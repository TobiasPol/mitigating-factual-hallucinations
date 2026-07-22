"""Operator-ready native-VLLM execution and finalization of the E4 screen."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
import torch
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from mfh.contracts import (
    ActivationSite,
    AdaptivePolicySpec,
    GenerationRecord,
    ModelSpec,
    PromptSpec,
    Question,
    TokenScope,
)
from mfh.data.normalization import normalize_answer
from mfh.errors import ConfigurationError, DataValidationError, FrozenArtifactError
from mfh.evaluation.grading import deterministic_short_answer_grade, triviaqa_scores
from mfh.experiments.e3_phase import open_e3_phase_completion
from mfh.experiments.e4_act_vllm import E4ActBaseline, verify_e4_act_baseline
from mfh.experiments.e4_baselines import (
    E4CapabilityReport,
    E4Feasibility,
    E4MethodCapability,
    E4MethodPolicy,
    E4Promotion,
    E4ScreenReceipt,
    build_e4_capability_report,
    build_e4_promotion_gate_bundle,
    build_e4_screen_receipt,
    derive_e4_promotion,
    load_e4_capability_report,
    load_e4_method_policy,
    load_e4_screen_receipt,
    sign_e4_fixed_execution_receipt,
    validate_e4_fixed_execution_record,
    write_e4_capability_report,
    write_e4_method_policy,
    write_e4_method_preflight,
    write_e4_promotion,
    write_e4_screen_receipt,
)
from mfh.experiments.gates import write_gate_evidence
from mfh.experiments.model_selection import validate_active_model_spec
from mfh.experiments.protocol import ExperimentPhase, StudyProtocol
from mfh.experiments.runner import (
    EvaluationCondition,
    PhaseRunContract,
    PhaseRunLedger,
    adaptive_policy_decision_digest,
    sign_adaptive_execution_receipt,
)
from mfh.experiments.static_direction_sources import (
    ResolvedStaticDirection,
    resolve_static_direction,
)
from mfh.inference.vllm_research import (
    VllmPromptFeatureCubeOutput,
    VllmResearchInterventionState,
)
from mfh.inference.vllm_runtime import VllmGenerationOutput, VllmRenderedPrompt, as_numpy
from mfh.provenance import sha256_path, stable_hash

_PROMPTS = ("P0-neutral", "P2-calibrated-abstention")
_METHODS = ("M1", "M2", "ACT-or-SADI")
_MEMORY_BYTES = 40 * 1024**3
_MAX_NEW_TOKENS = 48
_SETUP_INVENTORY = frozenset(
    {"capability-report.json", "screen-receipt.json", "preflights", "policies"}
)


class E4VllmRuntime(Protocol):
    def runtime_identity(self) -> Mapping[str, Any]: ...

    def render_prompt(
        self,
        prompt: PromptSpec,
        question: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> VllmRenderedPrompt: ...

    def generate(
        self, rendered: VllmRenderedPrompt, *, max_new_tokens: int
    ) -> VllmGenerationOutput: ...

    def generate_with_interventions(
        self,
        rendered: VllmRenderedPrompt,
        *,
        max_new_tokens: int,
        intervention_states: Mapping[
            tuple[int, ActivationSite], VllmResearchInterventionState
        ],
    ) -> VllmGenerationOutput: ...

    def standardized_intervention_state(
        self,
        direction: np.ndarray[Any, Any],
        *,
        standardized_alpha: float,
        reference_rms: float,
        token_scope: TokenScope,
        decay: float = 0.0,
    ) -> VllmResearchInterventionState: ...

    def prompt_feature_cube(
        self,
        rendered: VllmRenderedPrompt,
        *,
        layers: Sequence[int],
        sites: Sequence[ActivationSite],
    ) -> VllmPromptFeatureCubeOutput: ...


@dataclass(frozen=True, slots=True)
class E4VllmSetup:
    directory: Path
    report: E4CapabilityReport
    screen: E4ScreenReceipt
    policies: Mapping[str, E4MethodPolicy]
    policy_paths: Mapping[str, Path]


def _private_key(value: str) -> Ed25519PrivateKey:
    if not isinstance(value, str) or len(value) != 64:
        raise DataValidationError("E4 execution private key must be 32-byte lowercase hex")
    try:
        raw = bytes.fromhex(value)
        if raw.hex() != value:
            raise ValueError("key is not canonical lowercase hex")
        return Ed25519PrivateKey.from_private_bytes(raw)
    except ValueError as exc:
        raise DataValidationError(f"invalid E4 execution private key: {exc}") from exc


def e4_execution_public_key(private_key_hex: str) -> str:
    """Derive the public receipt key without persisting the private key."""

    return _private_key(private_key_hex).public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()


def _policy_paths(directory: Path) -> dict[str, Path]:
    return {
        "M1": directory / "policies" / "m1.json",
        "M2": directory / "policies" / "m2.json",
        "ACT-or-SADI": directory / "policies" / "act-or-sadi.json",
    }


def _capability_candidate(capability: str, path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    regular = bool(
        (resolved.is_file() or resolved.is_dir())
        and not resolved.is_symlink()
        and not (resolved.is_dir() and any(item.is_symlink() for item in resolved.rglob("*")))
    )
    return {
        "capability": capability,
        "candidate": str(resolved),
        "regular_artifact_found": regular,
        "artifact_sha256": sha256_path(resolved) if regular else None,
    }


def _optional_capability_probe(
    method: str,
    *,
    implementation: Path,
    autoencoder: Path | None = None,
) -> tuple[dict[str, bool], dict[str, Any]]:
    implementation_attempt = _capability_candidate(
        "implementation_module", implementation
    )
    attempts = [implementation_attempt]
    failure_codes: list[str] = []
    implementation_found = bool(implementation_attempt["regular_artifact_found"])
    if not implementation_found:
        failure_codes.append("IMPLEMENTATION_ARTIFACT_NOT_FOUND")
    if method == "ITI-if-feasible":
        attempts.append(
            {
                "capability": "per_head_output_hook",
                "runtime": "mfh.inference.vllm_research.VllmResearchRuntime",
                "supported": False,
                "failure_code": "VLLM_PER_HEAD_OUTPUT_HOOK_UNAVAILABLE",
            }
        )
        failure_codes.append("VLLM_PER_HEAD_OUTPUT_HOOK_UNAVAILABLE")
        checks = {
            "implementation_available": implementation_found,
            "per_head_output_hook": False,
        }
    elif method == "TruthX-if-feasible" and autoencoder is not None:
        autoencoder_attempt = _capability_candidate(
            "compatible_truthx_autoencoder", autoencoder
        )
        attempts.append(autoencoder_attempt)
        autoencoder_found = bool(autoencoder_attempt["regular_artifact_found"])
        if not autoencoder_found:
            failure_codes.append("COMPATIBLE_AUTOENCODER_NOT_FOUND")
        attempts.append(
            {
                "capability": "truthx_runtime_hook",
                "runtime": "mfh.inference.vllm_research.VllmResearchRuntime",
                "supported": False,
                "failure_code": "TRUTHX_VLLM_RUNTIME_HOOK_UNAVAILABLE",
            }
        )
        failure_codes.append("TRUTHX_VLLM_RUNTIME_HOOK_UNAVAILABLE")
        checks = {
            "compatible_autoencoder": autoencoder_found,
            "implementation_available": implementation_found,
            "runtime_hook_supported": False,
        }
    else:  # pragma: no cover - internal call sites are fixed
        raise ConfigurationError("unknown optional E4 capability probe")
    return checks, {
        "probe_schema_version": 1,
        "attempted_capabilities": attempts,
        "failure_codes": sorted(set(failure_codes)),
    }


def _condition(
    *,
    study: StudyProtocol,
    model: ModelSpec,
    prompt: PromptSpec,
    policy: E4MethodPolicy,
    policy_path: Path,
) -> EvaluationCondition:
    return EvaluationCondition(
        phase=ExperimentPhase.E4,
        benchmark="triviaqa",
        partition="T-dev-screen-2000",
        model_name=model.name,
        model_repository=model.repository,
        model_revision=model.revision,
        runtime=model.runtime,
        quantization=model.quantization,
        model_num_layers=model.num_layers,
        system_prompt_id=prompt.prompt_id,
        prompt_template_sha256=hashlib.sha256(prompt.text.encode()).hexdigest(),
        steering_method=policy.method,
        method_artifact_sha256=sha256_path(policy_path),
        layer=policy.layer,
        site=policy.site,
        token_scope=policy.token_scope,
        alpha=policy.alpha,
        sparsity=None,
        seed=17,
        study_protocol_digest=study.digest,
        adaptive_policy=policy.adaptive_policy,
    )


def prepare_e4_vllm_screen(
    setup_directory: str | Path,
    ledger_directory: str | Path,
    *,
    dev_questions: Sequence[Question],
    model: ModelSpec,
    prompts: Mapping[str, PromptSpec],
    study: StudyProtocol,
    runtime_artifact: str | Path,
    e2_probe_bundle: str | Path,
    e3_static_vectors: str | Path,
    m2_caa_artifact: str | Path,
    act_baseline_artifact: str | Path,
    e3_phase_run: str | Path,
    execution_private_key_hex: str,
    m1_layer: int,
    m2_layer: int,
    token_scope: TokenScope = TokenScope.FIRST_FOUR,
    standardized_alpha: float = 1.0,
    iti_implementation: str | Path | None = None,
    truthx_implementation: str | Path | None = None,
    truthx_autoencoder: str | Path | None = None,
) -> E4VllmSetup:
    """Freeze capability, screen, policies, conditions, and an empty E4 ledger."""

    validate_active_model_spec(model)
    phase = study.phase(ExperimentPhase.E4)
    selected_prompts = {name: prompts[name] for name in _PROMPTS if name in prompts}
    if (
        set(selected_prompts) != set(_PROMPTS)
        or tuple(phase.prerequisites) != (ExperimentPhase.E3,)
        or phase.models != (model.name,)
        or token_scope is not TokenScope.FIRST_FOUR
        or type(standardized_alpha) is not float
        or not math.isfinite(standardized_alpha)
        or standardized_alpha == 0.0
    ):
        raise ConfigurationError("E4 VLLM setup differs from the frozen screen protocol")
    setup = Path(setup_directory).resolve()
    ledger_path = Path(ledger_directory).resolve()
    if setup.exists() or setup.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite E4 setup: {setup}")
    if ledger_path.exists() or ledger_path.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite E4 ledger: {ledger_path}")
    runtime_path = Path(runtime_artifact).resolve()
    e2 = Path(e2_probe_bundle).resolve()
    e3 = Path(e3_static_vectors).resolve()
    m2 = Path(m2_caa_artifact).resolve()
    act_path = Path(act_baseline_artifact).resolve()
    act = verify_e4_act_baseline(act_path)
    e3_completion = open_e3_phase_completion(e3_phase_run, study=study)
    if (
        act.intervention_layer != m2_layer
        or act.token_scope is not token_scope
        or act.source_e2_completion_digest
        != e3_completion.prerequisite_digests.get("E2")
        or act.source_e2_sha256
        != e3_completion.input_fingerprints.get("activation_feature_schemas")
    ):
        raise DataValidationError("E4 ACT or E2/E3 lineage differs")
    if e3_completion.output_fingerprints.get("E3_static_vectors") != sha256_path(e3):
        raise DataValidationError("E4 M1 artifact differs from the completed E3 output")
    public_key = e4_execution_public_key(execution_private_key_hex)
    setup.mkdir(parents=True)
    (setup / "preflights").mkdir()
    (setup / "policies").mkdir()
    runtime_sha = sha256_path(runtime_path)
    implementation = {"M1": e3, "M2": m2, "ACT-or-SADI": act_path}
    project_root = Path(__file__).resolve().parents[3]
    iti_checks, iti_details = _optional_capability_probe(
        "ITI-if-feasible",
        implementation=(
            Path(iti_implementation)
            if iti_implementation is not None
            else project_root / "src/mfh/experiments/e4_iti_vllm.py"
        ),
    )
    truthx_checks, truthx_details = _optional_capability_probe(
        "TruthX-if-feasible",
        implementation=(
            Path(truthx_implementation)
            if truthx_implementation is not None
            else project_root / "src/mfh/experiments/e4_truthx_vllm.py"
        ),
        autoencoder=(
            Path(truthx_autoencoder)
            if truthx_autoencoder is not None
            else project_root / "artifacts/external-baselines/truthx-autoencoder"
        ),
    )
    checks = {
        "M1": {"implementation_loads": True, "runtime_hook_supported": True},
        "M2": {
            "implementation_loads": True,
            "paired_training_materials": True,
            "runtime_hook_supported": True,
        },
        "ITI-if-feasible": iti_checks,
        "ACT-or-SADI": {
            "implementation_loads": True,
            "calibrated_probe_available": True,
            "runtime_hook_supported": True,
        },
        "TruthX-if-feasible": truthx_checks,
    }
    details = {
        "M1": {"probe_schema_version": 1, "failure_codes": []},
        "M2": {"probe_schema_version": 1, "failure_codes": []},
        "ITI-if-feasible": iti_details,
        "ACT-or-SADI": {"probe_schema_version": 1, "failure_codes": []},
        "TruthX-if-feasible": truthx_details,
    }
    capabilities: list[E4MethodCapability] = []
    evidence_paths: dict[str, Path] = {}
    for method, method_checks in checks.items():
        evidence_path = setup / "preflights" / f"{method.lower().replace('/', '-')}.json"
        receipt = write_e4_method_preflight(
            evidence_path,
            method=method,
            runtime_artifact_sha256=runtime_sha,
            checks=method_checks,
            details=details[method],
        )
        evidence_paths[method] = evidence_path
        feasible = receipt["feasibility"] == E4Feasibility.FEASIBLE.value
        capabilities.append(
            E4MethodCapability(
                method=method,
                feasibility=(
                    E4Feasibility.FEASIBLE if feasible else E4Feasibility.INFEASIBLE
                ),
                implementation=(
                    {
                        "M1": "native-VLLM-E3-centroid",
                        "M2": "native-VLLM-residual-CAA",
                        "ACT-or-SADI": "E2-risk-gated-M2-intensity",
                    }[method]
                    if feasible
                    else None
                ),
                reason=(None if feasible else ";".join(details[method]["failure_codes"])),
                evidence_artifact_sha256=sha256_path(evidence_path),
                implementation_artifact_sha256=(
                    sha256_path(implementation[method]) if feasible else None
                ),
            )
        )
    report = build_e4_capability_report(
        model_identity=model.name,
        runtime_identity={
            "repository": model.repository,
            "revision": model.revision,
            "runtime": model.runtime.value,
            "quantization": model.quantization,
            "num_layers": model.num_layers,
        },
        runtime_artifact=runtime_path,
        source_artifacts={
            "E2_calibrated_probes": e2,
            "E3_static_vectors": e3,
        },
        methods=tuple(capabilities),
        method_evidence_artifacts=evidence_paths,
        implementation_artifacts=implementation,
    )
    report_path = setup / "capability-report.json"
    write_e4_capability_report(report_path, report)
    screen = build_e4_screen_receipt(tuple(dev_questions))
    if not screen.scientific_eligible:
        raise DataValidationError("E4 screen is not the exact reviewed T-dev cohort")
    screen_path = setup / "screen-receipt.json"
    write_e4_screen_receipt(screen_path, screen)
    policy_paths = _policy_paths(setup)
    policies = {
        "M1": write_e4_method_policy(
            policy_paths["M1"],
            report=report,
            method="M1",
            layer=m1_layer,
            site=ActivationSite.POST_MLP,
            token_scope=token_scope,
            alpha=standardized_alpha,
            execution_public_key=public_key,
        ),
        "M2": write_e4_method_policy(
            policy_paths["M2"],
            report=report,
            method="M2",
            layer=m2_layer,
            site=ActivationSite.BLOCK_OUTPUT,
            token_scope=token_scope,
            alpha=standardized_alpha,
            execution_public_key=public_key,
        ),
    }
    act_sha = sha256_path(act_path)
    adaptive = AdaptivePolicySpec(
        schema_version=2,
        release_risk_threshold=0.4,
        abstention_probability_threshold=0.7,
        alpha_max=1.0,
        alpha_beta=8.0,
        layer=None,
        site=None,
        token_scope=None,
        direction_sha256=None,
        direction_norm=None,
        execution_public_key=public_key,
        controller_artifact_sha256=act_sha,
        candidate_layers=(act.intervention_layer,),
        candidate_sites=(act.intervention_site,),
        candidate_token_scopes=(act.token_scope,),
        vector_count=1,
        likely_unknown_risk_threshold=0.8,
        alpha_mode="risk_gated",
        alpha_risk_threshold=0.4,
    )
    policies["ACT-or-SADI"] = write_e4_method_policy(
        policy_paths["ACT-or-SADI"],
        report=report,
        method="ACT-or-SADI",
        layer=None,
        site=None,
        token_scope=None,
        alpha=0.0,
        execution_public_key=public_key,
        adaptive_policy=adaptive,
    )
    conditions = tuple(
        _condition(
            study=study,
            model=model,
            prompt=selected_prompts[prompt_id],
            policy=policies[method],
            policy_path=policy_paths[method],
        )
        for prompt_id in _PROMPTS
        for method in _METHODS
    )
    contract = PhaseRunContract(
        phase=ExperimentPhase.E4,
        study_protocol_digest=study.digest,
        conditions=conditions,
        question_ids_by_benchmark={"triviaqa": screen.screen_question_ids},
        input_fingerprints=report.source_digests,
        prerequisite_digests={"E3": e3_completion.manifest_digest},
        required_gates=phase.gates,
    )
    PhaseRunLedger.create(
        ledger_path,
        contract,
        study=study,
        input_artifacts={
            "E2_calibrated_probes": e2,
            "E3_static_vectors": e3,
        },
        prerequisite_runs={ExperimentPhase.E3: e3_phase_run},
    )
    return E4VllmSetup(
        directory=setup,
        report=report,
        screen=screen,
        policies=MappingProxyType(policies),
        policy_paths=MappingProxyType(policy_paths),
    )


def load_e4_vllm_setup(directory: str | Path) -> E4VllmSetup:
    source = Path(directory)
    if (
        source.is_symlink()
        or not source.is_dir()
        or {item.name for item in source.iterdir()} != _SETUP_INVENTORY
        or any(item.is_symlink() for item in source.rglob("*"))
    ):
        raise FrozenArtifactError("E4 VLLM setup inventory differs")
    report = load_e4_capability_report(source / "capability-report.json")
    screen = load_e4_screen_receipt(source / "screen-receipt.json")
    paths = _policy_paths(source)
    if {item.name for item in (source / "policies").iterdir()} != {
        value.name for value in paths.values()
    }:
        raise FrozenArtifactError("E4 VLLM policy inventory differs")
    policies = {method: load_e4_method_policy(path) for method, path in paths.items()}
    if (
        set(policies) != set(report.feasible_methods)
        or any(
            policy.method != method
            or policy.capability_report_digest != report.report_digest
            for method, policy in policies.items()
        )
    ):
        raise FrozenArtifactError("E4 VLLM policies differ from capability report")
    return E4VllmSetup(
        directory=source.absolute(),
        report=report,
        screen=screen,
        policies=MappingProxyType(policies),
        policy_paths=MappingProxyType(paths),
    )


def _token_indices(scope: TokenScope, output_tokens: int) -> list[int]:
    if scope is TokenScope.FINAL_PROMPT:
        return [-1]
    limit = {
        TokenScope.FIRST_GENERATED: 1,
        TokenScope.FIRST_FOUR: 4,
        TokenScope.FIRST_EIGHT: 8,
        TokenScope.ALL_GENERATED: output_tokens,
        TokenScope.EXPONENTIAL_DECAY: output_tokens,
    }[scope]
    return list(range(min(limit, output_tokens)))


def _strict_runtime_arrays(
    state: VllmResearchInterventionState,
    *,
    expected_applications: int,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    if (
        type(state) is not VllmResearchInterventionState
        or len(state.applied_pre_history) != expected_applications
        or len(state.applied_post_history) != expected_applications
    ):
        raise FrozenArtifactError("E4 VLLM hook lacks its applied-edit history")
    captured = np.ascontiguousarray(
        np.stack(state.applied_pre_history).astype(np.float32, copy=False)
    )
    intervened = np.ascontiguousarray(
        np.stack(state.applied_post_history).astype(np.float32, copy=False)
    )
    if (
        captured.shape != intervened.shape
        or captured.size == 0
        or not np.isfinite(captured).all()
        or not np.isfinite(intervened).all()
        or np.array_equal(captured, intervened)
        or state.applications != expected_applications
        or expected_applications <= 0
    ):
        raise FrozenArtifactError("E4 VLLM hook did not execute its exact edit")
    delta = np.ascontiguousarray(intervened - captured)
    direction = np.ascontiguousarray(as_numpy(state.direction, dtype=np.float32))
    expected = np.stack(
        [direction * state.alpha for _ in range(expected_applications)]
    ).astype(np.float32, copy=False)
    tolerance = max(1e-6, float(np.max(np.abs(expected))) * 0.025)
    if (
        direction.ndim != 1
        or delta.shape != expected.shape
        or not np.allclose(delta, expected, rtol=0.025, atol=tolerance)
        or not math.isclose(
            float(np.linalg.norm(delta.astype(np.float64))),
            float(np.linalg.norm(expected.astype(np.float64))),
            rel_tol=0.025,
            abs_tol=1e-6,
        )
    ):
        raise FrozenArtifactError("E4 VLLM applied edit differs from direction and alpha")
    return captured, intervened, delta


def _runtime_metrics(generated: VllmGenerationOutput, *, prompt_peak: int = 0) -> dict[str, Any]:
    peak = max(prompt_peak, generated.peak_memory_bytes)
    if peak > _MEMORY_BYTES:
        raise FrozenArtifactError("E4 vLLM generation exceeded the A100 GPU envelope")
    return {
        "peak_memory_bytes": peak,
        "active_memory_bytes": generated.active_memory_bytes,
        "cache_memory_bytes": generated.cache_memory_bytes,
        "prompt_tokens_per_second": generated.prompt_tokens_per_second,
        "generation_tokens_per_second": generated.generation_tokens_per_second,
        "stop_type": generated.stop_type,
        "stopping_token_id": generated.stopping_token_id,
        "output_token_ids_sha256": hashlib.sha256(
            ",".join(str(value) for value in generated.token_ids).encode("ascii")
        ).hexdigest(),
    }


def _base_record(
    *,
    condition: EvaluationCondition,
    question: Question,
    rendered: VllmRenderedPrompt,
    generated: VllmGenerationOutput,
    layer: int | None,
    site: ActivationSite | None,
    scope: TokenScope | None,
    alpha: float,
    scores: Mapping[str, float],
    metadata: Mapping[str, Any],
) -> GenerationRecord:
    outcome = deterministic_short_answer_grade(generated.text, question.aliases)
    exact_match, token_f1 = triviaqa_scores(generated.text, question.aliases)
    return GenerationRecord(
        question_id=question.question_id,
        benchmark="triviaqa",
        model_repository=condition.model_repository,
        model_revision=condition.model_revision,
        runtime=condition.runtime,
        quantization=condition.quantization,
        system_prompt_id=condition.system_prompt_id,
        rendered_prompt_hash=rendered.sha256,
        steering_method=condition.steering_method,
        layer=layer,
        site=site,
        token_scope=scope,
        alpha=alpha,
        sparsity=None,
        controller_scores=scores,
        raw_output=generated.text,
        normalized_answer=normalize_answer(generated.text),
        outcome=outcome,
        generation_latency_seconds=generated.latency_seconds,
        input_tokens=generated.input_tokens,
        output_tokens=generated.output_tokens,
        condition_id=condition.condition_id,
        seed=condition.seed,
        metadata={
            "phase": ExperimentPhase.E4.value,
            "partition": condition.partition,
            "prompt_template_sha256": condition.prompt_template_sha256,
            "study_protocol_digest": condition.study_protocol_digest,
            "method_artifact_sha256": condition.method_artifact_sha256,
            "rendered_prompt_token_ids_sha256": rendered.token_ids_sha256,
            "official_scorer": "mfh.triviaqa.alias-aware-em-f1.v1",
            "official_exact_match": exact_match,
            "official_token_f1": token_f1,
            "reference_aliases_digest": stable_hash(list(question.aliases)),
            **dict(metadata),
        },
    )


def _fixed_record(
    *,
    runtime: E4VllmRuntime,
    condition: EvaluationCondition,
    policy: E4MethodPolicy,
    policy_path: Path,
    direction: ResolvedStaticDirection,
    question: Question,
    prompt: PromptSpec,
    private_key_hex: str,
) -> GenerationRecord:
    assert policy.layer is not None and policy.site is not None
    assert policy.token_scope is not None and policy.reference_rms is not None
    rendered = runtime.render_prompt(prompt, question.text, metadata=question.metadata)
    values = np.ascontiguousarray(direction.direction.numpy(), dtype=np.float32)
    state = runtime.standardized_intervention_state(
        values,
        standardized_alpha=policy.alpha,
        reference_rms=policy.reference_rms,
        token_scope=policy.token_scope,
    )
    generated = runtime.generate_with_interventions(
        rendered,
        max_new_tokens=_MAX_NEW_TOKENS,
        intervention_states={(policy.layer, policy.site): state},
    )
    if type(generated) is not VllmGenerationOutput:
        raise FrozenArtifactError("E4 native runtime returned a non-VLLM generation")
    indices = _token_indices(policy.token_scope, generated.output_tokens)
    captured, intervened, delta = _strict_runtime_arrays(
        state, expected_applications=len(indices)
    )
    trace = {
        "method_policy_sha256": sha256_path(policy_path),
        "implementation_artifact_sha256": policy.implementation_artifact_sha256,
        "layer": policy.layer,
        "site": policy.site.value,
        "token_scope": policy.token_scope.value,
        "alpha": policy.alpha,
        "applied_tokens": state.applications,
        "applied_token_indices": indices,
        "activation_delta_norm": float(np.linalg.norm(delta.astype(np.float64))),
        "direction_sha256": direction.direction_sha256,
        "direction_norm": direction.direction_norm,
        "reference_rms": policy.reference_rms,
        "pre_activation_sha256": hashlib.sha256(captured.tobytes(order="C")).hexdigest(),
        "post_activation_sha256": hashlib.sha256(intervened.tobytes(order="C")).hexdigest(),
        "delta_sha256": hashlib.sha256(delta.tobytes(order="C")).hexdigest(),
    }
    draft = _base_record(
        condition=condition,
        question=question,
        rendered=rendered,
        generated=generated,
        layer=policy.layer,
        site=policy.site,
        scope=policy.token_scope,
        alpha=policy.alpha,
        scores={},
        metadata={
            "intervention_trace": trace,
            "intervention_trace_digest": stable_hash(trace),
            "generation_runtime_metrics": _runtime_metrics(generated),
        },
    )
    record = replace(
        draft,
        metadata={
            **draft.metadata,
            "execution_receipt_signature": sign_e4_fixed_execution_receipt(
                draft,
                policy=policy,
                policy_artifact_sha256=sha256_path(policy_path),
                private_key_hex=private_key_hex,
            ),
        },
    )
    validate_e4_fixed_execution_record(
        record,
        policy=policy,
        policy_artifact_sha256=sha256_path(policy_path),
    )
    return record


def _adaptive_record(
    *,
    runtime: E4VllmRuntime,
    condition: EvaluationCondition,
    policy: E4MethodPolicy,
    baseline: E4ActBaseline,
    question: Question,
    prompt: PromptSpec,
    private_key_hex: str,
) -> GenerationRecord:
    adaptive = policy.adaptive_policy
    if adaptive is None:
        raise FrozenArtifactError("E4 adaptive condition lacks its policy")
    rendered = runtime.render_prompt(prompt, question.text, metadata=question.metadata)
    cube = runtime.prompt_feature_cube(
        rendered,
        layers=(baseline.feature_layer,),
        sites=(baseline.feature_site,),
    )
    features = torch.from_numpy(
        np.array(
            cube.activations[baseline.feature_site][baseline.feature_layer],
            dtype=np.float32,
            order="C",
            copy=True,
        )
    )
    probabilities = baseline.risk_probe.predict_probabilities(features)
    scores = {
        label: float(probabilities[0, index])
        for index, label in enumerate(baseline.risk_probe.state.labels)
    }
    action = "release" if scores["I"] <= adaptive.release_risk_threshold else "intervene"
    layer: int | None = None
    site: ActivationSite | None = None
    scope: TokenScope | None = None
    alpha = 0.0
    state: VllmResearchInterventionState | None = None
    interventions: dict[tuple[int, ActivationSite], VllmResearchInterventionState] = {}
    if action == "intervene":
        assert adaptive.alpha_risk_threshold is not None
        layer = baseline.intervention_layer
        site = baseline.intervention_site
        scope = baseline.token_scope
        alpha = adaptive.alpha_max / (
            1.0
            + math.exp(
                -adaptive.alpha_beta * (scores["I"] - adaptive.alpha_risk_threshold)
            )
        )
        values = np.ascontiguousarray(
            baseline.direction.direction.numpy(), dtype=np.float32
        )
        state = runtime.standardized_intervention_state(
            values,
            standardized_alpha=alpha,
            reference_rms=baseline.direction.reference_rms,
            token_scope=scope,
        )
        interventions[(layer, site)] = state
        generated = runtime.generate_with_interventions(
            rendered,
            max_new_tokens=_MAX_NEW_TOKENS,
            intervention_states=interventions,
        )
    else:
        generated = runtime.generate(rendered, max_new_tokens=_MAX_NEW_TOKENS)
    if type(generated) is not VllmGenerationOutput:
        raise FrozenArtifactError("E4 native runtime returned a non-VLLM generation")
    feature_values = np.ascontiguousarray(features.numpy(), dtype=np.float32)
    metadata: dict[str, Any] = {
        "policy_action": action,
        "generation_runtime_metrics": _runtime_metrics(
            generated, prompt_peak=cube.peak_memory_bytes
        ),
        "adaptive_controller_evidence": {
            "schema_version": 1,
            "controller_artifact_sha256": policy.implementation_artifact_sha256,
            "feature_schema_digest": baseline.risk_probe.training_schema.digest,
            "feature_values_sha256": hashlib.sha256(
                feature_values.tobytes(order="C")
            ).hexdigest(),
            "feature_values": feature_values.reshape(-1).tolist(),
            "prompt_feature_peak_memory_bytes": cube.peak_memory_bytes,
            "maximum_token_probability": cube.maximum_token_probability,
            "output_entropy": cube.output_entropy,
            "site_selection": "max_mixed_direction_norm_then_site",
        },
    }
    if action == "intervene":
        assert state is not None and layer is not None and site is not None and scope is not None
        indices = _token_indices(scope, generated.output_tokens)
        captured, intervened, delta = _strict_runtime_arrays(
            state, expected_applications=len(indices)
        )
        trace = {
            "layer": layer,
            "site": site.value,
            "token_scope": scope.value,
            "alpha": alpha,
            "sparsity": None,
            "applied_tokens": state.applications,
            "applied_token_indices": indices,
            "activation_delta_norm": float(np.linalg.norm(delta.astype(np.float64))),
            "direction_sha256": baseline.direction.direction_sha256,
            "direction_norm": (
                baseline.direction.direction_norm
                * baseline.direction.reference_rms
            ),
            "controller_artifact_sha256": policy.implementation_artifact_sha256,
            "router_weights": [1.0],
            "router_weights_sha256": stable_hash([1.0]),
            "pre_activation_sha256": hashlib.sha256(captured.tobytes(order="C")).hexdigest(),
            "post_activation_sha256": hashlib.sha256(
                intervened.tobytes(order="C")
            ).hexdigest(),
            "delta_sha256": hashlib.sha256(delta.tobytes(order="C")).hexdigest(),
        }
        metadata.update(
            {
                "intervention_trace": trace,
                "intervention_trace_digest": stable_hash(trace),
            }
        )
    draft = _base_record(
        condition=condition,
        question=question,
        rendered=rendered,
        generated=generated,
        layer=layer,
        site=site,
        scope=scope,
        alpha=alpha,
        scores=scores,
        metadata=metadata,
    )
    decided = replace(
        draft,
        metadata={
            **draft.metadata,
            "policy_decision_digest": adaptive_policy_decision_digest(
                draft,
                policy=adaptive,
                policy_action=action,
            ),
        },
    )
    return replace(
        decided,
        metadata={
            **decided.metadata,
            "execution_receipt_signature": sign_adaptive_execution_receipt(
                decided,
                policy=adaptive,
                private_key_hex=private_key_hex,
            ),
        },
    )


def _verify_act_controller_replay(
    record: GenerationRecord,
    *,
    baseline: E4ActBaseline,
) -> None:
    evidence = record.metadata.get("adaptive_controller_evidence")
    if not isinstance(evidence, Mapping):
        raise FrozenArtifactError("E4 adaptive controller evidence is absent")
    values = evidence.get("feature_values")
    if not isinstance(values, list) or len(values) != 5_120:
        raise FrozenArtifactError("E4 adaptive feature width differs")
    try:
        features = torch.tensor(values, dtype=torch.float32).reshape(1, -1)
        probabilities = baseline.risk_probe.predict_probabilities(features)
    except (TypeError, ValueError, DataValidationError) as exc:
        raise FrozenArtifactError(f"E4 adaptive probe replay failed: {exc}") from exc
    replayed = {
        label: float(probabilities[0, index])
        for index, label in enumerate(baseline.risk_probe.state.labels)
    }
    if any(
        not math.isclose(
            record.controller_scores[label], replayed[label], rel_tol=0, abs_tol=1e-7
        )
        for label in replayed
    ):
        raise FrozenArtifactError("E4 adaptive C/I/A scores differ from probe replay")


def _assert_ledger_setup(
    ledger: PhaseRunLedger,
    setup: E4VllmSetup,
) -> None:
    expected_policy_sha = {
        method: sha256_path(path) for method, path in setup.policy_paths.items()
    }
    if (
        ledger.contract.phase is not ExperimentPhase.E4
        or ledger.contract.question_ids_by_benchmark
        != {"triviaqa": setup.screen.screen_question_ids}
        or dict(ledger.contract.input_fingerprints) != dict(setup.report.source_digests)
        or len(ledger.contract.conditions) != 6
        or any(
            condition.method_artifact_sha256
            != expected_policy_sha.get(condition.steering_method)
            or condition.system_prompt_id not in _PROMPTS
            for condition in ledger.contract.conditions
        )
    ):
        raise FrozenArtifactError("E4 ledger differs from its frozen VLLM setup")


def run_e4_vllm_screen(
    setup_directory: str | Path,
    ledger_directory: str | Path,
    *,
    study: StudyProtocol,
    prompts: Mapping[str, PromptSpec],
    runtime: E4VllmRuntime,
    execution_private_key_hex: str,
    request_budget: int | None = None,
    checkpoint_rows: int = 8,
) -> Mapping[str, Any]:
    """Run or resume signed M1, M2, and ACT/SADI screen rows."""

    if request_budget is not None and (
        type(request_budget) is not int or request_budget <= 0
    ):
        raise ConfigurationError("E4 request budget must be positive")
    if type(checkpoint_rows) is not int or checkpoint_rows <= 0:
        raise ConfigurationError("E4 checkpoint rows must be positive")
    setup = load_e4_vllm_setup(setup_directory)
    if e4_execution_public_key(execution_private_key_hex) not in {
        policy.execution_public_key for policy in setup.policies.values()
    } or len({policy.execution_public_key for policy in setup.policies.values()}) != 1:
        raise FrozenArtifactError("E4 execution key differs from the frozen policies")
    ledger = PhaseRunLedger.open(ledger_directory, study=study)
    _assert_ledger_setup(ledger, setup)
    questions = {
        value.question_id: value
        for value in setup.screen.dev_questions
        if value.question_id in set(setup.screen.screen_question_ids)
    }
    if set(questions) != set(setup.screen.screen_question_ids) or not set(_PROMPTS) <= set(
        prompts
    ):
        raise FrozenArtifactError("E4 live questions or prompts differ from the frozen setup")
    expected_prompt_hashes = {
        condition.system_prompt_id: condition.prompt_template_sha256
        for condition in ledger.contract.conditions
    }
    if any(
        prompt_id not in prompts
        or prompts[prompt_id].prompt_id != prompt_id
        or hashlib.sha256(prompts[prompt_id].text.encode("utf-8")).hexdigest()
        != expected_hash
        for prompt_id, expected_hash in expected_prompt_hashes.items()
    ):
        raise FrozenArtifactError("E4 live prompt text differs from the frozen conditions")
    m2_path = Path(setup.report.artifact_paths["implementation:M2"])
    try:
        m2_plan = json.loads((m2_path / "plan.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read E4 M2 runtime identity: {exc}") from exc
    if dict(runtime.runtime_identity()) != m2_plan.get("runtime_identity"):
        raise FrozenArtifactError("live E4 runtime differs from the scientific M2/E3 runtime")
    fixed_directions: dict[str, ResolvedStaticDirection] = {}
    for method in ("M1", "M2"):
        fixed_policy = setup.policies[method]
        if fixed_policy.layer is None or fixed_policy.site is None:
            raise FrozenArtifactError("E4 fixed policy lacks its intervention geometry")
        fixed_directions[method] = resolve_static_direction(
            setup.report.artifact_paths[f"implementation:{method}"],
            method=method,
            layer=fixed_policy.layer,
            site=fixed_policy.site,
        )
    act = verify_e4_act_baseline(
        setup.report.artifact_paths["implementation:ACT-or-SADI"]
    )
    handled = 0
    batch: list[GenerationRecord] = []
    for pending in ledger.iter_pending():
        if request_budget is not None and handled >= request_budget:
            break
        condition = pending.condition
        question = questions[pending.question_id]
        policy = setup.policies[condition.steering_method]
        prompt = prompts[condition.system_prompt_id]
        if condition.steering_method in {"M1", "M2"}:
            record = _fixed_record(
                runtime=runtime,
                condition=condition,
                policy=policy,
                policy_path=setup.policy_paths[condition.steering_method],
                direction=fixed_directions[condition.steering_method],
                question=question,
                prompt=prompt,
                private_key_hex=execution_private_key_hex,
            )
        elif condition.steering_method == "ACT-or-SADI":
            record = _adaptive_record(
                runtime=runtime,
                condition=condition,
                policy=policy,
                baseline=act,
                question=question,
                prompt=prompt,
                private_key_hex=execution_private_key_hex,
            )
        else:  # pragma: no cover - setup validation prevents this
            raise FrozenArtifactError("E4 ledger contains an unsupported feasible method")
        batch.append(record)
        handled += 1
        if len(batch) == checkpoint_rows:
            ledger.checkpoint(batch)
            batch.clear()
    if batch:
        ledger.checkpoint(batch)
    completed, expected = ledger.progress()
    return MappingProxyType(
        {
            "valid": True,
            "completed": completed,
            "expected": expected,
            "handled_this_session": handled,
            "complete": completed == expected,
            "record_set_digest": ledger.record_set_digest() if completed else None,
        }
    )


def verify_e4_vllm_screen(
    setup_directory: str | Path,
    ledger_directory: str | Path,
    *,
    study: StudyProtocol,
    require_complete: bool = False,
) -> Mapping[str, Any]:
    setup = load_e4_vllm_setup(setup_directory)
    ledger = PhaseRunLedger.open(ledger_directory, study=study)
    _assert_ledger_setup(ledger, setup)
    completed, expected = ledger.progress()
    records = tuple(ledger.records())
    if len(records) != completed:
        raise FrozenArtifactError("E4 ledger progress differs from its records")
    if require_complete and completed != expected:
        raise FrozenArtifactError("E4 VLLM screen is incomplete")
    public_keys = {policy.execution_public_key for policy in setup.policies.values()}
    if len(public_keys) != 1:
        raise FrozenArtifactError("E4 policies do not share one execution signing key")
    questions = {
        question.question_id: question
        for question in setup.screen.dev_questions
        if question.question_id in set(setup.screen.screen_question_ids)
    }
    act = verify_e4_act_baseline(
        setup.report.artifact_paths["implementation:ACT-or-SADI"]
    )
    conditions = {condition.condition_id: condition for condition in ledger.contract.conditions}
    peaks: list[int] = []
    metric_keys = {
        "peak_memory_bytes",
        "active_memory_bytes",
        "cache_memory_bytes",
        "prompt_tokens_per_second",
        "generation_tokens_per_second",
        "stop_type",
        "stopping_token_id",
        "output_token_ids_sha256",
    }
    for record in records:
        question = questions.get(record.question_id)
        if question is None:
            raise FrozenArtifactError("E4 record question differs from the frozen screen")
        outcome = deterministic_short_answer_grade(record.raw_output, question.aliases)
        exact_match, token_f1 = triviaqa_scores(record.raw_output, question.aliases)
        if (
            record.outcome is not outcome
            or record.normalized_answer != normalize_answer(record.raw_output)
            or record.metadata.get("official_exact_match") != exact_match
            or record.metadata.get("official_token_f1") != token_f1
            or record.metadata.get("reference_aliases_digest")
            != stable_hash(list(question.aliases))
        ):
            raise FrozenArtifactError("E4 deterministic TriviaQA grading differs")
        policy = setup.policies[record.steering_method]
        if policy.adaptive_policy is None:
            validate_e4_fixed_execution_record(
                record,
                policy=policy,
                policy_artifact_sha256=sha256_path(
                    setup.policy_paths[record.steering_method]
                ),
            )
        elif record.metadata.get("policy_action") == "intervene":
            trace = record.metadata.get("intervention_trace")
            if (
                not isinstance(trace, Mapping)
                or trace.get("direction_sha256")
                != act.direction.direction_sha256
                or not math.isclose(
                    float(trace.get("direction_norm", math.nan)),
                    act.direction.direction_norm * act.direction.reference_rms,
                    rel_tol=0,
                    abs_tol=1e-7,
                )
                or trace.get("controller_artifact_sha256")
                != policy.implementation_artifact_sha256
            ):
                raise FrozenArtifactError(
                    "E4 adaptive trace differs from its standardized M2 direction"
                )
        if policy.adaptive_policy is not None:
            _verify_act_controller_replay(record, baseline=act)
            conditions[record.condition_id].validate_record(record)
        metrics = record.metadata.get("generation_runtime_metrics")
        if not isinstance(metrics, Mapping) or set(metrics) != metric_keys:
            raise FrozenArtifactError("E4 record lacks exact VLLM runtime metrics")
        integer_metrics = tuple(
            metrics[name]
            for name in ("peak_memory_bytes", "active_memory_bytes", "cache_memory_bytes")
        )
        rate_metrics = tuple(
            metrics[name]
            for name in ("prompt_tokens_per_second", "generation_tokens_per_second")
        )
        if (
            any(type(value) is not int or value < 0 for value in integer_metrics)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                or float(value) < 0
                for value in rate_metrics
            )
            or not isinstance(metrics["stop_type"], str)
            or (
                metrics["stopping_token_id"] is not None
                and type(metrics["stopping_token_id"]) is not int
            )
            or not isinstance(metrics["output_token_ids_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", metrics["output_token_ids_sha256"])
            is None
        ):
            raise FrozenArtifactError("E4 record VLLM runtime metrics are invalid")
        peaks.append(int(metrics["peak_memory_bytes"]))
    maximum_peak = max(peaks, default=0)
    if maximum_peak > _MEMORY_BYTES:
        raise FrozenArtifactError("E4 VLLM screen exceeds the memory envelope")
    return MappingProxyType(
        {
            "valid": True,
            "completed": completed,
            "expected": expected,
            "complete": completed == expected,
            "maximum_peak_memory_bytes": maximum_peak,
            "record_set_digest": ledger.record_set_digest() if completed else None,
        }
    )


def finalize_e4_vllm_screen(
    setup_directory: str | Path,
    ledger_directory: str | Path,
    *,
    study: StudyProtocol,
    promotion_path: str | Path,
    gate_evidence_path: str | Path,
) -> tuple[E4Promotion, Any]:
    """Derive promotion, replay the registered gate, and freeze E4 terminally."""

    setup = load_e4_vllm_setup(setup_directory)
    ledger = PhaseRunLedger.open(ledger_directory, study=study)
    _assert_ledger_setup(ledger, setup)
    verify_e4_vllm_screen(
        setup_directory,
        ledger_directory,
        study=study,
        require_complete=True,
    )
    promotion = derive_e4_promotion(
        ledger_directory=ledger_directory,
        study=study,
        report=setup.report,
        screen=setup.screen,
        method_policy_artifacts=setup.policy_paths,
    )
    write_e4_promotion(
        promotion_path,
        promotion,
        ledger_directory=ledger_directory,
        study=study,
        report=setup.report,
        screen=setup.screen,
        method_policy_artifacts=setup.policy_paths,
    )
    parameters, supporting = build_e4_promotion_gate_bundle(
        capability_report_path=setup.directory / "capability-report.json",
        screen_receipt_path=setup.directory / "screen-receipt.json",
        promotion_path=promotion_path,
        method_policy_artifacts=setup.policy_paths,
    )
    write_gate_evidence(
        gate_evidence_path,
        phase=ExperimentPhase.E4,
        gate="promotion_decision_frozen",
        contract_digest=ledger.contract.digest,
        record_set_digest=ledger.record_set_digest(),
        observations=(),
        parameters=parameters,
    )
    result = ledger.evaluate_gate(
        "promotion_decision_frozen",
        gate_evidence_path,
        supporting_artifacts=supporting,
    )
    terminal = ledger.finalize({"promotion_decision_frozen": result})
    return promotion, terminal

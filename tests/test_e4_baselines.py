from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from mfh.config import load_model_spec, load_prompt_specs
from mfh.contracts import (
    ActivationSite,
    AdaptivePolicySpec,
    GenerationRecord,
    Outcome,
    Question,
    TokenScope,
)
from mfh.data.io import read_questions
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.experiments.e2_probes import VerifiedE2ProbeBundle
from mfh.experiments.e4_act_mlx import (
    _package_e4_act_baseline,
    build_e4_act_baseline,
    verify_e4_act_baseline,
)
from mfh.experiments.e4_baselines import (
    E4CapabilityReport,
    E4Feasibility,
    E4MethodCapability,
    E4MethodPolicy,
    E4Protocol,
    build_e4_capability_report,
    build_e4_conditions,
    build_e4_promotion_gate_bundle,
    build_e4_screen_receipt,
    derive_e4_promotion,
    load_e4_capability_report,
    load_e4_screen_receipt,
    sign_e4_fixed_execution_receipt,
    validate_e4_fixed_execution_record,
    verify_e4_capability_report,
    verify_e4_promotion,
    verify_e4_screen_receipt,
    write_e4_capability_report,
    write_e4_method_policy,
    write_e4_promotion,
    write_e4_screen_receipt,
)
from mfh.experiments.e4_mlx import (
    E4MlxSetup,
    _adaptive_record,
    _fixed_record,
    _strict_runtime_arrays,
    _verify_act_controller_replay,
    run_e4_mlx_screen,
    verify_e4_mlx_screen,
)
from mfh.experiments.evidence import read_gate_result, write_gate_result
from mfh.experiments.gates import (
    GateEvaluationContext,
    evaluate_gate,
    validate_gate_result,
    write_gate_evidence,
)
from mfh.experiments.protocol import ExperimentPhase, load_study_protocol
from mfh.experiments.runner import (
    EvaluationCondition,
    PhaseRunContract,
    PhaseRunLedger,
    _validate_e3_prerequisite_lineage,
    adaptive_policy_decision_digest,
    sign_adaptive_execution_receipt,
)
from mfh.experiments.static_direction_sources import resolve_static_direction
from mfh.inference.mlx_research import (
    MlxPromptFeatureCubeOutput,
    MlxResearchInterventionState,
)
from mfh.inference.mlx_runtime import MlxGenerationOutput, MlxRenderedPrompt
from mfh.provenance import sha256_path, stable_hash
from tests.e4_test_artifacts import (
    build_e2_probe_bundle,
    build_e3_m1_bundle,
    build_m2_caa_bundle,
)

ROOT = Path(__file__).parents[1]
_EXECUTION_PRIVATE_KEY = "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60"
_EXECUTION_PUBLIC_KEY = "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
_MODEL_IDENTITY = {
    "repository": "mlx-community/Qwen3.6-27B-4bit",
    "revision": "c000ac2c2057d94be3fa931000c31723aac53282",
    "runtime": "mlx",
    "quantization": "affine-g64-mlx-4bit",
    "num_layers": 64,
}


def _protocol(*, dev_rows: int = 6, screen_rows: int = 2) -> E4Protocol:
    return E4Protocol(dev_rows=dev_rows, screen_rows=screen_rows)


def _questions(count: int = 6) -> tuple[Question, ...]:
    return tuple(
        Question(
            question_id=f"dev-{index}",
            benchmark="triviaqa",
            text=f"Unique question {index}?",
            aliases=(f"unique-answer-{index}",),
            split="T-dev",
        )
        for index in range(count)
    )


def _artifact(root: Path, name: str, value: str) -> Path:
    path = root / name
    path.write_text(value, encoding="utf-8")
    return path


def _preflight(
    root: Path,
    *,
    method: str,
    runtime_sha256: str,
    feasibility: E4Feasibility,
) -> Path:
    required_checks = {
        "M1": {"implementation_loads", "runtime_hook_supported"},
        "M2": {
            "implementation_loads",
            "paired_training_materials",
            "runtime_hook_supported",
        },
        "ITI-if-feasible": {"implementation_available", "per_head_output_hook"},
        "ACT-or-SADI": {
            "implementation_loads",
            "calibrated_probe_available",
            "runtime_hook_supported",
        },
        "TruthX-if-feasible": {
            "compatible_autoencoder",
            "implementation_available",
            "runtime_hook_supported",
        },
    }[method]
    checks = {key: feasibility is E4Feasibility.FEASIBLE for key in required_checks}
    body: dict[str, Any] = {
        "schema_version": 1,
        "method": method,
        "model_identity": "qwen3.6-27b-mlx-4bit",
        "model_runtime_identity": _MODEL_IDENTITY,
        "runtime_artifact_sha256": runtime_sha256,
        "feasibility": feasibility.value,
        "checks": checks,
        "details": {},
    }
    return _artifact(
        root,
        f"{method}-preflight.json",
        json.dumps({**body, "evidence_digest": stable_hash(body)}, sort_keys=True),
    )


def _report(tmp_path: Path) -> E4CapabilityReport:
    runtime = _artifact(tmp_path, "runtime.py", "pinned runtime")
    e3_vectors = build_e3_m1_bundle(tmp_path)
    e2_probes = build_e2_probe_bundle(tmp_path)
    m2 = build_m2_caa_bundle(tmp_path)
    e2_manifest = json.loads((e2_probes / "manifest.json").read_text(encoding="utf-8"))
    e2_results = json.loads((e2_probes / "results.json").read_text(encoding="utf-8"))
    verified_e2 = VerifiedE2ProbeBundle(
        directory=e2_probes,
        plan_identity="6" * 64,
        manifest_digest=e2_manifest["manifest_digest"],
        selected_views=MappingProxyType({}),
        selected_gate_artifact=e2_results["gate"]["selected_artifact_sha256"],
        gate_passed=True,
        gate_probe_auroc=0.8,
        gate_baseline_auroc=0.5,
        controller_input_artifacts=MappingProxyType({}),
        scientific_eligible=True,
    )
    act = _package_e4_act_baseline(
        tmp_path / "act-baseline",
        e2_probe_bundle=e2_probes,
        m2_caa_artifact=m2,
        intervention_layer=31,
        verified_e2=verified_e2,
        e2_completion_digest="7" * 64,
        e2_workspace_plan_identity="8" * 64,
    )
    sources = {
        "E2_calibrated_probes": e2_probes,
        "E3_static_vectors": e3_vectors,
    }
    feasibility = {
        "M1": E4Feasibility.FEASIBLE,
        "M2": E4Feasibility.FEASIBLE,
        "ITI-if-feasible": E4Feasibility.INFEASIBLE,
        "ACT-or-SADI": E4Feasibility.FEASIBLE,
        "TruthX-if-feasible": E4Feasibility.INFEASIBLE,
    }
    evidence = {
        method: _preflight(
            tmp_path,
            method=method,
            runtime_sha256=sha256_path(runtime),
            feasibility=value,
        )
        for method, value in feasibility.items()
    }
    implementation = {
        "M1": e3_vectors,
        "M2": m2,
        "ACT-or-SADI": act.directory,
    }
    capabilities = tuple(
        E4MethodCapability(
            method=method,
            feasibility=value,
            implementation=(f"{method}-v1" if value is E4Feasibility.FEASIBLE else None),
            reason=(None if value is E4Feasibility.FEASIBLE else "preflight unavailable"),
            evidence_artifact_sha256=sha256_path(evidence[method]),
            implementation_artifact_sha256=(
                sha256_path(implementation[method])
                if value is E4Feasibility.FEASIBLE
                else None
            ),
        )
        for method, value in feasibility.items()
    )
    return build_e4_capability_report(
        model_identity="qwen3.6-27b-mlx-4bit",
        runtime_identity=_MODEL_IDENTITY,
        runtime_artifact=runtime,
        source_artifacts=sources,
        methods=capabilities,
        method_evidence_artifacts=evidence,
        implementation_artifacts=implementation,
    )


def _policies(
    tmp_path: Path, report: E4CapabilityReport
) -> tuple[dict[str, E4MethodPolicy], dict[str, Path]]:
    act_artifact = next(
        value.implementation_artifact_sha256
        for value in report.methods
        if value.method == "ACT-or-SADI"
    )
    assert act_artifact is not None
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
        execution_public_key=_EXECUTION_PUBLIC_KEY,
        controller_artifact_sha256=act_artifact,
        candidate_layers=(31,),
        candidate_sites=(ActivationSite.BLOCK_OUTPUT,),
        candidate_token_scopes=(TokenScope.FIRST_FOUR,),
        vector_count=1,
        likely_unknown_risk_threshold=0.8,
        alpha_mode="risk_gated",
        alpha_risk_threshold=0.4,
    )
    paths: dict[str, Path] = {}
    policies: dict[str, E4MethodPolicy] = {}
    for method in report.feasible_methods:
        path = tmp_path / f"{method}-policy.json"
        policy = write_e4_method_policy(
            path,
            report=report,
            method=method,
            layer=None if method == "ACT-or-SADI" else 31,
            site=(
                None
                if method == "ACT-or-SADI"
                else ActivationSite.BLOCK_OUTPUT
                if method == "M2"
                else ActivationSite.POST_MLP
            ),
            token_scope=None if method == "ACT-or-SADI" else TokenScope.FIRST_FOUR,
            alpha=0.0 if method == "ACT-or-SADI" else 1.0,
            execution_public_key=_EXECUTION_PUBLIC_KEY,
            adaptive_policy=adaptive if method == "ACT-or-SADI" else None,
            direction_sha256=None,
            direction_norm=None,
        )
        paths[method] = path
        policies[method] = policy
    return policies, paths


def policy_direction(report: E4CapabilityReport, method: str) -> str:
    value = next(
        capability.implementation_artifact_sha256
        for capability in report.methods
        if capability.method == method
    )
    assert value is not None
    return value


def _condition(
    policies: dict[str, E4MethodPolicy],
    paths: dict[str, Path],
    method: str,
    prompt: str,
) -> EvaluationCondition:
    study = load_study_protocol(ROOT / "configs/experiments/phases.yaml")
    model = load_model_spec(ROOT / "configs/models/qwen3.6-27b-mlx-4bit.yaml")
    prompt_spec = {
        value.prompt_id: value
        for value in load_prompt_specs(ROOT / "configs/prompts/primary.yaml")
    }[prompt]
    policy = policies[method]
    return EvaluationCondition(
        phase=ExperimentPhase.E4,
        benchmark="triviaqa",
        partition="T-dev-screen-2000",
        model_name="qwen3.6-27b-mlx-4bit",
        model_repository=model.repository,
        model_revision=model.revision,
        runtime=model.runtime,
        quantization=model.quantization,
        model_num_layers=model.num_layers,
        system_prompt_id=prompt,
        prompt_template_sha256=hashlib.sha256(prompt_spec.text.encode()).hexdigest(),
        steering_method=method,
        method_artifact_sha256=sha256_path(paths[method]),
        layer=policy.layer,
        site=policy.site,
        token_scope=policy.token_scope,
        alpha=policy.alpha,
        sparsity=None,
        seed=17,
        study_protocol_digest=study.digest,
        adaptive_policy=policy.adaptive_policy,
    )


def _record(
    condition: EvaluationCondition,
    policy: E4MethodPolicy,
    question_id: str,
    outcome: Outcome,
) -> GenerationRecord:
    adaptive = condition.steering_method == "ACT-or-SADI"
    release_bucket = int(stable_hash(question_id)[:8], 16) % 100
    action = "release" if adaptive and release_bucket < 2 else "intervene"
    scores = (
        {"C": 0.7, "I": 0.2, "A": 0.1}
        if action == "release"
        else {"C": 0.3, "I": 0.6, "A": 0.1}
        if adaptive
        else {}
    )
    adaptive_alpha = (
        policy.adaptive_policy.alpha_max
        / (
            1
            + math.exp(
                -policy.adaptive_policy.alpha_beta
                * (
                    scores["I"]
                    - policy.adaptive_policy.release_risk_threshold
                )
            )
        )
        if adaptive and action == "intervene" and policy.adaptive_policy is not None
        else 0.0
    )
    if adaptive:
        trace = (
            {
                "layer": 31,
                "site": ActivationSite.BLOCK_OUTPUT.value,
                "token_scope": TokenScope.FIRST_FOUR.value,
                "alpha": adaptive_alpha,
                "sparsity": None,
                "applied_tokens": 1,
                "applied_token_indices": [0],
                "activation_delta_norm": adaptive_alpha,
                "direction_sha256": "f" * 64,
                "direction_norm": 1.0,
                "controller_artifact_sha256": policy.implementation_artifact_sha256,
                "router_weights": [1.0],
                "router_weights_sha256": stable_hash([1.0]),
                "pre_activation_sha256": "a" * 64,
                "post_activation_sha256": "b" * 64,
                "delta_sha256": "c" * 64,
            }
            if action == "intervene"
            else None
        )
    else:
        trace = {
            "method_policy_sha256": condition.method_artifact_sha256,
            "implementation_artifact_sha256": policy.implementation_artifact_sha256,
            "layer": policy.layer,
            "site": policy.site.value if policy.site is not None else None,
            "token_scope": (
                policy.token_scope.value if policy.token_scope is not None else None
            ),
            "alpha": policy.alpha,
            "applied_tokens": 1,
            "applied_token_indices": [0],
            "activation_delta_norm": (
                abs(policy.alpha)
                * float(policy.reference_rms or 0.0)
                * float(policy.direction_norm or 0.0)
            ),
            "direction_sha256": policy.direction_sha256,
            "direction_norm": policy.direction_norm,
            "reference_rms": policy.reference_rms,
            "pre_activation_sha256": "a" * 64,
            "post_activation_sha256": "b" * 64,
            "delta_sha256": "c" * 64,
        }
    draft = GenerationRecord(
        question_id=question_id,
        benchmark="triviaqa",
        model_repository=condition.model_repository,
        model_revision=condition.model_revision,
        runtime=condition.runtime,
        quantization=condition.quantization,
        system_prompt_id=condition.system_prompt_id,
        rendered_prompt_hash="d" * 64,
        steering_method=condition.steering_method,
        layer=31 if adaptive and action == "intervene" else condition.layer,
        site=(
            ActivationSite.BLOCK_OUTPUT
            if adaptive and action == "intervene"
            else condition.site
        ),
        token_scope=(
            TokenScope.FIRST_FOUR
            if adaptive and action == "intervene"
            else condition.token_scope
        ),
        alpha=adaptive_alpha if adaptive else condition.alpha,
        sparsity=condition.sparsity,
        controller_scores=scores,
        raw_output=outcome.value,
        normalized_answer=outcome.value,
        outcome=outcome,
        generation_latency_seconds=0.1,
        input_tokens=10,
        output_tokens=1,
        condition_id=condition.condition_id,
        seed=17,
        metadata={
            "phase": ExperimentPhase.E4.value,
            "partition": condition.partition,
            "prompt_template_sha256": condition.prompt_template_sha256,
            "study_protocol_digest": condition.study_protocol_digest,
            "method_artifact_sha256": condition.method_artifact_sha256,
            "intervention_trace": trace,
            "intervention_trace_digest": (
                stable_hash(trace) if trace is not None else None
            ),
            **({"policy_action": action} if adaptive else {}),
        },
    )
    if adaptive:
        assert policy.adaptive_policy is not None
        decided = replace(
            draft,
            metadata={
                **draft.metadata,
                "policy_decision_digest": adaptive_policy_decision_digest(
                    draft,
                    policy=policy.adaptive_policy,
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
                    policy=policy.adaptive_policy,
                    private_key_hex=_EXECUTION_PRIVATE_KEY,
                ),
            },
        )
    return replace(
        draft,
        metadata={
            **draft.metadata,
            "execution_receipt_signature": sign_e4_fixed_execution_receipt(
                draft,
                policy=policy,
                policy_artifact_sha256=condition.method_artifact_sha256 or "",
                private_key_hex=_EXECUTION_PRIVATE_KEY,
            ),
        },
    )


def _ledger(
    report: E4CapabilityReport,
    policies: dict[str, E4MethodPolicy],
    policy_paths: dict[str, Path],
    question_ids: tuple[str, ...],
) -> tuple[PhaseRunLedger, tuple[GenerationRecord, ...]]:
    conditions = tuple(
        _condition(policies, policy_paths, method, prompt)
        for prompt in ("P0-neutral", "P2-calibrated-abstention")
        for method in report.feasible_methods
    )
    contract = PhaseRunContract(
        phase=ExperimentPhase.E4,
        study_protocol_digest=load_study_protocol(
            ROOT / "configs/experiments/phases.yaml"
        ).digest,
        conditions=conditions,
        question_ids_by_benchmark={"triviaqa": question_ids},
        input_fingerprints=report.source_digests,
        prerequisite_digests={"E3": "e" * 64},
        required_gates=("promotion_decision_frozen",),
    )
    records = tuple(
        _record(
            condition,
            policies[condition.steering_method],
            question,
            (
                Outcome.CORRECT
                if condition.steering_method == "ACT-or-SADI"
                or (condition.steering_method == "M2" and index % 2 == 0)
                or index == 0
                else Outcome.INCORRECT
            ),
        )
        for condition in conditions
        for index, question in enumerate(question_ids)
    )
    ledger = object.__new__(PhaseRunLedger)
    ledger.contract = contract
    return ledger, records


class _NativeE4Runtime:
    """Deterministic stand-in that exercises the native signed-record boundary."""

    def __init__(self, identity: dict[str, Any] | None = None) -> None:
        self.identity = identity or {}

    def runtime_identity(self) -> dict[str, Any]:
        return dict(self.identity)

    def render_prompt(
        self,
        prompt: Any,
        question: str,
        *,
        metadata: Any | None = None,
    ) -> MlxRenderedPrompt:
        del metadata
        text = f"{prompt.text}\n{question}"
        token_ids = (1, 2, 3)
        return MlxRenderedPrompt(
            text=text,
            sha256=hashlib.sha256(text.encode()).hexdigest(),
            token_ids=token_ids,
            token_ids_sha256=hashlib.sha256(b"1,2,3").hexdigest(),
            messages=(),
        )

    @staticmethod
    def _generation(rendered: MlxRenderedPrompt) -> MlxGenerationOutput:
        return MlxGenerationOutput(
            rendered_prompt=rendered,
            token_ids=(7, 8, 9, 10, 11, 12),
            text="gold",
            input_tokens=3,
            output_tokens=6,
            latency_seconds=0.1,
            stop_type="length",
            stopping_token_id=None,
            prompt_tokens_per_second=10.0,
            generation_tokens_per_second=5.0,
            peak_memory_bytes=1_024,
            active_memory_bytes=512,
            cache_memory_bytes=256,
        )

    def generate(
        self, rendered: MlxRenderedPrompt, *, max_new_tokens: int
    ) -> MlxGenerationOutput:
        assert max_new_tokens == 48
        return self._generation(rendered)

    def generate_with_interventions(
        self,
        rendered: MlxRenderedPrompt,
        *,
        max_new_tokens: int,
        intervention_states: dict[
            tuple[int, ActivationSite], MlxResearchInterventionState
        ],
    ) -> MlxGenerationOutput:
        assert max_new_tokens == 48
        for state in intervention_states.values():
            for index in range(4):
                captured = np.full(5_120, index, dtype=np.float32)
                intervened = captured + np.asarray(state.direction) * state.alpha
                state.applied_pre_history.append(captured)
                state.applied_post_history.append(intervened)
            state.captured = np.full((1, 1, 5_120), 5.0, dtype=np.float32)
            state.intervened = state.captured
            state.applications = 4
        return self._generation(rendered)

    def standardized_intervention_state(
        self,
        direction: np.ndarray[Any, Any],
        *,
        standardized_alpha: float,
        reference_rms: float,
        token_scope: TokenScope,
        decay: float = 0.0,
    ) -> MlxResearchInterventionState:
        return MlxResearchInterventionState(
            direction=direction,
            alpha=standardized_alpha * reference_rms,
            token_scope=token_scope,
            decay=decay,
        )

    def prompt_feature_cube(
        self,
        rendered: MlxRenderedPrompt,
        *,
        layers: tuple[int, ...],
        sites: tuple[ActivationSite, ...],
    ) -> MlxPromptFeatureCubeOutput:
        del rendered
        return MlxPromptFeatureCubeOutput(
            activations={
                site: {
                    layer: np.zeros((1, 5_120), dtype=np.float32)
                    for layer in layers
                }
                for site in sites
            },
            maximum_token_probability=0.5,
            output_entropy=1.0,
            peak_memory_bytes=512,
        )


def test_native_e4_record_builders_execute_and_sign_exact_edits(tmp_path: Path) -> None:
    report = _report(tmp_path)
    policies, paths = _policies(tmp_path, report)
    runtime = _NativeE4Runtime()
    question = _questions(1)[0]
    prompt = {
        value.prompt_id: value
        for value in load_prompt_specs(ROOT / "configs/prompts/primary.yaml")
    }["P0-neutral"]

    fixed_condition = _condition(policies, paths, "M2", prompt.prompt_id)
    fixed = _fixed_record(
        runtime=runtime,
        condition=fixed_condition,
        policy=policies["M2"],
        policy_path=paths["M2"],
        direction=resolve_static_direction(
            report.artifact_paths["implementation:M2"],
            method="M2",
            layer=31,
            site=ActivationSite.BLOCK_OUTPUT,
        ),
        question=question,
        prompt=prompt,
        private_key_hex=_EXECUTION_PRIVATE_KEY,
    )
    fixed_condition.validate_record(fixed)
    validate_e4_fixed_execution_record(
        fixed,
        policy=policies["M2"],
        policy_artifact_sha256=sha256_path(paths["M2"]),
    )
    assert fixed.metadata["intervention_trace"]["applied_token_indices"] == [0, 1, 2, 3]
    fixed_metrics = dict(fixed.metadata["generation_runtime_metrics"])
    fixed_metrics["peak_memory_bytes"] = 1
    with pytest.raises(DataValidationError, match="signed by the frozen runtime"):
        validate_e4_fixed_execution_record(
            replace(
                fixed,
                metadata={**fixed.metadata, "generation_runtime_metrics": fixed_metrics},
            ),
            policy=policies["M2"],
            policy_artifact_sha256=sha256_path(paths["M2"]),
        )

    adaptive_condition = _condition(policies, paths, "ACT-or-SADI", prompt.prompt_id)
    adaptive = _adaptive_record(
        runtime=runtime,
        condition=adaptive_condition,
        policy=policies["ACT-or-SADI"],
        baseline=verify_e4_act_baseline(
            report.artifact_paths["implementation:ACT-or-SADI"]
        ),
        question=question,
        prompt=prompt,
        private_key_hex=_EXECUTION_PRIVATE_KEY,
    )
    adaptive_condition.validate_record(adaptive)
    assert adaptive.metadata["policy_action"] == "intervene"
    assert adaptive.metadata["intervention_trace"]["applied_token_indices"] == [0, 1, 2, 3]
    baseline = verify_e4_act_baseline(
        report.artifact_paths["implementation:ACT-or-SADI"]
    )
    assert baseline.direction.reference_rms != 1.0
    assert adaptive.metadata["intervention_trace"]["direction_norm"] == pytest.approx(
        baseline.direction.direction_norm * baseline.direction.reference_rms
    )
    _verify_act_controller_replay(adaptive, baseline=baseline)
    forged_scores = replace(
        adaptive,
        controller_scores={"C": 0.2, "I": 0.7, "A": 0.1},
    )
    with pytest.raises(FrozenArtifactError, match="scores differ"):
        _verify_act_controller_replay(forged_scores, baseline=baseline)
    controller_evidence = dict(adaptive.metadata["adaptive_controller_evidence"])
    controller_evidence["feature_values"] = [1.0, *controller_evidence["feature_values"][1:]]
    tampered = replace(
        adaptive,
        metadata={**adaptive.metadata, "adaptive_controller_evidence": controller_evidence},
    )
    with pytest.raises(DataValidationError):
        adaptive_condition.validate_record(tampered)
    adaptive_metrics = dict(adaptive.metadata["generation_runtime_metrics"])
    adaptive_metrics["peak_memory_bytes"] = 1
    with pytest.raises(DataValidationError, match="signed by the frozen runtime key"):
        adaptive_condition.validate_record(
            replace(
                adaptive,
                metadata={
                    **adaptive.metadata,
                    "generation_runtime_metrics": adaptive_metrics,
                },
            )
        )


def test_public_act_builder_rejects_self_declared_miniature_e2(tmp_path: Path) -> None:
    miniature = build_e2_probe_bundle(tmp_path)
    with pytest.raises(FrozenArtifactError):
        build_e4_act_baseline(
            tmp_path / "act",
            e2_probe_bundle=miniature,
            e2_workspace=miniature,
            e2_phase_run=miniature,
            m2_caa_artifact=miniature,
            intervention_layer=31,
            study=load_study_protocol(ROOT / "configs/experiments/phases.yaml"),
        )


def test_e4_applied_edit_history_rejects_wrong_scale_and_direction() -> None:
    direction = np.zeros(5_120, dtype=np.float32)
    direction[0] = 1.0
    state = MlxResearchInterventionState(
        direction=direction,
        alpha=2.0,
        token_scope=TokenScope.FIRST_FOUR,
    )
    for _ in range(4):
        before = np.zeros(5_120, dtype=np.float32)
        after = before.copy()
        after[1] = 4.0
        state.applied_pre_history.append(before)
        state.applied_post_history.append(after)
    state.applications = 4
    state.captured = np.zeros((1, 1, 5_120), dtype=np.float32)
    state.intervened = state.captured
    with pytest.raises(FrozenArtifactError, match="direction and alpha"):
        _strict_runtime_arrays(state, expected_applications=4)


def test_native_e4_run_and_portable_verify_cover_long_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mfh.experiments import e4_mlx as module

    report = _report(tmp_path)
    policies, paths = _policies(tmp_path, report)
    screen = build_e4_screen_receipt(_questions(), protocol=_protocol())
    conditions = tuple(
        _condition(policies, paths, method, prompt)
        for prompt in ("P0-neutral", "P2-calibrated-abstention")
        for method in report.feasible_methods
    )
    contract = PhaseRunContract(
        phase=ExperimentPhase.E4,
        study_protocol_digest=load_study_protocol(
            ROOT / "configs/experiments/phases.yaml"
        ).digest,
        conditions=conditions,
        question_ids_by_benchmark={"triviaqa": screen.screen_question_ids},
        input_fingerprints=report.source_digests,
        prerequisite_digests={"E3": "e" * 64},
        required_gates=("promotion_decision_frozen",),
    )

    class MemoryLedger:
        def __init__(self) -> None:
            self.contract = contract
            self.values: list[GenerationRecord] = []

        def iter_pending(self):
            completed = {(value.condition_id, value.question_id) for value in self.values}
            for condition in self.contract.conditions:
                for question_id in screen.screen_question_ids:
                    if (condition.condition_id, question_id) not in completed:
                        yield SimpleNamespace(condition=condition, question_id=question_id)

        def checkpoint(self, values):
            rows = tuple(values)
            condition_by_id = {
                condition.condition_id: condition for condition in self.contract.conditions
            }
            for row in rows:
                condition_by_id[row.condition_id].validate_record(row)
            self.values.extend(rows)
            return len(rows)

        def progress(self):
            return len(self.values), self.contract.expected_record_count

        def records(self):
            return iter(self.values)

        def record_set_digest(self):
            return stable_hash([value.to_dict() for value in self.values])

    ledger = MemoryLedger()
    setup = E4MlxSetup(
        directory=tmp_path,
        report=report,
        screen=screen,
        policies=MappingProxyType(policies),
        policy_paths=MappingProxyType(paths),
    )
    monkeypatch.setattr(module, "load_e4_mlx_setup", lambda _path: setup)
    monkeypatch.setattr(
        module,
        "PhaseRunLedger",
        SimpleNamespace(open=lambda _path, *, study: ledger),
    )
    m2_plan = json.loads(
        (
            Path(report.artifact_paths["implementation:M2"]) / "plan.json"
        ).read_text(encoding="utf-8")
    )
    runtime = _NativeE4Runtime(cast(dict[str, Any], m2_plan["runtime_identity"]))
    prompts = {
        value.prompt_id: value
        for value in load_prompt_specs(ROOT / "configs/prompts/primary.yaml")
    }
    study = load_study_protocol(ROOT / "configs/experiments/phases.yaml")
    mutated_prompts = {
        **prompts,
        "P0-neutral": replace(
            prompts["P0-neutral"], text=prompts["P0-neutral"].text + " changed"
        ),
    }
    with pytest.raises(FrozenArtifactError, match="prompt text"):
        run_e4_mlx_screen(
            tmp_path,
            tmp_path,
            study=study,
            prompts=mutated_prompts,
            runtime=runtime,
            execution_private_key_hex=_EXECUTION_PRIVATE_KEY,
        )
    with pytest.raises(FrozenArtifactError, match="execution key"):
        run_e4_mlx_screen(
            tmp_path,
            tmp_path,
            study=study,
            prompts=prompts,
            runtime=runtime,
            execution_private_key_hex="0" * 64,
        )
    wrong_identity = dict(runtime.identity)
    wrong_identity["model_revision"] = "0" * 40
    with pytest.raises(FrozenArtifactError, match="scientific M2/E3 runtime"):
        run_e4_mlx_screen(
            tmp_path,
            tmp_path,
            study=study,
            prompts=prompts,
            runtime=_NativeE4Runtime(wrong_identity),
            execution_private_key_hex=_EXECUTION_PRIVATE_KEY,
        )
    partial = run_e4_mlx_screen(
        tmp_path,
        tmp_path,
        study=study,
        prompts=prompts,
        runtime=runtime,
        execution_private_key_hex=_EXECUTION_PRIVATE_KEY,
        request_budget=5,
        checkpoint_rows=3,
    )
    assert partial["complete"] is False
    assert partial["completed"] == 5
    assert verify_e4_mlx_screen(
        tmp_path,
        tmp_path,
        study=study,
    )["complete"] is False
    result = run_e4_mlx_screen(
        tmp_path,
        tmp_path,
        study=study,
        prompts=prompts,
        runtime=runtime,
        execution_private_key_hex=_EXECUTION_PRIVATE_KEY,
        checkpoint_rows=3,
    )
    assert result["complete"] is True
    assert len(ledger.values) == 12
    assert all(value.output_tokens == 6 for value in ledger.values)
    replay = verify_e4_mlx_screen(
        tmp_path,
        tmp_path,
        study=study,
        require_complete=True,
    )
    assert replay["valid"] is True
    assert replay["completed"] == 12


def test_e4_capabilities_screen_and_restart_replay(tmp_path: Path) -> None:
    report = _report(tmp_path)
    assert report.feasible_methods == ("M1", "M2", "ACT-or-SADI")
    assert len(build_e4_conditions(report)) == 6
    screen = build_e4_screen_receipt(_questions(), protocol=_protocol())
    assert len(screen.screen_question_ids) == 2

    report_path = tmp_path / "capabilities.json"
    write_e4_capability_report(report_path, report)
    assert verify_e4_capability_report(report_path, expected=report)["valid"] is True
    screen_path = tmp_path / "screen.json"
    write_e4_screen_receipt(screen_path, screen)
    assert verify_e4_screen_receipt(screen_path, expected=screen)["valid"] is True
    assert load_e4_capability_report(report_path) == report
    assert load_e4_screen_receipt(screen_path) == screen


def test_e4_promotion_passes_registered_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _report(tmp_path)
    protocol = E4Protocol()
    dev_questions = tuple(
        read_questions(ROOT / "artifacts/splits/triviaqa-reviewed/T-dev.jsonl")
    )
    screen = build_e4_screen_receipt(dev_questions, protocol=protocol)
    assert screen.scientific_eligible is True
    policies, policy_paths = _policies(tmp_path, report)
    ledger, records = _ledger(
        report,
        policies,
        policy_paths,
        screen.screen_question_ids,
    )
    record_set_digest = stable_hash([value.to_dict() for value in records])
    ledger_directory = tmp_path / "restarted-ledger"
    ledger_directory.mkdir()
    monkeypatch.setattr(
        PhaseRunLedger,
        "open",
        classmethod(lambda cls, directory, *, study: ledger),
    )
    monkeypatch.setattr(
        PhaseRunLedger,
        "_verify_creation_evidence",
        lambda self: None,
    )
    monkeypatch.setattr(
        PhaseRunLedger,
        "progress",
        lambda self: (len(records), len(records)),
    )
    monkeypatch.setattr(PhaseRunLedger, "records", lambda self: iter(records))
    monkeypatch.setattr(
        PhaseRunLedger,
        "record_set_digest",
        lambda self: record_set_digest,
    )
    study = load_study_protocol(ROOT / "configs/experiments/phases.yaml")
    report_path = tmp_path / "promotion-capabilities.json"
    screen_path = tmp_path / "promotion-screen.json"
    write_e4_capability_report(report_path, report)
    write_e4_screen_receipt(screen_path, screen)
    promotion = derive_e4_promotion(
        ledger_directory=ledger_directory,
        study=study,
        report=report,
        screen=screen,
        method_policy_artifacts=policy_paths,
        protocol=protocol,
    )
    assert promotion.promoted_methods == ("ACT-or-SADI", "M2")
    assert len(promotion.selection_manifest["selected_condition_ids"]) == 6
    path = tmp_path / "promotion.json"
    write_e4_promotion(
        path,
        promotion,
        ledger_directory=ledger_directory,
        study=study,
        report=report,
        screen=screen,
        method_policy_artifacts=policy_paths,
        protocol=protocol,
    )
    assert verify_e4_promotion(
        path,
        ledger_directory=ledger_directory,
        study=study,
        report=report,
        screen=screen,
        method_policy_artifacts=policy_paths,
        protocol=protocol,
    )["valid"] is True
    gate_parameters, supporting_artifacts = build_e4_promotion_gate_bundle(
        capability_report_path=report_path,
        screen_receipt_path=screen_path,
        promotion_path=path,
        method_policy_artifacts=policy_paths,
    )

    facts = {
        condition.condition_id: {
            "model_repository": condition.model_repository,
            "model_name": condition.model_name,
            "model_revision": condition.model_revision,
            "runtime": condition.runtime.value,
            "quantization": condition.quantization,
            "model_num_layers": condition.model_num_layers,
            "benchmark": condition.benchmark,
            "system_prompt_id": condition.system_prompt_id,
            "partition": condition.partition,
                "steering_method": condition.steering_method,
                "method_artifact_sha256": condition.method_artifact_sha256,
                "comparison_group": condition.comparison_group,
        }
        for condition in ledger.contract.conditions
    }
    context = GateEvaluationContext(
        expected_record_count=len(records),
        records_factory=lambda: iter(records),
        expected_condition_ids=frozenset(facts),
        condition_facts=facts,
        input_fingerprints=report.source_digests,
    )
    evidence = tmp_path / "gate-evidence.json"
    write_gate_evidence(
        evidence,
        phase=ExperimentPhase.E4,
        gate="promotion_decision_frozen",
        contract_digest=ledger.contract.digest,
        record_set_digest=ledger.record_set_digest(),
        observations=(),
        parameters=gate_parameters,
    )
    with pytest.raises(DataValidationError, match="supporting artifacts"):
        evaluate_gate(
            phase=ExperimentPhase.E4,
            gate="promotion_decision_frozen",
            contract_digest=ledger.contract.digest,
            record_set_digest=ledger.record_set_digest(),
            evidence_path=evidence,
            context=context,
        )
    result = evaluate_gate(
        phase=ExperimentPhase.E4,
        gate="promotion_decision_frozen",
        contract_digest=ledger.contract.digest,
        record_set_digest=ledger.record_set_digest(),
        evidence_path=evidence,
        context=context,
        supporting_artifacts=supporting_artifacts,
    )
    assert result.passed is True
    assert set(result.artifact_fingerprints) == {
        "evaluation",
        *supporting_artifacts,
    }
    packaged = tmp_path / "packaged-gate"
    packaged.mkdir()
    shutil.copy2(evidence, packaged / "evaluation")
    for name, source in supporting_artifacts.items():
        if source.is_dir():
            shutil.copytree(source, packaged / name)
        else:
            shutil.copy2(source, packaged / name)
    result_path = tmp_path / "gate-result.json"
    write_gate_result(result_path, result)
    restarted_result = read_gate_result(result_path)
    validate_gate_result(
        restarted_result,
        evidence_path=packaged / "evaluation",
        context=context,
    )


def test_e4_promotion_explicitly_rejects_mandatory_m2_coverage_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _report(tmp_path)
    protocol = E4Protocol()
    screen = build_e4_screen_receipt(
        tuple(read_questions(ROOT / "artifacts/splits/triviaqa-reviewed/T-dev.jsonl")),
        protocol=protocol,
    )
    policies, policy_paths = _policies(tmp_path, report)
    ledger, _ = _ledger(report, policies, policy_paths, screen.screen_question_ids)
    records = tuple(
        _record(
            condition,
            policies[condition.steering_method],
            question_id,
            (
                Outcome.INCORRECT
                if condition.steering_method == "M2" and index == 0
                else Outcome.ABSTENTION
                if condition.steering_method == "M2"
                else Outcome.CORRECT
            ),
        )
        for condition in ledger.contract.conditions
        for index, question_id in enumerate(screen.screen_question_ids)
    )
    monkeypatch.setattr(
        PhaseRunLedger,
        "open",
        classmethod(lambda cls, directory, *, study: ledger),
    )
    monkeypatch.setattr(PhaseRunLedger, "_verify_creation_evidence", lambda self: None)
    monkeypatch.setattr(
        PhaseRunLedger, "progress", lambda self: (len(records), len(records))
    )
    monkeypatch.setattr(PhaseRunLedger, "records", lambda self: iter(records))
    monkeypatch.setattr(
        PhaseRunLedger,
        "record_set_digest",
        lambda self: stable_hash([value.to_dict() for value in records]),
    )
    with pytest.raises(DataValidationError, match="mandatory M2"):
        derive_e4_promotion(
            ledger_directory=tmp_path / "ledger",
            study=load_study_protocol(ROOT / "configs/experiments/phases.yaml"),
            report=report,
            screen=screen,
            method_policy_artifacts=policy_paths,
            protocol=protocol,
        )


def test_e4_static_vectors_must_equal_completed_e3_output(tmp_path: Path) -> None:
    report = _report(tmp_path)
    policies, policy_paths = _policies(tmp_path, report)
    ledger, _ = _ledger(report, policies, policy_paths, ("q-0", "q-1"))
    e3 = SimpleNamespace(
        prerequisite_digests={"E1": "a" * 64, "E2": "b" * 64},
        output_fingerprints={"E3_static_vectors": "0" * 64},
    )
    with pytest.raises(FrozenArtifactError, match="completed E3 output"):
        _validate_e3_prerequisite_lineage(
            {ExperimentPhase.E3: e3},
            contract=ledger.contract,
        )


def test_e4_receipts_detect_mutation_and_live_source_changes(tmp_path: Path) -> None:
    report = _report(tmp_path)
    screen = build_e4_screen_receipt(_questions(), protocol=_protocol())
    object.__setattr__(screen, "screen_question_ids", ("forged-a", "forged-b"))
    with pytest.raises((FrozenArtifactError, DataValidationError)):
        screen.assert_current()

    runtime = tmp_path / "runtime.py"
    runtime.write_text("changed", encoding="utf-8")
    with pytest.raises(FrozenArtifactError, match="changed"):
        report.assert_current()


def test_e4_screen_detects_mutated_question_content(tmp_path: Path) -> None:
    questions = _questions()
    screen = build_e4_screen_receipt(questions, protocol=_protocol())
    cast(dict[str, Any], questions[0].metadata)["changed"] = True
    with pytest.raises((FrozenArtifactError, DataValidationError)):
        screen.assert_current()


def test_e4_fixed_policy_and_signed_execution_are_bounded(tmp_path: Path) -> None:
    report = _report(tmp_path)
    with pytest.raises(DataValidationError, match="method policy"):
        write_e4_method_policy(
            tmp_path / "invalid-policy.json",
            report=report,
            method="M1",
            layer=64,
            site=ActivationSite.POST_MLP,
            token_scope=TokenScope.FIRST_FOUR,
            alpha=1.0,
            execution_public_key=_EXECUTION_PUBLIC_KEY,
            direction_sha256=policy_direction(report, "M1"),
            direction_norm=1.0,
        )
    with pytest.raises(DataValidationError, match="differs from its construction"):
        write_e4_method_policy(
            tmp_path / "forged-direction.json",
            report=report,
            method="M1",
            layer=31,
            site=ActivationSite.POST_MLP,
            token_scope=TokenScope.FIRST_FOUR,
            alpha=1.0,
            execution_public_key=_EXECUTION_PUBLIC_KEY,
            direction_sha256="0" * 64,
        )
    with pytest.raises(DataValidationError, match="differs from its construction"):
        write_e4_method_policy(
            tmp_path / "forged-rms.json",
            report=report,
            method="M2",
            layer=31,
            site=ActivationSite.BLOCK_OUTPUT,
            token_scope=TokenScope.FIRST_FOUR,
            alpha=1.0,
            execution_public_key=_EXECUTION_PUBLIC_KEY,
            reference_rms=99.0,
        )
    with pytest.raises(DataValidationError, match="residual block output"):
        write_e4_method_policy(
            tmp_path / "wrong-m2-site.json",
            report=report,
            method="M2",
            layer=31,
            site=ActivationSite.POST_MLP,
            token_scope=TokenScope.FIRST_FOUR,
            alpha=1.0,
            execution_public_key=_EXECUTION_PUBLIC_KEY,
        )

    policies, paths = _policies(tmp_path, report)
    condition = _condition(policies, paths, "M1", "P0-neutral")
    record = _record(condition, policies["M1"], "dev-0", Outcome.CORRECT)
    validate_e4_fixed_execution_record(
        record,
        policy=policies["M1"],
        policy_artifact_sha256=sha256_path(paths["M1"]),
    )
    trace = dict(record.metadata["intervention_trace"])
    trace["post_activation_sha256"] = trace["pre_activation_sha256"]
    forged = replace(
        record,
        metadata={
            **record.metadata,
            "intervention_trace": trace,
            "intervention_trace_digest": stable_hash(trace),
        },
    )
    with pytest.raises(DataValidationError, match="does not prove"):
        validate_e4_fixed_execution_record(
            forged,
            policy=policies["M1"],
            policy_artifact_sha256=sha256_path(paths["M1"]),
        )

    coerced_trace = dict(record.metadata["intervention_trace"])
    coerced_trace.update(
        {
            "applied_tokens": True,
            "applied_token_indices": [False],
            "alpha": True,
            "direction_norm": True,
        }
    )
    coerced_draft = replace(
        record,
        metadata={
            **record.metadata,
            "intervention_trace": coerced_trace,
            "intervention_trace_digest": stable_hash(coerced_trace),
        },
    )
    coerced = replace(
        coerced_draft,
        metadata={
            **coerced_draft.metadata,
            "execution_receipt_signature": sign_e4_fixed_execution_receipt(
                coerced_draft,
                policy=policies["M1"],
                policy_artifact_sha256=sha256_path(paths["M1"]),
                private_key_hex=_EXECUTION_PRIVATE_KEY,
            ),
        },
    )
    with pytest.raises(DataValidationError, match="does not prove"):
        validate_e4_fixed_execution_record(
            coerced,
            policy=policies["M1"],
            policy_artifact_sha256=sha256_path(paths["M1"]),
        )

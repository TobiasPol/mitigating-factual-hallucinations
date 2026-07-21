"""Resumable native-MLX operator lifecycle for the registered E8 study."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from mfh.config import load_model_spec, load_prompt_specs
from mfh.contracts import (
    ActivationSite,
    AdaptivePolicySpec,
    GenerationRecord,
    ModelSpec,
    Outcome,
    PromptSpec,
    Question,
    TokenScope,
)
from mfh.data.io import read_questions
from mfh.data.language_suite import load_reviewed_language_suite
from mfh.data.side_effect_sampling import select_mmlu_pro_stratified
from mfh.data.source_snapshots import SOURCE_SNAPSHOTS, iter_source_questions
from mfh.errors import ConfigurationError, DataValidationError, FrozenArtifactError
from mfh.experiments.e6_likelihood import E6RuntimeAttestor, _load_e6_runtime_attestation
from mfh.experiments.e6_operator import _e3_slice
from mfh.experiments.e7_e8_grading import E7E8DevelopmentGrader
from mfh.experiments.e7_sparse import verify_e7_phase
from mfh.experiments.e8_protected import (
    BehaviorActivationEvidence,
    BehaviorLabelPair,
    E8CandidatePoint,
    E8CandidateScreen,
    E8ProtectedArtifact,
    M5VariantScreen,
    _complete_e7_behavior_label_pairs,
    _unit,
    _within_class_behavior_changes,
    _write_e8_behavior_activation_bundle,
    build_e8_protected_artifact,
    execute_e8_adaptive_generation,
    execute_e8_generation,
    finalize_e8_phase,
    load_e8_behavior_activation_bundle,
    load_e8_candidate_screen,
    load_e8_protected_artifact,
    question_source_fingerprint,
    response_verbosity_style_preserved,
    save_e8_candidate_screen,
    save_e8_protected_artifact,
    verify_e8_phase,
)
from mfh.experiments.model_selection import (
    validate_active_model_spec,
    validate_active_study_artifact_paths,
)
from mfh.experiments.protocol import ExperimentPhase, StudyProtocol, load_study_protocol
from mfh.experiments.runner import (
    EvaluationCondition,
    PhaseRunContract,
    PhaseRunLedger,
    open_phase_prerequisite,
    validate_side_effect_evaluation_bundle,
    write_side_effect_evaluation_bundle,
)
from mfh.inference.mlx_research import MlxResearchRuntime
from mfh.inference.transformers_snapshot import verify_transformers_snapshot
from mfh.methods.adaptive import load_adaptive_controller
from mfh.methods.features import (
    ActivationFeatureSchema,
    ActivationKind,
    FeatureComposition,
)
from mfh.methods.protected import (
    E8OperatingPointRegistry,
    behavior_covariance,
    build_protected_subspace,
    covariance_aware_direction,
    load_e8_operating_point_registry,
    save_e8_operating_point_registry,
)
from mfh.methods.sparse import load_coordinate_sparse_artifact, load_sae_intervention
from mfh.provenance import canonical_json, sha256_file, sha256_path, stable_hash

_PROMPTS = ("P0-neutral", "P2-calibrated-abstention")
_METHODS = ("M0", "M1", "M3", "M4", "M5")
_SCREEN_METHODS = ("M1", "M3", "M4", "M5")
_ALPHAS = (0.1, 0.25, 0.5, 1.0, 2.0)
_BEHAVIORS = (
    "correct_to_abstain",
    "xstest_safe_refusal",
    "harmful_refusal",
    "language_switching",
    "instruction_following_failure",
    "verbosity_style",
)
_BEHAVIOR_BENCHMARKS = {
    "correct_to_abstain": "triviaqa",
    "xstest_safe_refusal": "xstest",
    "harmful_refusal": "strongreject_or_harmbench",
    "language_switching": "language_consistency",
    "instruction_following_failure": "ifeval",
    "verbosity_style": "ifeval",
}
_QUESTION_COUNTS = {
    "triviaqa": 5_000,
    "ifeval": 541,
    "mmlu_pro": 1_000,
    "wikitext103": 1_000,
    "xstest": 250,
    "strongreject_or_harmbench": 313,
    "language_consistency": 500,
}
_RUNBOOK_KEYS = {
    "schema_version",
    "phase",
    "study_protocol",
    "analysis_protocol",
    "research_plan",
    "model_config",
    "prompt_config",
    "snapshot_directory",
    "snapshot_manifest",
    "environment_file",
    "execution_key_file",
    "runtime_artifact",
    "package_lock",
    "reviewed_splits",
    "source_artifacts",
    "reviewed_language_suite",
    "ifeval_evaluator",
    "e6_transition_evidence",
    "e7_finalization",
    "prerequisite_runs",
    "outputs",
    "seed",
    "max_new_tokens",
    "candidate_question_count",
    "variant_factual_rows",
    "variant_protected_rows",
    "matching_dimension",
    "matching_tolerance",
    "m1_tensor_index",
    "m5_alpha",
}
_SOURCE_KEYS = {
    "triviaqa",
    "ifeval",
    "mmlu_pro",
    "wikitext103",
    "xstest",
    "strongreject_or_harmbench",
}
_OUTPUT_KEYS = {
    "development_side_effect_bundle",
    "side_effect_bundle",
    "activation_work",
    "protected_behavior_activations",
    "variant_work",
    "variant_screens",
    "protected_artifact",
    "candidate_work",
    "candidate_screen",
    "operating_point_registry",
    "final_work",
    "run_directory",
    "final_directory",
}


def _path(root: Path, value: object, context: str) -> Path:
    if type(value) is not str or not value.strip():
        raise DataValidationError(f"E8 runbook {context} path is invalid")
    raw = Path(value)
    return (raw if raw.is_absolute() else root / raw).resolve()


def _path_map(root: Path, value: object, expected: set[str], context: str) -> Mapping[str, Path]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise DataValidationError(f"E8 runbook {context} keys differ")
    return MappingProxyType(
        {name: _path(root, raw, f"{context}.{name}") for name, raw in value.items()}
    )


@dataclass(frozen=True, slots=True)
class E8Runbook:
    source: Path
    study_protocol: Path
    analysis_protocol: Path
    research_plan: Path
    model_config: Path
    prompt_config: Path
    snapshot_directory: Path
    snapshot_manifest: Path
    environment_file: Path
    execution_key_file: Path
    runtime_artifact: Path
    package_lock: Path
    reviewed_splits: Path
    source_artifacts: Mapping[str, Path]
    reviewed_language_suite: Path
    ifeval_evaluator: Path
    e6_transition_evidence: Path
    e7_finalization: Path
    prerequisite_runs: Mapping[ExperimentPhase, Path]
    outputs: Mapping[str, Path]
    seed: int
    max_new_tokens: int
    candidate_question_count: int
    variant_factual_rows: int
    variant_protected_rows: int
    matching_dimension: str
    matching_tolerance: float
    m1_tensor_index: tuple[str, str, str, int]
    m5_alpha: float
    runbook_digest: str

    @classmethod
    def load(cls, path: str | Path) -> E8Runbook:
        source = Path(path).resolve()
        try:
            value = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FrozenArtifactError(f"cannot read E8 runbook: {exc}") from exc
        prerequisites = value.get("prerequisite_runs") if isinstance(value, dict) else None
        tensor = value.get("m1_tensor_index") if isinstance(value, dict) else None
        if (
            not isinstance(value, dict)
            or set(value) != _RUNBOOK_KEYS
            or value["schema_version"] != 1
            or value["phase"] != "E8"
            or not isinstance(prerequisites, Mapping)
            or set(prerequisites) != {"E6", "E7"}
            or type(tensor) is not list
            or len(tensor) != 4
            or any(type(item) is not str for item in tensor[:3])
            or type(tensor[3]) is not int
            or value["matching_dimension"] not in {"hallucination_risk", "coverage"}
            or type(value["max_new_tokens"]) is not int
            or not 32 <= value["max_new_tokens"] <= 48
            or isinstance(value["matching_tolerance"], bool)
            or not isinstance(value["matching_tolerance"], int | float)
            or not 0 <= float(value["matching_tolerance"]) <= 0.02
            or value["m5_alpha"] not in _ALPHAS
            or any(
                type(value[name]) is not int or value[name] <= (0 if name != "seed" else -1)
                for name in (
                    "seed",
                    "candidate_question_count",
                    "variant_factual_rows",
                    "variant_protected_rows",
                )
            )
        ):
            raise DataValidationError("E8 runbook differs from schema version 1")
        root = source.parent
        return cls(
            source=source,
            study_protocol=_path(root, value["study_protocol"], "study_protocol"),
            analysis_protocol=_path(root, value["analysis_protocol"], "analysis_protocol"),
            research_plan=_path(root, value["research_plan"], "research_plan"),
            model_config=_path(root, value["model_config"], "model_config"),
            prompt_config=_path(root, value["prompt_config"], "prompt_config"),
            snapshot_directory=_path(root, value["snapshot_directory"], "snapshot_directory"),
            snapshot_manifest=_path(root, value["snapshot_manifest"], "snapshot_manifest"),
            environment_file=_path(root, value["environment_file"], "environment_file"),
            execution_key_file=_path(root, value["execution_key_file"], "execution_key_file"),
            runtime_artifact=_path(root, value["runtime_artifact"], "runtime_artifact"),
            package_lock=_path(root, value["package_lock"], "package_lock"),
            reviewed_splits=_path(root, value["reviewed_splits"], "reviewed_splits"),
            source_artifacts=_path_map(
                root, value["source_artifacts"], _SOURCE_KEYS, "source_artifacts"
            ),
            reviewed_language_suite=_path(
                root, value["reviewed_language_suite"], "reviewed_language_suite"
            ),
            ifeval_evaluator=_path(root, value["ifeval_evaluator"], "ifeval_evaluator"),
            e6_transition_evidence=_path(
                root, value["e6_transition_evidence"], "e6_transition_evidence"
            ),
            e7_finalization=_path(root, value["e7_finalization"], "e7_finalization"),
            prerequisite_runs=MappingProxyType(
                {
                    ExperimentPhase(name): _path(root, raw, f"prerequisite_runs.{name}")
                    for name, raw in prerequisites.items()
                }
            ),
            outputs=_path_map(root, value["outputs"], _OUTPUT_KEYS, "outputs"),
            seed=value["seed"],
            max_new_tokens=value["max_new_tokens"],
            candidate_question_count=value["candidate_question_count"],
            variant_factual_rows=value["variant_factual_rows"],
            variant_protected_rows=value["variant_protected_rows"],
            matching_dimension=value["matching_dimension"],
            matching_tolerance=float(value["matching_tolerance"]),
            m1_tensor_index=(tensor[0], tensor[1], tensor[2], tensor[3]),
            m5_alpha=float(value["m5_alpha"]),
            runbook_digest=stable_hash(value),
        )


def write_e8_runbook_template(path: str | Path, *, m1_layer: int) -> str:
    """Write the secret-free E8 runbook for an Apple-MLX 48-GiB host."""

    if isinstance(m1_layer, bool) or not isinstance(m1_layer, int) or not 0 <= m1_layer < 64:
        raise DataValidationError("E8 M1 layer must be an explicit Qwen layer index")
    destination = Path(path).resolve()
    if destination.exists() or destination.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite E8 runbook: {destination}")
    body = {
        "schema_version": 1,
        "phase": "E8",
        "study_protocol": "../../../../configs/experiments/phases.yaml",
        "analysis_protocol": "../../../../configs/analysis/confirmatory.yaml",
        "research_plan": "../../../../docs/research-plan.md",
        "model_config": "../../../../configs/models/qwen3.6-27b-mlx-4bit.yaml",
        "prompt_config": "../../../../configs/prompts/primary.yaml",
        "snapshot_directory": (
            "../../../models/qwen3.6-27b-mlx-4bit/"
            "c000ac2c2057d94be3fa931000c31723aac53282"
        ),
        "snapshot_manifest": "../../../../configs/models/qwen3.6-27b-mlx-4bit.snapshot.json",
        "environment_file": "../../../../.env",
        "execution_key_file": "../secrets/execution-private-key.hex",
        "runtime_artifact": (
            "../final/E6/gate-artifacts/"
            "knowledge_recovery_separated_from_abstention_substitution/"
            "likelihood-bundle/runtime-artifact"
        ),
        "package_lock": "../../../../uv.lock",
        "reviewed_splits": "../frozen/E7-E8-external-inputs/reviewed-splits",
        "source_artifacts": {
            "triviaqa": "../frozen/E7-E8-external-inputs/sources/triviaqa.parquet",
            "ifeval": "../frozen/E7-E8-external-inputs/sources/ifeval.jsonl",
            "mmlu_pro": "../frozen/E7-E8-external-inputs/sources/mmlu_pro.parquet",
            "wikitext103": "../frozen/E7-E8-external-inputs/sources/wikitext103.parquet",
            "xstest": "../frozen/E7-E8-external-inputs/sources/xstest.csv",
            "strongreject_or_harmbench": (
                "../frozen/E7-E8-external-inputs/sources/"
                "strongreject_or_harmbench.csv"
            ),
        },
        "reviewed_language_suite": "../frozen/E7-E8-external-inputs/language-suite",
        "ifeval_evaluator": "../frozen/E7-E8-external-inputs/ifeval-evaluator",
        "e6_transition_evidence": (
            "../runs/E6/gate-artifacts/"
            "knowledge_recovery_separated_from_abstention_substitution/likelihood-bundle"
        ),
        "e7_finalization": "../final/E7",
        "prerequisite_runs": {"E6": "../runs/E6", "E7": "../runs/E7"},
        "outputs": {
            "development_side_effect_bundle": "../work/E8/development-side-effects",
            "side_effect_bundle": "../frozen/E8-side-effects",
            "activation_work": "../work/E8/activations",
            "protected_behavior_activations": "../frozen/E8-protected-activations",
            "variant_work": "../work/E8/variant-screen",
            "variant_screens": "../frozen/E8-variant-screens.json",
            "protected_artifact": "../frozen/E8-protected-artifact",
            "candidate_work": "../work/E8/candidate-screen",
            "candidate_screen": "../frozen/E8-candidate-screen.json",
            "operating_point_registry": "../frozen/E8-operating-points.json",
            "final_work": "../work/E8/final-rows",
            "run_directory": "../runs/E8",
            "final_directory": "../final/E8",
        },
        "seed": 17,
        "max_new_tokens": 32,
        "candidate_question_count": 500,
        "variant_factual_rows": 500,
        "variant_protected_rows": 100,
        "matching_dimension": "hallucination_risk",
        "matching_tolerance": 0.02,
        "m1_tensor_index": ["P0-neutral", "M1-P", "post_mlp", m1_layer],
        "m5_alpha": 0.5,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sha256_file(destination)


@dataclass(frozen=True, slots=True)
class _E8Context:
    study: StudyProtocol
    model: ModelSpec
    prompts: Mapping[str, PromptSpec]
    questions: Mapping[str, tuple[Question, ...]]
    execution_private_key: str
    execution_public_key: str
    runtime_artifact_sha256: str
    runtime_identity: Mapping[str, Any]
    seed: int


def _private_key(runbook: E8Runbook) -> tuple[str, str]:
    try:
        private_hex = runbook.execution_key_file.read_text(encoding="utf-8").strip()
        private = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_hex))
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"cannot load E8 execution key: {exc}") from exc
    if len(private_hex) != 64 or private_hex.lower() != private_hex:
        raise ConfigurationError("E8 execution key must be one lowercase 32-byte hex key")
    public_hex = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    return private_hex, public_hex


def _load_source(path: Path, benchmark: str) -> tuple[Question, ...]:
    return tuple(iter_source_questions(SOURCE_SNAPSHOTS[benchmark], path))


def _select_wikitext(values: Sequence[Question], seed: int) -> tuple[Question, ...]:
    ranked = sorted(
        values,
        key=lambda item: (
            stable_hash(
                {"sampler": "wikitext-sha256-rank-v1", "seed": seed, "id": item.question_id}
            ),
            item.question_id,
        ),
    )
    selected = tuple(ranked[:1_000])
    if len(selected) != 1_000:
        raise DataValidationError("WikiText source cannot fill the registered E8 sample")
    return selected


def _base_context(runbook: E8Runbook) -> _E8Context:
    validate_active_study_artifact_paths(
        {
            "E8 runbook": runbook.source,
            "E8 reviewed splits": runbook.reviewed_splits,
            "E8 language suite": runbook.reviewed_language_suite,
            "E8 IFEval evaluator": runbook.ifeval_evaluator,
            "E8 transition evidence": runbook.e6_transition_evidence,
            "E8 E7 finalization": runbook.e7_finalization,
            "E8 runtime": runbook.runtime_artifact,
            **{f"E8 source {name}": path for name, path in runbook.source_artifacts.items()},
            **{
                f"E8 prerequisite {phase.value}": path
                for phase, path in runbook.prerequisite_runs.items()
            },
            **{f"E8 output {name}": path for name, path in runbook.outputs.items()},
        }
    )
    repository_lock = Path(__file__).resolve().parents[3] / "uv.lock"
    if sha256_file(runbook.package_lock) != sha256_file(repository_lock):
        raise FrozenArtifactError(
            "E8 runbook package lock differs from the executing repository lock"
        )
    study = load_study_protocol(runbook.study_protocol)
    model = load_model_spec(runbook.model_config)
    validate_active_model_spec(model)
    verify_transformers_snapshot(model, runbook.snapshot_directory, runbook.snapshot_manifest)
    prompt_values = {value.prompt_id: value for value in load_prompt_specs(runbook.prompt_config)}
    if not set(_PROMPTS) <= set(prompt_values):
        raise DataValidationError("E8 prompt configuration lacks P0/P2")
    prompts = MappingProxyType({name: prompt_values[name] for name in _PROMPTS})
    private_hex, public_hex = _private_key(runbook)
    runtime = _load_e6_runtime_attestation(runbook.runtime_artifact)
    runtime_identity = runtime["runtime_identity"]
    if (
        runtime["execution_public_key"] != public_hex
        or not isinstance(runtime_identity, Mapping)
        or runtime_identity.get("snapshot_sha256") != sha256_path(runbook.snapshot_directory)
    ):
        raise FrozenArtifactError("E8 key or snapshot differs from the E6 attestation")
    mmlu, _receipt = select_mmlu_pro_stratified(
        _load_source(runbook.source_artifacts["mmlu_pro"], "mmlu_pro"), seed=runbook.seed
    )
    questions = {
        "triviaqa": tuple(read_questions(runbook.reviewed_splits / "T-dev.jsonl")),
        "ifeval": _load_source(runbook.source_artifacts["ifeval"], "ifeval"),
        "mmlu_pro": mmlu,
        "wikitext103": _select_wikitext(
            _load_source(runbook.source_artifacts["wikitext103"], "wikitext103"),
            runbook.seed,
        ),
        "xstest": _load_source(runbook.source_artifacts["xstest"], "xstest"),
        "strongreject_or_harmbench": _load_source(
            runbook.source_artifacts["strongreject_or_harmbench"],
            "strongreject_or_harmbench",
        ),
        "language_consistency": load_reviewed_language_suite(runbook.reviewed_language_suite),
    }
    if any(len(questions[name]) != count for name, count in _QUESTION_COUNTS.items()) or len(
        {item.question_id for values in questions.values() for item in values}
    ) != sum(_QUESTION_COUNTS.values()):
        raise DataValidationError("E8 frozen question schedules differ from registration")
    prerequisite_completions = {
        phase: open_phase_prerequisite(path, phase=phase, study=study).verify_complete()
        for phase, path in runbook.prerequisite_runs.items()
    }
    terminal = verify_e7_phase(runbook.e7_finalization)
    if (
        terminal.get("status") != "complete"
        or terminal.get("scientific_eligible") is not True
        or terminal.get("terminal_digest")
        != prerequisite_completions[ExperimentPhase.E7].completion_digest
    ):
        raise FrozenArtifactError("E8 requires a scientifically eligible E7 finalization")
    e7_portable_ledger = PhaseRunLedger.open(
        runbook.e7_finalization / "portable-ledger", study=study
    )
    if tuple(item.question_id for item in questions["triviaqa"]) != tuple(
        e7_portable_ledger.contract.question_ids_by_benchmark["triviaqa"]
    ):
        raise FrozenArtifactError("E8 T-dev schedule differs from its promoted E7 run")
    expected_transition = (
        runbook.prerequisite_runs[ExperimentPhase.E6]
        / "gate-artifacts"
        / "knowledge_recovery_separated_from_abstention_substitution"
        / "likelihood-bundle"
    )
    if sha256_path(runbook.e6_transition_evidence) != sha256_path(expected_transition):
        raise FrozenArtifactError("E8 transition evidence differs from its E6 prerequisite")
    return _E8Context(
        study=study,
        model=model,
        prompts=prompts,
        questions=MappingProxyType(questions),
        execution_private_key=private_hex,
        execution_public_key=public_hex,
        runtime_artifact_sha256=sha256_file(runbook.runtime_artifact),
        runtime_identity=MappingProxyType(dict(runtime_identity)),
        seed=runbook.seed,
    )


def _completion_digests(runbook: E8Runbook, context: _E8Context) -> Mapping[str, str]:
    return MappingProxyType(
        {
            phase.value: open_phase_prerequisite(path, phase=phase, study=context.study)
            .verify_complete()
            .completion_digest
            for phase, path in runbook.prerequisite_runs.items()
        }
    )


def _source_artifacts(runbook: E8Runbook) -> Mapping[str, Path]:
    return MappingProxyType(
        {
            **dict(runbook.source_artifacts),
            "language_consistency": runbook.reviewed_language_suite,
        }
    )


def _condition(
    context: _E8Context,
    *,
    benchmark: str,
    prompt: PromptSpec,
    method: str,
    method_artifact_sha256: str | None = None,
    layer: int | None = None,
    site: ActivationSite | None = None,
    token_scope: TokenScope | None = None,
    alpha: float = 0.0,
    sparsity: float | None = None,
    adaptive_policy: AdaptivePolicySpec | None = None,
    comparison_group: str | None = None,
) -> EvaluationCondition:
    return EvaluationCondition(
        phase=ExperimentPhase.E8,
        benchmark=benchmark,
        partition="T-dev" if benchmark == "triviaqa" else "side-effect-eval",
        model_name=context.model.name,
        model_repository=context.model.repository,
        model_revision=context.model.revision,
        runtime=context.model.runtime,
        quantization=context.model.quantization,
        model_num_layers=context.model.num_layers,
        system_prompt_id=prompt.prompt_id,
        prompt_template_sha256=hashlib.sha256(prompt.text.encode()).hexdigest(),
        steering_method=method,
        method_artifact_sha256=method_artifact_sha256,
        layer=layer,
        site=site,
        token_scope=token_scope,
        alpha=alpha,
        sparsity=sparsity,
        seed=context.seed,
        study_protocol_digest=context.study.digest,
        adaptive_policy=adaptive_policy,
        comparison_group=comparison_group or f"e8-{benchmark}-{prompt.prompt_id}",
    )


def _provisional_contract(runbook: E8Runbook, context: _E8Context) -> PhaseRunContract:
    return PhaseRunContract(
        phase=ExperimentPhase.E8,
        study_protocol_digest=context.study.digest,
        conditions=tuple(
            _condition(
                context,
                benchmark=benchmark,
                prompt=context.prompts[prompt_id],
                method="M0",
            )
            for benchmark in _QUESTION_COUNTS
            for prompt_id in _PROMPTS
        ),
        question_ids_by_benchmark={
            name: tuple(item.question_id for item in values)
            for name, values in context.questions.items()
        },
        input_fingerprints={"provisional": "0" * 64},
        prerequisite_digests=_completion_digests(runbook, context),
        required_gates=context.study.phase(ExperimentPhase.E8).gates,
    )


def prepare_e8_runbook(runbook: E8Runbook) -> Mapping[str, Any]:
    """Freeze the E8 question/scorer bundle used by all construction screens."""

    context = _base_context(runbook)
    destination = runbook.outputs["development_side_effect_bundle"]
    contract = _provisional_contract(runbook, context)
    if destination.exists():
        side_sha = validate_side_effect_evaluation_bundle(destination, contract)
    else:
        side_sha = write_side_effect_evaluation_bundle(
            destination,
            contract,
            context.questions,
            source_artifacts=_source_artifacts(runbook),
            scorer_execution_public_key=context.execution_public_key,
            ifeval_evaluator=runbook.ifeval_evaluator,
        )
    return MappingProxyType(
        {
            "valid": True,
            "phase": "E8",
            "runbook_digest": runbook.runbook_digest,
            "development_side_effect_bundle_sha256": side_sha,
            "question_bundle_sha256": sha256_path(destination / "questions"),
            "execution_public_key": context.execution_public_key,
        }
    )


def preflight_e8_runbook(runbook: E8Runbook) -> Mapping[str, Any]:
    """Replay every immutable E8 source without loading MLX or writing outputs."""

    context = _base_context(runbook)
    if runbook.outputs["development_side_effect_bundle"].exists():
        validate_side_effect_evaluation_bundle(
            runbook.outputs["development_side_effect_bundle"],
            _provisional_contract(runbook, context),
        )
    return MappingProxyType(
        {
            "valid": True,
            "phase": "E8",
            "runbook_digest": runbook.runbook_digest,
            "runtime_artifact_sha256": context.runtime_artifact_sha256,
            "final_question_rows": sum(len(value) for value in context.questions.values()),
            "expected_final_records": 86_040,
            "outputs": {
                name: ("present" if path.exists() else "pending")
                for name, path in runbook.outputs.items()
            },
        }
    )


def _native_runtime(
    runbook: E8Runbook, context: _E8Context
) -> tuple[MlxResearchRuntime, E6RuntimeAttestor]:
    provenance = context.runtime_identity.get("research_provenance")
    if not isinstance(provenance, Mapping):
        raise FrozenArtifactError("E6 runtime attestation lacks research provenance")
    runtime = MlxResearchRuntime.from_spec(
        context.model,
        snapshot_path=runbook.snapshot_directory,
        seed=runbook.seed,
        research_provenance=dict(provenance),
    )
    attestor = E6RuntimeAttestor(runtime, execution_private_key=context.execution_private_key)
    attestor.verify_runtime_artifact(runbook.runtime_artifact)
    return runtime, attestor


def _feature_schema(runbook: E8Runbook, context: _E8Context) -> ActivationFeatureSchema:
    dense, _rms, _sha = _e3_slice(
        runbook.e6_transition_evidence / "e3-static-vectors", runbook.m1_tensor_index
    )
    prompt = context.prompts[runbook.m1_tensor_index[0]]
    return ActivationFeatureSchema(
        benchmark="multi-protected-behavior",
        partition="side-effect-construction",
        split_manifest_digest=sha256_path(
            runbook.e7_finalization
            / "portable-ledger"
            / "inputs"
            / "frozen_side_effect_scorers"
            / "questions"
        ),
        model_repository=context.model.repository,
        model_revision=context.model.revision,
        runtime=context.model.runtime,
        quantization=context.model.quantization,
        prompt_id=prompt.prompt_id,
        prompt_sha256=hashlib.sha256(prompt.text.encode()).hexdigest(),
        activation_kind=ActivationKind.FINAL_PROMPT,
        layers=(runbook.m1_tensor_index[3],),
        sites=(ActivationSite(runbook.m1_tensor_index[2]),),
        composition=FeatureComposition.SINGLE_LAYER,
        width=int(dense.size),
        token_scope=None,
    )


def _e7_label_material(
    runbook: E8Runbook, schema: ActivationFeatureSchema
) -> tuple[Mapping[str, tuple[BehaviorLabelPair, ...]], Mapping[str, Question], Path]:
    e7_study = load_study_protocol(
        runbook.e7_finalization / "configs" / "experiments" / "phases.yaml"
    )
    ledger = PhaseRunLedger.open(runbook.e7_finalization / "portable-ledger", study=e7_study)
    ledger.verify_complete()
    pairs = MappingProxyType(
        {
            behavior: _complete_e7_behavior_label_pairs(
                ledger, behavior=behavior, feature_schema=schema
            )
            for behavior in _BEHAVIORS
        }
    )
    source = (
        runbook.e7_finalization
        / "portable-ledger"
        / "inputs"
        / "frozen_side_effect_scorers"
        / "questions"
    )
    questions = {
        item.question_id: item for path in source.glob("*.jsonl") for item in read_questions(path)
    }
    if any(
        not values
        or {value.label for value in values} != {"positive", "negative"}
        or any(value.question_id not in questions for value in values)
        for values in pairs.values()
    ):
        raise DataValidationError("E7 lacks two-class protected construction evidence")
    return pairs, MappingProxyType(questions), source


def _activation_manifest(
    runbook: E8Runbook,
    context: _E8Context,
    schema: ActivationFeatureSchema,
    pairs: Mapping[str, tuple[BehaviorLabelPair, ...]],
    source: Path,
) -> dict[str, Any]:
    body = {
        "schema_version": 1,
        "runbook_digest": runbook.runbook_digest,
        "feature_schema": schema.to_dict(),
        "runtime_artifact_sha256": context.runtime_artifact_sha256,
        "execution_public_key": context.execution_public_key,
        "source_question_bundle_sha256": sha256_path(source),
        "pairs": {
            behavior: [value.fingerprint for value in values] for behavior, values in pairs.items()
        },
    }
    return {**body, "manifest_digest": stable_hash(body)}


def _activation_row_path(root: Path, behavior: str, pair: BehaviorLabelPair) -> tuple[Path, Path]:
    stem = f"{behavior}-{pair.label}-{stable_hash([pair.question_id, pair.fingerprint])}"
    return root / "rows" / f"{stem}.npy", root / "rows" / f"{stem}.json"


def _activation_receipt_body(
    *,
    behavior: str,
    pair: BehaviorLabelPair,
    question: Question,
    values_sha256: str,
    schema: ActivationFeatureSchema,
    context: _E8Context,
) -> dict[str, Any]:
    return {
        "receipt_kind": "e8-protected-activation-row-v1",
        "behavior": behavior,
        "label": pair.label,
        "pair_fingerprint": pair.fingerprint,
        "question_id": pair.question_id,
        "source_question_sha256": question_source_fingerprint(question),
        "activation_sha256": values_sha256,
        "feature_schema": schema.to_dict(),
        "runtime_artifact_sha256": context.runtime_artifact_sha256,
        "execution_public_key": context.execution_public_key,
    }


def _write_activation_row(
    root: Path,
    *,
    behavior: str,
    pair: BehaviorLabelPair,
    question: Question,
    values: np.ndarray[Any, Any],
    schema: ActivationFeatureSchema,
    context: _E8Context,
    attestor: E6RuntimeAttestor,
) -> None:
    array_path, receipt_path = _activation_row_path(root, behavior, pair)
    array_path.parent.mkdir(parents=True, exist_ok=True)
    values = np.ascontiguousarray(values, dtype=np.float32)
    body = _activation_receipt_body(
        behavior=behavior,
        pair=pair,
        question=question,
        values_sha256=hashlib.sha256(values.tobytes(order="C")).hexdigest(),
        schema=schema,
        context=context,
    )
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{array_path.name}.", dir=array_path.parent, delete=False
    ) as handle:
        temporary_array = Path(handle.name)
        np.save(handle, values, allow_pickle=False)
    try:
        os.replace(temporary_array, array_path)
    finally:
        temporary_array.unlink(missing_ok=True)
    payload = {"body": body, "signature": attestor._sign(body)}
    temporary_receipt = receipt_path.with_name(f".{receipt_path.name}.{os.getpid()}.tmp")
    temporary_receipt.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_receipt, receipt_path)


def _load_activation_row(
    root: Path,
    *,
    behavior: str,
    pair: BehaviorLabelPair,
    question: Question,
    schema: ActivationFeatureSchema,
    context: _E8Context,
) -> torch.Tensor:
    array_path, receipt_path = _activation_row_path(root, behavior, pair)
    if not array_path.is_file() or not receipt_path.is_file():
        raise FrozenArtifactError("E8 activation row is partially persisted")
    try:
        values = np.load(array_path, allow_pickle=False)
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        body = payload["body"]
        signature = payload["signature"]
        expected = _activation_receipt_body(
            behavior=behavior,
            pair=pair,
            question=question,
            values_sha256=hashlib.sha256(
                np.ascontiguousarray(values, dtype=np.float32).tobytes(order="C")
            ).hexdigest(),
            schema=schema,
            context=context,
        )
        if body != expected or type(signature) is not str:
            raise FrozenArtifactError("E8 activation row receipt differs")
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(context.execution_public_key)).verify(
            bytes.fromhex(signature), canonical_json(body).encode()
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError, InvalidSignature) as exc:
        if isinstance(exc, FrozenArtifactError):
            raise
        raise FrozenArtifactError(f"cannot replay E8 activation row: {exc}") from exc
    if (
        values.dtype != np.float32
        or values.shape != (schema.width,)
        or not np.isfinite(values).all()
    ):
        raise FrozenArtifactError("E8 activation row values differ")
    return torch.from_numpy(values.copy())


def _materialize_activation_bundle(
    runbook: E8Runbook,
    context: _E8Context,
    schema: ActivationFeatureSchema,
    pairs: Mapping[str, tuple[BehaviorLabelPair, ...]],
    questions: Mapping[str, Question],
    source: Path,
    root: Path,
) -> str:
    evidence: list[BehaviorActivationEvidence] = []
    for behavior in _BEHAVIORS:
        values = pairs[behavior]
        positive = tuple(item for item in values if item.label == "positive")
        negative = tuple(item for item in values if item.label == "negative")
        evidence.append(
            BehaviorActivationEvidence(
                behavior=behavior,
                positive_question_ids=tuple(item.question_id for item in positive),
                negative_question_ids=tuple(item.question_id for item in negative),
                positive_activations=torch.stack(
                    tuple(
                        _load_activation_row(
                            root,
                            behavior=behavior,
                            pair=item,
                            question=questions[item.question_id],
                            schema=schema,
                            context=context,
                        )
                        for item in positive
                    )
                ),
                negative_activations=torch.stack(
                    tuple(
                        _load_activation_row(
                            root,
                            behavior=behavior,
                            pair=item,
                            question=questions[item.question_id],
                            schema=schema,
                            context=context,
                        )
                        for item in negative
                    )
                ),
                label_pairs=values,
            )
        )
    destination = runbook.outputs["protected_behavior_activations"]
    if destination.exists():
        load_e8_behavior_activation_bundle(destination)
        return sha256_path(destination)
    private = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(context.execution_private_key))
    return _write_e8_behavior_activation_bundle(
        destination,
        evidence,
        feature_schema=schema,
        runtime_artifact_sha256=context.runtime_artifact_sha256,
        execution_public_key=context.execution_public_key,
        source_question_bundle_sha256=sha256_path(source),
        execution_signer=lambda body: private.sign(canonical_json(body).encode()).hex(),
    )


def execute_e8_activation_capture(
    runbook: E8Runbook, *, limit: int | None = None
) -> Mapping[str, Any]:
    """Resume signed protected-behavior activation capture from the E7 ledger."""

    prepare_e8_runbook(runbook)
    context = _base_context(runbook)
    schema = _feature_schema(runbook, context)
    pairs, questions, source = _e7_label_material(runbook, schema)
    root = runbook.outputs["activation_work"]
    manifest = _activation_manifest(runbook, context, schema, pairs, source)
    if root.exists():
        try:
            observed = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FrozenArtifactError(f"cannot read E8 activation manifest: {exc}") from exc
        if observed != manifest:
            raise FrozenArtifactError("E8 activation resume manifest differs")
    else:
        root.mkdir(parents=True)
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    expected = sum(len(values) for values in pairs.values())
    budget = expected if limit is None else limit
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise DataValidationError("E8 activation limit must be positive")
    runtime: MlxResearchRuntime | None = None
    attestor: E6RuntimeAttestor | None = None
    processed = 0
    completed = 0
    try:
        for behavior in _BEHAVIORS:
            for pair in pairs[behavior]:
                array_path, receipt_path = _activation_row_path(root, behavior, pair)
                question = questions[pair.question_id]
                if array_path.exists() or receipt_path.exists():
                    _load_activation_row(
                        root,
                        behavior=behavior,
                        pair=pair,
                        question=question,
                        schema=schema,
                        context=context,
                    )
                    completed += 1
                    continue
                if processed >= budget:
                    break
                if runtime is None:
                    runtime, attestor = _native_runtime(runbook, context)
                assert attestor is not None
                rendered = runtime.render_prompt(
                    context.prompts[schema.prompt_id],
                    question.text,
                    metadata=question.metadata,
                )
                cube = runtime.prompt_feature_cube(
                    rendered, layers=schema.layers, sites=schema.sites
                )
                values = np.asarray(
                    cube.activations[schema.sites[0]][schema.layers[0]][0],
                    dtype=np.float32,
                )
                _write_activation_row(
                    root,
                    behavior=behavior,
                    pair=pair,
                    question=question,
                    values=values,
                    schema=schema,
                    context=context,
                    attestor=attestor,
                )
                processed += 1
                completed += 1
            if processed >= budget:
                break
    finally:
        if runtime is not None:
            runtime.close()
    complete = completed == expected
    bundle_sha: str | None = None
    if complete:
        bundle_sha = _materialize_activation_bundle(
            runbook, context, schema, pairs, questions, source, root
        )
    return MappingProxyType(
        {
            "valid": True,
            "phase": "E8",
            "processed_rows": processed,
            "completed_rows": completed,
            "expected_rows": expected,
            "complete": complete,
            "protected_behavior_activations_sha256": bundle_sha,
        }
    )


@dataclass(frozen=True, slots=True)
class _E8Components:
    dense_direction: torch.Tensor
    reference_rms: float
    layer: int
    site: ActivationSite
    token_scope: TokenScope
    m3_policies: Mapping[str, AdaptivePolicySpec]
    m3_method_artifacts: Mapping[str, str]
    controller_path: Path
    coordinate_path: Path
    sae_path: Path


def _e6_components(
    runbook: E8Runbook, context: _E8Context, schema: ActivationFeatureSchema
) -> _E8Components:
    e6 = open_phase_prerequisite(
        runbook.prerequisite_runs[ExperimentPhase.E6],
        phase=ExperimentPhase.E6,
        study=context.study,
    )
    e6.verify_complete()
    geometries = {
        (item.layer, item.site, item.token_scope)
        for item in e6.contract.conditions
        if item.steering_method == "M1" and item.system_prompt_id == schema.prompt_id
    }
    policies: dict[str, dict[str, tuple[AdaptivePolicySpec, str]]] = {}
    for condition in e6.contract.conditions:
        if condition.steering_method != "M3" or condition.system_prompt_id not in _PROMPTS:
            continue
        if condition.adaptive_policy is None or condition.method_artifact_sha256 is None:
            raise FrozenArtifactError("E6 M3 condition lacks its frozen controller policy")
        policies.setdefault(condition.system_prompt_id, {})[
            stable_hash(condition.adaptive_policy.to_dict())
        ] = (condition.adaptive_policy, condition.method_artifact_sha256)
    if (
        len(geometries) != 1
        or set(policies) != set(_PROMPTS)
        or any(len(values) != 1 for values in policies.values())
    ):
        raise FrozenArtifactError("E6 lacks one promoted M1/M3 source per E8 prompt")
    layer, site, scope = next(iter(geometries))
    if (
        type(layer) is not int
        or not isinstance(site, ActivationSite)
        or not isinstance(scope, TokenScope)
        or layer != schema.layers[0]
        or site is not schema.sites[0]
    ):
        raise FrozenArtifactError("E8 protected geometry differs from E6 M1")
    try:
        creation = json.loads((e6.directory / "creation-evidence.json").read_text(encoding="utf-8"))
        descriptor = creation["input_artifacts"]["E5_adaptive_controllers"]
        raw = Path(str(descriptor["location"]))
        controller_source = raw if raw.is_absolute() else (e6.directory / raw).resolve()
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise FrozenArtifactError(f"cannot locate the E6 controller source: {exc}") from exc
    from mfh.experiments.e5_adaptive import (
        load_e5_controller_binding,
        validate_e5_selected_controller_bundle,
    )

    if controller_source.is_file():
        controller_path = Path(load_e5_controller_binding(controller_source).controller_directory)
    else:
        bundle = (
            controller_source / "selected-controller"
            if (controller_source / "selected-controller").is_dir()
            else controller_source
        )
        selected = validate_e5_selected_controller_bundle(bundle)
        controller_value = selected["controller_path"]
        if not isinstance(controller_value, Path):
            raise FrozenArtifactError("E6 selected controller path is invalid")
        controller_path = controller_value
    controller_sha = sha256_path(controller_path)
    resolved = {prompt: next(iter(values.values())) for prompt, values in policies.items()}
    if any(
        policy.controller_artifact_sha256 != controller_sha
        or policy.execution_public_key != context.execution_public_key
        for policy, _artifact in resolved.values()
    ):
        raise FrozenArtifactError("E6 policies differ from the resolved E5 controller")
    dense, reference_rms, _sha = _e3_slice(
        runbook.e6_transition_evidence / "e3-static-vectors", runbook.m1_tensor_index
    )
    coordinate = runbook.e7_finalization / "coordinate-artifact"
    sae = runbook.e7_finalization / "sae-intervention"
    load_coordinate_sparse_artifact(coordinate)
    load_sae_intervention(sae)
    return _E8Components(
        dense_direction=_unit(torch.from_numpy(dense.copy()), width=schema.width),
        reference_rms=reference_rms,
        layer=layer,
        site=site,
        token_scope=scope,
        m3_policies=MappingProxyType({prompt: value[0] for prompt, value in resolved.items()}),
        m3_method_artifacts=MappingProxyType(
            {prompt: value[1] for prompt, value in resolved.items()}
        ),
        controller_path=controller_path,
        coordinate_path=coordinate,
        sae_path=sae,
    )


def _draft_record(
    context: _E8Context,
    *,
    question: Question,
    prompt: PromptSpec,
    condition: EvaluationCondition,
    rendered_prompt_hash: str,
) -> GenerationRecord:
    return GenerationRecord(
        question_id=question.question_id,
        benchmark=question.benchmark,
        model_repository=context.model.repository,
        model_revision=context.model.revision,
        runtime=context.model.runtime,
        quantization=context.model.quantization,
        system_prompt_id=prompt.prompt_id,
        rendered_prompt_hash=rendered_prompt_hash,
        steering_method=condition.steering_method,
        layer=condition.layer,
        site=condition.site,
        token_scope=condition.token_scope,
        alpha=condition.alpha,
        sparsity=condition.sparsity,
        controller_scores={},
        raw_output="",
        normalized_answer="",
        outcome=Outcome.INCORRECT,
        generation_latency_seconds=0.0,
        input_tokens=0,
        output_tokens=0,
        condition_id=condition.condition_id,
        seed=context.seed,
        metadata={
            "phase": "E8",
            "partition": condition.partition,
            "prompt_template_sha256": condition.prompt_template_sha256,
            "study_protocol_digest": context.study.digest,
            "comparison_group": condition.comparison_group,
            **(
                {"method_artifact_sha256": condition.method_artifact_sha256}
                if condition.method_artifact_sha256 is not None
                else {}
            ),
        },
    )


def _write_generation_record(path: Path, record: GenerationRecord) -> None:
    if path.exists():
        raise FrozenArtifactError(f"refusing to overwrite E8 execution row: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    body = record.to_dict()
    payload = {"record": body, "record_digest": stable_hash(body)}
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    stage = Path(temporary)
    try:
        stage.write_text(
            json.dumps(payload, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        os.replace(stage, path)
    finally:
        stage.unlink(missing_ok=True)


def _load_generation_record(path: Path) -> GenerationRecord:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        body = payload["record"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read E8 execution row: {exc}") from exc
    if payload.get("record_digest") != stable_hash(body):
        raise FrozenArtifactError("E8 execution row digest differs")
    return GenerationRecord.from_dict(body)


def _bind_grader(
    grader: E7E8DevelopmentGrader, question: Question
) -> Callable[[GenerationRecord], GenerationRecord]:
    return lambda record: grader(record, question)


def _fixed_row(
    *,
    runbook: E8Runbook,
    context: _E8Context,
    runtime: MlxResearchRuntime,
    attestor: E6RuntimeAttestor,
    grader: E7E8DevelopmentGrader,
    question: Question,
    prompt: PromptSpec,
    condition: EvaluationCondition,
    direction: torch.Tensor | np.ndarray[Any, Any] | None,
    reference_rms: float | None,
) -> GenerationRecord:
    rendered = runtime.render_prompt(prompt, question.text, metadata=question.metadata)
    return execute_e8_generation(
        attestor=attestor,
        runtime_artifact=runbook.runtime_artifact,
        question=question,
        prompt=prompt,
        generation_record=_draft_record(
            context,
            question=question,
            prompt=prompt,
            condition=condition,
            rendered_prompt_hash=rendered.sha256,
        ),
        condition=condition,
        direction=direction,
        reference_rms=reference_rms,
        max_new_tokens=runbook.max_new_tokens,
        populate_generation=True,
        generation_grader=_bind_grader(grader, question),
    )


def _variant_directions(bundle: Any, dense_direction: torch.Tensor) -> Mapping[str, torch.Tensor]:
    ordered = tuple(bundle.evidence)
    fingerprint = stable_hash({value.behavior: value.data_fingerprint for value in ordered})
    subspace = build_protected_subspace(
        tuple(value.direction for value in ordered),
        data_fingerprint=fingerprint,
        feature_schema=bundle.feature_schema,
    )
    covariance = behavior_covariance(_within_class_behavior_changes(ordered), center=False)
    return MappingProxyType(
        {
            "orthogonal_projection": subspace.project(dense_direction, normalize=True),
            "covariance_aware": covariance_aware_direction(
                dense_direction, covariance, lambda_penalty=1.0, ridge=1e-4
            ),
        }
    )


def _variant_questions(
    runbook: E8Runbook, context: _E8Context
) -> tuple[tuple[Question, ...], Mapping[str, tuple[Question, ...]]]:
    factual = context.questions["triviaqa"][: runbook.variant_factual_rows]
    protected = MappingProxyType(
        {
            behavior: context.questions[benchmark][: runbook.variant_protected_rows]
            for behavior, benchmark in _BEHAVIOR_BENCHMARKS.items()
        }
    )
    if len(factual) != runbook.variant_factual_rows or any(
        len(values) != runbook.variant_protected_rows for values in protected.values()
    ):
        raise DataValidationError("E8 variant screen cohorts are underfilled")
    return factual, protected


def _variant_row_path(
    runbook: E8Runbook,
    *,
    variant: str,
    mode: str,
    behavior: str,
    question: Question,
) -> Path:
    key = stable_hash([variant, mode, behavior, question.benchmark, question.question_id])
    return runbook.outputs["variant_work"] / "rows" / f"{key}.json"


def _screen_boolean(
    behavior: str,
    baseline: GenerationRecord,
    intervention: GenerationRecord,
) -> tuple[bool, bool]:
    if behavior == "correct_to_abstain":
        return (
            baseline.outcome is not Outcome.ABSTENTION,
            intervention.outcome is not Outcome.ABSTENTION,
        )
    if behavior == "verbosity_style":
        return True, response_verbosity_style_preserved(
            baseline.raw_output, intervention.raw_output
        )
    if behavior == "instruction_following_failure":
        before = baseline.metadata.get("official_pass")
        after = intervention.metadata.get("official_pass")
    elif behavior == "xstest_safe_refusal":
        baseline_refused = baseline.metadata.get("refused")
        intervention_refused = intervention.metadata.get("refused")
        before = not baseline_refused if isinstance(baseline_refused, bool) else None
        after = not intervention_refused if isinstance(intervention_refused, bool) else None
    elif behavior == "harmful_refusal":
        before = baseline.metadata.get("refused")
        after = intervention.metadata.get("refused")
    elif behavior == "language_switching":
        before = baseline.metadata.get("requested_language_correct")
        after = intervention.metadata.get("requested_language_correct")
    else:  # pragma: no cover - frozen caller
        raise DataValidationError("unknown E8 protected screen behavior")
    if not isinstance(before, bool) or not isinstance(after, bool):
        raise DataValidationError(f"E8 protected screen score is absent for {behavior}")
    return before, after


def _save_variant_screens(path: Path, screens: Sequence[M5VariantScreen]) -> str:
    if path.exists() or path.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite E8 variant screens: {path}")
    body = {"schema_version": 1, "screens": [item.to_dict() for item in screens]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({**body, "screens_digest": stable_hash(body)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sha256_file(path)


def _load_variant_screens(path: Path) -> tuple[M5VariantScreen, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        body = dict(payload)
        digest = body.pop("screens_digest")
        raw = body["screens"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise FrozenArtifactError(f"cannot read E8 variant screens: {exc}") from exc
    if digest != stable_hash(body) or body.get("schema_version") != 1 or not isinstance(raw, list):
        raise FrozenArtifactError("E8 variant screen digest differs")
    return tuple(M5VariantScreen.from_dict(value) for value in raw)


def execute_e8_variant_screen(runbook: E8Runbook, *, limit: int | None = None) -> Mapping[str, Any]:
    """Resume the exactly paired orthogonal/covariance M5 construction screen."""

    context = _base_context(runbook)
    bundle = load_e8_behavior_activation_bundle(runbook.outputs["protected_behavior_activations"])
    components = _e6_components(runbook, context, bundle.feature_schema)
    directions = _variant_directions(bundle, components.dense_direction)
    factual, protected = _variant_questions(runbook, context)
    prompt = context.prompts[bundle.feature_schema.prompt_id]
    work = runbook.outputs["variant_work"]
    manifest_body = {
        "schema_version": 1,
        "runbook_digest": runbook.runbook_digest,
        "activation_bundle_sha256": sha256_path(runbook.outputs["protected_behavior_activations"]),
        "factual_question_ids": [item.question_id for item in factual],
        "protected_question_ids": {
            name: [item.question_id for item in values] for name, values in protected.items()
        },
        "direction_sha256s": {
            name: hashlib.sha256(
                np.ascontiguousarray(value.numpy(), dtype=np.float32).tobytes(order="C")
            ).hexdigest()
            for name, value in directions.items()
        },
        "alpha": runbook.m5_alpha,
        "geometry": [
            components.layer,
            components.site.value,
            components.token_scope.value,
        ],
    }
    manifest = {**manifest_body, "manifest_digest": stable_hash(manifest_body)}
    if work.exists():
        try:
            observed = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FrozenArtifactError(f"cannot read E8 variant manifest: {exc}") from exc
        if observed != manifest:
            raise FrozenArtifactError("E8 variant resume manifest differs")
    else:
        work.mkdir(parents=True)
        (work / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    unique_baseline = {
        (item.benchmark, item.question_id): item
        for item in (
            *factual,
            *(item for values in protected.values() for item in values),
        )
    }
    expected = len(unique_baseline) + 2 * (
        len(factual) + sum(len(values) for values in protected.values())
    )
    budget = expected if limit is None else limit
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise DataValidationError("E8 variant screen limit must be positive")
    runtime: MlxResearchRuntime | None = None
    attestor: E6RuntimeAttestor | None = None
    grader: E7E8DevelopmentGrader | None = None
    processed = 0
    completed = 0

    def execute(
        *,
        variant: str,
        mode: str,
        behavior: str,
        question: Question,
        direction: torch.Tensor | None,
    ) -> None:
        nonlocal runtime, attestor, grader, processed, completed
        path = _variant_row_path(
            runbook,
            variant=variant,
            mode=mode,
            behavior=behavior,
            question=question,
        )
        if path.exists():
            _load_generation_record(path)
            completed += 1
            return
        if processed >= budget:
            return
        if runtime is None:
            runtime, attestor = _native_runtime(runbook, context)
            grader = E7E8DevelopmentGrader(
                grader_bundle=runbook.outputs["development_side_effect_bundle"],
                attestor=attestor,
                environment_file=runbook.environment_file,
            )
        assert attestor is not None and grader is not None
        if mode == "baseline":
            condition = _condition(
                context,
                benchmark=question.benchmark,
                prompt=prompt,
                method="M0",
                comparison_group=f"e8-variant-{question.benchmark}",
            )
            reference = None
        else:
            assert direction is not None
            direction_sha = hashlib.sha256(
                np.ascontiguousarray(direction.numpy(), dtype=np.float32).tobytes(order="C")
            ).hexdigest()
            condition = _condition(
                context,
                benchmark=question.benchmark,
                prompt=prompt,
                method="M5",
                method_artifact_sha256=direction_sha,
                layer=components.layer,
                site=components.site,
                token_scope=components.token_scope,
                alpha=runbook.m5_alpha,
                comparison_group=f"e8-variant-{question.benchmark}",
            )
            reference = components.reference_rms
        record = _fixed_row(
            runbook=runbook,
            context=context,
            runtime=runtime,
            attestor=attestor,
            grader=grader,
            question=question,
            prompt=prompt,
            condition=condition,
            direction=direction,
            reference_rms=reference,
        )
        _write_generation_record(path, record)
        processed += 1
        completed += 1

    try:
        for question in unique_baseline.values():
            execute(
                variant="shared",
                mode="baseline",
                behavior="shared",
                question=question,
                direction=None,
            )
            if processed >= budget:
                break
        if processed < budget:
            for variant, direction in directions.items():
                for question in factual:
                    execute(
                        variant=variant,
                        mode="intervention",
                        behavior="factual",
                        question=question,
                        direction=direction,
                    )
                    if processed >= budget:
                        break
                if processed >= budget:
                    break
                for behavior, values in protected.items():
                    for question in values:
                        execute(
                            variant=variant,
                            mode="intervention",
                            behavior=behavior,
                            question=question,
                            direction=direction,
                        )
                        if processed >= budget:
                            break
                    if processed >= budget:
                        break
                if processed >= budget:
                    break
    finally:
        if runtime is not None:
            runtime.close()
    complete = completed == expected
    if complete and not runbook.outputs["variant_screens"].exists():
        baseline_index = {
            key: _load_generation_record(
                _variant_row_path(
                    runbook,
                    variant="shared",
                    mode="baseline",
                    behavior="shared",
                    question=question,
                )
            )
            for key, question in unique_baseline.items()
        }
        screens: list[M5VariantScreen] = []
        for variant, direction in directions.items():
            factual_baseline = tuple(
                baseline_index[(item.benchmark, item.question_id)] for item in factual
            )
            factual_intervention = tuple(
                _load_generation_record(
                    _variant_row_path(
                        runbook,
                        variant=variant,
                        mode="intervention",
                        behavior="factual",
                        question=item,
                    )
                )
                for item in factual
            )
            protected_baseline_records = {
                behavior: tuple(
                    baseline_index[(item.benchmark, item.question_id)] for item in values
                )
                for behavior, values in protected.items()
            }
            protected_intervention_records = {
                behavior: tuple(
                    _load_generation_record(
                        _variant_row_path(
                            runbook,
                            variant=variant,
                            mode="intervention",
                            behavior=behavior,
                            question=item,
                        )
                    )
                    for item in values
                )
                for behavior, values in protected.items()
            }
            booleans = {
                behavior: tuple(
                    _screen_boolean(behavior, before, after)
                    for before, after in zip(
                        protected_baseline_records[behavior],
                        protected_intervention_records[behavior],
                        strict=True,
                    )
                )
                for behavior in _BEHAVIORS
            }
            direction_sha = hashlib.sha256(
                np.ascontiguousarray(direction.numpy(), dtype=np.float32).tobytes(order="C")
            ).hexdigest()
            screens.append(
                M5VariantScreen(
                    variant=variant,
                    question_ids=tuple(item.question_id for item in factual),
                    baseline_outcomes=tuple(item.outcome for item in factual_baseline),
                    intervention_outcomes=tuple(item.outcome for item in factual_intervention),
                    protected_baseline={
                        behavior: tuple(value[0] for value in values)
                        for behavior, values in booleans.items()
                    },
                    protected_intervention={
                        behavior: tuple(value[1] for value in values)
                        for behavior, values in booleans.items()
                    },
                    baseline_execution_records=factual_baseline,
                    intervention_execution_records=factual_intervention,
                    runtime_artifact_sha256=context.runtime_artifact_sha256,
                    execution_public_key=context.execution_public_key,
                    protected_question_ids={
                        behavior: tuple(item.question_id for item in values)
                        for behavior, values in protected.items()
                    },
                    protected_baseline_execution_records=protected_baseline_records,
                    protected_intervention_execution_records=(protected_intervention_records),
                    direction_sha256=direction_sha,
                    layer=components.layer,
                    site=components.site,
                    token_scope=components.token_scope,
                    alpha=runbook.m5_alpha,
                    reference_rms=components.reference_rms,
                )
            )
        _save_variant_screens(runbook.outputs["variant_screens"], screens)
    frozen_screens = (
        _load_variant_screens(runbook.outputs["variant_screens"])
        if runbook.outputs["variant_screens"].exists()
        else ()
    )
    return MappingProxyType(
        {
            "valid": True,
            "phase": "E8",
            "processed_rows": processed,
            "completed_rows": completed,
            "expected_rows": expected,
            "complete": complete,
            "variant_screens_sha256": (
                sha256_file(runbook.outputs["variant_screens"]) if frozen_screens else None
            ),
            "selected_variant": (
                build_e8_protected_artifact(
                    evidence=bundle.evidence,
                    feature_schema=bundle.feature_schema,
                    dense_direction=components.dense_direction,
                    source_fingerprints={
                        "E6_transition_evidence": sha256_path(runbook.e6_transition_evidence),
                        "E7_sparse_artifacts": sha256_path(runbook.e7_finalization),
                        "protected_behavior_activations": sha256_path(
                            runbook.outputs["protected_behavior_activations"]
                        ),
                    },
                    variant_screens=frozen_screens,
                    layer=components.layer,
                    site=components.site,
                    token_scope=components.token_scope,
                    alpha=runbook.m5_alpha,
                    reference_rms=components.reference_rms,
                ).selected_variant
                if frozen_screens
                else None
            ),
        }
    )


def promote_e8_protected_artifact(runbook: E8Runbook) -> Mapping[str, Any]:
    """Freeze the M5 protected direction selected by the paired native screen."""

    context = _base_context(runbook)
    bundle = load_e8_behavior_activation_bundle(runbook.outputs["protected_behavior_activations"])
    components = _e6_components(runbook, context, bundle.feature_schema)
    screens = _load_variant_screens(runbook.outputs["variant_screens"])
    artifact = build_e8_protected_artifact(
        evidence=bundle.evidence,
        feature_schema=bundle.feature_schema,
        dense_direction=components.dense_direction,
        source_fingerprints={
            "E6_transition_evidence": sha256_path(runbook.e6_transition_evidence),
            "E7_sparse_artifacts": sha256_path(runbook.e7_finalization),
            "protected_behavior_activations": sha256_path(
                runbook.outputs["protected_behavior_activations"]
            ),
        },
        variant_screens=screens,
        layer=components.layer,
        site=components.site,
        token_scope=components.token_scope,
        alpha=runbook.m5_alpha,
        reference_rms=components.reference_rms,
    )
    destination = runbook.outputs["protected_artifact"]
    if destination.exists():
        observed = load_e8_protected_artifact(destination)
        if observed.selected_variant != artifact.selected_variant:
            raise FrozenArtifactError("existing E8 protected artifact differs")
    else:
        save_e8_protected_artifact(destination, artifact)
    return MappingProxyType(
        {
            "valid": True,
            "phase": "E8",
            "protected_artifact_sha256": sha256_path(destination),
            "selected_variant": artifact.selected_variant,
            "maximum_protected_change": max(item.maximum_protected_change for item in screens),
        }
    )


def _candidate_question_set(runbook: E8Runbook, context: _E8Context) -> tuple[Question, ...]:
    values = context.questions["triviaqa"][: runbook.candidate_question_count]
    if len(values) != runbook.candidate_question_count:
        raise DataValidationError("E8 candidate TriviaQA cohort is underfilled")
    return values


def _candidate_row_path(runbook: E8Runbook, condition_id: str, question_id: str) -> Path:
    return (
        runbook.outputs["candidate_work"]
        / "rows"
        / f"{stable_hash([condition_id, question_id])}.json"
    )


def _candidate_conditions(
    runbook: E8Runbook,
    context: _E8Context,
    components: _E8Components,
    artifact: E8ProtectedArtifact,
) -> tuple[tuple[EvaluationCondition, Any, float | None], ...]:
    sae = load_sae_intervention(components.sae_path)
    sae_geometry = next(iter(sae.evidence)).spec
    sae_sparsity = len(sae.latent_direction.selected_features) / float(
        sae.training.config.resolved_latent_width
    )
    protected_sha = sha256_path(runbook.outputs["protected_artifact"])
    sae_sha = sha256_path(components.sae_path)
    values: list[tuple[EvaluationCondition, Any, float | None]] = []
    for prompt_id in _PROMPTS:
        prompt = context.prompts[prompt_id]
        for alpha in _ALPHAS:
            values.append(
                (
                    _condition(
                        context,
                        benchmark="triviaqa",
                        prompt=prompt,
                        method="M1",
                        layer=components.layer,
                        site=components.site,
                        token_scope=components.token_scope,
                        alpha=alpha,
                    ),
                    components.dense_direction,
                    components.reference_rms,
                )
            )
            policy = replace(components.m3_policies[prompt_id], alpha_max=alpha)
            values.append(
                (
                    _condition(
                        context,
                        benchmark="triviaqa",
                        prompt=prompt,
                        method="M3",
                        method_artifact_sha256=components.m3_method_artifacts[prompt_id],
                        adaptive_policy=policy,
                    ),
                    policy,
                    None,
                )
            )
            values.append(
                (
                    _condition(
                        context,
                        benchmark="triviaqa",
                        prompt=prompt,
                        method="M4",
                        method_artifact_sha256=sae_sha,
                        layer=sae_geometry.layer,
                        site=sae_geometry.site,
                        token_scope=sae_geometry.token_scope,
                        alpha=alpha,
                        sparsity=sae_sparsity,
                    ),
                    sae.decoded_direction,
                    artifact.reference_rms,
                )
            )
            values.append(
                (
                    _condition(
                        context,
                        benchmark="triviaqa",
                        prompt=prompt,
                        method="M5",
                        method_artifact_sha256=protected_sha,
                        layer=artifact.layer,
                        site=artifact.site,
                        token_scope=artifact.token_scope,
                        alpha=alpha,
                    ),
                    artifact.selected_direction,
                    artifact.reference_rms,
                )
            )
    if len(values) != len(_PROMPTS) * len(_ALPHAS) * len(_SCREEN_METHODS):
        raise DataValidationError("E8 candidate condition grid is incomplete")
    return tuple(values)


def _candidate_target(
    runbook: E8Runbook, points: Sequence[E8CandidatePoint]
) -> tuple[float, Mapping[tuple[str, str], E8CandidatePoint]]:
    grouped: dict[tuple[str, str], tuple[E8CandidatePoint, ...]] = {}
    for prompt in _PROMPTS:
        for method in _SCREEN_METHODS:
            grouped[(prompt, method)] = tuple(
                item for item in points if item.prompt_id == prompt and item.method == method
            )
    if any(len(values) != len(_ALPHAS) for values in grouped.values()):
        raise DataValidationError("E8 candidate point groups are incomplete")
    dimension = runbook.matching_dimension
    targets = tuple(index / 10_000 for index in range(10_001))
    eligible: list[
        tuple[tuple[float, float, float], float, Mapping[tuple[str, str], E8CandidatePoint]]
    ] = []
    m5_registered = {
        prompt: next(
            item
            for item in grouped[(prompt, "M5")]
            if math.isclose(item.alpha, runbook.m5_alpha, rel_tol=0, abs_tol=1e-12)
        )
        for prompt in _PROMPTS
    }
    desired = sum(getattr(item, dimension) for item in m5_registered.values()) / len(m5_registered)
    for target in targets:
        winners = {
            key: min(
                values,
                key=lambda item: (
                    abs(getattr(item, dimension) - target),
                    item.alpha,
                    item.candidate_condition_id,
                ),
            )
            for key, values in grouped.items()
        }
        mismatches = tuple(abs(getattr(item, dimension) - target) for item in winners.values())
        if max(mismatches) <= runbook.matching_tolerance and all(
            math.isclose(
                winners[(prompt, "M5")].alpha,
                runbook.m5_alpha,
                rel_tol=0,
                abs_tol=1e-12,
            )
            for prompt in _PROMPTS
        ):
            eligible.append(
                (
                    (max(mismatches), abs(target - desired), target),
                    target,
                    MappingProxyType(winners),
                )
            )
    if not eligible:
        raise DataValidationError(
            "E8 empirical screen falsified the registered matched-point tolerance"
        )
    _score, target, selected_winners = min(eligible, key=lambda item: item[0])
    return target, selected_winners


def execute_e8_candidate_screen(
    runbook: E8Runbook, *, limit: int | None = None
) -> Mapping[str, Any]:
    """Resume the registered empirical strength grid and freeze matched winners."""

    context = _base_context(runbook)
    artifact = load_e8_protected_artifact(runbook.outputs["protected_artifact"])
    components = _e6_components(runbook, context, artifact.feature_schema)
    questions = _candidate_question_set(runbook, context)
    conditions = _candidate_conditions(runbook, context, components, artifact)
    work = runbook.outputs["candidate_work"]
    manifest_body = {
        "schema_version": 1,
        "runbook_digest": runbook.runbook_digest,
        "protected_artifact_sha256": sha256_path(runbook.outputs["protected_artifact"]),
        "condition_ids": [item[0].condition_id for item in conditions],
        "question_ids": [item.question_id for item in questions],
        "matching_dimension": runbook.matching_dimension,
        "matching_tolerance": runbook.matching_tolerance,
        "m5_alpha": runbook.m5_alpha,
    }
    manifest = {**manifest_body, "manifest_digest": stable_hash(manifest_body)}
    if work.exists():
        try:
            observed = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FrozenArtifactError(f"cannot read E8 candidate manifest: {exc}") from exc
        if observed != manifest:
            raise FrozenArtifactError("E8 candidate resume manifest differs")
    else:
        work.mkdir(parents=True)
        (work / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    expected = len(conditions) * len(questions)
    budget = expected if limit is None else limit
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise DataValidationError("E8 candidate screen limit must be positive")
    runtime: MlxResearchRuntime | None = None
    attestor: E6RuntimeAttestor | None = None
    grader: E7E8DevelopmentGrader | None = None
    controller_schema_prompt = context.prompts[
        load_adaptive_controller(components.controller_path).risk_probe.training_schema.prompt_id
    ]
    processed = 0
    completed = 0
    try:
        for condition, material, reference_rms in conditions:
            prompt = context.prompts[condition.system_prompt_id]
            for question in questions:
                path = _candidate_row_path(runbook, condition.condition_id, question.question_id)
                if path.exists():
                    _load_generation_record(path)
                    completed += 1
                    continue
                if processed >= budget:
                    break
                if runtime is None:
                    runtime, attestor = _native_runtime(runbook, context)
                    grader = E7E8DevelopmentGrader(
                        grader_bundle=runbook.outputs["development_side_effect_bundle"],
                        attestor=attestor,
                        environment_file=runbook.environment_file,
                    )
                assert attestor is not None and grader is not None
                if condition.steering_method == "M3":
                    if not isinstance(material, AdaptivePolicySpec):
                        raise DataValidationError("E8 M3 candidate material is invalid")
                    rendered = runtime.render_prompt(
                        prompt, question.text, metadata=question.metadata
                    )
                    record = execute_e8_adaptive_generation(
                        attestor=attestor,
                        runtime_artifact=runbook.runtime_artifact,
                        controller_artifact=components.controller_path,
                        question=question,
                        prompt=prompt,
                        controller_prompt=controller_schema_prompt,
                        generation_record=_draft_record(
                            context,
                            question=question,
                            prompt=prompt,
                            condition=condition,
                            rendered_prompt_hash=rendered.sha256,
                        ),
                        condition=condition,
                        max_new_tokens=runbook.max_new_tokens,
                        populate_generation=True,
                        generation_grader=_bind_grader(grader, question),
                    )
                else:
                    if not isinstance(material, torch.Tensor):
                        raise DataValidationError("E8 fixed candidate material is invalid")
                    record = _fixed_row(
                        runbook=runbook,
                        context=context,
                        runtime=runtime,
                        attestor=attestor,
                        grader=grader,
                        question=question,
                        prompt=prompt,
                        condition=condition,
                        direction=material,
                        reference_rms=reference_rms,
                    )
                _write_generation_record(path, record)
                processed += 1
                completed += 1
            if processed >= budget:
                break
    finally:
        if runtime is not None:
            runtime.close()
    complete = completed == expected
    if complete and not runbook.outputs["candidate_screen"].exists():
        raw_points: list[E8CandidatePoint] = []
        for condition, material, _reference in conditions:
            records = tuple(
                _load_generation_record(
                    _candidate_row_path(runbook, condition.condition_id, question.question_id)
                )
                for question in questions
            )
            policy = material if isinstance(material, AdaptivePolicySpec) else None
            alpha = policy.alpha_max if policy is not None else condition.alpha
            raw_points.append(
                E8CandidatePoint(
                    prompt_id=condition.system_prompt_id,
                    method=condition.steering_method,
                    candidate_condition_id=condition.condition_id,
                    alpha=alpha,
                    records=records,
                    adaptive_policy=policy,
                )
            )
        target, winners = _candidate_target(runbook, raw_points)
        selected_points = tuple(
            replace(
                point,
                selected_condition_id=(
                    point.candidate_condition_id
                    if winners[(point.prompt_id, point.method)] is point
                    else None
                ),
            )
            for point in raw_points
        )
        screen = E8CandidateScreen(
            matching_dimension=runbook.matching_dimension,
            target=target,
            tolerance=runbook.matching_tolerance,
            points=selected_points,
            runtime_artifact_sha256=context.runtime_artifact_sha256,
            execution_public_key=context.execution_public_key,
            source_question_bundle_sha256=sha256_path(
                runbook.outputs["development_side_effect_bundle"] / "questions"
            ),
            max_new_tokens=runbook.max_new_tokens,
        )
        candidate_sha = save_e8_candidate_screen(runbook.outputs["candidate_screen"], screen)
        registry = E8OperatingPointRegistry(
            matching_dimension=screen.matching_dimension,
            target=screen.target,
            tolerance=screen.tolerance,
            condition_ids_by_prompt={
                prompt: {
                    method: winners[(prompt, method)].candidate_condition_id
                    for method in _SCREEN_METHODS
                }
                for prompt in _PROMPTS
            },
            candidate_screen_sha256=candidate_sha,
        )
        save_e8_operating_point_registry(runbook.outputs["operating_point_registry"], registry)
    frozen_screen = (
        load_e8_candidate_screen(runbook.outputs["candidate_screen"])
        if runbook.outputs["candidate_screen"].exists()
        else None
    )
    return MappingProxyType(
        {
            "valid": True,
            "phase": "E8",
            "processed_rows": processed,
            "completed_rows": completed,
            "expected_rows": expected,
            "complete": complete,
            "candidate_screen_sha256": (
                sha256_file(runbook.outputs["candidate_screen"])
                if frozen_screen is not None
                else None
            ),
            "operating_point_registry_sha256": (
                sha256_file(runbook.outputs["operating_point_registry"])
                if frozen_screen is not None
                else None
            ),
            "matching_dimension": (
                frozen_screen.matching_dimension if frozen_screen is not None else None
            ),
            "target": frozen_screen.target if frozen_screen is not None else None,
        }
    )


def _selected_points(
    screen: E8CandidateScreen,
) -> Mapping[tuple[str, str], E8CandidatePoint]:
    values = {
        (item.prompt_id, item.method): item
        for item in screen.points
        if item.selected_condition_id is not None
    }
    if set(values) != {(prompt, method) for prompt in _PROMPTS for method in _SCREEN_METHODS}:
        raise FrozenArtifactError("E8 candidate screen selected-point inventory differs")
    return MappingProxyType(values)


def _final_contract(
    runbook: E8Runbook,
    context: _E8Context,
    *,
    side_effect_bundle_sha256: str,
) -> PhaseRunContract:
    screen = load_e8_candidate_screen(runbook.outputs["candidate_screen"])
    registry = load_e8_operating_point_registry(runbook.outputs["operating_point_registry"])
    if (
        registry.candidate_screen_sha256 != sha256_file(runbook.outputs["candidate_screen"])
        or screen.max_new_tokens != runbook.max_new_tokens
    ):
        raise FrozenArtifactError("E8 operating-point registry changed")
    selected = _selected_points(screen)
    conditions: list[EvaluationCondition] = []
    for benchmark in _QUESTION_COUNTS:
        for prompt_id in _PROMPTS:
            prompt = context.prompts[prompt_id]
            conditions.append(
                _condition(
                    context,
                    benchmark=benchmark,
                    prompt=prompt,
                    method="M0",
                )
            )
            for method in _SCREEN_METHODS:
                point = selected[(prompt_id, method)]
                source = point.records[0]
                if method == "M3":
                    if point.adaptive_policy is None:
                        raise FrozenArtifactError("E8 selected M3 point lacks its policy")
                    condition = _condition(
                        context,
                        benchmark=benchmark,
                        prompt=prompt,
                        method="M3",
                        method_artifact_sha256=source.metadata.get("method_artifact_sha256"),
                        adaptive_policy=point.adaptive_policy,
                    )
                else:
                    method_artifact = source.metadata.get("method_artifact_sha256")
                    if method != "M1" and type(method_artifact) is not str:
                        raise FrozenArtifactError(f"E8 selected {method} point lacks its artifact")
                    condition = _condition(
                        context,
                        benchmark=benchmark,
                        prompt=prompt,
                        method=method,
                        method_artifact_sha256=(
                            method_artifact if isinstance(method_artifact, str) else None
                        ),
                        layer=source.layer,
                        site=source.site,
                        token_scope=source.token_scope,
                        alpha=point.alpha,
                        sparsity=source.sparsity,
                    )
                if benchmark == "triviaqa" and (
                    condition.condition_id != point.selected_condition_id
                    or registry.condition_ids_by_prompt[prompt_id][method] != condition.condition_id
                ):
                    raise FrozenArtifactError(
                        "E8 selected candidate identity differs from the final TriviaQA condition"
                    )
                conditions.append(condition)
    contract = PhaseRunContract(
        phase=ExperimentPhase.E8,
        study_protocol_digest=context.study.digest,
        conditions=tuple(conditions),
        question_ids_by_benchmark={
            name: tuple(item.question_id for item in values)
            for name, values in context.questions.items()
        },
        input_fingerprints={
            "E6_transition_evidence": sha256_path(runbook.e6_transition_evidence),
            "E7_sparse_artifacts": sha256_path(runbook.e7_finalization),
            "protected_behavior_activations": sha256_path(
                runbook.outputs["protected_behavior_activations"]
            ),
            "frozen_side_effect_scorers": side_effect_bundle_sha256,
        },
        prerequisite_digests=_completion_digests(runbook, context),
        required_gates=context.study.phase(ExperimentPhase.E8).gates,
    )
    contract.assert_matches_study(context.study)
    if contract.expected_record_count != 86_040:
        raise DataValidationError("E8 final contract is not the registered 86,040 rows")
    return contract


def _freeze_final_side_effect_bundle(
    runbook: E8Runbook, context: _E8Context
) -> tuple[PhaseRunContract, str]:
    destination = runbook.outputs["side_effect_bundle"]
    provisional = _final_contract(runbook, context, side_effect_bundle_sha256="0" * 64)
    if destination.exists():
        side_sha = sha256_path(destination)
        contract = _final_contract(runbook, context, side_effect_bundle_sha256=side_sha)
        validate_side_effect_evaluation_bundle(destination, contract)
    else:
        side_sha = write_side_effect_evaluation_bundle(
            destination,
            provisional,
            context.questions,
            source_artifacts=_source_artifacts(runbook),
            scorer_execution_public_key=context.execution_public_key,
            ifeval_evaluator=runbook.ifeval_evaluator,
        )
        contract = _final_contract(runbook, context, side_effect_bundle_sha256=side_sha)
        validate_side_effect_evaluation_bundle(destination, contract)
    if sha256_path(runbook.outputs["development_side_effect_bundle"] / "questions") != sha256_path(
        destination / "questions"
    ):
        raise FrozenArtifactError("E8 promotion changed the screened question schedule")
    return contract, side_sha


def prepare_e8_ledger(runbook: E8Runbook) -> PhaseRunLedger:
    """Freeze the promoted 70-condition E8 matrix and create/reopen its ledger."""

    context = _base_context(runbook)
    contract, _side_sha = _freeze_final_side_effect_bundle(runbook, context)
    directory = runbook.outputs["run_directory"]
    if directory.exists():
        ledger = PhaseRunLedger.open(directory, study=context.study)
        if ledger.contract != contract:
            raise FrozenArtifactError("existing E8 ledger differs from its runbook")
        return ledger
    ledger = PhaseRunLedger.create(
        directory,
        contract,
        study=context.study,
        input_artifacts={
            "E6_transition_evidence": runbook.e6_transition_evidence,
            "E7_sparse_artifacts": runbook.e7_finalization,
            "protected_behavior_activations": runbook.outputs["protected_behavior_activations"],
            "frozen_side_effect_scorers": runbook.outputs["side_effect_bundle"],
        },
        prerequisite_runs={phase: path for phase, path in runbook.prerequisite_runs.items()},
    )
    runbook.outputs["final_work"].mkdir(parents=True, exist_ok=True)
    return ledger


def _final_row_path(runbook: E8Runbook, condition_id: str, question_id: str) -> Path:
    return runbook.outputs["final_work"] / f"{stable_hash([condition_id, question_id])}.json"


def _execution_material(
    runbook: E8Runbook,
    components: _E8Components,
    artifact: E8ProtectedArtifact,
    condition: EvaluationCondition,
) -> tuple[Any, float | None]:
    if condition.steering_method == "M0":
        return None, None
    if condition.steering_method == "M1":
        return components.dense_direction, artifact.reference_rms
    if condition.steering_method == "M3":
        if condition.adaptive_policy is None:
            raise FrozenArtifactError("E8 M3 final condition lacks its policy")
        return condition.adaptive_policy, None
    if condition.steering_method == "M4":
        sae = load_sae_intervention(components.sae_path)
        return sae.decoded_direction, artifact.reference_rms
    if condition.steering_method == "M5":
        return artifact.selected_direction, artifact.reference_rms
    raise DataValidationError("unknown E8 final method")


def execute_e8_final(runbook: E8Runbook, *, limit: int | None = None) -> Mapping[str, Any]:
    """Resume the exact signed 86,040-row E8 development matrix."""

    context = _base_context(runbook)
    ledger = prepare_e8_ledger(runbook)
    artifact = load_e8_protected_artifact(runbook.outputs["protected_artifact"])
    components = _e6_components(runbook, context, artifact.feature_schema)
    budget = ledger.contract.expected_record_count if limit is None else limit
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise DataValidationError("E8 final execution limit must be positive")
    questions = {
        (item.benchmark, item.question_id): item
        for values in context.questions.values()
        for item in values
    }
    controller_prompt = context.prompts[
        load_adaptive_controller(components.controller_path).risk_probe.training_schema.prompt_id
    ]
    runtime: MlxResearchRuntime | None = None
    attestor: E6RuntimeAttestor | None = None
    grader: E7E8DevelopmentGrader | None = None
    processed = 0
    try:
        for pending in ledger.iter_pending():
            if processed >= budget:
                break
            condition = pending.condition
            question = questions[(condition.benchmark, pending.question_id)]
            path = _final_row_path(runbook, condition.condition_id, pending.question_id)
            if path.exists():
                record = _load_generation_record(path)
            else:
                if runtime is None:
                    runtime, attestor = _native_runtime(runbook, context)
                    grader = E7E8DevelopmentGrader(
                        grader_bundle=runbook.outputs["side_effect_bundle"],
                        attestor=attestor,
                        environment_file=runbook.environment_file,
                    )
                assert attestor is not None and grader is not None
                prompt = context.prompts[condition.system_prompt_id]
                material, reference = _execution_material(runbook, components, artifact, condition)
                if condition.steering_method == "M3":
                    rendered = runtime.render_prompt(
                        prompt, question.text, metadata=question.metadata
                    )
                    record = execute_e8_adaptive_generation(
                        attestor=attestor,
                        runtime_artifact=runbook.runtime_artifact,
                        controller_artifact=components.controller_path,
                        question=question,
                        prompt=prompt,
                        controller_prompt=controller_prompt,
                        generation_record=_draft_record(
                            context,
                            question=question,
                            prompt=prompt,
                            condition=condition,
                            rendered_prompt_hash=rendered.sha256,
                        ),
                        condition=condition,
                        max_new_tokens=runbook.max_new_tokens,
                        populate_generation=True,
                        generation_grader=_bind_grader(grader, question),
                    )
                else:
                    record = _fixed_row(
                        runbook=runbook,
                        context=context,
                        runtime=runtime,
                        attestor=attestor,
                        grader=grader,
                        question=question,
                        prompt=prompt,
                        condition=condition,
                        direction=material,
                        reference_rms=reference,
                    )
                _write_generation_record(path, record)
            ledger.checkpoint((record,))
            processed += 1
    finally:
        if runtime is not None:
            runtime.close()
    completed, expected = ledger.progress()
    return MappingProxyType(
        {
            "valid": True,
            "phase": "E8",
            "processed_records": processed,
            "completed_records": completed,
            "expected_records": expected,
            "remaining_records": expected - completed,
            "contract_digest": ledger.contract.digest,
        }
    )


def finalize_e8_runbook(runbook: E8Runbook) -> Mapping[str, Any]:
    """Derive both E8 gates and publish the self-contained terminal artifact."""

    context = _base_context(runbook)
    ledger = PhaseRunLedger.open(runbook.outputs["run_directory"], study=context.study)
    completed, expected = ledger.progress()
    if completed != expected:
        raise DataValidationError("E8 finalization requires a complete ledger")
    return finalize_e8_phase(
        runbook.outputs["final_directory"],
        ledger_directory=ledger.directory,
        study=context.study,
        protected_artifact=runbook.outputs["protected_artifact"],
        operating_point_registry=runbook.outputs["operating_point_registry"],
        candidate_screen=runbook.outputs["candidate_screen"],
        runtime_artifact=runbook.runtime_artifact,
        analysis_protocol=runbook.analysis_protocol,
        research_plan=runbook.research_plan,
    )


def _protected_artifact_identity(artifact: E8ProtectedArtifact) -> str:
    def tensor_sha(value: torch.Tensor) -> str:
        frozen = np.ascontiguousarray(value.detach().cpu().float().numpy())
        return hashlib.sha256(frozen.tobytes(order="C")).hexdigest()

    return stable_hash(
        {
            "feature_schema": artifact.feature_schema.to_dict(),
            "evidence": [value.data_fingerprint for value in artifact.evidence],
            "dense_direction": tensor_sha(artifact.dense_direction),
            "variant_screens": [value.to_dict() for value in artifact.variant_screens],
            "source_fingerprints": dict(artifact.source_fingerprints),
            "layer": artifact.layer,
            "site": artifact.site.value,
            "token_scope": artifact.token_scope.value,
            "alpha": artifact.alpha,
            "reference_rms": artifact.reference_rms,
            "lambda_penalty": artifact.lambda_penalty,
            "ridge": artifact.ridge,
            "covariance_estimator": artifact.covariance_estimator,
        }
    )


def _candidate_selection_mapping(
    runbook: E8Runbook,
    candidate: E8CandidateScreen,
) -> Mapping[str, Mapping[str, str]]:
    target, winners = _candidate_target(runbook, candidate.points)
    if (
        candidate.matching_dimension != runbook.matching_dimension
        or candidate.tolerance != runbook.matching_tolerance
        or candidate.target != target
        or any(
            point.selected_condition_id
            != (
                point.candidate_condition_id
                if winners[(point.prompt_id, point.method)] is point
                else None
            )
            for point in candidate.points
        )
    ):
        raise FrozenArtifactError("E8 candidate matching policy does not replay")
    return MappingProxyType(
        {
            prompt: MappingProxyType(
                {
                    method: winners[(prompt, method)].candidate_condition_id
                    for method in _SCREEN_METHODS
                }
            )
            for prompt in _PROMPTS
        }
    )


def _verify_e8_intermediate_stages(
    runbook: E8Runbook,
    context: _E8Context,
    stages: Mapping[str, Any],
) -> E8CandidateScreen | None:
    """Replay every present construction artifact against the live runbook."""

    bundle = None
    components = None
    screens: tuple[M5VariantScreen, ...] | None = None
    artifact = None
    if stages["activations"]:
        bundle = load_e8_behavior_activation_bundle(
            runbook.outputs["protected_behavior_activations"]
        )
        expected_schema = _feature_schema(runbook, context)
        expected_pairs, _questions, question_source = _e7_label_material(runbook, expected_schema)
        expected_question_source = sha256_path(question_source)
        if (
            bundle.runtime_artifact_sha256 != context.runtime_artifact_sha256
            or bundle.execution_public_key != context.execution_public_key
            or bundle.source_question_bundle_sha256 != expected_question_source
            or bundle.feature_schema != expected_schema
            or any(
                evidence.label_pairs != expected_pairs[evidence.behavior]
                for evidence in bundle.evidence
            )
        ):
            raise FrozenArtifactError("E8 activation bundle derives from another runbook")
        components = _e6_components(runbook, context, bundle.feature_schema)
    if stages["variant_screens"]:
        if bundle is None or components is None:
            raise FrozenArtifactError("E8 variant screens lack their activation source")
        screens = _load_variant_screens(runbook.outputs["variant_screens"])
        factual, protected = _variant_questions(runbook, context)
        factual_ids = tuple(value.question_id for value in factual)
        protected_ids = {
            name: tuple(value.question_id for value in values) for name, values in protected.items()
        }
        if any(
            screen.runtime_artifact_sha256 != context.runtime_artifact_sha256
            or screen.execution_public_key != context.execution_public_key
            or screen.question_ids != factual_ids
            or dict(screen.protected_question_ids or {}) != protected_ids
            for screen in screens
        ):
            raise FrozenArtifactError("E8 variant screens derive from another runbook")
        prompt = context.prompts[bundle.feature_schema.prompt_id]
        questions_by_id = {
            value.question_id: value
            for value in (
                *factual,
                *(item for values in protected.values() for item in values),
            )
        }
        for screen in screens:
            assert screen.direction_sha256 is not None
            records_by_method = {
                "M0": (
                    *screen.baseline_execution_records,
                    *(
                        record
                        for records in (screen.protected_baseline_execution_records or {}).values()
                        for record in records
                    ),
                ),
                "M5": (
                    *screen.intervention_execution_records,
                    *(
                        record
                        for records in (
                            screen.protected_intervention_execution_records or {}
                        ).values()
                        for record in records
                    ),
                ),
            }
            for method, records in records_by_method.items():
                for record in records:
                    question = questions_by_id.get(record.question_id)
                    if question is None:
                        raise FrozenArtifactError("E8 variant record question is not registered")
                    condition = _condition(
                        context,
                        benchmark=question.benchmark,
                        prompt=prompt,
                        method=method,
                        method_artifact_sha256=(
                            screen.direction_sha256 if method == "M5" else None
                        ),
                        layer=components.layer if method == "M5" else None,
                        site=components.site if method == "M5" else None,
                        token_scope=components.token_scope if method == "M5" else None,
                        alpha=runbook.m5_alpha if method == "M5" else 0.0,
                        comparison_group=f"e8-variant-{question.benchmark}",
                    )
                    if (
                        record.condition_id != condition.condition_id
                        or record.metadata.get("source_question_sha256")
                        != question_source_fingerprint(question)
                        or record.metadata.get("prompt_template_sha256")
                        != condition.prompt_template_sha256
                        or record.metadata.get("method_artifact_sha256")
                        != condition.method_artifact_sha256
                        or record.metadata.get("decoding_max_new_tokens") != runbook.max_new_tokens
                    ):
                        raise FrozenArtifactError(
                            "E8 variant records derive from another prompt or question source"
                        )
        build_e8_protected_artifact(
            evidence=bundle.evidence,
            feature_schema=bundle.feature_schema,
            dense_direction=components.dense_direction,
            source_fingerprints={
                "E6_transition_evidence": sha256_path(runbook.e6_transition_evidence),
                "E7_sparse_artifacts": sha256_path(runbook.e7_finalization),
                "protected_behavior_activations": sha256_path(
                    runbook.outputs["protected_behavior_activations"]
                ),
            },
            variant_screens=screens,
            layer=components.layer,
            site=components.site,
            token_scope=components.token_scope,
            alpha=runbook.m5_alpha,
            reference_rms=components.reference_rms,
        )
    if stages["protected_artifact"]:
        if bundle is None or components is None or screens is None:
            raise FrozenArtifactError("E8 protected artifact lacks its screened sources")
        artifact = load_e8_protected_artifact(runbook.outputs["protected_artifact"])
        expected = build_e8_protected_artifact(
            evidence=bundle.evidence,
            feature_schema=bundle.feature_schema,
            dense_direction=components.dense_direction,
            source_fingerprints={
                "E6_transition_evidence": sha256_path(runbook.e6_transition_evidence),
                "E7_sparse_artifacts": sha256_path(runbook.e7_finalization),
                "protected_behavior_activations": sha256_path(
                    runbook.outputs["protected_behavior_activations"]
                ),
            },
            variant_screens=screens,
            layer=components.layer,
            site=components.site,
            token_scope=components.token_scope,
            alpha=runbook.m5_alpha,
            reference_rms=components.reference_rms,
        )
        if _protected_artifact_identity(artifact) != _protected_artifact_identity(expected):
            raise FrozenArtifactError("E8 protected artifact derives from another runbook")
    candidate = None
    if stages["candidate_screen"]:
        if artifact is None or components is None:
            raise FrozenArtifactError("E8 candidate screen lacks its protected source")
        candidate = load_e8_candidate_screen(runbook.outputs["candidate_screen"])
        questions = _candidate_question_set(runbook, context)
        expected_conditions = {
            condition.condition_id: condition
            for condition, _material, _reference_rms in _candidate_conditions(
                runbook, context, components, artifact
            )
        }
        _candidate_selection_mapping(runbook, candidate)
        question_by_id = {value.question_id: value for value in questions}
        if (
            candidate.runtime_artifact_sha256 != context.runtime_artifact_sha256
            or candidate.execution_public_key != context.execution_public_key
            or candidate.max_new_tokens != runbook.max_new_tokens
            or candidate.source_question_bundle_sha256
            != sha256_path(runbook.outputs["development_side_effect_bundle"] / "questions")
            or {point.candidate_condition_id for point in candidate.points}
            != set(expected_conditions)
            or any(
                tuple(record.question_id for record in point.records)
                != tuple(value.question_id for value in questions)
                for point in candidate.points
            )
        ):
            raise FrozenArtifactError("E8 candidate screen derives from another runbook")
        for point in candidate.points:
            expected_condition = expected_conditions[point.candidate_condition_id]
            for record in point.records:
                question = question_by_id[record.question_id]
                if (
                    record.metadata.get("source_question_sha256")
                    != question_source_fingerprint(question)
                    or record.metadata.get("prompt_template_sha256")
                    != expected_condition.prompt_template_sha256
                    or record.metadata.get("method_artifact_sha256")
                    != expected_condition.method_artifact_sha256
                ):
                    raise FrozenArtifactError(
                        "E8 candidate records derive from another prompt or question source"
                    )
    return candidate


def verify_e8_runbook(runbook: E8Runbook) -> Mapping[str, Any]:
    """Replay available E8 stages and terminal package without loading MLX."""

    context = _base_context(runbook)
    stages: dict[str, Any] = {
        "prepared": runbook.outputs["development_side_effect_bundle"].exists(),
        "activations": runbook.outputs["protected_behavior_activations"].exists(),
        "variant_screens": runbook.outputs["variant_screens"].exists(),
        "protected_artifact": runbook.outputs["protected_artifact"].exists(),
        "candidate_screen": runbook.outputs["candidate_screen"].exists(),
        "operating_points": runbook.outputs["operating_point_registry"].exists(),
        "ledger": runbook.outputs["run_directory"].exists(),
        "terminal": runbook.outputs["final_directory"].exists(),
    }
    if stages["prepared"]:
        validate_side_effect_evaluation_bundle(
            runbook.outputs["development_side_effect_bundle"],
            _provisional_contract(runbook, context),
        )
    candidate = _verify_e8_intermediate_stages(runbook, context, stages)
    if stages["operating_points"]:
        registry = load_e8_operating_point_registry(runbook.outputs["operating_point_registry"])
        selected_mapping = (
            _candidate_selection_mapping(runbook, candidate) if candidate is not None else None
        )
        if (
            candidate is None
            or registry.candidate_screen_sha256 != sha256_file(runbook.outputs["candidate_screen"])
            or registry.matching_dimension != candidate.matching_dimension
            or registry.target != candidate.target
            or registry.tolerance != candidate.tolerance
            or registry.condition_ids_by_prompt != selected_mapping
        ):
            raise FrozenArtifactError("E8 operating points differ from candidate screening")
    completed = 0
    expected = 0
    contract_digest: str | None = None
    ledger: PhaseRunLedger | None = None
    if stages["ledger"]:
        ledger = PhaseRunLedger.open(runbook.outputs["run_directory"], study=context.study)
        side_bundle = runbook.outputs["side_effect_bundle"]
        if not side_bundle.exists():
            raise FrozenArtifactError("E8 ledger lacks its final side-effect bundle")
        expected_contract = _final_contract(
            runbook,
            context,
            side_effect_bundle_sha256=sha256_path(side_bundle),
        )
        validate_side_effect_evaluation_bundle(side_bundle, expected_contract)
        if ledger.contract.digest != expected_contract.digest:
            raise FrozenArtifactError("E8 ledger differs from current screened artifacts")
        completed, expected = ledger.progress()
        contract_digest = ledger.contract.digest
    terminal: Mapping[str, Any] | None = None
    if stages["terminal"]:
        terminal = verify_e8_phase(runbook.outputs["final_directory"])
        if not stages["ledger"]:
            raise FrozenArtifactError("E8 terminal package lacks its source ledger")
        assert ledger is not None
        if terminal["status"] == "complete":
            live_digest = ledger.verify_complete().completion_digest
        else:
            live_digest = ledger.verify_falsified().falsification_digest
        if terminal["terminal_digest"] != live_digest:
            raise FrozenArtifactError("E8 terminal package differs from its live ledger")
    return MappingProxyType(
        {
            "valid": True,
            "phase": "E8",
            "runbook_digest": runbook.runbook_digest,
            "contract_digest": contract_digest,
            "completed_records": completed,
            "expected_records": expected,
            "remaining_records": expected - completed,
            "stages": stages,
            "terminal": dict(terminal) if terminal is not None else None,
        }
    )

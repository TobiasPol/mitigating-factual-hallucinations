"""Resumable native-VLLM operator lifecycle for the registered E7 study."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from mfh.config import load_model_spec, load_prompt_specs
from mfh.contracts import (
    ActivationSite,
    GenerationRecord,
    ModelSpec,
    Outcome,
    PromptSpec,
    Question,
    TokenScope,
)
from mfh.data.io import read_questions, write_questions
from mfh.data.language_suite import load_reviewed_language_suite
from mfh.data.source_snapshots import SOURCE_SNAPSHOTS, iter_source_questions
from mfh.data.splits import semantic_group_ids
from mfh.errors import ConfigurationError, DataValidationError, FrozenArtifactError
from mfh.experiments.e6_likelihood import (
    E6RuntimeAttestor,
    _load_e6_runtime_attestation,
)
from mfh.experiments.e6_operator import _e3_slice
from mfh.experiments.e7_e8_grading import E7E8DevelopmentGrader
from mfh.experiments.e7_sparse import (
    execute_coordinate_screen_generation,
    execute_e7_activation_capture_batch,
    execute_e7_causal_feature_generation,
    execute_e7_generation,
    execute_e7_interpretability_generation,
    execute_e7_tsteer_activation_capture_batch,
    finalize_e7_phase,
    validate_separate_sae_corpus,
    verify_e7_phase,
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
from mfh.inference.transformers_snapshot import verify_transformers_snapshot
from mfh.inference.vllm_research import VllmResearchRuntime
from mfh.methods.features import (
    ActivationFeatureSchema,
    ActivationKind,
    FeatureComposition,
)
from mfh.methods.probes import ProbeDataset
from mfh.methods.sae_stability import (
    _aligned_stability,
    _oriented_decoder_directions,
    load_sae_stability_bundle,
    write_sae_stability_bundle,
)
from mfh.methods.sparse import (
    ActivationBatch,
    ActivationCorpus,
    CoordinateScreenPoint,
    LongComputationReceipt,
    SAECheckpointResultSequence,
    SAEConfig,
    SAEInterpretabilityAudit,
    SAEPairedExecutionAudit,
    SAEPromotionCriteria,
    SAESparsity,
    SAESparsitySweepPoint,
    SAETrainingResult,
    coordinate_screen_condition_id,
    coordinate_screen_contract_digest,
    coordinate_sparse_direction,
    fit_coordinate_sparse_artifact,
    fit_e7_sae_sweep_measured,
    latent_factuality_direction,
    load_activation_corpus,
    load_coordinate_sparse_artifact,
    load_sae_intervention,
    measure_feature_intervention_evidence,
    promote_sae_intervention,
    sae_checkpoint_fingerprint,
    sae_config_fingerprint,
    save_coordinate_sparse_artifact,
    save_sae_intervention,
    write_activation_corpus,
)
from mfh.provenance import sha256_file, sha256_path, stable_hash

_PROMPTS = ("P0-neutral", "P2-calibrated-abstention")
_METHODS = ("M0", "M4a", "M4b")
_QUESTION_COUNTS = {
    "triviaqa": 5_000,
    "ifeval": 541,
    "xstest": 250,
    "strongreject_or_harmbench": 313,
    "language_consistency": 500,
}
_FRACTIONS = (0.01, 0.05, 0.10, 0.25)
_ALPHAS = (0.1, 0.25, 0.5, 1.0, 2.0)
_RUNBOOK_KEYS = {
    "schema_version",
    "phase",
    "study_protocol",
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
    "e3_static_vectors",
    "e5_adaptive_controllers",
    "prerequisite_runs",
    "outputs",
    "seed",
    "max_new_tokens",
    "capture_batch_size",
    "activation_shard_rows",
    "sae_training_rows",
    "sae_validation_rows",
    "coordinate_question_count",
    "interpretability_question_count",
    "feature_count",
    "m1_tensor_index",
    "sae_configs",
}
_OUTPUT_KEYS = {
    "sae_question_bundle",
    "development_side_effect_bundle",
    "side_effect_bundle",
    "capture_work",
    "tsteer_corpus",
    "separate_sae_corpus",
    "coordinate_work",
    "coordinate_artifact",
    "sae_sweep",
    "sae_selection",
    "causal_work",
    "interpretability_work",
    "final_work",
    "sae_intervention",
    "sae_stability_bundle",
    "run_directory",
    "final_directory",
}
_SOURCE_KEYS = {
    "triviaqa",
    "ifeval",
    "xstest",
    "strongreject_or_harmbench",
}


def _path(root: Path, value: object, context: str) -> Path:
    if type(value) is not str or not value.strip():
        raise DataValidationError(f"E7 runbook {context} path is invalid")
    raw = Path(value)
    return (raw if raw.is_absolute() else root / raw).resolve()


def _path_map(root: Path, value: object, expected: set[str], context: str) -> Mapping[str, Path]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise DataValidationError(f"E7 runbook {context} keys differ")
    return MappingProxyType(
        {name: _path(root, raw, f"{context}.{name}") for name, raw in value.items()}
    )


@dataclass(frozen=True, slots=True)
class E7Runbook:
    source: Path
    study_protocol: Path
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
    e3_static_vectors: Path
    e5_adaptive_controllers: Path
    prerequisite_runs: Mapping[ExperimentPhase, Path]
    outputs: Mapping[str, Path]
    seed: int
    max_new_tokens: int
    capture_batch_size: int
    activation_shard_rows: int
    sae_training_rows: int
    sae_validation_rows: int
    coordinate_question_count: int
    interpretability_question_count: int
    feature_count: int
    m1_tensor_index: tuple[str, str, str, int]
    sae_configs: tuple[SAEConfig, ...]
    runbook_digest: str

    @classmethod
    def load(cls, path: str | Path) -> E7Runbook:
        source = Path(path).resolve()
        try:
            value = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FrozenArtifactError(f"cannot read E7 runbook: {exc}") from exc
        if not isinstance(value, dict) or set(value) != _RUNBOOK_KEYS:
            raise DataValidationError("E7 runbook keys differ from schema version 1")
        root = source.parent
        prerequisites = value["prerequisite_runs"]
        tensor = value["m1_tensor_index"]
        raw_configs = value["sae_configs"]
        integer_fields = (
            "seed",
            "max_new_tokens",
            "capture_batch_size",
            "activation_shard_rows",
            "sae_training_rows",
            "sae_validation_rows",
            "coordinate_question_count",
            "interpretability_question_count",
            "feature_count",
        )
        if (
            value["schema_version"] != 1
            or value["phase"] != "E7"
            or not isinstance(prerequisites, Mapping)
            or set(prerequisites) != {"E3", "E5", "E6"}
            or type(tensor) is not list
            or len(tensor) != 4
            or any(type(item) is not str for item in tensor[:3])
            or type(tensor[3]) is not int
            or type(value["max_new_tokens"]) is not int
            or not 32 <= value["max_new_tokens"] <= 48
            or any(
                type(value[name]) is not int or value[name] <= (0 if name != "seed" else -1)
                for name in integer_fields
                if name != "max_new_tokens"
            )
            or type(raw_configs) is not list
            or len(raw_configs) < 6
        ):
            raise DataValidationError("E7 runbook values are invalid")
        configs: list[SAEConfig] = []
        try:
            for raw in raw_configs:
                if not isinstance(raw, Mapping):
                    raise TypeError
                config = dict(raw)
                config["sparsity"] = SAESparsity(str(config["sparsity"]))
                configs.append(SAEConfig(**config))
        except (KeyError, TypeError, ValueError) as exc:
            raise DataValidationError(f"E7 SAE config grid is invalid: {exc}") from exc
        return cls(
            source=source,
            study_protocol=_path(root, value["study_protocol"], "study_protocol"),
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
            e3_static_vectors=_path(root, value["e3_static_vectors"], "e3_static_vectors"),
            e5_adaptive_controllers=_path(
                root, value["e5_adaptive_controllers"], "e5_adaptive_controllers"
            ),
            prerequisite_runs=MappingProxyType(
                {
                    ExperimentPhase(name): _path(root, raw, f"prerequisite_runs.{name}")
                    for name, raw in prerequisites.items()
                }
            ),
            outputs=_path_map(root, value["outputs"], _OUTPUT_KEYS, "outputs"),
            seed=value["seed"],
            max_new_tokens=value["max_new_tokens"],
            capture_batch_size=value["capture_batch_size"],
            activation_shard_rows=value["activation_shard_rows"],
            sae_training_rows=value["sae_training_rows"],
            sae_validation_rows=value["sae_validation_rows"],
            coordinate_question_count=value["coordinate_question_count"],
            interpretability_question_count=value["interpretability_question_count"],
            feature_count=value["feature_count"],
            m1_tensor_index=(tensor[0], tensor[1], tensor[2], tensor[3]),
            sae_configs=tuple(configs),
            runbook_digest=stable_hash(value),
        )


def write_e7_runbook_template(path: str | Path, *, m1_layer: int) -> str:
    """Write a secret-free E7 runbook for the single-A100 vLLM host."""

    if isinstance(m1_layer, bool) or not isinstance(m1_layer, int) or not 0 <= m1_layer < 64:
        raise DataValidationError("E7 M1 layer must be an explicit Qwen layer index")
    destination = Path(path).resolve()
    if destination.exists() or destination.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite E7 runbook: {destination}")
    configs = [
        {
            "input_width": 5_120,
            "expansion_factor": 8,
            "sparsity": "top_k",
            "top_k": top_k,
            "l1_coefficient": 0.001,
            "epochs": 30,
            "batch_size": 256,
            "learning_rate": 0.001,
            "seed": seed,
        }
        for top_k in (16, 32, 64)
        for seed in (17, 29)
    ]
    body = {
        "schema_version": 1,
        "phase": "E7",
        "study_protocol": "../../../../configs/experiments/phases.yaml",
        "model_config": "../../../../configs/models/qwen3.6-27b-nvfp4.yaml",
        "prompt_config": "../../../../configs/prompts/primary.yaml",
        "snapshot_directory": (
            "../../../models/qwen3.6-27b-nvfp4/"
            "0893e1606ff3d5f97a441f405d5fc541a6bdf404"
        ),
        "snapshot_manifest": "../../../../configs/models/qwen3.6-27b-nvfp4.snapshot.json",
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
            "xstest": "../frozen/E7-E8-external-inputs/sources/xstest.csv",
            "strongreject_or_harmbench": (
                "../frozen/E7-E8-external-inputs/sources/"
                "strongreject_or_harmbench.csv"
            ),
        },
        "reviewed_language_suite": "../frozen/E7-E8-external-inputs/language-suite",
        "ifeval_evaluator": "../frozen/E7-E8-external-inputs/ifeval-evaluator",
        "e3_static_vectors": "../E3-operator/vectors",
        "e5_adaptive_controllers": "../frozen/E5-phase/selected-controller",
        "prerequisite_runs": {
            "E3": "../E3-operator/phase",
            "E5": "../runs/E5",
            "E6": "../runs/E6",
        },
        "outputs": {
            "sae_question_bundle": "../work/E7/sae-questions",
            "development_side_effect_bundle": "../work/E7/development-side-effects",
            "side_effect_bundle": "../frozen/E7-side-effects",
            "capture_work": "../work/E7/capture",
            "tsteer_corpus": "../frozen/E7-tsteer-corpus",
            "separate_sae_corpus": "../frozen/E7-separate-sae-corpus",
            "coordinate_work": "../work/E7/coordinate",
            "coordinate_artifact": "../frozen/E7-coordinate",
            "sae_sweep": "../work/E7/sae-sweep",
            "sae_selection": "../work/E7/sae-selection",
            "causal_work": "../work/E7/causal",
            "interpretability_work": "../work/E7/interpretability",
            "final_work": "../work/E7/final-rows",
            "sae_intervention": "../frozen/E7-sae-intervention",
            "sae_stability_bundle": "../frozen/E7-sae-stability",
            "run_directory": "../runs/E7",
            "final_directory": "../final/E7",
        },
        "seed": 17,
        "max_new_tokens": 32,
        "capture_batch_size": 64,
        "activation_shard_rows": 2_048,
        "sae_training_rows": 10_000,
        "sae_validation_rows": 2_000,
        "coordinate_question_count": 500,
        "interpretability_question_count": 100,
        "feature_count": 8,
        "m1_tensor_index": ["P0-neutral", "M1-P", "post_mlp", m1_layer],
        "sae_configs": configs,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sha256_file(destination)


def _private_key(runbook: E7Runbook) -> tuple[str, str]:
    try:
        private_hex = runbook.execution_key_file.read_text(encoding="utf-8").strip()
        private = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_hex))
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"cannot load E7 execution key: {exc}") from exc
    if len(private_hex) != 64 or private_hex.lower() != private_hex:
        raise ConfigurationError("E7 execution key must be one lowercase 32-byte hex key")
    public_hex = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    return private_hex, public_hex


@dataclass(frozen=True, slots=True)
class _E7BaseContext:
    study: StudyProtocol
    model: ModelSpec
    prompts: Mapping[str, PromptSpec]
    final_questions: Mapping[str, tuple[Question, ...]]
    tsteer_questions: tuple[Question, ...]
    sae_training_questions: tuple[Question, ...]
    sae_validation_questions: tuple[Question, ...]
    execution_private_key: str
    execution_public_key: str
    runtime_artifact_sha256: str
    runtime_identity: Mapping[str, Any]
    seed: int


def _load_source_questions(path: Path, benchmark: str) -> tuple[Question, ...]:
    snapshot = SOURCE_SNAPSHOTS[benchmark]
    return tuple(iter_source_questions(snapshot, path))


def _select_sae_questions(
    reserved: Sequence[Question], *, training_rows: int, validation_rows: int, seed: int
) -> tuple[tuple[Question, ...], tuple[Question, ...]]:
    groups = semantic_group_ids(reserved)
    by_group: dict[str, list[Question]] = {}
    for question in reserved:
        by_group.setdefault(groups[question.question_id], []).append(question)
    ordered = sorted(
        by_group.values(),
        key=lambda values: stable_hash(
            {"seed": seed, "ids": sorted(item.question_id for item in values)}
        ),
    )
    selected: dict[str, list[Question]] = {"sae-train": [], "sae-validation": []}
    targets = {"sae-train": training_rows, "sae-validation": validation_rows}
    for values in ordered:
        eligible = [name for name in targets if len(selected[name]) + len(values) <= targets[name]]
        if not eligible:
            continue
        name = max(
            eligible,
            key=lambda item: (targets[item] - len(selected[item])) / targets[item],
        )
        selected[name].extend(replace(item, split=name) for item in values)
        if all(len(selected[item]) == targets[item] for item in targets):
            break
    if any(len(selected[name]) != target for name, target in targets.items()):
        raise DataValidationError("reserved TriviaQA cannot fill exact disjoint SAE cohorts")
    return tuple(selected["sae-train"]), tuple(selected["sae-validation"])


def _base_context(runbook: E7Runbook) -> _E7BaseContext:
    validate_active_study_artifact_paths(
        {
            "E7 runbook": runbook.source,
            "E7 reviewed splits": runbook.reviewed_splits,
            "E7 language suite": runbook.reviewed_language_suite,
            "E7 IFEval evaluator": runbook.ifeval_evaluator,
            "E7 E3 vectors": runbook.e3_static_vectors,
            "E7 E5 controllers": runbook.e5_adaptive_controllers,
            "E7 runtime": runbook.runtime_artifact,
            **{f"E7 source {name}": path for name, path in runbook.source_artifacts.items()},
            **{
                f"E7 prerequisite {phase.value}": path
                for phase, path in runbook.prerequisite_runs.items()
            },
            **{f"E7 output {name}": path for name, path in runbook.outputs.items()},
        }
    )
    repository_lock = Path(__file__).resolve().parents[3] / "uv.lock"
    if sha256_file(runbook.package_lock) != sha256_file(repository_lock):
        raise FrozenArtifactError(
            "E7 runbook package lock differs from the executing repository lock"
        )
    study = load_study_protocol(runbook.study_protocol)
    model = load_model_spec(runbook.model_config)
    validate_active_model_spec(model)
    verify_transformers_snapshot(model, runbook.snapshot_directory, runbook.snapshot_manifest)
    prompts_all = {item.prompt_id: item for item in load_prompt_specs(runbook.prompt_config)}
    if not set(_PROMPTS) <= set(prompts_all):
        raise DataValidationError("E7 prompt configuration lacks P0/P2")
    prompts = MappingProxyType({name: prompts_all[name] for name in _PROMPTS})
    private_hex, public_hex = _private_key(runbook)
    runtime = _load_e6_runtime_attestation(runbook.runtime_artifact)
    runtime_identity = runtime["runtime_identity"]
    if (
        runtime["execution_public_key"] != public_hex
        or not isinstance(runtime_identity, Mapping)
        or runtime_identity.get("snapshot_sha256") != sha256_path(runbook.snapshot_directory)
    ):
        raise FrozenArtifactError("E7 key or snapshot differs from the E6 runtime attestation")
    tsteer = tuple(read_questions(runbook.reviewed_splits / "T-steer.jsonl"))
    reserved = tuple(read_questions(runbook.reviewed_splits / "reserved.jsonl"))
    trivia = tuple(read_questions(runbook.reviewed_splits / "T-dev.jsonl"))
    try:
        split_manifest = json.loads(
            (runbook.reviewed_splits / "manifest.json").read_text(encoding="utf-8")
        )
        split_body = dict(split_manifest)
        split_digest = split_body.pop("manifest_digest")
        split_artifacts = split_body["artifacts"]
        split_ids = split_body["split_question_ids_sha256"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise FrozenArtifactError(f"cannot replay E7 reviewed split manifest: {exc}") from exc
    if (
        split_digest != stable_hash(split_body)
        or split_body.get("scientific_eligible") is not True
        or not isinstance(split_artifacts, Mapping)
        or not isinstance(split_ids, Mapping)
        or split_ids.get("T-steer")
        != stable_hash([value.question_id for value in tsteer])
        or split_artifacts.get("T-steer.jsonl", {}).get("sha256")
        != sha256_file(runbook.reviewed_splits / "T-steer.jsonl")
        or split_artifacts.get("reserved.jsonl", {}).get("sha256")
        != sha256_file(runbook.reviewed_splits / "reserved.jsonl")
    ):
        raise FrozenArtifactError("E7 cohorts differ from the canonical E3 reviewed split")
    train, validation = _select_sae_questions(
        reserved,
        training_rows=runbook.sae_training_rows,
        validation_rows=runbook.sae_validation_rows,
        seed=runbook.seed,
    )
    final_questions = {
        "triviaqa": trivia,
        "ifeval": _load_source_questions(runbook.source_artifacts["ifeval"], "ifeval"),
        "xstest": _load_source_questions(runbook.source_artifacts["xstest"], "xstest"),
        "strongreject_or_harmbench": _load_source_questions(
            runbook.source_artifacts["strongreject_or_harmbench"],
            "strongreject_or_harmbench",
        ),
        "language_consistency": load_reviewed_language_suite(runbook.reviewed_language_suite),
    }
    if (
        len(tsteer) != 30_000
        or any(item.benchmark != "triviaqa" or item.split != "T-steer" for item in tsteer)
        or any(len(final_questions[name]) != count for name, count in _QUESTION_COUNTS.items())
        or len({item.question_id for values in final_questions.values() for item in values})
        != sum(_QUESTION_COUNTS.values())
    ):
        raise DataValidationError("E7 frozen question schedules differ from registration")
    for phase, path in runbook.prerequisite_runs.items():
        prerequisite = open_phase_prerequisite(path, phase=phase, study=study)
        prerequisite.verify_complete()
        if (
            phase is ExperimentPhase.E3
            and prerequisite.contract.input_fingerprints.get("reviewed_splits")
            != sha256_path(runbook.reviewed_splits)
        ):
            raise FrozenArtifactError("E7 reviewed splits differ from completed E3")
    return _E7BaseContext(
        study=study,
        model=model,
        prompts=prompts,
        final_questions=MappingProxyType(final_questions),
        tsteer_questions=tsteer,
        sae_training_questions=train,
        sae_validation_questions=validation,
        execution_private_key=private_hex,
        execution_public_key=public_hex,
        runtime_artifact_sha256=sha256_path(runbook.runtime_artifact),
        runtime_identity=MappingProxyType(dict(runtime_identity)),
        seed=runbook.seed,
    )


def _completion_digests(runbook: E7Runbook, context: _E7BaseContext) -> Mapping[str, str]:
    return MappingProxyType(
        {
            phase.value: open_phase_prerequisite(path, phase=phase, study=context.study)
            .verify_complete()
            .completion_digest
            for phase, path in runbook.prerequisite_runs.items()
        }
    )


def _evaluation_condition(
    context: _E7BaseContext,
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
) -> EvaluationCondition:
    return EvaluationCondition(
        phase=ExperimentPhase.E7,
        benchmark=benchmark,
        partition="T-dev",
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
        comparison_group=f"e7-{benchmark}-{prompt.prompt_id}",
    )


def runbook_seed(context: _E7BaseContext) -> int:
    """Recover the seed attested by the E6 runtime identity."""

    value = context.runtime_identity.get("seed")
    if type(value) is not int or value < 0:
        # Older attestation schemas bind deterministic decoding without exposing
        # the seed.  The registered study seed is fixed at 17 in that schema.
        return context.seed
    return value


def _provisional_contract(runbook: E7Runbook, context: _E7BaseContext) -> PhaseRunContract:
    conditions = tuple(
        _evaluation_condition(
            context,
            benchmark=benchmark,
            prompt=context.prompts[prompt_id],
            method="M0",
        )
        for benchmark in _QUESTION_COUNTS
        for prompt_id in _PROMPTS
    )
    return PhaseRunContract(
        phase=ExperimentPhase.E7,
        study_protocol_digest=context.study.digest,
        conditions=conditions,
        question_ids_by_benchmark={
            benchmark: tuple(item.question_id for item in values)
            for benchmark, values in context.final_questions.items()
        },
        input_fingerprints={"provisional": "0" * 64},
        prerequisite_digests=_completion_digests(runbook, context),
        required_gates=context.study.phase(ExperimentPhase.E7).gates,
    )


def _source_artifacts(runbook: E7Runbook) -> Mapping[str, Path]:
    return MappingProxyType(
        {
            "triviaqa": runbook.source_artifacts["triviaqa"],
            "ifeval": runbook.source_artifacts["ifeval"],
            "xstest": runbook.source_artifacts["xstest"],
            "strongreject_or_harmbench": runbook.source_artifacts["strongreject_or_harmbench"],
            "language_consistency": runbook.reviewed_language_suite,
        }
    )


def _write_sae_question_bundle(runbook: E7Runbook, context: _E7BaseContext) -> str:
    destination = runbook.outputs["sae_question_bundle"]
    expected = {
        "train.jsonl": context.sae_training_questions,
        "validation.jsonl": context.sae_validation_questions,
    }
    if destination.exists():
        if (
            destination.is_symlink()
            or not destination.is_dir()
            or {item.name for item in destination.iterdir()} != set(expected)
            or any(
                tuple(read_questions(destination / name)) != questions
                for name, questions in expected.items()
            )
        ):
            raise FrozenArtifactError("existing E7 SAE question bundle differs")
        return sha256_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        for name, questions in expected.items():
            write_questions(stage / name, questions)
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return sha256_path(destination)


def prepare_e7_runbook(runbook: E7Runbook) -> Mapping[str, Any]:
    """Freeze E7 construction questions and the scorer bundle used before promotion."""

    context = _base_context(runbook)
    question_sha = _write_sae_question_bundle(runbook, context)
    destination = runbook.outputs["development_side_effect_bundle"]
    contract = _provisional_contract(runbook, context)
    if destination.exists():
        side_sha = validate_side_effect_evaluation_bundle(destination, contract)
    else:
        side_sha = write_side_effect_evaluation_bundle(
            destination,
            contract,
            context.final_questions,
            source_artifacts=_source_artifacts(runbook),
            scorer_execution_public_key=context.execution_public_key,
            ifeval_evaluator=runbook.ifeval_evaluator,
        )
    return MappingProxyType(
        {
            "valid": True,
            "phase": "E7",
            "runbook_digest": runbook.runbook_digest,
            "sae_question_bundle_sha256": question_sha,
            "development_side_effect_bundle_sha256": side_sha,
            "question_bundle_sha256": sha256_path(destination / "questions"),
            "execution_public_key": context.execution_public_key,
        }
    )


def preflight_e7_runbook(runbook: E7Runbook) -> Mapping[str, Any]:
    """Replay every immutable E7 source without loading VLLM or writing outputs."""

    context = _base_context(runbook)
    status = {
        name: ("present" if path.exists() else "pending") for name, path in runbook.outputs.items()
    }
    if runbook.outputs["development_side_effect_bundle"].exists():
        validate_side_effect_evaluation_bundle(
            runbook.outputs["development_side_effect_bundle"],
            _provisional_contract(runbook, context),
        )
    return MappingProxyType(
        {
            "valid": True,
            "phase": "E7",
            "runbook_digest": runbook.runbook_digest,
            "runtime_artifact_sha256": context.runtime_artifact_sha256,
            "execution_public_key": context.execution_public_key,
            "tsteer_rows": len(context.tsteer_questions),
            "sae_training_rows": len(context.sae_training_questions),
            "sae_validation_rows": len(context.sae_validation_questions),
            "final_question_rows": sum(len(value) for value in context.final_questions.values()),
            "outputs": status,
        }
    )


def _native_runtime(
    runbook: E7Runbook, context: _E7BaseContext
) -> tuple[VllmResearchRuntime, E6RuntimeAttestor]:
    provenance = context.runtime_identity.get("research_provenance")
    if not isinstance(provenance, Mapping):
        raise FrozenArtifactError("E6 runtime attestation lacks research provenance")
    runtime = VllmResearchRuntime.from_spec(
        context.model,
        snapshot_path=runbook.snapshot_directory,
        seed=runbook.seed,
        research_provenance=dict(provenance),
    )
    attestor = E6RuntimeAttestor(runtime, execution_private_key=context.execution_private_key)
    attestor.verify_runtime_artifact(runbook.runtime_artifact)
    return runtime, attestor


def _feature_schema(
    runbook: E7Runbook, context: _E7BaseContext, partition: str
) -> ActivationFeatureSchema:
    direction, _rms, _sha = _e3_slice(runbook.e3_static_vectors, runbook.m1_tensor_index)
    prompt = context.prompts[runbook.m1_tensor_index[0]]
    return ActivationFeatureSchema(
        benchmark="triviaqa",
        partition=partition,
        split_manifest_digest=sha256_path(runbook.reviewed_splits),
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
        width=int(direction.size),
        token_scope=None,
    )


def _capture_questions(
    runbook: E7Runbook, context: _E7BaseContext, partition: str
) -> tuple[tuple[Question, ...], Path]:
    if partition == "T-steer":
        return context.tsteer_questions, runbook.reviewed_splits / "T-steer.jsonl"
    if partition == "sae-train":
        return (
            context.sae_training_questions,
            runbook.outputs["sae_question_bundle"],
        )
    if partition == "sae-validation":
        return (
            context.sae_validation_questions,
            runbook.outputs["sae_question_bundle"],
        )
    raise DataValidationError("E7 capture partition is invalid")


def _capture_manifest(
    runbook: E7Runbook,
    context: _E7BaseContext,
    partition: str,
    questions: Sequence[Question],
    source: Path,
    schema: ActivationFeatureSchema,
) -> dict[str, Any]:
    body = {
        "schema_version": 1,
        "partition": partition,
        "runbook_digest": runbook.runbook_digest,
        "runtime_artifact_sha256": context.runtime_artifact_sha256,
        "execution_public_key": context.execution_public_key,
        "source_question_bundle_sha256": sha256_path(source),
        "feature_schema": schema.to_dict(),
        "question_ids": [item.question_id for item in questions],
        "batch_size": runbook.capture_batch_size,
        "dtype": "float16",
    }
    return {**body, "manifest_digest": stable_hash(body)}


def _capture_batch_path(root: Path, index: int) -> tuple[Path, Path]:
    return root / f"batch-{index:05d}.npy", root / f"batch-{index:05d}.json"


def _write_capture_batch(root: Path, index: int, batch: ActivationBatch) -> None:
    array_path, json_path = _capture_batch_path(root, index)
    if array_path.exists() or json_path.exists():
        raise FrozenArtifactError("refusing to overwrite E7 capture batch")
    stage = Path(tempfile.mkdtemp(prefix=f".batch-{index:05d}-", dir=root))
    try:
        staged_array = stage / array_path.name
        values = np.asarray(batch.activations, dtype=np.float32)
        np.save(staged_array, values, allow_pickle=False)
        body = {
            "schema_version": 1,
            "question_ids": list(batch.question_ids),
            "outcomes": [item.value for item in batch.outcomes],
            "group_ids": list(batch.group_ids),
            "capture_receipts": [dict(item) for item in batch.capture_receipts],
            "activations_sha256": sha256_file(staged_array),
        }
        (stage / json_path.name).write_text(
            json.dumps({**body, "batch_digest": stable_hash(body)}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staged_array, array_path)
        os.replace(stage / json_path.name, json_path)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _load_capture_batch(root: Path, index: int) -> ActivationBatch:
    array_path, json_path = _capture_batch_path(root, index)
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        digest = payload.pop("batch_digest")
        values = np.load(array_path, allow_pickle=False)
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot load E7 capture batch: {exc}") from exc
    if (
        digest != stable_hash(payload)
        or payload.get("activations_sha256") != sha256_file(array_path)
        or not isinstance(payload.get("capture_receipts"), list)
    ):
        raise FrozenArtifactError("E7 capture batch digest differs")
    return ActivationBatch(
        question_ids=tuple(payload["question_ids"]),
        activations=torch.from_numpy(np.asarray(values, dtype=np.float32).copy()),
        outcomes=tuple(Outcome(item) for item in payload["outcomes"]),
        group_ids=tuple(payload["group_ids"]),
        capture_receipts=tuple(payload["capture_receipts"]),
    )


def _capture_root(
    runbook: E7Runbook,
    context: _E7BaseContext,
    partition: str,
    questions: Sequence[Question],
    source: Path,
    schema: ActivationFeatureSchema,
) -> Path:
    root = runbook.outputs["capture_work"] / partition
    manifest = _capture_manifest(runbook, context, partition, questions, source, schema)
    if root.exists():
        try:
            observed = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FrozenArtifactError(f"cannot read E7 capture manifest: {exc}") from exc
        if observed != manifest:
            raise FrozenArtifactError("E7 capture resume manifest differs")
    else:
        root.mkdir(parents=True)
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return root


def _materialize_activation_corpus(
    runbook: E7Runbook,
    context: _E7BaseContext,
    partition: str,
    questions: Sequence[Question],
    source: Path,
    schema: ActivationFeatureSchema,
    root: Path,
) -> Path:
    expected_batches = math.ceil(len(questions) / runbook.capture_batch_size)
    batches = tuple(_load_capture_batch(root, index) for index in range(expected_batches))
    private = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(context.execution_private_key))

    def sign(body: Mapping[str, Any]) -> str:
        from mfh.provenance import canonical_json

        return private.sign(canonical_json(body).encode()).hex()

    if partition == "T-steer":
        destination = runbook.outputs["tsteer_corpus"]
    else:
        destination = runbook.outputs["capture_work"] / f"materialized-{partition}"
    if not destination.exists():
        write_activation_corpus(
            destination,
            batches,
            feature_schema=schema,
            shard_rows=runbook.activation_shard_rows,
            dtype="float16",
            runtime_artifact_sha256=context.runtime_artifact_sha256,
            execution_public_key=context.execution_public_key,
            source_question_bundle_sha256=sha256_path(source),
            capture_signer=sign,
        )
    return destination


def _materialize_separate_sae_root(runbook: E7Runbook) -> None:
    destination = runbook.outputs["separate_sae_corpus"]
    if destination.exists():
        validate_separate_sae_corpus(destination)
        return
    train = runbook.outputs["capture_work"] / "materialized-sae-train"
    validation = runbook.outputs["capture_work"] / "materialized-sae-validation"
    questions = runbook.outputs["sae_question_bundle"]
    if not all(item.exists() for item in (train, validation, questions)):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        shutil.copytree(train, stage / "train")
        shutil.copytree(validation, stage / "validation")
        shutil.copytree(questions, stage / "questions")
        validate_separate_sae_corpus(stage)
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def execute_e7_capture(
    runbook: E7Runbook, *, partition: str, limit: int | None = None
) -> Mapping[str, Any]:
    """Resume one native activation-capture partition and freeze it when complete."""

    prepare_e7_runbook(runbook)
    context = _base_context(runbook)
    questions, source = _capture_questions(runbook, context, partition)
    schema = _feature_schema(runbook, context, partition)
    root = _capture_root(runbook, context, partition, questions, source, schema)
    expected_batches = math.ceil(len(questions) / runbook.capture_batch_size)
    completed = 0
    for index in range(expected_batches):
        array_path, json_path = _capture_batch_path(root, index)
        if array_path.exists() and json_path.exists():
            _load_capture_batch(root, index)
            completed += 1
        elif array_path.exists() or json_path.exists():
            raise FrozenArtifactError("E7 capture batch is partially persisted")
    budget = len(questions) if limit is None else limit
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise DataValidationError("E7 capture limit must be positive")
    runtime: VllmResearchRuntime | None = None
    attestor: E6RuntimeAttestor | None = None
    processed = 0
    groups = semantic_group_ids(questions)
    try:
        for index in range(expected_batches):
            array_path, _json_path = _capture_batch_path(root, index)
            if array_path.exists():
                continue
            start = index * runbook.capture_batch_size
            stop = min(start + runbook.capture_batch_size, len(questions))
            if processed + (stop - start) > budget:
                break
            if runtime is None:
                runtime, attestor = _native_runtime(runbook, context)
            assert attestor is not None
            batch_questions = tuple(questions[start:stop])
            group_ids = tuple(groups[item.question_id] for item in batch_questions)
            prompt = context.prompts[schema.prompt_id]
            if partition == "T-steer":
                batch = execute_e7_tsteer_activation_capture_batch(
                    attestor=attestor,
                    runtime_artifact=runbook.runtime_artifact,
                    questions=batch_questions,
                    prompt=prompt,
                    group_ids=group_ids,
                    feature_schema=schema,
                    source_question_bundle=source,
                    dtype="float16",
                    max_new_tokens=runbook.max_new_tokens,
                )
            else:
                batch = execute_e7_activation_capture_batch(
                    attestor=attestor,
                    runtime_artifact=runbook.runtime_artifact,
                    questions=batch_questions,
                    prompt=prompt,
                    outcomes=tuple(Outcome.UNSCORABLE for _ in batch_questions),
                    group_ids=group_ids,
                    feature_schema=schema,
                    source_question_bundle=source,
                    dtype="float16",
                )
            _write_capture_batch(root, index, batch)
            processed += len(batch_questions)
            completed += 1
            if processed >= budget:
                break
    finally:
        if runtime is not None:
            runtime.close()
    complete = completed == expected_batches
    if complete:
        _materialize_activation_corpus(runbook, context, partition, questions, source, schema, root)
        _materialize_separate_sae_root(runbook)
    return MappingProxyType(
        {
            "valid": True,
            "phase": "E7",
            "partition": partition,
            "processed_rows": processed,
            "completed_batches": completed,
            "expected_batches": expected_batches,
            "complete": complete,
        }
    )


def _probe_from_corpus(corpus: ActivationCorpus) -> ProbeDataset:
    identifiers: list[str] = []
    groups: list[str] = []
    outcomes: list[Outcome] = []
    features: list[torch.Tensor] = []
    for shard in corpus.iter_shards():
        identifiers.extend(shard.question_ids)
        groups.extend(shard.group_ids)
        outcomes.extend(shard.outcomes)
        features.append(torch.from_numpy(np.asarray(shard.activations, dtype=np.float32).copy()))
    return ProbeDataset(
        question_ids=tuple(identifiers),
        features=torch.cat(features),
        outcomes=tuple(outcomes),
        group_ids=tuple(groups),
        feature_schema=corpus.feature_schema,
    )


def _e5_geometry(
    runbook: E7Runbook, context: _E7BaseContext
) -> tuple[int, ActivationSite, TokenScope]:
    ledger = open_phase_prerequisite(
        runbook.prerequisite_runs[ExperimentPhase.E5],
        phase=ExperimentPhase.E5,
        study=context.study,
    )
    values = {
        (item.layer, item.site, item.token_scope)
        for item in ledger.contract.conditions
        if item.steering_method == "M1"
    }
    if len(values) != 1:
        raise FrozenArtifactError("E5 prerequisite lacks one promoted M1 geometry")
    layer, site, token_scope = next(iter(values))
    if (
        type(layer) is not int
        or not isinstance(site, ActivationSite)
        or not isinstance(token_scope, TokenScope)
        or layer != runbook.m1_tensor_index[3]
        or site.value != runbook.m1_tensor_index[2]
    ):
        raise FrozenArtifactError("E5 geometry differs from the E7 E3 tensor slice")
    return layer, site, token_scope


def _coordinate_plan(
    runbook: E7Runbook,
    context: _E7BaseContext,
    dataset: ProbeDataset,
) -> tuple[
    str,
    tuple[Question, ...],
    np.ndarray[Any, Any],
    torch.Tensor,
    float,
    str,
    int,
    ActivationSite,
    TokenScope,
]:
    if dataset.feature_schema is None:
        raise DataValidationError("E7 T-steer dataset lacks its feature schema")
    questions = context.final_questions["triviaqa"][: runbook.coordinate_question_count]
    if len(questions) != runbook.coordinate_question_count:
        raise DataValidationError("E7 coordinate screen question count is unavailable")
    dense, reference_rms, direction_sha = _e3_slice(
        runbook.e3_static_vectors, runbook.m1_tensor_index
    )
    correct = torch.tensor([item is Outcome.CORRECT for item in dataset.outcomes])
    incorrect = torch.tensor([item is Outcome.INCORRECT for item in dataset.outcomes])
    if not correct.any() or not incorrect.any():
        raise DataValidationError("E7 T-steer corpus lacks both correct and incorrect labels")
    from mfh.methods.sparse import standardized_effect_size

    effect = standardized_effect_size(dataset.features[correct], dataset.features[incorrect])
    layer, site, token_scope = _e5_geometry(runbook, context)
    dummy = tuple(
        CoordinateScreenPoint(
            retained_fraction=fraction,
            alpha=alpha,
            baseline_condition_id="0" * 64,
            intervention_condition_id="1" * 64,
            question_ids=tuple(item.question_id for item in questions),
            baseline_outcomes=tuple(Outcome.INCORRECT for _ in questions),
            intervention_outcomes=tuple(Outcome.INCORRECT for _ in questions),
        )
        for fraction in _FRACTIONS
        for alpha in _ALPHAS
    )
    digest = coordinate_screen_contract_digest(
        feature_schema=dataset.feature_schema,
        source_artifact_sha256=sha256_path(runbook.e3_static_vectors),
        source_tensor_index=runbook.m1_tensor_index,
        source_direction_sha256=direction_sha,
        layer=layer,
        site=site,
        token_scope=token_scope,
        runtime_artifact_sha256=context.runtime_artifact_sha256,
        execution_public_key=context.execution_public_key,
        points=dummy,
        source_question_bundle_sha256=sha256_path(
            runbook.outputs["development_side_effect_bundle"] / "questions"
        ),
    )
    return (
        digest,
        questions,
        dense,
        effect,
        reference_rms,
        direction_sha,
        layer,
        site,
        token_scope,
    )


def _coordinate_record_path(root: Path, condition_id: str, question_id: str) -> Path:
    return root / "rows" / f"{stable_hash([condition_id, question_id])}.json"


def _write_generation_record(path: Path, record: GenerationRecord) -> None:
    if path.exists():
        raise FrozenArtifactError(f"refusing to overwrite execution row: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    body = record.to_dict()
    payload = {"record": body, "record_digest": stable_hash(body)}
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    stage = Path(temporary)
    try:
        stage.write_text(
            json.dumps(payload, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(stage, path)
    finally:
        if stage.exists():
            stage.unlink()


def _load_generation_record(path: Path) -> GenerationRecord:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        body = payload["record"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read execution row: {exc}") from exc
    if payload.get("record_digest") != stable_hash(body):
        raise FrozenArtifactError("execution row digest differs")
    return GenerationRecord.from_dict(body)


def _draft_record(
    context: _E7BaseContext,
    *,
    question: Question,
    prompt: PromptSpec,
    rendered_prompt_hash: str,
    method: str,
    condition_id: str,
    layer: int | None,
    site: ActivationSite | None,
    token_scope: TokenScope | None,
    alpha: float,
    sparsity: float | None,
    partition: str = "T-dev",
    comparison_group: str = "development",
    method_artifact_sha256: str | None = None,
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
        steering_method=method,
        layer=layer,
        site=site,
        token_scope=token_scope,
        alpha=alpha,
        sparsity=sparsity,
        controller_scores={},
        raw_output="",
        normalized_answer="",
        outcome=Outcome.INCORRECT,
        generation_latency_seconds=0.0,
        input_tokens=0,
        output_tokens=0,
        condition_id=condition_id,
        seed=runbook_seed(context),
        metadata={
            "phase": "E7",
            "partition": partition,
            "prompt_template_sha256": hashlib.sha256(prompt.text.encode()).hexdigest(),
            "study_protocol_digest": context.study.digest,
            "comparison_group": comparison_group,
            **(
                {"method_artifact_sha256": method_artifact_sha256}
                if method_artifact_sha256 is not None
                else {}
            ),
        },
    )


def execute_e7_coordinate_screen(
    runbook: E7Runbook, *, limit: int | None = None
) -> Mapping[str, Any]:
    """Resume the registered 4x5 coordinate-sparsity screen and freeze its winner."""

    prepare_e7_runbook(runbook)
    context = _base_context(runbook)
    corpus = load_activation_corpus(runbook.outputs["tsteer_corpus"])
    dataset = _probe_from_corpus(corpus)
    (
        contract_digest,
        questions,
        dense,
        effect,
        reference_rms,
        direction_sha,
        layer,
        site,
        token_scope,
    ) = _coordinate_plan(runbook, context, dataset)
    root = runbook.outputs["coordinate_work"]
    manifest_body = {
        "schema_version": 1,
        "runbook_digest": runbook.runbook_digest,
        "contract_digest": contract_digest,
        "question_ids": [item.question_id for item in questions],
        "runtime_artifact_sha256": context.runtime_artifact_sha256,
        "execution_public_key": context.execution_public_key,
    }
    manifest = {**manifest_body, "manifest_digest": stable_hash(manifest_body)}
    if root.exists():
        try:
            observed = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FrozenArtifactError(f"cannot read E7 coordinate work: {exc}") from exc
        if observed != manifest:
            raise FrozenArtifactError("E7 coordinate resume plan differs")
    else:
        (root / "rows").mkdir(parents=True)
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    baseline_id = coordinate_screen_condition_id(contract_digest)
    cells = (
        (None, 0.0, baseline_id),
        *(
            (
                fraction,
                alpha,
                coordinate_screen_condition_id(
                    contract_digest, retained_fraction=fraction, alpha=alpha
                ),
            )
            for fraction in _FRACTIONS
            for alpha in _ALPHAS
        ),
    )
    budget = len(cells) * len(questions) if limit is None else limit
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise DataValidationError("E7 coordinate limit must be positive")
    runtime: VllmResearchRuntime | None = None
    attestor: E6RuntimeAttestor | None = None
    processed = 0
    completed = 0
    prompt = context.prompts[runbook.m1_tensor_index[0]]
    source_questions = runbook.outputs["development_side_effect_bundle"] / "questions"
    try:
        for fraction, alpha, condition_id in cells:
            sparse = (
                None
                if fraction is None
                else coordinate_sparse_direction(
                    torch.from_numpy(dense), effect, retained_fraction=fraction
                ).direction
            )
            for question in questions:
                path = _coordinate_record_path(root, condition_id, question.question_id)
                if path.exists():
                    _load_generation_record(path)
                    completed += 1
                    continue
                if processed >= budget:
                    break
                if runtime is None:
                    runtime, attestor = _native_runtime(runbook, context)
                assert attestor is not None
                rendered = runtime.render_prompt(prompt, question.text, metadata=question.metadata)
                intervened = fraction is not None
                draft = _draft_record(
                    context,
                    question=question,
                    prompt=prompt,
                    rendered_prompt_hash=rendered.sha256,
                    method="M4a" if intervened else "M0",
                    condition_id=condition_id,
                    layer=layer if intervened else None,
                    site=site if intervened else None,
                    token_scope=token_scope if intervened else None,
                    alpha=alpha if intervened else 0.0,
                    sparsity=fraction,
                    comparison_group="e7-coordinate-screen",
                )
                executed = execute_coordinate_screen_generation(
                    attestor=attestor,
                    runtime_artifact=runbook.runtime_artifact,
                    source_question_bundle=source_questions,
                    question=question,
                    prompt=prompt,
                    prompt_template_sha256=hashlib.sha256(prompt.text.encode()).hexdigest(),
                    generation_record=draft,
                    contract_digest=contract_digest,
                    source_artifact_sha256=sha256_path(runbook.e3_static_vectors),
                    retained_fraction=fraction,
                    direction=sparse,
                    reference_rms=reference_rms if intervened else None,
                    layer=layer if intervened else None,
                    site=site if intervened else None,
                    token_scope=token_scope if intervened else None,
                    alpha=alpha if intervened else 0.0,
                    max_new_tokens=runbook.max_new_tokens,
                    populate_generation=True,
                )
                _write_generation_record(path, executed)
                processed += 1
                completed += 1
            if processed >= budget:
                break
    finally:
        if runtime is not None:
            runtime.close()
    expected = len(cells) * len(questions)
    complete = completed == expected
    if complete and not runbook.outputs["coordinate_artifact"].exists():
        baseline_records = tuple(
            _load_generation_record(
                _coordinate_record_path(root, baseline_id, question.question_id)
            )
            for question in questions
        )
        points: list[CoordinateScreenPoint] = []
        records = list(baseline_records)
        for fraction in _FRACTIONS:
            for alpha in _ALPHAS:
                condition_id = coordinate_screen_condition_id(
                    contract_digest, retained_fraction=fraction, alpha=alpha
                )
                interventions = tuple(
                    _load_generation_record(
                        _coordinate_record_path(root, condition_id, question.question_id)
                    )
                    for question in questions
                )
                records.extend(interventions)
                points.append(
                    CoordinateScreenPoint(
                        retained_fraction=fraction,
                        alpha=alpha,
                        baseline_condition_id=baseline_id,
                        intervention_condition_id=condition_id,
                        question_ids=tuple(item.question_id for item in questions),
                        baseline_outcomes=tuple(item.outcome for item in baseline_records),
                        intervention_outcomes=tuple(item.outcome for item in interventions),
                    )
                )
        artifact = fit_coordinate_sparse_artifact(
            dataset,
            torch.from_numpy(dense),
            screen_points=points,
            screen_records=records,
            source_artifact_sha256=sha256_path(runbook.e3_static_vectors),
            source_tensor_index=runbook.m1_tensor_index,
            source_direction_sha256=direction_sha,
            reference_rms=reference_rms,
            layer=layer,
            site=site,
            token_scope=token_scope,
            screen_runtime_artifact_sha256=context.runtime_artifact_sha256,
            screen_execution_public_key=context.execution_public_key,
            screen_question_bundle_sha256=sha256_path(source_questions),
        )
        save_coordinate_sparse_artifact(runbook.outputs["coordinate_artifact"], artifact)
    if runbook.outputs["coordinate_artifact"].exists():
        load_coordinate_sparse_artifact(runbook.outputs["coordinate_artifact"])
    return MappingProxyType(
        {
            "valid": True,
            "phase": "E7",
            "processed_rows": processed,
            "completed_rows": completed,
            "expected_rows": expected,
            "complete": complete,
            "contract_digest": contract_digest,
        }
    )


@dataclass(frozen=True, slots=True)
class _SelectedSAE:
    training: SAETrainingResult
    seed_runs: tuple[SAETrainingResult, ...]
    latent_direction: Any
    stability_selections: tuple[Any, ...]
    aligned_stability: float
    sweep_results: SAECheckpointResultSequence
    receipt: LongComputationReceipt
    selected_index: int


def _execution_signer(private_hex: str) -> Callable[[Mapping[str, Any]], str]:
    from mfh.provenance import canonical_json

    private = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_hex))

    def sign(body: Mapping[str, Any]) -> str:
        return private.sign(canonical_json(body).encode()).hex()

    return sign


def _run_sae_sweep(runbook: E7Runbook, context: _E7BaseContext) -> Any:
    training, validation, _source_sha = validate_separate_sae_corpus(
        runbook.outputs["separate_sae_corpus"]
    )
    if any(config.input_width != training.feature_schema.width for config in runbook.sae_configs):
        raise DataValidationError("E7 SAE config width differs from captured activations")
    snapshot_sha = context.runtime_identity.get("snapshot_sha256")
    if type(snapshot_sha) is not str:
        raise FrozenArtifactError("E6 runtime identity lacks its snapshot SHA-256")
    return fit_e7_sae_sweep_measured(
        training,
        validation,
        runbook.sae_configs,
        package_lock=runbook.package_lock,
        model_snapshot_sha256=snapshot_sha,
        runtime_artifact_sha256=context.runtime_artifact_sha256,
        execution_public_key=context.execution_public_key,
        execution_signer=_execution_signer(context.execution_private_key),
        checkpoint_directory=runbook.outputs["sae_sweep"],
    )


def _selection_payload(runbook: E7Runbook, context: _E7BaseContext, sweep: Any) -> dict[str, Any]:
    dataset = _probe_from_corpus(load_activation_corpus(runbook.outputs["tsteer_corpus"]))
    grouped: dict[tuple[str, int | float, int], list[int]] = {}
    for index, config in enumerate(runbook.sae_configs):
        level: int | float = (
            config.top_k if config.sparsity is SAESparsity.TOP_K else config.l1_coefficient
        )
        grouped.setdefault((config.sparsity.value, level, config.expansion_factor), []).append(
            index
        )
    candidates: list[tuple[tuple[float, ...], dict[str, Any]]] = []
    for level_key, indices in grouped.items():
        if len(indices) < 2:
            continue
        runs = tuple(sweep.results[index] for index in indices)
        selections = []
        directions = []
        for training in runs:
            selection, oriented = _oriented_decoder_directions(
                training, dataset, feature_count=runbook.feature_count
            )
            selections.append(selection)
            directions.append(oriented)
        aligned = _aligned_stability(directions)
        minimum_fve = min(item.metrics.fraction_variance_explained for item in runs)
        maximum_mse = max(item.metrics.reconstruction_mse for item in runs)
        maximum_active = max(item.metrics.average_active_features for item in runs)
        active_limit = (
            float(runs[0].config.top_k)
            if runs[0].config.sparsity is SAESparsity.TOP_K
            else max(1.0, runs[0].config.resolved_latent_width * 0.10)
        )
        if (
            minimum_fve < 0.20
            or maximum_mse > 1.0
            or maximum_active > active_limit
            or aligned < 0.80
        ):
            continue
        primary = min(indices, key=lambda index: runbook.sae_configs[index].seed)
        payload = {
            "level": list(level_key),
            "seed_indices": indices,
            "selected_index": primary,
            "aligned_feature_stability": aligned,
            "stability_selections": [
                {
                    "seed": item.seed,
                    "checkpoint_fingerprint": item.checkpoint_fingerprint,
                    "selected_features": list(item.selected_features),
                }
                for item in selections
            ],
        }
        score = (
            minimum_fve,
            aligned,
            -maximum_mse,
            -maximum_active,
            -float(level_key[1]),
        )
        candidates.append((score, payload))
    if not candidates:
        raise DataValidationError(
            "E7 SAE sweep falsified the registered reconstruction/stability gate"
        )
    selected = max(candidates, key=lambda item: item[0])[1]
    body = {
        "schema_version": 1,
        "runbook_digest": runbook.runbook_digest,
        "tsteer_data_fingerprint": dataset.data_fingerprint,
        "sweep_plan_sha256": sha256_file(runbook.outputs["sae_sweep"] / "plan.json"),
        **selected,
    }
    return {**body, "selection_digest": stable_hash(body)}


def execute_e7_sae_sweep(runbook: E7Runbook) -> Mapping[str, Any]:
    """Run/resume the measured SAE grid and freeze its deterministic winner."""

    context = _base_context(runbook)
    sweep = _run_sae_sweep(runbook, context)
    selection = _selection_payload(runbook, context, sweep)
    selection_root = runbook.outputs["sae_selection"]
    selection_root.mkdir(parents=True, exist_ok=True)
    selection_path = selection_root / "selection.json"
    receipt_path = selection_root / "long-computation-receipt.json"
    if selection_path.exists():
        try:
            observed = json.loads(selection_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FrozenArtifactError(f"cannot read E7 SAE selection: {exc}") from exc
        if observed != selection:
            raise FrozenArtifactError("E7 SAE selection changed on replay")
    else:
        selection_path.write_text(
            json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    receipt_body = asdict(sweep.receipt)
    receipt = {"receipt": receipt_body, "receipt_digest": stable_hash(receipt_body)}
    if receipt_path.exists():
        try:
            observed_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FrozenArtifactError(f"cannot read E7 SAE computation receipt: {exc}") from exc
        observed_body = observed_receipt.get("receipt")
        if not isinstance(observed_body, Mapping) or observed_receipt.get(
            "receipt_digest"
        ) != stable_hash(observed_body):
            raise FrozenArtifactError("existing E7 SAE computation receipt is invalid")
        LongComputationReceipt(**observed_body)
    else:
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return MappingProxyType(
        {
            "valid": True,
            "phase": "E7",
            "checkpoint_count": len(sweep.results),
            "selected_index": selection["selected_index"],
            "aligned_feature_stability": selection["aligned_feature_stability"],
            "selection_digest": selection["selection_digest"],
        }
    )


def _selected_sae(runbook: E7Runbook, context: _E7BaseContext) -> _SelectedSAE:
    sweep = _run_sae_sweep(runbook, context)
    selection_path = runbook.outputs["sae_selection"] / "selection.json"
    receipt_path = runbook.outputs["sae_selection"] / "long-computation-receipt.json"
    try:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        receipt_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read E7 SAE promotion selection: {exc}") from exc
    recomputed = _selection_payload(runbook, context, sweep)
    receipt_body = receipt_payload.get("receipt")
    if (
        selection != recomputed
        or not isinstance(receipt_body, Mapping)
        or receipt_payload.get("receipt_digest") != stable_hash(receipt_body)
    ):
        raise FrozenArtifactError("E7 SAE selection or measured receipt differs")
    selected_index = int(selection["selected_index"])
    seed_indices = tuple(int(value) for value in selection["seed_indices"])
    primary = sweep.results[selected_index]
    seed_runs = tuple(sweep.results[index] for index in seed_indices)
    dataset = _probe_from_corpus(load_activation_corpus(runbook.outputs["tsteer_corpus"]))
    latent = latent_factuality_direction(
        primary.model, dataset, feature_count=runbook.feature_count
    )
    from mfh.methods.sparse import SeedFeatureSelection

    selections = tuple(
        SeedFeatureSelection(
            seed=int(value["seed"]),
            checkpoint_fingerprint=str(value["checkpoint_fingerprint"]),
            selected_features=tuple(int(item) for item in value["selected_features"]),
        )
        for value in selection["stability_selections"]
    )
    if (
        next(
            item
            for item in selections
            if item.checkpoint_fingerprint == sae_checkpoint_fingerprint(primary)
        ).selected_features
        != latent.selected_features
    ):
        raise FrozenArtifactError("E7 primary SAE selection differs from replay")
    return _SelectedSAE(
        training=primary,
        seed_runs=seed_runs,
        latent_direction=latent,
        stability_selections=selections,
        aligned_stability=float(selection["aligned_feature_stability"]),
        sweep_results=sweep.results,
        receipt=LongComputationReceipt(**receipt_body),
        selected_index=selected_index,
    )


_CAUSAL_BEHAVIOR_BENCHMARK = {
    "instruction_following": "ifeval",
    "safe_non_refusal": "xstest",
    "harmful_refusal": "strongreject_or_harmbench",
    "language_consistency": "language_consistency",
    "abstention_association": "triviaqa",
}


def _causal_questions(
    context: _E7BaseContext,
) -> tuple[tuple[Question, ...], Mapping[str, tuple[str, ...]]]:
    factual = context.final_questions["triviaqa"][:100]
    behavior_ids = {
        behavior: tuple(item.question_id for item in context.final_questions[benchmark][:50])
        for behavior, benchmark in _CAUSAL_BEHAVIOR_BENCHMARK.items()
    }
    by_key: dict[tuple[str, str], Question] = {}
    for question in factual:
        by_key[(question.benchmark, question.question_id)] = question
    for behavior, benchmark in _CAUSAL_BEHAVIOR_BENCHMARK.items():
        expected = set(behavior_ids[behavior])
        for question in context.final_questions[benchmark]:
            if question.question_id in expected:
                by_key[(question.benchmark, question.question_id)] = question
    questions = tuple(sorted(by_key.values(), key=lambda item: (item.benchmark, item.question_id)))
    if len(factual) != 100 or any(len(values) != 50 for values in behavior_ids.values()):
        raise DataValidationError("E7 causal cohorts do not meet registered minimums")
    return questions, MappingProxyType(behavior_ids)


def _causal_condition(
    runbook: E7Runbook,
    context: _E7BaseContext,
    *,
    benchmark: str,
    prompt: PromptSpec,
    feature: int,
    mode: str,
    checkpoint_sha: str,
    layer: int,
    site: ActivationSite,
    token_scope: TokenScope,
    alpha: float,
    sparsity: float,
) -> EvaluationCondition:
    active = mode != "baseline"
    return EvaluationCondition(
        phase=ExperimentPhase.E7,
        benchmark=benchmark,
        partition="T-dev",
        model_name=context.model.name,
        model_repository=context.model.repository,
        model_revision=context.model.revision,
        runtime=context.model.runtime,
        quantization=context.model.quantization,
        model_num_layers=context.model.num_layers,
        system_prompt_id=prompt.prompt_id,
        prompt_template_sha256=hashlib.sha256(prompt.text.encode()).hexdigest(),
        steering_method="M4b" if active else "M0",
        method_artifact_sha256=checkpoint_sha if active else None,
        layer=layer if active else None,
        site=site if active else None,
        token_scope=token_scope if active else None,
        alpha=alpha if active else 0.0,
        sparsity=sparsity if active else None,
        seed=runbook.seed,
        study_protocol_digest=context.study.digest,
        comparison_group=f"e7-causal-{feature}-{mode}",
    )


def _causal_row_path(runbook: E7Runbook, feature: int, mode: str, question: Question) -> Path:
    return (
        runbook.outputs["causal_work"]
        / "rows"
        / f"{feature:06d}-{mode}-{stable_hash([question.benchmark, question.question_id])}.json"
    )


def _bind_grader(
    grader: E7E8DevelopmentGrader, question: Question
) -> Callable[[GenerationRecord], GenerationRecord]:
    def grade(record: GenerationRecord) -> GenerationRecord:
        return grader(record, question)

    return grade


def execute_e7_causal_audit(runbook: E7Runbook, *, limit: int | None = None) -> Mapping[str, Any]:
    """Resume native activation/suppression tests for every selected SAE feature."""

    context = _base_context(runbook)
    selected = _selected_sae(runbook, context)
    coordinate = load_coordinate_sparse_artifact(runbook.outputs["coordinate_artifact"])
    questions, _behavior_ids = _causal_questions(context)
    prompt = context.prompts[selected.latent_direction.selection_schema.prompt_id]
    checkpoint_path = selected.sweep_results.directories[selected.selected_index]
    checkpoint_sha = sae_checkpoint_fingerprint(selected.training)
    sparsity = len(selected.latent_direction.selected_features) / float(
        selected.training.config.resolved_latent_width
    )
    modes = ("baseline", "activated", "suppressed")
    expected = len(selected.latent_direction.selected_features) * len(modes) * len(questions)
    budget = expected if limit is None else limit
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise DataValidationError("E7 causal limit must be positive")
    runtime: VllmResearchRuntime | None = None
    attestor: E6RuntimeAttestor | None = None
    grader: E7E8DevelopmentGrader | None = None
    processed = 0
    completed = 0
    source = runbook.outputs["development_side_effect_bundle"] / "questions"
    try:
        for feature in selected.latent_direction.selected_features:
            for mode in modes:
                for question in questions:
                    path = _causal_row_path(runbook, feature, mode, question)
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
                    condition = _causal_condition(
                        runbook,
                        context,
                        benchmark=question.benchmark,
                        prompt=prompt,
                        feature=feature,
                        mode=mode,
                        checkpoint_sha=checkpoint_sha,
                        layer=coordinate.layer,
                        site=coordinate.site,
                        token_scope=coordinate.token_scope,
                        alpha=coordinate.alpha,
                        sparsity=sparsity,
                    )
                    rendered = runtime.render_prompt(
                        prompt, question.text, metadata=question.metadata
                    )
                    draft = _draft_record(
                        context,
                        question=question,
                        prompt=prompt,
                        rendered_prompt_hash=rendered.sha256,
                        method=condition.steering_method,
                        condition_id=condition.condition_id,
                        layer=condition.layer,
                        site=condition.site,
                        token_scope=condition.token_scope,
                        alpha=condition.alpha,
                        sparsity=condition.sparsity,
                        comparison_group=condition.comparison_group,
                        method_artifact_sha256=condition.method_artifact_sha256,
                    )
                    executed = execute_e7_causal_feature_generation(
                        attestor=attestor,
                        runtime_artifact=runbook.runtime_artifact,
                        question=question,
                        prompt=prompt,
                        generation_record=draft,
                        condition=condition,
                        sae_training=checkpoint_path,
                        feature_schema=selected.latent_direction.selection_schema,
                        source_question_bundle=source,
                        feature_index=feature,
                        mode=mode,
                        layer=coordinate.layer,
                        site=coordinate.site,
                        token_scope=coordinate.token_scope,
                        alpha=coordinate.alpha,
                        reference_rms=coordinate.reference_rms,
                        max_new_tokens=runbook.max_new_tokens,
                        populate_generation=True,
                        generation_grader=_bind_grader(grader, question),
                    )
                    _write_generation_record(path, executed)
                    processed += 1
                    completed += 1
                if processed >= budget:
                    break
            if processed >= budget:
                break
    finally:
        if runtime is not None:
            runtime.close()
    complete = completed == expected
    if complete:
        _measure_causal_evidence(runbook, context, selected, coordinate)
    return MappingProxyType(
        {
            "valid": True,
            "phase": "E7",
            "processed_rows": processed,
            "completed_rows": completed,
            "expected_rows": expected,
            "complete": complete,
        }
    )


def _protected_boolean(behavior: str, record: GenerationRecord) -> bool:
    if behavior == "abstention_association":
        return record.outcome is Outcome.ABSTENTION
    if behavior == "instruction_following":
        value = record.metadata.get("official_pass")
    elif behavior == "safe_non_refusal":
        refused = record.metadata.get("refused")
        value = not refused if isinstance(refused, bool) else None
    elif behavior == "harmful_refusal":
        value = record.metadata.get("refused")
    elif behavior == "language_consistency":
        value = record.metadata.get("requested_language_correct")
    else:
        raise DataValidationError("E7 causal behavior is invalid")
    if not isinstance(value, bool):
        raise DataValidationError(f"E7 causal {behavior} score is absent")
    return value


def _measure_causal_evidence(
    runbook: E7Runbook,
    context: _E7BaseContext,
    selected: _SelectedSAE,
    coordinate: Any,
) -> tuple[Any, ...]:
    questions, behavior_ids = _causal_questions(context)
    factual_ids = {item.question_id for item in context.final_questions["triviaqa"][:100]}
    modes = ("baseline", "activated", "suppressed")
    results = []
    source_sha = sha256_path(runbook.outputs["development_side_effect_bundle"] / "questions")
    for feature in selected.latent_direction.selected_features:
        native = {
            mode: tuple(
                _load_generation_record(_causal_row_path(runbook, feature, mode, question))
                for question in questions
            )
            for mode in modes
        }
        outcome_maps = {
            mode: {
                record.question_id: record.outcome
                for record in native[mode]
                if record.question_id in factual_ids
            }
            for mode in modes
        }
        protected_maps = {
            mode: {
                behavior: {
                    record.question_id: _protected_boolean(behavior, record)
                    for record in native[mode]
                    if record.question_id in set(ids)
                    and record.benchmark == _CAUSAL_BEHAVIOR_BENCHMARK[behavior]
                }
                for behavior, ids in behavior_ids.items()
            }
            for mode in modes
        }
        results.append(
            measure_feature_intervention_evidence(
                feature,
                baseline_outcomes=outcome_maps["baseline"],
                activated_outcomes=outcome_maps["activated"],
                suppressed_outcomes=outcome_maps["suppressed"],
                protected_baseline=protected_maps["baseline"],
                protected_activated=protected_maps["activated"],
                protected_suppressed=protected_maps["suppressed"],
                feature_schema=selected.latent_direction.selection_schema,
                alpha=coordinate.alpha,
                token_scope=coordinate.token_scope,
                layer=coordinate.layer,
                site=coordinate.site,
                runtime_artifact_sha256=context.runtime_artifact_sha256,
                execution_public_key=context.execution_public_key,
                source_question_bundle_sha256=source_sha,
                execution_signer=_execution_signer(context.execution_private_key),
                native_execution_records=native,
            )
        )
    return tuple(results)


_INTERPRETABILITY_AUDITS = (
    "P0-neutral",
    "P2-calibrated-abstention",
    "negative_alpha",
    "label_shuffled",
    "matched_random",
    "unrelated_layer",
    "gaussian",
    "zero_hook",
    "different_prompt",
)


def _interpretability_prompt_id(audit_name: str, selection_prompt_id: str) -> str:
    if audit_name in _PROMPTS:
        return audit_name
    if audit_name == "different_prompt":
        return "P2-calibrated-abstention" if selection_prompt_id == "P0-neutral" else "P0-neutral"
    return selection_prompt_id


def _interpretability_condition(
    runbook: E7Runbook,
    context: _E7BaseContext,
    *,
    feature: int,
    audit_name: str,
    mode: str,
    prompt: PromptSpec,
    checkpoint_sha: str,
    layer: int,
    site: ActivationSite,
    token_scope: TokenScope,
    alpha: float,
) -> EvaluationCondition:
    active = mode == "intervention" and audit_name != "zero_hook"
    selected_layer = (
        (layer + 1) % context.model.num_layers if audit_name == "unrelated_layer" else layer
    )
    selected_alpha = -abs(alpha) if audit_name == "negative_alpha" else alpha
    return EvaluationCondition(
        phase=ExperimentPhase.E7,
        benchmark="triviaqa",
        partition="T-dev",
        model_name=context.model.name,
        model_repository=context.model.repository,
        model_revision=context.model.revision,
        runtime=context.model.runtime,
        quantization=context.model.quantization,
        model_num_layers=context.model.num_layers,
        system_prompt_id=prompt.prompt_id,
        prompt_template_sha256=hashlib.sha256(prompt.text.encode()).hexdigest(),
        steering_method="M4b" if active else "M0",
        method_artifact_sha256=checkpoint_sha if active else None,
        layer=selected_layer if active else None,
        site=site if active else None,
        token_scope=token_scope if active else None,
        alpha=selected_alpha if active else 0.0,
        sparsity=None,
        seed=runbook.seed,
        study_protocol_digest=context.study.digest,
        comparison_group=f"e7-interpretability-{feature}-{audit_name}-{mode}",
    )


def _interpretability_row_path(
    runbook: E7Runbook,
    feature: int,
    audit_name: str,
    mode: str,
    question_id: str,
) -> Path:
    return (
        runbook.outputs["interpretability_work"]
        / "rows"
        / (f"{feature:06d}-{audit_name}-{mode}-{stable_hash(question_id)}.json")
    )


def _top_activating_ids(
    validation: ActivationCorpus,
    training: SAETrainingResult,
    features: Sequence[int],
) -> Mapping[int, tuple[str, ...]]:
    ranked: dict[int, list[tuple[float, str]]] = {feature: [] for feature in features}
    with torch.no_grad():
        for shard in validation.iter_shards():
            values = torch.from_numpy(np.asarray(shard.activations, dtype=np.float32).copy())
            latents = training.model.encode(values)
            for row, question_id in enumerate(shard.question_ids):
                for feature in features:
                    ranked[feature].append((float(latents[row, feature]), question_id))
    return MappingProxyType(
        {
            feature: tuple(
                question_id
                for _value, question_id in sorted(values, key=lambda item: (-item[0], item[1]))[:10]
            )
            for feature, values in ranked.items()
        }
    )


def execute_e7_interpretability_audit(
    runbook: E7Runbook, *, limit: int | None = None
) -> Mapping[str, Any]:
    """Resume prompt-transfer and all seven registered negative-control pairs."""

    context = _base_context(runbook)
    selected = _selected_sae(runbook, context)
    coordinate = load_coordinate_sparse_artifact(runbook.outputs["coordinate_artifact"])
    selection = _probe_from_corpus(load_activation_corpus(runbook.outputs["tsteer_corpus"]))
    questions = context.final_questions["triviaqa"][: runbook.interpretability_question_count]
    checkpoint_path = selected.sweep_results.directories[selected.selected_index]
    checkpoint_sha = sae_checkpoint_fingerprint(selected.training)
    expected = (
        len(selected.latent_direction.selected_features)
        * len(_INTERPRETABILITY_AUDITS)
        * 2
        * len(questions)
    )
    budget = expected if limit is None else limit
    if (
        len(questions) != runbook.interpretability_question_count
        or isinstance(budget, bool)
        or not isinstance(budget, int)
        or budget <= 0
    ):
        raise DataValidationError("E7 interpretability schedule or limit is invalid")
    runtime: VllmResearchRuntime | None = None
    attestor: E6RuntimeAttestor | None = None
    grader: E7E8DevelopmentGrader | None = None
    processed = 0
    completed = 0
    source = runbook.outputs["development_side_effect_bundle"] / "questions"
    try:
        for feature in selected.latent_direction.selected_features:
            for audit_name in _INTERPRETABILITY_AUDITS:
                prompt_id = _interpretability_prompt_id(
                    audit_name,
                    selection.feature_schema.prompt_id
                    if selection.feature_schema is not None
                    else "",
                )
                prompt = context.prompts[prompt_id]
                for mode in ("baseline", "intervention"):
                    condition = _interpretability_condition(
                        runbook,
                        context,
                        feature=feature,
                        audit_name=audit_name,
                        mode=mode,
                        prompt=prompt,
                        checkpoint_sha=checkpoint_sha,
                        layer=coordinate.layer,
                        site=coordinate.site,
                        token_scope=coordinate.token_scope,
                        alpha=coordinate.alpha,
                    )
                    for question in questions:
                        path = _interpretability_row_path(
                            runbook, feature, audit_name, mode, question.question_id
                        )
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
                        rendered = runtime.render_prompt(
                            prompt, question.text, metadata=question.metadata
                        )
                        draft = _draft_record(
                            context,
                            question=question,
                            prompt=prompt,
                            rendered_prompt_hash=rendered.sha256,
                            method=condition.steering_method,
                            condition_id=condition.condition_id,
                            layer=condition.layer,
                            site=condition.site,
                            token_scope=condition.token_scope,
                            alpha=condition.alpha,
                            sparsity=condition.sparsity,
                            comparison_group=condition.comparison_group,
                            method_artifact_sha256=condition.method_artifact_sha256,
                        )
                        executed = execute_e7_interpretability_generation(
                            attestor=attestor,
                            runtime_artifact=runbook.runtime_artifact,
                            question=question,
                            prompt=prompt,
                            generation_record=draft,
                            condition=condition,
                            sae_training=checkpoint_path,
                            selection=selection,
                            source_question_bundle=source,
                            feature_index=feature,
                            audit_name=audit_name,
                            mode=mode,
                            layer=coordinate.layer,
                            site=coordinate.site,
                            token_scope=coordinate.token_scope,
                            alpha=coordinate.alpha,
                            reference_rms=coordinate.reference_rms,
                            max_new_tokens=runbook.max_new_tokens,
                            populate_generation=True,
                            generation_grader=_bind_grader(grader, question),
                        )
                        _write_generation_record(path, executed)
                        processed += 1
                        completed += 1
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
    if complete:
        _build_interpretability_audit(runbook, context, selected)
    return MappingProxyType(
        {
            "valid": True,
            "phase": "E7",
            "processed_rows": processed,
            "completed_rows": completed,
            "expected_rows": expected,
            "complete": complete,
        }
    )


def _paired_audit(
    baseline: tuple[GenerationRecord, ...],
    intervention: tuple[GenerationRecord, ...],
) -> SAEPairedExecutionAudit:
    effect = (
        sum(item.outcome is Outcome.CORRECT for item in intervention)
        - sum(item.outcome is Outcome.CORRECT for item in baseline)
    ) / len(baseline)
    return SAEPairedExecutionAudit(baseline, intervention, effect)


def _build_interpretability_audit(
    runbook: E7Runbook,
    context: _E7BaseContext,
    selected: _SelectedSAE,
) -> SAEInterpretabilityAudit:
    _training, validation, _sha = validate_separate_sae_corpus(
        runbook.outputs["separate_sae_corpus"]
    )
    questions = context.final_questions["triviaqa"][: runbook.interpretability_question_count]
    top = _top_activating_ids(
        validation,
        selected.training,
        selected.latent_direction.selected_features,
    )
    transfers: dict[int, dict[str, SAEPairedExecutionAudit]] = {}
    controls: dict[int, dict[str, SAEPairedExecutionAudit]] = {}
    for feature in selected.latent_direction.selected_features:
        transfers[feature] = {}
        controls[feature] = {}
        for audit_name in _INTERPRETABILITY_AUDITS:
            baseline = tuple(
                _load_generation_record(
                    _interpretability_row_path(
                        runbook,
                        feature,
                        audit_name,
                        "baseline",
                        question.question_id,
                    )
                )
                for question in questions
            )
            intervention = tuple(
                _load_generation_record(
                    _interpretability_row_path(
                        runbook,
                        feature,
                        audit_name,
                        "intervention",
                        question.question_id,
                    )
                )
                for question in questions
            )
            audit = _paired_audit(baseline, intervention)
            target = transfers if audit_name in _PROMPTS else controls
            target[feature][audit_name] = audit
    return SAEInterpretabilityAudit(
        top_activating_question_ids=top,
        evaluation_question_ids=tuple(item.question_id for item in questions),
        control_seed=runbook.seed,
        prompt_transfer_effects={
            feature: {name: audit.factuality_delta for name, audit in values.items()}
            for feature, values in transfers.items()
        },
        negative_control_effects={
            feature: {name: audit.factuality_delta for name, audit in values.items()}
            for feature, values in controls.items()
        },
        source_question_bundle_sha256=sha256_path(
            runbook.outputs["development_side_effect_bundle"] / "questions"
        ),
        prompt_transfer_execution=transfers,
        negative_control_execution=controls,
    )


def promote_e7_sae(runbook: E7Runbook) -> Mapping[str, Any]:
    """Apply all preregistered promotion gates and freeze M4b plus seed evidence."""

    context = _base_context(runbook)
    selected = _selected_sae(runbook, context)
    coordinate = load_coordinate_sparse_artifact(runbook.outputs["coordinate_artifact"])
    evidence = _measure_causal_evidence(runbook, context, selected, coordinate)
    audit = _build_interpretability_audit(runbook, context, selected)
    criteria = SAEPromotionCriteria(
        minimum_fve=0.20,
        maximum_reconstruction_mse=1.0,
        maximum_average_active_features=(
            float(selected.training.config.top_k)
            if selected.training.config.sparsity is SAESparsity.TOP_K
            else max(1.0, selected.training.config.resolved_latent_width * 0.10)
        ),
        minimum_feature_stability=0.80,
        minimum_causal_effect=0.02,
        maximum_protected_effect=0.02,
    )
    checkpoint_sha = sae_checkpoint_fingerprint(selected.training)
    sweep_points = tuple(
        SAESparsitySweepPoint(
            config_fingerprint=sae_config_fingerprint(result.config),
            checkpoint_fingerprint=sae_checkpoint_fingerprint(result),
            fraction_variance_explained=result.metrics.fraction_variance_explained,
            reconstruction_mse=result.metrics.reconstruction_mse,
            average_active_features=result.metrics.average_active_features,
            selected=sae_checkpoint_fingerprint(result) == checkpoint_sha,
        )
        for result in selected.sweep_results
    )
    artifact = promote_sae_intervention(
        selected.training,
        selected.latent_direction,
        evidence=evidence,
        stability_selections=selected.stability_selections,
        aligned_feature_stability=selected.aligned_stability,
        criteria=criteria,
        sparsity_sweep=sweep_points,
        sparsity_sweep_results=selected.sweep_results,
        interpretability_audit=audit,
        long_computation_receipt=selected.receipt,
    )
    destination = runbook.outputs["sae_intervention"]
    if destination.exists():
        existing = load_sae_intervention(destination)
        if existing.feature_stability != artifact.feature_stability:
            raise FrozenArtifactError("existing E7 SAE intervention differs")
    else:
        save_sae_intervention(destination, artifact)
    sae_sha = sha256_path(destination)
    stability_path = runbook.outputs["sae_stability_bundle"]
    if stability_path.exists():
        stability = load_sae_stability_bundle(stability_path)
        if stability.promoted_method_artifacts.get(context.model.repository) != sae_sha:
            raise FrozenArtifactError("existing E7 stability bundle differs")
    else:
        write_sae_stability_bundle(
            stability_path,
            runs_by_model={context.model.repository: selected.seed_runs},
            selection_corpora={context.model.repository: runbook.outputs["tsteer_corpus"]},
            selection_question_sources={
                context.model.repository: runbook.reviewed_splits / "T-steer.jsonl"
            },
            feature_count=runbook.feature_count,
            promoted_method_artifacts={context.model.repository: sae_sha},
        )
        stability = load_sae_stability_bundle(stability_path)
    if (
        not math.isclose(
            stability.stability_by_model[context.model.repository],
            artifact.feature_stability,
            rel_tol=0,
            abs_tol=1e-12,
        )
        or stability.selections_by_model[context.model.repository] != artifact.stability_selections
    ):
        raise FrozenArtifactError("E7 promoted SAE differs from recomputed seed stability")
    return MappingProxyType(
        {
            "valid": True,
            "phase": "E7",
            "sae_intervention_sha256": sae_sha,
            "sae_stability_bundle_sha256": sha256_path(stability_path),
            "aligned_feature_stability": artifact.feature_stability,
            "selected_features": list(artifact.latent_direction.selected_features),
        }
    )


def _final_contract(
    runbook: E7Runbook,
    context: _E7BaseContext,
    *,
    side_effect_bundle_sha256: str,
) -> PhaseRunContract:
    coordinate_path = runbook.outputs["coordinate_artifact"]
    sae_path = runbook.outputs["sae_intervention"]
    coordinate = load_coordinate_sparse_artifact(coordinate_path)
    sae = load_sae_intervention(sae_path)
    if not sae.evidence:
        raise FrozenArtifactError("E7 promoted SAE lacks causal geometry")
    geometry = sae.evidence[0].spec
    coordinate_sha = sha256_path(coordinate_path)
    sae_sha = sha256_path(sae_path)
    sae_sparsity = len(sae.latent_direction.selected_features) / float(
        sae.training.config.resolved_latent_width
    )
    conditions: list[EvaluationCondition] = []
    for benchmark in _QUESTION_COUNTS:
        for prompt_id in _PROMPTS:
            prompt = context.prompts[prompt_id]
            conditions.extend(
                (
                    _evaluation_condition(
                        context,
                        benchmark=benchmark,
                        prompt=prompt,
                        method="M0",
                    ),
                    _evaluation_condition(
                        context,
                        benchmark=benchmark,
                        prompt=prompt,
                        method="M4a",
                        method_artifact_sha256=coordinate_sha,
                        layer=coordinate.layer,
                        site=coordinate.site,
                        token_scope=coordinate.token_scope,
                        alpha=coordinate.alpha,
                        sparsity=coordinate.sparse_direction.retained_fraction,
                    ),
                    _evaluation_condition(
                        context,
                        benchmark=benchmark,
                        prompt=prompt,
                        method="M4b",
                        method_artifact_sha256=sae_sha,
                        layer=geometry.layer,
                        site=geometry.site,
                        token_scope=geometry.token_scope,
                        alpha=geometry.alpha,
                        sparsity=sae_sparsity,
                    ),
                )
            )
    contract = PhaseRunContract(
        phase=ExperimentPhase.E7,
        study_protocol_digest=context.study.digest,
        conditions=tuple(conditions),
        question_ids_by_benchmark={
            benchmark: tuple(item.question_id for item in values)
            for benchmark, values in context.final_questions.items()
        },
        input_fingerprints={
            "E3_static_vectors": sha256_path(runbook.e3_static_vectors),
            "E5_adaptive_controllers": sha256_path(runbook.e5_adaptive_controllers),
            "separate_sae_corpus": sha256_path(runbook.outputs["separate_sae_corpus"]),
            "frozen_sae_seed_runs": sha256_path(runbook.outputs["sae_stability_bundle"]),
            "frozen_tsteer_questions": sha256_file(runbook.reviewed_splits / "T-steer.jsonl"),
            "frozen_side_effect_scorers": side_effect_bundle_sha256,
        },
        prerequisite_digests=_completion_digests(runbook, context),
        required_gates=context.study.phase(ExperimentPhase.E7).gates,
    )
    contract.assert_matches_study(context.study)
    return contract


def _freeze_final_side_effect_bundle(
    runbook: E7Runbook, context: _E7BaseContext
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
            context.final_questions,
            source_artifacts=_source_artifacts(runbook),
            scorer_execution_public_key=context.execution_public_key,
            ifeval_evaluator=runbook.ifeval_evaluator,
        )
        contract = _final_contract(runbook, context, side_effect_bundle_sha256=side_sha)
        validate_side_effect_evaluation_bundle(destination, contract)
    development_questions_sha = sha256_path(
        runbook.outputs["development_side_effect_bundle"] / "questions"
    )
    final_questions_sha = sha256_path(destination / "questions")
    if development_questions_sha != final_questions_sha:
        raise FrozenArtifactError("E7 promoted schedule changed the causal evaluation questions")
    return contract, side_sha


def prepare_e7_ledger(runbook: E7Runbook) -> PhaseRunLedger:
    """Freeze the promoted method matrix and create/reopen the exact E7 ledger."""

    context = _base_context(runbook)
    contract, _side_sha = _freeze_final_side_effect_bundle(runbook, context)
    directory = runbook.outputs["run_directory"]
    if directory.exists():
        ledger = PhaseRunLedger.open(directory, study=context.study)
        if ledger.contract != contract:
            raise FrozenArtifactError("existing E7 ledger differs from promoted runbook")
        return ledger
    ledger = PhaseRunLedger.create(
        directory,
        contract,
        study=context.study,
        input_artifacts={
            "E3_static_vectors": runbook.e3_static_vectors,
            "E5_adaptive_controllers": runbook.e5_adaptive_controllers,
            "separate_sae_corpus": runbook.outputs["separate_sae_corpus"],
            "frozen_sae_seed_runs": runbook.outputs["sae_stability_bundle"],
            "frozen_tsteer_questions": runbook.reviewed_splits / "T-steer.jsonl",
            "frozen_side_effect_scorers": runbook.outputs["side_effect_bundle"],
        },
        prerequisite_runs={phase: path for phase, path in runbook.prerequisite_runs.items()},
    )
    runbook.outputs["final_work"].mkdir(parents=True, exist_ok=True)
    return ledger


def _final_row_path(runbook: E7Runbook, condition_id: str, question_id: str) -> Path:
    return runbook.outputs["final_work"] / f"{stable_hash([condition_id, question_id])}.json"


def execute_e7_final(runbook: E7Runbook, *, limit: int | None = None) -> Mapping[str, Any]:
    """Resume the exact promoted 39,624-row E7 development matrix."""

    context = _base_context(runbook)
    ledger = prepare_e7_ledger(runbook)
    budget = ledger.contract.expected_record_count if limit is None else limit
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise DataValidationError("E7 final execution limit must be positive")
    questions = {
        (item.benchmark, item.question_id): item
        for values in context.final_questions.values()
        for item in values
    }
    runtime: VllmResearchRuntime | None = None
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
                executed = _load_generation_record(path)
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
                rendered = runtime.render_prompt(prompt, question.text, metadata=question.metadata)
                draft = _draft_record(
                    context,
                    question=question,
                    prompt=prompt,
                    rendered_prompt_hash=rendered.sha256,
                    method=condition.steering_method,
                    condition_id=condition.condition_id,
                    layer=condition.layer,
                    site=condition.site,
                    token_scope=condition.token_scope,
                    alpha=condition.alpha,
                    sparsity=condition.sparsity,
                    comparison_group=condition.comparison_group,
                    method_artifact_sha256=condition.method_artifact_sha256,
                )
                executed = execute_e7_generation(
                    attestor=attestor,
                    runtime_artifact=runbook.runtime_artifact,
                    source_question_bundle=(runbook.outputs["side_effect_bundle"] / "questions"),
                    question=question,
                    prompt=prompt,
                    generation_record=draft,
                    condition=condition,
                    coordinate_artifact=runbook.outputs["coordinate_artifact"],
                    sae_intervention=runbook.outputs["sae_intervention"],
                    max_new_tokens=runbook.max_new_tokens,
                    populate_generation=True,
                    generation_grader=_bind_grader(grader, question),
                )
                _write_generation_record(path, executed)
            ledger.checkpoint((executed,))
            processed += 1
    finally:
        if runtime is not None:
            runtime.close()
    completed, expected = ledger.progress()
    return MappingProxyType(
        {
            "valid": True,
            "phase": "E7",
            "processed_records": processed,
            "completed_records": completed,
            "expected_records": expected,
            "remaining_records": expected - completed,
            "contract_digest": ledger.contract.digest,
        }
    )


def finalize_e7_runbook(runbook: E7Runbook) -> Mapping[str, Any]:
    """Derive all E7 gates and publish the self-contained terminal artifact."""

    context = _base_context(runbook)
    ledger = PhaseRunLedger.open(runbook.outputs["run_directory"], study=context.study)
    completed, expected = ledger.progress()
    if completed != expected:
        raise DataValidationError("E7 finalization requires a complete ledger")
    return finalize_e7_phase(
        runbook.outputs["final_directory"],
        ledger_directory=ledger.directory,
        study=context.study,
        coordinate_artifact=runbook.outputs["coordinate_artifact"],
        sae_intervention=runbook.outputs["sae_intervention"],
    )


def verify_e7_runbook(runbook: E7Runbook) -> Mapping[str, Any]:
    """Replay available E7 stages and the terminal package without loading VLLM."""

    context = _base_context(runbook)
    selected_feature_count = 0
    selection_path = runbook.outputs["sae_selection"] / "selection.json"
    if selection_path.is_file():
        try:
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            selections = selection["stability_selections"]
            selected_feature_count = len(selections[0]["selected_features"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, IndexError) as exc:
            raise FrozenArtifactError(f"cannot replay E7 selected feature count: {exc}") from exc
    causal_rows = len(tuple((runbook.outputs["causal_work"] / "rows").glob("*.json")))
    interpretability_rows = len(
        tuple((runbook.outputs["interpretability_work"] / "rows").glob("*.json"))
    )
    causal_question_count = len(_causal_questions(context)[0])
    causal_expected = selected_feature_count * 3 * causal_question_count
    interpretability_expected = (
        selected_feature_count
        * len(_INTERPRETABILITY_AUDITS)
        * 2
        * runbook.interpretability_question_count
    )
    stage_status = {
        "prepared": runbook.outputs["development_side_effect_bundle"].exists(),
        "tsteer_capture": runbook.outputs["tsteer_corpus"].exists(),
        "sae_corpus": runbook.outputs["separate_sae_corpus"].exists(),
        "coordinate": runbook.outputs["coordinate_artifact"].exists(),
        "sae_selection": (runbook.outputs["sae_selection"] / "selection.json").exists(),
        "causal": {
            "complete": causal_expected > 0 and causal_rows == causal_expected,
            "completed_rows": causal_rows,
            "expected_rows": causal_expected,
        },
        "interpretability": {
            "complete": interpretability_expected > 0
            and interpretability_rows == interpretability_expected,
            "completed_rows": interpretability_rows,
            "expected_rows": interpretability_expected,
        },
        "sae_promoted": runbook.outputs["sae_intervention"].exists(),
        "ledger": runbook.outputs["run_directory"].exists(),
        "terminal": runbook.outputs["final_directory"].exists(),
    }
    completed = 0
    expected = 0
    contract_digest: str | None = None
    if stage_status["ledger"]:
        ledger = PhaseRunLedger.open(runbook.outputs["run_directory"], study=context.study)
        completed, expected = ledger.progress()
        contract_digest = ledger.contract.digest
    terminal: Mapping[str, Any] | None = None
    if stage_status["terminal"]:
        terminal = verify_e7_phase(runbook.outputs["final_directory"])
    return MappingProxyType(
        {
            "valid": True,
            "phase": "E7",
            "runbook_digest": runbook.runbook_digest,
            "contract_digest": contract_digest,
            "completed_records": completed,
            "expected_records": expected,
            "remaining_records": expected - completed,
            "stages": stage_status,
            "terminal": dict(terminal) if terminal is not None else None,
        }
    )

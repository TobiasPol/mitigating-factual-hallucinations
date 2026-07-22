"""Resumable, provenance-bound native-VLLM operator lifecycle for E6."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
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
    AdaptivePolicySpec,
    GenerationRecord,
    ModelSpec,
    Outcome,
    PromptSpec,
    Question,
    TokenScope,
)
from mfh.data.io import read_questions
from mfh.data.reviewed_splits import validate_reviewed_split_snapshot
from mfh.errors import ConfigurationError, DataValidationError, FrozenArtifactError
from mfh.experiments.e5_adaptive import validate_e5_selected_controller_bundle
from mfh.experiments.e6_grading import (
    E6FactualGrader,
    E6OfficialGraderBundle,
    load_e6_official_grader_bundle,
    verify_e6_factual_grade,
)
from mfh.experiments.e6_likelihood import (
    E6ExecutedRow,
    E6RuntimeAttestor,
    E6VerifiedLikelihoodRecord,
    _e3_direction_index,
    _load_e6_runtime_attestation,
    e6_e3_slice_digest,
    execute_and_bind_e6_likelihood,
    finalize_e6_phase,
    verify_e6_bound_record,
    verify_e6_likelihood_bundle,
    verify_e6_phase,
    write_e6_likelihood_bundle,
)
from mfh.experiments.e8_protected import execute_e6_adaptive_generation
from mfh.experiments.model_selection import (
    validate_active_model_spec,
    validate_active_study_artifact_paths,
)
from mfh.experiments.protocol import (
    ExperimentPhase,
    StudyProtocol,
    load_study_protocol,
)
from mfh.experiments.runner import (
    EvaluationCondition,
    PhaseRunContract,
    PhaseRunLedger,
    _validate_question_bundle,
    open_phase_prerequisite,
    write_frozen_question_bundle,
)
from mfh.inference.transformers_snapshot import verify_transformers_snapshot
from mfh.inference.vllm_research import (
    VllmResearchInterventionState,
    VllmResearchRuntime,
)
from mfh.methods.adaptive import AdaptiveController
from mfh.provenance import sha256_file, sha256_path, stable_hash

_PROMPTS = ("P0-neutral", "P2-calibrated-abstention", "P3-forced-answer")
_METHODS = ("M0", "M1", "M3")
_QUESTION_COUNTS = {
    "triviaqa": 5_000,
    "simpleqa_verified": 1_000,
    "aa_omniscience_public_600": 600,
}
_PARTITIONS = {
    "triviaqa": "T-dev",
    "simpleqa_verified": "simpleqa-eval",
    "aa_omniscience_public_600": "aa-eval",
}
_REVIEWED_QUESTION_FILES = {
    "triviaqa": "T-dev.jsonl",
    "simpleqa_verified": "simpleqa-eval.jsonl",
    "aa_omniscience_public_600": "aa-eval.jsonl",
}
_TIMINGS = {
    "final_prompt": TokenScope.FINAL_PROMPT,
    "first_generated": TokenScope.FIRST_GENERATED,
    "first_four_generated": TokenScope.FIRST_FOUR,
}
_RUNBOOK_KEYS = {
    "schema_version",
    "phase",
    "study_protocol",
    "model_config",
    "prompt_config",
    "snapshot_directory",
    "snapshot_manifest",
    "frozen_question_bundle",
    "official_grader_bundle",
    "expected_grader_manifest_digest",
    "environment_file",
    "e3_static_vectors",
    "e5_adaptive_controllers",
    "runtime_artifact",
    "execution_key_file",
    "run_directory",
    "work_directory",
    "likelihood_directory",
    "final_directory",
    "prerequisite_runs",
    "seed",
    "max_new_tokens",
    "m1_tensor_index",
    "controller_source_prompt_id",
}


def _path(root: Path, value: object, context: str) -> Path:
    if type(value) is not str or not value.strip():
        raise DataValidationError(f"E6 runbook {context} path is invalid")
    raw = Path(value)
    return (raw if raw.is_absolute() else root / raw).resolve()


@dataclass(frozen=True, slots=True)
class E6Runbook:
    """All paths and frozen source choices needed for one E6 lifecycle."""

    source: Path
    study_protocol: Path
    model_config: Path
    prompt_config: Path
    snapshot_directory: Path
    snapshot_manifest: Path
    frozen_question_bundle: Path
    official_grader_bundle: Path
    expected_grader_manifest_digest: str
    environment_file: Path
    e3_static_vectors: Path
    e5_adaptive_controllers: Path
    runtime_artifact: Path
    execution_key_file: Path
    run_directory: Path
    work_directory: Path
    likelihood_directory: Path
    final_directory: Path
    prerequisite_runs: Mapping[ExperimentPhase, Path]
    seed: int
    max_new_tokens: int
    m1_tensor_index: tuple[str, str, str, int]
    controller_source_prompt_id: str
    runbook_digest: str

    @classmethod
    def load(cls, path: str | Path) -> E6Runbook:
        source = Path(path).resolve()
        try:
            value = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FrozenArtifactError(f"cannot read E6 runbook: {exc}") from exc
        if not isinstance(value, dict) or set(value) != _RUNBOOK_KEYS:
            raise DataValidationError("E6 runbook keys differ from schema version 1")
        prerequisites = value["prerequisite_runs"]
        tensor_index = value["m1_tensor_index"]
        if (
            value["schema_version"] != 1
            or value["phase"] != ExperimentPhase.E6.value
            or not isinstance(prerequisites, Mapping)
            or set(prerequisites) != {"E3", "E5"}
            or type(tensor_index) is not list
            or len(tensor_index) != 4
            or any(type(item) is not str for item in tensor_index[:3])
            or type(tensor_index[3]) is not int
            or type(value["seed"]) is not int
            or isinstance(value["seed"], bool)
            or value["seed"] < 0
            or type(value["max_new_tokens"]) is not int
            or isinstance(value["max_new_tokens"], bool)
            or not 32 <= value["max_new_tokens"] <= 48
            or value["controller_source_prompt_id"]
            not in {"P0-neutral", "P2-calibrated-abstention"}
            or type(value["expected_grader_manifest_digest"]) is not str
            or len(value["expected_grader_manifest_digest"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in value["expected_grader_manifest_digest"]
            )
        ):
            raise DataValidationError("E6 runbook values are invalid")
        root = source.parent
        return cls(
            source=source,
            study_protocol=_path(root, value["study_protocol"], "study_protocol"),
            model_config=_path(root, value["model_config"], "model_config"),
            prompt_config=_path(root, value["prompt_config"], "prompt_config"),
            snapshot_directory=_path(root, value["snapshot_directory"], "snapshot_directory"),
            snapshot_manifest=_path(root, value["snapshot_manifest"], "snapshot_manifest"),
            frozen_question_bundle=_path(
                root, value["frozen_question_bundle"], "frozen_question_bundle"
            ),
            official_grader_bundle=_path(
                root, value["official_grader_bundle"], "official_grader_bundle"
            ),
            expected_grader_manifest_digest=value["expected_grader_manifest_digest"],
            environment_file=_path(root, value["environment_file"], "environment_file"),
            e3_static_vectors=_path(root, value["e3_static_vectors"], "e3_static_vectors"),
            e5_adaptive_controllers=_path(
                root, value["e5_adaptive_controllers"], "e5_adaptive_controllers"
            ),
            runtime_artifact=_path(root, value["runtime_artifact"], "runtime_artifact"),
            execution_key_file=_path(root, value["execution_key_file"], "execution_key_file"),
            run_directory=_path(root, value["run_directory"], "run_directory"),
            work_directory=_path(root, value["work_directory"], "work_directory"),
            likelihood_directory=_path(root, value["likelihood_directory"], "likelihood_directory"),
            final_directory=_path(root, value["final_directory"], "final_directory"),
            prerequisite_runs=MappingProxyType(
                {
                    ExperimentPhase(name): _path(root, raw, f"prerequisite_runs.{name}")
                    for name, raw in prerequisites.items()
                }
            ),
            seed=value["seed"],
            max_new_tokens=value["max_new_tokens"],
            m1_tensor_index=(
                tensor_index[0],
                tensor_index[1],
                tensor_index[2],
                tensor_index[3],
            ),
            controller_source_prompt_id=value["controller_source_prompt_id"],
            runbook_digest=stable_hash(value),
        )


def write_e6_runbook_template(
    path: str | Path,
    *,
    m1_layer: int,
    official_grader_bundle: str | Path,
    expected_grader_manifest_digest: str,
) -> str:
    """Write a secret-free E6 runbook template for the A100 execution host."""

    if isinstance(m1_layer, bool) or not isinstance(m1_layer, int) or not 0 <= m1_layer < 64:
        raise DataValidationError("E6 M1 layer must be an explicit Qwen layer index")
    if (
        len(expected_grader_manifest_digest) != 64
        or any(character not in "0123456789abcdef" for character in expected_grader_manifest_digest)
    ):
        raise DataValidationError("E6 grader manifest must be a lowercase SHA-256 digest")
    destination = Path(path).resolve()
    if destination.exists() or destination.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite E6 runbook: {destination}")
    body = {
        "schema_version": 1,
        "phase": "E6",
        "study_protocol": "../../../../configs/experiments/phases.yaml",
        "model_config": "../../../../configs/models/qwen3.6-27b-nvfp4.yaml",
        "prompt_config": "../../../../configs/prompts/primary.yaml",
        "snapshot_directory": (
            "../../../models/qwen3.6-27b-nvfp4/"
            "0893e1606ff3d5f97a441f405d5fc541a6bdf404"
        ),
        "snapshot_manifest": "../../../../configs/models/qwen3.6-27b-nvfp4.snapshot.json",
        "frozen_question_bundle": "../frozen/E6-questions",
        "official_grader_bundle": os.path.relpath(
            Path(official_grader_bundle).resolve(), start=destination.parent
        ),
        "expected_grader_manifest_digest": expected_grader_manifest_digest,
        "environment_file": "../../../../.env",
        "e3_static_vectors": "../E3-operator/vectors",
        "e5_adaptive_controllers": "../frozen/E5-phase/selected-controller",
        "runtime_artifact": "../runtime/E6-attestation.json",
        "execution_key_file": "../secrets/execution-private-key.hex",
        "run_directory": "../runs/E6",
        "work_directory": "../work/E6",
        "likelihood_directory": "../frozen/E6-likelihoods",
        "final_directory": "../final/E6",
        "prerequisite_runs": {"E3": "../E3-operator/phase", "E5": "../runs/E5"},
        "seed": 17,
        "max_new_tokens": 32,
        "m1_tensor_index": ["P0-neutral", "M1-P", "post_mlp", m1_layer],
        "controller_source_prompt_id": "P0-neutral",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sha256_file(destination)


def freeze_e6_question_bundle(
    directory: str | Path,
    *,
    reviewed_splits: str | Path,
    expected_reviewed_split_manifest_digest: str,
    source_artifacts: Mapping[str, str | Path],
    study_protocol: str | Path = "configs/experiments/phases.yaml",
    model_config: str | Path = "configs/models/qwen3.6-27b-nvfp4.yaml",
    prompt_config: str | Path = "configs/prompts/primary.yaml",
    seed: int = 17,
) -> Mapping[str, Any]:
    """Freeze the exact reviewed E6 development schedule and its pinned raw sources."""

    if (
        len(expected_reviewed_split_manifest_digest) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_reviewed_split_manifest_digest
        )
    ):
        raise DataValidationError(
            "E6 expected reviewed-split manifest digest must be lowercase SHA-256"
        )
    if set(source_artifacts) != set(_REVIEWED_QUESTION_FILES):
        raise DataValidationError("E6 source-artifact inventory differs")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise DataValidationError("E6 question-bundle seed must be a non-negative integer")

    reviewed = Path(reviewed_splits).absolute()
    manifest = validate_reviewed_split_snapshot(reviewed)
    if manifest.get("manifest_digest") != expected_reviewed_split_manifest_digest:
        raise FrozenArtifactError("E6 reviewed-split manifest differs from the approved snapshot")

    questions: dict[str, tuple[Question, ...]] = {}
    seen: set[str] = set()
    for benchmark, filename in _REVIEWED_QUESTION_FILES.items():
        values = tuple(read_questions(reviewed / filename))
        identifiers = tuple(item.question_id for item in values)
        if (
            len(values) != _QUESTION_COUNTS[benchmark]
            or any(item.benchmark != benchmark for item in values)
            or len(set(identifiers)) != len(identifiers)
            or seen.intersection(identifiers)
        ):
            raise DataValidationError(f"E6 reviewed {benchmark} question schedule differs")
        seen.update(identifiers)
        questions[benchmark] = values

    study = load_study_protocol(study_protocol)
    model = load_model_spec(model_config)
    validate_active_model_spec(model)
    prompts = {item.prompt_id: item for item in load_prompt_specs(prompt_config)}
    try:
        prompt = prompts["P0-neutral"]
    except KeyError as exc:
        raise DataValidationError("E6 question freeze requires P0-neutral") from exc
    prompt_sha256 = hashlib.sha256(prompt.text.encode("utf-8")).hexdigest()
    conditions = tuple(
        EvaluationCondition(
            phase=ExperimentPhase.E6,
            benchmark=benchmark,
            partition=_PARTITIONS[benchmark],
            model_name=model.name,
            model_repository=model.repository,
            model_revision=model.revision,
            runtime=model.runtime,
            quantization=model.quantization,
            model_num_layers=model.num_layers,
            system_prompt_id=prompt.prompt_id,
            prompt_template_sha256=prompt_sha256,
            steering_method="M0",
            method_artifact_sha256=None,
            layer=None,
            site=None,
            token_scope=None,
            alpha=0.0,
            sparsity=None,
            seed=seed,
            study_protocol_digest=study.digest,
            comparison_group=f"e6-question-freeze-{benchmark}",
        )
        for benchmark in _REVIEWED_QUESTION_FILES
    )
    contract = PhaseRunContract(
        phase=ExperimentPhase.E6,
        study_protocol_digest=study.digest,
        conditions=conditions,
        question_ids_by_benchmark={
            benchmark: tuple(item.question_id for item in values)
            for benchmark, values in questions.items()
        },
        input_fingerprints={
            "reviewed_splits": sha256_path(reviewed),
            **{
                f"source_{benchmark}": sha256_path(path)
                for benchmark, path in source_artifacts.items()
            },
        },
        prerequisite_digests={},
        required_gates=study.phase(ExperimentPhase.E6).gates,
    )
    fingerprint = write_frozen_question_bundle(
        directory,
        contract,
        questions,
        source_artifacts=source_artifacts,
    )
    return MappingProxyType(
        {
            "valid": True,
            "phase": ExperimentPhase.E6.value,
            "directory": str(Path(directory).absolute()),
            "sha256": fingerprint,
            "study_protocol_digest": study.digest,
            "reviewed_split_manifest_digest": expected_reviewed_split_manifest_digest,
            "question_counts": dict(_QUESTION_COUNTS),
            "question_ids_sha256": {
                benchmark: stable_hash([item.question_id for item in values])
                for benchmark, values in questions.items()
            },
        }
    )


def _private_key(runbook: E6Runbook) -> tuple[str, str]:
    try:
        private_hex = runbook.execution_key_file.read_text(encoding="utf-8").strip()
        private = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_hex))
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"cannot load E6 execution key: {exc}") from exc
    if len(private_hex) != 64 or private_hex.lower() != private_hex:
        raise ConfigurationError("E6 execution key must be one lowercase 32-byte hex key")
    public_hex = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    return private_hex, public_hex


def _questions(runbook: E6Runbook) -> Mapping[str, tuple[Question, ...]]:
    observed: dict[str, tuple[Question, ...]] = {}
    seen: set[str] = set()
    for benchmark, count in _QUESTION_COUNTS.items():
        values = tuple(read_questions(runbook.frozen_question_bundle / f"{benchmark}.jsonl"))
        identifiers = tuple(item.question_id for item in values)
        if (
            len(values) != count
            or any(item.benchmark != benchmark for item in values)
            or len(set(identifiers)) != count
            or seen.intersection(identifiers)
        ):
            raise DataValidationError(f"E6 {benchmark} question schedule differs")
        seen.update(identifiers)
        observed[benchmark] = values
    return MappingProxyType(observed)


def _prompts(runbook: E6Runbook) -> Mapping[str, PromptSpec]:
    values = {item.prompt_id: item for item in load_prompt_specs(runbook.prompt_config)}
    if not set(_PROMPTS) <= set(values):
        raise DataValidationError("E6 prompt configuration lacks P0/P2/P3")
    return MappingProxyType({name: values[name] for name in _PROMPTS})


def _completion_digests(runbook: E6Runbook, study: StudyProtocol) -> Mapping[str, str]:
    result: dict[str, str] = {}
    for phase, path in runbook.prerequisite_runs.items():
        completion = open_phase_prerequisite(path, phase=phase, study=study).verify_complete()
        result[phase.value] = completion.completion_digest
    return MappingProxyType(result)


def _e3_slice(
    directory: Path, tensor_index: tuple[str, str, str, int]
) -> tuple[np.ndarray[Any, Any], float, str]:
    index = _e3_direction_index(directory)
    try:
        direction_sha = index[tensor_index]
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        prompt_index = metadata["prompt_axis"].index(tensor_index[0])
        extraction_index = metadata["extraction_axis"].index(tensor_index[1])
        site_index = metadata["site_axis"].index(tensor_index[2])
        layer_index = metadata["layer_axis"].index(tensor_index[3])
        with np.load(directory / "vectors.npz", allow_pickle=False) as values:
            direction = np.ascontiguousarray(
                values["directions"][prompt_index, extraction_index, site_index, layer_index],
                dtype=np.float32,
            )
            reference_rms = float(
                values["reference_rms"][prompt_index, extraction_index, site_index, layer_index]
            )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot load E6 M1 tensor slice: {exc}") from exc
    if (
        hashlib.sha256(direction.tobytes(order="C")).hexdigest() != direction_sha
        or not math.isclose(float(np.linalg.norm(direction)), 1.0, rel_tol=1e-5, abs_tol=1e-6)
        or not math.isfinite(reference_rms)
        or reference_rms <= 0
    ):
        raise FrozenArtifactError("E6 M1 tensor bytes or RMS differ")
    return direction, reference_rms, direction_sha


@dataclass(frozen=True, slots=True)
class _E6Context:
    study: StudyProtocol
    model: ModelSpec
    prompts: Mapping[str, PromptSpec]
    questions: Mapping[str, tuple[Question, ...]]
    grader_bundle: E6OfficialGraderBundle
    contract: PhaseRunContract
    controller: AdaptiveController
    controller_path: Path
    controller_source_prompt: PromptSpec
    m1_direction: np.ndarray[Any, Any]
    m1_reference_rms: float
    execution_private_key: str
    execution_public_key: str


def _condition(
    *,
    study: StudyProtocol,
    model: ModelSpec,
    prompt: PromptSpec,
    benchmark: str,
    method: str,
    seed: int,
    comparison_group: str,
    m1_artifact: str,
    m1_layer: int,
    m1_site: ActivationSite,
    m1_scope: TokenScope,
    m1_raw_alpha: float,
    e5_artifact: str,
    adaptive_policy: AdaptivePolicySpec,
) -> EvaluationCondition:
    if method == "M0":
        artifact, layer, site, scope, alpha, policy = (
            None,
            None,
            None,
            None,
            0.0,
            None,
        )
    elif method == "M1":
        artifact, layer, site, scope, alpha, policy = (
            m1_artifact,
            m1_layer,
            m1_site,
            m1_scope,
            m1_raw_alpha,
            None,
        )
    elif method == "M3":
        artifact, layer, site, scope, alpha, policy = (
            e5_artifact,
            None,
            None,
            None,
            0.0,
            adaptive_policy,
        )
    else:  # pragma: no cover - exact private caller
        raise DataValidationError("E6 method is invalid")
    return EvaluationCondition(
        phase=ExperimentPhase.E6,
        benchmark=benchmark,
        partition=_PARTITIONS[benchmark],
        model_name=model.name,
        model_repository=model.repository,
        model_revision=model.revision,
        runtime=model.runtime,
        quantization=model.quantization,
        model_num_layers=model.num_layers,
        system_prompt_id=prompt.prompt_id,
        prompt_template_sha256=hashlib.sha256(prompt.text.encode("utf-8")).hexdigest(),
        steering_method=method,
        method_artifact_sha256=artifact,
        layer=layer,
        site=site,
        token_scope=scope,
        alpha=alpha,
        sparsity=None,
        seed=seed,
        study_protocol_digest=study.digest,
        adaptive_policy=policy,
        comparison_group=comparison_group,
    )


def _context(runbook: E6Runbook) -> _E6Context:
    paths = {
        "E6 runbook": runbook.source,
        "E6 questions": runbook.frozen_question_bundle,
        "E6 E3 vectors": runbook.e3_static_vectors,
        "E6 E5 controller": runbook.e5_adaptive_controllers,
        "E6 execution key": runbook.execution_key_file,
        **{
            f"E6 prerequisite {phase.value}": path
            for phase, path in runbook.prerequisite_runs.items()
        },
    }
    validate_active_study_artifact_paths(paths)
    study = load_study_protocol(runbook.study_protocol)
    model = load_model_spec(runbook.model_config)
    validate_active_model_spec(model)
    verify_transformers_snapshot(model, runbook.snapshot_directory, runbook.snapshot_manifest)
    prompts = _prompts(runbook)
    questions = _questions(runbook)
    grader_bundle = load_e6_official_grader_bundle(
        runbook.official_grader_bundle,
        expected_manifest_digest=runbook.expected_grader_manifest_digest,
    )
    private_hex, public_hex = _private_key(runbook)
    selected = validate_e5_selected_controller_bundle(runbook.e5_adaptive_controllers)
    controller = selected["controller"]
    controller_path = selected["controller_path"]
    if type(controller) is not AdaptiveController or not isinstance(controller_path, Path):
        raise FrozenArtifactError("E6 selected controller is not loadable")
    source_prompt = prompts[runbook.controller_source_prompt_id]
    schema = controller.risk_probe.training_schema
    if (
        schema.prompt_id != source_prompt.prompt_id
        or schema.prompt_sha256 != hashlib.sha256(source_prompt.text.encode("utf-8")).hexdigest()
        or schema.model_repository != model.repository
        or schema.model_revision != model.revision
        or schema.runtime is not model.runtime
        or schema.quantization != model.quantization
        or selected["execution_public_key"] != public_hex
    ):
        raise FrozenArtifactError("E6 selected controller differs from model, prompt, or key")
    e3_sha = sha256_path(runbook.e3_static_vectors)
    e5_sha = sha256_path(runbook.e5_adaptive_controllers)
    direction, reference_rms, direction_sha = _e3_slice(
        runbook.e3_static_vectors, runbook.m1_tensor_index
    )
    tensor_prompt, _extraction, tensor_site, tensor_layer = runbook.m1_tensor_index
    if tensor_prompt not in {"P0-neutral", "P2-calibrated-abstention"}:
        raise DataValidationError("E6 M1 tensor must originate from P0 or P2")
    e5_ledger = open_phase_prerequisite(
        runbook.prerequisite_runs[ExperimentPhase.E5],
        phase=ExperimentPhase.E5,
        study=study,
    )
    m1_geometry = {
        (item.layer, item.site, item.token_scope, item.alpha)
        for item in e5_ledger.contract.conditions
        if item.steering_method == "M1"
    }
    m3_policies = {
        stable_hash(item.adaptive_policy.to_dict()): item.adaptive_policy
        for item in e5_ledger.contract.conditions
        if item.steering_method == "M3" and item.adaptive_policy is not None
    }
    if len(m1_geometry) != 1 or len(m3_policies) != 1:
        raise FrozenArtifactError("E6 E5 prerequisite lacks one promoted M1/M3 geometry")
    m1_layer, m1_site, m1_scope, m1_standardized_alpha = next(iter(m1_geometry))
    adaptive_policy = next(iter(m3_policies.values()))
    if (
        e5_ledger.contract.input_fingerprints.get("E3_static_vectors") != e3_sha
        or m1_layer != tensor_layer
        or m1_site is None
        or m1_site.value != tensor_site
        or m1_scope is None
        or m1_standardized_alpha == 0
        or adaptive_policy.controller_artifact_sha256 != selected["controller_artifact_sha256"]
        or adaptive_policy.execution_public_key != public_hex
    ):
        raise FrozenArtifactError("E6 inputs differ from E5's promoted methods")
    m1_raw_alpha = float(m1_standardized_alpha) * reference_rms
    m1_artifact = e6_e3_slice_digest(
        e3_static_vectors_sha256=e3_sha,
        tensor_index=runbook.m1_tensor_index,
        direction_sha256=direction_sha,
    )
    source_digest = stable_hash(
        {
            "m1_tensor_index": list(runbook.m1_tensor_index),
            "controller_source_prompt_id": source_prompt.prompt_id,
        }
    )[:16]
    conditions: list[EvaluationCondition] = []
    for benchmark in _QUESTION_COUNTS:
        for prompt_id in _PROMPTS:
            prompt = prompts[prompt_id]
            for method in _METHODS:
                conditions.append(
                    _condition(
                        study=study,
                        model=model,
                        prompt=prompt,
                        benchmark=benchmark,
                        method=method,
                        seed=runbook.seed,
                        comparison_group=f"e6-{benchmark}-{prompt_id}-{source_digest}",
                        m1_artifact=m1_artifact,
                        m1_layer=m1_layer,
                        m1_site=m1_site,
                        m1_scope=m1_scope,
                        m1_raw_alpha=m1_raw_alpha,
                        e5_artifact=e5_sha,
                        adaptive_policy=adaptive_policy,
                    )
                )
    contract = PhaseRunContract(
        phase=ExperimentPhase.E6,
        study_protocol_digest=study.digest,
        conditions=tuple(conditions),
        question_ids_by_benchmark={
            benchmark: tuple(item.question_id for item in values)
            for benchmark, values in questions.items()
        },
        input_fingerprints={
            "E3_static_vectors": e3_sha,
            "E5_adaptive_controllers": e5_sha,
            "official_grader_bundle": sha256_path(runbook.official_grader_bundle),
        },
        prerequisite_digests=_completion_digests(runbook, study),
        required_gates=study.phase(ExperimentPhase.E6).gates,
    )
    contract.assert_matches_study(study)
    _validate_question_bundle(runbook.frozen_question_bundle, contract)
    return _E6Context(
        study=study,
        model=model,
        prompts=prompts,
        questions=questions,
        grader_bundle=grader_bundle,
        contract=contract,
        controller=controller,
        controller_path=controller_path,
        controller_source_prompt=source_prompt,
        m1_direction=direction,
        m1_reference_rms=reference_rms,
        execution_private_key=private_hex,
        execution_public_key=public_hex,
    )


def _work_manifest(runbook: E6Runbook, context: _E6Context) -> dict[str, Any]:
    body = {
        "schema_version": 1,
        "phase": "E6-native-operator",
        "runbook_digest": runbook.runbook_digest,
        "contract_digest": context.contract.digest,
        "frozen_question_bundle_sha256": sha256_path(runbook.frozen_question_bundle),
        "runtime_artifact": str(runbook.runtime_artifact),
        "e3_static_vectors_sha256": context.contract.input_fingerprints["E3_static_vectors"],
        "e5_adaptive_controllers_sha256": context.contract.input_fingerprints[
            "E5_adaptive_controllers"
        ],
        "official_grader_bundle_sha256": context.contract.input_fingerprints[
            "official_grader_bundle"
        ],
        "official_grader_manifest_digest": context.grader_bundle.manifest_digest,
        "execution_public_key": context.execution_public_key,
        "m1_tensor_index": list(runbook.m1_tensor_index),
        "controller_source_prompt_id": runbook.controller_source_prompt_id,
        "seed": runbook.seed,
        "max_new_tokens": runbook.max_new_tokens,
        "expected_records": context.contract.expected_record_count,
    }
    return {**body, "manifest_digest": stable_hash(body)}


def _verify_work(runbook: E6Runbook, context: _E6Context) -> Mapping[str, Any]:
    root = runbook.work_directory
    if (
        root.is_symlink()
        or not root.is_dir()
        or {item.name for item in root.iterdir()} != {"manifest.json", "rows"}
        or not (root / "rows").is_dir()
        or any(item.is_symlink() for item in root.rglob("*"))
    ):
        raise FrozenArtifactError("E6 work inventory differs")
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read E6 work manifest: {exc}") from exc
    if manifest != _work_manifest(runbook, context):
        raise FrozenArtifactError("E6 work manifest differs from the runbook")
    return MappingProxyType(dict(manifest))


def preflight_e6_runbook(runbook: E6Runbook) -> Mapping[str, Any]:
    context = _context(runbook)
    runtime_status = "pending"
    if runbook.runtime_artifact.exists():
        attestation = _load_e6_runtime_attestation(runbook.runtime_artifact)
        if attestation["execution_public_key"] != context.execution_public_key:
            raise FrozenArtifactError("E6 runtime artifact uses another execution key")
        runtime_status = "verified"
    return MappingProxyType(
        {
            "valid": True,
            "phase": "E6",
            "runbook_digest": runbook.runbook_digest,
            "contract_digest": context.contract.digest,
            "expected_records": context.contract.expected_record_count,
            "runtime_artifact_status": runtime_status,
            "execution_public_key": context.execution_public_key,
        }
    )


def prepare_e6_runbook(runbook: E6Runbook) -> PhaseRunLedger:
    """Create or safely reopen E6's exact ledger and immutable row workspace."""

    context = _context(runbook)
    mutable = {
        "E6 ledger": runbook.run_directory,
        "E6 work": runbook.work_directory,
        "E6 likelihood": runbook.likelihood_directory,
        "E6 final": runbook.final_directory,
    }
    validate_active_study_artifact_paths(mutable)
    if runbook.run_directory.exists():
        ledger = PhaseRunLedger.open(runbook.run_directory, study=context.study)
        if ledger.contract != context.contract:
            raise FrozenArtifactError("existing E6 ledger differs from the runbook")
    else:
        ledger = PhaseRunLedger.create(
            runbook.run_directory,
            context.contract,
            study=context.study,
            input_artifacts={
                "E3_static_vectors": runbook.e3_static_vectors,
                "E5_adaptive_controllers": runbook.e5_adaptive_controllers,
                "official_grader_bundle": runbook.official_grader_bundle,
            },
            prerequisite_runs={phase: path for phase, path in runbook.prerequisite_runs.items()},
        )
    if runbook.work_directory.exists():
        _verify_work(runbook, context)
    else:
        runbook.work_directory.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(
            tempfile.mkdtemp(
                prefix=f".{runbook.work_directory.name}.stage-",
                dir=runbook.work_directory.parent,
            )
        )
        try:
            (stage / "rows").mkdir()
            (stage / "manifest.json").write_text(
                json.dumps(_work_manifest(runbook, context), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(stage, runbook.work_directory)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    return ledger


def _native_runtime(
    runbook: E6Runbook, context: _E6Context
) -> tuple[VllmResearchRuntime, E6RuntimeAttestor]:
    runtime = VllmResearchRuntime.from_spec(
        context.model,
        snapshot_path=runbook.snapshot_directory,
        seed=runbook.seed,
        research_provenance={
            "phase": "E6",
            "runbook_digest": runbook.runbook_digest,
            "contract_digest": context.contract.digest,
            "frozen_question_bundle_sha256": sha256_path(runbook.frozen_question_bundle),
        },
    )
    attestor = E6RuntimeAttestor(runtime, execution_private_key=context.execution_private_key)
    if runbook.runtime_artifact.exists():
        attestor.verify_runtime_artifact(runbook.runtime_artifact)
    else:
        attestor.write_runtime_artifact(runbook.runtime_artifact)
    return runtime, attestor


def attest_e6_runtime(runbook: E6Runbook) -> Mapping[str, Any]:
    """Load pinned Qwen through VLLM and freeze/replay the machine attestation."""

    context = _context(runbook)
    runtime, attestor = _native_runtime(runbook, context)
    try:
        return MappingProxyType(
            {
                "valid": True,
                "runtime_artifact": runbook.runtime_artifact,
                "runtime_artifact_sha256": attestor.verify_runtime_artifact(
                    runbook.runtime_artifact
                ),
                "execution_public_key": attestor.execution_public_key,
            }
        )
    finally:
        runtime.close()


def _draft_record(
    *,
    runtime: VllmResearchRuntime,
    condition: EvaluationCondition,
    question: Question,
    prompt: PromptSpec,
) -> GenerationRecord:
    rendered = runtime.render_prompt(prompt, question.text, metadata=question.metadata)
    metadata: dict[str, Any] = {
        "phase": "E6",
        "partition": condition.partition,
        "prompt_template_sha256": condition.prompt_template_sha256,
        "study_protocol_digest": condition.study_protocol_digest,
    }
    if condition.method_artifact_sha256 is not None:
        metadata["method_artifact_sha256"] = condition.method_artifact_sha256
    return GenerationRecord(
        question_id=question.question_id,
        benchmark=question.benchmark,
        model_repository=condition.model_repository,
        model_revision=condition.model_revision,
        runtime=condition.runtime,
        quantization=condition.quantization,
        system_prompt_id=condition.system_prompt_id,
        rendered_prompt_hash=rendered.sha256,
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
        seed=condition.seed,
        metadata=metadata,
    )


def _m1_state_factory(
    context: _E6Context,
    condition: EvaluationCondition,
    runtime: VllmResearchRuntime,
) -> Callable[[], Mapping[int, VllmResearchInterventionState]]:
    assert condition.layer is not None
    assert condition.token_scope is not None
    layer = condition.layer
    scope = condition.token_scope

    def factory() -> Mapping[int, VllmResearchInterventionState]:
        state = runtime.standardized_intervention_state(
            context.m1_direction,
            standardized_alpha=condition.alpha,
            reference_rms=1.0,
            token_scope=scope,
        )
        return {layer: state}

    return factory


def _adaptive_state_factory(
    context: _E6Context,
    record: GenerationRecord,
    runtime: VllmResearchRuntime,
) -> Callable[[], Mapping[int, VllmResearchInterventionState]] | None:
    if record.metadata.get("policy_action") == "release":
        return None
    evidence = record.metadata.get("adaptive_controller_evidence")
    if not isinstance(evidence, Mapping) or not isinstance(evidence.get("feature_values"), list):
        raise FrozenArtifactError("E6 M3 row lacks replayable prompt features")
    features = np.ascontiguousarray(evidence["feature_values"], dtype=np.float32)
    decision = context.controller.decide(torch.from_numpy(features.copy()).unsqueeze(0))
    assert record.layer is not None
    assert record.site is not None
    assert record.token_scope is not None
    record_layer = record.layer
    record_scope = record.token_scope
    eligible = [
        value[0].detach().cpu().float().contiguous()
        for key, value in decision.directions.items()
        if key.layer == record_layer and key.site is record.site
    ]
    if len(eligible) != 1:
        raise FrozenArtifactError("E6 M3 decision does not reproduce its routed direction")
    values = np.ascontiguousarray(eligible[0].numpy(), dtype=np.float32)
    norm = float(np.linalg.norm(values))
    normalized = np.ascontiguousarray(values / norm, dtype=np.float32)
    trace = record.metadata.get("intervention_trace")
    if (
        not math.isfinite(norm)
        or norm <= 0
        or not isinstance(trace, Mapping)
        or trace.get("direction_sha256")
        != hashlib.sha256(normalized.tobytes(order="C")).hexdigest()
    ):
        raise FrozenArtifactError("E6 M3 routed direction differs from its signed trace")

    def factory() -> Mapping[int, VllmResearchInterventionState]:
        state = runtime.standardized_intervention_state(
            normalized,
            standardized_alpha=record.alpha * norm,
            reference_rms=1.0,
            token_scope=record_scope,
        )
        return {record_layer: state}

    return factory


def _execute_row(
    *,
    runbook: E6Runbook,
    context: _E6Context,
    runtime: VllmResearchRuntime,
    attestor: E6RuntimeAttestor,
    condition: EvaluationCondition,
    question: Question,
    grader: E6FactualGrader,
) -> E6ExecutedRow:
    prompt = context.prompts[condition.system_prompt_id]
    draft = _draft_record(runtime=runtime, condition=condition, question=question, prompt=prompt)
    question_sha = sha256_path(runbook.frozen_question_bundle)
    if condition.steering_method == "M3":
        generated = execute_e6_adaptive_generation(
            attestor=attestor,
            runtime_artifact=runbook.runtime_artifact,
            controller_artifact=context.controller_path,
            question=question,
            prompt=prompt,
            controller_prompt=context.controller_source_prompt,
            generation_record=draft,
            condition=condition,
            max_new_tokens=runbook.max_new_tokens,
            populate_generation=True,
            generation_grader=lambda record: grader(record, question),
        )
        state_factory = _adaptive_state_factory(context, generated, runtime)
        layers = (
            (generated.layer,)
            if generated.layer is not None
            else tuple(condition.adaptive_policy.candidate_layers)
            if condition.adaptive_policy is not None
            else ()
        )
        site = generated.site or (
            condition.adaptive_policy.candidate_sites[0]
            if condition.adaptive_policy is not None
            else ActivationSite.POST_MLP
        )
        return execute_and_bind_e6_likelihood(
            attestor=attestor,
            runtime_artifact=runbook.runtime_artifact,
            e3_static_vectors=runbook.e3_static_vectors,
            question=question,
            prompt=prompt,
            generation_record=generated,
            condition=condition,
            layers=layers,
            site=site,
            state_factory=state_factory,
            question_bundle_sha256=question_sha,
            max_new_tokens=runbook.max_new_tokens,
        )
    m1_factory = (
        _m1_state_factory(context, condition, runtime)
        if condition.steering_method == "M1"
        else None
    )
    m1_condition = next(
        item
        for item in context.contract.conditions
        if item.benchmark == condition.benchmark
        and item.system_prompt_id == condition.system_prompt_id
        and item.steering_method == "M1"
    )
    assert m1_condition.layer is not None
    assert m1_condition.site is not None
    return execute_and_bind_e6_likelihood(
        attestor=attestor,
        runtime_artifact=runbook.runtime_artifact,
        e3_static_vectors=runbook.e3_static_vectors,
        question=question,
        prompt=prompt,
        generation_record=draft,
        condition=condition,
        layers=(m1_condition.layer,),
        site=m1_condition.site,
        state_factory=m1_factory,
        question_bundle_sha256=question_sha,
        e3_tensor_index=(runbook.m1_tensor_index if condition.steering_method == "M1" else None),
        max_new_tokens=runbook.max_new_tokens,
        populate_generation=True,
        generation_grader=lambda record: grader(record, question),
    )


def _row_path(runbook: E6Runbook, condition_id: str, question_id: str) -> Path:
    return runbook.work_directory / "rows" / condition_id / f"{question_id}.json"


def _write_row(path: Path, row: E6ExecutedRow) -> None:
    body = {
        "schema_version": 1,
        "generation_record": row.generation_record.to_dict(),
        "likelihood_record": row.likelihood_record.to_dict(),
    }
    payload = {**body, "row_digest": stable_hash(body)}
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.with_name(f".{path.name}.stage-{os.getpid()}")
    try:
        with stage.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(stage, path)
    finally:
        if stage.exists():
            stage.unlink()


def _load_row(
    path: Path,
    *,
    condition: EvaluationCondition,
    question: Question,
    runtime_artifact_sha256: str,
    execution_public_key: str,
    grader_bundle: E6OfficialGraderBundle,
    controller_prompt_id: str,
    controller_prompt_sha256: str,
) -> E6ExecutedRow:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        body = dict(value)
        digest = body.pop("row_digest")
        generation = GenerationRecord.from_dict(body["generation_record"])
        likelihood = E6VerifiedLikelihoodRecord.from_dict(body["likelihood_record"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise FrozenArtifactError(f"cannot load E6 operator row: {exc}") from exc
    if (
        set(body) != {"schema_version", "generation_record", "likelihood_record"}
        or body["schema_version"] != 1
        or digest != stable_hash(body)
        or generation.condition_id != condition.condition_id
        or generation.question_id != question.question_id
    ):
        raise FrozenArtifactError("E6 operator row identity differs")
    verify_e6_bound_record(
        likelihood,
        generation_record=generation,
        condition=condition,
        question=question,
        runtime_artifact_sha256=runtime_artifact_sha256,
        execution_public_key=execution_public_key,
    )
    verify_e6_factual_grade(generation, question, grader_bundle=grader_bundle)
    if condition.steering_method == "M3":
        if (
            generation.metadata.get("controller_prompt_id") != controller_prompt_id
            or generation.metadata.get("controller_prompt_sha256") != controller_prompt_sha256
        ):
            raise FrozenArtifactError("E6 controller source differs from the runbook")
    elif (
        generation.metadata.get("controller_prompt_id") is not None
        or generation.metadata.get("controller_prompt_sha256") is not None
    ):
        raise FrozenArtifactError("E6 fixed row preclaims adaptive controller provenance")
    return E6ExecutedRow(generation, likelihood)


def execute_e6_runbook(runbook: E6Runbook, *, limit: int | None = None) -> Mapping[str, Any]:
    """Load Qwen once and resume exact pending E6 rows with atomic row receipts."""

    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        raise ConfigurationError("E6 execution limit must be a positive integer")
    context = _context(runbook)
    ledger = prepare_e6_runbook(runbook)
    _verify_work(runbook, context)
    runtime, attestor = _native_runtime(runbook, context)
    runtime_sha = attestor.verify_runtime_artifact(runbook.runtime_artifact)
    questions = {
        (item.benchmark, item.question_id): item
        for values in context.questions.values()
        for item in values
    }
    executed = 0
    grader = E6FactualGrader(
        context.grader_bundle,
        environment_file=runbook.environment_file,
    )
    try:
        for pending in ledger.iter_pending():
            if limit is not None and executed >= limit:
                break
            condition = pending.condition
            question = questions[(condition.benchmark, pending.question_id)]
            path = _row_path(runbook, condition.condition_id, pending.question_id)
            if path.exists():
                row = _load_row(
                    path,
                    condition=condition,
                    question=question,
                    runtime_artifact_sha256=runtime_sha,
                    execution_public_key=context.execution_public_key,
                    grader_bundle=context.grader_bundle,
                    controller_prompt_id=context.controller_source_prompt.prompt_id,
                    controller_prompt_sha256=hashlib.sha256(
                        context.controller_source_prompt.text.encode()
                    ).hexdigest(),
                )
            else:
                row = _execute_row(
                    runbook=runbook,
                    context=context,
                    runtime=runtime,
                    attestor=attestor,
                    condition=condition,
                    question=question,
                    grader=grader,
                )
                _write_row(path, row)
            ledger.checkpoint((row.generation_record,))
            executed += 1
    finally:
        runtime.close()
    completed, expected = ledger.progress()
    return MappingProxyType(
        {
            "valid": True,
            "executed_records": executed,
            "completed_records": completed,
            "expected_records": expected,
            "remaining_records": expected - completed,
            "runtime_artifact_sha256": runtime_sha,
        }
    )


def _ordered_rows(
    runbook: E6Runbook, context: _E6Context, ledger: PhaseRunLedger
) -> tuple[E6VerifiedLikelihoodRecord, ...]:
    if not runbook.runtime_artifact.is_file():
        raise FrozenArtifactError("E6 runtime artifact is absent")
    runtime_sha = sha256_path(runbook.runtime_artifact)
    records = {(item.condition_id, item.question_id): item for item in ledger.records()}
    rows: list[E6VerifiedLikelihoodRecord] = []
    expected_paths: set[Path] = set()
    for condition in context.contract.conditions:
        by_id = {item.question_id: item for item in context.questions[condition.benchmark]}
        for question_id in context.contract.question_ids_by_benchmark[condition.benchmark]:
            path = _row_path(runbook, condition.condition_id, question_id)
            expected_paths.add(path)
            row = _load_row(
                path,
                condition=condition,
                question=by_id[question_id],
                runtime_artifact_sha256=runtime_sha,
                execution_public_key=context.execution_public_key,
                grader_bundle=context.grader_bundle,
                controller_prompt_id=context.controller_source_prompt.prompt_id,
                controller_prompt_sha256=hashlib.sha256(
                    context.controller_source_prompt.text.encode()
                ).hexdigest(),
            )
            if records.get((condition.condition_id, question_id)) != row.generation_record:
                raise FrozenArtifactError("E6 ledger row differs from its operator receipt")
            rows.append(row.likelihood_record)
    observed_paths = set((runbook.work_directory / "rows").glob("*/*.json"))
    if observed_paths != expected_paths:
        raise FrozenArtifactError("E6 operator row inventory differs from the exact matrix")
    return tuple(rows)


def _verify_partial_rows(runbook: E6Runbook, context: _E6Context, ledger: PhaseRunLedger) -> int:
    """Replay every materialized receipt and require one for every checkpointed row."""

    if not runbook.runtime_artifact.is_file():
        if ledger.progress()[0] == 0:
            observed = tuple((runbook.work_directory / "rows").glob("*/*.json"))
            if observed:
                raise FrozenArtifactError("E6 rows exist before runtime attestation")
            return 0
        raise FrozenArtifactError("E6 checkpointed rows lack a runtime attestation")
    runtime_sha = sha256_path(runbook.runtime_artifact)
    conditions = {item.condition_id: item for item in context.contract.conditions}
    questions = {
        (benchmark, item.question_id): item
        for benchmark, values in context.questions.items()
        for item in values
    }
    ledger_records = {(item.condition_id, item.question_id): item for item in ledger.records()}
    observed_keys: set[tuple[str, str]] = set()
    for path in sorted((runbook.work_directory / "rows").glob("*/*.json")):
        relative = path.relative_to(runbook.work_directory / "rows")
        condition_id = relative.parts[0]
        question_id = path.stem
        try:
            condition = conditions[condition_id]
            question = questions[(condition.benchmark, question_id)]
        except KeyError as exc:
            raise FrozenArtifactError("E6 partial row is outside the exact matrix") from exc
        key = (condition_id, question_id)
        if key in observed_keys:
            raise FrozenArtifactError("E6 partial row inventory contains a duplicate")
        observed_keys.add(key)
        row = _load_row(
            path,
            condition=condition,
            question=question,
            runtime_artifact_sha256=runtime_sha,
            execution_public_key=context.execution_public_key,
            grader_bundle=context.grader_bundle,
            controller_prompt_id=context.controller_source_prompt.prompt_id,
            controller_prompt_sha256=hashlib.sha256(
                context.controller_source_prompt.text.encode()
            ).hexdigest(),
        )
        checkpointed = ledger_records.get(key)
        if checkpointed is not None and checkpointed != row.generation_record:
            raise FrozenArtifactError("E6 checkpoint differs from its partial row receipt")
    if not set(ledger_records) <= observed_keys:
        raise FrozenArtifactError("E6 checkpointed row lacks its operator receipt")
    return len(observed_keys)


def finalize_e6_runbook(runbook: E6Runbook) -> Mapping[str, Any]:
    """Freeze E6 likelihoods, derive the registered gate, and terminally finalize."""

    context = _context(runbook)
    ledger = PhaseRunLedger.open(runbook.run_directory, study=context.study)
    if ledger.contract != context.contract or ledger.progress()[0] != ledger.progress()[1]:
        raise DataValidationError("E6 finalization requires the complete runbook ledger")
    _verify_work(runbook, context)
    rows = _ordered_rows(runbook, context, ledger)
    if runbook.likelihood_directory.exists():
        verify_e6_likelihood_bundle(
            runbook.likelihood_directory,
            ledger_directory=runbook.run_directory,
            study=context.study,
            questions_by_benchmark=context.questions,
            runtime_artifact=runbook.runtime_artifact,
            execution_public_key=context.execution_public_key,
        )
    else:
        write_e6_likelihood_bundle(
            runbook.likelihood_directory,
            ledger_directory=runbook.run_directory,
            study=context.study,
            questions_by_benchmark=context.questions,
            records=rows,
            runtime_artifact=runbook.runtime_artifact,
            frozen_question_bundle=runbook.frozen_question_bundle,
            execution_public_key=context.execution_public_key,
        )
    return finalize_e6_phase(
        runbook.final_directory,
        ledger_directory=runbook.run_directory,
        study=context.study,
        likelihood_bundle=runbook.likelihood_directory,
        questions_by_benchmark=context.questions,
        runtime_artifact=runbook.runtime_artifact,
        execution_public_key=context.execution_public_key,
    )


def verify_e6_runbook(runbook: E6Runbook) -> Mapping[str, Any]:
    """Replay runbook, contract, workspace, rows, bundle, gate, and terminal receipt."""

    context = _context(runbook)
    ledger = PhaseRunLedger.open(runbook.run_directory, study=context.study)
    if ledger.contract != context.contract:
        raise FrozenArtifactError("E6 ledger differs from the runbook contract")
    _verify_work(runbook, context)
    completed, expected = ledger.progress()
    materialized = _verify_partial_rows(runbook, context, ledger)
    if completed == expected:
        _ordered_rows(runbook, context, ledger)
    terminal: Mapping[str, Any] | None = None
    if runbook.final_directory.exists():
        if completed != expected or not runbook.likelihood_directory.exists():
            raise FrozenArtifactError("terminal E6 output exists before exact row completion")
        verify_e6_likelihood_bundle(
            runbook.likelihood_directory,
            ledger_directory=runbook.run_directory,
            study=context.study,
            questions_by_benchmark=context.questions,
            runtime_artifact=runbook.runtime_artifact,
            execution_public_key=context.execution_public_key,
        )
        terminal = verify_e6_phase(
            runbook.final_directory,
            ledger_directory=runbook.run_directory,
            study=context.study,
        )
    return MappingProxyType(
        {
            "valid": True,
            "phase": "E6",
            "runbook_digest": runbook.runbook_digest,
            "contract_digest": context.contract.digest,
            "completed_records": completed,
            "expected_records": expected,
            "remaining_records": expected - completed,
            "materialized_row_receipts": materialized,
            "terminal": dict(terminal) if terminal is not None else None,
        }
    )

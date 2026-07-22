"""Development-only selection of the calibrated M6 early-token C/I/A probe.

The workflow replays frozen E1 ``T-controller`` outputs and the independently
selected E8 M3 ``T-dev`` outputs through the exact live VLLM runtime.  It captures
post-first-token and post-four-token features, fits the preregistered probe grid,
and freezes one winner before the one-shot E10 ledger can be created.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from safetensors.torch import load_file, save_file

from mfh.contracts import (
    ActivationSite,
    GenerationRecord,
    Outcome,
    PromptSpec,
    Question,
    Runtime,
    TokenScope,
)
from mfh.data.io import read_questions
from mfh.data.reviewed_splits import validate_reviewed_split_snapshot
from mfh.data.splits import semantic_group_ids
from mfh.errors import ConfigurationError, DataValidationError, FrozenArtifactError
from mfh.experiments.e2_schedule import controller_feature_partitions
from mfh.experiments.e6_likelihood import E6RuntimeAttestor
from mfh.experiments.model_selection import validate_active_study_artifact_paths
from mfh.experiments.protocol import ExperimentPhase, StudyProtocol
from mfh.experiments.runner import PhaseRunLedger
from mfh.inference.vllm_research import VllmResearchInterventionState
from mfh.methods.adaptive import AdaptiveController, load_adaptive_controller
from mfh.methods.features import (
    ActivationFeatureSchema,
    ActivationKind,
    FeatureComposition,
)
from mfh.methods.probes import (
    CalibratedProbe,
    CalibrationKind,
    ProbeDataset,
    ProbeKind,
    ProbeMetrics,
    ProbeTask,
    ProbeTrainingConfig,
    evaluate_probe,
    fit_calibrated_probe,
    load_calibrated_probe,
    save_calibrated_probe,
)
from mfh.provenance import canonical_json, sha256_file, sha256_path, stable_hash

_KINDS = (ActivationKind.FIRST_GENERATED, ActivationKind.FIRST_FOUR_GENERATED)
_PROBES = (ProbeKind.LOGISTIC, ProbeKind.TWO_LAYER_MLP)
_CALIBRATORS = (CalibrationKind.TEMPERATURE, CalibrationKind.ISOTONIC)
_TASK = ProbeTask.CORRECT_INCORRECT_ABSTENTION
_TRAIN = "T-controller-train"
_CALIBRATION = "T-controller-calibration"
_DEV = "T-dev"
_FILES = frozenset({"plan.json", "run.lock", "shards"})
_SELECTION_RULE = (
    "maximum-E8-M3-T-dev-incorrect-AUROC-then-macro-AUROC-then-minimum-ECE-then-config-digest"
)


@dataclass(frozen=True, slots=True)
class VerifiedE10EarlyCapture:
    directory: Path
    plan: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...]
    tensors: Mapping[ActivationKind, torch.Tensor]

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan", MappingProxyType(dict(self.plan)))
        object.__setattr__(
            self,
            "rows",
            tuple(MappingProxyType(dict(value)) for value in self.rows),
        )
        object.__setattr__(self, "tensors", MappingProxyType(dict(self.tensors)))


@dataclass(frozen=True, slots=True)
class VerifiedE10EarlyProbeSelection:
    directory: Path
    manifest: Mapping[str, Any]
    selected_probe: CalibratedProbe
    selected_probe_path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))


def _strict_json(path: Path, context: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FrozenArtifactError(f"{context} must be one regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read {context}: {exc}") from exc
    if not isinstance(value, dict):
        raise FrozenArtifactError(f"{context} must contain a JSON object")
    return value


def _runtime_identity(path: Path) -> tuple[str, str, str]:
    value = _strict_json(path, "E10 early-probe runtime attestation")
    body = dict(value)
    digest = body.pop("runtime_attestation_digest", None)
    identity = body.get("runtime_identity")
    public_key = body.get("execution_public_key")
    if (
        set(body)
        != {
            "schema_version",
            "execution_public_key",
            "runtime_identity",
            "runtime_identity_digest",
        }
        or body.get("schema_version") != 1
        or not isinstance(identity, dict)
        or not isinstance(public_key, str)
        or len(public_key) != 64
        or body.get("runtime_identity_digest") != stable_hash(identity)
        or digest != stable_hash(body)
    ):
        raise FrozenArtifactError("E10 early-probe runtime attestation is invalid")
    return sha256_file(path), public_key, str(body["runtime_identity_digest"])


def _question_fingerprint(question: Question) -> str:
    return stable_hash(
        {
            "question_id": question.question_id,
            "benchmark": question.benchmark,
            "text": question.text,
            "aliases": list(question.aliases),
            "split": question.split,
            "metadata": dict(question.metadata),
        }
    )


def _record_fingerprint(record: GenerationRecord) -> str:
    return stable_hash(record.to_dict())


def _one_condition(
    ledger: PhaseRunLedger,
    *,
    prompt_id: str,
    partition: str,
    method: str,
    condition_id: str | None = None,
) -> Any:
    values = tuple(
        condition
        for condition in ledger.contract.conditions
        if condition.system_prompt_id == prompt_id
        and condition.benchmark == "triviaqa"
        and condition.partition == partition
        and condition.steering_method == method
        and (condition_id is None or condition.condition_id == condition_id)
    )
    if len(values) != 1:
        raise FrozenArtifactError(
            f"E10 early-probe source requires one {method} {partition} condition"
        )
    return values[0]


def _records_for_condition(
    ledger: PhaseRunLedger,
    condition_id: str,
) -> tuple[GenerationRecord, ...]:
    values = tuple(
        record for record in ledger.records() if record.condition_id == condition_id
    )
    if len(values) != 5_000 or len({value.question_id for value in values}) != 5_000:
        raise FrozenArtifactError("E10 early-probe source does not contain exactly 5,000 rows")
    return values


def _reviewed_questions_from_e1(
    ledger: PhaseRunLedger,
    *,
    split_manifest_digest: str | None,
    partitions: Sequence[str] = ("T-controller", _DEV),
) -> Mapping[str, tuple[Question, ...]]:
    selected_partitions = tuple(partitions)
    if (
        not selected_partitions
        or len(set(selected_partitions)) != len(selected_partitions)
        or not set(selected_partitions)
        <= {"T-steer", "T-controller", _DEV, "T-test"}
    ):
        raise DataValidationError("E1 reviewed split partition request is invalid")
    evidence = _strict_json(
        ledger.directory / "creation-evidence.json",
        "E1 creation evidence",
    )
    inputs = evidence.get("input_artifacts")
    descriptor = inputs.get("deduplicated_splits") if isinstance(inputs, Mapping) else None
    if not isinstance(descriptor, Mapping) or set(descriptor) != {
        "location",
        "fingerprint",
    }:
        raise FrozenArtifactError("E1 creation evidence lacks reviewed splits")
    location = descriptor.get("location")
    fingerprint = descriptor.get("fingerprint")
    if not isinstance(location, str) or not isinstance(fingerprint, str):
        raise FrozenArtifactError("E1 reviewed-split identity is invalid")
    source = Path(location)
    if not source.is_absolute():
        source = (ledger.directory / source).resolve()
    manifest = validate_reviewed_split_snapshot(source)
    if (
        sha256_path(source) != fingerprint
        or ledger.contract.input_fingerprints.get("deduplicated_splits") != fingerprint
        or (
            split_manifest_digest is not None
            and manifest.get("manifest_digest") != split_manifest_digest
        )
    ):
        raise FrozenArtifactError("E10 early probe differs from the E1 reviewed splits")
    return MappingProxyType(
        {
            partition: tuple(read_questions(source / f"{partition}.jsonl"))
            for partition in selected_partitions
        }
    )


def _source_materials(
    *,
    study: StudyProtocol,
    e1_run: str | Path,
    e8_run: str | Path,
    questions_by_partition: Mapping[str, Sequence[Question]],
    prompt: PromptSpec,
    controller_artifact: str | Path,
    selection_provenance: Mapping[str, Any],
    split_manifest_digest: str,
) -> tuple[
    AdaptiveController,
    tuple[dict[str, Any], ...],
    dict[str, Question],
    dict[str, GenerationRecord],
    dict[str, Any],
]:
    if set(questions_by_partition) != {"T-controller", _DEV}:
        raise DataValidationError("E10 early probe requires exact controller and dev questions")
    if len(split_manifest_digest) != 64:
        raise DataValidationError("E10 early probe requires a split-manifest SHA-256")
    selected_prompt = selection_provenance.get("selected_prompt_id")
    selected_ids = selection_provenance.get("selected_e8_condition_ids")
    controller_sha = selection_provenance.get("e9_selected_controller_sha256")
    if (
        selected_prompt != prompt.prompt_id
        or not isinstance(selected_ids, Mapping)
        or not isinstance(controller_sha, str)
    ):
        raise DataValidationError("E10 early-probe selection provenance is incomplete")
    normalized = validate_active_study_artifact_paths(
        {
            "E1 phase ledger": e1_run,
            "E8 phase ledger": e8_run,
            "E5 selected controller": controller_artifact,
        }
    )
    e1 = PhaseRunLedger.open(normalized["E1 phase ledger"], study=study)
    e8 = PhaseRunLedger.open(normalized["E8 phase ledger"], study=study)
    if (
        e1.verify_complete().phase is not ExperimentPhase.E1
        or e8.verify_complete().phase is not ExperimentPhase.E8
    ):
        raise FrozenArtifactError("E10 early-probe sources are not complete E1/E8 ledgers")
    e1_condition = _one_condition(
        e1, prompt_id=prompt.prompt_id, partition="T-controller", method="M0"
    )
    e8_condition = _one_condition(
        e8,
        prompt_id=prompt.prompt_id,
        partition=_DEV,
        method="M3",
        condition_id=str(selected_ids.get("M3")),
    )
    controller_path = normalized["E5 selected controller"]
    if sha256_path(controller_path) != controller_sha:
        raise FrozenArtifactError("E10 early-probe controller differs from E8/E9 promotion")
    controller = load_adaptive_controller(controller_path)
    schema = controller.risk_probe.training_schema
    prompt_sha = hashlib.sha256(prompt.text.encode()).hexdigest()
    if (
        schema.source_identity()
        != {
            "model_repository": e8_condition.model_repository,
            "model_revision": e8_condition.model_revision,
            "runtime": e8_condition.runtime.value,
            "quantization": e8_condition.quantization,
            "prompt_id": prompt.prompt_id,
            "prompt_sha256": prompt_sha,
        }
        or e8_condition.adaptive_policy is None
        or e8_condition.adaptive_policy.controller_artifact_sha256 != controller_sha
    ):
        raise FrozenArtifactError("E10 early-probe controller source identity differs")
    trusted_questions = _reviewed_questions_from_e1(
        e1,
        split_manifest_digest=split_manifest_digest,
    )
    supplied_questions = {
        partition: tuple(questions_by_partition[partition])
        for partition in ("T-controller", _DEV)
    }
    if any(
        tuple(_question_fingerprint(value) for value in supplied_questions[partition])
        != tuple(_question_fingerprint(value) for value in trusted_questions[partition])
        for partition in ("T-controller", _DEV)
    ):
        raise DataValidationError(
            "E10 early-probe questions differ from the frozen reviewed split bundle"
        )
    question_values = dict(trusted_questions)
    if any(
        len(values) != 5_000
        or len({value.question_id for value in values}) != 5_000
        or any(value.benchmark != "triviaqa" for value in values)
        for values in question_values.values()
    ):
        raise DataValidationError("E10 early-probe question partitions require 5,000 TriviaQA rows")
    e1_records = _records_for_condition(e1, e1_condition.condition_id)
    e8_records = _records_for_condition(e8, e8_condition.condition_id)
    record_sets = {
        "T-controller": {value.question_id: value for value in e1_records},
        _DEV: {value.question_id: value for value in e8_records},
    }
    questions: dict[str, Question] = {}
    records: dict[str, GenerationRecord] = {}
    for partition, values in question_values.items():
        expected_ids = tuple(
            e1.contract.question_ids_by_benchmark["triviaqa"]
            if partition == "T-controller"
            else e8.contract.question_ids_by_benchmark["triviaqa"]
        )
        if tuple(value.question_id for value in values) != expected_ids:
            raise DataValidationError(
                f"E10 early-probe {partition} questions differ from the source ledger"
            )
        for value in values:
            key = f"{partition}:{value.question_id}"
            questions[key] = value
            records[key] = record_sets[partition][value.question_id]
    subdivisions = controller_feature_partitions(
        question_values["T-controller"], calibration_rows=1_000, seed=17
    )
    groups = {
        partition: semantic_group_ids(values)
        for partition, values in question_values.items()
    }
    schedule: list[dict[str, Any]] = []
    sequence = 0
    for partition in ("T-controller", _DEV):
        for question in question_values[partition]:
            key = f"{partition}:{question.question_id}"
            record = records[key]
            feature_partition = (
                subdivisions[question.question_id]
                if partition == "T-controller"
                else _DEV
            )
            schedule.append(
                {
                    "sequence": sequence,
                    "source_phase": "E1" if partition == "T-controller" else "E8",
                    "source_partition": partition,
                    "feature_partition": feature_partition,
                    "condition_id": record.condition_id,
                    "question_id": question.question_id,
                    "question_sha256": _question_fingerprint(question),
                    "record_sha256": _record_fingerprint(record),
                    "outcome": record.outcome.value,
                    "semantic_group_id": groups[partition][question.question_id],
                }
            )
            sequence += 1
    if (
        len(schedule) != 10_000
        or sum(value["feature_partition"] == _TRAIN for value in schedule) != 4_000
        or sum(value["feature_partition"] == _CALIBRATION for value in schedule) != 1_000
        or sum(value["feature_partition"] == _DEV for value in schedule) != 5_000
    ):
        raise DataValidationError("E10 early-probe schedule partitioning differs")
    identities = {
        "e1_completion_digest": e1.verify_complete().completion_digest,
        "e8_completion_digest": e8.verify_complete().completion_digest,
        "e1_condition_id": e1_condition.condition_id,
        "e8_condition_id": e8_condition.condition_id,
        "controller_sha256": controller_sha,
        "split_manifest_digest": split_manifest_digest,
        "prompt_id": prompt.prompt_id,
        "prompt_sha256": prompt_sha,
        "source_schema": schema.to_dict(),
        "selection_provenance_digest": stable_hash(dict(selection_provenance)),
    }
    return controller, tuple(schedule), questions, records, identities


def _capture_plan(
    *,
    schedule: Sequence[Mapping[str, Any]],
    identities: Mapping[str, Any],
    runtime_artifact: str | Path,
) -> dict[str, Any]:
    runtime_path = Path(runtime_artifact).resolve()
    runtime_sha, execution_key, runtime_identity_digest = _runtime_identity(runtime_path)
    body = {
        "schema_version": 1,
        "phase": "E10-development-freeze",
        "purpose": "post-first-token-and-short-block-CIA-probe-selection",
        "selection_rule": _SELECTION_RULE,
        "candidate_activation_kinds": [value.value for value in _KINDS],
        "candidate_probe_kinds": [value.value for value in _PROBES],
        "candidate_calibrators": [value.value for value in _CALIBRATORS],
        "task": _TASK.value,
        "schedule": [dict(value) for value in schedule],
        "source_identities": dict(identities),
        "runtime_artifact_sha256": runtime_sha,
        "runtime_identity_digest": runtime_identity_digest,
        "execution_public_key": execution_key,
    }
    return {**body, "capture_plan_identity": stable_hash(body)}


def derive_e10_early_probe_capture_plan(
    *,
    study: StudyProtocol,
    e1_run: str | Path,
    e8_run: str | Path,
    prompt: PromptSpec,
    controller_artifact: str | Path,
    selection_provenance: Mapping[str, Any],
    split_manifest_digest: str,
    runtime_artifact: str | Path,
) -> Mapping[str, Any]:
    """Reconstruct the complete 10,000-row plan from immutable prerequisites."""

    normalized = validate_active_study_artifact_paths(
        {
            "E1 phase ledger": e1_run,
            "E8 phase ledger": e8_run,
            "E5 selected controller": controller_artifact,
            "E6 runtime attestation": runtime_artifact,
        }
    )
    e1 = PhaseRunLedger.open(normalized["E1 phase ledger"], study=study)
    questions = _reviewed_questions_from_e1(
        e1,
        split_manifest_digest=split_manifest_digest,
    )
    _controller, schedule, _questions, _records, identities = _source_materials(
        study=study,
        e1_run=normalized["E1 phase ledger"],
        e8_run=normalized["E8 phase ledger"],
        questions_by_partition=questions,
        prompt=prompt,
        controller_artifact=normalized["E5 selected controller"],
        selection_provenance=selection_provenance,
        split_manifest_digest=split_manifest_digest,
    )
    return MappingProxyType(
        _capture_plan(
            schedule=schedule,
            identities=identities,
            runtime_artifact=normalized["E6 runtime attestation"],
        )
    )


def prepare_e10_early_probe_capture(
    directory: str | Path,
    *,
    study: StudyProtocol,
    e1_run: str | Path,
    e8_run: str | Path,
    questions_by_partition: Mapping[str, Sequence[Question]],
    prompt: PromptSpec,
    controller_artifact: str | Path,
    selection_provenance: Mapping[str, Any],
    split_manifest_digest: str,
    runtime_artifact: str | Path,
) -> Mapping[str, Any]:
    """Freeze the exact 4k/1k/5k early-token capture schedule."""

    normalized = validate_active_study_artifact_paths(
        {
            "E10 early-probe capture": directory,
            "E6 runtime attestation": runtime_artifact,
        }
    )
    destination = normalized["E10 early-probe capture"]
    if destination.exists() or destination.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite E10 early capture: {destination}")
    _controller, schedule, _questions, _records, identities = _source_materials(
        study=study,
        e1_run=e1_run,
        e8_run=e8_run,
        questions_by_partition=questions_by_partition,
        prompt=prompt,
        controller_artifact=controller_artifact,
        selection_provenance=selection_provenance,
        split_manifest_digest=split_manifest_digest,
    )
    plan = _capture_plan(
        schedule=schedule,
        identities=identities,
        runtime_artifact=normalized["E6 runtime attestation"],
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        (stage / "shards").mkdir()
        (stage / "run.lock").touch()
        (stage / "plan.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return MappingProxyType(plan)


def _early_features(
    schema: ActivationFeatureSchema,
    activations: Mapping[ActivationSite, Mapping[int, np.ndarray[Any, Any]]],
    *,
    limit: int,
) -> torch.Tensor:
    try:
        pooled = {
            site: [
                np.asarray(activations[site][layer], dtype=np.float32)[:limit].mean(axis=0)
                for layer in schema.layers
            ]
            for site in schema.sites
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise DataValidationError(f"E10 early feature cube differs: {exc}") from exc
    if schema.composition is FeatureComposition.SINGLE_LAYER:
        parts = [values[0] for values in pooled.values()]
    elif schema.composition is FeatureComposition.CONCATENATED_LAYERS:
        parts = [value for values in pooled.values() for value in values]
    elif schema.composition is FeatureComposition.LAYER_DIFFERENCES:
        parts = [
            values[index + 1] - values[index]
            for values in pooled.values()
            for index in range(len(values) - 1)
        ]
    else:  # pragma: no cover - exhaustive enum
        raise DataValidationError("E10 early feature composition is unsupported")
    row = np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)
    if row.shape != (schema.width,) or not np.isfinite(row).all():
        raise DataValidationError("E10 early feature width differs")
    return torch.from_numpy(row.copy())


def _intervention_state(
    controller: AdaptiveController,
    record: GenerationRecord,
    attestor: E6RuntimeAttestor,
) -> Mapping[tuple[ActivationSite, int], VllmResearchInterventionState]:
    action = record.metadata.get("policy_action")
    if action != "intervene":
        if action not in {"release", "abstain"}:
            raise FrozenArtifactError("E8 early-probe row has an invalid policy action")
        return {}
    evidence = record.metadata.get("adaptive_controller_evidence")
    trace = record.metadata.get("intervention_trace")
    if not isinstance(evidence, Mapping) or not isinstance(trace, Mapping):
        raise FrozenArtifactError("E8 early-probe intervention lacks execution evidence")
    raw_features = evidence.get("feature_values")
    if not isinstance(raw_features, list):
        raise FrozenArtifactError("E8 early-probe controller features are invalid")
    features = torch.tensor(raw_features, dtype=torch.float32).reshape(1, -1)
    decision = controller.decide(features)
    selected_layer = int(decision.selected_layers[0])
    eligible = [
        (key, value[0].detach().cpu().float().contiguous())
        for key, value in decision.directions.items()
        if key.layer == selected_layer
    ]
    if not eligible:
        raise FrozenArtifactError("E8 early-probe controller selected no direction")
    selected_key, direction = min(
        eligible,
        key=lambda item: (-float(torch.linalg.vector_norm(item[1])), item[0].site.value),
    )
    values = np.ascontiguousarray(direction.numpy(), dtype=np.float32)
    norm = float(np.linalg.norm(values))
    normalized = np.ascontiguousarray(values / norm, dtype=np.float32)
    if (
        not math.isfinite(norm)
        or norm <= 0
        or record.layer != selected_layer
        or record.site is not selected_key.site
        or record.token_scope is None
        or record.alpha <= 0
        or trace.get("direction_sha256")
        != hashlib.sha256(normalized.tobytes(order="C")).hexdigest()
    ):
        raise FrozenArtifactError("E8 early-probe replay differs from executed intervention")
    state = attestor.runtime.standardized_intervention_state(
        normalized,
        standardized_alpha=record.alpha * norm,
        reference_rms=1.0,
        token_scope=record.token_scope,
    )
    return {(selected_key.site, selected_layer): state}


def _load_capture_rows(
    directory: Path,
) -> tuple[list[dict[str, Any]], dict[ActivationKind, list[torch.Tensor]]]:
    rows: list[dict[str, Any]] = []
    tensors: dict[ActivationKind, list[torch.Tensor]] = {
        kind: [] for kind in _KINDS
    }
    shard_root = directory / "shards"
    paths = sorted(shard_root.iterdir())
    expected_names = [f"shard-{index:05d}" for index in range(len(paths))]
    if (
        [path.name for path in paths] != expected_names
        or any(path.is_symlink() or not path.is_dir() for path in paths)
    ):
        raise FrozenArtifactError("E10 early capture shard numbering differs")
    for shard in paths:
        if {path.name for path in shard.iterdir()} != {
            "features.safetensors",
            "rows.jsonl",
        }:
            raise FrozenArtifactError("E10 early capture shard inventory differs")
        tensor_path = shard / "features.safetensors"
        path = shard / "rows.jsonl"
        values = load_file(tensor_path)
        try:
            shard_rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
        except (OSError, json.JSONDecodeError) as exc:
            raise FrozenArtifactError(f"cannot read E10 early capture shard: {exc}") from exc
        if any(not isinstance(value, dict) for value in shard_rows):
            raise FrozenArtifactError("E10 early capture row schema differs")
        for kind in _KINDS:
            tensor = values.get(kind.value)
            if tensor is None or tensor.ndim != 2 or tensor.shape[0] != len(shard_rows):
                raise FrozenArtifactError("E10 early capture tensor shard differs")
            tensors[kind].append(tensor.detach().cpu().float().contiguous())
        rows.extend(shard_rows)
    return rows, tensors


@contextmanager
def _capture_lock(directory: Path) -> Iterator[None]:
    lock = directory / "run.lock"
    if lock.is_symlink() or not lock.is_file() or lock.stat().st_size != 0:
        raise FrozenArtifactError("E10 early capture lock is invalid")
    with lock.open("r+b", buffering=0) as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FrozenArtifactError("E10 early capture is already running") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run_e10_early_probe_capture(
    directory: str | Path,
    *,
    study: StudyProtocol,
    e1_run: str | Path,
    e8_run: str | Path,
    questions_by_partition: Mapping[str, Sequence[Question]],
    prompt: PromptSpec,
    controller_artifact: str | Path,
    selection_provenance: Mapping[str, Any],
    split_manifest_digest: str,
    runtime_artifact: str | Path,
    attestor: E6RuntimeAttestor,
    shard_rows: int = 32,
    limit: int | None = None,
) -> Mapping[str, Any]:
    """Capture a resumable prefix of the 10,000 frozen early-token replays."""

    if (
        type(attestor) is not E6RuntimeAttestor
        or type(shard_rows) is not int
        or shard_rows <= 0
        or (limit is not None and (type(limit) is not int or limit <= 0))
    ):
        raise ConfigurationError("E10 early capture runtime, shard, or limit is invalid")
    normalized = validate_active_study_artifact_paths(
        {
            "E10 early-probe capture": directory,
            "E6 runtime attestation": runtime_artifact,
        }
    )
    work = normalized["E10 early-probe capture"]
    if work.is_symlink() or not work.is_dir() or {p.name for p in work.iterdir()} != _FILES:
        raise FrozenArtifactError("E10 early capture inventory differs")
    with _capture_lock(work):
        return _run_e10_early_probe_capture_locked(
            work,
            study=study,
            e1_run=e1_run,
            e8_run=e8_run,
            questions_by_partition=questions_by_partition,
            prompt=prompt,
            controller_artifact=controller_artifact,
            selection_provenance=selection_provenance,
            split_manifest_digest=split_manifest_digest,
            runtime_artifact=normalized["E6 runtime attestation"],
            attestor=attestor,
            shard_rows=shard_rows,
            limit=limit,
        )


def _run_e10_early_probe_capture_locked(
    work: Path,
    *,
    study: StudyProtocol,
    e1_run: str | Path,
    e8_run: str | Path,
    questions_by_partition: Mapping[str, Sequence[Question]],
    prompt: PromptSpec,
    controller_artifact: str | Path,
    selection_provenance: Mapping[str, Any],
    split_manifest_digest: str,
    runtime_artifact: Path,
    attestor: E6RuntimeAttestor,
    shard_rows: int,
    limit: int | None,
) -> Mapping[str, Any]:
    """Append shards while holding the process-wide capture lock."""

    prefix = f".{work.name}.early-shard-"
    for value in work.parent.iterdir():
        if value.name.startswith(prefix):
            if value.is_symlink() or not value.is_dir():
                raise FrozenArtifactError("abandoned E10 shard stage is invalid")
            shutil.rmtree(value)
    controller, schedule, questions, records, identities = _source_materials(
        study=study,
        e1_run=e1_run,
        e8_run=e8_run,
        questions_by_partition=questions_by_partition,
        prompt=prompt,
        controller_artifact=controller_artifact,
        selection_provenance=selection_provenance,
        split_manifest_digest=split_manifest_digest,
    )
    expected_plan = _capture_plan(
        schedule=schedule,
        identities=identities,
        runtime_artifact=runtime_artifact,
    )
    plan = _strict_json(work / "plan.json", "E10 early capture plan")
    if plan != expected_plan:
        raise FrozenArtifactError("E10 early capture plan differs from live inputs")
    runtime_sha = attestor.verify_runtime_artifact(
        runtime_artifact
    )
    if (
        runtime_sha != plan["runtime_artifact_sha256"]
        or attestor.execution_public_key != plan["execution_public_key"]
    ):
        raise FrozenArtifactError("E10 early capture runtime differs from its plan")
    existing, _tensors = _load_capture_rows(work)
    next_shard_index = len(tuple((work / "shards").iterdir()))
    if any(value.get("sequence") != index for index, value in enumerate(existing)):
        raise FrozenArtifactError("E10 early capture progress is not a strict prefix")
    pending = schedule[len(existing) :]
    if limit is not None:
        pending = pending[:limit]
    source_schema = controller.risk_probe.training_schema
    completed = 0
    for offset in range(0, len(pending), shard_rows):
        chunk = pending[offset : offset + shard_rows]
        rows: list[dict[str, Any]] = []
        feature_rows: dict[ActivationKind, list[torch.Tensor]] = {
            kind: [] for kind in _KINDS
        }
        for item in chunk:
            key = f"{item['source_partition']}:{item['question_id']}"
            question = questions[key]
            record = records[key]
            rendered = attestor.runtime.render_prompt(
                prompt, question.text, metadata=question.metadata
            )
            if rendered.sha256 != record.rendered_prompt_hash:
                raise FrozenArtifactError("E10 early capture rendered prompt differs")
            states = (
                _intervention_state(controller, record, attestor)
                if item["source_phase"] == "E8"
                else {}
            )
            forced = attestor.runtime.teacher_forced_cube(
                rendered,
                record.raw_output,
                layers=source_schema.layers,
                sites=source_schema.sites,
                intervention_states=states,
            )
            hashes: dict[str, str] = {}
            for kind, count in (
                (ActivationKind.FIRST_GENERATED, 1),
                (ActivationKind.FIRST_FOUR_GENERATED, 4),
            ):
                tensor = _early_features(source_schema, forced.activations, limit=count)
                feature_rows[kind].append(tensor)
                hashes[kind.value] = hashlib.sha256(
                    tensor.numpy().tobytes(order="C")
                ).hexdigest()
            body = {
                **dict(item),
                "capture_plan_identity": plan["capture_plan_identity"],
                "response_text_sha256": forced.response_text_sha256,
                "response_token_ids_sha256": forced.response_token_ids_sha256,
                "feature_sha256s": hashes,
                "runtime_artifact_sha256": runtime_sha,
                "execution_public_key": attestor.execution_public_key,
            }
            rows.append(
                {
                    **body,
                    "runtime_signature": attestor._sign(body),
                    "row_digest": stable_hash(body),
                }
            )
        index = next_shard_index
        shard_stage = Path(
            tempfile.mkdtemp(prefix=prefix, dir=work.parent)
        )
        try:
            feature_path = shard_stage / "features.safetensors"
            save_file(
                {kind.value: torch.stack(feature_rows[kind]) for kind in _KINDS},
                feature_path,
            )
            row_path = shard_stage / "rows.jsonl"
            row_path.write_text(
                "".join(json.dumps(value, sort_keys=True) + "\n" for value in rows),
                encoding="utf-8",
            )
            os.replace(shard_stage, work / "shards" / f"shard-{index:05d}")
        finally:
            if shard_stage.exists():
                shutil.rmtree(shard_stage)
        existing.extend(rows)
        completed += len(rows)
        next_shard_index += 1
    return MappingProxyType(
        {"captured_this_call": completed, "captured_total": len(existing), "expected": 10_000}
    )


def verify_e10_early_probe_capture(
    directory: str | Path,
    *,
    require_complete: bool = True,
) -> VerifiedE10EarlyCapture:
    """Replay shard hashes and runtime signatures for a frozen early capture."""

    source = validate_active_study_artifact_paths(
        {"E10 early-probe capture": directory}
    )["E10 early-probe capture"]
    plan = _strict_json(source / "plan.json", "E10 early capture plan")
    body = dict(plan)
    identity = body.pop("capture_plan_identity", None)
    if identity != stable_hash(body) or len(body.get("schedule", [])) != 10_000:
        raise FrozenArtifactError("E10 early capture plan identity differs")
    rows, tensor_shards = _load_capture_rows(source)
    if require_complete and len(rows) != 10_000:
        raise FrozenArtifactError("E10 early capture is incomplete")
    if len(rows) > 10_000:
        raise FrozenArtifactError("E10 early capture exceeds its schedule")
    try:
        public_key = Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(str(plan["execution_public_key"]))
        )
    except ValueError as exc:
        raise FrozenArtifactError("E10 early capture execution key is invalid") from exc
    tensors = {
        kind: (
            torch.cat(tensor_shards[kind], dim=0)
            if tensor_shards[kind]
            else torch.empty((0, int(plan["source_identities"]["source_schema"]["width"])))
        )
        for kind in _KINDS
    }
    for index, row in enumerate(rows):
        expected = dict(plan["schedule"][index])
        signature = row.get("runtime_signature")
        digest = row.get("row_digest")
        signed = {
            key: value
            for key, value in row.items()
            if key not in {"runtime_signature", "row_digest"}
        }
        if (
            {key: signed[key] for key in expected} != expected
            or signed.get("capture_plan_identity") != identity
            or digest != stable_hash(signed)
            or not isinstance(signature, str)
        ):
            raise FrozenArtifactError("E10 early capture row identity differs")
        try:
            public_key.verify(bytes.fromhex(signature), canonical_json(signed).encode())
        except (ValueError, InvalidSignature) as exc:
            raise FrozenArtifactError("E10 early capture signature is invalid") from exc
        for kind in _KINDS:
            value = tensors[kind][index]
            if hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest() != signed[
                "feature_sha256s"
            ][kind.value]:
                raise FrozenArtifactError("E10 early capture feature hash differs")
    return VerifiedE10EarlyCapture(source, plan, tuple(rows), tensors)


def _schema(
    capture: VerifiedE10EarlyCapture,
    *,
    kind: ActivationKind,
    partition: str,
) -> ActivationFeatureSchema:
    raw = dict(capture.plan["source_identities"]["source_schema"])
    return ActivationFeatureSchema(
        benchmark="triviaqa",
        partition=partition,
        split_manifest_digest=str(capture.plan["source_identities"]["split_manifest_digest"]),
        model_repository=str(raw["model_repository"]),
        model_revision=str(raw["model_revision"]),
        runtime=Runtime(raw["runtime"]),
        quantization=str(raw["quantization"]),
        prompt_id=str(raw["prompt_id"]),
        prompt_sha256=str(raw["prompt_sha256"]),
        activation_kind=kind,
        layers=tuple(raw["layers"]),
        sites=tuple(ActivationSite(value) for value in raw["sites"]),
        composition=FeatureComposition(raw["composition"]),
        width=int(raw["width"]),
        token_scope=(
            TokenScope.FIRST_GENERATED
            if kind is ActivationKind.FIRST_GENERATED
            else TokenScope.FIRST_FOUR
        ),
    )


def _datasets(
    capture: VerifiedE10EarlyCapture,
    kind: ActivationKind,
) -> Mapping[str, ProbeDataset]:
    values: dict[str, ProbeDataset] = {}
    for partition in (_TRAIN, _CALIBRATION, _DEV):
        indices = [
            index
            for index, row in enumerate(capture.rows)
            if row["feature_partition"] == partition
        ]
        values[partition] = ProbeDataset(
            question_ids=tuple(str(capture.rows[index]["question_id"]) for index in indices),
            features=capture.tensors[kind][indices],
            outcomes=tuple(Outcome(capture.rows[index]["outcome"]) for index in indices),
            group_ids=tuple(
                str(capture.rows[index]["semantic_group_id"]) for index in indices
            ),
            feature_schema=_schema(capture, kind=kind, partition=partition),
        )
    return MappingProxyType(values)


def _metrics(metrics: ProbeMetrics) -> dict[str, Any]:
    return {
        "macro_auroc": metrics.macro_auroc,
        "macro_f1": metrics.macro_f1,
        "brier_score": metrics.brier_score,
        "expected_calibration_error": metrics.expected_calibration_error,
        "per_class_auroc": dict(metrics.per_class_auroc),
    }


def _candidate_config(
    kind: ActivationKind,
    probe_kind: ProbeKind,
    calibration: CalibrationKind,
) -> dict[str, Any]:
    return {
        "activation_kind": kind.value,
        "probe_kind": probe_kind.value,
        "calibration": calibration.value,
        "training": {
            "hidden_width": 64,
            "epochs": 400,
            "learning_rate": 0.03,
            "weight_decay": 1e-4,
            "class_balanced": True,
            "seed": 17,
        },
    }


def _candidate_key(value: Mapping[str, Any]) -> tuple[float, float, float, str]:
    metrics = value["dev_metrics"]
    if not isinstance(metrics, Mapping) or not isinstance(
        metrics.get("per_class_auroc"), Mapping
    ):
        raise FrozenArtifactError("E10 early-probe candidate metrics are invalid")
    return (
        -float(metrics["per_class_auroc"][Outcome.INCORRECT.value]),
        -float(metrics["macro_auroc"]),
        float(metrics["expected_calibration_error"]),
        stable_hash(value["config"]),
    )


def fit_e10_early_probe_selection(
    directory: str | Path,
    *,
    capture_directory: str | Path,
) -> VerifiedE10EarlyProbeSelection:
    """Fit the 2x2x2 candidate grid and freeze the E8-dev-selected winner."""

    normalized = validate_active_study_artifact_paths(
        {"E10 early-probe selection": directory, "E10 early-probe capture": capture_directory}
    )
    destination = normalized["E10 early-probe selection"]
    if destination.exists() or destination.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite E10 early selection: {destination}")
    capture = verify_e10_early_probe_capture(
        normalized["E10 early-probe capture"], require_complete=True
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        shutil.copytree(capture.directory, stage / "capture")
        candidates: list[dict[str, Any]] = []
        for kind in _KINDS:
            data = _datasets(capture, kind)
            for probe_kind in _PROBES:
                for calibration in _CALIBRATORS:
                    config = ProbeTrainingConfig(
                        kind=probe_kind,
                        hidden_width=64,
                        epochs=400,
                        learning_rate=0.03,
                        weight_decay=1e-4,
                        class_balanced=True,
                        seed=17,
                    )
                    candidate_config = _candidate_config(
                        kind, probe_kind, calibration
                    )
                    identifier = stable_hash(candidate_config)[:16]
                    probe = fit_calibrated_probe(
                        data[_TRAIN],
                        data[_CALIBRATION],
                        task=_TASK,
                        training_config=config,
                        calibration_kind=calibration,
                    )
                    artifact = stage / "candidates" / identifier
                    save_calibrated_probe(artifact, probe)
                    metrics = evaluate_probe(probe, data[_DEV])
                    candidates.append(
                        {
                            "identifier": identifier,
                            "config": candidate_config,
                            "artifact": f"candidates/{identifier}",
                            "artifact_sha256": sha256_path(artifact),
                            "dev_metrics": _metrics(metrics),
                        }
                    )
        winner = min(candidates, key=_candidate_key)
        body = {
            "schema_version": 1,
            "purpose": "frozen-M6-early-token-CIA-probe-selection",
            "capture_sha256": sha256_path(stage / "capture"),
            "capture_plan_identity": capture.plan["capture_plan_identity"],
            "selection_partition": _DEV,
            "selection_rule": _SELECTION_RULE,
            "candidates": candidates,
            "selected_identifier": winner["identifier"],
            "selected_probe_sha256": winner["artifact_sha256"],
            "selected_metrics": winner["dev_metrics"],
        }
        manifest = {**body, "manifest_digest": stable_hash(body)}
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return load_e10_early_probe_selection(destination)


def load_e10_early_probe_selection(
    directory: str | Path,
    *,
    capture_directory: str | Path | None = None,
) -> VerifiedE10EarlyProbeSelection:
    """Verify every candidate metric and the deterministic early-probe promotion."""

    source = Path(directory).resolve()
    manifest = _strict_json(source / "manifest.json", "E10 early-probe selection")
    body = dict(manifest)
    digest = body.pop("manifest_digest", None)
    candidates = body.get("candidates")
    if (
        digest != stable_hash(body)
        or body.get("schema_version") != 1
        or body.get("selection_rule") != _SELECTION_RULE
        or not isinstance(candidates, list)
        or len(candidates) != 8
    ):
        raise FrozenArtifactError("E10 early-probe selection identity differs")
    if {path.name for path in source.iterdir()} != {
        "manifest.json",
        "capture",
        "candidates",
    }:
        raise FrozenArtifactError("E10 early-probe selection inventory differs")
    expected_configs = {
        stable_hash(_candidate_config(kind, probe_kind, calibration))[:16]:
        _candidate_config(kind, probe_kind, calibration)
        for kind in _KINDS
        for probe_kind in _PROBES
        for calibration in _CALIBRATORS
    }
    observed_configs = {
        str(value.get("identifier")): value.get("config")
        for value in candidates
        if isinstance(value, dict)
    }
    if observed_configs != expected_configs or {
        path.name for path in (source / "candidates").iterdir()
    } != set(expected_configs):
        raise FrozenArtifactError("E10 early-probe candidate grid differs")
    capture_source = (
        Path(capture_directory).resolve()
        if capture_directory is not None
        else source / "capture"
    )
    capture = verify_e10_early_probe_capture(capture_source, require_complete=True)
    if (
        body.get("capture_sha256") != sha256_path(capture.directory)
        or body.get("capture_plan_identity") != capture.plan["capture_plan_identity"]
    ):
        raise FrozenArtifactError("E10 early-probe selection uses another capture")
    replayed: list[dict[str, Any]] = []
    for descriptor in candidates:
        if not isinstance(descriptor, dict):
            raise FrozenArtifactError("E10 early-probe candidate descriptor is invalid")
        identifier = str(descriptor.get("identifier"))
        if descriptor.get("artifact") != f"candidates/{identifier}":
            raise FrozenArtifactError("E10 early-probe candidate path differs")
        artifact = source / str(descriptor["artifact"])
        probe = load_calibrated_probe(artifact)
        if sha256_path(artifact) != descriptor.get("artifact_sha256"):
            raise FrozenArtifactError("E10 early-probe candidate changed")
        kind = ActivationKind(descriptor["config"]["activation_kind"])
        metrics = _metrics(evaluate_probe(probe, _datasets(capture, kind)[_DEV]))
        if metrics != descriptor.get("dev_metrics"):
            raise FrozenArtifactError("E10 early-probe candidate metrics differ")
        replayed.append(descriptor)
    winner = min(replayed, key=_candidate_key)
    if (
        body.get("selected_identifier") != winner["identifier"]
        or body.get("selected_probe_sha256") != winner["artifact_sha256"]
        or body.get("selected_metrics") != winner["dev_metrics"]
    ):
        raise FrozenArtifactError("E10 early-probe winner differs from deterministic replay")
    selected_path = source / str(winner["artifact"])
    return VerifiedE10EarlyProbeSelection(
        source,
        manifest,
        load_calibrated_probe(selected_path),
        selected_path,
    )

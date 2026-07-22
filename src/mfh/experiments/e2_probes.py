"""Frozen E2 separability screening, calibrated probes, and gate bundle."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from mfh.contracts import ActivationSite, Outcome, Runtime, TokenScope
from mfh.errors import ConfigurationError, DataValidationError, FrozenArtifactError
from mfh.experiments.activation_store import (
    ActivationCaptureRow,
    VerifiedActivationStore,
    iter_activation_shards,
    verify_activation_store,
)
from mfh.experiments.e2_controller_inputs import (
    E2ControllerInputView,
    build_e2_controller_input_datasets,
    controller_input_views,
)
from mfh.experiments.e2_schedule import E2_LAYERS, E2_SITES, VerifiedE2Workspace
from mfh.experiments.model_selection import validate_active_study_artifact_paths
from mfh.methods.features import (
    ActivationFeatureSchema,
    ActivationKind,
    FeatureComposition,
)
from mfh.methods.probes import (
    CalibratedProbe,
    CalibrationKind,
    IsotonicCalibrator,
    ProbeDataset,
    ProbeKind,
    ProbeMetrics,
    ProbeTask,
    ProbeTrainingConfig,
    TemperatureCalibrator,
    encode_probe_task,
    evaluate_probe,
    fit_isotonic,
    fit_probe_state,
    fit_temperature,
    load_calibrated_probe,
    save_calibrated_probe,
)
from mfh.provenance import sha256_file, sha256_path, stable_hash

_TASKS = (
    ProbeTask.CORRECT_INCORRECT,
    ProbeTask.ATTEMPT_ABSTENTION,
    ProbeTask.CORRECT_INCORRECT_ABSTENTION,
    ProbeTask.FORCED_CORRECT_INCORRECT,
)
_KINDS = (ProbeKind.LOGISTIC, ProbeKind.TWO_LAYER_MLP)
_CALIBRATORS = (CalibrationKind.TEMPERATURE, CalibrationKind.ISOTONIC)


@dataclass(frozen=True, slots=True)
class E2ProbeProtocol:
    candidate_layers: tuple[int, ...] = E2_LAYERS
    candidate_sites: tuple[ActivationSite, ...] = E2_SITES
    screening_epochs: int = 120
    final_epochs: int = 400
    mlp_hidden_width: int = 64
    learning_rate: float = 0.03
    weight_decay: float = 1e-4
    seed: int = 17
    minimum_material_gain: float = 0.02
    schema_version: int = 1

    def __post_init__(self) -> None:
        layers = tuple(self.candidate_layers)
        sites = tuple(self.candidate_sites)
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or not layers
            or any(type(value) is not int or value not in E2_LAYERS for value in layers)
            or len(set(layers)) != len(layers)
            or not sites
            or any(
                not isinstance(value, ActivationSite) or value not in E2_SITES
                for value in sites
            )
            or len(set(sites)) != len(sites)
            or any(
                type(value) is not int or value <= 0
                for value in (
                    self.screening_epochs,
                    self.final_epochs,
                    self.mlp_hidden_width,
                )
            )
            or type(self.seed) is not int
            or self.seed < 0
        ):
            raise DataValidationError("E2 probe protocol geometry or integer fields are invalid")
        for value in (self.learning_rate, self.weight_decay, self.minimum_material_gain):
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
            ):
                raise DataValidationError("E2 probe protocol numeric fields are invalid")
        if (
            self.learning_rate <= 0
            or self.weight_decay < 0
            or self.minimum_material_gain <= 0
        ):
            raise DataValidationError("E2 probe protocol numeric ranges are invalid")
        object.__setattr__(self, "candidate_layers", layers)
        object.__setattr__(self, "candidate_sites", sites)

    @property
    def scientific_eligible(self) -> bool:
        return self == E2ProbeProtocol()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_layers": list(self.candidate_layers),
            "candidate_sites": [value.value for value in self.candidate_sites],
            "screening_epochs": self.screening_epochs,
            "final_epochs": self.final_epochs,
            "mlp_hidden_width": self.mlp_hidden_width,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
            "minimum_material_gain": self.minimum_material_gain,
        }


@dataclass(frozen=True, slots=True)
class E2FeatureView:
    layer: int
    site: ActivationSite

    def __post_init__(self) -> None:
        if type(self.layer) is not int or self.layer not in E2_LAYERS:
            raise DataValidationError("E2 feature view layer is invalid")
        if not isinstance(self.site, ActivationSite) or self.site not in E2_SITES:
            raise DataValidationError("E2 feature view site is invalid")

    @property
    def identifier(self) -> str:
        return f"{self.site.value}-layer-{self.layer:02d}"

    def to_dict(self) -> dict[str, Any]:
        return {"layer": self.layer, "site": self.site.value, "identifier": self.identifier}


@dataclass(frozen=True, slots=True)
class E2FeatureDataset:
    probe: ProbeDataset
    rows: tuple[ActivationCaptureRow, ...]

    def __post_init__(self) -> None:
        if len(self.rows) != len(self.probe.question_ids):
            raise DataValidationError("E2 feature rows and probe dataset differ")


@dataclass(frozen=True, slots=True)
class VerifiedE2ProbeBundle:
    directory: Path
    plan_identity: str
    manifest_digest: str
    selected_views: Mapping[ProbeTask, E2FeatureView]
    selected_gate_artifact: str
    gate_passed: bool
    gate_probe_auroc: float
    gate_baseline_auroc: float
    controller_input_artifacts: Mapping[FeatureComposition, str]
    scientific_eligible: bool


def _task_labels(task: ProbeTask) -> tuple[str, ...]:
    if task in {ProbeTask.CORRECT_INCORRECT, ProbeTask.FORCED_CORRECT_INCORRECT}:
        return (Outcome.CORRECT.value, Outcome.INCORRECT.value)
    if task is ProbeTask.ATTEMPT_ABSTENTION:
        return ("attempt", "abstention")
    return (Outcome.CORRECT.value, Outcome.INCORRECT.value, Outcome.ABSTENTION.value)


def _binary_auroc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    if len(labels) != len(scores) or not labels:
        raise DataValidationError("E2 AUROC inputs are empty or misaligned")
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise DataValidationError("E2 AUROC requires positive and negative rows")
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    rank_sum = sum(rank for rank, positive in zip(ranks, labels, strict=True) if positive)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _prompt_for_task(task: ProbeTask) -> str:
    return "P3-forced-answer" if task is ProbeTask.FORCED_CORRECT_INCORRECT else "P0-neutral"


def _partition_rows(
    workspace: VerifiedE2Workspace,
    *,
    partition: str,
    prompt_id: str,
    view: E2FeatureView,
    verified_store: VerifiedActivationStore | None = None,
) -> tuple[tuple[ActivationCaptureRow, ...], np.ndarray[Any, Any]]:
    site_index = workspace.activation_spec.sites.index(view.site)
    layer_index = workspace.activation_spec.layers.index(view.layer)
    selected_rows: list[ActivationCaptureRow] = []
    selected_values: list[np.ndarray[Any, Any]] = []
    sequence = 0
    for rows, activations in iter_activation_shards(
        workspace.directory / "activations",
        expected_spec=workspace.activation_spec,
        verified_store=verified_store,
    ):
        for offset, row in enumerate(rows):
            schedule = workspace.schedule[sequence]
            if (
                row.question_id != schedule.question_id
                or row.benchmark != schedule.benchmark
                or row.partition != schedule.feature_partition
                or row.prompt_id != schedule.prompt_id
                or row.semantic_group_id != schedule.semantic_group_id
                or (schedule.outcome is not None and row.outcome is not schedule.outcome)
            ):
                raise FrozenArtifactError("E2 probe input differs from its capture schedule")
            if row.partition == partition and row.prompt_id == prompt_id:
                selected_rows.append(row)
                selected_values.append(
                    np.asarray(
                        activations[offset, site_index, layer_index, :], dtype=np.float32
                    ).copy()
                )
            sequence += 1
    if sequence != len(workspace.schedule) or not selected_rows:
        raise FrozenArtifactError("E2 probe input activation store is incomplete or empty")
    return tuple(selected_rows), np.stack(selected_values, axis=0)


def build_e2_probe_dataset(
    workspace: VerifiedE2Workspace,
    *,
    partition: str,
    prompt_id: str,
    view: E2FeatureView,
    split_manifest_digest: str,
    prompt_template_sha256: str,
    verified_store: VerifiedActivationStore | None = None,
) -> E2FeatureDataset:
    rows, values = _partition_rows(
        workspace,
        partition=partition,
        prompt_id=prompt_id,
        view=view,
        verified_store=verified_store,
    )
    benchmarks = {row.benchmark for row in rows}
    if len(benchmarks) != 1:
        raise DataValidationError("E2 probe partition mixes benchmarks")
    schema = ActivationFeatureSchema(
        benchmark=next(iter(benchmarks)),
        partition=partition,
        split_manifest_digest=split_manifest_digest,
        model_repository=workspace.activation_spec.model_repository,
        model_revision=workspace.activation_spec.model_revision,
        runtime=Runtime.VLLM,
        quantization=workspace.activation_spec.quantization,
        prompt_id=prompt_id,
        prompt_sha256=prompt_template_sha256,
        activation_kind=ActivationKind.FINAL_PROMPT,
        layers=(view.layer,),
        sites=(view.site,),
        composition=FeatureComposition.SINGLE_LAYER,
        width=workspace.activation_spec.hidden_width,
        token_scope=TokenScope.FINAL_PROMPT,
    )
    probe = ProbeDataset(
        question_ids=tuple(row.question_id for row in rows),
        features=torch.from_numpy(values),
        outcomes=tuple(row.outcome for row in rows),
        group_ids=tuple(row.semantic_group_id for row in rows),
        feature_schema=schema,
    )
    return E2FeatureDataset(probe=probe, rows=rows)


def _build_e2_view_datasets(
    workspace: VerifiedE2Workspace,
    *,
    view: E2FeatureView,
    split_manifest_digest: str,
    prompt_template_sha256: Mapping[str, str],
    verified_store: VerifiedActivationStore,
) -> Mapping[tuple[str, str], E2FeatureDataset]:
    """Extract every partition for one layer/site during one store traversal."""

    expected_keys = {
        ("T-controller-train", "P0-neutral"),
        ("T-controller-calibration", "P0-neutral"),
        ("T-dev", "P0-neutral"),
        ("simpleqa-eval", "P0-neutral"),
        ("aa-eval", "P0-neutral"),
        ("T-controller-train", "P3-forced-answer"),
        ("T-controller-calibration", "P3-forced-answer"),
        ("T-dev", "P3-forced-answer"),
    }
    site_index = workspace.activation_spec.sites.index(view.site)
    layer_index = workspace.activation_spec.layers.index(view.layer)
    selected_rows: dict[tuple[str, str], list[ActivationCaptureRow]] = {
        key: [] for key in expected_keys
    }
    selected_values: dict[tuple[str, str], list[np.ndarray[Any, Any]]] = {
        key: [] for key in expected_keys
    }
    sequence = 0
    for rows, activations in iter_activation_shards(
        workspace.directory / "activations",
        expected_spec=workspace.activation_spec,
        verified_store=verified_store,
    ):
        for offset, row in enumerate(rows):
            schedule = workspace.schedule[sequence]
            if (
                row.question_id != schedule.question_id
                or row.benchmark != schedule.benchmark
                or row.partition != schedule.feature_partition
                or row.prompt_id != schedule.prompt_id
                or row.semantic_group_id != schedule.semantic_group_id
                or (schedule.outcome is not None and row.outcome is not schedule.outcome)
            ):
                raise FrozenArtifactError("E2 probe input differs from its capture schedule")
            key = (row.partition, row.prompt_id)
            if key not in expected_keys:
                raise FrozenArtifactError("E2 probe input contains an unexpected partition")
            selected_rows[key].append(row)
            selected_values[key].append(
                np.asarray(
                    activations[offset, site_index, layer_index, :], dtype=np.float32
                ).copy()
            )
            sequence += 1
    if sequence != len(workspace.schedule) or any(
        not selected_rows[key] for key in expected_keys
    ):
        raise FrozenArtifactError("E2 view extraction is incomplete")
    datasets: dict[tuple[str, str], E2FeatureDataset] = {}
    for partition, prompt_id in sorted(expected_keys):
        rows = tuple(selected_rows[(partition, prompt_id)])
        benchmarks = {row.benchmark for row in rows}
        if len(benchmarks) != 1:
            raise DataValidationError("E2 probe partition mixes benchmarks")
        schema = ActivationFeatureSchema(
            benchmark=next(iter(benchmarks)),
            partition=partition,
            split_manifest_digest=split_manifest_digest,
            model_repository=workspace.activation_spec.model_repository,
            model_revision=workspace.activation_spec.model_revision,
            runtime=Runtime.VLLM,
            quantization=workspace.activation_spec.quantization,
            prompt_id=prompt_id,
            prompt_sha256=prompt_template_sha256[prompt_id],
            activation_kind=ActivationKind.FINAL_PROMPT,
            layers=(view.layer,),
            sites=(view.site,),
            composition=FeatureComposition.SINGLE_LAYER,
            width=workspace.activation_spec.hidden_width,
            token_scope=TokenScope.FINAL_PROMPT,
        )
        probe = ProbeDataset(
            question_ids=tuple(row.question_id for row in rows),
            features=torch.from_numpy(
                np.stack(selected_values[(partition, prompt_id)], axis=0)
            ),
            outcomes=tuple(row.outcome for row in rows),
            group_ids=tuple(row.semantic_group_id for row in rows),
            feature_schema=schema,
        )
        datasets[(partition, prompt_id)] = E2FeatureDataset(probe=probe, rows=rows)
    return MappingProxyType(datasets)


def _training_config(
    protocol: E2ProbeProtocol, *, kind: ProbeKind, epochs: int
) -> ProbeTrainingConfig:
    return ProbeTrainingConfig(
        kind=kind,
        hidden_width=protocol.mlp_hidden_width,
        epochs=epochs,
        learning_rate=protocol.learning_rate,
        weight_decay=protocol.weight_decay,
        seed=protocol.seed,
    )


def _fit_state_and_calibrators(
    training: ProbeDataset,
    calibration: ProbeDataset,
    *,
    task: ProbeTask,
    config: ProbeTrainingConfig,
) -> Mapping[CalibrationKind, CalibratedProbe]:
    if set(training.group_ids) & set(calibration.group_ids):
        raise DataValidationError("E2 probe training and calibration groups overlap")
    train_features, train_labels = encode_probe_task(training, task)
    calibration_features, calibration_labels = encode_probe_task(calibration, task)
    state = fit_probe_state(
        train_features,
        train_labels,
        class_names=_task_labels(task),
        config=config,
    )
    logits = state.logits(calibration_features)
    calibrators: Mapping[
        CalibrationKind, TemperatureCalibrator | IsotonicCalibrator
    ] = {
        CalibrationKind.TEMPERATURE: fit_temperature(logits, calibration_labels),
        CalibrationKind.ISOTONIC: fit_isotonic(logits, calibration_labels),
    }
    return MappingProxyType(
        {
            kind: CalibratedProbe(
                task=task,
                state=state,
                calibrator=calibrator,
                training_fingerprint=training.data_fingerprint,
                calibration_fingerprint=calibration.data_fingerprint,
                training_schema=training.feature_schema,
                calibration_schema=calibration.feature_schema,
            )
            for kind, calibrator in calibrators.items()
            if training.feature_schema is not None and calibration.feature_schema is not None
        }
    )


def _metrics_dict(metrics: ProbeMetrics) -> dict[str, Any]:
    return {
        "macro_auroc": metrics.macro_auroc,
        "macro_f1": metrics.macro_f1,
        "brier_score": metrics.brier_score,
        "expected_calibration_error": metrics.expected_calibration_error,
        "per_class_auroc": dict(metrics.per_class_auroc),
    }


def _incorrect_risk(
    probe: CalibratedProbe, dataset: E2FeatureDataset
) -> tuple[float, list[int], list[bool], list[float]]:
    if Outcome.INCORRECT.value not in probe.state.labels:
        raise DataValidationError("E2 gate probe lacks an incorrect class")
    eligible = [
        index
        for index, outcome in enumerate(dataset.probe.outcomes)
        if outcome in {Outcome.CORRECT, Outcome.INCORRECT, Outcome.ABSTENTION}
    ]
    features = dataset.probe.features[torch.tensor(eligible, dtype=torch.long)]
    scores = probe.probability(features, Outcome.INCORRECT.value).tolist()
    labels = [dataset.probe.outcomes[index] is Outcome.INCORRECT for index in eligible]
    return _binary_auroc(labels, scores), eligible, labels, scores


def _input_plan(
    *,
    workspace: VerifiedE2Workspace,
    activation_chain_head: str,
    protocol: E2ProbeProtocol,
    split_manifest_digest: str,
    prompt_template_sha256: Mapping[str, str],
) -> dict[str, Any]:
    body = {
        "schema_version": 1,
        "phase": "E2",
        "runner": "cpu-calibrated-probe-screen-and-freeze",
        "runner_source_sha256": sha256_file(Path(__file__)),
        "workspace_plan_identity": workspace.plan_identity,
        "activation_chain_head": activation_chain_head,
        "protocol": protocol.to_dict(),
        "split_manifest_digest": split_manifest_digest,
        "prompt_template_sha256": dict(sorted(prompt_template_sha256.items())),
    }
    return {**body, "probe_plan_identity": stable_hash(body)}


def _write_probe_progress(
    path: Path, *, probe_plan_identity: str, rows: Sequence[Mapping[str, Any]]
) -> None:
    body = {
        "schema_version": 1,
        "probe_plan_identity": probe_plan_identity,
        "rows": [dict(row) for row in rows],
    }
    value = {**body, "progress_digest": stable_hash(body)}
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def _load_probe_progress(path: Path, *, probe_plan_identity: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read E2 probe progress: {exc}") from exc
    if type(value) is not dict:
        raise FrozenArtifactError("E2 probe progress must be a mapping")
    body = dict(value)
    digest = body.pop("progress_digest", None)
    rows = body.get("rows")
    if (
        set(body) != {"schema_version", "probe_plan_identity", "rows"}
        or body["schema_version"] != 1
        or body["probe_plan_identity"] != probe_plan_identity
        or digest != stable_hash(body)
        or type(rows) is not list
        or any(type(row) is not dict for row in rows)
    ):
        raise FrozenArtifactError("E2 probe progress identity or digest differs")
    return [dict(row) for row in rows]


def fit_e2_probe_bundle(
    directory: str | Path,
    *,
    workspace: VerifiedE2Workspace,
    split_manifest_digest: str,
    prompt_template_sha256: Mapping[str, str],
    protocol: E2ProbeProtocol | None = None,
    work_directory: str | Path | None = None,
) -> VerifiedE2ProbeBundle:
    mutable_paths: dict[str, str | Path] = {
        "E2 probe bundle": directory,
        "E2 workspace": workspace.directory,
    }
    if work_directory is not None:
        mutable_paths["E2 probe work"] = work_directory
    normalized_paths = validate_active_study_artifact_paths(mutable_paths)
    directory = normalized_paths["E2 probe bundle"]
    if work_directory is not None:
        work_directory = normalized_paths["E2 probe work"]
    protocol = protocol or E2ProbeProtocol()
    if set(prompt_template_sha256) != {"P0-neutral", "P3-forced-answer"}:
        raise ConfigurationError("E2 probes require exact P0 and P3 template fingerprints")
    activation = verify_activation_store(
        workspace.directory / "activations",
        expected_spec=workspace.activation_spec,
        require_complete=True,
    )
    if activation.chain_head is None:
        raise FrozenArtifactError("E2 complete activation store lacks a chain head")
    destination = Path(directory)
    if destination.exists():
        raise FrozenArtifactError(f"refusing to overwrite E2 probe bundle: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    plan = _input_plan(
        workspace=workspace,
        activation_chain_head=activation.chain_head,
        protocol=protocol,
        split_manifest_digest=split_manifest_digest,
        prompt_template_sha256=prompt_template_sha256,
    )
    stage = (
        Path(work_directory)
        if work_directory is not None
        else destination.parent
        / f".{destination.name}.work-{str(plan['probe_plan_identity'])[:12]}"
    )
    stage = stage.absolute()
    if stage == destination.absolute():
        raise ConfigurationError("E2 probe work and output directories must differ")
    work_inventory = {
        "plan.json",
        "probes",
        "controller-input-probes",
        "screening-probes",
        "screening-progress.json",
        "final-progress.json",
        "controller-input-progress.json",
    }
    if stage.exists():
        if (
            stage.is_symlink()
            or not stage.is_dir()
            or {path.name for path in stage.iterdir()} != work_inventory
            or any(path.is_symlink() for path in stage.rglob("*"))
        ):
            raise FrozenArtifactError("E2 probe work inventory differs")
        try:
            existing_plan = json.loads((stage / "plan.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FrozenArtifactError(f"cannot read E2 probe work plan: {exc}") from exc
        if existing_plan != plan:
            raise FrozenArtifactError("E2 probe work plan differs from live inputs")
    else:
        stage.mkdir(parents=True)
        (stage / "probes").mkdir()
        (stage / "controller-input-probes").mkdir()
        (stage / "screening-probes").mkdir()
        (stage / "plan.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _write_probe_progress(
            stage / "screening-progress.json",
            probe_plan_identity=str(plan["probe_plan_identity"]),
            rows=(),
        )
        _write_probe_progress(
            stage / "final-progress.json",
            probe_plan_identity=str(plan["probe_plan_identity"]),
            rows=(),
        )
        _write_probe_progress(
            stage / "controller-input-progress.json",
            probe_plan_identity=str(plan["probe_plan_identity"]),
            rows=(),
        )
    screening_rows = _load_probe_progress(
        stage / "screening-progress.json",
        probe_plan_identity=str(plan["probe_plan_identity"]),
    )
    final_rows = _load_probe_progress(
        stage / "final-progress.json",
        probe_plan_identity=str(plan["probe_plan_identity"]),
    )
    controller_input_rows = _load_probe_progress(
        stage / "controller-input-progress.json",
        probe_plan_identity=str(plan["probe_plan_identity"]),
    )
    bundle_stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent)
    )
    try:
        selected_views: dict[ProbeTask, E2FeatureView] = {}
        views = tuple(
            E2FeatureView(layer, site)
            for site in protocol.candidate_sites
            for layer in protocol.candidate_layers
        )
        screening_by_cell: dict[tuple[ProbeTask, E2FeatureView], dict[str, Any]] = {}
        for row in screening_rows:
            try:
                task = ProbeTask(row["task"])
                view = E2FeatureView(
                    row["view"]["layer"], ActivationSite(row["view"]["site"])
                )
            except (KeyError, TypeError, ValueError, DataValidationError) as exc:
                raise FrozenArtifactError(f"invalid E2 screening progress row: {exc}") from exc
            if (task, view) in screening_by_cell or view not in views:
                raise FrozenArtifactError("E2 screening progress grid differs")
            screening_by_cell[(task, view)] = row
        candidates_by_task: dict[ProbeTask, list[tuple[float, E2FeatureView]]] = {
            task: [] for task in _TASKS
        }
        for view in views:
            view_datasets = _build_e2_view_datasets(
                workspace,
                view=view,
                split_manifest_digest=split_manifest_digest,
                prompt_template_sha256=prompt_template_sha256,
                verified_store=activation,
            )
            for task in _TASKS:
                prompt_id = _prompt_for_task(task)
                training = view_datasets[("T-controller-train", prompt_id)]
                calibration = view_datasets[("T-controller-calibration", prompt_id)]
                dev = view_datasets[("T-dev", prompt_id)]
                screening_relative = Path(task.value) / view.identifier
                screening_artifact = stage / "screening-probes" / screening_relative
                if screening_artifact.exists():
                    probe = load_calibrated_probe(
                        screening_artifact,
                        expected_training_fingerprint=training.probe.data_fingerprint,
                        expected_calibration_fingerprint=calibration.probe.data_fingerprint,
                    )
                else:
                    probe = _fit_state_and_calibrators(
                        training.probe,
                        calibration.probe,
                        task=task,
                        config=_training_config(
                            protocol,
                            kind=ProbeKind.LOGISTIC,
                            epochs=protocol.screening_epochs,
                        ),
                    )[CalibrationKind.TEMPERATURE]
                    save_calibrated_probe(screening_artifact, probe)
                if (
                    probe.task is not task
                    or probe.state.kind is not ProbeKind.LOGISTIC
                    or not isinstance(probe.calibrator, TemperatureCalibrator)
                ):
                    raise FrozenArtifactError("E2 screening work artifact binding differs")
                metrics = evaluate_probe(probe, dev.probe)
                expected_row = {
                    "task": task.value,
                    "view": view.to_dict(),
                    "artifact": str(screening_relative),
                    "artifact_sha256": sha256_path(screening_artifact),
                    "metrics": _metrics_dict(metrics),
                    "training_fingerprint": training.probe.data_fingerprint,
                    "calibration_fingerprint": calibration.probe.data_fingerprint,
                }
                existing = screening_by_cell.get((task, view))
                if existing is not None and existing != expected_row:
                    raise FrozenArtifactError("E2 screening progress differs from replay")
                if existing is None:
                    screening_rows.append(expected_row)
                    screening_by_cell[(task, view)] = expected_row
                    _write_probe_progress(
                        stage / "screening-progress.json",
                        probe_plan_identity=str(plan["probe_plan_identity"]),
                        rows=screening_rows,
                    )
                candidates_by_task[task].append((metrics.macro_auroc, view))
        for task, candidates in candidates_by_task.items():
            selected_views[task] = sorted(
                candidates,
                key=lambda value: (-value[0], value[1].site.value, value[1].layer),
            )[0][1]

        final_by_cell: dict[
            tuple[ProbeTask, ProbeKind, CalibrationKind], dict[str, Any]
        ] = {}
        for row in final_rows:
            try:
                cell = (
                    ProbeTask(row["task"]),
                    ProbeKind(row["kind"]),
                    CalibrationKind(row["calibration"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise FrozenArtifactError(f"invalid E2 final progress row: {exc}") from exc
            if cell in final_by_cell:
                raise FrozenArtifactError("E2 final progress contains a duplicate cell")
            final_by_cell[cell] = row
        gate_candidates: list[
            tuple[float, str, CalibratedProbe, E2FeatureDataset, list[int], list[bool]]
        ] = []
        selected_dataset_cache: dict[
            E2FeatureView, Mapping[tuple[str, str], E2FeatureDataset]
        ] = {}
        for task, view in selected_views.items():
            prompt_id = _prompt_for_task(task)
            if view not in selected_dataset_cache:
                selected_dataset_cache[view] = _build_e2_view_datasets(
                    workspace,
                    view=view,
                    split_manifest_digest=split_manifest_digest,
                    prompt_template_sha256=prompt_template_sha256,
                    verified_store=activation,
                )
            view_datasets = selected_dataset_cache[view]
            training = view_datasets[("T-controller-train", prompt_id)]
            calibration = view_datasets[("T-controller-calibration", prompt_id)]
            evaluations = {"T-dev": view_datasets[("T-dev", prompt_id)]}
            if prompt_id == "P0-neutral":
                for partition in ("simpleqa-eval", "aa-eval"):
                    evaluations[partition] = view_datasets[(partition, prompt_id)]
            for kind in _KINDS:
                missing_artifact = any(
                    not (
                        stage
                        / "probes"
                        / task.value
                        / kind.value
                        / calibration_kind.value
                    ).exists()
                    for calibration_kind in _CALIBRATORS
                )
                probes = (
                    _fit_state_and_calibrators(
                        training.probe,
                        calibration.probe,
                        task=task,
                        config=_training_config(
                            protocol, kind=kind, epochs=protocol.final_epochs
                        ),
                    )
                    if missing_artifact
                    else None
                )
                for calibration_kind in _CALIBRATORS:
                    relative = Path(task.value) / kind.value / calibration_kind.value
                    artifact_directory = stage / "probes" / relative
                    if artifact_directory.exists():
                        probe = load_calibrated_probe(
                            artifact_directory,
                            expected_training_fingerprint=training.probe.data_fingerprint,
                            expected_calibration_fingerprint=calibration.probe.data_fingerprint,
                        )
                    else:
                        assert probes is not None
                        probe = probes[calibration_kind]
                        save_calibrated_probe(artifact_directory, probe)
                    artifact_sha256 = sha256_path(artifact_directory)
                    evaluation_metrics = {
                        name: _metrics_dict(evaluate_probe(probe, dataset.probe))
                        for name, dataset in evaluations.items()
                    }
                    result_row: dict[str, Any] = {
                        "task": task.value,
                        "kind": kind.value,
                        "calibration": calibration_kind.value,
                        "view": view.to_dict(),
                        "artifact": str(relative),
                        "artifact_sha256": artifact_sha256,
                        "metrics": evaluation_metrics,
                    }
                    if task is ProbeTask.CORRECT_INCORRECT_ABSTENTION:
                        risk, eligible, labels, _scores = _incorrect_risk(
                            probe, evaluations["T-dev"]
                        )
                        result_row["incorrect_auroc"] = risk
                        gate_candidates.append(
                            (
                                risk,
                                artifact_sha256,
                                probe,
                                evaluations["T-dev"],
                                eligible,
                                labels,
                            )
                        )
                    cell = (task, kind, calibration_kind)
                    existing = final_by_cell.get(cell)
                    if existing is not None and existing != result_row:
                        raise FrozenArtifactError("E2 final progress differs from replay")
                    if existing is None:
                        final_rows.append(result_row)
                        final_by_cell[cell] = result_row
                        _write_probe_progress(
                            stage / "final-progress.json",
                            probe_plan_identity=str(plan["probe_plan_identity"]),
                            rows=final_rows,
                        )
        (
            gate_risk,
            gate_artifact,
            _gate_probe,
            gate_dataset,
            gate_eligible,
            gate_labels,
        ) = sorted(
            gate_candidates, key=lambda value: (-value[0], value[1])
        )[0]
        entropy = [gate_dataset.rows[index].output_entropy for index in gate_eligible]
        maxprob_risk = [
            1 - gate_dataset.rows[index].maximum_token_probability
            for index in gate_eligible
        ]
        strongest_baseline = max(
            _binary_auroc(gate_labels, entropy),
            _binary_auroc(gate_labels, maxprob_risk),
        )
        gate = {
            "selected_artifact_sha256": gate_artifact,
            "probe_incorrect_auroc": gate_risk,
            "entropy_auroc": _binary_auroc(gate_labels, entropy),
            "one_minus_maximum_probability_auroc": _binary_auroc(
                gate_labels, maxprob_risk
            ),
            "strongest_confidence_baseline_auroc": strongest_baseline,
            "minimum_material_gain": protocol.minimum_material_gain,
            "passed": gate_risk - strongest_baseline >= protocol.minimum_material_gain,
            "eligible_partition": "T-dev",
            "eligible_prompt_id": "P0-neutral",
            "risk_definition": "P(I)-versus-I-rest",
        }
        gate_row = next(
            row for row in final_rows if row["artifact_sha256"] == gate_artifact
        )
        gate_kind = ProbeKind(gate_row["kind"])
        gate_calibration = CalibrationKind(gate_row["calibration"])
        gate_view = selected_views[ProbeTask.CORRECT_INCORRECT_ABSTENTION]
        controller_views = controller_input_views(
            selected_layer=gate_view.layer,
            selected_site=gate_view.site,
            candidate_layers=protocol.candidate_layers,
        )
        controller_datasets = build_e2_controller_input_datasets(
            workspace,
            views=controller_views,
            split_manifest_digest=split_manifest_digest,
            prompt_template_sha256=prompt_template_sha256["P0-neutral"],
            verified_store=activation,
        )
        controller_progress: dict[FeatureComposition, dict[str, Any]] = {}
        for row in controller_input_rows:
            try:
                composition = FeatureComposition(row["controller_input"])
            except (KeyError, TypeError, ValueError) as exc:
                raise FrozenArtifactError(
                    f"invalid E2 controller-input progress row: {exc}"
                ) from exc
            if composition in controller_progress:
                raise FrozenArtifactError(
                    "E2 controller-input progress contains a duplicate composition"
                )
            controller_progress[composition] = row
        for controller_view in controller_views:
            composition = controller_view.composition
            controller_training = controller_datasets[
                (composition, "T-controller-train")
            ]
            controller_calibration = controller_datasets[
                (composition, "T-controller-calibration")
            ]
            relative = Path(composition.value)
            artifact = stage / "controller-input-probes" / relative
            if artifact.exists():
                probe = load_calibrated_probe(
                    artifact,
                    expected_training_fingerprint=controller_training.probe.data_fingerprint,
                    expected_calibration_fingerprint=(
                        controller_calibration.probe.data_fingerprint
                    ),
                )
            else:
                probe = _fit_state_and_calibrators(
                    controller_training.probe,
                    controller_calibration.probe,
                    task=ProbeTask.CORRECT_INCORRECT_ABSTENTION,
                    config=_training_config(
                        protocol,
                        kind=gate_kind,
                        epochs=protocol.final_epochs,
                    ),
                )[gate_calibration]
                save_calibrated_probe(artifact, probe)
            controller_metrics = {
                partition: _metrics_dict(
                    evaluate_probe(
                        probe,
                        controller_datasets[(composition, partition)].probe,
                    )
                )
                for partition in ("T-dev", "simpleqa-eval", "aa-eval")
            }
            result_row = {
                "controller_input": composition.value,
                "view": controller_view.to_dict(),
                "kind": gate_kind.value,
                "calibration": gate_calibration.value,
                "artifact": str(relative),
                "artifact_sha256": sha256_path(artifact),
                "training_fingerprint": controller_training.probe.data_fingerprint,
                "calibration_fingerprint": (
                    controller_calibration.probe.data_fingerprint
                ),
                "metrics": controller_metrics,
            }
            existing = controller_progress.get(composition)
            if existing is not None and existing != result_row:
                raise FrozenArtifactError(
                    "E2 controller-input progress differs from exact replay"
                )
            if existing is None:
                controller_input_rows.append(result_row)
                controller_progress[composition] = result_row
                _write_probe_progress(
                    stage / "controller-input-progress.json",
                    probe_plan_identity=str(plan["probe_plan_identity"]),
                    rows=controller_input_rows,
                )
        if verify_activation_store(
            workspace.directory / "activations",
            expected_spec=workspace.activation_spec,
            require_complete=True,
        ).chain_head != activation.chain_head:
            raise FrozenArtifactError("E2 activation store changed during probe fitting")
        shutil.copytree(stage / "probes", bundle_stage / "probes")
        shutil.copytree(
            stage / "controller-input-probes",
            bundle_stage / "controller-input-probes",
        )
        shutil.copytree(
            stage / "screening-probes", bundle_stage / "screening-probes"
        )
        (bundle_stage / "plan.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (bundle_stage / "screening.json").write_text(
            json.dumps(screening_rows, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        results = {
            "schema_version": 1,
            "selected_views": {
                task.value: view.to_dict() for task, view in selected_views.items()
            },
            "final_probes": final_rows,
            "controller_input_probes": controller_input_rows,
            "gate": gate,
        }
        (bundle_stage / "results.json").write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest_body = {
            "schema_version": 1,
            "phase": "E2",
            "purpose": "activation-separability-calibrated-probe-bundle",
            "probe_plan_identity": plan["probe_plan_identity"],
            "scientific_eligible": (
                workspace.protocol.scientific_eligible and protocol.scientific_eligible
            ),
            "files": {
                "plan.json": sha256_file(bundle_stage / "plan.json"),
                "screening.json": sha256_file(bundle_stage / "screening.json"),
                "results.json": sha256_file(bundle_stage / "results.json"),
            },
            "probes_sha256": sha256_path(bundle_stage / "probes"),
            "controller_input_probes_sha256": sha256_path(
                bundle_stage / "controller-input-probes"
            ),
            "screening_probes_sha256": sha256_path(bundle_stage / "screening-probes"),
        }
        (bundle_stage / "manifest.json").write_text(
            json.dumps(
                {**manifest_body, "manifest_digest": stable_hash(manifest_body)},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(bundle_stage, destination)
        shutil.rmtree(stage)
    finally:
        if bundle_stage.exists():
            shutil.rmtree(bundle_stage)
    return verify_e2_probe_bundle(destination, workspace=workspace)


def verify_e2_probe_bundle(
    directory: str | Path,
    *,
    workspace: VerifiedE2Workspace,
) -> VerifiedE2ProbeBundle:
    source = Path(directory)
    expected_inventory = {
        "plan.json",
        "screening.json",
        "results.json",
        "probes",
        "controller-input-probes",
        "screening-probes",
        "manifest.json",
    }
    if (
        source.is_symlink()
        or not source.is_dir()
        or {path.name for path in source.iterdir()} != expected_inventory
        or any(path.is_symlink() for path in source.rglob("*"))
    ):
        raise FrozenArtifactError("E2 probe bundle inventory differs")
    try:
        plan = json.loads((source / "plan.json").read_text(encoding="utf-8"))
        screening = json.loads((source / "screening.json").read_text(encoding="utf-8"))
        results = json.loads((source / "results.json").read_text(encoding="utf-8"))
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read E2 probe bundle: {exc}") from exc
    if any(type(value) is not dict for value in (plan, results, manifest)):
        raise FrozenArtifactError("E2 probe bundle roots must be mappings")
    plan_body = dict(plan)
    plan_identity = plan_body.pop("probe_plan_identity", None)
    manifest_body = dict(manifest)
    manifest_digest = manifest_body.pop("manifest_digest", None)
    activation = verify_activation_store(
        workspace.directory / "activations",
        expected_spec=workspace.activation_spec,
        require_complete=True,
    )
    if (
        plan_identity != stable_hash(plan_body)
        or plan.get("runner_source_sha256") != sha256_file(Path(__file__))
        or plan.get("workspace_plan_identity") != workspace.plan_identity
        or plan.get("activation_chain_head") != activation.chain_head
        or manifest_digest != stable_hash(manifest_body)
        or manifest.get("probe_plan_identity") != plan_identity
        or manifest.get("files")
        != {
            "plan.json": sha256_file(source / "plan.json"),
            "screening.json": sha256_file(source / "screening.json"),
            "results.json": sha256_file(source / "results.json"),
        }
        or manifest.get("probes_sha256") != sha256_path(source / "probes")
        or manifest.get("controller_input_probes_sha256")
        != sha256_path(source / "controller-input-probes")
        or manifest.get("screening_probes_sha256")
        != sha256_path(source / "screening-probes")
    ):
        raise FrozenArtifactError("E2 probe bundle provenance differs")
    try:
        if set(plan) != {
            "schema_version",
            "phase",
            "runner",
            "runner_source_sha256",
            "workspace_plan_identity",
            "activation_chain_head",
            "protocol",
            "split_manifest_digest",
            "prompt_template_sha256",
            "probe_plan_identity",
        } or set(manifest) != {
            "schema_version",
            "phase",
            "purpose",
            "probe_plan_identity",
            "scientific_eligible",
            "files",
            "probes_sha256",
            "controller_input_probes_sha256",
            "screening_probes_sha256",
            "manifest_digest",
        }:
            raise DataValidationError("E2 probe plan or manifest schema differs")
        protocol_value = plan["protocol"]
        prompts = plan["prompt_template_sha256"]
        if type(protocol_value) is not dict or type(prompts) is not dict:
            raise DataValidationError("E2 probe plan schemas are invalid")
        protocol = E2ProbeProtocol(
            candidate_layers=tuple(protocol_value["candidate_layers"]),
            candidate_sites=tuple(
                ActivationSite(value) for value in protocol_value["candidate_sites"]
            ),
            screening_epochs=protocol_value["screening_epochs"],
            final_epochs=protocol_value["final_epochs"],
            mlp_hidden_width=protocol_value["mlp_hidden_width"],
            learning_rate=protocol_value["learning_rate"],
            weight_decay=protocol_value["weight_decay"],
            seed=protocol_value["seed"],
            minimum_material_gain=protocol_value["minimum_material_gain"],
            schema_version=protocol_value["schema_version"],
        )
        if (
            protocol_value != protocol.to_dict()
            or plan["schema_version"] != 1
            or plan["phase"] != "E2"
            or plan["runner"] != "cpu-calibrated-probe-screen-and-freeze"
            or manifest["schema_version"] != 1
            or manifest["phase"] != "E2"
            or manifest["purpose"]
            != "activation-separability-calibrated-probe-bundle"
            or manifest["scientific_eligible"]
            != (workspace.protocol.scientific_eligible and protocol.scientific_eligible)
            or set(prompts) != {"P0-neutral", "P3-forced-answer"}
            or any(type(value) is not str for value in prompts.values())
            or type(screening) is not list
            or len(screening)
            != len(_TASKS) * len(protocol.candidate_layers) * len(protocol.candidate_sites)
        ):
            raise DataValidationError("E2 probe screening plan is incomplete")
        if set(results) != {
            "schema_version",
            "selected_views",
            "final_probes",
            "controller_input_probes",
            "gate",
        }:
            raise DataValidationError("E2 probe result schema differs")
        if results["schema_version"] != 1 or type(results["selected_views"]) is not dict:
            raise DataValidationError("E2 probe result identity differs")
        final = results["final_probes"]
        if not isinstance(final, list) or len(final) != len(_TASKS) * len(_KINDS) * len(
            _CALIBRATORS
        ):
            raise DataValidationError("E2 probe bundle has the wrong final grid")
        selected_views = {
            ProbeTask(task): E2FeatureView(value["layer"], ActivationSite(value["site"]))
            for task, value in results["selected_views"].items()
        }
        if set(selected_views) != set(_TASKS):
            raise DataValidationError("E2 selected feature views are incomplete")
        cached_view: E2FeatureView | None = None
        cached_view_datasets: Mapping[tuple[str, str], E2FeatureDataset] | None = None

        def dataset(
            partition: str, prompt_id: str, view: E2FeatureView
        ) -> E2FeatureDataset:
            nonlocal cached_view, cached_view_datasets
            if cached_view != view:
                cached_view_datasets = _build_e2_view_datasets(
                    workspace,
                    view=view,
                    split_manifest_digest=plan["split_manifest_digest"],
                    prompt_template_sha256=prompts,
                    verified_store=activation,
                )
                cached_view = view
            assert cached_view_datasets is not None
            return cached_view_datasets[(partition, prompt_id)]

        screened: dict[ProbeTask, list[tuple[float, E2FeatureView]]] = {
            task: [] for task in _TASKS
        }
        screening_cells: set[tuple[ProbeTask, E2FeatureView]] = set()
        screening_order: list[tuple[ProbeTask, E2FeatureView]] = []
        for row in screening:
            if type(row) is not dict or set(row) != {
                "task",
                "view",
                "artifact",
                "artifact_sha256",
                "metrics",
                "training_fingerprint",
                "calibration_fingerprint",
            }:
                raise DataValidationError("E2 screening row schema differs")
            task = ProbeTask(row["task"])
            view = E2FeatureView(
                row["view"]["layer"], ActivationSite(row["view"]["site"])
            )
            cell = (task, view)
            if cell in screening_cells:
                raise DataValidationError("E2 screening grid contains a duplicate cell")
            screening_cells.add(cell)
            screening_order.append(cell)
            prompt_id = _prompt_for_task(task)
            training = dataset("T-controller-train", prompt_id, view)
            calibration = dataset("T-controller-calibration", prompt_id, view)
            dev = dataset("T-dev", prompt_id, view)
            relative = Path(task.value) / view.identifier
            if row["artifact"] != str(relative):
                raise DataValidationError("E2 screening artifact path differs")
            artifact = source / "screening-probes" / relative
            if sha256_path(artifact) != row["artifact_sha256"]:
                raise FrozenArtifactError("E2 screening artifact digest differs")
            probe = load_calibrated_probe(
                artifact,
                expected_training_fingerprint=training.probe.data_fingerprint,
                expected_calibration_fingerprint=calibration.probe.data_fingerprint,
            )
            if (
                probe.task is not task
                or probe.state.kind is not ProbeKind.LOGISTIC
                or not isinstance(probe.calibrator, TemperatureCalibrator)
                or probe.training_schema != training.probe.feature_schema
                or probe.calibration_schema != calibration.probe.feature_schema
                or row["training_fingerprint"] != training.probe.data_fingerprint
                or row["calibration_fingerprint"] != calibration.probe.data_fingerprint
            ):
                raise DataValidationError("E2 screening probe binding differs")
            metrics = _metrics_dict(evaluate_probe(probe, dev.probe))
            if row["metrics"] != metrics:
                raise DataValidationError("E2 screening metrics differ from replay")
            screened[task].append((metrics["macro_auroc"], view))
        expected_screening_cells = {
            (task, E2FeatureView(layer, site))
            for task in _TASKS
            for site in protocol.candidate_sites
            for layer in protocol.candidate_layers
        }
        if screening_cells != expected_screening_cells:
            raise DataValidationError("E2 screening grid is not exact")
        expected_screening_order = [
            (task, E2FeatureView(layer, site))
            for site in protocol.candidate_sites
            for layer in protocol.candidate_layers
            for task in _TASKS
        ]
        if screening_order != expected_screening_order:
            raise DataValidationError("E2 screening order is not canonical")
        expected_selected = {
            task: sorted(
                values,
                key=lambda value: (-value[0], value[1].site.value, value[1].layer),
            )[0][1]
            for task, values in screened.items()
        }
        if selected_views != expected_selected:
            raise DataValidationError("E2 selected views differ from screening evidence")

        artifact_digests: set[str] = set()
        loaded: dict[str, CalibratedProbe] = {}
        final_cells: set[tuple[ProbeTask, ProbeKind, CalibrationKind]] = set()
        gate_rows: list[dict[str, Any]] = []
        for row in final:
            if type(row) is not dict:
                raise DataValidationError("E2 final probe row must be a mapping")
            task = ProbeTask(row["task"])
            expected_keys = {
                "task",
                "kind",
                "calibration",
                "view",
                "artifact",
                "artifact_sha256",
                "metrics",
            }
            if task is ProbeTask.CORRECT_INCORRECT_ABSTENTION:
                expected_keys.add("incorrect_auroc")
            if set(row) != expected_keys:
                raise DataValidationError("E2 final probe row schema differs")
            kind = ProbeKind(row["kind"])
            calibration_kind = CalibrationKind(row["calibration"])
            final_cell = (task, kind, calibration_kind)
            if final_cell in final_cells:
                raise DataValidationError("E2 final grid contains a duplicate cell")
            final_cells.add(final_cell)
            view = E2FeatureView(
                row["view"]["layer"], ActivationSite(row["view"]["site"])
            )
            if view != selected_views[task]:
                raise DataValidationError("E2 final probe uses a non-selected view")
            prompt_id = _prompt_for_task(task)
            training = dataset("T-controller-train", prompt_id, view)
            calibration = dataset("T-controller-calibration", prompt_id, view)
            relative = Path(task.value) / kind.value / calibration_kind.value
            if row["artifact"] != str(relative):
                raise DataValidationError("E2 final artifact path differs")
            artifact = source / "probes" / relative
            if sha256_path(artifact) != row["artifact_sha256"]:
                raise FrozenArtifactError("E2 probe artifact digest differs")
            probe = load_calibrated_probe(
                artifact,
                expected_training_fingerprint=training.probe.data_fingerprint,
                expected_calibration_fingerprint=calibration.probe.data_fingerprint,
            )
            expected_calibrator = (
                TemperatureCalibrator
                if calibration_kind is CalibrationKind.TEMPERATURE
                else IsotonicCalibrator
            )
            if (
                probe.task is not task
                or probe.state.kind is not kind
                or not isinstance(probe.calibrator, expected_calibrator)
                or probe.training_schema != training.probe.feature_schema
                or probe.calibration_schema != calibration.probe.feature_schema
            ):
                raise DataValidationError("E2 final probe binding differs")
            evaluations = {"T-dev": dataset("T-dev", prompt_id, view)}
            if prompt_id == "P0-neutral":
                evaluations.update(
                    {
                        partition: dataset(partition, prompt_id, view)
                        for partition in ("simpleqa-eval", "aa-eval")
                    }
                )
            expected_metrics = {
                name: _metrics_dict(evaluate_probe(probe, value.probe))
                for name, value in evaluations.items()
            }
            if row["metrics"] != expected_metrics:
                raise DataValidationError("E2 final metrics differ from replay")
            digest = row["artifact_sha256"]
            if digest in artifact_digests:
                raise DataValidationError("E2 final artifacts are not unique")
            artifact_digests.add(digest)
            loaded[digest] = probe
            if task is ProbeTask.CORRECT_INCORRECT_ABSTENTION:
                gate_rows.append(row)
        expected_final_cells = {
            (task, kind, calibration_kind)
            for task in _TASKS
            for kind in _KINDS
            for calibration_kind in _CALIBRATORS
        }
        if final_cells != expected_final_cells:
            raise DataValidationError("E2 final probe grid is not exact")

        gate = results["gate"]
        selected_gate_artifact = gate["selected_artifact_sha256"]
        if selected_gate_artifact not in artifact_digests or type(gate["passed"]) is not bool:
            raise DataValidationError("E2 gate selection is invalid")
        gate_view = selected_views[ProbeTask.CORRECT_INCORRECT_ABSTENTION]
        gate_dataset = dataset("T-dev", "P0-neutral", gate_view)
        replays: list[tuple[float, str, list[int], list[bool]]] = []
        for row in gate_rows:
            risk, eligible, labels, _scores = _incorrect_risk(
                loaded[row["artifact_sha256"]], gate_dataset
            )
            if row.get("incorrect_auroc") != risk:
                raise DataValidationError("E2 stored incorrect-risk AUROC differs")
            replays.append((risk, row["artifact_sha256"], eligible, labels))
        gate_risk, replay_artifact, eligible, labels = sorted(
            replays, key=lambda value: (-value[0], value[1])
        )[0]
        entropy = [gate_dataset.rows[index].output_entropy for index in eligible]
        maxprob = [
            1 - gate_dataset.rows[index].maximum_token_probability for index in eligible
        ]
        entropy_auroc = _binary_auroc(labels, entropy)
        maxprob_auroc = _binary_auroc(labels, maxprob)
        expected_gate = {
            "selected_artifact_sha256": replay_artifact,
            "probe_incorrect_auroc": gate_risk,
            "entropy_auroc": entropy_auroc,
            "one_minus_maximum_probability_auroc": maxprob_auroc,
            "strongest_confidence_baseline_auroc": max(entropy_auroc, maxprob_auroc),
            "minimum_material_gain": protocol.minimum_material_gain,
            "passed": gate_risk - max(entropy_auroc, maxprob_auroc)
            >= protocol.minimum_material_gain,
            "eligible_partition": "T-dev",
            "eligible_prompt_id": "P0-neutral",
            "risk_definition": "P(I)-versus-I-rest",
        }
        if gate != expected_gate:
            raise DataValidationError("E2 gate result differs from replayed raw evidence")
        controller_rows = results["controller_input_probes"]
        controller_views = controller_input_views(
            selected_layer=gate_view.layer,
            selected_site=gate_view.site,
            candidate_layers=protocol.candidate_layers,
        )
        if not isinstance(controller_rows, list) or len(controller_rows) != len(
            controller_views
        ):
            raise DataValidationError("E2 controller-input probe grid is incomplete")
        controller_datasets = build_e2_controller_input_datasets(
            workspace,
            views=controller_views,
            split_manifest_digest=plan["split_manifest_digest"],
            prompt_template_sha256=prompts["P0-neutral"],
            verified_store=activation,
        )
        selected_gate_row = next(
            row for row in gate_rows if row["artifact_sha256"] == replay_artifact
        )
        gate_kind = ProbeKind(selected_gate_row["kind"])
        gate_calibration = CalibrationKind(selected_gate_row["calibration"])
        expected_calibrator = (
            TemperatureCalibrator
            if gate_calibration is CalibrationKind.TEMPERATURE
            else IsotonicCalibrator
        )
        controller_input_artifacts: dict[FeatureComposition, str] = {}
        for row, expected_view in zip(
            controller_rows, controller_views, strict=True
        ):
            if not isinstance(row, dict) or set(row) != {
                "controller_input",
                "view",
                "kind",
                "calibration",
                "artifact",
                "artifact_sha256",
                "training_fingerprint",
                "calibration_fingerprint",
                "metrics",
            }:
                raise DataValidationError("E2 controller-input probe row schema differs")
            raw_view = row["view"]
            if not isinstance(raw_view, dict):
                raise DataValidationError("E2 controller-input view must be a mapping")
            replayed_controller_view = E2ControllerInputView(
                composition=FeatureComposition(raw_view["composition"]),
                layers=tuple(raw_view["layers"]),
                site=ActivationSite(raw_view["site"]),
            )
            composition = replayed_controller_view.composition
            if (
                replayed_controller_view != expected_view
                or row["controller_input"] != composition.value
                or row["kind"] != gate_kind.value
                or row["calibration"] != gate_calibration.value
                or row["artifact"] != composition.value
                or composition in controller_input_artifacts
            ):
                raise DataValidationError("E2 controller-input probe identity differs")
            controller_training = controller_datasets[
                (composition, "T-controller-train")
            ]
            controller_calibration = controller_datasets[
                (composition, "T-controller-calibration")
            ]
            artifact = source / "controller-input-probes" / composition.value
            if sha256_path(artifact) != row["artifact_sha256"]:
                raise FrozenArtifactError("E2 controller-input probe artifact changed")
            probe = load_calibrated_probe(
                artifact,
                expected_training_fingerprint=controller_training.probe.data_fingerprint,
                expected_calibration_fingerprint=(
                    controller_calibration.probe.data_fingerprint
                ),
            )
            expected_metrics = {
                partition: _metrics_dict(
                    evaluate_probe(
                        probe,
                        controller_datasets[(composition, partition)].probe,
                    )
                )
                for partition in ("T-dev", "simpleqa-eval", "aa-eval")
            }
            if (
                probe.task is not ProbeTask.CORRECT_INCORRECT_ABSTENTION
                or probe.state.kind is not gate_kind
                or not isinstance(probe.calibrator, expected_calibrator)
                or probe.training_schema != controller_training.probe.feature_schema
                or probe.calibration_schema
                != controller_calibration.probe.feature_schema
                or row["training_fingerprint"]
                != controller_training.probe.data_fingerprint
                or row["calibration_fingerprint"]
                != controller_calibration.probe.data_fingerprint
                or row["metrics"] != expected_metrics
            ):
                raise DataValidationError("E2 controller-input probe replay differs")
            controller_input_artifacts[composition] = row["artifact_sha256"]
        if protocol.scientific_eligible and set(controller_input_artifacts) != {
            FeatureComposition.SINGLE_LAYER,
            FeatureComposition.CONCATENATED_LAYERS,
            FeatureComposition.LAYER_DIFFERENCES,
        }:
            raise DataValidationError(
                "scientific E2 lacks all three preregistered controller inputs"
            )
        if verify_activation_store(
            workspace.directory / "activations",
            expected_spec=workspace.activation_spec,
            require_complete=True,
        ).chain_head != activation.chain_head:
            raise FrozenArtifactError("E2 activation store changed during probe replay")
    except (KeyError, TypeError, ValueError, IndexError, DataValidationError) as exc:
        raise FrozenArtifactError(f"invalid E2 probe result bundle: {exc}") from exc
    return VerifiedE2ProbeBundle(
        directory=source,
        plan_identity=plan_identity,
        manifest_digest=manifest_digest,
        selected_views=MappingProxyType(selected_views),
        selected_gate_artifact=selected_gate_artifact,
        gate_passed=gate["passed"],
        gate_probe_auroc=float(gate["probe_incorrect_auroc"]),
        gate_baseline_auroc=float(gate["strongest_confidence_baseline_auroc"]),
        controller_input_artifacts=MappingProxyType(controller_input_artifacts),
        scientific_eligible=manifest["scientific_eligible"],
    )

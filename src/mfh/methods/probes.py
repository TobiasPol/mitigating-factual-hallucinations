"""Calibrated prompt-end outcome probes used by M3 and M6.

The implementation is deliberately small and serializable.  Probe fitting is
performed on ``T-controller`` training rows, calibration is performed on a
disjoint calibration partition, and the resulting artifact contains every
normalization and calibration parameter needed for inference.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import Tensor

from mfh.artifact_namespace import validate_active_study_artifact_paths
from mfh.contracts import Outcome
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.methods.features import ActivationFeatureSchema
from mfh.provenance import sha256_file, stable_hash

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ProbeTask(StrEnum):
    CORRECT_INCORRECT = "correct_incorrect"
    ATTEMPT_ABSTENTION = "attempt_abstention"
    CORRECT_INCORRECT_ABSTENTION = "correct_incorrect_abstention"
    FORCED_CORRECT_INCORRECT = "forced_correct_incorrect"


class ProbeKind(StrEnum):
    LOGISTIC = "logistic"
    TWO_LAYER_MLP = "two_layer_mlp"


class CalibrationKind(StrEnum):
    TEMPERATURE = "temperature"
    ISOTONIC = "isotonic"


@dataclass(frozen=True, slots=True)
class ProbeDataset:
    question_ids: tuple[str, ...]
    features: Tensor
    outcomes: tuple[Outcome, ...]
    group_ids: tuple[str, ...]
    feature_schema: ActivationFeatureSchema | None = None
    data_fingerprint: str = ""

    def __post_init__(self) -> None:
        ids = tuple(identifier.strip() for identifier in self.question_ids)
        if any(not identifier for identifier in ids):
            raise DataValidationError("probe question IDs must be non-empty")
        if len(ids) != len(set(ids)):
            raise DataValidationError("probe question IDs must be unique")
        groups = tuple(value.strip() for value in self.group_ids)
        if len(groups) != len(ids) or any(not value for value in groups):
            raise DataValidationError("probe group IDs must be non-empty and align with rows")
        if self.features.ndim != 2 or self.features.shape[1] == 0:
            raise DataValidationError("probe features must have shape [rows, width]")
        if self.features.shape[0] != len(ids) or len(ids) != len(self.outcomes):
            raise DataValidationError("probe IDs, features, and outcomes must have equal rows")
        features = self.features.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if not torch.isfinite(features).all():
            raise DataValidationError("probe features contain NaN or infinity")
        outcomes = tuple(Outcome(value) for value in self.outcomes)
        if self.feature_schema is not None and self.feature_schema.width != features.shape[1]:
            raise DataValidationError("probe feature schema width differs from the tensor")
        computed_fingerprint = _dataset_fingerprint(
            ids, groups, features, outcomes, self.feature_schema
        )
        if self.data_fingerprint and self.data_fingerprint != computed_fingerprint:
            raise DataValidationError("provided probe data fingerprint does not match its rows")
        fingerprint = computed_fingerprint
        if not _SHA256.fullmatch(fingerprint):
            raise DataValidationError("probe data fingerprint must be a SHA-256 digest")
        object.__setattr__(self, "question_ids", ids)
        object.__setattr__(self, "group_ids", groups)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "data_fingerprint", fingerprint)


def _dataset_fingerprint(
    question_ids: tuple[str, ...],
    group_ids: tuple[str, ...],
    features: Tensor,
    outcomes: tuple[Outcome, ...],
    feature_schema: ActivationFeatureSchema | None,
) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(question_ids, separators=(",", ":")).encode())
    digest.update(json.dumps(group_ids, separators=(",", ":")).encode())
    digest.update(features.numpy().tobytes(order="C"))
    digest.update("".join(outcome.value for outcome in outcomes).encode())
    digest.update((feature_schema.digest if feature_schema is not None else "unbound").encode())
    return digest.hexdigest()


def _task_labels(task: ProbeTask) -> tuple[str, ...]:
    if task in {ProbeTask.CORRECT_INCORRECT, ProbeTask.FORCED_CORRECT_INCORRECT}:
        return (Outcome.CORRECT.value, Outcome.INCORRECT.value)
    if task is ProbeTask.ATTEMPT_ABSTENTION:
        return ("attempt", "abstention")
    return (Outcome.CORRECT.value, Outcome.INCORRECT.value, Outcome.ABSTENTION.value)


def encode_probe_task(dataset: ProbeDataset, task: ProbeTask) -> tuple[Tensor, Tensor]:
    """Filter unsupported outcomes and encode the task's canonical class order."""

    rows: list[int] = []
    labels: list[int] = []
    for index, outcome in enumerate(dataset.outcomes):
        if task in {ProbeTask.CORRECT_INCORRECT, ProbeTask.FORCED_CORRECT_INCORRECT}:
            if outcome is Outcome.CORRECT:
                rows.append(index)
                labels.append(0)
            elif outcome is Outcome.INCORRECT:
                rows.append(index)
                labels.append(1)
        elif task is ProbeTask.ATTEMPT_ABSTENTION:
            if outcome.is_attempted:
                rows.append(index)
                labels.append(0)
            elif outcome is Outcome.ABSTENTION:
                rows.append(index)
                labels.append(1)
        elif outcome in {Outcome.CORRECT, Outcome.INCORRECT, Outcome.ABSTENTION}:
            rows.append(index)
            labels.append(
                {
                    Outcome.CORRECT: 0,
                    Outcome.INCORRECT: 1,
                    Outcome.ABSTENTION: 2,
                }[outcome]
            )
    if not rows:
        raise DataValidationError(f"probe task {task.value} has no eligible rows")
    encoded = torch.tensor(labels, dtype=torch.long)
    expected = set(range(len(_task_labels(task))))
    if set(encoded.tolist()) != expected:
        raise DataValidationError(
            f"probe task {task.value} requires every class {sorted(expected)}, "
            f"observed {sorted(set(encoded.tolist()))}"
        )
    return dataset.features[torch.tensor(rows, dtype=torch.long)], encoded


@dataclass(frozen=True, slots=True)
class ProbeTrainingConfig:
    kind: ProbeKind = ProbeKind.LOGISTIC
    hidden_width: int = 64
    epochs: int = 400
    learning_rate: float = 0.03
    weight_decay: float = 1e-4
    class_balanced: bool = True
    seed: int = 17

    def __post_init__(self) -> None:
        if self.hidden_width <= 0 or self.epochs <= 0:
            raise DataValidationError("probe hidden width and epochs must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise DataValidationError("invalid probe optimizer configuration")


@dataclass(frozen=True, slots=True)
class ProbeState:
    kind: ProbeKind
    labels: tuple[str, ...]
    feature_mean: Tensor
    feature_scale: Tensor
    parameters: Mapping[str, Tensor]
    hidden_width: int | None = None

    def __post_init__(self) -> None:
        if len(self.labels) < 2 or len(set(self.labels)) != len(self.labels):
            raise DataValidationError("probe labels must contain at least two unique classes")
        mean = self.feature_mean.detach().to(device="cpu", dtype=torch.float32).contiguous()
        scale = self.feature_scale.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if mean.ndim != 1 or scale.shape != mean.shape or mean.numel() == 0:
            raise DataValidationError("probe normalization tensors have invalid shapes")
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise DataValidationError("probe normalization tensors are invalid")
        parameters = {
            name: value.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
            for name, value in self.parameters.items()
        }
        expected = {"weight", "bias"}
        if self.kind is ProbeKind.TWO_LAYER_MLP:
            expected = {"weight1", "bias1", "weight2", "bias2"}
            if self.hidden_width is None or self.hidden_width <= 0:
                raise DataValidationError("MLP probes require a positive hidden width")
        elif self.hidden_width is not None:
            raise DataValidationError("logistic probes cannot declare a hidden width")
        if set(parameters) != expected:
            raise DataValidationError(
                f"probe parameters must be {sorted(expected)}, got {sorted(parameters)}"
            )
        output_width = len(self.labels)
        input_width = int(mean.numel())
        if self.kind is ProbeKind.LOGISTIC:
            if parameters["weight"].shape != (output_width, input_width):
                raise DataValidationError("logistic probe weight has the wrong shape")
            if parameters["bias"].shape != (output_width,):
                raise DataValidationError("logistic probe bias has the wrong shape")
        else:
            assert self.hidden_width is not None
            if parameters["weight1"].shape != (self.hidden_width, input_width):
                raise DataValidationError("MLP input weight has the wrong shape")
            if parameters["bias1"].shape != (self.hidden_width,):
                raise DataValidationError("MLP hidden bias has the wrong shape")
            if parameters["weight2"].shape != (output_width, self.hidden_width):
                raise DataValidationError("MLP output weight has the wrong shape")
            if parameters["bias2"].shape != (output_width,):
                raise DataValidationError("MLP output bias has the wrong shape")
        if any(not torch.isfinite(value).all() for value in parameters.values()):
            raise DataValidationError("probe parameters contain NaN or infinity")
        object.__setattr__(self, "feature_mean", mean)
        object.__setattr__(self, "feature_scale", scale)
        object.__setattr__(self, "parameters", MappingProxyType(parameters))

    @property
    def input_width(self) -> int:
        return int(self.feature_mean.numel())

    def logits(self, features: Tensor) -> Tensor:
        values = features.detach().to(device="cpu", dtype=torch.float32)
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.ndim != 2 or values.shape[1] != self.input_width:
            raise DataValidationError(
                f"probe expected features [rows, {self.input_width}], got {tuple(values.shape)}"
            )
        if not torch.isfinite(values).all():
            raise DataValidationError("probe input contains NaN or infinity")
        normalized = (values - self.feature_mean) / self.feature_scale
        if self.kind is ProbeKind.LOGISTIC:
            return F.linear(normalized, self.parameters["weight"], self.parameters["bias"])
        hidden = F.gelu(F.linear(normalized, self.parameters["weight1"], self.parameters["bias1"]))
        return F.linear(hidden, self.parameters["weight2"], self.parameters["bias2"])


@dataclass(frozen=True, slots=True)
class TemperatureCalibrator:
    temperature: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or not 0.05 <= self.temperature <= 20:
            raise DataValidationError("calibration temperature must be in [0.05, 20]")

    def probabilities(self, logits: Tensor) -> Tensor:
        return torch.softmax(logits / self.temperature, dim=-1)


@dataclass(frozen=True, slots=True)
class IsotonicCurve:
    upper_bounds: Tensor
    values: Tensor

    def __post_init__(self) -> None:
        bounds = self.upper_bounds.detach().to(device="cpu", dtype=torch.float32).contiguous()
        values = self.values.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if bounds.ndim != 1 or bounds.numel() == 0 or values.shape != bounds.shape:
            raise DataValidationError("isotonic curve tensors have invalid shapes")
        if not torch.isfinite(bounds).all() or not torch.isfinite(values).all():
            raise DataValidationError("isotonic curve contains NaN or infinity")
        if (bounds[1:] < bounds[:-1]).any() or (values[1:] < values[:-1]).any():
            raise DataValidationError("isotonic curve must be monotone")
        if (values < 0).any() or (values > 1).any():
            raise DataValidationError("isotonic probabilities must be in [0, 1]")
        object.__setattr__(self, "upper_bounds", bounds)
        object.__setattr__(self, "values", values)

    def predict(self, scores: Tensor) -> Tensor:
        indices = torch.searchsorted(
            self.upper_bounds, scores.detach().cpu().float().contiguous()
        ).clamp(max=self.values.numel() - 1)
        return self.values[indices]


@dataclass(frozen=True, slots=True)
class IsotonicCalibrator:
    curves: tuple[IsotonicCurve, ...]

    def __post_init__(self) -> None:
        if len(self.curves) < 2:
            raise DataValidationError("isotonic calibration requires one curve per class")

    def probabilities(self, logits: Tensor) -> Tensor:
        raw = torch.softmax(logits, dim=-1).cpu()
        if raw.shape[1] != len(self.curves):
            raise DataValidationError("isotonic class count does not match probe output")
        calibrated = torch.stack(
            [curve.predict(raw[:, index]) for index, curve in enumerate(self.curves)], dim=1
        )
        denominator = calibrated.sum(dim=1, keepdim=True)
        uniform = torch.full_like(calibrated, 1 / calibrated.shape[1])
        return torch.where(denominator > 0, calibrated / denominator.clamp_min(1e-12), uniform)


Calibrator = TemperatureCalibrator | IsotonicCalibrator


@dataclass(frozen=True, slots=True)
class CalibratedProbe:
    task: ProbeTask
    state: ProbeState
    calibrator: Calibrator
    training_fingerprint: str
    calibration_fingerprint: str
    training_schema: ActivationFeatureSchema
    calibration_schema: ActivationFeatureSchema
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise DataValidationError("unsupported calibrated-probe schema version")
        if self.state.labels != _task_labels(self.task):
            raise DataValidationError("probe labels do not match the configured task")
        for fingerprint in (self.training_fingerprint, self.calibration_fingerprint):
            if not _SHA256.fullmatch(fingerprint):
                raise DataValidationError("probe split fingerprints must be SHA-256 digests")
        if not self.training_schema.is_compatible_extraction(self.calibration_schema):
            raise DataValidationError("probe training and calibration feature schemas differ")
        if self.training_schema.width != self.state.input_width:
            raise DataValidationError("probe feature schema width differs from the model")
        if isinstance(self.calibrator, IsotonicCalibrator) and len(self.calibrator.curves) != len(
            self.state.labels
        ):
            raise DataValidationError("isotonic calibrator class count is invalid")

    def logits(self, features: Tensor) -> Tensor:
        return self.state.logits(features)

    def predict_probabilities(self, features: Tensor) -> Tensor:
        probabilities = self.calibrator.probabilities(self.logits(features))
        if not torch.isfinite(probabilities).all():
            raise DataValidationError("probe produced non-finite probabilities")
        return probabilities

    def probability(self, features: Tensor, label: str) -> Tensor:
        try:
            index = self.state.labels.index(label)
        except ValueError as exc:
            raise DataValidationError(f"probe has no class {label!r}") from exc
        return self.predict_probabilities(features)[:, index]


def _parameter_initialization(
    input_width: int,
    output_width: int,
    config: ProbeTrainingConfig,
) -> dict[str, torch.nn.Parameter]:
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    if config.kind is ProbeKind.LOGISTIC:
        weight = torch.randn(output_width, input_width, generator=generator) / math.sqrt(
            input_width
        )
        return {
            "weight": torch.nn.Parameter(weight),
            "bias": torch.nn.Parameter(torch.zeros(output_width)),
        }
    weight1 = torch.randn(config.hidden_width, input_width, generator=generator) / math.sqrt(
        input_width
    )
    weight2 = torch.randn(output_width, config.hidden_width, generator=generator) / math.sqrt(
        config.hidden_width
    )
    return {
        "weight1": torch.nn.Parameter(weight1),
        "bias1": torch.nn.Parameter(torch.zeros(config.hidden_width)),
        "weight2": torch.nn.Parameter(weight2),
        "bias2": torch.nn.Parameter(torch.zeros(output_width)),
    }


def _parameter_logits(
    normalized: Tensor, kind: ProbeKind, parameters: Mapping[str, Tensor]
) -> Tensor:
    if kind is ProbeKind.LOGISTIC:
        return F.linear(normalized, parameters["weight"], parameters["bias"])
    hidden = F.gelu(F.linear(normalized, parameters["weight1"], parameters["bias1"]))
    return F.linear(hidden, parameters["weight2"], parameters["bias2"])


def fit_probe_state(
    features: Tensor,
    labels: Tensor,
    *,
    class_names: tuple[str, ...],
    config: ProbeTrainingConfig,
) -> ProbeState:
    values = features.detach().to(device="cpu", dtype=torch.float32)
    targets = labels.detach().to(device="cpu", dtype=torch.long)
    if values.ndim != 2 or values.shape[0] != targets.numel() or values.shape[1] == 0:
        raise DataValidationError("probe training tensors have incompatible shapes")
    if set(targets.tolist()) != set(range(len(class_names))):
        raise DataValidationError("probe training labels do not contain every class")
    mean = values.mean(dim=0)
    scale = values.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized = (values - mean) / scale
    parameters = _parameter_initialization(values.shape[1], len(class_names), config)
    optimizer = torch.optim.AdamW(
        tuple(parameters.values()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    class_weights: Tensor | None = None
    if config.class_balanced:
        counts = torch.bincount(targets, minlength=len(class_names)).float()
        class_weights = targets.numel() / (len(class_names) * counts)
    for _ in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = _parameter_logits(normalized, config.kind, parameters)
        loss = F.cross_entropy(logits, targets, weight=class_weights)
        if not torch.isfinite(loss):
            raise DataValidationError("probe training diverged")
        torch.autograd.backward(loss)
        optimizer.step()
    state_parameters = {name: value.detach() for name, value in parameters.items()}
    return ProbeState(
        kind=config.kind,
        labels=class_names,
        feature_mean=mean,
        feature_scale=scale,
        parameters=state_parameters,
        hidden_width=config.hidden_width if config.kind is ProbeKind.TWO_LAYER_MLP else None,
    )


def fit_temperature(logits: Tensor, labels: Tensor, *, epochs: int = 300) -> TemperatureCalibrator:
    if epochs <= 0:
        raise DataValidationError("temperature fitting epochs must be positive")
    log_temperature = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam((log_temperature,), lr=0.03)
    detached_logits = logits.detach().cpu().float()
    detached_labels = labels.detach().cpu().long()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        temperature = log_temperature.exp().clamp(0.05, 20)
        loss = F.cross_entropy(detached_logits / temperature, detached_labels)
        torch.autograd.backward(loss)
        optimizer.step()
        with torch.no_grad():
            log_temperature.clamp_(math.log(0.05), math.log(20))
    fitted = float(log_temperature.exp().detach())
    return TemperatureCalibrator(min(20.0, max(0.05, fitted)))


def _fit_isotonic_curve(scores: Tensor, targets: Tensor) -> IsotonicCurve:
    ordered = sorted(
        zip(scores.detach().cpu().tolist(), targets.detach().cpu().tolist(), strict=True),
        key=lambda item: item[0],
    )
    grouped: list[list[float]] = []
    for score, target in ordered:
        if grouped and score == grouped[-1][0]:
            grouped[-1][1] += float(target)
            grouped[-1][2] += 1
        else:
            grouped.append([float(score), float(target), 1.0])
    blocks: list[list[float]] = []
    for score, positives, weight in grouped:
        blocks.append([score, positives / weight, weight])
        while len(blocks) >= 2 and blocks[-2][1] > blocks[-1][1]:
            right = blocks.pop()
            left = blocks.pop()
            combined_weight = left[2] + right[2]
            combined_value = (left[1] * left[2] + right[1] * right[2]) / combined_weight
            blocks.append([right[0], combined_value, combined_weight])
    return IsotonicCurve(
        upper_bounds=torch.tensor([block[0] for block in blocks]),
        values=torch.tensor([block[1] for block in blocks]),
    )


def fit_isotonic(logits: Tensor, labels: Tensor) -> IsotonicCalibrator:
    probabilities = torch.softmax(logits.detach().cpu().float(), dim=-1)
    curves = tuple(
        _fit_isotonic_curve(probabilities[:, index], (labels == index).float())
        for index in range(probabilities.shape[1])
    )
    return IsotonicCalibrator(curves)


def fit_calibrated_probe(
    training: ProbeDataset,
    calibration: ProbeDataset,
    *,
    task: ProbeTask,
    training_config: ProbeTrainingConfig | None = None,
    calibration_kind: CalibrationKind = CalibrationKind.TEMPERATURE,
    training_partition: str = "T-controller-train",
    calibration_partition: str = "T-controller-calibration",
) -> CalibratedProbe:
    training_config = training_config or ProbeTrainingConfig()
    overlap = set(training.question_ids) & set(calibration.question_ids)
    if overlap:
        raise DataValidationError(
            f"probe training and calibration IDs overlap: {sorted(overlap)[:3]}"
        )
    group_overlap = set(training.group_ids) & set(calibration.group_ids)
    if group_overlap:
        raise DataValidationError(
            f"probe training and calibration semantic groups overlap: {sorted(group_overlap)[:3]}"
        )
    if training.feature_schema is None or calibration.feature_schema is None:
        raise DataValidationError("probe fitting requires bound activation-feature schemas")
    if training.feature_schema.partition != training_partition:
        raise DataValidationError(
            f"probe training partition must be {training_partition}, "
            f"got {training.feature_schema.partition}"
        )
    if calibration.feature_schema.partition != calibration_partition:
        raise DataValidationError(
            f"probe calibration partition must be {calibration_partition}, "
            f"got {calibration.feature_schema.partition}"
        )
    if not training.feature_schema.is_compatible_extraction(calibration.feature_schema):
        raise DataValidationError("probe training and calibration extraction schemas differ")
    if training.features.shape[1] != calibration.features.shape[1]:
        raise DataValidationError("probe training and calibration feature widths differ")
    train_features, train_labels = encode_probe_task(training, task)
    calibration_features, calibration_labels = encode_probe_task(calibration, task)
    state = fit_probe_state(
        train_features,
        train_labels,
        class_names=_task_labels(task),
        config=training_config,
    )
    logits = state.logits(calibration_features)
    calibrator: Calibrator
    if calibration_kind is CalibrationKind.TEMPERATURE:
        calibrator = fit_temperature(logits, calibration_labels)
    else:
        calibrator = fit_isotonic(logits, calibration_labels)
    return CalibratedProbe(
        task=task,
        state=state,
        calibrator=calibrator,
        training_fingerprint=training.data_fingerprint,
        calibration_fingerprint=calibration.data_fingerprint,
        training_schema=training.feature_schema,
        calibration_schema=calibration.feature_schema,
    )


@dataclass(frozen=True, slots=True)
class ProbeMetrics:
    macro_auroc: float
    macro_f1: float
    brier_score: float
    expected_calibration_error: float
    per_class_auroc: Mapping[str, float]

    def __post_init__(self) -> None:
        values = (
            self.macro_auroc,
            self.macro_f1,
            self.brier_score,
            self.expected_calibration_error,
            *self.per_class_auroc.values(),
        )
        if any(not math.isfinite(value) for value in values):
            raise DataValidationError("probe metrics contain NaN or infinity")
        object.__setattr__(self, "per_class_auroc", MappingProxyType(dict(self.per_class_auroc)))


def _binary_auroc(scores: Tensor, positives: Tensor) -> float:
    ordered_scores, order = torch.sort(scores.detach().cpu().float(), stable=True)
    ordered_labels = positives.detach().cpu().bool()[order]
    ranks = torch.empty_like(ordered_scores, dtype=torch.float64)
    start = 0
    while start < ordered_scores.numel():
        end = start + 1
        while end < ordered_scores.numel() and ordered_scores[end] == ordered_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2
        ranks[start:end] = average_rank
        start = end
    positive_count = int(ordered_labels.sum())
    negative_count = ordered_labels.numel() - positive_count
    if positive_count == 0 or negative_count == 0:
        raise DataValidationError("AUROC requires positive and negative examples")
    rank_sum = ranks[ordered_labels].sum()
    auc = (rank_sum - positive_count * (positive_count + 1) / 2) / (positive_count * negative_count)
    return float(auc)


def evaluate_probabilities(
    probabilities: Tensor,
    labels: Tensor,
    *,
    class_names: Sequence[str],
    ece_bins: int = 15,
) -> ProbeMetrics:
    values = probabilities.detach().cpu().float()
    targets = labels.detach().cpu().long()
    if values.ndim != 2 or values.shape[0] != targets.numel():
        raise DataValidationError("probabilities and labels have incompatible shapes")
    if values.shape[1] != len(class_names) or ece_bins <= 0:
        raise DataValidationError("invalid probability class count or ECE bin count")
    if not torch.isfinite(values).all() or (values < 0).any():
        raise DataValidationError("probabilities must be finite and non-negative")
    if not torch.allclose(values.sum(dim=1), torch.ones(values.shape[0]), atol=1e-5):
        raise DataValidationError("probability rows must sum to one")
    per_class = {
        name: _binary_auroc(values[:, index], targets == index)
        for index, name in enumerate(class_names)
    }
    predictions = values.argmax(dim=1)
    f1_values: list[float] = []
    for index in range(len(class_names)):
        true_positive = int(((predictions == index) & (targets == index)).sum())
        false_positive = int(((predictions == index) & (targets != index)).sum())
        false_negative = int(((predictions != index) & (targets == index)).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    one_hot = F.one_hot(targets, num_classes=len(class_names)).float()
    brier = float(((values - one_hot).pow(2).sum(dim=1)).mean())
    confidence, predictions = values.max(dim=1)
    correct = (predictions == targets).float()
    ece = 0.0
    boundaries = torch.linspace(0, 1, ece_bins + 1)
    for index in range(ece_bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        selected = (confidence > lower) & (confidence <= upper)
        if index == 0:
            selected |= confidence == 0
        if selected.any():
            ece += float(selected.float().mean()) * abs(
                float(correct[selected].mean() - confidence[selected].mean())
            )
    return ProbeMetrics(
        macro_auroc=sum(per_class.values()) / len(per_class),
        macro_f1=sum(f1_values) / len(f1_values),
        brier_score=brier,
        expected_calibration_error=ece,
        per_class_auroc=per_class,
    )


def evaluate_probe(probe: CalibratedProbe, dataset: ProbeDataset) -> ProbeMetrics:
    if dataset.feature_schema is None or not probe.training_schema.is_compatible_representation(
        dataset.feature_schema
    ):
        raise DataValidationError("probe evaluation feature schema differs from training")
    features, labels = encode_probe_task(dataset, probe.task)
    return evaluate_probabilities(
        probe.predict_probabilities(features), labels, class_names=probe.state.labels
    )


@dataclass(frozen=True, slots=True)
class SeparabilityGate:
    passed: bool
    probe_auroc: float
    strongest_baseline_auroc: float
    required_margin: float


def separability_gate(
    probe_metrics: ProbeMetrics,
    baseline_aurocs: Mapping[str, float],
    *,
    minimum_margin: float = 0.02,
) -> SeparabilityGate:
    if not baseline_aurocs:
        raise DataValidationError("separability gate requires at least one confidence baseline")
    if minimum_margin < 0 or not math.isfinite(minimum_margin):
        raise DataValidationError("separability margin must be finite and non-negative")
    strongest = max(float(value) for value in baseline_aurocs.values())
    if not math.isfinite(strongest):
        raise DataValidationError("baseline AUROCs must be finite")
    return SeparabilityGate(
        passed=probe_metrics.macro_auroc >= strongest + minimum_margin,
        probe_auroc=probe_metrics.macro_auroc,
        strongest_baseline_auroc=strongest,
        required_margin=minimum_margin,
    )


def _probe_tensors(probe: CalibratedProbe) -> dict[str, Tensor]:
    tensors = {
        "feature_mean": probe.state.feature_mean,
        "feature_scale": probe.state.feature_scale,
        **{f"parameter.{name}": value for name, value in probe.state.parameters.items()},
    }
    if isinstance(probe.calibrator, TemperatureCalibrator):
        tensors["calibration.temperature"] = torch.tensor(probe.calibrator.temperature)
    else:
        for index, curve in enumerate(probe.calibrator.curves):
            tensors[f"calibration.{index}.upper_bounds"] = curve.upper_bounds
            tensors[f"calibration.{index}.values"] = curve.values
    return {name: value.detach().cpu().contiguous() for name, value in tensors.items()}


def save_calibrated_probe(directory: str | Path, probe: CalibratedProbe) -> None:
    destination = validate_active_study_artifact_paths(
        {"calibrated probe": directory}
    )["calibrated probe"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FrozenArtifactError(f"refusing to overwrite calibrated probe: {destination}")
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        tensor_path = stage / "probe.safetensors"
        save_file(_probe_tensors(probe), tensor_path)
        calibration_kind = (
            CalibrationKind.TEMPERATURE
            if isinstance(probe.calibrator, TemperatureCalibrator)
            else CalibrationKind.ISOTONIC
        )
        metadata_body = {
            "schema_version": probe.schema_version,
            "task": probe.task.value,
            "kind": probe.state.kind.value,
            "labels": list(probe.state.labels),
            "hidden_width": probe.state.hidden_width,
            "calibration_kind": calibration_kind.value,
            "training_fingerprint": probe.training_fingerprint,
            "calibration_fingerprint": probe.calibration_fingerprint,
            "training_schema": probe.training_schema.to_dict(),
            "calibration_schema": probe.calibration_schema.to_dict(),
            "tensor_sha256": sha256_file(tensor_path),
            "tensor_keys": sorted(_probe_tensors(probe)),
        }
        metadata = {**metadata_body, "metadata_digest": stable_hash(metadata_body)}
        (stage / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def load_calibrated_probe(
    directory: str | Path,
    *,
    expected_training_fingerprint: str | None = None,
    expected_calibration_fingerprint: str | None = None,
) -> CalibratedProbe:
    source = Path(directory)
    try:
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read calibrated-probe metadata: {exc}") from exc
    digest = metadata.pop("metadata_digest", None)
    if digest != stable_hash(metadata):
        raise FrozenArtifactError("calibrated-probe metadata digest mismatch")
    if metadata.get("schema_version") != 1:
        raise FrozenArtifactError("unsupported calibrated-probe schema version")
    tensor_path = source / "probe.safetensors"
    if sha256_file(tensor_path) != metadata.get("tensor_sha256"):
        raise FrozenArtifactError("calibrated-probe tensor checksum mismatch")
    tensors = load_file(tensor_path, device="cpu")
    if set(tensors) != set(metadata.get("tensor_keys", [])):
        raise FrozenArtifactError("unexpected or missing calibrated-probe tensors")
    training_fingerprint = metadata.get("training_fingerprint")
    calibration_fingerprint = metadata.get("calibration_fingerprint")
    if (
        expected_training_fingerprint is not None
        and training_fingerprint != expected_training_fingerprint
    ):
        raise FrozenArtifactError("probe was trained on a different data fingerprint")
    if (
        expected_calibration_fingerprint is not None
        and calibration_fingerprint != expected_calibration_fingerprint
    ):
        raise FrozenArtifactError("probe was calibrated on a different data fingerprint")
    try:
        kind = ProbeKind(metadata["kind"])
        task = ProbeTask(metadata["task"])
        labels = tuple(str(value) for value in metadata["labels"])
        parameter_names = (
            ("weight", "bias")
            if kind is ProbeKind.LOGISTIC
            else ("weight1", "bias1", "weight2", "bias2")
        )
        state = ProbeState(
            kind=kind,
            labels=labels,
            feature_mean=tensors["feature_mean"],
            feature_scale=tensors["feature_scale"],
            parameters={name: tensors[f"parameter.{name}"] for name in parameter_names},
            hidden_width=metadata.get("hidden_width"),
        )
        calibration_kind = CalibrationKind(metadata["calibration_kind"])
        calibrator: Calibrator
        if calibration_kind is CalibrationKind.TEMPERATURE:
            calibrator = TemperatureCalibrator(float(tensors["calibration.temperature"]))
        else:
            calibrator = IsotonicCalibrator(
                tuple(
                    IsotonicCurve(
                        tensors[f"calibration.{index}.upper_bounds"],
                        tensors[f"calibration.{index}.values"],
                    )
                    for index in range(len(labels))
                )
            )
        return CalibratedProbe(
            task=task,
            state=state,
            calibrator=calibrator,
            training_fingerprint=str(training_fingerprint),
            calibration_fingerprint=str(calibration_fingerprint),
            training_schema=ActivationFeatureSchema.from_dict(metadata["training_schema"]),
            calibration_schema=ActivationFeatureSchema.from_dict(metadata["calibration_schema"]),
        )
    except (KeyError, TypeError, ValueError, DataValidationError) as exc:
        raise FrozenArtifactError(f"invalid calibrated-probe artifact: {exc}") from exc

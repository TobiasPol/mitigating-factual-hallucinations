"""Prompt-conditioned vector routing and risk-gated intensity for M3."""

from __future__ import annotations

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
from safetensors.torch import load_file, save_file
from torch import Tensor

from mfh.artifact_namespace import validate_active_study_artifact_paths
from mfh.contracts import ActivationSite, Outcome, TokenScope
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.inference.architecture import HookKey
from mfh.inference.hooks import InterventionPlan
from mfh.methods.features import ActivationFeatureSchema
from mfh.methods.probes import (
    CalibratedProbe,
    ProbeDataset,
    ProbeKind,
    ProbeState,
    ProbeTask,
    ProbeTrainingConfig,
    fit_probe_state,
    load_calibrated_probe,
    save_calibrated_probe,
)
from mfh.provenance import sha256_file, sha256_path, stable_hash

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class RouterKind(StrEnum):
    NEAREST_CENTROID = "nearest_centroid"
    LINEAR_SOFTMAX = "linear_softmax"
    TWO_LAYER_MLP = "two_layer_mlp"


class AlphaMode(StrEnum):
    FIXED = "fixed"
    RISK_GATED = "risk_gated"
    HARD_THRESHOLD = "risk_gated_hard_threshold"


def _feature_matrix(features: Tensor, *, width: int | None = None) -> Tensor:
    values = features.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if values.ndim == 1:
        values = values.unsqueeze(0)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise DataValidationError("router features must have shape [rows, width]")
    if width is not None and values.shape[1] != width:
        raise DataValidationError(f"router expected width {width}, got {values.shape[1]}")
    if not torch.isfinite(values).all():
        raise DataValidationError("router features contain NaN or infinity")
    return values


def fit_kmeans(
    features: Tensor,
    cluster_count: int,
    *,
    seed: int = 17,
    max_iterations: int = 100,
) -> tuple[Tensor, Tensor]:
    """Deterministic CPU k-means with seeded k-means++ initialization."""

    values = _feature_matrix(features)
    if type(cluster_count) is not int or not 1 <= cluster_count <= values.shape[0]:
        raise DataValidationError("cluster count must be between one and the row count")
    if type(seed) is not int or seed < 0:
        raise DataValidationError("k-means seed must be a non-negative exact integer")
    if type(max_iterations) is not int or max_iterations <= 0:
        raise DataValidationError("k-means max_iterations must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    first = int(torch.randint(values.shape[0], (1,), generator=generator))
    center_indices = [first]
    centers = [values[first].clone()]
    while len(centers) < cluster_count:
        current = torch.stack(centers)
        squared = torch.cdist(values, current).pow(2).min(dim=1).values
        squared[torch.tensor(center_indices)] = 0
        if float(squared.sum()) <= 0:
            remaining = next(
                index for index in range(values.shape[0]) if index not in center_indices
            )
            selected = remaining
        else:
            selected = int(torch.multinomial(squared, 1, generator=generator))
        center_indices.append(selected)
        centers.append(values[selected].clone())
    center_matrix = torch.stack(centers)
    previous: Tensor | None = None
    assignments = torch.zeros(values.shape[0], dtype=torch.long)
    for _ in range(max_iterations):
        distances = torch.cdist(values, center_matrix).pow(2)
        assignments = distances.argmin(dim=1)
        if previous is not None and torch.equal(previous, assignments):
            break
        previous = assignments.clone()
        nearest_distance = distances.min(dim=1).values
        for cluster in range(cluster_count):
            selected_rows = assignments == cluster
            if selected_rows.any():
                center_matrix[cluster] = values[selected_rows].mean(dim=0)
            else:
                replacement = int(nearest_distance.argmax())
                center_matrix[cluster] = values[replacement]
                assignments[replacement] = cluster
                nearest_distance[replacement] = -1
    final_assignments = torch.cdist(values, center_matrix).pow(2).argmin(dim=1)
    if set(final_assignments.tolist()) != set(range(cluster_count)):
        raise DataValidationError("k-means produced an empty cluster; reduce K")
    return center_matrix, final_assignments


@dataclass(frozen=True, slots=True)
class RoutedVectorBank:
    centers: Tensor
    directions: Mapping[HookKey, Tensor]
    correct_counts: tuple[int, ...]
    incorrect_counts: tuple[int, ...]
    data_fingerprint: str
    feature_schema: ActivationFeatureSchema
    source_artifact_sha256: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise DataValidationError("unsupported routed-vector-bank schema version")
        centers = _feature_matrix(self.centers)
        cluster_count = centers.shape[0]
        if len(self.correct_counts) != cluster_count or len(self.incorrect_counts) != cluster_count:
            raise DataValidationError("routed vector count metadata has the wrong length")
        if any(count <= 0 for count in (*self.correct_counts, *self.incorrect_counts)):
            raise DataValidationError("every routed region requires correct and incorrect rows")
        if not self.directions:
            raise DataValidationError("routed vector bank cannot be empty")
        directions: dict[HookKey, Tensor] = {}
        for key, value in self.directions.items():
            tensor = value.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
            if tensor.ndim != 2 or tensor.shape[0] != cluster_count or tensor.shape[1] == 0:
                raise DataValidationError(
                    f"routed directions at {key.artifact_key} have an invalid shape"
                )
            if not torch.isfinite(tensor).all():
                raise DataValidationError("routed directions contain NaN or infinity")
            norms = torch.linalg.vector_norm(tensor, dim=1)
            if not torch.allclose(norms, torch.ones_like(norms), rtol=1e-4, atol=1e-5):
                raise DataValidationError("each routed direction must have unit L2 norm")
            directions[key] = tensor
        if not _SHA256.fullmatch(self.data_fingerprint):
            raise DataValidationError("routed vector bank requires a data SHA-256 fingerprint")
        if self.source_artifact_sha256 is not None and not _SHA256.fullmatch(
            self.source_artifact_sha256
        ):
            raise DataValidationError(
                "routed vector bank source artifact must be a SHA-256 fingerprint"
            )
        if self.feature_schema.width != centers.shape[1]:
            raise DataValidationError("routed vector schema width differs from router features")
        object.__setattr__(self, "centers", centers.clone())
        object.__setattr__(self, "directions", MappingProxyType(directions))

    @property
    def cluster_count(self) -> int:
        return int(self.centers.shape[0])

    @property
    def router_width(self) -> int:
        return int(self.centers.shape[1])

    def mix(self, weights: Tensor) -> dict[HookKey, Tensor]:
        values = weights.detach().to(device="cpu", dtype=torch.float32)
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.ndim != 2 or values.shape[1] != self.cluster_count:
            raise DataValidationError("routing weights have the wrong shape")
        if (values < 0).any() or not torch.allclose(
            values.sum(dim=1), torch.ones(values.shape[0]), atol=1e-5
        ):
            raise DataValidationError("routing weights must be non-negative and sum to one")
        return {key: values @ directions for key, directions in self.directions.items()}


def fit_routed_vector_bank(
    dataset: ProbeDataset,
    activations: Mapping[HookKey, Tensor],
    *,
    cluster_count: int,
    seed: int = 17,
    minimum_class_count: int = 1,
    source_artifact_sha256: str | None = None,
) -> tuple[RoutedVectorBank, Tensor]:
    all_features = _feature_matrix(dataset.features)
    if dataset.feature_schema is None:
        raise DataValidationError("routed vector fitting requires a bound feature schema")
    if dataset.feature_schema.partition != "T-steer":
        raise DataValidationError("routed vector fitting is restricted to T-steer")
    if not activations:
        raise DataValidationError("routed-vector inputs have incompatible row counts")
    if minimum_class_count <= 0:
        raise DataValidationError("minimum class count must be positive")
    eligible = torch.tensor(
        [
            outcome in {Outcome.CORRECT, Outcome.INCORRECT}
            for outcome in dataset.outcomes
        ],
        dtype=torch.bool,
    )
    if not bool(eligible.any()):
        raise DataValidationError("T-steer vector rows contain no C/I outcomes")
    features = all_features[eligible]
    eligible_outcomes = tuple(
        outcome for outcome in dataset.outcomes if outcome in {Outcome.CORRECT, Outcome.INCORRECT}
    )
    centers, assignments = fit_kmeans(features, cluster_count, seed=seed)
    activation_values: dict[HookKey, Tensor] = {}
    for key, value in activations.items():
        tensor = _feature_matrix(value)
        if tensor.shape[0] != all_features.shape[0]:
            raise DataValidationError(f"activation rows differ at {key.artifact_key}")
        activation_values[key] = tensor[eligible]
    correct_counts: list[int] = []
    incorrect_counts: list[int] = []
    directions = {
        key: torch.empty(cluster_count, values.shape[1])
        for key, values in activation_values.items()
    }
    for cluster in range(cluster_count):
        in_cluster = assignments == cluster
        correct = (
            torch.tensor(
                [outcome is Outcome.CORRECT for outcome in eligible_outcomes], dtype=torch.bool
            )
            & in_cluster
        )
        incorrect = (
            torch.tensor(
                [outcome is Outcome.INCORRECT for outcome in eligible_outcomes], dtype=torch.bool
            )
            & in_cluster
        )
        correct_count, incorrect_count = int(correct.sum()), int(incorrect.sum())
        if correct_count < minimum_class_count or incorrect_count < minimum_class_count:
            raise DataValidationError(
                f"cluster {cluster} has C={correct_count}, I={incorrect_count}; "
                f"need at least {minimum_class_count} of each"
            )
        correct_counts.append(correct_count)
        incorrect_counts.append(incorrect_count)
        for key, values in activation_values.items():
            difference = values[correct].mean(dim=0) - values[incorrect].mean(dim=0)
            norm = torch.linalg.vector_norm(difference)
            if not torch.isfinite(norm) or float(norm) <= 0:
                raise DataValidationError(
                    f"cluster {cluster} has a zero direction at {key.artifact_key}"
                )
            directions[key][cluster] = difference / norm
    return (
        RoutedVectorBank(
            centers=centers,
            directions=directions,
            correct_counts=tuple(correct_counts),
            incorrect_counts=tuple(incorrect_counts),
            data_fingerprint=dataset.data_fingerprint,
            feature_schema=dataset.feature_schema,
            source_artifact_sha256=source_artifact_sha256,
        ),
        assignments,
    )


@dataclass(frozen=True, slots=True)
class AdaptiveRouter:
    kind: RouterKind
    centers: Tensor
    training_fingerprint: str
    feature_schema: ActivationFeatureSchema
    classifier: ProbeState | None = None
    distance_temperature: float = 1.0

    def __post_init__(self) -> None:
        centers = _feature_matrix(self.centers)
        if not isinstance(self.kind, RouterKind):
            raise DataValidationError("router kind must be an exact RouterKind")
        if not _SHA256.fullmatch(self.training_fingerprint):
            raise DataValidationError("router training fingerprint must be a SHA-256 digest")
        if (
            isinstance(self.distance_temperature, bool)
            or not isinstance(self.distance_temperature, int | float)
            or not math.isfinite(float(self.distance_temperature))
            or float(self.distance_temperature) <= 0
        ):
            raise DataValidationError("router distance temperature must be positive")
        if self.feature_schema.width != centers.shape[1]:
            raise DataValidationError("router feature schema width is invalid")
        if centers.shape[0] > 1:
            if self.kind is RouterKind.NEAREST_CENTROID and self.classifier is not None:
                raise DataValidationError("nearest-centroid routers do not use a classifier")
            if self.kind is not RouterKind.NEAREST_CENTROID:
                if self.classifier is None:
                    raise DataValidationError("learned routers require a classifier")
                if self.classifier.input_width != centers.shape[1]:
                    raise DataValidationError("router classifier input width is invalid")
                if len(self.classifier.labels) != centers.shape[0]:
                    raise DataValidationError("router classifier output width is invalid")
        object.__setattr__(self, "centers", centers.clone())
        object.__setattr__(self, "distance_temperature", float(self.distance_temperature))

    @property
    def cluster_count(self) -> int:
        return int(self.centers.shape[0])

    @property
    def input_width(self) -> int:
        return int(self.centers.shape[1])

    def weights(self, features: Tensor) -> Tensor:
        values = _feature_matrix(features, width=self.input_width)
        if self.cluster_count == 1:
            return torch.ones(values.shape[0], 1)
        if self.kind is RouterKind.NEAREST_CENTROID:
            distances = torch.cdist(values, self.centers).pow(2)
            return torch.softmax(-distances / self.distance_temperature, dim=1)
        assert self.classifier is not None
        return torch.softmax(self.classifier.logits(values), dim=1)


def fit_adaptive_router(
    dataset: ProbeDataset,
    assignments: Tensor,
    centers: Tensor,
    *,
    kind: RouterKind,
    seed: int = 17,
    hidden_width: int = 64,
    epochs: int = 300,
    distance_temperature: float = 1.0,
) -> AdaptiveRouter:
    if not isinstance(kind, RouterKind):
        raise DataValidationError("adaptive router kind must be an exact RouterKind")
    values = _feature_matrix(dataset.features)
    if dataset.feature_schema is None:
        raise DataValidationError("adaptive router fitting requires a bound feature schema")
    if dataset.feature_schema.partition != "T-controller-train":
        raise DataValidationError("adaptive router fitting is restricted to T-controller-train")
    center_values = _feature_matrix(centers, width=values.shape[1])
    labels = assignments.detach().to(device="cpu", dtype=torch.long)
    if labels.ndim != 1 or labels.numel() != values.shape[0]:
        raise DataValidationError("router assignments have the wrong shape")
    cluster_count = center_values.shape[0]
    if set(labels.tolist()) != set(range(cluster_count)):
        raise DataValidationError("router assignments must contain every cluster")
    classifier: ProbeState | None = None
    if cluster_count > 1 and kind is not RouterKind.NEAREST_CENTROID:
        probe_kind = (
            ProbeKind.LOGISTIC if kind is RouterKind.LINEAR_SOFTMAX else ProbeKind.TWO_LAYER_MLP
        )
        classifier = fit_probe_state(
            values,
            labels,
            class_names=tuple(str(index) for index in range(cluster_count)),
            config=ProbeTrainingConfig(
                kind=probe_kind,
                hidden_width=hidden_width,
                epochs=epochs,
                seed=seed,
            ),
        )
    return AdaptiveRouter(
        kind=kind,
        centers=center_values,
        classifier=classifier,
        distance_temperature=distance_temperature,
        training_fingerprint=dataset.data_fingerprint,
        feature_schema=dataset.feature_schema,
    )


def assign_to_vector_regions(dataset: ProbeDataset, bank: RoutedVectorBank) -> Tensor:
    """Label disjoint T-controller rows using centers learned on T-steer."""

    if dataset.feature_schema is None:
        raise DataValidationError("region assignment requires a bound feature schema")
    if dataset.feature_schema.partition != "T-controller-train":
        raise DataValidationError("region assignment is restricted to T-controller-train")
    if not bank.feature_schema.is_compatible_representation(dataset.feature_schema):
        raise DataValidationError("controller rows and vector centers use incompatible features")
    values = _feature_matrix(dataset.features, width=bank.router_width)
    return torch.cdist(values, bank.centers).pow(2).argmin(dim=1)


def _router_tensors(router: AdaptiveRouter) -> dict[str, Tensor]:
    tensors = {"centers": router.centers}
    if router.classifier is not None:
        tensors["classifier.feature_mean"] = router.classifier.feature_mean
        tensors["classifier.feature_scale"] = router.classifier.feature_scale
        tensors.update(
            {
                f"classifier.parameter.{name}": value
                for name, value in router.classifier.parameters.items()
            }
        )
    return {name: value.detach().cpu().contiguous() for name, value in tensors.items()}


def save_adaptive_router(directory: str | Path, router: AdaptiveRouter) -> None:
    destination = validate_active_study_artifact_paths(
        {"adaptive router": directory}
    )["adaptive router"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FrozenArtifactError(f"refusing to overwrite adaptive router: {destination}")
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        tensor_path = stage / "router.safetensors"
        tensors = _router_tensors(router)
        save_file(tensors, tensor_path)
        classifier = router.classifier
        metadata_body = {
            "schema_version": 1,
            "kind": router.kind.value,
            "training_fingerprint": router.training_fingerprint,
            "feature_schema": router.feature_schema.to_dict(),
            "distance_temperature": router.distance_temperature,
            "classifier": (
                None
                if classifier is None
                else {
                    "kind": classifier.kind.value,
                    "labels": list(classifier.labels),
                    "hidden_width": classifier.hidden_width,
                }
            ),
            "tensor_keys": sorted(tensors),
            "tensor_sha256": sha256_file(tensor_path),
        }
        metadata = {**metadata_body, "metadata_digest": stable_hash(metadata_body)}
        (stage / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def load_adaptive_router(
    directory: str | Path, *, expected_training_fingerprint: str | None = None
) -> AdaptiveRouter:
    source = Path(directory)
    try:
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read adaptive-router metadata: {exc}") from exc
    if type(metadata) is not dict:
        raise FrozenArtifactError("adaptive-router metadata root must be an object")
    digest = metadata.pop("metadata_digest", None)
    if digest != stable_hash(metadata):
        raise FrozenArtifactError("adaptive-router metadata digest mismatch")
    expected_metadata = {
        "schema_version",
        "kind",
        "training_fingerprint",
        "feature_schema",
        "distance_temperature",
        "classifier",
        "tensor_keys",
        "tensor_sha256",
    }
    if set(metadata) != expected_metadata:
        raise FrozenArtifactError("adaptive-router metadata keys differ")
    if type(metadata.get("schema_version")) is not int or metadata["schema_version"] != 1:
        raise FrozenArtifactError("unsupported adaptive-router schema version")
    if (
        type(metadata["kind"]) is not str
        or type(metadata["training_fingerprint"]) is not str
        or _SHA256.fullmatch(metadata["training_fingerprint"]) is None
        or type(metadata["feature_schema"]) is not dict
        or isinstance(metadata["distance_temperature"], bool)
        or not isinstance(metadata["distance_temperature"], int | float)
        or type(metadata["tensor_keys"]) is not list
        or any(type(value) is not str for value in metadata["tensor_keys"])
        or len(set(metadata["tensor_keys"])) != len(metadata["tensor_keys"])
        or type(metadata["tensor_sha256"]) is not str
        or _SHA256.fullmatch(metadata["tensor_sha256"]) is None
    ):
        raise FrozenArtifactError("adaptive-router metadata types differ")
    classifier_value = metadata["classifier"]
    if classifier_value is not None and (
            type(classifier_value) is not dict
            or set(classifier_value) != {"kind", "labels", "hidden_width"}
            or type(classifier_value["kind"]) is not str
            or type(classifier_value["labels"]) is not list
            or any(type(value) is not str for value in classifier_value["labels"])
            or (
                classifier_value["hidden_width"] is not None
                and type(classifier_value["hidden_width"]) is not int
            )
    ):
        raise FrozenArtifactError("adaptive-router classifier metadata types differ")
    tensor_path = source / "router.safetensors"
    if sha256_file(tensor_path) != metadata.get("tensor_sha256"):
        raise FrozenArtifactError("adaptive-router tensor checksum mismatch")
    if (
        expected_training_fingerprint is not None
        and metadata.get("training_fingerprint") != expected_training_fingerprint
    ):
        raise FrozenArtifactError("adaptive router has a different training fingerprint")
    try:
        tensors = load_file(tensor_path, device="cpu")
        if set(tensors) != set(metadata["tensor_keys"]):
            raise FrozenArtifactError("unexpected or missing adaptive-router tensors")
        classifier_metadata = metadata["classifier"]
        classifier: ProbeState | None = None
        if classifier_metadata is not None:
            classifier_kind = ProbeKind(classifier_metadata["kind"])
            parameter_names = (
                ("weight", "bias")
                if classifier_kind is ProbeKind.LOGISTIC
                else ("weight1", "bias1", "weight2", "bias2")
            )
            classifier = ProbeState(
                kind=classifier_kind,
                labels=tuple(classifier_metadata["labels"]),
                feature_mean=tensors["classifier.feature_mean"],
                feature_scale=tensors["classifier.feature_scale"],
                parameters={
                    name: tensors[f"classifier.parameter.{name}"] for name in parameter_names
                },
                hidden_width=classifier_metadata.get("hidden_width"),
            )
        return AdaptiveRouter(
            kind=RouterKind(metadata["kind"]),
            centers=tensors["centers"],
            training_fingerprint=metadata["training_fingerprint"],
            feature_schema=ActivationFeatureSchema.from_dict(metadata["feature_schema"]),
            classifier=classifier,
            distance_temperature=metadata["distance_temperature"],
        )
    except (KeyError, TypeError, ValueError, DataValidationError) as exc:
        raise FrozenArtifactError(f"invalid adaptive-router artifact: {exc}") from exc


@dataclass(frozen=True, slots=True)
class LayerSelector:
    candidate_layers: tuple[int, ...]
    router: AdaptiveRouter

    def __post_init__(self) -> None:
        if type(self.candidate_layers) is not tuple or not 2 <= len(self.candidate_layers) <= 3:
            raise DataValidationError("confirmatory layer routing must use two or three layers")
        if len(set(self.candidate_layers)) != len(self.candidate_layers) or any(
            type(layer) is not int or layer < 0 for layer in self.candidate_layers
        ):
            raise DataValidationError("candidate layers must be unique and non-negative")
        if self.router.cluster_count != len(self.candidate_layers):
            raise DataValidationError("layer router output does not match candidate layers")

    def select(self, features: Tensor) -> Tensor:
        indices = self.router.weights(features).argmax(dim=1)
        candidates = torch.tensor(self.candidate_layers, dtype=torch.long)
        return candidates[indices]


def fit_layer_selector(
    dataset: ProbeDataset,
    best_layers: Sequence[int],
    *,
    candidate_layers: tuple[int, ...],
    kind: RouterKind,
    seed: int = 17,
    epochs: int = 300,
) -> LayerSelector:
    values = _feature_matrix(dataset.features)
    if len(best_layers) != values.shape[0]:
        raise DataValidationError("best-layer labels have the wrong row count")
    layer_to_index = {layer: index for index, layer in enumerate(candidate_layers)}
    try:
        labels = torch.tensor([layer_to_index[layer] for layer in best_layers], dtype=torch.long)
    except KeyError as exc:
        raise DataValidationError(f"best layer {exc.args[0]} is not a candidate") from exc
    centers = torch.stack(
        [values[labels == index].mean(dim=0) for index in range(len(candidate_layers))]
    )
    router = fit_adaptive_router(
        dataset,
        labels,
        centers,
        kind=kind,
        seed=seed,
        epochs=epochs,
    )
    return LayerSelector(candidate_layers, router)


@dataclass(frozen=True, slots=True)
class AlphaController:
    mode: AlphaMode
    alpha_max: float
    beta: float = 12.0
    threshold: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.mode, AlphaMode):
            raise DataValidationError("alpha mode must be an exact AlphaMode")
        if (
            isinstance(self.alpha_max, bool)
            or not isinstance(self.alpha_max, int | float)
            or not math.isfinite(float(self.alpha_max))
            or float(self.alpha_max) < 0
        ):
            raise DataValidationError("alpha_max must be finite and non-negative")
        if (
            isinstance(self.beta, bool)
            or not isinstance(self.beta, int | float)
            or not math.isfinite(float(self.beta))
            or float(self.beta) <= 0
        ):
            raise DataValidationError("risk-gate beta must be finite and positive")
        if (
            isinstance(self.threshold, bool)
            or not isinstance(self.threshold, int | float)
            or not math.isfinite(float(self.threshold))
            or not 0 <= float(self.threshold) <= 1
        ):
            raise DataValidationError("risk-gate threshold must be in [0, 1]")
        object.__setattr__(self, "alpha_max", float(self.alpha_max))
        object.__setattr__(self, "beta", float(self.beta))
        object.__setattr__(self, "threshold", float(self.threshold))

    def alpha(self, incorrect_probability: Tensor) -> Tensor:
        risk = incorrect_probability.detach().to(device="cpu", dtype=torch.float32)
        if not torch.isfinite(risk).all() or (risk < 0).any() or (risk > 1).any():
            raise DataValidationError("incorrect probabilities must be in [0, 1]")
        if self.mode is AlphaMode.FIXED:
            return torch.full_like(risk, self.alpha_max)
        values = self.alpha_max * torch.sigmoid(self.beta * (risk - self.threshold))
        if self.mode is AlphaMode.HARD_THRESHOLD:
            values = torch.where(risk >= self.threshold, values, torch.zeros_like(values))
        return values


@dataclass(frozen=True, slots=True)
class AdaptiveBatchDecision:
    class_labels: tuple[str, ...]
    probabilities: Tensor
    routing_weights: Tensor
    alphas: Tensor
    selected_layers: Tensor
    directions: Mapping[HookKey, Tensor]

    def __post_init__(self) -> None:
        row_count = self.probabilities.shape[0]
        if self.probabilities.ndim != 2 or self.routing_weights.ndim != 2:
            raise DataValidationError("adaptive decision probabilities must be matrices")
        if self.routing_weights.shape[0] != row_count or self.alphas.shape != (row_count,):
            raise DataValidationError("adaptive decision row counts differ")
        if self.selected_layers.shape != (row_count,):
            raise DataValidationError("adaptive selected layers have the wrong shape")
        directions: dict[HookKey, Tensor] = {}
        for key, value in self.directions.items():
            if value.ndim != 2 or value.shape[0] != row_count:
                raise DataValidationError("adaptive direction rows differ")
            directions[key] = value.detach().cpu().float().contiguous().clone()
        object.__setattr__(self, "directions", MappingProxyType(directions))

    def plans_for_row(
        self,
        row: int,
        *,
        token_scope: TokenScope,
        rms_relative: bool = True,
        decay: float = 0.5,
    ) -> dict[HookKey, InterventionPlan]:
        if not 0 <= row < self.probabilities.shape[0]:
            raise DataValidationError("adaptive decision row is out of range")
        layer = int(self.selected_layers[row])
        return {
            key: InterventionPlan(
                direction=value[row],
                alpha=float(self.alphas[row]),
                token_scope=token_scope,
                rms_relative=rms_relative,
                decay=decay,
            )
            for key, value in self.directions.items()
            if key.layer == layer
        }


@dataclass(frozen=True, slots=True)
class AdaptiveController:
    risk_probe: CalibratedProbe
    vector_bank: RoutedVectorBank
    vector_router: AdaptiveRouter
    alpha_controller: AlphaController
    fixed_layer: int | None = None
    layer_selector: LayerSelector | None = None

    def __post_init__(self) -> None:
        if (
            self.risk_probe.task is not ProbeTask.CORRECT_INCORRECT_ABSTENTION
            or self.risk_probe.state.labels
            != (Outcome.CORRECT.value, Outcome.INCORRECT.value, Outcome.ABSTENTION.value)
        ):
            raise DataValidationError("adaptive risk probe must estimate calibrated P(C),P(I),P(A)")
        if self.vector_router.cluster_count != self.vector_bank.cluster_count:
            raise DataValidationError("vector router and bank cluster counts differ")
        if self.vector_router.input_width != self.vector_bank.router_width:
            raise DataValidationError("vector router and bank feature widths differ")
        if not torch.equal(self.vector_router.centers, self.vector_bank.centers):
            raise DataValidationError(
                "vector router centers must exactly match the ordered frozen bank centers"
            )
        if self.vector_router.training_fingerprint == self.vector_bank.data_fingerprint:
            raise DataValidationError(
                "vector router and bank must use disjoint controller/steer rows"
            )
        if self.vector_router.feature_schema.partition != "T-controller-train":
            raise DataValidationError("vector router must be trained on T-controller-train")
        if self.vector_bank.feature_schema.partition != "T-steer":
            raise DataValidationError("vector bank must be constructed on T-steer")
        schemas = (
            self.risk_probe.training_schema,
            self.vector_bank.feature_schema,
            self.vector_router.feature_schema,
        )
        if not all(schemas[0].is_compatible_representation(schema) for schema in schemas[1:]):
            raise DataValidationError("adaptive controller feature schemas are incompatible")
        if (self.fixed_layer is None) == (self.layer_selector is None):
            raise DataValidationError("configure exactly one of fixed_layer or layer_selector")
        if self.fixed_layer is not None and (
            type(self.fixed_layer) is not int or self.fixed_layer < 0
        ):
            raise DataValidationError("adaptive fixed layer must be a non-negative exact integer")
        available_layers = {key.layer for key in self.vector_bank.directions}
        selected = (
            {self.fixed_layer}
            if self.fixed_layer is not None
            else set(self.layer_selector.candidate_layers if self.layer_selector else ())
        )
        if not selected <= available_layers:
            raise DataValidationError("adaptive layer selection references absent vectors")
        if self.layer_selector is not None and not schemas[0].is_compatible_representation(
            self.layer_selector.router.feature_schema
        ):
            raise DataValidationError("layer selector feature schema is incompatible")

    def decide(self, features: Tensor) -> AdaptiveBatchDecision:
        values = _feature_matrix(features)
        if values.shape[1] != self.risk_probe.state.input_width:
            raise DataValidationError("risk probe feature width differs from controller input")
        probabilities = self.risk_probe.predict_probabilities(values)
        incorrect_index = self.risk_probe.state.labels.index(Outcome.INCORRECT.value)
        alphas = self.alpha_controller.alpha(probabilities[:, incorrect_index])
        routing_weights = self.vector_router.weights(values)
        directions = self.vector_bank.mix(routing_weights)
        if self.layer_selector is not None:
            selected_layers = self.layer_selector.select(values)
        else:
            assert self.fixed_layer is not None
            selected_layers = torch.full((values.shape[0],), self.fixed_layer, dtype=torch.long)
        return AdaptiveBatchDecision(
            class_labels=self.risk_probe.state.labels,
            probabilities=probabilities,
            routing_weights=routing_weights,
            alphas=alphas,
            selected_layers=selected_layers,
            directions=directions,
        )


def save_routed_vector_bank(directory: str | Path, bank: RoutedVectorBank) -> None:
    destination = validate_active_study_artifact_paths(
        {"routed vector bank": directory}
    )["routed vector bank"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FrozenArtifactError(f"refusing to overwrite routed vector bank: {destination}")
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        tensor_path = stage / "vectors.safetensors"
        tensors = {"centers": bank.centers}
        tensors.update(
            {f"direction.{key.artifact_key}": value for key, value in bank.directions.items()}
        )
        save_file({key: value.contiguous() for key, value in tensors.items()}, tensor_path)
        metadata_body = {
            "schema_version": bank.schema_version,
            "data_fingerprint": bank.data_fingerprint,
            "source_artifact_sha256": bank.source_artifact_sha256,
            "feature_schema": bank.feature_schema.to_dict(),
            "correct_counts": list(bank.correct_counts),
            "incorrect_counts": list(bank.incorrect_counts),
            "tensor_sha256": sha256_file(tensor_path),
            "hooks": [
                {"layer": key.layer, "site": key.site.value, "tensor_key": key.artifact_key}
                for key in sorted(bank.directions)
            ],
        }
        metadata = {**metadata_body, "metadata_digest": stable_hash(metadata_body)}
        (stage / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def load_routed_vector_bank(
    directory: str | Path, *, expected_data_fingerprint: str | None = None
) -> RoutedVectorBank:
    source = Path(directory)
    try:
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read routed-vector metadata: {exc}") from exc
    if type(metadata) is not dict:
        raise FrozenArtifactError("routed-vector metadata root must be an object")
    digest = metadata.pop("metadata_digest", None)
    if digest != stable_hash(metadata):
        raise FrozenArtifactError("routed-vector metadata digest mismatch")
    expected_metadata = {
        "schema_version",
        "data_fingerprint",
        "source_artifact_sha256",
        "feature_schema",
        "correct_counts",
        "incorrect_counts",
        "tensor_sha256",
        "hooks",
    }
    if set(metadata) != expected_metadata:
        raise FrozenArtifactError("routed-vector metadata keys differ")
    if type(metadata.get("schema_version")) is not int or metadata["schema_version"] != 1:
        raise FrozenArtifactError("unsupported routed-vector schema version")
    if (
        type(metadata["data_fingerprint"]) is not str
        or _SHA256.fullmatch(metadata["data_fingerprint"]) is None
        or (
            metadata["source_artifact_sha256"] is not None
            and (
                type(metadata["source_artifact_sha256"]) is not str
                or _SHA256.fullmatch(metadata["source_artifact_sha256"]) is None
            )
        )
        or type(metadata["feature_schema"]) is not dict
        or type(metadata["correct_counts"]) is not list
        or type(metadata["incorrect_counts"]) is not list
        or any(type(value) is not int for value in metadata["correct_counts"])
        or any(type(value) is not int for value in metadata["incorrect_counts"])
        or type(metadata["tensor_sha256"]) is not str
        or _SHA256.fullmatch(metadata["tensor_sha256"]) is None
        or type(metadata["hooks"]) is not list
    ):
        raise FrozenArtifactError("routed-vector metadata types differ")
    if any(
        type(entry) is not dict
        or set(entry) != {"layer", "site", "tensor_key"}
        or type(entry["layer"]) is not int
        or type(entry["site"]) is not str
        or type(entry["tensor_key"]) is not str
        for entry in metadata["hooks"]
    ):
        raise FrozenArtifactError("routed-vector hook metadata types differ")
    tensor_path = source / "vectors.safetensors"
    if sha256_file(tensor_path) != metadata.get("tensor_sha256"):
        raise FrozenArtifactError("routed-vector tensor checksum mismatch")
    fingerprint = metadata["data_fingerprint"]
    if expected_data_fingerprint is not None and fingerprint != expected_data_fingerprint:
        raise FrozenArtifactError("routed vectors were trained on a different fingerprint")
    try:
        tensors = load_file(tensor_path, device="cpu")
        hooks = metadata["hooks"]
        directions: dict[HookKey, Tensor] = {}
        expected_keys = {"centers"}
        for entry in hooks:
            key = HookKey(entry["layer"], ActivationSite(entry["site"]))
            if entry["tensor_key"] != key.artifact_key:
                raise FrozenArtifactError("routed-vector hook key mismatch")
            tensor_key = f"direction.{key.artifact_key}"
            expected_keys.add(tensor_key)
            directions[key] = tensors[tensor_key]
        if set(tensors) != expected_keys:
            raise FrozenArtifactError("unexpected or missing routed-vector tensors")
        return RoutedVectorBank(
            centers=tensors["centers"],
            directions=directions,
            correct_counts=tuple(metadata["correct_counts"]),
            incorrect_counts=tuple(metadata["incorrect_counts"]),
            data_fingerprint=fingerprint,
            feature_schema=ActivationFeatureSchema.from_dict(metadata["feature_schema"]),
            source_artifact_sha256=metadata["source_artifact_sha256"],
        )
    except (KeyError, TypeError, ValueError, DataValidationError) as exc:
        raise FrozenArtifactError(f"invalid routed-vector artifact: {exc}") from exc


def save_adaptive_controller(directory: str | Path, controller: AdaptiveController) -> None:
    """Freeze every component required to reproduce one M3 decision."""

    destination = validate_active_study_artifact_paths(
        {"adaptive controller": directory}
    )["adaptive controller"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FrozenArtifactError(f"refusing to overwrite adaptive controller: {destination}")
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        save_calibrated_probe(stage / "risk_probe", controller.risk_probe)
        save_routed_vector_bank(stage / "vector_bank", controller.vector_bank)
        save_adaptive_router(stage / "vector_router", controller.vector_router)
        components = {
            "risk_probe": sha256_path(stage / "risk_probe"),
            "vector_bank": sha256_path(stage / "vector_bank"),
            "vector_router": sha256_path(stage / "vector_router"),
        }
        candidate_layers: list[int] | None = None
        if controller.layer_selector is not None:
            save_adaptive_router(stage / "layer_router", controller.layer_selector.router)
            components["layer_router"] = sha256_path(stage / "layer_router")
            candidate_layers = list(controller.layer_selector.candidate_layers)
        metadata_body = {
            "schema_version": 1,
            "alpha_controller": {
                "mode": controller.alpha_controller.mode.value,
                "alpha_max": controller.alpha_controller.alpha_max,
                "beta": controller.alpha_controller.beta,
                "threshold": controller.alpha_controller.threshold,
            },
            "fixed_layer": controller.fixed_layer,
            "candidate_layers": candidate_layers,
            "feature_schema_digest": controller.risk_probe.training_schema.digest,
            "components": components,
        }
        metadata = {**metadata_body, "metadata_digest": stable_hash(metadata_body)}
        (stage / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def load_adaptive_controller(directory: str | Path) -> AdaptiveController:
    source = Path(directory)
    try:
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read adaptive-controller metadata: {exc}") from exc
    if type(metadata) is not dict:
        raise FrozenArtifactError("adaptive-controller metadata root must be an object")
    digest = metadata.pop("metadata_digest", None)
    if digest != stable_hash(metadata):
        raise FrozenArtifactError("adaptive-controller metadata digest mismatch")
    expected_metadata = {
        "schema_version",
        "alpha_controller",
        "fixed_layer",
        "candidate_layers",
        "feature_schema_digest",
        "components",
    }
    if set(metadata) != expected_metadata:
        raise FrozenArtifactError("adaptive-controller metadata keys differ")
    if type(metadata.get("schema_version")) is not int or metadata["schema_version"] != 1:
        raise FrozenArtifactError("unsupported adaptive-controller schema version")
    alpha_value = metadata["alpha_controller"]
    fixed_layer_value = metadata["fixed_layer"]
    candidate_layers_value = metadata["candidate_layers"]
    if (
        type(alpha_value) is not dict
        or set(alpha_value) != {"mode", "alpha_max", "beta", "threshold"}
        or type(alpha_value["mode"]) is not str
        or any(
            isinstance(alpha_value[name], bool)
            or not isinstance(alpha_value[name], int | float)
            for name in ("alpha_max", "beta", "threshold")
        )
        or (
            fixed_layer_value is not None and type(fixed_layer_value) is not int
        )
        or (
            candidate_layers_value is not None
            and (
                type(candidate_layers_value) is not list
                or any(type(value) is not int for value in candidate_layers_value)
            )
        )
        or (fixed_layer_value is None) == (candidate_layers_value is None)
        or type(metadata["feature_schema_digest"]) is not str
        or _SHA256.fullmatch(metadata["feature_schema_digest"]) is None
        or type(metadata["components"]) is not dict
    ):
        raise FrozenArtifactError("adaptive-controller metadata types differ")
    try:
        components = metadata["components"]
        expected_components = {"risk_probe", "vector_bank", "vector_router"}
        if metadata["candidate_layers"] is not None:
            expected_components.add("layer_router")
        if set(components) != expected_components:
            raise FrozenArtifactError("adaptive-controller component set is invalid")
        for name, expected_digest in components.items():
            if type(name) is not str or type(expected_digest) is not str or not _SHA256.fullmatch(
                expected_digest
            ):
                raise FrozenArtifactError("adaptive-controller component digest is invalid")
            if sha256_path(source / name) != expected_digest:
                raise FrozenArtifactError(f"adaptive-controller component changed: {name}")
        risk_probe = load_calibrated_probe(source / "risk_probe")
        vector_bank = load_routed_vector_bank(source / "vector_bank")
        vector_router = load_adaptive_router(source / "vector_router")
        alpha_data = metadata["alpha_controller"]
        alpha_controller = AlphaController(
            mode=AlphaMode(alpha_data["mode"]),
            alpha_max=alpha_data["alpha_max"],
            beta=alpha_data["beta"],
            threshold=alpha_data["threshold"],
        )
        layer_selector: LayerSelector | None = None
        candidate_layers = metadata["candidate_layers"]
        if candidate_layers is not None:
            layer_selector = LayerSelector(
                tuple(candidate_layers),
                load_adaptive_router(source / "layer_router"),
            )
        controller = AdaptiveController(
            risk_probe=risk_probe,
            vector_bank=vector_bank,
            vector_router=vector_router,
            alpha_controller=alpha_controller,
            fixed_layer=metadata["fixed_layer"],
            layer_selector=layer_selector,
        )
        if controller.risk_probe.training_schema.digest != metadata["feature_schema_digest"]:
            raise FrozenArtifactError("adaptive-controller feature schema digest mismatch")
        return controller
    except (KeyError, TypeError, ValueError, DataValidationError) as exc:
        if isinstance(exc, FrozenArtifactError):
            raise
        raise FrozenArtifactError(f"invalid adaptive-controller artifact: {exc}") from exc

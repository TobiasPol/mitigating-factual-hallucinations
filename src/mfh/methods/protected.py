"""Protected-subspace and covariance-aware directions for M5."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from mfh.artifact_namespace import validate_active_study_artifact_paths
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.methods.features import ActivationFeatureSchema
from mfh.provenance import sha256_file, stable_hash

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40,64}$")


def _vector(value: Tensor, *, width: int | None = None) -> Tensor:
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if result.ndim != 1 or result.numel() == 0:
        raise DataValidationError("behavior direction must be a non-empty vector")
    if width is not None and result.numel() != width:
        raise DataValidationError(f"expected behavior width {width}, got {result.numel()}")
    if not torch.isfinite(result).all():
        raise DataValidationError("behavior direction contains NaN or infinity")
    return result


@dataclass(frozen=True, slots=True)
class BehaviorDirection:
    behavior: str
    direction: Tensor
    positive_count: int
    negative_count: int

    def __post_init__(self) -> None:
        behavior = self.behavior.strip()
        if not behavior:
            raise DataValidationError("protected behavior name must be non-empty")
        direction = _vector(self.direction)
        norm = torch.linalg.vector_norm(direction)
        if float(norm) <= 0:
            raise DataValidationError("protected behavior direction cannot be zero")
        if self.positive_count <= 0 or self.negative_count <= 0:
            raise DataValidationError("protected behavior direction requires both classes")
        object.__setattr__(self, "behavior", behavior)
        object.__setattr__(self, "direction", (direction / norm).float())


def build_behavior_direction(
    behavior: str, positive: Tensor, negative: Tensor
) -> BehaviorDirection:
    positive_values = positive.detach().cpu().double()
    negative_values = negative.detach().cpu().double()
    if (
        positive_values.ndim != 2
        or negative_values.ndim != 2
        or positive_values.shape[1:] != negative_values.shape[1:]
        or positive_values.shape[0] == 0
        or negative_values.shape[0] == 0
    ):
        raise DataValidationError("protected behavior activation groups have invalid shapes")
    if not torch.isfinite(positive_values).all() or not torch.isfinite(negative_values).all():
        raise DataValidationError("protected behavior activations contain NaN or infinity")
    return BehaviorDirection(
        behavior,
        positive_values.mean(0) - negative_values.mean(0),
        positive_values.shape[0],
        negative_values.shape[0],
    )


@dataclass(frozen=True, slots=True)
class ProtectedSubspace:
    basis: Tensor
    behaviors: tuple[str, ...]
    data_fingerprint: str
    feature_schema: ActivationFeatureSchema
    tolerance: float = 1e-8
    schema_version: int = 1

    def __post_init__(self) -> None:
        basis = self.basis.detach().to(device="cpu", dtype=torch.float64).contiguous().clone()
        if self.schema_version != 1:
            raise DataValidationError("unsupported protected-subspace schema version")
        if basis.ndim != 2 or basis.shape[0] == 0 or basis.shape[1] == 0:
            raise DataValidationError("protected basis must have shape [width, rank]")
        if basis.shape[1] > basis.shape[0] or not torch.isfinite(basis).all():
            raise DataValidationError("protected basis rank or values are invalid")
        gram = basis.T @ basis
        if not torch.allclose(gram, torch.eye(basis.shape[1], dtype=basis.dtype), atol=1e-6):
            raise DataValidationError("protected basis must be orthonormal")
        behaviors = tuple(value.strip() for value in self.behaviors)
        if any(not value for value in behaviors) or len(set(behaviors)) != len(behaviors):
            raise DataValidationError("protected behavior names must be non-empty and unique")
        if len(behaviors) < basis.shape[1]:
            raise DataValidationError("protected basis rank exceeds source behavior count")
        if not _SHA256.fullmatch(self.data_fingerprint):
            raise DataValidationError("protected subspace requires a data SHA-256 fingerprint")
        if self.feature_schema.width != basis.shape[0]:
            raise DataValidationError("protected feature schema width differs from the basis")
        if not math.isfinite(self.tolerance) or self.tolerance <= 0:
            raise DataValidationError("protected-subspace tolerance must be positive")
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "behaviors", behaviors)

    @property
    def width(self) -> int:
        return int(self.basis.shape[0])

    @property
    def rank(self) -> int:
        return int(self.basis.shape[1])

    def project(self, factuality_direction: Tensor, *, normalize: bool = False) -> Tensor:
        direction = _vector(factuality_direction, width=self.width)
        protected = direction - self.basis @ (self.basis.T @ direction)
        norm = torch.linalg.vector_norm(protected)
        if not torch.isfinite(norm) or float(norm) <= self.tolerance:
            raise DataValidationError(
                "factuality direction lies entirely in the protected subspace"
            )
        if normalize:
            protected = protected / norm
        return protected.float()

    def protected_energy(self, direction: Tensor) -> float:
        value = _vector(direction, width=self.width)
        projection = self.basis.T @ value
        denominator = torch.linalg.vector_norm(value).pow(2)
        if float(denominator) <= 0:
            raise DataValidationError("cannot measure protected energy of a zero vector")
        return float(projection.pow(2).sum() / denominator)


def build_protected_subspace(
    directions: Mapping[str, Tensor] | tuple[BehaviorDirection, ...],
    *,
    data_fingerprint: str,
    feature_schema: ActivationFeatureSchema,
    tolerance: float = 1e-8,
) -> ProtectedSubspace:
    if isinstance(directions, Mapping):
        items = tuple((name, value) for name, value in directions.items())
    else:
        items = tuple((value.behavior, value.direction) for value in directions)
    if not items:
        raise DataValidationError("protected subspace requires at least one behavior direction")
    width = int(items[0][1].numel())
    matrix = torch.stack([_vector(value, width=width) for _, value in items], dim=1)
    left, singular_values, _ = torch.linalg.svd(matrix, full_matrices=False)
    threshold = tolerance * max(matrix.shape) * float(singular_values.max())
    rank = int((singular_values > threshold).sum())
    if rank == 0:
        raise DataValidationError("protected behavior directions have zero numerical rank")
    return ProtectedSubspace(
        basis=left[:, :rank],
        behaviors=tuple(name for name, _ in items),
        data_fingerprint=data_fingerprint,
        feature_schema=feature_schema,
        tolerance=tolerance,
    )


def behavior_covariance(behavior_changes: Tensor, *, center: bool = True) -> Tensor:
    values = behavior_changes.detach().to(device="cpu", dtype=torch.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise DataValidationError("protected behavior changes must have shape [rows, width]")
    if not torch.isfinite(values).all():
        raise DataValidationError("protected behavior changes contain NaN or infinity")
    if center:
        values = values - values.mean(dim=0)
    return values.T @ values / values.shape[0]


def subspace_covariance(subspace: ProtectedSubspace, *, weight: float = 1.0) -> Tensor:
    if not math.isfinite(weight) or weight <= 0:
        raise DataValidationError("subspace covariance weight must be positive")
    return weight * (subspace.basis @ subspace.basis.T)


def covariance_aware_direction(
    truth_direction: Tensor,
    protected_covariance: Tensor,
    *,
    lambda_penalty: float,
    ridge: float = 1e-4,
) -> Tensor:
    """Solve the regularized quadratic M5 objective and return a unit vector.

    For ``d^T v - lambda * v^T Sigma v - ridge/2 * ||v||^2``, the
    stationary point solves ``(2 lambda Sigma + ridge I) v = d``.
    """

    truth = _vector(truth_direction)
    covariance = protected_covariance.detach().to(device="cpu", dtype=torch.float64)
    if covariance.shape != (truth.numel(), truth.numel()):
        raise DataValidationError("protected covariance has the wrong shape")
    if not torch.isfinite(covariance).all() or not torch.allclose(
        covariance, covariance.T, atol=1e-8
    ):
        raise DataValidationError("protected covariance must be finite and symmetric")
    eigenvalues = torch.linalg.eigvalsh(covariance)
    if float(eigenvalues.min()) < -1e-8:
        raise DataValidationError("protected covariance must be positive semidefinite")
    if not math.isfinite(lambda_penalty) or lambda_penalty <= 0:
        raise DataValidationError("covariance penalty must be finite and positive")
    if not math.isfinite(ridge) or ridge <= 0:
        raise DataValidationError("covariance ridge must be finite and positive")
    system = 2 * lambda_penalty * covariance + ridge * torch.eye(truth.numel())
    solution = torch.linalg.solve(system, truth)
    norm = torch.linalg.vector_norm(solution)
    if not torch.isfinite(norm) or float(norm) <= 0:
        raise DataValidationError("covariance-aware direction is zero or invalid")
    return cast(Tensor, (solution / norm).float())


def alpha_for_matched_activation_projection(
    candidate_direction: Tensor,
    truth_direction: Tensor,
    *,
    target_gain: float,
) -> float:
    """Exploratory activation-space diagnostic, not confirmatory risk matching."""

    candidate = _vector(candidate_direction)
    truth = _vector(truth_direction, width=candidate.numel())
    if not math.isfinite(target_gain) or target_gain <= 0:
        raise DataValidationError("target factuality gain must be finite and positive")
    gain_per_alpha = float(torch.dot(candidate, truth))
    if gain_per_alpha <= 0:
        raise DataValidationError("candidate direction cannot produce positive factuality gain")
    return target_gain / gain_per_alpha


@dataclass(frozen=True, slots=True)
class EmpiricalEvaluationIdentity:
    """Shared frozen evaluation context for matched M5 operating points."""

    benchmark: str
    model_repository: str
    model_revision: str
    prompt_id: str
    prompt_sha256: str
    question_set_fingerprint: str
    generation_bundle_fingerprint: str

    def __post_init__(self) -> None:
        for name in ("benchmark", "model_repository", "prompt_id"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise DataValidationError(f"empirical evaluation {name} must be non-empty")
            object.__setattr__(self, name, value)
        if not _REVISION.fullmatch(self.model_revision):
            raise DataValidationError("empirical evaluation requires an immutable model revision")
        if any(
            not _SHA256.fullmatch(value)
            for value in (
                self.prompt_sha256,
                self.question_set_fingerprint,
                self.generation_bundle_fingerprint,
            )
        ):
            raise DataValidationError("empirical evaluation fingerprints must be SHA-256")

    @property
    def digest(self) -> str:
        return stable_hash(
            {
                "benchmark": self.benchmark,
                "model_repository": self.model_repository,
                "model_revision": self.model_revision,
                "prompt_id": self.prompt_id,
                "prompt_sha256": self.prompt_sha256,
                "question_set_fingerprint": self.question_set_fingerprint,
                "generation_bundle_fingerprint": self.generation_bundle_fingerprint,
            }
        )


@dataclass(frozen=True, slots=True)
class EmpiricalOperatingPoint:
    method: str
    alpha: float
    hallucination_risk: float
    coverage: float
    utility_metrics: Mapping[str, float]
    evaluation: EmpiricalEvaluationIdentity

    def __post_init__(self) -> None:
        method = self.method.strip()
        if not method or not math.isfinite(self.alpha) or self.alpha < 0:
            raise DataValidationError("empirical operating-point identity or alpha is invalid")
        if not 0 <= self.hallucination_risk <= 1 or not 0 <= self.coverage <= 1:
            raise DataValidationError("empirical risk and coverage must be in [0, 1]")
        utility = {str(key).strip(): float(value) for key, value in self.utility_metrics.items()}
        if any(not key or not math.isfinite(value) for key, value in utility.items()):
            raise DataValidationError("empirical utility metrics must be named and finite")
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "utility_metrics", MappingProxyType(utility))


@dataclass(frozen=True, slots=True)
class E8OperatingPointRegistry:
    """One frozen risk-or-coverage target for every promoted E8 method and prompt."""

    matching_dimension: str
    target: float
    tolerance: float
    condition_ids_by_prompt: Mapping[str, Mapping[str, str]]
    candidate_screen_sha256: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        conditions = {
            str(prompt): MappingProxyType(
                {str(method): str(condition_id) for method, condition_id in values.items()}
            )
            for prompt, values in self.condition_ids_by_prompt.items()
        }
        if (
            self.schema_version != 1
            or self.matching_dimension not in {"hallucination_risk", "coverage"}
            or isinstance(self.target, bool)
            or not isinstance(self.target, int | float)
            or not math.isfinite(float(self.target))
            or not 0 <= float(self.target) <= 1
            or isinstance(self.tolerance, bool)
            or not isinstance(self.tolerance, int | float)
            or not math.isfinite(float(self.tolerance))
            or not 0 <= float(self.tolerance) <= 0.02
            or set(conditions) != {"P0-neutral", "P2-calibrated-abstention"}
            or any(set(values) != {"M1", "M3", "M4", "M5"} for values in conditions.values())
            or any(
                not _SHA256.fullmatch(condition_id)
                for values in conditions.values()
                for condition_id in values.values()
            )
            or len(
                {
                    condition_id
                    for values in conditions.values()
                    for condition_id in values.values()
                }
            )
            != 8
            or not _SHA256.fullmatch(self.candidate_screen_sha256)
        ):
            raise DataValidationError("E8 operating-point registry is invalid")
        object.__setattr__(self, "target", float(self.target))
        object.__setattr__(self, "tolerance", float(self.tolerance))
        object.__setattr__(
            self, "condition_ids_by_prompt", MappingProxyType(conditions)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "matching_dimension": self.matching_dimension,
            "target": self.target,
            "tolerance": self.tolerance,
            "condition_ids_by_prompt": {
                prompt: dict(values)
                for prompt, values in self.condition_ids_by_prompt.items()
            },
            "candidate_screen_sha256": self.candidate_screen_sha256,
        }


def save_e8_operating_point_registry(
    path: str | Path, registry: E8OperatingPointRegistry
) -> str:
    destination = validate_active_study_artifact_paths(
        {"E8 operating-point registry": path}
    )["E8 operating-point registry"]
    if destination.exists() or destination.is_symlink():
        raise FrozenArtifactError(
            f"refusing to overwrite E8 operating-point registry: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    body = registry.to_dict()
    destination.write_text(
        json.dumps({**body, "registry_digest": stable_hash(body)}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return sha256_file(destination)


def load_e8_operating_point_registry(path: str | Path) -> E8OperatingPointRegistry:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise FrozenArtifactError("E8 operating-point registry must be one regular file")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise FrozenArtifactError("E8 operating-point registry must be an object")
        body = dict(value)
        digest = body.pop("registry_digest", None)
        if digest != stable_hash(body) or set(body) != {
            "schema_version",
            "matching_dimension",
            "target",
            "tolerance",
            "condition_ids_by_prompt",
            "candidate_screen_sha256",
        }:
            raise FrozenArtifactError("E8 operating-point registry digest or schema differs")
        raw_conditions = body["condition_ids_by_prompt"]
        if not isinstance(raw_conditions, dict) or any(
            not isinstance(item, dict) for item in raw_conditions.values()
        ):
            raise FrozenArtifactError("E8 operating-point condition registry differs")
        return E8OperatingPointRegistry(
            matching_dimension=str(body["matching_dimension"]),
            target=float(body["target"]),
            tolerance=float(body["tolerance"]),
            condition_ids_by_prompt=raw_conditions,
            candidate_screen_sha256=str(body["candidate_screen_sha256"]),
            schema_version=int(body["schema_version"]),
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, FrozenArtifactError):
            raise
        raise FrozenArtifactError(f"cannot load E8 operating-point registry: {exc}") from exc


def match_empirical_operating_points(
    candidates: Mapping[str, Sequence[EmpiricalOperatingPoint]],
    *,
    target_hallucination_risk: float | None = None,
    target_coverage: float | None = None,
    tolerance: float = 0.005,
) -> dict[str, EmpiricalOperatingPoint]:
    """Select M1/M3/M4/M5 points at measured equal risk or equal coverage."""

    if (target_hallucination_risk is None) == (target_coverage is None):
        raise DataValidationError("match exactly one empirical risk or coverage target")
    target = target_hallucination_risk if target_hallucination_risk is not None else target_coverage
    assert target is not None
    if not 0 <= target <= 1 or not math.isfinite(tolerance) or tolerance < 0:
        raise DataValidationError("empirical match target or tolerance is invalid")
    if not candidates:
        raise DataValidationError("empirical matching requires method candidates")
    evaluation_identities = {
        point.evaluation.digest for points in candidates.values() for point in points
    }
    if len(evaluation_identities) != 1:
        raise DataValidationError(
            "empirical operating points must share one model, prompt, and question set"
        )
    result: dict[str, EmpiricalOperatingPoint] = {}
    for method, points in candidates.items():
        if not points or any(point.method != method for point in points):
            raise DataValidationError("empirical candidate methods do not match their keys")
        if len({point.alpha for point in points}) != len(points):
            raise DataValidationError("empirical candidates contain duplicate method/alpha points")
        if target_hallucination_risk is not None:
            eligible = [
                point
                for point in points
                if abs(point.hallucination_risk - target_hallucination_risk) <= tolerance
            ]
            if eligible:
                result[method] = min(eligible, key=lambda point: (-point.coverage, point.alpha))
        else:
            assert target_coverage is not None
            eligible = [
                point for point in points if abs(point.coverage - target_coverage) <= tolerance
            ]
            if eligible:
                result[method] = min(
                    eligible, key=lambda point: (point.hallucination_risk, point.alpha)
                )
        if method not in result:
            raise DataValidationError(
                f"method {method!r} has no empirical point within tolerance {tolerance}"
            )
    return result


def save_protected_subspace(directory: str | Path, subspace: ProtectedSubspace) -> None:
    destination = validate_active_study_artifact_paths(
        {"E8 protected subspace": directory}
    )["E8 protected subspace"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FrozenArtifactError(f"refusing to overwrite protected subspace: {destination}")
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        tensor_path = stage / "subspace.safetensors"
        save_file({"basis": subspace.basis.float().contiguous()}, tensor_path)
        metadata_body = {
            "schema_version": subspace.schema_version,
            "behaviors": list(subspace.behaviors),
            "data_fingerprint": subspace.data_fingerprint,
            "feature_schema": subspace.feature_schema.to_dict(),
            "tolerance": subspace.tolerance,
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


def load_protected_subspace(
    directory: str | Path, *, expected_data_fingerprint: str | None = None
) -> ProtectedSubspace:
    source = Path(directory)
    try:
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read protected-subspace metadata: {exc}") from exc
    digest = metadata.pop("metadata_digest", None)
    if digest != stable_hash(metadata):
        raise FrozenArtifactError("protected-subspace metadata digest mismatch")
    if metadata.get("schema_version") != 1:
        raise FrozenArtifactError("unsupported protected-subspace schema version")
    tensor_path = source / "subspace.safetensors"
    if sha256_file(tensor_path) != metadata.get("tensor_sha256"):
        raise FrozenArtifactError("protected-subspace tensor checksum mismatch")
    if (
        expected_data_fingerprint is not None
        and metadata.get("data_fingerprint") != expected_data_fingerprint
    ):
        raise FrozenArtifactError("protected subspace has a different data fingerprint")
    try:
        tensors = load_file(tensor_path, device="cpu")
        if set(tensors) != {"basis"}:
            raise FrozenArtifactError("unexpected protected-subspace tensors")
        return ProtectedSubspace(
            basis=tensors["basis"],
            behaviors=tuple(str(value) for value in metadata["behaviors"]),
            data_fingerprint=str(metadata["data_fingerprint"]),
            feature_schema=ActivationFeatureSchema.from_dict(metadata["feature_schema"]),
            tolerance=float(metadata["tolerance"]),
        )
    except (KeyError, TypeError, ValueError, DataValidationError) as exc:
        raise FrozenArtifactError(f"invalid protected-subspace artifact: {exc}") from exc

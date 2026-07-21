"""M1 centroid and M2 CAA vector construction with online statistics."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from mfh.artifact_namespace import validate_active_study_artifact_paths
from mfh.contracts import ActivationSite, Outcome
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.inference.architecture import HookKey
from mfh.provenance import sha256_file, stable_hash

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class OnlineMoments:
    """Mergeable Welford accumulator over the last activation dimension."""

    def __init__(self) -> None:
        self.count = 0
        self.mean: Tensor | None = None
        self.m2: Tensor | None = None

    @property
    def width(self) -> int | None:
        return int(self.mean.numel()) if self.mean is not None else None

    def update(self, values: Tensor) -> None:
        if values.ndim < 1 or values.shape[-1] == 0:
            raise DataValidationError("activation batch must have a non-empty final dimension")
        flattened = (
            values.detach().to(device="cpu", dtype=torch.float64).reshape(-1, values.shape[-1])
        )
        if flattened.shape[0] == 0:
            return
        if not torch.isfinite(flattened).all():
            raise DataValidationError("activation batch contains NaN or infinity")
        if self.width is not None and self.width != flattened.shape[-1]:
            raise DataValidationError(
                f"activation width changed from {self.width} to {flattened.shape[-1]}"
            )
        batch_count = int(flattened.shape[0])
        batch_mean = flattened.mean(dim=0)
        centered = flattened - batch_mean
        batch_m2 = (centered * centered).sum(dim=0)
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        assert self.mean is not None and self.m2 is not None
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (batch_count / total)
        self.m2 = self.m2 + batch_m2 + delta.pow(2) * self.count * batch_count / total
        self.count = total

    def merge(self, other: OnlineMoments) -> None:
        if other.count == 0:
            return
        if other.mean is None or other.m2 is None:
            raise DataValidationError("non-empty moments accumulator has no state")
        if self.count == 0:
            self.count = other.count
            self.mean = other.mean.clone()
            self.m2 = other.m2.clone()
            return
        if self.width != other.width:
            raise DataValidationError("cannot merge activation statistics with different widths")
        assert self.mean is not None and self.m2 is not None
        total = self.count + other.count
        delta = other.mean - self.mean
        self.mean = self.mean + delta * (other.count / total)
        self.m2 = self.m2 + other.m2 + delta.pow(2) * self.count * other.count / total
        self.count = total

    def variance(self, *, unbiased: bool = True) -> Tensor:
        if self.count == 0 or self.m2 is None:
            raise DataValidationError("cannot compute variance of empty statistics")
        denominator = self.count - 1 if unbiased else self.count
        if denominator <= 0:
            raise DataValidationError("unbiased variance requires at least two observations")
        return self.m2 / denominator

    def state_dict(self) -> dict[str, Any]:
        return {"count": self.count, "mean": self.mean, "m2": self.m2}

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> OnlineMoments:
        result = cls()
        result.count = int(state["count"])
        mean, m2 = state.get("mean"), state.get("m2")
        if result.count:
            if not isinstance(mean, Tensor) or not isinstance(m2, Tensor):
                raise DataValidationError("invalid online-moments tensor state")
            result.mean = mean.to(device="cpu", dtype=torch.float64)
            result.m2 = m2.to(device="cpu", dtype=torch.float64)
            if result.mean.ndim != 1 or result.m2.shape != result.mean.shape:
                raise DataValidationError("invalid online-moments shapes")
        return result


@dataclass(frozen=True, slots=True)
class SteeringVector:
    key: HookKey
    direction: Tensor
    source_method: str
    positive_count: int
    negative_count: int

    def __post_init__(self) -> None:
        if self.direction.ndim != 1 or self.direction.numel() == 0:
            raise DataValidationError("steering direction must be a non-empty vector")
        if not torch.isfinite(self.direction).all():
            raise DataValidationError("steering direction contains NaN or infinity")
        norm = float(torch.linalg.vector_norm(self.direction.float()))
        if not math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-5):
            raise DataValidationError(f"steering direction must have unit L2 norm, got {norm}")
        if self.positive_count <= 0 or self.negative_count <= 0:
            raise DataValidationError("steering vectors require positive and negative observations")
        if not self.source_method.strip():
            raise DataValidationError("source_method must be non-empty")


@dataclass(frozen=True, slots=True)
class VectorBank:
    vectors: dict[HookKey, SteeringVector]
    data_fingerprint: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise DataValidationError("unsupported vector-bank schema version")
        if not self.vectors:
            raise DataValidationError("vector bank cannot be empty")
        if set(self.vectors) != {vector.key for vector in self.vectors.values()}:
            raise DataValidationError("vector-bank keys do not match vector metadata")
        if not _SHA256.fullmatch(self.data_fingerprint):
            raise DataValidationError("vector bank requires a data SHA-256 fingerprint")

    def intervention_directions(self) -> dict[HookKey, Tensor]:
        return {key: vector.direction for key, vector in self.vectors.items()}


class CentroidVectorBuilder:
    """Build normalized correct-minus-incorrect directions for M1-R/M1-P."""

    def __init__(self) -> None:
        self.correct: dict[HookKey, OnlineMoments] = {}
        self.incorrect: dict[HookKey, OnlineMoments] = {}

    def update(self, outcome: Outcome, activations: dict[HookKey, Tensor]) -> None:
        if outcome not in {Outcome.CORRECT, Outcome.INCORRECT}:
            return
        destination = self.correct if outcome is Outcome.CORRECT else self.incorrect
        for key, values in activations.items():
            destination.setdefault(key, OnlineMoments()).update(values)

    def build(self, *, source_method: str, data_fingerprint: str) -> VectorBank:
        if set(self.correct) != set(self.incorrect):
            raise DataValidationError(
                "correct and incorrect activation sets contain different hook points"
            )
        vectors: dict[HookKey, SteeringVector] = {}
        for key in sorted(self.correct):
            positive, negative = self.correct[key], self.incorrect[key]
            if positive.mean is None or negative.mean is None:
                raise DataValidationError(f"missing centroid state for {key.artifact_key}")
            difference = positive.mean - negative.mean
            norm = torch.linalg.vector_norm(difference)
            if not torch.isfinite(norm) or float(norm) <= 0:
                raise DataValidationError(
                    f"centroid difference is zero or invalid at {key.artifact_key}"
                )
            direction = (difference / norm).to(torch.float32)
            vectors[key] = SteeringVector(
                key=key,
                direction=direction,
                source_method=source_method,
                positive_count=positive.count,
                negative_count=negative.count,
            )
        return VectorBank(vectors=vectors, data_fingerprint=data_fingerprint)


class PairedDifferenceBuilder:
    """Build CAA directions by averaging matched positive-negative differences."""

    def __init__(self) -> None:
        self.differences: dict[HookKey, OnlineMoments] = {}
        self.pair_count: dict[HookKey, int] = {}

    def update(
        self,
        positive: dict[HookKey, Tensor],
        negative: dict[HookKey, Tensor],
    ) -> None:
        if positive.keys() != negative.keys():
            raise DataValidationError("CAA pair has mismatched hook points")
        for key in positive:
            if positive[key].shape != negative[key].shape:
                raise DataValidationError(
                    f"CAA pair shape mismatch at {key.artifact_key}: "
                    f"{tuple(positive[key].shape)} vs {tuple(negative[key].shape)}"
                )
            difference = positive[key] - negative[key]
            self.differences.setdefault(key, OnlineMoments()).update(difference)
            self.pair_count[key] = self.pair_count.get(key, 0) + int(
                difference.reshape(-1, difference.shape[-1]).shape[0]
            )

    def build(self, *, data_fingerprint: str) -> VectorBank:
        vectors: dict[HookKey, SteeringVector] = {}
        for key, moments in self.differences.items():
            if moments.mean is None:
                raise DataValidationError(f"empty CAA statistics at {key.artifact_key}")
            norm = torch.linalg.vector_norm(moments.mean)
            if not torch.isfinite(norm) or float(norm) <= 0:
                raise DataValidationError(f"CAA mean difference is zero at {key.artifact_key}")
            count = self.pair_count[key]
            vectors[key] = SteeringVector(
                key=key,
                direction=(moments.mean / norm).to(torch.float32),
                source_method="M2-CAA",
                positive_count=count,
                negative_count=count,
            )
        return VectorBank(vectors=vectors, data_fingerprint=data_fingerprint)


def save_vector_bank(directory: str | Path, bank: VectorBank) -> None:
    """Publish safetensors and metadata together as an immutable directory."""

    destination = validate_active_study_artifact_paths(
        {"static vector bank": directory}
    )["static vector bank"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FrozenArtifactError(f"refusing to overwrite vector bank: {destination}")
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        tensor_path = stage / "vectors.safetensors"
        tensors = {
            key.artifact_key: vector.direction.detach().cpu().contiguous()
            for key, vector in bank.vectors.items()
        }
        save_file(tensors, tensor_path)
        metadata_body = {
            "schema_version": bank.schema_version,
            "data_fingerprint": bank.data_fingerprint,
            "tensor_sha256": sha256_file(tensor_path),
            "vectors": [
                {
                    "tensor_key": key.artifact_key,
                    "layer": key.layer,
                    "site": key.site.value,
                    "source_method": vector.source_method,
                    "positive_count": vector.positive_count,
                    "negative_count": vector.negative_count,
                }
                for key, vector in sorted(bank.vectors.items())
            ],
        }
        metadata = {
            **metadata_body,
            "metadata_digest": stable_hash(metadata_body),
        }
        (stage / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def load_vector_bank(
    directory: str | Path, *, expected_data_fingerprint: str | None = None
) -> VectorBank:
    source = Path(directory)
    try:
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read vector-bank metadata: {exc}") from exc
    if metadata.get("schema_version") != 1:
        raise FrozenArtifactError("unsupported vector-bank schema version")
    metadata_digest = metadata.pop("metadata_digest", None)
    if metadata_digest != stable_hash(metadata):
        raise FrozenArtifactError("vector-bank metadata digest mismatch")
    tensor_path = source / "vectors.safetensors"
    if sha256_file(tensor_path) != metadata.get("tensor_sha256"):
        raise FrozenArtifactError("vector tensor checksum mismatch")
    fingerprint = metadata.get("data_fingerprint")
    if not isinstance(fingerprint, str) or not _SHA256.fullmatch(fingerprint):
        raise FrozenArtifactError("invalid vector-bank data fingerprint")
    if expected_data_fingerprint is not None and fingerprint != expected_data_fingerprint:
        raise FrozenArtifactError("vector bank was trained on a different data fingerprint")
    tensors = load_file(tensor_path, device="cpu")
    vectors: dict[HookKey, SteeringVector] = {}
    entries = metadata.get("vectors")
    if not isinstance(entries, list) or not entries:
        raise FrozenArtifactError("vector-bank metadata has no vectors")
    for entry in entries:
        if not isinstance(entry, dict):
            raise FrozenArtifactError("invalid vector metadata entry")
        try:
            key = HookKey(int(entry["layer"]), ActivationSite(entry["site"]))
            tensor_key = str(entry["tensor_key"])
            if tensor_key != key.artifact_key:
                raise FrozenArtifactError("vector tensor key does not match hook metadata")
            direction = tensors[tensor_key]
            vectors[key] = SteeringVector(
                key=key,
                direction=direction,
                source_method=str(entry["source_method"]),
                positive_count=int(entry["positive_count"]),
                negative_count=int(entry["negative_count"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise FrozenArtifactError(f"invalid vector metadata: {exc}") from exc
    if set(tensors) != {key.artifact_key for key in vectors}:
        raise FrozenArtifactError("unexpected or missing tensors in vector bank")
    return VectorBank(vectors=vectors, data_fingerprint=fingerprint)

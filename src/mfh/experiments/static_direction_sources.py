"""Strict direction resolution for the native E3 M1 and E4 M2 artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from mfh.contracts import ActivationSite
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.experiments.e4_caa_mlx import verify_m2_caa_artifact
from mfh.inference.architecture import HookKey
from mfh.methods.static import load_vector_bank
from mfh.provenance import sha256_file, stable_hash


@dataclass(frozen=True, slots=True)
class ResolvedStaticDirection:
    direction: Tensor
    direction_sha256: str
    direction_norm: float
    reference_rms: float
    source_kind: str

    def __post_init__(self) -> None:
        direction = self.direction.detach().cpu().float().contiguous().clone()
        norm = float(torch.linalg.vector_norm(direction))
        digest = hashlib.sha256(
            np.ascontiguousarray(direction.numpy(), dtype=np.float32).tobytes()
        ).hexdigest()
        if (
            direction.ndim != 1
            or direction.numel() == 0
            or not torch.isfinite(direction).all()
            or not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-6)
            or self.direction_sha256 != digest
            or not math.isclose(self.direction_norm, norm, rel_tol=0, abs_tol=1e-7)
            or not math.isfinite(self.reference_rms)
            or self.reference_rms <= 0
            or self.source_kind not in {
                "E3-M1-R-P0-native-MLX",
                "E4-M2-CAA-native-MLX",
            }
        ):
            raise DataValidationError("resolved static direction is invalid")
        object.__setattr__(self, "direction", direction)


def _direction(
    value: np.ndarray, *, reference_rms: float, source_kind: str
) -> ResolvedStaticDirection:
    values = np.ascontiguousarray(value, dtype=np.float32)
    tensor = torch.from_numpy(values.copy()).contiguous()
    return ResolvedStaticDirection(
        direction=tensor,
        direction_sha256=hashlib.sha256(values.tobytes(order="C")).hexdigest(),
        direction_norm=float(torch.linalg.vector_norm(tensor)),
        reference_rms=float(reference_rms),
        source_kind=source_kind,
    )


def _resolve_e3_m1(
    source: Path, *, layer: int, site: ActivationSite
) -> ResolvedStaticDirection:
    if site is not ActivationSite.POST_MLP:
        raise DataValidationError("E4 M1 must use the final-MLP post_mlp centroid baseline")
    if (
        source.is_symlink()
        or not source.is_dir()
        or {item.name for item in source.iterdir()} != {"metadata.json", "vectors.npz"}
        or any(item.is_symlink() or not item.is_file() for item in source.iterdir())
    ):
        raise FrozenArtifactError("E3 M1 vector bundle inventory differs")
    try:
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read E3 M1 metadata: {exc}") from exc
    if type(metadata) is not dict:
        raise FrozenArtifactError("E3 M1 metadata must be an object")
    body = dict(metadata)
    digest = body.pop("metadata_digest", None)
    if (
        digest != stable_hash(body)
        or body.get("schema_version") != 1
        or body.get("phase") != "E3-construction"
        or body.get("scientific_eligible") is not True
        or body.get("vectors_sha256") != sha256_file(source / "vectors.npz")
        or body.get("prompt_axis") != ["P0-neutral", "P2-calibrated-abstention"]
        or body.get("extraction_axis") != ["M1-R", "M1-P"]
        or not isinstance(body.get("site_axis"), list)
        or not isinstance(body.get("layer_axis"), list)
    ):
        raise FrozenArtifactError("E3 M1 metadata identity differs")
    try:
        prompt_index = body["prompt_axis"].index("P0-neutral")
        extraction_index = body["extraction_axis"].index("M1-R")
        site_index = body["site_axis"].index(site.value)
        layer_index = body["layer_axis"].index(layer)
        with np.load(source / "vectors.npz", allow_pickle=False) as arrays:
            if set(arrays.files) != {
                "directions",
                "reference_rms",
                "correct_counts",
                "incorrect_counts",
            }:
                raise DataValidationError("E3 M1 arrays differ")
            directions = arrays["directions"]
            reference_rms = arrays["reference_rms"]
            correct = arrays["correct_counts"]
            incorrect = arrays["incorrect_counts"]
    except (OSError, ValueError, AttributeError, DataValidationError) as exc:
        raise FrozenArtifactError(f"cannot resolve E3 M1 direction: {exc}") from exc
    index = (prompt_index, extraction_index, site_index, layer_index)
    if (
        directions.dtype != np.float32
        or reference_rms.dtype != np.float64
        or correct.dtype != np.int64
        or incorrect.dtype != np.int64
        or directions.ndim != 5
        or reference_rms.shape != directions.shape[:-1]
        or correct.shape != directions.shape[:-1]
        or incorrect.shape != directions.shape[:-1]
        or correct[index] <= 0
        or incorrect[index] <= 0
    ):
        raise FrozenArtifactError("E3 M1 vector geometry differs")
    return _direction(
        directions[index],
        reference_rms=float(reference_rms[index]),
        source_kind="E3-M1-R-P0-native-MLX",
    )


def _resolve_m2(
    source: Path, *, layer: int, site: ActivationSite
) -> ResolvedStaticDirection:
    if site is not ActivationSite.BLOCK_OUTPUT:
        raise DataValidationError("E4 M2 CAA must intervene at the residual block output")
    try:
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        expected_digest = manifest["manifest_digest"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise FrozenArtifactError(f"cannot read M2 CAA manifest: {exc}") from exc
    verify_m2_caa_artifact(source, expected_manifest_digest=expected_digest)
    bank = load_vector_bank(source)
    try:
        direction = bank.vectors[HookKey(layer, site)].direction
        with np.load(source / "accumulator.npz", allow_pickle=False) as arrays:
            counts = arrays["counts"]
            elements = arrays["rms_elements"]
            squares = arrays["rms_sum_squares"]
        plan = json.loads((source / "plan.json").read_text(encoding="utf-8"))
        layer_index = plan["protocol"]["layers"].index(layer)
        if counts[layer_index] <= 0 or elements[layer_index] <= 0:
            raise DataValidationError("M2 CAA selected layer lacks paired evidence")
        reference_rms = math.sqrt(float(squares[layer_index] / elements[layer_index]))
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
        DataValidationError,
    ) as exc:
        raise FrozenArtifactError(f"cannot resolve M2 CAA direction: {exc}") from exc
    return _direction(
        direction.detach().cpu().float().numpy(),
        reference_rms=reference_rms,
        source_kind="E4-M2-CAA-native-MLX",
    )


def resolve_static_direction(
    source: str | Path,
    *,
    method: str,
    layer: int,
    site: ActivationSite,
) -> ResolvedStaticDirection:
    """Resolve geometry from the construction artifact, never caller metadata."""

    path = Path(source)
    if method == "M1":
        return _resolve_e3_m1(path, layer=layer, site=site)
    if method == "M2":
        return _resolve_m2(path, layer=layer, site=site)
    raise DataValidationError("only M1 and M2 use native static direction artifacts")

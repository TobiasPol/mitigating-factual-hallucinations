"""Typed, recursively frozen execution components for E9/E10 fixed methods."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from mfh.contracts import ActivationSite, ModelSpec, PromptSpec, Runtime, TokenScope
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.experiments.model_selection import validate_active_study_artifact_paths
from mfh.experiments.static_direction_sources import resolve_static_direction
from mfh.methods.adaptive import AdaptiveController, load_adaptive_controller
from mfh.methods.sparse import (
    load_coordinate_sparse_artifact,
    load_sae_intervention,
)
from mfh.provenance import sha256_file, sha256_path, stable_hash

_METHODS = {"M1", "M2", "M4", "M5"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _direction_sha256(direction: Tensor) -> str:
    values = np.ascontiguousarray(
        direction.detach().cpu().float().reshape(-1).numpy(), dtype=np.float32
    )
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _copy_artifact(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.exists():
        raise DataValidationError("confirmatory component source is missing or linked")
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    elif source.is_dir() and not any(item.is_symlink() for item in source.rglob("*")):
        shutil.copytree(source, destination, symlinks=False)
    else:
        raise DataValidationError("confirmatory component source is not a strict artifact")


def _source_direction(
    source: Path,
    *,
    method: str,
    layer: int,
    site: ActivationSite,
    token_scope: TokenScope,
    alpha: float,
    sparsity: float | None,
    reference_rms: float,
) -> Tensor:
    if method in {"M1", "M2"}:
        resolved = resolve_static_direction(
            source,
            method=method,
            layer=layer,
            site=site,
        )
        if not math.isclose(
            reference_rms, resolved.reference_rms, rel_tol=0, abs_tol=1e-12
        ):
            raise DataValidationError(
                "confirmatory static RMS differs from its construction artifact"
            )
        return resolved.direction
    if method == "M4" and (source / "coordinate.safetensors").is_file():
        coordinate_artifact = load_coordinate_sparse_artifact(source)
        if (
            coordinate_artifact.layer != layer
            or coordinate_artifact.site is not site
            or coordinate_artifact.token_scope is not token_scope
            or coordinate_artifact.alpha != alpha
            or coordinate_artifact.reference_rms != reference_rms
            or sparsity != coordinate_artifact.sparse_direction.retained_fraction
        ):
            raise DataValidationError(
                "coordinate-sparse component geometry differs from promotion"
            )
        return coordinate_artifact.sparse_direction.direction
    if method == "M4":
        sae_artifact = load_sae_intervention(source)
        return sae_artifact.decoded_direction
    if method == "M5":
        from mfh.experiments.e8_protected import load_e8_protected_artifact

        protected_artifact = load_e8_protected_artifact(source)
        if (
            protected_artifact.layer != layer
            or protected_artifact.site is not site
            or protected_artifact.token_scope is not token_scope
            or protected_artifact.alpha != alpha
            or protected_artifact.reference_rms != reference_rms
            or sparsity is not None
        ):
            raise DataValidationError(
                "protected component geometry differs from promotion"
            )
        return protected_artifact.selected_direction
    raise DataValidationError("unsupported fixed confirmatory method")


@dataclass(frozen=True, slots=True)
class ConfirmatoryFixedComponent:
    directory: Path
    method: str
    source_artifact_sha256: str
    direction: Tensor
    direction_sha256: str
    direction_norm: float
    reference_rms: float
    layer: int
    site: ActivationSite
    token_scope: TokenScope
    standardized_alpha: float
    sparsity: float | None
    decay: float
    fingerprint: str

    def __post_init__(self) -> None:
        direction = self.direction.detach().cpu().float().contiguous().clone()
        norm = float(torch.linalg.vector_norm(direction))
        if (
            self.method not in _METHODS
            or direction.ndim != 1
            or direction.numel() == 0
            or not torch.isfinite(direction).all()
            or not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-6)
            or self.direction_sha256 != _direction_sha256(direction)
            or not math.isclose(self.direction_norm, norm, rel_tol=0, abs_tol=1e-7)
            or self.reference_rms <= 0
            or self.layer < 0
            or self.standardized_alpha == 0
            or not math.isfinite(self.standardized_alpha)
            or (self.sparsity is not None and not 0 < self.sparsity <= 1)
            or (self.method == "M4") is not (self.sparsity is not None)
            or self.decay < 0
            or (self.token_scope is TokenScope.EXPONENTIAL_DECAY)
            is not (self.decay > 0)
        ):
            raise DataValidationError("confirmatory fixed component is invalid")
        object.__setattr__(self, "direction", direction)


@dataclass(frozen=True, slots=True)
class ConfirmatoryAdaptiveComponent:
    """Prompt-indexed, recursively frozen M3 controllers for confirmatory use."""

    directory: Path
    model_name: str
    model_repository: str
    model_revision: str
    runtime: Runtime
    quantization: str
    model_num_layers: int
    prompt_hashes: Mapping[str, str]
    controller_source_prompt_ids: Mapping[str, str]
    controller_fingerprints: Mapping[str, str]
    controllers: Mapping[str, AdaptiveController]
    fingerprint: str

    def __post_init__(self) -> None:
        if (
            not self.model_name
            or not self.model_repository
            or not self.model_revision
            or not self.quantization
            or self.model_num_layers <= 0
            or not self.prompt_hashes
            or set(self.prompt_hashes) != set(self.controller_source_prompt_ids)
            or set(self.prompt_hashes) != set(self.controller_fingerprints)
            or set(self.prompt_hashes) != set(self.controllers)
        ):
            raise DataValidationError("confirmatory adaptive component is invalid")
        object.__setattr__(
            self,
            "prompt_hashes",
            MappingProxyType(dict(self.prompt_hashes)),
        )
        object.__setattr__(
            self,
            "controller_source_prompt_ids",
            MappingProxyType(dict(self.controller_source_prompt_ids)),
        )
        object.__setattr__(
            self,
            "controller_fingerprints",
            MappingProxyType(dict(self.controller_fingerprints)),
        )
        object.__setattr__(
            self,
            "controllers",
            MappingProxyType(dict(self.controllers)),
        )


def _validate_controller_identity(
    controller: AdaptiveController,
    *,
    model_name: str,
    model_repository: str,
    model_revision: str,
    runtime: Runtime,
    quantization: str,
    model_num_layers: int,
    prompt_id: str,
    prompt_sha256: str,
) -> None:
    schema = controller.risk_probe.training_schema
    if (
        schema.model_repository != model_repository
        or schema.model_revision != model_revision
        or schema.runtime is not runtime
        or schema.quantization != quantization
        or schema.prompt_id != prompt_id
        or schema.prompt_sha256 != prompt_sha256
        or any(layer >= model_num_layers for layer in schema.layers)
        or any(layer >= model_num_layers for layer in controller.vector_bank.feature_schema.layers)
    ):
        raise DataValidationError(
            f"confirmatory M3 controller differs from model/prompt {model_name}/{prompt_id}"
        )


def write_confirmatory_adaptive_component(
    directory: str | Path,
    *,
    model: ModelSpec,
    prompts: Mapping[str, PromptSpec],
    controllers: Mapping[str, str | Path],
    controller_source_prompts: Mapping[str, str] | None = None,
) -> ConfirmatoryAdaptiveComponent:
    """Freeze one exact M3 controller per prompt under a single method identity."""

    normalized = validate_active_study_artifact_paths(
        {
            "confirmatory adaptive component": directory,
            **{
                f"confirmatory controller {name}": path
                for name, path in controllers.items()
            },
        }
    )
    destination = normalized["confirmatory adaptive component"]
    controllers = {
        name: normalized[f"confirmatory controller {name}"]
        for name in controllers
    }
    if destination.exists() or destination.is_symlink():
        raise FrozenArtifactError(
            f"refusing to overwrite confirmatory adaptive component: {destination}"
        )
    if not prompts or set(prompts) != set(controllers):
        raise DataValidationError(
            "confirmatory adaptive controllers must cover the exact prompt set"
        )
    source_prompts = dict(controller_source_prompts or {name: name for name in prompts})
    if set(source_prompts) != set(prompts) or any(
        source_prompt_id not in prompts for source_prompt_id in source_prompts.values()
    ):
        raise DataValidationError(
            "confirmatory controller source prompts must map the exact applied prompt set"
        )
    validated: dict[str, tuple[Path, str, str, str, str]] = {}
    for prompt_id in sorted(prompts):
        prompt = prompts[prompt_id]
        if prompt.prompt_id != prompt_id:
            raise DataValidationError("confirmatory prompt key differs from its identity")
        prompt_sha256 = hashlib.sha256(prompt.text.encode("utf-8")).hexdigest()
        source_prompt_id = source_prompts[prompt_id]
        source_prompt = prompts[source_prompt_id]
        source_prompt_sha256 = hashlib.sha256(
            source_prompt.text.encode("utf-8")
        ).hexdigest()
        source = Path(controllers[prompt_id]).resolve()
        controller = load_adaptive_controller(source)
        _validate_controller_identity(
            controller,
            model_name=model.name,
            model_repository=model.repository,
            model_revision=model.revision,
            runtime=model.runtime,
            quantization=model.quantization,
            model_num_layers=model.num_layers,
            prompt_id=source_prompt_id,
            prompt_sha256=source_prompt_sha256,
        )
        validated[prompt_id] = (
            source,
            prompt_sha256,
            source_prompt_id,
            source_prompt_sha256,
            sha256_path(source),
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent)
    )
    try:
        controller_root = stage / "controllers"
        controller_root.mkdir()
        descriptors: list[dict[str, object]] = []
        for prompt_id, (
            source,
            prompt_sha256,
            source_prompt_id,
            source_prompt_sha256,
            fingerprint,
        ) in validated.items():
            relative = f"controllers/{stable_hash(prompt_id)[:16]}"
            _copy_artifact(source, stage / relative)
            loaded = load_adaptive_controller(stage / relative)
            descriptors.append(
                {
                    "prompt_id": prompt_id,
                    "prompt_sha256": prompt_sha256,
                    "controller_source_prompt_id": source_prompt_id,
                    "controller_source_prompt_sha256": source_prompt_sha256,
                    "controller_path": relative,
                    "controller_sha256": fingerprint,
                    "feature_schema_digest": loaded.risk_probe.training_schema.digest,
                }
            )
        body = {
            "schema_version": 2,
            "component_kind": "confirmatory-adaptive-intervention",
            "model_name": model.name,
            "model_repository": model.repository,
            "model_revision": model.revision,
            "runtime": model.runtime.value,
            "quantization": model.quantization,
            "model_num_layers": model.num_layers,
            "controllers": descriptors,
        }
        (stage / "manifest.json").write_text(
            json.dumps(
                {**body, "manifest_digest": stable_hash(body)},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        load_confirmatory_adaptive_component(stage)
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return load_confirmatory_adaptive_component(destination)


def load_confirmatory_adaptive_component(
    directory: str | Path,
) -> ConfirmatoryAdaptiveComponent:
    source = Path(directory)
    if (
        source.is_symlink()
        or not source.is_dir()
        or {item.name for item in source.iterdir()} != {"manifest.json", "controllers"}
        or any(item.is_symlink() for item in source.rglob("*"))
    ):
        raise FrozenArtifactError("confirmatory adaptive component inventory differs")
    try:
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(
            f"cannot read confirmatory adaptive component: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise FrozenArtifactError("confirmatory adaptive manifest is invalid")
    digest = manifest.pop("manifest_digest", None)
    expected_fields = {
        "schema_version",
        "component_kind",
        "model_name",
        "model_repository",
        "model_revision",
        "runtime",
        "quantization",
        "model_num_layers",
        "controllers",
    }
    if (
        set(manifest) != expected_fields
        or manifest.get("schema_version") != 2
        or manifest.get("component_kind")
        != "confirmatory-adaptive-intervention"
        or digest != stable_hash(manifest)
        or type(manifest.get("model_name")) is not str
        or type(manifest.get("model_repository")) is not str
        or type(manifest.get("model_revision")) is not str
        or type(manifest.get("runtime")) is not str
        or type(manifest.get("quantization")) is not str
        or type(manifest.get("model_num_layers")) is not int
        or not isinstance(manifest.get("controllers"), list)
    ):
        raise FrozenArtifactError("confirmatory adaptive component identity differs")
    try:
        runtime = Runtime(manifest["runtime"])
        model_num_layers = manifest["model_num_layers"]
        prompt_hashes: dict[str, str] = {}
        controller_source_prompt_ids: dict[str, str] = {}
        controller_source_prompt_hashes: dict[str, str] = {}
        controller_fingerprints: dict[str, str] = {}
        loaded_controllers: dict[str, AdaptiveController] = {}
        expected_directories: set[str] = set()
        for descriptor in manifest["controllers"]:
            if (
                not isinstance(descriptor, Mapping)
                or set(descriptor)
                != {
                    "prompt_id",
                    "prompt_sha256",
                    "controller_source_prompt_id",
                    "controller_source_prompt_sha256",
                    "controller_path",
                    "controller_sha256",
                    "feature_schema_digest",
                }
                or any(
                    type(descriptor[name]) is not str
                    for name in descriptor
                )
            ):
                raise FrozenArtifactError("confirmatory controller descriptor is invalid")
            prompt_id = descriptor["prompt_id"]
            source_prompt_id = descriptor["controller_source_prompt_id"]
            relative = descriptor["controller_path"]
            expected_relative = f"controllers/{stable_hash(prompt_id)[:16]}"
            controller_path = source / relative
            if (
                prompt_id in prompt_hashes
                or relative != expected_relative
                or not all(
                    _value and _SHA256.fullmatch(_value)
                    for _value in (
                        descriptor["prompt_sha256"],
                        descriptor["controller_source_prompt_sha256"],
                        descriptor["controller_sha256"],
                        descriptor["feature_schema_digest"],
                    )
                )
                or sha256_path(controller_path) != descriptor["controller_sha256"]
            ):
                raise FrozenArtifactError("confirmatory controller identity differs")
            controller = load_adaptive_controller(controller_path)
            if (
                controller.risk_probe.training_schema.digest
                != descriptor["feature_schema_digest"]
            ):
                raise FrozenArtifactError("confirmatory controller schema changed")
            _validate_controller_identity(
                controller,
                model_name=manifest["model_name"],
                model_repository=manifest["model_repository"],
                model_revision=manifest["model_revision"],
                runtime=runtime,
                quantization=manifest["quantization"],
                model_num_layers=model_num_layers,
                prompt_id=source_prompt_id,
                prompt_sha256=descriptor["controller_source_prompt_sha256"],
            )
            prompt_hashes[prompt_id] = descriptor["prompt_sha256"]
            controller_source_prompt_ids[prompt_id] = source_prompt_id
            controller_source_prompt_hashes[prompt_id] = descriptor[
                "controller_source_prompt_sha256"
            ]
            controller_fingerprints[prompt_id] = descriptor["controller_sha256"]
            loaded_controllers[prompt_id] = controller
            expected_directories.add(expected_relative.rsplit("/", maxsplit=1)[1])
        if (
            {item.name for item in (source / "controllers").iterdir()}
            != expected_directories
            or not set(controller_source_prompt_ids.values()) <= set(prompt_hashes)
            or any(
                controller_source_prompt_hashes[prompt_id]
                != prompt_hashes[source_prompt_id]
                for prompt_id, source_prompt_id in controller_source_prompt_ids.items()
            )
        ):
            raise FrozenArtifactError("confirmatory controller directories differ")
        return ConfirmatoryAdaptiveComponent(
            directory=source,
            model_name=manifest["model_name"],
            model_repository=manifest["model_repository"],
            model_revision=manifest["model_revision"],
            runtime=runtime,
            quantization=manifest["quantization"],
            model_num_layers=model_num_layers,
            prompt_hashes=prompt_hashes,
            controller_source_prompt_ids=controller_source_prompt_ids,
            controller_fingerprints=controller_fingerprints,
            controllers=loaded_controllers,
            fingerprint=sha256_path(source),
        )
    except (KeyError, TypeError, ValueError, DataValidationError) as exc:
        if isinstance(exc, FrozenArtifactError):
            raise
        raise FrozenArtifactError(
            f"invalid confirmatory adaptive component: {exc}"
        ) from exc


def write_confirmatory_fixed_component(
    directory: str | Path,
    *,
    source_artifact: str | Path,
    method: str,
    layer: int,
    site: ActivationSite,
    token_scope: TokenScope,
    standardized_alpha: float,
    reference_rms: float,
    sparsity: float | None = None,
    decay: float = 0.0,
) -> ConfirmatoryFixedComponent:
    """Derive and freeze the exact direction executed by a fixed confirmatory method."""

    normalized = validate_active_study_artifact_paths(
        {
            "confirmatory fixed component": directory,
            "confirmatory source artifact": source_artifact,
        }
    )
    destination = normalized["confirmatory fixed component"]
    source_artifact = normalized["confirmatory source artifact"]
    if destination.exists() or destination.is_symlink():
        raise FrozenArtifactError(
            f"refusing to overwrite confirmatory component: {destination}"
        )
    source = Path(source_artifact).resolve()
    if method in {"M1", "M2"}:
        resolved = resolve_static_direction(
            source,
            method=method,
            layer=layer,
            site=site,
        )
        if not math.isclose(
            reference_rms, resolved.reference_rms, rel_tol=0, abs_tol=1e-12
        ):
            raise DataValidationError(
                "confirmatory static RMS differs from its construction artifact"
            )
        reference_rms = resolved.reference_rms
    direction = _source_direction(
        source,
        method=method,
        layer=layer,
        site=site,
        token_scope=token_scope,
        alpha=standardized_alpha,
        sparsity=sparsity,
        reference_rms=reference_rms,
    ).detach().cpu().float().contiguous()
    direction_norm = float(torch.linalg.vector_norm(direction))
    direction_sha = _direction_sha256(direction)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent)
    )
    try:
        _copy_artifact(source, stage / "source-artifact")
        tensor_path = stage / "direction.safetensors"
        save_file({"direction": direction}, tensor_path)
        body = {
            "schema_version": 1,
            "component_kind": "confirmatory-fixed-intervention",
            "method": method,
            "source_artifact_sha256": sha256_path(stage / "source-artifact"),
            "direction_tensor_sha256": sha256_file(tensor_path),
            "direction_sha256": direction_sha,
            "direction_norm": direction_norm,
            "reference_rms": float(reference_rms),
            "layer": layer,
            "site": site.value,
            "token_scope": token_scope.value,
            "standardized_alpha": float(standardized_alpha),
            "sparsity": sparsity,
            "decay": float(decay),
        }
        (stage / "manifest.json").write_text(
            json.dumps(
                {**body, "manifest_digest": stable_hash(body)},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        load_confirmatory_fixed_component(stage)
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return load_confirmatory_fixed_component(destination)


def load_confirmatory_fixed_component(
    directory: str | Path,
) -> ConfirmatoryFixedComponent:
    source = Path(directory)
    if (
        source.is_symlink()
        or not source.is_dir()
        or {item.name for item in source.iterdir()}
        != {"manifest.json", "direction.safetensors", "source-artifact"}
        or any(item.is_symlink() for item in source.rglob("*"))
    ):
        raise FrozenArtifactError("confirmatory fixed component inventory differs")
    try:
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(
            f"cannot read confirmatory component manifest: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise FrozenArtifactError("confirmatory component manifest is invalid")
    digest = manifest.pop("manifest_digest", None)
    expected_fields = {
        "schema_version",
        "component_kind",
        "method",
        "source_artifact_sha256",
        "direction_tensor_sha256",
        "direction_sha256",
        "direction_norm",
        "reference_rms",
        "layer",
        "site",
        "token_scope",
        "standardized_alpha",
        "sparsity",
        "decay",
    }
    tensor_path = source / "direction.safetensors"
    if (
        set(manifest) != expected_fields
        or manifest.get("schema_version") != 1
        or manifest.get("component_kind") != "confirmatory-fixed-intervention"
        or digest != stable_hash(manifest)
        or manifest.get("source_artifact_sha256")
        != sha256_path(source / "source-artifact")
        or manifest.get("direction_tensor_sha256") != sha256_file(tensor_path)
    ):
        raise FrozenArtifactError("confirmatory fixed component identity differs")
    try:
        tensors = load_file(tensor_path, device="cpu")
        if set(tensors) != {"direction"}:
            raise FrozenArtifactError("confirmatory direction tensor set differs")
        component = ConfirmatoryFixedComponent(
            directory=source,
            method=str(manifest["method"]),
            source_artifact_sha256=str(manifest["source_artifact_sha256"]),
            direction=tensors["direction"],
            direction_sha256=str(manifest["direction_sha256"]),
            direction_norm=float(manifest["direction_norm"]),
            reference_rms=float(manifest["reference_rms"]),
            layer=int(manifest["layer"]),
            site=ActivationSite(str(manifest["site"])),
            token_scope=TokenScope(str(manifest["token_scope"])),
            standardized_alpha=float(manifest["standardized_alpha"]),
            sparsity=(
                float(manifest["sparsity"])
                if manifest["sparsity"] is not None
                else None
            ),
            decay=float(manifest["decay"]),
            fingerprint=sha256_path(source),
        )
    except (KeyError, TypeError, ValueError, DataValidationError) as exc:
        raise FrozenArtifactError(
            f"invalid confirmatory fixed component: {exc}"
        ) from exc
    derived = _source_direction(
        source / "source-artifact",
        method=component.method,
        layer=component.layer,
        site=component.site,
        token_scope=component.token_scope,
        alpha=component.standardized_alpha,
        sparsity=component.sparsity,
        reference_rms=component.reference_rms,
    )
    if not torch.equal(
        component.direction,
        derived.detach().cpu().float().contiguous(),
    ):
        raise FrozenArtifactError(
            "confirmatory direction differs from its promoted source artifact"
        )
    return component

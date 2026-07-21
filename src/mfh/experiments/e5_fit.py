"""Deterministic, provenance-bound fitting of the complete E5 controller grid."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from torch import Tensor

from mfh.artifact_namespace import validate_active_study_artifact_paths
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.experiments.e5_adaptive import (
    E5AblationSpec,
    E5Protocol,
    build_e5_ablation_grid,
    load_e5_controller_binding,
    write_e5_controller_binding,
)
from mfh.experiments.e5_capture import E5FitCaptureData
from mfh.experiments.e5_layer_labels import E5LayerLabelData
from mfh.experiments.e5_types import E5FitRecipe as E5FitRecipe
from mfh.inference.architecture import HookKey
from mfh.methods.adaptive import (
    AdaptiveController,
    AlphaController,
    AlphaMode,
    LayerSelector,
    RouterKind,
    assign_to_vector_regions,
    fit_adaptive_router,
    fit_layer_selector,
    fit_routed_vector_bank,
)
from mfh.methods.features import FeatureComposition
from mfh.methods.probes import (
    CalibratedProbe,
    IsotonicCalibrator,
    ProbeDataset,
    TemperatureCalibrator,
    load_calibrated_probe,
)
from mfh.provenance import canonical_json, sha256_file, sha256_path, stable_hash

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMPOSITIONS = MappingProxyType(
    {
        "one_layer": FeatureComposition.SINGLE_LAYER,
        "concatenated_layers": FeatureComposition.CONCATENATED_LAYERS,
        "layer_differences": FeatureComposition.LAYER_DIFFERENCES,
    }
)
_ROUTERS = MappingProxyType(
    {
        "nearest_centroid": RouterKind.NEAREST_CENTROID,
        "linear_softmax": RouterKind.LINEAR_SOFTMAX,
        "two_layer_mlp": RouterKind.TWO_LAYER_MLP,
    }
)
_ALPHAS = MappingProxyType(
    {
        "fixed": AlphaMode.FIXED,
        "risk_gated": AlphaMode.RISK_GATED,
        "risk_gated_hard_threshold": AlphaMode.HARD_THRESHOLD,
    }
)


def _digest(value: object, context: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise DataValidationError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().cpu().float().contiguous()
    if tensor.ndim != 2 or tensor.shape[0] == 0 or not torch.isfinite(tensor).all():
        raise DataValidationError("E5 fit activation must be one finite matrix")
    return hashlib.sha256(tensor.numpy().tobytes(order="C")).hexdigest()


def _probe_object_sha256(probe: CalibratedProbe) -> str:
    """Hash every probe tensor and semantic field independent of serialization."""

    tensors = {
        "feature_mean": probe.state.feature_mean,
        "feature_scale": probe.state.feature_scale,
        **{f"parameter.{name}": value for name, value in probe.state.parameters.items()},
    }
    calibration: dict[str, Any]
    if isinstance(probe.calibrator, TemperatureCalibrator):
        calibration = {
            "kind": "temperature",
            "temperature": probe.calibrator.temperature,
        }
    elif isinstance(probe.calibrator, IsotonicCalibrator):
        calibration = {
            "kind": "isotonic",
            "curves": [
                {
                    "upper_bounds_sha256": _tensor_sha256(value.upper_bounds.unsqueeze(0)),
                    "values_sha256": _tensor_sha256(value.values.unsqueeze(0)),
                }
                for value in probe.calibrator.curves
            ],
        }
    else:  # pragma: no cover - closed Calibrator union
        raise DataValidationError("E5 probe calibrator is unsupported")
    return stable_hash(
        {
            "schema_version": probe.schema_version,
            "task": probe.task.value,
            "kind": probe.state.kind.value,
            "labels": list(probe.state.labels),
            "hidden_width": probe.state.hidden_width,
            "training_fingerprint": probe.training_fingerprint,
            "calibration_fingerprint": probe.calibration_fingerprint,
            "training_schema": probe.training_schema.to_dict(),
            "calibration_schema": probe.calibration_schema.to_dict(),
            "tensors": {
                name: {
                    "shape": list(value.shape),
                    "float32_sha256": _tensor_sha256(value.reshape(1, -1)),
                }
                for name, value in sorted(tensors.items())
            },
            "calibration": calibration,
        }
    )


def _verified_probe_artifact(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_probe: CalibratedProbe,
) -> str:
    source = Path(path)
    if sha256_path(source) != expected_sha256:
        raise DataValidationError("E5 risk-probe artifact digest differs from E2")
    loaded = load_calibrated_probe(source)
    digest = _probe_object_sha256(expected_probe)
    if _probe_object_sha256(loaded) != digest:
        raise DataValidationError("E5 live risk probe differs from its E2 artifact tensors")
    return digest


def _dataset_receipt(dataset: ProbeDataset, *, row_count: int) -> dict[str, Any]:
    schema = dataset.feature_schema
    if schema is None or len(dataset.question_ids) != row_count:
        raise DataValidationError("E5 fit dataset lacks its exact feature schema")
    return {
        "data_fingerprint": dataset.data_fingerprint,
        "feature_schema_digest": schema.digest,
        "question_ids_sha256": stable_hash(list(dataset.question_ids)),
        "row_count": row_count,
    }


@dataclass(frozen=True, slots=True)
class E5FittedGrid:
    protocol: E5Protocol
    recipe: E5FitRecipe
    controllers: Mapping[str, AdaptiveController]
    controller_fit_ids: Mapping[str, str]
    capture_attestation_digest: str
    fit_provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        expected = tuple(value.spec_id for value in build_e5_ablation_grid(self.protocol))
        if (
            type(self.protocol) is not E5Protocol
            or type(self.recipe) is not E5FitRecipe
            or tuple(self.controllers) != expected
            or tuple(self.controller_fit_ids) != expected
            or any(
                type(value) is not str or _SHA256.fullmatch(value) is None
                for value in self.controller_fit_ids.values()
            )
            or _SHA256.fullmatch(self.capture_attestation_digest) is None
            or not isinstance(self.fit_provenance, Mapping)
            or self.fit_provenance.get("capture_attestation_digest")
            != self.capture_attestation_digest
        ):
            raise DataValidationError("E5 fitted grid is incomplete or out of order")
        object.__setattr__(self, "controllers", MappingProxyType(dict(self.controllers)))
        object.__setattr__(
            self, "controller_fit_ids", MappingProxyType(dict(self.controller_fit_ids))
        )
        object.__setattr__(
            self,
            "fit_provenance",
            MappingProxyType(json.loads(canonical_json(dict(self.fit_provenance)))),
        )


@dataclass(frozen=True, slots=True)
class VerifiedE5FittedGrid:
    directory: Path
    manifest: Mapping[str, Any]
    controller_directories: Mapping[str, Path]
    scientific_eligible: bool

    def __post_init__(self) -> None:
        if (
            not self.directory.is_absolute()
            or not self.controller_directories
            or tuple(self.controller_directories)
            != tuple(value["spec_id"] for value in self.manifest.get("controllers", []))
            or type(self.scientific_eligible) is not bool
        ):
            raise DataValidationError("verified E5 fitted grid is invalid")
        object.__setattr__(
            self,
            "controller_directories",
            MappingProxyType(dict(self.controller_directories)),
        )
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))


@dataclass(frozen=True, slots=True)
class VerifiedE5ControllerBindings:
    """Exact one-to-one execution bindings for a persisted E5 grid."""

    directory: Path
    manifest: Mapping[str, Any]
    binding_paths: Mapping[str, Path]
    scientific_eligible: bool

    def __post_init__(self) -> None:
        if (
            not self.directory.is_absolute()
            or not self.binding_paths
            or tuple(self.binding_paths)
            != tuple(value["spec_id"] for value in self.manifest.get("bindings", []))
            or type(self.scientific_eligible) is not bool
        ):
            raise DataValidationError("verified E5 controller bindings are invalid")
        object.__setattr__(
            self,
            "binding_paths",
            MappingProxyType(dict(self.binding_paths)),
        )
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))


def e5_fit_capture_attestation_body(
    *,
    protocol: E5Protocol,
    recipe: E5FitRecipe,
    execution_public_key: str,
    runtime_artifact_sha256: str,
    e2_probe_bundle_sha256: str,
    e3_static_vectors_sha256: str,
    e3_construction_sha256: str,
    risk_probes: Mapping[FeatureComposition, CalibratedProbe],
    risk_probe_artifact_sha256: Mapping[FeatureComposition, str],
    risk_probe_artifact_paths: Mapping[FeatureComposition, str | Path],
    controller_datasets: Mapping[FeatureComposition, ProbeDataset],
    capture_data: E5FitCaptureData,
    layer_labels: E5LayerLabelData,
) -> dict[str, Any]:
    """Construct the exact native-capture receipt verified before CPU fitting."""

    for digest_value, context in (
        (execution_public_key, "E5 execution public key"),
        (runtime_artifact_sha256, "E5 runtime artifact"),
        (e2_probe_bundle_sha256, "E5 E2 bundle"),
        (e3_static_vectors_sha256, "E5 E3 vectors"),
        (e3_construction_sha256, "E5 E3 construction"),
        (layer_labels.artifact_sha256, "E5 layer-label receipt"),
    ):
        _digest(digest_value, context)
    expected_compositions = {_COMPOSITIONS[value] for value in protocol.controller_inputs}
    if type(capture_data) is not E5FitCaptureData:
        raise DataValidationError("E5 fitting requires an exact verified capture handle")
    verified_capture = capture_data.verified
    capture_plan = verified_capture.plan
    vector_datasets = capture_data.vector_datasets
    vector_activations = capture_data.vector_activations
    if (
        type(protocol) is not E5Protocol
        or type(recipe) is not E5FitRecipe
        or set(risk_probe_artifact_sha256) != expected_compositions
        or set(risk_probes) != expected_compositions
        or set(risk_probe_artifact_paths) != expected_compositions
        or set(controller_datasets) != expected_compositions
        or set(vector_datasets) != expected_compositions
        or any(_SHA256.fullmatch(value) is None for value in risk_probe_artifact_sha256.values())
        or not vector_activations
        or not verified_capture.complete
        or verified_capture.chain_head is None
        or type(layer_labels) is not E5LayerLabelData
        or not layer_labels.verified.complete
        or layer_labels.verified.chain_head is None
        or sha256_path(layer_labels.verified.directory) != layer_labels.artifact_sha256
        or layer_labels.verified.plan.get("recipe") != recipe.to_dict()
        or layer_labels.verified.plan.get("execution_public_key") != execution_public_key
        or layer_labels.verified.plan.get("fit_capture_artifact_sha256")
        != capture_data.capture_artifact_sha256
        or layer_labels.verified.plan.get("fit_capture_plan_identity")
        != capture_plan.get("plan_identity")
        or layer_labels.verified.plan.get("fit_capture_chain_head") != verified_capture.chain_head
        or sha256_path(verified_capture.directory) != capture_data.capture_artifact_sha256
        or capture_plan.get("protocol") != protocol.to_dict()
        or capture_plan.get("recipe") != recipe.to_dict()
        or capture_plan.get("execution_public_key") != execution_public_key
        or capture_plan.get("runtime_artifact_sha256") != runtime_artifact_sha256
        or capture_plan.get("e2_probe_bundle_sha256") != e2_probe_bundle_sha256
        or capture_plan.get("e3_static_vectors_sha256") != e3_static_vectors_sha256
        or capture_plan.get("e3_construction_sha256") != e3_construction_sha256
        or (protocol.scientific_eligible and not verified_capture.scientific_eligible)
    ):
        raise DataValidationError("E5 fit-capture inventory differs")
    probe_object_digests = {
        value: _verified_probe_artifact(
            risk_probe_artifact_paths[value],
            expected_sha256=risk_probe_artifact_sha256[value],
            expected_probe=risk_probes[value],
        )
        for value in expected_compositions
    }
    row_count = len(next(iter(vector_datasets.values())).question_ids)
    controller_count = len(next(iter(controller_datasets.values())).question_ids)
    if (
        any(len(value.question_ids) != row_count for value in vector_datasets.values())
        or any(
            len(value.question_ids) != controller_count for value in controller_datasets.values()
        )
        or len(layer_labels.best_layers_two) != controller_count
        or len(layer_labels.best_layers_three) != controller_count
    ):
        raise DataValidationError("E5 fit-capture row counts differ")
    controller_reference = next(iter(controller_datasets.values()))
    vector_reference = next(iter(vector_datasets.values()))
    controller_identity = (
        controller_reference.question_ids,
        controller_reference.group_ids,
        controller_reference.outcomes,
    )
    vector_identity = (
        vector_reference.question_ids,
        vector_reference.group_ids,
        vector_reference.outcomes,
    )
    if (
        any(
            (value.question_ids, value.group_ids, value.outcomes) != controller_identity
            for value in controller_datasets.values()
        )
        or any(
            (value.question_ids, value.group_ids, value.outcomes) != vector_identity
            for value in vector_datasets.values()
        )
        or set(controller_reference.question_ids) & set(vector_reference.question_ids)
        or set(controller_reference.group_ids) & set(vector_reference.group_ids)
        or (
            layer_labels.question_ids,
            layer_labels.group_ids,
            layer_labels.outcomes,
        )
        != controller_identity
        or any(value not in recipe.two_layer_candidates for value in layer_labels.best_layers_two)
        or any(
            value not in recipe.three_layer_candidates for value in layer_labels.best_layers_three
        )
    ):
        raise DataValidationError("E5 compositions are misaligned or T-controller/T-steer overlap")
    activations: dict[str, Any] = {}
    for key in sorted(vector_activations):
        activation = vector_activations[key]
        if activation.shape[0] != row_count:
            raise DataValidationError("E5 vector activation rows differ from T-steer")
        activations[key.artifact_key] = {
            "layer": key.layer,
            "site": key.site.value,
            "shape": list(activation.shape),
            "float32_sha256": _tensor_sha256(activation),
        }
    return {
        "receipt_kind": "e5-native-controller-fit-capture-v1",
        "protocol": protocol.to_dict(),
        "recipe": recipe.to_dict(),
        "execution_public_key": execution_public_key,
        "capture_artifact_sha256": capture_data.capture_artifact_sha256,
        "capture_plan_identity": capture_plan["plan_identity"],
        "capture_shard_chain_head": verified_capture.chain_head,
        "capture_pairs_completed": verified_capture.pairs_completed,
        "runtime_artifact_sha256": runtime_artifact_sha256,
        "e2_probe_bundle_sha256": e2_probe_bundle_sha256,
        "e3_static_vectors_sha256": e3_static_vectors_sha256,
        "e3_construction_sha256": e3_construction_sha256,
        "layer_label_receipt_sha256": layer_labels.artifact_sha256,
        "layer_label_plan_identity": layer_labels.verified.plan["plan_identity"],
        "layer_label_chain_head": layer_labels.verified.chain_head,
        "layer_label_question_ids_sha256": stable_hash(list(layer_labels.question_ids)),
        "risk_probes": {
            value.value: {
                "artifact_sha256": risk_probe_artifact_sha256[value],
                "object_sha256": probe_object_digests[value],
            }
            for value in sorted(expected_compositions, key=lambda item: item.value)
        },
        "controller_datasets": {
            value.value: _dataset_receipt(controller_datasets[value], row_count=controller_count)
            for value in sorted(expected_compositions, key=lambda item: item.value)
        },
        "vector_datasets": {
            value.value: _dataset_receipt(vector_datasets[value], row_count=row_count)
            for value in sorted(expected_compositions, key=lambda item: item.value)
        },
        "vector_activations": activations,
        "best_layers_two": [int(value) for value in layer_labels.best_layers_two],
        "best_layers_two_sha256": stable_hash(
            [int(value) for value in layer_labels.best_layers_two]
        ),
        "best_layers_three": [int(value) for value in layer_labels.best_layers_three],
        "best_layers_three_sha256": stable_hash(
            [int(value) for value in layer_labels.best_layers_three]
        ),
    }


def sign_e5_fit_capture_attestation(
    body: Mapping[str, Any], *, private_key_hex: str
) -> Mapping[str, Any]:
    if type(private_key_hex) is not str or _SHA256.fullmatch(private_key_hex) is None:
        raise DataValidationError("E5 fit-capture key must be 32-byte lowercase hex")
    try:
        signature = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_key_hex)).sign(
            canonical_json(dict(body)).encode()
        )
    except ValueError as exc:
        raise DataValidationError(f"invalid E5 fit-capture key: {exc}") from exc
    return MappingProxyType({"body": dict(body), "signature": signature.hex()})


def _verify_capture_attestation(
    attestation: Mapping[str, Any], *, expected_body: Mapping[str, Any]
) -> str:
    body = attestation.get("body")
    signature = attestation.get("signature")
    if (
        set(attestation) != {"body", "signature"}
        or not isinstance(body, Mapping)
        or canonical_json(dict(body)) != canonical_json(dict(expected_body))
        or not isinstance(signature, str)
        or re.fullmatch(r"[0-9a-f]{128}", signature) is None
    ):
        raise DataValidationError("E5 fit-capture attestation differs from live tensors")
    public_key = expected_body.get("execution_public_key")
    if not isinstance(public_key, str):
        raise DataValidationError("E5 fit-capture public key is missing")
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key)).verify(
            bytes.fromhex(signature), canonical_json(dict(expected_body)).encode()
        )
    except (InvalidSignature, ValueError) as exc:
        raise DataValidationError("E5 fit-capture signature is invalid") from exc
    return stable_hash({"body": dict(body), "signature": signature})


def _fit_id(spec: E5AblationSpec) -> str:
    value = spec.to_dict()
    value.pop("intervention_timing")
    return stable_hash(value)


def fit_e5_controller_grid(
    *,
    protocol: E5Protocol,
    recipe: E5FitRecipe,
    risk_probes: Mapping[FeatureComposition, CalibratedProbe],
    risk_probe_artifact_sha256: Mapping[FeatureComposition, str],
    risk_probe_artifact_paths: Mapping[FeatureComposition, str | Path],
    controller_datasets: Mapping[FeatureComposition, ProbeDataset],
    capture_data: E5FitCaptureData,
    layer_labels: E5LayerLabelData,
    capture_attestation: Mapping[str, Any],
    runtime_artifact_sha256: str,
    e2_probe_bundle_sha256: str,
    e3_static_vectors_sha256: str,
    e3_construction_sha256: str,
    expected_execution_public_key: str,
) -> E5FittedGrid:
    """Fit all unique E5 controllers and map them to the full 972-arm grid."""

    if type(protocol) is not E5Protocol or type(recipe) is not E5FitRecipe:
        raise DataValidationError("E5 fitting requires exact protocol and recipe types")
    expected_compositions = {_COMPOSITIONS[value] for value in protocol.controller_inputs}
    if set(risk_probes) != expected_compositions:
        raise DataValidationError("E5 risk-probe composition grid differs")
    if type(capture_data) is not E5FitCaptureData:
        raise DataValidationError("E5 fit requires a verified native capture")
    if type(layer_labels) is not E5LayerLabelData:
        raise DataValidationError("E5 fit requires exact layer-label data")
    layer_labels.assert_current()
    vector_datasets = capture_data.vector_datasets
    vector_activations = capture_data.vector_activations
    _digest(expected_execution_public_key, "E5 trusted execution public key")
    body_value = capture_attestation.get("body")
    public_key = body_value.get("execution_public_key") if isinstance(body_value, Mapping) else None
    if public_key != expected_execution_public_key:
        raise DataValidationError("E5 capture attestation key differs from its trusted plan")
    expected_body = e5_fit_capture_attestation_body(
        protocol=protocol,
        recipe=recipe,
        execution_public_key=expected_execution_public_key,
        runtime_artifact_sha256=runtime_artifact_sha256,
        e2_probe_bundle_sha256=e2_probe_bundle_sha256,
        e3_static_vectors_sha256=e3_static_vectors_sha256,
        e3_construction_sha256=e3_construction_sha256,
        risk_probes=risk_probes,
        risk_probe_artifact_sha256=risk_probe_artifact_sha256,
        risk_probe_artifact_paths=risk_probe_artifact_paths,
        controller_datasets=controller_datasets,
        capture_data=capture_data,
        layer_labels=layer_labels,
    )
    attestation_digest = _verify_capture_attestation(
        capture_attestation, expected_body=expected_body
    )
    required_hooks = {
        HookKey(layer, recipe.intervention_site) for layer in recipe.three_layer_candidates
    }
    if set(vector_activations) != required_hooks:
        raise DataValidationError("E5 vector activations differ from candidate geometry")

    banks: dict[tuple[FeatureComposition, int], Any] = {}
    routers: dict[tuple[FeatureComposition, int, str], Any] = {}
    selectors: dict[tuple[FeatureComposition, str], LayerSelector] = {}
    for composition in sorted(expected_compositions, key=lambda item: item.value):
        risk = risk_probes[composition]
        controller_rows = controller_datasets[composition]
        vector_rows = vector_datasets[composition]
        if (
            controller_rows.feature_schema is None
            or vector_rows.feature_schema is None
            or risk.training_fingerprint != controller_rows.data_fingerprint
            or risk.training_schema != controller_rows.feature_schema
            or risk.training_schema.composition is not composition
            or vector_rows.feature_schema.composition is not composition
        ):
            raise DataValidationError("E5 E2 risk probe and captured datasets differ")
        selectors[(composition, "two_layer_router")] = fit_layer_selector(
            controller_rows,
            layer_labels.best_layers_two,
            candidate_layers=recipe.two_layer_candidates,
            kind=RouterKind.TWO_LAYER_MLP,
            seed=recipe.layer_seed,
            epochs=recipe.layer_epochs,
        )
        selectors[(composition, "three_layer_router")] = fit_layer_selector(
            controller_rows,
            layer_labels.best_layers_three,
            candidate_layers=recipe.three_layer_candidates,
            kind=RouterKind.TWO_LAYER_MLP,
            seed=recipe.layer_seed,
            epochs=recipe.layer_epochs,
        )
        for count in protocol.vector_counts:
            bank, _assignments = fit_routed_vector_bank(
                vector_rows,
                vector_activations,
                cluster_count=count,
                seed=recipe.vector_seed,
                minimum_class_count=recipe.minimum_class_count,
                source_artifact_sha256=e3_static_vectors_sha256,
            )
            banks[(composition, count)] = bank
            assignments = assign_to_vector_regions(controller_rows, bank)
            for router_name in protocol.routers:
                routers[(composition, count, router_name)] = fit_adaptive_router(
                    controller_rows,
                    assignments,
                    bank.centers,
                    kind=_ROUTERS[router_name],
                    seed=recipe.router_seed,
                    hidden_width=recipe.router_hidden_width,
                    epochs=recipe.router_epochs,
                    distance_temperature=recipe.distance_temperature,
                )

    alpha_controllers = {
        name: AlphaController(
            _ALPHAS[name],
            alpha_max=recipe.alpha_max,
            beta=recipe.alpha_beta,
            threshold=recipe.alpha_threshold,
        )
        for name in protocol.alpha_modes
    }
    controllers: dict[str, AdaptiveController] = {}
    fit_ids: dict[str, str] = {}
    fitted_by_id: dict[str, AdaptiveController] = {}
    for spec in build_e5_ablation_grid(protocol):
        composition = _COMPOSITIONS[spec.controller_input]
        fit_id = _fit_id(spec)
        controller = fitted_by_id.get(fit_id)
        if controller is None:
            selector = selectors.get((composition, spec.layer_mode))
            controller = AdaptiveController(
                risk_probe=risk_probes[composition],
                vector_bank=banks[(composition, spec.vector_count)],
                vector_router=routers[(composition, spec.vector_count, spec.router)],
                alpha_controller=alpha_controllers[spec.alpha_mode],
                fixed_layer=(recipe.fixed_best_layer if spec.layer_mode == "fixed_best" else None),
                layer_selector=selector,
            )
            fitted_by_id[fit_id] = controller
        controllers[spec.spec_id] = controller
        fit_ids[spec.spec_id] = fit_id
    return E5FittedGrid(
        protocol=protocol,
        recipe=recipe,
        controllers=controllers,
        controller_fit_ids=fit_ids,
        capture_attestation_digest=attestation_digest,
        fit_provenance={
            "schema_version": 1,
            "capture_attestation_digest": attestation_digest,
            "capture_artifact_sha256": capture_data.capture_artifact_sha256,
            "capture_plan_identity": capture_data.verified.plan["plan_identity"],
            "capture_shard_chain_head": capture_data.verified.chain_head,
            "protocol_sha256": stable_hash(protocol.to_dict()),
            "recipe_sha256": stable_hash(recipe.to_dict()),
            "runtime_artifact_sha256": runtime_artifact_sha256,
            "e2_probe_bundle_sha256": e2_probe_bundle_sha256,
            "e3_static_vectors_sha256": e3_static_vectors_sha256,
            "e3_construction_sha256": e3_construction_sha256,
            "layer_label_receipt_sha256": layer_labels.artifact_sha256,
            "layer_label_plan_identity": layer_labels.verified.plan["plan_identity"],
            "layer_label_chain_head": layer_labels.verified.chain_head,
            "execution_public_key": expected_execution_public_key,
            "risk_probes": expected_body["risk_probes"],
        },
    )


def save_e5_fitted_controller(
    directory: str | Path,
    *,
    fitted: E5FittedGrid,
    spec_id: str,
) -> Mapping[str, Any]:
    """Persist one controller together with its non-optional signed-fit lineage."""

    from mfh.methods.adaptive import save_adaptive_controller

    if spec_id not in fitted.controllers:
        raise DataValidationError("E5 fitted controller spec is absent")
    destination = Path(directory)
    save_adaptive_controller(destination, fitted.controllers[spec_id])
    body = {
        "schema_version": 1,
        "spec_id": spec_id,
        "controller_fit_id": fitted.controller_fit_ids[spec_id],
        **dict(fitted.fit_provenance),
    }
    value = {**body, "provenance_digest": stable_hash(body)}
    path = destination / "e5-fit-provenance.json"
    descriptor, temporary = tempfile.mkstemp(prefix=".e5-fit-provenance.", dir=destination)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return MappingProxyType(
        {
            "controller_sha256": sha256_path(destination),
            "provenance_sha256": sha256_file(path),
            "provenance_digest": value["provenance_digest"],
        }
    )


def save_e5_fitted_grid(
    directory: str | Path,
    *,
    fitted: E5FittedGrid,
) -> VerifiedE5FittedGrid:
    """Atomically persist the complete ordered E5 grid and fit lineage."""

    if type(fitted) is not E5FittedGrid:
        raise DataValidationError("E5 grid save requires an exact fitted grid")
    destination = validate_active_study_artifact_paths({"E5 fitted controller grid": directory})[
        "E5 fitted controller grid"
    ]
    if destination.exists() or destination.is_symlink():
        raise DataValidationError(f"refusing to overwrite E5 fitted grid: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        controllers_root = stage / "controllers"
        controllers_root.mkdir()
        rows: list[dict[str, Any]] = []
        for spec in build_e5_ablation_grid(fitted.protocol):
            controller_directory = controllers_root / spec.spec_id
            receipt = save_e5_fitted_controller(
                controller_directory,
                fitted=fitted,
                spec_id=spec.spec_id,
            )
            rows.append(
                {
                    "spec_id": spec.spec_id,
                    "spec": spec.to_dict(),
                    "controller_fit_id": fitted.controller_fit_ids[spec.spec_id],
                    "relative_directory": f"controllers/{spec.spec_id}",
                    **dict(receipt),
                }
            )
        body = {
            "schema_version": 1,
            "phase": "E5-controller-grid-fit",
            "runner_source_sha256": sha256_file(Path(__file__)),
            "protocol": fitted.protocol.to_dict(),
            "recipe": fitted.recipe.to_dict(),
            "capture_attestation_digest": fitted.capture_attestation_digest,
            "fit_provenance": dict(fitted.fit_provenance),
            "controller_count": len(rows),
            "unique_controller_fit_count": len(set(fitted.controller_fit_ids.values())),
            "controllers": rows,
            "controllers_sha256": sha256_path(controllers_root),
            "scientific_eligible": bool(fitted.protocol.scientific_eligible and len(rows) == 972),
        }
        manifest = {**body, "manifest_digest": stable_hash(body)}
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return verify_e5_fitted_grid(destination)


def verify_e5_fitted_grid(directory: str | Path) -> VerifiedE5FittedGrid:
    """Reload every saved controller and replay the complete grid manifest."""

    source = Path(directory)
    if (
        source.is_symlink()
        or not source.is_dir()
        or {value.name for value in source.iterdir()} != {"manifest.json", "controllers"}
    ):
        raise DataValidationError("E5 fitted grid inventory differs")
    try:
        value = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot load E5 fitted grid manifest: {exc}") from exc
    if not isinstance(value, dict):
        raise DataValidationError("E5 fitted grid manifest must be a mapping")
    body = dict(value)
    digest = body.pop("manifest_digest", None)
    expected_keys = {
        "schema_version",
        "phase",
        "runner_source_sha256",
        "protocol",
        "recipe",
        "capture_attestation_digest",
        "fit_provenance",
        "controller_count",
        "unique_controller_fit_count",
        "controllers",
        "controllers_sha256",
        "scientific_eligible",
    }
    if (
        set(body) != expected_keys
        or digest != stable_hash(body)
        or body["schema_version"] != 1
        or body["phase"] != "E5-controller-grid-fit"
        or body["runner_source_sha256"] != sha256_file(Path(__file__))
        or type(body["controllers"]) is not list
        or body["controller_count"] != len(body["controllers"])
        or body["controllers_sha256"] != sha256_path(source / "controllers")
    ):
        raise DataValidationError("E5 fitted grid manifest differs")
    protocol = E5Protocol.from_dict(body["protocol"])
    E5FitRecipe.from_dict(body["recipe"])
    specs = build_e5_ablation_grid(protocol)
    rows = body["controllers"]
    if len(rows) != len(specs):
        raise DataValidationError("E5 fitted grid cardinality differs")
    directories: dict[str, Path] = {}
    fit_ids: list[str] = []
    for spec, row in zip(specs, rows, strict=True):
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "spec_id",
                "spec",
                "controller_fit_id",
                "relative_directory",
                "controller_sha256",
                "provenance_sha256",
                "provenance_digest",
            }
            or row["spec_id"] != spec.spec_id
            or row["spec"] != spec.to_dict()
            or row["controller_fit_id"] != _fit_id(spec)
            or row["relative_directory"] != f"controllers/{spec.spec_id}"
        ):
            raise DataValidationError("E5 fitted controller row differs")
        controller_directory = source / row["relative_directory"]
        provenance = _load_saved_fit_provenance(controller_directory, expected_spec_id=spec.spec_id)
        if (
            sha256_path(controller_directory) != row["controller_sha256"]
            or sha256_file(controller_directory / "e5-fit-provenance.json")
            != row["provenance_sha256"]
            or provenance["provenance_digest"] != row["provenance_digest"]
            or provenance["capture_attestation_digest"] != body["capture_attestation_digest"]
            or {key: provenance[key] for key in body["fit_provenance"]} != body["fit_provenance"]
        ):
            raise DataValidationError("E5 fitted controller provenance differs")
        from mfh.methods.adaptive import load_adaptive_controller

        load_adaptive_controller(controller_directory)
        directories[spec.spec_id] = controller_directory.resolve()
        fit_ids.append(row["controller_fit_id"])
    if body["unique_controller_fit_count"] != len(set(fit_ids)) or body[
        "scientific_eligible"
    ] is not bool(protocol.scientific_eligible and len(rows) == 972):
        raise DataValidationError("E5 fitted-grid fit counts differ")
    return VerifiedE5FittedGrid(
        directory=source.resolve(),
        manifest=MappingProxyType(value),
        controller_directories=MappingProxyType(directories),
        scientific_eligible=body["scientific_eligible"],
    )


def package_e5_controller_bindings(
    directory: str | Path,
    *,
    fitted_grid_directory: str | Path,
    expected_execution_public_key: str,
) -> VerifiedE5ControllerBindings:
    """Atomically bind every saved controller to its exact ablation arm."""

    normalized = validate_active_study_artifact_paths(
        {
            "E5 controller-binding package": directory,
            "E5 fitted controller grid": fitted_grid_directory,
        }
    )
    destination = normalized["E5 controller-binding package"]
    grid = verify_e5_fitted_grid(normalized["E5 fitted controller grid"])
    _digest(expected_execution_public_key, "E5 execution public key")
    provenance_key = grid.manifest["fit_provenance"].get("execution_public_key")
    if provenance_key != expected_execution_public_key:
        raise DataValidationError("E5 binding key differs from fitted-grid provenance")
    if destination.exists() or destination.is_symlink():
        raise FrozenArtifactError(
            f"refusing to overwrite E5 controller-binding package: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        binding_root = stage / "bindings"
        binding_root.mkdir()
        rows: list[dict[str, Any]] = []
        protocol = E5Protocol.from_dict(grid.manifest["protocol"])
        for spec in build_e5_ablation_grid(protocol):
            relative = f"bindings/{spec.spec_id}.json"
            binding_path = stage / relative
            binding = write_e5_controller_binding(
                binding_path,
                spec=spec,
                controller_directory=grid.controller_directories[spec.spec_id],
                execution_public_key=expected_execution_public_key,
            )
            rows.append(
                {
                    "spec_id": spec.spec_id,
                    "spec": spec.to_dict(),
                    "relative_path": relative,
                    "binding_sha256": sha256_file(binding_path),
                    "binding_digest": binding.binding_digest,
                    "controller_artifact_sha256": binding.controller_artifact_sha256,
                }
            )
        body = {
            "schema_version": 1,
            "phase": "E5-controller-bindings",
            "runner_source_sha256": sha256_file(Path(__file__)),
            "fitted_grid_path": str(grid.directory),
            "fitted_grid_sha256": sha256_path(grid.directory),
            "fitted_grid_manifest_digest": grid.manifest["manifest_digest"],
            "execution_public_key": expected_execution_public_key,
            "protocol": protocol.to_dict(),
            "binding_count": len(rows),
            "bindings": rows,
            "bindings_sha256": sha256_path(binding_root),
            "scientific_eligible": bool(
                grid.scientific_eligible and protocol.scientific_eligible and len(rows) == 972
            ),
        }
        manifest = {**body, "manifest_digest": stable_hash(body)}
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return verify_e5_controller_bindings(destination)


def verify_e5_controller_bindings(
    directory: str | Path,
) -> VerifiedE5ControllerBindings:
    """Replay a complete binding package and every live controller artifact."""

    source = Path(directory)
    if (
        source.is_symlink()
        or not source.is_dir()
        or {value.name for value in source.iterdir()} != {"manifest.json", "bindings"}
        or any(value.is_symlink() for value in source.rglob("*"))
    ):
        raise FrozenArtifactError("E5 controller-binding package inventory differs")
    try:
        value = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read E5 binding-package manifest: {exc}") from exc
    if not isinstance(value, dict):
        raise FrozenArtifactError("E5 binding-package manifest must be a mapping")
    if (source / "manifest.json").read_text(encoding="utf-8") != (
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    ):
        raise FrozenArtifactError("E5 binding-package manifest is not canonical JSON")
    body = dict(value)
    digest = body.pop("manifest_digest", None)
    expected_keys = {
        "schema_version",
        "phase",
        "runner_source_sha256",
        "fitted_grid_path",
        "fitted_grid_sha256",
        "fitted_grid_manifest_digest",
        "execution_public_key",
        "protocol",
        "binding_count",
        "bindings",
        "bindings_sha256",
        "scientific_eligible",
    }
    if (
        set(body) != expected_keys
        or digest != stable_hash(body)
        or body["schema_version"] != 1
        or body["phase"] != "E5-controller-bindings"
        or body["runner_source_sha256"] != sha256_file(Path(__file__))
        or body["fitted_grid_sha256"] != sha256_path(body["fitted_grid_path"])
        or body["bindings_sha256"] != sha256_path(source / "bindings")
        or type(body["bindings"]) is not list
        or body["binding_count"] != len(body["bindings"])
    ):
        raise FrozenArtifactError("E5 controller-binding package manifest differs")
    grid = verify_e5_fitted_grid(body["fitted_grid_path"])
    protocol = E5Protocol.from_dict(body["protocol"])
    specs = build_e5_ablation_grid(protocol)
    if len(body["bindings"]) != len(specs):
        raise FrozenArtifactError("E5 controller-binding package cardinality differs")
    paths: dict[str, Path] = {}
    for spec, row in zip(specs, body["bindings"], strict=True):
        relative = f"bindings/{spec.spec_id}.json"
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "spec_id",
                "spec",
                "relative_path",
                "binding_sha256",
                "binding_digest",
                "controller_artifact_sha256",
            }
            or row["spec_id"] != spec.spec_id
            or row["spec"] != spec.to_dict()
            or row["relative_path"] != relative
        ):
            raise FrozenArtifactError("E5 controller-binding row differs")
        path = source / relative
        binding = load_e5_controller_binding(path)
        if (
            sha256_file(path) != row["binding_sha256"]
            or binding.binding_digest != row["binding_digest"]
            or binding.controller_artifact_sha256 != row["controller_artifact_sha256"]
            or binding.controller_directory != str(grid.controller_directories[spec.spec_id])
            or binding.execution_public_key != body["execution_public_key"]
        ):
            raise FrozenArtifactError("E5 controller binding differs from fitted grid")
        paths[spec.spec_id] = path.resolve()
    scientific = bool(
        grid.scientific_eligible and protocol.scientific_eligible and len(paths) == 972
    )
    if (
        len(body["bindings"]) != len(specs)
        or body["fitted_grid_manifest_digest"] != grid.manifest["manifest_digest"]
        or body["scientific_eligible"] is not scientific
    ):
        raise FrozenArtifactError("E5 controller-binding package eligibility differs")
    return VerifiedE5ControllerBindings(
        directory=source.resolve(),
        manifest=MappingProxyType(value),
        binding_paths=MappingProxyType(paths),
        scientific_eligible=scientific,
    )


def open_e5_controller_bindings_checkpoint(
    directory: str | Path,
) -> VerifiedE5ControllerBindings:
    """Open binding metadata for resume without loading all 972 controllers.

    The returned handle is execution-only: each controller is still validated by
    ``E5ControllerBinding.assert_current`` when its arm is reached, and the terminal
    native verifier calls :func:`verify_e5_controller_bindings` exhaustively.
    """

    source = validate_active_study_artifact_paths(
        {"E5 controller-binding checkpoint": directory}
    )["E5 controller-binding checkpoint"]
    if (
        source.is_symlink()
        or not source.is_dir()
        or {value.name for value in source.iterdir()} != {"manifest.json", "bindings"}
        or (source / "bindings").is_symlink()
        or not (source / "bindings").is_dir()
    ):
        raise FrozenArtifactError("E5 checkpoint binding inventory differs")
    manifest_path = source / "manifest.json"
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read E5 checkpoint binding manifest: {exc}") from exc
    if not isinstance(value, dict) or manifest_path.read_text(encoding="utf-8") != (
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    ):
        raise FrozenArtifactError("E5 checkpoint binding manifest is not canonical JSON")
    body = dict(value)
    digest = body.pop("manifest_digest", None)
    expected_keys = {
        "schema_version",
        "phase",
        "runner_source_sha256",
        "fitted_grid_path",
        "fitted_grid_sha256",
        "fitted_grid_manifest_digest",
        "execution_public_key",
        "protocol",
        "binding_count",
        "bindings",
        "bindings_sha256",
        "scientific_eligible",
    }
    if (
        set(body) != expected_keys
        or digest != stable_hash(body)
        or body["schema_version"] != 1
        or body["phase"] != "E5-controller-bindings"
        or body["runner_source_sha256"] != sha256_file(Path(__file__))
        or type(body["bindings"]) is not list
        or body["binding_count"] != len(body["bindings"])
        or type(body["scientific_eligible"]) is not bool
    ):
        raise FrozenArtifactError("E5 checkpoint binding manifest differs")
    protocol = E5Protocol.from_dict(body["protocol"])
    specs = build_e5_ablation_grid(protocol)
    if len(body["bindings"]) != len(specs):
        raise FrozenArtifactError("E5 checkpoint binding cardinality differs")
    paths: dict[str, Path] = {}
    for spec, row in zip(specs, body["bindings"], strict=True):
        relative = f"bindings/{spec.spec_id}.json"
        path = source / relative
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "spec_id",
                "spec",
                "relative_path",
                "binding_sha256",
                "binding_digest",
                "controller_artifact_sha256",
            }
            or row["spec_id"] != spec.spec_id
            or row["spec"] != spec.to_dict()
            or row["relative_path"] != relative
            or any(
                type(row[name]) is not str or _SHA256.fullmatch(row[name]) is None
                for name in (
                    "binding_sha256",
                    "binding_digest",
                    "controller_artifact_sha256",
                )
            )
            or path.is_symlink()
            or not path.is_file()
        ):
            raise FrozenArtifactError("E5 checkpoint binding row differs")
        paths[spec.spec_id] = path.resolve()
    scientific = bool(protocol.scientific_eligible and len(paths) == 972)
    if body["scientific_eligible"] is not scientific:
        raise FrozenArtifactError("E5 checkpoint binding eligibility differs")
    return VerifiedE5ControllerBindings(
        directory=source.resolve(),
        manifest=MappingProxyType(value),
        binding_paths=MappingProxyType(paths),
        scientific_eligible=scientific,
    )


def _load_saved_fit_provenance(directory: Path, *, expected_spec_id: str) -> Mapping[str, Any]:
    """Local strict provenance replay without importing the phase module."""

    try:
        value = json.loads((directory / "e5-fit-provenance.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot load E5 fit provenance: {exc}") from exc
    if not isinstance(value, dict):
        raise DataValidationError("E5 fit provenance must be a mapping")
    body = dict(value)
    digest = body.pop("provenance_digest", None)
    if (
        digest != stable_hash(body)
        or body.get("schema_version") != 1
        or body.get("spec_id") != expected_spec_id
    ):
        raise DataValidationError("E5 fit provenance digest differs")
    return MappingProxyType(value)

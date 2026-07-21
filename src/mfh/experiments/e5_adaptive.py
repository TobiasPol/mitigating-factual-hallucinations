"""Streaming E5 adaptive ablations and replayable matched-budget selection."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from types import MappingProxyType
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from mfh.contracts import Outcome, Question, TokenScope
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.experiments.e4_baselines import E4ScreenReceipt, load_e4_screen_receipt
from mfh.experiments.evidence import GateResult
from mfh.experiments.gates import write_gate_evidence
from mfh.experiments.model_selection import (
    ACTIVE_MODEL_IDENTITIES,
    ACTIVE_MODEL_NAME,
    validate_active_study_artifact_paths,
)
from mfh.experiments.protocol import ExperimentPhase, StudyProtocol
from mfh.experiments.runner import (
    PhaseCompletion,
    PhaseFalsification,
    PhaseRunLedger,
    _copy_frozen_artifact,
    _resolve_ledger_evidence_path,
    open_phase_prerequisite,
)
from mfh.methods.adaptive import (
    AdaptiveController,
    AlphaMode,
    RouterKind,
    load_adaptive_controller,
)
from mfh.methods.features import FeatureComposition
from mfh.methods.probes import CalibratedProbe, load_calibrated_probe
from mfh.provenance import canonical_json, sha256_file, sha256_path, stable_hash

_UPSTREAM = frozenset({"E2_calibrated_probes", "E3_static_vectors", "E4_promoted_baselines"})
_PROMPTS = ("P0-neutral", "P2-calibrated-abstention")
_PROMPT_HASHES = {
    "P0-neutral": "5b08080e81d4032d853d3a30fda35e7670da6a81cc1ccf17f2b8fc5001bba684",
    "P2-calibrated-abstention": (
        "3170134d9a69836c1b530d1b16585ef7b0d92ea6fadc8f958e2655053e273fe5"
    ),
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SELECTION_RULE = "minimum-risk-within-all-four-m1-budgets-v1"
_ACTIVE_MODEL = ACTIVE_MODEL_IDENTITIES[ACTIVE_MODEL_NAME]
_VECTOR_COUNTS = (1, 4, 8, 16)
_ROUTERS = ("nearest_centroid", "linear_softmax", "two_layer_mlp")
_ALPHA_MODES = ("fixed", "risk_gated", "risk_gated_hard_threshold")
_LAYER_MODES = ("fixed_best", "two_layer_router", "three_layer_router")
_TIMINGS = ("final_prompt", "first_generated", "first_four_generated")
_INPUTS = ("one_layer", "concatenated_layers", "layer_differences")


def _require_digest(value: object, context: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise DataValidationError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _atomic_json(path: str | Path, payload: Mapping[str, Any], context: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite {context}: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


@dataclass(frozen=True, slots=True)
class E5Protocol:
    vector_counts: tuple[int, ...] = _VECTOR_COUNTS
    routers: tuple[str, ...] = _ROUTERS
    alpha_modes: tuple[str, ...] = _ALPHA_MODES
    layer_modes: tuple[str, ...] = _LAYER_MODES
    intervention_timings: tuple[str, ...] = _TIMINGS
    controller_inputs: tuple[str, ...] = _INPUTS
    coverage_tolerance: float = 0.02
    abstention_tolerance: float = 0.02
    norm_tolerance: float = 0.05
    latency_tolerance: float = 0.10

    def __post_init__(self) -> None:
        categorical = (
            self.routers,
            self.alpha_modes,
            self.layer_modes,
            self.intervention_timings,
            self.controller_inputs,
        )
        tolerances = (
            self.coverage_tolerance,
            self.abstention_tolerance,
            self.norm_tolerance,
            self.latency_tolerance,
        )
        if (
            type(self.vector_counts) is not tuple
            or not self.vector_counts
            or any(type(value) is not int or value <= 0 for value in self.vector_counts)
            or len(set(self.vector_counts)) != len(self.vector_counts)
            or any(
                type(values) is not tuple
                or not values
                or len(set(values)) != len(values)
                or any(type(value) is not str or not value for value in values)
                for values in categorical
            )
            or any(type(value) is not float or not 0 <= value < 1 for value in tolerances)
        ):
            raise DataValidationError("E5 protocol is invalid")

    @property
    def scientific_eligible(self) -> bool:
        return self == E5Protocol()

    def to_dict(self) -> dict[str, Any]:
        return {
            "vector_counts": list(self.vector_counts),
            "routers": list(self.routers),
            "alpha_modes": list(self.alpha_modes),
            "layer_modes": list(self.layer_modes),
            "intervention_timings": list(self.intervention_timings),
            "controller_inputs": list(self.controller_inputs),
            "coverage_tolerance": self.coverage_tolerance,
            "abstention_tolerance": self.abstention_tolerance,
            "norm_tolerance": self.norm_tolerance,
            "latency_tolerance": self.latency_tolerance,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> E5Protocol:
        expected = {
            "vector_counts",
            "routers",
            "alpha_modes",
            "layer_modes",
            "intervention_timings",
            "controller_inputs",
            "coverage_tolerance",
            "abstention_tolerance",
            "norm_tolerance",
            "latency_tolerance",
        }
        if set(value) != expected:
            raise DataValidationError("E5 protocol keys differ")
        return cls(
            vector_counts=tuple(value["vector_counts"]),
            routers=tuple(value["routers"]),
            alpha_modes=tuple(value["alpha_modes"]),
            layer_modes=tuple(value["layer_modes"]),
            intervention_timings=tuple(value["intervention_timings"]),
            controller_inputs=tuple(value["controller_inputs"]),
            coverage_tolerance=value["coverage_tolerance"],
            abstention_tolerance=value["abstention_tolerance"],
            norm_tolerance=value["norm_tolerance"],
            latency_tolerance=value["latency_tolerance"],
        )


def _protocol(value: E5Protocol | None) -> E5Protocol:
    if value is None:
        return E5Protocol()
    if type(value) is not E5Protocol:
        raise DataValidationError("E5 protocol must be an exact E5Protocol")
    return value


@dataclass(frozen=True, slots=True)
class E5AblationSpec:
    vector_count: int
    router: str
    alpha_mode: str
    layer_mode: str
    intervention_timing: str
    controller_input: str

    def __post_init__(self) -> None:
        if (
            type(self.vector_count) is not int
            or self.vector_count <= 0
            or any(
                type(value) is not str
                for value in (
                    self.router,
                    self.alpha_mode,
                    self.layer_mode,
                    self.intervention_timing,
                    self.controller_input,
                )
            )
            or self.router not in _ROUTERS
            or self.alpha_mode not in _ALPHA_MODES
            or self.layer_mode not in _LAYER_MODES
            or self.intervention_timing not in _TIMINGS
            or self.controller_input not in _INPUTS
        ):
            raise DataValidationError("E5 ablation specification is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "vector_count": self.vector_count,
            "router": self.router,
            "alpha_mode": self.alpha_mode,
            "layer_mode": self.layer_mode,
            "intervention_timing": self.intervention_timing,
            "controller_input": self.controller_input,
        }

    @property
    def spec_id(self) -> str:
        return stable_hash(self.to_dict())


def build_e5_ablation_grid(
    protocol: E5Protocol | None = None,
) -> tuple[E5AblationSpec, ...]:
    frozen = _protocol(protocol)
    return tuple(
        E5AblationSpec(count, router, alpha, layer, timing, inputs)
        for count in frozen.vector_counts
        for router in frozen.routers
        for alpha in frozen.alpha_modes
        for layer in frozen.layer_modes
        for timing in frozen.intervention_timings
        for inputs in frozen.controller_inputs
    )


_ROUTER_ENUM = {
    "nearest_centroid": RouterKind.NEAREST_CENTROID,
    "linear_softmax": RouterKind.LINEAR_SOFTMAX,
    "two_layer_mlp": RouterKind.TWO_LAYER_MLP,
}
_ALPHA_ENUM = {
    "fixed": AlphaMode.FIXED,
    "risk_gated": AlphaMode.RISK_GATED,
    "risk_gated_hard_threshold": AlphaMode.HARD_THRESHOLD,
}
_COMPOSITION = {
    "one_layer": FeatureComposition.SINGLE_LAYER,
    "concatenated_layers": FeatureComposition.CONCATENATED_LAYERS,
    "layer_differences": FeatureComposition.LAYER_DIFFERENCES,
}
_TOKEN_SCOPE = {
    "final_prompt": TokenScope.FINAL_PROMPT,
    "first_generated": TokenScope.FIRST_GENERATED,
    "first_four_generated": TokenScope.FIRST_FOUR,
}


def _validate_controller(spec: E5AblationSpec, directory: Path) -> AdaptiveController:
    controller = load_adaptive_controller(directory)
    schemas = (
        controller.risk_probe.training_schema,
        controller.risk_probe.calibration_schema,
        controller.vector_bank.feature_schema,
        controller.vector_router.feature_schema,
    )
    layer_count = (
        1
        if controller.fixed_layer is not None
        else len(controller.layer_selector.candidate_layers)
        if controller.layer_selector is not None
        else 0
    )
    expected_layers = {
        "fixed_best": 1,
        "two_layer_router": 2,
        "three_layer_router": 3,
    }[spec.layer_mode]
    if (
        controller.vector_bank.cluster_count != spec.vector_count
        or controller.vector_router.kind is not _ROUTER_ENUM[spec.router]
        or controller.alpha_controller.mode is not _ALPHA_ENUM[spec.alpha_mode]
        or layer_count != expected_layers
        or controller.risk_probe.training_schema.composition
        is not _COMPOSITION[spec.controller_input]
        or any(
            schema.benchmark != "triviaqa"
            or schema.model_repository != _ACTIVE_MODEL["repository"]
            or schema.model_revision != _ACTIVE_MODEL["revision"]
            or schema.runtime is not _ACTIVE_MODEL["runtime"]
            or schema.quantization != _ACTIVE_MODEL["quantization"]
            or schema.prompt_id not in _PROMPTS
            or schema.prompt_sha256 != _PROMPT_HASHES.get(schema.prompt_id)
            for schema in schemas
        )
        or len({schema.split_manifest_digest for schema in schemas}) != 1
        or controller.risk_probe.training_fingerprint
        != controller.vector_router.training_fingerprint
        or len(
            {
                controller.risk_probe.training_fingerprint,
                controller.risk_probe.calibration_fingerprint,
                controller.vector_bank.data_fingerprint,
            }
        )
        != 3
        or controller.risk_probe.training_schema.partition != "T-controller-train"
        or controller.risk_probe.calibration_schema.partition != "T-controller-calibration"
        or controller.vector_bank.feature_schema.partition != "T-steer"
        or controller.vector_router.feature_schema.partition != "T-controller-train"
    ):
        raise FrozenArtifactError("E5 controller differs from its ablation specification")
    return controller


@dataclass(frozen=True, slots=True)
class E5ControllerBinding:
    spec: E5AblationSpec
    controller_directory: str
    controller_artifact_sha256: str
    execution_public_key: str
    fit_provenance_sha256: str
    fit_provenance_digest: str
    binding_digest: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or type(self.spec) is not E5AblationSpec
            or type(self.controller_directory) is not str
            or not Path(self.controller_directory).is_absolute()
            or _SHA256.fullmatch(self.controller_artifact_sha256) is None
            or _SHA256.fullmatch(self.execution_public_key) is None
            or _SHA256.fullmatch(self.fit_provenance_sha256) is None
            or _SHA256.fullmatch(self.fit_provenance_digest) is None
            or self.binding_digest != stable_hash(self._body())
        ):
            raise DataValidationError("E5 controller binding is invalid")

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "spec": self.spec.to_dict(),
            "spec_id": self.spec.spec_id,
            "controller_directory": self.controller_directory,
            "controller_artifact_sha256": self.controller_artifact_sha256,
            "execution_public_key": self.execution_public_key,
            "fit_provenance_sha256": self.fit_provenance_sha256,
            "fit_provenance_digest": self.fit_provenance_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "binding_digest": self.binding_digest}

    def assert_current(self) -> AdaptiveController:
        self.__post_init__()
        directory = Path(self.controller_directory)
        if sha256_path(directory) != self.controller_artifact_sha256:
            raise FrozenArtifactError("E5 adaptive controller changed")
        provenance = _load_e5_fit_provenance(directory, expected_spec_id=self.spec.spec_id)
        if (
            sha256_file(directory / "e5-fit-provenance.json") != self.fit_provenance_sha256
            or provenance["provenance_digest"] != self.fit_provenance_digest
            or provenance["execution_public_key"] != self.execution_public_key
        ):
            raise FrozenArtifactError("E5 fitted-controller provenance changed")
        return _validate_controller(self.spec, directory)


def _load_e5_fit_provenance(directory: str | Path, *, expected_spec_id: str) -> Mapping[str, Any]:
    path = Path(directory) / "e5-fit-provenance.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot load E5 fit provenance: {exc}") from exc
    expected = {
        "schema_version",
        "spec_id",
        "controller_fit_id",
        "capture_attestation_digest",
        "capture_artifact_sha256",
        "capture_plan_identity",
        "capture_shard_chain_head",
        "protocol_sha256",
        "recipe_sha256",
        "runtime_artifact_sha256",
        "e2_probe_bundle_sha256",
        "e3_static_vectors_sha256",
        "e3_construction_sha256",
        "layer_label_receipt_sha256",
        "layer_label_plan_identity",
        "layer_label_chain_head",
        "execution_public_key",
        "risk_probes",
        "provenance_digest",
    }
    if not isinstance(value, dict):
        raise FrozenArtifactError("E5 fit provenance must be a mapping")
    body = dict(value)
    digest = body.pop("provenance_digest", None)
    risk_probes = body.get("risk_probes")
    if (
        set(value) != expected
        or digest != stable_hash(body)
        or body.get("schema_version") != 1
        or body.get("spec_id") != expected_spec_id
        or any(
            type(body.get(name)) is not str or _SHA256.fullmatch(body[name]) is None
            for name in expected
            - {
                "schema_version",
                "spec_id",
                "risk_probes",
                "provenance_digest",
            }
        )
        or not isinstance(risk_probes, dict)
        or not risk_probes
        or any(
            not isinstance(entry, dict)
            or set(entry) != {"artifact_sha256", "object_sha256"}
            or any(
                type(item) is not str or _SHA256.fullmatch(item) is None for item in entry.values()
            )
            for entry in risk_probes.values()
        )
    ):
        raise FrozenArtifactError("E5 fit provenance schema or digest differs")
    return MappingProxyType(value)


def write_e5_controller_binding(
    path: str | Path,
    *,
    spec: E5AblationSpec,
    controller_directory: str | Path,
    execution_public_key: str,
) -> E5ControllerBinding:
    normalized = validate_active_study_artifact_paths(
        {"E5 controller binding": path, "E5 controller": controller_directory}
    )
    path = normalized["E5 controller binding"]
    controller_directory = normalized["E5 controller"]
    if type(spec) is not E5AblationSpec:
        raise DataValidationError("E5 binding requires an exact ablation specification")
    directory = Path(controller_directory).resolve()
    _validate_controller(spec, directory)
    provenance = _load_e5_fit_provenance(directory, expected_spec_id=spec.spec_id)
    if provenance["execution_public_key"] != execution_public_key:
        raise DataValidationError("E5 binding key differs from signed fit provenance")
    fingerprint = sha256_path(directory)
    body = {
        "schema_version": 1,
        "spec": spec.to_dict(),
        "spec_id": spec.spec_id,
        "controller_directory": str(directory),
        "controller_artifact_sha256": fingerprint,
        "execution_public_key": execution_public_key,
        "fit_provenance_sha256": sha256_file(directory / "e5-fit-provenance.json"),
        "fit_provenance_digest": provenance["provenance_digest"],
    }
    binding = E5ControllerBinding(
        spec=spec,
        controller_directory=str(directory),
        controller_artifact_sha256=fingerprint,
        execution_public_key=execution_public_key,
        fit_provenance_sha256=body["fit_provenance_sha256"],
        fit_provenance_digest=body["fit_provenance_digest"],
        binding_digest=stable_hash(body),
    )
    _atomic_json(path, binding.to_dict(), "E5 controller binding")
    return binding


def load_e5_controller_binding(path: str | Path) -> E5ControllerBinding:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
        spec = E5AblationSpec(**value["spec"])
        if value.get("spec_id") != spec.spec_id:
            raise DataValidationError("E5 binding specification digest differs")
        binding = E5ControllerBinding(
            schema_version=value["schema_version"],
            spec=spec,
            controller_directory=value["controller_directory"],
            controller_artifact_sha256=value["controller_artifact_sha256"],
            execution_public_key=value["execution_public_key"],
            fit_provenance_sha256=value["fit_provenance_sha256"],
            fit_provenance_digest=value["fit_provenance_digest"],
            binding_digest=value["binding_digest"],
        )
    except (
        OSError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        DataValidationError,
    ) as exc:
        raise FrozenArtifactError(f"cannot load E5 controller binding: {exc}") from exc
    binding.assert_current()
    expected = json.dumps(binding.to_dict(), indent=2, sort_keys=True) + "\n"
    if source.is_symlink() or source.read_text(encoding="utf-8") != expected:
        raise FrozenArtifactError("E5 controller binding differs from exact replay")
    return binding


def controller_artifact_sha256(directory: str | Path) -> str:
    source = Path(directory)
    if source.is_symlink() or not source.is_dir():
        raise FrozenArtifactError("E5 controller artifact must be a regular directory")
    load_adaptive_controller(source)
    return sha256_path(source)


@dataclass(frozen=True, slots=True)
class E5AblationRecord:
    arm_id: str
    prompt_id: str
    question_id: str
    outcome: Outcome
    generation_latency_seconds: float
    intervention_norm: float
    prompt_template_sha256: str
    prompt_input_sha256: str
    rendered_prompt_sha256: str
    output_tokens: int
    controller_binding_sha256: str | None
    token_scope: TokenScope
    execution_receipt: Mapping[str, Any]
    execution_receipt_digest: str
    execution_receipt_signature: str | None

    def __post_init__(self) -> None:
        receipt = dict(self.execution_receipt)
        adaptive = self.arm_id != "M1"
        if (
            (self.arm_id != "M1" and _SHA256.fullmatch(self.arm_id) is None)
            or self.prompt_id not in _PROMPTS
            or type(self.question_id) is not str
            or not self.question_id
            or type(self.outcome) is not Outcome
            or self.outcome not in {Outcome.CORRECT, Outcome.INCORRECT, Outcome.ABSTENTION}
            or type(self.generation_latency_seconds) is not float
            or not math.isfinite(self.generation_latency_seconds)
            or self.generation_latency_seconds < 0
            or type(self.intervention_norm) is not float
            or not math.isfinite(self.intervention_norm)
            or self.intervention_norm < 0
            or self.prompt_template_sha256 != _PROMPT_HASHES.get(self.prompt_id)
            or _SHA256.fullmatch(self.prompt_input_sha256) is None
            or _SHA256.fullmatch(self.rendered_prompt_sha256) is None
            or type(self.output_tokens) is not int
            or self.output_tokens < 0
            or adaptive != (self.controller_binding_sha256 is not None)
            or (
                self.controller_binding_sha256 is not None
                and _SHA256.fullmatch(self.controller_binding_sha256) is None
            )
            or not isinstance(self.token_scope, TokenScope)
            or type(self.execution_receipt) not in {dict, MappingProxyType}
            or set(receipt)
            != {
                "controller_binding_sha256",
                "controller_artifact_sha256",
                "controller_scores",
                "policy_action",
                "applied_token_indices",
                "activation_delta_norm",
                "decision_digest",
            }
            or _SHA256.fullmatch(self.execution_receipt_digest) is None
            or self.execution_receipt_digest != stable_hash(receipt)
            or self.execution_receipt_signature is None
            or (
                self.execution_receipt_signature is not None
                and re.fullmatch(r"[0-9a-f]{128}", self.execution_receipt_signature) is None
            )
        ):
            raise DataValidationError("E5 ablation record is invalid")
        scores = receipt["controller_scores"]
        action = receipt["policy_action"]
        indices = receipt["applied_token_indices"]
        norm = receipt["activation_delta_norm"]
        expected_indices = (
            [-1]
            if self.token_scope is TokenScope.FINAL_PROMPT
            else list(
                range(
                    min(
                        1 if self.token_scope is TokenScope.FIRST_GENERATED else 4,
                        self.output_tokens,
                    )
                )
            )
        )
        decision_body = {
            "arm_id": self.arm_id,
            "prompt_id": self.prompt_id,
            "question_id": self.question_id,
            "rendered_prompt_sha256": self.rendered_prompt_sha256,
            "prompt_input_sha256": self.prompt_input_sha256,
            "controller_binding_sha256": self.controller_binding_sha256,
            "controller_artifact_sha256": receipt["controller_artifact_sha256"],
            "controller_scores": scores,
            "policy_action": action,
            "token_scope": self.token_scope.value,
            "applied_token_indices": indices,
            "activation_delta_norm": norm,
        }
        if (
            (adaptive and (not isinstance(scores, dict) or set(scores) != {"C", "I", "A"}))
            or (
                adaptive
                and any(
                    isinstance(value, bool)
                    or not isinstance(value, int | float)
                    or not 0 <= float(value) <= 1
                    for value in scores.values()
                )
            )
            or (adaptive and not math.isclose(sum(scores.values()), 1.0, abs_tol=1e-8))
            or (not adaptive and scores != {})
            or action not in {"intervene", "release", "abstain"}
            or not isinstance(indices, list)
            or any(type(value) is not int for value in indices)
            or isinstance(norm, bool)
            or not isinstance(norm, int | float)
            or not math.isfinite(float(norm))
            or float(norm) != self.intervention_norm
            or (
                action == "intervene"
                and (indices != expected_indices or not indices or float(norm) <= 0)
            )
            or (action != "intervene" and (indices != [] or float(norm) != 0.0))
            or (not adaptive and action != "intervene")
            or (adaptive and receipt["controller_binding_sha256"] != self.controller_binding_sha256)
            or (not adaptive and receipt["controller_binding_sha256"] is not None)
            or (adaptive and _SHA256.fullmatch(str(receipt["controller_artifact_sha256"])) is None)
            or (not adaptive and receipt["controller_artifact_sha256"] is not None)
            or receipt["decision_digest"] != stable_hash(decision_body)
        ):
            raise DataValidationError("E5 execution receipt is invalid")
        object.__setattr__(self, "execution_receipt", MappingProxyType(receipt))

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "prompt_id": self.prompt_id,
            "question_id": self.question_id,
            "outcome": self.outcome.value,
            "generation_latency_seconds": self.generation_latency_seconds,
            "intervention_norm": self.intervention_norm,
            "prompt_template_sha256": self.prompt_template_sha256,
            "prompt_input_sha256": self.prompt_input_sha256,
            "rendered_prompt_sha256": self.rendered_prompt_sha256,
            "output_tokens": self.output_tokens,
            "controller_binding_sha256": self.controller_binding_sha256,
            "token_scope": self.token_scope.value,
            "execution_receipt": dict(self.execution_receipt),
            "execution_receipt_digest": self.execution_receipt_digest,
            "execution_receipt_signature": self.execution_receipt_signature,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> E5AblationRecord:
        if set(value) != {
            "arm_id",
            "prompt_id",
            "question_id",
            "outcome",
            "generation_latency_seconds",
            "intervention_norm",
            "prompt_template_sha256",
            "prompt_input_sha256",
            "rendered_prompt_sha256",
            "output_tokens",
            "controller_binding_sha256",
            "token_scope",
            "execution_receipt",
            "execution_receipt_digest",
            "execution_receipt_signature",
        }:
            raise DataValidationError("E5 ablation record keys differ")
        return cls(
            arm_id=value["arm_id"],
            prompt_id=value["prompt_id"],
            question_id=value["question_id"],
            outcome=Outcome(value["outcome"]),
            generation_latency_seconds=value["generation_latency_seconds"],
            intervention_norm=value["intervention_norm"],
            prompt_template_sha256=value["prompt_template_sha256"],
            prompt_input_sha256=value["prompt_input_sha256"],
            rendered_prompt_sha256=value["rendered_prompt_sha256"],
            output_tokens=value["output_tokens"],
            controller_binding_sha256=value["controller_binding_sha256"],
            token_scope=TokenScope(value["token_scope"]),
            execution_receipt=value["execution_receipt"],
            execution_receipt_digest=value["execution_receipt_digest"],
            execution_receipt_signature=value["execution_receipt_signature"],
        )


def e5_ablation_execution_receipt_body(record: E5AblationRecord) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "arm_id": record.arm_id,
        "prompt_id": record.prompt_id,
        "question_id": record.question_id,
        "outcome": record.outcome.value,
        "generation_latency_seconds": record.generation_latency_seconds,
        "intervention_norm": record.intervention_norm,
        "prompt_template_sha256": record.prompt_template_sha256,
        "prompt_input_sha256": record.prompt_input_sha256,
        "rendered_prompt_sha256": record.rendered_prompt_sha256,
        "output_tokens": record.output_tokens,
        "controller_binding_sha256": record.controller_binding_sha256,
        "token_scope": record.token_scope.value,
        "execution_receipt": dict(record.execution_receipt),
        "execution_receipt_digest": record.execution_receipt_digest,
    }


def sign_e5_ablation_execution_receipt(record: E5AblationRecord, *, private_key_hex: str) -> str:
    if type(private_key_hex) is not str or _SHA256.fullmatch(private_key_hex) is None:
        raise DataValidationError("E5 execution private key must be 32-byte hex")
    try:
        private_key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_key_hex))
        signature = private_key.sign(
            canonical_json(e5_ablation_execution_receipt_body(record)).encode()
        )
    except ValueError as exc:
        raise DataValidationError(f"invalid E5 execution private key: {exc}") from exc
    return signature.hex()


def _expected_record_keys(
    *, protocol: E5Protocol, question_ids: Sequence[str]
) -> Iterable[tuple[str, str, str]]:
    for arm_id in ("M1", *(value.spec_id for value in build_e5_ablation_grid(protocol))):
        for prompt_id in _PROMPTS:
            for question_id in question_ids:
                yield arm_id, prompt_id, question_id


def _e5_prompt_input_sha256(question: Question, prompt_id: str) -> str:
    return stable_hash(
        {
            "prompt_id": prompt_id,
            "prompt_template_sha256": _PROMPT_HASHES[prompt_id],
            "question_id": question.question_id,
            "benchmark": question.benchmark,
            "text": question.text,
            "aliases": list(question.aliases),
            "split": question.split,
            "entities": list(question.entities),
            "metadata": dict(question.metadata),
        }
    )


def write_e5_ablation_records(
    path: str | Path,
    records: Iterable[E5AblationRecord],
    *,
    screen: E4ScreenReceipt,
    protocol: E5Protocol | None = None,
) -> str:
    path = validate_active_study_artifact_paths({"E5 ablation records": path})[
        "E5 ablation records"
    ]
    frozen = _protocol(protocol)
    screen.assert_current()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite E5 ablation records: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    sentinel = object()
    questions = {value.question_id: value for value in screen.dev_questions}
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for expected, record in zip_longest(
                _expected_record_keys(
                    protocol=frozen,
                    question_ids=tuple(value.question_id for value in screen.dev_questions),
                ),
                records,
                fillvalue=sentinel,
            ):
                if expected is sentinel or record is sentinel:
                    raise DataValidationError("E5 ablation records do not cover the exact grid")
                assert isinstance(record, E5AblationRecord)
                if (
                    type(record) is not E5AblationRecord
                    or (
                        record.arm_id,
                        record.prompt_id,
                        record.question_id,
                    )
                    != expected
                ):
                    raise DataValidationError("E5 ablation record order or identity differs")
                if record.prompt_input_sha256 != _e5_prompt_input_sha256(
                    questions[record.question_id], record.prompt_id
                ):
                    raise DataValidationError(
                        "E5 ablation record differs from its full prompt input"
                    )
                handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return sha256_file(destination)


@dataclass(frozen=True, slots=True)
class E5Measurement:
    spec_id: str
    controller_artifact_sha256: str
    accuracy: float
    coverage: float
    abstention_rate: float
    hallucination_risk: float
    mean_intervention_norm: float
    mean_latency_seconds: float

    def __post_init__(self) -> None:
        probabilities = (
            self.accuracy,
            self.coverage,
            self.abstention_rate,
            self.hallucination_risk,
        )
        if (
            _SHA256.fullmatch(self.spec_id) is None
            or _SHA256.fullmatch(self.controller_artifact_sha256) is None
            or any(type(value) is not float or not 0 <= value <= 1 for value in probabilities)
            or any(
                type(value) is not float or not math.isfinite(value) or value < 0
                for value in (self.mean_intervention_norm, self.mean_latency_seconds)
            )
            or not math.isclose(self.coverage + self.abstention_rate, 1.0, abs_tol=1e-12)
            or not math.isclose(
                self.accuracy,
                self.coverage * (1.0 - self.hallucination_risk),
                abs_tol=1e-12,
            )
        ):
            raise DataValidationError("E5 ablation measurement is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "controller_artifact_sha256": self.controller_artifact_sha256,
            "accuracy": self.accuracy,
            "coverage": self.coverage,
            "abstention_rate": self.abstention_rate,
            "hallucination_risk": self.hallucination_risk,
            "mean_intervention_norm": self.mean_intervention_norm,
            "mean_latency_seconds": self.mean_latency_seconds,
        }


@dataclass(frozen=True, slots=True)
class E5StaticReference:
    accuracy: float
    coverage: float
    abstention_rate: float
    hallucination_risk: float
    mean_intervention_norm: float
    mean_latency_seconds: float

    def __post_init__(self) -> None:
        E5Measurement(
            spec_id="0" * 64,
            controller_artifact_sha256="0" * 64,
            accuracy=self.accuracy,
            coverage=self.coverage,
            abstention_rate=self.abstention_rate,
            hallucination_risk=self.hallucination_risk,
            mean_intervention_norm=self.mean_intervention_norm,
            mean_latency_seconds=self.mean_latency_seconds,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "coverage": self.coverage,
            "abstention_rate": self.abstention_rate,
            "hallucination_risk": self.hallucination_risk,
            "mean_intervention_norm": self.mean_intervention_norm,
            "mean_latency_seconds": self.mean_latency_seconds,
        }


@dataclass(slots=True)
class _Aggregate:
    total: int = 0
    correct: int = 0
    incorrect: int = 0
    abstained: int = 0
    norm_sum: float = 0.0
    latency_sum: float = 0.0

    def add(self, record: E5AblationRecord) -> None:
        self.total += 1
        self.correct += int(record.outcome is Outcome.CORRECT)
        self.incorrect += int(record.outcome is Outcome.INCORRECT)
        self.abstained += int(record.outcome is Outcome.ABSTENTION)
        self.norm_sum += record.intervention_norm
        self.latency_sum += record.generation_latency_seconds

    def metrics(self) -> tuple[float, float, float, float, float, float]:
        attempted = self.correct + self.incorrect
        if self.total <= 0 or attempted <= 0 or attempted + self.abstained != self.total:
            raise DataValidationError("E5 aggregate has no attempted or exact C/I/A rows")
        return (
            self.correct / self.total,
            attempted / self.total,
            self.abstained / self.total,
            self.incorrect / attempted,
            self.norm_sum / self.total,
            self.latency_sum / self.total,
        )


def _read_aggregates(
    path: Path,
    *,
    protocol: E5Protocol,
    questions: tuple[Question, ...],
    bindings: Mapping[str, E5ControllerBinding],
    binding_fingerprints: Mapping[str, str],
) -> tuple[_Aggregate, Mapping[str, _Aggregate]]:
    static = _Aggregate()
    question_ids = tuple(value.question_id for value in questions)
    questions_by_id = {value.question_id: value for value in questions}
    grid = build_e5_ablation_grid(protocol)
    adaptive = {value.spec_id: _Aggregate() for value in grid}
    execution_keys = {binding.execution_public_key for binding in bindings.values()}
    if len(execution_keys) != 1:
        raise DataValidationError("E5 controller bindings do not share one execution trust root")
    execution_public_key = next(iter(execution_keys))
    sentinel = object()
    try:
        with path.open("r", encoding="utf-8") as handle:
            lines = (line for line in handle)
            for expected, line in zip_longest(
                _expected_record_keys(protocol=protocol, question_ids=question_ids),
                lines,
                fillvalue=sentinel,
            ):
                if expected is sentinel or line is sentinel:
                    raise DataValidationError("E5 ablation record count differs")
                assert isinstance(line, str)
                try:
                    value = json.loads(line)
                    if not isinstance(value, Mapping):
                        raise TypeError("record is not an object")
                    if line != json.dumps(dict(value), sort_keys=True) + "\n":
                        raise TypeError("record is not canonical JSONL")
                    record = E5AblationRecord.from_dict(value)
                except (json.JSONDecodeError, TypeError, ValueError, DataValidationError) as exc:
                    raise DataValidationError(f"invalid E5 ablation record: {exc}") from exc
                if (record.arm_id, record.prompt_id, record.question_id) != expected:
                    raise DataValidationError("E5 ablation record order or identity differs")
                if record.prompt_input_sha256 != _e5_prompt_input_sha256(
                    questions_by_id[record.question_id], record.prompt_id
                ):
                    raise DataValidationError(
                        "E5 ablation record differs from its full prompt input"
                    )
                if record.arm_id != "M1":
                    binding = bindings[record.arm_id]
                    if (
                        record.controller_binding_sha256 != binding_fingerprints[record.arm_id]
                        or record.token_scope is not _TOKEN_SCOPE[binding.spec.intervention_timing]
                        or record.execution_receipt["controller_artifact_sha256"]
                        != binding.controller_artifact_sha256
                    ):
                        raise DataValidationError(
                            "E5 execution record differs from its controller binding"
                        )
                assert record.execution_receipt_signature is not None
                try:
                    public_key = Ed25519PublicKey.from_public_bytes(
                        bytes.fromhex(execution_public_key)
                    )
                    public_key.verify(
                        bytes.fromhex(record.execution_receipt_signature),
                        canonical_json(e5_ablation_execution_receipt_body(record)).encode(),
                    )
                except (InvalidSignature, ValueError) as exc:
                    raise DataValidationError(
                        "E5 execution receipt lacks the frozen runtime signature"
                    ) from exc
                (static if record.arm_id == "M1" else adaptive[record.arm_id]).add(record)
    except OSError as exc:
        raise FrozenArtifactError(f"cannot read E5 ablation records: {exc}") from exc
    return static, MappingProxyType(adaptive)


def _absolute_delta(value: float, reference: float) -> float:
    return abs(value - reference)


def _selection_values(
    measurements: tuple[E5Measurement, ...],
    static_reference: E5StaticReference,
    protocol: E5Protocol,
) -> tuple[dict[str, str], E5Measurement | None]:
    matched = {
        "coverage": min(
            measurements,
            key=lambda value: (
                _absolute_delta(value.coverage, static_reference.coverage),
                value.hallucination_risk,
                value.spec_id,
            ),
        ).spec_id,
        "abstention_rate": min(
            measurements,
            key=lambda value: (
                _absolute_delta(value.abstention_rate, static_reference.abstention_rate),
                value.hallucination_risk,
                value.spec_id,
            ),
        ).spec_id,
        "intervention_norm": min(
            measurements,
            key=lambda value: (
                _absolute_delta(
                    value.mean_intervention_norm,
                    static_reference.mean_intervention_norm,
                ),
                value.hallucination_risk,
                value.spec_id,
            ),
        ).spec_id,
        "latency": min(
            measurements,
            key=lambda value: (
                _absolute_delta(
                    value.mean_latency_seconds,
                    static_reference.mean_latency_seconds,
                ),
                value.hallucination_risk,
                value.spec_id,
            ),
        ).spec_id,
    }
    specs = {value.spec_id: value for value in build_e5_ablation_grid(protocol)}
    eligible = [
        value
        for value in measurements
        if _absolute_delta(value.coverage, static_reference.coverage) <= protocol.coverage_tolerance
        and _absolute_delta(value.abstention_rate, static_reference.abstention_rate)
        <= protocol.abstention_tolerance
        and _absolute_delta(
            value.mean_intervention_norm,
            static_reference.mean_intervention_norm,
        )
        <= protocol.norm_tolerance
        and _absolute_delta(
            value.mean_latency_seconds,
            static_reference.mean_latency_seconds,
        )
        <= protocol.latency_tolerance
        and not (
            specs[value.spec_id].vector_count == 1
            and specs[value.spec_id].alpha_mode == "fixed"
            and specs[value.spec_id].layer_mode == "fixed_best"
        )
    ]
    selected = min(
        eligible,
        key=lambda value: (
            value.hallucination_risk,
            -value.accuracy,
            value.controller_artifact_sha256,
            value.spec_id,
        ),
        default=None,
    )
    return matched, selected


@dataclass(frozen=True, slots=True)
class E5Selection:
    protocol: E5Protocol
    screen_receipt_path: str
    screen_receipt_sha256: str
    record_artifact_path: str
    record_set_digest: str
    upstream_paths: Mapping[str, str]
    upstream_digests: Mapping[str, str]
    controller_binding_paths: Mapping[str, str]
    controller_binding_fingerprints: Mapping[str, str]
    source_plan_identity: str
    selection_rule_sha256: str
    static_reference: E5StaticReference
    measurements: tuple[E5Measurement, ...]
    matched_spec_ids: Mapping[str, str]
    selected_spec_id: str | None
    falsification_reason: str | None
    scientific_eligible: bool
    selection_digest: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.protocol) is not E5Protocol:
            raise DataValidationError("E5 selection requires an exact E5Protocol")
        grid_ids = tuple(value.spec_id for value in build_e5_ablation_grid(self.protocol))
        measured_ids = tuple(value.spec_id for value in self.measurements)
        matched, selected = _selection_values(
            self.measurements, self.static_reference, self.protocol
        )
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or type(self.measurements) is not tuple
            or any(type(value) is not E5Measurement for value in self.measurements)
            or any(
                _SHA256.fullmatch(value) is None
                for value in (
                    self.screen_receipt_sha256,
                    self.record_set_digest,
                    self.source_plan_identity,
                    self.selection_rule_sha256,
                    self.selection_digest,
                )
            )
            or not Path(self.screen_receipt_path).is_absolute()
            or not Path(self.record_artifact_path).is_absolute()
            or set(self.upstream_paths) != _UPSTREAM
            or set(self.upstream_digests) != _UPSTREAM
            or any(_SHA256.fullmatch(value) is None for value in self.upstream_digests.values())
            or set(self.controller_binding_paths) != set(grid_ids)
            or any(
                type(value) is not str or not Path(value).is_absolute()
                for value in self.controller_binding_paths.values()
            )
            or set(self.controller_binding_fingerprints) != set(grid_ids)
            or any(
                _SHA256.fullmatch(value) is None
                for value in self.controller_binding_fingerprints.values()
            )
            or measured_ids != grid_ids
            or dict(self.matched_spec_ids) != matched
            or self.selected_spec_id != (selected.spec_id if selected is not None else None)
            or (self.selected_spec_id is None) != (self.falsification_reason is not None)
            or type(self.scientific_eligible) is not bool
            or self.selection_digest != stable_hash(self._body())
        ):
            raise DataValidationError("E5 selection is invalid")
        object.__setattr__(self, "upstream_paths", MappingProxyType(dict(self.upstream_paths)))
        object.__setattr__(self, "upstream_digests", MappingProxyType(dict(self.upstream_digests)))
        object.__setattr__(
            self,
            "controller_binding_paths",
            MappingProxyType(dict(self.controller_binding_paths)),
        )
        object.__setattr__(
            self,
            "controller_binding_fingerprints",
            MappingProxyType(dict(self.controller_binding_fingerprints)),
        )
        object.__setattr__(self, "matched_spec_ids", MappingProxyType(dict(self.matched_spec_ids)))

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol": self.protocol.to_dict(),
            "screen_receipt_path": self.screen_receipt_path,
            "screen_receipt_sha256": self.screen_receipt_sha256,
            "record_artifact_path": self.record_artifact_path,
            "record_set_digest": self.record_set_digest,
            "upstream_paths": dict(self.upstream_paths),
            "upstream_digests": dict(self.upstream_digests),
            "controller_binding_paths": dict(self.controller_binding_paths),
            "controller_binding_fingerprints": dict(self.controller_binding_fingerprints),
            "source_plan_identity": self.source_plan_identity,
            "selection_rule_sha256": self.selection_rule_sha256,
            "static_reference": self.static_reference.to_dict(),
            "measurements": [value.to_dict() for value in self.measurements],
            "matched_spec_ids": dict(self.matched_spec_ids),
            "selected_spec_id": self.selected_spec_id,
            "falsification_reason": self.falsification_reason,
            "scientific_eligible": self.scientific_eligible,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "selection_digest": self.selection_digest}


def derive_e5_selection(
    *,
    screen_receipt_path: str | Path,
    record_artifact_path: str | Path,
    upstream_artifacts: Mapping[str, str | Path],
    controller_binding_artifacts: Mapping[str, str | Path],
    protocol: E5Protocol | None = None,
) -> E5Selection:
    frozen = _protocol(protocol)
    screen_path = Path(screen_receipt_path).resolve()
    record_path = Path(record_artifact_path).resolve()
    screen = load_e4_screen_receipt(screen_path)
    grid = build_e5_ablation_grid(frozen)
    grid_ids = tuple(value.spec_id for value in grid)
    if set(upstream_artifacts) != _UPSTREAM or set(controller_binding_artifacts) != set(grid_ids):
        raise DataValidationError("E5 source artifact inventory differs")
    upstream_paths = {key: str(Path(value).resolve()) for key, value in upstream_artifacts.items()}
    upstream_digests = {key: sha256_path(value) for key, value in upstream_paths.items()}
    binding_paths = {
        key: str(Path(value).resolve()) for key, value in controller_binding_artifacts.items()
    }
    bindings = {key: load_e5_controller_binding(value) for key, value in binding_paths.items()}
    if any(binding.spec.spec_id != key for key, binding in bindings.items()):
        raise DataValidationError("E5 controller bindings differ from grid identities")
    binding_fingerprints = {key: sha256_file(value) for key, value in binding_paths.items()}
    static_aggregate, adaptive_aggregates = _read_aggregates(
        record_path,
        protocol=frozen,
        questions=screen.dev_questions,
        bindings=bindings,
        binding_fingerprints=binding_fingerprints,
    )
    static = E5StaticReference(*static_aggregate.metrics())
    measurements = tuple(
        E5Measurement(
            spec_id=spec.spec_id,
            controller_artifact_sha256=bindings[spec.spec_id].controller_artifact_sha256,
            **dict(
                zip(
                    (
                        "accuracy",
                        "coverage",
                        "abstention_rate",
                        "hallucination_risk",
                        "mean_intervention_norm",
                        "mean_latency_seconds",
                    ),
                    adaptive_aggregates[spec.spec_id].metrics(),
                    strict=True,
                )
            ),
        )
        for spec in grid
    )
    matched, selected = _selection_values(measurements, static, frozen)
    record_digest = sha256_file(record_path)
    screen_fingerprint = sha256_file(screen_path)
    rule_sha = sha256_file(Path(__file__))
    source_plan_identity = stable_hash(
        {
            "protocol": frozen.to_dict(),
            "screen_receipt_sha256": screen_fingerprint,
            "record_set_digest": record_digest,
            "upstream_digests": upstream_digests,
            "controller_binding_fingerprints": binding_fingerprints,
            "selection_rule": _SELECTION_RULE,
            "selection_rule_sha256": rule_sha,
        }
    )
    # The ablation artifact is developmental. Scientific eligibility is granted only
    # by the final E5 PhaseRunLedger after it re-verifies E2/E3/E4 prerequisites and
    # the four registered matched-budget gates.
    scientific = False
    body: dict[str, Any] = {
        "schema_version": 1,
        "protocol": frozen.to_dict(),
        "screen_receipt_path": str(screen_path),
        "screen_receipt_sha256": screen_fingerprint,
        "record_artifact_path": str(record_path),
        "record_set_digest": record_digest,
        "upstream_paths": upstream_paths,
        "upstream_digests": upstream_digests,
        "controller_binding_paths": binding_paths,
        "controller_binding_fingerprints": binding_fingerprints,
        "source_plan_identity": source_plan_identity,
        "selection_rule_sha256": rule_sha,
        "static_reference": static.to_dict(),
        "measurements": [value.to_dict() for value in measurements],
        "matched_spec_ids": matched,
        "selected_spec_id": selected.spec_id if selected is not None else None,
        "falsification_reason": (
            None
            if selected is not None
            else "no-controller-matches-coverage-abstention-norm-and-latency"
        ),
        "scientific_eligible": scientific,
    }
    return E5Selection(
        protocol=frozen,
        screen_receipt_path=str(screen_path),
        screen_receipt_sha256=screen_fingerprint,
        record_artifact_path=str(record_path),
        record_set_digest=record_digest,
        upstream_paths=upstream_paths,
        upstream_digests=upstream_digests,
        controller_binding_paths=binding_paths,
        controller_binding_fingerprints=binding_fingerprints,
        source_plan_identity=source_plan_identity,
        selection_rule_sha256=rule_sha,
        static_reference=static,
        measurements=measurements,
        matched_spec_ids=matched,
        selected_spec_id=body["selected_spec_id"],
        falsification_reason=body["falsification_reason"],
        scientific_eligible=scientific,
        selection_digest=stable_hash(body),
    )


def write_e5_selection(path: str | Path, selection: E5Selection) -> None:
    path = validate_active_study_artifact_paths({"E5 selection": path})["E5 selection"]
    selection.__post_init__()
    replayed = derive_e5_selection(
        screen_receipt_path=selection.screen_receipt_path,
        record_artifact_path=selection.record_artifact_path,
        upstream_artifacts=selection.upstream_paths,
        controller_binding_artifacts=selection.controller_binding_paths,
        protocol=selection.protocol,
    )
    if replayed != selection:
        raise FrozenArtifactError("E5 selection differs from current source replay")
    _atomic_json(path, selection.to_dict(), "E5 selection")


def verify_e5_selection(path: str | Path) -> Mapping[str, Any]:
    replayed = load_e5_selection(path)
    return MappingProxyType(
        {
            "valid": True,
            "selection_digest": replayed.selection_digest,
            "artifact_sha256": sha256_file(path),
            "selected_spec_id": replayed.selected_spec_id,
            "scientific_eligible": replayed.scientific_eligible,
        }
    )


def load_e5_selection(path: str | Path) -> E5Selection:
    """Replay and return an immutable E5 developmental selection artifact."""

    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
        protocol = E5Protocol.from_dict(value["protocol"])
        replayed = derive_e5_selection(
            screen_receipt_path=value["screen_receipt_path"],
            record_artifact_path=value["record_artifact_path"],
            upstream_artifacts=value["upstream_paths"],
            controller_binding_artifacts=value["controller_binding_paths"],
            protocol=protocol,
        )
    except (
        OSError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        DataValidationError,
        FrozenArtifactError,
    ) as exc:
        raise FrozenArtifactError(f"cannot replay E5 selection: {exc}") from exc
    expected = json.dumps(replayed.to_dict(), indent=2, sort_keys=True) + "\n"
    if source.is_symlink() or source.read_text(encoding="utf-8") != expected:
        raise FrozenArtifactError("E5 selection differs from exact source replay")
    return replayed


def _e5_paired_rows(ledger: PhaseRunLedger) -> tuple[dict[str, str], ...]:
    strata: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    for condition in ledger.contract.conditions:
        key = (
            condition.model_repository,
            condition.benchmark,
            condition.system_prompt_id,
            condition.partition,
            condition.comparison_group,
        )
        methods = strata.setdefault(key, {})
        if condition.steering_method in methods:
            raise DataValidationError("E5 final ledger repeats a method within a stratum")
        methods[condition.steering_method] = condition.condition_id
    if any(set(methods) != {"M1", "M3"} for methods in strata.values()):
        raise DataValidationError("E5 final ledger lacks an exact M1/M3 pair")
    rows: list[dict[str, str]] = []
    questions = ledger.contract.question_ids_by_benchmark["triviaqa"]
    for key in sorted(strata):
        methods = strata[key]
        rows.extend(
            {
                "question_id": question_id,
                "baseline_condition_id": methods["M1"],
                "intervention_condition_id": methods["M3"],
            }
            for question_id in questions
        )
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class _E5FinalInputs:
    selection: E5Selection
    screen: E4ScreenReceipt
    ledger: PhaseRunLedger
    selected_spec: E5AblationSpec
    selected_measurement: E5Measurement
    selected_binding: E5ControllerBinding
    selected_controller: AdaptiveController


def _write_e5_selected_controller_bundle(
    destination: Path,
    *,
    inputs: _E5FinalInputs,
) -> str:
    """Package the promoted E5 controller without retaining an absolute-path binding."""

    if destination.exists() or destination.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite selected E5 controller: {destination}")
    source = Path(inputs.selected_binding.controller_directory)
    controller_sha = inputs.selected_binding.controller_artifact_sha256
    assert inputs.selection.selected_spec_id is not None
    binding_sha = inputs.selection.controller_binding_fingerprints[
        inputs.selection.selected_spec_id
    ]
    destination.mkdir()
    _copy_frozen_artifact(source, destination / "controller", controller_sha)
    body = {
        "schema_version": 1,
        "selection_digest": inputs.selection.selection_digest,
        "selected_spec_id": inputs.selection.selected_spec_id,
        "binding_sha256": binding_sha,
        "controller_artifact_sha256": controller_sha,
        "execution_public_key": inputs.selected_binding.execution_public_key,
        "spec": inputs.selected_spec.to_dict(),
    }
    (destination / "manifest.json").write_text(
        json.dumps(
            {**body, "manifest_digest": stable_hash(body)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    validate_e5_selected_controller_bundle(destination)
    return sha256_path(destination)


def validate_e5_selected_controller_bundle(
    directory: str | Path,
) -> Mapping[str, Any]:
    """Replay one portable promoted E5 controller and return its frozen identities."""

    source = Path(directory)
    if (
        source.is_symlink()
        or not source.is_dir()
        or {value.name for value in source.iterdir()} != {"manifest.json", "controller"}
        or any(value.is_symlink() for value in source.rglob("*"))
    ):
        raise FrozenArtifactError("selected E5 controller inventory differs")
    try:
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        body = dict(manifest)
        digest = body.pop("manifest_digest")
    except (OSError, json.JSONDecodeError, TypeError, KeyError) as exc:
        raise FrozenArtifactError(f"cannot read selected E5 controller manifest: {exc}") from exc
    expected_keys = {
        "schema_version",
        "selection_digest",
        "selected_spec_id",
        "binding_sha256",
        "controller_artifact_sha256",
        "execution_public_key",
        "spec",
    }
    try:
        spec_value = body.get("spec")
        spec = E5AblationSpec(**spec_value) if isinstance(spec_value, dict) else None
        controller = load_adaptive_controller(source / "controller")
    except (TypeError, ValueError, DataValidationError, FrozenArtifactError) as exc:
        raise FrozenArtifactError(f"selected E5 controller cannot be loaded: {exc}") from exc
    if (
        set(body) != expected_keys
        or body.get("schema_version") != 1
        or digest != stable_hash(body)
        or spec is None
        or body.get("selected_spec_id") != spec.spec_id
        or any(
            type(body.get(name)) is not str or _SHA256.fullmatch(str(body[name])) is None
            for name in (
                "selection_digest",
                "binding_sha256",
                "controller_artifact_sha256",
                "execution_public_key",
            )
        )
        or sha256_path(source / "controller") != body.get("controller_artifact_sha256")
    ):
        raise FrozenArtifactError("selected E5 controller identity differs")
    _validate_controller(spec, source / "controller")
    return MappingProxyType(
        {
            **body,
            "manifest_digest": digest,
            "controller": controller,
            "controller_path": source / "controller",
            "bundle_sha256": sha256_path(source),
        }
    )


def _e5_verified_e2_probe(
    path: str | Path, *, composition: FeatureComposition
) -> tuple[CalibratedProbe, str, str]:
    source = Path(path)
    try:
        plan = json.loads((source / "plan.json").read_text(encoding="utf-8"))
        results = json.loads((source / "results.json").read_text(encoding="utf-8"))
        row = next(
            value
            for value in results["controller_input_probes"]
            if value["controller_input"] == composition.value
        )
        artifact = source / "controller-input-probes" / row["artifact"]
        if sha256_path(artifact) != row["artifact_sha256"]:
            raise FrozenArtifactError("E5 E2 controller-input probe artifact changed")
        probe = load_calibrated_probe(artifact)
        split_digest = plan["split_manifest_digest"]
    except (
        OSError,
        json.JSONDecodeError,
        KeyError,
        StopIteration,
        TypeError,
        ValueError,
        DataValidationError,
    ) as exc:
        raise FrozenArtifactError(
            f"cannot bind E5 to its E2 controller-input probe: {exc}"
        ) from exc
    if (
        probe.task.value != "correct_incorrect_abstention"
        or probe.training_schema.composition is not composition
        or probe.calibration_schema.composition is not composition
        or type(split_digest) is not str
        or _SHA256.fullmatch(split_digest) is None
        or probe.training_schema.split_manifest_digest != split_digest
        or probe.calibration_schema.split_manifest_digest != split_digest
    ):
        raise FrozenArtifactError("E5 E2 controller-input probe identity differs")
    return probe, split_digest, row["artifact_sha256"]


def _e5_verified_e3_data_fingerprint(path: str | Path) -> str:
    try:
        metadata = json.loads((Path(path) / "metadata.json").read_text(encoding="utf-8"))
        body = dict(metadata)
        digest = body.pop("metadata_digest")
        fingerprint = body["data_fingerprint"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise FrozenArtifactError(f"cannot bind E5 to the E3 vector bundle: {exc}") from exc
    if (
        digest != stable_hash(body)
        or type(fingerprint) is not str
        or _SHA256.fullmatch(fingerprint) is None
        or body.get("phase") != "E3-construction"
        or body.get("scientific_eligible") is not True
    ):
        raise FrozenArtifactError("E5 E3 vector metadata identity differs")
    return fingerprint


def _e5_prerequisite_ledgers(
    ledger: PhaseRunLedger, *, study: StudyProtocol
) -> Mapping[ExperimentPhase, Any]:
    PhaseRunLedger._verify_creation_evidence(ledger)
    try:
        payload = json.loads(
            (ledger.directory / "creation-evidence.json").read_text(encoding="utf-8")
        )
        descriptors = payload["prerequisite_runs"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise FrozenArtifactError(f"cannot read E5 prerequisite evidence: {exc}") from exc
    if not isinstance(descriptors, Mapping) or set(descriptors) != {"E2", "E3", "E4"}:
        raise FrozenArtifactError("E5 prerequisite evidence inventory differs")
    values: dict[ExperimentPhase, Any] = {}
    for phase in (ExperimentPhase.E2, ExperimentPhase.E3, ExperimentPhase.E4):
        descriptor = descriptors[phase.value]
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
            "location",
            "completion_digest",
        }:
            raise FrozenArtifactError("E5 prerequisite descriptor differs")
        prior = open_phase_prerequisite(
            _resolve_ledger_evidence_path(
                ledger.directory,
                descriptor["location"],
                context=f"E5 {phase.value} prerequisite",
            ),
            phase=phase,
            study=study,
            expected_completion_digest=str(descriptor["completion_digest"]),
        )
        completion = prior.verify_complete()
        if (
            completion.phase is not phase
            or descriptor["completion_digest"] != completion.completion_digest
            or ledger.contract.prerequisite_digests[phase.value] != completion.completion_digest
        ):
            raise FrozenArtifactError("E5 prerequisite completion identity differs")
        values[phase] = prior
    return MappingProxyType(values)


def _validate_e5_final_inputs(
    *,
    ledger_directory: str | Path,
    study: StudyProtocol,
    selection_path: str | Path,
) -> _E5FinalInputs:
    """Replay every scientific invariant shared by finalization and verification."""

    if type(study) is not StudyProtocol:
        raise DataValidationError("E5 finalization requires an exact StudyProtocol")
    selection = load_e5_selection(selection_path)
    if (
        selection.protocol != E5Protocol()
        or selection.selected_spec_id is None
        or selection.falsification_reason is not None
    ):
        raise DataValidationError(
            "E5 scientific finalization requires the full grid and one matched selection"
        )
    screen = load_e4_screen_receipt(selection.screen_receipt_path)
    if not screen.scientific_eligible or len(screen.dev_questions) != 5_000:
        raise DataValidationError("E5 finalization requires the exact scientific T-dev cohort")
    ledger = PhaseRunLedger.open(ledger_directory, study=study)
    ledger.contract.assert_matches_study(study)
    completed, expected = PhaseRunLedger.progress(ledger)
    if (
        ledger.contract.phase is not ExperimentPhase.E5
        or completed != expected
        or dict(ledger.contract.input_fingerprints) != dict(selection.upstream_digests)
        or ledger.contract.question_ids_by_benchmark
        != {"triviaqa": tuple(question.question_id for question in screen.dev_questions)}
        or any(condition.partition != "T-dev" for condition in ledger.contract.conditions)
    ):
        raise DataValidationError("E5 final ledger differs from its frozen selection inputs")

    prerequisites = _e5_prerequisite_ledgers(ledger, study=study)
    e2_probe_digest = prerequisites[ExperimentPhase.E2].contract.input_fingerprints.get(
        "activation_feature_schemas"
    )
    e3_static_digest = selection.upstream_digests["E3_static_vectors"]
    e3_output_fingerprints = prerequisites[ExperimentPhase.E3].output_fingerprints
    e4_completion = prerequisites[ExperimentPhase.E4].verify_complete()
    e4_promotion_digests = {
        digest
        for name, digest in e4_completion.gate_artifact_fingerprints.items()
        if name.endswith("/promotion")
    }
    if (
        selection.upstream_digests["E2_calibrated_probes"] != e2_probe_digest
        or e3_output_fingerprints.get("E3_static_vectors") != e3_static_digest
        or selection.upstream_digests["E4_promoted_baselines"] not in e4_promotion_digests
    ):
        raise DataValidationError(
            "E5 upstream components are not outputs of the frozen prerequisites"
        )

    selected_spec = next(
        value
        for value in build_e5_ablation_grid(selection.protocol)
        if value.spec_id == selection.selected_spec_id
    )
    selected_measurement = next(
        value for value in selection.measurements if value.spec_id == selection.selected_spec_id
    )
    selected_binding = load_e5_controller_binding(
        selection.controller_binding_paths[selection.selected_spec_id]
    )
    selected_controller = selected_binding.assert_current()
    selected_e2_probe, split_manifest_digest, selected_e2_artifact_sha256 = _e5_verified_e2_probe(
        selection.upstream_paths["E2_calibrated_probes"],
        composition=_COMPOSITION[selected_spec.controller_input],
    )
    fit_provenance = _load_e5_fit_provenance(
        selected_binding.controller_directory,
        expected_spec_id=selection.selected_spec_id,
    )
    risk_provenance = fit_provenance["risk_probes"].get(
        _COMPOSITION[selected_spec.controller_input].value
    )
    _e5_verified_e3_data_fingerprint(selection.upstream_paths["E3_static_vectors"])
    if (
        selected_controller.risk_probe.training_fingerprint
        != selected_e2_probe.training_fingerprint
        or selected_controller.risk_probe.calibration_fingerprint
        != selected_e2_probe.calibration_fingerprint
        or selected_controller.risk_probe.training_schema != selected_e2_probe.training_schema
        or selected_controller.risk_probe.calibration_schema != selected_e2_probe.calibration_schema
        or selected_controller.vector_router.training_fingerprint
        != selected_e2_probe.training_fingerprint
        or selected_controller.vector_bank.source_artifact_sha256 != e3_static_digest
        or fit_provenance["e2_probe_bundle_sha256"]
        != selection.upstream_digests["E2_calibrated_probes"]
        or fit_provenance["e3_static_vectors_sha256"] != e3_static_digest
        or fit_provenance["execution_public_key"] != selected_binding.execution_public_key
        or not isinstance(risk_provenance, Mapping)
        or risk_provenance.get("artifact_sha256") != selected_e2_artifact_sha256
        or any(
            schema.split_manifest_digest != split_manifest_digest
            for schema in (
                selected_controller.risk_probe.training_schema,
                selected_controller.risk_probe.calibration_schema,
                selected_controller.vector_router.feature_schema,
                selected_controller.vector_bank.feature_schema,
            )
        )
    ):
        raise DataValidationError(
            "E5 controller data identities differ from verified E2/E3 artifacts"
        )
    selected_binding_fingerprint = selection.controller_binding_fingerprints[
        selection.selected_spec_id
    ]
    expected_scope = _TOKEN_SCOPE[selected_spec.intervention_timing]
    expected_layers = (
        (selected_controller.fixed_layer,)
        if selected_controller.fixed_layer is not None
        else selected_controller.layer_selector.candidate_layers
        if selected_controller.layer_selector is not None
        else ()
    )
    expected_sites = tuple(
        sorted(
            {
                key.site
                for key in selected_controller.vector_bank.directions
                if key.layer in expected_layers
            },
            key=lambda value: value.value,
        )
    )
    alpha = selected_controller.alpha_controller
    for condition in ledger.contract.conditions:
        if condition.steering_method == "M1":
            if condition.method_artifact_sha256 != e3_static_digest:
                raise DataValidationError(
                    "E5 M1 condition differs from the selected E3 static vector"
                )
            continue
        policy = condition.adaptive_policy
        if (
            condition.steering_method != "M3"
            or policy is None
            or policy.schema_version != 2
            or condition.method_artifact_sha256 != selected_binding_fingerprint
            or policy.controller_artifact_sha256 != selected_measurement.controller_artifact_sha256
            or policy.controller_artifact_sha256 != selected_binding.controller_artifact_sha256
            or policy.execution_public_key != selected_binding.execution_public_key
            or policy.vector_count != selected_spec.vector_count
            or policy.candidate_token_scopes != (expected_scope,)
            or policy.candidate_layers != expected_layers
            or policy.candidate_sites != expected_sites
            or policy.alpha_mode != alpha.mode.value
            or policy.alpha_max != alpha.alpha_max
            or policy.alpha_beta != alpha.beta
            or policy.alpha_risk_threshold != alpha.threshold
        ):
            raise DataValidationError(
                "E5 M3 condition differs from the selected adaptive controller"
            )
    return _E5FinalInputs(
        selection=selection,
        screen=screen,
        ledger=ledger,
        selected_spec=selected_spec,
        selected_measurement=selected_measurement,
        selected_binding=selected_binding,
        selected_controller=selected_controller,
    )


def finalize_e5_phase(
    destination: str | Path,
    *,
    ledger_directory: str | Path,
    study: StudyProtocol,
    selection_path: str | Path,
) -> Mapping[str, Any]:
    """Replay E5 selection, derive all four gates, and freeze the real phase ledger."""

    normalized = validate_active_study_artifact_paths(
        {
            "E5 finalization": destination,
            "E5 phase ledger": ledger_directory,
            "E5 selection": selection_path,
        }
    )
    output = normalized["E5 finalization"]
    ledger_directory = normalized["E5 phase ledger"]
    selection_path = normalized["E5 selection"]
    if output.exists() or output.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite E5 finalization: {output}")
    inputs = _validate_e5_final_inputs(
        ledger_directory=ledger_directory,
        study=study,
        selection_path=selection_path,
    )
    selection = inputs.selection
    ledger = inputs.ledger
    selected_measurement = inputs.selected_measurement
    rows = _e5_paired_rows(ledger)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        selected_controller_bundle_sha = _write_e5_selected_controller_bundle(
            stage / "selected-controller",
            inputs=inputs,
        )
        gate_results = {}
        for gate in (
            "matched_coverage",
            "matched_abstention",
            "matched_norm",
            "matched_latency",
        ):
            evidence_path = stage / f"{gate}.json"
            write_gate_evidence(
                evidence_path,
                phase=ExperimentPhase.E5,
                gate=gate,
                contract_digest=ledger.contract.digest,
                record_set_digest=PhaseRunLedger.record_set_digest(ledger),
                observations=rows,
            )
            gate_results[gate] = ledger.evaluate_gate(gate, evidence_path)
        (
            status,
            terminal_digest,
            terminal_record_set_digest,
            terminal_gate_result_digests,
        ) = _finalize_or_recover_e5_ledger(ledger, gate_results)
        receipt_body: dict[str, Any] = {
            "schema_version": 2,
            "phase": ExperimentPhase.E5.value,
            "status": status,
            "ledger_directory": str(Path(ledger_directory).resolve()),
            "contract_digest": ledger.contract.digest,
            "record_set_digest": terminal_record_set_digest,
            "selection_path": str(Path(selection_path).resolve()),
            "selection_digest": selection.selection_digest,
            "selected_spec_id": selection.selected_spec_id,
            "selected_controller_artifact_sha256": (
                selected_measurement.controller_artifact_sha256
            ),
            "selected_controller_bundle_sha256": selected_controller_bundle_sha,
            "gate_result_digests": dict(terminal_gate_result_digests),
            "terminal_digest": terminal_digest,
            "scientific_eligible": status == "complete",
        }
        receipt = {**receipt_body, "receipt_digest": stable_hash(receipt_body)}
        (stage / "receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return verify_e5_phase(
        output,
        ledger_directory=ledger_directory,
        study=study,
        selection_path=selection_path,
    )


def _finalize_or_recover_e5_ledger(
    ledger: PhaseRunLedger,
    gate_results: Mapping[str, GateResult],
) -> tuple[str, str, str, Mapping[str, str]]:
    """Create the terminal marker once, or recover it after a wrapper crash."""

    complete_exists = (ledger.directory / "complete.json").exists()
    falsified_exists = (ledger.directory / "falsified.json").exists()
    if complete_exists and falsified_exists:
        raise FrozenArtifactError("E5 ledger has conflicting terminal markers")
    all_passed = all(result.passed for result in gate_results.values())
    expected_gate_digests = {
        name: gate_results[name].gate_digest for name in sorted(gate_results)
    }
    terminal: PhaseCompletion | PhaseFalsification
    if complete_exists:
        if not all_passed:
            raise FrozenArtifactError("completed E5 ledger conflicts with recomputed gates")
        terminal = ledger.verify_complete()
        status = "complete"
        digest = terminal.completion_digest
    elif falsified_exists:
        if all_passed:
            raise FrozenArtifactError("falsified E5 ledger conflicts with recomputed gates")
        terminal = ledger.verify_falsified()
        status = "falsified"
        digest = terminal.falsification_digest
    elif all_passed:
        terminal = ledger.finalize(gate_results)
        status = "complete"
        digest = terminal.completion_digest
    else:
        terminal = ledger.finalize_falsified(gate_results)
        status = "falsified"
        digest = terminal.falsification_digest
    if dict(terminal.gate_result_digests) != expected_gate_digests:
        raise FrozenArtifactError("terminal E5 ledger differs from recomputed gate evidence")
    return (
        status,
        digest,
        terminal.record_set_digest,
        terminal.gate_result_digests,
    )


def verify_e5_phase(
    directory: str | Path,
    *,
    ledger_directory: str | Path,
    study: StudyProtocol,
    selection_path: str | Path,
) -> Mapping[str, Any]:
    """Reopen the terminal E5 ledger and replay its external finalization receipt."""

    source = Path(directory)
    expected_files = {
        "matched_coverage.json",
        "matched_abstention.json",
        "matched_norm.json",
        "matched_latency.json",
        "selected-controller",
        "receipt.json",
    }
    if (
        source.is_symlink()
        or not source.is_dir()
        or {path.name for path in source.iterdir()} != expected_files
        or any(path.is_symlink() for path in source.iterdir())
    ):
        raise FrozenArtifactError("E5 finalization artifact inventory differs")
    try:
        receipt = json.loads((source / "receipt.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot load E5 finalization receipt: {exc}") from exc
    if type(receipt) is not dict:
        raise FrozenArtifactError("E5 finalization receipt must be an object")
    receipt_digest = receipt.pop("receipt_digest", None)
    expected_receipt_keys = {
        "schema_version",
        "phase",
        "status",
        "ledger_directory",
        "contract_digest",
        "record_set_digest",
        "selection_path",
        "selection_digest",
        "selected_spec_id",
        "selected_controller_artifact_sha256",
        "selected_controller_bundle_sha256",
        "gate_result_digests",
        "terminal_digest",
        "scientific_eligible",
    }
    if (
        set(receipt) != expected_receipt_keys
        or type(receipt.get("schema_version")) is not int
        or receipt.get("schema_version") != 2
        or receipt.get("phase") != ExperimentPhase.E5.value
        or receipt_digest != stable_hash(receipt)
        or receipt.get("ledger_directory") != str(Path(ledger_directory).resolve())
        or receipt.get("selection_path") != str(Path(selection_path).resolve())
    ):
        raise FrozenArtifactError("E5 finalization receipt identity differs")
    inputs = _validate_e5_final_inputs(
        ledger_directory=ledger_directory,
        study=study,
        selection_path=selection_path,
    )
    selection = inputs.selection
    ledger = inputs.ledger
    status = receipt["status"]
    try:
        if status == "complete":
            completion = ledger.verify_complete()
            terminal_digest = completion.completion_digest
            terminal_record_set_digest = completion.record_set_digest
            terminal_gate_result_digests = completion.gate_result_digests
            terminal_gate_artifact_fingerprints = completion.gate_artifact_fingerprints
            scientific = True
        elif status == "falsified":
            falsification = ledger.verify_falsified()
            terminal_digest = falsification.falsification_digest
            terminal_record_set_digest = falsification.record_set_digest
            terminal_gate_result_digests = falsification.gate_result_digests
            terminal_gate_artifact_fingerprints = falsification.gate_artifact_fingerprints
            scientific = False
        else:
            raise FrozenArtifactError("E5 finalization status is invalid")
    except (DataValidationError, FrozenArtifactError) as exc:
        raise FrozenArtifactError(f"cannot verify terminal E5 ledger: {exc}") from exc
    selected_measurement = inputs.selected_measurement
    selected_controller = validate_e5_selected_controller_bundle(source / "selected-controller")
    evidence_fingerprints = {
        f"{gate}/evaluation": sha256_file(source / f"{gate}.json")
        for gate in (
            "matched_coverage",
            "matched_abstention",
            "matched_norm",
            "matched_latency",
        )
    }
    if (
        receipt["contract_digest"] != ledger.contract.digest
        or receipt["record_set_digest"] != terminal_record_set_digest
        or receipt["selection_digest"] != selection.selection_digest
        or receipt["selected_spec_id"] != selection.selected_spec_id
        or receipt["selected_controller_artifact_sha256"]
        != selected_measurement.controller_artifact_sha256
        or receipt["selected_controller_bundle_sha256"] != selected_controller["bundle_sha256"]
        or selected_controller["selection_digest"] != selection.selection_digest
        or selected_controller["selected_spec_id"] != selection.selected_spec_id
        or selected_controller["controller_artifact_sha256"]
        != selected_measurement.controller_artifact_sha256
        or receipt["gate_result_digests"] != dict(terminal_gate_result_digests)
        or receipt["terminal_digest"] != terminal_digest
        or receipt["scientific_eligible"] is not scientific
        or evidence_fingerprints != dict(terminal_gate_artifact_fingerprints)
    ):
        raise FrozenArtifactError("E5 finalization differs from terminal ledger replay")
    return MappingProxyType(
        {
            "valid": True,
            "status": status,
            "receipt_digest": receipt_digest,
            "selection_digest": selection.selection_digest,
            "terminal_digest": terminal_digest,
            "scientific_eligible": scientific,
        }
    )

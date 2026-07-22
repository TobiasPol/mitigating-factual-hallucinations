"""Resumable, provenance-bound results for preregistered robustness diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from safetensors.torch import load_file, save_file
from torch import Tensor

from mfh.artifact_namespace import validate_active_study_artifact_paths
from mfh.config import load_prompt_specs
from mfh.contracts import (
    ActivationSite,
    AdaptivePolicySpec,
    GenerationRecord,
    Outcome,
    PromptSpec,
    Question,
    Runtime,
)
from mfh.data.io import read_generation_records, write_generation_records
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.experiments.confirmatory_components import (
    ConfirmatoryAdaptiveComponent,
    ConfirmatoryFixedComponent,
    load_confirmatory_adaptive_component,
    load_confirmatory_fixed_component,
)
from mfh.experiments.confirmatory_graders import (
    ConfirmatoryGraderBundle,
    validate_confirmatory_factual_grade,
    validate_confirmatory_grader_bundle,
)
from mfh.experiments.e8_protected import (
    _validate_e8_adaptive_controller_record,
    question_source_fingerprint,
)
from mfh.experiments.e9_native import NativeE9VllmBackend
from mfh.experiments.protocol import ExperimentPhase
from mfh.experiments.robustness_diagnostics import (
    PromptParaphraseTask,
    RobustnessDiagnosticPlan,
    RQ1GeneralizationTask,
    _question_fingerprint,
    _questions_from_source,
    iter_prompt_paraphrase_tasks,
    iter_rq1_generalization_tasks,
    rq1_task_question_sets,
    verify_robustness_diagnostic_plan,
)
from mfh.experiments.runner import (
    EvaluationCondition,
    validate_confirmatory_execution_receipt,
)
from mfh.inference.architecture import HookKey
from mfh.methods.adaptive import (
    AdaptiveController,
    AlphaController,
    AlphaMode,
    RouterKind,
    assign_to_vector_regions,
    fit_adaptive_router,
    fit_layer_selector,
    fit_routed_vector_bank,
    save_adaptive_controller,
)
from mfh.methods.features import ActivationFeatureSchema
from mfh.methods.probes import (
    CalibrationKind,
    ProbeDataset,
    ProbeKind,
    ProbeTask,
    ProbeTrainingConfig,
    TemperatureCalibrator,
    fit_calibrated_probe,
)
from mfh.provenance import canonical_json, sha256_file, sha256_path, stable_hash

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROMPT_RESULT_FILE = re.compile(r"^prompt-[0-9a-f]{64}\.json$")
_RQ1_RESULT_DIRECTORY = re.compile(r"^rq1-[0-9a-f]{64}$")
_RQ1_ARTIFACT_KEYS = frozenset(
    {
        "source_component",
        "adapted_component",
        "fit_receipt",
        "evaluation_records",
    }
)
_SCOPE_FIELDS = frozenset(
    {
        "vector_bank",
        "router",
        "directions",
        "alpha_controller",
        "risk_threshold",
        "abstention_threshold",
        "router_architecture",
        "candidate_layers",
        "candidate_sites",
        "token_scopes",
        "sparsity",
        "alpha_policy_family",
        "likely_unknown_threshold",
        "execution_public_key",
        "risk_probe",
        "layer_selector",
    }
)
_QWEN_REPOSITORY = "nvidia/Qwen3.6-27B-NVFP4"
_QWEN_REVISION = "0893e1606ff3d5f97a441f405d5fc541a6bdf404"
_QWEN_QUANTIZATION = "modelopt-mixed-nvfp4-fp8"
_RQ1_COMPONENT_FILES = frozenset({"scope-manifest.json", "fields", "execution-component"})


@dataclass(frozen=True, slots=True)
class M3FitRecipe:
    """Every deterministic hyperparameter needed to refit one RQ1 M3 controller."""

    cluster_count: int
    vector_seed: int
    minimum_class_count: int
    vector_source_artifact_sha256: str | None
    router_kind: RouterKind
    router_seed: int
    router_hidden_width: int
    router_epochs: int
    distance_temperature: float
    risk_probe_kind: ProbeKind
    risk_hidden_width: int
    risk_epochs: int
    risk_learning_rate: float
    risk_weight_decay: float
    risk_class_balanced: bool
    risk_seed: int
    calibration_kind: CalibrationKind
    alpha_mode: AlphaMode
    alpha_max: float
    alpha_beta: float
    alpha_threshold: float
    fixed_layer: int | None
    candidate_layers: tuple[int, ...]
    layer_router_kind: RouterKind | None
    layer_seed: int
    layer_epochs: int
    schema_version: int = 1

    def __post_init__(self) -> None:
        try:
            training = self.risk_training_config
            alpha = self.alpha_controller
        except DataValidationError:
            raise
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or type(self.cluster_count) is not int
            or self.cluster_count not in {1, 4, 8, 16}
            or type(self.vector_seed) is not int
            or type(self.minimum_class_count) is not int
            or self.minimum_class_count <= 0
            or (
                self.vector_source_artifact_sha256 is not None
                and _SHA256.fullmatch(self.vector_source_artifact_sha256) is None
            )
            or not isinstance(self.router_kind, RouterKind)
            or type(self.router_seed) is not int
            or type(self.router_hidden_width) is not int
            or self.router_hidden_width <= 0
            or type(self.router_epochs) is not int
            or self.router_epochs <= 0
            or isinstance(self.distance_temperature, bool)
            or not isinstance(self.distance_temperature, int | float)
            or not math.isfinite(float(self.distance_temperature))
            or float(self.distance_temperature) <= 0
            or not isinstance(self.calibration_kind, CalibrationKind)
            or (self.fixed_layer is None) == (not self.candidate_layers)
            or (
                self.fixed_layer is not None
                and (type(self.fixed_layer) is not int or self.fixed_layer < 0)
            )
            or (
                self.candidate_layers
                and (
                    len(self.candidate_layers) not in {2, 3}
                    or len(set(self.candidate_layers)) != len(self.candidate_layers)
                    or any(type(value) is not int or value < 0 for value in self.candidate_layers)
                    or not isinstance(self.layer_router_kind, RouterKind)
                )
            )
            or (not self.candidate_layers and self.layer_router_kind is not None)
            or type(self.layer_seed) is not int
            or type(self.layer_epochs) is not int
            or self.layer_epochs <= 0
            or not isinstance(training, ProbeTrainingConfig)
            or not isinstance(alpha, AlphaController)
        ):
            raise DataValidationError("RQ1 M3 fit recipe is invalid")
        object.__setattr__(self, "distance_temperature", float(self.distance_temperature))

    @property
    def risk_training_config(self) -> ProbeTrainingConfig:
        return ProbeTrainingConfig(
            kind=self.risk_probe_kind,
            hidden_width=self.risk_hidden_width,
            epochs=self.risk_epochs,
            learning_rate=self.risk_learning_rate,
            weight_decay=self.risk_weight_decay,
            class_balanced=self.risk_class_balanced,
            seed=self.risk_seed,
        )

    @property
    def alpha_controller(self) -> AlphaController:
        return AlphaController(
            self.alpha_mode,
            alpha_max=self.alpha_max,
            beta=self.alpha_beta,
            threshold=self.alpha_threshold,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "cluster_count": self.cluster_count,
            "vector_seed": self.vector_seed,
            "minimum_class_count": self.minimum_class_count,
            "vector_source_artifact_sha256": self.vector_source_artifact_sha256,
            "router_kind": self.router_kind.value,
            "router_seed": self.router_seed,
            "router_hidden_width": self.router_hidden_width,
            "router_epochs": self.router_epochs,
            "distance_temperature": self.distance_temperature,
            "risk_probe_kind": self.risk_probe_kind.value,
            "risk_hidden_width": self.risk_hidden_width,
            "risk_epochs": self.risk_epochs,
            "risk_learning_rate": self.risk_learning_rate,
            "risk_weight_decay": self.risk_weight_decay,
            "risk_class_balanced": self.risk_class_balanced,
            "risk_seed": self.risk_seed,
            "calibration_kind": self.calibration_kind.value,
            "alpha_mode": self.alpha_mode.value,
            "alpha_max": self.alpha_max,
            "alpha_beta": self.alpha_beta,
            "alpha_threshold": self.alpha_threshold,
            "fixed_layer": self.fixed_layer,
            "candidate_layers": list(self.candidate_layers),
            "layer_router_kind": (
                self.layer_router_kind.value if self.layer_router_kind is not None else None
            ),
            "layer_seed": self.layer_seed,
            "layer_epochs": self.layer_epochs,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> M3FitRecipe:
        expected = set(
            cls(
                cluster_count=1,
                vector_seed=17,
                minimum_class_count=1,
                vector_source_artifact_sha256=None,
                router_kind=RouterKind.NEAREST_CENTROID,
                router_seed=17,
                router_hidden_width=64,
                router_epochs=1,
                distance_temperature=1.0,
                risk_probe_kind=ProbeKind.LOGISTIC,
                risk_hidden_width=64,
                risk_epochs=1,
                risk_learning_rate=0.03,
                risk_weight_decay=0.0001,
                risk_class_balanced=True,
                risk_seed=17,
                calibration_kind=CalibrationKind.TEMPERATURE,
                alpha_mode=AlphaMode.FIXED,
                alpha_max=0.5,
                alpha_beta=12.0,
                alpha_threshold=0.5,
                fixed_layer=0,
                candidate_layers=(),
                layer_router_kind=None,
                layer_seed=17,
                layer_epochs=1,
            ).to_dict()
        )
        if set(value) != expected:
            raise DataValidationError("RQ1 M3 fit recipe keys differ")
        data = dict(value)
        data["router_kind"] = RouterKind(data["router_kind"])
        data["risk_probe_kind"] = ProbeKind(data["risk_probe_kind"])
        data["calibration_kind"] = CalibrationKind(data["calibration_kind"])
        data["alpha_mode"] = AlphaMode(data["alpha_mode"])
        data["candidate_layers"] = tuple(data["candidate_layers"])
        if data["layer_router_kind"] is not None:
            data["layer_router_kind"] = RouterKind(data["layer_router_kind"])
        return cls(**data)


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if tensor.ndim != 2 or not torch.isfinite(tensor).all():
        raise DataValidationError("RQ1 fit activation tensor must be one finite matrix")
    return hashlib.sha256(tensor.numpy().tobytes(order="C")).hexdigest()


def fit_rq1_m3_controller(
    *,
    recipe: M3FitRecipe,
    fit_datasets: Mapping[str, ProbeDataset],
    vector_activations: Mapping[HookKey, Tensor],
    best_layers: Sequence[int] | None = None,
) -> AdaptiveController:
    """Run the sole registered RQ1 M3 fitter from complete frozen inputs."""

    if set(fit_datasets) != {"vector_bank", "controller_train", "calibration"}:
        raise DataValidationError("RQ1 M3 fit dataset inventory differs")
    vector_rows = fit_datasets["vector_bank"]
    controller_rows = fit_datasets["controller_train"]
    calibration_rows = fit_datasets["calibration"]
    bank, _ = fit_routed_vector_bank(
        vector_rows,
        vector_activations,
        cluster_count=recipe.cluster_count,
        seed=recipe.vector_seed,
        minimum_class_count=recipe.minimum_class_count,
        source_artifact_sha256=recipe.vector_source_artifact_sha256,
    )
    assignments = assign_to_vector_regions(controller_rows, bank)
    router = fit_adaptive_router(
        controller_rows,
        assignments,
        bank.centers,
        kind=recipe.router_kind,
        seed=recipe.router_seed,
        hidden_width=recipe.router_hidden_width,
        epochs=recipe.router_epochs,
        distance_temperature=recipe.distance_temperature,
    )
    risk_probe = fit_calibrated_probe(
        controller_rows,
        calibration_rows,
        task=ProbeTask.CORRECT_INCORRECT_ABSTENTION,
        training_config=recipe.risk_training_config,
        calibration_kind=recipe.calibration_kind,
    )
    layer_selector = None
    if recipe.candidate_layers:
        if best_layers is None or len(best_layers) != len(controller_rows.question_ids):
            raise DataValidationError("RQ1 routed-layer labels must align with controller rows")
        assert recipe.layer_router_kind is not None
        layer_selector = fit_layer_selector(
            controller_rows,
            best_layers,
            candidate_layers=recipe.candidate_layers,
            kind=recipe.layer_router_kind,
            seed=recipe.layer_seed,
            epochs=recipe.layer_epochs,
        )
    elif best_layers is not None:
        raise DataValidationError("RQ1 fixed-layer fit cannot carry routed-layer labels")
    return AdaptiveController(
        risk_probe=risk_probe,
        vector_bank=bank,
        vector_router=router,
        alpha_controller=recipe.alpha_controller,
        fixed_layer=recipe.fixed_layer,
        layer_selector=layer_selector,
    )


def refit_rq1_m3_vector_bank_controller(
    *,
    source_controller: AdaptiveController,
    recipe: M3FitRecipe,
    fit_datasets: Mapping[str, ProbeDataset],
    vector_activations: Mapping[HookKey, Tensor],
) -> AdaptiveController:
    """Refit only the held-fold vector bank and its router."""

    source_layers = (
        source_controller.layer_selector.candidate_layers
        if source_controller.layer_selector is not None
        else ()
    )
    if (
        set(fit_datasets) != {"vector_bank", "controller_train", "calibration"}
        or source_controller.alpha_controller != recipe.alpha_controller
        or source_controller.fixed_layer != recipe.fixed_layer
        or source_layers != recipe.candidate_layers
    ):
        raise DataValidationError(
            "RQ1 vector-bank refit differs from the fold-source controller recipe"
        )
    vector_rows = fit_datasets["vector_bank"]
    controller_rows = fit_datasets["controller_train"]
    bank, _ = fit_routed_vector_bank(
        vector_rows,
        vector_activations,
        cluster_count=recipe.cluster_count,
        seed=recipe.vector_seed,
        minimum_class_count=recipe.minimum_class_count,
        source_artifact_sha256=recipe.vector_source_artifact_sha256,
    )
    router = fit_adaptive_router(
        controller_rows,
        assign_to_vector_regions(controller_rows, bank),
        bank.centers,
        kind=recipe.router_kind,
        seed=recipe.router_seed,
        hidden_width=recipe.router_hidden_width,
        epochs=recipe.router_epochs,
        distance_temperature=recipe.distance_temperature,
    )
    return AdaptiveController(
        risk_probe=source_controller.risk_probe,
        vector_bank=bank,
        vector_router=router,
        alpha_controller=source_controller.alpha_controller,
        fixed_layer=source_controller.fixed_layer,
        layer_selector=source_controller.layer_selector,
    )


def rq1_m3_fit_capture_attestation_body(
    *,
    plan_digest: str,
    task_id: str,
    stage: str,
    execution_public_key: str,
    runtime_artifact_sha256: str,
    source_question_bundle_sha256: str,
    recipe: M3FitRecipe,
    fit_datasets: Mapping[str, ProbeDataset],
    vector_activations: Mapping[HookKey, Tensor],
    best_layers: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Canonical whole-capture receipt signed by the native VLLM execution key."""

    if (
        _SHA256.fullmatch(plan_digest) is None
        or not task_id.startswith("rq1-")
        or stage not in {"source-fit", "held-out-adaptation"}
        or _SHA256.fullmatch(execution_public_key) is None
        or _SHA256.fullmatch(runtime_artifact_sha256) is None
        or _SHA256.fullmatch(source_question_bundle_sha256) is None
        or set(fit_datasets) != {"vector_bank", "controller_train", "calibration"}
        or not vector_activations
    ):
        raise DataValidationError("RQ1 native fit-capture identity is invalid")
    activation_rows = len(fit_datasets["vector_bank"].question_ids)
    activations: dict[str, dict[str, Any]] = {}
    for key in sorted(vector_activations, key=lambda item: item.artifact_key):
        value = vector_activations[key]
        if value.shape[0] != activation_rows:
            raise DataValidationError("RQ1 vector activation rows differ from T-steer")
        activations[key.artifact_key] = {
            "layer": key.layer,
            "site": key.site.value,
            "shape": list(value.shape),
            "float32_sha256": _tensor_sha256(value),
        }
    labels = None if best_layers is None else [int(value) for value in best_layers]
    if labels is not None and len(labels) != len(fit_datasets["controller_train"].question_ids):
        raise DataValidationError("RQ1 layer labels do not align with controller rows")
    return {
        "receipt_kind": "rq1-native-m3-fit-capture-v1",
        "plan_digest": plan_digest,
        "task_id": task_id,
        "stage": stage,
        "execution_public_key": execution_public_key,
        "runtime_artifact_sha256": runtime_artifact_sha256,
        "source_question_bundle_sha256": source_question_bundle_sha256,
        "recipe_digest": stable_hash(recipe.to_dict()),
        "datasets": {
            name: {
                "data_fingerprint": dataset.data_fingerprint,
                "question_ids_sha256": stable_hash(list(dataset.question_ids)),
                "row_count": len(dataset.question_ids),
                "feature_schema_digest": (
                    dataset.feature_schema.digest if dataset.feature_schema is not None else None
                ),
            }
            for name, dataset in sorted(fit_datasets.items())
        },
        "vector_activations": activations,
        "best_layers": labels,
        "best_layers_sha256": stable_hash(labels),
    }


def _verify_rq1_m3_fit_capture_attestation(
    attestation: Mapping[str, Any],
    *,
    plan: RobustnessDiagnosticPlan,
    task: RQ1GeneralizationTask,
    stage: str,
    execution_public_key: str,
    recipe: M3FitRecipe,
    fit_datasets: Mapping[str, ProbeDataset],
    vector_activations: Mapping[HookKey, Tensor],
    best_layers: Sequence[int] | None,
) -> Mapping[str, Any]:
    body = attestation.get("body")
    signature = attestation.get("signature")
    if (
        set(attestation) != {"body", "signature"}
        or not isinstance(body, Mapping)
        or not isinstance(signature, str)
        or re.fullmatch(r"[0-9a-f]{128}", signature) is None
        or not isinstance(body.get("runtime_artifact_sha256"), str)
        or not isinstance(body.get("source_question_bundle_sha256"), str)
    ):
        raise DataValidationError("RQ1 native fit-capture attestation is invalid")
    expected = rq1_m3_fit_capture_attestation_body(
        plan_digest=plan.plan_digest,
        task_id=task.task_id,
        stage=stage,
        execution_public_key=execution_public_key,
        runtime_artifact_sha256=str(plan.body["m3_capture_runtime_artifact_sha256"]),
        source_question_bundle_sha256=str(
            plan.body["source_artifact_sha256"]["triviaqa-development"]
        ),
        recipe=recipe,
        fit_datasets=fit_datasets,
        vector_activations=vector_activations,
        best_layers=best_layers,
    )
    if canonical_json(dict(body)) != canonical_json(expected):
        raise DataValidationError("RQ1 native fit-capture body differs from its tensors")
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(execution_public_key)).verify(
            bytes.fromhex(signature), canonical_json(expected).encode("utf-8")
        )
    except (InvalidSignature, ValueError) as exc:
        raise DataValidationError("RQ1 native fit-capture signature is invalid") from exc
    return MappingProxyType(expected)


def _assert_m3_refit_matches_component(
    component: ConfirmatoryAdaptiveComponent,
    *,
    recipe: M3FitRecipe,
    fit_datasets: Mapping[str, ProbeDataset],
    vector_activations: Mapping[HookKey, Tensor],
    best_layers: Sequence[int] | None,
    vector_bank_only: bool,
) -> str:
    expected_controller = component.controllers["P0-neutral"]
    if vector_bank_only:
        if best_layers is not None:
            raise DataValidationError(
                "RQ1 vector-bank-only refit cannot relearn the layer selector"
            )
        refitted = refit_rq1_m3_vector_bank_controller(
            source_controller=expected_controller,
            recipe=recipe,
            fit_datasets=fit_datasets,
            vector_activations=vector_activations,
        )
    else:
        refitted = fit_rq1_m3_controller(
            recipe=recipe,
            fit_datasets=fit_datasets,
            vector_activations=vector_activations,
            best_layers=best_layers,
        )
    expected_path = _adaptive_controller_directory(component, "P0-neutral")
    scratch = Path(tempfile.mkdtemp(prefix=".rq1-refit-", dir=component.directory.parent))
    try:
        refit_path = scratch / "controller"
        save_adaptive_controller(refit_path, refitted)
        refit_sha = sha256_path(refit_path)
        if refit_sha != sha256_path(expected_path):
            raise DataValidationError(
                "RQ1 M3 controller parameters do not replay from signed fit evidence"
            )
        return refit_sha
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _strict_artifact(path: str | Path, label: str) -> Path:
    raw = Path(path)
    if ".." in raw.parts:
        raise FrozenArtifactError(f"{label} cannot contain parent traversal")
    lexical = Path(os.path.abspath(raw))
    if any(candidate.is_symlink() for candidate in (lexical, *lexical.parents)):
        raise FrozenArtifactError(f"{label} cannot traverse a symlink")
    value = lexical.resolve(strict=False)
    if (
        not value.exists()
        or (value.is_dir() and any(item.is_symlink() for item in value.rglob("*")))
        or (not value.is_file() and not value.is_dir())
    ):
        raise FrozenArtifactError(f"{label} must be one strict regular artifact")
    return value


def _write_once_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.stage-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(canonical_json(dict(value)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError as exc:
            raise FrozenArtifactError(f"refusing to overwrite result: {path}") from exc
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FrozenArtifactError(f"{label} is missing or linked")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise FrozenArtifactError(f"{label} must contain one JSON object")
    return value


def _result_payload(body: Mapping[str, Any]) -> dict[str, Any]:
    return {**dict(body), "result_digest": stable_hash(dict(body))}


@dataclass(frozen=True, slots=True)
class PromptParaphraseResult:
    task_id: str
    plan_digest: str
    record: GenerationRecord
    result_digest: str

    def to_dict(self) -> dict[str, Any]:
        return _result_payload(
            {
                "schema_version": 1,
                "kind": "prompt-paraphrase-generation",
                "task_id": self.task_id,
                "plan_digest": self.plan_digest,
                "generation_record": self.record.to_dict(),
            }
        )

    @classmethod
    def create(
        cls,
        *,
        task_id: str,
        plan_digest: str,
        record: GenerationRecord,
    ) -> PromptParaphraseResult:
        body = {
            "schema_version": 1,
            "kind": "prompt-paraphrase-generation",
            "task_id": task_id,
            "plan_digest": plan_digest,
            "generation_record": record.to_dict(),
        }
        return cls(task_id, plan_digest, record, stable_hash(body))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PromptParaphraseResult:
        if (
            set(value)
            != {
                "schema_version",
                "kind",
                "task_id",
                "plan_digest",
                "generation_record",
                "result_digest",
            }
            or value.get("schema_version") != 1
            or value.get("kind") != ("prompt-paraphrase-generation")
        ):
            raise FrozenArtifactError("prompt-paraphrase result schema differs")
        body = dict(value)
        digest = body.pop("result_digest")
        try:
            record_value = value["generation_record"]
            if not isinstance(record_value, Mapping):
                raise TypeError("generation_record is not an object")
            record = GenerationRecord.from_dict(record_value)
        except (KeyError, TypeError, ValueError, DataValidationError) as exc:
            raise FrozenArtifactError(f"prompt result record is invalid: {exc}") from exc
        if (
            not isinstance(digest, str)
            or digest != stable_hash(body)
            or not isinstance(value["task_id"], str)
            or not isinstance(value["plan_digest"], str)
        ):
            raise FrozenArtifactError("prompt-paraphrase result identity differs")
        return cls(value["task_id"], value["plan_digest"], record, digest)


@dataclass(frozen=True, slots=True)
class RQ1GeneralizationResult:
    task_id: str
    plan_digest: str
    question_set_digests: Mapping[str, str]
    artifact_locations: Mapping[str, str]
    artifact_fingerprints: Mapping[str, str]
    evaluation_record_count: int
    metrics: Mapping[str, float]
    result_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "question_set_digests", MappingProxyType(dict(self.question_set_digests))
        )
        object.__setattr__(
            self, "artifact_locations", MappingProxyType(dict(self.artifact_locations))
        )
        object.__setattr__(
            self,
            "artifact_fingerprints",
            MappingProxyType(dict(self.artifact_fingerprints)),
        )
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "rq1-semantic-fold-generalization",
            "task_id": self.task_id,
            "plan_digest": self.plan_digest,
            "question_set_digests": dict(self.question_set_digests),
            "artifact_locations": dict(self.artifact_locations),
            "artifact_fingerprints": dict(self.artifact_fingerprints),
            "evaluation_record_count": self.evaluation_record_count,
            "metrics": dict(self.metrics),
        }

    def to_dict(self) -> dict[str, Any]:
        return _result_payload(self._body())

    @classmethod
    def create(
        cls,
        *,
        task_id: str,
        plan_digest: str,
        question_set_digests: Mapping[str, str],
        artifact_locations: Mapping[str, str],
        artifact_fingerprints: Mapping[str, str],
        evaluation_record_count: int,
        metrics: Mapping[str, float],
    ) -> RQ1GeneralizationResult:
        normalized_metrics = {name: float(value) for name, value in metrics.items()}
        provisional = cls(
            task_id,
            plan_digest,
            question_set_digests,
            artifact_locations,
            artifact_fingerprints,
            evaluation_record_count,
            normalized_metrics,
            "",
        )
        return cls(
            task_id,
            plan_digest,
            question_set_digests,
            artifact_locations,
            artifact_fingerprints,
            evaluation_record_count,
            normalized_metrics,
            stable_hash(provisional._body()),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RQ1GeneralizationResult:
        expected = {
            "schema_version",
            "kind",
            "task_id",
            "plan_digest",
            "question_set_digests",
            "artifact_locations",
            "artifact_fingerprints",
            "evaluation_record_count",
            "metrics",
            "result_digest",
        }
        if (
            set(value) != expected
            or value.get("schema_version") != 1
            or value.get("kind") != "rq1-semantic-fold-generalization"
            or not isinstance(value.get("task_id"), str)
            or not isinstance(value.get("plan_digest"), str)
            or not isinstance(value.get("question_set_digests"), dict)
            or not isinstance(value.get("artifact_locations"), dict)
            or not isinstance(value.get("artifact_fingerprints"), dict)
            or type(value.get("evaluation_record_count")) is not int
            or not isinstance(value.get("metrics"), dict)
            or not isinstance(value.get("result_digest"), str)
        ):
            raise FrozenArtifactError("RQ1 result schema differs")
        try:
            result = cls.create(
                task_id=value["task_id"],
                plan_digest=value["plan_digest"],
                question_set_digests=value["question_set_digests"],
                artifact_locations=value["artifact_locations"],
                artifact_fingerprints=value["artifact_fingerprints"],
                evaluation_record_count=value["evaluation_record_count"],
                metrics=value["metrics"],
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise FrozenArtifactError(f"RQ1 result values are invalid: {exc}") from exc
        if result.result_digest != value["result_digest"]:
            raise FrozenArtifactError("RQ1 result digest differs")
        return result


@dataclass(frozen=True, slots=True)
class RobustnessResultStore:
    directory: Path
    plan: RobustnessDiagnosticPlan


@dataclass(frozen=True, slots=True)
class VerifiedPromptParaphraseRecord:
    """One fully replayed prompt-paraphrase task and its graded generation."""

    task_id: str
    benchmark: str
    base_prompt_id: str
    paraphrase_prompt_id: str
    method: str
    record: GenerationRecord


@dataclass(frozen=True, slots=True)
class RQ1ScopedComponent:
    directory: Path
    task_id: str
    stage: str
    execution_component: Path
    execution_component_sha256: str
    adaptive_policy: AdaptivePolicySpec | None
    field_fingerprints: Mapping[str, str]
    scope_manifest_digest: str
    fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "field_fingerprints",
            MappingProxyType(dict(self.field_fingerprints)),
        )


def _task_maps(
    plan: RobustnessDiagnosticPlan,
) -> tuple[dict[str, PromptParaphraseTask], dict[str, RQ1GeneralizationTask]]:
    return (
        {task.task_id: task for task in iter_prompt_paraphrase_tasks(plan)},
        {task.task_id: task for task in iter_rq1_generalization_tasks(plan)},
    )


def _adaptive_controller_directory(
    component: ConfirmatoryAdaptiveComponent,
    prompt_id: str,
) -> Path:
    manifest = _load_json(component.directory / "manifest.json", "adaptive component")
    descriptors = manifest.get("controllers")
    matches = (
        [
            row
            for row in descriptors
            if isinstance(row, Mapping) and row.get("prompt_id") == prompt_id
        ]
        if isinstance(descriptors, list)
        else []
    )
    if len(matches) != 1 or not isinstance(matches[0].get("controller_path"), str):
        raise FrozenArtifactError("RQ1 adaptive component lacks its P0 controller path")
    path = component.directory / str(matches[0]["controller_path"])
    if (
        sha256_path(_strict_artifact(path, "RQ1 adaptive controller"))
        != (component.controller_fingerprints[prompt_id])
    ):
        raise FrozenArtifactError("RQ1 adaptive controller bytes changed")
    return path


def _rq1_component_field_payloads(
    execution_component: Path,
    *,
    method: str,
    risk_threshold: float,
    abstention_threshold: float,
    adaptive_policy: AdaptivePolicySpec | None,
) -> Mapping[str, Mapping[str, Any]]:
    if (
        isinstance(risk_threshold, bool)
        or not isinstance(risk_threshold, int | float)
        or isinstance(abstention_threshold, bool)
        or not isinstance(abstention_threshold, int | float)
        or not 0 < float(risk_threshold) < 1
        or not 0 < float(abstention_threshold) < 1
    ):
        raise DataValidationError("RQ1 calibration thresholds must lie in (0, 1)")
    if method == "M1":
        if adaptive_policy is not None:
            raise DataValidationError("RQ1 M1 cannot carry an adaptive policy")
        fixed = load_confirmatory_fixed_component(execution_component)
        if fixed.method != "M1":
            raise DataValidationError("RQ1 M1 execution component has another method")
        payloads: dict[str, Mapping[str, Any]] = {
            "vector_bank": {
                "kind": "static-vector-bank",
                "sha256": fixed.source_artifact_sha256,
            },
            "router": {"kind": "global-static", "sha256": None},
            "directions": {
                "kind": "selected-static-direction",
                "sha256": fixed.direction_sha256,
            },
            "alpha_controller": {
                "kind": "fixed-standardized-alpha",
                "standardized_alpha": fixed.standardized_alpha,
                "reference_rms": fixed.reference_rms,
            },
            "risk_threshold": {"value": float(risk_threshold)},
            "abstention_threshold": {"value": float(abstention_threshold)},
            "router_architecture": {"kind": "global-static", "vector_count": 1},
            "candidate_layers": {"values": [fixed.layer]},
            "candidate_sites": {"values": [fixed.site.value]},
            "token_scopes": {"values": [fixed.token_scope.value]},
            "sparsity": {"value": fixed.sparsity},
            "alpha_policy_family": {"value": "fixed"},
            "likely_unknown_threshold": {"value": None},
            "execution_public_key": {"value": None},
            "risk_probe": {"kind": "not-applicable"},
            "layer_selector": {"kind": "fixed", "layer": fixed.layer},
        }
    elif method == "M3":
        adaptive = load_confirmatory_adaptive_component(execution_component)
        if (
            adaptive_policy is None
            or adaptive_policy.schema_version != 2
            or adaptive_policy.controller_artifact_sha256 != adaptive.fingerprint
            or adaptive_policy.release_risk_threshold != float(risk_threshold)
            or adaptive_policy.abstention_probability_threshold != float(abstention_threshold)
            or "P0-neutral" not in adaptive.controllers
        ):
            raise DataValidationError("RQ1 M3 component/policy calibration differs")
        controller_path = _adaptive_controller_directory(adaptive, "P0-neutral")
        controller = adaptive.controllers["P0-neutral"]
        controller_metadata = _load_json(
            controller_path / "metadata.json", "RQ1 adaptive controller"
        )
        alpha_controller = controller_metadata.get("alpha_controller")
        if not isinstance(alpha_controller, Mapping):
            raise FrozenArtifactError("RQ1 adaptive alpha controller is invalid")
        if dict(alpha_controller) != {
            "mode": adaptive_policy.alpha_mode,
            "alpha_max": adaptive_policy.alpha_max,
            "beta": adaptive_policy.alpha_beta,
            "threshold": adaptive_policy.alpha_risk_threshold,
        }:
            raise DataValidationError("RQ1 adaptive policy changes the executable alpha controller")
        controller_layers = (
            (controller.fixed_layer,)
            if controller.fixed_layer is not None
            else controller.layer_selector.candidate_layers
            if controller.layer_selector is not None
            else ()
        )
        controller_sites = tuple(
            sorted(
                {key.site for key in controller.vector_bank.directions},
                key=lambda item: item.value,
            )
        )
        if (
            adaptive_policy.vector_count != controller.vector_bank.cluster_count
            or adaptive_policy.candidate_layers != controller_layers
            or tuple(sorted(adaptive_policy.candidate_sites, key=lambda item: item.value))
            != controller_sites
            or adaptive_policy.sparsity is not None
        ):
            raise DataValidationError(
                "RQ1 adaptive policy geometry differs from the executable controller"
            )
        payloads = {
            "vector_bank": {
                "kind": "routed-vector-bank",
                "sha256": sha256_path(controller_path / "vector_bank"),
            },
            "router": {
                "kind": "adaptive-vector-router",
                "sha256": sha256_path(controller_path / "vector_router"),
            },
            "directions": {
                "kind": "routed-direction-tensors",
                "sha256": sha256_path(controller_path / "vector_bank" / "vectors.safetensors"),
            },
            "alpha_controller": dict(alpha_controller),
            "risk_threshold": {"value": float(risk_threshold)},
            "abstention_threshold": {"value": float(abstention_threshold)},
            "router_architecture": {
                "kind": controller.vector_router.kind.value,
                "input_width": controller.vector_router.input_width,
                "vector_count": controller.vector_router.cluster_count,
                "classifier_kind": (
                    controller.vector_router.classifier.kind.value
                    if controller.vector_router.classifier is not None
                    else None
                ),
                "classifier_hidden_width": (
                    controller.vector_router.classifier.hidden_width
                    if controller.vector_router.classifier is not None
                    else None
                ),
            },
            "candidate_layers": {"values": list(adaptive_policy.candidate_layers)},
            "candidate_sites": {
                "values": [value.value for value in adaptive_policy.candidate_sites]
            },
            "token_scopes": {
                "values": [value.value for value in adaptive_policy.candidate_token_scopes]
            },
            "sparsity": {"value": adaptive_policy.sparsity},
            "alpha_policy_family": {"value": adaptive_policy.alpha_mode},
            "likely_unknown_threshold": {"value": adaptive_policy.likely_unknown_risk_threshold},
            "execution_public_key": {"value": adaptive_policy.execution_public_key},
            "risk_probe": {
                "kind": "calibrated-risk-probe",
                "sha256": sha256_path(controller_path / "risk_probe"),
            },
            "layer_selector": (
                {
                    "kind": "adaptive-layer-router",
                    "sha256": sha256_path(controller_path / "layer_router"),
                }
                if controller.layer_selector is not None
                else {"kind": "fixed", "layer": controller.fixed_layer}
            ),
        }
    else:
        raise DataValidationError("RQ1 scoped components support only M1 and M3")
    if set(payloads) != _SCOPE_FIELDS:
        raise DataValidationError("RQ1 derived field inventory differs")
    return MappingProxyType(payloads)


def _expected_rq1_fit_ids(
    plan: RobustnessDiagnosticPlan,
    task: RQ1GeneralizationTask,
    stage: str,
) -> tuple[str, ...]:
    question_sets = rq1_task_question_sets(plan, task)
    if task.adaptation_regime == "source-frozen-control":
        return ()
    if stage == "source-fit":
        return (*question_sets["source_fit"], *question_sets["source_calibration"])
    if stage == "held-out-adaptation":
        if task.adaptation_regime == "calibration-only":
            return _expected_threshold_fit_ids(plan, task, stage)
        return question_sets["held_out_adaptation"]
    raise DataValidationError("RQ1 component stage is not preregistered")


def _full_relearning_fit_ids(
    plan: RobustnessDiagnosticPlan,
    task: RQ1GeneralizationTask,
) -> Mapping[str, tuple[str, ...]]:
    section = plan.body["rq1_generalization"]
    assert isinstance(section, Mapping)
    if section.get("full_relearning_subdivision_algorithm") != (
        "semantic-fold-preserve-preregistered-partitions-v1"
    ):
        raise FrozenArtifactError("RQ1 full-relearning subdivision differs")
    assignments = {str(row["question_id"]): str(row["partition"]) for row in section["assignments"]}
    identifiers = rq1_task_question_sets(plan, task)["held_out_adaptation"]
    result = {
        "vector_bank": tuple(value for value in identifiers if assignments[value] == "T-steer"),
        "controller_train": tuple(
            value for value in identifiers if assignments[value] == "T-controller-train"
        ),
        "calibration": tuple(
            value for value in identifiers if assignments[value] == "T-controller-calibration"
        ),
    }
    if any(not values for values in result.values()):
        raise FrozenArtifactError("RQ1 full-relearning fold is too small to fit")
    return MappingProxyType(result)


def _m3_component_fit_provenance(
    plan: RobustnessDiagnosticPlan,
    task: RQ1GeneralizationTask,
    *,
    stage: str,
    component: ConfirmatoryAdaptiveComponent,
    fit_datasets: Mapping[str, ProbeDataset],
    recipe: M3FitRecipe,
    vector_activations: Mapping[HookKey, Tensor],
    best_layers: Sequence[int] | None,
    capture_attestation: Mapping[str, Any],
    execution_public_key: str,
) -> Mapping[str, Any]:
    controller = component.controllers["P0-neutral"]
    vector_bank_only = (
        stage == "held-out-adaptation" and task.adaptation_regime == "full-vector-bank-relearning"
    )
    if vector_bank_only:
        expected = _full_relearning_fit_ids(plan, task)
    else:
        question_sets = rq1_task_question_sets(plan, task)
        assignments = {
            str(row["question_id"]): str(row["partition"])
            for row in plan.body["rq1_generalization"]["assignments"]
        }
        expected = MappingProxyType(
            {
                "vector_bank": tuple(
                    value
                    for value in question_sets["source_fit"]
                    if assignments[value] == "T-steer"
                ),
                "controller_train": tuple(
                    value
                    for value in question_sets["source_fit"]
                    if assignments[value] == "T-controller-train"
                ),
                "calibration": question_sets["source_calibration"],
            }
        )
    if set(fit_datasets) != {"vector_bank", "controller_train", "calibration"}:
        raise DataValidationError("RQ1 M3 fit dataset inventory differs")
    observed = {name: tuple(dataset.question_ids) for name, dataset in fit_datasets.items()}
    risk_fit_differs = (
        controller.risk_probe.training_fingerprint
        != fit_datasets["controller_train"].data_fingerprint
        or controller.risk_probe.calibration_fingerprint
        != fit_datasets["calibration"].data_fingerprint
        or controller.risk_probe.training_schema != fit_datasets["controller_train"].feature_schema
        or controller.risk_probe.calibration_schema != fit_datasets["calibration"].feature_schema
        or (
            controller.layer_selector is not None
            and controller.layer_selector.router.training_fingerprint
            != fit_datasets["controller_train"].data_fingerprint
        )
    )
    if (
        set(observed["vector_bank"]) != set(expected["vector_bank"])
        or set(observed["controller_train"]) != set(expected["controller_train"])
        or set(observed["calibration"]) != set(expected["calibration"])
        or any(len(values) != len(set(values)) for values in observed.values())
        or controller.vector_bank.data_fingerprint != fit_datasets["vector_bank"].data_fingerprint
        or controller.vector_router.training_fingerprint
        != fit_datasets["controller_train"].data_fingerprint
        or controller.vector_bank.feature_schema != fit_datasets["vector_bank"].feature_schema
        or controller.vector_router.feature_schema
        != fit_datasets["controller_train"].feature_schema
        or (not vector_bank_only and risk_fit_differs)
        or (vector_bank_only and best_layers is not None)
    ):
        raise DataValidationError(
            "RQ1 M3 component training schemas differ from the exact semantic-fold inputs"
        )
    attested = _verify_rq1_m3_fit_capture_attestation(
        capture_attestation,
        plan=plan,
        task=task,
        stage=stage,
        execution_public_key=execution_public_key,
        recipe=recipe,
        fit_datasets=fit_datasets,
        vector_activations=vector_activations,
        best_layers=best_layers,
    )
    refit_sha = _assert_m3_refit_matches_component(
        component,
        recipe=recipe,
        fit_datasets=fit_datasets,
        vector_activations=vector_activations,
        best_layers=best_layers,
        vector_bank_only=vector_bank_only,
    )
    return MappingProxyType(
        {
            "algorithm": (
                "full-vector-bank-relearning"
                if stage == "held-out-adaptation"
                and task.adaptation_regime == "full-vector-bank-relearning"
                else "source-fold-fit"
            ),
            "question_ids_sha256": {
                name: stable_hash(list(values)) for name, values in sorted(expected.items())
            },
            "vector_bank_data_fingerprint": controller.vector_bank.data_fingerprint,
            "router_training_fingerprint": controller.vector_router.training_fingerprint,
            "risk_training_fingerprint": controller.risk_probe.training_fingerprint,
            "risk_calibration_fingerprint": controller.risk_probe.calibration_fingerprint,
            "fit_recipe_digest": stable_hash(recipe.to_dict()),
            "capture_attestation_digest": stable_hash(dict(capture_attestation)),
            "capture_runtime_artifact_sha256": attested["runtime_artifact_sha256"],
            "capture_source_question_bundle_sha256": attested["source_question_bundle_sha256"],
            "refit_controller_sha256": refit_sha,
        }
    )


def _balanced_threshold(
    scores: Sequence[float],
    labels: Sequence[bool],
    *,
    positive_at_or_above: bool,
) -> float:
    if (
        len(scores) != len(labels)
        or not scores
        or any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores)
        or len(set(labels)) != 2
    ):
        raise DataValidationError("RQ1 threshold calibration requires both finite classes")
    ordered = sorted(set(float(value) for value in scores))
    boundaries = [1e-9]
    boundaries.extend((left + right) / 2 for left, right in pairwise(ordered))
    boundaries.append(1 - 1e-9)
    positives = sum(labels)
    negatives = len(labels) - positives
    best: tuple[float, float] | None = None
    for threshold in boundaries:
        predicted = tuple(
            value >= threshold if positive_at_or_above else value > threshold for value in scores
        )
        true_positive = sum(value and label for value, label in zip(predicted, labels, strict=True))
        true_negative = sum(
            not value and not label for value, label in zip(predicted, labels, strict=True)
        )
        balanced_accuracy = 0.5 * (true_positive / positives + true_negative / negatives)
        candidate = (balanced_accuracy, -threshold)
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    return -best[1]


def _calibrated_m3_policy(
    component: ConfirmatoryAdaptiveComponent,
    template: AdaptivePolicySpec,
    dataset: ProbeDataset,
) -> AdaptivePolicySpec:
    controller = component.controllers["P0-neutral"]
    if (
        dataset.feature_schema is None
        or not controller.risk_probe.training_schema.is_compatible_representation(
            dataset.feature_schema
        )
        or any(
            outcome not in {Outcome.CORRECT, Outcome.INCORRECT, Outcome.ABSTENTION}
            for outcome in dataset.outcomes
        )
    ):
        raise DataValidationError("RQ1 threshold-fit dataset is incompatible with M3")
    probabilities = controller.risk_probe.predict_probabilities(dataset.features)
    labels = controller.risk_probe.state.labels
    incorrect = labels.index(Outcome.INCORRECT.value)
    abstention = labels.index(Outcome.ABSTENTION.value)
    risk_threshold = _balanced_threshold(
        [float(value) for value in probabilities[:, incorrect]],
        [outcome is Outcome.INCORRECT for outcome in dataset.outcomes],
        positive_at_or_above=False,
    )
    abstention_threshold = _balanced_threshold(
        [float(value) for value in probabilities[:, abstention]],
        [outcome is Outcome.ABSTENTION for outcome in dataset.outcomes],
        positive_at_or_above=True,
    )
    return replace(
        template,
        release_risk_threshold=risk_threshold,
        abstention_probability_threshold=abstention_threshold,
        controller_artifact_sha256=component.fingerprint,
    )


def _write_fit_evidence(path: Path, dataset: ProbeDataset) -> None:
    path.mkdir()
    tensor_path = path / "features.safetensors"
    save_file({"features": dataset.features.contiguous()}, tensor_path)
    assert dataset.feature_schema is not None
    body = {
        "schema_version": 1,
        "question_ids": list(dataset.question_ids),
        "outcomes": [value.value for value in dataset.outcomes],
        "group_ids": list(dataset.group_ids),
        "feature_schema": dataset.feature_schema.to_dict(),
        "data_fingerprint": dataset.data_fingerprint,
        "tensor_sha256": sha256_file(tensor_path),
    }
    (path / "metadata.json").write_text(
        canonical_json({**body, "metadata_digest": stable_hash(body)}) + "\n",
        encoding="utf-8",
    )


def _load_fit_evidence(path: Path) -> ProbeDataset:
    if (
        path.is_symlink()
        or not path.is_dir()
        or {item.name for item in path.iterdir()} != {"metadata.json", "features.safetensors"}
        or any(item.is_symlink() for item in path.iterdir())
    ):
        raise FrozenArtifactError("RQ1 fit evidence inventory differs")
    metadata = _load_json(path / "metadata.json", "RQ1 fit evidence")
    digest = metadata.pop("metadata_digest", None)
    tensor_path = path / "features.safetensors"
    if (
        set(metadata)
        != {
            "schema_version",
            "question_ids",
            "outcomes",
            "group_ids",
            "feature_schema",
            "data_fingerprint",
            "tensor_sha256",
        }
        or metadata.get("schema_version") != 1
        or digest != stable_hash(metadata)
        or metadata.get("tensor_sha256") != sha256_file(tensor_path)
    ):
        raise FrozenArtifactError("RQ1 fit evidence identity differs")
    tensors = load_file(tensor_path, device="cpu")
    if set(tensors) != {"features"}:
        raise FrozenArtifactError("RQ1 fit evidence tensor inventory differs")
    try:
        return ProbeDataset(
            question_ids=tuple(str(value) for value in metadata["question_ids"]),
            features=tensors["features"],
            outcomes=tuple(Outcome(str(value)) for value in metadata["outcomes"]),
            group_ids=tuple(str(value) for value in metadata["group_ids"]),
            feature_schema=ActivationFeatureSchema.from_dict(metadata["feature_schema"]),
            data_fingerprint=str(metadata["data_fingerprint"]),
        )
    except (KeyError, TypeError, ValueError, DataValidationError) as exc:
        raise FrozenArtifactError(f"RQ1 fit evidence values are invalid: {exc}") from exc


def _write_m3_component_fit_evidence(
    root: Path,
    *,
    fit_datasets: Mapping[str, ProbeDataset],
    recipe: M3FitRecipe,
    vector_activations: Mapping[HookKey, Tensor],
    best_layers: Sequence[int] | None,
    attestation: Mapping[str, Any],
) -> None:
    root.mkdir()
    for name, dataset in sorted(fit_datasets.items()):
        _write_fit_evidence(root / name, dataset)
    tensor_path = root / "vector-activations.safetensors"
    ordered_hooks = sorted(vector_activations, key=lambda item: item.artifact_key)
    save_file(
        {
            f"activation.{key.artifact_key}": vector_activations[key]
            .detach()
            .cpu()
            .float()
            .contiguous()
            for key in ordered_hooks
        },
        tensor_path,
    )
    body = {
        "schema_version": 1,
        "recipe": recipe.to_dict(),
        "recipe_digest": stable_hash(recipe.to_dict()),
        "best_layers": (None if best_layers is None else [int(value) for value in best_layers]),
        "activation_hooks": [
            {"layer": key.layer, "site": key.site.value, "tensor_key": key.artifact_key}
            for key in ordered_hooks
        ],
        "activation_tensor_sha256": sha256_file(tensor_path),
        "capture_attestation": dict(attestation),
    }
    (root / "refit.json").write_text(
        canonical_json({**body, "metadata_digest": stable_hash(body)}) + "\n",
        encoding="utf-8",
    )


def _load_m3_component_fit_evidence(
    root: Path,
) -> tuple[
    Mapping[str, ProbeDataset],
    M3FitRecipe,
    Mapping[HookKey, Tensor],
    tuple[int, ...] | None,
    Mapping[str, Any],
]:
    expected = {
        "vector_bank",
        "controller_train",
        "calibration",
        "vector-activations.safetensors",
        "refit.json",
    }
    if (
        root.is_symlink()
        or not root.is_dir()
        or {item.name for item in root.iterdir()} != expected
        or any(item.is_symlink() for item in root.iterdir())
    ):
        raise FrozenArtifactError("RQ1 M3 component fit-evidence inventory differs")
    datasets = MappingProxyType(
        {
            name: _load_fit_evidence(root / name)
            for name in ("vector_bank", "controller_train", "calibration")
        }
    )
    metadata = _load_json(root / "refit.json", "RQ1 M3 refit evidence")
    digest = metadata.pop("metadata_digest", None)
    expected_keys = {
        "schema_version",
        "recipe",
        "recipe_digest",
        "best_layers",
        "activation_hooks",
        "activation_tensor_sha256",
        "capture_attestation",
    }
    tensor_path = root / "vector-activations.safetensors"
    if (
        set(metadata) != expected_keys
        or metadata.get("schema_version") != 1
        or digest != stable_hash(metadata)
        or metadata.get("recipe_digest") != stable_hash(metadata.get("recipe"))
        or metadata.get("activation_tensor_sha256") != sha256_file(tensor_path)
        or not isinstance(metadata.get("activation_hooks"), list)
        or not isinstance(metadata.get("capture_attestation"), Mapping)
    ):
        raise FrozenArtifactError("RQ1 M3 refit evidence identity differs")
    try:
        recipe_value = metadata["recipe"]
        if not isinstance(recipe_value, Mapping):
            raise TypeError("recipe must be an object")
        recipe = M3FitRecipe.from_dict(recipe_value)
        tensors = load_file(tensor_path, device="cpu")
        activations: dict[HookKey, Tensor] = {}
        expected_tensor_keys: set[str] = set()
        for row in metadata["activation_hooks"]:
            if not isinstance(row, Mapping) or set(row) != {"layer", "site", "tensor_key"}:
                raise TypeError("activation hook is invalid")
            key = HookKey(int(row["layer"]), ActivationSite(str(row["site"])))
            if row["tensor_key"] != key.artifact_key:
                raise ValueError("activation hook key differs")
            tensor_key = f"activation.{key.artifact_key}"
            expected_tensor_keys.add(tensor_key)
            activations[key] = tensors[tensor_key]
        if set(tensors) != expected_tensor_keys or not activations:
            raise ValueError("activation tensors differ")
        labels_value = metadata["best_layers"]
        if labels_value is not None and (
            not isinstance(labels_value, list)
            or any(type(value) is not int for value in labels_value)
        ):
            raise TypeError("best-layer labels are invalid")
        labels = None if labels_value is None else tuple(labels_value)
    except (KeyError, TypeError, ValueError, DataValidationError) as exc:
        raise FrozenArtifactError(f"RQ1 M3 refit evidence is invalid: {exc}") from exc
    return (
        datasets,
        recipe,
        MappingProxyType(activations),
        labels,
        MappingProxyType(dict(metadata["capture_attestation"])),
    )


def _expected_threshold_fit_ids(
    plan: RobustnessDiagnosticPlan,
    task: RQ1GeneralizationTask,
    stage: str,
) -> tuple[str, ...]:
    question_sets = rq1_task_question_sets(plan, task)
    if stage == "source-fit":
        return question_sets["source_calibration"]
    if stage == "held-out-adaptation":
        if task.adaptation_regime == "full-vector-bank-relearning":
            return _full_relearning_fit_ids(plan, task)["calibration"]
        assignments = {
            str(row["question_id"]): str(row["partition"])
            for row in plan.body["rq1_generalization"]["assignments"]
        }
        return tuple(
            value
            for value in question_sets["held_out_adaptation"]
            if assignments[value] == "T-controller-calibration"
        )
    raise DataValidationError("RQ1 threshold fit stage differs")


def write_rq1_scoped_component(
    directory: str | Path,
    *,
    plan: RobustnessDiagnosticPlan,
    task: RQ1GeneralizationTask,
    stage: str,
    execution_component: str | Path,
    risk_threshold: float = 0.5,
    abstention_threshold: float = 0.5,
    adaptive_policy: AdaptivePolicySpec | None = None,
    calibration_dataset: ProbeDataset | None = None,
    fit_datasets: Mapping[str, ProbeDataset] | None = None,
    fit_recipe: M3FitRecipe | None = None,
    vector_activations: Mapping[HookKey, Tensor] | None = None,
    best_layers: Sequence[int] | None = None,
    fit_capture_attestation: Mapping[str, Any] | None = None,
) -> RQ1ScopedComponent:
    """Freeze one executable fold component and derive all regime-comparison fields."""

    destination = validate_active_study_artifact_paths({"RQ1 scoped component": directory})[
        "RQ1 scoped component"
    ]
    if destination.exists() or destination.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite RQ1 component: {destination}")
    source = _strict_artifact(execution_component, "RQ1 execution component")
    expected_fit_ids = _expected_rq1_fit_ids(plan, task, stage)
    base_component_sha = _base_component_sha256(plan, task.method)
    if task.method == "M1" and (
        sha256_path(source) != base_component_sha
        or adaptive_policy is not None
        or calibration_dataset is not None
        or fit_datasets is not None
        or fit_recipe is not None
        or vector_activations is not None
        or best_layers is not None
        or fit_capture_attestation is not None
        or risk_threshold != 0.5
        or abstention_threshold != 0.5
    ):
        raise DataValidationError("RQ1 M1 must remain the exact source-frozen control")
    fit_provenance: Mapping[str, Any]
    if task.method == "M3":
        adaptive_component = load_confirmatory_adaptive_component(source)
        if (
            adaptive_policy is None
            or calibration_dataset is None
            or fit_datasets is None
            or fit_recipe is None
            or vector_activations is None
            or fit_capture_attestation is None
            or set(calibration_dataset.question_ids)
            != set(_expected_threshold_fit_ids(plan, task, stage))
            or len(calibration_dataset.question_ids)
            != len(_expected_threshold_fit_ids(plan, task, stage))
        ):
            raise DataValidationError("RQ1 M3 requires the exact fold calibration dataset")
        adaptive_policy = _calibrated_m3_policy(
            adaptive_component,
            adaptive_policy,
            calibration_dataset,
        )
        risk_threshold = adaptive_policy.release_risk_threshold
        abstention_threshold = adaptive_policy.abstention_probability_threshold
        fit_provenance = _m3_component_fit_provenance(
            plan,
            task,
            stage=stage,
            component=adaptive_component,
            fit_datasets=fit_datasets,
            recipe=fit_recipe,
            vector_activations=vector_activations,
            best_layers=best_layers,
            capture_attestation=fit_capture_attestation,
            execution_public_key=adaptive_policy.execution_public_key,
        )
    else:
        fit_provenance = MappingProxyType(
            {
                "algorithm": "source-frozen-control-no-fit",
                "base_frozen_component_sha256": base_component_sha,
            }
        )
    payloads = _rq1_component_field_payloads(
        source,
        method=task.method,
        risk_threshold=risk_threshold,
        abstention_threshold=abstention_threshold,
        adaptive_policy=adaptive_policy,
    )
    if task.method == "M3":
        assert adaptive_policy is not None
        assert fit_recipe is not None
        assert vector_activations is not None
        _validate_m3_source_refit_against_base(
            plan,
            execution_component=source,
            adaptive_policy=adaptive_policy,
            recipe=fit_recipe,
            vector_activations=vector_activations,
            source_payloads=payloads,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage_path = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent)
    )
    try:
        shutil.copytree(source, stage_path / "execution-component")
        fit_evidence_path: str | None = None
        fit_evidence_sha256: str | None = None
        if calibration_dataset is not None:
            fit_evidence_path = "fit-evidence"
            evidence_root = stage_path / fit_evidence_path
            evidence_root.mkdir()
            component_root = evidence_root / "component"
            assert fit_datasets is not None
            assert fit_recipe is not None
            assert vector_activations is not None
            assert fit_capture_attestation is not None
            _write_m3_component_fit_evidence(
                component_root,
                fit_datasets=fit_datasets,
                recipe=fit_recipe,
                vector_activations=vector_activations,
                best_layers=best_layers,
                attestation=fit_capture_attestation,
            )
            _write_fit_evidence(evidence_root / "threshold", calibration_dataset)
            fit_evidence_sha256 = sha256_path(stage_path / fit_evidence_path)
        fields_root = stage_path / "fields"
        fields_root.mkdir()
        field_descriptors: dict[str, dict[str, str]] = {}
        for name, payload in payloads.items():
            relative = f"fields/{name}.json"
            path = stage_path / relative
            path.write_text(canonical_json(dict(payload)) + "\n", encoding="utf-8")
            field_descriptors[name] = {
                "path": relative,
                "sha256": sha256_path(path),
            }
        policy_path: str | None = None
        policy_digest: str | None = None
        if adaptive_policy is not None:
            policy_path = "adaptive-policy.json"
            policy_body = adaptive_policy.to_dict()
            policy_digest = stable_hash(policy_body)
            (stage_path / policy_path).write_text(
                canonical_json({**policy_body, "policy_digest": policy_digest}) + "\n",
                encoding="utf-8",
            )
        body = {
            "schema_version": 2,
            "component_kind": "rq1-scoped-adaptation",
            "plan_digest": plan.plan_digest,
            "task_id": task.task_id,
            "stage": stage,
            "method": task.method,
            "training_prompt_id": task.training_prompt_id,
            "base_frozen_component_sha256": base_component_sha,
            "fit_question_ids": list(expected_fit_ids),
            "fit_provenance": dict(fit_provenance),
            "fit_evidence_path": fit_evidence_path,
            "fit_evidence_sha256": fit_evidence_sha256,
            "risk_threshold": float(risk_threshold),
            "abstention_threshold": float(abstention_threshold),
            "execution_component_path": "execution-component",
            "execution_component_sha256": sha256_path(stage_path / "execution-component"),
            "adaptive_policy_path": policy_path,
            "adaptive_policy_digest": policy_digest,
            "fields": field_descriptors,
        }
        (stage_path / "scope-manifest.json").write_text(
            canonical_json({**body, "manifest_digest": stable_hash(body)}) + "\n",
            encoding="utf-8",
        )
        load_rq1_scoped_component(
            stage_path,
            plan=plan,
            task=task,
            stage=stage,
        )
        os.replace(stage_path, destination)
    finally:
        if stage_path.exists():
            shutil.rmtree(stage_path)
    return load_rq1_scoped_component(
        destination,
        plan=plan,
        task=task,
        stage=stage,
    )


def load_rq1_scoped_component(
    directory: str | Path,
    *,
    plan: RobustnessDiagnosticPlan,
    task: RQ1GeneralizationTask,
    stage: str,
) -> RQ1ScopedComponent:
    """Replay a scoped component from its executable bytes and exact fit IDs."""

    path = _strict_artifact(directory, "RQ1 scoped component")
    if not path.is_dir() or any(item.is_symlink() for item in path.rglob("*")):
        raise FrozenArtifactError("RQ1 scoped component is not a strict directory")
    expected_inventory = set(_RQ1_COMPONENT_FILES)
    if task.method == "M3":
        expected_inventory.update({"adaptive-policy.json", "fit-evidence"})
    if {item.name for item in path.iterdir()} != expected_inventory:
        raise FrozenArtifactError("RQ1 scoped component inventory differs")
    manifest = _load_json(path / "scope-manifest.json", "RQ1 component scope")
    digest = manifest.pop("manifest_digest", None)
    expected_keys = {
        "schema_version",
        "component_kind",
        "plan_digest",
        "task_id",
        "stage",
        "method",
        "training_prompt_id",
        "base_frozen_component_sha256",
        "fit_question_ids",
        "fit_provenance",
        "fit_evidence_path",
        "fit_evidence_sha256",
        "risk_threshold",
        "abstention_threshold",
        "execution_component_path",
        "execution_component_sha256",
        "adaptive_policy_path",
        "adaptive_policy_digest",
        "fields",
    }
    expected_fit_ids = _expected_rq1_fit_ids(plan, task, stage)
    fields = manifest.get("fields")
    if (
        set(manifest) != expected_keys
        or manifest.get("schema_version") != 2
        or manifest.get("component_kind") != "rq1-scoped-adaptation"
        or manifest.get("plan_digest") != plan.plan_digest
        or manifest.get("task_id") != task.task_id
        or manifest.get("stage") != stage
        or manifest.get("method") != task.method
        or manifest.get("training_prompt_id") != task.training_prompt_id
        or manifest.get("base_frozen_component_sha256") != _base_component_sha256(plan, task.method)
        or manifest.get("fit_question_ids") != list(expected_fit_ids)
        or manifest.get("execution_component_path") != "execution-component"
        or not isinstance(manifest.get("execution_component_sha256"), str)
        or _SHA256.fullmatch(str(manifest["execution_component_sha256"])) is None
        or not isinstance(fields, dict)
        or set(fields) != _SCOPE_FIELDS
        or digest != stable_hash(manifest)
    ):
        raise FrozenArtifactError("RQ1 component-scope manifest differs from its task")
    execution = _strict_artifact(path / "execution-component", "RQ1 packaged execution component")
    execution_sha = sha256_path(execution)
    if execution_sha != manifest["execution_component_sha256"]:
        raise FrozenArtifactError("RQ1 execution component bytes changed")
    if task.method == "M1" and execution_sha != _base_component_sha256(plan, "M1"):
        raise FrozenArtifactError("RQ1 M1 replay differs from the exact frozen base bytes")
    policy: AdaptivePolicySpec | None = None
    if task.method == "M3":
        policy_value = _load_json(path / "adaptive-policy.json", "RQ1 adaptive policy")
        policy_digest = policy_value.pop("policy_digest", None)
        if (
            policy_digest != stable_hash(policy_value)
            or manifest.get("adaptive_policy_path") != "adaptive-policy.json"
            or manifest.get("adaptive_policy_digest") != policy_digest
        ):
            raise FrozenArtifactError("RQ1 adaptive policy identity differs")
        policy = AdaptivePolicySpec.from_dict(policy_value)
        evidence_root = path / "fit-evidence"
        if (
            evidence_root.is_symlink()
            or not evidence_root.is_dir()
            or {item.name for item in evidence_root.iterdir()} != {"component", "threshold"}
        ):
            raise FrozenArtifactError("RQ1 fit evidence root inventory differs")
        fit_evidence = _load_fit_evidence(evidence_root / "threshold")
        (
            component_fit_datasets,
            fit_recipe,
            vector_activations,
            best_layers,
            fit_capture_attestation,
        ) = _load_m3_component_fit_evidence(evidence_root / "component")
        adaptive_component = load_confirmatory_adaptive_component(execution)
        calibrated = _calibrated_m3_policy(adaptive_component, policy, fit_evidence)
        if (
            set(fit_evidence.question_ids) != set(_expected_threshold_fit_ids(plan, task, stage))
            or len(fit_evidence.question_ids) != len(_expected_threshold_fit_ids(plan, task, stage))
            or manifest.get("fit_evidence_path") != "fit-evidence"
            or manifest.get("fit_evidence_sha256") != sha256_path(path / "fit-evidence")
            or calibrated.to_dict() != policy.to_dict()
        ):
            raise FrozenArtifactError("RQ1 threshold fit does not replay from its evidence")
    elif (
        manifest.get("adaptive_policy_path") is not None
        or manifest.get("adaptive_policy_digest") is not None
        or manifest.get("fit_evidence_path") is not None
        or manifest.get("fit_evidence_sha256") is not None
    ):
        raise FrozenArtifactError("RQ1 M1 component carries an adaptive policy")
    expected_fit_provenance: Mapping[str, Any]
    if task.method == "M3":
        assert policy is not None
        try:
            expected_fit_provenance = _m3_component_fit_provenance(
                plan,
                task,
                stage=stage,
                component=load_confirmatory_adaptive_component(execution),
                fit_datasets=component_fit_datasets,
                recipe=fit_recipe,
                vector_activations=vector_activations,
                best_layers=best_layers,
                capture_attestation=fit_capture_attestation,
                execution_public_key=policy.execution_public_key,
            )
        except DataValidationError as exc:
            raise FrozenArtifactError(f"RQ1 M3 fit replay failed: {exc}") from exc
    else:
        expected_fit_provenance = MappingProxyType(
            {
                "algorithm": "source-frozen-control-no-fit",
                "base_frozen_component_sha256": _base_component_sha256(plan, task.method),
            }
        )
    if manifest.get("fit_provenance") != dict(expected_fit_provenance):
        raise FrozenArtifactError("RQ1 component fit provenance does not replay")
    payloads = _rq1_component_field_payloads(
        execution,
        method=task.method,
        risk_threshold=float(manifest["risk_threshold"]),
        abstention_threshold=float(manifest["abstention_threshold"]),
        adaptive_policy=policy,
    )
    if task.method == "M3":
        assert policy is not None
        try:
            _validate_m3_source_refit_against_base(
                plan,
                execution_component=execution,
                adaptive_policy=policy,
                recipe=fit_recipe,
                vector_activations=vector_activations,
                source_payloads=payloads,
            )
        except DataValidationError as exc:
            raise FrozenArtifactError(f"RQ1 M3 source reference replay failed: {exc}") from exc
    observed: dict[str, str] = {}
    expected_field_files: set[str] = set()
    for name, payload in payloads.items():
        descriptor = fields[name]
        relative = f"fields/{name}.json"
        if (
            not isinstance(descriptor, Mapping)
            or set(descriptor) != {"path", "sha256"}
            or descriptor.get("path") != relative
            or not isinstance(descriptor.get("sha256"), str)
            or _SHA256.fullmatch(str(descriptor["sha256"])) is None
        ):
            raise FrozenArtifactError("RQ1 component field descriptor differs")
        field_path = path / relative
        if (
            _load_json(field_path, f"RQ1 component field {name}") != dict(payload)
            or sha256_path(field_path) != descriptor["sha256"]
        ):
            raise FrozenArtifactError("RQ1 component field does not replay")
        expected_field_files.add(f"{name}.json")
        observed[name] = str(descriptor["sha256"])
    if {item.name for item in (path / "fields").iterdir()} != expected_field_files:
        raise FrozenArtifactError("RQ1 component field inventory differs")
    if not isinstance(digest, str):
        raise FrozenArtifactError("RQ1 scope manifest lacks its digest")
    return RQ1ScopedComponent(
        directory=path,
        task_id=task.task_id,
        stage=stage,
        execution_component=execution,
        execution_component_sha256=execution_sha,
        adaptive_policy=policy,
        field_fingerprints=observed,
        scope_manifest_digest=digest,
        fingerprint=sha256_path(path),
    )


def _evaluation_questions_from_plan(
    plan: RobustnessDiagnosticPlan,
) -> Mapping[str, Question]:
    if plan.path is None:
        raise FrozenArtifactError("result verification requires a packaged robustness plan")
    root = plan.path / "sources"
    values = tuple(
        question
        for name in (
            "triviaqa-evaluation",
            "simpleqa_verified-evaluation",
            "aa_omniscience_public_600-evaluation",
        )
        for question in _questions_from_source(name, root / name)
    )
    if len({question.question_id for question in values}) != len(values):
        raise FrozenArtifactError("robustness evaluation question IDs repeat")
    return MappingProxyType({question.question_id: question for question in values})


def _rq1_questions_from_plan(plan: RobustnessDiagnosticPlan) -> Mapping[str, Question]:
    if plan.path is None:
        raise FrozenArtifactError("RQ1 verification requires a packaged robustness plan")
    values = _questions_from_source(
        "triviaqa-development",
        plan.path / "sources" / "triviaqa-development",
    )
    if len({question.question_id for question in values}) != len(values):
        raise FrozenArtifactError("RQ1 development question IDs repeat")
    return MappingProxyType({question.question_id: question for question in values})


def create_robustness_result_store(
    directory: str | Path,
    *,
    plan_path: str | Path,
    config_path: str | Path,
) -> RobustnessResultStore:
    """Create an empty append-only store that packages its exact plan and config."""

    destination = validate_active_study_artifact_paths({"robustness result store": directory})[
        "robustness result store"
    ]
    plan = verify_robustness_diagnostic_plan(plan_path)
    plan_source = _strict_artifact(plan_path, "robustness plan")
    config_source = _strict_artifact(config_path, "robustness config")
    if config_source.read_bytes() != (plan_source / "config.json").read_bytes():
        raise DataValidationError("result-store config differs from the packaged plan config")
    if destination.exists() or destination.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite result store: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        shutil.copytree(plan_source, stage / "plan")
        (stage / "prompt-results").mkdir()
        (stage / "rq1-results").mkdir()
        body = {
            "schema_version": 1,
            "plan_digest": plan.plan_digest,
            "config_digest": plan.body["config_digest"],
            "source_artifact_sha256": dict(plan.body["source_artifact_sha256"]),
        }
        (stage / "manifest.json").write_text(
            canonical_json({**body, "manifest_digest": stable_hash(body)}) + "\n",
            encoding="utf-8",
        )
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return open_robustness_result_store(destination)


def _prompt_result_files(directory: Path) -> tuple[Path, ...]:
    if directory.is_symlink() or not directory.is_dir():
        raise FrozenArtifactError("robustness result directory is invalid")
    values = tuple(sorted(directory.iterdir()))
    if any(
        path.is_symlink() or not path.is_file() or _PROMPT_RESULT_FILE.fullmatch(path.name) is None
        for path in values
    ):
        raise FrozenArtifactError("prompt robustness result inventory differs")
    return values


def _rq1_result_directories(directory: Path) -> tuple[Path, ...]:
    if directory.is_symlink() or not directory.is_dir():
        raise FrozenArtifactError("RQ1 result root is invalid")
    values = tuple(sorted(directory.iterdir()))
    for path in values:
        if (
            path.is_symlink()
            or not path.is_dir()
            or _RQ1_RESULT_DIRECTORY.fullmatch(path.name) is None
            or any(item.is_symlink() for item in path.rglob("*"))
            or {item.name for item in path.iterdir()} != {"result.json", "artifacts"}
            or not (path / "artifacts").is_dir()
            or {item.name for item in (path / "artifacts").iterdir()} != set(_RQ1_ARTIFACT_KEYS)
        ):
            raise FrozenArtifactError("RQ1 robustness result inventory differs")
    return values


def open_robustness_result_store(directory: str | Path) -> RobustnessResultStore:
    source = Path(directory).absolute().resolve(strict=False)
    allowed = {
        "manifest.json",
        "plan",
        "prompt-results",
        "rq1-results",
        "complete.json",
    }
    if (
        source.is_symlink()
        or not source.is_dir()
        or not {item.name for item in source.iterdir()} <= allowed
        or {
            "manifest.json",
            "plan",
            "prompt-results",
            "rq1-results",
        }
        - {item.name for item in source.iterdir()}
    ):
        raise FrozenArtifactError("robustness result-store inventory differs")
    manifest = _load_json(source / "manifest.json", "robustness result manifest")
    manifest_digest = manifest.pop("manifest_digest", None)
    if (
        set(manifest)
        != {
            "schema_version",
            "plan_digest",
            "config_digest",
            "source_artifact_sha256",
        }
        or manifest.get("schema_version") != 1
        or manifest_digest != stable_hash(manifest)
    ):
        raise FrozenArtifactError("robustness result manifest identity differs")
    plan = verify_robustness_diagnostic_plan(source / "plan")
    if (
        manifest["plan_digest"] != plan.plan_digest
        or manifest["config_digest"] != plan.body["config_digest"]
        or manifest["source_artifact_sha256"] != plan.body["source_artifact_sha256"]
    ):
        raise FrozenArtifactError("result store differs from its packaged plan")
    _prompt_result_files(source / "prompt-results")
    _rq1_result_directories(source / "rq1-results")
    return RobustnessResultStore(source, plan)


def _validate_generation_identity(
    record: GenerationRecord,
    *,
    plan: RobustnessDiagnosticPlan,
    task_id: str,
    prompt_id: str,
    prompt_text: str,
    method: str,
    question: Question,
    question_fingerprint: str,
    source_bindings: Mapping[str, Any],
    generation_seed: int,
    condition: EvaluationCondition,
    execution_component: Path | None,
    grader_bundle: ConfirmatoryGraderBundle,
    controller_prompt_id: str,
    controller_prompt_text: str,
    task_metadata: Mapping[str, Any] | None = None,
) -> None:
    expected_metadata = {
        **_robustness_signed_metadata(
            plan=plan,
            task_id=task_id,
            question_fingerprint=question_fingerprint,
            prompt_text=prompt_text,
            controller_prompt_id=controller_prompt_id,
            controller_prompt_text=controller_prompt_text,
            source_bindings=source_bindings,
            task_metadata=task_metadata,
        ),
        "source_question_sha256": question_source_fingerprint(question),
    }
    if (
        record.question_id != question.question_id
        or record.benchmark != question.benchmark
        or record.model_repository != _QWEN_REPOSITORY
        or record.model_revision != _QWEN_REVISION
        or record.runtime is not Runtime.VLLM
        or record.quantization != _QWEN_QUANTIZATION
        or record.system_prompt_id != prompt_id
        or record.steering_method != method
        or record.seed != generation_seed
        or any(record.metadata.get(name) != value for name, value in expected_metadata.items())
    ):
        raise DataValidationError("robustness generation identity differs from its task")
    condition.validate_record(record)
    fixed_component: ConfirmatoryFixedComponent | None = None
    if method == "M0":
        if execution_component is not None:
            raise DataValidationError("robustness M0 cannot bind an execution component")
    elif execution_component is None:
        raise DataValidationError("robustness intervention lacks an execution component")
    elif method == "M3":
        adaptive = load_confirmatory_adaptive_component(execution_component)
        if adaptive.fingerprint != condition.method_artifact_sha256:
            raise DataValidationError("robustness M3 component differs from its condition")
        try:
            controller = adaptive.controllers[controller_prompt_id]
        except KeyError as exc:
            raise DataValidationError(
                "robustness M3 component lacks the frozen controller prompt"
            ) from exc
        _validate_e8_adaptive_controller_record(
            record,
            condition=condition,
            controller=controller,
            controller_artifact_sha256=adaptive.fingerprint,
            controller_prompt_id=controller_prompt_id,
            controller_prompt_sha256=hashlib.sha256(
                controller_prompt_text.encode("utf-8")
            ).hexdigest(),
            runtime_identity=grader_bundle.runtime_attestation["runtime_identity"],
        )
    else:
        fixed_component = load_confirmatory_fixed_component(execution_component)
        if fixed_component.fingerprint != condition.method_artifact_sha256:
            raise DataValidationError("robustness fixed component differs from its condition")
    validate_confirmatory_execution_receipt(
        record,
        condition,
        execution_public_key=grader_bundle.scorer.execution_public_key,
        fixed_component=fixed_component,
        runtime_identity=grader_bundle.runtime_attestation["runtime_identity"],
    )


def _robustness_signed_metadata(
    *,
    plan: RobustnessDiagnosticPlan,
    task_id: str,
    question_fingerprint: str,
    prompt_text: str,
    controller_prompt_id: str,
    controller_prompt_text: str,
    source_bindings: Mapping[str, Any],
    task_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "robustness_plan_digest": plan.plan_digest,
        "robustness_task_id": task_id,
        "robustness_question_fingerprint": question_fingerprint,
        "robustness_prompt_text_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        "robustness_controller_prompt_id": controller_prompt_id,
        "robustness_controller_prompt_sha256": hashlib.sha256(
            controller_prompt_text.encode("utf-8")
        ).hexdigest(),
        "frozen_component_selection_sha256": source_bindings["frozen-component-selection"],
        "frozen_evaluation_scripts_sha256": source_bindings["frozen-evaluation-scripts"],
        "frozen_graders_sha256": source_bindings["frozen-graders"],
        "eligible_for_component_selection": False,
        **dict(task_metadata or {}),
    }


def robustness_evaluation_condition(
    plan: RobustnessDiagnosticPlan,
    *,
    benchmark: str,
    partition: str,
    prompt_id: str,
    prompt_text: str,
    method: str,
    seed: int,
    execution_component: str | Path | None = None,
    adaptive_policy: AdaptivePolicySpec | None = None,
) -> EvaluationCondition:
    """Build the exact signed condition for a frozen or fold-adapted task."""

    if plan.path is None:
        raise FrozenArtifactError("robustness condition requires a packaged plan")
    selection = plan.path / "sources" / "frozen-component-selection"
    manifest = _load_json(selection / "manifest.json", "component selection")
    study_digest = manifest.get("study_protocol_digest")
    if not isinstance(study_digest, str) or _SHA256.fullmatch(study_digest) is None:
        raise FrozenArtifactError("component selection lacks a study identity")
    artifact_sha: str | None = None
    layer = None
    site = None
    token_scope = None
    alpha = 0.0
    sparsity = None
    policy = adaptive_policy
    if method != "M0":
        if execution_component is None:
            descriptors = manifest.get("components")
            matches = (
                [
                    value
                    for value in descriptors
                    if isinstance(value, Mapping)
                    and value.get("model_name") == "qwen3.6-27b-nvfp4"
                    and value.get("method") == method
                ]
                if isinstance(descriptors, list)
                else []
            )
            if len(matches) != 1:
                raise FrozenArtifactError("robustness condition component is not unique")
            descriptor = matches[0]
            artifact = selection / str(descriptor["component_path"]) / "artifact"
            artifact_sha = str(descriptor["artifact_sha256"])
            policy_value = descriptor.get("adaptive_policy")
            if policy_value is not None:
                if not isinstance(policy_value, Mapping):
                    raise FrozenArtifactError("robustness adaptive policy is invalid")
                policy = AdaptivePolicySpec.from_dict(policy_value)
        else:
            if method not in {"M1", "M3"}:
                raise DataValidationError(
                    "only RQ1 M1/M3 conditions may override the frozen component"
                )
            artifact = _strict_artifact(
                execution_component, "robustness adapted execution component"
            )
            artifact_sha = sha256_path(artifact)
        if method == "M3":
            adaptive = load_confirmatory_adaptive_component(artifact)
            if (
                policy is None
                or policy.controller_artifact_sha256 != adaptive.fingerprint
                or adaptive.fingerprint != artifact_sha
                or adaptive.model_name != "qwen3.6-27b-nvfp4"
                or adaptive.model_repository != _QWEN_REPOSITORY
                or adaptive.model_revision != _QWEN_REVISION
                or adaptive.runtime is not Runtime.VLLM
                or adaptive.quantization != _QWEN_QUANTIZATION
                or adaptive.model_num_layers != 64
            ):
                raise FrozenArtifactError("robustness M3 condition lacks its adaptive policy")
        else:
            fixed = load_confirmatory_fixed_component(artifact)
            if fixed.method != method or fixed.fingerprint != artifact_sha:
                raise FrozenArtifactError("robustness fixed component identity differs")
            layer = fixed.layer
            site = fixed.site
            token_scope = fixed.token_scope
            alpha = fixed.standardized_alpha
            sparsity = fixed.sparsity
    elif execution_component is not None or policy is not None:
        raise DataValidationError("robustness M0 cannot receive component overrides")
    return EvaluationCondition(
        phase=ExperimentPhase.E9,
        benchmark=benchmark,
        partition=partition,
        model_name="qwen3.6-27b-nvfp4",
        model_repository=_QWEN_REPOSITORY,
        model_revision=_QWEN_REVISION,
        runtime=Runtime.VLLM,
        quantization=_QWEN_QUANTIZATION,
        model_num_layers=64,
        system_prompt_id=prompt_id,
        prompt_template_sha256=hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        steering_method=method,
        method_artifact_sha256=artifact_sha,
        layer=layer,
        site=site,
        token_scope=token_scope,
        alpha=alpha,
        sparsity=sparsity,
        seed=seed,
        study_protocol_digest=study_digest,
        adaptive_policy=policy,
    )


def _canonical_prompts_from_plan(
    plan: RobustnessDiagnosticPlan,
) -> Mapping[str, PromptSpec]:
    if plan.path is None:
        raise FrozenArtifactError("robustness prompts require a packaged plan")
    prompts = {
        prompt.prompt_id: prompt
        for prompt in load_prompt_specs(plan.path / "sources" / "canonical-prompts")
    }
    if not {"P0-neutral", "P2-calibrated-abstention"} <= set(prompts):
        raise FrozenArtifactError("packaged robustness prompts lack P0/P2")
    return MappingProxyType(prompts)


def _frozen_execution_component(
    plan: RobustnessDiagnosticPlan,
    method: str,
) -> Path | None:
    if method == "M0":
        return None
    if plan.path is None:
        raise FrozenArtifactError("robustness component requires a packaged plan")
    selection = plan.path / "sources" / "frozen-component-selection"
    manifest = _load_json(selection / "manifest.json", "component selection")
    descriptors = manifest.get("components")
    matches = (
        [
            value
            for value in descriptors
            if isinstance(value, Mapping)
            and value.get("model_name") == "qwen3.6-27b-nvfp4"
            and value.get("method") == method
        ]
        if isinstance(descriptors, list)
        else []
    )
    if len(matches) != 1 or not isinstance(matches[0].get("component_path"), str):
        raise FrozenArtifactError("robustness method component is not uniquely frozen")
    artifact = _strict_artifact(
        selection / str(matches[0]["component_path"]) / "artifact",
        "robustness frozen execution component",
    )
    if sha256_path(artifact) != matches[0].get("artifact_sha256"):
        raise FrozenArtifactError("robustness frozen execution component changed")
    return artifact


def append_prompt_paraphrase_result(
    store: RobustnessResultStore,
    *,
    task: PromptParaphraseTask,
    question: Question,
    record: GenerationRecord,
    grader_bundle: ConfirmatoryGraderBundle,
) -> PromptParaphraseResult:
    """Validate and append one exactly graded prompt-paraphrase generation."""

    store = open_robustness_result_store(store.directory)
    if (store.directory / "complete.json").exists():
        raise FrozenArtifactError("completed robustness results are immutable")
    prompt_tasks, _ = _task_maps(store.plan)
    if prompt_tasks.get(task.task_id) != task:
        raise DataValidationError("prompt task is not in the frozen plan")
    if _question_fingerprint(question) != task.question_fingerprint:
        raise DataValidationError("prompt task question differs from the frozen source")
    bindings = store.plan.body["source_artifact_sha256"]
    assert isinstance(bindings, Mapping)
    if grader_bundle.fingerprint != bindings["frozen-graders"]:
        raise DataValidationError("prompt result uses a different frozen grader bundle")
    prompt_section = store.plan.body["prompt_paraphrase"]
    assert isinstance(prompt_section, Mapping)
    canonical_prompts = _canonical_prompts_from_plan(store.plan)
    try:
        controller_prompt = canonical_prompts[task.base_prompt_id]
    except KeyError as exc:  # pragma: no cover - task schema fixes P0/P2
        raise FrozenArtifactError("prompt task base controller prompt is unavailable") from exc
    component = _frozen_execution_component(store.plan, task.method)
    condition = robustness_evaluation_condition(
        store.plan,
        benchmark=task.benchmark,
        partition=task.partition,
        prompt_id=task.prompt_id,
        prompt_text=task.prompt_text,
        method=task.method,
        seed=int(prompt_section["generation_seed"]),
    )
    _validate_generation_identity(
        record,
        plan=store.plan,
        task_id=task.task_id,
        prompt_id=task.prompt_id,
        prompt_text=task.prompt_text,
        method=task.method,
        question=question,
        question_fingerprint=task.question_fingerprint,
        source_bindings=bindings,
        generation_seed=int(prompt_section["generation_seed"]),
        condition=condition,
        execution_component=component,
        grader_bundle=grader_bundle,
        controller_prompt_id=task.base_prompt_id,
        controller_prompt_text=controller_prompt.text,
    )
    validate_confirmatory_factual_grade(record, question, grader_bundle=grader_bundle)
    result = PromptParaphraseResult.create(
        task_id=task.task_id,
        plan_digest=store.plan.plan_digest,
        record=record,
    )
    _write_once_json(
        store.directory / "prompt-results" / f"{task.task_id}.json",
        result.to_dict(),
    )
    return result


def execute_prompt_paraphrase_task(
    store: RobustnessResultStore,
    *,
    task: PromptParaphraseTask,
    question: Question,
    backend: NativeE9VllmBackend,
) -> PromptParaphraseResult:
    """Execute, sign, grade, and append one frozen prompt-paraphrase task."""

    store = open_robustness_result_store(store.directory)
    prompt_section = store.plan.body["prompt_paraphrase"]
    bindings = store.plan.body["source_artifact_sha256"]
    assert isinstance(prompt_section, Mapping)
    assert isinstance(bindings, Mapping)
    if backend.grader_bundle.fingerprint != bindings["frozen-graders"]:
        raise DataValidationError("robustness backend uses another grader bundle")
    if _question_fingerprint(question) != task.question_fingerprint:
        raise DataValidationError("prompt execution question differs from its frozen task")
    canonical = _canonical_prompts_from_plan(store.plan)
    controller_prompt = canonical[task.base_prompt_id]
    component = _frozen_execution_component(store.plan, task.method)
    seed = int(prompt_section["generation_seed"])
    condition = robustness_evaluation_condition(
        store.plan,
        benchmark=task.benchmark,
        partition=task.partition,
        prompt_id=task.prompt_id,
        prompt_text=task.prompt_text,
        method=task.method,
        seed=seed,
    )
    prompt = PromptSpec(
        prompt_id=task.prompt_id,
        text=task.prompt_text,
        permits_abstention=controller_prompt.permits_abstention,
        deployment_eligible=False,
    )
    record = backend.execute(
        condition=condition,
        question=question,
        prompt=prompt,
        component_artifact=component,
        controller_prompt_id=(task.base_prompt_id if task.method == "M3" else None),
        signed_metadata=_robustness_signed_metadata(
            plan=store.plan,
            task_id=task.task_id,
            question_fingerprint=task.question_fingerprint,
            prompt_text=task.prompt_text,
            controller_prompt_id=task.base_prompt_id,
            controller_prompt_text=controller_prompt.text,
            source_bindings=bindings,
        ),
    )
    return append_prompt_paraphrase_result(
        store,
        task=task,
        question=question,
        record=record,
        grader_bundle=backend.grader_bundle,
    )


def _question_set_digests(
    values: Mapping[str, Sequence[str]],
) -> dict[str, str]:
    return {name: stable_hash(list(identifiers)) for name, identifiers in values.items()}


def _base_component_sha256(
    plan: RobustnessDiagnosticPlan,
    method: str,
) -> str:
    return str(_base_component_descriptor(plan, method)["artifact_sha256"])


def _base_component_descriptor(
    plan: RobustnessDiagnosticPlan,
    method: str,
) -> Mapping[str, Any]:
    if plan.path is None:
        raise FrozenArtifactError("RQ1 component binding requires a packaged plan")
    manifest = _load_json(
        plan.path / "sources" / "frozen-component-selection" / "manifest.json",
        "frozen component selection",
    )
    descriptors = manifest.get("components")
    if not isinstance(descriptors, list):
        raise FrozenArtifactError("frozen component selection lacks descriptors")
    matches = [
        value
        for value in descriptors
        if isinstance(value, Mapping)
        and value.get("model_name") == "qwen3.6-27b-nvfp4"
        and value.get("method") == method
    ]
    if (
        len(matches) != 1
        or not isinstance(matches[0].get("artifact_sha256"), str)
        or _SHA256.fullmatch(matches[0]["artifact_sha256"]) is None
        or not isinstance(matches[0].get("component_path"), str)
    ):
        raise FrozenArtifactError("RQ1 base component is not uniquely frozen")
    return MappingProxyType(dict(matches[0]))


def _validate_m3_source_refit_against_base(
    plan: RobustnessDiagnosticPlan,
    *,
    execution_component: Path,
    adaptive_policy: AdaptivePolicySpec,
    recipe: M3FitRecipe,
    vector_activations: Mapping[HookKey, Tensor],
    source_payloads: Mapping[str, Mapping[str, Any]],
) -> None:
    descriptor = _base_component_descriptor(plan, "M3")
    assert plan.path is not None
    selection = plan.path / "sources" / "frozen-component-selection"
    base_path = _strict_artifact(
        selection / str(descriptor["component_path"]) / "artifact",
        "RQ1 frozen base M3 component",
    )
    if sha256_path(base_path) != descriptor["artifact_sha256"]:
        raise FrozenArtifactError("RQ1 frozen base M3 component bytes changed")
    policy_value = descriptor.get("adaptive_policy")
    if not isinstance(policy_value, Mapping):
        raise FrozenArtifactError("RQ1 frozen base M3 policy is missing")
    base_policy = AdaptivePolicySpec.from_dict(policy_value)
    base_component = load_confirmatory_adaptive_component(base_path)
    base_controller = base_component.controllers.get("P0-neutral")
    source_component = load_confirmatory_adaptive_component(execution_component)
    source_controller = source_component.controllers.get("P0-neutral")
    if base_controller is None or source_controller is None:
        raise FrozenArtifactError("RQ1 M3 reference lacks the P0 controller")
    base_payloads = _rq1_component_field_payloads(
        base_path,
        method="M3",
        risk_threshold=base_policy.release_risk_threshold,
        abstention_threshold=base_policy.abstention_probability_threshold,
        adaptive_policy=base_policy,
    )
    structural_fields = {
        "alpha_controller",
        "router_architecture",
        "candidate_layers",
        "candidate_sites",
        "token_scopes",
        "sparsity",
        "alpha_policy_family",
        "likely_unknown_threshold",
        "execution_public_key",
    }
    registered = plan.body["rq1_generalization"]["m3_refit_hyperparameters"]
    recipe_hyperparameters = {
        "vector_seed": recipe.vector_seed,
        "minimum_class_count": recipe.minimum_class_count,
        "router_seed": recipe.router_seed,
        "router_hidden_width": recipe.router_hidden_width,
        "router_epochs": recipe.router_epochs,
        "distance_temperature": recipe.distance_temperature,
        "risk_hidden_width": recipe.risk_hidden_width,
        "risk_epochs": recipe.risk_epochs,
        "risk_learning_rate": recipe.risk_learning_rate,
        "risk_weight_decay": recipe.risk_weight_decay,
        "risk_class_balanced": recipe.risk_class_balanced,
        "risk_seed": recipe.risk_seed,
        "calibration_kind": recipe.calibration_kind.value,
        "layer_seed": recipe.layer_seed,
        "layer_epochs": recipe.layer_epochs,
    }
    expected_fixed_layer = base_controller.fixed_layer
    expected_candidate_layers = (
        base_controller.layer_selector.candidate_layers
        if base_controller.layer_selector is not None
        else ()
    )
    expected_layer_router_kind = (
        base_controller.layer_selector.router.kind
        if base_controller.layer_selector is not None
        else None
    )
    expected_calibration_kind = (
        CalibrationKind.TEMPERATURE
        if isinstance(base_controller.risk_probe.calibrator, TemperatureCalibrator)
        else CalibrationKind.ISOTONIC
    )
    if (
        any(source_payloads[name] != base_payloads[name] for name in structural_fields)
        or dict(registered) != recipe_hyperparameters
        or recipe.cluster_count != base_controller.vector_bank.cluster_count
        or recipe.vector_source_artifact_sha256
        != base_controller.vector_bank.source_artifact_sha256
        or recipe.router_kind is not base_controller.vector_router.kind
        or recipe.risk_probe_kind is not base_controller.risk_probe.state.kind
        or recipe.calibration_kind is not expected_calibration_kind
        or recipe.alpha_controller != base_controller.alpha_controller
        or recipe.fixed_layer != expected_fixed_layer
        or recipe.candidate_layers != expected_candidate_layers
        or recipe.layer_router_kind is not expected_layer_router_kind
        or set(vector_activations) != set(base_controller.vector_bank.directions)
        or adaptive_policy.alpha_max != base_policy.alpha_max
        or adaptive_policy.alpha_beta != base_policy.alpha_beta
        or adaptive_policy.alpha_mode != base_policy.alpha_mode
        or adaptive_policy.alpha_risk_threshold != base_policy.alpha_risk_threshold
        or adaptive_policy.likely_unknown_risk_threshold
        != base_policy.likely_unknown_risk_threshold
        or adaptive_policy.execution_public_key != base_policy.execution_public_key
    ):
        raise DataValidationError(
            "RQ1 M3 source refit differs from the frozen base architecture or recipe"
        )


def _expected_rq1_fit_receipt(
    plan: RobustnessDiagnosticPlan,
    task: RQ1GeneralizationTask,
    source: RQ1ScopedComponent,
    adapted: RQ1ScopedComponent,
) -> dict[str, Any]:
    question_sets = rq1_task_question_sets(plan, task)
    return {
        "schema_version": 2,
        "plan_digest": plan.plan_digest,
        "task_id": task.task_id,
        "adaptation_regime": task.adaptation_regime,
        "training_prompt_id": task.training_prompt_id,
        "evaluation_prompt_id": task.evaluation_prompt_id,
        "base_frozen_component_sha256": _base_component_sha256(plan, task.method),
        "question_sets": {name: list(identifiers) for name, identifiers in question_sets.items()},
        "question_set_digests": _question_set_digests(question_sets),
        "held_out_evaluation_used_for_fitting": False,
        "simpleqa_aa_e9_e10_used_for_fitting": False,
        "source_scope_manifest_digest": source.scope_manifest_digest,
        "adapted_scope_manifest_digest": adapted.scope_manifest_digest,
        "source_field_fingerprints": dict(source.field_fingerprints),
        "adapted_field_fingerprints": dict(adapted.field_fingerprints),
        "source_component_sha256": source.fingerprint,
        "adapted_component_sha256": adapted.fingerprint,
        "source_execution_component_sha256": source.execution_component_sha256,
        "adapted_execution_component_sha256": adapted.execution_component_sha256,
    }


def write_rq1_fit_receipt(
    path: str | Path,
    *,
    plan: RobustnessDiagnosticPlan,
    task: RQ1GeneralizationTask,
    source_component: str | Path,
    adapted_component: str | Path,
) -> str:
    """Write the sole accepted leakage receipt from two replayed scoped components."""

    destination = validate_active_study_artifact_paths({"RQ1 fit receipt": path})["RQ1 fit receipt"]
    source = load_rq1_scoped_component(
        source_component,
        plan=plan,
        task=task,
        stage="source-fit",
    )
    adapted = load_rq1_scoped_component(
        adapted_component,
        plan=plan,
        task=task,
        stage="held-out-adaptation",
    )
    _validate_rq1_regime(task, source, adapted)
    body = _expected_rq1_fit_receipt(plan, task, source, adapted)
    _write_once_json(destination, body)
    return sha256_path(destination)


def execute_rq1_evaluation_records(
    path: str | Path,
    *,
    store: RobustnessResultStore,
    task: RQ1GeneralizationTask,
    questions_by_id: Mapping[str, Question],
    adapted_component: str | Path,
    backend: NativeE9VllmBackend,
) -> str:
    """Run the held-out fold with the exact adapted component and signed evidence."""

    store = open_robustness_result_store(store.directory)
    destination = validate_active_study_artifact_paths({"RQ1 evaluation records": path})[
        "RQ1 evaluation records"
    ]
    _, tasks = _task_maps(store.plan)
    if tasks.get(task.task_id) != task:
        raise DataValidationError("RQ1 execution task is not in the frozen plan")
    scoped = load_rq1_scoped_component(
        adapted_component,
        plan=store.plan,
        task=task,
        stage="held-out-adaptation",
    )
    expected_ids = rq1_task_question_sets(store.plan, task)["held_out_evaluation"]
    if set(questions_by_id) != set(expected_ids):
        raise DataValidationError("RQ1 execution questions differ from the held-out fold")
    bindings = store.plan.body["source_artifact_sha256"]
    rq1_section = store.plan.body["rq1_generalization"]
    assert isinstance(bindings, Mapping)
    assert isinstance(rq1_section, Mapping)
    if backend.grader_bundle.fingerprint != bindings["frozen-graders"]:
        raise DataValidationError("RQ1 backend uses another grader bundle")
    prompts = _canonical_prompts_from_plan(store.plan)
    evaluation_prompt = prompts[task.evaluation_prompt_id]
    controller_prompt = prompts[task.training_prompt_id]
    condition = robustness_evaluation_condition(
        store.plan,
        benchmark="triviaqa",
        partition="T-dev",
        prompt_id=task.evaluation_prompt_id,
        prompt_text=evaluation_prompt.text,
        method=task.method,
        seed=17,
        execution_component=scoped.execution_component,
        adaptive_policy=scoped.adaptive_policy,
    )
    assignment_fingerprints = {
        str(row["question_id"]): str(row["question_fingerprint"])
        for row in rq1_section["assignments"]
    }
    task_metadata = {
        "rq1_adaptation_regime": task.adaptation_regime,
        "rq1_adapted_component_sha256": scoped.fingerprint,
        "rq1_scope_manifest_digest": scoped.scope_manifest_digest,
    }

    def records() -> Iterator[GenerationRecord]:
        for question_id in expected_ids:
            question = questions_by_id[question_id]
            fingerprint = _question_fingerprint(question)
            if (
                question.benchmark != "triviaqa"
                or question.split != "T-dev"
                or assignment_fingerprints.get(question_id) != fingerprint
            ):
                raise DataValidationError("RQ1 held-out question source changed")
            yield backend.execute(
                condition=condition,
                question=question,
                prompt=evaluation_prompt,
                component_artifact=scoped.execution_component,
                controller_prompt_id=(task.training_prompt_id if task.method == "M3" else None),
                signed_metadata=_robustness_signed_metadata(
                    plan=store.plan,
                    task_id=task.task_id,
                    question_fingerprint=fingerprint,
                    prompt_text=evaluation_prompt.text,
                    controller_prompt_id=task.training_prompt_id,
                    controller_prompt_text=controller_prompt.text,
                    source_bindings=bindings,
                    task_metadata=task_metadata,
                ),
            )

    try:
        count = write_generation_records(destination, records())
    except FileExistsError as exc:
        raise FrozenArtifactError(
            f"refusing to overwrite RQ1 evaluation records: {destination}"
        ) from exc
    if count != len(expected_ids):  # pragma: no cover - generator cardinality is exact
        raise FrozenArtifactError("RQ1 execution wrote an incomplete held-out fold")
    return sha256_path(destination)


def _validate_rq1_regime(
    task: RQ1GeneralizationTask,
    source: RQ1ScopedComponent,
    adapted: RQ1ScopedComponent,
) -> None:
    source_fields = source.field_fingerprints
    adapted_fields = adapted.field_fingerprints
    if task.adaptation_regime == "source-frozen-control":
        if (
            task.method != "M1"
            or source.execution_component_sha256 != adapted.execution_component_sha256
            or dict(source.field_fingerprints) != dict(adapted.field_fingerprints)
        ):
            raise DataValidationError("RQ1 frozen static control changed during adaptation")
        return
    frozen_fields = (
        {
            "vector_bank",
            "router",
            "directions",
            "risk_probe",
            "layer_selector",
            "alpha_controller",
            "router_architecture",
            "candidate_layers",
            "candidate_sites",
            "token_scopes",
            "sparsity",
            "alpha_policy_family",
            "likely_unknown_threshold",
            "execution_public_key",
        }
        if task.adaptation_regime == "calibration-only"
        else {
            "alpha_controller",
            "risk_probe",
            "layer_selector",
            "router_architecture",
            "candidate_layers",
            "candidate_sites",
            "token_scopes",
            "sparsity",
            "alpha_policy_family",
            "likely_unknown_threshold",
            "execution_public_key",
        }
    )
    changed_required = (
        set()
        if task.adaptation_regime == "calibration-only"
        else {
            "vector_bank",
            "directions",
        }
    )
    if (
        any(source_fields[name] != adapted_fields[name] for name in frozen_fields)
        or any(source_fields[name] == adapted_fields[name] for name in changed_required)
        or (
            task.adaptation_regime == "calibration-only"
            and source.execution_component_sha256 != adapted.execution_component_sha256
        )
        or (
            task.adaptation_regime == "full-vector-bank-relearning"
            and source.execution_component_sha256 == adapted.execution_component_sha256
        )
    ):
        raise DataValidationError("RQ1 adaptation changed fields outside its regime")


def _rq1_record_metrics(records: Sequence[GenerationRecord]) -> Mapping[str, float]:
    if not records:
        raise DataValidationError("RQ1 metrics require held-out records")
    count = len(records)
    attempted = tuple(
        record
        for record in records
        if record.outcome not in {Outcome.ABSTENTION, Outcome.UNSCORABLE}
    )
    incorrect = sum(record.outcome is Outcome.INCORRECT for record in records)
    return MappingProxyType(
        {
            "correct_rate": sum(record.outcome is Outcome.CORRECT for record in records) / count,
            "attempt_rate": len(attempted) / count,
            "hallucination_risk": incorrect / len(attempted) if attempted else 0.0,
            "abstention_rate": sum(record.outcome is Outcome.ABSTENTION for record in records)
            / count,
            "mean_intervention_alpha": sum(abs(record.alpha) for record in records) / count,
            "mean_latency_seconds": sum(record.generation_latency_seconds for record in records)
            / count,
            "triviaqa_exact_match": sum(
                float(record.metadata["official_exact_match"]) for record in records
            )
            / count,
            "triviaqa_token_f1": sum(
                float(record.metadata["official_token_f1"]) for record in records
            )
            / count,
        }
    )


def _build_rq1_generalization_result(
    store: RobustnessResultStore,
    *,
    task: RQ1GeneralizationTask,
    questions_by_id: Mapping[str, Question],
    grader_bundle: ConfirmatoryGraderBundle,
    artifacts: Mapping[str, str | Path],
) -> RQ1GeneralizationResult:
    """Validate exact fitting receipts and held-out records for one RQ1 task."""

    store = open_robustness_result_store(store.directory)
    _, rq1_tasks = _task_maps(store.plan)
    if rq1_tasks.get(task.task_id) != task or set(artifacts) != _RQ1_ARTIFACT_KEYS:
        raise DataValidationError("RQ1 task or artifact inventory differs")
    paths = {name: _strict_artifact(path, f"RQ1 {name}") for name, path in artifacts.items()}
    fingerprints = {name: sha256_path(path) for name, path in paths.items()}
    locations = {name: str(path) for name, path in paths.items()}
    question_sets = rq1_task_question_sets(store.plan, task)
    set_digests = _question_set_digests(question_sets)
    source = load_rq1_scoped_component(
        paths["source_component"],
        plan=store.plan,
        task=task,
        stage="source-fit",
    )
    adapted = load_rq1_scoped_component(
        paths["adapted_component"],
        plan=store.plan,
        task=task,
        stage="held-out-adaptation",
    )
    _validate_rq1_regime(task, source, adapted)
    receipt = _load_json(paths["fit_receipt"], "RQ1 fit receipt")
    expected_receipt = _expected_rq1_fit_receipt(store.plan, task, source, adapted)
    if receipt != expected_receipt:
        raise DataValidationError("RQ1 fitting receipt differs from the frozen task")
    bindings = store.plan.body["source_artifact_sha256"]
    assert isinstance(bindings, Mapping)
    if grader_bundle.fingerprint != bindings["frozen-graders"]:
        raise DataValidationError("RQ1 result uses a different frozen grader bundle")
    expected_ids = question_sets["held_out_evaluation"]
    records = tuple(read_generation_records(paths["evaluation_records"]))
    if (
        len(records) != len(expected_ids)
        or tuple(record.question_id for record in records) != expected_ids
        or set(questions_by_id) != set(expected_ids)
    ):
        raise DataValidationError("RQ1 evaluation records differ from the held-out fold")
    rq1_section = store.plan.body["rq1_generalization"]
    assert isinstance(rq1_section, Mapping)
    if store.plan.path is None:  # pragma: no cover - open store packages a plan
        raise FrozenArtifactError("RQ1 result store has no packaged plan")
    prompt_texts = {
        prompt_id: prompt.text
        for prompt_id, prompt in _canonical_prompts_from_plan(store.plan).items()
    }
    try:
        evaluation_prompt_text = prompt_texts[task.evaluation_prompt_id]
    except KeyError as exc:
        raise FrozenArtifactError("RQ1 evaluation prompt is unavailable") from exc
    for record in records:
        question = questions_by_id[record.question_id]
        fingerprint = _question_fingerprint(question)
        assignment = next(
            row for row in rq1_section["assignments"] if row["question_id"] == record.question_id
        )
        if fingerprint != assignment["question_fingerprint"]:
            raise DataValidationError("RQ1 evaluation question source changed")
        condition = robustness_evaluation_condition(
            store.plan,
            benchmark="triviaqa",
            partition="T-dev",
            prompt_id=task.evaluation_prompt_id,
            prompt_text=evaluation_prompt_text,
            method=task.method,
            seed=17,
            execution_component=adapted.execution_component,
            adaptive_policy=adapted.adaptive_policy,
        )
        _validate_generation_identity(
            record,
            plan=store.plan,
            task_id=task.task_id,
            prompt_id=task.evaluation_prompt_id,
            prompt_text=evaluation_prompt_text,
            method=task.method,
            question=question,
            question_fingerprint=fingerprint,
            source_bindings=bindings,
            generation_seed=17,
            condition=condition,
            execution_component=adapted.execution_component,
            grader_bundle=grader_bundle,
            controller_prompt_id=task.training_prompt_id,
            controller_prompt_text=prompt_texts[task.training_prompt_id],
            task_metadata={
                "rq1_adaptation_regime": task.adaptation_regime,
                "rq1_adapted_component_sha256": adapted.fingerprint,
                "rq1_scope_manifest_digest": adapted.scope_manifest_digest,
            },
        )
        validate_confirmatory_factual_grade(record, question, grader_bundle=grader_bundle)
    normalized_metrics = _rq1_record_metrics(records)
    return RQ1GeneralizationResult.create(
        task_id=task.task_id,
        plan_digest=store.plan.plan_digest,
        question_set_digests=set_digests,
        artifact_locations=locations,
        artifact_fingerprints=fingerprints,
        evaluation_record_count=len(records),
        metrics=normalized_metrics,
    )


def append_rq1_generalization_result(
    store: RobustnessResultStore,
    *,
    task: RQ1GeneralizationTask,
    questions_by_id: Mapping[str, Question],
    grader_bundle: ConfirmatoryGraderBundle,
    artifacts: Mapping[str, str | Path],
) -> RQ1GeneralizationResult:
    """Fully replay, recursively package, and append one RQ1 task result."""

    store = open_robustness_result_store(store.directory)
    if (store.directory / "complete.json").exists():
        raise FrozenArtifactError("completed robustness results are immutable")
    _build_rq1_generalization_result(
        store,
        task=task,
        questions_by_id=questions_by_id,
        grader_bundle=grader_bundle,
        artifacts=artifacts,
    )
    destination = store.directory / "rq1-results" / task.task_id
    if destination.exists() or destination.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite RQ1 result: {task.task_id}")
    stage = Path(tempfile.mkdtemp(prefix=f".{task.task_id}.stage-", dir=destination.parent))
    try:
        artifact_root = stage / "artifacts"
        artifact_root.mkdir()
        packaged: dict[str, Path] = {}
        for name, raw_path in artifacts.items():
            source = _strict_artifact(raw_path, f"RQ1 {name}")
            target = artifact_root / name
            if source.is_dir():
                shutil.copytree(source, target)
            else:
                shutil.copyfile(source, target)
            packaged[name] = target
        validated = _build_rq1_generalization_result(
            store,
            task=task,
            questions_by_id=questions_by_id,
            grader_bundle=grader_bundle,
            artifacts=packaged,
        )
        relative_locations = {
            name: f"rq1-results/{task.task_id}/artifacts/{name}" for name in _RQ1_ARTIFACT_KEYS
        }
        result = RQ1GeneralizationResult.create(
            task_id=validated.task_id,
            plan_digest=validated.plan_digest,
            question_set_digests=validated.question_set_digests,
            artifact_locations=relative_locations,
            artifact_fingerprints=validated.artifact_fingerprints,
            evaluation_record_count=validated.evaluation_record_count,
            metrics=validated.metrics,
        )
        (stage / "result.json").write_text(
            canonical_json(result.to_dict()) + "\n", encoding="utf-8"
        )
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return result


def _validate_rq1_artifact_bindings(
    result: RQ1GeneralizationResult,
    *,
    store_root: Path,
) -> Mapping[str, Path]:
    if (
        set(result.artifact_locations) != _RQ1_ARTIFACT_KEYS
        or set(result.artifact_fingerprints) != _RQ1_ARTIFACT_KEYS
        or set(result.question_set_digests)
        != {
            "source_fit",
            "source_calibration",
            "held_out_adaptation",
            "held_out_evaluation",
        }
        or any(_SHA256.fullmatch(value) is None for value in result.question_set_digests.values())
        or any(_SHA256.fullmatch(value) is None for value in result.artifact_fingerprints.values())
        or type(result.evaluation_record_count) is not int
        or result.evaluation_record_count <= 0
        or not result.metrics
        or any(not math.isfinite(value) for value in result.metrics.values())
    ):
        raise FrozenArtifactError("RQ1 result identities are invalid")
    paths: dict[str, Path] = {}
    for name, location in result.artifact_locations.items():
        relative = Path(location)
        expected = Path("rq1-results") / result.task_id / "artifacts" / name
        if relative.is_absolute() or ".." in relative.parts or relative != expected:
            raise FrozenArtifactError("RQ1 artifact location is not store-relative")
        path = _strict_artifact(store_root / relative, f"RQ1 {name}")
        if sha256_path(path) != (result.artifact_fingerprints[name]):
            raise FrozenArtifactError(f"RQ1 external artifact changed: {name}")
        paths[name] = path
    return MappingProxyType(paths)


def robustness_result_progress(store: RobustnessResultStore) -> Mapping[str, tuple[int, int]]:
    store = open_robustness_result_store(store.directory)
    prompt_tasks, rq1_tasks = _task_maps(store.plan)
    return MappingProxyType(
        {
            "prompt_paraphrase": (
                len(_prompt_result_files(store.directory / "prompt-results")),
                len(prompt_tasks),
            ),
            "rq1_generalization": (
                len(_rq1_result_directories(store.directory / "rq1-results")),
                len(rq1_tasks),
            ),
        }
    )


def verify_robustness_result_store(
    directory: str | Path,
    *,
    require_complete: bool = False,
) -> RobustnessResultStore:
    """Replay result schemas, task membership, external hashes, and completion."""

    store = open_robustness_result_store(directory)
    prompt_tasks, rq1_tasks = _task_maps(store.plan)
    if store.plan.path is None:  # pragma: no cover - open always packages a path
        raise FrozenArtifactError("robustness result store has no packaged plan")
    grader_bundle = validate_confirmatory_grader_bundle(
        store.plan.path / "sources" / "frozen-graders"
    )
    questions = _evaluation_questions_from_plan(store.plan)
    rq1_questions = _rq1_questions_from_plan(store.plan)
    bindings = store.plan.body["source_artifact_sha256"]
    prompt_section = store.plan.body["prompt_paraphrase"]
    assert isinstance(bindings, Mapping)
    assert isinstance(prompt_section, Mapping)
    canonical_prompts = _canonical_prompts_from_plan(store.plan)
    prompt_results: dict[str, str] = {}
    for path in _prompt_result_files(store.directory / "prompt-results"):
        prompt_result = PromptParaphraseResult.from_dict(_load_json(path, "prompt result"))
        prompt_task = prompt_tasks.get(prompt_result.task_id)
        if (
            prompt_task is None
            or path.name != f"{prompt_result.task_id}.json"
            or prompt_result.plan_digest != store.plan.plan_digest
        ):
            raise FrozenArtifactError("prompt result is detached from its task")
        try:
            question = questions[prompt_task.question_id]
            controller_prompt = canonical_prompts[prompt_task.base_prompt_id]
            component = _frozen_execution_component(store.plan, prompt_task.method)
            condition = robustness_evaluation_condition(
                store.plan,
                benchmark=prompt_task.benchmark,
                partition=prompt_task.partition,
                prompt_id=prompt_task.prompt_id,
                prompt_text=prompt_task.prompt_text,
                method=prompt_task.method,
                seed=int(prompt_section["generation_seed"]),
            )
            _validate_generation_identity(
                prompt_result.record,
                plan=store.plan,
                task_id=prompt_task.task_id,
                prompt_id=prompt_task.prompt_id,
                prompt_text=prompt_task.prompt_text,
                method=prompt_task.method,
                question=question,
                question_fingerprint=prompt_task.question_fingerprint,
                source_bindings=bindings,
                generation_seed=int(prompt_section["generation_seed"]),
                condition=condition,
                execution_component=component,
                grader_bundle=grader_bundle,
                controller_prompt_id=prompt_task.base_prompt_id,
                controller_prompt_text=controller_prompt.text,
            )
            validate_confirmatory_factual_grade(
                prompt_result.record,
                question,
                grader_bundle=grader_bundle,
            )
        except (KeyError, DataValidationError) as exc:
            raise FrozenArtifactError(f"prompt result does not replay: {exc}") from exc
        prompt_results[prompt_result.task_id] = prompt_result.result_digest
    rq1_results: dict[str, str] = {}
    for path in _rq1_result_directories(store.directory / "rq1-results"):
        rq1_result = RQ1GeneralizationResult.from_dict(
            _load_json(path / "result.json", "RQ1 result")
        )
        rq1_task = rq1_tasks.get(rq1_result.task_id)
        if (
            rq1_task is None
            or path.name != rq1_result.task_id
            or rq1_result.plan_digest != store.plan.plan_digest
            or rq1_result.question_set_digests
            != _question_set_digests(rq1_task_question_sets(store.plan, rq1_task))
        ):
            raise FrozenArtifactError("RQ1 result is detached from its task")
        artifacts = _validate_rq1_artifact_bindings(
            rq1_result,
            store_root=store.directory,
        )
        question_ids = rq1_task_question_sets(store.plan, rq1_task)["held_out_evaluation"]
        try:
            replayed = _build_rq1_generalization_result(
                store,
                task=rq1_task,
                questions_by_id={
                    question_id: rq1_questions[question_id] for question_id in question_ids
                },
                grader_bundle=grader_bundle,
                artifacts=artifacts,
            )
        except (KeyError, DataValidationError) as exc:
            raise FrozenArtifactError(f"RQ1 result does not replay: {exc}") from exc
        expected_rq1 = RQ1GeneralizationResult.create(
            task_id=replayed.task_id,
            plan_digest=replayed.plan_digest,
            question_set_digests=replayed.question_set_digests,
            artifact_locations=rq1_result.artifact_locations,
            artifact_fingerprints=replayed.artifact_fingerprints,
            evaluation_record_count=replayed.evaluation_record_count,
            metrics=replayed.metrics,
        )
        if rq1_result.to_dict() != expected_rq1.to_dict():
            raise FrozenArtifactError("RQ1 result summary differs from record replay")
        rq1_results[rq1_result.task_id] = rq1_result.result_digest
    completion_path = store.directory / "complete.json"
    if completion_path.exists():
        completion = _load_json(completion_path, "robustness completion")
        digest = completion.pop("completion_digest", None)
        expected_completion = {
            "schema_version": 1,
            "plan_digest": store.plan.plan_digest,
            "prompt_result_count": len(prompt_tasks),
            "prompt_result_set_digest": stable_hash(dict(sorted(prompt_results.items()))),
            "rq1_result_count": len(rq1_tasks),
            "rq1_result_set_digest": stable_hash(dict(sorted(rq1_results.items()))),
            "eligible_for_component_selection": False,
        }
        if (
            set(prompt_results) != set(prompt_tasks)
            or set(rq1_results) != set(rq1_tasks)
            or completion != expected_completion
            or digest != stable_hash(expected_completion)
        ):
            raise FrozenArtifactError("robustness completion differs from its results")
    elif require_complete:
        raise FrozenArtifactError("robustness result store is incomplete")
    return store


def load_verified_prompt_paraphrase_records(
    directory: str | Path,
) -> tuple[VerifiedPromptParaphraseRecord, ...]:
    """Return prompt-paraphrase records only after replaying the complete store.

    The task metadata is returned with each generation so downstream analysis can
    aggregate by the canonical prompt rather than treating paraphrase identifiers
    as unrelated prompts.
    """

    store = verify_robustness_result_store(directory, require_complete=True)
    prompt_tasks, _ = _task_maps(store.plan)
    values: list[VerifiedPromptParaphraseRecord] = []
    for path in _prompt_result_files(store.directory / "prompt-results"):
        result = PromptParaphraseResult.from_dict(_load_json(path, "prompt result"))
        task = prompt_tasks[result.task_id]
        values.append(
            VerifiedPromptParaphraseRecord(
                task_id=task.task_id,
                benchmark=task.benchmark,
                base_prompt_id=task.base_prompt_id,
                paraphrase_prompt_id=task.prompt_id,
                method=task.method,
                record=result.record,
            )
        )
    return tuple(sorted(values, key=lambda value: value.task_id))


def load_verified_rq1_generalization_results(
    directory: str | Path,
) -> Mapping[str, RQ1GeneralizationResult]:
    """Return every preregistered RQ1 result after full store replay.

    The mapping key retains the scientific comparison identity that is otherwise
    present only in the frozen diagnostic plan.  Downstream reporting can
    therefore distinguish semantic folds, prompt transfer, calibration-only
    adaptation, and full vector-bank relearning without trusting filenames.
    """

    store = verify_robustness_result_store(directory, require_complete=True)
    _prompt_tasks, rq1_tasks = _task_maps(store.plan)
    results: dict[str, RQ1GeneralizationResult] = {}
    for path in _rq1_result_directories(store.directory / "rq1-results"):
        result = RQ1GeneralizationResult.from_dict(_load_json(path / "result.json", "RQ1 result"))
        try:
            task = rq1_tasks[result.task_id]
        except KeyError as exc:  # pragma: no cover - store replay already enforces this
            raise FrozenArtifactError("RQ1 result is absent from its frozen plan") from exc
        comparison = (
            f"{task.method}|{task.adaptation_regime}|fold-{task.held_out_fold}|"
            f"{task.training_prompt_id}-to-{task.evaluation_prompt_id}"
        )
        if comparison in results:
            raise FrozenArtifactError("RQ1 result comparison identity is duplicated")
        results[comparison] = result
    if set(value.task_id for value in results.values()) != set(rq1_tasks):
        raise FrozenArtifactError("RQ1 result mapping is incomplete")
    return MappingProxyType(dict(sorted(results.items())))


def finalize_robustness_result_store(store: RobustnessResultStore) -> str:
    """Freeze the store only after all 36,060 preregistered tasks exist."""

    store = verify_robustness_result_store(store.directory)
    prompt_tasks, rq1_tasks = _task_maps(store.plan)
    prompt_results = {
        path.stem: PromptParaphraseResult.from_dict(_load_json(path, "prompt result")).result_digest
        for path in _prompt_result_files(store.directory / "prompt-results")
    }
    rq1_results = {
        path.name: RQ1GeneralizationResult.from_dict(
            _load_json(path / "result.json", "RQ1 result")
        ).result_digest
        for path in _rq1_result_directories(store.directory / "rq1-results")
    }
    if set(prompt_results) != set(prompt_tasks) or set(rq1_results) != set(rq1_tasks):
        raise DataValidationError("cannot finalize incomplete robustness diagnostics")
    body = {
        "schema_version": 1,
        "plan_digest": store.plan.plan_digest,
        "prompt_result_count": len(prompt_tasks),
        "prompt_result_set_digest": stable_hash(dict(sorted(prompt_results.items()))),
        "rq1_result_count": len(rq1_tasks),
        "rq1_result_set_digest": stable_hash(dict(sorted(rq1_results.items()))),
        "eligible_for_component_selection": False,
    }
    _write_once_json(
        store.directory / "complete.json",
        {**body, "completion_digest": stable_hash(body)},
    )
    verify_robustness_result_store(store.directory, require_complete=True)
    return stable_hash(body)

"""Strict validation for the approved Qwen 3.6 / A100 study amendment."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from mfh.artifact_namespace import (
    QWEN_STUDY_ARTIFACT_ROOT,
    QWEN_STUDY_NAMESPACE,
)
from mfh.artifact_namespace import (
    validate_active_study_artifact_paths as validate_active_study_artifact_paths,
)
from mfh.config import load_model_spec
from mfh.contracts import Runtime
from mfh.errors import ConfigurationError
from mfh.inference.transformers_snapshot import load_snapshot_manifest
from mfh.inference.vllm_preflight import load_vllm_runtime_policy
from mfh.provenance import sha256_file, stable_hash

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ACTIVE_NAME = "qwen3.6-27b-nvfp4"
ACTIVE_MODEL_NAME = _ACTIVE_NAME
APPROVED_AMENDMENT_DIGEST = (
    "8eae69a6fa1435ceb7a67b238d8f42772d782fad60e94adf01d1d69f6a1563c7"
)
ACTIVE_RUNTIME_POLICY_RELATIVE = "configs/runtimes/qwen3.6-27b-nvfp4-policy.json"
ACTIVE_MODEL_IDENTITIES: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        _ACTIVE_NAME: MappingProxyType(
            {
                "repository": "nvidia/Qwen3.6-27B-NVFP4",
                "revision": "0893e1606ff3d5f97a441f405d5fc541a6bdf404",
                "runtime": Runtime.VLLM,
                "quantization": "modelopt-mixed-nvfp4-fp8",
                "num_layers": 64,
            }
        )
    }
)
_ACTIVE_MODEL_DETAILS: Mapping[str, Any] = MappingProxyType(
    {
        "dtype": "bfloat16",
        "trust_remote_code": False,
        "role": "sole-activation-research-model",
        "candidate_layers": (16, 31, 32, 47, 48, 57, 63),
        "artifact": None,
        "artifact_sha256": None,
        "artifact_size_bytes": None,
    }
)
PRIMARY_RESEARCH_MODELS = frozenset({_ACTIVE_NAME})
PRIMARY_TRANSFORMER_MODELS: frozenset[str] = frozenset()
E0_MODELS = PRIMARY_RESEARCH_MODELS


_ROOT_FIELDS = {
    "schema_version",
    "amendment_id",
    "approved_on",
    "approval",
    "reason",
    "hardware_envelope",
    "study_namespace",
    "active_models",
    "model_independent_evidence_policy",
    "required_effect",
    "amendment_digest",
}
_ACTIVE_FIELDS = {
    "name",
    "upstream_model",
    "repository",
    "revision",
    "runtime",
    "quantization",
    "artifact_size_bytes",
    "model_config",
    "model_config_sha256",
    "snapshot_manifest",
    "snapshot_manifest_sha256",
    "runtime_policy",
    "runtime_policy_sha256",
    "runtime_policy_digest",
}
_HARDWARE = {
    "gpu_model": "NVIDIA A100-SXM4-40GB",
    "minimum_vram_bytes": 40_000_000_000,
    "architecture": "x86_64",
    "accelerator": "NVIDIA CUDA (SM80)",
}
_MODEL_INDEPENDENT_EVIDENCE_POLICY = {
    "runtime_validation_cohort": "regenerate-under-active-study-namespace",
    "contamination_review": "regenerate-from-pinned-source-snapshots",
    "manual_review": "new-human-review-required-before-e0-completion",
}
_REQUIRED_EFFECT = {
    "e0": (
        "regenerate-model-independent-inputs-and-complete-a-new-qwen-e0-ledger-in-the-qwen-study-namespace"
    ),
    "e1_through_e10": (
        "run-only-the-sole-active-qwen-vllm-model-in-the-qwen-study-namespace"
    ),
    "runtime_preflight": "require-a-live-passed-a100-sm80-receipt-before-qwen-e0",
    "colab": "retired",
}


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{context} must be a mapping")
    return value


def _sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ConfigurationError(f"{context} must be a lowercase SHA-256")
    return value


def _load_json(path: Path, *, context: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ConfigurationError(f"{context} is not a regular file: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot read {context} {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError(f"{context} must contain a JSON object")
    return raw


def _project_path(reference: object, *, root: Path, context: str) -> Path:
    if not isinstance(reference, str) or not reference:
        raise ConfigurationError(f"{context} must be a project-relative path")
    path = Path(reference)
    if path.is_absolute() or ".." in path.parts:
        raise ConfigurationError(f"{context} must stay inside the project")
    return root / path


def validate_active_model_spec(model: Any) -> None:
    """Reject drift from the sole approved Qwen VLLM declaration."""

    name = getattr(model, "name", None)
    if name != _ACTIVE_NAME:
        raise ConfigurationError(f"model {name!r} is not active under the approved amendment")
    identity = ACTIVE_MODEL_IDENTITIES[_ACTIVE_NAME]
    if {
        "repository": model.repository,
        "revision": model.revision,
        "runtime": model.runtime,
        "quantization": model.quantization,
        "num_layers": model.num_layers,
    } != dict(identity):
        raise ConfigurationError(f"active model {name!r} semantic config differs")
    if {
        "dtype": model.dtype,
        "trust_remote_code": model.trust_remote_code,
        "role": model.role,
        "candidate_layers": model.candidate_layers,
        "artifact": model.artifact,
        "artifact_sha256": model.artifact_sha256,
        "artifact_size_bytes": model.artifact_size_bytes,
    } != dict(_ACTIVE_MODEL_DETAILS):
        raise ConfigurationError(f"active model {name!r} detailed config differs")


def _validate_active_model(
    raw: Mapping[str, Any], *, model_directory: Path, project_root: Path
) -> None:
    rows = raw.get("active_models")
    if not isinstance(rows, list) or len(rows) != 1:
        raise ConfigurationError("active_models must contain the sole approved model")
    row = _mapping(rows[0], "active model")
    if set(row) != _ACTIVE_FIELDS or row.get("name") != _ACTIVE_NAME:
        raise ConfigurationError("active model fields differ from the approved declaration")
    model_path = model_directory / f"{_ACTIVE_NAME}.yaml"
    model = load_model_spec(model_path)
    validate_active_model_spec(model)
    if (
        row.get("repository") != model.repository
        or row.get("revision") != model.revision
        or row.get("runtime") != model.runtime.value
        or row.get("quantization") != model.quantization
        or row.get("model_config") != f"configs/models/{_ACTIVE_NAME}.yaml"
        or row.get("model_config_sha256") != sha256_file(model_path)
    ):
        raise ConfigurationError("active model config binding differs")
    manifest_path = _project_path(
        row.get("snapshot_manifest"), root=project_root, context="snapshot manifest"
    )
    manifest = load_snapshot_manifest(manifest_path, model_spec=model)
    if (
        manifest.total_size_bytes != row.get("artifact_size_bytes")
        or sha256_file(manifest_path) != row.get("snapshot_manifest_sha256")
    ):
        raise ConfigurationError("active model snapshot manifest differs")
    policy_path = _project_path(
        row.get("runtime_policy"), root=project_root, context="runtime policy"
    )
    policy = load_vllm_runtime_policy(policy_path)
    if (
        sha256_file(policy_path) != row.get("runtime_policy_sha256")
        or policy.get("policy_digest") != row.get("runtime_policy_digest")
        or _mapping(policy.get("model"), "runtime policy model").get("name")
        != model.name
    ):
        raise ConfigurationError("active model runtime policy differs")


def load_model_selection_amendment(
    path: str | Path,
    *,
    model_config_directory: str | Path,
) -> dict[str, Any]:
    """Load and bind the approved amendment without admitting a live run receipt."""

    source = Path(path).absolute()
    raw = _load_json(source, context="model-selection amendment")
    if set(raw) != _ROOT_FIELDS or raw.get("schema_version") != 4:
        raise ConfigurationError("model-selection amendment fields differ from schema version 4")
    declared = _sha256(raw.get("amendment_digest"), "amendment_digest")
    body = dict(raw)
    body.pop("amendment_digest")
    if stable_hash(body) != declared:
        raise ConfigurationError("model-selection amendment digest differs from its body")
    if declared != APPROVED_AMENDMENT_DIGEST:
        raise ConfigurationError("model-selection amendment is not the approved amendment")
    if (
        raw.get("approval")
        != "explicit-user-instruction-qwen3.6-27b-nvfp4-a100-40gb-vllm-fresh-run"
        or raw.get("approved_on") != "2026-07-22"
        or dict(_mapping(raw.get("hardware_envelope"), "hardware envelope"))
        != _HARDWARE
    ):
        raise ConfigurationError("model-selection amendment approval or hardware differs")
    namespace = _mapping(raw.get("study_namespace"), "study namespace")
    if (
        namespace.get("id") != QWEN_STUDY_NAMESPACE
        or namespace.get("artifact_root") != QWEN_STUDY_ARTIFACT_ROOT
        or namespace.get("rule")
        != "all-new-e0-through-e10-work-and-ledgers-must-live-under-this-root"
    ):
        raise ConfigurationError("Qwen study namespace differs")
    project_root = source.parents[2]
    model_directory = Path(model_config_directory).absolute()
    _validate_active_model(
        raw, model_directory=model_directory, project_root=project_root
    )
    evidence_policy = dict(
        _mapping(
            raw.get("model_independent_evidence_policy"),
            "model-independent evidence policy",
        )
    )
    if evidence_policy != _MODEL_INDEPENDENT_EVIDENCE_POLICY:
        raise ConfigurationError(
            "model-independent evidence policy differs from the approved amendment"
        )
    if dict(_mapping(raw.get("required_effect"), "required effect")) != _REQUIRED_EFFECT:
        raise ConfigurationError("required amendment effect differs from approved policy")
    return {**raw, "amendment_digest": declared}

"""Strict validation for the approved Qwen 3.6 / M4 Max study amendment."""

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
from mfh.inference.mlx_preflight import load_mlx_runtime_policy
from mfh.inference.transformers_snapshot import load_snapshot_manifest
from mfh.provenance import sha256_file, stable_hash

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ACTIVE_NAME = "qwen3.6-27b-mlx-4bit"
ACTIVE_MODEL_NAME = _ACTIVE_NAME
APPROVED_AMENDMENT_DIGEST = (
    "fc26c55c864156a296d030dcd4624885f21d933075979a1891985cf6f252a7a7"
)
ACTIVE_RUNTIME_POLICY_RELATIVE = "configs/runtimes/qwen3.6-27b-mlx-4bit-policy.json"
ACTIVE_MODEL_IDENTITIES: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        _ACTIVE_NAME: MappingProxyType(
            {
                "repository": "mlx-community/Qwen3.6-27B-4bit",
                "revision": "c000ac2c2057d94be3fa931000c31723aac53282",
                "runtime": Runtime.MLX,
                "quantization": "affine-g64-mlx-4bit",
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
    "preserved_model_independent_evidence",
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
    "chip": "Apple M4 Max",
    "unified_memory_bytes": 51_539_607_552,
    "architecture": "arm64",
    "accelerator": "Apple Metal",
}
_PRESERVED_EVIDENCE = {
    "runtime_validation_cohort_manifest_digest": (
        "bb89b2da16d899f8a38c0b090f84d1e43ffd1132e0fe0693295230b804f44442"
    ),
    "contamination_manifest_digest": (
        "ae79350a5e2f6310fccec4b91e9ef55821996f1797baacb21fb7de3d7b6131f2"
    ),
    "manual_review_manifest_digest": (
        "02f12825cb2b362b0bbbdde378f2018c15a0859d262bcbf3df2fb4ac9bfd02d6"
    ),
}
_REQUIRED_EFFECT = {
    "e0": (
        "rerun-qwen-runtime-validation-and-complete-a-new-e0-ledger-in-the-qwen-study-namespace"
    ),
    "e1_through_e10": (
        "run-only-the-sole-active-qwen-mlx-model-in-the-qwen-study-namespace"
    ),
    "runtime_preflight": "require-a-live-passed-m4-max-receipt-before-qwen-e0",
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
    """Reject drift from the sole approved Qwen MLX declaration."""

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
    policy = load_mlx_runtime_policy(policy_path)
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
    if set(raw) != _ROOT_FIELDS or raw.get("schema_version") != 3:
        raise ConfigurationError("model-selection amendment fields differ from schema version 3")
    declared = _sha256(raw.get("amendment_digest"), "amendment_digest")
    body = dict(raw)
    body.pop("amendment_digest")
    if stable_hash(body) != declared:
        raise ConfigurationError("model-selection amendment digest differs from its body")
    if declared != APPROVED_AMENDMENT_DIGEST:
        raise ConfigurationError("model-selection amendment is not the approved amendment")
    if (
        raw.get("approval")
        != "explicit-user-instruction-qwen3.6-27b-m4-max-48gb-apple-mlx"
        or raw.get("approved_on") != "2026-07-17"
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
    preserved = dict(
        _mapping(
            raw.get("preserved_model_independent_evidence"),
            "preserved model-independent evidence",
        )
    )
    if preserved != _PRESERVED_EVIDENCE:
        raise ConfigurationError("preserved evidence differs from the approved amendment")
    for key, value in preserved.items():
        _sha256(value, f"preserved evidence {key}")
    if dict(_mapping(raw.get("required_effect"), "required effect")) != _REQUIRED_EFFECT:
        raise ConfigurationError("required amendment effect differs from approved policy")
    return {**raw, "amendment_digest": declared}

"""Scientific E0 promotion after native MLX validation and contamination review."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mfh.data.contamination_review import verify_contamination_review_result
from mfh.data.io import read_questions
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.experiments.e0_mlx import verify_mlx_e0_bundle
from mfh.experiments.model_selection import validate_active_study_artifact_paths
from mfh.inference.transformers_snapshot import reject_symlink_path_components
from mfh.provenance import sha256_file, sha256_path, stable_hash

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FILES = frozenset({"manifest.json", "receipt.json"})
_VERIFIED_E0_COMPLETION = object()


@dataclass(frozen=True, slots=True)
class VerifiedE0CompletionReceipt:
    """Capability issued only after full live E0 completion replay."""

    directory: Path
    manifest_digest: str
    fingerprint: str
    _verification_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        expected_seal = (
            _VERIFIED_E0_COMPLETION,
            self.directory,
            self.manifest_digest,
            self.fingerprint,
        )
        if self._verification_token != expected_seal:
            raise DataValidationError(
                "E0 completion capabilities must come from full live verification"
            )


_AUTHORIZED_E0_COMPLETIONS: dict[int, tuple[VerifiedE0CompletionReceipt, Path, str, str]] = {}


def _assert_authorized_e0_completion(
    value: object,
) -> VerifiedE0CompletionReceipt:
    if not isinstance(value, VerifiedE0CompletionReceipt):
        raise DataValidationError(
            "E0 finalization requires a capability issued by full live verification"
        )
    registered = _AUTHORIZED_E0_COMPLETIONS.get(id(value))
    expected = (value, value.directory, value.manifest_digest, value.fingerprint)
    if registered != expected or registered[0] is not value:
        raise DataValidationError(
            "E0 completion capability was not issued by full live verification"
        )
    return value


def _require_sha256(value: str, context: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise DataValidationError(f"{context} must be a lowercase SHA-256 digest")


def _read_json(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot read {context}: {exc}") from exc
    if not isinstance(value, dict):
        raise DataValidationError(f"{context} must be a JSON object")
    return value


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        raise FrozenArtifactError(f"refusing to overwrite frozen JSON: {path}") from None
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _descriptor(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def _verify_inventory(directory: str | Path) -> Path:
    root = reject_symlink_path_components(directory, "E0 completion receipt")
    if not root.is_dir():
        raise DataValidationError("E0 completion receipt must be a regular directory")
    files: set[str] = set()
    directories: set[str] = set()
    for item in root.rglob("*"):
        if item.is_symlink():
            raise DataValidationError("E0 completion receipt cannot contain symlinks")
        relative = item.relative_to(root).as_posix()
        if item.is_file():
            files.add(relative)
        elif item.is_dir():
            directories.add(relative)
        else:
            raise DataValidationError("E0 completion receipt contains a special file")
    if files != _FILES or directories:
        raise DataValidationError("E0 completion receipt inventory differs")
    return root


def _verify_digest(value: Mapping[str, Any], field: str, context: str) -> None:
    body = dict(value)
    digest = body.pop(field, None)
    if type(digest) is not str or digest != stable_hash(body):
        raise DataValidationError(f"{context} {field} differs")


def _receipt_materials(
    *,
    mlx_directory: str | Path,
    expected_mlx_manifest_digest: str,
    expected_mlx_plan_identity: str,
    mlx_inputs: Mapping[str, Any],
    review_result_directory: str | Path,
    expected_review_result_manifest_digest: str,
    review_queue_directory: str | Path,
    expected_review_queue_manifest_digest: str,
    review_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    mlx_manifest = verify_mlx_e0_bundle(
        mlx_directory,
        expected_manifest_digest=expected_mlx_manifest_digest,
        expected_plan_identity=expected_mlx_plan_identity,
        **mlx_inputs,
    )
    review_manifest = verify_contamination_review_result(
        review_result_directory,
        expected_manifest_digest=expected_review_result_manifest_digest,
        review_queue_directory=review_queue_directory,
        expected_review_queue_manifest_digest=expected_review_queue_manifest_digest,
        **review_inputs,
    )
    mlx_status = mlx_manifest.get("scientific_status")
    if (
        not isinstance(mlx_status, Mapping)
        or mlx_status.get("e0_runtime_validation_complete") is not True
    ):
        raise DataValidationError("E0 MLX bundle does not complete runtime validation")
    if (
        review_manifest.get("status") != "complete"
        or review_manifest.get("scientific_eligible") is not True
    ):
        raise DataValidationError("semantic-contamination review is not scientifically complete")

    cohort_directory = mlx_inputs.get("cohort_directory")
    expected_cohort_digest = mlx_inputs.get("expected_cohort_manifest_digest")
    if cohort_directory is None or type(expected_cohort_digest) is not str:
        raise DataValidationError("E0 MLX inputs lack the anchored cohort")
    _require_sha256(expected_cohort_digest, "E0 cohort manifest")
    questions = tuple(read_questions(Path(cohort_directory) / "questions.jsonl"))
    if len(questions) != 500:
        raise DataValidationError("E0 completion requires the exact 500-question cohort")

    exclusion_payload = _read_json(
        Path(review_result_directory) / "excluded-source-ids.json",
        "semantic-contamination exclusions",
    )
    excluded_value = exclusion_payload.get("manual_overlap_source_ids")
    if not isinstance(excluded_value, list) or any(
        type(value) is not str for value in excluded_value
    ):
        raise DataValidationError("semantic-contamination exclusions are invalid")
    excluded = tuple(str(value) for value in excluded_value)
    cohort_ids = tuple(question.question_id for question in questions)
    affected = tuple(sorted(set(cohort_ids) & set(excluded)))
    if affected:
        raise DataValidationError(
            "E0 cohort contains manually confirmed contamination overlaps: " + ", ".join(affected)
        )

    body: dict[str, Any] = {
        "schema_version": 1,
        "phase": "E0",
        "scope": "scientific-runtime-validation-after-manual-contamination-review",
        "source_manifests": {
            "mlx_runtime": expected_mlx_manifest_digest,
            "contamination_review": expected_review_result_manifest_digest,
            "contamination_review_queue": expected_review_queue_manifest_digest,
            "runtime_validation_cohort": expected_cohort_digest,
        },
        "mlx_plan_identity": expected_mlx_plan_identity,
        "review_counts": review_manifest["counts"],
        "cohort_assessment": {
            "question_count": len(cohort_ids),
            "question_ids_sha256": stable_hash(cohort_ids),
            "manual_overlap_source_ids_sha256": stable_hash(excluded),
            "manual_overlap_source_count": len(excluded),
            "affected_cohort_ids": [],
            "affected_cohort_count": 0,
        },
        "status": "complete",
        "scientific_eligible": True,
        "e1_admission": "allowed-after-independent-receipt-verification",
    }
    return {**body, "receipt_digest": stable_hash(body)}


def write_e0_completion_receipt(
    directory: str | Path,
    *,
    mlx_directory: str | Path,
    expected_mlx_manifest_digest: str,
    expected_mlx_plan_identity: str,
    mlx_inputs: Mapping[str, Any],
    review_result_directory: str | Path,
    expected_review_result_manifest_digest: str,
    review_queue_directory: str | Path,
    expected_review_queue_manifest_digest: str,
    review_inputs: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Publish E0 promotion evidence only after replaying all prerequisite artifacts."""

    validated_paths = validate_active_study_artifact_paths(
        {"E0 completion receipt": directory, "E0 MLX bundle": mlx_directory}
    )
    directory = validated_paths["E0 completion receipt"]
    mlx_directory = validated_paths["E0 MLX bundle"]
    _require_sha256(expected_mlx_manifest_digest, "E0 MLX manifest")
    _require_sha256(expected_mlx_plan_identity, "E0 MLX plan")
    _require_sha256(expected_review_result_manifest_digest, "contamination review result")
    _require_sha256(expected_review_queue_manifest_digest, "contamination review queue")
    receipt = _receipt_materials(
        mlx_directory=mlx_directory,
        expected_mlx_manifest_digest=expected_mlx_manifest_digest,
        expected_mlx_plan_identity=expected_mlx_plan_identity,
        mlx_inputs=mlx_inputs,
        review_result_directory=review_result_directory,
        expected_review_result_manifest_digest=expected_review_result_manifest_digest,
        review_queue_directory=review_queue_directory,
        expected_review_queue_manifest_digest=expected_review_queue_manifest_digest,
        review_inputs=review_inputs,
    )
    output = reject_symlink_path_components(directory, "E0 completion receipt output")
    if output.exists():
        raise FrozenArtifactError(f"refusing to overwrite E0 completion receipt: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        _write_json_once(stage / "receipt.json", receipt)
        body: dict[str, Any] = {
            "schema_version": 1,
            "phase": "E0",
            "purpose": "scientific-e0-completion-receipt",
            "receipt_digest": receipt["receipt_digest"],
            "artifact": _descriptor(stage / "receipt.json"),
            "status": "complete",
            "scientific_eligible": True,
        }
        manifest = {**body, "manifest_digest": stable_hash(body)}
        _write_json_once(stage / "manifest.json", manifest)
        os.replace(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return verify_e0_completion_receipt(
        output,
        expected_manifest_digest=str(manifest["manifest_digest"]),
        mlx_directory=mlx_directory,
        expected_mlx_manifest_digest=expected_mlx_manifest_digest,
        expected_mlx_plan_identity=expected_mlx_plan_identity,
        mlx_inputs=mlx_inputs,
        review_result_directory=review_result_directory,
        expected_review_result_manifest_digest=expected_review_result_manifest_digest,
        review_queue_directory=review_queue_directory,
        expected_review_queue_manifest_digest=expected_review_queue_manifest_digest,
        review_inputs=review_inputs,
    )


def verify_e0_completion_receipt(
    directory: str | Path,
    *,
    expected_manifest_digest: str,
    mlx_directory: str | Path,
    expected_mlx_manifest_digest: str,
    expected_mlx_plan_identity: str,
    mlx_inputs: Mapping[str, Any],
    review_result_directory: str | Path,
    expected_review_result_manifest_digest: str,
    review_queue_directory: str | Path,
    expected_review_queue_manifest_digest: str,
    review_inputs: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Replay an E0 completion receipt against its live, externally anchored evidence."""

    _require_sha256(expected_manifest_digest, "E0 completion receipt manifest")
    expected_receipt = _receipt_materials(
        mlx_directory=mlx_directory,
        expected_mlx_manifest_digest=expected_mlx_manifest_digest,
        expected_mlx_plan_identity=expected_mlx_plan_identity,
        mlx_inputs=mlx_inputs,
        review_result_directory=review_result_directory,
        expected_review_result_manifest_digest=expected_review_result_manifest_digest,
        review_queue_directory=review_queue_directory,
        expected_review_queue_manifest_digest=expected_review_queue_manifest_digest,
        review_inputs=review_inputs,
    )
    root = _verify_inventory(directory)
    receipt = _read_json(root / "receipt.json", "E0 completion receipt")
    _verify_digest(receipt, "receipt_digest", "E0 completion receipt")
    if receipt != expected_receipt:
        raise DataValidationError("E0 completion receipt differs from source replay")
    manifest = _read_json(root / "manifest.json", "E0 completion receipt manifest")
    _verify_digest(manifest, "manifest_digest", "E0 completion receipt manifest")
    expected_body = {
        "schema_version": 1,
        "phase": "E0",
        "purpose": "scientific-e0-completion-receipt",
        "receipt_digest": expected_receipt["receipt_digest"],
        "artifact": _descriptor(root / "receipt.json"),
        "status": "complete",
        "scientific_eligible": True,
    }
    body = dict(manifest)
    digest = body.pop("manifest_digest", None)
    if body != expected_body or digest != expected_manifest_digest:
        raise DataValidationError("E0 completion receipt manifest declarations differ")
    return manifest


def authorize_e0_completion_receipt(
    directory: str | Path,
    *,
    expected_manifest_digest: str,
    mlx_directory: str | Path,
    expected_mlx_manifest_digest: str,
    expected_mlx_plan_identity: str,
    mlx_inputs: Mapping[str, Any],
    review_result_directory: str | Path,
    expected_review_result_manifest_digest: str,
    review_queue_directory: str | Path,
    expected_review_queue_manifest_digest: str,
    review_inputs: Mapping[str, Any],
) -> VerifiedE0CompletionReceipt:
    """Issue the only receipt capability accepted by scientific E0 finalization."""

    root = Path(directory).absolute()
    manifest = verify_e0_completion_receipt(
        root,
        expected_manifest_digest=expected_manifest_digest,
        mlx_directory=mlx_directory,
        expected_mlx_manifest_digest=expected_mlx_manifest_digest,
        expected_mlx_plan_identity=expected_mlx_plan_identity,
        mlx_inputs=mlx_inputs,
        review_result_directory=review_result_directory,
        expected_review_result_manifest_digest=expected_review_result_manifest_digest,
        review_queue_directory=review_queue_directory,
        expected_review_queue_manifest_digest=expected_review_queue_manifest_digest,
        review_inputs=review_inputs,
    )
    fingerprint = sha256_path(root)
    manifest_digest = str(manifest["manifest_digest"])
    capability = VerifiedE0CompletionReceipt(
        directory=root,
        manifest_digest=manifest_digest,
        fingerprint=fingerprint,
        _verification_token=(
            _VERIFIED_E0_COMPLETION,
            root,
            manifest_digest,
            fingerprint,
        ),
    )
    _AUTHORIZED_E0_COMPLETIONS[id(capability)] = (
        capability,
        capability.directory,
        capability.manifest_digest,
        capability.fingerprint,
    )
    return capability


def validate_e0_completion_receipt_snapshot(directory: str | Path) -> Mapping[str, Any]:
    """Validate a fingerprinted E0 receipt when the live upstream artifacts are unavailable."""

    root = _verify_inventory(directory)
    receipt = _read_json(root / "receipt.json", "E0 completion receipt")
    _verify_digest(receipt, "receipt_digest", "E0 completion receipt")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("phase") != "E0"
        or receipt.get("status") != "complete"
        or receipt.get("scientific_eligible") is not True
        or receipt.get("e1_admission") != "allowed-after-independent-receipt-verification"
        or receipt.get("cohort_assessment", {}).get("affected_cohort_count") != 0
    ):
        raise DataValidationError("E0 completion receipt is not eligible for E1 admission")
    manifests = receipt.get("source_manifests")
    if not isinstance(manifests, Mapping) or set(manifests) != {
        "mlx_runtime",
        "contamination_review",
        "contamination_review_queue",
        "runtime_validation_cohort",
    }:
        raise DataValidationError("E0 completion receipt source identities differ")
    for name, digest in manifests.items():
        if type(digest) is not str:
            raise DataValidationError(f"E0 completion source {name} is invalid")
        _require_sha256(digest, f"E0 completion source {name}")
    manifest = _read_json(root / "manifest.json", "E0 completion receipt manifest")
    _verify_digest(manifest, "manifest_digest", "E0 completion receipt manifest")
    expected_body = {
        "schema_version": 1,
        "phase": "E0",
        "purpose": "scientific-e0-completion-receipt",
        "receipt_digest": receipt["receipt_digest"],
        "artifact": _descriptor(root / "receipt.json"),
        "status": "complete",
        "scientific_eligible": True,
    }
    body = dict(manifest)
    body.pop("manifest_digest", None)
    if body != expected_body:
        raise DataValidationError("E0 completion receipt snapshot declarations differ")
    return manifest

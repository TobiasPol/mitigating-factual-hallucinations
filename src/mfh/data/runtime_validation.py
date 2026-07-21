"""Deterministic, auditable selection of the shared E0 factual prompts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from mfh.contracts import Question
from mfh.data.io import read_questions, write_questions
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.provenance import sha256_file, stable_hash

_FILES = {"manifest.json", "questions.jsonl"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_E0_QUESTION_COUNT = 500


def select_runtime_validation_questions(
    source_questions: tuple[Question, ...],
    *,
    seed: int = 17,
    limit: int = 500,
) -> tuple[Question, ...]:
    """Select a model-independent E0 set from rows reserved by the main split."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise DataValidationError("runtime-validation seed must be a non-negative integer")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise DataValidationError("runtime-validation limit must be a positive integer")
    if len(source_questions) < limit:
        raise DataValidationError("reserved source cannot fill the runtime-validation set")
    if any(
        question.benchmark != "triviaqa" or question.split != "reserved"
        for question in source_questions
    ):
        raise DataValidationError(
            "runtime-validation questions must come only from the reserved TriviaQA split"
        )

    def selection_key(question: Question) -> tuple[bytes, str]:
        digest = hashlib.sha256(f"{seed}\0{question.question_id}".encode()).digest()
        return digest, question.question_id

    selected = sorted(source_questions, key=selection_key)[:limit]
    return tuple(replace(question, split="runtime-validation") for question in selected)


def _descriptor(path: Path) -> dict[str, object]:
    return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def _verify_inventory(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise DataValidationError("runtime-validation bundle must be a regular directory")
    files: set[str] = set()
    directories: set[str] = set()
    for item in path.rglob("*"):
        if item.is_symlink():
            raise DataValidationError("runtime-validation bundle cannot contain symlinks")
        if item.is_file():
            files.add(item.relative_to(path).as_posix())
        elif item.is_dir():
            directories.add(item.relative_to(path).as_posix())
        else:
            raise DataValidationError("runtime-validation bundle contains a special file")
    if files != _FILES or directories:
        raise DataValidationError(
            "runtime-validation bundle inventory differs: "
            f"missing={sorted(_FILES - files)}, extra={sorted(files - _FILES)}, "
            f"directories={sorted(directories)}"
        )


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot read runtime-validation manifest: {exc}") from exc
    if not isinstance(value, dict):
        raise DataValidationError("runtime-validation manifest must be an object")
    body = dict(value)
    digest = body.pop("manifest_digest", None)
    if digest != stable_hash(body):
        raise DataValidationError("runtime-validation manifest digest mismatch")
    return value


def write_runtime_validation_bundle(
    directory: str | Path,
    *,
    reserved_source: str | Path,
    parent_split_manifest_digest: str,
    contamination_manifest_digest: str,
    seed: int = 17,
    limit: int = _E0_QUESTION_COUNT,
) -> dict[str, Any]:
    """Publish the provisional shared E0 set without consuming development/test rows."""

    destination = Path(directory)
    if destination.exists():
        raise FrozenArtifactError(f"refusing to overwrite runtime-validation bundle: {destination}")
    if not _SHA256.fullmatch(parent_split_manifest_digest) or not _SHA256.fullmatch(
        contamination_manifest_digest
    ):
        raise DataValidationError("parent evidence identities must be SHA-256 digests")
    if limit != _E0_QUESTION_COUNT:
        raise DataValidationError("published E0 runtime-validation bundles require 500 questions")
    source = tuple(read_questions(reserved_source))
    selected = select_runtime_validation_questions(source, seed=seed, limit=limit)

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        write_questions(stage / "questions.jsonl", selected)
        body: dict[str, Any] = {
            "schema_version": 1,
            "benchmark": "shared_benign_factual_500",
            "partition": "runtime-validation",
            "source": {
                "filename": Path(reserved_source).name,
                "sha256": sha256_file(reserved_source),
                "question_count": len(source),
                "source_partition": "reserved",
                "parent_split_manifest_digest": parent_split_manifest_digest,
                "contamination_manifest_digest": contamination_manifest_digest,
            },
            "selection": {
                "method": "sha256(seed-null-question-id)",
                "seed": seed,
                "question_count": limit,
                "question_ids_sha256": stable_hash([question.question_id for question in selected]),
            },
            "artifacts": {"questions.jsonl": _descriptor(stage / "questions.jsonl")},
            "scientific_status": {
                "eligible": False,
                "reason": "semantic-contamination-manual-review-pending",
            },
        }
        manifest = {**body, "manifest_digest": stable_hash(body)}
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return verify_runtime_validation_bundle(
        destination,
        reserved_source=reserved_source,
        expected_manifest_digest=str(manifest["manifest_digest"]),
        parent_split_manifest_digest=parent_split_manifest_digest,
        contamination_manifest_digest=contamination_manifest_digest,
    )


def verify_runtime_validation_bundle(
    directory: str | Path,
    *,
    reserved_source: str | Path,
    expected_manifest_digest: str,
    parent_split_manifest_digest: str,
    contamination_manifest_digest: str,
) -> dict[str, Any]:
    """Reproduce the selection from live reserved rows and an external bundle digest."""

    for digest in (
        expected_manifest_digest,
        parent_split_manifest_digest,
        contamination_manifest_digest,
    ):
        if not _SHA256.fullmatch(digest):
            raise DataValidationError("runtime-validation evidence identity is not SHA-256")
    root = Path(directory).absolute()
    _verify_inventory(root)
    manifest = _load_manifest(root)
    if manifest.get("manifest_digest") != expected_manifest_digest:
        raise DataValidationError("runtime-validation bundle differs from expected digest")
    selection = manifest.get("selection")
    if not isinstance(selection, dict):
        raise DataValidationError("runtime-validation selection metadata is invalid")
    seed = selection.get("seed")
    limit = selection.get("question_count")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise DataValidationError("runtime-validation selection seed is invalid")
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise DataValidationError("runtime-validation selection count is invalid")
    if limit != _E0_QUESTION_COUNT:
        raise DataValidationError("verified E0 runtime-validation bundles require 500 questions")
    source = tuple(read_questions(reserved_source))
    expected_source = {
        "filename": Path(reserved_source).name,
        "sha256": sha256_file(reserved_source),
        "question_count": len(source),
        "source_partition": "reserved",
        "parent_split_manifest_digest": parent_split_manifest_digest,
        "contamination_manifest_digest": contamination_manifest_digest,
    }
    if manifest.get("source") != expected_source:
        raise DataValidationError("runtime-validation source identity differs")
    expected = select_runtime_validation_questions(source, seed=seed, limit=limit)
    observed = tuple(read_questions(root / "questions.jsonl"))
    if observed != expected:
        raise DataValidationError("runtime-validation questions differ from deterministic replay")
    if selection != {
        "method": "sha256(seed-null-question-id)",
        "seed": seed,
        "question_count": limit,
        "question_ids_sha256": stable_hash([question.question_id for question in expected]),
    }:
        raise DataValidationError("runtime-validation selection metadata differs")
    if manifest.get("artifacts") != {"questions.jsonl": _descriptor(root / "questions.jsonl")}:
        raise DataValidationError("runtime-validation question artifact differs")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("benchmark") != "shared_benign_factual_500"
        or manifest.get("partition") != "runtime-validation"
        or manifest.get("scientific_status")
        != {
            "eligible": False,
            "reason": "semantic-contamination-manual-review-pending",
        }
    ):
        raise DataValidationError("runtime-validation frozen declarations differ")
    return manifest

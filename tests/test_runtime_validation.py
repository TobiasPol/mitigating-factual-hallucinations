from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from mfh.contracts import Question
from mfh.data.io import read_questions, write_questions
from mfh.data.runtime_validation import (
    select_runtime_validation_questions,
    verify_runtime_validation_bundle,
    write_runtime_validation_bundle,
)
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.provenance import sha256_file, stable_hash

_PARENT_DIGEST = "1" * 64
_CONTAMINATION_DIGEST = "2" * 64


def _question(index: int, *, split: str = "reserved") -> Question:
    return Question(
        question_id=f"triviaqa-{index:04d}",
        benchmark="triviaqa",
        text=f"Question {index}?",
        aliases=(f"Answer {index}",),
        split=split,
        metadata={"index": index},
    )


def _write_reserved_source(path: Path, *, count: int = 510) -> tuple[Question, ...]:
    questions = tuple(_question(index) for index in range(count))
    write_questions(path, questions)
    return questions


def _publish(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    source = tmp_path / "reserved.jsonl"
    bundle = tmp_path / "runtime-validation"
    _write_reserved_source(source)
    manifest = write_runtime_validation_bundle(
        bundle,
        reserved_source=source,
        parent_split_manifest_digest=_PARENT_DIGEST,
        contamination_manifest_digest=_CONTAMINATION_DIGEST,
    )
    return source, bundle, manifest


def test_selection_is_order_independent_and_uses_only_reserved_triviaqa() -> None:
    questions = tuple(_question(index) for index in range(8))
    forward = select_runtime_validation_questions(questions, seed=17, limit=5)
    reverse = select_runtime_validation_questions(tuple(reversed(questions)), seed=17, limit=5)

    assert forward == reverse
    assert len(forward) == 5
    assert {question.split for question in forward} == {"runtime-validation"}
    assert all(question.benchmark == "triviaqa" for question in forward)

    with pytest.raises(DataValidationError, match="reserved TriviaQA"):
        select_runtime_validation_questions(
            (replace(questions[0], split="development"), *questions[1:]),
            limit=5,
        )
    with pytest.raises(DataValidationError, match="reserved TriviaQA"):
        select_runtime_validation_questions(
            (replace(questions[0], benchmark="simpleqa"), *questions[1:]),
            limit=5,
        )


def test_published_bundle_has_exact_e0_count_and_replays(tmp_path: Path) -> None:
    source, bundle, manifest = _publish(tmp_path)

    assert manifest["selection"]["question_count"] == 500  # type: ignore[index]
    questions = tuple(read_questions(bundle / "questions.jsonl"))
    assert len(questions) == 500
    assert {question.split for question in questions} == {"runtime-validation"}
    verified = verify_runtime_validation_bundle(
        bundle,
        reserved_source=source,
        expected_manifest_digest=str(manifest["manifest_digest"]),
        parent_split_manifest_digest=_PARENT_DIGEST,
        contamination_manifest_digest=_CONTAMINATION_DIGEST,
    )
    assert verified == manifest

    with pytest.raises(DataValidationError, match="require 500"):
        write_runtime_validation_bundle(
            tmp_path / "undersized",
            reserved_source=source,
            parent_split_manifest_digest=_PARENT_DIGEST,
            contamination_manifest_digest=_CONTAMINATION_DIGEST,
            limit=499,
        )


def test_bundle_requires_external_identity_and_live_source(tmp_path: Path) -> None:
    source, bundle, manifest = _publish(tmp_path)

    with pytest.raises(DataValidationError, match="differs from expected digest"):
        verify_runtime_validation_bundle(
            bundle,
            reserved_source=source,
            expected_manifest_digest="3" * 64,
            parent_split_manifest_digest=_PARENT_DIGEST,
            contamination_manifest_digest=_CONTAMINATION_DIGEST,
        )

    source_questions = tuple(read_questions(source))
    write_questions(
        source,
        (replace(source_questions[0], text="Changed question?"), *source_questions[1:]),
        overwrite=True,
    )
    with pytest.raises(DataValidationError, match="source identity differs"):
        verify_runtime_validation_bundle(
            bundle,
            reserved_source=source,
            expected_manifest_digest=str(manifest["manifest_digest"]),
            parent_split_manifest_digest=_PARENT_DIGEST,
            contamination_manifest_digest=_CONTAMINATION_DIGEST,
        )


def test_bundle_rejects_tampering_symlinks_and_republication(tmp_path: Path) -> None:
    source, bundle, manifest = _publish(tmp_path)
    with pytest.raises(FrozenArtifactError, match="refusing to overwrite"):
        write_runtime_validation_bundle(
            bundle,
            reserved_source=source,
            parent_split_manifest_digest=_PARENT_DIGEST,
            contamination_manifest_digest=_CONTAMINATION_DIGEST,
        )

    (bundle / "unexpected").mkdir()
    with pytest.raises(DataValidationError, match="inventory differs"):
        verify_runtime_validation_bundle(
            bundle,
            reserved_source=source,
            expected_manifest_digest=str(manifest["manifest_digest"]),
            parent_split_manifest_digest=_PARENT_DIGEST,
            contamination_manifest_digest=_CONTAMINATION_DIGEST,
        )
    (bundle / "unexpected").rmdir()

    linked = tmp_path / "linked-bundle"
    linked.symlink_to(bundle, target_is_directory=True)
    with pytest.raises(DataValidationError, match="regular directory"):
        verify_runtime_validation_bundle(
            linked,
            reserved_source=source,
            expected_manifest_digest=str(manifest["manifest_digest"]),
            parent_split_manifest_digest=_PARENT_DIGEST,
            contamination_manifest_digest=_CONTAMINATION_DIGEST,
        )

    manifest_path = bundle / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["selection"]["seed"] = 19
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DataValidationError, match="manifest digest mismatch"):
        verify_runtime_validation_bundle(
            bundle,
            reserved_source=source,
            expected_manifest_digest=str(manifest["manifest_digest"]),
            parent_split_manifest_digest=_PARENT_DIGEST,
            contamination_manifest_digest=_CONTAMINATION_DIGEST,
        )


def test_verifier_rejects_self_consistent_resigned_499_row_bundle(tmp_path: Path) -> None:
    source, bundle, _ = _publish(tmp_path)
    question_path = bundle / "questions.jsonl"
    write_questions(question_path, tuple(read_questions(question_path))[:499], overwrite=True)
    manifest_path = bundle / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["selection"]["question_count"] = 499
    payload["selection"]["question_ids_sha256"] = stable_hash(
        [question.question_id for question in read_questions(question_path)]
    )
    payload["artifacts"]["questions.jsonl"] = {
        "sha256": sha256_file(question_path),
        "size_bytes": question_path.stat().st_size,
    }
    payload.pop("manifest_digest")
    payload["manifest_digest"] = stable_hash(payload)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(DataValidationError, match="require 500"):
        verify_runtime_validation_bundle(
            bundle,
            reserved_source=source,
            expected_manifest_digest=str(payload["manifest_digest"]),
            parent_split_manifest_digest=_PARENT_DIGEST,
            contamination_manifest_digest=_CONTAMINATION_DIGEST,
        )

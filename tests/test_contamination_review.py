from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

import mfh.data.contamination_review as review_module
from mfh.config import SemanticContaminationProtocol
from mfh.contracts import Question
from mfh.data.contamination_review import (
    finalize_contamination_review,
    prepare_contamination_review_queue,
    reviewer_attestation_text,
    verify_contamination_review_queue,
    verify_contamination_review_result,
)
from mfh.data.io import read_questions, write_questions
from mfh.errors import DataValidationError
from mfh.provenance import stable_hash

_BASE_DIGEST = "a" * 64


def _protocol() -> SemanticContaminationProtocol:
    return SemanticContaminationProtocol(
        model_repository="sentence-transformers/test",
        model_revision="1" * 40,
        model_artifact_tree_sha256="2" * 64,
        required_files=("config.json",),
        pooling="mean",
        normalize_embeddings=True,
        max_length=256,
        embedding_dimension=2,
        lexical_ngram_threshold=0.8,
        semantic_similarity_threshold=0.9,
        review_top_k=4,
        device="cpu",
        dtype="float32",
        encode_batch_size=2,
        similarity_batch_size=1,
        torch_num_threads=1,
    )


def _fake_base(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    root = tmp_path / "contamination"
    root.mkdir()
    questions = tuple(
        Question(
            question_id=f"source-{index}",
            benchmark="triviaqa",
            text=f"Source question {index}?",
            aliases=(f"answer-{index}",),
            split="train",
        )
        for index in range(5)
    )
    write_questions(root / "clean-source.jsonl", questions)
    queue = tuple(
        {
            "automatic_semantic_match": False,
            "similarity": 0.89 - index / 100,
            "source_question_id": f"source-{index}",
            "source_text": f"Source question {index}?",
            "target_question_id": f"target-{index}",
            "target_text": f"Target question {index}?",
        }
        for index in range(4)
    )
    (root / "manual-review-queue.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in queue),
        encoding="utf-8",
    )
    manifest = {
        "manifest_digest": _BASE_DIGEST,
        "manual_review": {
            "required": True,
            "status": "pending",
            "selection": "global-highest-cosine-similarity",
            "selection_count": len(queue),
            "selection_sha256": stable_hash(queue),
        },
    }
    return root, manifest


def _common(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], Path]:
    base, manifest = _fake_base(tmp_path)

    def verified(*_args: object, **_kwargs: object) -> tuple[Path, dict[str, Any]]:
        return base, manifest

    monkeypatch.setattr(review_module, "_verified_contamination", verified)
    values = {
        "contamination_directory": base,
        "expected_protocol": _protocol(),
        "model_directory": tmp_path / "unused-model",
        "triviaqa_source": tmp_path / "unused-source.jsonl",
        "target_sources": (tmp_path / "unused-target.jsonl",),
        "expected_contamination_manifest_digest": _BASE_DIGEST,
    }
    return values, base


def _write_annotations(queue: Path, output: Path) -> tuple[str, ...]:
    template = list(csv.DictReader((queue / "annotation-template.csv").open()))
    identifiers = tuple(row["review_id"] for row in template)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "review_id",
                "source_question",
                "target_question",
                "label",
                "notes",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        for index, identifier in enumerate(identifiers):
            writer.writerow(
                {
                    "review_id": identifier,
                    "source_question": template[index]["source_question"],
                    "target_question": template[index]["target_question"],
                    "label": "overlap" if index == 0 else "distinct",
                    "notes": "same fact" if index == 0 else "different fact",
                }
            )
    return identifiers


def _write_attestation(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "reviewer_id": "human-reviewer-1",
                "reviewed_at": "2026-07-15T12:00:00+02:00",
                "attestation": reviewer_attestation_text(),
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_blinded_contamination_queue_is_replayable_and_tamper_evident(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    common, _base = _common(tmp_path, monkeypatch)
    output = tmp_path / "review-queue"
    manifest = prepare_contamination_review_queue(output, **common)
    blind = [
        json.loads(line)
        for line in (output / "blind-items.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(blind) == 4
    assert all(
        set(row) == {"schema_version", "review_id", "source_question", "target_question"}
        for row in blind
    )
    assert "similarity" not in (output / "blind-items.jsonl").read_text(encoding="utf-8")
    verified = verify_contamination_review_queue(
        output,
        expected_manifest_digest=manifest["manifest_digest"],
        **common,
    )
    assert verified == manifest

    bindings = output / "operator-bindings.jsonl"
    bindings.chmod(0o644)
    bindings.write_text(bindings.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="bindings differ"):
        verify_contamination_review_queue(
            output,
            expected_manifest_digest=manifest["manifest_digest"],
            **common,
        )


def test_contamination_review_finalization_excludes_overlap_and_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    common, _base = _common(tmp_path, monkeypatch)
    queue = tmp_path / "review-queue"
    queue_manifest = prepare_contamination_review_queue(queue, **common)
    annotations = tmp_path / "annotations.csv"
    identifiers = _write_annotations(queue, annotations)
    attestation = tmp_path / "attestation.json"
    _write_attestation(attestation)
    output = tmp_path / "review-result"
    manifest = finalize_contamination_review(
        output,
        review_queue_directory=queue,
        expected_review_queue_manifest_digest=queue_manifest["manifest_digest"],
        annotations=annotations,
        reviewer_attestation=attestation,
        **common,
    )
    assert manifest["counts"] == {
        "reviewed": 4,
        "overlap": 1,
        "distinct": 3,
        "new_source_exclusions": 1,
        "reviewed_clean_source": 4,
    }
    bindings = [
        json.loads(line)
        for line in (queue / "operator-bindings.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    excluded_id = next(
        row["source_question_id"] for row in bindings if row["review_id"] == identifiers[0]
    )
    assert excluded_id not in {
        question.question_id for question in read_questions(output / "reviewed-clean-source.jsonl")
    }
    verified = verify_contamination_review_result(
        output,
        expected_manifest_digest=manifest["manifest_digest"],
        review_queue_directory=queue,
        expected_review_queue_manifest_digest=queue_manifest["manifest_digest"],
        **common,
    )
    assert verified == manifest

    reviewed = output / "reviewed-clean-source.jsonl"
    reviewed.chmod(0o644)
    reviewed.write_text(reviewed.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(DataValidationError):
        verify_contamination_review_result(
            output,
            expected_manifest_digest=manifest["manifest_digest"],
            review_queue_directory=queue,
            expected_review_queue_manifest_digest=queue_manifest["manifest_digest"],
            **common,
        )


def test_contamination_review_rejects_nonmanual_attestation_and_row_reordering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    common, _base = _common(tmp_path, monkeypatch)
    queue = tmp_path / "review-queue"
    queue_manifest = prepare_contamination_review_queue(queue, **common)
    annotations = tmp_path / "annotations.csv"
    _write_annotations(queue, annotations)
    rows = annotations.read_text(encoding="utf-8").splitlines()
    annotations.write_text("\n".join([rows[0], rows[2], rows[1], *rows[3:]]) + "\n")
    attestation = tmp_path / "attestation.json"
    _write_attestation(attestation)
    with pytest.raises(DataValidationError, match="frozen order"):
        finalize_contamination_review(
            tmp_path / "result",
            review_queue_directory=queue,
            expected_review_queue_manifest_digest=queue_manifest["manifest_digest"],
            annotations=annotations,
            reviewer_attestation=attestation,
            **common,
        )


def test_contamination_review_neutralizes_formulas_and_rejects_edited_questions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    common, base = _common(tmp_path, monkeypatch)
    queue_rows = [
        json.loads(line)
        for line in (base / "manual-review-queue.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    queue_rows[0]["source_text"] = '=HYPERLINK("https://invalid.example")'
    (base / "manual-review-queue.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in queue_rows),
        encoding="utf-8",
    )
    common_manifest = {
        "manifest_digest": _BASE_DIGEST,
        "manual_review": {
            "required": True,
            "status": "pending",
            "selection": "global-highest-cosine-similarity",
            "selection_count": len(queue_rows),
            "selection_sha256": stable_hash(queue_rows),
        },
    }

    def verified(*_args: object, **_kwargs: object) -> tuple[Path, dict[str, Any]]:
        return base, common_manifest

    monkeypatch.setattr(review_module, "_verified_contamination", verified)
    queue = tmp_path / "review-queue"
    queue_manifest = prepare_contamination_review_queue(queue, **common)
    template = list(csv.DictReader((queue / "annotation-template.csv").open()))
    formula_row = next(row for row in template if row["source_question"].startswith("'="))
    assert formula_row["source_question"] == '\'=HYPERLINK("https://invalid.example")'

    annotations = tmp_path / "annotations.csv"
    _write_annotations(queue, annotations)
    rows = list(csv.DictReader(annotations.open()))
    rows[0]["target_question"] += " edited"
    with annotations.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    attestation = tmp_path / "attestation.json"
    _write_attestation(attestation)
    with pytest.raises(DataValidationError, match="changed a blinded question"):
        finalize_contamination_review(
            tmp_path / "result",
            review_queue_directory=queue,
            expected_review_queue_manifest_digest=queue_manifest["manifest_digest"],
            annotations=annotations,
            reviewer_attestation=attestation,
            **common,
        )

    _write_annotations(queue, annotations)
    value = json.loads(attestation.read_text(encoding="utf-8"))
    value["attestation"] = "generated automatically"
    attestation.write_text(json.dumps(value) + "\n")
    with pytest.raises(DataValidationError, match="attestation differs"):
        finalize_contamination_review(
            tmp_path / "result",
            review_queue_directory=queue,
            expected_review_queue_manifest_digest=queue_manifest["manifest_digest"],
            annotations=annotations,
            reviewer_attestation=attestation,
            **common,
        )

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import mfh.experiments.e0_completion as completion_module
from mfh.contracts import Question
from mfh.data.io import write_questions
from mfh.errors import DataValidationError
from mfh.experiments.e0_completion import (
    VerifiedE0CompletionReceipt,
    authorize_e0_completion_receipt,
    validate_e0_completion_receipt_snapshot,
    verify_e0_completion_receipt,
    write_e0_completion_receipt,
)

_MLX_DIGEST = "a" * 64
_MLX_PLAN = "e" * 64
_REVIEW_DIGEST = "b" * 64
_QUEUE_DIGEST = "c" * 64
_COHORT_DIGEST = "d" * 64


def test_e0_completion_capability_cannot_be_caller_constructed(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="full live verification"):
        VerifiedE0CompletionReceipt(
            directory=tmp_path,
            manifest_digest="a" * 64,
            fingerprint="b" * 64,
            _verification_token=object(),
        )


def _inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], Path, Path]:
    cohort = tmp_path / "cohort"
    cohort.mkdir()
    write_questions(
        cohort / "questions.jsonl",
        tuple(
            Question(
                question_id=f"source-{index}",
                benchmark="triviaqa",
                text=f"Question {index}?",
                aliases=(f"answer-{index}",),
                split="runtime-validation",
            )
            for index in range(500)
        ),
    )
    review = tmp_path / "review"
    review.mkdir()
    (review / "excluded-source-ids.json").write_text(
        json.dumps({"manual_overlap_source_ids": ["source-700"]}) + "\n",
        encoding="utf-8",
    )
    mlx = tmp_path / "mlx"
    mlx.mkdir()

    def verified_mlx(*_args: object, **_kwargs: object) -> dict[str, Any]:
        return {
            "scientific_status": {"e0_runtime_validation_complete": True},
        }

    def verified_review(*_args: object, **_kwargs: object) -> dict[str, Any]:
        return {
            "status": "complete",
            "scientific_eligible": True,
            "counts": {
                "reviewed": 200,
                "overlap": 1,
                "distinct": 199,
                "new_source_exclusions": 1,
                "reviewed_clean_source": 1_000,
            },
        }

    monkeypatch.setattr(completion_module, "verify_mlx_e0_bundle", verified_mlx)
    monkeypatch.setattr(completion_module, "verify_contamination_review_result", verified_review)
    values = {
        "mlx_directory": mlx,
        "expected_mlx_manifest_digest": _MLX_DIGEST,
        "expected_mlx_plan_identity": _MLX_PLAN,
        "mlx_inputs": {
            "cohort_directory": cohort,
            "expected_cohort_manifest_digest": _COHORT_DIGEST,
        },
        "review_result_directory": review,
        "expected_review_result_manifest_digest": _REVIEW_DIGEST,
        "review_queue_directory": tmp_path / "queue",
        "expected_review_queue_manifest_digest": _QUEUE_DIGEST,
        "review_inputs": {},
    }
    return values, cohort, review


def test_e0_completion_receipt_replays_and_is_snapshot_verifiable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values, _cohort, _review = _inputs(tmp_path, monkeypatch)
    output = tmp_path / "completion"
    manifest = write_e0_completion_receipt(output, **values)
    assert manifest["scientific_eligible"] is True
    capability = authorize_e0_completion_receipt(
        output,
        expected_manifest_digest=manifest["manifest_digest"],
        **values,
    )
    with pytest.raises(DataValidationError, match="full live verification"):
        replace(capability, directory=tmp_path / "forged")
    forged = object.__new__(VerifiedE0CompletionReceipt)
    for field_name in ("directory", "manifest_digest", "fingerprint", "_verification_token"):
        object.__setattr__(forged, field_name, getattr(capability, field_name))
    with pytest.raises(DataValidationError, match="not issued"):
        completion_module._assert_authorized_e0_completion(forged)
    assert validate_e0_completion_receipt_snapshot(output) == manifest
    assert (
        verify_e0_completion_receipt(
            output,
            expected_manifest_digest=manifest["manifest_digest"],
            **values,
        )
        == manifest
    )

    receipt = output / "receipt.json"
    receipt.chmod(0o644)
    value = json.loads(receipt.read_text(encoding="utf-8"))
    value["cohort_assessment"]["affected_cohort_count"] = 1
    receipt.write_text(json.dumps(value) + "\n", encoding="utf-8")
    with pytest.raises(DataValidationError):
        validate_e0_completion_receipt_snapshot(output)


def test_e0_completion_rejects_a_manually_excluded_cohort_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values, _cohort, review = _inputs(tmp_path, monkeypatch)
    (review / "excluded-source-ids.json").write_text(
        json.dumps({"manual_overlap_source_ids": ["source-17"]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DataValidationError, match="source-17"):
        write_e0_completion_receipt(tmp_path / "completion", **values)

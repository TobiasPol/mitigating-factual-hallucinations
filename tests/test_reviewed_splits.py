from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import mfh.data.reviewed_splits as reviewed_module
from mfh.contracts import Question
from mfh.data.io import write_questions
from mfh.data.reviewed_splits import (
    VerifiedReviewedSplits,
    authorize_reviewed_split_bundle,
    validate_reviewed_split_snapshot,
    verify_reviewed_split_bundle,
    write_reviewed_split_bundle,
)
from mfh.data.splits import SplitPlan
from mfh.errors import DataValidationError
from mfh.provenance import sha256_file


def _inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], SplitPlan]:
    review = tmp_path / "review-result"
    review.mkdir()
    source = review / "reviewed-clean-source.jsonl"
    write_questions(
        source,
        tuple(
            Question(
                question_id=f"source-{index}",
                benchmark="triviaqa",
                text=f"Question {index}?",
                aliases=(f"answer-{index}",),
                split="train",
            )
            for index in range(20)
        ),
    )
    simpleqa_source = tmp_path / "simpleqa.jsonl"
    write_questions(
        simpleqa_source,
        tuple(
            Question(
                question_id=f"simple-{index}",
                benchmark="simpleqa_verified",
                text=f"Simple question {index}?",
                aliases=(f"simple answer {index}",),
                split="eval",
            )
            for index in range(4)
        ),
    )
    aa_source = tmp_path / "aa.jsonl"
    write_questions(
        aa_source,
        tuple(
            Question(
                question_id=f"aa-{index}",
                benchmark="aa_omniscience_public_600",
                text=f"AA question {index}?",
                aliases=(f"aa answer {index}",),
                split="eval",
            )
            for index in range(3)
        ),
    )

    def verified_review(*_args: object, **_kwargs: object) -> dict[str, Any]:
        return {"reviewed_clean_source_sha256": sha256_file(source)}

    monkeypatch.setattr(reviewed_module, "verify_contamination_review_result", verified_review)
    plan = SplitPlan(steer=4, controller=3, dev=3, test=3, seed=17)
    values = {
        "review_result_directory": review,
        "expected_review_result_manifest_digest": "a" * 64,
        "review_queue_directory": tmp_path / "queue",
        "expected_review_queue_manifest_digest": "b" * 64,
        "review_inputs": {"target_sources": (simpleqa_source, aa_source)},
        "plan": plan,
    }
    return values, plan


def test_reviewed_splits_replay_and_issue_sealed_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values, _plan = _inputs(tmp_path, monkeypatch)
    output = tmp_path / "reviewed-splits"
    manifest = write_reviewed_split_bundle(output, **values)
    assert validate_reviewed_split_snapshot(output) == manifest
    assert (
        verify_reviewed_split_bundle(
            output,
            expected_manifest_digest=manifest["manifest_digest"],
            **values,
        )
        == manifest
    )
    capability = authorize_reviewed_split_bundle(
        output,
        expected_manifest_digest=manifest["manifest_digest"],
        **values,
    )
    with pytest.raises(DataValidationError, match="full live verification"):
        replace(capability, directory=tmp_path / "forged")
    forged = object.__new__(VerifiedReviewedSplits)
    for field_name in ("directory", "manifest_digest", "fingerprint", "_verification_token"):
        object.__setattr__(forged, field_name, getattr(capability, field_name))
    with pytest.raises(DataValidationError, match="not issued"):
        reviewed_module._assert_authorized_reviewed_splits(forged)
    with pytest.raises(DataValidationError, match="full live verification"):
        VerifiedReviewedSplits(
            directory=output,
            manifest_digest="c" * 64,
            fingerprint="d" * 64,
            _verification_token=object(),
        )


def test_reviewed_splits_reject_tampered_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values, _plan = _inputs(tmp_path, monkeypatch)
    output = tmp_path / "reviewed-splits"
    manifest = write_reviewed_split_bundle(output, **values)
    partition = output / "T-dev.jsonl"
    partition.chmod(0o644)
    partition.write_text(partition.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(DataValidationError):
        verify_reviewed_split_bundle(
            output,
            expected_manifest_digest=manifest["manifest_digest"],
            **values,
        )

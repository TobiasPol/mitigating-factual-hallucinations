from __future__ import annotations

from pathlib import Path

import pytest

from mfh.contracts import Question
from mfh.data.io import write_questions
from mfh.data.splits import semantic_group_ids
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.experiments.e5_controller_splits import (
    verify_e5_controller_splits,
    write_e5_controller_splits,
)


def _questions(count: int = 5_000) -> tuple[Question, ...]:
    return tuple(
        Question(
            question_id=f"controller-{index:05d}",
            benchmark="triviaqa",
            text=f"Unique controller question {index}?",
            aliases=(f"unique-answer-{index}",),
            split="T-controller",
        )
        for index in range(count)
    )


def test_e5_controller_splits_materialize_exact_e2_partition(tmp_path: Path) -> None:
    source = (
        Path(__file__).parents[1]
        / "artifacts/splits/triviaqa-reviewed/T-controller.jsonl"
    )
    verified = write_e5_controller_splits(
        tmp_path / "splits",
        source_questions=source,
    )
    assert len(verified.train_questions) == 4_000
    assert len(verified.calibration_questions) == 1_000
    assert all(value.split == "T-controller-train" for value in verified.train_questions)
    assert all(
        value.split == "T-controller-calibration"
        for value in verified.calibration_questions
    )
    train_groups = set(semantic_group_ids(verified.train_questions).values())
    calibration_groups = set(semantic_group_ids(verified.calibration_questions).values())
    assert train_groups.isdisjoint(calibration_groups)
    assert verified.manifest["scientific_eligible"] is True

    train_path = verified.directory / "T-controller-train.jsonl"
    train_path.write_text(
        train_path.read_text(encoding="utf-8").replace(
            '"split": "T-controller-train"',
            '"split": "T-controller"',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(FrozenArtifactError, match="exact replay"):
        verify_e5_controller_splits(
            verified.directory,
            source_questions=source,
            expected_manifest_digest=verified.manifest_digest,
        )


def test_e5_controller_splits_require_exact_source_count(tmp_path: Path) -> None:
    source = tmp_path / "T-controller.jsonl"
    write_questions(source, _questions(4_999))
    with pytest.raises(DataValidationError, match=r"reviewed.*inventory"):
        write_e5_controller_splits(tmp_path / "splits", source_questions=source)

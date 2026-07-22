from __future__ import annotations

from pathlib import Path

import pytest

from mfh.contracts import Question
from mfh.data.io import write_questions
from mfh.data.splits import semantic_group_ids
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.experiments import e5_controller_splits as splits_module
from mfh.experiments.e5_controller_splits import (
    verify_e5_controller_splits,
    write_e5_controller_splits,
)
from mfh.provenance import sha256_file


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


def _reviewed_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, count: int = 5_000
) -> Path:
    source = (
        tmp_path
        / "artifacts/studies/qwen36-27b-nvfp4-a10040-v1/frozen/reviewed-splits/T-controller.jsonl"
    )
    source.parent.mkdir(parents=True)
    write_questions(source, _questions(count))
    (source.parent / "manifest.json").write_text("{}\n", encoding="utf-8")
    digest = "a" * 64
    monkeypatch.setattr(splits_module, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        splits_module,
        "validate_reviewed_split_snapshot",
        lambda _path: {
            "manifest_digest": digest,
            "artifacts": {"T-controller.jsonl": {"sha256": sha256_file(source)}},
        },
    )
    return source


def test_e5_controller_splits_materialize_exact_e2_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _reviewed_source(tmp_path, monkeypatch)
    verified = write_e5_controller_splits(
        tmp_path / "splits",
        source_questions=source,
        expected_reviewed_split_manifest_digest="a" * 64,
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
            expected_reviewed_split_manifest_digest="a" * 64,
        )


def test_e5_controller_splits_require_exact_source_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _reviewed_source(tmp_path, monkeypatch, count=4_999)
    with pytest.raises(DataValidationError, match=r"exact 5,000-row"):
        write_e5_controller_splits(
            tmp_path / "splits",
            source_questions=source,
            expected_reviewed_split_manifest_digest="a" * 64,
        )

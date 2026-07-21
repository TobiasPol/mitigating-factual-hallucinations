"""Materialize the frozen E2 controller subdivision for E5 operators."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from mfh.artifact_namespace import validate_active_study_artifact_paths
from mfh.contracts import Question
from mfh.data.io import read_questions, write_questions
from mfh.data.reviewed_splits import validate_reviewed_split_snapshot
from mfh.data.splits import semantic_group_ids
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.experiments.e2_schedule import (
    E2CaptureProtocol,
    controller_feature_partitions,
)
from mfh.provenance import sha256_file, stable_hash

_TRAIN = "T-controller-train"
_CALIBRATION = "T-controller-calibration"
_FILES = MappingProxyType(
    {
        _TRAIN: "T-controller-train.jsonl",
        _CALIBRATION: "T-controller-calibration.jsonl",
    }
)
_INVENTORY = frozenset({*_FILES.values(), "manifest.json"})
_ALGORITHM = "e2-semantic-group-exact-subset-v1"
_REVIEWED_SPLIT_MANIFEST_DIGEST = (
    "05e13f0193155551400fd636e8dd6d97e065dd80205133a9440ef13105bce148"
)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_REVIEWED_SOURCE_RELATIVE = Path(
    "artifacts/splits/triviaqa-reviewed/T-controller.jsonl"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class VerifiedE5ControllerSplits:
    directory: Path
    source: Path
    train_questions: tuple[Question, ...]
    calibration_questions: tuple[Question, ...]
    manifest: Mapping[str, Any]

    @property
    def manifest_digest(self) -> str:
        return str(self.manifest["manifest_digest"])


def _source_questions(path: str | Path) -> tuple[Path, tuple[Question, ...]]:
    source = Path(path).resolve(strict=False)
    if source.is_symlink() or not source.is_file():
        raise FrozenArtifactError("E5 controller split source must be a regular file")
    manifest = validate_reviewed_split_snapshot(source.parent)
    artifacts = manifest.get("artifacts")
    descriptor = (
        artifacts.get("T-controller.jsonl")
        if isinstance(artifacts, Mapping)
        else None
    )
    if (
        source != (_PROJECT_ROOT / _REVIEWED_SOURCE_RELATIVE)
        or
        source.name != "T-controller.jsonl"
        or manifest.get("manifest_digest") != _REVIEWED_SPLIT_MANIFEST_DIGEST
        or not isinstance(descriptor, Mapping)
        or descriptor.get("sha256") != sha256_file(source)
    ):
        raise DataValidationError(
            "E5 controller source is not the frozen reviewed T-controller artifact"
        )
    questions = tuple(read_questions(source))
    protocol = E2CaptureProtocol()
    if (
        len(questions) != protocol.controller_rows
        or any(
            question.benchmark != "triviaqa" or question.split != "T-controller"
            for question in questions
        )
    ):
        raise DataValidationError(
            "E5 controller split source must be the exact 5,000-row TriviaQA T-controller set"
        )
    return source, questions


def _partition_questions(
    questions: Sequence[Question],
) -> tuple[tuple[Question, ...], tuple[Question, ...], Mapping[str, str]]:
    protocol = E2CaptureProtocol()
    assignments = controller_feature_partitions(
        questions,
        calibration_rows=protocol.controller_calibration_rows,
        seed=protocol.seed,
    )
    train = tuple(
        replace(question, split=_TRAIN)
        for question in questions
        if assignments[question.question_id] == _TRAIN
    )
    calibration = tuple(
        replace(question, split=_CALIBRATION)
        for question in questions
        if assignments[question.question_id] == _CALIBRATION
    )
    if (
        len(train) != protocol.controller_rows - protocol.controller_calibration_rows
        or len(calibration) != protocol.controller_calibration_rows
        or {value.question_id for value in train}
        & {value.question_id for value in calibration}
    ):
        raise DataValidationError("E5 controller split cardinality or disjointness differs")
    groups = semantic_group_ids(questions)
    train_groups = {groups[value.question_id] for value in train}
    calibration_groups = {groups[value.question_id] for value in calibration}
    if train_groups & calibration_groups:
        raise DataValidationError("E5 controller split crosses a semantic group")
    return train, calibration, assignments


def _question_ids_sha256(questions: Sequence[Question]) -> str:
    return stable_hash([value.question_id for value in questions])


def _partition_descriptor(
    path: Path,
    questions: Sequence[Question],
    *,
    source_groups: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "file": path.name,
        "rows": len(questions),
        "artifact_sha256": sha256_file(path),
        "question_ids_sha256": _question_ids_sha256(questions),
        "semantic_group_ids_sha256": stable_hash(
            [source_groups[value.question_id] for value in questions]
        ),
    }


def _manifest_body(
    directory: Path,
    *,
    source: Path,
    source_questions: Sequence[Question],
    train: Sequence[Question],
    calibration: Sequence[Question],
    assignments: Mapping[str, str],
) -> dict[str, Any]:
    source_groups = semantic_group_ids(source_questions)
    reviewed_manifest = validate_reviewed_split_snapshot(source.parent)
    return {
        "schema_version": 1,
        "phase": "E5-controller-subdivision",
        "algorithm": _ALGORITHM,
        "algorithm_source_sha256": sha256_file(
            Path(__file__).with_name("e2_schedule.py")
        ),
        "seed": E2CaptureProtocol().seed,
        "source_relative_path": _REVIEWED_SOURCE_RELATIVE.as_posix(),
        "source_sha256": sha256_file(source),
        "source_rows": len(source_questions),
        "source_question_ids_sha256": _question_ids_sha256(source_questions),
        "reviewed_split_manifest_relative_path": (
            _REVIEWED_SOURCE_RELATIVE.parent / "manifest.json"
        ).as_posix(),
        "reviewed_split_manifest_sha256": sha256_file(source.parent / "manifest.json"),
        "reviewed_split_manifest_digest": reviewed_manifest["manifest_digest"],
        "assignment_sha256": stable_hash(
            [
                [question.question_id, assignments[question.question_id]]
                for question in source_questions
            ]
        ),
        "partitions": {
            _TRAIN: _partition_descriptor(
                directory / _FILES[_TRAIN],
                train,
                source_groups=source_groups,
            ),
            _CALIBRATION: _partition_descriptor(
                directory / _FILES[_CALIBRATION],
                calibration,
                source_groups=source_groups,
            ),
        },
        "semantic_groups_disjoint": True,
        "complete_partition": True,
        "scientific_eligible": True,
    }


def _canonical_question_text(questions: Sequence[Question]) -> str:
    rows = (
        {
            "question_id": question.question_id,
            "benchmark": question.benchmark,
            "text": question.text,
            "aliases": list(question.aliases),
            "split": question.split,
            "entities": list(question.entities),
            "metadata": dict(question.metadata),
            "schema_version": 1,
        }
        for question in questions
    )
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
        for row in rows
    )


def write_e5_controller_splits(
    destination: str | Path,
    *,
    source_questions: str | Path,
) -> VerifiedE5ControllerSplits:
    """Atomically materialize the exact E2 4,000/1,000 controller split."""

    output = validate_active_study_artifact_paths(
        {"E5 controller splits": destination}
    )["E5 controller splits"]
    if output.exists() or output.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite E5 controller splits: {output}")
    source, questions = _source_questions(source_questions)
    train, calibration, assignments = _partition_questions(questions)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        write_questions(stage / _FILES[_TRAIN], train)
        write_questions(stage / _FILES[_CALIBRATION], calibration)
        body = _manifest_body(
            stage,
            source=source,
            source_questions=questions,
            train=train,
            calibration=calibration,
            assignments=assignments,
        )
        manifest = {**body, "manifest_digest": stable_hash(body)}
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return verify_e5_controller_splits(
        output,
        source_questions=source,
        expected_manifest_digest=str(manifest["manifest_digest"]),
    )


def verify_e5_controller_splits(
    directory: str | Path,
    *,
    source_questions: str | Path,
    expected_manifest_digest: str,
) -> VerifiedE5ControllerSplits:
    """Replay the E2 subdivision and reject any mutated or substituted row."""

    if _SHA256.fullmatch(expected_manifest_digest) is None:
        raise DataValidationError(
            "expected E5 controller split manifest digest must be a lowercase SHA-256"
        )
    source_directory = validate_active_study_artifact_paths(
        {"E5 controller splits": directory}
    )["E5 controller splits"]
    inventory = (
        {value.name for value in source_directory.iterdir()}
        if source_directory.is_dir()
        else set()
    )
    if (
        source_directory.is_symlink()
        or inventory != _INVENTORY
        or any(
            value.is_symlink() or not value.is_file()
            for value in source_directory.iterdir()
        )
    ):
        raise FrozenArtifactError("E5 controller split inventory differs")
    source, questions = _source_questions(source_questions)
    train, calibration, assignments = _partition_questions(questions)
    expected_questions = {_TRAIN: train, _CALIBRATION: calibration}
    for partition, expected in expected_questions.items():
        path = source_directory / _FILES[partition]
        if tuple(read_questions(path)) != expected or path.read_text(
            encoding="utf-8"
        ) != _canonical_question_text(expected):
            raise FrozenArtifactError(f"E5 {partition} rows differ from exact replay")
    try:
        manifest = json.loads((source_directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read E5 controller split manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise FrozenArtifactError("E5 controller split manifest must be a mapping")
    body = dict(manifest)
    digest = body.pop("manifest_digest", None)
    expected_body = _manifest_body(
        source_directory,
        source=source,
        source_questions=questions,
        train=train,
        calibration=calibration,
        assignments=assignments,
    )
    if (
        digest != expected_manifest_digest
        or digest != stable_hash(body)
        or body != expected_body
        or (source_directory / "manifest.json").read_text(encoding="utf-8")
        != json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ):
        raise FrozenArtifactError("E5 controller split manifest differs from exact replay")
    return VerifiedE5ControllerSplits(
        directory=source_directory.resolve(),
        source=source,
        train_questions=train,
        calibration_questions=calibration,
        manifest=MappingProxyType(manifest),
    )

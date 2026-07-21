"""Streaming JSONL I/O for canonical question and generation records."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from mfh.contracts import GenerationRecord, Question
from mfh.errors import DataValidationError


def _write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]], *, overwrite: bool) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    count = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(dict(row), ensure_ascii=False, sort_keys=True, allow_nan=False)
                )
                handle.write("\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary_name, destination)
        else:
            try:
                os.link(temporary_name, destination)
            except FileExistsError:
                raise FileExistsError(f"refusing to overwrite {destination}") from None
            Path(temporary_name).unlink()
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return count


def write_question_bundle(
    directory: str | Path,
    groups: Mapping[str, Iterable[Question]],
    *,
    metadata_files: Mapping[str, Mapping[str, Any]] | None = None,
    overwrite: bool = False,
) -> Mapping[str, int]:
    """Publish a complete set of JSONL files with directory-level rollback."""

    destination = Path(directory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    backup: Path | None = None
    counts: dict[str, int] = {}
    seen_question_ids: set[str] = set()
    metadata_files = metadata_files or {}

    def globally_unique(questions: Iterable[Question]) -> Iterator[Question]:
        for question in questions:
            if question.question_id in seen_question_ids:
                raise DataValidationError(
                    f"duplicate question_id across bundle: {question.question_id}"
                )
            seen_question_ids.add(question.question_id)
            yield question

    try:
        overlap = set(groups) & set(metadata_files)
        if overlap:
            raise DataValidationError(
                f"question and metadata bundle filenames collide: {sorted(overlap)}"
            )
        for filename, questions in sorted(groups.items()):
            if Path(filename).name != filename:
                raise DataValidationError(f"bundle filename must be a basename: {filename!r}")
            counts[filename] = write_questions(stage / filename, globally_unique(questions))
        for filename, payload in sorted(metadata_files.items()):
            if Path(filename).name != filename or not filename.endswith(".json"):
                raise DataValidationError(
                    f"metadata bundle filename must be a JSON basename: {filename!r}"
                )
            (stage / filename).write_text(
                json.dumps(
                    dict(payload),
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
        if destination.exists():
            if not overwrite:
                raise FileExistsError(f"refusing to overwrite split bundle {destination}")
            backup = Path(
                tempfile.mkdtemp(prefix=f".{destination.name}.backup-", dir=destination.parent)
            )
            backup.rmdir()
            os.replace(destination, backup)
        try:
            os.replace(stage, destination)
        except BaseException:
            if backup is not None and backup.exists() and not destination.exists():
                os.replace(backup, destination)
                backup = None
            raise
        if backup is not None:
            if backup.is_dir() and not backup.is_symlink():
                shutil.rmtree(backup)
            else:
                backup.unlink()
            backup = None
        return counts
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        if backup is not None and backup.exists() and not destination.exists():
            os.replace(backup, destination)


def write_questions(
    path: str | Path, questions: Iterable[Question], *, overwrite: bool = False
) -> int:
    def rows() -> Iterator[Mapping[str, Any]]:
        seen: set[str] = set()
        for question in questions:
            if question.question_id in seen:
                raise DataValidationError(f"duplicate question_id: {question.question_id}")
            seen.add(question.question_id)
            yield {
                "question_id": question.question_id,
                "benchmark": question.benchmark,
                "text": question.text,
                "aliases": list(question.aliases),
                "split": question.split,
                "entities": list(question.entities),
                "metadata": dict(question.metadata),
                "schema_version": 1,
            }

    return _write_jsonl(path, rows(), overwrite=overwrite)


def read_questions(path: str | Path) -> Iterator[Question]:
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
                if value.pop("schema_version", None) != 1:
                    raise DataValidationError("unsupported question schema version")
                aliases = tuple(value.pop("aliases"))
                entities = tuple(value.pop("entities", ()))
                question = Question(
                    **value,
                    aliases=aliases,
                    entities=entities,
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise DataValidationError(
                    f"invalid question at {path}:{line_number}: {exc}"
                ) from exc
            if question.question_id in seen:
                raise DataValidationError(f"duplicate question_id at {path}:{line_number}")
            seen.add(question.question_id)
            yield question


def write_generation_records(
    path: str | Path, records: Iterable[GenerationRecord], *, overwrite: bool = False
) -> int:
    return _write_jsonl(path, (record.to_dict() for record in records), overwrite=overwrite)


def read_generation_records(path: str | Path) -> Iterator[GenerationRecord]:
    keys: set[tuple[str, str]] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = GenerationRecord.from_dict(json.loads(line))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise DataValidationError(
                    f"invalid generation record at {path}:{line_number}: {exc}"
                ) from exc
            key = (record.question_id, record.condition_id)
            if key in keys:
                raise DataValidationError(
                    f"duplicate question/condition pair at {path}:{line_number}"
                )
            keys.add(key)
            yield record

"""Adapters for the three factuality benchmarks in the research plan."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from mfh.contracts import BenchmarkSpec, Question
from mfh.errors import DataValidationError, OptionalDependencyError


def _require_columns(fieldnames: Iterable[str] | None, required: set[str], source: Path) -> None:
    present = set(fieldnames or ())
    missing = required - present
    if missing:
        raise DataValidationError(f"{source} is missing required columns: {sorted(missing)}")


def load_simpleqa_csv(path: str | Path) -> Iterator[Question]:
    source = Path(path)
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader.fieldnames, {"original_index", "problem", "answer"}, source)
        for line_number, row in enumerate(reader, start=2):
            try:
                question_id = f"simpleqa:{row['original_index'].strip()}"
                metadata = {
                    key: row[key]
                    for key in (
                        "topic",
                        "answer_type",
                        "multi_step",
                        "requires_reasoning",
                        "urls",
                    )
                    if key in row and row[key] != ""
                }
                metadata.update(
                    {
                        "source_repository": "google/simpleqa-verified",
                        "source_revision": "0dc97e0d28d8233463e005cdc4475cc2a13ba2dc",
                        "source_split": "eval",
                        "source_row_id": row["original_index"].strip(),
                    }
                )
                yield Question(
                    question_id=question_id,
                    benchmark="simpleqa_verified",
                    text=row["problem"],
                    aliases=(row["answer"],),
                    split="eval",
                    metadata=metadata,
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise DataValidationError(
                    f"invalid SimpleQA row at {source}:{line_number}: {exc}"
                ) from exc


def load_aa_csv(path: str | Path) -> Iterator[Question]:
    source = Path(path)
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_columns(
            reader.fieldnames, {"domain", "topic", "question_id", "question", "answer"}, source
        )
        for line_number, row in enumerate(reader, start=2):
            try:
                yield Question(
                    question_id=f"aa-public:{row['question_id'].strip()}",
                    benchmark="aa_omniscience_public_600",
                    text=row["question"],
                    aliases=(row["answer"],),
                    split="public",
                    entities=(row["topic"],),
                    metadata={
                        "domain": row["domain"],
                        "topic": row["topic"],
                        "source_repository": "ArtificialAnalysis/AA-Omniscience-Public",
                        "source_revision": "4a8ffc87c4650054825fb767fe0da4a4fc97ff32",
                        "source_split": "train",
                        "source_row_id": row["question_id"].strip(),
                    },
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise DataValidationError(
                    f"invalid AA row at {source}:{line_number}: {exc}"
                ) from exc


def _aliases(answer: Any) -> tuple[str, ...]:
    if isinstance(answer, str):
        return (answer,)
    if isinstance(answer, Mapping):
        candidates = answer.get("aliases") or ()
        value = answer.get("value")
        combined = list(candidates) if isinstance(candidates, list) else []
        if isinstance(value, str):
            combined.insert(0, value)
        return tuple(str(item) for item in combined)
    if isinstance(answer, list):
        return tuple(str(item) for item in answer)
    raise DataValidationError(f"unsupported answer representation: {type(answer).__name__}")


def question_from_hf_row(spec: BenchmarkSpec, row: Mapping[str, Any]) -> Question:
    try:
        raw_id = str(row[spec.id_column])
        metadata = {
            key: value
            for key, value in row.items()
            if key not in {spec.id_column, spec.question_column, spec.answer_column}
            and key not in {"entity_pages", "search_results"}
        }
        metadata.update(
            {
                "source_repository": spec.repository,
                "source_revision": spec.revision,
                "source_split": spec.split,
                "source_row_id": raw_id,
            }
        )
        entities: tuple[str, ...] = ()
        if spec.name == "triviaqa":
            answer = row[spec.answer_column]
            if isinstance(answer, Mapping):
                entity = answer.get("matched_wiki_entity_name")
                entities = (str(entity),) if entity else ()
        return Question(
            question_id=f"{spec.name}:{raw_id}",
            benchmark=spec.name,
            text=str(row[spec.question_column]),
            aliases=_aliases(row[spec.answer_column]),
            split=spec.split,
            entities=entities,
            metadata=metadata,
        )
    except KeyError as exc:
        raise DataValidationError(
            f"row for {spec.name} does not match pinned schema; missing {exc.args[0]!r}"
        ) from exc


def load_hf_benchmark(spec: BenchmarkSpec, *, streaming: bool = False) -> Iterator[Question]:
    """Load a commit-pinned Hub dataset and discard evidence/context fields.

    Network access is explicit: merely importing this module never contacts the
    Hub. `revision` has already been validated as an immutable commit SHA.
    """

    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as exc:
        raise OptionalDependencyError(
            "Hub loading requires the 'research' extra: uv sync --extra research"
        ) from exc

    dataset = load_dataset(
        spec.repository,
        spec.config,
        split=spec.split,
        revision=spec.revision,
        streaming=streaming,
    )
    for row in dataset:
        yield question_from_hf_row(spec, row)

"""Immutable source snapshots and canonical row reconstruction for frozen evaluations.

Confirmatory question files are only views over these source artifacts.  A
claimed repository name or row identifier is not evidence of membership: the
artifact bytes are verified first, then the canonical :class:`Question` is
reconstructed and compared with the frozen view.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from mfh.contracts import BenchmarkSpec, Question
from mfh.data.benchmarks import load_aa_csv, load_simpleqa_csv, question_from_hf_row
from mfh.errors import DataValidationError, OptionalDependencyError
from mfh.provenance import sha256_file

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    """One byte-exact benchmark source used by E9 or E10."""

    benchmark: str
    repository: str
    revision: str
    split: str
    identifier_prefix: str
    artifact_name: str
    artifact_sha256: str
    artifact_size_bytes: int
    artifact_format: str
    canonical_question_count: int
    confirmatory_eligible: bool = True

    def __post_init__(self) -> None:
        text_fields = (
            self.benchmark,
            self.repository,
            self.revision,
            self.split,
            self.identifier_prefix,
            self.artifact_name,
            self.artifact_format,
        )
        if any(not value.strip() for value in text_fields):
            raise DataValidationError("source-snapshot text fields must be non-empty")
        if Path(self.artifact_name).name != self.artifact_name:
            raise DataValidationError("source-snapshot artifact name must be a basename")
        if not _SHA256.fullmatch(self.artifact_sha256):
            raise DataValidationError("source-snapshot artifact requires a SHA-256")
        if self.artifact_size_bytes <= 0 or self.canonical_question_count <= 0:
            raise DataValidationError("source-snapshot sizes and counts must be positive")
        if not isinstance(self.confirmatory_eligible, bool):
            raise DataValidationError("source-snapshot eligibility must be boolean")


SOURCE_SNAPSHOTS: Mapping[str, SourceSnapshot] = MappingProxyType(
    {
        "triviaqa": SourceSnapshot(
            benchmark="triviaqa",
            repository="lighteval/trivia_qa",
            revision="d2ff7f468d3642dbd33123596331950db8a63d0e",
            split="train",
            identifier_prefix="triviaqa:",
            artifact_name="train-00000-of-00001-e93ee6c1ba181971.parquet",
            artifact_sha256="455223e4a3ba6977611914d96b1b812fe19a6cd4ea3aef4df381f093b297670c",
            artifact_size_bytes=55_391_498,
            artifact_format="triviaqa-parquet",
            canonical_question_count=76_523,
        ),
        "simpleqa_verified": SourceSnapshot(
            benchmark="simpleqa_verified",
            repository="google/simpleqa-verified",
            revision="0dc97e0d28d8233463e005cdc4475cc2a13ba2dc",
            split="eval",
            identifier_prefix="simpleqa:",
            artifact_name="simpleqa_verified.csv",
            artifact_sha256="b5db21155444763543fe31b67e7cf28ce2bb225742a5b889421c5f182e2f92f5",
            artifact_size_bytes=349_184,
            artifact_format="simpleqa-csv",
            canonical_question_count=1_000,
        ),
        "aa_omniscience_public_600": SourceSnapshot(
            benchmark="aa_omniscience_public_600",
            repository="ArtificialAnalysis/AA-Omniscience-Public",
            revision="4a8ffc87c4650054825fb767fe0da4a4fc97ff32",
            split="train",
            identifier_prefix="aa-public:",
            artifact_name="AA-Omniscience_dataset_public.csv",
            artifact_sha256="4829fc1d50b18ce3282bd6148baf71b172428c710658acd8b6e66183a34ce83c",
            artifact_size_bytes=145_337,
            artifact_format="aa-csv",
            canonical_question_count=600,
        ),
        "ifeval": SourceSnapshot(
            benchmark="ifeval",
            repository="google/IFEval",
            revision="966cd89545d6b6acfd7638bc708b98261ca58e84",
            split="train",
            identifier_prefix="ifeval:",
            artifact_name="ifeval_input_data.jsonl",
            artifact_sha256="6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2",
            artifact_size_bytes=207_111,
            artifact_format="ifeval-jsonl",
            canonical_question_count=541,
        ),
        "mmlu_pro": SourceSnapshot(
            benchmark="mmlu_pro",
            repository="TIGER-Lab/MMLU-Pro",
            revision="b189ec765aa7ed75c8acfea42df31fdae71f97be",
            split="test",
            identifier_prefix="mmlu_pro:",
            artifact_name="test-00000-of-00001.parquet",
            artifact_sha256="0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8",
            artifact_size_bytes=4_144_185,
            artifact_format="mmlu-pro-parquet",
            canonical_question_count=12_032,
        ),
        "wikitext103": SourceSnapshot(
            benchmark="wikitext103",
            repository="Salesforce/wikitext",
            revision="b08601e04326c79dfdd32d625aee71d232d685c3",
            split="test",
            identifier_prefix="wikitext103:",
            artifact_name="test-00000-of-00001.parquet",
            artifact_sha256="5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91",
            artifact_size_bytes=732_610,
            artifact_format="wikitext-parquet",
            canonical_question_count=2_891,
        ),
        "xstest": SourceSnapshot(
            benchmark="xstest",
            repository="paul-rottger/xstest",
            revision="d7bb5bd738c1fcbc36edd83d5e7d1b71a3e2d84d",
            split="test",
            identifier_prefix="xstest:",
            artifact_name="xstest_prompts.csv",
            artifact_sha256="11783fb294ed017473ee53c207d71f2161c7672c8d0b037501e78387f801cb5a",
            artifact_size_bytes=38_719,
            artifact_format="xstest-csv",
            canonical_question_count=250,
        ),
        "strongreject_or_harmbench": SourceSnapshot(
            benchmark="strongreject_or_harmbench",
            repository="alexandrasouly/strongreject",
            revision="f7cad6c17e624e21d8df2278e918ae1dddb4cb56",
            split="evaluation",
            identifier_prefix="strongreject:",
            artifact_name="strongreject_dataset.csv",
            artifact_sha256="4dd70357e4ff8b5d0ba5ebafecab5d6dd5633ce8046e3dd1c8bd93e64de44381",
            artifact_size_bytes=56_359,
            artifact_format="strongreject-csv",
            canonical_question_count=313,
        ),
        "language_consistency": SourceSnapshot(
            benchmark="language_consistency",
            repository="local/mfh-language-consistency-v1",
            revision="d4d401ad9e54ff17220a091c6ece546f9968fa81d58c5c4aa7c09dd2aaf5eacb",
            split="evaluation",
            identifier_prefix="language_consistency:",
            artifact_name="language-consistency-v1.json",
            artifact_sha256="d4d401ad9e54ff17220a091c6ece546f9968fa81d58c5c4aa7c09dd2aaf5eacb",
            artifact_size_bytes=1_245,
            artifact_format="language-consistency-json",
            canonical_question_count=500,
            confirmatory_eligible=False,
        ),
    }
)


def verify_source_artifact(snapshot: SourceSnapshot, path: str | Path) -> Path:
    """Return a resolved source path only after exact byte and shape checks."""

    source = Path(path).resolve()
    if source.is_symlink() or not source.is_file():
        raise DataValidationError(f"{snapshot.benchmark} source must be a regular file")
    try:
        size = source.stat().st_size
        digest = sha256_file(source)
    except OSError as exc:
        raise DataValidationError(
            f"cannot inspect {snapshot.benchmark} source artifact: {exc}"
        ) from exc
    if size != snapshot.artifact_size_bytes or digest != snapshot.artifact_sha256:
        raise DataValidationError(
            f"{snapshot.benchmark} source artifact differs from its pinned bytes"
        )
    return source


def _source_metadata(snapshot: SourceSnapshot, row_id: str) -> dict[str, str]:
    return {
        "source_repository": snapshot.repository,
        "source_revision": snapshot.revision,
        "source_split": snapshot.split,
        "source_row_id": row_id,
    }


def _parquet_rows(path: Path, columns: list[str]) -> Iterator[Mapping[str, Any]]:
    try:
        import pyarrow.parquet as parquet  # type: ignore[import-untyped]
    except ImportError as exc:
        raise OptionalDependencyError(
            "parquet source verification requires the 'research' extra: uv sync --extra research"
        ) from exc
    try:
        parquet_file = parquet.ParquetFile(path)
        for batch in parquet_file.iter_batches(columns=columns, batch_size=4_096):
            yield from batch.to_pylist()
    except Exception as exc:
        raise DataValidationError(f"cannot parse pinned parquet source {path}: {exc}") from exc


def _triviaqa(path: Path, snapshot: SourceSnapshot) -> Iterator[Question]:
    spec = BenchmarkSpec(
        name=snapshot.benchmark,
        repository=snapshot.repository,
        revision=snapshot.revision,
        config="rc.nocontext",
        split=snapshot.split,
        format="parquet",
        question_column="question",
        answer_column="answer",
        id_column="question_id",
    )
    observed: dict[str, Question] = {}
    for row in _parquet_rows(path, ["question", "question_id", "question_source", "answer"]):
        question = question_from_hf_row(spec, row)
        prior = observed.setdefault(question.question_id, question)
        if prior != question:
            raise DataValidationError(
                f"duplicate TriviaQA row {question.question_id!r} has conflicting content"
            )
        if prior is question:
            yield question


def _ifeval(path: Path, snapshot: SourceSnapshot) -> Iterator[Question]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
                if not isinstance(row, Mapping) or set(row) != {
                    "key",
                    "prompt",
                    "instruction_id_list",
                    "kwargs",
                }:
                    raise DataValidationError("IFEval row has an unexpected schema")
                row_id = str(row["key"])
                metadata: dict[str, Any] = {
                    "instruction_id_list": row["instruction_id_list"],
                    "kwargs": row["kwargs"],
                    **_source_metadata(snapshot, row_id),
                }
                yield Question(
                    question_id=f"{snapshot.identifier_prefix}{row_id}",
                    benchmark=snapshot.benchmark,
                    text=str(row["prompt"]),
                    aliases=("__ifeval_official_instruction_checker__",),
                    split=snapshot.split,
                    metadata=metadata,
                )
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise DataValidationError(
                    f"invalid IFEval source row at {path}:{line_number}: {exc}"
                ) from exc


def _mmlu_pro(path: Path, snapshot: SourceSnapshot) -> Iterator[Question]:
    columns = ["question_id", "question", "options", "answer", "answer_index", "category", "src"]
    for row in _parquet_rows(path, columns):
        row_id = str(row["question_id"])
        options = row["options"]
        answer_index = row["answer_index"]
        if not isinstance(options, list) or not isinstance(answer_index, int):
            raise DataValidationError("MMLU-Pro options or answer index has an invalid schema")
        if answer_index < 0 or answer_index >= len(options):
            raise DataValidationError("MMLU-Pro answer index is outside its option list")
        yield Question(
            question_id=f"{snapshot.identifier_prefix}{row_id}",
            benchmark=snapshot.benchmark,
            text=str(row["question"]),
            aliases=(str(options[answer_index]), str(row["answer"])),
            split=snapshot.split,
            entities=(str(row["category"]),),
            metadata={
                "options": options,
                "answer_index": answer_index,
                "category": row["category"],
                "src": row["src"],
                **_source_metadata(snapshot, row_id),
            },
        )


def _wikitext(path: Path, snapshot: SourceSnapshot) -> Iterator[Question]:
    row_index = 0
    for row in _parquet_rows(path, ["text"]):
        text = str(row["text"])
        current = row_index
        row_index += 1
        if not text.strip():
            continue
        row_id = str(current)
        yield Question(
            question_id=f"{snapshot.identifier_prefix}{row_id}",
            benchmark=snapshot.benchmark,
            text=text,
            aliases=("__wikitext_official_perplexity_scorer__",),
            split=snapshot.split,
            metadata=_source_metadata(snapshot, row_id),
        )


def _xstest(path: Path, snapshot: SourceSnapshot) -> Iterator[Question]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "prompt", "type", "label", "focus", "note"}
        if set(reader.fieldnames or ()) != required:
            raise DataValidationError("XSTest source columns differ from the pinned schema")
        for row in reader:
            if row["label"] != "safe":
                continue
            row_id = row["id"].strip()
            yield Question(
                question_id=f"{snapshot.identifier_prefix}{row_id}",
                benchmark=snapshot.benchmark,
                text=row["prompt"],
                aliases=("__xstest_safe_prompt_refusal_grader__",),
                split=snapshot.split,
                metadata={
                    "type": row["type"],
                    "label": row["label"],
                    "focus": row["focus"],
                    "note": row["note"],
                    **_source_metadata(snapshot, row_id),
                },
            )


def _strongreject(path: Path, snapshot: SourceSnapshot) -> Iterator[Question]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or ()) != {"category", "source", "forbidden_prompt"}:
            raise DataValidationError("StrongREJECT columns differ from the pinned schema")
        for index, row in enumerate(reader, start=1):
            row_id = str(index)
            yield Question(
                question_id=f"{snapshot.identifier_prefix}{row_id}",
                benchmark=snapshot.benchmark,
                text=row["forbidden_prompt"],
                aliases=("__strongreject_official_safety_grader__",),
                split=snapshot.split,
                entities=(row["category"],),
                metadata={
                    "category": row["category"],
                    "source": row["source"],
                    **_source_metadata(snapshot, row_id),
                },
            )


def _language_consistency(path: Path, snapshot: SourceSnapshot) -> Iterator[Question]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot parse language-consistency source: {exc}") from exc
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "suite_id",
        "description",
        "items_per_language",
        "operand_rule",
        "languages",
    }:
        raise DataValidationError("language-consistency source has an invalid schema")
    languages = payload["languages"]
    if (
        payload["schema_version"] != 1
        or payload["suite_id"] != "mfh-language-consistency-v1"
        or payload["items_per_language"] != 100
        or payload["operand_rule"] != {"left": "17 + index", "right": "3 + ((index * 7) % 19)"}
        or not isinstance(languages, Mapping)
        or set(languages) != {"de", "en", "es", "fr", "ja"}
    ):
        raise DataValidationError("language-consistency generator contract changed")
    for language in sorted(languages):
        specification = languages[language]
        if not isinstance(specification, Mapping) or set(specification) != {
            "name",
            "prompt",
            "answer",
        }:
            raise DataValidationError("language-consistency language schema changed")
        for index in range(100):
            left = 17 + index
            right = 3 + ((index * 7) % 19)
            result = left + right
            row_id = f"{language}:{index:03d}"
            substitutions = {"left": left, "right": right, "result": result}
            yield Question(
                question_id=f"{snapshot.identifier_prefix}{row_id}",
                benchmark=snapshot.benchmark,
                text=str(specification["prompt"]).format(**substitutions),
                aliases=(str(specification["answer"]).format(**substitutions),),
                split=snapshot.split,
                metadata={
                    "requested_language": language,
                    "language_name": specification["name"],
                    "suite_id": payload["suite_id"],
                    **_source_metadata(snapshot, row_id),
                },
            )


def iter_source_questions(snapshot: SourceSnapshot, path: str | Path) -> Iterator[Question]:
    """Reconstruct all canonical questions from a verified source snapshot."""

    source = verify_source_artifact(snapshot, path)
    if snapshot.artifact_format == "triviaqa-parquet":
        values = _triviaqa(source, snapshot)
    elif snapshot.artifact_format == "simpleqa-csv":
        values = load_simpleqa_csv(source)
    elif snapshot.artifact_format == "aa-csv":
        values = load_aa_csv(source)
    elif snapshot.artifact_format == "ifeval-jsonl":
        values = _ifeval(source, snapshot)
    elif snapshot.artifact_format == "mmlu-pro-parquet":
        values = _mmlu_pro(source, snapshot)
    elif snapshot.artifact_format == "wikitext-parquet":
        values = _wikitext(source, snapshot)
    elif snapshot.artifact_format == "xstest-csv":
        values = _xstest(source, snapshot)
    elif snapshot.artifact_format == "strongreject-csv":
        values = _strongreject(source, snapshot)
    elif snapshot.artifact_format == "language-consistency-json":
        values = _language_consistency(source, snapshot)
    else:
        raise DataValidationError(
            f"unsupported source format for {snapshot.benchmark}: {snapshot.artifact_format}"
        )
    count = 0
    seen: set[str] = set()
    for question in values:
        count += 1
        if question.question_id in seen:
            raise DataValidationError(
                f"pinned {snapshot.benchmark} source contains duplicate question IDs"
            )
        seen.add(question.question_id)
        yield question
    if count != snapshot.canonical_question_count:
        raise DataValidationError(
            f"pinned {snapshot.benchmark} source yielded {count} canonical rows, "
            f"expected {snapshot.canonical_question_count}"
        )


def source_question_index(
    snapshot: SourceSnapshot,
    path: str | Path,
) -> Mapping[str, Question]:
    """Materialize an immutable canonical question index for membership checks."""

    return MappingProxyType(
        {question.question_id: question for question in iter_source_questions(snapshot, path)}
    )


def validate_source_membership(
    snapshot: SourceSnapshot,
    path: str | Path,
    questions: tuple[Question, ...],
) -> None:
    """Require each selected question to equal its byte-derived canonical row."""

    selected = {question.question_id: question for question in questions}
    if len(selected) != len(questions):
        raise DataValidationError(
            f"selected {snapshot.benchmark} questions contain duplicate identifiers"
        )
    found: set[str] = set()
    for canonical in iter_source_questions(snapshot, path):
        claimed = selected.get(canonical.question_id)
        if claimed is None:
            continue
        if claimed != canonical:
            raise DataValidationError(
                f"frozen {snapshot.benchmark} row {canonical.question_id!r} differs "
                "from its pinned source"
            )
        found.add(canonical.question_id)
    missing = set(selected) - found
    if missing:
        examples = sorted(missing)[:3]
        raise DataValidationError(
            f"frozen {snapshot.benchmark} questions are absent from the pinned source: {examples}"
        )

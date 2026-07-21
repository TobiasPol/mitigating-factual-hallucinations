"""Human-review-bound TriviaQA splits authorized for E1 and later phases."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mfh.contracts import Question
from mfh.data.contamination_review import verify_contamination_review_result
from mfh.data.io import read_questions, write_questions
from mfh.data.splits import (
    ResearchSplit,
    SplitPlan,
    assert_disjoint,
    exclude_exact_duplicate_groups,
    make_research_splits,
)
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.inference.transformers_snapshot import reject_symlink_path_components
from mfh.provenance import sha256_file, sha256_path, stable_hash

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SPLIT_NAMES = tuple(f"{split.value}.jsonl" for split in ResearchSplit)
_EXTERNAL_FILES = {
    "simpleqa_verified": "simpleqa-eval.jsonl",
    "aa_omniscience_public_600": "aa-eval.jsonl",
}
_FILES = frozenset(
    (*_SPLIT_NAMES, *_EXTERNAL_FILES.values(), "curation-report.json", "manifest.json")
)
_VERIFIED_REVIEWED_SPLITS = object()


@dataclass(frozen=True, slots=True)
class VerifiedReviewedSplits:
    """Capability issued only by full live reviewed-split replay."""

    directory: Path
    manifest_digest: str
    fingerprint: str
    _verification_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        expected_seal = (
            _VERIFIED_REVIEWED_SPLITS,
            self.directory,
            self.manifest_digest,
            self.fingerprint,
        )
        if self._verification_token != expected_seal:
            raise DataValidationError(
                "reviewed-split capabilities must come from full live verification"
            )


_AUTHORIZED_REVIEWED_SPLITS: dict[int, tuple[VerifiedReviewedSplits, Path, str, str]] = {}


def _assert_authorized_reviewed_splits(value: object) -> VerifiedReviewedSplits:
    if not isinstance(value, VerifiedReviewedSplits):
        raise DataValidationError(
            "E1 creation requires reviewed splits issued by full live verification"
        )
    registered = _AUTHORIZED_REVIEWED_SPLITS.get(id(value))
    expected = (value, value.directory, value.manifest_digest, value.fingerprint)
    if registered != expected or registered[0] is not value:
        raise DataValidationError(
            "reviewed-split capability was not issued by full live verification"
        )
    return value


def _require_sha256(value: str, context: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise DataValidationError(f"{context} must be a lowercase SHA-256 digest")


def _read_json(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot read {context}: {exc}") from exc
    if not isinstance(value, dict):
        raise DataValidationError(f"{context} must be a JSON object")
    return value


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        raise FrozenArtifactError(f"refusing to overwrite frozen JSON: {path}") from None
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _descriptor(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def _verify_inventory(directory: str | Path) -> Path:
    root = reject_symlink_path_components(directory, "reviewed TriviaQA split bundle")
    if not root.is_dir():
        raise DataValidationError("reviewed TriviaQA split bundle must be a regular directory")
    files: set[str] = set()
    directories: set[str] = set()
    for item in root.rglob("*"):
        if item.is_symlink():
            raise DataValidationError("reviewed TriviaQA split bundle cannot contain symlinks")
        relative = item.relative_to(root).as_posix()
        if item.is_file():
            files.add(relative)
        elif item.is_dir():
            directories.add(relative)
        else:
            raise DataValidationError("reviewed TriviaQA split bundle contains a special file")
    if files != _FILES or directories:
        raise DataValidationError("reviewed TriviaQA split bundle inventory differs")
    return root


def _verify_digest(value: Mapping[str, Any], field: str, context: str) -> None:
    body = dict(value)
    digest = body.pop(field, None)
    if type(digest) is not str or digest != stable_hash(body):
        raise DataValidationError(f"{context} {field} differs")


def _materials(
    source: Path,
    *,
    plan: SplitPlan,
) -> tuple[dict[ResearchSplit, tuple[Any, ...]], dict[str, Any], dict[str, Any]]:
    questions = tuple(read_questions(source))
    curation = exclude_exact_duplicate_groups(questions)
    result = make_research_splits(curation.questions, plan, require_exact_sizes=True)
    assert_disjoint(result)
    curation_body: dict[str, Any] = {
        "schema_version": 1,
        "source_questions_sha256": sha256_file(source),
        "curation": curation.report.to_dict(),
    }
    curation_report = {**curation_body, "manifest_digest": stable_hash(curation_body)}
    report = {
        **asdict(result.report),
        "curation_manifest_digest": curation_report["manifest_digest"],
    }
    return dict(result.splits), curation_report, report


def _external_materials(
    review_inputs: Mapping[str, Any],
) -> tuple[dict[str, tuple[Question, ...]], dict[str, Path]]:
    value = review_inputs.get("target_sources")
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != len(_EXTERNAL_FILES)
    ):
        raise DataValidationError("reviewed splits require both contamination target sources")
    questions_by_benchmark: dict[str, tuple[Question, ...]] = {}
    paths_by_benchmark: dict[str, Path] = {}
    for item in value:
        path = Path(item).absolute()
        questions = tuple(read_questions(path))
        benchmarks = {question.benchmark for question in questions}
        if len(benchmarks) != 1:
            raise DataValidationError("contamination target source mixes benchmark identities")
        benchmark = next(iter(benchmarks))
        if benchmark not in _EXTERNAL_FILES or benchmark in questions_by_benchmark:
            raise DataValidationError("contamination target benchmark identity differs")
        questions_by_benchmark[benchmark] = questions
        paths_by_benchmark[benchmark] = path
    if set(questions_by_benchmark) != set(_EXTERNAL_FILES):
        raise DataValidationError("reviewed splits lack a required external benchmark")
    return questions_by_benchmark, paths_by_benchmark


def _manifest_body(
    root: Path,
    *,
    review_result_manifest_digest: str,
    reviewed_clean_source_sha256: str,
    plan: SplitPlan,
    report: Mapping[str, Any],
    external_questions: Mapping[str, tuple[Question, ...]],
    external_sources: Mapping[str, Path],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "purpose": "human-reviewed-contamination-controlled-triviaqa-splits",
        "review_result_manifest_digest": review_result_manifest_digest,
        "reviewed_clean_source_sha256": reviewed_clean_source_sha256,
        "split_plan": asdict(plan),
        "split_report": dict(report),
        "split_question_ids_sha256": {
            split.value: stable_hash(
                [question.question_id for question in read_questions(root / f"{split.value}.jsonl")]
            )
            for split in ResearchSplit
        },
        "external_benchmarks": {
            benchmark: {
                "artifact": _EXTERNAL_FILES[benchmark],
                "source_sha256": sha256_file(external_sources[benchmark]),
                "question_count": len(external_questions[benchmark]),
                "question_ids_sha256": stable_hash(
                    [question.question_id for question in external_questions[benchmark]]
                ),
            }
            for benchmark in sorted(_EXTERNAL_FILES)
        },
        "artifacts": {
            name: _descriptor(root / name) for name in sorted(_FILES - {"manifest.json"})
        },
        "status": "complete",
        "scientific_eligible": True,
    }


def write_reviewed_split_bundle(
    directory: str | Path,
    *,
    review_result_directory: str | Path,
    expected_review_result_manifest_digest: str,
    review_queue_directory: str | Path,
    expected_review_queue_manifest_digest: str,
    review_inputs: Mapping[str, Any],
    plan: SplitPlan | None = None,
) -> Mapping[str, Any]:
    """Publish deterministic splits only from the fully reviewed clean source."""

    plan = plan or SplitPlan()
    review_manifest = verify_contamination_review_result(
        review_result_directory,
        expected_manifest_digest=expected_review_result_manifest_digest,
        review_queue_directory=review_queue_directory,
        expected_review_queue_manifest_digest=expected_review_queue_manifest_digest,
        **review_inputs,
    )
    source = Path(review_result_directory).absolute() / "reviewed-clean-source.jsonl"
    splits, curation_report, report = _materials(source, plan=plan)
    external_questions, external_sources = _external_materials(review_inputs)
    output = reject_symlink_path_components(directory, "reviewed TriviaQA split output")
    if output.exists():
        raise FrozenArtifactError(f"refusing to overwrite reviewed TriviaQA splits: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        for split in ResearchSplit:
            write_questions(stage / f"{split.value}.jsonl", splits[split])
        for benchmark, filename in _EXTERNAL_FILES.items():
            write_questions(stage / filename, external_questions[benchmark])
        _write_json_once(stage / "curation-report.json", curation_report)
        body = _manifest_body(
            stage,
            review_result_manifest_digest=expected_review_result_manifest_digest,
            reviewed_clean_source_sha256=str(review_manifest["reviewed_clean_source_sha256"]),
            plan=plan,
            report=report,
            external_questions=external_questions,
            external_sources=external_sources,
        )
        manifest = {**body, "manifest_digest": stable_hash(body)}
        _write_json_once(stage / "manifest.json", manifest)
        os.replace(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return verify_reviewed_split_bundle(
        output,
        expected_manifest_digest=str(manifest["manifest_digest"]),
        review_result_directory=review_result_directory,
        expected_review_result_manifest_digest=expected_review_result_manifest_digest,
        review_queue_directory=review_queue_directory,
        expected_review_queue_manifest_digest=expected_review_queue_manifest_digest,
        review_inputs=review_inputs,
        plan=plan,
    )


def verify_reviewed_split_bundle(
    directory: str | Path,
    *,
    expected_manifest_digest: str,
    review_result_directory: str | Path,
    expected_review_result_manifest_digest: str,
    review_queue_directory: str | Path,
    expected_review_queue_manifest_digest: str,
    review_inputs: Mapping[str, Any],
    plan: SplitPlan | None = None,
) -> Mapping[str, Any]:
    """Replay reviewed splits from the live finalized human-review result."""

    _require_sha256(expected_manifest_digest, "reviewed split manifest")
    plan = plan or SplitPlan()
    review_manifest = verify_contamination_review_result(
        review_result_directory,
        expected_manifest_digest=expected_review_result_manifest_digest,
        review_queue_directory=review_queue_directory,
        expected_review_queue_manifest_digest=expected_review_queue_manifest_digest,
        **review_inputs,
    )
    source = Path(review_result_directory).absolute() / "reviewed-clean-source.jsonl"
    expected_splits, expected_curation, report = _materials(source, plan=plan)
    external_questions, external_sources = _external_materials(review_inputs)
    root = _verify_inventory(directory)
    for split in ResearchSplit:
        if tuple(read_questions(root / f"{split.value}.jsonl")) != expected_splits[split]:
            raise DataValidationError(f"reviewed TriviaQA {split.value} split differs")
    for benchmark, filename in _EXTERNAL_FILES.items():
        if tuple(read_questions(root / filename)) != external_questions[benchmark]:
            raise DataValidationError(f"reviewed {benchmark} evaluation schedule differs")
    if _read_json(root / "curation-report.json", "reviewed split curation") != expected_curation:
        raise DataValidationError("reviewed TriviaQA split curation differs")
    manifest = _read_json(root / "manifest.json", "reviewed split manifest")
    _verify_digest(manifest, "manifest_digest", "reviewed split manifest")
    expected_body = _manifest_body(
        root,
        review_result_manifest_digest=expected_review_result_manifest_digest,
        reviewed_clean_source_sha256=str(review_manifest["reviewed_clean_source_sha256"]),
        plan=plan,
        report=report,
        external_questions=external_questions,
        external_sources=external_sources,
    )
    body = dict(manifest)
    digest = body.pop("manifest_digest", None)
    if body != expected_body or digest != expected_manifest_digest:
        raise DataValidationError("reviewed TriviaQA split manifest declarations differ")
    return manifest


def authorize_reviewed_split_bundle(
    directory: str | Path,
    *,
    expected_manifest_digest: str,
    review_result_directory: str | Path,
    expected_review_result_manifest_digest: str,
    review_queue_directory: str | Path,
    expected_review_queue_manifest_digest: str,
    review_inputs: Mapping[str, Any],
    plan: SplitPlan | None = None,
) -> VerifiedReviewedSplits:
    """Issue the only reviewed-split capability accepted by E1 creation."""

    root = Path(directory).absolute()
    manifest = verify_reviewed_split_bundle(
        root,
        expected_manifest_digest=expected_manifest_digest,
        review_result_directory=review_result_directory,
        expected_review_result_manifest_digest=expected_review_result_manifest_digest,
        review_queue_directory=review_queue_directory,
        expected_review_queue_manifest_digest=expected_review_queue_manifest_digest,
        review_inputs=review_inputs,
        plan=plan,
    )
    fingerprint = sha256_path(root)
    manifest_digest = str(manifest["manifest_digest"])
    capability = VerifiedReviewedSplits(
        directory=root,
        manifest_digest=manifest_digest,
        fingerprint=fingerprint,
        _verification_token=(
            _VERIFIED_REVIEWED_SPLITS,
            root,
            manifest_digest,
            fingerprint,
        ),
    )
    _AUTHORIZED_REVIEWED_SPLITS[id(capability)] = (
        capability,
        capability.directory,
        capability.manifest_digest,
        capability.fingerprint,
    )
    return capability


def validate_reviewed_split_snapshot(directory: str | Path) -> Mapping[str, Any]:
    """Validate the packaged shape bound by a prior live-verification capability."""

    root = _verify_inventory(directory)
    manifest = _read_json(root / "manifest.json", "reviewed split manifest")
    _verify_digest(manifest, "manifest_digest", "reviewed split manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("purpose") != "human-reviewed-contamination-controlled-triviaqa-splits"
        or manifest.get("status") != "complete"
        or manifest.get("scientific_eligible") is not True
    ):
        raise DataValidationError("reviewed TriviaQA splits are not scientifically eligible")
    for field_name in (
        "review_result_manifest_digest",
        "reviewed_clean_source_sha256",
        "manifest_digest",
    ):
        value = manifest.get(field_name)
        if type(value) is not str:
            raise DataValidationError(f"reviewed TriviaQA split {field_name} is invalid")
        _require_sha256(value, f"reviewed TriviaQA split {field_name}")
    if manifest.get("artifacts") != {
        name: _descriptor(root / name) for name in sorted(_FILES - {"manifest.json"})
    }:
        raise DataValidationError("reviewed TriviaQA split artifacts differ")
    external = manifest.get("external_benchmarks")
    if not isinstance(external, Mapping) or set(external) != set(_EXTERNAL_FILES):
        raise DataValidationError("reviewed split external benchmark declarations differ")
    for benchmark, filename in _EXTERNAL_FILES.items():
        descriptor = external.get(benchmark)
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
            "artifact",
            "source_sha256",
            "question_count",
            "question_ids_sha256",
        }:
            raise DataValidationError("reviewed split external benchmark descriptor differs")
        if descriptor.get("artifact") != filename:
            raise DataValidationError("reviewed split external benchmark artifact differs")
        for field_name in ("source_sha256", "question_ids_sha256"):
            value = descriptor.get(field_name)
            if type(value) is not str:
                raise DataValidationError("reviewed split external benchmark digest is invalid")
            _require_sha256(value, "reviewed split external benchmark")
        if type(descriptor.get("question_count")) is not int:
            raise DataValidationError("reviewed split external benchmark count is invalid")
    return manifest

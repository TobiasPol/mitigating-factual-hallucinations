"""Blinded human review and finalization for semantic-contamination candidates."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from mfh.config import SemanticContaminationProtocol
from mfh.data.io import read_questions, write_questions
from mfh.data.semantic_contamination import verify_contamination_bundle
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.inference.transformers_snapshot import reject_symlink_path_components
from mfh.provenance import sha256_file, stable_hash

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LABELS = {"distinct", "overlap"}
_QUEUE_FILES = frozenset(
    {
        "annotation-template.csv",
        "blind-items.jsonl",
        "manifest.json",
        "operator-bindings.jsonl",
        "rubric.md",
    }
)
_RESULT_FILES = frozenset(
    {
        "annotations.csv",
        "decisions.jsonl",
        "excluded-source-ids.json",
        "manifest.json",
        "reviewed-clean-source.jsonl",
        "reviewer-attestation.json",
    }
)
_ANNOTATION_FIELDS = (
    "review_id",
    "source_question",
    "target_question",
    "label",
    "notes",
)
_ATTESTATION = (
    "I manually compared every blinded question pair using the frozen overlap rubric "
    "without automated label generation."
)
_RUBRIC = """# Semantic-contamination manual-review rubric

Review every pair using only the two displayed questions. Do not use similarity scores,
automatic-match flags, source IDs, target IDs, or model-generated labels.

Choose exactly one label:

- `overlap`: both questions seek the same underlying fact or would be correctly answered by
  the same fact, even if wording, scope qualifiers, or requested answer form differ.
- `distinct`: the questions concern different facts. Topic similarity, shared entities, or one
  question being a nearby superlative is not sufficient for overlap.

If outside factual research is needed to understand the wording, record it in `notes`. Confirmed
`overlap` rows are removed conservatively from the TriviaQA representation-training source.
Every row must be labeled. Do not edit `review_id` values or add/remove rows.
"""


def _require_sha256(value: str, context: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise DataValidationError(f"{context} must be a lowercase SHA-256 digest")


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _read_json(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot read {context}: {exc}") from exc
    if not isinstance(value, dict):
        raise DataValidationError(f"{context} must be a JSON object")
    return value


def _read_jsonl(path: Path, context: str) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise DataValidationError(f"{context} row {line_number} must be a JSON object")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot read {context}: {exc}") from exc
    return tuple(rows)


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        raise FrozenArtifactError(f"refusing to overwrite frozen JSON: {path}") from None
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode())
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl_once(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        raise FrozenArtifactError(f"refusing to overwrite frozen JSONL: {path}") from None
    with os.fdopen(descriptor, "wb") as handle:
        for row in rows:
            handle.write(_json_bytes(row))
        handle.flush()
        os.fsync(handle.fileno())


def _write_csv_once(
    path: Path,
    rows: Sequence[Mapping[str, str]],
) -> None:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        raise FrozenArtifactError(f"refusing to overwrite frozen CSV: {path}") from None
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_ANNOTATION_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _spreadsheet_text(value: object) -> str:
    text = str(value)
    return "'" + text if text.startswith(("=", "+", "-", "@")) else text


def _read_annotations(path: Path, *, require_labels: bool) -> tuple[dict[str, str], ...]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != _ANNOTATION_FIELDS:
                raise DataValidationError("contamination annotations have the wrong CSV fields")
            rows = []
            for line_number, raw in enumerate(reader, start=2):
                row = {field: str(raw.get(field, "")).strip() for field in _ANNOTATION_FIELDS}
                if not row["review_id"]:
                    raise DataValidationError(
                        f"contamination annotation row {line_number} lacks a review ID"
                    )
                if require_labels and row["label"] not in _LABELS:
                    raise DataValidationError(
                        f"contamination annotation row {line_number} has an invalid label"
                    )
                if not require_labels and row["label"]:
                    raise DataValidationError("frozen annotation template must be unlabeled")
                rows.append(row)
    except OSError as exc:
        raise DataValidationError(f"cannot read contamination annotations: {exc}") from exc
    identifiers = [row["review_id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise DataValidationError("contamination annotations repeat a review ID")
    return tuple(rows)


def _descriptor(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def _verify_inventory(root: Path, expected: frozenset[str], context: str) -> Path:
    root = reject_symlink_path_components(root, context)
    if not root.is_dir():
        raise DataValidationError(f"{context} must be a regular directory")
    files: set[str] = set()
    directories: set[str] = set()
    for item in root.rglob("*"):
        if item.is_symlink():
            raise DataValidationError(f"{context} cannot contain symlinks")
        relative = item.relative_to(root).as_posix()
        if item.is_file():
            files.add(relative)
        elif item.is_dir():
            directories.add(relative)
        else:
            raise DataValidationError(f"{context} contains a special file")
    if files != expected or directories:
        raise DataValidationError(f"{context} inventory differs")
    return root


def _verify_manifest(
    root: Path,
    *,
    expected_digest: str,
    context: str,
) -> dict[str, Any]:
    _require_sha256(expected_digest, context)
    manifest = _read_json(root / "manifest.json", context)
    body = dict(manifest)
    digest = body.pop("manifest_digest", None)
    if digest != stable_hash(body) or digest != expected_digest:
        raise DataValidationError(f"{context} identity differs")
    return manifest


def _verified_contamination(
    directory: str | Path,
    *,
    expected_protocol: SemanticContaminationProtocol,
    model_directory: str | Path,
    triviaqa_source: str | Path,
    target_sources: Sequence[str | Path],
    expected_manifest_digest: str,
) -> tuple[Path, Mapping[str, Any]]:
    manifest = verify_contamination_bundle(
        directory,
        expected_protocol=expected_protocol,
        model_directory=model_directory,
        triviaqa_source=triviaqa_source,
        target_sources=target_sources,
        expected_manifest_digest=expected_manifest_digest,
    )
    return Path(directory).absolute(), manifest


def _queue_materials(
    contamination_root: Path,
    contamination_manifest: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise DataValidationError("contamination review seed must be a non-negative integer")
    queue = _read_jsonl(
        contamination_root / "manual-review-queue.jsonl",
        "semantic-contamination manual-review queue",
    )
    manual_review = contamination_manifest.get("manual_review")
    if (
        not isinstance(manual_review, Mapping)
        or manual_review.get("status") != "pending"
        or manual_review.get("selection_count") != len(queue)
        or manual_review.get("selection_sha256") != stable_hash(queue)
        or len(queue) == 0
    ):
        raise DataValidationError("semantic-contamination manual-review declaration differs")
    bindings = []
    for queue_index, row in enumerate(queue):
        required = {
            "automatic_semantic_match",
            "similarity",
            "source_question_id",
            "source_text",
            "target_question_id",
            "target_text",
        }
        if set(row) != required:
            raise DataValidationError("semantic-contamination review row fields differ")
        review_id = (
            "cr-"
            + stable_hash(
                {
                    "schema_version": 1,
                    "seed": seed,
                    "selection_sha256": manual_review["selection_sha256"],
                    "source_question_id": row["source_question_id"],
                    "target_question_id": row["target_question_id"],
                }
            )[:20]
        )
        bindings.append(
            {
                "schema_version": 1,
                "review_id": review_id,
                "queue_index": queue_index,
                "source_question_id": row["source_question_id"],
                "target_question_id": row["target_question_id"],
                "source_text": row["source_text"],
                "target_text": row["target_text"],
                "source_text_sha256": stable_hash(row["source_text"]),
                "target_text_sha256": stable_hash(row["target_text"]),
                "similarity": row["similarity"],
                "automatic_semantic_match": row["automatic_semantic_match"],
            }
        )
    if len({row["review_id"] for row in bindings}) != len(bindings):
        raise DataValidationError("semantic-contamination review IDs collide")
    ordered = tuple(
        sorted(
            bindings,
            key=lambda row: (
                stable_hash({"seed": seed, "review_id": row["review_id"]}),
                str(row["review_id"]),
            ),
        )
    )
    blind = tuple(
        {
            "schema_version": 1,
            "review_id": row["review_id"],
            "source_question": row["source_text"],
            "target_question": row["target_text"],
        }
        for row in ordered
    )
    return ordered, blind


def prepare_contamination_review_queue(
    directory: str | Path,
    *,
    contamination_directory: str | Path,
    expected_protocol: SemanticContaminationProtocol,
    model_directory: str | Path,
    triviaqa_source: str | Path,
    target_sources: Sequence[str | Path],
    expected_contamination_manifest_digest: str,
    seed: int = 17,
) -> Mapping[str, Any]:
    """Publish a deterministic blinded copy of the frozen top-k review queue."""

    contamination_root, contamination_manifest = _verified_contamination(
        contamination_directory,
        expected_protocol=expected_protocol,
        model_directory=model_directory,
        triviaqa_source=triviaqa_source,
        target_sources=target_sources,
        expected_manifest_digest=expected_contamination_manifest_digest,
    )
    bindings, blind = _queue_materials(contamination_root, contamination_manifest, seed=seed)
    output = reject_symlink_path_components(directory, "contamination review queue output")
    if output.exists():
        raise FrozenArtifactError(f"refusing to overwrite contamination review queue: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        _write_jsonl_once(stage / "blind-items.jsonl", blind)
        _write_jsonl_once(stage / "operator-bindings.jsonl", bindings)
        _write_csv_once(
            stage / "annotation-template.csv",
            tuple(
                {
                    "review_id": str(row["review_id"]),
                    "source_question": _spreadsheet_text(row["source_question"]),
                    "target_question": _spreadsheet_text(row["target_question"]),
                    "label": "",
                    "notes": "",
                }
                for row in blind
            ),
        )
        (stage / "rubric.md").write_text(_RUBRIC, encoding="utf-8")
        artifact_names = sorted(_QUEUE_FILES - {"manifest.json"})
        body: dict[str, Any] = {
            "schema_version": 1,
            "purpose": "blinded-semantic-contamination-manual-review",
            "contamination_manifest_digest": expected_contamination_manifest_digest,
            "contamination_review_selection_sha256": contamination_manifest["manual_review"][
                "selection_sha256"
            ],
            "seed": seed,
            "review_count": len(bindings),
            "review_ids_sha256": stable_hash([row["review_id"] for row in bindings]),
            "bindings_digest": stable_hash(bindings),
            "rubric_sha256": sha256_file(stage / "rubric.md"),
            "artifacts": {name: _descriptor(stage / name) for name in artifact_names},
            "status": "awaiting-human-annotation",
            "scientific_eligible": False,
        }
        manifest = {**body, "manifest_digest": stable_hash(body)}
        _write_json_once(stage / "manifest.json", manifest)
        os.replace(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return verify_contamination_review_queue(
        output,
        expected_manifest_digest=str(manifest["manifest_digest"]),
        contamination_directory=contamination_directory,
        expected_protocol=expected_protocol,
        model_directory=model_directory,
        triviaqa_source=triviaqa_source,
        target_sources=target_sources,
        expected_contamination_manifest_digest=expected_contamination_manifest_digest,
    )


def verify_contamination_review_queue(
    directory: str | Path,
    *,
    expected_manifest_digest: str,
    contamination_directory: str | Path,
    expected_protocol: SemanticContaminationProtocol,
    model_directory: str | Path,
    triviaqa_source: str | Path,
    target_sources: Sequence[str | Path],
    expected_contamination_manifest_digest: str,
) -> Mapping[str, Any]:
    """Replay a blinded queue from its exact contamination candidate evidence."""

    contamination_root, contamination_manifest = _verified_contamination(
        contamination_directory,
        expected_protocol=expected_protocol,
        model_directory=model_directory,
        triviaqa_source=triviaqa_source,
        target_sources=target_sources,
        expected_manifest_digest=expected_contamination_manifest_digest,
    )
    root = _verify_inventory(Path(directory).absolute(), _QUEUE_FILES, "contamination review queue")
    manifest = _verify_manifest(
        root,
        expected_digest=expected_manifest_digest,
        context="contamination review queue manifest",
    )
    seed = manifest.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise DataValidationError("contamination review queue seed differs")
    bindings, blind = _queue_materials(contamination_root, contamination_manifest, seed=seed)
    if _read_jsonl(root / "operator-bindings.jsonl", "review bindings") != bindings:
        raise DataValidationError("contamination review bindings differ")
    if _read_jsonl(root / "blind-items.jsonl", "blind review items") != blind:
        raise DataValidationError("blind contamination review items differ")
    annotations = _read_annotations(root / "annotation-template.csv", require_labels=False)
    if annotations != tuple(
        {
            "review_id": str(row["review_id"]),
            "source_question": _spreadsheet_text(row["source_question"]),
            "target_question": _spreadsheet_text(row["target_question"]),
            "label": "",
            "notes": "",
        }
        for row in blind
    ):
        raise DataValidationError("contamination annotation template differs")
    if (root / "rubric.md").read_text(encoding="utf-8") != _RUBRIC:
        raise DataValidationError("contamination review rubric differs")
    artifact_names = sorted(_QUEUE_FILES - {"manifest.json"})
    expected_body = {
        "schema_version": 1,
        "purpose": "blinded-semantic-contamination-manual-review",
        "contamination_manifest_digest": expected_contamination_manifest_digest,
        "contamination_review_selection_sha256": contamination_manifest["manual_review"][
            "selection_sha256"
        ],
        "seed": seed,
        "review_count": len(bindings),
        "review_ids_sha256": stable_hash([row["review_id"] for row in bindings]),
        "bindings_digest": stable_hash(bindings),
        "rubric_sha256": sha256_file(root / "rubric.md"),
        "artifacts": {name: _descriptor(root / name) for name in artifact_names},
        "status": "awaiting-human-annotation",
        "scientific_eligible": False,
    }
    body = dict(manifest)
    body.pop("manifest_digest", None)
    if body != expected_body:
        raise DataValidationError("contamination review queue declarations differ")
    return manifest


def _review_attestation(path: Path) -> dict[str, Any]:
    value = _read_json(path, "contamination reviewer attestation")
    if set(value) != {"schema_version", "reviewer_id", "reviewed_at", "attestation"}:
        raise DataValidationError("contamination reviewer attestation fields differ")
    reviewer_id = value.get("reviewer_id")
    reviewed_at = value.get("reviewed_at")
    if type(reviewer_id) is not str or not reviewer_id.strip():
        raise DataValidationError("contamination reviewer ID must be non-empty")
    if type(reviewed_at) is not str:
        raise DataValidationError("contamination review timestamp must be text")
    try:
        timestamp = datetime.fromisoformat(reviewed_at)
    except ValueError as exc:
        raise DataValidationError("contamination review timestamp is invalid") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise DataValidationError("contamination review timestamp must include a timezone")
    if value.get("schema_version") != 1 or value.get("attestation") != _ATTESTATION:
        raise DataValidationError("contamination reviewer attestation differs")
    return value


def _decision_materials(
    queue_root: Path,
    annotations_path: Path,
    attestation_path: Path,
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[str, ...],
    dict[str, Any],
    tuple[dict[str, str], ...],
]:
    bindings = _read_jsonl(queue_root / "operator-bindings.jsonl", "review bindings")
    annotations = _read_annotations(annotations_path, require_labels=True)
    expected_ids = [str(row["review_id"]) for row in bindings]
    if [row["review_id"] for row in annotations] != expected_ids:
        raise DataValidationError(
            "contamination annotations must preserve every blinded row in frozen order"
        )
    for binding, annotation in zip(bindings, annotations, strict=True):
        if annotation["source_question"] != _spreadsheet_text(binding["source_text"]) or annotation[
            "target_question"
        ] != _spreadsheet_text(binding["target_text"]):
            raise DataValidationError("contamination annotations changed a blinded question")
    attestation = _review_attestation(attestation_path)
    decisions = tuple(
        {
            "schema_version": 1,
            **binding,
            "label": annotation["label"],
            "notes": annotation["notes"],
            "reviewer_id": attestation["reviewer_id"],
        }
        for binding, annotation in zip(bindings, annotations, strict=True)
    )
    excluded = tuple(
        sorted({str(row["source_question_id"]) for row in decisions if row["label"] == "overlap"})
    )
    return decisions, excluded, attestation, annotations


def finalize_contamination_review(
    directory: str | Path,
    *,
    review_queue_directory: str | Path,
    expected_review_queue_manifest_digest: str,
    annotations: str | Path,
    reviewer_attestation: str | Path,
    contamination_directory: str | Path,
    expected_protocol: SemanticContaminationProtocol,
    model_directory: str | Path,
    triviaqa_source: str | Path,
    target_sources: Sequence[str | Path],
    expected_contamination_manifest_digest: str,
) -> Mapping[str, Any]:
    """Freeze complete human decisions and the conservatively reviewed clean source."""

    queue_manifest = verify_contamination_review_queue(
        review_queue_directory,
        expected_manifest_digest=expected_review_queue_manifest_digest,
        contamination_directory=contamination_directory,
        expected_protocol=expected_protocol,
        model_directory=model_directory,
        triviaqa_source=triviaqa_source,
        target_sources=target_sources,
        expected_contamination_manifest_digest=expected_contamination_manifest_digest,
    )
    queue_root = Path(review_queue_directory).absolute()
    annotations_path = reject_symlink_path_components(annotations, "contamination annotations")
    attestation_path = reject_symlink_path_components(
        reviewer_attestation, "contamination reviewer attestation"
    )
    if not annotations_path.is_file() or not attestation_path.is_file():
        raise DataValidationError("contamination review inputs must be regular files")
    decisions, excluded, attestation, annotation_rows = _decision_materials(
        queue_root, annotations_path, attestation_path
    )
    contamination_root = Path(contamination_directory).absolute()
    base_clean = tuple(read_questions(contamination_root / "clean-source.jsonl"))
    base_ids = {question.question_id for question in base_clean}
    reviewed_clean = tuple(
        question for question in base_clean if question.question_id not in set(excluded)
    )
    newly_excluded = tuple(sorted(set(excluded) & base_ids))
    output = reject_symlink_path_components(directory, "contamination review result output")
    if output.exists():
        raise FrozenArtifactError(f"refusing to overwrite contamination review result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        _write_jsonl_once(stage / "decisions.jsonl", decisions)
        _write_csv_once(stage / "annotations.csv", annotation_rows)
        _write_json_once(stage / "reviewer-attestation.json", attestation)
        write_questions(stage / "reviewed-clean-source.jsonl", reviewed_clean)
        excluded_body: dict[str, Any] = {
            "schema_version": 1,
            "manual_overlap_source_ids": list(excluded),
            "manual_overlap_source_ids_sha256": stable_hash(excluded),
            "newly_excluded_source_ids": list(newly_excluded),
            "newly_excluded_source_ids_sha256": stable_hash(newly_excluded),
        }
        excluded_payload = {
            **excluded_body,
            "excluded_source_ids_digest": stable_hash(excluded_body),
        }
        _write_json_once(stage / "excluded-source-ids.json", excluded_payload)
        counts = Counter(str(row["label"]) for row in decisions)
        artifact_names = sorted(_RESULT_FILES - {"manifest.json"})
        body: dict[str, Any] = {
            "schema_version": 1,
            "purpose": "completed-semantic-contamination-manual-review",
            "contamination_manifest_digest": expected_contamination_manifest_digest,
            "review_queue_manifest_digest": expected_review_queue_manifest_digest,
            "review_queue_bindings_digest": queue_manifest["bindings_digest"],
            "reviewer_id": attestation["reviewer_id"],
            "reviewed_at": attestation["reviewed_at"],
            "counts": {
                "reviewed": len(decisions),
                "overlap": counts["overlap"],
                "distinct": counts["distinct"],
                "new_source_exclusions": len(newly_excluded),
                "reviewed_clean_source": len(reviewed_clean),
            },
            "decisions_digest": stable_hash(decisions),
            "excluded_source_ids_digest": excluded_payload["excluded_source_ids_digest"],
            "reviewed_clean_source_sha256": sha256_file(stage / "reviewed-clean-source.jsonl"),
            "artifacts": {name: _descriptor(stage / name) for name in artifact_names},
            "status": "complete",
            "scientific_eligible": True,
        }
        manifest = {**body, "manifest_digest": stable_hash(body)}
        _write_json_once(stage / "manifest.json", manifest)
        os.replace(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return verify_contamination_review_result(
        output,
        expected_manifest_digest=str(manifest["manifest_digest"]),
        review_queue_directory=review_queue_directory,
        expected_review_queue_manifest_digest=expected_review_queue_manifest_digest,
        contamination_directory=contamination_directory,
        expected_protocol=expected_protocol,
        model_directory=model_directory,
        triviaqa_source=triviaqa_source,
        target_sources=target_sources,
        expected_contamination_manifest_digest=expected_contamination_manifest_digest,
    )


def verify_contamination_review_result(
    directory: str | Path,
    *,
    expected_manifest_digest: str,
    review_queue_directory: str | Path,
    expected_review_queue_manifest_digest: str,
    contamination_directory: str | Path,
    expected_protocol: SemanticContaminationProtocol,
    model_directory: str | Path,
    triviaqa_source: str | Path,
    target_sources: Sequence[str | Path],
    expected_contamination_manifest_digest: str,
) -> Mapping[str, Any]:
    """Replay finalized human review evidence and its reviewed clean source."""

    queue_manifest = verify_contamination_review_queue(
        review_queue_directory,
        expected_manifest_digest=expected_review_queue_manifest_digest,
        contamination_directory=contamination_directory,
        expected_protocol=expected_protocol,
        model_directory=model_directory,
        triviaqa_source=triviaqa_source,
        target_sources=target_sources,
        expected_contamination_manifest_digest=expected_contamination_manifest_digest,
    )
    root = _verify_inventory(
        Path(directory).absolute(), _RESULT_FILES, "contamination review result"
    )
    manifest = _verify_manifest(
        root,
        expected_digest=expected_manifest_digest,
        context="contamination review result manifest",
    )
    decisions, excluded, attestation, annotation_rows = _decision_materials(
        Path(review_queue_directory).absolute(),
        root / "annotations.csv",
        root / "reviewer-attestation.json",
    )
    if _read_jsonl(root / "decisions.jsonl", "contamination decisions") != decisions:
        raise DataValidationError("contamination review decisions differ")
    if _read_annotations(root / "annotations.csv", require_labels=True) != annotation_rows:
        raise DataValidationError("contamination review annotations differ")
    if _read_json(root / "reviewer-attestation.json", "reviewer attestation") != attestation:
        raise DataValidationError("contamination reviewer attestation differs")
    contamination_root = Path(contamination_directory).absolute()
    base_clean = tuple(read_questions(contamination_root / "clean-source.jsonl"))
    base_ids = {question.question_id for question in base_clean}
    reviewed_clean = tuple(
        question for question in base_clean if question.question_id not in set(excluded)
    )
    if tuple(read_questions(root / "reviewed-clean-source.jsonl")) != reviewed_clean:
        raise DataValidationError("reviewed contamination clean source differs")
    newly_excluded = tuple(sorted(set(excluded) & base_ids))
    excluded_body: dict[str, Any] = {
        "schema_version": 1,
        "manual_overlap_source_ids": list(excluded),
        "manual_overlap_source_ids_sha256": stable_hash(excluded),
        "newly_excluded_source_ids": list(newly_excluded),
        "newly_excluded_source_ids_sha256": stable_hash(newly_excluded),
    }
    expected_excluded = {
        **excluded_body,
        "excluded_source_ids_digest": stable_hash(excluded_body),
    }
    if _read_json(root / "excluded-source-ids.json", "excluded source IDs") != expected_excluded:
        raise DataValidationError("contamination excluded source IDs differ")
    counts = Counter(str(row["label"]) for row in decisions)
    artifact_names = sorted(_RESULT_FILES - {"manifest.json"})
    expected_body = {
        "schema_version": 1,
        "purpose": "completed-semantic-contamination-manual-review",
        "contamination_manifest_digest": expected_contamination_manifest_digest,
        "review_queue_manifest_digest": expected_review_queue_manifest_digest,
        "review_queue_bindings_digest": queue_manifest["bindings_digest"],
        "reviewer_id": attestation["reviewer_id"],
        "reviewed_at": attestation["reviewed_at"],
        "counts": {
            "reviewed": len(decisions),
            "overlap": counts["overlap"],
            "distinct": counts["distinct"],
            "new_source_exclusions": len(newly_excluded),
            "reviewed_clean_source": len(reviewed_clean),
        },
        "decisions_digest": stable_hash(decisions),
        "excluded_source_ids_digest": expected_excluded["excluded_source_ids_digest"],
        "reviewed_clean_source_sha256": sha256_file(root / "reviewed-clean-source.jsonl"),
        "artifacts": {name: _descriptor(root / name) for name in artifact_names},
        "status": "complete",
        "scientific_eligible": True,
    }
    body = dict(manifest)
    body.pop("manifest_digest", None)
    if body != expected_body:
        raise DataValidationError("contamination review result declarations differ")
    return manifest


def reviewer_attestation_text() -> str:
    """Return the exact attestation sentence required by finalization."""

    return _ATTESTATION

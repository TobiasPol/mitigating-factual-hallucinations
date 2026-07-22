"""Exact resumable label-shuffled centroid control for E3."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from mfh.contracts import ActivationSite, Outcome, PromptSpec, Question
from mfh.errors import ConfigurationError, DataValidationError, FrozenArtifactError
from mfh.experiments.e3_construction import (
    VerifiedE3ConstructionSnapshot,
    load_verified_e3_construction_snapshot,
    verify_e3_vector_bundle,
)
from mfh.experiments.e3_schedule import E3OperatingPoint, E3Protocol
from mfh.experiments.e3_selection import VerifiedE3StageSelection
from mfh.experiments.model_selection import validate_active_study_artifact_paths
from mfh.inference.vllm_research import (
    VllmPromptFeatureCubeOutput,
    VllmTeacherForcedCubeOutput,
)
from mfh.inference.vllm_runtime import VllmRenderedPrompt
from mfh.provenance import sha256_file, stable_hash

_SHA256 = frozenset("0123456789abcdef")
_EXTRACTIONS = ("M1-R", "M1-P")
_LABELS = (Outcome.CORRECT, Outcome.INCORRECT)
_INVENTORY = frozenset({"plan.json", "sessions.jsonl", "checkpoints"})
_UNIFIED_MEMORY_BYTES = 42_949_672_960


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _loads_json(value: str) -> Any:
    return json.loads(value, object_pairs_hook=_reject_duplicate_keys)


class E3ShuffleRuntime(Protocol):
    def render_prompt(
        self,
        prompt: PromptSpec,
        question: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> VllmRenderedPrompt: ...

    def prompt_feature_cube(
        self,
        rendered: VllmRenderedPrompt,
        *,
        layers: Sequence[int],
        sites: Sequence[ActivationSite],
    ) -> VllmPromptFeatureCubeOutput: ...

    def teacher_forced_cube(
        self,
        rendered: VllmRenderedPrompt,
        response: str,
        *,
        layers: Sequence[int],
        sites: Sequence[ActivationSite],
    ) -> VllmTeacherForcedCubeOutput: ...

    def runtime_identity(self) -> Mapping[str, Any]: ...


@dataclass(slots=True)
class _ShuffleAccumulator:
    processed_rows: int
    counts: np.ndarray[Any, Any]
    sums: np.ndarray[Any, Any]
    maximum_peak_memory_bytes: int
    wall_time_seconds: float


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _SHA256 for character in value)
    )


def _points(
    selection: VerifiedE3StageSelection, protocol: E3Protocol
) -> Mapping[str, E3OperatingPoint]:
    if isinstance(selection, VerifiedE3StageSelection):
        selection.assert_current()
    if (
        not isinstance(selection, VerifiedE3StageSelection)
        or selection.stage != "scope"
        or selection.falsified
        or set(selection.selected) != set(_EXTRACTIONS)
    ):
        raise DataValidationError("E3 shuffled control requires successful scope selection")
    for name, point in selection.selected.items():
        if (
            point.extraction_method != name
            or point.layer not in protocol.candidate_layers
            or point.site is not protocol.primary_replication_site
            or point.standardized_alpha not in protocol.standardized_alphas
            or point.token_scope not in protocol.token_scopes
        ):
            raise DataValidationError("E3 shuffled-control operating point differs")
    return MappingProxyType(dict(selection.selected))


def _source_rows(
    snapshot: VerifiedE3ConstructionSnapshot,
) -> tuple[tuple[int, str, Outcome], ...]:
    rows = tuple(
        (record.sequence, record.question_id, record.outcome)
        for record in snapshot.generations
        if record.prompt_id == "P0-neutral" and record.outcome in _LABELS
    )
    labels = {value[2] for value in rows}
    if not rows or labels != set(_LABELS):
        raise DataValidationError("E3 shuffled control requires both P0 C/I classes")
    return rows


def _shuffle_labels(
    rows: Sequence[tuple[int, str, Outcome]], *, seed: int
) -> tuple[tuple[Outcome, ...], int, int, str, str]:
    original = tuple(value[2] for value in rows)
    for attempt in range(1_000):
        permutation = sorted(
            range(len(rows)),
            key=lambda index: hashlib.sha256(
                f"e3-label-shuffle:{seed}:{attempt}:{rows[index][0]}:{rows[index][1]}".encode()
            ).digest(),
        )
        shuffled = tuple(original[index] for index in permutation)
        changed = sum(left is not right for left, right in zip(original, shuffled, strict=True))
        inversion = all(left is not right for left, right in zip(original, shuffled, strict=True))
        if changed and not inversion:
            if {label: shuffled.count(label) for label in _LABELS} != {
                label: original.count(label) for label in _LABELS
            }:
                raise AssertionError("label permutation did not preserve class counts")
            return (
                shuffled,
                changed,
                attempt,
                stable_hash([value.value for value in original]),
                stable_hash([value.value for value in shuffled]),
            )
    raise DataValidationError("E3 could not construct a nondegenerate label permutation")


def _source_context(
    *,
    construction_directory: str | Path,
    vector_bundle_directory: str | Path,
    questions: Sequence[Question],
    prompts: Mapping[str, PromptSpec],
    scope_selection: VerifiedE3StageSelection,
    protocol: E3Protocol,
) -> tuple[
    VerifiedE3ConstructionSnapshot,
    Mapping[str, Any],
    Mapping[str, E3OperatingPoint],
    tuple[tuple[int, str, Outcome], ...],
    tuple[Outcome, ...],
    Mapping[str, Any],
]:
    snapshot = load_verified_e3_construction_snapshot(
        construction_directory,
        questions=questions,
        prompts=prompts,
        protocol=protocol,
    )
    vector = verify_e3_vector_bundle(
        vector_bundle_directory,
        work_directory=construction_directory,
        questions=questions,
        prompts=prompts,
        protocol=protocol,
    )
    if scope_selection.source_plan_identity != snapshot.plan["plan_identity"]:
        raise FrozenArtifactError(
            "E3 shuffled-control selection belongs to another construction plan"
        )
    points = _points(scope_selection, protocol)
    rows = _source_rows(snapshot)
    shuffled, changed, attempt, original_digest, shuffled_digest = _shuffle_labels(
        rows, seed=protocol.seed
    )
    labels = MappingProxyType(
        {
            "changed_labels": changed,
            "permutation_attempt": attempt,
            "original_labels_digest": original_digest,
            "shuffled_labels_digest": shuffled_digest,
        }
    )
    return snapshot, vector, points, rows, shuffled, labels


def _plan_body(
    *,
    snapshot: VerifiedE3ConstructionSnapshot,
    vector: Mapping[str, Any],
    points: Mapping[str, E3OperatingPoint],
    rows: Sequence[tuple[int, str, Outcome]],
    labels: Mapping[str, Any],
    scope_selection: VerifiedE3StageSelection,
    runtime_identity: Mapping[str, Any],
    protocol: E3Protocol,
    checkpoint_rows: int,
) -> dict[str, Any]:
    if type(checkpoint_rows) is not int or checkpoint_rows <= 0:
        raise DataValidationError("E3 shuffle checkpoint size is invalid")
    identity = json.loads(json.dumps(dict(runtime_identity), sort_keys=True, allow_nan=False))
    if identity != dict(snapshot.plan["runtime_identity"]):
        raise FrozenArtifactError("E3 shuffle runtime differs from construction runtime")
    body = {
        "schema_version": 1,
        "phase": "E3-shuffled-control",
        "runner_source_sha256": sha256_file(Path(__file__)),
        "construction_plan_identity": snapshot.plan["plan_identity"],
        "construction_generation_chain_head": snapshot.generation_chain_head,
        "vector_data_fingerprint": vector["data_fingerprint"],
        "scope_selection_digest": scope_selection.selection_digest,
        "operating_points": {
            name: {
                "layer": point.layer,
                "site": point.site.value,
                "standardized_alpha": point.standardized_alpha,
                "token_scope": point.token_scope.value,
            }
            for name, point in sorted(points.items())
        },
        "source_rows_digest": stable_hash(
            [
                {"sequence": sequence, "question_id": question, "outcome": outcome.value}
                for sequence, question, outcome in rows
            ]
        ),
        "expected_rows": len(rows),
        "hidden_width": int(snapshot.plan["hidden_width"]),
        "checkpoint_rows": checkpoint_rows,
        "label_permutation": dict(labels),
        "runtime_identity": identity,
        "scientific_eligible": bool(
            snapshot.scientific_eligible
            and vector["scientific_eligible"]
            and scope_selection.scientific_eligible
            and protocol.scientific_eligible
        ),
    }
    return body


def _plan(body: Mapping[str, Any]) -> dict[str, Any]:
    return {**body, "plan_identity": stable_hash(dict(body))}


def prepare_e3_shuffled_control_work(
    directory: str | Path,
    *,
    construction_directory: str | Path,
    vector_bundle_directory: str | Path,
    questions: Sequence[Question],
    prompts: Mapping[str, PromptSpec],
    scope_selection: VerifiedE3StageSelection,
    runtime_identity: Mapping[str, Any],
    protocol: E3Protocol | None = None,
    checkpoint_rows: int = 64,
) -> Mapping[str, Any]:
    normalized = validate_active_study_artifact_paths(
        {
            "E3 shuffled-control work": directory,
            "E3 construction work": construction_directory,
            "E3 vector bundle": vector_bundle_directory,
        }
    )
    directory = normalized["E3 shuffled-control work"]
    construction_directory = normalized["E3 construction work"]
    vector_bundle_directory = normalized["E3 vector bundle"]
    frozen_protocol = protocol or E3Protocol()
    snapshot, vector, points, rows, _shuffled, labels = _source_context(
        construction_directory=construction_directory,
        vector_bundle_directory=vector_bundle_directory,
        questions=questions,
        prompts=prompts,
        scope_selection=scope_selection,
        protocol=frozen_protocol,
    )
    plan = _plan(
        _plan_body(
            snapshot=snapshot,
            vector=vector,
            points=points,
            rows=rows,
            labels=labels,
            scope_selection=scope_selection,
            runtime_identity=runtime_identity,
            protocol=frozen_protocol,
            checkpoint_rows=checkpoint_rows,
        )
    )
    destination = Path(directory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite E3 shuffle work: {destination}")
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        (stage / "checkpoints").mkdir()
        (stage / "plan.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (stage / "sessions.jsonl").touch()
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return MappingProxyType(plan)


def _read_plan(path: Path) -> dict[str, Any]:
    try:
        value = _loads_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read E3 shuffle plan: {exc}") from exc
    if type(value) is not dict:
        raise FrozenArtifactError("E3 shuffle plan is invalid")
    body = dict(value)
    identity = body.pop("plan_identity", None)
    if (
        identity != stable_hash(body)
        or body.get("runner_source_sha256") != sha256_file(Path(__file__))
        or body.get("schema_version") != 1
        or body.get("phase") != "E3-shuffled-control"
    ):
        raise FrozenArtifactError("E3 shuffle plan identity differs")
    return value


def _context(
    directory: str | Path,
    *,
    construction_directory: str | Path,
    vector_bundle_directory: str | Path,
    questions: Sequence[Question],
    prompts: Mapping[str, PromptSpec],
    scope_selection: VerifiedE3StageSelection,
    protocol: E3Protocol,
) -> tuple[
    Path,
    dict[str, Any],
    VerifiedE3ConstructionSnapshot,
    Mapping[str, E3OperatingPoint],
    tuple[tuple[int, str, Outcome], ...],
    tuple[Outcome, ...],
]:
    work = Path(directory)
    if (
        work.is_symlink()
        or not work.is_dir()
        or {value.name for value in work.iterdir()} != _INVENTORY
        or not (work / "checkpoints").is_dir()
        or any(value.is_symlink() for value in work.iterdir())
    ):
        raise FrozenArtifactError("E3 shuffle work inventory differs")
    plan = _read_plan(work / "plan.json")
    snapshot, vector, points, rows, shuffled, labels = _source_context(
        construction_directory=construction_directory,
        vector_bundle_directory=vector_bundle_directory,
        questions=questions,
        prompts=prompts,
        scope_selection=scope_selection,
        protocol=protocol,
    )
    expected = _plan(
        _plan_body(
            snapshot=snapshot,
            vector=vector,
            points=points,
            rows=rows,
            labels=labels,
            scope_selection=scope_selection,
            runtime_identity=plan["runtime_identity"],
            protocol=protocol,
            checkpoint_rows=plan["checkpoint_rows"],
        )
    )
    if plan != expected:
        raise FrozenArtifactError("E3 shuffle plan differs from live sources")
    return work, plan, snapshot, points, rows, shuffled


def _empty(width: int) -> _ShuffleAccumulator:
    return _ShuffleAccumulator(
        processed_rows=0,
        counts=np.zeros((len(_EXTRACTIONS), len(_LABELS)), dtype=np.int64),
        sums=np.zeros((len(_EXTRACTIONS), len(_LABELS), width), dtype=np.float64),
        maximum_peak_memory_bytes=0,
        wall_time_seconds=0.0,
    )


def _checkpoint_paths(directory: Path) -> tuple[Path, ...]:
    paths = tuple(sorted(directory.glob("checkpoint-*.npz")))
    if {value.name for value in directory.iterdir()} != {value.name for value in paths}:
        raise FrozenArtifactError("E3 shuffle checkpoint inventory differs")
    return paths


def _write_checkpoint(
    directory: Path,
    *,
    accumulator: _ShuffleAccumulator,
    plan_identity: str,
    previous_sha256: str | None,
    index: int,
) -> str:
    metadata = {
        "schema_version": 1,
        "checkpoint_index": index,
        "processed_rows": accumulator.processed_rows,
        "maximum_peak_memory_bytes": accumulator.maximum_peak_memory_bytes,
        "wall_time_seconds": accumulator.wall_time_seconds,
        "plan_identity": plan_identity,
        "previous_checkpoint_sha256": previous_sha256,
    }
    destination = directory / f"checkpoint-{index:08d}.npz"
    if destination.exists():
        raise FrozenArtifactError("refusing to overwrite E3 shuffle checkpoint")
    descriptor, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                counts=accumulator.counts,
                sums=accumulator.sums,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return sha256_file(destination)


def _load_checkpoints(
    directory: Path,
    *,
    plan_identity: str,
    width: int,
    shuffled: Sequence[Outcome],
) -> tuple[_ShuffleAccumulator, str | None, int]:
    accumulator = _empty(width)
    previous: str | None = None
    paths = _checkpoint_paths(directory)
    for index, path in enumerate(paths):
        if path.name != f"checkpoint-{index:08d}.npz":
            raise FrozenArtifactError("E3 shuffle checkpoint numbering differs")
        try:
            with np.load(path, allow_pickle=False) as value:
                if set(value.files) != {"metadata", "counts", "sums"}:
                    raise DataValidationError("checkpoint arrays differ")
                metadata = _loads_json(str(value["metadata"].item()))
                if (
                    type(metadata) is not dict
                    or set(metadata)
                    != {
                        "schema_version",
                        "checkpoint_index",
                        "processed_rows",
                        "maximum_peak_memory_bytes",
                        "wall_time_seconds",
                        "plan_identity",
                        "previous_checkpoint_sha256",
                    }
                    or type(metadata.get("schema_version")) is not int
                    or type(metadata.get("checkpoint_index")) is not int
                    or type(metadata.get("processed_rows")) is not int
                    or type(metadata.get("maximum_peak_memory_bytes")) is not int
                    or metadata["maximum_peak_memory_bytes"] < 0
                    or type(metadata.get("wall_time_seconds")) is not float
                    or not np.isfinite(metadata["wall_time_seconds"])
                    or metadata["wall_time_seconds"] < 0
                ):
                    raise DataValidationError("checkpoint metadata schema differs")
                candidate = _ShuffleAccumulator(
                    processed_rows=metadata["processed_rows"],
                    counts=value["counts"].copy(),
                    sums=value["sums"].copy(),
                    maximum_peak_memory_bytes=metadata["maximum_peak_memory_bytes"],
                    wall_time_seconds=metadata["wall_time_seconds"],
                )
        except (OSError, ValueError, json.JSONDecodeError, DataValidationError) as exc:
            raise FrozenArtifactError(f"cannot read E3 shuffle checkpoint: {exc}") from exc
        expected_counts = np.zeros((len(_EXTRACTIONS), len(_LABELS)), dtype=np.int64)
        for label in shuffled[: candidate.processed_rows]:
            expected_counts[:, _LABELS.index(label)] += 1
        expected_metadata = {
            "schema_version": 1,
            "checkpoint_index": index,
            "processed_rows": candidate.processed_rows,
            "maximum_peak_memory_bytes": candidate.maximum_peak_memory_bytes,
            "wall_time_seconds": candidate.wall_time_seconds,
            "plan_identity": plan_identity,
            "previous_checkpoint_sha256": previous,
        }
        if (
            metadata != expected_metadata
            or not 0 < candidate.processed_rows <= len(shuffled)
            or candidate.counts.dtype != np.int64
            or candidate.counts.shape != (2, 2)
            or not np.array_equal(candidate.counts, expected_counts)
            or candidate.sums.dtype != np.float64
            or candidate.sums.shape != (2, 2, width)
            or not np.isfinite(candidate.sums).all()
            or candidate.maximum_peak_memory_bytes
            < accumulator.maximum_peak_memory_bytes
            or candidate.wall_time_seconds < accumulator.wall_time_seconds
        ):
            raise FrozenArtifactError("E3 shuffle checkpoint state differs")
        accumulator = candidate
        previous = sha256_file(path)
    return accumulator, previous, len(paths)


def _repair_tails(work: Path) -> None:
    session = work / "sessions.jsonl"
    data = session.read_bytes()
    if data and not data.endswith(b"\n"):
        boundary = data.rfind(b"\n")
        tail = data[boundary + 1 :]
        try:
            parsed = _loads_json(tail.decode())
        except (UnicodeDecodeError, json.JSONDecodeError):
            repaired = data[: boundary + 1] if boundary >= 0 else b""
        else:
            repaired = data + b"\n" if type(parsed) is dict else data[: boundary + 1]
        session.write_bytes(repaired)
    for path in (work / "checkpoints").iterdir():
        if path.name.startswith(".checkpoint-"):
            if path.is_symlink() or not path.is_file():
                raise FrozenArtifactError("E3 shuffle orphan checkpoint is invalid")
            path.unlink()


def _sessions(path: Path, *, plan_identity: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous: str | None = None
    open_index: int | None = None
    last_processed = 0
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = _loads_json(line)
            if type(row) is not dict:
                raise DataValidationError("session row is invalid")
            body = dict(row)
            digest = body.pop("session_digest", None)
            if (
                digest != stable_hash(body)
                or body.get("previous_session_digest") != previous
                or body.get("plan_identity") != plan_identity
                or body.get("schema_version") != 1
            ):
                raise DataValidationError("session chain differs")
            event = body.get("event")
            index = body.get("session_index")
            if event == "start":
                if (
                    set(body)
                    != {
                        "schema_version",
                        "event",
                        "session_index",
                        "plan_identity",
                        "processed_rows",
                        "created_unix_ns",
                        "previous_session_digest",
                    }
                    or open_index is not None
                    or type(index) is not int
                    or index != sum(value["event"] == "start" for value in rows)
                    or type(body.get("processed_rows")) is not int
                    or body["processed_rows"] != last_processed
                    or type(body.get("created_unix_ns")) is not int
                    or body["created_unix_ns"] <= 0
                ):
                    raise DataValidationError("session start differs")
                open_index = index
            elif event == "end":
                if (
                    set(body)
                    != {
                        "schema_version",
                        "event",
                        "session_index",
                        "plan_identity",
                        "status",
                        "processed_rows",
                        "checkpoint_chain_head",
                        "peak_memory_bytes",
                        "wall_time_seconds",
                        "created_unix_ns",
                        "previous_session_digest",
                    }
                    or open_index != index
                    or body.get("status")
                    not in {"partial", "complete", "error", "interrupted-recovered"}
                    or type(body.get("processed_rows")) is not int
                    or body["processed_rows"] < last_processed
                    or type(body.get("peak_memory_bytes")) is not int
                    or body["peak_memory_bytes"] < 0
                    or type(body.get("wall_time_seconds")) is not float
                    or not np.isfinite(body["wall_time_seconds"])
                    or body["wall_time_seconds"] < 0
                    or type(body.get("created_unix_ns")) is not int
                    or body["created_unix_ns"] <= 0
                    or (
                        not _is_sha256(body.get("checkpoint_chain_head"))
                        and body.get("checkpoint_chain_head") is not None
                    )
                ):
                    raise DataValidationError("session end differs")
                last_processed = body["processed_rows"]
                open_index = None
            else:
                raise DataValidationError("session event differs")
            rows.append(row)
            previous = digest
    except (OSError, json.JSONDecodeError, TypeError, ValueError, DataValidationError) as exc:
        raise FrozenArtifactError(f"cannot read E3 shuffle sessions: {exc}") from exc
    return rows


def _append_session(path: Path, body: Mapping[str, Any], previous: str | None) -> str:
    value = {**body, "previous_session_digest": previous}
    digest = stable_hash(value)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({**value, "session_digest": digest}, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return digest


@contextmanager
def _lock(path: Path) -> Iterator[None]:
    with path.open("rb") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ConfigurationError("E3 shuffle work is already running") from exc
        yield


def _observation(
    *,
    runtime: E3ShuffleRuntime,
    rendered: VllmRenderedPrompt,
    response: str,
    points: Mapping[str, E3OperatingPoint],
    width: int,
) -> tuple[Mapping[str, np.ndarray[Any, Any]], int]:
    layers = tuple(dict.fromkeys(point.layer for point in points.values()))
    prompt = runtime.prompt_feature_cube(
        rendered, layers=layers, sites=(ActivationSite.POST_MLP,)
    )
    teacher = runtime.teacher_forced_cube(
        rendered, response, layers=layers, sites=(ActivationSite.POST_MLP,)
    )
    if teacher.response_text_sha256 != hashlib.sha256(response.encode()).hexdigest():
        raise DataValidationError("E3 shuffled response cube differs from journal")
    values: dict[str, np.ndarray[Any, Any]] = {}
    for name, point in points.items():
        raw = np.asarray(
            (
                teacher.activations[point.site][point.layer]
                if name == "M1-R"
                else prompt.activations[point.site][point.layer]
            ),
            dtype=np.float32,
        )
        if (
            raw.ndim != 2
            or raw.shape[0] <= 0
            or raw.shape[1] != width
            or not np.isfinite(raw).all()
        ):
            raise DataValidationError("E3 shuffled activation geometry differs")
        values[name] = raw.mean(axis=0, dtype=np.float64)
    return MappingProxyType(values), max(
        prompt.peak_memory_bytes, teacher.peak_memory_bytes
    )


def verify_e3_shuffled_control_work(
    directory: str | Path,
    *,
    construction_directory: str | Path,
    vector_bundle_directory: str | Path,
    questions: Sequence[Question],
    prompts: Mapping[str, PromptSpec],
    scope_selection: VerifiedE3StageSelection,
    protocol: E3Protocol | None = None,
    require_complete: bool = False,
) -> Mapping[str, Any]:
    frozen_protocol = protocol or E3Protocol()
    work, plan, _snapshot, _points_value, rows, shuffled = _context(
        directory,
        construction_directory=construction_directory,
        vector_bundle_directory=vector_bundle_directory,
        questions=questions,
        prompts=prompts,
        scope_selection=scope_selection,
        protocol=frozen_protocol,
    )
    accumulator, checkpoint_head, checkpoint_count = _load_checkpoints(
        work / "checkpoints",
        plan_identity=plan["plan_identity"],
        width=plan["hidden_width"],
        shuffled=shuffled,
    )
    sessions = _sessions(work / "sessions.jsonl", plan_identity=plan["plan_identity"])
    if sessions and sessions[-1]["event"] == "start":
        raise FrozenArtifactError("E3 shuffle session is unclosed")
    complete = accumulator.processed_rows == len(rows)
    if sessions and (
        sessions[-1]["processed_rows"] != accumulator.processed_rows
        or sessions[-1]["checkpoint_chain_head"] != checkpoint_head
        or (sessions[-1]["status"] == "complete") != complete
    ):
        raise FrozenArtifactError("E3 shuffle session differs from checkpoint")
    if not sessions and accumulator.processed_rows:
        raise FrozenArtifactError("E3 shuffle checkpoint lacks session evidence")
    if require_complete and not complete:
        raise FrozenArtifactError("E3 shuffled control work is incomplete")
    session_peak = max(
        (row["peak_memory_bytes"] for row in sessions if row["event"] == "end"),
        default=0,
    )
    peak = max(session_peak, accumulator.maximum_peak_memory_bytes)
    return MappingProxyType(
        {
            "valid": True,
            "complete": complete,
            "processed_rows": accumulator.processed_rows,
            "expected_rows": len(rows),
            "checkpoint_count": checkpoint_count,
            "checkpoint_chain_head": checkpoint_head,
            "maximum_peak_memory_bytes": peak,
            "memory_within_envelope": peak <= _UNIFIED_MEMORY_BYTES,
            "plan_identity": plan["plan_identity"],
            "scientific_eligible": bool(
                plan["scientific_eligible"] and peak <= _UNIFIED_MEMORY_BYTES
            ),
        }
    )


def run_e3_shuffled_control(
    directory: str | Path,
    *,
    construction_directory: str | Path,
    vector_bundle_directory: str | Path,
    questions: Sequence[Question],
    prompts: Mapping[str, PromptSpec],
    scope_selection: VerifiedE3StageSelection,
    runtime: E3ShuffleRuntime,
    protocol: E3Protocol | None = None,
    request_budget: int | None = None,
) -> Mapping[str, Any]:
    normalized = validate_active_study_artifact_paths(
        {
            "E3 shuffled-control work": directory,
            "E3 construction work": construction_directory,
            "E3 vector bundle": vector_bundle_directory,
        }
    )
    directory = normalized["E3 shuffled-control work"]
    construction_directory = normalized["E3 construction work"]
    vector_bundle_directory = normalized["E3 vector bundle"]
    frozen_protocol = protocol or E3Protocol()
    if request_budget is not None and (type(request_budget) is not int or request_budget <= 0):
        raise ConfigurationError("E3 shuffle request budget must be positive")
    work, plan, snapshot, points, rows, shuffled = _context(
        directory,
        construction_directory=construction_directory,
        vector_bundle_directory=vector_bundle_directory,
        questions=questions,
        prompts=prompts,
        scope_selection=scope_selection,
        protocol=frozen_protocol,
    )
    if json.loads(json.dumps(dict(runtime.runtime_identity()), sort_keys=True)) != plan[
        "runtime_identity"
    ]:
        raise FrozenArtifactError("E3 shuffle live runtime identity differs")
    question_map = {value.question_id: value for value in questions}
    records = {value.sequence: value for value in snapshot.generations}
    with _lock(work / "plan.json"):
        _repair_tails(work)
        accumulator, checkpoint_head, checkpoint_index = _load_checkpoints(
            work / "checkpoints",
            plan_identity=plan["plan_identity"],
            width=plan["hidden_width"],
            shuffled=shuffled,
        )
        durable_rows = accumulator.processed_rows
        sessions = _sessions(work / "sessions.jsonl", plan_identity=plan["plan_identity"])
        session_head = sessions[-1]["session_digest"] if sessions else None
        if sessions and sessions[-1]["event"] == "start":
            session_head = _append_session(
                work / "sessions.jsonl",
                {
                    "schema_version": 1,
                    "event": "end",
                    "session_index": sessions[-1]["session_index"],
                    "plan_identity": plan["plan_identity"],
                    "status": "interrupted-recovered",
                    "processed_rows": durable_rows,
                    "checkpoint_chain_head": checkpoint_head,
                    "peak_memory_bytes": accumulator.maximum_peak_memory_bytes,
                    "wall_time_seconds": accumulator.wall_time_seconds,
                    "created_unix_ns": time.time_ns(),
                },
                session_head,
            )
            sessions = _sessions(
                work / "sessions.jsonl", plan_identity=plan["plan_identity"]
            )
        session_index = sum(value["event"] == "start" for value in sessions)
        session_head = _append_session(
            work / "sessions.jsonl",
            {
                "schema_version": 1,
                "event": "start",
                "session_index": session_index,
                "plan_identity": plan["plan_identity"],
                "processed_rows": durable_rows,
                "created_unix_ns": time.time_ns(),
            },
            session_head,
        )
        handled = 0
        peak = accumulator.maximum_peak_memory_bytes
        status = "partial"
        started_ns = time.monotonic_ns()
        starting_wall_time = accumulator.wall_time_seconds
        try:
            while accumulator.processed_rows < len(rows) and (
                request_budget is None or handled < request_budget
            ):
                index = accumulator.processed_rows
                sequence, question_id, _outcome = rows[index]
                source = records[sequence]
                question = question_map[question_id]
                rendered = runtime.render_prompt(
                    prompts["P0-neutral"], question.text, metadata=question.metadata
                )
                if (
                    rendered.sha256 != source.rendered_prompt_sha256
                    or rendered.token_ids_sha256 != source.prompt_token_ids_sha256
                ):
                    raise FrozenArtifactError("E3 shuffle rerendered prompt differs")
                values, row_peak = _observation(
                    runtime=runtime,
                    rendered=rendered,
                    response=str(source.evidence["raw_output"]),
                    points=points,
                    width=plan["hidden_width"],
                )
                label_index = _LABELS.index(shuffled[index])
                for extraction_index, extraction in enumerate(_EXTRACTIONS):
                    accumulator.counts[extraction_index, label_index] += 1
                    accumulator.sums[extraction_index, label_index] += values[extraction]
                accumulator.processed_rows += 1
                handled += 1
                peak = max(peak, row_peak)
                accumulator.maximum_peak_memory_bytes = peak
                accumulator.wall_time_seconds = starting_wall_time + (
                    time.monotonic_ns() - started_ns
                ) / 1e9
                if accumulator.processed_rows % plan["checkpoint_rows"] == 0:
                    checkpoint_head = _write_checkpoint(
                        work / "checkpoints",
                        accumulator=accumulator,
                        plan_identity=plan["plan_identity"],
                        previous_sha256=checkpoint_head,
                        index=checkpoint_index,
                    )
                    checkpoint_index += 1
                    durable_rows = accumulator.processed_rows
            if accumulator.processed_rows > durable_rows:
                accumulator.wall_time_seconds = starting_wall_time + (
                    time.monotonic_ns() - started_ns
                ) / 1e9
                checkpoint_head = _write_checkpoint(
                    work / "checkpoints",
                    accumulator=accumulator,
                    plan_identity=plan["plan_identity"],
                    previous_sha256=checkpoint_head,
                    index=checkpoint_index,
                )
                durable_rows = accumulator.processed_rows
            status = "complete" if durable_rows == len(rows) else "partial"
        except BaseException:
            status = "error"
            if accumulator.processed_rows > durable_rows:
                try:
                    accumulator.wall_time_seconds = starting_wall_time + (
                        time.monotonic_ns() - started_ns
                    ) / 1e9
                    checkpoint_head = _write_checkpoint(
                        work / "checkpoints",
                        accumulator=accumulator,
                        plan_identity=plan["plan_identity"],
                        previous_sha256=checkpoint_head,
                        index=checkpoint_index,
                    )
                    durable_rows = accumulator.processed_rows
                except BaseException:
                    pass
            raise
        finally:
            _append_session(
                work / "sessions.jsonl",
                {
                    "schema_version": 1,
                    "event": "end",
                    "session_index": session_index,
                    "plan_identity": plan["plan_identity"],
                    "status": status,
                    "processed_rows": durable_rows,
                    "checkpoint_chain_head": checkpoint_head,
                    "peak_memory_bytes": peak,
                    "wall_time_seconds": accumulator.wall_time_seconds,
                    "created_unix_ns": time.time_ns(),
                },
                session_head,
            )
    return verify_e3_shuffled_control_work(
        work,
        construction_directory=construction_directory,
        vector_bundle_directory=vector_bundle_directory,
        questions=questions,
        prompts=prompts,
        scope_selection=scope_selection,
        protocol=frozen_protocol,
    )


def finalize_e3_shuffled_control_bundle(
    destination: str | Path,
    *,
    work_directory: str | Path,
    construction_directory: str | Path,
    vector_bundle_directory: str | Path,
    questions: Sequence[Question],
    prompts: Mapping[str, PromptSpec],
    scope_selection: VerifiedE3StageSelection,
    protocol: E3Protocol | None = None,
    allow_non_scientific: bool = False,
) -> Mapping[str, Any]:
    normalized = validate_active_study_artifact_paths(
        {
            "E3 shuffled-control bundle": destination,
            "E3 shuffled-control work": work_directory,
            "E3 construction work": construction_directory,
            "E3 vector bundle": vector_bundle_directory,
        }
    )
    destination = normalized["E3 shuffled-control bundle"]
    work_directory = normalized["E3 shuffled-control work"]
    construction_directory = normalized["E3 construction work"]
    vector_bundle_directory = normalized["E3 vector bundle"]
    frozen_protocol = protocol or E3Protocol()
    verification = verify_e3_shuffled_control_work(
        work_directory,
        construction_directory=construction_directory,
        vector_bundle_directory=vector_bundle_directory,
        questions=questions,
        prompts=prompts,
        scope_selection=scope_selection,
        protocol=frozen_protocol,
        require_complete=True,
    )
    if not verification["scientific_eligible"] and not allow_non_scientific:
        raise FrozenArtifactError("E3 shuffled control is not scientifically eligible")
    work, plan, _snapshot, _points_value, _rows, shuffled = _context(
        work_directory,
        construction_directory=construction_directory,
        vector_bundle_directory=vector_bundle_directory,
        questions=questions,
        prompts=prompts,
        scope_selection=scope_selection,
        protocol=frozen_protocol,
    )
    accumulator, checkpoint_head, _count = _load_checkpoints(
        work / "checkpoints",
        plan_identity=plan["plan_identity"],
        width=plan["hidden_width"],
        shuffled=shuffled,
    )
    means = accumulator.sums / accumulator.counts[..., None]
    differences = means[:, 0] - means[:, 1]
    norms = np.linalg.norm(differences, axis=1)
    if np.any(accumulator.counts <= 0) or np.any(norms <= 0) or not np.isfinite(norms).all():
        raise DataValidationError("E3 shuffled centroid is degenerate")
    directions = (differences / norms[:, None]).astype(np.float32)
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite E3 shuffled bundle: {output}")
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        tensor_path = stage / "directions.npz"
        with tensor_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                directions=directions,
                correct_counts=accumulator.counts[:, 0],
                incorrect_counts=accumulator.counts[:, 1],
            )
        body = {
            "schema_version": 1,
            "phase": "E3-shuffled-control",
            "plan_identity": plan["plan_identity"],
            "extraction_axis": list(_EXTRACTIONS),
            "hidden_width": plan["hidden_width"],
            "processed_rows": accumulator.processed_rows,
            "label_permutation": plan["label_permutation"],
            "scope_selection_digest": plan["scope_selection_digest"],
            "vector_data_fingerprint": plan["vector_data_fingerprint"],
            "checkpoint_chain_head": checkpoint_head,
            "directions_sha256": sha256_file(tensor_path),
            "scientific_eligible": verification["scientific_eligible"],
        }
        metadata = {**body, "metadata_digest": stable_hash(body)}
        (stage / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return verify_e3_shuffled_control_bundle(
        output,
        work_directory=work,
        construction_directory=construction_directory,
        vector_bundle_directory=vector_bundle_directory,
        questions=questions,
        prompts=prompts,
        scope_selection=scope_selection,
        protocol=frozen_protocol,
    )


def verify_e3_shuffled_control_bundle(
    directory: str | Path,
    *,
    work_directory: str | Path,
    construction_directory: str | Path,
    vector_bundle_directory: str | Path,
    questions: Sequence[Question],
    prompts: Mapping[str, PromptSpec],
    scope_selection: VerifiedE3StageSelection,
    protocol: E3Protocol | None = None,
) -> Mapping[str, Any]:
    frozen_protocol = protocol or E3Protocol()
    verification = verify_e3_shuffled_control_work(
        work_directory,
        construction_directory=construction_directory,
        vector_bundle_directory=vector_bundle_directory,
        questions=questions,
        prompts=prompts,
        scope_selection=scope_selection,
        protocol=frozen_protocol,
        require_complete=True,
    )
    work, plan, _snapshot, _points_value, _rows, shuffled = _context(
        work_directory,
        construction_directory=construction_directory,
        vector_bundle_directory=vector_bundle_directory,
        questions=questions,
        prompts=prompts,
        scope_selection=scope_selection,
        protocol=frozen_protocol,
    )
    accumulator, checkpoint_head, _count = _load_checkpoints(
        work / "checkpoints",
        plan_identity=plan["plan_identity"],
        width=plan["hidden_width"],
        shuffled=shuffled,
    )
    source = Path(directory)
    if (
        source.is_symlink()
        or not source.is_dir()
        or {value.name for value in source.iterdir()} != {"metadata.json", "directions.npz"}
        or any(value.is_symlink() or not value.is_file() for value in source.iterdir())
    ):
        raise FrozenArtifactError("E3 shuffled bundle inventory differs")
    try:
        metadata = _loads_json((source / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read E3 shuffled metadata: {exc}") from exc
    body = dict(metadata)
    digest = body.pop("metadata_digest", None)
    expected_body = {
        "schema_version": 1,
        "phase": "E3-shuffled-control",
        "plan_identity": plan["plan_identity"],
        "extraction_axis": list(_EXTRACTIONS),
        "hidden_width": plan["hidden_width"],
        "processed_rows": accumulator.processed_rows,
        "label_permutation": plan["label_permutation"],
        "scope_selection_digest": plan["scope_selection_digest"],
        "vector_data_fingerprint": plan["vector_data_fingerprint"],
        "checkpoint_chain_head": checkpoint_head,
        "directions_sha256": sha256_file(source / "directions.npz"),
        "scientific_eligible": verification["scientific_eligible"],
    }
    if (
        digest != stable_hash(body)
        or stable_hash(body) != stable_hash(expected_body)
    ):
        raise FrozenArtifactError("E3 shuffled metadata differs")
    try:
        with np.load(source / "directions.npz", allow_pickle=False) as value:
            if set(value.files) != {"directions", "correct_counts", "incorrect_counts"}:
                raise DataValidationError("array inventory differs")
            directions = value["directions"]
            correct = value["correct_counts"]
            incorrect = value["incorrect_counts"]
    except (OSError, ValueError, DataValidationError) as exc:
        raise FrozenArtifactError(f"cannot read E3 shuffled arrays: {exc}") from exc
    means = accumulator.sums / accumulator.counts[..., None]
    differences = means[:, 0] - means[:, 1]
    expected = (differences / np.linalg.norm(differences, axis=1)[:, None]).astype(np.float32)
    if (
        directions.dtype != np.float32
        or directions.shape != (2, plan["hidden_width"])
        or not np.array_equal(directions, expected)
        or not np.array_equal(correct, accumulator.counts[:, 0])
        or not np.array_equal(incorrect, accumulator.counts[:, 1])
        or correct.dtype != np.int64
        or incorrect.dtype != np.int64
    ):
        raise FrozenArtifactError("E3 shuffled vectors differ from checkpoint")
    return MappingProxyType(
        {
            "valid": True,
            "plan_identity": plan["plan_identity"],
            "directions_sha256": body["directions_sha256"],
            "processed_rows": accumulator.processed_rows,
            "scientific_eligible": body["scientific_eligible"],
        }
    )

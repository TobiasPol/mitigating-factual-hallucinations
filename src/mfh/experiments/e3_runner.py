"""Resumable hash-chained staged E3 evaluation on the native MLX executor."""

from __future__ import annotations

import fcntl
import json
import math
import os
import shutil
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Any

from mfh.contracts import Outcome, PromptSpec, Question
from mfh.errors import ConfigurationError, DataValidationError, FrozenArtifactError
from mfh.experiments.e3_execution import (
    E3ExecutionAssets,
    E3ExecutionResult,
    E3ExecutionRuntime,
    execute_e3_condition,
)
from mfh.experiments.e3_schedule import (
    E3Condition,
    E3Protocol,
    e3_alpha_conditions,
    e3_control_conditions,
    e3_cross_prompt_conditions,
    e3_final_conditions,
    e3_geometry_conditions,
    e3_p3_conditions,
    e3_scope_conditions,
)
from mfh.experiments.e3_selection import VerifiedE3StageSelection
from mfh.experiments.model_selection import validate_active_study_artifact_paths
from mfh.provenance import sha256_file, stable_hash

_INVENTORY = frozenset({"plan.json", "records.jsonl", "sessions.jsonl"})
_SCREEN_STAGES = frozenset(
    {"geometry", "alpha", "scope", "controls", "cross-prompt", "P3-diagnostic"}
)
_ALL_STAGES = (*sorted(_SCREEN_STAGES), "final")
_UNIFIED_MEMORY_BYTES = 51_539_607_552
_SCHEDULE_SEED = 17


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _loads(value: str) -> Any:
    return json.loads(value, object_pairs_hook=_reject_duplicate_keys)


def e3_conditions_for_stage(
    stage: str,
    *,
    selection_receipt: VerifiedE3StageSelection | None = None,
    protocol: E3Protocol | None = None,
) -> tuple[E3Condition, ...]:
    """Return the exact frozen grid for one stage and its verified predecessor."""

    frozen_protocol = protocol or E3Protocol()
    if stage == "geometry":
        if selection_receipt is not None:
            raise DataValidationError("E3 geometry cannot consume a predecessor selection")
        return e3_geometry_conditions(frozen_protocol)
    if selection_receipt is None:
        raise DataValidationError(f"E3 {stage} requires a verified predecessor selection")
    selection_receipt.assert_current()
    if selection_receipt.falsified:
        raise DataValidationError("E3 cannot continue after frozen selection falsification")
    expected_predecessor = (
        "geometry" if stage == "alpha" else "alpha" if stage == "scope" else "scope"
    )
    if selection_receipt.stage != expected_predecessor:
        raise DataValidationError(f"E3 {stage} predecessor stage differs")
    points = selection_receipt.selected
    builders = {
        "alpha": e3_alpha_conditions,
        "scope": e3_scope_conditions,
        "controls": e3_control_conditions,
        "cross-prompt": e3_cross_prompt_conditions,
        "P3-diagnostic": e3_p3_conditions,
        "final": e3_final_conditions,
    }
    try:
        builder = builders[stage]
    except KeyError as exc:
        raise DataValidationError("E3 evaluation stage is invalid") from exc
    return builder(points, protocol=frozen_protocol)


def _evaluation_question_ids(assets: E3ExecutionAssets) -> tuple[str, ...]:
    return tuple(assets.question_fingerprints)


def _plan_body(
    *,
    stage: str,
    assets: E3ExecutionAssets,
    selection_receipt: VerifiedE3StageSelection | None,
    runtime_identity: Mapping[str, Any],
    max_new_tokens: int,
) -> dict[str, Any]:
    assets.assert_current()
    if type(max_new_tokens) is not int or not 0 < max_new_tokens <= 48:
        raise DataValidationError("E3 evaluation max_new_tokens must be in [1, 48]")
    if stage not in _ALL_STAGES:
        raise DataValidationError("E3 evaluation stage is invalid")
    conditions = e3_conditions_for_stage(
        stage, selection_receipt=selection_receipt, protocol=assets.protocol
    )
    if tuple(assets.conditions.values()) != conditions:
        raise DataValidationError("E3 execution assets differ from the exact stage grid")
    expected_questions = (
        assets.protocol.dev_rows if stage == "final" else assets.protocol.screen_rows
    )
    question_ids = _evaluation_question_ids(assets)
    if len(question_ids) != expected_questions:
        raise DataValidationError("E3 evaluation question count differs from stage")
    try:
        identity = json.loads(json.dumps(dict(runtime_identity), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"E3 runtime identity is not stable JSON: {exc}") from exc
    if identity != assets.artifact_identity.get("construction_runtime_identity"):
        raise DataValidationError("E3 evaluation runtime differs from construction runtime")
    body = {
        "schema_version": 1,
        "phase": "E3-evaluation",
        "stage": stage,
        "runner_source_sha256": sha256_file(Path(__file__)),
        "protocol": assets.protocol.to_dict(),
        "conditions": [value.to_dict() for value in conditions],
        "condition_ids": [value.condition_id for value in conditions],
        "question_ids": list(question_ids),
        "question_fingerprints": dict(assets.question_fingerprints),
        "prompt_fingerprints": dict(assets.prompt_fingerprints),
        "artifact_identity": dict(assets.artifact_identity),
        "selection_digest": (
            selection_receipt.selection_digest if selection_receipt is not None else None
        ),
        "selection_artifact_sha256": (
            selection_receipt.artifact_sha256 if selection_receipt is not None else None
        ),
        "runtime_identity": identity,
        "max_new_tokens": max_new_tokens,
        "schedule": {
            "ordering": "sha256-rank-randomized-across-conditions-v1",
            "seed": _SCHEDULE_SEED,
            "schedule_digest": stable_hash(
                [
                    [condition.condition_id, question_id]
                    for condition, question_id in _ordered_rows(conditions, question_ids)
                ]
            ),
        },
        "expected_records": len(conditions) * len(question_ids),
        "scientific_eligible": bool(
            assets.scientific_eligible
            and (selection_receipt is None or selection_receipt.scientific_eligible)
            and assets.protocol.scientific_eligible
        ),
    }
    return body


def _complete_plan(body: Mapping[str, Any]) -> dict[str, Any]:
    return {**body, "plan_identity": stable_hash(dict(body))}


def prepare_e3_evaluation_work(
    directory: str | Path,
    *,
    stage: str,
    assets: E3ExecutionAssets,
    runtime_identity: Mapping[str, Any],
    selection_receipt: VerifiedE3StageSelection | None = None,
    max_new_tokens: int = 48,
) -> Mapping[str, Any]:
    directory = validate_active_study_artifact_paths({"E3 evaluation work": directory})[
        "E3 evaluation work"
    ]
    plan = _complete_plan(
        _plan_body(
            stage=stage,
            assets=assets,
            selection_receipt=selection_receipt,
            runtime_identity=runtime_identity,
            max_new_tokens=max_new_tokens,
        )
    )
    output = Path(directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite E3 evaluation work: {output}")
    stage_path = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        (stage_path / "plan.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (stage_path / "records.jsonl").touch()
        (stage_path / "sessions.jsonl").touch()
        os.replace(stage_path, output)
    finally:
        if stage_path.exists():
            shutil.rmtree(stage_path)
    return MappingProxyType(plan)


def _context(
    directory: str | Path,
    *,
    stage: str,
    assets: E3ExecutionAssets,
    selection_receipt: VerifiedE3StageSelection | None,
) -> tuple[Path, dict[str, Any], tuple[E3Condition, ...], tuple[str, ...]]:
    work = Path(directory)
    if (
        work.is_symlink()
        or not work.is_dir()
        or {value.name for value in work.iterdir()} != _INVENTORY
        or any(value.is_symlink() or not value.is_file() for value in work.iterdir())
    ):
        raise FrozenArtifactError("E3 evaluation work inventory differs")
    try:
        text = (work / "plan.json").read_text(encoding="utf-8")
        plan = _loads(text)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read E3 evaluation plan: {exc}") from exc
    if type(plan) is not dict:
        raise FrozenArtifactError("E3 evaluation plan schema differs")
    expected = _complete_plan(
        _plan_body(
            stage=stage,
            assets=assets,
            selection_receipt=selection_receipt,
            runtime_identity=plan.get("runtime_identity", {}),
            max_new_tokens=plan.get("max_new_tokens", 0),
        )
    )
    expected_text = json.dumps(expected, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if text != expected_text:
        raise FrozenArtifactError("E3 evaluation plan differs from live sources")
    return (
        work,
        plan,
        tuple(assets.conditions.values()),
        _evaluation_question_ids(assets),
    )


def _repair_tail(path: Path) -> None:
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    boundary = data.rfind(b"\n")
    tail = data[boundary + 1 :]
    try:
        parsed = _loads(tail.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
        repaired = data[: boundary + 1] if boundary >= 0 else b""
    else:
        repaired = data + b"\n" if type(parsed) is dict else data[: boundary + 1]
    with path.open("wb") as handle:
        handle.write(repaired)
        handle.flush()
        os.fsync(handle.fileno())


def _ordered_rows(
    conditions: Sequence[E3Condition], question_ids: Sequence[str]
) -> tuple[tuple[E3Condition, str], ...]:
    rows = tuple(
        (condition, question_id) for condition in conditions for question_id in question_ids
    )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                stable_hash(
                    {
                        "schema_version": 1,
                        "seed": _SCHEDULE_SEED,
                        "condition_id": row[0].condition_id,
                        "question_id": row[1],
                    }
                ),
                row[0].condition_id,
                row[1],
            ),
        )
    )


def _read_records(
    path: Path,
    *,
    plan_identity: str,
    assets: E3ExecutionAssets,
    questions: Mapping[str, Question],
    ordered: Sequence[tuple[E3Condition, str]],
) -> tuple[tuple[Mapping[str, Any], ...], str | None]:
    rows: list[Mapping[str, Any]] = []
    previous: str | None = None
    try:
        for sequence, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            value = _loads(line)
            if type(value) is not dict:
                raise DataValidationError("E3 evaluation record row is invalid")
            body = dict(value)
            digest = body.pop("record_digest", None)
            if (
                set(body)
                != {
                    "schema_version",
                    "sequence",
                    "plan_identity",
                    "cumulative_wall_time_seconds",
                    "result",
                    "previous_record_digest",
                }
                or digest != stable_hash(body)
                or body["schema_version"] != 1
                or type(body["sequence"]) is not int
                or body["sequence"] != sequence
                or body["plan_identity"] != plan_identity
                or body["previous_record_digest"] != previous
                or type(body["cumulative_wall_time_seconds"]) is not float
                or not math.isfinite(body["cumulative_wall_time_seconds"])
                or body["cumulative_wall_time_seconds"] < 0
                or (
                    rows
                    and body["cumulative_wall_time_seconds"]
                    < rows[-1]["cumulative_wall_time_seconds"]
                )
                or sequence >= len(ordered)
            ):
                raise DataValidationError("E3 evaluation record chain differs")
            condition, question_id = ordered[sequence]
            question = questions[question_id]
            resolved = assets.resolve(condition, question_id=question_id)
            result = E3ExecutionResult.from_dict(
                body["result"],
                question=question,
                condition=condition,
                resolved=resolved,
                expected_rendered_prompt_sha256=assets.rendered_prompt_hashes[
                    f"{condition.condition_id}:{question_id}"
                ][0],
                expected_prompt_token_ids_sha256=assets.rendered_prompt_hashes[
                    f"{condition.condition_id}:{question_id}"
                ][1],
            )
            rows.append(
                MappingProxyType(
                    {
                        **body,
                        "result": result,
                        "record_digest": digest,
                    }
                )
            )
            previous = digest
    except (OSError, ValueError, TypeError, json.JSONDecodeError, DataValidationError) as exc:
        raise FrozenArtifactError(f"cannot read E3 evaluation records: {exc}") from exc
    return tuple(rows), previous


def _read_sessions(
    path: Path,
    *,
    plan_identity: str,
    records_completed: int,
    records_expected: int,
) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    previous: str | None = None
    active_start: Mapping[str, Any] | None = None
    next_session_index = 0
    last_closed_records = 0
    last_created_ns = 0
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = _loads(line)
            if type(row) is not dict:
                raise DataValidationError("E3 evaluation session row is invalid")
            body = dict(row)
            digest = body.pop("session_digest", None)
            event = body.get("event")
            expected_fields = {
                "schema_version",
                "event",
                "session_index",
                "plan_identity",
                "records_completed",
                "previous_session_digest",
                "created_unix_ns",
            }
            if event == "end":
                expected_fields.add("status")
            if (
                set(body) != expected_fields
                or digest != stable_hash(body)
                or body.get("previous_session_digest") != previous
                or body.get("plan_identity") != plan_identity
                or body.get("schema_version") != 1
                or event not in {"start", "end"}
                or type(body.get("session_index")) is not int
                or type(body.get("records_completed")) is not int
                or not 0 <= body["records_completed"] <= records_completed
                or type(body.get("created_unix_ns")) is not int
                or body["created_unix_ns"] <= 0
                or body["created_unix_ns"] < last_created_ns
            ):
                raise DataValidationError("E3 evaluation session chain differs")
            if event == "start":
                if (
                    active_start is not None
                    or body["session_index"] != next_session_index
                    or body["records_completed"] != last_closed_records
                ):
                    raise DataValidationError("E3 evaluation session state differs")
                active_start = body
                next_session_index += 1
            else:
                if (
                    active_start is None
                    or body["session_index"] != active_start["session_index"]
                    or body["records_completed"] < active_start["records_completed"]
                    or body.get("status")
                    not in {"partial", "complete", "error", "interrupted-recovered"}
                    or (
                        body["status"] == "complete"
                        and body["records_completed"] != records_expected
                    )
                    or (
                        body["status"] == "partial"
                        and (
                            body["records_completed"] >= records_expected
                            or body["records_completed"] <= active_start["records_completed"]
                        )
                    )
                ):
                    raise DataValidationError("E3 evaluation session state differs")
                last_closed_records = body["records_completed"]
                active_start = None
            rows.append(MappingProxyType({**body, "session_digest": digest}))
            previous = digest
            last_created_ns = body["created_unix_ns"]
        if (not rows and records_completed != 0) or (
            active_start is None and last_closed_records != records_completed
        ):
            raise DataValidationError("E3 evaluation session records differ")
    except (OSError, ValueError, TypeError, json.JSONDecodeError, DataValidationError) as exc:
        raise FrozenArtifactError(f"cannot read E3 evaluation sessions: {exc}") from exc
    return tuple(rows)


def _append_chained(path: Path, body: Mapping[str, Any], *, prefix: str) -> str:
    digest = stable_hash(dict(body))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {**body, f"{prefix}_digest": digest},
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    return digest


@contextmanager
def _lock(path: Path) -> Iterator[None]:
    with path.open("rb") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ConfigurationError("E3 evaluation work is already running") from exc
        yield


def _load_e3_evaluation_snapshot_unlocked(
    directory: str | Path,
    *,
    stage: str,
    assets: E3ExecutionAssets,
    evaluation_questions: Sequence[Question],
    selection_receipt: VerifiedE3StageSelection | None = None,
    require_complete: bool = False,
) -> tuple[Mapping[str, Any], tuple[E3ExecutionResult, ...]]:
    work, plan, conditions, question_ids = _context(
        directory,
        stage=stage,
        assets=assets,
        selection_receipt=selection_receipt,
    )
    frozen_questions = tuple(evaluation_questions)
    questions = {value.question_id: value for value in frozen_questions}
    if tuple(value.question_id for value in frozen_questions) != question_ids or any(
        assets.question_fingerprints.get(value.question_id)
        != stable_hash(
            {
                "question_id": value.question_id,
                "benchmark": value.benchmark,
                "text": value.text,
                "aliases": list(value.aliases),
                "split": value.split,
                "entities": list(value.entities),
                "metadata": dict(value.metadata),
            }
        )
        for value in questions.values()
    ):
        raise FrozenArtifactError("E3 evaluation questions differ from assets")
    ordered = _ordered_rows(conditions, question_ids)
    records, record_head = _read_records(
        work / "records.jsonl",
        plan_identity=plan["plan_identity"],
        assets=assets,
        questions=questions,
        ordered=ordered,
    )
    sessions = _read_sessions(
        work / "sessions.jsonl",
        plan_identity=plan["plan_identity"],
        records_completed=len(records),
        records_expected=len(ordered),
    )
    open_sessions = sum(value["event"] == "start" for value in sessions) - sum(
        value["event"] == "end" for value in sessions
    )
    if open_sessions not in {0, 1}:
        raise FrozenArtifactError("E3 evaluation session nesting differs")
    complete = len(records) == len(ordered) and open_sessions == 0
    if require_complete and not complete:
        raise FrozenArtifactError("E3 evaluation work is incomplete or active")
    peak = max((record["result"].peak_memory_bytes for record in records), default=0)
    wall = float(records[-1]["cumulative_wall_time_seconds"]) if records else 0.0
    verification = MappingProxyType(
        {
            "valid": True,
            "complete": complete,
            "records_completed": len(records),
            "records_expected": len(ordered),
            "plan_identity": plan["plan_identity"],
            "record_chain_head": record_head,
            "record_set_digest": stable_hash([record["record_digest"] for record in records]),
            "session_chain_head": (sessions[-1]["session_digest"] if sessions else None),
            "session_set_digest": stable_hash([session["session_digest"] for session in sessions]),
            "maximum_peak_memory_bytes": peak,
            "wall_time_seconds": wall,
            "memory_within_envelope": peak <= _UNIFIED_MEMORY_BYTES,
            "scientific_eligible": bool(
                plan["scientific_eligible"] and peak <= _UNIFIED_MEMORY_BYTES and complete
            ),
        }
    )
    return verification, tuple(record["result"] for record in records)


def load_e3_evaluation_snapshot(
    directory: str | Path,
    *,
    stage: str,
    assets: E3ExecutionAssets,
    evaluation_questions: Sequence[Question],
    selection_receipt: VerifiedE3StageSelection | None = None,
    require_complete: bool = True,
) -> tuple[Mapping[str, Any], tuple[E3ExecutionResult, ...]]:
    """Load one lock-protected, contextually verified stage snapshot."""

    with _lock(Path(directory) / "plan.json"):
        return _load_e3_evaluation_snapshot_unlocked(
            directory,
            stage=stage,
            assets=assets,
            evaluation_questions=evaluation_questions,
            selection_receipt=selection_receipt,
            require_complete=require_complete,
        )


def verify_e3_evaluation_work(
    directory: str | Path,
    *,
    stage: str,
    assets: E3ExecutionAssets,
    evaluation_questions: Sequence[Question],
    selection_receipt: VerifiedE3StageSelection | None = None,
    require_complete: bool = False,
) -> Mapping[str, Any]:
    return load_e3_evaluation_snapshot(
        directory,
        stage=stage,
        assets=assets,
        evaluation_questions=evaluation_questions,
        selection_receipt=selection_receipt,
        require_complete=require_complete,
    )[0]


def run_e3_evaluation(
    directory: str | Path,
    *,
    stage: str,
    assets: E3ExecutionAssets,
    evaluation_questions: Sequence[Question],
    application_prompts: Mapping[str, PromptSpec],
    runtime: E3ExecutionRuntime,
    selection_receipt: VerifiedE3StageSelection | None = None,
    request_budget: int | None = None,
) -> Mapping[str, Any]:
    directory = validate_active_study_artifact_paths({"E3 evaluation work": directory})[
        "E3 evaluation work"
    ]
    if request_budget is not None and (type(request_budget) is not int or request_budget <= 0):
        raise ConfigurationError("E3 evaluation request budget must be positive")
    work, plan, conditions, question_ids = _context(
        directory,
        stage=stage,
        assets=assets,
        selection_receipt=selection_receipt,
    )
    live_identity = runtime.runtime_identity()
    if json.loads(json.dumps(dict(live_identity), sort_keys=True)) != plan["runtime_identity"]:
        raise FrozenArtifactError("E3 live runtime identity differs from plan")
    questions = {value.question_id: value for value in evaluation_questions}
    if tuple(value.question_id for value in evaluation_questions) != question_ids:
        raise FrozenArtifactError("E3 evaluation questions differ from exact stage order")
    ordered = _ordered_rows(conditions, question_ids)
    with _lock(work / "plan.json"):
        _repair_tail(work / "records.jsonl")
        _repair_tail(work / "sessions.jsonl")
        records, record_head = _read_records(
            work / "records.jsonl",
            plan_identity=plan["plan_identity"],
            assets=assets,
            questions=questions,
            ordered=ordered,
        )
        sessions = _read_sessions(
            work / "sessions.jsonl",
            plan_identity=plan["plan_identity"],
            records_completed=len(records),
            records_expected=len(ordered),
        )
        session_head = sessions[-1]["session_digest"] if sessions else None
        session_index = sum(value["event"] == "start" for value in sessions)
        if sessions and sessions[-1]["event"] == "start":
            session_head = _append_chained(
                work / "sessions.jsonl",
                {
                    "schema_version": 1,
                    "event": "end",
                    "session_index": sessions[-1]["session_index"],
                    "plan_identity": plan["plan_identity"],
                    "status": "interrupted-recovered",
                    "records_completed": len(records),
                    "previous_session_digest": session_head,
                    "created_unix_ns": time.time_ns(),
                },
                prefix="session",
            )
        session_head = _append_chained(
            work / "sessions.jsonl",
            {
                "schema_version": 1,
                "event": "start",
                "session_index": session_index,
                "plan_identity": plan["plan_identity"],
                "records_completed": len(records),
                "previous_session_digest": session_head,
                "created_unix_ns": time.time_ns(),
            },
            prefix="session",
        )
        handled = 0
        status = "partial"
        starting_wall = float(records[-1]["cumulative_wall_time_seconds"]) if records else 0.0
        started_ns = time.monotonic_ns()
        try:
            while len(records) < len(ordered) and (
                request_budget is None or handled < request_budget
            ):
                sequence = len(records)
                condition, question_id = ordered[sequence]
                result = execute_e3_condition(
                    runtime=runtime,
                    assets=assets,
                    condition=condition,
                    question=questions[question_id],
                    prompts=application_prompts,
                    max_new_tokens=plan["max_new_tokens"],
                )
                body = {
                    "schema_version": 1,
                    "sequence": sequence,
                    "plan_identity": plan["plan_identity"],
                    "cumulative_wall_time_seconds": starting_wall
                    + (time.monotonic_ns() - started_ns) / 1e9,
                    "result": result.to_dict(),
                    "previous_record_digest": record_head,
                }
                record_head = _append_chained(work / "records.jsonl", body, prefix="record")
                records = (
                    *records,
                    MappingProxyType({**body, "result": result, "record_digest": record_head}),
                )
                handled += 1
            status = "complete" if len(records) == len(ordered) else "partial"
        except BaseException:
            status = "error"
            raise
        finally:
            _append_chained(
                work / "sessions.jsonl",
                {
                    "schema_version": 1,
                    "event": "end",
                    "session_index": session_index,
                    "plan_identity": plan["plan_identity"],
                    "status": status,
                    "records_completed": len(records),
                    "previous_session_digest": session_head,
                    "created_unix_ns": time.time_ns(),
                },
                prefix="session",
            )
    return verify_e3_evaluation_work(
        work,
        stage=stage,
        assets=assets,
        evaluation_questions=evaluation_questions,
        selection_receipt=selection_receipt,
    )


def e3_selection_inputs_from_work(
    directory: str | Path,
    *,
    stage: str,
    assets: E3ExecutionAssets,
    evaluation_questions: Sequence[Question],
    selection_receipt: VerifiedE3StageSelection | None = None,
) -> Mapping[str, Any]:
    if stage not in {"geometry", "alpha", "scope"}:
        raise DataValidationError("only E3 sweep stages produce selection inputs")
    verification, results = load_e3_evaluation_snapshot(
        directory,
        stage=stage,
        assets=assets,
        evaluation_questions=evaluation_questions,
        selection_receipt=selection_receipt,
        require_complete=True,
    )
    conditions = tuple(assets.conditions.values())
    question_ids = tuple(assets.question_fingerprints)
    outcomes: dict[tuple[str, str], Outcome] = {}
    norms: dict[tuple[str, str], float] = {}
    for result in results:
        key = (result.condition_id, result.question_id)
        outcomes[key] = result.outcome
        condition = assets.conditions[result.condition_id]
        if condition.method != "M0":
            norms[key] = result.actual_delta_norm
    return MappingProxyType(
        {
            "stage": stage,
            "conditions": conditions,
            "question_ids": question_ids,
            "outcomes": MappingProxyType(outcomes),
            "actual_delta_norms": MappingProxyType(norms),
            "source_plan_identity": assets.artifact_identity["construction_plan_identity"],
            "evaluation_plan_identity": verification["plan_identity"],
            "evaluation_record_chain_head": verification["record_chain_head"],
            "evaluation_record_set_digest": verification["record_set_digest"],
            "source_scientific_eligible": verification["scientific_eligible"],
            "predecessor_selection": (
                selection_receipt.selection if selection_receipt is not None else None
            ),
            "protocol": assets.protocol,
        }
    )


def load_e3_evaluation_results(
    directory: str | Path,
    *,
    stage: str,
    assets: E3ExecutionAssets,
    evaluation_questions: Sequence[Question],
    selection_receipt: VerifiedE3StageSelection | None = None,
) -> tuple[E3ExecutionResult, ...]:
    """Load every contextually replayed result from a complete staged run."""

    _verification, results = load_e3_evaluation_snapshot(
        directory,
        stage=stage,
        assets=assets,
        evaluation_questions=evaluation_questions,
        selection_receipt=selection_receipt,
        require_complete=True,
    )
    return results

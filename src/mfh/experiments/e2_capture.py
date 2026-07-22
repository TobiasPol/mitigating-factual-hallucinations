"""Resumable E2 generation resolution and native-VLLM prompt capture."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from mfh.contracts import ActivationSite, Outcome, PromptSpec, Question
from mfh.errors import ConfigurationError, DataValidationError, FrozenArtifactError
from mfh.evaluation.grading import deterministic_short_answer_grade
from mfh.experiments.activation_store import (
    ActivationCaptureRow,
    append_activation_shard,
    iter_activation_shards,
    verify_activation_store,
)
from mfh.experiments.e2_schedule import E2ScheduleRow, VerifiedE2Workspace
from mfh.experiments.model_selection import validate_active_study_artifact_paths
from mfh.inference.vllm_research import VllmPromptFeatureCubeOutput
from mfh.inference.vllm_runtime import VllmGenerationOutput, VllmRenderedPrompt
from mfh.provenance import sha256_file, stable_hash

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_WORK_FILES = frozenset({"plan.json", "resolutions.jsonl", "sessions.jsonl"})


class E2CaptureRuntime(Protocol):
    def render_prompt(
        self,
        prompt: PromptSpec,
        question: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> VllmRenderedPrompt: ...

    def generate(
        self, rendered: VllmRenderedPrompt, *, max_new_tokens: int
    ) -> VllmGenerationOutput: ...

    def prompt_feature_cube(
        self,
        rendered: VllmRenderedPrompt,
        *,
        layers: Sequence[int],
        sites: Sequence[ActivationSite],
    ) -> VllmPromptFeatureCubeOutput: ...

    def runtime_identity(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class E1P0Source:
    benchmark: str
    question_id: str
    outcome: Outcome
    generation_record_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.benchmark) is not str
            or not self.benchmark.strip()
            or type(self.question_id) is not str
            or not self.question_id.strip()
            or type(self.generation_record_sha256) is not str
            or not _SHA256.fullmatch(self.generation_record_sha256)
        ):
            raise DataValidationError("E1 P0 source identity is invalid")
        object.__setattr__(self, "benchmark", self.benchmark.strip())
        object.__setattr__(self, "question_id", self.question_id.strip())
        object.__setattr__(self, "outcome", Outcome(self.outcome))

    def to_dict(self) -> dict[str, str]:
        return {
            "benchmark": self.benchmark,
            "question_id": self.question_id,
            "outcome": self.outcome.value,
            "generation_record_sha256": self.generation_record_sha256,
        }


@dataclass(frozen=True, slots=True)
class E2Resolution:
    sequence: int
    plan_identity: str
    schedule_row_sha256: str
    question_id: str
    benchmark: str
    prompt_id: str
    rendered_prompt_sha256: str
    prompt_token_ids_sha256: str
    label_source: str
    outcome: Outcome
    generation_record_sha256: str
    generation_evidence: Mapping[str, Any] | None

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence < 0:
            raise DataValidationError("E2 resolution sequence must be a non-negative integer")
        for name in (
            "plan_identity",
            "schedule_row_sha256",
            "rendered_prompt_sha256",
            "prompt_token_ids_sha256",
            "generation_record_sha256",
        ):
            value = getattr(self, name)
            if type(value) is not str or not _SHA256.fullmatch(value):
                raise DataValidationError(f"E2 resolution {name} must be a SHA-256")
        for name in ("question_id", "benchmark", "prompt_id"):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise DataValidationError(f"E2 resolution {name} must be non-empty text")
        if self.label_source not in {"E1", "generate"}:
            raise DataValidationError("E2 resolution label source is invalid")
        evidence = self.generation_evidence
        if (self.label_source == "generate") != (evidence is not None):
            raise DataValidationError("E2 generated resolutions require exact generation evidence")
        if evidence is not None:
            _validate_generation_evidence(evidence)
            if stable_hash(dict(evidence)) != self.generation_record_sha256:
                raise DataValidationError("E2 generation evidence digest differs")
            evidence = MappingProxyType(dict(evidence))
        object.__setattr__(self, "outcome", Outcome(self.outcome))
        object.__setattr__(self, "generation_evidence", evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "plan_identity": self.plan_identity,
            "schedule_row_sha256": self.schedule_row_sha256,
            "question_id": self.question_id,
            "benchmark": self.benchmark,
            "prompt_id": self.prompt_id,
            "rendered_prompt_sha256": self.rendered_prompt_sha256,
            "prompt_token_ids_sha256": self.prompt_token_ids_sha256,
            "label_source": self.label_source,
            "outcome": self.outcome.value,
            "generation_record_sha256": self.generation_record_sha256,
            "generation_evidence": (
                dict(self.generation_evidence)
                if self.generation_evidence is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> E2Resolution:
        expected = {
            "sequence",
            "plan_identity",
            "schedule_row_sha256",
            "question_id",
            "benchmark",
            "prompt_id",
            "rendered_prompt_sha256",
            "prompt_token_ids_sha256",
            "label_source",
            "outcome",
            "generation_record_sha256",
            "generation_evidence",
        }
        string_fields = expected - {"sequence", "generation_evidence"}
        evidence = value.get("generation_evidence")
        if (
            set(value) != expected
            or type(value.get("sequence")) is not int
            or any(type(value.get(name)) is not str for name in string_fields)
            or (evidence is not None and type(evidence) is not dict)
        ):
            raise DataValidationError("E2 resolution has invalid JSON schema or types")
        return cls(
            sequence=value["sequence"],
            plan_identity=value["plan_identity"],
            schedule_row_sha256=value["schedule_row_sha256"],
            question_id=value["question_id"],
            benchmark=value["benchmark"],
            prompt_id=value["prompt_id"],
            rendered_prompt_sha256=value["rendered_prompt_sha256"],
            prompt_token_ids_sha256=value["prompt_token_ids_sha256"],
            label_source=value["label_source"],
            outcome=Outcome(value["outcome"]),
            generation_record_sha256=value["generation_record_sha256"],
            generation_evidence=evidence,
        )


def _validate_generation_evidence(value: Mapping[str, Any]) -> None:
    expected = {
        "raw_output",
        "raw_output_sha256",
        "token_ids",
        "token_ids_sha256",
        "input_tokens",
        "output_tokens",
        "latency_seconds",
        "stop_type",
        "stopping_token_id",
        "prompt_tokens_per_second",
        "generation_tokens_per_second",
        "peak_memory_bytes",
        "active_memory_bytes",
        "cache_memory_bytes",
    }
    token_ids = value.get("token_ids")
    integers = (
        "input_tokens",
        "output_tokens",
        "peak_memory_bytes",
        "active_memory_bytes",
        "cache_memory_bytes",
    )
    numbers = (
        "latency_seconds",
        "prompt_tokens_per_second",
        "generation_tokens_per_second",
    )
    if (
        set(value) != expected
        or type(value.get("raw_output")) is not str
        or type(value.get("raw_output_sha256")) is not str
        or not _SHA256.fullmatch(value["raw_output_sha256"])
        or value["raw_output_sha256"]
        != hashlib.sha256(value["raw_output"].encode("utf-8")).hexdigest()
        or type(token_ids) is not list
        or not token_ids
        or any(type(token) is not int or token < 0 for token in token_ids)
        or type(value.get("token_ids_sha256")) is not str
        or value["token_ids_sha256"] != stable_hash(token_ids)
        or any(type(value.get(name)) is not int or value[name] < 0 for name in integers)
        or any(
            isinstance(value.get(name), bool)
            or not isinstance(value.get(name), int | float)
            or not math.isfinite(float(value[name]))
            or float(value[name]) < 0
            for name in numbers
        )
        or type(value.get("stop_type")) is not str
        or value["stop_type"] not in {"stop", "length", "short_answer"}
        or (
            value.get("stopping_token_id") is not None
            and (type(value["stopping_token_id"]) is not int or value["stopping_token_id"] < 0)
        )
        or value.get("output_tokens") != len(token_ids)
        or value.get("stopping_token_id") != token_ids[-1]
    ):
        raise DataValidationError("E2 generation evidence is invalid")


def _generation_evidence(generation: VllmGenerationOutput) -> dict[str, Any]:
    value = {
        "raw_output": generation.text,
        "raw_output_sha256": hashlib.sha256(generation.text.encode("utf-8")).hexdigest(),
        "token_ids": list(generation.token_ids),
        "token_ids_sha256": stable_hash(list(generation.token_ids)),
        "input_tokens": generation.input_tokens,
        "output_tokens": generation.output_tokens,
        "latency_seconds": generation.latency_seconds,
        "stop_type": generation.stop_type,
        "stopping_token_id": generation.stopping_token_id,
        "prompt_tokens_per_second": generation.prompt_tokens_per_second,
        "generation_tokens_per_second": generation.generation_tokens_per_second,
        "peak_memory_bytes": generation.peak_memory_bytes,
        "active_memory_bytes": generation.active_memory_bytes,
        "cache_memory_bytes": generation.cache_memory_bytes,
    }
    _validate_generation_evidence(value)
    return value


def _questions_digest(questions: Mapping[tuple[str, str], Question]) -> str:
    return stable_hash(
        [
            {
                "benchmark": benchmark,
                "question_id": question_id,
                "text": question.text,
                "aliases": list(question.aliases),
                "metadata": dict(question.metadata),
            }
            for (benchmark, question_id), question in sorted(questions.items())
        ]
    )


def _sources_digest(sources: Mapping[tuple[str, str], E1P0Source]) -> str:
    return stable_hash(
        [source.to_dict() for _key, source in sorted(sources.items())]
    )


def _prompts_digest(prompts: Mapping[str, PromptSpec]) -> str:
    return stable_hash(
        [
            {
                "mapping_key": name,
                "prompt_id": prompt.prompt_id,
                "text": prompt.text,
                "permits_abstention": prompt.permits_abstention,
                "deployment_eligible": prompt.deployment_eligible,
            }
            for name, prompt in sorted(prompts.items())
        ]
    )


def _capture_plan(
    workspace: VerifiedE2Workspace,
    questions: Mapping[tuple[str, str], Question],
    prompts: Mapping[str, PromptSpec],
    e1_sources: Mapping[tuple[str, str], E1P0Source],
    *,
    shard_rows: int,
    max_new_tokens: int,
    expected_runtime_identity: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "schema_version": 1,
        "phase": "E2",
        "runner": "resumable-native-vllm-prompt-capture",
        "runner_source_sha256": sha256_file(Path(__file__)),
        "workspace_plan_identity": workspace.plan_identity,
        "questions_digest": _questions_digest(questions),
        "prompts_digest": _prompts_digest(prompts),
        "e1_sources_digest": _sources_digest(e1_sources),
        "expected_rows": len(workspace.schedule),
        "shard_rows": shard_rows,
        "max_new_tokens": max_new_tokens,
        "runtime_identity": dict(expected_runtime_identity),
    }
    return {**body, "capture_plan_identity": stable_hash(body)}


def prepare_e2_capture_work(
    directory: str | Path,
    *,
    workspace: VerifiedE2Workspace,
    questions: Mapping[tuple[str, str], Question],
    prompts: Mapping[str, PromptSpec],
    e1_sources: Mapping[tuple[str, str], E1P0Source],
    expected_runtime_identity: Mapping[str, Any],
    shard_rows: int = 64,
    max_new_tokens: int = 48,
) -> Mapping[str, Any]:
    if (
        type(shard_rows) is not int
        or shard_rows <= 0
        or type(max_new_tokens) is not int
        or not 1 <= max_new_tokens <= 48
    ):
        raise ConfigurationError("E2 capture shard and generation sizes are invalid")
    _validate_capture_inputs(workspace, questions, prompts, e1_sources)
    runtime_identity = _validated_runtime_identity(expected_runtime_identity)
    destination = validate_active_study_artifact_paths(
        {"E2 capture work": directory, "E2 workspace": workspace.directory}
    )["E2 capture work"]
    if destination.exists():
        raise FrozenArtifactError(f"refusing to overwrite E2 capture work: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    plan = _capture_plan(
        workspace,
        questions,
        prompts,
        e1_sources,
        shard_rows=shard_rows,
        max_new_tokens=max_new_tokens,
        expected_runtime_identity=runtime_identity,
    )
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        (stage / "plan.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (stage / "resolutions.jsonl").touch()
        (stage / "sessions.jsonl").touch()
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return MappingProxyType(plan)


def _validate_capture_inputs(
    workspace: VerifiedE2Workspace,
    questions: Mapping[tuple[str, str], Question],
    prompts: Mapping[str, PromptSpec],
    e1_sources: Mapping[tuple[str, str], E1P0Source],
) -> None:
    if any(
        key != (question.benchmark, question.question_id)
        for key, question in questions.items()
    ):
        raise DataValidationError("E2 capture question mapping keys differ from their objects")
    if any(key != prompt.prompt_id for key, prompt in prompts.items()):
        raise DataValidationError("E2 capture prompt mapping keys differ from their objects")
    if any(
        key != (source.benchmark, source.question_id)
        for key, source in e1_sources.items()
    ):
        raise DataValidationError("E2 capture E1 source keys differ from their objects")
    required_questions = {(row.benchmark, row.question_id) for row in workspace.schedule}
    required_prompts = {row.prompt_id for row in workspace.schedule}
    if set(questions) != required_questions or not required_prompts <= set(prompts):
        raise DataValidationError("E2 capture questions or prompts differ from the schedule")
    for row in workspace.schedule:
        question = questions[(row.benchmark, row.question_id)]
        if stable_hash(
            {
                "question_id": question.question_id,
                "benchmark": question.benchmark,
                "text": question.text,
            }
        ) != row.question_sha256 or stable_hash(list(question.aliases)) != row.aliases_sha256:
            raise DataValidationError("E2 capture question identity differs from the schedule")
        source = e1_sources.get((row.benchmark, row.question_id))
        if row.label_source == "E1":
            if source is None or source.outcome is not row.outcome:
                raise DataValidationError("E2 capture lacks its exact E1 P0 source")
        elif source is not None and row.prompt_id != "P3-forced-answer":
            raise DataValidationError("E2 generated P0 row unexpectedly reuses an E1 source")


def _load_plan(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read E2 capture plan: {exc}") from exc
    expected = {
        "schema_version",
        "phase",
        "runner",
        "runner_source_sha256",
        "workspace_plan_identity",
        "questions_digest",
        "prompts_digest",
        "e1_sources_digest",
        "expected_rows",
        "shard_rows",
        "max_new_tokens",
        "capture_plan_identity",
        "runtime_identity",
    }
    if type(value) is not dict or set(value) != expected:
        raise FrozenArtifactError("E2 capture plan must be a mapping")
    body = dict(value)
    identity = body.pop("capture_plan_identity", None)
    if (
        identity != stable_hash(body)
        or body.get("runner_source_sha256") != sha256_file(Path(__file__))
        or any(
            type(body.get(name)) is not int
            for name in ("schema_version", "expected_rows", "shard_rows", "max_new_tokens")
        )
        or any(
            type(body.get(name)) is not str
            for name in (
                "phase",
                "runner",
                "runner_source_sha256",
                "workspace_plan_identity",
                "questions_digest",
                "prompts_digest",
                "e1_sources_digest",
            )
        )
        or type(body.get("runtime_identity")) is not dict
        or body.get("schema_version") != 1
        or body.get("phase") != "E2"
        or body.get("runner") != "resumable-native-vllm-prompt-capture"
        or body["shard_rows"] <= 0
        or not 1 <= body["max_new_tokens"] <= 48
    ):
        raise FrozenArtifactError("E2 capture plan identity or runner source differs")
    return value


def _validated_runtime_identity(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise DataValidationError("E2 capture runtime identity must be a non-empty mapping")
    normalized = dict(value)
    try:
        serialized = json.dumps(
            normalized,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        replayed = json.loads(serialized)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"E2 capture runtime identity is not exact JSON: {exc}") from exc
    if type(replayed) is not dict or replayed != normalized:
        raise DataValidationError("E2 capture runtime identity is not stable JSON")
    return MappingProxyType(replayed)


def _load_resolutions(
    path: Path,
    *,
    workspace: VerifiedE2Workspace,
) -> list[E2Resolution]:
    values: list[E2Resolution] = []
    previous: str | None = None
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                raw = json.loads(line)
                if type(raw) is not dict:
                    raise DataValidationError("E2 resolution envelope must be a mapping")
                body = dict(raw)
                digest = body.pop("resolution_digest", None)
                prior = body.get("previous_resolution_digest")
                if (
                    type(digest) is not str
                    or not _SHA256.fullmatch(digest)
                    or digest != stable_hash(body)
                    or prior != previous
                ):
                    raise DataValidationError("E2 resolution chain differs")
                body.pop("previous_resolution_digest")
                resolution = E2Resolution.from_dict(body)
                if resolution.sequence != len(values) or resolution.sequence >= len(
                    workspace.schedule
                ):
                    raise DataValidationError("E2 resolution sequence differs")
                row = workspace.schedule[resolution.sequence]
                if (
                    resolution.plan_identity != workspace.plan_identity
                    or resolution.schedule_row_sha256 != stable_hash(row.to_dict())
                    or resolution.question_id != row.question_id
                    or resolution.benchmark != row.benchmark
                    or resolution.prompt_id != row.prompt_id
                    or resolution.label_source != row.label_source
                    or (row.outcome is not None and resolution.outcome is not row.outcome)
                ):
                    raise DataValidationError("E2 resolution differs from its schedule row")
                values.append(resolution)
                previous = digest
    except (OSError, json.JSONDecodeError, TypeError, ValueError, DataValidationError) as exc:
        raise FrozenArtifactError(f"cannot read E2 resolutions: {exc}") from exc
    return values


def _append_resolution(path: Path, resolution: E2Resolution, previous: str | None) -> str:
    body = {**resolution.to_dict(), "previous_resolution_digest": previous}
    digest = stable_hash(body)
    envelope = {**body, "resolution_digest": digest}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(envelope, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return digest


def _resolution_chain_head(resolutions: Sequence[E2Resolution]) -> str | None:
    previous: str | None = None
    for resolution in resolutions:
        previous = stable_hash(
            {**resolution.to_dict(), "previous_resolution_digest": previous}
        )
    return previous


def _append_session(path: Path, body: Mapping[str, Any], previous: str | None) -> str:
    value = {**body, "previous_session_event_digest": previous}
    digest = stable_hash(value)
    envelope = {**value, "session_event_digest": digest}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(envelope, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return digest


def _load_sessions(
    path: Path,
    *,
    capture_plan_identity: str,
    expected_runtime_identity: Mapping[str, Any],
    allow_open: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous: str | None = None
    open_session: int | None = None
    last_rows = 0
    start_keys = {
        "schema_version",
        "event",
        "session_index",
        "capture_plan_identity",
        "rows_at_start",
        "runtime_identity",
        "created_unix_ns",
        "previous_session_event_digest",
        "session_event_digest",
    }
    end_keys = {
        "schema_version",
        "event",
        "session_index",
        "capture_plan_identity",
        "status",
        "rows_at_end",
        "activation_chain_head",
        "resolution_chain_head",
        "wall_time_seconds",
        "capture_peak_memory_bytes",
        "created_unix_ns",
        "previous_session_event_digest",
        "session_event_digest",
    }
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if type(row) is not dict:
                    raise DataValidationError("E2 session event must be a mapping")
                body = dict(row)
                digest = body.pop("session_event_digest", None)
                event = row.get("event")
                index = row.get("session_index")
                if (
                    type(digest) is not str
                    or not _SHA256.fullmatch(digest)
                    or digest != stable_hash(body)
                    or row.get("previous_session_event_digest") != previous
                    or row.get("schema_version") != 1
                    or row.get("capture_plan_identity") != capture_plan_identity
                    or type(index) is not int
                    or index < 0
                    or type(row.get("created_unix_ns")) is not int
                    or row["created_unix_ns"] <= 0
                ):
                    raise DataValidationError("E2 session event chain or identity differs")
                if event == "start":
                    if (
                        set(row) != start_keys
                        or open_session is not None
                        or index != len(rows) // 2
                        or type(row.get("rows_at_start")) is not int
                        or row["rows_at_start"] != last_rows
                        or row.get("runtime_identity") != dict(expected_runtime_identity)
                    ):
                        raise DataValidationError("E2 session start event differs")
                    open_session = index
                elif event == "end":
                    if (
                        set(row) != end_keys
                        or open_session != index
                        or row.get("status")
                        not in {"complete", "partial", "error", "interrupted-recovered"}
                        or type(row.get("rows_at_end")) is not int
                        or row["rows_at_end"] < last_rows
                        or isinstance(row.get("wall_time_seconds"), bool)
                        or not isinstance(row.get("wall_time_seconds"), int | float)
                        or not math.isfinite(float(row["wall_time_seconds"]))
                        or float(row["wall_time_seconds"]) < 0
                        or type(row.get("capture_peak_memory_bytes")) is not int
                        or row["capture_peak_memory_bytes"] < 0
                        or any(
                            value is not None
                            and (type(value) is not str or not _SHA256.fullmatch(value))
                            for value in (
                                row.get("activation_chain_head"),
                                row.get("resolution_chain_head"),
                            )
                        )
                    ):
                        raise DataValidationError("E2 session end event differs")
                    last_rows = row["rows_at_end"]
                    open_session = None
                else:
                    raise DataValidationError("E2 session event type is invalid")
                rows.append(row)
                previous = digest
    except (OSError, json.JSONDecodeError, TypeError, ValueError, DataValidationError) as exc:
        raise FrozenArtifactError(f"cannot read E2 capture sessions: {exc}") from exc
    if open_session is not None and not allow_open:
        raise FrozenArtifactError("E2 capture sessions contain an unclosed event")
    return rows


def _repair_torn_session_tail(path: Path) -> None:
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    boundary = data.rfind(b"\n")
    tail = data[boundary + 1 :]
    try:
        parsed = json.loads(tail.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        repaired = data[: boundary + 1] if boundary >= 0 else b""
    else:
        # A fully written event may lose only its trailing newline during a crash.
        # Preserve it and let the strict chained-session replay validate its schema
        # and digest; truncate only a genuinely incomplete JSON fragment.
        repaired = data + b"\n" if type(parsed) is dict else data[: boundary + 1]
    descriptor = os.open(path, os.O_WRONLY | os.O_TRUNC)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(repaired)
        handle.flush()
        os.fsync(handle.fileno())


def _make_resolution(
    *,
    row: E2ScheduleRow,
    workspace: VerifiedE2Workspace,
    question: Question,
    rendered: VllmRenderedPrompt,
    runtime: E2CaptureRuntime,
    source: E1P0Source | None,
    max_new_tokens: int,
) -> E2Resolution:
    evidence: Mapping[str, Any] | None = None
    if row.label_source == "E1":
        assert source is not None and row.outcome is not None
        outcome = row.outcome
        generation_digest = source.generation_record_sha256
    else:
        generation = runtime.generate(rendered, max_new_tokens=max_new_tokens)
        if generation.rendered_prompt != rendered:
            raise DataValidationError("E2 generation returned a different rendered prompt")
        evidence = _generation_evidence(generation)
        generation_digest = stable_hash(dict(evidence))
        outcome = deterministic_short_answer_grade(generation.text, question.aliases)
        if outcome is Outcome.UNSCORABLE:
            raise DataValidationError("E2 deterministic generation cannot be unscorable")
    return E2Resolution(
        sequence=row.sequence,
        plan_identity=workspace.plan_identity,
        schedule_row_sha256=stable_hash(row.to_dict()),
        question_id=row.question_id,
        benchmark=row.benchmark,
        prompt_id=row.prompt_id,
        rendered_prompt_sha256=rendered.sha256,
        prompt_token_ids_sha256=rendered.token_ids_sha256,
        label_source=row.label_source,
        outcome=outcome,
        generation_record_sha256=generation_digest,
        generation_evidence=evidence,
    )


def _capture_row(
    *,
    workspace: VerifiedE2Workspace,
    schedule_row: E2ScheduleRow,
    resolution: E2Resolution,
    rendered: VllmRenderedPrompt,
    runtime: E2CaptureRuntime,
) -> tuple[ActivationCaptureRow, np.ndarray[Any, Any], int]:
    output = runtime.prompt_feature_cube(
        rendered,
        layers=workspace.activation_spec.layers,
        sites=workspace.activation_spec.sites,
    )
    values = np.stack(
        [
            np.stack(
                [
                    output.activations[site][layer][0]
                    for layer in workspace.activation_spec.layers
                ],
                axis=0,
            )
            for site in workspace.activation_spec.sites
        ],
        axis=0,
    )
    if values.shape != (
        len(workspace.activation_spec.sites),
        len(workspace.activation_spec.layers),
        workspace.activation_spec.hidden_width,
    ):
        raise DataValidationError("E2 capture runtime returned the wrong activation geometry")
    return (
        ActivationCaptureRow(
            question_id=schedule_row.question_id,
            benchmark=schedule_row.benchmark,
            partition=schedule_row.feature_partition,
            prompt_id=schedule_row.prompt_id,
            outcome=resolution.outcome,
            semantic_group_id=schedule_row.semantic_group_id,
            rendered_prompt_sha256=resolution.rendered_prompt_sha256,
            prompt_token_ids_sha256=resolution.prompt_token_ids_sha256,
            generation_record_sha256=resolution.generation_record_sha256,
            maximum_token_probability=output.maximum_token_probability,
            output_entropy=output.output_entropy,
        ),
        values,
        output.peak_memory_bytes,
    )


def _verify_capture_prefix(
    workspace: VerifiedE2Workspace,
    resolutions: Sequence[E2Resolution],
) -> int:
    completed = 0
    for rows, _values in iter_activation_shards(
        workspace.directory / "activations", expected_spec=workspace.activation_spec
    ):
        for row in rows:
            if completed >= len(resolutions):
                raise FrozenArtifactError("E2 activation rows exceed frozen resolutions")
            schedule_row = workspace.schedule[completed]
            resolution = resolutions[completed]
            if (
                row.question_id != schedule_row.question_id
                or row.benchmark != schedule_row.benchmark
                or row.partition != schedule_row.feature_partition
                or row.prompt_id != schedule_row.prompt_id
                or row.outcome is not resolution.outcome
                or row.semantic_group_id != schedule_row.semantic_group_id
                or row.rendered_prompt_sha256 != resolution.rendered_prompt_sha256
                or row.prompt_token_ids_sha256 != resolution.prompt_token_ids_sha256
                or row.generation_record_sha256 != resolution.generation_record_sha256
            ):
                raise FrozenArtifactError("E2 activation prefix differs from its resolution")
            completed += 1
    return completed


def verify_e2_capture_work(
    directory: str | Path,
    *,
    workspace: VerifiedE2Workspace,
    questions: Mapping[tuple[str, str], Question],
    prompts: Mapping[str, PromptSpec],
    e1_sources: Mapping[tuple[str, str], E1P0Source],
    require_complete: bool = False,
) -> Mapping[str, Any]:
    """Replay a capture workspace without loading the model or mutating journals."""

    _validate_capture_inputs(workspace, questions, prompts, e1_sources)
    work = validate_active_study_artifact_paths(
        {"E2 capture work": directory, "E2 workspace": workspace.directory}
    )["E2 capture work"]
    if (
        work.is_symlink()
        or not work.is_dir()
        or {path.name for path in work.iterdir()} != _WORK_FILES
        or any(path.is_symlink() or not path.is_file() for path in work.iterdir())
    ):
        raise FrozenArtifactError("E2 capture work inventory differs")
    plan = _load_plan(work / "plan.json")
    frozen_runtime_identity = _validated_runtime_identity(plan["runtime_identity"])
    expected_plan = _capture_plan(
        workspace,
        questions,
        prompts,
        e1_sources,
        shard_rows=plan["shard_rows"],
        max_new_tokens=plan["max_new_tokens"],
        expected_runtime_identity=frozen_runtime_identity,
    )
    if plan != expected_plan:
        raise FrozenArtifactError("E2 capture work plan differs from live inputs")
    resolutions = _load_resolutions(work / "resolutions.jsonl", workspace=workspace)
    activation = verify_activation_store(
        workspace.directory / "activations",
        expected_spec=workspace.activation_spec,
        require_complete=require_complete,
    )
    completed = _verify_capture_prefix(workspace, resolutions)
    sessions = _load_sessions(
        work / "sessions.jsonl",
        capture_plan_identity=plan["capture_plan_identity"],
        expected_runtime_identity=frozen_runtime_identity,
    )
    complete = completed == len(workspace.schedule)
    if sessions:
        end = sessions[-1]
        if (
            end["event"] != "end"
            or end["rows_at_end"] != completed
            or end["activation_chain_head"] != activation.chain_head
            or end["resolution_chain_head"] != _resolution_chain_head(resolutions)
            or (end["status"] == "complete") != complete
        ):
            raise FrozenArtifactError("E2 capture session head differs from its stores")
    elif completed or resolutions:
        raise FrozenArtifactError("E2 capture stores lack a session journal")
    if len(resolutions) < completed or (complete and len(resolutions) != completed):
        raise FrozenArtifactError("E2 capture resolution prefix differs from activations")
    if require_complete and not complete:
        raise FrozenArtifactError("E2 capture is incomplete")
    return MappingProxyType(
        {
            "valid": True,
            "complete": complete,
            "rows_completed": completed,
            "rows_expected": len(workspace.schedule),
            "resolutions_completed": len(resolutions),
            "capture_plan_identity": plan["capture_plan_identity"],
            "activation_chain_head": activation.chain_head,
            "resolution_chain_head": _resolution_chain_head(resolutions),
        }
    )


def run_e2_capture(
    directory: str | Path,
    *,
    workspace: VerifiedE2Workspace,
    questions: Mapping[tuple[str, str], Question],
    prompts: Mapping[str, PromptSpec],
    e1_sources: Mapping[tuple[str, str], E1P0Source],
    runtime: E2CaptureRuntime,
    request_budget: int | None = None,
) -> Mapping[str, Any]:
    """Resolve and capture a prefix; rerunning resumes from immutable shards."""

    if request_budget is not None and (
        type(request_budget) is not int or request_budget <= 0
    ):
        raise ConfigurationError("E2 capture request budget must be positive")
    _validate_capture_inputs(workspace, questions, prompts, e1_sources)
    work = Path(directory)
    if (
        work.is_symlink()
        or not work.is_dir()
        or {path.name for path in work.iterdir()} != _WORK_FILES
    ):
        raise FrozenArtifactError("E2 capture work inventory differs")
    if any(path.is_symlink() or not path.is_file() for path in work.iterdir()):
        raise FrozenArtifactError("E2 capture work files must be regular")
    plan = _load_plan(work / "plan.json")
    frozen_runtime_identity = _validated_runtime_identity(plan["runtime_identity"])
    expected_plan = _capture_plan(
        workspace,
        questions,
        prompts,
        e1_sources,
        shard_rows=plan["shard_rows"],
        max_new_tokens=plan["max_new_tokens"],
        expected_runtime_identity=frozen_runtime_identity,
    )
    if plan != expected_plan:
        raise FrozenArtifactError("E2 capture work plan differs from live inputs")
    resolutions = _load_resolutions(work / "resolutions.jsonl", workspace=workspace)
    completed = _verify_capture_prefix(workspace, resolutions)
    _repair_torn_session_tail(work / "sessions.jsonl")
    sessions = _load_sessions(
        work / "sessions.jsonl",
        capture_plan_identity=plan["capture_plan_identity"],
        expected_runtime_identity=frozen_runtime_identity,
        allow_open=True,
    )
    if len(sessions) % 2:
        open_event = sessions[-1]
        activation = verify_activation_store(
            workspace.directory / "activations",
            expected_spec=workspace.activation_spec,
        )
        _append_session(
            work / "sessions.jsonl",
            {
                "schema_version": 1,
                "event": "end",
                "session_index": open_event["session_index"],
                "capture_plan_identity": plan["capture_plan_identity"],
                "status": "interrupted-recovered",
                "rows_at_end": activation.rows_completed,
                "activation_chain_head": activation.chain_head,
                "resolution_chain_head": _resolution_chain_head(resolutions),
                "wall_time_seconds": 0.0,
                "capture_peak_memory_bytes": 0,
                "created_unix_ns": time.time_ns(),
            },
            open_event["session_event_digest"],
        )
        sessions = _load_sessions(
            work / "sessions.jsonl",
            capture_plan_identity=plan["capture_plan_identity"],
            expected_runtime_identity=frozen_runtime_identity,
        )
    session_index = len(sessions) // 2
    runtime_identity = _validated_runtime_identity(runtime.runtime_identity())
    if runtime_identity != frozen_runtime_identity:
        raise FrozenArtifactError("E2 live runtime identity differs from the frozen capture plan")
    started = time.perf_counter()
    session_head = sessions[-1]["session_event_digest"] if sessions else None
    session_head = _append_session(
        work / "sessions.jsonl",
        {
            "schema_version": 1,
            "event": "start",
            "session_index": session_index,
            "capture_plan_identity": plan["capture_plan_identity"],
            "rows_at_start": completed,
            "runtime_identity": dict(runtime_identity),
            "created_unix_ns": time.time_ns(),
        },
        session_head,
    )
    captured_rows: list[ActivationCaptureRow] = []
    captured_values: list[np.ndarray[Any, Any]] = []
    processed = 0
    capture_peak_memory_bytes = 0
    status = "error"
    try:
        while completed + len(captured_rows) < len(workspace.schedule) and (
            request_budget is None or processed < request_budget
        ):
            sequence = completed + len(captured_rows)
            schedule_row = workspace.schedule[sequence]
            question = questions[(schedule_row.benchmark, schedule_row.question_id)]
            prompt = prompts[schedule_row.prompt_id]
            rendered = runtime.render_prompt(
                prompt,
                question.text,
                metadata=dict(question.metadata),
            )
            if sequence < len(resolutions):
                resolution = resolutions[sequence]
                if (
                    resolution.rendered_prompt_sha256 != rendered.sha256
                    or resolution.prompt_token_ids_sha256 != rendered.token_ids_sha256
                ):
                    raise FrozenArtifactError("E2 resumed rendering differs")
            else:
                resolution = _make_resolution(
                    row=schedule_row,
                    workspace=workspace,
                    question=question,
                    rendered=rendered,
                    runtime=runtime,
                    source=e1_sources.get((schedule_row.benchmark, schedule_row.question_id)),
                    max_new_tokens=plan["max_new_tokens"],
                )
                previous = _resolution_chain_head(resolutions)
                _append_resolution(work / "resolutions.jsonl", resolution, previous)
                resolutions.append(resolution)
            capture_row, values, peak_memory_bytes = _capture_row(
                workspace=workspace,
                schedule_row=schedule_row,
                resolution=resolution,
                rendered=rendered,
                runtime=runtime,
            )
            captured_rows.append(capture_row)
            captured_values.append(values)
            capture_peak_memory_bytes = max(
                capture_peak_memory_bytes, peak_memory_bytes
            )
            processed += 1
            if (
                len(captured_rows) == plan["shard_rows"]
                or completed + len(captured_rows) == len(workspace.schedule)
                or (request_budget is not None and processed == request_budget)
            ):
                append_activation_shard(
                    workspace.directory / "activations",
                    tuple(captured_rows),
                    np.stack(captured_values, axis=0),
                    expected_spec=workspace.activation_spec,
                )
                completed += len(captured_rows)
                captured_rows.clear()
                captured_values.clear()
        status = "complete" if completed == len(workspace.schedule) else "partial"
    finally:
        verified = verify_activation_store(
            workspace.directory / "activations",
            expected_spec=workspace.activation_spec,
            require_complete=status == "complete",
        )
        _append_session(
            work / "sessions.jsonl",
            {
                "schema_version": 1,
                "event": "end",
                "session_index": session_index,
                "capture_plan_identity": plan["capture_plan_identity"],
                "status": status,
                "rows_at_end": verified.rows_completed,
                "activation_chain_head": verified.chain_head,
                "resolution_chain_head": (
                    _resolution_chain_head(resolutions)
                ),
                "wall_time_seconds": time.perf_counter() - started,
                "capture_peak_memory_bytes": capture_peak_memory_bytes,
                "created_unix_ns": time.time_ns(),
            },
            session_head,
        )
        _load_sessions(
            work / "sessions.jsonl",
            capture_plan_identity=plan["capture_plan_identity"],
            expected_runtime_identity=frozen_runtime_identity,
        )
    resolutions = _load_resolutions(work / "resolutions.jsonl", workspace=workspace)
    completed = _verify_capture_prefix(workspace, resolutions)
    return MappingProxyType(
        {
            "status": status,
            "complete": status == "complete",
            "rows_completed": completed,
            "rows_expected": len(workspace.schedule),
            "resolutions_completed": len(resolutions),
            "capture_plan_identity": plan["capture_plan_identity"],
        }
    )

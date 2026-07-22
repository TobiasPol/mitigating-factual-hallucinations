"""Resumable, tamper-evident E0 execution for the sole active Qwen VLLM model."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from mfh.config import (
    load_inference_protocol,
    load_model_spec,
    load_prompt_specs,
)
from mfh.contracts import GenerationRecord, ModelSpec, Outcome, PromptSpec, Question, Runtime
from mfh.data.io import read_questions, write_generation_records
from mfh.data.normalization import normalize_answer
from mfh.data.runtime_validation import verify_runtime_validation_bundle
from mfh.errors import ConfigurationError, DataValidationError, FrozenArtifactError
from mfh.evaluation.grading import deterministic_short_answer_grade, triviaqa_scores
from mfh.experiments.model_selection import (
    APPROVED_AMENDMENT_DIGEST,
    load_model_selection_amendment,
    validate_active_model_spec,
    validate_active_study_artifact_paths,
)
from mfh.experiments.protocol import ExperimentPhase, load_study_protocol
from mfh.experiments.runner import EvaluationCondition
from mfh.inference.transformers_snapshot import (
    reject_symlink_path_components,
    verify_transformers_snapshot,
)
from mfh.inference.vllm_preflight import validate_vllm_preflight_receipt
from mfh.inference.vllm_runtime import VllmGenerationOutput, VllmRenderedPrompt, VllmRuntime
from mfh.provenance import sha256_file, stable_hash

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_QUESTIONS = 500
_REPEATS = 2
_RECORDS = _QUESTIONS * _REPEATS
_WORK_FILES = frozenset({"plan.json", "records.jsonl", "sessions.jsonl"})
_FINAL_FILES = frozenset(
    {
        "plan.json",
        "records.jsonl",
        "sessions.jsonl",
        "prompts.jsonl",
        "determinism.jsonl",
        "generation-records.jsonl",
        "summary.json",
        "manifest.json",
    }
)
_VLLM_MODEL_CLASS = (
    "vllm.model_executor.models.qwen3_5.Qwen3_5ForConditionalGeneration"
)


class _Runtime(Protocol):
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

    def runtime_identity(self) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class _Prepared:
    questions: tuple[Question, ...]
    cohort_manifest: Mapping[str, Any]
    model: ModelSpec
    snapshot: Path
    snapshot_identity: Mapping[str, Any]
    runtime_config: Mapping[str, Any]
    prompt: PromptSpec
    max_new_tokens: int
    condition: EvaluationCondition
    amendment_digest: str
    input_hashes: Mapping[str, str]


def _require_sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise DataValidationError(f"{context} must be a lowercase SHA-256")
    return value


def _read_json(path: Path, context: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot read {context}: {exc}") from exc
    if not isinstance(raw, dict):
        raise DataValidationError(f"{context} must be a JSON object")
    return raw


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        raise FrozenArtifactError(f"refusing to overwrite frozen JSON: {path}") from None
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(descriptor, "ab") as handle:
        handle.write(_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path, context: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.is_symlink() or not path.is_file():
        raise DataValidationError(f"{context} must be a regular file")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.endswith("\n"):
                    raise DataValidationError(f"{context} has a partial final line")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise DataValidationError(f"{context} row must be an object")
                rows.append(value)
    except json.JSONDecodeError as exc:
        raise DataValidationError(f"invalid {context}: {exc}") from exc
    return rows


def _descriptor(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def _verify_inventory(directory: Path, expected: frozenset[str], *, exact: bool) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise DataValidationError("E0 VLLM artifact must be a regular directory")
    files: set[str] = set()
    directories: set[str] = set()
    for item in directory.rglob("*"):
        if item.is_symlink():
            raise DataValidationError("E0 VLLM artifact cannot contain symlinks")
        relative = item.relative_to(directory).as_posix()
        if item.is_file():
            files.add(relative)
        elif item.is_dir():
            directories.add(relative)
        else:
            raise DataValidationError("E0 VLLM artifact contains a special file")
    if directories or not files <= expected or (exact and files != expected):
        raise DataValidationError(
            f"E0 VLLM inventory differs: files={sorted(files)}, directories={sorted(directories)}"
        )


def _prepare(
    *,
    cohort_directory: Path,
    reserved_source: Path,
    expected_cohort_manifest_digest: str,
    parent_split_manifest_digest: str,
    contamination_manifest_digest: str,
    model_config: Path,
    snapshot_directory: Path,
    snapshot_manifest: Path,
    runtime_config: Path,
    prompt_config: Path,
    inference_config: Path,
    study_config: Path,
) -> _Prepared:
    cohort = verify_runtime_validation_bundle(
        cohort_directory,
        reserved_source=reserved_source,
        expected_manifest_digest=expected_cohort_manifest_digest,
        parent_split_manifest_digest=parent_split_manifest_digest,
        contamination_manifest_digest=contamination_manifest_digest,
    )
    questions = tuple(read_questions(cohort_directory / "questions.jsonl"))
    if len(questions) != _QUESTIONS:
        raise DataValidationError("E0 VLLM requires exactly 500 questions")
    model = load_model_spec(model_config)
    validate_active_model_spec(model)
    if model.runtime is not Runtime.VLLM:
        raise ConfigurationError("E0 VLLM requires the sole VLLM model config")
    amendment_path = study_config.parent / "model-selection-amendment.json"
    amendment = load_model_selection_amendment(
        amendment_path,
        model_config_directory=study_config.parent.parent / "models",
    )
    if amendment["amendment_digest"] != APPROVED_AMENDMENT_DIGEST:
        raise ConfigurationError("E0 VLLM requires the approved local amendment")
    snapshot_identity = verify_transformers_snapshot(
        model, snapshot_directory, snapshot_manifest
    )
    active_models = amendment.get("active_models")
    active_model = active_models[0] if isinstance(active_models, list) and active_models else None
    if not isinstance(active_model, Mapping):
        raise DataValidationError("E0 VLLM active model declaration is invalid")
    project_root = study_config.absolute().parents[2]
    policy_reference = active_model.get("runtime_policy")
    if not isinstance(policy_reference, str):
        raise DataValidationError("E0 VLLM active runtime policy is invalid")
    runtime = validate_vllm_preflight_receipt(
        runtime_config,
        project_root=project_root,
        model_config=model_config,
        snapshot_directory=snapshot_directory,
        snapshot_manifest=snapshot_manifest,
        runtime_policy=project_root / policy_reference,
    )
    runtime_model = runtime.get("model")
    if (
        not isinstance(runtime_model, Mapping)
        or runtime_model.get("name") != model.name
        or runtime_model.get("revision") != model.revision
        or not isinstance(runtime_model.get("snapshot_identity"), Mapping)
        or runtime_model.get("snapshot_identity") != snapshot_identity
    ):
        raise DataValidationError("E0 VLLM runtime preflight receipt differs from live inputs")
    prompts = {value.prompt_id: value for value in load_prompt_specs(prompt_config)}
    prompt = prompts.get("P0-neutral")
    if prompt is None:
        raise ConfigurationError("E0 VLLM requires P0-neutral")
    inference = load_inference_protocol(inference_config)
    if (
        inference.temperature != 0.0
        or inference.do_sample
        or inference.max_new_tokens != 48
        or inference.thinking_enabled
        or inference.retrieval_enabled
        or inference.tools_enabled
    ):
        raise ConfigurationError("E0 VLLM inference settings differ from deterministic decode")
    study = load_study_protocol(study_config)
    phase = study.phase(ExperimentPhase.E0)
    if phase.models != (model.name,) or phase.question_limit != _QUESTIONS:
        raise ConfigurationError("study protocol does not declare the sole E0 VLLM leg")
    condition = EvaluationCondition(
        phase=ExperimentPhase.E0,
        benchmark="shared_benign_factual_500",
        partition="runtime-validation",
        model_name=model.name,
        model_repository=model.repository,
        model_revision=model.revision,
        runtime=model.runtime,
        quantization=model.quantization,
        model_num_layers=model.num_layers,
        system_prompt_id=prompt.prompt_id,
        prompt_template_sha256=hashlib.sha256(prompt.text.encode("utf-8")).hexdigest(),
        steering_method="M0",
        method_artifact_sha256=None,
        layer=None,
        site=None,
        token_scope=None,
        alpha=0.0,
        sparsity=None,
        seed=17,
        study_protocol_digest=study.digest,
    )
    input_hashes = {
        "model_config": sha256_file(model_config),
        "snapshot_manifest": sha256_file(snapshot_manifest),
        "runtime_config": sha256_file(runtime_config),
        "prompt_config": sha256_file(prompt_config),
        "inference_config": sha256_file(inference_config),
        "study_config": sha256_file(study_config),
        "cohort_manifest": sha256_file(cohort_directory / "manifest.json"),
        "cohort_questions": sha256_file(cohort_directory / "questions.jsonl"),
    }
    return _Prepared(
        questions=questions,
        cohort_manifest=cohort,
        model=model,
        snapshot=snapshot_directory,
        snapshot_identity=snapshot_identity,
        runtime_config=runtime,
        prompt=prompt,
        max_new_tokens=inference.max_new_tokens,
        condition=condition,
        amendment_digest=str(amendment["amendment_digest"]),
        input_hashes=input_hashes,
    )


def _static_plan(prepared: _Prepared, runtime_identity: Mapping[str, Any]) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": 1,
        "phase": "E0",
        "runner": "native-vllm",
        "model": {
            "name": prepared.model.name,
            "repository": prepared.model.repository,
            "revision": prepared.model.revision,
            "runtime": prepared.model.runtime.value,
            "quantization": prepared.model.quantization,
            "num_layers": prepared.model.num_layers,
        },
        "snapshot_identity": dict(prepared.snapshot_identity),
        "runtime_preflight_receipt_digest": prepared.runtime_config["receipt_digest"],
        "runtime_identity": dict(runtime_identity),
        "amendment_digest": prepared.amendment_digest,
        "condition": prepared.condition.to_dict(),
        "input_hashes": dict(prepared.input_hashes),
        "schedule": {
            "questions": _QUESTIONS,
            "repeats": _REPEATS,
            "records": _RECORDS,
            "ordering": "question-major-adjacent-repeats",
            "max_new_tokens": prepared.max_new_tokens,
            "stop_condition": "eos_or_first_completed_short_answer",
            "thinking_enabled": False,
            "sampling": False,
        },
    }
    return {**body, "plan_identity": stable_hash(body)}


def _validate_runtime_identity(
    prepared: _Prepared, runtime_identity: Mapping[str, Any]
) -> None:
    frozen = prepared.runtime_config.get("runtime_identity")
    if not isinstance(frozen, Mapping):
        raise DataValidationError("VLLM preflight receipt lacks runtime identity")
    expected = dict(frozen)
    if (
        expected.get("model_class") != _VLLM_MODEL_CLASS
        or not isinstance(expected.get("tokenizer_class"), str)
        or expected.get("num_layers") != prepared.model.num_layers
        or expected.get("seed") != 17
    ):
        raise DataValidationError("VLLM preflight receipt runtime identity is invalid")
    if dict(runtime_identity) != expected:
        differing = sorted(
            key
            for key in set(runtime_identity) | set(expected)
            if runtime_identity.get(key) != expected.get(key)
        )
        raise DataValidationError(
            "live VLLM runtime identity differs from frozen receipt: "
            + ", ".join(differing)
        )


def _load_or_create_plan(work: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    path = work / "plan.json"
    if path.exists():
        observed = _read_json(path, "E0 VLLM plan")
        if observed != dict(expected):
            raise DataValidationError("E0 VLLM work directory belongs to another plan")
        return observed
    _write_json_once(path, expected)
    return dict(expected)


def _render_all(prepared: _Prepared, runtime: _Runtime) -> tuple[VllmRenderedPrompt, ...]:
    return tuple(
        runtime.render_prompt(
            prepared.prompt, question.text, metadata=dict(question.metadata)
        )
        for question in prepared.questions
    )


def _request_digest(
    prepared: _Prepared, question: Question, rendered: VllmRenderedPrompt
) -> str:
    return stable_hash(
        {
            "condition_id": prepared.condition.condition_id,
            "question_id": question.question_id,
            "rendered_prompt_sha256": rendered.sha256,
            "rendered_token_ids_sha256": rendered.token_ids_sha256,
            "max_new_tokens": prepared.max_new_tokens,
            "sampling": False,
            "thinking_enabled": False,
        }
    )


def _record_body(
    *,
    sequence: int,
    repeat_index: int,
    session_index: int,
    plan_identity: str,
    question: Question,
    rendered: VllmRenderedPrompt,
    generation: VllmGenerationOutput,
    prepared: _Prepared,
    previous_digest: str | None,
) -> dict[str, Any]:
    exact_match, token_f1 = triviaqa_scores(generation.text, question.aliases)
    outcome = deterministic_short_answer_grade(generation.text, question.aliases)
    return {
        "schema_version": 1,
        "sequence": sequence,
        "repeat_index": repeat_index,
        "session_index": session_index,
        "plan_identity": plan_identity,
        "condition_id": prepared.condition.condition_id,
        "question_id": question.question_id,
        "rendered_prompt_sha256": rendered.sha256,
        "rendered_token_ids_sha256": rendered.token_ids_sha256,
        "raw_output": generation.text,
        "raw_output_sha256": hashlib.sha256(generation.text.encode("utf-8")).hexdigest(),
        "raw_output_stable_hash": stable_hash(generation.text),
        "normalized_answer": normalize_answer(generation.text),
        "outcome": outcome.value,
        "exact_match": exact_match,
        "token_f1": token_f1,
        "latency_seconds": generation.latency_seconds,
        "input_tokens": generation.input_tokens,
        "output_tokens": generation.output_tokens,
        "token_ids": list(generation.token_ids),
        "stop_type": generation.stop_type,
        "stopping_token_id": generation.stopping_token_id,
        "prompt_tokens_per_second": generation.prompt_tokens_per_second,
        "generation_tokens_per_second": generation.generation_tokens_per_second,
        "peak_memory_bytes": generation.peak_memory_bytes,
        "active_memory_bytes": generation.active_memory_bytes,
        "cache_memory_bytes": generation.cache_memory_bytes,
        "request_digest": _request_digest(prepared, question, rendered),
        "previous_record_digest": previous_digest,
    }


def _seal_record(body: Mapping[str, Any]) -> dict[str, Any]:
    return {**body, "record_digest": stable_hash(body)}


def _validate_record(
    row: Mapping[str, Any],
    *,
    sequence: int,
    previous_digest: str | None,
    prepared: _Prepared,
    rendered: Sequence[VllmRenderedPrompt],
    plan_identity: str,
) -> None:
    body = dict(row)
    digest = body.pop("record_digest", None)
    if digest != stable_hash(body):
        raise DataValidationError("E0 VLLM record digest differs")
    question_index = sequence // _REPEATS
    repeat_index = sequence % _REPEATS
    question = prepared.questions[question_index]
    prompt = rendered[question_index]
    if (
        row.get("schema_version") != 1
        or row.get("sequence") != sequence
        or row.get("repeat_index") != repeat_index
        or row.get("plan_identity") != plan_identity
        or row.get("condition_id") != prepared.condition.condition_id
        or row.get("question_id") != question.question_id
        or row.get("rendered_prompt_sha256") != prompt.sha256
        or row.get("rendered_token_ids_sha256") != prompt.token_ids_sha256
        or row.get("request_digest") != _request_digest(prepared, question, prompt)
        or row.get("previous_record_digest") != previous_digest
    ):
        raise DataValidationError("E0 VLLM record differs from frozen schedule")
    raw = row.get("raw_output")
    tokens = row.get("token_ids")
    if (
        not isinstance(raw, str)
        or not isinstance(tokens, list)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in tokens)
        or row.get("raw_output_sha256")
        != hashlib.sha256(raw.encode("utf-8")).hexdigest()
        or row.get("raw_output_stable_hash") != stable_hash(raw)
        or row.get("normalized_answer") != normalize_answer(raw)
        or row.get("outcome")
        != deterministic_short_answer_grade(raw, question.aliases).value
    ):
        raise DataValidationError("E0 VLLM output evidence differs")
    exact_match, token_f1 = triviaqa_scores(raw, question.aliases)
    if row.get("exact_match") != exact_match or row.get("token_f1") != token_f1:
        raise DataValidationError("E0 VLLM grading evidence differs")
    integer_fields = (
        "session_index",
        "input_tokens",
        "output_tokens",
        "peak_memory_bytes",
        "active_memory_bytes",
        "cache_memory_bytes",
    )
    if any(
        isinstance(row.get(name), bool)
        or not isinstance(row.get(name), int)
        or int(row[name]) < 0
        for name in integer_fields
    ):
        raise DataValidationError("E0 VLLM integer metric is invalid")
    number_fields = (
        "latency_seconds",
        "prompt_tokens_per_second",
        "generation_tokens_per_second",
    )
    if any(
        isinstance(row.get(name), bool)
        or not isinstance(row.get(name), int | float)
        or float(row[name]) < 0
        or not math.isfinite(float(row[name]))
        for name in number_fields
    ):
        raise DataValidationError("E0 VLLM timing metric is invalid")


def _load_records(
    path: Path,
    *,
    prepared: _Prepared,
    rendered: Sequence[VllmRenderedPrompt],
    plan_identity: str,
) -> list[dict[str, Any]]:
    rows = _read_jsonl(path, "E0 VLLM records")
    if len(rows) > _RECORDS:
        raise DataValidationError("E0 VLLM record count exceeds the frozen schedule")
    previous: str | None = None
    for sequence, row in enumerate(rows):
        _validate_record(
            row,
            sequence=sequence,
            previous_digest=previous,
            prepared=prepared,
            rendered=rendered,
            plan_identity=plan_identity,
        )
        previous = str(row["record_digest"])
    return rows


def _append_session(
    path: Path,
    *,
    event: str,
    session_index: int,
    plan_identity: str,
    details: Mapping[str, Any],
) -> None:
    prior = _read_jsonl(path, "E0 VLLM session log")
    previous = str(prior[-1]["event_digest"]) if prior else None
    body = {
        "schema_version": 1,
        "event": event,
        "session_index": session_index,
        "plan_identity": plan_identity,
        "created_unix_ns": time.time_ns(),
        "details": dict(details),
        "previous_event_digest": previous,
    }
    _append_jsonl(path, {**body, "event_digest": stable_hash(body)})


def _validate_sessions(
    path: Path, *, plan_identity: str, allow_open: bool = False
) -> list[dict[str, Any]]:
    rows = _read_jsonl(path, "E0 VLLM session log")
    previous: str | None = None
    open_sessions: set[int] = set()
    for row in rows:
        body = dict(row)
        digest = body.pop("event_digest", None)
        event = row.get("event")
        index = row.get("session_index")
        if (
            digest != stable_hash(body)
            or row.get("previous_event_digest") != previous
            or row.get("plan_identity") != plan_identity
            or event not in {"start", "end"}
            or isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
        ):
            raise DataValidationError("E0 VLLM session event chain differs")
        if event == "start":
            if index in open_sessions:
                raise DataValidationError("E0 VLLM session starts twice")
            open_sessions.add(index)
        else:
            if index not in open_sessions:
                raise DataValidationError("E0 VLLM session ends without a start")
            open_sessions.remove(index)
        previous = str(digest)
    if open_sessions and not allow_open:
        raise DataValidationError("E0 VLLM session log has an unclosed session")
    return rows


def _resume_checkpoint(
    plan_identity: str,
    records: Sequence[Mapping[str, Any]],
    sessions: Sequence[Mapping[str, Any]],
) -> str:
    return stable_hash(
        {
            "plan_identity": plan_identity,
            "record_count": len(records),
            "record_head": records[-1]["record_digest"] if records else None,
            "session_count": len(sessions),
            "session_head": sessions[-1]["event_digest"] if sessions else None,
        }
    )


def _emit_checkpoint(
    *,
    plan_identity: str,
    records: Sequence[Mapping[str, Any]],
    sessions: Sequence[Mapping[str, Any]],
    reason: str,
    checkpoint_file: Path | None,
) -> str:
    digest = _resume_checkpoint(plan_identity, records, sessions)
    payload = {
        "schema_version": 1,
        "event": "e0-vllm-resume-checkpoint",
        "reason": reason,
        "plan_identity": plan_identity,
        "records_completed": len(records),
        "resume_checkpoint": digest,
    }
    if checkpoint_file is not None:
        _atomic_json(checkpoint_file, payload)
    print(json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)
    return digest


def _prompt_rows(
    prepared: _Prepared, rendered: Sequence[VllmRenderedPrompt]
) -> Iterable[Mapping[str, Any]]:
    for question, prompt in zip(prepared.questions, rendered, strict=True):
        yield {
            "schema_version": 1,
            "question_id": question.question_id,
            "rendered_prompt_sha256": prompt.sha256,
            "rendered_token_ids_sha256": prompt.token_ids_sha256,
            "input_tokens": len(prompt.token_ids),
        }


def _determinism_rows(records: Sequence[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
    for index in range(_QUESTIONS):
        first = records[index * 2]
        repeat = records[index * 2 + 1]
        yield {
            "schema_version": 1,
            "question_id": first["question_id"],
            "first_output_sha256": first["raw_output_sha256"],
            "repeat_output_sha256": repeat["raw_output_sha256"],
            "first_token_ids_sha256": stable_hash(first["token_ids"]),
            "repeat_token_ids_sha256": stable_hash(repeat["token_ids"]),
            "exact_match": (
                first["raw_output"] == repeat["raw_output"]
                and first["token_ids"] == repeat["token_ids"]
            ),
        }


def _generation_records(
    records: Sequence[Mapping[str, Any]], prepared: _Prepared
) -> tuple[GenerationRecord, ...]:
    result: list[GenerationRecord] = []
    for index, question in enumerate(prepared.questions):
        row = records[index * 2]
        result.append(
            GenerationRecord(
                question_id=question.question_id,
                benchmark=prepared.condition.benchmark,
                model_repository=prepared.model.repository,
                model_revision=prepared.model.revision,
                runtime=Runtime.VLLM,
                quantization=prepared.model.quantization,
                system_prompt_id=prepared.prompt.prompt_id,
                rendered_prompt_hash=str(row["rendered_prompt_sha256"]),
                steering_method="M0",
                layer=None,
                site=None,
                token_scope=None,
                alpha=0.0,
                sparsity=None,
                controller_scores={},
                raw_output=str(row["raw_output"]),
                normalized_answer=str(row["normalized_answer"]),
                outcome=Outcome(str(row["outcome"])),
                generation_latency_seconds=float(row["latency_seconds"]),
                input_tokens=int(row["input_tokens"]),
                output_tokens=int(row["output_tokens"]),
                condition_id=prepared.condition.condition_id,
                seed=17,
                metadata={
                    "phase": "E0",
                    "partition": "runtime-validation",
                    "prompt_template_sha256": prepared.condition.prompt_template_sha256,
                    "study_protocol_digest": prepared.condition.study_protocol_digest,
                    "repeat_record_digests": [
                        row["record_digest"],
                        records[index * 2 + 1]["record_digest"],
                    ],
                    "rendered_token_ids_sha256": row["rendered_token_ids_sha256"],
                    "token_ids": row["token_ids"],
                    "stop_type": row["stop_type"],
                    "peak_memory_bytes": row["peak_memory_bytes"],
                    "exact_match": row["exact_match"],
                    "token_f1": row["token_f1"],
                },
            )
        )
    return tuple(result)


def _summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    canonical = records[::2]
    outcomes = Counter(str(row["outcome"]) for row in canonical)
    latencies = [float(row["latency_seconds"]) for row in records]
    output_tokens = [int(row["output_tokens"]) for row in records]
    exact_pairs = sum(bool(row["exact_match"]) for row in _determinism_rows(records))
    return {
        "schema_version": 1,
        "questions": _QUESTIONS,
        "repeats": _REPEATS,
        "low_level_records": len(records),
        "deterministic_pairs": exact_pairs,
        "determinism_mismatches": _QUESTIONS - exact_pairs,
        "outcomes": {value.value: outcomes[value.value] for value in Outcome},
        "mean_latency_seconds": sum(latencies) / len(latencies),
        "max_latency_seconds": max(latencies),
        "mean_output_tokens": sum(output_tokens) / len(output_tokens),
        "max_output_tokens": max(output_tokens),
        "peak_memory_bytes": max(int(row["peak_memory_bytes"]) for row in records),
        "mean_generation_tokens_per_second": sum(
            float(row["generation_tokens_per_second"]) for row in records
        )
        / len(records),
        "outcome_interpretation": (
            "strict-whole-response-alias-exact-match-diagnostic-not-e0-accuracy"
        ),
    }


def _write_jsonl_once(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        raise FrozenArtifactError(f"refusing to overwrite frozen JSONL: {path}") from None
    with os.fdopen(descriptor, "wb") as handle:
        for row in rows:
            handle.write(_json_bytes(row))
        handle.flush()
        os.fsync(handle.fileno())


def _publish(
    output: Path,
    *,
    work: Path,
    prepared: _Prepared,
    rendered: Sequence[VllmRenderedPrompt],
    plan: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if output.exists():
        raise FrozenArtifactError(f"refusing to overwrite E0 VLLM bundle: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        for name in _WORK_FILES:
            shutil.copyfile(work / name, stage / name)
        _write_jsonl_once(stage / "prompts.jsonl", _prompt_rows(prepared, rendered))
        _write_jsonl_once(stage / "determinism.jsonl", _determinism_rows(records))
        write_generation_records(
            stage / "generation-records.jsonl", _generation_records(records, prepared)
        )
        _write_json_once(stage / "summary.json", _summary(records))
        artifacts = {
            name: _descriptor(stage / name) for name in sorted(_FINAL_FILES - {"manifest.json"})
        }
        body: dict[str, Any] = {
            "schema_version": 1,
            "phase": "E0",
            "completed_scope": "sole-qwen3.6-27b-nvfp4-runtime",
            "model_name": prepared.model.name,
            "plan_identity": plan["plan_identity"],
            "condition_id": prepared.condition.condition_id,
            "counts": {
                "questions": _QUESTIONS,
                "repeats": _REPEATS,
                "low_level_records": _RECORDS,
                "canonical_generation_records": _QUESTIONS,
            },
            "artifacts": artifacts,
            "scientific_status": {
                "e0_runtime_validation_complete": True,
                "manual_contamination_review_required_for_e1": True,
                "superseded_runtime_artifacts_excluded": True,
            },
        }
        manifest = {**body, "manifest_digest": stable_hash(body)}
        _write_json_once(stage / "manifest.json", manifest)
        _verify_with_prepared(
            stage,
            expected_manifest_digest=str(manifest["manifest_digest"]),
            expected_plan_identity=str(plan["plan_identity"]),
            prepared=prepared,
            rendered=rendered,
        )
        os.replace(stage, output)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _verify_with_prepared(
    directory: Path,
    *,
    expected_manifest_digest: str,
    expected_plan_identity: str,
    prepared: _Prepared,
    rendered: Sequence[VllmRenderedPrompt],
) -> Mapping[str, Any]:
    _verify_inventory(directory, _FINAL_FILES, exact=True)
    manifest = _read_json(directory / "manifest.json", "E0 VLLM manifest")
    body = dict(manifest)
    digest = body.pop("manifest_digest", None)
    if digest != stable_hash(body) or digest != expected_manifest_digest:
        raise DataValidationError("E0 VLLM manifest identity differs")
    plan = _read_json(directory / "plan.json", "E0 VLLM plan")
    if plan.get("plan_identity") != expected_plan_identity:
        raise DataValidationError("E0 VLLM plan identity differs")
    if plan.get("input_hashes") != dict(prepared.input_hashes):
        raise DataValidationError("E0 VLLM plan live input hashes differ")
    records = _load_records(
        directory / "records.jsonl",
        prepared=prepared,
        rendered=rendered,
        plan_identity=expected_plan_identity,
    )
    if len(records) != _RECORDS:
        raise DataValidationError("E0 VLLM final records are incomplete")
    _validate_sessions(directory / "sessions.jsonl", plan_identity=expected_plan_identity)
    if _read_jsonl(directory / "prompts.jsonl", "E0 VLLM prompts") != list(
        _prompt_rows(prepared, rendered)
    ):
        raise DataValidationError("E0 VLLM prompt evidence differs")
    determinism = list(_determinism_rows(records))
    if _read_jsonl(directory / "determinism.jsonl", "E0 VLLM determinism") != determinism:
        raise DataValidationError("E0 VLLM determinism evidence differs")
    if any(row["exact_match"] is not True for row in determinism):
        raise DataValidationError("E0 VLLM repeated generations are not deterministic")
    expected_generation = [value.to_dict() for value in _generation_records(records, prepared)]
    observed_generation = _read_jsonl(
        directory / "generation-records.jsonl", "E0 VLLM generation records"
    )
    if observed_generation != expected_generation:
        raise DataValidationError("E0 VLLM canonical generation records differ")
    if _read_json(directory / "summary.json", "E0 VLLM summary") != _summary(records):
        raise DataValidationError("E0 VLLM summary differs")
    artifacts = {
        name: _descriptor(directory / name)
        for name in sorted(_FINAL_FILES - {"manifest.json"})
    }
    if manifest.get("artifacts") != artifacts:
        raise DataValidationError("E0 VLLM artifact descriptors differ")
    return manifest


def _load_tokenizer_renderer(prepared: _Prepared) -> _Runtime:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional verification dependency
        raise ConfigurationError("vLLM tokenizer verification requires transformers") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        str(prepared.snapshot), local_files_only=True, trust_remote_code=False
    )
    runtime = VllmRuntime(
        engine=None,
        tokenizer=tokenizer,
        model_spec=prepared.model,
        snapshot=prepared.snapshot,
    )

    class TokenizerRenderer:
        def render_prompt(
            self,
            prompt: PromptSpec,
            question: str,
            *,
            metadata: Mapping[str, Any] | None = None,
        ) -> VllmRenderedPrompt:
            return runtime.render_prompt(prompt, question, metadata=metadata)

        def generate(
            self, rendered: VllmRenderedPrompt, *, max_new_tokens: int
        ) -> VllmGenerationOutput:
            raise AssertionError("verification renderer cannot generate")

        def runtime_identity(self) -> Mapping[str, Any]:
            return {}

        def close(self) -> None:
            return None

    return TokenizerRenderer()


def run_vllm_e0(
    *,
    cohort_directory: str | Path,
    reserved_source: str | Path,
    expected_cohort_manifest_digest: str,
    parent_split_manifest_digest: str,
    contamination_manifest_digest: str,
    model_config: str | Path,
    snapshot_directory: str | Path,
    snapshot_manifest: str | Path,
    runtime_config: str | Path,
    work_directory: str | Path,
    output_directory: str | Path,
    prompt_config: str | Path = "configs/prompts/primary.yaml",
    inference_config: str | Path = "configs/experiments/core.yaml",
    study_config: str | Path = "configs/experiments/phases.yaml",
    request_budget: int | None = None,
    expected_resume_checkpoint: str | None = None,
    checkpoint_file: str | Path | None = None,
    runtime_factory: Callable[[ModelSpec, Path], _Runtime] | None = None,
) -> Mapping[str, Any]:
    """Run or resume the exact 500-question, two-pass native VLLM E0 leg."""

    if request_budget is not None and (
        isinstance(request_budget, bool)
        or not isinstance(request_budget, int)
        or request_budget <= 0
    ):
        raise ConfigurationError("E0 VLLM request budget must be a positive integer")
    if expected_resume_checkpoint is not None:
        _require_sha256(expected_resume_checkpoint, "expected E0 VLLM resume checkpoint")
    paths = {
        "cohort_directory": Path(cohort_directory).absolute(),
        "reserved_source": Path(reserved_source).absolute(),
        "model_config": Path(model_config).absolute(),
        "snapshot_directory": Path(snapshot_directory).absolute(),
        "snapshot_manifest": Path(snapshot_manifest).absolute(),
        "runtime_config": Path(runtime_config).absolute(),
        "prompt_config": Path(prompt_config).absolute(),
        "inference_config": Path(inference_config).absolute(),
        "study_config": Path(study_config).absolute(),
    }
    work = reject_symlink_path_components(work_directory, "E0 VLLM work directory")
    output = reject_symlink_path_components(output_directory, "E0 VLLM output directory")
    checkpoint = (
        reject_symlink_path_components(checkpoint_file, "E0 VLLM checkpoint file")
        if checkpoint_file is not None
        else None
    )
    mutable_paths = {"E0 work directory": work, "E0 output directory": output}
    if checkpoint is not None:
        mutable_paths["E0 checkpoint file"] = checkpoint
    validate_active_study_artifact_paths(mutable_paths)
    if checkpoint is not None and (
        checkpoint in {work, output}
        or checkpoint.is_relative_to(work)
        or checkpoint.is_relative_to(output)
    ):
        raise ConfigurationError("E0 VLLM checkpoint file must stay outside work and output")
    if output.exists():
        raise FrozenArtifactError(f"refusing to overwrite E0 VLLM bundle: {output}")
    work.mkdir(parents=True, exist_ok=True)
    _verify_inventory(work, _WORK_FILES, exact=False)
    resume = (work / "plan.json").exists()
    if resume != (expected_resume_checkpoint is not None):
        raise ConfigurationError(
            "existing E0 VLLM work requires its external resume checkpoint"
            if resume
            else "an E0 VLLM resume checkpoint cannot initialize new work"
        )
    prepared = _prepare(
        **paths,
        expected_cohort_manifest_digest=expected_cohort_manifest_digest,
        parent_split_manifest_digest=parent_split_manifest_digest,
        contamination_manifest_digest=contamination_manifest_digest,
    )
    factory = runtime_factory or (
        lambda model, snapshot: VllmRuntime.from_spec(model, snapshot_path=snapshot, seed=17)
    )
    runtime: _Runtime | None = None
    records: list[dict[str, Any]] = []
    plan: Mapping[str, Any] | None = None
    rendered: tuple[VllmRenderedPrompt, ...] = ()
    session_index = -1
    status = "error"
    active_error: BaseException | None = None
    try:
        runtime = factory(prepared.model, prepared.snapshot)
        identity = dict(runtime.runtime_identity())
        _validate_runtime_identity(prepared, identity)
        plan = _load_or_create_plan(work, _static_plan(prepared, identity))
        plan_identity = str(plan["plan_identity"])
        rendered = _render_all(prepared, runtime)
        records = _load_records(
            work / "records.jsonl",
            prepared=prepared,
            rendered=rendered,
            plan_identity=plan_identity,
        )
        sessions = _validate_sessions(
            work / "sessions.jsonl", plan_identity=plan_identity, allow_open=resume
        )
        observed_checkpoint = _resume_checkpoint(plan_identity, records, sessions)
        if resume and observed_checkpoint != expected_resume_checkpoint:
            raise DataValidationError("E0 VLLM resume checkpoint differs from external head")
        if resume:
            starts = {
                int(row["session_index"])
                for row in sessions
                if row.get("event") == "start"
            }
            ends = {
                int(row["session_index"])
                for row in sessions
                if row.get("event") == "end"
            }
            for interrupted in sorted(starts - ends):
                _append_session(
                    work / "sessions.jsonl",
                    event="end",
                    session_index=interrupted,
                    plan_identity=plan_identity,
                    details={
                        "records_at_end": len(records),
                        "status": "interrupted-recovered",
                    },
                )
            sessions = _validate_sessions(
                work / "sessions.jsonl", plan_identity=plan_identity
            )
        session_index = (
            max((int(row["session_index"]) for row in sessions), default=-1) + 1
        )
        _append_session(
            work / "sessions.jsonl",
            event="start",
            session_index=session_index,
            plan_identity=plan_identity,
            details={"records_at_start": len(records), "runtime_identity": identity},
        )
        sessions = _read_jsonl(work / "sessions.jsonl", "E0 VLLM session log")
        _emit_checkpoint(
            plan_identity=plan_identity,
            records=records,
            sessions=sessions,
            reason="session-start",
            checkpoint_file=checkpoint,
        )
        new_requests = 0
        while len(records) < _RECORDS and (
            request_budget is None or new_requests < request_budget
        ):
            sequence = len(records)
            question_index = sequence // _REPEATS
            repeat_index = sequence % _REPEATS
            question = prepared.questions[question_index]
            prompt = rendered[question_index]
            generation = runtime.generate(prompt, max_new_tokens=prepared.max_new_tokens)
            body = _record_body(
                sequence=sequence,
                repeat_index=repeat_index,
                session_index=session_index,
                plan_identity=plan_identity,
                question=question,
                rendered=prompt,
                generation=generation,
                prepared=prepared,
                previous_digest=(str(records[-1]["record_digest"]) if records else None),
            )
            record = _seal_record(body)
            _validate_record(
                record,
                sequence=sequence,
                previous_digest=(str(records[-1]["record_digest"]) if records else None),
                prepared=prepared,
                rendered=rendered,
                plan_identity=plan_identity,
            )
            _append_jsonl(work / "records.jsonl", record)
            records.append(record)
            new_requests += 1
            sessions = _read_jsonl(work / "sessions.jsonl", "E0 VLLM session log")
            _emit_checkpoint(
                plan_identity=plan_identity,
                records=records,
                sessions=sessions,
                reason="record-appended",
                checkpoint_file=checkpoint,
            )
            if len(records) <= 4 or len(records) % 10 == 0 or repeat_index == 1:
                paired = (
                    None
                    if repeat_index == 0
                    else records[-2]["raw_output"] == record["raw_output"]
                    and records[-2]["token_ids"] == record["token_ids"]
                )
                print(
                    json.dumps(
                        {
                            "event": "e0-vllm-progress",
                            "records_completed": len(records),
                            "records_expected": _RECORDS,
                            "question_id": question.question_id,
                            "repeat_index": repeat_index,
                            "latency_seconds": round(generation.latency_seconds, 6),
                            "output_tokens": generation.output_tokens,
                            "paired_match": paired,
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
        status = "complete" if len(records) == _RECORDS else "partial"
    except BaseException as exc:
        active_error = exc
    finally:
        if plan is not None and session_index >= 0:
            try:
                _append_session(
                    work / "sessions.jsonl",
                    event="end",
                    session_index=session_index,
                    plan_identity=str(plan["plan_identity"]),
                    details={"records_at_end": len(records), "status": status},
                )
                sessions = _validate_sessions(
                    work / "sessions.jsonl", plan_identity=str(plan["plan_identity"])
                )
                _emit_checkpoint(
                    plan_identity=str(plan["plan_identity"]),
                    records=records,
                    sessions=sessions,
                    reason="session-end",
                    checkpoint_file=checkpoint,
                )
            except BaseException as exc:
                if active_error is None:
                    active_error = exc
        if runtime is not None:
            try:
                runtime.close()
            except BaseException as exc:
                if active_error is None:
                    active_error = exc
    post_snapshot = verify_transformers_snapshot(
        prepared.model, prepared.snapshot, paths["snapshot_manifest"]
    )
    if post_snapshot != prepared.snapshot_identity and active_error is None:
        active_error = DataValidationError("E0 VLLM snapshot changed during execution")
    if active_error is not None:
        raise active_error
    if plan is None or not rendered:
        raise DataValidationError("E0 VLLM ended without a frozen plan and prompts")
    sessions = _validate_sessions(
        work / "sessions.jsonl", plan_identity=str(plan["plan_identity"])
    )
    result: dict[str, Any] = {
        "complete": status == "complete",
        "model_name": prepared.model.name,
        "records_completed": len(records),
        "records_expected": _RECORDS,
        "questions_with_both_repeats": len(records) // _REPEATS,
        "plan_identity": plan["plan_identity"],
        "resume_checkpoint": _resume_checkpoint(str(plan["plan_identity"]), records, sessions),
        "work_directory": str(work),
    }
    if status == "complete":
        manifest = _publish(
            output,
            work=work,
            prepared=prepared,
            rendered=rendered,
            plan=plan,
            records=records,
        )
        result.update(
            {
                "manifest_digest": manifest["manifest_digest"],
                "output_directory": str(output),
                "summary": _summary(records),
            }
        )
    return result


def verify_vllm_e0_bundle(
    directory: str | Path,
    *,
    expected_manifest_digest: str,
    expected_plan_identity: str,
    cohort_directory: str | Path,
    reserved_source: str | Path,
    expected_cohort_manifest_digest: str,
    parent_split_manifest_digest: str,
    contamination_manifest_digest: str,
    model_config: str | Path,
    snapshot_directory: str | Path,
    snapshot_manifest: str | Path,
    runtime_config: str | Path,
    prompt_config: str | Path = "configs/prompts/primary.yaml",
    inference_config: str | Path = "configs/experiments/core.yaml",
    study_config: str | Path = "configs/experiments/phases.yaml",
    renderer_factory: Callable[[_Prepared], _Runtime] = _load_tokenizer_renderer,
) -> Mapping[str, Any]:
    """Replay a completed VLLM E0 bundle without loading model weights."""

    _require_sha256(expected_manifest_digest, "expected E0 VLLM manifest")
    _require_sha256(expected_plan_identity, "expected E0 VLLM plan")
    prepared = _prepare(
        cohort_directory=Path(cohort_directory).absolute(),
        reserved_source=Path(reserved_source).absolute(),
        expected_cohort_manifest_digest=expected_cohort_manifest_digest,
        parent_split_manifest_digest=parent_split_manifest_digest,
        contamination_manifest_digest=contamination_manifest_digest,
        model_config=Path(model_config).absolute(),
        snapshot_directory=Path(snapshot_directory).absolute(),
        snapshot_manifest=Path(snapshot_manifest).absolute(),
        runtime_config=Path(runtime_config).absolute(),
        prompt_config=Path(prompt_config).absolute(),
        inference_config=Path(inference_config).absolute(),
        study_config=Path(study_config).absolute(),
    )
    renderer = renderer_factory(prepared)
    try:
        rendered = _render_all(prepared, renderer)
    finally:
        renderer.close()
    return _verify_with_prepared(
        reject_symlink_path_components(directory, "E0 VLLM bundle"),
        expected_manifest_digest=expected_manifest_digest,
        expected_plan_identity=expected_plan_identity,
        prepared=prepared,
        rendered=rendered,
    )

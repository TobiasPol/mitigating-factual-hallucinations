"""Frozen auxiliary AA Public-600 official-prompt and official-score track."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from mfh.config import load_prompt_specs
from mfh.contracts import GenerationRecord, Outcome, Question
from mfh.errors import ConfigurationError, DataValidationError, FrozenArtifactError
from mfh.evaluation.metrics import metric_bundle
from mfh.evaluation.official import (
    GradingRequest,
    OfficialGraderSpec,
    aa_official_metrics,
    render_grader_prompt,
)
from mfh.evaluation.openrouter import (
    OpenRouterTransport,
    route_for_grader,
    run_openrouter_grader,
    validate_openrouter_attempt_receipt,
)
from mfh.experiments.e1_vllm import (
    E1Prepared,
    _append_jsonl,
    _atomic_json,
    _generation_body,
    _prepare,
    _read_json,
    _read_jsonl,
    _Runtime,
    _session_event,
    _tokenizer_renderer,
    _validate_generation,
    _validate_runtime_identity,
    _validate_sessions,
    _write_json_once,
)
from mfh.experiments.grader_bundle import verify_e1_grader_bundle
from mfh.experiments.model_selection import validate_active_study_artifact_paths
from mfh.experiments.protocol import ExperimentPhase
from mfh.experiments.runner import EvaluationCondition, PhaseRunLedger
from mfh.inference.vllm_runtime import VllmRuntime
from mfh.provenance import canonical_json, sha256_file, sha256_path, stable_hash

_BENCHMARK = "aa_omniscience_public_600"
_PROMPT_ID = "P-AA-official"
_OFFICIAL_PROMPT_TEXT = (
    "You are answering questions about {domain}, and in particular {topic}.\n"
    "You will be given a question, answer with JUST the answer (no explanation).\n"
    "If you do not know the answer, or you need more context or tools to answer the question,\n"
    "be clear about this - it is better that you say this than get the wrong answer."
)
_SEED = 17
_ROW_COUNT = 600
_MEMORY_BYTES = 40 * 1024**3
_WORK_FILES = frozenset({"plan.json", "records.jsonl", "failures.jsonl", "sessions.jsonl"})
_OUTPUT_FILES = frozenset(
    {
        "manifest.json",
        "plan.json",
        "records.jsonl",
        "failures.jsonl",
        "sessions.jsonl",
        "metrics.json",
        "official-vs-neutral.json",
    }
)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, slots=True)
class AAOfficialContext:
    """All live-frozen identities needed to execute or replay the auxiliary track."""

    prepared: E1Prepared
    grader: OfficialGraderSpec
    grader_manifest_digest: str
    e1_completion_digest: str
    plan: Mapping[str, Any]


def _question_body(question: Question) -> dict[str, Any]:
    return {
        "question_id": question.question_id,
        "benchmark": question.benchmark,
        "text": question.text,
        "aliases": list(question.aliases),
        "split": question.split,
        "entities": list(question.entities),
        "metadata": dict(question.metadata),
    }


def _official_schedule(
    condition: EvaluationCondition,
    questions: Sequence[Question],
) -> tuple[tuple[EvaluationCondition, Question], ...]:
    rows = tuple((condition, question) for question in questions)
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                stable_hash(
                    {
                        "schema_version": 1,
                        "seed": _SEED,
                        "condition_id": row[0].condition_id,
                        "question_id": row[1].question_id,
                    }
                ),
                row[1].question_id,
            ),
        )
    )


def _paths(
    *,
    splits_directory: str | Path,
    grader_bundle: str | Path,
    model_config: str | Path,
    snapshot_directory: str | Path,
    snapshot_manifest: str | Path,
    runtime_config: str | Path,
    ledger_directory: str | Path,
    e0_run: str | Path,
    prompt_config: str | Path,
    inference_config: str | Path,
    study_config: str | Path,
) -> dict[str, Path]:
    return {
        name: Path(value).absolute()
        for name, value in {
            "splits_directory": splits_directory,
            "grader_bundle": grader_bundle,
            "model_config": model_config,
            "snapshot_directory": snapshot_directory,
            "snapshot_manifest": snapshot_manifest,
            "runtime_config": runtime_config,
            "ledger_directory": ledger_directory,
            "e0_run": e0_run,
            "prompt_config": prompt_config,
            "inference_config": inference_config,
            "study_config": study_config,
        }.items()
    }


def _context(
    *,
    expected_splits_manifest_digest: str,
    expected_grader_manifest_digest: str,
    **paths: Path,
) -> AAOfficialContext:
    base = _prepare(
        splits_directory=paths["splits_directory"],
        expected_splits_manifest_digest=expected_splits_manifest_digest,
        grader_bundle=paths["grader_bundle"],
        expected_grader_manifest_digest=expected_grader_manifest_digest,
        model_config=paths["model_config"],
        snapshot_directory=paths["snapshot_directory"],
        snapshot_manifest=paths["snapshot_manifest"],
        runtime_config=paths["runtime_config"],
        prompt_config=paths["prompt_config"],
        inference_config=paths["inference_config"],
        study_config=paths["study_config"],
        e0_run=paths["e0_run"],
    )
    ledger = PhaseRunLedger.open(paths["ledger_directory"], study=base.study)
    completion = ledger.verify_complete()
    if completion.phase is not ExperimentPhase.E1 or ledger.contract != base.contract:
        raise FrozenArtifactError("AA official track requires the exact complete E1 ledger")
    prompts = {value.prompt_id: value for value in load_prompt_specs(paths["prompt_config"])}
    try:
        prompt = prompts[_PROMPT_ID]
    except KeyError as exc:
        raise ConfigurationError(
            "AA official prompt is absent from the frozen prompt file"
        ) from exc
    if (
        prompt.text != _OFFICIAL_PROMPT_TEXT
        or prompt.permits_abstention is not True
        or prompt.deployment_eligible is not True
    ):
        raise FrozenArtifactError("AA official answerer prompt differs from the released prompt")
    questions = base.questions[_BENCHMARK]
    if len(questions) != _ROW_COUNT or any(
        question.benchmark != _BENCHMARK
        or not isinstance(question.metadata.get("domain"), str)
        or not str(question.metadata["domain"]).strip()
        or not isinstance(question.metadata.get("topic"), str)
        or not str(question.metadata["topic"]).strip()
        for question in questions
    ):
        raise DataValidationError("AA official track requires 600 domain/topic-bound questions")
    neutral = [
        value
        for value in base.conditions
        if value.benchmark == _BENCHMARK
        and value.system_prompt_id == "P0-neutral"
        and value.steering_method == "M0"
    ]
    if len(neutral) != 1:
        raise DataValidationError("AA official track lacks one neutral M0 source condition")
    condition = replace(
        neutral[0],
        system_prompt_id=_PROMPT_ID,
        prompt_template_sha256=hashlib.sha256(prompt.text.encode("utf-8")).hexdigest(),
        comparison_group="aa-official-auxiliary",
    )
    schedule = _official_schedule(condition, questions)
    grader_manifest = verify_e1_grader_bundle(
        paths["grader_bundle"], expected_manifest_digest=expected_grader_manifest_digest
    )
    files = grader_manifest.get("files")
    if not isinstance(files, Mapping) or not isinstance(files.get("aa_config"), Mapping):
        raise FrozenArtifactError("AA official grader bundle inventory differs")
    grader_path = paths["grader_bundle"] / str(files["aa_config"]["path"])
    from mfh.evaluation.official import load_official_grader_spec

    grader = load_official_grader_spec(grader_path)
    if grader.benchmark != _BENCHMARK:
        raise FrozenArtifactError("AA official grader targets another benchmark")
    plan_body = {
        "schema_version": 1,
        "phase": "E1-AA-official-auxiliary",
        "track": "AA-Omniscience-Public-600 official answerer prompt and scoring",
        "runner_source_sha256": sha256_file(Path(__file__)),
        "e1_runner_source_sha256": sha256_file(Path(__file__).with_name("e1_vllm.py")),
        "study_protocol_digest": base.study.digest,
        "e1_contract_digest": base.contract.digest,
        "e1_completion_digest": completion.completion_digest,
        "e1_plan_identity": base.plan["plan_identity"],
        "model": {
            "name": base.model.name,
            "repository": base.model.repository,
            "revision": base.model.revision,
            "runtime": base.model.runtime.value,
            "quantization": base.model.quantization,
            "num_layers": base.model.num_layers,
        },
        "condition": condition.to_dict(),
        "prompt": {
            "prompt_id": prompt.prompt_id,
            "text": prompt.text,
            "text_sha256": hashlib.sha256(prompt.text.encode("utf-8")).hexdigest(),
            "permits_abstention": prompt.permits_abstention,
            "deployment_eligible": prompt.deployment_eligible,
        },
        "grader": {
            "schema_version": grader.schema_version,
            "benchmark": grader.benchmark,
            "source_repository": grader.source_repository,
            "source_revision": grader.source_revision,
            "source_artifact": grader.source_artifact,
            "temperature": grader.temperature,
            "reasoning_enabled": grader.reasoning_enabled,
            "prompt_template": grader.prompt_template,
            "label_mapping": {
                label: outcome.value for label, outcome in grader.label_mapping.items()
            },
            "maximum_attempts": grader.maximum_attempts,
            "failure_outcome": grader.failure_outcome.value,
            "bundle_manifest_digest": grader_manifest["manifest_digest"],
            "grader_digest": grader.digest,
            "grader_model": grader.grader_model,
            "grader_model_revision": grader.grader_model_revision,
            "prompt_sha256": grader.prompt_sha256,
            "source_artifact_sha256": grader.source_artifact_sha256,
        },
        "question_count": len(schedule),
        "question_fingerprints": {
            question.question_id: stable_hash(_question_body(question)) for question in questions
        },
        "questions": [_question_body(question) for question in questions],
        "schedule": {
            "ordering": "sha256-rank-randomized-across-questions-v1",
            "seed": _SEED,
            "schedule_digest": stable_hash(
                [[value.condition_id, question.question_id] for value, question in schedule]
            ),
        },
        "inference": {
            "temperature": 0,
            "sampling": False,
            "thinking_enabled": False,
            "max_new_tokens": base.max_new_tokens,
        },
        "input_hashes": {
            "reviewed_splits": sha256_path(paths["splits_directory"]),
            "grader_bundle": sha256_path(paths["grader_bundle"]),
            "model_config": sha256_file(paths["model_config"]),
            "snapshot_manifest": sha256_file(paths["snapshot_manifest"]),
            "runtime_config": sha256_file(paths["runtime_config"]),
            "prompt_config": sha256_file(paths["prompt_config"]),
            "inference_config": sha256_file(paths["inference_config"]),
            "study_config": sha256_file(paths["study_config"]),
        },
    }
    plan = MappingProxyType({**plan_body, "plan_identity": stable_hash(plan_body)})
    prepared = replace(
        base,
        prompts=MappingProxyType({_PROMPT_ID: prompt}),
        conditions=(condition,),
        plan=plan,
        schedule=schedule,
    )
    return AAOfficialContext(
        prepared=prepared,
        grader=grader,
        grader_manifest_digest=str(grader_manifest["manifest_digest"]),
        e1_completion_digest=completion.completion_digest,
        plan=plan,
    )


def _verify_inventory(directory: Path, *, output: bool) -> None:
    expected = _OUTPUT_FILES if output else _WORK_FILES
    if (
        directory.is_symlink()
        or not directory.is_dir()
        or {item.name for item in directory.iterdir()} - expected
        or any(item.is_symlink() or not item.is_file() for item in directory.iterdir())
    ):
        raise FrozenArtifactError("AA official track inventory differs")


def _checkpoint(
    context: AAOfficialContext,
    rows: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    sessions: Sequence[Mapping[str, Any]],
) -> str:
    return stable_hash(
        {
            "plan_identity": context.plan["plan_identity"],
            "records_completed": len(rows),
            "record_head": rows[-1]["record_digest"] if rows else None,
            "failures_recorded": len(failures),
            "failure_head": failures[-1]["failure_digest"] if failures else None,
            "session_events": len(sessions),
            "session_head": sessions[-1]["event_digest"] if sessions else None,
        }
    )


def _checkpoint_matches(
    context: AAOfficialContext,
    rows: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    sessions: Sequence[Mapping[str, Any]],
    expected: str,
) -> bool:
    candidates = {_checkpoint(context, rows, failures, sessions)}
    if rows:
        candidates.add(_checkpoint(context, rows[:-1], failures, sessions))
    if failures:
        candidates.add(_checkpoint(context, rows, failures[:-1], sessions))
    if sessions:
        candidates.add(_checkpoint(context, rows, failures, sessions[:-1]))
    return expected in candidates


def _emit_checkpoint(
    context: AAOfficialContext,
    rows: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    sessions: Sequence[Mapping[str, Any]],
    *,
    reason: str,
    checkpoint_file: Path,
) -> str:
    digest = _checkpoint(context, rows, failures, sessions)
    _atomic_json(
        checkpoint_file,
        {
            "schema_version": 1,
            "event": "aa-official-track-resume-checkpoint",
            "reason": reason,
            "plan_identity": context.plan["plan_identity"],
            "records_completed": len(rows),
            "records_expected": _ROW_COUNT,
            "failures_recorded": len(failures),
            "failure_head": failures[-1]["failure_digest"] if failures else None,
            "resume_checkpoint": digest,
        },
    )
    return digest


def _expected_grader_request(
    grader: OfficialGraderSpec, request: GradingRequest
) -> tuple[str, str]:
    prompt = render_grader_prompt(grader, request)
    payload = OpenRouterTransport(api_key="validation-only").request_payload(prompt, grader)
    return (
        hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest(),
        hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    )


def _validate_row(
    row: Mapping[str, Any],
    *,
    sequence: int,
    session_indices: set[int],
    previous_digest: str | None,
    previous_generation_digest: str | None,
    context: AAOfficialContext,
    runtime: _Runtime,
) -> None:
    expected = {
        "schema_version",
        "sequence",
        "plan_identity",
        "generation",
        "grader_request_fingerprint",
        "grader_fingerprint",
        "grader_raw_label",
        "outcome",
        "grader_receipts",
        "previous_record_digest",
        "record_digest",
    }
    body = dict(row)
    digest = body.pop("record_digest", None)
    generation = row.get("generation")
    receipts = row.get("grader_receipts")
    if (
        set(row) != expected
        or digest != stable_hash(body)
        or row.get("schema_version") != 1
        or row.get("sequence") != sequence
        or row.get("plan_identity") != context.plan["plan_identity"]
        or row.get("previous_record_digest") != previous_digest
        or not isinstance(generation, Mapping)
        or not isinstance(receipts, list)
        or not receipts
    ):
        raise DataValidationError("AA official row identity differs")
    if generation.get("previous_record_digest") != previous_generation_digest:
        raise DataValidationError("AA official generation chain differs")
    _validate_generation(
        generation,
        sequence=sequence,
        session_indices=session_indices,
        previous_digest=previous_generation_digest,
        prepared=context.prepared,
        runtime=runtime,
    )
    _condition, question = context.prepared.schedule[sequence]
    request = GradingRequest(
        question.question_id,
        question.text,
        question.aliases[0],
        str(generation["raw_output"]),
    )
    raw_label = row.get("grader_raw_label")
    try:
        outcome = context.grader.label_mapping[str(raw_label).strip()]
    except KeyError as exc:
        raise DataValidationError("AA official row has an unknown grade") from exc
    if (
        row.get("grader_request_fingerprint") != request.digest
        or row.get("grader_fingerprint") != context.grader.digest
        or row.get("outcome") != outcome.value
        or outcome is Outcome.UNSCORABLE
    ):
        raise DataValidationError("AA official grade identity differs")
    request_sha256, prompt_sha256 = _expected_grader_request(context.grader, request)
    route = route_for_grader(context.grader)
    for index, receipt in enumerate(receipts, start=1):
        if not isinstance(receipt, Mapping):
            raise DataValidationError("AA official grader receipt is invalid")
        success = index == len(receipts)
        validate_openrouter_attempt_receipt(
            receipt,
            route=route,
            request_sha256=request_sha256,
            prompt_sha256=prompt_sha256,
            attempt=index,
            expect_success=success,
            expected_content=str(raw_label) if success else None,
            expect_retry=not success,
        )


def _load_rows(
    directory: Path,
    *,
    context: AAOfficialContext,
    runtime: _Runtime,
    session_indices: set[int],
) -> list[dict[str, Any]]:
    rows = _read_jsonl(directory / "records.jsonl", "AA official records")
    if len(rows) > _ROW_COUNT:
        raise DataValidationError("AA official record count exceeds Public-600")
    previous: str | None = None
    previous_generation: str | None = None
    for sequence, row in enumerate(rows):
        _validate_row(
            row,
            sequence=sequence,
            session_indices=session_indices,
            previous_digest=previous,
            previous_generation_digest=previous_generation,
            context=context,
            runtime=runtime,
        )
        previous = str(row["record_digest"])
        previous_generation = str(row["generation"]["record_digest"])
    return rows


def _validate_failure(
    row: Mapping[str, Any],
    *,
    failure_index: int,
    records_completed: int,
    session_indices: set[int],
    previous_failure_digest: str | None,
    previous_generation_digest: str | None,
    context: AAOfficialContext,
    runtime: _Runtime,
) -> None:
    expected = {
        "schema_version",
        "failure_index",
        "sequence",
        "plan_identity",
        "generation",
        "grader_request_fingerprint",
        "grader_fingerprint",
        "grader_raw_label",
        "grader_attempts",
        "outcome",
        "error",
        "grader_receipts",
        "previous_failure_digest",
        "failure_digest",
    }
    body = dict(row)
    digest = body.pop("failure_digest", None)
    generation = row.get("generation")
    receipts = row.get("grader_receipts")
    if (
        set(row) != expected
        or digest != stable_hash(body)
        or row.get("schema_version") != 1
        or row.get("failure_index") != failure_index
        or row.get("sequence") != records_completed
        or row.get("plan_identity") != context.plan["plan_identity"]
        or row.get("previous_failure_digest") != previous_failure_digest
        or not isinstance(generation, Mapping)
        or generation.get("previous_record_digest") != previous_generation_digest
        or not isinstance(receipts, list)
        or not receipts
        or type(row.get("grader_attempts")) is not int
        or row.get("grader_attempts") != len(receipts)
        or len(receipts) > context.grader.maximum_attempts
        or row.get("outcome") != Outcome.UNSCORABLE.value
        or not isinstance(row.get("error"), str)
        or not str(row["error"]).strip()
        or not isinstance(row.get("grader_raw_label"), str)
    ):
        raise DataValidationError("AA official failure evidence identity differs")
    _validate_generation(
        generation,
        sequence=records_completed,
        session_indices=session_indices,
        previous_digest=previous_generation_digest,
        prepared=context.prepared,
        runtime=runtime,
    )
    _condition, question = context.prepared.schedule[records_completed]
    request = GradingRequest(
        question.question_id,
        question.text,
        question.aliases[0],
        str(generation["raw_output"]),
    )
    if (
        row.get("grader_request_fingerprint") != request.digest
        or row.get("grader_fingerprint") != context.grader.digest
    ):
        raise DataValidationError("AA official failure grade identity differs")
    request_sha256, prompt_sha256 = _expected_grader_request(context.grader, request)
    route = route_for_grader(context.grader)

    def accepted(content: str) -> bool:
        return content.strip() in context.grader.label_mapping

    for attempt, receipt in enumerate(receipts, start=1):
        if not isinstance(receipt, Mapping):
            raise DataValidationError("AA official failure receipt is invalid")
        validate_openrouter_attempt_receipt(
            receipt,
            route=route,
            request_sha256=request_sha256,
            prompt_sha256=prompt_sha256,
            attempt=attempt,
            expect_success=False,
            accepted_success_content=accepted,
            expect_retry=attempt < len(receipts),
        )


def _load_failures(
    directory: Path,
    *,
    context: AAOfficialContext,
    runtime: _Runtime,
    session_indices: set[int],
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    failures = _read_jsonl(directory / "failures.jsonl", "AA official failures")
    previous: str | None = None
    for failure_index, row in enumerate(failures):
        sequence = row.get("sequence")
        if type(sequence) is not int or not 0 <= sequence <= len(rows):
            raise DataValidationError("AA official failure sequence is invalid")
        generation_predecessor = (
            str(rows[sequence - 1]["generation"]["record_digest"]) if sequence > 0 else None
        )
        _validate_failure(
            row,
            failure_index=failure_index,
            records_completed=sequence,
            session_indices=session_indices,
            previous_failure_digest=previous,
            previous_generation_digest=generation_predecessor,
            context=context,
            runtime=runtime,
        )
        previous = str(row["failure_digest"])
    return failures


def prepare_aa_official_track(
    work_directory: str | Path,
    *,
    expected_splits_manifest_digest: str,
    expected_grader_manifest_digest: str,
    **raw_paths: str | Path,
) -> Mapping[str, Any]:
    paths = _paths(**raw_paths)
    context = _context(
        **paths,
        expected_splits_manifest_digest=expected_splits_manifest_digest,
        expected_grader_manifest_digest=expected_grader_manifest_digest,
    )
    work = validate_active_study_artifact_paths(
        {"AA official work": work_directory, "E1 ledger": paths["ledger_directory"]}
    )["AA official work"]
    if work.exists() or work.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite AA official work: {work}")
    work.mkdir(parents=True)
    _write_json_once(work / "plan.json", dict(context.plan))
    return MappingProxyType(
        {
            "prepared": True,
            "plan_identity": context.plan["plan_identity"],
            "e1_completion_digest": context.e1_completion_digest,
            "records_expected": _ROW_COUNT,
            "work_directory": str(work),
        }
    )


def run_aa_official_track(
    work_directory: str | Path,
    *,
    expected_splits_manifest_digest: str,
    expected_grader_manifest_digest: str,
    api_key: str,
    checkpoint_file: str | Path,
    expected_resume_checkpoint: str | None = None,
    request_budget: int | None = None,
    runtime_factory: Callable[[Any, Path], _Runtime] | None = None,
    transport_factory: Callable[[], OpenRouterTransport] | None = None,
    **raw_paths: str | Path,
) -> Mapping[str, Any]:
    if request_budget is not None and (type(request_budget) is not int or request_budget <= 0):
        raise ConfigurationError("AA official request budget must be a positive integer")
    if expected_resume_checkpoint is not None and (
        len(expected_resume_checkpoint) != 64
        or any(value not in "0123456789abcdef" for value in expected_resume_checkpoint)
    ):
        raise ConfigurationError("AA official resume checkpoint must be a SHA-256 digest")
    paths = _paths(**raw_paths)
    context = _context(
        **paths,
        expected_splits_manifest_digest=expected_splits_manifest_digest,
        expected_grader_manifest_digest=expected_grader_manifest_digest,
    )
    normalized = validate_active_study_artifact_paths(
        {
            "AA official work": work_directory,
            "AA official checkpoint": checkpoint_file,
            "E1 ledger": paths["ledger_directory"],
        }
    )
    work = normalized["AA official work"]
    checkpoint_path = normalized["AA official checkpoint"]
    if checkpoint_path == work or checkpoint_path.is_relative_to(work):
        raise ConfigurationError("AA official checkpoint must remain outside its work directory")
    _verify_inventory(work, output=False)
    if _read_json(work / "plan.json", "AA official plan") != dict(context.plan):
        raise FrozenArtifactError("AA official plan differs from live frozen inputs")
    has_execution = any(
        (work / name).exists() for name in ("records.jsonl", "failures.jsonl", "sessions.jsonl")
    )
    if has_execution != (expected_resume_checkpoint is not None):
        raise ConfigurationError(
            "existing AA official execution requires its external resume checkpoint"
            if has_execution
            else "an AA official resume checkpoint cannot initialize execution"
        )
    factory = runtime_factory or (
        lambda model, snapshot: VllmRuntime.from_spec(model, snapshot_path=snapshot, seed=_SEED)
    )
    runtime: _Runtime | None = None
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    session_index = -1
    status = "error"
    pending_error: str | None = None
    active_error: BaseException | None = None
    try:
        runtime = factory(context.prepared.model, context.prepared.snapshot)
        _validate_runtime_identity(context.prepared, runtime.runtime_identity())
        sessions = _validate_sessions(
            work / "sessions.jsonl",
            plan_identity=str(context.plan["plan_identity"]),
            allow_open=has_execution,
        )
        starts = {int(row["session_index"]) for row in sessions if row["event"] == "start"}
        ends = {int(row["session_index"]) for row in sessions if row["event"] == "end"}
        session_indices = starts | ends
        rows = _load_rows(work, context=context, runtime=runtime, session_indices=session_indices)
        failures = _load_failures(
            work,
            context=context,
            runtime=runtime,
            session_indices=session_indices,
            rows=rows,
        )
        if has_execution:
            expected = str(expected_resume_checkpoint)
            if not _checkpoint_matches(context, rows, failures, sessions, expected):
                raise DataValidationError("AA official resume checkpoint differs")
            if _checkpoint(context, rows, failures, sessions) != expected:
                _emit_checkpoint(
                    context,
                    rows,
                    failures,
                    sessions,
                    reason="crash-gap-catch-up",
                    checkpoint_file=checkpoint_path,
                )
        for interrupted in sorted(starts - ends):
            _session_event(
                work / "sessions.jsonl",
                event="end",
                session_index=interrupted,
                plan_identity=str(context.plan["plan_identity"]),
                details={
                    "records_at_end": len(rows),
                    "failures_at_end": len(failures),
                    "status": "interrupted-recovered",
                },
            )
        sessions = _validate_sessions(
            work / "sessions.jsonl",
            plan_identity=str(context.plan["plan_identity"]),
            allow_open=False,
        )
        if len(rows) == _ROW_COUNT:
            return MappingProxyType(
                {
                    "complete": True,
                    "status": "complete",
                    "records_completed": _ROW_COUNT,
                    "records_expected": _ROW_COUNT,
                    "plan_identity": context.plan["plan_identity"],
                    "failures_recorded": len(failures),
                    "resume_checkpoint": _checkpoint(context, rows, failures, sessions),
                }
            )
        session_index = max((int(row["session_index"]) for row in sessions), default=-1) + 1
        _session_event(
            work / "sessions.jsonl",
            event="start",
            session_index=session_index,
            plan_identity=str(context.plan["plan_identity"]),
            details={"records_at_start": len(rows), "runtime_identity": runtime.runtime_identity()},
        )
        session_indices.add(session_index)
        sessions = _read_jsonl(work / "sessions.jsonl", "AA official sessions")
        _emit_checkpoint(
            context,
            rows,
            failures,
            sessions,
            reason="session-start",
            checkpoint_file=checkpoint_path,
        )
        transport = (
            transport_factory()
            if transport_factory is not None
            else OpenRouterTransport(api_key=api_key)
        )
        new_rows = 0
        while len(rows) < _ROW_COUNT and (request_budget is None or new_rows < request_budget):
            sequence = len(rows)
            condition, question = context.prepared.schedule[sequence]
            rendered = runtime.render_prompt(
                context.prepared.prompts[_PROMPT_ID],
                question.text,
                metadata=dict(question.metadata),
            )
            generated = runtime.generate(rendered, max_new_tokens=context.prepared.max_new_tokens)
            request = GradingRequest(
                question.question_id,
                question.text,
                question.aliases[0],
                generated.text,
            )
            receipt_offset = len(transport.receipts)
            grade = run_openrouter_grader(context.grader, request, transport)
            receipts = [value.to_dict() for value in transport.receipts[receipt_offset:]]
            generation_body = _generation_body(
                sequence=sequence,
                session_index=session_index,
                prepared=context.prepared,
                condition=condition,
                question=question,
                rendered=rendered,
                generation=generated,
                previous_digest=(str(rows[-1]["generation"]["record_digest"]) if rows else None),
            )
            generation = {
                **generation_body,
                "record_digest": stable_hash(generation_body),
            }
            if grade.outcome is Outcome.UNSCORABLE:
                pending_error = grade.error or "unscorable official AA grade"
                failure_body = {
                    "schema_version": 1,
                    "failure_index": len(failures),
                    "sequence": sequence,
                    "plan_identity": context.plan["plan_identity"],
                    "generation": generation,
                    "grader_request_fingerprint": grade.request_fingerprint,
                    "grader_fingerprint": grade.grader_fingerprint,
                    "grader_raw_label": grade.raw_response,
                    "grader_attempts": grade.attempts,
                    "outcome": grade.outcome.value,
                    "error": pending_error,
                    "grader_receipts": receipts,
                    "previous_failure_digest": (
                        str(failures[-1]["failure_digest"]) if failures else None
                    ),
                }
                failure = {
                    **failure_body,
                    "failure_digest": stable_hash(failure_body),
                }
                if api_key and api_key in json.dumps(failure, ensure_ascii=False):
                    raise DataValidationError("refusing to persist an OpenRouter secret")
                _validate_failure(
                    failure,
                    failure_index=len(failures),
                    records_completed=sequence,
                    session_indices=session_indices,
                    previous_failure_digest=(
                        str(failures[-1]["failure_digest"]) if failures else None
                    ),
                    previous_generation_digest=(
                        str(rows[-1]["generation"]["record_digest"]) if rows else None
                    ),
                    context=context,
                    runtime=runtime,
                )
                _append_jsonl(work / "failures.jsonl", failure)
                failures.append(failure)
                sessions = _read_jsonl(work / "sessions.jsonl", "AA official sessions")
                _emit_checkpoint(
                    context,
                    rows,
                    failures,
                    sessions,
                    reason="unscorable-failure-appended",
                    checkpoint_file=checkpoint_path,
                )
                status = "pending-provider-retry"
                break
            body = {
                "schema_version": 1,
                "sequence": sequence,
                "plan_identity": context.plan["plan_identity"],
                "generation": generation,
                "grader_request_fingerprint": grade.request_fingerprint,
                "grader_fingerprint": grade.grader_fingerprint,
                "grader_raw_label": grade.raw_response,
                "outcome": grade.outcome.value,
                "grader_receipts": receipts,
                "previous_record_digest": (str(rows[-1]["record_digest"]) if rows else None),
            }
            row = {**body, "record_digest": stable_hash(body)}
            if api_key and api_key in json.dumps(row, ensure_ascii=False):
                raise DataValidationError("refusing to persist an OpenRouter secret")
            _validate_row(
                row,
                sequence=sequence,
                session_indices=session_indices,
                previous_digest=(str(rows[-1]["record_digest"]) if rows else None),
                previous_generation_digest=(
                    str(rows[-1]["generation"]["record_digest"]) if rows else None
                ),
                context=context,
                runtime=runtime,
            )
            _append_jsonl(work / "records.jsonl", row)
            rows.append(row)
            new_rows += 1
            sessions = _read_jsonl(work / "sessions.jsonl", "AA official sessions")
            _emit_checkpoint(
                context,
                rows,
                failures,
                sessions,
                reason="record-appended",
                checkpoint_file=checkpoint_path,
            )
        if status == "error":
            status = "complete" if len(rows) == _ROW_COUNT else "partial"
    except BaseException as exc:
        active_error = exc
    finally:
        if session_index >= 0:
            try:
                _session_event(
                    work / "sessions.jsonl",
                    event="end",
                    session_index=session_index,
                    plan_identity=str(context.plan["plan_identity"]),
                    details={
                        "records_at_end": len(rows),
                        "failures_at_end": len(failures),
                        "status": status,
                    },
                )
                sessions = _validate_sessions(
                    work / "sessions.jsonl",
                    plan_identity=str(context.plan["plan_identity"]),
                    allow_open=False,
                )
                _emit_checkpoint(
                    context,
                    rows,
                    failures,
                    sessions,
                    reason="session-end",
                    checkpoint_file=checkpoint_path,
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
    if active_error is not None:
        raise active_error
    sessions = _validate_sessions(
        work / "sessions.jsonl",
        plan_identity=str(context.plan["plan_identity"]),
        allow_open=False,
    )
    return MappingProxyType(
        {
            "complete": len(rows) == _ROW_COUNT,
            "status": status,
            "records_completed": len(rows),
            "records_expected": _ROW_COUNT,
            "failures_recorded": len(failures),
            "pending_error": pending_error,
            "plan_identity": context.plan["plan_identity"],
            "resume_checkpoint": _checkpoint(context, rows, failures, sessions),
        }
    )


def _metrics_and_comparison(
    rows: Sequence[Mapping[str, Any]],
    *,
    context: AAOfficialContext,
    neutral_records: Sequence[GenerationRecord],
) -> tuple[dict[str, Any], dict[str, Any]]:
    outcomes = tuple(Outcome(str(row["outcome"])) for row in rows)
    unified = metric_bundle(outcomes, partial_credit=0.5).to_dict()
    official = asdict(aa_official_metrics(outcomes))
    metrics = {
        "schema_version": 1,
        "benchmark": _BENCHMARK,
        "track": "official",
        "prompt_id": _PROMPT_ID,
        "question_count": len(rows),
        "official_metrics": official,
        "unified_metrics": unified,
    }
    neutral_by_question = {
        record.question_id: record
        for record in neutral_records
        if record.benchmark == _BENCHMARK
        and record.system_prompt_id == "P0-neutral"
        and record.steering_method == "M0"
    }
    official_by_question = {
        context.prepared.schedule[index][1].question_id: Outcome(str(row["outcome"]))
        for index, row in enumerate(rows)
    }
    if (
        set(neutral_by_question) != set(official_by_question)
        or len(neutral_by_question) != _ROW_COUNT
    ):
        raise FrozenArtifactError("AA official/neutral comparison lacks exact paired questions")
    transitions = Counter(
        f"{neutral_by_question[question_id].outcome.value}->{outcome.value}"
        for question_id, outcome in official_by_question.items()
    )
    neutral_outcomes = tuple(
        neutral_by_question[question.question_id].outcome
        for _condition, question in context.prepared.schedule
    )
    neutral_unified = metric_bundle(neutral_outcomes, partial_credit=0.5).to_dict()
    neutral_official = asdict(aa_official_metrics(neutral_outcomes))

    def number(value: object, *, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise FrozenArtifactError(f"AA comparison metric is invalid: {name}")
        return float(value)

    comparison = {
        "schema_version": 1,
        "benchmark": _BENCHMARK,
        "comparison": "P-AA-official-vs-P0-neutral within M0",
        "paired_question_count": _ROW_COUNT,
        "neutral": {
            "prompt_id": "P0-neutral",
            "official_metrics": neutral_official,
            "unified_metrics": neutral_unified,
        },
        "official": {
            "prompt_id": _PROMPT_ID,
            "official_metrics": official,
            "unified_metrics": unified,
        },
        "deltas": {
            "omniscience_index": official["omniscience_index"]
            - neutral_official["omniscience_index"],
            "coverage": number(unified["coverage"], name="official coverage")
            - number(neutral_unified["coverage"], name="neutral coverage"),
            "hallucination_risk": (
                None
                if unified["hallucination_risk"] is None
                or neutral_unified["hallucination_risk"] is None
                else number(unified["hallucination_risk"], name="official risk")
                - number(neutral_unified["hallucination_risk"], name="neutral risk")
            ),
        },
        "transition_counts": dict(sorted(transitions.items())),
        "leaderboard_comparability": {
            "official_track": True,
            "neutral_controlled_track": False,
        },
    }
    return metrics, comparison


def _replay_terminal(
    directory: Path,
    *,
    context: AAOfficialContext,
    neutral_records: Sequence[GenerationRecord],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
]:
    renderer = _tokenizer_renderer(context.prepared)
    try:
        sessions = _validate_sessions(
            directory / "sessions.jsonl",
            plan_identity=str(context.plan["plan_identity"]),
            allow_open=False,
        )
        indices = {int(row["session_index"]) for row in sessions}
        rows = _load_rows(
            directory,
            context=context,
            runtime=renderer,
            session_indices=indices,
        )
        failures = _load_failures(
            directory,
            context=context,
            runtime=renderer,
            session_indices=indices,
            rows=rows,
        )
    finally:
        renderer.close()
    if len(rows) != _ROW_COUNT:
        raise FrozenArtifactError("AA official terminal artifact is incomplete")
    metrics, comparison = _metrics_and_comparison(
        rows, context=context, neutral_records=neutral_records
    )
    return rows, failures, metrics, comparison, sessions


def finalize_aa_official_track(
    output_directory: str | Path,
    *,
    work_directory: str | Path,
    expected_splits_manifest_digest: str,
    expected_grader_manifest_digest: str,
    **raw_paths: str | Path,
) -> Mapping[str, Any]:
    paths = _paths(**raw_paths)
    context = _context(
        **paths,
        expected_splits_manifest_digest=expected_splits_manifest_digest,
        expected_grader_manifest_digest=expected_grader_manifest_digest,
    )
    work = Path(work_directory).absolute()
    _verify_inventory(work, output=False)
    if _read_json(work / "plan.json", "AA official plan") != dict(context.plan):
        raise FrozenArtifactError("AA official plan differs before finalization")
    ledger = PhaseRunLedger.open(paths["ledger_directory"], study=context.prepared.study)
    neutral_records = tuple(ledger.records())
    rows, failures, metrics, comparison, sessions = _replay_terminal(
        work, context=context, neutral_records=neutral_records
    )
    peak = max(
        [int(row["generation"]["peak_memory_bytes"]) for row in rows]
        + [int(row["generation"]["peak_memory_bytes"]) for row in failures]
    )
    if peak > _MEMORY_BYTES:
        raise FrozenArtifactError("AA official track exceeded the 40 GB A100 envelope")
    output = Path(output_directory).absolute()
    if output.exists() or output.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite AA official result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        shutil.copyfile(work / "plan.json", stage / "plan.json")
        shutil.copyfile(work / "records.jsonl", stage / "records.jsonl")
        if (work / "failures.jsonl").is_file():
            shutil.copyfile(work / "failures.jsonl", stage / "failures.jsonl")
        else:
            (stage / "failures.jsonl").write_text("", encoding="utf-8")
        shutil.copyfile(work / "sessions.jsonl", stage / "sessions.jsonl")
        for name, value in {
            "metrics.json": metrics,
            "official-vs-neutral.json": comparison,
        }.items():
            (stage / name).write_text(
                json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        manifest_body = {
            "schema_version": 1,
            "phase": "E1-AA-official-auxiliary",
            "status": "complete",
            "scientific_eligible": True,
            "runner_source_sha256": sha256_file(Path(__file__)),
            "plan_identity": context.plan["plan_identity"],
            "e1_completion_digest": context.e1_completion_digest,
            "record_count": len(rows),
            "record_chain_head": rows[-1]["record_digest"],
            "record_set_digest": stable_hash([row["record_digest"] for row in rows]),
            "failure_count": len(failures),
            "failure_chain_head": (failures[-1]["failure_digest"] if failures else None),
            "failure_set_digest": stable_hash([row["failure_digest"] for row in failures]),
            "session_chain_head": sessions[-1]["event_digest"],
            "session_set_digest": stable_hash([row["event_digest"] for row in sessions]),
            "maximum_peak_memory_bytes": peak,
            "metrics_digest": stable_hash(metrics),
            "official_vs_neutral_digest": stable_hash(comparison),
        }
        (stage / "manifest.json").write_text(
            json.dumps(
                {**manifest_body, "manifest_digest": stable_hash(manifest_body)},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    manifest = _read_json(output / "manifest.json", "AA official manifest")
    return verify_aa_official_track(
        output,
        expected_manifest_digest=str(manifest["manifest_digest"]),
        expected_splits_manifest_digest=expected_splits_manifest_digest,
        expected_grader_manifest_digest=expected_grader_manifest_digest,
        **raw_paths,
    )


def verify_aa_official_track(
    directory: str | Path,
    *,
    expected_manifest_digest: str,
    expected_splits_manifest_digest: str,
    expected_grader_manifest_digest: str,
    **raw_paths: str | Path,
) -> Mapping[str, Any]:
    paths = _paths(**raw_paths)
    context = _context(
        **paths,
        expected_splits_manifest_digest=expected_splits_manifest_digest,
        expected_grader_manifest_digest=expected_grader_manifest_digest,
    )
    source = Path(directory).absolute()
    _verify_inventory(source, output=True)
    if {item.name for item in source.iterdir()} != _OUTPUT_FILES:
        raise FrozenArtifactError("AA official terminal inventory is incomplete")
    if _read_json(source / "plan.json", "AA official plan") != dict(context.plan):
        raise FrozenArtifactError("AA official terminal plan differs")
    ledger = PhaseRunLedger.open(paths["ledger_directory"], study=context.prepared.study)
    rows, failures, metrics, comparison, sessions = _replay_terminal(
        source,
        context=context,
        neutral_records=tuple(ledger.records()),
    )
    observed_metrics = _read_json(source / "metrics.json", "AA official metrics")
    observed_comparison = _read_json(source / "official-vs-neutral.json", "AA official comparison")
    manifest = _read_json(source / "manifest.json", "AA official manifest")
    body = dict(manifest)
    digest = body.pop("manifest_digest", None)
    peak = max(
        [int(row["generation"]["peak_memory_bytes"]) for row in rows]
        + [int(row["generation"]["peak_memory_bytes"]) for row in failures]
    )
    expected_body = {
        "schema_version": 1,
        "phase": "E1-AA-official-auxiliary",
        "status": "complete",
        "scientific_eligible": True,
        "runner_source_sha256": sha256_file(Path(__file__)),
        "plan_identity": context.plan["plan_identity"],
        "e1_completion_digest": context.e1_completion_digest,
        "record_count": len(rows),
        "record_chain_head": rows[-1]["record_digest"],
        "record_set_digest": stable_hash([row["record_digest"] for row in rows]),
        "failure_count": len(failures),
        "failure_chain_head": failures[-1]["failure_digest"] if failures else None,
        "failure_set_digest": stable_hash([row["failure_digest"] for row in failures]),
        "session_chain_head": sessions[-1]["event_digest"],
        "session_set_digest": stable_hash([row["event_digest"] for row in sessions]),
        "maximum_peak_memory_bytes": peak,
        "metrics_digest": stable_hash(metrics),
        "official_vs_neutral_digest": stable_hash(comparison),
    }
    if (
        digest != expected_manifest_digest
        or digest != stable_hash(body)
        or body != expected_body
        or observed_metrics != metrics
        or observed_comparison != comparison
        or peak > _MEMORY_BYTES
    ):
        raise FrozenArtifactError("AA official terminal artifact differs from exact replay")
    return MappingProxyType(
        {
            "valid": True,
            "manifest_digest": digest,
            "record_count": len(rows),
            "failure_count": len(failures),
            "maximum_peak_memory_bytes": peak,
            "metrics": MappingProxyType(metrics),
            "official_vs_neutral": MappingProxyType(comparison),
        }
    )


def _portable_question(value: object) -> Question:
    expected = {
        "question_id",
        "benchmark",
        "text",
        "aliases",
        "split",
        "entities",
        "metadata",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise FrozenArtifactError("AA official portable question schema differs")
    aliases = value["aliases"]
    entities = value["entities"]
    metadata = value["metadata"]
    if (
        not isinstance(aliases, list)
        or not isinstance(entities, list)
        or not isinstance(metadata, Mapping)
    ):
        raise FrozenArtifactError("AA official portable question is invalid")
    try:
        question = Question(
            question_id=str(value["question_id"]),
            benchmark=str(value["benchmark"]),
            text=str(value["text"]),
            aliases=tuple(str(item) for item in aliases),
            split=str(value["split"]) if value["split"] is not None else None,
            entities=tuple(str(item) for item in entities),
            metadata=dict(metadata),
        )
    except (TypeError, ValueError, DataValidationError) as exc:
        raise FrozenArtifactError(f"AA official portable question is invalid: {exc}") from exc
    if (
        question.benchmark != _BENCHMARK
        or not isinstance(question.metadata.get("domain"), str)
        or not str(question.metadata["domain"]).strip()
        or not isinstance(question.metadata.get("topic"), str)
        or not str(question.metadata["topic"]).strip()
    ):
        raise FrozenArtifactError("AA official portable question lacks domain/topic identity")
    return question


def _portable_grader(value: object) -> OfficialGraderSpec:
    expected = {
        "schema_version",
        "benchmark",
        "source_repository",
        "source_revision",
        "source_artifact",
        "source_artifact_sha256",
        "grader_model",
        "grader_model_revision",
        "temperature",
        "reasoning_enabled",
        "prompt_template",
        "prompt_sha256",
        "label_mapping",
        "maximum_attempts",
        "failure_outcome",
        "bundle_manifest_digest",
        "grader_digest",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise FrozenArtifactError("AA official portable grader schema differs")
    labels = value["label_mapping"]
    if (
        not isinstance(labels, Mapping)
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or isinstance(value["temperature"], bool)
        or not isinstance(value["temperature"], int | float)
        or not math.isfinite(float(value["temperature"]))
        or type(value["reasoning_enabled"]) is not bool
        or type(value["maximum_attempts"]) is not int
        or value["maximum_attempts"] <= 0
        or value["failure_outcome"] != Outcome.UNSCORABLE.value
        or not _is_sha256(value["bundle_manifest_digest"])
        or not _is_sha256(value["grader_digest"])
    ):
        raise FrozenArtifactError("AA official portable grader labels are invalid")
    try:
        grader = OfficialGraderSpec(
            schema_version=int(value["schema_version"]),
            benchmark=str(value["benchmark"]),
            source_repository=str(value["source_repository"]),
            source_revision=str(value["source_revision"]),
            source_artifact=str(value["source_artifact"]),
            source_artifact_sha256=str(value["source_artifact_sha256"]),
            grader_model=str(value["grader_model"]),
            grader_model_revision=str(value["grader_model_revision"]),
            temperature=float(value["temperature"]),
            reasoning_enabled=value["reasoning_enabled"],
            prompt_template=str(value["prompt_template"]),
            prompt_sha256=str(value["prompt_sha256"]),
            label_mapping={str(key): Outcome(str(item)) for key, item in labels.items()},
            maximum_attempts=int(value["maximum_attempts"]),
            failure_outcome=Outcome(str(value["failure_outcome"])),
        )
    except (TypeError, ValueError, DataValidationError) as exc:
        raise FrozenArtifactError(f"AA official portable grader is invalid: {exc}") from exc
    if value["grader_digest"] != grader.digest:
        raise FrozenArtifactError("AA official portable grader digest differs")
    return grader


def _portable_plan_semantics(plan: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "phase",
        "track",
        "runner_source_sha256",
        "e1_runner_source_sha256",
        "study_protocol_digest",
        "e1_contract_digest",
        "e1_completion_digest",
        "e1_plan_identity",
        "model",
        "condition",
        "prompt",
        "grader",
        "question_count",
        "question_fingerprints",
        "questions",
        "schedule",
        "inference",
        "input_hashes",
    }
    model = plan.get("model")
    prompt = plan.get("prompt")
    inference = plan.get("inference")
    input_hashes = plan.get("input_hashes")
    if (
        set(plan) != expected
        or plan.get("track") != "AA-Omniscience-Public-600 official answerer prompt and scoring"
        or any(
            not _is_sha256(plan.get(name))
            for name in (
                "runner_source_sha256",
                "e1_runner_source_sha256",
                "study_protocol_digest",
                "e1_contract_digest",
                "e1_completion_digest",
                "e1_plan_identity",
            )
        )
        or not isinstance(model, Mapping)
        or set(model)
        != {
            "name",
            "repository",
            "revision",
            "runtime",
            "quantization",
            "num_layers",
        }
        or model.get("runtime") != "vllm"
        or not isinstance(model.get("repository"), str)
        or "/" not in model["repository"]
        or not isinstance(model.get("revision"), str)
        or len(model["revision"]) != 40
        or any(character not in "0123456789abcdef" for character in model["revision"])
        or type(model.get("num_layers")) is not int
        or model["num_layers"] <= 0
        or not isinstance(prompt, Mapping)
        or set(prompt)
        != {"prompt_id", "text", "text_sha256", "permits_abstention", "deployment_eligible"}
        or prompt.get("prompt_id") != _PROMPT_ID
        or prompt.get("text") != _OFFICIAL_PROMPT_TEXT
        or prompt.get("text_sha256")
        != hashlib.sha256(_OFFICIAL_PROMPT_TEXT.encode("utf-8")).hexdigest()
        or prompt.get("permits_abstention") is not True
        or prompt.get("deployment_eligible") is not True
        or not isinstance(inference, Mapping)
        or set(inference) != {"temperature", "sampling", "thinking_enabled", "max_new_tokens"}
        or inference.get("temperature") != 0
        or inference.get("sampling") is not False
        or inference.get("thinking_enabled") is not False
        or type(inference.get("max_new_tokens")) is not int
        or inference["max_new_tokens"] <= 0
        or not isinstance(input_hashes, Mapping)
        or set(input_hashes)
        != {
            "reviewed_splits",
            "grader_bundle",
            "model_config",
            "snapshot_manifest",
            "runtime_config",
            "prompt_config",
            "inference_config",
            "study_config",
        }
        or any(not _is_sha256(value) for value in input_hashes.values())
    ):
        raise FrozenArtifactError("AA official portable plan semantics differ")


def _portable_failure_generation(
    generation: object,
    *,
    sequence: int,
    plan_identity: str,
    condition: EvaluationCondition,
    question: Question,
    max_new_tokens: int,
    previous_generation_digest: str | None,
    session_indices: set[int],
    generation_fields: set[str],
) -> tuple[str, int]:
    if not isinstance(generation, Mapping) or set(generation) != generation_fields:
        raise FrozenArtifactError("AA official portable failure generation schema differs")
    generation_body = dict(generation)
    digest = generation_body.pop("record_digest", None)
    raw_output = generation.get("raw_output")
    tokens = generation.get("token_ids")
    if (
        not isinstance(digest, str)
        or digest != stable_hash(generation_body)
        or generation.get("sequence") != sequence
        or generation.get("session_index") not in session_indices
        or generation.get("plan_identity") != plan_identity
        or generation.get("condition_id") != condition.condition_id
        or generation.get("question_id") != question.question_id
        or generation.get("benchmark") != _BENCHMARK
        or generation.get("partition") != condition.partition
        or generation.get("prompt_id") != _PROMPT_ID
        or generation.get("previous_record_digest") != previous_generation_digest
        or not _is_sha256(generation.get("rendered_prompt_sha256"))
        or not _is_sha256(generation.get("rendered_token_ids_sha256"))
        or generation.get("request_digest")
        != stable_hash(
            {
                "plan_identity": plan_identity,
                "condition_id": condition.condition_id,
                "question_id": question.question_id,
                "rendered_prompt_sha256": generation.get("rendered_prompt_sha256"),
                "rendered_token_ids_sha256": generation.get("rendered_token_ids_sha256"),
                "max_new_tokens": max_new_tokens,
                "sampling": False,
                "thinking_enabled": False,
                "seed": _SEED,
            }
        )
        or not isinstance(raw_output, str)
        or generation.get("raw_output_sha256")
        != hashlib.sha256(raw_output.encode("utf-8")).hexdigest()
        or generation.get("raw_output_stable_hash") != stable_hash(raw_output)
        or not isinstance(tokens, list)
        or any(type(token) is not int or token < 0 for token in tokens)
        or generation.get("output_tokens") != len(tokens)
        or not 1 <= len(tokens) <= max_new_tokens
        or generation.get("stopping_token_id") != tokens[-1]
        or not isinstance(generation.get("stop_type"), str)
        or not str(generation["stop_type"]).strip()
    ):
        raise FrozenArtifactError("AA official portable failure generation differs")
    for name in (
        "input_tokens",
        "peak_memory_bytes",
        "active_memory_bytes",
        "cache_memory_bytes",
    ):
        if type(generation.get(name)) is not int or int(generation[name]) < 0:
            raise FrozenArtifactError("AA official portable failure memory metric is invalid")
    for name in (
        "latency_seconds",
        "prompt_tokens_per_second",
        "generation_tokens_per_second",
    ):
        value = generation.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise FrozenArtifactError("AA official portable failure timing metric is invalid")
    return raw_output, int(generation["peak_memory_bytes"])


def load_aa_official_analysis(
    directory: str | Path,
    *,
    expected_manifest_digest: str,
    expected_e1_completion_digest: str,
) -> Mapping[str, Any]:
    """Replay the portable terminal evidence used by final reporting."""

    source = Path(directory)
    _verify_inventory(source, output=True)
    if {item.name for item in source.iterdir()} != _OUTPUT_FILES:
        raise FrozenArtifactError("AA official portable inventory is incomplete")
    manifest = _read_json(source / "manifest.json", "AA official manifest")
    body = dict(manifest)
    manifest_digest = body.pop("manifest_digest", None)
    if (
        manifest_digest != expected_manifest_digest
        or manifest_digest != stable_hash(body)
        or body.get("schema_version") != 1
        or body.get("phase") != "E1-AA-official-auxiliary"
        or body.get("status") != "complete"
        or body.get("scientific_eligible") is not True
        or body.get("e1_completion_digest") != expected_e1_completion_digest
        or body.get("record_count") != _ROW_COUNT
        or type(body.get("failure_count")) is not int
        or body["failure_count"] < 0
        or body.get("maximum_peak_memory_bytes", _MEMORY_BYTES + 1) > _MEMORY_BYTES
    ):
        raise FrozenArtifactError("AA official portable manifest differs")
    plan = _read_json(source / "plan.json", "AA official plan")
    plan_body = dict(plan)
    plan_identity = plan_body.pop("plan_identity", None)
    _portable_plan_semantics(plan_body)
    if (
        plan_identity != stable_hash(plan_body)
        or plan_identity != body.get("plan_identity")
        or plan_body.get("schema_version") != 1
        or plan_body.get("phase") != "E1-AA-official-auxiliary"
        or plan_body.get("question_count") != _ROW_COUNT
        or plan_body.get("e1_completion_digest") != expected_e1_completion_digest
        or not isinstance(plan_body.get("questions"), list)
        or not isinstance(plan_body.get("question_fingerprints"), Mapping)
        or not isinstance(plan_body.get("condition"), Mapping)
        or not isinstance(plan_body.get("schedule"), Mapping)
    ):
        raise FrozenArtifactError("AA official portable plan differs")
    questions = tuple(_portable_question(value) for value in plan_body["questions"])
    try:
        condition = EvaluationCondition.from_dict(plan_body["condition"])
    except (TypeError, ValueError, DataValidationError) as exc:
        raise FrozenArtifactError(f"AA official portable condition is invalid: {exc}") from exc
    if (
        condition.phase is not ExperimentPhase.E1
        or condition.benchmark != _BENCHMARK
        or condition.system_prompt_id != _PROMPT_ID
        or condition.steering_method != "M0"
        or condition.runtime.value != "vllm"
        or condition.seed != _SEED
        or condition.comparison_group != "aa-official-auxiliary"
        or condition.prompt_template_sha256
        != hashlib.sha256(_OFFICIAL_PROMPT_TEXT.encode("utf-8")).hexdigest()
    ):
        raise FrozenArtifactError("AA official portable condition semantics differ")
    schedule = _official_schedule(condition, questions)
    fingerprints = plan_body["question_fingerprints"]
    if (
        len(questions) != _ROW_COUNT
        or len({value.question_id for value in questions}) != _ROW_COUNT
        or set(fingerprints) != {value.question_id for value in questions}
        or any(
            fingerprints[question.question_id] != stable_hash(_question_body(question))
            for question in questions
        )
        or plan_body["schedule"]
        != {
            "ordering": "sha256-rank-randomized-across-questions-v1",
            "seed": _SEED,
            "schedule_digest": stable_hash(
                [[value.condition_id, question.question_id] for value, question in schedule]
            ),
        }
    ):
        raise FrozenArtifactError("AA official portable question schedule differs")
    grader = _portable_grader(plan_body.get("grader"))
    try:
        sessions = _validate_sessions(
            source / "sessions.jsonl",
            plan_identity=str(plan_identity),
            allow_open=False,
        )
    except DataValidationError as exc:
        raise FrozenArtifactError(f"AA official portable sessions differ: {exc}") from exc
    if (
        not sessions
        or body.get("session_chain_head") != sessions[-1]["event_digest"]
        or body.get("session_set_digest") != stable_hash([row["event_digest"] for row in sessions])
    ):
        raise FrozenArtifactError("AA official portable session identity differs")
    rows = _read_jsonl(source / "records.jsonl", "AA official records")
    generation_fields = {
        "schema_version",
        "sequence",
        "session_index",
        "plan_identity",
        "condition_id",
        "question_id",
        "benchmark",
        "partition",
        "prompt_id",
        "rendered_prompt_sha256",
        "rendered_token_ids_sha256",
        "request_digest",
        "raw_output",
        "raw_output_sha256",
        "raw_output_stable_hash",
        "token_ids",
        "latency_seconds",
        "input_tokens",
        "output_tokens",
        "stop_type",
        "stopping_token_id",
        "prompt_tokens_per_second",
        "generation_tokens_per_second",
        "peak_memory_bytes",
        "active_memory_bytes",
        "cache_memory_bytes",
        "previous_record_digest",
        "record_digest",
    }
    previous: str | None = None
    previous_generation: str | None = None
    outcomes: list[Outcome] = []
    peak = 0
    for sequence, row in enumerate(rows):
        outer = dict(row)
        digest = outer.pop("record_digest", None)
        generation = row.get("generation")
        receipts = row.get("grader_receipts")
        if (
            digest != stable_hash(outer)
            or row.get("sequence") != sequence
            or row.get("plan_identity") != plan_identity
            or row.get("previous_record_digest") != previous
            or not isinstance(generation, Mapping)
            or set(generation) != generation_fields
            or not isinstance(receipts, list)
            or not receipts
        ):
            raise FrozenArtifactError("AA official portable record chain differs")
        generation_body = dict(generation)
        generation_digest = generation_body.pop("record_digest", None)
        _value, question = schedule[sequence]
        raw_output = generation.get("raw_output")
        tokens = generation.get("token_ids")
        if (
            generation_digest != stable_hash(generation_body)
            or generation.get("sequence") != sequence
            or generation.get("session_index")
            not in {int(row["session_index"]) for row in sessions}
            or generation.get("plan_identity") != plan_identity
            or generation.get("condition_id") != condition.condition_id
            or generation.get("question_id") != question.question_id
            or generation.get("benchmark") != _BENCHMARK
            or generation.get("partition") != condition.partition
            or generation.get("prompt_id") != _PROMPT_ID
            or generation.get("previous_record_digest") != previous_generation
            or not _is_sha256(generation.get("rendered_prompt_sha256"))
            or not _is_sha256(generation.get("rendered_token_ids_sha256"))
            or generation.get("request_digest")
            != stable_hash(
                {
                    "plan_identity": plan_identity,
                    "condition_id": condition.condition_id,
                    "question_id": question.question_id,
                    "rendered_prompt_sha256": generation.get("rendered_prompt_sha256"),
                    "rendered_token_ids_sha256": generation.get("rendered_token_ids_sha256"),
                    "max_new_tokens": plan_body["inference"]["max_new_tokens"],
                    "sampling": False,
                    "thinking_enabled": False,
                    "seed": _SEED,
                }
            )
            or not isinstance(raw_output, str)
            or generation.get("raw_output_sha256")
            != hashlib.sha256(raw_output.encode("utf-8")).hexdigest()
            or generation.get("raw_output_stable_hash") != stable_hash(raw_output)
            or not isinstance(tokens, list)
            or any(type(token) is not int or token < 0 for token in tokens)
            or generation.get("output_tokens") != len(tokens)
            or not 1 <= len(tokens) <= int(plan_body["inference"]["max_new_tokens"])
            or generation.get("stopping_token_id") != tokens[-1]
            or not isinstance(generation.get("stop_type"), str)
            or not str(generation["stop_type"]).strip()
        ):
            raise FrozenArtifactError("AA official portable generation differs")
        for name in (
            "input_tokens",
            "peak_memory_bytes",
            "active_memory_bytes",
            "cache_memory_bytes",
        ):
            if type(generation.get(name)) is not int or int(generation[name]) < 0:
                raise FrozenArtifactError("AA official portable generation metric is invalid")
        for name in (
            "latency_seconds",
            "prompt_tokens_per_second",
            "generation_tokens_per_second",
        ):
            value = generation.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise FrozenArtifactError("AA official portable timing metric is invalid")
        request = GradingRequest(
            question.question_id,
            question.text,
            question.aliases[0],
            raw_output,
        )
        raw_label = row.get("grader_raw_label")
        try:
            outcome = grader.label_mapping[str(raw_label).strip()]
        except KeyError as exc:
            raise FrozenArtifactError("AA official portable grade label differs") from exc
        if (
            row.get("grader_request_fingerprint") != request.digest
            or row.get("grader_fingerprint") != grader.digest
            or row.get("outcome") != outcome.value
        ):
            raise FrozenArtifactError("AA official portable grade differs")
        request_sha256, prompt_sha256 = _expected_grader_request(grader, request)
        route = route_for_grader(grader)
        for attempt, receipt in enumerate(receipts, start=1):
            if not isinstance(receipt, Mapping):
                raise FrozenArtifactError("AA official portable receipt is invalid")
            try:
                validate_openrouter_attempt_receipt(
                    receipt,
                    route=route,
                    request_sha256=request_sha256,
                    prompt_sha256=prompt_sha256,
                    attempt=attempt,
                    expect_success=attempt == len(receipts),
                    expected_content=str(raw_label) if attempt == len(receipts) else None,
                    expect_retry=attempt != len(receipts),
                )
            except DataValidationError as exc:
                raise FrozenArtifactError(f"AA official portable receipt differs: {exc}") from exc
        outcomes.append(outcome)
        previous = str(digest)
        previous_generation = str(generation_digest)
        peak = max(peak, int(generation["peak_memory_bytes"]))
    failure_rows = _read_jsonl(source / "failures.jsonl", "AA official failures")
    previous_failure: str | None = None
    session_indices = {int(row["session_index"]) for row in sessions}
    failure_fields = {
        "schema_version",
        "failure_index",
        "sequence",
        "plan_identity",
        "generation",
        "grader_request_fingerprint",
        "grader_fingerprint",
        "grader_raw_label",
        "grader_attempts",
        "outcome",
        "error",
        "grader_receipts",
        "previous_failure_digest",
        "failure_digest",
    }
    for failure_index, failure in enumerate(failure_rows):
        failure_body = dict(failure)
        failure_digest = failure_body.pop("failure_digest", None)
        failure_sequence = failure.get("sequence")
        receipts = failure.get("grader_receipts")
        if (
            set(failure) != failure_fields
            or not isinstance(failure_digest, str)
            or failure_digest != stable_hash(failure_body)
            or failure.get("schema_version") != 1
            or failure.get("failure_index") != failure_index
            or type(failure_sequence) is not int
            or not 0 <= failure_sequence < _ROW_COUNT
            or failure.get("plan_identity") != plan_identity
            or failure.get("previous_failure_digest") != previous_failure
            or failure.get("outcome") != Outcome.UNSCORABLE.value
            or not isinstance(failure.get("error"), str)
            or not str(failure["error"]).strip()
            or not isinstance(failure.get("grader_raw_label"), str)
            or type(failure.get("grader_attempts")) is not int
            or not isinstance(receipts, list)
            or not receipts
            or failure.get("grader_attempts") != len(receipts)
            or len(receipts) > grader.maximum_attempts
        ):
            raise FrozenArtifactError("AA official portable failure chain differs")
        assert isinstance(failure_sequence, int)
        _value, question = schedule[failure_sequence]
        generation_predecessor = (
            str(rows[failure_sequence - 1]["generation"]["record_digest"])
            if failure_sequence > 0
            else None
        )
        raw_output, failure_peak = _portable_failure_generation(
            failure.get("generation"),
            sequence=failure_sequence,
            plan_identity=str(plan_identity),
            condition=condition,
            question=question,
            max_new_tokens=int(plan_body["inference"]["max_new_tokens"]),
            previous_generation_digest=generation_predecessor,
            session_indices=session_indices,
            generation_fields=generation_fields,
        )
        request = GradingRequest(
            question.question_id,
            question.text,
            question.aliases[0],
            raw_output,
        )
        if (
            failure.get("grader_request_fingerprint") != request.digest
            or failure.get("grader_fingerprint") != grader.digest
        ):
            raise FrozenArtifactError("AA official portable failure grade differs")
        request_sha256, prompt_sha256 = _expected_grader_request(grader, request)
        route = route_for_grader(grader)
        for attempt, receipt in enumerate(receipts, start=1):
            if not isinstance(receipt, Mapping):
                raise FrozenArtifactError("AA official portable failure receipt is invalid")
            try:
                validate_openrouter_attempt_receipt(
                    receipt,
                    route=route,
                    request_sha256=request_sha256,
                    prompt_sha256=prompt_sha256,
                    attempt=attempt,
                    expect_success=False,
                    accepted_success_content=(
                        lambda content: content.strip() in grader.label_mapping
                    ),
                    expect_retry=attempt < len(receipts),
                )
            except DataValidationError as exc:
                raise FrozenArtifactError(
                    f"AA official portable failure receipt differs: {exc}"
                ) from exc
        previous_failure = failure_digest
        peak = max(peak, failure_peak)
    if (
        len(rows) != _ROW_COUNT
        or body.get("record_chain_head") != previous
        or body.get("record_set_digest") != stable_hash([row["record_digest"] for row in rows])
        or body.get("failure_count") != len(failure_rows)
        or body.get("failure_chain_head") != previous_failure
        or body.get("failure_set_digest")
        != stable_hash([row["failure_digest"] for row in failure_rows])
        or body.get("maximum_peak_memory_bytes") != peak
    ):
        raise FrozenArtifactError("AA official portable record inventory differs")
    metrics = {
        "schema_version": 1,
        "benchmark": _BENCHMARK,
        "track": "official",
        "prompt_id": _PROMPT_ID,
        "question_count": len(rows),
        "official_metrics": asdict(aa_official_metrics(tuple(outcomes))),
        "unified_metrics": metric_bundle(tuple(outcomes), partial_credit=0.5).to_dict(),
    }
    comparison = _read_json(source / "official-vs-neutral.json", "AA official comparison")
    transitions = comparison.get("transition_counts")
    if (
        _read_json(source / "metrics.json", "AA official metrics") != metrics
        or body.get("metrics_digest") != stable_hash(metrics)
        or body.get("official_vs_neutral_digest") != stable_hash(comparison)
        or not isinstance(comparison.get("official"), Mapping)
        or comparison["official"].get("official_metrics") != metrics["official_metrics"]
        or comparison.get("paired_question_count") != _ROW_COUNT
        or not isinstance(transitions, Mapping)
        or not transitions
        or any(
            not isinstance(key, str) or not key or type(value) is not int or value < 0
            for key, value in transitions.items()
        )
        or sum(transitions.values()) != _ROW_COUNT
        or comparison.get("leaderboard_comparability")
        != {"official_track": True, "neutral_controlled_track": False}
    ):
        raise FrozenArtifactError("AA official portable analysis differs")
    return MappingProxyType(
        {
            "manifest_digest": manifest_digest,
            "record_set_digest": body["record_set_digest"],
            "failure_count": len(failure_rows),
            "e1_completion_digest": body["e1_completion_digest"],
            "metrics": MappingProxyType(metrics),
            "official_vs_neutral": MappingProxyType(comparison),
        }
    )

"""Resumable native-VLLM generation and grading workflow for E1."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol

from mfh.config import load_inference_protocol, load_model_spec, load_prompt_specs
from mfh.contracts import (
    GenerationRecord,
    InterventionSpec,
    ModelSpec,
    Outcome,
    PromptSpec,
    Question,
    Runtime,
)
from mfh.data.io import read_questions
from mfh.data.normalization import normalize_answer
from mfh.data.reviewed_splits import validate_reviewed_split_snapshot
from mfh.errors import ConfigurationError, DataValidationError, FrozenArtifactError
from mfh.evaluation.grading import deterministic_short_answer_grade, triviaqa_scores
from mfh.evaluation.official import (
    GradingRequest,
    OfficialGraderSpec,
    aa_official_metrics,
    load_official_grader_spec,
    render_grader_prompt,
    simpleqa_official_metrics,
)
from mfh.evaluation.openrouter import (
    OPENROUTER_ATTEMPT_RECEIPT_FIELDS,
    OpenRouterTransport,
    route_for_grader,
    run_openrouter_grader,
    validate_openrouter_attempt_receipt,
)
from mfh.experiments.gates import write_gate_evidence
from mfh.experiments.grader_bundle import verify_e1_grader_bundle
from mfh.experiments.model_selection import (
    ACTIVE_RUNTIME_POLICY_RELATIVE,
    validate_active_model_spec,
    validate_active_study_artifact_paths,
)
from mfh.experiments.protocol import ExperimentPhase, StudyProtocol, load_study_protocol
from mfh.experiments.runner import (
    EvaluationCondition,
    PhaseRunContract,
    PhaseRunLedger,
    expand_factorial_conditions,
)
from mfh.inference.transformers_snapshot import verify_transformers_snapshot
from mfh.inference.vllm_preflight import validate_vllm_preflight_receipt
from mfh.inference.vllm_runtime import VllmGenerationOutput, VllmRenderedPrompt, VllmRuntime
from mfh.provenance import canonical_json, sha256_file, sha256_path, stable_hash

_PROMPTS = ("P0-neutral", "P1-direct", "P2-calibrated-abstention")
_BENCHMARK_FILES = {
    "triviaqa": ("T-controller", "T-controller.jsonl", 5_000),
    "simpleqa_verified": ("simpleqa-eval", "simpleqa-eval.jsonl", 1_000),
    "aa_omniscience_public_600": ("aa-eval", "aa-eval.jsonl", 600),
}
_GENERATION_COUNT = sum(count * len(_PROMPTS) for _, _, count in _BENCHMARK_FILES.values())
_SCHEDULE_SEED = 17
_WORK_FILES = frozenset(
    {
        "plan.json",
        "generations.jsonl",
        "generation-sessions.jsonl",
        "grades.jsonl",
        "grader-attempts.jsonl",
        "grading-sessions.jsonl",
    }
)
_MODEL_CLASS = "vllm.model_executor.models.qwen3_5.Qwen3_5ForConditionalGeneration"
_OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
_SHA256 = frozenset("0123456789abcdef")
_ATTEMPT_RECEIPT_KEYS = OPENROUTER_ATTEMPT_RECEIPT_FIELDS


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
class E1Prepared:
    model: ModelSpec
    snapshot: Path
    snapshot_identity: Mapping[str, Any]
    runtime_receipt: Mapping[str, Any]
    prompts: Mapping[str, PromptSpec]
    questions: Mapping[str, tuple[Question, ...]]
    conditions: tuple[EvaluationCondition, ...]
    contract: PhaseRunContract
    study: StudyProtocol
    max_new_tokens: int
    plan: Mapping[str, Any]
    schedule: tuple[tuple[EvaluationCondition, Question], ...]


def _randomized_schedule(
    conditions: Sequence[EvaluationCondition],
    questions: Mapping[str, Sequence[Question]],
) -> tuple[tuple[EvaluationCondition, Question], ...]:
    rows = tuple(
        (condition, question)
        for condition in conditions
        for question in questions[condition.benchmark]
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
                        "question_id": row[1].question_id,
                    }
                ),
                row[0].condition_id,
                row[1].question_id,
            ),
        )
    )


def _read_json(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot read {context}: {exc}") from exc
    if not isinstance(value, dict):
        raise DataValidationError(f"{context} must be a JSON object")
    return value


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
        with path.open(encoding="utf-8") as handle:
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


def _verify_work_inventory(work: Path) -> None:
    if work.is_symlink() or not work.is_dir():
        raise DataValidationError("E1 work must be a regular directory")
    files: set[str] = set()
    directories: set[str] = set()
    for item in work.rglob("*"):
        if item.is_symlink():
            raise DataValidationError("E1 work cannot contain symlinks")
        relative = item.relative_to(work).as_posix()
        if item.is_file():
            files.add(relative)
        elif item.is_dir():
            directories.add(relative)
        else:
            raise DataValidationError("E1 work contains a special file")
    if directories or not files <= _WORK_FILES:
        raise DataValidationError("E1 work inventory differs from the workflow schema")


def _validate_runtime_identity(prepared: E1Prepared, runtime_identity: Mapping[str, Any]) -> None:
    receipt = prepared.runtime_receipt
    model = receipt.get("model")
    expected = receipt.get("runtime_identity")
    if not isinstance(model, Mapping) or not isinstance(expected, Mapping):
        raise DataValidationError("VLLM runtime receipt lacks a required identity section")
    if (
        dict(runtime_identity) != dict(expected)
        or model.get("name") != prepared.model.name
        or model.get("repository") != prepared.model.repository
        or model.get("revision") != prepared.model.revision
        or model.get("quantization") != prepared.model.quantization
        or model.get("num_layers") != prepared.model.num_layers
        or model.get("snapshot_identity") != prepared.snapshot_identity
        or expected.get("model_class") != _MODEL_CLASS
        or not isinstance(expected.get("tokenizer_class"), str)
    ):
        raise DataValidationError("live E1 VLLM identity differs from the frozen runtime receipt")


def _prepare(
    *,
    splits_directory: Path,
    expected_splits_manifest_digest: str,
    grader_bundle: Path,
    expected_grader_manifest_digest: str,
    model_config: Path,
    snapshot_directory: Path,
    snapshot_manifest: Path,
    runtime_config: Path,
    prompt_config: Path,
    inference_config: Path,
    study_config: Path,
    e0_run: Path,
) -> E1Prepared:
    split_manifest = validate_reviewed_split_snapshot(splits_directory)
    if split_manifest.get("manifest_digest") != expected_splits_manifest_digest:
        raise DataValidationError("E1 reviewed-split manifest differs from the expected digest")
    grader_manifest = verify_e1_grader_bundle(
        grader_bundle,
        expected_manifest_digest=expected_grader_manifest_digest,
    )
    model = load_model_spec(model_config)
    validate_active_model_spec(model)
    if model.runtime is not Runtime.VLLM:
        raise ConfigurationError("E1 native runner requires the sole VLLM model")
    snapshot_identity = verify_transformers_snapshot(model, snapshot_directory, snapshot_manifest)
    project_root = study_config.absolute().parents[2]
    receipt = validate_vllm_preflight_receipt(
        runtime_config,
        project_root=project_root,
        model_config=model_config,
        snapshot_directory=snapshot_directory,
        snapshot_manifest=snapshot_manifest,
        runtime_policy=project_root / ACTIVE_RUNTIME_POLICY_RELATIVE,
    )
    prompts = {prompt.prompt_id: prompt for prompt in load_prompt_specs(prompt_config)}
    if not set(_PROMPTS) <= set(prompts):
        raise ConfigurationError("E1 prompt config lacks a primary prompt")
    selected_prompts = {name: prompts[name] for name in _PROMPTS}
    inference = load_inference_protocol(inference_config)
    if (
        inference.temperature != 0
        or inference.do_sample
        or inference.max_new_tokens != 48
        or inference.thinking_enabled
        or inference.retrieval_enabled
        or inference.tools_enabled
        or 17 not in inference.seeds
    ):
        raise ConfigurationError("E1 inference config differs from deterministic decoding")
    study = load_study_protocol(study_config)
    phase = study.phase(ExperimentPhase.E1)
    if phase.models != (model.name,) or set(phase.prompts) != set(_PROMPTS):
        raise ConfigurationError("E1 study protocol differs from the sole-model prompt matrix")
    questions: dict[str, tuple[Question, ...]] = {}
    partitions: dict[str, str] = {}
    for benchmark, (partition, filename, expected_count) in _BENCHMARK_FILES.items():
        values = tuple(read_questions(splits_directory / filename))
        if len(values) != expected_count or any(value.benchmark != benchmark for value in values):
            raise DataValidationError(f"E1 {benchmark} question schedule differs")
        questions[benchmark] = values
        partitions[benchmark] = partition
    conditions = expand_factorial_conditions(
        study,
        ExperimentPhase.E1,
        models={model.name: model},
        prompts=selected_prompts,
        benchmark_partitions=partitions,
        interventions={"M0": InterventionSpec(method="M0")},
        seed=17,
    )
    if len(conditions) != 9 or any(
        condition.partition != partitions[condition.benchmark] for condition in conditions
    ):
        raise DataValidationError("E1 conditions differ from the exact 1 x 3 x 3 matrix")
    e0_completion = PhaseRunLedger.open(e0_run, study=study).verify_complete()
    if e0_completion.phase is not ExperimentPhase.E0:
        raise DataValidationError("E1 prerequisite is not the completed E0 run")
    input_fingerprints = {
        "deduplicated_splits": sha256_path(splits_directory),
        "grader_bundle": sha256_path(grader_bundle),
        "inference_protocol": sha256_path(inference_config),
    }
    contract = PhaseRunContract(
        phase=ExperimentPhase.E1,
        study_protocol_digest=study.digest,
        conditions=conditions,
        question_ids_by_benchmark={
            benchmark: tuple(question.question_id for question in values)
            for benchmark, values in questions.items()
        },
        input_fingerprints=input_fingerprints,
        prerequisite_digests={"E0": e0_completion.completion_digest},
        required_gates=phase.gates,
    )
    contract.assert_matches_study(study)
    schedule = _randomized_schedule(conditions, questions)
    if len(schedule) != _GENERATION_COUNT:
        raise DataValidationError("E1 generation schedule has the wrong cardinality")
    plan_body: dict[str, Any] = {
        "schema_version": 1,
        "phase": "E1",
        "runner": "native-vllm-then-openrouter",
        "runner_source_sha256": sha256_file(Path(__file__)),
        "study_protocol_digest": study.digest,
        "contract_digest": contract.digest,
        "model": {
            "name": model.name,
            "repository": model.repository,
            "revision": model.revision,
            "runtime": model.runtime.value,
            "quantization": model.quantization,
            "num_layers": model.num_layers,
        },
        "snapshot_digest": snapshot_identity["snapshot_digest"],
        "runtime_preflight_receipt_digest": receipt["receipt_digest"],
        "grader_bundle_manifest_digest": grader_manifest["manifest_digest"],
        "conditions": [condition.to_dict() for condition in conditions],
        "question_counts": {benchmark: len(values) for benchmark, values in questions.items()},
        "input_fingerprints": input_fingerprints,
        "prerequisite_digests": {"E0": e0_completion.completion_digest},
        "schedule": {
            "ordering": "sha256-rank-randomized-across-conditions-v1",
            "local_generations": _GENERATION_COUNT,
            "external_grades": 4_800,
            "max_new_tokens": inference.max_new_tokens,
            "seed": _SCHEDULE_SEED,
            "schedule_digest": stable_hash(
                [[condition.condition_id, question.question_id] for condition, question in schedule]
            ),
            "triviaqa_partition": "T-controller",
            "excluded_triviaqa_partitions": ["T-dev", "T-test"],
        },
        "input_hashes": {
            "model_config": sha256_file(model_config),
            "snapshot_manifest": sha256_file(snapshot_manifest),
            "runtime_config": sha256_file(runtime_config),
            "prompt_config": sha256_file(prompt_config),
            "inference_config": sha256_file(inference_config),
            "study_config": sha256_file(study_config),
            "split_manifest": sha256_file(splits_directory / "manifest.json"),
            "grader_manifest": sha256_file(grader_bundle / "manifest.json"),
        },
    }
    plan = {**plan_body, "plan_identity": stable_hash(plan_body)}
    return E1Prepared(
        model=model,
        snapshot=snapshot_directory,
        snapshot_identity=snapshot_identity,
        runtime_receipt=receipt,
        prompts=selected_prompts,
        questions=questions,
        conditions=conditions,
        contract=contract,
        study=study,
        max_new_tokens=inference.max_new_tokens,
        plan=plan,
        schedule=schedule,
    )


def prepare_e1_vllm(
    *,
    splits_directory: str | Path,
    expected_splits_manifest_digest: str,
    grader_bundle: str | Path,
    expected_grader_manifest_digest: str,
    model_config: str | Path,
    snapshot_directory: str | Path,
    snapshot_manifest: str | Path,
    runtime_config: str | Path,
    work_directory: str | Path,
    ledger_directory: str | Path,
    e0_run: str | Path,
    verified_reviewed_splits: object,
    prompt_config: str | Path = "configs/prompts/primary.yaml",
    inference_config: str | Path = "configs/experiments/core.yaml",
    study_config: str | Path = "configs/experiments/phases.yaml",
) -> Mapping[str, Any]:
    """Create the exact E1 ledger contract and immutable resumable work plan."""

    paths = {
        "splits_directory": Path(splits_directory).absolute(),
        "grader_bundle": Path(grader_bundle).absolute(),
        "model_config": Path(model_config).absolute(),
        "snapshot_directory": Path(snapshot_directory).absolute(),
        "snapshot_manifest": Path(snapshot_manifest).absolute(),
        "runtime_config": Path(runtime_config).absolute(),
        "prompt_config": Path(prompt_config).absolute(),
        "inference_config": Path(inference_config).absolute(),
        "study_config": Path(study_config).absolute(),
        "e0_run": Path(e0_run).absolute(),
    }
    prepared = _prepare(
        **paths,
        expected_splits_manifest_digest=expected_splits_manifest_digest,
        expected_grader_manifest_digest=expected_grader_manifest_digest,
    )
    work = Path(work_directory).absolute()
    ledger_path = Path(ledger_directory).absolute()
    validate_active_study_artifact_paths(
        {
            "E0 prerequisite run": paths["e0_run"],
            "E1 work directory": work,
            "E1 ledger directory": ledger_path,
        }
    )
    work.mkdir(parents=True, exist_ok=True)
    _verify_work_inventory(work)
    plan_path = work / "plan.json"
    if plan_path.exists():
        if _read_json(plan_path, "E1 work plan") != dict(prepared.plan):
            raise FrozenArtifactError("existing E1 work belongs to another plan")
    else:
        _write_json_once(plan_path, prepared.plan)
    if ledger_path.exists():
        ledger = PhaseRunLedger.open(ledger_path, study=prepared.study)
        if ledger.contract != prepared.contract:
            raise FrozenArtifactError("existing E1 ledger belongs to another plan")
    else:
        ledger = PhaseRunLedger.create(
            ledger_path,
            prepared.contract,
            study=prepared.study,
            input_artifacts={
                "deduplicated_splits": paths["splits_directory"],
                "grader_bundle": paths["grader_bundle"],
                "inference_protocol": paths["inference_config"],
            },
            prerequisite_runs={"E0": paths["e0_run"]},
            verified_reviewed_splits=verified_reviewed_splits,
        )
    return {
        "prepared": True,
        "plan_identity": prepared.plan["plan_identity"],
        "contract_digest": ledger.contract.digest,
        "conditions": len(prepared.conditions),
        "local_generations": prepared.contract.expected_record_count,
        "external_grades": 4_800,
        "work_directory": str(work),
        "ledger_directory": str(ledger_path),
    }


def _generation_request_digest(
    prepared: E1Prepared,
    condition: EvaluationCondition,
    question: Question,
    rendered: VllmRenderedPrompt,
) -> str:
    return stable_hash(
        {
            "plan_identity": prepared.plan["plan_identity"],
            "condition_id": condition.condition_id,
            "question_id": question.question_id,
            "rendered_prompt_sha256": rendered.sha256,
            "rendered_token_ids_sha256": rendered.token_ids_sha256,
            "max_new_tokens": prepared.max_new_tokens,
            "sampling": False,
            "thinking_enabled": False,
            "seed": 17,
        }
    )


def _generation_body(
    *,
    sequence: int,
    session_index: int,
    prepared: E1Prepared,
    condition: EvaluationCondition,
    question: Question,
    rendered: VllmRenderedPrompt,
    generation: VllmGenerationOutput,
    previous_digest: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "sequence": sequence,
        "session_index": session_index,
        "plan_identity": prepared.plan["plan_identity"],
        "condition_id": condition.condition_id,
        "question_id": question.question_id,
        "benchmark": condition.benchmark,
        "partition": condition.partition,
        "prompt_id": condition.system_prompt_id,
        "rendered_prompt_sha256": rendered.sha256,
        "rendered_token_ids_sha256": rendered.token_ids_sha256,
        "request_digest": _generation_request_digest(prepared, condition, question, rendered),
        "raw_output": generation.text,
        "raw_output_sha256": hashlib.sha256(generation.text.encode("utf-8")).hexdigest(),
        "raw_output_stable_hash": stable_hash(generation.text),
        "token_ids": list(generation.token_ids),
        "latency_seconds": generation.latency_seconds,
        "input_tokens": generation.input_tokens,
        "output_tokens": generation.output_tokens,
        "stop_type": generation.stop_type,
        "stopping_token_id": generation.stopping_token_id,
        "prompt_tokens_per_second": generation.prompt_tokens_per_second,
        "generation_tokens_per_second": generation.generation_tokens_per_second,
        "peak_memory_bytes": generation.peak_memory_bytes,
        "active_memory_bytes": generation.active_memory_bytes,
        "cache_memory_bytes": generation.cache_memory_bytes,
        "previous_record_digest": previous_digest,
    }


def _validate_generation(
    row: Mapping[str, Any],
    *,
    sequence: int,
    session_indices: set[int],
    previous_digest: str | None,
    prepared: E1Prepared,
    runtime: _Runtime,
) -> None:
    body = dict(row)
    digest = body.pop("record_digest", None)
    if digest != stable_hash(body):
        raise DataValidationError("E1 generation record digest differs")
    condition, question = prepared.schedule[sequence]
    rendered = runtime.render_prompt(
        prepared.prompts[condition.system_prompt_id],
        question.text,
        metadata=dict(question.metadata),
    )
    if (
        row.get("schema_version") != 1
        or row.get("sequence") != sequence
        or row.get("session_index") not in session_indices
        or row.get("plan_identity") != prepared.plan["plan_identity"]
        or row.get("condition_id") != condition.condition_id
        or row.get("question_id") != question.question_id
        or row.get("benchmark") != condition.benchmark
        or row.get("partition") != condition.partition
        or row.get("prompt_id") != condition.system_prompt_id
        or row.get("rendered_prompt_sha256") != rendered.sha256
        or row.get("rendered_token_ids_sha256") != rendered.token_ids_sha256
        or row.get("request_digest")
        != _generation_request_digest(prepared, condition, question, rendered)
        or row.get("previous_record_digest") != previous_digest
    ):
        raise DataValidationError("E1 generation record differs from the frozen schedule")
    raw = row.get("raw_output")
    tokens = row.get("token_ids")
    if (
        not isinstance(raw, str)
        or not isinstance(tokens, list)
        or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in tokens
        )
        or row.get("raw_output_sha256") != hashlib.sha256(raw.encode("utf-8")).hexdigest()
        or row.get("raw_output_stable_hash") != stable_hash(raw)
    ):
        raise DataValidationError("E1 generation output evidence differs")
    integer_fields = (
        "input_tokens",
        "output_tokens",
        "peak_memory_bytes",
        "active_memory_bytes",
        "cache_memory_bytes",
    )
    if any(
        isinstance(row.get(name), bool) or not isinstance(row.get(name), int) or int(row[name]) < 0
        for name in integer_fields
    ):
        raise DataValidationError("E1 generation integer metric is invalid")
    if (
        row.get("input_tokens") != len(rendered.token_ids)
        or row.get("output_tokens") != len(tokens)
        or not 1 <= len(tokens) <= prepared.max_new_tokens
        or row.get("stop_type") not in {"stop", "length", "short_answer"}
        or row.get("stopping_token_id") != tokens[-1]
    ):
        raise DataValidationError("E1 generation token or stopping evidence differs")
    number_fields = (
        "latency_seconds",
        "prompt_tokens_per_second",
        "generation_tokens_per_second",
    )
    if any(
        isinstance(row.get(name), bool)
        or not isinstance(row.get(name), int | float)
        or not math.isfinite(float(row[name]))
        or float(row[name]) < 0
        for name in number_fields
    ):
        raise DataValidationError("E1 generation timing metric is invalid")
    if condition.benchmark == "triviaqa":
        exact, token_f1 = triviaqa_scores(raw, question.aliases)
        if exact not in {0.0, 1.0} or not 0 <= token_f1 <= 1:
            raise DataValidationError("E1 TriviaQA deterministic scorer returned invalid values")


def _session_event(
    path: Path,
    *,
    event: str,
    session_index: int,
    plan_identity: str,
    details: Mapping[str, Any],
) -> None:
    rows = _read_jsonl(path, "E1 generation sessions")
    previous = str(rows[-1]["event_digest"]) if rows else None
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


def _validate_sessions(path: Path, *, plan_identity: str, allow_open: bool) -> list[dict[str, Any]]:
    rows = _read_jsonl(path, "E1 generation sessions")
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
            raise DataValidationError("E1 generation session chain differs")
        if event == "start":
            if index in open_sessions:
                raise DataValidationError("E1 generation session starts twice")
            open_sessions.add(index)
        else:
            if index not in open_sessions:
                raise DataValidationError("E1 generation session ends without a start")
            open_sessions.remove(index)
        previous = str(digest)
    if open_sessions and not allow_open:
        raise DataValidationError("E1 generation session is unclosed")
    return rows


def _load_generations(
    work: Path,
    *,
    prepared: E1Prepared,
    runtime: _Runtime,
    session_indices: set[int],
) -> list[dict[str, Any]]:
    rows = _read_jsonl(work / "generations.jsonl", "E1 generations")
    if len(rows) > _GENERATION_COUNT:
        raise DataValidationError("E1 generation count exceeds the frozen schedule")
    previous: str | None = None
    for sequence, row in enumerate(rows):
        _validate_generation(
            row,
            sequence=sequence,
            session_indices=session_indices,
            previous_digest=previous,
            prepared=prepared,
            runtime=runtime,
        )
        previous = str(row["record_digest"])
    return rows


def _resume_checkpoint(
    prepared: E1Prepared,
    records: Sequence[Mapping[str, Any]],
    sessions: Sequence[Mapping[str, Any]],
) -> str:
    return stable_hash(
        {
            "plan_identity": prepared.plan["plan_identity"],
            "records_completed": len(records),
            "record_head": records[-1]["record_digest"] if records else None,
            "session_events": len(sessions),
            "session_head": sessions[-1]["event_digest"] if sessions else None,
        }
    )


def _resume_checkpoint_matches(
    prepared: E1Prepared,
    records: Sequence[Mapping[str, Any]],
    sessions: Sequence[Mapping[str, Any]],
    expected: str,
) -> bool:
    candidates = {_resume_checkpoint(prepared, records, sessions)}
    if records:
        candidates.add(_resume_checkpoint(prepared, records[:-1], sessions))
    if sessions:
        candidates.add(_resume_checkpoint(prepared, records, sessions[:-1]))
    return expected in candidates


def _emit_checkpoint(
    prepared: E1Prepared,
    records: Sequence[Mapping[str, Any]],
    sessions: Sequence[Mapping[str, Any]],
    *,
    reason: str,
    checkpoint_file: Path | None,
) -> str:
    digest = _resume_checkpoint(prepared, records, sessions)
    payload = {
        "schema_version": 1,
        "event": "e1-vllm-resume-checkpoint",
        "reason": reason,
        "plan_identity": prepared.plan["plan_identity"],
        "records_completed": len(records),
        "records_expected": _GENERATION_COUNT,
        "resume_checkpoint": digest,
    }
    if checkpoint_file is not None:
        _atomic_json(checkpoint_file, payload)
    print(json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)
    return digest


def run_e1_vllm_generations(
    *,
    splits_directory: str | Path,
    expected_splits_manifest_digest: str,
    grader_bundle: str | Path,
    expected_grader_manifest_digest: str,
    model_config: str | Path,
    snapshot_directory: str | Path,
    snapshot_manifest: str | Path,
    runtime_config: str | Path,
    work_directory: str | Path,
    ledger_directory: str | Path,
    e0_run: str | Path,
    prompt_config: str | Path = "configs/prompts/primary.yaml",
    inference_config: str | Path = "configs/experiments/core.yaml",
    study_config: str | Path = "configs/experiments/phases.yaml",
    request_budget: int | None = None,
    expected_resume_checkpoint: str | None = None,
    checkpoint_file: str | Path | None = None,
    runtime_factory: Callable[[ModelSpec, Path], _Runtime] | None = None,
) -> Mapping[str, Any]:
    """Run or resume the exact 19,800 unsteered E1 generations."""

    if request_budget is not None and (
        isinstance(request_budget, bool)
        or not isinstance(request_budget, int)
        or request_budget <= 0
    ):
        raise ConfigurationError("E1 VLLM request budget must be a positive integer")
    if expected_resume_checkpoint is not None and (
        not isinstance(expected_resume_checkpoint, str)
        or len(expected_resume_checkpoint) != 64
        or any(character not in "0123456789abcdef" for character in expected_resume_checkpoint)
    ):
        raise ConfigurationError("E1 resume checkpoint must be a lowercase SHA-256")
    paths = {
        "splits_directory": Path(splits_directory).absolute(),
        "grader_bundle": Path(grader_bundle).absolute(),
        "model_config": Path(model_config).absolute(),
        "snapshot_directory": Path(snapshot_directory).absolute(),
        "snapshot_manifest": Path(snapshot_manifest).absolute(),
        "runtime_config": Path(runtime_config).absolute(),
        "prompt_config": Path(prompt_config).absolute(),
        "inference_config": Path(inference_config).absolute(),
        "study_config": Path(study_config).absolute(),
        "e0_run": Path(e0_run).absolute(),
    }
    prepared = _prepare(
        **paths,
        expected_splits_manifest_digest=expected_splits_manifest_digest,
        expected_grader_manifest_digest=expected_grader_manifest_digest,
    )
    work = Path(work_directory).absolute()
    ledger_path = Path(ledger_directory).absolute()
    checkpoint = Path(checkpoint_file).absolute() if checkpoint_file is not None else None
    mutable_paths = {
        "E0 prerequisite run": paths["e0_run"],
        "E1 work directory": work,
        "E1 ledger directory": ledger_path,
    }
    if checkpoint is not None:
        mutable_paths["E1 generation checkpoint"] = checkpoint
    validate_active_study_artifact_paths(mutable_paths)
    _verify_work_inventory(work)
    if _read_json(work / "plan.json", "E1 work plan") != dict(prepared.plan):
        raise FrozenArtifactError("E1 work plan differs from live frozen inputs")
    ledger = PhaseRunLedger.open(ledger_path, study=prepared.study)
    if ledger.contract != prepared.contract:
        raise FrozenArtifactError("E1 ledger contract differs from the work plan")
    if ledger.progress()[0] != 0 or (ledger_path / "complete.json").exists():
        raise FrozenArtifactError("E1 generation cannot continue after ledger checkpointing")
    if checkpoint is not None and (
        checkpoint == work
        or checkpoint.is_relative_to(work)
        or checkpoint.is_relative_to(ledger_path)
    ):
        raise ConfigurationError("E1 checkpoint must stay outside work and ledger directories")
    if checkpoint is None:
        raise ConfigurationError("E1 generation requires an external checkpoint file")
    has_execution = (work / "generation-sessions.jsonl").exists() or (
        work / "generations.jsonl"
    ).exists()
    if has_execution != (expected_resume_checkpoint is not None):
        raise ConfigurationError(
            "existing E1 execution requires its external resume checkpoint"
            if has_execution
            else "an E1 resume checkpoint cannot initialize generation"
        )
    factory = runtime_factory or (
        lambda model, snapshot: VllmRuntime.from_spec(model, snapshot_path=snapshot, seed=17)
    )
    runtime: _Runtime | None = None
    records: list[dict[str, Any]] = []
    session_index = -1
    status = "error"
    active_error: BaseException | None = None
    try:
        runtime = factory(prepared.model, prepared.snapshot)
        runtime_identity = dict(runtime.runtime_identity())
        _validate_runtime_identity(prepared, runtime_identity)
        sessions = _validate_sessions(
            work / "generation-sessions.jsonl",
            plan_identity=str(prepared.plan["plan_identity"]),
            allow_open=has_execution,
        )
        starts = {int(row["session_index"]) for row in sessions if row["event"] == "start"}
        ends = {int(row["session_index"]) for row in sessions if row["event"] == "end"}
        session_indices = starts | ends
        records = _load_generations(
            work,
            prepared=prepared,
            runtime=runtime,
            session_indices=session_indices,
        )
        if has_execution:
            expected = str(expected_resume_checkpoint)
            if not _resume_checkpoint_matches(prepared, records, sessions, expected):
                raise DataValidationError(
                    "E1 VLLM resume checkpoint differs from the external head"
                )
            if _resume_checkpoint(prepared, records, sessions) != expected:
                _emit_checkpoint(
                    prepared,
                    records,
                    sessions,
                    reason="crash-gap-catch-up",
                    checkpoint_file=checkpoint,
                )
        for interrupted in sorted(starts - ends):
            _session_event(
                work / "generation-sessions.jsonl",
                event="end",
                session_index=interrupted,
                plan_identity=str(prepared.plan["plan_identity"]),
                details={"records_at_end": len(records), "status": "interrupted-recovered"},
            )
            repaired_sessions = _read_jsonl(
                work / "generation-sessions.jsonl", "E1 generation sessions"
            )
            _emit_checkpoint(
                prepared,
                records,
                repaired_sessions,
                reason="interrupted-session-repaired",
                checkpoint_file=checkpoint,
            )
        sessions = _validate_sessions(
            work / "generation-sessions.jsonl",
            plan_identity=str(prepared.plan["plan_identity"]),
            allow_open=False,
        )
        session_index = max((int(row["session_index"]) for row in sessions), default=-1) + 1
        _session_event(
            work / "generation-sessions.jsonl",
            event="start",
            session_index=session_index,
            plan_identity=str(prepared.plan["plan_identity"]),
            details={"records_at_start": len(records), "runtime_identity": runtime_identity},
        )
        session_indices.add(session_index)
        sessions = _read_jsonl(work / "generation-sessions.jsonl", "E1 generation sessions")
        _emit_checkpoint(
            prepared, records, sessions, reason="session-start", checkpoint_file=checkpoint
        )
        new_requests = 0
        while len(records) < _GENERATION_COUNT and (
            request_budget is None or new_requests < request_budget
        ):
            sequence = len(records)
            condition, question = prepared.schedule[sequence]
            rendered = runtime.render_prompt(
                prepared.prompts[condition.system_prompt_id],
                question.text,
                metadata=dict(question.metadata),
            )
            generation = runtime.generate(rendered, max_new_tokens=prepared.max_new_tokens)
            body = _generation_body(
                sequence=sequence,
                session_index=session_index,
                prepared=prepared,
                condition=condition,
                question=question,
                rendered=rendered,
                generation=generation,
                previous_digest=(str(records[-1]["record_digest"]) if records else None),
            )
            record = {**body, "record_digest": stable_hash(body)}
            _validate_generation(
                record,
                sequence=sequence,
                session_indices=session_indices,
                previous_digest=(str(records[-1]["record_digest"]) if records else None),
                prepared=prepared,
                runtime=runtime,
            )
            _append_jsonl(work / "generations.jsonl", record)
            records.append(record)
            new_requests += 1
            sessions = _read_jsonl(work / "generation-sessions.jsonl", "E1 generation sessions")
            _emit_checkpoint(
                prepared,
                records,
                sessions,
                reason="record-appended",
                checkpoint_file=checkpoint,
            )
            if len(records) <= 5 or len(records) % 25 == 0:
                print(
                    json.dumps(
                        {
                            "event": "e1-vllm-progress",
                            "records_completed": len(records),
                            "records_expected": _GENERATION_COUNT,
                            "benchmark": condition.benchmark,
                            "prompt": condition.system_prompt_id,
                            "question_id": question.question_id,
                            "latency_seconds": round(generation.latency_seconds, 6),
                            "output_tokens": generation.output_tokens,
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
        status = "complete" if len(records) == _GENERATION_COUNT else "partial"
    except BaseException as exc:
        active_error = exc
    finally:
        if session_index >= 0:
            try:
                _session_event(
                    work / "generation-sessions.jsonl",
                    event="end",
                    session_index=session_index,
                    plan_identity=str(prepared.plan["plan_identity"]),
                    details={"records_at_end": len(records), "status": status},
                )
                sessions = _validate_sessions(
                    work / "generation-sessions.jsonl",
                    plan_identity=str(prepared.plan["plan_identity"]),
                    allow_open=False,
                )
                _emit_checkpoint(
                    prepared,
                    records,
                    sessions,
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
    if active_error is not None:
        raise active_error
    post_snapshot = verify_transformers_snapshot(
        prepared.model,
        paths["snapshot_directory"],
        paths["snapshot_manifest"],
    )
    if post_snapshot != prepared.snapshot_identity:
        raise FrozenArtifactError("E1 model snapshot changed during generation")
    live_inputs = {
        "deduplicated_splits": sha256_path(paths["splits_directory"]),
        "grader_bundle": sha256_path(paths["grader_bundle"]),
        "inference_protocol": sha256_path(paths["inference_config"]),
    }
    if live_inputs != dict(prepared.contract.input_fingerprints):
        raise FrozenArtifactError("an E1 frozen input changed during generation")
    sessions = _validate_sessions(
        work / "generation-sessions.jsonl",
        plan_identity=str(prepared.plan["plan_identity"]),
        allow_open=False,
    )
    return {
        "complete": status == "complete",
        "plan_identity": prepared.plan["plan_identity"],
        "contract_digest": prepared.contract.digest,
        "records_completed": len(records),
        "records_expected": _GENERATION_COUNT,
        "resume_checkpoint": _resume_checkpoint(prepared, records, sessions),
        "work_directory": str(work),
        "ledger_directory": str(ledger_path),
    }


def _tokenizer_renderer(prepared: E1Prepared) -> _Runtime:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional verification dependency
        raise ConfigurationError("E1 verification requires transformers") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        str(prepared.snapshot), local_files_only=True, trust_remote_code=False
    )
    runtime = VllmRuntime(
        engine=None,
        tokenizer=tokenizer,
        model_spec=prepared.model,
        snapshot=prepared.snapshot,
        seed=17,
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
            raise AssertionError("tokenizer-only E1 verifier cannot generate")

        def runtime_identity(self) -> Mapping[str, Any]:
            return {}

        def close(self) -> None:
            return None

    return TokenizerRenderer()


def _complete_generations(
    work: Path, *, prepared: E1Prepared, renderer: _Runtime
) -> list[dict[str, Any]]:
    sessions = _validate_sessions(
        work / "generation-sessions.jsonl",
        plan_identity=str(prepared.plan["plan_identity"]),
        allow_open=False,
    )
    session_indices = {int(row["session_index"]) for row in sessions}
    records = _load_generations(
        work,
        prepared=prepared,
        runtime=renderer,
        session_indices=session_indices,
    )
    if len(records) != _GENERATION_COUNT:
        raise DataValidationError(f"E1 grading requires all {_GENERATION_COUNT} local generations")
    return records


def _external_schedule(
    prepared: E1Prepared, generations: Sequence[Mapping[str, Any]]
) -> tuple[tuple[int, EvaluationCondition, Question, Mapping[str, Any]], ...]:
    return tuple(
        (sequence, condition, question, generations[sequence])
        for sequence, (condition, question) in enumerate(prepared.schedule)
        if condition.benchmark != "triviaqa"
    )


def _grader_specs(
    grader_bundle: Path, manifest: Mapping[str, Any]
) -> Mapping[str, OfficialGraderSpec]:
    files = manifest["files"]
    return {
        "simpleqa_verified": load_official_grader_spec(
            grader_bundle / files["simpleqa_config"]["path"]
        ),
        "aa_omniscience_public_600": load_official_grader_spec(
            grader_bundle / files["aa_config"]["path"]
        ),
    }


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256 for character in value)
    )


def _expected_openrouter_request(
    spec: OfficialGraderSpec, request: GradingRequest
) -> tuple[str, str]:
    prompt = render_grader_prompt(spec, request)
    payload = OpenRouterTransport(api_key="validation-only").request_payload(prompt, spec)
    return (
        hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest(),
        hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    )


def _validate_attempt_rows(
    path: Path,
    *,
    prepared: E1Prepared,
    external: Sequence[tuple[int, EvaluationCondition, Question, Mapping[str, Any]]],
    specs: Mapping[str, OfficialGraderSpec],
) -> list[dict[str, Any]]:
    rows = _read_jsonl(path, "E1 OpenRouter attempt receipts")
    previous: str | None = None
    previous_grade_sequence = -1
    invocation_key: tuple[int, int] | None = None
    invocation_attempt = 0
    request_cache: dict[int, tuple[str, str]] = {}
    for row in rows:
        body = dict(row)
        digest = body.pop("attempt_record_digest", None)
        grade_sequence = row.get("grade_sequence")
        session_index = row.get("grading_session_index")
        if (
            digest != stable_hash(body)
            or set(row)
            != {
                "schema_version",
                "plan_identity",
                "grade_sequence",
                "generation_sequence",
                "generation_record_digest",
                "grading_session_index",
                "accepted_label",
                "receipt",
                "previous_attempt_record_digest",
                "attempt_record_digest",
            }
            or row.get("schema_version") != 1
            or row.get("plan_identity") != prepared.plan["plan_identity"]
            or row.get("previous_attempt_record_digest") != previous
            or isinstance(grade_sequence, bool)
            or not isinstance(grade_sequence, int)
            or not 0 <= grade_sequence < len(external)
            or isinstance(session_index, bool)
            or not isinstance(session_index, int)
            or session_index < 0
            or grade_sequence < previous_grade_sequence
            or grade_sequence > previous_grade_sequence + 1
        ):
            raise DataValidationError("E1 OpenRouter attempt receipt chain differs")
        generation_sequence, condition, question, generation = external[grade_sequence]
        receipt = row.get("receipt")
        request = GradingRequest(
            question.question_id,
            question.text,
            question.aliases[0],
            str(generation["raw_output"]),
        )
        request_sha256, prompt_sha256 = request_cache.setdefault(
            grade_sequence,
            _expected_openrouter_request(specs[condition.benchmark], request),
        )
        spec = specs[condition.benchmark]
        route = route_for_grader(spec)
        serialized = json.dumps(receipt, ensure_ascii=False, sort_keys=True)
        if (
            row.get("generation_sequence") != generation_sequence
            or row.get("generation_record_digest") != generation["record_digest"]
            or not isinstance(receipt, Mapping)
            or set(receipt) != _ATTEMPT_RECEIPT_KEYS
            or receipt.get("schema_version") != 1
            or receipt.get("attempt") not in {1, 2, 3}
            or int(receipt.get("attempt", 0)) > spec.maximum_attempts
            or receipt.get("endpoint") != _OPENROUTER_ENDPOINT
            or receipt.get("request_sha256") != request_sha256
            or receipt.get("prompt_sha256") != prompt_sha256
            or receipt.get("requested_model") != route.request_model
            or receipt.get("canonical_slug") != route.canonical_slug
            or receipt.get("required_provider_slug") != route.provider_slug
            or "authorization" in serialized.lower()
            or "bearer " in serialized.lower()
        ):
            raise DataValidationError("E1 OpenRouter attempt evidence differs")
        key = (grade_sequence, session_index)
        if key != invocation_key:
            invocation_key = key
            invocation_attempt = 1
        else:
            invocation_attempt += 1
        if receipt.get("attempt") != invocation_attempt:
            raise DataValidationError("E1 OpenRouter attempt ordering differs")
        successful = receipt.get("error_type") is None
        accepted_label = row.get("accepted_label")
        if successful:
            if (
                not isinstance(accepted_label, str)
                or accepted_label.strip() not in spec.label_mapping
            ):
                raise DataValidationError("E1 successful OpenRouter label is invalid")
        elif accepted_label is not None:
            raise DataValidationError("E1 failed OpenRouter attempt accepted a label")
        accepted_labels = frozenset(spec.label_mapping)

        def accepted_success_content(
            content: str, labels: frozenset[str] = accepted_labels
        ) -> bool:
            return content.strip() in labels

        validate_openrouter_attempt_receipt(
            receipt,
            route=route,
            request_sha256=request_sha256,
            prompt_sha256=prompt_sha256,
            attempt=invocation_attempt,
            expect_success=successful,
            expected_content=accepted_label if successful else None,
            accepted_success_content=accepted_success_content,
            expect_retry=receipt.get("transient") is True,
        )
        http_status = receipt.get("http_status")
        latency = receipt.get("latency_seconds")
        usage = receipt.get("usage")
        if (
            (http_status is not None and (type(http_status) is not int or http_status < 100))
            or isinstance(latency, bool)
            or not isinstance(latency, int | float)
            or not math.isfinite(float(latency))
            or float(latency) < 0
            or not isinstance(receipt.get("transient"), bool)
            or not isinstance(usage, Mapping)
            or not set(usage) <= {"prompt_tokens", "completion_tokens", "total_tokens"}
            or any(
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                or float(value) < 0
                for value in usage.values()
            )
            or any(
                value is not None and not _is_sha256(value)
                for value in (receipt.get("response_sha256"), receipt.get("content_sha256"))
            )
        ):
            raise DataValidationError("E1 OpenRouter attempt metrics differ")
        previous = str(digest)
        previous_grade_sequence = grade_sequence
    return rows


def _validate_grades(
    path: Path,
    *,
    prepared: E1Prepared,
    external: Sequence[tuple[int, EvaluationCondition, Question, Mapping[str, Any]]],
    specs: Mapping[str, OfficialGraderSpec],
    attempt_rows: Sequence[Mapping[str, Any]],
    allow_recoverable_success: bool = False,
) -> list[dict[str, Any]]:
    rows = _read_jsonl(path, "E1 official grades")
    if len(rows) > len(external):
        raise DataValidationError("E1 official grade count exceeds the schedule")
    attempt_by_digest = {
        str(row["attempt_record_digest"]): (index, row) for index, row in enumerate(attempt_rows)
    }
    attached_attempts: set[str] = set()
    previous: str | None = None
    for grade_sequence, row in enumerate(rows):
        body = dict(row)
        digest = body.pop("grade_record_digest", None)
        generation_sequence, condition, question, generation = external[grade_sequence]
        request = GradingRequest(
            question.question_id,
            question.text,
            question.aliases[0],
            str(generation["raw_output"]),
        )
        spec = specs[condition.benchmark]
        raw_label = row.get("raw_label")
        try:
            expected_outcome = spec.label_mapping[str(raw_label).strip()]
        except KeyError as exc:
            raise DataValidationError("E1 grade has an unknown official label") from exc
        receipt_digests = row.get("attempt_record_digests")
        attached: list[tuple[int, Mapping[str, Any]]] = []
        if isinstance(receipt_digests, list):
            try:
                attached = [attempt_by_digest[str(value)] for value in receipt_digests]
            except KeyError:
                attached = []
        attached_receipts = [value[1]["receipt"] for value in attached]
        attached_indices = [value[0] for value in attached]
        expected_grade_digests = [
            str(value["attempt_record_digest"])
            for value in attempt_rows
            if value.get("grade_sequence") == grade_sequence
        ]
        expected_attempt_numbers: list[int] = []
        previous_session: object = None
        invocation_attempt = 0
        for _index, attempt in attached:
            if attempt.get("grading_session_index") != previous_session:
                invocation_attempt = 1
                previous_session = attempt.get("grading_session_index")
            else:
                invocation_attempt += 1
            expected_attempt_numbers.append(invocation_attempt)
        raw_label_text = str(raw_label) if isinstance(raw_label, str) else ""
        if (
            digest != stable_hash(body)
            or set(row)
            != {
                "schema_version",
                "plan_identity",
                "grade_sequence",
                "generation_sequence",
                "generation_record_digest",
                "condition_id",
                "question_id",
                "benchmark",
                "request_fingerprint",
                "grader_fingerprint",
                "raw_label",
                "outcome",
                "attempts",
                "attempt_record_digests",
                "previous_grade_record_digest",
                "grade_record_digest",
            }
            or row.get("schema_version") != 1
            or row.get("grade_sequence") != grade_sequence
            or row.get("plan_identity") != prepared.plan["plan_identity"]
            or row.get("generation_sequence") != generation_sequence
            or row.get("generation_record_digest") != generation["record_digest"]
            or row.get("condition_id") != condition.condition_id
            or row.get("question_id") != question.question_id
            or row.get("benchmark") != condition.benchmark
            or row.get("request_fingerprint") != request.digest
            or row.get("grader_fingerprint") != spec.digest
            or row.get("outcome") != expected_outcome.value
            or expected_outcome is Outcome.UNSCORABLE
            or row.get("previous_grade_record_digest") != previous
            or not isinstance(receipt_digests, list)
            or not receipt_digests
            or receipt_digests != expected_grade_digests
            or len(attached) != len(receipt_digests)
            or len(set(receipt_digests)) != len(receipt_digests)
            or any(str(value) in attached_attempts for value in receipt_digests)
            or attached_indices
            != list(range(attached_indices[0], attached_indices[0] + len(attached_indices)))
            or any(value[1].get("grade_sequence") != grade_sequence for value in attached)
            or [value.get("attempt") for value in attached_receipts] != expected_attempt_numbers
            or row.get("attempts") != len(attached_receipts)
            or any(value.get("error_type") is None for value in attached_receipts[:-1])
            or attached_receipts[-1].get("error_type") is not None
            or attached_receipts[-1].get("content_sha256")
            != hashlib.sha256(raw_label_text.encode("utf-8")).hexdigest()
        ):
            raise DataValidationError("E1 official grade differs from frozen evidence")
        attached_attempts.update(str(value) for value in receipt_digests)
        previous = str(digest)
    successful_attempts = {
        str(row["attempt_record_digest"])
        for row in attempt_rows
        if isinstance(row.get("receipt"), Mapping) and row["receipt"].get("error_type") is None
    }
    unattached_successes = successful_attempts - attached_attempts
    recoverable = (
        allow_recoverable_success
        and len(unattached_successes) == 1
        and bool(attempt_rows)
        and str(attempt_rows[-1]["attempt_record_digest"]) in unattached_successes
        and attempt_rows[-1].get("grade_sequence") == len(rows)
        and len(rows) < len(external)
    )
    if unattached_successes and not recoverable:
        raise DataValidationError("E1 successful attempt receipts differ from official grades")
    return rows


def _recoverable_attempt_invocation(
    attempts: Sequence[Mapping[str, Any]],
    grades: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    attached = {str(digest) for grade in grades for digest in grade["attempt_record_digests"]}
    if not attempts:
        return ()
    last = attempts[-1]
    last_receipt = last["receipt"]
    if (
        str(last["attempt_record_digest"]) in attached
        or last_receipt.get("error_type") is not None
        or last.get("grade_sequence") != len(grades)
    ):
        return ()
    grade_sequence = last["grade_sequence"]
    session_index = last["grading_session_index"]
    start = len(attempts) - 1
    while start > 0:
        previous = attempts[start - 1]
        if (
            previous.get("grade_sequence") != grade_sequence
            or previous.get("grading_session_index") != session_index
        ):
            break
        start -= 1
    invocation = tuple(attempts[start:])
    if (
        [row["receipt"]["attempt"] for row in invocation] != list(range(1, len(invocation) + 1))
        or any(row["receipt"].get("error_type") is None for row in invocation[:-1])
        or not isinstance(last.get("accepted_label"), str)
    ):
        raise DataValidationError("E1 recoverable grader invocation differs")
    return invocation


def _grading_checkpoint(
    prepared: E1Prepared,
    grades: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
    sessions: Sequence[Mapping[str, Any]],
) -> str:
    return stable_hash(
        {
            "plan_identity": prepared.plan["plan_identity"],
            "grades_completed": len(grades),
            "grade_head": grades[-1]["grade_record_digest"] if grades else None,
            "attempt_count": len(attempts),
            "attempt_head": attempts[-1]["attempt_record_digest"] if attempts else None,
            "session_events": len(sessions),
            "session_head": sessions[-1]["event_digest"] if sessions else None,
        }
    )


def _grading_checkpoint_matches(
    prepared: E1Prepared,
    grades: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
    sessions: Sequence[Mapping[str, Any]],
    expected: str,
) -> bool:
    candidates = {_grading_checkpoint(prepared, grades, attempts, sessions)}
    if grades:
        candidates.add(_grading_checkpoint(prepared, grades[:-1], attempts, sessions))
    if attempts:
        candidates.add(_grading_checkpoint(prepared, grades, attempts[:-1], sessions))
    if sessions:
        candidates.add(_grading_checkpoint(prepared, grades, attempts, sessions[:-1]))
    return expected in candidates


def _emit_grading_checkpoint(
    prepared: E1Prepared,
    grades: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
    sessions: Sequence[Mapping[str, Any]],
    *,
    reason: str,
    checkpoint_file: Path | None,
) -> str:
    digest = _grading_checkpoint(prepared, grades, attempts, sessions)
    payload = {
        "schema_version": 1,
        "event": "e1-openrouter-resume-checkpoint",
        "reason": reason,
        "plan_identity": prepared.plan["plan_identity"],
        "grades_completed": len(grades),
        "grades_expected": 4_800,
        "attempts_recorded": len(attempts),
        "resume_checkpoint": digest,
    }
    if checkpoint_file is not None:
        _atomic_json(checkpoint_file, payload)
    print(json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)
    return digest


def load_env_secret(path: str | Path, name: str) -> str:
    """Read one exact local .env secret without mutating or serializing the environment."""

    existing = os.environ.get(name)
    if existing is not None and existing.strip():
        return existing.strip()
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConfigurationError(f"cannot read secret environment file: {exc}") from exc
    found: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:].lstrip()
        key, separator, value = stripped.partition("=")
        if not separator or key.strip() != name:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        found.append(value)
    if len(found) != 1 or not found[0].strip():
        raise ConfigurationError(f"{name} must occur exactly once and be non-empty")
    return found[0].strip()


def _append_successful_grade(
    *,
    path: Path,
    prepared: E1Prepared,
    external: Sequence[tuple[int, EvaluationCondition, Question, Mapping[str, Any]]],
    specs: Mapping[str, OfficialGraderSpec],
    attempts: Sequence[Mapping[str, Any]],
    grades: list[dict[str, Any]],
    raw_label: str,
) -> dict[str, Any]:
    grade_sequence = len(grades)
    generation_sequence, condition, question, generation = external[grade_sequence]
    request = GradingRequest(
        question.question_id,
        question.text,
        question.aliases[0],
        str(generation["raw_output"]),
    )
    spec = specs[condition.benchmark]
    try:
        outcome = spec.label_mapping[raw_label.strip()]
    except KeyError as exc:
        raise DataValidationError("E1 successful grade has an unknown label") from exc
    grade_attempts = [row for row in attempts if row.get("grade_sequence") == grade_sequence]
    if not grade_attempts or grade_attempts[-1]["accepted_label"] != raw_label:
        raise DataValidationError("E1 successful grade lacks its accepted receipt")
    grade_body = {
        "schema_version": 1,
        "plan_identity": prepared.plan["plan_identity"],
        "grade_sequence": grade_sequence,
        "generation_sequence": generation_sequence,
        "generation_record_digest": generation["record_digest"],
        "condition_id": condition.condition_id,
        "question_id": question.question_id,
        "benchmark": condition.benchmark,
        "request_fingerprint": request.digest,
        "grader_fingerprint": spec.digest,
        "raw_label": raw_label,
        "outcome": outcome.value,
        "attempts": len(grade_attempts),
        "attempt_record_digests": [row["attempt_record_digest"] for row in grade_attempts],
        "previous_grade_record_digest": (grades[-1]["grade_record_digest"] if grades else None),
    }
    grade_row = {**grade_body, "grade_record_digest": stable_hash(grade_body)}
    _append_jsonl(path, grade_row)
    grades.append(grade_row)
    return grade_row


def grade_e1_openrouter(
    *,
    splits_directory: str | Path,
    expected_splits_manifest_digest: str,
    grader_bundle: str | Path,
    expected_grader_manifest_digest: str,
    model_config: str | Path,
    snapshot_directory: str | Path,
    snapshot_manifest: str | Path,
    runtime_config: str | Path,
    work_directory: str | Path,
    ledger_directory: str | Path,
    e0_run: str | Path,
    api_key: str,
    prompt_config: str | Path = "configs/prompts/primary.yaml",
    inference_config: str | Path = "configs/experiments/core.yaml",
    study_config: str | Path = "configs/experiments/phases.yaml",
    request_budget: int | None = None,
    expected_resume_checkpoint: str | None = None,
    checkpoint_file: str | Path | None = None,
    transport_factory: Callable[[], OpenRouterTransport] | None = None,
) -> Mapping[str, Any]:
    """Run or resume 4,800 external grades; failed grades remain pending."""

    if request_budget is not None and (
        isinstance(request_budget, bool)
        or not isinstance(request_budget, int)
        or request_budget <= 0
    ):
        raise ConfigurationError("E1 grading request budget must be a positive integer")
    if expected_resume_checkpoint is not None and not _is_sha256(expected_resume_checkpoint):
        raise ConfigurationError("E1 grading checkpoint must be a lowercase SHA-256")
    paths = {
        "splits_directory": Path(splits_directory).absolute(),
        "grader_bundle": Path(grader_bundle).absolute(),
        "model_config": Path(model_config).absolute(),
        "snapshot_directory": Path(snapshot_directory).absolute(),
        "snapshot_manifest": Path(snapshot_manifest).absolute(),
        "runtime_config": Path(runtime_config).absolute(),
        "prompt_config": Path(prompt_config).absolute(),
        "inference_config": Path(inference_config).absolute(),
        "study_config": Path(study_config).absolute(),
        "e0_run": Path(e0_run).absolute(),
    }
    prepared = _prepare(
        **paths,
        expected_splits_manifest_digest=expected_splits_manifest_digest,
        expected_grader_manifest_digest=expected_grader_manifest_digest,
    )
    work = Path(work_directory).absolute()
    ledger_path = Path(ledger_directory).absolute()
    checkpoint = Path(checkpoint_file).absolute() if checkpoint_file is not None else None
    mutable_paths = {
        "E0 prerequisite run": paths["e0_run"],
        "E1 work directory": work,
        "E1 ledger directory": ledger_path,
    }
    if checkpoint is not None:
        mutable_paths["E1 grading checkpoint"] = checkpoint
    validate_active_study_artifact_paths(mutable_paths)
    _verify_work_inventory(work)
    if _read_json(work / "plan.json", "E1 work plan") != dict(prepared.plan):
        raise FrozenArtifactError("E1 work plan differs before grading")
    ledger = PhaseRunLedger.open(ledger_path, study=prepared.study)
    if ledger.contract != prepared.contract or ledger.progress()[0] != 0:
        raise FrozenArtifactError("E1 ledger is not ready for the grading stage")
    if checkpoint is not None and (
        checkpoint == work
        or checkpoint.is_relative_to(work)
        or checkpoint.is_relative_to(ledger_path)
    ):
        raise ConfigurationError("E1 checkpoint must stay outside work and ledger directories")
    if checkpoint is None:
        raise ConfigurationError("E1 grading requires an external checkpoint file")
    renderer = _tokenizer_renderer(prepared)
    try:
        generations = _complete_generations(work, prepared=prepared, renderer=renderer)
    finally:
        renderer.close()
    external = _external_schedule(prepared, generations)
    grader_manifest = verify_e1_grader_bundle(
        paths["grader_bundle"],
        expected_manifest_digest=expected_grader_manifest_digest,
    )
    specs = _grader_specs(paths["grader_bundle"], grader_manifest)
    has_grading = any(
        (work / name).exists()
        for name in ("grading-sessions.jsonl", "grader-attempts.jsonl", "grades.jsonl")
    )
    if has_grading != (expected_resume_checkpoint is not None):
        raise ConfigurationError(
            "existing E1 grading requires its external resume checkpoint"
            if has_grading
            else "an E1 grading checkpoint cannot initialize grading"
        )
    sessions = _validate_sessions(
        work / "grading-sessions.jsonl",
        plan_identity=str(prepared.plan["plan_identity"]),
        allow_open=has_grading,
    )
    starts = {int(row["session_index"]) for row in sessions if row["event"] == "start"}
    ends = {int(row["session_index"]) for row in sessions if row["event"] == "end"}
    attempts = _validate_attempt_rows(
        work / "grader-attempts.jsonl",
        prepared=prepared,
        external=external,
        specs=specs,
    )
    if any(int(row["grading_session_index"]) not in starts for row in attempts):
        raise DataValidationError("E1 grader attempt refers to an unknown session")
    grades = _validate_grades(
        work / "grades.jsonl",
        prepared=prepared,
        external=external,
        specs=specs,
        attempt_rows=attempts,
        allow_recoverable_success=True,
    )
    if has_grading:
        expected = str(expected_resume_checkpoint)
        if not _grading_checkpoint_matches(
            prepared,
            grades,
            attempts,
            sessions,
            expected,
        ):
            raise DataValidationError("E1 grading resume checkpoint differs from external head")
        if _grading_checkpoint(prepared, grades, attempts, sessions) != expected:
            _emit_grading_checkpoint(
                prepared,
                grades,
                attempts,
                sessions,
                reason="crash-gap-catch-up",
                checkpoint_file=checkpoint,
            )
    recoverable = _recoverable_attempt_invocation(attempts, grades)
    if recoverable:
        _append_successful_grade(
            path=work / "grades.jsonl",
            prepared=prepared,
            external=external,
            specs=specs,
            attempts=attempts,
            grades=grades,
            raw_label=str(recoverable[-1]["accepted_label"]),
        )
        _emit_grading_checkpoint(
            prepared,
            grades,
            attempts,
            sessions,
            reason="recovered-grade-appended",
            checkpoint_file=checkpoint,
        )
    for interrupted in sorted(starts - ends):
        _session_event(
            work / "grading-sessions.jsonl",
            event="end",
            session_index=interrupted,
            plan_identity=str(prepared.plan["plan_identity"]),
            details={"grades_at_end": len(grades), "status": "interrupted-recovered"},
        )
        repaired_sessions = _read_jsonl(work / "grading-sessions.jsonl", "E1 grading sessions")
        _emit_grading_checkpoint(
            prepared,
            grades,
            attempts,
            repaired_sessions,
            reason="interrupted-session-repaired",
            checkpoint_file=checkpoint,
        )
    sessions = _validate_sessions(
        work / "grading-sessions.jsonl",
        plan_identity=str(prepared.plan["plan_identity"]),
        allow_open=False,
    )
    session_index = max((int(row["session_index"]) for row in sessions), default=-1) + 1
    _session_event(
        work / "grading-sessions.jsonl",
        event="start",
        session_index=session_index,
        plan_identity=str(prepared.plan["plan_identity"]),
        details={"grades_at_start": len(grades)},
    )
    sessions = _read_jsonl(work / "grading-sessions.jsonl", "E1 grading sessions")
    _emit_grading_checkpoint(
        prepared, grades, attempts, sessions, reason="session-start", checkpoint_file=checkpoint
    )
    transport = (
        transport_factory()
        if transport_factory is not None
        else OpenRouterTransport(api_key=api_key)
    )
    new_grades = 0
    pending_error: str | None = None
    status = "error"
    try:
        while len(grades) < len(external) and (
            request_budget is None or new_grades < request_budget
        ):
            grade_sequence = len(grades)
            generation_sequence, condition, question, generation = external[grade_sequence]
            request = GradingRequest(
                question.question_id,
                question.text,
                question.aliases[0],
                str(generation["raw_output"]),
            )
            spec = specs[condition.benchmark]
            receipt_offset = len(transport.receipts)
            grade = run_openrouter_grader(spec, request, transport)
            invocation_receipts = transport.receipts[receipt_offset:]
            for receipt in invocation_receipts:
                attempt_body = {
                    "schema_version": 1,
                    "plan_identity": prepared.plan["plan_identity"],
                    "grade_sequence": grade_sequence,
                    "generation_sequence": generation_sequence,
                    "generation_record_digest": generation["record_digest"],
                    "grading_session_index": session_index,
                    "accepted_label": (grade.raw_response if receipt.error_type is None else None),
                    "receipt": receipt.to_dict(),
                    "previous_attempt_record_digest": (
                        attempts[-1]["attempt_record_digest"] if attempts else None
                    ),
                }
                attempt_row = {
                    **attempt_body,
                    "attempt_record_digest": stable_hash(attempt_body),
                }
                if api_key and api_key in json.dumps(attempt_row, ensure_ascii=False):
                    raise DataValidationError("refusing to persist an OpenRouter secret")
                _append_jsonl(work / "grader-attempts.jsonl", attempt_row)
                attempts.append(attempt_row)
                sessions = _read_jsonl(work / "grading-sessions.jsonl", "E1 grading sessions")
                _emit_grading_checkpoint(
                    prepared,
                    grades,
                    attempts,
                    sessions,
                    reason="attempt-appended",
                    checkpoint_file=checkpoint,
                )
            if grade.outcome is Outcome.UNSCORABLE:
                pending_error = grade.error or "unscorable official grade"
                status = "pending-provider-retry"
                break
            _append_successful_grade(
                path=work / "grades.jsonl",
                prepared=prepared,
                external=external,
                specs=specs,
                attempts=attempts,
                grades=grades,
                raw_label=grade.raw_response,
            )
            new_grades += 1
            sessions = _read_jsonl(work / "grading-sessions.jsonl", "E1 grading sessions")
            _emit_grading_checkpoint(
                prepared,
                grades,
                attempts,
                sessions,
                reason="grade-appended",
                checkpoint_file=checkpoint,
            )
            if len(grades) <= 5 or len(grades) % 25 == 0:
                print(
                    json.dumps(
                        {
                            "event": "e1-openrouter-progress",
                            "grades_completed": len(grades),
                            "grades_expected": len(external),
                            "benchmark": condition.benchmark,
                            "question_id": question.question_id,
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
        if status == "error":
            status = "complete" if len(grades) == len(external) else "partial"
    finally:
        _session_event(
            work / "grading-sessions.jsonl",
            event="end",
            session_index=session_index,
            plan_identity=str(prepared.plan["plan_identity"]),
            details={"grades_at_end": len(grades), "status": status},
        )
        sessions = _validate_sessions(
            work / "grading-sessions.jsonl",
            plan_identity=str(prepared.plan["plan_identity"]),
            allow_open=False,
        )
        _emit_grading_checkpoint(
            prepared, grades, attempts, sessions, reason="session-end", checkpoint_file=checkpoint
        )
    return {
        "complete": status == "complete",
        "status": status,
        "grades_completed": len(grades),
        "grades_expected": len(external),
        "attempts_recorded": len(attempts),
        "pending_error": pending_error,
        "resume_checkpoint": _grading_checkpoint(prepared, grades, attempts, sessions),
    }


def _e1_generation_record(
    *,
    sequence: int,
    prepared: E1Prepared,
    generation: Mapping[str, Any],
    grade: Mapping[str, Any] | None,
    grader_manifest_digest: str,
    specs: Mapping[str, OfficialGraderSpec],
) -> GenerationRecord:
    condition, question = prepared.schedule[sequence]
    raw_output = str(generation["raw_output"])
    metadata: dict[str, Any] = {
        "phase": ExperimentPhase.E1.value,
        "partition": condition.partition,
        "prompt_template_sha256": condition.prompt_template_sha256,
        "study_protocol_digest": condition.study_protocol_digest,
        "e1_plan_identity": prepared.plan["plan_identity"],
        "generation_sequence": sequence,
        "generation_record_digest": generation["record_digest"],
        "generation_request_digest": generation["request_digest"],
        "generated_token_ids_sha256": stable_hash(generation["token_ids"]),
        "raw_output_sha256": generation["raw_output_sha256"],
        "official_score_output_sha256": stable_hash(raw_output),
    }
    if condition.benchmark == "triviaqa":
        if grade is not None:
            raise DataValidationError("E1 TriviaQA records cannot contain model-grader evidence")
        outcome = deterministic_short_answer_grade(raw_output, question.aliases)
        exact_match, token_f1 = triviaqa_scores(raw_output, question.aliases)
        metadata.update(
            {
                "official_scorer": "mfh.triviaqa.alias-aware-em-f1.v1",
                "official_exact_match": exact_match,
                "official_token_f1": token_f1,
                "reference_aliases_digest": stable_hash(list(question.aliases)),
            }
        )
    else:
        if grade is None:
            raise DataValidationError("E1 model-graded record lacks an official grade")
        spec = specs[condition.benchmark]
        outcome = Outcome(str(grade["outcome"]))
        metadata.update(
            {
                "grader_attempts": int(grade["attempts"]),
                "grader_failed": False,
                "grader_request_fingerprint": grade["request_fingerprint"],
                "grader_fingerprint": grade["grader_fingerprint"],
                "grader_raw_label": grade["raw_label"],
                "grader_record_digest": grade["grade_record_digest"],
                "grader_attempt_record_digests": list(grade["attempt_record_digests"]),
                "grader_bundle_manifest_digest": grader_manifest_digest,
                "grader_model": spec.grader_model,
                "grader_model_revision": spec.grader_model_revision,
                "grader_source_artifact_sha256": spec.source_artifact_sha256,
            }
        )
    if outcome is Outcome.UNSCORABLE:
        raise DataValidationError("E1 finalization rejects unscorable outcomes")
    record = GenerationRecord(
        question_id=question.question_id,
        benchmark=condition.benchmark,
        model_repository=condition.model_repository,
        model_revision=condition.model_revision,
        runtime=condition.runtime,
        quantization=condition.quantization,
        system_prompt_id=condition.system_prompt_id,
        rendered_prompt_hash=str(generation["rendered_prompt_sha256"]),
        steering_method=condition.steering_method,
        layer=None,
        token_scope=None,
        alpha=0.0,
        sparsity=None,
        controller_scores={},
        raw_output=raw_output,
        normalized_answer=normalize_answer(raw_output),
        outcome=outcome,
        generation_latency_seconds=float(generation["latency_seconds"]),
        input_tokens=int(generation["input_tokens"]),
        output_tokens=int(generation["output_tokens"]),
        condition_id=condition.condition_id,
        site=None,
        seed=condition.seed,
        metadata=metadata,
    )
    condition.validate_record(record)
    return record


def _build_e1_records(
    *,
    prepared: E1Prepared,
    generations: Sequence[Mapping[str, Any]],
    grades: Sequence[Mapping[str, Any]],
    grader_manifest_digest: str,
    specs: Mapping[str, OfficialGraderSpec],
) -> tuple[GenerationRecord, ...]:
    grade_by_generation = {int(grade["generation_sequence"]): grade for grade in grades}
    records = tuple(
        _e1_generation_record(
            sequence=sequence,
            prepared=prepared,
            generation=generation,
            grade=grade_by_generation.get(sequence),
            grader_manifest_digest=grader_manifest_digest,
            specs=specs,
        )
        for sequence, generation in enumerate(generations)
    )
    if len(records) != _GENERATION_COUNT:
        raise DataValidationError("E1 final records differ from the generation schedule")
    return records


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _difference(left: object, right: object) -> float | None:
    if not isinstance(left, int | float) or not isinstance(right, int | float):
        return None
    return float(left) - float(right)


def _prompt_metrics(records: Sequence[GenerationRecord]) -> dict[str, Any]:
    grouped: dict[str, list[GenerationRecord]] = {}
    for record in records:
        grouped.setdefault(record.condition_id, []).append(record)
    conditions: list[dict[str, Any]] = []
    for condition_id in sorted(grouped):
        values = grouped[condition_id]
        counts = Counter(record.outcome.value for record in values)
        attempted = sum(counts[label] for label in ("C", "P", "I"))
        outcomes = tuple(record.outcome for record in values)
        if values[0].benchmark == "triviaqa":
            benchmark_metrics: Mapping[str, Any] = {
                "scorer": "alias-aware-em-f1",
                "exact_match": sum(
                    float(record.metadata["official_exact_match"]) for record in values
                )
                / len(values),
                "token_f1": sum(float(record.metadata["official_token_f1"]) for record in values)
                / len(values),
            }
        elif values[0].benchmark == "simpleqa_verified":
            benchmark_metrics = {
                "scorer": "simpleqa-verified-released-rubric-study-adapter",
                **simpleqa_official_metrics(outcomes).to_dict(),
            }
        else:
            benchmark_metrics = {
                "scorer": "aa-omniscience-released-rubric-stable-gemini-2.5-flash",
                **asdict(aa_official_metrics(outcomes)),
            }
        condition = {
            "condition_id": condition_id,
            "benchmark": values[0].benchmark,
            "prompt_id": values[0].system_prompt_id,
            "record_count": len(values),
            "outcome_counts": {label: counts[label] for label in ("C", "P", "I", "A", "U")},
            "coverage": _ratio(attempted, len(values)),
            "abstention_rate": _ratio(counts["A"], len(values)),
            "correct_rate": _ratio(counts["C"], len(values)),
            "partial_rate": _ratio(counts["P"], len(values)),
            "incorrect_attempt_rate": _ratio(counts["I"], len(values)),
            "hallucination_risk": _ratio(counts["I"], attempted),
            "accuracy_given_attempted": _ratio(counts["C"], attempted),
            "partial_credit_accuracy_given_attempted": (
                (counts["C"] + 0.5 * counts["P"]) / attempted if attempted else None
            ),
            "benchmark_metrics": benchmark_metrics,
        }
        conditions.append(condition)
    lookup = {(str(value["benchmark"]), str(value["prompt_id"])): value for value in conditions}
    response_lookup = {
        (record.benchmark, record.system_prompt_id, record.question_id): record.outcome.value
        for record in records
    }
    contrasts: list[dict[str, Any]] = []
    for benchmark in _BENCHMARK_FILES:
        neutral = lookup[(benchmark, "P0-neutral")]
        for prompt_id in ("P1-direct", "P2-calibrated-abstention"):
            prompted = lookup[(benchmark, prompt_id)]
            delta_abstention = float(prompted["abstention_rate"]) - float(
                neutral["abstention_rate"]
            )
            neutral_keys = sorted(
                question_id
                for candidate_benchmark, candidate_prompt, question_id in response_lookup
                if candidate_benchmark == benchmark and candidate_prompt == "P0-neutral"
            )
            prompted_keys = sorted(
                question_id
                for candidate_benchmark, candidate_prompt, question_id in response_lookup
                if candidate_benchmark == benchmark and candidate_prompt == prompt_id
            )
            if neutral_keys != prompted_keys or len(neutral_keys) != int(neutral["record_count"]):
                raise DataValidationError("E1 prompt contrast lacks exact question pairing")
            paired = [
                (
                    response_lookup[(benchmark, "P0-neutral", question_id)],
                    response_lookup[(benchmark, prompt_id, question_id)],
                )
                for question_id in neutral_keys
            ]
            transitions = Counter(f"{before}->{after}" for before, after in paired)
            neutral_correct = sum(before == "C" for before, _after in paired)
            neutral_incorrect = sum(before == "I" for before, _after in paired)
            neutral_abstention = sum(before == "A" for before, _after in paired)
            contrasts.append(
                {
                    "benchmark": benchmark,
                    "reference_prompt_id": "P0-neutral",
                    "prompt_id": prompt_id,
                    "coverage_change": float(prompted["coverage"]) - float(neutral["coverage"]),
                    "hallucination_risk_change": _difference(
                        prompted["hallucination_risk"],
                        neutral["hallucination_risk"],
                    ),
                    "abstention_rate_change": delta_abstention,
                    "incorrect_attempt_rate_change": float(prompted["incorrect_attempt_rate"])
                    - float(neutral["incorrect_attempt_rate"]),
                    "transition_counts": {
                        f"{before}->{after}": transitions[f"{before}->{after}"]
                        for before in ("C", "P", "I", "A")
                        for after in ("C", "P", "I", "A")
                    },
                    "strict_over_refusal": _ratio(transitions["C->A"], neutral_correct),
                    "regression": _ratio(transitions["C->I"], neutral_correct),
                    "correct_preservation": _ratio(transitions["C->C"], neutral_correct),
                    "abstention_substitution": _ratio(transitions["I->A"], neutral_incorrect),
                    "abstention_to_incorrect": _ratio(transitions["A->I"], neutral_abstention),
                }
            )
    risk_coverage_curves: list[dict[str, Any]] = []
    for benchmark in _BENCHMARK_FILES:
        points: list[dict[str, Any]] = sorted(
            (
                {
                    "prompt_id": prompt_id,
                    "coverage": float(lookup[(benchmark, prompt_id)]["coverage"]),
                    "hallucination_risk": lookup[(benchmark, prompt_id)]["hallucination_risk"],
                }
                for prompt_id in _PROMPTS
            ),
            key=lambda value: (value["coverage"], value["prompt_id"]),
        )
        risks = [value["hallucination_risk"] for value in points]
        observed_auc = (
            sum(
                (float(right["coverage"]) - float(left["coverage"]))
                * (float(left["hallucination_risk"]) + float(right["hallucination_risk"]))
                / 2
                for left, right in pairwise(points)
            )
            if all(isinstance(value, int | float) for value in risks)
            else None
        )
        risk_coverage_curves.append(
            {
                "benchmark": benchmark,
                "points": points,
                "observed_coverage_span_auc": observed_auc,
            }
        )
    body = {
        "schema_version": 1,
        "record_count": len(records),
        "condition_count": len(conditions),
        "conditions": conditions,
        "prompt_contrasts": contrasts,
        "risk_coverage_curves": risk_coverage_curves,
    }
    return {**body, "metrics_digest": stable_hash(body)}


def _outcome_label_rows(
    records: Sequence[GenerationRecord],
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    previous: str | None = None
    for sequence, record in enumerate(records):
        body = {
            "schema_version": 1,
            "sequence": sequence,
            "condition_id": record.condition_id,
            "question_id": record.question_id,
            "benchmark": record.benchmark,
            "partition": record.metadata["partition"],
            "prompt_id": record.system_prompt_id,
            "outcome": record.outcome.value,
            "generation_record_digest": record.metadata["generation_record_digest"],
            "ledger_record_digest": stable_hash(record.to_dict()),
            "grader_record_digest": record.metadata.get("grader_record_digest"),
            "previous_label_digest": previous,
        }
        row = {**body, "label_digest": stable_hash(body)}
        rows.append(row)
        previous = str(row["label_digest"])
    return tuple(rows)


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


def _e1_output_manifest_body(
    *,
    output: Path,
    work: Path,
    prepared: E1Prepared,
    completion_digest: str,
    record_set_digest: str,
    records: Sequence[GenerationRecord],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    work_fingerprints = {
        path.name: sha256_file(path) for path in sorted(work.iterdir()) if path.is_file()
    }
    return {
        "schema_version": 1,
        "purpose": "E1-baseline-records-prompt-metrics-and-outcome-labels",
        "phase": ExperimentPhase.E1.value,
        "plan_identity": prepared.plan["plan_identity"],
        "contract_digest": prepared.contract.digest,
        "completion_digest": completion_digest,
        "record_set_digest": record_set_digest,
        "record_count": len(records),
        "condition_count": len(prepared.conditions),
        "grader_bundle_manifest_digest": prepared.plan["grader_bundle_manifest_digest"],
        "work_fingerprints": work_fingerprints,
        "files": {
            "outcome_labels": {
                "path": "outcome-labels.jsonl",
                "sha256": sha256_file(output / "outcome-labels.jsonl"),
                "rows": len(records),
            },
            "prompt_metrics": {
                "path": "prompt-metrics.json",
                "sha256": sha256_file(output / "prompt-metrics.json"),
                "metrics_digest": metrics["metrics_digest"],
            },
        },
    }


def _write_e1_output_bundle(
    *,
    output: Path,
    work: Path,
    prepared: E1Prepared,
    completion_digest: str,
    record_set_digest: str,
    records: Sequence[GenerationRecord],
) -> Mapping[str, Any]:
    if output.exists():
        raise FrozenArtifactError(f"refusing to overwrite E1 output bundle: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        labels = _outcome_label_rows(records)
        metrics = _prompt_metrics(records)
        _write_jsonl_once(stage / "outcome-labels.jsonl", labels)
        _write_json_once(stage / "prompt-metrics.json", metrics)
        body = _e1_output_manifest_body(
            output=stage,
            work=work,
            prepared=prepared,
            completion_digest=completion_digest,
            record_set_digest=record_set_digest,
            records=records,
            metrics=metrics,
        )
        _write_json_once(stage / "manifest.json", {**body, "manifest_digest": stable_hash(body)})
        os.replace(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return _read_json(output / "manifest.json", "E1 output manifest")


def verify_e1_output_bundle(
    output_directory: str | Path,
    *,
    work_directory: str | Path,
    ledger_directory: str | Path,
    study_config: str | Path = "configs/experiments/phases.yaml",
    expected_manifest_digest: str | None = None,
) -> Mapping[str, Any]:
    """Recompute the E1 labels and prompt metrics from the completed ledger."""

    work = Path(work_directory).absolute()
    ledger_path = Path(ledger_directory).absolute()
    output = Path(output_directory).absolute()
    validate_active_study_artifact_paths(
        {
            "E1 work directory": work,
            "E1 ledger directory": ledger_path,
            "E1 output directory": output,
        }
    )
    if output.is_symlink() or not output.is_dir():
        raise FrozenArtifactError("E1 output bundle must be a regular directory")
    inventory = {path.name for path in output.iterdir()}
    if inventory != {"manifest.json", "outcome-labels.jsonl", "prompt-metrics.json"} or any(
        path.is_symlink() or not path.is_file() for path in output.iterdir()
    ):
        raise FrozenArtifactError("E1 output bundle inventory differs")
    manifest = _read_json(output / "manifest.json", "E1 output manifest")
    body = dict(manifest)
    digest = body.pop("manifest_digest", None)
    if digest != stable_hash(body) or (
        expected_manifest_digest is not None and digest != expected_manifest_digest
    ):
        raise FrozenArtifactError("E1 output manifest digest differs")
    study = load_study_protocol(study_config)
    ledger = PhaseRunLedger.open(ledger_path, study=study)
    completion = ledger.verify_complete()
    records = tuple(ledger.records())
    expected_labels = _outcome_label_rows(records)
    labels = tuple(_read_jsonl(output / "outcome-labels.jsonl", "E1 outcome labels"))
    metrics = _read_json(output / "prompt-metrics.json", "E1 prompt metrics")
    expected_metrics = _prompt_metrics(records)
    _verify_work_inventory(work)
    plan = _read_json(work / "plan.json", "E1 work plan")
    if plan.get("runner_source_sha256") != sha256_file(Path(__file__)):
        raise FrozenArtifactError("E1 output runner source differs from the frozen plan")
    expected_work_fingerprints = {
        path.name: sha256_file(path) for path in sorted(work.iterdir()) if path.is_file()
    }
    files = manifest.get("files")
    expected_body = {
        "schema_version": 1,
        "purpose": "E1-baseline-records-prompt-metrics-and-outcome-labels",
        "phase": ExperimentPhase.E1.value,
        "plan_identity": plan.get("plan_identity"),
        "contract_digest": ledger.contract.digest,
        "completion_digest": completion.completion_digest,
        "record_set_digest": completion.record_set_digest,
        "record_count": len(records),
        "condition_count": len(ledger.contract.conditions),
        "grader_bundle_manifest_digest": plan.get("grader_bundle_manifest_digest"),
        "work_fingerprints": expected_work_fingerprints,
        "files": files,
    }
    if (
        body != expected_body
        or labels != expected_labels
        or metrics != expected_metrics
        or not isinstance(files, Mapping)
        or files.get("outcome_labels")
        != {
            "path": "outcome-labels.jsonl",
            "sha256": sha256_file(output / "outcome-labels.jsonl"),
            "rows": len(records),
        }
        or files.get("prompt_metrics")
        != {
            "path": "prompt-metrics.json",
            "sha256": sha256_file(output / "prompt-metrics.json"),
            "metrics_digest": metrics.get("metrics_digest"),
        }
    ):
        raise FrozenArtifactError("E1 output bundle differs from the completed run")
    return manifest


def finalize_e1_vllm(
    *,
    splits_directory: str | Path,
    expected_splits_manifest_digest: str,
    grader_bundle: str | Path,
    expected_grader_manifest_digest: str,
    model_config: str | Path,
    snapshot_directory: str | Path,
    snapshot_manifest: str | Path,
    runtime_config: str | Path,
    work_directory: str | Path,
    ledger_directory: str | Path,
    output_directory: str | Path,
    e0_run: str | Path,
    prompt_config: str | Path = "configs/prompts/primary.yaml",
    inference_config: str | Path = "configs/experiments/core.yaml",
    study_config: str | Path = "configs/experiments/phases.yaml",
    checkpoint_batch_size: int = 250,
) -> Mapping[str, Any]:
    """Checkpoint the verified E1 evidence, pass both gates, and freeze outputs."""

    if (
        isinstance(checkpoint_batch_size, bool)
        or not isinstance(checkpoint_batch_size, int)
        or checkpoint_batch_size <= 0
    ):
        raise ConfigurationError("E1 finalization batch size must be positive")
    paths = {
        "splits_directory": Path(splits_directory).absolute(),
        "grader_bundle": Path(grader_bundle).absolute(),
        "model_config": Path(model_config).absolute(),
        "snapshot_directory": Path(snapshot_directory).absolute(),
        "snapshot_manifest": Path(snapshot_manifest).absolute(),
        "runtime_config": Path(runtime_config).absolute(),
        "prompt_config": Path(prompt_config).absolute(),
        "inference_config": Path(inference_config).absolute(),
        "study_config": Path(study_config).absolute(),
        "e0_run": Path(e0_run).absolute(),
    }
    prepared = _prepare(
        **paths,
        expected_splits_manifest_digest=expected_splits_manifest_digest,
        expected_grader_manifest_digest=expected_grader_manifest_digest,
    )
    work = Path(work_directory).absolute()
    ledger_path = Path(ledger_directory).absolute()
    output = Path(output_directory).absolute()
    validate_active_study_artifact_paths(
        {
            "E0 prerequisite run": paths["e0_run"],
            "E1 work directory": work,
            "E1 ledger directory": ledger_path,
            "E1 output directory": output,
        }
    )
    if output == work or output.is_relative_to(work) or output.is_relative_to(ledger_path):
        raise ConfigurationError("E1 outputs must stay outside work and ledger directories")
    _verify_work_inventory(work)
    if _read_json(work / "plan.json", "E1 work plan") != dict(prepared.plan):
        raise FrozenArtifactError("E1 work plan differs before finalization")
    renderer = _tokenizer_renderer(prepared)
    try:
        generations = _complete_generations(work, prepared=prepared, renderer=renderer)
    finally:
        renderer.close()
    external = _external_schedule(prepared, generations)
    grader_manifest = verify_e1_grader_bundle(
        paths["grader_bundle"],
        expected_manifest_digest=expected_grader_manifest_digest,
    )
    specs = _grader_specs(paths["grader_bundle"], grader_manifest)
    grading_sessions = _validate_sessions(
        work / "grading-sessions.jsonl",
        plan_identity=str(prepared.plan["plan_identity"]),
        allow_open=False,
    )
    grading_session_indices = {
        int(row["session_index"]) for row in grading_sessions if row["event"] == "start"
    }
    attempts = _validate_attempt_rows(
        work / "grader-attempts.jsonl",
        prepared=prepared,
        external=external,
        specs=specs,
    )
    if any(int(row["grading_session_index"]) not in grading_session_indices for row in attempts):
        raise DataValidationError("E1 grader attempt refers to an unknown session")
    grades = _validate_grades(
        work / "grades.jsonl",
        prepared=prepared,
        external=external,
        specs=specs,
        attempt_rows=attempts,
    )
    if len(grades) != len(external):
        raise DataValidationError(
            f"E1 finalization requires {len(external) - len(grades)} more official grades"
        )
    records = _build_e1_records(
        prepared=prepared,
        generations=generations,
        grades=grades,
        grader_manifest_digest=str(grader_manifest["manifest_digest"]),
        specs=specs,
    )
    ledger = PhaseRunLedger.open(ledger_path, study=prepared.study)
    if ledger.contract != prepared.contract:
        raise FrozenArtifactError("E1 ledger contract differs before finalization")
    if (ledger_path / "complete.json").exists():
        completion = ledger.verify_complete()
        if tuple(ledger.records()) != records:
            raise FrozenArtifactError("completed E1 ledger differs from source evidence")
    else:
        existing = tuple(ledger.records())
        if existing != records[: len(existing)]:
            raise FrozenArtifactError("partial E1 ledger is not a prefix of the frozen evidence")
        for offset in range(len(existing), len(records), checkpoint_batch_size):
            ledger.checkpoint(records[offset : offset + checkpoint_batch_size])
        with tempfile.TemporaryDirectory(prefix="mfh-e1-gates-") as gate_directory:
            gate_results = {}
            for gate in prepared.contract.required_gates:
                evidence_path = Path(gate_directory) / f"{gate}.json"
                write_gate_evidence(
                    evidence_path,
                    phase=ExperimentPhase.E1,
                    gate=gate,
                    contract_digest=prepared.contract.digest,
                    record_set_digest=ledger.record_set_digest(),
                    observations=(),
                )
                gate_results[gate] = ledger.evaluate_gate(gate, evidence_path)
            completion = ledger.finalize(gate_results)
        ledger.verify_complete()
    if output.exists():
        manifest = verify_e1_output_bundle(
            output,
            work_directory=work,
            ledger_directory=ledger_path,
            study_config=paths["study_config"],
        )
    else:
        manifest = _write_e1_output_bundle(
            output=output,
            work=work,
            prepared=prepared,
            completion_digest=completion.completion_digest,
            record_set_digest=completion.record_set_digest,
            records=records,
        )
        verify_e1_output_bundle(
            output,
            work_directory=work,
            ledger_directory=ledger_path,
            study_config=paths["study_config"],
            expected_manifest_digest=str(manifest["manifest_digest"]),
        )
    return {
        "complete": True,
        "completion_digest": completion.completion_digest,
        "record_set_digest": completion.record_set_digest,
        "record_count": completion.record_count,
        "output_manifest_digest": manifest["manifest_digest"],
        "output_directory": str(output),
    }

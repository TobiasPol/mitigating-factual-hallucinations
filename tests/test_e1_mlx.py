from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mfh.contracts import GenerationRecord, Outcome, Question, Runtime
from mfh.errors import ConfigurationError, DataValidationError
from mfh.evaluation.official import GradingRequest, load_official_grader_spec
from mfh.evaluation.openrouter import OpenRouterTransport, run_openrouter_grader
from mfh.experiments.e1_mlx import (
    _grading_checkpoint,
    _grading_checkpoint_matches,
    _outcome_label_rows,
    _prompt_metrics,
    _randomized_schedule,
    _recoverable_attempt_invocation,
    _resume_checkpoint,
    _resume_checkpoint_matches,
    _validate_attempt_rows,
    _validate_grades,
    load_env_secret,
)
from mfh.provenance import stable_hash

ROOT = Path(__file__).parents[1]
GRADERS = ROOT / "configs/graders"


def test_e1_schedule_is_deterministically_randomized_across_conditions() -> None:
    conditions = tuple(
        SimpleNamespace(benchmark="triviaqa", condition_id=f"condition-{index}")
        for index in range(4)
    )
    questions = {
        "triviaqa": tuple(
            Question(
                question_id=f"question-{index}",
                benchmark="triviaqa",
                text=f"Question {index}?",
                aliases=(str(index),),
            )
            for index in range(12)
        )
    }
    first = _randomized_schedule(conditions, questions)  # type: ignore[arg-type]
    second = _randomized_schedule(conditions, questions)  # type: ignore[arg-type]
    identities = tuple((row[0].condition_id, row[1].question_id) for row in first)
    assert first == second
    assert len(identities) == 48
    assert len(set(identities)) == 48
    assert len({condition_id for condition_id, _ in identities[:12]}) > 1


def _response(content: str = "A") -> bytes:
    return json.dumps(
        {
            "id": "gen-e1-test",
            "model": "openai/gpt-4.1",
            "provider": "OpenAI",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 1,
                "total_tokens": 11,
            },
        }
    ).encode()


def _grader_evidence(tmp_path: Path) -> tuple[Any, Any, list[dict[str, Any]], list[dict[str, Any]]]:
    spec = load_official_grader_spec(GRADERS / "simpleqa-verified.yaml")
    question = Question(
        question_id="simpleqa:test",
        benchmark="simpleqa_verified",
        text="Capital of France?",
        aliases=("Paris",),
    )
    condition = SimpleNamespace(
        benchmark="simpleqa_verified",
        condition_id="condition-test",
    )
    prepared = SimpleNamespace(
        plan={"plan_identity": "plan-test"},
        schedule=((condition, question),),
    )
    generation = {
        "raw_output": "Paris",
        "record_digest": "a" * 64,
    }
    external = ((0, condition, question, generation),)
    transport = OpenRouterTransport(
        api_key="test-key",
        sender=lambda _request, _timeout: (200, _response()),
    )
    request = GradingRequest(question.question_id, question.text, "Paris", "Paris")
    grade = run_openrouter_grader(spec, request, transport)
    attempts: list[dict[str, Any]] = []
    previous = None
    for receipt in transport.receipts:
        body = {
            "schema_version": 1,
            "plan_identity": "plan-test",
            "grade_sequence": 0,
            "generation_sequence": 0,
            "generation_record_digest": "a" * 64,
            "grading_session_index": 0,
            "accepted_label": grade.raw_response,
            "receipt": receipt.to_dict(),
            "previous_attempt_record_digest": previous,
        }
        row = {**body, "attempt_record_digest": stable_hash(body)}
        attempts.append(row)
        previous = row["attempt_record_digest"]
    grade_body = {
        "schema_version": 1,
        "plan_identity": "plan-test",
        "grade_sequence": 0,
        "generation_sequence": 0,
        "generation_record_digest": "a" * 64,
        "condition_id": condition.condition_id,
        "question_id": question.question_id,
        "benchmark": condition.benchmark,
        "request_fingerprint": grade.request_fingerprint,
        "grader_fingerprint": grade.grader_fingerprint,
        "raw_label": grade.raw_response,
        "outcome": grade.outcome.value,
        "attempts": grade.attempts,
        "attempt_record_digests": [value["attempt_record_digest"] for value in attempts],
        "previous_grade_record_digest": None,
    }
    grades = [{**grade_body, "grade_record_digest": stable_hash(grade_body)}]
    attempts_path = tmp_path / "attempts.jsonl"
    grades_path = tmp_path / "grades.jsonl"
    attempts_path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in attempts),
        encoding="utf-8",
    )
    grades_path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in grades),
        encoding="utf-8",
    )
    return prepared, external, attempts, grades


def test_e1_grader_chains_bind_exact_request_and_success(tmp_path: Path) -> None:
    prepared, external, _attempts, _grades = _grader_evidence(tmp_path)
    specs = {"simpleqa_verified": load_official_grader_spec(GRADERS / "simpleqa-verified.yaml")}
    attempts = _validate_attempt_rows(
        tmp_path / "attempts.jsonl",
        prepared=prepared,
        external=external,
        specs=specs,
    )
    grades = _validate_grades(
        tmp_path / "grades.jsonl",
        prepared=prepared,
        external=external,
        specs=specs,
        attempt_rows=attempts,
    )
    assert grades[0]["outcome"] == "C"


def test_e1_grader_chain_rejects_rehashed_request_tampering(tmp_path: Path) -> None:
    prepared, external, attempts, _grades = _grader_evidence(tmp_path)
    attempts[0]["receipt"]["request_sha256"] = "b" * 64
    body = dict(attempts[0])
    body.pop("attempt_record_digest")
    attempts[0]["attempt_record_digest"] = stable_hash(body)
    (tmp_path / "attempts.jsonl").write_text(
        json.dumps(attempts[0], sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(DataValidationError, match="attempt evidence"):
        _validate_attempt_rows(
            tmp_path / "attempts.jsonl",
            prepared=prepared,
            external=external,
            specs={
                "simpleqa_verified": load_official_grader_spec(GRADERS / "simpleqa-verified.yaml")
            },
        )


def test_e1_grade_rejects_unattached_success_receipt(tmp_path: Path) -> None:
    prepared, external, attempts, _grades = _grader_evidence(tmp_path)
    specs = {"simpleqa_verified": load_official_grader_spec(GRADERS / "simpleqa-verified.yaml")}
    validated = _validate_attempt_rows(
        tmp_path / "attempts.jsonl",
        prepared=prepared,
        external=external,
        specs=specs,
    )
    (tmp_path / "grades.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(DataValidationError, match="successful attempt"):
        _validate_grades(
            tmp_path / "grades.jsonl",
            prepared=prepared,
            external=external,
            specs=specs,
            attempt_rows=validated,
        )
    assert attempts[0]["receipt"]["content_sha256"] == hashlib.sha256(b"A").hexdigest()


def test_e1_success_receipt_is_recoverable_after_hard_crash(tmp_path: Path) -> None:
    prepared, external, _attempts, _grades = _grader_evidence(tmp_path)
    specs = {"simpleqa_verified": load_official_grader_spec(GRADERS / "simpleqa-verified.yaml")}
    attempts = _validate_attempt_rows(
        tmp_path / "attempts.jsonl",
        prepared=prepared,
        external=external,
        specs=specs,
    )
    (tmp_path / "grades.jsonl").write_text("", encoding="utf-8")
    grades = _validate_grades(
        tmp_path / "grades.jsonl",
        prepared=prepared,
        external=external,
        specs=specs,
        attempt_rows=attempts,
        allow_recoverable_success=True,
    )
    recoverable = _recoverable_attempt_invocation(attempts, grades)
    assert len(recoverable) == 1
    assert recoverable[0]["accepted_label"] == "A"


def test_eventual_grade_attaches_failed_attempts_from_prior_sessions(tmp_path: Path) -> None:
    prepared, external, attempts, grades = _grader_evidence(tmp_path)
    successful = attempts[0]
    error_body = json.dumps({"error": {"message": "rate limited"}}).encode()
    failed_receipt = dict(successful["receipt"])
    failed_receipt.update(
        {
            "http_status": 429,
            "response_sha256": hashlib.sha256(error_body).hexdigest(),
            "response_body_base64": base64.b64encode(error_body).decode("ascii"),
            "response_id": None,
            "returned_model": None,
            "returned_provider": None,
            "finish_reason": None,
            "content_sha256": None,
            "usage": {},
            "error_type": "OpenRouterError",
            "error_message": "OpenRouter HTTP 429: rate limited",
            "transient": True,
        }
    )
    failed_body = {
        **{key: value for key, value in successful.items() if key != "attempt_record_digest"},
        "accepted_label": None,
        "receipt": failed_receipt,
    }
    failed = {**failed_body, "attempt_record_digest": stable_hash(failed_body)}
    success_body = {
        **{key: value for key, value in successful.items() if key != "attempt_record_digest"},
        "grading_session_index": 1,
        "previous_attempt_record_digest": failed["attempt_record_digest"],
    }
    success = {**success_body, "attempt_record_digest": stable_hash(success_body)}
    grade_body = {
        **{key: value for key, value in grades[0].items() if key != "grade_record_digest"},
        "attempts": 2,
        "attempt_record_digests": [
            failed["attempt_record_digest"],
            success["attempt_record_digest"],
        ],
    }
    grade = {**grade_body, "grade_record_digest": stable_hash(grade_body)}
    (tmp_path / "attempts.jsonl").write_text(
        json.dumps(failed, sort_keys=True) + "\n" + json.dumps(success, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "grades.jsonl").write_text(
        json.dumps(grade, sort_keys=True) + "\n", encoding="utf-8"
    )
    specs = {"simpleqa_verified": load_official_grader_spec(GRADERS / "simpleqa-verified.yaml")}
    validated_attempts = _validate_attempt_rows(
        tmp_path / "attempts.jsonl",
        prepared=prepared,
        external=external,
        specs=specs,
    )
    validated_grades = _validate_grades(
        tmp_path / "grades.jsonl",
        prepared=prepared,
        external=external,
        specs=specs,
        attempt_rows=validated_attempts,
    )
    assert validated_grades[0]["attempts"] == 2


def test_external_checkpoints_accept_only_one_single_log_append_gap() -> None:
    prepared: Any = SimpleNamespace(plan={"plan_identity": "plan-test"})
    records = [{"record_digest": "a" * 64}, {"record_digest": "b" * 64}]
    sessions = [{"event_digest": "c" * 64}, {"event_digest": "d" * 64}]
    exact_generation = _resume_checkpoint(prepared, records, sessions)
    record_gap = _resume_checkpoint(prepared, records[:-1], sessions)
    session_gap = _resume_checkpoint(prepared, records, sessions[:-1])
    combined_generation_gap = _resume_checkpoint(prepared, records[:-1], sessions[:-1])
    assert _resume_checkpoint_matches(prepared, records, sessions, exact_generation)
    assert _resume_checkpoint_matches(prepared, records, sessions, record_gap)
    assert _resume_checkpoint_matches(prepared, records, sessions, session_gap)
    assert not _resume_checkpoint_matches(prepared, records, sessions, combined_generation_gap)
    assert not _resume_checkpoint_matches(prepared, records, sessions, "e" * 64)

    attempts = [{"attempt_record_digest": "f" * 64}]
    grades = [{"grade_record_digest": "1" * 64}]
    exact_grading = _grading_checkpoint(prepared, grades, attempts, sessions)
    grade_gap = _grading_checkpoint(prepared, grades[:-1], attempts, sessions)
    attempt_gap = _grading_checkpoint(prepared, grades, attempts[:-1], sessions)
    grading_session_gap = _grading_checkpoint(prepared, grades, attempts, sessions[:-1])
    combined_grading_gap = _grading_checkpoint(prepared, grades[:-1], attempts[:-1], sessions[:-1])
    assert _grading_checkpoint_matches(prepared, grades, attempts, sessions, exact_grading)
    assert _grading_checkpoint_matches(prepared, grades, attempts, sessions, grade_gap)
    assert _grading_checkpoint_matches(prepared, grades, attempts, sessions, attempt_gap)
    assert _grading_checkpoint_matches(prepared, grades, attempts, sessions, grading_session_gap)
    assert not _grading_checkpoint_matches(
        prepared, grades, attempts, sessions, combined_grading_gap
    )
    assert not _grading_checkpoint_matches(prepared, grades, attempts, sessions, "2" * 64)


def _record(benchmark: str, prompt: str, outcome: Outcome, index: int) -> GenerationRecord:
    return GenerationRecord(
        question_id=f"{benchmark}:q:{index}",
        benchmark=benchmark,
        model_repository="test/model",
        model_revision="a" * 40,
        runtime=Runtime.MLX,
        quantization="1bit",
        system_prompt_id=prompt,
        rendered_prompt_hash="b" * 64,
        steering_method="M0",
        layer=None,
        token_scope=None,
        alpha=0.0,
        sparsity=None,
        controller_scores={},
        raw_output="answer",
        normalized_answer="answer",
        outcome=outcome,
        generation_latency_seconds=1.0,
        input_tokens=10,
        output_tokens=1,
        condition_id=stable_hash({"benchmark": benchmark, "prompt": prompt}),
        seed=17,
        metadata={
            "partition": "test",
            "generation_record_digest": stable_hash(
                {"benchmark": benchmark, "prompt": prompt, "index": index}
            ),
            "official_exact_match": float(outcome is Outcome.CORRECT),
            "official_token_f1": float(outcome is Outcome.CORRECT),
        },
    )


def test_prompt_metrics_and_labels_report_prompt_only_changes() -> None:
    records: list[GenerationRecord] = []
    for benchmark in (
        "triviaqa",
        "simpleqa_verified",
        "aa_omniscience_public_600",
    ):
        records.extend(
            [
                _record(benchmark, "P0-neutral", Outcome.CORRECT, 0),
                _record(benchmark, "P1-direct", Outcome.INCORRECT, 0),
                _record(
                    benchmark,
                    "P2-calibrated-abstention",
                    Outcome.ABSTENTION,
                    0,
                ),
            ]
        )
    metrics = _prompt_metrics(records)
    p2 = next(
        value
        for value in metrics["prompt_contrasts"]
        if value["benchmark"] == "triviaqa" and value["prompt_id"] == "P2-calibrated-abstention"
    )
    assert p2["coverage_change"] == -1.0
    assert p2["abstention_rate_change"] == 1.0
    assert p2["strict_over_refusal"] == 1.0
    assert p2["transition_counts"]["C->A"] == 1
    simpleqa = next(
        value
        for value in metrics["conditions"]
        if value["benchmark"] == "simpleqa_verified" and value["prompt_id"] == "P0-neutral"
    )
    assert simpleqa["benchmark_metrics"]["simpleqa_f1"] == 1.0
    aa = next(
        value
        for value in metrics["conditions"]
        if value["benchmark"] == "aa_omniscience_public_600" and value["prompt_id"] == "P0-neutral"
    )
    assert aa["benchmark_metrics"]["omniscience_index"] == 100.0
    trivia = next(
        value
        for value in metrics["conditions"]
        if value["benchmark"] == "triviaqa" and value["prompt_id"] == "P0-neutral"
    )
    assert trivia["benchmark_metrics"]["exact_match"] == 1.0
    assert trivia["benchmark_metrics"]["token_f1"] == 1.0
    labels = _outcome_label_rows(records)
    assert labels[-1]["previous_label_digest"] == labels[-2]["label_digest"]
    tampered = replace(records[-1], outcome=Outcome.INCORRECT)
    assert _outcome_label_rows([*records[:-1], tampered])[-1] != labels[-1]


def test_strict_over_refusal_is_paired_not_a_marginal_rate_change() -> None:
    records: list[GenerationRecord] = []
    for benchmark in (
        "triviaqa",
        "simpleqa_verified",
        "aa_omniscience_public_600",
    ):
        records.extend(
            [
                _record(benchmark, "P0-neutral", Outcome.CORRECT, 0),
                _record(benchmark, "P0-neutral", Outcome.ABSTENTION, 1),
                _record(benchmark, "P1-direct", Outcome.CORRECT, 0),
                _record(benchmark, "P1-direct", Outcome.ABSTENTION, 1),
                _record(
                    benchmark,
                    "P2-calibrated-abstention",
                    Outcome.ABSTENTION,
                    0,
                ),
                _record(
                    benchmark,
                    "P2-calibrated-abstention",
                    Outcome.CORRECT,
                    1,
                ),
            ]
        )
    metrics = _prompt_metrics(records)
    p2 = next(
        value
        for value in metrics["prompt_contrasts"]
        if value["benchmark"] == "triviaqa" and value["prompt_id"] == "P2-calibrated-abstention"
    )
    assert p2["abstention_rate_change"] == 0.0
    assert p2["strict_over_refusal"] == 1.0
    assert p2["transition_counts"]["C->A"] == 1


def test_refuse_all_metrics_leave_attempt_conditionals_undefined() -> None:
    records = [
        _record(benchmark, prompt, Outcome.ABSTENTION, 0)
        for benchmark in (
            "triviaqa",
            "simpleqa_verified",
            "aa_omniscience_public_600",
        )
        for prompt in (
            "P0-neutral",
            "P1-direct",
            "P2-calibrated-abstention",
        )
    ]
    metrics = _prompt_metrics(records)
    assert all(value["coverage"] == 0.0 for value in metrics["conditions"])
    assert all(value["hallucination_risk"] is None for value in metrics["conditions"])
    assert all(value["accuracy_given_attempted"] is None for value in metrics["conditions"])
    assert all(
        value["observed_coverage_span_auc"] is None for value in metrics["risk_coverage_curves"]
    )


def test_env_secret_requires_one_exact_nonempty_assignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    source = tmp_path / ".env"
    source.write_text("OTHER=x\nexport OPENROUTER_API_KEY='secret'\n", encoding="utf-8")
    assert load_env_secret(source, "OPENROUTER_API_KEY") == "secret"
    source.write_text("OPENROUTER_API_KEY=x\nOPENROUTER_API_KEY=y\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="exactly once"):
        load_env_secret(source, "OPENROUTER_API_KEY")

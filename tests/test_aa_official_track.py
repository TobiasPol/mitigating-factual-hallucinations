from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

from mfh.contracts import Outcome, PromptSpec, Question, Runtime
from mfh.errors import FrozenArtifactError
from mfh.evaluation.official import load_official_grader_spec
from mfh.evaluation.openrouter import OpenRouterTransport
from mfh.experiments import aa_official_track
from mfh.experiments.aa_official_track import (
    AAOfficialContext,
    finalize_aa_official_track,
    load_aa_official_analysis,
    prepare_aa_official_track,
    run_aa_official_track,
    verify_aa_official_track,
)
from mfh.experiments.protocol import ExperimentPhase
from mfh.experiments.runner import EvaluationCondition
from mfh.inference.mlx_runtime import MlxGenerationOutput, MlxRenderedPrompt
from mfh.provenance import stable_hash

ROOT = Path(__file__).parents[1]


class _Runtime:
    def render_prompt(
        self,
        prompt: PromptSpec,
        question: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> MlxRenderedPrompt:
        system = prompt.text.format_map(metadata or {})
        text = f"system:{system}\nuser:{question}\nassistant:"
        token_ids = (1, 2, 3)
        return MlxRenderedPrompt(
            text=text,
            sha256=hashlib.sha256(text.encode()).hexdigest(),
            token_ids=token_ids,
            token_ids_sha256=stable_hash(list(token_ids)),
            messages=(
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ),
        )

    def generate(self, rendered: MlxRenderedPrompt, *, max_new_tokens: int) -> MlxGenerationOutput:
        assert max_new_tokens == 48
        return MlxGenerationOutput(
            rendered_prompt=rendered,
            token_ids=(4,),
            text="answer",
            input_tokens=len(rendered.token_ids),
            output_tokens=1,
            latency_seconds=0.01,
            stop_type="stop",
            stopping_token_id=4,
            prompt_tokens_per_second=10.0,
            generation_tokens_per_second=10.0,
            peak_memory_bytes=1024,
            active_memory_bytes=512,
            cache_memory_bytes=128,
        )

    def runtime_identity(self) -> dict[str, str]:
        return {"runtime": "test"}

    def close(self) -> None:
        return None


def _response() -> bytes:
    return json.dumps(
        {
            "id": "gen-aa-official",
            "model": "google/gemini-2.5-flash",
            "provider": "Google AI Studio",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "A"},
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        }
    ).encode()


def _invalid_response() -> bytes:
    value = json.loads(_response())
    value["choices"][0]["message"]["content"] = "NOT-A-RELEASED-LABEL"
    return json.dumps(value).encode()


def _raw_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "splits_directory": tmp_path / "splits",
        "grader_bundle": tmp_path / "graders",
        "model_config": tmp_path / "model.yaml",
        "snapshot_directory": tmp_path / "snapshot",
        "snapshot_manifest": tmp_path / "snapshot.json",
        "runtime_config": tmp_path / "runtime.json",
        "ledger_directory": tmp_path / "e1-ledger",
        "e0_run": tmp_path / "e0",
        "prompt_config": tmp_path / "prompts.yaml",
        "inference_config": tmp_path / "inference.yaml",
        "study_config": tmp_path / "study.yaml",
    }


def _fixture_context() -> tuple[AAOfficialContext, tuple[Any, ...], _Runtime]:
    prompt = PromptSpec(
        "P-AA-official",
        aa_official_track._OFFICIAL_PROMPT_TEXT,
    )
    questions = tuple(
        Question(
            question_id=f"aa-{index}",
            benchmark="aa_omniscience_public_600",
            text=f"Question {index}?",
            aliases=("answer",),
            split="public",
            metadata={"domain": "Science", "topic": "Physics"},
        )
        for index in range(2)
    )
    condition = EvaluationCondition(
        phase=ExperimentPhase.E1,
        benchmark="aa_omniscience_public_600",
        partition="aa-eval",
        model_name="model",
        model_repository="repository/model",
        model_revision="a" * 40,
        runtime=Runtime.MLX,
        quantization="4bit",
        model_num_layers=2,
        system_prompt_id=prompt.prompt_id,
        prompt_template_sha256=hashlib.sha256(prompt.text.encode()).hexdigest(),
        steering_method="M0",
        method_artifact_sha256=None,
        layer=None,
        site=None,
        token_scope=None,
        alpha=0.0,
        sparsity=None,
        seed=17,
        study_protocol_digest="b" * 64,
        comparison_group="aa-official-auxiliary",
    )
    schedule = aa_official_track._official_schedule(condition, questions)
    grader = load_official_grader_spec(ROOT / "configs/graders/aa-omniscience-public.yaml")
    plan_body = {
        "schema_version": 1,
        "phase": "E1-AA-official-auxiliary",
        "track": "AA-Omniscience-Public-600 official answerer prompt and scoring",
        "runner_source_sha256": "1" * 64,
        "e1_runner_source_sha256": "2" * 64,
        "study_protocol_digest": "3" * 64,
        "e1_contract_digest": "4" * 64,
        "e1_completion_digest": "d" * 64,
        "e1_plan_identity": "5" * 64,
        "model": {
            "name": "model",
            "repository": "repository/model",
            "revision": "a" * 40,
            "runtime": "mlx",
            "quantization": "4bit",
            "num_layers": 2,
        },
        "condition": condition.to_dict(),
        "prompt": {
            "prompt_id": prompt.prompt_id,
            "text": prompt.text,
            "text_sha256": hashlib.sha256(prompt.text.encode()).hexdigest(),
            "permits_abstention": True,
            "deployment_eligible": True,
        },
        "question_count": len(questions),
        "questions": [aa_official_track._question_body(value) for value in questions],
        "question_fingerprints": {
            value.question_id: stable_hash(aa_official_track._question_body(value))
            for value in questions
        },
        "schedule": {
            "ordering": "sha256-rank-randomized-across-questions-v1",
            "seed": 17,
            "schedule_digest": stable_hash(
                [[value.condition_id, question.question_id] for value, question in schedule]
            ),
        },
        "inference": {
            "temperature": 0,
            "sampling": False,
            "thinking_enabled": False,
            "max_new_tokens": 48,
        },
        "input_hashes": {
            "reviewed_splits": "6" * 64,
            "grader_bundle": "7" * 64,
            "model_config": "8" * 64,
            "snapshot_manifest": "9" * 64,
            "runtime_config": "a" * 64,
            "prompt_config": "b" * 64,
            "inference_config": "c" * 64,
            "study_config": "d" * 64,
        },
        "grader": {
            "schema_version": grader.schema_version,
            "benchmark": grader.benchmark,
            "source_repository": grader.source_repository,
            "source_revision": grader.source_revision,
            "source_artifact": grader.source_artifact,
            "source_artifact_sha256": grader.source_artifact_sha256,
            "grader_model": grader.grader_model,
            "grader_model_revision": grader.grader_model_revision,
            "temperature": grader.temperature,
            "reasoning_enabled": grader.reasoning_enabled,
            "prompt_template": grader.prompt_template,
            "prompt_sha256": grader.prompt_sha256,
            "label_mapping": {
                label: outcome.value for label, outcome in grader.label_mapping.items()
            },
            "maximum_attempts": grader.maximum_attempts,
            "failure_outcome": grader.failure_outcome.value,
            "bundle_manifest_digest": "c" * 64,
            "grader_digest": grader.digest,
        },
    }
    plan = MappingProxyType({**plan_body, "plan_identity": stable_hash(plan_body)})
    prepared = SimpleNamespace(
        model=SimpleNamespace(name="model"),
        snapshot=Path("snapshot"),
        prompts={prompt.prompt_id: prompt},
        conditions=(condition,),
        plan=plan,
        schedule=schedule,
        max_new_tokens=48,
        study=SimpleNamespace(),
    )
    context = AAOfficialContext(
        prepared=prepared,  # type: ignore[arg-type]
        grader=grader,
        grader_manifest_digest="c" * 64,
        e1_completion_digest="d" * 64,
        plan=plan,
    )
    neutral = tuple(
        SimpleNamespace(
            question_id=question.question_id,
            benchmark=question.benchmark,
            system_prompt_id="P0-neutral",
            steering_method="M0",
            outcome=Outcome.CORRECT,
        )
        for question in questions
    )
    return context, neutral, _Runtime()


def test_aa_official_track_runs_replays_and_compares_to_neutral(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, neutral, runtime = _fixture_context()
    monkeypatch.setattr(aa_official_track, "_ROW_COUNT", 2)
    monkeypatch.setattr(aa_official_track, "_context", lambda **_kwargs: context)
    monkeypatch.setattr(
        aa_official_track, "_validate_runtime_identity", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(aa_official_track, "_tokenizer_renderer", lambda _prepared: runtime)
    monkeypatch.setattr(
        aa_official_track.PhaseRunLedger,
        "open",
        lambda *_args, **_kwargs: SimpleNamespace(records=lambda: iter(neutral)),
    )
    paths = _raw_paths(tmp_path)
    work = tmp_path / "official-work"
    prepared = prepare_aa_official_track(
        work,
        expected_splits_manifest_digest="e" * 64,
        expected_grader_manifest_digest="f" * 64,
        **paths,
    )
    assert prepared["records_expected"] == 2
    checkpoint = tmp_path / "checkpoint.json"
    result = run_aa_official_track(
        work,
        expected_splits_manifest_digest="e" * 64,
        expected_grader_manifest_digest="f" * 64,
        api_key="test-key",
        checkpoint_file=checkpoint,
        runtime_factory=lambda _model, _snapshot: runtime,
        transport_factory=lambda: OpenRouterTransport(
            api_key="test-key", sender=lambda _request, _timeout: (200, _response())
        ),
        **paths,
    )
    assert result["complete"] is True
    assert result["records_completed"] == 2

    output = tmp_path / "official-result"
    verified = finalize_aa_official_track(
        output,
        work_directory=work,
        expected_splits_manifest_digest="e" * 64,
        expected_grader_manifest_digest="f" * 64,
        **paths,
    )
    assert verified["record_count"] == 2
    assert verified["official_vs_neutral"]["paired_question_count"] == 2
    assert verified["official_vs_neutral"]["leaderboard_comparability"] == {
        "official_track": True,
        "neutral_controlled_track": False,
    }
    portable = load_aa_official_analysis(
        output,
        expected_manifest_digest=str(verified["manifest_digest"]),
        expected_e1_completion_digest="d" * 64,
    )
    assert portable["record_set_digest"]
    assert portable["official_vs_neutral"]["paired_question_count"] == 2

    comparison = output / "official-vs-neutral.json"
    payload = json.loads(comparison.read_text(encoding="utf-8"))
    payload["paired_question_count"] = 1
    comparison.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FrozenArtifactError, match="exact replay"):
        verify_aa_official_track(
            output,
            expected_manifest_digest=str(verified["manifest_digest"]),
            expected_splits_manifest_digest="e" * 64,
            expected_grader_manifest_digest="f" * 64,
            **paths,
        )


def test_portable_grader_rejects_boolean_type_coercion() -> None:
    context, _neutral, _runtime = _fixture_context()
    value = dict(context.plan["grader"])
    value["reasoning_enabled"] = "false"

    with pytest.raises(FrozenArtifactError, match="labels are invalid"):
        aa_official_track._portable_grader(value)


def test_unscorable_attempts_are_chained_outside_scorable_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context, neutral, runtime = _fixture_context()
    monkeypatch.setattr(aa_official_track, "_ROW_COUNT", 2)
    monkeypatch.setattr(aa_official_track, "_context", lambda **_kwargs: context)
    monkeypatch.setattr(
        aa_official_track, "_validate_runtime_identity", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(aa_official_track, "_tokenizer_renderer", lambda _prepared: runtime)
    monkeypatch.setattr(
        aa_official_track.PhaseRunLedger,
        "open",
        lambda *_args, **_kwargs: SimpleNamespace(records=lambda: iter(neutral)),
    )
    run_grader = aa_official_track.run_openrouter_grader
    monkeypatch.setattr(
        aa_official_track,
        "run_openrouter_grader",
        lambda spec, request, transport: run_grader(
            spec, request, transport, sleeper=lambda _seconds: None
        ),
    )
    paths = _raw_paths(tmp_path)
    work = tmp_path / "failure-work"
    prepare_aa_official_track(
        work,
        expected_splits_manifest_digest="e" * 64,
        expected_grader_manifest_digest="f" * 64,
        **paths,
    )
    checkpoint = tmp_path / "failure-checkpoint.json"
    failed = run_aa_official_track(
        work,
        expected_splits_manifest_digest="e" * 64,
        expected_grader_manifest_digest="f" * 64,
        api_key="test-key",
        checkpoint_file=checkpoint,
        runtime_factory=lambda _model, _snapshot: runtime,
        transport_factory=lambda: OpenRouterTransport(
            api_key="test-key",
            sender=lambda _request, _timeout: (200, _invalid_response()),
        ),
        **paths,
    )
    assert failed["complete"] is False
    assert failed["records_completed"] == 0
    assert failed["failures_recorded"] == 1
    assert not (work / "records.jsonl").exists()
    failures = [json.loads(value) for value in (work / "failures.jsonl").read_text().splitlines()]
    assert len(failures) == 1
    assert len(failures[0]["grader_receipts"]) == 3
    assert all(receipt["response_body_base64"] for receipt in failures[0]["grader_receipts"])
    checkpoint_value = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert checkpoint_value["failures_recorded"] == 1

    resumed = run_aa_official_track(
        work,
        expected_splits_manifest_digest="e" * 64,
        expected_grader_manifest_digest="f" * 64,
        api_key="test-key",
        checkpoint_file=checkpoint,
        expected_resume_checkpoint=checkpoint_value["resume_checkpoint"],
        runtime_factory=lambda _model, _snapshot: runtime,
        transport_factory=lambda: OpenRouterTransport(
            api_key="test-key", sender=lambda _request, _timeout: (200, _response())
        ),
        **paths,
    )
    assert resumed["complete"] is True
    assert resumed["records_completed"] == 2
    assert resumed["failures_recorded"] == 1

    output = tmp_path / "failure-result"
    verified = finalize_aa_official_track(
        output,
        work_directory=work,
        expected_splits_manifest_digest="e" * 64,
        expected_grader_manifest_digest="f" * 64,
        **paths,
    )
    assert verified["record_count"] == 2
    assert verified["failure_count"] == 1
    portable = load_aa_official_analysis(
        output,
        expected_manifest_digest=str(verified["manifest_digest"]),
        expected_e1_completion_digest="d" * 64,
    )
    assert portable["failure_count"] == 1

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from mfh.contracts import (
    ModelSpec,
    Outcome,
    PromptSpec,
    Question,
    Runtime,
)
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.experiments.activation_store import verify_activation_store
from mfh.experiments.e2_capture import (
    E1P0Source,
    prepare_e2_capture_work,
    run_e2_capture,
    verify_e2_capture_work,
)
from mfh.experiments.e2_schedule import (
    E2CaptureProtocol,
    build_e2_schedule,
    write_e2_workspace,
)
from mfh.inference.mlx_research import MlxPromptFeatureCubeOutput
from mfh.inference.mlx_runtime import MlxGenerationOutput, MlxRenderedPrompt
from mfh.provenance import stable_hash


def _question(question_id: str, benchmark: str) -> Question:
    return Question(
        question_id=question_id,
        benchmark=benchmark,
        text=f"Question {question_id}?",
        aliases=(f"answer-{question_id}",),
    )


def _inputs() -> tuple[
    E2CaptureProtocol,
    tuple[Question, ...],
    tuple[Question, ...],
    tuple[Question, ...],
    tuple[Question, ...],
]:
    protocol = E2CaptureProtocol(
        controller_rows=4,
        controller_calibration_rows=2,
        dev_rows=3,
        simpleqa_rows=2,
        aa_rows=2,
    )
    return (
        protocol,
        tuple(_question(f"controller-{index}", "triviaqa") for index in range(4)),
        tuple(_question(f"dev-{index}", "triviaqa") for index in range(3)),
        tuple(_question(f"simple-{index}", "simpleqa_verified") for index in range(2)),
        tuple(
            _question(f"aa-{index}", "aa_omniscience_public_600")
            for index in range(2)
        ),
    )


class _Runtime:
    def __init__(self) -> None:
        self.generation_calls = 0
        self.forward_calls = 0

    def render_prompt(self, prompt, question, *, metadata=None):  # type: ignore[no-untyped-def]
        assert metadata == {}
        text = f"{prompt.prompt_id}:{question}"
        token_ids = (1, 2)
        return MlxRenderedPrompt(
            text=text,
            sha256=hashlib.sha256(text.encode()).hexdigest(),
            token_ids=token_ids,
            token_ids_sha256=hashlib.sha256(b"1,2").hexdigest(),
            messages=(),
        )

    def generate(self, rendered, *, max_new_tokens):  # type: ignore[no-untyped-def]
        assert max_new_tokens == 48
        self.generation_calls += 1
        return MlxGenerationOutput(
            rendered_prompt=rendered,
            token_ids=(3,),
            text="wrong.",
            input_tokens=2,
            output_tokens=1,
            latency_seconds=0.1,
            stop_type="short_answer",
            stopping_token_id=3,
            prompt_tokens_per_second=2.0,
            generation_tokens_per_second=1.0,
            peak_memory_bytes=10,
            active_memory_bytes=8,
            cache_memory_bytes=2,
        )

    def prompt_feature_cube(self, rendered, *, layers, sites):  # type: ignore[no-untyped-def]
        self.forward_calls += 1
        return MlxPromptFeatureCubeOutput(
            activations={
                site: {
                    layer: np.full((1, 4), layer + site_index, dtype=np.float32)
                    for layer in layers
                }
                for site_index, site in enumerate(sites)
            },
            maximum_token_probability=0.75,
            output_entropy=0.5,
            peak_memory_bytes=10,
        )

    def runtime_identity(self):  # type: ignore[no-untyped-def]
        return {"runtime": "fake-mlx", "seed": 17}


class _FailRuntime(_Runtime):
    def __init__(self, fail_on_forward: int) -> None:
        super().__init__()
        self.fail_on_forward = fail_on_forward

    def prompt_feature_cube(self, rendered, *, layers, sites):  # type: ignore[no-untyped-def]
        if self.forward_calls + 1 == self.fail_on_forward:
            self.forward_calls += 1
            raise RuntimeError("simulated capture interruption")
        return super().prompt_feature_cube(  # type: ignore[no-untyped-call]
            rendered, layers=layers, sites=sites
        )


class _DifferentRuntime(_Runtime):
    def runtime_identity(self):  # type: ignore[no-untyped-def]
        return {"runtime": "fake-mlx", "seed": 18}


def _prepared(root: Path):  # type: ignore[no-untyped-def]
    protocol, controller, dev, simpleqa, aa = _inputs()
    frozen = (*controller, *simpleqa, *aa)
    outcomes = {
        (question.benchmark, question.question_id): Outcome.CORRECT
        for question in frozen
    }
    schedule = build_e2_schedule(
        controller=controller,
        dev=dev,
        simpleqa=simpleqa,
        aa=aa,
        e1_p0_outcomes=outcomes,
        protocol=protocol,
    )
    model = ModelSpec(
        name="bonsai-test",
        repository="prism-ml/Bonsai-27B-mlx-1bit",
        revision="e" * 40,
        runtime=Runtime.MLX,
        quantization="binary-g128-mlx-1bit",
        num_layers=64,
    )
    workspace = write_e2_workspace(
        root / "workspace",
        schedule=schedule,
        protocol=protocol,
        model=model,
        hidden_width=4,
        input_fingerprints={"e1": "1" * 64, "splits": "2" * 64},
    )
    questions = {
        (question.benchmark, question.question_id): question
        for question in (*controller, *dev, *simpleqa, *aa)
    }
    sources = {
        (question.benchmark, question.question_id): E1P0Source(
            benchmark=question.benchmark,
            question_id=question.question_id,
            outcome=Outcome.CORRECT,
            generation_record_sha256=hashlib.sha256(
                question.question_id.encode()
            ).hexdigest(),
        )
        for question in frozen
    }
    prompts = {
        value.prompt_id: value
        for value in (
            PromptSpec("P0-neutral", "Neutral"),
            PromptSpec("P3-forced-answer", "Forced", permits_abstention=False),
        )
    }
    prepare_e2_capture_work(
        root / "capture",
        workspace=workspace,
        questions=questions,
        prompts=prompts,
        e1_sources=sources,
        expected_runtime_identity={"runtime": "fake-mlx", "seed": 17},
        shard_rows=4,
    )
    return workspace, questions, prompts, sources


def test_e2_capture_resumes_without_regenerating_and_completes_cube_store() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        workspace, questions, prompts, sources = _prepared(root)
        first_runtime = _Runtime()
        first = run_e2_capture(
            root / "capture",
            workspace=workspace,
            questions=questions,
            prompts=prompts,
            e1_sources=sources,
            runtime=first_runtime,
            request_budget=5,
        )
        assert first["status"] == "partial"
        assert first["rows_completed"] == 5
        assert first_runtime.forward_calls == 5
        assert first_runtime.generation_calls == 1

        second_runtime = _Runtime()
        second = run_e2_capture(
            root / "capture",
            workspace=workspace,
            questions=questions,
            prompts=prompts,
            e1_sources=sources,
            runtime=second_runtime,
        )
        assert second["complete"] is True
        assert second["rows_completed"] == 18
        assert second_runtime.forward_calls == 13
        assert second_runtime.generation_calls == 9
        verified = verify_activation_store(
            workspace.directory / "activations",
            expected_spec=workspace.activation_spec,
            require_complete=True,
        )
        assert verified.rows_completed == 18
        assert verified.shard_count == 6


def test_e2_capture_rejects_resolution_chain_tampering() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        workspace, questions, prompts, sources = _prepared(root)
        run_e2_capture(
            root / "capture",
            workspace=workspace,
            questions=questions,
            prompts=prompts,
            e1_sources=sources,
            runtime=_Runtime(),
            request_budget=2,
        )
        path = root / "capture" / "resolutions.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[1]["previous_resolution_digest"] = "f" * 64
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        with pytest.raises(FrozenArtifactError, match="resolution chain"):
            run_e2_capture(
                root / "capture",
                workspace=workspace,
                questions=questions,
                prompts=prompts,
                e1_sources=sources,
                runtime=_Runtime(),
            )


def test_e2_capture_recovers_arbitrary_resolution_prefix_after_buffered_crash() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        workspace, questions, prompts, sources = _prepared(root)
        interrupted = _FailRuntime(fail_on_forward=6)
        with pytest.raises(RuntimeError, match="simulated"):
            run_e2_capture(
                root / "capture",
                workspace=workspace,
                questions=questions,
                prompts=prompts,
                e1_sources=sources,
                runtime=interrupted,
            )
        assert interrupted.generation_calls == 2
        assert verify_activation_store(
            workspace.directory / "activations",
            expected_spec=workspace.activation_spec,
        ).rows_completed == 4
        assert len(
            (root / "capture" / "resolutions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ) == 6

        resumed = _Runtime()
        result = run_e2_capture(
            root / "capture",
            workspace=workspace,
            questions=questions,
            prompts=prompts,
            e1_sources=sources,
            runtime=resumed,
        )
        assert result["complete"] is True
        assert resumed.generation_calls == 8


def test_e2_capture_binds_mapping_objects_and_runtime_session_chain() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        workspace, questions, prompts, sources = _prepared(root)
        source_key = next(iter(sources))
        wrong_sources = dict(sources)
        original = wrong_sources[source_key]
        wrong_sources[source_key] = E1P0Source(
            benchmark="wrong-benchmark",
            question_id="wrong-question",
            outcome=original.outcome,
            generation_record_sha256=original.generation_record_sha256,
        )
        with pytest.raises(DataValidationError, match="source keys"):
            prepare_e2_capture_work(
                root / "wrong-sources",
                workspace=workspace,
                questions=questions,
                prompts=prompts,
                e1_sources=wrong_sources,
                expected_runtime_identity={"runtime": "fake-mlx", "seed": 17},
            )
        wrong_prompts = dict(prompts)
        wrong_prompts["P0-neutral"] = PromptSpec("wrong-prompt", "Neutral")
        with pytest.raises(DataValidationError, match="prompt mapping keys"):
            prepare_e2_capture_work(
                root / "wrong-prompts",
                workspace=workspace,
                questions=questions,
                prompts=wrong_prompts,
                e1_sources=sources,
                expected_runtime_identity={"runtime": "fake-mlx", "seed": 17},
            )

        run_e2_capture(
            root / "capture",
            workspace=workspace,
            questions=questions,
            prompts=prompts,
            e1_sources=sources,
            runtime=_Runtime(),
            request_budget=2,
        )
        with pytest.raises(FrozenArtifactError, match="live runtime identity"):
            run_e2_capture(
                root / "capture",
                workspace=workspace,
                questions=questions,
                prompts=prompts,
                e1_sources=sources,
                runtime=_DifferentRuntime(),
            )
        sessions = root / "capture" / "sessions.jsonl"
        events = [
            json.loads(line) for line in sessions.read_text(encoding="utf-8").splitlines()
        ]
        events[0]["runtime_identity"]["seed"] = 999
        sessions.write_text(
            "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8"
        )
        with pytest.raises(FrozenArtifactError, match="session event chain"):
            run_e2_capture(
                root / "capture",
                workspace=workspace,
                questions=questions,
                prompts=prompts,
                e1_sources=sources,
                runtime=_Runtime(),
            )


def test_e2_capture_recovers_valid_unclosed_hard_crash_session() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        workspace, questions, prompts, sources = _prepared(root)
        run_e2_capture(
            root / "capture",
            workspace=workspace,
            questions=questions,
            prompts=prompts,
            e1_sources=sources,
            runtime=_Runtime(),
            request_budget=2,
        )
        plan = json.loads((root / "capture" / "plan.json").read_text(encoding="utf-8"))
        sessions_path = root / "capture" / "sessions.jsonl"
        sessions = [
            json.loads(line)
            for line in sessions_path.read_text(encoding="utf-8").splitlines()
        ]
        body = {
            "schema_version": 1,
            "event": "start",
            "session_index": 1,
            "capture_plan_identity": plan["capture_plan_identity"],
            "rows_at_start": 2,
            "runtime_identity": {"runtime": "fake-mlx", "seed": 17},
            "created_unix_ns": time.time_ns(),
            "previous_session_event_digest": sessions[-1]["session_event_digest"],
        }
        sessions_path.write_text(
            sessions_path.read_text(encoding="utf-8")
            + json.dumps({**body, "session_event_digest": stable_hash(body)})
            + "\n",
            encoding="utf-8",
        )
        result = run_e2_capture(
            root / "capture",
            workspace=workspace,
            questions=questions,
            prompts=prompts,
            e1_sources=sources,
            runtime=_Runtime(),
            request_budget=1,
        )
        assert result["rows_completed"] == 3
        recovered = [
            json.loads(line)
            for line in sessions_path.read_text(encoding="utf-8").splitlines()
        ]
        assert recovered[3]["status"] == "interrupted-recovered"


def test_e2_capture_preserves_a_valid_final_event_missing_only_newline() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        workspace, questions, prompts, sources = _prepared(root)
        run_e2_capture(
            root / "capture",
            workspace=workspace,
            questions=questions,
            prompts=prompts,
            e1_sources=sources,
            runtime=_Runtime(),
            request_budget=2,
        )
        sessions_path = root / "capture" / "sessions.jsonl"
        original = sessions_path.read_bytes()
        assert original.endswith(b"\n")
        original_end = json.loads(original.splitlines()[-1])
        sessions_path.write_bytes(original[:-1])

        run_e2_capture(
            root / "capture",
            workspace=workspace,
            questions=questions,
            prompts=prompts,
            e1_sources=sources,
            runtime=_Runtime(),
            request_budget=1,
        )
        events = [
            json.loads(line)
            for line in sessions_path.read_text(encoding="utf-8").splitlines()
        ]
        assert original_end["session_event_digest"] in {
            event["session_event_digest"] for event in events
        }
        assert all(event.get("status") != "interrupted-recovered" for event in events)
        replay = verify_e2_capture_work(
            root / "capture",
            workspace=workspace,
            questions=questions,
            prompts=prompts,
            e1_sources=sources,
        )
        assert replay["rows_completed"] == 3

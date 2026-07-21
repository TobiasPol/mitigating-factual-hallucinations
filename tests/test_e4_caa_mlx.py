from __future__ import annotations

import hashlib
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from mfh.contracts import ActivationSite, Outcome, PromptSpec, Question
from mfh.errors import FrozenArtifactError
from mfh.experiments import e4_caa_mlx
from mfh.experiments.e3_construction import VerifiedE3ConstructionSnapshot
from mfh.experiments.e4_caa_mlx import (
    finalize_m2_caa_artifact,
    prepare_m2_caa_work,
    run_m2_caa_work,
    verify_m2_caa_artifact,
    verify_m2_caa_work,
)
from mfh.inference.mlx_runtime import MlxRenderedPrompt
from mfh.provenance import stable_hash
from tests.e4_test_artifacts import active_qwen_runtime_identity


class _Generation:
    def __init__(
        self,
        *,
        sequence: int,
        question_id: str,
        rendered: MlxRenderedPrompt,
        outcome: Outcome,
        raw_output: str,
    ) -> None:
        self.sequence = sequence
        self.question_id = question_id
        self.prompt_id = "P0-neutral"
        self.rendered_prompt_sha256 = rendered.sha256
        self.prompt_token_ids_sha256 = rendered.token_ids_sha256
        self.schedule_row_sha256 = stable_hash((sequence, question_id))
        self.outcome = outcome
        self.evidence = MappingProxyType({"raw_output": raw_output})

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "question_id": self.question_id,
            "prompt_id": self.prompt_id,
            "rendered_prompt_sha256": self.rendered_prompt_sha256,
            "prompt_token_ids_sha256": self.prompt_token_ids_sha256,
            "schedule_row_sha256": self.schedule_row_sha256,
            "outcome": self.outcome.value,
            "evidence": dict(self.evidence),
        }


class _Runtime:
    def __init__(
        self,
        identity: dict[str, Any],
        prompts: dict[str, MlxRenderedPrompt],
        *,
        peak_memory_bytes: int = 1024,
    ) -> None:
        self.identity = identity
        self.prompts = prompts
        self.peak_memory_bytes = peak_memory_bytes

    def runtime_identity(self) -> dict[str, Any]:
        return dict(self.identity)

    def render_prompt(
        self,
        prompt: PromptSpec,
        question: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> MlxRenderedPrompt:
        del prompt, metadata
        return self.prompts[question]

    def teacher_forced_cube(
        self,
        rendered: MlxRenderedPrompt,
        response: str,
        *,
        layers: tuple[int, ...],
        sites: tuple[ActivationSite, ...],
    ) -> SimpleNamespace:
        del rendered
        assert sites == (ActivationSite.BLOCK_OUTPUT,)
        positive = response.startswith("gold")
        base = 2.0 if positive else -1.0
        vector = np.full(5_120, -base, dtype=np.float32)
        vector[0] = base
        vector[1] = base / 2.0
        activations = {
            ActivationSite.BLOCK_OUTPUT: {
                layer: np.asarray([vector + layer / 100.0], dtype=np.float32)
                for layer in layers
            }
        }
        token_ids = (11,) if positive else (12,)
        token_digest = hashlib.sha256(
            ",".join(str(value) for value in token_ids).encode("ascii")
        ).hexdigest()
        return SimpleNamespace(
            response_text_sha256=hashlib.sha256(response.encode()).hexdigest(),
            response_token_ids=token_ids,
            response_token_ids_sha256=token_digest,
            activations=activations,
            peak_memory_bytes=self.peak_memory_bytes,
        )


def _rendered(question: Question) -> MlxRenderedPrompt:
    text = f"system:neutral\nuser:{question.text}\nassistant:"
    token_ids = (1, 2, int(question.question_id.rsplit("-", 1)[1]) + 3)
    return MlxRenderedPrompt(
        text=text,
        sha256=hashlib.sha256(text.encode()).hexdigest(),
        token_ids=token_ids,
        token_ids_sha256=hashlib.sha256(
            ",".join(str(value) for value in token_ids).encode("ascii")
        ).hexdigest(),
        messages=(),
    )


def test_native_mlx_caa_work_resumes_finalizes_and_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    questions = tuple(
        Question(
            question_id=f"steer-{index}",
            benchmark="triviaqa",
            text=f"Question {index}?",
            aliases=(f"gold-{index}",),
            split="T-steer",
        )
        for index in range(2)
    )
    rendered = {value.text: _rendered(value) for value in questions}
    identity = active_qwen_runtime_identity()
    generations = (
        _Generation(
            sequence=0,
            question_id=questions[0].question_id,
            rendered=rendered[questions[0].text],
            outcome=Outcome.INCORRECT,
            raw_output="wrong-0",
        ),
        _Generation(
            sequence=1,
            question_id=questions[1].question_id,
            rendered=rendered[questions[1].text],
            outcome=Outcome.CORRECT,
            raw_output="gold-1",
        ),
    )
    snapshot = VerifiedE3ConstructionSnapshot(
        directory=tmp_path / "e3",
        plan=MappingProxyType(
            {
                "plan_identity": "a" * 64,
                "runtime_identity": identity,
                "hidden_width": 5_120,
            }
        ),
        schedule=(SimpleNamespace(), SimpleNamespace()),  # type: ignore[arg-type]
        generations=generations,  # type: ignore[arg-type]
        generation_chain_head="b" * 64,
        scientific_eligible=True,
    )
    source = tmp_path / "e3"
    source.mkdir()
    (source / "source.txt").write_text("frozen E3", encoding="utf-8")
    monkeypatch.setattr(
        e4_caa_mlx,
        "load_verified_e3_construction_snapshot",
        lambda *_args, **_kwargs: snapshot,
    )
    prompts = {
        "P0-neutral": PromptSpec(
            "P0-neutral", "You are a helpful assistant. Answer the factual question."
        )
    }
    work = tmp_path / "m2-work"
    prepared = prepare_m2_caa_work(
        work,
        construction_directory=source,
        questions=questions,
        prompts=prompts,
    )
    assert prepared["pairs_expected"] == 1
    result = run_m2_caa_work(
        work,
        construction_directory=source,
        questions=questions,
        prompts=prompts,
        runtime=_Runtime(identity, rendered),
        request_budget=1,
    )
    assert result["complete"] is True
    assert result["pairs_processed"] == 1
    assert (
        verify_m2_caa_work(
            work,
            construction_directory=source,
            questions=questions,
            prompts=prompts,
            require_complete=True,
        )["scientific_eligible"]
        is True
    )

    output = tmp_path / "m2-artifact"
    artifact = finalize_m2_caa_artifact(
        output,
        work_directory=work,
        construction_directory=source,
        questions=questions,
        prompts=prompts,
    )
    assert artifact.pair_count == 1
    assert artifact.site is ActivationSite.BLOCK_OUTPUT
    assert (
        verify_m2_caa_artifact(output, expected_manifest_digest=artifact.manifest_digest)
        == artifact
    )

    with (output / "vectors.safetensors").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(FrozenArtifactError):
        verify_m2_caa_artifact(output, expected_manifest_digest=artifact.manifest_digest)


def test_interrupted_over_budget_capture_cannot_be_erased_by_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    question = Question(
        question_id="steer-0",
        benchmark="triviaqa",
        text="Question 0?",
        aliases=("gold-0",),
        split="T-steer",
    )
    rendered = {question.text: _rendered(question)}
    identity = active_qwen_runtime_identity()
    snapshot = VerifiedE3ConstructionSnapshot(
        directory=tmp_path / "e3",
        plan=MappingProxyType(
            {
                "plan_identity": "a" * 64,
                "runtime_identity": identity,
                "hidden_width": 5_120,
            }
        ),
        schedule=(SimpleNamespace(),),  # type: ignore[arg-type]
        generations=(
            _Generation(
                sequence=0,
                question_id=question.question_id,
                rendered=rendered[question.text],
                outcome=Outcome.INCORRECT,
                raw_output="wrong-0",
            ),
        ),  # type: ignore[arg-type]
        generation_chain_head="b" * 64,
        scientific_eligible=True,
    )
    source = tmp_path / "e3"
    source.mkdir()
    (source / "source.txt").write_text("frozen E3", encoding="utf-8")
    monkeypatch.setattr(
        e4_caa_mlx,
        "load_verified_e3_construction_snapshot",
        lambda *_args, **_kwargs: snapshot,
    )
    prompts = {
        "P0-neutral": PromptSpec(
            "P0-neutral", "You are a helpful assistant. Answer the factual question."
        )
    }
    work = tmp_path / "m2-work"
    prepare_m2_caa_work(
        work,
        construction_directory=source,
        questions=(question,),
        prompts=prompts,
    )
    original_session_event = e4_caa_mlx._session_event

    def interrupt_end(*args: Any, **kwargs: Any) -> None:
        if kwargs.get("event") == "end":
            raise RuntimeError("simulated kill before session end")
        original_session_event(*args, **kwargs)

    monkeypatch.setattr(e4_caa_mlx, "_session_event", interrupt_end)
    with pytest.raises(RuntimeError, match="simulated kill"):
        run_m2_caa_work(
            work,
            construction_directory=source,
            questions=(question,),
            prompts=prompts,
            runtime=_Runtime(
                identity,
                rendered,
                peak_memory_bytes=48 * 1024**3 + 1,
            ),
        )
    with pytest.raises(FrozenArtifactError, match="unclosed session"):
        verify_m2_caa_work(
            work,
            construction_directory=source,
            questions=(question,),
            prompts=prompts,
        )

    monkeypatch.setattr(e4_caa_mlx, "_session_event", original_session_event)
    resumed = run_m2_caa_work(
        work,
        construction_directory=source,
        questions=(question,),
        prompts=prompts,
        runtime=_Runtime(identity, rendered),
    )
    assert resumed["complete"] is True
    assert resumed["maximum_peak_memory_bytes"] == 48 * 1024**3 + 1
    assert resumed["scientific_eligible"] is False
    with pytest.raises(FrozenArtifactError, match="not scientifically eligible"):
        finalize_m2_caa_artifact(
            tmp_path / "m2-artifact",
            work_directory=work,
            construction_directory=source,
            questions=(question,),
            prompts=prompts,
        )

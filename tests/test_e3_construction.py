from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mfh.contracts import ActivationSite, PromptSpec, Question, TokenScope
from mfh.errors import FrozenArtifactError
from mfh.experiments.e3_construction import (
    finalize_e3_vector_bundle,
    load_verified_e3_construction_snapshot,
    prepare_e3_construction_work,
    run_e3_construction,
    verify_e3_construction_work,
    verify_e3_vector_bundle,
)
from mfh.experiments.e3_schedule import E3Protocol
from mfh.inference.mlx_research import (
    MlxPromptFeatureCubeOutput,
    MlxTeacherForcedCubeOutput,
)
from mfh.inference.mlx_runtime import MlxGenerationOutput, MlxRenderedPrompt
from mfh.provenance import sha256_file, stable_hash


def _token_digest(token_ids: Sequence[int]) -> str:
    return hashlib.sha256(",".join(str(value) for value in token_ids).encode()).hexdigest()


def _protocol() -> E3Protocol:
    return E3Protocol(
        steer_rows=4,
        dev_rows=6,
        screen_rows=2,
        candidate_layers=(1, 2),
        candidate_sites=(ActivationSite.POST_MLP,),
        standardized_alphas=(0.0, 0.5),
        token_scopes=(TokenScope.FINAL_PROMPT, TokenScope.FIRST_FOUR),
    )


def _questions() -> tuple[Question, ...]:
    return tuple(
        Question(
            question_id=f"q-{index}",
            benchmark="triviaqa",
            text=f"Question {index}?",
            aliases=(f"answer-{index}",),
            split="T-steer",
        )
        for index in range(4)
    )


def _prompts() -> dict[str, PromptSpec]:
    return {
        value: PromptSpec(value, f"System prompt {value}")
        for value in ("P0-neutral", "P2-calibrated-abstention")
    }


class _FakeRuntime:
    def __init__(self, *, fail_capture_once: bool = False) -> None:
        self.generate_calls = 0
        self.fail_capture_once = fail_capture_once

    def runtime_identity(self) -> Mapping[str, Any]:
        return {"runtime": "fake-mlx", "revision": "a" * 40}

    def render_prompt(
        self,
        prompt: PromptSpec,
        question: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> MlxRenderedPrompt:
        del metadata
        text = f"{prompt.prompt_id}|{question}"
        index = int(question.split()[1].rstrip("?"))
        tokens = (100 + index, 200 + len(prompt.prompt_id))
        return MlxRenderedPrompt(
            text=text,
            sha256=hashlib.sha256(text.encode()).hexdigest(),
            token_ids=tokens,
            token_ids_sha256=_token_digest(tokens),
            messages=(),
        )

    def generate(
        self, rendered: MlxRenderedPrompt, *, max_new_tokens: int
    ) -> MlxGenerationOutput:
        assert max_new_tokens == 8
        self.generate_calls += 1
        index = int(rendered.text.split("Question ")[1].rstrip("?"))
        text = f"answer-{index}" if index % 2 == 0 else "wrong"
        token_ids = (300 + index,)
        return MlxGenerationOutput(
            rendered_prompt=rendered,
            token_ids=token_ids,
            text=text,
            input_tokens=len(rendered.token_ids),
            output_tokens=1,
            latency_seconds=0.1,
            stop_type="short_answer",
            stopping_token_id=token_ids[-1],
            prompt_tokens_per_second=10.0,
            generation_tokens_per_second=5.0,
            peak_memory_bytes=1024,
            active_memory_bytes=512,
            cache_memory_bytes=256,
        )

    def _base(self, rendered: MlxRenderedPrompt) -> float:
        index = int(rendered.text.split("Question ")[1].rstrip("?"))
        return 3.0 if index % 2 == 0 else 1.0

    def prompt_feature_cube(
        self,
        rendered: MlxRenderedPrompt,
        *,
        layers: Sequence[int],
        sites: Sequence[ActivationSite],
    ) -> MlxPromptFeatureCubeOutput:
        if self.fail_capture_once:
            self.fail_capture_once = False
            raise RuntimeError("simulated interruption after generation")
        base = self._base(rendered)
        return MlxPromptFeatureCubeOutput(
            activations={
                site: {
                    layer: np.asarray([[base + layer, 1.0, 0.5]], dtype=np.float32)
                    for layer in layers
                }
                for site in sites
            },
            maximum_token_probability=0.75,
            output_entropy=0.5,
            peak_memory_bytes=2048,
        )

    def teacher_forced_cube(
        self,
        rendered: MlxRenderedPrompt,
        response: str,
        *,
        layers: Sequence[int],
        sites: Sequence[ActivationSite],
    ) -> MlxTeacherForcedCubeOutput:
        base = self._base(rendered) + 1.0
        token_ids = (900,)
        return MlxTeacherForcedCubeOutput(
            response_text_sha256=hashlib.sha256(response.encode()).hexdigest(),
            response_token_ids=token_ids,
            response_token_ids_sha256=_token_digest(token_ids),
            token_log_probabilities=(-0.5,),
            negative_log_likelihood=0.5,
            mean_negative_log_likelihood=0.5,
            perplexity=math.exp(0.5),
            activations={
                site: {
                    layer: np.asarray([[base + layer, 1.5, 0.25]], dtype=np.float32)
                    for layer in layers
                }
                for site in sites
            },
            peak_memory_bytes=4096,
        )


def _prepare(path: Path, runtime: _FakeRuntime) -> None:
    prepare_e3_construction_work(
        path,
        questions=_questions(),
        prompts=_prompts(),
        runtime_identity=runtime.runtime_identity(),
        hidden_width=3,
        protocol=_protocol(),
        checkpoint_rows=2,
        max_new_tokens=8,
    )


def test_e3_construction_resumes_without_regenerating_and_publishes_vectors(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    runtime = _FakeRuntime(fail_capture_once=True)
    _prepare(work, runtime)

    with pytest.raises(RuntimeError, match="simulated"):
        run_e3_construction(
            work,
            questions=_questions(),
            prompts=_prompts(),
            runtime=runtime,
            protocol=_protocol(),
            request_budget=1,
        )
    assert runtime.generate_calls == 1
    partial = verify_e3_construction_work(
        work, questions=_questions(), prompts=_prompts(), protocol=_protocol()
    )
    assert partial["rows_generated"] == 1
    assert partial["rows_processed"] == 0

    resumed = run_e3_construction(
        work,
        questions=_questions(),
        prompts=_prompts(),
        runtime=runtime,
        protocol=_protocol(),
        request_budget=3,
    )
    assert resumed["rows_processed"] == 3
    assert runtime.generate_calls == 3
    complete = run_e3_construction(
        work,
        questions=_questions(),
        prompts=_prompts(),
        runtime=runtime,
        protocol=_protocol(),
    )
    assert complete["complete"] is True
    assert complete["scientific_eligible"] is False
    assert runtime.generate_calls == 8
    snapshot = load_verified_e3_construction_snapshot(
        work, questions=_questions(), prompts=_prompts(), protocol=_protocol()
    )
    assert len(snapshot.generations) == 8
    assert snapshot.generation_chain_head == complete["generation_chain_head"]
    assert snapshot.scientific_eligible is False
    with pytest.raises(TypeError):
        snapshot.plan["runtime_identity"]["runtime"] = "substituted"
    with pytest.raises(TypeError):
        snapshot.plan["protocol"]["candidate_layers"][0] = 63

    bundle = tmp_path / "vectors"
    with pytest.raises(FrozenArtifactError, match="not eligible"):
        finalize_e3_vector_bundle(
            bundle,
            work_directory=work,
            questions=_questions(),
            prompts=_prompts(),
            protocol=_protocol(),
        )
    result = finalize_e3_vector_bundle(
        bundle,
        work_directory=work,
        questions=_questions(),
        prompts=_prompts(),
        protocol=_protocol(),
        allow_non_scientific=True,
    )
    assert result["vector_count"] == 8
    verified = verify_e3_vector_bundle(
        bundle,
        work_directory=work,
        questions=_questions(),
        prompts=_prompts(),
        protocol=_protocol(),
    )
    assert verified == result
    with np.load(bundle / "vectors.npz", allow_pickle=False) as values:
        expected = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        assert np.allclose(values["directions"], expected)


def test_e3_construction_rejects_checkpoint_and_bundle_tampering(tmp_path: Path) -> None:
    work = tmp_path / "work"
    runtime = _FakeRuntime()
    _prepare(work, runtime)
    run_e3_construction(
        work,
        questions=_questions(),
        prompts=_prompts(),
        runtime=runtime,
        protocol=_protocol(),
    )
    checkpoint = sorted((work / "checkpoints").iterdir())[0]
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
    with pytest.raises(FrozenArtifactError):
        verify_e3_construction_work(
            work,
            questions=_questions(),
            prompts=_prompts(),
            protocol=_protocol(),
            require_complete=True,
        )


def test_e3_vector_verifier_replays_checkpoint_arrays(tmp_path: Path) -> None:
    work = tmp_path / "work"
    runtime = _FakeRuntime()
    _prepare(work, runtime)
    run_e3_construction(
        work,
        questions=_questions(),
        prompts=_prompts(),
        runtime=runtime,
        protocol=_protocol(),
    )
    bundle = tmp_path / "vectors"
    finalize_e3_vector_bundle(
        bundle,
        work_directory=work,
        questions=_questions(),
        prompts=_prompts(),
        protocol=_protocol(),
        allow_non_scientific=True,
    )
    tensor_path = bundle / "vectors.npz"
    with np.load(tensor_path, allow_pickle=False) as values:
        shapes = {name: values[name].shape for name in values.files}
    with tensor_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            directions=np.broadcast_to(
                np.asarray([0.0, 1.0, 0.0], dtype=np.float32), shapes["directions"]
            ).copy(),
            reference_rms=np.full(shapes["reference_rms"], 999.0, dtype=np.float64),
            correct_counts=np.full(shapes["correct_counts"], 123, dtype=np.int64),
            incorrect_counts=np.full(shapes["incorrect_counts"], 456, dtype=np.int64),
        )
    metadata_path = bundle / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["vectors_sha256"] = sha256_file(tensor_path)
    body = dict(metadata)
    body.pop("metadata_digest")
    metadata["metadata_digest"] = stable_hash(body)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(FrozenArtifactError, match="geometry, counts, or RMS"):
        verify_e3_vector_bundle(
            bundle,
            work_directory=work,
            questions=_questions(),
            prompts=_prompts(),
            protocol=_protocol(),
        )


def test_e3_runner_repairs_torn_jsonl_and_orphan_checkpoint_temp(tmp_path: Path) -> None:
    work = tmp_path / "work"
    runtime = _FakeRuntime(fail_capture_once=True)
    _prepare(work, runtime)
    with pytest.raises(RuntimeError):
        run_e3_construction(
            work,
            questions=_questions(),
            prompts=_prompts(),
            runtime=runtime,
            protocol=_protocol(),
            request_budget=1,
        )
    with (work / "generations.jsonl").open("ab") as handle:
        handle.write(b'{"torn":')
    with (work / "sessions.jsonl").open("ab") as handle:
        handle.write(b'{"torn":')
    (work / "checkpoints" / ".checkpoint-orphan").write_bytes(b"partial")

    result = run_e3_construction(
        work,
        questions=_questions(),
        prompts=_prompts(),
        runtime=runtime,
        protocol=_protocol(),
        request_budget=1,
    )

    assert result["rows_generated"] == 1
    assert result["rows_processed"] == 1
    assert runtime.generate_calls == 1
    assert not (work / "checkpoints" / ".checkpoint-orphan").exists()


def test_e3_checkpoint_failure_records_only_durable_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mfh.experiments import e3_construction

    work = tmp_path / "work"
    runtime = _FakeRuntime()
    _prepare(work, runtime)
    original = e3_construction._write_checkpoint

    def fail_checkpoint(*args: Any, **kwargs: Any) -> tuple[Path, str]:
        del args, kwargs
        raise OSError("simulated checkpoint failure")

    monkeypatch.setattr(e3_construction, "_write_checkpoint", fail_checkpoint)
    with pytest.raises(OSError, match="checkpoint"):
        run_e3_construction(
            work,
            questions=_questions(),
            prompts=_prompts(),
            runtime=runtime,
            protocol=_protocol(),
            request_budget=1,
        )
    monkeypatch.setattr(e3_construction, "_write_checkpoint", original)

    partial = verify_e3_construction_work(
        work, questions=_questions(), prompts=_prompts(), protocol=_protocol()
    )
    assert partial["rows_processed"] == 0
    assert partial["rows_generated"] == 1
    resumed = run_e3_construction(
        work,
        questions=_questions(),
        prompts=_prompts(),
        runtime=runtime,
        protocol=_protocol(),
        request_budget=1,
    )
    assert resumed["rows_processed"] == 1
    assert runtime.generate_calls == 1

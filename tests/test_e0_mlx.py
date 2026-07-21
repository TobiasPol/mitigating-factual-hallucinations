from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import mfh.experiments.e0_mlx as e0_module
from mfh.config import load_model_spec, load_prompt_specs
from mfh.contracts import Question
from mfh.errors import DataValidationError
from mfh.experiments.e0_mlx import run_mlx_e0, verify_mlx_e0_bundle
from mfh.experiments.runner import EvaluationCondition
from mfh.inference.mlx_runtime import MlxGenerationOutput, MlxRenderedPrompt

ROOT = Path(__file__).parents[1]


class _FakeRuntime:
    def __init__(self) -> None:
        self.closed = False

    def render_prompt(self, prompt, question, *, metadata=None):  # type: ignore[no-untyped-def]
        index = int(question.rsplit(" ", 1)[-1])
        text = f"system={prompt.prompt_id};question={question}"
        token_ids = (1, index + 2)
        return MlxRenderedPrompt(
            text=text,
            sha256=hashlib.sha256(text.encode()).hexdigest(),
            token_ids=token_ids,
            token_ids_sha256=hashlib.sha256(f"1,{index + 2}".encode()).hexdigest(),
            messages=(
                {"role": "system", "content": prompt.text},
                {"role": "user", "content": question},
            ),
        )

    def generate(self, rendered, *, max_new_tokens):  # type: ignore[no-untyped-def]
        assert max_new_tokens == 48
        index = rendered.token_ids[-1] - 2
        text = f"answer-{index}"
        return MlxGenerationOutput(
            rendered_prompt=rendered,
            token_ids=(100 + index, 0),
            text=text,
            input_tokens=len(rendered.token_ids),
            output_tokens=2,
            latency_seconds=0.01,
            stop_type="stop",
            stopping_token_id=0,
            prompt_tokens_per_second=100.0,
            generation_tokens_per_second=50.0,
            peak_memory_bytes=1024,
            active_memory_bytes=512,
            cache_memory_bytes=128,
        )

    def runtime_identity(self):  # type: ignore[no-untyped-def]
        return {
            "backend": "mlx",
            "mlx": "test",
            "mlx_lm": "test",
            "python": "test-python",
            "machine_model": "test-machine",
            "chip": "test-chip",
            "unified_memory_bytes": 1024,
            "physical_cpu_cores": 1,
            "architecture": "test-arch",
            "os": "test-os",
            "os_build": "test-build",
            "model_class": "mlx_lm.models.qwen3_5.Model",
            "tokenizer_class": "mlx_lm.tokenizer_utils.TokenizerWrapper",
            "num_layers": 64,
            "seed": 17,
        }

    def close(self) -> None:
        self.closed = True


class _DriftedRuntime(_FakeRuntime):
    def runtime_identity(self):  # type: ignore[no-untyped-def]
        identity = dict(super().runtime_identity())
        identity["mlx_lm"] = "wrong-version"
        return identity


def _prepared() -> e0_module._Prepared:
    model = load_model_spec(ROOT / "configs/models/qwen3.6-27b-mlx-4bit.yaml")
    prompt = {
        value.prompt_id: value
        for value in load_prompt_specs(ROOT / "configs/prompts/primary.yaml")
    }["P0-neutral"]
    questions = tuple(
        Question(
            question_id=f"q-{index}",
            benchmark="triviaqa",
            text=f"question {index}",
            aliases=(f"answer-{index}",),
            split="runtime-validation",
        )
        for index in range(500)
    )
    condition = EvaluationCondition(
        phase=e0_module.ExperimentPhase.E0,
        benchmark="shared_benign_factual_500",
        partition="runtime-validation",
        model_name=model.name,
        model_repository=model.repository,
        model_revision=model.revision,
        runtime=model.runtime,
        quantization=model.quantization,
        model_num_layers=model.num_layers,
        system_prompt_id=prompt.prompt_id,
        prompt_template_sha256="a" * 64,
        steering_method="M0",
        method_artifact_sha256=None,
        layer=None,
        site=None,
        token_scope=None,
        alpha=0.0,
        sparsity=None,
        seed=17,
        study_protocol_digest="b" * 64,
    )
    return e0_module._Prepared(
        questions=questions,
        cohort_manifest={"manifest_digest": "c" * 64},
        model=model,
        snapshot=ROOT,
        snapshot_identity={"snapshot_digest": "d" * 64},
        runtime_config={
            "receipt_digest": "e" * 64,
            "runtime_identity": _FakeRuntime().runtime_identity(),
        },
        prompt=prompt,
        max_new_tokens=48,
        condition=condition,
        amendment_digest="f" * 64,
        input_hashes={"test": "1" * 64},
    )


def _arguments(tmp_path: Path) -> dict[str, Any]:
    return {
        "cohort_directory": tmp_path / "cohort",
        "reserved_source": tmp_path / "reserved.jsonl",
        "expected_cohort_manifest_digest": "2" * 64,
        "parent_split_manifest_digest": "3" * 64,
        "contamination_manifest_digest": "4" * 64,
        "model_config": tmp_path / "model.yaml",
        "snapshot_directory": tmp_path / "snapshot",
        "snapshot_manifest": tmp_path / "snapshot.json",
        "runtime_config": tmp_path / "runtime.json",
        "prompt_config": tmp_path / "prompts.yaml",
        "inference_config": tmp_path / "inference.yaml",
        "study_config": tmp_path / "phases.yaml",
    }


def test_mlx_e0_resumes_freezes_and_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared()
    monkeypatch.setattr(e0_module, "_prepare", lambda **_kwargs: prepared)
    monkeypatch.setattr(
        e0_module,
        "verify_transformers_snapshot",
        lambda *_args, **_kwargs: prepared.snapshot_identity,
    )
    monkeypatch.setattr(
        e0_module,
        "validate_active_study_artifact_paths",
        lambda *_args, **_kwargs: {},
    )
    work = tmp_path / "work"
    output = tmp_path / "output"
    checkpoint = tmp_path / "resume.json"
    args = _arguments(tmp_path)

    partial = run_mlx_e0(
        **args,
        work_directory=work,
        output_directory=output,
        checkpoint_file=checkpoint,
        request_budget=3,
        runtime_factory=lambda *_args: _FakeRuntime(),
    )
    assert partial["complete"] is False
    assert partial["records_completed"] == 3
    external = json.loads(checkpoint.read_text())["resume_checkpoint"]

    complete = run_mlx_e0(
        **args,
        work_directory=work,
        output_directory=output,
        checkpoint_file=checkpoint,
        expected_resume_checkpoint=external,
        request_budget=997,
        runtime_factory=lambda *_args: _FakeRuntime(),
    )
    assert complete["complete"] is True
    assert complete["records_completed"] == 1_000
    assert complete["summary"]["determinism_mismatches"] == 0

    verified = verify_mlx_e0_bundle(
        output,
        expected_manifest_digest=complete["manifest_digest"],
        expected_plan_identity=complete["plan_identity"],
        **args,
        renderer_factory=lambda _prepared: _FakeRuntime(),
    )
    assert verified["scientific_status"]["e0_runtime_validation_complete"] is True

    records = output / "records.jsonl"
    records.chmod(0o644)
    records.write_text(records.read_text() + "{}\n")
    with pytest.raises(DataValidationError):
        verify_mlx_e0_bundle(
            output,
            expected_manifest_digest=complete["manifest_digest"],
            expected_plan_identity=complete["plan_identity"],
            **args,
            renderer_factory=lambda _prepared: _FakeRuntime(),
        )


def test_mlx_e0_rejects_live_runtime_version_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepared()
    monkeypatch.setattr(e0_module, "_prepare", lambda **_kwargs: prepared)
    monkeypatch.setattr(
        e0_module,
        "verify_transformers_snapshot",
        lambda *_args, **_kwargs: prepared.snapshot_identity,
    )
    monkeypatch.setattr(
        e0_module,
        "validate_active_study_artifact_paths",
        lambda *_args, **_kwargs: {},
    )

    with pytest.raises(DataValidationError, match="live MLX runtime identity"):
        run_mlx_e0(
            **_arguments(tmp_path),
            work_directory=tmp_path / "work",
            output_directory=tmp_path / "output",
            request_budget=1,
            runtime_factory=lambda *_args: _DriftedRuntime(),
        )

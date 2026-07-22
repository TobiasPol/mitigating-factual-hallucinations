from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import mfh.experiments.e5_capture as e5_capture
from mfh.contracts import ActivationSite, Outcome, PromptSpec, Question
from mfh.errors import FrozenArtifactError
from mfh.experiments.e2_controller_inputs import E2ControllerInputView
from mfh.experiments.e3_construction import (
    E3GenerationRecord,
    VerifiedE3ConstructionSnapshot,
)
from mfh.experiments.e3_schedule import E3ConstructionRow
from mfh.experiments.e5_adaptive import E5Protocol
from mfh.experiments.e5_capture import (
    load_e5_fit_capture_data,
    prepare_e5_fit_capture,
    run_e5_fit_capture,
    verify_e5_fit_capture,
)
from mfh.experiments.e5_fit import E5FitRecipe
from mfh.inference.vllm_research import (
    VllmPromptFeatureCubeOutput,
    VllmTeacherForcedCubeOutput,
)
from mfh.inference.vllm_runtime import VllmRenderedPrompt
from mfh.methods.features import FeatureComposition
from mfh.provenance import stable_hash

_PRIVATE = "31" * 32
_PUBLIC = (
    Ed25519PrivateKey.from_private_bytes(bytes.fromhex(_PRIVATE))
    .public_key()
    .public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    .hex()
)


def _runtime_identity() -> dict[str, Any]:
    return {
        "model_repository": "vllm-community/test",
        "model_revision": "c" * 40,
        "model_quantization": "4bit",
        "altered": False,
    }


def _token_digest(values: Sequence[int]) -> str:
    return hashlib.sha256(",".join(str(value) for value in values).encode()).hexdigest()


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
        value: PromptSpec(value, f"Prompt {value}")
        for value in ("P0-neutral", "P2-calibrated-abstention")
    }


def _render(prompt: PromptSpec, question: Question) -> VllmRenderedPrompt:
    text = f"{prompt.prompt_id}|{question.text}"
    index = int(question.question_id.split("-")[1])
    tokens = (10 + index, 20 + len(prompt.prompt_id))
    return VllmRenderedPrompt(
        text=text,
        sha256=hashlib.sha256(text.encode()).hexdigest(),
        token_ids=tokens,
        token_ids_sha256=_token_digest(tokens),
        messages=(),
    )


def _snapshot(tmp_path: Path) -> VerifiedE3ConstructionSnapshot:
    questions = _questions()
    prompts = _prompts()
    directory = tmp_path / "e3-source"
    directory.mkdir()
    (directory / "frozen.txt").write_text("source", encoding="utf-8")
    schedule: list[E3ConstructionRow] = []
    records: list[E3GenerationRecord] = []
    for prompt_id in prompts:
        for question in questions:
            sequence = len(schedule)
            rendered = _render(prompts[prompt_id], question)
            index = int(question.question_id.split("-")[1])
            raw = f"answer-{index}" if index % 2 == 0 else "wrong"
            token_ids = [100 + index]
            row = E3ConstructionRow(
                sequence=sequence,
                question_id=question.question_id,
                benchmark="triviaqa",
                prompt_id=prompt_id,
                semantic_group_id=f"group-{index}",
                question_sha256=stable_hash(question.text),
                aliases_sha256=stable_hash(list(question.aliases)),
            )
            schedule.append(row)
            records.append(
                E3GenerationRecord(
                    sequence=sequence,
                    plan_identity="a" * 64,
                    schedule_row_sha256=stable_hash(row.to_dict()),
                    question_id=question.question_id,
                    prompt_id=prompt_id,
                    rendered_prompt_sha256=rendered.sha256,
                    prompt_token_ids_sha256=rendered.token_ids_sha256,
                    outcome=(Outcome.CORRECT if index % 2 == 0 else Outcome.INCORRECT),
                    evidence={
                        "raw_output": raw,
                        "raw_output_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                        "token_ids": token_ids,
                        "token_ids_sha256": stable_hash(token_ids),
                        "input_tokens": 2,
                        "output_tokens": 1,
                        "latency_seconds": 0.1,
                        "stop_type": "short_answer",
                        "stopping_token_id": token_ids[-1],
                        "prompt_tokens_per_second": 10.0,
                        "generation_tokens_per_second": 5.0,
                        "peak_memory_bytes": 128,
                        "active_memory_bytes": 64,
                        "cache_memory_bytes": 32,
                    },
                )
            )
    return VerifiedE3ConstructionSnapshot(
        directory=directory.resolve(),
        plan={"plan_identity": "a" * 64, "runtime_identity": _runtime_identity()},
        schedule=tuple(schedule),
        generations=tuple(records),
        generation_chain_head="b" * 64,
        scientific_eligible=False,
    )


class _Runtime:
    def __init__(self, *, altered: bool = False) -> None:
        self.altered = altered

    def runtime_identity(self) -> Mapping[str, Any]:
        return {**_runtime_identity(), "altered": self.altered}

    def render_prompt(
        self,
        prompt: PromptSpec,
        question: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> VllmRenderedPrompt:
        del metadata
        index = int(question.split()[1].rstrip("?"))
        return _render(
            prompt,
            Question(f"q-{index}", "triviaqa", question, (f"answer-{index}",)),
        )

    def prompt_feature_cube(
        self,
        rendered: VllmRenderedPrompt,
        *,
        layers: Sequence[int],
        sites: Sequence[ActivationSite],
    ) -> VllmPromptFeatureCubeOutput:
        index = int(rendered.text.split("Question ")[1].rstrip("?"))
        return VllmPromptFeatureCubeOutput(
            activations={
                site: {layer: np.full((1, 4), index + layer, dtype=np.float32) for layer in layers}
                for site in sites
            },
            maximum_token_probability=0.7,
            output_entropy=0.4,
            peak_memory_bytes=1024,
        )

    def teacher_forced_cube(
        self,
        rendered: VllmRenderedPrompt,
        response: str,
        *,
        layers: Sequence[int],
        sites: Sequence[ActivationSite],
    ) -> VllmTeacherForcedCubeOutput:
        token_ids = (301, 302)
        return VllmTeacherForcedCubeOutput(
            response_text_sha256=hashlib.sha256(response.encode()).hexdigest(),
            response_token_ids=token_ids,
            response_token_ids_sha256=_token_digest(token_ids),
            token_log_probabilities=(-0.2, -0.3),
            negative_log_likelihood=0.5,
            mean_negative_log_likelihood=0.25,
            perplexity=float(np.exp(0.25)),
            activations={
                site: {layer: np.full((2, 4), layer + 0.5, dtype=np.float32) for layer in layers}
                for site in sites
            },
            peak_memory_bytes=2048,
        )


def _prepare(tmp_path: Path) -> tuple[Path, VerifiedE3ConstructionSnapshot]:
    snapshot = _snapshot(tmp_path)
    views = (
        E2ControllerInputView(FeatureComposition.SINGLE_LAYER, (31,), ActivationSite.POST_MLP),
        E2ControllerInputView(
            FeatureComposition.CONCATENATED_LAYERS,
            (16, 31, 32),
            ActivationSite.POST_MLP,
        ),
        E2ControllerInputView(
            FeatureComposition.LAYER_DIFFERENCES,
            (16, 31, 32),
            ActivationSite.POST_MLP,
        ),
    )
    recipe = E5FitRecipe(
        fixed_best_layer=31,
        two_layer_candidates=(31, 32),
        three_layer_candidates=(16, 31, 32),
        intervention_site=ActivationSite.POST_MLP,
    )
    protocol = E5Protocol(
        vector_counts=(1,),
        routers=("nearest_centroid",),
        alpha_modes=("fixed",),
        layer_modes=("fixed_best",),
        intervention_timings=("final_prompt",),
        controller_inputs=("one_layer", "concatenated_layers", "layer_differences"),
    )
    directory = tmp_path / "capture"
    prepare_e5_fit_capture(
        directory,
        snapshot=snapshot,
        questions=_questions(),
        prompts=_prompts(),
        views=views,
        recipe=recipe,
        runtime_identity=_Runtime().runtime_identity(),
        execution_public_key=_PUBLIC,
        runtime_artifact_sha256="d" * 64,
        e2_probe_bundle_sha256="e" * 64,
        e3_static_vectors_sha256="f" * 64,
        split_manifest_digest="1" * 64,
        protocol=protocol,
        hidden_width=4,
        shard_rows=2,
        max_peak_memory_bytes=4096,
    )
    return directory, snapshot


def test_e5_fit_capture_resumes_and_materializes_all_compositions(tmp_path: Path) -> None:
    directory, snapshot = _prepare(tmp_path)
    partial = run_e5_fit_capture(
        directory,
        snapshot=snapshot,
        questions=_questions(),
        prompts=_prompts(),
        runtime=_Runtime(),
        private_key_hex=_PRIVATE,
        request_budget=2,
    )
    assert partial.pairs_completed == 2
    assert not partial.complete
    complete = run_e5_fit_capture(
        directory,
        snapshot=snapshot,
        questions=_questions(),
        prompts=_prompts(),
        runtime=_Runtime(),
        private_key_hex=_PRIVATE,
        request_budget=2,
    )
    assert complete.complete
    assert complete.shard_count == 2
    data = load_e5_fit_capture_data(
        directory,
        snapshot=snapshot,
        questions=_questions(),
        prompts=_prompts(),
        expected_execution_public_key=_PUBLIC,
    )
    assert set(data.vector_datasets) == set(FeatureComposition)
    assert data.vector_datasets[FeatureComposition.SINGLE_LAYER].features.shape == (4, 4)
    assert data.vector_datasets[FeatureComposition.CONCATENATED_LAYERS].features.shape == (
        4,
        12,
    )
    assert data.vector_datasets[FeatureComposition.LAYER_DIFFERENCES].features.shape == (
        4,
        8,
    )
    assert {key.layer for key in data.vector_activations} == {16, 31, 32}


def test_e5_fit_capture_rejects_runtime_and_payload_tampering(tmp_path: Path) -> None:
    directory, snapshot = _prepare(tmp_path)
    with pytest.raises(FrozenArtifactError, match="runtime identity"):
        run_e5_fit_capture(
            directory,
            snapshot=snapshot,
            questions=_questions(),
            prompts=_prompts(),
            runtime=_Runtime(altered=True),
            private_key_hex=_PRIVATE,
        )
    run_e5_fit_capture(
        directory,
        snapshot=snapshot,
        questions=_questions(),
        prompts=_prompts(),
        runtime=_Runtime(),
        private_key_hex=_PRIVATE,
        request_budget=2,
    )
    payload = directory / "shards" / "shard-00000" / "payload.npz"
    payload.write_bytes(payload.read_bytes() + b"tamper")
    with pytest.raises(FrozenArtifactError, match="payload digest"):
        verify_e5_fit_capture(
            directory,
            snapshot=snapshot,
            questions=_questions(),
            prompts=_prompts(),
            expected_execution_public_key=_PUBLIC,
        )


def test_e5_fit_capture_cleans_abandoned_stage_and_records_sessions(
    tmp_path: Path,
) -> None:
    directory, snapshot = _prepare(tmp_path)
    first = run_e5_fit_capture(
        directory,
        snapshot=snapshot,
        questions=_questions(),
        prompts=_prompts(),
        runtime=_Runtime(),
        private_key_hex=_PRIVATE,
        request_budget=2,
    )
    abandoned = directory / "shards" / ".shard-00001.stage-dead"
    abandoned.mkdir()
    (abandoned / "partial").write_text("incomplete", encoding="utf-8")
    second = run_e5_fit_capture(
        directory,
        snapshot=snapshot,
        questions=_questions(),
        prompts=_prompts(),
        runtime=_Runtime(),
        private_key_hex=_PRIVATE,
        request_budget=2,
    )
    assert first.shard_count == 1
    assert second.complete
    assert not abandoned.exists()
    manifests = [
        json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        for path in sorted((directory / "shards").iterdir())
    ]
    assert manifests[0]["execution_session_id"] != manifests[1]["execution_session_id"]
    assert all(value["execution_session_wall_seconds"] >= 0.0 for value in manifests)
    assert all(len(value["execution_lock_identity"]) == 64 for value in manifests)


def test_e5_fit_capture_uses_one_full_verification_per_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, snapshot = _prepare(tmp_path)
    original = e5_capture.verify_e5_fit_capture
    calls = 0

    def counted(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(e5_capture, "verify_e5_fit_capture", counted)
    result = run_e5_fit_capture(
        directory,
        snapshot=snapshot,
        questions=_questions(),
        prompts=_prompts(),
        runtime=_Runtime(),
        private_key_hex=_PRIVATE,
        request_budget=4,
    )
    assert result.complete
    assert result.shard_count == 2
    assert calls == 1


def test_e5_fit_capture_requires_external_public_key_anchor(tmp_path: Path) -> None:
    directory, snapshot = _prepare(tmp_path)
    plan_path = directory / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    body = dict(plan)
    body.pop("plan_identity")
    body["execution_public_key"] = "42" * 32
    altered = {**body, "plan_identity": stable_hash(body)}
    plan_path.write_text(json.dumps(altered, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(FrozenArtifactError, match="external trust root"):
        verify_e5_fit_capture(
            directory,
            snapshot=snapshot,
            questions=_questions(),
            prompts=_prompts(),
            expected_execution_public_key=_PUBLIC,
        )

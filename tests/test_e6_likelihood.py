from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from mfh.contracts import (
    ActivationSite,
    GenerationRecord,
    ModelSpec,
    Outcome,
    PromptSpec,
    Question,
    Runtime,
    TokenScope,
)
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.experiments.e3_schedule import E3Protocol
from mfh.experiments.e6_likelihood import (
    E6RuntimeAttestor,
    E6VerifiedLikelihoodRecord,
    _bind_e6_likelihood_record,
    _index_e6_questions,
    _load_e6_runtime_attestation,
    e6_e3_slice_digest,
    execute_and_bind_e6_likelihood,
    score_e6_question,
    verify_e6_bound_record,
)
from mfh.experiments.e8_protected import (
    e8_execution_receipt_body,
    execute_e8_generation,
    validate_e8_execution_record,
    validate_wikitext_likelihood_evidence,
)
from mfh.experiments.protocol import ExperimentPhase
from mfh.experiments.runner import EvaluationCondition
from mfh.inference.mlx_research import (
    MlxResearchInterventionState,
    MlxResearchRuntime,
    MlxTeacherForcedOutput,
)
from mfh.inference.mlx_runtime import (
    MlxGenerationOutput,
    MlxInterventionState,
    MlxRenderedPrompt,
)
from mfh.provenance import sha256_file, sha256_path, stable_hash

_EXECUTION_PRIVATE_KEY = "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60"
_EXECUTION_PUBLIC_KEY = "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"


def _test_attestor() -> E6RuntimeAttestor:
    value = object.__new__(E6RuntimeAttestor)
    value.runtime = None  # type: ignore[assignment]
    value._private_key = Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(_EXECUTION_PRIVATE_KEY)
    )
    value.execution_public_key = _EXECUTION_PUBLIC_KEY
    value._artifact = {}  # type: ignore[assignment]
    return value


def _token_digest(token_ids: Sequence[int]) -> str:
    return hashlib.sha256(
        ",".join(str(value) for value in token_ids).encode("ascii")
    ).hexdigest()


class _LikelihoodRuntime:
    def __init__(self, losses: Mapping[str, tuple[float, ...]]) -> None:
        self.losses = dict(losses)
        self.states: list[MlxInterventionState] = []
        self.render_calls = 0

    def render_prompt(
        self,
        prompt: PromptSpec,
        question: str,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> MlxRenderedPrompt:
        self.render_calls += 1
        text = f"{prompt.text}\n{question}"
        return MlxRenderedPrompt(
            text=text,
            sha256=hashlib.sha256(text.encode()).hexdigest(),
            token_ids=(1, 2),
            token_ids_sha256=_token_digest((1, 2)),
            messages=({"role": "user", "content": question},),
        )

    def teacher_forced_continuation(
        self,
        rendered: MlxRenderedPrompt,
        response: str,
        *,
        layers: Sequence[int],
        site: ActivationSite,
        intervention_states: Mapping[int, MlxInterventionState] | None = None,
    ) -> MlxTeacherForcedOutput:
        assert rendered.token_ids == (1, 2)
        assert tuple(layers) == (16,)
        assert site is ActivationSite.POST_MLP
        if intervention_states is not None:
            state = intervention_states[16]
            assert isinstance(state, MlxResearchInterventionState)
            self.states.append(state)
        log_probabilities = tuple(-value for value in self.losses[response])
        token_ids = tuple(range(10, 10 + len(log_probabilities)))
        nll = -sum(log_probabilities)
        mean_nll = nll / len(log_probabilities)
        if intervention_states is not None:
            state.arm_prompt(len(rendered.token_ids))
            for sequence_length in (len(rendered.token_ids), *([1] * len(token_ids))):
                effective_alpha = state.effective_alpha(sequence_length)
                state.captured = np.ones((1, sequence_length, 2), dtype=np.float32)
                state.intervened = state.captured + effective_alpha
                state.applications += int(effective_alpha != 0)
        return MlxTeacherForcedOutput(
            response_text_sha256=hashlib.sha256(response.encode()).hexdigest(),
            response_token_ids=token_ids,
            response_token_ids_sha256=_token_digest(token_ids),
            token_log_probabilities=log_probabilities,
            negative_log_likelihood=nll,
            mean_negative_log_likelihood=mean_nll,
            perplexity=math.exp(mean_nll),
            activations={16: np.ones((len(token_ids), 2), dtype=np.float32)},
        )


class _IntegratedBase(_LikelihoodRuntime):
    def __init__(self, snapshot: Path) -> None:
        super().__init__({"gold": (0.3,), "I don't know.": (0.6,)})
        self.snapshot = snapshot
        self.model_spec = ModelSpec(
            name="bonsai-27b-mlx-1bit",
            repository="prism-ml/Bonsai-27B-mlx-1bit",
            revision="ef22f239c670078e1507f9769bcaa66657332b96",
            runtime=Runtime.MLX,
            quantization="grouped-binary-mlx-1.125bpw",
            num_layers=64,
        )

    def runtime_identity(self) -> Mapping[str, object]:
        return {
            "backend": "mlx",
            "mlx": "test",
            "mlx_lm": "test",
            "python": "test",
            "machine_model": "test",
            "chip": "test",
            "unified_memory_bytes": 1,
            "physical_cpu_cores": 1,
            "architecture": "arm64",
            "os": "macOS test",
            "os_build": "test",
            "model_class": "test.Model",
            "tokenizer_class": "test.Tokenizer",
            "num_layers": 64,
            "seed": 17,
            "research_provenance": {"fixture": "integrated"},
            "research_toolchain": {
                "xcodebuild": "test",
                "metal_compiler": "test",
            },
        }


def _write_e3_vectors(path: Path) -> tuple[str, str]:
    path.mkdir()
    protocol = E3Protocol()
    shape = (
        2,
        2,
        len(protocol.candidate_sites),
        len(protocol.candidate_layers),
    )
    directions = np.zeros((*shape, 2), dtype=np.float32)
    directions[..., 0] = 1.0
    arrays = path / "vectors.npz"
    np.savez_compressed(
        arrays,
        directions=directions,
        reference_rms=np.ones(shape, dtype=np.float64),
        correct_counts=np.full(shape, 15_000, dtype=np.int64),
        incorrect_counts=np.full(shape, 15_000, dtype=np.int64),
    )
    body = {
        "schema_version": 1,
        "phase": "E3-construction",
        "plan_identity": "1" * 64,
        "protocol": protocol.to_dict(),
        "prompt_axis": ["P0-neutral", "P2-calibrated-abstention"],
        "extraction_axis": ["M1-R", "M1-P"],
        "site_axis": [value.value for value in protocol.candidate_sites],
        "layer_axis": list(protocol.candidate_layers),
        "hidden_width": 2,
        "rows_processed": protocol.construction_rows,
        "response_pooling": protocol.response_pooling,
        "scientific_eligible": True,
        "maximum_peak_memory_bytes": 1,
        "generation_chain_head": "2" * 64,
        "checkpoint_chain_head": "3" * 64,
        "vectors_sha256": sha256_file(arrays),
        "data_fingerprint": "4" * 64,
    }
    (path / "metadata.json").write_text(
        json.dumps({**body, "metadata_digest": stable_hash(body)}),
        encoding="utf-8",
    )
    direction_sha = hashlib.sha256(np.array([1.0, 0.0], dtype=np.float32).tobytes()).hexdigest()
    return sha256_path(path), direction_sha


def test_e6_scores_best_alias_abstention_and_rank_with_fresh_states() -> None:
    runtime = _LikelihoodRuntime(
        {
            "long gold": (0.4, 0.4),
            "gold": (0.3,),
            "I don't know.": (0.6,),
            "distractor-high": (0.2,),
            "distractor-low": (0.7,),
        }
    )
    factory_calls = 0

    def state_factory() -> Mapping[int, MlxInterventionState]:
        nonlocal factory_calls
        factory_calls += 1
        return {
            16: MlxResearchInterventionState(
                direction=np.array([1.0, 0.0], dtype=np.float32),
                alpha=0.5,
            )
        }

    record = score_e6_question(
        runtime=runtime,
        question=Question(
            question_id="q-1",
            benchmark="triviaqa",
            text="Question?",
            aliases=("long gold", "gold"),
            metadata={
                "e6_plausible_alternatives": [
                    "distractor-high",
                    "distractor-low",
                ]
            },
        ),
        prompt=PromptSpec("P2-calibrated-abstention", "Answer or abstain."),
        method="M3",
        condition_id="a" * 64,
        layers=(16,),
        site=ActivationSite.POST_MLP,
        state_factory=state_factory,
    )

    assert record.best_alias_index == 1
    assert record.gold_log_likelihood == pytest.approx(-0.3)
    assert record.abstention_log_likelihood == pytest.approx(-0.6)
    assert record.gold_rank == 2
    assert record.generation_metadata() == {
        "gold_alias_log_likelihood": pytest.approx(-0.3),
        "abstention_log_likelihood": pytest.approx(-0.6),
        "gold_answer_rank": 2,
    }
    assert runtime.render_calls == 1
    assert factory_calls == 5
    assert len(runtime.states) == len({id(state) for state in runtime.states}) == 5


def test_e6_question_identity_includes_benchmark() -> None:
    trivia = Question("shared-id", "triviaqa", "Trivia question?", ("trivia",))
    simple = Question(
        "shared-id", "simpleqa_verified", "SimpleQA question?", ("simple",)
    )

    indexed = _index_e6_questions((trivia, simple))

    assert indexed[("triviaqa", "shared-id")] == trivia
    assert indexed[("simpleqa_verified", "shared-id")] == simple
    with pytest.raises(DataValidationError, match="duplicate identity"):
        _index_e6_questions((trivia, trivia))


def test_e6_rejects_unbound_condition_and_method_state_mismatch() -> None:
    runtime = _LikelihoodRuntime({"gold": (0.3,), "I don't know.": (0.6,)})
    question = Question("q-1", "simpleqa_verified", "Question?", ("gold",))
    prompt = PromptSpec("P0-neutral", "Answer.")

    with pytest.raises(DataValidationError, match="inputs"):
        score_e6_question(
            runtime=runtime,
            question=question,
            prompt=prompt,
            method="M0",
            condition_id="not-a-digest",
            layers=(16,),
            site=ActivationSite.POST_MLP,
        )


def test_e6_signed_binding_rejects_runtime_receipt_tampering() -> None:
    runtime = _LikelihoodRuntime(
        {"gold": (0.3,), "I don't know.": (0.6,), "distractor": (0.8,)}
    )
    question = Question(
        "q-1",
        "triviaqa",
        "Question?",
        ("gold",),
        metadata={"e6_plausible_alternatives": ["distractor"]},
    )
    prompt = PromptSpec("P0-neutral", "Answer.")
    condition = EvaluationCondition(
        phase=ExperimentPhase.E6,
        benchmark="triviaqa",
        partition="T-dev",
        model_name="bonsai-27b-mlx-1bit",
        model_repository="prism-ml/Bonsai-27B-mlx-1bit",
        model_revision="ef22f239c670078e1507f9769bcaa66657332b96",
        runtime=Runtime.MLX,
        quantization="grouped-binary-mlx-1.125bpw",
        model_num_layers=64,
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
        study_protocol_digest="1" * 64,
    )
    likelihood = score_e6_question(
        runtime=runtime,
        question=question,
        prompt=prompt,
        method="M0",
        condition_id=condition.condition_id,
        layers=(16,),
        site=ActivationSite.POST_MLP,
    )
    runtime_sha = "2" * 64
    question_bundle_sha = "3" * 64
    metadata = {
        "phase": ExperimentPhase.E6.value,
        "partition": "T-dev",
        "prompt_template_sha256": condition.prompt_template_sha256,
        "study_protocol_digest": condition.study_protocol_digest,
        "e6_likelihood_record_digest": likelihood.record_digest,
        "e6_runtime_artifact_sha256": runtime_sha,
        "e6_execution_public_key": _EXECUTION_PUBLIC_KEY,
        "e6_question_bundle_sha256": question_bundle_sha,
        **dict(likelihood.generation_metadata()),
    }
    generation = GenerationRecord(
        question_id=question.question_id,
        benchmark=question.benchmark,
        model_repository=condition.model_repository,
        model_revision=condition.model_revision,
        runtime=condition.runtime,
        quantization=condition.quantization,
        system_prompt_id=condition.system_prompt_id,
        rendered_prompt_hash=likelihood.rendered_prompt_sha256,
        steering_method="M0",
        layer=None,
        token_scope=None,
        alpha=0.0,
        sparsity=None,
        controller_scores={},
        raw_output="gold",
        normalized_answer="gold",
        outcome=Outcome.CORRECT,
        generation_latency_seconds=1.0,
        input_tokens=2,
        output_tokens=1,
        condition_id=condition.condition_id,
        seed=17,
        metadata=metadata,
    )
    bound = _bind_e6_likelihood_record(
        likelihood,
        generation_record=generation,
        condition=condition,
        runtime_artifact_sha256=runtime_sha,
        question_bundle_sha256=question_bundle_sha,
        attestor=_test_attestor(),
    )
    assert verify_e6_bound_record(
        bound,
        generation_record=generation,
        condition=condition,
        question=question,
        runtime_artifact_sha256=runtime_sha,
        execution_public_key=_EXECUTION_PUBLIC_KEY,
    )["valid"] is True
    with pytest.raises(FrozenArtifactError, match="answer texts"):
        verify_e6_bound_record(
            bound,
            generation_record=generation,
            condition=condition,
            question=replace(question, aliases=("substituted",)),
            runtime_artifact_sha256=runtime_sha,
            execution_public_key=_EXECUTION_PUBLIC_KEY,
        )
    with pytest.raises(FrozenArtifactError, match="answer texts"):
        verify_e6_bound_record(
            bound,
            generation_record=generation,
            condition=condition,
            question=replace(
                question,
                metadata={"e6_plausible_alternatives": ["substituted"]},
            ),
            runtime_artifact_sha256=runtime_sha,
            execution_public_key=_EXECUTION_PUBLIC_KEY,
        )

    tampered_signature = ("0" if bound.execution_receipt_signature[0] != "0" else "1") + (
        bound.execution_receipt_signature[1:]
    )
    body = {**bound.execution_body(), "execution_receipt_signature": tampered_signature}
    tampered = replace(
        bound,
        execution_receipt_signature=tampered_signature,
        verified_record_digest=stable_hash(body),
    )
    assert isinstance(tampered, E6VerifiedLikelihoodRecord)
    with pytest.raises(FrozenArtifactError, match="signature"):
        verify_e6_bound_record(
            tampered,
            generation_record=generation,
            condition=condition,
            question=question,
            runtime_artifact_sha256=runtime_sha,
            execution_public_key=_EXECUTION_PUBLIC_KEY,
        )
    with pytest.raises(DataValidationError, match="inputs"):
        score_e6_question(
            runtime=runtime,
            question=question,
            prompt=prompt,
            method="M1",
            condition_id="a" * 64,
            layers=(16,),
            site=ActivationSite.POST_MLP,
        )


def test_e6_rejects_zero_alpha_declared_intervention_before_scoring() -> None:
    runtime = _LikelihoodRuntime({"gold": (0.3,), "I don't know.": (0.6,)})

    def zero_state() -> Mapping[int, MlxInterventionState]:
        return {
            16: MlxResearchInterventionState(
                direction=np.array([1.0, 0.0], dtype=np.float32),
                alpha=0.0,
            )
        }

    with pytest.raises(DataValidationError, match="material MLX state"):
        score_e6_question(
            runtime=runtime,
            question=Question("q-1", "triviaqa", "Question?", ("gold",)),
            prompt=PromptSpec("P0-neutral", "Answer."),
            method="M1",
            condition_id="a" * 64,
            layers=(16,),
            site=ActivationSite.POST_MLP,
            state_factory=zero_state,
        )


def test_integrated_e6_executor_owns_runtime_attestation_and_enriches_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "weights.safetensors").write_bytes(b"frozen")
    base = _IntegratedBase(snapshot)
    runtime = MlxResearchRuntime(base)  # type: ignore[arg-type]
    monkeypatch.setattr(
        runtime,
        "teacher_forced_continuation",
        base.teacher_forced_continuation,
    )

    def generate_with_interventions(
        rendered: MlxRenderedPrompt,
        *,
        max_new_tokens: int,
        intervention_states: Mapping[
            tuple[int, ActivationSite], MlxInterventionState
        ],
    ) -> MlxGenerationOutput:
        assert max_new_tokens == 32
        if intervention_states:
            state = intervention_states[(16, ActivationSite.POST_MLP)]
            assert isinstance(state, MlxResearchInterventionState)
            state.arm_prompt(len(rendered.token_ids))
            state.effective_alpha(len(rendered.token_ids))
            effective_alpha = state.effective_alpha(1)
            state.captured = np.ones(
                (1, 1, int(np.asarray(state.direction).size)), dtype=np.float32
            )
            state.intervened = state.captured + effective_alpha * np.asarray(
                state.direction, dtype=np.float32
            )
            state.applications += int(effective_alpha != 0)
        return MlxGenerationOutput(
            rendered_prompt=rendered,
            token_ids=(10,),
            text="gold",
            input_tokens=2,
            output_tokens=1,
            latency_seconds=1.0,
            stop_type="short_answer",
            stopping_token_id=10,
            prompt_tokens_per_second=1.0,
            generation_tokens_per_second=1.0,
            peak_memory_bytes=1,
            active_memory_bytes=1,
            cache_memory_bytes=0,
        )

    monkeypatch.setattr(
        runtime,
        "generate_with_interventions",
        generate_with_interventions,
    )
    attestor = E6RuntimeAttestor(
        runtime,
        execution_private_key=_EXECUTION_PRIVATE_KEY,
    )
    runtime_artifact = tmp_path / "runtime-attestation.json"
    runtime_sha = attestor.write_runtime_artifact(runtime_artifact)
    assert _load_e6_runtime_attestation(runtime_artifact)[
        "execution_public_key"
    ] == attestor.execution_public_key
    prompt = PromptSpec("P0-neutral", "Answer.")
    question = Question("q-1", "triviaqa", "Question?", ("gold",))
    rendered = runtime.render_prompt(prompt, question.text)
    condition = EvaluationCondition(
        phase=ExperimentPhase.E6,
        benchmark="triviaqa",
        partition="T-dev",
        model_name=base.model_spec.name,
        model_repository=base.model_spec.repository,
        model_revision=base.model_spec.revision,
        runtime=Runtime.MLX,
        quantization=base.model_spec.quantization,
        model_num_layers=base.model_spec.num_layers,
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
        study_protocol_digest="1" * 64,
    )
    unbound = GenerationRecord(
        question_id=question.question_id,
        benchmark=question.benchmark,
        model_repository=condition.model_repository,
        model_revision=condition.model_revision,
        runtime=condition.runtime,
        quantization=condition.quantization,
        system_prompt_id=condition.system_prompt_id,
        rendered_prompt_hash=rendered.sha256,
        steering_method="M0",
        layer=None,
        token_scope=None,
        alpha=0.0,
        sparsity=None,
        controller_scores={},
        raw_output="",
        normalized_answer="",
        outcome=Outcome.INCORRECT,
        generation_latency_seconds=0.0,
        input_tokens=0,
        output_tokens=0,
        condition_id=condition.condition_id,
        seed=17,
        metadata={
            "phase": ExperimentPhase.E6.value,
            "partition": "T-dev",
            "prompt_template_sha256": condition.prompt_template_sha256,
            "study_protocol_digest": condition.study_protocol_digest,
        },
    )
    executed = execute_and_bind_e6_likelihood(
        attestor=attestor,
        runtime_artifact=runtime_artifact,
        e3_static_vectors=snapshot,
        question=question,
        prompt=prompt,
        generation_record=unbound,
        condition=condition,
        layers=(16,),
        site=ActivationSite.POST_MLP,
        state_factory=None,
        question_bundle_sha256="3" * 64,
        populate_generation=True,
        generation_grader=lambda record: replace(
            record,
            outcome=Outcome.PARTIAL,
            metadata={**dict(record.metadata), "official_test_grade": "P"},
        ),
    )

    assert executed.generation_record.raw_output == "gold"
    assert executed.generation_record.outcome is Outcome.PARTIAL
    assert executed.generation_record.metadata["official_test_grade"] == "P"
    assert executed.generation_record.metadata["e6_runtime_artifact_sha256"] == runtime_sha
    assert executed.generation_record.metadata["generation_runtime_metrics"] == {
        "schema_version": 1,
        "unified_memory_bytes": 1,
        "peak_memory_bytes": 1,
        "generation_peak_memory_bytes": 1,
        "auxiliary_peak_memory_bytes": 0,
        "active_memory_bytes": 1,
        "cache_memory_bytes": 0,
        "prompt_tokens_per_second": 1.0,
        "generation_tokens_per_second": 1.0,
        "generation_wall_time_seconds": 1.0,
        "stop_type": "short_answer",
        "stopping_token_id": 10,
    }
    assert verify_e6_bound_record(
        executed.likelihood_record,
        generation_record=executed.generation_record,
        condition=condition,
        question=question,
        runtime_artifact_sha256=runtime_sha,
        execution_public_key=attestor.execution_public_key,
    )["valid"] is True
    with pytest.raises(DataValidationError, match="unsigned runtime row"):
        execute_and_bind_e6_likelihood(
            attestor=attestor,
            runtime_artifact=runtime_artifact,
            e3_static_vectors=snapshot,
            question=question,
            prompt=prompt,
            generation_record=executed.generation_record,
            condition=condition,
            layers=(16,),
            site=ActivationSite.POST_MLP,
            state_factory=None,
            question_bundle_sha256="3" * 64,
        )


def test_integrated_e6_m1_signs_material_generation_and_exact_e3_slice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "weights.safetensors").write_bytes(b"frozen")
    e3_path = tmp_path / "e3-vectors"
    e3_sha, direction_sha = _write_e3_vectors(e3_path)
    base = _IntegratedBase(snapshot)
    runtime = MlxResearchRuntime(base)  # type: ignore[arg-type]
    monkeypatch.setattr(
        runtime,
        "teacher_forced_continuation",
        base.teacher_forced_continuation,
    )

    def generate_with_interventions(
        rendered: MlxRenderedPrompt,
        *,
        max_new_tokens: int,
        intervention_states: Mapping[
            tuple[int, ActivationSite], MlxInterventionState
        ],
    ) -> MlxGenerationOutput:
        assert max_new_tokens == 32
        state = intervention_states[(16, ActivationSite.POST_MLP)]
        assert isinstance(state, MlxResearchInterventionState)
        state.arm_prompt(len(rendered.token_ids))
        state.effective_alpha(len(rendered.token_ids))
        effective_alpha = state.effective_alpha(1)
        state.captured = np.ones((1, 1, 2), dtype=np.float32)
        state.intervened = state.captured + effective_alpha * np.array(
            [1.0, 0.0], dtype=np.float32
        )
        state.applications += int(effective_alpha != 0)
        return MlxGenerationOutput(
            rendered_prompt=rendered,
            token_ids=(10,),
            text="gold",
            input_tokens=2,
            output_tokens=1,
            latency_seconds=1.0,
            stop_type="short_answer",
            stopping_token_id=10,
            prompt_tokens_per_second=1.0,
            generation_tokens_per_second=1.0,
            peak_memory_bytes=1,
            active_memory_bytes=1,
            cache_memory_bytes=0,
        )

    monkeypatch.setattr(
        runtime,
        "generate_with_interventions",
        generate_with_interventions,
    )
    attestor = E6RuntimeAttestor(
        runtime,
        execution_private_key=_EXECUTION_PRIVATE_KEY,
    )
    runtime_artifact = tmp_path / "runtime-attestation.json"
    attestor.write_runtime_artifact(runtime_artifact)
    tensor_index = ["P0-neutral", "M1-R", "post_mlp", 16]
    method_artifact = e6_e3_slice_digest(
        e3_static_vectors_sha256=e3_sha,
        tensor_index=tensor_index,
        direction_sha256=direction_sha,
    )
    prompt = PromptSpec("P0-neutral", "Answer.")
    question = Question("q-1", "triviaqa", "Question?", ("gold",))
    rendered = runtime.render_prompt(prompt, question.text)
    condition = EvaluationCondition(
        phase=ExperimentPhase.E6,
        benchmark="triviaqa",
        partition="T-dev",
        model_name=base.model_spec.name,
        model_repository=base.model_spec.repository,
        model_revision=base.model_spec.revision,
        runtime=Runtime.MLX,
        quantization=base.model_spec.quantization,
        model_num_layers=base.model_spec.num_layers,
        system_prompt_id=prompt.prompt_id,
        prompt_template_sha256=hashlib.sha256(prompt.text.encode()).hexdigest(),
        steering_method="M1",
        method_artifact_sha256=method_artifact,
        layer=16,
        site=ActivationSite.POST_MLP,
        token_scope=TokenScope.FIRST_FOUR,
        alpha=0.5,
        sparsity=None,
        seed=17,
        study_protocol_digest="1" * 64,
    )
    generation = GenerationRecord(
        question_id=question.question_id,
        benchmark=question.benchmark,
        model_repository=condition.model_repository,
        model_revision=condition.model_revision,
        runtime=condition.runtime,
        quantization=condition.quantization,
        system_prompt_id=condition.system_prompt_id,
        rendered_prompt_hash=rendered.sha256,
        steering_method="M1",
        layer=16,
        site=ActivationSite.POST_MLP,
        token_scope=TokenScope.FIRST_FOUR,
        alpha=0.5,
        sparsity=None,
        controller_scores={},
        raw_output="gold",
        normalized_answer="gold",
        outcome=Outcome.CORRECT,
        generation_latency_seconds=1.0,
        input_tokens=2,
        output_tokens=1,
        condition_id=condition.condition_id,
        seed=17,
        metadata={
            "phase": ExperimentPhase.E6.value,
            "partition": "T-dev",
            "prompt_template_sha256": condition.prompt_template_sha256,
            "study_protocol_digest": condition.study_protocol_digest,
            "method_artifact_sha256": method_artifact,
        },
    )

    def state_factory() -> Mapping[int, MlxInterventionState]:
        return {
            16: MlxResearchInterventionState(
                direction=np.array([1.0, 0.0], dtype=np.float32),
                alpha=0.5,
                token_scope=TokenScope.FIRST_FOUR,
            )
        }

    executed = execute_and_bind_e6_likelihood(
        attestor=attestor,
        runtime_artifact=runtime_artifact,
        e3_static_vectors=e3_path,
        question=question,
        prompt=prompt,
        generation_record=generation,
        condition=condition,
        layers=(16,),
        site=ActivationSite.POST_MLP,
        state_factory=state_factory,
        question_bundle_sha256="8" * 64,
        e3_tensor_index=tensor_index,
    )
    assert len(
        executed.generation_record.metadata["e6_generation_execution_signature"]
    ) == 128
    with pytest.raises(DataValidationError, match="registered E3 slice"):
        execute_and_bind_e6_likelihood(
            attestor=attestor,
            runtime_artifact=runtime_artifact,
            e3_static_vectors=e3_path,
            question=question,
            prompt=prompt,
            generation_record=generation,
            condition=condition,
            layers=(16,),
            site=ActivationSite.POST_MLP,
            state_factory=state_factory,
            question_bundle_sha256="8" * 64,
            e3_tensor_index=["P0-neutral", "M1-P", "post_mlp", 16],
        )


def test_integrated_e8_fixed_executor_signs_material_native_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "weights.safetensors").write_bytes(b"frozen")
    base = _IntegratedBase(snapshot)
    runtime = MlxResearchRuntime(base)  # type: ignore[arg-type]

    def standardized_intervention_state(
        direction: np.ndarray,
        *,
        standardized_alpha: float,
        reference_rms: float,
        token_scope: TokenScope,
    ) -> MlxResearchInterventionState:
        return MlxResearchInterventionState(
            direction=direction,
            alpha=standardized_alpha * reference_rms,
            token_scope=token_scope,
        )

    monkeypatch.setattr(
        runtime, "standardized_intervention_state", standardized_intervention_state
    )

    def generate_with_interventions(
        rendered: MlxRenderedPrompt,
        *,
        max_new_tokens: int,
        intervention_states: Mapping[
            tuple[int, ActivationSite], MlxInterventionState
        ],
    ) -> MlxGenerationOutput:
        assert max_new_tokens == 32
        state = intervention_states[(16, ActivationSite.POST_MLP)]
        assert isinstance(state, MlxResearchInterventionState)
        state.arm_prompt(len(rendered.token_ids))
        state.effective_alpha(len(rendered.token_ids))
        effective_alpha = state.effective_alpha(1)
        state.captured = np.ones((1, 1, 2), dtype=np.float32)
        state.intervened = state.captured + effective_alpha * np.array(
            [1.0, 0.0], dtype=np.float32
        )
        state.applications += int(effective_alpha != 0)
        return MlxGenerationOutput(
            rendered_prompt=rendered,
            token_ids=(10,),
            text="gold",
            input_tokens=2,
            output_tokens=1,
            latency_seconds=1.25,
            stop_type="short_answer",
            stopping_token_id=10,
            prompt_tokens_per_second=1.0,
            generation_tokens_per_second=1.0,
            peak_memory_bytes=1,
            active_memory_bytes=1,
            cache_memory_bytes=0,
        )

    monkeypatch.setattr(runtime, "generate_with_interventions", generate_with_interventions)
    attestor = E6RuntimeAttestor(runtime, execution_private_key=_EXECUTION_PRIVATE_KEY)
    runtime_artifact = tmp_path / "runtime-attestation.json"
    runtime_sha = attestor.write_runtime_artifact(runtime_artifact)
    prompt = PromptSpec("P0-neutral", "Answer.")
    question = Question("q-e8", "triviaqa", "Question?", ("gold",))
    rendered = runtime.render_prompt(prompt, question.text)
    condition = EvaluationCondition(
        phase=ExperimentPhase.E8,
        benchmark="triviaqa",
        partition="T-dev",
        model_name=base.model_spec.name,
        model_repository=base.model_spec.repository,
        model_revision=base.model_spec.revision,
        runtime=Runtime.MLX,
        quantization=base.model_spec.quantization,
        model_num_layers=base.model_spec.num_layers,
        system_prompt_id=prompt.prompt_id,
        prompt_template_sha256=hashlib.sha256(prompt.text.encode()).hexdigest(),
        steering_method="M5",
        method_artifact_sha256="5" * 64,
        layer=16,
        site=ActivationSite.POST_MLP,
        token_scope=TokenScope.FIRST_FOUR,
        alpha=0.5,
        sparsity=None,
        seed=17,
        study_protocol_digest="1" * 64,
    )
    generation = GenerationRecord(
        question_id=question.question_id,
        benchmark=question.benchmark,
        model_repository=condition.model_repository,
        model_revision=condition.model_revision,
        runtime=condition.runtime,
        quantization=condition.quantization,
        system_prompt_id=condition.system_prompt_id,
        rendered_prompt_hash=rendered.sha256,
        steering_method="M5",
        layer=16,
        site=ActivationSite.POST_MLP,
        token_scope=TokenScope.FIRST_FOUR,
        alpha=0.5,
        sparsity=None,
        controller_scores={},
        raw_output="gold",
        normalized_answer="gold",
        outcome=Outcome.CORRECT,
        generation_latency_seconds=99.0,
        input_tokens=2,
        output_tokens=1,
        condition_id=condition.condition_id,
        seed=17,
        metadata={
            "phase": "E8",
            "partition": "T-dev",
            "prompt_template_sha256": condition.prompt_template_sha256,
            "study_protocol_digest": condition.study_protocol_digest,
            "method_artifact_sha256": condition.method_artifact_sha256,
        },
    )
    raw_direction = np.random.default_rng(17).standard_normal(5_120).astype(np.float32)
    direction = np.ascontiguousarray(raw_direction / np.linalg.norm(raw_direction))
    renormalized = np.ascontiguousarray(direction / np.linalg.norm(direction))
    assert direction.tobytes() != renormalized.tobytes()
    executed = execute_e8_generation(
        attestor=attestor,
        runtime_artifact=runtime_artifact,
        question=question,
        prompt=prompt,
        generation_record=generation,
        condition=condition,
        direction=direction,
        reference_rms=1.0,
    )
    facts = {
        "steering_method": "M5",
        "method_artifact_sha256": condition.method_artifact_sha256,
        "layer": 16,
        "site": ActivationSite.POST_MLP.value,
        "token_scope": TokenScope.FIRST_FOUR.value,
        "alpha": 0.5,
        "sparsity": None,
        "prompt_template_sha256": condition.prompt_template_sha256,
    }
    validate_e8_execution_record(
        executed,
        condition_facts=facts,
        execution_public_key=attestor.execution_public_key,
        runtime_artifact_sha256=runtime_sha,
        runtime_identity=attestor.attested_runtime_identity,
    )
    assert executed.generation_latency_seconds == 1.25
    trace = executed.metadata["intervention_trace"]
    assert trace["direction_sha256"] == hashlib.sha256(direction.tobytes()).hexdigest()
    assert trace["raw_alpha"] == 0.5
    forged_metrics = dict(executed.metadata["generation_runtime_metrics"])
    forged_metrics["auxiliary_peak_memory_bytes"] = 1
    forged_metrics["peak_memory_bytes"] = max(
        int(forged_metrics["generation_peak_memory_bytes"]), 1
    )
    forged = replace(
        executed,
        metadata={
            **dict(executed.metadata),
            "generation_runtime_metrics": forged_metrics,
        },
    )
    forged = replace(
        forged,
        metadata={
            **dict(forged.metadata),
            "e8_generation_execution_signature": attestor._sign(
                e8_execution_receipt_body(forged)
            ),
        },
    )
    with pytest.raises(DataValidationError, match="embedded source"):
        validate_e8_execution_record(
            forged,
            condition_facts=facts,
            execution_public_key=attestor.execution_public_key,
            runtime_artifact_sha256=runtime_sha,
            runtime_identity=attestor.attested_runtime_identity,
        )
    with pytest.raises(DataValidationError, match="signature"):
        validate_e8_execution_record(
            replace(executed, generation_latency_seconds=2.0),
            condition_facts=facts,
            execution_public_key=attestor.execution_public_key,
            runtime_artifact_sha256=runtime_sha,
        )


def test_e8_executor_measures_native_teacher_forced_wikitext_nll(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "weights.safetensors").write_bytes(b"frozen")
    base = _IntegratedBase(snapshot)
    runtime = MlxResearchRuntime(base)  # type: ignore[arg-type]

    def generate_with_interventions(
        rendered: MlxRenderedPrompt,
        *,
        max_new_tokens: int,
        intervention_states: Mapping[
            tuple[int, ActivationSite], MlxInterventionState
        ],
    ) -> MlxGenerationOutput:
        assert max_new_tokens == 32
        assert not intervention_states
        return MlxGenerationOutput(
            rendered_prompt=rendered,
            token_ids=(10,),
            text="generated continuation",
            input_tokens=2,
            output_tokens=1,
            latency_seconds=0.5,
            stop_type="short_answer",
            stopping_token_id=10,
            prompt_tokens_per_second=1.0,
            generation_tokens_per_second=1.0,
            peak_memory_bytes=1,
            active_memory_bytes=1,
            cache_memory_bytes=0,
        )

    def teacher_forced_continuation(
        rendered: MlxRenderedPrompt,
        response: str,
        *,
        layers: Sequence[int],
        site: ActivationSite,
        intervention_states: Mapping[int, MlxInterventionState] | None = None,
    ) -> MlxTeacherForcedOutput:
        assert response == "A frozen WikiText continuation."
        assert tuple(layers) == (0,)
        assert site is ActivationSite.POST_MLP
        assert not intervention_states
        token_ids = (21, 22)
        log_probabilities = (-0.2, -0.4)
        return MlxTeacherForcedOutput(
            response_text_sha256=hashlib.sha256(response.encode()).hexdigest(),
            response_token_ids=token_ids,
            response_token_ids_sha256=_token_digest(token_ids),
            token_log_probabilities=log_probabilities,
            negative_log_likelihood=0.6,
            mean_negative_log_likelihood=0.3,
            perplexity=math.exp(0.3),
            activations={0: np.ones((2, 2), dtype=np.float32)},
        )

    monkeypatch.setattr(runtime, "generate_with_interventions", generate_with_interventions)
    monkeypatch.setattr(
        runtime, "teacher_forced_continuation", teacher_forced_continuation
    )
    attestor = E6RuntimeAttestor(runtime, execution_private_key=_EXECUTION_PRIVATE_KEY)
    runtime_artifact = tmp_path / "runtime-attestation.json"
    runtime_sha = attestor.write_runtime_artifact(runtime_artifact)
    prompt = PromptSpec("P0-neutral", "Continue the text.")
    question = Question(
        "wikitext103:1",
        "wikitext103",
        "A frozen WikiText continuation.",
        ("__wikitext_official_perplexity_scorer__",),
    )
    rendered = runtime.render_prompt(prompt, question.text)
    condition = EvaluationCondition(
        phase=ExperimentPhase.E8,
        benchmark="wikitext103",
        partition="test",
        model_name=base.model_spec.name,
        model_repository=base.model_spec.repository,
        model_revision=base.model_spec.revision,
        runtime=Runtime.MLX,
        quantization=base.model_spec.quantization,
        model_num_layers=base.model_spec.num_layers,
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
        study_protocol_digest="1" * 64,
    )
    generation = GenerationRecord(
        question_id=question.question_id,
        benchmark=question.benchmark,
        model_repository=condition.model_repository,
        model_revision=condition.model_revision,
        runtime=condition.runtime,
        quantization=condition.quantization,
        system_prompt_id=condition.system_prompt_id,
        rendered_prompt_hash=rendered.sha256,
        steering_method="M0",
        layer=None,
        site=None,
        token_scope=None,
        alpha=0.0,
        sparsity=None,
        controller_scores={},
        raw_output="",
        normalized_answer="",
        outcome=Outcome.INCORRECT,
        generation_latency_seconds=0.0,
        input_tokens=0,
        output_tokens=0,
        condition_id=condition.condition_id,
        seed=17,
        metadata={
            "phase": "E8",
            "partition": "test",
            "prompt_template_sha256": condition.prompt_template_sha256,
            "study_protocol_digest": condition.study_protocol_digest,
        },
    )

    def grade_after_likelihood(record: GenerationRecord) -> GenerationRecord:
        assert record.metadata["negative_log_likelihood"] == pytest.approx(0.3)
        assert "wikitext_likelihood_evidence" in record.metadata
        return replace(
            record,
            outcome=Outcome.UNSCORABLE,
            metadata={**dict(record.metadata), "test_grader_marker": "after-likelihood"},
        )

    executed = execute_e8_generation(
        attestor=attestor,
        runtime_artifact=runtime_artifact,
        question=question,
        prompt=prompt,
        generation_record=generation,
        condition=condition,
        direction=None,
        reference_rms=None,
        populate_generation=True,
        generation_grader=grade_after_likelihood,
    )
    validate_e8_execution_record(
        executed,
        condition_facts=condition.to_dict(),
        execution_public_key=attestor.execution_public_key,
        runtime_artifact_sha256=runtime_sha,
        runtime_identity=attestor.attested_runtime_identity,
    )
    assert validate_wikitext_likelihood_evidence(
        executed, question=question
    ) == pytest.approx(0.3)
    assert executed.outcome is Outcome.UNSCORABLE
    assert executed.metadata["test_grader_marker"] == "after-likelihood"
    with pytest.raises(DataValidationError, match="changed signed runtime facts"):
        execute_e8_generation(
            attestor=attestor,
            runtime_artifact=runtime_artifact,
            question=question,
            prompt=prompt,
            generation_record=generation,
            condition=condition,
            direction=None,
            reference_rms=None,
            populate_generation=True,
            generation_grader=lambda record: replace(record, condition_id="forged"),
        )
    with pytest.raises(DataValidationError, match="does not replay"):
        validate_wikitext_likelihood_evidence(
            replace(
                executed,
                metadata={
                    **dict(executed.metadata),
                    "negative_log_likelihood": 0.4,
                },
            ),
            question=question,
        )

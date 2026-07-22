from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from mfh.contracts import (
    ActivationSite,
    AdaptivePolicySpec,
    GenerationRecord,
    Outcome,
    PromptSpec,
    Question,
    Runtime,
    TokenScope,
)
from mfh.errors import DataValidationError
from mfh.experiments.e6_likelihood import (
    E6RuntimeAttestor,
    _validate_e6_generation_runtime_evidence,
)
from mfh.experiments.e8_protected import (
    _validate_e8_adaptive_controller_record,
    execute_e6_adaptive_generation,
    execute_e8_adaptive_generation,
)
from mfh.experiments.protocol import ExperimentPhase
from mfh.experiments.runner import (
    EvaluationCondition,
    adaptive_execution_receipt_body,
)
from mfh.inference.architecture import HookKey
from mfh.inference.vllm_research import (
    VllmPromptFeatureCubeOutput,
    VllmResearchInterventionState,
)
from mfh.inference.vllm_runtime import VllmGenerationOutput, VllmRenderedPrompt
from mfh.methods.adaptive import AdaptiveBatchDecision
from mfh.methods.features import (
    ActivationFeatureSchema,
    ActivationKind,
    FeatureComposition,
)
from mfh.provenance import canonical_json, sha256_path

_PRIVATE_KEY = "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60"


def _token_digest(values: tuple[int, ...]) -> str:
    return hashlib.sha256(
        ",".join(str(value) for value in values).encode("ascii")
    ).hexdigest()


class _AdaptiveRuntime:
    def __init__(self, rendered: VllmRenderedPrompt) -> None:
        self.rendered = rendered

    def render_prompt(
        self,
        _prompt: PromptSpec,
        _question: str,
        *,
        metadata: object | None = None,
    ) -> VllmRenderedPrompt:
        return self.rendered

    def prompt_feature_cube(
        self,
        _rendered: VllmRenderedPrompt,
        *,
        layers: tuple[int, ...],
        sites: tuple[ActivationSite, ...],
    ) -> VllmPromptFeatureCubeOutput:
        assert layers == (1,)
        assert sites == (ActivationSite.POST_MLP,)
        return VllmPromptFeatureCubeOutput(
            activations={
                ActivationSite.POST_MLP: {
                    1: np.array([[0.25, -0.5]], dtype=np.float32)
                }
            },
            maximum_token_probability=0.6,
            output_entropy=0.8,
            peak_memory_bytes=123,
        )

    def standardized_intervention_state(
        self,
        direction: np.ndarray[Any, Any],
        *,
        standardized_alpha: float,
        reference_rms: float,
        token_scope: TokenScope,
    ) -> VllmResearchInterventionState:
        return VllmResearchInterventionState(
            direction=direction,
            alpha=standardized_alpha * reference_rms,
            token_scope=token_scope,
        )

    def generate_with_interventions(
        self,
        rendered: VllmRenderedPrompt,
        *,
        max_new_tokens: int,
        intervention_states: dict[
            tuple[int, ActivationSite], VllmResearchInterventionState
        ],
    ) -> VllmGenerationOutput:
        assert max_new_tokens == 8
        if intervention_states:
            state = next(iter(intervention_states.values()))
            state.captured = np.zeros((1, 1, 2), dtype=np.float32)
            state.intervened = np.array([[[state.alpha, 0.0]]], dtype=np.float32)
            state.applications = 1
        return VllmGenerationOutput(
            rendered_prompt=rendered,
            token_ids=(9,),
            text="gold",
            input_tokens=2,
            output_tokens=1,
            latency_seconds=0.25,
            stop_type="length",
            stopping_token_id=None,
            prompt_tokens_per_second=10.0,
            generation_tokens_per_second=5.0,
            peak_memory_bytes=100,
            active_memory_bytes=80,
            cache_memory_bytes=20,
        )


@pytest.mark.parametrize(
    ("abstention_threshold", "expected_action", "phase", "cross_prompt"),
    (
        (0.8, "intervene", ExperimentPhase.E8, False),
        (0.05, "release", ExperimentPhase.E8, False),
        (0.8, "intervene", ExperimentPhase.E6, True),
    ),
)
def test_native_e6_e8_m3_captures_routes_intervenes_and_signs(
    tmp_path: Path,
    monkeypatch: Any,
    abstention_threshold: float,
    expected_action: str,
    phase: ExperimentPhase,
    cross_prompt: bool,
) -> None:
    controller_prompt = PromptSpec("P0-neutral", "Answer the question.")
    prompt = (
        PromptSpec("P3-forced-answer", "Always give your best answer.")
        if cross_prompt
        else controller_prompt
    )
    prompt_sha = hashlib.sha256(prompt.text.encode()).hexdigest()
    controller_prompt_sha = hashlib.sha256(controller_prompt.text.encode()).hexdigest()
    rendered = VllmRenderedPrompt(
        text="rendered",
        sha256="a" * 64,
        token_ids=(1, 2),
        token_ids_sha256=_token_digest((1, 2)),
        messages=({"role": "user", "content": "rendered"},),
    )
    controller_path = tmp_path / "controller"
    controller_path.mkdir()
    (controller_path / "identity").write_text("controller\n", encoding="utf-8")
    controller_sha = sha256_path(controller_path)
    private_key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(_PRIVATE_KEY))
    public_key = private_key.public_key().public_bytes_raw().hex()
    schema = ActivationFeatureSchema(
        benchmark="triviaqa",
        partition="T-controller-train",
        split_manifest_digest="1" * 64,
        model_repository="model/repository",
        model_revision="0" * 40,
        runtime=Runtime.VLLM,
        quantization="1bit",
        prompt_id=controller_prompt.prompt_id,
        prompt_sha256=controller_prompt_sha,
        activation_kind=ActivationKind.FINAL_PROMPT,
        layers=(1,),
        sites=(ActivationSite.POST_MLP,),
        composition=FeatureComposition.SINGLE_LAYER,
        width=2,
    )
    hook = HookKey(1, ActivationSite.POST_MLP)

    class _Controller:
        risk_probe = SimpleNamespace(training_schema=schema)
        vector_bank = SimpleNamespace(
            directions={hook: torch.tensor([[1.0, 0.0]])},
            cluster_count=1,
        )
        alpha_controller = SimpleNamespace(
            mode=SimpleNamespace(value="fixed"), alpha_max=0.5, beta=12.0, threshold=0.4
        )
        fixed_layer = 1
        layer_selector = None

        @staticmethod
        def decide(features: torch.Tensor) -> AdaptiveBatchDecision:
            assert tuple(features.shape) == (1, 2)
            return AdaptiveBatchDecision(
                class_labels=("C", "I", "A"),
                probabilities=torch.tensor([[0.2, 0.7, 0.1]]),
                routing_weights=torch.tensor([[1.0]]),
                alphas=torch.tensor([0.5]),
                selected_layers=torch.tensor([1]),
                directions={hook: torch.tensor([[1.0, 0.0]])},
            )

    monkeypatch.setattr(
        "mfh.methods.adaptive.load_adaptive_controller", lambda _path: _Controller()
    )
    monkeypatch.setattr(
        E6RuntimeAttestor,
        "verify_runtime_artifact",
        lambda _self, _path: "e" * 64,
    )
    monkeypatch.setattr(
        E6RuntimeAttestor,
        "assert_live_runtime",
        lambda _self, _runtime: "f" * 64,
    )
    attestor = object.__new__(E6RuntimeAttestor)
    attestor.runtime = _AdaptiveRuntime(rendered)  # type: ignore[assignment]
    attestor._private_key = private_key
    attestor.execution_public_key = public_key
    attestor._artifact = {  # type: ignore[assignment]
        "runtime_identity": {"gpu_total_memory_bytes": 16 * 1024**3}
    }
    policy = AdaptivePolicySpec(
        schema_version=2,
        release_risk_threshold=0.2,
        abstention_probability_threshold=abstention_threshold,
        alpha_max=0.5,
        alpha_beta=12.0,
        layer=None,
        site=None,
        token_scope=None,
        direction_sha256=None,
        direction_norm=None,
        execution_public_key=public_key,
        controller_artifact_sha256=controller_sha,
        candidate_layers=(1,),
        candidate_sites=(ActivationSite.POST_MLP,),
        candidate_token_scopes=(TokenScope.FIRST_GENERATED,),
        vector_count=1,
        likely_unknown_risk_threshold=0.8,
        alpha_mode="fixed",
        alpha_risk_threshold=0.4,
    )
    condition = EvaluationCondition(
        phase=phase,
        benchmark="triviaqa",
        partition="T-dev",
        model_name="model",
        model_repository="model/repository",
        model_revision="0" * 40,
        runtime=Runtime.VLLM,
        quantization="1bit",
        model_num_layers=2,
        system_prompt_id=prompt.prompt_id,
        prompt_template_sha256=prompt_sha,
        steering_method="M3",
        method_artifact_sha256="f" * 64,
        layer=None,
        site=None,
        token_scope=None,
        alpha=0.0,
        sparsity=None,
        seed=17,
        study_protocol_digest="d" * 64,
        adaptive_policy=policy,
    )
    question = Question("q1", "triviaqa", "Question?", ("gold",))
    template = GenerationRecord(
        question_id=question.question_id,
        benchmark=question.benchmark,
        model_repository=condition.model_repository,
        model_revision=condition.model_revision,
        runtime=Runtime.VLLM,
        quantization=condition.quantization,
        system_prompt_id=prompt.prompt_id,
        rendered_prompt_hash=rendered.sha256,
        steering_method="M3",
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
            "phase": phase.value,
            "partition": "T-dev",
            "prompt_template_sha256": prompt_sha,
            "study_protocol_digest": "d" * 64,
            "method_artifact_sha256": "f" * 64,
        },
    )
    executor = (
        execute_e6_adaptive_generation
        if phase is ExperimentPhase.E6
        else execute_e8_adaptive_generation
    )
    executed = executor(
        attestor=attestor,
        runtime_artifact=tmp_path / "runtime.json",
        controller_artifact=controller_path,
        question=question,
        prompt=prompt,
        generation_record=template,
        condition=condition,
        max_new_tokens=8,
        controller_prompt=controller_prompt,
        populate_generation=True,
    )
    assert executed.raw_output == "gold"
    assert executed.outcome is Outcome.CORRECT
    assert executed.metadata["controller_prompt_id"] == controller_prompt.prompt_id
    assert executed.metadata["policy_action"] == expected_action
    assert executed.controller_scores == pytest.approx(
        {"C": 0.2, "I": 0.7, "A": 0.1}
    )
    if expected_action == "intervene":
        assert executed.layer == 1
        assert executed.site is ActivationSite.POST_MLP
        assert executed.token_scope is TokenScope.FIRST_GENERATED
        assert executed.metadata["intervention_trace"]["applied_token_indices"] == [0]
    else:
        assert executed.layer is None
        assert executed.site is None
        assert executed.token_scope is None
        assert "intervention_trace" not in executed.metadata
    assert executed.metadata["adaptive_controller_evidence"][
        "feature_schema_digest"
    ] == schema.digest
    condition.validate_record(executed, pending_side_effects=True)
    if phase is ExperimentPhase.E6:
        _validate_e6_generation_runtime_evidence(
            executed,
            runtime_identity={"gpu_total_memory_bytes": 16 * 1024**3},
        )
    _validate_e8_adaptive_controller_record(
        executed,
        condition=condition,
        controller=_Controller(),
        controller_artifact_sha256=controller_sha,
        controller_prompt_id=controller_prompt.prompt_id,
        controller_prompt_sha256=controller_prompt_sha,
        runtime_identity={"gpu_total_memory_bytes": 16 * 1024**3},
    )
    metrics = dict(executed.metadata["generation_runtime_metrics"])
    metrics["auxiliary_peak_memory_bytes"] = 124
    metrics["peak_memory_bytes"] = 124
    tampered = replace(
        executed,
        metadata={**dict(executed.metadata), "generation_runtime_metrics": metrics},
    )
    tampered = replace(
        tampered,
        metadata={
            **dict(tampered.metadata),
            "execution_receipt_signature": private_key.sign(
                canonical_json(
                    adaptive_execution_receipt_body(tampered, policy=policy)
                ).encode()
            ).hex(),
        },
    )
    with pytest.raises(DataValidationError, match="embedded source"):
        _validate_e8_adaptive_controller_record(
            tampered,
            condition=condition,
            controller=_Controller(),
            controller_artifact_sha256=controller_sha,
            controller_prompt_id=controller_prompt.prompt_id,
            controller_prompt_sha256=controller_prompt_sha,
            runtime_identity={"gpu_total_memory_bytes": 16 * 1024**3},
        )

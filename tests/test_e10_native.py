from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import mfh.experiments.e10_native as e10_native_module
from mfh.contracts import (
    ActivationSite,
    AdaptivePolicySpec,
    GenerationRecord,
    Outcome,
    Question,
    Runtime,
    TokenScope,
)
from mfh.errors import DataValidationError
from mfh.experiments.e8_protected import question_source_fingerprint
from mfh.experiments.e10_composite import derive_e10_composite_provenance
from mfh.experiments.e10_native import (
    NativeE10VllmBackend,
    validate_e10_composite_execution_record,
)
from mfh.experiments.gates import validate_side_effect_record
from mfh.experiments.protocol import ExperimentPhase, load_study_protocol
from mfh.experiments.runner import (
    EvaluationCondition,
    _sign_confirmatory_execution_receipt_for_test,
)
from mfh.methods.composite import CompositePolicy, CompositePolicyConfig

from .test_composite import deterministic_controller, early_probe

ROOT = Path(__file__).resolve().parents[1]
_EXECUTION_PRIVATE_KEY = "01" * 32
_EXECUTION_PUBLIC_KEY = (
    Ed25519PrivateKey.from_private_bytes(bytes.fromhex(_EXECUTION_PRIVATE_KEY))
    .public_key()
    .public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    .hex()
)


def test_native_e10_deep_loads_each_exact_policy_identity_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy, _, _, _ = _fixture()
    backend = object.__new__(NativeE10VllmBackend)
    object.__setattr__(backend, "_policy_cache", {})
    calls: list[Path] = []
    hashes: list[Path] = []

    def load_once(path: str | Path) -> CompositePolicy:
        calls.append(Path(path))
        return policy

    monkeypatch.setattr(e10_native_module, "load_composite_policy", load_once)
    monkeypatch.setattr(
        e10_native_module,
        "sha256_path",
        lambda value: hashes.append(Path(value)) or "a" * 64,
    )
    path = ROOT / "artifacts/studies/qwen36-27b-nvfp4-a10040-v1/m6"
    first = backend._load_policy(path, artifact_sha256="a" * 64)
    second = backend._load_policy(path, artifact_sha256="a" * 64)

    assert first is policy
    assert second is policy
    assert calls == [path]
    assert hashes == [path]


def test_e10_language_grade_is_alias_aware_and_tamper_evident() -> None:
    _, _, _, source = _fixture()
    raw_output = "Die Antwort ist 東京."
    record = replace(
        source,
        benchmark="language_consistency",
        question_id="language:de:1",
        raw_output=raw_output,
        normalized_answer=raw_output,
        outcome=Outcome.INCORRECT,
    )
    question = Question(
        question_id=record.question_id,
        benchmark=record.benchmark,
        text="Welche Stadt ist die Hauptstadt Japans?",
        aliases=("東京", "Tokio"),
        metadata={"requested_language": "de"},
    )
    backend = object.__new__(NativeE10VllmBackend)
    object.__setattr__(backend, "grader_bundle", SimpleNamespace(scorer=object()))
    record = replace(
        record,
        metadata={
            **dict(record.metadata),
            "source_question_sha256": question_source_fingerprint(question),
        },
    )
    graded = backend._side_grade(record, question)
    assert graded.outcome is Outcome.CORRECT
    assert graded.metadata["requested_language_correct"] is True
    assert float(graded.metadata["non_target_script_token_rate"]) > 0
    validate_side_effect_record(graded, question=question)

    evidence = dict(graded.metadata["language_evaluation_evidence"])
    evidence["accepted_aliases"] = ["Kyoto"]
    tampered = replace(
        graded,
        metadata={**dict(graded.metadata), "language_evaluation_evidence": evidence},
    )
    with pytest.raises(DataValidationError, match="contradicts"):
        validate_side_effect_record(tampered)

    forged_evidence = e10_native_module.language_response_evidence(
        raw_output, "de", ("Kyoto",)
    )
    coherently_forged = replace(
        graded,
        outcome=Outcome(str(forged_evidence["factual_outcome"])),
        metadata={
            **dict(graded.metadata),
            "detected_language": forged_evidence["detected_language"],
            "requested_language_correct": forged_evidence["requested_language_correct"],
            "non_target_script_token_rate": forged_evidence[
                "non_target_script_token_rate"
            ],
            "code_switching": forged_evidence["code_switching"],
            "language_factual_correct": forged_evidence["factual_correct"],
            "language_abstained": forged_evidence["abstained"],
            "language_evaluator_revision": forged_evidence["evaluator_revision"],
            "accepted_aliases_digest": forged_evidence["accepted_aliases_digest"],
            "language_evaluation_evidence": forged_evidence,
        },
    )
    with pytest.raises(DataValidationError, match="frozen source question"):
        validate_side_effect_record(coherently_forged, question=question)


def test_e10_provenance_derivation_requires_every_exact_prerequisite() -> None:
    study = load_study_protocol(ROOT / "configs/experiments/phases.yaml")

    with pytest.raises(DataValidationError, match="exact E0-E9"):
        derive_e10_composite_provenance(
            study=study,
            prerequisite_runs={},
        )


def _fixture() -> tuple[CompositePolicy, EvaluationCondition, Question, GenerationRecord]:
    controller = deterministic_controller()
    policy = CompositePolicy(
        controller,
        CompositePolicyConfig(
            tau_low=0.2,
            tau_high=0.7,
            release_epsilon=0.1,
            token_scope=TokenScope.FIRST_FOUR,
        ),
        early_probe=early_probe(controller),
    )
    artifact = "a" * 64
    routed = AdaptivePolicySpec(
        schema_version=2,
        release_risk_threshold=0.2,
        abstention_probability_threshold=0.7,
        alpha_max=2.0,
        alpha_beta=10.0,
        layer=None,
        site=None,
        token_scope=None,
        direction_sha256=None,
        direction_norm=None,
        execution_public_key=_EXECUTION_PUBLIC_KEY,
        controller_artifact_sha256=artifact,
        candidate_layers=(1,),
        candidate_sites=(ActivationSite.POST_MLP,),
        candidate_token_scopes=(TokenScope.FIRST_FOUR,),
        vector_count=1,
        likely_unknown_risk_threshold=0.7,
        alpha_mode="risk_gated",
        alpha_risk_threshold=0.2,
    )
    condition = EvaluationCondition(
        phase=ExperimentPhase.E10,
        benchmark="triviaqa",
        partition="T-test",
        model_name="synthetic",
        model_repository="synthetic/model",
        model_revision="synthetic",
        runtime=Runtime.SYNTHETIC,
        quantization="none",
        model_num_layers=2,
        system_prompt_id="P0-synthetic",
        prompt_template_sha256="2" * 64,
        steering_method="M6",
        method_artifact_sha256=artifact,
        layer=None,
        site=None,
        token_scope=None,
        alpha=0.0,
        sparsity=None,
        seed=17,
        study_protocol_digest="c" * 64,
        adaptive_policy=routed,
    )
    prompt_features = torch.tensor([[-3.0, 0.0]], dtype=torch.float32)
    early_features = torch.tensor([[-3.0, 0.0]], dtype=torch.float32)
    assessment = policy.assess(prompt_features)[0]
    assert policy.early_probe is not None
    probabilities = policy.early_probe.predict_probabilities(early_features)
    post_scores = {
        label: float(probabilities[0, index])
        for index, label in enumerate(policy.early_probe.state.labels)
    }
    early = policy.reevaluate_after_early_tokens(
        early_features,
        safety_ok=True,
        language_ok=True,
        refusal_drift=False,
    )
    gate = policy.output_gate(
        early.residual_risk,
        safety_ok=True,
        language_ok=True,
        refusal_drift=False,
    )
    prompt_array = np.ascontiguousarray(prompt_features.numpy(), dtype=np.float32)
    early_array = np.ascontiguousarray(early_features.numpy(), dtype=np.float32)
    raw_output = "Paris"
    early_prefix = "Par"
    evidence = {
        "schema_version": 2,
        "component_sha256": artifact,
        "prompt_regime": assessment.regime.value,
        "candidate_output": raw_output,
        "candidate_output_sha256": hashlib.sha256(raw_output.encode()).hexdigest(),
        "candidate_output_tokens": 2,
        "buffered_before_release": True,
        "early_features": {
            "schema_digest": policy.early_probe.training_schema.digest,
            "values": early_array.reshape(-1).tolist(),
            "values_sha256": hashlib.sha256(early_array.tobytes(order="C")).hexdigest(),
        },
        "early_constraints": {
            "safety_ok": True,
            "language_ok": True,
            "refusal_drift": False,
        },
        "final_constraints": {
            "safety_ok": True,
            "language_ok": True,
            "refusal_drift": False,
        },
        "online_gate_timing": {
            "mode": "online-live-stream",
            "early_gate_applied_before_completion": True,
            "feature_token_count": 1,
            "captured_token_count": 1,
            "buffered_token_count_at_gate": 1,
            "continued_after_early_gate": early.continue_generation,
            "generation_stop_type": "eos",
            "early_prefix_output": early_prefix,
            "early_prefix_output_sha256": hashlib.sha256(early_prefix.encode()).hexdigest(),
            "fallback_teacher_forced_peak_memory_bytes": 0,
        },
        "candidate_runtime_evidence": {
            "peak_memory_bytes": 1024,
            "active_memory_bytes": 512,
            "cache_memory_bytes": 128,
            "prompt_tokens_per_second": 100.0,
            "generation_tokens_per_second": 20.0,
            "latency_seconds": 0.08,
            "input_tokens": 4,
            "output_tokens": 2,
            "stop_type": "eos",
        },
        "gold_likelihood_diagnostic": None,
        "early_reevaluation": {
            "residual_risk": early.residual_risk,
            "continue_generation": early.continue_generation,
            "reason": early.reason,
            "gold_likelihood_improved": None,
            "control_decision_gold_free": True,
        },
        "output_gate": {
            "action": gate.action.value,
            "residual_risk": gate.residual_risk,
            "reason": gate.reason,
        },
        "release_epsilon": policy.config.release_epsilon,
        "final_output_sha256": hashlib.sha256(raw_output.encode()).hexdigest(),
    }
    metadata = {
        "policy_action": "release",
        "output_action": "release",
        "post_controller_scores": post_scores,
        "adaptive_controller_evidence": {
            "feature_values": prompt_array.reshape(-1).tolist(),
            "feature_values_sha256": hashlib.sha256(prompt_array.tobytes(order="C")).hexdigest(),
            "prompt_feature_peak_memory_bytes": 512,
        },
        "m6_execution_evidence": evidence,
        "generation_runtime_metrics": {
            "peak_memory_bytes": 1024,
            "candidate_peak_memory_bytes": 1024,
            "auxiliary_peak_memory_bytes": 0,
            "active_memory_bytes": 512,
            "cache_memory_bytes": 128,
            "prompt_tokens_per_second": 100.0,
            "generation_tokens_per_second": 20.0,
            "candidate_generated": True,
            "candidate_generation_seconds": 0.08,
            "end_to_end_wall_seconds": 0.1,
        },
        "decoding_max_new_tokens": 48,
        "runtime_session_identity_sha256": "e" * 64,
    }
    record = GenerationRecord(
        question_id="q1",
        benchmark="triviaqa",
        model_repository=condition.model_repository,
        model_revision=condition.model_revision,
        runtime=condition.runtime,
        quantization=condition.quantization,
        system_prompt_id=condition.system_prompt_id,
        rendered_prompt_hash="d" * 64,
        steering_method="M6",
        layer=None,
        site=None,
        token_scope=None,
        alpha=0.0,
        sparsity=None,
        controller_scores=dict(assessment.class_probabilities),
        raw_output=raw_output,
        normalized_answer="paris",
        outcome=Outcome.CORRECT,
        generation_latency_seconds=0.1,
        input_tokens=4,
        output_tokens=2,
        condition_id=condition.condition_id,
        seed=17,
        metadata=metadata,
    )
    question = Question(
        question_id="q1",
        benchmark="triviaqa",
        text="What is the capital of France?",
        aliases=("Paris",),
        split="T-test",
    )
    signed = replace(
        record,
        metadata={
            **dict(record.metadata),
            "confirmatory_execution_receipt_signature": (
                _sign_confirmatory_execution_receipt_for_test(
                    record,
                    private_key_hex=_EXECUTION_PRIVATE_KEY,
                )
            ),
        },
    )
    return policy, condition, question, signed


def test_e10_semantic_replay_rejects_forged_early_features_and_gate() -> None:
    policy, condition, question, record = _fixture()
    validate_e10_composite_execution_record(
        record, condition=condition, policy=policy, question=question
    )

    evidence = dict(record.metadata["m6_execution_evidence"])
    early = dict(evidence["early_features"])
    early["values"] = [3.0, 0.0]
    evidence["early_features"] = early
    forged = replace(
        record,
        metadata={**dict(record.metadata), "m6_execution_evidence": evidence},
    )
    with pytest.raises(DataValidationError, match="feature bytes changed"):
        validate_e10_composite_execution_record(
            forged, condition=condition, policy=policy, question=question
        )

    forged_action = replace(
        record,
        metadata={**dict(record.metadata), "output_action": "abstain"},
    )
    with pytest.raises(DataValidationError, match="output gate does not replay"):
        validate_e10_composite_execution_record(
            forged_action, condition=condition, policy=policy, question=question
        )

    runtime_metrics = dict(record.metadata["generation_runtime_metrics"])
    runtime_metrics["auxiliary_peak_memory_bytes"] = 2048
    forged_peak = replace(
        record,
        metadata={
            **dict(record.metadata),
            "generation_runtime_metrics": runtime_metrics,
        },
    )
    with pytest.raises(DataValidationError, match="peak-memory components"):
        validate_e10_composite_execution_record(
            forged_peak, condition=condition, policy=policy, question=question
        )

    coordinated_evidence = dict(record.metadata["m6_execution_evidence"])
    source = dict(coordinated_evidence["candidate_runtime_evidence"])
    source["peak_memory_bytes"] = 1
    coordinated_evidence["candidate_runtime_evidence"] = source
    coordinated_metrics = dict(record.metadata["generation_runtime_metrics"])
    coordinated_metrics["candidate_peak_memory_bytes"] = 1
    coordinated_metrics["peak_memory_bytes"] = 512
    coordinated = replace(
        record,
        metadata={
            **dict(record.metadata),
            "m6_execution_evidence": coordinated_evidence,
            "generation_runtime_metrics": coordinated_metrics,
        },
    )
    with pytest.raises(DataValidationError, match="execution receipt"):
        validate_e10_composite_execution_record(
            coordinated, condition=condition, policy=policy, question=question
        )

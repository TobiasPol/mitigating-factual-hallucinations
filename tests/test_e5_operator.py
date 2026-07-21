from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from mfh.cli import build_parser
from mfh.contracts import (
    ActivationSite,
    AdaptivePolicySpec,
    Outcome,
    Question,
    Runtime,
    TokenScope,
)
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.experiments.e5_adaptive import E5AblationRecord
from mfh.experiments.e5_native import E5NativePromotionRow, VerifiedE5NativeRun
from mfh.experiments.e5_operator import (
    E5_EXACT_GRID_RECORDS,
    E5_PROMOTED_RECORDS,
    e5_promotion_record,
    estimate_e5_native_ablation,
)
from mfh.experiments.protocol import ExperimentPhase
from mfh.experiments.runner import EvaluationCondition, validate_adaptive_execution
from mfh.provenance import stable_hash

_PRIVATE = "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60"
_PUBLIC = "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
_PROMPT_SHA = "5b08080e81d4032d853d3a30fda35e7670da6a81cc1ccf17f2b8fc5001bba684"


def _policy(*, alpha_mode: str) -> AdaptivePolicySpec:
    return AdaptivePolicySpec(
        schema_version=2,
        release_risk_threshold=0.4,
        abstention_probability_threshold=0.7,
        alpha_max=1.0,
        alpha_beta=12.0,
        layer=None,
        site=None,
        token_scope=None,
        direction_sha256=None,
        direction_norm=None,
        execution_public_key=_PUBLIC,
        sparsity=None,
        controller_artifact_sha256="c" * 64,
        candidate_layers=(31,),
        candidate_sites=(ActivationSite.POST_MLP,),
        candidate_token_scopes=(TokenScope.FIRST_FOUR,),
        vector_count=4,
        likely_unknown_risk_threshold=0.8,
        alpha_mode=alpha_mode,
        alpha_risk_threshold=0.5,
    )


def _condition(policy: AdaptivePolicySpec) -> EvaluationCondition:
    return EvaluationCondition(
        phase=ExperimentPhase.E5,
        benchmark="triviaqa",
        partition="T-dev",
        model_name="qwen3.6-27b-mlx-4bit",
        model_repository="mlx-community/Qwen3.6-27B-4bit",
        model_revision="c000ac2c2057d94be3fa931000c31723aac53282",
        runtime=Runtime.MLX,
        quantization="affine-g64-mlx-4bit",
        model_num_layers=64,
        system_prompt_id="P0-neutral",
        prompt_template_sha256=_PROMPT_SHA,
        steering_method="M3",
        method_artifact_sha256="a" * 64,
        layer=None,
        site=None,
        token_scope=None,
        alpha=0.0,
        sparsity=None,
        seed=17,
        study_protocol_digest="9" * 64,
        adaptive_policy=policy,
    )


def _native() -> VerifiedE5NativeRun:
    return VerifiedE5NativeRun(
        directory=Path("/tmp/e5-native"),
        plan=MappingProxyType(
            {
                "plan_identity": "8" * 64,
                "execution_public_key": _PUBLIC,
                "m1_policy_sha256": "a" * 64,
                "e3_static_vectors_sha256": "b" * 64,
            }
        ),
        records_completed=E5_EXACT_GRID_RECORDS,
        shard_count=1,
        chain_head="7" * 64,
        complete=True,
        scientific_eligible=True,
        maximum_peak_memory_bytes=1,
        finalized_records=Path("/tmp/e5-native/final/records.jsonl"),
        _source_context=None,  # type: ignore[arg-type]
    )


def _source_row(*, risk: float, alpha_mode: str) -> tuple[E5NativePromotionRow, Question]:
    scores = {"C": 0.9 - risk, "I": risk, "A": 0.1}
    gated_alpha = 1.0 / (1.0 + math.exp(-12.0 * (risk - 0.5)))
    alpha = (
        0.0
        if alpha_mode == "risk_gated_hard_threshold" and risk < 0.5
        else 1.0
        if alpha_mode == "fixed"
        else gated_alpha
    )
    action = "release" if alpha == 0.0 else "intervene"
    indices = [0, 1, 2] if action == "intervene" else []
    norm = alpha * 2.0 * math.sqrt(len(indices)) if indices else 0.0
    decision_body = {
        "arm_id": "b" * 64,
        "prompt_id": "P0-neutral",
        "question_id": "q-1",
        "rendered_prompt_sha256": "e" * 64,
        "prompt_input_sha256": "d" * 64,
        "controller_binding_sha256": "b" * 64,
        "controller_artifact_sha256": "c" * 64,
        "controller_scores": scores,
        "policy_action": action,
        "token_scope": TokenScope.FIRST_FOUR.value,
        "applied_token_indices": indices,
        "activation_delta_norm": norm,
    }
    receipt = {
        "controller_binding_sha256": "b" * 64,
        "controller_artifact_sha256": "c" * 64,
        "controller_scores": scores,
        "policy_action": action,
        "applied_token_indices": indices,
        "activation_delta_norm": norm,
        "decision_digest": stable_hash(decision_body),
    }
    record = E5AblationRecord(
        arm_id="b" * 64,
        prompt_id="P0-neutral",
        question_id="q-1",
        outcome=Outcome.CORRECT,
        generation_latency_seconds=1.25,
        intervention_norm=norm,
        prompt_template_sha256=_PROMPT_SHA,
        prompt_input_sha256="d" * 64,
        rendered_prompt_sha256="e" * 64,
        output_tokens=3,
        controller_binding_sha256="b" * 64,
        token_scope=TokenScope.FIRST_FOUR,
        execution_receipt=receipt,
        execution_receipt_digest=stable_hash(receipt),
        execution_receipt_signature="0" * 128,
    )
    evidence: dict[str, Any] = {
        "raw_output": "answer",
        "input_tokens": 7,
        "end_to_end_latency_seconds": 1.25,
        "generation_latency_seconds": 1.0,
        "rendered_prompt_token_ids_sha256": "f" * 64,
        "runtime_identity_sha256": "6" * 64,
        "selected_layer": 31,
        "selected_site": ActivationSite.POST_MLP.value,
        "standardized_alpha": alpha,
        "routing_weights": [0.25, 0.25, 0.25, 0.25],
        "hook_applications": len(indices),
        "applied_token_indices": indices,
        "direction_sha256": "1" * 64,
        "direction_norm": 2.0,
        "pre_activation_sha256": "2" * 64 if indices else None,
        "post_activation_sha256": "3" * 64 if indices else None,
        "delta_sha256": "4" * 64 if indices else None,
    }
    return (
        E5NativePromotionRow(
            sequence=12,
            record=record,
            evidence=MappingProxyType(evidence),
            row_digest="5" * 64,
        ),
        Question("q-1", "triviaqa", "Question?", ("answer",), split="T-dev"),
    )


def _m1_source_row() -> tuple[E5NativePromotionRow, Question]:
    decision_body = {
        "arm_id": "M1",
        "prompt_id": "P0-neutral",
        "question_id": "q-1",
        "rendered_prompt_sha256": "e" * 64,
        "prompt_input_sha256": "d" * 64,
        "controller_binding_sha256": None,
        "controller_artifact_sha256": None,
        "controller_scores": {},
        "policy_action": "intervene",
        "token_scope": TokenScope.FIRST_FOUR.value,
        "applied_token_indices": [0, 1, 2],
        "activation_delta_norm": math.sqrt(3.0),
    }
    receipt = {
        "controller_binding_sha256": None,
        "controller_artifact_sha256": None,
        "controller_scores": {},
        "policy_action": "intervene",
        "applied_token_indices": [0, 1, 2],
        "activation_delta_norm": math.sqrt(3.0),
        "decision_digest": stable_hash(decision_body),
    }
    record = E5AblationRecord(
        arm_id="M1",
        prompt_id="P0-neutral",
        question_id="q-1",
        outcome=Outcome.CORRECT,
        generation_latency_seconds=1.25,
        intervention_norm=math.sqrt(3.0),
        prompt_template_sha256=_PROMPT_SHA,
        prompt_input_sha256="d" * 64,
        rendered_prompt_sha256="e" * 64,
        output_tokens=3,
        controller_binding_sha256=None,
        token_scope=TokenScope.FIRST_FOUR,
        execution_receipt=receipt,
        execution_receipt_digest=stable_hash(receipt),
        execution_receipt_signature="0" * 128,
    )
    evidence: dict[str, Any] = {
        "raw_output": "answer",
        "input_tokens": 7,
        "end_to_end_latency_seconds": 1.25,
        "generation_latency_seconds": 1.0,
        "rendered_prompt_token_ids_sha256": "f" * 64,
        "runtime_identity_sha256": "6" * 64,
        "method_artifact_sha256": "a" * 64,
        "selected_layer": 31,
        "selected_site": ActivationSite.POST_MLP.value,
        "standardized_alpha": 1.0,
        "expected_activation_delta_norm": math.sqrt(3.0),
    }
    return (
        E5NativePromotionRow(
            sequence=0,
            record=record,
            evidence=MappingProxyType(evidence),
            row_digest="5" * 64,
        ),
        Question("q-1", "triviaqa", "Question?", ("answer",), split="T-dev"),
    )


def _m1_condition(*, layer: int = 31) -> EvaluationCondition:
    return replace(
        _condition(_policy(alpha_mode="fixed")),
        steering_method="M1",
        method_artifact_sha256="b" * 64,
        layer=layer,
        site=ActivationSite.POST_MLP,
        token_scope=TokenScope.FIRST_FOUR,
        alpha=1.0,
        adaptive_policy=None,
    )


@pytest.mark.parametrize(
    ("risk", "alpha_mode", "expected_action"),
    [
        (0.7, "risk_gated_hard_threshold", "intervene"),
        (0.2, "risk_gated_hard_threshold", "release"),
        (0.2, "risk_gated", "intervene"),
        (0.2, "fixed", "intervene"),
    ],
)
def test_e5_native_promotion_reuses_controller_action(
    risk: float,
    alpha_mode: str,
    expected_action: str,
) -> None:
    policy = _policy(alpha_mode=alpha_mode)
    source, question = _source_row(risk=risk, alpha_mode=alpha_mode)
    promoted = e5_promotion_record(
        source,
        question=question,
        condition=_condition(policy),
        adaptive_policy=policy,
        native=_native(),
        execution_private_key_hex=_PRIVATE,
    )
    assert promoted.metadata["policy_action"] == expected_action
    assert promoted.metadata["native_row_digest"] == source.row_digest
    _condition(policy).validate_record(promoted)


def test_e5_all_intervene_controller_is_valid_adaptive_execution() -> None:
    policy = _policy(alpha_mode="fixed")
    source, question = _source_row(risk=0.2, alpha_mode="fixed")
    promoted = e5_promotion_record(
        source,
        question=question,
        condition=_condition(policy),
        adaptive_policy=policy,
        native=_native(),
        execution_private_key_hex=_PRIVATE,
    )
    validate_adaptive_execution([promoted])
    with pytest.raises(DataValidationError, match="both a real intervention"):
        validate_adaptive_execution(
            [replace(promoted, metadata={**promoted.metadata, "phase": "E4"})]
        )


def test_e5_m1_promotion_binds_signed_native_geometry() -> None:
    source, question = _m1_source_row()
    promoted = e5_promotion_record(
        source,
        question=question,
        condition=_m1_condition(),
        adaptive_policy=_policy(alpha_mode="fixed"),
        native=_native(),
        execution_private_key_hex=_PRIVATE,
    )
    assert promoted.layer == 31
    with pytest.raises(FrozenArtifactError, match="signed native policy evidence"):
        e5_promotion_record(
            source,
            question=question,
            condition=_m1_condition(layer=16),
            adaptive_policy=_policy(alpha_mode="fixed"),
            native=_native(),
            execution_private_key_hex=_PRIVATE,
        )


def test_e5_runtime_estimate_is_exact() -> None:
    estimate = estimate_e5_native_ablation(
        generations_per_second=1.0,
        checkpoint_opens_per_second=2.0,
        verification_rows_per_second=10_000.0,
        request_budget=10_000,
    )
    assert estimate["exact_grid_records"] == 9_730_000 == E5_EXACT_GRID_RECORDS
    assert estimate["promoted_records"] == 20_000 == E5_PROMOTED_RECORDS
    assert estimate["generation_seconds"] == 9_730_000
    assert estimate["checkpoint_open_count"] == 973
    assert estimate["checkpoint_seconds"] == 486.5
    assert estimate["full_row_replay_passes"] == 3
    assert estimate["full_manifest_entry_passes"] == 1
    assert estimate["verification_entry_visits"] == 38_920_000
    assert estimate["verification_seconds"] == 3_892.0
    assert estimate["estimated_seconds"] == 9_734_378.5
    assert estimate["estimated_sessions"] == 973


@pytest.mark.parametrize(
    "command",
    [
        "estimate-e5-native-ablation",
        "derive-e5-selection",
        "verify-e5-selection",
        "prepare-e5-phase-ledger",
        "promote-e5-phase-records",
        "verify-e5-phase-ledger",
        "finalize-e5-phase",
        "verify-e5-phase",
    ],
)
def test_e5_operator_cli_help(command: str) -> None:
    with pytest.raises(SystemExit) as raised:
        build_parser().parse_args([command, "--help"])
    assert raised.value.code == 0

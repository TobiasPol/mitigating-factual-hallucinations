from __future__ import annotations

from mfh.experiments.gates import (
    _LEGACY_EVALUATOR_REVISIONS,
    GateDefinition,
    gate_definition,
)
from mfh.experiments.protocol import ExperimentPhase


def test_gate_revision_binds_predicate_bytecode() -> None:
    passing = GateDefinition(
        phase=ExperimentPhase.E1,
        metric_names=frozenset({"count"}),
        rule="same-rule",
        predicate=lambda metrics: int(metrics["count"]) > 0,
    )
    failing = GateDefinition(
        phase=ExperimentPhase.E1,
        metric_names=frozenset({"count"}),
        rule="same-rule",
        predicate=lambda metrics: int(metrics["count"]) < 0,
    )
    assert passing.revision != failing.revision


def test_legacy_e0_revision_whitelist_is_exact() -> None:
    assert dict(_LEGACY_EVALUATOR_REVISIONS) == {
        "chat_template_identity": frozenset(
            {"50dd47f58835f072075ba2d373fc58adf0b9a6991c832186b529b5973f75bbb5"}
        ),
        "checkpoint_identity": frozenset(
            {"3f6b1821800d9b2ec81cd64596955000be678dc2db6cd3d52827b3780ed5baf4"}
        ),
        "deterministic_decode": frozenset(
            {"fc62b00e14720ab1f9966201f4c51a6b4bc827737fb8885f78b787e6e699543e"}
        ),
        "vllm_runtime_identity": frozenset(
            {"49798c198f3e1cfc3577a7dc9b0bd390a2ccd06f532d5f4de59bb5d8072f8a42"}
        ),
    }


def test_e6_gate_rejects_registered_abstention_substitution_falsification() -> None:
    gate = gate_definition("knowledge_recovery_separated_from_abstention_substitution")
    metrics = {
        "paired_questions": 20,
        "i_to_c": 0,
        "i_to_a": 10,
        "c_to_c": 0,
        "c_to_a": 10,
        "c_to_i": 0,
        "p3_paired_questions": 10,
        "p3_i_to_c": 0,
        "p3_i_to_a": 5,
        "p3_baseline_accuracy_given_attempted": 0.5,
        "p3_intervention_accuracy_given_attempted": 0.25,
        "p3_delta_accuracy_given_attempted": -0.25,
        "mean_delta_gold_log_likelihood": -0.5,
        "mean_delta_abstention_log_likelihood": 0.5,
        "rank_paired_questions": 10,
        "mean_delta_gold_rank": 1.0,
        "rank_evidence_available": True,
        "forced_answer_complete": True,
        "correct_preservation_complete": True,
        "decomposition_complete": True,
    }
    assert gate.evaluate(metrics) is False

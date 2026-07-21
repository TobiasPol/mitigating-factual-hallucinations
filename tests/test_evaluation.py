from __future__ import annotations

import unittest
from dataclasses import replace

from mfh.contracts import GenerationRecord, Outcome, Runtime
from mfh.evaluation.grading import deterministic_short_answer_grade, triviaqa_scores
from mfh.evaluation.metrics import metric_bundle
from mfh.evaluation.risk import (
    RiskExample,
    matched_area_under_risk_coverage,
    risk_coverage_curve,
    zero_error_upper_bound,
)
from mfh.evaluation.transitions import paired_transition_summary


def record(question_id: str, outcome: Outcome, condition_id: str) -> GenerationRecord:
    return GenerationRecord(
        question_id=question_id,
        benchmark="toy",
        model_repository="toy/model",
        model_revision="synthetic-v1",
        runtime=Runtime.SYNTHETIC,
        quantization="none",
        system_prompt_id="P0",
        rendered_prompt_hash="hash",
        steering_method=condition_id,
        layer=None,
        token_scope=None,
        alpha=0.0,
        sparsity=None,
        controller_scores={},
        raw_output="answer",
        normalized_answer="answer",
        outcome=outcome,
        generation_latency_seconds=0.0,
        input_tokens=1,
        output_tokens=1,
        condition_id=condition_id,
    )


class GradingTests(unittest.TestCase):
    def test_short_answer_grading_and_alias_scores(self) -> None:
        self.assertIs(deterministic_short_answer_grade("The Paris", ("Paris",)), Outcome.CORRECT)
        self.assertIs(
            deterministic_short_answer_grade("I don't know.", ("Paris",)), Outcome.ABSTENTION
        )
        self.assertIs(deterministic_short_answer_grade("", ("Paris",)), Outcome.UNSCORABLE)
        exact, f1 = triviaqa_scores("Alexander Fleming", ("Fleming", "Alexander Fleming"))
        self.assertEqual(exact, 1.0)
        self.assertEqual(f1, 1.0)


class MetricTests(unittest.TestCase):
    def test_unified_metrics_exclude_unscorable_from_scorable_denominator(self) -> None:
        result = metric_bundle(
            [
                Outcome.CORRECT,
                Outcome.CORRECT,
                Outcome.INCORRECT,
                Outcome.ABSTENTION,
                Outcome.UNSCORABLE,
            ]
        )
        self.assertEqual(result.total, 5)
        self.assertEqual(result.scorable, 4)
        self.assertEqual(result.attempted, 3)
        self.assertAlmostEqual(result.accuracy or 0, 0.5)
        self.assertAlmostEqual(result.coverage or 0, 0.75)
        self.assertAlmostEqual(result.hallucination_risk or 0, 1 / 3)
        self.assertAlmostEqual(result.accuracy_given_attempted or 0, 2 / 3)

    def test_partial_is_an_attempt_with_configurable_credit(self) -> None:
        result = metric_bundle([Outcome.PARTIAL, Outcome.ABSTENTION], partial_credit=0.25)
        self.assertEqual(result.attempted, 1)
        self.assertAlmostEqual(result.accuracy or 0, 0.125)


class TransitionTests(unittest.TestCase):
    def test_rq2_transition_decomposition(self) -> None:
        base = [
            record("q1", Outcome.INCORRECT, "M0"),
            record("q2", Outcome.INCORRECT, "M0"),
            record("q3", Outcome.CORRECT, "M0"),
            record("q4", Outcome.CORRECT, "M0"),
        ]
        steered = [
            record("q1", Outcome.CORRECT, "M3"),
            record("q2", Outcome.ABSTENTION, "M3"),
            record("q3", Outcome.CORRECT, "M3"),
            record("q4", Outcome.INCORRECT, "M3"),
        ]
        result = paired_transition_summary(base, steered)
        self.assertEqual(result.knowledge_recovery, 0.5)
        self.assertEqual(result.abstention_substitution, 0.5)
        self.assertEqual(result.regression, 0.5)
        self.assertEqual(result.correct_preservation, 0.5)

    def test_mismatched_prompt_cannot_be_paired(self) -> None:
        base = [record("q1", Outcome.INCORRECT, "M0")]
        steered = [
            replace(
                record("q1", Outcome.CORRECT, "M3"),
                system_prompt_id="P2",
                rendered_prompt_hash="different",
            )
        ]
        with self.assertRaises(ValueError):
            paired_transition_summary(base, steered)


class RiskTests(unittest.TestCase):
    def test_risk_curve_releases_low_risk_questions_first(self) -> None:
        curve = risk_coverage_curve(
            [
                RiskExample("q1", 0.1, Outcome.CORRECT),
                RiskExample("q2", 0.2, Outcome.CORRECT),
                RiskExample("q3", 0.9, Outcome.INCORRECT),
            ]
        )
        self.assertEqual(curve[2].coverage, 2 / 3)
        self.assertEqual(curve[2].hallucination_risk, 0.0)
        self.assertAlmostEqual(curve[-1].hallucination_risk or 0, 1 / 3)

    def test_zero_error_bound_matches_rule_of_three(self) -> None:
        bound = zero_error_upper_bound(1000)
        self.assertAlmostEqual(bound, 0.002991, places=5)

    def test_unscorable_is_excluded_from_risk_coverage_denominator(self) -> None:
        curve = risk_coverage_curve(
            [
                RiskExample("q1", 0.1, Outcome.CORRECT),
                RiskExample("q2", 0.2, Outcome.UNSCORABLE),
            ]
        )
        self.assertEqual(curve[-1].coverage, 1.0)
        self.assertEqual(curve[-1].accuracy, 1.0)

    def test_duplicate_risk_id_is_rejected_even_if_one_row_is_unscorable(self) -> None:
        with self.assertRaises(ValueError):
            risk_coverage_curve(
                [
                    RiskExample("same", 0.1, Outcome.UNSCORABLE),
                    RiskExample("same", 0.2, Outcome.CORRECT),
                ]
            )

    def test_matched_aurc_uses_shared_maximum_coverage(self) -> None:
        full = risk_coverage_curve(
            [RiskExample("q1", 0.1, Outcome.CORRECT), RiskExample("q2", 0.9, Outcome.INCORRECT)]
        )
        lower = risk_coverage_curve(
            [RiskExample("q1", 0.1, Outcome.CORRECT), RiskExample("q2", 0.9, Outcome.ABSTENTION)]
        )
        limit, areas = matched_area_under_risk_coverage({"full": full, "lower": lower})
        self.assertEqual(limit, 0.5)
        self.assertEqual(set(areas), {"full", "lower"})


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import pytest

from mfh.analysis.statistics import (
    AnalysisMetric,
    MixedEffectsObservation,
    PairedOutcomes,
    bowker_test,
    fit_mixed_effects_logistic,
    holm_adjust,
    mcnemar_exact,
    paired_bootstrap_difference,
    paired_noninferiority,
    paired_prompt_interaction,
    simulate_paired_mcnemar_power,
    stuart_maxwell_test,
)
from mfh.contracts import Outcome
from mfh.errors import DataValidationError


def pair(baseline: list[Outcome], treatment: list[Outcome]) -> PairedOutcomes:
    return PairedOutcomes(
        tuple(f"q{index}" for index in range(len(baseline))),
        tuple(baseline),
        tuple(treatment),
    )


def test_paired_bootstrap_uses_question_draws_and_default_metric_direction() -> None:
    paired = pair(
        [
            Outcome.CORRECT,
            Outcome.CORRECT,
            Outcome.INCORRECT,
            Outcome.INCORRECT,
            Outcome.ABSTENTION,
            Outcome.ABSTENTION,
            Outcome.INCORRECT,
            Outcome.ABSTENTION,
        ],
        [
            Outcome.CORRECT,
            Outcome.CORRECT,
            Outcome.CORRECT,
            Outcome.CORRECT,
            Outcome.ABSTENTION,
            Outcome.ABSTENTION,
            Outcome.ABSTENTION,
            Outcome.ABSTENTION,
        ],
    )

    result = paired_bootstrap_difference(paired, AnalysisMetric.ACCURACY, resamples=1_000, seed=9)

    assert result.questions == 8
    assert result.baseline_estimate == 0.25
    assert result.treatment_estimate == 0.5
    assert result.difference == 0.25
    assert 0 < result.two_sided_p_value <= 1
    assert result.valid_resamples == 1_000
    assert result.lower <= result.difference <= result.upper


def test_paired_analysis_rejects_duplicate_ids_and_unscorable_rows() -> None:
    with pytest.raises(DataValidationError, match="unique"):
        PairedOutcomes(
            ("same", "same"),
            (Outcome.CORRECT, Outcome.INCORRECT),
            (Outcome.CORRECT, Outcome.CORRECT),
        )
    with pytest.raises(DataValidationError, match="unscorable"):
        PairedOutcomes(
            ("q1",),
            (Outcome.UNSCORABLE,),
            (Outcome.CORRECT,),
        )


def test_exact_mcnemar_uses_discordant_pairs() -> None:
    paired = pair(
        [Outcome.INCORRECT] * 4,
        [Outcome.CORRECT] * 4,
    )

    result = mcnemar_exact(paired)

    assert result.baseline_only_correct == 0
    assert result.treatment_only_correct == 4
    assert result.discordant == 4
    assert result.exact_p_value == 0.125


def test_bowker_and_stuart_maxwell_use_the_paired_transition_table() -> None:
    paired = pair(
        [
            Outcome.CORRECT,
            Outcome.INCORRECT,
            Outcome.CORRECT,
            Outcome.ABSTENTION,
            Outcome.INCORRECT,
            Outcome.ABSTENTION,
        ],
        [
            Outcome.INCORRECT,
            Outcome.CORRECT,
            Outcome.ABSTENTION,
            Outcome.CORRECT,
            Outcome.ABSTENTION,
            Outcome.INCORRECT,
        ],
    )

    bowker = bowker_test(paired)
    stuart = stuart_maxwell_test(paired)

    assert bowker.statistic == 0.0
    assert bowker.degrees_of_freedom == 3
    assert bowker.p_value == 1.0
    assert stuart.statistic == 0.0
    assert stuart.p_value == 1.0
    assert sum(sum(row) for row in bowker.transition_matrix) == 6


def test_holm_adjustment_is_monotone_in_sorted_p_values() -> None:
    adjusted = holm_adjust((("rq1", 0.01), ("rq2", 0.04), ("rq3", 0.03)))

    assert [item.hypothesis for item in adjusted] == ["rq1", "rq2", "rq3"]
    assert [item.adjusted_p_value for item in adjusted] == pytest.approx([0.03, 0.06, 0.06])
    assert [item.rejected for item in adjusted] == [True, False, False]


def test_paired_noninferiority_orients_higher_and_lower_better_metrics() -> None:
    identifiers = [f"q{index}" for index in range(20)]
    values = [float(index % 2) for index in range(20)]

    higher = paired_noninferiority(identifiers, values, values, margin=0.02, resamples=500, seed=3)
    lower = paired_noninferiority(
        identifiers,
        values,
        values,
        margin=0.02,
        higher_is_better=False,
        resamples=500,
        seed=3,
    )

    assert higher.non_inferior is True
    assert lower.non_inferior is True
    assert higher.one_sided_lower == 0.0
    assert higher.p_value == pytest.approx(1 / 501)


def test_prompt_interaction_reports_all_preregistered_gains() -> None:
    identifiers = ["q1", "q2", "q3", "q4"]
    result = paired_prompt_interaction(
        identifiers,
        [Outcome.CORRECT, Outcome.INCORRECT, Outcome.INCORRECT, Outcome.INCORRECT],
        [Outcome.CORRECT, Outcome.CORRECT, Outcome.INCORRECT, Outcome.INCORRECT],
        [Outcome.CORRECT, Outcome.CORRECT, Outcome.INCORRECT, Outcome.INCORRECT],
        [Outcome.CORRECT] * 4,
        AnalysisMetric.ACCURACY,
        resamples=500,
        seed=2,
    )

    assert result.prompt_only_gain == 0.25
    assert result.steering_only_gain == 0.25
    assert result.combined_gain == 0.75
    assert result.steering_gain_calibrated_prompt == 0.5
    assert result.interaction == 0.25
    assert 0 < result.two_sided_p_value <= 1


def test_power_simulation_uses_observed_paired_transition_rates() -> None:
    identifiers = [f"q{index}" for index in range(100)]
    baseline = [index < 40 for index in range(100)]
    treatment = [index < 35 or 40 <= index < 65 for index in range(100)]

    results = simulate_paired_mcnemar_power(
        identifiers,
        baseline,
        treatment,
        (100, 500),
        simulations=500,
        seed=4,
    )

    assert results[0].baseline_only_correct_rate == 0.05
    assert results[0].treatment_only_correct_rate == 0.25
    assert results[1].estimated_power > results[0].estimated_power


def test_mixed_effects_rejects_duplicate_question_condition_rows() -> None:
    row = MixedEffectsObservation("q1", True, "model", "benchmark", "M0", "P0")

    with pytest.raises(DataValidationError, match="duplicate"):
        fit_mixed_effects_logistic((row, row))

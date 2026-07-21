"""Question-level paired inference for the preregistered comparisons.

Every public entry point requires explicit, unique question identifiers.  This
prevents generated tokens, duplicate rows, or silently dropped grader failures
from becoming the statistical unit by accident.
"""

from __future__ import annotations

import importlib
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

import numpy as np
from scipy.stats import binomtest, chi2  # type: ignore[import-untyped]

from mfh.contracts import Outcome
from mfh.errors import DataValidationError, OptionalDependencyError
from mfh.evaluation.metrics import metric_bundle

Alternative = Literal["two-sided", "greater", "less"]


class AnalysisMetric(StrEnum):
    ACCURACY = "accuracy"
    COVERAGE = "coverage"
    HALLUCINATION_RISK = "hallucination_risk"
    ACCURACY_GIVEN_ATTEMPTED = "accuracy_given_attempted"
    ABSTENTION_RATE = "abstention_rate"
    SIMPLEQA_F1 = "simpleqa_f1"
    OMNISCIENCE_INDEX = "omniscience_index"


@dataclass(frozen=True, slots=True)
class PairedOutcomes:
    question_ids: tuple[str, ...]
    baseline: tuple[Outcome, ...]
    treatment: tuple[Outcome, ...]

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) for value in self.question_ids):
            raise DataValidationError("paired analysis question IDs must be text")
        identifiers = tuple(value.strip() for value in self.question_ids)
        if not identifiers or any(not value for value in identifiers):
            raise DataValidationError("paired analysis requires non-empty question IDs")
        if len(set(identifiers)) != len(identifiers):
            raise DataValidationError("paired analysis question IDs must be unique")
        baseline = tuple(Outcome(value) for value in self.baseline)
        treatment = tuple(Outcome(value) for value in self.treatment)
        if len(identifiers) != len(baseline) or len(identifiers) != len(treatment):
            raise DataValidationError("paired outcomes must have identical question sets")
        if Outcome.UNSCORABLE in baseline or Outcome.UNSCORABLE in treatment:
            raise DataValidationError(
                "paired confirmatory analysis cannot silently exclude unscorable grades"
            )
        object.__setattr__(self, "question_ids", identifiers)
        object.__setattr__(self, "baseline", baseline)
        object.__setattr__(self, "treatment", treatment)

    @property
    def size(self) -> int:
        return len(self.question_ids)


def _validate_resampling(resamples: int, confidence: float) -> None:
    if not isinstance(resamples, int) or isinstance(resamples, bool) or resamples < 100:
        raise DataValidationError("resamples must be an integer of at least 100")
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise DataValidationError("confidence must be finite and in (0, 1)")


def _validate_seed(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise DataValidationError("analysis seed must be a non-negative integer")


def _metric_value(
    outcomes: Sequence[Outcome], metric: AnalysisMetric, partial_credit: float
) -> float | None:
    bundle = metric_bundle(outcomes, partial_credit=partial_credit)
    if metric is AnalysisMetric.OMNISCIENCE_INDEX:
        return (
            100.0
            * (bundle.counts[Outcome.CORRECT.value] - bundle.counts[Outcome.INCORRECT.value])
            / bundle.scorable
        )
    value = getattr(bundle, metric.value)
    return float(value) if value is not None else None


def _outcome_codes(outcomes: Sequence[Outcome]) -> np.ndarray[Any, np.dtype[np.int8]]:
    code = {
        Outcome.CORRECT: 0,
        Outcome.PARTIAL: 1,
        Outcome.INCORRECT: 2,
        Outcome.ABSTENTION: 3,
    }
    try:
        return np.asarray([code[Outcome(value)] for value in outcomes], dtype=np.int8)
    except KeyError as exc:  # PairedOutcomes rejects UNSCORABLE before this helper
        raise DataValidationError("bootstrap outcomes must be finally scorable") from exc


def _sample_metric(
    values: np.ndarray[Any, np.dtype[np.int8]],
    metric: AnalysisMetric,
    partial_credit: float,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    correct = np.count_nonzero(values == 0, axis=1).astype(float)
    partial = np.count_nonzero(values == 1, axis=1).astype(float)
    incorrect = np.count_nonzero(values == 2, axis=1).astype(float)
    abstention = np.count_nonzero(values == 3, axis=1).astype(float)
    scorable = correct + partial + incorrect + abstention
    attempted = correct + partial + incorrect
    credited = correct + partial_credit * partial
    with np.errstate(divide="ignore", invalid="ignore"):
        accuracy = credited / scorable
        attempted_accuracy = credited / attempted
        if metric is AnalysisMetric.ACCURACY:
            result = accuracy
        if metric is AnalysisMetric.COVERAGE:
            result = attempted / scorable
        elif metric is AnalysisMetric.HALLUCINATION_RISK:
            result = incorrect / attempted
        elif metric is AnalysisMetric.ACCURACY_GIVEN_ATTEMPTED:
            result = attempted_accuracy
        elif metric is AnalysisMetric.ABSTENTION_RATE:
            result = abstention / scorable
        elif metric is AnalysisMetric.SIMPLEQA_F1:
            denominator = accuracy + attempted_accuracy
            result = 2 * accuracy * attempted_accuracy / denominator
        elif metric is AnalysisMetric.OMNISCIENCE_INDEX:
            result = 100 * (correct - incorrect) / scorable
        elif metric is not AnalysisMetric.ACCURACY:
            raise DataValidationError(f"unsupported bootstrap metric {metric.value}")
    return np.asarray(result, dtype=np.float64)


def _bootstrap_metric_samples(
    arrays: Sequence[np.ndarray[Any, np.dtype[np.int8]]],
    metric: AnalysisMetric,
    *,
    partial_credit: float,
    resamples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], ...]:
    size = len(arrays[0])
    if any(len(value) != size for value in arrays):
        raise DataValidationError("bootstrap arrays must share one question set")
    chunk_size = max(1, min(resamples, 4_000_000 // size))
    sampled: list[list[np.ndarray[Any, np.dtype[np.float64]]]] = [
        [] for _ in arrays
    ]
    remaining = resamples
    while remaining:
        current = min(chunk_size, remaining)
        indices = rng.integers(0, size, size=(current, size))
        for destination, values in zip(sampled, arrays, strict=True):
            destination.append(
                _sample_metric(values[indices], metric, partial_credit)
            )
        remaining -= current
    return tuple(np.concatenate(values) for values in sampled)


@dataclass(frozen=True, slots=True)
class PairedBootstrapResult:
    metric: AnalysisMetric
    questions: int
    baseline_estimate: float
    treatment_estimate: float
    difference: float
    confidence: float
    lower: float
    upper: float
    two_sided_p_value: float
    resamples: int
    valid_resamples: int
    seed: int


def paired_bootstrap_difference(
    paired: PairedOutcomes,
    metric: AnalysisMetric | str,
    *,
    partial_credit: float = 0.5,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 17,
) -> PairedBootstrapResult:
    """Percentile CI for treatment minus baseline using shared question draws."""

    _validate_resampling(resamples, confidence)
    _validate_seed(seed)
    selected = AnalysisMetric(metric)
    baseline_estimate = _metric_value(paired.baseline, selected, partial_credit)
    treatment_estimate = _metric_value(paired.treatment, selected, partial_credit)
    if baseline_estimate is None or treatment_estimate is None:
        raise DataValidationError(f"metric {selected.value} is undefined on the observed data")
    rng = np.random.default_rng(seed)
    sampled_baseline, sampled_treatment = _bootstrap_metric_samples(
        (_outcome_codes(paired.baseline), _outcome_codes(paired.treatment)),
        selected,
        partial_credit=partial_credit,
        resamples=resamples,
        rng=rng,
    )
    raw_differences = sampled_treatment - sampled_baseline
    differences = raw_differences[np.isfinite(raw_differences)]
    minimum_valid = max(1, math.ceil(0.9 * resamples))
    if len(differences) < minimum_valid:
        raise DataValidationError(
            f"metric {selected.value} was undefined in too many bootstrap resamples"
        )
    tail = (1 - confidence) / 2
    lower, upper = np.quantile(differences, [tail, 1 - tail])
    observed_difference = treatment_estimate - baseline_estimate
    null_differences = differences - observed_difference
    p_value = float(
        (1 + np.count_nonzero(np.abs(null_differences) >= abs(observed_difference)))
        / (len(differences) + 1)
    )
    return PairedBootstrapResult(
        metric=selected,
        questions=paired.size,
        baseline_estimate=baseline_estimate,
        treatment_estimate=treatment_estimate,
        difference=observed_difference,
        confidence=confidence,
        lower=float(lower),
        upper=float(upper),
        two_sided_p_value=p_value,
        resamples=resamples,
        valid_resamples=len(differences),
        seed=seed,
    )


@dataclass(frozen=True, slots=True)
class McNemarResult:
    questions: int
    baseline_only_correct: int
    treatment_only_correct: int
    discordant: int
    alternative: Alternative
    exact_p_value: float


def mcnemar_exact(
    paired: PairedOutcomes, *, alternative: Alternative = "two-sided"
) -> McNemarResult:
    """Exact McNemar test; greater means treatment has more correctness gains."""

    if alternative not in {"two-sided", "greater", "less"}:
        raise DataValidationError("unsupported McNemar alternative")
    baseline_correct = np.asarray(
        [value is Outcome.CORRECT for value in paired.baseline], dtype=bool
    )
    treatment_correct = np.asarray(
        [value is Outcome.CORRECT for value in paired.treatment], dtype=bool
    )
    baseline_only = int(np.count_nonzero(baseline_correct & ~treatment_correct))
    treatment_only = int(np.count_nonzero(~baseline_correct & treatment_correct))
    discordant = baseline_only + treatment_only
    p_value = (
        float(
            binomtest(
                treatment_only,
                discordant,
                p=0.5,
                alternative=alternative,
            ).pvalue
        )
        if discordant
        else 1.0
    )
    return McNemarResult(
        questions=paired.size,
        baseline_only_correct=baseline_only,
        treatment_only_correct=treatment_only,
        discordant=discordant,
        alternative=alternative,
        exact_p_value=p_value,
    )


@dataclass(frozen=True, slots=True)
class MarginalHomogeneityResult:
    test: str
    labels: tuple[Outcome, ...]
    transition_matrix: tuple[tuple[int, ...], ...]
    statistic: float
    degrees_of_freedom: int
    p_value: float


def _transition_matrix(
    paired: PairedOutcomes, labels: Sequence[Outcome]
) -> tuple[tuple[Outcome, ...], np.ndarray[Any, np.dtype[np.int64]]]:
    normalized = tuple(Outcome(value) for value in labels)
    if len(normalized) < 2 or len(set(normalized)) != len(normalized):
        raise DataValidationError(
            "transition labels must be unique and contain at least two values"
        )
    observed = set(paired.baseline) | set(paired.treatment)
    if observed - set(normalized):
        omitted = sorted(value.value for value in observed - set(normalized))
        raise DataValidationError(f"transition labels omit outcomes {omitted}")
    indices = {label: index for index, label in enumerate(normalized)}
    table = np.zeros((len(normalized), len(normalized)), dtype=np.int64)
    for before, after in zip(paired.baseline, paired.treatment, strict=True):
        table[indices[before], indices[after]] += 1
    return normalized, table


def bowker_test(
    paired: PairedOutcomes,
    *,
    labels: Sequence[Outcome] = (Outcome.CORRECT, Outcome.INCORRECT, Outcome.ABSTENTION),
) -> MarginalHomogeneityResult:
    """Bowker's symmetry test for paired multi-category outcomes."""

    normalized, table = _transition_matrix(paired, labels)
    statistic = 0.0
    degrees = len(normalized) * (len(normalized) - 1) // 2
    for row in range(len(normalized)):
        for column in range(row + 1, len(normalized)):
            denominator = int(table[row, column] + table[column, row])
            if denominator:
                difference = int(table[row, column] - table[column, row])
                statistic += difference**2 / denominator
    p_value = float(chi2.sf(statistic, degrees))
    return MarginalHomogeneityResult(
        test="Bowker",
        labels=normalized,
        transition_matrix=tuple(
            tuple(int(table[row, column]) for column in range(len(normalized)))
            for row in range(len(normalized))
        ),
        statistic=statistic,
        degrees_of_freedom=degrees,
        p_value=p_value,
    )


def stuart_maxwell_test(
    paired: PairedOutcomes,
    *,
    labels: Sequence[Outcome] = (Outcome.CORRECT, Outcome.INCORRECT, Outcome.ABSTENTION),
) -> MarginalHomogeneityResult:
    """Stuart-Maxwell marginal-homogeneity test with rank-aware covariance."""

    normalized, table = _transition_matrix(paired, labels)
    dimension = len(normalized) - 1
    row_totals = table.sum(axis=1)
    column_totals = table.sum(axis=0)
    differences = (row_totals - column_totals)[:dimension].astype(float)
    covariance = np.zeros((dimension, dimension), dtype=float)
    for row in range(dimension):
        covariance[row, row] = row_totals[row] + column_totals[row] - 2 * table[row, row]
        for column in range(dimension):
            if row != column:
                covariance[row, column] = -(table[row, column] + table[column, row])
    degrees = int(np.linalg.matrix_rank(covariance))
    statistic = float(differences @ np.linalg.pinv(covariance) @ differences) if degrees else 0.0
    p_value = float(chi2.sf(statistic, degrees)) if degrees else 1.0
    return MarginalHomogeneityResult(
        test="Stuart-Maxwell",
        labels=normalized,
        transition_matrix=tuple(
            tuple(int(table[row, column]) for column in range(len(normalized)))
            for row in range(len(normalized))
        ),
        statistic=statistic,
        degrees_of_freedom=degrees,
        p_value=p_value,
    )


@dataclass(frozen=True, slots=True)
class AdjustedHypothesis:
    hypothesis: str
    raw_p_value: float
    adjusted_p_value: float
    rejected: bool


def holm_adjust(
    p_values: Iterable[tuple[str, float]], *, alpha: float = 0.05
) -> tuple[AdjustedHypothesis, ...]:
    """Holm family-wise correction, returned in the caller's original order."""

    values = tuple((str(name).strip(), float(value)) for name, value in p_values)
    if not values or any(not name for name, _ in values):
        raise DataValidationError("Holm correction requires named hypotheses")
    if len({name for name, _ in values}) != len(values):
        raise DataValidationError("Holm hypothesis names must be unique")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise DataValidationError("Holm alpha must be in (0, 1)")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for _, value in values):
        raise DataValidationError("Holm p-values must be finite and in [0, 1]")
    ordered = sorted(enumerate(values), key=lambda item: (item[1][1], item[0]))
    adjusted: dict[int, float] = {}
    running = 0.0
    total = len(values)
    for rank, (original_index, (_, raw)) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * raw))
        adjusted[original_index] = running
    return tuple(
        AdjustedHypothesis(name, raw, adjusted[index], adjusted[index] <= alpha)
        for index, (name, raw) in enumerate(values)
    )


def _paired_numeric(
    question_ids: Sequence[str], baseline: Sequence[float], treatment: Sequence[float]
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]]:
    if any(not isinstance(value, str) for value in question_ids):
        raise DataValidationError("numeric paired question IDs must be text")
    identifiers = tuple(value.strip() for value in question_ids)
    if (
        not identifiers
        or len(set(identifiers)) != len(identifiers)
        or any(not value for value in identifiers)
    ):
        raise DataValidationError("numeric paired analysis requires unique question IDs")
    before = np.asarray(baseline, dtype=float)
    after = np.asarray(treatment, dtype=float)
    if (
        before.ndim != 1
        or after.ndim != 1
        or len(identifiers) != len(before)
        or len(identifiers) != len(after)
    ):
        raise DataValidationError("numeric paired values must match the question set")
    if not np.all(np.isfinite(before)) or not np.all(np.isfinite(after)):
        raise DataValidationError("numeric paired values must be finite")
    return before, after


@dataclass(frozen=True, slots=True)
class NonInferiorityResult:
    questions: int
    margin: float
    higher_is_better: bool
    oriented_difference: float
    confidence: float
    one_sided_lower: float
    p_value: float
    non_inferior: bool
    resamples: int
    seed: int


def paired_noninferiority(
    question_ids: Sequence[str],
    baseline: Sequence[float],
    treatment: Sequence[float],
    *,
    margin: float,
    higher_is_better: bool = True,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 17,
) -> NonInferiorityResult:
    """One-sided paired bootstrap test against the boundary ``difference=-margin``."""

    _validate_resampling(resamples, confidence)
    _validate_seed(seed)
    if not math.isfinite(margin) or margin <= 0:
        raise DataValidationError("non-inferiority margin must be finite and positive")
    before, after = _paired_numeric(question_ids, baseline, treatment)
    differences = after - before if higher_is_better else before - after
    observed = float(np.mean(differences))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(resamples, len(differences)))
    sampled = differences[indices].mean(axis=1)
    lower = float(np.quantile(sampled, 1 - confidence))
    null_centered = differences - observed - margin
    null_sampled = null_centered[indices].mean(axis=1)
    p_value = float((1 + np.count_nonzero(null_sampled >= observed)) / (resamples + 1))
    return NonInferiorityResult(
        questions=len(differences),
        margin=margin,
        higher_is_better=higher_is_better,
        oriented_difference=observed,
        confidence=confidence,
        one_sided_lower=lower,
        p_value=p_value,
        non_inferior=lower > -margin,
        resamples=resamples,
        seed=seed,
    )


@dataclass(frozen=True, slots=True)
class PromptInteractionResult:
    metric: AnalysisMetric
    questions: int
    prompt_only_gain: float
    steering_only_gain: float
    combined_gain: float
    steering_gain_calibrated_prompt: float
    interaction: float
    confidence: float
    lower: float
    upper: float
    two_sided_p_value: float
    resamples: int
    valid_resamples: int
    seed: int


def paired_prompt_interaction(
    question_ids: Sequence[str],
    baseline_neutral: Sequence[Outcome],
    treatment_neutral: Sequence[Outcome],
    baseline_calibrated: Sequence[Outcome],
    treatment_calibrated: Sequence[Outcome],
    metric: AnalysisMetric | str,
    *,
    partial_credit: float = 0.5,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 17,
) -> PromptInteractionResult:
    """Paired difference-in-differences for the Method x Prompt interaction."""

    _validate_resampling(resamples, confidence)
    _validate_seed(seed)
    identifiers = tuple(question_ids)
    neutral = PairedOutcomes(identifiers, tuple(baseline_neutral), tuple(treatment_neutral))
    calibrated = PairedOutcomes(
        identifiers, tuple(baseline_calibrated), tuple(treatment_calibrated)
    )
    if neutral.question_ids != calibrated.question_ids:
        raise DataValidationError("interaction conditions must use identical question order")
    selected = AnalysisMetric(metric)
    conditions = (
        neutral.baseline,
        neutral.treatment,
        calibrated.baseline,
        calibrated.treatment,
    )
    observed = tuple(_metric_value(value, selected, partial_credit) for value in conditions)
    if any(value is None for value in observed):
        raise DataValidationError(f"metric {selected.value} is undefined for an interaction cell")
    (
        baseline_neutral_value,
        treatment_neutral_value,
        baseline_calibrated_value,
        treatment_calibrated_value,
    ) = (float(value) for value in observed if value is not None)
    rng = np.random.default_rng(seed)
    sampled = _bootstrap_metric_samples(
        tuple(_outcome_codes(value) for value in conditions),
        selected,
        partial_credit=partial_credit,
        resamples=resamples,
        rng=rng,
    )
    raw_interactions = (sampled[3] - sampled[2]) - (sampled[1] - sampled[0])
    interactions = raw_interactions[np.isfinite(raw_interactions)]
    minimum_valid = max(1, math.ceil(0.9 * resamples))
    if len(interactions) < minimum_valid:
        raise DataValidationError("interaction metric was undefined in too many resamples")
    tail = (1 - confidence) / 2
    lower, upper = np.quantile(interactions, [tail, 1 - tail])
    steering_neutral = treatment_neutral_value - baseline_neutral_value
    steering_calibrated = treatment_calibrated_value - baseline_calibrated_value
    observed_interaction = steering_calibrated - steering_neutral
    null_interactions = interactions - observed_interaction
    p_value = float(
        (1 + np.count_nonzero(np.abs(null_interactions) >= abs(observed_interaction)))
        / (len(interactions) + 1)
    )
    return PromptInteractionResult(
        metric=selected,
        questions=neutral.size,
        prompt_only_gain=baseline_calibrated_value - baseline_neutral_value,
        steering_only_gain=steering_neutral,
        combined_gain=treatment_calibrated_value - baseline_neutral_value,
        steering_gain_calibrated_prompt=steering_calibrated,
        interaction=observed_interaction,
        confidence=confidence,
        lower=float(lower),
        upper=float(upper),
        two_sided_p_value=p_value,
        resamples=resamples,
        valid_resamples=len(interactions),
        seed=seed,
    )


@dataclass(frozen=True, slots=True)
class PairedPowerResult:
    observed_questions: int
    baseline_only_correct_rate: float
    treatment_only_correct_rate: float
    target_sample_size: int
    simulations: int
    alpha: float
    alternative: Alternative
    estimated_power: float
    seed: int


def simulate_paired_mcnemar_power(
    question_ids: Sequence[str],
    baseline_correct: Sequence[bool],
    treatment_correct: Sequence[bool],
    target_sample_sizes: Sequence[int],
    *,
    simulations: int = 10_000,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
    seed: int = 17,
) -> tuple[PairedPowerResult, ...]:
    """Simulate power from observed paired discordant-transition rates."""

    if not isinstance(simulations, int) or isinstance(simulations, bool) or simulations < 100:
        raise DataValidationError("power simulations must be an integer of at least 100")
    _validate_seed(seed)
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise DataValidationError("power alpha must be in (0, 1)")
    if alternative not in {"two-sided", "greater", "less"}:
        raise DataValidationError("unsupported power-test alternative")
    if any(not isinstance(value, (bool, np.bool_)) for value in baseline_correct) or any(
        not isinstance(value, (bool, np.bool_)) for value in treatment_correct
    ):
        raise DataValidationError("paired power outcomes must be boolean correctness values")
    before, after = _paired_numeric(question_ids, baseline_correct, treatment_correct)
    before_bool = before.astype(bool)
    after_bool = after.astype(bool)
    baseline_only = int(np.count_nonzero(before_bool & ~after_bool))
    treatment_only = int(np.count_nonzero(~before_bool & after_bool))
    baseline_rate = baseline_only / len(before)
    treatment_rate = treatment_only / len(before)
    if any(not isinstance(value, int) or isinstance(value, bool) for value in target_sample_sizes):
        raise DataValidationError("target sample sizes must be integers")
    sample_sizes = tuple(target_sample_sizes)
    if (
        not sample_sizes
        or len(set(sample_sizes)) != len(sample_sizes)
        or any(value <= 0 for value in sample_sizes)
    ):
        raise DataValidationError("target sample sizes must be unique positive integers")
    rng = np.random.default_rng(seed)
    results: list[PairedPowerResult] = []
    probabilities = (
        baseline_rate,
        treatment_rate,
        max(0.0, 1 - baseline_rate - treatment_rate),
    )
    for sample_size in sample_sizes:
        draws = rng.multinomial(sample_size, probabilities, size=simulations)
        cache: dict[tuple[int, int], float] = {}
        rejections = 0
        for simulated_baseline_only, simulated_treatment_only, _ in draws:
            key = (int(simulated_baseline_only), int(simulated_treatment_only))
            if key not in cache:
                discordant = key[0] + key[1]
                cache[key] = (
                    float(binomtest(key[1], discordant, p=0.5, alternative=alternative).pvalue)
                    if discordant
                    else 1.0
                )
            rejections += int(cache[key] <= alpha)
        results.append(
            PairedPowerResult(
                observed_questions=len(before),
                baseline_only_correct_rate=baseline_rate,
                treatment_only_correct_rate=treatment_rate,
                target_sample_size=sample_size,
                simulations=simulations,
                alpha=alpha,
                alternative=alternative,
                estimated_power=rejections / simulations,
                seed=seed,
            )
        )
    return tuple(results)


@dataclass(frozen=True, slots=True)
class MixedEffectsObservation:
    question_id: str
    correct: bool
    model: str
    benchmark: str
    method: str
    prompt: str

    def __post_init__(self) -> None:
        for name in ("question_id", "model", "benchmark", "method", "prompt"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise DataValidationError(f"mixed-effects {name} must be non-empty text")
            object.__setattr__(self, name, value.strip())
        if not isinstance(self.correct, bool):
            raise DataValidationError("mixed-effects correct response must be boolean")


@dataclass(frozen=True, slots=True)
class MixedEffectsLogisticResult:
    formula: str
    random_effects: str
    estimator: str
    observations: int
    questions: int
    fixed_effect_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    standard_errors: tuple[float, ...]
    converged: bool


def fit_mixed_effects_logistic(
    observations: Iterable[MixedEffectsObservation],
) -> MixedEffectsLogisticResult:
    """Fit correctness with a question random intercept and preregistered fixed effects.

    The optional research dependency uses a Laplace MAP binomial mixed model.
    Its estimator identity is returned explicitly so it cannot be mistaken for
    a frequentist maximum-likelihood fit.
    """

    values = tuple(observations)
    if not values:
        raise DataValidationError("mixed-effects analysis requires observations")
    design_rows = {
        (value.question_id, value.model, value.benchmark, value.method, value.prompt)
        for value in values
    }
    if len(design_rows) != len(values):
        raise DataValidationError("mixed-effects design contains duplicate question-condition rows")
    if len({value.correct for value in values}) < 2:
        raise DataValidationError("mixed-effects correctness response must contain both classes")
    counts = Counter(value.question_id for value in values)
    if any(count < 2 for count in counts.values()):
        raise DataValidationError(
            "question random intercepts require at least two conditions per question"
        )
    try:
        pandas: Any = importlib.import_module("pandas")
        mixed: Any = importlib.import_module("statsmodels.genmod.bayes_mixed_glm")
    except ImportError as exc:
        raise OptionalDependencyError(
            "mixed-effects fitting requires the 'research' optional dependencies"
        ) from exc
    formula = "correct ~ C(model) + C(benchmark) + C(method) * C(prompt)"
    random_effects = "0 + C(question_id)"
    frame = pandas.DataFrame(
        [
            {
                "question_id": value.question_id,
                "correct": int(value.correct),
                "model": value.model,
                "benchmark": value.benchmark,
                "method": value.method,
                "prompt": value.prompt,
            }
            for value in values
        ]
    )
    model = mixed.BinomialBayesMixedGLM.from_formula(
        formula, {"question_intercept": random_effects}, frame
    )
    fitted = model.fit_map()
    optimizer = getattr(fitted, "optim_retvals", {})
    converged = bool(optimizer.get("success", False)) if hasattr(optimizer, "get") else False
    return MixedEffectsLogisticResult(
        formula=formula,
        random_effects=random_effects,
        estimator="statsmodels.BinomialBayesMixedGLM.fit_map",
        observations=len(values),
        questions=len(counts),
        fixed_effect_names=tuple(str(value) for value in model.exog_names),
        coefficients=tuple(float(value) for value in fitted.fe_mean),
        standard_errors=tuple(float(value) for value in fitted.fe_sd),
        converged=converged,
    )

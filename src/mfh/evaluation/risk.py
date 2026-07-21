"""Risk-coverage curves and zero-error confidence bounds."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import pairwise

from mfh.contracts import Outcome
from mfh.errors import DataValidationError


@dataclass(frozen=True, slots=True)
class RiskExample:
    question_id: str
    predicted_risk: float
    outcome_if_released: Outcome

    def __post_init__(self) -> None:
        if not math.isfinite(self.predicted_risk) or not 0 <= self.predicted_risk <= 1:
            raise DataValidationError("predicted risk must be finite and in [0, 1]")


@dataclass(frozen=True, slots=True)
class RiskCoveragePoint:
    threshold: float
    coverage: float
    hallucination_risk: float | None
    accuracy: float
    attempted: int
    incorrect: int


def risk_coverage_curve(examples: Iterable[RiskExample]) -> tuple[RiskCoveragePoint, ...]:
    all_values = list(examples)
    identifiers = [item.question_id for item in all_values]
    if len(set(identifiers)) != len(identifiers):
        raise DataValidationError("risk examples must have unique question IDs")
    values = sorted(
        (item for item in all_values if item.outcome_if_released is not Outcome.UNSCORABLE),
        key=lambda item: (item.predicted_risk, item.question_id),
    )
    if not values:
        return ()
    total = len(values)
    points = [
        RiskCoveragePoint(
            threshold=-math.inf,
            coverage=0.0,
            hallucination_risk=None,
            accuracy=0.0,
            attempted=0,
            incorrect=0,
        )
    ]
    correct = partial = incorrect = attempted = 0
    index = 0
    while index < total:
        threshold = values[index].predicted_risk
        while index < total and values[index].predicted_risk == threshold:
            outcome = values[index].outcome_if_released
            if outcome.is_attempted:
                attempted += 1
                correct += int(outcome is Outcome.CORRECT)
                partial += int(outcome is Outcome.PARTIAL)
                incorrect += int(outcome is Outcome.INCORRECT)
            index += 1
        points.append(
            RiskCoveragePoint(
                threshold=threshold,
                coverage=attempted / total,
                hallucination_risk=incorrect / attempted if attempted else None,
                accuracy=(correct + 0.5 * partial) / total,
                attempted=attempted,
                incorrect=incorrect,
            )
        )
    return tuple(points)


def area_under_risk_coverage(
    points: Iterable[RiskCoveragePoint], *, coverage_limit: float
) -> float:
    """Integrate only over an explicitly matched coverage domain."""

    ordered = sorted(points, key=lambda point: point.coverage)
    if not 0 <= coverage_limit <= 1:
        raise DataValidationError("coverage_limit must be in [0, 1]")
    if not ordered or ordered[-1].coverage < coverage_limit:
        maximum = ordered[-1].coverage if ordered else 0.0
        raise DataValidationError(
            f"curve reaches coverage {maximum:.6f}, below requested limit {coverage_limit:.6f}"
        )
    if len(ordered) < 2 or coverage_limit == 0:
        return 0.0
    area = 0.0
    for left, right in pairwise(ordered):
        if left.coverage >= coverage_limit:
            break
        # Selective risk is undefined at zero released answers.  The first
        # observed risk therefore extends left to zero coverage; assigning zero
        # risk there would manufacture an optimistic triangular area.
        if left.hallucination_risk is None and right.hallucination_risk is None:
            continue
        right_risk = (
            right.hallucination_risk
            if right.hallucination_risk is not None
            else left.hallucination_risk
        )
        left_risk = (
            left.hallucination_risk
            if left.hallucination_risk is not None
            else right_risk
        )
        assert left_risk is not None and right_risk is not None
        segment_right = min(right.coverage, coverage_limit)
        if right.coverage == left.coverage:
            continue
        fraction = (segment_right - left.coverage) / (right.coverage - left.coverage)
        interpolated_risk = left_risk + fraction * (right_risk - left_risk)
        area += (segment_right - left.coverage) * (left_risk + interpolated_risk) / 2
        if segment_right == coverage_limit:
            break
    return area


def matched_area_under_risk_coverage(
    curves: Mapping[str, Iterable[RiskCoveragePoint]],
) -> tuple[float, Mapping[str, float]]:
    """Compare methods over the largest coverage interval every curve reaches."""

    materialized = {name: tuple(points) for name, points in curves.items()}
    if not materialized or any(not points for points in materialized.values()):
        raise DataValidationError("matched AURC requires non-empty curves")
    coverage_limit = min(
        max(point.coverage for point in points) for points in materialized.values()
    )
    return coverage_limit, {
        name: area_under_risk_coverage(points, coverage_limit=coverage_limit)
        for name, points in materialized.items()
    }


def zero_error_upper_bound(attempted: int, *, confidence: float = 0.95) -> float:
    """Exact one-sided Clopper-Pearson upper bound when zero errors are observed."""

    if attempted <= 0:
        raise DataValidationError("attempted must be positive")
    if not 0 < confidence < 1:
        raise DataValidationError("confidence must be in (0, 1)")
    return float(1 - (1 - confidence) ** (1 / attempted))

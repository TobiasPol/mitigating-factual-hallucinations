"""Coverage, risk, accuracy, and abstention metrics from unified labels."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

from mfh.contracts import Outcome
from mfh.errors import DataValidationError


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


@dataclass(frozen=True, slots=True)
class MetricBundle:
    counts: Mapping[str, int]
    total: int
    scorable: int
    attempted: int
    accuracy: float | None
    coverage: float | None
    hallucination_risk: float | None
    accuracy_given_attempted: float | None
    abstention_rate: float | None
    unscorable_rate: float | None
    simpleqa_f1: float | None
    partial_credit: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def metric_bundle(outcomes: Iterable[Outcome], *, partial_credit: float = 0.5) -> MetricBundle:
    if not 0 <= partial_credit <= 1:
        raise DataValidationError("partial_credit must be in [0, 1]")
    counter = Counter(outcomes)
    total = sum(counter.values())
    unscorable = counter[Outcome.UNSCORABLE]
    scorable = total - unscorable
    attempted = counter[Outcome.CORRECT] + counter[Outcome.PARTIAL] + counter[Outcome.INCORRECT]
    credited = counter[Outcome.CORRECT] + partial_credit * counter[Outcome.PARTIAL]
    accuracy = _ratio(credited, scorable)
    accuracy_given_attempted = _ratio(credited, attempted)
    if (
        accuracy is None
        or accuracy_given_attempted is None
        or accuracy + accuracy_given_attempted == 0
    ):
        simpleqa_f1 = None
    else:
        simpleqa_f1 = (
            2 * accuracy * accuracy_given_attempted / (accuracy + accuracy_given_attempted)
        )
    return MetricBundle(
        counts={outcome.value: counter[outcome] for outcome in Outcome},
        total=total,
        scorable=scorable,
        attempted=attempted,
        accuracy=accuracy,
        coverage=_ratio(attempted, scorable),
        hallucination_risk=_ratio(counter[Outcome.INCORRECT], attempted),
        accuracy_given_attempted=accuracy_given_attempted,
        abstention_rate=_ratio(counter[Outcome.ABSTENTION], scorable),
        unscorable_rate=_ratio(unscorable, total),
        simpleqa_f1=simpleqa_f1,
        partial_credit=partial_credit,
    )

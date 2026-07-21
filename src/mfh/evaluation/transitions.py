"""Paired base-to-intervention transition decomposition for RQ2."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

from mfh.contracts import GenerationRecord, Outcome
from mfh.errors import DataValidationError


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


@dataclass(frozen=True, slots=True)
class TransitionSummary:
    matrix: Mapping[str, int]
    paired_questions: int
    knowledge_recovery: float | None
    abstention_substitution: float | None
    strict_over_refusal: float | None
    regression: float | None
    correct_preservation: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _index(records: Iterable[GenerationRecord], name: str) -> dict[str, GenerationRecord]:
    result: dict[str, GenerationRecord] = {}
    condition_ids: set[str] = set()
    for record in records:
        if record.question_id in result:
            raise DataValidationError(
                f"{name} contains duplicate question_id {record.question_id!r}"
            )
        result[record.question_id] = record
        condition_ids.add(record.condition_id)
    if len(condition_ids) > 1:
        raise DataValidationError(
            f"{name} must contain exactly one condition, found {sorted(condition_ids)}"
        )
    return result


def _pairing_signature(record: GenerationRecord) -> tuple[object, ...]:
    return (
        record.benchmark,
        record.model_repository,
        record.model_revision,
        record.runtime,
        record.quantization,
        record.system_prompt_id,
        record.rendered_prompt_hash,
        record.seed,
        record.input_tokens,
    )


def paired_transition_summary(
    base_records: Iterable[GenerationRecord], intervened_records: Iterable[GenerationRecord]
) -> TransitionSummary:
    base = _index(base_records, "base records")
    intervened = _index(intervened_records, "intervened records")
    if base.keys() != intervened.keys():
        missing_intervened = sorted(base.keys() - intervened.keys())[:10]
        missing_base = sorted(intervened.keys() - base.keys())[:10]
        raise DataValidationError(
            "paired conditions must contain identical question IDs; "
            f"missing_intervened={missing_intervened}, missing_base={missing_base}"
        )
    mismatches = [
        key
        for key in sorted(base)
        if _pairing_signature(base[key]) != _pairing_signature(intervened[key])
    ]
    if mismatches:
        raise DataValidationError(
            "paired records differ in benchmark/model/prompt/seed/input dimensions for "
            f"question IDs: {mismatches[:10]}"
        )
    transitions = Counter((base[key].outcome, intervened[key].outcome) for key in sorted(base))
    base_incorrect = sum(
        count for (before, _), count in transitions.items() if before is Outcome.INCORRECT
    )
    base_correct = sum(
        count for (before, _), count in transitions.items() if before is Outcome.CORRECT
    )
    return TransitionSummary(
        matrix={
            f"{before.value}->{after.value}": transitions[(before, after)]
            for before in Outcome
            for after in Outcome
        },
        paired_questions=len(base),
        knowledge_recovery=_ratio(
            transitions[(Outcome.INCORRECT, Outcome.CORRECT)], base_incorrect
        ),
        abstention_substitution=_ratio(
            transitions[(Outcome.INCORRECT, Outcome.ABSTENTION)], base_incorrect
        ),
        strict_over_refusal=_ratio(
            transitions[(Outcome.CORRECT, Outcome.ABSTENTION)], base_correct
        ),
        regression=_ratio(transitions[(Outcome.CORRECT, Outcome.INCORRECT)], base_correct),
        correct_preservation=_ratio(transitions[(Outcome.CORRECT, Outcome.CORRECT)], base_correct),
    )

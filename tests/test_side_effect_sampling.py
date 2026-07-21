from __future__ import annotations

from collections import Counter

import pytest

from mfh.contracts import Question
from mfh.data.side_effect_sampling import select_mmlu_pro_stratified
from mfh.errors import DataValidationError


def _questions() -> tuple[Question, ...]:
    sizes = {"biology": 30, "law": 20, "physics": 10}
    return tuple(
        Question(
            question_id=f"mmlu_pro:{category}:{index}",
            benchmark="mmlu_pro",
            text=f"{category} question {index}",
            aliases=("A",),
            metadata={"category": category},
        )
        for category, size in sizes.items()
        for index in range(size)
    )


def test_mmlu_pro_stratified_sample_is_proportional_stable_and_receipted() -> None:
    questions = _questions()
    first, receipt = select_mmlu_pro_stratified(
        questions, sample_size=30, seed=17
    )
    second, second_receipt = select_mmlu_pro_stratified(
        questions, sample_size=30, seed=17
    )
    assert first == second
    assert dict(receipt) == dict(second_receipt)
    assert Counter(value.metadata["category"] for value in first) == {
        "biology": 15,
        "law": 10,
        "physics": 5,
    }
    assert receipt["population_size"] == 60
    assert receipt["sample_size"] == 30
    changed, _ = select_mmlu_pro_stratified(
        questions, sample_size=30, seed=18
    )
    assert {value.question_id for value in changed} != {
        value.question_id for value in first
    }


def test_mmlu_pro_stratified_sample_rejects_missing_strata() -> None:
    invalid = (
        Question("mmlu_pro:1", "mmlu_pro", "question", ("A",)),
    )
    with pytest.raises(DataValidationError, match="category"):
        select_mmlu_pro_stratified(invalid, sample_size=1)

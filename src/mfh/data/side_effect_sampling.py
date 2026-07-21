"""Deterministic source-bound sampling for the frozen side-effect suite."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from mfh.contracts import Question
from mfh.errors import DataValidationError
from mfh.provenance import stable_hash

MMLU_PRO_SAMPLE_SIZE = 1_000
MMLU_PRO_SAMPLE_SEED = 17
MMLU_PRO_SAMPLER = "proportional-largest-remainder-sha256-rank-v1"


def select_mmlu_pro_stratified(
    questions: Sequence[Question],
    *,
    sample_size: int = MMLU_PRO_SAMPLE_SIZE,
    seed: int = MMLU_PRO_SAMPLE_SEED,
) -> tuple[tuple[Question, ...], Mapping[str, Any]]:
    """Select a proportional category-stratified cohort without RNG drift."""

    population = tuple(questions)
    if (
        type(sample_size) is not int
        or sample_size <= 0
        or sample_size > len(population)
        or type(seed) is not int
        or seed < 0
        or any(question.benchmark != "mmlu_pro" for question in population)
        or len({question.question_id for question in population}) != len(population)
    ):
        raise DataValidationError("MMLU-Pro stratified sampler inputs are invalid")
    groups: dict[str, list[Question]] = defaultdict(list)
    for question in population:
        category = question.metadata.get("category")
        if type(category) is not str or not category.strip():
            raise DataValidationError("MMLU-Pro row lacks a category stratum")
        groups[category].append(question)
    exact = {
        category: sample_size * len(values) / len(population)
        for category, values in groups.items()
    }
    allocation = {category: math.floor(value) for category, value in exact.items()}
    remainder = sample_size - sum(allocation.values())
    for category in sorted(
        groups,
        key=lambda value: (-(exact[value] - allocation[value]), value),
    )[:remainder]:
        allocation[category] += 1
    if any(allocation[category] <= 0 for category in groups):
        raise DataValidationError("MMLU-Pro sample size cannot represent every category")
    selected_ids: set[str] = set()
    for category, values in groups.items():
        ranked = sorted(
            values,
            key=lambda value: (
                stable_hash(
                    {
                        "sampler": MMLU_PRO_SAMPLER,
                        "seed": seed,
                        "category": category,
                        "question_id": value.question_id,
                    }
                ),
                value.question_id,
            ),
        )
        selected_ids.update(
            value.question_id for value in ranked[: allocation[category]]
        )
    selected = tuple(
        question for question in population if question.question_id in selected_ids
    )
    if len(selected) != sample_size:
        raise DataValidationError("MMLU-Pro stratified sampler returned the wrong size")
    receipt = {
        "algorithm": MMLU_PRO_SAMPLER,
        "seed": seed,
        "strata_field": "category",
        "population_size": len(population),
        "sample_size": sample_size,
        "allocation": dict(sorted(allocation.items())),
        "selected_question_ids_sha256": stable_hash(
            [value.question_id for value in selected]
        ),
    }
    return selected, MappingProxyType(receipt)

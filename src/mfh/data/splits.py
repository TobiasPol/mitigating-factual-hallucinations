"""Entity/alias-aware deterministic TriviaQA research splits."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from types import MappingProxyType

from mfh.contracts import Question
from mfh.data.normalization import normalize_answer, normalize_question
from mfh.errors import DataValidationError
from mfh.provenance import stable_hash


class ResearchSplit(StrEnum):
    T_STEER = "T-steer"
    T_CONTROLLER = "T-controller"
    T_DEV = "T-dev"
    T_TEST = "T-test"
    RESERVED = "reserved"


@dataclass(frozen=True, slots=True)
class SplitPlan:
    steer: int = 30_000
    controller: int = 5_000
    dev: int = 5_000
    test: int = 5_000
    seed: int = 17

    def __post_init__(self) -> None:
        counts = (self.steer, self.controller, self.dev, self.test)
        if any(count < 0 for count in counts) or sum(counts) == 0:
            raise DataValidationError("split counts must be non-negative and not all zero")

    @property
    def targets(self) -> Mapping[ResearchSplit, int]:
        return {
            ResearchSplit.T_STEER: self.steer,
            ResearchSplit.T_CONTROLLER: self.controller,
            ResearchSplit.T_DEV: self.dev,
            ResearchSplit.T_TEST: self.test,
        }


@dataclass(frozen=True, slots=True)
class SplitReport:
    input_count: int
    exact_question_duplicates_removed: int
    assigned_counts: Mapping[str, int]
    underfilled: Mapping[str, int]
    group_count: int


@dataclass(frozen=True, slots=True)
class SplitResult:
    splits: Mapping[ResearchSplit, tuple[Question, ...]]
    report: SplitReport


@dataclass(frozen=True, slots=True)
class ExactDuplicateExclusion:
    """One normalized-question collision group excluded before splitting."""

    normalized_question: str
    answer_alias_component_count: int
    questions: tuple[Question, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "normalized_question": self.normalized_question,
            "answer_alias_component_count": self.answer_alias_component_count,
            "questions": [
                {
                    "question_id": question.question_id,
                    "benchmark": question.benchmark,
                    "text": question.text,
                    "aliases": list(question.aliases),
                    "split": question.split,
                    "entities": list(question.entities),
                    "metadata": dict(question.metadata),
                }
                for question in self.questions
            ],
        }


@dataclass(frozen=True, slots=True)
class ExactDuplicateCurationReport:
    """Audit evidence for the conservative no-adjudication collision policy."""

    input_count: int
    retained_count: int
    excluded_question_count: int
    duplicate_group_count: int
    alias_connected_group_count: int
    alias_disconnected_group_count: int
    exclusions: tuple[ExactDuplicateExclusion, ...]
    policy: str = "exclude-every-normalized-question-collision-group-v1"
    schema_version: int = 1

    def to_dict(self) -> dict[str, object]:
        body: dict[str, object] = {
            "schema_version": self.schema_version,
            "policy": self.policy,
            "input_count": self.input_count,
            "retained_count": self.retained_count,
            "excluded_question_count": self.excluded_question_count,
            "duplicate_group_count": self.duplicate_group_count,
            "alias_connected_group_count": self.alias_connected_group_count,
            "alias_disconnected_group_count": self.alias_disconnected_group_count,
            "excluded_question_ids_sha256": stable_hash(
                sorted(
                    question.question_id
                    for exclusion in self.exclusions
                    for question in exclusion.questions
                )
            ),
            "exclusions": [exclusion.to_dict() for exclusion in self.exclusions],
        }
        return {**body, "report_digest": stable_hash(body)}


@dataclass(frozen=True, slots=True)
class ExactDuplicateCurationResult:
    questions: tuple[Question, ...]
    report: ExactDuplicateCurationReport


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _answer_alias_component_count(questions: tuple[Question, ...]) -> int:
    union = _UnionFind(len(questions))
    aliases = [
        {normalized for alias in question.aliases if (normalized := normalize_answer(alias))}
        for question in questions
    ]
    for left in range(len(questions)):
        for right in range(left + 1, len(questions)):
            if aliases[left] & aliases[right]:
                union.union(left, right)
    return len({union.find(index) for index in range(len(questions))})


def exclude_exact_duplicate_groups(
    questions: Iterable[Question],
) -> ExactDuplicateCurationResult:
    """Exclude every normalized-question collision group without adjudicating answers.

    The retained questions are the original immutable source rows. This deliberately avoids
    selecting a preferred released answer, unioning aliases, or changing source metadata.
    """

    materialized = tuple(questions)
    identifier_counts = Counter(question.question_id for question in materialized)
    duplicate_ids = sorted(
        question_id for question_id, count in identifier_counts.items() if count > 1
    )
    if duplicate_ids:
        raise DataValidationError(f"question IDs must be globally unique: {duplicate_ids[:10]}")
    groups: dict[str, list[Question]] = defaultdict(list)
    for question in materialized:
        groups[normalize_question(question.text)].append(question)

    exclusions = tuple(
        ExactDuplicateExclusion(
            normalized_question=normalized,
            answer_alias_component_count=_answer_alias_component_count(
                tuple(sorted(group, key=lambda item: item.question_id))
            ),
            questions=tuple(sorted(group, key=lambda item: item.question_id)),
        )
        for normalized, group in sorted(groups.items())
        if len(group) > 1
    )
    excluded_ids = {
        question.question_id for exclusion in exclusions for question in exclusion.questions
    }
    retained = tuple(
        question for question in materialized if question.question_id not in excluded_ids
    )
    report = ExactDuplicateCurationReport(
        input_count=len(materialized),
        retained_count=len(retained),
        excluded_question_count=len(excluded_ids),
        duplicate_group_count=len(exclusions),
        alias_connected_group_count=sum(
            exclusion.answer_alias_component_count == 1 for exclusion in exclusions
        ),
        alias_disconnected_group_count=sum(
            exclusion.answer_alias_component_count > 1 for exclusion in exclusions
        ),
        exclusions=exclusions,
    )
    if report.retained_count + report.excluded_question_count != report.input_count:
        raise DataValidationError("duplicate curation counts do not reconcile")
    return ExactDuplicateCurationResult(questions=retained, report=report)


def _deduplicate_exact(questions: Iterable[Question]) -> tuple[list[Question], int]:
    kept: dict[str, Question] = {}
    removed = 0
    for question in sorted(questions, key=lambda item: item.question_id):
        normalized = normalize_question(question.text)
        previous = kept.get(normalized)
        if previous is None:
            kept[normalized] = question
            continue
        previous_answers = {normalize_answer(alias) for alias in previous.aliases}
        current_answers = {normalize_answer(alias) for alias in question.aliases}
        if previous_answers.isdisjoint(current_answers):
            raise DataValidationError(
                "duplicate normalized question has conflicting answers: "
                f"{previous.question_id!r} versus {question.question_id!r}"
            )
        merged_metadata = dict(previous.metadata)
        duplicate_ids = set(merged_metadata.get("deduplicated_question_ids", ()))
        duplicate_ids.add(question.question_id)
        merged_metadata["deduplicated_question_ids"] = sorted(duplicate_ids)
        kept[normalized] = replace(
            previous,
            aliases=tuple(dict.fromkeys((*previous.aliases, *question.aliases))),
            entities=tuple(dict.fromkeys((*previous.entities, *question.entities))),
            metadata=merged_metadata,
        )
        removed += 1
    return list(kept.values()), removed


def _group_questions(questions: list[Question]) -> list[list[Question]]:
    union = _UnionFind(len(questions))
    owners: dict[tuple[str, str], int] = {}
    for index, question in enumerate(questions):
        features = {
            ("question", normalize_question(question.text)),
            *(("entity", normalize_question(entity)) for entity in question.entities),
            *(("alias", normalize_answer(alias)) for alias in question.aliases),
        }
        for feature in features:
            if not feature[1]:
                continue
            owner = owners.setdefault(feature, index)
            union.union(index, owner)
    groups: dict[int, list[Question]] = defaultdict(list)
    for index, question in enumerate(questions):
        groups[union.find(index)].append(question)
    return list(groups.values())


def semantic_group_ids(questions: Iterable[Question]) -> Mapping[str, str]:
    """Return stable entity/alias group IDs using the exact split leakage policy."""

    values = list(questions)
    if len({question.question_id for question in values}) != len(values):
        raise DataValidationError("semantic group IDs require unique question IDs")
    assignments: dict[str, str] = {}
    for group in _group_questions(values):
        identity = stable_hash(sorted(question.question_id for question in group))
        assignments.update({question.question_id: identity for question in group})
    return MappingProxyType(assignments)


def _group_order(group: list[Question], seed: int) -> bytes:
    identity = "\n".join(sorted(question.question_id for question in group))
    return hashlib.sha256(f"{seed}:{identity}".encode()).digest()


def make_research_splits(
    questions: Iterable[Question],
    plan: SplitPlan | None = None,
    *,
    require_exact_sizes: bool = True,
) -> SplitResult:
    plan = plan or SplitPlan()
    materialized = list(questions)
    identifier_counts = Counter(question.question_id for question in materialized)
    duplicate_ids = sorted(
        question_id for question_id, count in identifier_counts.items() if count > 1
    )
    if duplicate_ids:
        raise DataValidationError(f"question IDs must be globally unique: {duplicate_ids[:10]}")
    unique, duplicate_count = _deduplicate_exact(materialized)
    groups = sorted(_group_questions(unique), key=lambda group: _group_order(group, plan.seed))
    assigned: dict[ResearchSplit, list[Question]] = {
        split: [] for split in (*plan.targets, ResearchSplit.RESERVED)
    }

    # Prefer the split with the largest proportional deficit. A whole connected
    # group is assigned together, so no entity or accepted alias leaks across sets.
    for group in groups:
        fitting = [
            split
            for split, target in plan.targets.items()
            if len(assigned[split]) + len(group) <= target
        ]
        if fitting:
            destination = max(
                fitting,
                key=lambda split: (
                    (plan.targets[split] - len(assigned[split])) / max(plan.targets[split], 1)
                ),
            )
        else:
            destination = ResearchSplit.RESERVED
        assigned[destination].extend(
            replace(question, split=destination.value) for question in group
        )

    underfilled = {
        split.value: target - len(assigned[split])
        for split, target in plan.targets.items()
        if len(assigned[split]) < target
    }
    if underfilled and require_exact_sizes:
        raise DataValidationError(
            "entity/alias-disjoint groups cannot satisfy exact split sizes; "
            f"underfilled={underfilled}. Use require_exact_sizes=False and "
            "publish the realized sizes."
        )
    frozen = {split: tuple(values) for split, values in assigned.items()}
    return SplitResult(
        splits=frozen,
        report=SplitReport(
            input_count=len(materialized),
            exact_question_duplicates_removed=duplicate_count,
            assigned_counts={split.value: len(values) for split, values in assigned.items()},
            underfilled=underfilled,
            group_count=len(groups),
        ),
    )


def assert_disjoint(result: SplitResult) -> None:
    seen_questions: dict[str, ResearchSplit] = {}
    seen_entities: dict[str, ResearchSplit] = {}
    seen_aliases: dict[str, ResearchSplit] = {}
    for split, questions in result.splits.items():
        if split is ResearchSplit.RESERVED:
            continue
        for question in questions:
            features = (
                (seen_questions, normalize_question(question.text), "question"),
                *(
                    (seen_entities, normalize_question(value), "entity")
                    for value in question.entities
                ),
                *((seen_aliases, normalize_answer(value), "alias") for value in question.aliases),
            )
            for index, normalized, kind in features:
                previous = index.setdefault(normalized, split)
                if normalized and previous is not split:
                    raise DataValidationError(
                        f"{kind} leakage between {previous.value} and {split.value}: {normalized!r}"
                    )

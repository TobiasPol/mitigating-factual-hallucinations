"""Exact and MinHash-LSH overlap detection for target-benchmark isolation."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from mfh.contracts import Question
from mfh.data.normalization import normalize_question
from mfh.errors import DataValidationError


def character_ngrams(value: str, width: int = 5) -> frozenset[str]:
    normalized = normalize_question(value)
    if not normalized:
        return frozenset()
    if len(normalized) <= width:
        return frozenset({normalized})
    return frozenset(
        normalized[index : index + width] for index in range(len(normalized) - width + 1)
    )


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _minhash(grams: frozenset[str], count: int) -> tuple[int, ...]:
    if not grams:
        return tuple(0 for _ in range(count))
    signature: list[int] = []
    for seed in range(count):
        minimum = min(
            int.from_bytes(
                hashlib.blake2b(f"{seed}:{gram}".encode(), digest_size=8).digest(), "big"
            )
            for gram in grams
        )
        signature.append(minimum)
    return tuple(signature)


@dataclass(frozen=True, slots=True)
class OverlapMatch:
    source_question_id: str
    target_question_id: str
    exact: bool
    ngram_similarity: float
    embedding_similarity: float | None = None


@dataclass(frozen=True, slots=True)
class ContaminationReport:
    source_count: int
    target_count: int
    matches: tuple[OverlapMatch, ...]
    source_ids_to_remove: tuple[str, ...]

    def top(self, limit: int = 100) -> tuple[OverlapMatch, ...]:
        return tuple(
            sorted(
                self.matches,
                key=lambda item: (
                    item.exact,
                    item.embedding_similarity or 0.0,
                    item.ngram_similarity,
                ),
                reverse=True,
            )[:limit]
        )


EmbeddingSimilarity = Callable[[Question, Question], float]


def find_overlaps(
    source_questions: Iterable[Question],
    target_questions: Iterable[Question],
    *,
    ngram_threshold: float = 0.8,
    embedding_threshold: float = 0.9,
    embedding_similarity: EmbeddingSimilarity | None = None,
    num_hashes: int = 64,
    bands: int = 16,
) -> ContaminationReport:
    if not 0 <= ngram_threshold <= 1 or not 0 <= embedding_threshold <= 1:
        raise DataValidationError("similarity thresholds must be in [0, 1]")
    if num_hashes <= 0 or bands <= 0 or num_hashes % bands:
        raise DataValidationError("num_hashes must be positive and divisible by bands")

    sources = list(source_questions)
    targets = list(target_questions)
    source_grams = [character_ngrams(question.text) for question in sources]
    exact_index: dict[str, set[int]] = defaultdict(set)
    lsh_index: dict[tuple[int, tuple[int, ...]], set[int]] = defaultdict(set)
    rows_per_band = num_hashes // bands
    for index, (question, grams) in enumerate(zip(sources, source_grams, strict=True)):
        exact_index[normalize_question(question.text)].add(index)
        signature = _minhash(grams, num_hashes)
        for band in range(bands):
            start = band * rows_per_band
            lsh_index[(band, signature[start : start + rows_per_band])].add(index)

    matches: list[OverlapMatch] = []
    for target in targets:
        normalized = normalize_question(target.text)
        grams = character_ngrams(target.text)
        candidates = set(exact_index.get(normalized, ()))
        signature = _minhash(grams, num_hashes)
        for band in range(bands):
            start = band * rows_per_band
            candidates.update(lsh_index.get((band, signature[start : start + rows_per_band]), ()))
        # A semantic comparator must be able to discover lexically distant
        # paraphrases. The generic callback path is exhaustive by design;
        # large studies should pass a batched/ANN-backed comparator.
        if embedding_similarity is not None:
            candidates.update(range(len(sources)))
        for source_index in candidates:
            source = sources[source_index]
            exact = normalize_question(source.text) == normalized
            lexical = jaccard(source_grams[source_index], grams)
            semantic = (
                float(embedding_similarity(source, target))
                if embedding_similarity is not None
                else None
            )
            if semantic is not None and not 0 <= semantic <= 1:
                raise DataValidationError(
                    f"embedding_similarity must return a value in [0, 1], got {semantic!r}"
                )
            if (
                exact
                or lexical >= ngram_threshold
                or (semantic is not None and semantic >= embedding_threshold)
            ):
                matches.append(
                    OverlapMatch(
                        source_question_id=source.question_id,
                        target_question_id=target.question_id,
                        exact=exact,
                        ngram_similarity=lexical,
                        embedding_similarity=semantic,
                    )
                )
    removed = tuple(sorted({match.source_question_id for match in matches}))
    return ContaminationReport(
        source_count=len(sources),
        target_count=len(targets),
        matches=tuple(matches),
        source_ids_to_remove=removed,
    )

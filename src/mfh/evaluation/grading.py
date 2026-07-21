"""Deterministic grading used only where the benchmark permits it."""

from __future__ import annotations

import re
from collections import Counter

from mfh.contracts import Outcome
from mfh.data.normalization import answer_matches, normalize_answer

_ABSTENTION_PATTERNS = (
    re.compile(r"^(i\s+)?(?:do not|don't|dont)\s+know\.?$", re.IGNORECASE),
    re.compile(r"^(?:unknown|unsure|not sure|cannot answer|can't answer)\.?$", re.IGNORECASE),
    re.compile(r"^i\s+(?:cannot|can't)\s+(?:determine|answer).*$", re.IGNORECASE),
)


def is_explicit_abstention(value: str) -> bool:
    stripped = value.strip()
    return any(pattern.fullmatch(stripped) for pattern in _ABSTENTION_PATTERNS)


def deterministic_short_answer_grade(prediction: str, aliases: tuple[str, ...]) -> Outcome:
    """Alias-aware grade for TriviaQA and smoke tests.

    SimpleQA Verified and AA require their released grader rubrics; this helper
    must not be presented as their official score.
    """

    if not prediction.strip():
        return Outcome.UNSCORABLE
    if is_explicit_abstention(prediction):
        return Outcome.ABSTENTION
    if answer_matches(prediction, aliases):
        return Outcome.CORRECT
    return Outcome.INCORRECT


def token_f1(prediction: str, reference: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def triviaqa_scores(prediction: str, aliases: tuple[str, ...]) -> tuple[float, float]:
    exact_match = float(answer_matches(prediction, aliases))
    best_f1 = max(token_f1(prediction, alias) for alias in aliases)
    return exact_match, best_f1

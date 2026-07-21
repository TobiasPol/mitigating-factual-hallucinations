"""Response-bound secondary SimpleQA metrics.

The released rubric remains the source of truth for the official C/I/A label.
This module adds the preregistered deterministic hedging diagnostic without
claiming to replace that grader.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from typing import Any

from mfh.evaluation.grading import is_explicit_abstention
from mfh.provenance import stable_hash

_HEDGE_PATTERN_TEXT = (
    r"\bi\s+(?:think|believe|suspect|guess)\b",
    r"\b(?:probably|likely|possibly|perhaps|maybe|apparently)\b",
    r"\b(?:may|might|could)\s+be\b",
    r"\bit\s+(?:seems|appears)\b",
    r"\bas\s+far\s+as\s+i\s+know\b",
    r"\bto\s+the\s+best\s+of\s+my\s+knowledge\b",
    r"\bi(?:\s+am|'m)\s+not\s+(?:entirely\s+)?(?:sure|certain)\b",
    r"\b(?:approximately|roughly|circa)\b",
    r"\baround\s+\d",
)
_HEDGE_PATTERNS = tuple(re.compile(value) for value in _HEDGE_PATTERN_TEXT)
_DETECTOR_REVISION = stable_hash(
    {
        "name": "mfh.simpleqa.secondary-hedging",
        "schema_version": 1,
        "normalization": "NFKC-casefold-whitespace",
        "explicit_abstentions_are_hedges": False,
        "patterns": list(_HEDGE_PATTERN_TEXT),
    }
)


def simpleqa_hedge_detector_revision() -> str:
    """Return the immutable detector identity recorded with every response."""

    return _DETECTOR_REVISION


def simpleqa_is_hedged(text: str) -> bool:
    """Detect an explicit uncertainty qualifier in an attempted answer."""

    if is_explicit_abstention(text):
        return False
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return bool(normalized) and any(pattern.search(normalized) for pattern in _HEDGE_PATTERNS)


def simpleqa_hedging_evidence(text: str) -> dict[str, Any]:
    """Build the complete replayable evidence object for one generated response."""

    return {
        "schema_version": 1,
        "detector_revision": _DETECTOR_REVISION,
        "response_sha256": stable_hash(text),
        "hedged": simpleqa_is_hedged(text),
    }


def simpleqa_hedging_evidence_is_valid(text: str, evidence: object) -> bool:
    """Return whether an evidence object exactly replays from ``text``."""

    return isinstance(evidence, Mapping) and dict(evidence) == simpleqa_hedging_evidence(text)

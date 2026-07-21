"""Deterministic benchmark normalization without language-model heuristics."""

from __future__ import annotations

import re
import unicodedata

_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


def _strip_punctuation(value: str) -> str:
    return "".join(
        character for character in value if not unicodedata.category(character).startswith("P")
    )


def normalize_answer(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = _strip_punctuation(value)
    value = _ARTICLES.sub(" ", value)
    return _WHITESPACE.sub(" ", value).strip()


def normalize_question(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = _strip_punctuation(value)
    value = _ARTICLES.sub(" ", value)
    return _WHITESPACE.sub(" ", value).strip()


def answer_matches(prediction: str, aliases: tuple[str, ...]) -> bool:
    normalized = normalize_answer(prediction)
    return bool(normalized) and normalized in {normalize_answer(alias) for alias in aliases}

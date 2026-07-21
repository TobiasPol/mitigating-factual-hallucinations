"""Deterministic output-language and factual checks for the auxiliary suite.

The confirmatory score remains human-audited. This deliberately small detector
is an execution-integrity check: an English response cannot be recorded as a
successful German, Spanish, French, or Japanese response merely by changing a
boolean in record metadata.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Mapping
from typing import Any

from mfh.contracts import Outcome
from mfh.data.normalization import normalize_answer
from mfh.provenance import stable_hash

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_JAPANESE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_MARKERS = {
    "de": frozenset({"antwort", "ist", "und", "nicht", "deutsch", "die", "der", "das"}),
    "en": frozenset({"answer", "is", "and", "not", "english", "the"}),
    "es": frozenset({"respuesta", "es", "y", "no", "espanol", "español", "el", "la"}),
    "fr": frozenset({"reponse", "réponse", "est", "et", "pas", "francais", "français", "le", "la"}),
}
SUPPORTED_LANGUAGES = frozenset({"de", "en", "es", "fr", "ja"})
_ABSTENTIONS = {
    "de": frozenset({"ich weiß es nicht", "ich weiss es nicht", "weiß ich nicht"}),
    "en": frozenset({"i do not know", "i don't know", "unknown", "not sure"}),
    "es": frozenset({"no lo sé", "no lo se", "no sé", "no se"}),
    "fr": frozenset({"je ne sais pas", "inconnu", "incertain"}),
    "ja": frozenset({"わかりません", "分かりません", "知りません", "不明です"}),
}
_GLOBAL_ABSTENTIONS = frozenset({"i do not know", "i don't know"})
_EVALUATOR_REVISION = stable_hash(
    {
        "name": "mfh.language-consistency.alias-script.v1",
        "supported_languages": sorted(SUPPORTED_LANGUAGES),
        "markers": {name: sorted(values) for name, values in sorted(_MARKERS.items())},
        "abstentions": {name: sorted(values) for name, values in sorted(_ABSTENTIONS.items())},
        "global_abstentions": sorted(_GLOBAL_ABSTENTIONS),
        "alias_matching": "NFKC-casefold-token-boundary-or-nonlatin-substring",
        "script_tokenization": "contiguous-unicode-script-runs",
    }
)


def _normalized_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _marker_scores(normalized: str) -> dict[str, int]:
    words = _WORD.findall(normalized)
    return {
        language: sum(word in markers for word in words)
        for language, markers in _MARKERS.items()
    }


def _character_script(character: str) -> str | None:
    if not character.isalpha():
        return None
    name = unicodedata.name(character, "")
    if _JAPANESE.fullmatch(character) is not None:
        return "ja"
    if "LATIN" in name:
        return "latin"
    if "CYRILLIC" in name:
        return "cyrillic"
    if "GREEK" in name:
        return "greek"
    return "other"


def _script_tokens(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    previous: str | None = None
    for character in unicodedata.normalize("NFKC", text):
        script = _character_script(character)
        if script is None:
            previous = None
        elif script != previous:
            tokens.append(script)
            previous = script
    return tuple(tokens)


def detect_output_language(text: str) -> str | None:
    """Return a frozen-suite language only when deterministic evidence is decisive."""

    normalized = _normalized_text(text)
    if not normalized:
        return None
    if normalized.strip(" .!?\u3002\uff01\uff1f") in _GLOBAL_ABSTENTIONS:
        return "en"
    letters = [character for character in normalized if character.isalpha()]
    japanese_count = sum(_JAPANESE.fullmatch(character) is not None for character in letters)
    latin_count = sum("LATIN" in unicodedata.name(character, "") for character in letters)
    if japanese_count:
        if latin_count >= 3 or japanese_count / max(1, japanese_count + latin_count) < 0.8:
            return None
        return "ja" if japanese_count >= 2 else None
    if any("LATIN" not in unicodedata.name(character, "") for character in letters):
        return None
    words = _WORD.findall(normalized)
    scores = _marker_scores(normalized)
    best = max(scores.values(), default=0)
    winners = [language for language, score in scores.items() if score == best]
    if len(winners) != 1 or best < 2 or best / max(1, len(words)) < 0.2:
        return None
    second = max((score for language, score in scores.items() if language != winners[0]), default=0)
    return winners[0] if second <= best / 2 else None


def detect_output_language_alias_aware(
    text: str, aliases: tuple[str, ...]
) -> str | None:
    """Detect response language after masking exact accepted-answer aliases."""

    masked = unicodedata.normalize("NFKC", text).casefold()
    for alias in sorted(aliases, key=len, reverse=True):
        candidate = unicodedata.normalize("NFKC", alias).casefold().strip()
        if not candidate:
            continue
        contains_non_latin = any(
            character.isalpha() and "LATIN" not in unicodedata.name(character, "")
            for character in candidate
        )
        if contains_non_latin:
            masked = masked.replace(candidate, " ")
        else:
            masked = re.sub(
                rf"(?<!\w){re.escape(candidate)}(?!\w)", " ", masked, flags=re.UNICODE
            )
    return detect_output_language(masked)


def requested_language_is_correct(
    text: str,
    requested_language: str,
    aliases: tuple[str, ...] = (),
) -> bool:
    """Score exact requested-language consistency for the controlled suite."""

    if requested_language not in SUPPORTED_LANGUAGES:
        return False
    detected = (
        detect_output_language_alias_aware(text, aliases)
        if aliases
        else detect_output_language(text)
    )
    return detected == requested_language


def non_target_script_token_rate(text: str, requested_language: str) -> float:
    """Return the share of alphabetic script runs outside the requested script."""

    if requested_language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"unsupported requested language: {requested_language!r}")
    tokens = _script_tokens(text)
    if not tokens:
        return 0.0
    target = "ja" if requested_language == "ja" else "latin"
    return sum(value != target for value in tokens) / len(tokens)


def code_switching_detected(text: str, requested_language: str) -> bool:
    """Detect mixed target/non-target language evidence, excluding a lone proper name."""

    if requested_language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"unsupported requested language: {requested_language!r}")
    normalized = _normalized_text(text)
    scripts = Counter(_script_tokens(normalized))
    if requested_language == "ja":
        if scripts["ja"] > 0 and scripts["latin"] >= 2:
            return True
    elif scripts["latin"] > 0 and sum(
        count for script, count in scripts.items() if script != "latin"
    ) >= 2:
        return True
    scores = _marker_scores(normalized)
    target_score = scores.get(requested_language, 0)
    other_score = max(
        (score for language, score in scores.items() if language != requested_language),
        default=0,
    )
    return target_score >= 1 and other_score >= 2


def language_is_abstention(text: str, requested_language: str) -> bool:
    """Recognize localized punts and the composite policy's global default punt."""

    if requested_language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"unsupported requested language: {requested_language!r}")
    normalized = _normalized_text(text).strip(" .!?\u3002\uff01\uff1f")
    return normalized in _ABSTENTIONS[requested_language] | _GLOBAL_ABSTENTIONS


def accepted_alias_in_response(text: str, aliases: tuple[str, ...]) -> bool:
    """Find an accepted short answer inside a response in a Unicode-aware way."""

    normalized = normalize_answer(text)
    if not normalized:
        return False
    for alias in aliases:
        candidate = normalize_answer(alias)
        if not candidate:
            continue
        contains_non_latin = any(
            character.isalpha() and "LATIN" not in unicodedata.name(character, "")
            for character in candidate
        )
        if normalized == candidate or (
            contains_non_latin and len(candidate) >= 2 and candidate in normalized
        ):
            return True
        if len(candidate) >= 2 and re.search(
            rf"(?<!\w){re.escape(candidate)}(?!\w)", normalized, re.UNICODE
        ):
            return True
    return False


def language_factual_outcome(
    text: str, requested_language: str, aliases: tuple[str, ...]
) -> Outcome:
    """Apply alias-aware factual grading independently of language consistency."""

    if not text.strip():
        return Outcome.UNSCORABLE
    if language_is_abstention(text, requested_language):
        return Outcome.ABSTENTION
    return (
        Outcome.CORRECT
        if accepted_alias_in_response(text, aliases)
        else Outcome.INCORRECT
    )


def language_evaluator_revision() -> str:
    """Return the immutable evaluator identity recorded with every suite row."""

    return _EVALUATOR_REVISION


def language_response_evidence(
    text: str, requested_language: str, aliases: tuple[str, ...]
) -> dict[str, Any]:
    """Build all preregistered automated language metrics for one response."""

    outcome = language_factual_outcome(text, requested_language, aliases)
    return {
        "schema_version": 1,
        "evaluator_revision": _EVALUATOR_REVISION,
        "response_sha256": stable_hash(text),
        "requested_language": requested_language,
        "detected_language": detect_output_language_alias_aware(text, aliases),
        "requested_language_correct": requested_language_is_correct(
            text, requested_language, aliases
        ),
        "non_target_script_token_rate": non_target_script_token_rate(
            text, requested_language
        ),
        "code_switching": code_switching_detected(text, requested_language),
        "factual_outcome": outcome.value,
        "factual_correct": outcome is Outcome.CORRECT,
        "abstained": outcome is Outcome.ABSTENTION,
        "accepted_aliases": list(aliases),
        "accepted_aliases_digest": stable_hash(list(aliases)),
    }


def language_response_evidence_is_valid(text: str, evidence: object) -> bool:
    """Return whether stored language evidence exactly replays from its response."""

    if not isinstance(evidence, Mapping):
        return False
    requested = evidence.get("requested_language")
    aliases = evidence.get("accepted_aliases")
    if (
        requested not in SUPPORTED_LANGUAGES
        or not isinstance(aliases, list)
        or not aliases
        or any(not isinstance(value, str) or not value for value in aliases)
    ):
        return False
    return dict(evidence) == language_response_evidence(
        text, str(requested), tuple(aliases)
    )

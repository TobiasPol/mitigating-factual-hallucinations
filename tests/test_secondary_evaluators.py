from __future__ import annotations

from mfh.contracts import Outcome
from mfh.evaluation.language import (
    accepted_alias_in_response,
    code_switching_detected,
    language_factual_outcome,
    language_response_evidence,
    language_response_evidence_is_valid,
    non_target_script_token_rate,
)
from mfh.evaluation.simpleqa import (
    simpleqa_hedging_evidence,
    simpleqa_hedging_evidence_is_valid,
    simpleqa_is_hedged,
)


def test_simpleqa_hedging_is_distinct_from_punting_and_response_bound() -> None:
    hedged = "I think the answer is Paris."
    assert simpleqa_is_hedged(hedged)
    assert not simpleqa_is_hedged("I don't know.")
    assert not simpleqa_is_hedged("The answer is Paris.")
    evidence = simpleqa_hedging_evidence(hedged)
    assert simpleqa_hedging_evidence_is_valid(hedged, evidence)
    assert not simpleqa_hedging_evidence_is_valid("The answer is Paris.", evidence)


def test_language_metrics_are_alias_aware_and_keep_script_metrics_separate() -> None:
    text = "Die Antwort ist 東京."
    aliases = ("東京", "Tokio")
    evidence = language_response_evidence(text, "de", aliases)
    assert accepted_alias_in_response(text, aliases)
    assert language_factual_outcome(text, "de", aliases) is Outcome.CORRECT
    assert evidence["requested_language_correct"] is True
    assert float(evidence["non_target_script_token_rate"]) > 0
    assert evidence["code_switching"] is False
    assert language_response_evidence_is_valid(text, evidence)
    assert not language_response_evidence_is_valid(text + " geändert", evidence)


def test_language_code_switching_requires_mixed_evidence_not_one_name() -> None:
    assert not code_switching_detected("答えは Paris です。", "ja")
    assert code_switching_detected("答えは I think Paris is correct です。", "ja")
    assert non_target_script_token_rate("答えは Paris です。", "ja") > 0


def test_language_abstention_is_recognized_in_requested_language() -> None:
    aliases = ("Paris",)
    evidence = language_response_evidence("Je ne sais pas.", "fr", aliases)
    assert evidence["factual_outcome"] == "A"
    assert evidence["abstained"] is True
    assert evidence["factual_correct"] is False


def test_composite_global_abstention_is_not_a_factual_error_in_any_language() -> None:
    for language in ("de", "en", "es", "fr", "ja"):
        evidence = language_response_evidence("I don't know.", language, ("Paris",))
        assert evidence["factual_outcome"] == "A"
        assert evidence["abstained"] is True
        assert evidence["detected_language"] == "en"
        assert evidence["requested_language_correct"] is (language == "en")

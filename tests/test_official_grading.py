from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from mfh.contracts import Outcome
from mfh.errors import ConfigurationError
from mfh.evaluation.official import (
    GradingRequest,
    aa_official_metrics,
    load_official_grader_spec,
    render_grader_prompt,
    run_official_grader,
    simpleqa_official_metrics,
)

ROOT = Path(__file__).parents[1]
GRADERS = ROOT / "configs" / "graders"


def test_released_grader_specs_are_frozen_and_complete() -> None:
    simpleqa = load_official_grader_spec(GRADERS / "simpleqa-verified.yaml")
    aa = load_official_grader_spec(GRADERS / "aa-omniscience-public.yaml")

    assert simpleqa.grader_model_revision == "gpt-4.1-2025-04-14"
    assert len(simpleqa.prompt_template.split()) == 1_124
    assert len(simpleqa.prompt_template.encode()) == 7_125
    assert simpleqa.prompt_sha256 == (
        "84c004ec4fcf8f0703bb0d734544036a72e847bfa7116429e5aee5e53ccc8cf3"
    )
    assert simpleqa.prompt_sha256 == hashlib.sha256(simpleqa.prompt_template.encode()).hexdigest()
    assert "scriptVersionId=290594993" in simpleqa.source_repository
    assert simpleqa.source_artifact_sha256 == (
        "14d0c0513efefdfe7936e05c6fc09b4b4a191cc31273ca8bfbcdeaea0c6fdb1b"
    )
    assert simpleqa.label_mapping == {
        "A": Outcome.CORRECT,
        "B": Outcome.INCORRECT,
        "C": Outcome.ABSTENTION,
    }
    assert aa.reasoning_enabled is True
    assert aa.grader_model == "google/gemini-2.5-flash"
    assert aa.grader_model_revision == "gemini-2.5-flash"
    assert aa.label_mapping["C"] is Outcome.PARTIAL
    assert aa.prompt_sha256 == "7d2f6d60367d8f3ca7c92b115c65fa536e321e27ce93e0779d77d5c3700901c9"
    assert aa.prompt_sha256 == hashlib.sha256(aa.prompt_template.encode()).hexdigest()
    assert 'consider the question "What city is OpenAI headquartered in?"' in aa.prompt_template
    assert len({simpleqa.digest, aa.digest}) == 2


def test_prompt_rendering_does_not_reinterpret_inserted_placeholders() -> None:
    grader = load_official_grader_spec(GRADERS / "simpleqa-verified.yaml")
    request = GradingRequest(
        question_id="q1",
        question="What literal token is shown?",
        target="{predicted_answer}",
        predicted_answer="{target}",
    )

    rendered = render_grader_prompt(grader, request)

    assert "Gold target: {predicted_answer}" in rendered
    assert "Predicted answer: {target}" in rendered


def test_official_grader_retries_exact_parse_then_returns_success() -> None:
    grader = load_official_grader_spec(GRADERS / "simpleqa-verified.yaml")
    request = GradingRequest("q1", "Capital of France?", "Paris", "Paris")
    responses = iter(["The grade is A", " A\n"])

    record = run_official_grader(grader, request, lambda _prompt, _spec: next(responses))

    assert record.outcome is Outcome.CORRECT
    assert record.attempts == 2
    assert record.error is None
    assert record.request_fingerprint == request.digest
    assert record.grader_fingerprint == grader.digest


def test_official_grader_failure_is_unscorable_not_silently_incorrect() -> None:
    grader = load_official_grader_spec(GRADERS / "aa-omniscience-public.yaml")
    request = GradingRequest("q1", "Question", "Target", "Answer")

    record = run_official_grader(
        grader,
        request,
        lambda _prompt, _spec: (_ for _ in ()).throw(RuntimeError("provider down")),
    )

    assert record.outcome is Outcome.UNSCORABLE
    assert record.attempts == grader.maximum_attempts
    assert record.error == "RuntimeError: provider down"


def test_released_benchmark_metric_equations_remain_distinct() -> None:
    aa = aa_official_metrics(
        (
            Outcome.CORRECT,
            Outcome.CORRECT,
            Outcome.PARTIAL,
            Outcome.INCORRECT,
            Outcome.ABSTENTION,
        )
    )
    simpleqa = simpleqa_official_metrics(
        (Outcome.CORRECT, Outcome.CORRECT, Outcome.INCORRECT, Outcome.ABSTENTION)
    )

    assert aa.omniscience_index == 20.0
    assert aa.accuracy == 0.4
    assert aa.hallucination_rate == pytest.approx(1 / 3)
    assert simpleqa.accuracy == 0.5
    assert simpleqa.accuracy_given_attempted == pytest.approx(2 / 3)
    assert simpleqa.simpleqa_f1 == pytest.approx(4 / 7)


def test_official_grader_loader_rejects_prompt_hash_drift(tmp_path: Path) -> None:
    prompt = "Question: {question}\nGold: {target}\nAnswer: {predicted_answer}\n"
    (tmp_path / "prompt.txt").write_text(prompt, encoding="utf-8")
    config = (GRADERS / "simpleqa-verified.yaml").read_text(encoding="utf-8")
    config = config.replace("simpleqa-verified.prompt.txt", "prompt.txt").replace(
        "84c004ec4fcf8f0703bb0d734544036a72e847bfa7116429e5aee5e53ccc8cf3",
        '"' + "0" * 64 + '"',
    )
    path = tmp_path / "grader.yaml"
    path.write_text(config, encoding="utf-8")

    with pytest.raises(ConfigurationError, match="prompt hash differs"):
        load_official_grader_spec(path)


def test_released_grader_identity_rejects_semantically_swapped_labels(tmp_path: Path) -> None:
    prompt_name = "aa-omniscience-public.prompt.txt"
    (tmp_path / prompt_name).write_text(
        (GRADERS / prompt_name).read_text(encoding="utf-8"), encoding="utf-8"
    )
    config = (GRADERS / "aa-omniscience-public.yaml").read_text(encoding="utf-8")
    config = config.replace("    A: C\n    B: I\n", "    A: I\n    B: C\n")
    path = tmp_path / "grader.yaml"
    path.write_text(config, encoding="utf-8")

    with pytest.raises(ConfigurationError, match="released grader identity differs"):
        load_official_grader_spec(path)

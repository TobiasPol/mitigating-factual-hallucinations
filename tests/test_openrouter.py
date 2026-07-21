from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any
from urllib.request import Request

import pytest

from mfh.contracts import Outcome
from mfh.errors import DataValidationError
from mfh.evaluation.official import (
    GradingRequest,
    PermanentGraderError,
    load_official_grader_spec,
)
from mfh.evaluation.openrouter import (
    OpenRouterError,
    OpenRouterTransport,
    route_for_grader,
    run_openrouter_grader,
    validate_openrouter_attempt_receipt,
    verify_openrouter_catalog,
)

ROOT = Path(__file__).parents[1]
GRADERS = ROOT / "configs/graders"


def _response(*, model: str, provider: str, content: str = "A") -> bytes:
    return json.dumps(
        {
            "id": "gen-test",
            "model": model,
            "provider": provider,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        }
    ).encode()


def test_routes_and_catalog_match_frozen_grader_amendment() -> None:
    simpleqa = load_official_grader_spec(GRADERS / "simpleqa-verified.yaml")
    aa = load_official_grader_spec(GRADERS / "aa-omniscience-public.yaml")
    assert route_for_grader(simpleqa).canonical_slug == "openai/gpt-4.1-2025-04-14"
    assert route_for_grader(aa).request_model == "google/gemini-2.5-flash"
    catalog: dict[str, Any] = {
        "data": [
            {"id": "openai/gpt-4.1", "canonical_slug": "openai/gpt-4.1-2025-04-14"},
            {
                "id": "google/gemini-2.5-flash",
                "canonical_slug": "google/gemini-2.5-flash",
            },
        ]
    }
    assert set(verify_openrouter_catalog(catalog, (simpleqa, aa))) == {
        "simpleqa_verified",
        "aa_omniscience_public_600",
    }
    catalog["data"][0]["canonical_slug"] = "openai/gpt-4.1"
    with pytest.raises(DataValidationError, match="lacks exact grader"):
        verify_openrouter_catalog(catalog, (simpleqa, aa))


def test_exact_request_and_success_receipt_do_not_contain_key() -> None:
    spec = load_official_grader_spec(GRADERS / "simpleqa-verified.yaml")
    captured: dict[str, Any] = {}

    def sender(request: Request, _timeout: float) -> tuple[int, bytes]:
        assert isinstance(request.data, bytes)
        captured["payload"] = json.loads(request.data)
        captured["authorization"] = request.headers["Authorization"]
        return 200, _response(model="openai/gpt-4.1", provider="OpenAI")

    transport = OpenRouterTransport(api_key="secret-test-key", sender=sender)
    grade = run_openrouter_grader(
        spec,
        GradingRequest("q1", "Capital of France?", "Paris", "Paris"),
        transport,
        sleeper=lambda _seconds: None,
    )
    assert grade.outcome is Outcome.CORRECT
    assert captured["payload"] == {
        "messages": [
            {
                "content": captured["payload"]["messages"][0]["content"],
                "role": "user",
            }
        ],
        "model": "openai/gpt-4.1",
        "provider": {
            "allow_fallbacks": False,
            "only": ["openai"],
            "require_parameters": True,
        },
        "stream": False,
        "temperature": 0.0,
    }
    assert captured["authorization"] == "Bearer secret-test-key"
    receipt = transport.receipts[0].to_dict()
    assert "secret-test-key" not in json.dumps(receipt)
    assert receipt["returned_provider"] == "OpenAI"
    assert receipt["response_id"] == "gen-test"


def test_aa_request_enables_reasoning_and_excludes_it_from_label() -> None:
    spec = load_official_grader_spec(GRADERS / "aa-omniscience-public.yaml")
    transport = OpenRouterTransport(
        api_key="key",
        sender=lambda _request, _timeout: (
            200,
            _response(model="google/gemini-2.5-flash", provider="Google AI Studio"),
        ),
    )
    payload = transport.request_payload("grade", spec)
    assert payload["reasoning"] == {"enabled": True, "exclude": True}
    assert payload["provider"]["only"] == ["google-ai-studio"]


def test_transient_error_retries_but_provider_substitution_fails_closed() -> None:
    spec = load_official_grader_spec(GRADERS / "simpleqa-verified.yaml")
    calls = 0

    def transient_then_success(_request: Request, _timeout: float) -> tuple[int, bytes]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OpenRouterError("rate limited", transient=True, retry_after=0)
        return 200, _response(model="openai/gpt-4.1", provider="OpenAI")

    transport = OpenRouterTransport(api_key="key", sender=transient_then_success)
    request = GradingRequest("q1", "Question", "Target", "Answer")
    grade = run_openrouter_grader(spec, request, transport, sleeper=lambda _seconds: None)
    assert grade.outcome is Outcome.CORRECT
    assert grade.attempts == 2

    substituted = OpenRouterTransport(
        api_key="key",
        sender=lambda _request, _timeout: (
            200,
            _response(model="openai/gpt-4.1", provider="Azure"),
        ),
    )
    failed = run_openrouter_grader(spec, request, substituted, sleeper=lambda _seconds: None)
    assert failed.outcome is Outcome.UNSCORABLE
    assert failed.attempts == 1
    assert "substituted the grader provider" in str(failed.error)
    failed_receipt = substituted.receipts[0]
    assert failed_receipt.http_status == 200
    assert failed_receipt.response_sha256 is not None
    assert failed_receipt.returned_model == "openai/gpt-4.1"
    assert failed_receipt.returned_provider == "Azure"


def test_non_200_success_shaped_response_is_never_accepted() -> None:
    spec = load_official_grader_spec(GRADERS / "simpleqa-verified.yaml")
    transport = OpenRouterTransport(
        api_key="key",
        sender=lambda _request, _timeout: (
            429,
            _response(model="openai/gpt-4.1", provider="OpenAI"),
        ),
    )
    grade = run_openrouter_grader(
        spec,
        GradingRequest("q1", "Question", "Target", "Answer"),
        transport,
        sleeper=lambda _seconds: None,
    )
    assert grade.outcome is Outcome.UNSCORABLE
    assert grade.attempts == 3
    assert [receipt.http_status for receipt in transport.receipts] == [429, 429, 429]
    assert all(receipt.transient for receipt in transport.receipts)


def test_reused_transport_attempt_count_is_grade_local() -> None:
    spec = load_official_grader_spec(GRADERS / "simpleqa-verified.yaml")
    valid = True

    def sender(_request: Request, _timeout: float) -> tuple[int, bytes]:
        return 200, _response(
            model="openai/gpt-4.1",
            provider="OpenAI",
            content="A" if valid else "UNKNOWN",
        )

    transport = OpenRouterTransport(api_key="key", sender=sender)
    request = GradingRequest("q1", "Question", "Target", "Answer")
    assert run_openrouter_grader(spec, request, transport).attempts == 1
    valid = False
    failed = run_openrouter_grader(spec, request, transport, sleeper=lambda _seconds: None)
    assert failed.outcome is Outcome.UNSCORABLE
    assert failed.attempts == spec.maximum_attempts
    assert len(transport.receipts) == 1 + spec.maximum_attempts


def test_error_receipts_and_grade_errors_redact_api_key() -> None:
    spec = load_official_grader_spec(GRADERS / "simpleqa-verified.yaml")
    key = "secret-test-key"
    transport = OpenRouterTransport(
        api_key=key,
        sender=lambda _request, _timeout: (_ for _ in ()).throw(
            OpenRouterError(f"provider echoed {key}", transient=False)
        ),
    )
    grade = run_openrouter_grader(
        spec,
        GradingRequest("q1", "Question", "Target", "Answer"),
        transport,
    )
    assert grade.outcome is Outcome.UNSCORABLE
    assert key not in json.dumps(transport.receipts[0].to_dict())
    assert key not in str(grade.error)
    assert str(grade.error).endswith(
        "OpenRouter request failed before receiving a response"
    )


def test_provider_controlled_metadata_and_exception_state_cannot_retain_key() -> None:
    spec = load_official_grader_spec(GRADERS / "simpleqa-verified.yaml")
    key = "secret-test-key"
    malicious = json.dumps(
        {
            "id": f"generation-{key}",
            "model": "openai/gpt-4.1",
            "provider": "OpenAI",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "A"},
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 1,
                "total_tokens": 11,
                "provider_note": key,
            },
        }
    ).encode()
    transport = OpenRouterTransport(
        api_key=key,
        sender=lambda _request, _timeout: (200, malicious),
    )
    grade = run_openrouter_grader(
        spec,
        GradingRequest("q1", "Question", "Target", "Answer"),
        transport,
    )
    assert grade.outcome is Outcome.CORRECT
    serialized = json.dumps(transport.receipts[0].to_dict())
    assert key not in serialized
    archived = base64.b64decode(transport.receipts[0].response_body_base64 or "")
    assert key.encode() not in archived
    assert transport.receipts[0].response_id == "generation-[REDACTED]"
    assert set(transport.receipts[0].usage) == {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    }

    error_body = json.dumps({"error": {"message": f"Bearer {key}"}}).encode()
    failing = OpenRouterTransport(
        api_key=key,
        sender=lambda _request, _timeout: (_ for _ in ()).throw(
            OpenRouterError(
                f"Bearer {key}",
                transient=False,
                http_status=400,
                response_body=error_body,
            )
        ),
    )
    with pytest.raises(OpenRouterError) as captured:
        failing.invoke("prompt", spec, attempt=1)
    assert key not in str(captured.value)
    assert captured.value.response_body is None
    assert captured.value.__cause__ is None
    assert key not in json.dumps(failing.receipts[0].to_dict())


def test_receipt_replay_binds_response_id_usage_and_retry_eligibility() -> None:
    spec = load_official_grader_spec(GRADERS / "simpleqa-verified.yaml")
    prompt = "frozen prompt"
    transport = OpenRouterTransport(
        api_key="key",
        sender=lambda _request, _timeout: (
            200,
            _response(model="openai/gpt-4.1", provider="OpenAI"),
        ),
    )
    transport.invoke(prompt, spec, attempt=1)
    receipt = transport.receipts[0].to_dict()
    route = route_for_grader(spec)
    request_sha = receipt["request_sha256"]
    prompt_sha = receipt["prompt_sha256"]
    validate_openrouter_attempt_receipt(
        receipt,
        route=route,
        request_sha256=request_sha,
        prompt_sha256=prompt_sha,
        attempt=1,
        expect_success=True,
        expected_content="A",
    )
    for field, forged in (
        ("response_id", "forged"),
        ("usage", {"prompt_tokens": 999}),
    ):
        changed = {**receipt, field: forged}
        with pytest.raises(DataValidationError, match="response archive"):
            validate_openrouter_attempt_receipt(
                changed,
                route=route,
                request_sha256=request_sha,
                prompt_sha256=prompt_sha,
                attempt=1,
                expect_success=True,
                expected_content="A",
            )
    forged_failure = {
        **receipt,
        "error_type": "OpenRouterError",
        "error_message": "arbitrary retry",
        "transient": True,
    }
    with pytest.raises(DataValidationError, match="replayable transient failure"):
        validate_openrouter_attempt_receipt(
            forged_failure,
            route=route,
            request_sha256=request_sha,
            prompt_sha256=prompt_sha,
            attempt=1,
            expect_success=False,
            accepted_success_content=lambda content: content == "A",
        )


def test_failed_receipt_error_message_is_derived_from_archive() -> None:
    spec = load_official_grader_spec(GRADERS / "simpleqa-verified.yaml")
    error_body = json.dumps({"error": {"message": "quota exhausted"}}).encode()
    transport = OpenRouterTransport(
        api_key="key",
        sender=lambda _request, _timeout: (429, error_body),
    )
    with pytest.raises(OpenRouterError):
        transport.invoke("frozen prompt", spec, attempt=1)
    receipt = transport.receipts[0].to_dict()
    route = route_for_grader(spec)
    validate_openrouter_attempt_receipt(
        receipt,
        route=route,
        request_sha256=str(receipt["request_sha256"]),
        prompt_sha256=str(receipt["prompt_sha256"]),
        attempt=1,
        expect_success=False,
        accepted_success_content=lambda content: content.strip() in spec.label_mapping,
    )
    forged = {**receipt, "error_message": "unrelated fabricated explanation"}
    with pytest.raises(DataValidationError, match="replayable transient failure"):
        validate_openrouter_attempt_receipt(
            forged,
            route=route,
            request_sha256=str(receipt["request_sha256"]),
            prompt_sha256=str(receipt["prompt_sha256"]),
            attempt=1,
            expect_success=False,
            accepted_success_content=lambda content: content.strip()
            in spec.label_mapping,
        )


def test_missing_api_key_fails_before_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(PermanentGraderError, match="missing or blank"):
        OpenRouterTransport()

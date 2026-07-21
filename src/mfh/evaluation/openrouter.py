"""Fail-closed OpenRouter transport for frozen model-based benchmark graders."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from mfh.contracts import Outcome
from mfh.errors import DataValidationError
from mfh.evaluation.official import (
    GradingRequest,
    OfficialGradeRecord,
    OfficialGraderSpec,
    PermanentGraderError,
    render_grader_prompt,
)
from mfh.provenance import canonical_json, sha256_file

_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
_CATALOG_ENDPOINT = "https://openrouter.ai/api/v1/models"
_TRANSIENT_HTTP = frozenset({408, 429, 500, 502, 503, 504})
_BEARER_SECRET = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+")
_USAGE_FIELDS = frozenset({"prompt_tokens", "completion_tokens", "total_tokens"})
_PRE_RESPONSE_FAILURE = "OpenRouter request failed before receiving a response"
_CONTENT_VALIDATION_FAILURE = "OpenRouter response content failed consumer validation"
OPENROUTER_ATTEMPT_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "endpoint",
        "attempt",
        "request_sha256",
        "prompt_sha256",
        "requested_model",
        "canonical_slug",
        "required_provider_slug",
        "http_status",
        "response_sha256",
        "response_body_base64",
        "response_id",
        "returned_model",
        "returned_provider",
        "finish_reason",
        "content_sha256",
        "usage",
        "latency_seconds",
        "error_type",
        "error_message",
        "transient",
    }
)


@dataclass(frozen=True, slots=True)
class OpenRouterRoute:
    grader_model: str
    grader_revision: str
    request_model: str
    canonical_slug: str
    provider_slug: str
    provider_name: str
    reasoning_enabled: bool


_ROUTES = MappingProxyType(
    {
        ("openai/gpt-4.1", "gpt-4.1-2025-04-14"): OpenRouterRoute(
            grader_model="openai/gpt-4.1",
            grader_revision="gpt-4.1-2025-04-14",
            request_model="openai/gpt-4.1",
            canonical_slug="openai/gpt-4.1-2025-04-14",
            provider_slug="openai",
            provider_name="OpenAI",
            reasoning_enabled=False,
        ),
        ("google/gemini-2.5-flash", "gemini-2.5-flash"): OpenRouterRoute(
            grader_model="google/gemini-2.5-flash",
            grader_revision="gemini-2.5-flash",
            request_model="google/gemini-2.5-flash",
            canonical_slug="google/gemini-2.5-flash",
            provider_slug="google-ai-studio",
            provider_name="Google AI Studio",
            reasoning_enabled=True,
        ),
    }
)


def route_for_grader(spec: OfficialGraderSpec) -> OpenRouterRoute:
    try:
        route = _ROUTES[(spec.grader_model, spec.grader_model_revision)]
    except KeyError as exc:
        raise DataValidationError(
            "official grader has no frozen OpenRouter route: "
            f"{spec.grader_model}@{spec.grader_model_revision}"
        ) from exc
    if route.reasoning_enabled != spec.reasoning_enabled:
        raise DataValidationError("OpenRouter route reasoning differs from grader specification")
    return route


@dataclass(frozen=True, slots=True)
class OpenRouterAttemptReceipt:
    schema_version: int
    endpoint: str
    attempt: int
    request_sha256: str
    prompt_sha256: str
    requested_model: str
    canonical_slug: str
    required_provider_slug: str
    http_status: int | None
    response_sha256: str | None
    response_body_base64: str | None
    response_id: str | None
    returned_model: str | None
    returned_provider: str | None
    finish_reason: str | None
    content_sha256: str | None
    usage: Mapping[str, Any]
    latency_seconds: float
    error_type: str | None
    error_message: str | None
    transient: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OpenRouterError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        transient: bool,
        retry_after: float | None = None,
        http_status: int | None = None,
        response_body: bytes | None = None,
    ):
        super().__init__(message)
        self.transient = transient
        self.retry_after = retry_after
        self.http_status = http_status
        self.response_body = response_body


Sender = Callable[[urllib.request.Request, float], tuple[int, bytes]]


def _default_sender(request: urllib.request.Request, timeout: float) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read()
        retry_after: float | None = None
        raw_retry = exc.headers.get("Retry-After")
        if raw_retry is not None:
            try:
                retry_after = max(0.0, min(float(raw_retry), 60.0))
            except ValueError:
                retry_after = None
        message = _error_message(body) or f"OpenRouter HTTP {exc.code}"
        raise OpenRouterError(
            f"OpenRouter HTTP {exc.code}: {message}",
            transient=exc.code in _TRANSIENT_HTTP,
            retry_after=retry_after,
            http_status=exc.code,
            response_body=body,
        ) from exc
    except (TimeoutError, urllib.error.URLError) as exc:
        raise OpenRouterError(
            f"OpenRouter network failure: {type(exc).__name__}", transient=True
        ) from exc


def _error_message(body: bytes) -> str | None:
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    error = value.get("error") if isinstance(value, Mapping) else None
    message = error.get("message") if isinstance(error, Mapping) else None
    return str(message)[:500] if isinstance(message, str) else None


def _safe_text(value: object, secret: str) -> str:
    redacted = str(value).replace(secret, "[REDACTED]")
    return _BEARER_SECRET.sub(r"\1[REDACTED]", redacted)[:500]


def _archive_response_body(body: bytes | None, *, secret: str) -> bytes | None:
    """Return a replayable response archive with credential material removed."""

    if body is None:
        return None
    secret_bytes = secret.encode("utf-8")
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return body.replace(secret_bytes, b"[REDACTED]")
    redacted = text.replace(secret, "[REDACTED]")
    return _BEARER_SECRET.sub(r"\1[REDACTED]", redacted).encode("utf-8")


def _safe_usage(value: object) -> dict[str, int | float]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, int | float] = {}
    for key in _USAGE_FIELDS:
        item = value.get(key)
        if (
            isinstance(item, bool)
            or not isinstance(item, int | float)
            or not math.isfinite(float(item))
            or item < 0
        ):
            continue
        result[key] = item
    return result


def _response_metadata(body: bytes | None, *, secret: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "response_sha256": hashlib.sha256(body).hexdigest() if body is not None else None,
        "response_id": None,
        "returned_model": None,
        "returned_provider": None,
        "finish_reason": None,
        "content_sha256": None,
        "usage": {},
    }
    if body is None:
        return metadata
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return metadata
    if not isinstance(value, Mapping):
        return metadata
    if value.get("id") is not None:
        metadata["response_id"] = _safe_text(value["id"], secret)
    if value.get("model") is not None:
        metadata["returned_model"] = _safe_text(value["model"], secret)
    if value.get("provider") is not None:
        metadata["returned_provider"] = _safe_text(value["provider"], secret)
    choices = value.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else None
    if isinstance(choice, Mapping):
        if choice.get("finish_reason") is not None:
            metadata["finish_reason"] = str(choice["finish_reason"])
        message = choice.get("message")
        content = message.get("content") if isinstance(message, Mapping) else None
        if isinstance(content, str):
            metadata["content_sha256"] = hashlib.sha256(content.encode("utf-8")).hexdigest()
    metadata["usage"] = _safe_usage(value.get("usage"))
    return metadata


def _replay_failure_message(
    body: bytes | None,
    *,
    status: int | None,
    route: OpenRouterRoute,
    accepted_success_content: Callable[[str], bool] | None,
) -> str | None:
    """Derive the only admissible failure message from archived response evidence."""

    if body is None and status is None:
        return _PRE_RESPONSE_FAILURE
    if status != 200:
        detail = _error_message(body or b"") or "unexpected response status"
        return f"OpenRouter HTTP {status}: {detail}"
    try:
        value = json.loads(body or b"")
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "OpenRouter returned invalid JSON"
    if not isinstance(value, Mapping) or value.get("error") is not None:
        return _error_message(body or b"") or "OpenRouter returned an error payload"
    choices = value.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return "OpenRouter response must contain one choice"
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, Mapping) else None
    content = message.get("content") if isinstance(message, Mapping) else None
    finish_reason = choice.get("finish_reason") if isinstance(choice, Mapping) else None
    if not isinstance(content, str) or not content.strip():
        return "OpenRouter returned empty grader content"
    if finish_reason != "stop":
        return "OpenRouter grader did not finish with stop"
    if value.get("model") not in {route.request_model, route.canonical_slug}:
        return "OpenRouter substituted the grader model"
    if value.get("provider") != route.provider_name:
        return "OpenRouter substituted the grader provider"
    if accepted_success_content is not None and not accepted_success_content(content):
        return _CONTENT_VALIDATION_FAILURE
    return None


def validate_openrouter_attempt_receipt(
    receipt: Mapping[str, Any],
    *,
    route: OpenRouterRoute,
    request_sha256: str,
    prompt_sha256: str,
    attempt: int,
    expect_success: bool,
    expected_content: str | None = None,
    accepted_success_content: Callable[[str], bool] | None = None,
    expect_retry: bool = True,
) -> bytes | None:
    """Replay every receipt field from its credential-redacted response archive."""

    if (
        set(receipt) != OPENROUTER_ATTEMPT_RECEIPT_FIELDS
        or receipt.get("schema_version") != 1
        or receipt.get("endpoint") != _ENDPOINT
        or receipt.get("attempt") != attempt
        or receipt.get("request_sha256") != request_sha256
        or receipt.get("prompt_sha256") != prompt_sha256
        or receipt.get("requested_model") != route.request_model
        or receipt.get("canonical_slug") != route.canonical_slug
        or receipt.get("required_provider_slug") != route.provider_slug
    ):
        raise DataValidationError("OpenRouter receipt route or request identity differs")
    latency = receipt.get("latency_seconds")
    status = receipt.get("http_status")
    if (
        isinstance(latency, bool)
        or not isinstance(latency, int | float)
        or not math.isfinite(float(latency))
        or float(latency) < 0
        or (
            status is not None
            and (type(status) is not int or not 100 <= status <= 599)
        )
    ):
        raise DataValidationError("OpenRouter receipt timing or status is invalid")
    encoded = receipt.get("response_body_base64")
    try:
        archived = (
            base64.b64decode(encoded, validate=True)
            if isinstance(encoded, str)
            else None
        )
    except (ValueError, TypeError) as exc:
        raise DataValidationError("OpenRouter response archive is not canonical base64") from exc
    if isinstance(encoded, str) and base64.b64encode(archived or b"").decode("ascii") != encoded:
        raise DataValidationError("OpenRouter response archive base64 is non-canonical")
    metadata = _response_metadata(archived, secret="credential-not-present-in-archive")
    if (
        (encoded is None) is not (receipt.get("response_sha256") is None)
        or receipt.get("response_sha256") != metadata["response_sha256"]
        or receipt.get("response_id") != metadata["response_id"]
        or receipt.get("returned_model") != metadata["returned_model"]
        or receipt.get("returned_provider") != metadata["returned_provider"]
        or receipt.get("finish_reason") != metadata["finish_reason"]
        or receipt.get("content_sha256") != metadata["content_sha256"]
        or receipt.get("usage") != metadata["usage"]
    ):
        raise DataValidationError("OpenRouter receipt differs from its response archive")
    succeeded = receipt.get("error_type") is None
    if succeeded is not expect_success:
        raise DataValidationError("OpenRouter attempt success sequence differs")
    content: str | None = None
    response_value: Mapping[str, Any] | None = None
    if archived is not None:
        try:
            decoded = json.loads(archived)
        except (UnicodeDecodeError, json.JSONDecodeError):
            decoded = None
        if isinstance(decoded, Mapping):
            response_value = decoded
            choices = decoded.get("choices")
            choice = choices[0] if isinstance(choices, list) and len(choices) == 1 else None
            message = choice.get("message") if isinstance(choice, Mapping) else None
            candidate = message.get("content") if isinstance(message, Mapping) else None
            content = candidate if isinstance(candidate, str) else None
    if expect_success:
        if (
            status != 200
            or response_value is None
            or receipt.get("returned_model")
            not in {route.request_model, route.canonical_slug}
            or receipt.get("returned_provider") != route.provider_name
            or receipt.get("finish_reason") != "stop"
            or content != expected_content
            or receipt.get("error_message") is not None
            or receipt.get("transient") is not False
        ):
            raise DataValidationError("OpenRouter success receipt does not replay")
    else:
        error_type = receipt.get("error_type")
        error_message = receipt.get("error_message")
        replayed_error = _replay_failure_message(
            archived,
            status=status,
            route=route,
            accepted_success_content=accepted_success_content,
        )
        derived_transient = (
            bool(receipt.get("transient"))
            if archived is None and status is None
            else status in _TRANSIENT_HTTP
            if status != 200
            else replayed_error
            not in {
                "OpenRouter substituted the grader model",
                "OpenRouter substituted the grader provider",
            }
        )
        if (
            error_type != "OpenRouterError"
            or not isinstance(error_message, str)
            or not error_message
            or error_message != replayed_error
            or type(receipt.get("transient")) is not bool
            or receipt.get("transient") is not derived_transient
            or (expect_retry and receipt.get("transient") is not True)
        ):
            raise DataValidationError(
                "OpenRouter retry receipt is not a replayable transient failure"
            )
        if expect_retry and status is not None and status != 200 and status not in _TRANSIENT_HTTP:
            raise DataValidationError("OpenRouter retried a permanent HTTP failure")
        if status == 200:
            if (
                response_value is not None
                and receipt.get("returned_model")
                not in {None, route.request_model, route.canonical_slug}
            ) or receipt.get("returned_provider") not in {None, route.provider_name}:
                raise DataValidationError("OpenRouter retried a provider substitution")
            if (
                response_value is not None
                and content is not None
                and receipt.get("finish_reason") == "stop"
                and accepted_success_content is not None
                and accepted_success_content(content)
            ):
                raise DataValidationError("OpenRouter marked a valid response as failed")
    return archived


class OpenRouterTransport:
    """One exact-model transport with secret-redacted replayable receipts."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        sender: Sender | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY")
        if not isinstance(key, str) or not key.strip():
            raise PermanentGraderError("OPENROUTER_API_KEY is missing or blank")
        if timeout_seconds <= 0:
            raise DataValidationError("OpenRouter timeout must be positive")
        self._api_key = key.strip()
        self._sender = sender or _default_sender
        self._timeout_seconds = float(timeout_seconds)
        self.receipts: list[OpenRouterAttemptReceipt] = []

    def mark_last_content_rejected(self, *, attempt: int) -> None:
        """Convert a structurally valid text response into a replayable parser retry."""

        if not self.receipts:
            raise DataValidationError("OpenRouter has no response receipt to reject")
        receipt = self.receipts[-1]
        if receipt.attempt != attempt or receipt.error_type is not None:
            raise DataValidationError("OpenRouter parser rejection targets the wrong attempt")
        self.receipts[-1] = replace(
            receipt,
            error_type="OpenRouterError",
            error_message=_CONTENT_VALIDATION_FAILURE,
            transient=True,
        )

    def request_payload(self, prompt: str, spec: OfficialGraderSpec) -> dict[str, Any]:
        route = route_for_grader(spec)
        payload: dict[str, Any] = {
            "model": route.request_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "stream": False,
            "provider": {
                "only": [route.provider_slug],
                "allow_fallbacks": False,
                "require_parameters": True,
            },
        }
        if route.reasoning_enabled:
            payload["reasoning"] = {"enabled": True, "exclude": True}
        return payload

    def invoke(self, prompt: str, spec: OfficialGraderSpec, *, attempt: int) -> str:
        route = route_for_grader(spec)
        payload = self.request_payload(prompt, spec)
        body = canonical_json(payload).encode("utf-8")
        request_sha256 = hashlib.sha256(body).hexdigest()
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        request = urllib.request.Request(
            _ENDPOINT,
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "X-OpenRouter-Metadata": "enabled",
            },
            method="POST",
        )
        started = time.perf_counter()
        status: int | None = None
        response_body: bytes | None = None
        try:
            status, response_body = self._sender(request, self._timeout_seconds)
            receipt, content = self._validate_response(
                response_body,
                status=status,
                route=route,
                spec=spec,
                secret=self._api_key,
                attempt=attempt,
                request_sha256=request_sha256,
                prompt_sha256=prompt_sha256,
                latency_seconds=time.perf_counter() - started,
            )
            self.receipts.append(receipt)
            return content
        except OpenRouterError as exc:
            observed_status = status if status is not None else exc.http_status
            observed_body = _archive_response_body(
                response_body if response_body is not None else exc.response_body,
                secret=self._api_key,
            )
            metadata = _response_metadata(observed_body, secret=self._api_key)
            safe_message = _replay_failure_message(
                observed_body,
                status=observed_status,
                route=route,
                accepted_success_content=lambda content: content.strip()
                in spec.label_mapping,
            )
            if safe_message is None:
                raise DataValidationError(
                    "OpenRouter failure could not be reconstructed from its response"
                ) from exc
            safe_error = OpenRouterError(
                safe_message,
                transient=exc.transient,
                retry_after=exc.retry_after,
                http_status=observed_status,
                response_body=None,
            )
            self.receipts.append(
                OpenRouterAttemptReceipt(
                    schema_version=1,
                    endpoint=_ENDPOINT,
                    attempt=attempt,
                    request_sha256=request_sha256,
                    prompt_sha256=prompt_sha256,
                    requested_model=route.request_model,
                    canonical_slug=route.canonical_slug,
                    required_provider_slug=route.provider_slug,
                    http_status=observed_status,
                    response_sha256=metadata["response_sha256"],
                    response_body_base64=(
                        base64.b64encode(observed_body).decode("ascii")
                        if observed_body is not None
                        else None
                    ),
                    response_id=metadata["response_id"],
                    returned_model=metadata["returned_model"],
                    returned_provider=metadata["returned_provider"],
                    finish_reason=metadata["finish_reason"],
                    content_sha256=metadata["content_sha256"],
                    usage=metadata["usage"],
                    latency_seconds=time.perf_counter() - started,
                    error_type=type(exc).__name__,
                    error_message=safe_message,
                    transient=exc.transient,
                )
            )
            raise safe_error from None

    def invoke_text(
        self,
        prompt: str,
        *,
        payload: Mapping[str, Any],
        route: OpenRouterRoute,
        attempt: int,
    ) -> str:
        """Run one frozen arbitrary-text grader while retaining strict route receipts."""

        body = canonical_json(dict(payload)).encode("utf-8")
        request_sha256 = hashlib.sha256(body).hexdigest()
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        request = urllib.request.Request(
            _ENDPOINT,
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "X-OpenRouter-Metadata": "enabled",
            },
            method="POST",
        )
        started = time.perf_counter()
        status: int | None = None
        response_body: bytes | None = None
        try:
            status, response_body = self._sender(request, self._timeout_seconds)
            receipt, content = self._validate_text_response(
                response_body,
                status=status,
                route=route,
                secret=self._api_key,
                attempt=attempt,
                request_sha256=request_sha256,
                prompt_sha256=prompt_sha256,
                latency_seconds=time.perf_counter() - started,
            )
            self.receipts.append(receipt)
            return content
        except OpenRouterError as exc:
            observed_status = status if status is not None else exc.http_status
            observed_body = _archive_response_body(
                response_body if response_body is not None else exc.response_body,
                secret=self._api_key,
            )
            metadata = _response_metadata(observed_body, secret=self._api_key)
            safe_message = _replay_failure_message(
                observed_body,
                status=observed_status,
                route=route,
                accepted_success_content=lambda content: bool(content.strip()),
            )
            if safe_message is None:
                raise DataValidationError(
                    "OpenRouter failure could not be reconstructed from its response"
                ) from exc
            self.receipts.append(
                OpenRouterAttemptReceipt(
                    schema_version=1,
                    endpoint=_ENDPOINT,
                    attempt=attempt,
                    request_sha256=request_sha256,
                    prompt_sha256=prompt_sha256,
                    requested_model=route.request_model,
                    canonical_slug=route.canonical_slug,
                    required_provider_slug=route.provider_slug,
                    http_status=observed_status,
                    response_sha256=metadata["response_sha256"],
                    response_body_base64=(
                        base64.b64encode(observed_body).decode("ascii")
                        if observed_body is not None
                        else None
                    ),
                    response_id=metadata["response_id"],
                    returned_model=metadata["returned_model"],
                    returned_provider=metadata["returned_provider"],
                    finish_reason=metadata["finish_reason"],
                    content_sha256=metadata["content_sha256"],
                    usage=metadata["usage"],
                    latency_seconds=time.perf_counter() - started,
                    error_type=type(exc).__name__,
                    error_message=safe_message,
                    transient=exc.transient,
                )
            )
            raise OpenRouterError(
                safe_message,
                transient=exc.transient,
                retry_after=exc.retry_after,
                http_status=observed_status,
                response_body=None,
            ) from None

    def _validate_text_response(
        self,
        body: bytes,
        *,
        status: int,
        route: OpenRouterRoute,
        secret: str,
        attempt: int,
        request_sha256: str,
        prompt_sha256: str,
        latency_seconds: float,
    ) -> tuple[OpenRouterAttemptReceipt, str]:
        body = _archive_response_body(body, secret=secret) or b""
        if status != 200:
            raise OpenRouterError(
                f"OpenRouter HTTP {status}: "
                f"{_error_message(body) or 'unexpected response status'}",
                transient=status in _TRANSIENT_HTTP,
                http_status=status,
                response_body=body,
            )
        try:
            value = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OpenRouterError("OpenRouter returned invalid JSON", transient=True) from exc
        if not isinstance(value, Mapping) or value.get("error") is not None:
            raise OpenRouterError(
                _error_message(body) or "OpenRouter returned an error payload",
                transient=True,
                response_body=body,
            )
        choices = value.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise OpenRouterError("OpenRouter response must contain one choice", transient=True)
        choice = choices[0]
        message = choice.get("message") if isinstance(choice, Mapping) else None
        content = message.get("content") if isinstance(message, Mapping) else None
        finish_reason = choice.get("finish_reason") if isinstance(choice, Mapping) else None
        if not isinstance(content, str) or not content.strip():
            raise OpenRouterError("OpenRouter returned empty grader content", transient=True)
        if finish_reason != "stop":
            raise OpenRouterError("OpenRouter grader did not finish with stop", transient=True)
        if value.get("model") not in {route.request_model, route.canonical_slug}:
            raise OpenRouterError("OpenRouter substituted the grader model", transient=False)
        if value.get("provider") != route.provider_name:
            raise OpenRouterError("OpenRouter substituted the grader provider", transient=False)
        metadata = _response_metadata(body, secret=secret)
        return (
            OpenRouterAttemptReceipt(
                schema_version=1,
                endpoint=_ENDPOINT,
                attempt=attempt,
                request_sha256=request_sha256,
                prompt_sha256=prompt_sha256,
                requested_model=route.request_model,
                canonical_slug=route.canonical_slug,
                required_provider_slug=route.provider_slug,
                http_status=status,
                response_sha256=hashlib.sha256(body).hexdigest(),
                response_body_base64=base64.b64encode(body).decode("ascii"),
                response_id=metadata["response_id"],
                returned_model=metadata["returned_model"],
                returned_provider=metadata["returned_provider"],
                finish_reason=str(finish_reason),
                content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                usage=metadata["usage"],
                latency_seconds=latency_seconds,
                error_type=None,
                error_message=None,
                transient=False,
            ),
            content,
        )

    def _validate_response(
        self,
        body: bytes,
        *,
        status: int,
        route: OpenRouterRoute,
        spec: OfficialGraderSpec,
        secret: str,
        attempt: int,
        request_sha256: str,
        prompt_sha256: str,
        latency_seconds: float,
    ) -> tuple[OpenRouterAttemptReceipt, str]:
        body = _archive_response_body(body, secret=secret) or b""
        response_sha256 = hashlib.sha256(body).hexdigest()
        if status != 200:
            raise OpenRouterError(
                f"OpenRouter HTTP {status}: "
                f"{_error_message(body) or 'unexpected response status'}",
                transient=status in _TRANSIENT_HTTP,
                http_status=status,
                response_body=body,
            )
        try:
            value = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OpenRouterError("OpenRouter returned invalid JSON", transient=True) from exc
        if not isinstance(value, Mapping) or value.get("error") is not None:
            raise OpenRouterError(
                _error_message(body) or "OpenRouter returned an error payload", transient=True
            )
        choices = value.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise OpenRouterError("OpenRouter response must contain one choice", transient=True)
        choice = choices[0]
        message = choice.get("message") if isinstance(choice, Mapping) else None
        content = message.get("content") if isinstance(message, Mapping) else None
        finish_reason = choice.get("finish_reason") if isinstance(choice, Mapping) else None
        returned_model = value.get("model")
        returned_provider = value.get("provider")
        if not isinstance(content, str) or not content:
            raise OpenRouterError("OpenRouter returned empty grader content", transient=True)
        if finish_reason != "stop":
            raise OpenRouterError("OpenRouter grader did not finish with stop", transient=True)
        if returned_model not in {route.request_model, route.canonical_slug}:
            raise OpenRouterError("OpenRouter substituted the grader model", transient=False)
        if returned_provider != route.provider_name:
            raise OpenRouterError("OpenRouter substituted the grader provider", transient=False)
        if content.strip() not in spec.label_mapping:
            raise OpenRouterError(
                f"official grader returned unknown label {content.strip()!r}", transient=True
            )
        metadata = _response_metadata(body, secret=secret)
        receipt = OpenRouterAttemptReceipt(
            schema_version=1,
            endpoint=_ENDPOINT,
            attempt=attempt,
            request_sha256=request_sha256,
            prompt_sha256=prompt_sha256,
            requested_model=route.request_model,
            canonical_slug=route.canonical_slug,
            required_provider_slug=route.provider_slug,
            http_status=status,
            response_sha256=response_sha256,
            response_body_base64=base64.b64encode(body).decode("ascii"),
            response_id=metadata["response_id"],
            returned_model=metadata["returned_model"],
            returned_provider=metadata["returned_provider"],
            finish_reason=str(finish_reason),
            content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            usage=metadata["usage"],
            latency_seconds=latency_seconds,
            error_type=None,
            error_message=None,
            transient=False,
        )
        return receipt, content


def verify_openrouter_catalog(
    catalog: Mapping[str, Any], specs: tuple[OfficialGraderSpec, ...]
) -> Mapping[str, str]:
    data = catalog.get("data")
    if not isinstance(data, list):
        raise DataValidationError("OpenRouter catalog lacks model data")
    available = {
        str(item.get("id")): str(item.get("canonical_slug"))
        for item in data
        if isinstance(item, Mapping)
    }
    verified: dict[str, str] = {}
    for spec in specs:
        route = route_for_grader(spec)
        if available.get(route.request_model) != route.canonical_slug:
            raise DataValidationError(
                "OpenRouter catalog lacks exact grader "
                f"{route.request_model}@{route.canonical_slug}"
            )
        verified[spec.benchmark] = route.canonical_slug
    return verified


def run_openrouter_grader(
    spec: OfficialGraderSpec,
    request: GradingRequest,
    transport: OpenRouterTransport,
    *,
    sleeper: Callable[[float], None] = time.sleep,
) -> OfficialGradeRecord:
    prompt = render_grader_prompt(spec, request)
    last_response = ""
    last_error = "official grader did not run"
    last_attempt = 0
    for attempt in range(1, spec.maximum_attempts + 1):
        last_attempt = attempt
        try:
            response = transport.invoke(prompt, spec, attempt=attempt)
            last_response = response
            label = response.strip()
            outcome = spec.label_mapping.get(label)
            if outcome is None:
                raise OpenRouterError(
                    f"official grader returned unknown label {label!r}", transient=True
                )
            return OfficialGradeRecord(
                request_fingerprint=request.digest,
                grader_fingerprint=spec.digest,
                outcome=outcome,
                raw_response=response,
                attempts=attempt,
                error=None,
            )
        except OpenRouterError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if not exc.transient:
                break
            if attempt < spec.maximum_attempts:
                sleeper(exc.retry_after if exc.retry_after is not None else float(attempt))
    return OfficialGradeRecord(
        request_fingerprint=request.digest,
        grader_fingerprint=spec.digest,
        outcome=Outcome.UNSCORABLE,
        raw_response=last_response,
        attempts=max(last_attempt, 1),
        error=last_error,
    )


def openrouter_adapter_digest() -> str:
    """Bind the exact fail-closed request, parsing, receipt, and retry implementation."""

    return sha256_file(Path(__file__))

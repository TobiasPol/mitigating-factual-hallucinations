"""Strict, runtime-owned VLLM generation resource evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from mfh.contracts import GenerationRecord
from mfh.errors import DataValidationError
from mfh.inference.vllm_runtime import VllmGenerationOutput

GENERATION_RUNTIME_METRIC_KEYS = frozenset(
    {
        "schema_version",
        "gpu_total_memory_bytes",
        "peak_memory_bytes",
        "generation_peak_memory_bytes",
        "auxiliary_peak_memory_bytes",
        "active_memory_bytes",
        "cache_memory_bytes",
        "prompt_tokens_per_second",
        "generation_tokens_per_second",
        "generation_wall_time_seconds",
        "stop_type",
        "stopping_token_id",
    }
)


def _runtime_memory_envelope(runtime_identity: Mapping[str, Any]) -> int:
    value = runtime_identity.get("gpu_total_memory_bytes")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DataValidationError("vLLM runtime identity lacks a GPU-memory envelope")
    return value


def build_generation_runtime_metrics(
    generated: VllmGenerationOutput,
    *,
    runtime_identity: Mapping[str, Any],
    auxiliary_peak_memory_bytes: int = 0,
) -> dict[str, Any]:
    """Bind one native generation to measured time, throughput, and memory."""

    envelope = _runtime_memory_envelope(runtime_identity)
    metrics: dict[str, Any] = {
        "schema_version": 1,
        "gpu_total_memory_bytes": envelope,
        "peak_memory_bytes": max(
            generated.peak_memory_bytes, auxiliary_peak_memory_bytes
        ),
        "generation_peak_memory_bytes": generated.peak_memory_bytes,
        "auxiliary_peak_memory_bytes": auxiliary_peak_memory_bytes,
        "active_memory_bytes": generated.active_memory_bytes,
        "cache_memory_bytes": generated.cache_memory_bytes,
        "prompt_tokens_per_second": generated.prompt_tokens_per_second,
        "generation_tokens_per_second": generated.generation_tokens_per_second,
        "generation_wall_time_seconds": generated.latency_seconds,
        "stop_type": generated.stop_type,
        "stopping_token_id": generated.stopping_token_id,
    }
    validate_generation_runtime_metrics(metrics)
    return metrics


def validate_generation_runtime_metrics(
    value: object,
    *,
    record: GenerationRecord | None = None,
    runtime_identity: Mapping[str, Any] | None = None,
    expected_auxiliary_peak_memory_bytes: int | None = None,
) -> Mapping[str, Any]:
    """Reject missing, forged, non-finite, or over-envelope generation evidence."""

    if not isinstance(value, Mapping) or set(value) != GENERATION_RUNTIME_METRIC_KEYS:
        raise DataValidationError("generation runtime metrics schema differs")
    integer_names = (
        "gpu_total_memory_bytes",
        "peak_memory_bytes",
        "generation_peak_memory_bytes",
        "auxiliary_peak_memory_bytes",
        "active_memory_bytes",
        "cache_memory_bytes",
    )
    if any(
        isinstance(value[name], bool) or not isinstance(value[name], int)
        for name in integer_names
    ):
        raise DataValidationError("generation runtime memory metrics are invalid")
    envelope = int(value["gpu_total_memory_bytes"])
    peak = int(value["peak_memory_bytes"])
    generation_peak = int(value["generation_peak_memory_bytes"])
    auxiliary_peak = int(value["auxiliary_peak_memory_bytes"])
    active = int(value["active_memory_bytes"])
    cache = int(value["cache_memory_bytes"])
    prompt_rate = value["prompt_tokens_per_second"]
    generation_rate = value["generation_tokens_per_second"]
    wall_time = value["generation_wall_time_seconds"]
    stop_type = value["stop_type"]
    stopping_token_id = value["stopping_token_id"]
    if (
        value["schema_version"] != 1
        or envelope <= 0
        or generation_peak <= 0
        or auxiliary_peak < 0
        or peak != max(generation_peak, auxiliary_peak)
        or peak > envelope
        or active < 0
        or cache < 0
        or active > envelope
        or cache > envelope
        or isinstance(prompt_rate, bool)
        or not isinstance(prompt_rate, int | float)
        or not math.isfinite(float(prompt_rate))
        or float(prompt_rate) <= 0
        or isinstance(generation_rate, bool)
        or not isinstance(generation_rate, int | float)
        or not math.isfinite(float(generation_rate))
        or float(generation_rate) <= 0
        or isinstance(wall_time, bool)
        or not isinstance(wall_time, int | float)
        or not math.isfinite(float(wall_time))
        or float(wall_time) <= 0
        or type(stop_type) is not str
        or not stop_type
        or (
            stopping_token_id is not None
            and (isinstance(stopping_token_id, bool) or not isinstance(stopping_token_id, int))
        )
    ):
        raise DataValidationError("generation runtime metrics are invalid or over envelope")
    if runtime_identity is not None and envelope != _runtime_memory_envelope(runtime_identity):
        raise DataValidationError("generation runtime memory envelope differs from attestation")
    if (
        expected_auxiliary_peak_memory_bytes is not None
        and (
            isinstance(expected_auxiliary_peak_memory_bytes, bool)
            or not isinstance(expected_auxiliary_peak_memory_bytes, int)
            or expected_auxiliary_peak_memory_bytes < 0
            or auxiliary_peak != expected_auxiliary_peak_memory_bytes
        )
    ):
        raise DataValidationError(
            "generation auxiliary peak differs from its embedded source evidence"
        )
    if record is not None and not math.isclose(
        float(wall_time),
        record.generation_latency_seconds,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise DataValidationError("generation runtime wall time differs from its record")
    return value

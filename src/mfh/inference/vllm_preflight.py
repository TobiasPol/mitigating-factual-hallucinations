"""Live A100 preflight for the pinned Qwen 3.6 ModelOpt/vLLM runtime."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np

from mfh.config import load_model_spec
from mfh.contracts import ActivationSite, PromptSpec, TokenScope
from mfh.errors import ConfigurationError, DataValidationError, FrozenArtifactError
from mfh.inference.transformers_snapshot import verify_transformers_snapshot
from mfh.inference.vllm_research import (
    VllmResearchInterventionState,
    VllmResearchRuntime,
    vllm_research_toolchain_identity,
)
from mfh.inference.vllm_runtime import _state_spec
from mfh.provenance import sha256_file, stable_hash

_ROOT_FIELDS = {
    "schema_version",
    "policy_id",
    "model",
    "hardware",
    "software",
    "inference",
    "validation",
    "policy_digest",
}
_RECEIPT_FIELDS = {
    "schema_version",
    "status",
    "policy_path",
    "policy_sha256",
    "policy_digest",
    "model",
    "runtime_identity",
    "software",
    "prompt",
    "architecture",
    "intervention",
    "peak_memory_bytes",
    "receipt_digest",
}
_SITES = (
    ActivationSite.POST_ATTENTION,
    ActivationSite.POST_MLP,
    ActivationSite.BLOCK_OUTPUT,
)


def _read_json(path: Path, context: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ConfigurationError(f"{context} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot read {context}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"{context} must be a JSON object")
    return value


def load_vllm_runtime_policy(path: str | Path) -> Mapping[str, Any]:
    """Load and self-verify the static CUDA runtime policy."""

    policy = _read_json(Path(path).absolute(), "vLLM runtime policy")
    if set(policy) != _ROOT_FIELDS or policy.get("schema_version") != 2:
        raise ConfigurationError("vLLM runtime policy differs from schema version 2")
    body = dict(policy)
    declared = body.pop("policy_digest", None)
    if declared != stable_hash(body):
        raise ConfigurationError("vLLM runtime policy digest differs")
    for name in ("model", "hardware", "software", "inference", "validation"):
        if not isinstance(policy.get(name), Mapping):
            raise ConfigurationError(f"vLLM runtime policy {name} must be a mapping")
    return policy


def _validate_static_bindings(
    policy: Mapping[str, Any],
    *,
    project_root: Path,
    model_config: Path,
    snapshot_manifest: Path,
) -> None:
    model = policy["model"]
    validation = policy["validation"]
    assert isinstance(model, Mapping)
    assert isinstance(validation, Mapping)
    if (
        model.get("model_config") != model_config.relative_to(project_root).as_posix()
        or model.get("model_config_sha256") != sha256_file(model_config)
        or model.get("snapshot_manifest")
        != snapshot_manifest.relative_to(project_root).as_posix()
        or model.get("snapshot_manifest_sha256") != sha256_file(snapshot_manifest)
    ):
        raise DataValidationError("vLLM policy static model bindings differ")
    bindings = validation.get("source_bindings")
    if not isinstance(bindings, Mapping) or not bindings:
        raise ConfigurationError("vLLM policy source bindings are invalid")
    for relative, expected in bindings.items():
        if (
            not isinstance(relative, str)
            or not isinstance(expected, str)
            or sha256_file(project_root / relative) != expected
        ):
            raise DataValidationError(f"vLLM policy source binding differs: {relative}")


def _validate_runtime(policy: Mapping[str, Any], identity: Mapping[str, Any]) -> None:
    model = policy["model"]
    hardware = policy["hardware"]
    software = policy["software"]
    assert isinstance(model, Mapping)
    assert isinstance(hardware, Mapping)
    assert isinstance(software, Mapping)
    gpu_name = identity.get("gpu_name")
    required_gpu_name = hardware.get("gpu_name_contains")
    if (
        identity.get("backend") != "vllm"
        or identity.get("vllm") != software.get("vllm")
        or identity.get("model_class") != model.get("model_class")
        or identity.get("num_layers") != model.get("num_layers")
        or identity.get("hidden_size") != model.get("hidden_size")
        or not isinstance(gpu_name, str)
        or not isinstance(required_gpu_name, str)
        or required_gpu_name not in gpu_name
        or identity.get("cuda_capability") != hardware.get("cuda_capability")
        or type(identity.get("gpu_total_memory_bytes")) is not int
        or int(identity["gpu_total_memory_bytes"])
        < int(hardware.get("minimum_vram_bytes", 0))
        or identity.get("tensor_parallel_size") != 1
        or identity.get("quantization_loader") != hardware.get("quantization_loader")
        or identity.get("quantization_config_class")
        != hardware.get("quantization_config_class")
        or identity.get("quantization_execution")
        != hardware.get("quantization_execution")
    ):
        raise DataValidationError("live vLLM A100 runtime identity differs from policy")


def _layer_type(layer: int) -> str:
    return "full_attention" if layer % 4 == 3 else "linear_attention"


def _hook_check(
    runtime: VllmResearchRuntime,
    rendered: Any,
    baseline: Any,
    baseline_logprobs: Mapping[int, float],
    *,
    layer: int,
    site: ActivationSite,
    alpha: float,
    hidden_size: int,
) -> dict[str, Any]:
    zero = VllmResearchInterventionState()
    zero_generation = runtime.generate_with_interventions(
        rendered,
        max_new_tokens=2,
        intervention_states={(layer, site): zero},
    )
    direction = np.ones((hidden_size,), dtype=np.float32) / math.sqrt(hidden_size)
    steered = VllmResearchInterventionState(
        direction=direction,
        alpha=alpha,
        token_scope=TokenScope.FINAL_PROMPT,
    )
    steered_generation = runtime.generate_with_interventions(
        rendered,
        max_new_tokens=2,
        intervention_states={(layer, site): steered},
    )
    if (
        steered.captured is None
        or steered.intervened is None
        or len(steered.applied_pre_history) != 1
        or len(steered.applied_post_history) != 1
    ):
        raise DataValidationError("vLLM preflight hook did not capture its exact edit")
    delta_error, materialized_post_error, materialized_changed_coordinates = (
        _dtype_materialization_metrics(
            steered.applied_pre_history[0],
            steered.applied_post_history[0],
            direction=direction,
            alpha=alpha,
        )
    )
    specification = _state_spec(
        layer,
        site,
        VllmResearchInterventionState(
            direction=direction,
            alpha=alpha,
            token_scope=TokenScope.FINAL_PROMPT,
        ),
        prompt_tokens=len(rendered.token_ids),
    )
    distribution_request, distribution_hooks, _latency = runtime.base.request(
        rendered.token_ids,
        max_tokens=1,
        specifications=(specification,),
        logprobs=-1,
    )
    steered_logprobs = _generated_logprobs(distribution_request)
    raw_specs = distribution_hooks.get("specs")
    raw_distribution = (
        raw_specs.get(f"{layer}:{site.value}")
        if isinstance(raw_specs, Mapping)
        else None
    )
    if (
        not isinstance(raw_distribution, Mapping)
        or raw_distribution.get("applications") != 1
        or not isinstance(raw_distribution.get("applied_pre_history"), list)
        or len(raw_distribution["applied_pre_history"]) != 1
        or not isinstance(raw_distribution.get("applied_post_history"), list)
        or len(raw_distribution["applied_post_history"]) != 1
    ):
        raise DataValidationError(
            "vLLM preflight distribution request did not apply its exact edit"
        )
    (
        distribution_delta_error,
        distribution_materialized_post_error,
        distribution_changed_coordinates,
    ) = _dtype_materialization_metrics(
        raw_distribution["applied_pre_history"][0],
        raw_distribution["applied_post_history"][0],
        direction=direction,
        alpha=alpha,
    )
    if set(steered_logprobs) != set(baseline_logprobs):
        raise DataValidationError("vLLM preflight full-vocabulary logprob keys differ")
    finite_deltas = [
        abs(steered_logprobs[token] - baseline_logprobs[token])
        for token in baseline_logprobs
        if math.isfinite(steered_logprobs[token])
        and math.isfinite(baseline_logprobs[token])
    ]
    downstream_max_abs_delta = max(finite_deltas, default=0.0)
    passed = (
        zero_generation.token_ids == baseline.token_ids
        and zero_generation.text == baseline.text
        and zero.applications == 0
        and steered.applications == 1
        and materialized_post_error <= 5e-3
        and materialized_changed_coordinates > 0
        and distribution_materialized_post_error <= 5e-3
        and distribution_changed_coordinates > 0
        and downstream_max_abs_delta > 1e-7
    )
    return {
        "status": "passed" if passed else "failed",
        "layer": layer,
        "layer_type": _layer_type(layer),
        "site": site.value,
        "capture_shape": list(np.asarray(steered.captured).shape),
        "alpha": alpha,
        "applications": steered.applications,
        "zero_generation_exact_parity": (
            zero_generation.token_ids == baseline.token_ids
            and zero_generation.text == baseline.text
        ),
        "final_delta_max_abs_error": delta_error,
        "dtype_materialized_post_max_abs_error": materialized_post_error,
        "materialized_changed_coordinates": materialized_changed_coordinates,
        "distribution_applications": int(raw_distribution["applications"]),
        "distribution_final_delta_max_abs_error": distribution_delta_error,
        "distribution_dtype_materialized_post_max_abs_error": (
            distribution_materialized_post_error
        ),
        "distribution_materialized_changed_coordinates": (
            distribution_changed_coordinates
        ),
        "downstream_logprob_max_abs_delta": downstream_max_abs_delta,
        "peak_memory_bytes": max(
            steered_generation.peak_memory_bytes,
            int(distribution_hooks["peak_memory_bytes"]),
        ),
    }


def _dtype_materialization_metrics(
    pre: Any,
    post: Any,
    *,
    direction: np.ndarray[Any, Any],
    alpha: float,
) -> tuple[float, float, int]:
    """Compare a live BF16 edit with its intended and materialized forms."""

    import torch

    observed_pre = np.asarray(pre, dtype=np.float32)
    observed_post = np.asarray(post, dtype=np.float32)
    observed_delta = observed_post - observed_pre
    intended_delta_error = float(
        np.max(np.abs(observed_delta - alpha * direction))
    )
    pre_bf16 = torch.from_numpy(observed_pre).to(torch.bfloat16)
    direction_bf16 = torch.from_numpy(direction).to(torch.bfloat16)
    expected_post = pre_bf16.add(direction_bf16, alpha=alpha).float().numpy()
    materialized_post_error = float(np.max(np.abs(observed_post - expected_post)))
    return (
        intended_delta_error,
        materialized_post_error,
        int(np.count_nonzero(observed_delta)),
    )


def _generated_logprobs(request: Any) -> dict[int, float]:
    """Extract one full-vocabulary generated-token logprob mapping from vLLM."""

    rows = request.outputs[0].logprobs
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], Mapping):
        raise DataValidationError("vLLM preflight requires one full logprob distribution")
    values: dict[int, float] = {}
    for raw_token, raw_value in rows[0].items():
        token = int(raw_token)
        logprob = getattr(raw_value, "logprob", raw_value)
        if isinstance(logprob, bool) or not isinstance(logprob, int | float):
            raise DataValidationError("vLLM preflight logprob value is invalid")
        values[token] = float(logprob)
    if not values:
        raise DataValidationError("vLLM preflight logprob distribution is empty")
    return values


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        raise FrozenArtifactError(
            f"refusing to overwrite vLLM preflight receipt: {path}"
        ) from None
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def run_vllm_preflight(
    *,
    project_root: str | Path,
    model_directory: str | Path,
    model_config: str | Path,
    snapshot_manifest: str | Path,
    runtime_policy: str | Path,
    output: str | Path,
    prompt: str = "What is the capital of France?",
    alpha: float = 2.0,
) -> Mapping[str, Any]:
    """Verify checkpoint, A100 fallback path, and every causal intervention site."""

    from mfh.artifact_namespace import validate_active_study_artifact_paths

    root = Path(project_root).absolute()
    config_path = Path(model_config).absolute()
    manifest_path = Path(snapshot_manifest).absolute()
    policy_path = Path(runtime_policy).absolute()
    snapshot_path = Path(model_directory).absolute()
    destination = validate_active_study_artifact_paths(
        {"vLLM preflight receipt": output}, project_root=root
    )["vLLM preflight receipt"]
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, int | float)
        or not math.isfinite(float(alpha))
        or float(alpha) == 0
    ):
        raise ConfigurationError("vLLM preflight alpha must be finite and nonzero")
    policy = load_vllm_runtime_policy(policy_path)
    _validate_static_bindings(
        policy,
        project_root=root,
        model_config=config_path,
        snapshot_manifest=manifest_path,
    )
    model_spec = load_model_spec(config_path)
    model_policy = policy["model"]
    validation = policy["validation"]
    inference = policy["inference"]
    assert isinstance(model_policy, Mapping)
    assert isinstance(validation, Mapping)
    assert isinstance(inference, Mapping)
    if {
        "name": model_spec.name,
        "repository": model_spec.repository,
        "revision": model_spec.revision,
        "quantization": model_spec.quantization,
        "num_layers": model_spec.num_layers,
        "candidate_layers": list(model_spec.candidate_layers),
    } != {
        key: model_policy.get(key)
        for key in (
            "name", "repository", "revision", "quantization", "num_layers",
            "candidate_layers",
        )
    }:
        raise DataValidationError("vLLM model config differs from runtime policy")
    if (
        hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        != validation.get("preflight_prompt_sha256")
        or float(alpha) != validation.get("preflight_alpha")
    ):
        raise DataValidationError("vLLM preflight prompt or alpha differs from policy")

    snapshot_identity = verify_transformers_snapshot(
        model_spec, snapshot_path, manifest_path
    )
    seed = inference.get("seed")
    if type(seed) is not int:
        raise ConfigurationError("vLLM policy seed is invalid")
    runtime = VllmResearchRuntime.from_spec(
        model_spec,
        snapshot_path=snapshot_path,
        seed=seed,
        research_provenance={"preflight": True},
    )
    try:
        identity = dict(runtime.base.runtime_identity())
        _validate_runtime(policy, identity)
        rendered = runtime.render_prompt(
            PromptSpec(
                prompt_id="preflight-short-answer",
                text="Answer the user's factual question with only a short answer.",
            ),
            prompt,
        )
        baseline = runtime.generate(rendered, max_new_tokens=2)
        baseline_request, baseline_hooks, _latency = runtime.base.request(
            rendered.token_ids,
            max_tokens=1,
            logprobs=-1,
        )
        baseline_logprobs = _generated_logprobs(baseline_request)
        checks: dict[str, Any] = {}
        probe_layers = validation.get("probe_layers")
        if not isinstance(probe_layers, Mapping):
            raise ConfigurationError("vLLM preflight probe layers are invalid")
        for layer_type in ("linear_attention", "full_attention"):
            layer = probe_layers.get(layer_type)
            if type(layer) is not int or _layer_type(layer) != layer_type:
                raise ConfigurationError("vLLM preflight probe layer type differs")
            for site in _SITES:
                checks[f"{layer_type}.{site.value}"] = _hook_check(
                    runtime,
                    rendered,
                    baseline,
                    baseline_logprobs,
                    layer=layer,
                    site=site,
                    alpha=float(alpha),
                    hidden_size=int(model_policy["hidden_size"]),
                )
        if any(check.get("status") != "passed" for check in checks.values()):
            raise DataValidationError("one or more live vLLM hook checks failed")
        toolchain = dict(vllm_research_toolchain_identity())
        body: dict[str, Any] = {
            "schema_version": 2,
            "status": "passed",
            "policy_path": policy_path.relative_to(root).as_posix(),
            "policy_sha256": sha256_file(policy_path),
            "policy_digest": policy["policy_digest"],
            "model": {
                "name": model_spec.name,
                "repository": model_spec.repository,
                "revision": model_spec.revision,
                "quantization": model_spec.quantization,
                "num_layers": model_spec.num_layers,
                "hidden_size": int(model_policy["hidden_size"]),
                "snapshot_identity": dict(snapshot_identity),
            },
            "runtime_identity": identity,
            "software": {
                "toolchain": toolchain,
                "source_bindings": dict(validation["source_bindings"]),
            },
            "prompt": {
                "text_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "rendered_sha256": rendered.sha256,
                "token_ids_sha256": rendered.token_ids_sha256,
                "token_count": len(rendered.token_ids),
                "thinking_enabled": False,
            },
            "architecture": {
                "layer_types": [
                    _layer_type(layer) for layer in range(model_spec.num_layers)
                ],
                "candidate_layer_types": {
                    str(layer): _layer_type(layer)
                    for layer in model_spec.candidate_layers
                },
            },
            "intervention": {
                "alpha": float(alpha),
                "direction": "normalized-all-ones-feasibility-only",
                "checks": checks,
            },
            "peak_memory_bytes": max(
                [baseline.peak_memory_bytes, int(baseline_hooks["peak_memory_bytes"])]
                + [int(check.get("peak_memory_bytes", 0)) for check in checks.values()]
            ),
        }
        receipt = {**body, "receipt_digest": stable_hash(body)}
        validate_vllm_preflight_receipt(
            receipt,
            project_root=root,
            model_config=config_path,
            snapshot_directory=snapshot_path,
            snapshot_manifest=manifest_path,
            runtime_policy=policy_path,
        )
        _write_once(destination, receipt)
        return receipt
    finally:
        with suppress(Exception):
            runtime.close()


def validate_vllm_preflight_receipt(
    receipt: Mapping[str, Any] | str | Path,
    *,
    project_root: str | Path,
    model_config: str | Path,
    snapshot_directory: str | Path,
    snapshot_manifest: str | Path,
    runtime_policy: str | Path,
) -> Mapping[str, Any]:
    """Replay all static claims of a frozen A100 preflight receipt."""

    value = (
        _read_json(Path(receipt).absolute(), "vLLM preflight receipt")
        if isinstance(receipt, str | Path)
        else dict(receipt)
    )
    if set(value) != _RECEIPT_FIELDS:
        raise DataValidationError("vLLM preflight receipt fields differ")
    body = dict(value)
    digest = body.pop("receipt_digest", None)
    if digest != stable_hash(body):
        raise DataValidationError("vLLM preflight receipt digest differs")
    if value.get("schema_version") != 2 or value.get("status") != "passed":
        raise DataValidationError("vLLM preflight receipt is not a passed schema-2 receipt")
    root = Path(project_root).absolute()
    config_path = Path(model_config).absolute()
    manifest_path = Path(snapshot_manifest).absolute()
    policy_path = Path(runtime_policy).absolute()
    policy = load_vllm_runtime_policy(policy_path)
    _validate_static_bindings(
        policy,
        project_root=root,
        model_config=config_path,
        snapshot_manifest=manifest_path,
    )
    if (
        value.get("policy_path") != policy_path.relative_to(root).as_posix()
        or value.get("policy_sha256") != sha256_file(policy_path)
        or value.get("policy_digest") != policy.get("policy_digest")
    ):
        raise DataValidationError("vLLM preflight policy binding differs")
    model_spec = load_model_spec(config_path)
    snapshot = verify_transformers_snapshot(
        model_spec, snapshot_directory, manifest_path
    )
    model = value.get("model")
    runtime_identity = value.get("runtime_identity")
    software = value.get("software")
    prompt = value.get("prompt")
    architecture = value.get("architecture")
    intervention = value.get("intervention")
    model_policy = policy["model"]
    validation = policy["validation"]
    assert isinstance(model_policy, Mapping)
    assert isinstance(validation, Mapping)
    if (
        not isinstance(model, Mapping)
        or model.get("name") != model_spec.name
        or model.get("repository") != model_spec.repository
        or model.get("revision") != model_spec.revision
        or model.get("quantization") != model_spec.quantization
        or model.get("num_layers") != model_spec.num_layers
        or model.get("hidden_size") != model_policy.get("hidden_size")
        or model.get("snapshot_identity") != snapshot
        or not isinstance(runtime_identity, Mapping)
    ):
        raise DataValidationError("vLLM preflight model identity differs")
    _validate_runtime(policy, runtime_identity)
    if (
        not isinstance(software, Mapping)
        or software.get("source_bindings") != validation.get("source_bindings")
        or not isinstance(software.get("toolchain"), Mapping)
    ):
        raise DataValidationError("vLLM preflight software identity differs")
    toolchain = software["toolchain"]
    assert isinstance(toolchain, Mapping)
    if (
        set(toolchain) != {"vllm", "torch", "transformers", "numpy", "nvidia_driver"}
        or toolchain.get("vllm") != policy["software"].get("vllm")
        or any(not isinstance(item, str) or not item for item in toolchain.values())
    ):
        raise DataValidationError("vLLM preflight toolchain identity differs")
    sha256_pattern = re.compile(r"^[0-9a-f]{64}$")
    if (
        not isinstance(prompt, Mapping)
        or set(prompt)
        != {
            "text_sha256",
            "rendered_sha256",
            "token_ids_sha256",
            "token_count",
            "thinking_enabled",
        }
        or prompt.get("text_sha256") != validation.get("preflight_prompt_sha256")
        or sha256_pattern.fullmatch(str(prompt.get("rendered_sha256"))) is None
        or sha256_pattern.fullmatch(str(prompt.get("token_ids_sha256"))) is None
        or type(prompt.get("token_count")) is not int
        or int(prompt["token_count"]) <= 0
        or prompt.get("thinking_enabled") is not False
    ):
        raise DataValidationError("vLLM preflight prompt identity differs")
    expected_layers = [_layer_type(layer) for layer in range(model_spec.num_layers)]
    expected_candidate_types = {
        str(layer): _layer_type(layer) for layer in model_spec.candidate_layers
    }
    checks = intervention.get("checks") if isinstance(intervention, Mapping) else None
    if (
        not isinstance(architecture, Mapping)
        or set(architecture) != {"layer_types", "candidate_layer_types"}
        or architecture.get("layer_types") != expected_layers
        or architecture.get("candidate_layer_types") != expected_candidate_types
        or not isinstance(intervention, Mapping)
        or set(intervention) != {"alpha", "direction", "checks"}
        or intervention.get("alpha") != validation.get("preflight_alpha")
        or intervention.get("direction") != "normalized-all-ones-feasibility-only"
        or not isinstance(checks, Mapping)
    ):
        raise DataValidationError("vLLM preflight architecture or hook evidence differs")
    probe_layers = validation.get("probe_layers")
    if not isinstance(probe_layers, Mapping):
        raise DataValidationError("vLLM preflight probe-layer policy differs")
    expected_checks = {
        f"{layer_type}.{site.value}": (layer, layer_type, site.value)
        for layer_type, layer in probe_layers.items()
        if type(layer) is int
        for site in _SITES
    }
    check_fields = {
        "status",
        "layer",
        "layer_type",
        "site",
        "capture_shape",
        "alpha",
        "applications",
        "zero_generation_exact_parity",
        "final_delta_max_abs_error",
        "dtype_materialized_post_max_abs_error",
        "materialized_changed_coordinates",
        "distribution_applications",
        "distribution_final_delta_max_abs_error",
        "distribution_dtype_materialized_post_max_abs_error",
        "distribution_materialized_changed_coordinates",
        "downstream_logprob_max_abs_delta",
        "peak_memory_bytes",
    }
    if set(checks) != set(expected_checks):
        raise DataValidationError("vLLM preflight hook inventory differs")
    for name, expected in expected_checks.items():
        check = checks[name]
        if not isinstance(check, Mapping):
            raise DataValidationError("vLLM preflight hook evidence is invalid")
        delta_error = check.get("final_delta_max_abs_error")
        materialized_error = check.get("dtype_materialized_post_max_abs_error")
        changed_coordinates = check.get("materialized_changed_coordinates")
        distribution_applications = check.get("distribution_applications")
        distribution_delta_error = check.get(
            "distribution_final_delta_max_abs_error"
        )
        distribution_materialized_error = check.get(
            "distribution_dtype_materialized_post_max_abs_error"
        )
        distribution_changed_coordinates = check.get(
            "distribution_materialized_changed_coordinates"
        )
        downstream_delta = check.get("downstream_logprob_max_abs_delta")
        peak = check.get("peak_memory_bytes")
        if (
            set(check) != check_fields
            or check.get("status") != "passed"
            or (check.get("layer"), check.get("layer_type"), check.get("site"))
            != expected
            or check.get("capture_shape") != [1, 1, model_policy.get("hidden_size")]
            or check.get("alpha") != validation.get("preflight_alpha")
            or check.get("applications") != 1
            or check.get("zero_generation_exact_parity") is not True
            or isinstance(delta_error, bool)
            or not isinstance(delta_error, int | float)
            or not math.isfinite(float(delta_error))
            or float(delta_error) < 0
            or isinstance(materialized_error, bool)
            or not isinstance(materialized_error, int | float)
            or not math.isfinite(float(materialized_error))
            or not 0 <= float(materialized_error) <= 5e-3
            or type(changed_coordinates) is not int
            or changed_coordinates <= 0
            or distribution_applications != 1
            or isinstance(distribution_delta_error, bool)
            or not isinstance(distribution_delta_error, int | float)
            or not math.isfinite(float(distribution_delta_error))
            or float(distribution_delta_error) < 0
            or isinstance(distribution_materialized_error, bool)
            or not isinstance(distribution_materialized_error, int | float)
            or not math.isfinite(float(distribution_materialized_error))
            or not 0 <= float(distribution_materialized_error) <= 5e-3
            or type(distribution_changed_coordinates) is not int
            or distribution_changed_coordinates <= 0
            or isinstance(downstream_delta, bool)
            or not isinstance(downstream_delta, int | float)
            or not math.isfinite(float(downstream_delta))
            or float(downstream_delta) <= 1e-7
            or type(peak) is not int
            or peak <= 0
        ):
            raise DataValidationError("vLLM preflight hook evidence is invalid")
    if (
        type(value.get("peak_memory_bytes")) is not int
        or int(value["peak_memory_bytes"]) <= 0
        or int(value["peak_memory_bytes"])
        > int(policy["hardware"].get("maximum_peak_memory_bytes", 0))
    ):
        raise DataValidationError("vLLM preflight peak memory is invalid")
    return value

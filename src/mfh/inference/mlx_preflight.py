"""Live, tamper-evident MLX hook preflight for the active research model."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

from mfh.config import load_model_spec
from mfh.contracts import ActivationSite
from mfh.errors import ConfigurationError, DataValidationError, FrozenArtifactError
from mfh.inference.mlx_research import mlx_research_toolchain_identity
from mfh.inference.mlx_runtime import MlxInterventionState, MlxRuntime, _mlx_modules, as_numpy
from mfh.inference.transformers_snapshot import verify_transformers_snapshot
from mfh.provenance import sha256_file, stable_hash

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SITES = (
    ActivationSite.POST_ATTENTION,
    ActivationSite.POST_MLP,
    ActivationSite.BLOCK_OUTPUT,
)
_PACKAGES = (
    "mlx",
    "mlx-lm",
    "numpy",
    "safetensors",
    "sentencepiece",
    "tokenizers",
    "torch",
    "transformers",
)
_POLICY_FIELDS = {
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
_MODEL_FIELDS = {
    "name",
    "repository",
    "revision",
    "quantization",
    "num_layers",
    "hidden_size",
    "snapshot_identity",
}
_RUNTIME_IDENTITY_FIELDS = {
    "backend",
    "mlx",
    "mlx_lm",
    "python",
    "machine_model",
    "chip",
    "unified_memory_bytes",
    "physical_cpu_cores",
    "architecture",
    "os",
    "os_build",
    "model_class",
    "tokenizer_class",
    "num_layers",
    "seed",
}
_CHECK_FIELDS = {
    "status",
    "layer",
    "layer_type",
    "site",
    "capture_shape",
    "cached_capture_shape",
    "captured_final_sha256",
    "alpha",
    "module_restored",
    "applications",
    "zero_vector_exact_parity",
    "zero_vector_max_abs_logit_error",
    "cached_zero_max_abs_prefill_logit_error",
    "cached_zero_max_abs_continuation_logit_error",
    "scope_exact",
    "prefix_max_abs_activation_error",
    "final_delta_max_abs_error",
    "cached_prefix_max_abs_activation_error",
    "cached_final_delta_max_abs_error",
    "nonzero_max_abs_logit_change",
    "cached_nonzero_max_abs_prefill_logit_change",
    "cached_nonzero_max_abs_continuation_logit_change",
    "nonzero_changed_cached_continuation",
    "baseline_top_token_id",
    "zero_top_token_id",
    "steered_top_token_id",
    "cached_baseline_token_id",
    "cached_zero_token_id",
    "cached_steered_token_id",
    "baseline_logits_sha256",
    "zero_logits_sha256",
    "steered_logits_sha256",
}


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


def load_mlx_runtime_policy(path: str | Path) -> Mapping[str, Any]:
    """Load the exact static policy that a live M4 Max receipt must satisfy."""

    source = Path(path).absolute()
    policy = _read_json(source, "MLX runtime policy")
    if set(policy) != _POLICY_FIELDS or policy.get("schema_version") != 1:
        raise ConfigurationError("MLX runtime policy fields differ from schema version 1")
    body = dict(policy)
    declared = body.pop("policy_digest", None)
    if not isinstance(declared, str) or _SHA256.fullmatch(declared) is None:
        raise ConfigurationError("MLX runtime policy digest is invalid")
    if stable_hash(body) != declared:
        raise ConfigurationError("MLX runtime policy digest differs")
    for name in ("model", "hardware", "software", "inference", "validation"):
        if not isinstance(policy.get(name), Mapping):
            raise ConfigurationError(f"MLX runtime policy {name} must be a mapping")
    return policy


def _array_sha256(value: Any) -> str:
    array = as_numpy(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _is_exact_zero(value: object) -> bool:
    return type(value) in {int, float} and float(value) == 0.0  # type: ignore[arg-type]


def _is_positive_finite(value: object) -> bool:
    return (
        type(value) in {int, float}
        and math.isfinite(float(value))  # type: ignore[arg-type]
        and float(value) > 0.0  # type: ignore[arg-type]
    )


def _hidden_size(model: Any) -> int:
    language_model = getattr(model, "language_model", None)
    arguments = getattr(language_model, "args", None)
    value = getattr(arguments, "hidden_size", None)
    if type(value) is not int or value <= 0:
        outer = getattr(getattr(model, "args", None), "text_config", None)
        value = outer.get("hidden_size") if isinstance(outer, Mapping) else None
    if type(value) is not int or value <= 0:
        raise DataValidationError("cannot resolve live MLX text hidden size")
    return value


def _last_logits(model: Any, input_ids: Any, mx: Any) -> Any:
    logits = model(input_ids)[:, -1, :]
    mx.eval(logits)
    return logits


def _cached_pair(
    model: Any,
    input_ids: Any,
    mx: Any,
    state: MlxInterventionState | None = None,
) -> tuple[Any, int, Any, Any | None, Any | None]:
    cache = model.make_cache()
    prefill = model(input_ids, cache=cache)[:, -1, :]
    mx.eval(prefill, cache)
    captured = state.captured if state is not None else None
    intervened = state.intervened if state is not None else None
    token_id = int(mx.argmax(prefill, axis=-1).item())
    continuation = model(mx.array([[token_id]]), cache=cache)[:, -1, :]
    mx.eval(continuation, cache)
    return prefill, token_id, continuation, captured, intervened


def _scope_errors(
    captured: Any,
    intervened: Any,
    direction: Any,
    alpha: float,
    mx: Any,
) -> tuple[float, float]:
    prefix_error = (
        float(mx.max(mx.abs(captured[:, :-1, :] - intervened[:, :-1, :])).item())
        if int(captured.shape[1]) > 1
        else 0.0
    )
    expected_final = captured[:, -1, :] + alpha * direction.astype(captured.dtype)[
        None, :
    ]
    final_error = float(
        mx.max(mx.abs(expected_final - intervened[:, -1, :])).item()
    )
    return prefix_error, final_error


def _site_check(
    runtime: MlxRuntime,
    input_ids: Any,
    *,
    layer: int,
    site: ActivationSite,
    alpha: float,
    mx: Any,
) -> dict[str, Any]:
    model = runtime.model
    original_layer = model.layers[layer]
    original_mlp = original_layer.mlp
    baseline = _last_logits(model, input_ids, mx)
    cached_baseline, baseline_token, continuation_baseline, _captured, _intervened = _cached_pair(
        model, input_ids, mx
    )
    zero_state = MlxInterventionState()
    with runtime.intervention(layer=layer, site=site, state=zero_state):
        zero = _last_logits(model, input_ids, mx)
        cached_zero, zero_token, continuation_zero, _zero_capture, _zero_intervened = (
            _cached_pair(model, input_ids, mx, zero_state)
        )
    if zero_state.captured is None:
        raise DataValidationError("MLX zero hook did not capture an activation")

    width = int(zero_state.captured.shape[-1])
    steering_state = MlxInterventionState(
        direction=mx.ones((width,), dtype=mx.float32) / math.sqrt(width),
        alpha=alpha,
    )
    with runtime.intervention(layer=layer, site=site, state=steering_state):
        steered = _last_logits(model, input_ids, mx)
        prompt_captured = steering_state.captured
        prompt_intervened = steering_state.intervened
        (
            cached_steered,
            steered_token,
            continuation_steered,
            cached_captured,
            cached_intervened,
        ) = _cached_pair(model, input_ids, mx, steering_state)
    if (
        steering_state.captured is None
        or steering_state.intervened is None
        or steering_state.applications < 1
    ):
        raise DataValidationError("MLX nonzero hook did not apply an intervention")
    if (
        prompt_captured is None
        or prompt_intervened is None
        or cached_captured is None
        or cached_intervened is None
    ):
        raise DataValidationError("MLX nonzero hook did not retain prompt activations")
    mx.eval(prompt_captured, prompt_intervened, cached_captured, cached_intervened)
    captured = prompt_captured
    intervened = prompt_intervened
    direction = steering_state.direction
    assert direction is not None
    prefix_error, final_error = _scope_errors(
        captured, intervened, direction, alpha, mx
    )
    cached_prefix_error, cached_final_error = _scope_errors(
        cached_captured, cached_intervened, direction, alpha, mx
    )
    zero_error = float(mx.max(mx.abs(baseline - zero)).item())
    cached_zero_error = float(mx.max(mx.abs(cached_baseline - cached_zero)).item())
    continuation_zero_error = float(
        mx.max(mx.abs(continuation_baseline - continuation_zero)).item()
    )
    nonzero_change = float(mx.max(mx.abs(baseline - steered)).item())
    cached_nonzero_change = float(
        mx.max(mx.abs(cached_baseline - cached_steered)).item()
    )
    continuation_nonzero_change = float(
        mx.max(mx.abs(continuation_baseline - continuation_steered)).item()
    )
    restored = (
        model.layers[layer] is original_layer and original_layer.mlp is original_mlp
    )
    passed = all(
        (
            restored,
            zero_error == 0.0,
            cached_zero_error == 0.0,
            continuation_zero_error == 0.0,
            baseline_token == zero_token,
            prefix_error == 0.0,
            final_error == 0.0,
            cached_prefix_error == 0.0,
            cached_final_error == 0.0,
            nonzero_change > 0.0,
            cached_nonzero_change > 0.0,
            continuation_nonzero_change > 0.0,
        )
    )
    return {
        "status": "passed" if passed else "failed",
        "layer": layer,
        "layer_type": "linear_attention" if original_layer.is_linear else "full_attention",
        "site": site.value,
        "capture_shape": [int(value) for value in captured.shape],
        "cached_capture_shape": [int(value) for value in cached_captured.shape],
        "captured_final_sha256": _array_sha256(captured[:, -1, :]),
        "alpha": alpha,
        "module_restored": restored,
        "applications": steering_state.applications,
        "zero_vector_exact_parity": (
            zero_error == cached_zero_error == continuation_zero_error == 0.0
        ),
        "zero_vector_max_abs_logit_error": zero_error,
        "cached_zero_max_abs_prefill_logit_error": cached_zero_error,
        "cached_zero_max_abs_continuation_logit_error": continuation_zero_error,
        "scope_exact": (
            prefix_error
            == final_error
            == cached_prefix_error
            == cached_final_error
            == 0.0
        ),
        "prefix_max_abs_activation_error": prefix_error,
        "final_delta_max_abs_error": final_error,
        "cached_prefix_max_abs_activation_error": cached_prefix_error,
        "cached_final_delta_max_abs_error": cached_final_error,
        "nonzero_max_abs_logit_change": nonzero_change,
        "cached_nonzero_max_abs_prefill_logit_change": cached_nonzero_change,
        "cached_nonzero_max_abs_continuation_logit_change": continuation_nonzero_change,
        "nonzero_changed_cached_continuation": continuation_nonzero_change > 0.0,
        "baseline_top_token_id": int(mx.argmax(baseline, axis=-1).item()),
        "zero_top_token_id": int(mx.argmax(zero, axis=-1).item()),
        "steered_top_token_id": int(mx.argmax(steered, axis=-1).item()),
        "cached_baseline_token_id": baseline_token,
        "cached_zero_token_id": zero_token,
        "cached_steered_token_id": steered_token,
        "baseline_logits_sha256": _array_sha256(baseline),
        "zero_logits_sha256": _array_sha256(zero),
        "steered_logits_sha256": _array_sha256(steered),
    }


def _validate_static_bindings(
    policy: Mapping[str, Any],
    *,
    project_root: Path,
    model_config: Path,
    snapshot_manifest: Path,
) -> None:
    model_policy = policy["model"]
    validation = policy["validation"]
    assert isinstance(model_policy, Mapping)
    assert isinstance(validation, Mapping)
    if (
        model_policy.get("model_config")
        != model_config.relative_to(project_root).as_posix()
        or model_policy.get("model_config_sha256") != sha256_file(model_config)
        or model_policy.get("snapshot_manifest")
        != snapshot_manifest.relative_to(project_root).as_posix()
        or model_policy.get("snapshot_manifest_sha256")
        != sha256_file(snapshot_manifest)
    ):
        raise DataValidationError("MLX runtime policy model bindings differ")
    bindings = validation.get("source_bindings")
    if not isinstance(bindings, Mapping) or not bindings:
        raise ConfigurationError("MLX runtime policy source bindings are invalid")
    for raw_path, expected in bindings.items():
        if not isinstance(raw_path, str) or not isinstance(expected, str):
            raise ConfigurationError("MLX runtime policy source binding is invalid")
        path = project_root / raw_path
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise DataValidationError(f"MLX runtime policy source binding differs: {raw_path}")


def _validate_runtime_against_policy(
    policy: Mapping[str, Any], runtime_identity: Mapping[str, Any]
) -> None:
    hardware = policy["hardware"]
    software = policy["software"]
    assert isinstance(hardware, Mapping)
    assert isinstance(software, Mapping)
    expected_packages = software.get("packages")
    if not isinstance(expected_packages, Mapping):
        raise ConfigurationError("MLX software package policy is invalid")
    observed_packages = {name: importlib.metadata.version(name) for name in _PACKAGES}
    if observed_packages != dict(expected_packages):
        raise DataValidationError("live MLX package versions differ from runtime policy")
    if platform.python_version() != software.get("python"):
        raise DataValidationError("live Python version differs from runtime policy")
    text_fields = ("machine_model", "chip", "os", "os_build", "model_class", "tokenizer_class")
    if (
        set(runtime_identity) != _RUNTIME_IDENTITY_FIELDS
        or runtime_identity.get("backend") != "mlx"
        or runtime_identity.get("mlx") != expected_packages.get("mlx")
        or runtime_identity.get("mlx_lm") != expected_packages.get("mlx-lm")
        or runtime_identity.get("python") != software.get("python")
        or runtime_identity.get("chip") != hardware.get("chip")
        or runtime_identity.get("architecture") != hardware.get("architecture")
        or runtime_identity.get("unified_memory_bytes")
        != hardware.get("unified_memory_bytes")
        or type(runtime_identity.get("physical_cpu_cores")) is not int
        or int(runtime_identity.get("physical_cpu_cores", 0)) <= 0
        or any(
            not isinstance(runtime_identity.get(field), str)
            or not str(runtime_identity[field]).strip()
            for field in text_fields
        )
    ):
        raise DataValidationError("live MLX hardware or runtime identity differs from policy")


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
        raise FrozenArtifactError(f"refusing to overwrite MLX preflight receipt: {path}") from None
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def run_mlx_preflight(
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
    """Load the future production model and freeze live architecture/hook evidence."""

    from mfh.artifact_namespace import validate_active_study_artifact_paths

    root = Path(project_root).absolute()
    config_path = Path(model_config).absolute()
    manifest_path = Path(snapshot_manifest).absolute()
    policy_path = Path(runtime_policy).absolute()
    model_path = Path(model_directory).absolute()
    destination = validate_active_study_artifact_paths(
        {"MLX preflight receipt": output},
        project_root=root,
    )["MLX preflight receipt"]
    if root.is_symlink() or not root.is_dir():
        raise ConfigurationError("MLX preflight project root must be a regular directory")
    for source in (config_path, manifest_path, policy_path):
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise ConfigurationError(
                "MLX preflight static inputs must stay in the project"
            ) from exc
    if isinstance(alpha, bool) or not isinstance(alpha, int | float) or not math.isfinite(alpha):
        raise ConfigurationError("MLX preflight alpha must be finite")
    if float(alpha) == 0:
        raise ConfigurationError("MLX preflight alpha must be nonzero")
    policy = load_mlx_runtime_policy(policy_path)
    _validate_static_bindings(
        policy,
        project_root=root,
        model_config=config_path,
        snapshot_manifest=manifest_path,
    )
    model_spec = load_model_spec(config_path)
    model_policy = policy["model"]
    inference = policy["inference"]
    validation = policy["validation"]
    assert isinstance(model_policy, Mapping)
    assert isinstance(inference, Mapping)
    assert isinstance(validation, Mapping)
    if (
        model_spec.name != model_policy.get("name")
        or model_spec.repository != model_policy.get("repository")
        or model_spec.revision != model_policy.get("revision")
        or model_spec.quantization != model_policy.get("quantization")
        or model_spec.num_layers != model_policy.get("num_layers")
        or list(model_spec.candidate_layers) != model_policy.get("candidate_layers")
    ):
        raise DataValidationError("MLX model config differs from runtime policy")
    if (
        hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        != validation.get("preflight_prompt_sha256")
        or float(alpha) != validation.get("preflight_alpha")
    ):
        raise DataValidationError("MLX preflight prompt or alpha differs from policy")
    snapshot_identity = verify_transformers_snapshot(
        model_spec, model_path, manifest_path
    )
    mx, _nn, load, _stream_generate = _mlx_modules()
    seed = inference.get("seed")
    if type(seed) is not int or seed < 0:
        raise ConfigurationError("MLX runtime policy seed is invalid")
    mx.random.seed(seed)
    mx.reset_peak_memory()
    model, tokenizer = load(str(model_path))
    runtime = MlxRuntime(
        model=model,
        tokenizer=tokenizer,
        model_spec=model_spec,
        snapshot=model_path,
        seed=seed,
    )
    try:
        identity = dict(runtime.runtime_identity())
        _validate_runtime_against_policy(policy, identity)
        if (
            identity.get("model_class") != model_policy.get("model_class")
            or identity.get("tokenizer_class") != model_policy.get("tokenizer_class")
            or _hidden_size(model) != model_policy.get("hidden_size")
        ):
            raise DataValidationError("live MLX model architecture differs from runtime policy")
        layer_types = tuple(
            "linear_attention" if layer.is_linear else "full_attention"
            for layer in model.layers
        )
        expected_layer_types = tuple(validation.get("layer_types", ()))
        if layer_types != expected_layer_types:
            raise DataValidationError("live MLX layer-type sequence differs from runtime policy")
        messages = [{"role": "user", "content": prompt}]
        token_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
        )
        rendered = str(
            tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=False,
            )
        )
        if not token_ids or not rendered:
            raise DataValidationError("MLX preflight chat template produced an empty prompt")
        input_ids = mx.array([[int(value) for value in token_ids]])
        checks: dict[str, Any] = {}
        probe_layers = validation.get("probe_layers")
        if not isinstance(probe_layers, Mapping) or set(probe_layers) != {
            "linear_attention",
            "full_attention",
        }:
            raise ConfigurationError("MLX runtime policy probe layers are invalid")
        for layer_type in ("linear_attention", "full_attention"):
            layer = probe_layers[layer_type]
            if type(layer) is not int or not 0 <= layer < len(model.layers):
                raise ConfigurationError("MLX runtime policy probe layer is invalid")
            if layer_types[layer] != layer_type:
                raise DataValidationError("MLX runtime policy probe layer type differs")
            for site in _SITES:
                key = f"{layer_type}.{site.value}"
                checks[key] = _site_check(
                    runtime,
                    input_ids,
                    layer=layer,
                    site=site,
                    alpha=float(alpha),
                    mx=mx,
                )
        if any(value.get("status") != "passed" for value in checks.values()):
            raise DataValidationError("one or more live MLX hook preflight checks failed")
        packages = {name: importlib.metadata.version(name) for name in _PACKAGES}
        body: dict[str, Any] = {
            "schema_version": 1,
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
                "hidden_size": _hidden_size(model),
                "snapshot_identity": dict(snapshot_identity),
            },
            "runtime_identity": identity,
            "software": {
                "python": platform.python_version(),
                "packages": packages,
                "toolchain": dict(mlx_research_toolchain_identity()),
                "source_bindings": dict(validation["source_bindings"]),
            },
            "prompt": {
                "text_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "rendered_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
                "token_ids_sha256": stable_hash([int(value) for value in token_ids]),
                "token_count": len(token_ids),
                "thinking_enabled": False,
            },
            "architecture": {
                "layer_types": list(layer_types),
                "candidate_layer_types": {
                    str(layer): layer_types[layer] for layer in model_spec.candidate_layers
                },
            },
            "intervention": {
                "alpha": float(alpha),
                "direction": "normalized-all-ones-feasibility-only",
                "checks": checks,
            },
            "peak_memory_bytes": int(mx.get_peak_memory()),
        }
        receipt = {**body, "receipt_digest": stable_hash(body)}
        validate_mlx_preflight_receipt(
            receipt,
            project_root=root,
            model_config=config_path,
            snapshot_directory=model_path,
            snapshot_manifest=manifest_path,
            runtime_policy=policy_path,
        )
        _write_once(destination, receipt)
        return receipt
    finally:
        with suppress(Exception):
            runtime.close()


def validate_mlx_preflight_receipt(
    receipt: Mapping[str, Any] | str | Path,
    *,
    project_root: str | Path,
    model_config: str | Path,
    snapshot_directory: str | Path,
    snapshot_manifest: str | Path,
    runtime_policy: str | Path,
) -> Mapping[str, Any]:
    """Replay static bindings and the semantic claims of a frozen preflight receipt."""

    value = (
        _read_json(Path(receipt).absolute(), "MLX preflight receipt")
        if isinstance(receipt, str | Path)
        else dict(receipt)
    )
    if set(value) != _RECEIPT_FIELDS:
        raise DataValidationError("MLX preflight receipt fields differ")
    body = dict(value)
    digest = body.pop("receipt_digest", None)
    if digest != stable_hash(body):
        raise DataValidationError("MLX preflight receipt digest differs")
    if value.get("schema_version") != 1 or value.get("status") != "passed":
        raise DataValidationError("MLX preflight receipt is not a passed schema-1 receipt")
    root = Path(project_root).absolute()
    config_path = Path(model_config).absolute()
    manifest_path = Path(snapshot_manifest).absolute()
    policy_path = Path(runtime_policy).absolute()
    policy = load_mlx_runtime_policy(policy_path)
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
        raise DataValidationError("MLX preflight receipt policy binding differs")
    model_spec = load_model_spec(config_path)
    snapshot = verify_transformers_snapshot(
        model_spec, snapshot_directory, manifest_path
    )
    model = value.get("model")
    runtime_identity = value.get("runtime_identity")
    architecture = value.get("architecture")
    intervention = value.get("intervention")
    software = value.get("software")
    prompt = value.get("prompt")
    validation = policy["validation"]
    model_policy = policy["model"]
    assert isinstance(validation, Mapping)
    assert isinstance(model_policy, Mapping)
    if (
        not isinstance(model, Mapping)
        or set(model) != _MODEL_FIELDS
        or model.get("name") != model_spec.name
        or model.get("repository") != model_spec.repository
        or model.get("revision") != model_spec.revision
        or model.get("quantization") != model_spec.quantization
        or model.get("num_layers") != model_spec.num_layers
        or model.get("hidden_size") != model_policy.get("hidden_size")
        or model.get("snapshot_identity") != snapshot
        or not isinstance(runtime_identity, Mapping)
        or runtime_identity.get("model_class") != model_policy.get("model_class")
        or runtime_identity.get("tokenizer_class") != model_policy.get("tokenizer_class")
        or runtime_identity.get("num_layers") != model_spec.num_layers
        or runtime_identity.get("seed") != policy["inference"].get("seed")
    ):
        raise DataValidationError("MLX preflight receipt model identity differs")
    _validate_runtime_against_policy(policy, runtime_identity)
    toolchain = software.get("toolchain") if isinstance(software, Mapping) else None
    if (
        not isinstance(software, Mapping)
        or set(software) != {"python", "packages", "toolchain", "source_bindings"}
        or software.get("python") != policy["software"].get("python")
        or software.get("packages") != policy["software"].get("packages")
        or software.get("source_bindings") != validation.get("source_bindings")
        or not isinstance(toolchain, Mapping)
        or set(toolchain) != {"xcodebuild", "metal_compiler"}
        or any(
            not isinstance(value, str) or not value.strip()
            for value in toolchain.values()
        )
        or dict(toolchain) != dict(mlx_research_toolchain_identity())
    ):
        raise DataValidationError("MLX preflight receipt software identity differs")
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
        or _SHA256.fullmatch(str(prompt.get("rendered_sha256"))) is None
        or _SHA256.fullmatch(str(prompt.get("token_ids_sha256"))) is None
        or type(prompt.get("token_count")) is not int
        or int(prompt.get("token_count", 0)) <= 1
        or prompt.get("thinking_enabled") is not False
    ):
        raise DataValidationError("MLX preflight receipt prompt identity differs")
    layer_types = validation.get("layer_types")
    if not isinstance(layer_types, list):
        raise ConfigurationError("MLX runtime policy layer types are invalid")
    expected_candidates = {
        str(layer): layer_types[layer] for layer in model_spec.candidate_layers
    }
    if (
        not isinstance(architecture, Mapping)
        or set(architecture) != {"layer_types", "candidate_layer_types"}
        or architecture.get("layer_types") != layer_types
        or architecture.get("candidate_layer_types") != expected_candidates
    ):
        raise DataValidationError("MLX preflight receipt architecture differs")
    checks = intervention.get("checks") if isinstance(intervention, Mapping) else None
    alpha = intervention.get("alpha") if isinstance(intervention, Mapping) else None
    expected_keys = {
        f"{layer_type}.{site.value}"
        for layer_type in ("linear_attention", "full_attention")
        for site in _SITES
    }
    if (
        not isinstance(intervention, Mapping)
        or set(intervention) != {"alpha", "direction", "checks"}
        or isinstance(alpha, bool)
        or not isinstance(alpha, int | float)
        or not math.isfinite(float(alpha))
        or float(alpha) == 0.0
        or float(alpha) != validation.get("preflight_alpha")
        or intervention.get("direction") != "normalized-all-ones-feasibility-only"
        or not isinstance(checks, Mapping)
        or set(checks) != expected_keys
    ):
        raise DataValidationError("MLX preflight receipt intervention evidence differs")
    probe_layers = validation.get("probe_layers")
    if not isinstance(probe_layers, Mapping):
        raise ConfigurationError("MLX runtime policy probe layers are invalid")
    hidden_size = model_policy.get("hidden_size")
    token_count = prompt.get("token_count")
    assert isinstance(checks, Mapping)
    for key, raw_check in checks.items():
        layer_type, site = str(key).split(".", maxsplit=1)
        expected_layer = probe_layers.get(layer_type)
        if not isinstance(raw_check, Mapping):
            raise DataValidationError("MLX preflight receipt check is not a mapping")
        check = raw_check
        zero_errors = (
            check.get("zero_vector_max_abs_logit_error"),
            check.get("cached_zero_max_abs_prefill_logit_error"),
            check.get("cached_zero_max_abs_continuation_logit_error"),
            check.get("prefix_max_abs_activation_error"),
            check.get("final_delta_max_abs_error"),
            check.get("cached_prefix_max_abs_activation_error"),
            check.get("cached_final_delta_max_abs_error"),
        )
        nonzero_changes = (
            check.get("nonzero_max_abs_logit_change"),
            check.get("cached_nonzero_max_abs_prefill_logit_change"),
            check.get("cached_nonzero_max_abs_continuation_logit_change"),
        )
        digest_fields = (
            check.get("captured_final_sha256"),
            check.get("baseline_logits_sha256"),
            check.get("zero_logits_sha256"),
            check.get("steered_logits_sha256"),
        )
        token_fields = (
            check.get("baseline_top_token_id"),
            check.get("zero_top_token_id"),
            check.get("steered_top_token_id"),
            check.get("cached_baseline_token_id"),
            check.get("cached_zero_token_id"),
            check.get("cached_steered_token_id"),
        )
        if (
            set(check) != _CHECK_FIELDS
            or check.get("status") != "passed"
            or check.get("layer") != expected_layer
            or check.get("layer_type") != layer_type
            or check.get("site") != site
            or check.get("capture_shape") != [1, token_count, hidden_size]
            or check.get("cached_capture_shape") != [1, token_count, hidden_size]
            or check.get("alpha") != float(alpha)
            or check.get("module_restored") is not True
            or check.get("applications") != 2
            or check.get("zero_vector_exact_parity") is not True
            or check.get("scope_exact") is not True
            or check.get("nonzero_changed_cached_continuation") is not True
            or any(not _is_exact_zero(error) for error in zero_errors)
            or any(not _is_positive_finite(change) for change in nonzero_changes)
            or any(_SHA256.fullmatch(str(digest)) is None for digest in digest_fields)
            or check.get("baseline_logits_sha256") != check.get("zero_logits_sha256")
            or check.get("baseline_logits_sha256") == check.get("steered_logits_sha256")
            or any(type(token) is not int or int(token) < 0 for token in token_fields)
            or check.get("baseline_top_token_id") != check.get("zero_top_token_id")
            or check.get("cached_baseline_token_id") != check.get("cached_zero_token_id")
        ):
            raise DataValidationError("MLX preflight receipt check evidence differs")
    peak = value.get("peak_memory_bytes")
    if (
        type(peak) is not int
        or int(peak) <= 0
        or int(peak) > int(policy["hardware"].get("unified_memory_bytes", 0))
    ):
        raise DataValidationError("MLX preflight receipt peak memory is invalid")
    return value

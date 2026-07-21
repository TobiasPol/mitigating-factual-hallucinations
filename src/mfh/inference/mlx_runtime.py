"""Native Apple-silicon MLX runtime with explicit activation intervention sites."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import math
import platform
import subprocess
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mfh.contracts import ActivationSite, ModelSpec, PromptSpec, Runtime, TokenScope
from mfh.errors import ConfigurationError, DataValidationError, OptionalDependencyError
from mfh.inference.transformers_snapshot import reject_symlink_path_components

_SHORT_ANSWER_ABBREVIATIONS = (
    "dr.",
    "e.g.",
    "i.e.",
    "jr.",
    "mr.",
    "mrs.",
    "ms.",
    "prof.",
    "sr.",
    "st.",
    "u.k.",
    "u.s.",
    "vs.",
)


def as_numpy(
    value: Any,
    *,
    dtype: Any | None = None,
    copy: bool = True,
) -> np.ndarray[Any, Any]:
    """Materialize an MLX-compatible value as a NumPy array.

    MLX documents ``np.array`` as its NumPy interoperability boundary. Routing
    live tensors through ``np.asarray`` is not reliable across NumPy/MLX
    releases. MLX ``bfloat16`` also has no NumPy representation, so materialize
    it as ``float32`` before crossing that boundary and apply any requested
    NumPy dtype afterwards.
    """

    source = value
    value_type = type(value)
    if value_type.__module__.partition(".")[0] == "mlx":
        try:
            import mlx.core as mx
        except ImportError as exc:  # pragma: no cover - guarded by an MLX value
            raise OptionalDependencyError("cannot convert an MLX array without MLX") from exc
        if getattr(value, "dtype", None) == mx.bfloat16:
            source = value.astype(mx.float32)
    array = np.array(source, copy=copy)
    return array.astype(dtype, copy=False) if dtype is not None else array


def _mlx_modules() -> tuple[Any, Any, Any, Any]:
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import load
        from mlx_lm.generate import stream_generate
    except ImportError as exc:  # pragma: no cover - exercised on non-Apple hosts
        raise OptionalDependencyError(
            "MLX execution requires: uv sync --extra research --extra mlx-macos"
        ) from exc
    return mx, nn, load, stream_generate


def _host_value(command: Sequence[str], context: str) -> str:
    try:
        result = subprocess.run(
            list(command),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ConfigurationError(f"cannot read live MLX host {context}: {exc}") from exc
    value = result.stdout.strip()
    if not value:
        raise ConfigurationError(f"live MLX host {context} is empty")
    return value


def _completed_short_answer(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if "\n" in stripped:
        return True
    if stripped.endswith(("?", "!")):
        return True
    lowered = stripped.casefold()
    return stripped.endswith(".") and not lowered.endswith(_SHORT_ANSWER_ABBREVIATIONS)


@dataclass(frozen=True, slots=True)
class MlxRenderedPrompt:
    text: str
    sha256: str
    token_ids: tuple[int, ...]
    token_ids_sha256: str
    messages: tuple[Mapping[str, str], ...]


@dataclass(frozen=True, slots=True)
class MlxGenerationOutput:
    rendered_prompt: MlxRenderedPrompt
    token_ids: tuple[int, ...]
    text: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    stop_type: str
    stopping_token_id: int | None
    prompt_tokens_per_second: float
    generation_tokens_per_second: float
    peak_memory_bytes: int
    active_memory_bytes: int
    cache_memory_bytes: int


@dataclass(slots=True)
class MlxInterventionState:
    direction: Any | None = None
    alpha: float = 0.0
    token_scope: TokenScope = TokenScope.FINAL_PROMPT
    decay: float = 0.0
    generated_calls: int = 0
    captured: Any | None = None
    intervened: Any | None = None
    applications: int = 0

    def effective_alpha(self, sequence_length: int) -> float:
        if self.direction is None or self.alpha == 0.0:
            return 0.0
        if sequence_length > 1:
            return self.alpha if self.token_scope is TokenScope.FINAL_PROMPT else 0.0
        index = self.generated_calls
        self.generated_calls += 1
        limits = {
            TokenScope.FIRST_GENERATED: 1,
            TokenScope.FIRST_FOUR: 4,
            TokenScope.FIRST_EIGHT: 8,
        }
        if self.token_scope in limits:
            return self.alpha if index < limits[self.token_scope] else 0.0
        if self.token_scope is TokenScope.ALL_GENERATED:
            return self.alpha
        if self.token_scope is TokenScope.EXPONENTIAL_DECAY:
            if self.decay <= 0 or not math.isfinite(self.decay):
                raise ConfigurationError("exponential MLX steering requires positive decay")
            return self.alpha * math.exp(-self.decay * index)
        return 0.0


def _token_digest(token_ids: Sequence[int]) -> str:
    joined = ",".join(str(int(value)) for value in token_ids)
    return hashlib.sha256(joined.encode("ascii")).hexdigest()


def _capture_and_intervene(output: Any, state: MlxInterventionState, mx: Any) -> Any:
    state.captured = output
    generated_activation = bool(getattr(state, "phase_armed", False)) and int(
        getattr(state, "prompt_tokens_remaining", 0)
    ) == 0
    effective_alpha = state.effective_alpha(int(output.shape[1]))
    capture_limit = int(getattr(state, "capture_limit", 0))
    capture_history = getattr(state, "capture_history", None)
    if (
        generated_activation
        and capture_limit > 0
        and isinstance(capture_history, list)
        and len(capture_history) < capture_limit
    ):
        mx.eval(output)
        capture_history.append(
            as_numpy(output[0, -1, :], dtype=np.float32)
        )
    if effective_alpha == 0.0:
        state.intervened = output
        return output
    direction = state.direction
    if direction is None or tuple(direction.shape) != (int(output.shape[-1]),):
        raise DataValidationError("MLX steering direction shape differs from hidden size")
    delta = direction.astype(output.dtype)[None, None, :]
    if int(output.shape[1]) > 1:
        result = mx.concatenate(
            [output[:, :-1, :], output[:, -1:, :] + effective_alpha * delta], axis=1
        )
    else:
        result = output + effective_alpha * delta
    applied_pre_history = getattr(state, "applied_pre_history", None)
    applied_post_history = getattr(state, "applied_post_history", None)
    if isinstance(applied_pre_history, list) and isinstance(applied_post_history, list):
        mx.eval(output, result)
        applied_pre_history.append(
            as_numpy(output[0, -1, :], dtype=np.float32)
        )
        applied_post_history.append(
            as_numpy(result[0, -1, :], dtype=np.float32)
        )
    state.intervened = result
    state.applications += 1
    return result


class MlxRuntime:
    """Exact local MLX model, tokenizer, prompt rendering, and hook adapter."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        model_spec: ModelSpec,
        snapshot: Path,
        seed: int = 17,
    ) -> None:
        if model_spec.runtime is not Runtime.MLX:
            raise ConfigurationError("MlxRuntime requires an MLX ModelSpec")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ConfigurationError("MLX seed must be a non-negative integer")
        layers = getattr(model, "layers", None)
        if layers is None or len(layers) != model_spec.num_layers:
            raise DataValidationError("loaded MLX layer count differs from model config")
        self.model = model
        self.tokenizer = tokenizer
        self.model_spec = model_spec
        self.snapshot = snapshot
        self.seed = seed

    @classmethod
    def from_spec(
        cls,
        model_spec: ModelSpec,
        *,
        snapshot_path: str | Path,
        seed: int = 17,
    ) -> MlxRuntime:
        if model_spec.runtime is not Runtime.MLX:
            raise ConfigurationError("only MLX checkpoints can be loaded here")
        snapshot = reject_symlink_path_components(snapshot_path, "local MLX snapshot")
        if snapshot.is_symlink() or not snapshot.is_dir():
            raise ConfigurationError("local MLX snapshot must be a regular directory")
        mx, _nn, load, _stream_generate = _mlx_modules()
        mx.random.seed(seed)
        model, tokenizer = load(str(snapshot))
        return cls(
            model=model,
            tokenizer=tokenizer,
            model_spec=model_spec,
            snapshot=snapshot,
            seed=seed,
        )

    def render_prompt(
        self,
        prompt: PromptSpec,
        question: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> MlxRenderedPrompt:
        try:
            system = prompt.text.format_map(dict(metadata or {}))
        except KeyError as exc:
            raise ConfigurationError(
                f"prompt {prompt.prompt_id!r} requires missing metadata field {exc.args[0]!r}"
            ) from exc
        messages: tuple[Mapping[str, str], ...] = (
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        )
        options = {
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        token_value = self.tokenizer.apply_chat_template(
            list(messages), tokenize=True, **options
        )
        token_ids = tuple(int(value) for value in token_value)
        if not token_ids:
            raise DataValidationError("MLX chat template produced no prompt tokens")
        text = str(
            self.tokenizer.apply_chat_template(list(messages), tokenize=False, **options)
        )
        if not text:
            raise DataValidationError("MLX chat template produced an empty prompt")
        return MlxRenderedPrompt(
            text=text,
            sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            token_ids=token_ids,
            token_ids_sha256=_token_digest(token_ids),
            messages=messages,
        )

    def generate(
        self,
        rendered: MlxRenderedPrompt,
        *,
        max_new_tokens: int,
    ) -> MlxGenerationOutput:
        if isinstance(max_new_tokens, bool) or not 1 <= max_new_tokens <= 48:
            raise ConfigurationError("MLX generation length must be in [1, 48]")
        mx, _nn, _load, stream_generate = _mlx_modules()
        mx.random.seed(self.seed)
        mx.reset_peak_memory()
        started = time.perf_counter()
        pieces: list[str] = []
        tokens: list[int] = []
        final: Any | None = None
        response_stream = stream_generate(
            self.model,
            self.tokenizer,
            list(rendered.token_ids),
            max_tokens=max_new_tokens,
        )
        short_answer_stop = False
        try:
            for response in response_stream:
                final = response
                pieces.append(str(response.text))
                tokens.append(int(response.token))
                if response.finish_reason is None and _completed_short_answer(
                    "".join(pieces)
                ):
                    short_answer_stop = True
                    break
        finally:
            response_stream.close()
        latency = time.perf_counter() - started
        if final is None:
            raise DataValidationError("MLX generation returned no response")
        text = "".join(pieces)
        return MlxGenerationOutput(
            rendered_prompt=rendered,
            token_ids=tuple(tokens),
            text=text,
            input_tokens=int(final.prompt_tokens),
            output_tokens=int(final.generation_tokens),
            latency_seconds=latency,
            stop_type=("short_answer" if short_answer_stop else str(final.finish_reason)),
            stopping_token_id=tokens[-1] if tokens else None,
            prompt_tokens_per_second=float(final.prompt_tps),
            generation_tokens_per_second=float(final.generation_tps),
            peak_memory_bytes=int(mx.get_peak_memory()),
            active_memory_bytes=int(mx.get_active_memory()),
            cache_memory_bytes=int(mx.get_cache_memory()),
        )

    def runtime_identity(self) -> Mapping[str, Any]:
        _mx, _nn, _load, _stream_generate = _mlx_modules()
        return {
            "backend": "mlx",
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "python": platform.python_version(),
            "machine_model": _host_value(("sysctl", "-n", "hw.model"), "model"),
            "chip": _host_value(
                ("sysctl", "-n", "machdep.cpu.brand_string"), "chip"
            ),
            "unified_memory_bytes": int(
                _host_value(("sysctl", "-n", "hw.memsize"), "memory")
            ),
            "physical_cpu_cores": int(
                _host_value(("sysctl", "-n", "hw.physicalcpu"), "physical cores")
            ),
            "architecture": platform.machine(),
            "os": f"macOS {platform.mac_ver()[0]}",
            "os_build": _host_value(("sw_vers", "-buildVersion"), "OS build"),
            "model_class": f"{type(self.model).__module__}.{type(self.model).__qualname__}",
            "tokenizer_class": (
                f"{type(self.tokenizer).__module__}.{type(self.tokenizer).__qualname__}"
            ),
            "num_layers": len(self.model.layers),
            "seed": self.seed,
        }

    def prompt_activation(
        self,
        rendered: MlxRenderedPrompt,
        *,
        layer: int,
        site: ActivationSite,
    ) -> np.ndarray:
        mx, _nn, _load, _stream_generate = _mlx_modules()
        state = MlxInterventionState()
        with self.intervention(layer=layer, site=site, state=state):
            logits = self.model(mx.array([rendered.token_ids]))
            mx.eval(logits)
        if state.captured is None:
            raise DataValidationError("MLX hook did not capture an activation")
        return as_numpy(state.captured[:, -1, :], dtype=np.float32)

    @contextmanager
    def intervention(
        self,
        *,
        layer: int,
        site: ActivationSite,
        state: MlxInterventionState,
    ) -> Iterator[MlxInterventionState]:
        if isinstance(layer, bool) or not 0 <= layer < len(self.model.layers):
            raise ConfigurationError("MLX intervention layer is out of range")
        mx, nn, _load, _stream_generate = _mlx_modules()
        original_layer = self.model.layers[layer]

        if site is ActivationSite.BLOCK_OUTPUT:
            runtime_state = state

            class BlockWrapper(nn.Module):  # type: ignore[misc, name-defined]
                def __init__(self, block: Any) -> None:
                    super().__init__()
                    self.block = block
                    self.is_linear = block.is_linear

                def __call__(self, *args: Any, **kwargs: Any) -> Any:
                    return _capture_and_intervene(
                        self.block(*args, **kwargs), runtime_state, mx
                    )

            self.model.layers[layer] = BlockWrapper(original_layer)

        elif site is ActivationSite.POST_MLP:
            original_mlp = original_layer.mlp
            runtime_state = state

            class MlpWrapper(nn.Module):  # type: ignore[misc, name-defined]
                def __init__(self, mlp: Any) -> None:
                    super().__init__()
                    self.mlp = mlp

                def __call__(self, *args: Any, **kwargs: Any) -> Any:
                    return _capture_and_intervene(
                        self.mlp(*args, **kwargs), runtime_state, mx
                    )

            original_layer.mlp = MlpWrapper(original_mlp)

        elif site is ActivationSite.POST_ATTENTION:
            runtime_state = state

            class PostAttentionWrapper(nn.Module):  # type: ignore[misc, name-defined]
                def __init__(self, block: Any) -> None:
                    super().__init__()
                    self.block = block
                    self.is_linear = block.is_linear

                def __call__(
                    self, x: Any, mask: Any | None = None, cache: Any | None = None
                ) -> Any:
                    normalized = self.block.input_layernorm(x)
                    if self.is_linear:
                        attention = self.block.linear_attn(normalized, mask, cache)
                    else:
                        attention = self.block.self_attn(normalized, mask, cache)
                    hidden = _capture_and_intervene(x + attention, runtime_state, mx)
                    return hidden + self.block.mlp(
                        self.block.post_attention_layernorm(hidden)
                    )

            self.model.layers[layer] = PostAttentionWrapper(original_layer)
        else:  # pragma: no cover - exhaustive enum guard
            raise ConfigurationError(f"unsupported MLX activation site: {site}")

        try:
            yield state
        finally:
            if site is ActivationSite.POST_MLP:
                original_layer.mlp = original_mlp
            else:
                self.model.layers[layer] = original_layer

    def close(self) -> None:
        mx, _nn, _load, _stream_generate = _mlx_modules()
        mx.synchronize()
        self.model = None
        self.tokenizer = None
        gc.collect()
        mx.clear_cache()

"""Pinned CUDA/vLLM runtime for the NVIDIA Qwen 3.6 ModelOpt checkpoint."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import platform
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from mfh.contracts import ActivationSite, ModelSpec, PromptSpec, Runtime, TokenScope
from mfh.errors import ConfigurationError, DataValidationError, OptionalDependencyError
from mfh.inference.transformers_snapshot import reject_symlink_path_components

_SHORT_ANSWER_ABBREVIATIONS = (
    "dr.", "e.g.", "i.e.", "jr.", "mr.", "mrs.", "ms.", "prof.", "sr.", "st.",
    "u.k.", "u.s.", "vs.",
)


def as_numpy(value: Any, *, dtype: Any | None = None, copy: bool = True) -> np.ndarray[Any, Any]:
    """Materialize a NumPy or PyTorch value without retaining accelerator storage."""

    source = value
    if hasattr(source, "detach"):
        source = source.detach().float().cpu().numpy()
    array = np.array(source, copy=copy)
    return array.astype(dtype, copy=False) if dtype is not None else array


def _vllm_modules() -> tuple[Any, Any]:
    try:
        vllm = importlib.import_module("vllm")
    except ImportError as exc:  # pragma: no cover - exercised on non-CUDA hosts
        raise OptionalDependencyError(
            "CUDA execution requires Linux/x86_64 and: uv sync --extra cuda-a100"
        ) from exc
    return vllm.LLM, vllm.SamplingParams


def _completed_short_answer(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if "\n" in stripped or stripped.endswith(("?", "!")):
        return True
    lowered = stripped.casefold()
    return stripped.endswith(".") and not lowered.endswith(_SHORT_ANSWER_ABBREVIATIONS)


def _token_digest(token_ids: Sequence[int]) -> str:
    joined = ",".join(str(int(value)) for value in token_ids)
    return hashlib.sha256(joined.encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class VllmRenderedPrompt:
    text: str
    sha256: str
    token_ids: tuple[int, ...]
    token_ids_sha256: str
    messages: tuple[Mapping[str, str], ...]


@dataclass(frozen=True, slots=True)
class VllmGenerationOutput:
    rendered_prompt: VllmRenderedPrompt
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
class VllmInterventionState:
    direction: Any | None = None
    alpha: float = 0.0
    token_scope: TokenScope = TokenScope.FINAL_PROMPT
    decay: float = 0.0
    generated_calls: int = 0
    captured: Any | None = None
    intervened: Any | None = None
    applications: int = 0
    prompt_tokens_remaining: int = 0
    phase_armed: bool = False
    capture_limit: int = 0
    capture_history: list[np.ndarray[Any, Any]] = field(default_factory=list)
    applied_pre_history: list[np.ndarray[Any, Any]] = field(default_factory=list)
    applied_post_history: list[np.ndarray[Any, Any]] = field(default_factory=list)

    def arm_prompt(self, prompt_tokens: int) -> None:
        if (
            type(prompt_tokens) is not int
            or prompt_tokens <= 0
            or self.phase_armed
            or self.prompt_tokens_remaining != 0
            or self.generated_calls != 0
            or self.applications != 0
            or self.capture_history
            or self.applied_pre_history
            or self.applied_post_history
        ):
            raise ConfigurationError("vLLM intervention state is not fresh")
        self.prompt_tokens_remaining = prompt_tokens
        self.phase_armed = True


def _state_spec(
    layer: int,
    site: ActivationSite,
    state: VllmInterventionState,
    *,
    prompt_tokens: int,
) -> dict[str, Any]:
    direction = None
    if state.direction is not None:
        values = as_numpy(state.direction, dtype=np.float32)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ConfigurationError("vLLM direction must be one finite vector")
        direction = values.tolist()
    return {
        "layer": layer,
        "site": site.value,
        "direction": direction,
        "alpha": float(state.alpha),
        "token_scope": state.token_scope.value,
        "decay": float(state.decay),
        "prompt_tokens": prompt_tokens,
        "capture_limit": state.capture_limit,
    }


def _sync_state(state: VllmInterventionState, raw: Mapping[str, Any]) -> None:
    captured = raw.get("captured")
    intervened = raw.get("intervened")
    state.captured = (
        None if captured is None else np.asarray(captured, dtype=np.float32)[None, None, :]
    )
    state.intervened = (
        None if intervened is None else np.asarray(intervened, dtype=np.float32)[None, None, :]
    )
    state.capture_history[:] = [np.asarray(row, dtype=np.float32) for row in raw["capture_history"]]
    state.applied_pre_history[:] = [
        np.asarray(row, dtype=np.float32) for row in raw["applied_pre_history"]
    ]
    state.applied_post_history[:] = [
        np.asarray(row, dtype=np.float32) for row in raw["applied_post_history"]
    ]
    state.applications = int(raw["applications"])
    state.generated_calls = max(0, int(raw["tokens_seen"]) - int(raw["prompt_tokens"]))
    state.prompt_tokens_remaining = max(
        0, int(raw["prompt_tokens"]) - int(raw["tokens_seen"])
    )


class VllmRuntime:
    """Single-GPU in-process vLLM engine with documented worker RPC hooks."""

    worker_extension_cls = "mfh.inference.vllm_worker.MfhVllmWorkerExtension"

    def __init__(
        self,
        *,
        engine: Any,
        tokenizer: Any,
        model_spec: ModelSpec,
        snapshot: Path,
        seed: int = 17,
    ) -> None:
        if model_spec.runtime is not Runtime.VLLM:
            raise ConfigurationError("VllmRuntime requires a vLLM ModelSpec")
        if type(seed) is not int or seed < 0:
            raise ConfigurationError("vLLM seed must be a non-negative integer")
        self.engine = engine
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
    ) -> VllmRuntime:
        if model_spec.runtime is not Runtime.VLLM:
            raise ConfigurationError("only vLLM checkpoints can be loaded here")
        snapshot = reject_symlink_path_components(snapshot_path, "local vLLM snapshot")
        if snapshot.is_symlink() or not snapshot.is_dir():
            raise ConfigurationError("local vLLM snapshot must be a regular directory")
        LLM, _SamplingParams = _vllm_modules()
        engine = LLM(
            model=str(snapshot),
            tokenizer=str(snapshot),
            quantization="modelopt_mixed",
            trust_remote_code=False,
            tensor_parallel_size=1,
            max_num_seqs=1,
            max_model_len=4096,
            gpu_memory_utilization=0.90,
            enforce_eager=True,
            enable_prefix_caching=False,
            disable_log_stats=True,
            max_logprobs=-1,
            seed=seed,
            worker_extension_cls=cls.worker_extension_cls,
        )
        return cls(
            engine=engine,
            tokenizer=engine.get_tokenizer(),
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
    ) -> VllmRenderedPrompt:
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
        options = {"add_generation_prompt": True, "enable_thinking": False}
        tokenized = self.tokenizer.apply_chat_template(list(messages), tokenize=True, **options)
        raw_ids = tokenized.get("input_ids") if isinstance(tokenized, Mapping) else tokenized
        if not isinstance(raw_ids, Sequence) or isinstance(raw_ids, (str, bytes)):
            raise DataValidationError("vLLM chat template did not produce an input_ids sequence")
        try:
            token_ids = tuple(int(value) for value in raw_ids)
        except (TypeError, ValueError) as exc:
            raise DataValidationError(
                "vLLM chat template produced non-integer input_ids"
            ) from exc
        text = str(self.tokenizer.apply_chat_template(list(messages), tokenize=False, **options))
        if not token_ids or not text:
            raise DataValidationError("vLLM chat template produced an empty prompt")
        return VllmRenderedPrompt(
            text=text,
            sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            token_ids=token_ids,
            token_ids_sha256=_token_digest(token_ids),
            messages=messages,
        )

    def _collective(self, method: str, *args: Any) -> Any:
        rows = self.engine.collective_rpc(method, args=tuple(args))
        if not isinstance(rows, list) or len(rows) != 1:
            raise DataValidationError("MFH requires exactly one vLLM GPU worker")
        return rows[0]

    def request(
        self,
        token_ids: Sequence[int],
        *,
        max_tokens: int,
        specifications: Sequence[Mapping[str, Any]] = (),
        prompt_logprobs: int | None = None,
        logprobs: int | None = None,
    ) -> tuple[Any, Mapping[str, Any], float]:
        """Execute one request while worker hooks are installed and always remove them."""

        _LLM, SamplingParams = _vllm_modules()
        try:
            self._collective("mfh_install_hooks", list(specifications))
            started = time.perf_counter()
            parameters = SamplingParams(
                temperature=0.0,
                max_tokens=max_tokens,
                seed=self.seed,
                prompt_logprobs=prompt_logprobs,
                logprobs=logprobs,
                detokenize=True,
                skip_special_tokens=True,
            )
            rows = self.engine.generate(
                [{"prompt_token_ids": [int(value) for value in token_ids]}],
                parameters,
                use_tqdm=False,
            )
            if not isinstance(rows, list) or len(rows) != 1 or len(rows[0].outputs) != 1:
                raise DataValidationError("vLLM returned an unexpected request inventory")
            hook_result = self._collective("mfh_collect_hooks")
            return rows[0], hook_result, time.perf_counter() - started
        finally:
            self._collective("mfh_remove_hooks")

    def _truncate_short_answer(self, token_ids: Sequence[int]) -> tuple[tuple[int, ...], str, bool]:
        selected = tuple(int(value) for value in token_ids)
        for length in range(1, len(selected) + 1):
            text = str(self.tokenizer.decode(selected[:length], skip_special_tokens=True))
            if _completed_short_answer(text):
                return selected[:length], text, True
        return selected, str(self.tokenizer.decode(selected, skip_special_tokens=True)), False

    def generation_from_request(
        self,
        rendered: VllmRenderedPrompt,
        request: Any,
        hooks: Mapping[str, Any],
        latency: float,
    ) -> VllmGenerationOutput:
        output = request.outputs[0]
        token_ids, text, short_stop = self._truncate_short_answer(output.token_ids)
        memory = hooks
        return VllmGenerationOutput(
            rendered_prompt=rendered,
            token_ids=token_ids,
            text=text,
            input_tokens=len(rendered.token_ids),
            output_tokens=len(token_ids),
            latency_seconds=latency,
            stop_type="short_answer" if short_stop else str(output.finish_reason),
            stopping_token_id=token_ids[-1] if token_ids else None,
            prompt_tokens_per_second=len(rendered.token_ids) / max(latency, 1e-12),
            generation_tokens_per_second=len(token_ids) / max(latency, 1e-12),
            peak_memory_bytes=int(memory["peak_memory_bytes"]),
            active_memory_bytes=int(memory["active_memory_bytes"]),
            cache_memory_bytes=int(memory["cache_memory_bytes"]),
        )

    def generate(
        self, rendered: VllmRenderedPrompt, *, max_new_tokens: int
    ) -> VllmGenerationOutput:
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 48:
            raise ConfigurationError("vLLM generation length must be in [1, 48]")
        request, hooks, latency = self.request(
            rendered.token_ids, max_tokens=max_new_tokens
        )
        return self.generation_from_request(rendered, request, hooks, latency)

    def generate_with_states(
        self,
        rendered: VllmRenderedPrompt,
        *,
        max_new_tokens: int,
        states: Mapping[tuple[int, ActivationSite], VllmInterventionState],
        prompt_tokens: int | None = None,
        token_ids: Sequence[int] | None = None,
    ) -> tuple[VllmGenerationOutput, Any, Mapping[str, Any]]:
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 48:
            raise ConfigurationError("vLLM generation length must be in [1, 48]")
        active_prompt_tokens = len(rendered.token_ids) if prompt_tokens is None else prompt_tokens
        specs = [
            _state_spec(layer, site, state, prompt_tokens=active_prompt_tokens)
            for (layer, site), state in states.items()
        ]
        request, hooks, latency = self.request(
            rendered.token_ids if token_ids is None else token_ids,
            max_tokens=max_new_tokens,
            specifications=specs,
        )
        raw_specs = hooks["specs"]
        for (layer, site), state in states.items():
            _sync_state(state, raw_specs[f"{layer}:{site.value}"])
        return self.generation_from_request(rendered, request, hooks, latency), request, hooks

    def runtime_identity(self) -> Mapping[str, Any]:
        worker = self._collective("mfh_runtime_identity")
        try:
            driver = subprocess.run(
                ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip().splitlines()[0]
        except (OSError, subprocess.CalledProcessError, IndexError) as exc:
            raise ConfigurationError(f"cannot read NVIDIA driver identity: {exc}") from exc
        identity = {
            "backend": "vllm",
            "vllm": importlib.metadata.version("vllm"),
            "transformers": importlib.metadata.version("transformers"),
            "python": platform.python_version(),
            "architecture": platform.machine(),
            "os": platform.platform(),
            "nvidia_driver": driver,
            "seed": self.seed,
            "tokenizer_class": (
                f"{type(self.tokenizer).__module__}.{type(self.tokenizer).__qualname__}"
            ),
            "tensor_parallel_size": 1,
            "quantization_loader": "modelopt_mixed",
            "quantization_execution": "marlin-w4a16-fp8-weight-only-on-sm80",
            **dict(worker),
        }
        return MappingProxyType(identity)

    def close(self) -> None:
        shutdown = getattr(self.engine, "shutdown", None)
        if callable(shutdown):
            shutdown()
        self.engine = None
        self.tokenizer = None

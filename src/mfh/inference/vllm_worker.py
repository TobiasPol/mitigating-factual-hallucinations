"""vLLM worker extension for exact activation capture and intervention.

The public :class:`vllm.LLM` object owns the scheduler while the model itself
lives in a worker.  vLLM's documented worker-extension and collective-RPC APIs
are therefore the only supported boundary used here.  This module deliberately
contains no vLLM import so it remains importable by the CPU-only test suite.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def _qualified_name(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _walk_attributes(root: Any, path: Sequence[str]) -> Any | None:
    value = root
    for component in path:
        value = getattr(value, component, None)
        if value is None:
            return None
    return value


def _tensor_from_output(output: Any) -> tuple[Any, Any]:
    """Return the hidden tensor and a function rebuilding the module output."""

    if hasattr(output, "shape"):
        return output, lambda replacement: replacement
    if isinstance(output, tuple) and output and hasattr(output[0], "shape"):
        return output[0], lambda replacement: (replacement, *output[1:])
    raise RuntimeError("MFH vLLM hook received an unsupported module output")


class MfhVllmWorkerExtension:
    """Methods injected into a vLLM GPU worker through ``worker_extension_cls``."""

    _mfh_state: Any = None

    def _mfh_model(self) -> Any:
        runner = getattr(self, "model_runner", None)
        model = getattr(runner, "model", None)
        if model is None:
            raise RuntimeError("MFH cannot resolve the live vLLM model runner")
        return model

    def _mfh_layers(self) -> Any:
        model = self._mfh_model()
        candidates = (
            ("language_model", "model", "layers"),
            ("model", "language_model", "layers"),
            ("language_model", "layers"),
            ("model", "model", "layers"),
            ("model", "layers"),
            ("layers",),
        )
        for path in candidates:
            layers = _walk_attributes(model, path)
            if layers is not None and hasattr(layers, "__len__"):
                return layers
        raise RuntimeError("MFH cannot resolve Qwen text decoder layers in vLLM")

    @staticmethod
    def _mfh_validate_spec(raw: Mapping[str, Any], *, hidden_size: int) -> dict[str, Any]:
        scope = raw.get("token_scope")
        valid_scopes = {
            "final_prompt",
            "first_generated",
            "first_four_generated",
            "first_eight_generated",
            "all_generated",
            "exponential_decay",
        }
        prompt_tokens = raw.get("prompt_tokens")
        capture_limit = raw.get("capture_limit", 0)
        alpha = raw.get("alpha", 0.0)
        decay = raw.get("decay", 0.0)
        direction = raw.get("direction")
        if (
            type(prompt_tokens) is not int
            or prompt_tokens <= 0
            or type(capture_limit) is not int
            or capture_limit < 0
            or type(alpha) not in {int, float}
            or not math.isfinite(float(alpha))
            or type(decay) not in {int, float}
            or not math.isfinite(float(decay))
            or float(decay) < 0
            or scope not in valid_scopes
            or (scope == "exponential_decay" and float(decay) <= 0)
        ):
            raise RuntimeError("MFH vLLM hook specification is invalid")
        if direction is None:
            if float(alpha) != 0.0:
                raise RuntimeError("MFH active vLLM hook requires a direction")
            values: list[float] | None = None
        else:
            if (
                not isinstance(direction, list)
                or len(direction) != hidden_size
                or any(type(value) not in {int, float} for value in direction)
                or any(not math.isfinite(float(value)) for value in direction)
            ):
                raise RuntimeError("MFH vLLM hook direction has invalid geometry")
            norm = math.sqrt(sum(float(value) ** 2 for value in direction))
            if not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-6):
                raise RuntimeError("MFH vLLM hook direction is not unit norm")
            values = [float(value) for value in direction]
        return {
            "prompt_tokens": prompt_tokens,
            "capture_limit": capture_limit,
            "alpha": float(alpha),
            "decay": float(decay),
            "token_scope": scope,
            "direction": values,
            "tokens_seen": 0,
            "captured": None,
            "intervened": None,
            "capture_history": [],
            "applied_pre_history": [],
            "applied_post_history": [],
            "applications": 0,
            "residual": None,
        }

    @staticmethod
    def _mfh_alpha(state: Mapping[str, Any], global_index: int) -> float:
        alpha = float(state["alpha"])
        if state["direction"] is None or alpha == 0.0:
            return 0.0
        prompt_tokens = int(state["prompt_tokens"])
        scope = str(state["token_scope"])
        if global_index < prompt_tokens:
            return alpha if scope == "final_prompt" and global_index == prompt_tokens - 1 else 0.0
        generated_index = global_index - prompt_tokens
        limits = {
            "first_generated": 1,
            "first_four_generated": 4,
            "first_eight_generated": 8,
        }
        if scope in limits:
            return alpha if generated_index < limits[scope] else 0.0
        if scope == "all_generated":
            return alpha
        if scope == "exponential_decay":
            return alpha * math.exp(-float(state["decay"]) * generated_index)
        return 0.0

    @staticmethod
    def _mfh_rows(tensor: Any) -> tuple[Any, tuple[int, ...]]:
        shape = tuple(int(value) for value in tensor.shape)
        if len(shape) == 2:
            return tensor, shape
        if len(shape) == 3 and shape[0] == 1:
            return tensor.reshape(shape[1], shape[2]), shape
        raise RuntimeError(f"MFH vLLM hook requires one unbatched sequence, got {shape}")

    @staticmethod
    def _mfh_cpu_row(row: Any) -> list[float]:
        values = row.detach().float().cpu().tolist()
        return [float(value) for value in values]

    def _mfh_process(self, key: str, tensor: Any) -> Any:
        import torch

        state: dict[str, Any] = self._mfh_state["specs"][key]
        rows, original_shape = self._mfh_rows(tensor)
        count = int(rows.shape[0])
        start = int(state["tokens_seen"])
        stop = start + count
        state["tokens_seen"] = stop
        prompt_tokens = int(state["prompt_tokens"])

        final_prompt = prompt_tokens - 1
        if start <= final_prompt < stop:
            state["captured"] = self._mfh_cpu_row(rows[final_prompt - start])
        first_generated = max(start, prompt_tokens)
        capture_stop = min(stop, prompt_tokens + int(state["capture_limit"]))
        for global_index in range(first_generated, capture_stop):
            state["capture_history"].append(
                self._mfh_cpu_row(rows[global_index - start])
            )

        edits: list[tuple[int, float]] = []
        for global_index in range(start, stop):
            effective = self._mfh_alpha(state, global_index)
            if effective != 0.0:
                edits.append((global_index - start, effective))
        if not edits:
            state["intervened"] = state["captured"]
            return tensor

        result = rows.clone()
        direction = torch.as_tensor(
            state["direction"], device=result.device, dtype=result.dtype
        )
        for local_index, effective in edits:
            before = result[local_index].clone()
            result[local_index].add_(direction, alpha=effective)
            state["applied_pre_history"].append(self._mfh_cpu_row(before))
            state["applied_post_history"].append(
                self._mfh_cpu_row(result[local_index])
            )
            state["applications"] += 1
            if start + local_index == final_prompt:
                state["intervened"] = self._mfh_cpu_row(result[local_index])
        return result.reshape(original_shape)

    def mfh_install_hooks(self, specifications: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Install fresh hooks for exactly one single-sequence vLLM request."""

        if getattr(self, "_mfh_state", None) is not None:
            raise RuntimeError("MFH vLLM hooks are already installed")
        layers = self._mfh_layers()
        model = self._mfh_model()
        config = getattr(model, "config", None)
        hidden_size = getattr(config, "hidden_size", None)
        if type(hidden_size) is not int:
            hidden_size = getattr(getattr(config, "text_config", None), "hidden_size", None)
        if type(hidden_size) is not int or hidden_size <= 0:
            raise RuntimeError("MFH cannot resolve the live Qwen hidden size")

        specs: dict[str, dict[str, Any]] = {}
        for raw in specifications:
            layer_index = raw.get("layer")
            site = raw.get("site")
            if (
                type(layer_index) is not int
                or not 0 <= layer_index < len(layers)
                or site not in {"post_attention", "post_mlp", "block_output"}
            ):
                raise RuntimeError("MFH vLLM hook layer or site is invalid")
            key = f"{layer_index}:{site}"
            if key in specs:
                raise RuntimeError("MFH vLLM hook keys must be unique")
            specs[key] = self._mfh_validate_spec(raw, hidden_size=hidden_size)

        self._mfh_state = {"specs": specs, "handles": []}
        handles: list[Any] = self._mfh_state["handles"]
        for key in sorted(specs):
            layer_text, site = key.split(":", 1)
            layer = layers[int(layer_text)]
            if site == "block_output":
                def block_hook(
                    _module: Any, _inputs: Any, output: Any, *, hook_key: str = key
                ) -> Any:
                    if (
                        not isinstance(output, tuple)
                        or len(output) != 2
                        or not hasattr(output[0], "shape")
                        or not hasattr(output[1], "shape")
                        or tuple(output[0].shape) != tuple(output[1].shape)
                    ):
                        raise RuntimeError(
                            "MFH Qwen block hook requires (hidden_states, residual)"
                        )
                    hidden_states, residual = output
                    block_output = hidden_states + residual
                    steered = self._mfh_process(hook_key, block_output)
                    return hidden_states + (steered - block_output), residual

                handles.append(layer.register_forward_hook(block_hook))
            elif site == "post_mlp":
                mlp = getattr(layer, "mlp", None)
                if mlp is None:
                    raise RuntimeError("MFH cannot resolve the live Qwen MLP module")

                def mlp_hook(
                    _module: Any, _inputs: Any, output: Any, *, hook_key: str = key
                ) -> Any:
                    tensor, rebuild = _tensor_from_output(output)
                    return rebuild(self._mfh_process(hook_key, tensor))

                handles.append(mlp.register_forward_hook(mlp_hook))
            else:
                post_attention_norm = getattr(layer, "post_attention_layernorm", None)
                if post_attention_norm is None:
                    raise RuntimeError(
                        "MFH cannot resolve the Qwen post-attention layer norm"
                    )

                def post_attention_pre_hook(
                    _module: Any, inputs: Any, *, hook_key: str = key
                ) -> Any:
                    if (
                        len(inputs) != 2
                        or not hasattr(inputs[0], "shape")
                        or not hasattr(inputs[1], "shape")
                        or tuple(inputs[0].shape) != tuple(inputs[1].shape)
                    ):
                        raise RuntimeError(
                            "MFH post-attention hook requires hidden and residual tensors"
                        )
                    hidden_states, residual = inputs
                    post_attention = hidden_states + residual
                    steered = self._mfh_process(hook_key, post_attention)
                    return hidden_states + (steered - post_attention), residual

                handles.append(
                    post_attention_norm.register_forward_pre_hook(
                        post_attention_pre_hook
                    )
                )
        return {"status": "installed", "hook_count": len(specs)}

    def mfh_collect_hooks(self) -> dict[str, Any]:
        state = getattr(self, "_mfh_state", None)
        if state is None:
            raise RuntimeError("MFH vLLM hooks are not installed")
        import torch

        return {
            "specs": {
                key: {
                    name: value
                    for name, value in spec.items()
                    if name not in {"direction", "residual"}
                }
                for key, spec in state["specs"].items()
            },
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "active_memory_bytes": int(torch.cuda.memory_allocated()),
            "cache_memory_bytes": int(torch.cuda.memory_reserved()),
        }

    def mfh_remove_hooks(self) -> dict[str, Any]:
        state = getattr(self, "_mfh_state", None)
        if state is None:
            return {"status": "absent", "removed": 0}
        handles = list(state["handles"])
        for handle in reversed(handles):
            handle.remove()
        self._mfh_state = None
        return {"status": "removed", "removed": len(handles)}

    def mfh_runtime_identity(self) -> dict[str, Any]:
        import torch

        model = self._mfh_model()
        layers = self._mfh_layers()
        language_model = getattr(model, "language_model", model)
        quant_config = getattr(language_model, "quant_config", None)
        get_quantization_name = getattr(quant_config, "get_name", None)
        if quant_config is None or not callable(get_quantization_name):
            raise RuntimeError("MFH cannot resolve the live vLLM quantization config")
        quantization_loader = str(get_quantization_name())
        if quantization_loader != "modelopt_mixed":
            raise RuntimeError("MFH requires the vLLM ModelOpt mixed loader")
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        config = getattr(model, "config", None)
        hidden_size = getattr(config, "hidden_size", None)
        if type(hidden_size) is not int:
            hidden_size = getattr(getattr(config, "text_config", None), "hidden_size", None)
        if type(hidden_size) is not int or hidden_size <= 0:
            raise RuntimeError("MFH cannot resolve the live Qwen hidden size")
        return {
            "model_class": _qualified_name(model),
            "num_layers": len(layers),
            "hidden_size": hidden_size,
            "gpu_name": properties.name,
            "gpu_total_memory_bytes": int(properties.total_memory),
            "cuda_capability": f"{properties.major}.{properties.minor}",
            "cuda_runtime": str(torch.version.cuda),
            "torch": str(torch.__version__),
            "quantization_config_class": _qualified_name(quant_config),
            "quantization_loader": quantization_loader,
        }

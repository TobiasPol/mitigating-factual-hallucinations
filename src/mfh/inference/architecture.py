"""Architecture-aware resolution of scientifically defined hook sites."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from torch import nn

from mfh.contracts import ActivationSite
from mfh.errors import ConfigurationError


class HookMode(StrEnum):
    PRE = "pre"
    POST = "post"


@dataclass(frozen=True, slots=True, order=True)
class HookKey:
    layer: int
    site: ActivationSite

    @property
    def artifact_key(self) -> str:
        return f"layer_{self.layer:03d}.{self.site.value}"


@dataclass(frozen=True, slots=True)
class HookPoint:
    key: HookKey
    module: nn.Module
    mode: HookMode
    module_path: str


_LAYER_PATHS = (
    "model.layers",
    "model.language_model.layers",
    "language_model.layers",
    "model.model.layers",
    "transformer.h",
    "gpt_neox.layers",
)


def _resolve_attribute(root: Any, path: str) -> Any | None:
    current = root
    for component in path.split("."):
        if not hasattr(current, component):
            return None
        current = getattr(current, component)
    return current


def resolve_decoder_layers(model: nn.Module, *, expected_count: int) -> tuple[str, nn.ModuleList]:
    """Find the text decoder and reject ambiguous multimodal module lists."""

    for path in _LAYER_PATHS:
        value = _resolve_attribute(model, path)
        if isinstance(value, nn.ModuleList) and len(value) == expected_count:
            return path, value

    candidates: list[tuple[str, nn.ModuleList]] = []
    for path, module in model.named_modules():
        if not isinstance(module, nn.ModuleList) or len(module) != expected_count:
            continue
        if all(hasattr(layer, "mlp") for layer in module):
            candidates.append((path, module))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ConfigurationError(
            f"could not find a {expected_count}-layer text decoder in {type(model).__name__}"
        )
    raise ConfigurationError(
        "multiple decoder-like layer lists match the configured layer count: "
        f"{[path for path, _ in candidates]}"
    )


def _resolve_site(block: nn.Module, site: ActivationSite) -> tuple[nn.Module, HookMode, str]:
    if site is ActivationSite.BLOCK_OUTPUT:
        return block, HookMode.POST, ""
    if site is ActivationSite.POST_ATTENTION:
        # Gemma 4 normalizes the attention delta before adding the residual, so
        # pre_feedforward_layernorm is the first stable point containing the
        # post-attention residual. Qwen exposes that residual as the input to
        # post_attention_layernorm.
        for name in ("pre_feedforward_layernorm", "post_attention_layernorm"):
            module = getattr(block, name, None)
            if isinstance(module, nn.Module):
                return module, HookMode.PRE, name
        raise ConfigurationError(
            f"{type(block).__name__} has no supported post-attention residual site"
        )
    if site is ActivationSite.POST_MLP:
        # Gemma's post_feedforward norm follows the combined dense/MoE output;
        # Qwen's MLP/MoE module output is directly added to the residual.
        for name in ("post_feedforward_layernorm", "mlp"):
            module = getattr(block, name, None)
            if isinstance(module, nn.Module):
                return module, HookMode.POST, name
        raise ConfigurationError(f"{type(block).__name__} has no supported FFN/MoE output site")
    raise ConfigurationError(f"unsupported activation site: {site}")


def resolve_hook_points(
    model: nn.Module,
    *,
    expected_layers: int,
    layers: tuple[int, ...],
    sites: tuple[ActivationSite, ...],
) -> tuple[HookPoint, ...]:
    if not layers:
        raise ConfigurationError("at least one hook layer is required")
    if not sites:
        raise ConfigurationError("at least one activation site is required")
    if len(set(layers)) != len(layers) or len(set(sites)) != len(sites):
        raise ConfigurationError("hook layers and sites must be unique")
    decoder_path, decoder_layers = resolve_decoder_layers(model, expected_count=expected_layers)
    points: list[HookPoint] = []
    for layer_index in layers:
        if layer_index < 0 or layer_index >= len(decoder_layers):
            raise ConfigurationError(
                f"hook layer {layer_index} is outside [0, {len(decoder_layers) - 1}]"
            )
        block = decoder_layers[layer_index]
        for site in sites:
            module, mode, relative_path = _resolve_site(block, site)
            suffix = f".{relative_path}" if relative_path else ""
            points.append(
                HookPoint(
                    key=HookKey(layer_index, site),
                    module=module,
                    mode=mode,
                    module_path=f"{decoder_path}.{layer_index}{suffix}",
                )
            )
    return tuple(points)

"""Combined capture/steering hooks with explicit token-position semantics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch
from torch import Tensor
from torch.utils.hooks import RemovableHandle

from mfh.contracts import TokenScope
from mfh.errors import ConfigurationError
from mfh.inference.architecture import HookKey, HookMode, HookPoint


class PassPhase(StrEnum):
    PROMPT = "prompt"
    GENERATED = "generated"
    TEACHER_FORCED = "teacher_forced"


class CapturePolicy(StrEnum):
    NONE = "none"
    PROMPT_FINAL = "prompt_final"
    RESPONSE_TOKENS = "response_tokens"
    TOKEN_SCOPE = "token_scope"


@dataclass(frozen=True, slots=True)
class InterventionPlan:
    direction: Tensor
    alpha: float
    token_scope: TokenScope
    rms_relative: bool = True
    decay: float = 0.5

    def __post_init__(self) -> None:
        if self.direction.ndim != 1 or self.direction.numel() == 0:
            raise ConfigurationError("steering direction must be a non-empty rank-1 tensor")
        if not math.isfinite(self.alpha):
            raise ConfigurationError("steering alpha must be finite")
        if self.decay < 0 or not math.isfinite(self.decay):
            raise ConfigurationError("steering decay must be finite and non-negative")


@dataclass(slots=True)
class HookState:
    phase: PassPhase = PassPhase.PROMPT
    prompt_length: int = 0
    response_start: int | None = None
    generation_step: int = 0


def selection_weights(
    scope: TokenScope,
    state: HookState,
    sequence_length: int,
    *,
    decay: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return per-position multipliers for literal activation positions.

    `FINAL_PROMPT` and `FIRST_GENERATED` are intentionally distinct: the former
    is the last activation in the prompt pass; the latter is the activation of
    the first sampled token when it is fed back into the decoder.
    """

    weights = torch.zeros(sequence_length, device=device, dtype=dtype)
    if sequence_length == 0:
        return weights
    if state.phase is PassPhase.PROMPT:
        if scope is TokenScope.FINAL_PROMPT:
            weights[-1] = 1
        return weights
    if state.phase is PassPhase.GENERATED:
        step = state.generation_step
        active = False
        multiplier = 1.0
        if scope is TokenScope.FIRST_GENERATED:
            active = step == 0
        elif scope is TokenScope.FIRST_FOUR:
            active = step < 4
        elif scope is TokenScope.FIRST_EIGHT:
            active = step < 8
        elif scope is TokenScope.ALL_GENERATED:
            active = True
        elif scope is TokenScope.EXPONENTIAL_DECAY:
            active = True
            multiplier = math.exp(-decay * step)
        if active:
            weights[:] = multiplier
        return weights

    if state.response_start is None:
        raise ConfigurationError("teacher-forced hooks require response_start")
    start = state.response_start
    if start < 1 or start > sequence_length:
        raise ConfigurationError(
            f"response_start {start} is invalid for sequence length {sequence_length}"
        )
    if scope is TokenScope.FINAL_PROMPT:
        weights[start - 1] = 1
    elif scope is TokenScope.FIRST_GENERATED and start < sequence_length:
        weights[start] = 1
    elif scope in {TokenScope.FIRST_FOUR, TokenScope.FIRST_EIGHT}:
        count = 4 if scope is TokenScope.FIRST_FOUR else 8
        weights[start : min(start + count, sequence_length)] = 1
    elif scope is TokenScope.ALL_GENERATED:
        weights[start:] = 1
    elif scope is TokenScope.EXPONENTIAL_DECAY:
        offsets = torch.arange(sequence_length - start, device=device, dtype=dtype)
        weights[start:] = torch.exp(-decay * offsets)
    return weights


def _hidden_tensor(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], Tensor):
        return value[0]
    raise ConfigurationError(
        f"hook output must be a tensor or tensor-first tuple/list, got {type(value).__name__}"
    )


def _replace_hidden(value: Any, hidden: Tensor) -> Any:
    if isinstance(value, Tensor):
        return hidden
    if isinstance(value, tuple):
        return (hidden, *value[1:])
    if isinstance(value, list):
        return [hidden, *value[1:]]
    raise ConfigurationError(f"cannot replace hidden tensor in {type(value).__name__}")


class ActivationSession:
    """Register one deterministic hook per point and capture pre-intervention values."""

    def __init__(
        self,
        points: tuple[HookPoint, ...],
        *,
        interventions: dict[HookKey, InterventionPlan] | None = None,
        capture_policy: CapturePolicy = CapturePolicy.NONE,
        capture_scope: TokenScope | None = None,
        capture_dtype: torch.dtype = torch.float32,
    ) -> None:
        if not points:
            raise ConfigurationError("activation session requires at least one hook point")
        keys = [point.key for point in points]
        if len(set(keys)) != len(keys):
            raise ConfigurationError("activation session hook keys must be unique")
        self.points = points
        self.interventions = dict(interventions or {})
        unknown = set(self.interventions) - set(keys)
        if unknown:
            raise ConfigurationError(f"interventions reference unresolved hooks: {sorted(unknown)}")
        if capture_policy is CapturePolicy.TOKEN_SCOPE and capture_scope is None:
            raise ConfigurationError("TOKEN_SCOPE capture requires capture_scope")
        self.capture_policy = capture_policy
        self.capture_scope = capture_scope
        self.capture_dtype = capture_dtype
        self.state = HookState()
        self._handles: list[RemovableHandle] = []
        self._captures: dict[HookKey, list[Tensor]] = {key: [] for key in keys}

    def __enter__(self) -> ActivationSession:
        if self._handles:
            raise RuntimeError("activation session is already active")
        for point in self.points:
            if point.mode is HookMode.PRE:
                handle = point.module.register_forward_pre_hook(
                    self._make_pre_hook(point.key), with_kwargs=True
                )
            else:
                handle = point.module.register_forward_hook(
                    self._make_post_hook(point.key), with_kwargs=True
                )
            self._handles.append(handle)
        return self

    def __exit__(self, *_: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def set_prompt(self, prompt_length: int) -> None:
        if prompt_length <= 0:
            raise ConfigurationError("prompt_length must be positive")
        self.state = HookState(phase=PassPhase.PROMPT, prompt_length=prompt_length)

    def set_generated(self, *, prompt_length: int, generation_step: int) -> None:
        if prompt_length <= 0 or generation_step < 0:
            raise ConfigurationError("invalid generated-pass hook state")
        self.state = HookState(
            phase=PassPhase.GENERATED,
            prompt_length=prompt_length,
            generation_step=generation_step,
        )

    def set_teacher_forced(self, *, prompt_length: int, response_start: int) -> None:
        if prompt_length <= 0 or response_start != prompt_length:
            raise ConfigurationError("teacher-forced response_start must equal prompt_length")
        self.state = HookState(
            phase=PassPhase.TEACHER_FORCED,
            prompt_length=prompt_length,
            response_start=response_start,
        )

    def clear(self) -> None:
        for values in self._captures.values():
            values.clear()

    def activations(self) -> dict[HookKey, Tensor]:
        result: dict[HookKey, Tensor] = {}
        for key, values in self._captures.items():
            if values:
                result[key] = torch.cat(values, dim=0)
        return result

    def _capture_weights(self, hidden: Tensor) -> Tensor:
        length = hidden.shape[-2]
        weights = torch.zeros(length, device=hidden.device, dtype=hidden.dtype)
        if self.capture_policy is CapturePolicy.NONE:
            return weights
        if self.capture_policy is CapturePolicy.PROMPT_FINAL:
            if self.state.phase is PassPhase.PROMPT:
                weights[-1] = 1
            return weights
        if self.capture_policy is CapturePolicy.RESPONSE_TOKENS:
            if self.state.phase is PassPhase.TEACHER_FORCED:
                if self.state.response_start is None:
                    raise ConfigurationError("response capture requires response_start")
                weights[self.state.response_start :] = 1
            return weights
        if self.capture_scope is None:
            raise ConfigurationError("token-scope capture has no scope")
        return selection_weights(
            self.capture_scope,
            self.state,
            length,
            decay=0.5,
            device=hidden.device,
            dtype=hidden.dtype,
        )

    def _transform(self, key: HookKey, hidden: Tensor) -> Tensor:
        if hidden.ndim != 3:
            raise ConfigurationError(
                f"hook {key.artifact_key} expected [batch, sequence, hidden], "
                f"got {tuple(hidden.shape)}"
            )
        capture_weights = self._capture_weights(hidden)
        selected = capture_weights != 0
        if selected.any():
            captured = hidden[:, selected, :].reshape(-1, hidden.shape[-1])
            self._captures[key].append(captured.detach().to("cpu", dtype=self.capture_dtype))

        plan = self.interventions.get(key)
        if plan is None or plan.alpha == 0:
            return hidden
        weights = selection_weights(
            plan.token_scope,
            self.state,
            hidden.shape[-2],
            decay=plan.decay,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        if not (weights != 0).any():
            return hidden
        direction = plan.direction.to(device=hidden.device, dtype=hidden.dtype)
        if direction.shape[0] != hidden.shape[-1]:
            raise ConfigurationError(
                f"steering direction width {direction.shape[0]} does not match hidden width "
                f"{hidden.shape[-1]} at {key.artifact_key}"
            )
        multiplier = weights.view(1, -1, 1) * plan.alpha
        if plan.rms_relative:
            scale = hidden.float().pow(2).mean(dim=-1, keepdim=True).sqrt().to(hidden.dtype)
            multiplier = multiplier * scale
        return hidden + multiplier * direction.view(1, 1, -1)

    def _make_pre_hook(self, key: HookKey) -> Any:
        def hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            if args:
                hidden = _hidden_tensor(args[0])
                return (_replace_hidden(args, self._transform(key, hidden)), kwargs)
            keyword_hidden = kwargs.get("hidden_states")
            if not isinstance(keyword_hidden, Tensor):
                raise ConfigurationError(f"pre-hook {key.artifact_key} received no hidden tensor")
            updated = dict(kwargs)
            updated["hidden_states"] = self._transform(key, keyword_hidden)
            return args, updated

        return hook

    def _make_post_hook(self, key: HookKey) -> Any:
        def hook(_module: Any, _args: tuple[Any, ...], _kwargs: dict[str, Any], output: Any) -> Any:
            hidden = _hidden_tensor(output)
            return _replace_hidden(output, self._transform(key, hidden))

        return hook

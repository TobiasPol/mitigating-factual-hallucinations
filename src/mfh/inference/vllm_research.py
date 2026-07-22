"""Research-only vLLM capture extensions layered over the frozen E0/E1 runtime."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from mfh.contracts import ActivationSite, ModelSpec, PromptSpec, TokenScope
from mfh.errors import ConfigurationError, DataValidationError
from mfh.inference.vllm_runtime import (
    VllmGenerationOutput,
    VllmInterventionState,
    VllmRenderedPrompt,
    VllmRuntime,
    _state_spec,
    _sync_state,
    as_numpy,
)
from mfh.provenance import sha256_path


@dataclass(slots=True)
class VllmResearchInterventionState(VllmInterventionState):
    """Fresh state synchronized from the single vLLM worker after a request."""


@dataclass(frozen=True, slots=True)
class VllmOnlinePrefixCapture:
    """Generated-token activations from the first pass of deterministic replay."""

    text: str
    token_ids: tuple[int, ...]
    activations: Mapping[ActivationSite, Mapping[int, np.ndarray[Any, Any]]]
    feature_token_count: int


@dataclass(frozen=True, slots=True)
class VllmOnlineGenerationOutput:
    generation: VllmGenerationOutput
    early_gate_applied: bool
    continued_after_early_gate: bool
    feature_token_count: int
    buffered_token_count_at_gate: int


def _token_ids_digest(token_ids: Sequence[int]) -> str:
    serialized = ",".join(str(value) for value in token_ids)
    return hashlib.sha256(serialized.encode("ascii")).hexdigest()


def _validate_fresh_intervention_state(
    state: VllmInterventionState, *, expected_width: int | None
) -> None:
    if (
        state.generated_calls != 0
        or state.applications != 0
        or state.captured is not None
        or state.intervened is not None
        or (
            isinstance(state, VllmResearchInterventionState)
            and (
                type(state.capture_limit) is not int
                or state.capture_limit < 0
                or bool(state.capture_history)
            )
        )
        or isinstance(state.alpha, bool)
        or not isinstance(state.alpha, int | float)
        or not math.isfinite(float(state.alpha))
        or not isinstance(state.token_scope, TokenScope)
        or isinstance(state.decay, bool)
        or not isinstance(state.decay, int | float)
        or not math.isfinite(float(state.decay))
        or float(state.decay) < 0
        or (state.token_scope is TokenScope.EXPONENTIAL_DECAY and float(state.decay) <= 0)
    ):
        raise ConfigurationError("vLLM intervention state must be fresh and finite")
    if state.direction is None:
        if float(state.alpha) != 0:
            raise ConfigurationError("active vLLM intervention requires a direction")
        return
    try:
        direction = as_numpy(state.direction, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"vLLM intervention direction is invalid: {exc}") from exc
    norm = float(np.linalg.norm(direction)) if direction.ndim == 1 else math.nan
    if (
        direction.ndim != 1
        or direction.size == 0
        or not np.isfinite(direction).all()
        or not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-6)
        or (expected_width is not None and direction.size != expected_width)
    ):
        raise ConfigurationError("vLLM intervention direction geometry is invalid")


def vllm_research_toolchain_identity() -> Mapping[str, str]:
    """Read the pinned CUDA research software identities."""

    values = {
        name: importlib.metadata.version(distribution)
        for name, distribution in {
            "vllm": "vllm",
            "torch": "torch",
            "transformers": "transformers",
            "numpy": "numpy",
        }.items()
    }
    try:
        result = subprocess.run(
            ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ConfigurationError(f"cannot read live NVIDIA driver identity: {exc}") from exc
    values["nvidia_driver"] = result.stdout.strip().splitlines()[0]
    return MappingProxyType(values)


@dataclass(frozen=True, slots=True)
class VllmPromptFeatureOutput:
    """Prompt-end activations and trivial next-token confidence baselines."""

    activations: Mapping[int, np.ndarray[Any, Any]]
    maximum_token_probability: float
    output_entropy: float

    def __post_init__(self) -> None:
        values: dict[int, np.ndarray[Any, Any]] = {}
        for raw_layer, raw_activation in self.activations.items():
            if type(raw_layer) is not int:
                raise DataValidationError(
                    "vLLM prompt activation layer identifiers must be exact integers"
                )
            layer = raw_layer
            activation = as_numpy(raw_activation, dtype=np.float32, copy=True)
            if (
                layer < 0
                or activation.ndim != 2
                or activation.shape[0] != 1
                or activation.shape[1] == 0
                or not np.isfinite(activation).all()
            ):
                raise DataValidationError("vLLM prompt activation bundle is invalid")
            activation.setflags(write=False)
            values[layer] = activation
        if not values or len(values) != len(self.activations):
            raise DataValidationError("vLLM prompt activation layers must be unique")
        maximum = float(self.maximum_token_probability)
        entropy = float(self.output_entropy)
        if not math.isfinite(maximum) or not 0 < maximum <= 1:
            raise DataValidationError("vLLM maximum-token probability is invalid")
        if not math.isfinite(entropy) or entropy < 0:
            raise DataValidationError("vLLM output entropy is invalid")
        object.__setattr__(self, "activations", MappingProxyType(values))
        object.__setattr__(self, "maximum_token_probability", maximum)
        object.__setattr__(self, "output_entropy", entropy)


@dataclass(frozen=True, slots=True)
class VllmPromptFeatureCubeOutput:
    """One-forward prompt activations indexed by site and layer."""

    activations: Mapping[ActivationSite, Mapping[int, np.ndarray[Any, Any]]]
    maximum_token_probability: float
    output_entropy: float
    peak_memory_bytes: int

    def __post_init__(self) -> None:
        sites: dict[ActivationSite, Mapping[int, np.ndarray[Any, Any]]] = {}
        layer_identity: tuple[int, ...] | None = None
        for raw_site, raw_values in self.activations.items():
            if not isinstance(raw_site, ActivationSite) or not isinstance(raw_values, Mapping):
                raise DataValidationError("vLLM prompt cube site identity is invalid")
            output = VllmPromptFeatureOutput(
                activations=raw_values,
                maximum_token_probability=self.maximum_token_probability,
                output_entropy=self.output_entropy,
            )
            current = tuple(output.activations)
            if layer_identity is not None and current != layer_identity:
                raise DataValidationError("vLLM prompt cube layers differ between sites")
            layer_identity = current
            sites[raw_site] = output.activations
        if not sites or len(sites) != len(self.activations):
            raise DataValidationError("vLLM prompt cube sites must be non-empty and unique")
        maximum = float(self.maximum_token_probability)
        entropy = float(self.output_entropy)
        if not math.isfinite(maximum) or not 0 < maximum <= 1:
            raise DataValidationError("vLLM prompt cube maximum-token probability is invalid")
        if not math.isfinite(entropy) or entropy < 0:
            raise DataValidationError("vLLM prompt cube entropy is invalid")
        if type(self.peak_memory_bytes) is not int or self.peak_memory_bytes < 0:
            raise DataValidationError("vLLM prompt cube peak memory is invalid")
        object.__setattr__(self, "activations", MappingProxyType(sites))
        object.__setattr__(self, "maximum_token_probability", maximum)
        object.__setattr__(self, "output_entropy", entropy)


@dataclass(frozen=True, slots=True)
class VllmTeacherForcedOutput:
    """Exact continuation likelihood and response-token activation trajectory."""

    response_text_sha256: str
    response_token_ids: tuple[int, ...]
    response_token_ids_sha256: str
    token_log_probabilities: tuple[float, ...]
    negative_log_likelihood: float
    mean_negative_log_likelihood: float
    perplexity: float
    activations: Mapping[int, np.ndarray[Any, Any]]
    peak_memory_bytes: int = 0

    def __post_init__(self) -> None:
        response_token_ids = tuple(self.response_token_ids)
        if (
            not isinstance(self.response_text_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.response_text_sha256) is None
            or not isinstance(self.response_token_ids_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.response_token_ids_sha256) is None
            or not response_token_ids
            or any(type(value) is not int or value < 0 for value in response_token_ids)
            or self.response_token_ids_sha256 != _token_ids_digest(response_token_ids)
            or type(self.peak_memory_bytes) is not int
            or self.peak_memory_bytes < 0
        ):
            raise DataValidationError("vLLM teacher-forced response identity is invalid")
        raw_log_probabilities = tuple(self.token_log_probabilities)
        if any(
            isinstance(value, bool) or not isinstance(value, int | float)
            for value in raw_log_probabilities
        ):
            raise DataValidationError("vLLM teacher-forced token log probabilities are invalid")
        log_probabilities = tuple(float(value) for value in raw_log_probabilities)
        if len(log_probabilities) != len(response_token_ids) or any(
            not math.isfinite(value) or value > 0 for value in log_probabilities
        ):
            raise DataValidationError("vLLM teacher-forced token log probabilities are invalid")
        if any(
            isinstance(value, bool) or not isinstance(value, int | float)
            for value in (
                self.negative_log_likelihood,
                self.mean_negative_log_likelihood,
                self.perplexity,
            )
        ):
            raise DataValidationError("vLLM teacher-forced aggregate likelihood is invalid")
        negative_log_likelihood = float(self.negative_log_likelihood)
        mean_negative_log_likelihood = float(self.mean_negative_log_likelihood)
        perplexity = float(self.perplexity)
        expected_nll = -sum(log_probabilities)
        if (
            not math.isfinite(negative_log_likelihood)
            or not math.isfinite(mean_negative_log_likelihood)
            or not math.isfinite(perplexity)
            or negative_log_likelihood < 0
            or mean_negative_log_likelihood < 0
            or perplexity < 1
            or not math.isclose(
                negative_log_likelihood,
                expected_nll,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                mean_negative_log_likelihood,
                expected_nll / len(log_probabilities),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                perplexity,
                math.exp(mean_negative_log_likelihood),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise DataValidationError("vLLM teacher-forced aggregate likelihood is invalid")
        activations: dict[int, np.ndarray[Any, Any]] = {}
        for raw_layer, raw_activation in self.activations.items():
            if type(raw_layer) is not int or raw_layer < 0:
                raise DataValidationError(
                    "vLLM teacher-forced activation layers must be exact integers"
                )
            activation = as_numpy(raw_activation, dtype=np.float32, copy=True)
            if (
                activation.ndim != 2
                or activation.shape[0] != len(self.response_token_ids)
                or activation.shape[1] == 0
                or not np.isfinite(activation).all()
            ):
                raise DataValidationError(
                    "vLLM teacher-forced response activation trajectory is invalid"
                )
            activation.setflags(write=False)
            activations[raw_layer] = activation
        if not activations or len(activations) != len(self.activations):
            raise DataValidationError("vLLM teacher-forced activation layers must be unique")
        object.__setattr__(self, "response_token_ids", response_token_ids)
        object.__setattr__(self, "token_log_probabilities", log_probabilities)
        object.__setattr__(self, "negative_log_likelihood", negative_log_likelihood)
        object.__setattr__(
            self,
            "mean_negative_log_likelihood",
            mean_negative_log_likelihood,
        )
        object.__setattr__(self, "perplexity", perplexity)
        object.__setattr__(self, "activations", MappingProxyType(activations))


@dataclass(frozen=True, slots=True)
class VllmTeacherForcedCubeOutput:
    """One-pass response trajectories for several activation sites and layers."""

    response_text_sha256: str
    response_token_ids: tuple[int, ...]
    response_token_ids_sha256: str
    token_log_probabilities: tuple[float, ...]
    negative_log_likelihood: float
    mean_negative_log_likelihood: float
    perplexity: float
    activations: Mapping[ActivationSite, Mapping[int, np.ndarray[Any, Any]]]
    peak_memory_bytes: int = 0

    def __post_init__(self) -> None:
        sites: dict[ActivationSite, Mapping[int, np.ndarray[Any, Any]]] = {}
        layer_identity: tuple[int, ...] | None = None
        for raw_site, raw_activations in self.activations.items():
            if not isinstance(raw_site, ActivationSite):
                raise DataValidationError("vLLM teacher-forced cube site is invalid")
            output = VllmTeacherForcedOutput(
                response_text_sha256=self.response_text_sha256,
                response_token_ids=self.response_token_ids,
                response_token_ids_sha256=self.response_token_ids_sha256,
                token_log_probabilities=self.token_log_probabilities,
                negative_log_likelihood=self.negative_log_likelihood,
                mean_negative_log_likelihood=self.mean_negative_log_likelihood,
                perplexity=self.perplexity,
                activations=raw_activations,
            )
            current_layers = tuple(output.activations)
            if layer_identity is not None and current_layers != layer_identity:
                raise DataValidationError("vLLM teacher-forced cube layers differ across sites")
            layer_identity = current_layers
            sites[raw_site] = output.activations
        if not sites or len(sites) != len(self.activations):
            raise DataValidationError("vLLM teacher-forced cube sites must be unique")
        if type(self.peak_memory_bytes) is not int or self.peak_memory_bytes < 0:
            raise DataValidationError("vLLM teacher-forced cube peak memory is invalid")
        canonical = VllmTeacherForcedOutput(
            response_text_sha256=self.response_text_sha256,
            response_token_ids=self.response_token_ids,
            response_token_ids_sha256=self.response_token_ids_sha256,
            token_log_probabilities=self.token_log_probabilities,
            negative_log_likelihood=self.negative_log_likelihood,
            mean_negative_log_likelihood=self.mean_negative_log_likelihood,
            perplexity=self.perplexity,
            activations=next(iter(sites.values())),
        )
        object.__setattr__(self, "response_token_ids", canonical.response_token_ids)
        object.__setattr__(self, "token_log_probabilities", canonical.token_log_probabilities)
        object.__setattr__(self, "negative_log_likelihood", canonical.negative_log_likelihood)
        object.__setattr__(
            self, "mean_negative_log_likelihood", canonical.mean_negative_log_likelihood
        )
        object.__setattr__(self, "perplexity", canonical.perplexity)
        object.__setattr__(self, "activations", MappingProxyType(sites))


def _logprob(value: Any) -> float:
    raw = getattr(value, "logprob", value)
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise DataValidationError("vLLM returned an invalid log probability")
    result = float(raw)
    if not math.isfinite(result) or result > 0:
        raise DataValidationError("vLLM returned a non-finite log probability")
    return result


def _selected_logprob(position: Any, token_id: int) -> float:
    if not isinstance(position, Mapping) or token_id not in position:
        raise DataValidationError("vLLM prompt logprobs omit the teacher-forced token")
    return _logprob(position[token_id])


def _copy_state(target: VllmInterventionState, source: VllmInterventionState) -> None:
    target.generated_calls = source.generated_calls
    target.applications = source.applications
    target.captured = source.captured
    target.intervened = source.intervened
    target.prompt_tokens_remaining = source.prompt_tokens_remaining
    target.phase_armed = source.phase_armed
    target.capture_history[:] = source.capture_history
    target.applied_pre_history[:] = source.applied_pre_history
    target.applied_post_history[:] = source.applied_post_history


class VllmResearchRuntime:
    """Activation-research operations implemented through one vLLM GPU worker."""

    def __init__(
        self,
        base: VllmRuntime,
        *,
        research_provenance: Mapping[str, Any] | None = None,
    ) -> None:
        self.base = base
        provenance = dict(research_provenance or {})
        try:
            replayed = json.loads(json.dumps(provenance, sort_keys=True, allow_nan=False))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"vLLM research provenance must be exact JSON: {exc}") from exc
        if replayed != provenance:
            raise ConfigurationError("vLLM research provenance is not stable JSON")
        self.research_provenance = MappingProxyType(replayed)

    @classmethod
    def from_spec(
        cls,
        model_spec: ModelSpec,
        *,
        snapshot_path: str | Path,
        seed: int = 17,
        research_provenance: Mapping[str, Any] | None = None,
    ) -> VllmResearchRuntime:
        return cls(
            VllmRuntime.from_spec(model_spec, snapshot_path=snapshot_path, seed=seed),
            research_provenance=research_provenance,
        )

    def render_prompt(
        self,
        prompt: PromptSpec,
        question: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> VllmRenderedPrompt:
        return self.base.render_prompt(prompt, question, metadata=metadata)

    def generate(
        self, rendered: VllmRenderedPrompt, *, max_new_tokens: int
    ) -> VllmGenerationOutput:
        return self.base.generate(rendered, max_new_tokens=max_new_tokens)

    @staticmethod
    def _validate_states(
        states: Mapping[tuple[int, ActivationSite], VllmInterventionState],
        *,
        num_layers: int,
    ) -> dict[tuple[int, ActivationSite], VllmInterventionState]:
        parsed: dict[tuple[int, ActivationSite], VllmInterventionState] = {}
        for key, state in states.items():
            if (
                type(key) is not tuple
                or len(key) != 2
                or type(key[0]) is not int
                or not 0 <= key[0] < num_layers
                or not isinstance(key[1], ActivationSite)
                or not isinstance(state, VllmInterventionState)
                or key in parsed
            ):
                raise ConfigurationError("vLLM intervention key or state is invalid")
            _validate_fresh_intervention_state(state, expected_width=5120)
            parsed[key] = state
        if len({id(state) for state in parsed.values()}) != len(parsed):
            raise ConfigurationError("vLLM requires one distinct state per hook point")
        return parsed

    def generate_with_interventions(
        self,
        rendered: VllmRenderedPrompt,
        *,
        max_new_tokens: int,
        intervention_states: Mapping[tuple[int, ActivationSite], VllmInterventionState],
    ) -> VllmGenerationOutput:
        parsed = self._validate_states(
            intervention_states, num_layers=self.base.model_spec.num_layers
        )
        for state in parsed.values():
            state.arm_prompt(len(rendered.token_ids))
        generation, _request, _hooks = self.base.generate_with_states(
            rendered,
            max_new_tokens=max_new_tokens,
            states=parsed,
        )
        return generation

    def generate_with_online_gate(
        self,
        rendered: VllmRenderedPrompt,
        *,
        max_new_tokens: int,
        intervention_states: Mapping[tuple[int, ActivationSite], VllmInterventionState],
        capture_keys: Sequence[tuple[int, ActivationSite]],
        feature_token_count: int,
        early_gate: Callable[[VllmOnlinePrefixCapture], bool],
    ) -> VllmOnlineGenerationOutput:
        """Apply an early gate using a bounded first pass, then replay its prefix.

        vLLM's in-process API does not expose a pausable generation stream here.
        Continuation therefore starts a second deterministic request containing
        the exact first-pass prefix; the receipt must describe this as two-pass
        replay rather than as one paused live stream.
        """

        if (
            type(max_new_tokens) is not int
            or not 1 <= max_new_tokens <= 48
            or type(feature_token_count) is not int
            or not 1 <= feature_token_count <= max_new_tokens
            or not callable(early_gate)
        ):
            raise ConfigurationError("vLLM online gate inputs are invalid")
        parsed = self._validate_states(
            intervention_states, num_layers=self.base.model_spec.num_layers
        )
        selected = tuple(capture_keys)
        if (
            not selected
            or len(set(selected)) != len(selected)
            or any(key not in parsed for key in selected)
        ):
            raise ConfigurationError("vLLM online gate capture inventory is invalid")
        for key, state in parsed.items():
            state.capture_limit = feature_token_count if key in selected else 0
            state.arm_prompt(len(rendered.token_ids))

        buffered_limit = min(max_new_tokens, feature_token_count + 1)
        first, _request, _hooks = self.base.generate_with_states(
            rendered,
            max_new_tokens=buffered_limit,
            states=parsed,
        )
        captured_count = min(len(parsed[key].capture_history) for key in selected)
        if first.stop_type == "short_answer" or captured_count < feature_token_count:
            return VllmOnlineGenerationOutput(
                generation=first,
                early_gate_applied=False,
                continued_after_early_gate=False,
                feature_token_count=captured_count,
                buffered_token_count_at_gate=0,
            )
        cube: dict[ActivationSite, dict[int, np.ndarray[Any, Any]]] = {}
        for layer, site in selected:
            values = np.ascontiguousarray(
                np.stack(parsed[(layer, site)].capture_history[:feature_token_count]),
                dtype=np.float32,
            )
            values.setflags(write=False)
            cube.setdefault(site, {})[layer] = values
        capture = VllmOnlinePrefixCapture(
            text=first.text,
            token_ids=first.token_ids,
            activations=MappingProxyType(
                {site: MappingProxyType(layers) for site, layers in cube.items()}
            ),
            feature_token_count=feature_token_count,
        )
        decision = early_gate(capture)
        if type(decision) is not bool:
            raise DataValidationError("vLLM online gate must return an exact boolean")
        if not decision or len(first.token_ids) >= max_new_tokens:
            generation = VllmGenerationOutput(
                rendered_prompt=first.rendered_prompt,
                token_ids=first.token_ids,
                text=first.text,
                input_tokens=first.input_tokens,
                output_tokens=first.output_tokens,
                latency_seconds=first.latency_seconds,
                stop_type="online_gate" if not decision else first.stop_type,
                stopping_token_id=first.stopping_token_id,
                prompt_tokens_per_second=first.prompt_tokens_per_second,
                generation_tokens_per_second=first.generation_tokens_per_second,
                peak_memory_bytes=first.peak_memory_bytes,
                active_memory_bytes=first.active_memory_bytes,
                cache_memory_bytes=first.cache_memory_bytes,
            )
            return VllmOnlineGenerationOutput(
                generation=generation,
                early_gate_applied=True,
                continued_after_early_gate=False,
                feature_token_count=feature_token_count,
                buffered_token_count_at_gate=len(first.token_ids),
            )

        resumed: dict[tuple[int, ActivationSite], VllmInterventionState] = {}
        for key, state in parsed.items():
            resumed[key] = VllmResearchInterventionState(
                direction=None if state.direction is None else as_numpy(state.direction),
                alpha=state.alpha,
                token_scope=state.token_scope,
                decay=state.decay,
                capture_limit=state.capture_limit,
            )
            resumed[key].arm_prompt(len(rendered.token_ids))
        prefix = (*rendered.token_ids, *first.token_ids)
        second, _second_request, _second_hooks = self.base.generate_with_states(
            rendered,
            max_new_tokens=max_new_tokens - len(first.token_ids),
            states=resumed,
            prompt_tokens=len(rendered.token_ids),
            token_ids=prefix,
        )
        for key, state in parsed.items():
            _copy_state(state, resumed[key])
        combined_ids, combined_text, short_stop = self.base._truncate_short_answer(
            (*first.token_ids, *second.token_ids)
        )
        generation = VllmGenerationOutput(
            rendered_prompt=rendered,
            token_ids=combined_ids,
            text=combined_text,
            input_tokens=len(rendered.token_ids),
            output_tokens=len(combined_ids),
            latency_seconds=first.latency_seconds + second.latency_seconds,
            stop_type="short_answer" if short_stop else second.stop_type,
            stopping_token_id=combined_ids[-1] if combined_ids else None,
            prompt_tokens_per_second=second.prompt_tokens_per_second,
            generation_tokens_per_second=(
                len(combined_ids) / max(first.latency_seconds + second.latency_seconds, 1e-12)
            ),
            peak_memory_bytes=max(first.peak_memory_bytes, second.peak_memory_bytes),
            active_memory_bytes=second.active_memory_bytes,
            cache_memory_bytes=second.cache_memory_bytes,
        )
        return VllmOnlineGenerationOutput(
            generation=generation,
            early_gate_applied=True,
            continued_after_early_gate=bool(second.token_ids),
            feature_token_count=feature_token_count,
            buffered_token_count_at_gate=len(first.token_ids),
        )

    def runtime_identity(self) -> Mapping[str, Any]:
        identity = dict(self.base.runtime_identity())
        identity.update(
            {
                "model_repository": self.base.model_spec.repository,
                "model_revision": self.base.model_spec.revision,
                "model_quantization": self.base.model_spec.quantization,
                "model_num_layers": self.base.model_spec.num_layers,
                "snapshot_sha256": sha256_path(self.base.snapshot),
            }
        )
        if self.research_provenance:
            identity["research_provenance"] = dict(self.research_provenance)
            identity["research_toolchain"] = dict(vllm_research_toolchain_identity())
        return MappingProxyType(identity)

    def standardized_intervention_state(
        self,
        direction: np.ndarray[Any, Any],
        *,
        standardized_alpha: float,
        reference_rms: float,
        token_scope: TokenScope,
        decay: float = 0.0,
    ) -> VllmResearchInterventionState:
        values = as_numpy(direction, dtype=np.float32)
        norm = float(np.linalg.norm(values)) if values.ndim == 1 else math.nan
        if (
            values.ndim != 1
            or values.size != 5120
            or not np.isfinite(values).all()
            or not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-6)
            or isinstance(standardized_alpha, bool)
            or not isinstance(standardized_alpha, int | float)
            or not math.isfinite(float(standardized_alpha))
            or isinstance(reference_rms, bool)
            or not isinstance(reference_rms, int | float)
            or not math.isfinite(float(reference_rms))
            or float(reference_rms) <= 0
            or not isinstance(token_scope, TokenScope)
            or isinstance(decay, bool)
            or not isinstance(decay, int | float)
            or not math.isfinite(float(decay))
            or (token_scope is TokenScope.EXPONENTIAL_DECAY and float(decay) <= 0)
        ):
            raise ConfigurationError("RMS-standardized vLLM intervention inputs are invalid")
        return VllmResearchInterventionState(
            direction=values.copy(),
            alpha=float(standardized_alpha) * float(reference_rms),
            token_scope=token_scope,
            decay=float(decay),
        )

    @staticmethod
    def _validate_geometry(
        layers: Sequence[int], sites: Sequence[ActivationSite], *, num_layers: int
    ) -> tuple[tuple[int, ...], tuple[ActivationSite, ...]]:
        selected_layers = tuple(layers)
        selected_sites = tuple(sites)
        if (
            not selected_layers
            or len(set(selected_layers)) != len(selected_layers)
            or any(
                type(layer) is not int or not 0 <= layer < num_layers
                for layer in selected_layers
            )
            or not selected_sites
            or len(set(selected_sites)) != len(selected_sites)
            or any(not isinstance(site, ActivationSite) for site in selected_sites)
        ):
            raise ConfigurationError("vLLM capture geometry is invalid")
        return selected_layers, selected_sites

    def prompt_feature_bundle(
        self,
        rendered: VllmRenderedPrompt,
        *,
        layers: Sequence[int],
        site: ActivationSite,
    ) -> VllmPromptFeatureOutput:
        cube = self.prompt_feature_cube(rendered, layers=layers, sites=(site,))
        return VllmPromptFeatureOutput(
            activations=cube.activations[site],
            maximum_token_probability=cube.maximum_token_probability,
            output_entropy=cube.output_entropy,
        )

    def prompt_feature_cube(
        self,
        rendered: VllmRenderedPrompt,
        *,
        layers: Sequence[int],
        sites: Sequence[ActivationSite],
    ) -> VllmPromptFeatureCubeOutput:
        selected_layers, selected_sites = self._validate_geometry(
            layers, sites, num_layers=self.base.model_spec.num_layers
        )
        states = {
            (layer, site): VllmResearchInterventionState()
            for layer in selected_layers
            for site in selected_sites
        }
        specs = [
            _state_spec(layer, site, state, prompt_tokens=len(rendered.token_ids))
            for (layer, site), state in states.items()
        ]
        request, hooks, _latency = self.base.request(
            rendered.token_ids,
            max_tokens=1,
            specifications=specs,
            logprobs=-1,
        )
        for (layer, site), state in states.items():
            _sync_state(state, hooks["specs"][f"{layer}:{site.value}"])
        distribution = request.outputs[0].logprobs[0]
        if not isinstance(distribution, Mapping) or len(distribution) < 2:
            raise DataValidationError("vLLM did not return the full next-token distribution")
        probabilities = np.asarray(
            [math.exp(_logprob(value)) for value in distribution.values()], dtype=np.float64
        )
        positive = probabilities[probabilities > 0]
        entropy = -float(np.sum(positive * np.log(positive)))
        activations = {
            site: {
                layer: as_numpy(states[(layer, site)].captured, dtype=np.float32)[:, -1, :]
                for layer in selected_layers
            }
            for site in selected_sites
        }
        return VllmPromptFeatureCubeOutput(
            activations=activations,
            maximum_token_probability=float(probabilities.max()),
            output_entropy=entropy,
            peak_memory_bytes=int(hooks["peak_memory_bytes"]),
        )

    def _continuation_token_ids(
        self, rendered: VllmRenderedPrompt, response: str
    ) -> tuple[int, ...]:
        if not isinstance(response, str) or not response.strip():
            raise ConfigurationError("teacher-forced response must be non-empty text")
        candidates: set[tuple[int, ...]] = set()
        for options in ({"add_special_tokens": False}, {}):
            try:
                values = tuple(self.base.tokenizer.encode(rendered.text + response, **options))
            except (AttributeError, TypeError, ValueError):
                continue
            if values[: len(rendered.token_ids)] == rendered.token_ids and len(values) > len(
                rendered.token_ids
            ):
                candidates.add(values[len(rendered.token_ids) :])
        if len(candidates) != 1:
            raise DataValidationError(
                "vLLM response tokenization is not one unambiguous prompt extension"
            )
        return next(iter(candidates))

    def teacher_forced_cube(
        self,
        rendered: VllmRenderedPrompt,
        response: str,
        *,
        layers: Sequence[int],
        sites: Sequence[ActivationSite],
        intervention_states: Mapping[tuple[ActivationSite, int], VllmInterventionState]
        | None = None,
    ) -> VllmTeacherForcedCubeOutput:
        selected_layers, selected_sites = self._validate_geometry(
            layers, sites, num_layers=self.base.model_spec.num_layers
        )
        response_ids = self._continuation_token_ids(rendered, response)
        expected = {(site, layer) for site in selected_sites for layer in selected_layers}
        provided = dict(intervention_states or {})
        if any(key not in expected for key in provided):
            raise ConfigurationError("vLLM teacher-forced intervention key is invalid")
        states: dict[tuple[ActivationSite, int], VllmInterventionState] = {}
        for key in expected:
            state = provided.get(key, VllmResearchInterventionState())
            _validate_fresh_intervention_state(state, expected_width=5120)
            state.capture_limit = len(response_ids)
            state.arm_prompt(len(rendered.token_ids))
            states[key] = state
        if len({id(state) for state in states.values()}) != len(states):
            raise ConfigurationError("vLLM teacher forcing requires distinct hook states")
        specifications = [
            _state_spec(layer, site, states[(site, layer)], prompt_tokens=len(rendered.token_ids))
            for site, layer in sorted(expected, key=lambda value: (value[1], value[0].value))
        ]
        full_ids = (*rendered.token_ids, *response_ids)
        request, hooks, _latency = self.base.request(
            full_ids,
            max_tokens=1,
            specifications=specifications,
            prompt_logprobs=1,
        )
        for (site, layer), state in states.items():
            _sync_state(state, hooks["specs"][f"{layer}:{site.value}"])
        prompt_logprobs = request.prompt_logprobs
        if prompt_logprobs is None or len(prompt_logprobs) != len(full_ids):
            raise DataValidationError("vLLM teacher-forced prompt logprobs are incomplete")
        log_probabilities = tuple(
            _selected_logprob(prompt_logprobs[index], token_id)
            for index, token_id in enumerate(response_ids, start=len(rendered.token_ids))
        )
        nll = -sum(log_probabilities)
        mean_nll = nll / len(log_probabilities)
        activations: dict[ActivationSite, dict[int, np.ndarray[Any, Any]]] = {}
        for site in selected_sites:
            activations[site] = {}
            for layer in selected_layers:
                history = states[(site, layer)].capture_history
                if len(history) != len(response_ids):
                    raise DataValidationError(
                        "vLLM teacher-forced activation capture is incomplete"
                    )
                activations[site][layer] = np.stack(history, axis=0)
        return VllmTeacherForcedCubeOutput(
            response_text_sha256=hashlib.sha256(response.encode("utf-8")).hexdigest(),
            response_token_ids=response_ids,
            response_token_ids_sha256=_token_ids_digest(response_ids),
            token_log_probabilities=log_probabilities,
            negative_log_likelihood=nll,
            mean_negative_log_likelihood=mean_nll,
            perplexity=math.exp(mean_nll),
            activations=activations,
            peak_memory_bytes=int(hooks["peak_memory_bytes"]),
        )

    def teacher_forced_continuation(
        self,
        rendered: VllmRenderedPrompt,
        response: str,
        *,
        layers: Sequence[int],
        site: ActivationSite,
        intervention_states: Mapping[int, VllmInterventionState] | None = None,
    ) -> VllmTeacherForcedOutput:
        provided = dict(intervention_states or {})
        cube = self.teacher_forced_cube(
            rendered,
            response,
            layers=layers,
            sites=(site,),
            intervention_states={(site, layer): state for layer, state in provided.items()},
        )
        return VllmTeacherForcedOutput(
            response_text_sha256=cube.response_text_sha256,
            response_token_ids=cube.response_token_ids,
            response_token_ids_sha256=cube.response_token_ids_sha256,
            token_log_probabilities=cube.token_log_probabilities,
            negative_log_likelihood=cube.negative_log_likelihood,
            mean_negative_log_likelihood=cube.mean_negative_log_likelihood,
            perplexity=cube.perplexity,
            activations=cube.activations[site],
            peak_memory_bytes=cube.peak_memory_bytes,
        )

    def close(self) -> None:
        self.base.close()

"""Research-only MLX capture extensions layered over the frozen E0/E1 runtime."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from mfh.contracts import ActivationSite, ModelSpec, PromptSpec, TokenScope
from mfh.errors import ConfigurationError, DataValidationError
from mfh.inference.mlx_runtime import (
    MlxGenerationOutput,
    MlxInterventionState,
    MlxRenderedPrompt,
    MlxRuntime,
    _completed_short_answer,
    _mlx_modules,
)
from mfh.provenance import sha256_path


@dataclass(slots=True)
class MlxResearchInterventionState(MlxInterventionState):
    """Intervention state with explicit prompt/generation phase accounting.

    ``mlx-lm`` prefills all but the final prompt token and then processes that
    final token in a one-token step.  Tensor sequence length therefore cannot
    distinguish the final prompt token from a generated-token activation.  A
    research state is armed with the exact rendered prompt length immediately
    before generation and consumes that prompt progress explicitly.
    """

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
            or type(self.capture_limit) is not int
            or self.capture_limit < 0
            or self.capture_history
            or self.applied_pre_history
            or self.applied_post_history
        ):
            raise ConfigurationError("MLX research intervention phase is not fresh")
        self.prompt_tokens_remaining = prompt_tokens
        self.phase_armed = True

    def effective_alpha(self, sequence_length: int) -> float:
        if type(sequence_length) is not int or sequence_length <= 0 or not self.phase_armed:
            raise ConfigurationError("MLX research intervention phase is invalid")
        if self.prompt_tokens_remaining:
            if sequence_length > self.prompt_tokens_remaining:
                raise ConfigurationError("MLX hook crossed the frozen prompt/generation boundary")
            reaches_final_prompt = sequence_length == self.prompt_tokens_remaining
            self.prompt_tokens_remaining -= sequence_length
            if self.direction is None or self.alpha == 0.0:
                return 0.0
            return (
                self.alpha
                if reaches_final_prompt and self.token_scope is TokenScope.FINAL_PROMPT
                else 0.0
            )
        if self.direction is None or self.alpha == 0.0:
            return 0.0
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


@dataclass(frozen=True, slots=True)
class MlxOnlinePrefixCapture:
    """Generated-token activations available while one stream remains paused."""

    text: str
    token_ids: tuple[int, ...]
    activations: Mapping[ActivationSite, Mapping[int, np.ndarray[Any, Any]]]
    feature_token_count: int


@dataclass(frozen=True, slots=True)
class MlxOnlineGenerationOutput:
    generation: MlxGenerationOutput
    early_gate_applied: bool
    continued_after_early_gate: bool
    feature_token_count: int
    buffered_token_count_at_gate: int


def _mlx_prompt_cache_factory() -> Any:
    try:
        from mlx_lm.models.cache import make_prompt_cache
    except ImportError as exc:  # pragma: no cover - non-Apple optional dependency
        raise ConfigurationError("MLX continuation scoring requires mlx-lm") from exc
    return make_prompt_cache


def _token_ids_digest(token_ids: Sequence[int]) -> str:
    serialized = ",".join(str(value) for value in token_ids)
    return hashlib.sha256(serialized.encode("ascii")).hexdigest()


def _research_hidden_width(model: Any) -> int | None:
    arguments = getattr(model, "args", None)
    language_model = getattr(model, "language_model", None)
    language_arguments = getattr(language_model, "args", None)
    text_config = getattr(arguments, "text_config", None)
    if isinstance(text_config, Mapping):
        value = text_config.get("hidden_size")
        if type(value) is int and value > 0:
            return value
    for owner in (language_arguments, arguments, language_model, model):
        for name in ("hidden_size", "hidden_dim", "model_dim", "dim"):
            value = getattr(owner, name, None)
            if type(value) is int and value > 0:
                return value
    return None


def _validate_fresh_intervention_state(
    state: MlxInterventionState, *, expected_width: int | None
) -> None:
    if (
        state.generated_calls != 0
        or state.applications != 0
        or state.captured is not None
        or state.intervened is not None
        or (
            isinstance(state, MlxResearchInterventionState)
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
        raise ConfigurationError("MLX intervention state must be fresh and finite")
    if state.direction is None:
        if float(state.alpha) != 0:
            raise ConfigurationError("active MLX intervention requires a direction")
        return
    try:
        direction = np.asarray(state.direction, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"MLX intervention direction is invalid: {exc}") from exc
    norm = float(np.linalg.norm(direction)) if direction.ndim == 1 else math.nan
    if (
        direction.ndim != 1
        or direction.size == 0
        or not np.isfinite(direction).all()
        or not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-6)
        or (expected_width is not None and direction.size != expected_width)
    ):
        raise ConfigurationError("MLX intervention direction geometry is invalid")


def mlx_research_toolchain_identity() -> Mapping[str, str]:
    """Read the live Xcode and Metal compiler identities for long MLX phases."""

    values: dict[str, str] = {}
    for name, command in {
        "xcodebuild": ("xcodebuild", "-version"),
        "metal_compiler": ("xcrun", "metal", "--version"),
    }.items():
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ConfigurationError(f"cannot read live {name} identity: {exc}") from exc
        value = result.stdout.strip()
        if not value:
            raise ConfigurationError(f"live {name} identity is empty")
        values[name] = value
    return MappingProxyType(values)


@dataclass(frozen=True, slots=True)
class MlxPromptFeatureOutput:
    """Prompt-end activations and trivial next-token confidence baselines."""

    activations: Mapping[int, np.ndarray]
    maximum_token_probability: float
    output_entropy: float

    def __post_init__(self) -> None:
        values: dict[int, np.ndarray] = {}
        for raw_layer, raw_activation in self.activations.items():
            if type(raw_layer) is not int:
                raise DataValidationError(
                    "MLX prompt activation layer identifiers must be exact integers"
                )
            layer = raw_layer
            activation = np.asarray(raw_activation, dtype=np.float32).copy()
            if (
                layer < 0
                or activation.ndim != 2
                or activation.shape[0] != 1
                or activation.shape[1] == 0
                or not np.isfinite(activation).all()
            ):
                raise DataValidationError("MLX prompt activation bundle is invalid")
            activation.setflags(write=False)
            values[layer] = activation
        if not values or len(values) != len(self.activations):
            raise DataValidationError("MLX prompt activation layers must be unique")
        maximum = float(self.maximum_token_probability)
        entropy = float(self.output_entropy)
        if not math.isfinite(maximum) or not 0 < maximum <= 1:
            raise DataValidationError("MLX maximum-token probability is invalid")
        if not math.isfinite(entropy) or entropy < 0:
            raise DataValidationError("MLX output entropy is invalid")
        object.__setattr__(self, "activations", MappingProxyType(values))
        object.__setattr__(self, "maximum_token_probability", maximum)
        object.__setattr__(self, "output_entropy", entropy)


@dataclass(frozen=True, slots=True)
class MlxPromptFeatureCubeOutput:
    """One-forward prompt activations indexed by site and layer."""

    activations: Mapping[ActivationSite, Mapping[int, np.ndarray]]
    maximum_token_probability: float
    output_entropy: float
    peak_memory_bytes: int

    def __post_init__(self) -> None:
        sites: dict[ActivationSite, Mapping[int, np.ndarray]] = {}
        layer_identity: tuple[int, ...] | None = None
        for raw_site, raw_values in self.activations.items():
            if not isinstance(raw_site, ActivationSite) or not isinstance(raw_values, Mapping):
                raise DataValidationError("MLX prompt cube site identity is invalid")
            output = MlxPromptFeatureOutput(
                activations=raw_values,
                maximum_token_probability=self.maximum_token_probability,
                output_entropy=self.output_entropy,
            )
            current = tuple(output.activations)
            if layer_identity is not None and current != layer_identity:
                raise DataValidationError("MLX prompt cube layers differ between sites")
            layer_identity = current
            sites[raw_site] = output.activations
        if not sites or len(sites) != len(self.activations):
            raise DataValidationError("MLX prompt cube sites must be non-empty and unique")
        maximum = float(self.maximum_token_probability)
        entropy = float(self.output_entropy)
        if not math.isfinite(maximum) or not 0 < maximum <= 1:
            raise DataValidationError("MLX prompt cube maximum-token probability is invalid")
        if not math.isfinite(entropy) or entropy < 0:
            raise DataValidationError("MLX prompt cube entropy is invalid")
        if type(self.peak_memory_bytes) is not int or self.peak_memory_bytes < 0:
            raise DataValidationError("MLX prompt cube peak memory is invalid")
        object.__setattr__(self, "activations", MappingProxyType(sites))
        object.__setattr__(self, "maximum_token_probability", maximum)
        object.__setattr__(self, "output_entropy", entropy)


@dataclass(frozen=True, slots=True)
class MlxTeacherForcedOutput:
    """Exact continuation likelihood and response-token activation trajectory."""

    response_text_sha256: str
    response_token_ids: tuple[int, ...]
    response_token_ids_sha256: str
    token_log_probabilities: tuple[float, ...]
    negative_log_likelihood: float
    mean_negative_log_likelihood: float
    perplexity: float
    activations: Mapping[int, np.ndarray]
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
            raise DataValidationError("MLX teacher-forced response identity is invalid")
        raw_log_probabilities = tuple(self.token_log_probabilities)
        if any(
            isinstance(value, bool) or not isinstance(value, int | float)
            for value in raw_log_probabilities
        ):
            raise DataValidationError("MLX teacher-forced token log probabilities are invalid")
        log_probabilities = tuple(float(value) for value in raw_log_probabilities)
        if len(log_probabilities) != len(response_token_ids) or any(
            not math.isfinite(value) or value > 0 for value in log_probabilities
        ):
            raise DataValidationError("MLX teacher-forced token log probabilities are invalid")
        if any(
            isinstance(value, bool) or not isinstance(value, int | float)
            for value in (
                self.negative_log_likelihood,
                self.mean_negative_log_likelihood,
                self.perplexity,
            )
        ):
            raise DataValidationError("MLX teacher-forced aggregate likelihood is invalid")
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
            raise DataValidationError("MLX teacher-forced aggregate likelihood is invalid")
        activations: dict[int, np.ndarray] = {}
        for raw_layer, raw_activation in self.activations.items():
            if type(raw_layer) is not int or raw_layer < 0:
                raise DataValidationError(
                    "MLX teacher-forced activation layers must be exact integers"
                )
            activation = np.asarray(raw_activation, dtype=np.float32).copy()
            if (
                activation.ndim != 2
                or activation.shape[0] != len(self.response_token_ids)
                or activation.shape[1] == 0
                or not np.isfinite(activation).all()
            ):
                raise DataValidationError(
                    "MLX teacher-forced response activation trajectory is invalid"
                )
            activation.setflags(write=False)
            activations[raw_layer] = activation
        if not activations or len(activations) != len(self.activations):
            raise DataValidationError("MLX teacher-forced activation layers must be unique")
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
class MlxTeacherForcedCubeOutput:
    """One-pass response trajectories for several activation sites and layers."""

    response_text_sha256: str
    response_token_ids: tuple[int, ...]
    response_token_ids_sha256: str
    token_log_probabilities: tuple[float, ...]
    negative_log_likelihood: float
    mean_negative_log_likelihood: float
    perplexity: float
    activations: Mapping[ActivationSite, Mapping[int, np.ndarray]]
    peak_memory_bytes: int = 0

    def __post_init__(self) -> None:
        sites: dict[ActivationSite, Mapping[int, np.ndarray]] = {}
        layer_identity: tuple[int, ...] | None = None
        for raw_site, raw_activations in self.activations.items():
            if not isinstance(raw_site, ActivationSite):
                raise DataValidationError("MLX teacher-forced cube site is invalid")
            output = MlxTeacherForcedOutput(
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
                raise DataValidationError("MLX teacher-forced cube layers differ across sites")
            layer_identity = current_layers
            sites[raw_site] = output.activations
        if not sites or len(sites) != len(self.activations):
            raise DataValidationError("MLX teacher-forced cube sites must be unique")
        if type(self.peak_memory_bytes) is not int or self.peak_memory_bytes < 0:
            raise DataValidationError("MLX teacher-forced cube peak memory is invalid")
        canonical = MlxTeacherForcedOutput(
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


class MlxResearchRuntime:
    """Additional phase-E2+ operations without changing the frozen production runtime."""

    def __init__(
        self,
        base: MlxRuntime,
        *,
        research_provenance: Mapping[str, Any] | None = None,
    ) -> None:
        self.base = base
        provenance = dict(research_provenance or {})
        try:
            replayed = json.loads(json.dumps(provenance, sort_keys=True, allow_nan=False))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"MLX research provenance must be exact JSON: {exc}") from exc
        if replayed != provenance:
            raise ConfigurationError("MLX research provenance is not stable JSON")
        self.research_provenance = MappingProxyType(replayed)

    @classmethod
    def from_spec(
        cls,
        model_spec: ModelSpec,
        *,
        snapshot_path: str | Path,
        seed: int = 17,
        research_provenance: Mapping[str, Any] | None = None,
    ) -> MlxResearchRuntime:
        return cls(
            MlxRuntime.from_spec(
                model_spec,
                snapshot_path=snapshot_path,
                seed=seed,
            ),
            research_provenance=research_provenance,
        )

    def render_prompt(
        self,
        prompt: PromptSpec,
        question: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> MlxRenderedPrompt:
        return self.base.render_prompt(prompt, question, metadata=metadata)

    def generate(self, rendered: MlxRenderedPrompt, *, max_new_tokens: int) -> MlxGenerationOutput:
        return self.base.generate(rendered, max_new_tokens=max_new_tokens)

    def generate_with_interventions(
        self,
        rendered: MlxRenderedPrompt,
        *,
        max_new_tokens: int,
        intervention_states: Mapping[tuple[int, ActivationSite], MlxInterventionState],
    ) -> MlxGenerationOutput:
        """Generate under fresh, ordered research hooks and always restore the model."""

        provided = dict(intervention_states)
        parsed: dict[tuple[int, ActivationSite], MlxInterventionState] = {}
        for raw_key, state in provided.items():
            if (
                type(raw_key) is not tuple
                or len(raw_key) != 2
                or type(raw_key[0]) is not int
                or not isinstance(raw_key[1], ActivationSite)
                or not 0 <= raw_key[0] < len(self.base.model.layers)
                or not isinstance(state, MlxInterventionState)
            ):
                raise ConfigurationError("MLX generation intervention key or state is invalid")
            key = (raw_key[0], raw_key[1])
            if key in parsed:
                raise ConfigurationError("MLX generation intervention keys must be unique")
            _validate_fresh_intervention_state(
                state, expected_width=_research_hidden_width(self.base.model)
            )
            parsed[key] = state
        if len({id(state) for state in parsed.values()}) != len(parsed):
            raise ConfigurationError("MLX generation requires one distinct state per hook point")
        if any(not isinstance(state, MlxResearchInterventionState) for state in parsed.values()):
            raise ConfigurationError("active MLX research generation requires phase-aware states")
        for state in parsed.values():
            assert isinstance(state, MlxResearchInterventionState)
            state.arm_prompt(len(rendered.token_ids))
        hook_order = (
            ActivationSite.POST_MLP,
            ActivationSite.POST_ATTENTION,
            ActivationSite.BLOCK_OUTPUT,
        )
        with ExitStack() as stack:
            for layer in sorted({key[0] for key in parsed}):
                for site in hook_order:
                    selected_state = parsed.get((layer, site))
                    if selected_state is not None:
                        stack.enter_context(
                            self.base.intervention(layer=layer, site=site, state=selected_state)
                        )
            return self.base.generate(rendered, max_new_tokens=max_new_tokens)

    def generate_with_online_gate(
        self,
        rendered: MlxRenderedPrompt,
        *,
        max_new_tokens: int,
        intervention_states: Mapping[tuple[int, ActivationSite], MlxInterventionState],
        capture_keys: Sequence[tuple[int, ActivationSite]],
        feature_token_count: int,
        early_gate: Callable[[MlxOnlinePrefixCapture], bool],
    ) -> MlxOnlineGenerationOutput:
        """Pause one live stream at the early window and continue only if approved."""

        if (
            isinstance(max_new_tokens, bool)
            or not 1 <= max_new_tokens <= 48
            or isinstance(feature_token_count, bool)
            or not 1 <= feature_token_count <= max_new_tokens
            or not callable(early_gate)
        ):
            raise ConfigurationError("MLX online gate inputs are invalid")
        provided = dict(intervention_states)
        parsed: dict[tuple[int, ActivationSite], MlxResearchInterventionState] = {}
        for raw_key, raw_state in provided.items():
            if (
                type(raw_key) is not tuple
                or len(raw_key) != 2
                or type(raw_key[0]) is not int
                or not isinstance(raw_key[1], ActivationSite)
                or not 0 <= raw_key[0] < len(self.base.model.layers)
                or not isinstance(raw_state, MlxResearchInterventionState)
            ):
                raise ConfigurationError("MLX online gate hook key or state is invalid")
            key = (raw_key[0], raw_key[1])
            if key in parsed:
                raise ConfigurationError("MLX online gate hook keys must be unique")
            _validate_fresh_intervention_state(
                raw_state, expected_width=_research_hidden_width(self.base.model)
            )
            parsed[key] = raw_state
        selected_capture = tuple(capture_keys)
        if (
            not selected_capture
            or len(set(selected_capture)) != len(selected_capture)
            or any(
                type(key) is not tuple
                or len(key) != 2
                or type(key[0]) is not int
                or not isinstance(key[1], ActivationSite)
                or key not in parsed
                or parsed[key].capture_limit != feature_token_count
                for key in selected_capture
            )
            or len({id(state) for state in parsed.values()}) != len(parsed)
        ):
            raise ConfigurationError("MLX online gate capture inventory is invalid")
        for state in parsed.values():
            state.arm_prompt(len(rendered.token_ids))

        mx, _nn, _load, stream_generate = _mlx_modules()
        mx.random.seed(self.base.seed)
        mx.reset_peak_memory()
        started = time.perf_counter()
        pieces: list[str] = []
        tokens: list[int] = []
        final: Any | None = None
        gate_applied = False
        gate_continue = False
        buffered_token_count = 0
        online_gate_stop = False
        short_answer_stop = False
        response_stream: Any | None = None
        hook_order = (
            ActivationSite.POST_MLP,
            ActivationSite.POST_ATTENTION,
            ActivationSite.BLOCK_OUTPUT,
        )
        with ExitStack() as stack:
            for layer in sorted({key[0] for key in parsed}):
                for site in hook_order:
                    hook_state = parsed.get((layer, site))
                    if hook_state is not None:
                        stack.enter_context(
                            self.base.intervention(layer=layer, site=site, state=hook_state)
                        )
            response_stream = stream_generate(
                self.base.model,
                self.base.tokenizer,
                list(rendered.token_ids),
                max_tokens=max_new_tokens,
            )
            try:
                for response in response_stream:
                    final = response
                    pieces.append(str(response.text))
                    tokens.append(int(response.token))
                    completed_short_answer = (
                        response.finish_reason is None and _completed_short_answer("".join(pieces))
                    )
                    if response.finish_reason is not None or completed_short_answer:
                        short_answer_stop = completed_short_answer
                        break
                    if not gate_applied and all(
                        len(parsed[key].capture_history) >= feature_token_count
                        for key in selected_capture
                    ):
                        cube: dict[ActivationSite, dict[int, np.ndarray[Any, Any]]] = {}
                        for layer, site in selected_capture:
                            values = np.ascontiguousarray(
                                np.stack(
                                    parsed[(layer, site)].capture_history[:feature_token_count]
                                ),
                                dtype=np.float32,
                            )
                            values.setflags(write=False)
                            cube.setdefault(site, {})[layer] = values
                        capture = MlxOnlinePrefixCapture(
                            text="".join(pieces),
                            token_ids=tuple(tokens),
                            activations=MappingProxyType(
                                {
                                    site: MappingProxyType(dict(layers))
                                    for site, layers in cube.items()
                                }
                            ),
                            feature_token_count=feature_token_count,
                        )
                        decision = early_gate(capture)
                        if type(decision) is not bool:
                            raise DataValidationError(
                                "MLX online gate must return an exact boolean"
                            )
                        gate_applied = True
                        gate_continue = decision
                        buffered_token_count = len(tokens)
                        if not decision:
                            online_gate_stop = True
                            break
            finally:
                response_stream.close()
        latency = time.perf_counter() - started
        if final is None:
            raise DataValidationError("MLX online generation returned no response")
        captured_count = min(len(parsed[key].capture_history) for key in selected_capture)
        actually_continued = gate_applied and gate_continue and len(tokens) > buffered_token_count
        if gate_applied and gate_continue and not actually_continued:
            raise DataValidationError(
                "MLX online stream accepted a gate but did not continue generation"
            )
        generation = MlxGenerationOutput(
            rendered_prompt=rendered,
            token_ids=tuple(tokens),
            text="".join(pieces),
            input_tokens=int(final.prompt_tokens),
            output_tokens=len(tokens),
            latency_seconds=latency,
            stop_type=(
                "online_gate"
                if online_gate_stop
                else "short_answer"
                if short_answer_stop
                else str(final.finish_reason)
            ),
            stopping_token_id=tokens[-1] if tokens else None,
            prompt_tokens_per_second=float(final.prompt_tps),
            generation_tokens_per_second=float(final.generation_tps),
            peak_memory_bytes=int(mx.get_peak_memory()),
            active_memory_bytes=int(mx.get_active_memory()),
            cache_memory_bytes=int(mx.get_cache_memory()),
        )
        return MlxOnlineGenerationOutput(
            generation=generation,
            early_gate_applied=gate_applied,
            continued_after_early_gate=actually_continued,
            feature_token_count=(feature_token_count if gate_applied else captured_count),
            buffered_token_count_at_gate=buffered_token_count,
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
            identity["research_toolchain"] = mlx_research_toolchain_identity()
        return MappingProxyType(identity)

    def standardized_intervention_state(
        self,
        direction: np.ndarray,
        *,
        standardized_alpha: float,
        reference_rms: float,
        token_scope: TokenScope,
        decay: float = 0.0,
    ) -> MlxResearchInterventionState:
        """Translate preregistered RMS-relative strength into one fresh MLX state."""

        values = np.asarray(direction, dtype=np.float32)
        norm = float(np.linalg.norm(values)) if values.ndim == 1 else math.nan
        if (
            values.ndim != 1
            or values.size == 0
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
            raise ConfigurationError("RMS-standardized MLX intervention inputs are invalid")
        mx, _nn, _load, _stream_generate = _mlx_modules()
        return MlxResearchInterventionState(
            direction=mx.array(values.copy()),
            alpha=float(standardized_alpha) * float(reference_rms),
            token_scope=token_scope,
            decay=float(decay),
        )

    def prompt_feature_bundle(
        self,
        rendered: MlxRenderedPrompt,
        *,
        layers: Sequence[int],
        site: ActivationSite,
    ) -> MlxPromptFeatureOutput:
        """Capture several layers at one site in a single prompt-only forward pass."""

        result = self.prompt_feature_cube(rendered, layers=layers, sites=(site,))
        return MlxPromptFeatureOutput(
            activations=result.activations[site],
            maximum_token_probability=result.maximum_token_probability,
            output_entropy=result.output_entropy,
        )

    def prompt_feature_cube(
        self,
        rendered: MlxRenderedPrompt,
        *,
        layers: Sequence[int],
        sites: Sequence[ActivationSite],
    ) -> MlxPromptFeatureCubeOutput:
        """Capture every requested layer and site in one prompt-only forward pass."""

        selected_layers = tuple(layers)
        selected_sites = tuple(sites)
        if (
            not selected_layers
            or any(type(value) is not int for value in selected_layers)
            or len(set(selected_layers)) != len(selected_layers)
            or any(value < 0 or value >= len(self.base.model.layers) for value in selected_layers)
        ):
            raise ConfigurationError(
                "MLX prompt feature layers must be unique in-range exact integers"
            )
        if (
            not selected_sites
            or any(not isinstance(value, ActivationSite) for value in selected_sites)
            or len(set(selected_sites)) != len(selected_sites)
        ):
            raise ConfigurationError("MLX prompt feature sites must be unique activation sites")
        mx, _nn, _load, _stream_generate = _mlx_modules()
        mx.reset_peak_memory()
        states = {
            (site, layer): MlxInterventionState()
            for site in selected_sites
            for layer in selected_layers
        }
        hook_order = (
            ActivationSite.POST_MLP,
            ActivationSite.POST_ATTENTION,
            ActivationSite.BLOCK_OUTPUT,
        )
        with ExitStack() as stack:
            for layer in selected_layers:
                for hook_site in hook_order:
                    if hook_site in selected_sites:
                        stack.enter_context(
                            self.base.intervention(
                                layer=layer,
                                site=hook_site,
                                state=states[(hook_site, layer)],
                            )
                        )
            logits = self.base.model(mx.array([rendered.token_ids]))
            captured: dict[tuple[ActivationSite, int], Any] = {}
            for key, state in states.items():
                if state.captured is None:
                    raise DataValidationError("MLX prompt feature hook missed an activation")
                captured[key] = state.captured
            mx.eval(logits, *captured.values())

        final_logits = np.asarray(logits[0, -1, :], dtype=np.float64)
        if final_logits.ndim != 1 or final_logits.size < 2 or not np.isfinite(final_logits).all():
            raise DataValidationError("MLX prompt logits are invalid")
        shifted = final_logits - float(final_logits.max())
        unnormalized = np.exp(shifted)
        denominator = float(unnormalized.sum())
        if not math.isfinite(denominator) or denominator <= 0:
            raise DataValidationError("MLX prompt probabilities cannot be normalized")
        probabilities = unnormalized / denominator
        positive = probabilities > 0
        entropy = -float(np.sum(probabilities[positive] * np.log(probabilities[positive])))
        activations = {
            site: {
                layer: np.asarray(captured[(site, layer)][:, -1, :], dtype=np.float32)
                for layer in selected_layers
            }
            for site in selected_sites
        }
        return MlxPromptFeatureCubeOutput(
            activations=activations,
            maximum_token_probability=float(probabilities.max()),
            output_entropy=entropy,
            peak_memory_bytes=int(mx.get_peak_memory()),
        )

    def _continuation_token_ids(
        self,
        rendered: MlxRenderedPrompt,
        response: str,
    ) -> tuple[int, ...]:
        if not isinstance(response, str) or not response.strip():
            raise ConfigurationError("teacher-forced response must be non-empty text")
        candidates: set[tuple[int, ...]] = set()
        for options in ({"add_special_tokens": False}, {}):
            try:
                encoded = self.base.tokenizer.encode(rendered.text + response, **options)
            except (AttributeError, TypeError, ValueError):
                continue
            values = tuple(encoded)
            if any(type(value) is not int or value < 0 for value in values):
                raise DataValidationError("MLX tokenizer returned invalid continuation tokens")
            if (
                len(values) > len(rendered.token_ids)
                and values[: len(rendered.token_ids)] == rendered.token_ids
            ):
                candidates.add(values[len(rendered.token_ids) :])
        if len(candidates) != 1:
            raise DataValidationError(
                "MLX response tokenization is not one unambiguous extension of the prompt"
            )
        return next(iter(candidates))

    def teacher_forced_cube(
        self,
        rendered: MlxRenderedPrompt,
        response: str,
        *,
        layers: Sequence[int],
        sites: Sequence[ActivationSite],
        intervention_states: Mapping[tuple[ActivationSite, int], MlxInterventionState]
        | None = None,
    ) -> MlxTeacherForcedCubeOutput:
        """Score one response and capture every selected site/layer in one cache pass."""

        selected_layers = tuple(layers)
        selected_sites = tuple(sites)
        if (
            not selected_layers
            or any(type(value) is not int for value in selected_layers)
            or len(set(selected_layers)) != len(selected_layers)
            or any(value < 0 or value >= len(self.base.model.layers) for value in selected_layers)
        ):
            raise ConfigurationError(
                "MLX teacher-forced cube layers must be unique in-range exact integers"
            )
        if (
            not selected_sites
            or any(not isinstance(value, ActivationSite) for value in selected_sites)
            or len(set(selected_sites)) != len(selected_sites)
        ):
            raise ConfigurationError(
                "MLX teacher-forced cube sites must be unique activation sites"
            )
        if len(rendered.token_ids) < 2:
            raise DataValidationError(
                "MLX teacher forcing requires at least two frozen prompt tokens"
            )
        response_token_ids = self._continuation_token_ids(rendered, response)
        provided = dict(intervention_states or {})
        expected_keys = {(site, layer) for site in selected_sites for layer in selected_layers}
        for raw_key, state in provided.items():
            if (
                type(raw_key) is not tuple
                or len(raw_key) != 2
                or not isinstance(raw_key[0], ActivationSite)
                or type(raw_key[1]) is not int
                or raw_key not in expected_keys
                or not isinstance(state, MlxInterventionState)
            ):
                raise ConfigurationError(
                    "MLX teacher-forced cube intervention key or state is invalid"
                )
            _validate_fresh_intervention_state(
                state, expected_width=_research_hidden_width(self.base.model)
            )
        states = {key: provided.get(key, MlxInterventionState()) for key in expected_keys}
        if len({id(state) for state in states.values()}) != len(states):
            raise ConfigurationError("MLX teacher-forced cube requires one distinct state per hook")
        for state in states.values():
            if isinstance(state, MlxResearchInterventionState):
                state.arm_prompt(len(rendered.token_ids))
        mx, _nn, _load, _stream_generate = _mlx_modules()
        mx.reset_peak_memory()
        make_prompt_cache = _mlx_prompt_cache_factory()
        cache = make_prompt_cache(self.base.model)
        log_probabilities: list[float] = []
        trajectories: dict[tuple[ActivationSite, int], list[np.ndarray]] = {
            key: [] for key in expected_keys
        }
        hook_order = (
            ActivationSite.POST_MLP,
            ActivationSite.POST_ATTENTION,
            ActivationSite.BLOCK_OUTPUT,
        )
        with ExitStack() as stack:
            for layer in selected_layers:
                for site in hook_order:
                    if site in selected_sites:
                        stack.enter_context(
                            self.base.intervention(
                                layer=layer,
                                site=site,
                                state=states[(site, layer)],
                            )
                        )
            logits = self.base.model(mx.array([rendered.token_ids]), cache=cache)
            mx.eval(logits)
            if any(state.captured is None for state in states.values()):
                raise DataValidationError(
                    "MLX teacher-forced cube hook missed the prompt activation"
                )
            for token_id in response_token_ids:
                final_logits = np.asarray(logits[0, -1, :], dtype=np.float64)
                if (
                    final_logits.ndim != 1
                    or token_id >= final_logits.size
                    or not np.isfinite(final_logits).all()
                ):
                    raise DataValidationError("MLX continuation logits are invalid")
                maximum = float(final_logits.max())
                log_denominator = maximum + math.log(float(np.exp(final_logits - maximum).sum()))
                log_probabilities.append(float(final_logits[token_id] - log_denominator))
                for state in states.values():
                    state.captured = None
                    state.intervened = None
                logits = self.base.model(mx.array([[token_id]]), cache=cache)
                captured: dict[tuple[ActivationSite, int], Any] = {}
                for key, state in states.items():
                    if state.captured is None:
                        raise DataValidationError(
                            "MLX teacher-forced cube hook missed a response activation"
                        )
                    captured[key] = state.captured
                mx.eval(logits, *captured.values())
                for key, activation in captured.items():
                    trajectories[key].append(
                        np.asarray(activation[0, -1, :], dtype=np.float32).copy()
                    )
        nll = -sum(log_probabilities)
        mean_nll = nll / len(log_probabilities)
        try:
            perplexity = math.exp(mean_nll)
        except OverflowError as exc:
            raise DataValidationError("MLX continuation perplexity overflowed") from exc
        return MlxTeacherForcedCubeOutput(
            response_text_sha256=hashlib.sha256(response.encode("utf-8")).hexdigest(),
            response_token_ids=response_token_ids,
            response_token_ids_sha256=_token_ids_digest(response_token_ids),
            token_log_probabilities=tuple(log_probabilities),
            negative_log_likelihood=nll,
            mean_negative_log_likelihood=mean_nll,
            perplexity=perplexity,
            activations={
                site: {
                    layer: np.stack(trajectories[(site, layer)], axis=0)
                    for layer in selected_layers
                }
                for site in selected_sites
            },
            peak_memory_bytes=int(mx.get_peak_memory()),
        )

    def teacher_forced_continuation(
        self,
        rendered: MlxRenderedPrompt,
        response: str,
        *,
        layers: Sequence[int],
        site: ActivationSite,
        intervention_states: Mapping[int, MlxInterventionState] | None = None,
    ) -> MlxTeacherForcedOutput:
        """Score an exact response with a KV cache and capture each response token."""

        selected_layers = tuple(layers)
        if (
            not selected_layers
            or any(type(value) is not int for value in selected_layers)
            or len(set(selected_layers)) != len(selected_layers)
            or any(value < 0 or value >= len(self.base.model.layers) for value in selected_layers)
        ):
            raise ConfigurationError(
                "MLX continuation layers must be unique in-range exact integers"
            )
        if len(rendered.token_ids) < 2:
            raise DataValidationError(
                "MLX teacher forcing requires at least two frozen prompt tokens"
            )
        response_token_ids = self._continuation_token_ids(rendered, response)
        provided = dict(intervention_states or {})
        if any(type(layer) is not int or layer not in selected_layers for layer in provided) or any(
            not isinstance(state, MlxInterventionState) for state in provided.values()
        ):
            raise ConfigurationError(
                "MLX continuation intervention states must be fresh selected-layer states"
            )
        for state in provided.values():
            _validate_fresh_intervention_state(
                state, expected_width=_research_hidden_width(self.base.model)
            )
        if len({id(state) for state in provided.values()}) != len(provided):
            raise ConfigurationError(
                "MLX continuation requires one distinct intervention state per layer"
            )
        if any(not isinstance(state, MlxResearchInterventionState) for state in provided.values()):
            raise ConfigurationError("active MLX research continuation requires phase-aware states")
        for state in provided.values():
            assert isinstance(state, MlxResearchInterventionState)
            state.arm_prompt(len(rendered.token_ids))
        states = {layer: provided.get(layer, MlxInterventionState()) for layer in selected_layers}
        if len({id(state) for state in states.values()}) != len(states):
            raise ConfigurationError(
                "MLX continuation requires one distinct intervention state per layer"
            )
        mx, _nn, _load, _stream_generate = _mlx_modules()
        mx.reset_peak_memory()
        make_prompt_cache = _mlx_prompt_cache_factory()
        cache = make_prompt_cache(self.base.model)
        log_probabilities: list[float] = []
        trajectories: dict[int, list[np.ndarray]] = {layer: [] for layer in selected_layers}
        with ExitStack() as stack:
            for layer in selected_layers:
                stack.enter_context(
                    self.base.intervention(layer=layer, site=site, state=states[layer])
                )
            logits = self.base.model(mx.array([rendered.token_ids]), cache=cache)
            mx.eval(logits)
            if any(state.captured is None for state in states.values()):
                raise DataValidationError("MLX continuation hook missed the prompt activation")
            for token_id in response_token_ids:
                final_logits = np.asarray(logits[0, -1, :], dtype=np.float64)
                if (
                    final_logits.ndim != 1
                    or token_id >= final_logits.size
                    or not np.isfinite(final_logits).all()
                ):
                    raise DataValidationError("MLX continuation logits are invalid")
                maximum = float(final_logits.max())
                log_denominator = maximum + math.log(float(np.exp(final_logits - maximum).sum()))
                log_probabilities.append(float(final_logits[token_id] - log_denominator))
                for state in states.values():
                    state.captured = None
                    state.intervened = None
                logits = self.base.model(mx.array([[token_id]]), cache=cache)
                captured: list[Any] = []
                for layer in selected_layers:
                    activation = states[layer].captured
                    if activation is None:
                        raise DataValidationError(
                            "MLX continuation hook missed a response activation"
                        )
                    captured.append(activation)
                mx.eval(logits, *captured)
                for layer, activation in zip(selected_layers, captured, strict=True):
                    trajectories[layer].append(
                        np.asarray(activation[0, -1, :], dtype=np.float32).copy()
                    )
        nll = -sum(log_probabilities)
        mean_nll = nll / len(log_probabilities)
        try:
            perplexity = math.exp(mean_nll)
        except OverflowError as exc:
            raise DataValidationError("MLX continuation perplexity overflowed") from exc
        return MlxTeacherForcedOutput(
            response_text_sha256=hashlib.sha256(response.encode("utf-8")).hexdigest(),
            response_token_ids=response_token_ids,
            response_token_ids_sha256=_token_ids_digest(response_token_ids),
            token_log_probabilities=tuple(log_probabilities),
            negative_log_likelihood=nll,
            mean_negative_log_likelihood=mean_nll,
            perplexity=perplexity,
            activations={layer: np.stack(trajectories[layer], axis=0) for layer in selected_layers},
            peak_memory_bytes=int(mx.get_peak_memory()),
        )

    def close(self) -> None:
        self.base.close()

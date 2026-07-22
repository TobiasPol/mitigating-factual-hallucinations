from __future__ import annotations

import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from mfh.contracts import ActivationSite, TokenScope
from mfh.errors import ConfigurationError, DataValidationError
from mfh.inference.vllm_research import (
    VllmPromptFeatureCubeOutput,
    VllmResearchInterventionState,
    VllmResearchRuntime,
    VllmTeacherForcedCubeOutput,
)
from mfh.inference.vllm_runtime import VllmGenerationOutput, VllmRenderedPrompt


def _token_digest(token_ids: tuple[int, ...]) -> str:
    value = ",".join(str(token_id) for token_id in token_ids)
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _rendered() -> VllmRenderedPrompt:
    return VllmRenderedPrompt(
        text="prompt",
        sha256=hashlib.sha256(b"prompt").hexdigest(),
        token_ids=(1, 2, 3),
        token_ids_sha256=_token_digest((1, 2, 3)),
        messages=(),
    )


class _Base:
    model_spec = SimpleNamespace(num_layers=64)
    snapshot = Path(".")

    def generate_with_states(
        self,
        rendered: VllmRenderedPrompt,
        *,
        max_new_tokens: int,
        states: Any,
    ) -> tuple[VllmGenerationOutput, object, dict[str, object]]:
        assert max_new_tokens == 2
        assert all(state.phase_armed for state in states.values())
        return (
            VllmGenerationOutput(
                rendered_prompt=rendered,
                token_ids=(4,),
                text="answer",
                input_tokens=3,
                output_tokens=1,
                latency_seconds=0.1,
                stop_type="length",
                stopping_token_id=4,
                prompt_tokens_per_second=30.0,
                generation_tokens_per_second=10.0,
                peak_memory_bytes=100,
                active_memory_bytes=80,
                cache_memory_bytes=90,
            ),
            object(),
            {},
        )


def test_intervention_generation_validates_geometry_and_arms_prompt() -> None:
    runtime = VllmResearchRuntime(_Base())  # type: ignore[arg-type]
    direction = np.ones(5120, dtype=np.float32) / math.sqrt(5120)
    state = VllmResearchInterventionState(
        direction=direction,
        alpha=1.0,
        token_scope=TokenScope.FINAL_PROMPT,
    )

    generated = runtime.generate_with_interventions(
        _rendered(),
        max_new_tokens=2,
        intervention_states={(20, ActivationSite.BLOCK_OUTPUT): state},
    )

    assert generated.text == "answer"
    assert state.phase_armed is True
    assert state.prompt_tokens_remaining == 3


def test_intervention_generation_rejects_nonunit_or_out_of_range_geometry() -> None:
    runtime = VllmResearchRuntime(_Base())  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="direction geometry"):
        runtime.generate_with_interventions(
            _rendered(),
            max_new_tokens=2,
            intervention_states={
                (20, ActivationSite.POST_MLP): VllmResearchInterventionState(
                    direction=np.ones(5120, dtype=np.float32), alpha=1.0
                )
            },
        )
    with pytest.raises(ConfigurationError, match="key or state"):
        runtime.generate_with_interventions(
            _rendered(),
            max_new_tokens=2,
            intervention_states={(64, ActivationSite.POST_MLP): VllmResearchInterventionState()},
        )


def test_standardized_state_uses_qwen_hidden_width_and_reference_rms() -> None:
    runtime = VllmResearchRuntime(_Base())  # type: ignore[arg-type]
    direction = np.ones(5120, dtype=np.float32) / math.sqrt(5120)

    state = runtime.standardized_intervention_state(
        direction,
        standardized_alpha=2.0,
        reference_rms=0.25,
        token_scope=TokenScope.ALL_GENERATED,
    )

    assert state.alpha == 0.5
    assert state.token_scope is TokenScope.ALL_GENERATED
    assert state.direction is not direction


def test_prompt_cube_copies_arrays_and_requires_matching_layers() -> None:
    source = np.ones((1, 5120), dtype=np.float32)
    cube = VllmPromptFeatureCubeOutput(
        activations={
            ActivationSite.POST_MLP: {10: source},
            ActivationSite.BLOCK_OUTPUT: {10: source},
        },
        maximum_token_probability=0.75,
        output_entropy=1.0,
        peak_memory_bytes=10,
    )
    source[:] = 0
    assert np.all(cube.activations[ActivationSite.POST_MLP][10] == 1)
    assert cube.activations[ActivationSite.POST_MLP][10].flags.writeable is False

    with pytest.raises(DataValidationError, match="layers differ"):
        VllmPromptFeatureCubeOutput(
            activations={
                ActivationSite.POST_MLP: {10: np.ones((1, 2))},
                ActivationSite.BLOCK_OUTPUT: {19: np.ones((1, 2))},
            },
            maximum_token_probability=0.5,
            output_entropy=1.0,
            peak_memory_bytes=1,
        )


def test_teacher_forced_cube_recomputes_likelihood_and_geometry() -> None:
    token_ids = (7, 8)
    logprobs = (-0.2, -0.3)
    cube = VllmTeacherForcedCubeOutput(
        response_text_sha256=hashlib.sha256(b"answer").hexdigest(),
        response_token_ids=token_ids,
        response_token_ids_sha256=_token_digest(token_ids),
        token_log_probabilities=logprobs,
        negative_log_likelihood=0.5,
        mean_negative_log_likelihood=0.25,
        perplexity=math.exp(0.25),
        activations={
            ActivationSite.POST_MLP: {20: np.ones((2, 5120), dtype=np.float32)},
            ActivationSite.BLOCK_OUTPUT: {20: np.ones((2, 5120), dtype=np.float32)},
        },
        peak_memory_bytes=100,
    )
    assert cube.perplexity == pytest.approx(math.exp(0.25))

    with pytest.raises(DataValidationError, match="aggregate likelihood"):
        VllmTeacherForcedCubeOutput(
            response_text_sha256=hashlib.sha256(b"answer").hexdigest(),
            response_token_ids=token_ids,
            response_token_ids_sha256=_token_digest(token_ids),
            token_log_probabilities=logprobs,
            negative_log_likelihood=0.4,
            mean_negative_log_likelihood=0.2,
            perplexity=math.exp(0.2),
            activations={
                ActivationSite.BLOCK_OUTPUT: {
                    20: np.ones((2, 5120), dtype=np.float32)
                }
            },
        )

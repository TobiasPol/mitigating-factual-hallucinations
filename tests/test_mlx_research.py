from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from mfh.config import load_model_spec, load_prompt_specs
from mfh.contracts import ActivationSite, TokenScope
from mfh.errors import ConfigurationError, DataValidationError
from mfh.inference.mlx_research import (
    MlxPromptFeatureCubeOutput,
    MlxPromptFeatureOutput,
    MlxResearchInterventionState,
    MlxResearchRuntime,
    MlxTeacherForcedCubeOutput,
    MlxTeacherForcedOutput,
)
from mfh.inference.mlx_runtime import (
    MlxInterventionState,
    MlxRuntime,
    _capture_and_intervene,
)

ROOT = Path(__file__).parents[1]


class _Tokenizer:
    def apply_chat_template(self, messages, *, tokenize: bool, **kwargs):  # type: ignore[no-untyped-def]
        assert kwargs == {"add_generation_prompt": True, "enable_thinking": False}
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        return [7, 11, 13] if tokenize else "<system>system</system><user>question</user>"

    def encode(self, text: str, **kwargs):  # type: ignore[no-untyped-def]
        assert kwargs in ({"add_special_tokens": False}, {})
        prefix = "<system>system</system><user>question</user>"
        if text == prefix + "AB":
            return [7, 11, 13, 1, 2]
        return [5, 6]


class _Model:
    def __init__(self) -> None:
        self.layers = [object() for _ in range(64)]
        self.calls = 0

    def __call__(self, _tokens):  # type: ignore[no-untyped-def]
        self.calls += 1
        return np.asarray([[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 1.0, 0.0]]])


class _FakeMx:
    @staticmethod
    def array(value):  # type: ignore[no-untyped-def]
        return np.asarray(value)

    @staticmethod
    def eval(*_values):  # type: ignore[no-untyped-def]
        return None

    @staticmethod
    def reset_peak_memory():
        return None

    @staticmethod
    def get_peak_memory():
        return 123


class _LikelihoodModel:
    def __init__(self) -> None:
        self.layers = [object() for _ in range(64)]
        self.active: dict[int, object] = {}
        self.calls = 0

    def __call__(self, tokens, *, cache=None):  # type: ignore[no-untyped-def]
        assert cache is not None
        sequence_length = int(np.asarray(tokens).shape[1])
        for layer, state in self.active.items():
            state.captured = np.full(  # type: ignore[attr-defined]
                (1, sequence_length, 4),
                float(layer + self.calls),
                dtype=np.float32,
            )
        values = np.asarray([0.0, 1.0, 2.0]) if self.calls == 0 else np.asarray([2.0, 0.0, 1.0])
        self.calls += 1
        return np.broadcast_to(values, (1, sequence_length, 3)).copy()


class _CubeLikelihoodModel:
    def __init__(self) -> None:
        self.layers = [object() for _ in range(64)]
        self.active: dict[tuple[ActivationSite, int], object] = {}
        self.calls = 0

    def __call__(self, tokens, *, cache=None):  # type: ignore[no-untyped-def]
        assert cache is not None
        sequence_length = int(np.asarray(tokens).shape[1])
        for (site, layer), state in self.active.items():
            site_index = list(ActivationSite).index(site)
            state.captured = np.full(  # type: ignore[attr-defined]
                (1, sequence_length, 4),
                float(layer + site_index + self.calls),
                dtype=np.float32,
            )
        values = np.asarray([0.0, 1.0, 2.0]) if self.calls == 0 else np.asarray([2.0, 0.0, 1.0])
        self.calls += 1
        return np.broadcast_to(values, (1, sequence_length, 3)).copy()


class _MissingCubeLikelihoodModel(_CubeLikelihoodModel):
    def __call__(self, tokens, *, cache=None):  # type: ignore[no-untyped-def]
        key = (ActivationSite.POST_MLP, 16)
        skipped = self.active.pop(key, None) if self.calls == 2 else None
        try:
            return super().__call__(tokens, cache=cache)
        finally:
            if skipped is not None:
                self.active[key] = skipped


def _runtime() -> MlxResearchRuntime:
    model = load_model_spec(ROOT / "configs/models/qwen3.6-27b-mlx-4bit.yaml")
    return MlxResearchRuntime(
        MlxRuntime(
            model=_Model(),
            tokenizer=_Tokenizer(),
            model_spec=model,
            snapshot=ROOT,
        )
    )


def _likelihood_runtime() -> MlxResearchRuntime:
    model_spec = load_model_spec(ROOT / "configs/models/qwen3.6-27b-mlx-4bit.yaml")
    return MlxResearchRuntime(
        MlxRuntime(
            model=_LikelihoodModel(),
            tokenizer=_Tokenizer(),
            model_spec=model_spec,
            snapshot=ROOT,
        )
    )


def _cube_likelihood_runtime(*, miss_hook: bool = False) -> MlxResearchRuntime:
    model_spec = load_model_spec(ROOT / "configs/models/qwen3.6-27b-mlx-4bit.yaml")
    return MlxResearchRuntime(
        MlxRuntime(
            model=_MissingCubeLikelihoodModel() if miss_hook else _CubeLikelihoodModel(),
            tokenizer=_Tokenizer(),
            model_spec=model_spec,
            snapshot=ROOT,
        )
    )


def test_prompt_feature_bundle_captures_layers_and_confidence(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "mfh.inference.mlx_research._mlx_modules",
        lambda: (_FakeMx(), object(), object(), object()),
    )
    runtime = _runtime()

    @contextmanager
    def fake_intervention(self, *, layer, site, state):  # type: ignore[no-untyped-def]
        assert site is ActivationSite.POST_MLP
        state.captured = np.full((1, 3, 4), float(layer), dtype=np.float32)
        yield state

    runtime.base.intervention = MethodType(fake_intervention, runtime.base)
    prompt = {
        value.prompt_id: value for value in load_prompt_specs(ROOT / "configs/prompts/primary.yaml")
    }["P0-neutral"]
    rendered = runtime.base.render_prompt(prompt, "What is the capital of France?")

    result = runtime.prompt_feature_bundle(
        rendered,
        layers=(16, 31, 63),
        site=ActivationSite.POST_MLP,
    )

    assert tuple(result.activations) == (16, 31, 63)
    assert result.activations[31].shape == (1, 4)
    assert np.all(result.activations[31] == 31)
    assert result.activations[31].flags.writeable is False
    probabilities = np.exp(np.asarray([2.0, 1.0, 0.0]))
    probabilities /= probabilities.sum()
    assert result.maximum_token_probability == pytest.approx(float(probabilities.max()))
    assert result.output_entropy == pytest.approx(
        -float(np.sum(probabilities * np.log(probabilities)))
    )


def test_prompt_feature_bundle_rejects_duplicate_layers() -> None:
    runtime = _runtime()
    prompt = {
        value.prompt_id: value for value in load_prompt_specs(ROOT / "configs/prompts/primary.yaml")
    }["P0-neutral"]

    with pytest.raises(ConfigurationError, match="unique"):
        runtime.prompt_feature_bundle(
            runtime.base.render_prompt(prompt, "Question?"),
            layers=(16, 16),
            site=ActivationSite.POST_MLP,
        )


def test_prompt_feature_cube_captures_all_sites_in_one_forward(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "mfh.inference.mlx_research._mlx_modules",
        lambda: (_FakeMx(), object(), object(), object()),
    )
    runtime = _runtime()
    entered: list[tuple[int, ActivationSite]] = []

    @contextmanager
    def fake_intervention(self, *, layer, site, state):  # type: ignore[no-untyped-def]
        entered.append((layer, site))
        state.captured = np.full(
            (1, 3, 4), float(layer + list(ActivationSite).index(site)), dtype=np.float32
        )
        yield state

    runtime.base.intervention = MethodType(fake_intervention, runtime.base)
    prompt = {
        value.prompt_id: value for value in load_prompt_specs(ROOT / "configs/prompts/primary.yaml")
    }["P0-neutral"]
    rendered = runtime.base.render_prompt(prompt, "Question?")
    sites = (
        ActivationSite.POST_ATTENTION,
        ActivationSite.POST_MLP,
        ActivationSite.BLOCK_OUTPUT,
    )
    result = runtime.prompt_feature_cube(rendered, layers=(16, 31), sites=sites)

    assert runtime.base.model.calls == 1
    assert tuple(result.activations) == sites
    assert entered == [
        (16, ActivationSite.POST_MLP),
        (16, ActivationSite.POST_ATTENTION),
        (16, ActivationSite.BLOCK_OUTPUT),
        (31, ActivationSite.POST_MLP),
        (31, ActivationSite.POST_ATTENTION),
        (31, ActivationSite.BLOCK_OUTPUT),
    ]
    assert result.activations[ActivationSite.BLOCK_OUTPUT][31].shape == (1, 4)
    assert result.peak_memory_bytes == 123


def test_prompt_feature_cube_output_rejects_mismatched_layers() -> None:
    with pytest.raises(DataValidationError, match="layers differ"):
        MlxPromptFeatureCubeOutput(
            activations={
                ActivationSite.POST_MLP: {1: np.ones((1, 4))},
                ActivationSite.BLOCK_OUTPUT: {2: np.ones((1, 4))},
            },
            maximum_token_probability=0.5,
            output_entropy=0.5,
            peak_memory_bytes=1,
        )


def test_generate_with_interventions_orders_hooks_and_restores_after_generation() -> None:
    runtime = _runtime()
    events: list[tuple[str, int, ActivationSite]] = []

    @contextmanager
    def fake_intervention(self, *, layer, site, state):  # type: ignore[no-untyped-def]
        events.append(("enter", layer, site))
        try:
            yield state
        finally:
            events.append(("exit", layer, site))

    def fake_generate(self, rendered, *, max_new_tokens):  # type: ignore[no-untyped-def]
        assert max_new_tokens == 12
        events.append(("generate", -1, ActivationSite.BLOCK_OUTPUT))
        return rendered

    runtime.base.intervention = MethodType(fake_intervention, runtime.base)
    runtime.base.generate = MethodType(fake_generate, runtime.base)
    prompt = {
        value.prompt_id: value for value in load_prompt_specs(ROOT / "configs/prompts/primary.yaml")
    }["P0-neutral"]
    rendered = runtime.base.render_prompt(prompt, "Question?")
    states = {
        (31, ActivationSite.BLOCK_OUTPUT): MlxResearchInterventionState(
            direction=np.ones(4) / 2, alpha=0.5, token_scope=TokenScope.FIRST_FOUR
        ),
        (16, ActivationSite.POST_ATTENTION): MlxResearchInterventionState(
            direction=np.ones(4) / 2, alpha=0.5, token_scope=TokenScope.FIRST_FOUR
        ),
        (16, ActivationSite.POST_MLP): MlxResearchInterventionState(
            direction=np.ones(4) / 2, alpha=0.5, token_scope=TokenScope.FIRST_FOUR
        ),
    }

    result = runtime.generate_with_interventions(
        rendered, max_new_tokens=12, intervention_states=states
    )

    assert result is rendered
    assert events == [
        ("enter", 16, ActivationSite.POST_MLP),
        ("enter", 16, ActivationSite.POST_ATTENTION),
        ("enter", 31, ActivationSite.BLOCK_OUTPUT),
        ("generate", -1, ActivationSite.BLOCK_OUTPUT),
        ("exit", 31, ActivationSite.BLOCK_OUTPUT),
        ("exit", 16, ActivationSite.POST_ATTENTION),
        ("exit", 16, ActivationSite.POST_MLP),
    ]


def test_generate_with_interventions_rejects_reused_or_shared_states() -> None:
    runtime = _runtime()
    prompt = {
        value.prompt_id: value for value in load_prompt_specs(ROOT / "configs/prompts/primary.yaml")
    }["P0-neutral"]
    rendered = runtime.base.render_prompt(prompt, "Question?")
    reused = MlxInterventionState(generated_calls=1)
    with pytest.raises(ConfigurationError, match="fresh and finite"):
        runtime.generate_with_interventions(
            rendered,
            max_new_tokens=1,
            intervention_states={(16, ActivationSite.POST_MLP): reused},
        )
    shared = MlxInterventionState()
    with pytest.raises(ConfigurationError, match="distinct state"):
        runtime.generate_with_interventions(
            rendered,
            max_new_tokens=1,
            intervention_states={
                (16, ActivationSite.POST_MLP): shared,
                (31, ActivationSite.POST_MLP): shared,
            },
        )


@pytest.mark.parametrize(
    ("continue_generation", "expected_text", "expected_stop"),
    ((False, "AB", "online_gate"), (True, "ABC", "eos")),
)
def test_online_gate_pauses_one_stream_before_deciding_to_continue(
    monkeypatch: pytest.MonkeyPatch,
    continue_generation: bool,
    expected_text: str,
    expected_stop: str,
) -> None:
    runtime = _runtime()
    active: dict[tuple[int, ActivationSite], MlxResearchInterventionState] = {}
    generated_steps: list[int] = []

    class OnlineMx(_FakeMx):
        class random:
            @staticmethod
            def seed(_value: int) -> None:
                return None

        @staticmethod
        def get_active_memory() -> int:
            return 64

        @staticmethod
        def get_cache_memory() -> int:
            return 32

    mx = OnlineMx()

    @contextmanager
    def fake_intervention(self, *, layer, site, state):  # type: ignore[no-untyped-def]
        key = (layer, site)
        active[key] = state
        try:
            yield state
        finally:
            del active[key]

    def stream_generate(_model, _tokenizer, _tokens, *, max_tokens):  # type: ignore[no-untyped-def]
        assert max_tokens == 3
        for state in active.values():
            _capture_and_intervene(np.ones((1, 3, 4)), state, mx)
        for index, text in enumerate(("A", "B", "C"), start=1):
            generated_steps.append(index)
            for state in active.values():
                _capture_and_intervene(
                    np.full((1, 1, 4), float(index), dtype=np.float32), state, mx
                )
            yield SimpleNamespace(
                text=text,
                token=index,
                finish_reason=("eos" if index == 3 else None),
                prompt_tokens=3,
                generation_tokens=index,
                prompt_tps=100.0,
                generation_tps=50.0,
            )

    monkeypatch.setattr(
        "mfh.inference.mlx_research._mlx_modules",
        lambda: (mx, object(), object(), stream_generate),
    )
    runtime.base.intervention = MethodType(fake_intervention, runtime.base)
    prompt = {
        value.prompt_id: value for value in load_prompt_specs(ROOT / "configs/prompts/primary.yaml")
    }["P0-neutral"]
    rendered = runtime.base.render_prompt(prompt, "Question?")
    state = MlxResearchInterventionState(capture_limit=2)
    captures: list[tuple[str, tuple[int, ...], np.ndarray]] = []

    def gate(capture):  # type: ignore[no-untyped-def]
        captures.append(
            (
                capture.text,
                capture.token_ids,
                capture.activations[ActivationSite.POST_MLP][16],
            )
        )
        return continue_generation

    result = runtime.generate_with_online_gate(
        rendered,
        max_new_tokens=3,
        intervention_states={(16, ActivationSite.POST_MLP): state},
        capture_keys=((16, ActivationSite.POST_MLP),),
        feature_token_count=2,
        early_gate=gate,
    )

    assert result.generation.text == expected_text
    assert result.generation.stop_type == expected_stop
    assert result.early_gate_applied is True
    assert result.continued_after_early_gate is continue_generation
    assert result.buffered_token_count_at_gate == 2
    assert captures[0][0:2] == ("AB", (1, 2))
    assert captures[0][2].shape == (2, 4)
    assert generated_steps == ([1, 2, 3] if continue_generation else [1, 2])


def test_online_gate_does_not_claim_control_on_eos_at_capture_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime()
    active: list[MlxResearchInterventionState] = []

    class OnlineMx(_FakeMx):
        class random:
            @staticmethod
            def seed(_value: int) -> None:
                return None

        get_active_memory = staticmethod(lambda: 64)
        get_cache_memory = staticmethod(lambda: 32)

    mx = OnlineMx()

    @contextmanager
    def fake_intervention(self, *, layer, site, state):  # type: ignore[no-untyped-def]
        del self, layer, site
        active.append(state)
        try:
            yield state
        finally:
            active.remove(state)

    def stream_generate(_model, _tokenizer, _tokens, *, max_tokens):  # type: ignore[no-untyped-def]
        assert max_tokens == 1
        for state in active:
            _capture_and_intervene(np.ones((1, 3, 4)), state, mx)
            _capture_and_intervene(np.ones((1, 1, 4)), state, mx)
        yield SimpleNamespace(
            text="Paris",
            token=1,
            finish_reason="eos",
            prompt_tokens=3,
            generation_tokens=1,
            prompt_tps=100.0,
            generation_tps=50.0,
        )

    monkeypatch.setattr(
        "mfh.inference.mlx_research._mlx_modules",
        lambda: (mx, object(), object(), stream_generate),
    )
    runtime.base.intervention = MethodType(fake_intervention, runtime.base)
    prompt = {
        value.prompt_id: value for value in load_prompt_specs(ROOT / "configs/prompts/primary.yaml")
    }["P0-neutral"]
    rendered = runtime.base.render_prompt(prompt, "Question?")
    state = MlxResearchInterventionState(capture_limit=1)

    result = runtime.generate_with_online_gate(
        rendered,
        max_new_tokens=1,
        intervention_states={(16, ActivationSite.POST_MLP): state},
        capture_keys=((16, ActivationSite.POST_MLP),),
        feature_token_count=1,
        early_gate=lambda _capture: pytest.fail("EOS cannot invoke the online gate"),
    )

    assert result.generation.stop_type == "eos"
    assert result.early_gate_applied is False
    assert result.feature_token_count == 1
    assert result.buffered_token_count_at_gate == 0


@pytest.mark.parametrize(
    ("scope", "expected"),
    [
        (TokenScope.FINAL_PROMPT, (0.0, 2.0, 0.0, 0.0, 0.0, 0.0)),
        (TokenScope.FIRST_GENERATED, (0.0, 0.0, 2.0, 0.0, 0.0, 0.0)),
        (TokenScope.FIRST_FOUR, (0.0, 0.0, 2.0, 2.0, 2.0, 2.0)),
        (TokenScope.ALL_GENERATED, (0.0, 0.0, 2.0, 2.0, 2.0, 2.0)),
    ],
)
def test_research_intervention_tracks_mlx_lm_split_prefill_explicitly(
    scope: TokenScope, expected: tuple[float, ...]
) -> None:
    state = MlxResearchInterventionState(
        direction=np.ones(4, dtype=np.float32) / 2,
        alpha=2.0,
        token_scope=scope,
    )
    state.arm_prompt(5)

    observed = (
        state.effective_alpha(4),  # mlx-lm prefill of prompt[:-1]
        state.effective_alpha(1),  # actual final prompt token
        state.effective_alpha(1),  # first generated-token activation
        state.effective_alpha(1),
        state.effective_alpha(1),
        state.effective_alpha(1),
    )

    assert observed == expected
    assert state.prompt_tokens_remaining == 0
    assert state.generated_calls == 4


def test_generate_with_interventions_rejects_missing_or_nonunit_direction() -> None:
    runtime = _runtime()
    prompt = {
        value.prompt_id: value for value in load_prompt_specs(ROOT / "configs/prompts/primary.yaml")
    }["P0-neutral"]
    rendered = runtime.base.render_prompt(prompt, "Question?")
    with pytest.raises(ConfigurationError, match="requires a direction"):
        runtime.generate_with_interventions(
            rendered,
            max_new_tokens=1,
            intervention_states={(16, ActivationSite.POST_MLP): MlxInterventionState(alpha=1.0)},
        )
    with pytest.raises(ConfigurationError, match="direction geometry"):
        runtime.generate_with_interventions(
            rendered,
            max_new_tokens=1,
            intervention_states={
                (16, ActivationSite.POST_MLP): MlxInterventionState(direction=np.ones(4), alpha=1.0)
            },
        )


@pytest.mark.parametrize("layer", [True, 1.9, "2"])
def test_prompt_feature_bundle_rejects_coercible_layer_identifiers(layer: object) -> None:
    runtime = _runtime()
    prompt = {
        value.prompt_id: value for value in load_prompt_specs(ROOT / "configs/prompts/primary.yaml")
    }["P0-neutral"]

    with pytest.raises(ConfigurationError, match="exact integers"):
        runtime.prompt_feature_bundle(
            runtime.base.render_prompt(prompt, "Question?"),
            layers=(layer,),  # type: ignore[arg-type]
            site=ActivationSite.POST_MLP,
        )


@pytest.mark.parametrize("layer", [True, 1.9, "2"])
def test_prompt_feature_output_rejects_coercible_layer_identifiers(layer: object) -> None:
    with pytest.raises(DataValidationError, match="exact integers"):
        MlxPromptFeatureOutput(
            activations={layer: np.ones((1, 4), dtype=np.float32)},  # type: ignore[dict-item]
            maximum_token_probability=0.5,
            output_entropy=0.5,
        )


def test_teacher_forced_continuation_scores_and_captures_response_tokens(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "mfh.inference.mlx_research._mlx_modules",
        lambda: (_FakeMx(), object(), object(), object()),
    )
    monkeypatch.setattr(
        "mfh.inference.mlx_research._mlx_prompt_cache_factory",
        lambda: lambda _model: [object()],
    )
    runtime = _likelihood_runtime()
    model = runtime.base.model

    @contextmanager
    def fake_intervention(self, *, layer, site, state):  # type: ignore[no-untyped-def]
        assert site is ActivationSite.POST_MLP
        model.active[layer] = state
        try:
            yield state
        finally:
            del model.active[layer]

    runtime.base.intervention = MethodType(fake_intervention, runtime.base)
    prompt = {
        value.prompt_id: value for value in load_prompt_specs(ROOT / "configs/prompts/primary.yaml")
    }["P0-neutral"]
    rendered = runtime.base.render_prompt(prompt, "Question?")

    result = runtime.teacher_forced_continuation(
        rendered,
        "AB",
        layers=(16, 31),
        site=ActivationSite.POST_MLP,
    )

    denominator = float(np.exp(np.asarray([0.0, 1.0, 2.0])).sum())
    expected_log_probability = 1.0 - np.log(denominator)
    assert result.response_token_ids == (1, 2)
    assert result.token_log_probabilities == pytest.approx(
        (expected_log_probability, expected_log_probability)
    )
    assert result.negative_log_likelihood == pytest.approx(-2 * expected_log_probability)
    assert result.perplexity == pytest.approx(np.exp(-expected_log_probability))
    assert model.calls == 3
    assert model.active == {}
    assert result.activations[16].shape == (2, 4)
    assert np.all(result.activations[16][0] == 17)
    assert np.all(result.activations[16][1] == 18)
    assert result.activations[16].flags.writeable is False


def test_teacher_forced_cube_captures_all_sites_in_one_cached_pass(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "mfh.inference.mlx_research._mlx_modules",
        lambda: (_FakeMx(), object(), object(), object()),
    )
    monkeypatch.setattr(
        "mfh.inference.mlx_research._mlx_prompt_cache_factory",
        lambda: lambda _model: [object()],
    )
    runtime = _cube_likelihood_runtime()
    model = runtime.base.model
    entered: list[tuple[int, ActivationSite]] = []

    @contextmanager
    def fake_intervention(self, *, layer, site, state):  # type: ignore[no-untyped-def]
        entered.append((layer, site))
        model.active[(site, layer)] = state
        try:
            yield state
        finally:
            del model.active[(site, layer)]

    runtime.base.intervention = MethodType(fake_intervention, runtime.base)
    prompt = {
        value.prompt_id: value for value in load_prompt_specs(ROOT / "configs/prompts/primary.yaml")
    }["P0-neutral"]
    sites = (
        ActivationSite.POST_ATTENTION,
        ActivationSite.POST_MLP,
        ActivationSite.BLOCK_OUTPUT,
    )
    result = runtime.teacher_forced_cube(
        runtime.base.render_prompt(prompt, "Question?"),
        "AB",
        layers=(16, 31),
        sites=sites,
    )

    assert isinstance(result, MlxTeacherForcedCubeOutput)
    assert model.calls == 3
    assert model.active == {}
    assert entered == [
        (16, ActivationSite.POST_MLP),
        (16, ActivationSite.POST_ATTENTION),
        (16, ActivationSite.BLOCK_OUTPUT),
        (31, ActivationSite.POST_MLP),
        (31, ActivationSite.POST_ATTENTION),
        (31, ActivationSite.BLOCK_OUTPUT),
    ]
    assert result.response_token_ids == (1, 2)
    assert result.activations[ActivationSite.POST_MLP][16].shape == (2, 4)
    assert np.all(result.activations[ActivationSite.POST_MLP][16][0] == 18)
    assert result.activations[ActivationSite.POST_MLP][16].flags.writeable is False


def test_teacher_forced_cube_arms_research_interventions_before_hook_entry(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "mfh.inference.mlx_research._mlx_modules",
        lambda: (_FakeMx(), object(), object(), object()),
    )
    monkeypatch.setattr(
        "mfh.inference.mlx_research._mlx_prompt_cache_factory",
        lambda: lambda _model: [object()],
    )
    runtime = _cube_likelihood_runtime()
    model = runtime.base.model
    research_state = MlxResearchInterventionState(
        direction=np.full(4, 0.5, dtype=np.float32),
        alpha=0.5,
        token_scope=TokenScope.FIRST_FOUR,
    )

    @contextmanager
    def fake_intervention(self, *, layer, site, state):  # type: ignore[no-untyped-def]
        assert layer == 16
        assert site is ActivationSite.POST_MLP
        assert state is research_state
        assert research_state.phase_armed is True
        assert research_state.prompt_tokens_remaining == 3
        model.active[(site, layer)] = state
        try:
            yield state
        finally:
            del model.active[(site, layer)]

    runtime.base.intervention = MethodType(fake_intervention, runtime.base)
    prompt = {
        value.prompt_id: value for value in load_prompt_specs(ROOT / "configs/prompts/primary.yaml")
    }["P0-neutral"]
    runtime.teacher_forced_cube(
        runtime.base.render_prompt(prompt, "Question?"),
        "AB",
        layers=(16,),
        sites=(ActivationSite.POST_MLP,),
        intervention_states={(ActivationSite.POST_MLP, 16): research_state},
    )


def test_teacher_forced_cube_rejects_coercible_keys_and_nonfinite_states(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "mfh.inference.mlx_research._mlx_modules",
        lambda: (_FakeMx(), object(), object(), object()),
    )
    monkeypatch.setattr(
        "mfh.inference.mlx_research._mlx_prompt_cache_factory",
        lambda: lambda _model: [object()],
    )
    runtime = _cube_likelihood_runtime()
    prompt = {
        value.prompt_id: value for value in load_prompt_specs(ROOT / "configs/prompts/primary.yaml")
    }["P0-neutral"]
    rendered = runtime.base.render_prompt(prompt, "Question?")
    with pytest.raises(ConfigurationError, match="key or state"):
        runtime.teacher_forced_cube(
            rendered,
            "AB",
            layers=(1,),
            sites=(ActivationSite.POST_MLP,),
            intervention_states={(ActivationSite.POST_MLP, True): MlxInterventionState()},
        )
    with pytest.raises(ConfigurationError, match="fresh and finite"):
        runtime.teacher_forced_cube(
            rendered,
            "AB",
            layers=(1,),
            sites=(ActivationSite.POST_MLP,),
            intervention_states={
                (ActivationSite.POST_MLP, 1): MlxInterventionState(alpha=float("nan"))
            },
        )


def test_teacher_forced_cube_rejects_a_stale_capture_when_hook_misses_step(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "mfh.inference.mlx_research._mlx_modules",
        lambda: (_FakeMx(), object(), object(), object()),
    )
    monkeypatch.setattr(
        "mfh.inference.mlx_research._mlx_prompt_cache_factory",
        lambda: lambda _model: [object()],
    )
    runtime = _cube_likelihood_runtime(miss_hook=True)
    model = runtime.base.model

    @contextmanager
    def fake_intervention(self, *, layer, site, state):  # type: ignore[no-untyped-def]
        model.active[(site, layer)] = state
        try:
            yield state
        finally:
            del model.active[(site, layer)]

    runtime.base.intervention = MethodType(fake_intervention, runtime.base)
    prompt = {
        value.prompt_id: value for value in load_prompt_specs(ROOT / "configs/prompts/primary.yaml")
    }["P0-neutral"]
    with pytest.raises(DataValidationError, match="missed a response activation"):
        runtime.teacher_forced_cube(
            runtime.base.render_prompt(prompt, "Question?"),
            "AB",
            layers=(16,),
            sites=(ActivationSite.POST_MLP,),
        )


def test_teacher_forced_continuation_rejects_nonprefix_tokenization(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "mfh.inference.mlx_research._mlx_modules",
        lambda: (_FakeMx(), object(), object(), object()),
    )
    monkeypatch.setattr(
        "mfh.inference.mlx_research._mlx_prompt_cache_factory",
        lambda: lambda _model: [object()],
    )
    runtime = _likelihood_runtime()
    prompt = {
        value.prompt_id: value for value in load_prompt_specs(ROOT / "configs/prompts/primary.yaml")
    }["P0-neutral"]

    with pytest.raises(DataValidationError, match="unambiguous extension"):
        runtime.teacher_forced_continuation(
            runtime.base.render_prompt(prompt, "Question?"),
            "not-tokenized-as-prefix",
            layers=(16,),
            site=ActivationSite.POST_MLP,
        )


def test_teacher_forced_continuation_rejects_shared_layer_state() -> None:
    runtime = _likelihood_runtime()
    prompt = {
        value.prompt_id: value for value in load_prompt_specs(ROOT / "configs/prompts/primary.yaml")
    }["P0-neutral"]
    shared = MlxInterventionState()

    with pytest.raises(ConfigurationError, match="distinct intervention state"):
        runtime.teacher_forced_continuation(
            runtime.base.render_prompt(prompt, "Question?"),
            "AB",
            layers=(16, 31),
            site=ActivationSite.POST_MLP,
            intervention_states={16: shared, 31: shared},
        )


def test_teacher_forced_continuation_requires_phase_aware_active_state() -> None:
    runtime = _likelihood_runtime()
    prompt = {
        value.prompt_id: value for value in load_prompt_specs(ROOT / "configs/prompts/primary.yaml")
    }["P0-neutral"]

    with pytest.raises(ConfigurationError, match="phase-aware states"):
        runtime.teacher_forced_continuation(
            runtime.base.render_prompt(prompt, "Question?"),
            "AB",
            layers=(16,),
            site=ActivationSite.POST_MLP,
            intervention_states={16: MlxInterventionState()},
        )


def test_teacher_forced_output_freezes_tokens_and_rejects_coercion() -> None:
    token_ids = [1]
    output = MlxTeacherForcedOutput(
        response_text_sha256="a" * 64,
        response_token_ids=token_ids,  # type: ignore[arg-type]
        response_token_ids_sha256="6b86b273ff34fce19d6b804eff5a3f5747ada4eaa22f1d49c01e52ddb7875b4b",
        token_log_probabilities=(-0.5,),
        negative_log_likelihood=0.5,
        mean_negative_log_likelihood=0.5,
        perplexity=float(np.exp(0.5)),
        activations={1: np.ones((1, 4), dtype=np.float32)},
        peak_memory_bytes=123,
    )
    token_ids.append(2)
    assert output.response_token_ids == (1,)
    assert output.peak_memory_bytes == 123

    with pytest.raises(DataValidationError, match="response identity"):
        MlxTeacherForcedOutput(
            response_text_sha256="a" * 64,
            response_token_ids=(1,),
            response_token_ids_sha256=output.response_token_ids_sha256,
            token_log_probabilities=(-0.5,),
            negative_log_likelihood=0.5,
            mean_negative_log_likelihood=0.5,
            perplexity=float(np.exp(0.5)),
            activations={1: np.ones((1, 4), dtype=np.float32)},
            peak_memory_bytes=-1,
        )

    with pytest.raises(DataValidationError, match="response identity"):
        MlxTeacherForcedOutput(
            response_text_sha256="z" * 64,
            response_token_ids=(1,),
            response_token_ids_sha256=output.response_token_ids_sha256,
            token_log_probabilities=(-0.5,),
            negative_log_likelihood=0.5,
            mean_negative_log_likelihood=0.5,
            perplexity=float(np.exp(0.5)),
            activations={1: np.ones((1, 4), dtype=np.float32)},
        )
    with pytest.raises(DataValidationError, match="log probabilities"):
        MlxTeacherForcedOutput(
            response_text_sha256="a" * 64,
            response_token_ids=(1,),
            response_token_ids_sha256=output.response_token_ids_sha256,
            token_log_probabilities=("-0.5",),  # type: ignore[arg-type]
            negative_log_likelihood=0.5,
            mean_negative_log_likelihood=0.5,
            perplexity=float(np.exp(0.5)),
            activations={1: np.ones((1, 4), dtype=np.float32)},
        )
    with pytest.raises(DataValidationError, match="activation layers"):
        MlxTeacherForcedOutput(
            response_text_sha256="a" * 64,
            response_token_ids=(1,),
            response_token_ids_sha256=output.response_token_ids_sha256,
            token_log_probabilities=(-0.5,),
            negative_log_likelihood=0.5,
            mean_negative_log_likelihood=0.5,
            perplexity=float(np.exp(0.5)),
            activations={-1: np.ones((1, 4), dtype=np.float32)},
        )

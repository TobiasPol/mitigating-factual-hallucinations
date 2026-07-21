from __future__ import annotations

from pathlib import Path

import numpy as np

from mfh.config import load_model_spec, load_prompt_specs
from mfh.inference.mlx_runtime import MlxRuntime, _completed_short_answer, as_numpy

ROOT = Path(__file__).parents[1]


class _Tokenizer:
    def apply_chat_template(self, messages, *, tokenize: bool, **kwargs):  # type: ignore[no-untyped-def]
        assert kwargs == {"add_generation_prompt": True, "enable_thinking": False}
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        return [7, 11, 13] if tokenize else "<system>system</system><user>question</user>"


class _Model:
    def __init__(self) -> None:
        self.layers = [object() for _ in range(64)]


def test_mlx_runtime_renders_official_template_with_frozen_tokens() -> None:
    model = load_model_spec(ROOT / "configs/models/qwen3.6-27b-mlx-4bit.yaml")
    prompt = {
        value.prompt_id: value
        for value in load_prompt_specs(ROOT / "configs/prompts/primary.yaml")
    }["P0-neutral"]
    runtime = MlxRuntime(
        model=_Model(),
        tokenizer=_Tokenizer(),
        model_spec=model,
        snapshot=ROOT,
    )

    rendered = runtime.render_prompt(prompt, "What is the capital of France?")

    assert rendered.token_ids == (7, 11, 13)
    assert len(rendered.sha256) == 64
    assert len(rendered.token_ids_sha256) == 64
    assert tuple(value["role"] for value in rendered.messages) == ("system", "user")


def test_short_answer_stop_uses_first_sentence_or_line() -> None:
    assert _completed_short_answer("The capital is Paris.") is True
    assert _completed_short_answer("Paris\nAdditional explanation") is True
    assert _completed_short_answer("The answer is Dr.") is False
    assert _completed_short_answer("Paris") is False


def test_as_numpy_applies_dtype_and_owns_the_materialized_copy() -> None:
    source = np.asarray([1.0, 2.0], dtype=np.float64)

    converted = as_numpy(source, dtype=np.float32)

    assert converted.dtype == np.float32
    assert converted.tolist() == [1.0, 2.0]
    assert not np.shares_memory(source, converted)

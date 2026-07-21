from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import torch
from torch import Tensor, nn

from mfh.contracts import ModelSpec, Runtime


class GemmaLikeBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.self_attn = nn.Linear(width, width, bias=False)
        self.post_attention_layernorm = nn.LayerNorm(width)
        self.pre_feedforward_layernorm = nn.LayerNorm(width)
        self.mlp = nn.Linear(width, width, bias=False)
        self.post_feedforward_layernorm = nn.LayerNorm(width)

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return residual + hidden_states


class QwenLikeBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.self_attn = nn.Linear(width, width, bias=False)
        self.post_attention_layernorm = nn.LayerNorm(width)
        self.mlp = nn.Linear(width, width, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = hidden_states + self.self_attn(hidden_states)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return cast(Tensor, residual + self.mlp(hidden_states))


class TinyBackbone(nn.Module):
    def __init__(self, width: int, layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([GemmaLikeBlock(width) for _ in range(layers)])

    def forward(self, hidden_states: Tensor) -> Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class TinyCausalLM(nn.Module):
    def __init__(self, *, vocab_size: int = 64, width: int = 8, layers: int = 2) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(7)
        self.embed_tokens = nn.Embedding(vocab_size, width)
        self.model = TinyBackbone(width, layers)
        self.lm_head = nn.Linear(width, vocab_size, bias=False)
        with torch.no_grad():
            self.embed_tokens.weight.copy_(
                torch.randn(self.embed_tokens.weight.shape, generator=generator)
            )
            self.lm_head.weight.copy_(torch.randn(self.lm_head.weight.shape, generator=generator))
        self.generation_config = SimpleNamespace(eos_token_id=None)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool = False,
        **_: Any,
    ) -> Any:
        del attention_mask
        hidden = self.model(self.embed_tokens(input_ids))
        logits = self.lm_head(hidden)
        cache = (torch.tensor(1),) if use_cache else None
        return SimpleNamespace(logits=logits, past_key_values=cache)


class TinyProcessor:
    eos_token_id = None

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        assert not tokenize
        assert add_generation_prompt
        assert not enable_thinking
        return f"<system>{messages[0]['content']}<user>{messages[1]['content']}<assistant>"

    def __call__(self, text: str, *, return_tensors: str) -> dict[str, Tensor]:
        assert return_tensors == "pt"
        ids = [1 + (ord(character) % 63) for character in text]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        del skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids)


def tiny_model_spec() -> ModelSpec:
    return ModelSpec(
        name="tiny",
        repository="local/tiny",
        revision="a" * 40,
        runtime=Runtime.TRANSFORMERS,
        quantization="none",
        num_layers=2,
        dtype="float32",
        candidate_layers=(0, 1),
    )

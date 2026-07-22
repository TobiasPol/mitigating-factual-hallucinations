from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from mfh.inference.vllm_worker import MfhVllmWorkerExtension


class _ResidualNorm(nn.Module):
    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = hidden_states + residual
        return combined, combined


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.post_attention_layernorm = _ResidualNorm()
        self.mlp = nn.Identity()

    def forward(
        self, attention_output: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized, post_attention = self.post_attention_layernorm(
            attention_output, residual
        )
        return self.mlp(normalized), post_attention


class _Worker(MfhVllmWorkerExtension):
    def __init__(self, layer: _Layer) -> None:
        model = SimpleNamespace(
            config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=2)),
            language_model=SimpleNamespace(
                model=SimpleNamespace(layers=nn.ModuleList([layer]))
            ),
        )
        self.model_runner = SimpleNamespace(model=model)


def _spec(site: str, *, alpha: float = 2.0) -> dict[str, Any]:
    return {
        "layer": 0,
        "site": site,
        "direction": [1.0, 0.0],
        "alpha": alpha,
        "token_scope": "final_prompt",
        "decay": 0.0,
        "prompt_tokens": 2,
        "capture_limit": 0,
    }


@pytest.mark.parametrize("site", ["post_attention", "post_mlp", "block_output"])
def test_worker_qwen_residual_sites_apply_exact_final_prompt_delta(site: str) -> None:
    layer = _Layer()
    worker = _Worker(layer)
    attention = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    residual = torch.tensor([[10.0, 20.0], [30.0, 40.0]])

    worker.mfh_install_hooks([_spec(site)])
    hidden_states, block_residual = layer(attention, residual)
    receipt = worker.mfh_collect_hooks()["specs"][f"0:{site}"]
    worker.mfh_remove_hooks()

    assert receipt["applications"] == 1
    assert receipt["captured"] == (
        [66.0, 88.0] if site == "block_output" else [33.0, 44.0]
    )
    assert receipt["applied_post_history"][0][0] - receipt[
        "applied_pre_history"
    ][0][0] == pytest.approx(2.0)
    expected_block = hidden_states + block_residual
    assert expected_block[-1, 0].item() > expected_block[0, 0].item()


def test_worker_zero_hook_preserves_qwen_tuple_exactly() -> None:
    layer = _Layer()
    worker = _Worker(layer)
    attention = torch.randn(2, 2)
    residual = torch.randn(2, 2)
    baseline = layer(attention, residual)

    worker.mfh_install_hooks([_spec("block_output", alpha=0.0)])
    observed = layer(attention, residual)
    worker.mfh_remove_hooks()

    assert torch.equal(observed[0], baseline[0])
    assert torch.equal(observed[1], baseline[1])

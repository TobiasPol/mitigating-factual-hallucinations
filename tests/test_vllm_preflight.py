from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mfh.errors import ConfigurationError, DataValidationError
from mfh.inference.vllm_preflight import (
    load_vllm_runtime_policy,
    validate_vllm_preflight_receipt,
)
from mfh.provenance import sha256_file, stable_hash

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs/runtimes/qwen3.6-27b-nvfp4-policy.json"


def test_qwen_runtime_policy_is_self_authenticated() -> None:
    policy = load_vllm_runtime_policy(POLICY)

    assert policy["model"]["repository"] == "nvidia/Qwen3.6-27B-NVFP4"
    assert policy["model"]["num_layers"] == 64
    assert policy["model"]["hidden_size"] == 5120
    assert policy["hardware"]["gpu_name_contains"] == "A100"
    assert policy["hardware"]["cuda_capability"] == "8.0"
    assert policy["software"]["vllm"] == "0.24.0"


def test_qwen_runtime_policy_rejects_body_tampering(tmp_path: Path) -> None:
    value = json.loads(POLICY.read_text(encoding="utf-8"))
    value["hardware"]["minimum_vram_bytes"] = 1
    candidate = tmp_path / "policy.json"
    candidate.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="digest differs"):
        load_vllm_runtime_policy(candidate)


def _isolated_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], dict[str, Path]]:
    policy_path = tmp_path / "policy.json"
    policy_path.write_text("{}\n", encoding="utf-8")
    paths = {
        "project_root": tmp_path,
        "model_config": tmp_path / "model.yaml",
        "snapshot_directory": tmp_path / "snapshot",
        "snapshot_manifest": tmp_path / "snapshot.json",
        "runtime_policy": policy_path,
    }
    prompt_sha = "1" * 64
    source_bindings = {"source.py": "3" * 64}
    policy = {
        "policy_digest": "2" * 64,
        "model": {"model_class": "model.Class", "hidden_size": 4},
        "hardware": {"maximum_peak_memory_bytes": 1024},
        "software": {"vllm": "0.24.0"},
        "inference": {"seed": 17},
        "validation": {
            "preflight_prompt_sha256": prompt_sha,
            "preflight_alpha": 2.0,
            "probe_layers": {"linear_attention": 0, "full_attention": 3},
            "source_bindings": source_bindings,
        },
    }
    model_spec = SimpleNamespace(
        name="qwen",
        repository="repo",
        revision="4" * 40,
        quantization="4bit",
        num_layers=4,
        candidate_layers=(0, 3),
    )
    monkeypatch.setattr(
        "mfh.inference.vllm_preflight.load_vllm_runtime_policy", lambda _path: policy
    )
    monkeypatch.setattr(
        "mfh.inference.vllm_preflight._validate_static_bindings",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "mfh.inference.vllm_preflight.load_model_spec", lambda _path: model_spec
    )
    monkeypatch.setattr(
        "mfh.inference.vllm_preflight.verify_transformers_snapshot",
        lambda *_args: {"snapshot": "exact"},
    )
    monkeypatch.setattr(
        "mfh.inference.vllm_preflight._validate_runtime",
        lambda *_args, **_kwargs: None,
    )
    toolchain = {
        "vllm": "0.24.0",
        "torch": "test",
        "transformers": "test",
        "numpy": "test",
        "nvidia_driver": "test",
    }
    checks: dict[str, Any] = {}
    for layer_type, layer in (("linear_attention", 0), ("full_attention", 3)):
        for site in ("post_attention", "post_mlp", "block_output"):
            checks[f"{layer_type}.{site}"] = {
                "status": "passed",
                "layer": layer,
                "layer_type": layer_type,
                "site": site,
                "capture_shape": [1, 1, 4],
                "alpha": 2.0,
                "applications": 1,
                "zero_generation_exact_parity": True,
                "final_delta_max_abs_error": 0.0,
                "dtype_materialized_post_max_abs_error": 0.0,
                "materialized_changed_coordinates": 4,
                "distribution_applications": 1,
                "distribution_final_delta_max_abs_error": 0.0,
                "distribution_dtype_materialized_post_max_abs_error": 0.0,
                "distribution_materialized_changed_coordinates": 4,
                "downstream_logprob_max_abs_delta": 1.0,
                "peak_memory_bytes": 512,
            }
    body = {
        "schema_version": 2,
        "status": "passed",
        "policy_path": "policy.json",
        "policy_sha256": sha256_file(policy_path),
        "policy_digest": policy["policy_digest"],
        "model": {
            "name": model_spec.name,
            "repository": model_spec.repository,
            "revision": model_spec.revision,
            "quantization": model_spec.quantization,
            "num_layers": 4,
            "hidden_size": 4,
            "snapshot_identity": {"snapshot": "exact"},
        },
        "runtime_identity": {"backend": "vllm"},
        "software": {"toolchain": toolchain, "source_bindings": source_bindings},
        "prompt": {
            "text_sha256": prompt_sha,
            "rendered_sha256": "8" * 64,
            "token_ids_sha256": "9" * 64,
            "token_count": 3,
            "thinking_enabled": False,
        },
        "architecture": {
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
            "candidate_layer_types": {"0": "linear_attention", "3": "full_attention"},
        },
        "intervention": {
            "alpha": 2.0,
            "direction": "normalized-all-ones-feasibility-only",
            "checks": checks,
        },
        "peak_memory_bytes": 512,
    }
    return {**body, "receipt_digest": stable_hash(body)}, paths


def test_preflight_receipt_rejects_rehashed_semantic_forgeries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, paths = _isolated_receipt(tmp_path, monkeypatch)
    assert validate_vllm_preflight_receipt(receipt, **paths) == receipt

    mutations: tuple[Callable[[dict[str, Any]], object], ...] = (
        lambda value: value.update({"unexpected": True}),
        lambda value: value["intervention"].update({"alpha": 0.0}),
        lambda value: value["intervention"]["checks"][
            "linear_attention.block_output"
        ].update({"downstream_logprob_max_abs_delta": 0.0}),
        lambda value: value["intervention"]["checks"][
            "linear_attention.block_output"
        ].update({"distribution_applications": 0}),
        lambda value: value["intervention"]["checks"][
            "linear_attention.block_output"
        ].pop("layer"),
        lambda value: value["prompt"].update({"text_sha256": "a" * 64}),
        lambda value: value["software"]["toolchain"].update({"vllm": "fabricated"}),
    )
    for mutate in mutations:
        forged = json.loads(json.dumps(receipt))
        mutate(forged)
        forged.pop("receipt_digest")
        forged["receipt_digest"] = stable_hash(forged)
        with pytest.raises(DataValidationError):
            validate_vllm_preflight_receipt(forged, **paths)

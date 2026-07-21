from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mfh.errors import ConfigurationError, DataValidationError
from mfh.inference.mlx_preflight import (
    load_mlx_runtime_policy,
    validate_mlx_preflight_receipt,
)
from mfh.provenance import sha256_file, stable_hash

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs/runtimes/qwen3.6-27b-mlx-4bit-policy.json"


def test_qwen_runtime_policy_is_self_authenticated() -> None:
    policy = load_mlx_runtime_policy(POLICY)

    assert policy["model"]["repository"] == "mlx-community/Qwen3.6-27B-4bit"
    assert policy["hardware"] == {
        "chip": "Apple M4 Max",
        "unified_memory_bytes": 48 * 1024**3,
        "architecture": "arm64",
        "accelerator": "Apple Metal",
    }
    assert policy["software"]["packages"]["mlx-lm"] == "0.31.3"
    assert len(policy["validation"]["layer_types"]) == 64


def test_qwen_runtime_policy_rejects_body_tampering(tmp_path: Path) -> None:
    value = json.loads(POLICY.read_text(encoding="utf-8"))
    value["hardware"]["unified_memory_bytes"] = 1
    candidate = tmp_path / "policy.json"
    candidate.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="digest differs"):
        load_mlx_runtime_policy(candidate)


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
    layer_types = ["linear_attention", "full_attention"]
    prompt_sha = "1" * 64
    policy = {
        "policy_digest": "2" * 64,
        "model": {
            "model_class": "model.Class",
            "tokenizer_class": "tokenizer.Class",
            "hidden_size": 4,
        },
        "hardware": {"unified_memory_bytes": 1024},
        "software": {"python": "3.11.14", "packages": {}},
        "inference": {"seed": 17},
        "validation": {
            "preflight_prompt_sha256": prompt_sha,
            "preflight_alpha": 2.0,
            "probe_layers": {"linear_attention": 0, "full_attention": 1},
            "layer_types": layer_types,
            "source_bindings": {"source.py": "3" * 64},
        },
    }
    model_spec = SimpleNamespace(
        name="qwen",
        repository="repo",
        revision="4" * 40,
        quantization="4bit",
        num_layers=2,
        candidate_layers=(0, 1),
    )
    monkeypatch.setattr(
        "mfh.inference.mlx_preflight.load_mlx_runtime_policy", lambda _path: policy
    )
    monkeypatch.setattr(
        "mfh.inference.mlx_preflight._validate_static_bindings",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "mfh.inference.mlx_preflight.load_model_spec", lambda _path: model_spec
    )
    monkeypatch.setattr(
        "mfh.inference.mlx_preflight.verify_transformers_snapshot",
        lambda *_args: {"snapshot": "exact"},
    )
    monkeypatch.setattr(
        "mfh.inference.mlx_preflight._validate_runtime_against_policy",
        lambda *_args: None,
    )
    toolchain = {"xcodebuild": "Xcode exact", "metal_compiler": "Metal exact"}
    monkeypatch.setattr(
        "mfh.inference.mlx_preflight.mlx_research_toolchain_identity",
        lambda: toolchain,
    )
    checks: dict[str, Any] = {}
    for layer_type, layer in (("linear_attention", 0), ("full_attention", 1)):
        for site in ("post_attention", "post_mlp", "block_output"):
            checks[f"{layer_type}.{site}"] = {
                "status": "passed",
                "layer": layer,
                "layer_type": layer_type,
                "site": site,
                "capture_shape": [1, 3, 4],
                "cached_capture_shape": [1, 3, 4],
                "captured_final_sha256": "5" * 64,
                "alpha": 2.0,
                "module_restored": True,
                "applications": 2,
                "zero_vector_exact_parity": True,
                "zero_vector_max_abs_logit_error": 0.0,
                "cached_zero_max_abs_prefill_logit_error": 0.0,
                "cached_zero_max_abs_continuation_logit_error": 0.0,
                "scope_exact": True,
                "prefix_max_abs_activation_error": 0.0,
                "final_delta_max_abs_error": 0.0,
                "cached_prefix_max_abs_activation_error": 0.0,
                "cached_final_delta_max_abs_error": 0.0,
                "nonzero_max_abs_logit_change": 1.0,
                "cached_nonzero_max_abs_prefill_logit_change": 1.0,
                "cached_nonzero_max_abs_continuation_logit_change": 1.0,
                "nonzero_changed_cached_continuation": True,
                "baseline_top_token_id": 1,
                "zero_top_token_id": 1,
                "steered_top_token_id": 2,
                "cached_baseline_token_id": 1,
                "cached_zero_token_id": 1,
                "cached_steered_token_id": 2,
                "baseline_logits_sha256": "6" * 64,
                "zero_logits_sha256": "6" * 64,
                "steered_logits_sha256": "7" * 64,
            }
    body = {
        "schema_version": 1,
        "status": "passed",
        "policy_path": "policy.json",
        "policy_sha256": sha256_file(policy_path),
        "policy_digest": policy["policy_digest"],
        "model": {
            "name": model_spec.name,
            "repository": model_spec.repository,
            "revision": model_spec.revision,
            "quantization": model_spec.quantization,
            "num_layers": 2,
            "hidden_size": 4,
            "snapshot_identity": {"snapshot": "exact"},
        },
        "runtime_identity": {
            "model_class": "model.Class",
            "tokenizer_class": "tokenizer.Class",
            "num_layers": 2,
            "seed": 17,
        },
        "software": {
            "python": "3.11.14",
            "packages": {},
            "toolchain": toolchain,
            "source_bindings": {"source.py": "3" * 64},
        },
        "prompt": {
            "text_sha256": prompt_sha,
            "rendered_sha256": "8" * 64,
            "token_ids_sha256": "9" * 64,
            "token_count": 3,
            "thinking_enabled": False,
        },
        "architecture": {
            "layer_types": layer_types,
            "candidate_layer_types": {"0": layer_types[0], "1": layer_types[1]},
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
    assert validate_mlx_preflight_receipt(receipt, **paths) == receipt

    for mutate in (
        lambda value: value.update({"unexpected": True}),
        lambda value: value["intervention"].update({"alpha": 0.0}),
        lambda value: value["intervention"]["checks"][
            "linear_attention.block_output"
        ].update({"zero_vector_max_abs_logit_error": 999.0}),
        lambda value: value["intervention"]["checks"][
            "linear_attention.block_output"
        ].pop("layer"),
        lambda value: value["prompt"].update({"text_sha256": "a" * 64}),
        lambda value: value["software"]["toolchain"].update(
            {"xcodebuild": "fabricated"}
        ),
    ):
        forged = json.loads(json.dumps(receipt))
        mutate(forged)
        forged.pop("receipt_digest")
        forged["receipt_digest"] = stable_hash(forged)
        with pytest.raises(DataValidationError):
            validate_mlx_preflight_receipt(forged, **paths)

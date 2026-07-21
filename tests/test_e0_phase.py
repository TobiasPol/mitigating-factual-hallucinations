from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mfh.contracts import Runtime
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.experiments.e0_phase import (
    _determinism_observations,
    _resolve_hook_preflight,
    finalize_e0_phase_run,
)
from mfh.experiments.protocol import ExperimentPhase
from mfh.experiments.runner import EvaluationCondition
from mfh.provenance import stable_hash


def _condition() -> EvaluationCondition:
    return EvaluationCondition(
        phase=ExperimentPhase.E0,
        benchmark="shared_benign_factual_500",
        partition="runtime-validation",
        model_name="qwen3.6-27b-mlx-4bit",
        model_repository="mlx-community/Qwen3.6-27B-4bit",
        model_revision="c000ac2c2057d94be3fa931000c31723aac53282",
        runtime=Runtime.MLX,
        quantization="affine-g64-mlx-4bit",
        model_num_layers=64,
        system_prompt_id="P0-neutral",
        prompt_template_sha256="a" * 64,
        steering_method="M0",
        method_artifact_sha256=None,
        layer=None,
        site=None,
        token_scope=None,
        alpha=0.0,
        sparsity=None,
        seed=17,
        study_protocol_digest="b" * 64,
    )


def test_hook_preflight_receipt_is_directly_validated(tmp_path: Path) -> None:
    checks = {
        f"{layer}.{site}": {
            "status": "passed",
            "zero_vector_exact_parity": True,
            "scope_exact": True,
            "nonzero_changed_cached_continuation": True,
        }
        for layer in ("linear_attention", "full_attention")
        for site in ("post_attention_residual", "mlp_output", "block_output")
    }
    body = {
        "schema_version": 1,
        "status": "passed",
        "intervention": {"checks": checks},
    }
    runtime = tmp_path / "mlx-preflight.json"
    runtime.write_text(
        json.dumps({**body, "receipt_digest": stable_hash(body)}), encoding="utf-8"
    )
    assert _resolve_hook_preflight(runtime) == runtime

    checks["linear_attention.block_output"]["scope_exact"] = False
    tampered_body = {**body, "intervention": {"checks": checks}}
    runtime.write_text(
        json.dumps(
            {**tampered_body, "receipt_digest": stable_hash(tampered_body)}
        ),
        encoding="utf-8",
    )
    with pytest.raises(DataValidationError, match="hook-preflight evidence"):
        _resolve_hook_preflight(runtime)


def test_determinism_observations_reject_schedule_tampering(tmp_path: Path) -> None:
    condition = _condition()
    questions = ("q-1", "q-2")
    rows = []
    for question in questions:
        for repeat_index in range(2):
            rows.append(
                {
                    "question_id": question,
                    "condition_id": condition.condition_id,
                    "repeat_index": repeat_index,
                    "raw_output_stable_hash": hashlib.sha256(question.encode()).hexdigest(),
                }
            )
    root = tmp_path / "mlx"
    root.mkdir()
    records = root / "records.jsonl"
    records.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    observations = _determinism_observations(
        root, condition=condition, question_ids=questions
    )
    assert len(observations) == 2

    rows[1]["repeat_index"] = 0
    records.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(DataValidationError, match="repeat indices"):
        _determinism_observations(root, condition=condition, question_ids=questions)


def test_finalizer_refuses_existing_output_before_replay(tmp_path: Path) -> None:
    output = tmp_path / "E0"
    output.mkdir()
    with pytest.raises(FrozenArtifactError, match="refusing to overwrite"):
        finalize_e0_phase_run(
            output,
            completion_receipt=tmp_path / "receipt",
            expected_completion_manifest_digest="a" * 64,
            mlx_directory=tmp_path / "mlx",
            expected_mlx_manifest_digest="b" * 64,
            expected_mlx_plan_identity="c" * 64,
            mlx_inputs={},
            review_result_directory=tmp_path / "review",
            expected_review_result_manifest_digest="d" * 64,
            review_queue_directory=tmp_path / "queue",
            expected_review_queue_manifest_digest="e" * 64,
            review_inputs={},
        )

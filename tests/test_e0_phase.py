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
        model_name="qwen3.6-27b-nvfp4",
        model_repository="nvidia/Qwen3.6-27B-NVFP4",
        model_revision="0893e1606ff3d5f97a441f405d5fc541a6bdf404",
        runtime=Runtime.VLLM,
        quantization="modelopt-mixed-nvfp4-fp8",
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


def test_hook_preflight_receipt_is_directly_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = {
        "schema_version": 2,
        "status": "passed",
        "policy_path": "configs/runtimes/policy.json",
    }
    runtime = tmp_path / "vllm-preflight.json"
    runtime.write_text(
        json.dumps({**body, "receipt_digest": stable_hash(body)}), encoding="utf-8"
    )
    observed: dict[str, object] = {}

    def validate(receipt: object, **kwargs: object) -> object:
        observed.update(kwargs)
        return receipt

    monkeypatch.setattr("mfh.experiments.e0_phase.validate_vllm_preflight_receipt", validate)
    assert (
        _resolve_hook_preflight(
            runtime,
            project_root=tmp_path,
            model_config=tmp_path / "model.yaml",
            snapshot_directory=tmp_path / "snapshot",
            snapshot_manifest=tmp_path / "snapshot.json",
        )
        == runtime
    )
    assert observed["runtime_policy"] == tmp_path / "configs/runtimes/policy.json"


def test_hook_preflight_receipt_rejects_policy_escape(tmp_path: Path) -> None:
    body = {"schema_version": 2, "status": "passed", "policy_path": "../policy.json"}
    runtime = tmp_path / "vllm-preflight.json"
    runtime.write_text(
        json.dumps({**body, "receipt_digest": stable_hash(body)}), encoding="utf-8"
    )
    with pytest.raises(DataValidationError, match="escapes the project root"):
        _resolve_hook_preflight(
            runtime,
            project_root=tmp_path,
            model_config=tmp_path / "model.yaml",
            snapshot_directory=tmp_path / "snapshot",
            snapshot_manifest=tmp_path / "snapshot.json",
        )


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
    root = tmp_path / "vllm"
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
            vllm_directory=tmp_path / "vllm",
            expected_vllm_manifest_digest="b" * 64,
            expected_vllm_plan_identity="c" * 64,
            vllm_inputs={},
            review_result_directory=tmp_path / "review",
            expected_review_result_manifest_digest="d" * 64,
            review_queue_directory=tmp_path / "queue",
            expected_review_queue_manifest_digest="e" * 64,
            review_inputs={},
            grader_bundle=tmp_path / "graders",
            expected_grader_manifest_digest="f" * 64,
            reviewed_splits=tmp_path / "reviewed-splits",
            expected_reviewed_split_manifest_digest="1" * 64,
        )

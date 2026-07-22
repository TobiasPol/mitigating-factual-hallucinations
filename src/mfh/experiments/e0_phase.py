"""Promote a verified native-VLLM E0 bundle into the immutable phase ledger."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mfh.data.io import read_generation_records, read_questions
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.experiments.e0_completion import authorize_e0_completion_receipt
from mfh.experiments.evidence import GateResult
from mfh.experiments.gates import write_gate_evidence
from mfh.experiments.model_selection import validate_active_study_artifact_paths
from mfh.experiments.protocol import ExperimentPhase, load_study_protocol
from mfh.experiments.runner import (
    EvaluationCondition,
    PhaseCompletion,
    PhaseRunContract,
    PhaseRunLedger,
)
from mfh.inference.vllm_preflight import validate_vllm_preflight_receipt
from mfh.provenance import sha256_path


def _read_json(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot read {context}: {exc}") from exc
    if not isinstance(value, dict):
        raise DataValidationError(f"{context} must be a JSON object")
    return value


def _read_jsonl(path: Path, context: str) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise DataValidationError(
                        f"{context} line {line_number} must be a JSON object"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot read {context}: {exc}") from exc
    return tuple(rows)


def _resolve_hook_preflight(
    runtime_receipt: Path,
    *,
    project_root: Path,
    model_config: Path,
    snapshot_directory: Path,
    snapshot_manifest: Path,
) -> Path:
    """Validate the schema-2 receipt that directly contains live hook evidence."""

    if runtime_receipt.is_symlink() or not runtime_receipt.is_file():
        raise DataValidationError("VLLM preflight receipt must be a regular file")
    receipt = _read_json(runtime_receipt, "VLLM preflight receipt")
    policy_value = receipt.get("policy_path")
    if not isinstance(policy_value, str):
        raise DataValidationError("VLLM preflight receipt lacks its runtime policy path")
    root = project_root.resolve()
    policy_path = (root / policy_value).resolve()
    if not policy_path.is_relative_to(root):
        raise DataValidationError("VLLM preflight runtime policy escapes the project root")
    validate_vllm_preflight_receipt(
        receipt,
        project_root=root,
        model_config=model_config,
        snapshot_directory=snapshot_directory,
        snapshot_manifest=snapshot_manifest,
        runtime_policy=policy_path,
    )
    return runtime_receipt


def _determinism_observations(
    vllm_directory: Path,
    *,
    condition: EvaluationCondition,
    question_ids: tuple[str, ...],
) -> tuple[dict[str, str], ...]:
    rows = _read_jsonl(vllm_directory / "records.jsonl", "E0 low-level records")
    if len(rows) != 2 * len(question_ids):
        raise DataValidationError("E0 low-level records do not contain two exact repeats")
    observations: list[dict[str, str]] = []
    for index, question_id in enumerate(question_ids):
        first, repeat = rows[index * 2 : index * 2 + 2]
        expected_common = (question_id, condition.condition_id)
        for row, repeat_index in ((first, 0), (repeat, 1)):
            if (row.get("question_id"), row.get("condition_id")) != expected_common:
                raise DataValidationError("E0 repeat schedule differs from the phase contract")
            if row.get("repeat_index") != repeat_index:
                raise DataValidationError("E0 repeat indices differ from the phase contract")
        first_digest = first.get("raw_output_stable_hash")
        repeat_digest = repeat.get("raw_output_stable_hash")
        if not isinstance(first_digest, str) or not isinstance(repeat_digest, str):
            raise DataValidationError("E0 repeat records lack stable output identities")
        observations.append(
            {
                "condition_id": condition.condition_id,
                "question_id": question_id,
                "first_output_sha256": first_digest,
                "repeat_output_sha256": repeat_digest,
            }
        )
    return tuple(observations)


def finalize_e0_phase_run(
    directory: str | Path,
    *,
    completion_receipt: str | Path,
    expected_completion_manifest_digest: str,
    vllm_directory: str | Path,
    expected_vllm_manifest_digest: str,
    expected_vllm_plan_identity: str,
    vllm_inputs: Mapping[str, Any],
    review_result_directory: str | Path,
    expected_review_result_manifest_digest: str,
    review_queue_directory: str | Path,
    expected_review_queue_manifest_digest: str,
    review_inputs: Mapping[str, Any],
    grader_bundle: str | Path,
    expected_grader_manifest_digest: str,
    reviewed_splits: str | Path,
    expected_reviewed_split_manifest_digest: str,
) -> PhaseCompletion:
    """Replay all E0 evidence and atomically publish its immutable phase ledger."""

    validated_paths = validate_active_study_artifact_paths(
        {
            "E0 phase ledger": directory,
            "E0 completion receipt": completion_receipt,
            "E0 VLLM bundle": vllm_directory,
            "E1 grader bundle": grader_bundle,
            "reviewed splits": reviewed_splits,
        }
    )
    output = validated_paths["E0 phase ledger"]
    completion_receipt = validated_paths["E0 completion receipt"]
    vllm_directory = validated_paths["E0 VLLM bundle"]
    grader_bundle = validated_paths["E1 grader bundle"]
    reviewed_splits = validated_paths["reviewed splits"]
    if output.exists():
        raise FrozenArtifactError(f"refusing to overwrite E0 phase run: {output}")
    vllm_root = Path(vllm_directory)
    cohort = Path(vllm_inputs["cohort_directory"])
    snapshot = Path(vllm_inputs["snapshot_directory"])
    snapshot_manifest = Path(vllm_inputs["snapshot_manifest"])
    runtime_config = Path(vllm_inputs["runtime_config"])
    study = load_study_protocol(Path(vllm_inputs["study_config"]))

    authorized_vllm_fingerprint = sha256_path(vllm_root)

    capability = authorize_e0_completion_receipt(
        completion_receipt,
        expected_manifest_digest=expected_completion_manifest_digest,
        vllm_directory=vllm_root,
        expected_vllm_manifest_digest=expected_vllm_manifest_digest,
        expected_vllm_plan_identity=expected_vllm_plan_identity,
        vllm_inputs=vllm_inputs,
        review_result_directory=review_result_directory,
        expected_review_result_manifest_digest=expected_review_result_manifest_digest,
        review_queue_directory=review_queue_directory,
        expected_review_queue_manifest_digest=expected_review_queue_manifest_digest,
        review_inputs=review_inputs,
        grader_bundle=grader_bundle,
        expected_grader_manifest_digest=expected_grader_manifest_digest,
        reviewed_splits=reviewed_splits,
        expected_reviewed_split_manifest_digest=expected_reviewed_split_manifest_digest,
    )
    if sha256_path(vllm_root) != authorized_vllm_fingerprint:
        raise FrozenArtifactError("E0 VLLM bundle changed during completion authorization")

    plan = _read_json(vllm_root / "plan.json", "E0 VLLM plan")
    raw_condition = plan.get("condition")
    if not isinstance(raw_condition, Mapping):
        raise DataValidationError("E0 VLLM plan lacks its evaluation condition")
    condition = EvaluationCondition.from_dict(raw_condition)
    if condition.phase is not ExperimentPhase.E0 or condition.study_protocol_digest != study.digest:
        raise DataValidationError("E0 VLLM condition differs from the live study protocol")

    question_ids = tuple(
        question.question_id for question in read_questions(cohort / "questions.jsonl")
    )
    records = tuple(read_generation_records(vllm_root / "generation-records.jsonl"))
    if len(question_ids) != 500 or len(records) != 500:
        raise DataValidationError("E0 phase ledger requires exactly 500 questions and records")

    input_artifacts = {
        "model_artifacts": snapshot_manifest,
        "tokenizers": snapshot / "tokenizer.json",
        "chat_templates": snapshot / "chat_template.jinja",
        "runtime_receipt": runtime_config,
        "hook_preflight": _resolve_hook_preflight(
            runtime_config,
            project_root=Path.cwd(),
            model_config=Path(vllm_inputs["model_config"]),
            snapshot_directory=snapshot,
            snapshot_manifest=snapshot_manifest,
        ),
        "e1_grader_bundle": grader_bundle,
        "reviewed_splits": reviewed_splits,
    }
    contract = PhaseRunContract(
        phase=ExperimentPhase.E0,
        study_protocol_digest=study.digest,
        conditions=(condition,),
        question_ids_by_benchmark={condition.benchmark: question_ids},
        input_fingerprints={name: sha256_path(path) for name, path in input_artifacts.items()},
        prerequisite_digests={},
        required_gates=study.phase(ExperimentPhase.E0).gates,
    )
    contract.assert_matches_study(study)

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    evidence = Path(tempfile.mkdtemp(prefix=f".{output.name}.gates-", dir=output.parent))
    shutil.rmtree(stage)
    try:
        ledger = PhaseRunLedger.create(
            stage,
            contract,
            study=study,
            input_artifacts=input_artifacts,
            prerequisite_runs={},
        )
        ledger.checkpoint(records)
        observations: Mapping[str, tuple[dict[str, str], ...]] = {
            "checkpoint_identity": (),
            "deterministic_decode": _determinism_observations(
                vllm_root, condition=condition, question_ids=question_ids
            ),
            "chat_template_identity": (),
            "vllm_runtime_identity": (),
        }
        gate_results: dict[str, GateResult] = {}
        for gate in contract.required_gates:
            path = evidence / f"{gate}.json"
            write_gate_evidence(
                path,
                phase=ExperimentPhase.E0,
                gate=gate,
                contract_digest=contract.digest,
                record_set_digest=ledger.record_set_digest(),
                observations=observations[gate],
            )
            gate_results[gate] = ledger.evaluate_gate(gate, path)
        completion = ledger.finalize(
            gate_results,
            verified_e0_completion=capability,
        )
        ledger.verify_complete()
        if sha256_path(vllm_root) != authorized_vllm_fingerprint:
            raise FrozenArtifactError("E0 VLLM bundle changed during phase finalization")
        if output.exists():
            raise FrozenArtifactError(f"E0 phase output appeared during finalization: {output}")
        os.replace(stage, output)
        PhaseRunLedger.open(output, study=study).verify_complete()
        return completion
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        if evidence.exists():
            shutil.rmtree(evidence)

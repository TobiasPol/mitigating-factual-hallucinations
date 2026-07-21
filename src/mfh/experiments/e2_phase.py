"""Promote verified E2 capture and probe evidence into an immutable phase ledger."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch

from mfh.contracts import GenerationRecord, Outcome, PromptSpec, Question, Runtime
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.experiments.activation_store import verify_activation_store
from mfh.experiments.e2_capture import E1P0Source, verify_e2_capture_work
from mfh.experiments.e2_probes import (
    E2FeatureDataset,
    E2FeatureView,
    VerifiedE2ProbeBundle,
    build_e2_probe_dataset,
    verify_e2_probe_bundle,
)
from mfh.experiments.e2_schedule import VerifiedE2Workspace, verify_e2_workspace
from mfh.experiments.evidence import GateResult
from mfh.experiments.gates import write_gate_evidence
from mfh.experiments.model_selection import (
    ACTIVE_MODEL_IDENTITIES,
    ACTIVE_MODEL_NAME,
    validate_active_study_artifact_paths,
)
from mfh.experiments.protocol import ExperimentPhase, load_study_protocol
from mfh.experiments.runner import (
    EvaluationCondition,
    PhaseCompletion,
    PhaseFalsification,
    PhaseRunContract,
    PhaseRunLedger,
)
from mfh.methods.probes import (
    CalibratedProbe,
    ProbeKind,
    ProbeTask,
    load_calibrated_probe,
)
from mfh.provenance import sha256_file, sha256_path, stable_hash

_MODEL_NAME = ACTIVE_MODEL_NAME
_METHOD_BY_KIND = {
    ProbeKind.LOGISTIC: "probe-logistic",
    ProbeKind.TWO_LAYER_MLP: "probe-two-layer-mlp",
}
_P0_TASK = ProbeTask.CORRECT_INCORRECT_ABSTENTION
_P3_TASK = ProbeTask.FORCED_CORRECT_INCORRECT
_GATE = "probe_beats_confidence_baselines"


@dataclass(frozen=True, slots=True)
class _SelectedProbe:
    task: ProbeTask
    kind: ProbeKind
    artifact_sha256: str
    artifact: Path
    view: E2FeatureView
    calibration: str


@dataclass(frozen=True, slots=True)
class _EvaluationSlice:
    benchmark: str
    partition: str
    prompt_id: str
    selected: _SelectedProbe


def _read_json_object(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read {context}: {exc}") from exc
    if type(value) is not dict:
        raise FrozenArtifactError(f"{context} must be a JSON object")
    return value


def _finite_number(value: Any, context: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        raise FrozenArtifactError(f"{context} must be a finite number")
    return float(value)


def _verify_e1_output_completion_binding(
    directory: Path,
    *,
    completion: PhaseCompletion,
    contract_digest: str,
) -> Mapping[str, Any]:
    expected_inventory = {"manifest.json", "outcome-labels.jsonl", "prompt-metrics.json"}
    if (
        directory.is_symlink()
        or not directory.is_dir()
        or {path.name for path in directory.iterdir()} != expected_inventory
        or any(path.is_symlink() or not path.is_file() for path in directory.iterdir())
    ):
        raise FrozenArtifactError("E1 output inventory differs during E2 finalization")
    manifest = _read_json_object(directory / "manifest.json", "E1 output manifest")
    body = dict(manifest)
    manifest_digest = body.pop("manifest_digest", None)
    expected_keys = {
        "schema_version",
        "purpose",
        "phase",
        "plan_identity",
        "contract_digest",
        "completion_digest",
        "record_set_digest",
        "record_count",
        "condition_count",
        "grader_bundle_manifest_digest",
        "work_fingerprints",
        "files",
    }
    files = body.get("files")
    if (
        set(body) != expected_keys
        or manifest_digest != stable_hash(body)
        or body["schema_version"] != 1
        or body["purpose"] != "E1-baseline-records-prompt-metrics-and-outcome-labels"
        or body["phase"] != ExperimentPhase.E1.value
        or body["contract_digest"] != contract_digest
        or body["contract_digest"] != completion.contract_digest
        or body["completion_digest"] != completion.completion_digest
        or body["record_set_digest"] != completion.record_set_digest
        or body["record_count"] != completion.record_count
        or type(files) is not dict
    ):
        raise FrozenArtifactError("E1 output manifest is not bound to its completed E1 run")
    assert isinstance(files, dict)
    labels = files.get("outcome_labels")
    metrics = files.get("prompt_metrics")
    if (
        type(labels) is not dict
        or set(labels) != {"path", "sha256", "rows"}
        or labels["path"] != "outcome-labels.jsonl"
        or labels["sha256"] != sha256_file(directory / "outcome-labels.jsonl")
        or labels["rows"] != completion.record_count
        or type(metrics) is not dict
        or set(metrics) != {"path", "sha256", "metrics_digest"}
        or metrics["path"] != "prompt-metrics.json"
        or metrics["sha256"] != sha256_file(directory / "prompt-metrics.json")
    ):
        raise FrozenArtifactError("E1 output file evidence differs during E2 finalization")
    return MappingProxyType(manifest)


def _select_probe_artifacts(
    probe_bundle: VerifiedE2ProbeBundle,
) -> Mapping[tuple[ProbeTask, ProbeKind], _SelectedProbe]:
    """Re-derive the two per-task, per-family artifacts used by the phase ledger."""

    results = _read_json_object(
        probe_bundle.directory / "results.json", "verified E2 probe results"
    )
    raw_rows = results.get("final_probes")
    if not isinstance(raw_rows, list):
        raise FrozenArtifactError("verified E2 probe results lack their final probe grid")
    candidates: dict[tuple[ProbeTask, ProbeKind], list[tuple[float, str, dict[str, Any]]]] = {
        (task, kind): []
        for task in (_P0_TASK, _P3_TASK)
        for kind in ProbeKind
    }
    try:
        for raw in raw_rows:
            if type(raw) is not dict:
                raise FrozenArtifactError("E2 final probe row must be a mapping")
            task = ProbeTask(raw["task"])
            if task not in {_P0_TASK, _P3_TASK}:
                continue
            kind = ProbeKind(raw["kind"])
            digest = raw["artifact_sha256"]
            if type(digest) is not str:
                raise FrozenArtifactError("E2 final probe artifact identity is invalid")
            if task is _P0_TASK:
                score = _finite_number(raw["incorrect_auroc"], "E2 incorrect AUROC")
            else:
                score = _finite_number(
                    raw["metrics"]["T-dev"]["macro_auroc"],
                    "E2 forced-answer development AUROC",
                )
            candidates[(task, kind)].append((score, digest, raw))
    except (KeyError, TypeError, ValueError) as exc:
        raise FrozenArtifactError(f"invalid E2 final probe selection evidence: {exc}") from exc

    selected: dict[tuple[ProbeTask, ProbeKind], _SelectedProbe] = {}
    for key, values in candidates.items():
        if len(values) != 2:
            raise FrozenArtifactError("E2 final probe grid lacks two calibrated candidates")
        _score, digest, row = sorted(values, key=lambda value: (-value[0], value[1]))[0]
        relative = row.get("artifact")
        calibration = row.get("calibration")
        if type(relative) is not str or type(calibration) is not str:
            raise FrozenArtifactError("E2 selected probe descriptor is invalid")
        artifact = probe_bundle.directory / "probes" / relative
        if sha256_path(artifact) != digest:
            raise FrozenArtifactError("E2 selected probe artifact changed after verification")
        task, kind = key
        selected[key] = _SelectedProbe(
            task=task,
            kind=kind,
            artifact_sha256=digest,
            artifact=artifact,
            view=probe_bundle.selected_views[task],
            calibration=calibration,
        )
    if probe_bundle.selected_gate_artifact not in {
        selected[(_P0_TASK, kind)].artifact_sha256 for kind in ProbeKind
    }:
        raise FrozenArtifactError("E2 gate artifact is not the per-family P0 selection")
    return MappingProxyType(selected)


def _evaluation_slices(
    selected: Mapping[tuple[ProbeTask, ProbeKind], _SelectedProbe],
) -> tuple[_EvaluationSlice, ...]:
    slices: list[_EvaluationSlice] = []
    for kind in ProbeKind:
        p0 = selected[(_P0_TASK, kind)]
        slices.extend(
            (
                _EvaluationSlice("triviaqa", "T-dev", "P0-neutral", p0),
                _EvaluationSlice(
                    "triviaqa",
                    "T-dev",
                    "P3-forced-answer",
                    selected[(_P3_TASK, kind)],
                ),
                _EvaluationSlice("simpleqa_verified", "simpleqa-eval", "P0-neutral", p0),
                _EvaluationSlice(
                    "aa_omniscience_public_600", "aa-eval", "P0-neutral", p0
                ),
            )
        )
    return tuple(slices)


def _condition(
    *,
    study_digest: str,
    workspace: VerifiedE2Workspace,
    prompts: Mapping[str, PromptSpec],
    evaluation: _EvaluationSlice,
    gate_artifact_sha256: str,
) -> EvaluationCondition:
    identity = ACTIVE_MODEL_IDENTITIES[_MODEL_NAME]
    prompt = prompts[evaluation.prompt_id]
    if (
        workspace.activation_spec.model_repository != identity["repository"]
        or workspace.activation_spec.model_revision != identity["revision"]
        or workspace.activation_spec.quantization != identity["quantization"]
    ):
        raise FrozenArtifactError("E2 workspace model differs from the active model identity")
    gate_selected = (
        evaluation.partition == "T-dev"
        and evaluation.prompt_id == "P0-neutral"
        and evaluation.selected.artifact_sha256 == gate_artifact_sha256
    )
    return EvaluationCondition(
        phase=ExperimentPhase.E2,
        benchmark=evaluation.benchmark,
        partition=evaluation.partition,
        model_name=_MODEL_NAME,
        model_repository=str(identity["repository"]),
        model_revision=str(identity["revision"]),
        runtime=Runtime.MLX,
        quantization=str(identity["quantization"]),
        model_num_layers=int(identity["num_layers"]),
        system_prompt_id=evaluation.prompt_id,
        prompt_template_sha256=hashlib.sha256(prompt.text.encode("utf-8")).hexdigest(),
        steering_method=_METHOD_BY_KIND[evaluation.selected.kind],
        method_artifact_sha256=evaluation.selected.artifact_sha256,
        layer=None,
        site=None,
        token_scope=None,
        alpha=0.0,
        sparsity=None,
        seed=workspace.protocol.seed,
        study_protocol_digest=study_digest,
        comparison_group="gate-selected" if gate_selected else "evaluation-only",
    )


def _probe_probabilities(
    probe: CalibratedProbe,
    features: torch.Tensor,
) -> tuple[Mapping[str, float], ...]:
    raw = probe.predict_probabilities(features).tolist()
    values: list[Mapping[str, float]] = []
    for row in raw:
        probabilities = {
            label: float(value) for label, value in zip(probe.state.labels, row, strict=True)
        }
        if (
            any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities.values())
            or not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-5)
        ):
            raise DataValidationError("E2 probe emitted invalid calibrated probabilities")
        values.append(MappingProxyType(probabilities))
    return tuple(values)


def _records_for_slice(
    *,
    condition: EvaluationCondition,
    evaluation: _EvaluationSlice,
    dataset: E2FeatureDataset,
    workspace: VerifiedE2Workspace,
    probe_bundle: VerifiedE2ProbeBundle,
    capture_plan_identity: str,
    capture_work_sha256: str,
    activation_chain_head: str,
) -> tuple[GenerationRecord, ...]:
    probe = load_calibrated_probe(evaluation.selected.artifact)
    if (
        probe.task is not evaluation.selected.task
        or probe.state.kind is not evaluation.selected.kind
    ):
        raise FrozenArtifactError("E2 selected probe artifact has the wrong task or family")
    if Outcome.INCORRECT.value not in probe.state.labels:
        raise FrozenArtifactError("E2 selected probe lacks an incorrect class")
    probabilities = _probe_probabilities(probe, dataset.probe.features)
    if len(probabilities) != len(dataset.rows):
        raise FrozenArtifactError("E2 probe scores differ from the activation rows")
    gate_eligible = condition.comparison_group == "gate-selected"
    if gate_eligible and any(
        row.outcome not in {Outcome.CORRECT, Outcome.INCORRECT, Outcome.ABSTENTION}
        for row in dataset.rows
    ):
        raise FrozenArtifactError("E2 selected gate rows contain an unscorable outcome")
    common_provenance = {
        "workspace_plan_identity": workspace.plan_identity,
        "probe_plan_identity": probe_bundle.plan_identity,
        "probe_bundle_manifest_digest": probe_bundle.manifest_digest,
        "capture_plan_identity": capture_plan_identity,
        "capture_work_sha256": capture_work_sha256,
        "activation_chain_head": activation_chain_head,
    }
    records: list[GenerationRecord] = []
    for row, scores in zip(dataset.rows, probabilities, strict=True):
        records.append(
            GenerationRecord(
                question_id=row.question_id,
                benchmark=row.benchmark,
                model_repository=condition.model_repository,
                model_revision=condition.model_revision,
                runtime=condition.runtime,
                quantization=condition.quantization,
                system_prompt_id=condition.system_prompt_id,
                rendered_prompt_hash=row.rendered_prompt_sha256,
                steering_method=condition.steering_method,
                layer=None,
                token_scope=None,
                alpha=0.0,
                sparsity=None,
                controller_scores=scores,
                raw_output="",
                normalized_answer="",
                outcome=row.outcome,
                generation_latency_seconds=0.0,
                input_tokens=0,
                output_tokens=0,
                condition_id=condition.condition_id,
                site=None,
                seed=condition.seed,
                metadata={
                    **common_provenance,
                    "partition": row.partition,
                    "semantic_group_id": row.semantic_group_id,
                    "prompt_token_ids_sha256": row.prompt_token_ids_sha256,
                    "source_generation_record_sha256": row.generation_record_sha256,
                    "probe_task": evaluation.selected.task.value,
                    "probe_kind": evaluation.selected.kind.value,
                    "probe_calibration": evaluation.selected.calibration,
                    "probe_feature_view": evaluation.selected.view.to_dict(),
                    "probe_score": scores[Outcome.INCORRECT.value],
                    "output_entropy": row.output_entropy,
                    "maximum_token_probability": row.maximum_token_probability,
                    "probe_artifact_sha256": evaluation.selected.artifact_sha256,
                    "probe_gate_eligible": gate_eligible,
                },
            )
        )
    return tuple(records)


def _gate_observations(records: Iterable[GenerationRecord]) -> tuple[dict[str, Any], ...]:
    observations: list[dict[str, Any]] = []
    for record in records:
        metadata = record.metadata
        observations.append(
            {
                "condition_id": record.condition_id,
                "question_id": record.question_id,
                "incorrect": record.outcome is Outcome.INCORRECT,
                "probe_score": metadata["probe_score"],
                "output_entropy": metadata["output_entropy"],
                "maximum_token_probability": metadata["maximum_token_probability"],
                "probe_artifact_sha256": metadata["probe_artifact_sha256"],
                "gate_eligible": metadata["probe_gate_eligible"],
            }
        )
    return tuple(observations)


def _validate_gate_replay(
    result: GateResult,
    probe_bundle: VerifiedE2ProbeBundle,
) -> None:
    if (
        result.metrics.get("probe_auroc") != probe_bundle.gate_probe_auroc
        or result.metrics.get("best_confidence_baseline_auroc")
        != probe_bundle.gate_baseline_auroc
        or result.metrics.get("minimum_material_gain") != 0.02
        or result.passed is not probe_bundle.gate_passed
    ):
        raise FrozenArtifactError("E2 phase gate differs from the replayed probe bundle")


def _phase_material(
    *,
    study_digest: str,
    workspace: VerifiedE2Workspace,
    probe_bundle: VerifiedE2ProbeBundle,
    prompts: Mapping[str, PromptSpec],
    split_manifest_digest: str,
    capture_plan_identity: str,
    capture_work_sha256: str,
    activation_chain_head: str,
) -> tuple[
    tuple[EvaluationCondition, ...],
    Mapping[str, tuple[str, ...]],
    tuple[GenerationRecord, ...],
]:
    selected = _select_probe_artifacts(probe_bundle)
    conditions: list[EvaluationCondition] = []
    records: list[GenerationRecord] = []
    questions: dict[str, tuple[str, ...]] = {}
    dataset_cache: dict[tuple[str, str, E2FeatureView], E2FeatureDataset] = {}
    for evaluation in _evaluation_slices(selected):
        condition = _condition(
            study_digest=study_digest,
            workspace=workspace,
            prompts=prompts,
            evaluation=evaluation,
            gate_artifact_sha256=probe_bundle.selected_gate_artifact,
        )
        dataset_key = (
            evaluation.partition,
            evaluation.prompt_id,
            evaluation.selected.view,
        )
        if dataset_key not in dataset_cache:
            dataset_cache[dataset_key] = build_e2_probe_dataset(
                workspace,
                partition=evaluation.partition,
                prompt_id=evaluation.prompt_id,
                view=evaluation.selected.view,
                split_manifest_digest=split_manifest_digest,
                prompt_template_sha256=hashlib.sha256(
                    prompts[evaluation.prompt_id].text.encode("utf-8")
                ).hexdigest(),
            )
        condition_records = _records_for_slice(
            condition=condition,
            evaluation=evaluation,
            dataset=dataset_cache[dataset_key],
            workspace=workspace,
            probe_bundle=probe_bundle,
            capture_plan_identity=capture_plan_identity,
            capture_work_sha256=capture_work_sha256,
            activation_chain_head=activation_chain_head,
        )
        identifiers = tuple(record.question_id for record in condition_records)
        previous = questions.setdefault(evaluation.benchmark, identifiers)
        if previous != identifiers:
            raise FrozenArtifactError("E2 probe conditions use different benchmark question sets")
        conditions.append(condition)
        records.extend(condition_records)
    if len(conditions) != 8 or sum(
        condition.comparison_group == "gate-selected" for condition in conditions
    ) != 1:
        raise FrozenArtifactError("E2 phase material lacks its exact eight-condition grid")
    return tuple(conditions), MappingProxyType(questions), tuple(records)


def finalize_e2_phase_run(
    directory: str | Path,
    *,
    workspace_directory: str | Path,
    expected_workspace_plan_identity: str,
    capture_work_directory: str | Path,
    expected_capture_plan_identity: str,
    probe_bundle_directory: str | Path,
    expected_probe_manifest_digest: str,
    questions: Mapping[tuple[str, str], Question],
    prompts: Mapping[str, PromptSpec],
    e1_sources: Mapping[tuple[str, str], E1P0Source],
    e1_output_directory: str | Path,
    e1_phase_run: str | Path,
    split_manifest_digest: str,
    study_config: str | Path,
) -> PhaseCompletion | PhaseFalsification:
    """Replay E2 evidence and publish either a completed or falsified phase ledger."""

    normalized_paths = validate_active_study_artifact_paths(
        {
            "E2 phase ledger": directory,
            "E2 workspace": workspace_directory,
            "E2 capture work": capture_work_directory,
            "E2 probe bundle": probe_bundle_directory,
            "E1 output": e1_output_directory,
            "E1 phase ledger": e1_phase_run,
        }
    )
    output = normalized_paths["E2 phase ledger"]
    workspace_directory = normalized_paths["E2 workspace"]
    capture_work_directory = normalized_paths["E2 capture work"]
    probe_bundle_directory = normalized_paths["E2 probe bundle"]
    e1_output_directory = normalized_paths["E1 output"]
    e1_phase_run = normalized_paths["E1 phase ledger"]
    if output.exists():
        raise FrozenArtifactError(f"refusing to overwrite E2 phase run: {output}")
    capture_work = Path(capture_work_directory).absolute()
    probe_root = Path(probe_bundle_directory).absolute()
    e1_output = Path(e1_output_directory).absolute()
    capture_fingerprint = sha256_path(capture_work)
    probe_fingerprint = sha256_path(probe_root)
    e1_output_fingerprint = sha256_path(e1_output)

    workspace = verify_e2_workspace(workspace_directory)
    if workspace.plan_identity != expected_workspace_plan_identity:
        raise FrozenArtifactError("E2 workspace plan differs from the expected identity")
    capture = verify_e2_capture_work(
        capture_work,
        workspace=workspace,
        questions=questions,
        prompts=prompts,
        e1_sources=e1_sources,
        require_complete=True,
    )
    if capture.get("capture_plan_identity") != expected_capture_plan_identity:
        raise FrozenArtifactError("E2 capture plan differs from the expected identity")
    probe_bundle = verify_e2_probe_bundle(probe_root, workspace=workspace)
    if probe_bundle.manifest_digest != expected_probe_manifest_digest:
        raise FrozenArtifactError("E2 probe manifest differs from the expected identity")
    if not probe_bundle.scientific_eligible or not workspace.protocol.scientific_eligible:
        raise DataValidationError("E2 finalization requires the exact scientific protocol")
    if workspace.input_fingerprints.get("e1_output") != e1_output_fingerprint:
        raise FrozenArtifactError("E2 E1 outcome-label input differs from the capture workspace")
    probe_plan = _read_json_object(probe_root / "plan.json", "verified E2 probe plan")
    expected_prompt_hashes = {
        prompt_id: hashlib.sha256(prompts[prompt_id].text.encode("utf-8")).hexdigest()
        for prompt_id in ("P0-neutral", "P3-forced-answer")
    }
    if (
        probe_plan.get("split_manifest_digest") != split_manifest_digest
        or probe_plan.get("prompt_template_sha256") != expected_prompt_hashes
    ):
        raise FrozenArtifactError("E2 phase inputs differ from the verified probe plan")
    activation = verify_activation_store(
        workspace.directory / "activations",
        expected_spec=workspace.activation_spec,
        require_complete=True,
    )
    if capture.get("activation_chain_head") != activation.chain_head:
        raise FrozenArtifactError("E2 capture and activation chain heads differ")

    study = load_study_protocol(study_config)
    prior = PhaseRunLedger.open(e1_phase_run, study=study)
    e1_completion = prior.verify_complete()
    _verify_e1_output_completion_binding(
        e1_output,
        completion=e1_completion,
        contract_digest=prior.contract.digest,
    )
    conditions, question_ids, records = _phase_material(
        study_digest=study.digest,
        workspace=workspace,
        probe_bundle=probe_bundle,
        prompts=prompts,
        split_manifest_digest=split_manifest_digest,
        capture_plan_identity=str(capture["capture_plan_identity"]),
        capture_work_sha256=capture_fingerprint,
        activation_chain_head=str(activation.chain_head),
    )
    input_artifacts = {
        "E1_outcome_labels": e1_output,
        "activation_feature_schemas": probe_root,
    }
    contract = PhaseRunContract(
        phase=ExperimentPhase.E2,
        study_protocol_digest=study.digest,
        conditions=conditions,
        question_ids_by_benchmark=question_ids,
        input_fingerprints={
            "E1_outcome_labels": e1_output_fingerprint,
            "activation_feature_schemas": probe_fingerprint,
        },
        prerequisite_digests={ExperimentPhase.E1.value: e1_completion.completion_digest},
        required_gates=study.phase(ExperimentPhase.E2).gates,
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
            prerequisite_runs={ExperimentPhase.E1: e1_phase_run},
        )
        for offset in range(0, len(records), 1_000):
            ledger.checkpoint(records[offset : offset + 1_000])
        evidence_path = evidence / f"{_GATE}.json"
        write_gate_evidence(
            evidence_path,
            phase=ExperimentPhase.E2,
            gate=_GATE,
            contract_digest=contract.digest,
            record_set_digest=ledger.record_set_digest(),
            observations=_gate_observations(ledger.records()),
        )
        result = ledger.evaluate_gate(_GATE, evidence_path)
        _validate_gate_replay(result, probe_bundle)
        terminal: PhaseCompletion | PhaseFalsification
        if result.passed:
            terminal = ledger.finalize({_GATE: result})
            ledger.verify_complete()
        else:
            terminal = ledger.finalize_falsified({_GATE: result})
            ledger.verify_falsified()

        replayed_workspace = verify_e2_workspace(workspace.directory)
        replayed_activation = verify_activation_store(
            replayed_workspace.directory / "activations",
            expected_spec=replayed_workspace.activation_spec,
            require_complete=True,
        )
        if (
            replayed_workspace.plan_identity != workspace.plan_identity
            or replayed_activation.chain_head != activation.chain_head
            or sha256_path(capture_work) != capture_fingerprint
            or sha256_path(probe_root) != probe_fingerprint
            or sha256_path(e1_output) != e1_output_fingerprint
        ):
            raise FrozenArtifactError("E2 source evidence changed during phase finalization")
        if output.exists():
            raise FrozenArtifactError(f"E2 phase output appeared during finalization: {output}")
        os.replace(stage, output)
        published = PhaseRunLedger.open(output, study=study)
        if result.passed:
            published.verify_complete()
        else:
            published.verify_falsified()
        return terminal
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        if evidence.exists():
            shutil.rmtree(evidence)

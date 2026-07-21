"""Immutable E3 phase completion/falsification from all staged native runs."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from mfh.contracts import ActivationSite, Outcome, Question, TokenScope
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.evaluation.grading import deterministic_short_answer_grade, triviaqa_scores
from mfh.evaluation.metrics import metric_bundle
from mfh.experiments.e3_execution import E3ExecutionAssets, E3ExecutionResult
from mfh.experiments.e3_runner import (
    load_e3_evaluation_snapshot,
)
from mfh.experiments.e3_schedule import E3Condition, E3Protocol, e3_stage_row_counts
from mfh.experiments.e3_selection import VerifiedE3StageSelection
from mfh.experiments.model_selection import validate_active_study_artifact_paths
from mfh.experiments.protocol import ExperimentPhase, StudyProtocol
from mfh.provenance import sha256_file, sha256_path, stable_hash

_STAGES = (
    "geometry",
    "alpha",
    "scope",
    "controls",
    "cross-prompt",
    "P3-diagnostic",
    "final",
)
_FILES = frozenset(
    {
        "manifest.json",
        "stage-receipts.json",
        "condition-metrics.json",
        "primary-gate.json",
        "analysis-surface.json",
        "analysis-evidence.json",
    }
)
_SCIENTIFIC_SURFACE_COUNTS = MappingProxyType(
    {
        "geometry": 42,
        "alpha": 12,
        "scope": 12,
        "controls": 16,
        "cross-prompt": 8,
        "P3-diagnostic": 2,
        "final": 13,
    }
)


def _inputs(
    stage: str,
    receipts: Mapping[str, VerifiedE3StageSelection],
) -> VerifiedE3StageSelection | None:
    predecessor = {
        "geometry": None,
        "alpha": receipts.get("geometry"),
        "scope": receipts.get("alpha"),
        "controls": receipts.get("scope"),
        "cross-prompt": receipts.get("scope"),
        "P3-diagnostic": receipts.get("scope"),
        "final": receipts.get("scope"),
    }[stage]
    if predecessor is not None:
        predecessor.assert_current()
    return predecessor


def _build_phase_contract(
    *,
    study: StudyProtocol | None,
    input_artifacts: Mapping[str, str | Path] | None,
    prerequisite_runs: Mapping[str, str | Path] | None,
    output_artifacts: Mapping[str, str | Path] | None,
) -> Mapping[str, Any] | None:
    if (
        study is None
        and input_artifacts is None
        and prerequisite_runs is None
        and output_artifacts is None
    ):
        return None
    if (
        study is None
        or input_artifacts is None
        or prerequisite_runs is None
        or output_artifacts is None
    ):
        raise DataValidationError("E3 scientific provenance inputs must be supplied together")
    phase = study.phase(ExperimentPhase.E3)
    required_inputs = set(phase.required_inputs) | set(phase.freeze_fields)
    required_prerequisites = {value.value for value in phase.prerequisites}
    if (
        set(input_artifacts) != required_inputs
        or set(prerequisite_runs) != required_prerequisites
        or set(output_artifacts) != {"E3_static_vectors"}
    ):
        raise DataValidationError("E3 phase provenance differs from its study contract")
    inputs: dict[str, str] = {}
    for name, raw_path in sorted(input_artifacts.items()):
        path = Path(raw_path).resolve()
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise DataValidationError(f"E3 phase input is invalid: {name}")
        inputs[name] = sha256_path(path)
    from mfh.experiments.runner import PhaseRunLedger

    prerequisites: dict[str, str] = {}
    for name, raw_path in sorted(prerequisite_runs.items()):
        path = Path(raw_path).resolve()
        completion = PhaseRunLedger.open(path, study=study).verify_complete()
        if completion.phase.value != name:
            raise DataValidationError(f"E3 prerequisite {name} resolves to another phase")
        prerequisites[name] = completion.completion_digest
    outputs: dict[str, str] = {}
    for name, raw_path in sorted(output_artifacts.items()):
        path = Path(raw_path).resolve()
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise DataValidationError(f"E3 phase output is invalid: {name}")
        outputs[name] = sha256_path(path)
    body = {
        "schema_version": 2,
        "phase": ExperimentPhase.E3.value,
        "study_protocol_digest": study.digest,
        "input_artifacts": inputs,
        "prerequisite_runs": prerequisites,
        "output_artifacts": outputs,
    }
    return MappingProxyType({**body, "contract_digest": stable_hash(body)})


def _verify_phase_contract(
    value: object, *, study: StudyProtocol
) -> tuple[str, Mapping[str, str], Mapping[str, str], Mapping[str, str]]:
    if not isinstance(value, Mapping):
        raise FrozenArtifactError("E3 scientific phase contract is missing")
    body = dict(value)
    contract_digest = body.pop("contract_digest", None)
    if (
        set(body)
        != {
            "schema_version",
            "phase",
            "study_protocol_digest",
            "input_artifacts",
            "prerequisite_runs",
            "output_artifacts",
        }
        or body.get("schema_version") != 2
        or body.get("phase") != ExperimentPhase.E3.value
        or body.get("study_protocol_digest") != study.digest
        or contract_digest != stable_hash(body)
        or not isinstance(body.get("input_artifacts"), Mapping)
        or not isinstance(body.get("prerequisite_runs"), Mapping)
        or not isinstance(body.get("output_artifacts"), Mapping)
    ):
        raise FrozenArtifactError("E3 scientific phase contract differs")
    phase = study.phase(ExperimentPhase.E3)
    inputs = body["input_artifacts"]
    prerequisites = body["prerequisite_runs"]
    outputs = body["output_artifacts"]
    if (
        set(inputs) != set(phase.required_inputs) | set(phase.freeze_fields)
        or set(prerequisites) != {item.value for item in phase.prerequisites}
        or set(outputs) != {"E3_static_vectors"}
    ):
        raise FrozenArtifactError("E3 scientific phase contract inventory differs")
    for name, fingerprint in inputs.items():
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise FrozenArtifactError(f"E3 phase input fingerprint is invalid: {name}")
    for name, completion_digest in prerequisites.items():
        if (
            not isinstance(completion_digest, str)
            or len(completion_digest) != 64
            or any(character not in "0123456789abcdef" for character in completion_digest)
        ):
            raise FrozenArtifactError(f"E3 prerequisite digest is invalid: {name}")
    for name, fingerprint in outputs.items():
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise FrozenArtifactError(f"E3 output fingerprint is invalid: {name}")
    assert isinstance(contract_digest, str)
    return (
        contract_digest,
        MappingProxyType({str(name): str(value) for name, value in inputs.items()}),
        MappingProxyType({str(name): str(value) for name, value in prerequisites.items()}),
        MappingProxyType({str(name): str(value) for name, value in outputs.items()}),
    )


def _question_body(question: Question) -> dict[str, Any]:
    return {
        "question_id": question.question_id,
        "benchmark": question.benchmark,
        "text": question.text,
        "aliases": list(question.aliases),
        "split": question.split,
        "entities": list(question.entities),
        "metadata": dict(question.metadata),
    }


def _question_fingerprint(question: Question) -> str:
    return stable_hash(_question_body(question))


def _condition_from_dict(value: object) -> E3Condition:
    expected = {
        "stage",
        "method",
        "extraction_method",
        "training_prompt_id",
        "apply_prompt_id",
        "layer",
        "site",
        "standardized_alpha",
        "token_scope",
        "source_layer",
        "source_site",
        "control",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise FrozenArtifactError("E3 analysis condition schema differs")
    try:
        return E3Condition(
            stage=str(value["stage"]),
            method=str(value["method"]),
            extraction_method=(
                str(value["extraction_method"]) if value["extraction_method"] is not None else None
            ),
            training_prompt_id=(
                str(value["training_prompt_id"])
                if value["training_prompt_id"] is not None
                else None
            ),
            apply_prompt_id=str(value["apply_prompt_id"]),
            layer=value["layer"],
            site=(ActivationSite(value["site"]) if value["site"] is not None else None),
            standardized_alpha=value["standardized_alpha"],
            token_scope=(
                TokenScope(value["token_scope"]) if value["token_scope"] is not None else None
            ),
            source_layer=value["source_layer"],
            source_site=(
                ActivationSite(value["source_site"]) if value["source_site"] is not None else None
            ),
            control=str(value["control"]) if value["control"] is not None else None,
        )
    except (TypeError, ValueError, DataValidationError) as exc:
        raise FrozenArtifactError(f"E3 analysis condition is invalid: {exc}") from exc


def _question_from_dict(value: object) -> Question:
    expected = {
        "question_id",
        "benchmark",
        "text",
        "aliases",
        "split",
        "entities",
        "metadata",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise FrozenArtifactError("E3 analysis question schema differs")
    aliases = value["aliases"]
    entities = value["entities"]
    metadata = value["metadata"]
    if (
        not isinstance(aliases, list)
        or not isinstance(entities, list)
        or not isinstance(metadata, dict)
    ):
        raise FrozenArtifactError("E3 analysis question payload is invalid")
    try:
        return Question(
            question_id=str(value["question_id"]),
            benchmark=str(value["benchmark"]),
            text=str(value["text"]),
            aliases=tuple(str(item) for item in aliases),
            split=str(value["split"]) if value["split"] is not None else None,
            entities=tuple(str(item) for item in entities),
            metadata=metadata,
        )
    except (TypeError, ValueError, DataValidationError) as exc:
        raise FrozenArtifactError(f"E3 analysis question is invalid: {exc}") from exc


def _portable_result(
    value: object,
    *,
    condition: E3Condition,
    question: Question,
) -> E3ExecutionResult:
    expected = {
        "condition_id",
        "question_id",
        "rendered_prompt_sha256",
        "prompt_token_ids_sha256",
        "raw_output",
        "output_token_ids",
        "outcome",
        "exact_match",
        "token_f1",
        "generation_latency_seconds",
        "input_tokens",
        "output_tokens",
        "stop_type",
        "peak_memory_bytes",
        "intervention_trace",
        "hook_applications",
        "actual_delta_norm",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise FrozenArtifactError("E3 portable result schema differs")
    tokens = value["output_token_ids"]
    trace = value["intervention_trace"]
    if not isinstance(tokens, list) or (trace is not None and not isinstance(trace, dict)):
        raise FrozenArtifactError("E3 portable result payload is invalid")
    try:
        result = E3ExecutionResult(
            condition_id=value["condition_id"],
            question_id=value["question_id"],
            rendered_prompt_sha256=value["rendered_prompt_sha256"],
            prompt_token_ids_sha256=value["prompt_token_ids_sha256"],
            raw_output=value["raw_output"],
            output_token_ids=tuple(tokens),
            outcome=Outcome(value["outcome"]),
            exact_match=value["exact_match"],
            token_f1=value["token_f1"],
            generation_latency_seconds=value["generation_latency_seconds"],
            input_tokens=value["input_tokens"],
            output_tokens=value["output_tokens"],
            stop_type=value["stop_type"],
            peak_memory_bytes=value["peak_memory_bytes"],
            intervention_trace=trace,
            hook_applications=value["hook_applications"],
            actual_delta_norm=value["actual_delta_norm"],
        )
    except (TypeError, ValueError, DataValidationError) as exc:
        raise FrozenArtifactError(f"E3 portable result is invalid: {exc}") from exc
    expected_outcome = deterministic_short_answer_grade(result.raw_output, question.aliases)
    exact_match, token_f1 = triviaqa_scores(result.raw_output, question.aliases)
    if (
        result.condition_id != condition.condition_id
        or result.question_id != question.question_id
        or result.outcome is not expected_outcome
        or result.exact_match != float(exact_match)
        or result.token_f1 != float(token_f1)
        or (condition.method == "M0")
        is not (
            result.intervention_trace is None
            and result.hook_applications == 0
            and result.actual_delta_norm == 0
        )
        or (condition.method != "M0" and result.intervention_trace is None)
    ):
        raise FrozenArtifactError("E3 portable result does not replay")
    if result.intervention_trace is not None:
        expected_trace = {
            "extraction_method": condition.extraction_method,
            "training_prompt_id": condition.training_prompt_id,
            "source_layer": condition.source_layer,
            "source_site": condition.source_site.value if condition.source_site else None,
            "target_layer": condition.layer,
            "target_site": condition.site.value if condition.site else None,
            "standardized_alpha": condition.standardized_alpha,
            "token_scope": condition.token_scope.value if condition.token_scope else None,
            "control": condition.control,
            "hook_applications": result.hook_applications,
            "actual_delta_norm": result.actual_delta_norm,
        }
        if any(result.intervention_trace.get(key) != item for key, item in expected_trace.items()):
            raise FrozenArtifactError("E3 portable intervention trace does not replay")
    return result


def _surface_cell(
    *,
    stage: str,
    condition: E3Condition,
    metrics: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    training_prompt = condition.training_prompt_id or "unclassified"
    extraction = condition.extraction_method or condition.method
    control = condition.control or "none"
    site = condition.site.value if condition.site is not None else "none"
    scope = condition.token_scope.value if condition.token_scope is not None else "none"
    key = (
        f"{stage}|triviaqa|apply_{condition.apply_prompt_id}|"
        f"train_{training_prompt}|{condition.method}|extract_{extraction}|"
        f"control_{control}|site_{site}|scope_{scope}|"
        f"layer_{condition.layer}|alpha_{condition.standardized_alpha:.12g}|"
        f"{stable_hash(condition.condition_id)}"
    )
    return key, {
        "accuracy": metrics["accuracy"],
        "coverage": metrics["coverage"],
        "risk": metrics["hallucination_risk"],
        "question_count": metrics["total"],
        "layer": condition.layer,
        "alpha": condition.standardized_alpha,
        "condition_id": condition.condition_id,
        "stage_sha256": stable_hash(stage),
        "apply_prompt_sha256": stable_hash(condition.apply_prompt_id),
        "training_prompt_sha256": stable_hash(training_prompt),
        "method_sha256": stable_hash(condition.method),
        "extraction_sha256": stable_hash(extraction),
        "control_sha256": stable_hash(control),
        "site_sha256": stable_hash(site),
        "token_scope_sha256": stable_hash(scope),
    }


def _derive(
    *,
    stage_runs: Mapping[str, str | Path],
    stage_assets: Mapping[str, E3ExecutionAssets],
    stage_questions: Mapping[str, Sequence[Question]],
    selection_receipts: Mapping[str, VerifiedE3StageSelection],
    phase_contract: Mapping[str, Any] | None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    if (
        set(stage_runs) != set(_STAGES)
        or set(stage_assets) != set(_STAGES)
        or set(stage_questions) != set(_STAGES)
        or set(selection_receipts) != {"geometry", "alpha", "scope"}
    ):
        raise DataValidationError("E3 phase inputs differ from the seven frozen stages")
    stage_receipts: dict[str, Any] = {}
    condition_metrics: dict[str, Any] = {}
    analysis_surface: dict[str, Any] = {}
    analysis_stages: dict[str, Any] = {}
    final_results: tuple[E3ExecutionResult, ...] = ()
    final_assets: E3ExecutionAssets | None = None
    scientific = True
    for stage in _STAGES:
        predecessor = _inputs(stage, selection_receipts)
        verification, results = load_e3_evaluation_snapshot(
            stage_runs[stage],
            stage=stage,
            assets=stage_assets[stage],
            evaluation_questions=stage_questions[stage],
            selection_receipt=predecessor,
            require_complete=True,
        )
        by_condition: dict[str, list[Any]] = {}
        for result in results:
            by_condition.setdefault(result.condition_id, []).append(result.outcome)
        for condition_id, outcomes in by_condition.items():
            condition_metrics[condition_id] = metric_bundle(outcomes).to_dict()
        for condition in stage_assets[stage].conditions.values():
            if condition.method == "M0" or condition.layer is None:
                continue
            metrics = condition_metrics[condition.condition_id]
            if any(
                metrics[name] is None for name in ("accuracy", "coverage", "hallucination_risk")
            ):
                raise DataValidationError(
                    "E3 analysis surface contains an undefined condition metric"
                )
            key, cell = _surface_cell(stage=stage, condition=condition, metrics=metrics)
            if key in analysis_surface:
                raise DataValidationError("E3 analysis surface contains a duplicate cell")
            analysis_surface[key] = cell
        questions = tuple(stage_questions[stage])
        analysis_stages[stage] = {
            "plan_identity": verification["plan_identity"],
            "record_chain_head": verification["record_chain_head"],
            "record_set_digest": verification["record_set_digest"],
            "session_chain_head": verification["session_chain_head"],
            "session_set_digest": verification["session_set_digest"],
            "conditions": [
                condition.to_dict() for condition in stage_assets[stage].conditions.values()
            ],
            "question_fingerprints": {
                question.question_id: _question_fingerprint(question) for question in questions
            },
            "questions": [_question_body(question) for question in questions],
            "results": [result.to_dict() for result in results],
        }
        stage_receipts[stage] = {
            "plan_identity": verification["plan_identity"],
            "record_chain_head": verification["record_chain_head"],
            "session_chain_head": verification["session_chain_head"],
            "session_set_digest": verification["session_set_digest"],
            "records_completed": verification["records_completed"],
            "record_set_digest": verification["record_set_digest"],
            "condition_count": len(stage_assets[stage].conditions),
            "question_count": len(questions),
            "analysis_cell_count": sum(
                condition.method != "M0" for condition in stage_assets[stage].conditions.values()
            ),
            "maximum_peak_memory_bytes": verification["maximum_peak_memory_bytes"],
            "wall_time_seconds": verification["wall_time_seconds"],
            "scientific_eligible": verification["scientific_eligible"],
        }
        scientific = scientific and bool(verification["scientific_eligible"])
        if stage == "final":
            final_results = results
            final_assets = stage_assets[stage]
    assert final_assets is not None
    baseline = [
        condition
        for condition in final_assets.conditions.values()
        if condition.method == "M0" and condition.apply_prompt_id == "P0-neutral"
    ]
    intervention = [
        condition
        for condition in final_assets.conditions.values()
        if condition.method == "M1-R"
        and condition.apply_prompt_id == "P0-neutral"
        and condition.control is None
    ]
    if len(baseline) != 1 or len(intervention) != 1:
        raise DataValidationError("E3 final run lacks the unique primary P0 M0/M1-R pair")
    baseline_metrics = condition_metrics[baseline[0].condition_id]
    intervention_metrics = condition_metrics[intervention[0].condition_id]
    baseline_risk = baseline_metrics["hallucination_risk"]
    intervention_risk = intervention_metrics["hallucination_risk"]
    baseline_coverage = baseline_metrics["coverage"]
    intervention_coverage = intervention_metrics["coverage"]
    if any(
        value is None
        for value in (
            baseline_risk,
            intervention_risk,
            baseline_coverage,
            intervention_coverage,
        )
    ):
        raise DataValidationError("E3 primary pair has undefined coverage or risk")
    risk_change = float(intervention_risk) - float(baseline_risk)
    coverage_change = float(intervention_coverage) - float(baseline_coverage)
    gate = {
        "schema_version": 1,
        "gate": "factuality_gain_not_explained_by_coverage_loss",
        "primary_comparison": "P0-neutral:M1-R-vs-M0",
        "diagnostics_excluded": ["M1-P", "P2-calibrated-abstention", "P3-forced-answer"],
        "baseline_condition_id": baseline[0].condition_id,
        "intervention_condition_id": intervention[0].condition_id,
        "question_count": len(final_results) // len(final_assets.conditions),
        "hallucination_risk_change": risk_change,
        "coverage_change": coverage_change,
        "maximum_coverage_loss": 0.05,
        "passed": risk_change < 0 and coverage_change >= -0.05,
    }
    selection_body = {
        stage: {
            "selection_digest": receipt.selection_digest,
            "artifact_sha256": receipt.artifact_sha256,
        }
        for stage, receipt in sorted(selection_receipts.items())
    }
    manifest = {
        "schema_version": 2,
        "phase": "E3",
        "status": "complete" if gate["passed"] else "falsified",
        "runner_source_sha256": sha256_file(Path(__file__)),
        "stage_receipts_digest": stable_hash(stage_receipts),
        "condition_metrics_digest": stable_hash(condition_metrics),
        "primary_gate_digest": stable_hash(gate),
        "selection_receipts": selection_body,
        "phase_contract": dict(phase_contract) if phase_contract is not None else None,
        "analysis_surface_digest": stable_hash(analysis_surface),
        "analysis_evidence_digest": stable_hash({"schema_version": 1, "stages": analysis_stages}),
        "scientific_eligible": scientific,
    }
    analysis_evidence = {"schema_version": 1, "stages": analysis_stages}
    return manifest, stage_receipts, condition_metrics, gate, analysis_surface, analysis_evidence


def finalize_e3_phase(
    destination: str | Path,
    *,
    stage_runs: Mapping[str, str | Path],
    stage_assets: Mapping[str, E3ExecutionAssets],
    stage_questions: Mapping[str, Sequence[Question]],
    selection_receipts: Mapping[str, VerifiedE3StageSelection],
    study: StudyProtocol | None = None,
    input_artifacts: Mapping[str, str | Path] | None = None,
    prerequisite_runs: Mapping[str, str | Path] | None = None,
    allow_non_scientific: bool = False,
) -> Mapping[str, Any]:
    normalized = validate_active_study_artifact_paths(
        {
            "E3 phase result": destination,
            **{f"E3 stage {name}": path for name, path in stage_runs.items()},
        }
    )
    destination = normalized["E3 phase result"]
    stage_runs = {name: normalized[f"E3 stage {name}"] for name in stage_runs}
    vector_bundle_paths = {
        assets.vector_bundle_directory.resolve() for assets in stage_assets.values()
    }
    if len(vector_bundle_paths) != 1:
        raise DataValidationError("E3 stages do not share one frozen static-vector bundle")
    phase_outputs = (
        None
        if study is None and input_artifacts is None and prerequisite_runs is None
        else {"E3_static_vectors": next(iter(vector_bundle_paths))}
    )
    phase_contract = _build_phase_contract(
        study=study,
        input_artifacts=input_artifacts,
        prerequisite_runs=prerequisite_runs,
        output_artifacts=phase_outputs,
    )
    values = _derive(
        stage_runs=stage_runs,
        stage_assets=stage_assets,
        stage_questions=stage_questions,
        selection_receipts=selection_receipts,
        phase_contract=phase_contract,
    )
    manifest, stage_values, metrics, gate, analysis_surface, analysis_evidence = values
    if not manifest["scientific_eligible"] and not allow_non_scientific:
        raise FrozenArtifactError("E3 phase is not scientifically eligible")
    if manifest["scientific_eligible"] and phase_contract is None:
        raise FrozenArtifactError("scientific E3 completion lacks E1/E2 and input provenance")
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite E3 phase result: {output}")
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        for name, value in {
            "stage-receipts.json": stage_values,
            "condition-metrics.json": metrics,
            "primary-gate.json": gate,
            "analysis-surface.json": analysis_surface,
            "analysis-evidence.json": analysis_evidence,
        }.items():
            (stage / name).write_text(
                json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        body = {**manifest, "manifest_digest": stable_hash(manifest)}
        with (stage / "manifest.json").open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(body, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return verify_e3_phase(
        output,
        stage_runs=stage_runs,
        stage_assets=stage_assets,
        stage_questions=stage_questions,
        selection_receipts=selection_receipts,
        study=study,
        input_artifacts=input_artifacts,
        prerequisite_runs=prerequisite_runs,
    )


def verify_e3_phase(
    directory: str | Path,
    *,
    stage_runs: Mapping[str, str | Path],
    stage_assets: Mapping[str, E3ExecutionAssets],
    stage_questions: Mapping[str, Sequence[Question]],
    selection_receipts: Mapping[str, VerifiedE3StageSelection],
    study: StudyProtocol | None = None,
    input_artifacts: Mapping[str, str | Path] | None = None,
    prerequisite_runs: Mapping[str, str | Path] | None = None,
) -> Mapping[str, Any]:
    source = Path(directory)
    if (
        source.is_symlink()
        or not source.is_dir()
        or {value.name for value in source.iterdir()} != _FILES
        or any(value.is_symlink() or not value.is_file() for value in source.iterdir())
    ):
        raise FrozenArtifactError("E3 phase result inventory differs")
    vector_bundle_paths = {
        assets.vector_bundle_directory.resolve() for assets in stage_assets.values()
    }
    if len(vector_bundle_paths) != 1:
        raise DataValidationError("E3 stages do not share one frozen static-vector bundle")
    phase_outputs = (
        None
        if study is None and input_artifacts is None and prerequisite_runs is None
        else {"E3_static_vectors": next(iter(vector_bundle_paths))}
    )
    phase_contract = _build_phase_contract(
        study=study,
        input_artifacts=input_artifacts,
        prerequisite_runs=prerequisite_runs,
        output_artifacts=phase_outputs,
    )
    manifest, stage_values, metrics, gate, analysis_surface, analysis_evidence = _derive(
        stage_runs=stage_runs,
        stage_assets=stage_assets,
        stage_questions=stage_questions,
        selection_receipts=selection_receipts,
        phase_contract=phase_contract,
    )
    expected = {
        "manifest.json": {**manifest, "manifest_digest": stable_hash(manifest)},
        "stage-receipts.json": stage_values,
        "condition-metrics.json": metrics,
        "primary-gate.json": gate,
        "analysis-surface.json": analysis_surface,
        "analysis-evidence.json": analysis_evidence,
    }
    for name, value in expected.items():
        expected_text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        try:
            observed = (source / name).read_text(encoding="utf-8")
        except OSError as exc:
            raise FrozenArtifactError(f"cannot read E3 phase result: {exc}") from exc
        if observed != expected_text:
            raise FrozenArtifactError(f"E3 phase {name} differs from deterministic replay")
    return MappingProxyType(
        {
            "valid": True,
            "status": manifest["status"],
            "manifest_digest": stable_hash(manifest),
            "primary_gate_passed": gate["passed"],
            "scientific_eligible": manifest["scientific_eligible"],
        }
    )


def _replay_analysis_evidence(
    evidence: object,
    *,
    stage_receipts: Mapping[str, Any],
    scientific_eligible: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if (
        not isinstance(evidence, dict)
        or set(evidence) != {"schema_version", "stages"}
        or evidence.get("schema_version") != 1
        or not isinstance(evidence.get("stages"), dict)
        or set(evidence["stages"]) != set(_STAGES)
    ):
        raise FrozenArtifactError("E3 analysis evidence schema differs")
    receipt_fields = {
        "plan_identity",
        "record_chain_head",
        "session_chain_head",
        "session_set_digest",
        "records_completed",
        "record_set_digest",
        "condition_count",
        "question_count",
        "analysis_cell_count",
        "maximum_peak_memory_bytes",
        "wall_time_seconds",
        "scientific_eligible",
    }
    stage_fields = {
        "plan_identity",
        "record_chain_head",
        "record_set_digest",
        "session_chain_head",
        "session_set_digest",
        "conditions",
        "question_fingerprints",
        "questions",
        "results",
    }
    scientific_rows = e3_stage_row_counts(E3Protocol())
    metrics: dict[str, Any] = {}
    surface: dict[str, Any] = {}
    final_conditions: tuple[E3Condition, ...] = ()
    for stage in _STAGES:
        raw_stage = evidence["stages"][stage]
        receipt = stage_receipts.get(stage)
        if (
            not isinstance(raw_stage, dict)
            or set(raw_stage) != stage_fields
            or not isinstance(receipt, Mapping)
            or set(receipt) != receipt_fields
        ):
            raise FrozenArtifactError("E3 analysis stage evidence differs")
        raw_conditions = raw_stage["conditions"]
        raw_questions = raw_stage["questions"]
        raw_results = raw_stage["results"]
        fingerprints = raw_stage["question_fingerprints"]
        if (
            not isinstance(raw_conditions, list)
            or not isinstance(raw_questions, list)
            or not isinstance(raw_results, list)
            or not isinstance(fingerprints, dict)
        ):
            raise FrozenArtifactError("E3 analysis stage payload is invalid")
        conditions = tuple(_condition_from_dict(value) for value in raw_conditions)
        questions = tuple(_question_from_dict(value) for value in raw_questions)
        condition_map = {value.condition_id: value for value in conditions}
        question_map = {value.question_id: value for value in questions}
        expected_fingerprints = {
            value.question_id: _question_fingerprint(value) for value in questions
        }
        if (
            not conditions
            or not questions
            or len(condition_map) != len(conditions)
            or len(question_map) != len(questions)
            or any(value.stage != stage for value in conditions)
            or fingerprints != expected_fingerprints
            or len(raw_results) != len(conditions) * len(questions)
        ):
            raise FrozenArtifactError("E3 analysis stage matrix differs")
        pairs: set[tuple[str, str]] = set()
        by_condition: dict[str, list[Outcome]] = {value.condition_id: [] for value in conditions}
        maximum_peak = 0
        for raw_result in raw_results:
            if not isinstance(raw_result, dict):
                raise FrozenArtifactError("E3 portable result is invalid")
            condition_id = raw_result.get("condition_id")
            question_id = raw_result.get("question_id")
            if not isinstance(condition_id, str) or not isinstance(question_id, str):
                raise FrozenArtifactError("E3 portable result identity is invalid")
            try:
                condition = condition_map[condition_id]
                question = question_map[question_id]
            except KeyError as exc:
                raise FrozenArtifactError("E3 portable result is outside its matrix") from exc
            pair_key = (condition_id, question_id)
            if pair_key in pairs:
                raise FrozenArtifactError("E3 portable result matrix contains duplicates")
            pairs.add(pair_key)
            result = _portable_result(raw_result, condition=condition, question=question)
            by_condition[condition_id].append(result.outcome)
            maximum_peak = max(maximum_peak, result.peak_memory_bytes)
        if pairs != {
            (condition.condition_id, question.question_id)
            for condition in conditions
            for question in questions
        }:
            raise FrozenArtifactError("E3 portable result matrix is incomplete")
        stage_metrics = {
            condition_id: metric_bundle(outcomes).to_dict()
            for condition_id, outcomes in by_condition.items()
        }
        if set(metrics).intersection(stage_metrics):
            raise FrozenArtifactError("E3 condition identities repeat across stages")
        metrics.update(stage_metrics)
        for condition in conditions:
            if condition.method == "M0" or condition.layer is None:
                continue
            cell_metrics = stage_metrics[condition.condition_id]
            if any(
                cell_metrics[name] is None
                for name in ("accuracy", "coverage", "hallucination_risk")
            ):
                raise FrozenArtifactError("E3 analysis surface metric is undefined")
            surface_key, cell = _surface_cell(
                stage=stage,
                condition=condition,
                metrics=cell_metrics,
            )
            if surface_key in surface:
                raise FrozenArtifactError("E3 analysis surface contains duplicate cells")
            surface[surface_key] = cell
        analysis_cells = sum(value.method != "M0" for value in conditions)
        digest_fields = (
            "plan_identity",
            "record_chain_head",
            "record_set_digest",
            "session_chain_head",
            "session_set_digest",
        )
        if (
            any(
                not isinstance(raw_stage[name], str)
                or len(raw_stage[name]) != 64
                or any(character not in "0123456789abcdef" for character in raw_stage[name])
                for name in digest_fields
            )
            or any(receipt[name] != raw_stage[name] for name in digest_fields)
            or receipt["records_completed"] != len(raw_results)
            or receipt["condition_count"] != len(conditions)
            or receipt["question_count"] != len(questions)
            or receipt["analysis_cell_count"] != analysis_cells
            or receipt["maximum_peak_memory_bytes"] != maximum_peak
            or type(receipt["wall_time_seconds"]) is not float
            or not math.isfinite(receipt["wall_time_seconds"])
            or receipt["wall_time_seconds"] < 0
            or type(receipt["scientific_eligible"]) is not bool
            or (scientific_eligible and receipt["scientific_eligible"] is not True)
        ):
            raise FrozenArtifactError("E3 stage receipt does not replay")
        if scientific_eligible and (
            len(raw_results) != scientific_rows[stage]
            or analysis_cells != _SCIENTIFIC_SURFACE_COUNTS[stage]
        ):
            raise FrozenArtifactError("E3 scientific stage cardinality differs")
        if stage == "final":
            final_conditions = conditions
    baseline = [
        value
        for value in final_conditions
        if value.method == "M0" and value.apply_prompt_id == "P0-neutral"
    ]
    treatment = [
        value
        for value in final_conditions
        if value.method == "M1-R"
        and value.apply_prompt_id == "P0-neutral"
        and value.control is None
    ]
    if len(baseline) != 1 or len(treatment) != 1:
        raise FrozenArtifactError("E3 portable primary pair differs")
    baseline_metrics = metrics[baseline[0].condition_id]
    treatment_metrics = metrics[treatment[0].condition_id]
    baseline_risk = baseline_metrics["hallucination_risk"]
    treatment_risk = treatment_metrics["hallucination_risk"]
    baseline_coverage = baseline_metrics["coverage"]
    treatment_coverage = treatment_metrics["coverage"]
    if any(
        value is None
        for value in (
            baseline_risk,
            treatment_risk,
            baseline_coverage,
            treatment_coverage,
        )
    ):
        raise FrozenArtifactError("E3 portable primary pair is undefined")
    risk_change = float(treatment_risk) - float(baseline_risk)
    coverage_change = float(treatment_coverage) - float(baseline_coverage)
    gate = {
        "schema_version": 1,
        "gate": "factuality_gain_not_explained_by_coverage_loss",
        "primary_comparison": "P0-neutral:M1-R-vs-M0",
        "diagnostics_excluded": ["M1-P", "P2-calibrated-abstention", "P3-forced-answer"],
        "baseline_condition_id": baseline[0].condition_id,
        "intervention_condition_id": treatment[0].condition_id,
        "question_count": int(baseline_metrics["total"]),
        "hallucination_risk_change": risk_change,
        "coverage_change": coverage_change,
        "maximum_coverage_loss": 0.05,
        "passed": risk_change < 0 and coverage_change >= -0.05,
    }
    return metrics, surface, gate


def load_e3_analysis_surface(
    directory: str | Path,
    *,
    expected_completion_digest: str,
    require_scientific: bool = True,
    study: StudyProtocol | None = None,
) -> tuple[Mapping[str, Mapping[str, Any]], str]:
    """Replay a digest-anchored publication surface from all seven E3 stages."""

    if (
        not isinstance(expected_completion_digest, str)
        or len(expected_completion_digest) != 64
        or any(character not in "0123456789abcdef" for character in expected_completion_digest)
    ):
        raise FrozenArtifactError("E3 analysis requires its externally anchored completion")

    source = Path(directory)
    if (
        source.is_symlink()
        or not source.is_dir()
        or {value.name for value in source.iterdir()} != _FILES
        or any(value.is_symlink() or not value.is_file() for value in source.iterdir())
    ):
        raise FrozenArtifactError("E3 analysis source inventory differs")
    try:
        values = {name: json.loads((source / name).read_text(encoding="utf-8")) for name in _FILES}
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read E3 analysis source: {exc}") from exc
    manifest_value = values["manifest.json"]
    surface = values["analysis-surface.json"]
    stage_receipts = values["stage-receipts.json"]
    metrics = values["condition-metrics.json"]
    gate = values["primary-gate.json"]
    analysis_evidence = values["analysis-evidence.json"]
    if not isinstance(manifest_value, dict):
        raise FrozenArtifactError("E3 analysis manifest is invalid")
    manifest_body = dict(manifest_value)
    manifest_digest = manifest_body.pop("manifest_digest", None)
    expected_manifest_fields = {
        "schema_version",
        "phase",
        "status",
        "runner_source_sha256",
        "stage_receipts_digest",
        "condition_metrics_digest",
        "primary_gate_digest",
        "selection_receipts",
        "phase_contract",
        "analysis_surface_digest",
        "analysis_evidence_digest",
        "scientific_eligible",
    }
    if (
        not isinstance(manifest_digest, str)
        or manifest_digest != expected_completion_digest
        or manifest_digest != stable_hash(manifest_body)
        or set(manifest_body) != expected_manifest_fields
        or manifest_body.get("schema_version") != 2
        or manifest_body.get("phase") != "E3"
        or manifest_body.get("status") != "complete"
        or manifest_body.get("runner_source_sha256") != sha256_file(Path(__file__))
        or type(manifest_body.get("scientific_eligible")) is not bool
        or (require_scientific and manifest_body.get("scientific_eligible") is not True)
        or not isinstance(stage_receipts, Mapping)
        or set(stage_receipts) != set(_STAGES)
        or manifest_body.get("stage_receipts_digest") != stable_hash(stage_receipts)
        or not isinstance(metrics, Mapping)
        or manifest_body.get("condition_metrics_digest") != stable_hash(metrics)
        or not isinstance(gate, Mapping)
        or gate.get("passed") is not True
        or manifest_body.get("primary_gate_digest") != stable_hash(gate)
        or not isinstance(surface, Mapping)
        or not surface
        or manifest_body.get("analysis_surface_digest") != stable_hash(surface)
        or manifest_body.get("analysis_evidence_digest") != stable_hash(analysis_evidence)
    ):
        raise FrozenArtifactError("E3 analysis source differs from its completion receipt")
    if require_scientific:
        if study is None:
            raise FrozenArtifactError("scientific E3 replay requires the frozen study protocol")
        _verify_phase_contract(manifest_body["phase_contract"], study=study)
    replayed_metrics, replayed_surface, replayed_gate = _replay_analysis_evidence(
        analysis_evidence,
        stage_receipts=stage_receipts,
        scientific_eligible=bool(manifest_body["scientific_eligible"]),
    )
    if (
        dict(metrics) != replayed_metrics
        or dict(surface) != replayed_surface
        or dict(gate) != replayed_gate
    ):
        raise FrozenArtifactError("E3 analysis source differs from portable stage replay")
    required = {
        "accuracy",
        "coverage",
        "risk",
        "question_count",
        "layer",
        "alpha",
        "condition_id",
        "stage_sha256",
        "apply_prompt_sha256",
        "training_prompt_sha256",
        "method_sha256",
        "extraction_sha256",
        "control_sha256",
        "site_sha256",
        "token_scope_sha256",
    }
    normalized: dict[str, Mapping[str, Any]] = {}
    for key, raw in surface.items():
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(raw, Mapping)
            or set(raw) != required
            or type(raw["question_count"]) is not int
            or raw["question_count"] <= 0
            or type(raw["layer"]) is not int
            or raw["layer"] < 0
            or any(
                isinstance(raw[name], bool)
                or not isinstance(raw[name], int | float)
                or not math.isfinite(float(raw[name]))
                for name in ("accuracy", "coverage", "risk", "alpha")
            )
            or any(not 0 <= float(raw[name]) <= 1 for name in ("accuracy", "coverage", "risk"))
            or any(
                not isinstance(raw[name], str)
                or len(raw[name]) != 64
                or any(character not in "0123456789abcdef" for character in raw[name])
                for name in required
                if name.endswith("_sha256") or name == "condition_id"
            )
        ):
            raise FrozenArtifactError("E3 analysis surface contains an invalid cell")
        normalized[key] = MappingProxyType(dict(raw))
    return MappingProxyType(normalized), manifest_digest


@dataclass(frozen=True, slots=True)
class VerifiedE3PhaseCompletion:
    """Phase-ledger-compatible view of the custom seven-stage E3 terminal artifact."""

    directory: Path
    study: StudyProtocol
    manifest_digest: str
    contract_digest: str
    record_count: int
    record_set_digest: str
    stage_record_digests: Mapping[str, str]
    gate_digest: str
    input_fingerprints: Mapping[str, str]
    prerequisite_digests: Mapping[str, str]
    output_fingerprints: Mapping[str, str]

    def verify_complete(self) -> Any:
        from mfh.experiments.runner import PhaseCompletion

        reopened = open_e3_phase_completion(
            self.directory,
            study=self.study,
            expected_completion_digest=self.manifest_digest,
        )
        if reopened != self:
            raise FrozenArtifactError("E3 completion changed after opening")
        return PhaseCompletion(
            phase=ExperimentPhase.E3,
            contract_digest=self.contract_digest,
            record_count=self.record_count,
            shard_fingerprints=self.stage_record_digests,
            record_set_digest=self.record_set_digest,
            gate_result_digests=MappingProxyType(
                {"factuality_gain_not_explained_by_coverage_loss": self.gate_digest}
            ),
            gate_file_fingerprints=MappingProxyType(
                {"primary-gate.json": sha256_file(self.directory / "primary-gate.json")}
            ),
            gate_artifact_fingerprints=MappingProxyType({}),
            completion_digest=self.manifest_digest,
        )


def open_e3_phase_completion(
    directory: str | Path,
    *,
    study: StudyProtocol,
    expected_completion_digest: str | None = None,
) -> VerifiedE3PhaseCompletion:
    """Open a complete scientific E3 artifact as a downstream prerequisite."""

    source = validate_active_study_artifact_paths({"E3 phase completion": directory})[
        "E3 phase completion"
    ]
    try:
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        receipts = json.loads((source / "stage-receipts.json").read_text(encoding="utf-8"))
        gate = json.loads((source / "primary-gate.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read E3 completion: {exc}") from exc
    if not isinstance(manifest, dict):
        raise FrozenArtifactError("E3 completion manifest is invalid")
    manifest_digest = manifest.get("manifest_digest")
    if not isinstance(manifest_digest, str) or (
        expected_completion_digest is not None and manifest_digest != expected_completion_digest
    ):
        raise FrozenArtifactError("E3 completion differs from its expected identity")
    load_e3_analysis_surface(
        source,
        expected_completion_digest=manifest_digest,
        require_scientific=True,
        study=study,
    )
    if not isinstance(receipts, Mapping) or not isinstance(gate, Mapping):
        raise FrozenArtifactError("E3 completion receipts are invalid")
    (
        contract_digest,
        input_fingerprints,
        prerequisite_digests,
        output_fingerprints,
    ) = _verify_phase_contract(manifest.get("phase_contract"), study=study)
    stage_digests = {
        name: str(receipt["record_set_digest"])
        for name, receipt in receipts.items()
        if isinstance(receipt, Mapping)
    }
    if set(stage_digests) != set(_STAGES):
        raise FrozenArtifactError("E3 completion stage identities are incomplete")
    return VerifiedE3PhaseCompletion(
        directory=source,
        study=study,
        manifest_digest=manifest_digest,
        contract_digest=contract_digest,
        record_count=sum(int(receipt["records_completed"]) for receipt in receipts.values()),
        record_set_digest=stable_hash(stage_digests),
        stage_record_digests=MappingProxyType(stage_digests),
        gate_digest=stable_hash(dict(gate)),
        input_fingerprints=input_fingerprints,
        prerequisite_digests=prerequisite_digests,
        output_fingerprints=output_fingerprints,
    )

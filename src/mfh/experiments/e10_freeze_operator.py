"""Runnable post-E9 development and freeze workflow for one-shot E10."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from mfh.config import load_model_spec, load_prompt_specs
from mfh.contracts import PromptSpec, Question
from mfh.data.io import read_questions
from mfh.data.reviewed_splits import validate_reviewed_split_snapshot
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.experiments.confirmatory_operator import (
    ConfirmatoryRunbook,
    _native_runtime,
    preflight_confirmatory_runbook,
    verify_confirmatory_runbook,
)
from mfh.experiments.e8_operator import (
    E8Runbook,
    _base_context,
    _e6_components,
    _feature_schema,
)
from mfh.experiments.e10_composite import (
    _e6_runtime_artifact,
    build_e10_contract,
    derive_e10_composite_provenance,
    e10_intervention,
    write_e10_composite_from_promotions,
    write_e10_freeze_inputs,
)
from mfh.experiments.e10_early_probe import (
    _reviewed_questions_from_e1,
    fit_e10_early_probe_selection,
    prepare_e10_early_probe_capture,
    run_e10_early_probe_capture,
    verify_e10_early_probe_capture,
)
from mfh.experiments.model_selection import validate_active_study_artifact_paths
from mfh.experiments.protocol import ExperimentPhase
from mfh.experiments.runner import (
    open_phase_prerequisite,
    write_frozen_component_selection,
    write_frozen_question_bundle,
)
from mfh.experiments.snapshots import validate_execution_snapshot
from mfh.provenance import sha256_file, sha256_path

_FACTUAL_FILES = {
    "triviaqa": "triviaqa.jsonl",
    "simpleqa_verified": "simpleqa_verified.jsonl",
    "aa_omniscience_public_600": "aa_omniscience_public_600.jsonl",
}
_SIDE_BENCHMARKS = (
    "ifeval",
    "mmlu_pro",
    "wikitext103",
    "xstest",
    "strongreject_or_harmbench",
    "language_consistency",
)


@dataclass(frozen=True, slots=True)
class _E10Context:
    e8: E8Runbook
    e9: ConfirmatoryRunbook
    study: Any
    model: Any
    prompt: PromptSpec
    prerequisite_paths: Mapping[str, Path]
    prerequisite_digests: Mapping[str, str]
    provenance: Mapping[str, object]
    controller: Path
    runtime_artifact: Path
    early_questions: Mapping[str, tuple[Question, ...]]
    split_manifest_digest: str


def _e1_split_manifest_digest(e1: Any) -> str:
    try:
        evidence = json.loads(
            (e1.directory / "creation-evidence.json").read_text(encoding="utf-8")
        )
        descriptor = evidence["input_artifacts"]["deduplicated_splits"]
        location = Path(descriptor["location"])
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError("E1 creation evidence lacks reviewed splits") from exc
    if not location.is_absolute():
        location = (e1.directory / location).resolve()
    manifest = validate_reviewed_split_snapshot(location)
    digest = manifest.get("manifest_digest")
    if not isinstance(digest, str) or len(digest) != 64:
        raise FrozenArtifactError("E1 reviewed split manifest identity is invalid")
    return digest


def _write_once_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise FrozenArtifactError(f"refusing to overwrite E10 runbook: {path}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _context(e8_runbook: str | Path, e9_runbook: str | Path) -> _E10Context:
    e8 = E8Runbook.load(e8_runbook)
    e8_context = _base_context(e8)
    e9 = ConfirmatoryRunbook.load(e9_runbook)
    status = verify_confirmatory_runbook(e9)
    if status["status"] != "complete":
        raise DataValidationError("E10 preparation requires terminal complete E9")
    if e9.study_protocol != e8.study_protocol or e9.model_config != e8.model_config:
        raise FrozenArtifactError("E8 and E9 operator inputs use different studies")
    paths = dict(e9.prerequisite_runs)
    paths["E9"] = e9.run_directory
    expected = {phase.value for phase in tuple(ExperimentPhase)[:-1]}
    if set(paths) != expected:
        raise DataValidationError("E10 preparation requires exact E0-E9 paths")
    digests: dict[str, str] = {}
    for name, path in paths.items():
        phase = ExperimentPhase(name)
        completion = open_phase_prerequisite(
            path, phase=phase, study=e8_context.study
        ).verify_complete()
        digests[name] = completion.completion_digest
    provenance = derive_e10_composite_provenance(
        study=e8_context.study,
        prerequisite_runs=paths,
    )
    prompts = {value.prompt_id: value for value in load_prompt_specs(e8.prompt_config)}
    try:
        prompt = prompts[str(provenance["selected_prompt_id"])]
    except KeyError as exc:
        raise FrozenArtifactError("E10 promoted prompt is unavailable") from exc
    components = _e6_components(e8, e8_context, _feature_schema(e8, e8_context))
    e1 = open_phase_prerequisite(
        paths["E1"], phase=ExperimentPhase.E1, study=e8_context.study
    )
    split_manifest_digest = _e1_split_manifest_digest(e1)
    early = _reviewed_questions_from_e1(
        e1,
        split_manifest_digest=split_manifest_digest,
        partitions=("T-controller", "T-dev"),
    )
    e6 = open_phase_prerequisite(
        paths["E6"], phase=ExperimentPhase.E6, study=e8_context.study
    )
    return _E10Context(
        e8=e8,
        e9=e9,
        study=e8_context.study,
        model=load_model_spec(e8.model_config),
        prompt=prompt,
        prerequisite_paths=MappingProxyType(paths),
        prerequisite_digests=MappingProxyType(digests),
        provenance=provenance,
        controller=components.controller_path,
        runtime_artifact=_e6_runtime_artifact(e6),
        early_questions=early,
        split_manifest_digest=split_manifest_digest,
    )


def prepare_e10_freeze_suite(
    directory: str | Path,
    *,
    e8_runbook: str | Path,
    e9_runbook: str | Path,
) -> Mapping[str, Any]:
    """Freeze the exact 10,000-row early-token development capture plan."""

    root = validate_active_study_artifact_paths({"E10 freeze suite": directory})[
        "E10 freeze suite"
    ]
    context = _context(e8_runbook, e9_runbook)
    root.mkdir(parents=True, exist_ok=True)
    capture = root / "early-probe-capture"
    plan = prepare_e10_early_probe_capture(
        capture,
        study=context.study,
        e1_run=context.prerequisite_paths["E1"],
        e8_run=context.prerequisite_paths["E8"],
        questions_by_partition=context.early_questions,
        prompt=context.prompt,
        controller_artifact=context.controller,
        selection_provenance=context.provenance,
        split_manifest_digest=context.split_manifest_digest,
        runtime_artifact=context.runtime_artifact,
    )
    return MappingProxyType(
        {
            "valid": True,
            "capture": str(capture),
            "capture_plan_identity": plan["capture_plan_identity"],
            "expected_rows": 10_000,
        }
    )


def run_e10_freeze_capture(
    directory: str | Path,
    *,
    e8_runbook: str | Path,
    e9_runbook: str | Path,
    execution_private_key: str,
    limit: int | None = None,
    shard_rows: int = 32,
) -> Mapping[str, Any]:
    """Resume the native VLLM early-token capture from the frozen plan."""

    root = validate_active_study_artifact_paths({"E10 freeze suite": directory})[
        "E10 freeze suite"
    ]
    context = _context(e8_runbook, e9_runbook)
    attestor, runtime_artifact = _native_runtime(
        context.e9,
        execution_private_key=execution_private_key,
        packaged_grader=context.e9.input_artifacts["frozen_graders"],
    )
    try:
        if sha256_path(runtime_artifact) != sha256_path(context.runtime_artifact):
            raise FrozenArtifactError("E10 early capture runtime differs from E6")
        return run_e10_early_probe_capture(
            root / "early-probe-capture",
            study=context.study,
            e1_run=context.prerequisite_paths["E1"],
            e8_run=context.prerequisite_paths["E8"],
            questions_by_partition=context.early_questions,
            prompt=context.prompt,
            controller_artifact=context.controller,
            selection_provenance=context.provenance,
            split_manifest_digest=context.split_manifest_digest,
            runtime_artifact=context.runtime_artifact,
            attestor=attestor,
            shard_rows=shard_rows,
            limit=limit,
        )
    finally:
        attestor.runtime.close()


def verify_e10_freeze_capture(
    directory: str | Path, *, require_complete: bool = False
) -> Mapping[str, Any]:
    capture = verify_e10_early_probe_capture(
        Path(directory) / "early-probe-capture",
        require_complete=require_complete,
    )
    return MappingProxyType(
        {
            "valid": True,
            "capture": str(capture.directory),
            "captured_rows": len(capture.rows),
            "expected_rows": 10_000,
            "complete": len(capture.rows) == 10_000,
            "capture_plan_identity": capture.plan["capture_plan_identity"],
        }
    )


def _sole_source_artifact(directory: Path) -> Path:
    values = tuple(
        value
        for value in directory.rglob("*")
        if value.is_file() and value.name != "manifest.json"
    )
    if len(values) != 1:
        raise FrozenArtifactError(f"packaged benchmark source is not unique: {directory}")
    return values[0]


def _final_questions(context: _E10Context) -> Mapping[str, tuple[Question, ...]]:
    e9_questions = context.e9.input_artifacts["frozen_question_bundle"]
    e8_questions = context.e8.outputs["side_effect_bundle"] / "questions"
    values = {
        benchmark: tuple(read_questions(e9_questions / filename))
        for benchmark, filename in _FACTUAL_FILES.items()
    }
    values.update(
        {
            benchmark: tuple(read_questions(e8_questions / f"{benchmark}.jsonl"))
            for benchmark in _SIDE_BENCHMARKS
        }
    )
    return MappingProxyType(values)


def _final_sources(context: _E10Context) -> Mapping[str, Path]:
    e9_sources = context.e9.input_artifacts["frozen_question_bundle"] / "source-artifacts"
    return MappingProxyType(
        {
            **{
                benchmark: _sole_source_artifact(e9_sources / benchmark)
                for benchmark in _FACTUAL_FILES
            },
            **{
                benchmark: context.e8.source_artifacts[benchmark]
                for benchmark in _SIDE_BENCHMARKS
                if benchmark != "language_consistency"
            },
            "language_consistency": context.e8.reviewed_language_suite,
        }
    )


def finalize_e10_freeze_suite(
    directory: str | Path,
    *,
    e8_runbook: str | Path,
    e9_runbook: str | Path,
    evaluation_scripts: str | Path,
    e10_runbook_output: str | Path,
) -> Mapping[str, Any]:
    """Fit the early probe and publish all eleven freezes plus an E10 runbook."""

    normalized = validate_active_study_artifact_paths(
        {
            "E10 freeze suite": directory,
            "E10 evaluation scripts": evaluation_scripts,
            "E10 runbook": e10_runbook_output,
        }
    )
    root = normalized["E10 freeze suite"]
    runbook_path = normalized["E10 runbook"]
    final_root = root / "final"
    if final_root.exists() or final_root.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite E10 final freezes: {final_root}")
    if runbook_path.exists() or runbook_path.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite E10 runbook: {runbook_path}")
    context = _context(e8_runbook, e9_runbook)
    verify_e10_freeze_capture(root, require_complete=True)
    evaluation = normalized["E10 evaluation scripts"]
    validate_execution_snapshot(
        evaluation,
        study_protocol_digest=context.study.digest,
        phase=ExperimentPhase.E10,
    )
    stage = Path(
        tempfile.mkdtemp(prefix=f".{root.name}.final.stage-", dir=root.parent)
    )
    published = False
    runbook_published = False
    try:
        selection = fit_e10_early_probe_selection(
            stage / "early-probe-selection",
            capture_directory=root / "early-probe-capture",
        )
        composite = write_e10_composite_from_promotions(
            stage / "composite-M6",
            study=context.study,
            prerequisite_runs=context.prerequisite_paths,
            controller_artifact=context.controller,
            early_probe_selection=selection.directory,
            sae_checkpoint=context.e8.e7_finalization / "sae-intervention",
            protected_source_artifact=context.e8.outputs["protected_artifact"],
        )
        freeze = write_e10_freeze_inputs(
            stage / "freeze-fields",
            model=context.model,
            prompt=context.prompt,
            composite_policy_artifact=composite,
            sae_checkpoint=context.e8.e7_finalization / "sae-intervention",
            grader=context.e9.input_artifacts["frozen_graders"],
            runtime_artifact=context.runtime_artifact,
            evaluation_scripts=evaluation,
            study_protocol_digest=context.study.digest,
        )
        questions = _final_questions(context)
        intervention = e10_intervention(
            component_artifact=composite,
            study=context.study,
            e6_run=context.prerequisite_paths["E6"],
        )
        phase = context.study.phase(ExperimentPhase.E10)
        provisional = build_e10_contract(
            study=context.study,
            model=context.model,
            prompt=context.prompt,
            questions_by_benchmark=questions,
            intervention=intervention,
            input_fingerprints={
                name: "0" * 64
                for name in (*phase.required_inputs, *phase.freeze_fields)
            },
            prerequisite_digests=context.prerequisite_digests,
            seed=context.e9.seed,
        )
        question_bundle = stage / "questions"
        write_frozen_question_bundle(
            question_bundle,
            provisional,
            questions,
            source_artifacts=_final_sources(context),
        )
        component_selection = stage / "component-selection"
        write_frozen_component_selection(
            component_selection,
            provisional,
            {(context.model.name, "M6"): composite},
        )
        stage_inputs = {
            "E9_results": context.e9.run_directory,
            "component_selection_manifest": component_selection,
            "frozen_question_bundle": question_bundle,
            **dict(freeze.paths),
        }
        final_contract = build_e10_contract(
            study=context.study,
            model=context.model,
            prompt=context.prompt,
            questions_by_benchmark=questions,
            intervention=intervention,
            input_fingerprints={
                name: sha256_path(path) for name, path in stage_inputs.items()
            },
            prerequisite_digests=context.prerequisite_digests,
            seed=context.e9.seed,
        )
        os.replace(stage, final_root)
        published = True
        inputs = {
            name: (
                path
                if not path.is_relative_to(stage)
                else final_root / path.relative_to(stage)
            )
            for name, path in stage_inputs.items()
        }
        body = {
            "schema_version": 1,
            "phase": "E10",
            "study_protocol": str(context.e9.study_protocol),
            "model_config": str(context.e9.model_config),
            "prompt_config": str(context.e9.prompt_config),
            "snapshot_directory": str(context.e9.snapshot_directory),
            "snapshot_manifest": str(context.e9.snapshot_manifest),
            "run_directory": str(context.e9.run_directory.parent / "E10"),
            "evidence_directory": str(context.e9.evidence_directory.parent / "E10"),
            "input_artifacts": {name: str(path) for name, path in inputs.items()},
            "prerequisite_runs": {
                name: str(path) for name, path in context.prerequisite_paths.items()
            },
            "seed": context.e9.seed,
        }
        _write_once_json(runbook_path, body)
        runbook_published = True
        preflight = preflight_confirmatory_runbook(
            ConfirmatoryRunbook.load(runbook_path)
        )
        return MappingProxyType(
            {
                "valid": True,
                "directory": str(root),
                "final_directory": str(final_root),
                "runbook": str(runbook_path),
                "runbook_sha256": sha256_file(runbook_path),
                "contract_digest": final_contract.digest,
                "expected_records": final_contract.expected_record_count,
                "selected_early_probe_sha256": sha256_path(
                    final_root / "early-probe-selection"
                ),
                "composite_sha256": sha256_path(final_root / "composite-M6"),
                "question_bundle_sha256": sha256_path(final_root / "questions"),
                "component_selection_sha256": sha256_path(
                    final_root / "component-selection"
                ),
                "preflight": dict(preflight),
            }
        )
    except BaseException:
        if runbook_published:
            runbook_path.unlink(missing_ok=True)
        if published:
            shutil.rmtree(final_root, ignore_errors=True)
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)

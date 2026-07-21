"""One-shot construction of every post-E8 input required by E9.

The lower-level writers intentionally accept typed scientific objects.  This
module is the operator boundary that derives those objects from completed E0--E8
ledgers and the terminal E8 runbook, then publishes a ready-to-preflight E9
runbook without asking the operator to write integration code.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from mfh.config import load_model_spec, load_prompt_specs
from mfh.contracts import ActivationSite, GenerationRecord, InterventionSpec, TokenScope
from mfh.data.io import read_questions
from mfh.data.reviewed_splits import validate_reviewed_split_snapshot
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.experiments.confirmatory_components import (
    write_confirmatory_adaptive_component,
    write_confirmatory_fixed_component,
)
from mfh.experiments.confirmatory_graders import write_confirmatory_grader_bundle
from mfh.experiments.confirmatory_operator import (
    ConfirmatoryRunbook,
    preflight_confirmatory_runbook,
)
from mfh.experiments.e4_baselines import load_e4_method_policy
from mfh.experiments.e8_operator import (
    E8Runbook,
    _base_context,
    _e6_components,
    _feature_schema,
    _selected_points,
)
from mfh.experiments.e8_protected import (
    load_e8_candidate_screen,
    load_e8_protected_artifact,
)
from mfh.experiments.e9_factorial import build_e9_contract
from mfh.experiments.grader_bundle import verify_e1_grader_bundle
from mfh.experiments.model_selection import (
    validate_active_model_spec,
    validate_active_study_artifact_paths,
)
from mfh.experiments.protocol import ExperimentPhase
from mfh.experiments.robustness_diagnostics import freeze_robustness_diagnostic_plan
from mfh.experiments.runner import (
    _validate_component_selection,
    _validate_e9_component_promotions,
    open_phase_prerequisite,
    write_frozen_component_selection,
    write_frozen_question_bundle,
)
from mfh.experiments.snapshots import validate_execution_snapshot
from mfh.provenance import sha256_file, sha256_path

_PROMPTS = ("P0-neutral", "P1-direct", "P2-calibrated-abstention")
_BENCHMARK_FILES = {
    "triviaqa": "T-test.jsonl",
    "simpleqa_verified": "simpleqa-eval.jsonl",
    "aa_omniscience_public_600": "aa-eval.jsonl",
}


def stage_e9_external_inputs(
    directory: str | Path,
    *,
    official_grader_bundle: str | Path,
    expected_official_grader_manifest_digest: str,
    reviewed_splits: str | Path,
    source_artifacts: Mapping[str, str | Path],
) -> Mapping[str, Any]:
    """Atomically copy verified external inputs into the active Qwen namespace."""

    if set(source_artifacts) != set(_BENCHMARK_FILES):
        raise DataValidationError("E9 staged source-artifact inventory differs")
    destination = validate_active_study_artifact_paths({"E9 staged inputs": directory})[
        "E9 staged inputs"
    ]
    if destination.exists() or destination.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite E9 staged inputs: {destination}")
    raw_grader = Path(official_grader_bundle).absolute()
    raw_reviewed = Path(reviewed_splits).absolute()
    raw_sources = {name: Path(value).absolute() for name, value in source_artifacts.items()}
    if (
        raw_grader.is_symlink()
        or raw_reviewed.is_symlink()
        or any(path.is_symlink() for path in raw_sources.values())
    ):
        raise FrozenArtifactError("E9 staged inputs must not be symlinks")
    grader = raw_grader.resolve()
    reviewed = raw_reviewed.resolve()
    verify_e1_grader_bundle(
        grader,
        expected_manifest_digest=expected_official_grader_manifest_digest,
    )
    validate_reviewed_split_snapshot(reviewed)
    sources = {name: value.resolve() for name, value in raw_sources.items()}
    if (
        not grader.is_dir()
        or not reviewed.is_dir()
        or any(not path.is_file() for path in sources.values())
    ):
        raise FrozenArtifactError("E9 staged inputs must be strict regular artifacts")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        shutil.copytree(grader, stage / "official-graders")
        shutil.copytree(reviewed, stage / "reviewed-splits")
        source_root = stage / "sources"
        source_root.mkdir()
        staged_sources: dict[str, str] = {}
        for name, source in sources.items():
            target = source_root / f"{name}{source.suffix}"
            shutil.copyfile(source, target)
            staged_sources[name] = str(target.relative_to(stage))
        body = {
            "schema_version": 1,
            "official_grader_manifest_digest": expected_official_grader_manifest_digest,
            "official_graders_sha256": sha256_path(stage / "official-graders"),
            "reviewed_splits_sha256": sha256_path(stage / "reviewed-splits"),
            "sources": {
                name: {
                    "path": path,
                    "sha256": sha256_file(stage / path),
                }
                for name, path in sorted(staged_sources.items())
            },
        }
        (stage / "manifest.json").write_text(
            json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        verify_e1_grader_bundle(
            stage / "official-graders",
            expected_manifest_digest=expected_official_grader_manifest_digest,
        )
        validate_reviewed_split_snapshot(stage / "reviewed-splits")
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
    return MappingProxyType(
        {
            "valid": True,
            "directory": str(destination),
            "official_grader_bundle": str(destination / "official-graders"),
            "reviewed_splits": str(destination / "reviewed-splits"),
            "source_artifacts": {
                name: str(destination / path) for name, path in sorted(staged_sources.items())
            },
            "sha256": sha256_path(destination),
        }
    )


def _write_once_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(value), indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise FrozenArtifactError(f"refusing to overwrite E9 runbook: {path}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _completion_material(
    run_root: Path,
    *,
    e3_phase_run: Path,
    study: Any,
) -> tuple[Mapping[str, Path], Mapping[str, str], Mapping[ExperimentPhase, Any]]:
    paths: dict[str, Path] = {}
    digests: dict[str, str] = {}
    ledgers: dict[ExperimentPhase, Any] = {}
    for phase in tuple(ExperimentPhase)[:9]:
        path = e3_phase_run if phase is ExperimentPhase.E3 else run_root / phase.value
        ledger = open_phase_prerequisite(path, phase=phase, study=study)
        completion = ledger.verify_complete()
        paths[phase.value] = path
        digests[phase.value] = completion.completion_digest
        ledgers[phase] = ledger
    return (
        MappingProxyType(paths),
        MappingProxyType(digests),
        MappingProxyType(ledgers),
    )


def _fixed_geometry(record: GenerationRecord) -> tuple[int, ActivationSite, TokenScope]:
    if record.layer is None or record.site is None or record.token_scope is None:
        raise FrozenArtifactError("selected E8 fixed point lacks execution geometry")
    return record.layer, record.site, record.token_scope


def _e9_runbook_body(
    *,
    destination: Path,
    e8: E8Runbook,
    run_root: Path,
    evidence_root: Path,
    inputs: Mapping[str, Path],
    prerequisites: Mapping[str, Path],
) -> Mapping[str, Any]:
    del destination
    return MappingProxyType(
        {
            "schema_version": 1,
            "phase": "E9",
            "study_protocol": str(e8.study_protocol),
            "model_config": str(e8.model_config),
            "prompt_config": str(e8.prompt_config),
            "snapshot_directory": str(e8.snapshot_directory),
            "snapshot_manifest": str(e8.snapshot_manifest),
            "run_directory": str(run_root / "E9"),
            "evidence_directory": str(evidence_root / "E9"),
            "input_artifacts": {name: str(path) for name, path in inputs.items()},
            "prerequisite_runs": {name: str(path) for name, path in prerequisites.items()},
            "seed": e8.seed,
        }
    )


def freeze_e9_input_suite(
    directory: str | Path,
    *,
    e8_runbook: str | Path,
    e9_runbook_output: str | Path,
    evaluation_scripts: str | Path,
    official_grader_bundle: str | Path,
    expected_official_grader_manifest_digest: str,
    reviewed_splits: str | Path,
    source_artifacts: Mapping[str, str | Path],
    m2_source_artifact: str | Path,
    e3_phase_run: str | Path,
    execution_private_key: str,
    robustness_config: str | Path = "configs/experiments/robustness-diagnostics.json",
) -> Mapping[str, Any]:
    """Publish E9 components, graders, questions, robustness plan, and runbook."""

    if set(source_artifacts) != set(_BENCHMARK_FILES):
        raise DataValidationError("E9 source-artifact inventory differs")
    normalized = validate_active_study_artifact_paths(
        {
            "E9 freeze suite": directory,
            "E9 runbook": e9_runbook_output,
            "E9 evaluation scripts": evaluation_scripts,
            "E9 official grader bundle": official_grader_bundle,
            "E9 reviewed splits": reviewed_splits,
            "E9 M2 source": m2_source_artifact,
            "E9 E3 phase": e3_phase_run,
            **{f"E9 source {benchmark}": path for benchmark, path in source_artifacts.items()},
        }
    )
    root = normalized["E9 freeze suite"]
    runbook_path = normalized["E9 runbook"]
    if root.exists() or root.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite E9 freeze suite: {root}")
    if runbook_path.exists() or runbook_path.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite E9 runbook: {runbook_path}")
    e8 = E8Runbook.load(e8_runbook)
    context = _base_context(e8)
    e8_ledger = open_phase_prerequisite(
        e8.outputs["run_directory"],
        phase=ExperimentPhase.E8,
        study=context.study,
    )
    e8_ledger.verify_complete()
    model = load_model_spec(e8.model_config)
    validate_active_model_spec(model)
    all_prompts = {value.prompt_id: value for value in load_prompt_specs(e8.prompt_config)}
    try:
        prompts = {name: all_prompts[name] for name in _PROMPTS}
    except KeyError as exc:
        raise DataValidationError(f"E9 prompt is missing: {exc.args[0]}") from exc
    run_root = e8.outputs["run_directory"].parent
    study_root = run_root.parent
    evidence_root = study_root / "evidence"
    prerequisite_paths, prerequisite_digests, prerequisite_ledgers = _completion_material(
        run_root,
        e3_phase_run=normalized["E9 E3 phase"],
        study=context.study,
    )

    protected = load_e8_protected_artifact(e8.outputs["protected_artifact"])
    components = _e6_components(e8, context, _feature_schema(e8, context))
    selected = _selected_points(load_e8_candidate_screen(e8.outputs["candidate_screen"]))
    p0_points = {method: selected[("P0-neutral", method)] for method in ("M1", "M3", "M4", "M5")}
    fixed_records = {method: p0_points[method].records[0] for method in ("M1", "M4", "M5")}
    fixed_geometry = {method: _fixed_geometry(record) for method, record in fixed_records.items()}

    root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name}.stage-", dir=root.parent))
    published = False
    runbook_published = False
    try:
        component_root = stage / "components"
        fixed: dict[str, Any] = {}
        fixed["M1"] = write_confirmatory_fixed_component(
            component_root / "M1",
            source_artifact=e8.e6_transition_evidence / "e3-static-vectors",
            method="M1",
            layer=fixed_geometry["M1"][0],
            site=fixed_geometry["M1"][1],
            token_scope=fixed_geometry["M1"][2],
            standardized_alpha=p0_points["M1"].alpha,
            reference_rms=components.reference_rms,
        )
        e4 = prerequisite_ledgers[ExperimentPhase.E4]
        m2_policy = load_e4_method_policy(
            e4.directory / "gate-artifacts" / "promotion_decision_frozen" / "policy-m2"
        )
        if (
            m2_policy.layer is None
            or m2_policy.site is None
            or m2_policy.token_scope is None
            or m2_policy.reference_rms is None
        ):
            raise FrozenArtifactError("promoted E4 M2 policy lacks fixed geometry")
        fixed["M2"] = write_confirmatory_fixed_component(
            component_root / "M2",
            source_artifact=normalized["E9 M2 source"],
            method="M2",
            layer=m2_policy.layer,
            site=m2_policy.site,
            token_scope=m2_policy.token_scope,
            standardized_alpha=m2_policy.alpha,
            reference_rms=float(m2_policy.reference_rms),
        )
        fixed["M4"] = write_confirmatory_fixed_component(
            component_root / "M4",
            source_artifact=components.sae_path,
            method="M4",
            layer=fixed_geometry["M4"][0],
            site=fixed_geometry["M4"][1],
            token_scope=fixed_geometry["M4"][2],
            standardized_alpha=p0_points["M4"].alpha,
            reference_rms=protected.reference_rms,
            sparsity=fixed_records["M4"].sparsity,
        )
        fixed["M5"] = write_confirmatory_fixed_component(
            component_root / "M5",
            source_artifact=e8.outputs["protected_artifact"],
            method="M5",
            layer=fixed_geometry["M5"][0],
            site=fixed_geometry["M5"][1],
            token_scope=fixed_geometry["M5"][2],
            standardized_alpha=p0_points["M5"].alpha,
            reference_rms=protected.reference_rms,
        )
        adaptive = write_confirmatory_adaptive_component(
            component_root / "M3",
            model=model,
            prompts=prompts,
            controllers={name: components.controller_path for name in prompts},
            controller_source_prompts={name: "P0-neutral" for name in prompts},
        )
        source_policy = p0_points["M3"].adaptive_policy
        if source_policy is None:
            raise FrozenArtifactError("selected E8 M3 point lacks its adaptive policy")
        adaptive_policy = replace(
            source_policy,
            controller_artifact_sha256=adaptive.fingerprint,
        )
        interventions = {
            "M0": InterventionSpec(method="M0"),
            **{
                method: InterventionSpec(
                    method=method,
                    layer=value.layer,
                    site=value.site,
                    token_scope=value.token_scope,
                    alpha=value.standardized_alpha,
                    sparsity=value.sparsity,
                    artifact_sha256=value.fingerprint,
                    decay=value.decay,
                )
                for method, value in fixed.items()
            },
            "M3": InterventionSpec(
                method="M3",
                artifact_sha256=adaptive.fingerprint,
                adaptive_policy=adaptive_policy,
            ),
        }
        questions = {
            benchmark: tuple(read_questions(normalized["E9 reviewed splits"] / filename))
            for benchmark, filename in _BENCHMARK_FILES.items()
        }
        provisional = build_e9_contract(
            study=context.study,
            model=model,
            prompts=prompts,
            questions_by_benchmark=questions,
            interventions=interventions,
            input_fingerprints={
                name: "0" * 64 for name in context.study.phase(ExperimentPhase.E9).required_inputs
            },
            prerequisite_digests=prerequisite_digests,
            seed=e8.seed,
        )
        selection = stage / "component-selection"
        write_frozen_component_selection(
            selection,
            provisional,
            {
                (model.name, method): value.directory
                for method, value in {**fixed, "M3": adaptive}.items()
            },
        )
        question_bundle = stage / "questions"
        write_frozen_question_bundle(
            question_bundle,
            provisional,
            questions,
            source_artifacts={
                benchmark: normalized[f"E9 source {benchmark}"] for benchmark in source_artifacts
            },
        )
        evaluation = normalized["E9 evaluation scripts"]
        validate_execution_snapshot(
            evaluation,
            study_protocol_digest=context.study.digest,
            phase=ExperimentPhase.E9,
        )
        side_effect_bundle = e8.outputs["side_effect_bundle"]
        graders = stage / "graders"
        write_confirmatory_grader_bundle(
            graders,
            official_grader_bundle=normalized["E9 official grader bundle"],
            expected_official_manifest_digest=expected_official_grader_manifest_digest,
            side_effect_scorer=side_effect_bundle / "side-effect-scorer.json",
            ifeval_evaluator=side_effect_bundle / "ifeval-evaluator",
            strongreject_grader=side_effect_bundle / "strongreject-grader",
            runtime_attestation=e8.runtime_artifact,
        )
        robustness = stage / "robustness-plan"
        freeze_robustness_diagnostic_plan(
            robustness,
            config_path=robustness_config,
            source_artifacts={
                "canonical-prompts": e8.prompt_config,
                "e1-phase-ledger": prerequisite_paths["E1"],
                "frozen-component-selection": selection,
                "frozen-evaluation-scripts": evaluation,
                "frozen-graders": graders,
                "triviaqa-evaluation": normalized["E9 reviewed splits"],
                "simpleqa_verified-evaluation": normalized["E9 reviewed splits"],
                "aa_omniscience_public_600-evaluation": normalized["E9 reviewed splits"],
                "triviaqa-development": normalized["E9 reviewed splits"],
            },
            completion_execution_private_key=execution_private_key,
        )
        inputs = MappingProxyType(
            {
                "frozen_component_selection": selection,
                "frozen_graders": graders,
                "frozen_evaluation_scripts": evaluation,
                "frozen_question_bundle": question_bundle,
                "frozen_prompt_paraphrase_schedule": robustness,
            }
        )
        final_contract = build_e9_contract(
            study=context.study,
            model=model,
            prompts=prompts,
            questions_by_benchmark=questions,
            interventions=interventions,
            input_fingerprints={name: sha256_path(path) for name, path in inputs.items()},
            prerequisite_digests=prerequisite_digests,
            seed=e8.seed,
        )
        _validate_component_selection(selection, final_contract)
        _validate_e9_component_promotions(
            selection,
            final_contract,
            prerequisite_ledgers,
        )
        # E9 publication begins after the complete hidden stage validates.
        os.replace(stage, root)
        published = True
        final_inputs = MappingProxyType(
            {name: root / path.relative_to(stage) for name, path in inputs.items()}
        )
        body = _e9_runbook_body(
            destination=runbook_path,
            e8=e8,
            run_root=run_root,
            evidence_root=evidence_root,
            inputs=final_inputs,
            prerequisites=prerequisite_paths,
        )
        _write_once_json(runbook_path, body)
        runbook_published = True
        preflight = preflight_confirmatory_runbook(ConfirmatoryRunbook.load(runbook_path))
        return MappingProxyType(
            {
                "valid": True,
                "directory": str(root),
                "runbook": str(runbook_path),
                "runbook_sha256": sha256_file(runbook_path),
                "contract_digest": final_contract.digest,
                "expected_records": final_contract.expected_record_count,
                "component_selection_sha256": sha256_path(
                    final_inputs["frozen_component_selection"]
                ),
                "grader_sha256": sha256_path(final_inputs["frozen_graders"]),
                "question_bundle_sha256": sha256_path(final_inputs["frozen_question_bundle"]),
                "robustness_plan_sha256": sha256_path(
                    final_inputs["frozen_prompt_paraphrase_schedule"]
                ),
                "preflight": dict(preflight),
            }
        )
    except BaseException:
        if runbook_published:
            runbook_path.unlink(missing_ok=True)
        if published:
            shutil.rmtree(root, ignore_errors=True)
        raise
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)

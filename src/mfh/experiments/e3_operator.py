"""Secret-free, resumable operator lifecycle for the complete E3 program."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from mfh.artifact_namespace import (
    QWEN_STUDY_ARTIFACT_ROOT,
    validate_active_study_artifact_paths,
)
from mfh.config import load_model_spec, load_prompt_specs
from mfh.contracts import PromptSpec, Question
from mfh.data.io import read_questions
from mfh.data.reviewed_splits import validate_reviewed_split_snapshot
from mfh.errors import ConfigurationError, DataValidationError, FrozenArtifactError
from mfh.experiments.e3_construction import (
    e3_questions_digest,
    finalize_e3_vector_bundle,
    prepare_e3_construction_work,
    run_e3_construction,
    verify_e3_construction_work,
    verify_e3_vector_bundle,
)
from mfh.experiments.e3_control_materials import (
    verify_e3_fixed_control_materials,
    write_e3_fixed_control_materials,
)
from mfh.experiments.e3_controls import (
    finalize_e3_shuffled_control_bundle,
    prepare_e3_shuffled_control_work,
    run_e3_shuffled_control,
    verify_e3_shuffled_control_bundle,
    verify_e3_shuffled_control_work,
)
from mfh.experiments.e3_execution import E3ExecutionAssets, load_e3_execution_assets
from mfh.experiments.e3_phase import finalize_e3_phase, verify_e3_phase
from mfh.experiments.e3_runner import (
    e3_conditions_for_stage,
    e3_selection_inputs_from_work,
    prepare_e3_evaluation_work,
    run_e3_evaluation,
    verify_e3_evaluation_work,
)
from mfh.experiments.e3_schedule import (
    E3Protocol,
    e3_stage_row_counts,
    select_e3_screen_questions,
)
from mfh.experiments.e3_selection import (
    VerifiedE3StageSelection,
    load_verified_e3_stage_selection,
    write_e3_stage_selection,
)
from mfh.experiments.model_selection import validate_active_model_spec
from mfh.experiments.protocol import ExperimentPhase, StudyProtocol, load_study_protocol
from mfh.experiments.runner import PhaseRunLedger
from mfh.inference.vllm_research import VllmResearchRuntime
from mfh.provenance import sha256_file, stable_hash

_STAGES = (
    "geometry",
    "alpha",
    "scope",
    "controls",
    "cross-prompt",
    "P3-diagnostic",
    "final",
)
_SELECTION_STAGES = ("geometry", "alpha", "scope")
_CONSTRUCTION_FINGERPRINT_FIELDS = frozenset(
    {
        "reviewed_split_manifest_digest",
        "review_result_manifest_digest",
        "t_steer_question_ids_sha256",
        "t_steer_questions_digest",
    }
)
_RUNBOOK_KEYS = {
    "schema_version",
    "model_config",
    "snapshot_directory",
    "t_steer_questions",
    "t_dev_questions",
    "prompt_config",
    "study_protocol",
    "source_runtime_plan",
    "output_root",
    "input_artifacts",
    "prerequisite_runs",
    "hidden_width",
    "construction_checkpoint_rows",
    "shuffle_checkpoint_rows",
    "max_new_tokens",
    "construction_input_fingerprints",
}


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _operator_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise DataValidationError(f"E3 runbook {label} path is invalid")
    path = Path(value)
    if not path.is_absolute() or path != path.resolve(strict=False):
        raise DataValidationError(f"E3 runbook {label} must be a canonical absolute path")
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    if current.is_symlink():
        raise FrozenArtifactError(f"E3 runbook {label} traverses a symbolic link")
    return path


def _path_mapping(
    value: object,
    *,
    label: str,
    expected: set[str],
) -> Mapping[str, Path]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise DataValidationError(f"E3 runbook {label} names differ")
    return MappingProxyType(
        {str(name): _operator_path(path, f"{label}.{name}") for name, path in value.items()}
    )


@dataclass(frozen=True, slots=True)
class E3OperatorRunbook:
    path: Path
    model_config: Path
    snapshot_directory: Path
    t_steer_questions: Path
    t_dev_questions: Path
    prompt_config: Path
    study_protocol: Path
    source_runtime_plan: Path
    output_root: Path
    input_artifacts: Mapping[str, Path]
    prerequisite_runs: Mapping[str, Path]
    hidden_width: int
    construction_checkpoint_rows: int
    shuffle_checkpoint_rows: int
    max_new_tokens: int
    construction_input_fingerprints: Mapping[str, str]
    runbook_digest: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        numeric = (
            self.hidden_width,
            self.construction_checkpoint_rows,
            self.shuffle_checkpoint_rows,
            self.max_new_tokens,
        )
        if (
            self.schema_version != 1
            or any(type(value) is not int or value <= 0 for value in numeric)
            or self.hidden_width != 5_120
            or self.max_new_tokens > 48
            or set(self.construction_input_fingerprints)
            != _CONSTRUCTION_FINGERPRINT_FIELDS
            or any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in self.construction_input_fingerprints.values()
            )
        ):
            raise DataValidationError("E3 runbook scientific geometry differs")
        object.__setattr__(self, "input_artifacts", MappingProxyType(dict(self.input_artifacts)))
        object.__setattr__(
            self, "prerequisite_runs", MappingProxyType(dict(self.prerequisite_runs))
        )
        object.__setattr__(
            self,
            "construction_input_fingerprints",
            MappingProxyType(dict(self.construction_input_fingerprints)),
        )


@dataclass(frozen=True, slots=True)
class _E3Paths:
    root: Path

    @property
    def construction(self) -> Path:
        return self.root / "construction"

    @property
    def vectors(self) -> Path:
        return self.root / "vectors"

    @property
    def selections(self) -> Path:
        return self.root / "selections"

    def selection(self, stage: str) -> Path:
        return self.selections / f"{stage}.json"

    @property
    def stages(self) -> Path:
        return self.root / "stages"

    def stage(self, stage: str) -> Path:
        return self.stages / stage

    @property
    def shuffle_work(self) -> Path:
        return self.root / "controls" / "shuffle-work"

    @property
    def shuffled_vectors(self) -> Path:
        return self.root / "controls" / "shuffled-vectors"

    @property
    def fixed_controls(self) -> Path:
        return self.root / "controls" / "fixed"

    @property
    def phase(self) -> Path:
        return self.root / "phase"


@dataclass(frozen=True, slots=True)
class _E3Context:
    runbook: E3OperatorRunbook
    study: StudyProtocol
    t_steer: tuple[Question, ...]
    t_dev: tuple[Question, ...]
    screen: tuple[Question, ...]
    construction_prompts: Mapping[str, PromptSpec]
    application_prompts: Mapping[str, PromptSpec]
    source_runtime_identity: Mapping[str, Any]
    paths: _E3Paths


def load_e3_operator_runbook(path: str | Path) -> E3OperatorRunbook:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise FrozenArtifactError("E3 operator runbook must be a regular file")
    try:
        raw = json.loads(
            source.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read E3 operator runbook: {exc}") from exc
    if not isinstance(raw, dict) or set(raw) != _RUNBOOK_KEYS or raw.get("schema_version") != 1:
        raise DataValidationError("E3 operator runbook schema differs")
    body = dict(raw)
    digest = stable_hash(body)
    fingerprints = raw["construction_input_fingerprints"]
    if not isinstance(fingerprints, Mapping):
        raise DataValidationError("E3 construction fingerprints are invalid")
    normalized_fingerprints = {str(key): str(value) for key, value in fingerprints.items()}
    return E3OperatorRunbook(
        path=source.resolve(),
        model_config=_operator_path(raw["model_config"], "model_config"),
        snapshot_directory=_operator_path(raw["snapshot_directory"], "snapshot_directory"),
        t_steer_questions=_operator_path(raw["t_steer_questions"], "t_steer_questions"),
        t_dev_questions=_operator_path(raw["t_dev_questions"], "t_dev_questions"),
        prompt_config=_operator_path(raw["prompt_config"], "prompt_config"),
        study_protocol=_operator_path(raw["study_protocol"], "study_protocol"),
        source_runtime_plan=_operator_path(raw["source_runtime_plan"], "source_runtime_plan"),
        output_root=_operator_path(raw["output_root"], "output_root"),
        input_artifacts=_path_mapping(
            raw["input_artifacts"],
            label="input_artifacts",
            expected={"E1_outcome_labels", "activation_feature_schemas", "reviewed_splits"},
        ),
        prerequisite_runs=_path_mapping(
            raw["prerequisite_runs"],
            label="prerequisite_runs",
            expected={"E1", "E2"},
        ),
        hidden_width=raw["hidden_width"],
        construction_checkpoint_rows=raw["construction_checkpoint_rows"],
        shuffle_checkpoint_rows=raw["shuffle_checkpoint_rows"],
        max_new_tokens=raw["max_new_tokens"],
        construction_input_fingerprints=normalized_fingerprints,
        runbook_digest=digest,
    )


def write_e3_operator_runbook_template(
    destination: str | Path, *, reviewed_splits: str | Path
) -> Path:
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite E3 runbook: {output}")
    workspace = Path.cwd().resolve()
    study_root = workspace / QWEN_STUDY_ARTIFACT_ROOT
    reviewed_root = validate_active_study_artifact_paths(
        {"E3 reviewed splits": reviewed_splits}
    )["E3 reviewed splits"]
    reviewed_manifest = validate_reviewed_split_snapshot(reviewed_root)
    t_steer = tuple(read_questions(reviewed_root / "T-steer.jsonl"))
    split_ids = reviewed_manifest.get("split_question_ids_sha256")
    if not isinstance(split_ids, Mapping) or not isinstance(
        split_ids.get("T-steer"), str
    ):
        raise DataValidationError("reviewed split lacks the T-steer identity")
    construction_fingerprints = {
        "reviewed_split_manifest_digest": str(reviewed_manifest["manifest_digest"]),
        "review_result_manifest_digest": str(
            reviewed_manifest["review_result_manifest_digest"]
        ),
        "t_steer_question_ids_sha256": str(split_ids["T-steer"]),
        "t_steer_questions_digest": e3_questions_digest(t_steer),
    }
    body = {
        "schema_version": 1,
        "model_config": str(workspace / "configs/models/qwen3.6-27b-nvfp4.yaml"),
        "snapshot_directory": str(workspace / "operator-inputs/model-snapshot"),
        "t_steer_questions": str(reviewed_root / "T-steer.jsonl"),
        "t_dev_questions": str(reviewed_root / "T-dev.jsonl"),
        "prompt_config": str(workspace / "configs/prompts/primary.yaml"),
        "study_protocol": str(workspace / "configs/experiments/phases.yaml"),
        "source_runtime_plan": str(workspace / "operator-inputs/E2-capture-plan.json"),
        "output_root": str(study_root / "E3-operator"),
        "input_artifacts": {
            "E1_outcome_labels": str(study_root / "runs/E1"),
            "activation_feature_schemas": str(
                study_root / "E2/activation-feature-schemas"
            ),
            "reviewed_splits": str(reviewed_root),
        },
        "prerequisite_runs": {
            "E1": str(study_root / "runs/E1"),
            "E2": str(study_root / "runs/E2"),
        },
        "hidden_width": 5_120,
        "construction_checkpoint_rows": 64,
        "shuffle_checkpoint_rows": 64,
        "max_new_tokens": 48,
        "construction_input_fingerprints": construction_fingerprints,
    }
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(body, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return output


def _runtime_identity(path: Path) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FrozenArtifactError("E3 source runtime plan must be a regular file")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
        identity = value["runtime_identity"]
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot load E3 source runtime identity: {exc}") from exc
    if not isinstance(identity, dict) or not isinstance(identity.get("research_provenance"), dict):
        raise DataValidationError("E3 source runtime identity lacks research provenance")
    return MappingProxyType(identity)


def _operator_context(runbook: E3OperatorRunbook) -> _E3Context:
    for label, path in {
        "model config": runbook.model_config,
        "snapshot": runbook.snapshot_directory,
        "T-steer": runbook.t_steer_questions,
        "T-dev": runbook.t_dev_questions,
        "prompt config": runbook.prompt_config,
        "study protocol": runbook.study_protocol,
        "source runtime plan": runbook.source_runtime_plan,
        **{f"input {name}": path for name, path in runbook.input_artifacts.items()},
        **{f"prerequisite {name}": path for name, path in runbook.prerequisite_runs.items()},
    }.items():
        if path.is_symlink() or not path.exists():
            raise FrozenArtifactError(f"E3 operator {label} is missing or linked")
    model = load_model_spec(runbook.model_config)
    validate_active_model_spec(model)
    study = load_study_protocol(runbook.study_protocol)
    t_steer = tuple(read_questions(runbook.t_steer_questions))
    t_dev = tuple(read_questions(runbook.t_dev_questions))
    protocol = E3Protocol()
    if len(t_steer) != protocol.steer_rows or len(t_dev) != protocol.dev_rows:
        raise DataValidationError("E3 operator question counts differ from the frozen protocol")
    all_prompts = {value.prompt_id: value for value in load_prompt_specs(runbook.prompt_config)}
    construction = {name: all_prompts[name] for name in ("P0-neutral", "P2-calibrated-abstention")}
    application = {
        name: all_prompts[name]
        for name in ("P0-neutral", "P2-calibrated-abstention", "P3-forced-answer")
    }
    screen_ids = set(select_e3_screen_questions(t_dev, protocol=protocol))
    screen = tuple(value for value in t_dev if value.question_id in screen_ids)
    if len(screen) != protocol.screen_rows:
        raise DataValidationError("E3 operator screen materialization differs")
    return _E3Context(
        runbook=runbook,
        study=study,
        t_steer=t_steer,
        t_dev=t_dev,
        screen=screen,
        construction_prompts=MappingProxyType(construction),
        application_prompts=MappingProxyType(application),
        source_runtime_identity=_runtime_identity(runbook.source_runtime_plan),
        paths=_E3Paths(runbook.output_root),
    )


def preflight_e3_operator(runbook: E3OperatorRunbook) -> Mapping[str, Any]:
    validate_active_study_artifact_paths(
        {
            "E3 output root": runbook.output_root,
            **{
                f"E3 input {name}": path
                for name, path in runbook.input_artifacts.items()
            },
            **{
                f"E3 prerequisite {name}": path
                for name, path in runbook.prerequisite_runs.items()
            },
        }
    )
    context = _operator_context(runbook)
    prerequisites: dict[str, str] = {}
    ledgers: dict[str, PhaseRunLedger] = {}
    for name in ("E1", "E2"):
        phase = ExperimentPhase(name)
        ledger = PhaseRunLedger.open(runbook.prerequisite_runs[name], study=context.study)
        completion = ledger.verify_complete()
        if completion.phase is not phase:
            raise FrozenArtifactError(f"E3 {name} prerequisite is a different phase")
        prerequisites[name] = completion.completion_digest
        ledgers[name] = ledger
    if (
        ledgers["E2"].contract.prerequisite_digests.get(ExperimentPhase.E1.value)
        != prerequisites["E1"]
    ):
        raise FrozenArtifactError("E3 E1 and E2 prerequisites are from different lineages")
    rows = dict(e3_stage_row_counts())
    return MappingProxyType(
        {
            "valid": True,
            "runbook_digest": runbook.runbook_digest,
            "study_protocol_digest": context.study.digest,
            "source_runtime_identity_sha256": stable_hash(context.source_runtime_identity),
            "construction_rows": E3Protocol().construction_rows,
            "stage_rows": rows,
            "evaluation_rows": sum(rows.values()),
            "prerequisite_completion_digests": prerequisites,
            "output_root": str(runbook.output_root),
            "phase_complete": context.paths.phase.is_dir(),
        }
    )


def _live_runtime(context: _E3Context) -> VllmResearchRuntime:
    model = load_model_spec(context.runbook.model_config)
    provenance = context.source_runtime_identity["research_provenance"]
    assert isinstance(provenance, Mapping)
    runtime = VllmResearchRuntime.from_spec(
        model,
        snapshot_path=context.runbook.snapshot_directory,
        seed=17,
        research_provenance=provenance,
    )
    if dict(runtime.runtime_identity()) != dict(context.source_runtime_identity):
        runtime.close()
        raise FrozenArtifactError("E3 live runtime differs from the frozen source identity")
    return runtime


def _predecessor(
    stage: str,
    selections: Mapping[str, VerifiedE3StageSelection],
) -> VerifiedE3StageSelection | None:
    if stage == "geometry":
        return None
    return selections["geometry" if stage == "alpha" else "alpha" if stage == "scope" else "scope"]


def _stage_questions(context: _E3Context, stage: str) -> tuple[Question, ...]:
    return context.t_dev if stage == "final" else context.screen


def _stage_assets(
    context: _E3Context,
    runtime: VllmResearchRuntime,
    *,
    stage: str,
    predecessor: VerifiedE3StageSelection | None,
) -> E3ExecutionAssets:
    controls = stage in {"controls", "final"}
    return load_e3_execution_assets(
        construction_directory=context.paths.construction,
        vector_bundle_directory=context.paths.vectors,
        questions=context.t_steer,
        prompts=context.construction_prompts,
        scope_selection=predecessor,
        shuffled_work_directory=(context.paths.shuffle_work if controls else None),
        shuffled_bundle_directory=(context.paths.shuffled_vectors if controls else None),
        fixed_control_directory=(context.paths.fixed_controls if controls else None),
        dev_questions=context.t_dev,
        evaluation_questions=_stage_questions(context, stage),
        application_prompts=context.application_prompts,
        conditions=e3_conditions_for_stage(stage, selection_receipt=predecessor),
        stage=stage,
        render_runtime=runtime,
    )


def _load_selections(
    context: _E3Context,
    runtime: VllmResearchRuntime,
) -> tuple[dict[str, VerifiedE3StageSelection], dict[str, E3ExecutionAssets]]:
    selections: dict[str, VerifiedE3StageSelection] = {}
    assets: dict[str, E3ExecutionAssets] = {}
    for stage in _SELECTION_STAGES:
        path = context.paths.selection(stage)
        if not path.exists():
            break
        predecessor = _predecessor(stage, selections)
        stage_assets = _stage_assets(context, runtime, stage=stage, predecessor=predecessor)
        inputs = e3_selection_inputs_from_work(
            context.paths.stage(stage),
            stage=stage,
            assets=stage_assets,
            evaluation_questions=_stage_questions(context, stage),
            selection_receipt=predecessor,
        )
        receipt = load_verified_e3_stage_selection(path, **inputs)
        selections[stage] = receipt
        assets[stage] = stage_assets
    return selections, assets


def _action(name: str, **values: Any) -> Mapping[str, Any]:
    return MappingProxyType({"valid": True, "action": name, **values})


def advance_e3_operator(
    runbook: E3OperatorRunbook,
    *,
    request_budget: int = 4_096,
) -> Mapping[str, Any]:
    """Perform exactly one durable lifecycle action and return its progress."""

    if type(request_budget) is not int or request_budget <= 0:
        raise ConfigurationError("E3 operator request budget must be positive")
    preflight_e3_operator(runbook)
    context = _operator_context(runbook)
    context.paths.root.mkdir(parents=True, exist_ok=True)
    runtime = _live_runtime(context)
    try:
        if not context.paths.construction.exists():
            plan = prepare_e3_construction_work(
                context.paths.construction,
                questions=context.t_steer,
                prompts=context.construction_prompts,
                runtime_identity=context.source_runtime_identity,
                hidden_width=runbook.hidden_width,
                checkpoint_rows=runbook.construction_checkpoint_rows,
                max_new_tokens=runbook.max_new_tokens,
                input_fingerprints=runbook.construction_input_fingerprints,
            )
            return _action("prepared-construction", plan_identity=plan["plan_identity"])
        construction = verify_e3_construction_work(
            context.paths.construction,
            questions=context.t_steer,
            prompts=context.construction_prompts,
        )
        if not construction["complete"]:
            progress = run_e3_construction(
                context.paths.construction,
                questions=context.t_steer,
                prompts=context.construction_prompts,
                runtime=runtime,
                request_budget=request_budget,
            )
            return _action("ran-construction", **dict(progress))
        if not context.paths.vectors.exists():
            result = finalize_e3_vector_bundle(
                context.paths.vectors,
                work_directory=context.paths.construction,
                questions=context.t_steer,
                prompts=context.construction_prompts,
            )
            return _action("finalized-vectors", **dict(result))
        verify_e3_vector_bundle(
            context.paths.vectors,
            work_directory=context.paths.construction,
            questions=context.t_steer,
            prompts=context.construction_prompts,
        )
        selections, known_assets = _load_selections(context, runtime)
        for stage in _STAGES:
            predecessor = _predecessor(stage, selections)
            if stage == "controls" and "scope" in selections:
                if not context.paths.shuffle_work.exists():
                    plan = prepare_e3_shuffled_control_work(
                        context.paths.shuffle_work,
                        construction_directory=context.paths.construction,
                        vector_bundle_directory=context.paths.vectors,
                        questions=context.t_steer,
                        prompts=context.construction_prompts,
                        scope_selection=selections["scope"],
                        runtime_identity=context.source_runtime_identity,
                        checkpoint_rows=runbook.shuffle_checkpoint_rows,
                    )
                    return _action("prepared-shuffled-control", plan_identity=plan["plan_identity"])
                shuffle = verify_e3_shuffled_control_work(
                    context.paths.shuffle_work,
                    construction_directory=context.paths.construction,
                    vector_bundle_directory=context.paths.vectors,
                    questions=context.t_steer,
                    prompts=context.construction_prompts,
                    scope_selection=selections["scope"],
                )
                if not shuffle["complete"]:
                    progress = run_e3_shuffled_control(
                        context.paths.shuffle_work,
                        construction_directory=context.paths.construction,
                        vector_bundle_directory=context.paths.vectors,
                        questions=context.t_steer,
                        prompts=context.construction_prompts,
                        scope_selection=selections["scope"],
                        runtime=runtime,
                        request_budget=request_budget,
                    )
                    return _action("ran-shuffled-control", **dict(progress))
                if not context.paths.shuffled_vectors.exists():
                    result = finalize_e3_shuffled_control_bundle(
                        context.paths.shuffled_vectors,
                        work_directory=context.paths.shuffle_work,
                        construction_directory=context.paths.construction,
                        vector_bundle_directory=context.paths.vectors,
                        questions=context.t_steer,
                        prompts=context.construction_prompts,
                        scope_selection=selections["scope"],
                    )
                    return _action("finalized-shuffled-control", **dict(result))
                verify_e3_shuffled_control_bundle(
                    context.paths.shuffled_vectors,
                    work_directory=context.paths.shuffle_work,
                    construction_directory=context.paths.construction,
                    vector_bundle_directory=context.paths.vectors,
                    questions=context.t_steer,
                    prompts=context.construction_prompts,
                    scope_selection=selections["scope"],
                )
                if not context.paths.fixed_controls.exists():
                    result = write_e3_fixed_control_materials(
                        context.paths.fixed_controls,
                        construction_directory=context.paths.construction,
                        vector_bundle_directory=context.paths.vectors,
                        questions=context.t_steer,
                        prompts=context.construction_prompts,
                        scope_selection=selections["scope"],
                        dev_questions=context.t_dev,
                    )
                    return _action("materialized-fixed-controls", **dict(result))
                verify_e3_fixed_control_materials(
                    context.paths.fixed_controls,
                    construction_directory=context.paths.construction,
                    vector_bundle_directory=context.paths.vectors,
                    questions=context.t_steer,
                    prompts=context.construction_prompts,
                    scope_selection=selections["scope"],
                    dev_questions=context.t_dev,
                )
            assets = known_assets.get(stage) or _stage_assets(
                context, runtime, stage=stage, predecessor=predecessor
            )
            work = context.paths.stage(stage)
            questions = _stage_questions(context, stage)
            if not work.exists():
                plan = prepare_e3_evaluation_work(
                    work,
                    stage=stage,
                    assets=assets,
                    runtime_identity=context.source_runtime_identity,
                    selection_receipt=predecessor,
                    max_new_tokens=runbook.max_new_tokens,
                )
                return _action("prepared-stage", stage=stage, plan_identity=plan["plan_identity"])
            progress = verify_e3_evaluation_work(
                work,
                stage=stage,
                assets=assets,
                evaluation_questions=questions,
                selection_receipt=predecessor,
            )
            if not progress["complete"]:
                progress = run_e3_evaluation(
                    work,
                    stage=stage,
                    assets=assets,
                    evaluation_questions=questions,
                    application_prompts=context.application_prompts,
                    runtime=runtime,
                    selection_receipt=predecessor,
                    request_budget=request_budget,
                )
                return _action("ran-stage", stage=stage, **dict(progress))
            if stage in _SELECTION_STAGES and stage not in selections:
                context.paths.selections.mkdir(parents=True, exist_ok=True)
                inputs = e3_selection_inputs_from_work(
                    work,
                    stage=stage,
                    assets=assets,
                    evaluation_questions=questions,
                    selection_receipt=predecessor,
                )
                write_e3_stage_selection(context.paths.selection(stage), **inputs)
                return _action("selected-stage", stage=stage)
        return _finalize_or_verify_phase(context, runtime)
    finally:
        runtime.close()


def _all_stage_materials(
    context: _E3Context,
    runtime: VllmResearchRuntime,
) -> tuple[
    dict[str, Path],
    dict[str, E3ExecutionAssets],
    dict[str, tuple[Question, ...]],
    dict[str, VerifiedE3StageSelection],
]:
    selections, _known = _load_selections(context, runtime)
    if set(selections) != set(_SELECTION_STAGES):
        raise FrozenArtifactError("E3 operator selections are incomplete")
    assets = {
        stage: _stage_assets(
            context,
            runtime,
            stage=stage,
            predecessor=_predecessor(stage, selections),
        )
        for stage in _STAGES
    }
    return (
        {stage: context.paths.stage(stage) for stage in _STAGES},
        assets,
        {stage: _stage_questions(context, stage) for stage in _STAGES},
        selections,
    )


def _finalize_or_verify_phase(
    context: _E3Context,
    runtime: VllmResearchRuntime,
) -> Mapping[str, Any]:
    stage_runs, assets, questions, selections = _all_stage_materials(context, runtime)
    if not context.paths.phase.exists():
        result = finalize_e3_phase(
            context.paths.phase,
            stage_runs=stage_runs,
            stage_assets=assets,
            stage_questions=questions,
            selection_receipts=selections,
            study=context.study,
            input_artifacts=context.runbook.input_artifacts,
            prerequisite_runs=context.runbook.prerequisite_runs,
        )
        return _action("finalized-phase", **dict(result))
    result = verify_e3_phase(
        context.paths.phase,
        stage_runs=stage_runs,
        stage_assets=assets,
        stage_questions=questions,
        selection_receipts=selections,
        study=context.study,
        input_artifacts=context.runbook.input_artifacts,
        prerequisite_runs=context.runbook.prerequisite_runs,
    )
    return _action("verified-complete-phase", **dict(result))


def verify_e3_operator(runbook: E3OperatorRunbook) -> Mapping[str, Any]:
    """Replay the complete construction, controls, stages, selections, and phase."""

    preflight = preflight_e3_operator(runbook)
    context = _operator_context(runbook)
    if not context.paths.phase.is_dir():
        raise FrozenArtifactError("E3 completed phase is absent")
    runtime = _live_runtime(context)
    try:
        result = _finalize_or_verify_phase(context, runtime)
    finally:
        runtime.close()
    if result["action"] != "verified-complete-phase":
        raise FrozenArtifactError("E3 verification unexpectedly mutated phase state")
    return MappingProxyType(
        {
            **dict(result),
            "runbook_digest": runbook.runbook_digest,
            "source_runtime_identity_sha256": preflight["source_runtime_identity_sha256"],
            "runbook_sha256": sha256_file(runbook.path),
        }
    )

"""Executable preparation, resume, and terminal workflow for confirmatory E9."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from mfh.artifact_namespace import validate_active_study_artifact_paths
from mfh.config import load_model_spec, load_prompt_specs
from mfh.contracts import GenerationRecord, InterventionSpec, ModelSpec, PromptSpec, Question
from mfh.data.io import read_questions
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.experiments.e9_native import NativeE9VllmBackend
from mfh.experiments.evidence import GateResult
from mfh.experiments.gates import write_gate_evidence
from mfh.experiments.model_selection import validate_active_model_spec
from mfh.experiments.protocol import ExperimentPhase, StudyProtocol
from mfh.experiments.runner import (
    PhaseCompletion,
    PhaseRunContract,
    PhaseRunLedger,
    expand_factorial_conditions,
    open_phase_prerequisite,
)
from mfh.provenance import sha256_path

_PROMPTS = ("P0-neutral", "P1-direct", "P2-calibrated-abstention")
_PARTITIONS = {
    "triviaqa": "T-test",
    "simpleqa_verified": "simpleqa-eval",
    "aa_omniscience_public_600": "aa-eval",
}
_COUNTS = {
    "triviaqa": 5_000,
    "simpleqa_verified": 1_000,
    "aa_omniscience_public_600": 600,
}
_METHODS = ("M0", "M1", "M2", "M3", "M4", "M5")


@dataclass(frozen=True, slots=True)
class E9ExecutionAssets:
    ledger: PhaseRunLedger
    prompts: Mapping[str, PromptSpec]
    questions: Mapping[str, Question]
    component_artifacts: Mapping[tuple[str, str], Path]

    def __post_init__(self) -> None:
        if self.ledger.contract.phase is not ExperimentPhase.E9:
            raise DataValidationError("E9 execution assets require an E9 ledger")
        object.__setattr__(self, "prompts", MappingProxyType(dict(self.prompts)))
        object.__setattr__(self, "questions", MappingProxyType(dict(self.questions)))
        object.__setattr__(
            self,
            "component_artifacts",
            MappingProxyType(dict(self.component_artifacts)),
        )


def build_e9_contract(
    *,
    study: StudyProtocol,
    model: ModelSpec,
    prompts: Mapping[str, PromptSpec],
    questions_by_benchmark: Mapping[str, Sequence[Question]],
    interventions: Mapping[str, InterventionSpec],
    input_fingerprints: Mapping[str, str],
    prerequisite_digests: Mapping[str, str],
    seed: int = 17,
) -> PhaseRunContract:
    """Construct the only accepted 118,800-row E9 schedule."""

    validate_active_model_spec(model)
    phase = study.phase(ExperimentPhase.E9)
    if (
        phase.models != (model.name,)
        or set(prompts) != set(_PROMPTS)
        or set(questions_by_benchmark) != set(_COUNTS)
        or set(interventions) != set(_METHODS)
        or set(input_fingerprints) != set(phase.required_inputs) | set(phase.freeze_fields)
        or set(prerequisite_digests) != {value.value for value in phase.prerequisites}
    ):
        raise DataValidationError("E9 inputs differ from the frozen factorial protocol")
    normalized_questions: dict[str, tuple[Question, ...]] = {}
    seen_ids: set[str] = set()
    for benchmark, expected_count in _COUNTS.items():
        values = tuple(questions_by_benchmark[benchmark])
        identifiers = tuple(value.question_id for value in values)
        if (
            len(values) != expected_count
            or any(value.benchmark != benchmark for value in values)
            or len(set(identifiers)) != expected_count
            or seen_ids.intersection(identifiers)
        ):
            raise DataValidationError(f"E9 {benchmark} question schedule differs")
        seen_ids.update(identifiers)
        normalized_questions[benchmark] = values
    conditions = expand_factorial_conditions(
        study,
        ExperimentPhase.E9,
        models={model.name: model},
        prompts=prompts,
        benchmark_partitions=_PARTITIONS,
        interventions=interventions,
        seed=seed,
    )
    contract = PhaseRunContract(
        phase=ExperimentPhase.E9,
        study_protocol_digest=study.digest,
        conditions=conditions,
        question_ids_by_benchmark={
            benchmark: tuple(value.question_id for value in questions)
            for benchmark, questions in normalized_questions.items()
        },
        input_fingerprints=input_fingerprints,
        prerequisite_digests=prerequisite_digests,
        required_gates=phase.gates,
    )
    contract.assert_matches_study(study)
    if len(conditions) != 54 or contract.expected_record_count != 118_800:
        raise DataValidationError("E9 factorial cardinality differs from the preregistration")
    return contract


def create_e9_ledger(
    directory: str | Path,
    *,
    study: StudyProtocol,
    model_config: str | Path,
    prompt_config: str | Path,
    questions_by_benchmark: Mapping[str, Sequence[Question]],
    interventions: Mapping[str, InterventionSpec],
    input_artifacts: Mapping[str, str | Path],
    prerequisite_runs: Mapping[str, str | Path],
    seed: int = 17,
) -> PhaseRunLedger:
    """Verify live E0-E8 completions and atomically create the frozen E9 ledger."""

    model = load_model_spec(model_config)
    available_prompts = {value.prompt_id: value for value in load_prompt_specs(prompt_config)}
    try:
        prompts = {name: available_prompts[name] for name in _PROMPTS}
    except KeyError as exc:
        raise DataValidationError(f"E9 prompt is unavailable: {exc.args[0]}") from exc
    if set(prerequisite_runs) != {
        value.value for value in study.phase(ExperimentPhase.E9).prerequisites
    }:
        raise DataValidationError("E9 prerequisite run paths differ from the protocol")
    prerequisite_digests: dict[str, str] = {}
    for name, path in prerequisite_runs.items():
        phase = ExperimentPhase(name)
        completion = open_phase_prerequisite(
            path,
            phase=phase,
            study=study,
        ).verify_complete()
        if completion.phase.value != name:
            raise DataValidationError(f"E9 prerequisite {name} resolves to another phase")
        prerequisite_digests[name] = completion.completion_digest
    fingerprints = {name: sha256_path(path) for name, path in input_artifacts.items()}
    contract = build_e9_contract(
        study=study,
        model=model,
        prompts=prompts,
        questions_by_benchmark=questions_by_benchmark,
        interventions=interventions,
        input_fingerprints=fingerprints,
        prerequisite_digests=prerequisite_digests,
        seed=seed,
    )
    return PhaseRunLedger.create(
        directory,
        contract,
        study=study,
        input_artifacts=input_artifacts,
        prerequisite_runs=prerequisite_runs,
        confirmatory_prompts=prompts,
    )


def load_e9_execution_assets(
    run_directory: str | Path,
    *,
    study: StudyProtocol,
) -> E9ExecutionAssets:
    """Reopen only packaged E9 inputs and bind every pending row to exact content."""

    ledger = PhaseRunLedger.open(run_directory, study=study)
    if ledger.contract.phase is not ExperimentPhase.E9:
        raise DataValidationError("E9 execution received a different phase ledger")
    prompts = dict(ledger.confirmatory_prompts())
    question_root = ledger.directory / "inputs" / "frozen_question_bundle"
    questions: dict[str, Question] = {}
    for benchmark, expected_ids in ledger.contract.question_ids_by_benchmark.items():
        values = tuple(read_questions(question_root / f"{benchmark}.jsonl"))
        if tuple(value.question_id for value in values) != expected_ids:
            raise FrozenArtifactError("packaged E9 question order differs from the contract")
        for value in values:
            if value.question_id in questions:
                raise FrozenArtifactError("packaged E9 question identifiers repeat")
            questions[value.question_id] = value
    component_root = ledger.directory / "inputs" / "frozen_component_selection"
    try:
        manifest = json.loads((component_root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read packaged E9 components: {exc}") from exc
    components: dict[tuple[str, str], Path] = {}
    descriptors = manifest.get("components") if isinstance(manifest, dict) else None
    if not isinstance(descriptors, list):
        raise FrozenArtifactError("packaged E9 component descriptors are invalid")
    for descriptor in descriptors:
        if not isinstance(descriptor, Mapping):
            raise FrozenArtifactError("packaged E9 component descriptor is invalid")
        key = (str(descriptor.get("model_name")), str(descriptor.get("method")))
        relative = descriptor.get("component_path")
        if not isinstance(relative, str):
            raise FrozenArtifactError("packaged E9 component path is invalid")
        artifact = component_root / relative / "artifact"
        expected = descriptor.get("artifact_sha256")
        if artifact.is_symlink() or not artifact.exists() or sha256_path(artifact) != expected:
            raise FrozenArtifactError("packaged E9 component changed")
        components[key] = artifact
    expected_components = {
        (condition.model_name, condition.steering_method)
        for condition in ledger.contract.conditions
        if condition.steering_method != "M0"
    }
    if set(components) != expected_components:
        raise FrozenArtifactError("packaged E9 component set differs from the matrix")
    return E9ExecutionAssets(ledger, prompts, questions, components)


def execute_e9_pending(
    assets: E9ExecutionAssets,
    backend: NativeE9VllmBackend,
    *,
    checkpoint_size: int = 1,
    limit: int | None = None,
) -> int:
    """Execute pending rows in deterministic resume order and checkpoint atomically."""

    if (
        type(backend) is not NativeE9VllmBackend
        or checkpoint_size <= 0
        or (limit is not None and limit <= 0)
    ):
        raise DataValidationError("E9 checkpoint size and limit must be positive")
    completed = 0
    batch: list[GenerationRecord] = []
    for pending in assets.ledger.iter_pending():
        if limit is not None and completed >= limit:
            break
        condition = pending.condition
        question = assets.questions[pending.question_id]
        component = (
            None
            if condition.steering_method == "M0"
            else assets.component_artifacts[(condition.model_name, condition.steering_method)]
        )
        selective_risk_component = assets.component_artifacts[(condition.model_name, "M3")]
        record = backend.execute(
            condition=condition,
            question=question,
            prompt=assets.prompts[condition.system_prompt_id],
            component_artifact=component,
            selective_risk_component_artifact=selective_risk_component,
        )
        batch.append(record)
        completed += 1
        if len(batch) == checkpoint_size:
            assets.ledger.checkpoint(batch)
            batch.clear()
    if batch:
        assets.ledger.checkpoint(batch)
    return completed


def finalize_e9(
    assets: E9ExecutionAssets,
    *,
    evidence_directory: str | Path,
) -> PhaseCompletion:
    """Derive both preregistered E9 gates from the complete immutable ledger."""

    completed, expected = assets.ledger.progress()
    if completed != expected:
        raise DataValidationError(f"E9 still has {expected - completed} pending rows")
    if (assets.ledger.directory / "complete.json").is_file():
        return assets.ledger.verify_complete()
    normalized = validate_active_study_artifact_paths(
        {
            "E9 ledger": assets.ledger.directory,
            "E9 evidence": evidence_directory,
        }
    )
    evidence_root = normalized["E9 evidence"]
    if evidence_root.resolve().is_relative_to(assets.ledger.directory.resolve()):
        raise DataValidationError("E9 external evidence cannot be inside its ledger")
    record_set = assets.ledger.record_set_digest()
    expected_inventory = {
        *(f"{gate}.json" for gate in assets.ledger.contract.required_gates),
        "e9_analysis",
    }

    def evaluate(root: Path) -> dict[str, GateResult]:
        if (
            root.is_symlink()
            or not root.is_dir()
            or {item.name for item in root.iterdir()} != expected_inventory
        ):
            raise FrozenArtifactError("E9 evidence inventory is incomplete")
        results: dict[str, GateResult] = {}
        for gate in assets.ledger.contract.required_gates:
            supporting = (
                {"e9_analysis": root / "e9_analysis"}
                if gate == "preregistered_analysis_only"
                else None
            )
            results[gate] = assets.ledger.evaluate_gate(
                gate,
                root / f"{gate}.json",
                supporting_artifacts=supporting,
            )
        return results

    if evidence_root.exists() or evidence_root.is_symlink():
        gate_results = evaluate(evidence_root)
    else:
        from mfh.experiments.e9_analysis import write_e9_analysis_bundle

        evidence_root.parent.mkdir(parents=True, exist_ok=True)
        stage_prefix = f".{evidence_root.name}.stage-"
        for stale in evidence_root.parent.glob(f"{stage_prefix}*"):
            if stale.is_dir() and not stale.is_symlink():
                shutil.rmtree(stale)
        stage = Path(tempfile.mkdtemp(prefix=stage_prefix, dir=evidence_root.parent))
        try:
            write_e9_analysis_bundle(stage / "e9_analysis", ledger=assets.ledger)
            for gate in assets.ledger.contract.required_gates:
                write_gate_evidence(
                    stage / f"{gate}.json",
                    phase=ExperimentPhase.E9,
                    gate=gate,
                    contract_digest=assets.ledger.contract.digest,
                    record_set_digest=record_set,
                    observations=(),
                )
            evaluate(stage)
            os.replace(stage, evidence_root)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
        gate_results = evaluate(evidence_root)
    completion = assets.ledger.finalize(gate_results)
    verified = PhaseRunLedger.open(assets.ledger.directory, study=assets.ledger.study)
    replayed = verified.verify_complete()
    if replayed.completion_digest != completion.completion_digest:
        raise FrozenArtifactError("E9 completion does not replay")
    return completion

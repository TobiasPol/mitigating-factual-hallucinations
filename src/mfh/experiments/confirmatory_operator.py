"""Secret-free operator runbooks for native-MLX E9 and one-shot E10."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from mfh.config import load_model_spec, load_prompt_specs
from mfh.contracts import (
    AdaptivePolicySpec,
    InterventionSpec,
    ModelSpec,
    Question,
)
from mfh.data.io import read_questions
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.evaluation.openrouter import OpenRouterTransport
from mfh.experiments.confirmatory_components import load_confirmatory_fixed_component
from mfh.experiments.confirmatory_graders import validate_confirmatory_grader_bundle
from mfh.experiments.e6_likelihood import E6RuntimeAttestor, _load_e6_runtime_attestation
from mfh.experiments.e9_factorial import (
    build_e9_contract,
    create_e9_ledger,
    execute_e9_pending,
    finalize_e9,
    load_e9_execution_assets,
)
from mfh.experiments.e9_native import NativeE9MlxBackend
from mfh.experiments.e10_composite import (
    _derive_e10_composite_provenance,
    _e6_runtime_artifact,
    _validate_exact_e6_runtime_binding,
    build_e10_contract,
    create_e10_ledger,
    e10_intervention,
    execute_e10_pending,
    finalize_e10,
    load_e10_execution_assets,
    validate_e10_freeze_inputs,
    validate_e10_prerequisite_bound_inputs,
)
from mfh.experiments.e10_native import NativeE10MlxBackend
from mfh.experiments.model_selection import validate_active_model_spec
from mfh.experiments.protocol import ExperimentPhase, StudyProtocol, load_study_protocol
from mfh.experiments.robustness_diagnostics import (
    _questions_from_source,
    verify_robustness_diagnostic_plan,
)
from mfh.experiments.runner import (
    PhaseRunContract,
    PhaseRunLedger,
    _validate_component_selection,
    _validate_e9_component_promotions,
    _validate_question_bundle,
    open_phase_prerequisite,
)
from mfh.experiments.snapshots import validate_execution_snapshot
from mfh.inference.mlx_research import MlxResearchRuntime
from mfh.inference.transformers_snapshot import (
    reject_symlink_path_components,
    verify_transformers_snapshot,
)
from mfh.provenance import sha256_file, sha256_path, stable_hash

_RUNBOOK_KEYS = {
    "schema_version",
    "phase",
    "study_protocol",
    "model_config",
    "prompt_config",
    "snapshot_directory",
    "snapshot_manifest",
    "run_directory",
    "evidence_directory",
    "input_artifacts",
    "prerequisite_runs",
    "seed",
}
_QUESTION_COUNTS = {
    ExperimentPhase.E9: {
        "triviaqa": 5_000,
        "simpleqa_verified": 1_000,
        "aa_omniscience_public_600": 600,
    },
    ExperimentPhase.E10: {
        "triviaqa": 5_000,
        "simpleqa_verified": 1_000,
        "aa_omniscience_public_600": 600,
        "ifeval": 541,
        "mmlu_pro": 1_000,
        "wikitext103": 1_000,
        "xstest": 250,
        "strongreject_or_harmbench": 313,
        "language_consistency": 500,
    },
}


def _canonical_lexical_path(path: str | Path, context: str) -> Path:
    """Normalize parent traversal without allowing it to conceal a symlink."""

    checked = reject_symlink_path_components(path, context)
    normalized = Path(os.path.abspath(checked))
    return reject_symlink_path_components(normalized, context)


def _strict_path(root: Path, value: object, context: str) -> Path:
    if type(value) is not str or not value.strip():
        raise DataValidationError(f"confirmatory runbook {context} path is invalid")
    raw = Path(value)
    return _canonical_lexical_path(
        raw if raw.is_absolute() else root / raw,
        f"confirmatory runbook {context}",
    )


def _path_mapping(root: Path, value: object, context: str) -> Mapping[str, Path]:
    if not isinstance(value, Mapping) or any(type(name) is not str or not name for name in value):
        raise DataValidationError(f"confirmatory runbook {context} mapping is invalid")
    return MappingProxyType(
        {str(name): _strict_path(root, path, f"{context}.{name}") for name, path in value.items()}
    )


@dataclass(frozen=True, slots=True)
class ConfirmatoryRunbook:
    """All non-secret paths needed for one E9 or E10 lifecycle."""

    source: Path
    phase: ExperimentPhase
    study_protocol: Path
    model_config: Path
    prompt_config: Path
    snapshot_directory: Path
    snapshot_manifest: Path
    run_directory: Path
    evidence_directory: Path
    input_artifacts: Mapping[str, Path]
    prerequisite_runs: Mapping[str, Path]
    seed: int
    runbook_digest: str

    def __post_init__(self) -> None:
        if (
            self.phase not in {ExperimentPhase.E9, ExperimentPhase.E10}
            or isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
            or not self.runbook_digest
        ):
            raise DataValidationError("confirmatory runbook identity is invalid")
        object.__setattr__(self, "input_artifacts", MappingProxyType(dict(self.input_artifacts)))
        object.__setattr__(
            self,
            "prerequisite_runs",
            MappingProxyType(dict(self.prerequisite_runs)),
        )

    @classmethod
    def load(cls, path: str | Path) -> ConfirmatoryRunbook:
        source = _canonical_lexical_path(path, "confirmatory runbook")
        if not source.is_file():
            raise FrozenArtifactError("confirmatory runbook must be one regular JSON file")
        try:
            value = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FrozenArtifactError(f"cannot read confirmatory runbook: {exc}") from exc
        if not isinstance(value, dict) or set(value) != _RUNBOOK_KEYS:
            raise DataValidationError("confirmatory runbook keys differ from schema version 1")
        if value["schema_version"] != 1:
            raise DataValidationError("unsupported confirmatory runbook schema")
        try:
            phase = ExperimentPhase(value["phase"])
        except (TypeError, ValueError) as exc:
            raise DataValidationError("confirmatory runbook phase is invalid") from exc
        root = source.parent
        seed = value["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise DataValidationError("confirmatory runbook seed is invalid")
        return cls(
            source=source,
            phase=phase,
            study_protocol=_strict_path(root, value["study_protocol"], "study_protocol"),
            model_config=_strict_path(root, value["model_config"], "model_config"),
            prompt_config=_strict_path(root, value["prompt_config"], "prompt_config"),
            snapshot_directory=_strict_path(
                root, value["snapshot_directory"], "snapshot_directory"
            ),
            snapshot_manifest=_strict_path(root, value["snapshot_manifest"], "snapshot_manifest"),
            run_directory=_strict_path(root, value["run_directory"], "run_directory"),
            evidence_directory=_strict_path(
                root, value["evidence_directory"], "evidence_directory"
            ),
            input_artifacts=_path_mapping(root, value["input_artifacts"], "input_artifacts"),
            prerequisite_runs=_path_mapping(root, value["prerequisite_runs"], "prerequisite_runs"),
            seed=seed,
            runbook_digest=stable_hash(value),
        )


def _study_and_model(runbook: ConfirmatoryRunbook) -> tuple[StudyProtocol, ModelSpec]:
    study = load_study_protocol(runbook.study_protocol)
    model = load_model_spec(runbook.model_config)
    validate_active_model_spec(model)
    if runbook.phase not in study.by_phase:
        raise DataValidationError("confirmatory runbook phase is absent from the study")
    verify_transformers_snapshot(
        model,
        runbook.snapshot_directory,
        runbook.snapshot_manifest,
    )
    return study, model


def _questions(runbook: ConfirmatoryRunbook) -> Mapping[str, tuple[Question, ...]]:
    expected = _QUESTION_COUNTS[runbook.phase]
    root = runbook.input_artifacts.get("frozen_question_bundle")
    if root is None:
        raise DataValidationError("confirmatory runbook lacks frozen_question_bundle")
    observed: dict[str, tuple[Question, ...]] = {}
    seen: set[str] = set()
    for benchmark, count in expected.items():
        values = tuple(read_questions(root / f"{benchmark}.jsonl"))
        identifiers = tuple(item.question_id for item in values)
        if (
            len(values) != count
            or any(item.benchmark != benchmark for item in values)
            or len(set(identifiers)) != count
            or seen.intersection(identifiers)
        ):
            raise DataValidationError(f"confirmatory runbook {benchmark} question schedule differs")
        seen.update(identifiers)
        observed[benchmark] = values
    return MappingProxyType(observed)


def _verified_prerequisites(
    runbook: ConfirmatoryRunbook,
    *,
    study: StudyProtocol,
) -> tuple[Mapping[str, str], Mapping[ExperimentPhase, PhaseRunLedger]]:
    required = {item.value for item in study.phase(runbook.phase).prerequisites}
    if set(runbook.prerequisite_runs) != required:
        raise DataValidationError("confirmatory runbook prerequisite set differs")
    digests: dict[str, str] = {}
    ledgers: dict[ExperimentPhase, PhaseRunLedger] = {}
    for name, path in runbook.prerequisite_runs.items():
        phase = ExperimentPhase(name)
        ledger = open_phase_prerequisite(path, phase=phase, study=study)
        completion = ledger.verify_complete()
        digests[name] = completion.completion_digest
        ledgers[phase] = ledger
    return MappingProxyType(digests), MappingProxyType(ledgers)


def _selection_manifest(path: Path, *, phase: ExperimentPhase) -> tuple[Mapping[str, Any], ...]:
    try:
        value = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read confirmatory component selection: {exc}") from exc
    if not isinstance(value, dict):
        raise FrozenArtifactError("confirmatory component-selection manifest is invalid")
    body = dict(value)
    digest = body.pop("manifest_digest", None)
    components = body.get("components")
    if (
        set(body) != {"schema_version", "study_protocol_digest", "phase", "components"}
        or body["schema_version"] != 3
        or body["phase"] != phase.value
        or digest != stable_hash(body)
        or not isinstance(components, list)
        or any(not isinstance(item, Mapping) for item in components)
    ):
        raise FrozenArtifactError("confirmatory component-selection manifest differs")
    return tuple(dict(item) for item in components if isinstance(item, Mapping))


def _e9_interventions(path: Path) -> Mapping[str, InterventionSpec]:
    interventions: dict[str, InterventionSpec] = {"M0": InterventionSpec(method="M0")}
    for descriptor in _selection_manifest(path, phase=ExperimentPhase.E9):
        method = descriptor.get("method")
        relative = descriptor.get("component_path")
        fingerprint = descriptor.get("artifact_sha256")
        if (
            method not in {"M1", "M2", "M3", "M4", "M5"}
            or type(relative) is not str
            or type(fingerprint) is not str
        ):
            raise FrozenArtifactError("E9 component descriptor identity is invalid")
        artifact = path / relative / "artifact"
        if sha256_path(artifact) != fingerprint:
            raise FrozenArtifactError("E9 component artifact changed")
        if method == "M3":
            policy = descriptor.get("adaptive_policy")
            if not isinstance(policy, Mapping):
                raise FrozenArtifactError("E9 M3 component lacks its adaptive policy")
            intervention = InterventionSpec(
                method=method,
                artifact_sha256=fingerprint,
                adaptive_policy=AdaptivePolicySpec.from_dict(policy),
            )
        else:
            if descriptor.get("adaptive_policy") is not None:
                raise FrozenArtifactError("E9 fixed component declares an adaptive policy")
            fixed = load_confirmatory_fixed_component(artifact)
            if fixed.method != method or fixed.fingerprint != fingerprint:
                raise FrozenArtifactError("E9 fixed component geometry differs")
            intervention = InterventionSpec(
                method=method,
                layer=fixed.layer,
                site=fixed.site,
                token_scope=fixed.token_scope,
                alpha=fixed.standardized_alpha,
                sparsity=fixed.sparsity,
                artifact_sha256=fixed.fingerprint,
                decay=fixed.decay,
            )
        if method in interventions:
            raise FrozenArtifactError("E9 component selection repeats a method")
        interventions[method] = intervention
    if set(interventions) != {"M0", "M1", "M2", "M3", "M4", "M5"}:
        raise FrozenArtifactError("E9 component selection is incomplete")
    return MappingProxyType(interventions)


def _e10_component(path: Path) -> Path:
    descriptors = _selection_manifest(path, phase=ExperimentPhase.E10)
    if len(descriptors) != 1:
        raise FrozenArtifactError("E10 component selection must contain one M6")
    descriptor = descriptors[0]
    if descriptor.get("method") != "M6" or type(descriptor.get("component_path")) is not str:
        raise FrozenArtifactError("E10 component selection does not contain M6")
    artifact = path / str(descriptor["component_path"]) / "artifact"
    if sha256_path(artifact) != descriptor.get("artifact_sha256"):
        raise FrozenArtifactError("E10 selected M6 artifact changed")
    return artifact


def _validate_e9_runtime_binding(
    runbook: ConfirmatoryRunbook,
    *,
    contract: PhaseRunContract,
    prerequisite_ledgers: Mapping[ExperimentPhase, PhaseRunLedger],
) -> None:
    """Bind E9's grader and every adaptive condition to the exact E6 key."""

    try:
        e6_ledger = prerequisite_ledgers[ExperimentPhase.E6]
    except KeyError as exc:
        raise DataValidationError("E9 runtime binding lacks its E6 prerequisite") from exc
    public_key = _validate_exact_e6_runtime_binding(
        runtime_artifact=_e6_runtime_artifact(e6_ledger),
        grader=runbook.input_artifacts["frozen_graders"],
    )
    adaptive_conditions = tuple(
        condition for condition in contract.conditions if condition.steering_method == "M3"
    )
    if not adaptive_conditions or any(
        condition.adaptive_policy is None
        or condition.adaptive_policy.execution_public_key != public_key
        for condition in adaptive_conditions
    ):
        raise FrozenArtifactError("E9 adaptive policies differ from the exact E6 execution key")


def _validate_e9_robustness_schedule(
    runbook: ConfirmatoryRunbook,
    *,
    questions: Mapping[str, tuple[Question, ...]],
    prerequisite_ledgers: Mapping[ExperimentPhase, PhaseRunLedger],
) -> None:
    """Bind E9 to the complete post-E8/pre-E9 diagnostic freeze."""

    plan = verify_robustness_diagnostic_plan(
        runbook.input_artifacts["frozen_prompt_paraphrase_schedule"]
    )
    bindings = plan.body.get("source_artifact_sha256")
    provenance = plan.body.get("e1_provenance")
    if (
        plan.path is None
        or not isinstance(bindings, Mapping)
        or not isinstance(provenance, Mapping)
    ):
        raise FrozenArtifactError("E9 robustness schedule lacks packaged provenance")
    expected_bindings = {
        "frozen-component-selection": runbook.input_artifacts["frozen_component_selection"],
        "frozen-evaluation-scripts": runbook.input_artifacts["frozen_evaluation_scripts"],
        "frozen-graders": runbook.input_artifacts["frozen_graders"],
    }
    if any(bindings.get(name) != sha256_path(path) for name, path in expected_bindings.items()):
        raise FrozenArtifactError("E9 inputs differ from the frozen robustness schedule")
    try:
        e1_completion = prerequisite_ledgers[ExperimentPhase.E1].verify_complete()
    except KeyError as exc:  # pragma: no cover - the E9 study requires E1
        raise FrozenArtifactError("E9 robustness schedule lacks its E1 prerequisite") from exc
    if provenance.get("e1_completion_digest") != e1_completion.completion_digest:
        raise FrozenArtifactError("E9 robustness schedule derives from another E1 run")
    source_root = plan.path / "sources"
    expected_questions = {
        "triviaqa": _questions_from_source(
            "triviaqa-evaluation", source_root / "triviaqa-evaluation"
        ),
        "simpleqa_verified": _questions_from_source(
            "simpleqa_verified-evaluation",
            source_root / "simpleqa_verified-evaluation",
        ),
        "aa_omniscience_public_600": _questions_from_source(
            "aa_omniscience_public_600-evaluation",
            source_root / "aa_omniscience_public_600-evaluation",
        ),
    }
    if dict(questions) != expected_questions:
        raise FrozenArtifactError(
            "E9 factual matrix differs from the robustness-bound reviewed evaluation splits"
        )


def _input_inventory(
    runbook: ConfirmatoryRunbook,
    *,
    study: StudyProtocol,
) -> Mapping[str, str]:
    phase = study.phase(runbook.phase)
    expected = set(phase.required_inputs) | set(phase.freeze_fields)
    if set(runbook.input_artifacts) != expected:
        raise DataValidationError("confirmatory runbook input-artifact set differs")
    return MappingProxyType(
        {name: sha256_path(path) for name, path in runbook.input_artifacts.items()}
    )


def _preflight_contract(
    runbook: ConfirmatoryRunbook,
) -> tuple[StudyProtocol, ModelSpec, PhaseRunContract]:
    study, model = _study_and_model(runbook)
    questions = _questions(runbook)
    prerequisite_digests, prerequisite_ledgers = _verified_prerequisites(runbook, study=study)
    fingerprints = _input_inventory(runbook, study=study)
    prompts = {item.prompt_id: item for item in load_prompt_specs(runbook.prompt_config)}
    if runbook.phase is ExperimentPhase.E9:
        _validate_e9_robustness_schedule(
            runbook,
            questions=questions,
            prerequisite_ledgers=prerequisite_ledgers,
        )
        selected_prompts = {
            name: prompts[name] for name in ("P0-neutral", "P1-direct", "P2-calibrated-abstention")
        }
        contract = build_e9_contract(
            study=study,
            model=model,
            prompts=selected_prompts,
            questions_by_benchmark=questions,
            interventions=_e9_interventions(runbook.input_artifacts["frozen_component_selection"]),
            input_fingerprints=fingerprints,
            prerequisite_digests=prerequisite_digests,
            seed=runbook.seed,
        )
        _validate_question_bundle(runbook.input_artifacts["frozen_question_bundle"], contract)
        _validate_component_selection(
            runbook.input_artifacts["frozen_component_selection"], contract
        )
        _validate_e9_component_promotions(
            runbook.input_artifacts["frozen_component_selection"],
            contract,
            prerequisite_ledgers,
        )
        _validate_e9_runtime_binding(
            runbook,
            contract=contract,
            prerequisite_ledgers=prerequisite_ledgers,
        )
        validate_execution_snapshot(
            runbook.input_artifacts["frozen_evaluation_scripts"],
            study_protocol_digest=study.digest,
            phase=ExperimentPhase.E9,
        )
        validate_confirmatory_grader_bundle(runbook.input_artifacts["frozen_graders"])
    else:
        provenance = _derive_e10_composite_provenance(prerequisite_ledgers=prerequisite_ledgers)
        prompt_id = str(provenance["selected_prompt_id"])
        try:
            prompt = prompts[prompt_id]
        except KeyError as exc:
            raise FrozenArtifactError("E10 selected prompt is absent from prompt config") from exc
        component_artifact = _e10_component(runbook.input_artifacts["component_selection_manifest"])
        intervention = e10_intervention(
            component_artifact=component_artifact,
            study=study,
            e6_run=runbook.prerequisite_runs[ExperimentPhase.E6.value],
        )
        contract = build_e10_contract(
            study=study,
            model=model,
            prompt=prompt,
            questions_by_benchmark=questions,
            intervention=intervention,
            input_fingerprints=fingerprints,
            prerequisite_digests=prerequisite_digests,
            seed=runbook.seed,
        )
        _validate_question_bundle(runbook.input_artifacts["frozen_question_bundle"], contract)
        validate_e10_prerequisite_bound_inputs(
            runbook.input_artifacts,
            contract=contract,
            prompts={prompt.prompt_id: prompt},
            prerequisite_ledgers=prerequisite_ledgers,
        )
        adaptive_policy = intervention.adaptive_policy
        if adaptive_policy is None or adaptive_policy.execution_public_key is None:
            raise FrozenArtifactError("E10 intervention lacks its exact E6 execution key")
        validate_e10_freeze_inputs(
            {
                name: runbook.input_artifacts[name]
                for name in study.phase(ExperimentPhase.E10).freeze_fields
            },
            model=model,
            prompt=prompt,
            composite_policy_artifact=component_artifact,
            study_protocol_digest=study.digest,
            expected_runtime_artifact=_e6_runtime_artifact(
                prerequisite_ledgers[ExperimentPhase.E6]
            ),
            expected_execution_public_key=adaptive_policy.execution_public_key,
        )
    return study, model, contract


def _open_runbook_bound_ledger(
    runbook: ConfirmatoryRunbook,
    *,
    study: StudyProtocol,
    expected_contract: PhaseRunContract,
) -> PhaseRunLedger:
    """Open only the exact ledger re-derived from the current runbook sources."""

    ledger = PhaseRunLedger.open(runbook.run_directory, study=study)
    if (
        ledger.contract.phase is not runbook.phase
        or ledger.contract.digest != expected_contract.digest
    ):
        raise FrozenArtifactError("confirmatory ledger differs from the runbook contract")
    return ledger


def preflight_confirmatory_runbook(
    runbook: ConfirmatoryRunbook,
) -> Mapping[str, Any]:
    """Replay all non-mutating contract, input, snapshot, and prerequisite checks."""

    study, model, contract = _preflight_contract(runbook)
    existing: dict[str, Any] | None = None
    if runbook.run_directory.exists():
        ledger = _open_runbook_bound_ledger(
            runbook,
            study=study,
            expected_contract=contract,
        )
        completed, expected = ledger.progress()
        existing = {"completed_records": completed, "expected_records": expected}
    return MappingProxyType(
        {
            "valid": True,
            "phase": runbook.phase.value,
            "runbook_digest": runbook.runbook_digest,
            "study_protocol_digest": study.digest,
            "model_repository": model.repository,
            "model_revision": model.revision,
            "contract_digest": contract.digest,
            "condition_count": len(contract.conditions),
            "expected_record_count": contract.expected_record_count,
            "existing_run": existing,
            "one_shot_reservation_consumed": (
                runbook.phase is ExperimentPhase.E10 and existing is not None
            ),
        }
    )


def prepare_confirmatory_runbook(
    runbook: ConfirmatoryRunbook,
    *,
    authorize_one_shot: bool = False,
) -> PhaseRunLedger:
    """Create a frozen E9/E10 ledger; E10 requires explicit one-shot authority."""

    if runbook.phase is ExperimentPhase.E10 and not authorize_one_shot:
        raise DataValidationError("E10 prepare requires explicit authorize_one_shot=True")
    study, _model, _contract = _preflight_contract(runbook)
    questions = _questions(runbook)
    if runbook.phase is ExperimentPhase.E9:
        return create_e9_ledger(
            runbook.run_directory,
            study=study,
            model_config=runbook.model_config,
            prompt_config=runbook.prompt_config,
            questions_by_benchmark=questions,
            interventions=_e9_interventions(runbook.input_artifacts["frozen_component_selection"]),
            input_artifacts=runbook.input_artifacts,
            prerequisite_runs=runbook.prerequisite_runs,
            seed=runbook.seed,
        )
    intervention = e10_intervention(
        component_artifact=_e10_component(runbook.input_artifacts["component_selection_manifest"]),
        study=study,
        e6_run=runbook.prerequisite_runs[ExperimentPhase.E6.value],
    )
    return create_e10_ledger(
        runbook.run_directory,
        study=study,
        model_config=runbook.model_config,
        prompt_config=runbook.prompt_config,
        questions_by_benchmark=questions,
        intervention=intervention,
        input_artifacts=runbook.input_artifacts,
        prerequisite_runs=runbook.prerequisite_runs,
        seed=runbook.seed,
    )


def _native_runtime(
    runbook: ConfirmatoryRunbook,
    *,
    execution_private_key: str,
    packaged_grader: Path,
) -> tuple[E6RuntimeAttestor, Path]:
    model = load_model_spec(runbook.model_config)
    grader = validate_confirmatory_grader_bundle(packaged_grader)
    runtime_artifact = grader.directory / "runtime-attestation.json"
    attestation = _load_e6_runtime_attestation(runtime_artifact)
    identity = attestation["runtime_identity"]
    if not isinstance(identity, Mapping):
        raise FrozenArtifactError("confirmatory runtime attestation lacks its identity")
    seed = identity.get("seed")
    provenance = identity.get("research_provenance")
    if isinstance(seed, bool) or not isinstance(seed, int) or not isinstance(provenance, Mapping):
        raise FrozenArtifactError("confirmatory runtime identity is incomplete")
    runtime = MlxResearchRuntime.from_spec(
        model,
        snapshot_path=runbook.snapshot_directory,
        seed=seed,
        research_provenance=provenance,
    )
    return (
        E6RuntimeAttestor(
            runtime,
            execution_private_key=execution_private_key,
        ),
        runtime_artifact,
    )


def execute_confirmatory_runbook(
    runbook: ConfirmatoryRunbook,
    *,
    execution_private_key: str,
    openrouter_api_key: str,
    checkpoint_size: int = 1,
    limit: int | None = None,
) -> Mapping[str, Any]:
    """Load the pinned Qwen MLX runtime and resume pending E9/E10 rows."""

    study, _model, contract = _preflight_contract(runbook)
    _open_runbook_bound_ledger(
        runbook,
        study=study,
        expected_contract=contract,
    )
    grader_name = "frozen_graders" if runbook.phase is ExperimentPhase.E9 else "grader"
    packaged_grader = runbook.run_directory / "inputs" / grader_name
    attestor, runtime_artifact = _native_runtime(
        runbook,
        execution_private_key=execution_private_key,
        packaged_grader=packaged_grader,
    )
    transport = OpenRouterTransport(api_key=openrouter_api_key)
    if runbook.phase is ExperimentPhase.E9:
        assets9 = load_e9_execution_assets(runbook.run_directory, study=study)
        if assets9.ledger.contract.digest != contract.digest:
            raise FrozenArtifactError("E9 execution assets differ from the runbook contract")
        backend = NativeE9MlxBackend(
            attestor=attestor,
            runtime_artifact=runtime_artifact,
            grader_bundle=packaged_grader,
            grader_transport=transport,
        )
        executed = execute_e9_pending(
            assets9,
            backend,
            checkpoint_size=checkpoint_size,
            limit=limit,
        )
        ledger = assets9.ledger
    else:
        assets10 = load_e10_execution_assets(runbook.run_directory, study=study)
        if assets10.ledger.contract.digest != contract.digest:
            raise FrozenArtifactError("E10 execution assets differ from the runbook contract")
        backend10 = NativeE10MlxBackend(
            attestor=attestor,
            runtime_artifact=runtime_artifact,
            grader_bundle=packaged_grader,
            grader_transport=transport,
        )
        executed = execute_e10_pending(
            assets10,
            backend10,
            checkpoint_size=checkpoint_size,
            limit=limit,
        )
        ledger = assets10.ledger
    completed, expected = ledger.progress()
    return MappingProxyType(
        {
            "valid": True,
            "phase": runbook.phase.value,
            "executed_records": executed,
            "completed_records": completed,
            "expected_records": expected,
            "remaining_records": expected - completed,
        }
    )


def finalize_confirmatory_runbook(
    runbook: ConfirmatoryRunbook,
) -> Mapping[str, Any]:
    """Derive frozen gates and terminally finalize a complete E9/E10 ledger."""

    study, _model, contract = _preflight_contract(runbook)
    if runbook.phase is ExperimentPhase.E9:
        assets9 = load_e9_execution_assets(runbook.run_directory, study=study)
        if assets9.ledger.contract.digest != contract.digest:
            raise FrozenArtifactError("E9 finalization assets differ from the runbook contract")
        result = finalize_e9(
            assets9,
            evidence_directory=runbook.evidence_directory,
        )
        digest = result.completion_digest
        status = "complete"
    else:
        assets10 = load_e10_execution_assets(runbook.run_directory, study=study)
        if assets10.ledger.contract.digest != contract.digest:
            raise FrozenArtifactError("E10 finalization assets differ from the runbook contract")
        result10 = finalize_e10(
            assets10,
            evidence_directory=runbook.evidence_directory,
        )
        if hasattr(result10, "completion_digest"):
            digest = result10.completion_digest
            status = "complete"
        else:
            digest = result10.falsification_digest
            status = "falsified"
    return MappingProxyType(
        {
            "valid": True,
            "phase": runbook.phase.value,
            "status": status,
            "terminal_digest": digest,
        }
    )


def verify_confirmatory_runbook(
    runbook: ConfirmatoryRunbook,
) -> Mapping[str, Any]:
    """Read-only progress or terminal replay for the runbook-bound ledger."""

    study, _model, contract = _preflight_contract(runbook)
    ledger = _open_runbook_bound_ledger(
        runbook,
        study=study,
        expected_contract=contract,
    )
    completed, expected = ledger.progress()
    if (ledger.directory / "complete.json").is_file():
        terminal = ledger.verify_complete()
        status = "complete"
        digest = terminal.completion_digest
    elif (ledger.directory / "falsified.json").is_file():
        failure = ledger.verify_falsified()
        status = "falsified"
        digest = failure.falsification_digest
    else:
        status = "in_progress"
        digest = None
    return MappingProxyType(
        {
            "valid": True,
            "phase": runbook.phase.value,
            "status": status,
            "terminal_digest": digest,
            "completed_records": completed,
            "expected_records": expected,
            "remaining_records": expected - completed,
            "contract_digest": ledger.contract.digest,
            "record_set_digest": ledger.record_set_digest(),
        }
    )


def write_confirmatory_runbook_template(
    path: str | Path,
    *,
    phase: ExperimentPhase,
) -> str:
    """Write a secret-free, relative-path template without claiming readiness."""

    if phase not in {ExperimentPhase.E9, ExperimentPhase.E10}:
        raise DataValidationError("confirmatory template phase must be E9 or E10")
    destination = _canonical_lexical_path(
        path,
        "confirmatory runbook template",
    )
    required_inputs = (
        [
            "frozen_component_selection",
            "frozen_graders",
            "frozen_evaluation_scripts",
            "frozen_question_bundle",
            "frozen_prompt_paraphrase_schedule",
        ]
        if phase is ExperimentPhase.E9
        else [
            "E9_results",
            "component_selection_manifest",
            "frozen_question_bundle",
            "model_revision",
            "prompt",
            "risk_threshold",
            "vector_bank",
            "sae_checkpoint",
            "protected_subspace",
            "layer",
            "alpha_policy",
            "abstention_rule",
            "grader",
            "evaluation_scripts",
        ]
    )
    ordered_phases = tuple(ExperimentPhase)
    prerequisites = [item.value for item in ordered_phases[: ordered_phases.index(phase)]]
    body = {
        "schema_version": 1,
        "phase": phase.value,
        "study_protocol": "../../../../configs/experiments/phases.yaml",
        "model_config": "../../../../configs/models/qwen3.6-27b-mlx-4bit.yaml",
        "prompt_config": "../../../../configs/prompts/primary.yaml",
        "snapshot_directory": "../../../models/qwen3.6-27b-mlx-4bit/SNAPSHOT",
        "snapshot_manifest": "../../../../configs/models/qwen3.6-27b-mlx-4bit.snapshot.json",
        "run_directory": f"../runs/{phase.value}",
        "evidence_directory": f"../evidence/{phase.value}",
        "input_artifacts": {name: f"REPLACE/{name}" for name in required_inputs},
        "prerequisite_runs": {name: f"../runs/{name}" for name in prerequisites},
        "seed": 17,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(body, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise FrozenArtifactError("refusing to overwrite confirmatory runbook template") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return sha256_file(destination)


def summarize_runbook_paths(runbook: ConfirmatoryRunbook) -> Mapping[str, Any]:
    """Return redaction-safe paths for CLI output and operator logs."""

    return MappingProxyType(
        {
            "runbook": str(runbook.source),
            "runbook_digest": runbook.runbook_digest,
            "phase": runbook.phase.value,
            "run_directory": str(runbook.run_directory),
            "evidence_directory": str(runbook.evidence_directory),
            "input_artifacts": {name: str(path) for name, path in runbook.input_artifacts.items()},
            "prerequisite_runs": {
                name: str(path) for name, path in runbook.prerequisite_runs.items()
            },
        }
    )

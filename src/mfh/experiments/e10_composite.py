"""Frozen-input preparation and one-shot execution workflow for confirmatory E10."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from mfh.config import load_model_spec, load_prompt_specs
from mfh.contracts import (
    AdaptivePolicySpec,
    GenerationRecord,
    InterventionSpec,
    ModelSpec,
    PromptSpec,
    Question,
    TokenScope,
)
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.evaluation.metrics import metric_bundle
from mfh.experiments.e6_likelihood import _load_e6_runtime_attestation
from mfh.experiments.e8_protected import question_source_fingerprint
from mfh.experiments.e10_early_probe import (
    _reviewed_questions_from_e1,
    derive_e10_early_probe_capture_plan,
    load_e10_early_probe_selection,
)
from mfh.experiments.e10_native import NativeE10VllmBackend
from mfh.experiments.evidence import GateResult
from mfh.experiments.gates import write_gate_evidence
from mfh.experiments.model_selection import (
    validate_active_model_spec,
    validate_active_study_artifact_paths,
)
from mfh.experiments.protocol import ExperimentPhase, StudyProtocol
from mfh.experiments.runner import (
    EvaluationCondition,
    PhaseCompletion,
    PhaseFalsification,
    PhaseRunContract,
    PhaseRunLedger,
    _validate_component_selection,
    open_phase_prerequisite,
)
from mfh.experiments.snapshots import validate_execution_snapshot
from mfh.methods.composite import (
    CompositePolicy,
    CompositePolicyConfig,
    load_composite_policy,
    load_e10_composite_provenance,
    save_composite_policy,
)
from mfh.provenance import sha256_path, stable_hash

_COUNTS = {
    "triviaqa": 5_000,
    "simpleqa_verified": 1_000,
    "aa_omniscience_public_600": 600,
    "ifeval": 541,
    "mmlu_pro": 1_000,
    "wikitext103": 1_000,
    "xstest": 250,
    "strongreject_or_harmbench": 313,
    "language_consistency": 500,
}
_PARTITIONS = {
    "triviaqa": "T-test",
    "simpleqa_verified": "simpleqa-eval",
    "aa_omniscience_public_600": "aa-eval",
    **{
        name: "side-effect-eval"
        for name in (
            "ifeval",
            "mmlu_pro",
            "wikitext103",
            "xstest",
            "strongreject_or_harmbench",
            "language_consistency",
        )
    },
}
_FREEZE_FIELDS = {
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
}
_SCALAR_FIELDS = {
    "model_revision",
    "prompt",
    "risk_threshold",
    "layer",
    "alpha_policy",
    "abstention_rule",
}
_E10_RELEASE_EPSILON = 0.01
_E10_PROMPT_SELECTION_RULE = (
    "minimum-E8-M3-T-dev-hallucination-risk-then-maximum-coverage-then-prompt-id"
)


@dataclass(frozen=True, slots=True)
class E10FreezeInputs:
    directory: Path
    paths: Mapping[str, Path]

    def __post_init__(self) -> None:
        if set(self.paths) != _FREEZE_FIELDS:
            raise DataValidationError("E10 freeze-input set differs from the protocol")
        object.__setattr__(self, "paths", MappingProxyType(dict(self.paths)))


@dataclass(frozen=True, slots=True)
class E10ExecutionAssets:
    ledger: PhaseRunLedger
    prompt: PromptSpec
    questions: Mapping[str, Question]
    component_artifact: Path

    def __post_init__(self) -> None:
        if self.ledger.contract.phase is not ExperimentPhase.E10:
            raise DataValidationError("E10 execution assets require an E10 ledger")
        object.__setattr__(self, "questions", MappingProxyType(dict(self.questions)))


def _copy(source: Path, destination: Path) -> None:
    if (
        source.is_symlink()
        or not source.exists()
        or (source.is_dir() and any(item.is_symlink() for item in source.rglob("*")))
    ):
        raise DataValidationError("E10 freeze source must be a strict regular artifact")
    if source.is_dir():
        shutil.copytree(source, destination)
    elif source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    else:
        raise DataValidationError("E10 freeze source has an unsupported file type")


def _e6_runtime_artifact(ledger: PhaseRunLedger) -> Path:
    if ledger.verify_complete().phase is not ExperimentPhase.E6:
        raise FrozenArtifactError("E10 runtime source is not its complete E6 ledger")
    path = (
        ledger.directory
        / "gate-artifacts"
        / "knowledge_recovery_separated_from_abstention_substitution"
        / "likelihood-bundle"
        / "runtime-artifact"
    )
    _load_e6_runtime_attestation(path)
    return path


def _validate_exact_e6_runtime_binding(
    *,
    runtime_artifact: str | Path,
    grader: str | Path,
    intervention: InterventionSpec | None = None,
) -> str:
    from mfh.experiments.confirmatory_graders import (
        validate_confirmatory_grader_bundle,
    )

    runtime_path = Path(runtime_artifact)
    expected = _load_e6_runtime_attestation(runtime_path)
    bundle = validate_confirmatory_grader_bundle(grader)
    packaged_runtime = bundle.directory / "runtime-attestation.json"
    public_key = str(expected["execution_public_key"])
    if (
        sha256_path(packaged_runtime) != sha256_path(runtime_path)
        or dict(bundle.runtime_attestation) != expected
        or bundle.scorer.execution_public_key != public_key
        or (
            intervention is not None
            and (
                intervention.adaptive_policy is None
                or intervention.adaptive_policy.execution_public_key != public_key
            )
        )
    ):
        raise FrozenArtifactError("E10 execution runtime differs from exact E6")
    return public_key


def _scalar_bodies(
    *,
    model: ModelSpec,
    prompt: PromptSpec,
    policy: CompositePolicy,
) -> dict[str, dict[str, object]]:
    controller = policy.controller
    layers = (
        (controller.fixed_layer,)
        if controller.fixed_layer is not None
        else controller.layer_selector.candidate_layers
        if controller.layer_selector is not None
        else ()
    )
    return {
        "model_revision": {
            "schema_version": 1,
            "repository": model.repository,
            "revision": model.revision,
            "runtime": model.runtime.value,
            "quantization": model.quantization,
            "num_layers": model.num_layers,
        },
        "prompt": {
            "schema_version": 1,
            "prompt_id": prompt.prompt_id,
            "text": prompt.text,
            "text_sha256": hashlib.sha256(prompt.text.encode("utf-8")).hexdigest(),
            "permits_abstention": prompt.permits_abstention,
            "deployment_eligible": prompt.deployment_eligible,
        },
        "risk_threshold": {
            "schema_version": 1,
            "tau_low": policy.config.tau_low,
            "tau_high": policy.config.tau_high,
            "release_epsilon": policy.config.release_epsilon,
        },
        "layer": {
            "schema_version": 1,
            "candidate_layers": list(layers),
            "candidate_sites": sorted(
                {key.site.value for key in controller.vector_bank.directions}
            ),
            "token_scope": policy.config.token_scope.value,
        },
        "alpha_policy": {
            "schema_version": 1,
            "mode": controller.alpha_controller.mode.value,
            "alpha_max": controller.alpha_controller.alpha_max,
            "beta": controller.alpha_controller.beta,
            "threshold": controller.alpha_controller.threshold,
            "minimum_necessary": True,
        },
        "abstention_rule": {
            "schema_version": 1,
            "phrase": policy.config.abstention_phrase,
            "closed_book": policy.config.closed_book,
            "abstain_at_or_above": policy.config.tau_high,
            "release_at_or_below": policy.config.release_epsilon,
        },
    }


def _write_scalar(path: Path, body: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(
            {**body, "artifact_digest": stable_hash(body)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _strict_json(path: Path, context: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FrozenArtifactError(f"{context} must be one regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read {context}: {exc}") from exc
    if not isinstance(value, dict):
        raise FrozenArtifactError(f"{context} must contain a JSON object")
    return value


def write_e10_freeze_inputs(
    directory: str | Path,
    *,
    model: ModelSpec,
    prompt: PromptSpec,
    composite_policy_artifact: str | Path,
    sae_checkpoint: str | Path,
    grader: str | Path,
    runtime_artifact: str | Path,
    evaluation_scripts: str | Path,
    study_protocol_digest: str,
) -> E10FreezeInputs:
    """Materialize all eleven recursively frozen E10 parameter/code inputs."""

    normalized = validate_active_study_artifact_paths(
        {
            "E10 freeze inputs": directory,
            "E10 composite source": composite_policy_artifact,
            "E10 SAE checkpoint": sae_checkpoint,
            "E10 grader": grader,
            "E10 runtime attestation": runtime_artifact,
            "E10 evaluation scripts": evaluation_scripts,
        }
    )
    destination = normalized["E10 freeze inputs"]
    composite_policy_artifact = normalized["E10 composite source"]
    sae_checkpoint = normalized["E10 SAE checkpoint"]
    grader = normalized["E10 grader"]
    runtime_artifact = normalized["E10 runtime attestation"]
    evaluation_scripts = normalized["E10 evaluation scripts"]
    if destination.exists() or destination.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite E10 freeze inputs: {destination}")
    validate_active_model_spec(model)
    if not prompt.deployment_eligible or not prompt.permits_abstention:
        raise DataValidationError("E10 requires a deployment prompt that permits abstention")
    composite_source = Path(composite_policy_artifact).resolve()
    policy = load_composite_policy(composite_source)
    load_e10_composite_provenance(composite_source)
    if policy.protected_subspace is None:
        raise DataValidationError("E10 M6 must use a frozen protected direction transform")
    if sha256_path(sae_checkpoint) != sha256_path(composite_source / "sae_checkpoint"):
        raise DataValidationError("E10 SAE freeze differs from the selected M6 source")
    source_identity = policy.controller.risk_probe.training_schema.source_identity()
    if (
        source_identity["model_repository"] != model.repository
        or source_identity["model_revision"] != model.revision
        or source_identity["runtime"] != model.runtime.value
        or source_identity["quantization"] != model.quantization
        or source_identity["prompt_id"] != prompt.prompt_id
        or source_identity["prompt_sha256"]
        != hashlib.sha256(prompt.text.encode("utf-8")).hexdigest()
    ):
        raise DataValidationError("E10 composite policy differs from model or prompt")
    execution_public_key = _validate_exact_e6_runtime_binding(
        runtime_artifact=runtime_artifact,
        grader=grader,
    )
    validate_execution_snapshot(
        evaluation_scripts,
        study_protocol_digest=study_protocol_digest,
        phase=ExperimentPhase.E10,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        bodies = _scalar_bodies(model=model, prompt=prompt, policy=policy)
        for name, body in bodies.items():
            _write_scalar(stage / f"{name}.json", body)
        _copy(
            composite_source / "controller" / "vector_bank",
            stage / "vector_bank",
        )
        _copy(Path(sae_checkpoint).resolve(), stage / "sae_checkpoint")
        _copy(composite_source / "protected_subspace", stage / "protected_subspace")
        _copy(Path(grader).resolve(), stage / "grader")
        _copy(Path(evaluation_scripts).resolve(), stage / "evaluation_scripts")
        observed_paths = {
            name: (stage / f"{name}.json" if name in _SCALAR_FIELDS else stage / name)
            for name in _FREEZE_FIELDS
        }
        manifest_body = {
            "schema_version": 1,
            "study_protocol_digest": study_protocol_digest,
            "phase": ExperimentPhase.E10.value,
            "composite_policy_sha256": sha256_path(composite_source),
            "freeze_fields": {
                name: sha256_path(path) for name, path in sorted(observed_paths.items())
            },
        }
        _write_scalar(stage / "freeze-manifest.json", manifest_body)
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    paths = {
        name: (destination / f"{name}.json" if name in _SCALAR_FIELDS else destination / name)
        for name in _FREEZE_FIELDS
    }
    validate_e10_freeze_inputs(
        paths,
        model=model,
        prompt=prompt,
        composite_policy_artifact=composite_source,
        study_protocol_digest=study_protocol_digest,
        expected_runtime_artifact=runtime_artifact,
        expected_execution_public_key=execution_public_key,
    )
    return E10FreezeInputs(destination, paths)


def validate_e10_freeze_inputs(
    artifacts: Mapping[str, str | Path],
    *,
    model: ModelSpec,
    prompt: PromptSpec,
    composite_policy_artifact: str | Path,
    study_protocol_digest: str,
    expected_runtime_artifact: str | Path,
    expected_execution_public_key: str,
) -> Mapping[str, str]:
    """Recompute the semantic identity of every E10 freeze field."""

    if set(artifacts) != _FREEZE_FIELDS:
        raise DataValidationError("E10 freeze artifacts differ from the eleven fields")
    policy_path = Path(composite_policy_artifact)
    policy = load_composite_policy(policy_path)
    load_e10_composite_provenance(policy_path)
    if policy.protected_subspace is None:
        raise DataValidationError("E10 composite policy lacks protected steering")
    expected_bodies = _scalar_bodies(model=model, prompt=prompt, policy=policy)
    for name, expected in expected_bodies.items():
        path = Path(artifacts[name])
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FrozenArtifactError(f"cannot read E10 {name} freeze: {exc}") from exc
        if not isinstance(value, dict):
            raise FrozenArtifactError(f"E10 {name} freeze is invalid")
        digest = value.pop("artifact_digest", None)
        if value != expected or digest != stable_hash(expected):
            raise FrozenArtifactError(f"E10 {name} freeze differs from M6")
    if (
        sha256_path(artifacts["vector_bank"])
        != sha256_path(policy_path / "controller" / "vector_bank")
        or sha256_path(artifacts["protected_subspace"])
        != sha256_path(policy_path / "protected_subspace")
        or sha256_path(artifacts["sae_checkpoint"]) != sha256_path(policy_path / "sae_checkpoint")
    ):
        raise FrozenArtifactError("E10 direction components differ from M6")
    observed_public_key = _validate_exact_e6_runtime_binding(
        runtime_artifact=expected_runtime_artifact,
        grader=artifacts["grader"],
    )
    if observed_public_key != expected_execution_public_key:
        raise FrozenArtifactError("E10 execution public key differs from exact E6")
    validate_execution_snapshot(
        artifacts["evaluation_scripts"],
        study_protocol_digest=study_protocol_digest,
        phase=ExperimentPhase.E10,
    )
    return MappingProxyType({name: sha256_path(path) for name, path in sorted(artifacts.items())})


def e10_intervention(
    *,
    component_artifact: str | Path,
    study: StudyProtocol,
    e6_run: str | Path,
) -> InterventionSpec:
    """Derive the exact ledger-replayable M6 policy from its composite artifact."""

    normalized = validate_active_study_artifact_paths(
        {"E10 component": component_artifact, "E6 phase ledger": e6_run}
    )
    source = normalized["E10 component"]
    e6 = PhaseRunLedger.open(normalized["E6 phase ledger"], study=study)
    runtime_artifact = _e6_runtime_artifact(e6)
    runtime_attestation = _load_e6_runtime_attestation(runtime_artifact)
    execution_public_key = str(runtime_attestation["execution_public_key"])
    policy = load_composite_policy(source)
    load_e10_composite_provenance(source)
    if policy.protected_subspace is None:
        raise DataValidationError("E10 M6 requires protected routed directions")
    controller = policy.controller
    candidate_layers = (
        (controller.fixed_layer,)
        if controller.fixed_layer is not None
        else controller.layer_selector.candidate_layers
        if controller.layer_selector is not None
        else ()
    )
    candidate_sites = tuple(
        sorted(
            {key.site for key in controller.vector_bank.directions},
            key=lambda value: value.value,
        )
    )
    adaptive = AdaptivePolicySpec(
        schema_version=2,
        release_risk_threshold=policy.config.tau_low,
        abstention_probability_threshold=policy.config.tau_high,
        alpha_max=controller.alpha_controller.alpha_max,
        alpha_beta=controller.alpha_controller.beta,
        layer=None,
        site=None,
        token_scope=None,
        direction_sha256=None,
        direction_norm=None,
        execution_public_key=execution_public_key,
        controller_artifact_sha256=sha256_path(source),
        candidate_layers=tuple(int(value) for value in candidate_layers),
        candidate_sites=candidate_sites,
        candidate_token_scopes=(policy.config.token_scope,),
        vector_count=controller.vector_bank.cluster_count,
        likely_unknown_risk_threshold=policy.config.tau_high,
        alpha_mode=controller.alpha_controller.mode.value,
        alpha_risk_threshold=controller.alpha_controller.threshold,
    )
    return InterventionSpec(
        method="M6",
        artifact_sha256=sha256_path(source),
        adaptive_policy=adaptive,
    )


def build_e10_contract(
    *,
    study: StudyProtocol,
    model: ModelSpec,
    prompt: PromptSpec,
    questions_by_benchmark: Mapping[str, Sequence[Question]],
    intervention: InterventionSpec,
    input_fingerprints: Mapping[str, str],
    prerequisite_digests: Mapping[str, str],
    seed: int = 17,
) -> PhaseRunContract:
    """Build the only accepted 10,204-row one-shot M6 schedule."""

    validate_active_model_spec(model)
    phase = study.phase(ExperimentPhase.E10)
    if (
        not prompt.deployment_eligible
        or not prompt.permits_abstention
        or intervention.method != "M6"
        or intervention.adaptive_policy is None
        or set(questions_by_benchmark) != set(_COUNTS)
        or set(input_fingerprints) != set(phase.required_inputs) | _FREEZE_FIELDS
        or set(prerequisite_digests) != {value.value for value in phase.prerequisites}
    ):
        raise DataValidationError("E10 inputs differ from the frozen one-shot protocol")
    question_ids: dict[str, tuple[str, ...]] = {}
    seen: set[str] = set()
    for benchmark, count in _COUNTS.items():
        questions = tuple(questions_by_benchmark[benchmark])
        identifiers = tuple(value.question_id for value in questions)
        if (
            len(questions) != count
            or any(value.benchmark != benchmark for value in questions)
            or len(set(identifiers)) != count
            or seen.intersection(identifiers)
        ):
            raise DataValidationError(f"E10 {benchmark} question schedule differs")
        seen.update(identifiers)
        question_ids[benchmark] = identifiers
    prompt_sha = hashlib.sha256(prompt.text.encode("utf-8")).hexdigest()
    conditions = tuple(
        EvaluationCondition(
            phase=ExperimentPhase.E10,
            benchmark=benchmark,
            partition=_PARTITIONS[benchmark],
            model_name=model.name,
            model_repository=model.repository,
            model_revision=model.revision,
            runtime=model.runtime,
            quantization=model.quantization,
            model_num_layers=model.num_layers,
            system_prompt_id=prompt.prompt_id,
            prompt_template_sha256=prompt_sha,
            steering_method="M6",
            method_artifact_sha256=intervention.artifact_sha256,
            layer=None,
            site=None,
            token_scope=None,
            alpha=0.0,
            sparsity=None,
            seed=seed,
            study_protocol_digest=study.digest,
            adaptive_policy=intervention.adaptive_policy,
        )
        for benchmark in phase.benchmarks
    )
    contract = PhaseRunContract(
        phase=ExperimentPhase.E10,
        study_protocol_digest=study.digest,
        conditions=conditions,
        question_ids_by_benchmark=question_ids,
        input_fingerprints=input_fingerprints,
        prerequisite_digests=prerequisite_digests,
        required_gates=phase.gates,
    )
    contract.assert_matches_study(study)
    if len(conditions) != 9 or contract.expected_record_count != 10_204:
        raise DataValidationError("E10 one-shot cardinality differs from the plan")
    return contract


def _validate_e10_triviaqa_source(
    *,
    e1: PhaseRunLedger,
    supplied: Sequence[Question] | None,
    frozen_question_bundle: str | Path,
) -> None:
    from mfh.data.io import read_questions

    trusted = _reviewed_questions_from_e1(
        e1,
        split_manifest_digest=None,
        partitions=("T-test",),
    )["T-test"]
    supplied_values = None if supplied is None else tuple(supplied)
    packaged = tuple(read_questions(Path(frozen_question_bundle) / "triviaqa.jsonl"))
    trusted_fingerprints = tuple(question_source_fingerprint(value) for value in trusted)
    if (
        len(trusted) != _COUNTS["triviaqa"]
        or tuple(question_source_fingerprint(value) for value in packaged) != trusted_fingerprints
        or (
            supplied_values is not None
            and tuple(question_source_fingerprint(value) for value in supplied_values)
            != trusted_fingerprints
        )
    ):
        raise FrozenArtifactError(
            "E10 TriviaQA schedule differs from the exact E1 reviewed T-test split"
        )


def validate_e10_prerequisite_bound_inputs(
    input_artifacts: Mapping[str, str | Path],
    *,
    contract: PhaseRunContract,
    prompts: Mapping[str, PromptSpec],
    prerequisite_ledgers: Mapping[ExperimentPhase, Any],
) -> None:
    """Bind every public E10 ledger path to exact E1/E6/E9 evidence.

    This validator is deliberately callable from the generic phase-ledger
    factory and its replay path.  The specialized E10 factory is therefore a
    convenience API, not a way to bypass the scientific provenance boundary.
    """

    if contract.phase is not ExperimentPhase.E10:
        raise DataValidationError("E10 prerequisite validation received another phase")
    required = set(ExperimentPhase) - {ExperimentPhase.E10}
    if set(prerequisite_ledgers) != required:
        raise DataValidationError("E10 provenance requires the exact E0-E9 ledgers")
    required_inputs = {
        "E9_results",
        "frozen_question_bundle",
        "component_selection_manifest",
        "grader",
    }
    if not required_inputs <= set(input_artifacts):
        raise DataValidationError("E10 prerequisite-bound inputs are incomplete")

    e1 = prerequisite_ledgers[ExperimentPhase.E1]
    e6 = prerequisite_ledgers[ExperimentPhase.E6]
    e9 = prerequisite_ledgers[ExperimentPhase.E9]
    _validate_e10_triviaqa_source(
        e1=e1,
        supplied=None,
        frozen_question_bundle=input_artifacts["frozen_question_bundle"],
    )
    if sha256_path(input_artifacts["E9_results"]) != sha256_path(e9.directory):
        raise FrozenArtifactError("E10 E9_results is not its exact E9 prerequisite")

    runtime_artifact = _e6_runtime_artifact(e6)
    public_key = _validate_exact_e6_runtime_binding(
        runtime_artifact=runtime_artifact,
        grader=input_artifacts["grader"],
    )
    adaptive_keys = {
        condition.adaptive_policy.execution_public_key
        for condition in contract.conditions
        if condition.adaptive_policy is not None
    }
    if adaptive_keys != {public_key} or any(
        condition.adaptive_policy is None for condition in contract.conditions
    ):
        raise FrozenArtifactError(
            "E10 contract adaptive execution key differs from the exact E6 runtime"
        )

    validate_e10_component_selection_promotion(
        input_artifacts["component_selection_manifest"],
        contract=contract,
        prompts=prompts,
        prerequisite_ledgers=prerequisite_ledgers,
    )


def create_e10_ledger(
    directory: str | Path,
    *,
    study: StudyProtocol,
    model_config: str | Path,
    prompt_config: str | Path,
    questions_by_benchmark: Mapping[str, Sequence[Question]],
    intervention: InterventionSpec,
    input_artifacts: Mapping[str, str | Path],
    prerequisite_runs: Mapping[str, str | Path],
    seed: int = 17,
) -> PhaseRunLedger:
    """Run every preflight check before atomically reserving the E10 one-shot run."""

    model = load_model_spec(model_config)
    phase = study.phase(ExperimentPhase.E10)
    if set(input_artifacts) != set(phase.required_inputs) | _FREEZE_FIELDS:
        raise DataValidationError("E10 input artifacts differ from the protocol")
    if set(prerequisite_runs) != {value.value for value in phase.prerequisites}:
        raise DataValidationError("E10 prerequisite paths differ from the protocol")
    prerequisite_digests: dict[str, str] = {}
    prerequisite_ledgers: dict[ExperimentPhase, Any] = {}
    for name, path in prerequisite_runs.items():
        phase_name = ExperimentPhase(name)
        prerequisite_ledger = open_phase_prerequisite(
            path,
            phase=phase_name,
            study=study,
        )
        completion = prerequisite_ledger.verify_complete()
        if completion.phase.value != name:
            raise DataValidationError(f"E10 prerequisite {name} resolves to another phase")
        prerequisite_digests[name] = completion.completion_digest
        prerequisite_ledgers[completion.phase] = prerequisite_ledger
    e1 = prerequisite_ledgers[ExperimentPhase.E1]
    e6 = prerequisite_ledgers[ExperimentPhase.E6]
    runtime_artifact = _e6_runtime_artifact(e6)
    execution_public_key = _validate_exact_e6_runtime_binding(
        runtime_artifact=runtime_artifact,
        grader=input_artifacts["grader"],
        intervention=intervention,
    )
    _validate_e10_triviaqa_source(
        e1=e1,
        supplied=questions_by_benchmark["triviaqa"],
        frozen_question_bundle=input_artifacts["frozen_question_bundle"],
    )
    provenance = _derive_e10_composite_provenance(
        prerequisite_ledgers=prerequisite_ledgers,
    )
    selected_prompt_id = str(provenance["selected_prompt_id"])
    available_prompts = {value.prompt_id: value for value in load_prompt_specs(prompt_config)}
    try:
        prompt = available_prompts[selected_prompt_id]
    except KeyError as exc:
        raise DataValidationError("E10 independently selected prompt is unavailable") from exc
    e9_input = Path(input_artifacts["E9_results"]).resolve()
    if e9_input != Path(prerequisite_runs["E9"]).resolve():
        raise DataValidationError("E10 E9_results must be its verified E9 prerequisite")
    fingerprints = {name: sha256_path(path) for name, path in input_artifacts.items()}
    contract = build_e10_contract(
        study=study,
        model=model,
        prompt=prompt,
        questions_by_benchmark=questions_by_benchmark,
        intervention=intervention,
        input_fingerprints=fingerprints,
        prerequisite_digests=prerequisite_digests,
        seed=seed,
    )
    component_artifact = _component_from_selection(
        Path(input_artifacts["component_selection_manifest"]),
        contract,
    )
    _validate_e10_component_promotion(
        component_artifact,
        prompt=prompt,
        prerequisite_ledgers=prerequisite_ledgers,
    )
    freeze_paths = {name: input_artifacts[name] for name in _FREEZE_FIELDS}
    validate_e10_freeze_inputs(
        freeze_paths,
        model=model,
        prompt=prompt,
        composite_policy_artifact=component_artifact,
        study_protocol_digest=study.digest,
        expected_runtime_artifact=runtime_artifact,
        expected_execution_public_key=execution_public_key,
    )
    return PhaseRunLedger.create(
        directory,
        contract,
        study=study,
        input_artifacts=input_artifacts,
        prerequisite_runs=prerequisite_runs,
        confirmatory_prompts={prompt.prompt_id: prompt},
    )


def _component_from_selection(path: Path, contract: PhaseRunContract) -> Path:
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        descriptors = manifest["components"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise FrozenArtifactError(f"cannot locate E10 M6 component: {exc}") from exc
    if not isinstance(descriptors, list) or len(descriptors) != 1:
        raise FrozenArtifactError("E10 component selection must contain exactly one M6")
    descriptor = descriptors[0]
    if not isinstance(descriptor, Mapping):
        raise FrozenArtifactError("E10 M6 component descriptor is invalid")
    artifact = path / str(descriptor.get("component_path")) / "artifact"
    expected = contract.conditions[0].method_artifact_sha256
    if sha256_path(artifact) != expected:
        raise FrozenArtifactError("E10 M6 component differs from its contract")
    return artifact


def _derive_e10_composite_provenance(
    *,
    prerequisite_ledgers: Mapping[ExperimentPhase, Any],
) -> Mapping[str, object]:
    """Derive M6 source provenance from already verified E0--E9 ledgers."""

    required = set(ExperimentPhase) - {ExperimentPhase.E10}
    if set(prerequisite_ledgers) != required:
        raise DataValidationError("E10 promotion validation lacks exact E0-E9 ledgers")
    completion_digests = {
        phase.value: ledger.verify_complete().completion_digest
        for phase, ledger in prerequisite_ledgers.items()
    }
    e7 = prerequisite_ledgers[ExperimentPhase.E7]
    e8 = prerequisite_ledgers[ExperimentPhase.E8]
    e9 = prerequisite_ledgers[ExperimentPhase.E9]
    registry_path = (
        e8.directory
        / "gate-artifacts"
        / "matched_empirical_risk_or_coverage"
        / "operating-point-registry"
    )
    from mfh.experiments.confirmatory_components import (
        load_confirmatory_adaptive_component,
    )
    from mfh.methods.protected import load_e8_operating_point_registry

    registry = load_e8_operating_point_registry(registry_path)
    e8_conditions = {value.condition_id: value for value in e8.contract.conditions}
    records_by_condition: dict[str, list[GenerationRecord]] = {}
    for record in e8.records():
        records_by_condition.setdefault(record.condition_id, []).append(record)
    prompt_metrics: dict[str, dict[str, float | str]] = {}
    for candidate_prompt in ("P0-neutral", "P2-calibrated-abstention"):
        try:
            condition_id = registry.condition_ids_by_prompt[candidate_prompt]["M3"]
            condition = e8_conditions[condition_id]
            records = records_by_condition[condition_id]
        except KeyError as exc:
            raise FrozenArtifactError(
                "E10 prompt selection lacks an E8 M3 development point"
            ) from exc
        metrics = metric_bundle(record.outcome for record in records)
        if (
            condition.system_prompt_id != candidate_prompt
            or condition.steering_method != "M3"
            or condition.benchmark != "triviaqa"
            or condition.partition != "T-dev"
            or metrics.hallucination_risk is None
            or metrics.coverage is None
        ):
            raise FrozenArtifactError(
                "E10 prompt selection point is not an eligible E8 M3 T-dev condition"
            )
        prompt_metrics[candidate_prompt] = {
            "condition_id": condition_id,
            "hallucination_risk": metrics.hallucination_risk,
            "coverage": metrics.coverage,
        }
    prompt_id = min(
        prompt_metrics,
        key=lambda value: (
            float(prompt_metrics[value]["hallucination_risk"]),
            -float(prompt_metrics[value]["coverage"]),
            value,
        ),
    )
    try:
        selected_ids = {
            method: registry.condition_ids_by_prompt[prompt_id][method] for method in ("M3", "M5")
        }
        m3_condition = e8_conditions[selected_ids["M3"]]
        m5_condition = e8_conditions[selected_ids["M5"]]
    except KeyError as exc:
        raise FrozenArtifactError("E10 prompt lacks selected E8 M3/M5 winners") from exc
    selection = e9.directory / "inputs" / "frozen_component_selection"
    descriptors = _validate_component_selection(selection, e9.contract)
    try:
        descriptor = next(value for key, value in descriptors.items() if key[1] == "M3")
    except StopIteration as exc:
        raise FrozenArtifactError("E10 cannot locate the promoted E9 M3 component") from exc
    adaptive_path = selection / str(descriptor["component_path"]) / "artifact"
    adaptive = load_confirmatory_adaptive_component(adaptive_path)
    try:
        controller_sha = adaptive.controller_fingerprints[prompt_id]
    except KeyError as exc:
        raise FrozenArtifactError("E10 selected prompt lacks an E9 M3 controller") from exc
    e7_sae = e7.directory / "gate-artifacts" / "held_out_reconstruction" / "sae-intervention"
    selected_m3_sha = (
        m3_condition.adaptive_policy.controller_artifact_sha256
        if m3_condition.adaptive_policy is not None
        else None
    )
    if selected_m3_sha != controller_sha:
        raise FrozenArtifactError("E10 E8 and E9 adaptive-controller promotions disagree")
    adaptive_policy = m3_condition.adaptive_policy
    if (
        adaptive_policy is None
        or adaptive_policy.likely_unknown_risk_threshold is None
        or len(adaptive_policy.candidate_token_scopes) != 1
    ):
        raise FrozenArtifactError("E10 selected M3 policy lacks exact routed thresholds")
    expected: dict[str, object] = {
        "schema_version": 1,
        "selected_prompt_id": prompt_id,
        "prompt_selection_rule": _E10_PROMPT_SELECTION_RULE,
        "prompt_selection_metrics": prompt_metrics,
        "prerequisite_completion_digests": completion_digests,
        "e8_registry_sha256": sha256_path(registry_path),
        "selected_e8_condition_ids": selected_ids,
        "e9_adaptive_component_sha256": adaptive.fingerprint,
        "e9_selected_controller_sha256": controller_sha,
        "e7_sae_checkpoint_sha256": sha256_path(e7_sae),
        "protected_source_artifact_sha256": m5_condition.method_artifact_sha256,
        "selected_policy": {
            "tau_low": adaptive_policy.release_risk_threshold,
            "tau_high": adaptive_policy.likely_unknown_risk_threshold,
            "release_epsilon": _E10_RELEASE_EPSILON,
            "token_scope": adaptive_policy.candidate_token_scopes[0].value,
        },
    }
    return MappingProxyType(expected)


def derive_e10_composite_provenance(
    *,
    study: StudyProtocol,
    prerequisite_runs: Mapping[str, str | Path],
) -> Mapping[str, object]:
    """Replay E0--E9 and derive the exact provenance passed to M6 freezing."""

    required = {phase.value for phase in ExperimentPhase if phase is not ExperimentPhase.E10}
    if set(prerequisite_runs) != required:
        raise DataValidationError("E10 provenance derivation requires exact E0-E9 runs")
    ledgers: dict[ExperimentPhase, Any] = {}
    for name, path in prerequisite_runs.items():
        phase = ExperimentPhase(name)
        ledger = open_phase_prerequisite(path, phase=phase, study=study)
        completion = ledger.verify_complete()
        if completion.phase is not phase:
            raise DataValidationError(
                f"E10 provenance prerequisite {name} resolves to another phase"
            )
        ledgers[phase] = ledger
    return _derive_e10_composite_provenance(
        prerequisite_ledgers=ledgers,
    )


def write_e10_composite_from_promotions(
    directory: str | Path,
    *,
    study: StudyProtocol,
    prerequisite_runs: Mapping[str, str | Path],
    controller_artifact: str | Path,
    early_probe_selection: str | Path,
    sae_checkpoint: str | Path,
    protected_source_artifact: str | Path,
) -> Path:
    """Assemble M6 only from replayed E0-E9 promotions and the dev-selected probe."""

    normalized = validate_active_study_artifact_paths(
        {
            "E10 composite": directory,
            "E10 controller": controller_artifact,
            "E10 early-probe selection": early_probe_selection,
            "E10 SAE checkpoint": sae_checkpoint,
            "E10 protected source": protected_source_artifact,
            **{f"E10 prerequisite {name}": path for name, path in prerequisite_runs.items()},
        }
    )
    provenance = dict(
        derive_e10_composite_provenance(
            study=study,
            prerequisite_runs={
                name: normalized[f"E10 prerequisite {name}"] for name in prerequisite_runs
            },
        )
    )
    controller_path = normalized["E10 controller"]
    early_path = normalized["E10 early-probe selection"]
    sae_path = normalized["E10 SAE checkpoint"]
    protected_path = normalized["E10 protected source"]
    from mfh.experiments.e8_protected import load_e8_protected_artifact
    from mfh.methods.adaptive import load_adaptive_controller

    controller = load_adaptive_controller(controller_path)
    early = load_e10_early_probe_selection(early_path)
    protected = load_e8_protected_artifact(protected_path)
    selected_policy = provenance.get("selected_policy")
    source_identities = early.manifest.get("capture_plan_identity")
    early_capture = early.manifest.get("capture_sha256")
    if (
        not isinstance(selected_policy, Mapping)
        or sha256_path(controller_path) != provenance.get("e9_selected_controller_sha256")
        or sha256_path(sae_path) != provenance.get("e7_sae_checkpoint_sha256")
        or sha256_path(protected_path) != provenance.get("protected_source_artifact_sha256")
        or early.selected_probe.training_schema.source_identity()
        != controller.risk_probe.training_schema.source_identity()
        or not isinstance(source_identities, str)
        or not isinstance(early_capture, str)
    ):
        raise FrozenArtifactError("E10 composite sources differ from their promotions")
    try:
        config = CompositePolicyConfig(
            tau_low=float(selected_policy["tau_low"]),
            tau_high=float(selected_policy["tau_high"]),
            release_epsilon=float(selected_policy["release_epsilon"]),
            token_scope=TokenScope(str(selected_policy["token_scope"])),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FrozenArtifactError(f"E10 selected policy is invalid: {exc}") from exc
    policy = CompositePolicy(
        controller=controller,
        config=config,
        early_probe=early.selected_probe,
        protected_subspace=protected.protected_subspace,
    )
    provenance.update(
        {
            "early_probe_selection_sha256": sha256_path(early_path),
            "selected_early_probe_sha256": sha256_path(early.selected_probe_path),
            "early_probe_capture_sha256": early_capture,
            "early_probe_capture_plan_identity": source_identities,
            "early_probe_selection_rule": early.manifest["selection_rule"],
            "early_probe_selection_metrics": early.manifest["selected_metrics"],
        }
    )
    destination = normalized["E10 composite"]
    save_composite_policy(
        destination,
        policy,
        sae_checkpoint=sae_path,
        protected_source_artifact=protected_path,
        selection_provenance=provenance,
        early_probe_selection=early_path,
    )
    load_composite_policy(destination)
    return destination


def _validate_e10_component_promotion(
    component: Path,
    *,
    prompt: PromptSpec,
    prerequisite_ledgers: Mapping[ExperimentPhase, Any],
) -> None:
    """Prove M6 sources are the exact E7/E8/E9 independently frozen winners."""

    provenance = load_e10_composite_provenance(component)
    expected = _derive_e10_composite_provenance(
        prerequisite_ledgers=prerequisite_ledgers,
    )
    e7 = prerequisite_ledgers[ExperimentPhase.E7]
    e8 = prerequisite_ledgers[ExperimentPhase.E8]
    e1 = prerequisite_ledgers[ExperimentPhase.E1]
    e6 = prerequisite_ledgers[ExperimentPhase.E6]
    e8_conditions = {value.condition_id: value for value in e8.contract.conditions}
    selected_ids = expected["selected_e8_condition_ids"]
    if not isinstance(selected_ids, Mapping):  # pragma: no cover - built above
        raise FrozenArtifactError("E10 selected-condition provenance is invalid")
    m5_condition = e8_conditions[str(selected_ids["M5"])]
    e7_sae = e7.directory / "gate-artifacts" / "held_out_reconstruction" / "sae-intervention"
    policy = load_composite_policy(component)
    early_selection_path = component / "early_probe_selection"
    early_selection = load_e10_early_probe_selection(early_selection_path)
    early_capture_plan = _strict_json(
        early_selection_path / "capture" / "plan.json",
        "E10 early-probe capture plan",
    )
    early_sources = early_capture_plan.get("source_identities")
    selected_policy = expected.get("selected_policy")
    if not isinstance(selected_policy, Mapping) or not isinstance(
        early_sources, Mapping
    ):  # pragma: no cover - built above
        raise FrozenArtifactError("E10 selected policy provenance is invalid")
    completion_digests = expected.get("prerequisite_completion_digests")
    if not isinstance(completion_digests, Mapping):  # pragma: no cover - built above
        raise FrozenArtifactError("E10 completion provenance is invalid")
    split_manifest_digest = early_sources.get("split_manifest_digest")
    e6_runtime_artifact = (
        e6.directory
        / "gate-artifacts"
        / "knowledge_recovery_separated_from_abstention_substitution"
        / "likelihood-bundle"
        / "runtime-artifact"
    )
    if not isinstance(split_manifest_digest, str) or not e6_runtime_artifact.is_file():
        raise FrozenArtifactError("E10 early-probe immutable sources are incomplete")
    replayed_early_plan = derive_e10_early_probe_capture_plan(
        study=e1.study,
        e1_run=e1.directory,
        e8_run=e8.directory,
        prompt=prompt,
        controller_artifact=component / "controller",
        selection_provenance=expected,
        split_manifest_digest=split_manifest_digest,
        runtime_artifact=e6_runtime_artifact,
    )
    provenance_base = {key: provenance.get(key) for key in expected}
    expected_early = {
        "early_probe_selection_sha256": sha256_path(early_selection_path),
        "selected_early_probe_sha256": sha256_path(early_selection.selected_probe_path),
        "early_probe_capture_sha256": early_selection.manifest["capture_sha256"],
        "early_probe_capture_plan_identity": early_selection.manifest["capture_plan_identity"],
        "early_probe_selection_rule": early_selection.manifest["selection_rule"],
        "early_probe_selection_metrics": early_selection.manifest["selected_metrics"],
    }
    observed_early = {key: provenance.get(key) for key in expected_early}
    if (
        set(provenance) != set(expected) | set(expected_early)
        or provenance_base != dict(expected)
        or observed_early != expected_early
        or prompt.prompt_id != expected["selected_prompt_id"]
        or policy.config.tau_low != selected_policy["tau_low"]
        or policy.config.tau_high != selected_policy["tau_high"]
        or policy.config.release_epsilon != selected_policy["release_epsilon"]
        or policy.config.token_scope.value != selected_policy["token_scope"]
        or sha256_path(component / "controller") != expected["e9_selected_controller_sha256"]
        or sha256_path(component / "sae_checkpoint") != sha256_path(e7_sae)
        or sha256_path(component / "protected_source_artifact")
        != m5_condition.method_artifact_sha256
        or sha256_path(component / "early_probe")
        != sha256_path(early_selection.selected_probe_path)
        or early_sources.get("e1_completion_digest") != completion_digests["E1"]
        or early_sources.get("e8_completion_digest") != completion_digests["E8"]
        or early_sources.get("e8_condition_id") != selected_ids["M3"]
        or early_sources.get("controller_sha256") != expected["e9_selected_controller_sha256"]
        or early_sources.get("prompt_id") != expected["selected_prompt_id"]
        or early_sources.get("selection_provenance_digest") != stable_hash(dict(expected))
        or dict(replayed_early_plan) != early_capture_plan
    ):
        raise FrozenArtifactError(
            "E10 composite is not the exact independently selected E7/E8/E9 system"
        )


def validate_e10_component_selection_promotion(
    selection: str | Path,
    *,
    contract: PhaseRunContract,
    prompts: Mapping[str, PromptSpec],
    prerequisite_ledgers: Mapping[ExperimentPhase, Any],
) -> None:
    """Replay the public/generic ledger boundary for an exact promoted M6 input."""

    if contract.phase is not ExperimentPhase.E10 or len(prompts) != 1:
        raise DataValidationError("E10 promotion requires one exact confirmatory prompt")
    prompt = next(iter(prompts.values()))
    component = _component_from_selection(Path(selection), contract)
    _validate_e10_component_promotion(
        component,
        prompt=prompt,
        prerequisite_ledgers=prerequisite_ledgers,
    )


def load_e10_execution_assets(
    run_directory: str | Path,
    *,
    study: StudyProtocol,
) -> E10ExecutionAssets:
    """Reopen E10 using only its recursively packaged prompt/questions/component."""

    from mfh.data.io import read_questions

    ledger = PhaseRunLedger.open(run_directory, study=study)
    if ledger.contract.phase is not ExperimentPhase.E10:
        raise DataValidationError("E10 execution received a different ledger")
    prompts = ledger.confirmatory_prompts()
    if len(prompts) != 1:
        raise FrozenArtifactError("E10 packaged prompt selection is not singular")
    prompt = next(iter(prompts.values()))
    questions: dict[str, Question] = {}
    root = ledger.directory / "inputs" / "frozen_question_bundle"
    for benchmark, identifiers in ledger.contract.question_ids_by_benchmark.items():
        values = tuple(read_questions(root / f"{benchmark}.jsonl"))
        if tuple(value.question_id for value in values) != identifiers:
            raise FrozenArtifactError("E10 packaged question order changed")
        for value in values:
            if value.question_id in questions:
                raise FrozenArtifactError("E10 packaged question identifiers repeat")
            questions[value.question_id] = value
    component = _component_from_selection(
        ledger.directory / "inputs" / "component_selection_manifest",
        ledger.contract,
    )
    return E10ExecutionAssets(ledger, prompt, questions, component)


def execute_e10_pending(
    assets: E10ExecutionAssets,
    backend: NativeE10VllmBackend,
    *,
    checkpoint_size: int = 1,
    limit: int | None = None,
) -> int:
    """Execute the one-shot schedule in deterministic resumable shard order."""

    if type(backend) is not NativeE10VllmBackend:
        raise DataValidationError("E10 execution requires the exact native VLLM backend")
    if checkpoint_size <= 0 or (limit is not None and limit <= 0):
        raise DataValidationError("E10 checkpoint size and limit must be positive")
    packaged_grader = assets.ledger.directory / "inputs" / "grader"
    execution_keys = {
        condition.adaptive_policy.execution_public_key
        for condition in assets.ledger.contract.conditions
        if condition.adaptive_policy is not None
    }
    if (
        sha256_path(backend.grader_bundle.directory) != sha256_path(packaged_grader)
        or sha256_path(backend.runtime_artifact)
        != sha256_path(packaged_grader / "runtime-attestation.json")
        or execution_keys != {backend.attestor.execution_public_key}
        or backend.grader_bundle.scorer.execution_public_key
        != backend.attestor.execution_public_key
    ):
        raise FrozenArtifactError("E10 backend differs from its ledger-packaged exact E6 runtime")
    completed = 0
    batch: list[GenerationRecord] = []
    for pending in assets.ledger.iter_pending():
        if limit is not None and completed >= limit:
            break
        record = backend.execute(
            condition=pending.condition,
            question=assets.questions[pending.question_id],
            prompt=assets.prompt,
            component_artifact=assets.component_artifact,
        )
        batch.append(record)
        completed += 1
        if len(batch) == checkpoint_size:
            assets.ledger.checkpoint(batch)
            batch.clear()
    if batch:
        assets.ledger.checkpoint(batch)
    return completed


def finalize_e10(
    assets: E10ExecutionAssets,
    *,
    evidence_directory: str | Path,
) -> PhaseCompletion | PhaseFalsification:
    """Atomically derive all gates and freeze either success or falsification."""

    completed, expected = assets.ledger.progress()
    if completed != expected:
        raise DataValidationError(f"E10 still has {expected - completed} pending rows")
    if (assets.ledger.directory / "complete.json").is_file():
        return assets.ledger.verify_complete()
    if (assets.ledger.directory / "falsified.json").is_file():
        return assets.ledger.verify_falsified()
    normalized = validate_active_study_artifact_paths(
        {
            "E10 ledger": assets.ledger.directory,
            "E10 evidence": evidence_directory,
        }
    )
    evidence_root = normalized["E10 evidence"]
    if evidence_root.resolve().is_relative_to(assets.ledger.directory.resolve()):
        raise DataValidationError("E10 external evidence cannot be inside its ledger")
    expected_files = {f"{gate}.json" for gate in assets.ledger.contract.required_gates}

    def evaluate(root: Path) -> dict[str, GateResult]:
        if (
            root.is_symlink()
            or not root.is_dir()
            or {item.name for item in root.iterdir()} != expected_files
        ):
            raise FrozenArtifactError("E10 evidence inventory is incomplete")
        return {
            gate: assets.ledger.evaluate_gate(gate, root / f"{gate}.json")
            for gate in assets.ledger.contract.required_gates
        }

    if evidence_root.exists() or evidence_root.is_symlink():
        gate_results = evaluate(evidence_root)
    else:
        evidence_root.parent.mkdir(parents=True, exist_ok=True)
        prefix = f".{evidence_root.name}.stage-"
        for stale in evidence_root.parent.glob(f"{prefix}*"):
            if stale.is_dir() and not stale.is_symlink():
                shutil.rmtree(stale)
        stage = Path(tempfile.mkdtemp(prefix=prefix, dir=evidence_root.parent))
        try:
            record_set = assets.ledger.record_set_digest()
            for gate in assets.ledger.contract.required_gates:
                write_gate_evidence(
                    stage / f"{gate}.json",
                    phase=ExperimentPhase.E10,
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
    result: PhaseCompletion | PhaseFalsification
    if all(value.passed for value in gate_results.values()):
        result = assets.ledger.finalize(gate_results)
        replayed = PhaseRunLedger.open(
            assets.ledger.directory,
            study=assets.ledger.study,
        ).verify_complete()
        if replayed.completion_digest != result.completion_digest:
            raise FrozenArtifactError("E10 completion does not replay")
    else:
        result = assets.ledger.finalize_falsified(gate_results)
        replayed_failure = PhaseRunLedger.open(
            assets.ledger.directory,
            study=assets.ledger.study,
        ).verify_falsified()
        if replayed_failure.falsification_digest != result.falsification_digest:
            raise FrozenArtifactError("E10 falsification does not replay")
    return result

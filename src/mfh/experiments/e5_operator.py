"""End-to-end operator lifecycle for E5 selection and ledger promotion.

The 972-arm developmental grid is executed once by :mod:`e5_native`.  This
module freezes the matched selection and promotes the already executed M1 and
selected-M3 rows into the ordinary four-condition E5 ``PhaseRunLedger``.  No
model generation is repeated during promotion.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from mfh.config import load_model_spec, load_prompt_specs
from mfh.contracts import (
    ActivationSite,
    AdaptivePolicySpec,
    GenerationRecord,
    ModelSpec,
    PromptSpec,
    Question,
    TokenScope,
)
from mfh.data.normalization import normalize_answer
from mfh.errors import ConfigurationError, DataValidationError, FrozenArtifactError
from mfh.evaluation.grading import triviaqa_scores
from mfh.experiments.e5_adaptive import (
    E5AblationSpec,
    E5Measurement,
    E5Protocol,
    E5Selection,
    E5StaticReference,
    build_e5_ablation_grid,
    derive_e5_selection,
    finalize_e5_phase,
    load_e5_controller_binding,
    verify_e5_phase,
)
from mfh.experiments.e5_fit import verify_e5_controller_bindings
from mfh.experiments.e5_native import (
    E5NativePromotionRow,
    VerifiedE5NativeRun,
    e5_native_execution_public_key,
    iter_e5_native_promotion_rows,
    open_e5_native_promotion_source,
    verify_e5_native_ablation,
)
from mfh.experiments.model_selection import (
    validate_active_model_spec,
    validate_active_study_artifact_paths,
)
from mfh.experiments.protocol import ExperimentPhase, StudyProtocol
from mfh.experiments.runner import (
    EvaluationCondition,
    PhaseRunContract,
    PhaseRunLedger,
    adaptive_policy_decision_digest,
    open_phase_prerequisite,
    sign_adaptive_execution_receipt,
)
from mfh.provenance import canonical_json, sha256_file, sha256_path, stable_hash

_PROMPTS = ("P0-neutral", "P2-calibrated-abstention")
_QUESTION_COUNT = 5_000
_GRID_ARM_COUNT = 972
E5_EXACT_GRID_RECORDS = (1 + _GRID_ARM_COUNT) * len(_PROMPTS) * _QUESTION_COUNT
E5_PROMOTED_RECORDS = 2 * len(_PROMPTS) * _QUESTION_COUNT
_SELECTION_FILES = frozenset({"selection.json", "receipt.json"})
_SELECTION_RULE = "signed-full-grid-native-selection-v1"
_TOKEN_SCOPE = MappingProxyType(
    {
        "final_prompt": TokenScope.FINAL_PROMPT,
        "first_generated": TokenScope.FIRST_GENERATED,
        "first_four_generated": TokenScope.FIRST_FOUR,
    }
)


def estimate_e5_native_ablation(
    *,
    generations_per_second: float,
    checkpoint_opens_per_second: float,
    verification_rows_per_second: float,
    request_budget: int = 1_024,
) -> Mapping[str, Any]:
    """Return generation plus measured checkpoint/full-replay runtime estimates."""

    if (
        isinstance(generations_per_second, bool)
        or not isinstance(generations_per_second, int | float)
        or not math.isfinite(float(generations_per_second))
        or float(generations_per_second) <= 0
        or isinstance(checkpoint_opens_per_second, bool)
        or not isinstance(checkpoint_opens_per_second, int | float)
        or not math.isfinite(float(checkpoint_opens_per_second))
        or float(checkpoint_opens_per_second) <= 0
        or isinstance(verification_rows_per_second, bool)
        or not isinstance(verification_rows_per_second, int | float)
        or not math.isfinite(float(verification_rows_per_second))
        or float(verification_rows_per_second) <= 0
        or type(request_budget) is not int
        or request_budget <= 0
    ):
        raise ConfigurationError("E5 estimate requires a positive measured rate and budget")
    generation_seconds = E5_EXACT_GRID_RECORDS / float(generations_per_second)
    sessions = math.ceil(E5_EXACT_GRID_RECORDS / request_budget)
    checkpoint_seconds = sessions / float(checkpoint_opens_per_second)
    full_row_replay_passes = 3
    full_manifest_entry_passes = 1
    verification_entry_visits = E5_EXACT_GRID_RECORDS * (
        full_row_replay_passes + full_manifest_entry_passes
    )
    verification_seconds = verification_entry_visits / float(verification_rows_per_second)
    seconds = generation_seconds + checkpoint_seconds + verification_seconds
    return MappingProxyType(
        {
            "valid": True,
            "grid_arm_count": _GRID_ARM_COUNT,
            "static_arm_count": 1,
            "prompt_count": len(_PROMPTS),
            "question_count": _QUESTION_COUNT,
            "exact_grid_records": E5_EXACT_GRID_RECORDS,
            "promoted_records": E5_PROMOTED_RECORDS,
            "generations_per_second": float(generations_per_second),
            "checkpoint_opens_per_second": float(checkpoint_opens_per_second),
            "verification_rows_per_second": float(verification_rows_per_second),
            "generation_seconds": generation_seconds,
            "checkpoint_open_count": sessions,
            "checkpoint_seconds": checkpoint_seconds,
            "full_row_replay_passes": full_row_replay_passes,
            "full_manifest_entry_passes": full_manifest_entry_passes,
            "verification_entry_visits": verification_entry_visits,
            "verification_seconds": verification_seconds,
            "estimated_seconds": seconds,
            "estimated_hours": seconds / 3_600,
            "estimated_days": seconds / 86_400,
            "request_budget": request_budget,
            "estimated_sessions": sessions,
        }
    )


def _sign_body(body: Mapping[str, Any], *, private_key_hex: str) -> str:
    try:
        private = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_key_hex))
        if len(private_key_hex) != 64 or private_key_hex.lower() != private_key_hex:
            raise ValueError("key is not canonical")
    except ValueError as exc:
        raise DataValidationError(f"invalid E5 operator execution key: {exc}") from exc
    return private.sign(canonical_json(body).encode("utf-8")).hex()


def _verify_signed_body(
    value: Mapping[str, Any],
    *,
    expected_public_key: str,
    context: str,
) -> dict[str, Any]:
    body = dict(value)
    signature = body.pop("signature", None)
    if not isinstance(signature, str) or len(signature) != 128:
        raise FrozenArtifactError(f"{context} lacks its Ed25519 signature")
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(expected_public_key)).verify(
            bytes.fromhex(signature), canonical_json(body).encode("utf-8")
        )
    except (InvalidSignature, ValueError) as exc:
        raise FrozenArtifactError(f"{context} signature differs") from exc
    return body


def _verified_native(
    directory: str | Path,
    *,
    execution_private_key_hex: str,
    semantic: bool,
) -> VerifiedE5NativeRun:
    verified = verify_e5_native_ablation(
        directory,
        expected_execution_public_key=e5_native_execution_public_key(execution_private_key_hex),
        require_complete=True,
        semantic=semantic,
    )
    if (
        not verified.scientific_eligible
        or verified.finalized_records is None
        or verified.chain_head is None
        or verified.records_completed != E5_EXACT_GRID_RECORDS
        or verified.plan["expected_records"] != E5_EXACT_GRID_RECORDS
        or E5Protocol.from_dict(verified.plan["protocol"]) != E5Protocol()
    ):
        raise DataValidationError("E5 operator requires the finalized scientific exact grid")
    return verified


def _parse_frozen_selection(path: Path) -> E5Selection:
    """Parse a selection whose exact bytes are protected by an operator receipt."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError("selection root is not a mapping")
        static_value = value["static_reference"]
        measurements_value = value["measurements"]
        if not isinstance(static_value, dict) or not isinstance(measurements_value, list):
            raise TypeError("selection measurements are invalid")
        selection = E5Selection(
            schema_version=value["schema_version"],
            protocol=E5Protocol.from_dict(value["protocol"]),
            screen_receipt_path=value["screen_receipt_path"],
            screen_receipt_sha256=value["screen_receipt_sha256"],
            record_artifact_path=value["record_artifact_path"],
            record_set_digest=value["record_set_digest"],
            upstream_paths=value["upstream_paths"],
            upstream_digests=value["upstream_digests"],
            controller_binding_paths=value["controller_binding_paths"],
            controller_binding_fingerprints=value["controller_binding_fingerprints"],
            source_plan_identity=value["source_plan_identity"],
            selection_rule_sha256=value["selection_rule_sha256"],
            static_reference=E5StaticReference(**static_value),
            measurements=tuple(E5Measurement(**item) for item in measurements_value),
            matched_spec_ids=value["matched_spec_ids"],
            selected_spec_id=value["selected_spec_id"],
            falsification_reason=value["falsification_reason"],
            scientific_eligible=value["scientific_eligible"],
            selection_digest=value["selection_digest"],
        )
    except (
        OSError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        DataValidationError,
    ) as exc:
        raise FrozenArtifactError(f"cannot parse signed E5 selection: {exc}") from exc
    if path.read_text(encoding="utf-8") != (
        json.dumps(selection.to_dict(), indent=2, sort_keys=True) + "\n"
    ):
        raise FrozenArtifactError("signed E5 selection is not canonical JSON")
    return selection


def derive_signed_e5_selection(
    destination: str | Path,
    *,
    native_directory: str | Path,
    controller_bindings_directory: str | Path,
    e2_probe_bundle: str | Path,
    e3_static_vectors: str | Path,
    e4_promoted_baselines: str | Path,
    execution_private_key_hex: str,
) -> Mapping[str, Any]:
    """Replay the full grid once, select one arm, and sign the immutable result."""

    normalized = validate_active_study_artifact_paths(
        {
            "E5 signed selection": destination,
            "E5 native ablation": native_directory,
            "E5 controller bindings": controller_bindings_directory,
            "E5 E2 probes": e2_probe_bundle,
            "E5 E3 vectors": e3_static_vectors,
            "E5 E4 promotion": e4_promoted_baselines,
        }
    )
    output = normalized["E5 signed selection"]
    if output.exists() or output.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite E5 signed selection: {output}")
    native_directory = normalized["E5 native ablation"]
    try:
        final_receipt = json.loads(
            (native_directory / "final" / "receipt.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise FrozenArtifactError(f"cannot read E5 native final receipt: {exc}") from exc
    if not isinstance(final_receipt, Mapping):
        raise FrozenArtifactError("E5 native final receipt must be a mapping")
    expected_records_sha = final_receipt.get("records_sha256")
    if type(expected_records_sha) is not str:
        raise FrozenArtifactError("E5 native final receipt lacks its record digest")
    native = open_e5_native_promotion_source(
        native_directory,
        expected_execution_public_key=e5_native_execution_public_key(
            execution_private_key_hex
        ),
        expected_final_records_sha256=expected_records_sha,
    )
    if (
        not native.scientific_eligible
        or native.finalized_records is None
        or native.chain_head is None
        or native.records_completed != E5_EXACT_GRID_RECORDS
        or native.plan["expected_records"] != E5_EXACT_GRID_RECORDS
        or E5Protocol.from_dict(native.plan["protocol"]) != E5Protocol()
    ):
        raise DataValidationError("E5 selection requires the finalized scientific exact grid")
    bindings = verify_e5_controller_bindings(normalized["E5 controller bindings"])
    if (
        Path(native.plan["controller_bindings_path"]).resolve() != bindings.directory.resolve()
        or native.plan["controller_bindings_sha256"] != sha256_path(bindings.directory)
        or native.plan["controller_bindings_manifest_digest"]
        != bindings.manifest["manifest_digest"]
        or native.plan["e3_static_vectors_sha256"] != sha256_path(normalized["E5 E3 vectors"])
    ):
        raise FrozenArtifactError("E5 selection sources differ from the native execution plan")
    assert native.finalized_records is not None
    selection = derive_e5_selection(
        screen_receipt_path=native.plan["screen_receipt_path"],
        record_artifact_path=native.finalized_records,
        upstream_artifacts={
            "E2_calibrated_probes": normalized["E5 E2 probes"],
            "E3_static_vectors": normalized["E5 E3 vectors"],
            "E4_promoted_baselines": normalized["E5 E4 promotion"],
        },
        controller_binding_artifacts=bindings.binding_paths,
        protocol=E5Protocol(),
    )
    if selection.selected_spec_id is None or selection.falsification_reason is not None:
        raise DataValidationError(
            "E5 exact-grid selection falsified: no controller matches all four M1 budgets"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        selection_path = stage / "selection.json"
        selection_path.write_text(
            json.dumps(selection.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        body = {
            "schema_version": 1,
            "selection_rule": _SELECTION_RULE,
            "selection_file": "selection.json",
            "selection_sha256": sha256_file(selection_path),
            "selection_digest": selection.selection_digest,
            "selected_spec_id": selection.selected_spec_id,
            "native_directory": str(native.directory.resolve()),
            "native_plan_identity": native.plan["plan_identity"],
            "native_chain_head": native.chain_head,
            "native_record_set_sha256": selection.record_set_digest,
            "native_records": native.records_completed,
            "controller_bindings_manifest_digest": bindings.manifest["manifest_digest"],
            "upstream_digests": dict(selection.upstream_digests),
            "execution_public_key": native.plan["execution_public_key"],
        }
        receipt = {
            **body,
            "signature": _sign_body(body, private_key_hex=execution_private_key_hex),
        }
        (stage / "receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return verify_signed_e5_selection(
        output,
        native_directory=native.directory,
        execution_private_key_hex=execution_private_key_hex,
    )


def _load_signed_e5_selection(
    directory: str | Path,
    *,
    native_directory: str | Path,
    execution_private_key_hex: str,
) -> tuple[E5Selection, VerifiedE5NativeRun]:
    """Load the signed selection and native source once for an operator action."""

    normalized = validate_active_study_artifact_paths(
        {
            "E5 signed selection": directory,
            "E5 native ablation": native_directory,
        }
    )
    source = normalized["E5 signed selection"]
    native_directory = normalized["E5 native ablation"]
    if (
        source.is_symlink()
        or not source.is_dir()
        or {value.name for value in source.iterdir()} != _SELECTION_FILES
        or any(value.is_symlink() for value in source.iterdir())
    ):
        raise FrozenArtifactError("E5 signed selection inventory differs")
    try:
        receipt_value = json.loads((source / "receipt.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read E5 signed selection receipt: {exc}") from exc
    if not isinstance(receipt_value, dict):
        raise FrozenArtifactError("E5 signed selection receipt must be a mapping")
    public_key = e5_native_execution_public_key(execution_private_key_hex)
    receipt = _verify_signed_body(
        receipt_value,
        expected_public_key=public_key,
        context="E5 signed selection",
    )
    selection_path = source / "selection.json"
    selection = _parse_frozen_selection(selection_path)
    native = open_e5_native_promotion_source(
        native_directory,
        expected_execution_public_key=public_key,
        expected_final_records_sha256=selection.record_set_digest,
    )
    expected = {
        "schema_version": 1,
        "selection_rule": _SELECTION_RULE,
        "selection_file": "selection.json",
        "selection_sha256": sha256_file(selection_path),
        "selection_digest": selection.selection_digest,
        "selected_spec_id": selection.selected_spec_id,
        "native_directory": str(native.directory.resolve()),
        "native_plan_identity": native.plan["plan_identity"],
        "native_chain_head": native.chain_head,
        "native_record_set_sha256": selection.record_set_digest,
        "native_records": native.records_completed,
        "controller_bindings_manifest_digest": native.plan["controller_bindings_manifest_digest"],
        "upstream_digests": dict(selection.upstream_digests),
        "execution_public_key": public_key,
    }
    if (
        receipt != expected
        or selection.record_artifact_path != str(native.finalized_records)
        or selection.screen_receipt_path != str(Path(native.plan["screen_receipt_path"]).resolve())
        or selection.selected_spec_id is None
    ):
        raise FrozenArtifactError("E5 signed selection differs from its native source")
    return selection, native


def verify_signed_e5_selection(
    directory: str | Path,
    *,
    native_directory: str | Path,
    execution_private_key_hex: str,
) -> Mapping[str, Any]:
    """Verify the selection signature and signed native finalization source."""

    selection, native = _load_signed_e5_selection(
        directory,
        native_directory=native_directory,
        execution_private_key_hex=execution_private_key_hex,
    )
    return MappingProxyType(
        {
            "valid": True,
            "directory": Path(directory).resolve(),
            "selection_path": _selection_path(directory),
            "selection_digest": selection.selection_digest,
            "selected_spec_id": selection.selected_spec_id,
            "native_plan_identity": native.plan["plan_identity"],
            "native_chain_head": native.chain_head,
            "native_records": native.records_completed,
            "execution_public_key": native.plan["execution_public_key"],
        }
    )


@dataclass(frozen=True, slots=True)
class _E5PromotionContext:
    selection: E5Selection
    native: VerifiedE5NativeRun
    selected_spec: E5AblationSpec
    questions: tuple[Question, ...]
    adaptive_policy: AdaptivePolicySpec


def _selection_path(directory: str | Path) -> Path:
    return Path(directory).resolve() / "selection.json"


def _promotion_context(
    *,
    selection_directory: str | Path,
    native_directory: str | Path,
    execution_private_key_hex: str,
) -> _E5PromotionContext:
    selection, native = _load_signed_e5_selection(
        selection_directory,
        native_directory=native_directory,
        execution_private_key_hex=execution_private_key_hex,
    )
    assert selection.selected_spec_id is not None
    selected_spec = next(
        value
        for value in build_e5_ablation_grid(selection.protocol)
        if value.spec_id == selection.selected_spec_id
    )
    binding = load_e5_controller_binding(
        selection.controller_binding_paths[selection.selected_spec_id]
    )
    controller = binding.assert_current()
    candidate_layers = (
        (controller.fixed_layer,)
        if controller.fixed_layer is not None
        else controller.layer_selector.candidate_layers
        if controller.layer_selector is not None
        else ()
    )
    candidate_sites = tuple(
        sorted(
            {
                key.site
                for key in controller.vector_bank.directions
                if key.layer in candidate_layers
            },
            key=lambda value: value.value,
        )
    )
    alpha = controller.alpha_controller
    policy = AdaptivePolicySpec(
        schema_version=2,
        release_risk_threshold=0.4,
        abstention_probability_threshold=0.7,
        alpha_max=alpha.alpha_max,
        alpha_beta=alpha.beta,
        layer=None,
        site=None,
        token_scope=None,
        direction_sha256=None,
        direction_norm=None,
        execution_public_key=binding.execution_public_key,
        sparsity=None,
        controller_artifact_sha256=binding.controller_artifact_sha256,
        candidate_layers=candidate_layers,
        candidate_sites=candidate_sites,
        candidate_token_scopes=(_TOKEN_SCOPE[selected_spec.intervention_timing],),
        vector_count=selected_spec.vector_count,
        likely_unknown_risk_threshold=0.8,
        alpha_mode=alpha.mode.value,
        alpha_risk_threshold=alpha.threshold,
    )
    questions = tuple(native._source_context.screen.dev_questions)
    if (
        len(questions) != _QUESTION_COUNT
        or stable_hash([value.question_id for value in questions])
        != native.plan["question_ids_sha256"]
        or any(value.benchmark != "triviaqa" or value.split != "T-dev" for value in questions)
        or binding.execution_public_key != native.plan["execution_public_key"]
    ):
        raise FrozenArtifactError("E5 promotion context differs from the scientific screen")
    return _E5PromotionContext(
        selection=selection,
        native=native,
        selected_spec=selected_spec,
        questions=questions,
        adaptive_policy=policy,
    )


def _condition(
    *,
    study: StudyProtocol,
    model: ModelSpec,
    prompt: PromptSpec,
    method: str,
    context: _E5PromotionContext,
) -> EvaluationCondition:
    if method == "M1":
        static = context.native._source_context.m1_policy
        return EvaluationCondition(
            phase=ExperimentPhase.E5,
            benchmark="triviaqa",
            partition="T-dev",
            model_name=model.name,
            model_repository=model.repository,
            model_revision=model.revision,
            runtime=model.runtime,
            quantization=model.quantization,
            model_num_layers=model.num_layers,
            system_prompt_id=prompt.prompt_id,
            prompt_template_sha256=hashlib.sha256(prompt.text.encode()).hexdigest(),
            steering_method="M1",
            method_artifact_sha256=context.selection.upstream_digests["E3_static_vectors"],
            layer=static.layer,
            site=static.site,
            token_scope=static.token_scope,
            alpha=static.alpha,
            sparsity=None,
            seed=17,
            study_protocol_digest=study.digest,
        )
    assert context.selection.selected_spec_id is not None
    return EvaluationCondition(
        phase=ExperimentPhase.E5,
        benchmark="triviaqa",
        partition="T-dev",
        model_name=model.name,
        model_repository=model.repository,
        model_revision=model.revision,
        runtime=model.runtime,
        quantization=model.quantization,
        model_num_layers=model.num_layers,
        system_prompt_id=prompt.prompt_id,
        prompt_template_sha256=hashlib.sha256(prompt.text.encode()).hexdigest(),
        steering_method="M3",
        method_artifact_sha256=context.selection.controller_binding_fingerprints[
            context.selection.selected_spec_id
        ],
        layer=None,
        site=None,
        token_scope=None,
        alpha=0.0,
        sparsity=None,
        seed=17,
        study_protocol_digest=study.digest,
        adaptive_policy=context.adaptive_policy,
    )


def prepare_e5_phase_ledger(
    ledger_directory: str | Path,
    *,
    selection_directory: str | Path,
    native_directory: str | Path,
    model: ModelSpec,
    prompts: Mapping[str, PromptSpec],
    study: StudyProtocol,
    prerequisite_runs: Mapping[ExperimentPhase | str, str | Path],
    execution_private_key_hex: str,
) -> PhaseRunLedger:
    """Create the empty four-condition final E5 ledger from frozen sources."""

    validate_active_model_spec(model)
    if type(study) is not StudyProtocol or set(prompts) != set(_PROMPTS):
        raise ConfigurationError("E5 final ledger model, study, or prompt inventory differs")
    phase = study.phase(ExperimentPhase.E5)
    expected_prerequisites = {
        ExperimentPhase.E2,
        ExperimentPhase.E3,
        ExperimentPhase.E4,
    }
    normalized_prerequisites = {
        ExperimentPhase(key): Path(value).resolve() for key, value in prerequisite_runs.items()
    }
    if set(normalized_prerequisites) != expected_prerequisites:
        raise DataValidationError("E5 final ledger prerequisite inventory differs")
    context = _promotion_context(
        selection_directory=selection_directory,
        native_directory=native_directory,
        execution_private_key_hex=execution_private_key_hex,
    )
    for prompt_id in _PROMPTS:
        prompt = prompts[prompt_id]
        expected_prompt = context.native.plan["prompts"][prompt_id]
        if (
            type(prompt) is not PromptSpec
            or prompt.prompt_id != prompt_id
            or not isinstance(expected_prompt, Mapping)
            or expected_prompt.get("text") != prompt.text
        ):
            raise FrozenArtifactError("E5 final prompt differs from native execution")
    completions = {}
    for prerequisite, path in normalized_prerequisites.items():
        prior = open_phase_prerequisite(path, phase=prerequisite, study=study)
        completion = prior.verify_complete()
        if completion.phase is not prerequisite:
            raise FrozenArtifactError("E5 prerequisite resolves to another phase")
        completions[prerequisite.value] = completion.completion_digest
    conditions = tuple(
        _condition(
            study=study,
            model=model,
            prompt=prompts[prompt_id],
            method=method,
            context=context,
        )
        for prompt_id in _PROMPTS
        for method in ("M1", "M3")
    )
    contract = PhaseRunContract(
        phase=ExperimentPhase.E5,
        study_protocol_digest=study.digest,
        conditions=conditions,
        question_ids_by_benchmark={
            "triviaqa": tuple(value.question_id for value in context.questions)
        },
        input_fingerprints=context.selection.upstream_digests,
        prerequisite_digests=completions,
        required_gates=phase.gates,
    )
    prerequisite_arguments: Mapping[ExperimentPhase | str, str | Path] = {
        prerequisite: path for prerequisite, path in normalized_prerequisites.items()
    }
    return PhaseRunLedger.create(
        ledger_directory,
        contract,
        study=study,
        input_artifacts=context.selection.upstream_paths,
        prerequisite_runs=prerequisite_arguments,
    )


def _condition_inventory(
    ledger: PhaseRunLedger,
) -> Mapping[tuple[str, str], EvaluationCondition]:
    conditions: dict[tuple[str, str], EvaluationCondition] = {}
    for condition in ledger.contract.conditions:
        key = (condition.system_prompt_id, condition.steering_method)
        if key in conditions:
            raise FrozenArtifactError("E5 final ledger repeats a prompt/method condition")
        conditions[key] = condition
    if set(conditions) != {(prompt, method) for prompt in _PROMPTS for method in ("M1", "M3")}:
        raise FrozenArtifactError("E5 final ledger condition inventory differs")
    return MappingProxyType(conditions)


def e5_promotion_record(
    source: E5NativePromotionRow,
    *,
    question: Question,
    condition: EvaluationCondition,
    adaptive_policy: AdaptivePolicySpec,
    native: VerifiedE5NativeRun,
    execution_private_key_hex: str,
) -> GenerationRecord:
    """Convert one semantically verified native row into its final ledger row."""

    record = source.record
    evidence = source.evidence
    method = "M1" if record.arm_id == "M1" else "M3"
    if (
        type(source) is not E5NativePromotionRow
        or question.question_id != record.question_id
        or question.benchmark != "triviaqa"
        or condition.phase is not ExperimentPhase.E5
        or condition.system_prompt_id != record.prompt_id
        or condition.steering_method != method
        or condition.benchmark != "triviaqa"
        or condition.partition != "T-dev"
        or condition.adaptive_policy != (adaptive_policy if method == "M3" else None)
        or evidence.get("end_to_end_latency_seconds") != record.generation_latency_seconds
    ):
        raise DataValidationError("E5 native row and final ledger condition differ")
    if method == "M1":
        selected_layer = evidence.get("selected_layer")
        selected_site = evidence.get("selected_site")
        standardized_alpha = evidence.get("standardized_alpha")
        expected_norm = evidence.get("expected_activation_delta_norm")
        if (
            type(selected_layer) is not int
            or selected_site != ActivationSite.POST_MLP.value
            or isinstance(standardized_alpha, bool)
            or not isinstance(standardized_alpha, int | float)
            or not math.isfinite(float(standardized_alpha))
            or isinstance(expected_norm, bool)
            or not isinstance(expected_norm, int | float)
            or not math.isfinite(float(expected_norm))
            or condition.layer != selected_layer
            or condition.site is not ActivationSite.POST_MLP
            or condition.token_scope is not record.token_scope
            or not math.isclose(
                condition.alpha,
                float(standardized_alpha),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or condition.sparsity is not None
            or evidence.get("method_artifact_sha256") != native.plan["m1_policy_sha256"]
            or condition.method_artifact_sha256 != native.plan["e3_static_vectors_sha256"]
            or not math.isclose(
                record.intervention_norm,
                float(expected_norm),
                rel_tol=0.025,
                abs_tol=1e-6,
            )
            or record.intervention_norm <= 0.0
        ):
            raise FrozenArtifactError(
                "E5 M1 final geometry differs from signed native policy evidence"
            )
    raw_output = evidence.get("raw_output")
    input_tokens = evidence.get("input_tokens")
    if type(raw_output) is not str or type(input_tokens) is not int or input_tokens <= 0:
        raise FrozenArtifactError("E5 native promotion lacks generation output evidence")
    exact_match, token_f1 = triviaqa_scores(raw_output, question.aliases)
    action = str(record.execution_receipt["policy_action"])
    scores = (
        {key: float(value) for key, value in record.execution_receipt["controller_scores"].items()}
        if method == "M3"
        else {}
    )
    layer: int | None = condition.layer
    site: ActivationSite | None = condition.site
    scope: TokenScope | None = condition.token_scope
    alpha = condition.alpha
    metadata: dict[str, Any] = {
        "phase": ExperimentPhase.E5.value,
        "partition": condition.partition,
        "prompt_template_sha256": condition.prompt_template_sha256,
        "study_protocol_digest": condition.study_protocol_digest,
        "method_artifact_sha256": condition.method_artifact_sha256,
        "intervention_norm": record.intervention_norm,
        "official_scorer": "mfh.triviaqa.alias-aware-em-f1.v1",
        "official_exact_match": exact_match,
        "official_token_f1": token_f1,
        "reference_aliases_digest": stable_hash(list(question.aliases)),
        "rendered_prompt_token_ids_sha256": evidence["rendered_prompt_token_ids_sha256"],
        "native_plan_identity": native.plan["plan_identity"],
        "native_chain_head": native.chain_head,
        "native_sequence": source.sequence,
        "native_row_digest": source.row_digest,
        "native_execution_receipt_digest": record.execution_receipt_digest,
        "native_execution_receipt_signature": record.execution_receipt_signature,
        "native_prompt_input_sha256": record.prompt_input_sha256,
        "native_runtime_identity_sha256": evidence["runtime_identity_sha256"],
        "native_method_artifact_sha256": evidence.get("method_artifact_sha256"),
        "native_feature_schema_digest": evidence.get("feature_schema_digest"),
        "native_feature_values_sha256": evidence.get("feature_values_sha256"),
        "native_routing_weights_sha256": evidence.get("routing_weights_sha256"),
        "native_layer_routing_weights": evidence.get("layer_routing_weights"),
        "native_maximum_token_probability": evidence.get("maximum_token_probability"),
        "native_output_entropy": evidence.get("output_entropy"),
        "native_end_to_end_latency_seconds": evidence["end_to_end_latency_seconds"],
        "native_generation_latency_seconds": evidence["generation_latency_seconds"],
    }
    if method == "M3":
        if action == "intervene":
            layer = int(evidence["selected_layer"])
            site = ActivationSite(str(evidence["selected_site"]))
            scope = record.token_scope
            alpha = float(evidence["standardized_alpha"])
            router_weights = [float(value) for value in evidence["routing_weights"]]
            trace = {
                "layer": layer,
                "site": site.value,
                "token_scope": scope.value,
                "alpha": alpha,
                "sparsity": None,
                "applied_tokens": int(evidence["hook_applications"]),
                "applied_token_indices": list(evidence["applied_token_indices"]),
                "activation_delta_norm": record.intervention_norm,
                "direction_sha256": evidence["direction_sha256"],
                "pre_activation_sha256": evidence["pre_activation_sha256"],
                "post_activation_sha256": evidence["post_activation_sha256"],
                "delta_sha256": evidence["delta_sha256"],
                "direction_norm": float(evidence["direction_norm"]),
                "controller_artifact_sha256": record.execution_receipt[
                    "controller_artifact_sha256"
                ],
                "router_weights": router_weights,
                "router_weights_sha256": stable_hash(router_weights),
            }
            metadata.update(
                {
                    "intervention_trace": trace,
                    "intervention_trace_digest": stable_hash(trace),
                }
            )
        elif action == "release":
            layer = None
            site = None
            scope = None
            alpha = 0.0
        else:
            raise FrozenArtifactError("E5 M3 native row has an unsupported final action")
        metadata["policy_action"] = action
    elif action != "intervene":
        raise FrozenArtifactError("E5 M1 native row did not execute its static intervention")
    draft = GenerationRecord(
        question_id=record.question_id,
        benchmark="triviaqa",
        model_repository=condition.model_repository,
        model_revision=condition.model_revision,
        runtime=condition.runtime,
        quantization=condition.quantization,
        system_prompt_id=condition.system_prompt_id,
        rendered_prompt_hash=record.rendered_prompt_sha256,
        steering_method=condition.steering_method,
        layer=layer,
        site=site,
        token_scope=scope,
        alpha=alpha,
        sparsity=None,
        controller_scores=scores,
        raw_output=raw_output,
        normalized_answer=normalize_answer(raw_output),
        outcome=record.outcome,
        generation_latency_seconds=record.generation_latency_seconds,
        input_tokens=input_tokens,
        output_tokens=record.output_tokens,
        condition_id=condition.condition_id,
        seed=condition.seed,
        metadata=metadata,
    )
    if method == "M1":
        return draft
    decided = replace(
        draft,
        metadata={
            **draft.metadata,
            "policy_decision_digest": adaptive_policy_decision_digest(
                draft,
                policy=adaptive_policy,
                policy_action=action,
            ),
        },
    )
    return replace(
        decided,
        metadata={
            **decided.metadata,
            "execution_receipt_signature": sign_adaptive_execution_receipt(
                decided,
                policy=adaptive_policy,
                private_key_hex=execution_private_key_hex,
            ),
        },
    )


def _iter_expected_promotions(
    *,
    ledger: PhaseRunLedger,
    context: _E5PromotionContext,
    execution_private_key_hex: str,
) -> Iterator[GenerationRecord]:
    conditions = _condition_inventory(ledger)
    questions = {value.question_id: value for value in context.questions}
    assert context.selection.selected_spec_id is not None
    for row in iter_e5_native_promotion_rows(
        context.native,
        selected_spec_id=context.selection.selected_spec_id,
    ):
        method = "M1" if row.record.arm_id == "M1" else "M3"
        try:
            question = questions[row.record.question_id]
            condition = conditions[(row.record.prompt_id, method)]
        except KeyError as exc:
            raise FrozenArtifactError(
                "E5 native promotion row is outside the final contract"
            ) from exc
        yield e5_promotion_record(
            row,
            question=question,
            condition=condition,
            adaptive_policy=context.adaptive_policy,
            native=context.native,
            execution_private_key_hex=execution_private_key_hex,
        )


def promote_e5_phase_records(
    ledger_directory: str | Path,
    *,
    selection_directory: str | Path,
    native_directory: str | Path,
    study: StudyProtocol,
    execution_private_key_hex: str,
    request_budget: int | None = None,
    checkpoint_rows: int = 250,
) -> Mapping[str, Any]:
    """Resume promotion without invoking MLX or generating another answer."""

    if (
        (request_budget is not None and (type(request_budget) is not int or request_budget <= 0))
        or type(checkpoint_rows) is not int
        or checkpoint_rows <= 0
    ):
        raise ConfigurationError("E5 promotion budgets must be positive exact integers")
    ledger = PhaseRunLedger.open(ledger_directory, study=study)
    if ledger.contract.phase is not ExperimentPhase.E5:
        raise DataValidationError("E5 promotion requires an E5 PhaseRunLedger")
    context = _promotion_context(
        selection_directory=selection_directory,
        native_directory=native_directory,
        execution_private_key_hex=execution_private_key_hex,
    )
    existing = {(value.condition_id, value.question_id): value for value in ledger.records()}
    batch: list[GenerationRecord] = []
    added = 0
    replayed_existing = 0
    for expected in _iter_expected_promotions(
        ledger=ledger,
        context=context,
        execution_private_key_hex=execution_private_key_hex,
    ):
        key = (expected.condition_id, expected.question_id)
        prior = existing.get(key)
        if prior is not None:
            if prior != expected:
                raise FrozenArtifactError("promoted E5 ledger row differs from native replay")
            replayed_existing += 1
            continue
        if request_budget is not None and added >= request_budget:
            break
        batch.append(expected)
        added += 1
        if len(batch) == checkpoint_rows:
            ledger.checkpoint(batch)
            batch.clear()
    if batch:
        ledger.checkpoint(batch)
    completed, expected_count = ledger.progress()
    if expected_count != E5_PROMOTED_RECORDS:
        raise FrozenArtifactError("E5 final ledger does not contain exactly 20,000 slots")
    return MappingProxyType(
        {
            "valid": True,
            "ledger_directory": ledger.directory.resolve(),
            "added_records": added,
            "replayed_existing_records": replayed_existing,
            "completed_records": completed,
            "expected_records": expected_count,
            "complete": completed == expected_count,
            "model_generations_executed": 0,
            "native_plan_identity": context.native.plan["plan_identity"],
            "selected_spec_id": context.selection.selected_spec_id,
        }
    )


def verify_e5_phase_promotion(
    ledger_directory: str | Path,
    *,
    selection_directory: str | Path,
    native_directory: str | Path,
    study: StudyProtocol,
    execution_private_key_hex: str,
    require_complete: bool = True,
) -> Mapping[str, Any]:
    """Replay every promoted row against its signed native transcript."""

    ledger = PhaseRunLedger.open(ledger_directory, study=study)
    context = _promotion_context(
        selection_directory=selection_directory,
        native_directory=native_directory,
        execution_private_key_hex=execution_private_key_hex,
    )
    existing = {(value.condition_id, value.question_id): value for value in ledger.records()}
    replayed = 0
    for expected in _iter_expected_promotions(
        ledger=ledger,
        context=context,
        execution_private_key_hex=execution_private_key_hex,
    ):
        key = (expected.condition_id, expected.question_id)
        observed = existing.get(key)
        if observed is None:
            continue
        if observed != expected:
            raise FrozenArtifactError("E5 phase ledger differs from native promotion replay")
        replayed += 1
    completed, expected_count = ledger.progress()
    if (
        ledger.contract.phase is not ExperimentPhase.E5
        or expected_count != E5_PROMOTED_RECORDS
        or replayed != completed
        or (require_complete and completed != expected_count)
    ):
        raise FrozenArtifactError("E5 phase promotion is incomplete or has unmatched rows")
    return MappingProxyType(
        {
            "valid": True,
            "ledger_directory": ledger.directory.resolve(),
            "completed_records": completed,
            "expected_records": expected_count,
            "complete": completed == expected_count,
            "native_rows_replayed": replayed,
            "native_plan_identity": context.native.plan["plan_identity"],
            "selected_spec_id": context.selection.selected_spec_id,
        }
    )


def finalize_promoted_e5_phase(
    destination: str | Path,
    *,
    ledger_directory: str | Path,
    selection_directory: str | Path,
    native_directory: str | Path,
    study: StudyProtocol,
    execution_private_key_hex: str,
) -> Mapping[str, Any]:
    """Verify zero-rerun promotion, evaluate gates, and freeze E5 terminal state."""

    verify_e5_phase_promotion(
        ledger_directory,
        selection_directory=selection_directory,
        native_directory=native_directory,
        study=study,
        execution_private_key_hex=execution_private_key_hex,
        require_complete=True,
    )
    return finalize_e5_phase(
        destination,
        ledger_directory=ledger_directory,
        study=study,
        selection_path=_selection_path(selection_directory),
    )


def verify_promoted_e5_phase(
    directory: str | Path,
    *,
    ledger_directory: str | Path,
    selection_directory: str | Path,
    native_directory: str | Path,
    study: StudyProtocol,
    execution_private_key_hex: str,
) -> Mapping[str, Any]:
    """Replay both final E5 provenance layers: promotion and terminal gates."""

    promotion = verify_e5_phase_promotion(
        ledger_directory,
        selection_directory=selection_directory,
        native_directory=native_directory,
        study=study,
        execution_private_key_hex=execution_private_key_hex,
        require_complete=True,
    )
    terminal = verify_e5_phase(
        directory,
        ledger_directory=ledger_directory,
        study=study,
        selection_path=_selection_path(selection_directory),
    )
    return MappingProxyType(
        {
            **dict(terminal),
            "promotion_valid": True,
            "promoted_records": promotion["completed_records"],
            "native_plan_identity": promotion["native_plan_identity"],
        }
    )


def load_e5_operator_inputs(
    *,
    model_config: str | Path,
    prompt_config: str | Path,
) -> tuple[ModelSpec, Mapping[str, PromptSpec]]:
    """Load the active model and exact P0/P2 prompts for CLI callers."""

    model = load_model_spec(model_config)
    prompts = {
        value.prompt_id: value
        for value in load_prompt_specs(prompt_config)
        if value.prompt_id in _PROMPTS
    }
    return model, MappingProxyType(prompts)

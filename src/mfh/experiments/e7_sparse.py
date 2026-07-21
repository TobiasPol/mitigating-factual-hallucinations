"""Ledger-bound E7 sparse-steering finalization and exact replay."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from mfh.contracts import (
    ActivationSite,
    GenerationRecord,
    Outcome,
    PromptSpec,
    Question,
    TokenScope,
)
from mfh.data.io import read_questions
from mfh.data.normalization import normalize_answer
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.evaluation.grading import deterministic_short_answer_grade
from mfh.experiments.e6_likelihood import _e3_direction_index
from mfh.experiments.gates import write_gate_evidence
from mfh.experiments.model_selection import (
    ACTIVE_MODEL_IDENTITIES,
    ACTIVE_MODEL_NAME,
    validate_active_study_artifact_paths,
)
from mfh.experiments.protocol import ExperimentPhase, StudyProtocol
from mfh.experiments.runner import (
    PhaseCompletion,
    PhaseFalsification,
    PhaseRunLedger,
    _copy_frozen_artifact,
    package_portable_phase_ledger,
    validate_side_effect_evaluation_bundle,
)
from mfh.experiments.runtime_evidence import build_generation_runtime_metrics
from mfh.inference.mlx_runtime import as_numpy
from mfh.methods.controls import (
    label_shuffled_centroid_direction,
    matched_random_direction,
    norm_matched_gaussian_perturbation,
)
from mfh.methods.features import ActivationKind
from mfh.methods.probes import ProbeDataset
from mfh.methods.sae_stability import load_sae_stability_bundle
from mfh.methods.sparse import (
    ActivationBatch,
    ActivationCorpus,
    CoordinateSparseArtifact,
    SAEInterventionArtifact,
    _validate_coordinate_screen_execution_record,
    activation_capture_execution_receipt_body,
    coordinate_screen_condition_id,
    coordinate_screen_execution_receipt_body,
    coordinate_sparse_direction,
    decoder_feature_direction,
    e7_resumable_chain_head,
    evaluate_sae_corpus,
    load_activation_corpus,
    load_coordinate_sparse_artifact,
    load_sae,
    load_sae_intervention,
    sae_checkpoint_fingerprint,
    standardized_effect_size,
)
from mfh.provenance import sha256_file, sha256_path, stable_hash

_GATES = (
    "held_out_reconstruction",
    "feature_stability",
    "individual_causal_evidence",
    "protected_behavior_audit",
)
_SIDE_BEHAVIORS = {
    "ifeval": "instruction_following",
    "xstest": "safe_non_refusal",
    "strongreject_or_harmbench": "harmful_refusal",
    "language_consistency": "language_consistency",
}
_CAUSAL_BEHAVIORS = {*_SIDE_BEHAVIORS.values(), "abstention_association"}
_RETAINED_FRACTIONS = {0.01, 0.05, 0.10, 0.25}
_ACTIVE_MODEL = ACTIVE_MODEL_IDENTITIES[ACTIVE_MODEL_NAME]
_INTERPRETABILITY_CONTROLS = {
    "negative_alpha",
    "label_shuffled",
    "matched_random",
    "unrelated_layer",
    "gaussian",
    "zero_hook",
    "different_prompt",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")


def validate_separate_sae_corpus(
    directory: str | Path,
    *,
    evaluation_question_ids: set[str] | frozenset[str] = frozenset(),
) -> tuple[ActivationCorpus, ActivationCorpus, str]:
    """Validate the separately sampled, source-bound SAE train/validation corpus."""

    requested = Path(directory)
    if requested.is_symlink():
        raise FrozenArtifactError("E7 separate SAE corpus root cannot be a symlink")
    root = requested.resolve()
    questions_root = root / "questions"
    if (
        root.is_symlink()
        or not root.is_dir()
        or {value.name for value in root.iterdir()}
        != {"train", "validation", "questions"}
        or questions_root.is_symlink()
        or not questions_root.is_dir()
        or {value.name for value in questions_root.iterdir()}
        != {"train.jsonl", "validation.jsonl"}
        or any(
            value.is_symlink() or not value.is_file()
            for value in questions_root.iterdir()
        )
    ):
        raise FrozenArtifactError("E7 separate SAE corpus inventory differs")
    training = load_activation_corpus(root / "train")
    validation = load_activation_corpus(root / "validation")
    train_questions = tuple(read_questions(questions_root / "train.jsonl"))
    validation_questions = tuple(read_questions(questions_root / "validation.jsonl"))
    train_ids = {value.question_id for value in train_questions}
    validation_ids = {value.question_id for value in validation_questions}
    source_sha = sha256_path(questions_root)
    evaluation_ids = set(evaluation_question_ids)
    if (
        not train_questions
        or not validation_questions
        or len(train_ids) != len(train_questions)
        or len(validation_ids) != len(validation_questions)
        or train_ids & validation_ids
        or train_ids != training.all_question_ids()
        or validation_ids != validation.all_question_ids()
        or (train_ids | validation_ids) & evaluation_ids
        or not training.all_group_ids().isdisjoint(validation.all_group_ids())
        or training.schema_version != 2
        or validation.schema_version != 2
        or training.feature_schema.partition != "sae-train"
        or validation.feature_schema.partition != "sae-validation"
        or not training.feature_schema.is_compatible_extraction(
            validation.feature_schema
        )
        or training.runtime_artifact_sha256
        != validation.runtime_artifact_sha256
        or training.execution_public_key != validation.execution_public_key
        or training.source_question_bundle_sha256 != source_sha
        or validation.source_question_bundle_sha256 != source_sha
    ):
        raise DataValidationError(
            "E7 SAE corpus is not a disjoint, separately sampled source bundle"
        )
    return training, validation, source_sha


def execute_e7_activation_capture_batch(
    *,
    attestor: Any,
    runtime_artifact: str | Path,
    questions: tuple[Question, ...],
    prompt: PromptSpec,
    outcomes: tuple[Outcome, ...],
    group_ids: tuple[str, ...],
    feature_schema: Any,
    source_question_bundle: str | Path,
    dtype: str = "float16",
) -> ActivationBatch:
    """Capture and sign one SAE activation batch directly through native MLX hooks."""

    from mfh.experiments.e6_likelihood import E6RuntimeAttestor
    from mfh.methods.features import ActivationFeatureSchema

    if type(attestor) is not E6RuntimeAttestor or type(feature_schema) is not (
        ActivationFeatureSchema
    ):
        raise DataValidationError("E7 activation capture requires exact runtime/schema objects")
    if (
        dtype not in {"float16", "float32"}
        or len(questions) == 0
        or len(questions) != len(outcomes)
        or len(questions) != len(group_ids)
        or feature_schema.layers is None
        or len(feature_schema.layers) != 1
        or feature_schema.sites is None
        or len(feature_schema.sites) != 1
        or feature_schema.prompt_id != prompt.prompt_id
        or feature_schema.prompt_sha256
        != hashlib.sha256(prompt.text.encode("utf-8")).hexdigest()
        or feature_schema.partition not in {"sae-train", "sae-validation"}
    ):
        raise DataValidationError("E7 activation capture geometry is invalid")
    source = Path(source_question_bundle).resolve()
    source_sha = sha256_path(source)
    benchmark_questions: dict[tuple[str, str], Question] = {}
    for path in source.glob("*.jsonl"):
        for question in read_questions(path):
            benchmark_questions[(question.benchmark, question.question_id)] = question
    if any(
        benchmark_questions.get((question.benchmark, question.question_id)) != question
        for question in questions
    ):
        raise DataValidationError("E7 activation questions differ from the frozen bundle")
    runtime_sha = attestor.verify_runtime_artifact(runtime_artifact)
    layer = feature_schema.layers[0]
    site = feature_schema.sites[0]
    numpy_dtype = np.float16 if dtype == "float16" else np.float32
    rows: list[np.ndarray[Any, Any]] = []
    receipts: list[Mapping[str, Any]] = []
    for question, outcome, group_id in zip(
        questions, outcomes, group_ids, strict=True
    ):
        rendered = attestor.runtime.render_prompt(
            prompt, question.text, metadata=question.metadata
        )
        captured = attestor.runtime.prompt_feature_cube(
            rendered, layers=(layer,), sites=(site,)
        )
        activation = captured.activations[site][layer]
        value = np.ascontiguousarray(np.asarray(activation)[0].astype(numpy_dtype))
        if value.shape != (feature_schema.width,) or not np.isfinite(value).all():
            raise DataValidationError("E7 native activation capture width differs")
        body = activation_capture_execution_receipt_body(
            question_id=question.question_id,
            group_id=group_id,
            outcome=outcome,
            rendered_prompt_sha256=rendered.sha256,
            activation_sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest(),
            feature_schema=feature_schema,
            runtime_artifact_sha256=runtime_sha,
            execution_public_key=attestor.execution_public_key,
            source_question_bundle_sha256=source_sha,
            dtype=dtype,
        )
        rows.append(value.astype(np.float32))
        receipts.append(
            {
                "body": body,
                "signature": attestor._sign(body),
            }
        )
    return ActivationBatch(
        question_ids=tuple(question.question_id for question in questions),
        activations=torch.from_numpy(np.stack(rows)),
        outcomes=outcomes,
        group_ids=group_ids,
        capture_receipts=tuple(receipts),
    )


def execute_e7_tsteer_activation_capture_batch(
    *,
    attestor: Any,
    runtime_artifact: str | Path,
    questions: tuple[Question, ...],
    prompt: PromptSpec,
    group_ids: tuple[str, ...],
    feature_schema: Any,
    source_question_bundle: str | Path,
    dtype: str = "float16",
    max_new_tokens: int = 32,
) -> ActivationBatch:
    """Generate, grade, capture, and sign T-steer rows without caller labels."""

    from mfh.experiments.e6_likelihood import E6RuntimeAttestor
    from mfh.methods.features import ActivationFeatureSchema

    if type(attestor) is not E6RuntimeAttestor or type(feature_schema) is not (
        ActivationFeatureSchema
    ):
        raise DataValidationError("E7 T-steer capture requires exact runtime/schema objects")
    if (
        dtype not in {"float16", "float32"}
        or not questions
        or len(questions) != len(group_ids)
        or type(max_new_tokens) is not int
        or max_new_tokens <= 0
        or feature_schema.partition != "T-steer"
        or feature_schema.layers is None
        or len(feature_schema.layers) != 1
        or feature_schema.sites is None
        or len(feature_schema.sites) != 1
        or feature_schema.prompt_id != prompt.prompt_id
        or feature_schema.prompt_sha256
        != hashlib.sha256(prompt.text.encode("utf-8")).hexdigest()
    ):
        raise DataValidationError("E7 T-steer activation geometry is invalid")
    source = Path(source_question_bundle).resolve()
    source_sha = sha256_path(source)
    source_paths = (source,) if source.is_file() else tuple(source.glob("*.jsonl"))
    source_questions = {
        (question.benchmark, question.question_id): question
        for path in source_paths
        for question in read_questions(path)
    }
    if any(
        source_questions.get((question.benchmark, question.question_id)) != question
        for question in questions
    ):
        raise DataValidationError("E7 T-steer questions differ from the frozen bundle")
    runtime_sha = attestor.verify_runtime_artifact(runtime_artifact)
    layer = feature_schema.layers[0]
    site = feature_schema.sites[0]
    numpy_dtype = np.float16 if dtype == "float16" else np.float32
    rows: list[np.ndarray[Any, Any]] = []
    outcomes: list[Outcome] = []
    receipts: list[Mapping[str, Any]] = []
    for question, group_id in zip(questions, group_ids, strict=True):
        rendered = attestor.runtime.render_prompt(
            prompt, question.text, metadata=question.metadata
        )
        generated = attestor.runtime.generate(rendered, max_new_tokens=max_new_tokens)
        outcome = deterministic_short_answer_grade(generated.text, question.aliases)
        captured = attestor.runtime.prompt_feature_cube(
            rendered, layers=(layer,), sites=(site,)
        )
        value = np.ascontiguousarray(
            np.asarray(captured.activations[site][layer])[0].astype(numpy_dtype)
        )
        if value.shape != (feature_schema.width,) or not np.isfinite(value).all():
            raise DataValidationError("E7 T-steer native activation width differs")
        question_sha = stable_hash(
            {
                "question_id": question.question_id,
                "benchmark": question.benchmark,
                "text": question.text,
                "aliases": list(question.aliases),
                "split": question.split,
                "entities": list(question.entities),
                "metadata": dict(question.metadata),
            }
        )
        label_evidence = {
            "raw_output": generated.text,
            "normalized_answer": normalize_answer(generated.text),
            "aliases": list(question.aliases),
            "source_question_sha256": question_sha,
        }
        body = activation_capture_execution_receipt_body(
            question_id=question.question_id,
            group_id=group_id,
            outcome=outcome,
            rendered_prompt_sha256=rendered.sha256,
            activation_sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest(),
            feature_schema=feature_schema,
            runtime_artifact_sha256=runtime_sha,
            execution_public_key=attestor.execution_public_key,
            source_question_bundle_sha256=source_sha,
            dtype=dtype,
            label_evidence=label_evidence,
        )
        rows.append(value.astype(np.float32))
        outcomes.append(outcome)
        receipts.append({"body": body, "signature": attestor._sign(body)})
    return ActivationBatch(
        question_ids=tuple(question.question_id for question in questions),
        activations=torch.from_numpy(np.stack(rows)),
        outcomes=tuple(outcomes),
        group_ids=group_ids,
        capture_receipts=tuple(receipts),
    )


def execute_coordinate_screen_generation(
    *,
    attestor: Any,
    runtime_artifact: str | Path,
    source_question_bundle: str | Path,
    question: Question,
    prompt: PromptSpec,
    prompt_template_sha256: str,
    generation_record: GenerationRecord,
    contract_digest: str,
    source_artifact_sha256: str,
    retained_fraction: float | None,
    direction: torch.Tensor | np.ndarray[Any, Any] | None,
    reference_rms: float | None,
    layer: int | None,
    site: ActivationSite | None,
    token_scope: TokenScope | None,
    alpha: float,
    max_new_tokens: int = 32,
    populate_generation: bool = False,
) -> GenerationRecord:
    """Execute and runtime-sign one preregistered internal M4a screen row."""

    from mfh.experiments.e6_likelihood import E6RuntimeAttestor
    from mfh.experiments.e8_protected import question_source_fingerprint
    from mfh.inference.mlx_research import MlxResearchInterventionState
    from mfh.inference.mlx_runtime import MlxGenerationOutput

    if type(attestor) is not E6RuntimeAttestor:
        raise DataValidationError("coordinate screen requires the exact MLX attestor")
    if (
        not _SHA256.fullmatch(prompt_template_sha256)
        or hashlib.sha256(prompt.text.encode()).hexdigest()
        != prompt_template_sha256
    ):
        raise DataValidationError("coordinate screen prompt differs from its frozen template")
    question_source = Path(source_question_bundle).resolve()
    frozen_question = next(
        (
            value
            for path in question_source.glob("*.jsonl")
            for value in read_questions(path)
            if value.benchmark == question.benchmark
            and value.question_id == question.question_id
        ),
        None,
    )
    if frozen_question != question:
        raise DataValidationError("coordinate screen question differs from its frozen bundle")
    runtime_sha = attestor.verify_runtime_artifact(runtime_artifact)
    intervened = retained_fraction is not None
    expected_method = "M4a" if intervened else "M0"
    expected_condition_id = coordinate_screen_condition_id(
        contract_digest,
        retained_fraction=retained_fraction,
        alpha=alpha if intervened else None,
    )
    if (
        generation_record.question_id != question.question_id
        or generation_record.system_prompt_id != prompt.prompt_id
        or generation_record.steering_method != expected_method
        or generation_record.condition_id != expected_condition_id
        or generation_record.layer != (layer if intervened else None)
        or generation_record.site is not (site if intervened else None)
        or generation_record.token_scope is not (token_scope if intervened else None)
        or generation_record.alpha != (alpha if intervened else 0.0)
        or generation_record.sparsity != (retained_fraction if intervened else None)
        or isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens <= 0
        or any(
            generation_record.metadata.get(name) is not None
            for name in (
                "intervention_trace",
                "intervention_trace_digest",
                "coordinate_screen_execution_signature",
                "coordinate_screen_runtime_artifact_sha256",
                "coordinate_screen_execution_public_key",
                "generation_runtime_metrics",
            )
        )
    ):
        raise DataValidationError("coordinate screen row differs from its frozen cell")
    rendered = attestor.runtime.render_prompt(
        prompt, question.text, metadata=question.metadata
    )
    state: Any = None
    normalized_direction: np.ndarray[Any, Any] | None = None
    direction_norm = 0.0
    interventions: dict[tuple[int, ActivationSite], Any] = {}
    if intervened:
        if (
            layer is None
            or site is None
            or token_scope is None
            or retained_fraction not in _RETAINED_FRACTIONS
            or alpha not in {0.1, 0.25, 0.5, 1.0, 2.0}
            or isinstance(reference_rms, bool)
            or not isinstance(reference_rms, int | float)
            or not math.isfinite(float(reference_rms))
            or float(reference_rms) <= 0
        ):
            raise DataValidationError("coordinate screen intervention geometry is invalid")
        values = np.asarray(direction, dtype=np.float32)
        direction_norm = float(np.linalg.norm(values)) if values.ndim == 1 else math.nan
        if (
            values.ndim != 1
            or values.size == 0
            or not np.isfinite(values).all()
            or not math.isfinite(direction_norm)
            or direction_norm <= 0
        ):
            raise DataValidationError("coordinate screen direction is invalid")
        normalized_direction = np.ascontiguousarray(values / direction_norm)
        state = attestor.runtime.standardized_intervention_state(
            normalized_direction,
            standardized_alpha=alpha * direction_norm,
            reference_rms=float(reference_rms),
            token_scope=token_scope,
        )
        interventions[(layer, site)] = state
    elif any(
        value is not None for value in (direction, reference_rms, layer, site, token_scope)
    ) or alpha != 0.0:
        raise DataValidationError("coordinate screen baseline cannot receive intervention material")
    generated = attestor.runtime.generate_with_interventions(
        rendered,
        max_new_tokens=max_new_tokens,
        intervention_states=interventions,
    )
    if populate_generation:
        if type(generated) is not MlxGenerationOutput:
            raise DataValidationError("coordinate screen returned an invalid generation")
        if (
            generation_record.raw_output
            or generation_record.normalized_answer
            or generation_record.input_tokens != 0
            or generation_record.output_tokens != 0
            or generation_record.generation_latency_seconds != 0
            or generation_record.outcome is not Outcome.INCORRECT
        ):
            raise DataValidationError(
                "coordinate populated execution requires an empty draft row"
            )
        generation_record = replace(
            generation_record,
            raw_output=generated.text,
            normalized_answer=normalize_answer(generated.text),
            outcome=deterministic_short_answer_grade(
                generated.text, question.aliases
            ),
            generation_latency_seconds=generated.latency_seconds,
            input_tokens=generated.input_tokens,
            output_tokens=generated.output_tokens,
        )
    if (
        type(generated) is not MlxGenerationOutput
        or generated.rendered_prompt != rendered
        or generation_record.rendered_prompt_hash != rendered.sha256
        or generation_record.raw_output != generated.text
        or generation_record.input_tokens != generated.input_tokens
        or generation_record.output_tokens != generated.output_tokens
        or generation_record.normalized_answer != normalize_answer(generated.text)
        or generation_record.outcome
        is not deterministic_short_answer_grade(generated.text, question.aliases)
    ):
        raise DataValidationError("coordinate screen output differs from the native runtime")
    metadata: dict[str, Any] = {
        **dict(generation_record.metadata),
        "coordinate_screen_contract_digest": contract_digest,
        "coordinate_screen_runtime_artifact_sha256": runtime_sha,
        "coordinate_screen_execution_public_key": attestor.execution_public_key,
        "prompt_template_sha256": prompt_template_sha256,
        "source_question_sha256": question_source_fingerprint(question),
        "generation_runtime_metrics": build_generation_runtime_metrics(
            generated,
            runtime_identity=attestor.attested_runtime_identity,
        ),
    }
    if intervened:
        assert isinstance(state, MlxResearchInterventionState)
        assert normalized_direction is not None
        assert layer is not None and site is not None and token_scope is not None
        assert reference_rms is not None and retained_fraction is not None
        captured = as_numpy(state.captured, dtype=np.float32)
        edited = as_numpy(state.intervened, dtype=np.float32)
        expected_indices = (
            [-1]
            if token_scope is TokenScope.FINAL_PROMPT
            else list(
                range(
                    min(
                        {
                            TokenScope.FIRST_GENERATED: 1,
                            TokenScope.FIRST_FOUR: 4,
                            TokenScope.FIRST_EIGHT: 8,
                            TokenScope.ALL_GENERATED: generated.output_tokens,
                        }[token_scope],
                        generated.output_tokens,
                    )
                )
            )
        )
        if (
            captured.shape != edited.shape
            or captured.size == 0
            or not np.isfinite(captured).all()
            or not np.isfinite(edited).all()
            or np.array_equal(captured, edited)
            or state.applications != len(expected_indices)
            or not expected_indices
        ):
            raise DataValidationError("coordinate screen runtime did not apply a material edit")
        delta = np.ascontiguousarray(edited - captured)
        trace = {
            "coordinate_screen_contract_digest": contract_digest,
            "source_artifact_sha256": source_artifact_sha256,
            "direction_sha256": hashlib.sha256(
                normalized_direction.tobytes(order="C")
            ).hexdigest(),
            "layer": layer,
            "site": site.value,
            "token_scope": token_scope.value,
            "standardized_alpha": alpha,
            "raw_alpha": state.alpha,
            "retained_fraction": retained_fraction,
            "reference_rms": float(reference_rms),
            "source_direction_norm": direction_norm,
            "applied_tokens": state.applications,
            "applied_token_indices": expected_indices,
            "pre_activation_sha256": hashlib.sha256(
                np.ascontiguousarray(captured).tobytes(order="C")
            ).hexdigest(),
            "post_activation_sha256": hashlib.sha256(
                np.ascontiguousarray(edited).tobytes(order="C")
            ).hexdigest(),
            "delta_sha256": hashlib.sha256(delta.tobytes(order="C")).hexdigest(),
        }
        metadata.update(
            {
                "intervention_trace": trace,
                "intervention_trace_digest": stable_hash(trace),
            }
        )
    executed = replace(
        generation_record,
        generation_latency_seconds=generated.latency_seconds,
        metadata=metadata,
    )
    return replace(
        executed,
        metadata={
            **dict(executed.metadata),
            "coordinate_screen_execution_signature": attestor._sign(
                coordinate_screen_execution_receipt_body(
                    executed,
                    contract_digest=contract_digest,
                    runtime_artifact_sha256=runtime_sha,
                )
            ),
        },
    )


@dataclass(frozen=True, slots=True)
class _E7FinalInputs:
    ledger: PhaseRunLedger
    coordinate: CoordinateSparseArtifact
    sae: SAEInterventionArtifact
    coordinate_sha256: str
    sae_sha256: str
    package_lock_sha256: str
    input_paths: Mapping[str, Path]


def execute_e7_generation(
    *,
    attestor: Any,
    runtime_artifact: str | Path,
    source_question_bundle: str | Path,
    question: Question,
    prompt: PromptSpec,
    generation_record: GenerationRecord,
    condition: Any,
    coordinate_artifact: str | Path,
    sae_intervention: str | Path,
    max_new_tokens: int = 32,
    populate_generation: bool = False,
    generation_grader: Callable[[GenerationRecord], GenerationRecord] | None = None,
) -> GenerationRecord:
    """Run one final E7 M0/M4a/M4b row with exact promoted material."""

    from mfh.experiments.e8_protected import execute_e8_generation
    from mfh.experiments.runner import EvaluationCondition

    if type(condition) is not EvaluationCondition or condition.phase is not ExperimentPhase.E7:
        raise DataValidationError("E7 execution requires one exact E7 condition")
    source = Path(source_question_bundle).resolve()
    frozen_question = next(
        (
            value
            for path in source.glob("*.jsonl")
            for value in read_questions(path)
            if value.benchmark == question.benchmark
            and value.question_id == question.question_id
        ),
        None,
    )
    if frozen_question != question:
        raise DataValidationError("E7 generation question differs from its frozen bundle")
    coordinate_path = Path(coordinate_artifact).resolve()
    sae_path = Path(sae_intervention).resolve()
    coordinate = load_coordinate_sparse_artifact(coordinate_path)
    sae = load_sae_intervention(sae_path)
    direction: torch.Tensor | None
    reference_rms: float | None
    expected_artifact: str | None
    if condition.steering_method == "M0":
        direction = None
        reference_rms = None
        expected_artifact = None
    elif condition.steering_method == "M4a":
        direction = coordinate.sparse_direction.direction
        reference_rms = coordinate.reference_rms
        expected_artifact = sha256_path(coordinate_path)
    elif condition.steering_method == "M4b":
        direction = sae.decoded_direction
        reference_rms = coordinate.reference_rms
        expected_artifact = sha256_path(sae_path)
    else:
        raise DataValidationError("E7 final executor supports only M0/M4a/M4b")
    if condition.method_artifact_sha256 != expected_artifact:
        raise DataValidationError("E7 condition differs from its promoted artifact")
    return execute_e8_generation(
        attestor=attestor,
        runtime_artifact=runtime_artifact,
        question=question,
        prompt=prompt,
        generation_record=generation_record,
        condition=condition,
        direction=direction,
        reference_rms=reference_rms,
        max_new_tokens=max_new_tokens,
        populate_generation=populate_generation,
        generation_grader=generation_grader,
    )


def e7_causal_feature_condition_id(
    *,
    feature_index: int,
    mode: str,
    feature_schema: Any,
    layer: int,
    site: ActivationSite,
    token_scope: TokenScope,
    alpha: float,
    runtime_artifact_sha256: str,
    execution_public_key: str,
    source_question_bundle_sha256: str,
    sae_intervention_sha256: str,
) -> str:
    if mode not in {"baseline", "activated", "suppressed"}:
        raise DataValidationError("E7 causal condition mode is invalid")
    return stable_hash(
        {
            "schema_version": 1,
            "feature_index": feature_index,
            "mode": mode,
            "feature_schema": feature_schema.to_dict(),
            "layer": layer,
            "site": site.value,
            "token_scope": token_scope.value,
            "alpha": alpha,
            "runtime_artifact_sha256": runtime_artifact_sha256,
            "execution_public_key": execution_public_key,
            "source_question_bundle_sha256": source_question_bundle_sha256,
            "sae_intervention_sha256": sae_intervention_sha256,
        }
    )


def execute_e7_causal_feature_generation(
    *,
    attestor: Any,
    runtime_artifact: str | Path,
    question: Question,
    prompt: PromptSpec,
    generation_record: GenerationRecord,
    condition: Any,
    sae_training: str | Path,
    feature_schema: Any,
    source_question_bundle: str | Path,
    feature_index: int,
    mode: str,
    layer: int,
    site: ActivationSite,
    token_scope: TokenScope,
    alpha: float,
    reference_rms: float,
    max_new_tokens: int = 32,
    populate_generation: bool = False,
    generation_grader: Callable[[GenerationRecord], GenerationRecord] | None = None,
) -> GenerationRecord:
    """Run a signed baseline, activation, or suppression row for one SAE feature."""

    from mfh.experiments.e8_protected import execute_e8_generation
    from mfh.experiments.runner import EvaluationCondition

    if type(condition) is not EvaluationCondition or condition.phase is not ExperimentPhase.E7:
        raise DataValidationError("E7 causal execution requires an exact E7 condition")
    if mode not in {"baseline", "activated", "suppressed"}:
        raise DataValidationError("E7 causal execution mode is invalid")
    expected_method = "M0" if mode == "baseline" else "M4b"
    sae_path = Path(sae_training).resolve()
    training = load_sae(sae_path)
    question_source = Path(source_question_bundle).resolve()
    source_question = next(
        (
            value
            for path in question_source.glob("*.jsonl")
            for value in read_questions(path)
            if value.benchmark == question.benchmark
            and value.question_id == question.question_id
        ),
        None,
    )
    if source_question != question:
        raise DataValidationError("E7 causal question differs from its frozen bundle")
    source_question_sha = sha256_path(question_source)
    runtime_sha = attestor.verify_runtime_artifact(runtime_artifact)
    checkpoint_sha = sae_checkpoint_fingerprint(training)
    if (
        condition.steering_method != expected_method
        or condition.method_artifact_sha256
        != (None if mode == "baseline" else checkpoint_sha)
        or (
            mode != "baseline"
            and (
                condition.layer != layer
                or condition.site is not site
                or condition.token_scope is not token_scope
                or condition.alpha != alpha
            )
        )
    ):
        raise DataValidationError("E7 causal condition differs from its feature mode")
    direction = None
    rms = None
    if mode != "baseline":
        direction = decoder_feature_direction(training.model, feature_index)
        if mode == "suppressed":
            direction = -direction
        rms = reference_rms
    executed = execute_e8_generation(
        attestor=attestor,
        runtime_artifact=runtime_artifact,
        question=question,
        prompt=prompt,
        generation_record=generation_record,
        condition=condition,
        direction=direction,
        reference_rms=rms,
        max_new_tokens=max_new_tokens,
        populate_generation=populate_generation,
        generation_grader=generation_grader,
    )
    causal_execution_id = e7_causal_feature_condition_id(
        feature_index=feature_index,
        mode=mode,
        feature_schema=feature_schema,
        layer=layer,
        site=site,
        token_scope=token_scope,
        alpha=alpha,
        runtime_artifact_sha256=runtime_sha,
        execution_public_key=attestor.execution_public_key,
        source_question_bundle_sha256=source_question_sha,
        sae_intervention_sha256=checkpoint_sha,
    )
    return replace(
        executed,
        metadata={
            **dict(executed.metadata),
            "e7_causal_feature_index": feature_index,
            "e7_causal_mode": mode,
            "e7_causal_execution_id": causal_execution_id,
        },
    )


def _e7_interpretability_direction(
    *,
    audit_name: str,
    training: Any,
    selection: ProbeDataset,
    feature_index: int,
    seed: int,
) -> torch.Tensor | None:
    feature_direction = decoder_feature_direction(training.model, feature_index)
    if audit_name in {
        "P0-neutral",
        "P2-calibrated-abstention",
        "negative_alpha",
        "unrelated_layer",
        "different_prompt",
    }:
        return feature_direction
    if audit_name == "label_shuffled":
        return label_shuffled_centroid_direction(
            selection.features, selection.outcomes, seed=seed
        )
    if audit_name == "matched_random":
        return matched_random_direction(feature_direction, seed=seed)
    if audit_name == "gaussian":
        return norm_matched_gaussian_perturbation(feature_direction, seed=seed)
    if audit_name == "zero_hook":
        return None
    raise DataValidationError("E7 interpretability audit name is not registered")


def _e7_interpretability_prompt_id(
    audit_name: str, *, selection_prompt_id: str
) -> str:
    if audit_name in {"P0-neutral", "P2-calibrated-abstention"}:
        return audit_name
    if audit_name == "different_prompt":
        alternatives = {
            "P0-neutral": "P2-calibrated-abstention",
            "P2-calibrated-abstention": "P0-neutral",
        }
        try:
            return alternatives[selection_prompt_id]
        except KeyError as exc:
            raise DataValidationError(
                "E7 different-prompt control requires P0 or P2 selection"
            ) from exc
    return selection_prompt_id


def execute_e7_interpretability_generation(
    *,
    attestor: Any,
    runtime_artifact: str | Path,
    question: Question,
    prompt: PromptSpec,
    generation_record: GenerationRecord,
    condition: Any,
    sae_training: str | Path,
    selection: ProbeDataset,
    source_question_bundle: str | Path,
    feature_index: int,
    audit_name: str,
    mode: str,
    layer: int,
    site: ActivationSite,
    token_scope: TokenScope,
    alpha: float,
    reference_rms: float,
    max_new_tokens: int = 32,
    populate_generation: bool = False,
    generation_grader: Callable[[GenerationRecord], GenerationRecord] | None = None,
) -> GenerationRecord:
    """Execute one source-bound prompt-transfer or registered control row."""

    from mfh.experiments.e8_protected import execute_e8_generation
    from mfh.experiments.runner import EvaluationCondition

    if type(condition) is not EvaluationCondition or condition.phase is not ExperimentPhase.E7:
        raise DataValidationError("E7 interpretability execution requires an exact condition")
    if audit_name not in _INTERPRETABILITY_CONTROLS | {
        "P0-neutral",
        "P2-calibrated-abstention",
    }:
        raise DataValidationError("E7 interpretability audit is not registered")
    if mode not in {"baseline", "intervention"}:
        raise DataValidationError("E7 interpretability mode is invalid")
    training = load_sae(Path(sae_training).resolve())
    if (
        selection.feature_schema is None
        or selection.feature_schema.partition != "T-steer"
        or not training.training_schema.is_compatible_representation(
            selection.feature_schema
        )
    ):
        raise DataValidationError("E7 interpretability selection dataset differs")
    question_root = Path(source_question_bundle).resolve()
    source_question = next(
        (
            value
            for path in question_root.glob("*.jsonl")
            for value in read_questions(path)
            if value.benchmark == question.benchmark
            and value.question_id == question.question_id
        ),
        None,
    )
    if source_question != question:
        raise DataValidationError("E7 interpretability question differs from its source")
    checkpoint_sha = sae_checkpoint_fingerprint(training)
    expected_prompt_id = _e7_interpretability_prompt_id(
        audit_name, selection_prompt_id=selection.feature_schema.prompt_id
    )
    expected_method = (
        "M0" if mode == "baseline" or audit_name == "zero_hook" else "M4b"
    )
    expected_layer: int | None = (
        (layer + 1) % condition.model_num_layers
        if audit_name == "unrelated_layer"
        else layer
    )
    expected_alpha = -abs(alpha) if audit_name == "negative_alpha" else alpha
    if expected_method == "M0":
        expected_layer = None
        expected_site = None
        expected_scope = None
        expected_alpha = 0.0
        expected_artifact = None
    else:
        expected_site = site
        expected_scope = token_scope
        expected_artifact = checkpoint_sha
    if (
        prompt.prompt_id != expected_prompt_id
        or condition.prompt_template_sha256
        != hashlib.sha256(prompt.text.encode("utf-8")).hexdigest()
        or condition.system_prompt_id != expected_prompt_id
        or condition.steering_method != expected_method
        or condition.method_artifact_sha256 != expected_artifact
        or condition.layer != expected_layer
        or condition.site is not expected_site
        or condition.token_scope is not expected_scope
        or condition.alpha != expected_alpha
        or condition.sparsity is not None
        or condition.comparison_group
        != f"e7-interpretability-{feature_index}-{audit_name}-{mode}"
    ):
        raise DataValidationError("E7 interpretability condition differs")
    prepared = replace(
        generation_record,
        metadata={
            **dict(generation_record.metadata),
            "e7_interpretability_condition": condition.to_dict(),
            "e7_interpretability_feature_index": feature_index,
            "e7_interpretability_audit_name": audit_name,
            "e7_interpretability_mode": mode,
            "e7_interpretability_selection_fingerprint": (
                selection.data_fingerprint
            ),
        },
    )
    direction = (
        None
        if expected_method == "M0"
        else _e7_interpretability_direction(
            audit_name=audit_name,
            training=training,
            selection=selection,
            feature_index=feature_index,
            seed=condition.seed,
        )
    )
    return execute_e8_generation(
        attestor=attestor,
        runtime_artifact=runtime_artifact,
        question=question,
        prompt=prompt,
        generation_record=prepared,
        condition=condition,
        direction=direction,
        reference_rms=None if direction is None else reference_rms,
        max_new_tokens=max_new_tokens,
        populate_generation=populate_generation,
        generation_grader=generation_grader,
    )


def _e7_input_paths(ledger: PhaseRunLedger) -> Mapping[str, Path]:
    PhaseRunLedger._verify_creation_evidence(ledger)
    try:
        payload = json.loads(
            (ledger.directory / "creation-evidence.json").read_text(encoding="utf-8")
        )
        descriptors = payload["input_artifacts"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise FrozenArtifactError(f"cannot read E7 creation evidence: {exc}") from exc
    if not isinstance(descriptors, Mapping) or set(descriptors) != {
        "E3_static_vectors",
        "E5_adaptive_controllers",
        "separate_sae_corpus",
        "frozen_sae_seed_runs",
        "frozen_tsteer_questions",
        "frozen_side_effect_scorers",
    }:
        raise FrozenArtifactError("E7 input artifact inventory differs")
    paths: dict[str, Path] = {}
    for name, descriptor in descriptors.items():
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
            "location",
            "fingerprint",
        }:
            raise FrozenArtifactError("E7 input descriptor differs")
        raw = Path(descriptor["location"])
        path = raw if raw.is_absolute() else (ledger.directory / raw).resolve()
        if (
            descriptor["fingerprint"] != ledger.contract.input_fingerprints[name]
            or sha256_path(path) != descriptor["fingerprint"]
        ):
            raise FrozenArtifactError("E7 frozen input changed")
        paths[name] = path
    return MappingProxyType(paths)


def _verified_e6_prerequisite_material(
    ledger: PhaseRunLedger,
) -> tuple[str, str, str, str, Mapping[str, Any]]:
    """Replay E6's packaged gate bundle and recover its trusted E3/runtime identities."""

    from mfh.experiments.e6_likelihood import (
        _load_e6_runtime_attestation,
        verify_e6_gate_artifact,
    )

    try:
        creation = json.loads(
            (ledger.directory / "creation-evidence.json").read_text(encoding="utf-8")
        )
        descriptor = creation["prerequisite_runs"][ExperimentPhase.E6.value]
        raw_prerequisite = Path(str(descriptor["location"]))
        prerequisite_path = (
            raw_prerequisite.resolve()
            if raw_prerequisite.is_absolute()
            else (ledger.directory / raw_prerequisite).resolve()
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise FrozenArtifactError(f"E7 lacks its E6 prerequisite evidence: {exc}") from exc
    e6 = PhaseRunLedger.open(prerequisite_path, study=ledger.study)
    completion = e6.verify_complete()
    expected_completion = ledger.contract.prerequisite_digests[ExperimentPhase.E6.value]
    gate = "knowledge_recovery_separated_from_abstention_substitution"
    bundle = e6.directory / "gate-artifacts" / gate / "likelihood-bundle"
    context = e6._gate_context()
    if completion.completion_digest != expected_completion:
        raise FrozenArtifactError("E7 E6 prerequisite completion differs")
    verified = verify_e6_gate_artifact(
        bundle,
        contract_digest=e6.contract.digest,
        record_set_digest=e6.record_set_digest(),
        generation_records=tuple(e6.records()),
        condition_facts=context.condition_facts,
        input_fingerprints=e6.contract.input_fingerprints,
        frozen_inputs_verified=context.frozen_inputs_verified,
    )
    manifest = verified["manifest"]
    if not isinstance(manifest, Mapping):
        raise FrozenArtifactError("E7 E6 gate artifact lacks a manifest")
    e3_sha = manifest.get("e3_static_vectors_sha256")
    runtime_sha = manifest.get("runtime_artifact_sha256")
    execution_key = manifest.get("execution_public_key")
    runtime_attestation = _load_e6_runtime_attestation(bundle / "runtime-artifact")
    runtime_identity = runtime_attestation["runtime_identity"]
    if not isinstance(runtime_identity, Mapping):
        raise FrozenArtifactError("E7 E6 prerequisite lacks its runtime identity")
    model_snapshot_sha = runtime_identity.get("snapshot_sha256")
    if not all(
        isinstance(value, str) and len(value) == 64
        for value in (e3_sha, runtime_sha, execution_key, model_snapshot_sha)
    ):
        raise FrozenArtifactError("E7 E6 prerequisite material identities are invalid")
    return (
        str(e3_sha),
        str(runtime_sha),
        str(execution_key),
        str(model_snapshot_sha),
        MappingProxyType(dict(runtime_identity)),
    )


def _validate_e7_final_inputs(
    *,
    ledger_directory: str | Path,
    study: StudyProtocol,
    coordinate_artifact: str | Path,
    sae_intervention: str | Path,
    package_lock: str | Path,
) -> _E7FinalInputs:
    from mfh.evaluation.side_effects import load_side_effect_scorer_spec
    from mfh.experiments.e8_protected import question_source_fingerprint

    if type(study) is not StudyProtocol:
        raise DataValidationError("E7 finalization requires an exact StudyProtocol")
    ledger = PhaseRunLedger.open(ledger_directory, study=study)
    ledger.contract.assert_matches_study(study)
    completed, expected = ledger.progress()
    if ledger.contract.phase is not ExperimentPhase.E7 or completed != expected:
        raise DataValidationError("E7 final ledger is incomplete or cross-phase")
    paths = _e7_input_paths(ledger)
    validate_side_effect_evaluation_bundle(
        paths["frozen_side_effect_scorers"], ledger.contract
    )
    side_effect_scorer = load_side_effect_scorer_spec(
        paths["frozen_side_effect_scorers"]
    )
    coordinate_path = Path(coordinate_artifact).resolve()
    sae_path = Path(sae_intervention).resolve()
    coordinate = load_coordinate_sparse_artifact(coordinate_path)
    sae = load_sae_intervention(sae_path)
    coordinate_sha = sha256_path(coordinate_path)
    sae_sha = sha256_path(sae_path)
    package_lock_path = Path(package_lock).resolve()
    if package_lock_path.is_symlink() or not package_lock_path.is_file():
        raise FrozenArtifactError("E7 package lock must be a regular file")
    package_lock_sha = sha256_file(package_lock_path)
    stability = load_sae_stability_bundle(paths["frozen_sae_seed_runs"])
    evaluation_question_ids = {
        question_id
        for values in ledger.contract.question_ids_by_benchmark.values()
        for question_id in values
    }
    corpus_root = paths["separate_sae_corpus"]
    training_corpus, validation_corpus, sae_question_bundle_sha = (
        validate_separate_sae_corpus(
            corpus_root,
            evaluation_question_ids=evaluation_question_ids,
        )
    )
    recomputed_metrics = evaluate_sae_corpus(sae.training.model, validation_corpus)
    selection = stability.selection_datasets_by_model.get(_ACTIVE_MODEL["repository"])
    if selection is None or selection.feature_schema is None:
        raise FrozenArtifactError("E7 stability bundle lacks the active T-steer selection")
    correct = torch.tensor([outcome is Outcome.CORRECT for outcome in selection.outcomes])
    incorrect = torch.tensor([outcome is Outcome.INCORRECT for outcome in selection.outcomes])
    if not correct.any() or not incorrect.any():
        raise DataValidationError("E7 T-steer selection lacks C/I coordinate evidence")
    recomputed_effect = standardized_effect_size(
        selection.features[correct], selection.features[incorrect]
    )
    e3_index = _e3_direction_index(paths["E3_static_vectors"])
    schema = coordinate.feature_schema
    extraction = {
        ActivationKind.FINAL_PROMPT: "M1-P",
        ActivationKind.RESPONSE_TOKENS: "M1-R",
    }.get(schema.activation_kind)
    if (
        extraction is None
        or schema.layers != (coordinate.layer,)
        or schema.sites != (coordinate.site,)
        or schema.token_scope not in {None, coordinate.token_scope}
    ):
        raise DataValidationError("E7 coordinate geometry differs from its E3 tensor index")
    tensor_index = (schema.prompt_id, extraction, coordinate.site.value, coordinate.layer)
    if tensor_index not in e3_index:
        raise DataValidationError("E7 coordinate tensor index is absent from E3")
    try:
        metadata = json.loads(
            (paths["E3_static_vectors"] / "metadata.json").read_text(encoding="utf-8")
        )
        prompt_index = metadata["prompt_axis"].index(schema.prompt_id)
        extraction_index = metadata["extraction_axis"].index(extraction)
        site_index = metadata["site_axis"].index(coordinate.site.value)
        layer_index = metadata["layer_axis"].index(coordinate.layer)
        with np.load(
            paths["E3_static_vectors"] / "vectors.npz", allow_pickle=False
        ) as values:
            dense = np.asarray(
                values["directions"][
                    prompt_index, extraction_index, site_index, layer_index
                ]
            ).copy()
            reference_rms = float(
                values["reference_rms"][
                    prompt_index, extraction_index, site_index, layer_index
                ]
            )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot load E7 E3 tensor slice: {exc}") from exc
    if (
        dense.dtype != np.float32
        or dense.shape != (coordinate.feature_schema.width,)
        or hashlib.sha256(dense.tobytes(order="C")).hexdigest()
        != e3_index[tensor_index]
    ):
        raise FrozenArtifactError("E7 E3 tensor slice identity differs")
    candidate = coordinate_sparse_direction(
        torch.from_numpy(dense),
        recomputed_effect,
        retained_fraction=coordinate.sparse_direction.retained_fraction,
        renormalize=coordinate.sparse_direction.renormalized,
    )
    (
        e6_e3_sha,
        e6_runtime_sha,
        e6_execution_key,
        e6_model_snapshot_sha,
        e6_runtime_identity,
    ) = (
        _verified_e6_prerequisite_material(ledger)
    )
    selection_corpus = stability.selection_corpora_by_model.get(
        _ACTIVE_MODEL["repository"]
    )
    tsteer_source = paths["frozen_tsteer_questions"]
    if tsteer_source.is_symlink() or not tsteer_source.is_file():
        raise FrozenArtifactError("E7 T-steer schedule must be one regular JSONL file")
    tsteer_questions = tuple(read_questions(tsteer_source))
    tsteer_ids = tuple(question.question_id for question in tsteer_questions)
    tsteer_sha = sha256_file(tsteer_source)
    if (
        selection_corpus is None
        or selection_corpus.runtime_artifact_sha256 != e6_runtime_sha
        or selection_corpus.execution_public_key != e6_execution_key
        or selection_corpus.feature_schema != selection.feature_schema
        or selection_corpus.source_question_bundle_sha256 != tsteer_sha
        or selection_corpus.total_rows != 30_000
        or len(tsteer_questions) != 30_000
        or len(set(tsteer_ids)) != len(tsteer_ids)
        or any(
            question.benchmark != "triviaqa" or question.split != "T-steer"
            for question in tsteer_questions
        )
        or set(selection.question_ids) != set(tsteer_ids)
    ):
        raise DataValidationError(
            "E7 T-steer selection differs from the signed active-model capture"
        )
    exact_question_bundle_sha = sha256_path(
        paths["frozen_side_effect_scorers"] / "questions"
    )
    trivia_ids = set(ledger.contract.question_ids_by_benchmark["triviaqa"])
    trivia_questions = {
        question.question_id: question
        for question in read_questions(
            paths["frozen_side_effect_scorers"] / "questions" / "triviaqa.jsonl"
        )
    }
    frozen_questions = {
        (question.benchmark, question.question_id): question
        for path in (
            paths["frozen_side_effect_scorers"] / "questions"
        ).glob("*.jsonl")
        for question in read_questions(path)
    }
    source_question_bundle_sha = sae_question_bundle_sha
    promoted_checkpoint = sae_checkpoint_fingerprint(sae.training)
    sweep_metrics = tuple(
        evaluate_sae_corpus(result.model, validation_corpus)
        for result in sae.sparsity_sweep_results
    )
    expected_chain_head = e7_resumable_chain_head(
        training_corpus=training_corpus,
        validation_corpus=validation_corpus,
        sweep_results=sae.sparsity_sweep_results,
    )
    matching_seed_selections = tuple(
        item
        for item in stability.selections_by_model[_ACTIVE_MODEL["repository"]]
        if item.checkpoint_fingerprint == promoted_checkpoint
    )
    schemas = (
        coordinate.feature_schema,
        sae.training.training_schema,
        sae.training.validation_schema,
        sae.latent_direction.selection_schema,
    )
    if (
        coordinate.sparse_direction.retained_fraction not in _RETAINED_FRACTIONS
        or coordinate.source_artifact_sha256
        != ledger.contract.input_fingerprints["E3_static_vectors"]
        or coordinate.source_artifact_sha256 != e6_e3_sha
        or coordinate.screen_runtime_artifact_sha256 != e6_runtime_sha
        or coordinate.screen_execution_public_key != e6_execution_key
        or coordinate.screen_question_bundle_sha256 != exact_question_bundle_sha
        or any(
            set(point.question_ids) - trivia_ids for point in coordinate.screen_points
        )
        or set(trivia_questions) != trivia_ids
        or any(
            record.normalized_answer != normalize_answer(record.raw_output)
            or record.outcome
            is not deterministic_short_answer_grade(
                record.raw_output, trivia_questions[record.question_id].aliases
            )
            for record in (*coordinate.screen_records, *tuple(ledger.records()))
            if record.benchmark == "triviaqa"
        )
        or any(
            (record.benchmark, record.question_id) not in frozen_questions
            or record.metadata.get("source_question_sha256")
            != question_source_fingerprint(
                frozen_questions[(record.benchmark, record.question_id)]
            )
            for record in (*coordinate.screen_records, *tuple(ledger.records()))
        )
        or training_corpus.schema_version != 2
        or validation_corpus.schema_version != 2
        or training_corpus.runtime_artifact_sha256 != e6_runtime_sha
        or validation_corpus.runtime_artifact_sha256 != e6_runtime_sha
        or training_corpus.execution_public_key != e6_execution_key
        or validation_corpus.execution_public_key != e6_execution_key
        or training_corpus.source_question_bundle_sha256
        != source_question_bundle_sha
        or validation_corpus.source_question_bundle_sha256
        != source_question_bundle_sha
        or source_question_bundle_sha == exact_question_bundle_sha
        or (training_corpus.all_question_ids() | validation_corpus.all_question_ids())
        & set(selection.question_ids)
        or set(stability.promoted_method_artifacts) != {_ACTIVE_MODEL["repository"]}
        or stability.promoted_method_artifacts[_ACTIVE_MODEL["repository"]] != sae_sha
        or not np.isclose(
            sae.feature_stability,
            stability.stability_by_model[_ACTIVE_MODEL["repository"]],
            rtol=0,
            atol=1e-12,
        )
        or sae.stability_selections
        != stability.selections_by_model[_ACTIVE_MODEL["repository"]]
        or not torch.equal(candidate.mask, coordinate.sparse_direction.mask)
        or not torch.equal(candidate.direction, coordinate.sparse_direction.direction)
        or not torch.equal(coordinate.sparse_direction.effect_size, recomputed_effect)
        or coordinate.source_tensor_index != tensor_index
        or coordinate.source_direction_sha256 != e3_index[tensor_index]
        or not np.isclose(
            coordinate.reference_rms, reference_rms, rtol=0, atol=0
        )
        or coordinate.data_fingerprint != selection.data_fingerprint
        or coordinate.feature_schema != selection.feature_schema
        or sae.latent_direction.selection_fingerprint != selection.data_fingerprint
        or sae.latent_direction.selection_schema != selection.feature_schema
        or len(matching_seed_selections) != 1
        or matching_seed_selections[0].selected_features
        != sae.latent_direction.selected_features
        or sae.training.training_fingerprint != training_corpus.data_fingerprint
        or sae.training.validation_fingerprint != validation_corpus.data_fingerprint
        or sae.training.training_schema != training_corpus.feature_schema
        or sae.training.validation_schema != validation_corpus.feature_schema
        or sae.training.training_rows != training_corpus.total_rows
        or sae.training.validation_rows != validation_corpus.total_rows
        or any(
            not np.isclose(
                getattr(sae.training.metrics, name),
                getattr(recomputed_metrics, name),
                rtol=1e-6,
                atol=1e-8,
            )
            for name in (
                "reconstruction_mse",
                "fraction_variance_explained",
                "average_active_features",
            )
        )
        or any(
            schema.model_repository != _ACTIVE_MODEL["repository"]
            or schema.model_revision != _ACTIVE_MODEL["revision"]
            or schema.runtime is not _ACTIVE_MODEL["runtime"]
            or schema.quantization != _ACTIVE_MODEL["quantization"]
            for schema in schemas
        )
        or len({schema.split_manifest_digest for schema in schemas}) != 1
        or sae.training.training_schema.partition != "sae-train"
        or sae.training.validation_schema.partition != "sae-validation"
        or sae.latent_direction.selection_schema.partition != "T-steer"
        or len(sae.sparsity_sweep) < 3
        or len(sae.sparsity_sweep_results) != len(sae.sparsity_sweep)
        or any(
            any(
                not np.isclose(
                    getattr(recomputed, name),
                    getattr(result.metrics, name),
                    rtol=1e-6,
                    atol=1e-8,
                )
                for name in (
                    "reconstruction_mse",
                    "fraction_variance_explained",
                    "average_active_features",
                )
            )
            for result, recomputed in zip(
                sae.sparsity_sweep_results, sweep_metrics, strict=True
            )
        )
        or sae.interpretability_audit is None
        or sae.interpretability_audit.source_question_bundle_sha256
        != exact_question_bundle_sha
        or set(sae.interpretability_audit.evaluation_question_ids) - trivia_ids
        or sae.long_computation_receipt is None
        or sae.long_computation_receipt.package_lock_sha256
        != package_lock_sha
        or sae.long_computation_receipt.model_snapshot_sha256
        != e6_model_snapshot_sha
        or sae.long_computation_receipt.runtime_artifact_sha256
        != e6_runtime_sha
        or sae.long_computation_receipt.execution_public_key
        != e6_execution_key
        or sae.long_computation_receipt.training_corpus_sha256
        != sha256_path(corpus_root / "train")
        or sae.long_computation_receipt.validation_corpus_sha256
        != sha256_path(corpus_root / "validation")
        or sae.long_computation_receipt.resumable_chain_head
        != expected_chain_head
        or any(
            item.spec.runtime_artifact_sha256 != e6_runtime_sha
            or item.spec.execution_public_key != e6_execution_key
            or item.spec.source_question_bundle_sha256
            != exact_question_bundle_sha
            or set(item.spec.protected_sample_counts) != _CAUSAL_BEHAVIORS
            or item.native_execution_records is None
            for item in sae.evidence
        )
    ):
        raise DataValidationError("E7 sparse artifacts differ from frozen scientific inputs")
    assert sae.interpretability_audit is not None
    if (
        sae.interpretability_audit.prompt_transfer_execution is None
        or sae.interpretability_audit.negative_control_execution is None
    ):
        raise DataValidationError("E7 interpretability audit lacks native executions")
    ranked: dict[int, list[tuple[float, str]]] = {
        feature: [] for feature in sae.latent_direction.selected_features
    }
    with torch.no_grad():
        for shard in validation_corpus.iter_shards():
            values = torch.from_numpy(
                np.array(shard.activations, dtype=np.float32, copy=True)
            )
            latents = sae.training.model.encode(values)
            for row, question_id in enumerate(shard.question_ids):
                for feature in ranked:
                    ranked[feature].append((float(latents[row, feature]), question_id))
    for feature, declared_ids in (
        sae.interpretability_audit.top_activating_question_ids.items()
    ):
        expected_ids = tuple(
            question_id
            for _value, question_id in sorted(
                ranked[feature], key=lambda value: (-value[0], value[1])
            )[: len(declared_ids)]
        )
        if declared_ids != expected_ids or any(
            question_id not in validation_corpus.all_question_ids()
            for question_id in declared_ids
        ):
            raise DataValidationError("E7 top-activating examples do not replay")
    from mfh.evaluation.side_effects import (
        recompute_and_verify_official_metric,
        verify_official_metric_receipt,
        verify_safety_score_receipt,
    )
    from mfh.evaluation.strongreject import (
        validate_strongreject_grade_evidence,
        validate_strongreject_terminal_failure,
    )
    from mfh.experiments.e8_protected import (
        validate_e8_execution_record,
        validate_wikitext_likelihood_evidence,
    )
    from mfh.experiments.runner import _PROMPT_HASHES, EvaluationCondition

    sae_geometry = next(iter(sae.evidence)).spec
    for feature in sae.latent_direction.selected_features:
        transfer = sae.interpretability_audit.prompt_transfer_execution[feature]
        controls = sae.interpretability_audit.negative_control_execution[feature]
        for audit_name, audit in {**dict(transfer), **dict(controls)}.items():
            expected_prompt_id = _e7_interpretability_prompt_id(
                audit_name,
                selection_prompt_id=selection.feature_schema.prompt_id,
            )
            for mode, records in (
                ("baseline", audit.baseline_records),
                ("intervention", audit.intervention_records),
            ):
                expected_method = (
                    "M0"
                    if mode == "baseline" or audit_name == "zero_hook"
                    else "M4b"
                )
                for record in records:
                    question = frozen_questions.get(
                        (record.benchmark, record.question_id)
                    )
                    condition_value = record.metadata.get(
                        "e7_interpretability_condition"
                    )
                    if not isinstance(condition_value, Mapping):
                        raise DataValidationError(
                            "E7 interpretability condition evidence is invalid"
                        )
                    try:
                        condition = EvaluationCondition.from_dict(condition_value)
                    except (TypeError, ValueError, DataValidationError) as exc:
                        raise DataValidationError(
                            "E7 interpretability condition evidence is invalid"
                        ) from exc
                    expected_layer: int | None = (
                        (sae_geometry.layer + 1) % condition.model_num_layers
                        if mode == "intervention"
                        and audit_name == "unrelated_layer"
                        else sae_geometry.layer
                    )
                    expected_alpha = (
                        -abs(sae_geometry.alpha)
                        if mode == "intervention"
                        and audit_name == "negative_alpha"
                        else sae_geometry.alpha
                    )
                    if expected_method == "M0":
                        expected_layer = None
                        expected_site = None
                        expected_scope = None
                        expected_alpha = 0.0
                        expected_artifact = None
                    else:
                        expected_site = sae_geometry.site
                        expected_scope = sae_geometry.token_scope
                        expected_artifact = promoted_checkpoint
                    if (
                        question is None
                        or record.benchmark != "triviaqa"
                        or record.metadata.get("source_question_sha256")
                        != question_source_fingerprint(question)
                        or record.normalized_answer
                        != normalize_answer(record.raw_output)
                        or record.outcome
                        is not deterministic_short_answer_grade(
                            record.raw_output, question.aliases
                        )
                        or condition.phase is not ExperimentPhase.E7
                        or condition.condition_id != record.condition_id
                        or condition.benchmark != "triviaqa"
                        or (
                            question.split is not None
                            and condition.partition != question.split
                        )
                        or condition.model_repository
                        != _ACTIVE_MODEL["repository"]
                        or condition.model_revision != _ACTIVE_MODEL["revision"]
                        or condition.runtime is not _ACTIVE_MODEL["runtime"]
                        or condition.quantization != _ACTIVE_MODEL["quantization"]
                        or condition.system_prompt_id != expected_prompt_id
                        or condition.prompt_template_sha256
                        != _PROMPT_HASHES[expected_prompt_id]
                        or condition.steering_method != expected_method
                        or condition.method_artifact_sha256 != expected_artifact
                        or condition.layer != expected_layer
                        or condition.site is not expected_site
                        or condition.token_scope is not expected_scope
                        or condition.alpha != expected_alpha
                        or condition.sparsity is not None
                        or condition.study_protocol_digest != study.digest
                        or condition.seed
                        != sae.interpretability_audit.control_seed
                        or condition.comparison_group
                        != (
                            f"e7-interpretability-{feature}-{audit_name}-{mode}"
                        )
                        or record.metadata.get(
                            "e7_interpretability_feature_index"
                        )
                        != feature
                        or record.metadata.get("e7_interpretability_audit_name")
                        != audit_name
                        or record.metadata.get("e7_interpretability_mode") != mode
                        or record.metadata.get(
                            "e7_interpretability_selection_fingerprint"
                        )
                        != selection.data_fingerprint
                    ):
                        raise DataValidationError(
                            "E7 interpretability row differs from its frozen design"
                        )
                    condition.validate_record(record)
                    validate_e8_execution_record(
                        record,
                        condition_facts=condition.to_dict(),
                        execution_public_key=e6_execution_key,
                        runtime_artifact_sha256=e6_runtime_sha,
                        runtime_identity=e6_runtime_identity,
                    )
                    trace = record.metadata.get("intervention_trace")
                    if expected_method == "M0":
                        if trace is not None:
                            raise DataValidationError(
                                "E7 interpretability baseline contains a hook"
                            )
                        continue
                    direction = _e7_interpretability_direction(
                        audit_name=audit_name,
                        training=sae.training,
                        selection=selection,
                        feature_index=feature,
                        seed=condition.seed,
                    )
                    assert direction is not None
                    values = np.ascontiguousarray(
                        direction.detach().cpu().float().numpy()
                    )
                    norm = float(np.linalg.norm(values))
                    expected_direction_sha = hashlib.sha256(
                        np.ascontiguousarray(values / norm).tobytes(order="C")
                    ).hexdigest()
                    if (
                        not isinstance(trace, Mapping)
                        or trace.get("direction_sha256")
                        != expected_direction_sha
                        or trace.get("source_direction_norm") != norm
                        or trace.get("reference_rms")
                        != coordinate.reference_rms
                        or not math.isclose(
                            float(trace.get("raw_alpha", math.nan)),
                            expected_alpha * norm * coordinate.reference_rms,
                            rel_tol=1e-6,
                            abs_tol=1e-8,
                        )
                    ):
                        raise DataValidationError(
                            "E7 interpretability hook differs from its exact control"
                        )
    screen_direction_facts: dict[float, tuple[str, float]] = {}
    for fraction in _RETAINED_FRACTIONS:
        screened = coordinate_sparse_direction(
            torch.from_numpy(dense),
            recomputed_effect,
            retained_fraction=fraction,
            renormalize=coordinate.sparse_direction.renormalized,
        ).direction
        values = np.ascontiguousarray(screened.detach().cpu().float().numpy())
        norm = float(np.linalg.norm(values))
        normalized = np.ascontiguousarray(values / norm)
        screen_direction_facts[fraction] = (
            hashlib.sha256(normalized.tobytes(order="C")).hexdigest(),
            norm,
        )
    for record in coordinate.screen_records:
        _validate_coordinate_screen_execution_record(
            record,
            contract_digest=coordinate.screen_contract_digest,
            runtime_artifact_sha256=e6_runtime_sha,
            execution_public_key=e6_execution_key,
            prompt_template_sha256=coordinate.feature_schema.prompt_sha256,
            runtime_identity=e6_runtime_identity,
        )
        if record.steering_method == "M0":
            continue
        assert record.sparsity is not None
        trace = record.metadata.get("intervention_trace")
        expected_direction_sha, expected_norm = screen_direction_facts[record.sparsity]
        if (
            not isinstance(trace, Mapping)
            or trace.get("direction_sha256") != expected_direction_sha
            or trace.get("source_direction_norm") != expected_norm
            or not math.isclose(
                float(trace.get("raw_alpha", math.nan)),
                record.alpha * expected_norm * coordinate.reference_rms,
                rel_tol=1e-6,
                abs_tol=1e-8,
            )
        ):
            raise DataValidationError(
                "E7 coordinate screen runtime trace differs from the exact sparse direction"
            )
    sae_sparsity = len(sae.latent_direction.selected_features) / float(
        sae.training.config.resolved_latent_width
    )
    for condition in ledger.contract.conditions:
        expected_artifact = {
            "M0": None,
            "M4a": coordinate_sha,
            "M4b": sae_sha,
        }[condition.steering_method]
        valid_geometry = True
        if condition.steering_method == "M4a":
            valid_geometry = (
                condition.layer == coordinate.layer
                and condition.site is coordinate.site
                and condition.token_scope is coordinate.token_scope
                and condition.alpha == coordinate.alpha
                and condition.sparsity == coordinate.sparse_direction.retained_fraction
            )
        elif condition.steering_method == "M4b":
            if condition.sparsity is None:
                raise DataValidationError("E7 M4b condition lacks frozen sparsity")
            valid_geometry = (
                condition.layer == sae_geometry.layer
                and condition.site is sae_geometry.site
                and condition.token_scope is sae_geometry.token_scope
                and condition.alpha == sae_geometry.alpha
                and bool(
                    np.isclose(
                        condition.sparsity,
                        sae_sparsity,
                        rtol=0,
                        atol=1e-12,
                    )
                )
            )
        if condition.method_artifact_sha256 != expected_artifact or not valid_geometry:
            raise DataValidationError("E7 condition differs from its promoted sparse artifact")
    from mfh.experiments.e8_protected import validate_e8_execution_record

    coordinate_values = np.ascontiguousarray(
        coordinate.sparse_direction.direction.detach().cpu().float().numpy()
    )
    coordinate_norm = float(np.linalg.norm(coordinate_values))
    coordinate_direction_sha = hashlib.sha256(
        np.ascontiguousarray(coordinate_values / coordinate_norm).tobytes(order="C")
    ).hexdigest()
    sae_values = np.ascontiguousarray(
        sae.decoded_direction.detach().cpu().float().numpy()
    )
    sae_norm = float(np.linalg.norm(sae_values))
    sae_direction_sha = hashlib.sha256(
        np.ascontiguousarray(sae_values / sae_norm).tobytes(order="C")
    ).hexdigest()
    facts_by_condition = ledger._gate_context().condition_facts
    for record in ledger.records():
        validate_e8_execution_record(
            record,
            condition_facts=facts_by_condition[record.condition_id],
            execution_public_key=e6_execution_key,
            runtime_artifact_sha256=e6_runtime_sha,
            runtime_identity=e6_runtime_identity,
        )
        if record.benchmark in {"ifeval", "mmlu_pro"}:
            recompute_and_verify_official_metric(
                record,
                frozen_questions[(record.benchmark, record.question_id)],
                scorer=side_effect_scorer,
                ifeval_evaluator_directory=(
                    paths["frozen_side_effect_scorers"] / "ifeval-evaluator"
                ),
            )
        elif record.benchmark == "strongreject_or_harmbench":
            if record.metadata.get("strongreject_grader_failed") is True:
                validate_strongreject_terminal_failure(
                    record,
                    frozen_questions[(record.benchmark, record.question_id)],
                    grader_directory=(
                        paths["frozen_side_effect_scorers"] / "strongreject-grader"
                    ),
                )
            else:
                validate_strongreject_grade_evidence(
                    record,
                    frozen_questions[(record.benchmark, record.question_id)],
                    grader_directory=(
                        paths["frozen_side_effect_scorers"] / "strongreject-grader"
                    ),
                )
            verify_safety_score_receipt(
                record,
                scorer_revision=side_effect_scorer.scorer_revision,
                execution_public_key=side_effect_scorer.execution_public_key,
            )
        elif record.benchmark == "wikitext103":
            mean_nll = validate_wikitext_likelihood_evidence(
                record,
                question=frozen_questions[(record.benchmark, record.question_id)],
            )
            verify_official_metric_receipt(
                record,
                metric="wikitext103_negative_log_likelihood",
                value=mean_nll,
                scorer_revision=side_effect_scorer.scorer_revision,
                execution_public_key=side_effect_scorer.execution_public_key,
            )
        if record.steering_method == "M0":
            continue
        trace = record.metadata.get("intervention_trace")
        expected_sha, expected_norm = (
            (coordinate_direction_sha, coordinate_norm)
            if record.steering_method == "M4a"
            else (sae_direction_sha, sae_norm)
        )
        if (
            not isinstance(trace, Mapping)
            or trace.get("direction_sha256") != expected_sha
            or trace.get("source_direction_norm") != expected_norm
            or trace.get("reference_rms") != coordinate.reference_rms
            or not math.isclose(
                float(trace.get("raw_alpha", math.nan)),
                record.alpha * expected_norm * coordinate.reference_rms,
                rel_tol=1e-6,
                abs_tol=1e-8,
            )
        ):
            raise DataValidationError(
                "E7 final runtime trace differs from its exact promoted direction"
            )
    from mfh.experiments.gates import _side_metric_value

    gate_context = ledger._gate_context()
    causal_metrics = {
        "instruction_following": "ifeval_pass_rate",
        "safe_non_refusal": "xstest_benign_non_refusal_rate",
        "harmful_refusal": "harmful_prompt_refusal_rate",
        "language_consistency": "requested_language_consistency",
    }
    modes = ("baseline", "activated", "suppressed")
    for item in sae.evidence:
        assert item.native_execution_records is not None
        feature_direction = np.ascontiguousarray(
            decoder_feature_direction(
                sae.training.model, item.feature_index
            ).numpy()
        )
        expected_feature_directions = {
            "activated": hashlib.sha256(
                feature_direction.tobytes(order="C")
            ).hexdigest(),
            "suppressed": hashlib.sha256(
                np.ascontiguousarray(-feature_direction).tobytes(order="C")
            ).hexdigest(),
        }
        for mode in ("activated", "suppressed"):
            if any(
                not isinstance(record.metadata.get("intervention_trace"), Mapping)
                or record.metadata["intervention_trace"].get("direction_sha256")
                != expected_feature_directions[mode]
                or record.metadata.get("method_artifact_sha256")
                != promoted_checkpoint
                or record.layer != item.spec.layer
                or record.site is not item.spec.site
                or record.token_scope is not item.spec.token_scope
                or record.alpha != item.spec.alpha
                or record.metadata["intervention_trace"].get("reference_rms")
                != coordinate.reference_rms
                for record in item.native_execution_records[mode]
            ):
                raise DataValidationError(
                    "E7 causal native row differs from its exact decoder feature"
                )
        for mode, records in item.native_execution_records.items():
            expected_execution_id = e7_causal_feature_condition_id(
                feature_index=item.feature_index,
                mode=mode,
                feature_schema=item.spec.feature_schema,
                layer=item.spec.layer,
                site=item.spec.site,
                token_scope=item.spec.token_scope,
                alpha=item.spec.alpha,
                runtime_artifact_sha256=e6_runtime_sha,
                execution_public_key=e6_execution_key,
                source_question_bundle_sha256=exact_question_bundle_sha,
                sae_intervention_sha256=promoted_checkpoint,
            )
            if any(
                record.metadata.get("e7_causal_execution_id")
                != expected_execution_id
                for record in records
            ):
                raise DataValidationError("E7 causal execution contract differs")
        for records in item.native_execution_records.values():
            for record in records:
                validate_e8_execution_record(
                    record,
                    condition_facts={
                        "steering_method": record.steering_method,
                        "method_artifact_sha256": record.metadata.get(
                            "method_artifact_sha256"
                        ),
                        "layer": record.layer,
                        "site": record.site.value if record.site is not None else None,
                        "token_scope": (
                            record.token_scope.value
                            if record.token_scope is not None
                            else None
                        ),
                        "alpha": record.alpha,
                        "sparsity": record.sparsity,
                        "prompt_template_sha256": record.metadata.get(
                            "prompt_template_sha256"
                        ),
                    },
                    execution_public_key=e6_execution_key,
                    runtime_artifact_sha256=e6_runtime_sha,
                    runtime_identity=e6_runtime_identity,
                )
                if (
                    (record.benchmark, record.question_id) not in frozen_questions
                    or record.system_prompt_id != item.spec.feature_schema.prompt_id
                    or record.metadata.get("prompt_template_sha256")
                    != item.spec.feature_schema.prompt_sha256
                    or record.metadata.get("source_question_sha256")
                    != question_source_fingerprint(
                        frozen_questions[(record.benchmark, record.question_id)]
                    )
                ):
                    raise DataValidationError(
                        "E7 causal question differs from its frozen source bundle"
                    )
                if record.benchmark == "triviaqa" and (
                    record.question_id not in trivia_questions
                    or record.normalized_answer != normalize_answer(record.raw_output)
                    or record.outcome
                    is not deterministic_short_answer_grade(
                        record.raw_output,
                        trivia_questions[record.question_id].aliases,
                    )
                ):
                    raise DataValidationError(
                        "E7 causal factual grade differs from frozen TriviaQA aliases"
                    )
                if record.benchmark == "strongreject_or_harmbench":
                    if record.metadata.get("strongreject_grader_failed") is True:
                        validate_strongreject_terminal_failure(
                            record,
                            frozen_questions[(record.benchmark, record.question_id)],
                            grader_directory=(
                                paths["frozen_side_effect_scorers"]
                                / "strongreject-grader"
                            ),
                        )
                    else:
                        validate_strongreject_grade_evidence(
                            record,
                            frozen_questions[(record.benchmark, record.question_id)],
                            grader_directory=(
                                paths["frozen_side_effect_scorers"]
                                / "strongreject-grader"
                            ),
                        )
                    verify_safety_score_receipt(
                        record,
                        scorer_revision=side_effect_scorer.scorer_revision,
                        execution_public_key=side_effect_scorer.execution_public_key,
                    )
                elif record.benchmark == "ifeval":
                    recompute_and_verify_official_metric(
                        record,
                        frozen_questions[(record.benchmark, record.question_id)],
                        scorer=side_effect_scorer,
                        ifeval_evaluator_directory=(
                            paths["frozen_side_effect_scorers"]
                            / "ifeval-evaluator"
                        ),
                    )
        for behavior, measurements in item.protected_outcomes.items():
            for question_id, expected_values in measurements.items():
                observed: list[bool] = []
                for mode in modes:
                    record = next(
                        value
                        for value in item.native_execution_records[mode]
                        if value.question_id == question_id
                    )
                    if (
                        behavior == "abstention_association"
                        and (
                            record.benchmark != "triviaqa"
                            or question_id not in trivia_questions
                        )
                    ):
                        raise DataValidationError(
                            "E7 abstention association must use frozen TriviaQA rows"
                        )
                    observed.append(
                        record.outcome is Outcome.ABSTENTION
                        if behavior == "abstention_association"
                        else bool(
                            _side_metric_value(
                                record, causal_metrics[behavior], gate_context
                            )
                        )
                    )
                if tuple(observed) != expected_values:
                    raise DataValidationError(
                        "E7 causal protected outcomes differ from official receipts"
                    )
    return _E7FinalInputs(
        ledger=ledger,
        coordinate=coordinate,
        sae=sae,
        coordinate_sha256=coordinate_sha,
        sae_sha256=sae_sha,
        package_lock_sha256=package_lock_sha,
        input_paths=paths,
    )


def _condition_pairs(
    ledger: PhaseRunLedger,
) -> Mapping[tuple[str, str, str, str, str], Mapping[str, str]]:
    strata: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    for condition in ledger.contract.conditions:
        key = (
            condition.model_repository,
            condition.benchmark,
            condition.system_prompt_id,
            condition.partition,
            condition.comparison_group,
        )
        methods = strata.setdefault(key, {})
        if condition.steering_method in methods:
            raise DataValidationError("E7 repeats a method within a stratum")
        methods[condition.steering_method] = condition.condition_id
    return MappingProxyType({key: MappingProxyType(value) for key, value in strata.items()})


def _e7_gate_rows(ledger: PhaseRunLedger) -> Mapping[str, tuple[dict[str, str], ...]]:
    protected: list[dict[str, str]] = []
    for key, methods in sorted(_condition_pairs(ledger).items()):
        benchmark = key[1]
        baseline_id = methods.get("M0")
        if baseline_id is None:
            raise DataValidationError("E7 stratum lacks M0")
        for method in ("M4a", "M4b"):
            intervention_id = methods.get(method)
            if intervention_id is None:
                continue
            for question_id in ledger.contract.question_ids_by_benchmark[benchmark]:
                if benchmark in _SIDE_BEHAVIORS:
                    protected.append(
                        {
                            "behavior": _SIDE_BEHAVIORS[benchmark],
                            "question_id": question_id,
                            "baseline_condition_id": baseline_id,
                            "intervention_condition_id": intervention_id,
                        }
                    )
    return MappingProxyType(
        {
            "held_out_reconstruction": (),
            "feature_stability": (),
            "individual_causal_evidence": (),
            "protected_behavior_audit": tuple(protected),
        }
    )


def finalize_e7_phase(
    destination: str | Path,
    *,
    ledger_directory: str | Path,
    study: StudyProtocol,
    coordinate_artifact: str | Path,
    sae_intervention: str | Path,
) -> Mapping[str, Any]:
    normalized = validate_active_study_artifact_paths(
        {
            "E7 finalization": destination,
            "E7 phase ledger": ledger_directory,
            "E7 coordinate artifact": coordinate_artifact,
            "E7 SAE intervention": sae_intervention,
        }
    )
    output = normalized["E7 finalization"]
    ledger_directory = normalized["E7 phase ledger"]
    coordinate_artifact = normalized["E7 coordinate artifact"]
    sae_intervention = normalized["E7 SAE intervention"]
    if output.is_symlink():
        raise FrozenArtifactError(f"refusing linked E7 finalization: {output}")
    if output.exists():
        return verify_e7_phase(output)
    package_lock = Path(__file__).parents[3] / "uv.lock"
    inputs = _validate_e7_final_inputs(
        ledger_directory=ledger_directory,
        study=study,
        coordinate_artifact=coordinate_artifact,
        sae_intervention=sae_intervention,
        package_lock=package_lock,
    )
    ledger = inputs.ledger
    rows = _e7_gate_rows(ledger)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        _copy_frozen_artifact(
            Path(coordinate_artifact).resolve(),
            stage / "coordinate-artifact",
            inputs.coordinate_sha256,
        )
        _copy_frozen_artifact(
            Path(sae_intervention).resolve(),
            stage / "sae-intervention",
            inputs.sae_sha256,
        )
        _copy_frozen_artifact(
            package_lock,
            stage / "uv.lock",
            inputs.package_lock_sha256,
        )
        sae_corpus_sha = ledger.contract.input_fingerprints["separate_sae_corpus"]
        _copy_frozen_artifact(
            inputs.input_paths["separate_sae_corpus"],
            stage / "sae-corpus",
            sae_corpus_sha,
        )
        results = {}
        for gate in _GATES:
            path = stage / f"{gate}.json"
            write_gate_evidence(
                path,
                phase=ExperimentPhase.E7,
                gate=gate,
                contract_digest=ledger.contract.digest,
                record_set_digest=ledger.record_set_digest(),
                observations=rows[gate],
                parameters=(
                    {
                        "sae_intervention_sha256": inputs.sae_sha256,
                        **(
                            {"sae_corpus_sha256": sae_corpus_sha}
                            if gate == "held_out_reconstruction"
                            else {}
                        ),
                    }
                    if gate in {"held_out_reconstruction", "individual_causal_evidence"}
                    else {"coordinate_artifact_sha256": inputs.coordinate_sha256}
                    if gate == "protected_behavior_audit"
                    else None
                ),
            )
            results[gate] = ledger.evaluate_gate(
                gate,
                path,
                supporting_artifacts=(
                    {
                        "sae-intervention": stage / "sae-intervention",
                        **(
                            {"sae-corpus": stage / "sae-corpus"}
                            if gate == "held_out_reconstruction"
                            else {}
                        ),
                    }
                    if gate in {"held_out_reconstruction", "individual_causal_evidence"}
                    else {"coordinate-artifact": stage / "coordinate-artifact"}
                    if gate == "protected_behavior_audit"
                    else None
                ),
            )
        terminal: PhaseCompletion | PhaseFalsification
        expected_gate_digests = {
            name: results[name].gate_digest for name in sorted(results)
        }
        complete_marker = ledger.directory / "complete.json"
        falsified_marker = ledger.directory / "falsified.json"
        if complete_marker.is_file():
            terminal = ledger.verify_complete()
            if (
                not all(value.passed for value in results.values())
                or dict(terminal.gate_result_digests) != expected_gate_digests
            ):
                raise FrozenArtifactError(
                    "E7 recovered terminal differs from re-derived gates"
                )
            status = "complete"
            terminal_digest = terminal.completion_digest
        elif falsified_marker.is_file():
            terminal = ledger.verify_falsified()
            if (
                all(value.passed for value in results.values())
                or dict(terminal.gate_result_digests) != expected_gate_digests
            ):
                raise FrozenArtifactError(
                    "E7 recovered falsification differs from re-derived gates"
                )
            status = "falsified"
            terminal_digest = terminal.falsification_digest
        elif all(value.passed for value in results.values()):
            terminal = ledger.finalize(results)
            status = "complete"
            terminal_digest = terminal.completion_digest
        else:
            terminal = ledger.finalize_falsified(results)
            status = "falsified"
            terminal_digest = terminal.falsification_digest
        portable_ledger_sha = package_portable_phase_ledger(
            ledger.directory,
            stage / "portable-ledger",
            study=study,
        )
        if study.source_path is None:
            raise DataValidationError(
                "E7 portable finalization requires the loaded phases.yaml protocol"
            )
        study_path = stage / "configs" / "experiments" / "phases.yaml"
        _copy_frozen_artifact(
            study.source_path,
            study_path,
            sha256_file(study.source_path),
        )
        study_sha = sha256_file(study_path)
        receipt_body: dict[str, Any] = {
            "schema_version": 2,
            "phase": ExperimentPhase.E7.value,
            "status": status,
            "portable_ledger_sha256": portable_ledger_sha,
            "study_protocol_sha256": study_sha,
            "contract_digest": ledger.contract.digest,
            "record_set_digest": terminal.record_set_digest,
            "coordinate_artifact_sha256": inputs.coordinate_sha256,
            "sae_intervention_sha256": inputs.sae_sha256,
            "package_lock_sha256": inputs.package_lock_sha256,
            "gate_result_digests": dict(terminal.gate_result_digests),
            "terminal_digest": terminal_digest,
            "scientific_eligible": status == "complete",
        }
        (stage / "receipt.json").write_text(
            json.dumps(
                {**receipt_body, "receipt_digest": stable_hash(receipt_body)},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return verify_e7_phase(output)


def verify_e7_phase(
    directory: str | Path,
) -> Mapping[str, Any]:
    from mfh.experiments.protocol import load_study_protocol

    source = Path(directory)
    expected_files = {
        *(f"{gate}.json" for gate in _GATES),
        "coordinate-artifact",
        "sae-intervention",
        "sae-corpus",
        "uv.lock",
        "portable-ledger",
        "configs",
        "receipt.json",
    }
    if (
        source.is_symlink()
        or not source.is_dir()
        or {value.name for value in source.iterdir()} != expected_files
        or any(value.is_symlink() for value in source.rglob("*"))
    ):
        raise FrozenArtifactError("E7 finalization artifact inventory differs")
    study_path = source / "configs" / "experiments" / "phases.yaml"
    study = load_study_protocol(study_path)
    ledger_directory = source / "portable-ledger"
    inputs = _validate_e7_final_inputs(
        ledger_directory=ledger_directory,
        study=study,
        coordinate_artifact=source / "coordinate-artifact",
        sae_intervention=source / "sae-intervention",
        package_lock=source / "uv.lock",
    )
    try:
        receipt = json.loads((source / "receipt.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot load E7 finalization receipt: {exc}") from exc
    if not isinstance(receipt, dict):
        raise FrozenArtifactError("E7 receipt must be an object")
    body = dict(receipt)
    receipt_digest = body.pop("receipt_digest", None)
    expected_keys = {
        "schema_version",
        "phase",
        "status",
        "portable_ledger_sha256",
        "study_protocol_sha256",
        "contract_digest",
        "record_set_digest",
        "coordinate_artifact_sha256",
        "sae_intervention_sha256",
        "package_lock_sha256",
        "gate_result_digests",
        "terminal_digest",
        "scientific_eligible",
    }
    if (
        set(body) != expected_keys
        or type(body.get("schema_version")) is not int
        or body.get("schema_version") != 2
        or body.get("phase") != ExperimentPhase.E7.value
        or body.get("portable_ledger_sha256") != sha256_path(ledger_directory)
        or body.get("study_protocol_sha256") != sha256_file(study_path)
        or body.get("package_lock_sha256") != sha256_file(source / "uv.lock")
        or receipt_digest != stable_hash(body)
    ):
        raise FrozenArtifactError("E7 finalization receipt identity differs")
    ledger = inputs.ledger
    status = body["status"]
    terminal: PhaseCompletion | PhaseFalsification
    if status == "complete":
        terminal = ledger.verify_complete()
        terminal_digest = terminal.completion_digest
        scientific = True
    elif status == "falsified":
        terminal = ledger.verify_falsified()
        terminal_digest = terminal.falsification_digest
        scientific = False
    else:
        raise FrozenArtifactError("E7 terminal status differs")
    expected_gate_artifacts = {
        f"{gate}/evaluation": sha256_file(source / f"{gate}.json") for gate in _GATES
    }
    for gate in {"held_out_reconstruction", "individual_causal_evidence"}:
        expected_gate_artifacts[f"{gate}/sae-intervention"] = sha256_path(
            source / "sae-intervention"
        )
    expected_gate_artifacts["held_out_reconstruction/sae-corpus"] = sha256_path(
        source / "sae-corpus"
    )
    expected_gate_artifacts["protected_behavior_audit/coordinate-artifact"] = (
        sha256_path(source / "coordinate-artifact")
    )
    if (
        body["contract_digest"] != ledger.contract.digest
        or body["record_set_digest"] != terminal.record_set_digest
        or body["coordinate_artifact_sha256"] != inputs.coordinate_sha256
        or body["sae_intervention_sha256"] != inputs.sae_sha256
        or body["gate_result_digests"] != dict(terminal.gate_result_digests)
        or body["terminal_digest"] != terminal_digest
        or body["scientific_eligible"] is not scientific
        or dict(terminal.gate_artifact_fingerprints) != expected_gate_artifacts
    ):
        raise FrozenArtifactError("E7 finalization differs from terminal replay")
    return MappingProxyType(
        {
            "valid": True,
            "status": status,
            "receipt_digest": receipt_digest,
            "terminal_digest": terminal_digest,
            "scientific_eligible": scientific,
        }
    )

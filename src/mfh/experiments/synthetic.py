"""Deterministic, explicitly non-scientific E0--E10 integration smoke workflow."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from torch import Tensor, nn

from mfh.analysis.statistics import (
    AnalysisMetric,
    PairedOutcomes,
    bowker_test,
    holm_adjust,
    mcnemar_exact,
    paired_bootstrap_difference,
    paired_noninferiority,
    paired_prompt_interaction,
)
from mfh.contracts import ActivationSite, GenerationRecord, Outcome, Runtime, TokenScope
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.evaluation.metrics import metric_bundle
from mfh.evaluation.risk import (
    RiskExample,
    matched_area_under_risk_coverage,
    risk_coverage_curve,
    zero_error_upper_bound,
)
from mfh.evaluation.transitions import paired_transition_summary
from mfh.inference.architecture import HookKey, HookMode, HookPoint
from mfh.inference.hooks import ActivationSession, CapturePolicy, InterventionPlan
from mfh.inference.runtime import TeacherForcedScore, set_deterministic_seed
from mfh.methods.adaptive import (
    AdaptiveController,
    AlphaController,
    AlphaMode,
    RouterKind,
    assign_to_vector_regions,
    fit_adaptive_router,
    fit_routed_vector_bank,
)
from mfh.methods.composite import (
    CompositePolicy,
    CompositePolicyConfig,
    minimum_alpha_for_risk,
)
from mfh.methods.controls import matched_random_direction, opposite_direction, zero_direction
from mfh.methods.features import ActivationFeatureSchema, ActivationKind
from mfh.methods.probes import (
    CalibratedProbe,
    ProbeDataset,
    ProbeTask,
    ProbeTrainingConfig,
    evaluate_probe,
    fit_calibrated_probe,
    separability_gate,
)
from mfh.methods.protected import (
    EmpiricalEvaluationIdentity,
    EmpiricalOperatingPoint,
    build_behavior_direction,
    build_protected_subspace,
    covariance_aware_direction,
    match_empirical_operating_points,
    subspace_covariance,
)
from mfh.methods.sparse import (
    CoordinateScreenPoint,
    SAEConfig,
    SeedFeatureSelection,
    coordinate_screen_condition_id,
    coordinate_screen_contract_digest,
    coordinate_screen_execution_receipt_body,
    fit_coordinate_sparse_artifact,
    fit_sparse_autoencoder,
    latent_factuality_direction,
    sae_checkpoint_fingerprint,
    selected_feature_stability,
)
from mfh.methods.static import CentroidVectorBuilder
from mfh.provenance import canonical_json, sha256_file, stable_hash

_PHASES = tuple(f"E{index}" for index in range(11))
_WARNING = (
    "SYNTHETIC INTEGRATION EVIDENCE ONLY: not a benchmark result, not confirmatory evidence, "
    "and not eligible for any E0-E10 phase gate or scientific claim."
)


@dataclass(frozen=True, slots=True)
class SyntheticStudyBundle:
    directory: Path
    seed: int
    phase_digests: Mapping[str, str]
    bundle_digest: str
    scientific_eligible: bool = False


def _tensor_digest(value: Tensor) -> str:
    tensor = value.detach().cpu().float().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _clustered_dataset(
    prefix: str,
    partition: str,
    *,
    seed: int,
    rows_per_class: int = 10,
) -> ProbeDataset:
    generator = torch.Generator().manual_seed(seed)
    centers = (
        torch.tensor([2.5, 0.0, 0.0]),
        torch.tensor([-2.5, 0.0, 0.0]),
        torch.tensor([0.0, 2.5, 0.0]),
    )
    outcomes = (Outcome.CORRECT, Outcome.INCORRECT, Outcome.ABSTENTION)
    features: list[Tensor] = []
    labels: list[Outcome] = []
    identifiers: list[str] = []
    for class_index, (center, outcome) in enumerate(zip(centers, outcomes, strict=True)):
        features.append(center + 0.18 * torch.randn(rows_per_class, 3, generator=generator))
        labels.extend([outcome] * rows_per_class)
        identifiers.extend(f"{prefix}-{class_index}-{row}" for row in range(rows_per_class))
    ids = tuple(identifiers)
    return ProbeDataset(
        ids,
        torch.cat(features),
        tuple(labels),
        group_ids=ids,
        feature_schema=ActivationFeatureSchema.synthetic(partition=partition, width=3),
    )


def _phase_e0(seed: int) -> dict[str, Any]:
    set_deterministic_seed(seed)
    module = nn.Identity()
    key = HookKey(1, ActivationSite.POST_MLP)
    point = HookPoint(key, module, HookMode.POST, "synthetic.identity")
    values = torch.arange(1, 25, dtype=torch.float32).reshape(2, 3, 4) / 10
    with ActivationSession((point,), capture_policy=CapturePolicy.PROMPT_FINAL) as capture_session:
        capture_session.set_prompt(3)
        baseline = module(values.clone())
        captured = capture_session.activations()[key]
    with ActivationSession(
        (point,),
        interventions={
            key: InterventionPlan(
                torch.tensor([1.0, 0.0, 0.0, 0.0]),
                0.5,
                TokenScope.FINAL_PROMPT,
                rms_relative=False,
            )
        },
        capture_policy=CapturePolicy.PROMPT_FINAL,
    ) as intervention_session:
        intervention_session.set_prompt(3)
        steered = module(values.clone())
        pre_intervention = intervention_session.activations()[key]
    delta = steered - baseline
    selected_only = bool(
        torch.equal(delta[:, :-1], torch.zeros_like(delta[:, :-1]))
        and torch.allclose(delta[:, -1, 0], torch.full((2,), 0.5))
        and torch.equal(delta[:, -1, 1:], torch.zeros_like(delta[:, -1, 1:]))
    )
    hooks_removed = bool(torch.equal(module(values), values))
    if not selected_only or not hooks_removed or not torch.equal(captured, pre_intervention):
        raise DataValidationError("synthetic E0 hook semantics failed")
    return {
        "deterministic_seed": seed,
        "capture_sha256": _tensor_digest(captured),
        "pre_intervention_capture": bool(torch.equal(captured, values[:, -1, :])),
        "selected_token_only": selected_only,
        "hooks_removed": hooks_removed,
        "delta_l2": float(torch.linalg.vector_norm(delta)),
    }


def _phase_e1() -> dict[str, Any]:
    outcomes = (
        Outcome.CORRECT,
        Outcome.CORRECT,
        Outcome.PARTIAL,
        Outcome.INCORRECT,
        Outcome.ABSTENTION,
        Outcome.UNSCORABLE,
    )
    metrics = metric_bundle(outcomes)
    base = risk_coverage_curve(
        RiskExample(f"q{index}", risk, outcome)
        for index, (risk, outcome) in enumerate(
            (
                (0.05, Outcome.CORRECT),
                (0.20, Outcome.INCORRECT),
                (0.40, Outcome.CORRECT),
                (0.80, Outcome.INCORRECT),
            )
        )
    )
    adaptive = risk_coverage_curve(
        RiskExample(f"q{index}", risk, outcome)
        for index, (risk, outcome) in enumerate(
            (
                (0.03, Outcome.CORRECT),
                (0.25, Outcome.CORRECT),
                (0.45, Outcome.INCORRECT),
                (0.90, Outcome.INCORRECT),
            )
        )
    )
    coverage_limit, areas = matched_area_under_risk_coverage({"M0": base, "M3": adaptive})
    return {
        "unified_metrics": metrics.to_dict(),
        "risk_curve_points": {"M0": len(base), "M3": len(adaptive)},
        "matched_coverage_limit": coverage_limit,
        "matched_aurc": dict(areas),
        "zero_error_upper_50": zero_error_upper_bound(50),
    }


def _phase_e2(
    seed: int,
) -> tuple[dict[str, Any], CalibratedProbe, ProbeDataset, ProbeDataset, ProbeDataset]:
    training = _clustered_dataset("controller-train", "T-controller-train", seed=seed + 11)
    calibration = _clustered_dataset(
        "controller-cal", "T-controller-calibration", seed=seed + 12, rows_per_class=6
    )
    evaluation = _clustered_dataset("dev", "T-dev", seed=seed + 13, rows_per_class=6)
    probe = fit_calibrated_probe(
        training,
        calibration,
        task=ProbeTask.CORRECT_INCORRECT_ABSTENTION,
        training_config=ProbeTrainingConfig(epochs=60, seed=seed),
    )
    metrics = evaluate_probe(probe, evaluation)
    gate = separability_gate(metrics, {"entropy": 0.60, "max_token": 0.65})
    if not gate.passed:
        raise DataValidationError("synthetic E2 separability gate unexpectedly failed")
    return (
        {
            "probe_metrics": {
                "macro_auroc": metrics.macro_auroc,
                "macro_f1": metrics.macro_f1,
                "brier_score": metrics.brier_score,
                "expected_calibration_error": metrics.expected_calibration_error,
                "per_class_auroc": dict(metrics.per_class_auroc),
            },
            "separability_gate": asdict(gate),
            "training_fingerprint": training.data_fingerprint,
            "calibration_fingerprint": calibration.data_fingerprint,
            "split_disjoint": not bool(set(training.question_ids) & set(calibration.question_ids)),
        },
        probe,
        training,
        calibration,
        evaluation,
    )


def _phase_e3(training: ProbeDataset, seed: int) -> tuple[dict[str, Any], Tensor]:
    key = HookKey(1, ActivationSite.POST_MLP)
    builder = CentroidVectorBuilder()
    for row, outcome in enumerate(training.outcomes):
        builder.update(outcome, {key: training.features[row : row + 1]})
    bank = builder.build(source_method="M1-P", data_fingerprint=training.data_fingerprint)
    direction = bank.vectors[key].direction
    random_control = matched_random_direction(direction, seed=seed)
    controls = {
        "opposite": float(torch.linalg.vector_norm(opposite_direction(direction))),
        "random": float(torch.linalg.vector_norm(random_control)),
        "zero": float(torch.linalg.vector_norm(zero_direction(direction))),
    }
    return (
        {
            "direction_sha256": _tensor_digest(direction),
            "direction_l2": float(torch.linalg.vector_norm(direction)),
            "positive_count": bank.vectors[key].positive_count,
            "negative_count": bank.vectors[key].negative_count,
            "control_norms": controls,
            "random_is_distinct": not bool(torch.equal(random_control, direction)),
        },
        direction,
    )


def _screening_points() -> dict[str, tuple[EmpiricalOperatingPoint, ...]]:
    identity = EmpiricalEvaluationIdentity(
        benchmark="synthetic",
        model_repository="synthetic/model",
        model_revision="0" * 40,
        prompt_id="P0-synthetic",
        prompt_sha256="1" * 64,
        question_set_fingerprint="2" * 64,
        generation_bundle_fingerprint="3" * 64,
    )
    coverages = {
        "M1": 0.70,
        "CAA": 0.74,
        "ITI": 0.72,
        "ACT": 0.76,
        "SADI": 0.73,
        "TruthX": 0.71,
    }
    return {
        method: (
            EmpiricalOperatingPoint(
                method,
                1.0,
                0.05,
                coverage,
                {"utility": 0.90 - index * 0.01},
                identity,
            ),
        )
        for index, (method, coverage) in enumerate(coverages.items())
    }


def _phase_e4() -> dict[str, Any]:
    matched = match_empirical_operating_points(
        _screening_points(), target_hallucination_risk=0.05, tolerance=0.001
    )
    promoted = max(matched, key=lambda method: matched[method].coverage)
    return {
        "screened_methods": sorted(matched),
        "matched_target_risk": 0.05,
        "matched_coverages": {method: point.coverage for method, point in sorted(matched.items())},
        "synthetic_promotion": promoted,
    }


def _routed_rows(seed: int) -> tuple[ProbeDataset, Mapping[HookKey, Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    features: list[Tensor] = []
    first: list[Tensor] = []
    second: list[Tensor] = []
    outcomes: list[Outcome] = []
    for cluster, center in enumerate((-3.0, 3.0)):
        for outcome in (Outcome.CORRECT, Outcome.INCORRECT):
            for _ in range(8):
                features.append(
                    torch.tensor([center, 0.0, 0.0]) + 0.06 * torch.randn(3, generator=generator)
                )
                correct = outcome is Outcome.CORRECT
                first.append(
                    torch.tensor([2.0, 0.0, 0.0])
                    if cluster == 0 and correct
                    else torch.tensor([0.0, 2.0, 0.0])
                    if correct
                    else torch.zeros(3)
                )
                second.append(
                    torch.tensor([0.0, 0.0, 2.0])
                    if cluster == 0 and correct
                    else torch.tensor([1.0, 0.0, 1.0])
                    if correct
                    else torch.zeros(3)
                )
                outcomes.append(outcome)
    ids = tuple(f"steer-{row}" for row in range(len(outcomes)))
    return (
        ProbeDataset(
            ids,
            torch.stack(features),
            tuple(outcomes),
            group_ids=ids,
            feature_schema=ActivationFeatureSchema.synthetic(partition="T-steer", width=3),
        ),
        {
            HookKey(1, ActivationSite.POST_MLP): torch.stack(first),
            HookKey(2, ActivationSite.POST_MLP): torch.stack(second),
        },
    )


def _phase_e5(seed: int, probe: CalibratedProbe) -> tuple[dict[str, Any], AdaptiveController]:
    steer, activations = _routed_rows(seed + 20)
    bank, _ = fit_routed_vector_bank(steer, activations, cluster_count=2, seed=seed)
    controller_ids = tuple(f"controller-{row}" for row in range(len(steer.question_ids)))
    controller_rows = ProbeDataset(
        controller_ids,
        steer.features.clone(),
        steer.outcomes,
        group_ids=controller_ids,
        feature_schema=ActivationFeatureSchema.synthetic(partition="T-controller-train", width=3),
    )
    assignments = assign_to_vector_regions(controller_rows, bank)
    router = fit_adaptive_router(
        controller_rows,
        assignments,
        bank.centers,
        kind=RouterKind.NEAREST_CENTROID,
    )
    controller = AdaptiveController(
        risk_probe=probe,
        vector_bank=bank,
        vector_router=router,
        alpha_controller=AlphaController(
            AlphaMode.RISK_GATED, alpha_max=2.0, beta=8.0, threshold=0.4
        ),
        fixed_layer=1,
    )
    decision = controller.decide(controller_rows.features)
    routed = Counter(assignments.tolist())
    if set(routed) != {0, 1}:
        raise DataValidationError("synthetic E5 router did not exercise both regions")
    return (
        {
            "cluster_count": bank.cluster_count,
            "region_counts": {str(key): routed[key] for key in sorted(routed)},
            "alpha_min": float(decision.alphas.min()),
            "alpha_max": float(decision.alphas.max()),
            "routing_rows_sum_to_one": bool(
                torch.allclose(
                    decision.routing_weights.sum(dim=1),
                    torch.ones(decision.routing_weights.shape[0]),
                )
            ),
            "selected_layers": sorted(set(decision.selected_layers.tolist())),
        },
        controller,
    )


def _synthetic_record(
    question_id: str,
    outcome: Outcome,
    *,
    method: str,
    condition_id: str,
    gold_log_likelihood: float,
    abstention_log_likelihood: float,
) -> GenerationRecord:
    return GenerationRecord(
        question_id=question_id,
        benchmark="synthetic",
        model_repository="synthetic/model",
        model_revision="synthetic",
        runtime=Runtime.SYNTHETIC,
        quantization="none",
        system_prompt_id="P0-synthetic",
        rendered_prompt_hash="4" * 64,
        steering_method=method,
        layer=None,
        token_scope=None,
        alpha=0.0,
        sparsity=None,
        controller_scores={},
        raw_output=f"{method}-{question_id}-{outcome.value}",
        normalized_answer=outcome.value,
        outcome=outcome,
        generation_latency_seconds=0.001,
        input_tokens=5,
        output_tokens=1,
        seed=17,
        condition_id=condition_id,
        metadata={
            "gold_alias_log_likelihood": gold_log_likelihood,
            "abstention_log_likelihood": abstention_log_likelihood,
        },
    )


def _phase_e6() -> dict[str, Any]:
    baseline_outcomes = (
        Outcome.INCORRECT,
        Outcome.INCORRECT,
        Outcome.CORRECT,
        Outcome.CORRECT,
        Outcome.ABSTENTION,
    )
    treatment_outcomes = (
        Outcome.CORRECT,
        Outcome.ABSTENTION,
        Outcome.CORRECT,
        Outcome.ABSTENTION,
        Outcome.ABSTENTION,
    )
    baseline = tuple(
        _synthetic_record(
            f"q{index}",
            outcome,
            method="M0",
            condition_id="synthetic-E6-M0",
            gold_log_likelihood=-2.0,
            abstention_log_likelihood=-1.0,
        )
        for index, outcome in enumerate(baseline_outcomes)
    )
    treatment = tuple(
        _synthetic_record(
            f"q{index}",
            outcome,
            method="M3",
            condition_id="synthetic-E6-M3",
            gold_log_likelihood=-1.5,
            abstention_log_likelihood=-1.2,
        )
        for index, outcome in enumerate(treatment_outcomes)
    )
    transition = paired_transition_summary(baseline, treatment)
    score = TeacherForcedScore("gold", (1, 2), -3.0, -1.5)
    return {
        "transitions": transition.to_dict(),
        "mean_delta_gold_log_likelihood": 0.5,
        "mean_delta_abstention_log_likelihood": -0.2,
        "teacher_forced_tokens": len(score.token_ids),
        "teacher_forced_mean": score.mean_log_likelihood,
    }


def _phase_e7(seed: int) -> tuple[dict[str, Any], Tensor]:
    generator = torch.Generator().manual_seed(seed + 30)
    sources = torch.randn(100, 2, generator=generator)
    mixing = torch.tensor([[1.0, 0.0, 0.8, -0.2], [0.0, 1.0, 0.3, 0.9]])
    values = sources @ mixing + 0.02 * torch.randn(100, 4, generator=generator)
    outcomes = tuple(Outcome.CORRECT if value > 0 else Outcome.INCORRECT for value in sources[:, 0])
    training_schema = ActivationFeatureSchema.synthetic(partition="sae-train", width=4)
    validation_schema = ActivationFeatureSchema.synthetic(partition="sae-validation", width=4)
    selection_ids = tuple(f"sae-steer-{row}" for row in range(len(outcomes)))
    selection = ProbeDataset(
        selection_ids,
        values,
        outcomes,
        group_ids=selection_ids,
        feature_schema=ActivationFeatureSchema.synthetic(partition="T-steer", width=4),
    )
    config = SAEConfig(
        input_width=4,
        expansion_factor=2,
        top_k=2,
        epochs=12,
        batch_size=20,
        learning_rate=0.02,
        seed=seed,
    )
    first = fit_sparse_autoencoder(
        values[:80],
        values[80:],
        config,
        training_schema=training_schema,
        validation_schema=validation_schema,
    )
    second = fit_sparse_autoencoder(
        values[:80],
        values[80:],
        replace(config, epochs=8, seed=seed + 1),
        training_schema=training_schema,
        validation_schema=validation_schema,
    )
    first_latent = latent_factuality_direction(first.model, selection, feature_count=2)
    second_latent = latent_factuality_direction(second.model, selection, feature_count=2)
    selections = (
        SeedFeatureSelection(
            seed,
            sae_checkpoint_fingerprint(first),
            first_latent.selected_features,
        ),
        SeedFeatureSelection(
            seed + 1,
            sae_checkpoint_fingerprint(second),
            second_latent.selected_features,
        ),
    )
    correct = values[torch.tensor([value is Outcome.CORRECT for value in outcomes])]
    incorrect = values[torch.tensor([value is Outcome.INCORRECT for value in outcomes])]
    dense = correct.mean(dim=0) - incorrect.mean(dim=0)
    dense = dense / torch.linalg.vector_norm(dense)
    private_key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex("2" * 64))
    execution_public_key = private_key.public_key().public_bytes(
        Encoding.Raw, PublicFormat.Raw
    ).hex()
    runtime_sha = "e" * 64
    baseline_condition_id = "b" * 64
    points = tuple(
        CoordinateScreenPoint(
            retained_fraction=fraction,
            alpha=alpha,
            baseline_condition_id=baseline_condition_id,
            intervention_condition_id=stable_hash(
                {"fraction": fraction, "alpha": alpha}
            ),
            question_ids=selection.question_ids,
            baseline_outcomes=selection.outcomes,
            intervention_outcomes=(
                (Outcome.CORRECT,) * len(selection.question_ids)
                if (fraction, alpha) == (0.25, 0.5)
                else selection.outcomes
            ),
        )
        for fraction in (0.01, 0.05, 0.10, 0.25)
        for alpha in (0.1, 0.25, 0.5, 1.0, 2.0)
    )
    assert selection.feature_schema is not None
    screen_feature_schema = selection.feature_schema
    source_index = ("P0-synthetic", "M1-P", "post_mlp", 0)
    direction_sha = hashlib.sha256(
        dense.numpy().astype("float32").tobytes()
    ).hexdigest()
    contract_digest = coordinate_screen_contract_digest(
        feature_schema=screen_feature_schema,
        source_artifact_sha256="a" * 64,
        source_tensor_index=source_index,
        source_direction_sha256=direction_sha,
        layer=0,
        site=ActivationSite.POST_MLP,
        token_scope=TokenScope.FIRST_FOUR,
        runtime_artifact_sha256=runtime_sha,
        execution_public_key=execution_public_key,
        points=points,
    )
    baseline_condition_id = coordinate_screen_condition_id(contract_digest)
    points = tuple(
        replace(
            point,
            baseline_condition_id=baseline_condition_id,
            intervention_condition_id=coordinate_screen_condition_id(
                contract_digest,
                retained_fraction=point.retained_fraction,
                alpha=point.alpha,
            ),
        )
        for point in points
    )

    def screen_record(
        question_id: str,
        outcome: Outcome,
        *,
        condition_id: str,
        method: str,
        alpha: float = 0.0,
        sparsity: float | None = None,
    ) -> GenerationRecord:
        intervened = method == "M4a"
        trace = (
            {
                "coordinate_screen_contract_digest": contract_digest,
                "source_artifact_sha256": "a" * 64,
                "direction_sha256": "9" * 64,
                "layer": 0,
                "site": ActivationSite.POST_MLP.value,
                "token_scope": TokenScope.FIRST_FOUR.value,
                "standardized_alpha": alpha,
                "raw_alpha": alpha,
                "retained_fraction": sparsity,
                "reference_rms": 1.0,
                "source_direction_norm": 1.0,
                "applied_tokens": 1,
                "applied_token_indices": [0],
                "pre_activation_sha256": "7" * 64,
                "post_activation_sha256": "8" * 64,
                "delta_sha256": "6" * 64,
            }
            if intervened
            else None
        )
        metadata = {
            "coordinate_screen_contract_digest": contract_digest,
            "coordinate_screen_runtime_artifact_sha256": runtime_sha,
            "coordinate_screen_execution_public_key": execution_public_key,
            "prompt_template_sha256": screen_feature_schema.prompt_sha256,
            "generation_runtime_metrics": {
                "schema_version": 1,
                "unified_memory_bytes": 1_000_000,
                "peak_memory_bytes": 1_000,
                "generation_peak_memory_bytes": 1_000,
                "auxiliary_peak_memory_bytes": 0,
                "active_memory_bytes": 800,
                "cache_memory_bytes": 200,
                "prompt_tokens_per_second": 10.0,
                "generation_tokens_per_second": 10.0,
                "generation_wall_time_seconds": 0.1,
                "stop_type": "length",
                "stopping_token_id": None,
            },
            **(
                {
                    "intervention_trace": trace,
                    "intervention_trace_digest": stable_hash(trace),
                }
                if trace is not None
                else {}
            ),
        }
        unsigned = GenerationRecord(
            question_id=question_id,
            benchmark="synthetic",
            model_repository="synthetic/model",
            model_revision="0" * 40,
            runtime=Runtime.SYNTHETIC,
            quantization="none",
            system_prompt_id="P0-synthetic",
            rendered_prompt_hash="d" * 64,
            steering_method=method,
            layer=0 if intervened else None,
            site=ActivationSite.POST_MLP if intervened else None,
            token_scope=TokenScope.FIRST_FOUR if intervened else None,
            alpha=alpha,
            sparsity=sparsity,
            controller_scores={},
            raw_output=outcome.value,
            normalized_answer=outcome.value,
            outcome=outcome,
            generation_latency_seconds=0.1,
            input_tokens=1,
            output_tokens=1,
            condition_id=condition_id,
            metadata=metadata,
        )
        signature = private_key.sign(
            canonical_json(
                coordinate_screen_execution_receipt_body(
                    unsigned,
                    contract_digest=contract_digest,
                    runtime_artifact_sha256=runtime_sha,
                )
            ).encode()
        ).hex()
        return replace(
            unsigned,
            metadata={
                **dict(unsigned.metadata),
                "coordinate_screen_execution_signature": signature,
            },
        )

    screen_records = [
        screen_record(
            question_id,
            outcome,
            condition_id=baseline_condition_id,
            method="M0",
        )
        for question_id, outcome in zip(
            selection.question_ids, selection.outcomes, strict=True
        )
    ]
    for point in points:
        screen_records.extend(
            screen_record(
                question_id,
                outcome,
                condition_id=point.intervention_condition_id,
                method="M4a",
                alpha=point.alpha,
                sparsity=point.retained_fraction,
            )
            for question_id, outcome in zip(
                point.question_ids, point.intervention_outcomes, strict=True
            )
        )
    coordinate = fit_coordinate_sparse_artifact(
        selection,
        dense,
        screen_points=points,
        screen_records=screen_records,
        source_artifact_sha256="a" * 64,
        source_tensor_index=source_index,
        source_direction_sha256=direction_sha,
        reference_rms=1.0,
        layer=0,
        site=ActivationSite.POST_MLP,
        token_scope=TokenScope.FIRST_FOUR,
        screen_runtime_artifact_sha256=runtime_sha,
        screen_execution_public_key=execution_public_key,
    )
    return (
        {
            "reconstruction_mse": first.metrics.reconstruction_mse,
            "fraction_variance_explained": first.metrics.fraction_variance_explained,
            "average_active_features": first.metrics.average_active_features,
            "loss_decreased": first.loss_history[-1] < first.loss_history[0],
            "selected_features": list(first_latent.selected_features),
            "two_seed_jaccard": selected_feature_stability(selections),
            "coordinate_retained_dimensions": (coordinate.sparse_direction.retained_dimensions),
        },
        dense,
    )


def _phase_e8(seed: int, dense_direction: Tensor) -> dict[str, Any]:
    refusal = build_behavior_direction(
        "safe_refusal",
        torch.tensor([[2.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]]),
        torch.zeros(2, 4),
    )
    language = build_behavior_direction(
        "language_switch",
        torch.tensor([[0.0, 2.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]]),
        torch.zeros(2, 4),
    )
    subspace = build_protected_subspace(
        (refusal, language),
        data_fingerprint="5" * 64,
        feature_schema=ActivationFeatureSchema.synthetic(
            partition="protected-construction", width=4
        ),
    )
    projected = subspace.project(dense_direction, normalize=True)
    covariance_aware = covariance_aware_direction(
        dense_direction,
        subspace_covariance(subspace),
        lambda_penalty=1.0,
        ridge=0.1,
    )
    ids = tuple(f"side-{row}" for row in range(20))
    values = tuple(float(row % 2) for row in range(20))
    noninferiority = paired_noninferiority(
        ids,
        values,
        values,
        margin=0.02,
        resamples=200,
        seed=seed,
    )
    screening = _screening_points()
    projected_point = screening["M1"][0]
    covariance_point = screening["ACT"][0]
    matched = match_empirical_operating_points(
        {
            "M4": (
                EmpiricalOperatingPoint(
                    "M4",
                    projected_point.alpha,
                    projected_point.hallucination_risk,
                    projected_point.coverage,
                    projected_point.utility_metrics,
                    projected_point.evaluation,
                ),
            ),
            "M5": (
                EmpiricalOperatingPoint(
                    "M5",
                    covariance_point.alpha,
                    covariance_point.hallucination_risk,
                    covariance_point.coverage,
                    covariance_point.utility_metrics,
                    covariance_point.evaluation,
                ),
            ),
        },
        target_hallucination_risk=0.05,
        tolerance=0.001,
    )
    return {
        "protected_rank": int(subspace.basis.shape[0]),
        "projected_protected_energy": subspace.protected_energy(projected),
        "projected_direction_l2": float(torch.linalg.vector_norm(projected)),
        "covariance_aware_sha256": _tensor_digest(covariance_aware),
        "noninferiority": asdict(noninferiority),
        "matched_coverages": {method: point.coverage for method, point in sorted(matched.items())},
    }


def _phase_e9(seed: int) -> dict[str, Any]:
    identifiers = tuple(f"q{row}" for row in range(8))
    baseline = (
        Outcome.CORRECT,
        Outcome.CORRECT,
        Outcome.INCORRECT,
        Outcome.INCORRECT,
        Outcome.ABSTENTION,
        Outcome.ABSTENTION,
        Outcome.INCORRECT,
        Outcome.ABSTENTION,
    )
    treatment = (
        Outcome.CORRECT,
        Outcome.CORRECT,
        Outcome.CORRECT,
        Outcome.CORRECT,
        Outcome.ABSTENTION,
        Outcome.ABSTENTION,
        Outcome.ABSTENTION,
        Outcome.ABSTENTION,
    )
    paired = PairedOutcomes(identifiers, baseline, treatment)
    bootstrap = paired_bootstrap_difference(
        paired, AnalysisMetric.ACCURACY, resamples=250, seed=seed
    )
    mcnemar = mcnemar_exact(paired)
    bowker = bowker_test(paired)
    adjusted = holm_adjust((("RQ1", mcnemar.exact_p_value), ("RQ2", bowker.p_value)))
    interaction = paired_prompt_interaction(
        identifiers[:4],
        baseline[:4],
        treatment[:4],
        treatment[:4],
        (Outcome.CORRECT,) * 4,
        AnalysisMetric.ACCURACY,
        resamples=200,
        seed=seed,
    )
    factorial = tuple(
        product(
            ("synthetic-small", "synthetic-large"),
            ("trivia", "simple", "aa"),
            ("P0", "P2"),
            ("M0", "M1", "M3"),
        )
    )
    return {
        "toy_factorial_conditions": len(factorial),
        "paired_bootstrap": asdict(bootstrap),
        "mcnemar": asdict(mcnemar),
        "bowker": asdict(bowker),
        "holm": [asdict(value) for value in adjusted],
        "prompt_interaction": asdict(interaction),
    }


def _phase_e10(probe: CalibratedProbe, controller: AdaptiveController) -> dict[str, Any]:
    early_probe = replace(
        probe,
        training_schema=ActivationFeatureSchema.synthetic(
            partition="T-controller-train",
            width=3,
            activation_kind=ActivationKind.FIRST_GENERATED,
            token_scope=TokenScope.FIRST_GENERATED,
        ),
        calibration_schema=ActivationFeatureSchema.synthetic(
            partition="T-controller-calibration",
            width=3,
            activation_kind=ActivationKind.FIRST_GENERATED,
            token_scope=TokenScope.FIRST_GENERATED,
        ),
    )
    policy = CompositePolicy(
        controller,
        CompositePolicyConfig(
            tau_low=0.2,
            tau_high=0.7,
            release_epsilon=0.1,
            token_scope=TokenScope.FIRST_FOUR,
        ),
        early_probe=early_probe,
    )
    assessments = policy.assess(torch.tensor([[2.5, 0.0, 0.0], [0.0, 0.0, 0.0], [-2.5, 0.0, 0.0]]))
    regimes = Counter(value.regime.value for value in assessments)
    safe = policy.output_gate(0.05, safety_ok=True, language_ok=True, refusal_drift=False)
    unsafe = policy.output_gate(0.05, safety_ok=False, language_ok=True, refusal_drift=False)
    early = policy.reevaluate_after_early_tokens(
        torch.tensor([[2.5, 0.0, 0.0]]),
        safety_ok=True,
        language_ok=True,
        refusal_drift=False,
        gold_log_likelihood_delta=0.1,
    )
    return {
        "risk_regimes": dict(sorted(regimes.items())),
        "safe_output_action": safe.action.value,
        "unsafe_output_action": unsafe.action.value,
        "early_continue": early.continue_generation,
        "gold_likelihood_improved": early.gold_likelihood_improved,
        "minimum_alpha_at_risk_target": minimum_alpha_for_risk(
            ((0.5, 0.20), (1.0, 0.08), (2.0, 0.03)), risk_epsilon=0.10
        ),
        "zero_error_upper_100": zero_error_upper_bound(100),
        "one_shot_confirmatory_executed": False,
    }


def _phase_payload(
    phase: str,
    result: Mapping[str, Any],
    dependencies: Mapping[str, str],
) -> dict[str, Any]:
    body = {
        "schema_version": 1,
        "phase": phase,
        "scientific_eligible": False,
        "runtime": "synthetic",
        "warning": _WARNING,
        "dependencies": dict(dependencies),
        "result": json.loads(canonical_json(result)),
    }
    return {**body, "phase_digest": stable_hash(body)}


def _execute(seed: int) -> dict[str, dict[str, Any]]:
    set_deterministic_seed(seed)
    results: dict[str, Mapping[str, Any]] = {}
    results["E0"] = _phase_e0(seed)
    results["E1"] = _phase_e1()
    e2, probe, training, _, _ = _phase_e2(seed)
    results["E2"] = e2
    e3, _ = _phase_e3(training, seed)
    results["E3"] = e3
    results["E4"] = _phase_e4()
    e5, controller = _phase_e5(seed, probe)
    results["E5"] = e5
    results["E6"] = _phase_e6()
    e7, dense_direction = _phase_e7(seed)
    results["E7"] = e7
    results["E8"] = _phase_e8(seed, dense_direction)
    results["E9"] = _phase_e9(seed)
    results["E10"] = _phase_e10(probe, controller)

    payloads: dict[str, dict[str, Any]] = {}
    for index, phase in enumerate(_PHASES):
        dependencies = (
            {}
            if index == 0
            else {_PHASES[index - 1]: str(payloads[_PHASES[index - 1]]["phase_digest"])}
        )
        payloads[phase] = _phase_payload(phase, results[phase], dependencies)
    return payloads


def _write_text(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _read_mapping(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read {context}: {exc}") from exc
    if not isinstance(value, dict):
        raise FrozenArtifactError(f"{context} must be a mapping")
    return value


def _exact_directory(
    path: Path,
    *,
    files: set[str],
    directories: set[str],
    context: str,
) -> None:
    if path.is_symlink() or not path.is_dir():
        raise FrozenArtifactError(f"{context} must be a regular directory")
    entries = tuple(path.iterdir())
    if {entry.name for entry in entries} != files | directories:
        raise FrozenArtifactError(f"{context} contains missing or unexpected entries")
    for entry in entries:
        if entry.is_symlink():
            raise FrozenArtifactError(f"{context} contains a symbolic link")
        if entry.name in files and not entry.is_file():
            raise FrozenArtifactError(f"{context} contains a non-regular file")
        if entry.name in directories and not entry.is_dir():
            raise FrozenArtifactError(f"{context} contains a non-regular directory")


def run_synthetic_study(
    output: str | Path,
    *,
    seed: int = 1701,
) -> SyntheticStudyBundle:
    """Execute the integration smoke and atomically freeze a non-scientific bundle."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise DataValidationError("synthetic study seed must be a non-negative integer")
    destination = Path(output)
    if destination.exists():
        raise FrozenArtifactError(f"refusing to overwrite synthetic study: {destination}")
    payloads = _execute(seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        phase_root = stage / "phases"
        phase_root.mkdir()
        descriptors: dict[str, dict[str, str]] = {}
        for phase in _PHASES:
            filename = f"{phase}.json"
            path = phase_root / filename
            _write_text(
                path,
                json.dumps(payloads[phase], indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            )
            descriptors[phase] = {
                "filename": f"phases/{filename}",
                "sha256": sha256_file(path),
                "phase_digest": str(payloads[phase]["phase_digest"]),
            }
        body = {
            "schema_version": 1,
            "scientific_eligible": False,
            "runtime": "synthetic",
            "warning": _WARNING,
            "seed": seed,
            "phase_order": list(_PHASES),
            "phases": descriptors,
            "execution_digest": stable_hash(
                [descriptors[phase]["phase_digest"] for phase in _PHASES]
            ),
        }
        _write_text(
            stage / "manifest.json",
            json.dumps(
                {**body, "bundle_digest": stable_hash(body)},
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return verify_synthetic_study(destination, replay=False)


def verify_synthetic_study(
    directory: str | Path,
    *,
    replay: bool = True,
) -> SyntheticStudyBundle:
    """Verify the frozen tree and optionally replay every deterministic phase."""

    source = Path(directory)
    _exact_directory(
        source,
        files={"manifest.json"},
        directories={"phases"},
        context="synthetic study bundle",
    )
    _exact_directory(
        source / "phases",
        files={f"{phase}.json" for phase in _PHASES},
        directories=set(),
        context="synthetic phase directory",
    )
    manifest = _read_mapping(source / "manifest.json", "synthetic study manifest")
    digest = manifest.pop("bundle_digest", None)
    expected_fields = {
        "schema_version",
        "scientific_eligible",
        "runtime",
        "warning",
        "seed",
        "phase_order",
        "phases",
        "execution_digest",
    }
    if (
        set(manifest) != expected_fields
        or manifest.get("schema_version") != 1
        or manifest.get("scientific_eligible") is not False
        or manifest.get("runtime") != "synthetic"
        or manifest.get("warning") != _WARNING
        or manifest.get("phase_order") != list(_PHASES)
        or not isinstance(digest, str)
        or digest != stable_hash(manifest)
    ):
        raise FrozenArtifactError("synthetic study manifest is invalid")
    seed = manifest["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise FrozenArtifactError("synthetic study seed is invalid")
    descriptors = manifest["phases"]
    if not isinstance(descriptors, Mapping) or set(descriptors) != set(_PHASES):
        raise FrozenArtifactError("synthetic study phase descriptors are invalid")
    payloads: dict[str, dict[str, Any]] = {}
    phase_digests: dict[str, str] = {}
    for index, phase in enumerate(_PHASES):
        descriptor = descriptors[phase]
        expected_filename = f"phases/{phase}.json"
        if (
            not isinstance(descriptor, Mapping)
            or set(descriptor) != {"filename", "sha256", "phase_digest"}
            or descriptor["filename"] != expected_filename
        ):
            raise FrozenArtifactError(f"synthetic {phase} descriptor is invalid")
        path = source / expected_filename
        if descriptor["sha256"] != sha256_file(path):
            raise FrozenArtifactError(f"synthetic {phase} file changed")
        payload = _read_mapping(path, f"synthetic {phase} result")
        phase_digest = payload.pop("phase_digest", None)
        expected_dependencies = (
            {} if index == 0 else {_PHASES[index - 1]: phase_digests[_PHASES[index - 1]]}
        )
        if (
            set(payload)
            != {
                "schema_version",
                "phase",
                "scientific_eligible",
                "runtime",
                "warning",
                "dependencies",
                "result",
            }
            or payload.get("schema_version") != 1
            or payload.get("phase") != phase
            or payload.get("scientific_eligible") is not False
            or payload.get("runtime") != "synthetic"
            or payload.get("warning") != _WARNING
            or payload.get("dependencies") != expected_dependencies
            or not isinstance(phase_digest, str)
            or phase_digest != stable_hash(payload)
            or descriptor["phase_digest"] != phase_digest
        ):
            raise FrozenArtifactError(f"synthetic {phase} payload is invalid")
        payloads[phase] = {**payload, "phase_digest": phase_digest}
        phase_digests[phase] = phase_digest
    if manifest["execution_digest"] != stable_hash([phase_digests[phase] for phase in _PHASES]):
        raise FrozenArtifactError("synthetic execution chain changed")
    if replay:
        replayed = _execute(seed)
        if canonical_json(replayed) != canonical_json(payloads):
            raise FrozenArtifactError("synthetic study differs from deterministic replay")
    return SyntheticStudyBundle(
        directory=source,
        seed=seed,
        phase_digests=MappingProxyType(phase_digests),
        bundle_digest=digest,
    )

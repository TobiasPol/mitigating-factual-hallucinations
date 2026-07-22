"""Replayable construction of every quantitative final-analysis section.

The reporting layer deliberately does not estimate statistics.  This module is
the single calculation boundary: it consumes verified generation records,
replayed E9 outputs, finalized human annotations, and replayed prompt-paraphrase
records, then returns the typed payload used by every table and figure.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from mfh.analysis.human_audit import HumanAuditResults
from mfh.analysis.protocol import AnalysisProtocol, MarginScale
from mfh.analysis.reporting import FinalAnalysisResults, _exact_binomial_upper
from mfh.analysis.statistics import (
    holm_adjust,
    paired_noninferiority,
    simulate_paired_mcnemar_power,
)
from mfh.contracts import GenerationRecord, Outcome, Runtime
from mfh.errors import DataValidationError
from mfh.evaluation.language import (
    SUPPORTED_LANGUAGES,
    language_response_evidence,
)
from mfh.evaluation.metrics import metric_bundle
from mfh.evaluation.risk import (
    RiskCoveragePoint,
    RiskExample,
    area_under_risk_coverage,
    matched_area_under_risk_coverage,
    risk_coverage_curve,
)
from mfh.evaluation.simpleqa import simpleqa_hedging_evidence_is_valid
from mfh.experiments.protocol import ExperimentPhase, StudyProtocol
from mfh.experiments.robustness_results import (
    RQ1GeneralizationResult,
    VerifiedPromptParaphraseRecord,
)
from mfh.provenance import sha256_path, stable_hash

_FACTUAL = (
    "triviaqa",
    "simpleqa_verified",
    "aa_omniscience_public_600",
)
_PROMPTS = ("P0-neutral", "P2-calibrated-abstention")
_SIDE_METRIC_BENCHMARK = {
    "ifeval_pass_rate": "ifeval",
    "mmlu_pro_accuracy": "mmlu_pro",
    "xstest_benign_non_refusal_rate": "xstest",
    "harmful_prompt_refusal_rate": "strongreject_or_harmbench",
    "harmful_prompt_compliance_score": "strongreject_or_harmbench",
    "requested_language_consistency": "language_consistency",
    "perplexity_relative_change": "wikitext103",
    "latency_relative_change": "wikitext103",
}


@dataclass(frozen=True, slots=True)
class DerivedFinalAnalysis:
    """Calculated results plus the exact raw-source identities used to obtain them."""

    results: FinalAnalysisResults
    source_record_digests: Mapping[str, str]
    e9_analysis_digest: str
    human_audit_manifest_digest: str
    robustness_record_digest: str
    runtime_attestation_digest: str
    phase_completion_digests: Mapping[str, str]

    @property
    def derivation_digest(self) -> str:
        return stable_hash(
            {
                "source_record_digests": dict(self.source_record_digests),
                "e9_analysis_digest": self.e9_analysis_digest,
                "human_audit_manifest_digest": self.human_audit_manifest_digest,
                "robustness_record_digest": self.robustness_record_digest,
                "runtime_attestation_digest": self.runtime_attestation_digest,
                "phase_completion_digests": dict(self.phase_completion_digests),
                "results": self.results.to_dict(),
            }
        )


def _ordered_records(
    values: Sequence[GenerationRecord], phase: str
) -> tuple[GenerationRecord, ...]:
    records = tuple(sorted(values, key=lambda value: (value.condition_id, value.question_id)))
    keys = [(value.condition_id, value.question_id) for value in records]
    if not records or len(keys) != len(set(keys)):
        raise DataValidationError(f"{phase} final-analysis records must be unique and non-empty")
    if phase != ExperimentPhase.E10.value and any(
        value.outcome is Outcome.UNSCORABLE for value in records
    ):
        raise DataValidationError(f"{phase} final-analysis records contain unscorable outcomes")
    return records


def _condition_cells(
    records: Sequence[GenerationRecord],
) -> Mapping[tuple[str, str, str, str], Mapping[str, GenerationRecord]]:
    result: dict[tuple[str, str, str, str], dict[str, GenerationRecord]] = {}
    for record in records:
        key = (
            record.condition_id,
            record.benchmark,
            record.system_prompt_id,
            record.steering_method,
        )
        cell = result.setdefault(key, {})
        if record.question_id in cell:
            raise DataValidationError("final analysis contains a duplicate condition question")
        cell[record.question_id] = record
    return result


def _method_cells(
    records: Sequence[GenerationRecord],
) -> Mapping[tuple[str, str, str], Mapping[str, GenerationRecord]]:
    result: dict[tuple[str, str, str], dict[str, GenerationRecord]] = {}
    for record in records:
        key = (record.benchmark, record.system_prompt_id, record.steering_method)
        cell = result.setdefault(key, {})
        if record.question_id in cell:
            raise DataValidationError(
                "final analysis has more than one operating point for a method cell"
            )
        cell[record.question_id] = record
    return result


def _paired_cells(
    cells: Mapping[tuple[str, str, str], Mapping[str, GenerationRecord]],
    *,
    benchmark: str,
    prompt: str,
    baseline: str,
    treatment: str,
) -> tuple[tuple[str, ...], tuple[GenerationRecord, ...], tuple[GenerationRecord, ...]]:
    try:
        before = cells[(benchmark, prompt, baseline)]
        after = cells[(benchmark, prompt, treatment)]
    except KeyError as exc:
        raise DataValidationError("final analysis lacks a paired method cell") from exc
    if set(before) != set(after) or not before:
        raise DataValidationError("final analysis method cells are not exactly question-paired")
    identifiers = tuple(sorted(before))
    return (
        identifiers,
        tuple(before[value] for value in identifiers),
        tuple(after[value] for value in identifiers),
    )


def _primary_and_e9_sections(
    outputs: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    required = {
        "primary_contrasts.json",
        "prompt_method_interactions.json",
        "mixed_effects.json",
        "holm_corrections.json",
        "condition_summaries.json",
    }
    if set(outputs) != required:
        raise DataValidationError("final analysis requires the complete replayed E9 output set")
    primary_value = outputs["primary_contrasts.json"].get("contrasts")
    if not isinstance(primary_value, Mapping) or set(primary_value) != {
        "RQ1",
        "RQ2",
        "RQ3",
        "RQ4",
    }:
        raise DataValidationError("replayed E9 primary contrasts are incomplete")
    primary: dict[str, Any] = {}
    for rq in ("RQ1", "RQ2", "RQ3", "RQ4"):
        raw_items = primary_value[rq]
        if not isinstance(raw_items, list) or not raw_items:
            raise DataValidationError(f"replayed E9 {rq} comparisons are empty")
        comparisons: dict[str, Any] = {}
        for raw in raw_items:
            if not isinstance(raw, Mapping):
                raise DataValidationError(f"replayed E9 {rq} comparison is invalid")
            if rq == "RQ3":
                nested = raw.get("risk_and_coverage_contrasts")
                candidates = nested if isinstance(nested, list) else []
            else:
                candidates = [raw]
            for candidate in candidates:
                if not isinstance(candidate, Mapping):
                    raise DataValidationError(f"replayed E9 {rq} result is invalid")
                comparison_id = candidate.get("comparison_id")
                if not isinstance(comparison_id, str) or not comparison_id:
                    raise DataValidationError(f"replayed E9 {rq} lacks a comparison identity")
                if rq == "RQ4":
                    result = candidate.get("result")
                    if not isinstance(result, Mapping):
                        raise DataValidationError("replayed E9 interaction is invalid")
                    estimate = result.get("interaction")
                else:
                    result = candidate.get("paired_bootstrap")
                    if not isinstance(result, Mapping):
                        raise DataValidationError("replayed E9 paired bootstrap is invalid")
                    estimate = result.get("difference")
                if isinstance(estimate, bool) or not isinstance(estimate, int | float):
                    raise DataValidationError("replayed E9 comparison estimate is invalid")
                try:
                    comparisons[comparison_id] = {
                        "estimate": float(estimate),
                        "confidence_interval": [float(result["lower"]), float(result["upper"])],
                        "p_value": float(result["two_sided_p_value"]),
                        "questions": int(result["questions"]),
                    }
                except (KeyError, TypeError, ValueError) as exc:
                    raise DataValidationError(
                        "replayed E9 comparison statistics are invalid"
                    ) from exc
        if not comparisons:
            raise DataValidationError(f"replayed E9 {rq} contains no quantitative comparisons")
        # The RQ-level entry is a conservative display sentinel: it is the member
        # with the largest raw p-value.  Inference remains in the full comparison
        # mapping and Holm family below; no p-values are averaged.
        sentinel_key = max(
            comparisons,
            key=lambda key: (float(comparisons[key]["p_value"]), key),
        )
        sentinel = comparisons[sentinel_key]
        primary[rq] = {
            **sentinel,
            "comparison_count": len(comparisons),
            "sentinel_comparison_sha256": stable_hash(sentinel_key),
            "comparisons": comparisons,
        }

    holm_rows = outputs["holm_corrections.json"].get("hypotheses")
    if not isinstance(holm_rows, list) or not holm_rows:
        raise DataValidationError("replayed E9 Holm family is empty")
    holm: dict[str, Any] = {}
    for row in holm_rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("hypothesis"), str):
            raise DataValidationError("replayed E9 Holm result is invalid")
        holm[str(row["hypothesis"])] = {
            "raw_p_value": float(row["raw_p_value"]),
            "adjusted_p_value": float(row["adjusted_p_value"]),
            "rejected": bool(row["rejected"]),
        }

    mixed_value = outputs["mixed_effects.json"].get("result")
    if not isinstance(mixed_value, Mapping):
        raise DataValidationError("replayed E9 mixed-effects result is invalid")
    names = mixed_value.get("fixed_effect_names")
    coefficients = mixed_value.get("coefficients")
    errors = mixed_value.get("standard_errors")
    if (
        not isinstance(names, list | tuple)
        or not isinstance(coefficients, list | tuple)
        or not isinstance(errors, list | tuple)
        or len(names) != len(coefficients)
        or len(names) != len(errors)
        or not names
    ):
        raise DataValidationError("replayed E9 fixed-effect arrays differ")
    mixed: dict[str, Any] = {}
    for name, coefficient, error in zip(names, coefficients, errors, strict=True):
        estimate = float(coefficient)
        standard_error = float(error)
        if not isinstance(name, str) or standard_error <= 0:
            raise DataValidationError("replayed E9 fixed effect is invalid")
        z_value = estimate / standard_error
        mixed[name] = {
            "estimate": estimate,
            "standard_error": standard_error,
            "z_value": z_value,
            "p_value": math.erfc(abs(z_value) / math.sqrt(2)),
        }

    interaction_rows = outputs["prompt_method_interactions.json"].get("interactions")
    if not isinstance(interaction_rows, list) or not interaction_rows:
        raise DataValidationError("replayed E9 prompt interactions are empty")
    interactions: dict[str, Any] = {}
    for row in interaction_rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("comparison_id"), str):
            raise DataValidationError("replayed E9 prompt interaction is invalid")
        result = row.get("result")
        if not isinstance(result, Mapping):
            raise DataValidationError("replayed E9 prompt interaction result is invalid")
        interactions[str(row["comparison_id"])] = {
            "estimate": float(result["interaction"]),
            "confidence_interval": [float(result["lower"]), float(result["upper"])],
            "p_value": float(result["two_sided_p_value"]),
            "questions": int(result["questions"]),
            "prompt_only_gain": float(result["prompt_only_gain"]),
            "steering_only_gain": float(result["steering_only_gain"]),
            "combined_gain": float(result["combined_gain"]),
        }
    return primary, holm, mixed, interactions


def _transition_decomposition(
    records: Sequence[GenerationRecord],
    composite_records: Sequence[GenerationRecord],
) -> dict[str, Any]:
    cells = _method_cells((*records, *composite_records))
    result: dict[str, Any] = {}
    comparisons = (
        *(
            (benchmark, prompt, treatment)
            for benchmark in _FACTUAL
            for prompt in _PROMPTS
            for treatment in ("M1", "M2", "M3", "M4", "M5")
        ),
        *((benchmark, "P0-neutral", "M6") for benchmark in _FACTUAL),
    )
    for benchmark, prompt, treatment in comparisons:
        identifiers, before, after = _paired_cells(
            cells,
            benchmark=benchmark,
            prompt=prompt,
            baseline="M0",
            treatment=treatment,
        )
        scorable = tuple(
            (left, right)
            for left, right in zip(before, after, strict=True)
            if left.outcome is not Outcome.UNSCORABLE and right.outcome is not Outcome.UNSCORABLE
        )
        excluded = len(identifiers) - len(scorable)
        if not scorable:
            raise DataValidationError(
                "transition decomposition has no pair with two scorable outcomes"
            )
        counts = Counter(f"{left.outcome.value}:{right.outcome.value}" for left, right in scorable)
        incorrect = sum(left.outcome is Outcome.INCORRECT for left, _right in scorable)
        correct = sum(left.outcome is Outcome.CORRECT for left, _right in scorable)
        if incorrect == 0 or correct == 0:
            raise DataValidationError(
                "transition decomposition requires base correct and incorrect outcomes"
            )
        key = f"{benchmark}|{prompt}|M0_to_{treatment}"
        result[key] = {
            "paired_questions": len(scorable),
            "unscorable_pairs_excluded": excluded,
            "knowledge_recovery": counts["I:C"] / incorrect,
            "abstention_substitution": counts["I:A"] / incorrect,
            "strict_overrefusal": counts["C:A"] / correct,
            "regression": counts["C:I"] / correct,
            "transition_counts": {
                f"{left}:{right}": counts[f"{left}:{right}"]
                for left in ("C", "P", "I", "A")
                for right in ("C", "P", "I", "A")
            },
        }
    return result


def _cell_metric(cell: Mapping[str, GenerationRecord]) -> dict[str, Any]:
    metrics = metric_bundle(tuple(value.outcome for value in cell.values()))
    if metrics.accuracy is None or metrics.coverage is None or metrics.hallucination_risk is None:
        raise DataValidationError("final analysis encountered an undefined risk/coverage cell")
    return {
        "accuracy": metrics.accuracy,
        "coverage": metrics.coverage,
        "risk": metrics.hallucination_risk,
        "question_count": len(cell),
    }


def _validated_selective_risk(record: GenerationRecord) -> tuple[float, str]:
    evidence = record.metadata.get("selective_risk_evidence")
    if not isinstance(evidence, Mapping):
        raise DataValidationError("E9 record lacks signed selective-risk evidence")
    scores = evidence.get("scores")
    if (
        evidence.get("score_semantics") != "frozen-pre-generation-CIA-prompt-risk"
        or not isinstance(scores, Mapping)
        or set(scores) != {"C", "I", "A"}
        or set(record.controller_scores) != {"C", "I", "A"}
    ):
        raise DataValidationError("E9 selective-risk evidence has an invalid schema")
    normalized = {label: float(scores[label]) for label in ("C", "I", "A")}
    controller_artifact = evidence.get("controller_artifact_sha256")
    controller_prompt = evidence.get("controller_prompt_id")
    feature_schema = evidence.get("feature_schema_digest")
    feature_values_digest = evidence.get("feature_values_sha256")
    feature_values = evidence.get("feature_values")
    if (
        any(not math.isfinite(value) or not 0 <= value <= 1 for value in normalized.values())
        or not math.isclose(sum(normalized.values()), 1.0, abs_tol=1e-5)
        or not isinstance(controller_artifact, str)
        or len(controller_artifact) != 64
        or controller_prompt != record.system_prompt_id
        or not isinstance(feature_schema, str)
        or len(feature_schema) != 64
        or not isinstance(feature_values_digest, str)
        or len(feature_values_digest) != 64
        or not isinstance(feature_values, Sequence)
        or isinstance(feature_values, str | bytes)
        or any(
            not math.isclose(
                normalized[label], float(record.controller_scores[label]), abs_tol=1e-12
            )
            for label in normalized
        )
        or not math.isclose(
            float(evidence.get("predicted_hallucination_risk", math.nan)),
            normalized["I"],
            abs_tol=1e-12,
        )
    ):
        raise DataValidationError("E9 selective-risk probabilities do not replay")
    try:
        normalized_features = tuple(float(value) for value in feature_values)
    except (TypeError, ValueError) as exc:
        raise DataValidationError("E9 selective-risk feature values are not numeric") from exc
    if not normalized_features or any(not math.isfinite(value) for value in normalized_features):
        raise DataValidationError("E9 selective-risk feature values are invalid")
    identity = stable_hash(
        {
            "controller_artifact_sha256": controller_artifact,
            "controller_prompt_id": controller_prompt,
            "feature_schema_digest": feature_schema,
            "feature_values_sha256": feature_values_digest,
            "feature_values": normalized_features,
            "scores": normalized,
        }
    )
    return normalized["I"], identity


def _reported_curve_points(
    curve: Sequence[RiskCoveragePoint],
) -> tuple[RiskCoveragePoint, ...]:
    points = tuple(
        point
        for point in curve
        if math.isfinite(point.threshold)
        and point.attempted > 0
        and point.hallucination_risk is not None
    )
    if len(points) < 2 or len({point.coverage for point in points}) < 2:
        raise DataValidationError(
            "selective risk curve requires at least two observed coverage levels"
        )
    return points


def _risk_at_target(points: Sequence[RiskCoveragePoint], target: float) -> dict[str, Any]:
    reached = [point for point in points if point.coverage >= target]
    point = min(reached, key=lambda value: value.coverage) if reached else points[-1]
    assert point.hallucination_risk is not None
    return {
        "target_coverage": target,
        "reached": bool(reached),
        "achieved_coverage": point.coverage,
        "risk": point.hallucination_risk,
        "attempted": point.attempted,
        "incorrect": point.incorrect,
        "threshold": point.threshold,
    }


def _risk_coverage(e9_records: Sequence[GenerationRecord]) -> dict[str, Any]:
    """Derive leakage-free threshold curves from per-question frozen E9 scores."""

    cells = _method_cells(e9_records)
    score_identities: dict[tuple[str, str, str], str] = {}
    curves: dict[tuple[str, str, str], tuple[RiskCoveragePoint, ...]] = {}
    for key, cell in sorted(cells.items()):
        benchmark, prompt, method = key
        if benchmark not in _FACTUAL:
            continue
        examples: list[RiskExample] = []
        for record in cell.values():
            predicted_risk, score_identity = _validated_selective_risk(record)
            identity_key = (benchmark, prompt, record.question_id)
            prior_identity = score_identities.setdefault(identity_key, score_identity)
            if prior_identity != score_identity:
                raise DataValidationError(
                    "E9 selective-risk score differs across methods for the same question"
                )
            examples.append(
                RiskExample(
                    question_id=record.question_id,
                    predicted_risk=predicted_risk,
                    outcome_if_released=record.outcome,
                )
            )
        curve = risk_coverage_curve(examples)
        _reported_curve_points(curve)
        curves[(benchmark, prompt, method)] = curve
    expected = {
        (benchmark, prompt, method)
        for benchmark in _FACTUAL
        for prompt in _PROMPTS
        for method in ("M0", "M1", "M2", "M3", "M4", "M5")
    }
    if set(curves) != expected:
        raise DataValidationError("E9 selective-risk curves do not cover the factorial matrix")

    result: dict[str, Any] = {}
    for benchmark in _FACTUAL:
        for prompt in _PROMPTS:
            family = {
                method: curves[(benchmark, prompt, method)]
                for method in ("M0", "M1", "M2", "M3", "M4", "M5")
            }
            matched_limit, matched_areas = matched_area_under_risk_coverage(family)
            if matched_limit <= 0:
                raise DataValidationError("matched selective-risk domain has zero coverage")
            for method, curve in family.items():
                points = _reported_curve_points(curve)
                maximum_coverage = max(point.coverage for point in curve)
                series_key = f"{benchmark}|{prompt}|{method}"
                result[series_key] = {
                    "aurc": matched_areas[method],
                    "coverage_limit": matched_limit,
                    "full_observed_aurc": area_under_risk_coverage(
                        curve, coverage_limit=maximum_coverage
                    ),
                    "maximum_coverage": maximum_coverage,
                    "point_count": len(points),
                    "target_risks": {
                        f"coverage_{int(target * 100):02d}": _risk_at_target(points, target)
                        for target in (0.25, 0.50, 0.75, 0.90)
                    },
                    "points": {
                        f"point_{index:03d}": {
                            "threshold": point.threshold,
                            "coverage": point.coverage,
                            "risk": point.hallucination_risk,
                            "accuracy": point.accuracy,
                            "attempted": point.attempted,
                            "incorrect": point.incorrect,
                        }
                        for index, point in enumerate(points)
                    },
                }
    return result


def _layer_alpha_surface(records: Sequence[GenerationRecord]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for (_, benchmark, prompt, method), cell in _condition_cells(records).items():
        representative = next(iter(cell.values()))
        if benchmark not in _FACTUAL or method == "M0" or representative.layer is None:
            continue
        trace = representative.metadata.get("intervention_trace")
        if not isinstance(trace, Mapping):
            trace = {}
        stage = str(
            representative.metadata.get(
                "e3_stage", representative.metadata.get("stage", "unclassified")
            )
        )
        training_prompt = str(trace.get("training_prompt_id", "unclassified"))
        extraction = str(trace.get("extraction_method", method))
        control = str(trace.get("control") or "none")
        site = representative.site.value if representative.site is not None else "none"
        scope = (
            representative.token_scope.value if representative.token_scope is not None else "none"
        )
        metrics = _cell_metric(cell)
        key = (
            f"{stage}|{benchmark}|apply_{prompt}|train_{training_prompt}|{method}|"
            f"extract_{extraction}|control_{control}|site_{site}|scope_{scope}|"
            f"layer_{representative.layer}|alpha_{representative.alpha:.12g}|"
            f"{stable_hash(representative.condition_id)}"
        )
        if key in result:
            raise DataValidationError("layer/alpha surface contains a duplicate point")
        result[key] = {
            **metrics,
            "layer": representative.layer,
            "alpha": representative.alpha,
            "condition_id": representative.condition_id,
            "stage_sha256": stable_hash(stage),
            "apply_prompt_sha256": stable_hash(prompt),
            "training_prompt_sha256": stable_hash(training_prompt),
            "method_sha256": stable_hash(method),
            "extraction_sha256": stable_hash(extraction),
            "control_sha256": stable_hash(control),
            "site_sha256": stable_hash(site),
            "token_scope_sha256": stable_hash(scope),
        }
    if not result:
        raise DataValidationError("E3 records contain no layer/alpha evaluation surface")
    return result


def _validated_layer_alpha_surface(
    surface: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
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
    if not surface:
        raise DataValidationError("E3 analysis surface is empty")
    result: dict[str, Any] = {}
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
            raise DataValidationError("E3 analysis surface contains an invalid cell")
        result[key] = dict(raw)
    return result


def _matched_coverage(records: Sequence[GenerationRecord]) -> dict[str, Any]:
    """Report all four preregistered matching dimensions, not coverage alone."""

    def intervention_norm(record: GenerationRecord) -> float:
        trace = record.metadata.get("intervention_trace")
        if isinstance(trace, Mapping):
            value = trace.get("activation_delta_norm")
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise DataValidationError("E9 intervention trace lacks a valid norm")
            return float(value)
        if trace is not None:
            raise DataValidationError("E9 intervention norm evidence is malformed")
        if record.steering_method == "M3" and record.metadata.get("policy_action") in {
            "release",
            "abstain",
        }:
            return 0.0
        fallback = record.metadata.get("intervention_norm")
        if (
            isinstance(fallback, bool)
            or not isinstance(fallback, int | float)
            or not math.isfinite(float(fallback))
            or float(fallback) < 0
        ):
            raise DataValidationError("E9 record lacks executed intervention-norm evidence")
        return float(fallback)

    def controls(values: Mapping[str, GenerationRecord]) -> dict[str, float]:
        records_in_cell = tuple(values.values())
        scorable = tuple(
            value for value in records_in_cell if value.outcome is not Outcome.UNSCORABLE
        )
        if not scorable:
            raise DataValidationError("matched controls have no scorable rows")
        return {
            "abstention_rate": sum(value.outcome is Outcome.ABSTENTION for value in scorable)
            / len(scorable),
            "mean_intervention_norm": fmean(intervention_norm(value) for value in scorable),
            "mean_latency_seconds": fmean(value.generation_latency_seconds for value in scorable),
        }

    cells = _method_cells(records)
    result: dict[str, Any] = {}
    for benchmark in _FACTUAL:
        for prompt in _PROMPTS:
            baseline_cell = cells[(benchmark, prompt, "M1")]
            baseline = _cell_metric(baseline_cell)
            baseline_controls = controls(baseline_cell)
            for treatment in ("M3", "M4", "M5"):
                target_cell = cells[(benchmark, prompt, treatment)]
                target = _cell_metric(target_cell)
                target_controls = controls(target_cell)
                result[f"{benchmark}|{prompt}|{treatment}_vs_M1"] = {
                    "coverage": float(target["coverage"]),
                    "baseline_coverage": float(baseline["coverage"]),
                    "coverage_mismatch": abs(
                        float(target["coverage"]) - float(baseline["coverage"])
                    ),
                    "risk_difference": float(target["risk"]) - float(baseline["risk"]),
                    "baseline_risk": float(baseline["risk"]),
                    "treatment_risk": float(target["risk"]),
                    "paired_questions": int(target["question_count"]),
                    "abstention_rate": target_controls["abstention_rate"],
                    "baseline_abstention_rate": baseline_controls["abstention_rate"],
                    "abstention_rate_mismatch": abs(
                        target_controls["abstention_rate"] - baseline_controls["abstention_rate"]
                    ),
                    "mean_intervention_norm": target_controls["mean_intervention_norm"],
                    "baseline_mean_intervention_norm": baseline_controls["mean_intervention_norm"],
                    "intervention_norm_mismatch": abs(
                        target_controls["mean_intervention_norm"]
                        - baseline_controls["mean_intervention_norm"]
                    ),
                    "mean_latency_seconds": target_controls["mean_latency_seconds"],
                    "baseline_mean_latency_seconds": baseline_controls["mean_latency_seconds"],
                    "latency_mismatch_seconds": abs(
                        target_controls["mean_latency_seconds"]
                        - baseline_controls["mean_latency_seconds"]
                    ),
                }
    return result


def _likelihood_changes(records: Sequence[GenerationRecord]) -> dict[str, Any]:
    cells = _method_cells(records)
    result: dict[str, Any] = {}
    for benchmark in _FACTUAL:
        for prompt in (*_PROMPTS, "P3-forced-answer"):
            for treatment in ("M1", "M3"):
                identifiers, before, after = _paired_cells(
                    cells,
                    benchmark=benchmark,
                    prompt=prompt,
                    baseline="M0",
                    treatment=treatment,
                )
                gold_changes: list[float] = []
                abstention_changes: list[float] = []
                baseline_ranks: list[int] = []
                treatment_ranks: list[int] = []
                for baseline, target in zip(before, after, strict=True):
                    try:
                        gold_changes.append(
                            float(target.metadata["gold_alias_log_likelihood"])
                            - float(baseline.metadata["gold_alias_log_likelihood"])
                        )
                        abstention_changes.append(
                            float(target.metadata["abstention_log_likelihood"])
                            - float(baseline.metadata["abstention_log_likelihood"])
                        )
                    except (KeyError, TypeError, ValueError) as exc:
                        raise DataValidationError(
                            "E6 records lack bound gold/abstention likelihoods"
                        ) from exc
                    baseline_rank = baseline.metadata.get("gold_answer_rank")
                    target_rank = target.metadata.get("gold_answer_rank")
                    if (baseline_rank is None) != (target_rank is None):
                        raise DataValidationError("E6 paired gold-rank evidence is incomplete")
                    if baseline_rank is not None:
                        if (
                            type(baseline_rank) is not int
                            or type(target_rank) is not int
                            or baseline_rank <= 0
                            or target_rank <= 0
                        ):
                            raise DataValidationError("E6 gold-answer rank is invalid")
                        baseline_ranks.append(baseline_rank)
                        treatment_ranks.append(target_rank)
                baseline_correct = sum(value.outcome is Outcome.CORRECT for value in before) / len(
                    before
                )
                treatment_correct = sum(value.outcome is Outcome.CORRECT for value in after) / len(
                    after
                )
                baseline_abstention = sum(
                    value.outcome is Outcome.ABSTENTION for value in before
                ) / len(before)
                treatment_abstention = sum(
                    value.outcome is Outcome.ABSTENTION for value in after
                ) / len(after)
                result[f"{benchmark}|{prompt}|M0_to_{treatment}"] = {
                    "gold": fmean(gold_changes),
                    "abstention": fmean(abstention_changes),
                    "gold_median": float(np.median(gold_changes)),
                    "abstention_median": float(np.median(abstention_changes)),
                    "paired_questions": len(identifiers),
                    "baseline_accuracy": baseline_correct,
                    "treatment_accuracy": treatment_correct,
                    "accuracy_difference": treatment_correct - baseline_correct,
                    "baseline_abstention_rate": baseline_abstention,
                    "treatment_abstention_rate": treatment_abstention,
                    "abstention_rate_difference": (treatment_abstention - baseline_abstention),
                    "forced_answer_condition": prompt == "P3-forced-answer",
                    "rank_eligible_questions": len(baseline_ranks),
                    "baseline_mean_gold_rank": (fmean(baseline_ranks) if baseline_ranks else 0.0),
                    "treatment_mean_gold_rank": (
                        fmean(treatment_ranks) if treatment_ranks else 0.0
                    ),
                    "mean_gold_rank_change": (
                        fmean(
                            right - left
                            for left, right in zip(baseline_ranks, treatment_ranks, strict=True)
                        )
                        if baseline_ranks
                        else 0.0
                    ),
                }
    return result


def _rq1_generalization_section(
    results: Mapping[str, RQ1GeneralizationResult],
) -> dict[str, Any]:
    if not results:
        raise DataValidationError("final analysis lacks RQ1 generalization results")
    section: dict[str, Any] = {}
    for comparison, result in sorted(results.items()):
        if not comparison or result.evaluation_record_count <= 0:
            raise DataValidationError("RQ1 generalization result is invalid")
        section[comparison] = {
            "task_identity_sha256": stable_hash(result.task_id),
            "question_set_digests": dict(result.question_set_digests),
            "artifact_fingerprints": dict(result.artifact_fingerprints),
            "evaluation_record_count": result.evaluation_record_count,
            "metrics": dict(result.metrics),
            "result_digest": result.result_digest,
        }
    regimes = set()
    for comparison in section:
        if "|calibration-only|" in comparison:
            regimes.add("calibration-only")
        if "|full-vector-bank-relearning|" in comparison:
            regimes.add("full-vector-bank-relearning")
    if regimes != {"calibration-only", "full-vector-bank-relearning"}:
        raise DataValidationError(
            "RQ1 results lack calibration-only or full-relearning comparisons"
        )
    return section


def _e7_interpretability_section(run_directory: str | Path) -> dict[str, Any]:
    """Replay and summarize the exact promoted E7 SAE evidence.

    Raw question identifiers are represented by SHA-256 identities so the
    quantitative result tree remains publication-safe while still binding the
    reported rank order to the reviewed examples.
    """

    from mfh.methods.sparse import load_sae_intervention

    artifact_directory = (
        Path(run_directory) / "gate-artifacts" / "individual_causal_evidence" / "sae-intervention"
    )
    sae = load_sae_intervention(artifact_directory)
    audit = sae.interpretability_audit
    if (
        audit is None
        or audit.prompt_transfer_execution is None
        or audit.negative_control_execution is None
    ):
        raise DataValidationError("E7 final analysis lacks native interpretability evidence")
    features: dict[str, Any] = {}
    evidence_by_feature = {value.feature_index: value for value in sae.evidence}
    for feature in sae.latent_direction.selected_features:
        evidence = evidence_by_feature[feature]
        top_examples = audit.top_activating_question_ids[feature]
        transfer = audit.prompt_transfer_execution[feature]
        controls = audit.negative_control_execution[feature]
        features[f"feature_{feature}"] = {
            "feature_index": feature,
            "top_activating_examples": {
                f"rank_{rank:03d}": stable_hash(question_id)
                for rank, question_id in enumerate(top_examples, start=1)
            },
            "prompt_transfer_effects": dict(audit.prompt_transfer_effects[feature]),
            "prompt_transfer_sample_counts": {
                name: len(value.baseline_records) for name, value in transfer.items()
            },
            "negative_control_effects": dict(audit.negative_control_effects[feature]),
            "negative_control_sample_counts": {
                name: len(value.baseline_records) for name, value in controls.items()
            },
            "activation_factuality_delta": evidence.activation_factuality_delta,
            "suppression_factuality_delta": evidence.suppression_factuality_delta,
            "protected_behavior_deltas": dict(evidence.protected_behavior_deltas),
            "factuality_sample_count": evidence.spec.factuality_sample_count,
            "protected_sample_counts": dict(evidence.spec.protected_sample_counts),
            "causal_evidence_spec_sha256": evidence.spec.digest,
        }
    receipt = sae.long_computation_receipt
    result: dict[str, Any] = {
        "artifact_sha256": sha256_path(artifact_directory),
        "source_question_bundle_sha256": audit.source_question_bundle_sha256,
        "evaluation_question_set_sha256": stable_hash(audit.evaluation_question_ids),
        "evaluation_question_count": len(audit.evaluation_question_ids),
        "control_seed": audit.control_seed,
        "reconstruction": {
            "reconstruction_mse": sae.training.metrics.reconstruction_mse,
            "fraction_variance_explained": (sae.training.metrics.fraction_variance_explained),
            "average_active_features": sae.training.metrics.average_active_features,
            "training_rows": sae.training.training_rows,
            "validation_rows": sae.training.validation_rows,
        },
        "feature_stability": sae.feature_stability,
        "selected_feature_count": len(sae.latent_direction.selected_features),
        "features": features,
    }
    if receipt is not None:
        result["measured_execution"] = {
            "wall_time_seconds": receipt.wall_time_seconds,
            "peak_gpu_memory_bytes": receipt.peak_gpu_memory_bytes,
            "package_lock_sha256": receipt.package_lock_sha256,
            "model_snapshot_sha256": receipt.model_snapshot_sha256,
            "runtime_artifact_sha256": receipt.runtime_artifact_sha256,
            "resumable_chain_head": receipt.resumable_chain_head,
        }
    return result


def _side_metric(record: GenerationRecord, metric: str) -> float:
    if record.benchmark != _SIDE_METRIC_BENCHMARK[metric]:
        raise DataValidationError("side-effect metric uses the wrong benchmark")
    if metric == "ifeval_pass_rate":
        value = record.metadata.get("official_pass")
        if not isinstance(value, bool):
            raise DataValidationError("IFEval record lacks an official pass decision")
        return float(value)
    if metric == "mmlu_pro_accuracy":
        value = record.metadata.get("official_correct")
        if not isinstance(value, bool) or value is not (record.outcome is Outcome.CORRECT):
            raise DataValidationError("MMLU-Pro record lacks a consistent official decision")
        return float(value)
    if metric == "xstest_benign_non_refusal_rate":
        refused = record.metadata.get("refused")
        if not isinstance(refused, bool):
            raise DataValidationError("XSTest record lacks a refusal decision")
        return float(not refused)
    if metric == "harmful_prompt_refusal_rate":
        refused = record.metadata.get("refused")
        if not isinstance(refused, bool):
            raise DataValidationError("harmful-prompt record lacks a refusal decision")
        return float(refused)
    if metric == "harmful_prompt_compliance_score":
        value = record.metadata.get("harmful_compliance_score")
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise DataValidationError("harmful-prompt record lacks a compliance score")
        return float(value)
    if metric == "requested_language_consistency":
        value = record.metadata.get("requested_language_correct")
        if not isinstance(value, bool):
            raise DataValidationError("language record lacks a consistency decision")
        return float(value)
    if metric == "latency_relative_change":
        return record.generation_latency_seconds
    value = record.metadata.get("negative_log_likelihood")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DataValidationError("WikiText record lacks token negative log likelihood")
    return math.exp(float(value))


def _noninferiority(
    records: Sequence[GenerationRecord], protocol: AnalysisProtocol
) -> dict[str, Any]:
    cells = _method_cells(records)
    result: dict[str, Any] = {}
    raw_family: list[tuple[str, float]] = []
    for metric, specification in protocol.noninferiority_margins.items():
        benchmark = _SIDE_METRIC_BENCHMARK[metric]
        comparisons: dict[str, Any] = {}
        for prompt in _PROMPTS:
            for treatment in ("M1", "M3", "M4", "M5"):
                identifiers, before, after = _paired_cells(
                    cells,
                    benchmark=benchmark,
                    prompt=prompt,
                    baseline="M0",
                    treatment=treatment,
                )
                baseline_values = [_side_metric(value, metric) for value in before]
                treatment_values = [_side_metric(value, metric) for value in after]
                if specification.scale is MarginScale.RELATIVE_FRACTION:
                    if any(value <= 0 for value in baseline_values):
                        raise DataValidationError(
                            "relative non-inferiority baseline must be positive"
                        )
                    treatment_values = [
                        target / baseline
                        for baseline, target in zip(baseline_values, treatment_values, strict=True)
                    ]
                    baseline_values = [1.0] * len(identifiers)
                inference = paired_noninferiority(
                    identifiers,
                    baseline_values,
                    treatment_values,
                    margin=specification.margin,
                    higher_is_better=specification.higher_is_better,
                    resamples=protocol.bootstrap_resamples,
                    confidence=protocol.confidence,
                    seed=17,
                )
                comparison_key = f"{prompt}|{treatment}_vs_M0"
                comparisons[comparison_key] = {
                    "estimate": inference.oriented_difference,
                    "one_sided_lower": inference.one_sided_lower,
                    "raw_p_value": inference.p_value,
                    "non_inferior": inference.non_inferior,
                    "questions": inference.questions,
                }
                raw_family.append((f"{metric}|{comparison_key}", inference.p_value))
        worst_key = min(
            comparisons,
            key=lambda key: (float(comparisons[key]["one_sided_lower"]), key),
        )
        worst = comparisons[worst_key]
        result[metric] = {
            "estimate": float(worst["estimate"]),
            "margin": specification.margin,
            "higher_is_better": specification.higher_is_better,
            "one_sided_lower": float(worst["one_sided_lower"]),
            "p_value": max(float(value["raw_p_value"]) for value in comparisons.values()),
            "passed": False,
            "comparison_count": len(comparisons),
            "worst_comparison_sha256": stable_hash(worst_key),
            "comparisons": comparisons,
        }
    adjusted = {value.hypothesis: value for value in holm_adjust(raw_family, alpha=protocol.alpha)}
    for metric, metric_result in result.items():
        comparisons = metric_result["comparisons"]
        assert isinstance(comparisons, dict)
        for comparison_key, comparison in comparisons.items():
            assert isinstance(comparison, dict)
            correction = adjusted[f"{metric}|{comparison_key}"]
            comparison["adjusted_p_value"] = correction.adjusted_p_value
            comparison["rejected"] = correction.rejected
            comparison["passed"] = bool(comparison["non_inferior"]) and correction.rejected
        metric_result["p_value"] = max(
            float(value["adjusted_p_value"]) for value in comparisons.values()
        )
        metric_result["passed"] = all(bool(value["passed"]) for value in comparisons.values())
    return result


def _composite_side_effects(
    e8_records: Sequence[GenerationRecord],
    e10_records: Sequence[GenerationRecord],
    protocol: AnalysisProtocol,
) -> dict[str, Any]:
    """Test final M6 against the exactly paired E8 unsteered baseline."""

    baseline_cells = _method_cells(e8_records)
    composite: dict[tuple[str, str], GenerationRecord] = {}
    prompts = {
        record.system_prompt_id
        for record in e10_records
        if record.benchmark in set(_SIDE_METRIC_BENCHMARK.values())
    }
    if len(prompts) != 1:
        raise DataValidationError("E10 side-effect suite must use one frozen selected prompt")
    prompt = next(iter(prompts))
    for record in e10_records:
        if record.benchmark not in set(_SIDE_METRIC_BENCHMARK.values()):
            continue
        key = (record.benchmark, record.question_id)
        if key in composite or record.steering_method != "M6":
            raise DataValidationError("E10 composite side-effect rows are invalid")
        composite[key] = record

    metrics: dict[str, Any] = {}
    raw_family: list[tuple[str, float]] = []
    for metric, specification in protocol.noninferiority_margins.items():
        benchmark = _SIDE_METRIC_BENCHMARK[metric]
        try:
            baseline = baseline_cells[(benchmark, prompt, "M0")]
        except KeyError as exc:
            raise DataValidationError("E8 lacks the frozen-prompt M0 baseline for E10") from exc
        treatment = {
            question_id: record
            for (record_benchmark, question_id), record in composite.items()
            if record_benchmark == benchmark
        }
        if set(baseline) != set(treatment) or not baseline:
            raise DataValidationError(f"E10 {benchmark} rows are not exactly paired to E8 M0")
        identifiers = tuple(sorted(baseline))
        baseline_values = [_side_metric(baseline[value], metric) for value in identifiers]
        treatment_values = [_side_metric(treatment[value], metric) for value in identifiers]
        reported_baseline = fmean(baseline_values)
        reported_treatment = fmean(treatment_values)
        if specification.scale is MarginScale.RELATIVE_FRACTION:
            if any(value <= 0 for value in baseline_values):
                raise DataValidationError("M6 relative non-inferiority baseline must be positive")
            treatment_values = [
                target / base
                for base, target in zip(baseline_values, treatment_values, strict=True)
            ]
            baseline_values = [1.0] * len(identifiers)
        inference = paired_noninferiority(
            identifiers,
            baseline_values,
            treatment_values,
            margin=specification.margin,
            higher_is_better=specification.higher_is_better,
            resamples=protocol.bootstrap_resamples,
            confidence=protocol.confidence,
            seed=29,
        )
        metrics[metric] = {
            "baseline_mean": reported_baseline,
            "composite_mean": reported_treatment,
            "estimate": inference.oriented_difference,
            "margin": specification.margin,
            "higher_is_better": specification.higher_is_better,
            "one_sided_lower": inference.one_sided_lower,
            "raw_p_value": inference.p_value,
            "questions": inference.questions,
            "non_inferior": inference.non_inferior,
        }
        raw_family.append((metric, inference.p_value))
    corrections = {
        value.hypothesis: value for value in holm_adjust(raw_family, alpha=protocol.alpha)
    }
    for metric, value in metrics.items():
        correction = corrections[metric]
        value["adjusted_p_value"] = correction.adjusted_p_value
        value["rejected"] = correction.rejected
        value["passed"] = bool(value["non_inferior"]) and correction.rejected

    def safety_state(record: GenerationRecord) -> str:
        refused = record.metadata.get("refused")
        if not isinstance(refused, bool):
            raise DataValidationError("safety transition record lacks refusal evidence")
        if record.benchmark == "xstest":
            return "REFUSE" if refused else "ANSWER"
        compliance = record.metadata.get("harmful_compliance_score")
        if isinstance(compliance, bool) or not isinstance(compliance, int | float):
            raise DataValidationError("harmful transition lacks compliance evidence")
        if refused:
            return "REFUSE"
        return "COMPLY" if float(compliance) > 0 else "OTHER"

    transitions: dict[str, Any] = {}
    for benchmark in ("xstest", "strongreject_or_harmbench"):
        baseline = baseline_cells[(benchmark, prompt, "M0")]
        treatment = {
            question_id: record
            for (record_benchmark, question_id), record in composite.items()
            if record_benchmark == benchmark
        }
        counts = Counter(
            f"{safety_state(baseline[question_id])}:{safety_state(treatment[question_id])}"
            for question_id in sorted(baseline)
        )
        labels = ("ANSWER", "REFUSE") if benchmark == "xstest" else ("REFUSE", "COMPLY", "OTHER")
        transitions[benchmark] = {
            "paired_questions": len(baseline),
            "transition_counts": {
                f"{left}:{right}": counts[f"{left}:{right}"] for left in labels for right in labels
            },
        }
    return {
        "selected_prompt_sha256": stable_hash(prompt),
        "metrics": metrics,
        "safety_transitions": transitions,
        "all_preregistered_noninferiority_tests_passed": all(
            bool(value["passed"]) for value in metrics.values()
        ),
    }


def _pareto(records: Sequence[GenerationRecord], protocol: AnalysisProtocol) -> dict[str, Any]:
    cells = _method_cells(records)
    result: dict[str, Any] = {}
    for prompt in _PROMPTS:
        baseline_risk = float(_cell_metric(cells[("triviaqa", prompt, "M0")])["risk"])
        for method in ("M1", "M3", "M4", "M5"):
            risk = float(_cell_metric(cells[("triviaqa", prompt, method)])["risk"])
            metric_slacks: dict[str, Any] = {}
            for metric, specification in protocol.noninferiority_margins.items():
                benchmark = _SIDE_METRIC_BENCHMARK[metric]
                identifiers, before, after = _paired_cells(
                    cells,
                    benchmark=benchmark,
                    prompt=prompt,
                    baseline="M0",
                    treatment=method,
                )
                baseline_values = [_side_metric(value, metric) for value in before]
                treatment_values = [_side_metric(value, metric) for value in after]
                if specification.scale is MarginScale.RELATIVE_FRACTION:
                    if any(value <= 0 for value in baseline_values):
                        raise DataValidationError(
                            "Pareto relative non-inferiority baseline must be positive"
                        )
                    treatment_values = [
                        target / baseline
                        for baseline, target in zip(baseline_values, treatment_values, strict=True)
                    ]
                    baseline_values = [1.0] * len(identifiers)
                inference = paired_noninferiority(
                    identifiers,
                    baseline_values,
                    treatment_values,
                    margin=specification.margin,
                    higher_is_better=specification.higher_is_better,
                    resamples=protocol.bootstrap_resamples,
                    confidence=protocol.confidence,
                    seed=17,
                )
                metric_slacks[metric] = {
                    "one_sided_lower": inference.one_sided_lower,
                    "margin": specification.margin,
                    "normalized_margin_slack": (inference.one_sided_lower + specification.margin)
                    / specification.margin,
                    "passed": inference.non_inferior,
                }
            worst_metric = min(
                metric_slacks,
                key=lambda name: (float(metric_slacks[name]["normalized_margin_slack"]), name),
            )
            worst = metric_slacks[worst_metric]
            result[f"{prompt}|{method}"] = {
                "risk": risk,
                "factuality_gain": baseline_risk - risk,
                "minimum_normalized_noninferiority_slack": worst["normalized_margin_slack"],
                "worst_one_sided_lower": worst["one_sided_lower"],
                "worst_margin": worst["margin"],
                "worst_metric_sha256": stable_hash(worst_metric),
                "all_noninferior": all(bool(value["passed"]) for value in metric_slacks.values()),
                "metric_slacks": metric_slacks,
            }
    return result


def _prompt_paraphrase(
    values: Sequence[VerifiedPromptParaphraseRecord],
) -> dict[str, Any]:
    if not values:
        raise DataValidationError("final analysis requires prompt-paraphrase records")
    cells: dict[tuple[str, str, str, str], list[GenerationRecord]] = defaultdict(list)
    seen: set[str] = set()
    for value in values:
        if value.task_id in seen:
            raise DataValidationError("prompt-paraphrase records repeat a task")
        seen.add(value.task_id)
        cells[
            (
                value.benchmark,
                value.base_prompt_id,
                value.method,
                value.paraphrase_prompt_id,
            )
        ].append(value.record)
    grouped: dict[tuple[str, str, str], list[dict[str, float]]] = defaultdict(list)
    for (benchmark, base_prompt, method, _), records in cells.items():
        metrics = metric_bundle(tuple(value.outcome for value in records))
        if metrics.accuracy is None or metrics.hallucination_risk is None:
            raise DataValidationError("prompt paraphrase cell has undefined risk")
        grouped[(benchmark, base_prompt, method)].append(
            {"accuracy": metrics.accuracy, "risk": metrics.hallucination_risk}
        )
    result: dict[str, Any] = {}
    for key, points in sorted(grouped.items()):
        accuracies = [value["accuracy"] for value in points]
        risks = [value["risk"] for value in points]
        result["|".join(key)] = {
            "variance": float(np.var(accuracies)),
            "mean_accuracy": fmean(accuracies),
            "minimum_accuracy": min(accuracies),
            "maximum_accuracy": max(accuracies),
            "maximum_risk": max(risks),
            "prompt_count": len(points),
            "question_count": sum(
                len(records) for cell_key, records in cells.items() if cell_key[:3] == key
            ),
        }
    return result


def _language_confusion(
    records: Sequence[GenerationRecord],
    baseline_records: Sequence[GenerationRecord],
    audit: HumanAuditResults,
) -> dict[str, Any]:
    matrix: Counter[str] = Counter()
    grouped: dict[str, list[GenerationRecord]] = defaultdict(list)
    for record in records:
        if record.benchmark != "language_consistency":
            continue
        requested = record.metadata.get("requested_language")
        detected = record.metadata.get("detected_language")
        evidence = record.metadata.get("language_evaluation_evidence")
        aliases = evidence.get("accepted_aliases") if isinstance(evidence, Mapping) else None
        if (
            requested not in SUPPORTED_LANGUAGES
            or not isinstance(aliases, list)
            or not aliases
            or any(not isinstance(value, str) or not value for value in aliases)
            or not isinstance(evidence, Mapping)
            or dict(evidence)
            != language_response_evidence(record.raw_output, str(requested), tuple(aliases))
        ):
            raise DataValidationError("E10 language record lacks a requested language")
        if detected is not None and (not isinstance(detected, str) or not detected):
            raise DataValidationError("E10 language record has an invalid detected language")
        matrix[f"{requested}:{detected or 'und'}"] += 1
        grouped[str(requested)].append(record)
    by_language: dict[str, dict[str, Any]] = {}
    for language, values in sorted(grouped.items()):
        rows = len(values)
        by_language[language] = {
            "rows": rows,
            "correct_output_language_rate": sum(
                value.metadata.get("requested_language_correct") is True for value in values
            )
            / rows,
            "non_target_script_token_rate": fmean(
                float(value.metadata["non_target_script_token_rate"]) for value in values
            ),
            "code_switching_rate": sum(
                value.metadata.get("code_switching") is True for value in values
            )
            / rows,
            "factual_accuracy": sum(value.outcome is Outcome.CORRECT for value in values) / rows,
            "abstention_rate": sum(value.outcome is Outcome.ABSTENTION for value in values) / rows,
        }
    baseline_index = {
        (
            value.model_repository,
            value.system_prompt_id,
            value.question_id,
        ): value
        for value in baseline_records
        if value.benchmark == "language_consistency" and value.steering_method == "M0"
    }
    eligible = 0
    wrong = 0
    transition_by_language: dict[str, dict[str, Any]] = {}
    for language, values in sorted(grouped.items()):
        language_eligible = 0
        language_wrong = 0
        for value in values:
            key = (value.model_repository, value.system_prompt_id, value.question_id)
            baseline = baseline_index.get(key)
            if baseline is None:
                raise DataValidationError("E10 language record lacks its paired M0 baseline")
            if (
                baseline.metadata.get("requested_language") != language
                or baseline.metadata.get("requested_language_correct") is not True
                or baseline.outcome is not Outcome.CORRECT
            ):
                continue
            language_eligible += 1
            language_wrong += int(value.metadata.get("requested_language_correct") is not True)
        eligible += language_eligible
        wrong += language_wrong
        transition_by_language[language] = {
            "eligible_baseline_correct_language_rows": language_eligible,
            "wrong_language_transitions": language_wrong,
            "rate": language_wrong / language_eligible if language_eligible else 0.0,
        }
    payload = audit.summary.get("language_reporting_payload")
    if not isinstance(payload, Mapping):
        raise DataValidationError("human audit lacks adjudicated language evidence")
    adjudication = payload.get("adjudication_summary")
    if not matrix or not isinstance(adjudication, Mapping) or int(adjudication.get("rows", 0)) <= 0:
        raise DataValidationError("language report requires non-empty automated and human evidence")
    return {
        "requested_detected_matrix": dict(sorted(matrix.items())),
        "automated_metrics_by_language": by_language,
        "correct_to_wrong_language": {
            "eligible_baseline_correct_language_rows": eligible,
            "wrong_language_transitions": wrong,
            "rate": wrong / eligible if eligible else 0.0,
            "by_language": transition_by_language,
        },
        "human_audit": dict(payload),
    }


def _runtime_replication(
    records: Sequence[GenerationRecord], runtime_identity: Mapping[str, Any]
) -> dict[str, Any]:
    from mfh.experiments.e6_likelihood import _validate_e6_runtime_identity

    _validate_e6_runtime_identity(runtime_identity)
    runtime_identity_digest = stable_hash(runtime_identity)
    latencies = [value.generation_latency_seconds for value in records]
    peaks: list[int] = []
    signed = 0
    session_identities: list[str] = []
    site_evidence: list[dict[str, Any]] = []
    site_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    delta_norms: dict[str, list[float]] = defaultdict(list)
    prompt_rates: list[float] = []
    generation_rates: list[float] = []
    candidate_latencies: list[float] = []
    pre_generation_abstentions = 0
    for record in records:
        controller_evidence = record.metadata.get("adaptive_controller_evidence")
        if isinstance(controller_evidence, Mapping):
            site_evidence.append(
                {
                    "site_selection": controller_evidence.get("site_selection"),
                    "feature_schema_digest": controller_evidence.get("feature_schema_digest"),
                    "controller_artifact_sha256": controller_evidence.get(
                        "controller_artifact_sha256"
                    ),
                }
            )
        runtime_metrics = record.metadata.get("generation_runtime_metrics")
        if not isinstance(runtime_metrics, Mapping):
            raise DataValidationError("E10 record lacks runtime measurement evidence")
        wall_seconds = runtime_metrics.get("end_to_end_wall_seconds")
        candidate_generated = runtime_metrics.get("candidate_generated")
        candidate_seconds = runtime_metrics.get("candidate_generation_seconds")
        if (
            isinstance(wall_seconds, bool)
            or not isinstance(wall_seconds, int | float)
            or not math.isclose(
                float(wall_seconds),
                record.generation_latency_seconds,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or record.generation_latency_seconds <= 0
            or not isinstance(candidate_generated, bool)
            or isinstance(candidate_seconds, bool)
            or not isinstance(candidate_seconds, int | float)
            or not math.isfinite(float(candidate_seconds))
            or float(candidate_seconds) < 0
        ):
            raise DataValidationError("E10 wall/candidate runtime evidence does not replay")
        if candidate_generated:
            if float(candidate_seconds) <= 0:
                raise DataValidationError("generated E10 candidate has zero runtime")
            candidate_latencies.append(float(candidate_seconds))
        else:
            if float(candidate_seconds) != 0:
                raise DataValidationError("pre-generation abstention has candidate runtime")
            pre_generation_abstentions += 1
        peak = runtime_metrics.get("peak_memory_bytes")
        if isinstance(peak, int) and not isinstance(peak, bool) and peak > 0:
            peaks.append(peak)
        prompt_rate = runtime_metrics.get("prompt_tokens_per_second")
        generation_rate = runtime_metrics.get("generation_tokens_per_second")
        if candidate_generated:
            if (
                isinstance(prompt_rate, bool)
                or not isinstance(prompt_rate, int | float)
                or float(prompt_rate) <= 0
                or isinstance(generation_rate, bool)
                or not isinstance(generation_rate, int | float)
                or float(generation_rate) <= 0
            ):
                raise DataValidationError("generated E10 candidate lacks throughput evidence")
            prompt_rates.append(float(prompt_rate))
            generation_rates.append(float(generation_rate))
        elif prompt_rate != 0 or generation_rate != 0:
            raise DataValidationError("pre-generation abstention has generated-token throughput")
        session = record.metadata.get("runtime_session_identity_sha256")
        if session == runtime_identity_digest:
            session_identities.append(session)
        action = record.metadata.get("policy_action")
        if isinstance(action, str) and action:
            action_counts[action] += 1
        if action not in {"release", "intervene", "abstain"} or candidate_generated is not (
            action != "abstain"
        ):
            raise DataValidationError("E10 runtime candidate flag differs from policy routing")
        site = record.site.value if record.site is not None else "no-intervention"
        site_counts[site] += 1
        trace = record.metadata.get("intervention_trace")
        if isinstance(trace, Mapping):
            delta = trace.get("activation_delta_norm")
            if (
                isinstance(delta, int | float)
                and not isinstance(delta, bool)
                and math.isfinite(float(delta))
                and float(delta) >= 0
            ):
                delta_norms[site].append(float(delta))
        signed += int(
            all(
                isinstance(record.metadata.get(key), str) and len(str(record.metadata[key])) == 128
                for key in (
                    "execution_receipt_signature",
                    "confirmatory_execution_receipt_signature",
                )
            )
        )
    runtime_counts = Counter(value.runtime.value for value in records)
    identity = sorted(
        {
            (
                value.model_repository,
                value.model_revision,
                value.quantization,
                value.runtime.value,
            )
            for value in records
        }
    )
    if not (len(prompt_rates) == len(generation_rates) == len(candidate_latencies)):
        raise DataValidationError("E10 generated-row runtime evidence is incomplete")
    mean_delta_by_site = {site: fmean(values) for site, values in sorted(delta_norms.items())}
    if not mean_delta_by_site:
        mean_delta_by_site = {"no-intervention": 0.0}
    passed = (
        set(runtime_counts) == {Runtime.VLLM.value}
        and len(identity) == 1
        and signed == len(records)
        and bool(peaks)
        and len(peaks) == len(records)
        and len(session_identities) == len(records)
        and len(site_evidence) == len(records)
    )
    return {
        "local_vllm_execution": {
            "passed": passed,
            "record_count": len(records),
            "mean_latency_seconds": fmean(latencies),
            "p95_latency_seconds": float(np.quantile(latencies, 0.95)),
            "maximum_latency_seconds": max(latencies),
            "mean_candidate_generation_seconds": (
                fmean(candidate_latencies) if candidate_latencies else 0.0
            ),
            "candidate_generated_record_count": len(candidate_latencies),
            "pre_generation_abstention_count": pre_generation_abstentions,
            "maximum_peak_memory_bytes": max(peaks, default=0),
            "signed_execution_receipts": signed,
            "mean_prompt_tokens_per_second": fmean(prompt_rates) if prompt_rates else 0.0,
            "mean_generation_tokens_per_second": (
                fmean(generation_rates) if generation_rates else 0.0
            ),
            "runtime_identity_sha256": runtime_identity_digest,
            "runtime_session_identity_sha256": runtime_identity_digest,
            "intervention_site_evidence_sha256": stable_hash(site_evidence),
            "runtime_counts": dict(sorted(runtime_counts.items())),
            "vllm_versions": {str(runtime_identity["vllm"]): 1},
            "torch_versions": {str(runtime_identity["torch"]): 1},
            "transformers_versions": {str(runtime_identity["transformers"]): 1},
            "nvidia_drivers": {str(runtime_identity["nvidia_driver"]): 1},
            "gpu_models": {str(runtime_identity["gpu_name"]): 1},
            "operating_systems": {str(runtime_identity["os"]): 1},
            "architectures": {str(runtime_identity["architecture"]): 1},
            "cuda_capabilities": {str(runtime_identity["cuda_capability"]): 1},
            "cuda_runtimes": {str(runtime_identity["cuda_runtime"]): 1},
            "quantization_executions": {
                str(runtime_identity["quantization_execution"]): 1
            },
            "gpu_total_memory_bytes": int(runtime_identity["gpu_total_memory_bytes"]),
            "intervention_site_counts": dict(sorted(site_counts.items())),
            "policy_action_counts": dict(sorted(action_counts.items())),
            "mean_activation_delta_norm_by_site": mean_delta_by_site,
        }
    }


def _power_analysis(records: Sequence[GenerationRecord], alpha: float) -> dict[str, Any]:
    """Use only post-E1 paired P0/P2 discordance to plan confirmatory sizes."""

    cells = _method_cells(records)
    result: dict[str, Any] = {}
    targets = {
        "triviaqa": 5_000,
        "simpleqa_verified": 1_000,
        "aa_omniscience_public_600": 600,
    }
    for benchmark, target in targets.items():
        before_cell = cells[(benchmark, "P0-neutral", "M0")]
        after_cell = cells[(benchmark, "P2-calibrated-abstention", "M0")]
        if set(before_cell) != set(after_cell):
            raise DataValidationError("E1 paired power prompts use different questions")
        identifiers = tuple(sorted(before_cell))
        before = tuple(before_cell[value] for value in identifiers)
        after = tuple(after_cell[value] for value in identifiers)
        value = simulate_paired_mcnemar_power(
            identifiers,
            [record.outcome is Outcome.CORRECT for record in before],
            [record.outcome is Outcome.CORRECT for record in after],
            target_sample_sizes=(target,),
            simulations=10_000,
            alpha=alpha,
            seed=17,
        )[0]
        result[f"{benchmark}|E1_P0_to_P2"] = {
            "estimated_power": value.estimated_power,
            "target_sample_size": value.target_sample_size,
            "observed_questions": value.observed_questions,
            "baseline_only_correct_rate": value.baseline_only_correct_rate,
            "treatment_only_correct_rate": value.treatment_only_correct_rate,
            "source_phase_sha256": stable_hash(ExperimentPhase.E1.value),
        }
    return result


def _official_sections(
    records: Sequence[GenerationRecord], confidence: float
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    grouped: dict[str, list[GenerationRecord]] = {benchmark: [] for benchmark in _FACTUAL}
    for record in records:
        if record.benchmark in grouped:
            grouped[record.benchmark].append(record)
    if any(not values for values in grouped.values()):
        raise DataValidationError("E10 records lack an official factual benchmark")
    official: dict[str, Any] = {}
    grader: dict[str, Any] = {}
    bounds: dict[str, Any] = {}
    for benchmark, values in grouped.items():
        counts = Counter(value.outcome.value for value in values)
        normalized_counts = {label: counts[label] for label in ("C", "P", "I", "A", "U")}
        total = len(values)
        scorable = total - counts["U"]
        attempted = counts["C"] + counts["P"] + counts["I"]
        if scorable <= 0 or attempted <= 0:
            raise DataValidationError("official factual result has no scorable attempts")
        common: dict[str, Any] = {
            "counts": normalized_counts,
            "total": total,
            "scorable": scorable,
            "attempted": attempted,
            "accuracy": counts["C"] / scorable,
            "coverage": attempted / scorable,
            "hallucination_risk": counts["I"] / attempted,
        }
        if benchmark == "triviaqa":
            try:
                exact = [float(value.metadata["official_exact_match"]) for value in values]
                token_f1 = [float(value.metadata["official_token_f1"]) for value in values]
            except (KeyError, TypeError, ValueError) as exc:
                raise DataValidationError(
                    "TriviaQA records lack official deterministic scores"
                ) from exc
            common.update({"exact_match": fmean(exact), "token_f1": fmean(token_f1)})
        elif benchmark == "simpleqa_verified":
            if counts["P"]:
                raise DataValidationError("SimpleQA official outcomes cannot contain partials")
            scorable_values = [value for value in values if value.outcome is not Outcome.UNSCORABLE]
            if any(
                not simpleqa_hedging_evidence_is_valid(
                    value.raw_output,
                    value.metadata.get("simpleqa_hedging_evidence"),
                )
                for value in values
            ):
                raise DataValidationError("SimpleQA hedging evidence does not replay")
            attempted_accuracy = counts["C"] / attempted
            accuracy = counts["C"] / scorable
            common.update(
                {
                    "simpleqa_f1": (
                        2 * accuracy * attempted_accuracy / (accuracy + attempted_accuracy)
                        if accuracy + attempted_accuracy
                        else 0.0
                    ),
                    "accuracy_given_attempted": attempted_accuracy,
                    "attempt_rate": attempted / scorable,
                    "incorrect_attempted_rate": counts["I"] / attempted,
                    "punting_rate": counts["A"] / scorable,
                    "hedging_rate": sum(
                        bool(value.metadata["simpleqa_hedging_evidence"]["hedged"])
                        for value in scorable_values
                    )
                    / scorable,
                }
            )
        else:
            common.update(
                {
                    "omniscience_index": 100 * (counts["C"] - counts["I"]) / scorable,
                    "accuracy_given_attempted": counts["C"] / attempted,
                    "correct_rate": counts["C"] / scorable,
                    "partial_rate": counts["P"] / scorable,
                    "incorrect_rate": counts["I"] / scorable,
                    "abstention_rate": counts["A"] / scorable,
                }
            )
        official[benchmark] = common
        if benchmark != "triviaqa":
            grader_calls = 0
            failed_calls = 0
            recovered_responses = 0
            terminal_failures = 0
            for value in values:
                evidence = value.metadata.get("official_grader_evidence")
                attempt_receipts = (
                    evidence.get("attempt_receipts") if isinstance(evidence, Mapping) else None
                )
                if not isinstance(attempt_receipts, list) or not attempt_receipts:
                    raise DataValidationError("official grader record lacks attempt-level receipts")
                errors = sum(
                    1
                    for receipt in attempt_receipts
                    if isinstance(receipt, Mapping) and receipt.get("error_type") is not None
                )
                if any(not isinstance(receipt, Mapping) for receipt in attempt_receipts):
                    raise DataValidationError("official grader attempt receipt is invalid")
                grader_calls += len(attempt_receipts)
                failed_calls += errors
                failed = value.metadata.get("grader_failed")
                if not isinstance(failed, bool):
                    raise DataValidationError("official grader terminal status is missing")
                terminal_failures += int(failed)
                recovered_responses += int(errors > 0 and not failed)
            grader[benchmark] = {
                "responses": total,
                "grader_calls": grader_calls,
                "failed_calls": failed_calls,
                "recovered_responses": recovered_responses,
                "terminal_failures": terminal_failures,
                "failure_rate": failed_calls / grader_calls,
            }
            bounds[benchmark] = {
                "attempted": attempted,
                "errors": counts["I"],
                "zero_errors_observed": counts["I"] == 0,
                "confidence": confidence,
                "one_sided_upper": _exact_binomial_upper(counts["I"], attempted, confidence),
            }
    return official, grader, bounds


def derive_final_analysis_results(
    *,
    protocol: AnalysisProtocol,
    phase_records: Mapping[ExperimentPhase | str, Sequence[GenerationRecord]],
    e9_analysis_outputs: Mapping[str, Mapping[str, Any]],
    human_audit: HumanAuditResults,
    prompt_paraphrase_records: Sequence[VerifiedPromptParaphraseRecord],
    rq1_generalization_results: Mapping[str, RQ1GeneralizationResult],
    e7_interpretability: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    runtime_attestation_digest: str,
    phase_completion_digests: Mapping[str, str] | None = None,
    e3_analysis_surface: Mapping[str, Mapping[str, Any]] | None = None,
    e3_source_digest: str | None = None,
    aa_official_prompt_comparison: Mapping[str, Any] | None = None,
    aa_official_source_digest: str | None = None,
) -> DerivedFinalAnalysis:
    """Calculate the complete final result payload from replayed raw evidence."""

    normalized = {ExperimentPhase(key): tuple(values) for key, values in phase_records.items()}
    generic_required = {
        ExperimentPhase.E1,
        ExperimentPhase.E6,
        ExperimentPhase.E8,
        ExperimentPhase.E9,
        ExperimentPhase.E10,
    }
    expected_records = (
        generic_required | {ExperimentPhase.E3} if e3_analysis_surface is None else generic_required
    )
    if set(normalized) != expected_records:
        raise DataValidationError("final result derivation received the wrong phase-record sources")
    records = {
        phase: _ordered_records(normalized[phase], phase.value) for phase in expected_records
    }
    if e3_analysis_surface is None:
        if e3_source_digest is not None:
            raise DataValidationError("raw E3 records cannot receive a surface digest")
        layer_alpha_surface = _layer_alpha_surface(records[ExperimentPhase.E3])
        e3_digest = stable_hash([value.to_dict() for value in records[ExperimentPhase.E3]])
    else:
        if (
            not isinstance(e3_source_digest, str)
            or len(e3_source_digest) != 64
            or any(character not in "0123456789abcdef" for character in e3_source_digest)
        ):
            raise DataValidationError("E3 analysis surface lacks its completion digest")
        layer_alpha_surface = _validated_layer_alpha_surface(e3_analysis_surface)
        e3_digest = e3_source_digest
    primary, holm, mixed, interactions = _primary_and_e9_sections(e9_analysis_outputs)
    if (aa_official_prompt_comparison is None) is not (aa_official_source_digest is None):
        raise DataValidationError("AA official comparison and source identity must be paired")
    interactions = dict(interactions)
    if aa_official_prompt_comparison is not None:
        deltas = aa_official_prompt_comparison.get("deltas")
        official_track = aa_official_prompt_comparison.get("official")
        neutral_track = aa_official_prompt_comparison.get("neutral")
        transitions = aa_official_prompt_comparison.get("transition_counts")
        comparability = aa_official_prompt_comparison.get("leaderboard_comparability")
        if (
            aa_official_prompt_comparison.get("comparison")
            != "P-AA-official-vs-P0-neutral within M0"
            or aa_official_prompt_comparison.get("paired_question_count") != 600
            or not isinstance(deltas, Mapping)
            or not isinstance(official_track, Mapping)
            or not isinstance(neutral_track, Mapping)
            or not isinstance(official_track.get("official_metrics"), Mapping)
            or not isinstance(official_track.get("unified_metrics"), Mapping)
            or not isinstance(neutral_track.get("official_metrics"), Mapping)
            or not isinstance(neutral_track.get("unified_metrics"), Mapping)
            or not isinstance(transitions, Mapping)
            or not transitions
            or any(
                not isinstance(key, str)
                or not key
                or isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for key, value in transitions.items()
            )
            or sum(transitions.values()) != 600
            or comparability != {"official_track": True, "neutral_controlled_track": False}
            or not isinstance(aa_official_source_digest, str)
            or len(aa_official_source_digest) != 64
            or any(value not in "0123456789abcdef" for value in aa_official_source_digest)
        ):
            raise DataValidationError("AA official prompt comparison is invalid")

        def numeric(value: object, name: str) -> float:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise DataValidationError(f"AA official prompt metric is invalid: {name}")
            result = float(value)
            if not math.isfinite(result):
                raise DataValidationError(f"AA official prompt metric is invalid: {name}")
            return result

        official_metrics = official_track["official_metrics"]
        official_unified = official_track["unified_metrics"]
        neutral_metrics = neutral_track["official_metrics"]
        neutral_unified = neutral_track["unified_metrics"]
        assert isinstance(official_metrics, Mapping)
        assert isinstance(official_unified, Mapping)
        assert isinstance(neutral_metrics, Mapping)
        assert isinstance(neutral_unified, Mapping)
        omniscience_delta = numeric(deltas.get("omniscience_index"), "omniscience-index change")
        coverage_delta = numeric(deltas.get("coverage"), "coverage change")
        official_omniscience = numeric(
            official_metrics.get("omniscience_index"), "official omniscience index"
        )
        official_accuracy = numeric(official_metrics.get("accuracy"), "official accuracy")
        official_coverage = numeric(official_unified.get("coverage"), "official coverage")
        neutral_omniscience = numeric(
            neutral_metrics.get("omniscience_index"), "neutral omniscience index"
        )
        neutral_accuracy = numeric(neutral_metrics.get("accuracy"), "neutral accuracy")
        neutral_coverage = numeric(neutral_unified.get("coverage"), "neutral coverage")
        risk_delta = deltas.get("hallucination_risk")
        risk_defined = risk_delta is not None
        interactions["AA|M0|P-AA-official-vs-P0-neutral"] = {
            "estimate": omniscience_delta,
            "paired_questions": 600,
            "omniscience_index_change": omniscience_delta,
            "coverage_change": coverage_delta,
            "hallucination_risk_change_defined": risk_defined,
            "hallucination_risk_change": (
                numeric(risk_delta, "hallucination-risk change") if risk_defined else 0.0
            ),
            "official_prompt": {
                "omniscience_index": official_omniscience,
                "accuracy": official_accuracy,
                "coverage": official_coverage,
            },
            "neutral_prompt": {
                "omniscience_index": neutral_omniscience,
                "accuracy": neutral_accuracy,
                "coverage": neutral_coverage,
            },
            "transition_counts": {
                str(key): int(value) for key, value in sorted(transitions.items())
            },
            "official_track_leaderboard_comparable": True,
            "neutral_track_leaderboard_comparable": False,
            "comparison_digest": stable_hash(dict(aa_official_prompt_comparison)),
            "source_digest": aa_official_source_digest,
        }
    official, grader, bounds = _official_sections(records[ExperimentPhase.E10], protocol.confidence)
    factual_audit = human_audit.summary.get("factual_reporting_payload")
    if not isinstance(factual_audit, Mapping):
        raise DataValidationError("finalized human audit lacks factual reporting evidence")
    results = FinalAnalysisResults(
        primary_contrasts=primary,
        holm_adjusted_tests=holm,
        mixed_effects=mixed,
        noninferiority=_noninferiority(records[ExperimentPhase.E8], protocol),
        composite_side_effects=_composite_side_effects(
            records[ExperimentPhase.E8], records[ExperimentPhase.E10], protocol
        ),
        risk_coverage=_risk_coverage(records[ExperimentPhase.E9]),
        transition_decomposition=_transition_decomposition(
            records[ExperimentPhase.E9], records[ExperimentPhase.E10]
        ),
        likelihood_changes=_likelihood_changes(records[ExperimentPhase.E6]),
        layer_alpha_surface=layer_alpha_surface,
        matched_coverage=_matched_coverage(records[ExperimentPhase.E9]),
        factuality_side_effect_pareto=_pareto(records[ExperimentPhase.E8], protocol),
        prompt_interactions=interactions,
        prompt_paraphrase=_prompt_paraphrase(prompt_paraphrase_records),
        rq1_generalization=_rq1_generalization_section(rq1_generalization_results),
        e7_interpretability=dict(e7_interpretability),
        language_confusion=_language_confusion(
            records[ExperimentPhase.E10],
            records[ExperimentPhase.E8],
            human_audit,
        ),
        runtime_replication=_runtime_replication(records[ExperimentPhase.E10], runtime_identity),
        power_analysis=_power_analysis(records[ExperimentPhase.E1], protocol.alpha),
        official_metrics=official,
        grader_failure_rates=grader,
        zero_error_bounds=bounds,
        human_audit=dict(factual_audit),
    )
    results.validate_against_protocol(protocol)
    results.validate_against_records(records[ExperimentPhase.E10])
    source_digests = {
        phase.value: stable_hash([value.to_dict() for value in records[phase]])
        for phase in sorted(generic_required, key=lambda value: value.ordinal)
    }
    source_digests[ExperimentPhase.E3.value] = e3_digest
    if aa_official_source_digest is not None:
        source_digests["AA-official-auxiliary"] = aa_official_source_digest
    completions = (
        {str(key): str(value) for key, value in phase_completion_digests.items()}
        if phase_completion_digests is not None
        else {}
    )
    all_required = generic_required | {ExperimentPhase.E3, ExperimentPhase.E7}
    if completions and set(completions) != {phase.value for phase in all_required}:
        raise DataValidationError("final derivation completion identities are incomplete")
    return DerivedFinalAnalysis(
        results=results,
        source_record_digests=source_digests,
        e9_analysis_digest=stable_hash(e9_analysis_outputs),
        human_audit_manifest_digest=human_audit.manifest_digest,
        robustness_record_digest=stable_hash(
            {
                "prompt_paraphrase": [
                    {
                        "task_id": value.task_id,
                        "benchmark": value.benchmark,
                        "base_prompt_id": value.base_prompt_id,
                        "paraphrase_prompt_id": value.paraphrase_prompt_id,
                        "method": value.method,
                        "record": value.record.to_dict(),
                    }
                    for value in prompt_paraphrase_records
                ],
                "rq1_generalization": {
                    key: value.to_dict()
                    for key, value in sorted(rq1_generalization_results.items())
                },
            }
        ),
        runtime_attestation_digest=runtime_attestation_digest,
        phase_completion_digests=completions,
    )


def derive_final_analysis_from_artifacts(
    *,
    protocol: AnalysisProtocol,
    study: StudyProtocol,
    phase_run_directories: Mapping[ExperimentPhase | str, str | Path],
    robustness_result_directory: str | Path,
    human_audit_queue_directory: str | Path,
    human_audit_results_directory: str | Path,
    human_audit_blinding_key: bytes,
    aa_official_directory: str | Path,
    expected_aa_official_manifest_digest: str,
) -> DerivedFinalAnalysis:
    """Verify every upstream artifact and derive the publication payload.

    ``open_phase_prerequisite`` performs the authoritative study-protocol
    validation before any record is admitted to a calculation, including the
    custom seven-stage E3 terminal artifact.
    """

    from mfh.analysis.human_audit import verify_human_audit_results
    from mfh.experiments.aa_official_track import load_aa_official_analysis
    from mfh.experiments.e3_phase import load_e3_analysis_surface
    from mfh.experiments.e6_likelihood import _load_e6_runtime_attestation
    from mfh.experiments.e9_analysis import _derive_analysis, load_e9_matching_basis
    from mfh.experiments.robustness_results import (
        load_verified_prompt_paraphrase_records,
        load_verified_rq1_generalization_results,
    )
    from mfh.experiments.runner import open_phase_prerequisite

    normalized = {ExperimentPhase(key): value for key, value in phase_run_directories.items()}
    required = {
        ExperimentPhase.E1,
        ExperimentPhase.E3,
        ExperimentPhase.E6,
        ExperimentPhase.E7,
        ExperimentPhase.E8,
        ExperimentPhase.E9,
        ExperimentPhase.E10,
    }
    if set(normalized) != required:
        raise DataValidationError(
            "artifact-derived final analysis requires E1, E3, E6, E7, E8, E9, and E10 runs"
        )
    ledgers: dict[ExperimentPhase, Any] = {}
    completions: dict[str, str] = {}
    phase_records: dict[ExperimentPhase, tuple[GenerationRecord, ...]] = {}
    record_phases = {
        ExperimentPhase.E1,
        ExperimentPhase.E6,
        ExperimentPhase.E8,
        ExperimentPhase.E9,
        ExperimentPhase.E10,
    }
    for phase in sorted(required, key=lambda value: value.ordinal):
        ledger = open_phase_prerequisite(
            normalized[phase],
            phase=phase,
            study=study,
        )
        completion = ledger.verify_complete()
        if completion.phase is not phase:
            raise DataValidationError("final-analysis source run is cross-phase")
        ledgers[phase] = ledger
        completions[phase.value] = completion.completion_digest
        if phase in record_phases:
            phase_records[phase] = tuple(ledger.records())
    e9_prerequisites = ledgers[ExperimentPhase.E9].contract.prerequisite_digests
    e10_prerequisites = ledgers[ExperimentPhase.E10].contract.prerequisite_digests
    e1_completion_digest = completions[ExperimentPhase.E1.value]
    if (
        e9_prerequisites.get(ExperimentPhase.E1.value) != e1_completion_digest
        or e10_prerequisites.get(ExperimentPhase.E1.value) != e1_completion_digest
    ):
        raise DataValidationError("E9/E10 do not share one frozen E1 completion")
    aa_official = load_aa_official_analysis(
        aa_official_directory,
        expected_manifest_digest=expected_aa_official_manifest_digest,
        expected_e1_completion_digest=e1_completion_digest,
    )
    e3_completion_digest = completions[ExperimentPhase.E3.value]
    if (
        e9_prerequisites.get(ExperimentPhase.E3.value) != e3_completion_digest
        or e10_prerequisites.get(ExperimentPhase.E3.value) != e3_completion_digest
    ):
        raise DataValidationError("E9/E10 do not share one frozen E3 completion")
    e3_surface, observed_e3_digest = load_e3_analysis_surface(
        normalized[ExperimentPhase.E3],
        expected_completion_digest=e3_completion_digest,
        study=study,
    )
    if observed_e3_digest != e3_completion_digest:
        raise DataValidationError("E3 analysis source differs from confirmatory lineage")
    for phase in (ExperimentPhase.E6, ExperimentPhase.E7, ExperimentPhase.E8):
        digest = completions[phase.value]
        if (
            e9_prerequisites.get(phase.value) != digest
            or e10_prerequisites.get(phase.value) != digest
        ):
            raise DataValidationError(
                f"{phase.value} analysis source differs from confirmatory lineage"
            )
    if e10_prerequisites.get(ExperimentPhase.E9.value) != completions[ExperimentPhase.E9.value]:
        raise DataValidationError("E9 analysis source differs from E10 lineage")
    e9 = ledgers[ExperimentPhase.E9]
    e9_outputs = _derive_analysis(
        phase_records[ExperimentPhase.E9],
        protocol,
        e9.contract.prerequisite_digests,
        load_e9_matching_basis(e9),
    )
    audit = verify_human_audit_results(
        human_audit_results_directory,
        queue_directory=human_audit_queue_directory,
        expected_protocol=protocol,
        study=study,
        phase_run_directories={
            ExperimentPhase.E9: normalized[ExperimentPhase.E9],
            ExperimentPhase.E10: normalized[ExperimentPhase.E10],
        },
        blinding_key=human_audit_blinding_key,
    )
    prompt_records = load_verified_prompt_paraphrase_records(robustness_result_directory)
    rq1_results = load_verified_rq1_generalization_results(robustness_result_directory)
    e7_interpretability = _e7_interpretability_section(normalized[ExperimentPhase.E7])
    runtime_attestation = _load_e6_runtime_attestation(
        Path(normalized[ExperimentPhase.E10]) / "inputs" / "grader" / "runtime-attestation.json"
    )
    runtime_identity = runtime_attestation["runtime_identity"]
    if not isinstance(runtime_identity, Mapping):
        raise DataValidationError("E10 runtime attestation lacks its identity")
    return derive_final_analysis_results(
        protocol=protocol,
        phase_records={phase.value: values for phase, values in phase_records.items()},
        e9_analysis_outputs=e9_outputs,
        human_audit=audit,
        prompt_paraphrase_records=prompt_records,
        rq1_generalization_results=rq1_results,
        e7_interpretability=e7_interpretability,
        runtime_identity=runtime_identity,
        runtime_attestation_digest=str(runtime_attestation["runtime_attestation_digest"]),
        phase_completion_digests=completions,
        e3_analysis_surface=e3_surface,
        e3_source_digest=e3_completion_digest,
        aa_official_prompt_comparison=aa_official["official_vs_neutral"],
        aa_official_source_digest=stable_hash(
            {
                "manifest_digest": aa_official["manifest_digest"],
                "record_set_digest": aa_official["record_set_digest"],
                "e1_completion_digest": aa_official["e1_completion_digest"],
            }
        ),
    )

"""Atomic, content-addressed final analysis bundles backed by completed runs."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from mfh.analysis.human_audit import HumanAuditResults, verify_human_audit_results
from mfh.analysis.protocol import AnalysisProtocol
from mfh.artifact_namespace import validate_active_study_artifact_paths
from mfh.contracts import GenerationRecord, Outcome
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.evaluation.language import SUPPORTED_LANGUAGES, language_response_evidence
from mfh.evaluation.simpleqa import simpleqa_hedging_evidence_is_valid
from mfh.experiments.protocol import ExperimentPhase, StudyProtocol
from mfh.experiments.runner import PhaseRunLedger, open_phase_prerequisite
from mfh.provenance import canonical_json, sha256_file, sha256_path, stable_hash

if TYPE_CHECKING:
    from mfh.analysis.derivation import DerivedFinalAnalysis

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_NAME = re.compile(r"^[a-z0-9_]+$")
_PRIMARY_CONTRASTS = {"RQ1", "RQ2", "RQ3", "RQ4"}
_FACTUAL_BENCHMARKS = {
    "triviaqa",
    "simpleqa_verified",
    "aa_omniscience_public_600",
}
_OFFICIAL_GRADER_BENCHMARKS = {
    "simpleqa_verified",
    "aa_omniscience_public_600",
}
_FINAL_ANALYSIS_PHASES = (
    ExperimentPhase.E1,
    ExperimentPhase.E3,
    ExperimentPhase.E6,
    ExperimentPhase.E7,
    ExperimentPhase.E8,
    ExperimentPhase.E9,
    ExperimentPhase.E10,
)
_MODEL_NAMES_BY_REPOSITORY = {
    "nvidia/Qwen3.6-27B-NVFP4": "qwen3.6-27b-nvfp4",
}
_NONINFERIORITY_METRICS = {
    "ifeval_pass_rate",
    "mmlu_pro_accuracy",
    "xstest_benign_non_refusal_rate",
    "harmful_prompt_refusal_rate",
    "harmful_prompt_compliance_score",
    "requested_language_consistency",
    "perplexity_relative_change",
    "latency_relative_change",
}
_RESULT_FIELDS = (
    "primary_contrasts",
    "holm_adjusted_tests",
    "mixed_effects",
    "noninferiority",
    "composite_side_effects",
    "risk_coverage",
    "transition_decomposition",
    "likelihood_changes",
    "layer_alpha_surface",
    "matched_coverage",
    "factuality_side_effect_pareto",
    "prompt_interactions",
    "prompt_paraphrase",
    "rq1_generalization",
    "e7_interpretability",
    "language_confusion",
    "runtime_replication",
    "power_analysis",
    "official_metrics",
    "grader_failure_rates",
    "zero_error_bounds",
    "human_audit",
)
_REPORT_RESULT_DEPENDENCIES: Mapping[str, tuple[str, ...]] = {
    "risk_coverage_curves": ("risk_coverage",),
    "outcome_transition_diagrams": ("transition_decomposition",),
    "gold_vs_abstention_likelihood_changes": ("likelihood_changes",),
    "layer_alpha_heatmaps": ("layer_alpha_surface",),
    "static_vs_adaptive_matched_coverage": (
        "matched_coverage",
        "rq1_generalization",
    ),
    "dense_sparse_disentangled_pareto": (
        "factuality_side_effect_pareto",
        "e7_interpretability",
    ),
    "prompt_method_interaction_heatmaps": ("prompt_interactions", "mixed_effects"),
    "prompt_paraphrase_robustness": ("prompt_paraphrase",),
    "safety_utility_noninferiority": (
        "noninferiority",
        "composite_side_effects",
    ),
    "language_switching_confusion_matrices": ("language_confusion",),
    "local_vllm_runtime_validation": ("runtime_replication",),
    "zero_error_confidence_bounds": ("zero_error_bounds",),
    "adjudicated_final_labels": ("human_audit",),
    "automated_human_confusion_matrix": ("human_audit",),
}
_OUTCOME_LABELS = {"C", "P", "I", "A", "U"}
_TRANSITION_OUTCOMES = ("C", "P", "I", "A")
_TRANSITION_COMPARISONS = frozenset(
    {
        f"{benchmark}|{prompt}|M0_to_{method}"
        for benchmark in _FACTUAL_BENCHMARKS
        for prompt in ("P0-neutral", "P2-calibrated-abstention")
        for method in ("M1", "M2", "M3", "M4", "M5")
    }
    | {f"{benchmark}|P0-neutral|M0_to_M6" for benchmark in _FACTUAL_BENCHMARKS}
)
_AUDIT_QUEUES = {
    "automated_grader_disagreements",
    "partial_aa_responses",
    "language_switch_detections",
    "suspected_safety_regressions",
    "random_abstentions",
    "random_incorrect_attempts",
    "minimum_stratified_sample",
}
_AUDIT_COLUMNS = (
    "audit_id",
    "question_id",
    "condition_id",
    "response_sha256",
    "benchmark",
    "model",
    "method",
    "prompt",
    "automated_label",
    "annotator_1_label",
    "annotator_2_label",
    "adjudicated_label",
    "queue",
)


def _normalized_nonempty_mapping(value: Mapping[str, Any], context: str) -> Mapping[str, Any]:
    result = {str(key).strip(): item for key, item in value.items()}
    if not result or any(not key for key in result):
        raise DataValidationError(f"{context} must be a non-empty named mapping")
    _validate_quantitative_tree(result, context)
    try:
        normalized = json.loads(canonical_json(result))
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{context} must contain finite JSON values: {exc}") from exc
    return MappingProxyType(normalized)


def _validate_quantitative_tree(value: Any, context: str) -> None:
    if isinstance(value, str) and _SHA256.fullmatch(value):
        return
    if isinstance(value, bool):
        return
    if isinstance(value, int | float):
        if not isinstance(value, bool) and math.isfinite(float(value)):
            return
        raise DataValidationError(f"{context} contains a non-finite numeric result")
    if isinstance(value, Mapping):
        if not value:
            raise DataValidationError(f"{context} contains an empty result mapping")
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise DataValidationError(f"{context} contains an invalid result key")
            _validate_quantitative_tree(item, f"{context}.{key}")
        return
    if isinstance(value, list | tuple):
        if not value:
            raise DataValidationError(f"{context} contains an empty result sequence")
        for index, item in enumerate(value):
            _validate_quantitative_tree(item, f"{context}[{index}]")
        return
    raise DataValidationError(
        f"{context} contains a non-quantitative result value of type {type(value).__name__}"
    )


def _entry(
    section: Mapping[str, Any], key: str, required: set[str], context: str
) -> Mapping[str, Any]:
    value = section[key]
    if not isinstance(value, Mapping) or not required <= set(value):
        raise DataValidationError(f"{context}.{key} lacks fields {sorted(required)}")
    return value


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DataValidationError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise DataValidationError(f"{context} must be finite")
    return result


def _probability(value: Any, context: str) -> float:
    result = _number(value, context)
    if not 0 <= result <= 1:
        raise DataValidationError(f"{context} must be in [0, 1]")
    return result


def _integer(value: Any, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DataValidationError(f"{context} must be an integer of at least {minimum}")
    return int(value)


def _exact_binomial_upper(errors: int, attempted: int, confidence: float) -> float:
    """One-sided Clopper-Pearson upper bound, including the zero-error special case."""

    if errors < 0 or attempted <= 0 or errors > attempted or not 0 < confidence < 1:
        raise DataValidationError("exact binomial bound inputs are invalid")
    if errors == 0:
        return float(1 - (1 - confidence) ** (1 / attempted))
    if errors == attempted:
        return 1.0

    def cumulative(probability: float) -> float:
        if probability <= 0:
            return 1.0
        if probability >= 1:
            return 0.0
        term = (1 - probability) ** attempted
        total = term
        odds = probability / (1 - probability)
        for value in range(errors):
            term *= (attempted - value) / (value + 1) * odds
            total += term
        return total

    target = 1 - confidence
    lower = errors / attempted
    upper = 1.0
    for _ in range(100):
        midpoint = (lower + upper) / 2
        if cumulative(midpoint) > target:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2


@dataclass(frozen=True, slots=True)
class FinalAnalysisResults:
    """Required statistical outputs; arbitrary result-shaped JSON is not accepted."""

    primary_contrasts: Mapping[str, Any]
    holm_adjusted_tests: Mapping[str, Any]
    mixed_effects: Mapping[str, Any]
    noninferiority: Mapping[str, Any]
    composite_side_effects: Mapping[str, Any]
    risk_coverage: Mapping[str, Any]
    transition_decomposition: Mapping[str, Any]
    likelihood_changes: Mapping[str, Any]
    layer_alpha_surface: Mapping[str, Any]
    matched_coverage: Mapping[str, Any]
    factuality_side_effect_pareto: Mapping[str, Any]
    prompt_interactions: Mapping[str, Any]
    prompt_paraphrase: Mapping[str, Any]
    rq1_generalization: Mapping[str, Any]
    e7_interpretability: Mapping[str, Any]
    language_confusion: Mapping[str, Any]
    runtime_replication: Mapping[str, Any]
    power_analysis: Mapping[str, Any]
    official_metrics: Mapping[str, Any]
    grader_failure_rates: Mapping[str, Any]
    zero_error_bounds: Mapping[str, Any]
    human_audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in _RESULT_FIELDS:
            normalized = _normalized_nonempty_mapping(getattr(self, name), name)
            object.__setattr__(self, name, normalized)
        if set(self.primary_contrasts) != _PRIMARY_CONTRASTS:
            raise DataValidationError("final results must contain the four primary RQ contrasts")
        if set(self.noninferiority) != _NONINFERIORITY_METRICS:
            raise DataValidationError("final results must contain every non-inferiority metric")
        if set(self.official_metrics) != _FACTUAL_BENCHMARKS:
            raise DataValidationError("final results must contain every official factual metric")
        if set(self.grader_failure_rates) != _OFFICIAL_GRADER_BENCHMARKS:
            raise DataValidationError(
                "final results must report both official-grader failure rates"
            )
        if set(self.zero_error_bounds) != _OFFICIAL_GRADER_BENCHMARKS:
            raise DataValidationError("zero-error bounds must cover SimpleQA and AA")
        if set(self.human_audit) != {
            "agreement_metrics",
            "adjudication_summary",
            "automated_human_confusion_matrix",
            "record_binding_digest",
        }:
            raise DataValidationError("final human-audit results are incomplete")
        agreement = self.human_audit["agreement_metrics"]
        if not isinstance(agreement, Mapping) or set(agreement) != {
            "cohen_kappa",
            "krippendorff_alpha",
        }:
            raise DataValidationError("human audit must report kappa and alpha")
        if not _SHA256.fullmatch(str(self.human_audit["record_binding_digest"])):
            raise DataValidationError("human audit requires a record-binding digest")
        self._validate_statistical_shapes()

    def _validate_statistical_shapes(self) -> None:
        for contrast in sorted(_PRIMARY_CONTRASTS):
            result = _entry(
                self.primary_contrasts,
                contrast,
                {"estimate", "confidence_interval", "p_value"},
                "primary_contrasts",
            )
            _number(result["estimate"], f"primary_contrasts.{contrast}.estimate")
            _probability(result["p_value"], f"primary_contrasts.{contrast}.p_value")
            interval = result["confidence_interval"]
            if not isinstance(interval, list | tuple) or len(interval) != 2:
                raise DataValidationError(
                    f"primary_contrasts.{contrast}.confidence_interval must have two bounds"
                )
            lower = _number(interval[0], f"primary_contrasts.{contrast}.confidence_interval[0]")
            upper = _number(interval[1], f"primary_contrasts.{contrast}.confidence_interval[1]")
            if lower > upper:
                raise DataValidationError(
                    f"primary_contrasts.{contrast} confidence bounds are reversed"
                )
        for family in self.holm_adjusted_tests:
            result = _entry(
                self.holm_adjusted_tests,
                family,
                {"raw_p_value", "adjusted_p_value", "rejected"},
                "holm_adjusted_tests",
            )
            raw_p_value = _probability(
                result["raw_p_value"], f"holm_adjusted_tests.{family}.raw_p_value"
            )
            adjusted_p_value = _probability(
                result["adjusted_p_value"],
                f"holm_adjusted_tests.{family}.adjusted_p_value",
            )
            if adjusted_p_value < raw_p_value:
                raise DataValidationError(
                    f"holm_adjusted_tests.{family} adjusted p-value is below the raw value"
                )
            if not isinstance(result["rejected"], bool):
                raise DataValidationError(f"holm_adjusted_tests.{family}.rejected must be boolean")
        for effect in self.mixed_effects:
            result = _entry(
                self.mixed_effects,
                effect,
                {"estimate", "p_value"},
                "mixed_effects",
            )
            _number(result["estimate"], f"mixed_effects.{effect}.estimate")
            _probability(result["p_value"], f"mixed_effects.{effect}.p_value")
        for metric in sorted(_NONINFERIORITY_METRICS):
            result = _entry(
                self.noninferiority,
                metric,
                {
                    "estimate",
                    "margin",
                    "higher_is_better",
                    "one_sided_lower",
                    "p_value",
                    "passed",
                },
                "noninferiority",
            )
            _number(result["estimate"], f"noninferiority.{metric}.estimate")
            if _number(result["margin"], f"noninferiority.{metric}.margin") <= 0:
                raise DataValidationError(f"noninferiority.{metric}.margin must be positive")
            if not isinstance(result["higher_is_better"], bool):
                raise DataValidationError(
                    f"noninferiority.{metric}.higher_is_better must be boolean"
                )
            _number(result["one_sided_lower"], f"noninferiority.{metric}.one_sided_lower")
            _probability(result["p_value"], f"noninferiority.{metric}.p_value")
            if not isinstance(result["passed"], bool):
                raise DataValidationError(f"noninferiority.{metric}.passed must be boolean")
            comparisons = result.get("comparisons")
            if not isinstance(comparisons, Mapping) or not comparisons:
                raise DataValidationError(
                    f"noninferiority.{metric} requires its complete comparison family"
                )
            for comparison_key, comparison_value in comparisons.items():
                if not isinstance(comparison_value, Mapping) or not {
                    "raw_p_value",
                    "adjusted_p_value",
                    "rejected",
                    "non_inferior",
                    "passed",
                } <= set(comparison_value):
                    raise DataValidationError(
                        f"noninferiority.{metric}.{comparison_key} is incomplete"
                    )
                raw = _probability(
                    comparison_value["raw_p_value"],
                    f"noninferiority.{metric}.{comparison_key}.raw_p_value",
                )
                adjusted = _probability(
                    comparison_value["adjusted_p_value"],
                    f"noninferiority.{metric}.{comparison_key}.adjusted_p_value",
                )
                if adjusted < raw or any(
                    not isinstance(comparison_value[name], bool)
                    for name in ("rejected", "non_inferior", "passed")
                ):
                    raise DataValidationError(
                        f"noninferiority.{metric}.{comparison_key} correction is invalid"
                    )
                if comparison_value["passed"] is not (
                    comparison_value["rejected"] and comparison_value["non_inferior"]
                ):
                    raise DataValidationError(
                        f"noninferiority.{metric}.{comparison_key} decision is invalid"
                    )
            if result["passed"] is not all(bool(value["passed"]) for value in comparisons.values()):
                raise DataValidationError(f"noninferiority.{metric} aggregate decision is invalid")
        if set(self.composite_side_effects) != {
            "selected_prompt_sha256",
            "metrics",
            "safety_transitions",
            "all_preregistered_noninferiority_tests_passed",
        } or not _SHA256.fullmatch(str(self.composite_side_effects["selected_prompt_sha256"])):
            raise DataValidationError("composite M6 side-effect evidence is incomplete")
        composite_metrics = self.composite_side_effects["metrics"]
        if not isinstance(composite_metrics, Mapping) or set(composite_metrics) != set(
            _NONINFERIORITY_METRICS
        ):
            raise DataValidationError("composite M6 metrics do not cover the side-effect suite")
        for metric, value in composite_metrics.items():
            if not isinstance(value, Mapping) or not {
                "baseline_mean",
                "composite_mean",
                "estimate",
                "margin",
                "higher_is_better",
                "one_sided_lower",
                "raw_p_value",
                "adjusted_p_value",
                "rejected",
                "non_inferior",
                "passed",
                "questions",
            } <= set(value):
                raise DataValidationError(f"composite side-effect metric {metric} is incomplete")
            for name in ("baseline_mean", "composite_mean", "estimate", "one_sided_lower"):
                _number(value[name], f"composite_side_effects.metrics.{metric}.{name}")
            if _number(value["margin"], f"composite_side_effects.metrics.{metric}.margin") <= 0:
                raise DataValidationError("composite side-effect margin must be positive")
            raw = _probability(
                value["raw_p_value"],
                f"composite_side_effects.metrics.{metric}.raw_p_value",
            )
            adjusted = _probability(
                value["adjusted_p_value"],
                f"composite_side_effects.metrics.{metric}.adjusted_p_value",
            )
            if (
                adjusted < raw
                or any(
                    not isinstance(value[name], bool)
                    for name in ("higher_is_better", "rejected", "non_inferior", "passed")
                )
                or value["passed"] is not (value["rejected"] and value["non_inferior"])
            ):
                raise DataValidationError(f"composite side-effect decision {metric} is invalid")
            _integer(
                value["questions"],
                f"composite_side_effects.metrics.{metric}.questions",
                minimum=1,
            )
        aggregate = self.composite_side_effects["all_preregistered_noninferiority_tests_passed"]
        if not isinstance(aggregate, bool) or aggregate is not all(
            bool(value["passed"]) for value in composite_metrics.values()
        ):
            raise DataValidationError("composite side-effect aggregate decision is invalid")
        safety_transitions = self.composite_side_effects["safety_transitions"]
        if not isinstance(safety_transitions, Mapping) or set(safety_transitions) != {
            "xstest",
            "strongreject_or_harmbench",
        }:
            raise DataValidationError("composite safety transitions are incomplete")
        for benchmark, labels in {
            "xstest": ("ANSWER", "REFUSE"),
            "strongreject_or_harmbench": ("REFUSE", "COMPLY", "OTHER"),
        }.items():
            transition = safety_transitions[benchmark]
            if not isinstance(transition, Mapping) or set(transition) != {
                "paired_questions",
                "transition_counts",
            }:
                raise DataValidationError(f"composite {benchmark} transition is incomplete")
            paired = _integer(
                transition["paired_questions"],
                f"composite_side_effects.safety_transitions.{benchmark}.paired_questions",
                minimum=1,
            )
            counts = transition["transition_counts"]
            expected_keys = {f"{left}:{right}" for left in labels for right in labels}
            if (
                not isinstance(counts, Mapping)
                or set(counts) != expected_keys
                or sum(
                    _integer(
                        count,
                        "composite_side_effects.safety_transitions."
                        f"{benchmark}.transition_counts.{key}",
                    )
                    for key, count in counts.items()
                )
                != paired
            ):
                raise DataValidationError(f"composite {benchmark} transition counts are invalid")
        for method in self.risk_coverage:
            result = _entry(
                self.risk_coverage,
                method,
                {
                    "aurc",
                    "coverage_limit",
                    "full_observed_aurc",
                    "maximum_coverage",
                    "point_count",
                    "target_risks",
                    "points",
                },
                "risk_coverage",
            )
            if _number(result["aurc"], f"risk_coverage.{method}.aurc") < 0:
                raise DataValidationError(f"risk_coverage.{method}.aurc cannot be negative")
            _probability(result["coverage_limit"], f"risk_coverage.{method}.coverage_limit")
            _probability(result["maximum_coverage"], f"risk_coverage.{method}.maximum_coverage")
            if (
                _number(
                    result["full_observed_aurc"],
                    f"risk_coverage.{method}.full_observed_aurc",
                )
                < 0
            ):
                raise DataValidationError("full observed AURC cannot be negative")
            point_count = _integer(
                result["point_count"], f"risk_coverage.{method}.point_count", minimum=2
            )
            points = result["points"]
            targets = result["target_risks"]
            if not isinstance(points, Mapping) or len(points) != point_count:
                raise DataValidationError(f"risk_coverage.{method} points are incomplete")
            if not isinstance(targets, Mapping) or set(targets) != {
                "coverage_25",
                "coverage_50",
                "coverage_75",
                "coverage_90",
            }:
                raise DataValidationError(f"risk_coverage.{method} target risks are incomplete")
            for point_key, point in points.items():
                if not isinstance(point, Mapping) or not {
                    "threshold",
                    "coverage",
                    "risk",
                    "attempted",
                    "incorrect",
                } <= set(point):
                    raise DataValidationError(f"risk_coverage.{method}.{point_key} is invalid")
                _probability(point["coverage"], f"risk_coverage.{method}.{point_key}.coverage")
                _probability(point["risk"], f"risk_coverage.{method}.{point_key}.risk")
                _number(point["threshold"], f"risk_coverage.{method}.{point_key}.threshold")
                attempted = _integer(
                    point["attempted"],
                    f"risk_coverage.{method}.{point_key}.attempted",
                    minimum=1,
                )
                incorrect = _integer(
                    point["incorrect"], f"risk_coverage.{method}.{point_key}.incorrect"
                )
                if incorrect > attempted:
                    raise DataValidationError("risk-coverage incorrect count exceeds attempted")
            for target_key, target in targets.items():
                if not isinstance(target, Mapping) or not {
                    "target_coverage",
                    "reached",
                    "achieved_coverage",
                    "risk",
                    "attempted",
                    "incorrect",
                    "threshold",
                } <= set(target):
                    raise DataValidationError(f"risk_coverage.{method}.{target_key} is invalid")
                if not isinstance(target["reached"], bool):
                    raise DataValidationError("risk target reach flag must be boolean")
                _probability(
                    target["target_coverage"],
                    f"risk_coverage.{method}.{target_key}.target",
                )
                achieved = _probability(
                    target["achieved_coverage"],
                    f"risk_coverage.{method}.{target_key}.achieved",
                )
                if target["reached"] and achieved < float(target["target_coverage"]):
                    raise DataValidationError("reached risk target is below target coverage")
        if set(self.transition_decomposition) != set(_TRANSITION_COMPARISONS):
            raise DataValidationError(
                "transition decomposition does not cover every preregistered M0 comparison"
            )
        for comparison in sorted(self.transition_decomposition):
            result = _entry(
                self.transition_decomposition,
                comparison,
                {
                    "paired_questions",
                    "unscorable_pairs_excluded",
                    "knowledge_recovery",
                    "abstention_substitution",
                    "strict_overrefusal",
                    "regression",
                    "transition_counts",
                },
                "transition_decomposition",
            )
            paired = _integer(
                result["paired_questions"],
                f"transition_decomposition.{comparison}.paired_questions",
                minimum=1,
            )
            _integer(
                result["unscorable_pairs_excluded"],
                f"transition_decomposition.{comparison}.unscorable_pairs_excluded",
            )
            _probability(
                result["knowledge_recovery"],
                f"transition_decomposition.{comparison}.knowledge_recovery",
            )
            _probability(
                result["abstention_substitution"],
                f"transition_decomposition.{comparison}.abstention_substitution",
            )
            _probability(
                result["strict_overrefusal"],
                f"transition_decomposition.{comparison}.strict_overrefusal",
            )
            _probability(
                result["regression"],
                f"transition_decomposition.{comparison}.regression",
            )
            counts = result["transition_counts"]
            expected_cells = {
                f"{left}:{right}" for left in _TRANSITION_OUTCOMES for right in _TRANSITION_OUTCOMES
            }
            if not isinstance(counts, Mapping) or set(counts) != expected_cells:
                raise DataValidationError(
                    f"transition_decomposition.{comparison}.transition_counts is incomplete"
                )
            total = sum(
                _integer(
                    counts[cell],
                    f"transition_decomposition.{comparison}.transition_counts.{cell}",
                )
                for cell in sorted(expected_cells)
            )
            if total != paired:
                raise DataValidationError(
                    f"transition_decomposition.{comparison} count total differs from paired rows"
                )
        for contrast in self.power_analysis:
            result = _entry(
                self.power_analysis,
                contrast,
                {"estimated_power", "target_sample_size"},
                "power_analysis",
            )
            _probability(result["estimated_power"], f"power_analysis.{contrast}.estimated_power")
            _integer(
                result["target_sample_size"],
                f"power_analysis.{contrast}.target_sample_size",
                minimum=1,
            )
        for benchmark in sorted(_FACTUAL_BENCHMARKS):
            benchmark_specific = {
                "triviaqa": {"exact_match", "token_f1"},
                "simpleqa_verified": {
                    "simpleqa_f1",
                    "accuracy_given_attempted",
                    "attempt_rate",
                    "incorrect_attempted_rate",
                    "punting_rate",
                    "hedging_rate",
                },
                "aa_omniscience_public_600": {
                    "omniscience_index",
                    "accuracy_given_attempted",
                    "correct_rate",
                    "partial_rate",
                    "incorrect_rate",
                    "abstention_rate",
                },
            }[benchmark]
            result = _entry(
                self.official_metrics,
                benchmark,
                {
                    "counts",
                    "total",
                    "scorable",
                    "attempted",
                    "accuracy",
                    "coverage",
                    "hallucination_risk",
                }
                | benchmark_specific,
                "official_metrics",
            )
            counts_value = result["counts"]
            if not isinstance(counts_value, Mapping) or set(counts_value) != _OUTCOME_LABELS:
                raise DataValidationError(f"official_metrics.{benchmark}.counts are incomplete")
            counts = {
                label: _integer(counts_value[label], f"official_metrics.{benchmark}.counts.{label}")
                for label in _OUTCOME_LABELS
            }
            total = _integer(result["total"], f"official_metrics.{benchmark}.total", minimum=1)
            scorable = _integer(
                result["scorable"], f"official_metrics.{benchmark}.scorable", minimum=1
            )
            attempted = _integer(
                result["attempted"], f"official_metrics.{benchmark}.attempted", minimum=1
            )
            if (
                total != sum(counts.values())
                or scorable != total - counts["U"]
                or attempted != counts["C"] + counts["P"] + counts["I"]
            ):
                raise DataValidationError(
                    f"official_metrics.{benchmark} counts and denominators disagree"
                )
            for metric in ("accuracy", "coverage", "hallucination_risk"):
                _probability(result[metric], f"official_metrics.{benchmark}.{metric}")
            expected_accuracy = counts["C"] / scorable
            expected_coverage = attempted / scorable
            expected_risk = counts["I"] / attempted
            for name, expected_value in {
                "accuracy": expected_accuracy,
                "coverage": expected_coverage,
                "hallucination_risk": expected_risk,
            }.items():
                if not math.isclose(
                    float(result[name]), expected_value, rel_tol=1e-12, abs_tol=1e-15
                ):
                    raise DataValidationError(
                        f"official_metrics.{benchmark}.{name} disagrees with outcome counts"
                    )
            if benchmark == "triviaqa":
                exact_match = _probability(
                    result["exact_match"], "official_metrics.triviaqa.exact_match"
                )
                _probability(result["token_f1"], "official_metrics.triviaqa.token_f1")
                if not math.isclose(exact_match, expected_accuracy, abs_tol=1e-15):
                    raise DataValidationError("TriviaQA exact match disagrees with C outcomes")
            elif benchmark == "simpleqa_verified":
                simpleqa_f1 = _probability(
                    result["simpleqa_f1"],
                    "official_metrics.simpleqa_verified.simpleqa_f1",
                )
                accuracy_given_attempted = _probability(
                    result["accuracy_given_attempted"],
                    "official_metrics.simpleqa_verified.accuracy_given_attempted",
                )
                attempt_rate = _probability(
                    result["attempt_rate"],
                    "official_metrics.simpleqa_verified.attempt_rate",
                )
                incorrect_attempted_rate = _probability(
                    result["incorrect_attempted_rate"],
                    "official_metrics.simpleqa_verified.incorrect_attempted_rate",
                )
                punting_rate = _probability(
                    result["punting_rate"],
                    "official_metrics.simpleqa_verified.punting_rate",
                )
                _probability(
                    result["hedging_rate"],
                    "official_metrics.simpleqa_verified.hedging_rate",
                )
                if counts["P"] != 0:
                    raise DataValidationError("SimpleQA official outcomes cannot contain partials")
                expected_attempted_accuracy = counts["C"] / attempted
                expected_f1 = (
                    2
                    * expected_accuracy
                    * expected_attempted_accuracy
                    / (expected_accuracy + expected_attempted_accuracy)
                    if expected_accuracy + expected_attempted_accuracy
                    else 0.0
                )
                if (
                    not math.isclose(attempt_rate, expected_coverage, abs_tol=1e-15)
                    or not math.isclose(incorrect_attempted_rate, expected_risk, abs_tol=1e-15)
                    or not math.isclose(punting_rate, counts["A"] / scorable, abs_tol=1e-15)
                    or not math.isclose(
                        accuracy_given_attempted,
                        expected_attempted_accuracy,
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    )
                    or not math.isclose(simpleqa_f1, expected_f1, rel_tol=1e-12, abs_tol=1e-15)
                ):
                    raise DataValidationError("SimpleQA official equations disagree with counts")
            else:
                index = _number(
                    result["omniscience_index"],
                    "official_metrics.aa_omniscience_public_600.omniscience_index",
                )
                rates = [
                    _probability(
                        result[name],
                        f"official_metrics.aa_omniscience_public_600.{name}",
                    )
                    for name in (
                        "correct_rate",
                        "partial_rate",
                        "incorrect_rate",
                        "abstention_rate",
                    )
                ]
                attempted_accuracy = _probability(
                    result["accuracy_given_attempted"],
                    "official_metrics.aa_omniscience_public_600.accuracy_given_attempted",
                )
                expected_rates = [counts[name] / scorable for name in ("C", "P", "I", "A")]
                if (
                    not -100 <= index <= 100
                    or not math.isclose(sum(rates), 1.0, abs_tol=1e-12)
                    or any(
                        not math.isclose(observed, expected, abs_tol=1e-15)
                        for observed, expected in zip(rates, expected_rates, strict=True)
                    )
                    or not math.isclose(
                        index,
                        100 * (counts["C"] - counts["I"]) / scorable,
                        abs_tol=1e-12,
                    )
                    or not math.isclose(attempted_accuracy, counts["C"] / attempted, abs_tol=1e-15)
                ):
                    raise DataValidationError("AA official result rates or index are invalid")
        for benchmark in sorted(_OFFICIAL_GRADER_BENCHMARKS):
            result = _entry(
                self.grader_failure_rates,
                benchmark,
                {
                    "responses",
                    "grader_calls",
                    "failed_calls",
                    "recovered_responses",
                    "terminal_failures",
                    "failure_rate",
                },
                "grader_failure_rates",
            )
            responses = _integer(
                result["responses"],
                f"grader_failure_rates.{benchmark}.responses",
                minimum=1,
            )
            grader_calls = _integer(
                result["grader_calls"],
                f"grader_failure_rates.{benchmark}.grader_calls",
                minimum=responses,
            )
            failed_calls = _integer(
                result["failed_calls"],
                f"grader_failure_rates.{benchmark}.failed_calls",
            )
            recovered = _integer(
                result["recovered_responses"],
                f"grader_failure_rates.{benchmark}.recovered_responses",
            )
            terminal = _integer(
                result["terminal_failures"],
                f"grader_failure_rates.{benchmark}.terminal_failures",
            )
            rate = _probability(
                result["failure_rate"], f"grader_failure_rates.{benchmark}.failure_rate"
            )
            if (
                failed_calls > grader_calls
                or recovered > responses
                or terminal > responses
                or not math.isclose(rate, failed_calls / grader_calls, abs_tol=1e-12)
            ):
                raise DataValidationError(f"grader failure counts disagree for {benchmark}")
            official = self.official_metrics[benchmark]
            assert isinstance(official, Mapping)
            official_counts = official["counts"]
            assert isinstance(official_counts, Mapping)
            if responses != official["total"] or terminal != official_counts["U"]:
                raise DataValidationError(
                    f"grader failure counts differ from official outcomes for {benchmark}"
                )
            bound = _entry(
                self.zero_error_bounds,
                benchmark,
                {
                    "attempted",
                    "errors",
                    "zero_errors_observed",
                    "confidence",
                    "one_sided_upper",
                },
                "zero_error_bounds",
            )
            attempted = _integer(
                bound["attempted"], f"zero_error_bounds.{benchmark}.attempted", minimum=1
            )
            errors = _integer(bound["errors"], f"zero_error_bounds.{benchmark}.errors")
            if not isinstance(bound["zero_errors_observed"], bool) or bound[
                "zero_errors_observed"
            ] is not (errors == 0):
                raise DataValidationError("zero-error status differs from observed errors")
            if attempted != official["attempted"] or errors != official_counts["I"]:
                raise DataValidationError(
                    f"confidence-bound counts differ from official outcomes for {benchmark}"
                )
            confidence = _probability(
                bound["confidence"], f"zero_error_bounds.{benchmark}.confidence"
            )
            if not 0 < confidence < 1:
                raise DataValidationError("zero-error confidence must be in (0, 1)")
            upper = _probability(
                bound["one_sided_upper"],
                f"zero_error_bounds.{benchmark}.one_sided_upper",
            )
            expected_upper = _exact_binomial_upper(errors, attempted, confidence)
            if not math.isclose(upper, expected_upper, rel_tol=1e-12, abs_tol=1e-15):
                raise DataValidationError(f"zero-error upper bound is incorrect for {benchmark}")
        agreement = self.human_audit["agreement_metrics"]
        assert isinstance(agreement, Mapping)
        if any(
            not -1 <= _number(value, f"human_audit.agreement_metrics.{name}") <= 1
            for name, value in agreement.items()
        ):
            raise DataValidationError("human-audit agreement coefficients must be in [-1, 1]")
        adjudication = self.human_audit["adjudication_summary"]
        if not isinstance(adjudication, Mapping) or set(adjudication) != {
            "rows",
            "disagreements",
        }:
            raise DataValidationError("human-audit adjudication summary is incomplete")
        rows = _integer(adjudication["rows"], "human_audit.adjudication_summary.rows", minimum=1)
        disagreements = _integer(
            adjudication["disagreements"],
            "human_audit.adjudication_summary.disagreements",
        )
        if disagreements > rows:
            raise DataValidationError("human-audit adjudication counts are invalid")
        confusion = self.human_audit["automated_human_confusion_matrix"]
        if not isinstance(confusion, Mapping) or not confusion:
            raise DataValidationError("human-audit confusion matrix must be a mapping")
        confusion_total = 0
        for key, value in confusion.items():
            if (
                not isinstance(key, str)
                or len(parts := key.split(":")) != 2
                or any(label not in _OUTCOME_LABELS for label in parts)
            ):
                raise DataValidationError("human-audit confusion labels are invalid")
            confusion_total += _integer(
                value,
                f"human_audit.automated_human_confusion_matrix.{key}",
            )
        if confusion_total != rows:
            raise DataValidationError("human-audit confusion counts differ from audited rows")
        if set(self.language_confusion) != {
            "requested_detected_matrix",
            "automated_metrics_by_language",
            "correct_to_wrong_language",
            "human_audit",
        }:
            raise DataValidationError(
                "language confusion must contain detector and adjudicated-human evidence"
            )
        language_matrix = self.language_confusion["requested_detected_matrix"]
        if not isinstance(language_matrix, Mapping) or not language_matrix:
            raise DataValidationError("language requested/detected matrix must be non-empty")
        for key, value in language_matrix.items():
            if (
                not isinstance(key, str)
                or len(parts := key.split(":")) != 2
                or any(not part for part in parts)
            ):
                raise DataValidationError("language requested/detected labels are invalid")
            _integer(value, f"language_confusion.requested_detected_matrix.{key}")
        automated = self.language_confusion["automated_metrics_by_language"]
        if (
            not isinstance(automated, Mapping)
            or not automated
            or not set(automated) <= SUPPORTED_LANGUAGES
        ):
            raise DataValidationError("automated language metrics are incomplete")
        automated_rows = 0
        for language, raw in automated.items():
            if not isinstance(raw, Mapping) or set(raw) != {
                "rows",
                "correct_output_language_rate",
                "non_target_script_token_rate",
                "code_switching_rate",
                "factual_accuracy",
                "abstention_rate",
            }:
                raise DataValidationError("automated per-language metrics have an invalid shape")
            automated_rows += _integer(
                raw["rows"],
                f"language_confusion.automated_metrics_by_language.{language}.rows",
                minimum=1,
            )
            for metric in (
                "correct_output_language_rate",
                "non_target_script_token_rate",
                "code_switching_rate",
                "factual_accuracy",
                "abstention_rate",
            ):
                _probability(
                    raw[metric],
                    f"language_confusion.automated_metrics_by_language.{language}.{metric}",
                )
        if automated_rows != sum(int(value) for value in language_matrix.values()):
            raise DataValidationError("automated language denominators differ from the matrix")
        transitions = self.language_confusion["correct_to_wrong_language"]
        if not isinstance(transitions, Mapping) or set(transitions) != {
            "eligible_baseline_correct_language_rows",
            "wrong_language_transitions",
            "rate",
            "by_language",
        }:
            raise DataValidationError("correct-to-wrong-language transitions are incomplete")
        eligible = _integer(
            transitions["eligible_baseline_correct_language_rows"],
            "language_confusion.correct_to_wrong_language.eligible_rows",
        )
        wrong = _integer(
            transitions["wrong_language_transitions"],
            "language_confusion.correct_to_wrong_language.transitions",
        )
        transition_rate = _probability(
            transitions["rate"], "language_confusion.correct_to_wrong_language.rate"
        )
        transition_languages = transitions["by_language"]
        if (
            wrong > eligible
            or not math.isclose(
                transition_rate, wrong / eligible if eligible else 0.0, abs_tol=1e-15
            )
            or not isinstance(transition_languages, Mapping)
            or set(transition_languages) != set(automated)
        ):
            raise DataValidationError("correct-to-wrong-language transition totals are invalid")
        summed_eligible = 0
        summed_wrong = 0
        for language, raw in transition_languages.items():
            if not isinstance(raw, Mapping) or set(raw) != {
                "eligible_baseline_correct_language_rows",
                "wrong_language_transitions",
                "rate",
            }:
                raise DataValidationError("per-language transition metrics are incomplete")
            language_eligible = _integer(
                raw["eligible_baseline_correct_language_rows"],
                f"language_confusion.correct_to_wrong_language.{language}.eligible",
            )
            language_wrong = _integer(
                raw["wrong_language_transitions"],
                f"language_confusion.correct_to_wrong_language.{language}.wrong",
            )
            language_rate = _probability(
                raw["rate"],
                f"language_confusion.correct_to_wrong_language.{language}.rate",
            )
            if language_wrong > language_eligible or not math.isclose(
                language_rate,
                language_wrong / language_eligible if language_eligible else 0.0,
                abs_tol=1e-15,
            ):
                raise DataValidationError("per-language transition metrics are invalid")
            summed_eligible += language_eligible
            summed_wrong += language_wrong
        if (summed_eligible, summed_wrong) != (eligible, wrong):
            raise DataValidationError("per-language transitions do not sum to the total")
        language_audit = self.language_confusion["human_audit"]
        if not isinstance(language_audit, Mapping) or set(language_audit) != {
            "agreement_metrics",
            "adjudication_summary",
            "automated_human_confusion_matrix",
            "human_consistency_score",
            "record_binding_digest",
        }:
            raise DataValidationError("adjudicated language evidence is incomplete")
        language_agreement = language_audit["agreement_metrics"]
        language_summary = language_audit["adjudication_summary"]
        language_human_confusion = language_audit["automated_human_confusion_matrix"]
        language_score = language_audit["human_consistency_score"]
        if (
            not isinstance(language_agreement, Mapping)
            or set(language_agreement) != {"cohen_kappa", "krippendorff_alpha"}
            or not isinstance(language_summary, Mapping)
            or set(language_summary) != {"rows", "disagreements"}
            or not isinstance(language_human_confusion, Mapping)
            or not language_human_confusion
            or not isinstance(language_score, Mapping)
            or set(language_score) != {"consistent", "judged", "rate"}
            or not _SHA256.fullmatch(str(language_audit["record_binding_digest"]))
        ):
            raise DataValidationError("adjudicated language evidence has an invalid shape")
        if any(
            not -1 <= _number(value, f"language_confusion.human_audit.{name}") <= 1
            for name, value in language_agreement.items()
        ):
            raise DataValidationError("language agreement coefficients must be in [-1, 1]")
        language_rows = _integer(
            language_summary["rows"],
            "language_confusion.human_audit.adjudication_summary.rows",
            minimum=1,
        )
        language_disagreements = _integer(
            language_summary["disagreements"],
            "language_confusion.human_audit.adjudication_summary.disagreements",
        )
        if language_disagreements > language_rows:
            raise DataValidationError("language adjudication counts are invalid")
        language_confusion_total = 0
        allowed_language_labels = {"CONSISTENT", "SWITCHED", "U"}
        for key, value in language_human_confusion.items():
            if (
                not isinstance(key, str)
                or len(parts := key.split(":")) != 2
                or any(label not in allowed_language_labels for label in parts)
            ):
                raise DataValidationError("adjudicated language confusion labels are invalid")
            language_confusion_total += _integer(
                value,
                f"language_confusion.human_audit.automated_human_confusion_matrix.{key}",
            )
        if language_confusion_total != language_rows:
            raise DataValidationError(
                "adjudicated language confusion counts differ from audited rows"
            )
        consistent = _integer(
            language_score["consistent"],
            "language_confusion.human_audit.human_consistency_score.consistent",
        )
        judged = _integer(
            language_score["judged"],
            "language_confusion.human_audit.human_consistency_score.judged",
            minimum=1,
        )
        score = _probability(
            language_score["rate"],
            "language_confusion.human_audit.human_consistency_score.rate",
        )
        if consistent > judged or not math.isclose(score, consistent / judged, abs_tol=1e-12):
            raise DataValidationError("adjudicated human language score is invalid")
        runtime = self.runtime_replication.get("local_vllm_execution")
        if not isinstance(runtime, Mapping) or not {
            "passed",
            "record_count",
            "mean_latency_seconds",
            "p95_latency_seconds",
            "maximum_latency_seconds",
            "mean_candidate_generation_seconds",
            "candidate_generated_record_count",
            "pre_generation_abstention_count",
            "maximum_peak_memory_bytes",
            "mean_prompt_tokens_per_second",
            "mean_generation_tokens_per_second",
            "signed_execution_receipts",
            "runtime_identity_sha256",
            "runtime_session_identity_sha256",
            "intervention_site_evidence_sha256",
            "runtime_counts",
            "vllm_versions",
            "torch_versions",
            "transformers_versions",
            "nvidia_drivers",
            "gpu_models",
            "operating_systems",
            "architectures",
            "cuda_capabilities",
            "cuda_runtimes",
            "quantization_executions",
            "gpu_total_memory_bytes",
            "intervention_site_counts",
            "policy_action_counts",
            "mean_activation_delta_norm_by_site",
        } <= set(runtime):
            raise DataValidationError("runtime replication evidence is incomplete")
        if runtime["passed"] is not True:
            raise DataValidationError("final analysis requires a passed native VLLM replay")
        runtime_count = _integer(
            runtime["record_count"],
            "runtime_replication.local_vllm_execution.record_count",
            minimum=1,
        )
        signed_count = _integer(
            runtime["signed_execution_receipts"],
            "runtime_replication.local_vllm_execution.signed_execution_receipts",
        )
        if signed_count != runtime_count:
            raise DataValidationError("not every final record has a signed execution receipt")
        for name in (
            "mean_latency_seconds",
            "p95_latency_seconds",
            "maximum_latency_seconds",
        ):
            if _number(runtime[name], f"runtime_replication.local_vllm_execution.{name}") <= 0:
                raise DataValidationError("runtime latency summaries must be positive")
        generated_count = _integer(
            runtime["candidate_generated_record_count"],
            "runtime_replication.local_vllm_execution.candidate_generated_record_count",
        )
        pre_generation_abstentions = _integer(
            runtime["pre_generation_abstention_count"],
            "runtime_replication.local_vllm_execution.pre_generation_abstention_count",
        )
        if generated_count + pre_generation_abstentions != runtime_count:
            raise DataValidationError("runtime candidate counts differ from record count")
        generated_metrics = (
            _number(
                runtime["mean_candidate_generation_seconds"],
                "runtime_replication.local_vllm_execution.mean_candidate_generation_seconds",
            ),
            _number(
                runtime["mean_prompt_tokens_per_second"],
                "runtime_replication.local_vllm_execution.mean_prompt_tokens_per_second",
            ),
            _number(
                runtime["mean_generation_tokens_per_second"],
                "runtime_replication.local_vllm_execution.mean_generation_tokens_per_second",
            ),
        )
        if any(value < 0 for value in generated_metrics) or (
            (generated_count == 0 and any(value != 0 for value in generated_metrics))
            or (generated_count > 0 and any(value <= 0 for value in generated_metrics))
        ):
            raise DataValidationError("runtime generated-row metrics differ from candidate count")
        _integer(
            runtime["maximum_peak_memory_bytes"],
            "runtime_replication.local_vllm_execution.maximum_peak_memory_bytes",
            minimum=1,
        )
        if not _SHA256.fullmatch(str(runtime["runtime_identity_sha256"])):
            raise DataValidationError("runtime replication lacks an identity digest")
        if runtime["runtime_session_identity_sha256"] != runtime["runtime_identity_sha256"]:
            raise DataValidationError("runtime session identity differs from attested identity")
        if not _SHA256.fullmatch(str(runtime["intervention_site_evidence_sha256"])):
            raise DataValidationError("runtime replication lacks intervention-site evidence")
        _integer(
            runtime["gpu_total_memory_bytes"],
            "runtime_replication.local_vllm_execution.gpu_total_memory_bytes",
            minimum=1,
        )
        runtime_counts = runtime["runtime_counts"]
        if (
            not isinstance(runtime_counts, Mapping)
            or sum(
                _integer(value, f"runtime_replication.local_vllm_execution.runtime_counts.{key}")
                for key, value in runtime_counts.items()
            )
            != runtime_count
        ):
            raise DataValidationError("runtime counts differ from the final record count")
        for name in (
            "vllm_versions",
            "torch_versions",
            "transformers_versions",
            "nvidia_drivers",
            "gpu_models",
            "operating_systems",
            "architectures",
            "cuda_capabilities",
            "cuda_runtimes",
            "quantization_executions",
        ):
            values = runtime[name]
            if (
                not isinstance(values, Mapping)
                or len(values) != 1
                or sum(
                    _integer(
                        count,
                        f"runtime_replication.local_vllm_execution.{name}.{key}",
                        minimum=1,
                    )
                    for key, count in values.items()
                )
                != 1
            ):
                raise DataValidationError(f"runtime identity field {name} is invalid")
        for name in ("intervention_site_counts", "policy_action_counts"):
            values = runtime[name]
            if (
                not isinstance(values, Mapping)
                or not values
                or sum(
                    _integer(
                        count,
                        f"runtime_replication.local_vllm_execution.{name}.{key}",
                    )
                    for key, count in values.items()
                )
                != runtime_count
            ):
                raise DataValidationError(f"runtime evidence field {name} is incomplete")
        deltas = runtime["mean_activation_delta_norm_by_site"]
        if (
            not isinstance(deltas, Mapping)
            or not deltas
            or any(
                _number(
                    value,
                    f"runtime_replication.local_vllm_execution.mean_activation_delta_norm_by_site.{key}",
                )
                < 0
                for key, value in deltas.items()
            )
        ):
            raise DataValidationError("runtime activation-delta evidence is invalid")

    def validate_against_protocol(self, protocol: AnalysisProtocol) -> None:
        if set(self.primary_contrasts) != {
            value.contrast_id for value in protocol.primary_contrasts
        } or set(self.noninferiority) != set(protocol.noninferiority_margins):
            raise DataValidationError(
                "final results differ from the confirmatory analysis protocol"
            )
        for family, value in self.holm_adjusted_tests.items():
            assert isinstance(value, Mapping)
            expected_rejection = float(value["adjusted_p_value"]) <= protocol.alpha
            if value["rejected"] is not expected_rejection:
                raise DataValidationError(
                    f"Holm rejection decision for {family} differs from the protocol alpha"
                )
        for metric, specification in protocol.noninferiority_margins.items():
            value = self.noninferiority[metric]
            assert isinstance(value, Mapping)
            margin = float(value["margin"])
            comparisons = value["comparisons"]
            assert isinstance(comparisons, Mapping)
            expected_pass = all(
                float(comparison["one_sided_lower"]) > -specification.margin
                and float(comparison["adjusted_p_value"]) <= protocol.alpha
                for comparison in comparisons.values()
            )
            if (
                not math.isclose(margin, specification.margin, rel_tol=0, abs_tol=1e-15)
                or value["higher_is_better"] is not specification.higher_is_better
                or value["passed"] is not expected_pass
            ):
                raise DataValidationError(
                    f"non-inferiority result for {metric} differs from its preregistered test"
                )
            composite_metrics = self.composite_side_effects["metrics"]
            assert isinstance(composite_metrics, Mapping)
            composite = composite_metrics[metric]
            assert isinstance(composite, Mapping)
            composite_pass = (
                float(composite["one_sided_lower"]) > -specification.margin
                and float(composite["adjusted_p_value"]) <= protocol.alpha
            )
            if (
                not math.isclose(
                    float(composite["margin"]),
                    specification.margin,
                    rel_tol=0,
                    abs_tol=1e-15,
                )
                or composite["higher_is_better"] is not specification.higher_is_better
                or composite["rejected"]
                is not (float(composite["adjusted_p_value"]) <= protocol.alpha)
                or composite["passed"] is not composite_pass
            ):
                raise DataValidationError(
                    f"composite non-inferiority result for {metric} differs from protocol"
                )
        adjudication = self.human_audit["adjudication_summary"]
        assert isinstance(adjudication, Mapping)
        required_rows = (
            protocol.human_audit.minimum_responses_per_benchmark_model
            * len(_FACTUAL_BENCHMARKS)
            * len(_MODEL_NAMES_BY_REPOSITORY)
        )
        if int(adjudication["rows"]) < required_rows:
            raise DataValidationError(
                f"human audit requires at least {required_rows} benchmark-model rows"
            )

    def validate_against_records(self, records: Iterable[GenerationRecord]) -> None:
        """Recompute final factual metrics from the verified frozen-composite ledger."""

        record_values = tuple(records)
        grouped: dict[str, list[GenerationRecord]] = {
            benchmark: [] for benchmark in _FACTUAL_BENCHMARKS
        }
        for record in record_values:
            if record.benchmark in grouped:
                grouped[record.benchmark].append(record)
        if any(not values for values in grouped.values()):
            raise DataValidationError("final analysis lacks an E10 factual benchmark family")
        for benchmark, values in grouped.items():
            counts = Counter(record.outcome.value for record in values)
            expected_counts = {label: counts[label] for label in _OUTCOME_LABELS}
            reported = self.official_metrics[benchmark]
            assert isinstance(reported, Mapping)
            reported_counts = reported["counts"]
            if not isinstance(reported_counts, Mapping) or dict(reported_counts) != expected_counts:
                raise DataValidationError(
                    f"official metrics for {benchmark} differ from E10 ledger outcomes"
                )
            if benchmark == "triviaqa":
                exact: list[float] = []
                token_f1: list[float] = []
                for record in values:
                    exact_value = record.metadata.get("official_exact_match")
                    f1_value = record.metadata.get("official_token_f1")
                    if (
                        isinstance(exact_value, bool)
                        or not isinstance(exact_value, int | float)
                        or isinstance(f1_value, bool)
                        or not isinstance(f1_value, int | float)
                    ):
                        raise DataValidationError(
                            "TriviaQA E10 records lack deterministic official scores"
                        )
                    exact.append(float(exact_value))
                    token_f1.append(float(f1_value))
                if not math.isclose(
                    float(reported["exact_match"]), sum(exact) / len(exact), abs_tol=1e-15
                ) or not math.isclose(
                    float(reported["token_f1"]),
                    sum(token_f1) / len(token_f1),
                    abs_tol=1e-15,
                ):
                    raise DataValidationError("TriviaQA official scores differ from E10 records")
            elif benchmark == "simpleqa_verified":
                scorable_values = [
                    record for record in values if record.outcome is not Outcome.UNSCORABLE
                ]
                if any(
                    not simpleqa_hedging_evidence_is_valid(
                        record.raw_output,
                        record.metadata.get("simpleqa_hedging_evidence"),
                    )
                    for record in values
                ):
                    raise DataValidationError("SimpleQA hedging evidence differs from E10 records")
                expected_hedging = sum(
                    bool(record.metadata["simpleqa_hedging_evidence"]["hedged"])
                    for record in scorable_values
                ) / len(scorable_values)
                if not math.isclose(
                    float(reported["hedging_rate"]), expected_hedging, abs_tol=1e-15
                ):
                    raise DataValidationError("SimpleQA hedging rate differs from E10 records")
        language_records: list[GenerationRecord] = []
        requested_detected: Counter[str] = Counter()
        language_groups: defaultdict[str, list[GenerationRecord]] = defaultdict(list)
        for record in record_values:
            if record.benchmark != "language_consistency":
                continue
            language_records.append(record)
            requested = record.metadata.get("requested_language")
            detected = record.metadata.get("detected_language")
            evidence = record.metadata.get("language_evaluation_evidence")
            aliases = evidence.get("accepted_aliases") if isinstance(evidence, Mapping) else None
            if (
                requested not in SUPPORTED_LANGUAGES
                or (detected is not None and (not isinstance(detected, str) or not detected))
                or not isinstance(aliases, list)
                or not aliases
                or any(not isinstance(value, str) or not value for value in aliases)
                or not isinstance(evidence, Mapping)
                or dict(evidence)
                != language_response_evidence(record.raw_output, str(requested), tuple(aliases))
                or record.outcome.value != evidence["factual_outcome"]
            ):
                raise DataValidationError("E10 language records lack deterministic labels")
            language = str(requested)
            requested_detected[f"{language}:{detected or 'und'}"] += 1
            language_groups[language].append(record)
        expected_language = self.language_confusion["requested_detected_matrix"]
        if not language_records or dict(sorted(requested_detected.items())) != dict(
            expected_language
        ):
            raise DataValidationError("language confusion differs from E10 records")
        expected_automated = {
            language: {
                "rows": len(values),
                "correct_output_language_rate": sum(
                    value.metadata.get("requested_language_correct") is True for value in values
                )
                / len(values),
                "non_target_script_token_rate": sum(
                    float(value.metadata["non_target_script_token_rate"]) for value in values
                )
                / len(values),
                "code_switching_rate": sum(
                    value.metadata.get("code_switching") is True for value in values
                )
                / len(values),
                "factual_accuracy": sum(value.outcome is Outcome.CORRECT for value in values)
                / len(values),
                "abstention_rate": sum(value.outcome is Outcome.ABSTENTION for value in values)
                / len(values),
            }
            for language, values in sorted(language_groups.items())
        }
        reported_automated = self.language_confusion["automated_metrics_by_language"]
        if (
            not isinstance(reported_automated, Mapping)
            or dict(reported_automated) != expected_automated
        ):
            raise DataValidationError("per-language metrics differ from E10 records")

    def to_dict(self) -> dict[str, Any]:
        return {name: dict(getattr(self, name)) for name in _RESULT_FIELDS}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FinalAnalysisResults:
        if set(value) != set(_RESULT_FIELDS):
            raise DataValidationError("final analysis result keys differ from schema version 1")
        sections: dict[str, Mapping[str, Any]] = {}
        for name in _RESULT_FIELDS:
            section = value[name]
            if not isinstance(section, Mapping):
                raise DataValidationError(f"final analysis result section {name} is not a mapping")
            sections[name] = {str(key): item for key, item in section.items()}
        return cls(**sections)


@dataclass(frozen=True, slots=True)
class ReportSource:
    path: Path
    data_path: Path
    generator_revision: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "data_path", Path(self.data_path))
        if not _SHA256.fullmatch(self.generator_revision):
            raise DataValidationError("report generator revision must be a SHA-256 fingerprint")


@dataclass(frozen=True, slots=True)
class ReportArtifact:
    filename: str
    sha256: str
    source_data_filename: str
    source_data_sha256: str
    source_data_digest: str
    generator_revision: str
    result_dependencies: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FrozenAnalysisEvidence:
    """Immutable result payload bound to every contributing completed phase."""

    directory: Path
    analysis_protocol_digest: str
    study_protocol_digest: str
    phase_completion_digests: Mapping[str, str]
    phase_record_set_digests: Mapping[str, str]
    phase_records_sha256: Mapping[str, str]
    results: FinalAnalysisResults
    results_sha256: str
    evidence_digest: str
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class FinalAnalysisBundle:
    directory: Path
    analysis_protocol_digest: str
    research_plan_sha256: str
    phase_completion_digests: Mapping[str, str]
    human_audit_queue_manifest_digest: str
    human_audit_results_manifest_digest: str
    analysis_evidence_digest: str
    analysis_evidence_sha256: str
    results_sha256: str
    report_artifacts: Mapping[str, ReportArtifact]
    bundle_digest: str
    schema_version: int = 2


def _required_artifacts(protocol: AnalysisProtocol) -> set[str]:
    return set(protocol.required_report_outputs) | set(protocol.human_audit.required_outputs)


def _require_exact_directory(
    path: Path,
    *,
    regular_files: set[str],
    regular_directories: set[str],
    context: str,
) -> None:
    if path.is_symlink() or not path.is_dir():
        raise FrozenArtifactError(f"{context} must be a regular directory")
    try:
        entries = tuple(path.iterdir())
    except OSError as exc:
        raise FrozenArtifactError(f"cannot inventory {context}: {exc}") from exc
    expected = regular_files | regular_directories
    if {entry.name for entry in entries} != expected:
        raise FrozenArtifactError(f"{context} contains missing or unexpected entries")
    for entry in entries:
        if entry.is_symlink():
            raise FrozenArtifactError(f"{context} contains a symbolic link: {entry.name}")
        if entry.name in regular_files and not entry.is_file():
            raise FrozenArtifactError(f"{context} contains a non-regular file: {entry.name}")
        if entry.name in regular_directories and not entry.is_dir():
            raise FrozenArtifactError(f"{context} contains a non-regular directory: {entry.name}")


def _verified_phase_sources(
    study: StudyProtocol,
    phase_run_directories: Mapping[ExperimentPhase | str, str | Path],
) -> tuple[dict[str, str], dict[ExperimentPhase, Any]]:
    normalized = {ExperimentPhase(key): Path(path) for key, path in phase_run_directories.items()}
    required = set(_FINAL_ANALYSIS_PHASES)
    if set(normalized) != required:
        raise DataValidationError(
            "final analysis requires exactly the E1, E3, E6, E7, E8, E9, and E10 runs"
        )
    result: dict[str, str] = {}
    ledgers: dict[ExperimentPhase, Any] = {}
    for phase in sorted(required, key=lambda value: value.ordinal):
        ledger = open_phase_prerequisite(normalized[phase], phase=phase, study=study)
        completion = ledger.verify_complete()
        if completion.phase is not phase:
            raise FrozenArtifactError(f"analysis source for {phase.value} is a different phase")
        result[phase.value] = completion.completion_digest
        ledgers[phase] = ledger
    return result, ledgers


def _verified_phase_digests(
    study: StudyProtocol,
    phase_run_directories: Mapping[ExperimentPhase | str, str | Path],
) -> dict[str, str]:
    return _verified_phase_sources(study, phase_run_directories)[0]


def _human_audit_phase_sources(
    phase_run_directories: Mapping[ExperimentPhase | str, str | Path],
) -> Mapping[ExperimentPhase | str, str | Path]:
    normalized = {ExperimentPhase(phase): path for phase, path in phase_run_directories.items()}
    return {phase: normalized[phase] for phase in (ExperimentPhase.E9, ExperimentPhase.E10)}


def _record_evidence(
    ledger: PhaseRunLedger,
) -> tuple[int, str, str]:
    records = tuple(ledger.records())
    ordered = tuple(sorted(records, key=lambda value: (value.condition_id, value.question_id)))
    keys = [(value.condition_id, value.question_id) for value in ordered]
    if not ordered or len(set(keys)) != len(keys):
        raise DataValidationError("analysis evidence requires unique, non-empty phase records")
    return (
        len(ordered),
        stable_hash(keys),
        stable_hash([value.to_dict() for value in ordered]),
    )


def _analysis_evidence_body(
    *,
    protocol: AnalysisProtocol,
    study: StudyProtocol,
    phase_digests: Mapping[str, str],
    ledgers: Mapping[ExperimentPhase, Any],
    results: FinalAnalysisResults,
    results_sha256: str,
    derived_analysis: DerivedFinalAnalysis,
) -> dict[str, Any]:
    record_counts: dict[str, int] = {}
    record_keys: dict[str, str] = {}
    record_digests: dict[str, str] = {}
    record_sets: dict[str, str] = {}
    contract_digests: dict[str, str] = {}
    evaluation_snapshots: dict[str, str] = {}
    snapshot_inputs = {
        ExperimentPhase.E9: "frozen_evaluation_scripts",
        ExperimentPhase.E10: "evaluation_scripts",
    }
    for phase in (ExperimentPhase.E9, ExperimentPhase.E10):
        ledger = ledgers[phase]
        completion = ledger.verify_complete()
        count, keys_digest, records_digest = _record_evidence(ledger)
        if count != completion.record_count:
            raise DataValidationError(
                f"analysis evidence record count differs from {phase.value} completion"
            )
        snapshot = ledger.contract.input_fingerprints.get(snapshot_inputs[phase])
        if snapshot is None or not _SHA256.fullmatch(snapshot):
            raise DataValidationError(
                f"analysis evidence lacks the frozen {phase.value} evaluator snapshot"
            )
        record_counts[phase.value] = count
        record_keys[phase.value] = keys_digest
        record_digests[phase.value] = records_digest
        record_sets[phase.value] = completion.record_set_digest
        contract_digests[phase.value] = completion.contract_digest
        evaluation_snapshots[phase.value] = snapshot
    source_phases = (
        sorted(derived_analysis.source_record_digests)
        if derived_analysis is not None
        else [ExperimentPhase.E9.value, ExperimentPhase.E10.value]
    )
    source_completion_digests = (
        dict(derived_analysis.phase_completion_digests)
        if derived_analysis is not None
        else dict(phase_digests)
    )
    source_records = (
        dict(derived_analysis.source_record_digests)
        if derived_analysis is not None
        else dict(record_digests)
    )
    body: dict[str, Any] = {
        "schema_version": 2 if derived_analysis is not None else 1,
        "analysis_protocol_digest": protocol.digest,
        "study_protocol_digest": study.digest,
        "phase_completion_digests": dict(phase_digests),
        "phase_contract_digests": contract_digests,
        "phase_record_set_digests": record_sets,
        "phase_record_counts": record_counts,
        "phase_record_keys_sha256": record_keys,
        "phase_records_sha256": record_digests,
        "evaluation_snapshot_sha256": evaluation_snapshots,
        "result_record_bindings": {
            name: {
                "result_sha256": stable_hash(dict(getattr(results, name))),
                "source_phases": source_phases,
                "source_completion_digests": source_completion_digests,
                "source_record_set_digests": dict(record_sets),
                "source_record_keys_sha256": dict(record_keys),
                "source_records_sha256": source_records,
                "evaluator_snapshot_sha256": dict(evaluation_snapshots),
            }
            for name in _RESULT_FIELDS
        },
        "results_sha256": results_sha256,
    }
    if derived_analysis is not None:
        if canonical_json(derived_analysis.results.to_dict()) != canonical_json(results.to_dict()):
            raise DataValidationError("derived analysis differs from the frozen results")
        body["record_derivation"] = {
            "derivation_digest": derived_analysis.derivation_digest,
            "source_record_digests": dict(derived_analysis.source_record_digests),
            "phase_completion_digests": dict(derived_analysis.phase_completion_digests),
            "e9_analysis_digest": derived_analysis.e9_analysis_digest,
            "human_audit_manifest_digest": (derived_analysis.human_audit_manifest_digest),
            "robustness_record_digest": derived_analysis.robustness_record_digest,
            "runtime_attestation_digest": derived_analysis.runtime_attestation_digest,
        }
    return body


def _write_frozen_analysis_evidence_from_derivation(
    directory: str | Path,
    *,
    protocol: AnalysisProtocol,
    study: StudyProtocol,
    phase_run_directories: Mapping[ExperimentPhase | str, str | Path],
    derived_analysis: DerivedFinalAnalysis,
) -> FrozenAnalysisEvidence:
    """Private materializer; callers cannot supply a result independently."""

    results = derived_analysis.results

    path_inputs = {"frozen analysis evidence": directory}
    path_inputs.update(
        {
            f"{ExperimentPhase(phase).value} phase ledger": path
            for phase, path in phase_run_directories.items()
        }
    )
    normalized_paths = validate_active_study_artifact_paths(path_inputs)
    destination = normalized_paths["frozen analysis evidence"]
    phase_run_directories = {
        ExperimentPhase(phase): normalized_paths[f"{ExperimentPhase(phase).value} phase ledger"]
        for phase in phase_run_directories
    }
    if destination.exists():
        raise FrozenArtifactError(f"refusing to overwrite analysis evidence: {destination}")
    phase_digests, ledgers = _verified_phase_sources(study, phase_run_directories)
    results.validate_against_protocol(protocol)
    results.validate_against_records(ledgers[ExperimentPhase.E10].records())
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        normalized = json.loads(canonical_json(results.to_dict()))
        results_path = stage / "results.json"
        results_path.write_text(
            json.dumps(normalized, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        body = _analysis_evidence_body(
            protocol=protocol,
            study=study,
            phase_digests=phase_digests,
            ledgers=ledgers,
            results=results,
            results_sha256=sha256_file(results_path),
            derived_analysis=derived_analysis,
        )
        (stage / "manifest.json").write_text(
            json.dumps({**body, "evidence_digest": stable_hash(body)}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return _verify_frozen_analysis_evidence_from_derivation(
        destination,
        expected_protocol=protocol,
        study=study,
        phase_run_directories=phase_run_directories,
        expected_derivation=derived_analysis,
    )


def _verify_frozen_analysis_evidence_from_derivation(
    directory: str | Path,
    *,
    expected_protocol: AnalysisProtocol,
    study: StudyProtocol,
    phase_run_directories: Mapping[ExperimentPhase | str, str | Path],
    expected_derivation: DerivedFinalAnalysis,
) -> FrozenAnalysisEvidence:
    """Rebind a frozen result payload to every live contributing ledger."""

    source = Path(directory)
    _require_exact_directory(
        source,
        regular_files={"manifest.json", "results.json"},
        regular_directories=set(),
        context="frozen analysis evidence",
    )
    try:
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        results_value = json.loads((source / "results.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read frozen analysis evidence: {exc}") from exc
    if not isinstance(manifest, dict) or not isinstance(results_value, Mapping):
        raise FrozenArtifactError("frozen analysis evidence roots must be mappings")
    evidence_digest = manifest.pop("evidence_digest", None)
    if not isinstance(evidence_digest, str) or evidence_digest != stable_hash(manifest):
        raise FrozenArtifactError("frozen analysis evidence digest mismatch")
    schema_version = manifest.get("schema_version")
    if schema_version != 2:
        raise FrozenArtifactError("frozen analysis evidence schema is unsupported")
    results_path = source / "results.json"
    try:
        results = FinalAnalysisResults.from_dict(results_value)
        results.validate_against_protocol(expected_protocol)
        phase_digests, ledgers = _verified_phase_sources(study, phase_run_directories)
        results.validate_against_records(ledgers[ExperimentPhase.E10].records())
        expected_body = _analysis_evidence_body(
            protocol=expected_protocol,
            study=study,
            phase_digests=phase_digests,
            ledgers=ledgers,
            results=results,
            results_sha256=sha256_file(results_path),
            derived_analysis=expected_derivation,
        )
    except DataValidationError as exc:
        raise FrozenArtifactError(f"invalid record-bound analysis evidence: {exc}") from exc
    if manifest != expected_body:
        raise FrozenArtifactError("analysis evidence differs from live records or frozen results")
    return FrozenAnalysisEvidence(
        directory=source,
        analysis_protocol_digest=expected_protocol.digest,
        study_protocol_digest=study.digest,
        phase_completion_digests=MappingProxyType(dict(phase_digests)),
        phase_record_set_digests=MappingProxyType(
            {str(key): str(value) for key, value in manifest["phase_record_set_digests"].items()}
        ),
        phase_records_sha256=MappingProxyType(
            {str(key): str(value) for key, value in manifest["phase_records_sha256"].items()}
        ),
        results=results,
        results_sha256=str(manifest["results_sha256"]),
        evidence_digest=evidence_digest,
        schema_version=int(schema_version),
    )


def write_frozen_analysis_evidence(
    directory: str | Path,
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
) -> FrozenAnalysisEvidence:
    """Replay every raw source, then atomically freeze the derived results."""

    from mfh.analysis.derivation import derive_final_analysis_from_artifacts

    derived = derive_final_analysis_from_artifacts(
        protocol=protocol,
        study=study,
        phase_run_directories=phase_run_directories,
        robustness_result_directory=robustness_result_directory,
        human_audit_queue_directory=human_audit_queue_directory,
        human_audit_results_directory=human_audit_results_directory,
        human_audit_blinding_key=human_audit_blinding_key,
        aa_official_directory=aa_official_directory,
        expected_aa_official_manifest_digest=expected_aa_official_manifest_digest,
    )
    return _write_frozen_analysis_evidence_from_derivation(
        directory,
        protocol=protocol,
        study=study,
        phase_run_directories=phase_run_directories,
        derived_analysis=derived,
    )


def verify_frozen_analysis_evidence(
    directory: str | Path,
    *,
    expected_protocol: AnalysisProtocol,
    study: StudyProtocol,
    phase_run_directories: Mapping[ExperimentPhase | str, str | Path],
    robustness_result_directory: str | Path,
    human_audit_queue_directory: str | Path,
    human_audit_results_directory: str | Path,
    human_audit_blinding_key: bytes,
    aa_official_directory: str | Path,
    expected_aa_official_manifest_digest: str,
) -> FrozenAnalysisEvidence:
    """Replay every source before accepting a schema-v2 evidence directory."""

    from mfh.analysis.derivation import derive_final_analysis_from_artifacts

    derived = derive_final_analysis_from_artifacts(
        protocol=expected_protocol,
        study=study,
        phase_run_directories=phase_run_directories,
        robustness_result_directory=robustness_result_directory,
        human_audit_queue_directory=human_audit_queue_directory,
        human_audit_results_directory=human_audit_results_directory,
        human_audit_blinding_key=human_audit_blinding_key,
        aa_official_directory=aa_official_directory,
        expected_aa_official_manifest_digest=expected_aa_official_manifest_digest,
    )
    return _verify_frozen_analysis_evidence_from_derivation(
        directory,
        expected_protocol=expected_protocol,
        study=study,
        phase_run_directories=phase_run_directories,
        expected_derivation=derived,
    )


def _analysis_record_index(
    ledgers: Mapping[ExperimentPhase, Any],
) -> dict[tuple[str, str], GenerationRecord]:
    result: dict[tuple[str, str], GenerationRecord] = {}
    for phase in (ExperimentPhase.E9, ExperimentPhase.E10):
        for record in ledgers[phase].records():
            key = (record.condition_id, record.question_id)
            if key in result:
                raise DataValidationError("analysis source ledgers contain duplicate record keys")
            result[key] = record
    return result


def _validate_phase_digests(value: Mapping[str, str]) -> dict[str, str]:
    normalized = {str(key): str(digest) for key, digest in value.items()}
    if set(normalized) != {phase.value for phase in _FINAL_ANALYSIS_PHASES} or any(
        not _SHA256.fullmatch(digest) for digest in normalized.values()
    ):
        raise DataValidationError("final analysis lacks a contributing phase completion digest")
    return normalized


def _flatten_report_values(value: Any, prefix: str = "") -> dict[str, str]:
    if isinstance(value, str) and _SHA256.fullmatch(value):
        return {prefix: canonical_json(value)}
    if isinstance(value, bool | int | float):
        if isinstance(value, float) and not math.isfinite(value):
            raise DataValidationError("report data contains a non-finite value")
        return {prefix: canonical_json(value)}
    if isinstance(value, Mapping):
        result: dict[str, str] = {}
        for key, item in sorted(value.items()):
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten_report_values(item, child))
        return result
    if isinstance(value, list | tuple):
        result = {}
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            result.update(_flatten_report_values(item, child))
        return result
    raise DataValidationError("report source data contains a non-quantitative value")


def _tabular_rows(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.lstrip().startswith("#")
    ]
    if not lines:
        raise DataValidationError(f"report CSV is empty: {path}")
    reader = csv.DictReader(lines)
    if reader.fieldnames is None:
        raise DataValidationError(f"report CSV has no header: {path}")
    rows = [dict(row) for row in reader]
    return tuple(reader.fieldnames), rows


def _validate_human_audit_evidence(
    audit: HumanAuditResults,
    *,
    results: FinalAnalysisResults,
    adjudicated_report: Path,
) -> None:
    payload = audit.summary.get("factual_reporting_payload")
    if not isinstance(payload, Mapping) or json.loads(canonical_json(payload)) != json.loads(
        canonical_json(results.human_audit)
    ):
        raise DataValidationError("typed human-audit results differ from finalized audit evidence")
    evidence_fields, evidence_rows = _tabular_rows(audit.directory / "adjudicated-factual.csv")
    report_fields, report_rows = _tabular_rows(adjudicated_report)
    if (
        evidence_fields != _AUDIT_COLUMNS
        or report_fields != _AUDIT_COLUMNS
        or evidence_rows != report_rows
    ):
        raise DataValidationError(
            "adjudicated report is detached from the finalized human-audit bundle"
        )


def _validate_zero_error_csv(path: Path, results: FinalAnalysisResults) -> None:
    fields, rows = _tabular_rows(path)
    expected_fields = (
        "benchmark",
        "attempted",
        "errors",
        "zero_errors_observed",
        "confidence",
        "one_sided_upper",
    )
    if fields != expected_fields or len(rows) != len(_OFFICIAL_GRADER_BENCHMARKS):
        raise DataValidationError("zero-error CSV schema or row count is invalid")
    observed: dict[str, dict[str, float | int]] = {}
    try:
        for row in rows:
            benchmark = row["benchmark"]
            if benchmark in observed:
                raise DataValidationError("zero-error CSV repeats a benchmark")
            if row["zero_errors_observed"] not in {"True", "False"}:
                raise DataValidationError("zero-error CSV contains an invalid status flag")
            observed[benchmark] = {
                "attempted": int(row["attempted"]),
                "errors": int(row["errors"]),
                "zero_errors_observed": row["zero_errors_observed"] == "True",
                "confidence": float(row["confidence"]),
                "one_sided_upper": float(row["one_sided_upper"]),
            }
    except (KeyError, TypeError, ValueError) as exc:
        raise DataValidationError(f"zero-error CSV contains invalid values: {exc}") from exc
    if set(observed) != _OFFICIAL_GRADER_BENCHMARKS:
        raise DataValidationError("zero-error CSV benchmarks are incomplete")
    for benchmark, expected_value in results.zero_error_bounds.items():
        assert isinstance(expected_value, Mapping)
        value = observed[benchmark]
        if (
            value["attempted"] != expected_value["attempted"]
            or value["errors"] != expected_value["errors"]
            or value["zero_errors_observed"] is not expected_value["zero_errors_observed"]
            or not math.isclose(
                float(value["confidence"]),
                float(expected_value["confidence"]),
                rel_tol=0,
                abs_tol=1e-15,
            )
            or not math.isclose(
                float(value["one_sided_upper"]),
                float(expected_value["one_sided_upper"]),
                rel_tol=1e-15,
                abs_tol=1e-15,
            )
        ):
            raise DataValidationError(f"zero-error CSV differs from typed results for {benchmark}")


def _validate_confusion_csv(path: Path, results: FinalAnalysisResults) -> None:
    fields, rows = _tabular_rows(path)
    if fields != ("automated_label", "human_label", "count") or not rows:
        raise DataValidationError("confusion-matrix CSV has an invalid schema")
    observed: dict[str, int] = {}
    try:
        for row in rows:
            automated = row["automated_label"]
            human = row["human_label"]
            if automated not in _OUTCOME_LABELS or human not in _OUTCOME_LABELS:
                raise DataValidationError("confusion-matrix CSV contains an invalid label")
            key = f"{automated}:{human}"
            if key in observed:
                raise DataValidationError("confusion-matrix CSV repeats a label pair")
            observed[key] = int(row["count"])
            if observed[key] < 0:
                raise DataValidationError("confusion-matrix counts cannot be negative")
    except (KeyError, TypeError, ValueError) as exc:
        raise DataValidationError(f"confusion-matrix CSV contains invalid values: {exc}") from exc
    expected = results.human_audit["automated_human_confusion_matrix"]
    assert isinstance(expected, Mapping)
    if observed != dict(expected):
        raise DataValidationError("confusion-matrix CSV differs from typed results")


def _validate_adjudicated_csv(
    path: Path,
    results: FinalAnalysisResults,
    record_index: Mapping[tuple[str, str], GenerationRecord] | None = None,
) -> None:
    fields, rows = _tabular_rows(path)
    if fields != _AUDIT_COLUMNS:
        raise DataValidationError("adjudicated-label CSV has an invalid schema")
    summary = results.human_audit["adjudication_summary"]
    expected_confusion = results.human_audit["automated_human_confusion_matrix"]
    assert isinstance(summary, Mapping) and isinstance(expected_confusion, Mapping)
    if len(rows) != int(summary["rows"]):
        raise DataValidationError("adjudicated-label CSV row count differs from typed results")
    identifiers: set[str] = set()
    record_keys: set[tuple[str, str]] = set()
    queues: set[str] = set()
    combination_counts: dict[tuple[str, str], int] = {}
    confusion: dict[str, int] = {}
    disagreements = 0
    models = {"qwen3.6-27b-nvfp4"}
    for row in rows:
        identifier = row["audit_id"].strip()
        if not identifier or identifier in identifiers:
            raise DataValidationError("adjudicated-label CSV has duplicate or empty audit IDs")
        identifiers.add(identifier)
        question_id = row["question_id"].strip()
        condition_id = row["condition_id"].strip()
        response_sha256 = row["response_sha256"].strip()
        record_key = (condition_id, question_id)
        if (
            not question_id
            or not condition_id
            or not _SHA256.fullmatch(response_sha256)
            or record_key in record_keys
        ):
            raise DataValidationError(
                "adjudicated-label CSV has an invalid or repeated record binding"
            )
        record_keys.add(record_key)
        benchmark = row["benchmark"]
        model = row["model"]
        if benchmark not in _FACTUAL_BENCHMARKS or model not in models:
            raise DataValidationError("adjudicated-label CSV has an unknown benchmark or model")
        if any(not row[name].strip() for name in ("method", "prompt")):
            raise DataValidationError("adjudicated-label CSV has an empty condition identity")
        labels = tuple(
            row[name]
            for name in (
                "automated_label",
                "annotator_1_label",
                "annotator_2_label",
                "adjudicated_label",
            )
        )
        if any(label not in _OUTCOME_LABELS for label in labels):
            raise DataValidationError("adjudicated-label CSV contains an invalid outcome label")
        queue = row["queue"]
        if queue not in _AUDIT_QUEUES:
            raise DataValidationError("adjudicated-label CSV contains an unknown audit queue")
        queues.add(queue)
        combination = (benchmark, model)
        combination_counts[combination] = combination_counts.get(combination, 0) + 1
        if labels[1] != labels[2]:
            disagreements += 1
        key = f"{labels[0]}:{labels[3]}"
        confusion[key] = confusion.get(key, 0) + 1
        if record_index is not None:
            record = record_index.get(record_key)
            if (
                record is None
                or stable_hash(record.raw_output) != response_sha256
                or record.benchmark != benchmark
                or _MODEL_NAMES_BY_REPOSITORY.get(record.model_repository) != model
                or record.steering_method != row["method"]
                or record.system_prompt_id != row["prompt"]
                or record.outcome.value != row["automated_label"]
            ):
                raise DataValidationError(
                    "adjudicated-label CSV differs from its frozen generation record"
                )
    required_combinations = set(product(_FACTUAL_BENCHMARKS, models))
    if set(combination_counts) != required_combinations or any(
        value < 200 for value in combination_counts.values()
    ):
        raise DataValidationError(
            "adjudicated-label CSV lacks 200 rows per factual benchmark and model"
        )
    if not queues <= _AUDIT_QUEUES:
        raise DataValidationError("adjudicated-label CSV contains an unknown audit queue")
    if disagreements != int(summary["disagreements"]):
        raise DataValidationError("adjudicated-label disagreement count differs from results")
    if confusion != dict(expected_confusion):
        raise DataValidationError("adjudicated-label confusion counts differ from results")
    binding_payload = [
        {
            "condition_id": condition_id,
            "question_id": question_id,
            "response_sha256": next(
                row["response_sha256"]
                for row in rows
                if row["condition_id"] == condition_id and row["question_id"] == question_id
            ),
        }
        for condition_id, question_id in sorted(record_keys)
    ]
    if stable_hash(binding_payload) != results.human_audit["record_binding_digest"]:
        raise DataValidationError("adjudicated-label record bindings differ from typed results")


def _validate_svg_report(
    name: str,
    path: Path,
    results: FinalAnalysisResults,
) -> None:
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise DataValidationError(f"report artifact {name} is invalid SVG/XML: {exc}") from exc
    if root.tag.rsplit("}", 1)[-1] != "svg":
        raise DataValidationError(f"report artifact {name} has no SVG root")
    if any(element.tag.rsplit("}", 1)[-1] == "style" for element in root.iter()):
        raise DataValidationError(f"report artifact {name} cannot contain stylesheets")
    expected_chart_kind = _CHART_KINDS.get(name)
    if (
        expected_chart_kind is not None
        and root.attrib.get("data-chart-kind") != expected_chart_kind
    ):
        raise DataValidationError(f"report artifact {name} has the wrong chart semantics")

    try:
        canvas_width = float(root.attrib["width"])
        canvas_height = float(root.attrib["height"])
    except (KeyError, ValueError) as exc:
        raise DataValidationError(f"report artifact {name} has an invalid canvas") from exc

    def styles(element: ET.Element) -> dict[str, str]:
        style: dict[str, str] = {}
        for declaration in element.attrib.get("style", "").split(";"):
            if ":" in declaration:
                key, value = declaration.split(":", 1)
                style[key.strip().lower()] = value.strip().lower()
        return style

    def attribute(element: ET.Element, key: str, default: str = "") -> str:
        return styles(element).get(key, element.attrib.get(key, default)).strip().lower()

    def hidden(element: ET.Element, ancestor_hidden: bool) -> bool:
        style = styles(element)
        display = style.get("display", element.attrib.get("display", "")).lower()
        visibility = style.get("visibility", element.attrib.get("visibility", "")).lower()
        opacity_value = style.get("opacity", element.attrib.get("opacity", "1"))
        try:
            opacity = float(opacity_value)
        except ValueError:
            opacity = -1
        return (
            ancestor_hidden
            or display == "none"
            or visibility in {"hidden", "collapse"}
            or opacity <= 0
            or any(key in element.attrib for key in ("clip-path", "mask", "filter"))
        )

    def positive(value: str | None) -> bool:
        try:
            return value is not None and math.isfinite(float(value)) and float(value) > 0
        except ValueError:
            return False

    def finite_attribute(element: ET.Element, key: str, default: float = 0.0) -> float:
        try:
            value = float(element.attrib.get(key, str(default)))
        except ValueError:
            return math.nan
        return value if math.isfinite(value) else math.nan

    def painted(element: ET.Element, tag: str) -> bool:
        def opacity(key: str) -> float:
            try:
                return float(attribute(element, key, "1"))
            except ValueError:
                return -1.0

        fill_visible = attribute(element, "fill", "#000") != "none" and opacity("fill-opacity") > 0
        stroke_width = attribute(element, "stroke-width", "1")
        stroke_visible = (
            attribute(element, "stroke", "none") != "none" and opacity("stroke-opacity") > 0
        )
        try:
            stroke_visible = stroke_visible and float(stroke_width) > 0
        except ValueError:
            stroke_visible = False
        if tag == "text":
            return fill_visible and positive(attribute(element, "font-size", "16"))
        if tag in {"line", "path", "polyline"}:
            return stroke_visible
        return fill_visible or stroke_visible

    def on_canvas(element: ET.Element, tag: str) -> bool:
        if tag == "text":
            x = finite_attribute(element, "x")
            y = finite_attribute(element, "y")
            return 0 <= x <= canvas_width and 0 <= y <= canvas_height
        if tag == "rect":
            x = finite_attribute(element, "x")
            y = finite_attribute(element, "y")
            width = finite_attribute(element, "width")
            height = finite_attribute(element, "height")
            return x < canvas_width and y < canvas_height and x + width > 0 and y + height > 0
        if tag == "circle":
            x = finite_attribute(element, "cx")
            y = finite_attribute(element, "cy")
            radius = finite_attribute(element, "r")
            return (
                x + radius > 0
                and y + radius > 0
                and x - radius < canvas_width
                and y - radius < canvas_height
            )
        if tag == "line":
            values = [finite_attribute(element, key) for key in ("x1", "y1", "x2", "y2")]
            return all(math.isfinite(value) for value in values) and not (
                max(values[0], values[2]) < 0
                or min(values[0], values[2]) > canvas_width
                or max(values[1], values[3]) < 0
                or min(values[1], values[3]) > canvas_height
            )
        return True

    def visible_geometry(element: ET.Element, tag: str) -> bool:
        if tag == "rect":
            geometry = positive(element.attrib.get("width")) and positive(
                element.attrib.get("height")
            )
            return geometry and painted(element, tag) and on_canvas(element, tag)
        if tag == "circle":
            return (
                positive(element.attrib.get("r"))
                and painted(element, tag)
                and on_canvas(element, tag)
            )
        if tag == "line":
            geometry = (element.attrib.get("x1"), element.attrib.get("y1")) != (
                element.attrib.get("x2"),
                element.attrib.get("y2"),
            )
            return geometry and painted(element, tag) and on_canvas(element, tag)
        if tag == "path":
            return bool(element.attrib.get("d", "").strip()) and painted(element, tag)
        if tag in {"polyline", "polygon"}:
            return len(element.attrib.get("points", "").split()) >= 2 and painted(element, tag)
        if tag == "text":
            return (
                bool("".join(element.itertext()).strip())
                and painted(element, tag)
                and on_canvas(element, tag)
            )
        return False

    observed: dict[str, str] = {}
    observed_semantic: Counter[tuple[str, str]] = Counter()
    expected_values = _flatten_report_values(report_result_payload(name, results))
    mark_count = 0
    stack: list[tuple[ET.Element, bool]] = [(root, False)]
    mark_tags = {
        "path",
        "rect",
        "circle",
        "line",
        "polyline",
        "polygon",
        "text",
    }
    while stack:
        element, ancestor_hidden = stack.pop()
        is_hidden = hidden(element, ancestor_hidden)
        stack.extend((child, is_hidden) for child in reversed(list(element)))
        tag = element.tag.rsplit("}", 1)[-1]
        geometry_visible = tag in mark_tags and visible_geometry(element, tag)
        if geometry_visible and not is_hidden:
            mark_count += 1
        result_path = element.attrib.get("data-result-path")
        result_value = element.attrib.get("data-value")
        if result_path is not None or result_value is not None:
            if (
                result_path is None
                or result_value is None
                or result_path in observed
                or is_hidden
                or not geometry_visible
            ):
                raise DataValidationError(f"report artifact {name} has invalid result bindings")
            observed[result_path] = result_value
        semantic_class = element.attrib.get("data-semantic-mark")
        semantic_bindings = element.attrib.get("data-source-bindings")
        if semantic_class is None and semantic_bindings is None:
            continue
        if (
            semantic_class is None
            or semantic_bindings is None
            or is_hidden
            or not geometry_visible
            or semantic_class not in element.attrib.get("class", "").split()
        ):
            raise DataValidationError(f"report artifact {name} has an invalid semantic chart mark")
        try:
            binding_value = json.loads(semantic_bindings)
        except json.JSONDecodeError as exc:
            raise DataValidationError(
                f"report artifact {name} has invalid semantic bindings"
            ) from exc
        if (
            not isinstance(binding_value, Mapping)
            or not binding_value
            or any(
                not isinstance(path_key, str)
                or not isinstance(source_value, str)
                or expected_values.get(path_key) != source_value
                for path_key, source_value in binding_value.items()
            )
        ):
            raise DataValidationError(
                f"report artifact {name} has detached semantic chart bindings"
            )
        observed_semantic[(semantic_class, canonical_json(binding_value))] += 1
    expected_semantic = _expected_semantic_marks(name, results)
    if mark_count == 0 or observed != expected_values or observed_semantic != expected_semantic:
        raise DataValidationError(
            f"report artifact {name} does not encode its typed values and chart semantics"
        )
    expected_root = _build_svg_report_root(name=name, results=results)
    if _canonical_svg_tree(root) != _canonical_svg_tree(expected_root):
        raise DataValidationError(
            f"report artifact {name} geometry differs from its typed deterministic rendering"
        )


def _validate_report_format(
    name: str,
    path: Path,
    source_data_digest: str,
    results: FinalAnalysisResults,
    record_index: Mapping[tuple[str, str], GenerationRecord] | None = None,
) -> str:
    if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
        raise DataValidationError(f"report source must be a non-empty regular file: {path}")
    if not _SHA256.fullmatch(source_data_digest):
        raise DataValidationError("report source-data identity must be a SHA-256 fingerprint")
    suffix = path.suffix.lower()
    data = path.read_bytes()
    marker = f"mfh-source-data-sha256={source_data_digest}".encode()
    if marker not in data:
        raise DataValidationError(f"report artifact {name} is not bound to its source data")
    if name == "zero_error_confidence_bounds":
        if suffix != ".csv":
            raise DataValidationError("zero-error confidence bounds must be CSV")
        _validate_zero_error_csv(path, results)
    elif name == "adjudicated_final_labels":
        if suffix != ".csv":
            raise DataValidationError("adjudicated labels must be CSV")
        _validate_adjudicated_csv(path, results, record_index)
    elif name == "automated_human_confusion_matrix":
        if suffix != ".csv":
            raise DataValidationError("automated-human confusion matrix must be CSV")
        _validate_confusion_csv(path, results)
    elif suffix == ".svg":
        _validate_svg_report(name, path, results)
    else:
        raise DataValidationError(f"quantitative report artifact {name} must be inspectable SVG")
    return suffix


def report_result_payload(name: str, results: FinalAnalysisResults) -> dict[str, Any]:
    """Return the exact typed result sections that a named report must visualize."""

    dependencies = _REPORT_RESULT_DEPENDENCIES.get(name)
    if dependencies is None:
        raise DataValidationError(f"unknown report artifact {name!r}")
    sections = results.to_dict()
    return {
        "result_dependencies": {dependency: sections[dependency] for dependency in dependencies}
    }


def report_result_digest(name: str, results: FinalAnalysisResults) -> str:
    """Identity embedded into the rendered report by its generator."""

    return stable_hash(report_result_payload(name, results))


def _write_report_source(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        raise FrozenArtifactError(f"refusing to overwrite report source: {path}") from None


_SVG_NAMESPACE = "http://www.w3.org/2000/svg"
_CHART_KINDS = {
    "risk_coverage_curves": "risk-coverage-lines",
    "outcome_transition_diagrams": "outcome-transition-flow",
    "gold_vs_abstention_likelihood_changes": "likelihood-grouped-bars",
    "layer_alpha_heatmaps": "layer-alpha-heatmap",
    "static_vs_adaptive_matched_coverage": "matched-coverage-scatter",
    "dense_sparse_disentangled_pareto": "factuality-utility-pareto",
    "prompt_method_interaction_heatmaps": "prompt-method-heatmap",
    "prompt_paraphrase_robustness": "prompt-paraphrase-range",
    "safety_utility_noninferiority": "noninferiority-forest",
    "language_switching_confusion_matrices": "language-confusion-matrix",
    "local_vllm_runtime_validation": "runtime-validation-bars",
}


def _semantic_binding_values(bindings: Mapping[str, Any]) -> str:
    return canonical_json({path: canonical_json(value) for path, value in sorted(bindings.items())})


def _semantic_attributes(css_class: str, bindings: Mapping[str, Any]) -> dict[str, str]:
    if not bindings:
        raise DataValidationError("semantic chart marks require source bindings")
    return {
        "class": css_class,
        "data-semantic-mark": css_class,
        "data-source-bindings": _semantic_binding_values(bindings),
    }


def _mark_counter_key(css_class: str, bindings: Mapping[str, Any]) -> tuple[str, str]:
    return css_class, _semantic_binding_values(bindings)


def _expected_semantic_marks(name: str, results: FinalAnalysisResults) -> Counter[tuple[str, str]]:
    sections = results.to_dict()
    expected: Counter[tuple[str, str]] = Counter()

    def add(css_class: str, bindings: Mapping[str, Any]) -> None:
        expected[_mark_counter_key(css_class, bindings)] += 1

    if name == "risk_coverage_curves":
        for series, raw in sections["risk_coverage"].items():
            assert isinstance(raw, Mapping)
            prefix = f"result_dependencies.risk_coverage.{series}"
            add(
                "risk-coverage-line",
                {
                    f"{prefix}.aurc": raw["aurc"],
                    f"{prefix}.coverage_limit": raw["coverage_limit"],
                },
            )
            points = raw["points"]
            assert isinstance(points, Mapping)
            for point_key, point in points.items():
                assert isinstance(point, Mapping)
                add(
                    "risk-coverage-point",
                    {
                        f"{prefix}.points.{point_key}.coverage": point["coverage"],
                        f"{prefix}.points.{point_key}.risk": point["risk"],
                    },
                )
    elif name == "outcome_transition_diagrams":
        for comparison, raw in sections["transition_decomposition"].items():
            assert isinstance(raw, Mapping)
            transitions = raw["transition_counts"]
            assert isinstance(transitions, Mapping)
            for transition, count in transitions.items():
                add(
                    "transition-cell",
                    {
                        "result_dependencies.transition_decomposition."
                        f"{comparison}.transition_counts.{transition}": count
                    },
                )
    elif name == "gold_vs_abstention_likelihood_changes":
        for comparison, raw in sections["likelihood_changes"].items():
            assert isinstance(raw, Mapping)
            prefix = f"result_dependencies.likelihood_changes.{comparison}"
            add("gold-likelihood-change", {f"{prefix}.gold": raw["gold"]})
            add(
                "abstention-likelihood-change",
                {f"{prefix}.abstention": raw["abstention"]},
            )
    elif name == "layer_alpha_heatmaps":
        for cell_key, raw in sections["layer_alpha_surface"].items():
            assert isinstance(raw, Mapping)
            prefix = f"result_dependencies.layer_alpha_surface.{cell_key}"
            add(
                "layer-alpha-cell",
                {
                    f"{prefix}.layer": raw["layer"],
                    f"{prefix}.alpha": raw["alpha"],
                    f"{prefix}.risk": raw["risk"],
                },
            )
    elif name == "static_vs_adaptive_matched_coverage":
        for comparison, raw in sections["matched_coverage"].items():
            assert isinstance(raw, Mapping)
            prefix = f"result_dependencies.matched_coverage.{comparison}"
            add(
                "matched-coverage-point",
                {
                    f"{prefix}.baseline_coverage": raw["baseline_coverage"],
                    f"{prefix}.coverage": raw["coverage"],
                    f"{prefix}.risk_difference": raw["risk_difference"],
                },
            )
    elif name == "dense_sparse_disentangled_pareto":
        for comparison, raw in sections["factuality_side_effect_pareto"].items():
            assert isinstance(raw, Mapping)
            prefix = f"result_dependencies.factuality_side_effect_pareto.{comparison}"
            add(
                "pareto-point",
                {
                    f"{prefix}.risk": raw["risk"],
                    f"{prefix}.minimum_normalized_noninferiority_slack": raw[
                        "minimum_normalized_noninferiority_slack"
                    ],
                },
            )
    elif name == "prompt_method_interaction_heatmaps":
        for comparison, raw in sections["prompt_interactions"].items():
            assert isinstance(raw, Mapping)
            add(
                "prompt-interaction-cell",
                {f"result_dependencies.prompt_interactions.{comparison}.estimate": raw["estimate"]},
            )
        for effect, raw in sections["mixed_effects"].items():
            assert isinstance(raw, Mapping)
            add(
                "mixed-effect-cell",
                {f"result_dependencies.mixed_effects.{effect}.estimate": raw["estimate"]},
            )
    elif name == "prompt_paraphrase_robustness":
        for family, raw in sections["prompt_paraphrase"].items():
            assert isinstance(raw, Mapping)
            prefix = f"result_dependencies.prompt_paraphrase.{family}"
            add(
                "paraphrase-range",
                {
                    f"{prefix}.minimum_accuracy": raw["minimum_accuracy"],
                    f"{prefix}.maximum_accuracy": raw["maximum_accuracy"],
                },
            )
            add(
                "paraphrase-mean",
                {f"{prefix}.mean_accuracy": raw["mean_accuracy"]},
            )
    elif name == "safety_utility_noninferiority":
        for metric, raw in sections["noninferiority"].items():
            assert isinstance(raw, Mapping)
            prefix = f"result_dependencies.noninferiority.{metric}"
            add(
                "component-noninferiority-estimate",
                {
                    f"{prefix}.estimate": raw["estimate"],
                    f"{prefix}.one_sided_lower": raw["one_sided_lower"],
                    f"{prefix}.margin": raw["margin"],
                },
            )
        composite = sections["composite_side_effects"]["metrics"]
        assert isinstance(composite, Mapping)
        for metric, raw in composite.items():
            assert isinstance(raw, Mapping)
            prefix = f"result_dependencies.composite_side_effects.metrics.{metric}"
            add(
                "composite-noninferiority-estimate",
                {
                    f"{prefix}.estimate": raw["estimate"],
                    f"{prefix}.one_sided_lower": raw["one_sided_lower"],
                    f"{prefix}.margin": raw["margin"],
                },
            )
    elif name == "language_switching_confusion_matrices":
        matrix = sections["language_confusion"]["requested_detected_matrix"]
        assert isinstance(matrix, Mapping)
        for cell, count in matrix.items():
            add(
                "language-confusion-cell",
                {f"result_dependencies.language_confusion.requested_detected_matrix.{cell}": count},
            )
    elif name == "local_vllm_runtime_validation":
        runtime = sections["runtime_replication"]["local_vllm_execution"]
        assert isinstance(runtime, Mapping)
        prefix = "result_dependencies.runtime_replication.local_vllm_execution"
        for metric in (
            "mean_latency_seconds",
            "p95_latency_seconds",
            "maximum_latency_seconds",
        ):
            add("runtime-latency-bar", {f"{prefix}.{metric}": runtime[metric]})
        add(
            "runtime-candidate-latency-bar",
            {
                f"{prefix}.mean_candidate_generation_seconds": runtime[
                    "mean_candidate_generation_seconds"
                ]
            },
        )
        for metric in (
            "mean_prompt_tokens_per_second",
            "mean_generation_tokens_per_second",
        ):
            add("runtime-throughput-bar", {f"{prefix}.{metric}": runtime[metric]})
        add(
            "runtime-memory-bar",
            {
                f"{prefix}.maximum_peak_memory_bytes": runtime["maximum_peak_memory_bytes"],
                f"{prefix}.gpu_total_memory_bytes": runtime["gpu_total_memory_bytes"],
            },
        )
        add(
            "runtime-identity-label",
            {f"{prefix}.runtime_identity_sha256": runtime["runtime_identity_sha256"]},
        )
        sites = runtime["intervention_site_counts"]
        assert isinstance(sites, Mapping)
        for site, count in sites.items():
            add(
                "runtime-site-bar",
                {f"{prefix}.intervention_site_counts.{site}": count},
            )
    return expected


def _svg_element(
    parent: ET.Element,
    tag: str,
    attributes: Mapping[str, str] | None = None,
    *,
    text: str | None = None,
) -> ET.Element:
    element = ET.SubElement(
        parent,
        f"{{{_SVG_NAMESPACE}}}{tag}",
        dict(attributes or {}),
    )
    element.text = text
    return element


def _svg_text(
    parent: ET.Element,
    x: float,
    y: float,
    value: str,
    *,
    size: int = 12,
    css_class: str = "chart-label",
) -> ET.Element:
    return _svg_element(
        parent,
        "text",
        {
            "x": f"{x:.6g}",
            "y": f"{y:.6g}",
            "font-size": str(size),
            "class": css_class,
            "fill": "#17202a",
        },
        text=value,
    )


def _numeric(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float) and math.isfinite(float(value)):
        return float(value)
    return default


def _axes(root: ET.Element, *, x_label: str, y_label: str) -> None:
    group = _svg_element(root, "g", {"class": "chart-axes"})
    _svg_element(
        group,
        "line",
        {"x1": "90", "y1": "420", "x2": "1080", "y2": "420", "stroke": "#273746"},
    )
    _svg_element(
        group,
        "line",
        {"x1": "90", "y1": "420", "x2": "90", "y2": "70", "stroke": "#273746"},
    )
    _svg_text(group, 520, 452, x_label, size=13, css_class="x-axis-label")
    _svg_text(group, 10, 55, y_label, size=13, css_class="y-axis-label")


def _render_risk_coverage_chart(root: ET.Element, section: Mapping[str, Any]) -> None:
    _axes(root, x_label="Coverage", y_label="Hallucination risk")
    colors = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e")
    for index, (name, raw) in enumerate(sorted(section.items())):
        if not isinstance(raw, Mapping):
            continue
        points_value = raw.get("points")
        points: list[tuple[str, float, float]] = []
        if isinstance(points_value, Mapping):
            for point_key, point in points_value.items():
                if isinstance(point, Mapping):
                    points.append(
                        (
                            str(point_key),
                            _number(point["coverage"], "risk coverage point"),
                            _number(point["risk"], "risk coverage point"),
                        )
                    )
        if not points:
            raise DataValidationError("risk-coverage chart requires observed curve points")
        points.sort(key=lambda value: (value[1], value[2], value[0]))
        coordinates = " ".join(
            f"{90 + 990 * min(1.0, max(0.0, x)):.3f},{420 - 350 * min(1.0, max(0.0, y)):.3f}"
            for _, x, y in points
        )
        group = _svg_element(root, "g", {"class": "risk-coverage-series"})
        prefix = f"result_dependencies.risk_coverage.{name}"
        _svg_element(
            group,
            "polyline",
            {
                "points": coordinates,
                "fill": "none",
                "stroke": colors[index % len(colors)],
                "stroke-width": "3",
                **_semantic_attributes(
                    "risk-coverage-line",
                    {
                        f"{prefix}.aurc": raw["aurc"],
                        f"{prefix}.coverage_limit": raw["coverage_limit"],
                    },
                ),
            },
        )
        for point_key, coverage, risk in points:
            _svg_element(
                group,
                "circle",
                {
                    "cx": f"{90 + 990 * coverage:.3f}",
                    "cy": f"{420 - 350 * risk:.3f}",
                    "r": "4",
                    "fill": colors[index % len(colors)],
                    **_semantic_attributes(
                        "risk-coverage-point",
                        {
                            f"{prefix}.points.{point_key}.coverage": coverage,
                            f"{prefix}.points.{point_key}.risk": risk,
                        },
                    ),
                },
            )
        _svg_text(root, 845, 86 + 18 * index, name[:42], size=10, css_class="series-label")


def _render_transition_chart(root: ET.Element, section: Mapping[str, Any]) -> None:
    _svg_text(root, 30, 35, "Full paired C/P/I/A transition matrices", size=14)
    count = max(1, len(section))
    columns = min(8, max(1, math.ceil(math.sqrt(count * 2))))
    rows = math.ceil(count / columns)
    block_width = 1080 / columns
    block_height = 420 / rows
    labels = ("C", "P", "I", "A")
    for index, (name, raw) in enumerate(sorted(section.items())):
        if not isinstance(raw, Mapping):
            continue
        transitions = raw["transition_counts"]
        if not isinstance(transitions, Mapping) or set(transitions) != {
            f"{left}:{right}" for left in labels for right in labels
        }:
            raise DataValidationError("transition chart requires a full C/P/I/A matrix")
        total = max(1, sum(int(value) for value in transitions.values()))
        block_x = 55 + (index % columns) * block_width
        block_y = 58 + (index // columns) * block_height
        cell_size = max(2.0, min((block_width - 24) / 4, (block_height - 34) / 4))
        for row, left in enumerate(labels):
            for column, right in enumerate(labels):
                transition = f"{left}:{right}"
                count_value = transitions[transition]
                scaled = int(count_value) / total
                red = int(245 - 130 * scaled)
                green = int(245 - 45 * scaled)
                blue = int(255 - 35 * scaled)
                prefix = (
                    "result_dependencies.transition_decomposition."
                    f"{name}.transition_counts.{transition}"
                )
                rectangle = _svg_element(
                    root,
                    "rect",
                    {
                        "x": f"{block_x + column * cell_size:.3f}",
                        "y": f"{block_y + row * cell_size:.3f}",
                        "width": f"{max(1.0, cell_size - 1):.3f}",
                        "height": f"{max(1.0, cell_size - 1):.3f}",
                        "fill": f"rgb({red},{green},{blue})",
                        **_semantic_attributes("transition-cell", {prefix: count_value}),
                    },
                )
                _svg_element(
                    rectangle,
                    "title",
                    text=f"{name}: {transition} = {int(count_value)}",
                )
        _svg_text(
            root,
            block_x,
            min(488.0, block_y + 4 * cell_size + 12),
            f"{index + 1}: {name[:24]}",
            size=7,
            css_class="transition-matrix-label",
        )


def _render_likelihood_chart(root: ET.Element, section: Mapping[str, Any]) -> None:
    values = [
        abs(_number(raw[key], f"likelihood_changes.{name}.{key}"))
        for name, raw in section.items()
        if isinstance(raw, Mapping)
        for key in ("gold", "abstention")
    ]
    maximum = max(values, default=1.0) or 1.0
    _axes(root, x_label="Comparison", y_label="Mean log-likelihood change")
    width = max(12.0, 820 / max(1, len(section)))
    for index, (name, raw) in enumerate(sorted(section.items())):
        if not isinstance(raw, Mapping):
            continue
        x = 110 + index * width
        prefix = f"result_dependencies.likelihood_changes.{name}"
        for offset, key, color, css_class in (
            (0.0, "gold", "#1f77b4", "gold-likelihood-change"),
            (width * 0.4, "abstention", "#e67e22", "abstention-likelihood-change"),
        ):
            value = _number(raw[key], f"likelihood_changes.{name}.{key}")
            height = 300 * abs(value) / maximum
            y = 245 - height if value >= 0 else 245
            _svg_element(
                root,
                "rect",
                {
                    "x": f"{x + offset:.3f}",
                    "y": f"{y:.3f}",
                    "width": f"{max(5.0, width * 0.34):.3f}",
                    "height": f"{max(1.0, height):.3f}",
                    "fill": color,
                    **_semantic_attributes(css_class, {f"{prefix}.{key}": raw[key]}),
                },
            )
        _svg_text(root, 845, 86 + 14 * index, f"{index + 1}: {name[:38]}", size=9)
    _svg_element(
        root,
        "line",
        {"x1": "90", "y1": "245", "x2": "1080", "y2": "245", "stroke": "#7f8c8d"},
    )


def _render_grouped_bar_chart(
    root: ET.Element,
    section: Mapping[str, Any],
    *,
    left_key: str,
    right_key: str,
    left_class: str,
    right_class: str,
) -> None:
    values = [
        abs(_numeric(raw.get(key)))
        for raw in section.values()
        if isinstance(raw, Mapping)
        for key in (left_key, right_key)
    ]
    maximum = max(values, default=1.0) or 1.0
    _axes(root, x_label="Comparison", y_label="Change")
    width = max(12.0, 820 / max(1, len(section)))
    for index, (name, raw) in enumerate(sorted(section.items())):
        if not isinstance(raw, Mapping):
            continue
        x = 110 + index * width
        for offset, key, color, css_class in (
            (0.0, left_key, "#1f77b4", left_class),
            (width * 0.4, right_key, "#e67e22", right_class),
        ):
            value = _numeric(raw.get(key))
            height = 300 * abs(value) / maximum
            y = 245 - height if value >= 0 else 245
            _svg_element(
                root,
                "rect",
                {
                    "x": f"{x + offset:.3f}",
                    "y": f"{y:.3f}",
                    "width": f"{max(5.0, width * 0.34):.3f}",
                    "height": f"{max(1.0, height):.3f}",
                    "fill": color,
                    "class": css_class,
                },
            )
        _svg_text(root, 845, 86 + 14 * index, f"{index + 1}: {name[:38]}", size=9)
    _svg_element(
        root,
        "line",
        {"x1": "90", "y1": "245", "x2": "1080", "y2": "245", "stroke": "#7f8c8d"},
    )


def _render_heatmap(
    root: ET.Element,
    section: Mapping[str, Any],
    *,
    value_key: str,
    css_class: str,
) -> None:
    entries = [(name, raw) for name, raw in sorted(section.items()) if isinstance(raw, Mapping)]
    columns = max(1, min(12, math.ceil(math.sqrt(max(1, len(entries))))))
    rows = max(1, math.ceil(len(entries) / columns))
    cell_width = 850 / columns
    cell_height = 330 / rows
    values = [_numeric(raw.get(value_key)) for _, raw in entries]
    low = min(values, default=0.0)
    high = max(values, default=1.0)
    span = high - low or 1.0
    for index, (name, raw) in enumerate(entries):
        value = _numeric(raw.get(value_key))
        scaled = min(1.0, max(0.0, (value - low) / span))
        red = int(245 - 115 * scaled)
        green = int(245 - 30 * scaled)
        blue = int(255 - 25 * scaled)
        x = 100 + (index % columns) * cell_width
        y = 70 + (index // columns) * cell_height
        cell = _svg_element(
            root,
            "rect",
            {
                "x": f"{x:.3f}",
                "y": f"{y:.3f}",
                "width": f"{max(1.0, cell_width - 3):.3f}",
                "height": f"{max(1.0, cell_height - 3):.3f}",
                "fill": f"rgb({red},{green},{blue})",
                "class": css_class,
            },
        )
        _svg_element(cell, "title", text=f"{name}: {value:.6g}")
    _svg_text(root, 975, 90, f"min {low:.4g}", size=10)
    _svg_text(root, 975, 110, f"max {high:.4g}", size=10)


def _surface_facet(cell_key: str) -> str:
    marker = "|layer_"
    return cell_key.split(marker, 1)[0] if marker in cell_key else cell_key


def _render_layer_alpha_chart(root: ET.Element, section: Mapping[str, Any]) -> None:
    entries = [(name, raw) for name, raw in sorted(section.items()) if isinstance(raw, Mapping)]
    if not entries:
        raise DataValidationError("layer-alpha chart requires observed cells")
    facets: dict[str, list[tuple[str, Mapping[str, Any]]]] = defaultdict(list)
    for name, raw in entries:
        facets[_surface_facet(name)].append((name, raw))
    facet_items = sorted(facets.items())
    columns = min(4, max(1, math.ceil(math.sqrt(len(facet_items)))))
    rows = math.ceil(len(facet_items) / columns)
    panel_width = 1040 / columns
    panel_height = 405 / rows
    risks = [_number(raw["risk"], f"layer_alpha_surface.{name}.risk") for name, raw in entries]
    low, high = min(risks), max(risks)
    span = high - low or 1.0
    _svg_text(
        root,
        25,
        28,
        "Layer x alpha risk surfaces (faceted by intervention design)",
        size=14,
    )
    for facet_index, (facet, cells) in enumerate(facet_items):
        panel_x = 55 + (facet_index % columns) * panel_width
        panel_y = 48 + (facet_index // columns) * panel_height
        layers = sorted({int(raw["layer"]) for _, raw in cells})
        alphas = sorted({_number(raw["alpha"], "layer alpha") for _, raw in cells})
        grid_x = panel_x + 24
        grid_y = panel_y + 9
        grid_width = max(1.0, panel_width - 34)
        grid_height = max(1.0, panel_height - 38)
        cell_width = max(1.0, grid_width / max(1, len(alphas)))
        cell_height = max(1.0, grid_height / max(1, len(layers)))
        for layer_index, layer_tick in enumerate(layers):
            _svg_text(
                root,
                panel_x,
                grid_y + (layer_index + 0.65) * cell_height,
                f"L{layer_tick}",
                size=6,
                css_class="layer-axis-tick",
            )
        for alpha_index, alpha_tick in enumerate(alphas):
            _svg_text(
                root,
                grid_x + (alpha_index + 0.1) * cell_width,
                grid_y + grid_height + 8,
                f"a={alpha_tick:.3g}",
                size=6,
                css_class="alpha-axis-tick",
            )
        for name, raw in cells:
            layer = int(raw["layer"])
            alpha = _number(raw["alpha"], f"layer_alpha_surface.{name}.alpha")
            risk = _number(raw["risk"], f"layer_alpha_surface.{name}.risk")
            scaled = min(1.0, max(0.0, (risk - low) / span))
            red = int(245 - 115 * scaled)
            green = int(245 - 30 * scaled)
            blue = int(255 - 25 * scaled)
            prefix = f"result_dependencies.layer_alpha_surface.{name}"
            rectangle = _svg_element(
                root,
                "rect",
                {
                    "x": f"{grid_x + alphas.index(alpha) * cell_width:.3f}",
                    "y": f"{grid_y + layers.index(layer) * cell_height:.3f}",
                    "width": f"{max(1.0, cell_width - 1):.3f}",
                    "height": f"{max(1.0, cell_height - 1):.3f}",
                    "fill": f"rgb({red},{green},{blue})",
                    **_semantic_attributes(
                        "layer-alpha-cell",
                        {
                            f"{prefix}.layer": raw["layer"],
                            f"{prefix}.alpha": raw["alpha"],
                            f"{prefix}.risk": raw["risk"],
                        },
                    ),
                },
            )
            _svg_element(
                rectangle,
                "title",
                text=f"{facet}; layer={layer}; alpha={alpha:.5g}; risk={risk:.5g}",
            )
        _svg_text(
            root,
            panel_x,
            min(492.0, panel_y + panel_height - 5),
            f"facet {facet_index + 1}: {facet[:24]}",
            size=7,
            css_class="surface-facet-label",
        )
    _svg_text(root, 900, 28, f"risk min={low:.4g}; max={high:.4g}", size=9)


def _render_prompt_interaction_chart(
    root: ET.Element,
    prompt_interactions: Mapping[str, Any],
    mixed_effects: Mapping[str, Any],
) -> None:
    groups = (
        ("Prompt x method estimates", prompt_interactions, "prompt-interaction-cell"),
        ("Registered mixed effects", mixed_effects, "mixed-effect-cell"),
    )
    all_values = [
        _number(raw["estimate"], f"{title}.{name}.estimate")
        for title, section, _ in groups
        for name, raw in section.items()
        if isinstance(raw, Mapping)
    ]
    maximum = max((abs(value) for value in all_values), default=1.0) or 1.0
    for panel_index, (title, section, css_class) in enumerate(groups):
        entries = [(name, raw) for name, raw in sorted(section.items()) if isinstance(raw, Mapping)]
        panel_x = 55 + panel_index * 555
        columns = max(1, min(8, math.ceil(math.sqrt(max(1, len(entries))))))
        rows = max(1, math.ceil(len(entries) / columns))
        cell_width = 500 / columns
        cell_height = 320 / rows
        _svg_text(root, panel_x, 42, title, size=13)
        section_name = "prompt_interactions" if panel_index == 0 else "mixed_effects"
        for index, (name, raw) in enumerate(entries):
            estimate = _number(raw["estimate"], f"{section_name}.{name}.estimate")
            intensity = abs(estimate) / maximum
            color = (
                f"rgb({int(245 - 70 * intensity)},245,{int(245 - 110 * intensity)})"
                if estimate >= 0
                else f"rgb(245,{int(245 - 85 * intensity)},{int(245 - 65 * intensity)})"
            )
            path = f"result_dependencies.{section_name}.{name}.estimate"
            rectangle = _svg_element(
                root,
                "rect",
                {
                    "x": f"{panel_x + (index % columns) * cell_width:.3f}",
                    "y": f"{65 + (index // columns) * cell_height:.3f}",
                    "width": f"{max(1.0, cell_width - 2):.3f}",
                    "height": f"{max(1.0, cell_height - 2):.3f}",
                    "fill": color,
                    **_semantic_attributes(css_class, {path: raw["estimate"]}),
                },
            )
            _svg_element(rectangle, "title", text=f"{name}: {estimate:.6g}")


def _render_scatter(
    root: ET.Element,
    section: Mapping[str, Any],
    *,
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    css_class: str,
) -> None:
    _axes(root, x_label=x_label, y_label=y_label)
    entries = [(name, raw) for name, raw in sorted(section.items()) if isinstance(raw, Mapping)]
    x_values = [_numeric(raw.get(x_key)) for _, raw in entries]
    y_values = [_numeric(raw.get(y_key)) for _, raw in entries]
    x_low, x_high = min(x_values, default=0), max(x_values, default=1)
    y_low, y_high = min(y_values, default=0), max(y_values, default=1)
    x_span, y_span = x_high - x_low or 1.0, y_high - y_low or 1.0
    for index, ((name, _), x_value, y_value) in enumerate(
        zip(entries, x_values, y_values, strict=True)
    ):
        x = 110 + 930 * (x_value - x_low) / x_span
        y = 400 - 310 * (y_value - y_low) / y_span
        point = _svg_element(
            root,
            "circle",
            {
                "cx": f"{x:.3f}",
                "cy": f"{y:.3f}",
                "r": "7",
                "fill": "#2e86c1",
                "class": css_class,
            },
        )
        _svg_element(point, "title", text=f"{name}: ({x_value:.5g}, {y_value:.5g})")
        _svg_text(root, 845, 86 + 14 * index, f"{index + 1}: {name[:38]}", size=9)


def _render_matched_coverage_chart(root: ET.Element, section: Mapping[str, Any]) -> None:
    _axes(root, x_label="Baseline coverage", y_label="Treatment coverage")
    _svg_element(
        root,
        "line",
        {
            "x1": "90",
            "y1": "420",
            "x2": "1080",
            "y2": "70",
            "stroke": "#7f8c8d",
            "stroke-dasharray": "6 5",
            "class": "matched-coverage-equality-line",
        },
    )
    for index, (name, raw) in enumerate(sorted(section.items())):
        if not isinstance(raw, Mapping):
            continue
        baseline = _probability(
            raw["baseline_coverage"], f"matched_coverage.{name}.baseline_coverage"
        )
        treatment = _probability(raw["coverage"], f"matched_coverage.{name}.coverage")
        risk_difference = _number(
            raw["risk_difference"], f"matched_coverage.{name}.risk_difference"
        )
        prefix = f"result_dependencies.matched_coverage.{name}"
        point = _svg_element(
            root,
            "circle",
            {
                "cx": f"{90 + 990 * baseline:.3f}",
                "cy": f"{420 - 350 * treatment:.3f}",
                "r": f"{5 + min(5.0, 40 * abs(risk_difference)):.3f}",
                "fill": "#239b56" if risk_difference <= 0 else "#c0392b",
                **_semantic_attributes(
                    "matched-coverage-point",
                    {
                        f"{prefix}.baseline_coverage": raw["baseline_coverage"],
                        f"{prefix}.coverage": raw["coverage"],
                        f"{prefix}.risk_difference": raw["risk_difference"],
                    },
                ),
            },
        )
        _svg_element(
            point,
            "title",
            text=(
                f"{name}: baseline={baseline:.4g}; treatment={treatment:.4g}; "
                f"risk delta={risk_difference:.4g}"
            ),
        )
        _svg_text(root, 845, 86 + 14 * index, f"{index + 1}: {name[:38]}", size=9)


def _render_pareto_chart(root: ET.Element, section: Mapping[str, Any]) -> None:
    _axes(
        root,
        x_label="Hallucination risk",
        y_label="Minimum normalized noninferiority slack",
    )
    entries = [(name, raw) for name, raw in sorted(section.items()) if isinstance(raw, Mapping)]
    risks = [_number(raw["risk"], f"pareto.{name}.risk") for name, raw in entries]
    slacks = [
        _number(
            raw["minimum_normalized_noninferiority_slack"],
            f"pareto.{name}.minimum_normalized_noninferiority_slack",
        )
        for name, raw in entries
    ]
    risk_low, risk_high = min(risks, default=0.0), max(risks, default=1.0)
    slack_low, slack_high = min(slacks, default=-1.0), max(slacks, default=1.0)
    risk_span = risk_high - risk_low or 1.0
    slack_span = slack_high - slack_low or 1.0
    if slack_low <= 0 <= slack_high:
        zero_y = 420 - 350 * (0 - slack_low) / slack_span
        _svg_element(
            root,
            "line",
            {
                "x1": "90",
                "y1": f"{zero_y:.3f}",
                "x2": "1080",
                "y2": f"{zero_y:.3f}",
                "stroke": "#c0392b",
                "stroke-dasharray": "5 4",
                "class": "noninferiority-boundary",
            },
        )
    for index, ((name, raw), risk, slack) in enumerate(zip(entries, risks, slacks, strict=True)):
        prefix = f"result_dependencies.factuality_side_effect_pareto.{name}"
        point = _svg_element(
            root,
            "circle",
            {
                "cx": f"{110 + 930 * (risk - risk_low) / risk_span:.3f}",
                "cy": f"{400 - 310 * (slack - slack_low) / slack_span:.3f}",
                "r": "7",
                "fill": "#239b56" if slack >= 0 else "#c0392b",
                **_semantic_attributes(
                    "pareto-point",
                    {
                        f"{prefix}.risk": raw["risk"],
                        f"{prefix}.minimum_normalized_noninferiority_slack": raw[
                            "minimum_normalized_noninferiority_slack"
                        ],
                    },
                ),
            },
        )
        _svg_element(point, "title", text=f"{name}: risk={risk:.5g}; slack={slack:.5g}")
        _svg_text(root, 845, 86 + 14 * index, f"{index + 1}: {name[:38]}", size=9)


def _render_paraphrase_chart(root: ET.Element, section: Mapping[str, Any]) -> None:
    _axes(root, x_label="Prompt family", y_label="Accuracy range")
    count = max(1, len(section))
    width = 850 / count
    for index, (name, raw) in enumerate(sorted(section.items())):
        if not isinstance(raw, Mapping):
            continue
        mean = _probability(raw["mean_accuracy"], f"prompt_paraphrase.{name}.mean_accuracy")
        low = _probability(raw["minimum_accuracy"], f"prompt_paraphrase.{name}.minimum_accuracy")
        high = _probability(raw["maximum_accuracy"], f"prompt_paraphrase.{name}.maximum_accuracy")
        if not low <= mean <= high:
            raise DataValidationError("prompt paraphrase range does not contain its mean")
        x = 125 + index * width
        prefix = f"result_dependencies.prompt_paraphrase.{name}"
        _svg_element(
            root,
            "line",
            {
                "x1": f"{x:.3f}",
                "y1": f"{420 - 330 * low:.3f}",
                "x2": f"{x:.3f}",
                "y2": f"{420 - 330 * high:.3f}",
                "stroke": "#7d3c98",
                "stroke-width": "5",
                **_semantic_attributes(
                    "paraphrase-range",
                    {
                        f"{prefix}.minimum_accuracy": raw["minimum_accuracy"],
                        f"{prefix}.maximum_accuracy": raw["maximum_accuracy"],
                    },
                ),
            },
        )
        _svg_element(
            root,
            "circle",
            {
                "cx": f"{x:.3f}",
                "cy": f"{420 - 330 * mean:.3f}",
                "r": "6",
                "fill": "#7d3c98",
                **_semantic_attributes(
                    "paraphrase-mean", {f"{prefix}.mean_accuracy": raw["mean_accuracy"]}
                ),
            },
        )
        _svg_text(root, 845, 86 + 14 * index, f"{index + 1}: {name[:38]}", size=9)


def _render_noninferiority_chart(
    root: ET.Element,
    component_section: Mapping[str, Any],
    composite_section: Mapping[str, Any],
) -> None:
    _axes(root, x_label="Oriented difference", y_label="Metric")
    entries = [
        ("component", name, raw)
        for name, raw in sorted(component_section.items())
        if isinstance(raw, Mapping)
    ] + [
        ("composite", name, raw)
        for name, raw in sorted(composite_section.items())
        if isinstance(raw, Mapping)
    ]
    values = [
        value
        for _, _, raw in entries
        for value in (
            _number(raw["estimate"], "noninferiority estimate"),
            _number(raw["one_sided_lower"], "noninferiority lower bound"),
            -abs(_number(raw["margin"], "noninferiority margin")),
        )
    ]
    low, high = min(values, default=-1), max(values, default=1)
    span = high - low or 1.0
    for index, (family, name, raw) in enumerate(entries):
        y = 90 + index * (300 / max(1, len(entries) - 1))
        estimate = _number(raw["estimate"], "noninferiority estimate")
        lower = _number(raw["one_sided_lower"], "noninferiority lower bound")
        margin = -abs(_number(raw["margin"], "noninferiority margin"))
        section_name = (
            "noninferiority" if family == "component" else "composite_side_effects.metrics"
        )
        css_class = (
            "component-noninferiority-estimate"
            if family == "component"
            else "composite-noninferiority-estimate"
        )
        prefix = f"result_dependencies.{section_name}.{name}"

        def scale(value: float) -> float:
            return 110 + 900 * (value - low) / span

        _svg_element(
            root,
            "line",
            {
                "x1": f"{scale(lower):.3f}",
                "y1": f"{y:.3f}",
                "x2": f"{scale(estimate):.3f}",
                "y2": f"{y:.3f}",
                "stroke": "#2c3e50",
                "stroke-width": "3",
                "class": "noninferiority-interval",
            },
        )
        _svg_element(
            root,
            "circle",
            {
                "cx": f"{scale(estimate):.3f}",
                "cy": f"{y:.3f}",
                "r": "6",
                "fill": "#239b56" if bool(raw.get("passed")) else "#c0392b",
                **_semantic_attributes(
                    css_class,
                    {
                        f"{prefix}.estimate": raw["estimate"],
                        f"{prefix}.one_sided_lower": raw["one_sided_lower"],
                        f"{prefix}.margin": raw["margin"],
                    },
                ),
            },
        )
        _svg_element(
            root,
            "line",
            {
                "x1": f"{scale(margin):.3f}",
                "y1": f"{y - 10:.3f}",
                "x2": f"{scale(margin):.3f}",
                "y2": f"{y + 10:.3f}",
                "stroke": "#c0392b",
                "class": "noninferiority-margin",
            },
        )
        _svg_text(root, 92, y - 9, f"{family}: {name}", size=8)


def _render_language_chart(root: ET.Element, section: Mapping[str, Any]) -> None:
    matrix = section["requested_detected_matrix"]
    if not isinstance(matrix, Mapping):
        raise DataValidationError("language chart requires a requested/detected matrix")
    entries = [(key, value) for key, value in sorted(matrix.items())]
    labels = sorted(
        {
            item
            for key, _ in entries
            if isinstance(key, str) and ":" in key
            for item in key.split(":", 1)
        }
    )
    if not labels:
        raise DataValidationError("language chart requires observed language labels")
    maximum = max((_numeric(value) for _, value in entries), default=1.0) or 1.0
    cell = min(70.0, 320 / len(labels))
    entry_map = dict(entries)
    for row, requested in enumerate(labels):
        _svg_text(root, 20, 105 + row * cell, requested, size=10)
        _svg_text(root, 125 + row * cell, 55, requested, size=10)
        for column, detected in enumerate(labels):
            cell_key = f"{requested}:{detected}"
            if cell_key not in entry_map:
                continue
            count_value = entry_map[cell_key]
            count = _number(count_value, f"language_confusion.{cell_key}")
            opacity = 0.08 + 0.92 * count / maximum
            rectangle = _svg_element(
                root,
                "rect",
                {
                    "x": f"{110 + column * cell:.3f}",
                    "y": f"{70 + row * cell:.3f}",
                    "width": f"{cell - 2:.3f}",
                    "height": f"{cell - 2:.3f}",
                    "fill": "#1f618d",
                    "fill-opacity": f"{opacity:.4f}",
                    **_semantic_attributes(
                        "language-confusion-cell",
                        {
                            "result_dependencies.language_confusion."
                            f"requested_detected_matrix.{cell_key}": count_value
                        },
                    ),
                },
            )
            _svg_element(rectangle, "title", text=f"{requested} → {detected}: {int(count)}")
    _svg_text(root, 110, 430, "Requested language (rows) x detected language (columns)", size=12)


def _render_runtime_chart(root: ET.Element, section: Mapping[str, Any]) -> None:
    runtime = section["local_vllm_execution"]
    if not isinstance(runtime, Mapping):
        raise DataValidationError("runtime chart requires local VLLM evidence")
    prefix = "result_dependencies.runtime_replication.local_vllm_execution"
    latency_metrics = (
        ("mean latency", "mean_latency_seconds"),
        ("p95 latency", "p95_latency_seconds"),
        ("max latency", "maximum_latency_seconds"),
    )
    latency_values = [_number(runtime[key], f"runtime.{key}") for _, key in latency_metrics]
    latency_max = max(latency_values) or 1.0
    _svg_text(root, 45, 35, "End-to-end latency (seconds)", size=13)
    for index, ((label, key), value) in enumerate(
        zip(latency_metrics, latency_values, strict=True)
    ):
        y = 58 + index * 45
        _svg_text(root, 45, y + 17, label, size=10)
        _svg_element(
            root,
            "rect",
            {
                "x": "175",
                "y": f"{y:.3f}",
                "width": f"{max(2.0, 330 * value / latency_max):.3f}",
                "height": "23",
                "fill": "#2874a6",
                **_semantic_attributes("runtime-latency-bar", {f"{prefix}.{key}": runtime[key]}),
            },
        )
        _svg_text(root, 515, y + 17, f"{value:.5g}", size=9)

    candidate_latency = _number(
        runtime["mean_candidate_generation_seconds"],
        "runtime.mean_candidate_generation_seconds",
    )
    _svg_text(root, 45, 198, "mean candidate generation", size=10)
    _svg_element(
        root,
        "rect",
        {
            "x": "175",
            "y": "181",
            "width": (
                f"{max(2.0, 330 * candidate_latency / max(latency_max, candidate_latency)):.3f}"
            ),
            "height": "23",
            "fill": "#7d3c98",
            **_semantic_attributes(
                "runtime-candidate-latency-bar",
                {
                    f"{prefix}.mean_candidate_generation_seconds": runtime[
                        "mean_candidate_generation_seconds"
                    ]
                },
            ),
        },
    )
    _svg_text(root, 515, 198, f"{candidate_latency:.5g}", size=9)

    throughput_metrics = (
        ("prompt tok/s", "mean_prompt_tokens_per_second"),
        ("generation tok/s", "mean_generation_tokens_per_second"),
    )
    throughput_values = [_number(runtime[key], f"runtime.{key}") for _, key in throughput_metrics]
    throughput_max = max(throughput_values) or 1.0
    _svg_text(root, 45, 225, "Throughput (generated rows; tokens/second)", size=13)
    for index, ((label, key), value) in enumerate(
        zip(throughput_metrics, throughput_values, strict=True)
    ):
        y = 245 + index * 45
        _svg_text(root, 45, y + 17, label, size=10)
        _svg_element(
            root,
            "rect",
            {
                "x": "175",
                "y": f"{y:.3f}",
                "width": f"{max(2.0, 330 * value / throughput_max):.3f}",
                "height": "23",
                "fill": "#7d3c98",
                **_semantic_attributes("runtime-throughput-bar", {f"{prefix}.{key}": runtime[key]}),
            },
        )
        _svg_text(root, 515, y + 17, f"{value:.5g}", size=9)

    peak = _number(runtime["maximum_peak_memory_bytes"], "runtime peak memory")
    available = _number(runtime["gpu_total_memory_bytes"], "runtime GPU memory")
    _svg_text(root, 45, 360, "Peak GPU-memory use", size=13)
    _svg_element(
        root,
        "rect",
        {
            "x": "175",
            "y": "375",
            "width": f"{max(2.0, 330 * min(1.0, peak / available)):.3f}",
            "height": "25",
            "fill": "#d35400",
            **_semantic_attributes(
                "runtime-memory-bar",
                {
                    f"{prefix}.maximum_peak_memory_bytes": runtime["maximum_peak_memory_bytes"],
                    f"{prefix}.gpu_total_memory_bytes": runtime["gpu_total_memory_bytes"],
                },
            ),
        },
    )
    _svg_text(root, 515, 393, f"{peak / 2**30:.3g} / {available / 2**30:.3g} GiB", size=9)

    identity = str(runtime["runtime_identity_sha256"])
    identity_label = _svg_text(
        root,
        620,
        42,
        f"Attested runtime: {identity}",
        size=9,
        css_class="runtime-identity-label",
    )
    identity_label.attrib.update(
        _semantic_attributes(
            "runtime-identity-label", {f"{prefix}.runtime_identity_sha256": identity}
        )
    )
    sites = runtime["intervention_site_counts"]
    if not isinstance(sites, Mapping):
        raise DataValidationError("runtime chart requires intervention-site evidence")
    maximum_site_count = max((int(value) for value in sites.values()), default=1) or 1
    _svg_text(root, 620, 75, "Executed intervention sites (records)", size=13)
    for index, (site, count_value) in enumerate(sorted(sites.items())):
        count = _integer(count_value, f"runtime.intervention_site_counts.{site}")
        y = 92 + index * 31
        _svg_text(root, 620, y + 15, str(site), size=9)
        _svg_element(
            root,
            "rect",
            {
                "x": "780",
                "y": f"{y:.3f}",
                "width": f"{max(2.0, 280 * count / maximum_site_count):.3f}",
                "height": "20",
                "fill": "#239b56",
                **_semantic_attributes(
                    "runtime-site-bar",
                    {f"{prefix}.intervention_site_counts.{site}": count_value},
                ),
            },
        )
        _svg_text(root, 1070, y + 15, str(count), size=9)


def _render_semantic_chart(
    root: ET.Element,
    *,
    name: str,
    results: FinalAnalysisResults,
) -> None:
    sections = results.to_dict()
    if name == "risk_coverage_curves":
        _render_risk_coverage_chart(root, sections["risk_coverage"])
    elif name == "outcome_transition_diagrams":
        _render_transition_chart(root, sections["transition_decomposition"])
    elif name == "gold_vs_abstention_likelihood_changes":
        _render_likelihood_chart(root, sections["likelihood_changes"])
    elif name == "layer_alpha_heatmaps":
        _render_layer_alpha_chart(root, sections["layer_alpha_surface"])
    elif name == "static_vs_adaptive_matched_coverage":
        _render_matched_coverage_chart(root, sections["matched_coverage"])
    elif name == "dense_sparse_disentangled_pareto":
        _render_pareto_chart(root, sections["factuality_side_effect_pareto"])
    elif name == "prompt_method_interaction_heatmaps":
        _render_prompt_interaction_chart(
            root,
            sections["prompt_interactions"],
            sections["mixed_effects"],
        )
    elif name == "prompt_paraphrase_robustness":
        _render_paraphrase_chart(root, sections["prompt_paraphrase"])
    elif name == "safety_utility_noninferiority":
        composite = sections["composite_side_effects"]["metrics"]
        if not isinstance(composite, Mapping):
            raise DataValidationError("composite noninferiority metrics are invalid")
        _render_noninferiority_chart(root, sections["noninferiority"], composite)
    elif name == "language_switching_confusion_matrices":
        _render_language_chart(root, sections["language_confusion"])
    elif name == "local_vllm_runtime_validation":
        _render_runtime_chart(root, sections["runtime_replication"])
    else:  # pragma: no cover - dependency registry and caller check agree
        raise DataValidationError(f"no semantic chart renderer for {name!r}")


def _build_svg_report_root(*, name: str, results: FinalAnalysisResults) -> ET.Element:
    if name in {
        "zero_error_confidence_bounds",
        "adjudicated_final_labels",
        "automated_human_confusion_matrix",
    }:
        raise DataValidationError(f"{name} is tabular and cannot be rendered as SVG")
    chart_kind = _CHART_KINDS.get(name)
    if chart_kind is None:
        raise DataValidationError(f"no SVG chart kind is registered for {name!r}")
    values = _flatten_report_values(report_result_payload(name, results))
    ET.register_namespace("", _SVG_NAMESPACE)
    height = max(560, 18 * len(values) + 530)
    root = ET.Element(
        f"{{{_SVG_NAMESPACE}}}svg",
        {
            "width": "1200",
            "height": str(height),
            "viewBox": f"0 0 1200 {height}",
            "data-chart-kind": chart_kind,
        },
    )
    _svg_element(
        root,
        "metadata",
        text=f"mfh-source-data-sha256={report_result_digest(name, results)}",
    )
    _svg_element(root, "title", text=name)
    _svg_text(root, 24, 34, name.replace("_", " ").title(), size=20, css_class="chart-title")
    _render_semantic_chart(root, name=name, results=results)
    _svg_text(
        root,
        12,
        500,
        "Exact source-value ledger (each line is cryptographically bound to report data)",
        size=11,
        css_class="source-ledger-title",
    )
    for index, (result_path, result_value) in enumerate(sorted(values.items())):
        y = 524 + index * 18
        _svg_element(
            root,
            "text",
            {
                "x": "12",
                "y": str(y),
                "font-size": "9",
                "class": "source-value-binding",
                "fill": "#34495e",
                "data-result-path": result_path,
                "data-value": result_value,
            },
            text=f"{result_path} = {result_value}",
        )
    return root


def _canonical_svg_tree(element: ET.Element) -> tuple[Any, ...]:
    return (
        element.tag,
        tuple(sorted(element.attrib.items())),
        element.text or "",
        tuple(_canonical_svg_tree(child) for child in element),
    )


def render_svg_report(
    path: str | Path,
    *,
    name: str,
    results: FinalAnalysisResults,
) -> None:
    """Render a figure-specific chart plus one visible binding per result scalar."""

    root = _build_svg_report_root(name=name, results=results)
    destination = validate_active_study_artifact_paths({f"analysis report {name}": path})[
        f"analysis report {name}"
    ]
    _write_report_source(destination, ET.tostring(root, encoding="unicode") + "\n")
    _validate_report_format(
        name,
        destination,
        report_result_digest(name, results),
        results,
    )


def write_zero_error_report(path: str | Path, results: FinalAnalysisResults) -> None:
    """Write the exact one-sided zero-error bounds represented in typed results."""

    output = io.StringIO(newline="")
    output.write(
        "# mfh-source-data-sha256="
        f"{report_result_digest('zero_error_confidence_bounds', results)}\n"
    )
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "benchmark",
            "attempted",
            "errors",
            "zero_errors_observed",
            "confidence",
            "one_sided_upper",
        )
    )
    for benchmark in sorted(results.zero_error_bounds):
        value = results.zero_error_bounds[benchmark]
        assert isinstance(value, Mapping)
        writer.writerow(
            (
                benchmark,
                value["attempted"],
                value["errors"],
                value["zero_errors_observed"],
                value["confidence"],
                value["one_sided_upper"],
            )
        )
    destination = validate_active_study_artifact_paths(
        {"analysis report zero_error_confidence_bounds": path}
    )["analysis report zero_error_confidence_bounds"]
    _write_report_source(destination, output.getvalue())
    _validate_report_format(
        "zero_error_confidence_bounds",
        destination,
        report_result_digest("zero_error_confidence_bounds", results),
        results,
    )


def write_confusion_matrix_report(path: str | Path, results: FinalAnalysisResults) -> None:
    """Write the exact automated-versus-adjudicated confusion counts."""

    output = io.StringIO(newline="")
    output.write(
        "# mfh-source-data-sha256="
        f"{report_result_digest('automated_human_confusion_matrix', results)}\n"
    )
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("automated_label", "human_label", "count"))
    confusion = results.human_audit["automated_human_confusion_matrix"]
    assert isinstance(confusion, Mapping)
    for key in sorted(confusion):
        automated, human = key.split(":")
        writer.writerow((automated, human, confusion[key]))
    destination = validate_active_study_artifact_paths(
        {"analysis report automated_human_confusion_matrix": path}
    )["analysis report automated_human_confusion_matrix"]
    _write_report_source(destination, output.getvalue())
    _validate_report_format(
        "automated_human_confusion_matrix",
        destination,
        report_result_digest("automated_human_confusion_matrix", results),
        results,
    )


def write_adjudicated_labels_report(
    path: str | Path,
    results: FinalAnalysisResults,
    rows: Iterable[Mapping[str, str]],
) -> None:
    """Write and validate the blinded human-audit record export."""

    output = io.StringIO(newline="")
    output.write(
        f"# mfh-source-data-sha256={report_result_digest('adjudicated_final_labels', results)}\n"
    )
    writer = csv.DictWriter(output, fieldnames=_AUDIT_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if set(row) != set(_AUDIT_COLUMNS):
            raise DataValidationError("adjudicated audit row differs from the report schema")
        writer.writerow({name: row[name] for name in _AUDIT_COLUMNS})
    destination = validate_active_study_artifact_paths(
        {"analysis report adjudicated_final_labels": path}
    )["analysis report adjudicated_final_labels"]
    _write_report_source(destination, output.getvalue())
    _validate_report_format(
        "adjudicated_final_labels",
        destination,
        report_result_digest("adjudicated_final_labels", results),
        results,
    )


def _validate_report_data(name: str, path: Path, results: FinalAnalysisResults) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
        raise DataValidationError(f"report source data must be a non-empty regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"report source data for {name} is invalid JSON: {exc}") from exc
    expected = report_result_payload(name, results)
    if not isinstance(value, Mapping) or json.loads(canonical_json(value)) != json.loads(
        canonical_json(expected)
    ):
        raise DataValidationError(f"report source data for {name} differs from typed results")
    return expected


def _manifest_body(
    *,
    protocol: AnalysisProtocol,
    phase_digests: Mapping[str, str],
    human_audit_queue_manifest_digest: str,
    human_audit_results_manifest_digest: str,
    human_audit_queue_sha256: str,
    human_audit_results_sha256: str,
    analysis_evidence_digest: str,
    analysis_evidence_sha256: str,
    results_sha256: str,
    artifacts: Mapping[str, ReportArtifact],
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "analysis_protocol_digest": protocol.digest,
        "research_plan_sha256": protocol.research_plan_sha256,
        "phase_completion_digests": dict(phase_digests),
        "human_audit_queue_manifest_digest": human_audit_queue_manifest_digest,
        "human_audit_results_manifest_digest": human_audit_results_manifest_digest,
        "human_audit_queue_sha256": human_audit_queue_sha256,
        "human_audit_results_sha256": human_audit_results_sha256,
        "analysis_evidence_digest": analysis_evidence_digest,
        "analysis_evidence_sha256": analysis_evidence_sha256,
        "results_sha256": results_sha256,
        "report_artifacts": {
            name: {
                "filename": value.filename,
                "sha256": value.sha256,
                "source_data_filename": value.source_data_filename,
                "source_data_sha256": value.source_data_sha256,
                "source_data_digest": value.source_data_digest,
                "generator_revision": value.generator_revision,
                "result_dependencies": list(value.result_dependencies),
            }
            for name, value in artifacts.items()
        },
    }


def write_final_analysis_bundle(
    directory: str | Path,
    *,
    protocol: AnalysisProtocol,
    study: StudyProtocol,
    phase_run_directories: Mapping[ExperimentPhase | str, str | Path],
    results: FinalAnalysisResults,
    analysis_evidence_directory: str | Path,
    report_sources: Mapping[str, ReportSource],
    human_audit_queue_directory: str | Path,
    human_audit_results_directory: str | Path,
    human_audit_blinding_key: bytes,
    derived_analysis: DerivedFinalAnalysis,
) -> FinalAnalysisBundle:
    """Copy required outputs after independently verifying the E9/E10 ledgers."""

    path_inputs: dict[str, str | Path] = {
        "final analysis bundle": directory,
        "frozen analysis evidence": analysis_evidence_directory,
        "human audit queue": human_audit_queue_directory,
        "human audit results": human_audit_results_directory,
    }
    path_inputs.update(
        {
            f"{ExperimentPhase(phase).value} phase ledger": path
            for phase, path in phase_run_directories.items()
        }
    )
    path_inputs.update(
        {
            key: value
            for name, source in report_sources.items()
            for key, value in (
                (f"analysis report {name}", source.path),
                (f"analysis report source data {name}", source.data_path),
            )
        }
    )
    normalized_paths = validate_active_study_artifact_paths(path_inputs)
    destination = normalized_paths["final analysis bundle"]
    analysis_evidence_directory = normalized_paths["frozen analysis evidence"]
    human_audit_queue_directory = normalized_paths["human audit queue"]
    human_audit_results_directory = normalized_paths["human audit results"]
    phase_run_directories = {
        ExperimentPhase(phase): normalized_paths[f"{ExperimentPhase(phase).value} phase ledger"]
        for phase in phase_run_directories
    }
    report_sources = {
        name: replace(
            source,
            path=normalized_paths[f"analysis report {name}"],
            data_path=normalized_paths[f"analysis report source data {name}"],
        )
        for name, source in report_sources.items()
    }
    if destination.exists():
        raise FrozenArtifactError(f"refusing to overwrite final analysis bundle: {destination}")
    phases, ledgers = _verified_phase_sources(study, phase_run_directories)
    record_index = _analysis_record_index(ledgers)
    results.validate_against_protocol(protocol)
    results.validate_against_records(ledgers[ExperimentPhase.E10].records())
    analysis_evidence = _verify_frozen_analysis_evidence_from_derivation(
        analysis_evidence_directory,
        expected_protocol=protocol,
        study=study,
        phase_run_directories=phase_run_directories,
        expected_derivation=derived_analysis,
    )
    if canonical_json(analysis_evidence.results.to_dict()) != canonical_json(results.to_dict()):
        raise DataValidationError("final results differ from the frozen record-bound evidence")
    required = _required_artifacts(protocol)
    if set(report_sources) != required or required != set(_REPORT_RESULT_DEPENDENCIES):
        raise DataValidationError(
            "final report artifacts differ; "
            f"missing={sorted(required - set(report_sources))}, "
            f"unknown={sorted(set(report_sources) - required)}"
        )
    audit = verify_human_audit_results(
        human_audit_results_directory,
        queue_directory=human_audit_queue_directory,
        expected_protocol=protocol,
        study=study,
        phase_run_directories=_human_audit_phase_sources(phase_run_directories),
        blinding_key=human_audit_blinding_key,
    )
    _validate_human_audit_evidence(
        audit,
        results=results,
        adjudicated_report=report_sources["adjudicated_final_labels"].path,
    )
    normalized_results = json.loads(canonical_json(results.to_dict()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        reports = stage / "reports"
        reports.mkdir()
        report_data = stage / "report-data"
        report_data.mkdir()
        audit_root = stage / "human-audit"
        audit_root.mkdir()
        shutil.copytree(analysis_evidence_directory, stage / "analysis-evidence")
        shutil.copytree(human_audit_queue_directory, audit_root / "queue")
        shutil.copytree(human_audit_results_directory, audit_root / "results")
        results_path = stage / "results.json"
        results_path.write_text(
            json.dumps(normalized_results, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        artifacts: dict[str, ReportArtifact] = {}
        for name in sorted(report_sources):
            if not _ARTIFACT_NAME.fullmatch(name):
                raise DataValidationError(f"invalid final report artifact name {name!r}")
            source = report_sources[name]
            source_data = _validate_report_data(name, source.data_path, results)
            source_data_digest = stable_hash(source_data)
            suffix = _validate_report_format(
                name,
                source.path,
                source_data_digest,
                results,
                record_index,
            )
            filename = f"{name}{suffix}"
            copied = reports / filename
            shutil.copyfile(source.path, copied)
            source_data_filename = f"{name}.json"
            copied_data = report_data / source_data_filename
            copied_data.write_text(
                json.dumps(source_data, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            artifacts[name] = ReportArtifact(
                filename=filename,
                sha256=sha256_file(copied),
                source_data_filename=source_data_filename,
                source_data_sha256=sha256_file(copied_data),
                source_data_digest=source_data_digest,
                generator_revision=source.generator_revision,
                result_dependencies=_REPORT_RESULT_DEPENDENCIES[name],
            )
        body = _manifest_body(
            protocol=protocol,
            phase_digests=phases,
            human_audit_queue_manifest_digest=audit.queue_manifest_digest,
            human_audit_results_manifest_digest=audit.manifest_digest,
            human_audit_queue_sha256=sha256_path(audit_root / "queue"),
            human_audit_results_sha256=sha256_path(audit_root / "results"),
            analysis_evidence_digest=analysis_evidence.evidence_digest,
            analysis_evidence_sha256=sha256_path(stage / "analysis-evidence"),
            results_sha256=sha256_file(results_path),
            artifacts=artifacts,
        )
        bundle_digest = stable_hash(body)
        (stage / "manifest.json").write_text(
            json.dumps({**body, "bundle_digest": bundle_digest}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return verify_final_analysis_bundle(
        destination,
        expected_protocol=protocol,
        study=study,
        phase_run_directories=phase_run_directories,
        human_audit_blinding_key=human_audit_blinding_key,
        expected_derivation=derived_analysis,
    )


def verify_final_analysis_bundle(
    directory: str | Path,
    *,
    expected_protocol: AnalysisProtocol | None = None,
    study: StudyProtocol | None = None,
    phase_run_directories: Mapping[ExperimentPhase | str, str | Path] | None = None,
    human_audit_blinding_key: bytes | None = None,
    expected_derivation: DerivedFinalAnalysis | None = None,
) -> FinalAnalysisBundle:
    source = Path(directory)
    _require_exact_directory(
        source,
        regular_files={"manifest.json", "results.json"},
        regular_directories={"reports", "report-data", "human-audit", "analysis-evidence"},
        context="final analysis bundle",
    )
    _require_exact_directory(
        source / "human-audit",
        regular_files=set(),
        regular_directories={"queue", "results"},
        context="packaged human-audit root",
    )
    try:
        payload = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read final analysis manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise FrozenArtifactError("final analysis manifest must be a mapping")
    digest = payload.pop("bundle_digest", None)
    expected_keys = {
        "schema_version",
        "analysis_protocol_digest",
        "research_plan_sha256",
        "phase_completion_digests",
        "human_audit_queue_manifest_digest",
        "human_audit_results_manifest_digest",
        "human_audit_queue_sha256",
        "human_audit_results_sha256",
        "analysis_evidence_digest",
        "analysis_evidence_sha256",
        "results_sha256",
        "report_artifacts",
    }
    if set(payload) != expected_keys or payload.get("schema_version") != 2:
        raise FrozenArtifactError("final analysis manifest keys differ from schema version 2")
    if not isinstance(digest, str) or digest != stable_hash(payload):
        raise FrozenArtifactError("final analysis bundle digest mismatch")
    protocol_digest = payload["analysis_protocol_digest"]
    research_plan_sha256 = payload["research_plan_sha256"]
    results_sha256 = payload["results_sha256"]
    audit_queue_manifest_digest = payload["human_audit_queue_manifest_digest"]
    audit_results_manifest_digest = payload["human_audit_results_manifest_digest"]
    audit_queue_sha256 = payload["human_audit_queue_sha256"]
    audit_results_sha256 = payload["human_audit_results_sha256"]
    analysis_evidence_digest = payload["analysis_evidence_digest"]
    analysis_evidence_sha256 = payload["analysis_evidence_sha256"]
    if any(
        not isinstance(value, str) or not _SHA256.fullmatch(value)
        for value in (
            protocol_digest,
            research_plan_sha256,
            results_sha256,
            audit_queue_manifest_digest,
            audit_results_manifest_digest,
            audit_queue_sha256,
            audit_results_sha256,
            analysis_evidence_digest,
            analysis_evidence_sha256,
        )
    ):
        raise FrozenArtifactError("final analysis manifest contains malformed fingerprints")
    phase_value = payload["phase_completion_digests"]
    if not isinstance(phase_value, Mapping):
        raise FrozenArtifactError("final analysis phase digests must be a mapping")
    try:
        phases = _validate_phase_digests(
            {str(key): str(value) for key, value in phase_value.items()}
        )
    except DataValidationError as exc:
        raise FrozenArtifactError(str(exc)) from exc
    results_path = source / "results.json"
    if not results_path.is_file() or sha256_file(results_path) != results_sha256:
        raise FrozenArtifactError("final analysis results changed")
    try:
        results_value = json.loads(results_path.read_text(encoding="utf-8"))
        if not isinstance(results_value, Mapping):
            raise DataValidationError("final analysis result root is not a mapping")
        parsed_results = FinalAnalysisResults.from_dict(results_value)
        if expected_protocol is not None:
            parsed_results.validate_against_protocol(expected_protocol)
    except (OSError, json.JSONDecodeError, DataValidationError) as exc:
        raise FrozenArtifactError(f"invalid final analysis results: {exc}") from exc
    artifact_value = payload["report_artifacts"]
    if not isinstance(artifact_value, Mapping):
        raise FrozenArtifactError("final analysis report artifacts must be a mapping")
    artifacts: dict[str, ReportArtifact] = {}
    for name, descriptor in artifact_value.items():
        if (
            not isinstance(name, str)
            or not _ARTIFACT_NAME.fullmatch(name)
            or not isinstance(descriptor, Mapping)
            or set(descriptor)
            != {
                "filename",
                "sha256",
                "source_data_filename",
                "source_data_sha256",
                "source_data_digest",
                "generator_revision",
                "result_dependencies",
            }
        ):
            raise FrozenArtifactError("final analysis report descriptor is invalid")
        filename = descriptor["filename"]
        fingerprint = descriptor["sha256"]
        source_data_filename = descriptor["source_data_filename"]
        source_data_fingerprint = descriptor["source_data_sha256"]
        source_data_digest = descriptor["source_data_digest"]
        revision = descriptor["generator_revision"]
        dependencies = descriptor["result_dependencies"]
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not isinstance(fingerprint, str)
            or not _SHA256.fullmatch(fingerprint)
            or not isinstance(source_data_filename, str)
            or Path(source_data_filename).name != source_data_filename
            or not isinstance(source_data_fingerprint, str)
            or not _SHA256.fullmatch(source_data_fingerprint)
            or not isinstance(source_data_digest, str)
            or not _SHA256.fullmatch(source_data_digest)
            or not isinstance(revision, str)
            or not _SHA256.fullmatch(revision)
            or not isinstance(dependencies, list)
            or tuple(dependencies) != _REPORT_RESULT_DEPENDENCIES.get(name)
        ):
            raise FrozenArtifactError("final analysis report identity is invalid")
        report_path = source / "reports" / filename
        try:
            _validate_report_format(
                name,
                report_path,
                source_data_digest,
                parsed_results,
            )
        except DataValidationError as exc:
            raise FrozenArtifactError(str(exc)) from exc
        if sha256_file(report_path) != fingerprint:
            raise FrozenArtifactError(f"final report artifact changed: {filename}")
        source_data_path = source / "report-data" / source_data_filename
        if (
            not source_data_path.is_file()
            or source_data_path.is_symlink()
            or sha256_file(source_data_path) != source_data_fingerprint
        ):
            raise FrozenArtifactError(f"final report source data changed: {source_data_filename}")
        try:
            validated_source_data = _validate_report_data(name, source_data_path, parsed_results)
        except DataValidationError as exc:
            raise FrozenArtifactError(str(exc)) from exc
        if stable_hash(validated_source_data) != source_data_digest:
            raise FrozenArtifactError(
                f"final report source-data identity changed: {source_data_filename}"
            )
        artifacts[name] = ReportArtifact(
            filename,
            fingerprint,
            source_data_filename,
            source_data_fingerprint,
            source_data_digest,
            revision,
            tuple(str(item) for item in dependencies),
        )
    _require_exact_directory(
        source / "reports",
        regular_files={value.filename for value in artifacts.values()},
        regular_directories=set(),
        context="final report directory",
    )
    _require_exact_directory(
        source / "report-data",
        regular_files={value.source_data_filename for value in artifacts.values()},
        regular_directories=set(),
        context="final report-data directory",
    )
    if expected_protocol is not None and (
        protocol_digest != expected_protocol.digest
        or research_plan_sha256 != expected_protocol.research_plan_sha256
        or set(artifacts) != _required_artifacts(expected_protocol)
    ):
        raise FrozenArtifactError("final analysis bundle differs from the expected protocol")
    if (
        expected_protocol is None
        or study is None
        or phase_run_directories is None
        or human_audit_blinding_key is None
        or expected_derivation is None
    ):
        raise DataValidationError(
            "final analysis verification requires protocol, blinding key, and live phase evidence"
        )
    packaged_analysis_evidence = source / "analysis-evidence"
    try:
        if sha256_path(packaged_analysis_evidence) != analysis_evidence_sha256:
            raise FrozenArtifactError("packaged analysis evidence changed")
        verified_analysis_evidence = _verify_frozen_analysis_evidence_from_derivation(
            packaged_analysis_evidence,
            expected_protocol=expected_protocol,
            study=study,
            phase_run_directories=phase_run_directories,
            expected_derivation=expected_derivation,
        )
    except (OSError, DataValidationError) as exc:
        raise FrozenArtifactError(f"invalid packaged analysis evidence: {exc}") from exc
    if (
        verified_analysis_evidence.evidence_digest != analysis_evidence_digest
        or verified_analysis_evidence.phase_completion_digests != phases
        or canonical_json(verified_analysis_evidence.results.to_dict())
        != canonical_json(parsed_results.to_dict())
    ):
        raise FrozenArtifactError(
            "packaged analysis evidence differs from the manifest or typed results"
        )
    audit_queue = source / "human-audit" / "queue"
    audit_results = source / "human-audit" / "results"
    try:
        if (
            sha256_path(audit_queue) != audit_queue_sha256
            or sha256_path(audit_results) != audit_results_sha256
        ):
            raise FrozenArtifactError("packaged human-audit evidence changed")
        verified_audit = verify_human_audit_results(
            audit_results,
            queue_directory=audit_queue,
            expected_protocol=expected_protocol,
            study=study,
            phase_run_directories=_human_audit_phase_sources(phase_run_directories),
            blinding_key=human_audit_blinding_key,
        )
    except (OSError, DataValidationError) as exc:
        raise FrozenArtifactError(f"invalid packaged human-audit evidence: {exc}") from exc
    if (
        verified_audit.queue_manifest_digest != audit_queue_manifest_digest
        or verified_audit.manifest_digest != audit_results_manifest_digest
    ):
        raise FrozenArtifactError("packaged human-audit manifest identity changed")
    audit_artifact = artifacts.get("adjudicated_final_labels")
    if audit_artifact is None:
        raise FrozenArtifactError("final analysis lacks adjudicated labels")
    try:
        _validate_human_audit_evidence(
            verified_audit,
            results=parsed_results,
            adjudicated_report=source / "reports" / audit_artifact.filename,
        )
    except DataValidationError as exc:
        raise FrozenArtifactError(str(exc)) from exc
    if study is not None and phase_run_directories is not None:
        live_phases, live_ledgers = _verified_phase_sources(study, phase_run_directories)
        if phases != live_phases:
            raise FrozenArtifactError("final analysis source phase runs changed")
        try:
            parsed_results.validate_against_records(live_ledgers[ExperimentPhase.E10].records())
            audit_artifact = artifacts.get("adjudicated_final_labels")
            if audit_artifact is None:
                raise DataValidationError("final analysis lacks adjudicated labels")
            _validate_adjudicated_csv(
                source / "reports" / audit_artifact.filename,
                parsed_results,
                _analysis_record_index(live_ledgers),
            )
        except DataValidationError as exc:
            raise FrozenArtifactError(
                f"final analysis differs from live phase records: {exc}"
            ) from exc
    return FinalAnalysisBundle(
        directory=source,
        analysis_protocol_digest=protocol_digest,
        research_plan_sha256=research_plan_sha256,
        phase_completion_digests=MappingProxyType(phases),
        human_audit_queue_manifest_digest=str(audit_queue_manifest_digest),
        human_audit_results_manifest_digest=str(audit_results_manifest_digest),
        analysis_evidence_digest=str(analysis_evidence_digest),
        analysis_evidence_sha256=str(analysis_evidence_sha256),
        results_sha256=results_sha256,
        report_artifacts=MappingProxyType(artifacts),
        bundle_digest=digest,
    )

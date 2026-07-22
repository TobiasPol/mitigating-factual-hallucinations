"""Frozen confirmatory analysis and human-audit protocol."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from mfh.config import load_yaml
from mfh.errors import ConfigurationError, FrozenArtifactError
from mfh.provenance import sha256_file, stable_hash

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROMPTS = {"P0-neutral", "P2-calibrated-abstention"}
_FACTUAL_BENCHMARKS = {
    "triviaqa",
    "simpleqa_verified",
    "aa_omniscience_public_600",
}
_REPORT_OUTPUTS = {
    "risk_coverage_curves",
    "outcome_transition_diagrams",
    "gold_vs_abstention_likelihood_changes",
    "layer_alpha_heatmaps",
    "static_vs_adaptive_matched_coverage",
    "dense_sparse_disentangled_pareto",
    "prompt_method_interaction_heatmaps",
    "prompt_paraphrase_robustness",
    "safety_utility_noninferiority",
    "language_switching_confusion_matrices",
    "local_vllm_runtime_validation",
    "zero_error_confidence_bounds",
}
_AUDIT_BLINDING = {"model", "steering_condition", "system_prompt", "experimental_hypothesis"}
_AUDIT_QUEUES = {
    "automated_grader_disagreements",
    "partial_aa_responses",
    "language_switch_detections",
    "suspected_safety_regressions",
    "random_abstentions",
    "random_incorrect_attempts",
}
_MARGINS = {
    "ifeval_pass_rate",
    "mmlu_pro_accuracy",
    "xstest_benign_non_refusal_rate",
    "harmful_prompt_refusal_rate",
    "harmful_prompt_compliance_score",
    "requested_language_consistency",
    "perplexity_relative_change",
    "latency_relative_change",
}


class MarginScale(StrEnum):
    ABSOLUTE_PROPORTION = "absolute_proportion"
    RELATIVE_FRACTION = "relative_fraction"


def _strict(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise ConfigurationError(
            f"{context} keys differ; missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}"
        )


def _strings(value: Any, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"{context} must be a non-empty list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ConfigurationError(f"{context} must contain non-empty text")
    result = tuple(item.strip() for item in value)
    if len(set(result)) != len(result):
        raise ConfigurationError(f"{context} values must be unique")
    return result


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{context} must be non-empty text")
    return value.strip()


def _boolean(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{context} must be a boolean")
    return value


def _number(value: Any, context: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigurationError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigurationError(f"{context} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class PrimaryContrast:
    contrast_id: str
    baseline_methods: tuple[str, ...]
    treatment_methods: tuple[str, ...]
    prompts: tuple[str, ...]
    benchmarks: tuple[str, ...]
    analysis: str


@dataclass(frozen=True, slots=True)
class NonInferiorityMargin:
    metric: str
    margin: float
    scale: MarginScale
    higher_is_better: bool

    def __post_init__(self) -> None:
        if self.metric not in _MARGINS or not 0 < self.margin < 1:
            raise ConfigurationError("non-inferiority metric or margin is invalid")


@dataclass(frozen=True, slots=True)
class MixedEffectsProtocol:
    response: str
    random_intercept: str
    fixed_effects: tuple[str, ...]
    estimator: str


@dataclass(frozen=True, slots=True)
class HumanAuditProtocol:
    annotators: int
    minimum_responses_per_benchmark_model: int
    sample_seed: int
    random_responses_per_benchmark_model_outcome: int
    blinded_to: tuple[str, ...]
    mandatory_queues: tuple[str, ...]
    agreement_metrics: tuple[str, ...]
    required_outputs: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.annotators != 2 or self.minimum_responses_per_benchmark_model < 200:
            raise ConfigurationError("human audit requires two annotators and at least 200 rows")
        if (
            isinstance(self.sample_seed, bool)
            or self.sample_seed < 0
            or isinstance(self.random_responses_per_benchmark_model_outcome, bool)
            or self.random_responses_per_benchmark_model_outcome < 1
        ):
            raise ConfigurationError("human-audit sampling must be preregistered and positive")
        if set(self.blinded_to) != _AUDIT_BLINDING:
            raise ConfigurationError("human-audit blinding differs from the research plan")
        if set(self.mandatory_queues) != _AUDIT_QUEUES:
            raise ConfigurationError("human-audit mandatory queues differ from the research plan")
        if set(self.agreement_metrics) != {"cohen_kappa", "krippendorff_alpha"}:
            raise ConfigurationError("human audit must support kappa and alpha agreement")
        if set(self.required_outputs) != {
            "adjudicated_final_labels",
            "automated_human_confusion_matrix",
        }:
            raise ConfigurationError("human-audit outputs differ from the research plan")


@dataclass(frozen=True, slots=True)
class AnalysisProtocol:
    research_plan_sha256: str
    statistical_unit: str
    bootstrap_resamples: int
    confidence: float
    alpha: float
    multiple_comparison_correction: str
    paired_tests: tuple[str, ...]
    mixed_effects: MixedEffectsProtocol
    primary_contrasts: tuple[PrimaryContrast, ...]
    noninferiority_margins: Mapping[str, NonInferiorityMargin]
    human_audit: HumanAuditProtocol
    required_report_outputs: tuple[str, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1 or not _SHA256.fullmatch(self.research_plan_sha256):
            raise ConfigurationError("analysis protocol requires schema version 1 and plan hash")
        if (
            self.statistical_unit != "question"
            or self.bootstrap_resamples != 10_000
            or self.confidence != 0.95
            or self.alpha != 0.05
            or self.multiple_comparison_correction != "holm"
        ):
            raise ConfigurationError("confirmatory statistical defaults differ from the plan")
        expected_tests = {
            "paired_bootstrap",
            "mcnemar_exact",
            "bowker",
            "stuart_maxwell_sensitivity",
        }
        if set(self.paired_tests) != expected_tests:
            raise ConfigurationError("paired test family differs from the research plan")
        if self.mixed_effects != MixedEffectsProtocol(
            response="binary_correctness",
            random_intercept="question_id",
            fixed_effects=("benchmark", "method", "prompt", "method_x_prompt"),
            estimator="statsmodels_binomial_bayes_mixed_glm_map",
        ):
            raise ConfigurationError("mixed-effects model differs from the preregistered design")
        self._validate_contrasts()
        margins = dict(self.noninferiority_margins)
        if set(margins) != _MARGINS or any(key != value.metric for key, value in margins.items()):
            raise ConfigurationError("non-inferiority margins must cover the side-effect suite")
        if set(self.required_report_outputs) != _REPORT_OUTPUTS:
            raise ConfigurationError("required report outputs differ from section 21")
        object.__setattr__(self, "noninferiority_margins", MappingProxyType(margins))

    def _validate_contrasts(self) -> None:
        if tuple(value.contrast_id for value in self.primary_contrasts) != (
            "RQ1",
            "RQ2",
            "RQ3",
            "RQ4",
        ):
            raise ConfigurationError("primary contrasts must contain RQ1 through RQ4 in order")
        by_id = {value.contrast_id: value for value in self.primary_contrasts}
        expectations = {
            "RQ1": (
                {"M1"},
                {"M3"},
                {"simpleqa_verified", "aa_omniscience_public_600"},
                "paired_metric_difference",
            ),
            "RQ2": (
                {"M0"},
                {"M1", "M3"},
                _FACTUAL_BENCHMARKS,
                "transition_decomposition",
            ),
            "RQ3": (
                {"M1"},
                {"M4", "M5"},
                _FACTUAL_BENCHMARKS,
                "matched_risk_or_coverage",
            ),
            "RQ4": (
                {"M0"},
                {"M1", "M3", "M5"},
                _FACTUAL_BENCHMARKS,
                "prompt_method_difference_in_differences",
            ),
        }
        for contrast_id, (baseline, treatment, benchmarks, analysis) in expectations.items():
            contrast = by_id[contrast_id]
            if (
                set(contrast.baseline_methods) != baseline
                or set(contrast.treatment_methods) != treatment
                or set(contrast.prompts) != _PROMPTS
                or set(contrast.benchmarks) != benchmarks
                or contrast.analysis != analysis
            ):
                raise ConfigurationError(f"{contrast_id} differs from section 16.1")

    @property
    def digest(self) -> str:
        return stable_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "research_plan_sha256": self.research_plan_sha256,
            "statistical_unit": self.statistical_unit,
            "bootstrap_resamples": self.bootstrap_resamples,
            "confidence": self.confidence,
            "alpha": self.alpha,
            "multiple_comparison_correction": self.multiple_comparison_correction,
            "paired_tests": list(self.paired_tests),
            "mixed_effects": {
                "response": self.mixed_effects.response,
                "random_intercept": self.mixed_effects.random_intercept,
                "fixed_effects": list(self.mixed_effects.fixed_effects),
                "estimator": self.mixed_effects.estimator,
            },
            "primary_contrasts": [
                {
                    "contrast_id": value.contrast_id,
                    "baseline_methods": list(value.baseline_methods),
                    "treatment_methods": list(value.treatment_methods),
                    "prompts": list(value.prompts),
                    "benchmarks": list(value.benchmarks),
                    "analysis": value.analysis,
                }
                for value in self.primary_contrasts
            ],
            "noninferiority_margins": {
                key: {
                    "margin": value.margin,
                    "scale": value.scale.value,
                    "higher_is_better": value.higher_is_better,
                }
                for key, value in self.noninferiority_margins.items()
            },
            "human_audit": {
                "annotators": self.human_audit.annotators,
                "minimum_responses_per_benchmark_model": (
                    self.human_audit.minimum_responses_per_benchmark_model
                ),
                "sample_seed": self.human_audit.sample_seed,
                "random_responses_per_benchmark_model_outcome": (
                    self.human_audit.random_responses_per_benchmark_model_outcome
                ),
                "blinded_to": list(self.human_audit.blinded_to),
                "mandatory_queues": list(self.human_audit.mandatory_queues),
                "agreement_metrics": list(self.human_audit.agreement_metrics),
                "required_outputs": list(self.human_audit.required_outputs),
            },
            "required_report_outputs": list(self.required_report_outputs),
        }

    def verify_research_plan(self, path: str | Path) -> None:
        actual = sha256_file(path)
        if actual != self.research_plan_sha256:
            raise FrozenArtifactError(
                f"research plan changed: expected {self.research_plan_sha256}, found {actual}"
            )


_CONTRAST_KEYS = {
    "contrast_id",
    "baseline_methods",
    "treatment_methods",
    "prompts",
    "benchmarks",
    "analysis",
}
_MIXED_KEYS = {"response", "random_intercept", "fixed_effects", "estimator"}
_AUDIT_KEYS = {
    "annotators",
    "minimum_responses_per_benchmark_model",
    "sample_seed",
    "random_responses_per_benchmark_model_outcome",
    "blinded_to",
    "mandatory_queues",
    "agreement_metrics",
    "required_outputs",
}
_ANALYSIS_KEYS = {
    "research_plan_sha256",
    "statistical_unit",
    "bootstrap_resamples",
    "confidence",
    "alpha",
    "multiple_comparison_correction",
    "paired_tests",
    "mixed_effects",
    "primary_contrasts",
    "noninferiority_margins",
    "human_audit",
    "required_report_outputs",
}


def _parse_contrasts(value: Any) -> tuple[PrimaryContrast, ...]:
    if not isinstance(value, list):
        raise ConfigurationError("analysis.primary_contrasts must be a list")
    results: list[PrimaryContrast] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ConfigurationError(f"primary_contrasts[{index}] must be a mapping")
        _strict(item, _CONTRAST_KEYS, f"primary_contrasts[{index}]")
        results.append(
            PrimaryContrast(
                contrast_id=_text(item["contrast_id"], f"primary_contrasts[{index}].id"),
                baseline_methods=_strings(
                    item["baseline_methods"], f"primary_contrasts[{index}].baseline_methods"
                ),
                treatment_methods=_strings(
                    item["treatment_methods"], f"primary_contrasts[{index}].treatment_methods"
                ),
                prompts=_strings(item["prompts"], f"primary_contrasts[{index}].prompts"),
                benchmarks=_strings(item["benchmarks"], f"primary_contrasts[{index}].benchmarks"),
                analysis=_text(item["analysis"], f"primary_contrasts[{index}].analysis"),
            )
        )
    return tuple(results)


def _parse_margins(value: Any) -> Mapping[str, NonInferiorityMargin]:
    if not isinstance(value, Mapping):
        raise ConfigurationError("analysis.noninferiority_margins must be a mapping")
    results: dict[str, NonInferiorityMargin] = {}
    for metric, item in value.items():
        if not isinstance(metric, str) or not isinstance(item, Mapping):
            raise ConfigurationError("non-inferiority margin entries must be named mappings")
        _strict(item, {"margin", "scale", "higher_is_better"}, f"margin.{metric}")
        try:
            scale = MarginScale(_text(item["scale"], f"margin.{metric}.scale"))
        except ValueError as exc:
            raise ConfigurationError(f"invalid scale for margin {metric}") from exc
        results[metric] = NonInferiorityMargin(
            metric=metric,
            margin=_number(item["margin"], f"margin.{metric}.margin"),
            scale=scale,
            higher_is_better=_boolean(
                item["higher_is_better"], f"margin.{metric}.higher_is_better"
            ),
        )
    return results


def _parse_human_audit(value: Any) -> HumanAuditProtocol:
    if not isinstance(value, Mapping):
        raise ConfigurationError("analysis.human_audit must be a mapping")
    _strict(value, _AUDIT_KEYS, "analysis.human_audit")
    annotators = value["annotators"]
    minimum = value["minimum_responses_per_benchmark_model"]
    sample_seed = value["sample_seed"]
    random_per_outcome = value["random_responses_per_benchmark_model_outcome"]
    if (
        not isinstance(annotators, int)
        or isinstance(annotators, bool)
        or not isinstance(minimum, int)
        or isinstance(minimum, bool)
        or not isinstance(sample_seed, int)
        or isinstance(sample_seed, bool)
        or not isinstance(random_per_outcome, int)
        or isinstance(random_per_outcome, bool)
    ):
        raise ConfigurationError("human-audit counts must be integers")
    return HumanAuditProtocol(
        annotators=annotators,
        minimum_responses_per_benchmark_model=minimum,
        sample_seed=sample_seed,
        random_responses_per_benchmark_model_outcome=random_per_outcome,
        blinded_to=_strings(value["blinded_to"], "human_audit.blinded_to"),
        mandatory_queues=_strings(value["mandatory_queues"], "human_audit.mandatory_queues"),
        agreement_metrics=_strings(value["agreement_metrics"], "human_audit.agreement_metrics"),
        required_outputs=_strings(value["required_outputs"], "human_audit.required_outputs"),
    )


def load_analysis_protocol(path: str | Path) -> AnalysisProtocol:
    raw = load_yaml(path)
    if set(raw) != {"schema_version", "analysis"} or raw.get("schema_version") != 1:
        raise ConfigurationError("analysis config must contain schema_version 1 and analysis")
    value = raw.get("analysis")
    if not isinstance(value, Mapping):
        raise ConfigurationError("analysis config section must be a mapping")
    _strict(value, _ANALYSIS_KEYS, "analysis")
    mixed = value["mixed_effects"]
    if not isinstance(mixed, Mapping):
        raise ConfigurationError("analysis.mixed_effects must be a mapping")
    _strict(mixed, _MIXED_KEYS, "analysis.mixed_effects")
    bootstrap = value["bootstrap_resamples"]
    if not isinstance(bootstrap, int) or isinstance(bootstrap, bool):
        raise ConfigurationError("analysis.bootstrap_resamples must be an integer")
    return AnalysisProtocol(
        research_plan_sha256=_text(value["research_plan_sha256"], "research_plan_sha256"),
        statistical_unit=_text(value["statistical_unit"], "statistical_unit"),
        bootstrap_resamples=bootstrap,
        confidence=_number(value["confidence"], "confidence"),
        alpha=_number(value["alpha"], "alpha"),
        multiple_comparison_correction=_text(
            value["multiple_comparison_correction"], "multiple_comparison_correction"
        ),
        paired_tests=_strings(value["paired_tests"], "paired_tests"),
        mixed_effects=MixedEffectsProtocol(
            response=_text(mixed["response"], "mixed_effects.response"),
            random_intercept=_text(mixed["random_intercept"], "mixed_effects.random_intercept"),
            fixed_effects=_strings(mixed["fixed_effects"], "mixed_effects.fixed_effects"),
            estimator=_text(mixed["estimator"], "mixed_effects.estimator"),
        ),
        primary_contrasts=_parse_contrasts(value["primary_contrasts"]),
        noninferiority_margins=_parse_margins(value["noninferiority_margins"]),
        human_audit=_parse_human_audit(value["human_audit"]),
        required_report_outputs=_strings(
            value["required_report_outputs"], "required_report_outputs"
        ),
    )

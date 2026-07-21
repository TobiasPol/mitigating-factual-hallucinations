"""Unified factuality metrics and paired selective-prediction analyses."""

from mfh.evaluation.grading import deterministic_short_answer_grade, triviaqa_scores
from mfh.evaluation.language import (
    SUPPORTED_LANGUAGES,
    detect_output_language,
    requested_language_is_correct,
)
from mfh.evaluation.metrics import MetricBundle, metric_bundle
from mfh.evaluation.official import (
    AAOfficialMetrics,
    GradingRequest,
    OfficialGradeRecord,
    OfficialGraderSpec,
    aa_official_metrics,
    load_official_grader_spec,
    render_grader_prompt,
    run_official_grader,
    simpleqa_official_metrics,
)
from mfh.evaluation.openrouter import (
    OpenRouterAttemptReceipt,
    OpenRouterRoute,
    OpenRouterTransport,
    openrouter_adapter_digest,
    route_for_grader,
    run_openrouter_grader,
    verify_openrouter_catalog,
)
from mfh.evaluation.risk import RiskCoveragePoint, risk_coverage_curve
from mfh.evaluation.transitions import TransitionSummary, paired_transition_summary

__all__ = [
    "SUPPORTED_LANGUAGES",
    "AAOfficialMetrics",
    "GradingRequest",
    "MetricBundle",
    "OfficialGradeRecord",
    "OfficialGraderSpec",
    "OpenRouterAttemptReceipt",
    "OpenRouterRoute",
    "OpenRouterTransport",
    "RiskCoveragePoint",
    "TransitionSummary",
    "aa_official_metrics",
    "detect_output_language",
    "deterministic_short_answer_grade",
    "load_official_grader_spec",
    "metric_bundle",
    "openrouter_adapter_digest",
    "paired_transition_summary",
    "render_grader_prompt",
    "requested_language_is_correct",
    "risk_coverage_curve",
    "route_for_grader",
    "run_official_grader",
    "run_openrouter_grader",
    "simpleqa_official_metrics",
    "triviaqa_scores",
    "verify_openrouter_catalog",
]

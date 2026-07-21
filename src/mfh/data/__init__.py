"""Benchmark ingestion, normalization, splitting, and contamination control."""

from mfh.data.benchmarks import load_aa_csv, load_hf_benchmark, load_simpleqa_csv
from mfh.data.contamination_review import (
    finalize_contamination_review,
    prepare_contamination_review_queue,
    verify_contamination_review_queue,
    verify_contamination_review_result,
)
from mfh.data.normalization import normalize_answer, normalize_question
from mfh.data.reviewed_splits import (
    authorize_reviewed_split_bundle,
    validate_reviewed_split_snapshot,
    verify_reviewed_split_bundle,
    write_reviewed_split_bundle,
)
from mfh.data.runtime_validation import (
    select_runtime_validation_questions,
    verify_runtime_validation_bundle,
    write_runtime_validation_bundle,
)
from mfh.data.semantic_contamination import (
    ReviewPair,
    SemanticOverlapPair,
    semantic_overlap_pairs,
    verify_contamination_bundle,
    write_contamination_bundle,
)
from mfh.data.splits import (
    ExactDuplicateCurationReport,
    ExactDuplicateCurationResult,
    ResearchSplit,
    SplitPlan,
    exclude_exact_duplicate_groups,
    make_research_splits,
    semantic_group_ids,
)

__all__ = [
    "ExactDuplicateCurationReport",
    "ExactDuplicateCurationResult",
    "ResearchSplit",
    "ReviewPair",
    "SemanticOverlapPair",
    "SplitPlan",
    "authorize_reviewed_split_bundle",
    "exclude_exact_duplicate_groups",
    "finalize_contamination_review",
    "load_aa_csv",
    "load_hf_benchmark",
    "load_simpleqa_csv",
    "make_research_splits",
    "normalize_answer",
    "normalize_question",
    "prepare_contamination_review_queue",
    "select_runtime_validation_questions",
    "semantic_group_ids",
    "semantic_overlap_pairs",
    "validate_reviewed_split_snapshot",
    "verify_contamination_bundle",
    "verify_contamination_review_queue",
    "verify_contamination_review_result",
    "verify_reviewed_split_bundle",
    "verify_runtime_validation_bundle",
    "write_contamination_bundle",
    "write_reviewed_split_bundle",
    "write_runtime_validation_bundle",
]

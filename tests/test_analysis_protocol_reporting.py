from __future__ import annotations

import inspect
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from mfh.analysis.derivation import DerivedFinalAnalysis
from mfh.analysis.human_audit import HumanAuditResults
from mfh.analysis.protocol import AnalysisProtocol, load_analysis_protocol
from mfh.analysis.reporting import (
    FinalAnalysisResults,
    ReportSource,
    _validate_svg_report,
    render_svg_report,
    report_result_payload,
    verify_final_analysis_bundle,
    verify_frozen_analysis_evidence,
    write_adjudicated_labels_report,
    write_confusion_matrix_report,
    write_final_analysis_bundle,
    write_frozen_analysis_evidence,
    write_zero_error_report,
)
from mfh.contracts import GenerationRecord, Outcome, Runtime
from mfh.errors import ConfigurationError, DataValidationError, FrozenArtifactError
from mfh.evaluation.language import language_response_evidence
from mfh.evaluation.simpleqa import simpleqa_hedging_evidence
from mfh.experiments.protocol import ExperimentPhase, StudyProtocol, load_study_protocol
from mfh.experiments.runner import PhaseCompletion
from mfh.provenance import stable_hash

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "analysis" / "confirmatory.yaml"
STUDY_CONFIG = ROOT / "configs" / "experiments" / "phases.yaml"
PLAN = ROOT / "docs" / "research-plan.md"
BLINDING_KEY = bytes(range(32))


def test_confirmatory_analysis_protocol_matches_the_research_plan() -> None:
    protocol = load_analysis_protocol(CONFIG)

    protocol.verify_research_plan(PLAN)

    assert protocol.statistical_unit == "question"
    assert protocol.bootstrap_resamples == 10_000
    assert tuple(value.contrast_id for value in protocol.primary_contrasts) == (
        "RQ1",
        "RQ2",
        "RQ3",
        "RQ4",
    )
    assert protocol.human_audit.annotators == 2
    assert protocol.human_audit.minimum_responses_per_benchmark_model == 200
    assert protocol.human_audit.sample_seed == 1701
    assert protocol.human_audit.random_responses_per_benchmark_model_outcome == 25
    assert len(protocol.required_report_outputs) == 12
    assert len(protocol.digest) == 64


def test_analysis_protocol_detects_research_plan_drift(tmp_path: Path) -> None:
    protocol = load_analysis_protocol(CONFIG)
    changed = tmp_path / "research-plan.md"
    changed.write_text(PLAN.read_text(encoding="utf-8") + "\nchanged\n", encoding="utf-8")

    with pytest.raises(FrozenArtifactError, match="research plan changed"):
        protocol.verify_research_plan(changed)


def test_analysis_protocol_rejects_changed_confirmatory_defaults(tmp_path: Path) -> None:
    config = CONFIG.read_text(encoding="utf-8").replace(
        "bootstrap_resamples: 10000", "bootstrap_resamples: 9999"
    )
    path = tmp_path / "analysis.yaml"
    path.write_text(config, encoding="utf-8")

    with pytest.raises(ConfigurationError, match="defaults differ"):
        load_analysis_protocol(path)


def test_protocol_only_analysis_import_does_not_load_optional_statistics() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import mfh.analysis; "
                "assert 'scipy' not in sys.modules; "
                "assert 'statsmodels' not in sys.modules"
            ),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def _audit_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    queues = (
        "automated_grader_disagreements",
        "partial_aa_responses",
        "language_switch_detections",
        "suspected_safety_regressions",
        "random_abstentions",
        "random_incorrect_attempts",
    )
    index = 0
    for benchmark in (
        "triviaqa",
        "simpleqa_verified",
        "aa_omniscience_public_600",
    ):
        for model in ("qwen3.6-27b-mlx-4bit",):
            condition_id = f"E9:{benchmark}:{model}:M0:P0"
            for model_index in range(200):
                adjudicated = "C" if index < 350 else "I"
                raw_output = f"answer-{index}"
                rows.append(
                    {
                        "audit_id": f"audit-{index:04d}",
                        "question_id": f"{benchmark}:{model}:{model_index:03d}",
                        "condition_id": condition_id,
                        "response_sha256": stable_hash(raw_output),
                        "benchmark": benchmark,
                        "model": model,
                        "method": "M0",
                        "prompt": "P0-neutral",
                        "automated_label": "C",
                        "annotator_1_label": adjudicated,
                        "annotator_2_label": "A" if index < 12 else adjudicated,
                        "adjudicated_label": adjudicated,
                        "queue": queues[index % len(queues)],
                    }
                )
                index += 1
    return rows


def _audit_binding_digest() -> str:
    return stable_hash(
        [
            {
                "condition_id": row["condition_id"],
                "question_id": row["question_id"],
                "response_sha256": row["response_sha256"],
            }
            for row in sorted(
                _audit_rows(), key=lambda value: (value["condition_id"], value["question_id"])
            )
        ]
    )


def _results(protocol: AnalysisProtocol) -> FinalAnalysisResults:
    return FinalAnalysisResults(
        primary_contrasts={
            contrast: {
                "estimate": 0.01,
                "confidence_interval": [-0.01, 0.03],
                "p_value": 0.04,
            }
            for contrast in ("RQ1", "RQ2", "RQ3", "RQ4")
        },
        holm_adjusted_tests={
            "primary_family": {
                "raw_p_value": 0.01,
                "adjusted_p_value": 0.04,
                "rejected": True,
            }
        },
        mixed_effects={"method_x_prompt": {"estimate": 0.02, "p_value": 0.03}},
        noninferiority={
            metric: {
                "estimate": 0.0,
                "margin": specification.margin,
                "higher_is_better": specification.higher_is_better,
                "one_sided_lower": 0.0,
                "p_value": 0.008,
                "passed": True,
                "comparisons": {
                    "P0-neutral|M1_vs_M0": {
                        "estimate": 0.0,
                        "one_sided_lower": 0.0,
                        "raw_p_value": 0.001,
                        "adjusted_p_value": 0.008,
                        "rejected": True,
                        "non_inferior": True,
                        "passed": True,
                        "questions": 200,
                    }
                },
            }
            for metric, specification in protocol.noninferiority_margins.items()
        },
        composite_side_effects={
            "selected_prompt_sha256": stable_hash("P0-neutral"),
            "metrics": {
                metric: {
                    "baseline_mean": 1.0,
                    "composite_mean": 1.0,
                    "estimate": 0.0,
                    "margin": specification.margin,
                    "higher_is_better": specification.higher_is_better,
                    "one_sided_lower": 0.0,
                    "raw_p_value": 0.001,
                    "adjusted_p_value": 0.008,
                    "rejected": True,
                    "non_inferior": True,
                    "passed": True,
                    "questions": 200,
                }
                for metric, specification in protocol.noninferiority_margins.items()
            },
            "safety_transitions": {
                "xstest": {
                    "paired_questions": 200,
                    "transition_counts": {
                        f"{left}:{right}": (200 if (left, right) == ("ANSWER", "ANSWER") else 0)
                        for left in ("ANSWER", "REFUSE")
                        for right in ("ANSWER", "REFUSE")
                    },
                },
                "strongreject_or_harmbench": {
                    "paired_questions": 200,
                    "transition_counts": {
                        f"{left}:{right}": (200 if (left, right) == ("REFUSE", "REFUSE") else 0)
                        for left in ("REFUSE", "COMPLY", "OTHER")
                        for right in ("REFUSE", "COMPLY", "OTHER")
                    },
                },
            },
            "all_preregistered_noninferiority_tests_passed": True,
        },
        risk_coverage={
            "M6": {
                "aurc": 0.01,
                "coverage_limit": 0.7,
                "full_observed_aurc": 0.02,
                "maximum_coverage": 0.9,
                "point_count": 2,
                "target_risks": {
                    f"coverage_{target:02d}": {
                        "target_coverage": target / 100,
                        "reached": True,
                        "achieved_coverage": 0.5 if target <= 50 else 0.9,
                        "risk": 0.01 if target <= 50 else 0.02,
                        "attempted": 100 if target <= 50 else 180,
                        "incorrect": 1 if target <= 50 else 4,
                        "threshold": 0.2 if target <= 50 else 0.8,
                    }
                    for target in (25, 50, 75, 90)
                },
                "points": {
                    "point_000": {
                        "threshold": 0.2,
                        "coverage": 0.5,
                        "risk": 0.01,
                        "accuracy": 0.49,
                        "attempted": 100,
                        "incorrect": 1,
                    },
                    "point_001": {
                        "threshold": 0.8,
                        "coverage": 0.9,
                        "risk": 0.02,
                        "accuracy": 0.88,
                        "attempted": 180,
                        "incorrect": 4,
                    },
                },
            }
        },
        transition_decomposition={
            comparison: {
                "paired_questions": 1_000,
                "unscorable_pairs_excluded": 0,
                "knowledge_recovery": 0.2,
                "abstention_substitution": 0.1,
                "strict_overrefusal": 0.01,
                "regression": 0.02,
                "transition_counts": {
                    f"{left}:{right}": (1_000 if (left, right) == ("I", "C") else 0)
                    for left in ("C", "P", "I", "A")
                    for right in ("C", "P", "I", "A")
                },
            }
            for comparison in (
                *(
                    f"{benchmark}|{prompt}|M0_to_{method}"
                    for benchmark in (
                        "triviaqa",
                        "simpleqa_verified",
                        "aa_omniscience_public_600",
                    )
                    for prompt in ("P0-neutral", "P2-calibrated-abstention")
                    for method in ("M1", "M2", "M3", "M4", "M5")
                ),
                *(
                    f"{benchmark}|P0-neutral|M0_to_M6"
                    for benchmark in (
                        "triviaqa",
                        "simpleqa_verified",
                        "aa_omniscience_public_600",
                    )
                ),
            )
        },
        likelihood_changes={"M0_to_M3": {"gold": 0.2, "abstention": 0.1}},
        layer_alpha_surface={
            "confirmatory|layer_21|alpha_1": {
                "risk": 0.01,
                "coverage": 0.7,
                "layer": 21,
                "alpha": 1.0,
            }
        },
        matched_coverage={
            "M3_vs_M1": {
                "baseline_coverage": 0.7,
                "coverage": 0.7,
                "risk_difference": -0.01,
            }
        },
        factuality_side_effect_pareto={
            "M5": {
                "risk": 0.01,
                "factuality_gain": 0.02,
                "minimum_normalized_noninferiority_slack": 0.5,
            }
        },
        prompt_interactions={"M3_x_P2": {"estimate": 0.02}},
        prompt_paraphrase={
            "M3": {
                "variance": 0.001,
                "minimum_accuracy": 0.7,
                "maximum_accuracy": 0.8,
                "mean_accuracy": 0.75,
            }
        },
        rq1_generalization={
            "M3|calibration-only|fold-0|P0-neutral-to-P2-calibrated-abstention": {
                "evaluation_record_count": 200,
                "accuracy": 0.75,
            },
            "M3|full-vector-bank-relearning|fold-0|P0-neutral-to-P2-calibrated-abstention": {
                "evaluation_record_count": 200,
                "accuracy": 0.76,
            },
        },
        e7_interpretability={
            "artifact_sha256": "d" * 64,
            "feature_stability": 0.9,
            "selected_feature_count": 2,
        },
        language_confusion={
            "requested_detected_matrix": {"de:de": 10},
            "automated_metrics_by_language": {
                "de": {
                    "rows": 10,
                    "correct_output_language_rate": 1.0,
                    "non_target_script_token_rate": 0.0,
                    "code_switching_rate": 0.0,
                    "factual_accuracy": 1.0,
                    "abstention_rate": 0.0,
                }
            },
            "correct_to_wrong_language": {
                "eligible_baseline_correct_language_rows": 10,
                "wrong_language_transitions": 0,
                "rate": 0.0,
                "by_language": {
                    "de": {
                        "eligible_baseline_correct_language_rows": 10,
                        "wrong_language_transitions": 0,
                        "rate": 0.0,
                    }
                },
            },
            "human_audit": {
                "agreement_metrics": {
                    "cohen_kappa": 1.0,
                    "krippendorff_alpha": 1.0,
                },
                "adjudication_summary": {"rows": 10, "disagreements": 0},
                "automated_human_confusion_matrix": {"CONSISTENT:CONSISTENT": 10},
                "human_consistency_score": {
                    "consistent": 10,
                    "judged": 10,
                    "rate": 1.0,
                },
                "record_binding_digest": "f" * 64,
            },
        },
        runtime_replication={
            "local_mlx_execution": {
                "passed": True,
                "record_count": 610,
                "mean_latency_seconds": 0.01,
                "p95_latency_seconds": 0.01,
                "maximum_latency_seconds": 0.01,
                "mean_candidate_generation_seconds": 0.008,
                "candidate_generated_record_count": 600,
                "pre_generation_abstention_count": 10,
                "maximum_peak_memory_bytes": 1024,
                "mean_prompt_tokens_per_second": 100.0,
                "mean_generation_tokens_per_second": 20.0,
                "signed_execution_receipts": 610,
                "runtime_identity_sha256": "9" * 64,
                "runtime_session_identity_sha256": "9" * 64,
                "intervention_site_evidence_sha256": "8" * 64,
                "runtime_counts": {"mlx": 610},
                "mlx_versions": {"0.31.0": 1},
                "mlx_lm_versions": {"0.31.3": 1},
                "machine_models": {"Mac16,7": 1},
                "chips": {"Apple M4 Max": 1},
                "operating_systems": {"macOS 15.5": 1},
                "os_builds": {"24F74": 1},
                "architectures": {"arm64": 1},
                "unified_memory_bytes": 48 * 2**30,
                "intervention_site_counts": {"no-intervention": 610},
                "policy_action_counts": {"release": 610},
                "mean_activation_delta_norm_by_site": {"no-intervention": 0.0},
            }
        },
        power_analysis={"RQ1": {"estimated_power": 0.8, "target_sample_size": 1_000}},
        official_metrics={
            "triviaqa": {
                "counts": {"C": 200, "P": 0, "I": 0, "A": 0, "U": 0},
                "total": 200,
                "scorable": 200,
                "attempted": 200,
                "accuracy": 1.0,
                "coverage": 1.0,
                "hallucination_risk": 0.0,
                "exact_match": 1.0,
                "token_f1": 1.0,
            },
            "simpleqa_verified": {
                "counts": {"C": 200, "P": 0, "I": 0, "A": 0, "U": 0},
                "total": 200,
                "scorable": 200,
                "attempted": 200,
                "accuracy": 1.0,
                "coverage": 1.0,
                "hallucination_risk": 0.0,
                "simpleqa_f1": 1.0,
                "accuracy_given_attempted": 1.0,
                "attempt_rate": 1.0,
                "incorrect_attempted_rate": 0.0,
                "punting_rate": 0.0,
                "hedging_rate": 0.0,
            },
            "aa_omniscience_public_600": {
                "counts": {"C": 200, "P": 0, "I": 0, "A": 0, "U": 0},
                "total": 200,
                "scorable": 200,
                "attempted": 200,
                "accuracy": 1.0,
                "coverage": 1.0,
                "hallucination_risk": 0.0,
                "omniscience_index": 100.0,
                "accuracy_given_attempted": 1.0,
                "correct_rate": 1.0,
                "partial_rate": 0.0,
                "incorrect_rate": 0.0,
                "abstention_rate": 0.0,
            },
        },
        grader_failure_rates={
            "simpleqa_verified": {
                "responses": 200,
                "grader_calls": 200,
                "failed_calls": 0,
                "recovered_responses": 0,
                "terminal_failures": 0,
                "failure_rate": 0.0,
            },
            "aa_omniscience_public_600": {
                "responses": 200,
                "grader_calls": 200,
                "failed_calls": 0,
                "recovered_responses": 0,
                "terminal_failures": 0,
                "failure_rate": 0.0,
            },
        },
        zero_error_bounds={
            "simpleqa_verified": {
                "attempted": 200,
                "errors": 0,
                "zero_errors_observed": True,
                "confidence": 0.95,
                "one_sided_upper": 1 - 0.05 ** (1 / 200),
            },
            "aa_omniscience_public_600": {
                "attempted": 200,
                "errors": 0,
                "zero_errors_observed": True,
                "confidence": 0.95,
                "one_sided_upper": 1 - 0.05 ** (1 / 200),
            },
        },
        human_audit={
            "agreement_metrics": {"cohen_kappa": 0.9, "krippendorff_alpha": 0.89},
            "adjudication_summary": {"rows": 600, "disagreements": 12},
            "automated_human_confusion_matrix": {"C:C": 350, "C:I": 250},
            "record_binding_digest": _audit_binding_digest(),
        },
    )


def _report_sources(tmp_path: Path, results: FinalAnalysisResults) -> dict[str, ReportSource]:
    protocol = load_analysis_protocol(CONFIG)
    names = set(protocol.required_report_outputs) | set(protocol.human_audit.required_outputs)
    source = tmp_path / "sources"
    source.mkdir()
    data = tmp_path / "report-source-data"
    data.mkdir()
    result: dict[str, ReportSource] = {}
    audit_rows = _audit_rows()
    for name in names:
        if name == "zero_error_confidence_bounds":
            path = source / f"{name}.csv"
            write_zero_error_report(path, results)
        elif name == "adjudicated_final_labels":
            path = source / f"{name}.csv"
            write_adjudicated_labels_report(path, results, audit_rows)
        elif name == "automated_human_confusion_matrix":
            path = source / f"{name}.csv"
            write_confusion_matrix_report(path, results)
        else:
            path = source / f"{name}.svg"
            render_svg_report(path, name=name, results=results)
        data_path = data / f"{name}.json"
        data_path.write_text(
            json.dumps(report_result_payload(name, results), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result[name] = ReportSource(path=path, data_path=data_path, generator_revision="a" * 64)
    return result


def _completion(
    phase: ExperimentPhase, digest_character: str, *, record_count: int
) -> PhaseCompletion:
    return PhaseCompletion(
        phase=phase,
        contract_digest="c" * 64,
        record_count=record_count,
        shard_fingerprints={"records-00000.jsonl": "d" * 64},
        record_set_digest="e" * 64,
        gate_result_digests={"gate": "f" * 64},
        gate_file_fingerprints={"gate.json": "1" * 64},
        gate_artifact_fingerprints={"gate/evaluation": "2" * 64},
        completion_digest=digest_character * 64,
    )


def _phase_records(phase: ExperimentPhase) -> tuple[GenerationRecord, ...]:
    repositories = {
        "qwen3.6-27b-mlx-4bit": "mlx-community/Qwen3.6-27B-4bit",
    }
    result: list[GenerationRecord] = []
    for index, row in enumerate(_audit_rows()):
        raw_output = f"answer-{index}"
        benchmark = row["benchmark"]
        metadata: dict[str, object] = {
            "phase": phase.value,
            "partition": "frozen-eval",
            "prompt_template_sha256": "1" * 64,
            "study_protocol_digest": "2" * 64,
            "official_score_output_sha256": stable_hash(raw_output),
            "prompt_feature_peak_memory_bytes": 1024,
            "e10_generation_execution_signature": "a" * 128,
        }
        if benchmark == "triviaqa":
            metadata.update({"official_exact_match": 1.0, "official_token_f1": 1.0})
        else:
            metadata.update({"grader_attempts": 1, "grader_failed": False})
        if benchmark == "simpleqa_verified":
            metadata["simpleqa_hedging_evidence"] = simpleqa_hedging_evidence(raw_output)
        result.append(
            GenerationRecord(
                question_id=row["question_id"],
                benchmark=benchmark,
                model_repository=repositories[row["model"]],
                model_revision="a" * 40,
                runtime=Runtime.MLX,
                quantization="synthetic",
                system_prompt_id="P0-neutral",
                rendered_prompt_hash="3" * 64,
                steering_method="M0" if phase is ExperimentPhase.E9 else "M6",
                layer=None,
                token_scope=None,
                alpha=0.0,
                sparsity=None,
                controller_scores={},
                raw_output=raw_output,
                normalized_answer=raw_output,
                outcome=Outcome.CORRECT,
                generation_latency_seconds=0.01,
                input_tokens=5,
                output_tokens=1,
                condition_id=(
                    row["condition_id"]
                    if phase is ExperimentPhase.E9
                    else row["condition_id"].replace("E9:", "E10:").replace(":M0:", ":M6:")
                ),
                metadata=metadata,
            )
        )
    if phase is ExperimentPhase.E10:
        for index in range(10):
            alias = f"antwort-{index}"
            raw_output = f"Die Antwort ist {alias}."
            evidence = language_response_evidence(raw_output, "de", (alias,))
            result.append(
                GenerationRecord(
                    question_id=f"language-{index:03d}",
                    benchmark="language_consistency",
                    model_repository="mlx-community/Qwen3.6-27B-4bit",
                    model_revision="a" * 40,
                    runtime=Runtime.MLX,
                    quantization="synthetic",
                    system_prompt_id="P0-neutral",
                    rendered_prompt_hash="4" * 64,
                    steering_method="M6",
                    layer=None,
                    token_scope=None,
                    alpha=0.0,
                    sparsity=None,
                    controller_scores={},
                    raw_output=raw_output,
                    normalized_answer=raw_output,
                    outcome=Outcome.CORRECT,
                    generation_latency_seconds=0.01,
                    input_tokens=5,
                    output_tokens=1,
                    condition_id=f"E10:language:M6:{index:03d}",
                    metadata={
                        "phase": "E10",
                        "partition": "frozen-eval",
                        "prompt_template_sha256": "1" * 64,
                        "study_protocol_digest": "2" * 64,
                        "requested_language": "de",
                        "detected_language": evidence["detected_language"],
                        "requested_language_correct": evidence["requested_language_correct"],
                        "non_target_script_token_rate": evidence["non_target_script_token_rate"],
                        "code_switching": evidence["code_switching"],
                        "language_factual_correct": evidence["factual_correct"],
                        "language_abstained": evidence["abstained"],
                        "language_evaluator_revision": evidence["evaluator_revision"],
                        "accepted_aliases_digest": evidence["accepted_aliases_digest"],
                        "language_score_output_sha256": stable_hash(raw_output),
                        "language_evaluation_evidence": evidence,
                        "prompt_feature_peak_memory_bytes": 1024,
                        "e10_generation_execution_signature": "a" * 128,
                    },
                )
            )
    return tuple(result)


def _phase_open_mock(
    path: Path,
    *,
    study: StudyProtocol,
    phase: ExperimentPhase | None = None,
) -> Mock:
    del study
    phase = phase or ExperimentPhase(Path(path).name.removesuffix("-run"))
    ledger = Mock()
    digest_character = {
        ExperimentPhase.E1: "1",
        ExperimentPhase.E3: "3",
        ExperimentPhase.E6: "6",
        ExperimentPhase.E7: "7",
        ExperimentPhase.E8: "8",
        ExperimentPhase.E9: "9",
        ExperimentPhase.E10: "a",
    }[phase]
    records = _phase_records(phase) if phase in {ExperimentPhase.E9, ExperimentPhase.E10} else ()
    ledger.verify_complete.return_value = _completion(
        phase,
        digest_character,
        record_count=len(records),
    )
    ledger.records.side_effect = lambda: iter(records)
    ledger.contract.input_fingerprints = {}
    if phase is ExperimentPhase.E9:
        ledger.contract.input_fingerprints = {"frozen_evaluation_scripts": "b" * 64}
    elif phase is ExperimentPhase.E10:
        ledger.contract.input_fingerprints = {"evaluation_scripts": "b" * 64}
    return ledger


def _all_phase_runs(tmp_path: Path) -> dict[str, Path]:
    return {
        phase: tmp_path / f"{phase}-run" for phase in ("E1", "E3", "E6", "E7", "E8", "E9", "E10")
    }


def _frozen_analysis_evidence(
    tmp_path: Path,
    protocol: AnalysisProtocol,
    study: StudyProtocol,
    results: FinalAnalysisResults,
) -> Path:
    path = tmp_path / "record-bound-analysis-evidence"
    with (
        patch("mfh.analysis.reporting.open_phase_prerequisite", side_effect=_phase_open_mock),
        patch(
            "mfh.analysis.derivation.derive_final_analysis_from_artifacts",
            return_value=_derived(results),
        ),
    ):
        write_frozen_analysis_evidence(
            path,
            protocol=protocol,
            study=study,
            phase_run_directories=_all_phase_runs(tmp_path),
            robustness_result_directory=tmp_path / "robustness-results",
            human_audit_queue_directory=tmp_path / "audit-queue",
            human_audit_results_directory=tmp_path / "audit-results",
            human_audit_blinding_key=BLINDING_KEY,
            aa_official_directory=tmp_path / "aa-official",
            expected_aa_official_manifest_digest="f" * 64,
        )
    return path


def _derived(results: FinalAnalysisResults) -> DerivedFinalAnalysis:
    phases = ("E1", "E3", "E6", "E7", "E8", "E9", "E10")
    return DerivedFinalAnalysis(
        results=results,
        source_record_digests={phase: stable_hash((phase, "records")) for phase in phases},
        e9_analysis_digest=stable_hash("e9-analysis"),
        human_audit_manifest_digest=stable_hash("human-audit"),
        robustness_record_digest=stable_hash("prompt-robustness"),
        runtime_attestation_digest=stable_hash("runtime-attestation"),
        phase_completion_digests={phase: stable_hash((phase, "completion")) for phase in phases},
    )


def _audit_evidence(
    tmp_path: Path,
    results: FinalAnalysisResults,
    sources: dict[str, ReportSource],
) -> tuple[Path, Path, HumanAuditResults]:
    queue = tmp_path / "verified-audit-queue"
    audit_results = tmp_path / "verified-audit-results"
    queue.mkdir()
    audit_results.mkdir()
    (queue / "evidence.txt").write_text("verified queue\n", encoding="utf-8")
    (audit_results / "adjudicated-factual.csv").write_bytes(
        sources["adjudicated_final_labels"].path.read_bytes()
    )
    verified = HumanAuditResults(
        directory=audit_results,
        manifest_digest="7" * 64,
        queue_manifest_digest="6" * 64,
        scientific_eligible=True,
        summary={"factual_reporting_payload": dict(results.human_audit)},
    )
    return queue, audit_results, verified


def test_final_analysis_bundle_is_atomic_complete_and_tamper_evident(
    tmp_path: Path,
) -> None:
    protocol = load_analysis_protocol(CONFIG)
    study = load_study_protocol(STUDY_CONFIG)
    results = _results(protocol)
    sources = _report_sources(tmp_path, results)
    destination = tmp_path / "final-analysis"
    phase_runs = _all_phase_runs(tmp_path)
    analysis_evidence = _frozen_analysis_evidence(tmp_path, protocol, study, results)
    audit_queue, audit_results, verified_audit = _audit_evidence(tmp_path, results, sources)

    with (
        patch(
            "mfh.analysis.reporting.open_phase_prerequisite", side_effect=_phase_open_mock
        ) as phase_open,
        patch(
            "mfh.analysis.reporting.verify_human_audit_results",
            return_value=verified_audit,
        ),
    ):
        written = write_final_analysis_bundle(
            destination,
            protocol=protocol,
            study=study,
            phase_run_directories=phase_runs,
            results=results,
            analysis_evidence_directory=analysis_evidence,
            report_sources=sources,
            human_audit_queue_directory=audit_queue,
            human_audit_results_directory=audit_results,
            human_audit_blinding_key=BLINDING_KEY,
            derived_analysis=_derived(results),
        )
    assert phase_open.call_count == 28
    with (
        patch("mfh.analysis.reporting.open_phase_prerequisite", side_effect=_phase_open_mock),
        patch(
            "mfh.analysis.reporting.verify_human_audit_results",
            return_value=verified_audit,
        ),
    ):
        verified = verify_final_analysis_bundle(
            destination,
            expected_protocol=protocol,
            study=study,
            phase_run_directories=phase_runs,
            human_audit_blinding_key=BLINDING_KEY,
            expected_derivation=_derived(results),
        )

    assert written.bundle_digest == verified.bundle_digest
    assert written.phase_completion_digests == {
        phase: character * 64
        for phase, character in {
            "E1": "1",
            "E3": "3",
            "E6": "6",
            "E7": "7",
            "E8": "8",
            "E9": "9",
            "E10": "a",
        }.items()
    }
    assert len(verified.report_artifacts) == 14
    unexpected = destination / "unexpected-empty-directory"
    unexpected.mkdir()
    with pytest.raises(FrozenArtifactError, match="missing or unexpected entries"):
        verify_final_analysis_bundle(
            destination,
            expected_protocol=protocol,
            study=study,
            phase_run_directories=phase_runs,
            human_audit_blinding_key=BLINDING_KEY,
            expected_derivation=_derived(results),
        )
    unexpected.rmdir()
    manifest = destination / "manifest.json"
    manifest_bytes = manifest.read_bytes()
    external_manifest = tmp_path / "external-manifest.json"
    external_manifest.write_bytes(manifest_bytes)
    manifest.unlink()
    manifest.symlink_to(external_manifest)
    with pytest.raises(FrozenArtifactError, match="symbolic link"):
        verify_final_analysis_bundle(
            destination,
            expected_protocol=protocol,
            study=study,
            phase_run_directories=phase_runs,
            human_audit_blinding_key=BLINDING_KEY,
            expected_derivation=_derived(results),
        )
    manifest.unlink()
    manifest.write_bytes(manifest_bytes)
    name = "risk_coverage_curves"
    artifact = verified.report_artifacts[name]
    report_path = destination / "reports" / artifact.filename
    report_path.write_text(
        report_path.read_text(encoding="utf-8").replace(
            'stroke-width="3"',
            'stroke-width="4"',
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(FrozenArtifactError, match="geometry differs"):
        verify_final_analysis_bundle(
            destination,
            expected_protocol=protocol,
            study=study,
            phase_run_directories=phase_runs,
            human_audit_blinding_key=BLINDING_KEY,
            expected_derivation=_derived(results),
        )


def test_final_analysis_bundle_requires_every_report_and_a_typed_result(
    tmp_path: Path,
) -> None:
    protocol = load_analysis_protocol(CONFIG)
    study = load_study_protocol(STUDY_CONFIG)
    results = _results(protocol)
    sources = _report_sources(tmp_path, results)
    sources.pop("risk_coverage_curves")
    analysis_evidence = _frozen_analysis_evidence(tmp_path, protocol, study, results)

    with (
        patch("mfh.analysis.reporting.open_phase_prerequisite", side_effect=_phase_open_mock),
        pytest.raises(DataValidationError, match="missing"),
    ):
        write_final_analysis_bundle(
            tmp_path / "incomplete",
            protocol=protocol,
            study=study,
            phase_run_directories=_all_phase_runs(tmp_path),
            results=results,
            analysis_evidence_directory=analysis_evidence,
            report_sources=sources,
            human_audit_queue_directory=tmp_path / "unused-queue",
            human_audit_results_directory=tmp_path / "unused-results",
            human_audit_blinding_key=BLINDING_KEY,
            derived_analysis=_derived(results),
        )

    with pytest.raises(DataValidationError, match="result keys differ"):
        FinalAnalysisResults.from_dict({"arbitrary": {"claim": True}})


def test_final_analysis_rejects_internally_valid_metrics_from_the_wrong_record_count() -> None:
    protocol = load_analysis_protocol(CONFIG)
    payload = json.loads(json.dumps(_results(protocol).to_dict()))
    for benchmark in payload["official_metrics"]:
        value = payload["official_metrics"][benchmark]
        value.update(
            {
                "counts": {"C": 1, "P": 0, "I": 0, "A": 0, "U": 0},
                "total": 1,
                "scorable": 1,
                "attempted": 1,
            }
        )
    for benchmark in payload["grader_failure_rates"]:
        payload["grader_failure_rates"][benchmark]["responses"] = 1
        payload["grader_failure_rates"][benchmark]["grader_calls"] = 1
        payload["zero_error_bounds"][benchmark].update({"attempted": 1, "one_sided_upper": 0.95})
    claimed = FinalAnalysisResults.from_dict(payload)

    with pytest.raises(DataValidationError, match="differ from E10 ledger outcomes"):
        claimed.validate_against_records(_phase_records(ExperimentPhase.E10))


def test_final_analysis_rejects_incomplete_or_miscounted_transition_matrices() -> None:
    protocol = load_analysis_protocol(CONFIG)
    missing = json.loads(json.dumps(_results(protocol).to_dict()))
    missing["transition_decomposition"].pop("simpleqa_verified|P0-neutral|M0_to_M2")
    with pytest.raises(DataValidationError, match="every preregistered"):
        FinalAnalysisResults.from_dict(missing)

    miscounted = json.loads(json.dumps(_results(protocol).to_dict()))
    comparison = miscounted["transition_decomposition"]["simpleqa_verified|P0-neutral|M0_to_M2"]
    comparison["transition_counts"]["I:C"] = 999
    with pytest.raises(DataValidationError, match="count total differs"):
        FinalAnalysisResults.from_dict(miscounted)


def test_hidden_or_zero_geometry_cannot_satisfy_svg_result_bindings(tmp_path: Path) -> None:
    results = _results(load_analysis_protocol(CONFIG))
    path = tmp_path / "risk.svg"
    render_svg_report(path, name="risk_coverage_curves", results=results)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            'class="source-value-binding"',
            'class="source-value-binding" display="none"',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(DataValidationError, match="invalid result bindings"):
        _validate_svg_report("risk_coverage_curves", path, results)


@pytest.mark.parametrize(
    "attack",
    (
        "coordinate",
        "style_clip",
        "transparent",
        "transform",
        "viewbox",
        "off_canvas_polyline",
        "remove_mark",
    ),
)
def test_svg_semantic_geometry_is_recomputed_from_typed_results(
    tmp_path: Path, attack: str
) -> None:
    results = _results(load_analysis_protocol(CONFIG))
    path = tmp_path / f"risk-{attack}.svg"
    render_svg_report(path, name="risk_coverage_curves", results=results)
    tree = ET.parse(path)
    root = tree.getroot()
    marks = [
        element
        for element in root.iter()
        if element.attrib.get("data-semantic-mark") == "risk-coverage-point"
    ]
    assert marks
    mark = marks[0]
    if attack == "coordinate":
        mark.attrib["cx"] = str(float(mark.attrib["cx"]) + 100)
    elif attack == "style_clip":
        mark.attrib["style"] = "clip-path:url(#empty)"
    elif attack == "transparent":
        mark.attrib["fill"] = "transparent"
    elif attack == "transform":
        mark.attrib["transform"] = "translate(2000 0)"
    elif attack == "viewbox":
        root.attrib["viewBox"] = "5000 5000 1200 560"
    elif attack == "off_canvas_polyline":
        line = next(
            element
            for element in root.iter()
            if element.attrib.get("data-semantic-mark") == "risk-coverage-line"
        )
        line.attrib["points"] = "5000,5000 5100,5100"
    else:
        parent = next(parent for parent in root.iter() if mark in list(parent))
        parent.remove(mark)
    tree.write(path, encoding="unicode")

    with pytest.raises(
        DataValidationError,
        match=r"geometry differs|does not encode|invalid semantic",
    ):
        _validate_svg_report("risk_coverage_curves", path, results)


@pytest.mark.parametrize(
    ("attribute", "value"),
    (
        ("font-size", "0"),
        ("fill", "none"),
        ("fill-opacity", "0"),
        ("transform", "translate(5000 5000)"),
    ),
)
def test_svg_source_ledger_must_remain_visibly_on_canvas(
    tmp_path: Path, attribute: str, value: str
) -> None:
    results = _results(load_analysis_protocol(CONFIG))
    path = tmp_path / f"ledger-{attribute}.svg"
    render_svg_report(path, name="risk_coverage_curves", results=results)
    tree = ET.parse(path)
    binding = next(
        element for element in tree.getroot().iter() if "data-result-path" in element.attrib
    )
    binding.attrib[attribute] = value
    tree.write(path, encoding="unicode")

    with pytest.raises(DataValidationError, match=r"invalid result bindings|geometry differs"):
        _validate_svg_report("risk_coverage_curves", path, results)


@pytest.mark.parametrize(
    ("name", "chart_kind", "mark_class"),
    (
        ("risk_coverage_curves", "risk-coverage-lines", "risk-coverage-series"),
        (
            "outcome_transition_diagrams",
            "outcome-transition-flow",
            "transition-cell",
        ),
        (
            "gold_vs_abstention_likelihood_changes",
            "likelihood-grouped-bars",
            "gold-likelihood-change",
        ),
        ("layer_alpha_heatmaps", "layer-alpha-heatmap", "layer-alpha-cell"),
        (
            "static_vs_adaptive_matched_coverage",
            "matched-coverage-scatter",
            "matched-coverage-point",
        ),
        (
            "dense_sparse_disentangled_pareto",
            "factuality-utility-pareto",
            "pareto-point",
        ),
        (
            "prompt_method_interaction_heatmaps",
            "prompt-method-heatmap",
            "prompt-interaction-cell",
        ),
        (
            "prompt_paraphrase_robustness",
            "prompt-paraphrase-range",
            "paraphrase-range",
        ),
        (
            "safety_utility_noninferiority",
            "noninferiority-forest",
            "component-noninferiority-estimate",
        ),
        (
            "language_switching_confusion_matrices",
            "language-confusion-matrix",
            "language-confusion-cell",
        ),
        (
            "local_mlx_runtime_validation",
            "runtime-validation-bars",
            "runtime-latency-bar",
        ),
    ),
)
def test_svg_reports_use_figure_specific_semantics(
    tmp_path: Path,
    name: str,
    chart_kind: str,
    mark_class: str,
) -> None:
    results = _results(load_analysis_protocol(CONFIG))
    path = tmp_path / f"{name}.svg"
    render_svg_report(path, name=name, results=results)
    value = path.read_text(encoding="utf-8")
    assert f'data-chart-kind="{chart_kind}"' in value
    assert f'class="{mark_class}"' in value


def test_final_analysis_rejects_results_detached_from_frozen_evidence(tmp_path: Path) -> None:
    protocol = load_analysis_protocol(CONFIG)
    study = load_study_protocol(STUDY_CONFIG)
    original = _results(protocol)
    evidence = _frozen_analysis_evidence(tmp_path, protocol, study, original)
    payload = json.loads(json.dumps(original.to_dict()))
    payload["primary_contrasts"]["RQ1"]["estimate"] = 0.25
    claimed = FinalAnalysisResults.from_dict(payload)

    with (
        patch("mfh.analysis.reporting.open_phase_prerequisite", side_effect=_phase_open_mock),
        pytest.raises(FrozenArtifactError, match="derived analysis differs"),
    ):
        write_final_analysis_bundle(
            tmp_path / "detached-results",
            protocol=protocol,
            study=study,
            phase_run_directories=_all_phase_runs(tmp_path),
            results=claimed,
            analysis_evidence_directory=evidence,
            report_sources=_report_sources(tmp_path, claimed),
            human_audit_queue_directory=tmp_path / "unused-queue",
            human_audit_results_directory=tmp_path / "unused-results",
            human_audit_blinding_key=BLINDING_KEY,
            derived_analysis=_derived(claimed),
        )


def test_record_derived_evidence_requires_full_source_replay(tmp_path: Path) -> None:
    protocol = load_analysis_protocol(CONFIG)
    study = load_study_protocol(STUDY_CONFIG)
    results = _results(protocol)
    derived = _derived(results)
    path = tmp_path / "derived-evidence"
    phase_runs = _all_phase_runs(tmp_path)
    parameters = inspect.signature(write_frozen_analysis_evidence).parameters
    assert "results" not in parameters
    assert "derived_analysis" not in parameters
    with (
        patch("mfh.analysis.reporting.open_phase_prerequisite", side_effect=_phase_open_mock),
        patch(
            "mfh.analysis.derivation.derive_final_analysis_from_artifacts",
            return_value=derived,
        ),
    ):
        written = write_frozen_analysis_evidence(
            path,
            protocol=protocol,
            study=study,
            phase_run_directories=phase_runs,
            robustness_result_directory=tmp_path / "robustness-results",
            human_audit_queue_directory=tmp_path / "audit-queue",
            human_audit_results_directory=tmp_path / "audit-results",
            human_audit_blinding_key=BLINDING_KEY,
            aa_official_directory=tmp_path / "aa-official",
            expected_aa_official_manifest_digest="f" * 64,
        )
    assert written.schema_version == 2

    changed_payload = json.loads(json.dumps(results.to_dict()))
    changed_payload["primary_contrasts"]["RQ1"]["estimate"] = 0.25
    changed = _derived(FinalAnalysisResults.from_dict(changed_payload))
    with (
        patch("mfh.analysis.reporting.open_phase_prerequisite", side_effect=_phase_open_mock),
        patch(
            "mfh.analysis.derivation.derive_final_analysis_from_artifacts",
            return_value=changed,
        ),
        pytest.raises(FrozenArtifactError, match="differs"),
    ):
        verify_frozen_analysis_evidence(
            path,
            expected_protocol=protocol,
            study=study,
            phase_run_directories=phase_runs,
            robustness_result_directory=tmp_path / "robustness-results",
            human_audit_queue_directory=tmp_path / "audit-queue",
            human_audit_results_directory=tmp_path / "audit-results",
            human_audit_blinding_key=BLINDING_KEY,
            aa_official_directory=tmp_path / "aa-official",
            expected_aa_official_manifest_digest="f" * 64,
        )


def test_human_audit_rows_must_bind_to_the_actual_generation_record(tmp_path: Path) -> None:
    protocol = load_analysis_protocol(CONFIG)
    study = load_study_protocol(STUDY_CONFIG)
    results = _results(protocol)
    sources = _report_sources(tmp_path, results)
    audit = sources["adjudicated_final_labels"].path
    audit.write_text(
        audit.read_text(encoding="utf-8").replace(",M0,P0-neutral,", ",M1,P0-neutral,", 1),
        encoding="utf-8",
    )
    audit_queue, audit_results, verified_audit = _audit_evidence(tmp_path, results, sources)
    analysis_evidence = _frozen_analysis_evidence(tmp_path, protocol, study, results)

    with (
        patch("mfh.analysis.reporting.open_phase_prerequisite", side_effect=_phase_open_mock),
        patch(
            "mfh.analysis.reporting.verify_human_audit_results",
            return_value=verified_audit,
        ),
        pytest.raises(DataValidationError, match="frozen generation record"),
    ):
        write_final_analysis_bundle(
            tmp_path / "bad-audit",
            protocol=protocol,
            study=study,
            phase_run_directories=_all_phase_runs(tmp_path),
            results=results,
            analysis_evidence_directory=analysis_evidence,
            report_sources=sources,
            human_audit_queue_directory=audit_queue,
            human_audit_results_directory=audit_results,
            human_audit_blinding_key=BLINDING_KEY,
            derived_analysis=_derived(results),
        )


def test_final_analysis_bundle_rejects_a_wrong_phase_behind_a_claimed_e9_path(
    tmp_path: Path,
) -> None:
    protocol = load_analysis_protocol(CONFIG)
    study = load_study_protocol(STUDY_CONFIG)
    results = _results(protocol)
    wrong = Mock()
    wrong.verify_complete.return_value = _completion(ExperimentPhase.E8, "8", record_count=600)

    with (
        patch("mfh.analysis.reporting.open_phase_prerequisite", return_value=wrong),
        pytest.raises(FrozenArtifactError, match="different phase"),
    ):
        write_final_analysis_bundle(
            tmp_path / "wrong-phase",
            protocol=protocol,
            study=study,
            phase_run_directories=_all_phase_runs(tmp_path),
            results=results,
            analysis_evidence_directory=tmp_path / "unused-analysis-evidence",
            report_sources=_report_sources(tmp_path, results),
            human_audit_queue_directory=tmp_path / "unused-queue",
            human_audit_results_directory=tmp_path / "unused-results",
            human_audit_blinding_key=BLINDING_KEY,
            derived_analysis=_derived(results),
        )

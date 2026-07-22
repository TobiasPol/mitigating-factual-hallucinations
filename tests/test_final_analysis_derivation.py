from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from mfh.analysis.derivation import (
    _transition_decomposition,
    derive_final_analysis_from_artifacts,
    derive_final_analysis_results,
)
from mfh.analysis.human_audit import HumanAuditResults
from mfh.analysis.protocol import load_analysis_protocol
from mfh.contracts import GenerationRecord, Outcome, Runtime, TokenScope
from mfh.errors import DataValidationError
from mfh.evaluation.language import language_response_evidence
from mfh.evaluation.simpleqa import simpleqa_hedging_evidence
from mfh.experiments.protocol import ExperimentPhase
from mfh.experiments.robustness_results import (
    RQ1GeneralizationResult,
    VerifiedPromptParaphraseRecord,
)
from mfh.provenance import stable_hash

ROOT = Path(__file__).parents[1]
PROTOCOL = ROOT / "configs" / "analysis" / "confirmatory.yaml"
MODEL = "nvidia/Qwen3.6-27B-NVFP4"


def test_artifact_derivation_opens_custom_e3_through_prerequisite_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    opened: list[ExperimentPhase] = []

    class StopAfterE3(Exception):
        pass

    def open_source(
        _directory: Path,
        *,
        phase: ExperimentPhase,
        study: object,
        **_kwargs: object,
    ) -> object:
        del study
        opened.append(phase)
        if phase is ExperimentPhase.E6:
            raise StopAfterE3
        return SimpleNamespace(
            verify_complete=lambda: SimpleNamespace(
                phase=phase,
                completion_digest=stable_hash((phase.value, "completion")),
            ),
            records=lambda: (),
        )

    monkeypatch.setattr(
        "mfh.experiments.runner.open_phase_prerequisite",
        open_source,
    )
    phase_runs = {
        phase: tmp_path / phase.value
        for phase in (
            ExperimentPhase.E1,
            ExperimentPhase.E3,
            ExperimentPhase.E6,
            ExperimentPhase.E7,
            ExperimentPhase.E8,
            ExperimentPhase.E9,
            ExperimentPhase.E10,
        )
    }

    with pytest.raises(StopAfterE3):
        derive_final_analysis_from_artifacts(
            protocol=None,  # type: ignore[arg-type]
            study=None,  # type: ignore[arg-type]
            phase_run_directories=phase_runs,
            robustness_result_directory=tmp_path / "robustness",
            human_audit_queue_directory=tmp_path / "audit-queue",
            human_audit_results_directory=tmp_path / "audit-results",
            human_audit_blinding_key=b"0" * 32,
            aa_official_directory=tmp_path / "aa",
            expected_aa_official_manifest_digest="0" * 64,
        )
    assert opened == [ExperimentPhase.E1, ExperimentPhase.E3, ExperimentPhase.E6]


def _runtime_identity() -> dict[str, object]:
    return {
        "backend": "vllm",
        "vllm": "0.24.0",
        "transformers": "5.2.0",
        "torch": "2.11.0",
        "python": "3.12.0",
        "architecture": "x86_64",
        "os": "Linux test",
        "nvidia_driver": "570.00",
        "gpu_name": "NVIDIA A100-SXM4-40GB",
        "gpu_total_memory_bytes": 40_000_000_000,
        "cuda_capability": "8.0",
        "cuda_runtime": "12.9",
        "tensor_parallel_size": 1,
        "quantization_loader": "modelopt_mixed",
        "quantization_config_class": (
            "vllm.model_executor.layers.quantization.modelopt."
            "ModelOptMixedPrecisionConfig"
        ),
        "quantization_execution": "marlin-w4a16-fp8-weight-only-on-sm80",
        "model_class": (
            "vllm.model_executor.models.qwen3_5."
            "Qwen3_5ForConditionalGeneration"
        ),
        "tokenizer_class": "transformers.Tokenizer",
        "num_layers": 64,
        "hidden_size": 5_120,
        "seed": 17,
        "model_repository": MODEL,
        "model_revision": "a" * 40,
        "model_quantization": "modelopt-mixed-nvfp4-fp8",
        "model_num_layers": 64,
        "snapshot_sha256": "6" * 64,
        "research_provenance": {"study": "synthetic-test"},
        "research_toolchain": {
            "vllm": "0.24.0",
            "torch": "2.11.0",
            "transformers": "5.2.0",
            "numpy": "2.4.3",
            "nvidia_driver": "570.00",
        },
    }


def _record(
    *,
    phase: ExperimentPhase,
    benchmark: str,
    prompt: str,
    method: str,
    question: int,
    outcome: Outcome,
    metadata: dict[str, object] | None = None,
    layer: int | None = None,
    alpha: float = 0.0,
) -> GenerationRecord:
    identity = {
        "phase": phase.value,
        "benchmark": benchmark,
        "prompt": prompt,
        "method": method,
        "layer": layer,
        "alpha": alpha,
    }
    risk = 0.1 + question * 0.2
    scores = {"C": (1 - risk) * 0.75, "I": risk, "A": (1 - risk) * 0.25}
    raw_output = (
        f"Die Antwort ist answer-{question}."
        if benchmark == "language_consistency"
        else f"answer-{question}"
    )
    values: dict[str, object] = {
        "phase": phase.value,
        "partition": "synthetic-replay",
        "runtime_session_identity_sha256": stable_hash(_runtime_identity()),
        "adaptive_controller_evidence": {
            "prompt_feature_peak_memory_bytes": 4096,
            "site_selection": "max_mixed_direction_norm_then_site",
            "feature_schema_digest": "8" * 64,
            "controller_artifact_sha256": "7" * 64,
        },
        "execution_receipt_signature": "a" * 128,
        "confirmatory_execution_receipt_signature": "b" * 128,
        "generation_runtime_metrics": {
            "peak_memory_bytes": 8192,
            "candidate_peak_memory_bytes": 8192,
            "auxiliary_peak_memory_bytes": 4096,
            "active_memory_bytes": 4096,
            "cache_memory_bytes": 1024,
            "prompt_tokens_per_second": 100.0,
            "generation_tokens_per_second": 20.0,
            "candidate_generated": True,
            "candidate_generation_seconds": 0.008,
            "end_to_end_wall_seconds": 0.1,
        },
        "policy_action": "release",
        "selective_risk_evidence": {
            "score_semantics": "frozen-pre-generation-CIA-prompt-risk",
            "controller_artifact_sha256": "7" * 64,
            "controller_prompt_id": prompt,
            "feature_schema_digest": "8" * 64,
            "feature_values_sha256": stable_hash([risk]),
            "feature_values": [risk],
            "scores": scores,
            "predicted_hallucination_risk": risk,
        },
        **(metadata or {}),
    }
    if benchmark == "simpleqa_verified" and phase in {
        ExperimentPhase.E9,
        ExperimentPhase.E10,
    }:
        values["simpleqa_hedging_evidence"] = simpleqa_hedging_evidence(raw_output)
    if benchmark == "language_consistency":
        evidence = language_response_evidence(raw_output, "de", (f"answer-{question}",))
        values.update(
            {
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
            }
        )
    return GenerationRecord(
        question_id=f"{benchmark}-q{question:03d}",
        benchmark=benchmark,
        model_repository=MODEL,
        model_revision="a" * 40,
        runtime=Runtime.VLLM,
        quantization="4bit",
        system_prompt_id=prompt,
        rendered_prompt_hash=stable_hash((identity, question)),
        steering_method=method,
        layer=layer,
        token_scope=TokenScope.FINAL_PROMPT if layer is not None else None,
        alpha=alpha,
        sparsity=None,
        controller_scores=scores,
        raw_output=raw_output,
        normalized_answer=raw_output,
        outcome=outcome,
        generation_latency_seconds=0.1,
        input_tokens=8,
        output_tokens=2,
        condition_id=stable_hash(identity),
        metadata=values,
    )


def _outcome(method: str, question: int) -> Outcome:
    baseline = (Outcome.CORRECT, Outcome.INCORRECT, Outcome.CORRECT, Outcome.INCORRECT)
    if method == "M0":
        return baseline[question]
    if method == "M1":
        return (Outcome.CORRECT, Outcome.CORRECT, Outcome.CORRECT, Outcome.ABSTENTION)[question]
    if method == "M3":
        return (Outcome.CORRECT, Outcome.CORRECT, Outcome.ABSTENTION, Outcome.CORRECT)[question]
    return (Outcome.CORRECT, Outcome.ABSTENTION, Outcome.CORRECT, Outcome.CORRECT)[question]


def _e3_records() -> tuple[GenerationRecord, ...]:
    return tuple(
        _record(
            phase=ExperimentPhase.E3,
            benchmark="triviaqa",
            prompt="P0-neutral",
            method="M1",
            question=question,
            outcome=_outcome("M1", question),
            layer=21,
            alpha=1.0,
            metadata={
                "e3_stage": "alpha",
                "intervention_trace": {
                    "training_prompt_id": "P0-neutral",
                    "extraction_method": "M1-R",
                    "control": None,
                },
            },
        )
        for question in range(4)
    )


def _e1_records() -> tuple[GenerationRecord, ...]:
    outcomes = {
        "P0-neutral": (
            Outcome.CORRECT,
            Outcome.INCORRECT,
            Outcome.CORRECT,
            Outcome.INCORRECT,
        ),
        "P2-calibrated-abstention": (
            Outcome.CORRECT,
            Outcome.CORRECT,
            Outcome.INCORRECT,
            Outcome.INCORRECT,
        ),
    }
    return tuple(
        _record(
            phase=ExperimentPhase.E1,
            benchmark=benchmark,
            prompt=prompt,
            method="M0",
            question=question,
            outcome=outcomes[prompt][question],
        )
        for benchmark in (
            "triviaqa",
            "simpleqa_verified",
            "aa_omniscience_public_600",
        )
        for prompt in ("P0-neutral", "P2-calibrated-abstention")
        for question in range(4)
    )


def _e6_records() -> tuple[GenerationRecord, ...]:
    result: list[GenerationRecord] = []
    for benchmark in (
        "triviaqa",
        "simpleqa_verified",
        "aa_omniscience_public_600",
    ):
        for prompt in (
            "P0-neutral",
            "P2-calibrated-abstention",
            "P3-forced-answer",
        ):
            for method_index, method in enumerate(("M0", "M1", "M3")):
                for question in range(4):
                    result.append(
                        _record(
                            phase=ExperimentPhase.E6,
                            benchmark=benchmark,
                            prompt=prompt,
                            method=method,
                            question=question,
                            outcome=_outcome(method, question),
                            metadata={
                                "gold_alias_log_likelihood": -2.0 + 0.2 * method_index,
                                "abstention_log_likelihood": -3.0 + 0.1 * method_index,
                                "gold_answer_rank": 3 - method_index,
                            },
                        )
                    )
    return tuple(result)


def _side_metadata(benchmark: str, question: int) -> dict[str, object]:
    if benchmark == "ifeval":
        return {"official_pass": question % 2 == 0}
    if benchmark == "mmlu_pro":
        return {"official_correct": question % 2 == 0}
    if benchmark == "xstest":
        return {"refused": False}
    if benchmark == "strongreject_or_harmbench":
        return {"refused": True, "harmful_compliance_score": 0.0}
    if benchmark == "language_consistency":
        return {
            "requested_language": "de",
            "detected_language": "de",
            "requested_language_correct": True,
        }
    if benchmark == "wikitext103":
        return {"negative_log_likelihood": 1.0}
    return {}


def _e8_records() -> tuple[GenerationRecord, ...]:
    result: list[GenerationRecord] = []
    benchmarks = (
        "triviaqa",
        "ifeval",
        "mmlu_pro",
        "xstest",
        "strongreject_or_harmbench",
        "language_consistency",
        "wikitext103",
    )
    for benchmark in benchmarks:
        for prompt in ("P0-neutral", "P2-calibrated-abstention"):
            for method in ("M0", "M1", "M3", "M4", "M5"):
                for question in range(4):
                    outcome = (
                        _outcome(method, question)
                        if benchmark == "triviaqa"
                        else (
                            Outcome.CORRECT
                            if benchmark != "mmlu_pro" or question % 2 == 0
                            else Outcome.INCORRECT
                        )
                    )
                    result.append(
                        _record(
                            phase=ExperimentPhase.E8,
                            benchmark=benchmark,
                            prompt=prompt,
                            method=method,
                            question=question,
                            outcome=outcome,
                            metadata=_side_metadata(benchmark, question),
                        )
                    )
    return tuple(result)


def _e9_records() -> tuple[GenerationRecord, ...]:
    return tuple(
        _record(
            phase=ExperimentPhase.E9,
            benchmark=benchmark,
            prompt=prompt,
            method=method,
            question=question,
            outcome=_outcome(method, question),
            metadata=(
                {
                    "intervention_trace": {
                        "activation_delta_norm": {
                            "M1": 1.0,
                            "M3": 1.2,
                            "M4": 0.8,
                            "M5": 0.7,
                        }[method]
                    }
                }
                if method in {"M1", "M3", "M4", "M5"}
                else {"intervention_norm": 0.0}
            ),
        )
        for benchmark in (
            "triviaqa",
            "simpleqa_verified",
            "aa_omniscience_public_600",
        )
        for prompt in ("P0-neutral", "P2-calibrated-abstention")
        for method in ("M0", "M1", "M2", "M3", "M4", "M5")
        for question in range(4)
    )


def _e10_records() -> tuple[GenerationRecord, ...]:
    records: list[GenerationRecord] = []
    for benchmark in (
        "triviaqa",
        "simpleqa_verified",
        "aa_omniscience_public_600",
    ):
        for question, outcome in enumerate(
            (Outcome.CORRECT, Outcome.CORRECT, Outcome.INCORRECT, Outcome.ABSTENTION)
        ):
            metadata: dict[str, object] = {}
            if benchmark == "triviaqa":
                score = float(outcome is Outcome.CORRECT)
                metadata.update({"official_exact_match": score, "official_token_f1": score})
            records.append(
                _record(
                    phase=ExperimentPhase.E10,
                    benchmark=benchmark,
                    prompt="P0-neutral",
                    method="M6",
                    question=question,
                    outcome=outcome,
                    metadata=metadata,
                )
            )
    for benchmark in (
        "ifeval",
        "mmlu_pro",
        "wikitext103",
        "xstest",
        "strongreject_or_harmbench",
        "language_consistency",
    ):
        for question in range(4):
            records.append(
                _record(
                    phase=ExperimentPhase.E10,
                    benchmark=benchmark,
                    prompt="P0-neutral",
                    method="M6",
                    question=question,
                    outcome=(
                        Outcome.INCORRECT
                        if benchmark == "mmlu_pro" and question % 2
                        else Outcome.CORRECT
                    ),
                    metadata=_side_metadata(benchmark, question),
                )
            )
    graded: list[GenerationRecord] = []
    for record in records:
        if record.benchmark in {"simpleqa_verified", "aa_omniscience_public_600"}:
            graded.append(
                GenerationRecord.from_dict(
                    {
                        **record.to_dict(),
                        "metadata": {
                            **dict(record.metadata),
                            "grader_failed": False,
                            "official_grader_evidence": {
                                "attempt_receipts": [{"error_type": None}]
                            },
                        },
                    }
                )
            )
        else:
            graded.append(record)
    return tuple(graded)


def _bootstrap(comparison_id: str, estimate: float = 0.1) -> dict[str, object]:
    return {
        "comparison_id": comparison_id,
        "paired_bootstrap": {
            "difference": estimate,
            "lower": estimate - 0.05,
            "upper": estimate + 0.05,
            "two_sided_p_value": 0.01,
            "questions": 4,
        },
    }


def _e9_outputs() -> dict[str, dict[str, object]]:
    interaction_result = {
        "interaction": 0.02,
        "lower": -0.01,
        "upper": 0.05,
        "two_sided_p_value": 0.2,
        "questions": 4,
        "prompt_only_gain": 0.01,
        "steering_only_gain": 0.1,
        "combined_gain": 0.13,
    }
    return {
        "primary_contrasts.json": {
            "contrasts": {
                "RQ1": [_bootstrap("RQ1:test")],
                "RQ2": [_bootstrap("RQ2:test")],
                "RQ3": [
                    {
                        "comparison_id": "RQ3:test",
                        "risk_and_coverage_contrasts": [
                            _bootstrap("RQ3:test:risk"),
                            _bootstrap("RQ3:test:coverage", 0.0),
                        ],
                    }
                ],
                "RQ4": [{"comparison_id": "RQ4:test", "result": interaction_result}],
            }
        },
        "prompt_method_interactions.json": {
            "interactions": [{"comparison_id": "RQ4:test", "result": interaction_result}]
        },
        "mixed_effects.json": {
            "result": {
                "fixed_effect_names": ["Intercept"],
                "coefficients": [0.1],
                "standard_errors": [0.2],
            }
        },
        "holm_corrections.json": {
            "hypotheses": [
                {
                    "hypothesis": "primary:test",
                    "raw_p_value": 0.01,
                    "adjusted_p_value": 0.04,
                    "rejected": True,
                }
            ]
        },
        "condition_summaries.json": {"conditions": [{"question_count": 4}]},
    }


def _audit(tmp_path: Path) -> HumanAuditResults:
    return HumanAuditResults(
        directory=tmp_path,
        manifest_digest="1" * 64,
        queue_manifest_digest="2" * 64,
        scientific_eligible=True,
        summary={
            "factual_reporting_payload": {
                "agreement_metrics": {
                    "cohen_kappa": 1.0,
                    "krippendorff_alpha": 1.0,
                },
                "adjudication_summary": {"rows": 600, "disagreements": 0},
                "automated_human_confusion_matrix": {"C:C": 600},
                "record_binding_digest": "3" * 64,
            },
            "language_reporting_payload": {
                "agreement_metrics": {
                    "cohen_kappa": 1.0,
                    "krippendorff_alpha": 1.0,
                },
                "adjudication_summary": {"rows": 4, "disagreements": 0},
                "automated_human_confusion_matrix": {"CONSISTENT:CONSISTENT": 4},
                "human_consistency_score": {
                    "consistent": 4,
                    "judged": 4,
                    "rate": 1.0,
                },
                "record_binding_digest": "4" * 64,
            },
        },
    )


def _prompt_records() -> tuple[VerifiedPromptParaphraseRecord, ...]:
    return tuple(
        VerifiedPromptParaphraseRecord(
            task_id=f"task-{variant}-{question}",
            benchmark="triviaqa",
            base_prompt_id="P0-neutral",
            paraphrase_prompt_id=f"P0-paraphrase-{variant}",
            method="M1",
            record=_record(
                phase=ExperimentPhase.E9,
                benchmark="triviaqa",
                prompt=f"P0-paraphrase-{variant}",
                method="M1",
                question=question,
                outcome=Outcome.CORRECT,
            ),
        )
        for variant in range(2)
        for question in range(4)
    )


def _rq1_results() -> dict[str, RQ1GeneralizationResult]:
    return {
        regime: RQ1GeneralizationResult.create(
            task_id=f"rq1-{regime}",
            plan_digest="a" * 64,
            question_set_digests={"held_out_evaluation": "b" * 64},
            artifact_locations={"component": f"artifacts/{regime}"},
            artifact_fingerprints={"component": "c" * 64},
            evaluation_record_count=200,
            metrics={"accuracy": 0.75, "coverage": 0.8},
        )
        for regime in (
            "M3|calibration-only|fold-0|P0-neutral-to-P2-calibrated-abstention",
            "M3|full-vector-bank-relearning|fold-0|P0-neutral-to-P2-calibrated-abstention",
        )
    }


def _e7_results() -> dict[str, object]:
    return {
        "artifact_sha256": "d" * 64,
        "feature_stability": 0.9,
        "selected_feature_count": 2,
        "features": {
            "feature_1": {
                "activation_factuality_delta": 0.08,
                "suppression_factuality_delta": -0.05,
            }
        },
    }


def test_final_analysis_is_derived_from_raw_phase_records(tmp_path: Path) -> None:
    protocol = load_analysis_protocol(PROTOCOL)
    derived = derive_final_analysis_results(
        protocol=protocol,
        phase_records={
            "E1": _e1_records(),
            "E3": _e3_records(),
            "E6": _e6_records(),
            "E8": _e8_records(),
            "E9": _e9_records(),
            "E10": _e10_records(),
        },
        e9_analysis_outputs=_e9_outputs(),
        human_audit=_audit(tmp_path),
        prompt_paraphrase_records=_prompt_records(),
        rq1_generalization_results=_rq1_results(),
        e7_interpretability=_e7_results(),
        runtime_identity=_runtime_identity(),
        runtime_attestation_digest="5" * 64,
        aa_official_prompt_comparison={
            "comparison": "P-AA-official-vs-P0-neutral within M0",
            "paired_question_count": 600,
            "official": {
                "official_metrics": {
                    "omniscience_index": 85.0,
                    "accuracy": 0.87,
                },
                "unified_metrics": {"coverage": 0.92},
            },
            "neutral": {
                "official_metrics": {
                    "omniscience_index": 82.5,
                    "accuracy": 0.84,
                },
                "unified_metrics": {"coverage": 0.82},
            },
            "deltas": {
                "omniscience_index": 2.5,
                "coverage": 0.1,
                "hallucination_risk": -0.05,
            },
            "transition_counts": {"C->C": 500, "I->C": 60, "A->A": 40},
            "leaderboard_comparability": {
                "official_track": True,
                "neutral_controlled_track": False,
            },
        },
        aa_official_source_digest="6" * 64,
    )

    results = derived.results
    assert results.likelihood_changes["triviaqa|P0-neutral|M0_to_M3"]["gold"] == pytest.approx(0.4)
    assert results.language_confusion["requested_detected_matrix"] == {"de:de": 4}
    assert results.runtime_replication["local_vllm_execution"]["passed"] is True
    assert results.primary_contrasts["RQ1"]["comparisons"]["RQ1:test"]["estimate"] == 0.1
    assert len(results.noninferiority["ifeval_pass_rate"]["comparisons"]) == 8
    aa_prompt = results.prompt_interactions["AA|M0|P-AA-official-vs-P0-neutral"]
    assert aa_prompt["estimate"] == 2.5
    assert aa_prompt["source_digest"] == "6" * 64
    assert len(derived.source_record_digests) == 7
    forced = results.likelihood_changes["triviaqa|P3-forced-answer|M0_to_M3"]
    assert forced["forced_answer_condition"] is True
    assert forced["mean_gold_rank_change"] == -2.0
    matched = results.matched_coverage["triviaqa|P0-neutral|M3_vs_M1"]
    assert matched["intervention_norm_mismatch"] == pytest.approx(0.2)
    assert matched["latency_mismatch_seconds"] == 0.0
    assert "triviaqa|E1_P0_to_P2" in results.power_analysis
    assert len(results.rq1_generalization) == 2
    assert results.e7_interpretability["feature_stability"] == 0.9


def test_transition_matrices_exclude_and_report_unscorable_pairs() -> None:
    e9 = list(_e9_records())
    index = next(
        index
        for index, record in enumerate(e9)
        if record.benchmark == "simpleqa_verified"
        and record.system_prompt_id == "P0-neutral"
        and record.steering_method == "M2"
        and record.question_id.endswith("q000")
    )
    e9[index] = replace(e9[index], outcome=Outcome.UNSCORABLE)
    transitions = _transition_decomposition(tuple(e9), _e10_records())
    comparison = transitions["simpleqa_verified|P0-neutral|M0_to_M2"]
    assert comparison["paired_questions"] == 3
    assert comparison["unscorable_pairs_excluded"] == 1
    assert sum(comparison["transition_counts"].values()) == 3


def test_record_change_changes_the_derivation_identity(tmp_path: Path) -> None:
    protocol = load_analysis_protocol(PROTOCOL)
    common = {
        "protocol": protocol,
        "e9_analysis_outputs": _e9_outputs(),
        "human_audit": _audit(tmp_path),
        "prompt_paraphrase_records": _prompt_records(),
        "rq1_generalization_results": _rq1_results(),
        "e7_interpretability": _e7_results(),
        "runtime_identity": _runtime_identity(),
        "runtime_attestation_digest": "5" * 64,
    }
    phases = {
        "E1": _e1_records(),
        "E3": _e3_records(),
        "E6": _e6_records(),
        "E8": _e8_records(),
        "E9": _e9_records(),
        "E10": _e10_records(),
    }
    first = derive_final_analysis_results(phase_records=phases, **common)
    changed = list(phases["E6"])
    row = changed[0]
    changed[0] = GenerationRecord.from_dict(
        {
            **row.to_dict(),
            "metadata": {
                **dict(row.metadata),
                "gold_alias_log_likelihood": -1.75,
            },
        }
    )
    second = derive_final_analysis_results(phase_records={**phases, "E6": tuple(changed)}, **common)
    assert first.derivation_digest != second.derivation_digest
    assert first.results.likelihood_changes != second.results.likelihood_changes


def test_selective_risk_scores_must_be_identical_across_methods(tmp_path: Path) -> None:
    e9 = list(_e9_records())
    index = next(
        index
        for index, record in enumerate(e9)
        if record.benchmark == "triviaqa"
        and record.system_prompt_id == "P0-neutral"
        and record.steering_method == "M1"
        and record.question_id.endswith("q000")
    )
    record = e9[index]
    evidence = dict(record.metadata["selective_risk_evidence"])
    changed_scores = {"C": 0.05, "I": 0.9, "A": 0.05}
    evidence.update(
        {
            "scores": changed_scores,
            "predicted_hallucination_risk": changed_scores["I"],
        }
    )
    e9[index] = GenerationRecord.from_dict(
        {
            **record.to_dict(),
            "controller_scores": changed_scores,
            "metadata": {
                **dict(record.metadata),
                "selective_risk_evidence": evidence,
            },
        }
    )

    with pytest.raises(DataValidationError, match="differs across methods"):
        derive_final_analysis_results(
            protocol=load_analysis_protocol(PROTOCOL),
            phase_records={
                "E1": _e1_records(),
                "E3": _e3_records(),
                "E6": _e6_records(),
                "E8": _e8_records(),
                "E9": tuple(e9),
                "E10": _e10_records(),
            },
            e9_analysis_outputs=_e9_outputs(),
            human_audit=_audit(tmp_path),
            prompt_paraphrase_records=_prompt_records(),
            rq1_generalization_results=_rq1_results(),
            e7_interpretability=_e7_results(),
            runtime_identity=_runtime_identity(),
            runtime_attestation_digest="5" * 64,
        )


def test_final_derivation_accepts_the_verified_compact_e3_surface(tmp_path: Path) -> None:
    protocol = load_analysis_protocol(PROTOCOL)
    common = {
        "protocol": protocol,
        "e9_analysis_outputs": _e9_outputs(),
        "human_audit": _audit(tmp_path),
        "prompt_paraphrase_records": _prompt_records(),
        "rq1_generalization_results": _rq1_results(),
        "e7_interpretability": _e7_results(),
        "runtime_identity": _runtime_identity(),
        "runtime_attestation_digest": "5" * 64,
    }
    full = derive_final_analysis_results(
        phase_records={
            "E1": _e1_records(),
            "E3": _e3_records(),
            "E6": _e6_records(),
            "E8": _e8_records(),
            "E9": _e9_records(),
            "E10": _e10_records(),
        },
        **common,
    )
    e3_digest = stable_hash("verified-e3-completion")
    compact = derive_final_analysis_results(
        phase_records={
            "E1": _e1_records(),
            "E6": _e6_records(),
            "E8": _e8_records(),
            "E9": _e9_records(),
            "E10": _e10_records(),
        },
        e3_analysis_surface=full.results.layer_alpha_surface,
        e3_source_digest=e3_digest,
        **common,
    )

    assert compact.results.layer_alpha_surface == full.results.layer_alpha_surface
    assert compact.source_record_digests["E3"] == e3_digest

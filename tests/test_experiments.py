from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from mfh.analysis.protocol import load_analysis_protocol
from mfh.config import load_model_spec, load_prompt_specs
from mfh.contracts import (
    ActivationSite,
    AdaptivePolicySpec,
    GenerationRecord,
    InterventionSpec,
    Outcome,
    Question,
    TokenScope,
)
from mfh.data.io import read_questions, write_questions
from mfh.data.reviewed_splits import (
    VerifiedReviewedSplits,
    authorize_reviewed_split_bundle,
    write_reviewed_split_bundle,
)
from mfh.data.source_snapshots import SOURCE_SNAPSHOTS, iter_source_questions
from mfh.data.splits import SplitPlan
from mfh.errors import ConfigurationError, DataValidationError, FrozenArtifactError
from mfh.experiments.e0_completion import (
    VerifiedE0CompletionReceipt,
    authorize_e0_completion_receipt,
)
from mfh.experiments.evidence import GateResult
from mfh.experiments.gates import (
    GateEvaluationContext,
    evaluate_gate,
    write_gate_evidence,
)
from mfh.experiments.protocol import ExperimentPhase, StudyProtocol, load_study_protocol
from mfh.experiments.runner import (
    EvaluationCondition,
    PhaseCompletion,
    PhaseRunContract,
    PhaseRunLedger,
    _side_effect_schedule_digest,
    adaptive_policy_decision_digest,
    expand_factorial_conditions,
    package_portable_phase_ledger,
    sign_adaptive_execution_receipt,
    validate_adaptive_execution,
    write_frozen_question_bundle,
)
from mfh.provenance import sha256_file, sha256_path, stable_hash

ROOT = Path(__file__).parents[1]
MODEL_CONFIGS = {
    "qwen3.6-27b-mlx-4bit": ROOT / "configs/models/qwen3.6-27b-mlx-4bit.yaml",
}
_EXECUTION_PRIVATE_KEY = "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60"
_EXECUTION_PUBLIC_KEY = "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"


def _allow_unit_test_artifact_paths(paths: object) -> dict[str, Path]:
    """Keep legacy ledger unit fixtures isolated from the live study namespace."""

    assert isinstance(paths, dict)
    return {
        str(name): Path(value).absolute().resolve(strict=False)
        for name, value in paths.items()
    }


def _adaptive_policy() -> AdaptivePolicySpec:
    return AdaptivePolicySpec(
        release_risk_threshold=0.4,
        abstention_probability_threshold=0.7,
        alpha_max=1.5,
        alpha_beta=8.0,
        layer=1,
        site=ActivationSite.POST_MLP,
        token_scope=TokenScope.FIRST_FOUR,
        direction_sha256="d" * 64,
        direction_norm=1.0,
        execution_public_key=_EXECUTION_PUBLIC_KEY,
    )


def _adaptive_policy_v2() -> AdaptivePolicySpec:
    return AdaptivePolicySpec(
        schema_version=2,
        release_risk_threshold=0.4,
        abstention_probability_threshold=0.7,
        likely_unknown_risk_threshold=0.8,
        alpha_max=1.5,
        alpha_beta=8.0,
        layer=None,
        site=None,
        token_scope=None,
        direction_sha256=None,
        direction_norm=None,
        execution_public_key=_EXECUTION_PUBLIC_KEY,
        controller_artifact_sha256="e" * 64,
        candidate_layers=(1, 2),
        candidate_sites=(ActivationSite.POST_MLP, ActivationSite.BLOCK_OUTPUT),
        candidate_token_scopes=(TokenScope.FIRST_FOUR,),
        vector_count=4,
        alpha_mode="risk_gated",
        alpha_risk_threshold=0.4,
    )


def _study() -> StudyProtocol:
    return load_study_protocol(ROOT / "configs/experiments/phases.yaml")


def _models(*names: str):  # type: ignore[no-untyped-def]
    return {name: load_model_spec(MODEL_CONFIGS[name]) for name in names}


def _prompts():  # type: ignore[no-untyped-def]
    return {
        value.prompt_id: value for value in load_prompt_specs(ROOT / "configs/prompts/primary.yaml")
    }


def _e0_conditions(study: StudyProtocol) -> tuple[EvaluationCondition, ...]:
    prompt = _prompts()["P0-neutral"]
    prompt_hash = hashlib.sha256(prompt.text.encode()).hexdigest()
    values: list[EvaluationCondition] = []
    for name, model in _models(*MODEL_CONFIGS).items():
        values.append(
            EvaluationCondition(
                phase=ExperimentPhase.E0,
                benchmark="shared_benign_factual_500",
                partition="runtime-validation",
                model_name=name,
                model_repository=model.repository,
                model_revision=model.revision,
                runtime=model.runtime,
                quantization=model.quantization,
                model_num_layers=model.num_layers,
                system_prompt_id=prompt.prompt_id,
                prompt_template_sha256=prompt_hash,
                steering_method="M0",
                method_artifact_sha256=None,
                layer=None,
                site=None,
                token_scope=None,
                alpha=0.0,
                sparsity=None,
                seed=17,
                study_protocol_digest=study.digest,
            )
        )
    return tuple(values)


def _record(condition: EvaluationCondition, question_id: str) -> GenerationRecord:
    return GenerationRecord(
        question_id=question_id,
        benchmark=condition.benchmark,
        model_repository=condition.model_repository,
        model_revision=condition.model_revision,
        runtime=condition.runtime,
        quantization=condition.quantization,
        system_prompt_id=condition.system_prompt_id,
        rendered_prompt_hash="9" * 64,
        steering_method=condition.steering_method,
        layer=condition.layer,
        site=condition.site,
        token_scope=condition.token_scope,
        alpha=condition.alpha,
        sparsity=condition.sparsity,
        controller_scores={},
        raw_output="answer",
        normalized_answer="answer",
        outcome=Outcome.CORRECT,
        generation_latency_seconds=0.01,
        input_tokens=5,
        output_tokens=1,
        condition_id=condition.condition_id,
        seed=condition.seed,
        metadata={
            "phase": condition.phase.value,
            "partition": condition.partition,
            "prompt_template_sha256": condition.prompt_template_sha256,
            "study_protocol_digest": condition.study_protocol_digest,
        },
    )


def _input_artifacts(directory: Path, names: tuple[str, ...]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for index, name in enumerate(names):
        path = directory / f"{name}.txt"
        path.write_text(f"frozen {name} {index}\n", encoding="utf-8")
        result[name] = path
    return result


def _e0_contract(study: StudyProtocol, input_paths: dict[str, Path]) -> PhaseRunContract:
    phase = study.phase("E0")
    return PhaseRunContract(
        phase=ExperimentPhase.E0,
        study_protocol_digest=study.digest,
        conditions=_e0_conditions(study),
        question_ids_by_benchmark={
            "shared_benign_factual_500": tuple(f"q-{index}" for index in range(500))
        },
        input_fingerprints={name: sha256_path(path) for name, path in input_paths.items()},
        prerequisite_digests={},
        required_gates=phase.gates,
    )


def _e10_contract(study: StudyProtocol, prompt_id: str) -> PhaseRunContract:
    phase = study.phase("E10")
    prompt = _prompts()[prompt_id]
    prompt_hash = hashlib.sha256(prompt.text.encode()).hexdigest()
    partitions = {
        "triviaqa": "T-test",
        "simpleqa_verified": "simpleqa-eval",
        "aa_omniscience_public_600": "aa-eval",
        **{
            benchmark: "side-effect-eval"
            for benchmark in (
                "ifeval",
                "mmlu_pro",
                "wikitext103",
                "xstest",
                "strongreject_or_harmbench",
                "language_consistency",
            )
        },
    }
    counts = {
        "triviaqa": 5_000,
        "simpleqa_verified": 1_000,
        "aa_omniscience_public_600": 600,
        "ifeval": 541,
        "mmlu_pro": 1_000,
        "wikitext103": 1_000,
        "xstest": 250,
        "strongreject_or_harmbench": 313,
        "language_consistency": 500,
    }
    conditions = tuple(
        EvaluationCondition(
            phase=ExperimentPhase.E10,
            benchmark=benchmark,
            partition=partitions[benchmark],
            model_name=model_name,
            model_repository=model.repository,
            model_revision=model.revision,
            runtime=model.runtime,
            quantization=model.quantization,
            model_num_layers=model.num_layers,
            system_prompt_id=prompt_id,
            prompt_template_sha256=prompt_hash,
            steering_method="M6",
            method_artifact_sha256="6" * 64,
            layer=None,
            site=None,
            token_scope=None,
            alpha=0.0,
            sparsity=None,
            seed=17,
            study_protocol_digest=study.digest,
            adaptive_policy=_adaptive_policy(),
        )
        for model_name, model in _models("qwen3.6-27b-mlx-4bit").items()
        for benchmark in phase.benchmarks
    )
    return PhaseRunContract(
        phase=ExperimentPhase.E10,
        study_protocol_digest=study.digest,
        conditions=conditions,
        question_ids_by_benchmark={
            benchmark: tuple(f"{benchmark}-{index}" for index in range(count))
            for benchmark, count in counts.items()
        },
        input_fingerprints={
            name: "a" * 64 for name in set(phase.required_inputs) | set(phase.freeze_fields)
        },
        prerequisite_digests={value.value: "b" * 64 for value in phase.prerequisites},
        required_gates=phase.gates,
    )


def _gate_results(ledger: PhaseRunLedger, directory: Path) -> dict[str, GateResult]:
    directory.mkdir()
    records = tuple(ledger.records())
    repeated = records
    observations = {
        "checkpoint_identity": [],
        "deterministic_decode": [
            {
                "condition_id": record.condition_id,
                "question_id": record.question_id,
                "first_output_sha256": stable_hash(record.raw_output),
                "repeat_output_sha256": stable_hash(record.raw_output),
            }
            for record in repeated
        ],
        "chat_template_identity": [],
        "mlx_runtime_identity": [],
    }
    results: dict[str, GateResult] = {}
    for gate in ledger.contract.required_gates:
        path = directory / f"{gate}.json"
        write_gate_evidence(
            path,
            phase=ledger.contract.phase,
            gate=gate,
            contract_digest=ledger.contract.digest,
            record_set_digest=ledger.record_set_digest(),
            observations=observations[gate],
        )
        results[gate] = ledger.evaluate_gate(gate, path)
    return results


def _scientific_e0_receipt(directory: Path) -> tuple[Path, str]:
    root = directory / "E0-scientific-receipt"
    root.mkdir()
    body = {
        "schema_version": 1,
        "phase": "E0",
        "scope": "scientific-runtime-validation-after-manual-contamination-review",
        "source_manifests": {
            "mlx_runtime": "a" * 64,
            "contamination_review": "b" * 64,
            "contamination_review_queue": "c" * 64,
            "runtime_validation_cohort": "d" * 64,
        },
        "mlx_plan_identity": "e" * 64,
        "review_counts": {
            "reviewed": 200,
            "overlap": 0,
            "distinct": 200,
            "new_source_exclusions": 0,
            "reviewed_clean_source": 1_000,
        },
        "cohort_assessment": {
            "question_count": 500,
            "question_ids_sha256": "f" * 64,
            "manual_overlap_source_ids_sha256": "0" * 64,
            "manual_overlap_source_count": 0,
            "affected_cohort_ids": [],
            "affected_cohort_count": 0,
        },
        "status": "complete",
        "scientific_eligible": True,
        "e1_admission": "allowed-after-independent-receipt-verification",
    }
    receipt = {**body, "receipt_digest": stable_hash(body)}
    (root / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_body = {
        "schema_version": 1,
        "phase": "E0",
        "purpose": "scientific-e0-completion-receipt",
        "receipt_digest": receipt["receipt_digest"],
        "artifact": {
            "sha256": sha256_file(root / "receipt.json"),
            "size_bytes": (root / "receipt.json").stat().st_size,
        },
        "status": "complete",
        "scientific_eligible": True,
    }
    manifest = {**manifest_body, "manifest_digest": stable_hash(manifest_body)}
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root.resolve(), str(manifest["manifest_digest"])


def _verified_scientific_e0_receipt(directory: Path) -> VerifiedE0CompletionReceipt:
    receipt, manifest_digest = _scientific_e0_receipt(directory)
    with patch(
        "mfh.experiments.e0_completion.verify_e0_completion_receipt",
        return_value={"manifest_digest": manifest_digest},
    ):
        return authorize_e0_completion_receipt(
            receipt,
            expected_manifest_digest=manifest_digest,
            mlx_directory=directory / "mlx",
            expected_mlx_manifest_digest="a" * 64,
            expected_mlx_plan_identity="e" * 64,
            mlx_inputs={},
            review_result_directory=directory / "review-result",
            expected_review_result_manifest_digest="b" * 64,
            review_queue_directory=directory / "review-queue",
            expected_review_queue_manifest_digest="c" * 64,
            review_inputs={},
        )


def _verified_reviewed_splits(
    directory: Path,
    *,
    review_digest: str = "b" * 64,
    suffix: str = "",
) -> VerifiedReviewedSplits:
    directory = directory.resolve()
    review_result = directory / f"reviewed-split-source{suffix}"
    review_result.mkdir()
    source = review_result / "reviewed-clean-source.jsonl"
    write_questions(
        source,
        tuple(
            Question(
                question_id=f"reviewed-{index}",
                benchmark="triviaqa",
                text=f"Reviewed question {index}?",
                aliases=(f"reviewed answer {index}",),
                split="train",
            )
            for index in range(20)
        ),
    )
    review_manifest = {"reviewed_clean_source_sha256": sha256_file(source)}
    simpleqa_source = directory / f"simpleqa-source{suffix}.jsonl"
    write_questions(
        simpleqa_source,
        tuple(
            Question(
                question_id=f"s-{index}",
                benchmark="simpleqa_verified",
                text=f"SimpleQA question {index}?",
                aliases=(f"simpleqa answer {index}",),
                split="eval",
            )
            for index in range(1_000)
        ),
    )
    aa_source = directory / f"aa-source{suffix}.jsonl"
    write_questions(
        aa_source,
        tuple(
            Question(
                question_id=f"a-{index}",
                benchmark="aa_omniscience_public_600",
                text=f"AA question {index}?",
                aliases=(f"aa answer {index}",),
                split="eval",
            )
            for index in range(600)
        ),
    )
    plan = SplitPlan(steer=4, controller=3, dev=3, test=3, seed=17)
    output = directory / f"reviewed-splits{suffix}"
    verification_inputs = {
        "review_result_directory": review_result,
        "expected_review_result_manifest_digest": review_digest,
        "review_queue_directory": directory / "review-queue",
        "expected_review_queue_manifest_digest": "c" * 64,
        "review_inputs": {"target_sources": (simpleqa_source, aa_source)},
        "plan": plan,
    }
    with patch(
        "mfh.data.reviewed_splits.verify_contamination_review_result",
        return_value=review_manifest,
    ):
        manifest = write_reviewed_split_bundle(output, **verification_inputs)
        return authorize_reviewed_split_bundle(
            output,
            expected_manifest_digest=manifest["manifest_digest"],
            **verification_inputs,
        )


def _complete_e0(directory: Path) -> tuple[Path, PhaseCompletion]:
    study = _study()
    inputs = _input_artifacts(directory, study.phase("E0").required_inputs)
    contract = _e0_contract(study, inputs)
    path = directory / "E0-run"
    ledger = PhaseRunLedger.create(
        path,
        contract,
        study=study,
        input_artifacts=inputs,
        prerequisite_runs={},
    )
    ledger.checkpoint(
        _record(condition, question_id)
        for condition in contract.conditions
        for question_id in contract.question_ids_by_benchmark[condition.benchmark]
    )
    receipt = _verified_scientific_e0_receipt(directory)
    return path, ledger.finalize(
        _gate_results(ledger, directory / "E0-gate-evidence"),
        verified_e0_completion=receipt,
    )


class StudyProtocolTests(unittest.TestCase):
    def test_side_effect_schedule_digest_excludes_self_referential_inputs(self) -> None:
        study = _study()
        condition = replace(
            _e0_conditions(study)[0],
            phase=ExperimentPhase.E8,
            benchmark="language_consistency",
            partition="side-effect-eval",
        )
        provisional = PhaseRunContract(
            phase=ExperimentPhase.E8,
            study_protocol_digest=study.digest,
            conditions=(condition,),
            question_ids_by_benchmark={"language_consistency": ("language:1",)},
            input_fingerprints={"frozen_side_effect_scorers": "a" * 64},
            prerequisite_digests={"E7": "b" * 64},
            required_gates=("provisional",),
        )
        final = replace(
            provisional,
            input_fingerprints={"frozen_side_effect_scorers": "c" * 64},
            prerequisite_digests={"E7": "d" * 64},
            required_gates=("final",),
        )
        self.assertNotEqual(provisional.digest, final.digest)
        self.assertEqual(
            _side_effect_schedule_digest(provisional),
            _side_effect_schedule_digest(final),
        )
        m1 = replace(
            condition,
            steering_method="M1",
            method_artifact_sha256="e" * 64,
            layer=1,
            site=ActivationSite.POST_MLP,
            token_scope=TokenScope.FIRST_FOUR,
            alpha=0.5,
        )
        m1_contract = replace(provisional, conditions=(m1,))
        artifact_only = replace(
            m1_contract,
            conditions=(replace(m1, method_artifact_sha256="f" * 64),),
        )
        self.assertEqual(
            _side_effect_schedule_digest(m1_contract),
            _side_effect_schedule_digest(artifact_only),
        )
        schedule_mutations = (
            replace(provisional, conditions=(condition, m1)),
            replace(m1_contract, conditions=(replace(m1, layer=2),)),
            replace(m1_contract, conditions=(replace(m1, alpha=1.0),)),
            replace(m1_contract, conditions=(replace(m1, sparsity=0.1),)),
            replace(
                m1_contract,
                conditions=(replace(m1, comparison_group="different"),),
            ),
            replace(
                provisional,
                conditions=(
                    replace(
                        condition,
                        steering_method="M3",
                        method_artifact_sha256="3" * 64,
                        adaptive_policy=_adaptive_policy(),
                    ),
                ),
            ),
        )
        for mutated in schedule_mutations:
            with self.subTest(mutated=mutated.conditions[0].steering_method):
                self.assertNotEqual(
                    _side_effect_schedule_digest(provisional),
                    _side_effect_schedule_digest(mutated),
                )

    def setUp(self) -> None:
        self._namespace_patcher = patch(
            "mfh.experiments.runner.validate_active_study_artifact_paths",
            side_effect=_allow_unit_test_artifact_paths,
        )
        self._namespace_patcher.start()

    def tearDown(self) -> None:
        self._namespace_patcher.stop()

    def test_terminal_ledger_package_reopens_without_external_input_paths(self) -> None:
        study = _study()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original, completion = _complete_e0(root)
            portable = root / "portable-E0"
            fingerprint = package_portable_phase_ledger(
                original, portable, study=study
            )
            reopened = PhaseRunLedger.open(portable, study=study).verify_complete()
            self.assertEqual(reopened.completion_digest, completion.completion_digest)
            self.assertEqual(sha256_path(portable), fingerprint)
            packaged_input = next((portable / "portable-inputs").iterdir())
            if packaged_input.is_dir():
                packaged_file = next(
                    value for value in packaged_input.rglob("*") if value.is_file()
                )
            else:
                packaged_file = packaged_input
            packaged_file.write_bytes(packaged_file.read_bytes() + b"tamper")
            with self.assertRaisesRegex(FrozenArtifactError, "input"):
                PhaseRunLedger.open(portable, study=study)

    def test_exact_e0_e10_contract_and_readiness(self) -> None:
        study = _study()
        self.assertEqual(tuple(item.phase for item in study.phases), tuple(ExperimentPhase))
        self.assertEqual(study.phase("E0").question_limit, 500)
        self.assertEqual(study.phase("E4").question_limit, 2_000)
        self.assertTrue(study.phase("E10").one_shot)
        with self.assertRaises(DataValidationError):
            study.assert_ready("E5", {"E2": "a" * 64})
        with self.assertRaises(DataValidationError):
            study.assert_ready("E5", {"E2": "z" * 64, "E3": "b" * 64, "E4": "c" * 64})
        study.assert_ready("E5", {"E2": "a" * 64, "E3": "b" * 64, "E4": "c" * 64})
        with self.assertRaises(DataValidationError):
            study.assert_frozen_inputs("E10", {"E9_results": "a" * 64})

    def test_e10_rejects_diagnostic_and_benchmark_specific_prompts(self) -> None:
        study = _study()
        _e10_contract(study, "P2-calibrated-abstention").assert_matches_study(study)
        for prompt_id in ("P3-forced-answer", "P-AA-official"):
            with (
                self.subTest(prompt_id=prompt_id),
                self.assertRaisesRegex(DataValidationError, "deployment-eligible prompt"),
            ):
                _e10_contract(study, prompt_id).assert_matches_study(study)

    def test_contract_rejects_model_layer_count_drift(self) -> None:
        study = _study()
        contract = _e0_contract(
            study,
            {
                name: ROOT / "configs/experiments/phases.yaml"
                for name in study.phase("E0").required_inputs
            },
        )
        first_condition, *remaining = contract.conditions
        drifted = replace(
            contract,
            conditions=(replace(first_condition, model_num_layers=999), *remaining),
        )
        with self.assertRaisesRegex(DataValidationError, "model identity differs"):
            drifted.assert_matches_study(study)

    def test_factorial_expansion_matches_confirmatory_matrix(self) -> None:
        study = _study()
        models = _models("qwen3.6-27b-mlx-4bit")
        prompts = _prompts()
        partitions = {
            "triviaqa": "T-test",
            "simpleqa_verified": "simpleqa-eval",
            "aa_omniscience_public_600": "aa-eval",
        }
        fixed = {
            method: InterventionSpec(
                method=method,
                layer=21,
                site=ActivationSite.POST_MLP,
                token_scope=TokenScope.FIRST_FOUR,
                alpha=1.0,
                sparsity=0.1 if method == "M4" else None,
                artifact_sha256=character * 64,
            )
            for method, character in (("M1", "1"), ("M2", "2"), ("M4", "4"), ("M5", "5"))
        }
        interventions = {
            "M0": InterventionSpec(method="M0"),
            "M3": InterventionSpec(
                method="M3",
                artifact_sha256="3" * 64,
                adaptive_policy=_adaptive_policy(),
            ),
            **fixed,
        }
        conditions = expand_factorial_conditions(
            study,
            "E9",
            models=models,
            prompts=prompts,
            benchmark_partitions=partitions,
            interventions=interventions,
        )
        self.assertEqual(len(conditions), 1 * 3 * 3 * 6)
        phase = study.phase("E9")
        contract = PhaseRunContract(
            phase=ExperimentPhase.E9,
            study_protocol_digest=study.digest,
            conditions=conditions,
            question_ids_by_benchmark={
                "triviaqa": tuple(f"t-{index}" for index in range(5_000)),
                "simpleqa_verified": tuple(f"s-{index}" for index in range(1_000)),
                "aa_omniscience_public_600": tuple(f"a-{index}" for index in range(600)),
            },
            input_fingerprints={name: "a" * 64 for name in phase.required_inputs},
            prerequisite_digests={name.value: "b" * 64 for name in phase.prerequisites},
            required_gates=phase.gates,
        )
        contract.assert_matches_study(study)
        self.assertEqual(contract.expected_record_count, 118_800)

        fake = replace(
            contract,
            conditions=(conditions[0],),
            question_ids_by_benchmark={"triviaqa": ("one-record",)},
        )
        with self.assertRaises(DataValidationError):
            fake.assert_matches_study(study)

        language_source = ROOT / "configs/benchmarks/language-consistency-v1.json"
        snapshot = SOURCE_SNAPSHOTS["language_consistency"]
        question = next(iter_source_questions(snapshot, language_source))
        language_condition = replace(
            conditions[0],
            benchmark="language_consistency",
            partition="side-effect-eval",
        )
        short_contract = replace(
            contract,
            conditions=(language_condition,),
            question_ids_by_benchmark={"language_consistency": (question.question_id,)},
        )
        questions = {"language_consistency": (question,)}
        source_artifacts = {"language_consistency": language_source}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(
                DataValidationError,
                msg="the arithmetic smoke suite must not enter a confirmatory bundle",
            ):
                write_frozen_question_bundle(
                    Path(directory) / "confirmatory-bundle",
                    short_contract,
                    questions,
                    source_artifacts=source_artifacts,
                )
            development_contract = replace(
                short_contract,
                phase=ExperimentPhase.E8,
                conditions=(replace(language_condition, phase=ExperimentPhase.E8),),
            )
            bundle = Path(directory) / "development-bundle"
            self.assertEqual(
                len(
                    write_frozen_question_bundle(
                        bundle,
                        development_contract,
                        questions,
                        source_artifacts=source_artifacts,
                    )
                ),
                64,
            )
            invalid = {"language_consistency": (replace(question, text="Invented"),)}
            with self.assertRaises(DataValidationError, msg="arbitrary IDs must be rejected"):
                write_frozen_question_bundle(
                    Path(directory) / "invalid-bundle",
                    development_contract,
                    invalid,
                    source_artifacts=source_artifacts,
                )

    def test_adaptive_records_bind_and_exercise_real_policy_actions(self) -> None:
        study = _study()
        condition = replace(
            _e0_conditions(study)[0],
            steering_method="M3",
            method_artifact_sha256="3" * 64,
            adaptive_policy=_adaptive_policy(),
        )

        def adaptive_record(question_id: str, action: str) -> GenerationRecord:
            geometry = action == "intervene"
            scores = {"C": 0.2, "I": 0.7, "A": 0.1} if geometry else {"C": 0.7, "I": 0.2, "A": 0.1}
            policy = condition.adaptive_policy
            assert policy is not None
            alpha = (
                policy.alpha_max
                / (1 + math.exp(-policy.alpha_beta * (scores["I"] - policy.release_risk_threshold)))
                if geometry
                else 0.0
            )
            trace = (
                {
                    "layer": policy.layer,
                    "site": policy.site.value,
                    "token_scope": policy.token_scope.value,
                    "alpha": alpha,
                    "sparsity": policy.sparsity,
                    "applied_tokens": 1,
                    "applied_token_indices": [0],
                    "activation_delta_norm": alpha,
                    "direction_sha256": policy.direction_sha256,
                    "pre_activation_sha256": "a" * 64,
                    "post_activation_sha256": "b" * 64,
                    "delta_sha256": "c" * 64,
                }
                if geometry
                else None
            )
            draft = replace(
                _record(condition, question_id),
                layer=policy.layer if geometry else None,
                site=policy.site if geometry else None,
                token_scope=policy.token_scope if geometry else None,
                alpha=alpha,
                controller_scores=scores,
                metadata={
                    **_record(condition, question_id).metadata,
                    "method_artifact_sha256": "3" * 64,
                    "policy_action": action,
                    "intervention_trace": trace,
                    "intervention_trace_digest": (
                        stable_hash(trace) if trace is not None else None
                    ),
                },
            )
            decided = replace(
                draft,
                metadata={
                    **draft.metadata,
                    "policy_decision_digest": adaptive_policy_decision_digest(
                        draft,
                        policy=policy,
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
                        policy=policy,
                        private_key_hex=_EXECUTION_PRIVATE_KEY,
                    ),
                },
            )

        released = adaptive_record("q-release", "release")
        intervened = adaptive_record("q-intervene", "intervene")
        condition.validate_record(released)
        condition.validate_record(intervened)
        validate_adaptive_execution((released, intervened))
        with self.assertRaises(
            DataValidationError,
            msg="a signed receipt cannot be replayed with a different normalized answer",
        ):
            condition.validate_record(
                replace(intervened, normalized_answer="forged normalized answer")
            )
        with self.assertRaises(DataValidationError, msg="all-zero M3 must not pass as adaptive"):
            validate_adaptive_execution((released, adaptive_record("q-release-2", "release")))
        with self.assertRaises(DataValidationError, msg="decision digest must bind the action"):
            condition.validate_record(
                replace(
                    intervened,
                    metadata={**intervened.metadata, "policy_decision_digest": "f" * 64},
                )
            )
        microscopic = replace(intervened, alpha=5e-324)
        microscopic = replace(
            microscopic,
            metadata={
                **microscopic.metadata,
                "policy_decision_digest": adaptive_policy_decision_digest(
                    microscopic,
                    policy=_adaptive_policy(),
                    policy_action="intervene",
                ),
            },
        )
        with self.assertRaises(DataValidationError, msg="numerical no-ops are not interventions"):
            condition.validate_record(microscopic)

    def test_adaptive_policy_v2_is_strict_and_v1_remains_byte_compatible(self) -> None:
        legacy = _adaptive_policy()
        legacy_body = legacy.to_dict()
        self.assertEqual(AdaptivePolicySpec.from_dict(legacy_body).to_dict(), legacy_body)
        self.assertEqual(
            list(legacy_body),
            [
                "schema_version",
                "release_risk_threshold",
                "abstention_probability_threshold",
                "alpha_max",
                "alpha_beta",
                "layer",
                "site",
                "token_scope",
                "direction_sha256",
                "direction_norm",
                "execution_public_key",
                "sparsity",
            ],
        )

        routed = _adaptive_policy_v2()
        routed_body = routed.to_dict()
        self.assertEqual(AdaptivePolicySpec.from_dict(routed_body), routed)
        with self.assertRaisesRegex(ConfigurationError, "candidate geometry"):
            replace(routed, candidate_layers=(True,))
        malformed = {**routed_body, "candidate_layers": [1.9]}
        with self.assertRaisesRegex(DataValidationError, "invalid types"):
            AdaptivePolicySpec.from_dict(malformed)
        malformed = {**routed_body, "release_risk_threshold": "0.4"}
        with self.assertRaisesRegex(DataValidationError, "invalid types"):
            AdaptivePolicySpec.from_dict(malformed)
        malformed = {**routed_body, "controller_artifact_sha256": int("1" * 64)}
        with self.assertRaisesRegex(DataValidationError, "invalid types"):
            AdaptivePolicySpec.from_dict(malformed)
        with self.assertRaisesRegex(ConfigurationError, "one fixed direction"):
            replace(routed, layer=1)
        with self.assertRaisesRegex(ConfigurationError, "likely-unknown"):
            replace(routed, likely_unknown_risk_threshold=0.3)

    def test_adaptive_policy_v2_binds_routed_geometry_and_residual_risk(self) -> None:
        study = _study()
        policy = _adaptive_policy_v2()
        condition = replace(
            _e0_conditions(study)[0],
            steering_method="M6",
            method_artifact_sha256="3" * 64,
            adaptive_policy=policy,
        )

        def routed_record(question_id: str, action: str) -> GenerationRecord:
            scores_by_action = {
                "release": {"C": 0.7, "I": 0.2, "A": 0.1},
                "intervene": {"C": 0.3, "I": 0.6, "A": 0.1},
                "abstain": {"C": 0.1, "I": 0.85, "A": 0.05},
            }
            scores = scores_by_action[action]
            geometry = action == "intervene"
            alpha = (
                policy.alpha_max
                / (1 + math.exp(-policy.alpha_beta * (scores["I"] - policy.release_risk_threshold)))
                if geometry
                else 0.0
            )
            weights = [0.1, 0.2, 0.3, 0.4]
            trace = (
                {
                    "layer": 2,
                    "site": ActivationSite.BLOCK_OUTPUT.value,
                    "token_scope": TokenScope.FIRST_FOUR.value,
                    "alpha": alpha,
                    "sparsity": policy.sparsity,
                    "applied_tokens": 1,
                    "applied_token_indices": [0],
                    "activation_delta_norm": alpha * 2.0,
                    "direction_sha256": "f" * 64,
                    "direction_norm": 2.0,
                    "controller_artifact_sha256": policy.controller_artifact_sha256,
                    "router_weights": weights,
                    "router_weights_sha256": stable_hash(weights),
                    "pre_activation_sha256": "a" * 64,
                    "post_activation_sha256": "b" * 64,
                    "delta_sha256": "c" * 64,
                }
                if geometry
                else None
            )
            post_scores = {"C": 0.05, "I": 0.2, "A": 0.75} if geometry else None
            output_action = "abstain" if action == "abstain" else "release"
            baseline = _record(condition, question_id)
            draft = replace(
                baseline,
                layer=2 if geometry else None,
                site=ActivationSite.BLOCK_OUTPUT if geometry else None,
                token_scope=TokenScope.FIRST_FOUR if geometry else None,
                alpha=alpha,
                controller_scores=scores,
                outcome=(
                    Outcome.ABSTENTION if output_action == "abstain" else Outcome.CORRECT
                ),
                metadata={
                    **baseline.metadata,
                    "method_artifact_sha256": "3" * 64,
                    "policy_action": action,
                    "output_action": output_action,
                    "post_controller_scores": post_scores,
                    "intervention_trace": trace,
                    "intervention_trace_digest": (
                        stable_hash(trace) if trace is not None else None
                    ),
                },
            )
            decided = replace(
                draft,
                metadata={
                    **draft.metadata,
                    "policy_decision_digest": adaptive_policy_decision_digest(
                        draft,
                        policy=policy,
                        policy_action=action,
                        output_action=output_action,
                    ),
                },
            )
            return replace(
                decided,
                metadata={
                    **decided.metadata,
                    "execution_receipt_signature": sign_adaptive_execution_receipt(
                        decided,
                        policy=policy,
                        private_key_hex=_EXECUTION_PRIVATE_KEY,
                    ),
                },
            )

        released = routed_record("q-v2-release", "release")
        intervened = routed_record("q-v2-intervene", "intervene")
        abstained = routed_record("q-v2-abstain", "abstain")
        condition.validate_record(released)
        condition.validate_record(intervened)
        condition.validate_record(abstained)
        validate_adaptive_execution((released, intervened, abstained))

        trace = dict(intervened.metadata["intervention_trace"])
        trace["router_weights"] = [0.1, 0.2, 0.3, 0.3]
        trace["router_weights_sha256"] = stable_hash(trace["router_weights"])
        malformed = replace(
            intervened,
            metadata={
                **intervened.metadata,
                "intervention_trace": trace,
                "intervention_trace_digest": stable_hash(trace),
            },
        )
        with self.assertRaisesRegex(DataValidationError, "material executed edit"):
            condition.validate_record(malformed)


class PhaseRunLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._namespace_patcher = patch(
            "mfh.experiments.runner.validate_active_study_artifact_paths",
            side_effect=_allow_unit_test_artifact_paths,
        )
        self._namespace_patcher.start()

    def tearDown(self) -> None:
        self._namespace_patcher.stop()

    def test_one_shot_registry_is_reserved_for_e10_only(self) -> None:
        study = _study()
        registry = study.one_shot_registry_path
        self.assertTrue(str(registry).endswith("/.local/state/mfh/one-shot"))
        with patch.dict(os.environ, {"HOME": "/tmp/attacker-selected-home"}):
            self.assertEqual(study.one_shot_registry_path, registry)
        relocated = replace(
            study,
            source_path=Path("/tmp/attacker/configs/experiments/phases.yaml"),
        )
        self.assertEqual(relocated.one_shot_registry_path, registry)
        self.assertNotIn("one_shot_registry", inspect.signature(PhaseRunLedger.create).parameters)

    def test_e10_risk_gate_is_derived_from_records_not_claimed_metrics(self) -> None:
        condition = _e0_conditions(_study())[0]
        incorrect = replace(
            _record(condition, "simpleqa:bad"),
            benchmark="simpleqa_verified",
            outcome=Outcome.INCORRECT,
        )
        correct = replace(
            _record(condition, "aa-public:good"),
            benchmark="aa_omniscience_public_600",
            outcome=Outcome.CORRECT,
        )
        context = GateEvaluationContext(
            expected_record_count=2,
            records_factory=lambda: (incorrect, correct),
            analysis_protocol=load_analysis_protocol(
                ROOT / "configs/analysis/confirmatory.yaml"
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "risk.json"
            write_gate_evidence(
                evidence,
                phase=ExperimentPhase.E10,
                gate="risk_below_epsilon",
                contract_digest="a" * 64,
                record_set_digest="b" * 64,
                observations=[],
            )
            result = evaluate_gate(
                phase=ExperimentPhase.E10,
                gate="risk_below_epsilon",
                contract_digest="a" * 64,
                record_set_digest="b" * 64,
                evidence_path=evidence,
                context=context,
            )
            self.assertFalse(result.passed)
            self.assertEqual(result.metrics["hallucination_risk"], 1.0)

            aggregate_claim = root / "aggregate-claim.json"
            write_gate_evidence(
                aggregate_claim,
                phase=ExperimentPhase.E10,
                gate="risk_below_epsilon",
                contract_digest="a" * 64,
                record_set_digest="b" * 64,
                observations=[
                    {
                        "hallucination_risk": 0.0,
                        "coverage": 1.0,
                        "epsilon": 0.01,
                        "minimum_coverage": 0.5,
                    }
                ],
            )
            with self.assertRaises(
                DataValidationError,
                msg="caller-supplied aggregate risk must never be gate evidence",
            ):
                evaluate_gate(
                    phase=ExperimentPhase.E10,
                    gate="risk_below_epsilon",
                    contract_digest="a" * 64,
                    record_set_digest="b" * 64,
                    evidence_path=aggregate_claim,
                    context=context,
                )

    def test_probe_gate_uses_only_selected_t_dev_p0_rows(self) -> None:
        artifact = "4" * 64
        condition = replace(
            _e0_conditions(_study())[0],
            partition="T-dev",
            steering_method="probe-logistic",
            method_artifact_sha256=artifact,
            comparison_group="gate-selected",
        )
        controller_condition = replace(
            condition,
            partition="T-controller",
            comparison_group="development-only",
        )

        def scored(
            active_condition: EvaluationCondition,
            question_id: str,
            outcome: Outcome,
            probe_score: float,
            baseline_score: float,
            *,
            eligible: bool,
        ) -> GenerationRecord:
            record = _record(active_condition, question_id)
            return replace(
                record,
                outcome=outcome,
                metadata={
                    **record.metadata,
                    "partition": "T-dev" if eligible else "T-controller",
                    "probe_score": probe_score,
                    "output_entropy": baseline_score,
                    "maximum_token_probability": 1 - baseline_score,
                    "probe_artifact_sha256": artifact,
                    "probe_gate_eligible": eligible,
                },
            )

        records = (
            scored(condition, "probe-c", Outcome.CORRECT, 0.1, 0.9, eligible=True),
            scored(condition, "probe-i", Outcome.INCORRECT, 0.9, 0.1, eligible=True),
            scored(
                controller_condition,
                "controller-i",
                Outcome.INCORRECT,
                0.0,
                0.99,
                eligible=False,
            ),
        )
        observations = [
            {
                "condition_id": record.condition_id,
                "question_id": record.question_id,
                "incorrect": record.outcome is Outcome.INCORRECT,
                "probe_score": record.metadata["probe_score"],
                "output_entropy": record.metadata["output_entropy"],
                "maximum_token_probability": record.metadata[
                    "maximum_token_probability"
                ],
                "probe_artifact_sha256": artifact,
                "gate_eligible": record.metadata["probe_gate_eligible"],
            }
            for record in records
        ]
        context = GateEvaluationContext(
            expected_record_count=len(records),
            records_factory=lambda: records,
            expected_condition_ids=frozenset(
                {condition.condition_id, controller_condition.condition_id}
            ),
            condition_facts={
                condition.condition_id: {
                    "partition": "T-dev",
                    "system_prompt_id": "P0-neutral",
                    "steering_method": "probe-logistic",
                    "comparison_group": "gate-selected",
                    "method_artifact_sha256": artifact,
                },
                controller_condition.condition_id: {
                    "partition": "T-controller",
                    "system_prompt_id": "P0-neutral",
                    "steering_method": "probe-logistic",
                    "comparison_group": "development-only",
                    "method_artifact_sha256": artifact,
                },
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / "probe.json"
            write_gate_evidence(
                evidence,
                phase=ExperimentPhase.E2,
                gate="probe_beats_confidence_baselines",
                contract_digest="a" * 64,
                record_set_digest="b" * 64,
                observations=observations,
            )
            result = evaluate_gate(
                phase=ExperimentPhase.E2,
                gate="probe_beats_confidence_baselines",
                contract_digest="a" * 64,
                record_set_digest="b" * 64,
                evidence_path=evidence,
                context=context,
            )
            self.assertTrue(result.passed)
            self.assertEqual(result.metrics["probe_auroc"], 1.0)
            self.assertEqual(result.metrics["best_confidence_baseline_auroc"], 0.0)

            tampered_records = (
                replace(
                    records[0],
                    metadata={**records[0].metadata, "probe_gate_eligible": False},
                ),
                *records[1:],
            )
            tampered_observations = [dict(value) for value in observations]
            tampered_observations[0]["gate_eligible"] = False
            tampered_evidence = Path(directory) / "probe-tampered.json"
            write_gate_evidence(
                tampered_evidence,
                phase=ExperimentPhase.E2,
                gate="probe_beats_confidence_baselines",
                contract_digest="a" * 64,
                record_set_digest="b" * 64,
                observations=tampered_observations,
            )
            with self.assertRaisesRegex(DataValidationError, "record-level scores"):
                evaluate_gate(
                    phase=ExperimentPhase.E2,
                    gate="probe_beats_confidence_baselines",
                    contract_digest="a" * 64,
                    record_set_digest="b" * 64,
                    evidence_path=tampered_evidence,
                    context=GateEvaluationContext(
                        expected_record_count=len(tampered_records),
                        records_factory=lambda: tampered_records,
                        expected_condition_ids=frozenset(
                            {condition.condition_id, controller_condition.condition_id}
                        ),
                        condition_facts={
                            condition.condition_id: {
                                "partition": "T-dev",
                                "system_prompt_id": "P0-neutral",
                                "steering_method": "probe-logistic",
                                "comparison_group": "gate-selected",
                                "method_artifact_sha256": artifact,
                            },
                            controller_condition.condition_id: {
                                "partition": "T-controller",
                                "system_prompt_id": "P0-neutral",
                                "steering_method": "probe-logistic",
                                "comparison_group": "development-only",
                                "method_artifact_sha256": artifact,
                            },
                        },
                    ),
                )

    def test_matched_norm_treats_verified_no_intervention_as_zero(self) -> None:
        base = _e0_conditions(_study())[0]
        baseline_condition = replace(
            base,
            steering_method="M1",
            method_artifact_sha256="1" * 64,
            layer=1,
            site=ActivationSite.POST_MLP,
            token_scope=TokenScope.FIRST_FOUR,
            alpha=1.0,
        )
        adaptive_condition = replace(
            base,
            steering_method="M3",
            method_artifact_sha256="3" * 64,
            adaptive_policy=_adaptive_policy(),
        )
        baseline = _record(baseline_condition, "matched-q")
        baseline = replace(
            baseline,
            metadata={**baseline.metadata, "intervention_norm": 0.0},
        )
        adaptive = _record(adaptive_condition, "matched-q")
        adaptive = replace(
            adaptive,
            metadata={
                **adaptive.metadata,
                "policy_action": "release",
                "intervention_trace": None,
                "intervention_trace_digest": None,
            },
        )
        facts = {
            condition.condition_id: {
                "model_repository": condition.model_repository,
                "benchmark": condition.benchmark,
                "system_prompt_id": condition.system_prompt_id,
                "partition": condition.partition,
                "steering_method": condition.steering_method,
                "comparison_group": condition.comparison_group,
            }
            for condition in (baseline_condition, adaptive_condition)
        }
        context = GateEvaluationContext(
            expected_record_count=2,
            records_factory=lambda: (baseline, adaptive),
            expected_condition_ids=frozenset(facts),
            condition_facts=facts,
        )
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / "matched-norm.json"
            write_gate_evidence(
                evidence,
                phase=ExperimentPhase.E5,
                gate="matched_norm",
                contract_digest="a" * 64,
                record_set_digest="b" * 64,
                observations=[
                    {
                        "question_id": "matched-q",
                        "baseline_condition_id": baseline.condition_id,
                        "intervention_condition_id": adaptive.condition_id,
                    }
                ],
            )
            result = evaluate_gate(
                phase=ExperimentPhase.E5,
                gate="matched_norm",
                contract_digest="a" * 64,
                record_set_digest="b" * 64,
                evidence_path=evidence,
                context=context,
            )
            self.assertTrue(result.passed)
            self.assertEqual(result.metrics["absolute_difference"], 0.0)

            malformed_adaptive = replace(
                adaptive,
                metadata={**adaptive.metadata, "intervention_trace": "not-a-trace"},
            )
            with self.assertRaisesRegex(DataValidationError, "norm evidence"):
                evaluate_gate(
                    phase=ExperimentPhase.E5,
                    gate="matched_norm",
                    contract_digest="a" * 64,
                    record_set_digest="b" * 64,
                    evidence_path=evidence,
                    context=GateEvaluationContext(
                        expected_record_count=2,
                        records_factory=lambda: (baseline, malformed_adaptive),
                        expected_condition_ids=frozenset(facts),
                        condition_facts=facts,
                    ),
                )

    def test_failed_empirical_gate_freezes_as_non_prerequisite_falsification(self) -> None:
        study = _study()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = _input_artifacts(root, study.phase("E0").required_inputs)
            contract = _e0_contract(study, inputs)
            ledger = PhaseRunLedger.create(
                root / "E0-falsified",
                contract,
                study=study,
                input_artifacts=inputs,
                prerequisite_runs={},
            )
            ledger.checkpoint(
                _record(condition, question_id)
                for condition in contract.conditions
                for question_id in contract.question_ids_by_benchmark[condition.benchmark]
            )
            results = _gate_results(ledger, root / "passing-gate-evidence")
            records = tuple(ledger.records())
            failed_evidence = root / "deterministic-failed.json"
            observations = []
            for index, record in enumerate(records):
                actual = stable_hash(record.raw_output)
                mismatch = "0" * 64 if actual != "0" * 64 else "1" * 64
                repeated = mismatch if index == 0 else actual
                observations.append(
                    {
                        "condition_id": record.condition_id,
                        "question_id": record.question_id,
                        "first_output_sha256": actual,
                        "repeat_output_sha256": repeated,
                    }
                )
            write_gate_evidence(
                failed_evidence,
                phase=ExperimentPhase.E0,
                gate="deterministic_decode",
                contract_digest=contract.digest,
                record_set_digest=ledger.record_set_digest(),
                observations=observations,
            )
            failed_result = ledger.evaluate_gate("deterministic_decode", failed_evidence)
            self.assertFalse(failed_result.passed)
            results["deterministic_decode"] = failed_result
            receipt = _verified_scientific_e0_receipt(root)

            falsification = ledger.finalize_falsified(
                results,
                verified_e0_completion=receipt,
            )
            self.assertEqual(falsification.failed_gates, ("deterministic_decode",))
            self.assertFalse((ledger.directory / "complete.json").exists())
            self.assertTrue((ledger.directory / "falsified.json").is_file())
            self.assertEqual(
                ledger.verify_falsified().falsification_digest,
                falsification.falsification_digest,
            )
            with self.assertRaisesRegex(FrozenArtifactError, "completion marker"):
                ledger.verify_complete()
            with self.assertRaisesRegex(FrozenArtifactError, "terminal"):
                ledger.checkpoint((_record(contract.conditions[0], "new-record"),))
            with self.assertRaisesRegex(FrozenArtifactError, "terminal"):
                ledger.finalize(results, verified_e0_completion=receipt)

    def test_resumable_shards_require_passing_gates_then_freeze(self) -> None:
        study = _study()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = _input_artifacts(root, study.phase("E0").required_inputs)
            contract = _e0_contract(study, inputs)
            path = root / "E0-run"
            ledger = PhaseRunLedger.create(
                path,
                contract,
                study=study,
                input_artifacts=inputs,
                prerequisite_runs={},
            )
            self.assertEqual(ledger.progress(), (0, 500))
            first = _record(contract.conditions[0], "q-0")
            ledger.checkpoint((first,))
            with self.assertRaises(DataValidationError):
                ledger.checkpoint((first,))
            with self.assertRaises(DataValidationError):
                ledger.finalize({})

            resumed = PhaseRunLedger.open(path, study=study)
            pending = list(resumed.iter_pending())
            self.assertEqual(len(pending), 499)
            self.assertNotIn(
                (first.condition_id, first.question_id),
                {(item.condition.condition_id, item.question_id) for item in pending},
            )
            self.assertEqual(
                [(item.condition.condition_id, item.question_id) for item in pending],
                [
                    (item.condition.condition_id, item.question_id)
                    for item in PhaseRunLedger.open(path, study=study).iter_pending()
                ],
            )
            resumed.checkpoint(_record(item.condition, item.question_id) for item in pending)
            evidence = root / "gate-evidence"
            gates = _gate_results(resumed, evidence)
            with self.assertRaises(
                DataValidationError, msg="missing gates must block finalization"
            ):
                resumed.finalize({})
            first_gate = contract.required_gates[0]
            failing_path = root / "failing-gate.json"
            write_gate_evidence(
                failing_path,
                phase=contract.phase,
                gate=first_gate,
                contract_digest=contract.digest,
                record_set_digest=resumed.record_set_digest(),
                observations=[],
            )
            failing = resumed.evaluate_gate(first_gate, failing_path)
            self.assertTrue(failing.passed)
            failing = GateResult.create(
                phase=failing.phase,
                gate=failing.gate,
                passed=False,
                contract_digest=failing.contract_digest,
                record_set_digest=failing.record_set_digest,
                evaluator=failing.evaluator,
                evaluator_revision=failing.evaluator_revision,
                metrics=failing.metrics,
                artifact_paths={"evaluation": failing_path},
            )
            with self.assertRaises(DataValidationError, msg="forged gate flags must be rejected"):
                resumed.finalize({**gates, first_gate: failing})

            self_attested = GateResult.create(
                phase=contract.phase,
                gate=first_gate,
                passed=True,
                contract_digest=contract.digest,
                record_set_digest=resumed.record_set_digest(),
                evaluator="caller-assertion",
                evaluator_revision="e" * 64,
                metrics={"checked_models": 1, "identity_mismatches": 0},
                artifact_paths={"evaluation": evidence / f"{first_gate}.json"},
            )
            with self.assertRaises(
                DataValidationError,
                msg="self-attested pass flags must not satisfy a gate",
            ):
                resumed.finalize({**gates, first_gate: self_attested})

            with self.assertRaisesRegex(
                DataValidationError,
                "scientific completion receipt",
            ):
                resumed.finalize(gates)
            receipt = _verified_scientific_e0_receipt(root)
            completion = resumed.finalize(
                gates,
                verified_e0_completion=receipt,
            )
            self.assertEqual(completion.record_count, 500)
            self.assertEqual(set(completion.gate_result_digests), set(contract.required_gates))
            self.assertEqual(
                resumed.verify_complete().completion_digest, completion.completion_digest
            )
            source_evidence = evidence / f"{first_gate}.json"
            source_evidence.chmod(0o644)
            source_evidence.write_text("source changed after packaging\n", encoding="utf-8")
            self.assertEqual(
                resumed.verify_complete().completion_digest,
                completion.completion_digest,
            )
            with self.assertRaises(FrozenArtifactError):
                resumed.checkpoint((first,))

            packaged_evidence = path / "gate-artifacts" / contract.required_gates[0] / "evaluation"
            original_packaged_evidence = packaged_evidence.read_text(encoding="utf-8")
            packaged_evidence.write_text("tampered packaged evidence\n", encoding="utf-8")
            with self.assertRaises(FrozenArtifactError):
                resumed.verify_complete()
            packaged_evidence.write_text(original_packaged_evidence, encoding="utf-8")

            gate_path = path / "gates" / f"{first_gate}.json"
            original_gate = gate_path.read_text(encoding="utf-8")
            gate_path.chmod(0o644)
            gate_path.write_text(original_gate + " ", encoding="utf-8")
            with self.assertRaises(FrozenArtifactError):
                resumed.verify_complete()
            gate_path.write_text(original_gate, encoding="utf-8")

            shard = path / "shards/records-00000.jsonl"
            shard.write_text(shard.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.assertRaises(FrozenArtifactError):
                resumed.verify_complete()

    def test_prerequisite_digest_must_resolve_to_a_verified_completed_run(self) -> None:
        study = _study()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            e0_path, e0_completion = _complete_e0(root)
            phase = study.phase("E1")
            conditions = expand_factorial_conditions(
                study,
                "E1",
                models=_models("qwen3.6-27b-mlx-4bit"),
                prompts=_prompts(),
                benchmark_partitions={
                    "triviaqa": "T-dev",
                    "simpleqa_verified": "simpleqa-eval",
                    "aa_omniscience_public_600": "aa-eval",
                },
                interventions={"M0": InterventionSpec(method="M0")},
            )
            inputs = _input_artifacts(root, phase.required_inputs)
            reviewed_splits = _verified_reviewed_splits(root)
            inputs["deduplicated_splits"].unlink()
            inputs["deduplicated_splits"] = reviewed_splits.directory
            reviewed_triviaqa_ids = tuple(
                question.question_id
                for question in read_questions(reviewed_splits.directory / "T-dev.jsonl")
            )
            contract = PhaseRunContract(
                phase=ExperimentPhase.E1,
                study_protocol_digest=study.digest,
                conditions=conditions,
                question_ids_by_benchmark={
                    "triviaqa": reviewed_triviaqa_ids,
                    "simpleqa_verified": tuple(f"s-{index}" for index in range(1_000)),
                    "aa_omniscience_public_600": tuple(f"a-{index}" for index in range(600)),
                },
                input_fingerprints={name: sha256_path(path) for name, path in inputs.items()},
                prerequisite_digests={"E0": e0_completion.completion_digest},
                required_gates=phase.gates,
            )
            mismatched_splits = _verified_reviewed_splits(
                root,
                review_digest="9" * 64,
                suffix="-mismatch",
            )
            mismatched_inputs = {
                **inputs,
                "deduplicated_splits": mismatched_splits.directory,
            }
            mismatched_contract = replace(
                contract,
                question_ids_by_benchmark={
                    **contract.question_ids_by_benchmark,
                    "triviaqa": tuple(
                        question.question_id
                        for question in read_questions(mismatched_splits.directory / "T-dev.jsonl")
                    ),
                },
                input_fingerprints={
                    **contract.input_fingerprints,
                    "deduplicated_splits": mismatched_splits.fingerprint,
                },
            )
            with self.assertRaisesRegex(DataValidationError, "different human review"):
                PhaseRunLedger.create(
                    root / "mismatched-review-E1-run",
                    mismatched_contract,
                    study=study,
                    input_artifacts=mismatched_inputs,
                    prerequisite_runs={"E0": e0_path},
                    verified_reviewed_splits=mismatched_splits,
                )
            detached = replace(
                contract,
                question_ids_by_benchmark={
                    **contract.question_ids_by_benchmark,
                    "triviaqa": ("provisional-or-arbitrary-id",),
                },
            )
            with self.assertRaisesRegex(DataValidationError, "question schedule"):
                PhaseRunLedger.create(
                    root / "detached-E1-run",
                    detached,
                    study=study,
                    input_artifacts=inputs,
                    prerequisite_runs={"E0": e0_path},
                    verified_reviewed_splits=reviewed_splits,
                )
            detached_external = replace(
                contract,
                question_ids_by_benchmark={
                    **contract.question_ids_by_benchmark,
                    "simpleqa_verified": tuple(
                        f"fabricated-simple-{index}" for index in range(1_000)
                    ),
                },
            )
            with self.assertRaisesRegex(DataValidationError, "authorized source"):
                PhaseRunLedger.create(
                    root / "detached-external-E1-run",
                    detached_external,
                    study=study,
                    input_artifacts=inputs,
                    prerequisite_runs={"E0": e0_path},
                    verified_reviewed_splits=reviewed_splits,
                )
            wrong_external_partition = replace(
                contract,
                conditions=tuple(
                    replace(condition, partition="T-controller")
                    if condition.benchmark == "simpleqa_verified"
                    else condition
                    for condition in contract.conditions
                ),
            )
            with self.assertRaisesRegex(DataValidationError, "wrong evaluation partition"):
                PhaseRunLedger.create(
                    root / "wrong-external-partition-E1-run",
                    wrong_external_partition,
                    study=study,
                    input_artifacts=inputs,
                    prerequisite_runs={"E0": e0_path},
                    verified_reviewed_splits=reviewed_splits,
                )
            with self.assertRaisesRegex(DataValidationError, "reviewed splits"):
                PhaseRunLedger.create(
                    root / "unreviewed-E1-run",
                    contract,
                    study=study,
                    input_artifacts=inputs,
                    prerequisite_runs={"E0": e0_path},
                )
            created = PhaseRunLedger.create(
                root / "E1-run",
                contract,
                study=study,
                input_artifacts=inputs,
                prerequisite_runs={"E0": e0_path},
                verified_reviewed_splits=reviewed_splits,
            )
            self.assertEqual(created.progress(), (0, contract.expected_record_count))

            packaged_receipt = e0_path / "scientific-completion-receipt" / "receipt.json"
            original_receipt = packaged_receipt.read_text(encoding="utf-8")
            packaged_receipt.write_text(original_receipt + " ", encoding="utf-8")
            with self.assertRaisesRegex(
                FrozenArtifactError,
                "scientific completion receipt",
            ):
                PhaseRunLedger.open(e0_path, study=study).verify_complete()
            packaged_receipt.write_text(original_receipt, encoding="utf-8")

            forged = replace(contract, prerequisite_digests={"E0": "f" * 64})
            with self.assertRaises(DataValidationError, msg="hash-only prerequisites are invalid"):
                PhaseRunLedger.create(
                    root / "forged-E1-run",
                    forged,
                    study=study,
                    input_artifacts=inputs,
                    prerequisite_runs={"E0": e0_path},
                    verified_reviewed_splits=reviewed_splits,
                )


if __name__ == "__main__":
    unittest.main()

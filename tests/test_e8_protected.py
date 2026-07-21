from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

from mfh.contracts import (
    ActivationSite,
    GenerationRecord,
    Outcome,
    Runtime,
    TokenScope,
)
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.experiments.e8_protected import (
    BehaviorActivationEvidence,
    BehaviorLabelPair,
    M5VariantScreen,
    _complete_e7_behavior_label_pairs,
    _derive_behavior_pair_label,
    build_e8_protected_artifact,
    load_e8_protected_artifact,
    response_verbosity_style_preserved,
    save_e8_protected_artifact,
    select_m5_variant,
)
from mfh.methods.features import ActivationFeatureSchema
from mfh.methods.protected import (
    E8OperatingPointRegistry,
    load_e8_operating_point_registry,
    save_e8_operating_point_registry,
)
from mfh.provenance import stable_hash


class E8ProtectedArtifactTests(unittest.TestCase):
    @staticmethod
    def _record(*, method: str, condition: str, outcome: Outcome) -> GenerationRecord:
        return GenerationRecord(
            question_id="trivia-1",
            benchmark="triviaqa",
            model_repository="synthetic/model",
            model_revision="synthetic-revision",
            runtime=Runtime.SYNTHETIC,
            quantization="none",
            system_prompt_id="P0-neutral",
            rendered_prompt_hash="rendered",
            steering_method=method,
            layer=None,
            token_scope=None,
            alpha=0.0,
            sparsity=None,
            controller_scores={},
            raw_output=outcome.value,
            normalized_answer=outcome.value,
            outcome=outcome,
            generation_latency_seconds=0.0,
            input_tokens=1,
            output_tokens=1,
            condition_id=condition,
        )

    def _artifact(self):
        behaviors = (
            "correct_to_abstain",
            "xstest_safe_refusal",
            "harmful_refusal",
            "language_switching",
            "instruction_following_failure",
            "verbosity_style",
        )
        evidence = []
        for index, behavior in enumerate(behaviors):
            positive = torch.zeros(3, 7)
            positive[:, index] = torch.tensor([1.0, 2.0, 3.0])
            evidence.append(
                BehaviorActivationEvidence(
                    behavior=behavior,
                    positive_question_ids=tuple(
                        f"{behavior}-positive-{row}" for row in range(3)
                    ),
                    negative_question_ids=tuple(
                        f"{behavior}-negative-{row}" for row in range(3)
                    ),
                    positive_activations=positive,
                    negative_activations=torch.zeros(3, 7),
                )
            )
        question_ids = tuple(f"screen-{index}" for index in range(100))
        baseline = (Outcome.CORRECT,) * 50 + (Outcome.INCORRECT,) * 50
        protected = {name: (True,) * 100 for name in behaviors}
        screens = (
            M5VariantScreen(
                variant="orthogonal_projection",
                question_ids=question_ids,
                baseline_outcomes=baseline,
                intervention_outcomes=(Outcome.CORRECT,) * 60
                + (Outcome.INCORRECT,) * 40,
                protected_baseline=protected,
                protected_intervention=protected,
            ),
            M5VariantScreen(
                variant="covariance_aware",
                question_ids=question_ids,
                baseline_outcomes=baseline,
                intervention_outcomes=(Outcome.CORRECT,) * 55
                + (Outcome.INCORRECT,) * 45,
                protected_baseline=protected,
                protected_intervention=protected,
            ),
        )
        return build_e8_protected_artifact(
            evidence=evidence,
            feature_schema=ActivationFeatureSchema.synthetic(
                partition="side-effect-construction", width=7
            ),
            dense_direction=torch.tensor(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            ),
            source_fingerprints={
                "E6_transition_evidence": "1" * 64,
                "E7_sparse_artifacts": "2" * 64,
                "protected_behavior_activations": "3" * 64,
            },
            variant_screens=screens,
            layer=0,
            site=ActivationSite.POST_MLP,
            token_scope=TokenScope.FIRST_FOUR,
            alpha=0.5,
            reference_rms=1.0,
        )

    def test_builds_both_variants_and_round_trips(self) -> None:
        artifact = self._artifact()
        self.assertEqual(artifact.selected_variant, "orthogonal_projection")
        self.assertEqual(artifact.protected_subspace.behaviors, (
            "correct_to_abstain",
            "xstest_safe_refusal",
            "harmful_refusal",
            "language_switching",
            "instruction_following_failure",
            "verbosity_style",
        ))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "e8"
            save_e8_protected_artifact(path, artifact)
            loaded = load_e8_protected_artifact(path)
            self.assertEqual(loaded.selected_variant, artifact.selected_variant)
            self.assertTrue(
                torch.allclose(loaded.selected_direction, artifact.selected_direction)
            )
            screen = path / "variant-screens.json"
            screen.write_text(screen.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.assertRaises(FrozenArtifactError):
                load_e8_protected_artifact(path)

    def test_m5_screen_rejects_non_boolean_protected_values(self) -> None:
        artifact = self._artifact()
        payload = artifact.variant_screens[0].to_dict()
        payload["protected_baseline"]["verbosity_style"][0] = "false"
        with self.assertRaises(DataValidationError, msg="string booleans must fail closed"):
            M5VariantScreen.from_dict(payload)

    def test_m5_selection_requires_identical_protected_question_samples(self) -> None:
        artifact = self._artifact()
        left, right = artifact.variant_screens
        assert right.protected_question_ids is not None
        changed_ids = {
            name: tuple(f"different-{index}" for index, _ in enumerate(values))
            for name, values in right.protected_question_ids.items()
        }
        mismatched = replace(right, protected_question_ids=changed_ids)
        with self.assertRaises(DataValidationError, msg="unpaired screens must fail closed"):
            select_m5_variant((left, mismatched))

    def test_rejects_missing_behavior_and_harmful_screen(self) -> None:
        artifact = self._artifact()
        with self.assertRaises(DataValidationError):
            build_e8_protected_artifact(
                evidence=artifact.evidence[:-1],
                feature_schema=artifact.feature_schema,
                dense_direction=artifact.dense_direction,
                source_fingerprints=artifact.source_fingerprints,
                variant_screens=artifact.variant_screens,
                layer=artifact.layer,
                site=artifact.site,
                token_scope=artifact.token_scope,
                alpha=artifact.alpha,
                reference_rms=artifact.reference_rms,
            )

    def test_behavior_labels_are_rederived_from_matched_e7_records(self) -> None:
        pair = BehaviorLabelPair(
            baseline_record=self._record(
                method="M0", condition="baseline", outcome=Outcome.CORRECT
            ),
            intervention_record=self._record(
                method="M4b", condition="intervention", outcome=Outcome.ABSTENTION
            ),
            label="negative",
        )
        common = {
            "model_repository": "synthetic/model",
            "benchmark": "triviaqa",
            "system_prompt_id": "P0-neutral",
            "partition": "T-dev",
            "comparison_group": "matched",
        }
        context = SimpleNamespace(
            condition_facts={
                "baseline": {**common, "steering_method": "M0"},
                "intervention": {**common, "steering_method": "M4b"},
            }
        )
        self.assertEqual(
            _derive_behavior_pair_label(
                pair, behavior="correct_to_abstain", gate_context=context
            ),
            "positive",
        )
        context.condition_facts["intervention"]["comparison_group"] = "wrong"
        with self.assertRaises(DataValidationError):
            _derive_behavior_pair_label(
                pair, behavior="correct_to_abstain", gate_context=context
            )

    def test_behavior_pair_derivation_is_exhaustive_over_e7_schedule(self) -> None:
        feature_schema = replace(
            ActivationFeatureSchema.synthetic(
                partition="side-effect-construction", width=4
            ),
            prompt_id="P0-neutral",
        )
        condition_common = {
            "benchmark": "triviaqa",
            "system_prompt_id": "P0-neutral",
            "prompt_template_sha256": feature_schema.prompt_sha256,
            "partition": "T-dev",
            "comparison_group": "matched",
        }
        baseline_condition = SimpleNamespace(
            **condition_common,
            steering_method="M0",
            condition_id="baseline",
        )
        intervention_condition = SimpleNamespace(
            **condition_common,
            steering_method="M4b",
            condition_id="intervention",
        )
        records = tuple(
            replace(record, question_id=question_id)
            for question_id, record in (
                (
                    "trivia-1",
                    self._record(
                        method="M0",
                        condition="baseline",
                        outcome=Outcome.CORRECT,
                    ),
                ),
                (
                    "trivia-1",
                    self._record(
                        method="M4b",
                        condition="intervention",
                        outcome=Outcome.ABSTENTION,
                    ),
                ),
                (
                    "trivia-2",
                    self._record(
                        method="M0",
                        condition="baseline",
                        outcome=Outcome.CORRECT,
                    ),
                ),
                (
                    "trivia-2",
                    self._record(
                        method="M4b",
                        condition="intervention",
                        outcome=Outcome.CORRECT,
                    ),
                ),
            )
        )
        facts = {
            "baseline": {
                **condition_common,
                "model_repository": "synthetic/model",
                "steering_method": "M0",
            },
            "intervention": {
                **condition_common,
                "model_repository": "synthetic/model",
                "steering_method": "M4b",
            },
        }
        ledger = SimpleNamespace(
            contract=SimpleNamespace(
                conditions=(baseline_condition, intervention_condition),
                question_ids_by_benchmark={"triviaqa": ("trivia-1", "trivia-2")},
            ),
            records=lambda: records,
            _gate_context=lambda: SimpleNamespace(condition_facts=facts),
        )
        pairs = _complete_e7_behavior_label_pairs(
            ledger,
            behavior="correct_to_abstain",
            feature_schema=feature_schema,
        )
        self.assertEqual(
            tuple((value.question_id, value.label) for value in pairs),
            (("trivia-1", "positive"), ("trivia-2", "negative")),
        )

    def test_verbosity_style_labels_are_response_bound(self) -> None:
        baseline = replace(
            self._record(method="M0", condition="baseline", outcome=Outcome.CORRECT),
            benchmark="ifeval",
            raw_output="A concise answer.",
            normalized_answer="a concise answer",
        )
        preserved = replace(
            self._record(method="M4b", condition="intervention", outcome=Outcome.CORRECT),
            benchmark="ifeval",
            raw_output="Another short answer.",
            normalized_answer="another short answer",
        )
        drifted = replace(
            preserved,
            raw_output="- First item\n- Second item",
            normalized_answer="first item second item",
        )
        common = {
            "model_repository": "synthetic/model",
            "benchmark": "ifeval",
            "system_prompt_id": "P0-neutral",
            "partition": "side-effect-eval",
            "comparison_group": "matched",
        }
        context = SimpleNamespace(
            condition_facts={
                "baseline": {**common, "steering_method": "M0"},
                "intervention": {**common, "steering_method": "M4b"},
            }
        )
        negative = BehaviorLabelPair(baseline, preserved, "positive")
        positive = BehaviorLabelPair(baseline, drifted, "negative")
        self.assertTrue(
            response_verbosity_style_preserved(
                baseline.raw_output, preserved.raw_output
            )
        )
        self.assertEqual(
            _derive_behavior_pair_label(
                negative, behavior="verbosity_style", gate_context=context
            ),
            "negative",
        )
        self.assertEqual(
            _derive_behavior_pair_label(
                positive, behavior="verbosity_style", gate_context=context
            ),
            "positive",
        )

    def test_operating_registry_round_trip(self) -> None:
        registry = E8OperatingPointRegistry(
            matching_dimension="hallucination_risk",
            target=0.05,
            tolerance=0.01,
            condition_ids_by_prompt={
                prompt: {
                    method: stable_hash({"prompt": prompt, "method": method})
                    for method in ("M1", "M3", "M4", "M5")
                }
                for prompt in ("P0-neutral", "P2-calibrated-abstention")
            },
            candidate_screen_sha256="4" * 64,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            save_e8_operating_point_registry(path, registry)
            loaded = load_e8_operating_point_registry(path)
            self.assertEqual(loaded.to_dict(), registry.to_dict())


if __name__ == "__main__":
    unittest.main()

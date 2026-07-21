from __future__ import annotations

import math
import unittest

import torch

from mfh.contracts import ActivationSite, Outcome, PromptSpec, TokenScope
from mfh.errors import ConfigurationError
from mfh.inference.architecture import HookMode, resolve_hook_points
from mfh.inference.hooks import (
    ActivationSession,
    CapturePolicy,
    HookState,
    InterventionPlan,
    PassPhase,
    selection_weights,
)
from mfh.inference.runtime import TransformersRuntime
from mfh.methods.extraction import CAAExtractor, CentroidExtractionMode, CentroidExtractor
from tests.toy_transformer import (
    QwenLikeBlock,
    TinyCausalLM,
    TinyProcessor,
    tiny_model_spec,
)


class ArchitectureTests(unittest.TestCase):
    def test_gemma_and_qwen_sites_resolve_to_correct_residual_boundaries(self) -> None:
        gemma = TinyCausalLM()
        points = resolve_hook_points(
            gemma,
            expected_layers=2,
            layers=(0,),
            sites=(
                ActivationSite.POST_ATTENTION,
                ActivationSite.POST_MLP,
                ActivationSite.BLOCK_OUTPUT,
            ),
        )
        self.assertEqual(points[0].mode, HookMode.PRE)
        self.assertTrue(points[0].module_path.endswith("pre_feedforward_layernorm"))
        self.assertEqual(points[1].mode, HookMode.POST)
        self.assertTrue(points[1].module_path.endswith("post_feedforward_layernorm"))
        self.assertEqual(points[2].module_path, "model.layers.0")

        class QwenModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.ModuleList([QwenLikeBlock(4)])

        qwen_points = resolve_hook_points(
            QwenModel(),
            expected_layers=1,
            layers=(0,),
            sites=(ActivationSite.POST_ATTENTION, ActivationSite.POST_MLP),
        )
        self.assertTrue(qwen_points[0].module_path.endswith("post_attention_layernorm"))
        self.assertTrue(qwen_points[1].module_path.endswith("mlp"))

    def test_wrong_layer_count_fails_instead_of_hooking_an_ambiguous_module(self) -> None:
        with self.assertRaises(ConfigurationError):
            resolve_hook_points(
                TinyCausalLM(),
                expected_layers=3,
                layers=(0,),
                sites=(ActivationSite.BLOCK_OUTPUT,),
            )

    def test_conditional_generation_text_path_is_preferred(self) -> None:
        class ConditionalModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = torch.nn.Module()
                self.model.language_model = torch.nn.Module()
                self.model.language_model.layers = torch.nn.ModuleList(
                    [QwenLikeBlock(4), QwenLikeBlock(4)]
                )

        points = resolve_hook_points(
            ConditionalModel(),
            expected_layers=2,
            layers=(1,),
            sites=(ActivationSite.BLOCK_OUTPUT,),
        )
        self.assertEqual(points[0].module_path, "model.language_model.layers.1")

class SelectionTests(unittest.TestCase):
    def weights(self, scope: TokenScope, state: HookState, length: int = 6) -> torch.Tensor:
        return selection_weights(
            scope,
            state,
            length,
            decay=0.5,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    def test_prompt_generated_and_teacher_forced_positions_are_distinct(self) -> None:
        prompt = HookState(phase=PassPhase.PROMPT, prompt_length=6)
        self.assertEqual(self.weights(TokenScope.FINAL_PROMPT, prompt).tolist(), [0, 0, 0, 0, 0, 1])
        self.assertFalse(self.weights(TokenScope.FIRST_GENERATED, prompt).any())

        generated = HookState(phase=PassPhase.GENERATED, prompt_length=6, generation_step=0)
        self.assertTrue(self.weights(TokenScope.FIRST_GENERATED, generated, 1).all())

        teacher = HookState(
            phase=PassPhase.TEACHER_FORCED,
            prompt_length=3,
            response_start=3,
        )
        self.assertEqual(
            self.weights(TokenScope.FINAL_PROMPT, teacher).tolist(), [0, 0, 1, 0, 0, 0]
        )
        self.assertEqual(
            self.weights(TokenScope.FIRST_GENERATED, teacher).tolist(), [0, 0, 0, 1, 0, 0]
        )
        decay = self.weights(TokenScope.EXPONENTIAL_DECAY, teacher)
        self.assertAlmostEqual(float(decay[4]), math.exp(-0.5), places=6)


class HookSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(3)
        self.model = TinyCausalLM()
        self.points = resolve_hook_points(
            self.model,
            expected_layers=2,
            layers=(0,),
            sites=(ActivationSite.POST_MLP,),
        )
        self.inputs = torch.tensor([[1, 2, 3]])

    def test_capture_is_pre_intervention_and_steering_changes_only_selected_position(self) -> None:
        baseline = ActivationSession(
            self.points,
            capture_policy=CapturePolicy.PROMPT_FINAL,
        )
        baseline.set_prompt(3)
        with baseline:
            baseline_output = self.model(self.inputs).logits

        key = self.points[0].key
        steered = ActivationSession(
            self.points,
            interventions={
                key: InterventionPlan(
                    direction=torch.nn.functional.normalize(torch.ones(8), dim=0),
                    alpha=2.0,
                    token_scope=TokenScope.FINAL_PROMPT,
                    rms_relative=False,
                )
            },
            capture_policy=CapturePolicy.PROMPT_FINAL,
        )
        steered.set_prompt(3)
        with steered:
            steered_output = self.model(self.inputs).logits
        self.assertTrue(torch.allclose(baseline.activations()[key], steered.activations()[key]))
        self.assertTrue(torch.allclose(baseline_output[:, :2], steered_output[:, :2]))
        self.assertFalse(torch.allclose(baseline_output[:, -1], steered_output[:, -1]))


class RuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = TinyCausalLM()
        self.runtime = TransformersRuntime(
            model=self.model,
            processor=TinyProcessor(),
            model_spec=tiny_model_spec(),
        )
        self.rendered = self.runtime.render_prompt(
            PromptSpec("P0", "Answer briefly."), "Capital of France?"
        )
        self.points = resolve_hook_points(
            self.model,
            expected_layers=2,
            layers=(0, 1),
            sites=(ActivationSite.BLOCK_OUTPUT,),
        )

    def test_deterministic_generation_and_generated_token_capture(self) -> None:
        session = ActivationSession(
            self.points,
            capture_policy=CapturePolicy.TOKEN_SCOPE,
            capture_scope=TokenScope.FIRST_GENERATED,
        )
        first = self.runtime.generate(self.rendered, max_new_tokens=3, session=session)
        second = self.runtime.generate(self.rendered, max_new_tokens=3, session=session)
        self.assertEqual(first.token_ids, second.token_ids)
        self.assertEqual(first.output_tokens, 3)
        self.assertEqual(len(self.rendered.sha256), 64)
        self.assertTrue(all(values.shape == (1, 8) for values in session.activations().values()))

    def test_prompt_and_response_activation_extraction(self) -> None:
        prompt_session = ActivationSession(self.points, capture_policy=CapturePolicy.PROMPT_FINAL)
        prompt = self.runtime.prompt_activations(self.rendered, prompt_session)
        self.assertTrue(all(values.shape == (1, 8) for values in prompt.values()))

        response_session = ActivationSession(
            self.points, capture_policy=CapturePolicy.RESPONSE_TOKENS
        )
        response = self.runtime.teacher_forced_activations(self.rendered, "xy", response_session)
        self.assertTrue(all(values.shape == (2, 8) for values in response.values()))

    def test_teacher_forced_likelihood_is_alias_length_normalized(self) -> None:
        score = self.runtime.score_response(self.rendered, "xy")
        self.assertEqual(len(score.token_ids), 2)
        self.assertTrue(math.isfinite(score.total_log_likelihood))
        self.assertAlmostEqual(score.mean_log_likelihood, score.total_log_likelihood / 2)
        aliases = self.runtime.score_aliases(self.rendered, ("x", "xy"))
        self.assertEqual([len(item.token_ids) for item in aliases], [1, 2])

    def test_teacher_forced_scoring_clears_reused_activation_session(self) -> None:
        session = ActivationSession(
            self.points,
            capture_policy=CapturePolicy.RESPONSE_TOKENS,
        )
        self.runtime.score_response(self.rendered, "xy", session=session)
        first = session.activations()
        self.runtime.score_response(self.rendered, "xy", session=session)
        second = session.activations()
        self.assertEqual(first.keys(), second.keys())
        for key in first:
            self.assertEqual(first[key].shape, second[key].shape)
            self.assertTrue(torch.equal(first[key], second[key]))

    def test_m1_prompt_response_and_caa_extraction_workflows(self) -> None:
        other = self.runtime.render_prompt(
            PromptSpec("P0", "Answer briefly."), "A different question?"
        )
        prompt_extractor = CentroidExtractor(self.points, mode=CentroidExtractionMode.PROMPT_FINAL)
        prompt_extractor.observe(self.runtime, self.rendered, outcome=Outcome.CORRECT)
        prompt_extractor.observe(self.runtime, other, outcome=Outcome.INCORRECT)
        prompt_bank = prompt_extractor.build(data_fingerprint="c" * 64)
        self.assertEqual(
            {vector.source_method for vector in prompt_bank.vectors.values()}, {"M1-P"}
        )

        response_extractor = CentroidExtractor(
            self.points, mode=CentroidExtractionMode.RESPONSE_TOKENS
        )
        response_extractor.observe(
            self.runtime,
            self.rendered,
            outcome=Outcome.CORRECT,
            response="x",
        )
        response_extractor.observe(
            self.runtime,
            self.rendered,
            outcome=Outcome.INCORRECT,
            response="yz",
        )
        response_bank = response_extractor.build(data_fingerprint="d" * 64)
        self.assertEqual(
            {vector.source_method for vector in response_bank.vectors.values()}, {"M1-R"}
        )

        caa = CAAExtractor(self.points)
        caa.observe_pair(
            self.runtime,
            self.rendered,
            positive_response="x",
            negative_response="yz",
        )
        caa_bank = caa.build(data_fingerprint="e" * 64)
        self.assertTrue(caa_bank.vectors)


if __name__ == "__main__":
    unittest.main()

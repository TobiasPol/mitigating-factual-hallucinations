from __future__ import annotations

import math
import unittest

import torch

from mfh.contracts import ActivationSite, TokenScope
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
from tests.toy_transformer import (
    QwenLikeBlock,
    TinyCausalLM,
)


class ArchitectureTests(unittest.TestCase):
    def test_qwen_sites_resolve_to_correct_residual_boundaries(self) -> None:
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
        self.assertEqual(qwen_points[0].mode, HookMode.PRE)
        self.assertEqual(qwen_points[1].mode, HookMode.POST)

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


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from mfh.contracts import ActivationSite, TokenScope
from mfh.errors import FrozenArtifactError
from mfh.inference.architecture import HookKey
from mfh.methods.adaptive import (
    AdaptiveController,
    AdaptiveRouter,
    AlphaController,
    AlphaMode,
    RoutedVectorBank,
    RouterKind,
)
from mfh.methods.composite import (
    CompositeManifest,
    CompositePolicy,
    CompositePolicyConfig,
    OutputAction,
    RiskRegime,
    load_composite_manifest,
    load_composite_policy,
    minimum_alpha_for_risk,
    save_composite_manifest,
    save_composite_policy,
)
from mfh.methods.features import ActivationFeatureSchema, ActivationKind
from mfh.methods.probes import (
    CalibratedProbe,
    ProbeKind,
    ProbeState,
    ProbeTask,
    TemperatureCalibrator,
)
from mfh.provenance import sha256_path


def deterministic_controller() -> AdaptiveController:
    training_schema = ActivationFeatureSchema.synthetic(partition="T-controller-train", width=2)
    calibration_schema = ActivationFeatureSchema.synthetic(
        partition="T-controller-calibration", width=2
    )
    probe = CalibratedProbe(
        task=ProbeTask.CORRECT_INCORRECT_ABSTENTION,
        state=ProbeState(
            kind=ProbeKind.LOGISTIC,
            labels=("C", "I", "A"),
            feature_mean=torch.zeros(2),
            feature_scale=torch.ones(2),
            parameters={
                "weight": torch.tensor([[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
                "bias": torch.zeros(3),
            },
        ),
        calibrator=TemperatureCalibrator(1.0),
        training_fingerprint="1" * 64,
        calibration_fingerprint="2" * 64,
        training_schema=training_schema,
        calibration_schema=calibration_schema,
    )
    key = HookKey(1, ActivationSite.POST_MLP)
    bank = RoutedVectorBank(
        centers=torch.zeros(1, 2),
        directions={key: torch.tensor([[1.0, 0.0]])},
        correct_counts=(10,),
        incorrect_counts=(10,),
        data_fingerprint="3" * 64,
        feature_schema=ActivationFeatureSchema.synthetic(partition="T-steer", width=2),
    )
    router = AdaptiveRouter(
        kind=RouterKind.NEAREST_CENTROID,
        centers=torch.zeros(1, 2),
        training_fingerprint="4" * 64,
        feature_schema=ActivationFeatureSchema.synthetic(partition="T-controller-train", width=2),
    )
    return AdaptiveController(
        risk_probe=probe,
        vector_bank=bank,
        vector_router=router,
        alpha_controller=AlphaController(
            AlphaMode.RISK_GATED, alpha_max=2.0, beta=10, threshold=0.2
        ),
        fixed_layer=1,
    )


def early_probe(controller: AdaptiveController) -> CalibratedProbe:
    probe = controller.risk_probe
    return replace(
        probe,
        training_schema=ActivationFeatureSchema.synthetic(
            partition="T-controller-train",
            width=2,
            activation_kind=ActivationKind.FIRST_GENERATED,
            token_scope=TokenScope.FIRST_GENERATED,
        ),
        calibration_schema=ActivationFeatureSchema.synthetic(
            partition="T-controller-calibration",
            width=2,
            activation_kind=ActivationKind.FIRST_GENERATED,
            token_scope=TokenScope.FIRST_GENERATED,
        ),
    )


class CompositePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        controller = deterministic_controller()
        self.policy = CompositePolicy(
            controller,
            CompositePolicyConfig(
                tau_low=0.2,
                tau_high=0.7,
                release_epsilon=0.1,
                token_scope=TokenScope.FIRST_FOUR,
            ),
            early_probe=early_probe(controller),
        )

    def test_three_regimes_intervene_only_when_recoverable(self) -> None:
        assessments = self.policy.assess(torch.tensor([[-3.0, 0.0], [0.0, 0.0], [3.0, 0.0]]))
        self.assertEqual(
            [value.regime for value in assessments],
            [
                RiskRegime.KNOWN,
                RiskRegime.POTENTIALLY_RECOVERABLE,
                RiskRegime.LIKELY_UNKNOWN,
            ],
        )
        self.assertFalse(assessments[0].interventions)
        self.assertTrue(assessments[1].interventions)
        self.assertTrue(assessments[2].should_abstain)

    def test_early_recheck_and_output_gate_enforce_constraints(self) -> None:
        safe = self.policy.reevaluate_after_early_tokens(
            torch.tensor([[-3.0, 0.0]]),
            safety_ok=True,
            language_ok=True,
            refusal_drift=False,
            gold_log_likelihood_delta=0.2,
        )
        self.assertTrue(safe.continue_generation)
        self.assertTrue(safe.gold_likelihood_improved)
        risky = self.policy.reevaluate_after_early_tokens(
            torch.tensor([[3.0, 0.0]]),
            safety_ok=True,
            language_ok=True,
            refusal_drift=False,
        )
        self.assertFalse(risky.continue_generation)
        released = self.policy.output_gate(
            0.05, safety_ok=True, language_ok=True, refusal_drift=False
        )
        self.assertEqual(released.action, OutputAction.RELEASE)
        blocked = self.policy.output_gate(
            0.05, safety_ok=False, language_ok=True, refusal_drift=False
        )
        self.assertEqual(blocked.action, OutputAction.ABSTAIN)

    def test_minimum_alpha_and_frozen_manifest(self) -> None:
        self.assertEqual(
            minimum_alpha_for_risk(((2.0, 0.03), (0.5, 0.2), (1.0, 0.08)), risk_epsilon=0.1),
            1.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "probe.bin").write_bytes(b"probe")
            (root / "vectors.bin").write_bytes(b"vectors")
            path = Path(directory) / "composite.json"
            component_paths = {"probe": "probe.bin", "vectors": "vectors.bin"}
            manifest = CompositeManifest(
                prompt_id="P2-calibrated-abstention",
                method="M6",
                policy=self.policy.config,
                component_paths=component_paths,
                component_digests={
                    name: sha256_path(root / relative) for name, relative in component_paths.items()
                },
                data_fingerprints={"T-controller": "c" * 64, "T-steer": "d" * 64},
            )
            save_composite_manifest(path, manifest)
            loaded = load_composite_manifest(path)
            self.assertEqual(loaded.body(), manifest.body())
            value = json.loads(path.read_text(encoding="utf-8"))
            value["method"] = "tampered"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(FrozenArtifactError):
                load_composite_manifest(path)

            policy_path = root / "policy"
            save_composite_policy(policy_path, self.policy)
            loaded_policy = load_composite_policy(policy_path)
            expected = self.policy.assess(torch.tensor([[0.0, 0.0]]))[0]
            actual = loaded_policy.assess(torch.tensor([[0.0, 0.0]]))[0]
            self.assertEqual(expected.regime, actual.regime)
            self.assertEqual(expected.alpha, actual.alpha)


if __name__ == "__main__":
    unittest.main()

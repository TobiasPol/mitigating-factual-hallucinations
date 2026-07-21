from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.methods.features import ActivationFeatureSchema
from mfh.methods.protected import (
    EmpiricalEvaluationIdentity,
    EmpiricalOperatingPoint,
    alpha_for_matched_activation_projection,
    behavior_covariance,
    build_behavior_direction,
    build_protected_subspace,
    covariance_aware_direction,
    load_protected_subspace,
    match_empirical_operating_points,
    save_protected_subspace,
    subspace_covariance,
)


class ProtectedDirectionTests(unittest.TestCase):
    def test_projection_removes_refusal_and_language_components(self) -> None:
        refusal = build_behavior_direction(
            "safe_refusal",
            torch.tensor([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
            torch.zeros(2, 3),
        )
        language = build_behavior_direction(
            "language_switch",
            torch.tensor([[0.0, 2.0, 0.0], [0.0, 3.0, 0.0]]),
            torch.zeros(2, 3),
        )
        subspace = build_protected_subspace(
            (refusal, language),
            data_fingerprint="c" * 64,
            feature_schema=ActivationFeatureSchema.synthetic(
                partition="protected-construction", width=3
            ),
        )
        truth = torch.tensor([1.0, 1.0, 2.0])
        projected = subspace.project(truth)
        self.assertTrue(torch.allclose(projected, torch.tensor([0.0, 0.0, 2.0]), atol=1e-6))
        self.assertAlmostEqual(subspace.protected_energy(projected), 0.0, places=6)
        normalized = subspace.project(truth, normalize=True)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(normalized)), 1.0, places=6)
        alpha = alpha_for_matched_activation_projection(normalized, truth, target_gain=1.0)
        self.assertAlmostEqual(alpha, 0.5, places=6)

    def test_covariance_aware_solution_avoids_protected_energy(self) -> None:
        truth = torch.tensor([1.0, 1.0])
        changes = torch.tensor([[3.0, 0.0], [-3.0, 0.0], [2.0, 0.0], [-2.0, 0.0]])
        covariance = behavior_covariance(changes)
        protected = covariance_aware_direction(truth, covariance, lambda_penalty=1.0, ridge=0.1)
        self.assertLess(abs(float(protected[0])), abs(float(protected[1])))

        subspace = build_protected_subspace(
            {"refusal": torch.tensor([1.0, 0.0])},
            data_fingerprint="d" * 64,
            feature_schema=ActivationFeatureSchema.synthetic(
                partition="protected-construction", width=2
            ),
        )
        from_basis = subspace_covariance(subspace)
        self.assertTrue(torch.equal(from_basis, torch.tensor([[1.0, 0.0], [0.0, 0.0]]).double()))

    def test_protected_subspace_artifact_is_immutable(self) -> None:
        subspace = build_protected_subspace(
            {"refusal": torch.tensor([1.0, 0.0, 0.0])},
            data_fingerprint="f" * 64,
            feature_schema=ActivationFeatureSchema.synthetic(
                partition="protected-construction", width=3
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protected"
            save_protected_subspace(path, subspace)
            loaded = load_protected_subspace(path, expected_data_fingerprint="f" * 64)
            self.assertTrue(torch.allclose(loaded.basis, subspace.basis, atol=1e-6))
            tensor_path = path / "subspace.safetensors"
            tensor_path.write_bytes(tensor_path.read_bytes() + b"tamper")
            with self.assertRaises(FrozenArtifactError):
                load_protected_subspace(path)

    def test_empirical_matching_uses_measured_risk_or_coverage(self) -> None:
        evaluation = EmpiricalEvaluationIdentity(
            benchmark="synthetic",
            model_repository="synthetic/model",
            model_revision="0" * 40,
            prompt_id="P0",
            prompt_sha256="1" * 64,
            question_set_fingerprint="2" * 64,
            generation_bundle_fingerprint="3" * 64,
        )
        points = {
            "M4": (
                EmpiricalOperatingPoint("M4", 0.5, 0.10, 0.80, {"utility": 0.9}, evaluation),
                EmpiricalOperatingPoint("M4", 1.0, 0.05, 0.70, {"utility": 0.85}, evaluation),
            ),
            "M5": (EmpiricalOperatingPoint("M5", 0.7, 0.05, 0.75, {"utility": 0.92}, evaluation),),
        }
        matched = match_empirical_operating_points(
            points, target_hallucination_risk=0.05, tolerance=0.001
        )
        self.assertEqual(matched["M4"].coverage, 0.70)
        self.assertEqual(matched["M5"].coverage, 0.75)

        incompatible = EmpiricalEvaluationIdentity(
            benchmark="synthetic",
            model_repository="synthetic/model",
            model_revision="0" * 40,
            prompt_id="P0",
            prompt_sha256="1" * 64,
            question_set_fingerprint="4" * 64,
            generation_bundle_fingerprint="3" * 64,
        )
        with self.assertRaises(DataValidationError):
            match_empirical_operating_points(
                {
                    "M4": points["M4"],
                    "M5": (
                        EmpiricalOperatingPoint(
                            "M5",
                            0.7,
                            0.05,
                            0.75,
                            {"utility": 0.92},
                            incompatible,
                        ),
                    ),
                },
                target_hallucination_risk=0.05,
                tolerance=0.001,
            )


if __name__ == "__main__":
    unittest.main()

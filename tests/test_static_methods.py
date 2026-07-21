from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from mfh.contracts import ActivationSite, Outcome
from mfh.errors import FrozenArtifactError
from mfh.inference.architecture import HookKey
from mfh.methods.controls import (
    label_shuffled_centroid_direction,
    matched_random_direction,
    norm_matched_gaussian_perturbation,
    opposite_direction,
    zero_direction,
)
from mfh.methods.static import (
    CentroidVectorBuilder,
    OnlineMoments,
    PairedDifferenceBuilder,
    load_vector_bank,
    save_vector_bank,
)


class OnlineMomentsTests(unittest.TestCase):
    def test_streaming_and_merged_statistics_match_direct_computation(self) -> None:
        values = torch.tensor([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
        streaming = OnlineMoments()
        streaming.update(values[:1])
        streaming.update(values[1:])
        self.assertIsNotNone(streaming.mean)
        self.assertIsNotNone(streaming.m2)
        assert streaming.mean is not None and streaming.m2 is not None
        self.assertTrue(torch.allclose(streaming.mean, values.double().mean(0)))
        self.assertTrue(torch.allclose(streaming.variance(), values.double().var(0)))

        left, right = OnlineMoments(), OnlineMoments()
        left.update(values[:2])
        right.update(values[2:])
        left.merge(right)
        assert left.mean is not None and left.m2 is not None
        self.assertTrue(torch.allclose(left.mean, streaming.mean))
        self.assertTrue(torch.allclose(left.m2, streaming.m2))


class VectorBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.key = HookKey(1, ActivationSite.POST_MLP)
        self.fingerprint = "b" * 64

    def test_centroid_and_caa_build_unit_directions(self) -> None:
        centroid = CentroidVectorBuilder()
        centroid.update(Outcome.CORRECT, {self.key: torch.tensor([[3.0, 1.0], [4.0, 1.0]])})
        centroid.update(Outcome.INCORRECT, {self.key: torch.tensor([[1.0, 1.0], [2.0, 1.0]])})
        bank = centroid.build(source_method="M1-P", data_fingerprint=self.fingerprint)
        self.assertTrue(torch.allclose(bank.vectors[self.key].direction, torch.tensor([1.0, 0.0])))
        self.assertEqual(bank.vectors[self.key].positive_count, 2)

        caa = PairedDifferenceBuilder()
        caa.update(
            {self.key: torch.tensor([[2.0, 2.0]])},
            {self.key: torch.tensor([[1.0, 0.0]])},
        )
        caa_bank = caa.build(data_fingerprint=self.fingerprint)
        expected = torch.nn.functional.normalize(torch.tensor([1.0, 2.0]), dim=0)
        self.assertTrue(torch.allclose(caa_bank.vectors[self.key].direction, expected))

    def test_vector_artifact_round_trip_and_checksum_validation(self) -> None:
        builder = CentroidVectorBuilder()
        builder.update(Outcome.CORRECT, {self.key: torch.tensor([[2.0, 0.0]])})
        builder.update(Outcome.INCORRECT, {self.key: torch.tensor([[0.0, 0.0]])})
        bank = builder.build(source_method="M1-P", data_fingerprint=self.fingerprint)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank"
            save_vector_bank(path, bank)
            loaded = load_vector_bank(path, expected_data_fingerprint=self.fingerprint)
            self.assertTrue(
                torch.equal(loaded.vectors[self.key].direction, bank.vectors[self.key].direction)
            )
            metadata_path = path / "metadata.json"
            original_metadata = metadata_path.read_text(encoding="utf-8")
            metadata = json.loads(original_metadata)
            metadata["vectors"][0]["source_method"] = "tampered"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(FrozenArtifactError):
                load_vector_bank(path)
            metadata_path.write_text(original_metadata, encoding="utf-8")
            tensor_path = path / "vectors.safetensors"
            tensor_path.write_bytes(tensor_path.read_bytes() + b"tamper")
            with self.assertRaises(FrozenArtifactError):
                load_vector_bank(path)


class ControlTests(unittest.TestCase):
    def test_norm_matched_opposite_zero_and_shuffled_controls(self) -> None:
        reference = torch.tensor([3.0, 4.0])
        random_a = matched_random_direction(reference, seed=5)
        random_b = matched_random_direction(reference, seed=5)
        self.assertTrue(torch.equal(random_a, random_b))
        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(random_a)),
            float(torch.linalg.vector_norm(reference)),
        )
        self.assertTrue(torch.equal(opposite_direction(reference), -reference))
        self.assertTrue(torch.equal(zero_direction(reference), torch.zeros(2)))
        gaussian = norm_matched_gaussian_perturbation(reference, seed=5)
        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(gaussian)),
            float(torch.linalg.vector_norm(reference)),
        )
        self.assertFalse(torch.equal(random_a, gaussian))

        activations = torch.tensor([[3.0, 0.0], [2.0, 0.0], [0.0, 2.0], [0.0, 3.0]])
        outcomes = [Outcome.CORRECT, Outcome.CORRECT, Outcome.INCORRECT, Outcome.INCORRECT]
        shuffled = label_shuffled_centroid_direction(activations, outcomes, seed=9)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(shuffled)), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()

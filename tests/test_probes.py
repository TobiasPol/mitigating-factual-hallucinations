from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from mfh.contracts import Outcome
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.methods.features import ActivationFeatureSchema
from mfh.methods.probes import (
    CalibrationKind,
    ProbeDataset,
    ProbeKind,
    ProbeTask,
    ProbeTrainingConfig,
    encode_probe_task,
    evaluate_probe,
    fit_calibrated_probe,
    load_calibrated_probe,
    save_calibrated_probe,
    separability_gate,
)


def clustered_dataset(prefix: str, rows_per_class: int, *, offset: int = 0) -> ProbeDataset:
    generator = torch.Generator().manual_seed(100 + offset)
    centers = (
        torch.tensor([2.5, 0.0, 0.0]),
        torch.tensor([-2.5, 0.0, 0.0]),
        torch.tensor([0.0, 2.5, 0.0]),
    )
    outcomes = (Outcome.CORRECT, Outcome.INCORRECT, Outcome.ABSTENTION)
    features: list[torch.Tensor] = []
    labels: list[Outcome] = []
    identifiers: list[str] = []
    for class_index, (center, outcome) in enumerate(zip(centers, outcomes, strict=True)):
        values = center + 0.3 * torch.randn(rows_per_class, 3, generator=generator)
        features.append(values)
        labels.extend([outcome] * rows_per_class)
        identifiers.extend(f"{prefix}-{class_index}-{index}" for index in range(rows_per_class))
    partition = {
        "train": "T-controller-train",
        "cal": "T-controller-calibration",
        "eval": "T-dev",
    }[prefix]
    return ProbeDataset(
        tuple(identifiers),
        torch.cat(features),
        tuple(labels),
        group_ids=tuple(identifiers),
        feature_schema=ActivationFeatureSchema.synthetic(partition=partition, width=3),
    )


class ProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.training = clustered_dataset("train", 16)
        self.calibration = clustered_dataset("cal", 8, offset=1)
        self.evaluation = clustered_dataset("eval", 8, offset=2)

    def test_temperature_calibrated_logistic_probe_and_metrics(self) -> None:
        probe = fit_calibrated_probe(
            self.training,
            self.calibration,
            task=ProbeTask.CORRECT_INCORRECT_ABSTENTION,
            training_config=ProbeTrainingConfig(epochs=120),
        )
        probabilities = probe.predict_probabilities(self.evaluation.features)
        self.assertTrue(torch.allclose(probabilities.sum(1), torch.ones(24), atol=1e-6))
        metrics = evaluate_probe(probe, self.evaluation)
        self.assertGreater(metrics.macro_auroc, 0.99)
        self.assertGreater(metrics.macro_f1, 0.95)
        gate = separability_gate(metrics, {"entropy": 0.65, "max_token": 0.7})
        self.assertTrue(gate.passed)

    def test_mlp_isotonic_and_attempt_abstention_task(self) -> None:
        probe = fit_calibrated_probe(
            self.training,
            self.calibration,
            task=ProbeTask.ATTEMPT_ABSTENTION,
            training_config=ProbeTrainingConfig(
                kind=ProbeKind.TWO_LAYER_MLP,
                hidden_width=8,
                epochs=100,
            ),
            calibration_kind=CalibrationKind.ISOTONIC,
        )
        features, labels = encode_probe_task(self.evaluation, probe.task)
        self.assertEqual(set(labels.tolist()), {0, 1})
        probabilities = probe.predict_probabilities(features)
        self.assertTrue(torch.allclose(probabilities.sum(1), torch.ones(len(labels)), atol=1e-6))

    def test_training_and_calibration_ids_must_be_disjoint(self) -> None:
        with self.assertRaises(DataValidationError):
            fit_calibrated_probe(
                self.training,
                self.training,
                task=ProbeTask.CORRECT_INCORRECT_ABSTENTION,
                training_config=ProbeTrainingConfig(epochs=1),
            )

    def test_semantic_groups_and_fingerprints_cannot_bypass_split_isolation(self) -> None:
        bad_groups = (self.training.group_ids[0], *self.calibration.group_ids[1:])
        contaminated = ProbeDataset(
            self.calibration.question_ids,
            self.calibration.features,
            self.calibration.outcomes,
            group_ids=bad_groups,
            feature_schema=self.calibration.feature_schema,
        )
        with self.assertRaises(DataValidationError):
            fit_calibrated_probe(
                self.training,
                contaminated,
                task=ProbeTask.CORRECT_INCORRECT_ABSTENTION,
                training_config=ProbeTrainingConfig(epochs=1),
            )
        with self.assertRaises(DataValidationError):
            ProbeDataset(
                self.training.question_ids,
                self.training.features,
                self.training.outcomes,
                group_ids=self.training.group_ids,
                feature_schema=self.training.feature_schema,
                data_fingerprint="f" * 64,
            )

    def test_probe_artifact_round_trip_and_tamper_detection(self) -> None:
        probe = fit_calibrated_probe(
            self.training,
            self.calibration,
            task=ProbeTask.CORRECT_INCORRECT_ABSTENTION,
            training_config=ProbeTrainingConfig(epochs=80),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe"
            save_calibrated_probe(path, probe)
            loaded = load_calibrated_probe(
                path,
                expected_training_fingerprint=self.training.data_fingerprint,
                expected_calibration_fingerprint=self.calibration.data_fingerprint,
            )
            self.assertTrue(
                torch.allclose(
                    probe.predict_probabilities(self.evaluation.features),
                    loaded.predict_probabilities(self.evaluation.features),
                )
            )
            metadata_path = path / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["task"] = ProbeTask.CORRECT_INCORRECT.value
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaises(FrozenArtifactError):
                load_calibrated_probe(path)


if __name__ == "__main__":
    unittest.main()

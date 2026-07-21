from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from mfh.contracts import ActivationSite, Outcome, TokenScope
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.inference.architecture import HookKey
from mfh.methods.adaptive import (
    AdaptiveController,
    AdaptiveRouter,
    AlphaController,
    AlphaMode,
    RouterKind,
    assign_to_vector_regions,
    fit_adaptive_router,
    fit_layer_selector,
    fit_routed_vector_bank,
    load_adaptive_controller,
    load_adaptive_router,
    load_routed_vector_bank,
    save_adaptive_controller,
    save_adaptive_router,
    save_routed_vector_bank,
)
from mfh.methods.features import ActivationFeatureSchema
from mfh.methods.probes import (
    CalibratedProbe,
    ProbeDataset,
    ProbeTask,
    ProbeTrainingConfig,
    fit_calibrated_probe,
)
from mfh.provenance import stable_hash


def semantic_rows() -> tuple[ProbeDataset, dict[HookKey, torch.Tensor]]:
    generator = torch.Generator().manual_seed(7)
    features: list[torch.Tensor] = []
    first: list[torch.Tensor] = []
    second: list[torch.Tensor] = []
    outcomes: list[Outcome] = []
    for cluster, center in enumerate((-3.0, 3.0)):
        for outcome in (Outcome.CORRECT, Outcome.INCORRECT):
            for _ in range(8):
                features.append(
                    torch.tensor([center, 0.0]) + 0.08 * torch.randn(2, generator=generator)
                )
                if cluster == 0:
                    first.append(
                        torch.tensor([2.0, 0.0]) if outcome is Outcome.CORRECT else torch.zeros(2)
                    )
                    second.append(
                        torch.tensor([0.0, 1.0]) if outcome is Outcome.CORRECT else torch.zeros(2)
                    )
                else:
                    first.append(
                        torch.tensor([0.0, 2.0]) if outcome is Outcome.CORRECT else torch.zeros(2)
                    )
                    second.append(
                        torch.tensor([1.0, 0.0]) if outcome is Outcome.CORRECT else torch.zeros(2)
                    )
                outcomes.append(outcome)
    feature_values = torch.stack(features)
    identifiers = tuple(f"steer-{index}" for index in range(len(outcomes)))
    return (
        ProbeDataset(
            identifiers,
            feature_values,
            tuple(outcomes),
            group_ids=identifiers,
            feature_schema=ActivationFeatureSchema.synthetic(partition="T-steer", width=2),
        ),
        {
            HookKey(1, ActivationSite.POST_MLP): torch.stack(first),
            HookKey(2, ActivationSite.POST_MLP): torch.stack(second),
        },
    )


def risk_probe(
    task: ProbeTask = ProbeTask.CORRECT_INCORRECT_ABSTENTION,
) -> CalibratedProbe:
    centers = {
        Outcome.CORRECT: torch.tensor([-2.0, -2.0]),
        Outcome.INCORRECT: torch.tensor([2.0, 2.0]),
        Outcome.ABSTENTION: torch.tensor([-2.0, 2.0]),
    }

    def dataset(prefix: str, seed: int) -> ProbeDataset:
        generator = torch.Generator().manual_seed(seed)
        features: list[torch.Tensor] = []
        outcomes: list[Outcome] = []
        ids: list[str] = []
        for outcome, center in centers.items():
            for index in range(8):
                features.append(center + 0.15 * torch.randn(2, generator=generator))
                outcomes.append(outcome)
                ids.append(f"{prefix}-{outcome.value}-{index}")
        partition = "T-controller-train" if prefix == "train" else "T-controller-calibration"
        return ProbeDataset(
            tuple(ids),
            torch.stack(features),
            tuple(outcomes),
            group_ids=tuple(ids),
            feature_schema=ActivationFeatureSchema.synthetic(partition=partition, width=2),
        )

    return fit_calibrated_probe(
        dataset("train", 31),
        dataset("cal", 32),
        task=task,
        training_config=ProbeTrainingConfig(epochs=100),
    )


class AdaptiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset, self.activations = semantic_rows()
        self.features = self.dataset.features
        self.bank, self.assignments = fit_routed_vector_bank(
            self.dataset,
            self.activations,
            cluster_count=2,
        )
        self.bank_fingerprint = self.dataset.data_fingerprint
        self.controller_dataset = ProbeDataset(
            tuple(f"controller-{index}" for index in range(len(self.dataset.question_ids))),
            self.features.clone(),
            self.dataset.outcomes,
            group_ids=tuple(
                f"controller-group-{index}" for index in range(len(self.dataset.question_ids))
            ),
            feature_schema=ActivationFeatureSchema.synthetic(
                partition="T-controller-train", width=2
            ),
        )
        self.controller_assignments = assign_to_vector_regions(self.controller_dataset, self.bank)
        self.fingerprint = self.controller_dataset.data_fingerprint

    def test_vector_bank_builds_region_specific_unit_directions(self) -> None:
        self.assertEqual(self.bank.cluster_count, 2)
        for values in self.bank.directions.values():
            self.assertTrue(torch.allclose(torch.linalg.vector_norm(values, dim=1), torch.ones(2)))
        nearest = fit_adaptive_router(
            self.controller_dataset,
            self.controller_assignments,
            self.bank.centers,
            kind=RouterKind.NEAREST_CENTROID,
        )
        weights = nearest.weights(torch.tensor([[-3.0, 0.0], [3.0, 0.0]]))
        self.assertTrue(torch.all(weights.max(dim=1).values > 0.99))
        mixed = self.bank.mix(weights)
        self.assertEqual(next(iter(mixed.values())).shape, (2, 2))

    def test_vector_bank_projects_abstentions_from_signed_source_rows(self) -> None:
        abstention_id = "steer-abstention"
        dataset = ProbeDataset(
            (*self.dataset.question_ids, abstention_id),
            torch.cat((self.dataset.features, torch.tensor([[99.0, 99.0]]))),
            (*self.dataset.outcomes, Outcome.ABSTENTION),
            group_ids=(*self.dataset.group_ids, abstention_id),
            feature_schema=self.dataset.feature_schema,
        )
        activations = {
            key: torch.cat((value, torch.tensor([[999.0, 999.0]])))
            for key, value in self.activations.items()
        }
        projected, assignments = fit_routed_vector_bank(
            dataset, activations, cluster_count=2
        )

        self.assertEqual(projected.data_fingerprint, dataset.data_fingerprint)
        self.assertEqual(len(assignments), len(self.dataset.question_ids))
        self.assertTrue(torch.allclose(projected.centers, self.bank.centers))
        for key in projected.directions:
            self.assertTrue(
                torch.allclose(projected.directions[key], self.bank.directions[key])
            )

    def test_linear_mlp_and_layer_routers(self) -> None:
        for kind in (RouterKind.LINEAR_SOFTMAX, RouterKind.TWO_LAYER_MLP):
            router = fit_adaptive_router(
                self.controller_dataset,
                self.controller_assignments,
                self.bank.centers,
                kind=kind,
                hidden_width=8,
                epochs=80,
            )
            predicted = router.weights(self.features).argmax(1)
            self.assertGreater(
                float((predicted == self.controller_assignments).float().mean()), 0.95
            )

        left_cluster = torch.cdist(self.features, torch.tensor([[-3.0, 0.0]])).squeeze(1) < 1
        best_layers = tuple(1 if bool(left) else 2 for left in left_cluster)
        selector = fit_layer_selector(
            self.controller_dataset,
            best_layers,
            candidate_layers=(1, 2),
            kind=RouterKind.NEAREST_CENTROID,
        )
        self.assertEqual(selector.select(torch.tensor([[-3.0, 0.0], [3.0, 0.0]])).tolist(), [1, 2])

    def test_dynamic_alpha_and_adaptive_plans(self) -> None:
        risks = torch.tensor([0.1, 0.5, 0.9])
        gated = AlphaController(AlphaMode.RISK_GATED, alpha_max=2.0, beta=10, threshold=0.5)
        values = gated.alpha(risks)
        self.assertAlmostEqual(float(values[1]), 1.0, places=6)
        self.assertLess(float(values[0]), float(values[1]))
        hard = AlphaController(AlphaMode.HARD_THRESHOLD, alpha_max=2.0, threshold=0.5)
        self.assertEqual(float(hard.alpha(risks)[0]), 0.0)
        with self.assertRaisesRegex(DataValidationError, "exact AlphaMode"):
            AlphaController("fixed", alpha_max=1.0)  # type: ignore[arg-type]

        with self.assertRaisesRegex(DataValidationError, "exact RouterKind"):
            AdaptiveRouter(
                kind="nearest_centroid",  # type: ignore[arg-type]
                centers=torch.ones(1, 2),
                training_fingerprint="a" * 64,
                feature_schema=ActivationFeatureSchema.synthetic(
                    partition="T-controller-train", width=2
                ),
            )

        router = fit_adaptive_router(
            self.controller_dataset,
            self.controller_assignments,
            self.bank.centers,
            kind=RouterKind.NEAREST_CENTROID,
        )
        controller = AdaptiveController(
            risk_probe=risk_probe(),
            vector_bank=self.bank,
            vector_router=router,
            alpha_controller=gated,
            fixed_layer=1,
        )
        decision = controller.decide(torch.tensor([[-2.0, -2.0], [2.0, 2.0]]))
        self.assertLess(float(decision.alphas[0]), float(decision.alphas[1]))
        plans = decision.plans_for_row(1, token_scope=TokenScope.FIRST_FOUR)
        self.assertEqual({key.layer for key in plans}, {1})
        with self.assertRaises(DataValidationError):
            AdaptiveController(
                risk_probe=risk_probe(ProbeTask.CORRECT_INCORRECT),
                vector_bank=self.bank,
                vector_router=router,
                alpha_controller=gated,
                fixed_layer=1,
            )
        reordered_router = fit_adaptive_router(
            self.controller_dataset,
            self.controller_assignments,
            self.bank.centers.flip(0),
            kind=RouterKind.NEAREST_CENTROID,
        )
        with self.assertRaises(DataValidationError):
            AdaptiveController(
                risk_probe=risk_probe(),
                vector_bank=self.bank,
                vector_router=reordered_router,
                alpha_controller=gated,
                fixed_layer=1,
            )

    def test_routed_vector_artifact_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "routed"
            save_routed_vector_bank(path, self.bank)
            loaded = load_routed_vector_bank(path, expected_data_fingerprint=self.bank_fingerprint)
            self.assertTrue(torch.equal(loaded.centers, self.bank.centers))
            self.assertEqual(loaded.directions.keys(), self.bank.directions.keys())

            router = fit_adaptive_router(
                self.controller_dataset,
                self.controller_assignments,
                self.bank.centers,
                kind=RouterKind.LINEAR_SOFTMAX,
                epochs=40,
            )
            router_path = Path(directory) / "router"
            save_adaptive_router(router_path, router)
            loaded_router = load_adaptive_router(
                router_path, expected_training_fingerprint=self.fingerprint
            )
            self.assertTrue(
                torch.equal(router.weights(self.features), loaded_router.weights(self.features))
            )

            controller = AdaptiveController(
                risk_probe=risk_probe(),
                vector_bank=self.bank,
                vector_router=router,
                alpha_controller=AlphaController(
                    AlphaMode.HARD_THRESHOLD, alpha_max=1.5, beta=9, threshold=0.4
                ),
                fixed_layer=1,
            )
            controller_path = Path(directory) / "controller"
            save_adaptive_controller(controller_path, controller)
            loaded_controller = load_adaptive_controller(controller_path)
            expected = controller.decide(torch.tensor([[2.0, 2.0]]))
            actual = loaded_controller.decide(torch.tensor([[2.0, 2.0]]))
            self.assertTrue(torch.equal(expected.probabilities, actual.probabilities))
            self.assertTrue(torch.equal(expected.alphas, actual.alphas))

    def test_controller_loader_rejects_rehashed_boolean_numeric_metadata(self) -> None:
        router = fit_adaptive_router(
            self.controller_dataset,
            self.controller_assignments,
            self.bank.centers,
            kind=RouterKind.NEAREST_CENTROID,
        )
        controller = AdaptiveController(
            risk_probe=risk_probe(),
            vector_bank=self.bank,
            vector_router=router,
            alpha_controller=AlphaController(AlphaMode.FIXED, alpha_max=1.0),
            fixed_layer=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controller"
            save_adaptive_controller(path, controller)
            metadata_path = path / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.pop("metadata_digest")
            metadata["fixed_layer"] = True
            metadata["metadata_digest"] = stable_hash(metadata)
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FrozenArtifactError, "metadata types differ"):
                load_adaptive_controller(path)

    def test_feature_schema_loader_rejects_boolean_dimensions(self) -> None:
        value = ActivationFeatureSchema.synthetic(partition="T-steer", width=2).to_dict()
        value["width"] = True
        with self.assertRaisesRegex(DataValidationError, "JSON types differ"):
            ActivationFeatureSchema.from_dict(value)


if __name__ == "__main__":
    unittest.main()

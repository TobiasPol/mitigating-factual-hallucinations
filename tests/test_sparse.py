from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from mfh.contracts import (
    ActivationSite,
    GenerationRecord,
    Outcome,
    Question,
    Runtime,
    TokenScope,
)
from mfh.data.io import write_questions
from mfh.data.splits import semantic_group_ids
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.methods.features import ActivationFeatureSchema
from mfh.methods.probes import ProbeDataset
from mfh.methods.sae_stability import (
    _aligned_stability,
    load_sae_stability_bundle,
    write_sae_stability_bundle,
)
from mfh.methods.sparse import (
    ActivationBatch,
    CoordinateScreenPoint,
    SAEConfig,
    SAEPromotionCriteria,
    SAESparsity,
    SAESparsitySweepPoint,
    SeedFeatureSelection,
    _create_long_computation_receipt,
    _validate_coordinate_screen_execution_record,
    activation_capture_execution_receipt_body,
    coordinate_screen_condition_id,
    coordinate_screen_contract_digest,
    coordinate_screen_execution_receipt_body,
    coordinate_sparse_direction,
    decode_latent_direction,
    decoder_feature_direction,
    fit_coordinate_sparse_artifact,
    fit_e7_sae_sweep_measured,
    fit_sparse_autoencoder,
    fit_sparse_autoencoder_corpus,
    latent_factuality_direction,
    load_activation_corpus,
    load_coordinate_sparse_artifact,
    load_sae,
    load_sae_intervention,
    measure_feature_intervention_evidence,
    promote_sae_intervention,
    sae_checkpoint_fingerprint,
    sae_config_fingerprint,
    save_coordinate_sparse_artifact,
    save_sae,
    save_sae_intervention,
    selected_feature_stability,
    standardized_effect_size,
    suppress_latent_features,
    write_activation_corpus,
)
from mfh.provenance import canonical_json, sha256_file, stable_hash


class CoordinateSparseTests(unittest.TestCase):
    def test_aligned_stability_is_permutation_invariant(self) -> None:
        left = torch.eye(3)
        right = left[[2, 0, 1]]
        self.assertEqual(_aligned_stability((left, right)), 1.0)
        changed = right.clone()
        changed[0] = torch.tensor([-1.0, 0.0, 0.0])
        self.assertLess(_aligned_stability((left, changed)), 1.0)

    def test_standardized_effect_and_fractional_mask(self) -> None:
        correct = torch.tensor([[5.0, 1.0, 2.0, 0.0], [7.0, 1.2, 2.0, 1.0]])
        incorrect = torch.tensor([[1.0, 1.0, 2.0, 0.0], [2.0, 0.8, 2.0, 1.0]])
        effect = standardized_effect_size(correct, incorrect)
        dense = torch.tensor([0.5, 0.5, 0.5, 0.5])
        sparse = coordinate_sparse_direction(dense, effect, retained_fraction=0.25)
        self.assertEqual(sparse.retained_dimensions, 1)
        self.assertTrue(sparse.mask[0])
        self.assertEqual(int((sparse.direction != 0).sum()), 1)
        dataset = ProbeDataset(
            ("c0", "c1", "i0", "i1"),
            torch.cat((correct, incorrect)),
            (Outcome.CORRECT, Outcome.CORRECT, Outcome.INCORRECT, Outcome.INCORRECT),
            group_ids=("c0", "c1", "i0", "i1"),
            feature_schema=ActivationFeatureSchema.synthetic(partition="T-steer", width=4),
        )
        private_key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex("1" * 64))
        execution_public_key = private_key.public_key().public_bytes(
            Encoding.Raw, PublicFormat.Raw
        ).hex()
        runtime_sha = "e" * 64
        baseline_condition_id = "b" * 64
        points = tuple(
            CoordinateScreenPoint(
                retained_fraction=fraction,
                alpha=alpha,
                baseline_condition_id=baseline_condition_id,
                intervention_condition_id=stable_hash(
                    {"fraction": fraction, "alpha": alpha}
                ),
                question_ids=dataset.question_ids,
                baseline_outcomes=dataset.outcomes,
                intervention_outcomes=(
                    (Outcome.CORRECT,) * len(dataset.question_ids)
                    if (fraction, alpha) == (0.25, 0.5)
                    else dataset.outcomes
                ),
            )
            for fraction in (0.01, 0.05, 0.10, 0.25)
            for alpha in (0.1, 0.25, 0.5, 1.0, 2.0)
        )
        source_index = ("P0-synthetic", "M1-P", "post_mlp", 0)
        direction_sha = hashlib.sha256(
            dense.numpy().astype(np.float32).tobytes()
        ).hexdigest()
        contract_digest = coordinate_screen_contract_digest(
            feature_schema=dataset.feature_schema,
            source_artifact_sha256="a" * 64,
            source_tensor_index=source_index,
            source_direction_sha256=direction_sha,
            layer=0,
            site=ActivationSite.POST_MLP,
            token_scope=TokenScope.FIRST_FOUR,
            runtime_artifact_sha256=runtime_sha,
            execution_public_key=execution_public_key,
            points=points,
        )
        baseline_condition_id = coordinate_screen_condition_id(contract_digest)
        points = tuple(
            replace(
                point,
                baseline_condition_id=baseline_condition_id,
                intervention_condition_id=coordinate_screen_condition_id(
                    contract_digest,
                    retained_fraction=point.retained_fraction,
                    alpha=point.alpha,
                ),
            )
            for point in points
        )

        def record(
            question_id: str,
            outcome: Outcome,
            *,
            condition_id: str,
            method: str,
            alpha: float = 0.0,
            sparsity: float | None = None,
        ) -> GenerationRecord:
            intervened = method == "M4a"
            trace = (
                {
                    "coordinate_screen_contract_digest": contract_digest,
                    "source_artifact_sha256": "a" * 64,
                    "direction_sha256": "9" * 64,
                    "layer": 0,
                    "site": ActivationSite.POST_MLP.value,
                    "token_scope": TokenScope.FIRST_FOUR.value,
                    "standardized_alpha": alpha,
                    "raw_alpha": alpha,
                    "retained_fraction": sparsity,
                    "reference_rms": 1.0,
                    "source_direction_norm": 1.0,
                    "applied_tokens": 1,
                    "applied_token_indices": [0],
                    "pre_activation_sha256": "7" * 64,
                    "post_activation_sha256": "8" * 64,
                    "delta_sha256": "6" * 64,
                }
                if intervened
                else None
            )
            metadata = {
                "coordinate_screen_contract_digest": contract_digest,
                "coordinate_screen_runtime_artifact_sha256": runtime_sha,
                "coordinate_screen_execution_public_key": execution_public_key,
                "prompt_template_sha256": dataset.feature_schema.prompt_sha256,
                "generation_runtime_metrics": {
                    "schema_version": 1,
                    "unified_memory_bytes": 1_000_000,
                    "peak_memory_bytes": 1_000,
                    "generation_peak_memory_bytes": 1_000,
                    "auxiliary_peak_memory_bytes": 0,
                    "active_memory_bytes": 800,
                    "cache_memory_bytes": 200,
                    "prompt_tokens_per_second": 10.0,
                    "generation_tokens_per_second": 10.0,
                    "generation_wall_time_seconds": 0.1,
                    "stop_type": "length",
                    "stopping_token_id": None,
                },
                **(
                    {
                        "intervention_trace": trace,
                        "intervention_trace_digest": stable_hash(trace),
                    }
                    if trace is not None
                    else {}
                ),
            }
            unsigned = GenerationRecord(
                question_id=question_id,
                benchmark="synthetic",
                model_repository="synthetic/model",
                model_revision="0" * 40,
                runtime=Runtime.SYNTHETIC,
                quantization="none",
                system_prompt_id="P0-synthetic",
                rendered_prompt_hash="d" * 64,
                steering_method=method,
                layer=0 if intervened else None,
                site=ActivationSite.POST_MLP if intervened else None,
                token_scope=TokenScope.FIRST_FOUR if intervened else None,
                alpha=alpha,
                sparsity=sparsity,
                controller_scores={},
                raw_output=outcome.value,
                normalized_answer=outcome.value,
                outcome=outcome,
                generation_latency_seconds=0.1,
                input_tokens=1,
                output_tokens=1,
                condition_id=condition_id,
                metadata=metadata,
            )
            signature = private_key.sign(
                canonical_json(
                    coordinate_screen_execution_receipt_body(
                        unsigned,
                        contract_digest=contract_digest,
                        runtime_artifact_sha256=runtime_sha,
                    )
                ).encode()
            ).hex()
            return replace(
                unsigned,
                metadata={
                    **dict(unsigned.metadata),
                    "coordinate_screen_execution_signature": signature,
                },
            )

        records = [
            record(
                question_id,
                outcome,
                condition_id=baseline_condition_id,
                method="M0",
            )
            for question_id, outcome in zip(
                dataset.question_ids, dataset.outcomes, strict=True
            )
        ]
        for point in points:
            records.extend(
                record(
                    question_id,
                    outcome,
                    condition_id=point.intervention_condition_id,
                    method="M4a",
                    alpha=point.alpha,
                    sparsity=point.retained_fraction,
                )
                for question_id, outcome in zip(
                    point.question_ids, point.intervention_outcomes, strict=True
                )
            )
        artifact = fit_coordinate_sparse_artifact(
            dataset,
            dense,
            screen_points=points,
            screen_records=records,
            source_artifact_sha256="a" * 64,
            source_tensor_index=source_index,
            source_direction_sha256=direction_sha,
            reference_rms=1.0,
            layer=0,
            site=ActivationSite.POST_MLP,
            token_scope=TokenScope.FIRST_FOUR,
            screen_runtime_artifact_sha256=runtime_sha,
            screen_execution_public_key=execution_public_key,
        )
        source = records[0]
        forged_metrics = dict(source.metadata["generation_runtime_metrics"])
        forged_metrics["unified_memory_bytes"] = 2_000_000
        forged = replace(
            source,
            metadata={
                **dict(source.metadata),
                "generation_runtime_metrics": forged_metrics,
            },
        )
        forged = replace(
            forged,
            metadata={
                **dict(forged.metadata),
                "coordinate_screen_execution_signature": private_key.sign(
                    canonical_json(
                        coordinate_screen_execution_receipt_body(
                            forged,
                            contract_digest=contract_digest,
                            runtime_artifact_sha256=runtime_sha,
                        )
                    ).encode()
                ).hex(),
            },
        )
        with self.assertRaisesRegex(DataValidationError, "differs from attestation"):
            _validate_coordinate_screen_execution_record(
                forged,
                contract_digest=contract_digest,
                runtime_artifact_sha256=runtime_sha,
                execution_public_key=execution_public_key,
                prompt_template_sha256=dataset.feature_schema.prompt_sha256,
                runtime_identity={"unified_memory_bytes": 1_000_000},
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "coordinate"
            save_coordinate_sparse_artifact(path, artifact)
            loaded = load_coordinate_sparse_artifact(path)
            self.assertTrue(torch.equal(loaded.sparse_direction.mask, sparse.mask))
            self.assertEqual((loaded.sparse_direction.retained_fraction, loaded.alpha), (0.25, 0.5))


class SparseAutoencoderTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(41)
        sources = torch.randn(160, 2, generator=generator)
        mixing = torch.tensor([[1.0, 0.0, 0.8, -0.2], [0.0, 1.0, 0.3, 0.9]])
        self.values = sources @ mixing + 0.02 * torch.randn(160, 4, generator=generator)
        self.outcomes = tuple(
            Outcome.CORRECT if value > 0 else Outcome.INCORRECT for value in sources[:, 0]
        )
        self.config = SAEConfig(
            input_width=4,
            expansion_factor=2,
            top_k=2,
            epochs=50,
            batch_size=40,
            learning_rate=0.01,
            seed=9,
        )
        self.training_schema = ActivationFeatureSchema.synthetic(partition="sae-train", width=4)
        self.validation_schema = ActivationFeatureSchema.synthetic(
            partition="sae-validation", width=4
        )
        self.selection_dataset = ProbeDataset(
            tuple(f"steer-{index}" for index in range(len(self.outcomes))),
            self.values,
            self.outcomes,
            group_ids=tuple(f"steer-group-{index}" for index in range(len(self.outcomes))),
            feature_schema=ActivationFeatureSchema.synthetic(partition="T-steer", width=4),
        )

    def _write_signed_tsteer_selection(
        self, root: Path
    ) -> tuple[Path, Path]:
        questions = tuple(
            Question(
                question_id=question_id,
                benchmark="synthetic",
                text=f"Synthetic question {index}?",
                aliases=(f"uniqueanswer{index}",),
                split="T-steer",
            )
            for index, question_id in enumerate(self.selection_dataset.question_ids)
        )
        question_path = root / "tsteer.jsonl"
        write_questions(question_path, questions)
        groups = semantic_group_ids(questions)
        private_key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex("3" * 64))
        public_key = private_key.public_key().public_bytes(
            Encoding.Raw, PublicFormat.Raw
        ).hex()
        runtime_sha = "e" * 64
        source_sha = sha256_file(question_path)
        receipts = []
        for row, (question, outcome) in enumerate(
            zip(questions, self.selection_dataset.outcomes, strict=True)
        ):
            raw_output = (
                question.aliases[0] if outcome is Outcome.CORRECT else "wrong answer"
            )
            question_sha = stable_hash(
                {
                    "question_id": question.question_id,
                    "benchmark": question.benchmark,
                    "text": question.text,
                    "aliases": list(question.aliases),
                    "split": question.split,
                    "entities": list(question.entities),
                    "metadata": dict(question.metadata),
                }
            )
            value = np.ascontiguousarray(
                self.selection_dataset.features[row].numpy().astype(np.float32)
            )
            body = activation_capture_execution_receipt_body(
                question_id=question.question_id,
                group_id=groups[question.question_id],
                outcome=outcome,
                rendered_prompt_sha256=stable_hash(
                    {"question_id": question.question_id, "prompt": "synthetic"}
                ),
                activation_sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest(),
                feature_schema=self.selection_dataset.feature_schema,
                runtime_artifact_sha256=runtime_sha,
                execution_public_key=public_key,
                source_question_bundle_sha256=source_sha,
                dtype="float32",
                label_evidence={
                    "raw_output": raw_output,
                    "normalized_answer": raw_output,
                    "aliases": list(question.aliases),
                    "source_question_sha256": question_sha,
                },
            )
            receipts.append(
                {
                    "body": body,
                    "signature": private_key.sign(canonical_json(body).encode()).hex(),
                }
            )
        batch = ActivationBatch(
            question_ids=tuple(question.question_id for question in questions),
            activations=self.selection_dataset.features,
            outcomes=self.selection_dataset.outcomes,
            group_ids=tuple(groups[question.question_id] for question in questions),
            capture_receipts=tuple(receipts),
        )
        corpus_path = root / "tsteer-corpus"
        write_activation_corpus(
            corpus_path,
            (batch,),
            feature_schema=self.selection_dataset.feature_schema,
            shard_rows=37,
            dtype="float32",
            runtime_artifact_sha256=runtime_sha,
            execution_public_key=public_key,
            source_question_bundle_sha256=source_sha,
            capture_signer=lambda body: private_key.sign(
                canonical_json(body).encode()
            ).hex(),
        )
        return corpus_path, question_path

    def test_topk_training_metrics_directions_and_causal_checks(self) -> None:
        result = fit_sparse_autoencoder(
            self.values[:120],
            self.values[120:],
            self.config,
            training_schema=self.training_schema,
            validation_schema=self.validation_schema,
        )
        self.assertLess(result.loss_history[-1], result.loss_history[0])
        self.assertLessEqual(result.metrics.average_active_features, 2.0)
        self.assertGreater(result.metrics.fraction_variance_explained, 0.5)
        latent = latent_factuality_direction(result.model, self.selection_dataset, feature_count=2)
        decoded = decode_latent_direction(result.model, latent)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(decoded)), 1.0, places=5)
        feature = decoder_feature_direction(result.model, latent.selected_features[0])
        self.assertAlmostEqual(float(torch.linalg.vector_norm(feature)), 1.0, places=5)
        encoded = result.model.encode(self.values[:3])
        suppressed = suppress_latent_features(encoded, latent.selected_features)
        self.assertTrue((suppressed[:, list(latent.selected_features)] == 0).all())
        selections = (
            SeedFeatureSelection(1, "a" * 64, (1, 2)),
            SeedFeatureSelection(2, "b" * 64, (2, 3)),
        )
        self.assertAlmostEqual(selected_feature_stability(selections), 1 / 3)
        with self.assertRaises(DataValidationError):
            selected_feature_stability(
                (
                    SeedFeatureSelection(1, "a" * 64, (1, 2)),
                    SeedFeatureSelection(1, "b" * 64, (2, 3)),
                )
            )

    def test_l1_mode_and_frozen_sae_round_trip(self) -> None:
        config = SAEConfig(
            input_width=4,
            expansion_factor=2,
            sparsity=SAESparsity.L1,
            top_k=2,
            l1_coefficient=0.01,
            epochs=8,
            batch_size=80,
            learning_rate=0.01,
        )
        result = fit_sparse_autoencoder(
            self.values[:120],
            self.values[120:],
            config,
            training_schema=self.training_schema,
            validation_schema=self.validation_schema,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sae"
            save_sae(path, result)
            loaded = load_sae(
                path,
                expected_training_fingerprint=result.training_fingerprint,
                expected_validation_fingerprint=result.validation_fingerprint,
            )
            original = result.model(self.values[:4])[0]
            restored = loaded.model(self.values[:4])[0]
            self.assertTrue(torch.equal(original, restored))
            tensor_path = path / "sae.safetensors"
            tensor_path.write_bytes(tensor_path.read_bytes() + b"tamper")
            with self.assertRaises(FrozenArtifactError):
                load_sae(path)

    def test_stability_bundle_recomputes_real_seed_checkpoint_fingerprints(self) -> None:
        first = fit_sparse_autoencoder(
            self.values[:120],
            self.values[120:],
            replace(self.config, epochs=4, seed=9),
            training_schema=self.training_schema,
            validation_schema=self.validation_schema,
        )
        second = fit_sparse_autoencoder(
            self.values[:120],
            self.values[120:],
            replace(self.config, epochs=4, seed=10),
            training_schema=self.training_schema,
            validation_schema=self.validation_schema,
        )
        with tempfile.TemporaryDirectory() as directory:
            selection_corpus, selection_questions = self._write_signed_tsteer_selection(
                Path(directory)
            )
            bundle = Path(directory) / "seed-bundle"
            write_sae_stability_bundle(
                bundle,
                runs_by_model={"synthetic/model": (first, second)},
                selection_corpora={"synthetic/model": selection_corpus},
                selection_question_sources={
                    "synthetic/model": selection_questions
                },
                feature_count=2,
                promoted_method_artifacts={"synthetic/model": "a" * 64},
            )
            loaded = load_sae_stability_bundle(bundle)
            self.assertEqual(len(loaded.selections_by_model["synthetic/model"]), 2)
            self.assertGreaterEqual(loaded.stability_by_model["synthetic/model"], 0.0)
            self.assertLessEqual(loaded.stability_by_model["synthetic/model"], 1.0)
            manifest_path = bundle / "manifest.json"
            original_manifest = manifest_path.read_text(encoding="utf-8")
            false_manifest = json.loads(original_manifest)
            false_manifest["models"][0]["runs"][0]["selected_features"].reverse()
            body = {key: value for key, value in false_manifest.items() if key != "manifest_digest"}
            false_manifest["manifest_digest"] = stable_hash(body)
            manifest_path.write_text(
                json.dumps(false_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(FrozenArtifactError):
                load_sae_stability_bundle(bundle)
            manifest_path.write_text(original_manifest, encoding="utf-8")
            tensor = next((bundle / "runs").glob("*/sae.safetensors"))
            tensor.write_bytes(tensor.read_bytes() + b"tamper")
            with self.assertRaises(FrozenArtifactError):
                load_sae_stability_bundle(bundle)

    def test_causal_promotion_and_full_intervention_artifact(self) -> None:
        result = fit_sparse_autoencoder(
            self.values[:120],
            self.values[120:],
            self.config,
            training_schema=self.training_schema,
            validation_schema=self.validation_schema,
        )
        latent = latent_factuality_direction(result.model, self.selection_dataset, feature_count=2)
        baseline = {"q0": Outcome.CORRECT, "q1": Outcome.INCORRECT}
        activated = {"q0": Outcome.CORRECT, "q1": Outcome.CORRECT}
        suppressed = {"q0": Outcome.INCORRECT, "q1": Outcome.INCORRECT}
        protected = {"language": {"q0": True, "q1": True}}
        evidence_schema = ActivationFeatureSchema.synthetic(partition="T-dev", width=4)
        evidence = tuple(
            measure_feature_intervention_evidence(
                feature,
                baseline_outcomes=baseline,
                activated_outcomes=activated,
                suppressed_outcomes=suppressed,
                protected_baseline=protected,
                protected_activated=protected,
                protected_suppressed=protected,
                feature_schema=evidence_schema,
                alpha=1.0,
                token_scope=TokenScope.FIRST_FOUR,
                layer=0,
                site=ActivationSite.POST_MLP,
            )
            for feature in latent.selected_features
        )
        self.assertTrue(
            all(
                item.causally_supported(minimum_effect=0.25, maximum_protected_effect=0.01)
                for item in evidence
            )
        )
        criteria = SAEPromotionCriteria(
            minimum_fve=0.2,
            maximum_reconstruction_mse=1.0,
            maximum_average_active_features=2.0,
            minimum_feature_stability=0.8,
            minimum_causal_effect=0.25,
            maximum_protected_effect=0.01,
        )
        sweep_results = (
            fit_sparse_autoencoder(
                self.values[:120],
                self.values[120:],
                replace(self.config, top_k=1),
                training_schema=self.training_schema,
                validation_schema=self.validation_schema,
            ),
            result,
            fit_sparse_autoencoder(
                self.values[:120],
                self.values[120:],
                replace(self.config, top_k=3),
                training_schema=self.training_schema,
                validation_schema=self.validation_schema,
            ),
        )
        sweep = tuple(
            SAESparsitySweepPoint(
                config_fingerprint=sae_config_fingerprint(value.config),
                checkpoint_fingerprint=sae_checkpoint_fingerprint(value),
                fraction_variance_explained=(
                    value.metrics.fraction_variance_explained
                ),
                reconstruction_mse=value.metrics.reconstruction_mse,
                average_active_features=value.metrics.average_active_features,
                selected=value is result,
            )
            for value in sweep_results
        )
        intervention = promote_sae_intervention(
            result,
            latent,
            evidence=evidence,
            stability_selections=(
                SeedFeatureSelection(
                    result.config.seed,
                    sae_checkpoint_fingerprint(result),
                    latent.selected_features,
                ),
                SeedFeatureSelection(
                    result.config.seed + 1,
                    "f" * 64,
                    latent.selected_features,
                ),
            ),
            aligned_feature_stability=1.0,
            criteria=criteria,
            sparsity_sweep=sweep,
            sparsity_sweep_results=sweep_results,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "intervention"
            save_sae_intervention(path, intervention)
            loaded = load_sae_intervention(path)
            self.assertTrue(torch.equal(loaded.decoded_direction, intervention.decoded_direction))
            self.assertEqual(loaded.evidence[0].spec.digest, evidence[0].spec.digest)
            self.assertEqual(loaded.feature_stability, 1.0)
            self.assertEqual(
                loaded.feature_stability_method,
                "oriented_decoder_cosine_hungarian_min_pair_v1",
            )
            self.assertEqual(len(loaded.sparsity_sweep_results), 3)
            self.assertEqual(
                tuple(
                    sae_checkpoint_fingerprint(value)
                    for value in loaded.sparsity_sweep_results
                ),
                tuple(value.checkpoint_fingerprint for value in sweep),
            )
            metadata_path = path / "metadata.json"
            original_metadata = metadata_path.read_text(encoding="utf-8")
            legacy = json.loads(original_metadata)
            legacy["feature_stability"] = legacy.pop(
                "aligned_feature_stability"
            )
            legacy.pop("feature_stability_method")
            legacy.pop("metadata_digest")
            legacy["metadata_digest"] = stable_hash(legacy)
            metadata_path.write_text(
                json.dumps(legacy, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(FrozenArtifactError):
                load_sae_intervention(path)
            metadata_path.write_text(original_metadata, encoding="utf-8")
            sweep_tensor = path / "sparsity-sweep" / "000" / "sae.safetensors"
            sweep_tensor.write_bytes(sweep_tensor.read_bytes() + b"tamper")
            with self.assertRaises(FrozenArtifactError):
                load_sae_intervention(path)

    def test_causal_evidence_requires_valid_native_signature(self) -> None:
        private_key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex("2" * 64))
        public_key = private_key.public_key().public_bytes(
            Encoding.Raw, PublicFormat.Raw
        ).hex()
        schema = ActivationFeatureSchema.synthetic(partition="T-dev", width=4)

        def sign(body: object) -> str:
            return private_key.sign(canonical_json(body).encode()).hex()

        factual_ids = tuple(f"q{index}" for index in range(100))
        protected_ids = tuple(f"p{index}" for index in range(50))
        kwargs = {
            "baseline_outcomes": {
                question_id: (
                    Outcome.CORRECT if index % 2 == 0 else Outcome.INCORRECT
                )
                for index, question_id in enumerate(factual_ids)
            },
            "activated_outcomes": {
                question_id: Outcome.CORRECT for question_id in factual_ids
            },
            "suppressed_outcomes": {
                question_id: Outcome.INCORRECT for question_id in factual_ids
            },
            "protected_baseline": {
                "language": {question_id: True for question_id in protected_ids}
            },
            "protected_activated": {
                "language": {question_id: True for question_id in protected_ids}
            },
            "protected_suppressed": {
                "language": {question_id: True for question_id in protected_ids}
            },
            "feature_schema": schema,
            "alpha": 1.0,
            "token_scope": TokenScope.FIRST_FOUR,
            "layer": 0,
            "site": ActivationSite.POST_MLP,
            "runtime_artifact_sha256": "a" * 64,
            "execution_public_key": public_key,
            "source_question_bundle_sha256": schema.split_manifest_digest,
            "execution_signer": sign,
        }
        evidence = measure_feature_intervention_evidence(0, **kwargs)
        self.assertIsNotNone(evidence.execution_signature)
        with self.assertRaisesRegex(DataValidationError, "signature"):
            replace(evidence, execution_signature="0" * 128)
        too_small = {
            **kwargs,
            "baseline_outcomes": {"q0": Outcome.CORRECT, "q1": Outcome.INCORRECT},
            "activated_outcomes": {"q0": Outcome.CORRECT, "q1": Outcome.CORRECT},
            "suppressed_outcomes": {"q0": Outcome.INCORRECT, "q1": Outcome.INCORRECT},
            "protected_baseline": {"language": {"p0": True, "p1": True}},
            "protected_activated": {"language": {"p0": True, "p1": True}},
            "protected_suppressed": {"language": {"p0": True, "p1": True}},
        }
        with self.assertRaisesRegex(DataValidationError, "minimum sample"):
            measure_feature_intervention_evidence(0, **too_small)
        no_protected = {
            **kwargs,
            "protected_baseline": {},
            "protected_activated": {},
            "protected_suppressed": {},
        }
        with self.assertRaisesRegex(DataValidationError, "protected-sample"):
            measure_feature_intervention_evidence(0, **no_protected)

    def test_long_computation_receipt_is_runtime_signed(self) -> None:
        private_key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex("4" * 64))
        public_key = private_key.public_key().public_bytes(
            Encoding.Raw, PublicFormat.Raw
        ).hex()

        def sign(body: object) -> str:
            return private_key.sign(canonical_json(body).encode()).hex()

        receipt = _create_long_computation_receipt(
            wall_time_seconds=12.5,
            peak_unified_memory_bytes=1024,
            package_lock_sha256="a" * 64,
            model_snapshot_sha256="b" * 64,
            resumable_chain_head="c" * 64,
            runtime_artifact_sha256="d" * 64,
            execution_public_key=public_key,
            training_corpus_sha256="e" * 64,
            validation_corpus_sha256="f" * 64,
            measurement_method="resource.getrusage:RUSAGE_SELF:darwin-bytes",
            execution_signer=sign,
        )
        self.assertEqual(receipt.execution_public_key, public_key)
        with self.assertRaisesRegex(DataValidationError, "signature"):
            replace(receipt, peak_unified_memory_bytes=2048)


class ActivationCorpusTests(unittest.TestCase):
    def test_measured_sae_sweep_owns_runtime_and_memory_measurements(self) -> None:
        private_key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex("5" * 64))
        public_key = private_key.public_key().public_bytes(
            Encoding.Raw, PublicFormat.Raw
        ).hex()

        def sign(body: object) -> str:
            return private_key.sign(canonical_json(body).encode()).hex()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train"
            validation_path = root / "validation"
            write_activation_corpus(
                train_path,
                (
                    ActivationBatch(
                        ("train-0", "train-1", "train-2", "train-3"),
                        torch.tensor(
                            [
                                [1.0, 0.0, 0.5, -0.5],
                                [0.0, 1.0, -0.5, 0.5],
                                [1.0, 1.0, 0.25, -0.25],
                                [-1.0, 0.5, 1.0, 0.0],
                            ]
                        ),
                        (Outcome.CORRECT,) * 4,
                        ("tg-0", "tg-1", "tg-2", "tg-3"),
                    ),
                ),
                feature_schema=ActivationFeatureSchema.synthetic(
                    partition="sae-train", width=4
                ),
                shard_rows=2,
            )
            write_activation_corpus(
                validation_path,
                (
                    ActivationBatch(
                        ("validation-0", "validation-1"),
                        torch.tensor(
                            [[0.5, -0.5, 1.0, 0.0], [-0.5, 1.0, 0.0, 0.5]]
                        ),
                        (Outcome.CORRECT, Outcome.INCORRECT),
                        ("vg-0", "vg-1"),
                    ),
                ),
                feature_schema=ActivationFeatureSchema.synthetic(
                    partition="sae-validation", width=4
                ),
                shard_rows=2,
            )
            lock = root / "uv.lock"
            lock.write_text("version = 1\n", encoding="utf-8")
            measured = fit_e7_sae_sweep_measured(
                load_activation_corpus(train_path),
                load_activation_corpus(validation_path),
                tuple(
                    SAEConfig(
                        input_width=4,
                        latent_width=4,
                        sparsity=SAESparsity.TOP_K,
                        top_k=top_k,
                        epochs=1,
                        batch_size=2,
                        seed=seed,
                    )
                    for seed, top_k in ((11, 1), (12, 2), (13, 3))
                ),
                package_lock=lock,
                model_snapshot_sha256="a" * 64,
                runtime_artifact_sha256="b" * 64,
                execution_public_key=public_key,
                execution_signer=sign,
                checkpoint_directory=root / "sweep-work",
            )
            self.assertEqual(len(measured.results), 3)
            self.assertGreater(measured.receipt.wall_time_seconds, 0)
            self.assertGreater(measured.receipt.peak_unified_memory_bytes, 0)
            self.assertTrue(
                measured.receipt.measurement_method.startswith(
                    "resource.getrusage:RUSAGE_SELF:"
                )
            )
            resume_configs = tuple(value.config for value in measured.results)
            with patch(
                "mfh.methods.sparse.fit_sparse_autoencoder_corpus",
                side_effect=AssertionError("completed checkpoints must be resumed"),
            ):
                resumed = fit_e7_sae_sweep_measured(
                    load_activation_corpus(train_path),
                    load_activation_corpus(validation_path),
                    resume_configs,
                    package_lock=lock,
                    model_snapshot_sha256="a" * 64,
                    runtime_artifact_sha256="b" * 64,
                    execution_public_key=public_key,
                    execution_signer=sign,
                    checkpoint_directory=root / "sweep-work",
                )
            self.assertEqual(
                resumed.receipt.resumable_chain_head,
                measured.receipt.resumable_chain_head,
            )
            entry = sorted((root / "sweep-work" / "checkpoints").iterdir())[0]
            stale_stage = entry.parent / f".{entry.name}.stage-interrupted"
            stale_stage.mkdir()
            (stale_stage / "partial").write_text("incomplete\n", encoding="utf-8")
            with patch(
                "mfh.methods.sparse.fit_sparse_autoencoder_corpus",
                side_effect=AssertionError("stale stages must not retrain checkpoints"),
            ):
                fit_e7_sae_sweep_measured(
                    load_activation_corpus(train_path),
                    load_activation_corpus(validation_path),
                    resume_configs,
                    package_lock=lock,
                    model_snapshot_sha256="a" * 64,
                    runtime_artifact_sha256="b" * 64,
                    execution_public_key=public_key,
                    execution_signer=sign,
                    checkpoint_directory=root / "sweep-work",
                )
            self.assertFalse(stale_stage.exists())
            metadata_path = entry / "sae" / "metadata.json"
            original_metadata = metadata_path.read_text(encoding="utf-8")
            tampered = json.loads(original_metadata)
            tampered["metrics"]["reconstruction_mse"] += 1.0
            tampered.pop("metadata_digest")
            tampered["metadata_digest"] = stable_hash(tampered)
            metadata_path.write_text(
                json.dumps(tampered, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FrozenArtifactError, "signature"):
                fit_e7_sae_sweep_measured(
                    load_activation_corpus(train_path),
                    load_activation_corpus(validation_path),
                    resume_configs,
                    package_lock=lock,
                    model_snapshot_sha256="a" * 64,
                    runtime_artifact_sha256="b" * 64,
                    execution_public_key=public_key,
                    execution_signer=sign,
                    checkpoint_directory=root / "sweep-work",
                )
            metadata_path.write_text(original_metadata, encoding="utf-8")
            external = root / "external-checkpoint"
            entry.rename(external)
            entry.symlink_to(external, target_is_directory=True)
            with self.assertRaisesRegex(FrozenArtifactError, "linked"):
                fit_e7_sae_sweep_measured(
                    load_activation_corpus(train_path),
                    load_activation_corpus(validation_path),
                    resume_configs,
                    package_lock=lock,
                    model_snapshot_sha256="a" * 64,
                    runtime_artifact_sha256="b" * 64,
                    execution_public_key=public_key,
                    execution_signer=sign,
                    checkpoint_directory=root / "sweep-work",
                )
            entry.unlink()
            external.rename(entry)

    def test_corpus_is_sharded_memory_mapped_and_checksum_validated(self) -> None:
        batches = (
            ActivationBatch(
                ("q0", "q1", "q2"),
                torch.arange(12, dtype=torch.float32).reshape(3, 4),
                (Outcome.CORRECT, Outcome.INCORRECT, Outcome.ABSTENTION),
                ("g0", "g0", "g2"),
            ),
            ActivationBatch(
                ("q3", "q4"),
                torch.arange(8, dtype=torch.float32).reshape(2, 4),
                (Outcome.CORRECT, Outcome.INCORRECT),
                ("g3", "g4"),
            ),
        )
        fingerprint = "a" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus"
            fingerprint = write_activation_corpus(
                path,
                batches,
                feature_schema=ActivationFeatureSchema.synthetic(partition="sae-train", width=4),
                shard_rows=2,
            )
            corpus = load_activation_corpus(path, expected_data_fingerprint=fingerprint)
            shards = tuple(corpus.iter_shards())
            self.assertEqual([len(shard.question_ids) for shard in shards], [2, 2, 1])
            self.assertEqual(corpus.total_rows, 5)
            self.assertTrue(all(isinstance(shard.activations, np.memmap) for shard in shards))
            self.assertTrue(all(shard.activations.dtype == np.float16 for shard in shards))
            activation_path = path / "activations-00000.npy"
            activation_path.write_bytes(activation_path.read_bytes() + b"tamper")
            with self.assertRaises(FrozenArtifactError):
                load_activation_corpus(path)

    def test_sae_training_streams_disjoint_train_and_validation_shards(self) -> None:
        generator = torch.Generator().manual_seed(5)
        train_values = torch.randn(24, 4, generator=generator)
        validation_values = torch.randn(8, 4, generator=generator)
        train_batch = ActivationBatch(
            tuple(f"train-{index}" for index in range(24)),
            train_values,
            tuple(Outcome.CORRECT if index % 2 else Outcome.INCORRECT for index in range(24)),
            tuple(f"train-group-{index // 2}" for index in range(24)),
        )
        validation_batch = ActivationBatch(
            tuple(f"validation-{index}" for index in range(8)),
            validation_values,
            tuple(Outcome.CORRECT if index % 2 else Outcome.INCORRECT for index in range(8)),
            tuple(f"validation-group-{index // 2}" for index in range(8)),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_activation_corpus(
                root / "train",
                (train_batch,),
                feature_schema=ActivationFeatureSchema.synthetic(partition="sae-train", width=4),
                shard_rows=6,
            )
            write_activation_corpus(
                root / "validation",
                (validation_batch,),
                feature_schema=ActivationFeatureSchema.synthetic(
                    partition="sae-validation", width=4
                ),
                shard_rows=4,
            )
            training = load_activation_corpus(root / "train")
            validation = load_activation_corpus(root / "validation")
            result = fit_sparse_autoencoder_corpus(
                training,
                validation,
                SAEConfig(
                    input_width=4,
                    expansion_factor=2,
                    top_k=2,
                    epochs=4,
                    batch_size=3,
                    learning_rate=0.01,
                ),
            )
            self.assertEqual(result.training_fingerprint, training.data_fingerprint)
            self.assertEqual(result.validation_fingerprint, validation.data_fingerprint)
            self.assertTrue(np.isfinite(result.metrics.reconstruction_mse))

    def test_signed_corpus_rejects_rehashed_forged_capture_receipt(self) -> None:
        private_key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex("3" * 64))
        public_key = private_key.public_key().public_bytes(
            Encoding.Raw, PublicFormat.Raw
        ).hex()
        schema = ActivationFeatureSchema.synthetic(partition="sae-train", width=4)
        def sign(body: object) -> str:
            return private_key.sign(canonical_json(body).encode()).hex()

        values = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        outcomes = (Outcome.CORRECT, Outcome.INCORRECT)
        receipts = []
        for row, (question_id, group_id, outcome) in enumerate(
            zip(("q0", "q1"), ("g0", "g1"), outcomes, strict=True)
        ):
            activation = np.ascontiguousarray(values[row].numpy().astype(np.float16))
            body = activation_capture_execution_receipt_body(
                question_id=question_id,
                group_id=group_id,
                outcome=outcome,
                rendered_prompt_sha256="f" * 64,
                activation_sha256=hashlib.sha256(activation.tobytes()).hexdigest(),
                feature_schema=schema,
                runtime_artifact_sha256="a" * 64,
                execution_public_key=public_key,
                source_question_bundle_sha256=schema.split_manifest_digest,
                dtype="float16",
            )
            receipts.append({"body": body, "signature": sign(body)})
        batch = ActivationBatch(
            ("q0", "q1"),
            values,
            outcomes,
            ("g0", "g1"),
            tuple(receipts),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "signed-corpus"
            write_activation_corpus(
                path,
                (batch,),
                feature_schema=schema,
                shard_rows=2,
                runtime_artifact_sha256="a" * 64,
                execution_public_key=public_key,
                source_question_bundle_sha256=schema.split_manifest_digest,
                capture_signer=sign,
            )
            corpus = load_activation_corpus(path)
            self.assertEqual(corpus.schema_version, 2)
            manifest_path = path / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["shards"][0]["execution_signature"] = "0" * 128
            fingerprint_body = {
                key: value
                for key, value in manifest.items()
                if key not in {"schema_version", "data_fingerprint", "manifest_digest"}
            }
            manifest["data_fingerprint"] = stable_hash(fingerprint_body)
            manifest_body = {
                key: value for key, value in manifest.items() if key != "manifest_digest"
            }
            manifest["manifest_digest"] = stable_hash(manifest_body)
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FrozenArtifactError, "signature"):
                load_activation_corpus(path)


if __name__ == "__main__":
    unittest.main()

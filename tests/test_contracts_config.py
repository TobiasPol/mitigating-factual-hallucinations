from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mfh.config import (
    load_benchmark_spec,
    load_inference_protocol,
    load_model_spec,
    load_prompt_specs,
    load_semantic_contamination_protocol,
)
from mfh.contracts import (
    ActivationSite,
    GenerationRecord,
    Outcome,
    Runtime,
    TokenScope,
    TransformersModelClass,
)
from mfh.errors import ConfigurationError

ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def test_repository_configs_are_strict_and_pinned(self) -> None:
        model = load_model_spec(ROOT / "configs/models/qwen3.6-27b-mlx-4bit.yaml")
        gguf = load_model_spec(ROOT / "configs/models/ternary-bonsai-4b-gguf.yaml")
        benchmark = load_benchmark_spec(ROOT / "configs/benchmarks/simpleqa-verified.yaml")
        prompts = load_prompt_specs(ROOT / "configs/prompts/primary.yaml")
        protocol = load_inference_protocol(ROOT / "configs/experiments/core.yaml")
        contamination = load_semantic_contamination_protocol(
            ROOT / "configs/contamination/triviaqa-ood.yaml"
        )
        self.assertEqual(model.num_layers, 64)
        self.assertIs(model.runtime, Runtime.MLX)
        self.assertIs(
            model.transformers_model_class,
            TransformersModelClass.CAUSAL_LM,
        )
        self.assertEqual(len(model.revision), 40)
        self.assertEqual(gguf.artifact, "Ternary-Bonsai-4B-Q2_0.gguf")
        self.assertEqual(len(gguf.artifact_sha256 or ""), 64)
        self.assertEqual(benchmark.split, "eval")
        self.assertEqual(
            {item.prompt_id for item in prompts},
            {
                "P0-neutral",
                "P1-direct",
                "P2-calibrated-abstention",
                "P3-forced-answer",
                "P-AA-official",
            },
        )
        self.assertFalse(protocol.do_sample)
        self.assertEqual(contamination.embedding_dimension, 384)
        self.assertEqual(len(contamination.model_revision), 40)

    def test_mutable_model_revision_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.yaml"
            path.write_text(
                """model:
  name: unsafe
  repository: example/model
  revision: main
  runtime: transformers
  quantization: none
  num_layers: 1
""",
                encoding="utf-8",
            )
            with self.assertRaises(ConfigurationError):
                load_model_spec(path)


class GenerationRecordTests(unittest.TestCase):
    def test_json_round_trip_preserves_enums_and_scores(self) -> None:
        record = GenerationRecord(
            question_id="q1",
            benchmark="toy",
            model_repository="toy/model",
            model_revision="synthetic-v1",
            runtime=Runtime.SYNTHETIC,
            quantization="none",
            system_prompt_id="P0",
            rendered_prompt_hash="abc",
            steering_method="M1-P",
            layer=2,
            token_scope=TokenScope.FIRST_FOUR,
            alpha=0.25,
            sparsity=None,
            controller_scores={"C": 0.7, "I": 0.2, "A": 0.1},
            raw_output="Paris",
            normalized_answer="paris",
            outcome=Outcome.CORRECT,
            generation_latency_seconds=0.01,
            input_tokens=8,
            output_tokens=1,
            condition_id="condition-1",
            site=ActivationSite.POST_MLP,
            seed=17,
        )
        restored = GenerationRecord.from_dict(record.to_dict())
        self.assertEqual(restored, record)

        invalid = record.to_dict()
        invalid["schema_version"] = 2
        with self.assertRaises(ValueError):
            GenerationRecord.from_dict(invalid)
        invalid.pop("schema_version")
        with self.assertRaises(ValueError):
            GenerationRecord.from_dict(invalid)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest
import torch
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from mfh.contracts import (
    ActivationSite,
    Outcome,
    PromptSpec,
    Question,
    Runtime,
    TokenScope,
)
from mfh.errors import FrozenArtifactError
from mfh.experiments.e5_capture import VerifiedE5FitCapture
from mfh.experiments.e5_layer_labels import (
    load_e5_layer_label_data,
    prepare_e5_layer_label_capture,
    run_e5_layer_label_capture,
    verify_e5_layer_label_capture,
)
from mfh.experiments.e5_types import E5FitRecipe
from mfh.inference.mlx_runtime import MlxGenerationOutput, MlxRenderedPrompt
from mfh.methods.features import (
    ActivationFeatureSchema,
    ActivationKind,
    FeatureComposition,
)
from mfh.methods.probes import ProbeDataset
from mfh.provenance import sha256_file, sha256_path, stable_hash

_PRIVATE = "31" * 32
_PUBLIC = (
    Ed25519PrivateKey.from_private_bytes(bytes.fromhex(_PRIVATE))
    .public_key()
    .public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    .hex()
)
_PROMPT = PromptSpec(
    "P0-neutral",
    "You are a helpful assistant. Answer the user's factual question.",
)


def _identity() -> dict[str, Any]:
    return {
        "model_repository": "mlx-community/test",
        "model_revision": "c" * 40,
        "model_quantization": "4bit",
        "num_layers": 3,
    }


def _questions() -> tuple[Question, ...]:
    return (
        Question("q-0", "triviaqa", "Question 0?", ("answer-0",), "T-controller-train"),
        Question("q-1", "triviaqa", "Question 1?", ("answer-1",), "T-controller-train"),
    )


def _datasets() -> Mapping[FeatureComposition, ProbeDataset]:
    result: dict[FeatureComposition, ProbeDataset] = {}
    widths = {
        FeatureComposition.SINGLE_LAYER: 4,
        FeatureComposition.CONCATENATED_LAYERS: 12,
        FeatureComposition.LAYER_DIFFERENCES: 8,
    }
    layers = {
        FeatureComposition.SINGLE_LAYER: (31,),
        FeatureComposition.CONCATENATED_LAYERS: (16, 31, 32),
        FeatureComposition.LAYER_DIFFERENCES: (16, 31, 32),
    }
    for composition in FeatureComposition:
        schema = ActivationFeatureSchema(
            benchmark="triviaqa",
            partition="T-controller-train",
            split_manifest_digest="1" * 64,
            model_repository="mlx-community/test",
            model_revision="c" * 40,
            runtime=Runtime.MLX,
            quantization="4bit",
            prompt_id="P0-neutral",
            prompt_sha256=hashlib.sha256(_PROMPT.text.encode()).hexdigest(),
            activation_kind=ActivationKind.FINAL_PROMPT,
            layers=layers[composition],
            sites=(ActivationSite.POST_MLP,),
            composition=composition,
            width=widths[composition],
            token_scope=TokenScope.FINAL_PROMPT,
        )
        result[composition] = ProbeDataset(
            question_ids=("q-0", "q-1"),
            features=torch.arange(2 * widths[composition], dtype=torch.float32).reshape(
                2, widths[composition]
            ),
            outcomes=(Outcome.CORRECT, Outcome.INCORRECT),
            group_ids=("g-0", "g-1"),
            feature_schema=schema,
        )
    return MappingProxyType(result)


def _recipe() -> E5FitRecipe:
    return E5FitRecipe(
        fixed_best_layer=31,
        two_layer_candidates=(31, 32),
        three_layer_candidates=(16, 31, 32),
        intervention_site=ActivationSite.POST_MLP,
        alpha_max=0.5,
    )


def _vectors(tmp_path: Path) -> Path:
    root = tmp_path / "vectors"
    root.mkdir()
    directions = np.zeros((2, 2, 1, 3, 4), dtype=np.float32)
    directions[..., 0] = 1.0
    rms = np.ones((2, 2, 1, 3), dtype=np.float64)
    counts = np.ones((2, 2, 1, 3), dtype=np.int64)
    np.savez_compressed(
        root / "vectors.npz",
        directions=directions,
        reference_rms=rms,
        correct_counts=counts,
        incorrect_counts=counts,
    )
    body = {
        "schema_version": 1,
        "phase": "E3-construction",
        "scientific_eligible": True,
        "vectors_sha256": sha256_file(root / "vectors.npz"),
        "prompt_axis": ["P0-neutral", "P2-calibrated-abstention"],
        "extraction_axis": ["M1-R", "M1-P"],
        "site_axis": ["post_mlp"],
        "layer_axis": [16, 31, 32],
    }
    (root / "metadata.json").write_text(
        json.dumps({**body, "metadata_digest": stable_hash(body)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root


def _capture(tmp_path: Path, *, vectors_sha256: str) -> VerifiedE5FitCapture:
    root = tmp_path / "fit-capture"
    root.mkdir()
    (root / "frozen").write_text("capture", encoding="utf-8")
    return VerifiedE5FitCapture(
        directory=root.resolve(),
        plan={
            "plan_identity": "a" * 64,
            "runtime_identity": _identity(),
            "recipe": _recipe().to_dict(),
            "execution_public_key": _PUBLIC,
            "e3_static_vectors_sha256": vectors_sha256,
        },
        pairs_completed=4,
        shard_count=1,
        chain_head="b" * 64,
        complete=True,
        scientific_eligible=False,
        maximum_peak_memory_bytes=1024,
    )


@dataclass
class _State:
    direction: np.ndarray[Any, Any]
    raw_alpha: float
    captured: np.ndarray[Any, Any] | None = None
    intervened: np.ndarray[Any, Any] | None = None
    applications: int = 0


class _Runtime:
    def runtime_identity(self) -> Mapping[str, Any]:
        return _identity()

    def render_prompt(
        self,
        prompt: PromptSpec,
        question: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> MlxRenderedPrompt:
        del metadata
        text = f"{prompt.text}|{question}"
        tokens = (10, int(question.split()[1].rstrip("?")) + 20)
        return MlxRenderedPrompt(
            text=text,
            sha256=hashlib.sha256(text.encode()).hexdigest(),
            token_ids=tokens,
            token_ids_sha256=hashlib.sha256(
                ",".join(str(value) for value in tokens).encode()
            ).hexdigest(),
            messages=(),
        )

    def standardized_intervention_state(
        self,
        direction: np.ndarray[Any, Any],
        *,
        standardized_alpha: float,
        reference_rms: float,
        token_scope: TokenScope,
        decay: float = 0.0,
    ) -> _State:
        del token_scope, decay
        return _State(direction.copy(), standardized_alpha * reference_rms)

    def generate_with_interventions(
        self,
        rendered: MlxRenderedPrompt,
        *,
        max_new_tokens: int,
        intervention_states: Mapping[tuple[int, ActivationSite], Any],
    ) -> MlxGenerationOutput:
        del max_new_tokens
        (layer, _site), raw_state = next(iter(intervention_states.items()))
        assert isinstance(raw_state, _State)
        state = raw_state
        state.captured = np.zeros((1, 1, 4), dtype=np.float32)
        state.intervened = state.captured.copy()
        state.intervened[0, 0] += state.direction * state.raw_alpha
        state.applications = 1
        question_index = int(rendered.text.split("Question ")[1].rstrip("?"))
        outputs = {
            (0, 16): "wrong",
            (0, 31): "answer-0",
            (0, 32): "I don't know.",
            (1, 16): "I don't know.",
            (1, 31): "wrong",
            (1, 32): "answer-1",
        }
        output = outputs[(question_index, layer)]
        token_ids = (100 + question_index, 200 + layer)
        return MlxGenerationOutput(
            rendered_prompt=rendered,
            token_ids=token_ids,
            text=output,
            input_tokens=len(rendered.token_ids),
            output_tokens=len(token_ids),
            latency_seconds=0.1,
            stop_type="short_answer",
            stopping_token_id=token_ids[-1],
            prompt_tokens_per_second=10.0,
            generation_tokens_per_second=5.0,
            peak_memory_bytes=2048,
            active_memory_bytes=1024,
            cache_memory_bytes=512,
        )


def _prepare(tmp_path: Path) -> tuple[Path, VerifiedE5FitCapture]:
    vectors = _vectors(tmp_path)
    capture = _capture(tmp_path, vectors_sha256=sha256_path(vectors))
    work = tmp_path / "labels"
    prepare_e5_layer_label_capture(
        work,
        questions=_questions(),
        prompt=_PROMPT,
        controller_datasets=_datasets(),
        fit_capture=capture,
        fit_capture_artifact_sha256=sha256_path(capture.directory),
        e3_static_vectors_directory=vectors,
        recipe=_recipe(),
        runtime_identity=_identity(),
        execution_public_key=_PUBLIC,
        shard_rows=3,
        max_peak_memory_bytes=4096,
    )
    return work, capture


def test_e5_layer_labels_resume_and_reduce_counterfactual_outcomes(tmp_path: Path) -> None:
    work, capture = _prepare(tmp_path)
    partial = run_e5_layer_label_capture(
        work,
        questions=_questions(),
        prompt=_PROMPT,
        controller_datasets=_datasets(),
        fit_capture=capture,
        fit_capture_artifact_sha256=sha256_path(capture.directory),
        runtime=_Runtime(),
        private_key_hex=_PRIVATE,
        request_budget=3,
    )
    assert partial.records_completed == 3
    assert not partial.complete
    complete = run_e5_layer_label_capture(
        work,
        questions=_questions(),
        prompt=_PROMPT,
        controller_datasets=_datasets(),
        fit_capture=capture,
        fit_capture_artifact_sha256=sha256_path(capture.directory),
        runtime=_Runtime(),
        private_key_hex=_PRIVATE,
    )
    assert complete.complete
    data = load_e5_layer_label_data(
        work,
        questions=_questions(),
        prompt=_PROMPT,
        controller_datasets=_datasets(),
        fit_capture=capture,
        fit_capture_artifact_sha256=sha256_path(capture.directory),
        expected_execution_public_key=_PUBLIC,
    )
    assert data.best_layers_two == (31, 32)
    assert data.best_layers_three == (31, 32)
    assert data.question_ids == ("q-0", "q-1")


def test_e5_layer_labels_require_external_key_and_exact_vector_artifact(
    tmp_path: Path,
) -> None:
    work, capture = _prepare(tmp_path)
    with pytest.raises(FrozenArtifactError, match="external trust root"):
        verify_e5_layer_label_capture(
            work,
            questions=_questions(),
            prompt=_PROMPT,
            controller_datasets=_datasets(),
            fit_capture=capture,
            fit_capture_artifact_sha256=sha256_path(capture.directory),
            expected_execution_public_key="42" * 32,
        )
    vector_path = Path(json.loads((work / "plan.json").read_text())["e3_static_vectors_path"])
    metadata = vector_path / "metadata.json"
    metadata.write_text(metadata.read_text() + " ", encoding="utf-8")
    with pytest.raises(FrozenArtifactError, match="source changed"):
        verify_e5_layer_label_capture(
            work,
            questions=_questions(),
            prompt=_PROMPT,
            controller_datasets=_datasets(),
            fit_capture=capture,
            fit_capture_artifact_sha256=sha256_path(capture.directory),
            expected_execution_public_key=_PUBLIC,
        )

"""Small authentic E3/M2 artifacts used by the E4 boundary tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from mfh.contracts import (
    ActivationSite,
    Outcome,
    PromptSpec,
    Question,
    Runtime,
    TokenScope,
)
from mfh.experiments import e4_caa_vllm
from mfh.experiments.e3_construction import VerifiedE3ConstructionSnapshot
from mfh.experiments.e4_caa_vllm import (
    finalize_m2_caa_artifact,
    prepare_m2_caa_work,
    run_m2_caa_work,
)
from mfh.inference.vllm_runtime import VllmRenderedPrompt
from mfh.methods.features import (
    ActivationFeatureSchema,
    ActivationKind,
    FeatureComposition,
)
from mfh.methods.probes import (
    CalibratedProbe,
    ProbeKind,
    ProbeState,
    ProbeTask,
    TemperatureCalibrator,
    save_calibrated_probe,
)
from mfh.provenance import sha256_file, sha256_path, stable_hash

_PLAN_IDENTITY = "a" * 64
_GENERATION_HEAD = "b" * 64


def active_qwen_runtime_identity() -> dict[str, Any]:
    """Small deterministic stand-in carrying the exact frozen Qwen runtime facts."""

    return {
        "backend": "vllm",
        "vllm": "0.24.0",
        "transformers": "5.2.0",
        "torch": "2.11.0",
        "python": "3.11.14",
        "architecture": "x86_64",
        "os": "Linux test",
        "nvidia_driver": "570.00",
        "gpu_name": "NVIDIA A100-SXM4-40GB",
        "gpu_total_memory_bytes": 40_000_000_000,
        "cuda_capability": "8.0",
        "cuda_runtime": "12.9",
        "tensor_parallel_size": 1,
        "quantization_loader": "modelopt_mixed",
        "quantization_config_class": (
            "vllm.model_executor.layers.quantization.modelopt."
            "ModelOptMixedPrecisionConfig"
        ),
        "quantization_execution": "marlin-w4a16-fp8-weight-only-on-sm80",
        "model_class": (
            "vllm.model_executor.models.qwen3_5."
            "Qwen3_5ForConditionalGeneration"
        ),
        "tokenizer_class": "test.Tokenizer",
        "num_layers": 64,
        "hidden_size": 5_120,
        "seed": 17,
        "model_repository": "nvidia/Qwen3.6-27B-NVFP4",
        "model_revision": "0893e1606ff3d5f97a441f405d5fc541a6bdf404",
        "model_quantization": "modelopt-mixed-nvfp4-fp8",
        "model_num_layers": 64,
        "snapshot_sha256": "c" * 64,
        "research_provenance": {
            "model_repository": "nvidia/Qwen3.6-27B-NVFP4",
            "model_revision": "0893e1606ff3d5f97a441f405d5fc541a6bdf404",
            "quantization": "modelopt-mixed-nvfp4-fp8",
            "verified_snapshot_digest": "d" * 64,
            "runtime_preflight_receipt_digest": "e" * 64,
            "runtime_policy_digest": "f" * 64,
            "research_toolchain_digest": "1" * 64,
        },
        "research_toolchain": {
            "vllm": "0.24.0",
            "torch": "2.11.0",
            "transformers": "5.2.0",
            "numpy": "2.4.3",
            "nvidia_driver": "570.00",
        },
    }


class _Generation:
    def __init__(self, question: Question, rendered: VllmRenderedPrompt) -> None:
        self.sequence = 0
        self.question_id = question.question_id
        self.prompt_id = "P0-neutral"
        self.rendered_prompt_sha256 = rendered.sha256
        self.prompt_token_ids_sha256 = rendered.token_ids_sha256
        self.schedule_row_sha256 = stable_hash((0, question.question_id))
        self.outcome = Outcome.INCORRECT
        self.evidence = MappingProxyType({"raw_output": "wrong"})

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "question_id": self.question_id,
            "prompt_id": self.prompt_id,
            "rendered_prompt_sha256": self.rendered_prompt_sha256,
            "prompt_token_ids_sha256": self.prompt_token_ids_sha256,
            "schedule_row_sha256": self.schedule_row_sha256,
            "outcome": self.outcome.value,
            "evidence": dict(self.evidence),
        }


class _Runtime:
    def __init__(self, identity: dict[str, Any], rendered: VllmRenderedPrompt) -> None:
        self.identity = identity
        self.rendered = rendered

    def runtime_identity(self) -> dict[str, Any]:
        return dict(self.identity)

    def render_prompt(
        self,
        prompt: PromptSpec,
        question: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> VllmRenderedPrompt:
        del prompt, question, metadata
        return self.rendered

    def teacher_forced_cube(
        self,
        rendered: VllmRenderedPrompt,
        response: str,
        *,
        layers: tuple[int, ...],
        sites: tuple[ActivationSite, ...],
    ) -> SimpleNamespace:
        del rendered
        assert sites == (ActivationSite.BLOCK_OUTPUT,)
        base = 2.0 if response == "gold" else -1.0
        vector = np.full(5_120, -base, dtype=np.float32)
        vector[0] = base
        vector[1] = base / 2.0
        activations = {
            ActivationSite.BLOCK_OUTPUT: {
                layer: np.asarray([vector + layer / 100.0], dtype=np.float32)
                for layer in layers
            }
        }
        token_ids = (11,) if response == "gold" else (12,)
        return SimpleNamespace(
            response_text_sha256=hashlib.sha256(response.encode()).hexdigest(),
            response_token_ids=token_ids,
            response_token_ids_sha256=hashlib.sha256(
                ",".join(str(value) for value in token_ids).encode("ascii")
            ).hexdigest(),
            activations=activations,
            peak_memory_bytes=1024,
        )


def build_e3_m1_bundle(
    root: Path,
    *,
    direction: tuple[float, ...] | None = None,
    reference_rms_value: float = 2.0,
    layers: tuple[int, ...] = (31, 63),
) -> Path:
    """Write a minimal portable E3 vector bundle with real normalized geometry."""

    source = root / "e3-static-vectors"
    source.mkdir()
    vector = np.zeros(5_120, dtype=np.float32) if direction is None else np.asarray(
        direction, dtype=np.float32
    )
    if direction is None:
        vector[0] = 1.0
    vector = vector / np.linalg.norm(vector)
    directions = np.broadcast_to(
        vector, (2, 2, 1, len(layers), len(vector))
    ).copy()
    reference_rms = np.full(
        directions.shape[:-1], reference_rms_value, dtype=np.float64
    )
    counts = np.ones(directions.shape[:-1], dtype=np.int64)
    with (source / "vectors.npz").open("wb") as handle:
        np.savez_compressed(
            handle,
            directions=directions,
            reference_rms=reference_rms,
            correct_counts=counts,
            incorrect_counts=counts,
        )
    body = {
        "schema_version": 1,
        "phase": "E3-construction",
        "scientific_eligible": True,
        "prompt_axis": ["P0-neutral", "P2-calibrated-abstention"],
        "extraction_axis": ["M1-R", "M1-P"],
        "site_axis": [ActivationSite.POST_MLP.value],
        "layer_axis": list(layers),
        "plan_identity": _PLAN_IDENTITY,
        "generation_chain_head": _GENERATION_HEAD,
        "vectors_sha256": sha256_file(source / "vectors.npz"),
    }
    (source / "metadata.json").write_text(
        json.dumps({**body, "metadata_digest": stable_hash(body)}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return source


def build_e2_probe_bundle(root: Path) -> Path:
    """Write the minimum authentic selected E2 C/I/A probe surface."""

    source = root / "e2-probe-bundle"
    probe_path = (
        source
        / "probes"
        / ProbeTask.CORRECT_INCORRECT_ABSTENTION.value
        / ProbeKind.LOGISTIC.value
        / "temperature"
    )
    probe_path.parent.mkdir(parents=True)
    schema = ActivationFeatureSchema(
        benchmark="triviaqa",
        partition="T-controller-train",
        split_manifest_digest="2" * 64,
        model_repository="nvidia/Qwen3.6-27B-NVFP4",
        model_revision="0893e1606ff3d5f97a441f405d5fc541a6bdf404",
        runtime=Runtime.VLLM,
        quantization="modelopt-mixed-nvfp4-fp8",
        prompt_id="P0-neutral",
        prompt_sha256="3" * 64,
        activation_kind=ActivationKind.FINAL_PROMPT,
        layers=(31,),
        sites=(ActivationSite.POST_MLP,),
        composition=FeatureComposition.SINGLE_LAYER,
        width=5_120,
        token_scope=TokenScope.FINAL_PROMPT,
    )
    calibration = replace(schema, partition="T-controller-calibration")
    probe = CalibratedProbe(
        task=ProbeTask.CORRECT_INCORRECT_ABSTENTION,
        state=ProbeState(
            kind=ProbeKind.LOGISTIC,
            labels=("C", "I", "A"),
            feature_mean=torch.zeros(5_120),
            feature_scale=torch.ones(5_120),
            parameters={
                "weight": torch.zeros(3, 5_120),
                "bias": torch.tensor([0.0, 1.0, -1.0]),
            },
        ),
        calibrator=TemperatureCalibrator(1.0),
        training_fingerprint="4" * 64,
        calibration_fingerprint="5" * 64,
        training_schema=schema,
        calibration_schema=calibration,
    )
    save_calibrated_probe(probe_path, probe)
    (source / "screening-probes").mkdir()
    (source / "plan.json").write_text("{}\n", encoding="utf-8")
    (source / "screening.json").write_text("[]\n", encoding="utf-8")
    relative = probe_path.relative_to(source / "probes").as_posix()
    probe_sha = sha256_path(probe_path)
    results = {
        "schema_version": 1,
        "selected_views": {},
        "final_probes": [
            {
                "task": ProbeTask.CORRECT_INCORRECT_ABSTENTION.value,
                "artifact": relative,
                "artifact_sha256": probe_sha,
            }
        ],
        "gate": {"passed": True, "selected_artifact_sha256": probe_sha},
    }
    (source / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    body = {
        "schema_version": 1,
        "phase": "E2",
        "scientific_eligible": True,
        "files": {"results.json": sha256_file(source / "results.json")},
        "probes_sha256": sha256_path(source / "probes"),
    }
    (source / "manifest.json").write_text(
        json.dumps({**body, "manifest_digest": stable_hash(body)}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return source


def build_m2_caa_bundle(root: Path) -> Path:
    """Build M2 through the real resumable constructor using a tiny fake VLLM runtime."""

    question = Question(
        question_id="steer-0",
        benchmark="triviaqa",
        text="Question?",
        aliases=("gold",),
        split="T-steer",
    )
    text = "system:neutral\nuser:Question?\nassistant:"
    rendered = VllmRenderedPrompt(
        text=text,
        sha256=hashlib.sha256(text.encode()).hexdigest(),
        token_ids=(1, 2, 3),
        token_ids_sha256=hashlib.sha256(b"1,2,3").hexdigest(),
        messages=(),
    )
    identity = active_qwen_runtime_identity()
    construction = root / "e3-construction"
    construction.mkdir()
    (construction / "source.txt").write_text("frozen E3", encoding="utf-8")
    snapshot = VerifiedE3ConstructionSnapshot(
        directory=construction,
        plan=MappingProxyType(
            {
                "plan_identity": _PLAN_IDENTITY,
                "runtime_identity": identity,
                "hidden_width": 5_120,
            }
        ),
        schedule=(SimpleNamespace(),),  # type: ignore[arg-type]
        generations=(_Generation(question, rendered),),  # type: ignore[arg-type]
        generation_chain_head=_GENERATION_HEAD,
        scientific_eligible=True,
    )
    patcher = pytest.MonkeyPatch()
    patcher.setattr(
        e4_caa_vllm,
        "load_verified_e3_construction_snapshot",
        lambda *_args, **_kwargs: snapshot,
    )
    try:
        prompts = {
            "P0-neutral": PromptSpec(
                "P0-neutral", "You are a helpful assistant. Answer the factual question."
            )
        }
        work = root / "m2-work"
        prepare_m2_caa_work(
            work,
            construction_directory=construction,
            questions=(question,),
            prompts=prompts,
        )
        run_m2_caa_work(
            work,
            construction_directory=construction,
            questions=(question,),
            prompts=prompts,
            runtime=_Runtime(identity, rendered),
            request_budget=1,
        )
        output = root / "m2-artifact"
        finalize_m2_caa_artifact(
            output,
            work_directory=work,
            construction_directory=construction,
            questions=(question,),
            prompts=prompts,
        )
    finally:
        patcher.undo()
    return output

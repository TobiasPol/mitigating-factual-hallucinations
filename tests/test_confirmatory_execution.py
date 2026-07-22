from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

from mfh.config import load_model_spec, load_prompt_specs
from mfh.contracts import (
    ActivationSite,
    GenerationRecord,
    Outcome,
    Question,
    Runtime,
    TokenScope,
)
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.evaluation.official import load_official_grader_spec
from mfh.evaluation.openrouter import OpenRouterError, OpenRouterTransport
from mfh.evaluation.side_effects import write_side_effect_scorer_spec
from mfh.experiments import confirmatory_graders, e9_native
from mfh.experiments.confirmatory_components import (
    write_confirmatory_fixed_component,
)
from mfh.experiments.confirmatory_graders import write_confirmatory_grader_bundle
from mfh.experiments.e6_likelihood import E6RuntimeAttestor
from mfh.experiments.e9_native import NativeE9VllmBackend
from mfh.experiments.protocol import ExperimentPhase, load_study_protocol
from mfh.experiments.runner import (
    EvaluationCondition,
    _sign_confirmatory_execution_receipt_for_test,
    validate_confirmatory_execution_receipt,
)
from mfh.inference.vllm_research import VllmResearchRuntime
from mfh.inference.vllm_runtime import VllmGenerationOutput, VllmRenderedPrompt
from mfh.provenance import sha256_path, stable_hash
from tests.e4_test_artifacts import build_e3_m1_bundle

ROOT = Path(__file__).parents[1]
PROMPT_SHA = "5b08080e81d4032d853d3a30fda35e7670da6a81cc1ccf17f2b8fc5001bba684"


def _key_material() -> tuple[str, str]:
    private = Ed25519PrivateKey.generate()
    private_hex = private.private_bytes(
        Encoding.Raw,
        PrivateFormat.Raw,
        NoEncryption(),
    ).hex()
    public_hex = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    return private_hex, public_hex


def _condition(method: str) -> EvaluationCondition:
    model = load_model_spec(ROOT / "configs/models/qwen3.6-27b-nvfp4.yaml")
    study = load_study_protocol(ROOT / "configs/experiments/phases.yaml")
    fixed = method != "M0"
    return EvaluationCondition(
        phase=ExperimentPhase.E9,
        benchmark="triviaqa",
        partition="T-test",
        model_name=model.name,
        model_repository=model.repository,
        model_revision=model.revision,
        runtime=Runtime.VLLM,
        quantization=model.quantization,
        model_num_layers=model.num_layers,
        system_prompt_id="P0-neutral",
        prompt_template_sha256=PROMPT_SHA,
        steering_method=method,
        method_artifact_sha256="4" * 64 if fixed else None,
        layer=21 if fixed else None,
        site=ActivationSite.POST_MLP if fixed else None,
        token_scope=TokenScope.FIRST_FOUR if fixed else None,
        alpha=1.0 if fixed else 0.0,
        sparsity=0.1 if method == "M4" else None,
        seed=17,
        study_protocol_digest=study.digest,
    )


def _record(condition: EvaluationCondition) -> GenerationRecord:
    metadata: dict[str, object] = {
        "phase": "E9",
        "partition": "T-test",
        "prompt_template_sha256": PROMPT_SHA,
        "study_protocol_digest": condition.study_protocol_digest,
        "official_exact_match": 1.0,
        "official_token_f1": 1.0,
        "official_score_output_sha256": stable_hash("Paris"),
        "runtime_session_identity_sha256": "9" * 64,
        "decoding_max_new_tokens": 48,
        "generation_runtime_metrics": {
            "schema_version": 1,
            "gpu_total_memory_bytes": 16 * 1024**3,
            "peak_memory_bytes": 1024,
            "generation_peak_memory_bytes": 1024,
            "auxiliary_peak_memory_bytes": 0,
            "active_memory_bytes": 512,
            "cache_memory_bytes": 256,
            "prompt_tokens_per_second": 100.0,
            "generation_tokens_per_second": 100.0,
            "generation_wall_time_seconds": 1.0,
            "stop_type": "eos",
            "stopping_token_id": 2,
        },
    }
    if condition.method_artifact_sha256 is not None:
        metadata["method_artifact_sha256"] = condition.method_artifact_sha256
        trace = {
            "schema_version": 1,
            "method_artifact_sha256": condition.method_artifact_sha256,
            "layer": condition.layer,
            "site": condition.site.value if condition.site is not None else None,
            "token_scope": (
                condition.token_scope.value if condition.token_scope is not None else None
            ),
            "standardized_alpha": condition.alpha,
            "sparsity": condition.sparsity,
            "direction_sha256": "8" * 64,
            "direction_norm": 1.0,
            "reference_rms": 2.0,
            "raw_alpha": 2.0,
            "decay": 0.0,
            "applied_tokens": 4,
            "applied_token_indices": [0, 1, 2, 3],
            "activation_delta_norm": 4.0,
            "pre_activation_sha256": "a" * 64,
            "post_activation_sha256": "b" * 64,
            "delta_sha256": "c" * 64,
            "runtime_session_identity_sha256": "9" * 64,
        }
        metadata["intervention_trace"] = trace
        metadata["intervention_trace_digest"] = stable_hash(trace)
    return GenerationRecord(
        question_id="triviaqa:test:1",
        benchmark="triviaqa",
        model_repository=condition.model_repository,
        model_revision=condition.model_revision,
        runtime=condition.runtime,
        quantization=condition.quantization,
        system_prompt_id=condition.system_prompt_id,
        rendered_prompt_hash="7" * 64,
        steering_method=condition.steering_method,
        layer=condition.layer,
        site=condition.site,
        token_scope=condition.token_scope,
        alpha=condition.alpha,
        sparsity=condition.sparsity,
        controller_scores={},
        raw_output="Paris",
        normalized_answer="paris",
        outcome=Outcome.CORRECT,
        generation_latency_seconds=1.0,
        input_tokens=12,
        output_tokens=4,
        condition_id=condition.condition_id,
        seed=17,
        metadata=metadata,
    )


@pytest.mark.parametrize("method", ["M0", "M4"])
def test_confirmatory_receipt_binds_base_and_fixed_execution(method: str) -> None:
    private_hex, public_hex = _key_material()
    condition = _condition(method)
    unsigned = _record(condition)
    condition.validate_record(unsigned)
    signature = _sign_confirmatory_execution_receipt_for_test(
        unsigned,
        private_key_hex=private_hex,
    )
    signed = replace(
        unsigned,
        metadata={
            **unsigned.metadata,
            "confirmatory_execution_receipt_signature": signature,
        },
    )
    validate_confirmatory_execution_receipt(
        signed,
        condition,
        execution_public_key=public_hex,
    )
    changed = replace(signed, raw_output="Lyon")
    with pytest.raises(DataValidationError, match="not signed"):
        validate_confirmatory_execution_receipt(
            changed,
            condition,
            execution_public_key=public_hex,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing", "schema differs"),
        ("over-envelope", "over envelope"),
        ("forged-attested-envelope", "differs from attestation"),
        ("forged-auxiliary-peak", "differs from its embedded source"),
        ("wall-time", "wall time differs"),
    ),
)
def test_confirmatory_receipt_rejects_resigned_invalid_runtime_metrics(
    mutation: str,
    message: str,
) -> None:
    private_hex, public_hex = _key_material()
    condition = _condition("M0")
    source = _record(condition)
    metadata = dict(source.metadata)
    if mutation == "missing":
        metadata.pop("generation_runtime_metrics")
    else:
        metrics = dict(metadata["generation_runtime_metrics"])
        if mutation == "over-envelope":
            metrics["generation_peak_memory_bytes"] = 16 * 1024**3 + 1
            metrics["peak_memory_bytes"] = 16 * 1024**3 + 1
        elif mutation == "forged-attested-envelope":
            metrics["gpu_total_memory_bytes"] = 32 * 1024**3
        elif mutation == "forged-auxiliary-peak":
            metrics["auxiliary_peak_memory_bytes"] = 1
        else:
            metrics["generation_wall_time_seconds"] = 2.0
        metadata["generation_runtime_metrics"] = metrics
    invalid = replace(source, metadata=metadata)
    signed = replace(
        invalid,
        metadata={
            **invalid.metadata,
            "confirmatory_execution_receipt_signature": (
                _sign_confirmatory_execution_receipt_for_test(
                    invalid,
                    private_key_hex=private_hex,
                )
            ),
        },
    )

    with pytest.raises(DataValidationError, match=message):
        validate_confirmatory_execution_receipt(
            signed,
            condition,
            execution_public_key=public_hex,
            runtime_identity={"gpu_total_memory_bytes": 16 * 1024**3},
        )


def test_confirmatory_fixed_trace_rejects_nonexecuted_delta() -> None:
    private_hex, public_hex = _key_material()
    condition = _condition("M4")
    record = _record(condition)
    trace = dict(record.metadata["intervention_trace"])
    trace["activation_delta_norm"] = math.nextafter(0.0, 1.0)
    tampered = replace(
        record,
        metadata={
            **record.metadata,
            "intervention_trace": trace,
            "intervention_trace_digest": stable_hash(trace),
        },
    )
    signature = _sign_confirmatory_execution_receipt_for_test(
        tampered,
        private_key_hex=private_hex,
    )
    tampered = replace(
        tampered,
        metadata={
            **tampered.metadata,
            "confirmatory_execution_receipt_signature": signature,
        },
    )
    with pytest.raises(DataValidationError, match="does not prove"):
        validate_confirmatory_execution_receipt(
            tampered,
            condition,
            execution_public_key=public_hex,
        )


def test_confirmatory_fixed_trace_is_derived_from_packaged_component(
    tmp_path: Path,
) -> None:
    bank_path = build_e3_m1_bundle(
        tmp_path,
        direction=(1.0, 0.0),
        layers=(21,),
    )
    component = write_confirmatory_fixed_component(
        tmp_path / "component",
        source_artifact=bank_path,
        method="M1",
        layer=21,
        site=ActivationSite.POST_MLP,
        token_scope=TokenScope.FIRST_FOUR,
        standardized_alpha=1.0,
        reference_rms=2.0,
    )
    condition = replace(
        _condition("M1"),
        method_artifact_sha256=component.fingerprint,
    )
    unsigned = _record(condition)
    trace = {
        **dict(unsigned.metadata["intervention_trace"]),
        "method_artifact_sha256": component.fingerprint,
        "direction_sha256": component.direction_sha256,
        "direction_norm": component.direction_norm,
        "reference_rms": component.reference_rms,
    }
    unsigned = replace(
        unsigned,
        metadata={
            **unsigned.metadata,
            "method_artifact_sha256": component.fingerprint,
            "intervention_trace": trace,
            "intervention_trace_digest": stable_hash(trace),
        },
    )
    private_hex, public_hex = _key_material()
    signed = replace(
        unsigned,
        metadata={
            **unsigned.metadata,
            "confirmatory_execution_receipt_signature": (
                _sign_confirmatory_execution_receipt_for_test(
                    unsigned,
                    private_key_hex=private_hex,
                )
            ),
        },
    )
    validate_confirmatory_execution_receipt(
        signed,
        condition,
        execution_public_key=public_hex,
        fixed_component=component,
    )

    wrong_trace = {**trace, "direction_sha256": "0" * 64}
    tampered = replace(
        unsigned,
        metadata={
            **unsigned.metadata,
            "intervention_trace": wrong_trace,
            "intervention_trace_digest": stable_hash(wrong_trace),
        },
    )
    tampered = replace(
        tampered,
        metadata={
            **tampered.metadata,
            "confirmatory_execution_receipt_signature": (
                _sign_confirmatory_execution_receipt_for_test(
                    tampered,
                    private_key_hex=private_hex,
                )
            ),
        },
    )
    with pytest.raises(DataValidationError, match="packaged execution component"):
        validate_confirmatory_execution_receipt(
            tampered,
            condition,
            execution_public_key=public_hex,
            fixed_component=component,
        )


def test_native_e9_m0_executes_live_runtime_and_runtime_signs_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    condition = _condition("M0")
    model = load_model_spec(ROOT / "configs/models/qwen3.6-27b-nvfp4.yaml")
    prompt = next(
        value
        for value in load_prompt_specs(ROOT / "configs/prompts/primary.yaml")
        if value.prompt_id == "P0-neutral"
    )
    question = Question(
        question_id="triviaqa:test:1",
        benchmark="triviaqa",
        text="What is the capital of France?",
        aliases=("Paris",),
    )
    rendered = VllmRenderedPrompt(
        text="rendered prompt",
        sha256="7" * 64,
        token_ids=(1, 2, 3),
        token_ids_sha256=stable_hash([1, 2, 3]),
        messages=({"role": "user", "content": question.text},),
    )
    identity = {
        "backend": "vllm",
        "vllm": "0.24.0",
        "transformers": "5.2.0",
        "torch": "2.11.0",
        "python": "3.12",
        "architecture": "x86_64",
        "os": "Linux-test",
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
        "tokenizer_class": "TestTokenizer",
        "num_layers": model.num_layers,
        "hidden_size": 5_120,
        "seed": condition.seed,
        "model_repository": model.repository,
        "model_revision": model.revision,
        "model_quantization": model.quantization,
        "model_num_layers": model.num_layers,
        "snapshot_sha256": "8" * 64,
        "research_provenance": {"test": "native-boundary"},
        "research_toolchain": {
            "vllm": "0.24.0",
            "torch": "2.11.0",
            "transformers": "5.2.0",
            "numpy": "2.4.3",
            "nvidia_driver": "570.00",
        },
    }
    monkeypatch.setattr(VllmResearchRuntime, "runtime_identity", lambda _self: identity)
    monkeypatch.setattr(
        VllmResearchRuntime,
        "render_prompt",
        lambda _self, _prompt, _question, metadata=None: rendered,
    )

    def generate(
        _self: VllmResearchRuntime,
        actual_rendered: VllmRenderedPrompt,
        *,
        max_new_tokens: int,
        intervention_states: object,
    ) -> VllmGenerationOutput:
        assert max_new_tokens == 48
        assert intervention_states == {}
        return VllmGenerationOutput(
            rendered_prompt=actual_rendered,
            token_ids=(10,),
            text="Paris",
            input_tokens=3,
            output_tokens=1,
            latency_seconds=0.01,
            stop_type="eos",
            stopping_token_id=2,
            prompt_tokens_per_second=100.0,
            generation_tokens_per_second=100.0,
            peak_memory_bytes=1024,
            active_memory_bytes=512,
            cache_memory_bytes=256,
        )

    monkeypatch.setattr(VllmResearchRuntime, "generate_with_interventions", generate)
    runtime = object.__new__(VllmResearchRuntime)
    private_hex, public_hex = _key_material()
    attestor = E6RuntimeAttestor(runtime, execution_private_key=private_hex)
    runtime_artifact = tmp_path / "runtime-attestation.json"
    attestor.write_runtime_artifact(runtime_artifact)
    scorer = tmp_path / "scorer.json"
    write_side_effect_scorer_spec(scorer, execution_public_key=public_hex)
    ifeval = tmp_path / "ifeval"
    strongreject = tmp_path / "strongreject"
    ifeval.mkdir()
    strongreject.mkdir()
    (ifeval / "evaluator").write_text("test\n", encoding="utf-8")
    (strongreject / "grader").write_text("test\n", encoding="utf-8")
    official = tmp_path / "official-graders"
    official.mkdir()
    (official / "fixture.txt").write_text("test\n", encoding="utf-8")
    official_digest = "b" * 64
    monkeypatch.setattr(
        confirmatory_graders,
        "verify_e1_grader_bundle",
        lambda *_args, **_kwargs: {"manifest_digest": official_digest},
    )
    monkeypatch.setattr(
        confirmatory_graders,
        "validate_ifeval_evaluator",
        lambda path: sha256_path(path),
    )
    monkeypatch.setattr(
        confirmatory_graders,
        "validate_strongreject_grader",
        lambda path: sha256_path(path),
    )
    grader_bundle = tmp_path / "graders"
    write_confirmatory_grader_bundle(
        grader_bundle,
        official_grader_bundle=official,
        expected_official_manifest_digest=official_digest,
        side_effect_scorer=scorer,
        ifeval_evaluator=ifeval,
        strongreject_grader=strongreject,
        runtime_attestation=runtime_artifact,
    )
    backend = NativeE9VllmBackend(
        attestor=attestor,
        runtime_artifact=runtime_artifact,
        grader_bundle=grader_bundle,
        grader_transport=OpenRouterTransport(
            api_key="unused",
            sender=lambda _request, _timeout: pytest.fail("TriviaQA is locally scored"),
        ),
    )
    record = backend.execute(
        condition=condition,
        question=question,
        prompt=prompt,
        component_artifact=None,
    )
    assert record.outcome is Outcome.CORRECT
    assert record.raw_output == "Paris"
    assert record.metadata["generation_runtime_metrics"] == {
        "schema_version": 1,
        "gpu_total_memory_bytes": 40_000_000_000,
        "peak_memory_bytes": 1024,
        "generation_peak_memory_bytes": 1024,
        "auxiliary_peak_memory_bytes": 0,
        "active_memory_bytes": 512,
        "cache_memory_bytes": 256,
        "prompt_tokens_per_second": 100.0,
        "generation_tokens_per_second": 100.0,
        "generation_wall_time_seconds": 0.01,
        "stop_type": "eos",
        "stopping_token_id": 2,
    }
    assert record.metadata["confirmatory_execution_receipt_signature"]
    condition.validate_record(record)

    replacement = object.__new__(VllmResearchRuntime)
    attestor.runtime = replacement
    with pytest.raises(FrozenArtifactError, match="runtime was replaced"):
        backend.execute(
            condition=condition,
            question=question,
            prompt=prompt,
            component_artifact=None,
        )


def test_native_e9_terminal_grader_failure_is_an_unscorable_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = load_official_grader_spec(
        ROOT / "configs/graders/simpleqa-verified.yaml"
    )
    monkeypatch.setattr(e9_native, "_official_grader_spec", lambda _bundle, _name: spec)
    error_body = b'{"error":{"message":"quota exhausted"}}'

    def fail(_request: object, _timeout: float) -> tuple[int, bytes]:
        raise OpenRouterError(
            "rate limited",
            transient=True,
            retry_after=0,
            http_status=429,
            response_body=error_body,
        )

    backend = object.__new__(NativeE9VllmBackend)
    object.__setattr__(
        backend,
        "grader_bundle",
        cast(
            Any,
            SimpleNamespace(
                manifest_digest="1" * 64,
                official_manifest_digest="2" * 64,
            ),
        ),
    )
    object.__setattr__(
        backend,
        "grader_transport",
        OpenRouterTransport(api_key="key", sender=fail),
    )
    question = Question(
        question_id="simpleqa:test:1",
        benchmark="simpleqa_verified",
        text="What is the capital of France?",
        aliases=("Paris",),
    )
    condition = _condition("M0")
    source = replace(
        _record(condition),
        question_id=question.question_id,
        benchmark=question.benchmark,
        outcome=Outcome.INCORRECT,
    )
    graded = backend._grade(record=source, question=question)
    evidence = graded.metadata["official_grader_evidence"]
    assert graded.metadata["simpleqa_hedging_evidence"]["response_sha256"] == stable_hash(
        graded.raw_output
    )
    assert graded.outcome is Outcome.UNSCORABLE
    assert graded.metadata["grader_failed"] is True
    assert len(evidence["attempt_receipts"]) == 3
    assert evidence["terminal_error"] == (
        "OpenRouterError: OpenRouter HTTP 429: quota exhausted"
    )

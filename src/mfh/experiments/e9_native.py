"""Concrete native-VLLM execution boundary for confirmatory E9 rows."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mfh.contracts import (
    ActivationSite,
    GenerationRecord,
    Outcome,
    PromptSpec,
    Question,
    TokenScope,
)
from mfh.data.normalization import normalize_answer
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.evaluation.grading import deterministic_short_answer_grade, triviaqa_scores
from mfh.evaluation.official import GradingRequest, render_grader_prompt
from mfh.evaluation.openrouter import OpenRouterTransport, run_openrouter_grader
from mfh.evaluation.simpleqa import simpleqa_hedging_evidence
from mfh.experiments.confirmatory_components import (
    ConfirmatoryAdaptiveComponent,
    ConfirmatoryFixedComponent,
    load_confirmatory_adaptive_component,
    load_confirmatory_fixed_component,
)
from mfh.experiments.confirmatory_graders import (
    ConfirmatoryGraderBundle,
    _official_grader_spec,
    _question_fingerprint,
    validate_confirmatory_factual_grade,
    validate_confirmatory_grader_bundle,
)
from mfh.experiments.e6_likelihood import E6RuntimeAttestor
from mfh.experiments.e8_protected import (
    _compose_e8_controller_features,
    question_source_fingerprint,
)
from mfh.experiments.protocol import ExperimentPhase
from mfh.experiments.runner import (
    EvaluationCondition,
    adaptive_execution_receipt_body,
    adaptive_policy_decision_digest,
    confirmatory_execution_receipt_body,
)
from mfh.experiments.runtime_evidence import build_generation_runtime_metrics
from mfh.inference.vllm_research import VllmResearchInterventionState
from mfh.inference.vllm_runtime import VllmGenerationOutput, as_numpy
from mfh.provenance import sha256_file, stable_hash


def _token_indices(scope: TokenScope, output_tokens: int) -> list[int]:
    if scope is TokenScope.FINAL_PROMPT:
        return [-1]
    limit = {
        TokenScope.FIRST_GENERATED: 1,
        TokenScope.FIRST_FOUR: 4,
        TokenScope.FIRST_EIGHT: 8,
        TokenScope.ALL_GENERATED: output_tokens,
        TokenScope.EXPONENTIAL_DECAY: output_tokens,
    }[scope]
    return list(range(min(limit, output_tokens)))


def _strict_runtime_arrays(
    state: VllmResearchInterventionState, *, expected_applications: int
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    if (
        type(state) is not VllmResearchInterventionState
        or len(state.applied_pre_history) != expected_applications
        or len(state.applied_post_history) != expected_applications
    ):
        raise FrozenArtifactError("native E9 hook lacks its applied-edit history")
    captured = np.ascontiguousarray(
        np.stack(state.applied_pre_history).astype(np.float32, copy=False)
    )
    intervened = np.ascontiguousarray(
        np.stack(state.applied_post_history).astype(np.float32, copy=False)
    )
    if (
        captured.shape != intervened.shape
        or captured.size == 0
        or not np.isfinite(captured).all()
        or not np.isfinite(intervened).all()
        or np.array_equal(captured, intervened)
        or state.applications != expected_applications
        or expected_applications <= 0
    ):
        raise FrozenArtifactError("confirmatory VLLM hook did not execute its exact edit")
    delta = np.ascontiguousarray(intervened - captured)
    direction = np.ascontiguousarray(as_numpy(state.direction, dtype=np.float32))
    alphas = [
        state.alpha * math.exp(-state.decay * index)
        if state.token_scope is TokenScope.EXPONENTIAL_DECAY
        else state.alpha
        for index in range(expected_applications)
    ]
    expected = np.stack([direction * alpha for alpha in alphas]).astype(
        np.float32, copy=False
    )
    tolerance = max(1e-6, float(np.max(np.abs(expected))) * 0.025)
    if (
        direction.ndim != 1
        or delta.shape != expected.shape
        or not np.allclose(delta, expected, rtol=0.025, atol=tolerance)
        or not math.isclose(
            float(np.linalg.norm(delta.astype(np.float64))),
            float(np.linalg.norm(expected.astype(np.float64))),
            rel_tol=0.025,
            abs_tol=1e-6,
        )
    ):
        raise FrozenArtifactError("native E9 applied edit differs from direction and alpha")
    return captured, intervened, delta


_CONFIRMATORY_MAX_NEW_TOKENS = 48


@dataclass(frozen=True, slots=True, init=False)
class NativeE9VllmBackend:
    """Only accepted E9 backend: live VLLM generation plus frozen official grading."""

    attestor: E6RuntimeAttestor
    runtime_artifact: Path
    grader_bundle: ConfirmatoryGraderBundle
    grader_transport: OpenRouterTransport
    _runtime: Any
    max_new_tokens: int

    def __init__(
        self,
        *,
        attestor: E6RuntimeAttestor,
        runtime_artifact: str | Path,
        grader_bundle: str | Path,
        grader_transport: OpenRouterTransport,
    ) -> None:
        if (
            type(attestor) is not E6RuntimeAttestor
            or type(grader_transport) is not OpenRouterTransport
        ):
            raise DataValidationError("native E9 requires exact VLLM attestor and transport")
        runtime_path = Path(runtime_artifact).resolve()
        bundle = validate_confirmatory_grader_bundle(grader_bundle)
        runtime_sha = attestor.verify_runtime_artifact(runtime_path)
        if (
            runtime_sha != sha256_file(bundle.directory / "runtime-attestation.json")
            or attestor.execution_public_key != bundle.scorer.execution_public_key
        ):
            raise FrozenArtifactError(
                "native E9 runtime differs from the packaged confirmatory attestation"
            )
        object.__setattr__(self, "attestor", attestor)
        object.__setattr__(self, "runtime_artifact", runtime_path)
        object.__setattr__(self, "grader_bundle", bundle)
        object.__setattr__(self, "grader_transport", grader_transport)
        object.__setattr__(self, "_runtime", attestor.runtime)
        object.__setattr__(self, "max_new_tokens", _CONFIRMATORY_MAX_NEW_TOKENS)

    def _live_runtime_identity(self) -> str:
        if self.attestor.runtime is not self._runtime:
            raise FrozenArtifactError("native E9 attestor runtime was replaced")
        self.attestor.verify_runtime_artifact(self.runtime_artifact)
        identity = self.attestor.assert_live_runtime(self._runtime)
        if identity != self.grader_bundle.runtime_identity_digest:
            raise FrozenArtifactError("native E9 live runtime differs from grader bundle")
        return identity

    def _selective_risk_score(
        self,
        *,
        component: ConfirmatoryAdaptiveComponent,
        rendered: Any,
        controller_prompt_id: str,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Apply the same frozen prompt-risk probe to every E9 method.

        The score is deliberately calculated before method-specific generation.
        It can therefore be threshold-swept without using the observed outcome and
        supports matched selective-prediction comparisons between static and
        adaptive methods.
        """

        try:
            controller = component.controllers[controller_prompt_id]
        except KeyError as exc:
            raise FrozenArtifactError(
                "selective-risk component lacks the row prompt controller"
            ) from exc
        schema = controller.risk_probe.training_schema
        cube = self._runtime.prompt_feature_cube(
            rendered,
            layers=schema.layers,
            sites=schema.sites,
        )
        features = _compose_e8_controller_features(schema, cube.activations)
        decision = controller.decide(features)
        if decision.class_labels != ("C", "I", "A"):
            raise DataValidationError("selective-risk controller labels differ from C/I/A")
        scores = {
            label: float(decision.probabilities[0, index])
            for index, label in enumerate(decision.class_labels)
        }
        feature_values = np.ascontiguousarray(features.numpy(), dtype=np.float32)
        evidence = {
            "schema_version": 1,
            "score_semantics": "frozen-pre-generation-CIA-prompt-risk",
            "controller_artifact_sha256": component.fingerprint,
            "controller_prompt_id": controller_prompt_id,
            "feature_schema_digest": schema.digest,
            "feature_values_sha256": hashlib.sha256(
                feature_values.tobytes(order="C")
            ).hexdigest(),
            "feature_values": feature_values.reshape(-1).tolist(),
            "prompt_feature_peak_memory_bytes": cube.peak_memory_bytes,
            "scores": dict(scores),
            "predicted_hallucination_risk": scores["I"],
        }
        return scores, evidence

    def _fixed_execution(
        self,
        *,
        condition: EvaluationCondition,
        component: ConfirmatoryFixedComponent,
        rendered: Any,
    ) -> tuple[VllmGenerationOutput, dict[str, Any]]:
        values = np.ascontiguousarray(
            component.direction.detach().cpu().float().numpy(), dtype=np.float32
        )
        state = self._runtime.standardized_intervention_state(
            values,
            standardized_alpha=component.standardized_alpha,
            reference_rms=component.reference_rms,
            token_scope=component.token_scope,
            decay=component.decay,
        )
        generated = self._runtime.generate_with_interventions(
            rendered,
            max_new_tokens=self.max_new_tokens,
            intervention_states={(component.layer, component.site): state},
        )
        if type(generated) is not VllmGenerationOutput:
            raise FrozenArtifactError("native E9 runtime returned a non-VLLM generation")
        indices = _token_indices(component.token_scope, generated.output_tokens)
        captured, intervened, delta = _strict_runtime_arrays(
            state, expected_applications=len(indices)
        )
        raw_alpha = component.standardized_alpha * component.reference_rms
        trace = {
            "schema_version": 1,
            "method_artifact_sha256": component.fingerprint,
            "layer": component.layer,
            "site": component.site.value,
            "token_scope": component.token_scope.value,
            "standardized_alpha": component.standardized_alpha,
            "sparsity": component.sparsity,
            "direction_sha256": component.direction_sha256,
            "direction_norm": component.direction_norm,
            "reference_rms": component.reference_rms,
            "raw_alpha": raw_alpha,
            "decay": component.decay,
            "applied_tokens": state.applications,
            "applied_token_indices": indices,
            "activation_delta_norm": float(np.linalg.norm(delta.astype(np.float64))),
            "pre_activation_sha256": hashlib.sha256(captured.tobytes(order="C")).hexdigest(),
            "post_activation_sha256": hashlib.sha256(intervened.tobytes(order="C")).hexdigest(),
            "delta_sha256": hashlib.sha256(delta.tobytes(order="C")).hexdigest(),
            "runtime_session_identity_sha256": self.grader_bundle.runtime_identity_digest,
        }
        if component.fingerprint != condition.method_artifact_sha256:
            raise FrozenArtifactError("native E9 fixed component differs from its condition")
        return generated, {
            "intervention_trace": trace,
            "intervention_trace_digest": stable_hash(trace),
        }

    def _adaptive_execution(
        self,
        *,
        condition: EvaluationCondition,
        component: ConfirmatoryAdaptiveComponent,
        rendered: Any,
        controller_prompt_id: str,
    ) -> tuple[
        VllmGenerationOutput,
        int | None,
        ActivationSite | None,
        TokenScope | None,
        float,
        float | None,
        dict[str, float],
        dict[str, Any],
    ]:
        try:
            controller = component.controllers[controller_prompt_id]
        except KeyError as exc:
            raise FrozenArtifactError("native E9 component lacks the row controller") from exc
        policy = condition.adaptive_policy
        if policy is None or component.fingerprint != condition.method_artifact_sha256:
            raise FrozenArtifactError("native E9 adaptive component differs from condition")
        schema = controller.risk_probe.training_schema
        cube = self._runtime.prompt_feature_cube(rendered, layers=schema.layers, sites=schema.sites)
        features = _compose_e8_controller_features(schema, cube.activations)
        decision = controller.decide(features)
        if decision.class_labels != ("C", "I", "A"):
            raise DataValidationError("native E9 controller labels differ from C/I/A")
        scores = {
            label: float(decision.probabilities[0, index])
            for index, label in enumerate(decision.class_labels)
        }
        assert policy.alpha_risk_threshold is not None
        action = (
            "release"
            if scores["I"] <= policy.release_risk_threshold
            or scores["A"] >= policy.abstention_probability_threshold
            or (
                policy.alpha_mode == "risk_gated_hard_threshold"
                and scores["I"] < policy.alpha_risk_threshold
            )
            else "intervene"
        )
        layer: int | None = None
        site: ActivationSite | None = None
        scope: TokenScope | None = None
        alpha = 0.0
        state: VllmResearchInterventionState | None = None
        normalized: np.ndarray[Any, Any] | None = None
        direction_norm = 0.0
        routing_weights = [float(value) for value in decision.routing_weights[0]]
        interventions: dict[tuple[int, ActivationSite], Any] = {}
        if action == "intervene":
            layer = int(decision.selected_layers[0])
            eligible = [
                (key, value[0].detach().cpu().float().contiguous())
                for key, value in decision.directions.items()
                if key.layer == layer and key.site in policy.candidate_sites
            ]
            if not eligible:
                raise DataValidationError("native E9 controller selected no direction")
            selected_key, selected = min(
                eligible,
                key=lambda item: (
                    -float(torch.linalg.vector_norm(item[1])),
                    item[0].site.value,
                ),
            )
            direction = np.ascontiguousarray(selected.numpy(), dtype=np.float32)
            direction_norm = float(np.linalg.norm(direction))
            if not math.isfinite(direction_norm) or direction_norm <= 0:
                raise DataValidationError("native E9 routed direction is invalid")
            normalized = np.ascontiguousarray(direction / direction_norm)
            site = selected_key.site
            scope = policy.candidate_token_scopes[0]
            alpha = (
                policy.alpha_max
                if policy.alpha_mode == "fixed"
                else policy.alpha_max
                / (1.0 + math.exp(-policy.alpha_beta * (scores["I"] - policy.alpha_risk_threshold)))
            )
            if not math.isclose(float(decision.alphas[0]), alpha, rel_tol=1e-5, abs_tol=1e-7):
                raise DataValidationError("native E9 controller alpha differs")
            state = self._runtime.standardized_intervention_state(
                normalized,
                standardized_alpha=alpha * direction_norm,
                reference_rms=1.0,
                token_scope=scope,
            )
            interventions[(layer, site)] = state
        generated = self._runtime.generate_with_interventions(
            rendered,
            max_new_tokens=self.max_new_tokens,
            intervention_states=interventions,
        )
        if type(generated) is not VllmGenerationOutput:
            raise FrozenArtifactError("native E9 runtime returned a non-VLLM generation")
        feature_values = np.ascontiguousarray(features.numpy(), dtype=np.float32)
        metadata: dict[str, Any] = {
            "policy_action": action,
            "adaptive_controller_evidence": {
                "schema_version": 1,
                "controller_artifact_sha256": component.fingerprint,
                "feature_schema_digest": schema.digest,
                "feature_values_sha256": hashlib.sha256(
                    feature_values.tobytes(order="C")
                ).hexdigest(),
                "feature_values": feature_values.reshape(-1).tolist(),
                "prompt_feature_peak_memory_bytes": cube.peak_memory_bytes,
                "maximum_token_probability": cube.maximum_token_probability,
                "output_entropy": cube.output_entropy,
                "site_selection": "max_mixed_direction_norm_then_site",
            },
        }
        if action == "intervene":
            assert state is not None and normalized is not None
            assert layer is not None and site is not None and scope is not None
            indices = _token_indices(scope, generated.output_tokens)
            captured, intervened, delta = _strict_runtime_arrays(
                state, expected_applications=len(indices)
            )
            trace = {
                "layer": layer,
                "site": site.value,
                "token_scope": scope.value,
                "alpha": alpha,
                "sparsity": policy.sparsity,
                "applied_tokens": state.applications,
                "applied_token_indices": indices,
                "activation_delta_norm": float(np.linalg.norm(delta.astype(np.float64))),
                "direction_sha256": hashlib.sha256(normalized.tobytes(order="C")).hexdigest(),
                "direction_norm": direction_norm,
                "controller_artifact_sha256": component.fingerprint,
                "router_weights": routing_weights,
                "router_weights_sha256": stable_hash(routing_weights),
                "pre_activation_sha256": hashlib.sha256(captured.tobytes(order="C")).hexdigest(),
                "post_activation_sha256": hashlib.sha256(intervened.tobytes(order="C")).hexdigest(),
                "delta_sha256": hashlib.sha256(delta.tobytes(order="C")).hexdigest(),
            }
            metadata.update(
                {
                    "intervention_trace": trace,
                    "intervention_trace_digest": stable_hash(trace),
                }
            )
        return (
            generated,
            layer,
            site,
            scope,
            alpha,
            policy.sparsity if action == "intervene" else None,
            scores,
            metadata,
        )

    def _grade(self, *, record: GenerationRecord, question: Question) -> GenerationRecord:
        metadata = dict(record.metadata)
        metadata["official_score_output_sha256"] = stable_hash(record.raw_output)
        if record.benchmark == "simpleqa_verified":
            metadata["simpleqa_hedging_evidence"] = simpleqa_hedging_evidence(
                record.raw_output
            )
        if record.benchmark == "triviaqa":
            exact_match, token_f1 = triviaqa_scores(record.raw_output, question.aliases)
            outcome = deterministic_short_answer_grade(record.raw_output, question.aliases)
            metadata.update(
                {
                    "official_scorer": "mfh.triviaqa.alias-aware-em-f1.v1",
                    "official_exact_match": exact_match,
                    "official_token_f1": token_f1,
                    "reference_aliases_digest": stable_hash(list(question.aliases)),
                }
            )
            return replace(
                record,
                outcome=outcome,
                normalized_answer=normalize_answer(record.raw_output),
                metadata=metadata,
            )
        spec = _official_grader_spec(self.grader_bundle, record.benchmark)
        request = GradingRequest(
            question.question_id,
            question.text,
            question.aliases[0],
            record.raw_output,
        )
        prompt = render_grader_prompt(spec, request)
        start = len(self.grader_transport.receipts)
        grade = run_openrouter_grader(spec, request, self.grader_transport)
        attempts = [value.to_dict() for value in self.grader_transport.receipts[start:]]
        evidence = {
            "schema_version": 2,
            "grader_bundle_manifest_digest": self.grader_bundle.manifest_digest,
            "official_grader_manifest_digest": self.grader_bundle.official_manifest_digest,
            "grader_spec_digest": spec.digest,
            "question_source_sha256": _question_fingerprint(question),
            "response_sha256": stable_hash(record.raw_output),
            "request_fingerprint": request.digest,
            "rendered_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "raw_label": grade.raw_response,
            "terminal_error": grade.error,
            "attempt_receipts": attempts,
        }
        metadata.update(
            {
                "official_grader_evidence": evidence,
                "grader_attempts": grade.attempts,
                "grader_failed": grade.error is not None,
                "grader_request_fingerprint": request.digest,
                "grader_fingerprint": spec.digest,
                "grader_raw_label": grade.raw_response,
                "grader_bundle_manifest_digest": self.grader_bundle.manifest_digest,
                "grader_model": spec.grader_model,
                "grader_model_revision": spec.grader_model_revision,
                "grader_source_artifact_sha256": spec.source_artifact_sha256,
            }
        )
        return replace(
            record,
            outcome=grade.outcome,
            normalized_answer=normalize_answer(record.raw_output),
            metadata=metadata,
        )

    def execute(
        self,
        *,
        condition: EvaluationCondition,
        question: Question,
        prompt: PromptSpec,
        component_artifact: Path | None,
        selective_risk_component_artifact: Path | None = None,
        controller_prompt_id: str | None = None,
        signed_metadata: Mapping[str, Any] | None = None,
    ) -> GenerationRecord:
        if (
            type(condition) is not EvaluationCondition
            or condition.phase is not ExperimentPhase.E9
            or question.benchmark != condition.benchmark
            or prompt.prompt_id != condition.system_prompt_id
            or hashlib.sha256(prompt.text.encode()).hexdigest() != condition.prompt_template_sha256
        ):
            raise DataValidationError("native E9 row differs from its frozen schedule")
        selected_controller_prompt = controller_prompt_id or condition.system_prompt_id
        if (
            condition.steering_method != "M3"
            and selected_controller_prompt != condition.system_prompt_id
        ):
            raise DataValidationError("only M3 may select a different controller prompt")
        extra_metadata = dict(signed_metadata or {})
        protected_metadata = {
            "phase",
            "partition",
            "prompt_template_sha256",
            "study_protocol_digest",
            "runtime_session_identity_sha256",
            "decoding_max_new_tokens",
            "source_question_sha256",
            "method_artifact_sha256",
            "controller_prompt_id",
            "generation_runtime_metrics",
        }
        if protected_metadata & set(extra_metadata):
            raise DataValidationError("additional signed metadata shadows runtime evidence")
        live_identity = self._live_runtime_identity()
        rendered = self._runtime.render_prompt(prompt, question.text, metadata=question.metadata)
        layer = condition.layer
        site = condition.site
        scope = condition.token_scope
        alpha = condition.alpha
        sparsity = condition.sparsity
        scores: dict[str, float] = {}
        execution_metadata: dict[str, Any] = {}
        selective_component: ConfirmatoryAdaptiveComponent | None = None
        if selective_risk_component_artifact is not None:
            selective_component = load_confirmatory_adaptive_component(
                selective_risk_component_artifact
            )
        if condition.steering_method != "M3" and selective_component is not None:
            scores, selective_evidence = self._selective_risk_score(
                component=selective_component,
                rendered=rendered,
                controller_prompt_id=condition.system_prompt_id,
            )
            execution_metadata["selective_risk_evidence"] = selective_evidence
        if condition.steering_method == "M0":
            if component_artifact is not None:
                raise DataValidationError("native E9 M0 cannot receive a component")
            generated = self._runtime.generate_with_interventions(
                rendered,
                max_new_tokens=self.max_new_tokens,
                intervention_states={},
            )
        elif condition.steering_method == "M3":
            if component_artifact is None:
                raise DataValidationError("native E9 M3 lacks its component")
            component = load_confirmatory_adaptive_component(component_artifact)
            (
                generated,
                layer,
                site,
                scope,
                alpha,
                sparsity,
                scores,
                execution_metadata,
            ) = self._adaptive_execution(
                condition=condition,
                component=component,
                rendered=rendered,
                controller_prompt_id=selected_controller_prompt,
            )
            if selective_component is not None and (
                component_artifact is None
                or selective_component.fingerprint
                != load_confirmatory_adaptive_component(component_artifact).fingerprint
            ):
                raise FrozenArtifactError(
                    "E9 M3 selective-risk component differs from its executed controller"
                )
        else:
            if component_artifact is None:
                raise DataValidationError("native E9 fixed method lacks its component")
            fixed = load_confirmatory_fixed_component(component_artifact)
            generated, execution_metadata = self._fixed_execution(
                condition=condition,
                component=fixed,
                rendered=rendered,
            )
        if condition.steering_method == "M3":
            adaptive_evidence = execution_metadata.get("adaptive_controller_evidence")
            if not isinstance(adaptive_evidence, Mapping):
                raise FrozenArtifactError("E9 M3 lacks signed selective-risk evidence")
            execution_metadata["selective_risk_evidence"] = {
                "schema_version": 1,
                "score_semantics": "frozen-pre-generation-CIA-prompt-risk",
                "controller_artifact_sha256": adaptive_evidence[
                    "controller_artifact_sha256"
                ],
                "controller_prompt_id": selected_controller_prompt,
                "feature_schema_digest": adaptive_evidence["feature_schema_digest"],
                "feature_values_sha256": adaptive_evidence["feature_values_sha256"],
                "feature_values": adaptive_evidence["feature_values"],
                "prompt_feature_peak_memory_bytes": adaptive_evidence[
                    "prompt_feature_peak_memory_bytes"
                ],
                "scores": dict(scores),
                "predicted_hallucination_risk": scores["I"],
            }
        if type(generated) is not VllmGenerationOutput or generated.rendered_prompt != rendered:
            raise FrozenArtifactError("native E9 generation differs from rendered prompt")
        risk_evidence = execution_metadata.get("selective_risk_evidence")
        auxiliary_peak = (
            int(risk_evidence["prompt_feature_peak_memory_bytes"])
            if isinstance(risk_evidence, Mapping)
            else 0
        )
        metadata = {
            "phase": ExperimentPhase.E9.value,
            "partition": condition.partition,
            "prompt_template_sha256": condition.prompt_template_sha256,
            "study_protocol_digest": condition.study_protocol_digest,
            "runtime_session_identity_sha256": live_identity,
            "decoding_max_new_tokens": self.max_new_tokens,
            "source_question_sha256": question_source_fingerprint(question),
            "generation_runtime_metrics": build_generation_runtime_metrics(
                generated,
                runtime_identity=self.attestor.attested_runtime_identity,
                auxiliary_peak_memory_bytes=auxiliary_peak,
            ),
            **(
                {"controller_prompt_id": selected_controller_prompt}
                if condition.steering_method == "M3"
                else {}
            ),
            **(
                {"method_artifact_sha256": condition.method_artifact_sha256}
                if condition.method_artifact_sha256 is not None
                else {}
            ),
            **execution_metadata,
            **extra_metadata,
        }
        unsigned = GenerationRecord(
            question_id=question.question_id,
            benchmark=question.benchmark,
            model_repository=condition.model_repository,
            model_revision=condition.model_revision,
            runtime=condition.runtime,
            quantization=condition.quantization,
            system_prompt_id=condition.system_prompt_id,
            rendered_prompt_hash=rendered.sha256,
            steering_method=condition.steering_method,
            layer=layer,
            site=site,
            token_scope=scope,
            alpha=alpha,
            sparsity=sparsity,
            controller_scores=scores,
            raw_output=generated.text,
            normalized_answer=normalize_answer(generated.text),
            outcome=Outcome.INCORRECT,
            generation_latency_seconds=generated.latency_seconds,
            input_tokens=generated.input_tokens,
            output_tokens=generated.output_tokens,
            condition_id=condition.condition_id,
            seed=condition.seed,
            metadata=metadata,
        )
        graded = self._grade(record=unsigned, question=question)
        self._live_runtime_identity()
        if condition.steering_method == "M3":
            policy = condition.adaptive_policy
            assert policy is not None
            action = str(graded.metadata["policy_action"])
            graded = replace(
                graded,
                metadata={
                    **dict(graded.metadata),
                    "policy_decision_digest": adaptive_policy_decision_digest(
                        graded, policy=policy, policy_action=action
                    ),
                },
            )
            graded = replace(
                graded,
                metadata={
                    **dict(graded.metadata),
                    "execution_receipt_signature": self.attestor._sign(
                        adaptive_execution_receipt_body(graded, policy=policy)
                    ),
                },
            )
        signed = replace(
            graded,
            metadata={
                **dict(graded.metadata),
                "confirmatory_execution_receipt_signature": self.attestor._sign(
                    confirmatory_execution_receipt_body(graded)
                ),
            },
        )
        condition.validate_record(signed)
        validate_confirmatory_factual_grade(signed, question, grader_bundle=self.grader_bundle)
        return signed

"""Native CUDA/vLLM execution and semantic replay for frozen E10 M6 rows."""

from __future__ import annotations

import hashlib
import math
import time
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
from mfh.evaluation.ifeval import evaluate_ifeval_strict
from mfh.evaluation.language import (
    language_response_evidence,
    requested_language_is_correct,
)
from mfh.evaluation.openrouter import OpenRouterTransport
from mfh.evaluation.side_effects import (
    deterministic_refusal_decision,
    official_metric_receipt_body,
    recompute_mmlu_pro_accuracy,
    safety_score_receipt_body,
)
from mfh.evaluation.strongreject import (
    StrongRejectTerminalFailure,
    grade_strongreject_openrouter,
)
from mfh.experiments.confirmatory_graders import (
    ConfirmatoryGraderBundle,
    validate_confirmatory_grader_bundle,
)
from mfh.experiments.e6_likelihood import E6RuntimeAttestor
from mfh.experiments.e8_protected import (
    _compose_e8_controller_features,
    question_source_fingerprint,
    validate_wikitext_likelihood_evidence,
)
from mfh.experiments.e9_native import (
    NativeE9VllmBackend,
    _strict_runtime_arrays,
    _token_indices,
)
from mfh.experiments.protocol import ExperimentPhase
from mfh.experiments.runner import (
    EvaluationCondition,
    adaptive_execution_receipt_body,
    adaptive_policy_decision_digest,
    confirmatory_execution_receipt_body,
    validate_confirmatory_execution_receipt,
)
from mfh.inference.vllm_research import (
    VllmOnlinePrefixCapture,
    VllmResearchInterventionState,
)
from mfh.inference.vllm_runtime import VllmGenerationOutput
from mfh.methods.composite import (
    CompositePolicy,
    EarlyReevaluation,
    OutputAction,
    RiskRegime,
    load_composite_policy,
)
from mfh.methods.features import ActivationKind, FeatureComposition
from mfh.provenance import sha256_file, sha256_path, stable_hash

_MAX_NEW_TOKENS = 48
_FACTUAL = {"triviaqa", "simpleqa_verified", "aa_omniscience_public_600"}


def _early_feature_limit(kind: ActivationKind) -> int:
    return {
        ActivationKind.FIRST_GENERATED: 1,
        ActivationKind.FIRST_FOUR_GENERATED: 4,
        ActivationKind.FIRST_EIGHT_GENERATED: 8,
    }[kind]


def _compose_early_features(
    policy: CompositePolicy,
    activations: dict[ActivationSite, dict[int, np.ndarray[Any, Any]]],
) -> torch.Tensor:
    """Pool the declared first-token block and reproduce its frozen composition."""

    assert policy.early_probe is not None
    schema = policy.early_probe.training_schema
    limit = _early_feature_limit(schema.activation_kind)
    try:
        pooled = {
            site: [
                np.asarray(activations[site][layer], dtype=np.float32)[:limit].mean(axis=0)
                for layer in schema.layers
            ]
            for site in schema.sites
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise DataValidationError(f"M6 early feature cube differs: {exc}") from exc
    if any(
        not values or any(value.ndim != 1 or not np.isfinite(value).all() for value in values)
        for values in pooled.values()
    ):
        raise DataValidationError("M6 early feature trajectories are empty or invalid")
    if schema.composition is FeatureComposition.SINGLE_LAYER:
        parts = [values[0] for values in pooled.values()]
    elif schema.composition is FeatureComposition.CONCATENATED_LAYERS:
        parts = [value for values in pooled.values() for value in values]
    elif schema.composition is FeatureComposition.LAYER_DIFFERENCES:
        parts = [
            values[index + 1] - values[index]
            for values in pooled.values()
            for index in range(len(values) - 1)
        ]
    else:  # pragma: no cover - exhaustive enum
        raise DataValidationError("M6 early feature composition is unsupported")
    row = np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)
    if row.shape != (schema.width,) or not np.isfinite(row).all():
        raise DataValidationError("M6 early feature width differs from its probe")
    return torch.from_numpy(row.copy()).unsqueeze(0)


def _probabilities(probe: Any, features: torch.Tensor) -> dict[str, float]:
    values = probe.predict_probabilities(features)
    return {label: float(values[0, index]) for index, label in enumerate(probe.state.labels)}


def _feature_evidence(
    values: torch.Tensor,
    *,
    schema_digest: str,
) -> dict[str, Any]:
    array = np.ascontiguousarray(values.detach().cpu().float().numpy(), dtype=np.float32)
    return {
        "schema_digest": schema_digest,
        "values": array.reshape(-1).tolist(),
        "values_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _candidate_constraints(question: Question, text: str) -> dict[str, bool]:
    refused = deterministic_refusal_decision(text)
    requested = question.metadata.get("requested_language")
    return {
        "safety_ok": (refused if question.benchmark == "strongreject_or_harmbench" else True),
        "language_ok": (
            requested_language_is_correct(text, str(requested), question.aliases)
            if question.benchmark == "language_consistency"
            else True
        ),
        "refusal_drift": question.benchmark == "xstest" and refused,
    }


@dataclass(frozen=True, slots=True, init=False)
class NativeE10VllmBackend:
    """Only accepted E10 backend: buffered M6 generation, gating, and scoring."""

    attestor: E6RuntimeAttestor
    runtime_artifact: Path
    grader_bundle: ConfirmatoryGraderBundle
    grader_transport: OpenRouterTransport
    _runtime: Any
    _factual_backend: NativeE9VllmBackend
    _policy_cache: dict[str, tuple[Path, CompositePolicy]]
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
            raise DataValidationError("native E10 requires exact VLLM attestor and transport")
        runtime_path = Path(runtime_artifact).resolve()
        bundle = validate_confirmatory_grader_bundle(grader_bundle)
        if (
            attestor.verify_runtime_artifact(runtime_path)
            != sha256_file(bundle.directory / "runtime-attestation.json")
            or attestor.execution_public_key != bundle.scorer.execution_public_key
        ):
            raise FrozenArtifactError("native E10 runtime differs from its grader bundle")
        factual = NativeE9VllmBackend(
            attestor=attestor,
            runtime_artifact=runtime_path,
            grader_bundle=bundle.directory,
            grader_transport=grader_transport,
        )
        object.__setattr__(self, "attestor", attestor)
        object.__setattr__(self, "runtime_artifact", runtime_path)
        object.__setattr__(self, "grader_bundle", bundle)
        object.__setattr__(self, "grader_transport", grader_transport)
        object.__setattr__(self, "_runtime", attestor.runtime)
        object.__setattr__(self, "_factual_backend", factual)
        object.__setattr__(self, "_policy_cache", {})
        object.__setattr__(self, "max_new_tokens", _MAX_NEW_TOKENS)

    def _load_policy(
        self,
        component_artifact: Path,
        *,
        artifact_sha256: str,
    ) -> CompositePolicy:
        """Deep-validate a frozen M6 policy once per exact artifact identity."""

        canonical = component_artifact.resolve()
        cached = self._policy_cache.get(artifact_sha256)
        if cached is not None:
            cached_path, policy = cached
            if canonical != cached_path:
                raise FrozenArtifactError("native E10 component path changed during execution")
            return policy
        if sha256_path(canonical) != artifact_sha256:
            raise DataValidationError("native E10 component differs from its frozen condition")
        policy = load_composite_policy(canonical)
        self._policy_cache[artifact_sha256] = (canonical, policy)
        return policy

    def _live_identity(self) -> str:
        if self.attestor.runtime is not self._runtime:
            raise FrozenArtifactError("native E10 attestor runtime was replaced")
        self.attestor.verify_runtime_artifact(self.runtime_artifact)
        identity = self.attestor.assert_live_runtime(self._runtime)
        if identity != self.grader_bundle.runtime_identity_digest:
            raise FrozenArtifactError("native E10 live runtime differs from grader bundle")
        return identity

    def _gold_likelihood_diagnostic(
        self,
        *,
        rendered: Any,
        question: Question,
        action: str,
        layer: int | None,
        site: ActivationSite | None,
        scope: TokenScope | None,
        normalized: np.ndarray[Any, Any] | None,
        standardized_alpha: float,
    ) -> dict[str, Any] | None:
        """Measure gold-LL improvement after routing without feeding it into control."""

        if action != "intervene" or question.benchmark not in _FACTUAL:
            return None
        if layer is None or site is None or scope is None or normalized is None:
            raise FrozenArtifactError("M6 gold-likelihood intervention is incomplete")
        baseline_nll: list[float] = []
        intervened_nll: list[float] = []
        peak_memory_bytes = 0
        for alias in question.aliases:
            baseline = self._runtime.teacher_forced_continuation(
                rendered,
                alias,
                layers=(layer,),
                site=site,
                intervention_states={},
            )
            state = self._runtime.standardized_intervention_state(
                normalized,
                standardized_alpha=standardized_alpha,
                reference_rms=1.0,
                token_scope=scope,
            )
            steered = self._runtime.teacher_forced_continuation(
                rendered,
                alias,
                layers=(layer,),
                site=site,
                intervention_states={layer: state},
            )
            baseline_nll.append(float(baseline.mean_negative_log_likelihood))
            intervened_nll.append(float(steered.mean_negative_log_likelihood))
            peak_memory_bytes = max(
                peak_memory_bytes,
                baseline.peak_memory_bytes,
                steered.peak_memory_bytes,
            )
        baseline_best = min(baseline_nll)
        intervened_best = min(intervened_nll)
        delta = baseline_best - intervened_best
        if not math.isfinite(delta):
            raise DataValidationError("M6 gold-likelihood diagnostic is non-finite")
        return {
            "schema_version": 1,
            "accepted_alias_set_sha256": stable_hash(list(question.aliases)),
            "alias_count": len(question.aliases),
            "baseline_best_mean_negative_log_likelihood": baseline_best,
            "intervened_best_mean_negative_log_likelihood": intervened_best,
            "gold_log_likelihood_delta": delta,
            "gold_likelihood_improved": delta > 0,
            "used_for_control": False,
            "computed_after_online_decision": True,
            "maximum_peak_memory_bytes": peak_memory_bytes,
        }

    def _safety_signature(self, record: GenerationRecord, revision: str) -> str:
        return self.attestor._sign(safety_score_receipt_body(record, scorer_revision=revision))

    def _side_grade(
        self,
        record: GenerationRecord,
        question: Question,
    ) -> GenerationRecord:
        benchmark = record.benchmark
        metadata = dict(record.metadata)
        scorer = self.grader_bundle.scorer
        if benchmark == "ifeval":
            passed, instruction_passes = evaluate_ifeval_strict(
                question,
                record.raw_output,
                evaluator_directory=self.grader_bundle.directory / "ifeval-evaluator",
            )
            metadata.update(
                {
                    "official_pass": passed,
                    "official_instruction_passes": list(instruction_passes),
                    "official_metric_scorer_revision": scorer.scorer_revision,
                }
            )
            unsigned = replace(
                record,
                outcome=(Outcome.CORRECT if passed else record.outcome),
                metadata=metadata,
            )
            signature = self.attestor._sign(
                official_metric_receipt_body(
                    unsigned,
                    metric="ifeval_pass_rate",
                    value=passed,
                    scorer_revision=scorer.scorer_revision,
                )
            )
            return replace(
                unsigned,
                metadata={
                    **dict(unsigned.metadata),
                    "official_metric_receipt_signatures": {"ifeval_pass_rate": signature},
                },
            )
        if benchmark == "mmlu_pro":
            correct = recompute_mmlu_pro_accuracy(record, question)
            metadata.update(
                {
                    "official_correct": correct,
                    "official_metric_scorer_revision": scorer.scorer_revision,
                }
            )
            unsigned = replace(
                record,
                outcome=Outcome.CORRECT if correct else Outcome.INCORRECT,
                metadata=metadata,
            )
            signature = self.attestor._sign(
                official_metric_receipt_body(
                    unsigned,
                    metric="mmlu_pro_accuracy",
                    value=correct,
                    scorer_revision=scorer.scorer_revision,
                )
            )
            return replace(
                unsigned,
                metadata={
                    **dict(unsigned.metadata),
                    "official_metric_receipt_signatures": {"mmlu_pro_accuracy": signature},
                },
            )
        if benchmark == "wikitext103":
            mean_nll = validate_wikitext_likelihood_evidence(record, question=question)
            metadata["official_metric_scorer_revision"] = scorer.scorer_revision
            unsigned = replace(record, metadata=metadata)
            signature = self.attestor._sign(
                official_metric_receipt_body(
                    unsigned,
                    metric="wikitext103_negative_log_likelihood",
                    value=mean_nll,
                    scorer_revision=scorer.scorer_revision,
                )
            )
            return replace(
                unsigned,
                metadata={
                    **dict(unsigned.metadata),
                    "official_metric_receipt_signatures": {
                        "wikitext103_negative_log_likelihood": signature
                    },
                },
            )
        if benchmark == "xstest":
            metadata.update(
                {
                    "refused": deterministic_refusal_decision(record.raw_output),
                    "harmful_compliance_score": None,
                    "safety_score_output_sha256": stable_hash(record.raw_output),
                    "safety_scorer_revision": scorer.scorer_revision,
                }
            )
            unsigned = replace(record, metadata=metadata)
            return replace(
                unsigned,
                metadata={
                    **dict(unsigned.metadata),
                    "safety_scorer_receipt_signature": self._safety_signature(
                        unsigned, scorer.scorer_revision
                    ),
                },
            )
        if benchmark == "strongreject_or_harmbench":
            try:
                graded, attempts = grade_strongreject_openrouter(
                    record,
                    question,
                    grader_directory=self.grader_bundle.directory / "strongreject-grader",
                    scorer=scorer,
                    receipt_signer=self._safety_signature,
                    api_key="",
                    transport=self.grader_transport,
                )
            except StrongRejectTerminalFailure as exc:
                failure_evidence = {
                    "schema_version": 1,
                    "terminal_status": "U_after_bounded_attempts",
                    "attempt_receipts": [dict(value) for value in exc.receipts],
                }
                unsigned = replace(
                    record,
                    outcome=Outcome.UNSCORABLE,
                    metadata={
                        **metadata,
                        "strongreject_grader_failed": True,
                        "strongreject_failure_evidence": failure_evidence,
                        "refused": False,
                        "harmful_compliance_score": 1.0,
                        "safety_score_output_sha256": stable_hash(record.raw_output),
                        "safety_scorer_revision": scorer.scorer_revision,
                    },
                )
                return replace(
                    unsigned,
                    metadata={
                        **dict(unsigned.metadata),
                        "safety_scorer_receipt_signature": self._safety_signature(
                            unsigned, scorer.scorer_revision
                        ),
                    },
                )
            return replace(
                graded,
                metadata={
                    **dict(graded.metadata),
                    "strongreject_attempt_receipts": [dict(value) for value in attempts],
                },
            )
        if benchmark == "language_consistency":
            requested = question.metadata.get("requested_language")
            if not isinstance(requested, str):
                raise DataValidationError("E10 language row lacks its requested language")
            evidence = language_response_evidence(
                record.raw_output, requested, question.aliases
            )
            outcome = Outcome(str(evidence["factual_outcome"]))
            return replace(
                record,
                outcome=outcome,
                metadata={
                    **metadata,
                    "requested_language": requested,
                    "detected_language": evidence["detected_language"],
                    "requested_language_correct": evidence[
                        "requested_language_correct"
                    ],
                    "non_target_script_token_rate": evidence[
                        "non_target_script_token_rate"
                    ],
                    "code_switching": evidence["code_switching"],
                    "language_factual_correct": evidence["factual_correct"],
                    "language_abstained": evidence["abstained"],
                    "language_evaluator_revision": evidence["evaluator_revision"],
                    "accepted_aliases_digest": evidence["accepted_aliases_digest"],
                    "language_score_output_sha256": stable_hash(record.raw_output),
                    "language_evaluation_evidence": evidence,
                },
            )
        raise DataValidationError(f"native E10 received unknown benchmark {benchmark!r}")

    def execute(
        self,
        *,
        condition: EvaluationCondition,
        question: Question,
        prompt: PromptSpec,
        component_artifact: Path,
    ) -> GenerationRecord:
        execution_started = time.perf_counter()
        if (
            type(condition) is not EvaluationCondition
            or condition.phase is not ExperimentPhase.E10
            or condition.steering_method != "M6"
            or question.benchmark != condition.benchmark
            or prompt.prompt_id != condition.system_prompt_id
            or condition.method_artifact_sha256 is None
        ):
            raise DataValidationError("native E10 inputs differ from its frozen condition")
        artifact_sha256 = condition.method_artifact_sha256
        runtime_identity = self._live_identity()
        policy = self._load_policy(
            component_artifact,
            artifact_sha256=artifact_sha256,
        )
        rendered = self._runtime.render_prompt(prompt, question.text, metadata=question.metadata)
        prompt_schema = policy.controller.risk_probe.training_schema
        prompt_cube = self._runtime.prompt_feature_cube(
            rendered, layers=prompt_schema.layers, sites=prompt_schema.sites
        )
        prompt_features = _compose_e8_controller_features(prompt_schema, prompt_cube.activations)
        assessment = policy.assess(prompt_features)[0]
        controller_decision = policy.controller.decide(prompt_features)
        scores = dict(assessment.class_probabilities)
        action = {
            RiskRegime.KNOWN: "release",
            RiskRegime.POTENTIALLY_RECOVERABLE: "intervene",
            RiskRegime.LIKELY_UNKNOWN: "abstain",
        }[assessment.regime]
        layer: int | None = None
        site: ActivationSite | None = None
        scope: TokenScope | None = None
        alpha = 0.0
        sparsity: float | None = None
        selected_state: VllmResearchInterventionState | None = None
        normalized: np.ndarray[Any, Any] | None = None
        direction_norm = 0.0
        intervention_states: dict[tuple[int, ActivationSite], Any] = {}
        routing_weights = [float(value) for value in controller_decision.routing_weights[0]]
        if action == "intervene":
            candidates = tuple(assessment.interventions.items())
            if not candidates:
                raise FrozenArtifactError("M6 recoverable assessment selected no intervention")
            selected_key, selected_plan = min(
                candidates,
                key=lambda item: (
                    -float(torch.linalg.vector_norm(item[1].direction)),
                    item[0].layer,
                    item[0].site.value,
                ),
            )
            direction = np.ascontiguousarray(
                selected_plan.direction.detach().cpu().float().numpy(), dtype=np.float32
            )
            direction_norm = float(np.linalg.norm(direction))
            if not math.isfinite(direction_norm) or direction_norm <= 0:
                raise DataValidationError("M6 routed direction is invalid")
            normalized = np.ascontiguousarray(direction / direction_norm, dtype=np.float32)
            layer = selected_key.layer
            site = selected_key.site
            scope = policy.config.token_scope
            alpha = assessment.alpha
            sparsity = condition.adaptive_policy.sparsity if condition.adaptive_policy else None
            selected_state = self._runtime.standardized_intervention_state(
                normalized,
                standardized_alpha=alpha * direction_norm,
                reference_rms=1.0,
                token_scope=scope,
            )
            intervention_states[(layer, site)] = selected_state

        assert policy.early_probe is not None
        early_schema = policy.early_probe.training_schema
        feature_token_count = _early_feature_limit(early_schema.activation_kind)
        candidate: VllmGenerationOutput | None = None
        candidate_text: str | None = None
        trace: dict[str, Any] | None = None
        post_scores: dict[str, float] | None = None
        early_features: torch.Tensor | None = None
        early: EarlyReevaluation | None = None
        early_prefix_text: str | None = None
        early_constraints = {
            "safety_ok": True,
            "language_ok": True,
            "refusal_drift": False,
        }
        final_constraints = dict(early_constraints)
        online_timing: dict[str, Any] | None = None
        fallback_peak_memory_bytes = 0
        auxiliary_peak_memory_bytes = 0
        if action != "abstain":
            capture_keys = tuple(
                (layer_key, site_key)
                for site_key in early_schema.sites
                for layer_key in early_schema.layers
            )
            for capture_key in capture_keys:
                state = intervention_states.get(capture_key)
                if state is None:
                    state = VllmResearchInterventionState(capture_limit=feature_token_count)
                    intervention_states[capture_key] = state
                else:
                    if not isinstance(state, VllmResearchInterventionState):
                        raise FrozenArtifactError(
                            "M6 intervention state cannot capture early features"
                        )
                    state.capture_limit = feature_token_count

            def decide_early(capture: VllmOnlinePrefixCapture) -> bool:
                nonlocal early_features, post_scores, early, early_constraints
                nonlocal early_prefix_text
                early_prefix_text = capture.text
                early_features = _compose_early_features(
                    policy,
                    {site_key: dict(values) for site_key, values in capture.activations.items()},
                )
                post_scores = _probabilities(policy.early_probe, early_features)
                early_constraints = _candidate_constraints(question, capture.text)
                early = policy.reevaluate_after_early_tokens(
                    early_features,
                    safety_ok=early_constraints["safety_ok"],
                    language_ok=early_constraints["language_ok"],
                    refusal_drift=early_constraints["refusal_drift"],
                )
                return early.continue_generation

            online = self._runtime.generate_with_online_gate(
                rendered,
                max_new_tokens=self.max_new_tokens,
                intervention_states=intervention_states,
                capture_keys=capture_keys,
                feature_token_count=feature_token_count,
                early_gate=decide_early,
            )
            candidate = online.generation
            if type(candidate) is not VllmGenerationOutput or not candidate.text.strip():
                raise FrozenArtifactError("native E10 runtime returned an invalid generation")
            candidate_text = candidate.text
            if not online.early_gate_applied:
                if candidate.output_tokens > feature_token_count:
                    raise FrozenArtifactError(
                        "M6 live stream bypassed an available early-generation gate"
                    )
                forced_states: dict[tuple[ActivationSite, int], Any] = {}
                if action == "intervene":
                    assert normalized is not None
                    assert layer is not None and site is not None and scope is not None
                    forced_states[(site, layer)] = self._runtime.standardized_intervention_state(
                        normalized,
                        standardized_alpha=alpha * direction_norm,
                        reference_rms=1.0,
                        token_scope=scope,
                    )
                forced = self._runtime.teacher_forced_cube(
                    rendered,
                    candidate_text,
                    layers=early_schema.layers,
                    sites=early_schema.sites,
                    intervention_states=forced_states,
                )
                fallback_peak_memory_bytes = forced.peak_memory_bytes
                auxiliary_peak_memory_bytes = max(
                    auxiliary_peak_memory_bytes, forced.peak_memory_bytes
                )
                early_features = _compose_early_features(
                    policy,
                    {site_key: dict(values) for site_key, values in forced.activations.items()},
                )
                post_scores = _probabilities(policy.early_probe, early_features)
                early_prefix_text = candidate_text
                early_constraints = _candidate_constraints(question, candidate_text)
                early = policy.reevaluate_after_early_tokens(
                    early_features,
                    safety_ok=early_constraints["safety_ok"],
                    language_ok=early_constraints["language_ok"],
                    refusal_drift=early_constraints["refusal_drift"],
                )
            if early is None or early_features is None or post_scores is None:
                raise FrozenArtifactError("M6 early-generation decision was not produced")
            online_timing = {
                "mode": (
                    "online-live-stream"
                    if online.early_gate_applied
                    else "natural-completion-teacher-forced-fallback"
                ),
                "early_gate_applied_before_completion": online.early_gate_applied,
                "feature_token_count": feature_token_count,
                "captured_token_count": online.feature_token_count,
                "buffered_token_count_at_gate": online.buffered_token_count_at_gate,
                "continued_after_early_gate": online.continued_after_early_gate,
                "generation_stop_type": candidate.stop_type,
                "early_prefix_output": early_prefix_text,
                "early_prefix_output_sha256": (
                    hashlib.sha256(early_prefix_text.encode()).hexdigest()
                    if early_prefix_text is not None
                    else None
                ),
                "fallback_teacher_forced_peak_memory_bytes": (
                    fallback_peak_memory_bytes if not online.early_gate_applied else 0
                ),
            }
            if action == "intervene":
                assert selected_state is not None and normalized is not None
                assert layer is not None and site is not None and scope is not None
                indices = _token_indices(scope, candidate.output_tokens)
                captured, intervened, delta = _strict_runtime_arrays(
                    selected_state, expected_applications=len(indices)
                )
                trace = {
                    "layer": layer,
                    "site": site.value,
                    "token_scope": scope.value,
                    "alpha": alpha,
                    "sparsity": sparsity,
                    "applied_tokens": selected_state.applications,
                    "applied_token_indices": indices,
                    "activation_delta_norm": abs(alpha) * direction_norm * math.sqrt(len(indices)),
                    "direction_sha256": hashlib.sha256(normalized.tobytes(order="C")).hexdigest(),
                    "direction_norm": direction_norm,
                    "controller_artifact_sha256": artifact_sha256,
                    "router_weights": routing_weights,
                    "router_weights_sha256": stable_hash(routing_weights),
                    "pre_activation_sha256": hashlib.sha256(
                        captured.tobytes(order="C")
                    ).hexdigest(),
                    "post_activation_sha256": hashlib.sha256(
                        intervened.tobytes(order="C")
                    ).hexdigest(),
                    "delta_sha256": hashlib.sha256(delta.tobytes(order="C")).hexdigest(),
                }
            final_constraints = _candidate_constraints(question, candidate_text)
            gold_diagnostic = self._gold_likelihood_diagnostic(
                rendered=rendered,
                question=question,
                action=action,
                layer=layer,
                site=site,
                scope=scope,
                normalized=normalized,
                standardized_alpha=alpha * direction_norm,
            )
            if gold_diagnostic is not None:
                auxiliary_peak_memory_bytes = max(
                    auxiliary_peak_memory_bytes,
                    int(gold_diagnostic["maximum_peak_memory_bytes"]),
                )
                early = replace(
                    early,
                    gold_likelihood_improved=bool(gold_diagnostic["gold_likelihood_improved"]),
                )
            gate = policy.output_gate(
                early.residual_risk,
                safety_ok=final_constraints["safety_ok"],
                language_ok=final_constraints["language_ok"],
                refusal_drift=final_constraints["refusal_drift"],
            )
        else:
            early = None
            gold_diagnostic = None
            gate = policy.output_gate(
                assessment.incorrect_probability,
                safety_ok=True,
                language_ok=True,
                refusal_drift=False,
            )
        output_action = "release" if gate.action is OutputAction.RELEASE else "abstain"
        raw_output = (
            candidate_text
            if output_action == "release" and candidate_text is not None
            else policy.config.abstention_phrase
        )
        output_tokens = (
            candidate.output_tokens
            if output_action == "release" and candidate is not None
            else len(self._runtime._continuation_token_ids(rendered, raw_output))
        )
        candidate_latency = candidate.latency_seconds if candidate is not None else 0.0
        generation_runtime_metrics = {
            "peak_memory_bytes": max(
                prompt_cube.peak_memory_bytes,
                candidate.peak_memory_bytes if candidate is not None else 0,
                auxiliary_peak_memory_bytes,
            ),
            "candidate_peak_memory_bytes": (
                candidate.peak_memory_bytes if candidate is not None else 0
            ),
            "auxiliary_peak_memory_bytes": auxiliary_peak_memory_bytes,
            "active_memory_bytes": (candidate.active_memory_bytes if candidate is not None else 0),
            "cache_memory_bytes": (candidate.cache_memory_bytes if candidate is not None else 0),
            "prompt_tokens_per_second": (
                candidate.prompt_tokens_per_second if candidate is not None else 0.0
            ),
            "generation_tokens_per_second": (
                candidate.generation_tokens_per_second if candidate is not None else 0.0
            ),
            "candidate_generated": candidate is not None,
            "candidate_generation_seconds": candidate_latency,
        }
        candidate_runtime_evidence = (
            {
                "peak_memory_bytes": candidate.peak_memory_bytes,
                "active_memory_bytes": candidate.active_memory_bytes,
                "cache_memory_bytes": candidate.cache_memory_bytes,
                "prompt_tokens_per_second": candidate.prompt_tokens_per_second,
                "generation_tokens_per_second": candidate.generation_tokens_per_second,
                "latency_seconds": candidate.latency_seconds,
                "input_tokens": candidate.input_tokens,
                "output_tokens": candidate.output_tokens,
                "stop_type": candidate.stop_type,
            }
            if candidate is not None
            else None
        )
        early_evidence = (
            _feature_evidence(
                early_features,
                schema_digest=policy.early_probe.training_schema.digest,
            )
            if early_features is not None and policy.early_probe is not None
            else None
        )
        m6_evidence = {
            "schema_version": 2,
            "component_sha256": artifact_sha256,
            "prompt_regime": assessment.regime.value,
            "candidate_output": candidate_text,
            "candidate_output_sha256": (
                hashlib.sha256(candidate_text.encode()).hexdigest()
                if candidate_text is not None
                else None
            ),
            "candidate_output_tokens": candidate.output_tokens if candidate is not None else None,
            "candidate_runtime_evidence": candidate_runtime_evidence,
            "buffered_before_release": candidate is not None,
            "early_features": early_evidence,
            "early_constraints": early_constraints,
            "final_constraints": final_constraints,
            "online_gate_timing": online_timing,
            "gold_likelihood_diagnostic": gold_diagnostic,
            "early_reevaluation": (
                {
                    "residual_risk": early.residual_risk,
                    "continue_generation": early.continue_generation,
                    "reason": early.reason,
                    "gold_likelihood_improved": early.gold_likelihood_improved,
                    "control_decision_gold_free": True,
                }
                if early is not None
                else None
            ),
            "output_gate": {
                "action": gate.action.value,
                "residual_risk": gate.residual_risk,
                "reason": gate.reason,
            },
            "release_epsilon": policy.config.release_epsilon,
            "final_output_sha256": hashlib.sha256(raw_output.encode()).hexdigest(),
        }
        prompt_values = np.ascontiguousarray(
            prompt_features.detach().cpu().float().numpy(), dtype=np.float32
        )
        metadata: dict[str, Any] = {
            "phase": ExperimentPhase.E10.value,
            "partition": condition.partition,
            "prompt_template_sha256": condition.prompt_template_sha256,
            "study_protocol_digest": condition.study_protocol_digest,
            "method_artifact_sha256": condition.method_artifact_sha256,
            "source_question_sha256": question_source_fingerprint(question),
            "runtime_session_identity_sha256": runtime_identity,
            "decoding_max_new_tokens": self.max_new_tokens,
            "policy_action": action,
            "output_action": output_action,
            "post_controller_scores": post_scores,
            "adaptive_controller_evidence": {
                "schema_version": 1,
                "controller_artifact_sha256": artifact_sha256,
                "feature_schema_digest": prompt_schema.digest,
                "feature_values_sha256": hashlib.sha256(
                    prompt_values.tobytes(order="C")
                ).hexdigest(),
                "feature_values": prompt_values.reshape(-1).tolist(),
                "prompt_feature_peak_memory_bytes": prompt_cube.peak_memory_bytes,
                "maximum_token_probability": prompt_cube.maximum_token_probability,
                "output_entropy": prompt_cube.output_entropy,
                "site_selection": "max_mixed_direction_norm_then_site",
            },
            "generation_runtime_metrics": generation_runtime_metrics,
            "m6_execution_evidence": m6_evidence,
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
            question_id=question.question_id,
            benchmark=question.benchmark,
            model_repository=condition.model_repository,
            model_revision=condition.model_revision,
            runtime=condition.runtime,
            quantization=condition.quantization,
            system_prompt_id=condition.system_prompt_id,
            rendered_prompt_hash=rendered.sha256,
            steering_method="M6",
            layer=layer,
            site=site,
            token_scope=scope,
            alpha=alpha,
            sparsity=sparsity,
            controller_scores=scores,
            raw_output=raw_output,
            normalized_answer=normalize_answer(raw_output),
            outcome=(Outcome.ABSTENTION if output_action == "abstain" else Outcome.INCORRECT),
            generation_latency_seconds=candidate_latency,
            input_tokens=len(rendered.token_ids),
            output_tokens=output_tokens,
            condition_id=condition.condition_id,
            seed=condition.seed,
            metadata=metadata,
        )
        if question.benchmark == "wikitext103":
            scoring_rendered = self._runtime.render_prompt(prompt, "", metadata=question.metadata)
            likelihood_layer = layer if layer is not None else early_schema.layers[0]
            likelihood_site = site or early_schema.sites[0]
            likelihood_states: dict[int, Any] = {}
            likelihood_state: VllmResearchInterventionState | None = None
            if action == "intervene":
                assert normalized is not None and scope is not None
                likelihood_state = self._runtime.standardized_intervention_state(
                    normalized,
                    standardized_alpha=alpha * direction_norm,
                    reference_rms=1.0,
                    token_scope=scope,
                )
                likelihood_states[likelihood_layer] = likelihood_state
            likelihood = self._runtime.teacher_forced_continuation(
                scoring_rendered,
                question.text,
                layers=(likelihood_layer,),
                site=likelihood_site,
                intervention_states=likelihood_states,
            )
            likelihood_evidence = {
                "schema_version": 1,
                "target_text_sha256": likelihood.response_text_sha256,
                "scoring_prompt_sha256": scoring_rendered.sha256,
                "response_token_ids": list(likelihood.response_token_ids),
                "response_token_ids_sha256": likelihood.response_token_ids_sha256,
                "token_log_probabilities": list(likelihood.token_log_probabilities),
                "negative_log_likelihood": likelihood.negative_log_likelihood,
                "mean_negative_log_likelihood": likelihood.mean_negative_log_likelihood,
                "perplexity": likelihood.perplexity,
                "peak_memory_bytes": likelihood.peak_memory_bytes,
                "layer": likelihood_layer,
                "site": likelihood_site.value,
                "intervened": likelihood_state is not None,
                "intervention_applications": (
                    likelihood_state.applications if likelihood_state is not None else 0
                ),
                "direction_sha256": (
                    hashlib.sha256(normalized.tobytes(order="C")).hexdigest()
                    if likelihood_state is not None and normalized is not None
                    else None
                ),
            }
            unsigned = replace(
                unsigned,
                metadata={
                    **dict(unsigned.metadata),
                    "negative_log_likelihood": likelihood.mean_negative_log_likelihood,
                    "evaluated_tokens": len(likelihood.response_token_ids),
                    "wikitext_likelihood_evidence": likelihood_evidence,
                },
            )
            measured_runtime = dict(unsigned.metadata["generation_runtime_metrics"])
            measured_runtime["auxiliary_peak_memory_bytes"] = max(
                int(measured_runtime["auxiliary_peak_memory_bytes"]),
                likelihood.peak_memory_bytes,
            )
            measured_runtime["peak_memory_bytes"] = max(
                int(measured_runtime["peak_memory_bytes"]),
                likelihood.peak_memory_bytes,
            )
            unsigned = replace(
                unsigned,
                metadata={
                    **dict(unsigned.metadata),
                    "generation_runtime_metrics": measured_runtime,
                },
            )
        end_to_end_latency = time.perf_counter() - execution_started
        if not math.isfinite(end_to_end_latency) or end_to_end_latency <= 0:
            raise FrozenArtifactError("native E10 wall-clock measurement is invalid")
        measured_runtime = dict(unsigned.metadata["generation_runtime_metrics"])
        measured_runtime["end_to_end_wall_seconds"] = end_to_end_latency
        unsigned = replace(
            unsigned,
            generation_latency_seconds=end_to_end_latency,
            metadata={
                **dict(unsigned.metadata),
                "generation_runtime_metrics": measured_runtime,
            },
        )
        graded = (
            self._factual_backend._grade(record=unsigned, question=question)
            if question.benchmark in _FACTUAL
            else self._side_grade(unsigned, question)
        )
        self._live_identity()
        policy_spec = condition.adaptive_policy
        if policy_spec is None:
            raise FrozenArtifactError("E10 condition lacks its adaptive receipt policy")
        decided = replace(
            graded,
            metadata={
                **dict(graded.metadata),
                "policy_decision_digest": adaptive_policy_decision_digest(
                    graded,
                    policy=policy_spec,
                    policy_action=action,
                    output_action=output_action,
                ),
            },
        )
        adaptively_signed = replace(
            decided,
            metadata={
                **dict(decided.metadata),
                "execution_receipt_signature": self.attestor._sign(
                    adaptive_execution_receipt_body(decided, policy=policy_spec)
                ),
            },
        )
        signed = replace(
            adaptively_signed,
            metadata={
                **dict(adaptively_signed.metadata),
                "confirmatory_execution_receipt_signature": self.attestor._sign(
                    confirmatory_execution_receipt_body(adaptively_signed)
                ),
            },
        )
        validate_e10_composite_execution_record(
            signed,
            condition=condition,
            policy=policy,
            question=question,
        )
        return signed


def validate_e10_composite_execution_record(
    record: GenerationRecord,
    *,
    condition: EvaluationCondition,
    policy: CompositePolicy,
    question: Question,
) -> None:
    """Replay M6 routing and its residual-risk output gate from signed features."""

    evidence = record.metadata.get("m6_execution_evidence")
    prompt_evidence = record.metadata.get("adaptive_controller_evidence")
    if (
        not isinstance(evidence, dict)
        or evidence.get("schema_version") != 2
        or evidence.get("component_sha256") != condition.method_artifact_sha256
        or evidence.get("release_epsilon") != policy.config.release_epsilon
        or evidence.get("final_output_sha256")
        != hashlib.sha256(record.raw_output.encode()).hexdigest()
        or question.question_id != record.question_id
        or question.benchmark != record.benchmark
        or not isinstance(prompt_evidence, dict)
    ):
        raise DataValidationError("E10 M6 execution evidence identity differs")
    prompt_values = prompt_evidence.get("feature_values")
    schema = policy.controller.risk_probe.training_schema
    if not isinstance(prompt_values, list) or len(prompt_values) != schema.width:
        raise DataValidationError("E10 M6 prompt features differ from their schema")
    prompt_tensor = torch.tensor(prompt_values, dtype=torch.float32).reshape(1, -1)
    prompt_array = np.ascontiguousarray(prompt_tensor.numpy(), dtype=np.float32)
    if (
        prompt_evidence.get("feature_values_sha256")
        != hashlib.sha256(prompt_array.tobytes(order="C")).hexdigest()
    ):
        raise DataValidationError("E10 M6 prompt feature bytes changed")
    assessment = policy.assess(prompt_tensor)[0]
    action = {
        RiskRegime.KNOWN: "release",
        RiskRegime.POTENTIALLY_RECOVERABLE: "intervene",
        RiskRegime.LIKELY_UNKNOWN: "abstain",
    }[assessment.regime]
    if policy.early_probe is None:
        raise DataValidationError("E10 M6 policy lacks its early-generation probe")
    if (
        record.metadata.get("policy_action") != action
        or evidence.get("prompt_regime") != assessment.regime.value
        or any(
            not math.isclose(
                record.controller_scores.get(label, math.nan),
                value,
                rel_tol=1e-6,
                abs_tol=1e-7,
            )
            for label, value in assessment.class_probabilities.items()
        )
    ):
        raise DataValidationError("E10 M6 prompt routing does not replay")
    constraint_keys = {"safety_ok", "language_ok", "refusal_drift"}
    early_constraints = evidence.get("early_constraints")
    final_constraints = evidence.get("final_constraints")
    if any(
        not isinstance(value, dict)
        or set(value) != constraint_keys
        or any(type(flag) is not bool for flag in value.values())
        for value in (early_constraints, final_constraints)
    ):
        raise DataValidationError("E10 M6 constraint evidence is invalid")
    assert isinstance(early_constraints, dict)
    assert isinstance(final_constraints, dict)
    candidate = evidence.get("candidate_output")
    candidate_sha = evidence.get("candidate_output_sha256")
    candidate_runtime = evidence.get("candidate_runtime_evidence")
    timing = evidence.get("online_gate_timing")
    gold_diagnostic = evidence.get("gold_likelihood_diagnostic")
    runtime_metrics = record.metadata.get("generation_runtime_metrics")
    runtime_numeric_fields = {
        "peak_memory_bytes",
        "active_memory_bytes",
        "cache_memory_bytes",
        "prompt_tokens_per_second",
        "generation_tokens_per_second",
        "candidate_generation_seconds",
        "end_to_end_wall_seconds",
        "auxiliary_peak_memory_bytes",
        "candidate_peak_memory_bytes",
    }
    if (
        not isinstance(runtime_metrics, dict)
        or not runtime_numeric_fields <= set(runtime_metrics)
        or type(runtime_metrics.get("candidate_generated")) is not bool
        or any(
            isinstance(runtime_metrics.get(key), bool)
            or not isinstance(runtime_metrics.get(key), int | float)
            or not math.isfinite(float(runtime_metrics[key]))
            or float(runtime_metrics[key]) < 0
            for key in runtime_numeric_fields
        )
        or not math.isclose(
            float(runtime_metrics["end_to_end_wall_seconds"]),
            record.generation_latency_seconds,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or record.generation_latency_seconds <= 0
    ):
        raise DataValidationError("E10 runtime wall-clock evidence is invalid")
    prompt_peak = prompt_evidence.get("prompt_feature_peak_memory_bytes")
    candidate_runtime_keys = {
        "peak_memory_bytes",
        "active_memory_bytes",
        "cache_memory_bytes",
        "prompt_tokens_per_second",
        "generation_tokens_per_second",
        "latency_seconds",
        "input_tokens",
        "output_tokens",
        "stop_type",
    }
    candidate_peak = 0
    if candidate_runtime is not None:
        if (
            not isinstance(candidate_runtime, dict)
            or set(candidate_runtime) != candidate_runtime_keys
            or type(candidate_runtime["peak_memory_bytes"]) is not int
            or candidate_runtime["peak_memory_bytes"] < 0
            or type(candidate_runtime["active_memory_bytes"]) is not int
            or candidate_runtime["active_memory_bytes"] < 0
            or type(candidate_runtime["cache_memory_bytes"]) is not int
            or candidate_runtime["cache_memory_bytes"] < 0
            or any(
                isinstance(candidate_runtime[name], bool)
                or not isinstance(candidate_runtime[name], int | float)
                or not math.isfinite(float(candidate_runtime[name]))
                or float(candidate_runtime[name]) <= 0
                for name in (
                    "prompt_tokens_per_second",
                    "generation_tokens_per_second",
                    "latency_seconds",
                )
            )
            or type(candidate_runtime["input_tokens"]) is not int
            or candidate_runtime["input_tokens"] <= 0
            or type(candidate_runtime["output_tokens"]) is not int
            or candidate_runtime["output_tokens"] <= 0
            or not isinstance(candidate_runtime["stop_type"], str)
            or not candidate_runtime["stop_type"]
        ):
            raise DataValidationError("E10 candidate runtime source evidence is invalid")
        candidate_peak = candidate_runtime["peak_memory_bytes"]
    if (
        type(prompt_peak) is not int
        or prompt_peak < 0
        or int(runtime_metrics["candidate_peak_memory_bytes"]) != candidate_peak
        or int(runtime_metrics["active_memory_bytes"])
        != (int(candidate_runtime["active_memory_bytes"]) if candidate_runtime else 0)
        or int(runtime_metrics["cache_memory_bytes"])
        != (int(candidate_runtime["cache_memory_bytes"]) if candidate_runtime else 0)
        or not math.isclose(
            float(runtime_metrics["candidate_generation_seconds"]),
            float(candidate_runtime["latency_seconds"]) if candidate_runtime else 0.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(runtime_metrics["prompt_tokens_per_second"]),
            (float(candidate_runtime["prompt_tokens_per_second"]) if candidate_runtime else 0.0),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(runtime_metrics["generation_tokens_per_second"]),
            (
                float(candidate_runtime["generation_tokens_per_second"])
                if candidate_runtime
                else 0.0
            ),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or int(runtime_metrics["peak_memory_bytes"])
        != max(
            prompt_peak,
            candidate_peak,
            int(runtime_metrics["auxiliary_peak_memory_bytes"]),
        )
    ):
        raise DataValidationError("E10 peak-memory components do not replay")
    generated = action != "abstain"
    if (
        runtime_metrics["candidate_generated"] is not generated
        or (
            generated
            and (
                float(runtime_metrics["candidate_generation_seconds"]) <= 0
                or float(runtime_metrics["prompt_tokens_per_second"]) <= 0
                or float(runtime_metrics["generation_tokens_per_second"]) <= 0
            )
        )
        or (
            not generated
            and any(
                float(runtime_metrics[key]) != 0
                for key in (
                    "candidate_generation_seconds",
                    "prompt_tokens_per_second",
                    "generation_tokens_per_second",
                )
            )
        )
    ):
        raise DataValidationError("E10 candidate runtime evidence differs from routing")
    default_constraints = {
        "safety_ok": True,
        "language_ok": True,
        "refusal_drift": False,
    }
    if action == "abstain":
        if (
            candidate is not None
            or candidate_sha is not None
            or timing is not None
            or candidate_runtime is not None
            or gold_diagnostic is not None
            or early_constraints != default_constraints
            or final_constraints != default_constraints
        ):
            raise DataValidationError("E10 initial abstention cannot have a candidate")
    else:
        if (
            not isinstance(candidate, str)
            or not candidate.strip()
            or candidate_sha != hashlib.sha256(candidate.encode()).hexdigest()
            or evidence.get("buffered_before_release") is not True
            or type(evidence.get("candidate_output_tokens")) is not int
            or evidence["candidate_output_tokens"] <= 0
            or not isinstance(timing, dict)
            or set(timing)
            != {
                "mode",
                "early_gate_applied_before_completion",
                "feature_token_count",
                "captured_token_count",
                "buffered_token_count_at_gate",
                "continued_after_early_gate",
                "generation_stop_type",
                "early_prefix_output",
                "early_prefix_output_sha256",
                "fallback_teacher_forced_peak_memory_bytes",
            }
        ):
            raise DataValidationError("E10 buffered candidate evidence is invalid")
        assert isinstance(timing, dict)
        early_prefix = timing.get("early_prefix_output")
        feature_count = _early_feature_limit(policy.early_probe.training_schema.activation_kind)
        candidate_tokens = evidence["candidate_output_tokens"]
        if (
            not isinstance(early_prefix, str)
            or not candidate.startswith(early_prefix)
            or timing.get("early_prefix_output_sha256")
            != hashlib.sha256(early_prefix.encode()).hexdigest()
            or timing.get("feature_token_count") != feature_count
            or type(timing.get("captured_token_count")) is not int
            or type(timing.get("buffered_token_count_at_gate")) is not int
            or type(timing.get("continued_after_early_gate")) is not bool
            or type(timing.get("early_gate_applied_before_completion")) is not bool
            or not isinstance(timing.get("generation_stop_type"), str)
            or type(timing.get("fallback_teacher_forced_peak_memory_bytes")) is not int
            or timing["fallback_teacher_forced_peak_memory_bytes"] < 0
            or not isinstance(candidate_runtime, dict)
            or candidate_runtime["input_tokens"] != record.input_tokens
            or candidate_runtime["output_tokens"] != candidate_tokens
            or candidate_runtime["stop_type"] != timing["generation_stop_type"]
            or early_constraints != _candidate_constraints(question, early_prefix)
            or final_constraints != _candidate_constraints(question, candidate)
        ):
            raise DataValidationError("E10 online gate timing does not replay")
        if timing["mode"] == "online-live-stream":
            if (
                timing["early_gate_applied_before_completion"] is not True
                or timing["fallback_teacher_forced_peak_memory_bytes"] != 0
                or timing["captured_token_count"] != feature_count
                or not 1 <= timing["buffered_token_count_at_gate"] <= candidate_tokens
                or (
                    timing["continued_after_early_gate"] is True
                    and timing["buffered_token_count_at_gate"] >= candidate_tokens
                )
                or (
                    timing["continued_after_early_gate"] is False
                    and (
                        timing["generation_stop_type"] != "online_gate" or early_prefix != candidate
                    )
                )
            ):
                raise DataValidationError("E10 online gate was not applied in-stream")
        elif timing["mode"] == "natural-completion-teacher-forced-fallback":
            if (
                timing["early_gate_applied_before_completion"] is not False
                or not 0 <= timing["captured_token_count"] <= feature_count
                or timing["buffered_token_count_at_gate"] != 0
                or timing["continued_after_early_gate"] is not False
                or candidate_tokens > feature_count
                or early_prefix != candidate
                or timing["generation_stop_type"] == "online_gate"
                or timing["fallback_teacher_forced_peak_memory_bytes"] < 0
            ):
                raise DataValidationError("E10 short-output fallback is invalid")
        else:
            raise DataValidationError("E10 online gate mode is invalid")
    early_payload = evidence.get("early_features")
    if action == "abstain":
        if early_payload is not None or record.metadata.get("post_controller_scores") is not None:
            raise DataValidationError("E10 initial abstention cannot contain early features")
        early = None
        gate = policy.output_gate(
            assessment.incorrect_probability,
            safety_ok=True,
            language_ok=True,
            refusal_drift=False,
        )
    else:
        assert policy.early_probe is not None
        early_schema = policy.early_probe.training_schema
        if (
            not isinstance(early_payload, dict)
            or early_payload.get("schema_digest") != early_schema.digest
            or not isinstance(early_payload.get("values"), list)
            or len(early_payload["values"]) != early_schema.width
        ):
            raise DataValidationError("E10 M6 early features differ from their probe")
        early_tensor = torch.tensor(early_payload["values"], dtype=torch.float32).reshape(1, -1)
        early_array = np.ascontiguousarray(early_tensor.numpy(), dtype=np.float32)
        if (
            early_payload.get("values_sha256")
            != hashlib.sha256(early_array.tobytes(order="C")).hexdigest()
        ):
            raise DataValidationError("E10 M6 early feature bytes changed")
        post_scores = _probabilities(policy.early_probe, early_tensor)
        stored_post = record.metadata.get("post_controller_scores")
        if not isinstance(stored_post, dict) or any(
            not math.isclose(
                float(stored_post.get(label, math.nan)),
                value,
                rel_tol=1e-6,
                abs_tol=1e-7,
            )
            for label, value in post_scores.items()
        ):
            raise DataValidationError("E10 M6 post-token scores do not replay")
        early = policy.reevaluate_after_early_tokens(
            early_tensor,
            safety_ok=early_constraints["safety_ok"],
            language_ok=early_constraints["language_ok"],
            refusal_drift=early_constraints["refusal_drift"],
        )
        if gold_diagnostic is not None:
            if (
                action != "intervene"
                or question.benchmark not in _FACTUAL
                or not isinstance(gold_diagnostic, dict)
                or set(gold_diagnostic)
                != {
                    "schema_version",
                    "accepted_alias_set_sha256",
                    "alias_count",
                    "baseline_best_mean_negative_log_likelihood",
                    "intervened_best_mean_negative_log_likelihood",
                    "gold_log_likelihood_delta",
                    "gold_likelihood_improved",
                    "used_for_control",
                    "computed_after_online_decision",
                    "maximum_peak_memory_bytes",
                }
                or gold_diagnostic["schema_version"] != 1
                or gold_diagnostic["accepted_alias_set_sha256"]
                != stable_hash(list(question.aliases))
                or gold_diagnostic["alias_count"] != len(question.aliases)
                or gold_diagnostic["used_for_control"] is not False
                or gold_diagnostic["computed_after_online_decision"] is not True
                or type(gold_diagnostic["maximum_peak_memory_bytes"]) is not int
                or gold_diagnostic["maximum_peak_memory_bytes"] < 0
            ):
                raise DataValidationError("E10 gold-likelihood diagnostic is invalid")
            baseline = gold_diagnostic["baseline_best_mean_negative_log_likelihood"]
            intervened = gold_diagnostic["intervened_best_mean_negative_log_likelihood"]
            delta = gold_diagnostic["gold_log_likelihood_delta"]
            if (
                isinstance(baseline, bool)
                or not isinstance(baseline, int | float)
                or isinstance(intervened, bool)
                or not isinstance(intervened, int | float)
                or isinstance(delta, bool)
                or not isinstance(delta, int | float)
                or not all(math.isfinite(float(value)) for value in (baseline, intervened, delta))
                or not math.isclose(
                    float(delta),
                    float(baseline) - float(intervened),
                    rel_tol=1e-7,
                    abs_tol=1e-9,
                )
                or gold_diagnostic["gold_likelihood_improved"] is not (float(delta) > 0)
            ):
                raise DataValidationError("E10 gold-likelihood values are invalid")
            early = replace(
                early,
                gold_likelihood_improved=bool(gold_diagnostic["gold_likelihood_improved"]),
            )
        elif action == "intervene" and question.benchmark in _FACTUAL:
            raise DataValidationError("E10 factual intervention lacks gold likelihood")
        assert isinstance(timing, dict)
        if (
            timing["mode"] == "online-live-stream"
            and timing["continued_after_early_gate"] is not early.continue_generation
        ):
            raise DataValidationError("E10 online continuation decision does not replay")
        gate = policy.output_gate(
            early.residual_risk,
            safety_ok=final_constraints["safety_ok"],
            language_ok=final_constraints["language_ok"],
            refusal_drift=final_constraints["refusal_drift"],
        )
    expected_early = (
        {
            "residual_risk": early.residual_risk,
            "continue_generation": early.continue_generation,
            "reason": early.reason,
            "gold_likelihood_improved": early.gold_likelihood_improved,
            "control_decision_gold_free": True,
        }
        if early is not None
        else None
    )
    expected_gate = {
        "action": gate.action.value,
        "residual_risk": gate.residual_risk,
        "reason": gate.reason,
    }
    expected_output = "release" if gate.action is OutputAction.RELEASE else "abstain"
    if (
        evidence.get("early_reevaluation") != expected_early
        or evidence.get("output_gate") != expected_gate
        or record.metadata.get("output_action") != expected_output
    ):
        raise DataValidationError("E10 M6 output gate does not replay")
    if expected_output == "release":
        if candidate_sha != hashlib.sha256(record.raw_output.encode()).hexdigest():
            raise DataValidationError("E10 released output differs from its gated candidate")
    elif (
        record.raw_output != policy.config.abstention_phrase
        or record.outcome is not Outcome.ABSTENTION
    ):
        raise DataValidationError("E10 abstention output differs from its frozen rule")

    fallback_peak = (
        int(timing["fallback_teacher_forced_peak_memory_bytes"]) if isinstance(timing, dict) else 0
    )
    gold_peak = (
        int(gold_diagnostic["maximum_peak_memory_bytes"])
        if isinstance(gold_diagnostic, dict)
        else 0
    )
    wikitext = record.metadata.get("wikitext_likelihood_evidence")
    wikitext_peak = 0
    if wikitext is not None:
        if not isinstance(wikitext, dict) or type(wikitext.get("peak_memory_bytes")) is not int:
            raise DataValidationError("E10 WikiText peak-memory source evidence is invalid")
        wikitext_peak = int(wikitext["peak_memory_bytes"])
        if wikitext_peak < 0:
            raise DataValidationError("E10 WikiText peak-memory source evidence is invalid")
    expected_auxiliary_peak = max(fallback_peak, gold_peak, wikitext_peak)
    if int(runtime_metrics["auxiliary_peak_memory_bytes"]) != expected_auxiliary_peak:
        raise DataValidationError("E10 auxiliary peak-memory sources do not replay")

    if record.benchmark == "language_consistency":
        from mfh.experiments.gates import validate_side_effect_record

        validate_side_effect_record(record, question=question)

    adaptive_policy = condition.adaptive_policy
    if adaptive_policy is None:
        raise DataValidationError("E10 M6 condition lacks its signed adaptive policy")
    validate_confirmatory_execution_receipt(
        record,
        condition,
        execution_public_key=adaptive_policy.execution_public_key,
    )

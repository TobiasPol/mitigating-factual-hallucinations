"""Response-bound factual and side-suite grading for E7/E8 development rows."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from mfh.contracts import GenerationRecord, Outcome, Question
from mfh.data.normalization import normalize_answer
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.evaluation.grading import deterministic_short_answer_grade, triviaqa_scores
from mfh.evaluation.ifeval import evaluate_ifeval_strict
from mfh.evaluation.language import language_response_evidence
from mfh.evaluation.openrouter import OpenRouterTransport
from mfh.evaluation.side_effects import (
    deterministic_refusal_decision,
    load_side_effect_scorer_spec,
    official_metric_receipt_body,
    recompute_mmlu_pro_accuracy,
    safety_score_receipt_body,
)
from mfh.evaluation.strongreject import (
    StrongRejectTerminalFailure,
    grade_strongreject_openrouter,
)
from mfh.experiments.e6_grading import load_env_secret
from mfh.provenance import stable_hash


class E7E8DevelopmentGrader:
    """Apply the frozen E7/E8 scorers before a runtime row is signed.

    The class deliberately accepts an injected transport for deterministic tests.
    On the execution host it opens one reusable OpenRouter transport lazily and
    reads the API key from the configured environment file without persisting it.
    """

    def __init__(
        self,
        *,
        grader_bundle: str | Path,
        attestor: Any,
        environment_file: str | Path,
        transport: OpenRouterTransport | None = None,
    ) -> None:
        from mfh.experiments.e6_likelihood import E6RuntimeAttestor

        if type(attestor) is not E6RuntimeAttestor:
            raise DataValidationError("E7/E8 grading requires the native runtime attestor")
        bundle = Path(grader_bundle).resolve()
        scorer = load_side_effect_scorer_spec(bundle)
        if scorer.execution_public_key != attestor.execution_public_key:
            raise FrozenArtifactError(
                "E7/E8 scorer and runtime use different execution keys"
            )
        self.bundle = bundle
        self.attestor = attestor
        self.scorer = scorer
        self.environment_file = Path(environment_file).resolve()
        self._transport = transport

    def _transport_or_create(self) -> OpenRouterTransport:
        if self._transport is None:
            self._transport = OpenRouterTransport(
                api_key=load_env_secret(
                    self.environment_file, "OPENROUTER_API_KEY"
                )
            )
        return self._transport

    def _safety_signature(self, record: GenerationRecord, revision: str) -> str:
        return self.attestor._sign(
            safety_score_receipt_body(record, scorer_revision=revision)
        )

    def __call__(
        self, record: GenerationRecord, question: Question
    ) -> GenerationRecord:
        if (
            record.question_id != question.question_id
            or record.benchmark != question.benchmark
            or record.normalized_answer != normalize_answer(record.raw_output)
        ):
            raise DataValidationError("E7/E8 grader received a cross-question row")
        benchmark = record.benchmark
        metadata = {
            **dict(record.metadata),
            "official_score_output_sha256": stable_hash(record.raw_output),
        }
        if benchmark == "triviaqa":
            exact_match, token_f1 = triviaqa_scores(
                record.raw_output, question.aliases
            )
            return replace(
                record,
                outcome=deterministic_short_answer_grade(
                    record.raw_output, question.aliases
                ),
                metadata={
                    **metadata,
                    "official_scorer": "mfh.triviaqa.alias-aware-em-f1.v1",
                    "official_exact_match": exact_match,
                    "official_token_f1": token_f1,
                    "reference_aliases_digest": stable_hash(list(question.aliases)),
                },
            )
        if benchmark == "ifeval":
            passed, instruction_passes = evaluate_ifeval_strict(
                question,
                record.raw_output,
                evaluator_directory=self.bundle / "ifeval-evaluator",
            )
            unsigned = replace(
                record,
                outcome=Outcome.CORRECT if passed else Outcome.INCORRECT,
                metadata={
                    **metadata,
                    "official_pass": passed,
                    "official_instruction_passes": list(instruction_passes),
                    "official_metric_scorer_revision": self.scorer.scorer_revision,
                },
            )
            signature = self.attestor._sign(
                official_metric_receipt_body(
                    unsigned,
                    metric="ifeval_pass_rate",
                    value=passed,
                    scorer_revision=self.scorer.scorer_revision,
                )
            )
            return replace(
                unsigned,
                metadata={
                    **dict(unsigned.metadata),
                    "official_metric_receipt_signatures": {
                        "ifeval_pass_rate": signature
                    },
                },
            )
        if benchmark == "mmlu_pro":
            correct = recompute_mmlu_pro_accuracy(record, question)
            unsigned = replace(
                record,
                outcome=Outcome.CORRECT if correct else Outcome.INCORRECT,
                metadata={
                    **metadata,
                    "official_correct": correct,
                    "official_metric_scorer_revision": self.scorer.scorer_revision,
                },
            )
            signature = self.attestor._sign(
                official_metric_receipt_body(
                    unsigned,
                    metric="mmlu_pro_accuracy",
                    value=correct,
                    scorer_revision=self.scorer.scorer_revision,
                )
            )
            return replace(
                unsigned,
                metadata={
                    **dict(unsigned.metadata),
                    "official_metric_receipt_signatures": {
                        "mmlu_pro_accuracy": signature
                    },
                },
            )
        if benchmark == "wikitext103":
            from mfh.experiments.e8_protected import (
                validate_wikitext_likelihood_evidence,
            )

            mean_nll = validate_wikitext_likelihood_evidence(
                record, question=question
            )
            unsigned = replace(
                record,
                metadata={
                    **metadata,
                    "official_metric_scorer_revision": self.scorer.scorer_revision,
                },
            )
            signature = self.attestor._sign(
                official_metric_receipt_body(
                    unsigned,
                    metric="wikitext103_negative_log_likelihood",
                    value=mean_nll,
                    scorer_revision=self.scorer.scorer_revision,
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
            unsigned = replace(
                record,
                metadata={
                    **metadata,
                    "refused": deterministic_refusal_decision(record.raw_output),
                    "harmful_compliance_score": None,
                    "safety_score_output_sha256": stable_hash(record.raw_output),
                    "safety_scorer_revision": self.scorer.scorer_revision,
                },
            )
            return replace(
                unsigned,
                metadata={
                    **dict(unsigned.metadata),
                    "safety_scorer_receipt_signature": self._safety_signature(
                        unsigned, self.scorer.scorer_revision
                    ),
                },
            )
        if benchmark == "strongreject_or_harmbench":
            try:
                graded, attempts = grade_strongreject_openrouter(
                    record,
                    question,
                    grader_directory=self.bundle / "strongreject-grader",
                    scorer=self.scorer,
                    receipt_signer=self._safety_signature,
                    api_key="",
                    transport=self._transport_or_create(),
                )
            except StrongRejectTerminalFailure as exc:
                unsigned = replace(
                    record,
                    outcome=Outcome.UNSCORABLE,
                    metadata={
                        **metadata,
                        "strongreject_grader_failed": True,
                        "strongreject_failure_evidence": {
                            "schema_version": 1,
                            "terminal_status": "U_after_bounded_attempts",
                            "attempt_receipts": [dict(value) for value in exc.receipts],
                        },
                        "refused": False,
                        "harmful_compliance_score": 1.0,
                        "safety_score_output_sha256": stable_hash(
                            record.raw_output
                        ),
                        "safety_scorer_revision": self.scorer.scorer_revision,
                    },
                )
                return replace(
                    unsigned,
                    metadata={
                        **dict(unsigned.metadata),
                        "safety_scorer_receipt_signature": self._safety_signature(
                            unsigned, self.scorer.scorer_revision
                        ),
                    },
                )
            return replace(
                graded,
                metadata={
                    **dict(graded.metadata),
                    "official_score_output_sha256": stable_hash(record.raw_output),
                    "strongreject_attempt_receipts": [
                        dict(value) for value in attempts
                    ],
                },
            )
        if benchmark == "language_consistency":
            requested = question.metadata.get("requested_language")
            if not isinstance(requested, str):
                raise DataValidationError(
                    "language-consistency row lacks its requested language"
                )
            evidence = language_response_evidence(
                record.raw_output, requested, question.aliases
            )
            return replace(
                record,
                outcome=Outcome(str(evidence["factual_outcome"])),
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
                    "accepted_aliases_digest": evidence[
                        "accepted_aliases_digest"
                    ],
                    "language_score_output_sha256": stable_hash(
                        record.raw_output
                    ),
                    "language_evaluation_evidence": evidence,
                },
            )
        raise DataValidationError(
            f"E7/E8 grader received unknown benchmark {benchmark!r}"
        )

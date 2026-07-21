"""Portable, response-bound factual grading for E6 development rows."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

from mfh.contracts import GenerationRecord, Outcome, Question
from mfh.data.normalization import normalize_answer
from mfh.errors import ConfigurationError, DataValidationError, FrozenArtifactError
from mfh.evaluation.grading import deterministic_short_answer_grade, triviaqa_scores
from mfh.evaluation.official import (
    GradingRequest,
    OfficialGraderSpec,
    load_official_grader_spec,
    render_grader_prompt,
)
from mfh.evaluation.openrouter import (
    OpenRouterTransport,
    route_for_grader,
    run_openrouter_grader,
    validate_openrouter_attempt_receipt,
)
from mfh.evaluation.simpleqa import (
    simpleqa_hedging_evidence,
    simpleqa_hedging_evidence_is_valid,
)
from mfh.experiments.e8_protected import question_source_fingerprint
from mfh.experiments.grader_bundle import verify_e1_grader_bundle
from mfh.provenance import canonical_json, sha256_path, stable_hash


@dataclass(frozen=True, slots=True)
class E6OfficialGraderBundle:
    directory: Path
    manifest_digest: str
    fingerprint: str
    specs: Mapping[str, OfficialGraderSpec]

    def __post_init__(self) -> None:
        object.__setattr__(self, "specs", MappingProxyType(dict(self.specs)))


def load_e6_official_grader_bundle(
    directory: str | Path, *, expected_manifest_digest: str | None = None
) -> E6OfficialGraderBundle:
    source = Path(directory).resolve()
    manifest = verify_e1_grader_bundle(
        source,
        expected_manifest_digest=expected_manifest_digest,
        verify_live_sources=False,
    )
    try:
        descriptors = manifest["files"]
        specs = {
            "simpleqa_verified": load_official_grader_spec(
                source / str(descriptors["simpleqa_config"]["path"])
            ),
            "aa_omniscience_public_600": load_official_grader_spec(
                source / str(descriptors["aa_config"]["path"])
            ),
        }
        digest = str(manifest["manifest_digest"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FrozenArtifactError(f"cannot load E6 official graders: {exc}") from exc
    if set(specs) != {value.benchmark for value in specs.values()}:
        raise FrozenArtifactError("E6 official grader identities differ")
    return E6OfficialGraderBundle(
        directory=source,
        manifest_digest=digest,
        fingerprint=sha256_path(source),
        specs=specs,
    )


def load_env_secret(path: str | Path, name: str) -> str:
    existing = os.environ.get(name)
    if existing is not None and existing.strip():
        return existing.strip()
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConfigurationError(f"cannot read E6 secret environment file: {exc}") from exc
    found: list[str] = []
    for line in lines:
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        if value.startswith("export "):
            value = value[7:].lstrip()
        key, separator, raw = value.partition("=")
        if not separator or key.strip() != name:
            continue
        raw = raw.strip()
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {'"', "'"}:
            raw = raw[1:-1]
        found.append(raw)
    if len(found) != 1 or not found[0].strip():
        raise ConfigurationError(f"{name} must occur exactly once and be non-empty")
    return found[0].strip()


class E6FactualGrader:
    """Grade E6 rows using frozen official rubrics and one reusable transport."""

    def __init__(
        self,
        bundle: E6OfficialGraderBundle,
        *,
        environment_file: str | Path,
        transport: OpenRouterTransport | None = None,
    ) -> None:
        self.bundle = bundle
        self.environment_file = Path(environment_file)
        self._transport = transport

    def __call__(self, record: GenerationRecord, question: Question) -> GenerationRecord:
        metadata = dict(record.metadata)
        metadata["official_score_output_sha256"] = stable_hash(record.raw_output)
        if record.benchmark == "triviaqa":
            exact_match, token_f1 = triviaqa_scores(record.raw_output, question.aliases)
            metadata.update(
                {
                    "official_scorer": "mfh.triviaqa.alias-aware-em-f1.v1",
                    "official_exact_match": exact_match,
                    "official_token_f1": token_f1,
                    "reference_aliases_digest": stable_hash(list(question.aliases)),
                }
            )
            graded = replace(
                record,
                normalized_answer=normalize_answer(record.raw_output),
                outcome=deterministic_short_answer_grade(record.raw_output, question.aliases),
                metadata=metadata,
            )
            verify_e6_factual_grade(graded, question, grader_bundle=self.bundle)
            return graded
        try:
            spec = self.bundle.specs[record.benchmark]
        except KeyError as exc:
            raise DataValidationError("E6 factual grader received an unknown benchmark") from exc
        if record.benchmark == "simpleqa_verified":
            metadata["simpleqa_hedging_evidence"] = simpleqa_hedging_evidence(
                record.raw_output
            )
        request = GradingRequest(
            question.question_id, question.text, question.aliases[0], record.raw_output
        )
        prompt = render_grader_prompt(spec, request)
        if self._transport is None:
            self._transport = OpenRouterTransport(
                api_key=load_env_secret(self.environment_file, "OPENROUTER_API_KEY")
            )
        start = len(self._transport.receipts)
        grade = run_openrouter_grader(spec, request, self._transport)
        attempts = [value.to_dict() for value in self._transport.receipts[start:]]
        evidence = {
            "schema_version": 1,
            "grader_bundle_manifest_digest": self.bundle.manifest_digest,
            "grader_spec_digest": spec.digest,
            "question_source_sha256": question_source_fingerprint(question),
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
                "grader_bundle_manifest_digest": self.bundle.manifest_digest,
                "grader_model": spec.grader_model,
                "grader_model_revision": spec.grader_model_revision,
                "grader_source_artifact_sha256": spec.source_artifact_sha256,
            }
        )
        graded = replace(
            record,
            normalized_answer=normalize_answer(record.raw_output),
            outcome=grade.outcome,
            metadata=metadata,
        )
        verify_e6_factual_grade(graded, question, grader_bundle=self.bundle)
        return graded


def verify_e6_factual_grade(
    record: GenerationRecord,
    question: Question,
    *,
    grader_bundle: E6OfficialGraderBundle,
) -> None:
    if (
        record.question_id != question.question_id
        or record.benchmark != question.benchmark
        or record.metadata.get("official_score_output_sha256")
        != stable_hash(record.raw_output)
        or record.normalized_answer != normalize_answer(record.raw_output)
    ):
        raise FrozenArtifactError("E6 factual grade is not bound to its source response")
    if record.benchmark == "triviaqa":
        exact_match, token_f1 = triviaqa_scores(record.raw_output, question.aliases)
        if (
            record.outcome
            is not deterministic_short_answer_grade(record.raw_output, question.aliases)
            or record.metadata.get("official_scorer")
            != "mfh.triviaqa.alias-aware-em-f1.v1"
            or record.metadata.get("official_exact_match") != exact_match
            or record.metadata.get("official_token_f1") != token_f1
            or record.metadata.get("reference_aliases_digest")
            != stable_hash(list(question.aliases))
            or record.metadata.get("official_grader_evidence") is not None
        ):
            raise FrozenArtifactError("E6 TriviaQA grade differs from frozen aliases")
        return
    try:
        spec = grader_bundle.specs[record.benchmark]
    except KeyError as exc:
        raise FrozenArtifactError("E6 grade names an unsupported benchmark") from exc
    if record.benchmark == "simpleqa_verified" and not simpleqa_hedging_evidence_is_valid(
        record.raw_output, record.metadata.get("simpleqa_hedging_evidence")
    ):
        raise FrozenArtifactError("E6 SimpleQA hedging evidence does not replay")
    request = GradingRequest(
        question.question_id, question.text, question.aliases[0], record.raw_output
    )
    prompt = render_grader_prompt(spec, request)
    route = route_for_grader(spec)
    payload = OpenRouterTransport(api_key="e6-validation-only").request_payload(prompt, spec)
    request_sha = hashlib.sha256(canonical_json(payload).encode()).hexdigest()
    prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
    evidence = record.metadata.get("official_grader_evidence")
    expected_keys = {
        "schema_version",
        "grader_bundle_manifest_digest",
        "grader_spec_digest",
        "question_source_sha256",
        "response_sha256",
        "request_fingerprint",
        "rendered_prompt_sha256",
        "raw_label",
        "terminal_error",
        "attempt_receipts",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != expected_keys:
        raise FrozenArtifactError("E6 official grade lacks strict replay evidence")
    attempts = evidence["attempt_receipts"]
    raw_label = evidence["raw_label"]
    terminal_error = evidence["terminal_error"]
    succeeded = terminal_error is None
    if (
        evidence["schema_version"] != 1
        or evidence["grader_bundle_manifest_digest"] != grader_bundle.manifest_digest
        or evidence["grader_spec_digest"] != spec.digest
        or evidence["question_source_sha256"] != question_source_fingerprint(question)
        or evidence["response_sha256"] != stable_hash(record.raw_output)
        or evidence["request_fingerprint"] != request.digest
        or evidence["rendered_prompt_sha256"] != prompt_sha
        or not isinstance(raw_label, str)
        or not isinstance(attempts, list)
        or not 1 <= len(attempts) <= spec.maximum_attempts
        or (
            not succeeded
            and (
                not isinstance(terminal_error, str)
                or not terminal_error
                or raw_label != ""
                or (
                    len(attempts) < spec.maximum_attempts
                    and attempts[-1].get("transient") is not False
                )
            )
        )
    ):
        raise FrozenArtifactError("E6 official grade evidence identity differs")
    for index, attempt in enumerate(attempts, start=1):
        if not isinstance(attempt, Mapping):
            raise FrozenArtifactError("E6 official grader receipt is not a mapping")
        validate_openrouter_attempt_receipt(
            attempt,
            route=route,
            request_sha256=request_sha,
            prompt_sha256=prompt_sha,
            attempt=index,
            expect_success=succeeded and index == len(attempts),
            expected_content=raw_label if succeeded and index == len(attempts) else None,
            accepted_success_content=lambda value: value.strip() in spec.label_mapping,
            expect_retry=not succeeded and index < len(attempts),
        )
    mapped = spec.label_mapping.get(raw_label.strip()) if succeeded else Outcome.UNSCORABLE
    expected_error = (
        None if succeeded else f"OpenRouterError: {attempts[-1]['error_message']}"
    )
    if (
        mapped is None
        or record.outcome is not mapped
        or terminal_error != expected_error
        or record.metadata.get("grader_attempts") != len(attempts)
        or record.metadata.get("grader_failed") is not (not succeeded)
        or record.metadata.get("grader_request_fingerprint") != request.digest
        or record.metadata.get("grader_fingerprint") != spec.digest
        or record.metadata.get("grader_raw_label") != raw_label
        or record.metadata.get("grader_bundle_manifest_digest")
        != grader_bundle.manifest_digest
        or record.metadata.get("grader_model") != spec.grader_model
        or record.metadata.get("grader_model_revision") != spec.grader_model_revision
        or record.metadata.get("grader_source_artifact_sha256")
        != spec.source_artifact_sha256
    ):
        raise FrozenArtifactError("E6 official grade does not replay")

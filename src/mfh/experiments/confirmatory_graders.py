"""Recursive grader freeze used by the confirmatory E9 and E10 runs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from mfh.contracts import GenerationRecord, Outcome, Question
from mfh.data.normalization import normalize_answer
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.evaluation.grading import deterministic_short_answer_grade, triviaqa_scores
from mfh.evaluation.ifeval import validate_ifeval_evaluator
from mfh.evaluation.official import (
    GradingRequest,
    OfficialGraderSpec,
    load_official_grader_spec,
    render_grader_prompt,
)
from mfh.evaluation.openrouter import (
    OpenRouterTransport,
    route_for_grader,
    validate_openrouter_attempt_receipt,
)
from mfh.evaluation.side_effects import SideEffectScorerSpec, load_side_effect_scorer_spec
from mfh.evaluation.simpleqa import simpleqa_hedging_evidence_is_valid
from mfh.evaluation.strongreject import validate_strongreject_grader
from mfh.experiments.e6_likelihood import _load_e6_runtime_attestation
from mfh.experiments.grader_bundle import verify_e1_grader_bundle
from mfh.experiments.model_selection import validate_active_study_artifact_paths
from mfh.provenance import canonical_json, sha256_path, stable_hash

_INVENTORY = {
    "manifest.json",
    "official-graders",
    "side-effect-scorer.json",
    "ifeval-evaluator",
    "strongreject-grader",
    "runtime-attestation.json",
}
@dataclass(frozen=True, slots=True)
class ConfirmatoryGraderBundle:
    """Verified identities and paths for every confirmatory scorer."""

    directory: Path
    manifest_digest: str
    fingerprint: str
    official_manifest_digest: str
    scorer: SideEffectScorerSpec
    runtime_attestation: Mapping[str, Any]
    component_fingerprints: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "component_fingerprints",
            MappingProxyType(dict(self.component_fingerprints)),
        )
        object.__setattr__(
            self,
            "runtime_attestation",
            MappingProxyType(dict(self.runtime_attestation)),
        )

    @property
    def runtime_identity_digest(self) -> str:
        return str(self.runtime_attestation["runtime_identity_digest"])


def _copy_regular_artifact(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.exists():
        raise DataValidationError(f"grader component is missing or linked: {source}")
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        return
    if not source.is_dir() or any(item.is_symlink() for item in source.rglob("*")):
        raise DataValidationError(f"grader component is not a strict directory: {source}")
    shutil.copytree(source, destination, symlinks=False)


def write_confirmatory_grader_bundle(
    directory: str | Path,
    *,
    official_grader_bundle: str | Path,
    expected_official_manifest_digest: str,
    side_effect_scorer: str | Path,
    ifeval_evaluator: str | Path,
    strongreject_grader: str | Path,
    runtime_attestation: str | Path,
) -> ConfirmatoryGraderBundle:
    """Freeze all factual and side-suite graders without changing frozen E1 bytes."""

    normalized = validate_active_study_artifact_paths(
        {
            "confirmatory grader bundle": directory,
            "confirmatory runtime attestation": runtime_attestation,
        }
    )
    destination = normalized["confirmatory grader bundle"]
    runtime_attestation = normalized["confirmatory runtime attestation"]
    if destination.exists() or destination.is_symlink():
        raise FrozenArtifactError(
            f"refusing to overwrite confirmatory grader bundle: {destination}"
        )
    official_source = Path(official_grader_bundle)
    official = verify_e1_grader_bundle(
        official_source,
        expected_manifest_digest=expected_official_manifest_digest,
    )
    scorer_source = Path(side_effect_scorer)
    scorer = load_side_effect_scorer_spec(scorer_source)
    ifeval_source = Path(ifeval_evaluator)
    strongreject_source = Path(strongreject_grader)
    runtime_source = Path(runtime_attestation)
    validate_ifeval_evaluator(ifeval_source)
    validate_strongreject_grader(strongreject_source)
    runtime = _load_e6_runtime_attestation(runtime_source)
    if runtime["execution_public_key"] != scorer.execution_public_key:
        raise DataValidationError(
            "confirmatory runtime and scorer use different execution keys"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent)
    )
    try:
        _copy_regular_artifact(official_source, stage / "official-graders")
        scorer_file = (
            scorer_source / "side-effect-scorer.json"
            if scorer_source.is_dir()
            else scorer_source
        )
        _copy_regular_artifact(scorer_file, stage / "side-effect-scorer.json")
        _copy_regular_artifact(ifeval_source, stage / "ifeval-evaluator")
        _copy_regular_artifact(strongreject_source, stage / "strongreject-grader")
        _copy_regular_artifact(runtime_source, stage / "runtime-attestation.json")
        components = {
            "official_graders": sha256_path(stage / "official-graders"),
            "side_effect_scorer": sha256_path(stage / "side-effect-scorer.json"),
            "ifeval_evaluator": sha256_path(stage / "ifeval-evaluator"),
            "strongreject_grader": sha256_path(stage / "strongreject-grader"),
            "runtime_attestation": sha256_path(stage / "runtime-attestation.json"),
        }
        body: dict[str, Any] = {
            "schema_version": 2,
            "bundle_kind": "e9-e10-confirmatory-graders",
            "official_grader_manifest_digest": official["manifest_digest"],
            "side_effect_scorer_digest": scorer.digest,
            "execution_public_key": scorer.execution_public_key,
            "runtime_identity_digest": runtime["runtime_identity_digest"],
            "components": components,
        }
        (stage / "manifest.json").write_text(
            json.dumps(
                {**body, "manifest_digest": stable_hash(body)},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        verified = validate_confirmatory_grader_bundle(
            stage,
            expected_official_manifest_digest=expected_official_manifest_digest,
        )
        os.replace(stage, destination)
        return ConfirmatoryGraderBundle(
            directory=destination,
            manifest_digest=verified.manifest_digest,
            fingerprint=sha256_path(destination),
            official_manifest_digest=verified.official_manifest_digest,
            scorer=verified.scorer,
            runtime_attestation=verified.runtime_attestation,
            component_fingerprints=verified.component_fingerprints,
        )
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def validate_confirmatory_grader_bundle(
    directory: str | Path,
    *,
    expected_official_manifest_digest: str | None = None,
) -> ConfirmatoryGraderBundle:
    """Independently replay the strict recursive grader inventory and identities."""

    source = Path(directory)
    if (
        source.is_symlink()
        or not source.is_dir()
        or {item.name for item in source.iterdir()} != _INVENTORY
        or any(item.is_symlink() for item in source.rglob("*"))
    ):
        raise FrozenArtifactError("confirmatory grader bundle inventory differs")
    try:
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read confirmatory grader manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise FrozenArtifactError("confirmatory grader manifest must be an object")
    manifest_digest = manifest.pop("manifest_digest", None)
    official = verify_e1_grader_bundle(
        source / "official-graders",
        expected_manifest_digest=expected_official_manifest_digest,
    )
    scorer = load_side_effect_scorer_spec(source / "side-effect-scorer.json")
    validate_ifeval_evaluator(source / "ifeval-evaluator")
    validate_strongreject_grader(source / "strongreject-grader")
    runtime = _load_e6_runtime_attestation(source / "runtime-attestation.json")
    components = {
        "official_graders": sha256_path(source / "official-graders"),
        "side_effect_scorer": sha256_path(source / "side-effect-scorer.json"),
        "ifeval_evaluator": sha256_path(source / "ifeval-evaluator"),
        "strongreject_grader": sha256_path(source / "strongreject-grader"),
        "runtime_attestation": sha256_path(source / "runtime-attestation.json"),
    }
    expected = {
        "schema_version": 2,
        "bundle_kind": "e9-e10-confirmatory-graders",
        "official_grader_manifest_digest": official["manifest_digest"],
        "side_effect_scorer_digest": scorer.digest,
        "execution_public_key": scorer.execution_public_key,
        "runtime_identity_digest": runtime["runtime_identity_digest"],
        "components": components,
    }
    if (
        runtime["execution_public_key"] != scorer.execution_public_key
        or manifest != expected
        or manifest_digest != stable_hash(expected)
    ):
        raise FrozenArtifactError("confirmatory grader bundle identity differs")
    return ConfirmatoryGraderBundle(
        directory=source,
        manifest_digest=str(manifest_digest),
        fingerprint=sha256_path(source),
        official_manifest_digest=str(official["manifest_digest"]),
        scorer=scorer,
        runtime_attestation=runtime,
        component_fingerprints=components,
    )


def _question_fingerprint(question: Question) -> str:
    return stable_hash(
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


def _official_grader_spec(
    bundle: ConfirmatoryGraderBundle,
    benchmark: str,
) -> OfficialGraderSpec:
    try:
        manifest = json.loads(
            (bundle.directory / "official-graders" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        key = {
            "simpleqa_verified": "simpleqa_config",
            "aa_omniscience_public_600": "aa_config",
        }[benchmark]
        relative = manifest["files"][key]["path"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise FrozenArtifactError(
            f"cannot locate confirmatory official grader: {exc}"
        ) from exc
    return load_official_grader_spec(
        bundle.directory / "official-graders" / str(relative)
    )


def validate_confirmatory_factual_grade(
    record: GenerationRecord,
    question: Question,
    *,
    grader_bundle: ConfirmatoryGraderBundle,
) -> None:
    """Replay one factual grade against the exact frozen question and grader route."""

    if (
        record.question_id != question.question_id
        or record.benchmark != question.benchmark
        or record.benchmark
        not in {
            "triviaqa",
            "simpleqa_verified",
            "aa_omniscience_public_600",
        }
    ):
        raise DataValidationError("confirmatory factual grade received a mismatched row")
    if record.metadata.get("official_score_output_sha256") != stable_hash(
        record.raw_output
    ):
        raise DataValidationError("confirmatory factual grade is not response-bound")
    if record.benchmark == "triviaqa":
        expected_outcome = deterministic_short_answer_grade(
            record.raw_output, question.aliases
        )
        exact_match, token_f1 = triviaqa_scores(record.raw_output, question.aliases)
        if (
            record.outcome is not expected_outcome
            or record.normalized_answer != normalize_answer(record.raw_output)
            or record.metadata.get("official_scorer")
            != "mfh.triviaqa.alias-aware-em-f1.v1"
            or record.metadata.get("official_exact_match") != exact_match
            or record.metadata.get("official_token_f1") != token_f1
            or record.metadata.get("reference_aliases_digest")
            != stable_hash(list(question.aliases))
            or record.metadata.get("official_grader_evidence") is not None
        ):
            raise DataValidationError(
                "confirmatory TriviaQA grade differs from frozen aliases"
            )
        return

    if record.benchmark == "simpleqa_verified" and not simpleqa_hedging_evidence_is_valid(
        record.raw_output, record.metadata.get("simpleqa_hedging_evidence")
    ):
        raise DataValidationError("confirmatory SimpleQA hedging evidence does not replay")

    spec = _official_grader_spec(grader_bundle, record.benchmark)
    request = GradingRequest(
        question.question_id,
        question.text,
        question.aliases[0],
        record.raw_output,
    )
    prompt = render_grader_prompt(spec, request)
    route = route_for_grader(spec)
    request_payload = OpenRouterTransport(
        api_key="confirmatory-validation-only"
    ).request_payload(prompt, spec)
    request_sha = hashlib.sha256(
        canonical_json(request_payload).encode("utf-8")
    ).hexdigest()
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    evidence = record.metadata.get("official_grader_evidence")
    if not isinstance(evidence, Mapping) or set(evidence) != {
        "schema_version",
        "grader_bundle_manifest_digest",
        "official_grader_manifest_digest",
        "grader_spec_digest",
        "question_source_sha256",
        "response_sha256",
        "request_fingerprint",
        "rendered_prompt_sha256",
        "raw_label",
        "terminal_error",
        "attempt_receipts",
    }:
        raise DataValidationError("confirmatory model grade lacks strict evidence")
    raw_label = evidence["raw_label"]
    attempts = evidence["attempt_receipts"]
    if (
        evidence["schema_version"] != 2
        or evidence["grader_bundle_manifest_digest"]
        != grader_bundle.manifest_digest
        or evidence["official_grader_manifest_digest"]
        != grader_bundle.official_manifest_digest
        or evidence["grader_spec_digest"] != spec.digest
        or evidence["question_source_sha256"] != _question_fingerprint(question)
        or evidence["response_sha256"] != stable_hash(record.raw_output)
        or evidence["request_fingerprint"] != request.digest
        or evidence["rendered_prompt_sha256"] != prompt_sha
        or not isinstance(raw_label, str)
        or not isinstance(attempts, list)
        or not 1 <= len(attempts) <= spec.maximum_attempts
    ):
        raise DataValidationError("confirmatory model-grade evidence identity differs")
    terminal_error = evidence["terminal_error"]
    succeeded = terminal_error is None
    if not succeeded and (
        not isinstance(terminal_error, str)
        or not terminal_error
        or raw_label != ""
        or (
            len(attempts) < spec.maximum_attempts
            and attempts[-1].get("transient") is not False
        )
    ):
        raise DataValidationError("confirmatory terminal grader failure is invalid")
    for index, attempt in enumerate(attempts, start=1):
        if not isinstance(attempt, Mapping):
            raise DataValidationError("confirmatory grader attempt is not a mapping")
        validate_openrouter_attempt_receipt(
            attempt,
            route=route,
            request_sha256=request_sha,
            prompt_sha256=prompt_sha,
            attempt=index,
            expect_success=succeeded and index == len(attempts),
            expected_content=(
                raw_label if succeeded and index == len(attempts) else None
            ),
            accepted_success_content=lambda content: content.strip()
            in spec.label_mapping,
            expect_retry=not succeeded and index < len(attempts),
        )
    mapped_outcome = (
        spec.label_mapping.get(raw_label.strip()) if succeeded else Outcome.UNSCORABLE
    )
    expected_terminal_error = (
        None
        if succeeded
        else f"OpenRouterError: {attempts[-1]['error_message']}"
    )
    if (
        mapped_outcome is None
        or record.outcome is not mapped_outcome
        or terminal_error != expected_terminal_error
        or record.normalized_answer != normalize_answer(record.raw_output)
        or record.metadata.get("grader_attempts") != len(attempts)
        or record.metadata.get("grader_failed") is not (not succeeded)
        or record.metadata.get("grader_request_fingerprint") != request.digest
        or record.metadata.get("grader_fingerprint") != spec.digest
        or record.metadata.get("grader_raw_label") != raw_label
        or record.metadata.get("grader_bundle_manifest_digest")
        != grader_bundle.manifest_digest
        or record.metadata.get("grader_model") != spec.grader_model
        or record.metadata.get("grader_model_revision")
        != spec.grader_model_revision
        or record.metadata.get("grader_source_artifact_sha256")
        != spec.source_artifact_sha256
    ):
        raise DataValidationError("confirmatory official grade does not replay")

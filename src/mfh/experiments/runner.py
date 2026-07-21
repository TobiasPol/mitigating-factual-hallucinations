"""Condition expansion and crash-safe, immutable phase-run ledgers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import struct
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from itertools import chain, product
from pathlib import Path
from types import MappingProxyType
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from mfh.contracts import (
    ActivationSite,
    AdaptivePolicySpec,
    GenerationRecord,
    InterventionSpec,
    ModelSpec,
    Outcome,
    PromptSpec,
    Question,
    Runtime,
    TokenScope,
)
from mfh.data.io import (
    read_generation_records,
    read_questions,
    write_generation_records,
    write_questions,
)
from mfh.data.language_suite import load_reviewed_language_suite
from mfh.data.side_effect_sampling import select_mmlu_pro_stratified
from mfh.data.source_snapshots import (
    SOURCE_SNAPSHOTS,
    SourceSnapshot,
    iter_source_questions,
    validate_source_membership,
    verify_source_artifact,
)
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.evaluation.side_effects import load_side_effect_scorer_spec
from mfh.experiments.evidence import GateResult, read_gate_result, write_gate_result
from mfh.experiments.gates import (
    GateEvaluationContext,
    validate_gate_result,
    validate_side_effect_record,
)
from mfh.experiments.gates import (
    evaluate_gate as evaluate_registered_gate,
)
from mfh.experiments.model_selection import (
    ACTIVE_MODEL_IDENTITIES,
    validate_active_study_artifact_paths,
)
from mfh.experiments.protocol import ExperimentPhase, StudyProtocol
from mfh.experiments.runtime_evidence import validate_generation_runtime_metrics
from mfh.experiments.snapshots import validate_execution_snapshot
from mfh.inference.transformers_snapshot import reject_symlink_path_components
from mfh.methods.sae_stability import load_sae_stability_bundle
from mfh.provenance import canonical_json, sha256_file, sha256_path, stable_hash

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
_SHARD = re.compile(r"^records-(\d{5})\.jsonl$")
_GATE_FILE = re.compile(r"^[a-z0-9][a-z0-9_-]*\.json$")
_ARTIFACT_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_MIN_ADAPTIVE_ALPHA = 1e-4
_MIN_ADAPTIVE_ACTION_FRACTION = 0.01
_VERIFIED_LEDGER = object()
_MODEL_IDENTITIES = {
    name: (
        identity["repository"],
        identity["revision"],
        identity["runtime"],
        identity["quantization"],
        identity["num_layers"],
    )
    for name, identity in ACTIVE_MODEL_IDENTITIES.items()
}
_PROMPT_HASHES = {
    "P0-neutral": "5b08080e81d4032d853d3a30fda35e7670da6a81cc1ccf17f2b8fc5001bba684",
    "P1-direct": "2c37eff2211a9f717f3d662bd97a3197c07a514ee1ad57b5ab500069b78ca70a",
    "P2-calibrated-abstention": (
        "3170134d9a69836c1b530d1b16585ef7b0d92ea6fadc8f958e2655053e273fe5"
    ),
    "P3-forced-answer": "db38cfa43eab1db9671c0d1d6d6a63f4b80f555c54d216b18813acdf61326ed2",
    "P-AA-official": "13310395995ff5d5caee478fbade133d5ab681d395c7439710df6131d74598c0",
}
_E10_DEPLOYMENT_PROMPTS = {"P0-neutral", "P2-calibrated-abstention"}
_FIXED_QUESTION_COUNTS = {
    ExperimentPhase.E0: {"shared_benign_factual_500": 500},
    ExperimentPhase.E4: {"triviaqa": 2_000},
    ExperimentPhase.E6: {
        "triviaqa": 5_000,
        "simpleqa_verified": 1_000,
        "aa_omniscience_public_600": 600,
    },
    ExperimentPhase.E7: {
        "triviaqa": 5_000,
        "ifeval": 541,
        "xstest": 250,
        "strongreject_or_harmbench": 313,
        "language_consistency": 500,
    },
    ExperimentPhase.E8: {
        "triviaqa": 5_000,
        "ifeval": 541,
        "mmlu_pro": 1_000,
        "wikitext103": 1_000,
        "xstest": 250,
        "strongreject_or_harmbench": 313,
        "language_consistency": 500,
    },
    ExperimentPhase.E9: {
        "triviaqa": 5_000,
        "simpleqa_verified": 1_000,
        "aa_omniscience_public_600": 600,
    },
    ExperimentPhase.E10: {
        "triviaqa": 5_000,
        "simpleqa_verified": 1_000,
        "aa_omniscience_public_600": 600,
        "ifeval": 541,
        "mmlu_pro": 1_000,
        "wikitext103": 1_000,
        "xstest": 250,
        "strongreject_or_harmbench": 313,
        "language_consistency": 500,
    },
}
_E9_PARTITIONS = {
    "triviaqa": "T-test",
    "simpleqa_verified": "simpleqa-eval",
    "aa_omniscience_public_600": "aa-eval",
}
_E6_PARTITIONS = {
    "triviaqa": "T-dev",
    "simpleqa_verified": "simpleqa-eval",
    "aa_omniscience_public_600": "aa-eval",
}
_E8_PARTITIONS = {
    "triviaqa": "T-dev",
    "ifeval": "side-effect-eval",
    "mmlu_pro": "side-effect-eval",
    "wikitext103": "side-effect-eval",
    "xstest": "side-effect-eval",
    "strongreject_or_harmbench": "side-effect-eval",
    "language_consistency": "side-effect-eval",
}
_E10_PARTITIONS = {
    **_E9_PARTITIONS,
    "ifeval": "side-effect-eval",
    "mmlu_pro": "side-effect-eval",
    "wikitext103": "side-effect-eval",
    "xstest": "side-effect-eval",
    "strongreject_or_harmbench": "side-effect-eval",
    "language_consistency": "side-effect-eval",
}
_QUESTION_BUNDLE = "frozen_question_bundle"
_COMPONENT_INPUT = {
    ExperimentPhase.E9: "frozen_component_selection",
    ExperimentPhase.E10: "component_selection_manifest",
}


def adaptive_policy_decision_digest(
    record: GenerationRecord,
    *,
    policy: AdaptivePolicySpec,
    policy_action: str,
    output_action: str | None = None,
) -> str:
    """Bind an adaptive routing decision to the exact action executed in a record."""

    decision: dict[str, Any] = {
        "schema_version": 1,
        "question_id": record.question_id,
        "condition_id": record.condition_id,
        "controller_scores": dict(sorted(record.controller_scores.items())),
        "adaptive_policy_digest": stable_hash(policy.to_dict()),
        "policy_action": policy_action,
        "layer": record.layer,
        "site": record.site.value if record.site is not None else None,
        "token_scope": record.token_scope.value if record.token_scope is not None else None,
        "alpha": record.alpha,
        "sparsity": record.sparsity,
        "intervention_trace_digest": record.metadata.get("intervention_trace_digest"),
    }
    if record.steering_method == "M6":
        decision["output_action"] = output_action
        decision["post_controller_scores"] = record.metadata.get("post_controller_scores")
    return stable_hash(decision)


def adaptive_execution_receipt_body(
    record: GenerationRecord,
    *,
    policy: AdaptivePolicySpec,
) -> dict[str, Any]:
    """Canonical body signed by the trusted inference runtime after hook execution."""

    post_scores = record.metadata.get("post_controller_scores")
    trace = record.metadata.get("intervention_trace")
    direction_sha256 = policy.direction_sha256
    direction_norm = policy.direction_norm
    if policy.schema_version == 2 and isinstance(trace, Mapping):
        direction_sha256 = trace.get("direction_sha256")
        direction_norm = trace.get("direction_norm")
    body = {
        "schema_version": 1,
        "question_id": record.question_id,
        "condition_id": record.condition_id,
        "rendered_prompt_hash": record.rendered_prompt_hash,
        "raw_output_sha256": hashlib.sha256(record.raw_output.encode("utf-8")).hexdigest(),
        "normalized_answer_sha256": hashlib.sha256(
            record.normalized_answer.encode("utf-8")
        ).hexdigest(),
        "outcome": record.outcome.value,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
        "generation_latency_seconds": record.generation_latency_seconds,
        "generation_runtime_metrics": record.metadata.get("generation_runtime_metrics"),
        "decoding_max_new_tokens": record.metadata.get("decoding_max_new_tokens"),
        "method_artifact_sha256": record.metadata.get("method_artifact_sha256"),
        "source_question_sha256": record.metadata.get("source_question_sha256"),
        "prompt_template_sha256": record.metadata.get("prompt_template_sha256"),
        "adaptive_policy_digest": stable_hash(policy.to_dict()),
        "direction_sha256": direction_sha256,
        "controller_scores": dict(sorted(record.controller_scores.items())),
        "post_controller_scores": (
            dict(sorted(post_scores.items())) if isinstance(post_scores, Mapping) else None
        ),
        "policy_action": record.metadata.get("policy_action"),
        "output_action": record.metadata.get("output_action"),
        "intervention_trace": trace,
        "adaptive_controller_evidence": record.metadata.get("adaptive_controller_evidence"),
        "wikitext_likelihood_evidence": record.metadata.get("wikitext_likelihood_evidence"),
    }
    if policy.schema_version == 2:
        body.update(
            {
                "receipt_schema_version": 2,
                "controller_artifact_sha256": policy.controller_artifact_sha256,
                "direction_norm": direction_norm,
            }
        )
    return body


def _validate_adaptive_controller_evidence(
    evidence: object,
    *,
    policy: AdaptivePolicySpec,
) -> None:
    """Replay the exact prompt-feature payload signed by the native M3 runtime."""

    expected_keys = {
        "schema_version",
        "controller_artifact_sha256",
        "feature_schema_digest",
        "feature_values_sha256",
        "feature_values",
        "prompt_feature_peak_memory_bytes",
        "maximum_token_probability",
        "output_entropy",
        "site_selection",
    }
    if (
        not isinstance(evidence, Mapping)
        or set(evidence) != expected_keys
        or evidence.get("schema_version") != 1
        or evidence.get("controller_artifact_sha256") != policy.controller_artifact_sha256
        or not isinstance(evidence.get("feature_schema_digest"), str)
        or _SHA256.fullmatch(str(evidence["feature_schema_digest"])) is None
        or evidence.get("site_selection") != "max_mixed_direction_norm_then_site"
    ):
        raise DataValidationError("adaptive controller evidence identity differs")
    feature_values = evidence.get("feature_values")
    if (
        not isinstance(feature_values, list)
        or not feature_values
        or any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            for value in feature_values
        )
    ):
        raise DataValidationError("adaptive controller feature values are invalid")
    try:
        feature_bytes = struct.pack(
            f"<{len(feature_values)}f", *(float(value) for value in feature_values)
        )
    except (OverflowError, struct.error) as exc:
        raise DataValidationError(
            f"adaptive controller feature values cannot be encoded: {exc}"
        ) from exc
    peak_memory = evidence.get("prompt_feature_peak_memory_bytes")
    maximum_probability = evidence.get("maximum_token_probability")
    entropy = evidence.get("output_entropy")
    if (
        evidence.get("feature_values_sha256") != hashlib.sha256(feature_bytes).hexdigest()
        or isinstance(peak_memory, bool)
        or not isinstance(peak_memory, int)
        or peak_memory <= 0
        or isinstance(maximum_probability, bool)
        or not isinstance(maximum_probability, int | float)
        or not math.isfinite(float(maximum_probability))
        or not 0 <= float(maximum_probability) <= 1
        or isinstance(entropy, bool)
        or not isinstance(entropy, int | float)
        or not math.isfinite(float(entropy))
        or float(entropy) < 0
    ):
        raise DataValidationError("adaptive controller feature evidence does not replay")


def sign_adaptive_execution_receipt(
    record: GenerationRecord,
    *,
    policy: AdaptivePolicySpec,
    private_key_hex: str,
) -> str:
    """Sign a completed runtime receipt with the key whose public half was frozen."""

    if not re.fullmatch(r"[0-9a-f]{64}", private_key_hex):
        raise DataValidationError("adaptive execution private key must be 32-byte hex")
    try:
        private_key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_key_hex))
        signature = private_key.sign(
            canonical_json(adaptive_execution_receipt_body(record, policy=policy)).encode("utf-8")
        )
    except ValueError as exc:
        raise DataValidationError(f"invalid adaptive execution private key: {exc}") from exc
    return signature.hex()


def confirmatory_execution_receipt_body(record: GenerationRecord) -> Mapping[str, Any]:
    """Bind an E9/E10 row, including its output, grade, and runtime evidence."""

    value = record.to_dict()
    metadata = value.get("metadata")
    if not isinstance(metadata, dict):  # pragma: no cover - GenerationRecord guarantees this
        raise DataValidationError("confirmatory record metadata is invalid")
    metadata.pop("confirmatory_execution_receipt_signature", None)
    runtime_identity = metadata.get("runtime_session_identity_sha256")
    if not isinstance(runtime_identity, str) or not _SHA256.fullmatch(runtime_identity):
        raise DataValidationError("confirmatory record lacks its exact runtime-session identity")
    return MappingProxyType({"schema_version": 1, "record": value})


def _sign_confirmatory_execution_receipt_for_test(
    record: GenerationRecord,
    *,
    private_key_hex: str,
) -> str:
    """Test-fixture signer; production E9/E10 signing is runtime-owned."""

    if not re.fullmatch(r"[0-9a-f]{64}", private_key_hex):
        raise DataValidationError("confirmatory execution private key must be 32-byte hex")
    try:
        private_key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_key_hex))
        signature = private_key.sign(
            canonical_json(confirmatory_execution_receipt_body(record)).encode("utf-8")
        )
    except ValueError as exc:
        raise DataValidationError(f"invalid confirmatory execution private key: {exc}") from exc
    return signature.hex()


def _fixed_intervention_token_indices(record: GenerationRecord) -> list[int]:
    if record.token_scope is TokenScope.FINAL_PROMPT:
        return [-1]
    if record.token_scope is None:
        return []
    limit = {
        TokenScope.FIRST_GENERATED: 1,
        TokenScope.FIRST_FOUR: 4,
        TokenScope.FIRST_EIGHT: 8,
        TokenScope.ALL_GENERATED: record.output_tokens,
        TokenScope.EXPONENTIAL_DECAY: record.output_tokens,
    }[record.token_scope]
    return list(range(min(limit, record.output_tokens)))


def _validate_confirmatory_fixed_trace(
    record: GenerationRecord,
    condition: EvaluationCondition,
    *,
    component: Any | None = None,
) -> None:
    trace = record.metadata.get("intervention_trace")
    expected_keys = {
        "schema_version",
        "method_artifact_sha256",
        "layer",
        "site",
        "token_scope",
        "standardized_alpha",
        "sparsity",
        "direction_sha256",
        "direction_norm",
        "reference_rms",
        "raw_alpha",
        "decay",
        "applied_tokens",
        "applied_token_indices",
        "activation_delta_norm",
        "pre_activation_sha256",
        "post_activation_sha256",
        "delta_sha256",
        "runtime_session_identity_sha256",
    }
    if not isinstance(trace, Mapping) or set(trace) != expected_keys:
        raise DataValidationError(
            "confirmatory fixed intervention requires a strict execution trace"
        )
    numeric_names = {
        "standardized_alpha",
        "direction_norm",
        "reference_rms",
        "raw_alpha",
        "decay",
        "activation_delta_norm",
    }
    if any(
        isinstance(trace[name], bool)
        or not isinstance(trace[name], int | float)
        or not math.isfinite(float(trace[name]))
        for name in numeric_names
    ):
        raise DataValidationError("confirmatory fixed intervention trace is non-finite")
    direction_norm = float(trace["direction_norm"])
    reference_rms = float(trace["reference_rms"])
    raw_alpha = float(trace["raw_alpha"])
    decay = float(trace["decay"])
    delta_norm = float(trace["activation_delta_norm"])
    indices = _fixed_intervention_token_indices(record)
    if record.token_scope is TokenScope.EXPONENTIAL_DECAY:
        scale = math.sqrt(sum(math.exp(-2 * decay * index) for index in indices))
    else:
        scale = math.sqrt(len(indices))
    expected_delta = abs(raw_alpha) * direction_norm * scale
    runtime_identity = record.metadata.get("runtime_session_identity_sha256")
    if (
        trace["schema_version"] != 1
        or trace["method_artifact_sha256"] != condition.method_artifact_sha256
        or trace["layer"] != record.layer
        or trace["site"] != (record.site.value if record.site is not None else None)
        or trace["token_scope"]
        != (record.token_scope.value if record.token_scope is not None else None)
        or not math.isclose(
            float(trace["standardized_alpha"]),
            record.alpha,
            rel_tol=0,
            abs_tol=1e-12,
        )
        or trace["sparsity"] != record.sparsity
        or not isinstance(trace["direction_sha256"], str)
        or not _SHA256.fullmatch(trace["direction_sha256"])
        or direction_norm <= 0
        or reference_rms <= 0
        or not math.isclose(
            raw_alpha,
            record.alpha * reference_rms,
            rel_tol=0,
            abs_tol=1e-12,
        )
        or decay < 0
        or (record.token_scope is TokenScope.EXPONENTIAL_DECAY and decay <= 0)
        or (record.token_scope is not TokenScope.EXPONENTIAL_DECAY and decay != 0)
        or isinstance(trace["applied_tokens"], bool)
        or not isinstance(trace["applied_tokens"], int)
        or trace["applied_tokens"] != len(indices)
        or trace["applied_token_indices"] != indices
        or not indices
        or delta_norm <= 0
        or not math.isclose(delta_norm, expected_delta, rel_tol=0.025, abs_tol=1e-6)
        or trace["runtime_session_identity_sha256"] != runtime_identity
        or any(
            not isinstance(trace[name], str) or not _SHA256.fullmatch(trace[name])
            for name in (
                "pre_activation_sha256",
                "post_activation_sha256",
                "delta_sha256",
            )
        )
        or trace["pre_activation_sha256"] == trace["post_activation_sha256"]
        or record.metadata.get("intervention_trace_digest") != stable_hash(dict(trace))
    ):
        raise DataValidationError(
            "confirmatory fixed intervention trace does not prove the frozen edit"
        )
    if component is not None and (
        component.fingerprint != condition.method_artifact_sha256
        or component.method != condition.steering_method
        or component.layer != condition.layer
        or component.site is not condition.site
        or component.token_scope is not condition.token_scope
        or component.standardized_alpha != condition.alpha
        or component.sparsity != condition.sparsity
        or trace["direction_sha256"] != component.direction_sha256
        or not math.isclose(
            direction_norm,
            component.direction_norm,
            rel_tol=0,
            abs_tol=1e-7,
        )
        or not math.isclose(
            reference_rms,
            component.reference_rms,
            rel_tol=0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            decay,
            component.decay,
            rel_tol=0,
            abs_tol=1e-12,
        )
    ):
        raise DataValidationError(
            "confirmatory trace differs from its packaged execution component"
        )


def validate_confirmatory_execution_receipt(
    record: GenerationRecord,
    condition: EvaluationCondition,
    *,
    execution_public_key: str,
    fixed_component: Any | None = None,
    runtime_identity: Mapping[str, Any] | None = None,
) -> None:
    if condition.phase is ExperimentPhase.E9:
        risk_evidence = record.metadata.get("selective_risk_evidence")
        expected_auxiliary_peak = 0
        if risk_evidence is not None:
            prompt_peak = (
                risk_evidence.get("prompt_feature_peak_memory_bytes")
                if isinstance(risk_evidence, Mapping)
                else None
            )
            if (
                isinstance(prompt_peak, bool)
                or not isinstance(prompt_peak, int)
                or prompt_peak <= 0
            ):
                raise DataValidationError(
                    "confirmatory selective-risk peak-memory evidence is invalid"
                )
            expected_auxiliary_peak = prompt_peak
        validate_generation_runtime_metrics(
            record.metadata.get("generation_runtime_metrics"),
            record=record,
            runtime_identity=runtime_identity,
            expected_auxiliary_peak_memory_bytes=expected_auxiliary_peak,
        )
    if record.metadata.get("decoding_max_new_tokens") != 48:
        raise DataValidationError(
            "confirmatory record differs from the frozen 48-token decode limit"
        )
    adaptive = condition.steering_method in {"M3", "M6", "ACT-or-SADI"}
    if condition.steering_method == "M0":
        if (
            record.metadata.get("intervention_trace") is not None
            or record.metadata.get("intervention_trace_digest") is not None
        ):
            raise DataValidationError("confirmatory M0 records cannot claim an intervention")
    elif not adaptive:
        _validate_confirmatory_fixed_trace(
            record,
            condition,
            component=fixed_component,
        )
    else:
        policy = condition.adaptive_policy
        assert policy is not None
        if policy.execution_public_key != execution_public_key:
            raise DataValidationError("adaptive and confirmatory execution keys differ")
    signature = record.metadata.get("confirmatory_execution_receipt_signature")
    if not isinstance(signature, str) or not re.fullmatch(r"[0-9a-f]{128}", signature):
        raise DataValidationError("confirmatory record lacks its execution receipt")
    try:
        public_key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(execution_public_key))
        public_key.verify(
            bytes.fromhex(signature),
            canonical_json(confirmatory_execution_receipt_body(record)).encode("utf-8"),
        )
    except (InvalidSignature, ValueError) as exc:
        raise DataValidationError(
            "confirmatory execution receipt was not signed by the frozen key"
        ) from exc


def _assert_regular_artifact(path: Path, context: str) -> None:
    if path.is_symlink() or not (path.is_file() or path.is_dir()):
        raise DataValidationError(f"{context} must be a regular file or directory")
    if path.is_dir() and any(value.is_symlink() for value in path.rglob("*")):
        raise DataValidationError(f"{context} cannot contain symbolic links")


def _copy_frozen_artifact(source: Path, destination: Path, fingerprint: str) -> None:
    """Copy a verified artifact without permitting symlink or overwrite ambiguity."""

    _assert_regular_artifact(source, f"frozen artifact {source}")
    if destination.exists():
        if destination.is_symlink() or sha256_path(destination) != fingerprint:
            raise FrozenArtifactError(f"packaged artifact changed: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copyfile(source, destination)
    if sha256_path(destination) != fingerprint:
        raise FrozenArtifactError(f"packaged artifact copy differs: {destination}")


def _resolve_ledger_evidence_path(
    ledger_directory: Path, location: object, *, context: str
) -> Path:
    if type(location) is not str or not location:
        raise FrozenArtifactError(f"{context} location is invalid")
    candidate = Path(location)
    if candidate.is_absolute():
        return candidate.resolve()
    resolved = (ledger_directory / candidate).resolve()
    if not resolved.is_relative_to(ledger_directory.resolve()):
        raise FrozenArtifactError(f"{context} location escapes its ledger")
    return resolved


def _reviewed_split_authorization(
    contract: PhaseRunContract,
    evidence: object | None,
    *,
    input_sources: Mapping[str, Path],
    input_fingerprints: Mapping[str, str],
) -> dict[str, dict[str, str]]:
    if contract.phase is not ExperimentPhase.E1:
        if evidence is not None:
            raise DataValidationError("only E1 may authorize reviewed TriviaQA splits")
        return {}
    from mfh.data.reviewed_splits import (
        _assert_authorized_reviewed_splits,
        validate_reviewed_split_snapshot,
    )

    evidence = _assert_authorized_reviewed_splits(evidence)
    source = input_sources.get("deduplicated_splits")
    fingerprint = input_fingerprints.get("deduplicated_splits")
    if source is None or fingerprint is None:
        raise DataValidationError("E1 lacks its deduplicated-splits input")
    manifest = validate_reviewed_split_snapshot(source)
    if (
        evidence.directory.resolve() != source.resolve()
        or evidence.fingerprint != fingerprint
        or sha256_path(source) != evidence.fingerprint
        or manifest.get("manifest_digest") != evidence.manifest_digest
    ):
        raise DataValidationError("E1 reviewed-split authorization differs from its input")
    triviaqa_partitions = {
        condition.partition
        for condition in contract.conditions
        if condition.benchmark == "triviaqa"
    }
    if len(triviaqa_partitions) != 1:
        raise DataValidationError("E1 must use one reviewed TriviaQA partition schedule")
    partition = next(iter(triviaqa_partitions))
    if partition not in {"T-controller", "T-dev", "T-test"}:
        raise DataValidationError("E1 TriviaQA partition is not eligible for baseline execution")
    reviewed_ids = tuple(
        question.question_id for question in read_questions(source / f"{partition}.jsonl")
    )
    if contract.question_ids_by_benchmark.get("triviaqa") != reviewed_ids:
        raise DataValidationError(
            "E1 TriviaQA question schedule differs from its authorized reviewed partition"
        )
    external_schedules = {
        "simpleqa_verified": ("simpleqa-eval", "simpleqa-eval.jsonl"),
        "aa_omniscience_public_600": ("aa-eval", "aa-eval.jsonl"),
    }
    for benchmark, (expected_partition, filename) in external_schedules.items():
        observed_partitions = {
            condition.partition
            for condition in contract.conditions
            if condition.benchmark == benchmark
        }
        if observed_partitions != {expected_partition}:
            raise DataValidationError(
                f"E1 {benchmark} conditions use the wrong evaluation partition"
            )
        expected_ids = tuple(question.question_id for question in read_questions(source / filename))
        if contract.question_ids_by_benchmark.get(benchmark) != expected_ids:
            raise DataValidationError(
                f"E1 {benchmark} question schedule differs from its authorized source"
            )
    return {
        "deduplicated_splits": {
            "kind": "human-reviewed-contamination-controlled-triviaqa-splits",
            "manifest_digest": evidence.manifest_digest,
            "review_result_manifest_digest": str(manifest["review_result_manifest_digest"]),
            "fingerprint": evidence.fingerprint,
        }
    }


def _e0_review_result_digest(run_directory: Path) -> str:
    try:
        value = json.loads(
            (run_directory.resolve() / "scientific-completion-receipt" / "receipt.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read E0 scientific receipt provenance: {exc}") from exc
    manifests = value.get("source_manifests") if isinstance(value, Mapping) else None
    digest = manifests.get("contamination_review") if isinstance(manifests, Mapping) else None
    if type(digest) is not str or not _SHA256.fullmatch(digest):
        raise FrozenArtifactError("E0 scientific receipt review provenance is invalid")
    return digest


def _reserve_one_shot(
    registry: Path,
    *,
    study: StudyProtocol,
    contract: PhaseRunContract,
    destination: Path,
) -> tuple[dict[str, str], Path]:
    try:
        registry = reject_symlink_path_components(registry, "one-shot registry")
    except DataValidationError as exc:
        raise FrozenArtifactError(str(exc)) from exc
    registry.mkdir(parents=True, exist_ok=True)
    try:
        verified_registry = reject_symlink_path_components(registry, "one-shot registry")
    except DataValidationError as exc:
        raise FrozenArtifactError(str(exc)) from exc
    if verified_registry != registry or registry.is_symlink() or not registry.is_dir():
        raise FrozenArtifactError("one-shot registry must be a regular directory")
    reservation = registry / f"{study.study_id}-{study.digest}-E10.json"
    body = {
        "schema_version": 1,
        "study_id": study.study_id,
        "study_protocol_digest": study.digest,
        "phase": ExperimentPhase.E10.value,
        "contract_digest": contract.digest,
        "run_directory": str(destination.resolve()),
    }
    payload = (
        json.dumps(
            {**body, "reservation_digest": stable_hash(body)},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    claims = registry / "claims"
    claims.mkdir(exist_ok=True)
    try:
        claims = reject_symlink_path_components(claims, "one-shot claims")
    except DataValidationError as exc:
        raise FrozenArtifactError(str(exc)) from exc
    if claims.is_symlink() or not claims.is_dir():
        raise FrozenArtifactError("one-shot claims must be a regular directory")
    claim_directory = claims / reservation.stem
    try:
        claim_directory.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise FrozenArtifactError(
            "E10 has already been claimed for this exact study protocol"
        ) from exc
    claim = claim_directory / "claim.json"
    try:
        claim_descriptor = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(claim_descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise FrozenArtifactError(f"cannot freeze E10 one-shot claim: {exc}") from exc
    try:
        descriptor = os.open(reservation, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as exc:
        raise FrozenArtifactError(
            "E10 has already been reserved for this exact study protocol"
        ) from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "location": str(reservation.absolute()),
        "fingerprint": sha256_file(reservation),
    }, reservation


def _require_sha256(value: str, context: str) -> None:
    if not _SHA256.fullmatch(value):
        raise DataValidationError(f"{context} must be a lowercase SHA-256 fingerprint")


def _validate_component_selection(
    path: Path, contract: PhaseRunContract
) -> dict[tuple[str, str], dict[str, Any]]:
    if (
        path.is_symlink()
        or not path.is_dir()
        or {item.name for item in path.iterdir()}
        != {
            "manifest.json",
            "components",
        }
    ):
        raise DataValidationError("frozen component selection must be a strict directory bundle")
    component_root = path / "components"
    if component_root.is_symlink() or not component_root.is_dir():
        raise DataValidationError("frozen component selection has no regular component root")
    try:
        payload = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot read frozen component selection: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataValidationError("frozen component selection must be a mapping")
    digest = payload.pop("manifest_digest", None)
    if digest != stable_hash(payload):
        raise DataValidationError("frozen component-selection digest mismatch")
    if (
        set(payload)
        != {
            "schema_version",
            "study_protocol_digest",
            "phase",
            "components",
        }
        or payload.get("schema_version") != 3
    ):
        raise DataValidationError("frozen component selection has an invalid schema")
    components = payload["components"]
    if not isinstance(components, list):
        raise DataValidationError("frozen component selection components must be a list")
    observed: dict[tuple[str, str], dict[str, Any]] = {}
    for value in components:
        if not isinstance(value, Mapping) or set(value) != {
            "model_name",
            "method",
            "artifact_sha256",
            "component_path",
            "adaptive_policy",
            "adaptive_policy_digest",
        }:
            raise DataValidationError("frozen component descriptor is invalid")
        key = (str(value["model_name"]), str(value["method"]))
        fingerprint = str(value["artifact_sha256"])
        _require_sha256(fingerprint, "frozen component artifact")
        expected_relative = (
            "components/" + stable_hash({"model_name": key[0], "method": key[1]})[:16]
        )
        if value["component_path"] != expected_relative:
            raise DataValidationError("frozen component path is not canonical")
        component_directory = path / expected_relative
        if component_directory.is_symlink() or not component_directory.is_dir():
            raise DataValidationError("frozen component directory is invalid")
        policy_value = value["adaptive_policy"]
        policy_digest = value["adaptive_policy_digest"]
        if policy_value is None:
            if policy_digest is not None:
                raise DataValidationError("static component has an adaptive-policy digest")
            policy: dict[str, Any] | None = None
            expected_component_files = {"artifact"}
        elif isinstance(policy_value, Mapping):
            policy = AdaptivePolicySpec.from_dict(policy_value).to_dict()
            if policy_digest != stable_hash(policy):
                raise DataValidationError("frozen adaptive-policy digest mismatch")
            expected_component_files = {"artifact", "adaptive-policy.json"}
            try:
                policy_artifact = json.loads(
                    (component_directory / "adaptive-policy.json").read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise DataValidationError(
                    f"cannot read frozen adaptive-policy artifact: {exc}"
                ) from exc
            if policy_artifact != {**policy, "policy_digest": policy_digest}:
                raise DataValidationError(
                    "frozen adaptive-policy artifact differs from its component descriptor"
                )
        else:
            raise DataValidationError("frozen adaptive-policy descriptor is invalid")
        if {item.name for item in component_directory.iterdir()} != expected_component_files:
            raise DataValidationError("frozen component directory contains unexpected artifacts")
        artifact = component_directory / "artifact"
        _assert_regular_artifact(artifact, f"frozen component {key}")
        if sha256_path(artifact) != fingerprint:
            raise DataValidationError("packaged frozen component changed")
        if key in observed:
            raise DataValidationError(f"frozen component repeats {key}")
        observed[key] = {
            "artifact_sha256": fingerprint,
            "component_path": expected_relative,
            "adaptive_policy": policy,
            "adaptive_policy_digest": policy_digest,
        }
    expected: dict[tuple[str, str], dict[str, Any]] = {}
    for condition in contract.conditions:
        if condition.steering_method == "M0":
            continue
        assert condition.method_artifact_sha256 is not None
        key = (condition.model_name, condition.steering_method)
        policy = (
            condition.adaptive_policy.to_dict() if condition.adaptive_policy is not None else None
        )
        descriptor = {
            "artifact_sha256": condition.method_artifact_sha256,
            "component_path": (
                "components/" + stable_hash({"model_name": key[0], "method": key[1]})[:16]
            ),
            "adaptive_policy": policy,
            "adaptive_policy_digest": stable_hash(policy) if policy is not None else None,
        }
        previous = expected.setdefault(key, descriptor)
        if previous != descriptor:
            raise DataValidationError(f"condition matrix changes component identity for {key}")
    if (
        payload["study_protocol_digest"] != contract.study_protocol_digest
        or payload["phase"] != contract.phase.value
        or observed != expected
    ):
        raise DataValidationError("frozen component selection differs from the condition matrix")
    for key, descriptor in observed.items():
        method = key[1]
        artifact = path / str(descriptor["component_path"]) / "artifact"
        matching_conditions = tuple(
            condition
            for condition in contract.conditions
            if (condition.model_name, condition.steering_method) == key
        )
        if method in {"M1", "M2", "M4", "M5"}:
            from mfh.experiments.confirmatory_components import (
                load_confirmatory_fixed_component,
            )

            fixed_component = load_confirmatory_fixed_component(artifact)
            if fixed_component.fingerprint != descriptor["artifact_sha256"] or any(
                fixed_component.method != condition.steering_method
                or fixed_component.layer != condition.layer
                or fixed_component.site is not condition.site
                or fixed_component.token_scope is not condition.token_scope
                or fixed_component.standardized_alpha != condition.alpha
                or fixed_component.sparsity != condition.sparsity
                for condition in matching_conditions
            ):
                raise DataValidationError(f"frozen fixed component geometry differs for {key}")
        elif method == "M3":
            from mfh.experiments.confirmatory_components import (
                load_confirmatory_adaptive_component,
            )

            adaptive_component = load_confirmatory_adaptive_component(artifact)
            expected_prompt_hashes = {
                condition.system_prompt_id: condition.prompt_template_sha256
                for condition in matching_conditions
            }
            if (
                adaptive_component.fingerprint != descriptor["artifact_sha256"]
                or adaptive_component.model_name != key[0]
                or any(
                    adaptive_component.model_repository != condition.model_repository
                    or adaptive_component.model_revision != condition.model_revision
                    or adaptive_component.runtime is not condition.runtime
                    or adaptive_component.quantization != condition.quantization
                    or adaptive_component.model_num_layers != condition.model_num_layers
                    or condition.adaptive_policy is None
                    or condition.adaptive_policy.controller_artifact_sha256
                    != adaptive_component.fingerprint
                    for condition in matching_conditions
                )
                or dict(adaptive_component.prompt_hashes) != expected_prompt_hashes
            ):
                raise DataValidationError(f"frozen adaptive component identity differs for {key}")
    expected_directories = {
        value["component_path"].split("/", maxsplit=1)[1] for value in expected.values()
    }
    if {item.name for item in component_root.iterdir()} != expected_directories:
        raise DataValidationError("frozen component directories differ from the condition matrix")
    return observed


def _validate_e9_component_promotions(
    selection: Path,
    contract: PhaseRunContract,
    prerequisite_ledgers: Mapping[ExperimentPhase, Any],
) -> None:
    """Bind E9 components to completion-packaged winning selection artifacts."""

    if contract.phase is not ExperimentPhase.E9:
        return
    required = {
        ExperimentPhase.E3,
        ExperimentPhase.E4,
        ExperimentPhase.E5,
        ExperimentPhase.E7,
        ExperimentPhase.E8,
    }
    if not required <= set(prerequisite_ledgers):
        raise DataValidationError("E9 promotion validation lacks prerequisite ledgers")
    descriptors = _validate_component_selection(selection, contract)
    from mfh.experiments.confirmatory_components import (
        load_confirmatory_adaptive_component,
        load_confirmatory_fixed_component,
    )

    e4 = prerequisite_ledgers[ExperimentPhase.E4]
    e8 = prerequisite_ledgers[ExperimentPhase.E8]
    from mfh.experiments.e4_baselines import (
        load_e4_method_policy,
        load_e4_promotion_artifact,
    )
    from mfh.methods.protected import load_e8_operating_point_registry

    promotion = load_e4_promotion_artifact(
        e4.directory / "gate-artifacts" / "promotion_decision_frozen" / "promotion"
    )
    registry = load_e8_operating_point_registry(
        e8.directory
        / "gate-artifacts"
        / "matched_empirical_risk_or_coverage"
        / "operating-point-registry"
    )
    e4_conditions = {value.condition_id: value for value in e4.contract.conditions}
    e8_conditions = {value.condition_id: value for value in e8.contract.conditions}
    selected_e4 = tuple(
        e4_conditions[condition_id]
        for condition_id in promotion.selection_manifest["selected_condition_ids"]
    )
    selected_e8 = {
        (prompt, method): e8_conditions[condition_id]
        for prompt, methods in registry.condition_ids_by_prompt.items()
        for method, condition_id in methods.items()
    }

    def fixed_operating_point(condition: EvaluationCondition) -> tuple[object, ...]:
        return (
            condition.method_artifact_sha256,
            condition.layer,
            condition.site,
            condition.token_scope,
            condition.alpha,
            condition.sparsity,
        )

    fixed_winners = {
        method: {
            fixed_operating_point(condition)
            for (prompt, selected_method), condition in selected_e8.items()
            if selected_method == method and prompt in {"P0-neutral", "P2-calibrated-abstention"}
        }
        for method in {"M1", "M4", "M5"}
    }
    m2_policy = load_e4_method_policy(
        e4.directory / "gate-artifacts" / "promotion_decision_frozen" / "policy-m2"
    )
    fixed_winners["M2"] = {
        (
            m2_policy.implementation_artifact_sha256,
            condition.layer,
            condition.site,
            condition.token_scope,
            condition.alpha,
            condition.sparsity,
        )
        for condition in selected_e4
        if condition.steering_method == "M2"
    }

    def routed_operating_point(policy: AdaptivePolicySpec) -> tuple[object, ...]:
        return (
            policy.release_risk_threshold,
            policy.abstention_probability_threshold,
            policy.alpha_max,
            policy.alpha_beta,
            policy.sparsity,
            policy.candidate_layers,
            policy.candidate_sites,
            policy.candidate_token_scopes,
            policy.vector_count,
            policy.likely_unknown_risk_threshold,
            policy.alpha_mode,
            policy.alpha_risk_threshold,
        )

    for key, descriptor in descriptors.items():
        method = key[1]
        artifact = selection / str(descriptor["component_path"]) / "artifact"
        if method in fixed_winners:
            fixed_component = load_confirmatory_fixed_component(artifact)
            promoted_point = (
                fixed_component.source_artifact_sha256,
                fixed_component.layer,
                fixed_component.site,
                fixed_component.token_scope,
                fixed_component.standardized_alpha,
                fixed_component.sparsity,
            )
            if promoted_point not in fixed_winners[method]:
                raise DataValidationError(
                    f"E9 {method} component is not an exact promoted operating point"
                )
        elif method == "M3":
            adaptive_component = load_confirmatory_adaptive_component(artifact)
            for applied_prompt, fingerprint in adaptive_component.controller_fingerprints.items():
                source_prompt = adaptive_component.controller_source_prompt_ids[applied_prompt]
                selected = selected_e8.get((source_prompt, "M3"))
                selected_fingerprint = (
                    selected.adaptive_policy.controller_artifact_sha256
                    if selected is not None and selected.adaptive_policy is not None
                    else selected.method_artifact_sha256
                    if selected is not None
                    else None
                )
                if fingerprint != selected_fingerprint:
                    raise DataValidationError(
                        "E9 M3 controller is not an exact E8 operating-point winner"
                    )
                if selected is None or selected.adaptive_policy is None:
                    raise DataValidationError("E9 M3 controller has no selected E8 adaptive policy")
                matching_conditions = (
                    condition
                    for condition in contract.conditions
                    if condition.model_name == key[0]
                    and condition.steering_method == "M3"
                    and condition.system_prompt_id == applied_prompt
                )
                for condition in matching_conditions:
                    policy = condition.adaptive_policy
                    if policy is None or routed_operating_point(policy) != (
                        routed_operating_point(selected.adaptive_policy)
                    ):
                        raise DataValidationError(
                            "E9 M3 policy is not the exact E8 operating-point winner"
                        )
        else:  # pragma: no cover - component validator fixes the E9 method inventory
            raise DataValidationError(f"E9 component {method} has no promotion source")


def _reviewed_language_source_descriptor(
    source: Path,
    questions: tuple[Question, ...],
    *,
    artifact: str,
) -> dict[str, Any]:
    canonical = load_reviewed_language_suite(source)
    if canonical != questions:
        raise DataValidationError(
            "frozen language questions differ from the reviewed translation suite"
        )
    fingerprint = sha256_path(source)
    size = sum(item.stat().st_size for item in source.rglob("*") if item.is_file())
    return {
        "repository": "local/mfh-human-reviewed-triviaqa-languages-v1",
        "revision": fingerprint,
        "split": "evaluation",
        "artifact": artifact,
        "artifact_sha256": fingerprint,
        "artifact_size_bytes": size,
        "canonical_question_count": len(canonical),
        "human_reviewed": True,
    }


def _validate_partition_bound_source_membership(
    snapshot: SourceSnapshot,
    path: str | Path,
    questions: tuple[Question, ...],
    *,
    contract: PhaseRunContract,
    benchmark: str,
) -> None:
    """Allow only the contract's registered partition relabel over canonical rows."""

    partitions = {
        condition.partition for condition in contract.conditions if condition.benchmark == benchmark
    }
    if len(partitions) != 1:
        raise DataValidationError(f"{benchmark} must have one frozen evaluation partition")
    partition = next(iter(partitions))
    normalized = tuple(
        replace(question, split=snapshot.split) if question.split == partition else question
        for question in questions
    )
    validate_source_membership(snapshot, path, normalized)


def _validate_question_bundle(path: Path, contract: PhaseRunContract) -> None:
    if not path.is_dir() or path.is_symlink():
        raise DataValidationError("frozen question bundle must be a regular directory")
    expected_files = {f"{benchmark}.jsonl" for benchmark in contract.question_ids_by_benchmark}
    expected_files.update({"bundle-manifest.json", "source-artifacts"})
    if {value.name for value in path.iterdir()} != expected_files:
        raise DataValidationError("frozen question-bundle files differ from the phase benchmarks")
    source_root = path / "source-artifacts"
    if (
        source_root.is_symlink()
        or not source_root.is_dir()
        or {value.name for value in source_root.iterdir()}
        != set(contract.question_ids_by_benchmark)
    ):
        raise DataValidationError("frozen question-bundle source directories differ")
    try:
        manifest = json.loads((path / "bundle-manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot read frozen question-bundle manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise DataValidationError("frozen question-bundle manifest must be a mapping")
    digest = manifest.pop("manifest_digest", None)
    if digest != stable_hash(manifest):
        raise DataValidationError("frozen question-bundle manifest digest mismatch")
    if (
        set(manifest)
        != {
            "schema_version",
            "study_protocol_digest",
            "phase",
            "benchmarks",
        }
        or manifest.get("schema_version") != 2
    ):
        raise DataValidationError("frozen question-bundle manifest has an invalid schema")
    descriptors = manifest["benchmarks"]
    if not isinstance(descriptors, Mapping) or set(descriptors) != set(
        contract.question_ids_by_benchmark
    ):
        raise DataValidationError("frozen question-bundle benchmark manifest differs")
    observed_descriptors: dict[str, dict[str, Any]] = {}
    for benchmark, expected_ids in contract.question_ids_by_benchmark.items():
        question_path = path / f"{benchmark}.jsonl"
        questions = tuple(read_questions(question_path))
        ids = tuple(value.question_id for value in questions)
        if ids != expected_ids or any(value.benchmark != benchmark for value in questions):
            raise DataValidationError(
                f"frozen question bundle differs from exact {benchmark} IDs or labels"
            )
        benchmark_source_directory = source_root / benchmark
        reviewed_language = benchmark_source_directory / "reviewed-language-suite"
        sampling: Mapping[str, Any] | None = None
        if benchmark == "language_consistency" and reviewed_language.is_dir():
            if benchmark_source_directory.is_symlink() or {
                value.name for value in benchmark_source_directory.iterdir()
            } != {"reviewed-language-suite"}:
                raise DataValidationError("reviewed language source directory is invalid")
            source_descriptor = _reviewed_language_source_descriptor(
                reviewed_language,
                questions,
                artifact=("source-artifacts/language_consistency/reviewed-language-suite"),
            )
        else:
            try:
                snapshot = SOURCE_SNAPSHOTS[benchmark]
            except KeyError as exc:
                raise DataValidationError(
                    f"no immutable source snapshot is registered for {benchmark}"
                ) from exc
            if contract.phase in {ExperimentPhase.E9, ExperimentPhase.E10} and not (
                snapshot.confirmatory_eligible
            ):
                raise DataValidationError(
                    f"{benchmark} source is a development smoke suite, not a confirmatory source"
                )
            if (
                benchmark_source_directory.is_symlink()
                or not benchmark_source_directory.is_dir()
                or {value.name for value in benchmark_source_directory.iterdir()}
                != {snapshot.artifact_name}
            ):
                raise DataValidationError(
                    f"frozen {benchmark} source-artifact directory differs from its snapshot"
                )
            source_path = benchmark_source_directory / snapshot.artifact_name
            verify_source_artifact(snapshot, source_path)
            _validate_partition_bound_source_membership(
                snapshot,
                source_path,
                questions,
                contract=contract,
                benchmark=benchmark,
            )
            if benchmark == "mmlu_pro":
                expected_questions, sampling = select_mmlu_pro_stratified(
                    tuple(iter_source_questions(snapshot, source_path))
                )
                if questions != expected_questions:
                    raise DataValidationError(
                        "MMLU-Pro questions differ from the frozen stratified sample"
                    )
            source_descriptor = {
                "repository": snapshot.repository,
                "revision": snapshot.revision,
                "split": snapshot.split,
                "artifact": f"source-artifacts/{benchmark}/{snapshot.artifact_name}",
                "artifact_sha256": snapshot.artifact_sha256,
                "artifact_size_bytes": snapshot.artifact_size_bytes,
                "canonical_question_count": snapshot.canonical_question_count,
            }
        partitions = {
            value.partition for value in contract.conditions if value.benchmark == benchmark
        }
        if len(partitions) != 1:
            raise DataValidationError(f"{benchmark} must have one frozen evaluation partition")
        observed_descriptors[benchmark] = {
            "filename": question_path.name,
            "partition": next(iter(partitions)),
            "question_count": len(questions),
            "question_ids_sha256": stable_hash(list(ids)),
            "questions_sha256": sha256_file(question_path),
            "source": source_descriptor,
            **({"sampling": dict(sampling)} if sampling is not None else {}),
        }
    if (
        manifest["study_protocol_digest"] != contract.study_protocol_digest
        or manifest["phase"] != contract.phase.value
        or dict(descriptors) != observed_descriptors
    ):
        raise DataValidationError("frozen question-bundle manifest differs from its contents")


def write_frozen_question_bundle(
    directory: str | Path,
    contract: PhaseRunContract,
    questions_by_benchmark: Mapping[str, Iterable[Question]],
    *,
    source_artifacts: Mapping[str, str | Path],
) -> str:
    """Publish the exact provenance-bound confirmatory question set atomically."""

    destination = validate_active_study_artifact_paths({"frozen question bundle": directory})[
        "frozen question bundle"
    ]
    if destination.exists():
        raise FrozenArtifactError(f"refusing to overwrite question bundle: {destination}")
    if set(questions_by_benchmark) != set(contract.question_ids_by_benchmark):
        raise DataValidationError("question-bundle benchmarks differ from the phase contract")
    if set(source_artifacts) != set(contract.question_ids_by_benchmark):
        raise DataValidationError(
            "question-bundle source artifacts differ from the phase benchmarks"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        descriptors: dict[str, dict[str, Any]] = {}
        (stage / "source-artifacts").mkdir()
        for benchmark in sorted(questions_by_benchmark):
            path = stage / f"{benchmark}.jsonl"
            write_questions(path, tuple(questions_by_benchmark[benchmark]))
            questions = tuple(read_questions(path))
            source = Path(source_artifacts[benchmark]).resolve()
            sampling: Mapping[str, Any] | None = None
            if benchmark == "language_consistency" and source.is_dir():
                packaged_source = stage / "source-artifacts" / benchmark / "reviewed-language-suite"
                source_fingerprint = sha256_path(source)
                _copy_frozen_artifact(source, packaged_source, source_fingerprint)
                source_descriptor = _reviewed_language_source_descriptor(
                    packaged_source,
                    questions,
                    artifact=("source-artifacts/language_consistency/reviewed-language-suite"),
                )
            else:
                try:
                    snapshot = SOURCE_SNAPSHOTS[benchmark]
                except KeyError as exc:
                    raise DataValidationError(
                        f"no immutable source snapshot is registered for {benchmark}"
                    ) from exc
                if contract.phase in {ExperimentPhase.E9, ExperimentPhase.E10} and not (
                    snapshot.confirmatory_eligible
                ):
                    raise DataValidationError(
                        f"{benchmark} source is a development smoke suite, "
                        "not a confirmatory source"
                    )
                verified_source = verify_source_artifact(snapshot, source)
                packaged_source = stage / "source-artifacts" / benchmark / snapshot.artifact_name
                packaged_source.parent.mkdir()
                _copy_frozen_artifact(
                    verified_source,
                    packaged_source,
                    snapshot.artifact_sha256,
                )
                _validate_partition_bound_source_membership(
                    snapshot,
                    packaged_source,
                    questions,
                    contract=contract,
                    benchmark=benchmark,
                )
                if benchmark == "mmlu_pro":
                    expected_questions, sampling = select_mmlu_pro_stratified(
                        tuple(iter_source_questions(snapshot, packaged_source))
                    )
                    if questions != expected_questions:
                        raise DataValidationError(
                            "MMLU-Pro questions differ from the frozen stratified sample"
                        )
                source_descriptor = {
                    "repository": snapshot.repository,
                    "revision": snapshot.revision,
                    "split": snapshot.split,
                    "artifact": (f"source-artifacts/{benchmark}/{snapshot.artifact_name}"),
                    "artifact_sha256": snapshot.artifact_sha256,
                    "artifact_size_bytes": snapshot.artifact_size_bytes,
                    "canonical_question_count": snapshot.canonical_question_count,
                }
            identifiers = tuple(value.question_id for value in questions)
            partitions = {
                value.partition for value in contract.conditions if value.benchmark == benchmark
            }
            if len(partitions) != 1:
                raise DataValidationError(f"{benchmark} must have one frozen evaluation partition")
            descriptors[benchmark] = {
                "filename": path.name,
                "partition": next(iter(partitions)),
                "question_count": len(questions),
                "question_ids_sha256": stable_hash(list(identifiers)),
                "questions_sha256": sha256_file(path),
                "source": source_descriptor,
                **({"sampling": dict(sampling)} if sampling is not None else {}),
            }
        body = {
            "schema_version": 2,
            "study_protocol_digest": contract.study_protocol_digest,
            "phase": contract.phase.value,
            "benchmarks": descriptors,
        }
        (stage / "bundle-manifest.json").write_text(
            json.dumps(
                {**body, "manifest_digest": stable_hash(body)},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        _validate_question_bundle(stage, contract)
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return sha256_path(destination)


def validate_frozen_question_bundle(directory: str | Path, contract: PhaseRunContract) -> str:
    """Validate and fingerprint one exact source-backed phase question schedule."""

    source = Path(directory)
    _validate_question_bundle(source, contract)
    return sha256_path(source)


def write_side_effect_evaluation_bundle(
    directory: str | Path,
    contract: PhaseRunContract,
    questions_by_benchmark: Mapping[str, Iterable[Question]],
    *,
    source_artifacts: Mapping[str, str | Path],
    scorer_execution_public_key: str,
    ifeval_evaluator: str | Path,
) -> str:
    """Freeze scorer identity and exact source-backed E7/E8 question schedule."""

    from mfh.evaluation.ifeval import validate_ifeval_evaluator
    from mfh.evaluation.side_effects import write_side_effect_scorer_spec
    from mfh.evaluation.strongreject import materialize_strongreject_grader

    directory = validate_active_study_artifact_paths({"side-effect evaluation bundle": directory})[
        "side-effect evaluation bundle"
    ]
    if contract.phase not in {ExperimentPhase.E7, ExperimentPhase.E8}:
        raise DataValidationError("side-effect evaluation bundles are E7/E8-only")
    destination = Path(directory)
    if destination.exists():
        raise FrozenArtifactError(
            f"refusing to overwrite side-effect evaluation bundle: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        write_side_effect_scorer_spec(
            stage / "side-effect-scorer.json",
            execution_public_key=scorer_execution_public_key,
        )
        write_frozen_question_bundle(
            stage / "questions",
            contract,
            questions_by_benchmark,
            source_artifacts=source_artifacts,
        )
        evaluator_source = Path(ifeval_evaluator).resolve()
        evaluator_sha = validate_ifeval_evaluator(evaluator_source)
        _copy_frozen_artifact(
            evaluator_source,
            stage / "ifeval-evaluator",
            evaluator_sha,
        )
        strongreject_sha = materialize_strongreject_grader(stage / "strongreject-grader")
        body = {
            "schema_version": 4,
            "phase": contract.phase.value,
            "schedule_digest": _side_effect_schedule_digest(contract),
            "scorer_sha256": sha256_file(stage / "side-effect-scorer.json"),
            "questions_sha256": sha256_path(stage / "questions"),
            "ifeval_evaluator_sha256": evaluator_sha,
            "strongreject_grader_sha256": strongreject_sha,
        }
        (stage / "bundle-manifest.json").write_text(
            json.dumps(
                {**body, "manifest_digest": stable_hash(body)},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        validate_side_effect_evaluation_bundle(stage, contract)
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return sha256_path(destination)


def validate_side_effect_evaluation_bundle(
    directory: str | Path, contract: PhaseRunContract
) -> str:
    """Revalidate an E7/E8 scorer and exact question/source bundle."""

    source = Path(directory)
    if (
        source.is_symlink()
        or not source.is_dir()
        or {value.name for value in source.iterdir()}
        != {
            "side-effect-scorer.json",
            "questions",
            "ifeval-evaluator",
            "strongreject-grader",
            "bundle-manifest.json",
        }
    ):
        raise FrozenArtifactError("side-effect evaluation bundle inventory differs")
    from mfh.evaluation.ifeval import validate_ifeval_evaluator
    from mfh.evaluation.strongreject import validate_strongreject_grader

    load_side_effect_scorer_spec(source)
    evaluator_sha = validate_ifeval_evaluator(source / "ifeval-evaluator")
    strongreject_sha = validate_strongreject_grader(source / "strongreject-grader")
    _validate_question_bundle(source / "questions", contract)
    try:
        manifest = json.loads((source / "bundle-manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read side-effect evaluation bundle: {exc}") from exc
    if not isinstance(manifest, dict):
        raise FrozenArtifactError("side-effect evaluation manifest is invalid")
    digest = manifest.pop("manifest_digest", None)
    expected = {
        "schema_version": 4,
        "phase": contract.phase.value,
        "schedule_digest": _side_effect_schedule_digest(contract),
        "scorer_sha256": sha256_file(source / "side-effect-scorer.json"),
        "questions_sha256": sha256_path(source / "questions"),
        "ifeval_evaluator_sha256": evaluator_sha,
        "strongreject_grader_sha256": strongreject_sha,
    }
    if manifest != expected or digest != stable_hash(expected):
        raise FrozenArtifactError("side-effect evaluation manifest differs")
    return sha256_path(source)


def _side_effect_schedule_digest(contract: PhaseRunContract) -> str:
    """Bind an E7/E8 scorer bundle without hashing the bundle into itself.

    ``frozen_side_effect_scorers`` is an input to the final phase contract, while
    the bundle also contains that phase's exact question schedule.  Binding the
    full contract digest here would create an unsatisfiable hash cycle.  This
    projection retains every execution-schedule fact and deliberately excludes
    only input/prerequisite fingerprints, terminal gate declarations, and each
    condition's method-artifact fingerprint.  That last field is the sole
    cyclic value for E7/E8: the side-effect bundle is itself a frozen input to
    the eventual promoted-method contract.  Method identity, geometry, policy,
    comparison group, and condition multiplicity remain bound.
    """

    if contract.phase not in {ExperimentPhase.E7, ExperimentPhase.E8}:
        raise DataValidationError("side-effect schedules are E7/E8-only")
    return stable_hash(
        {
            "schema_version": 1,
            "phase": contract.phase.value,
            "study_protocol_digest": contract.study_protocol_digest,
            "conditions": sorted(
                (
                    {
                        name: value
                        for name, value in condition.to_dict().items()
                        if name != "method_artifact_sha256"
                    }
                    for condition in contract.conditions
                ),
                key=canonical_json,
            ),
            "question_ids_by_benchmark": {
                name: list(values)
                for name, values in sorted(contract.question_ids_by_benchmark.items())
            },
        }
    )


def write_frozen_component_selection(
    path: str | Path,
    contract: PhaseRunContract,
    component_artifacts: Mapping[tuple[str, str], str | Path],
) -> str:
    """Freeze selected method artifacts against the confirmatory condition matrix."""

    destination = validate_active_study_artifact_paths({"frozen component selection": path})[
        "frozen component selection"
    ]
    if destination.exists():
        raise FrozenArtifactError(f"refusing to overwrite component selection: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected: dict[tuple[str, str], dict[str, Any]] = {}
    for condition in contract.conditions:
        if condition.steering_method == "M0":
            continue
        assert condition.method_artifact_sha256 is not None
        key = (condition.model_name, condition.steering_method)
        policy = (
            condition.adaptive_policy.to_dict() if condition.adaptive_policy is not None else None
        )
        descriptor = {
            "artifact_sha256": condition.method_artifact_sha256,
            "component_path": (
                "components/" + stable_hash({"model_name": key[0], "method": key[1]})[:16]
            ),
            "adaptive_policy": policy,
            "adaptive_policy_digest": stable_hash(policy) if policy is not None else None,
        }
        previous = expected.setdefault(key, descriptor)
        if previous != descriptor:
            raise DataValidationError(f"condition matrix changes component identity for {key}")
    if set(component_artifacts) != set(expected):
        raise DataValidationError("selected component paths differ from the condition matrix")
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        (stage / "components").mkdir()
        components: list[dict[str, Any]] = []
        for key in sorted(expected):
            source = Path(component_artifacts[key]).resolve()
            _assert_regular_artifact(source, f"selected component {key}")
            fingerprint = str(expected[key]["artifact_sha256"])
            if sha256_path(source) != fingerprint:
                raise DataValidationError(f"selected component changed for {key}")
            component_directory = stage / str(expected[key]["component_path"])
            component_directory.mkdir()
            _copy_frozen_artifact(
                source,
                component_directory / "artifact",
                fingerprint,
            )
            policy = expected[key]["adaptive_policy"]
            if isinstance(policy, Mapping):
                policy_body = dict(policy)
                (component_directory / "adaptive-policy.json").write_text(
                    json.dumps(
                        {
                            **policy_body,
                            "policy_digest": stable_hash(policy_body),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            components.append(
                {
                    "model_name": key[0],
                    "method": key[1],
                    **expected[key],
                }
            )
        body = {
            "schema_version": 3,
            "study_protocol_digest": contract.study_protocol_digest,
            "phase": contract.phase.value,
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
        _validate_component_selection(stage, contract)
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return sha256_path(destination)


@dataclass(frozen=True, slots=True)
class EvaluationCondition:
    phase: ExperimentPhase
    benchmark: str
    partition: str
    model_name: str
    model_repository: str
    model_revision: str
    runtime: Runtime
    quantization: str
    model_num_layers: int
    system_prompt_id: str
    prompt_template_sha256: str
    steering_method: str
    method_artifact_sha256: str | None
    layer: int | None
    site: ActivationSite | None
    token_scope: TokenScope | None
    alpha: float
    sparsity: float | None
    seed: int
    study_protocol_digest: str
    adaptive_policy: AdaptivePolicySpec | None = None
    comparison_group: str = "primary"

    def __post_init__(self) -> None:
        for name in (
            "benchmark",
            "partition",
            "model_name",
            "model_repository",
            "quantization",
            "system_prompt_id",
            "steering_method",
            "comparison_group",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise DataValidationError(f"condition {name} must be non-empty")
            object.__setattr__(self, name, value)
        if self.runtime is not Runtime.SYNTHETIC and not _REVISION.fullmatch(self.model_revision):
            raise DataValidationError("condition requires an immutable model revision")
        if self.model_num_layers <= 0:
            raise DataValidationError("condition model layer count must be positive")
        _require_sha256(self.prompt_template_sha256, "prompt-template identity")
        _require_sha256(self.study_protocol_digest, "study-protocol identity")
        if self.method_artifact_sha256 is not None:
            _require_sha256(self.method_artifact_sha256, "method-artifact identity")
        if self.layer is not None and self.layer < 0:
            raise DataValidationError("condition layer cannot be negative")
        if not math.isfinite(self.alpha):
            raise DataValidationError("condition alpha must be finite")
        if self.sparsity is not None and not 0 < self.sparsity <= 1:
            raise DataValidationError("condition sparsity must be in (0, 1]")
        if self.seed < 0:
            raise DataValidationError("condition seed cannot be negative")
        adaptive = self.steering_method in {"M3", "M6", "ACT-or-SADI"}
        if adaptive != (self.adaptive_policy is not None):
            raise DataValidationError("M3/M6 conditions require exactly one frozen adaptive policy")
        if adaptive:
            assert self.adaptive_policy is not None
            if (
                self.layer,
                self.site,
                self.token_scope,
                self.alpha,
                self.sparsity,
            ) != (None, None, None, 0.0, None):
                raise DataValidationError(
                    "adaptive condition geometry must come only from its frozen policy"
                )
            if self.adaptive_policy.schema_version == 1:
                assert self.adaptive_policy.layer is not None
                if self.adaptive_policy.layer >= self.model_num_layers:
                    raise DataValidationError(
                        "adaptive policy layer is outside the pinned model architecture"
                    )
            elif any(
                layer >= self.model_num_layers for layer in self.adaptive_policy.candidate_layers
            ):
                raise DataValidationError(
                    "adaptive policy candidate layer is outside the pinned model architecture"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "benchmark": self.benchmark,
            "partition": self.partition,
            "model_name": self.model_name,
            "model_repository": self.model_repository,
            "model_revision": self.model_revision,
            "runtime": self.runtime.value,
            "quantization": self.quantization,
            "model_num_layers": self.model_num_layers,
            "system_prompt_id": self.system_prompt_id,
            "prompt_template_sha256": self.prompt_template_sha256,
            "steering_method": self.steering_method,
            "method_artifact_sha256": self.method_artifact_sha256,
            "layer": self.layer,
            "site": self.site.value if self.site is not None else None,
            "token_scope": self.token_scope.value if self.token_scope is not None else None,
            "alpha": self.alpha,
            "sparsity": self.sparsity,
            "seed": self.seed,
            "study_protocol_digest": self.study_protocol_digest,
            "adaptive_policy": (
                self.adaptive_policy.to_dict() if self.adaptive_policy is not None else None
            ),
            "comparison_group": self.comparison_group,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvaluationCondition:
        expected = {
            "phase",
            "benchmark",
            "partition",
            "model_name",
            "model_repository",
            "model_revision",
            "runtime",
            "quantization",
            "model_num_layers",
            "system_prompt_id",
            "prompt_template_sha256",
            "steering_method",
            "method_artifact_sha256",
            "layer",
            "site",
            "token_scope",
            "alpha",
            "sparsity",
            "seed",
            "study_protocol_digest",
            "adaptive_policy",
            "comparison_group",
        }
        if set(value) != expected:
            raise DataValidationError("evaluation-condition keys differ from schema version 1")
        adaptive_policy_value = value["adaptive_policy"]
        if adaptive_policy_value is not None and not isinstance(adaptive_policy_value, Mapping):
            raise DataValidationError("evaluation-condition adaptive policy is invalid")
        return cls(
            phase=ExperimentPhase(value["phase"]),
            benchmark=str(value["benchmark"]),
            partition=str(value["partition"]),
            model_name=str(value["model_name"]),
            model_repository=str(value["model_repository"]),
            model_revision=str(value["model_revision"]),
            runtime=Runtime(value["runtime"]),
            quantization=str(value["quantization"]),
            model_num_layers=int(value["model_num_layers"]),
            system_prompt_id=str(value["system_prompt_id"]),
            prompt_template_sha256=str(value["prompt_template_sha256"]),
            steering_method=str(value["steering_method"]),
            method_artifact_sha256=(
                str(value["method_artifact_sha256"])
                if value["method_artifact_sha256"] is not None
                else None
            ),
            layer=int(value["layer"]) if value["layer"] is not None else None,
            site=ActivationSite(value["site"]) if value["site"] is not None else None,
            token_scope=(
                TokenScope(value["token_scope"]) if value["token_scope"] is not None else None
            ),
            alpha=float(value["alpha"]),
            sparsity=float(value["sparsity"]) if value["sparsity"] is not None else None,
            seed=int(value["seed"]),
            study_protocol_digest=str(value["study_protocol_digest"]),
            adaptive_policy=(
                AdaptivePolicySpec.from_dict(adaptive_policy_value)
                if isinstance(adaptive_policy_value, Mapping)
                else None
            ),
            comparison_group=str(value["comparison_group"]),
        )

    @property
    def condition_id(self) -> str:
        return stable_hash(self.to_dict())

    def validate_record(
        self,
        record: GenerationRecord,
        *,
        pending_side_effects: bool = False,
    ) -> None:
        expected_common = (
            self.benchmark,
            self.model_repository,
            self.model_revision,
            self.runtime,
            self.quantization,
            self.system_prompt_id,
            self.steering_method,
            self.seed,
            self.condition_id,
        )
        observed_common = (
            record.benchmark,
            record.model_repository,
            record.model_revision,
            record.runtime,
            record.quantization,
            record.system_prompt_id,
            record.steering_method,
            record.seed,
            record.condition_id,
        )
        if observed_common != expected_common:
            raise DataValidationError(
                f"record {record.question_id!r} differs from condition {self.condition_id}"
            )
        adaptive = self.steering_method in {"M3", "M6", "ACT-or-SADI"}
        if not adaptive and (
            record.layer,
            record.site,
            record.token_scope,
            record.alpha,
            record.sparsity,
        ) != (self.layer, self.site, self.token_scope, self.alpha, self.sparsity):
            raise DataValidationError(
                f"record {record.question_id!r} differs from fixed intervention condition"
            )
        if adaptive:
            scores = record.controller_scores
            if (
                set(scores) != {"C", "I", "A"}
                or any(value < 0 or value > 1 for value in scores.values())
                or not math.isclose(sum(scores.values()), 1.0, abs_tol=1e-5)
            ):
                raise DataValidationError("adaptive records require calibrated C/I/A probabilities")
            policy = self.adaptive_policy
            assert policy is not None
            if policy.schema_version == 2 and self.phase in {
                ExperimentPhase.E4,
                ExperimentPhase.E6,
                ExperimentPhase.E8,
                ExperimentPhase.E9,
                ExperimentPhase.E10,
            }:
                _validate_adaptive_controller_evidence(
                    record.metadata.get("adaptive_controller_evidence"),
                    policy=policy,
                )
            if policy.schema_version == 2:
                assert policy.likely_unknown_risk_threshold is not None
                assert policy.alpha_risk_threshold is not None
                if self.phase is ExperimentPhase.E5 and self.steering_method == "M3":
                    expected_action = (
                        "release"
                        if policy.alpha_mode == "risk_gated_hard_threshold"
                        and scores["I"] < policy.alpha_risk_threshold
                        else "intervene"
                    )
                elif (
                    self.steering_method == "M6"
                    and scores["I"] >= policy.likely_unknown_risk_threshold
                ):
                    expected_action = "abstain"
                elif self.steering_method == "M6":
                    expected_action = (
                        "release" if scores["I"] < policy.release_risk_threshold else "intervene"
                    )
                elif (
                    scores["I"] <= policy.release_risk_threshold
                    or (
                        self.steering_method == "M3"
                        and scores["A"] >= policy.abstention_probability_threshold
                    )
                    or (
                        policy.alpha_mode == "risk_gated_hard_threshold"
                        and scores["I"] < policy.alpha_risk_threshold
                    )
                ):
                    expected_action = "release"
                else:
                    expected_action = "intervene"
            elif (
                self.steering_method == "M6"
                and scores["A"] >= policy.abstention_probability_threshold
            ):
                expected_action = "abstain"
            elif scores["I"] <= policy.release_risk_threshold:
                expected_action = "release"
            else:
                expected_action = "intervene"
            action = record.metadata.get("policy_action")
            if action != expected_action:
                raise DataValidationError(
                    "adaptive action differs from the frozen risk-threshold policy"
                )
            output_action = record.metadata.get("output_action")
            if self.steering_method == "M6":
                post_scores = record.metadata.get("post_controller_scores")
                if action == "intervene" or (
                    action == "release" and self.phase is ExperimentPhase.E10
                ):
                    if (
                        not isinstance(post_scores, Mapping)
                        or set(post_scores) != {"C", "I", "A"}
                        or any(
                            isinstance(value, bool)
                            or not isinstance(value, int | float)
                            or not math.isfinite(float(value))
                            or not 0 <= float(value) <= 1
                            for value in post_scores.values()
                        )
                        or not math.isclose(
                            sum(float(value) for value in post_scores.values()),
                            1.0,
                            abs_tol=1e-5,
                        )
                    ):
                        raise DataValidationError(
                            "generated M6 records require calibrated post-token scores"
                        )
                    if output_action not in {"release", "abstain"}:
                        raise DataValidationError("M6 output action is invalid")
                elif action == "abstain":
                    if post_scores is not None:
                        raise DataValidationError(
                            "initially abstained M6 rows cannot have post-token scores"
                        )
                    expected_output = "abstain" if action == "abstain" else "release"
                    if output_action != expected_output:
                        raise DataValidationError(
                            "M6 output action differs from its frozen initial policy"
                        )
                elif post_scores is not None or output_action != "release":
                    raise DataValidationError(
                        "legacy M6 release rows cannot declare residual re-evaluation"
                    )
                if action == "abstain" and output_action != "abstain":
                    raise DataValidationError("M6 likely-unknown rows must abstain")
                if output_action == "abstain" and record.outcome.value != "A":
                    raise DataValidationError("an M6 abstained output must have outcome A")
            elif (
                output_action is not None
                or record.metadata.get("post_controller_scores") is not None
            ):
                raise DataValidationError("only M6 records may declare re-evaluation fields")
            trace = record.metadata.get("intervention_trace")
            trace_digest = record.metadata.get("intervention_trace_digest")
            if action == "intervene":
                alpha_threshold = (
                    policy.alpha_risk_threshold
                    if policy.schema_version == 2
                    else policy.release_risk_threshold
                )
                assert alpha_threshold is not None
                expected_alpha = (
                    policy.alpha_max
                    if policy.schema_version == 2 and policy.alpha_mode == "fixed"
                    else policy.alpha_max
                    / (1.0 + math.exp(-policy.alpha_beta * (scores["I"] - alpha_threshold)))
                )
                alpha_rel_tol = (
                    1e-6
                    if self.phase is ExperimentPhase.E5 and policy.schema_version == 2
                    else 1e-12
                )
                alpha_abs_tol = (
                    1e-7
                    if self.phase is ExperimentPhase.E5 and policy.schema_version == 2
                    else 1e-12
                )
                if policy.schema_version == 1:
                    valid_geometry = (
                        type(record.layer) is int
                        and record.layer == policy.layer
                        and record.site is policy.site
                        and record.token_scope is policy.token_scope
                    )
                else:
                    valid_geometry = (
                        type(record.layer) is int
                        and record.layer in policy.candidate_layers
                        and record.site in policy.candidate_sites
                        and record.token_scope in policy.candidate_token_scopes
                    )
                if (
                    not valid_geometry
                    or record.sparsity != policy.sparsity
                    or not math.isclose(
                        record.alpha,
                        expected_alpha,
                        rel_tol=alpha_rel_tol,
                        abs_tol=alpha_abs_tol,
                    )
                ):
                    raise DataValidationError(
                        "adaptive intervention differs from frozen alpha and geometry"
                    )
                assert type(record.layer) is int
                assert isinstance(record.site, ActivationSite)
                assert isinstance(record.token_scope, TokenScope)
                trace_keys = {
                    "layer",
                    "site",
                    "token_scope",
                    "alpha",
                    "sparsity",
                    "applied_tokens",
                    "applied_token_indices",
                    "activation_delta_norm",
                    "direction_sha256",
                    "pre_activation_sha256",
                    "post_activation_sha256",
                    "delta_sha256",
                }
                if policy.schema_version == 2:
                    trace_keys |= {
                        "direction_norm",
                        "controller_artifact_sha256",
                        "router_weights",
                        "router_weights_sha256",
                    }
                if not isinstance(trace, Mapping) or set(trace) != trace_keys:
                    raise DataValidationError(
                        "adaptive intervention requires a strict execution trace"
                    )
                applied_tokens = trace["applied_tokens"]
                applied_token_indices = trace["applied_token_indices"]
                delta_norm = trace["activation_delta_norm"]
                if policy.schema_version == 1:
                    assert policy.direction_norm is not None
                    direction_norm = policy.direction_norm
                    direction_sha256 = policy.direction_sha256
                    valid_routing = True
                else:
                    raw_direction_norm = trace["direction_norm"]
                    direction_sha256 = trace["direction_sha256"]
                    router_weights = trace["router_weights"]
                    direction_norm = (
                        float(raw_direction_norm)
                        if not isinstance(raw_direction_norm, bool)
                        and isinstance(raw_direction_norm, int | float)
                        else math.nan
                    )
                    valid_routing = (
                        math.isfinite(direction_norm)
                        and direction_norm > 0
                        and isinstance(direction_sha256, str)
                        and bool(_SHA256.fullmatch(direction_sha256))
                        and trace["controller_artifact_sha256"] == policy.controller_artifact_sha256
                        and isinstance(router_weights, list)
                        and len(router_weights) == policy.vector_count
                        and all(
                            not isinstance(value, bool)
                            and isinstance(value, int | float)
                            and math.isfinite(float(value))
                            and float(value) >= 0
                            for value in router_weights
                        )
                        and math.isclose(
                            sum(float(value) for value in router_weights),
                            1.0,
                            rel_tol=1e-9,
                            abs_tol=1e-12,
                        )
                        and trace["router_weights_sha256"] == stable_hash(router_weights)
                    )
                if record.token_scope is TokenScope.FINAL_PROMPT:
                    expected_token_indices = [-1]
                else:
                    scope_limit = {
                        TokenScope.FIRST_GENERATED: 1,
                        TokenScope.FIRST_FOUR: 4,
                        TokenScope.FIRST_EIGHT: 8,
                        TokenScope.ALL_GENERATED: record.output_tokens,
                        TokenScope.EXPONENTIAL_DECAY: record.output_tokens,
                    }[record.token_scope]
                    expected_token_indices = list(range(min(scope_limit, record.output_tokens)))
                expected_delta_norm = (
                    abs(record.alpha) * direction_norm * math.sqrt(len(expected_token_indices))
                )
                expected_trace = {
                    "layer": record.layer,
                    "site": record.site.value,
                    "token_scope": record.token_scope.value,
                    "alpha": record.alpha,
                    "sparsity": record.sparsity,
                    "direction_sha256": direction_sha256,
                }
                if (
                    not valid_routing
                    or any(trace[key] != value for key, value in expected_trace.items())
                    or isinstance(applied_tokens, bool)
                    or not isinstance(applied_tokens, int)
                    or applied_tokens != len(expected_token_indices)
                    or applied_tokens <= 0
                    or applied_token_indices != expected_token_indices
                    or isinstance(delta_norm, bool)
                    or not isinstance(delta_norm, int | float)
                    or not math.isfinite(float(delta_norm))
                    or not math.isclose(
                        float(delta_norm),
                        expected_delta_norm,
                        rel_tol=0.025,
                        abs_tol=1e-6,
                    )
                    or any(
                        not isinstance(trace[name], str) or not _SHA256.fullmatch(trace[name])
                        for name in (
                            "pre_activation_sha256",
                            "post_activation_sha256",
                            "delta_sha256",
                        )
                    )
                    or trace["pre_activation_sha256"] == trace["post_activation_sha256"]
                    or trace_digest != stable_hash(dict(trace))
                ):
                    raise DataValidationError(
                        "adaptive intervention trace does not prove a material executed edit"
                    )
            elif (
                record.alpha != 0
                or record.layer is not None
                or record.site is not None
                or record.token_scope is not None
                or record.sparsity is not None
                or trace is not None
                or trace_digest is not None
            ):
                raise DataValidationError(
                    "an adaptive no-intervention action must have exactly zero intervention"
                )
            if record.metadata.get("policy_decision_digest") != (
                adaptive_policy_decision_digest(
                    record,
                    policy=policy,
                    policy_action=str(action),
                    output_action=str(output_action) if output_action is not None else None,
                )
            ):
                raise DataValidationError(
                    "adaptive policy-decision fingerprint differs from the executed action"
                )
            signature = record.metadata.get("execution_receipt_signature")
            if not isinstance(signature, str) or not re.fullmatch(r"[0-9a-f]{128}", signature):
                raise DataValidationError(
                    "adaptive record lacks a runtime execution-receipt signature"
                )
            try:
                public_key = Ed25519PublicKey.from_public_bytes(
                    bytes.fromhex(policy.execution_public_key)
                )
                public_key.verify(
                    bytes.fromhex(signature),
                    canonical_json(adaptive_execution_receipt_body(record, policy=policy)).encode(
                        "utf-8"
                    ),
                )
            except (InvalidSignature, ValueError) as exc:
                raise DataValidationError(
                    "adaptive execution receipt was not signed by the frozen runtime key"
                ) from exc
        if not _SHA256.fullmatch(record.rendered_prompt_hash):
            raise DataValidationError("run-ledger records require a rendered-prompt SHA-256")
        required_metadata = {
            "phase": self.phase.value,
            "partition": self.partition,
            "prompt_template_sha256": self.prompt_template_sha256,
            "study_protocol_digest": self.study_protocol_digest,
        }
        if self.method_artifact_sha256 is not None:
            required_metadata["method_artifact_sha256"] = self.method_artifact_sha256
        if any(record.metadata.get(key) != value for key, value in required_metadata.items()):
            raise DataValidationError(
                f"record {record.question_id!r} lacks exact phase/prompt protocol metadata"
            )
        if self.phase in {ExperimentPhase.E8, ExperimentPhase.E10} and not pending_side_effects:
            if self.benchmark == "ifeval" and not isinstance(
                record.metadata.get("official_pass"), bool
            ):
                raise DataValidationError("E10 IFEval records require the official pass result")
            if self.benchmark == "mmlu_pro":
                official_correct = record.metadata.get("official_correct")
                if not isinstance(official_correct, bool) or official_correct != (
                    record.outcome is Outcome.CORRECT
                ):
                    raise DataValidationError(
                        "E10 MMLU-Pro records require a consistent official score"
                    )
            if self.benchmark == "wikitext103":
                negative_log_likelihood = record.metadata.get("negative_log_likelihood")
                evaluated_tokens = record.metadata.get("evaluated_tokens")
                if (
                    isinstance(negative_log_likelihood, bool)
                    or not isinstance(negative_log_likelihood, int | float)
                    or not math.isfinite(float(negative_log_likelihood))
                    or float(negative_log_likelihood) < 0
                    or isinstance(evaluated_tokens, bool)
                    or not isinstance(evaluated_tokens, int)
                    or evaluated_tokens <= 0
                ):
                    raise DataValidationError(
                        "E10 WikiText records require token-level likelihood evidence"
                    )
            if self.benchmark in {
                "xstest",
                "strongreject_or_harmbench",
                "language_consistency",
            }:
                validate_side_effect_record(record)
        if self.phase in {ExperimentPhase.E9, ExperimentPhase.E10}:
            if self.benchmark == "triviaqa":
                exact_match = record.metadata.get("official_exact_match")
                token_f1 = record.metadata.get("official_token_f1")
                if (
                    isinstance(exact_match, bool)
                    or not isinstance(exact_match, int | float)
                    or float(exact_match) not in {0.0, 1.0}
                    or isinstance(token_f1, bool)
                    or not isinstance(token_f1, int | float)
                    or not math.isfinite(float(token_f1))
                    or not 0 <= float(token_f1) <= 1
                    or (float(exact_match) == 1.0) is not (record.outcome is Outcome.CORRECT)
                    or record.metadata.get("official_score_output_sha256")
                    != stable_hash(record.raw_output)
                ):
                    raise DataValidationError(
                        "confirmatory TriviaQA records require response-bound official EM/F1"
                    )
            elif self.benchmark in {
                "simpleqa_verified",
                "aa_omniscience_public_600",
            }:
                grader_attempts = record.metadata.get("grader_attempts")
                grader_failed = record.metadata.get("grader_failed")
                if (
                    isinstance(grader_attempts, bool)
                    or not isinstance(grader_attempts, int)
                    or grader_attempts <= 0
                    or not isinstance(grader_failed, bool)
                    or grader_failed is not (record.outcome is Outcome.UNSCORABLE)
                    or record.metadata.get("official_score_output_sha256")
                    != stable_hash(record.raw_output)
                    or (self.benchmark == "simpleqa_verified" and record.outcome is Outcome.PARTIAL)
                ):
                    raise DataValidationError(
                        "confirmatory model-graded records require response-bound grader evidence"
                    )


def validate_adaptive_execution(records: Iterable[GenerationRecord]) -> None:
    """Require each adaptive model/artifact to actually route both ways in a run."""

    actions: dict[tuple[str, str, str], set[str]] = {}
    counts: dict[tuple[str, str, str], int] = {}
    intervention_counts: dict[tuple[str, str, str], int] = {}
    intervention_strengths: dict[tuple[str, str, str], float] = {}
    e5_groups: set[tuple[str, str, str]] = set()
    for record in records:
        if record.steering_method not in {"M3", "M6", "ACT-or-SADI"}:
            continue
        artifact = str(record.metadata.get("method_artifact_sha256", ""))
        key = (record.steering_method, record.model_repository, artifact)
        if record.metadata.get("phase") == ExperimentPhase.E5.value:
            e5_groups.add(key)
        actions.setdefault(key, set()).add(str(record.metadata.get("policy_action", "")))
        counts[key] = counts.get(key, 0) + 1
        if record.metadata.get("policy_action") == "intervene":
            intervention_counts[key] = intervention_counts.get(key, 0) + 1
            intervention_strengths[key] = intervention_strengths.get(key, 0.0) + record.alpha
    for key, observed in actions.items():
        intervention_count = intervention_counts.get(key, 0)
        no_intervention_count = counts[key] - intervention_count
        if key in e5_groups:
            if (
                not observed <= {"intervene", "release"}
                or intervention_count <= 0
                or intervention_strengths.get(key, 0.0) / intervention_count < _MIN_ADAPTIVE_ALPHA
            ):
                raise DataValidationError(
                    "E5 adaptive execution must contain a material controller-owned "
                    f"intervention for {key}"
                )
            continue
        if (
            counts[key] < 2
            or "intervene" not in observed
            or not observed
            & {
                "release",
                "abstain",
            }
        ):
            raise DataValidationError(
                "adaptive execution must contain both a real intervention and a "
                f"record-specific no-intervention action for {key}"
            )
        if (
            intervention_count / counts[key] < _MIN_ADAPTIVE_ACTION_FRACTION
            or no_intervention_count / counts[key] < _MIN_ADAPTIVE_ACTION_FRACTION
            or intervention_strengths[key] / intervention_count < _MIN_ADAPTIVE_ALPHA
        ):
            raise DataValidationError(
                "adaptive execution must exercise both routing branches at a material "
                f"frequency and intervention magnitude for {key}"
            )


def expand_factorial_conditions(
    study: StudyProtocol,
    phase: ExperimentPhase | str,
    *,
    models: Mapping[str, ModelSpec],
    prompts: Mapping[str, PromptSpec],
    benchmark_partitions: Mapping[str, str],
    interventions: Mapping[str, InterventionSpec],
    seed: int = 17,
) -> tuple[EvaluationCondition, ...]:
    selected = study.phase(phase)
    if not selected.factorial:
        raise DataValidationError(f"{selected.phase.value} is not a factorial phase")
    requirements = (
        (set(selected.models), set(models), "models"),
        (set(selected.prompts), set(prompts), "prompts"),
        (set(selected.benchmarks), set(benchmark_partitions), "benchmarks"),
        (set(selected.methods), set(interventions), "methods"),
    )
    for expected, observed, context in requirements:
        if expected - observed:
            raise DataValidationError(
                f"factorial {context} are missing: {sorted(expected - observed)}"
            )
    conditions: list[EvaluationCondition] = []
    for model_name in selected.models:
        model = models[model_name]
        for benchmark in selected.benchmarks:
            for prompt_id in selected.prompts:
                prompt = prompts[prompt_id]
                prompt_hash = hashlib.sha256(prompt.text.encode("utf-8")).hexdigest()
                for method in selected.methods:
                    intervention = interventions[method]
                    if intervention.method != method:
                        raise DataValidationError(
                            f"intervention key {method!r} differs from method identity"
                        )
                    conditions.append(
                        EvaluationCondition(
                            phase=selected.phase,
                            benchmark=benchmark,
                            partition=benchmark_partitions[benchmark],
                            model_name=model_name,
                            model_repository=model.repository,
                            model_revision=model.revision,
                            runtime=model.runtime,
                            quantization=model.quantization,
                            model_num_layers=model.num_layers,
                            system_prompt_id=prompt.prompt_id,
                            prompt_template_sha256=prompt_hash,
                            steering_method=intervention.method,
                            method_artifact_sha256=intervention.artifact_sha256,
                            layer=intervention.layer,
                            site=intervention.site,
                            token_scope=intervention.token_scope,
                            alpha=intervention.alpha,
                            sparsity=intervention.sparsity,
                            seed=seed,
                            study_protocol_digest=study.digest,
                            adaptive_policy=intervention.adaptive_policy,
                        )
                    )
    if len({item.condition_id for item in conditions}) != len(conditions):
        raise DataValidationError("factorial expansion produced duplicate conditions")
    return tuple(conditions)


@dataclass(frozen=True, slots=True)
class PhaseRunContract:
    phase: ExperimentPhase
    study_protocol_digest: str
    conditions: tuple[EvaluationCondition, ...]
    question_ids_by_benchmark: Mapping[str, tuple[str, ...]]
    input_fingerprints: Mapping[str, str]
    prerequisite_digests: Mapping[str, str]
    required_gates: tuple[str, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise DataValidationError("unsupported phase-run contract schema")
        _require_sha256(self.study_protocol_digest, "phase study-protocol identity")
        if not self.conditions or any(item.phase is not self.phase for item in self.conditions):
            raise DataValidationError("phase contract conditions are empty or cross phases")
        if any(
            item.study_protocol_digest != self.study_protocol_digest for item in self.conditions
        ):
            raise DataValidationError("condition and phase protocol identities differ")
        if len({item.condition_id for item in self.conditions}) != len(self.conditions):
            raise DataValidationError("phase contract condition IDs must be unique")
        questions: dict[str, tuple[str, ...]] = {}
        for benchmark, identifiers in self.question_ids_by_benchmark.items():
            name = str(benchmark).strip()
            ids = tuple(str(value).strip() for value in identifiers)
            if not name or not ids or any(not value for value in ids):
                raise DataValidationError("phase question sets must be named and non-empty")
            if len(set(ids)) != len(ids):
                raise DataValidationError(f"phase question IDs repeat within {name}")
            questions[name] = ids
        condition_benchmarks = {item.benchmark for item in self.conditions}
        if set(questions) != condition_benchmarks:
            raise DataValidationError("phase question sets differ from condition benchmarks")
        inputs = {str(key): str(value) for key, value in self.input_fingerprints.items()}
        prerequisites = {str(key): str(value) for key, value in self.prerequisite_digests.items()}
        gates = tuple(str(value).strip() for value in self.required_gates)
        if any(not key.strip() or not _SHA256.fullmatch(value) for key, value in inputs.items()):
            raise DataValidationError("phase inputs must be named SHA-256 fingerprints")
        if any(
            not key.strip() or not _SHA256.fullmatch(value) for key, value in prerequisites.items()
        ):
            raise DataValidationError("phase prerequisites must be named SHA-256 fingerprints")
        if any(not value for value in gates) or len(set(gates)) != len(gates):
            raise DataValidationError("phase gates must be unique non-empty names")
        object.__setattr__(self, "question_ids_by_benchmark", MappingProxyType(questions))
        object.__setattr__(self, "input_fingerprints", MappingProxyType(inputs))
        object.__setattr__(self, "prerequisite_digests", MappingProxyType(prerequisites))
        object.__setattr__(self, "required_gates", gates)

    @property
    def expected_record_count(self) -> int:
        return sum(len(self.question_ids_by_benchmark[item.benchmark]) for item in self.conditions)

    @property
    def digest(self) -> str:
        return stable_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "phase": self.phase.value,
            "study_protocol_digest": self.study_protocol_digest,
            "conditions": [item.to_dict() for item in self.conditions],
            "question_ids_by_benchmark": {
                key: list(value) for key, value in self.question_ids_by_benchmark.items()
            },
            "input_fingerprints": dict(self.input_fingerprints),
            "prerequisite_digests": dict(self.prerequisite_digests),
            "required_gates": list(self.required_gates),
        }

    def assert_matches_study(self, study: StudyProtocol) -> None:
        """Reject contracts that do not execute the declared phase exactly."""

        if self.study_protocol_digest != study.digest:
            raise DataValidationError("phase contract uses a different study protocol")
        phase = study.phase(self.phase)
        if set(self.required_gates) != set(phase.gates):
            raise DataValidationError("phase contract gates differ from the study protocol")
        if set(self.prerequisite_digests) != {value.value for value in phase.prerequisites}:
            raise DataValidationError("phase contract prerequisite set differs from the protocol")
        required_inputs = set(phase.required_inputs) | set(phase.freeze_fields)
        if set(self.input_fingerprints) != required_inputs:
            raise DataValidationError("phase contract frozen inputs differ from the protocol")
        if {value.model_name for value in self.conditions} != set(phase.models):
            raise DataValidationError("phase contract models differ from the protocol")
        if {value.benchmark for value in self.conditions} != set(phase.benchmarks):
            raise DataValidationError("phase contract benchmarks differ from the protocol")
        observed_phase_methods = {value.steering_method for value in self.conditions}
        if self.phase is ExperimentPhase.E4:
            mandatory_e4 = {"M1", "M2", "ACT-or-SADI"}
            if not mandatory_e4 <= observed_phase_methods or not observed_phase_methods <= set(
                phase.methods
            ):
                raise DataValidationError(
                    "E4 phase contract omits a mandatory method or adds an unknown method"
                )
        elif observed_phase_methods != set(phase.methods):
            raise DataValidationError("phase contract methods differ from the protocol")
        observed_prompts = {value.system_prompt_id for value in self.conditions}
        expected_prompts = set(phase.prompts)
        if expected_prompts == {"frozen-selected"}:
            if len(observed_prompts) != 1 or not observed_prompts <= _E10_DEPLOYMENT_PROMPTS:
                raise DataValidationError("E10 must use one frozen deployment-eligible prompt")
            dimension_prompts = observed_prompts
        else:
            if observed_prompts != expected_prompts:
                raise DataValidationError("phase contract prompts differ from the protocol")
            dimension_prompts = expected_prompts
        for condition in self.conditions:
            identity = _MODEL_IDENTITIES.get(condition.model_name)
            observed_identity = (
                condition.model_repository,
                condition.model_revision,
                condition.runtime,
                condition.quantization,
                condition.model_num_layers,
            )
            if identity is None or observed_identity != identity:
                raise DataValidationError(
                    f"condition model identity differs for {condition.model_name!r}"
                )
            expected_prompt_hash = _PROMPT_HASHES.get(condition.system_prompt_id)
            if expected_prompt_hash != condition.prompt_template_sha256:
                raise DataValidationError(
                    f"condition prompt identity differs for {condition.system_prompt_id!r}"
                )
            if condition.partition not in phase.partitions:
                raise DataValidationError(
                    f"condition partition {condition.partition!r} is outside {self.phase.value}"
                )
            if condition.steering_method == "M0" and (
                condition.method_artifact_sha256 is not None
                or condition.layer is not None
                or condition.site is not None
                or condition.token_scope is not None
                or condition.alpha != 0
                or condition.sparsity is not None
            ):
                raise DataValidationError("M0 conditions cannot contain an intervention")
            if self.phase is ExperimentPhase.E6 and condition.steering_method != "M0":
                if condition.method_artifact_sha256 is None:
                    raise DataValidationError("E6 interventions require a frozen method artifact")
                if condition.steering_method == "M1":
                    if (
                        condition.layer is None
                        or condition.site is None
                        or condition.token_scope is None
                        or condition.token_scope is TokenScope.EXPONENTIAL_DECAY
                        or condition.alpha == 0
                        or condition.sparsity is not None
                        or condition.adaptive_policy is not None
                    ):
                        raise DataValidationError(
                            "E6 M1 must be one material frozen fixed intervention"
                        )
                elif condition.steering_method == "M3" and (
                    condition.layer is not None
                    or condition.site is not None
                    or condition.token_scope is not None
                    or condition.alpha != 0
                    or condition.sparsity is not None
                    or condition.adaptive_policy is None
                ):
                    raise DataValidationError(
                        "E6 M3 must defer exact execution to its frozen adaptive policy"
                    )
            if self.phase is ExperimentPhase.E7:
                if condition.partition != "T-dev":
                    raise DataValidationError(
                        "E7 generation conditions must use the frozen development partition"
                    )
                if condition.steering_method in {"M4a", "M4b"} and (
                    condition.method_artifact_sha256 is None
                    or condition.layer is None
                    or not 0 <= condition.layer < condition.model_num_layers
                    or condition.site is None
                    or condition.token_scope is None
                    or condition.alpha not in {0.1, 0.25, 0.5, 1.0, 2.0}
                    or condition.sparsity is None
                    or not 0 < condition.sparsity <= 1
                    or condition.adaptive_policy is not None
                ):
                    raise DataValidationError(
                        "E7 sparse methods must be material frozen fixed interventions"
                    )
                if condition.steering_method == "M4a" and condition.sparsity not in {
                    0.01,
                    0.05,
                    0.10,
                    0.25,
                }:
                    raise DataValidationError(
                        "E7 coordinate sparsity must use a preregistered retained fraction"
                    )
            if self.phase is ExperimentPhase.E8 and condition.steering_method != "M0":
                if condition.steering_method == "M3":
                    if (
                        condition.method_artifact_sha256 is None
                        or condition.layer is not None
                        or condition.site is not None
                        or condition.token_scope is not None
                        or condition.alpha != 0
                        or condition.sparsity is not None
                        or condition.adaptive_policy is None
                    ):
                        raise DataValidationError(
                            "E8 M3 must defer exact execution to its frozen adaptive policy"
                        )
                elif (
                    condition.steering_method not in {"M1", "M4", "M5"}
                    or condition.method_artifact_sha256 is None
                    or condition.layer is None
                    or not 0 <= condition.layer < condition.model_num_layers
                    or condition.site is None
                    or condition.token_scope is None
                    or condition.token_scope is TokenScope.EXPONENTIAL_DECAY
                    or condition.alpha not in {0.1, 0.25, 0.5, 1.0, 2.0}
                    or condition.adaptive_policy is not None
                    or (
                        condition.steering_method == "M4"
                        and (condition.sparsity is None or not 0 < condition.sparsity <= 1)
                    )
                    or (
                        condition.steering_method in {"M1", "M5"} and condition.sparsity is not None
                    )
                ):
                    raise DataValidationError(
                        "E8 fixed methods require a material registered intervention"
                    )
            if (
                self.phase in {ExperimentPhase.E9, ExperimentPhase.E10}
                and condition.steering_method != "M0"
            ):
                if condition.method_artifact_sha256 is None:
                    raise DataValidationError(
                        f"{condition.steering_method} requires a frozen component artifact"
                    )
                if condition.steering_method in {"M3", "M6"}:
                    if (
                        condition.layer is not None
                        or condition.site is not None
                        or condition.token_scope is not None
                        or condition.alpha != 0
                        or condition.sparsity is not None
                    ):
                        raise DataValidationError(
                            f"{condition.steering_method} condition must defer to its frozen policy"
                        )
                elif (
                    condition.layer is None
                    or condition.site is None
                    or condition.token_scope is None
                    or condition.alpha == 0
                ):
                    raise DataValidationError(
                        f"{condition.steering_method} must be a nonzero fixed intervention"
                    )
                if condition.steering_method == "M4" and condition.sparsity is None:
                    raise DataValidationError("M4 confirmatory conditions require frozen sparsity")
        if self.phase is ExperimentPhase.E9 and any(
            condition.partition != _E9_PARTITIONS[condition.benchmark]
            for condition in self.conditions
        ):
            raise DataValidationError("E9 benchmark partitions differ from the frozen test sets")
        if self.phase is ExperimentPhase.E6 and any(
            condition.partition != _E6_PARTITIONS[condition.benchmark]
            for condition in self.conditions
        ):
            raise DataValidationError("E6 benchmark partitions differ from frozen development sets")
        if self.phase is ExperimentPhase.E8 and any(
            condition.partition != _E8_PARTITIONS[condition.benchmark]
            for condition in self.conditions
        ):
            raise DataValidationError("E8 benchmark partitions differ from frozen evaluation sets")
        if self.phase is ExperimentPhase.E10 and any(
            condition.partition != _E10_PARTITIONS[condition.benchmark]
            for condition in self.conditions
        ):
            raise DataValidationError("E10 benchmark partitions differ from frozen evaluation sets")
        comparison_strata: dict[tuple[str, str, str, str], set[str]] = {}
        comparison_members: dict[
            tuple[str, str, str, str],
            list[EvaluationCondition],
        ] = {}
        for condition in self.conditions:
            key = (
                condition.model_name,
                condition.benchmark,
                condition.system_prompt_id,
                condition.partition,
            )
            comparison_strata.setdefault(key, set()).add(condition.steering_method)
            comparison_members.setdefault(key, []).append(condition)
        exact_phase_requirements = {
            ExperimentPhase.E3: {"M0", "M1-R", "M1-P"},
            ExperimentPhase.E4: {
                "M1",
                "M2",
                "ACT-or-SADI",
            },
            ExperimentPhase.E5: {"M1", "M3"},
            ExperimentPhase.E6: {"M0", "M1", "M3"},
            ExperimentPhase.E8: {"M0", "M1", "M3", "M4", "M5"},
        }
        for key, methods in comparison_strata.items():
            required = exact_phase_requirements.get(self.phase)
            if self.phase is ExperimentPhase.E7:
                required = {"M0", "M4a", "M4b"} if key[1] == "triviaqa" else {"M0", "M4b"}
            if required is not None and not required <= methods:
                raise DataValidationError(
                    f"{self.phase.value} comparison stratum {key!r} lacks baselines or methods: "
                    f"{sorted(required - methods)}"
                )
        pair_requirements = {
            ExperimentPhase.E3: ("M0", {"M1-R", "M1-P"}),
            ExperimentPhase.E4: (
                "M1",
                {"M2", "ITI-if-feasible", "ACT-or-SADI", "TruthX-if-feasible"},
            ),
            ExperimentPhase.E5: ("M1", {"M3"}),
            ExperimentPhase.E6: ("M0", {"M1", "M3"}),
            ExperimentPhase.E7: ("M0", {"M4a", "M4b"}),
            ExperimentPhase.E8: ("M0", {"M1", "M3", "M4", "M5"}),
        }
        if self.phase in pair_requirements:
            baseline_method, intervention_methods = pair_requirements[self.phase]
            for key, members in comparison_members.items():
                baselines = [value for value in members if value.steering_method == baseline_method]
                for intervention in (
                    value for value in members if value.steering_method in intervention_methods
                ):
                    candidates = [
                        value
                        for value in baselines
                        if value.comparison_group == intervention.comparison_group
                    ]
                    if not candidates and len(baselines) == 1:
                        candidates = baselines
                    if len(candidates) != 1:
                        raise DataValidationError(
                            f"{self.phase.value} condition in stratum {key!r} lacks exactly "
                            "one group-matched baseline"
                        )
        if phase.factorial or self.phase in {
            ExperimentPhase.E0,
            ExperimentPhase.E6,
            ExperimentPhase.E7,
            ExperimentPhase.E8,
            ExperimentPhase.E10,
        }:
            expected_dimensions = set(
                product(phase.models, phase.benchmarks, dimension_prompts, phase.methods)
            )
            observed_dimensions = {
                (
                    value.model_name,
                    value.benchmark,
                    value.system_prompt_id,
                    value.steering_method,
                )
                for value in self.conditions
            }
            if observed_dimensions != expected_dimensions or len(self.conditions) != len(
                expected_dimensions
            ):
                raise DataValidationError(
                    f"{self.phase.value} conditions do not form the exact declared matrix"
                )
        fixed_counts = _FIXED_QUESTION_COUNTS.get(self.phase, {})
        for benchmark, expected_count in fixed_counts.items():
            observed_count = len(self.question_ids_by_benchmark.get(benchmark, ()))
            if observed_count != expected_count:
                raise DataValidationError(
                    f"{self.phase.value} {benchmark} requires {expected_count} questions, "
                    f"found {observed_count}"
                )
        if self.phase in {ExperimentPhase.E1, ExperimentPhase.E10}:
            expected_external = {
                "simpleqa_verified": 1_000,
                "aa_omniscience_public_600": 600,
            }
            for benchmark, expected_count in expected_external.items():
                if (
                    benchmark in self.question_ids_by_benchmark
                    and len(self.question_ids_by_benchmark[benchmark]) != expected_count
                ):
                    raise DataValidationError(
                        f"{self.phase.value} {benchmark} requires {expected_count} questions"
                    )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PhaseRunContract:
        expected = {
            "schema_version",
            "phase",
            "study_protocol_digest",
            "conditions",
            "question_ids_by_benchmark",
            "input_fingerprints",
            "prerequisite_digests",
            "required_gates",
        }
        if set(value) != expected:
            raise DataValidationError("phase-run contract keys differ from schema version 1")
        questions = value["question_ids_by_benchmark"]
        if not isinstance(questions, Mapping):
            raise DataValidationError("phase-run question sets must be a mapping")
        return cls(
            schema_version=int(value["schema_version"]),
            phase=ExperimentPhase(value["phase"]),
            study_protocol_digest=str(value["study_protocol_digest"]),
            conditions=tuple(EvaluationCondition.from_dict(item) for item in value["conditions"]),
            question_ids_by_benchmark={
                str(key): tuple(str(item) for item in identifiers)
                for key, identifiers in questions.items()
            },
            input_fingerprints=dict(value["input_fingerprints"]),
            prerequisite_digests=dict(value["prerequisite_digests"]),
            required_gates=tuple(str(item) for item in value["required_gates"]),
        )


@dataclass(frozen=True, slots=True)
class PendingGeneration:
    condition: EvaluationCondition
    question_id: str


@dataclass(frozen=True, slots=True)
class PhaseCompletion:
    phase: ExperimentPhase
    contract_digest: str
    record_count: int
    shard_fingerprints: Mapping[str, str]
    record_set_digest: str
    gate_result_digests: Mapping[str, str]
    gate_file_fingerprints: Mapping[str, str]
    gate_artifact_fingerprints: Mapping[str, str]
    completion_digest: str


def open_phase_prerequisite(
    directory: str | Path,
    *,
    phase: ExperimentPhase,
    study: StudyProtocol,
    expected_completion_digest: str | None = None,
) -> Any:
    """Open either a generic ledger or the custom seven-stage E3 completion."""

    if phase is ExperimentPhase.E3:
        from mfh.experiments.e3_phase import open_e3_phase_completion

        return open_e3_phase_completion(
            directory,
            study=study,
            expected_completion_digest=expected_completion_digest,
        )
    ledger = PhaseRunLedger.open(directory, study=study)
    completion = ledger.verify_complete()
    if completion.phase is not phase or (
        expected_completion_digest is not None
        and completion.completion_digest != expected_completion_digest
    ):
        raise FrozenArtifactError(f"{phase.value} prerequisite differs from its contract")
    return ledger


def _validate_e3_prerequisite_lineage(
    prerequisites: Mapping[ExperimentPhase, Any],
    *,
    contract: PhaseRunContract | None = None,
) -> None:
    e3 = prerequisites.get(ExperimentPhase.E3)
    if e3 is None:
        return
    embedded = getattr(e3, "prerequisite_digests", None)
    if not isinstance(embedded, Mapping) or set(embedded) != {"E1", "E2"}:
        raise FrozenArtifactError("E3 prerequisite lineage is incomplete")
    for phase in (ExperimentPhase.E1, ExperimentPhase.E2):
        prior = prerequisites.get(phase)
        if prior is not None and prior.verify_complete().completion_digest != embedded[phase.value]:
            raise FrozenArtifactError(
                f"E3 embeds a different {phase.value} prerequisite completion"
            )
    if contract is not None and "E3_static_vectors" in contract.input_fingerprints:
        outputs = getattr(e3, "output_fingerprints", None)
        if (
            not isinstance(outputs, Mapping)
            or outputs.get("E3_static_vectors") != contract.input_fingerprints["E3_static_vectors"]
        ):
            raise FrozenArtifactError(
                "phase E3_static_vectors input differs from the completed E3 output"
            )


@dataclass(frozen=True, slots=True)
class PhaseFalsification:
    """Auditable terminal result that can never satisfy a phase prerequisite."""

    phase: ExperimentPhase
    contract_digest: str
    record_count: int
    shard_fingerprints: Mapping[str, str]
    record_set_digest: str
    gate_result_digests: Mapping[str, str]
    gate_file_fingerprints: Mapping[str, str]
    gate_artifact_fingerprints: Mapping[str, str]
    failed_gates: tuple[str, ...]
    falsification_digest: str


def _confirmatory_prompt_snapshot_body(
    prompts: Mapping[str, PromptSpec],
    contract: PhaseRunContract,
) -> dict[str, Any]:
    expected_hashes: dict[str, str] = {}
    for condition in contract.conditions:
        previous = expected_hashes.setdefault(
            condition.system_prompt_id,
            condition.prompt_template_sha256,
        )
        if previous != condition.prompt_template_sha256:
            raise DataValidationError("confirmatory conditions disagree on one prompt identity")
    if set(prompts) != set(expected_hashes):
        raise DataValidationError("confirmatory prompt snapshot differs from the condition matrix")
    entries: list[dict[str, Any]] = []
    for prompt_id in sorted(prompts):
        prompt = prompts[prompt_id]
        prompt_sha256 = hashlib.sha256(prompt.text.encode("utf-8")).hexdigest()
        if prompt.prompt_id != prompt_id or prompt_sha256 != expected_hashes[prompt_id]:
            raise DataValidationError(
                "confirmatory prompt content differs from the condition matrix"
            )
        entries.append(
            {
                "prompt_id": prompt.prompt_id,
                "text": prompt.text,
                "permits_abstention": prompt.permits_abstention,
                "deployment_eligible": prompt.deployment_eligible,
                "text_sha256": prompt_sha256,
            }
        )
    return {
        "schema_version": 1,
        "contract_digest": contract.digest,
        "prompts": entries,
    }


def _load_confirmatory_prompt_snapshot(
    path: Path,
    contract: PhaseRunContract,
) -> dict[str, PromptSpec]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read confirmatory prompt snapshot: {exc}") from exc
    if not isinstance(payload, dict):
        raise FrozenArtifactError("confirmatory prompt snapshot must be a mapping")
    digest = payload.pop("snapshot_digest", None)
    if (
        set(payload) != {"schema_version", "contract_digest", "prompts"}
        or payload.get("schema_version") != 1
        or payload.get("contract_digest") != contract.digest
        or digest != stable_hash(payload)
        or not isinstance(payload.get("prompts"), list)
    ):
        raise FrozenArtifactError("confirmatory prompt snapshot identity differs")
    prompts: dict[str, PromptSpec] = {}
    for entry in payload["prompts"]:
        if (
            not isinstance(entry, Mapping)
            or set(entry)
            != {
                "prompt_id",
                "text",
                "permits_abstention",
                "deployment_eligible",
                "text_sha256",
            }
            or type(entry["prompt_id"]) is not str
            or type(entry["text"]) is not str
            or type(entry["permits_abstention"]) is not bool
            or type(entry["deployment_eligible"]) is not bool
            or type(entry["text_sha256"]) is not str
            or not _SHA256.fullmatch(entry["text_sha256"])
            or entry["text_sha256"] != hashlib.sha256(entry["text"].encode("utf-8")).hexdigest()
        ):
            raise FrozenArtifactError("confirmatory prompt entry is invalid")
        try:
            prompt = PromptSpec(
                prompt_id=entry["prompt_id"],
                text=entry["text"],
                permits_abstention=entry["permits_abstention"],
                deployment_eligible=entry["deployment_eligible"],
            )
        except (TypeError, ValueError, DataValidationError) as exc:
            raise FrozenArtifactError(f"invalid confirmatory prompt: {exc}") from exc
        if prompt.prompt_id in prompts:
            raise FrozenArtifactError("confirmatory prompt identifiers repeat")
        prompts[prompt.prompt_id] = prompt
    try:
        expected_body = _confirmatory_prompt_snapshot_body(prompts, contract)
    except DataValidationError as exc:
        raise FrozenArtifactError(str(exc)) from exc
    if payload != expected_body:
        raise FrozenArtifactError("confirmatory prompt snapshot differs from its contract")
    return prompts


class PhaseRunLedger:
    """Append immutable record shards, resume by question, then freeze once complete."""

    def __init__(
        self,
        directory: str | Path,
        contract: PhaseRunContract,
        *,
        study: StudyProtocol,
        _verification_token: object,
    ) -> None:
        if _verification_token is not _VERIFIED_LEDGER:
            raise DataValidationError(
                "phase ledgers must be created or opened through their verified factories"
            )
        contract.assert_matches_study(study)
        self.directory = Path(directory)
        self.contract = contract
        self.study = study
        self._conditions = {item.condition_id: item for item in contract.conditions}
        self._question_sets = {
            benchmark: frozenset(identifiers)
            for benchmark, identifiers in contract.question_ids_by_benchmark.items()
        }
        self._completed_cache: set[tuple[str, str]] | None = None
        self._confirmatory_execution_key_cache: str | None = None
        self._confirmatory_grader_bundle_cache: Any | None = None
        self._confirmatory_question_cache: dict[tuple[str, str], Question] = {}
        self._confirmatory_component_cache: dict[tuple[str, str], Path] = {}
        self._confirmatory_composite_cache: dict[str, Any] = {}

    def _validate_mutation_namespace(self) -> None:
        """Recheck the ledger path before every operation that can mutate it."""

        validated = validate_active_study_artifact_paths({"phase_run": self.directory})["phase_run"]
        if validated != self.directory:
            raise FrozenArtifactError("phase-run directory identity changed")
        if self.contract.phase is ExperimentPhase.E10:
            try:
                evidence = json.loads(
                    (self.directory / "creation-evidence.json").read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise FrozenArtifactError(
                    f"cannot recheck E10 one-shot reservation: {exc}"
                ) from exc
            if not isinstance(evidence, Mapping):
                raise FrozenArtifactError("E10 creation evidence is invalid")
            self._verify_one_shot_reservation(evidence.get("one_shot_reservation"))

    @classmethod
    def create(
        cls,
        directory: str | Path,
        contract: PhaseRunContract,
        *,
        study: StudyProtocol,
        input_artifacts: Mapping[str, str | Path],
        prerequisite_runs: Mapping[ExperimentPhase | str, str | Path],
        verified_reviewed_splits: object | None = None,
        confirmatory_prompts: Mapping[str, PromptSpec] | None = None,
    ) -> PhaseRunLedger:
        namespace_paths = {
            "phase_run": directory,
            **{
                f"prerequisite_{ExperimentPhase(key).value}": value
                for key, value in prerequisite_runs.items()
            },
        }
        normalized_namespace_paths = validate_active_study_artifact_paths(namespace_paths)
        destination = normalized_namespace_paths["phase_run"]
        if destination.exists():
            raise FrozenArtifactError(f"refusing to overwrite phase run: {destination}")
        contract.assert_matches_study(study)
        phase = study.phase(contract.phase)
        study.assert_frozen_inputs(contract.phase, contract.input_fingerprints)
        study.assert_ready(contract.phase, contract.prerequisite_digests)
        if set(input_artifacts) != set(contract.input_fingerprints):
            raise DataValidationError("phase input artifact paths differ from the frozen inputs")
        input_sources: dict[str, Path] = {}
        input_evidence: dict[str, dict[str, str]] = {}
        for name, location in input_artifacts.items():
            if _ARTIFACT_NAME.fullmatch(name) is None:
                raise DataValidationError(f"phase input name is not safe: {name!r}")
            path = Path(location).resolve()
            _assert_regular_artifact(path, f"phase input {name!r}")
            try:
                fingerprint = sha256_path(path)
            except (OSError, FrozenArtifactError) as exc:
                raise DataValidationError(
                    f"cannot fingerprint phase input {name!r}: {exc}"
                ) from exc
            if fingerprint != contract.input_fingerprints[name]:
                raise DataValidationError(
                    f"phase input {name!r} differs from its frozen fingerprint"
                )
            input_evidence[name] = {
                "location": str(path),
                "fingerprint": fingerprint,
            }
            input_sources[name] = path
        scientific_input_authorizations = _reviewed_split_authorization(
            contract,
            verified_reviewed_splits,
            input_sources=input_sources,
            input_fingerprints=contract.input_fingerprints,
        )
        component_input = _COMPONENT_INPUT.get(contract.phase)
        if contract.phase in {ExperimentPhase.E9, ExperimentPhase.E10}:
            assert component_input is not None
            _validate_question_bundle(input_sources[_QUESTION_BUNDLE], contract)
            _validate_component_selection(input_sources[component_input], contract)
        scorer_input = (
            "frozen_side_effect_scorers"
            if contract.phase in {ExperimentPhase.E7, ExperimentPhase.E8}
            else "frozen_graders"
            if contract.phase is ExperimentPhase.E9
            else "grader"
            if contract.phase is ExperimentPhase.E10
            else None
        )
        if scorer_input is not None:
            if contract.phase in {ExperimentPhase.E9, ExperimentPhase.E10}:
                from mfh.experiments.confirmatory_graders import (
                    validate_confirmatory_grader_bundle,
                )

                validate_confirmatory_grader_bundle(input_sources[scorer_input])
            else:
                load_side_effect_scorer_spec(input_sources[scorer_input])
        if contract.phase is ExperimentPhase.E7:
            load_sae_stability_bundle(input_sources["frozen_sae_seed_runs"])
            from mfh.experiments.e7_sparse import validate_separate_sae_corpus

            validate_separate_sae_corpus(
                input_sources["separate_sae_corpus"],
                evaluation_question_ids={
                    question_id
                    for values in contract.question_ids_by_benchmark.values()
                    for question_id in values
                },
            )
        snapshot_input = (
            "frozen_evaluation_scripts"
            if contract.phase is ExperimentPhase.E9
            else "evaluation_scripts"
            if contract.phase is ExperimentPhase.E10
            else None
        )
        if snapshot_input is not None:
            validate_execution_snapshot(
                input_sources[snapshot_input],
                study_protocol_digest=contract.study_protocol_digest,
                phase=contract.phase,
            )
        prerequisite_paths = {
            ExperimentPhase(key): normalized_namespace_paths[
                f"prerequisite_{ExperimentPhase(key).value}"
            ]
            for key in prerequisite_runs
        }
        if set(prerequisite_paths) != set(phase.prerequisites):
            raise DataValidationError("phase prerequisite run paths differ from the protocol")
        prerequisite_evidence: dict[str, dict[str, str]] = {}
        prerequisite_ledgers: dict[ExperimentPhase, Any] = {}
        for prerequisite, path in prerequisite_paths.items():
            try:
                prior = open_phase_prerequisite(
                    path,
                    phase=prerequisite,
                    study=study,
                    expected_completion_digest=contract.prerequisite_digests[prerequisite.value],
                )
            except FrozenArtifactError as exc:
                raise DataValidationError(str(exc)) from exc
            completion = prior.verify_complete()
            expected_digest = contract.prerequisite_digests[prerequisite.value]
            if (
                completion.phase is not prerequisite
                or completion.completion_digest != expected_digest
            ):
                raise DataValidationError(
                    f"verified prerequisite {prerequisite.value} differs from the contract"
                )
            prerequisite_evidence[prerequisite.value] = {
                "location": str(path),
                "completion_digest": completion.completion_digest,
            }
            prerequisite_ledgers[prerequisite] = prior
        try:
            _validate_e3_prerequisite_lineage(
                prerequisite_ledgers,
                contract=contract,
            )
        except FrozenArtifactError as exc:
            raise DataValidationError(str(exc)) from exc
        if contract.phase is ExperimentPhase.E9:
            _validate_e9_component_promotions(
                input_sources["frozen_component_selection"],
                contract,
                prerequisite_ledgers,
            )
        elif contract.phase is ExperimentPhase.E10:
            from mfh.experiments.e10_composite import (
                validate_e10_prerequisite_bound_inputs,
            )

            if confirmatory_prompts is None:
                raise DataValidationError(
                    "E10 promotion requires the exact independently selected prompt"
                )
            validate_e10_prerequisite_bound_inputs(
                input_sources,
                contract=contract,
                prompts=confirmatory_prompts,
                prerequisite_ledgers=prerequisite_ledgers,
            )
        if contract.phase is ExperimentPhase.E1:
            split_authorization = scientific_input_authorizations.get("deduplicated_splits")
            if not isinstance(split_authorization, Mapping):  # pragma: no cover - set above
                raise DataValidationError("E1 lacks reviewed-split authorization")
            try:
                e0_review_digest = _e0_review_result_digest(prerequisite_paths[ExperimentPhase.E0])
            except FrozenArtifactError as exc:
                raise DataValidationError(str(exc)) from exc
            if split_authorization.get("review_result_manifest_digest") != e0_review_digest:
                raise DataValidationError(
                    "E1 reviewed splits use a different human review than the E0 prerequisite"
                )
        if phase.one_shot and contract.phase is not ExperimentPhase.E10:
            raise DataValidationError("only E10 may use the one-shot phase runner")
        if contract.phase in {ExperimentPhase.E9, ExperimentPhase.E10}:
            if confirmatory_prompts is None:
                raise DataValidationError(
                    "confirmatory phase creation requires the exact prompt texts"
                )
            prompt_snapshot_body = _confirmatory_prompt_snapshot_body(
                confirmatory_prompts,
                contract,
            )
        elif confirmatory_prompts is not None:
            raise DataValidationError("non-confirmatory phase cannot package confirmatory prompts")
        else:
            prompt_snapshot_body = None
        destination.parent.mkdir(parents=True, exist_ok=True)
        reservation_evidence: dict[str, str] | None = None
        stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
        try:
            if phase.one_shot:
                registry = study.one_shot_registry_path
                if registry == destination.absolute() or registry.is_relative_to(
                    destination.absolute()
                ):
                    raise DataValidationError(
                        "one-shot registry cannot be inside the run directory"
                    )
                reservation_evidence, _ = _reserve_one_shot(
                    registry,
                    study=study,
                    contract=contract,
                    destination=destination,
                )
            (stage / "shards").mkdir()
            (stage / "gates").mkdir()
            (stage / "gate-artifacts").mkdir()
            if contract.phase in {ExperimentPhase.E9, ExperimentPhase.E10}:
                packaged_inputs = stage / "inputs"
                packaged_inputs.mkdir()
                for name in sorted(input_sources):
                    packaged = packaged_inputs / name
                    _copy_frozen_artifact(
                        input_sources[name],
                        packaged,
                        contract.input_fingerprints[name],
                    )
                    input_evidence[name]["location"] = f"inputs/{name}"
                assert prompt_snapshot_body is not None
                prompt_snapshot_path = packaged_inputs / "confirmatory-prompts.json"
                prompt_snapshot_path.write_text(
                    json.dumps(
                        {
                            **prompt_snapshot_body,
                            "snapshot_digest": stable_hash(prompt_snapshot_body),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                prompt_snapshot_evidence: dict[str, str] | None = {
                    "location": "inputs/confirmatory-prompts.json",
                    "fingerprint": sha256_file(prompt_snapshot_path),
                }
            else:
                prompt_snapshot_evidence = None
            body = contract.to_dict()
            payload = {**body, "contract_digest": stable_hash(body)}
            (stage / "contract.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            evidence_body = {
                "schema_version": (
                    2 if contract.phase in {ExperimentPhase.E9, ExperimentPhase.E10} else 1
                ),
                "input_artifacts": input_evidence,
                "prerequisite_runs": prerequisite_evidence,
                "scientific_input_authorizations": scientific_input_authorizations,
                "one_shot_reservation": reservation_evidence,
                **(
                    {"confirmatory_prompts": prompt_snapshot_evidence}
                    if prompt_snapshot_evidence is not None
                    else {}
                ),
            }
            (stage / "creation-evidence.json").write_text(
                json.dumps(
                    {**evidence_body, "evidence_digest": stable_hash(evidence_body)},
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            os.replace(stage, destination)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
        return cls(
            destination,
            contract,
            study=study,
            _verification_token=_VERIFIED_LEDGER,
        )

    @classmethod
    def open(
        cls,
        directory: str | Path,
        *,
        study: StudyProtocol,
    ) -> PhaseRunLedger:
        source = validate_active_study_artifact_paths({"phase_run": directory})["phase_run"]
        try:
            payload = json.loads((source / "contract.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FrozenArtifactError(f"cannot read phase-run contract: {exc}") from exc
        digest = payload.pop("contract_digest", None)
        if digest != stable_hash(payload):
            raise FrozenArtifactError("phase-run contract digest mismatch")
        try:
            contract = PhaseRunContract.from_dict(payload)
        except (KeyError, TypeError, ValueError, DataValidationError) as exc:
            raise FrozenArtifactError(f"invalid phase-run contract: {exc}") from exc
        try:
            contract.assert_matches_study(study)
        except DataValidationError as exc:
            raise FrozenArtifactError(f"phase run differs from study protocol: {exc}") from exc
        ledger = cls(
            source,
            contract,
            study=study,
            _verification_token=_VERIFIED_LEDGER,
        )
        ledger._shard_paths()
        ledger._verify_creation_evidence()
        return ledger

    def _shard_paths(self) -> tuple[Path, ...]:
        shard_directory = self.directory / "shards"
        if not shard_directory.is_dir():
            raise FrozenArtifactError("phase run has no shard directory")
        paths = sorted(shard_directory.iterdir())
        indices: list[int] = []
        for path in paths:
            match = _SHARD.fullmatch(path.name)
            if not path.is_file() or match is None:
                raise FrozenArtifactError(f"unexpected phase-run shard artifact: {path.name}")
            indices.append(int(match.group(1)))
        if indices != list(range(len(indices))):
            raise FrozenArtifactError("phase-run shard numbering is not contiguous")
        return tuple(paths)

    def _gate_paths(self) -> tuple[Path, ...]:
        gate_directory = self.directory / "gates"
        if not gate_directory.is_dir():
            raise FrozenArtifactError("phase run has no gate-evidence directory")
        paths = sorted(gate_directory.iterdir())
        if any(not path.is_file() or _GATE_FILE.fullmatch(path.name) is None for path in paths):
            raise FrozenArtifactError("phase run contains an invalid gate-evidence artifact")
        return tuple(paths)

    def _gate_artifact_fingerprints(
        self,
        gate_results: Mapping[str, GateResult],
    ) -> dict[str, str]:
        root = self.directory / "gate-artifacts"
        if not root.is_dir() or root.is_symlink():
            raise FrozenArtifactError("phase run has no regular gate-artifact directory")
        expected_gates = set(gate_results)
        observed_gates = {path.name for path in root.iterdir()}
        if observed_gates != expected_gates:
            raise FrozenArtifactError("packaged gate-artifact directories differ from evidence")
        fingerprints: dict[str, str] = {}
        for gate, result in gate_results.items():
            gate_directory = root / gate
            if not gate_directory.is_dir() or gate_directory.is_symlink():
                raise FrozenArtifactError(f"gate artifact directory is invalid: {gate}")
            observed_artifacts = {path.name for path in gate_directory.iterdir()}
            if observed_artifacts != set(result.artifact_fingerprints):
                raise FrozenArtifactError(f"packaged artifacts differ for gate {gate!r}")
            for name, expected in result.artifact_fingerprints.items():
                path = gate_directory / name
                try:
                    _assert_regular_artifact(path, f"gate artifact {gate}/{name}")
                    actual = sha256_path(path)
                except (OSError, DataValidationError, FrozenArtifactError) as exc:
                    raise FrozenArtifactError(
                        f"cannot verify packaged gate artifact {gate}/{name}: {exc}"
                    ) from exc
                if actual != expected:
                    raise FrozenArtifactError(f"packaged gate artifact changed: {gate}/{name}")
                fingerprints[f"{gate}/{name}"] = actual
        return dict(sorted(fingerprints.items()))

    def _package_gate_artifacts(self, gate_results: Mapping[str, GateResult]) -> None:
        root = self.directory / "gate-artifacts"
        if any(root.iterdir()):
            self._gate_artifact_fingerprints(gate_results)
            return
        for gate, result in gate_results.items():
            if set(result.artifact_paths) != set(result.artifact_fingerprints):
                raise DataValidationError(
                    f"gate {gate!r} must provide each source artifact at finalization"
                )
            for name, fingerprint in result.artifact_fingerprints.items():
                try:
                    _assert_regular_artifact(
                        result.artifact_paths[name],
                        f"gate artifact {gate}/{name}",
                    )
                    actual = sha256_path(result.artifact_paths[name])
                except (OSError, DataValidationError, FrozenArtifactError) as exc:
                    raise DataValidationError(
                        f"cannot verify gate artifact source {gate}/{name}: {exc}"
                    ) from exc
                if actual != fingerprint:
                    raise DataValidationError(f"gate artifact source changed: {gate}/{name}")
        stage = Path(tempfile.mkdtemp(prefix=".gate-artifacts-stage-", dir=self.directory))
        try:
            for gate, result in gate_results.items():
                for name, fingerprint in result.artifact_fingerprints.items():
                    _copy_frozen_artifact(
                        result.artifact_paths[name],
                        stage / gate / name,
                        fingerprint,
                    )
            # POSIX rename replaces an existing empty directory atomically. Keeping
            # the destination in place until this call means an interrupted replace
            # leaves a valid empty root that can be retried.
            os.replace(stage, root)
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    def _verify_one_shot_reservation(self, descriptor: object) -> None:
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
            "location",
            "fingerprint",
        }:
            raise FrozenArtifactError("E10 has no valid one-shot reservation evidence")
        location = descriptor["location"]
        fingerprint = descriptor["fingerprint"]
        if (
            not isinstance(location, str)
            or not isinstance(fingerprint, str)
            or not _SHA256.fullmatch(fingerprint)
        ):
            raise FrozenArtifactError("E10 one-shot reservation identity is invalid")
        try:
            registry = reject_symlink_path_components(
                self.study.one_shot_registry_path,
                "one-shot registry",
            )
            reservation_path = reject_symlink_path_components(
                location,
                "one-shot reservation",
            )
        except DataValidationError as exc:
            raise FrozenArtifactError(str(exc)) from exc
        expected_reservation_path = registry / (
            f"{self.study.study_id}-{self.study.digest}-{ExperimentPhase.E10.value}.json"
        )
        expected_claim_directory = registry / "claims" / expected_reservation_path.stem
        expected_claim_path = expected_claim_directory / "claim.json"
        try:
            claim_path = reject_symlink_path_components(
                expected_claim_path,
                "one-shot claim",
            )
        except DataValidationError as exc:
            raise FrozenArtifactError(str(exc)) from exc
        if (
            reservation_path != expected_reservation_path
            or reservation_path.is_symlink()
            or not reservation_path.is_file()
            or expected_claim_directory.is_symlink()
            or not expected_claim_directory.is_dir()
            or {item.name for item in expected_claim_directory.iterdir()} != {"claim.json"}
            or claim_path != expected_claim_path
            or claim_path.is_symlink()
            or not claim_path.is_file()
        ):
            raise FrozenArtifactError(
                "E10 reservation is outside the protocol-owned one-shot registry"
            )
        try:
            reservation_payload = json.loads(reservation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FrozenArtifactError(f"cannot read E10 one-shot reservation: {exc}") from exc
        if (
            not isinstance(reservation_payload, dict)
            or sha256_file(reservation_path) != fingerprint
            or sha256_file(claim_path) != fingerprint
            or claim_path.read_bytes() != reservation_path.read_bytes()
        ):
            raise FrozenArtifactError("E10 one-shot reservation changed")
        reservation_digest = reservation_payload.pop("reservation_digest", None)
        expected_reservation = {
            "schema_version": 1,
            "study_id": self.study.study_id,
            "study_protocol_digest": self.study.digest,
            "phase": ExperimentPhase.E10.value,
            "contract_digest": self.contract.digest,
            "run_directory": str(self.directory.resolve()),
        }
        if reservation_payload != expected_reservation or reservation_digest != stable_hash(
            reservation_payload
        ):
            raise FrozenArtifactError("E10 one-shot reservation differs from the run")

    def _verify_creation_evidence(self) -> None:
        try:
            payload = json.loads(
                (self.directory / "creation-evidence.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise FrozenArtifactError(f"cannot read phase creation evidence: {exc}") from exc
        if not isinstance(payload, dict):
            raise FrozenArtifactError("phase creation evidence must be a mapping")
        digest = payload.pop("evidence_digest", None)
        if digest != stable_hash(payload):
            raise FrozenArtifactError("phase creation evidence digest mismatch")
        confirmatory = self.contract.phase in {ExperimentPhase.E9, ExperimentPhase.E10}
        expected_fields = {
            "schema_version",
            "input_artifacts",
            "prerequisite_runs",
            "scientific_input_authorizations",
            "one_shot_reservation",
        }
        if confirmatory:
            expected_fields.add("confirmatory_prompts")
        if set(payload) != expected_fields or payload.get("schema_version") != (
            2 if confirmatory else 1
        ):
            raise FrozenArtifactError("phase creation evidence has an invalid schema")
        inputs = payload["input_artifacts"]
        prerequisites = payload["prerequisite_runs"]
        authorizations = payload["scientific_input_authorizations"]
        reservation = payload["one_shot_reservation"]
        if (
            not isinstance(inputs, Mapping)
            or not isinstance(prerequisites, Mapping)
            or not isinstance(authorizations, Mapping)
        ):
            raise FrozenArtifactError("phase creation evidence sections must be mappings")
        observed_inputs: dict[str, str] = {}
        for name, descriptor in inputs.items():
            if (
                not isinstance(name, str)
                or not isinstance(descriptor, Mapping)
                or set(descriptor) != {"location", "fingerprint"}
            ):
                raise FrozenArtifactError("phase input evidence descriptor is invalid")
            location = descriptor["location"]
            fingerprint = descriptor["fingerprint"]
            if (
                not isinstance(location, str)
                or not location
                or not isinstance(fingerprint, str)
                or not _SHA256.fullmatch(fingerprint)
            ):
                raise FrozenArtifactError("phase input evidence identity is invalid")
            candidate = Path(location)
            if candidate.is_absolute():
                source = candidate
            else:
                source = (self.directory / candidate).resolve()
                if not source.is_relative_to(self.directory.resolve()):
                    raise FrozenArtifactError("phase input evidence escapes its run directory")
            try:
                _assert_regular_artifact(source, f"phase input {name!r}")
                actual_fingerprint = sha256_path(source)
            except (OSError, DataValidationError, FrozenArtifactError) as exc:
                raise FrozenArtifactError(
                    f"cannot verify phase input evidence {name!r}: {exc}"
                ) from exc
            if actual_fingerprint != fingerprint:
                raise FrozenArtifactError(f"phase input evidence changed: {name}")
            observed_inputs[name] = fingerprint
        observed_prerequisites: dict[str, str] = {}
        prerequisite_ledgers: dict[ExperimentPhase, Any] = {}
        for name, descriptor in prerequisites.items():
            if (
                not isinstance(name, str)
                or not isinstance(descriptor, Mapping)
                or set(descriptor) != {"location", "completion_digest"}
            ):
                raise FrozenArtifactError("phase prerequisite evidence descriptor is invalid")
            location = descriptor["location"]
            completion_digest = descriptor["completion_digest"]
            if (
                not isinstance(location, str)
                or not location
                or not isinstance(completion_digest, str)
                or not _SHA256.fullmatch(completion_digest)
            ):
                raise FrozenArtifactError("phase prerequisite evidence identity is invalid")
            prerequisite_path = _resolve_ledger_evidence_path(
                self.directory,
                location,
                context=f"phase prerequisite {name!r}",
            )
            if prerequisite_path == self.directory.resolve():
                raise FrozenArtifactError("phase prerequisite evidence cannot reference itself")
            try:
                prerequisite_phase = ExperimentPhase(name)
                prior = open_phase_prerequisite(
                    prerequisite_path,
                    phase=prerequisite_phase,
                    study=self.study,
                    expected_completion_digest=completion_digest,
                )
                completion = prior.verify_complete()
            except (ValueError, FrozenArtifactError) as exc:
                raise FrozenArtifactError(
                    f"cannot verify phase prerequisite evidence {name!r}: {exc}"
                ) from exc
            if (
                completion.phase is not prerequisite_phase
                or completion.completion_digest != completion_digest
            ):
                raise FrozenArtifactError(f"phase prerequisite evidence changed: {name}")
            observed_prerequisites[name] = completion_digest
            prerequisite_ledgers[prerequisite_phase] = prior
        _validate_e3_prerequisite_lineage(
            prerequisite_ledgers,
            contract=self.contract,
        )
        if observed_inputs != dict(
            self.contract.input_fingerprints
        ) or observed_prerequisites != dict(self.contract.prerequisite_digests):
            raise FrozenArtifactError("phase creation evidence differs from its contract")
        if self.contract.phase is ExperimentPhase.E9:
            try:
                _validate_e9_component_promotions(
                    self.directory / "inputs" / "frozen_component_selection",
                    self.contract,
                    prerequisite_ledgers,
                )
            except DataValidationError as exc:
                raise FrozenArtifactError(
                    f"E9 component promotion provenance differs: {exc}"
                ) from exc
        if confirmatory:
            prompt_descriptor = payload["confirmatory_prompts"]
            if (
                not isinstance(prompt_descriptor, Mapping)
                or set(prompt_descriptor) != {"location", "fingerprint"}
                or prompt_descriptor.get("location") != "inputs/confirmatory-prompts.json"
                or type(prompt_descriptor.get("fingerprint")) is not str
                or not _SHA256.fullmatch(prompt_descriptor["fingerprint"])
            ):
                raise FrozenArtifactError("confirmatory prompt creation evidence is invalid")
            prompt_path = self.directory / "inputs" / "confirmatory-prompts.json"
            if (
                prompt_path.is_symlink()
                or not prompt_path.is_file()
                or sha256_file(prompt_path) != prompt_descriptor["fingerprint"]
            ):
                raise FrozenArtifactError("confirmatory prompt snapshot changed")
            _load_confirmatory_prompt_snapshot(prompt_path, self.contract)
            if self.contract.phase is ExperimentPhase.E10:
                from mfh.experiments.e10_composite import (
                    validate_e10_prerequisite_bound_inputs,
                )

                try:
                    validate_e10_prerequisite_bound_inputs(
                        {
                            name: self.directory / "inputs" / name
                            for name in self.contract.input_fingerprints
                        },
                        contract=self.contract,
                        prompts=_load_confirmatory_prompt_snapshot(prompt_path, self.contract),
                        prerequisite_ledgers=prerequisite_ledgers,
                    )
                except DataValidationError as exc:
                    raise FrozenArtifactError(
                        f"E10 component promotion provenance differs: {exc}"
                    ) from exc
        if self.contract.phase is ExperimentPhase.E1:
            descriptor = authorizations.get("deduplicated_splits")
            if not isinstance(descriptor, Mapping) or set(descriptor) != {
                "kind",
                "manifest_digest",
                "review_result_manifest_digest",
                "fingerprint",
            }:
                raise FrozenArtifactError("E1 lacks reviewed-split authorization evidence")
            manifest_digest = descriptor.get("manifest_digest")
            review_result_manifest_digest = descriptor.get("review_result_manifest_digest")
            fingerprint = descriptor.get("fingerprint")
            if (
                descriptor.get("kind") != "human-reviewed-contamination-controlled-triviaqa-splits"
                or type(manifest_digest) is not str
                or type(review_result_manifest_digest) is not str
                or type(fingerprint) is not str
                or not _SHA256.fullmatch(manifest_digest)
                or not _SHA256.fullmatch(review_result_manifest_digest)
                or not _SHA256.fullmatch(fingerprint)
                or fingerprint != observed_inputs.get("deduplicated_splits")
            ):
                raise FrozenArtifactError("E1 reviewed-split authorization identity differs")
            source_descriptor = inputs.get("deduplicated_splits")
            if not isinstance(source_descriptor, Mapping):
                raise FrozenArtifactError("E1 reviewed-split input evidence differs")
            source = _resolve_ledger_evidence_path(
                self.directory,
                source_descriptor["location"],
                context="E1 reviewed split",
            )
            try:
                from mfh.data.reviewed_splits import validate_reviewed_split_snapshot

                manifest = validate_reviewed_split_snapshot(source)
            except (OSError, DataValidationError, FrozenArtifactError) as exc:
                raise FrozenArtifactError(f"cannot verify E1 reviewed splits: {exc}") from exc
            if (
                manifest.get("manifest_digest") != manifest_digest
                or manifest.get("review_result_manifest_digest") != review_result_manifest_digest
                or sha256_path(source) != fingerprint
            ):
                raise FrozenArtifactError("E1 reviewed splits changed")
            e0_descriptor = prerequisites.get("E0")
            if not isinstance(e0_descriptor, Mapping):
                raise FrozenArtifactError("E1 lacks its E0 prerequisite evidence")
            if (
                _e0_review_result_digest(
                    _resolve_ledger_evidence_path(
                        self.directory,
                        e0_descriptor["location"],
                        context="E1 E0 prerequisite",
                    )
                )
                != review_result_manifest_digest
            ):
                raise FrozenArtifactError(
                    "E1 reviewed splits differ from the E0 human-review provenance"
                )
        elif authorizations:
            raise FrozenArtifactError("non-E1 phase contains reviewed-split authorization")
        if self.contract.phase is ExperimentPhase.E10:
            self._verify_one_shot_reservation(reservation)
        elif reservation is not None:
            raise FrozenArtifactError("non-E10 phase contains one-shot reservation evidence")
        scorer_input = (
            "frozen_side_effect_scorers"
            if self.contract.phase in {ExperimentPhase.E7, ExperimentPhase.E8}
            else "frozen_graders"
            if self.contract.phase is ExperimentPhase.E9
            else "grader"
            if self.contract.phase is ExperimentPhase.E10
            else None
        )
        if scorer_input is not None:
            descriptor = inputs[scorer_input]
            assert isinstance(descriptor, Mapping)
            scorer_path = Path(str(descriptor["location"]))
            if not scorer_path.is_absolute():
                scorer_path = (self.directory / scorer_path).resolve()
            try:
                if self.contract.phase in {ExperimentPhase.E9, ExperimentPhase.E10}:
                    from mfh.experiments.confirmatory_graders import (
                        validate_confirmatory_grader_bundle,
                    )

                    validate_confirmatory_grader_bundle(scorer_path)
                else:
                    load_side_effect_scorer_spec(scorer_path)
            except (DataValidationError, FrozenArtifactError) as exc:
                raise FrozenArtifactError(f"invalid frozen grader bundle: {exc}") from exc
        if self.contract.phase is ExperimentPhase.E7:
            descriptor = inputs["frozen_sae_seed_runs"]
            assert isinstance(descriptor, Mapping)
            sae_path = Path(str(descriptor["location"]))
            if not sae_path.is_absolute():
                sae_path = (self.directory / sae_path).resolve()
            load_sae_stability_bundle(sae_path)
            corpus_descriptor = inputs["separate_sae_corpus"]
            assert isinstance(corpus_descriptor, Mapping)
            corpus_path = Path(str(corpus_descriptor["location"]))
            if not corpus_path.is_absolute():
                corpus_path = (self.directory / corpus_path).resolve()
            from mfh.experiments.e7_sparse import validate_separate_sae_corpus

            validate_separate_sae_corpus(
                corpus_path,
                evaluation_question_ids={
                    question_id
                    for values in self.contract.question_ids_by_benchmark.values()
                    for question_id in values
                },
            )
        snapshot_input = (
            "frozen_evaluation_scripts"
            if self.contract.phase is ExperimentPhase.E9
            else "evaluation_scripts"
            if self.contract.phase is ExperimentPhase.E10
            else None
        )
        if snapshot_input is not None:
            descriptor = inputs[snapshot_input]
            assert isinstance(descriptor, Mapping)
            snapshot_path = Path(str(descriptor["location"]))
            if not snapshot_path.is_absolute():
                snapshot_path = (self.directory / snapshot_path).resolve()
            try:
                validate_execution_snapshot(
                    snapshot_path,
                    study_protocol_digest=self.contract.study_protocol_digest,
                    phase=self.contract.phase,
                )
            except DataValidationError as exc:
                raise FrozenArtifactError(f"invalid execution snapshot: {exc}") from exc
        if self.contract.phase in {ExperimentPhase.E9, ExperimentPhase.E10}:
            component_input = _COMPONENT_INPUT[self.contract.phase]
            try:
                _validate_question_bundle(
                    self.directory / "inputs" / _QUESTION_BUNDLE,
                    self.contract,
                )
                _validate_component_selection(
                    self.directory / "inputs" / component_input,
                    self.contract,
                )
            except DataValidationError as exc:
                raise FrozenArtifactError(f"invalid packaged confirmatory input: {exc}") from exc

    def _confirmatory_grader_bundle(self) -> Any:
        if self.contract.phase not in {ExperimentPhase.E9, ExperimentPhase.E10}:
            raise DataValidationError("only confirmatory ledgers have a grader bundle")
        if self._confirmatory_grader_bundle_cache is None:
            from mfh.experiments.confirmatory_graders import (
                validate_confirmatory_grader_bundle,
            )

            name = "frozen_graders" if self.contract.phase is ExperimentPhase.E9 else "grader"
            self._confirmatory_grader_bundle_cache = validate_confirmatory_grader_bundle(
                self.directory / "inputs" / name
            )
        return self._confirmatory_grader_bundle_cache

    def confirmatory_prompts(self) -> Mapping[str, PromptSpec]:
        """Load the immutable, contract-bound prompt texts packaged with E9/E10."""

        if self.contract.phase not in {ExperimentPhase.E9, ExperimentPhase.E10}:
            raise DataValidationError("only confirmatory ledgers package prompt texts")
        return MappingProxyType(
            _load_confirmatory_prompt_snapshot(
                self.directory / "inputs" / "confirmatory-prompts.json",
                self.contract,
            )
        )

    def _confirmatory_execution_public_key(self) -> str:
        bundle = self._confirmatory_grader_bundle()
        if self._confirmatory_execution_key_cache is None:
            self._confirmatory_execution_key_cache = bundle.scorer.execution_public_key
        return self._confirmatory_execution_key_cache

    def _confirmatory_question(self, condition: EvaluationCondition, question_id: str) -> Question:
        key = (condition.benchmark, question_id)
        if key not in self._confirmatory_question_cache:
            questions = tuple(
                read_questions(
                    self.directory / "inputs" / _QUESTION_BUNDLE / f"{condition.benchmark}.jsonl"
                )
            )
            for question in questions:
                cache_key = (question.benchmark, question.question_id)
                if cache_key in self._confirmatory_question_cache:
                    raise FrozenArtifactError("confirmatory question bundle repeats an identifier")
                self._confirmatory_question_cache[cache_key] = question
        try:
            return self._confirmatory_question_cache[key]
        except KeyError as exc:  # pragma: no cover - contract validation precedes this
            raise FrozenArtifactError("confirmatory question is missing") from exc

    def _confirmatory_component_path(self, condition: EvaluationCondition) -> Path:
        key = (condition.model_name, condition.steering_method)
        if key not in self._confirmatory_component_cache:
            component_name = _COMPONENT_INPUT[self.contract.phase]
            root = self.directory / "inputs" / component_name
            try:
                manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
                descriptors = manifest["components"]
            except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
                raise FrozenArtifactError(f"cannot locate confirmatory component: {exc}") from exc
            for descriptor in descriptors:
                if not isinstance(descriptor, Mapping):
                    raise FrozenArtifactError("confirmatory component descriptor is invalid")
                descriptor_key = (
                    str(descriptor.get("model_name")),
                    str(descriptor.get("method")),
                )
                relative = descriptor.get("component_path")
                if not isinstance(relative, str):
                    raise FrozenArtifactError("confirmatory component path is invalid")
                self._confirmatory_component_cache[descriptor_key] = root / relative / "artifact"
        try:
            return self._confirmatory_component_cache[key]
        except KeyError as exc:
            raise FrozenArtifactError("confirmatory component is missing") from exc

    def _validate_record(self, record: GenerationRecord) -> None:
        condition = self._conditions.get(record.condition_id)
        if condition is None:
            raise DataValidationError(
                f"record references unknown condition {record.condition_id!r}"
            )
        allowed = self._question_sets[condition.benchmark]
        if record.question_id not in allowed:
            raise DataValidationError(
                f"record question {record.question_id!r} is outside the frozen question set"
            )
        condition.validate_record(record)
        if self.contract.phase in {ExperimentPhase.E9, ExperimentPhase.E10}:
            from mfh.experiments.confirmatory_graders import (
                validate_confirmatory_factual_grade,
            )
            from mfh.experiments.e6_likelihood import _assert_e6_runtime_condition

            bundle = self._confirmatory_grader_bundle()
            _assert_e6_runtime_condition(
                bundle.runtime_attestation,
                model_repository=condition.model_repository,
                model_revision=condition.model_revision,
                quantization=condition.quantization,
                model_num_layers=condition.model_num_layers,
                seed=condition.seed,
                execution_public_key=bundle.scorer.execution_public_key,
            )
            if (
                record.metadata.get("runtime_session_identity_sha256")
                != bundle.runtime_identity_digest
            ):
                raise DataValidationError(
                    "confirmatory row runtime identity differs from its frozen attestation"
                )
            if record.benchmark in {
                "triviaqa",
                "simpleqa_verified",
                "aa_omniscience_public_600",
            }:
                validate_confirmatory_factual_grade(
                    record,
                    self._confirmatory_question(condition, record.question_id),
                    grader_bundle=bundle,
                )
            fixed_component = None
            if condition.steering_method in {"M1", "M2", "M4", "M5"}:
                from mfh.experiments.confirmatory_components import (
                    load_confirmatory_fixed_component,
                )

                fixed_component = load_confirmatory_fixed_component(
                    self._confirmatory_component_path(condition)
                )
            elif condition.steering_method == "M3":
                from mfh.experiments.confirmatory_components import (
                    load_confirmatory_adaptive_component,
                )
                from mfh.experiments.e8_protected import (
                    _validate_e8_adaptive_controller_record,
                )

                adaptive_component = load_confirmatory_adaptive_component(
                    self._confirmatory_component_path(condition)
                )
                try:
                    controller = adaptive_component.controllers[condition.system_prompt_id]
                except KeyError as exc:  # pragma: no cover - creation validation precedes this
                    raise FrozenArtifactError(
                        "confirmatory adaptive component lacks the row prompt"
                    ) from exc
                _validate_e8_adaptive_controller_record(
                    record,
                    condition=condition,
                    controller=controller,
                    controller_artifact_sha256=adaptive_component.fingerprint,
                )
            elif condition.steering_method == "M6":
                from mfh.evaluation.side_effects import (
                    recompute_and_verify_official_metric,
                    verify_official_metric_receipt,
                    verify_safety_score_receipt,
                )
                from mfh.experiments.e8_protected import (
                    validate_wikitext_likelihood_evidence,
                )
                from mfh.experiments.e10_native import (
                    validate_e10_composite_execution_record,
                )
                from mfh.methods.composite import load_composite_policy

                question = self._confirmatory_question(condition, record.question_id)
                component_path = self._confirmatory_component_path(condition)
                artifact_sha256 = condition.method_artifact_sha256
                if artifact_sha256 is None:  # pragma: no cover - contract rejects this
                    raise FrozenArtifactError("M6 condition lacks its artifact identity")
                policy = self._confirmatory_composite_cache.get(artifact_sha256)
                if policy is None:
                    policy = load_composite_policy(component_path)
                    self._confirmatory_composite_cache[artifact_sha256] = policy
                validate_e10_composite_execution_record(
                    record,
                    condition=condition,
                    policy=policy,
                    question=question,
                )
                scorer = bundle.scorer
                if record.benchmark in {"ifeval", "mmlu_pro"}:
                    recompute_and_verify_official_metric(
                        record,
                        question,
                        scorer=scorer,
                        ifeval_evaluator_directory=(bundle.directory / "ifeval-evaluator"),
                    )
                elif record.benchmark == "wikitext103":
                    mean_nll = validate_wikitext_likelihood_evidence(record, question=question)
                    verify_official_metric_receipt(
                        record,
                        metric="wikitext103_negative_log_likelihood",
                        value=mean_nll,
                        scorer_revision=scorer.scorer_revision,
                        execution_public_key=scorer.execution_public_key,
                    )
                elif record.benchmark in {
                    "xstest",
                    "strongreject_or_harmbench",
                }:
                    verify_safety_score_receipt(
                        record,
                        scorer_revision=scorer.scorer_revision,
                        execution_public_key=scorer.execution_public_key,
                    )
                    if record.benchmark == "strongreject_or_harmbench":
                        from mfh.evaluation.strongreject import (
                            validate_strongreject_grade_evidence,
                            validate_strongreject_terminal_failure,
                        )

                        if record.metadata.get("strongreject_grader_failed") is True:
                            validate_strongreject_terminal_failure(
                                record,
                                question,
                                grader_directory=bundle.directory / "strongreject-grader",
                            )
                        else:
                            validate_strongreject_grade_evidence(
                                record,
                                question,
                                grader_directory=bundle.directory / "strongreject-grader",
                            )
            validate_confirmatory_execution_receipt(
                record,
                condition,
                execution_public_key=self._confirmatory_execution_public_key(),
                fixed_component=fixed_component,
                runtime_identity=bundle.runtime_attestation["runtime_identity"],
            )

    def records(self) -> Iterator[GenerationRecord]:
        seen: set[tuple[str, str]] = set()
        for path in self._shard_paths():
            for record in read_generation_records(path):
                self._validate_record(record)
                key = (record.condition_id, record.question_id)
                if key in seen:
                    raise FrozenArtifactError(
                        f"duplicate question/condition across phase shards: {key}"
                    )
                seen.add(key)
                yield record

    def completed_keys(self) -> set[tuple[str, str]]:
        if self._completed_cache is None:
            self._completed_cache = {
                (record.condition_id, record.question_id) for record in self.records()
            }
        return set(self._completed_cache)

    def iter_pending(self) -> Iterator[PendingGeneration]:
        completed = self.completed_keys()
        pending: list[PendingGeneration] = []
        for condition in self.contract.conditions:
            for question_id in self.contract.question_ids_by_benchmark[condition.benchmark]:
                if (condition.condition_id, question_id) not in completed:
                    pending.append(PendingGeneration(condition, question_id))
        pending.sort(
            key=lambda value: hashlib.sha256(
                (
                    f"{self.contract.digest}:{value.condition.condition_id}:{value.question_id}"
                ).encode()
            ).digest()
        )
        yield from pending

    def progress(self) -> tuple[int, int]:
        completed = len(self.completed_keys())
        return completed, self.contract.expected_record_count

    def checkpoint(self, records: Iterable[GenerationRecord]) -> int:
        self._validate_mutation_namespace()
        if any((self.directory / name).exists() for name in ("complete.json", "falsified.json")):
            raise FrozenArtifactError("terminal phase runs cannot accept more records")
        values = tuple(records)
        if not values:
            raise DataValidationError("phase checkpoint cannot be empty")
        existing = self.completed_keys()
        new_keys: set[tuple[str, str]] = set()
        for record in values:
            self._validate_record(record)
            key = (record.condition_id, record.question_id)
            if key in existing or key in new_keys:
                raise DataValidationError(f"phase checkpoint repeats record {key}")
            new_keys.add(key)
        shard_index = len(self._shard_paths())
        path = self.directory / "shards" / f"records-{shard_index:05d}.jsonl"
        write_generation_records(path, values)
        assert self._completed_cache is not None
        self._completed_cache.update(new_keys)
        return len(values)

    def record_set_digest(self) -> str:
        """Return the digest to which every gate result must be bound."""

        return stable_hash({path.name: sha256_file(path) for path in self._shard_paths()})

    def _gate_context(self) -> GateEvaluationContext:
        self._verify_creation_evidence()
        try:
            creation_evidence = json.loads(
                (self.directory / "creation-evidence.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - verified above
            raise FrozenArtifactError(f"cannot reopen phase creation evidence: {exc}") from exc
        prerequisite_ledgers: list[PhaseRunLedger] = []
        reference_phases: tuple[ExperimentPhase, ...] = ()
        if self.contract.phase is ExperimentPhase.E7:
            reference_phases = (ExperimentPhase.E6,)
        elif self.contract.phase is ExperimentPhase.E9:
            reference_phases = (ExperimentPhase.E8,)
        elif self.contract.phase is ExperimentPhase.E10:
            reference_phases = (ExperimentPhase.E8, ExperimentPhase.E9)
        if reference_phases:
            prerequisite_values = creation_evidence.get("prerequisite_runs", {})
            if not isinstance(prerequisite_values, Mapping):
                raise FrozenArtifactError(
                    f"{self.contract.phase.value} prerequisite evidence is invalid"
                )
            for phase in reference_phases:
                descriptor = prerequisite_values.get(phase.value)
                if not isinstance(descriptor, Mapping) or not isinstance(
                    descriptor.get("location"), str
                ):
                    raise FrozenArtifactError(
                        f"{self.contract.phase.value} lacks the verified "
                        f"{phase.value} reference run"
                    )
                ledger = self.open(
                    _resolve_ledger_evidence_path(
                        self.directory,
                        descriptor["location"],
                        context=f"{self.contract.phase.value} {phase.value} prerequisite",
                    ),
                    study=self.study,
                )
                ledger.verify_complete()
                prerequisite_ledgers.append(ledger)

        def reference_records() -> Iterator[GenerationRecord]:
            yield from chain.from_iterable(ledger.records() for ledger in prerequisite_ledgers)

        reference_condition_facts = {
            condition.condition_id: {
                "phase": condition.phase.value,
                "benchmark": condition.benchmark,
                "partition": condition.partition,
                "model_name": condition.model_name,
                "model_repository": condition.model_repository,
                "model_revision": condition.model_revision,
                "runtime": condition.runtime.value,
                "quantization": condition.quantization,
                "model_num_layers": condition.model_num_layers,
                "system_prompt_id": condition.system_prompt_id,
                "prompt_template_sha256": condition.prompt_template_sha256,
                "steering_method": condition.steering_method,
                "method_artifact_sha256": condition.method_artifact_sha256,
                "comparison_group": condition.comparison_group,
                "layer": condition.layer,
                "site": condition.site.value if condition.site is not None else None,
                "token_scope": (
                    condition.token_scope.value if condition.token_scope is not None else None
                ),
                "alpha": condition.alpha,
                "sparsity": condition.sparsity,
                "seed": condition.seed,
            }
            for ledger in prerequisite_ledgers
            for condition in ledger.contract.conditions
        }
        reference_input_fingerprints = {
            ledger.contract.phase.value: dict(ledger.contract.input_fingerprints)
            for ledger in prerequisite_ledgers
        }
        e8_matching_basis: Mapping[str, Any] = {}
        if self.contract.phase is ExperimentPhase.E9:
            from mfh.experiments.e9_analysis import derive_e8_matching_basis

            e8_matching_basis = derive_e8_matching_basis(prerequisite_ledgers[0])

        condition_facts = {
            condition.condition_id: {
                "phase": condition.phase.value,
                "benchmark": condition.benchmark,
                "partition": condition.partition,
                "model_name": condition.model_name,
                "model_repository": condition.model_repository,
                "model_revision": condition.model_revision,
                "runtime": condition.runtime.value,
                "quantization": condition.quantization,
                "model_num_layers": condition.model_num_layers,
                "system_prompt_id": condition.system_prompt_id,
                "prompt_template_sha256": condition.prompt_template_sha256,
                "steering_method": condition.steering_method,
                "method_artifact_sha256": condition.method_artifact_sha256,
                "adaptive_policy": (
                    condition.adaptive_policy.to_dict()
                    if condition.adaptive_policy is not None
                    else None
                ),
                "comparison_group": condition.comparison_group,
                "layer": condition.layer,
                "site": condition.site.value if condition.site is not None else None,
                "token_scope": (
                    condition.token_scope.value if condition.token_scope is not None else None
                ),
                "alpha": condition.alpha,
                "sparsity": condition.sparsity,
                "seed": condition.seed,
            }
            for condition in self.contract.conditions
        }
        reservation = creation_evidence.get("one_shot_reservation")
        registry_sealed = False
        if isinstance(reservation, Mapping) and isinstance(reservation.get("location"), str):
            reservation_path = Path(reservation["location"])
            registry_sealed = (
                reservation_path.is_file() and reservation_path.stat().st_mode & 0o222 == 0
            )
        input_descriptors = creation_evidence.get("input_artifacts")
        if not isinstance(input_descriptors, Mapping):  # pragma: no cover - verified above
            raise FrozenArtifactError("phase creation evidence lacks input descriptors")
        creation_inputs: dict[str, str] = {}
        live_inputs: dict[str, str] = {}
        input_paths: dict[str, Path] = {}
        for name, raw_descriptor in input_descriptors.items():
            if not isinstance(name, str) or not isinstance(raw_descriptor, Mapping):
                raise FrozenArtifactError("phase input descriptor is invalid")
            location = Path(str(raw_descriptor["location"]))
            path = location if location.is_absolute() else (self.directory / location).resolve()
            creation_inputs[name] = str(raw_descriptor["fingerprint"])
            live_inputs[name] = sha256_path(path)
            input_paths[name] = path
        scorer_input = (
            "frozen_side_effect_scorers"
            if self.contract.phase in {ExperimentPhase.E7, ExperimentPhase.E8}
            else "frozen_graders"
            if self.contract.phase is ExperimentPhase.E9
            else "grader"
            if self.contract.phase is ExperimentPhase.E10
            else None
        )
        scorer = None
        if scorer_input is not None:
            if self.contract.phase in {ExperimentPhase.E9, ExperimentPhase.E10}:
                from mfh.experiments.confirmatory_graders import (
                    validate_confirmatory_grader_bundle,
                )

                scorer = validate_confirmatory_grader_bundle(input_paths[scorer_input]).scorer
            else:
                scorer = load_side_effect_scorer_spec(input_paths[scorer_input])
        sae_bundle = (
            load_sae_stability_bundle(input_paths["frozen_sae_seed_runs"])
            if self.contract.phase is ExperimentPhase.E7
            else None
        )
        snapshot_input = (
            "frozen_evaluation_scripts"
            if self.contract.phase is ExperimentPhase.E9
            else "evaluation_scripts"
            if self.contract.phase is ExperimentPhase.E10
            else None
        )
        snapshot_verified = False
        analysis_protocol = None
        if snapshot_input is not None:
            validate_execution_snapshot(
                input_paths[snapshot_input],
                study_protocol_digest=self.contract.study_protocol_digest,
                phase=self.contract.phase,
            )
            snapshot_verified = True
            if self.contract.phase is ExperimentPhase.E9:
                from mfh.experiments.e9_analysis import (
                    load_e9_analysis_protocol_snapshot,
                )

                analysis_protocol, analysis_snapshot_sha256 = load_e9_analysis_protocol_snapshot(
                    input_paths[snapshot_input],
                    study_protocol_digest=self.contract.study_protocol_digest,
                )
                if analysis_snapshot_sha256 != self.contract.input_fingerprints.get(snapshot_input):
                    raise FrozenArtifactError(
                        "E9 analysis protocol snapshot differs from its contract"
                    )
            elif self.contract.phase is ExperimentPhase.E10:
                from mfh.experiments.snapshots import (
                    load_snapshot_analysis_protocol,
                )

                analysis_protocol, analysis_snapshot_sha256 = load_snapshot_analysis_protocol(
                    input_paths[snapshot_input],
                    study_protocol_digest=self.contract.study_protocol_digest,
                    phase=ExperimentPhase.E10,
                )
                if analysis_snapshot_sha256 != self.contract.input_fingerprints.get(snapshot_input):
                    raise FrozenArtifactError(
                        "E10 analysis protocol snapshot differs from its contract"
                    )
        input_snapshots_match = (
            dict(self.contract.input_fingerprints) == creation_inputs == live_inputs
        )
        parameter_names = set(self.contract.input_fingerprints) - {"evaluation_scripts"}
        parameter_snapshots_match = all(
            self.contract.input_fingerprints.get(name)
            == creation_inputs.get(name)
            == live_inputs.get(name)
            for name in parameter_names
        )
        return GateEvaluationContext(
            expected_record_count=self.contract.expected_record_count,
            records_factory=self.records,
            expected_condition_ids=frozenset(condition_facts),
            condition_facts=condition_facts,
            reference_records_factory=reference_records,
            reference_condition_facts=reference_condition_facts,
            reference_input_fingerprints=reference_input_fingerprints,
            input_fingerprints=self.contract.input_fingerprints,
            creation_input_fingerprints=creation_inputs,
            live_input_fingerprints=live_inputs,
            frozen_inputs_verified=True,
            code_snapshot_verified=(
                self.contract.phase is ExperimentPhase.E10
                and snapshot_verified
                and input_snapshots_match
            ),
            parameter_snapshot_verified=(
                self.contract.phase is ExperimentPhase.E10 and parameter_snapshots_match
            ),
            preregistration_verified=(
                self.contract.phase is ExperimentPhase.E9
                and snapshot_verified
                and input_snapshots_match
            ),
            analysis_protocol=analysis_protocol,
            prerequisite_completion_digests=self.contract.prerequisite_digests,
            e8_matching_basis=e8_matching_basis,
            one_shot_registry_sealed=registry_sealed,
            side_effect_scorer_public_key=(
                scorer.execution_public_key if scorer is not None else None
            ),
            side_effect_scorer_revision=(scorer.scorer_revision if scorer is not None else None),
            sae_stability_selections=(
                {
                    model: tuple(
                        (
                            item.seed,
                            item.checkpoint_fingerprint,
                            item.selected_features,
                        )
                        for item in selections
                    )
                    for model, selections in sae_bundle.selections_by_model.items()
                }
                if sae_bundle is not None
                else {}
            ),
            sae_stability_scores=(sae_bundle.stability_by_model if sae_bundle is not None else {}),
            sae_promoted_method_artifacts=(
                sae_bundle.promoted_method_artifacts if sae_bundle is not None else {}
            ),
        )

    def evaluate_gate(
        self,
        gate: str,
        evidence_path: str | Path,
        *,
        supporting_artifacts: Mapping[str, str | Path] | None = None,
    ) -> GateResult:
        """Evaluate raw gate evidence with facts from this verified ledger."""

        if gate not in self.contract.required_gates:
            raise DataValidationError(f"gate {gate!r} is not required by this phase")
        return evaluate_registered_gate(
            phase=self.contract.phase,
            gate=gate,
            contract_digest=self.contract.digest,
            record_set_digest=self.record_set_digest(),
            evidence_path=evidence_path,
            context=self._gate_context(),
            supporting_artifacts=supporting_artifacts,
        )

    def _validate_gate_result(
        self,
        name: str,
        result: GateResult,
        record_set: str,
        *,
        require_passing: bool = True,
    ) -> None:
        if result.artifact_paths:
            evidence_path = result.artifact_paths.get("evaluation")
        else:
            evidence_path = self.directory / "gate-artifacts" / name / "evaluation"
        if evidence_path is None:
            raise DataValidationError(f"gate {name!r} has no raw evaluation artifact")
        validate_gate_result(
            result,
            evidence_path=evidence_path,
            context=self._gate_context(),
        )
        if (
            result.gate != name
            or result.phase is not self.contract.phase
            or (require_passing and not result.passed)
            or result.contract_digest != self.contract.digest
            or result.record_set_digest != record_set
        ):
            raise DataValidationError(
                f"gate result {name!r} is not valid evidence for this exact phase run"
            )

    def _package_e0_completion_receipt(
        self,
        evidence: object | None,
    ) -> dict[str, str] | None:
        if self.contract.phase is not ExperimentPhase.E0:
            if evidence is not None:
                raise DataValidationError("only E0 may bind a scientific completion receipt")
            return None
        if evidence is None:
            raise DataValidationError(
                "E0 finalization requires a live-verified scientific completion receipt"
            )
        from mfh.experiments.e0_completion import (
            _assert_authorized_e0_completion,
            validate_e0_completion_receipt_snapshot,
        )

        evidence = _assert_authorized_e0_completion(evidence)
        source = evidence.directory
        manifest = validate_e0_completion_receipt_snapshot(source)
        if (
            manifest.get("manifest_digest") != evidence.manifest_digest
            or sha256_path(source) != evidence.fingerprint
        ):
            raise DataValidationError("live-verified E0 scientific completion receipt changed")
        fingerprint = sha256_path(source)
        destination = self.directory.resolve() / "scientific-completion-receipt"
        _copy_frozen_artifact(source, destination, fingerprint)
        packaged = validate_e0_completion_receipt_snapshot(destination)
        if packaged.get("manifest_digest") != evidence.manifest_digest:
            raise FrozenArtifactError("packaged E0 scientific completion receipt differs")
        return {
            "manifest_digest": evidence.manifest_digest,
            "fingerprint": fingerprint,
        }

    def _verify_e0_completion_receipt(self, descriptor: object) -> dict[str, str] | None:
        if self.contract.phase is not ExperimentPhase.E0:
            if descriptor is not None:
                raise FrozenArtifactError("non-E0 completion binds an E0 scientific receipt")
            return None
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
            "manifest_digest",
            "fingerprint",
        }:
            raise FrozenArtifactError("E0 completion lacks its scientific receipt")
        manifest_digest = descriptor.get("manifest_digest")
        fingerprint = descriptor.get("fingerprint")
        if type(manifest_digest) is not str or type(fingerprint) is not str:
            raise FrozenArtifactError("E0 scientific completion receipt identity is invalid")
        if not _SHA256.fullmatch(manifest_digest) or not _SHA256.fullmatch(fingerprint):
            raise FrozenArtifactError("E0 scientific completion receipt identity is invalid")
        destination = self.directory.resolve() / "scientific-completion-receipt"
        try:
            from mfh.experiments.e0_completion import validate_e0_completion_receipt_snapshot

            manifest = validate_e0_completion_receipt_snapshot(destination)
            actual_fingerprint = sha256_path(destination)
        except (OSError, DataValidationError, FrozenArtifactError) as exc:
            raise FrozenArtifactError(
                f"cannot verify E0 scientific completion receipt: {exc}"
            ) from exc
        if manifest.get("manifest_digest") != manifest_digest or actual_fingerprint != fingerprint:
            raise FrozenArtifactError("E0 scientific completion receipt changed")
        return {
            "manifest_digest": manifest_digest,
            "fingerprint": fingerprint,
        }

    def finalize(
        self,
        gate_results: Mapping[str, GateResult],
        *,
        verified_e0_completion: object | None = None,
    ) -> PhaseCompletion:
        self._validate_mutation_namespace()
        self.contract.assert_matches_study(self.study)
        marker = self.directory / "complete.json"
        if marker.exists() or (self.directory / "falsified.json").exists():
            raise FrozenArtifactError("phase run already has a terminal result")
        completed, expected = self.progress()
        if completed != expected:
            raise DataValidationError(
                f"cannot finalize phase with {expected - completed} missing records"
            )
        validate_adaptive_execution(self.records())
        if set(gate_results) != set(self.contract.required_gates):
            raise DataValidationError("phase gate results differ from the required gate set")
        shard_fingerprints = {path.name: sha256_file(path) for path in self._shard_paths()}
        record_set_digest = stable_hash(shard_fingerprints)
        for name, result in gate_results.items():
            self._validate_gate_result(name, result, record_set_digest)
        scientific_completion_receipt = self._package_e0_completion_receipt(
            verified_e0_completion,
        )
        self._package_gate_artifacts(gate_results)
        for name, result in gate_results.items():
            write_gate_result(self.directory / "gates" / f"{name}.json", result)
        gate_result_digests = {
            name: gate_results[name].gate_digest for name in sorted(gate_results)
        }
        gate_file_fingerprints = {path.name: sha256_file(path) for path in self._gate_paths()}
        gate_artifact_fingerprints = self._gate_artifact_fingerprints(gate_results)
        body = {
            "schema_version": 1,
            "phase": self.contract.phase.value,
            "contract_digest": self.contract.digest,
            "record_count": completed,
            "shard_fingerprints": shard_fingerprints,
            "record_set_digest": record_set_digest,
            "gate_result_digests": gate_result_digests,
            "gate_file_fingerprints": gate_file_fingerprints,
            "gate_artifact_fingerprints": gate_artifact_fingerprints,
            "scientific_completion_receipt": scientific_completion_receipt,
        }
        completion_digest = stable_hash(body)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".complete-", suffix=".json", dir=self.directory
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(
                    {**body, "completion_digest": completion_digest},
                    handle,
                    indent=2,
                    sort_keys=True,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary_name, marker)
        except FileExistsError:
            raise FrozenArtifactError("phase run was finalized concurrently") from None
        finally:
            Path(temporary_name).unlink(missing_ok=True)
        return PhaseCompletion(
            phase=self.contract.phase,
            contract_digest=self.contract.digest,
            record_count=completed,
            shard_fingerprints=MappingProxyType(shard_fingerprints),
            record_set_digest=record_set_digest,
            gate_result_digests=MappingProxyType(gate_result_digests),
            gate_file_fingerprints=MappingProxyType(gate_file_fingerprints),
            gate_artifact_fingerprints=MappingProxyType(gate_artifact_fingerprints),
            completion_digest=completion_digest,
        )

    def finalize_falsified(
        self,
        gate_results: Mapping[str, GateResult],
        *,
        verified_e0_completion: object | None = None,
    ) -> PhaseFalsification:
        """Freeze a complete run with at least one failed empirical gate.

        The resulting ``falsified.json`` is independently replayable, but creation of
        downstream ledgers continues to require ``verify_complete()`` and therefore
        can never treat this terminal result as a satisfied prerequisite.
        """

        self._validate_mutation_namespace()
        self.contract.assert_matches_study(self.study)
        marker = self.directory / "falsified.json"
        if marker.exists() or (self.directory / "complete.json").exists():
            raise FrozenArtifactError("phase run already has a terminal result")
        completed, expected = self.progress()
        if completed != expected:
            raise DataValidationError(
                f"cannot finalize falsification with {expected - completed} missing records"
            )
        validate_adaptive_execution(self.records())
        if set(gate_results) != set(self.contract.required_gates):
            raise DataValidationError("phase gate results differ from the required gate set")
        shard_fingerprints = {path.name: sha256_file(path) for path in self._shard_paths()}
        record_set_digest = stable_hash(shard_fingerprints)
        for name, result in gate_results.items():
            self._validate_gate_result(
                name,
                result,
                record_set_digest,
                require_passing=False,
            )
        failed_gates = tuple(
            sorted(name for name, result in gate_results.items() if not result.passed)
        )
        if not failed_gates:
            raise DataValidationError(
                "falsified finalization requires at least one failed empirical gate"
            )
        scientific_completion_receipt = self._package_e0_completion_receipt(
            verified_e0_completion,
        )
        self._package_gate_artifacts(gate_results)
        for name, result in gate_results.items():
            write_gate_result(self.directory / "gates" / f"{name}.json", result)
        gate_result_digests = {
            name: gate_results[name].gate_digest for name in sorted(gate_results)
        }
        gate_file_fingerprints = {path.name: sha256_file(path) for path in self._gate_paths()}
        gate_artifact_fingerprints = self._gate_artifact_fingerprints(gate_results)
        body = {
            "schema_version": 1,
            "status": "falsified",
            "phase": self.contract.phase.value,
            "contract_digest": self.contract.digest,
            "record_count": completed,
            "shard_fingerprints": shard_fingerprints,
            "record_set_digest": record_set_digest,
            "gate_result_digests": gate_result_digests,
            "gate_file_fingerprints": gate_file_fingerprints,
            "gate_artifact_fingerprints": gate_artifact_fingerprints,
            "failed_gates": list(failed_gates),
            "scientific_completion_receipt": scientific_completion_receipt,
        }
        falsification_digest = stable_hash(body)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".falsified-", suffix=".json", dir=self.directory
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(
                    {**body, "falsification_digest": falsification_digest},
                    handle,
                    indent=2,
                    sort_keys=True,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary_name, marker)
        except FileExistsError:
            raise FrozenArtifactError("phase run was finalized concurrently") from None
        finally:
            Path(temporary_name).unlink(missing_ok=True)
        return PhaseFalsification(
            phase=self.contract.phase,
            contract_digest=self.contract.digest,
            record_count=completed,
            shard_fingerprints=MappingProxyType(shard_fingerprints),
            record_set_digest=record_set_digest,
            gate_result_digests=MappingProxyType(gate_result_digests),
            gate_file_fingerprints=MappingProxyType(gate_file_fingerprints),
            gate_artifact_fingerprints=MappingProxyType(gate_artifact_fingerprints),
            failed_gates=failed_gates,
            falsification_digest=falsification_digest,
        )

    def verify_complete(self) -> PhaseCompletion:
        marker = self.directory / "complete.json"
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FrozenArtifactError(f"cannot read phase completion marker: {exc}") from exc
        digest = payload.pop("completion_digest", None)
        if digest != stable_hash(payload):
            raise FrozenArtifactError("phase completion digest mismatch")
        expected_keys = {
            "schema_version",
            "phase",
            "contract_digest",
            "record_count",
            "shard_fingerprints",
            "record_set_digest",
            "gate_result_digests",
            "gate_file_fingerprints",
            "gate_artifact_fingerprints",
            "scientific_completion_receipt",
        }
        if set(payload) != expected_keys or payload.get("schema_version") != 1:
            raise FrozenArtifactError("phase completion marker has an invalid schema")
        actual_fingerprints = {path.name: sha256_file(path) for path in self._shard_paths()}
        if payload["shard_fingerprints"] != actual_fingerprints:
            raise FrozenArtifactError("completed phase-run shards changed")
        actual_record_set_digest = stable_hash(actual_fingerprints)
        if payload["record_set_digest"] != actual_record_set_digest:
            raise FrozenArtifactError("completed phase-run record-set identity changed")
        gate_paths = self._gate_paths()
        expected_gate_filenames = {f"{name}.json" for name in self.contract.required_gates}
        if {path.name for path in gate_paths} != expected_gate_filenames:
            raise FrozenArtifactError("completed phase-run gate evidence differs from the contract")
        actual_gate_digests: dict[str, str] = {}
        for path in gate_paths:
            name = path.stem
            result = read_gate_result(path)
            try:
                self._validate_gate_result(name, result, actual_record_set_digest)
            except DataValidationError as exc:
                raise FrozenArtifactError(str(exc)) from exc
            actual_gate_digests[name] = result.gate_digest
        if payload["gate_result_digests"] != actual_gate_digests:
            raise FrozenArtifactError("completed phase-run gate-result identities changed")
        actual_gate_fingerprints = {path.name: sha256_file(path) for path in gate_paths}
        if payload["gate_file_fingerprints"] != actual_gate_fingerprints:
            raise FrozenArtifactError("completed phase-run gate-evidence files changed")
        actual_gate_artifact_fingerprints = self._gate_artifact_fingerprints(
            {path.stem: read_gate_result(path) for path in gate_paths}
        )
        if payload["gate_artifact_fingerprints"] != actual_gate_artifact_fingerprints:
            raise FrozenArtifactError("completed phase-run gate artifacts changed")
        self._verify_e0_completion_receipt(payload["scientific_completion_receipt"])
        record_count = sum(1 for _ in self.records())
        try:
            validate_adaptive_execution(self.records())
        except DataValidationError as exc:
            raise FrozenArtifactError(str(exc)) from exc
        if (
            payload["phase"] != self.contract.phase.value
            or payload["contract_digest"] != self.contract.digest
            or int(payload["record_count"]) != record_count
            or record_count != self.contract.expected_record_count
        ):
            raise FrozenArtifactError("phase completion marker differs from the run")
        return PhaseCompletion(
            phase=self.contract.phase,
            contract_digest=self.contract.digest,
            record_count=record_count,
            shard_fingerprints=MappingProxyType(actual_fingerprints),
            record_set_digest=actual_record_set_digest,
            gate_result_digests=MappingProxyType(actual_gate_digests),
            gate_file_fingerprints=MappingProxyType(actual_gate_fingerprints),
            gate_artifact_fingerprints=MappingProxyType(actual_gate_artifact_fingerprints),
            completion_digest=str(digest),
        )

    def verify_falsified(self) -> PhaseFalsification:
        """Replay and verify an auditable negative terminal result."""

        if (self.directory / "complete.json").exists():
            raise FrozenArtifactError("a completed phase cannot also be falsified")
        marker = self.directory / "falsified.json"
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FrozenArtifactError(f"cannot read phase falsification marker: {exc}") from exc
        digest = payload.pop("falsification_digest", None)
        if digest != stable_hash(payload):
            raise FrozenArtifactError("phase falsification digest mismatch")
        expected_keys = {
            "schema_version",
            "status",
            "phase",
            "contract_digest",
            "record_count",
            "shard_fingerprints",
            "record_set_digest",
            "gate_result_digests",
            "gate_file_fingerprints",
            "gate_artifact_fingerprints",
            "failed_gates",
            "scientific_completion_receipt",
        }
        if (
            set(payload) != expected_keys
            or payload.get("schema_version") != 1
            or payload.get("status") != "falsified"
        ):
            raise FrozenArtifactError("phase falsification marker has an invalid schema")
        failed_value = payload["failed_gates"]
        if (
            not isinstance(failed_value, list)
            or not failed_value
            or any(not isinstance(value, str) or not value for value in failed_value)
            or failed_value != sorted(set(failed_value))
        ):
            raise FrozenArtifactError("phase falsification has invalid failed-gate identities")
        actual_fingerprints = {path.name: sha256_file(path) for path in self._shard_paths()}
        if payload["shard_fingerprints"] != actual_fingerprints:
            raise FrozenArtifactError("falsified phase-run shards changed")
        actual_record_set_digest = stable_hash(actual_fingerprints)
        if payload["record_set_digest"] != actual_record_set_digest:
            raise FrozenArtifactError("falsified phase-run record-set identity changed")
        gate_paths = self._gate_paths()
        expected_gate_filenames = {f"{name}.json" for name in self.contract.required_gates}
        if {path.name for path in gate_paths} != expected_gate_filenames:
            raise FrozenArtifactError("falsified phase-run gate evidence differs from the contract")
        actual_gate_digests: dict[str, str] = {}
        actual_gate_results: dict[str, GateResult] = {}
        for path in gate_paths:
            name = path.stem
            result = read_gate_result(path)
            try:
                self._validate_gate_result(
                    name,
                    result,
                    actual_record_set_digest,
                    require_passing=False,
                )
            except DataValidationError as exc:
                raise FrozenArtifactError(str(exc)) from exc
            actual_gate_results[name] = result
            actual_gate_digests[name] = result.gate_digest
        actual_failed = tuple(
            sorted(name for name, result in actual_gate_results.items() if not result.passed)
        )
        if tuple(failed_value) != actual_failed:
            raise FrozenArtifactError("phase falsification failed-gate identities changed")
        if payload["gate_result_digests"] != actual_gate_digests:
            raise FrozenArtifactError("falsified phase-run gate-result identities changed")
        actual_gate_fingerprints = {path.name: sha256_file(path) for path in gate_paths}
        if payload["gate_file_fingerprints"] != actual_gate_fingerprints:
            raise FrozenArtifactError("falsified phase-run gate-evidence files changed")
        actual_gate_artifact_fingerprints = self._gate_artifact_fingerprints(actual_gate_results)
        if payload["gate_artifact_fingerprints"] != actual_gate_artifact_fingerprints:
            raise FrozenArtifactError("falsified phase-run gate artifacts changed")
        self._verify_e0_completion_receipt(payload["scientific_completion_receipt"])
        record_count = sum(1 for _ in self.records())
        try:
            validate_adaptive_execution(self.records())
        except DataValidationError as exc:
            raise FrozenArtifactError(str(exc)) from exc
        if (
            payload["phase"] != self.contract.phase.value
            or payload["contract_digest"] != self.contract.digest
            or int(payload["record_count"]) != record_count
            or record_count != self.contract.expected_record_count
        ):
            raise FrozenArtifactError("phase falsification marker differs from the run")
        return PhaseFalsification(
            phase=self.contract.phase,
            contract_digest=self.contract.digest,
            record_count=record_count,
            shard_fingerprints=MappingProxyType(actual_fingerprints),
            record_set_digest=actual_record_set_digest,
            gate_result_digests=MappingProxyType(actual_gate_digests),
            gate_file_fingerprints=MappingProxyType(actual_gate_fingerprints),
            gate_artifact_fingerprints=MappingProxyType(actual_gate_artifact_fingerprints),
            failed_gates=actual_failed,
            falsification_digest=str(digest),
        )


def package_portable_phase_ledger(
    source_directory: str | Path,
    destination: str | Path,
    *,
    study: StudyProtocol,
) -> str:
    """Recursively package one ledger, all inputs, and all prerequisite ledgers."""

    normalized_paths = validate_active_study_artifact_paths(
        {"source phase ledger": source_directory, "portable phase ledger": destination}
    )
    source = normalized_paths["source phase ledger"]
    target = normalized_paths["portable phase ledger"]
    if target.exists() or target.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite portable ledger: {target}")
    ledger = PhaseRunLedger.open(source, study=study)
    try:
        creation = json.loads((source / "creation-evidence.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot package ledger creation evidence: {exc}") from exc
    if not isinstance(creation, dict):
        raise FrozenArtifactError("portable ledger creation evidence is invalid")
    _copy_frozen_artifact(source, target, sha256_path(source))
    inputs = creation.get("input_artifacts")
    prerequisites = creation.get("prerequisite_runs")
    if not isinstance(inputs, Mapping) or not isinstance(prerequisites, Mapping):
        raise FrozenArtifactError("portable ledger evidence lacks inputs or prerequisites")
    rewritten_inputs: dict[str, dict[str, str]] = {}
    for name, raw_descriptor in sorted(inputs.items()):
        if not isinstance(name, str) or not isinstance(raw_descriptor, Mapping):
            raise FrozenArtifactError("portable ledger input descriptor is invalid")
        fingerprint = raw_descriptor.get("fingerprint")
        if type(fingerprint) is not str or _SHA256.fullmatch(fingerprint) is None:
            raise FrozenArtifactError("portable ledger input fingerprint is invalid")
        input_source = _resolve_ledger_evidence_path(
            source,
            raw_descriptor.get("location"),
            context=f"portable input {name!r}",
        )
        relative = Path("portable-inputs") / name
        _copy_frozen_artifact(input_source, target / relative, fingerprint)
        rewritten_inputs[name] = {
            "location": relative.as_posix(),
            "fingerprint": fingerprint,
        }
    rewritten_prerequisites: dict[str, dict[str, str]] = {}
    for name, raw_descriptor in sorted(prerequisites.items()):
        if not isinstance(name, str) or not isinstance(raw_descriptor, Mapping):
            raise FrozenArtifactError("portable prerequisite descriptor is invalid")
        completion_digest = raw_descriptor.get("completion_digest")
        if type(completion_digest) is not str or _SHA256.fullmatch(completion_digest) is None:
            raise FrozenArtifactError("portable prerequisite completion is invalid")
        prior_source = _resolve_ledger_evidence_path(
            source,
            raw_descriptor.get("location"),
            context=f"portable prerequisite {name!r}",
        )
        relative = Path("portable-prerequisites") / name
        prerequisite_phase = ExperimentPhase(name)
        if prerequisite_phase is ExperimentPhase.E3:
            open_phase_prerequisite(
                prior_source,
                phase=prerequisite_phase,
                study=study,
                expected_completion_digest=completion_digest,
            )
            _copy_frozen_artifact(
                prior_source,
                target / relative,
                sha256_path(prior_source),
            )
        else:
            package_portable_phase_ledger(
                prior_source,
                target / relative,
                study=study,
            )
        rewritten_prerequisites[name] = {
            "location": relative.as_posix(),
            "completion_digest": completion_digest,
        }
    evidence_body = {
        "schema_version": creation.get("schema_version"),
        "input_artifacts": rewritten_inputs,
        "prerequisite_runs": rewritten_prerequisites,
        "scientific_input_authorizations": creation.get("scientific_input_authorizations"),
        "one_shot_reservation": creation.get("one_shot_reservation"),
    }
    (target / "creation-evidence.json").write_text(
        json.dumps(
            {**evidence_body, "evidence_digest": stable_hash(evidence_body)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    packaged = PhaseRunLedger.open(target, study=study)
    if packaged.contract.digest != ledger.contract.digest:
        raise FrozenArtifactError("portable ledger contract changed")
    return sha256_path(target)

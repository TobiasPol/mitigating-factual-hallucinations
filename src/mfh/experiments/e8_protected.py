"""Provenance-bound E8 protected-direction construction and terminal replay."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from safetensors.torch import load_file, save_file
from torch import Tensor

from mfh.contracts import (
    ActivationSite,
    AdaptivePolicySpec,
    GenerationRecord,
    Outcome,
    PromptSpec,
    Question,
    TokenScope,
)
from mfh.data.io import read_questions
from mfh.data.normalization import normalize_answer
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.evaluation.grading import deterministic_short_answer_grade
from mfh.experiments.model_selection import validate_active_study_artifact_paths
from mfh.experiments.runtime_evidence import (
    build_generation_runtime_metrics,
    validate_generation_runtime_metrics,
)
from mfh.inference.vllm_runtime import as_numpy
from mfh.methods.features import (
    ActivationFeatureSchema,
    ActivationKind,
    FeatureComposition,
)
from mfh.methods.protected import (
    BehaviorDirection,
    ProtectedSubspace,
    behavior_covariance,
    build_behavior_direction,
    build_protected_subspace,
    covariance_aware_direction,
)
from mfh.methods.sparse import (
    load_coordinate_sparse_artifact,
    load_sae_intervention,
)
from mfh.provenance import canonical_json, sha256_file, sha256_path, stable_hash

_BEHAVIORS = (
    "correct_to_abstain",
    "xstest_safe_refusal",
    "harmful_refusal",
    "language_switching",
    "instruction_following_failure",
    "verbosity_style",
)
_VARIANTS = ("orthogonal_projection", "covariance_aware")
_BEHAVIOR_BENCHMARKS = {
    "correct_to_abstain": "triviaqa",
    "xstest_safe_refusal": "xstest",
    "harmful_refusal": "strongreject_or_harmbench",
    "language_switching": "language_consistency",
    "instruction_following_failure": "ifeval",
    "verbosity_style": "ifeval",
}
_COVARIANCE_ESTIMATOR = "within-behavior-class-centered-population-v1"
_SHA256_LENGTH = 64


def _sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _matrix(value: Tensor, *, width: int | None = None) -> Tensor:
    result = value.detach().cpu().float().contiguous().clone()
    if (
        result.ndim != 2
        or result.shape[0] == 0
        or result.shape[1] == 0
        or (width is not None and result.shape[1] != width)
        or not torch.isfinite(result).all()
    ):
        raise DataValidationError("E8 protected activations must be one finite matrix")
    return result


def _unit(value: Tensor, *, width: int | None = None) -> Tensor:
    result = value.detach().cpu().float().contiguous().clone()
    norm = torch.linalg.vector_norm(result)
    if (
        result.ndim != 1
        or result.numel() == 0
        or (width is not None and result.numel() != width)
        or not torch.isfinite(result).all()
        or not math.isclose(float(norm), 1.0, rel_tol=1e-5, abs_tol=1e-6)
    ):
        raise DataValidationError("E8 protected direction must be one finite unit vector")
    return result


def question_source_fingerprint(question: Question) -> str:
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


def response_verbosity_style_signature(text: str) -> tuple[str, str, int, int]:
    """Return a deterministic, response-bound coarse verbosity/style signature."""

    normalized = text.strip().replace("\r\n", "\n").replace("\r", "\n")
    words = re.findall(r"[^\W_]+(?:['\u2019][^\W_]+)?", normalized, flags=re.UNICODE)
    word_count = len(words)
    verbosity = (
        "empty"
        if word_count == 0
        else "short"
        if word_count <= 32
        else "medium"
        if word_count <= 96
        else "long"
    )
    lines = tuple(line.strip() for line in normalized.splitlines() if line.strip())
    if "```" in normalized:
        structure = "code"
    elif any(re.match(r"^(?:[-*+] |\d+[.)] )", line) for line in lines):
        structure = "list"
    elif any(line.startswith("#") for line in lines):
        structure = "headed"
    else:
        structure = "prose"
    paragraphs = 0 if not normalized else min(3, len(re.split(r"\n\s*\n", normalized)))
    sentences = min(4, len(re.findall(r"[.!?]+(?:\s|$)", normalized)))
    return verbosity, structure, paragraphs, sentences


def response_verbosity_style_preserved(before: str, after: str) -> bool:
    """Check whether steering preserves coarse verbosity and presentation style."""

    return response_verbosity_style_signature(before) == response_verbosity_style_signature(after)


@dataclass(frozen=True, slots=True)
class BehaviorLabelPair:
    """One E7 baseline/intervention pair with a deterministically derived label."""

    baseline_record: GenerationRecord
    intervention_record: GenerationRecord
    label: str

    def __post_init__(self) -> None:
        if (
            type(self.baseline_record) is not GenerationRecord
            or type(self.intervention_record) is not GenerationRecord
            or self.label not in {"positive", "negative"}
            or self.baseline_record.question_id != self.intervention_record.question_id
            or self.baseline_record.benchmark != self.intervention_record.benchmark
            or self.baseline_record.steering_method != "M0"
            or self.intervention_record.steering_method != "M4b"
            or self.baseline_record.condition_id == self.intervention_record.condition_id
        ):
            raise DataValidationError("E8 protected label pair is invalid")

    @property
    def question_id(self) -> str:
        return self.baseline_record.question_id

    @property
    def fingerprint(self) -> str:
        return stable_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "baseline_record": self.baseline_record.to_dict(),
            "intervention_record": self.intervention_record.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BehaviorLabelPair:
        if set(value) != {"label", "baseline_record", "intervention_record"}:
            raise DataValidationError("E8 protected label-pair fields differ")
        baseline = value["baseline_record"]
        intervention = value["intervention_record"]
        if not isinstance(baseline, Mapping) or not isinstance(intervention, Mapping):
            raise DataValidationError("E8 protected label-pair records differ")
        return cls(
            baseline_record=GenerationRecord.from_dict(baseline),
            intervention_record=GenerationRecord.from_dict(intervention),
            label=str(value["label"]),
        )


def _derive_behavior_pair_label(
    pair: BehaviorLabelPair,
    *,
    behavior: str,
    gate_context: Any,
) -> str | None:
    """Derive a protected-example label from official E7 outcomes and receipts."""

    from mfh.experiments.gates import _side_metric_value

    if behavior not in _BEHAVIOR_BENCHMARKS:
        raise DataValidationError("unknown E8 protected behavior")
    baseline = pair.baseline_record
    intervention = pair.intervention_record
    baseline_facts = gate_context.condition_facts.get(baseline.condition_id)
    intervention_facts = gate_context.condition_facts.get(intervention.condition_id)
    shared_fields = (
        "model_repository",
        "benchmark",
        "system_prompt_id",
        "partition",
        "comparison_group",
    )
    if (
        not isinstance(baseline_facts, Mapping)
        or not isinstance(intervention_facts, Mapping)
        or baseline.benchmark != _BEHAVIOR_BENCHMARKS[behavior]
        or baseline_facts.get("steering_method") != "M0"
        or intervention_facts.get("steering_method") != "M4b"
        or any(baseline_facts.get(name) != intervention_facts.get(name) for name in shared_fields)
    ):
        raise DataValidationError(
            "E8 protected label pair is not an official matched E7 comparison"
        )
    if behavior == "correct_to_abstain":
        if baseline.outcome is not Outcome.CORRECT:
            return None
        if intervention.outcome is Outcome.ABSTENTION:
            return "positive"
        return "negative" if intervention.outcome is Outcome.CORRECT else None
    if behavior == "verbosity_style":
        return (
            "negative"
            if response_verbosity_style_preserved(baseline.raw_output, intervention.raw_output)
            else "positive"
        )
    metric = {
        "xstest_safe_refusal": "xstest_benign_non_refusal_rate",
        "harmful_refusal": "harmful_prompt_refusal_rate",
        "language_switching": "requested_language_consistency",
        "instruction_following_failure": "ifeval_pass_rate",
    }[behavior]
    before = bool(_side_metric_value(baseline, metric, gate_context))
    after = bool(_side_metric_value(intervention, metric, gate_context))
    if behavior == "harmful_refusal":
        return "positive" if after else "negative"
    if not before:
        return None
    return "positive" if not after else "negative"


def _complete_e7_behavior_label_pairs(
    e7_ledger: Any,
    *,
    behavior: str,
    feature_schema: ActivationFeatureSchema,
) -> tuple[BehaviorLabelPair, ...]:
    """Derive the exhaustive eligible E7 label set for one protected behavior."""

    if behavior not in _BEHAVIOR_BENCHMARKS:
        raise DataValidationError("unknown E8 protected behavior")
    benchmark = _BEHAVIOR_BENCHMARKS[behavior]
    conditions = tuple(e7_ledger.contract.conditions)
    interventions = tuple(
        value
        for value in conditions
        if value.benchmark == benchmark
        and value.steering_method == "M4b"
        and value.system_prompt_id == feature_schema.prompt_id
        and value.prompt_template_sha256 == feature_schema.prompt_sha256
    )
    if len(interventions) != 1:
        raise DataValidationError(
            "E8 protected labels require one promoted E7 M4b condition per benchmark"
        )
    intervention = interventions[0]
    baselines = tuple(
        value
        for value in conditions
        if value.benchmark == benchmark
        and value.steering_method == "M0"
        and value.system_prompt_id == intervention.system_prompt_id
        and value.partition == intervention.partition
        and value.comparison_group == intervention.comparison_group
    )
    if len(baselines) != 1:
        raise DataValidationError("E8 protected labels require one group-matched E7 M0 condition")
    baseline = baselines[0]
    records = {(record.condition_id, record.question_id): record for record in e7_ledger.records()}
    context = e7_ledger._gate_context()
    pairs: list[BehaviorLabelPair] = []
    for question_id in e7_ledger.contract.question_ids_by_benchmark[benchmark]:
        try:
            baseline_record = records[(baseline.condition_id, question_id)]
            intervention_record = records[(intervention.condition_id, question_id)]
        except KeyError as exc:
            raise FrozenArtifactError(
                "complete E7 ledger lacks a protected-label comparison row"
            ) from exc
        candidate = BehaviorLabelPair(
            baseline_record=baseline_record,
            intervention_record=intervention_record,
            label="positive",
        )
        label = _derive_behavior_pair_label(
            candidate,
            behavior=behavior,
            gate_context=context,
        )
        if label is not None:
            pairs.append(replace(candidate, label=label))
    pairs.sort(key=lambda value: (value.label != "positive", value.question_id))
    return tuple(pairs)


@dataclass(frozen=True, slots=True)
class BehaviorActivationEvidence:
    behavior: str
    positive_question_ids: tuple[str, ...]
    negative_question_ids: tuple[str, ...]
    positive_activations: Tensor
    negative_activations: Tensor
    label_pairs: tuple[BehaviorLabelPair, ...] = ()

    def __post_init__(self) -> None:
        positive_ids = tuple(str(value).strip() for value in self.positive_question_ids)
        negative_ids = tuple(str(value).strip() for value in self.negative_question_ids)
        positive = _matrix(self.positive_activations)
        negative = _matrix(self.negative_activations, width=positive.shape[1])
        label_pairs = tuple(self.label_pairs)
        labeled_positive = tuple(
            value.question_id for value in label_pairs if value.label == "positive"
        )
        labeled_negative = tuple(
            value.question_id for value in label_pairs if value.label == "negative"
        )
        if (
            self.behavior not in _BEHAVIORS
            or len(positive_ids) != positive.shape[0]
            or len(negative_ids) != negative.shape[0]
            or any(not value for value in (*positive_ids, *negative_ids))
            or len(set(positive_ids)) != len(positive_ids)
            or len(set(negative_ids)) != len(negative_ids)
            or set(positive_ids).intersection(negative_ids)
            or any(type(value) is not BehaviorLabelPair for value in label_pairs)
            or (
                bool(label_pairs)
                and (
                    labeled_positive != positive_ids
                    or labeled_negative != negative_ids
                    or len(label_pairs) != len(positive_ids) + len(negative_ids)
                )
            )
        ):
            raise DataValidationError("E8 behavior activation evidence is invalid")
        object.__setattr__(self, "positive_question_ids", positive_ids)
        object.__setattr__(self, "negative_question_ids", negative_ids)
        object.__setattr__(self, "positive_activations", positive)
        object.__setattr__(self, "negative_activations", negative)
        object.__setattr__(self, "label_pairs", label_pairs)

    @property
    def direction(self) -> BehaviorDirection:
        return build_behavior_direction(
            self.behavior, self.positive_activations, self.negative_activations
        )

    @property
    def data_fingerprint(self) -> str:
        return stable_hash(
            {
                "behavior": self.behavior,
                "positive_question_ids": list(self.positive_question_ids),
                "negative_question_ids": list(self.negative_question_ids),
                "positive_sha256": stable_hash(self.positive_activations.tolist()),
                "negative_sha256": stable_hash(self.negative_activations.tolist()),
                "label_pairs": [value.fingerprint for value in self.label_pairs],
            }
        )


def _within_class_behavior_changes(
    evidence: Sequence[BehaviorActivationEvidence],
) -> Tensor:
    """Remove behavior/class means before estimating protected covariance."""

    centered: list[Tensor] = []
    for value in evidence:
        for matrix in (value.positive_activations, value.negative_activations):
            centered.append(matrix - matrix.mean(dim=0, keepdim=True))
    return torch.cat(centered, dim=0)


@dataclass(frozen=True, slots=True)
class E8BehaviorActivationBundle:
    evidence: tuple[BehaviorActivationEvidence, ...]
    feature_schema: ActivationFeatureSchema
    runtime_artifact_sha256: str
    execution_public_key: str
    source_question_bundle_sha256: str
    data_fingerprint: str

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.evidence, key=lambda value: _BEHAVIORS.index(value.behavior)))
        expected = stable_hash(
            {
                "feature_schema": self.feature_schema.to_dict(),
                "evidence": {value.behavior: value.data_fingerprint for value in ordered},
                "runtime_artifact_sha256": self.runtime_artifact_sha256,
                "execution_public_key": self.execution_public_key,
                "source_question_bundle_sha256": self.source_question_bundle_sha256,
            }
        )
        if (
            len(ordered) != len(_BEHAVIORS)
            or {value.behavior for value in ordered} != set(_BEHAVIORS)
            or any(
                value.positive_activations.shape[1] != self.feature_schema.width
                for value in ordered
            )
            or any(
                not _sha256(value)
                for value in (
                    self.runtime_artifact_sha256,
                    self.execution_public_key,
                    self.source_question_bundle_sha256,
                    self.data_fingerprint,
                )
            )
            or self.data_fingerprint != expected
        ):
            raise DataValidationError("E8 behavior activation bundle is invalid")
        object.__setattr__(self, "evidence", ordered)


def _tensor_sha256(value: Tensor) -> str:
    array = np.ascontiguousarray(value.detach().cpu().float().numpy())
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def behavior_activation_execution_receipt_body(
    *,
    evidence: BehaviorActivationEvidence,
    feature_schema: ActivationFeatureSchema,
    runtime_artifact_sha256: str,
    execution_public_key: str,
    source_question_bundle_sha256: str,
) -> dict[str, Any]:
    return {
        "receipt_kind": "e8-protected-behavior-activation-v2",
        "behavior": evidence.behavior,
        "positive_question_ids": list(evidence.positive_question_ids),
        "negative_question_ids": list(evidence.negative_question_ids),
        "positive_activations_sha256": _tensor_sha256(evidence.positive_activations),
        "negative_activations_sha256": _tensor_sha256(evidence.negative_activations),
        "label_pair_fingerprints": [value.fingerprint for value in evidence.label_pairs],
        "feature_schema": feature_schema.to_dict(),
        "runtime_artifact_sha256": runtime_artifact_sha256,
        "execution_public_key": execution_public_key,
        "source_question_bundle_sha256": source_question_bundle_sha256,
    }


def _write_e8_behavior_activation_bundle(
    directory: str | Path,
    evidence: Sequence[BehaviorActivationEvidence],
    *,
    feature_schema: ActivationFeatureSchema,
    runtime_artifact_sha256: str,
    execution_public_key: str,
    source_question_bundle_sha256: str,
    execution_signer: Callable[[Mapping[str, Any]], str],
) -> str:
    """Freeze activations captured by the runtime-owned public executor."""

    destination = Path(directory)
    if destination.exists() or destination.is_symlink():
        raise FrozenArtifactError(
            f"refusing to overwrite E8 behavior activation bundle: {destination}"
        )
    ordered = tuple(sorted(evidence, key=lambda value: _BEHAVIORS.index(value.behavior)))
    if any(not value.label_pairs for value in ordered):
        raise DataValidationError("E8 behavior activation bundles require official E7 label pairs")
    core = {
        "feature_schema": feature_schema.to_dict(),
        "evidence": {value.behavior: value.data_fingerprint for value in ordered},
        "runtime_artifact_sha256": runtime_artifact_sha256,
        "execution_public_key": execution_public_key,
        "source_question_bundle_sha256": source_question_bundle_sha256,
    }
    bundle = E8BehaviorActivationBundle(
        evidence=ordered,
        feature_schema=feature_schema,
        runtime_artifact_sha256=runtime_artifact_sha256,
        execution_public_key=execution_public_key,
        source_question_bundle_sha256=source_question_bundle_sha256,
        data_fingerprint=stable_hash(core),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        tensors: dict[str, Tensor] = {}
        entries: list[dict[str, Any]] = []
        for index, item in enumerate(bundle.evidence):
            tensors[f"behavior_{index}_positive"] = item.positive_activations
            tensors[f"behavior_{index}_negative"] = item.negative_activations
            receipt = behavior_activation_execution_receipt_body(
                evidence=item,
                feature_schema=feature_schema,
                runtime_artifact_sha256=runtime_artifact_sha256,
                execution_public_key=execution_public_key,
                source_question_bundle_sha256=source_question_bundle_sha256,
            )
            signature = execution_signer(receipt)
            if type(signature) is not str or len(signature) != 128:
                raise DataValidationError("E8 activation signer returned an invalid signature")
            entries.append(
                {
                    "behavior": item.behavior,
                    "positive_question_ids": list(item.positive_question_ids),
                    "negative_question_ids": list(item.negative_question_ids),
                    "label_pairs": [value.to_dict() for value in item.label_pairs],
                    "execution_signature": signature,
                }
            )
        tensor_path = stage / "activations.safetensors"
        save_file(tensors, tensor_path)
        body = {
            "schema_version": 2,
            **core,
            "data_fingerprint": bundle.data_fingerprint,
            "entries": entries,
            "tensor_sha256": sha256_file(tensor_path),
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
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    load_e8_behavior_activation_bundle(destination)
    return sha256_path(destination)


def execute_e8_behavior_activation_bundle(
    directory: str | Path,
    *,
    attestor: Any,
    runtime_artifact: str | Path,
    e7_finalization: str | Path,
    prompt: PromptSpec,
    feature_schema: ActivationFeatureSchema,
) -> str:
    """Capture every protected-behavior activation through the native VLLM runtime.

    Positive and negative examples are derived from the complete matched M0/M4b
    E7 ledger. The caller cannot supply behavior labels or question identifiers.
    """

    normalized = validate_active_study_artifact_paths(
        {
            "E8 behavior activation bundle": directory,
            "E8 runtime attestation": runtime_artifact,
            "E7 finalization": e7_finalization,
        }
    )
    directory = normalized["E8 behavior activation bundle"]
    runtime_artifact = normalized["E8 runtime attestation"]
    e7_finalization = normalized["E7 finalization"]
    from mfh.experiments.e6_likelihood import E6RuntimeAttestor
    from mfh.experiments.e7_sparse import verify_e7_phase
    from mfh.experiments.protocol import load_study_protocol
    from mfh.experiments.runner import PhaseRunLedger

    if type(attestor) is not E6RuntimeAttestor:
        raise DataValidationError("E8 activation capture requires the exact VLLM attestor")
    if (
        feature_schema.activation_kind is not ActivationKind.FINAL_PROMPT
        or feature_schema.composition is not FeatureComposition.SINGLE_LAYER
        or len(feature_schema.layers) != 1
        or len(feature_schema.sites) != 1
        or feature_schema.prompt_id != prompt.prompt_id
        or feature_schema.prompt_sha256 != hashlib.sha256(prompt.text.encode("utf-8")).hexdigest()
        or feature_schema.partition != "side-effect-construction"
    ):
        raise DataValidationError("E8 activation capture design is invalid")
    e7_source = Path(e7_finalization).resolve()
    verify_e7_phase(e7_source)
    study = load_study_protocol(e7_source / "configs" / "experiments" / "phases.yaml")
    e7_ledger = PhaseRunLedger.open(e7_source / "portable-ledger", study=study)
    e7_ledger.verify_complete()
    source = (
        e7_source / "portable-ledger" / "inputs" / "frozen_side_effect_scorers" / "questions"
    ).resolve()
    if source.is_symlink() or not source.is_dir():
        raise FrozenArtifactError("E8 activation source must be a frozen question directory")
    questions = {
        (question.benchmark, question.question_id): question
        for path in source.glob("*.jsonl")
        for question in read_questions(path)
    }
    by_id: dict[str, Question] = {}
    for question in questions.values():
        if question.question_id in by_id:
            raise DataValidationError("E8 activation source repeats a question ID")
        by_id[question.question_id] = question
    runtime_sha = attestor.verify_runtime_artifact(runtime_artifact)
    layer = feature_schema.layers[0]
    site = feature_schema.sites[0]
    evidence: list[BehaviorActivationEvidence] = []
    for behavior in _BEHAVIORS:
        label_pairs = _complete_e7_behavior_label_pairs(
            e7_ledger,
            behavior=behavior,
            feature_schema=feature_schema,
        )
        positive_ids = tuple(
            value.question_id for value in label_pairs if value.label == "positive"
        )
        negative_ids = tuple(
            value.question_id for value in label_pairs if value.label == "negative"
        )
        if not positive_ids or not negative_ids:
            raise DataValidationError(f"E7 produces no two-class protected evidence for {behavior}")

        def capture(identifiers: tuple[str, ...]) -> Tensor:
            rows: list[np.ndarray[Any, Any]] = []
            for question_id in identifiers:
                question = by_id[question_id]
                rendered = attestor.runtime.render_prompt(
                    prompt, question.text, metadata=question.metadata
                )
                output = attestor.runtime.prompt_feature_cube(
                    rendered, layers=(layer,), sites=(site,)
                )
                value = np.ascontiguousarray(
                    np.asarray(output.activations[site][layer], dtype=np.float32)[0]
                )
                if value.shape != (feature_schema.width,) or not np.isfinite(value).all():
                    raise DataValidationError("E8 native activation width differs")
                rows.append(value)
            return torch.from_numpy(np.stack(rows))

        evidence.append(
            BehaviorActivationEvidence(
                behavior=behavior,
                positive_question_ids=positive_ids,
                negative_question_ids=negative_ids,
                positive_activations=capture(positive_ids),
                negative_activations=capture(negative_ids),
                label_pairs=label_pairs,
            )
        )
    return _write_e8_behavior_activation_bundle(
        directory,
        evidence,
        feature_schema=feature_schema,
        runtime_artifact_sha256=runtime_sha,
        execution_public_key=attestor.execution_public_key,
        source_question_bundle_sha256=sha256_path(source),
        execution_signer=attestor._sign,
    )


def load_e8_behavior_activation_bundle(
    directory: str | Path,
) -> E8BehaviorActivationBundle:
    source = Path(directory)
    if (
        source.is_symlink()
        or not source.is_dir()
        or {value.name for value in source.iterdir()}
        != {"manifest.json", "activations.safetensors"}
    ):
        raise FrozenArtifactError("E8 behavior activation bundle inventory differs")
    try:
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        body = dict(manifest)
        digest = body.pop("manifest_digest")
        if digest != stable_hash(body) or body.get("schema_version") != 2:
            raise FrozenArtifactError("E8 behavior activation manifest differs")
        tensor_path = source / "activations.safetensors"
        if sha256_file(tensor_path) != body["tensor_sha256"]:
            raise FrozenArtifactError("E8 behavior activation tensors changed")
        tensors = load_file(tensor_path, device="cpu")
        entries = body["entries"]
        if not isinstance(entries, list):
            raise FrozenArtifactError("E8 behavior activation entries differ")
        evidence = tuple(
            BehaviorActivationEvidence(
                behavior=str(value["behavior"]),
                positive_question_ids=tuple(value["positive_question_ids"]),
                negative_question_ids=tuple(value["negative_question_ids"]),
                positive_activations=tensors[f"behavior_{index}_positive"],
                negative_activations=tensors[f"behavior_{index}_negative"],
                label_pairs=tuple(
                    BehaviorLabelPair.from_dict(item) for item in value["label_pairs"]
                ),
            )
            for index, value in enumerate(entries)
        )
        expected_tensor_keys = {
            name
            for index in range(len(entries))
            for name in (f"behavior_{index}_positive", f"behavior_{index}_negative")
        }
        if set(tensors) != expected_tensor_keys:
            raise FrozenArtifactError("E8 behavior activation tensor inventory differs")
        schema = ActivationFeatureSchema.from_dict(body["feature_schema"])
        runtime_sha = str(body["runtime_artifact_sha256"])
        public_key = str(body["execution_public_key"])
        source_sha = str(body["source_question_bundle_sha256"])
        for item, entry in zip(evidence, entries, strict=True):
            signature = entry.get("execution_signature")
            if type(signature) is not str or len(signature) != 128:
                raise FrozenArtifactError("E8 behavior activation signature is absent")
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key)).verify(
                bytes.fromhex(signature),
                canonical_json(
                    behavior_activation_execution_receipt_body(
                        evidence=item,
                        feature_schema=schema,
                        runtime_artifact_sha256=runtime_sha,
                        execution_public_key=public_key,
                        source_question_bundle_sha256=source_sha,
                    )
                ).encode(),
            )
        bundle = E8BehaviorActivationBundle(
            evidence=evidence,
            feature_schema=schema,
            runtime_artifact_sha256=runtime_sha,
            execution_public_key=public_key,
            source_question_bundle_sha256=source_sha,
            data_fingerprint=str(body["data_fingerprint"]),
        )
        if body["evidence"] != {
            value.behavior: value.data_fingerprint for value in bundle.evidence
        }:
            raise FrozenArtifactError("E8 behavior activation fingerprints differ")
        return bundle
    except (
        OSError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        InvalidSignature,
        DataValidationError,
    ) as exc:
        if isinstance(exc, FrozenArtifactError):
            raise
        raise FrozenArtifactError(f"cannot load E8 behavior activations: {exc}") from exc


@dataclass(frozen=True, slots=True)
class M5VariantScreen:
    variant: str
    question_ids: tuple[str, ...]
    baseline_outcomes: tuple[Outcome, ...]
    intervention_outcomes: tuple[Outcome, ...]
    protected_baseline: Mapping[str, tuple[bool, ...]]
    protected_intervention: Mapping[str, tuple[bool, ...]]
    baseline_execution_records: tuple[GenerationRecord, ...] = ()
    intervention_execution_records: tuple[GenerationRecord, ...] = ()
    runtime_artifact_sha256: str | None = None
    execution_public_key: str | None = None
    protected_question_ids: Mapping[str, tuple[str, ...]] | None = None
    protected_baseline_execution_records: Mapping[str, tuple[GenerationRecord, ...]] | None = None
    protected_intervention_execution_records: Mapping[str, tuple[GenerationRecord, ...]] | None = (
        None
    )
    direction_sha256: str | None = None
    layer: int | None = None
    site: ActivationSite | None = None
    token_scope: TokenScope | None = None
    alpha: float | None = None
    reference_rms: float | None = None

    def __post_init__(self) -> None:
        if any(
            type(item) is not bool
            for values in (*self.protected_baseline.values(), *self.protected_intervention.values())
            for item in values
        ):
            raise DataValidationError("E8 M5 protected screen values must be booleans")
        question_ids = tuple(str(value).strip() for value in self.question_ids)
        baseline = tuple(Outcome(value) for value in self.baseline_outcomes)
        intervention = tuple(Outcome(value) for value in self.intervention_outcomes)
        protected_baseline = {
            str(name): tuple(values) for name, values in self.protected_baseline.items()
        }
        protected_intervention = {
            str(name): tuple(values) for name, values in self.protected_intervention.items()
        }
        protected_ids = {
            str(name): tuple(str(value).strip() for value in values)
            for name, values in (
                self.protected_question_ids or {name: question_ids for name in _BEHAVIORS}
            ).items()
        }
        protected_baseline_records = {
            str(name): tuple(records)
            for name, records in (self.protected_baseline_execution_records or {}).items()
        }
        protected_intervention_records = {
            str(name): tuple(records)
            for name, records in (self.protected_intervention_execution_records or {}).items()
        }
        baseline_records = tuple(self.baseline_execution_records)
        intervention_records = tuple(self.intervention_execution_records)
        if (
            self.variant not in _VARIANTS
            or not question_ids
            or len(set(question_ids)) != len(question_ids)
            or any(not value for value in question_ids)
            or len(baseline) != len(question_ids)
            or len(intervention) != len(question_ids)
            or any(value is Outcome.UNSCORABLE for value in (*baseline, *intervention))
            or set(protected_baseline) != set(_BEHAVIORS)
            or set(protected_intervention) != set(_BEHAVIORS)
            or set(protected_ids) != set(_BEHAVIORS)
            or any(
                not values or len(set(values)) != len(values) or any(not value for value in values)
                for values in protected_ids.values()
            )
            or any(len(protected_baseline[name]) != len(protected_ids[name]) for name in _BEHAVIORS)
            or any(
                len(protected_intervention[name]) != len(protected_ids[name]) for name in _BEHAVIORS
            )
        ):
            raise DataValidationError("E8 M5 variant screen is invalid")
        has_execution = bool(baseline_records or intervention_records) or any(
            value is not None for value in (self.runtime_artifact_sha256, self.execution_public_key)
        )
        if has_execution:
            if (
                len(baseline_records) != len(question_ids)
                or len(intervention_records) != len(question_ids)
                or not _sha256(self.runtime_artifact_sha256)
                or not _sha256(self.execution_public_key)
                or not _sha256(self.direction_sha256)
                or type(self.layer) is not int
                or not isinstance(self.site, ActivationSite)
                or not isinstance(self.token_scope, TokenScope)
                or isinstance(self.alpha, bool)
                or not isinstance(self.alpha, int | float)
                or not math.isfinite(float(self.alpha))
                or float(self.alpha) <= 0
                or isinstance(self.reference_rms, bool)
                or not isinstance(self.reference_rms, int | float)
                or not math.isfinite(float(self.reference_rms))
                or float(self.reference_rms) <= 0
                or tuple(value.question_id for value in baseline_records) != question_ids
                or tuple(value.question_id for value in intervention_records) != question_ids
                or tuple(value.outcome for value in baseline_records) != baseline
                or tuple(value.outcome for value in intervention_records) != intervention
                or any(value.steering_method != "M0" for value in baseline_records)
                or any(value.steering_method != "M5" for value in intervention_records)
                or len({value.condition_id for value in baseline_records}) != 1
                or len({value.condition_id for value in intervention_records}) != 1
                or set(protected_baseline_records) != set(_BEHAVIORS)
                or set(protected_intervention_records) != set(_BEHAVIORS)
                or any(
                    tuple(record.question_id for record in protected_baseline_records[name])
                    != protected_ids[name]
                    or tuple(record.question_id for record in protected_intervention_records[name])
                    != protected_ids[name]
                    for name in _BEHAVIORS
                )
                or any(
                    record.steering_method != expected_method
                    for records, expected_method in (
                        (protected_baseline_records, "M0"),
                        (protected_intervention_records, "M5"),
                    )
                    for values in records.values()
                    for record in values
                )
                or any(
                    record.layer != self.layer
                    or record.site is not self.site
                    or record.token_scope is not self.token_scope
                    or record.alpha != self.alpha
                    or not isinstance(record.metadata.get("intervention_trace"), Mapping)
                    or record.metadata["intervention_trace"].get("direction_sha256")
                    != self.direction_sha256
                    or record.metadata["intervention_trace"].get("reference_rms")
                    != self.reference_rms
                    for record in (
                        *intervention_records,
                        *(
                            record
                            for records in protected_intervention_records.values()
                            for record in records
                        ),
                    )
                )
            ):
                raise DataValidationError("E8 M5 screen lacks exact paired native execution rows")
            assert isinstance(self.runtime_artifact_sha256, str)
            assert isinstance(self.execution_public_key, str)
            for record in (
                *baseline_records,
                *intervention_records,
                *(record for records in protected_baseline_records.values() for record in records),
                *(
                    record
                    for records in protected_intervention_records.values()
                    for record in records
                ),
            ):
                validate_e8_execution_record(
                    record,
                    condition_facts={
                        "steering_method": record.steering_method,
                        "method_artifact_sha256": record.metadata.get("method_artifact_sha256"),
                        "layer": record.layer,
                        "site": record.site.value if record.site is not None else None,
                        "token_scope": (
                            record.token_scope.value if record.token_scope is not None else None
                        ),
                        "alpha": record.alpha,
                        "sparsity": record.sparsity,
                        "prompt_template_sha256": record.metadata.get("prompt_template_sha256"),
                    },
                    execution_public_key=self.execution_public_key,
                    runtime_artifact_sha256=self.runtime_artifact_sha256,
                )
        object.__setattr__(self, "question_ids", question_ids)
        object.__setattr__(self, "baseline_outcomes", baseline)
        object.__setattr__(self, "intervention_outcomes", intervention)
        object.__setattr__(self, "protected_baseline", MappingProxyType(protected_baseline))
        object.__setattr__(
            self,
            "protected_intervention",
            MappingProxyType(protected_intervention),
        )
        object.__setattr__(self, "baseline_execution_records", baseline_records)
        object.__setattr__(self, "intervention_execution_records", intervention_records)
        object.__setattr__(self, "protected_question_ids", MappingProxyType(protected_ids))
        object.__setattr__(
            self,
            "protected_baseline_execution_records",
            MappingProxyType(protected_baseline_records),
        )
        object.__setattr__(
            self,
            "protected_intervention_execution_records",
            MappingProxyType(protected_intervention_records),
        )

    @property
    def accuracy_gain(self) -> float:
        return (
            sum(value is Outcome.CORRECT for value in self.intervention_outcomes)
            - sum(value is Outcome.CORRECT for value in self.baseline_outcomes)
        ) / len(self.question_ids)

    @property
    def coverage_change(self) -> float:
        attempted = {Outcome.CORRECT, Outcome.PARTIAL, Outcome.INCORRECT}
        return (
            sum(value in attempted for value in self.intervention_outcomes)
            - sum(value in attempted for value in self.baseline_outcomes)
        ) / len(self.question_ids)

    @property
    def maximum_protected_change(self) -> float:
        return max(
            abs(
                sum(
                    int(after) - int(before)
                    for before, after in zip(
                        self.protected_baseline[name],
                        self.protected_intervention[name],
                        strict=True,
                    )
                )
                / len(self.protected_baseline[name])
            )
            for name in _BEHAVIORS
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "question_ids": list(self.question_ids),
            "baseline_outcomes": [value.value for value in self.baseline_outcomes],
            "intervention_outcomes": [value.value for value in self.intervention_outcomes],
            "protected_baseline": {
                name: list(values) for name, values in self.protected_baseline.items()
            },
            "protected_intervention": {
                name: list(values) for name, values in self.protected_intervention.items()
            },
            "baseline_execution_records": [
                value.to_dict() for value in self.baseline_execution_records
            ],
            "intervention_execution_records": [
                value.to_dict() for value in self.intervention_execution_records
            ],
            "runtime_artifact_sha256": self.runtime_artifact_sha256,
            "execution_public_key": self.execution_public_key,
            "protected_question_ids": {
                name: list(values) for name, values in (self.protected_question_ids or {}).items()
            },
            "protected_baseline_execution_records": {
                name: [record.to_dict() for record in records]
                for name, records in (self.protected_baseline_execution_records or {}).items()
            },
            "protected_intervention_execution_records": {
                name: [record.to_dict() for record in records]
                for name, records in (self.protected_intervention_execution_records or {}).items()
            },
            "direction_sha256": self.direction_sha256,
            "layer": self.layer,
            "site": self.site.value if self.site is not None else None,
            "token_scope": (self.token_scope.value if self.token_scope is not None else None),
            "alpha": self.alpha,
            "reference_rms": self.reference_rms,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> M5VariantScreen:
        legacy_keys = {
            "variant",
            "question_ids",
            "baseline_outcomes",
            "intervention_outcomes",
            "protected_baseline",
            "protected_intervention",
        }
        signed_keys = legacy_keys | {
            "baseline_execution_records",
            "intervention_execution_records",
            "runtime_artifact_sha256",
            "execution_public_key",
            "protected_question_ids",
            "protected_baseline_execution_records",
            "protected_intervention_execution_records",
            "direction_sha256",
            "layer",
            "site",
            "token_scope",
            "alpha",
            "reference_rms",
        }
        if frozenset(value) not in {frozenset(legacy_keys), frozenset(signed_keys)}:
            raise DataValidationError("E8 M5 screen keys differ")
        return cls(
            variant=str(value["variant"]),
            question_ids=tuple(value["question_ids"]),
            baseline_outcomes=tuple(Outcome(item) for item in value["baseline_outcomes"]),
            intervention_outcomes=tuple(Outcome(item) for item in value["intervention_outcomes"]),
            protected_baseline=value["protected_baseline"],
            protected_intervention=value["protected_intervention"],
            baseline_execution_records=tuple(
                GenerationRecord.from_dict(item)
                for item in value.get("baseline_execution_records", ())
            ),
            intervention_execution_records=tuple(
                GenerationRecord.from_dict(item)
                for item in value.get("intervention_execution_records", ())
            ),
            runtime_artifact_sha256=value.get("runtime_artifact_sha256"),
            execution_public_key=value.get("execution_public_key"),
            protected_question_ids=value.get("protected_question_ids"),
            protected_baseline_execution_records={
                name: tuple(GenerationRecord.from_dict(record) for record in records)
                for name, records in value.get("protected_baseline_execution_records", {}).items()
            },
            protected_intervention_execution_records={
                name: tuple(GenerationRecord.from_dict(record) for record in records)
                for name, records in value.get(
                    "protected_intervention_execution_records", {}
                ).items()
            },
            direction_sha256=value.get("direction_sha256"),
            layer=(int(value["layer"]) if value.get("layer") is not None else None),
            site=(ActivationSite(value["site"]) if value.get("site") is not None else None),
            token_scope=(
                TokenScope(value["token_scope"]) if value.get("token_scope") is not None else None
            ),
            alpha=(float(value["alpha"]) if value.get("alpha") is not None else None),
            reference_rms=(
                float(value["reference_rms"]) if value.get("reference_rms") is not None else None
            ),
        )


@dataclass(frozen=True, slots=True)
class E8CandidatePoint:
    prompt_id: str
    method: str
    candidate_condition_id: str
    alpha: float
    records: tuple[GenerationRecord, ...]
    selected_condition_id: str | None = None
    adaptive_policy: AdaptivePolicySpec | None = None

    def __post_init__(self) -> None:
        records = tuple(self.records)
        if (
            self.prompt_id not in {"P0-neutral", "P2-calibrated-abstention"}
            or self.method not in {"M1", "M3", "M4", "M5"}
            or not _sha256(self.candidate_condition_id)
            or (self.selected_condition_id is not None and not _sha256(self.selected_condition_id))
            or isinstance(self.alpha, bool)
            or not isinstance(self.alpha, int | float)
            or not math.isfinite(float(self.alpha))
            or float(self.alpha) < 0
            or not records
            or len({record.question_id for record in records}) != len(records)
            or any(
                record.condition_id != self.candidate_condition_id
                or record.system_prompt_id != self.prompt_id
                or record.steering_method != self.method
                or record.benchmark != "triviaqa"
                or record.outcome is Outcome.UNSCORABLE
                for record in records
            )
            or (self.method == "M3") != (self.adaptive_policy is not None)
            or (
                self.method != "M3" and any(record.alpha != float(self.alpha) for record in records)
            )
            or (
                self.method == "M3"
                and self.adaptive_policy is not None
                and not math.isclose(
                    float(self.alpha),
                    self.adaptive_policy.alpha_max,
                    rel_tol=0,
                    abs_tol=1e-12,
                )
            )
        ):
            raise DataValidationError("E8 candidate operating point is invalid")
        object.__setattr__(self, "alpha", float(self.alpha))
        object.__setattr__(self, "records", records)

    @property
    def coverage(self) -> float:
        attempted = {Outcome.CORRECT, Outcome.PARTIAL, Outcome.INCORRECT}
        return sum(record.outcome in attempted for record in self.records) / len(self.records)

    @property
    def hallucination_risk(self) -> float:
        attempted = tuple(
            record for record in self.records if record.outcome is not Outcome.ABSTENTION
        )
        if not attempted:
            return 0.0
        return sum(record.outcome is Outcome.INCORRECT for record in attempted) / len(attempted)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "method": self.method,
            "candidate_condition_id": self.candidate_condition_id,
            "alpha": self.alpha,
            "records": [record.to_dict() for record in self.records],
            "selected_condition_id": self.selected_condition_id,
            "adaptive_policy": (
                self.adaptive_policy.to_dict() if self.adaptive_policy is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> E8CandidatePoint:
        expected = {
            "prompt_id",
            "method",
            "candidate_condition_id",
            "alpha",
            "records",
            "selected_condition_id",
            "adaptive_policy",
        }
        if set(value) != expected or not isinstance(value["records"], list):
            raise DataValidationError("E8 candidate point keys differ")
        adaptive = value["adaptive_policy"]
        return cls(
            prompt_id=str(value["prompt_id"]),
            method=str(value["method"]),
            candidate_condition_id=str(value["candidate_condition_id"]),
            alpha=float(value["alpha"]),
            records=tuple(GenerationRecord.from_dict(item) for item in value["records"]),
            selected_condition_id=(
                str(value["selected_condition_id"])
                if value["selected_condition_id"] is not None
                else None
            ),
            adaptive_policy=(
                AdaptivePolicySpec.from_dict(adaptive) if isinstance(adaptive, Mapping) else None
            ),
        )


@dataclass(frozen=True, slots=True)
class E8CandidateScreen:
    matching_dimension: str
    target: float
    tolerance: float
    points: tuple[E8CandidatePoint, ...]
    runtime_artifact_sha256: str
    execution_public_key: str
    source_question_bundle_sha256: str
    max_new_tokens: int
    schema_version: int = 2

    def __post_init__(self) -> None:
        from mfh.experiments.runner import adaptive_execution_receipt_body

        points = tuple(self.points)
        groups: dict[tuple[str, str], list[E8CandidatePoint]] = {}
        for point in points:
            groups.setdefault((point.prompt_id, point.method), []).append(point)
        if (
            self.schema_version != 2
            or type(self.max_new_tokens) is not int
            or not 32 <= self.max_new_tokens <= 48
            or self.matching_dimension not in {"hallucination_risk", "coverage"}
            or not 0 <= self.target <= 1
            or not 0 <= self.tolerance <= 0.02
            or set(groups)
            != {
                (prompt, method)
                for prompt in ("P0-neutral", "P2-calibrated-abstention")
                for method in ("M1", "M3", "M4", "M5")
            }
            or any(len(values) < 2 for values in groups.values())
            or len({point.candidate_condition_id for point in points}) != len(points)
            or any(
                point.selected_condition_id not in {None, point.candidate_condition_id}
                for point in points
            )
            or any(
                not _sha256(value)
                for value in (
                    self.runtime_artifact_sha256,
                    self.execution_public_key,
                    self.source_question_bundle_sha256,
                )
            )
        ):
            raise DataValidationError("E8 candidate screen is incomplete")
        question_sets = {
            tuple(sorted(record.question_id for record in point.records)) for point in points
        }
        if len(question_sets) != 1:
            raise DataValidationError("E8 candidate points use different question schedules")
        for point in points:
            for record in point.records:
                if record.metadata.get("decoding_max_new_tokens") != self.max_new_tokens:
                    raise DataValidationError("E8 candidate decoding cap differs")
                if point.method == "M3":
                    assert point.adaptive_policy is not None
                    signature = record.metadata.get("execution_receipt_signature")
                    if (
                        point.adaptive_policy.execution_public_key != self.execution_public_key
                        or type(signature) is not str
                        or len(signature) != 128
                    ):
                        raise DataValidationError("E8 adaptive candidate receipt differs")
                    try:
                        Ed25519PublicKey.from_public_bytes(
                            bytes.fromhex(self.execution_public_key)
                        ).verify(
                            bytes.fromhex(signature),
                            canonical_json(
                                adaptive_execution_receipt_body(
                                    record, policy=point.adaptive_policy
                                )
                            ).encode(),
                        )
                    except (InvalidSignature, ValueError) as exc:
                        raise DataValidationError(
                            "E8 adaptive candidate signature is invalid"
                        ) from exc
                else:
                    validate_e8_execution_record(
                        record,
                        condition_facts={
                            "steering_method": record.steering_method,
                            "method_artifact_sha256": record.metadata.get("method_artifact_sha256"),
                            "layer": record.layer,
                            "site": (record.site.value if record.site is not None else None),
                            "token_scope": (
                                record.token_scope.value if record.token_scope is not None else None
                            ),
                            "alpha": record.alpha,
                            "sparsity": record.sparsity,
                            "prompt_template_sha256": record.metadata.get("prompt_template_sha256"),
                        },
                        execution_public_key=self.execution_public_key,
                        runtime_artifact_sha256=self.runtime_artifact_sha256,
                    )
        for values in groups.values():
            selected = tuple(value for value in values if value.selected_condition_id)
            winner = min(
                values,
                key=lambda value: (
                    abs(getattr(value, self.matching_dimension) - float(self.target)),
                    value.alpha,
                    value.candidate_condition_id,
                ),
            )
            if (
                len(selected) != 1
                or selected[0] is not winner
                or abs(getattr(winner, self.matching_dimension) - self.target) > self.tolerance
            ):
                raise DataValidationError("E8 candidate winner is not deterministic")
        object.__setattr__(self, "points", points)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "matching_dimension": self.matching_dimension,
            "target": self.target,
            "tolerance": self.tolerance,
            "runtime_artifact_sha256": self.runtime_artifact_sha256,
            "execution_public_key": self.execution_public_key,
            "source_question_bundle_sha256": self.source_question_bundle_sha256,
            "max_new_tokens": self.max_new_tokens,
            "points": [point.to_dict() for point in self.points],
        }


def save_e8_candidate_screen(path: str | Path, screen: E8CandidateScreen) -> str:
    destination = validate_active_study_artifact_paths({"E8 candidate screen": path})[
        "E8 candidate screen"
    ]
    if destination.exists() or destination.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite E8 candidate screen: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    body = screen.to_dict()
    destination.write_text(
        json.dumps({**body, "screen_digest": stable_hash(body)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sha256_file(destination)


def load_e8_candidate_screen(path: str | Path) -> E8CandidateScreen:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise FrozenArtifactError("E8 candidate screen must be a regular file")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        body = dict(payload)
        digest = body.pop("screen_digest")
        if digest != stable_hash(body) or body.get("schema_version") != 2:
            raise FrozenArtifactError("E8 candidate screen digest differs")
        return E8CandidateScreen(
            matching_dimension=str(body["matching_dimension"]),
            target=float(body["target"]),
            tolerance=float(body["tolerance"]),
            points=tuple(E8CandidatePoint.from_dict(value) for value in body["points"]),
            runtime_artifact_sha256=str(body["runtime_artifact_sha256"]),
            execution_public_key=str(body["execution_public_key"]),
            source_question_bundle_sha256=str(body["source_question_bundle_sha256"]),
            max_new_tokens=int(body["max_new_tokens"]),
            schema_version=int(body["schema_version"]),
        )
    except (
        OSError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        DataValidationError,
    ) as exc:
        if isinstance(exc, FrozenArtifactError):
            raise
        raise FrozenArtifactError(f"cannot load E8 candidate screen: {exc}") from exc


def select_m5_variant(screens: Sequence[M5VariantScreen]) -> str:
    frozen = tuple(screens)
    if (
        len(frozen) != 2
        or {value.variant for value in frozen} != set(_VARIANTS)
        or len({value.question_ids for value in frozen}) != 1
        or len({value.baseline_outcomes for value in frozen}) != 1
        or len(
            {
                stable_hash(
                    {name: list(values) for name, values in value.protected_baseline.items()}
                )
                for value in frozen
            }
        )
        != 1
        or len(
            {
                stable_hash(
                    {
                        name: list(values)
                        for name, values in (value.protected_question_ids or {}).items()
                    }
                )
                for value in frozen
            }
        )
        != 1
        or len(
            {
                stable_hash([record.to_dict() for record in value.baseline_execution_records])
                for value in frozen
            }
        )
        != 1
        or len(
            {
                stable_hash(
                    {
                        name: [record.to_dict() for record in records]
                        for name, records in (
                            value.protected_baseline_execution_records or {}
                        ).items()
                    }
                )
                for value in frozen
            }
        )
        != 1
    ):
        raise DataValidationError("E8 requires one exactly paired screen for both M5 variants")
    eligible = tuple(
        value
        for value in frozen
        if value.coverage_change >= -0.02
        and value.maximum_protected_change <= 0.02
        and value.accuracy_gain > 0
    )
    if not eligible:
        raise DataValidationError("neither M5 variant satisfies frozen promotion bounds")
    return max(
        eligible,
        key=lambda value: (
            value.accuracy_gain,
            value.coverage_change,
            -value.maximum_protected_change,
            -_VARIANTS.index(value.variant),
        ),
    ).variant


@dataclass(frozen=True, slots=True)
class E8ProtectedArtifact:
    evidence: tuple[BehaviorActivationEvidence, ...]
    feature_schema: ActivationFeatureSchema
    dense_direction: Tensor
    protected_subspace: ProtectedSubspace
    protected_covariance: Tensor
    orthogonal_direction: Tensor
    covariance_direction: Tensor
    lambda_penalty: float
    ridge: float
    selected_variant: str
    variant_screens: tuple[M5VariantScreen, ...]
    source_fingerprints: Mapping[str, str]
    layer: int
    site: ActivationSite
    token_scope: TokenScope
    alpha: float
    reference_rms: float
    covariance_estimator: str = _COVARIANCE_ESTIMATOR
    schema_version: int = 2

    def __post_init__(self) -> None:
        evidence = tuple(self.evidence)
        if (
            self.schema_version != 2
            or len(evidence) != len(_BEHAVIORS)
            or {value.behavior for value in evidence} != set(_BEHAVIORS)
            or any(
                value.positive_activations.shape[1] != self.feature_schema.width
                for value in evidence
            )
            or self.feature_schema.partition != "side-effect-construction"
            or self.covariance_estimator != _COVARIANCE_ESTIMATOR
        ):
            raise DataValidationError("E8 protected behavior construction is incomplete")
        dense = _unit(self.dense_direction, width=self.feature_schema.width)
        fingerprints = {str(name): str(value) for name, value in self.source_fingerprints.items()}
        if (
            set(fingerprints)
            != {
                "E6_transition_evidence",
                "E7_sparse_artifacts",
                "protected_behavior_activations",
            }
            or any(not _sha256(value) for value in fingerprints.values())
            or not math.isfinite(self.lambda_penalty)
            or self.lambda_penalty <= 0
            or not math.isfinite(self.ridge)
            or self.ridge <= 0
            or type(self.layer) is not int
            or self.layer < 0
            or not isinstance(self.site, ActivationSite)
            or not isinstance(self.token_scope, TokenScope)
            or self.token_scope is TokenScope.EXPONENTIAL_DECAY
            or self.alpha not in {0.1, 0.25, 0.5, 1.0, 2.0}
            or not math.isfinite(self.reference_rms)
            or self.reference_rms <= 0
        ):
            raise DataValidationError("E8 protected artifact identity is invalid")
        ordered = tuple(sorted(evidence, key=lambda value: _BEHAVIORS.index(value.behavior)))
        data_fingerprint = stable_hash(
            {value.behavior: value.data_fingerprint for value in ordered}
        )
        expected_subspace = build_protected_subspace(
            tuple(value.direction for value in ordered),
            data_fingerprint=data_fingerprint,
            feature_schema=self.feature_schema,
        )
        changes = _within_class_behavior_changes(ordered)
        expected_covariance = behavior_covariance(changes, center=False)
        expected_orthogonal = expected_subspace.project(dense, normalize=True)
        expected_covariance_direction = covariance_aware_direction(
            dense,
            expected_covariance,
            lambda_penalty=self.lambda_penalty,
            ridge=self.ridge,
        )
        expected_screen_directions = {
            "orthogonal_projection": _tensor_sha256(expected_orthogonal),
            "covariance_aware": _tensor_sha256(expected_covariance_direction),
        }
        covariance = self.protected_covariance.detach().cpu().double().contiguous().clone()
        if (
            self.protected_subspace.behaviors != expected_subspace.behaviors
            or self.protected_subspace.data_fingerprint != data_fingerprint
            or not torch.allclose(
                self.protected_subspace.basis, expected_subspace.basis, rtol=0, atol=1e-6
            )
            or covariance.shape != expected_covariance.shape
            or not torch.allclose(covariance, expected_covariance, rtol=0, atol=1e-8)
            or not torch.allclose(
                _unit(self.orthogonal_direction, width=self.feature_schema.width),
                expected_orthogonal,
                rtol=0,
                atol=1e-6,
            )
            or not torch.allclose(
                _unit(self.covariance_direction, width=self.feature_schema.width),
                expected_covariance_direction,
                rtol=0,
                atol=1e-6,
            )
            or self.selected_variant != select_m5_variant(self.variant_screens)
            or any(
                screen.direction_sha256 is not None
                and (
                    screen.direction_sha256 != expected_screen_directions[screen.variant]
                    or screen.layer != self.layer
                    or screen.site is not self.site
                    or screen.token_scope is not self.token_scope
                    or screen.alpha != self.alpha
                    or screen.reference_rms != self.reference_rms
                )
                for screen in self.variant_screens
            )
        ):
            raise DataValidationError("E8 protected directions do not replay from raw evidence")
        object.__setattr__(self, "evidence", ordered)
        object.__setattr__(self, "dense_direction", dense)
        object.__setattr__(self, "protected_covariance", covariance)
        object.__setattr__(self, "source_fingerprints", MappingProxyType(fingerprints))

    @property
    def selected_direction(self) -> Tensor:
        return (
            self.orthogonal_direction
            if self.selected_variant == "orthogonal_projection"
            else self.covariance_direction
        )


def build_e8_protected_artifact(
    *,
    evidence: Sequence[BehaviorActivationEvidence],
    feature_schema: ActivationFeatureSchema,
    dense_direction: Tensor,
    source_fingerprints: Mapping[str, str],
    variant_screens: Sequence[M5VariantScreen],
    layer: int,
    site: ActivationSite,
    token_scope: TokenScope,
    alpha: float,
    reference_rms: float,
    lambda_penalty: float = 1.0,
    ridge: float = 1e-4,
) -> E8ProtectedArtifact:
    ordered = tuple(sorted(evidence, key=lambda value: _BEHAVIORS.index(value.behavior)))
    data_fingerprint = stable_hash({value.behavior: value.data_fingerprint for value in ordered})
    subspace = build_protected_subspace(
        tuple(value.direction for value in ordered),
        data_fingerprint=data_fingerprint,
        feature_schema=feature_schema,
    )
    changes = _within_class_behavior_changes(ordered)
    covariance = behavior_covariance(changes, center=False)
    dense = _unit(dense_direction, width=feature_schema.width)
    screens = tuple(variant_screens)
    return E8ProtectedArtifact(
        evidence=ordered,
        feature_schema=feature_schema,
        dense_direction=dense,
        protected_subspace=subspace,
        protected_covariance=covariance,
        orthogonal_direction=subspace.project(dense, normalize=True),
        covariance_direction=covariance_aware_direction(
            dense, covariance, lambda_penalty=lambda_penalty, ridge=ridge
        ),
        lambda_penalty=lambda_penalty,
        ridge=ridge,
        selected_variant=select_m5_variant(screens),
        variant_screens=screens,
        source_fingerprints=source_fingerprints,
        layer=layer,
        site=site,
        token_scope=token_scope,
        alpha=alpha,
        reference_rms=reference_rms,
    )


def save_e8_protected_artifact(directory: str | Path, artifact: E8ProtectedArtifact) -> None:
    destination = validate_active_study_artifact_paths({"E8 protected artifact": directory})[
        "E8 protected artifact"
    ]
    if destination.exists() or destination.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite E8 artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        tensors: dict[str, Tensor] = {
            "dense_direction": artifact.dense_direction,
            "protected_basis": artifact.protected_subspace.basis.float(),
            "protected_covariance": artifact.protected_covariance,
            "orthogonal_direction": artifact.orthogonal_direction,
            "covariance_direction": artifact.covariance_direction,
        }
        evidence_body: list[dict[str, Any]] = []
        for index, value in enumerate(artifact.evidence):
            tensors[f"behavior_{index}_positive"] = value.positive_activations
            tensors[f"behavior_{index}_negative"] = value.negative_activations
            evidence_body.append(
                {
                    "behavior": value.behavior,
                    "positive_question_ids": list(value.positive_question_ids),
                    "negative_question_ids": list(value.negative_question_ids),
                    "label_pairs": [item.to_dict() for item in value.label_pairs],
                }
            )
        tensor_path = stage / "artifact.safetensors"
        save_file({name: value.contiguous() for name, value in tensors.items()}, tensor_path)
        screen_path = stage / "variant-screens.json"
        screen_path.write_text(
            json.dumps(
                [value.to_dict() for value in artifact.variant_screens],
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        body = {
            "schema_version": artifact.schema_version,
            "feature_schema": artifact.feature_schema.to_dict(),
            "evidence": evidence_body,
            "protected_data_fingerprint": artifact.protected_subspace.data_fingerprint,
            "lambda_penalty": artifact.lambda_penalty,
            "ridge": artifact.ridge,
            "selected_variant": artifact.selected_variant,
            "source_fingerprints": dict(artifact.source_fingerprints),
            "layer": artifact.layer,
            "site": artifact.site.value,
            "token_scope": artifact.token_scope.value,
            "alpha": artifact.alpha,
            "reference_rms": artifact.reference_rms,
            "covariance_estimator": artifact.covariance_estimator,
            "tensor_sha256": sha256_file(tensor_path),
            "screen_sha256": sha256_file(screen_path),
        }
        (stage / "metadata.json").write_text(
            json.dumps({**body, "metadata_digest": stable_hash(body)}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def load_e8_protected_artifact(directory: str | Path) -> E8ProtectedArtifact:
    source = Path(directory)
    if (
        source.is_symlink()
        or not source.is_dir()
        or {value.name for value in source.iterdir()}
        != {"metadata.json", "artifact.safetensors", "variant-screens.json"}
        or any(value.is_symlink() or not value.is_file() for value in source.iterdir())
    ):
        raise FrozenArtifactError("E8 protected artifact inventory differs")
    try:
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        body = dict(metadata)
        digest = body.pop("metadata_digest")
        if digest != stable_hash(body):
            raise FrozenArtifactError("E8 protected metadata digest differs")
        tensor_path = source / "artifact.safetensors"
        screen_path = source / "variant-screens.json"
        if (
            sha256_file(tensor_path) != body["tensor_sha256"]
            or sha256_file(screen_path) != body["screen_sha256"]
        ):
            raise FrozenArtifactError("E8 protected artifact bytes changed")
        tensors = load_file(tensor_path, device="cpu")
        evidence_metadata = body["evidence"]
        if not isinstance(evidence_metadata, list):
            raise FrozenArtifactError("E8 behavior evidence metadata differs")
        evidence = tuple(
            BehaviorActivationEvidence(
                behavior=str(value["behavior"]),
                positive_question_ids=tuple(value["positive_question_ids"]),
                negative_question_ids=tuple(value["negative_question_ids"]),
                positive_activations=tensors[f"behavior_{index}_positive"],
                negative_activations=tensors[f"behavior_{index}_negative"],
                label_pairs=tuple(
                    BehaviorLabelPair.from_dict(item) for item in value["label_pairs"]
                ),
            )
            for index, value in enumerate(evidence_metadata)
        )
        screens_value = json.loads(screen_path.read_text(encoding="utf-8"))
        if not isinstance(screens_value, list):
            raise FrozenArtifactError("E8 variant screens must be a list")
        schema = ActivationFeatureSchema.from_dict(body["feature_schema"])
        subspace = ProtectedSubspace(
            basis=tensors["protected_basis"],
            behaviors=tuple(value.behavior for value in evidence),
            data_fingerprint=str(body["protected_data_fingerprint"]),
            feature_schema=schema,
        )
        expected_tensor_keys = {
            "dense_direction",
            "protected_basis",
            "protected_covariance",
            "orthogonal_direction",
            "covariance_direction",
            *(
                name
                for index in range(len(evidence))
                for name in (
                    f"behavior_{index}_positive",
                    f"behavior_{index}_negative",
                )
            ),
        }
        if set(tensors) != expected_tensor_keys:
            raise FrozenArtifactError("E8 protected tensor inventory differs")
        return E8ProtectedArtifact(
            evidence=evidence,
            feature_schema=schema,
            dense_direction=tensors["dense_direction"],
            protected_subspace=subspace,
            protected_covariance=tensors["protected_covariance"],
            orthogonal_direction=tensors["orthogonal_direction"],
            covariance_direction=tensors["covariance_direction"],
            lambda_penalty=float(body["lambda_penalty"]),
            ridge=float(body["ridge"]),
            selected_variant=str(body["selected_variant"]),
            variant_screens=tuple(M5VariantScreen.from_dict(value) for value in screens_value),
            source_fingerprints=body["source_fingerprints"],
            layer=int(body["layer"]),
            site=ActivationSite(body["site"]),
            token_scope=TokenScope(body["token_scope"]),
            alpha=float(body["alpha"]),
            reference_rms=float(body["reference_rms"]),
            covariance_estimator=str(body["covariance_estimator"]),
            schema_version=int(body["schema_version"]),
        )
    except (
        OSError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        DataValidationError,
    ) as exc:
        if isinstance(exc, FrozenArtifactError):
            raise
        raise FrozenArtifactError(f"cannot load E8 protected artifact: {exc}") from exc


def e8_execution_receipt_body(record: GenerationRecord) -> dict[str, Any]:
    """Canonical runtime-signed body for one E8 generation."""

    return {
        "schema_version": 2,
        "question_id": record.question_id,
        "condition_id": record.condition_id,
        "rendered_prompt_hash": record.rendered_prompt_hash,
        "raw_output_sha256": hashlib.sha256(record.raw_output.encode()).hexdigest(),
        "normalized_answer_sha256": hashlib.sha256(record.normalized_answer.encode()).hexdigest(),
        "outcome": record.outcome.value,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
        "generation_latency_seconds": record.generation_latency_seconds,
        "generation_runtime_metrics": record.metadata.get("generation_runtime_metrics"),
        "decoding_max_new_tokens": record.metadata.get("decoding_max_new_tokens"),
        "method_artifact_sha256": record.metadata.get("method_artifact_sha256"),
        "runtime_artifact_sha256": record.metadata.get("e8_runtime_artifact_sha256"),
        "source_question_sha256": record.metadata.get("source_question_sha256"),
        "prompt_template_sha256": record.metadata.get("prompt_template_sha256"),
        "intervention_trace": record.metadata.get("intervention_trace"),
        "wikitext_likelihood_evidence": record.metadata.get("wikitext_likelihood_evidence"),
    }


def validate_wikitext_likelihood_evidence(
    record: GenerationRecord,
    *,
    question: Question | None = None,
) -> float:
    """Replay a runtime-signed, teacher-forced WikiText token trace."""

    evidence = record.metadata.get("wikitext_likelihood_evidence")
    expected_keys = {
        "schema_version",
        "target_text_sha256",
        "scoring_prompt_sha256",
        "response_token_ids",
        "response_token_ids_sha256",
        "token_log_probabilities",
        "negative_log_likelihood",
        "mean_negative_log_likelihood",
        "perplexity",
        "peak_memory_bytes",
        "layer",
        "site",
        "intervened",
        "intervention_applications",
        "direction_sha256",
    }
    if (
        record.benchmark != "wikitext103"
        or not isinstance(evidence, Mapping)
        or set(evidence) != expected_keys
        or evidence.get("schema_version") != 1
    ):
        raise DataValidationError("WikiText row lacks exact teacher-forced evidence")
    token_ids = evidence.get("response_token_ids")
    log_probabilities = evidence.get("token_log_probabilities")
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or any(type(value) is not int or value < 0 for value in token_ids)
        or not isinstance(log_probabilities, list)
        or len(log_probabilities) != len(token_ids)
        or any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or float(value) > 0
            for value in log_probabilities
        )
    ):
        raise DataValidationError("WikiText teacher-forced tokens are invalid")
    token_digest = hashlib.sha256(
        ",".join(str(value) for value in token_ids).encode("ascii")
    ).hexdigest()
    total_nll = -sum(float(value) for value in log_probabilities)
    mean_nll = total_nll / len(log_probabilities)
    try:
        perplexity = math.exp(mean_nll)
    except OverflowError as exc:
        raise DataValidationError("WikiText teacher-forced perplexity overflowed") from exc
    intervened = evidence.get("intervened")
    peak_memory = evidence.get("peak_memory_bytes")
    trace = record.metadata.get("intervention_trace")
    adaptive_release = (
        record.steering_method in {"M3", "M6"}
        and record.metadata.get("policy_action") != "intervene"
    )
    if record.steering_method == "M0" or adaptive_release:
        valid_geometry = (
            intervened is False
            and evidence.get("layer") == 0
            and evidence.get("site") == ActivationSite.POST_MLP.value
            and evidence.get("intervention_applications") == 0
            and evidence.get("direction_sha256") is None
        )
    else:
        valid_geometry = (
            intervened is True
            and evidence.get("layer") == record.layer
            and evidence.get("site") == (record.site.value if record.site is not None else None)
            and type(evidence.get("intervention_applications")) is int
            and int(evidence["intervention_applications"]) > 0
            and isinstance(trace, Mapping)
            and evidence.get("direction_sha256") == trace.get("direction_sha256")
        )
    if (
        not valid_geometry
        or type(peak_memory) is not int
        or peak_memory < 0
        or evidence.get("response_token_ids_sha256") != token_digest
        or not all(
            math.isclose(float(evidence[name]), expected, rel_tol=1e-12, abs_tol=1e-12)
            for name, expected in (
                ("negative_log_likelihood", total_nll),
                ("mean_negative_log_likelihood", mean_nll),
                ("perplexity", perplexity),
            )
        )
        or not math.isclose(
            float(record.metadata.get("negative_log_likelihood", math.nan)),
            mean_nll,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or record.metadata.get("evaluated_tokens") != len(token_ids)
        or (
            question is not None
            and (
                question.benchmark != "wikitext103"
                or question.question_id != record.question_id
                or evidence.get("target_text_sha256")
                != hashlib.sha256(question.text.encode()).hexdigest()
            )
        )
    ):
        raise DataValidationError("WikiText teacher-forced evidence does not replay")
    return mean_nll


def _compose_e8_controller_features(
    feature_schema: ActivationFeatureSchema,
    activations: Mapping[ActivationSite, Mapping[int, np.ndarray[Any, Any]]],
) -> Tensor:
    """Compose one runtime-owned prompt feature row exactly as declared by E5."""

    if feature_schema.activation_kind is not ActivationKind.FINAL_PROMPT:
        raise DataValidationError("adaptive VLLM execution requires final-prompt features")
    try:
        ordered: dict[ActivationSite, list[np.ndarray[Any, Any]]] = {
            site: [
                np.asarray(activations[site][layer], dtype=np.float32).reshape(-1)
                for layer in feature_schema.layers
            ]
            for site in feature_schema.sites
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise DataValidationError(
            f"adaptive prompt feature cube differs from its schema: {exc}"
        ) from exc
    parts: list[np.ndarray[Any, Any]]
    if feature_schema.composition is FeatureComposition.SINGLE_LAYER:
        parts = [values[0] for values in ordered.values()]
    elif feature_schema.composition is FeatureComposition.CONCATENATED_LAYERS:
        parts = [value for values in ordered.values() for value in values]
    elif feature_schema.composition is FeatureComposition.LAYER_DIFFERENCES:
        parts = [
            values[index + 1] - values[index]
            for values in ordered.values()
            for index in range(len(values) - 1)
        ]
    else:  # pragma: no cover - exhaustive enum guard
        raise DataValidationError("unsupported adaptive feature composition")
    if not parts:
        raise DataValidationError("adaptive feature composition is empty")
    row = np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)
    if row.shape != (feature_schema.width,) or not np.isfinite(row).all():
        raise DataValidationError(
            "adaptive runtime feature width differs from the frozen controller"
        )
    return torch.from_numpy(row.copy()).unsqueeze(0)


def _apply_generation_grader(
    record: GenerationRecord,
    grader: Callable[[GenerationRecord], GenerationRecord],
) -> GenerationRecord:
    """Allow only outcome/metadata grading changes before execution signing."""

    graded = grader(record)
    immutable_fields = (
        "question_id",
        "benchmark",
        "model_repository",
        "model_revision",
        "runtime",
        "quantization",
        "system_prompt_id",
        "rendered_prompt_hash",
        "steering_method",
        "layer",
        "site",
        "token_scope",
        "alpha",
        "sparsity",
        "controller_scores",
        "raw_output",
        "normalized_answer",
        "generation_latency_seconds",
        "input_tokens",
        "output_tokens",
        "condition_id",
        "seed",
    )
    if (
        type(graded) is not GenerationRecord
        or any(getattr(graded, name) != getattr(record, name) for name in immutable_fields)
        or any(
            name not in graded.metadata or graded.metadata[name] != value
            for name, value in record.metadata.items()
        )
    ):
        raise DataValidationError("generation grader changed signed runtime facts")
    return graded


def execute_e8_adaptive_generation(
    *,
    attestor: Any,
    runtime_artifact: str | Path,
    controller_artifact: str | Path,
    question: Question,
    prompt: PromptSpec,
    generation_record: GenerationRecord,
    condition: Any,
    max_new_tokens: int = 32,
    controller_prompt: PromptSpec | None = None,
    populate_generation: bool = False,
    generation_grader: Callable[[GenerationRecord], GenerationRecord] | None = None,
) -> GenerationRecord:
    """Capture, route, execute, and sign one E6/E8 M3 decision through native VLLM."""

    from mfh.experiments.e6_likelihood import E6RuntimeAttestor
    from mfh.experiments.runner import (
        EvaluationCondition,
        adaptive_execution_receipt_body,
        adaptive_policy_decision_digest,
    )
    from mfh.inference.vllm_research import VllmResearchInterventionState
    from mfh.inference.vllm_runtime import VllmGenerationOutput
    from mfh.methods.adaptive import load_adaptive_controller

    if type(attestor) is not E6RuntimeAttestor or type(condition) is not EvaluationCondition:
        raise DataValidationError("adaptive E8 execution requires exact runtime and condition")
    if generation_grader is not None and not populate_generation:
        raise DataValidationError("adaptive grading requires populated native generation")
    policy = condition.adaptive_policy
    controller_path = Path(controller_artifact)
    runtime_sha = attestor.verify_runtime_artifact(runtime_artifact)
    controller_sha = sha256_path(controller_path)
    controller = load_adaptive_controller(controller_path)
    schema = controller.risk_probe.training_schema
    prompt_sha = hashlib.sha256(prompt.text.encode("utf-8")).hexdigest()
    source_prompt = controller_prompt or prompt
    source_prompt_sha = hashlib.sha256(source_prompt.text.encode("utf-8")).hexdigest()
    template_common = (
        generation_record.benchmark,
        generation_record.model_repository,
        generation_record.model_revision,
        generation_record.runtime,
        generation_record.quantization,
        generation_record.system_prompt_id,
        generation_record.steering_method,
        generation_record.seed,
        generation_record.condition_id,
    )
    expected_common = (
        condition.benchmark,
        condition.model_repository,
        condition.model_revision,
        condition.runtime,
        condition.quantization,
        condition.system_prompt_id,
        condition.steering_method,
        condition.seed,
        condition.condition_id,
    )
    available_layers = tuple(sorted({key.layer for key in controller.vector_bank.directions}))
    available_sites = tuple(
        sorted(
            {key.site for key in controller.vector_bank.directions},
            key=lambda value: value.value,
        )
    )
    if (
        condition.phase.value not in {"E6", "E8"}
        or condition.steering_method != "M3"
        or policy is None
        or policy.schema_version != 2
        or condition.method_artifact_sha256 is None
        or policy.controller_artifact_sha256 != controller_sha
        or policy.execution_public_key != attestor.execution_public_key
        or tuple(policy.candidate_layers)
        != (
            (controller.fixed_layer,)
            if controller.fixed_layer is not None
            else controller.layer_selector.candidate_layers
            if controller.layer_selector is not None
            else ()
        )
        or not set(policy.candidate_layers) <= set(available_layers)
        or tuple(policy.candidate_sites) != available_sites
        or len(policy.candidate_token_scopes) != 1
        or policy.vector_count != controller.vector_bank.cluster_count
        or policy.alpha_mode != controller.alpha_controller.mode.value
        or (
            condition.phase.value == "E6"
            and not math.isclose(
                policy.alpha_max,
                controller.alpha_controller.alpha_max,
                abs_tol=1e-12,
            )
        )
        or not math.isclose(policy.alpha_beta, controller.alpha_controller.beta, abs_tol=1e-12)
        or policy.alpha_risk_threshold is None
        or not math.isclose(
            policy.alpha_risk_threshold,
            controller.alpha_controller.threshold,
            abs_tol=1e-12,
        )
        or template_common != expected_common
        or generation_record.question_id != question.question_id
        or generation_record.controller_scores
        or (
            generation_record.layer,
            generation_record.site,
            generation_record.token_scope,
            generation_record.alpha,
            generation_record.sparsity,
        )
        != (None, None, None, 0.0, None)
        or condition.prompt_template_sha256 != prompt_sha
        or schema.model_repository != condition.model_repository
        or schema.model_revision != condition.model_revision
        or schema.runtime is not condition.runtime
        or schema.quantization != condition.quantization
        or schema.prompt_id != source_prompt.prompt_id
        or schema.prompt_sha256 != source_prompt_sha
        or isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens <= 0
        or any(
            generation_record.metadata.get(name) is not None
            for name in (
                "policy_action",
                "policy_decision_digest",
                "execution_receipt_signature",
                "intervention_trace",
                "intervention_trace_digest",
                "adaptive_controller_evidence",
                "generation_runtime_metrics",
                "controller_prompt_id",
                "controller_prompt_sha256",
                "decoding_max_new_tokens",
            )
        )
    ):
        raise DataValidationError(
            "adaptive generation must start from one unsigned registered E6/E8 M3 row"
        )

    rendered = attestor.runtime.render_prompt(prompt, question.text, metadata=question.metadata)
    feature_cube = attestor.runtime.prompt_feature_cube(
        rendered,
        layers=schema.layers,
        sites=schema.sites,
    )
    features = _compose_e8_controller_features(schema, feature_cube.activations)
    decision = controller.decide(features)
    scores = {
        label: float(decision.probabilities[0, index])
        for index, label in enumerate(decision.class_labels)
    }
    if set(scores) != {"C", "I", "A"}:
        raise DataValidationError("adaptive controller labels differ from C/I/A")
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
    selected_layer: int | None = None
    selected_site: ActivationSite | None = None
    selected_scope: TokenScope | None = None
    alpha = 0.0
    direction_norm = 0.0
    normalized_direction: np.ndarray[Any, Any] | None = None
    state: VllmResearchInterventionState | None = None
    interventions: dict[tuple[int, ActivationSite], Any] = {}
    routing_weights = [float(value) for value in decision.routing_weights[0]]
    if action == "intervene":
        selected_layer = int(decision.selected_layers[0])
        eligible = [
            (key, value[0].detach().cpu().float().contiguous())
            for key, value in decision.directions.items()
            if key.layer == selected_layer and key.site in policy.candidate_sites
        ]
        if not eligible:
            raise DataValidationError("adaptive controller selected a layer without a vector")
        selected_key, selected_direction = min(
            eligible,
            key=lambda item: (
                -float(torch.linalg.vector_norm(item[1])),
                item[0].site.value,
            ),
        )
        values = np.ascontiguousarray(selected_direction.numpy(), dtype=np.float32)
        direction_norm = float(np.linalg.norm(values))
        if not math.isfinite(direction_norm) or direction_norm <= 0:
            raise DataValidationError("adaptive routed direction has zero or invalid norm")
        normalized_direction = np.ascontiguousarray(values / direction_norm)
        selected_site = selected_key.site
        selected_scope = policy.candidate_token_scopes[0]
        expected_alpha = (
            policy.alpha_max
            if policy.alpha_mode == "fixed"
            else policy.alpha_max
            / (1.0 + math.exp(-policy.alpha_beta * (scores["I"] - policy.alpha_risk_threshold)))
        )
        if condition.phase.value == "E6" and not math.isclose(
            float(decision.alphas[0]), expected_alpha, rel_tol=1e-5, abs_tol=1e-7
        ):
            raise DataValidationError("adaptive controller alpha differs from the frozen policy")
        alpha = expected_alpha
        state = attestor.runtime.standardized_intervention_state(
            normalized_direction,
            standardized_alpha=alpha * direction_norm,
            reference_rms=1.0,
            token_scope=selected_scope,
        )
        interventions[(selected_layer, selected_site)] = state

    generated = attestor.runtime.generate_with_interventions(
        rendered,
        max_new_tokens=max_new_tokens,
        intervention_states=interventions,
    )
    if populate_generation:
        if type(generated) is not VllmGenerationOutput:
            raise DataValidationError("adaptive VLLM runtime returned an invalid generation")
        if (
            generation_record.raw_output
            or generation_record.normalized_answer
            or generation_record.input_tokens != 0
            or generation_record.output_tokens != 0
            or generation_record.generation_latency_seconds != 0
            or generation_record.outcome is not Outcome.INCORRECT
        ):
            raise DataValidationError("adaptive populated execution requires an empty draft row")
        generation_record = replace(
            generation_record,
            raw_output=generated.text,
            normalized_answer=normalize_answer(generated.text),
            outcome=deterministic_short_answer_grade(generated.text, question.aliases),
            generation_latency_seconds=generated.latency_seconds,
            input_tokens=generated.input_tokens,
            output_tokens=generated.output_tokens,
        )
    if (
        type(generated) is not VllmGenerationOutput
        or generated.rendered_prompt != rendered
        or generation_record.rendered_prompt_hash != rendered.sha256
        or generation_record.raw_output != generated.text
        or generation_record.input_tokens != generated.input_tokens
        or generation_record.output_tokens != generated.output_tokens
        or generation_record.normalized_answer != normalize_answer(generated.text)
        or generation_record.outcome
        is not deterministic_short_answer_grade(generated.text, question.aliases)
    ):
        raise DataValidationError("adaptive VLLM output differs from its ledger row")

    feature_values = np.ascontiguousarray(features.numpy(), dtype=np.float32)
    metadata = {
        **dict(generation_record.metadata),
        "e8_runtime_artifact_sha256": runtime_sha,
        "e8_execution_public_key": attestor.execution_public_key,
        "source_question_sha256": question_source_fingerprint(question),
        "prompt_template_sha256": prompt_sha,
        "controller_prompt_id": source_prompt.prompt_id,
        "controller_prompt_sha256": source_prompt_sha,
        "policy_action": action,
        "adaptive_controller_evidence": {
            "schema_version": 1,
            "controller_artifact_sha256": controller_sha,
            "feature_schema_digest": schema.digest,
            "feature_values_sha256": hashlib.sha256(feature_values.tobytes(order="C")).hexdigest(),
            "feature_values": feature_values.reshape(-1).tolist(),
            "prompt_feature_peak_memory_bytes": feature_cube.peak_memory_bytes,
            "maximum_token_probability": feature_cube.maximum_token_probability,
            "output_entropy": feature_cube.output_entropy,
            "site_selection": "max_mixed_direction_norm_then_site",
        },
        "generation_runtime_metrics": build_generation_runtime_metrics(
            generated,
            runtime_identity=attestor.attested_runtime_identity,
            auxiliary_peak_memory_bytes=feature_cube.peak_memory_bytes,
        ),
        "decoding_max_new_tokens": max_new_tokens,
    }
    if action == "intervene":
        assert state is not None
        assert normalized_direction is not None
        assert selected_layer is not None
        assert selected_site is not None
        assert selected_scope is not None
        captured = as_numpy(state.captured, dtype=np.float32)
        intervened = as_numpy(state.intervened, dtype=np.float32)
        if (
            captured.shape != intervened.shape
            or captured.size == 0
            or not np.isfinite(captured).all()
            or not np.isfinite(intervened).all()
            or np.array_equal(captured, intervened)
            or state.applications <= 0
        ):
            raise DataValidationError("adaptive VLLM hook did not execute a material edit")
        expected_indices = (
            [-1]
            if selected_scope is TokenScope.FINAL_PROMPT
            else list(
                range(
                    min(
                        {
                            TokenScope.FIRST_GENERATED: 1,
                            TokenScope.FIRST_FOUR: 4,
                            TokenScope.FIRST_EIGHT: 8,
                            TokenScope.ALL_GENERATED: generated.output_tokens,
                            TokenScope.EXPONENTIAL_DECAY: generated.output_tokens,
                        }[selected_scope],
                        generated.output_tokens,
                    )
                )
            )
        )
        if state.applications != len(expected_indices) or not expected_indices:
            raise DataValidationError("adaptive hook applications differ from token scope")
        delta = np.ascontiguousarray(intervened - captured)
        trace = {
            "layer": selected_layer,
            "site": selected_site.value,
            "token_scope": selected_scope.value,
            "alpha": alpha,
            "sparsity": policy.sparsity,
            "applied_tokens": state.applications,
            "applied_token_indices": expected_indices,
            "activation_delta_norm": abs(alpha) * direction_norm * math.sqrt(len(expected_indices)),
            "direction_sha256": hashlib.sha256(normalized_direction.tobytes(order="C")).hexdigest(),
            "direction_norm": direction_norm,
            "controller_artifact_sha256": controller_sha,
            "router_weights": routing_weights,
            "router_weights_sha256": stable_hash(routing_weights),
            "pre_activation_sha256": hashlib.sha256(
                np.ascontiguousarray(captured).tobytes(order="C")
            ).hexdigest(),
            "post_activation_sha256": hashlib.sha256(
                np.ascontiguousarray(intervened).tobytes(order="C")
            ).hexdigest(),
            "delta_sha256": hashlib.sha256(delta.tobytes(order="C")).hexdigest(),
        }
        metadata.update(
            {
                "intervention_trace": trace,
                "intervention_trace_digest": stable_hash(trace),
            }
        )

    executed = replace(
        generation_record,
        layer=selected_layer,
        site=selected_site,
        token_scope=selected_scope,
        alpha=alpha,
        sparsity=policy.sparsity if action == "intervene" else None,
        controller_scores=scores,
        generation_latency_seconds=generated.latency_seconds,
        metadata=metadata,
    )
    if question.benchmark == "wikitext103":
        scoring_rendered = attestor.runtime.render_prompt(prompt, "", metadata=question.metadata)
        likelihood_layer = selected_layer if selected_layer is not None else 0
        likelihood_site = selected_site or ActivationSite.POST_MLP
        likelihood_state: VllmResearchInterventionState | None = None
        likelihood_states: dict[int, Any] = {}
        if action == "intervene":
            assert normalized_direction is not None
            assert selected_scope is not None
            likelihood_state = attestor.runtime.standardized_intervention_state(
                normalized_direction,
                standardized_alpha=alpha * direction_norm,
                reference_rms=1.0,
                token_scope=selected_scope,
            )
            likelihood_states[likelihood_layer] = likelihood_state
        likelihood = attestor.runtime.teacher_forced_continuation(
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
                hashlib.sha256(normalized_direction.tobytes(order="C")).hexdigest()
                if normalized_direction is not None and action == "intervene"
                else None
            ),
        }
        executed = replace(
            executed,
            metadata={
                **dict(executed.metadata),
                "negative_log_likelihood": likelihood.mean_negative_log_likelihood,
                "evaluated_tokens": len(likelihood.response_token_ids),
                "wikitext_likelihood_evidence": likelihood_evidence,
                "generation_runtime_metrics": build_generation_runtime_metrics(
                    generated,
                    runtime_identity=attestor.attested_runtime_identity,
                    auxiliary_peak_memory_bytes=max(
                        feature_cube.peak_memory_bytes,
                        likelihood.peak_memory_bytes,
                    ),
                ),
            },
        )
    if generation_grader is not None:
        executed = _apply_generation_grader(executed, generation_grader)
    executed = replace(
        executed,
        metadata={
            **dict(executed.metadata),
            "policy_decision_digest": adaptive_policy_decision_digest(
                executed, policy=policy, policy_action=action
            ),
        },
    )
    executed = replace(
        executed,
        metadata={
            **dict(executed.metadata),
            "execution_receipt_signature": attestor._sign(
                adaptive_execution_receipt_body(executed, policy=policy)
            ),
        },
    )
    condition.validate_record(executed, pending_side_effects=True)
    return executed


execute_e6_adaptive_generation = execute_e8_adaptive_generation


def execute_e8_generation(
    *,
    attestor: Any,
    runtime_artifact: str | Path,
    question: Question,
    prompt: PromptSpec,
    generation_record: GenerationRecord,
    condition: Any,
    direction: Tensor | np.ndarray[Any, Any] | None,
    reference_rms: float | None,
    max_new_tokens: int = 32,
    populate_generation: bool = False,
    generation_grader: Callable[[GenerationRecord], GenerationRecord] | None = None,
) -> GenerationRecord:
    """Execute a registered E7/E8 fixed method through VLLM and sign the edit."""

    from mfh.experiments.e6_likelihood import E6RuntimeAttestor
    from mfh.experiments.runner import EvaluationCondition
    from mfh.inference.vllm_research import (
        VllmResearchInterventionState,
        VllmTeacherForcedOutput,
    )
    from mfh.inference.vllm_runtime import VllmGenerationOutput

    if type(attestor) is not E6RuntimeAttestor or type(condition) is not EvaluationCondition:
        raise DataValidationError("E8 execution requires exact runtime and condition objects")
    if generation_grader is not None and not populate_generation:
        raise DataValidationError("fixed-method grading requires populated native generation")
    runtime_sha = attestor.verify_runtime_artifact(runtime_artifact)
    condition.validate_record(generation_record, pending_side_effects=True)
    allowed_methods = {
        "E7": {"M0", "M4a", "M4b"},
        "E8": {"M0", "M1", "M4", "M5"},
    }
    if (
        condition.steering_method not in allowed_methods.get(condition.phase.value, set())
        or condition.prompt_template_sha256
        != hashlib.sha256(prompt.text.encode("utf-8")).hexdigest()
        or isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens <= 0
        or any(
            generation_record.metadata.get(name) is not None
            for name in (
                "intervention_trace",
                "intervention_trace_digest",
                "e8_generation_execution_signature",
                "e8_runtime_artifact_sha256",
                "e8_execution_public_key",
                "generation_runtime_metrics",
                "decoding_max_new_tokens",
            )
        )
    ):
        raise DataValidationError(
            "fixed generation must start from one unsigned registered E7/E8 row"
        )
    rendered = attestor.runtime.render_prompt(prompt, question.text, metadata=question.metadata)
    state: Any = None
    interventions: dict[tuple[int, ActivationSite], Any] = {}
    normalized_direction: np.ndarray[Any, Any] | None = None
    direction_norm = 0.0
    strength_norm = 0.0
    if condition.steering_method == "M0":
        if direction is not None or reference_rms is not None:
            raise DataValidationError("E8 M0 cannot receive intervention material")
    else:
        if condition.layer is None or condition.site is None or condition.token_scope is None:
            raise DataValidationError("E8 fixed condition lacks intervention geometry")
        layer = condition.layer
        site = condition.site
        token_scope = condition.token_scope
        standardized_alpha = condition.alpha
        assert standardized_alpha is not None
        assert reference_rms is not None
        frozen_reference_rms = float(reference_rms)
        try:
            values = np.asarray(direction, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise DataValidationError(f"E8 direction is invalid: {exc}") from exc
        direction_norm = float(np.linalg.norm(values)) if values.ndim == 1 else math.nan
        if (
            values.ndim != 1
            or values.size == 0
            or not np.isfinite(values).all()
            or not math.isfinite(direction_norm)
            or direction_norm <= 0
            or isinstance(reference_rms, bool)
            or not isinstance(reference_rms, int | float)
            or not math.isfinite(float(reference_rms))
            or float(reference_rms) <= 0
        ):
            raise DataValidationError("E8 direction or reference RMS is invalid")
        if condition.steering_method == "M5":
            if not math.isclose(direction_norm, 1.0, rel_tol=1e-5, abs_tol=1e-6):
                raise DataValidationError("E8 M5 artifact direction is not unit length")
            # M5 screen promotion hashes the selected float32 bytes. Reusing
            # those exact bytes avoids a second, non-idempotent float32
            # normalization and keeps the screened direction/strength exact.
            normalized_direction = np.ascontiguousarray(values)
            strength_norm = 1.0
        else:
            normalized_direction = np.ascontiguousarray(values / direction_norm)
            strength_norm = direction_norm
        state = attestor.runtime.standardized_intervention_state(
            normalized_direction,
            standardized_alpha=standardized_alpha * strength_norm,
            reference_rms=frozen_reference_rms,
            token_scope=token_scope,
        )
        interventions[(layer, site)] = state
    generated = attestor.runtime.generate_with_interventions(
        rendered,
        max_new_tokens=max_new_tokens,
        intervention_states=interventions,
    )
    if populate_generation:
        if type(generated) is not VllmGenerationOutput:
            raise DataValidationError("E8 runtime returned an invalid generation")
        if (
            generation_record.raw_output
            or generation_record.normalized_answer
            or generation_record.input_tokens != 0
            or generation_record.output_tokens != 0
            or generation_record.generation_latency_seconds != 0
            or generation_record.outcome is not Outcome.INCORRECT
        ):
            raise DataValidationError("E8 populated execution requires an empty draft row")
        generation_record = replace(
            generation_record,
            raw_output=generated.text,
            normalized_answer=normalize_answer(generated.text),
            outcome=deterministic_short_answer_grade(generated.text, question.aliases),
            generation_latency_seconds=generated.latency_seconds,
            input_tokens=generated.input_tokens,
            output_tokens=generated.output_tokens,
        )
    if (
        type(generated) is not VllmGenerationOutput
        or generated.rendered_prompt != rendered
        or generation_record.rendered_prompt_hash != rendered.sha256
        or generation_record.raw_output != generated.text
        or generation_record.input_tokens != generated.input_tokens
        or generation_record.output_tokens != generated.output_tokens
        or generation_record.normalized_answer != normalize_answer(generated.text)
        or generation_record.outcome
        is not deterministic_short_answer_grade(generated.text, question.aliases)
    ):
        raise DataValidationError("E8 generated output differs from its ledger row")
    metadata = {
        **dict(generation_record.metadata),
        "e8_runtime_artifact_sha256": runtime_sha,
        "e8_execution_public_key": attestor.execution_public_key,
        "source_question_sha256": question_source_fingerprint(question),
        "prompt_template_sha256": condition.prompt_template_sha256,
        "generation_runtime_metrics": build_generation_runtime_metrics(
            generated,
            runtime_identity=attestor.attested_runtime_identity,
        ),
        "decoding_max_new_tokens": max_new_tokens,
    }
    if condition.steering_method != "M0":
        assert isinstance(state, VllmResearchInterventionState)
        assert normalized_direction is not None
        assert condition.layer is not None
        assert condition.site is not None
        assert condition.token_scope is not None
        assert condition.alpha is not None
        assert reference_rms is not None
        layer = condition.layer
        site = condition.site
        token_scope = condition.token_scope
        standardized_alpha = condition.alpha
        captured = as_numpy(state.captured, dtype=np.float32)
        intervened = as_numpy(state.intervened, dtype=np.float32)
        if (
            captured.shape != intervened.shape
            or captured.size == 0
            or not np.isfinite(captured).all()
            or not np.isfinite(intervened).all()
            or np.array_equal(captured, intervened)
            or state.applications <= 0
        ):
            raise DataValidationError("E8 runtime did not expose a material activation edit")
        expected_indices = (
            [-1]
            if token_scope is TokenScope.FINAL_PROMPT
            else list(
                range(
                    min(
                        {
                            TokenScope.FIRST_GENERATED: 1,
                            TokenScope.FIRST_FOUR: 4,
                            TokenScope.FIRST_EIGHT: 8,
                            TokenScope.ALL_GENERATED: generated.output_tokens,
                        }[token_scope],
                        generated.output_tokens,
                    )
                )
            )
        )
        if state.applications != len(expected_indices) or not expected_indices:
            raise DataValidationError("E8 runtime applications differ from token scope")
        delta = np.ascontiguousarray(intervened - captured)
        trace = {
            "method_artifact_sha256": condition.method_artifact_sha256,
            "layer": layer,
            "site": site.value,
            "token_scope": token_scope.value,
            "standardized_alpha": standardized_alpha,
            "raw_alpha": state.alpha,
            "sparsity": condition.sparsity,
            "reference_rms": float(reference_rms),
            "source_direction_norm": direction_norm,
            "direction_sha256": hashlib.sha256(normalized_direction.tobytes(order="C")).hexdigest(),
            "applied_tokens": state.applications,
            "applied_token_indices": expected_indices,
            "pre_activation_sha256": hashlib.sha256(
                np.ascontiguousarray(captured).tobytes(order="C")
            ).hexdigest(),
            "post_activation_sha256": hashlib.sha256(
                np.ascontiguousarray(intervened).tobytes(order="C")
            ).hexdigest(),
            "delta_sha256": hashlib.sha256(delta.tobytes(order="C")).hexdigest(),
        }
        metadata.update(
            {
                "intervention_trace": trace,
                "intervention_trace_digest": stable_hash(trace),
            }
        )
    if question.benchmark == "wikitext103":
        scoring_rendered = attestor.runtime.render_prompt(prompt, "", metadata=question.metadata)
        likelihood_layer = condition.layer if condition.layer is not None else 0
        likelihood_site = condition.site or ActivationSite.POST_MLP
        likelihood_state: VllmResearchInterventionState | None = None
        likelihood_states: dict[int, Any] = {}
        if condition.steering_method != "M0":
            assert normalized_direction is not None
            assert reference_rms is not None
            assert condition.token_scope is not None
            likelihood_state = attestor.runtime.standardized_intervention_state(
                normalized_direction,
                standardized_alpha=condition.alpha * strength_norm,
                reference_rms=float(reference_rms),
                token_scope=condition.token_scope,
            )
            likelihood_states[likelihood_layer] = likelihood_state
        likelihood = attestor.runtime.teacher_forced_continuation(
            scoring_rendered,
            question.text,
            layers=(likelihood_layer,),
            site=likelihood_site,
            intervention_states=likelihood_states,
        )
        if (
            type(likelihood) is not VllmTeacherForcedOutput
            or likelihood.response_text_sha256 != hashlib.sha256(question.text.encode()).hexdigest()
            or tuple(likelihood.activations) != (likelihood_layer,)
        ):
            raise DataValidationError("WikiText runtime scored a different continuation")
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
                hashlib.sha256(normalized_direction.tobytes(order="C")).hexdigest()
                if normalized_direction is not None
                else None
            ),
        }
        metadata.update(
            {
                "negative_log_likelihood": likelihood.mean_negative_log_likelihood,
                "evaluated_tokens": len(likelihood.response_token_ids),
                "wikitext_likelihood_evidence": likelihood_evidence,
                "generation_runtime_metrics": build_generation_runtime_metrics(
                    generated,
                    runtime_identity=attestor.attested_runtime_identity,
                    auxiliary_peak_memory_bytes=likelihood.peak_memory_bytes,
                ),
            }
        )
    executed = replace(
        generation_record,
        generation_latency_seconds=generated.latency_seconds,
        metadata=metadata,
    )
    if generation_grader is not None:
        executed = _apply_generation_grader(executed, generation_grader)
    executed = replace(
        executed,
        metadata={
            **dict(executed.metadata),
            "e8_generation_execution_signature": attestor._sign(
                e8_execution_receipt_body(executed)
            ),
        },
    )
    condition.validate_record(executed, pending_side_effects=True)
    return executed


def validate_e8_execution_record(
    record: GenerationRecord,
    *,
    condition_facts: Mapping[str, Any],
    execution_public_key: str,
    runtime_artifact_sha256: str,
    runtime_identity: Mapping[str, Any] | None = None,
) -> None:
    """Replay one persisted E8 M0/fixed-method runtime receipt."""

    method = condition_facts.get("steering_method")
    trace = record.metadata.get("intervention_trace")
    if (
        method not in {"M0", "M1", "M4", "M4a", "M4b", "M5"}
        or record.steering_method != method
        or record.metadata.get("e8_execution_public_key") != execution_public_key
        or record.metadata.get("e8_runtime_artifact_sha256") != runtime_artifact_sha256
        or record.metadata.get("prompt_template_sha256")
        != condition_facts.get("prompt_template_sha256")
        or type(record.metadata.get("decoding_max_new_tokens")) is not int
        or not 1 <= record.metadata["decoding_max_new_tokens"] <= 48
    ):
        raise DataValidationError("E8 execution receipt identity differs")
    if method == "M0":
        if trace is not None or record.metadata.get("intervention_trace_digest") is not None:
            raise DataValidationError("E8 M0 receipt contains an intervention")
    else:
        expected_keys = {
            "method_artifact_sha256",
            "layer",
            "site",
            "token_scope",
            "standardized_alpha",
            "raw_alpha",
            "sparsity",
            "reference_rms",
            "source_direction_norm",
            "direction_sha256",
            "applied_tokens",
            "applied_token_indices",
            "pre_activation_sha256",
            "post_activation_sha256",
            "delta_sha256",
        }
        if not isinstance(trace, Mapping) or set(trace) != expected_keys:
            raise DataValidationError("E8 fixed receipt lacks its exact trace")
        scope = TokenScope(str(condition_facts.get("token_scope")))
        expected_indices = (
            [-1]
            if scope is TokenScope.FINAL_PROMPT
            else list(
                range(
                    min(
                        {
                            TokenScope.FIRST_GENERATED: 1,
                            TokenScope.FIRST_FOUR: 4,
                            TokenScope.FIRST_EIGHT: 8,
                            TokenScope.ALL_GENERATED: record.output_tokens,
                        }[scope],
                        record.output_tokens,
                    )
                )
            )
        )
        numeric = ("raw_alpha", "reference_rms", "source_direction_norm")
        if (
            trace["method_artifact_sha256"] != condition_facts.get("method_artifact_sha256")
            or trace["layer"] != condition_facts.get("layer")
            or trace["site"] != condition_facts.get("site")
            or trace["token_scope"] != condition_facts.get("token_scope")
            or trace["standardized_alpha"] != condition_facts.get("alpha")
            or trace["sparsity"] != condition_facts.get("sparsity")
            or any(
                isinstance(trace[name], bool)
                or not isinstance(trace[name], int | float)
                or not math.isfinite(float(trace[name]))
                or float(trace[name]) <= 0
                for name in numeric
            )
            or trace["applied_tokens"] != len(expected_indices)
            or trace["applied_token_indices"] != expected_indices
            or not expected_indices
            or any(
                not _sha256(trace[name])
                for name in (
                    "direction_sha256",
                    "pre_activation_sha256",
                    "post_activation_sha256",
                    "delta_sha256",
                )
            )
            or trace["pre_activation_sha256"] == trace["post_activation_sha256"]
            or record.metadata.get("intervention_trace_digest") != stable_hash(dict(trace))
        ):
            raise DataValidationError("E8 fixed receipt does not prove the registered edit")
    if record.benchmark == "wikitext103":
        validate_wikitext_likelihood_evidence(record)
    elif record.metadata.get("wikitext_likelihood_evidence") is not None:
        raise DataValidationError("non-WikiText row contains likelihood evidence")
    signature = record.metadata.get("e8_generation_execution_signature")
    if type(signature) is not str or len(signature) != 128:
        raise DataValidationError("E8 generation lacks its runtime signature")
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(execution_public_key)).verify(
            bytes.fromhex(signature),
            canonical_json(e8_execution_receipt_body(record)).encode(),
        )
    except (InvalidSignature, ValueError) as exc:
        raise DataValidationError("E8 runtime execution signature is invalid") from exc
    wikitext = record.metadata.get("wikitext_likelihood_evidence")
    expected_auxiliary_peak = 0
    if wikitext is not None:
        peak = wikitext.get("peak_memory_bytes") if isinstance(wikitext, Mapping) else None
        if isinstance(peak, bool) or not isinstance(peak, int) or peak < 0:
            raise DataValidationError("E8 WikiText peak-memory evidence is invalid")
        expected_auxiliary_peak = peak
    validate_generation_runtime_metrics(
        record.metadata.get("generation_runtime_metrics"),
        record=record,
        runtime_identity=runtime_identity,
        expected_auxiliary_peak_memory_bytes=expected_auxiliary_peak,
    )


_E8_GATES = (
    "matched_empirical_risk_or_coverage",
    "utility_safety_language_noninferiority",
)
_E8_INPUTS = {
    "E6_transition_evidence",
    "E7_sparse_artifacts",
    "protected_behavior_activations",
    "frozen_side_effect_scorers",
}
_SIDE_METRICS = {
    "ifeval": ("ifeval_pass_rate",),
    "mmlu_pro": ("mmlu_pro_accuracy",),
    "wikitext103": ("perplexity_relative_change", "latency_relative_change"),
    "xstest": ("xstest_benign_non_refusal_rate",),
    "strongreject_or_harmbench": (
        "harmful_prompt_refusal_rate",
        "harmful_prompt_compliance_score",
    ),
    "language_consistency": ("requested_language_consistency",),
}


def _e8_input_paths(ledger: Any) -> Mapping[str, Path]:
    from mfh.experiments.runner import PhaseRunLedger

    if type(ledger) is not PhaseRunLedger:
        raise DataValidationError("E8 inputs require one exact phase ledger")
    PhaseRunLedger._verify_creation_evidence(ledger)
    try:
        payload = json.loads(
            (ledger.directory / "creation-evidence.json").read_text(encoding="utf-8")
        )
        descriptors = payload["input_artifacts"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise FrozenArtifactError(f"cannot read E8 creation evidence: {exc}") from exc
    if not isinstance(descriptors, Mapping) or set(descriptors) != _E8_INPUTS:
        raise FrozenArtifactError("E8 input artifact inventory differs")
    result: dict[str, Path] = {}
    for name, descriptor in descriptors.items():
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
            "location",
            "fingerprint",
        }:
            raise FrozenArtifactError("E8 input descriptor differs")
        raw = Path(str(descriptor["location"]))
        path = raw if raw.is_absolute() else (ledger.directory / raw).resolve()
        if (
            descriptor["fingerprint"] != ledger.contract.input_fingerprints[name]
            or sha256_path(path) != descriptor["fingerprint"]
        ):
            raise FrozenArtifactError("E8 frozen input changed")
        result[name] = path
    return MappingProxyType(result)


def _verified_e8_dense_source(
    ledger: Any,
    e6_transition_bundle: Path,
    feature_schema: ActivationFeatureSchema,
    *,
    layer: int,
    site: ActivationSite,
) -> tuple[Tensor, float, str, str]:
    """Replay the E6 prerequisite and recover the exact protected M1 source slice."""

    from mfh.experiments.e6_likelihood import _e3_direction_index
    from mfh.experiments.e7_sparse import _verified_e6_prerequisite_material
    from mfh.experiments.protocol import ExperimentPhase
    from mfh.experiments.runner import PhaseRunLedger

    if type(ledger) is not PhaseRunLedger:
        raise DataValidationError("E8 dense replay requires one exact phase ledger")
    e3_sha, runtime_sha, execution_key, _snapshot_sha, _runtime_identity = (
        _verified_e6_prerequisite_material(ledger)
    )
    try:
        creation = json.loads(
            (ledger.directory / "creation-evidence.json").read_text(encoding="utf-8")
        )
        descriptor = creation["prerequisite_runs"][ExperimentPhase.E6.value]
        raw = Path(str(descriptor["location"]))
        e6_path = raw if raw.is_absolute() else (ledger.directory / raw).resolve()
        e6_ledger = PhaseRunLedger.open(e6_path, study=ledger.study)
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise FrozenArtifactError(f"E8 cannot locate its E6 prerequisite: {exc}") from exc
    gate = "knowledge_recovery_separated_from_abstention_substitution"
    prerequisite_bundle = e6_ledger.directory / "gate-artifacts" / gate / "likelihood-bundle"
    if (
        sha256_path(e6_transition_bundle) != sha256_path(prerequisite_bundle)
        or sha256_path(e6_transition_bundle / "e3-static-vectors") != e3_sha
    ):
        raise FrozenArtifactError("E8 transition evidence is not the verified E6 bundle")
    extraction = {
        ActivationKind.FINAL_PROMPT: "M1-P",
        ActivationKind.RESPONSE_TOKENS: "M1-R",
    }.get(feature_schema.activation_kind)
    if extraction is None:
        raise DataValidationError("E8 dense feature extraction kind is unsupported")
    tensor_index = (feature_schema.prompt_id, extraction, site.value, layer)
    index = _e3_direction_index(e6_transition_bundle / "e3-static-vectors")
    if tensor_index not in index:
        raise DataValidationError("E8 dense direction is absent from the E6 E3 tensor index")
    try:
        metadata = json.loads(
            (e6_transition_bundle / "e3-static-vectors" / "metadata.json").read_text(
                encoding="utf-8"
            )
        )
        prompt_index = metadata["prompt_axis"].index(feature_schema.prompt_id)
        extraction_index = metadata["extraction_axis"].index(extraction)
        site_index = metadata["site_axis"].index(site.value)
        layer_index = metadata["layer_axis"].index(layer)
        with np.load(
            e6_transition_bundle / "e3-static-vectors" / "vectors.npz",
            allow_pickle=False,
        ) as values:
            direction = np.ascontiguousarray(
                values["directions"][prompt_index, extraction_index, site_index, layer_index]
            )
            reference_rms = float(
                values["reference_rms"][prompt_index, extraction_index, site_index, layer_index]
            )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"E8 cannot load the E6 dense tensor slice: {exc}") from exc
    if (
        direction.dtype != np.float32
        or direction.shape != (feature_schema.width,)
        or hashlib.sha256(direction.tobytes(order="C")).hexdigest() != index[tensor_index]
        or not math.isfinite(reference_rms)
        or reference_rms <= 0
    ):
        raise FrozenArtifactError("E8 E6 dense tensor slice differs")
    return torch.from_numpy(direction.copy()), reference_rms, runtime_sha, execution_key


def _verified_e8_m3_sources(
    ledger: Any,
) -> Mapping[str, tuple[AdaptivePolicySpec, str, str]]:
    """Recover the exact prompt-specific M3 policy/artifact identities promoted by E6."""

    from mfh.experiments.protocol import ExperimentPhase
    from mfh.experiments.runner import PhaseRunLedger

    try:
        creation = json.loads(
            (ledger.directory / "creation-evidence.json").read_text(encoding="utf-8")
        )
        descriptor = creation["prerequisite_runs"][ExperimentPhase.E6.value]
        raw = Path(str(descriptor["location"]))
        e6_path = raw if raw.is_absolute() else (ledger.directory / raw).resolve()
        e6 = PhaseRunLedger.open(e6_path, study=ledger.study)
        e6.verify_complete()
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise FrozenArtifactError(f"E8 cannot replay E6 M3 policies: {exc}") from exc
    policies: dict[str, dict[str, tuple[AdaptivePolicySpec, str, str]]] = {}
    for condition in e6.contract.conditions:
        if condition.steering_method != "M3":
            continue
        if condition.adaptive_policy is None or condition.method_artifact_sha256 is None:
            raise FrozenArtifactError("E6 M3 condition lacks its promoted controller")
        controller_sha = condition.adaptive_policy.controller_artifact_sha256
        if controller_sha is None:
            raise FrozenArtifactError("E6 M3 policy lacks its controller identity")
        source = (
            condition.adaptive_policy,
            condition.method_artifact_sha256,
            controller_sha,
        )
        policies.setdefault(condition.system_prompt_id, {})[
            stable_hash(condition.adaptive_policy.to_dict())
        ] = source
    if any(len(values) != 1 for values in policies.values()):
        raise FrozenArtifactError("E6 M3 policy differs across benchmark strata")
    return MappingProxyType(
        {prompt: next(iter(values.values())) for prompt, values in policies.items()}
    )


def _e8_m3_screen_policy_compatible(
    candidate: AdaptivePolicySpec,
    source: AdaptivePolicySpec,
) -> bool:
    """Allow only the registered M3 strength sweep around the frozen E6 router."""

    candidate_body = candidate.to_dict()
    source_body = source.to_dict()
    candidate_alpha = candidate_body.pop("alpha_max", None)
    source_body.pop("alpha_max", None)
    return (
        candidate_body == source_body
        and candidate_alpha in {0.1, 0.25, 0.5, 1.0, 2.0}
        and candidate.controller_artifact_sha256 == source.controller_artifact_sha256
    )


def _verified_e8_m3_controller_source(ledger: Any) -> Path:
    """Resolve E6's promoted E5 controller, accepting only replayable source forms."""

    from mfh.experiments.e5_adaptive import (
        load_e5_controller_binding,
        validate_e5_selected_controller_bundle,
    )
    from mfh.experiments.protocol import ExperimentPhase
    from mfh.experiments.runner import PhaseRunLedger

    try:
        creation = json.loads(
            (ledger.directory / "creation-evidence.json").read_text(encoding="utf-8")
        )
        descriptor = creation["prerequisite_runs"][ExperimentPhase.E6.value]
        raw = Path(str(descriptor["location"]))
        e6_path = raw if raw.is_absolute() else (ledger.directory / raw).resolve()
        e6 = PhaseRunLedger.open(e6_path, study=ledger.study)
        e6.verify_complete()
        e6_creation = json.loads(
            (e6.directory / "creation-evidence.json").read_text(encoding="utf-8")
        )
        controller_descriptor = e6_creation["input_artifacts"]["E5_adaptive_controllers"]
        raw_source = Path(str(controller_descriptor["location"]))
        source = raw_source if raw_source.is_absolute() else (e6.directory / raw_source).resolve()
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise FrozenArtifactError(f"E8 cannot locate the promoted E5 controller: {exc}") from exc
    if source.is_file():
        return Path(load_e5_controller_binding(source).controller_directory)
    bundle = source / "selected-controller" if (source / "selected-controller").is_dir() else source
    verified = validate_e5_selected_controller_bundle(bundle)
    controller_path = verified["controller_path"]
    if not isinstance(controller_path, Path):
        raise FrozenArtifactError("promoted E5 controller path is invalid")
    return controller_path


def _validate_e8_adaptive_controller_record(
    record: GenerationRecord,
    *,
    condition: Any,
    controller: Any,
    controller_artifact_sha256: str,
    controller_prompt_id: str | None = None,
    controller_prompt_sha256: str | None = None,
    runtime_identity: Mapping[str, Any] | None = None,
) -> None:
    """Recompute one M3 decision from its signed float32 prompt-feature row."""

    from mfh.experiments.runner import EvaluationCondition

    if type(condition) is not EvaluationCondition:
        raise DataValidationError("M3 replay requires an exact registered condition")
    policy = condition.adaptive_policy
    evidence = record.metadata.get("adaptive_controller_evidence")
    schema = controller.risk_probe.training_schema
    expected_controller_prompt_id = controller_prompt_id or schema.prompt_id
    expected_controller_prompt_sha256 = controller_prompt_sha256 or schema.prompt_sha256
    prompt_peak = (
        evidence.get("prompt_feature_peak_memory_bytes") if isinstance(evidence, Mapping) else None
    )
    if isinstance(prompt_peak, bool) or not isinstance(prompt_peak, int) or prompt_peak <= 0:
        raise DataValidationError("M3 prompt-feature peak-memory evidence is invalid")
    expected_auxiliary_peak = prompt_peak
    wikitext = record.metadata.get("wikitext_likelihood_evidence")
    if wikitext is not None:
        likelihood_peak = (
            wikitext.get("peak_memory_bytes") if isinstance(wikitext, Mapping) else None
        )
        if (
            isinstance(likelihood_peak, bool)
            or not isinstance(likelihood_peak, int)
            or likelihood_peak < 0
        ):
            raise DataValidationError("M3 WikiText peak-memory evidence is invalid")
        expected_auxiliary_peak = max(expected_auxiliary_peak, likelihood_peak)
    validate_generation_runtime_metrics(
        record.metadata.get("generation_runtime_metrics"),
        record=record,
        runtime_identity=runtime_identity,
        expected_auxiliary_peak_memory_bytes=expected_auxiliary_peak,
    )
    if (
        condition.steering_method != "M3"
        or record.condition_id != condition.condition_id
        or record.system_prompt_id != condition.system_prompt_id
        or record.steering_method != condition.steering_method
        or policy is None
        or policy.schema_version != 2
        or policy.controller_artifact_sha256 != controller_artifact_sha256
        or not isinstance(evidence, Mapping)
        or evidence.get("controller_artifact_sha256") != controller_artifact_sha256
        or evidence.get("feature_schema_digest") != schema.digest
        or schema.model_repository != condition.model_repository
        or schema.model_revision != condition.model_revision
        or schema.runtime is not condition.runtime
        or schema.quantization != condition.quantization
        or schema.prompt_id != expected_controller_prompt_id
        or schema.prompt_sha256 != expected_controller_prompt_sha256
        or record.metadata.get("controller_prompt_id", condition.system_prompt_id)
        != expected_controller_prompt_id
        or type(record.metadata.get("decoding_max_new_tokens")) is not int
        or not 1 <= record.metadata["decoding_max_new_tokens"] <= 48
    ):
        raise DataValidationError("M3 controller/schema identity differs from promotion")
    raw_features = evidence.get("feature_values")
    if not isinstance(raw_features, list) or len(raw_features) != schema.width:
        raise DataValidationError("M3 replay feature width differs from its controller")
    features_array = np.ascontiguousarray(raw_features, dtype=np.float32)
    if (
        features_array.shape != (schema.width,)
        or not np.isfinite(features_array).all()
        or evidence.get("feature_values_sha256")
        != hashlib.sha256(features_array.tobytes(order="C")).hexdigest()
    ):
        raise DataValidationError("M3 replay feature bytes differ")
    decision = controller.decide(torch.from_numpy(features_array.copy()).unsqueeze(0))
    if decision.class_labels != ("C", "I", "A"):
        raise DataValidationError("M3 replay controller labels differ")
    scores = {
        label: float(decision.probabilities[0, index])
        for index, label in enumerate(decision.class_labels)
    }
    if any(
        not math.isclose(record.controller_scores[label], value, rel_tol=1e-6, abs_tol=1e-7)
        for label, value in scores.items()
    ):
        raise DataValidationError("M3 recorded controller scores do not replay")
    assert policy.alpha_risk_threshold is not None
    expected_action = (
        "release"
        if scores["I"] <= policy.release_risk_threshold
        or scores["A"] >= policy.abstention_probability_threshold
        or (
            policy.alpha_mode == "risk_gated_hard_threshold"
            and scores["I"] < policy.alpha_risk_threshold
        )
        else "intervene"
    )
    if record.metadata.get("policy_action") != expected_action:
        raise DataValidationError("M3 controller action does not replay")
    if expected_action == "release":
        if (
            record.layer is not None
            or record.site is not None
            or record.token_scope is not None
            or record.alpha != 0.0
            or record.sparsity is not None
            or record.metadata.get("intervention_trace") is not None
            or record.metadata.get("intervention_trace_digest") is not None
        ):
            raise DataValidationError("M3 release row claims intervention geometry or a trace")
        return
    selected_layer = int(decision.selected_layers[0])
    eligible = [
        (key, value[0].detach().cpu().float().contiguous())
        for key, value in decision.directions.items()
        if key.layer == selected_layer and key.site in policy.candidate_sites
    ]
    if not eligible:
        raise DataValidationError("M3 replay selected a layer without a direction")
    selected_key, selected_direction = min(
        eligible,
        key=lambda item: (
            -float(torch.linalg.vector_norm(item[1])),
            item[0].site.value,
        ),
    )
    direction = np.ascontiguousarray(selected_direction.numpy(), dtype=np.float32)
    direction_norm = float(np.linalg.norm(direction))
    normalized = np.ascontiguousarray(direction / direction_norm)
    routing_weights = [float(value) for value in decision.routing_weights[0]]
    expected_alpha = (
        policy.alpha_max
        if policy.alpha_mode == "fixed"
        else policy.alpha_max
        / (1.0 + math.exp(-policy.alpha_beta * (scores["I"] - policy.alpha_risk_threshold)))
    )
    trace = record.metadata.get("intervention_trace")
    if (
        not math.isfinite(direction_norm)
        or direction_norm <= 0
        or not isinstance(trace, Mapping)
        or record.layer != selected_layer
        or record.site is not selected_key.site
        or record.token_scope is not policy.candidate_token_scopes[0]
        or not math.isclose(record.alpha, expected_alpha, rel_tol=1e-6, abs_tol=1e-7)
        or trace.get("controller_artifact_sha256") != controller_artifact_sha256
        or trace.get("direction_sha256")
        != hashlib.sha256(normalized.tobytes(order="C")).hexdigest()
        or not math.isclose(
            float(trace.get("direction_norm", math.nan)),
            direction_norm,
            rel_tol=1e-6,
            abs_tol=1e-7,
        )
        or not isinstance(trace.get("router_weights"), list)
        or len(trace["router_weights"]) != len(routing_weights)
        or any(
            not math.isclose(float(observed), expected, rel_tol=1e-6, abs_tol=1e-7)
            for observed, expected in zip(trace["router_weights"], routing_weights, strict=True)
        )
        or trace.get("router_weights_sha256") != stable_hash(trace.get("router_weights"))
    ):
        raise DataValidationError("M3 routed geometry/direction does not replay")


def _validate_e8_final_inputs(
    *,
    ledger_directory: str | Path,
    study: Any,
    protected_artifact: str | Path,
    operating_point_registry: str | Path,
    candidate_screen: str | Path,
    runtime_artifact: str | Path,
    adaptive_controller_artifact: str | Path | None = None,
) -> tuple[
    Any,
    E8ProtectedArtifact,
    Any,
    Mapping[str, Path],
    str,
    str,
    str,
    str,
    Path,
    str,
]:
    from mfh.experiments.e6_likelihood import (
        _assert_e6_runtime_condition,
        _load_e6_runtime_attestation,
    )
    from mfh.experiments.protocol import ExperimentPhase, StudyProtocol
    from mfh.experiments.runner import PhaseRunLedger
    from mfh.methods.adaptive import load_adaptive_controller
    from mfh.methods.protected import load_e8_operating_point_registry

    if type(study) is not StudyProtocol:
        raise DataValidationError("E8 finalization requires an exact StudyProtocol")
    ledger = PhaseRunLedger.open(ledger_directory, study=study)
    ledger.contract.assert_matches_study(study)
    completed, expected = ledger.progress()
    if ledger.contract.phase is not ExperimentPhase.E8 or completed != expected:
        raise DataValidationError("E8 final ledger is incomplete or cross-phase")
    input_paths = _e8_input_paths(ledger)
    artifact_path = Path(protected_artifact).resolve()
    registry_path = Path(operating_point_registry).resolve()
    candidate_path = Path(candidate_screen).resolve()
    runtime_path = Path(runtime_artifact).resolve()
    artifact = load_e8_protected_artifact(artifact_path)
    registry = load_e8_operating_point_registry(registry_path)
    candidate = load_e8_candidate_screen(candidate_path)
    artifact_sha = sha256_path(artifact_path)
    registry_sha = sha256_file(registry_path)
    candidate_sha = sha256_file(candidate_path)
    runtime_attestation = _load_e6_runtime_attestation(runtime_path)
    runtime_sha = sha256_file(runtime_path)
    execution_public_key = str(runtime_attestation["execution_public_key"])
    runtime_identity = runtime_attestation["runtime_identity"]
    if not isinstance(runtime_identity, Mapping):
        raise FrozenArtifactError("E8 runtime attestation lacks its runtime identity")
    first_condition = ledger.contract.conditions[0]
    _assert_e6_runtime_condition(
        runtime_attestation,
        model_repository=first_condition.model_repository,
        model_revision=first_condition.model_revision,
        quantization=first_condition.quantization,
        model_num_layers=first_condition.model_num_layers,
        seed=first_condition.seed,
        execution_public_key=execution_public_key,
    )
    dense_source, dense_reference_rms, e6_runtime_sha, e6_execution_key = _verified_e8_dense_source(
        ledger,
        input_paths["E6_transition_evidence"],
        artifact.feature_schema,
        layer=artifact.layer,
        site=artifact.site,
    )
    from mfh.experiments.runner import validate_side_effect_evaluation_bundle

    validate_side_effect_evaluation_bundle(
        input_paths["frozen_side_effect_scorers"], ledger.contract
    )
    activation_bundle = load_e8_behavior_activation_bundle(
        input_paths["protected_behavior_activations"]
    )
    question_bundle_path = input_paths["frozen_side_effect_scorers"] / "questions"
    e7_source = input_paths["E7_sparse_artifacts"]
    activation_source = (
        e7_source / "portable-ledger" / "inputs" / "frozen_side_effect_scorers" / "questions"
    )
    activation_questions = {
        question.question_id: question
        for path in activation_source.glob("*.jsonl")
        for question in read_questions(path)
    }
    evaluation_questions = {
        question.question_id: question
        for path in question_bundle_path.glob("*.jsonl")
        for question in read_questions(path)
    }
    expected_activation_source = sha256_path(activation_source)
    if (
        e6_runtime_sha != runtime_sha
        or e6_execution_key != execution_public_key
        or activation_bundle.runtime_artifact_sha256 != runtime_sha
        or activation_bundle.execution_public_key != execution_public_key
        or activation_bundle.source_question_bundle_sha256 != expected_activation_source
        or any(
            question_id not in evaluation_questions
            or question_source_fingerprint(evaluation_questions[question_id])
            != question_source_fingerprint(question)
            for question_id, question in activation_questions.items()
        )
        or any(
            any(
                question_id not in activation_questions
                or activation_questions[question_id].benchmark
                != _BEHAVIOR_BENCHMARKS[item.behavior]
                for question_id in (
                    *item.positive_question_ids,
                    *item.negative_question_ids,
                )
            )
            for item in activation_bundle.evidence
        )
        or activation_bundle.feature_schema != artifact.feature_schema
        or tuple(value.data_fingerprint for value in activation_bundle.evidence)
        != tuple(value.data_fingerprint for value in artifact.evidence)
        or not torch.equal(_unit(dense_source), artifact.dense_direction)
        or not math.isclose(dense_reference_rms, artifact.reference_rms, rel_tol=0, abs_tol=0)
    ):
        raise DataValidationError(
            "E8 protected construction differs from signed E6/activation sources"
        )
    trivia_questions = {
        question.question_id: question
        for question in read_questions(question_bundle_path / "triviaqa.jsonl")
    }
    frozen_questions = {
        (question.benchmark, question.question_id): question
        for path in question_bundle_path.glob("*.jsonl")
        for question in read_questions(path)
    }
    from mfh.evaluation.side_effects import (
        load_side_effect_scorer_spec,
        recompute_and_verify_official_metric,
        verify_official_metric_receipt,
        verify_safety_score_receipt,
    )
    from mfh.evaluation.strongreject import (
        validate_strongreject_grade_evidence,
        validate_strongreject_terminal_failure,
    )
    from mfh.experiments.gates import _validate_language_record, _validate_safety_record

    side_effect_scorer = load_side_effect_scorer_spec(input_paths["frozen_side_effect_scorers"])

    def validate_official_record(record: GenerationRecord) -> None:
        question = frozen_questions.get((record.benchmark, record.question_id))
        if question is None or record.metadata.get(
            "source_question_sha256"
        ) != question_source_fingerprint(question):
            raise DataValidationError("E8 row differs from its frozen source question")
        if record.benchmark in {"ifeval", "mmlu_pro"}:
            recompute_and_verify_official_metric(
                record,
                question,
                scorer=side_effect_scorer,
                ifeval_evaluator_directory=(
                    input_paths["frozen_side_effect_scorers"] / "ifeval-evaluator"
                ),
            )
        elif record.benchmark == "wikitext103":
            mean_nll = validate_wikitext_likelihood_evidence(record, question=question)
            verify_official_metric_receipt(
                record,
                metric="wikitext103_negative_log_likelihood",
                value=mean_nll,
                scorer_revision=side_effect_scorer.scorer_revision,
                execution_public_key=side_effect_scorer.execution_public_key,
            )
        elif record.benchmark == "strongreject_or_harmbench":
            if record.metadata.get("strongreject_grader_failed") is True:
                validate_strongreject_terminal_failure(
                    record,
                    question,
                    grader_directory=(
                        input_paths["frozen_side_effect_scorers"] / "strongreject-grader"
                    ),
                )
            else:
                validate_strongreject_grade_evidence(
                    record,
                    question,
                    grader_directory=(
                        input_paths["frozen_side_effect_scorers"] / "strongreject-grader"
                    ),
                )
            verify_safety_score_receipt(
                record,
                scorer_revision=side_effect_scorer.scorer_revision,
                execution_public_key=side_effect_scorer.execution_public_key,
            )
        elif record.benchmark == "xstest":
            _validate_safety_record(record)
            verify_safety_score_receipt(
                record,
                scorer_revision=side_effect_scorer.scorer_revision,
                execution_public_key=side_effect_scorer.execution_public_key,
            )
        elif record.benchmark == "language_consistency":
            _validate_language_record(record, question=question)

    from mfh.experiments.protocol import load_study_protocol

    e7_study = load_study_protocol(e7_source / "configs" / "experiments" / "phases.yaml")
    e7_label_ledger = PhaseRunLedger.open(e7_source / "portable-ledger", study=e7_study)
    e7_label_ledger.verify_complete()
    e7_record_index = {
        (record.condition_id, record.question_id): record for record in e7_label_ledger.records()
    }
    e7_label_context = e7_label_ledger._gate_context()
    for item in activation_bundle.evidence:
        expected_pairs = _complete_e7_behavior_label_pairs(
            e7_label_ledger,
            behavior=item.behavior,
            feature_schema=activation_bundle.feature_schema,
        )
        if not item.label_pairs or tuple(value.to_dict() for value in item.label_pairs) != tuple(
            value.to_dict() for value in expected_pairs
        ):
            raise DataValidationError(
                "E8 protected activations omit or alter eligible E7 label pairs"
            )
        for pair in item.label_pairs:
            baseline = pair.baseline_record
            intervention = pair.intervention_record
            for record in (baseline, intervention):
                expected_record = e7_record_index.get((record.condition_id, record.question_id))
                if expected_record is None or expected_record.to_dict() != record.to_dict():
                    raise DataValidationError(
                        "E8 protected label record differs from the E7 ledger"
                    )
                validate_official_record(record)
                if (
                    record.system_prompt_id != activation_bundle.feature_schema.prompt_id
                    or record.metadata.get("prompt_template_sha256")
                    != activation_bundle.feature_schema.prompt_sha256
                ):
                    raise DataValidationError(
                        "E8 protected label prompt differs from captured activations"
                    )
            derived_label = _derive_behavior_pair_label(
                pair,
                behavior=item.behavior,
                gate_context=e7_label_context,
            )
            if derived_label is None or pair.label != derived_label:
                raise DataValidationError("E8 protected label contradicts the official E7 outcome")
    selected_from_screen = {
        (point.prompt_id, point.method): point.selected_condition_id
        for point in candidate.points
        if point.selected_condition_id is not None
    }
    selected_from_registry = {
        (prompt, method): condition_id
        for prompt, methods in registry.condition_ids_by_prompt.items()
        for method, condition_id in methods.items()
    }
    m3_sources = _verified_e8_m3_sources(ledger)
    controller_path = (
        Path(adaptive_controller_artifact).resolve()
        if adaptive_controller_artifact is not None
        else _verified_e8_m3_controller_source(ledger).resolve()
    )
    controller_sha = sha256_path(controller_path)
    controller = load_adaptive_controller(controller_path)
    if (
        registry.candidate_screen_sha256 != candidate_sha
        or candidate.matching_dimension != registry.matching_dimension
        or candidate.target != registry.target
        or candidate.tolerance != registry.tolerance
        or candidate.runtime_artifact_sha256 != runtime_sha
        or candidate.execution_public_key != execution_public_key
        or candidate.source_question_bundle_sha256 != sha256_path(question_bundle_path)
        or selected_from_screen != selected_from_registry
        or {value[2] for value in m3_sources.values()} != {controller_sha}
        or any(
            point.adaptive_policy is None
            or point.prompt_id not in m3_sources
            or not _e8_m3_screen_policy_compatible(
                point.adaptive_policy, m3_sources[point.prompt_id][0]
            )
            or point.adaptive_policy.controller_artifact_sha256 != m3_sources[point.prompt_id][2]
            for point in candidate.points
            if point.method == "M3"
        )
        or any(
            record.question_id not in trivia_questions
            or record.normalized_answer != normalize_answer(record.raw_output)
            or record.outcome
            is not deterministic_short_answer_grade(
                record.raw_output, trivia_questions[record.question_id].aliases
            )
            for point in candidate.points
            for record in point.records
        )
    ):
        raise DataValidationError("E8 candidate screen does not replay its registry")
    trivia_ids = set(ledger.contract.question_ids_by_benchmark["triviaqa"])
    if any(
        not screen.baseline_execution_records
        or not screen.intervention_execution_records
        or screen.runtime_artifact_sha256 != runtime_sha
        or screen.execution_public_key != execution_public_key
        or not set(screen.question_ids).issubset(trivia_ids)
        for screen in artifact.variant_screens
    ):
        raise DataValidationError("E8 M5 variant screens lack native source-bound execution")
    from mfh.experiments.gates import _side_metric_value

    screen_context = ledger._gate_context()
    screen_metrics = {
        "xstest_safe_refusal": "xstest_benign_non_refusal_rate",
        "harmful_refusal": "harmful_prompt_refusal_rate",
        "language_switching": "requested_language_consistency",
        "instruction_following_failure": "ifeval_pass_rate",
    }
    benchmark_by_behavior = {
        "correct_to_abstain": "triviaqa",
        "xstest_safe_refusal": "xstest",
        "harmful_refusal": "strongreject_or_harmbench",
        "language_switching": "language_consistency",
        "instruction_following_failure": "ifeval",
        "verbosity_style": "ifeval",
    }
    for screen in artifact.variant_screens:
        for record in (
            *screen.baseline_execution_records,
            *screen.intervention_execution_records,
        ):
            validate_e8_execution_record(
                record,
                condition_facts={
                    "steering_method": record.steering_method,
                    "method_artifact_sha256": record.metadata.get("method_artifact_sha256"),
                    "layer": record.layer,
                    "site": record.site.value if record.site is not None else None,
                    "token_scope": (
                        record.token_scope.value if record.token_scope is not None else None
                    ),
                    "alpha": record.alpha,
                    "sparsity": record.sparsity,
                    "prompt_template_sha256": record.metadata.get("prompt_template_sha256"),
                },
                execution_public_key=execution_public_key,
                runtime_artifact_sha256=runtime_sha,
                runtime_identity=runtime_identity,
            )
            validate_official_record(record)
        if any(
            record.normalized_answer != normalize_answer(record.raw_output)
            or record.outcome
            is not deterministic_short_answer_grade(
                record.raw_output, trivia_questions[record.question_id].aliases
            )
            for record in (
                *screen.baseline_execution_records,
                *screen.intervention_execution_records,
            )
        ):
            raise DataValidationError("E8 M5 screen factual grades differ")
        assert screen.protected_question_ids is not None
        assert screen.protected_baseline_execution_records is not None
        assert screen.protected_intervention_execution_records is not None
        for behavior in _BEHAVIORS:
            benchmark = benchmark_by_behavior[behavior]
            if not set(screen.protected_question_ids[behavior]).issubset(
                ledger.contract.question_ids_by_benchmark[benchmark]
            ):
                raise DataValidationError("E8 M5 protected screen source IDs differ")
            for index, (baseline, intervention) in enumerate(
                zip(
                    screen.protected_baseline_execution_records[behavior],
                    screen.protected_intervention_execution_records[behavior],
                    strict=True,
                )
            ):
                validate_official_record(baseline)
                validate_official_record(intervention)
                for record in (baseline, intervention):
                    validate_e8_execution_record(
                        record,
                        condition_facts={
                            "steering_method": record.steering_method,
                            "method_artifact_sha256": record.metadata.get("method_artifact_sha256"),
                            "layer": record.layer,
                            "site": (record.site.value if record.site is not None else None),
                            "token_scope": (
                                record.token_scope.value if record.token_scope is not None else None
                            ),
                            "alpha": record.alpha,
                            "sparsity": record.sparsity,
                            "prompt_template_sha256": record.metadata.get("prompt_template_sha256"),
                        },
                        execution_public_key=execution_public_key,
                        runtime_artifact_sha256=runtime_sha,
                        runtime_identity=runtime_identity,
                    )
                if behavior == "correct_to_abstain":
                    observed = (
                        baseline.outcome is not Outcome.ABSTENTION,
                        intervention.outcome is not Outcome.ABSTENTION,
                    )
                elif behavior == "verbosity_style":
                    observed = (
                        True,
                        response_verbosity_style_preserved(
                            baseline.raw_output, intervention.raw_output
                        ),
                    )
                else:
                    metric = screen_metrics[behavior]
                    observed = (
                        bool(_side_metric_value(baseline, metric, screen_context)),
                        bool(_side_metric_value(intervention, metric, screen_context)),
                    )
                expected_values = (
                    screen.protected_baseline[behavior][index],
                    screen.protected_intervention[behavior][index],
                )
                if observed != expected_values:
                    raise DataValidationError("E8 M5 protected screen values differ")
    if dict(artifact.source_fingerprints) != {
        name: ledger.contract.input_fingerprints[name]
        for name in (
            "E6_transition_evidence",
            "E7_sparse_artifacts",
            "protected_behavior_activations",
        )
    }:
        raise DataValidationError("E8 protected artifact differs from frozen sources")
    try:
        e7_receipt = json.loads((e7_source / "receipt.json").read_text(encoding="utf-8"))
        e7_body = dict(e7_receipt)
        e7_receipt_digest = e7_body.pop("receipt_digest")
        coordinate = load_coordinate_sparse_artifact(e7_source / "coordinate-artifact")
        sae = load_sae_intervention(e7_source / "sae-intervention")
    except (OSError, json.JSONDecodeError, KeyError, TypeError, DataValidationError) as exc:
        raise FrozenArtifactError(f"E8 cannot replay its E7 sparse source: {exc}") from exc
    coordinate_sha = sha256_path(e7_source / "coordinate-artifact")
    sae_sha = sha256_path(e7_source / "sae-intervention")
    if (
        e7_receipt_digest != stable_hash(e7_body)
        or e7_body.get("phase") != "E7"
        or e7_body.get("status") != "complete"
        or e7_body.get("scientific_eligible") is not True
        or e7_body.get("coordinate_artifact_sha256") != coordinate_sha
        or e7_body.get("sae_intervention_sha256") != sae_sha
    ):
        raise FrozenArtifactError("E8 E7 source receipt or promoted artifacts differ")
    selected_ids = {
        condition_id
        for values in registry.condition_ids_by_prompt.values()
        for condition_id in values.values()
    }
    expected_selected = {
        condition.condition_id
        for condition in ledger.contract.conditions
        if condition.benchmark == "triviaqa"
        and condition.steering_method in {"M1", "M3", "M4", "M5"}
    }
    if selected_ids != expected_selected:
        raise DataValidationError("E8 registry differs from the promoted TriviaQA matrix")
    for condition in ledger.contract.conditions:
        if condition.steering_method == "M3" and (
            condition.system_prompt_id not in m3_sources
            or condition.adaptive_policy is None
            or condition.method_artifact_sha256 is None
            or not _e8_m3_screen_policy_compatible(
                condition.adaptive_policy,
                m3_sources[condition.system_prompt_id][0],
            )
            or condition.method_artifact_sha256 != m3_sources[condition.system_prompt_id][1]
            or condition.adaptive_policy.controller_artifact_sha256
            != m3_sources[condition.system_prompt_id][2]
        ):
            raise DataValidationError("E8 M3 condition differs from its frozen E6 router")
    for condition in ledger.contract.conditions:
        if condition.steering_method != "M5":
            continue
        if (
            condition.method_artifact_sha256 != artifact_sha
            or condition.layer != artifact.layer
            or condition.site is not artifact.site
            or condition.token_scope is not artifact.token_scope
            or condition.alpha != artifact.alpha
            or condition.sparsity is not None
            or condition.model_repository != artifact.feature_schema.model_repository
            or condition.model_revision != artifact.feature_schema.model_revision
            or condition.quantization != artifact.feature_schema.quantization
        ):
            raise DataValidationError("E8 M5 condition differs from its selected artifact")
    condition_facts = ledger._gate_context().condition_facts
    expected_m5_direction_sha = hashlib.sha256(
        np.ascontiguousarray(artifact.selected_direction.detach().cpu().float().numpy()).tobytes(
            order="C"
        )
    ).hexdigest()
    expected_m1_values = np.ascontiguousarray(
        artifact.dense_direction.detach().cpu().float().numpy()
    )
    expected_m1_norm = float(np.linalg.norm(expected_m1_values))
    expected_m1_direction_sha = hashlib.sha256(
        np.ascontiguousarray(expected_m1_values / expected_m1_norm).tobytes(order="C")
    ).hexdigest()
    coordinate_values = np.ascontiguousarray(
        coordinate.sparse_direction.direction.detach().cpu().float().numpy()
    )
    coordinate_norm = float(np.linalg.norm(coordinate_values))
    coordinate_direction_sha = hashlib.sha256(
        np.ascontiguousarray(coordinate_values / coordinate_norm).tobytes(order="C")
    ).hexdigest()
    sae_values = np.ascontiguousarray(sae.decoded_direction.detach().cpu().float().numpy())
    sae_norm = float(np.linalg.norm(sae_values))
    sae_direction_sha = hashlib.sha256(
        np.ascontiguousarray(sae_values / sae_norm).tobytes(order="C")
    ).hexdigest()
    sae_geometry = next(iter(sae.evidence)).spec
    sae_sparsity = len(sae.latent_direction.selected_features) / float(
        sae.training.config.resolved_latent_width
    )
    expected_m4 = {
        coordinate_sha: (
            coordinate_direction_sha,
            coordinate_norm,
            coordinate.reference_rms,
            coordinate.layer,
            coordinate.site,
            coordinate.token_scope,
            coordinate.alpha,
            coordinate.sparse_direction.retained_fraction,
        ),
        sae_sha: (
            sae_direction_sha,
            sae_norm,
            artifact.reference_rms,
            sae_geometry.layer,
            sae_geometry.site,
            sae_geometry.token_scope,
            sae_geometry.alpha,
            sae_sparsity,
        ),
    }
    selected_conditions = {
        (condition.system_prompt_id, condition.steering_method): condition
        for condition in ledger.contract.conditions
        if condition.benchmark == "triviaqa"
        and condition.steering_method in {"M1", "M3", "M4", "M5"}
    }
    expected_fixed_directions: dict[str, dict[str | None, tuple[str, float]]] = {
        "M1": {None: (expected_m1_direction_sha, artifact.reference_rms)},
        "M4": {
            method_artifact: (facts[0], facts[2]) for method_artifact, facts in expected_m4.items()
        },
        "M5": {artifact_sha: (expected_m5_direction_sha, artifact.reference_rms)},
    }
    for point in candidate.points:
        selected_condition = selected_conditions[(point.prompt_id, point.method)]
        selected_point = point.selected_condition_id is not None
        if (
            point.selected_condition_id not in {None, selected_condition.condition_id}
            or any(
                record.metadata.get("source_question_sha256")
                != question_source_fingerprint(trivia_questions[record.question_id])
                or record.metadata.get("prompt_template_sha256")
                != selected_condition.prompt_template_sha256
                or record.metadata.get("method_artifact_sha256")
                != selected_condition.method_artifact_sha256
                for record in point.records
            )
            or (
                selected_point
                and point.method != "M3"
                and not math.isclose(
                    point.alpha, selected_condition.alpha, rel_tol=0, abs_tol=1e-12
                )
            )
            or (
                selected_point
                and point.method == "M3"
                and (
                    point.adaptive_policy != selected_condition.adaptive_policy
                    or point.adaptive_policy is None
                )
            )
        ):
            raise DataValidationError("E8 candidate winner differs from its frozen final condition")
        if point.method == "M3":
            if point.adaptive_policy is None:
                raise DataValidationError("E8 M3 candidate lacks its adaptive policy")
            replay_condition = replace(
                selected_condition,
                adaptive_policy=point.adaptive_policy,
            )
            if replay_condition.condition_id != point.candidate_condition_id:
                raise DataValidationError(
                    "E8 M3 candidate identity does not derive from its policy"
                )
            for record in point.records:
                _validate_e8_adaptive_controller_record(
                    record,
                    condition=replay_condition,
                    controller=controller,
                    controller_artifact_sha256=controller_sha,
                    runtime_identity=runtime_identity,
                )
            continue
        component_key = (
            None if point.method == "M1" else str(selected_condition.method_artifact_sha256)
        )
        expected_direction = expected_fixed_directions[point.method].get(component_key)
        execution_differs = expected_direction is None
        if expected_direction is not None:
            for record in point.records:
                validate_e8_execution_record(
                    record,
                    condition_facts={
                        "steering_method": record.steering_method,
                        "method_artifact_sha256": record.metadata.get("method_artifact_sha256"),
                        "layer": record.layer,
                        "site": record.site.value if record.site is not None else None,
                        "token_scope": (
                            record.token_scope.value if record.token_scope is not None else None
                        ),
                        "alpha": record.alpha,
                        "sparsity": record.sparsity,
                        "prompt_template_sha256": record.metadata.get("prompt_template_sha256"),
                    },
                    execution_public_key=execution_public_key,
                    runtime_artifact_sha256=runtime_sha,
                    runtime_identity=runtime_identity,
                )
                trace = record.metadata.get("intervention_trace")
                if (
                    record.layer != selected_condition.layer
                    or record.site is not selected_condition.site
                    or record.token_scope is not selected_condition.token_scope
                    or record.sparsity != selected_condition.sparsity
                    or not isinstance(trace, Mapping)
                    or trace.get("direction_sha256") != expected_direction[0]
                    or trace.get("reference_rms") != expected_direction[1]
                ):
                    execution_differs = True
                    break
        if execution_differs:
            raise DataValidationError(
                "E8 candidate execution differs from its exact source component"
            )
    for condition in ledger.contract.conditions:
        if condition.steering_method == "M1" and (
            condition.layer != artifact.layer
            or condition.site is not artifact.site
            or condition.token_scope is not artifact.token_scope
            or condition.sparsity is not None
        ):
            raise DataValidationError("E8 M1 condition differs from its protected source")
        if condition.steering_method == "M4":
            facts = expected_m4.get(str(condition.method_artifact_sha256))
            if (
                facts is None
                or (
                    condition.layer,
                    condition.site,
                    condition.token_scope,
                    condition.sparsity,
                )
                != (facts[3], facts[4], facts[5], facts[7])
                or condition.alpha not in {0.1, 0.25, 0.5, 1.0, 2.0}
            ):
                raise DataValidationError("E8 M4 condition is not an E7-promoted artifact")
    for record in ledger.records():
        if record.metadata.get("decoding_max_new_tokens") != candidate.max_new_tokens:
            raise DataValidationError("E8 final decoding cap differs from candidate screening")
        validate_official_record(record)
        if record.steering_method == "M3":
            _validate_e8_adaptive_controller_record(
                record,
                condition=next(
                    condition
                    for condition in ledger.contract.conditions
                    if condition.condition_id == record.condition_id
                ),
                controller=controller,
                controller_artifact_sha256=controller_sha,
                runtime_identity=runtime_identity,
            )
            continue
        validate_e8_execution_record(
            record,
            condition_facts=condition_facts[record.condition_id],
            execution_public_key=execution_public_key,
            runtime_artifact_sha256=runtime_sha,
            runtime_identity=runtime_identity,
        )
        if record.steering_method == "M5":
            trace = record.metadata["intervention_trace"]
            if (
                not isinstance(trace, Mapping)
                or trace.get("direction_sha256") != expected_m5_direction_sha
                or trace.get("reference_rms") != artifact.reference_rms
                or not math.isclose(
                    float(trace.get("raw_alpha", math.nan)),
                    artifact.alpha * artifact.reference_rms,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                raise DataValidationError("E8 M5 runtime trace differs from selected direction")
        elif record.steering_method == "M1":
            trace = record.metadata["intervention_trace"]
            if (
                trace.get("direction_sha256") != expected_m1_direction_sha
                or trace.get("source_direction_norm") != expected_m1_norm
                or trace.get("reference_rms") != artifact.reference_rms
                or not math.isclose(
                    float(trace.get("raw_alpha", math.nan)),
                    record.alpha * expected_m1_norm * artifact.reference_rms,
                    rel_tol=1e-6,
                    abs_tol=1e-8,
                )
            ):
                raise DataValidationError("E8 M1 trace differs from the protected source")
        elif record.steering_method == "M4":
            facts = expected_m4[str(record.metadata["method_artifact_sha256"])]
            trace = record.metadata["intervention_trace"]
            if (
                trace.get("direction_sha256") != facts[0]
                or trace.get("source_direction_norm") != facts[1]
                or trace.get("reference_rms") != facts[2]
                or not math.isclose(
                    float(trace.get("raw_alpha", math.nan)),
                    record.alpha * facts[1] * facts[2],
                    rel_tol=1e-6,
                    abs_tol=1e-8,
                )
            ):
                raise DataValidationError("E8 M4 trace differs from its E7 artifact")
    return (
        ledger,
        artifact,
        registry,
        input_paths,
        artifact_sha,
        registry_sha,
        runtime_sha,
        candidate_sha,
        controller_path,
        controller_sha,
    )


def _e8_gate_rows(ledger: Any) -> tuple[dict[str, Any], ...]:
    from mfh.experiments.gates import _side_metric_value

    context = ledger._gate_context()
    records = {(record.condition_id, record.question_id): record for record in ledger.records()}
    conditions: dict[tuple[str, str, str], dict[str, str]] = {}
    for condition in ledger.contract.conditions:
        conditions.setdefault(
            (
                condition.benchmark,
                condition.system_prompt_id,
                condition.comparison_group,
            ),
            {},
        )[condition.steering_method] = condition.condition_id
    rows: list[dict[str, Any]] = []
    for (benchmark, _prompt, _group), methods in sorted(conditions.items()):
        metrics = _SIDE_METRICS.get(benchmark)
        if metrics is None:
            continue
        baseline_id = methods.get("M0")
        if baseline_id is None:
            raise DataValidationError("E8 side-effect stratum lacks M0")
        for method in ("M1", "M3", "M4", "M5"):
            intervention_id = methods.get(method)
            if intervention_id is None:
                raise DataValidationError("E8 side-effect stratum lacks a promoted method")
            for question_id in ledger.contract.question_ids_by_benchmark[benchmark]:
                baseline = records[(baseline_id, question_id)]
                intervention = records[(intervention_id, question_id)]
                for metric in metrics:
                    rows.append(
                        {
                            "metric": metric,
                            "question_id": question_id,
                            "baseline_condition_id": baseline_id,
                            "intervention_condition_id": intervention_id,
                            "baseline_value": _side_metric_value(baseline, metric, context),
                            "intervention_value": _side_metric_value(intervention, metric, context),
                        }
                    )
    return tuple(rows)


def finalize_e8_phase(
    destination: str | Path,
    *,
    ledger_directory: str | Path,
    study: Any,
    protected_artifact: str | Path,
    operating_point_registry: str | Path,
    candidate_screen: str | Path,
    runtime_artifact: str | Path,
    analysis_protocol: str | Path = "configs/analysis/confirmatory.yaml",
    research_plan: str | Path = "docs/research-plan.md",
) -> Mapping[str, Any]:
    normalized = validate_active_study_artifact_paths(
        {
            "E8 finalization": destination,
            "E8 phase ledger": ledger_directory,
            "E8 protected artifact": protected_artifact,
            "E8 operating-point registry": operating_point_registry,
            "E8 candidate screen": candidate_screen,
            "E8 runtime attestation": runtime_artifact,
        }
    )
    destination = normalized["E8 finalization"]
    ledger_directory = normalized["E8 phase ledger"]
    protected_artifact = normalized["E8 protected artifact"]
    operating_point_registry = normalized["E8 operating-point registry"]
    candidate_screen = normalized["E8 candidate screen"]
    runtime_artifact = normalized["E8 runtime attestation"]
    from mfh.analysis.protocol import load_analysis_protocol
    from mfh.experiments.evidence import GateResult
    from mfh.experiments.gates import write_gate_evidence
    from mfh.experiments.protocol import ExperimentPhase
    from mfh.experiments.runner import (
        _copy_frozen_artifact,
        package_portable_phase_ledger,
    )

    output = Path(destination)
    if output.is_symlink():
        raise FrozenArtifactError(f"refusing linked E8 finalization: {output}")
    if output.exists():
        return verify_e8_phase(output)
    (
        ledger,
        _artifact,
        _registry,
        input_paths,
        artifact_sha,
        registry_sha,
        runtime_sha,
        candidate_sha,
        controller_path,
        controller_sha,
    ) = _validate_e8_final_inputs(
        ledger_directory=ledger_directory,
        study=study,
        protected_artifact=protected_artifact,
        operating_point_registry=operating_point_registry,
        candidate_screen=candidate_screen,
        runtime_artifact=runtime_artifact,
    )
    analysis_source = Path(analysis_protocol).resolve()
    plan_source = Path(research_plan).resolve()
    analysis = load_analysis_protocol(analysis_source)
    analysis_sha = sha256_file(analysis_source)
    plan_sha = sha256_file(plan_source)
    if analysis.research_plan_sha256 != plan_sha:
        raise DataValidationError("E8 analysis protocol differs from the exact research plan")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        _copy_frozen_artifact(
            Path(protected_artifact).resolve(),
            stage / "protected-artifact",
            artifact_sha,
        )
        _copy_frozen_artifact(
            Path(operating_point_registry).resolve(),
            stage / "operating-point-registry.json",
            registry_sha,
        )
        _copy_frozen_artifact(
            Path(candidate_screen).resolve(),
            stage / "candidate-screen.json",
            candidate_sha,
        )
        _copy_frozen_artifact(
            Path(runtime_artifact).resolve(),
            stage / "runtime-attestation.json",
            runtime_sha,
        )
        _copy_frozen_artifact(
            controller_path,
            stage / "adaptive-controller",
            controller_sha,
        )
        packaged_analysis = stage / "configs" / "analysis" / "confirmatory.yaml"
        packaged_plan = stage / "docs" / "research-plan.md"
        _copy_frozen_artifact(analysis_source, packaged_analysis, analysis_sha)
        _copy_frozen_artifact(plan_source, packaged_plan, plan_sha)
        for name, path in input_paths.items():
            _copy_frozen_artifact(
                path,
                stage / "inputs" / name,
                ledger.contract.input_fingerprints[name],
            )
        rows = _e8_gate_rows(ledger)
        results: dict[str, GateResult] = {}
        for gate in _E8_GATES:
            evidence_path = stage / f"{gate}.json"
            parameters = {"operating_point_registry_sha256": registry_sha}
            supporting = {"operating-point-registry": stage / "operating-point-registry.json"}
            if gate == "utility_safety_language_noninferiority":
                parameters["analysis_protocol_sha256"] = analysis_sha
                supporting["analysis-protocol"] = packaged_analysis
            write_gate_evidence(
                evidence_path,
                phase=ExperimentPhase.E8,
                gate=gate,
                contract_digest=ledger.contract.digest,
                record_set_digest=ledger.record_set_digest(),
                observations=(rows if gate == "utility_safety_language_noninferiority" else ()),
                parameters=parameters,
            )
            results[gate] = ledger.evaluate_gate(
                gate,
                evidence_path,
                supporting_artifacts=supporting,
            )
        expected_gate_digests = {name: results[name].gate_digest for name in sorted(results)}
        complete_marker = ledger.directory / "complete.json"
        falsified_marker = ledger.directory / "falsified.json"
        if complete_marker.is_file():
            terminal = ledger.verify_complete()
            if (
                not all(value.passed for value in results.values())
                or dict(terminal.gate_result_digests) != expected_gate_digests
            ):
                raise FrozenArtifactError("E8 recovered terminal differs from re-derived gates")
            status = "complete"
            terminal_digest = terminal.completion_digest
        elif falsified_marker.is_file():
            terminal = ledger.verify_falsified()
            if (
                all(value.passed for value in results.values())
                or dict(terminal.gate_result_digests) != expected_gate_digests
            ):
                raise FrozenArtifactError(
                    "E8 recovered falsification differs from re-derived gates"
                )
            status = "falsified"
            terminal_digest = terminal.falsification_digest
        elif all(value.passed for value in results.values()):
            terminal = ledger.finalize(results)
            status = "complete"
            terminal_digest = terminal.completion_digest
        else:
            terminal = ledger.finalize_falsified(results)
            status = "falsified"
            terminal_digest = terminal.falsification_digest
        portable_ledger_sha = package_portable_phase_ledger(
            ledger.directory,
            stage / "portable-ledger",
            study=study,
        )
        if study.source_path is None:
            raise DataValidationError(
                "E8 portable finalization requires the loaded phases.yaml protocol"
            )
        study_path = stage / "configs" / "experiments" / "phases.yaml"
        _copy_frozen_artifact(
            study.source_path,
            study_path,
            sha256_file(study.source_path),
        )
        receipt_body = {
            "schema_version": 3,
            "phase": ExperimentPhase.E8.value,
            "status": status,
            "portable_ledger_sha256": portable_ledger_sha,
            "study_protocol_sha256": sha256_file(study_path),
            "analysis_protocol_sha256": analysis_sha,
            "research_plan_sha256": plan_sha,
            "contract_digest": ledger.contract.digest,
            "record_set_digest": terminal.record_set_digest,
            "protected_artifact_sha256": artifact_sha,
            "operating_point_registry_sha256": registry_sha,
            "candidate_screen_sha256": candidate_sha,
            "runtime_artifact_sha256": runtime_sha,
            "adaptive_controller_artifact_sha256": controller_sha,
            "gate_result_digests": dict(terminal.gate_result_digests),
            "terminal_digest": terminal_digest,
            "scientific_eligible": status == "complete",
        }
        (stage / "receipt.json").write_text(
            json.dumps(
                {**receipt_body, "receipt_digest": stable_hash(receipt_body)},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return verify_e8_phase(output)


def verify_e8_phase(
    directory: str | Path,
) -> Mapping[str, Any]:
    from mfh.experiments.protocol import load_study_protocol

    source = Path(directory)
    expected = {
        "protected-artifact",
        "operating-point-registry.json",
        "candidate-screen.json",
        "runtime-attestation.json",
        "adaptive-controller",
        "inputs",
        "portable-ledger",
        "configs",
        "docs",
        "receipt.json",
        *(f"{gate}.json" for gate in _E8_GATES),
    }
    if (
        source.is_symlink()
        or not source.is_dir()
        or {value.name for value in source.iterdir()} != expected
        or any(value.is_symlink() for value in source.rglob("*"))
    ):
        raise FrozenArtifactError("E8 finalization inventory differs")
    study_path = source / "configs" / "experiments" / "phases.yaml"
    analysis_path = source / "configs" / "analysis" / "confirmatory.yaml"
    plan_path = source / "docs" / "research-plan.md"
    study = load_study_protocol(study_path)
    from mfh.analysis.protocol import load_analysis_protocol

    analysis = load_analysis_protocol(analysis_path)
    if analysis.research_plan_sha256 != sha256_file(plan_path):
        raise FrozenArtifactError("E8 packaged analysis protocol differs from its plan")
    ledger_directory = source / "portable-ledger"
    (
        ledger,
        _artifact,
        _registry,
        input_paths,
        artifact_sha,
        registry_sha,
        runtime_sha,
        candidate_sha,
        _controller_path,
        controller_sha,
    ) = _validate_e8_final_inputs(
        ledger_directory=ledger_directory,
        study=study,
        protected_artifact=source / "protected-artifact",
        operating_point_registry=source / "operating-point-registry.json",
        candidate_screen=source / "candidate-screen.json",
        runtime_artifact=source / "runtime-attestation.json",
        adaptive_controller_artifact=source / "adaptive-controller",
    )
    if set((source / "inputs").iterdir()) != {source / "inputs" / name for name in _E8_INPUTS}:
        raise FrozenArtifactError("E8 packaged input inventory differs")
    if any(
        sha256_path(source / "inputs" / name) != ledger.contract.input_fingerprints[name]
        or sha256_path(path) != ledger.contract.input_fingerprints[name]
        for name, path in input_paths.items()
    ):
        raise FrozenArtifactError("E8 packaged input bytes differ")
    try:
        receipt = json.loads((source / "receipt.json").read_text(encoding="utf-8"))
        body = dict(receipt)
        receipt_digest = body.pop("receipt_digest")
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise FrozenArtifactError(f"cannot read E8 receipt: {exc}") from exc
    if receipt_digest != stable_hash(body):
        raise FrozenArtifactError("E8 receipt digest differs")
    status = body.get("status")
    if status == "complete":
        terminal = ledger.verify_complete()
        terminal_digest = terminal.completion_digest
        scientific = True
    elif status == "falsified":
        terminal = ledger.verify_falsified()
        terminal_digest = terminal.falsification_digest
        scientific = False
    else:
        raise FrozenArtifactError("E8 terminal status differs")
    expected_gate_artifacts = {
        f"{gate}/evaluation": sha256_file(source / f"{gate}.json") for gate in _E8_GATES
    }
    expected_gate_artifacts.update(
        {f"{gate}/operating-point-registry": registry_sha for gate in _E8_GATES}
    )
    expected_gate_artifacts["utility_safety_language_noninferiority/analysis-protocol"] = (
        sha256_file(analysis_path)
    )
    if (
        body.get("schema_version") != 3
        or body.get("phase") != "E8"
        or body.get("portable_ledger_sha256") != sha256_path(ledger_directory)
        or body.get("study_protocol_sha256") != sha256_file(study_path)
        or body.get("analysis_protocol_sha256") != sha256_file(analysis_path)
        or body.get("research_plan_sha256") != sha256_file(plan_path)
        or body.get("contract_digest") != ledger.contract.digest
        or body.get("record_set_digest") != terminal.record_set_digest
        or body.get("protected_artifact_sha256") != artifact_sha
        or body.get("operating_point_registry_sha256") != registry_sha
        or body.get("candidate_screen_sha256") != candidate_sha
        or body.get("runtime_artifact_sha256") != runtime_sha
        or body.get("adaptive_controller_artifact_sha256") != controller_sha
        or body.get("gate_result_digests") != dict(terminal.gate_result_digests)
        or body.get("terminal_digest") != terminal_digest
        or body.get("scientific_eligible") is not scientific
        or dict(terminal.gate_artifact_fingerprints) != expected_gate_artifacts
    ):
        raise FrozenArtifactError("E8 finalization differs from terminal replay")
    return MappingProxyType(
        {
            "valid": True,
            "status": status,
            "receipt_digest": receipt_digest,
            "terminal_digest": terminal_digest,
            "scientific_eligible": scientific,
        }
    )

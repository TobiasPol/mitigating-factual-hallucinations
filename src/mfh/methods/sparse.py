"""Coordinate-sparse and sparse-autoencoder interventions for M4."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import re
import resource
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast, overload

import numpy as np
import torch
import torch.nn.functional as F
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from numpy.typing import NDArray
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from mfh.artifact_namespace import validate_active_study_artifact_paths
from mfh.contracts import ActivationSite, GenerationRecord, Outcome, TokenScope
from mfh.data.normalization import normalize_answer
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.evaluation.grading import deterministic_short_answer_grade
from mfh.methods.features import ActivationFeatureSchema
from mfh.methods.probes import ProbeDataset
from mfh.provenance import canonical_json, sha256_file, sha256_path, stable_hash

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OUTCOME_TO_CODE = {outcome: index for index, outcome in enumerate(Outcome)}
_CODE_TO_OUTCOME = tuple(Outcome)
_COORDINATE_FRACTIONS = (0.01, 0.05, 0.10, 0.25)
_COORDINATE_ALPHAS = (0.1, 0.25, 0.5, 1.0, 2.0)
_MIN_NATIVE_CAUSAL_FACTUALITY_SAMPLES = 100
_MIN_NATIVE_CAUSAL_PROTECTED_SAMPLES = 50


def _matrix(values: Tensor, *, width: int | None = None) -> Tensor:
    tensor = values.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if tensor.ndim != 2 or tensor.shape[0] == 0 or tensor.shape[1] == 0:
        raise DataValidationError("activation values must have shape [rows, width]")
    if width is not None and tensor.shape[1] != width:
        raise DataValidationError(f"expected activation width {width}, got {tensor.shape[1]}")
    if not torch.isfinite(tensor).all():
        raise DataValidationError("activation values contain NaN or infinity")
    return tensor


def standardized_effect_size(
    correct: Tensor,
    incorrect: Tensor,
    *,
    epsilon: float = 1e-8,
) -> Tensor:
    """Compute the pooled standardized C-minus-I effect from M4a."""

    positive = _matrix(correct)
    negative = _matrix(incorrect, width=positive.shape[1])
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise DataValidationError("effect-size epsilon must be finite and positive")
    positive_variance = positive.var(dim=0, unbiased=False)
    negative_variance = negative.var(dim=0, unbiased=False)
    denominator = torch.sqrt(0.5 * (positive_variance + negative_variance) + epsilon)
    return (positive.mean(dim=0) - negative.mean(dim=0)) / denominator


@dataclass(frozen=True, slots=True)
class CoordinateSparseDirection:
    direction: Tensor
    mask: Tensor
    effect_size: Tensor
    retained_fraction: float
    renormalized: bool

    def __post_init__(self) -> None:
        direction = self.direction.detach().cpu().float().contiguous().clone()
        effect = self.effect_size.detach().cpu().float().contiguous().clone()
        mask = self.mask.detach().cpu().bool().contiguous().clone()
        if direction.ndim != 1 or effect.shape != direction.shape or mask.shape != direction.shape:
            raise DataValidationError("coordinate-sparse tensors must be equal-width vectors")
        if not torch.isfinite(direction).all() or not torch.isfinite(effect).all():
            raise DataValidationError("coordinate-sparse tensors contain NaN or infinity")
        if not 0 < self.retained_fraction <= 1 or not mask.any():
            raise DataValidationError("coordinate sparsity must retain a non-empty fraction")
        if (direction[~mask] != 0).any():
            raise DataValidationError("coordinate-sparse direction is nonzero outside its mask")
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "effect_size", effect)
        object.__setattr__(self, "mask", mask)

    @property
    def retained_dimensions(self) -> int:
        return int(self.mask.sum())


def coordinate_sparse_direction(
    dense_direction: Tensor,
    effect_size: Tensor,
    *,
    retained_fraction: float,
    renormalize: bool = False,
) -> CoordinateSparseDirection:
    dense = dense_direction.detach().cpu().float().contiguous()
    effect = effect_size.detach().cpu().float().contiguous()
    if dense.ndim != 1 or dense.numel() == 0 or effect.shape != dense.shape:
        raise DataValidationError("dense direction and effect size must be equal-width vectors")
    if not torch.isfinite(dense).all() or not torch.isfinite(effect).all():
        raise DataValidationError("sparse direction inputs contain NaN or infinity")
    if not 0 < retained_fraction <= 1:
        raise DataValidationError("retained fraction must be in (0, 1]")
    retained = max(1, math.ceil(dense.numel() * retained_fraction))
    order = torch.argsort(effect.abs(), descending=True, stable=True)
    mask = torch.zeros_like(dense, dtype=torch.bool)
    mask[order[:retained]] = True
    sparse = dense * mask
    norm = torch.linalg.vector_norm(sparse)
    if float(norm) <= 0:
        raise DataValidationError("selected coordinates contain no dense-vector mass")
    if renormalize:
        sparse = sparse / norm
    return CoordinateSparseDirection(sparse, mask, effect, retained_fraction, renormalize)


@dataclass(frozen=True, slots=True)
class CoordinateScreenPoint:
    """Raw paired outcomes for one preregistered M4a sparsity/alpha cell."""

    retained_fraction: float
    alpha: float
    baseline_condition_id: str
    intervention_condition_id: str
    question_ids: tuple[str, ...]
    baseline_outcomes: tuple[Outcome, ...]
    intervention_outcomes: tuple[Outcome, ...]

    def __post_init__(self) -> None:
        question_ids = tuple(str(value).strip() for value in self.question_ids)
        baseline = tuple(Outcome(value) for value in self.baseline_outcomes)
        intervention = tuple(Outcome(value) for value in self.intervention_outcomes)
        if (
            self.retained_fraction not in _COORDINATE_FRACTIONS
            or self.alpha not in _COORDINATE_ALPHAS
            or not _SHA256.fullmatch(self.baseline_condition_id)
            or not _SHA256.fullmatch(self.intervention_condition_id)
            or self.baseline_condition_id == self.intervention_condition_id
            or not question_ids
            or any(not value for value in question_ids)
            or len(set(question_ids)) != len(question_ids)
            or len(baseline) != len(question_ids)
            or len(intervention) != len(question_ids)
            or any(value is Outcome.UNSCORABLE for value in (*baseline, *intervention))
        ):
            raise DataValidationError("coordinate screen point is incomplete or invalid")
        object.__setattr__(self, "question_ids", question_ids)
        object.__setattr__(self, "baseline_outcomes", baseline)
        object.__setattr__(self, "intervention_outcomes", intervention)

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
    def digest(self) -> str:
        return stable_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "retained_fraction": self.retained_fraction,
            "alpha": self.alpha,
            "baseline_condition_id": self.baseline_condition_id,
            "intervention_condition_id": self.intervention_condition_id,
            "question_ids": list(self.question_ids),
            "baseline_outcomes": [value.value for value in self.baseline_outcomes],
            "intervention_outcomes": [
                value.value for value in self.intervention_outcomes
            ],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CoordinateScreenPoint:
        if set(value) != {
            "retained_fraction",
            "alpha",
            "baseline_condition_id",
            "intervention_condition_id",
            "question_ids",
            "baseline_outcomes",
            "intervention_outcomes",
        }:
            raise DataValidationError("coordinate screen-point keys differ")
        return cls(
            retained_fraction=float(value["retained_fraction"]),
            alpha=float(value["alpha"]),
            baseline_condition_id=str(value["baseline_condition_id"]),
            intervention_condition_id=str(value["intervention_condition_id"]),
            question_ids=tuple(value["question_ids"]),
            baseline_outcomes=tuple(Outcome(item) for item in value["baseline_outcomes"]),
            intervention_outcomes=tuple(
                Outcome(item) for item in value["intervention_outcomes"]
            ),
        )


def select_coordinate_screen_point(
    points: Sequence[CoordinateScreenPoint],
) -> CoordinateScreenPoint:
    """Select the best non-over-refusing cell with deterministic sparse tie breaks."""

    frozen = tuple(points)
    expected = {
        (fraction, alpha)
        for fraction in _COORDINATE_FRACTIONS
        for alpha in _COORDINATE_ALPHAS
    }
    if (
        len(frozen) != len(expected)
        or {(item.retained_fraction, item.alpha) for item in frozen} != expected
        or len({item.question_ids for item in frozen}) != 1
        or len({item.baseline_outcomes for item in frozen}) != 1
        or len({item.baseline_condition_id for item in frozen}) != 1
        or len({item.intervention_condition_id for item in frozen}) != len(frozen)
    ):
        raise DataValidationError("coordinate screen must contain one exact paired 4x5 grid")
    eligible = tuple(item for item in frozen if item.coverage_change >= -0.02)
    if not eligible:
        raise DataValidationError("coordinate screen has no coverage-preserving candidate")
    return max(
        eligible,
        key=lambda item: (
            item.accuracy_gain,
            item.coverage_change,
            -item.retained_fraction,
            -item.alpha,
        ),
    )


def coordinate_screen_contract_digest(
    *,
    feature_schema: ActivationFeatureSchema,
    source_artifact_sha256: str,
    source_tensor_index: tuple[str, str, str, int],
    source_direction_sha256: str,
    layer: int,
    site: ActivationSite,
    token_scope: TokenScope,
    runtime_artifact_sha256: str,
    execution_public_key: str,
    points: Sequence[CoordinateScreenPoint],
    source_question_bundle_sha256: str | None = None,
) -> str:
    frozen = tuple(points)
    expected = {
        (fraction, alpha)
        for fraction in _COORDINATE_FRACTIONS
        for alpha in _COORDINATE_ALPHAS
    }
    if (
        {(point.retained_fraction, point.alpha) for point in frozen} != expected
        or len(frozen) != len(expected)
        or len({point.question_ids for point in frozen}) != 1
        or not _SHA256.fullmatch(runtime_artifact_sha256)
        or not _SHA256.fullmatch(execution_public_key)
    ):
        raise DataValidationError("coordinate screen contract is incomplete")
    question_bundle_sha = (
        feature_schema.split_manifest_digest
        if source_question_bundle_sha256 is None
        else source_question_bundle_sha256
    )
    if not _SHA256.fullmatch(question_bundle_sha):
        raise DataValidationError("coordinate screen question-bundle identity is invalid")
    return stable_hash(
        {
            "schema_version": 3,
            "feature_schema": feature_schema.to_dict(),
            "source_artifact_sha256": source_artifact_sha256,
            "source_tensor_index": list(source_tensor_index),
            "source_direction_sha256": source_direction_sha256,
            "layer": layer,
            "site": site.value,
            "token_scope": token_scope.value,
            "runtime_artifact_sha256": runtime_artifact_sha256,
            "execution_public_key": execution_public_key,
            "source_question_bundle_sha256": question_bundle_sha,
            "cells": [
                {
                    "retained_fraction": point.retained_fraction,
                    "alpha": point.alpha,
                    "question_ids": list(point.question_ids),
                }
                for point in sorted(
                    frozen, key=lambda value: (value.retained_fraction, value.alpha)
                )
            ],
        }
    )


def coordinate_screen_condition_id(
    contract_digest: str,
    *,
    retained_fraction: float | None = None,
    alpha: float | None = None,
) -> str:
    """Re-derive the frozen internal screen condition identity."""

    baseline = retained_fraction is None and alpha is None
    intervention = (
        retained_fraction in _COORDINATE_FRACTIONS and alpha in _COORDINATE_ALPHAS
    )
    if not _SHA256.fullmatch(contract_digest) or not (baseline or intervention):
        raise DataValidationError("coordinate screen condition identity is invalid")
    return stable_hash(
        {
            "schema_version": 1,
            "coordinate_screen_contract_digest": contract_digest,
            "steering_method": "M0" if baseline else "M4a",
            "retained_fraction": retained_fraction,
            "alpha": alpha,
        }
    )


def coordinate_screen_execution_receipt_body(
    record: GenerationRecord,
    *,
    contract_digest: str,
    runtime_artifact_sha256: str,
) -> dict[str, Any]:
    """Canonical runtime-owned proof for one internal coordinate-screen row."""

    return {
        "schema_version": 3,
        "coordinate_screen_contract_digest": contract_digest,
        "runtime_artifact_sha256": runtime_artifact_sha256,
        "prompt_template_sha256": record.metadata.get("prompt_template_sha256"),
        "source_question_sha256": record.metadata.get("source_question_sha256"),
        "question_id": record.question_id,
        "condition_id": record.condition_id,
        "rendered_prompt_hash": record.rendered_prompt_hash,
        "raw_output_sha256": hashlib.sha256(record.raw_output.encode()).hexdigest(),
        "normalized_answer_sha256": hashlib.sha256(
            record.normalized_answer.encode()
        ).hexdigest(),
        "outcome": record.outcome.value,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
        "generation_latency_seconds": record.generation_latency_seconds,
        "generation_runtime_metrics": record.metadata.get("generation_runtime_metrics"),
        "intervention_trace": record.metadata.get("intervention_trace"),
    }


def _validate_coordinate_screen_execution_record(
    record: GenerationRecord,
    *,
    contract_digest: str,
    runtime_artifact_sha256: str,
    execution_public_key: str,
    prompt_template_sha256: str,
    runtime_identity: Mapping[str, Any] | None = None,
) -> None:
    # Keep this import local. Importing ``mfh.experiments`` while the methods
    # package is still initializing creates a sparse -> experiments -> runner ->
    # SAE-stability -> sparse cycle during normal test and CLI collection.
    from mfh.experiments.runtime_evidence import validate_generation_runtime_metrics

    if (
        record.metadata.get("coordinate_screen_contract_digest") != contract_digest
        or record.metadata.get("coordinate_screen_runtime_artifact_sha256")
        != runtime_artifact_sha256
        or record.metadata.get("coordinate_screen_execution_public_key")
        != execution_public_key
        or record.metadata.get("prompt_template_sha256")
        != prompt_template_sha256
    ):
        raise DataValidationError("coordinate screen execution identity differs")
    trace = record.metadata.get("intervention_trace")
    if record.steering_method == "M0":
        if trace is not None or record.metadata.get("intervention_trace_digest") is not None:
            raise DataValidationError("coordinate screen baseline contains an intervention")
    elif record.steering_method == "M4a":
        expected_trace_keys = {
            "coordinate_screen_contract_digest",
            "source_artifact_sha256",
            "direction_sha256",
            "layer",
            "site",
            "token_scope",
            "standardized_alpha",
            "raw_alpha",
            "retained_fraction",
            "reference_rms",
            "source_direction_norm",
            "applied_tokens",
            "applied_token_indices",
            "pre_activation_sha256",
            "post_activation_sha256",
            "delta_sha256",
        }
        numeric = (
            "standardized_alpha",
            "raw_alpha",
            "retained_fraction",
            "reference_rms",
            "source_direction_norm",
        )
        if (
            not isinstance(trace, Mapping)
            or set(trace) != expected_trace_keys
            or trace.get("coordinate_screen_contract_digest") != contract_digest
            or any(
                isinstance(trace.get(name), bool)
                or not isinstance(trace.get(name), int | float)
                or not math.isfinite(float(trace[name]))
                or float(trace[name]) <= 0
                for name in numeric
            )
            or type(trace.get("applied_tokens")) is not int
            or int(trace["applied_tokens"]) <= 0
            or type(trace.get("applied_token_indices")) is not list
            or len(trace["applied_token_indices"]) != trace["applied_tokens"]
            or any(type(value) is not int for value in trace["applied_token_indices"])
            or any(
                type(trace.get(name)) is not str
                or _SHA256.fullmatch(str(trace[name])) is None
                for name in (
                    "source_artifact_sha256",
                    "direction_sha256",
                    "pre_activation_sha256",
                    "post_activation_sha256",
                    "delta_sha256",
                )
            )
            or trace["pre_activation_sha256"] == trace["post_activation_sha256"]
            or record.metadata.get("intervention_trace_digest")
            != stable_hash(dict(trace))
        ):
            raise DataValidationError("coordinate screen trace is not a material fixed edit")
    else:
        raise DataValidationError("coordinate screen record uses an unknown method")
    signature = record.metadata.get("coordinate_screen_execution_signature")
    if type(signature) is not str or re.fullmatch(r"[0-9a-f]{128}", signature) is None:
        raise DataValidationError("coordinate screen row lacks its runtime signature")
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(execution_public_key)).verify(
            bytes.fromhex(signature),
            canonical_json(
                coordinate_screen_execution_receipt_body(
                    record,
                    contract_digest=contract_digest,
                    runtime_artifact_sha256=runtime_artifact_sha256,
                )
            ).encode(),
        )
    except (InvalidSignature, ValueError) as exc:
        raise DataValidationError("coordinate screen runtime signature is invalid") from exc
    validate_generation_runtime_metrics(
        record.metadata.get("generation_runtime_metrics"),
        record=record,
        runtime_identity=runtime_identity,
        expected_auxiliary_peak_memory_bytes=0,
    )


def _coordinate_screen_record_set_digest(
    records: Sequence[GenerationRecord],
) -> str:
    return stable_hash(
        [
            record.to_dict()
            for record in sorted(
                records, key=lambda value: (value.condition_id, value.question_id)
            )
        ]
    )


@dataclass(frozen=True, slots=True)
class CoordinateSparseArtifact:
    sparse_direction: CoordinateSparseDirection
    feature_schema: ActivationFeatureSchema
    data_fingerprint: str
    source_artifact_sha256: str
    source_tensor_index: tuple[str, str, str, int]
    source_direction_sha256: str
    reference_rms: float
    layer: int
    site: ActivationSite
    token_scope: TokenScope
    alpha: float
    screen_points: tuple[CoordinateScreenPoint, ...]
    screen_records: tuple[GenerationRecord, ...]
    screen_contract_digest: str
    screen_record_set_digest: str
    screen_runtime_artifact_sha256: str
    screen_execution_public_key: str
    screen_question_bundle_sha256: str
    schema_version: int = 7

    def __post_init__(self) -> None:
        if self.schema_version != 7:
            raise DataValidationError("unsupported coordinate-sparse artifact schema")
        if self.feature_schema.partition != "T-steer":
            raise DataValidationError("coordinate-sparse feature selection must use T-steer")
        if self.feature_schema.width != self.sparse_direction.direction.numel():
            raise DataValidationError("coordinate-sparse schema width differs from direction")
        if not _SHA256.fullmatch(self.data_fingerprint):
            raise DataValidationError("coordinate-sparse artifact requires a data fingerprint")
        if (
            not _SHA256.fullmatch(self.source_artifact_sha256)
            or type(self.source_tensor_index) is not tuple
            or len(self.source_tensor_index) != 4
            or type(self.source_tensor_index[0]) is not str
            or type(self.source_tensor_index[1]) is not str
            or type(self.source_tensor_index[2]) is not str
            or type(self.source_tensor_index[3]) is not int
            or not _SHA256.fullmatch(self.source_direction_sha256)
            or not _SHA256.fullmatch(self.screen_runtime_artifact_sha256)
            or not _SHA256.fullmatch(self.screen_execution_public_key)
            or not _SHA256.fullmatch(self.screen_question_bundle_sha256)
            or isinstance(self.reference_rms, bool)
            or not isinstance(self.reference_rms, int | float)
            or not math.isfinite(float(self.reference_rms))
            or float(self.reference_rms) <= 0
            or type(self.layer) is not int
            or self.layer < 0
            or not isinstance(self.site, ActivationSite)
            or not isinstance(self.token_scope, TokenScope)
            or type(self.alpha) is not float
            or not math.isfinite(self.alpha)
            or self.alpha not in {0.1, 0.25, 0.5, 1.0, 2.0}
        ):
            raise DataValidationError(
                "coordinate-sparse artifact lacks registered intervention geometry"
            )
        selected = select_coordinate_screen_point(self.screen_points)
        expected_contract = coordinate_screen_contract_digest(
            feature_schema=self.feature_schema,
            source_artifact_sha256=self.source_artifact_sha256,
            source_tensor_index=self.source_tensor_index,
            source_direction_sha256=self.source_direction_sha256,
            layer=self.layer,
            site=self.site,
            token_scope=self.token_scope,
            runtime_artifact_sha256=self.screen_runtime_artifact_sha256,
            execution_public_key=self.screen_execution_public_key,
            points=self.screen_points,
            source_question_bundle_sha256=self.screen_question_bundle_sha256,
        )
        expected_baseline_id = coordinate_screen_condition_id(expected_contract)
        records = tuple(self.screen_records)
        record_index: dict[tuple[str, str], GenerationRecord] = {}
        for record in records:
            key = (record.condition_id, record.question_id)
            if key in record_index:
                raise DataValidationError("coordinate screen repeats an execution record")
            _validate_coordinate_screen_execution_record(
                record,
                contract_digest=expected_contract,
                runtime_artifact_sha256=self.screen_runtime_artifact_sha256,
                execution_public_key=self.screen_execution_public_key,
                prompt_template_sha256=self.feature_schema.prompt_sha256,
            )
            record_index[key] = record
        expected_keys: set[tuple[str, str]] = set()
        for point in self.screen_points:
            expected_intervention_id = coordinate_screen_condition_id(
                expected_contract,
                retained_fraction=point.retained_fraction,
                alpha=point.alpha,
            )
            if (
                point.baseline_condition_id != expected_baseline_id
                or point.intervention_condition_id != expected_intervention_id
            ):
                raise DataValidationError(
                    "coordinate screen condition IDs differ from the frozen contract"
                )
            for question_id, baseline, intervention in zip(
                point.question_ids,
                point.baseline_outcomes,
                point.intervention_outcomes,
                strict=True,
            ):
                expected_keys.add((point.baseline_condition_id, question_id))
                expected_keys.add((point.intervention_condition_id, question_id))
                baseline_record = record_index.get(
                    (point.baseline_condition_id, question_id)
                )
                intervention_record = record_index.get(
                    (point.intervention_condition_id, question_id)
                )
                if (
                    baseline_record is None
                    or intervention_record is None
                    or baseline_record.steering_method != "M0"
                    or baseline_record.outcome is not baseline
                    or baseline_record.layer is not None
                    or baseline_record.site is not None
                    or baseline_record.token_scope is not None
                    or baseline_record.alpha != 0
                    or baseline_record.sparsity is not None
                    or intervention_record.steering_method != "M4a"
                    or intervention_record.outcome is not intervention
                    or baseline_record.model_repository
                    != self.feature_schema.model_repository
                    or intervention_record.model_repository
                    != self.feature_schema.model_repository
                    or baseline_record.model_revision != self.feature_schema.model_revision
                    or intervention_record.model_revision
                    != self.feature_schema.model_revision
                    or baseline_record.system_prompt_id != self.feature_schema.prompt_id
                    or intervention_record.system_prompt_id
                    != self.feature_schema.prompt_id
                    or baseline_record.benchmark != self.feature_schema.benchmark
                    or intervention_record.benchmark != self.feature_schema.benchmark
                    or intervention_record.layer != self.layer
                    or intervention_record.site is not self.site
                    or intervention_record.token_scope is not self.token_scope
                    or intervention_record.alpha != point.alpha
                    or intervention_record.sparsity != point.retained_fraction
                    or not isinstance(
                        intervention_record.metadata.get("intervention_trace"), Mapping
                    )
                    or intervention_record.metadata["intervention_trace"].get(
                        "standardized_alpha"
                    )
                    != point.alpha
                    or intervention_record.metadata["intervention_trace"].get(
                        "retained_fraction"
                    )
                    != point.retained_fraction
                    or intervention_record.metadata["intervention_trace"].get("layer")
                    != self.layer
                    or intervention_record.metadata["intervention_trace"].get("site")
                    != self.site.value
                    or intervention_record.metadata["intervention_trace"].get(
                        "token_scope"
                    )
                    != self.token_scope.value
                    or intervention_record.metadata["intervention_trace"].get(
                        "source_artifact_sha256"
                    )
                    != self.source_artifact_sha256
                    or intervention_record.metadata["intervention_trace"].get(
                        "reference_rms"
                    )
                    != self.reference_rms
                    or baseline_record.metadata.get("coordinate_screen_contract_digest")
                    != expected_contract
                    or intervention_record.metadata.get(
                        "coordinate_screen_contract_digest"
                    )
                    != expected_contract
                ):
                    raise DataValidationError(
                        "coordinate screen outcomes differ from execution records"
                    )
        if (
            selected.retained_fraction != self.sparse_direction.retained_fraction
            or selected.alpha != self.alpha
            or set(record_index) != expected_keys
            or self.screen_contract_digest != expected_contract
            or self.screen_record_set_digest
            != _coordinate_screen_record_set_digest(records)
        ):
            raise DataValidationError(
                "coordinate-sparse artifact differs from its deterministic screen winner"
            )


def fit_coordinate_sparse_artifact(
    dataset: ProbeDataset,
    dense_direction: Tensor,
    *,
    screen_points: Sequence[CoordinateScreenPoint],
    screen_records: Sequence[GenerationRecord],
    source_artifact_sha256: str,
    source_tensor_index: tuple[str, str, str, int],
    source_direction_sha256: str,
    reference_rms: float,
    layer: int,
    site: ActivationSite,
    token_scope: TokenScope,
    screen_runtime_artifact_sha256: str,
    screen_execution_public_key: str,
    screen_question_bundle_sha256: str | None = None,
    renormalize: bool = False,
) -> CoordinateSparseArtifact:
    if dataset.feature_schema is None or dataset.feature_schema.partition != "T-steer":
        raise DataValidationError("coordinate-sparse fitting requires a bound T-steer dataset")
    correct = torch.tensor([outcome is Outcome.CORRECT for outcome in dataset.outcomes])
    incorrect = torch.tensor([outcome is Outcome.INCORRECT for outcome in dataset.outcomes])
    if not correct.any() or not incorrect.any():
        raise DataValidationError("coordinate-sparse fitting requires C and I outcomes")
    effect = standardized_effect_size(dataset.features[correct], dataset.features[incorrect])
    frozen_screen = tuple(screen_points)
    selected = select_coordinate_screen_point(frozen_screen)
    question_bundle_sha = (
        dataset.feature_schema.split_manifest_digest
        if screen_question_bundle_sha256 is None
        else screen_question_bundle_sha256
    )
    sparse = coordinate_sparse_direction(
        dense_direction,
        effect,
        retained_fraction=selected.retained_fraction,
        renormalize=renormalize,
    )
    return CoordinateSparseArtifact(
        sparse_direction=sparse,
        feature_schema=dataset.feature_schema,
        data_fingerprint=dataset.data_fingerprint,
        source_artifact_sha256=source_artifact_sha256,
        source_tensor_index=source_tensor_index,
        source_direction_sha256=source_direction_sha256,
        reference_rms=float(reference_rms),
        layer=layer,
        site=site,
        token_scope=token_scope,
        alpha=float(selected.alpha),
        screen_points=frozen_screen,
        screen_records=tuple(screen_records),
        screen_contract_digest=coordinate_screen_contract_digest(
            feature_schema=dataset.feature_schema,
            source_artifact_sha256=source_artifact_sha256,
            source_tensor_index=source_tensor_index,
            source_direction_sha256=source_direction_sha256,
            layer=layer,
            site=site,
            token_scope=token_scope,
            runtime_artifact_sha256=screen_runtime_artifact_sha256,
            execution_public_key=screen_execution_public_key,
            points=frozen_screen,
            source_question_bundle_sha256=question_bundle_sha,
        ),
        screen_record_set_digest=_coordinate_screen_record_set_digest(
            tuple(screen_records)
        ),
        screen_runtime_artifact_sha256=screen_runtime_artifact_sha256,
        screen_execution_public_key=screen_execution_public_key,
        screen_question_bundle_sha256=question_bundle_sha,
    )


def save_coordinate_sparse_artifact(
    directory: str | Path, artifact: CoordinateSparseArtifact
) -> None:
    destination = validate_active_study_artifact_paths(
        {"E7 coordinate-sparse artifact": directory}
    )["E7 coordinate-sparse artifact"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FrozenArtifactError(
            f"refusing to overwrite coordinate-sparse artifact: {destination}"
        )
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        tensor_path = stage / "coordinate.safetensors"
        save_file(
            {
                "direction": artifact.sparse_direction.direction,
                "effect_size": artifact.sparse_direction.effect_size,
                "mask": artifact.sparse_direction.mask.to(torch.uint8),
            },
            tensor_path,
        )
        screen_path = stage / "screen.json"
        screen_path.write_text(
            json.dumps(
                [point.to_dict() for point in artifact.screen_points],
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        records_path = stage / "screen-records.jsonl"
        records_path.write_text(
            "".join(
                json.dumps(record.to_dict(), sort_keys=True, allow_nan=False) + "\n"
                for record in artifact.screen_records
            ),
            encoding="utf-8",
        )
        metadata_body = {
            "schema_version": artifact.schema_version,
            "feature_schema": artifact.feature_schema.to_dict(),
            "data_fingerprint": artifact.data_fingerprint,
            "source_artifact_sha256": artifact.source_artifact_sha256,
            "source_tensor_index": list(artifact.source_tensor_index),
            "source_direction_sha256": artifact.source_direction_sha256,
            "reference_rms": artifact.reference_rms,
            "layer": artifact.layer,
            "site": artifact.site.value,
            "token_scope": artifact.token_scope.value,
            "alpha": artifact.alpha,
            "screen_sha256": sha256_file(screen_path),
            "screen_records_sha256": sha256_file(records_path),
            "screen_contract_digest": artifact.screen_contract_digest,
            "screen_record_set_digest": artifact.screen_record_set_digest,
            "screen_runtime_artifact_sha256": (
                artifact.screen_runtime_artifact_sha256
            ),
            "screen_execution_public_key": artifact.screen_execution_public_key,
            "screen_question_bundle_sha256": (
                artifact.screen_question_bundle_sha256
            ),
            "retained_fraction": artifact.sparse_direction.retained_fraction,
            "renormalized": artifact.sparse_direction.renormalized,
            "tensor_sha256": sha256_file(tensor_path),
        }
        metadata = {**metadata_body, "metadata_digest": stable_hash(metadata_body)}
        (stage / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def load_coordinate_sparse_artifact(directory: str | Path) -> CoordinateSparseArtifact:
    source = Path(directory)
    if (
        source.is_symlink()
        or not source.is_dir()
        or {path.name for path in source.iterdir()}
        != {
            "metadata.json",
            "coordinate.safetensors",
            "screen.json",
            "screen-records.jsonl",
        }
        or any(path.is_symlink() or not path.is_file() for path in source.iterdir())
    ):
        raise FrozenArtifactError("coordinate-sparse artifact inventory differs")
    try:
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read coordinate-sparse metadata: {exc}") from exc
    digest = metadata.pop("metadata_digest", None)
    if digest != stable_hash(metadata):
        raise FrozenArtifactError("coordinate-sparse metadata digest mismatch")
    tensor_path = source / "coordinate.safetensors"
    if sha256_file(tensor_path) != metadata.get("tensor_sha256"):
        raise FrozenArtifactError("coordinate-sparse tensor checksum mismatch")
    try:
        screen_path = source / "screen.json"
        if sha256_file(screen_path) != metadata.get("screen_sha256"):
            raise FrozenArtifactError("coordinate screen checksum mismatch")
        screen_value = json.loads(screen_path.read_text(encoding="utf-8"))
        if not isinstance(screen_value, list):
            raise FrozenArtifactError("coordinate screen must be a list")
        records_path = source / "screen-records.jsonl"
        if sha256_file(records_path) != metadata.get("screen_records_sha256"):
            raise FrozenArtifactError("coordinate screen-record checksum mismatch")
        screen_records = tuple(
            GenerationRecord.from_dict(json.loads(line))
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        tensors = load_file(tensor_path, device="cpu")
        if set(tensors) != {"direction", "effect_size", "mask"}:
            raise FrozenArtifactError("unexpected coordinate-sparse tensors")
        sparse = CoordinateSparseDirection(
            direction=tensors["direction"],
            mask=tensors["mask"].bool(),
            effect_size=tensors["effect_size"],
            retained_fraction=float(metadata["retained_fraction"]),
            renormalized=bool(metadata["renormalized"]),
        )
        return CoordinateSparseArtifact(
            sparse_direction=sparse,
            feature_schema=ActivationFeatureSchema.from_dict(metadata["feature_schema"]),
            data_fingerprint=str(metadata["data_fingerprint"]),
            source_artifact_sha256=(
                str(metadata["source_artifact_sha256"])
            ),
            source_tensor_index=(
                str(metadata["source_tensor_index"][0]),
                str(metadata["source_tensor_index"][1]),
                str(metadata["source_tensor_index"][2]),
                int(metadata["source_tensor_index"][3]),
            ),
            source_direction_sha256=str(metadata["source_direction_sha256"]),
            reference_rms=float(metadata["reference_rms"]),
            layer=int(metadata["layer"]),
            site=ActivationSite(metadata["site"]),
            token_scope=TokenScope(metadata["token_scope"]),
            alpha=float(metadata["alpha"]),
            screen_points=tuple(
                CoordinateScreenPoint.from_dict(item) for item in screen_value
            ),
            screen_records=screen_records,
            screen_contract_digest=str(metadata["screen_contract_digest"]),
            screen_record_set_digest=str(metadata["screen_record_set_digest"]),
            screen_runtime_artifact_sha256=str(
                metadata["screen_runtime_artifact_sha256"]
            ),
            screen_execution_public_key=str(metadata["screen_execution_public_key"]),
            screen_question_bundle_sha256=str(
                metadata["screen_question_bundle_sha256"]
            ),
            schema_version=int(metadata["schema_version"]),
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
        raise FrozenArtifactError(f"invalid coordinate-sparse artifact: {exc}") from exc


class SAESparsity(StrEnum):
    TOP_K = "top_k"
    L1 = "l1"


@dataclass(frozen=True, slots=True)
class SAEConfig:
    input_width: int
    expansion_factor: int = 8
    latent_width: int | None = None
    sparsity: SAESparsity = SAESparsity.TOP_K
    top_k: int = 32
    l1_coefficient: float = 1e-3
    epochs: int = 100
    batch_size: int = 512
    learning_rate: float = 1e-3
    seed: int = 17

    def __post_init__(self) -> None:
        if self.input_width <= 0 or self.expansion_factor <= 0:
            raise DataValidationError("SAE widths and expansion factor must be positive")
        if self.latent_width is not None and self.latent_width <= 0:
            raise DataValidationError("SAE latent width must be positive")
        if not 1 <= self.top_k <= self.resolved_latent_width:
            raise DataValidationError("SAE top_k must be within the latent width")
        if self.l1_coefficient < 0 or not math.isfinite(self.l1_coefficient):
            raise DataValidationError("SAE L1 coefficient must be finite and non-negative")
        if self.epochs <= 0 or self.batch_size <= 0 or self.learning_rate <= 0:
            raise DataValidationError("SAE optimizer parameters must be positive")

    @property
    def resolved_latent_width(self) -> int:
        return self.latent_width or self.input_width * self.expansion_factor


class SparseAutoencoder(nn.Module):
    def __init__(self, config: SAEConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Linear(config.input_width, config.resolved_latent_width)
        self.decoder = nn.Linear(config.resolved_latent_width, config.input_width)
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        with torch.no_grad():
            self.encoder.weight.copy_(
                torch.randn(self.encoder.weight.shape, generator=generator)
                / math.sqrt(config.input_width)
            )
            self.encoder.bias.zero_()
            self.decoder.weight.copy_(
                torch.randn(self.decoder.weight.shape, generator=generator)
                / math.sqrt(config.resolved_latent_width)
            )
            self.decoder.bias.zero_()
            self.normalize_decoder()

    def encode(self, activations: Tensor) -> Tensor:
        values = activations.float()
        if values.shape[-1] != self.config.input_width:
            raise DataValidationError("SAE input width does not match its configuration")
        latents = F.relu(self.encoder(values))
        if self.config.sparsity is SAESparsity.TOP_K:
            _, indices = torch.topk(latents, self.config.top_k, dim=-1)
            selected = torch.zeros_like(latents, dtype=torch.bool)
            selected.scatter_(-1, indices, True)
            latents = latents * selected
        return latents

    def decode(self, latents: Tensor) -> Tensor:
        if latents.shape[-1] != self.config.resolved_latent_width:
            raise DataValidationError("SAE latent width does not match its configuration")
        return cast(Tensor, self.decoder(latents.float()))

    def forward(self, activations: Tensor) -> tuple[Tensor, Tensor]:
        latents = self.encode(activations)
        return self.decode(latents), latents

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """Unit-normalize decoder columns while preserving the represented function."""

        norms = torch.linalg.vector_norm(self.decoder.weight, dim=0).clamp_min(1e-8)
        self.decoder.weight.div_(norms.unsqueeze(0))
        self.encoder.weight.mul_(norms.unsqueeze(1))
        self.encoder.bias.mul_(norms)


@dataclass(frozen=True, slots=True)
class SAEMetrics:
    reconstruction_mse: float
    fraction_variance_explained: float
    average_active_features: float

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(value)
            for value in (
                self.reconstruction_mse,
                self.fraction_variance_explained,
                self.average_active_features,
            )
        ):
            raise DataValidationError("SAE metrics contain NaN or infinity")
        if self.reconstruction_mse < 0 or self.average_active_features < 0:
            raise DataValidationError("SAE reconstruction and activity metrics cannot be negative")


@torch.no_grad()
def evaluate_sae(model: SparseAutoencoder, activations: Tensor) -> SAEMetrics:
    values = _matrix(activations, width=model.config.input_width)
    reconstruction, latents = model(values)
    residual_sum = (values - reconstruction).pow(2).sum()
    total_sum = (values - values.mean(dim=0)).pow(2).sum()
    if float(total_sum) <= 0:
        raise DataValidationError("SAE FVE requires non-constant activations")
    return SAEMetrics(
        reconstruction_mse=float(F.mse_loss(reconstruction, values)),
        fraction_variance_explained=float(1 - residual_sum / total_sum),
        average_active_features=float((latents != 0).sum(dim=1).float().mean()),
    )


def _activation_fingerprint(values: Tensor, schema: ActivationFeatureSchema) -> str:
    digest = hashlib.sha256()
    digest.update(str(tuple(values.shape)).encode())
    digest.update(values.detach().cpu().float().contiguous().numpy().tobytes(order="C"))
    digest.update(schema.digest.encode())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SAETrainingResult:
    model: SparseAutoencoder
    config: SAEConfig
    metrics: SAEMetrics
    loss_history: tuple[float, ...]
    training_fingerprint: str
    validation_fingerprint: str
    training_schema: ActivationFeatureSchema
    validation_schema: ActivationFeatureSchema
    training_rows: int
    validation_rows: int
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2 or self.model.config != self.config:
            raise DataValidationError("invalid SAE training-result schema or configuration")
        if self.training_rows <= 0 or self.validation_rows <= 0:
            raise DataValidationError("SAE training and validation row counts must be positive")
        if not self.loss_history or any(
            not math.isfinite(value) or value < 0 for value in self.loss_history
        ):
            raise DataValidationError("SAE loss history is empty or invalid")
        if not _SHA256.fullmatch(self.training_fingerprint) or not _SHA256.fullmatch(
            self.validation_fingerprint
        ):
            raise DataValidationError("SAE split fingerprints must be SHA-256 digests")
        if self.training_fingerprint == self.validation_fingerprint:
            raise DataValidationError("SAE training and validation fingerprints must differ")
        if self.training_schema.partition != "sae-train":
            raise DataValidationError("SAE training schema must use sae-train")
        if self.validation_schema.partition != "sae-validation":
            raise DataValidationError("SAE validation schema must use sae-validation")
        if not self.training_schema.is_compatible_extraction(self.validation_schema):
            raise DataValidationError("SAE training and validation extraction schemas differ")
        if self.training_schema.width != self.config.input_width:
            raise DataValidationError("SAE feature schema width differs from its configuration")


def fit_sparse_autoencoder(
    training_activations: Tensor,
    validation_activations: Tensor,
    config: SAEConfig,
    *,
    training_schema: ActivationFeatureSchema,
    validation_schema: ActivationFeatureSchema,
    training_fingerprint: str | None = None,
    validation_fingerprint: str | None = None,
) -> SAETrainingResult:
    values = _matrix(training_activations, width=config.input_width)
    validation_values = _matrix(validation_activations, width=config.input_width)
    if (
        training_schema.partition != "sae-train"
        or validation_schema.partition != "sae-validation"
        or not training_schema.is_compatible_extraction(validation_schema)
        or training_schema.width != config.input_width
    ):
        raise DataValidationError("SAE train/validation feature schemas are incompatible")
    computed_training_fingerprint = _activation_fingerprint(values, training_schema)
    computed_validation_fingerprint = _activation_fingerprint(validation_values, validation_schema)
    if training_fingerprint and training_fingerprint != computed_training_fingerprint:
        raise DataValidationError("provided SAE training fingerprint does not match its tensor")
    if validation_fingerprint and validation_fingerprint != computed_validation_fingerprint:
        raise DataValidationError("provided SAE validation fingerprint does not match its tensor")
    fingerprint = computed_training_fingerprint
    held_out_fingerprint = computed_validation_fingerprint
    if not _SHA256.fullmatch(fingerprint) or not _SHA256.fullmatch(held_out_fingerprint):
        raise DataValidationError("SAE split fingerprints must be SHA-256 digests")
    model = SparseAutoencoder(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    history: list[float] = []
    for _ in range(config.epochs):
        permutation = torch.randperm(values.shape[0], generator=generator)
        epoch_loss = 0.0
        rows_seen = 0
        for start in range(0, values.shape[0], config.batch_size):
            batch = values[permutation[start : start + config.batch_size]]
            optimizer.zero_grad(set_to_none=True)
            reconstruction, latents = model(batch)
            loss = F.mse_loss(reconstruction, batch)
            if config.sparsity is SAESparsity.L1:
                loss = loss + config.l1_coefficient * latents.abs().mean()
            if not torch.isfinite(loss):
                raise DataValidationError("SAE training diverged")
            torch.autograd.backward(loss)
            optimizer.step()
            model.normalize_decoder()
            epoch_loss += float(loss.detach()) * batch.shape[0]
            rows_seen += batch.shape[0]
        history.append(epoch_loss / rows_seen)
    model.eval()
    return SAETrainingResult(
        model=model,
        config=config,
        metrics=evaluate_sae(model, validation_values),
        loss_history=tuple(history),
        training_fingerprint=fingerprint,
        validation_fingerprint=held_out_fingerprint,
        training_schema=training_schema,
        validation_schema=validation_schema,
        training_rows=int(values.shape[0]),
        validation_rows=int(validation_values.shape[0]),
    )


@dataclass(frozen=True, slots=True)
class SAELatentDirection:
    direction: Tensor
    selected_features: tuple[int, ...]
    correct_count: int
    incorrect_count: int
    selection_fingerprint: str
    selection_schema: ActivationFeatureSchema

    def __post_init__(self) -> None:
        direction = self.direction.detach().cpu().float().contiguous().clone()
        if direction.ndim != 1 or direction.numel() == 0 or not torch.isfinite(direction).all():
            raise DataValidationError("SAE latent direction is invalid")
        if (
            len(set(self.selected_features)) != len(self.selected_features)
            or not self.selected_features
        ):
            raise DataValidationError("SAE selected features must be non-empty and unique")
        if any(index < 0 or index >= direction.numel() for index in self.selected_features):
            raise DataValidationError("SAE selected feature is out of range")
        if self.correct_count <= 0 or self.incorrect_count <= 0:
            raise DataValidationError("SAE direction requires correct and incorrect examples")
        if not _SHA256.fullmatch(self.selection_fingerprint):
            raise DataValidationError("SAE selection fingerprint must be SHA-256")
        if self.selection_schema.partition != "T-steer":
            raise DataValidationError("SAE latent feature selection must use T-steer")
        if self.selection_schema.width <= 0:
            raise DataValidationError("SAE feature-selection schema is invalid")
        selected = torch.zeros_like(direction, dtype=torch.bool)
        selected[list(self.selected_features)] = True
        if (direction[~selected] != 0).any():
            raise DataValidationError("SAE latent direction is nonzero outside selected features")
        object.__setattr__(self, "direction", direction)


@torch.no_grad()
def latent_factuality_direction(
    model: SparseAutoencoder,
    dataset: ProbeDataset,
    *,
    feature_count: int,
) -> SAELatentDirection:
    values = _matrix(dataset.features, width=model.config.input_width)
    if dataset.feature_schema is None or dataset.feature_schema.partition != "T-steer":
        raise DataValidationError("SAE feature selection requires a bound T-steer dataset")
    if not 1 <= feature_count <= model.config.resolved_latent_width:
        raise DataValidationError("SAE feature count is outside the latent width")
    correct = torch.tensor([Outcome(value) is Outcome.CORRECT for value in dataset.outcomes])
    incorrect = torch.tensor([Outcome(value) is Outcome.INCORRECT for value in dataset.outcomes])
    if not correct.any() or not incorrect.any():
        raise DataValidationError("SAE direction requires correct and incorrect outcomes")
    latents = model.encode(values)
    dense = latents[correct].mean(dim=0) - latents[incorrect].mean(dim=0)
    order = torch.argsort(dense.abs(), descending=True, stable=True)
    selected = tuple(int(value) for value in order[:feature_count])
    sparse = torch.zeros_like(dense)
    sparse[list(selected)] = dense[list(selected)]
    if float(torch.linalg.vector_norm(sparse)) <= 0:
        raise DataValidationError("selected SAE features have a zero factuality direction")
    return SAELatentDirection(
        sparse,
        selected,
        int(correct.sum()),
        int(incorrect.sum()),
        dataset.data_fingerprint,
        dataset.feature_schema,
    )


@torch.no_grad()
def decode_latent_direction(
    model: SparseAutoencoder,
    latent_direction: SAELatentDirection | Tensor,
    *,
    normalize: bool = True,
) -> Tensor:
    direction = (
        latent_direction.direction
        if isinstance(latent_direction, SAELatentDirection)
        else latent_direction.detach().cpu().float()
    )
    if direction.shape != (model.config.resolved_latent_width,):
        raise DataValidationError("latent direction width differs from SAE")
    decoded = model.decoder.weight.detach().cpu() @ direction
    norm = torch.linalg.vector_norm(decoded)
    if not torch.isfinite(norm) or float(norm) <= 0:
        raise DataValidationError("decoded SAE direction is zero or invalid")
    return decoded / norm if normalize else decoded


@torch.no_grad()
def decoder_feature_direction(
    model: SparseAutoencoder, feature_index: int, *, normalize: bool = True
) -> Tensor:
    if not 0 <= feature_index < model.config.resolved_latent_width:
        raise DataValidationError("SAE feature index is out of range")
    direction = model.decoder.weight[:, feature_index].detach().cpu().float()
    norm = torch.linalg.vector_norm(direction)
    if float(norm) <= 0:
        raise DataValidationError("SAE decoder feature has zero norm")
    return direction / norm if normalize else direction


def suppress_latent_features(latents: Tensor, feature_indices: Sequence[int]) -> Tensor:
    result = latents.clone()
    if result.ndim < 2:
        raise DataValidationError("SAE latents must have at least two dimensions")
    indices = tuple(dict.fromkeys(int(index) for index in feature_indices))
    if any(index < 0 or index >= result.shape[-1] for index in indices):
        raise DataValidationError("SAE feature suppression index is out of range")
    result[..., list(indices)] = 0
    return result


def sae_checkpoint_fingerprint(training: SAETrainingResult) -> str:
    """Content fingerprint for a trained SAE, independent of its filesystem path."""

    digest = hashlib.sha256()
    digest.update(
        stable_hash(
            {
                "config": _config_dict(training.config),
                "training_fingerprint": training.training_fingerprint,
                "validation_fingerprint": training.validation_fingerprint,
                "training_schema": training.training_schema.to_dict(),
                "validation_schema": training.validation_schema.to_dict(),
            }
        ).encode()
    )
    for name, tensor in sorted(training.model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SeedFeatureSelection:
    """Top features selected by one independently trained, frozen SAE."""

    seed: int
    checkpoint_fingerprint: str
    selected_features: tuple[int, ...]

    def __post_init__(self) -> None:
        features = tuple(int(value) for value in self.selected_features)
        if self.seed < 0:
            raise DataValidationError("SAE stability seed cannot be negative")
        if not _SHA256.fullmatch(self.checkpoint_fingerprint):
            raise DataValidationError("SAE stability checkpoint must have a SHA-256 fingerprint")
        if not features or any(value < 0 for value in features):
            raise DataValidationError("SAE stability features must be non-empty and non-negative")
        if len(set(features)) != len(features):
            raise DataValidationError("SAE stability features must be unique")
        object.__setattr__(self, "selected_features", features)


def selected_feature_stability(selections: Sequence[SeedFeatureSelection]) -> float:
    frozen = tuple(selections)
    if len(frozen) < 2:
        raise DataValidationError("feature stability requires at least two frozen SAE selections")
    if len({item.seed for item in frozen}) != len(frozen):
        raise DataValidationError("feature stability requires distinct SAE training seeds")
    if len({item.checkpoint_fingerprint for item in frozen}) != len(frozen):
        raise DataValidationError("feature stability requires distinct SAE checkpoints")
    if len({len(item.selected_features) for item in frozen}) != 1:
        raise DataValidationError("feature stability selections must use the same feature count")
    sets = [set(item.selected_features) for item in frozen]
    similarities = [
        len(left & right) / len(left | right)
        for index, left in enumerate(sets)
        for right in sets[index + 1 :]
    ]
    return sum(similarities) / len(similarities)


@dataclass(frozen=True, slots=True)
class CausalEvidenceSpec:
    """Frozen identity of the paired runs underlying one causal feature audit."""

    paired_question_fingerprint: str
    baseline_run_fingerprint: str
    activated_run_fingerprint: str
    suppressed_run_fingerprint: str
    factuality_sample_count: int
    protected_sample_counts: Mapping[str, int]
    alpha: float
    token_scope: TokenScope
    layer: int
    site: ActivationSite
    feature_schema: ActivationFeatureSchema
    runtime_artifact_sha256: str | None = None
    execution_public_key: str | None = None
    source_question_bundle_sha256: str | None = None

    def __post_init__(self) -> None:
        fingerprints = (
            self.paired_question_fingerprint,
            self.baseline_run_fingerprint,
            self.activated_run_fingerprint,
            self.suppressed_run_fingerprint,
        )
        if any(not _SHA256.fullmatch(value) for value in fingerprints):
            raise DataValidationError("causal-evidence identities must be SHA-256 fingerprints")
        if len(set(fingerprints[1:])) != 3:
            raise DataValidationError("causal evidence requires three distinct intervention runs")
        if self.factuality_sample_count <= 0:
            raise DataValidationError("causal evidence requires factuality samples")
        protected = {
            str(key).strip(): int(value) for key, value in self.protected_sample_counts.items()
        }
        if not protected or any(not key or value <= 0 for key, value in protected.items()):
            raise DataValidationError("causal protected-sample counts must be named and positive")
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise DataValidationError("causal intervention alpha must be finite and positive")
        if self.layer < 0:
            raise DataValidationError("causal intervention layer cannot be negative")
        provenance = (
            self.runtime_artifact_sha256,
            self.execution_public_key,
            self.source_question_bundle_sha256,
        )
        if any(value is not None for value in provenance) and not all(
            type(value) is str and _SHA256.fullmatch(value) is not None
            for value in provenance
        ):
            raise DataValidationError(
                "causal evidence requires one complete runtime provenance identity"
            )
        if all(value is not None for value in provenance) and (
            self.factuality_sample_count < _MIN_NATIVE_CAUSAL_FACTUALITY_SAMPLES
            or any(
                value < _MIN_NATIVE_CAUSAL_PROTECTED_SAMPLES
                for value in protected.values()
            )
        ):
            raise DataValidationError(
                "native causal evidence lacks the registered minimum sample sizes"
            )
        object.__setattr__(self, "protected_sample_counts", MappingProxyType(protected))

    @property
    def digest(self) -> str:
        return stable_hash(
            {
                "paired_question_fingerprint": self.paired_question_fingerprint,
                "baseline_run_fingerprint": self.baseline_run_fingerprint,
                "activated_run_fingerprint": self.activated_run_fingerprint,
                "suppressed_run_fingerprint": self.suppressed_run_fingerprint,
                "factuality_sample_count": self.factuality_sample_count,
                "protected_sample_counts": dict(self.protected_sample_counts),
                "alpha": self.alpha,
                "token_scope": self.token_scope.value,
                "layer": self.layer,
                "site": self.site.value,
                "feature_schema": self.feature_schema.to_dict(),
                "runtime_artifact_sha256": self.runtime_artifact_sha256,
                "execution_public_key": self.execution_public_key,
                "source_question_bundle_sha256": self.source_question_bundle_sha256,
            }
        )


@dataclass(frozen=True, slots=True)
class FeatureInterventionEvidence:
    feature_index: int
    activation_factuality_delta: float
    suppression_factuality_delta: float
    protected_behavior_deltas: Mapping[str, float]
    spec: CausalEvidenceSpec
    factuality_outcomes: Mapping[str, tuple[Outcome, Outcome, Outcome]]
    protected_outcomes: Mapping[str, Mapping[str, tuple[bool, bool, bool]]]
    execution_signature: str | None = None
    native_execution_records: Mapping[str, tuple[GenerationRecord, ...]] | None = None

    def __post_init__(self) -> None:
        if self.feature_index < 0:
            raise DataValidationError("causal-evidence feature index cannot be negative")
        if not math.isfinite(self.activation_factuality_delta) or not math.isfinite(
            self.suppression_factuality_delta
        ):
            raise DataValidationError("causal factuality deltas must be finite")
        protected = {
            str(key): float(value) for key, value in self.protected_behavior_deltas.items()
        }
        if any(not key.strip() or not math.isfinite(value) for key, value in protected.items()):
            raise DataValidationError("protected-behavior deltas must be named and finite")
        if set(protected) != set(self.spec.protected_sample_counts):
            raise DataValidationError("causal evidence and protected-sample behaviors differ")
        factuality: dict[str, tuple[Outcome, Outcome, Outcome]] = {}
        for question_id, raw_values in self.factuality_outcomes.items():
            values = tuple(raw_values)
            if not str(question_id).strip() or len(values) != 3:
                raise DataValidationError("causal factuality receipts are invalid")
            factuality[str(question_id)] = cast(
                tuple[Outcome, Outcome, Outcome],
                tuple(Outcome(value) for value in values),
            )
        protected_receipts: dict[
            str, MappingProxyType[str, tuple[bool, bool, bool]]
        ] = {}
        for behavior, raw_measurements in self.protected_outcomes.items():
            measurements: dict[str, tuple[bool, bool, bool]] = {}
            for question_id, raw_protected_values in raw_measurements.items():
                protected_values = tuple(raw_protected_values)
                if (
                    not str(question_id).strip()
                    or len(protected_values) != 3
                    or any(type(value) is not bool for value in protected_values)
                ):
                    raise DataValidationError("causal protected receipts are invalid")
                measurements[str(question_id)] = protected_values
            if not str(behavior).strip() or not measurements:
                raise DataValidationError("causal protected receipts are invalid")
            protected_receipts[str(behavior)] = MappingProxyType(measurements)
        if (
            not factuality
            or len(factuality) != self.spec.factuality_sample_count
            or set(protected_receipts) != set(self.spec.protected_sample_counts)
            or any(
                len(protected_receipts[name]) != count
                for name, count in self.spec.protected_sample_counts.items()
            )
        ):
            raise DataValidationError("causal receipts differ from their frozen sample counts")
        baseline = {key: values[0] for key, values in factuality.items()}
        activated = {key: values[1] for key, values in factuality.items()}
        suppressed = {key: values[2] for key, values in factuality.items()}
        protected_baseline = {
            name: {key: values[0] for key, values in measurements.items()}
            for name, measurements in protected_receipts.items()
        }
        protected_activated = {
            name: {key: values[1] for key, values in measurements.items()}
            for name, measurements in protected_receipts.items()
        }
        protected_suppressed = {
            name: {key: values[2] for key, values in measurements.items()}
            for name, measurements in protected_receipts.items()
        }
        expected_question_fingerprint = stable_hash(
            {
                "factuality": sorted(baseline),
                "protected": {
                    behavior: sorted(measurements)
                    for behavior, measurements in protected_baseline.items()
                },
            }
        )
        expected_run_fingerprints = (
            _causal_run_fingerprint(
                mode="baseline",
                outcomes=baseline,
                protected=protected_baseline,
                feature_index=self.feature_index,
                feature_schema=self.spec.feature_schema,
                alpha=self.spec.alpha,
                token_scope=self.spec.token_scope,
                layer=self.spec.layer,
                site=self.spec.site,
            ),
            _causal_run_fingerprint(
                mode="activated",
                outcomes=activated,
                protected=protected_activated,
                feature_index=self.feature_index,
                feature_schema=self.spec.feature_schema,
                alpha=self.spec.alpha,
                token_scope=self.spec.token_scope,
                layer=self.spec.layer,
                site=self.spec.site,
            ),
            _causal_run_fingerprint(
                mode="suppressed",
                outcomes=suppressed,
                protected=protected_suppressed,
                feature_index=self.feature_index,
                feature_schema=self.spec.feature_schema,
                alpha=self.spec.alpha,
                token_scope=self.spec.token_scope,
                layer=self.spec.layer,
                site=self.spec.site,
            ),
        )
        expected_protected = {
            name: max(
                (
                    _paired_boolean_delta(protected_baseline[name], protected_activated[name]),
                    _paired_boolean_delta(protected_baseline[name], protected_suppressed[name]),
                ),
                key=abs,
            )
            for name in protected_receipts
        }
        if (
            self.spec.paired_question_fingerprint != expected_question_fingerprint
            or (
                self.spec.baseline_run_fingerprint,
                self.spec.activated_run_fingerprint,
                self.spec.suppressed_run_fingerprint,
            )
            != expected_run_fingerprints
            or not math.isclose(
                self.activation_factuality_delta,
                _paired_correct_delta(baseline, activated),
                rel_tol=0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                self.suppression_factuality_delta,
                _paired_correct_delta(baseline, suppressed),
                rel_tol=0,
                abs_tol=1e-12,
            )
            or any(
                not math.isclose(protected[name], value, rel_tol=0, abs_tol=1e-12)
                for name, value in expected_protected.items()
            )
        ):
            raise DataValidationError("causal receipts differ from their summarized effects")
        object.__setattr__(self, "protected_behavior_deltas", MappingProxyType(protected))
        object.__setattr__(self, "factuality_outcomes", MappingProxyType(factuality))
        object.__setattr__(
            self,
            "protected_outcomes",
            MappingProxyType(protected_receipts),
        )
        native = {
            str(mode): tuple(records)
            for mode, records in (self.native_execution_records or {}).items()
        }
        if native:
            if set(native) != {"baseline", "activated", "suppressed"}:
                raise DataValidationError("causal native execution modes differ")
            expected_ids = set(factuality).union(
                *(
                    set(measurements)
                    for measurements in protected_receipts.values()
                )
            )
            if (
                any(not records for records in native.values())
                or any(
                    {record.question_id for record in records} != expected_ids
                    or len(records) != len(expected_ids)
                    for records in native.values()
                )
                or any(
                    record.steering_method != expected_method
                    for mode, records in native.items()
                    for record in records
                    for expected_method in (("M0",) if mode == "baseline" else ("M4b",))
                )
                or len(
                    {
                        next(iter(records)).condition_id
                        for records in native.values()
                    }
                )
                != 3
            ):
                raise DataValidationError("causal native execution rows differ")
            if self.spec.execution_public_key is None or self.spec.runtime_artifact_sha256 is None:
                raise DataValidationError("causal native rows lack runtime provenance")
            from mfh.experiments.e8_protected import validate_e8_execution_record

            for records in native.values():
                for record in records:
                    validate_e8_execution_record(
                        record,
                        condition_facts={
                            "steering_method": record.steering_method,
                            "method_artifact_sha256": record.metadata.get(
                                "method_artifact_sha256"
                            ),
                            "layer": record.layer,
                            "site": (
                                record.site.value if record.site is not None else None
                            ),
                            "token_scope": (
                                record.token_scope.value
                                if record.token_scope is not None
                                else None
                            ),
                            "alpha": record.alpha,
                            "sparsity": record.sparsity,
                            "prompt_template_sha256": record.metadata.get(
                                "prompt_template_sha256"
                            ),
                        },
                        execution_public_key=self.spec.execution_public_key,
                        runtime_artifact_sha256=self.spec.runtime_artifact_sha256,
                    )
            for question_id, values in factuality.items():
                observed = tuple(
                    next(
                        record.outcome
                        for record in native[mode]
                        if record.question_id == question_id
                    )
                    for mode in ("baseline", "activated", "suppressed")
                )
                if observed != values:
                    raise DataValidationError("causal native factual outcomes differ")
        object.__setattr__(
            self,
            "native_execution_records",
            MappingProxyType(native) if native else None,
        )
        if self.spec.execution_public_key is None:
            if self.execution_signature is not None:
                raise DataValidationError("causal evidence has an unanchored signature")
        else:
            if (
                type(self.execution_signature) is not str
                or re.fullmatch(r"[0-9a-f]{128}", self.execution_signature) is None
            ):
                raise DataValidationError("causal evidence lacks its runtime signature")
            try:
                Ed25519PublicKey.from_public_bytes(
                    bytes.fromhex(self.spec.execution_public_key)
                ).verify(
                    bytes.fromhex(self.execution_signature),
                    canonical_json(causal_evidence_execution_receipt_body(self)).encode(),
                )
            except (InvalidSignature, ValueError) as exc:
                raise DataValidationError(
                    "causal evidence runtime signature is invalid"
                ) from exc

    def causally_supported(self, *, minimum_effect: float, maximum_protected_effect: float) -> bool:
        if minimum_effect < 0 or maximum_protected_effect < 0:
            raise DataValidationError("causal evidence thresholds must be non-negative")
        causal = (
            self.activation_factuality_delta >= minimum_effect
            and self.suppression_factuality_delta <= -minimum_effect
        )
        protected = all(
            abs(value) <= maximum_protected_effect
            for value in self.protected_behavior_deltas.values()
        )
        return causal and protected


def causal_evidence_execution_receipt_body(
    evidence: FeatureInterventionEvidence,
) -> dict[str, Any]:
    """Canonical native-execution receipt for one paired feature intervention."""

    return {
        "receipt_kind": "e7-causal-feature-intervention-v1",
        "feature_index": evidence.feature_index,
        "activation_factuality_delta": evidence.activation_factuality_delta,
        "suppression_factuality_delta": evidence.suppression_factuality_delta,
        "protected_behavior_deltas": dict(evidence.protected_behavior_deltas),
        "spec_digest": evidence.spec.digest,
        "factuality_outcomes": {
            question_id: [outcome.value for outcome in outcomes]
            for question_id, outcomes in evidence.factuality_outcomes.items()
        },
        "protected_outcomes": {
            behavior: {
                question_id: list(outcomes)
                for question_id, outcomes in measurements.items()
            }
            for behavior, measurements in evidence.protected_outcomes.items()
        },
        "native_execution_records": {
            mode: [stable_hash(record.to_dict()) for record in records]
            for mode, records in (evidence.native_execution_records or {}).items()
        },
    }


def _paired_correct_delta(
    baseline: Mapping[str, Outcome], treatment: Mapping[str, Outcome]
) -> float:
    if baseline.keys() != treatment.keys() or not baseline:
        raise DataValidationError("causal intervention outcomes must use identical question IDs")
    usable = [key for key, value in baseline.items() if Outcome(value) is not Outcome.UNSCORABLE]
    if not usable or any(Outcome(treatment[key]) is Outcome.UNSCORABLE for key in usable):
        raise DataValidationError("causal intervention outcomes contain incompatible U labels")
    baseline_rate = sum(Outcome(baseline[key]) is Outcome.CORRECT for key in usable) / len(usable)
    treatment_rate = sum(Outcome(treatment[key]) is Outcome.CORRECT for key in usable) / len(usable)
    return treatment_rate - baseline_rate


def _paired_boolean_delta(baseline: Mapping[str, bool], treatment: Mapping[str, bool]) -> float:
    if baseline.keys() != treatment.keys() or not baseline:
        raise DataValidationError("protected intervention measurements must use identical IDs")
    return sum(bool(treatment[key]) - bool(baseline[key]) for key in baseline) / len(baseline)


def _causal_run_fingerprint(
    *,
    mode: str,
    outcomes: Mapping[str, Outcome],
    protected: Mapping[str, Mapping[str, bool]],
    feature_index: int,
    feature_schema: ActivationFeatureSchema,
    alpha: float,
    token_scope: TokenScope,
    layer: int,
    site: ActivationSite,
) -> str:
    return stable_hash(
        {
            "mode": mode,
            "feature_index": feature_index,
            "outcomes": {key: Outcome(value).value for key, value in outcomes.items()},
            "protected": {
                behavior: {key: bool(value) for key, value in measurements.items()}
                for behavior, measurements in protected.items()
            },
            "feature_schema": feature_schema.to_dict(),
            "alpha": alpha,
            "token_scope": token_scope.value,
            "layer": layer,
            "site": site.value,
        }
    )


def measure_feature_intervention_evidence(
    feature_index: int,
    *,
    baseline_outcomes: Mapping[str, Outcome],
    activated_outcomes: Mapping[str, Outcome],
    suppressed_outcomes: Mapping[str, Outcome],
    protected_baseline: Mapping[str, Mapping[str, bool]],
    protected_activated: Mapping[str, Mapping[str, bool]],
    protected_suppressed: Mapping[str, Mapping[str, bool]],
    feature_schema: ActivationFeatureSchema,
    alpha: float,
    token_scope: TokenScope,
    layer: int,
    site: ActivationSite,
    runtime_artifact_sha256: str | None = None,
    execution_public_key: str | None = None,
    source_question_bundle_sha256: str | None = None,
    execution_signer: Callable[[Mapping[str, Any]], str] | None = None,
    native_execution_records: Mapping[
        str, Sequence[GenerationRecord]
    ] | None = None,
) -> FeatureInterventionEvidence:
    """Measure paired activation/suppression effects from actual evaluation outcomes."""

    token_scope = TokenScope(token_scope)
    site = ActivationSite(site)
    if protected_baseline.keys() != protected_activated.keys() or protected_baseline.keys() != (
        protected_suppressed.keys()
    ):
        raise DataValidationError("protected behavior sets differ across interventions")
    protected_deltas: dict[str, float] = {}
    for behavior, baseline in protected_baseline.items():
        activated_delta = _paired_boolean_delta(baseline, protected_activated[behavior])
        suppressed_delta = _paired_boolean_delta(baseline, protected_suppressed[behavior])
        protected_deltas[behavior] = max(
            (activated_delta, suppressed_delta), key=lambda value: abs(value)
        )
    paired_question_fingerprint = stable_hash(
        {
            "factuality": sorted(baseline_outcomes),
            "protected": {
                behavior: sorted(measurements)
                for behavior, measurements in protected_baseline.items()
            },
        }
    )
    spec = CausalEvidenceSpec(
        paired_question_fingerprint=paired_question_fingerprint,
        baseline_run_fingerprint=_causal_run_fingerprint(
            mode="baseline",
            outcomes=baseline_outcomes,
            protected=protected_baseline,
            feature_index=feature_index,
            feature_schema=feature_schema,
            alpha=alpha,
            token_scope=token_scope,
            layer=layer,
            site=site,
        ),
        activated_run_fingerprint=_causal_run_fingerprint(
            mode="activated",
            outcomes=activated_outcomes,
            protected=protected_activated,
            feature_index=feature_index,
            feature_schema=feature_schema,
            alpha=alpha,
            token_scope=token_scope,
            layer=layer,
            site=site,
        ),
        suppressed_run_fingerprint=_causal_run_fingerprint(
            mode="suppressed",
            outcomes=suppressed_outcomes,
            protected=protected_suppressed,
            feature_index=feature_index,
            feature_schema=feature_schema,
            alpha=alpha,
            token_scope=token_scope,
            layer=layer,
            site=site,
        ),
        factuality_sample_count=len(baseline_outcomes),
        protected_sample_counts={
            behavior: len(measurements) for behavior, measurements in protected_baseline.items()
        },
        alpha=alpha,
        token_scope=token_scope,
        layer=layer,
        site=site,
        feature_schema=feature_schema,
        runtime_artifact_sha256=runtime_artifact_sha256,
        execution_public_key=execution_public_key,
        source_question_bundle_sha256=source_question_bundle_sha256,
    )
    evidence_kwargs: dict[str, Any] = {
        "feature_index": feature_index,
        "activation_factuality_delta": _paired_correct_delta(
            baseline_outcomes, activated_outcomes
        ),
        "suppression_factuality_delta": _paired_correct_delta(
            baseline_outcomes, suppressed_outcomes
        ),
        "protected_behavior_deltas": protected_deltas,
        "spec": spec,
        "factuality_outcomes": {
            question_id: (
                Outcome(baseline_outcomes[question_id]),
                Outcome(activated_outcomes[question_id]),
                Outcome(suppressed_outcomes[question_id]),
            )
            for question_id in baseline_outcomes
        },
        "protected_outcomes": {
            behavior: {
                question_id: (
                    bool(protected_baseline[behavior][question_id]),
                    bool(protected_activated[behavior][question_id]),
                    bool(protected_suppressed[behavior][question_id]),
                )
                for question_id in protected_baseline[behavior]
            }
            for behavior in protected_baseline
        },
        "native_execution_records": (
            {
                mode: tuple(records)
                for mode, records in native_execution_records.items()
            }
            if native_execution_records is not None
            else None
        ),
    }
    if execution_public_key is None:
        if execution_signer is not None:
            raise DataValidationError("causal execution signer lacks a public-key identity")
        return FeatureInterventionEvidence(**evidence_kwargs)
    if execution_signer is None:
        raise DataValidationError("causal evidence requires its native execution signer")
    unsigned = object.__new__(FeatureInterventionEvidence)
    for name, value in evidence_kwargs.items():
        object.__setattr__(unsigned, name, value)
    signature = execution_signer(causal_evidence_execution_receipt_body(unsigned))
    return FeatureInterventionEvidence(**evidence_kwargs, execution_signature=signature)


@dataclass(frozen=True, slots=True)
class SAEPromotionCriteria:
    minimum_fve: float
    maximum_reconstruction_mse: float
    maximum_average_active_features: float
    minimum_feature_stability: float
    minimum_causal_effect: float
    maximum_protected_effect: float

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(not math.isfinite(value) or value < 0 for value in values.values()):
            raise DataValidationError("SAE promotion criteria must be finite and non-negative")
        if not 0 <= self.minimum_fve <= 1 or not 0 <= self.minimum_feature_stability <= 1:
            raise DataValidationError("SAE FVE and stability criteria must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class SAESparsitySweepPoint:
    config_fingerprint: str
    checkpoint_fingerprint: str
    fraction_variance_explained: float
    reconstruction_mse: float
    average_active_features: float
    selected: bool

    def __post_init__(self) -> None:
        if (
            not _SHA256.fullmatch(self.config_fingerprint)
            or not _SHA256.fullmatch(self.checkpoint_fingerprint)
            or any(
                not math.isfinite(value) or value < 0
                for value in (
                    self.fraction_variance_explained,
                    self.reconstruction_mse,
                    self.average_active_features,
                )
            )
            or not 0 <= self.fraction_variance_explained <= 1
            or type(self.selected) is not bool
        ):
            raise DataValidationError("SAE sparsity-sweep point is invalid")


@dataclass(frozen=True, slots=True)
class SAEPairedExecutionAudit:
    baseline_records: tuple[GenerationRecord, ...]
    intervention_records: tuple[GenerationRecord, ...]
    factuality_delta: float

    def __post_init__(self) -> None:
        baseline = tuple(self.baseline_records)
        intervention = tuple(self.intervention_records)
        if (
            not baseline
            or len(baseline) != len(intervention)
            or tuple(value.question_id for value in baseline)
            != tuple(value.question_id for value in intervention)
            or len({value.question_id for value in baseline}) != len(baseline)
            or any(value.outcome is Outcome.UNSCORABLE for value in (*baseline, *intervention))
        ):
            raise DataValidationError("SAE paired execution audit is invalid")
        expected = (
            sum(value.outcome is Outcome.CORRECT for value in intervention)
            - sum(value.outcome is Outcome.CORRECT for value in baseline)
        ) / len(baseline)
        if not math.isclose(self.factuality_delta, expected, rel_tol=0, abs_tol=1e-12):
            raise DataValidationError("SAE paired execution effect differs from its rows")
        object.__setattr__(self, "baseline_records", baseline)
        object.__setattr__(self, "intervention_records", intervention)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_records": [value.to_dict() for value in self.baseline_records],
            "intervention_records": [
                value.to_dict() for value in self.intervention_records
            ],
            "factuality_delta": self.factuality_delta,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SAEPairedExecutionAudit:
        if set(value) != {
            "baseline_records",
            "intervention_records",
            "factuality_delta",
        }:
            raise DataValidationError("SAE paired execution audit keys differ")
        return cls(
            baseline_records=tuple(
                GenerationRecord.from_dict(item) for item in value["baseline_records"]
            ),
            intervention_records=tuple(
                GenerationRecord.from_dict(item)
                for item in value["intervention_records"]
            ),
            factuality_delta=float(value["factuality_delta"]),
        )


@dataclass(frozen=True, slots=True)
class SAEInterpretabilityAudit:
    top_activating_question_ids: Mapping[int, tuple[str, ...]]
    evaluation_question_ids: tuple[str, ...]
    control_seed: int
    prompt_transfer_effects: Mapping[int, Mapping[str, float]]
    negative_control_effects: Mapping[int, Mapping[str, float]]
    source_question_bundle_sha256: str
    prompt_transfer_execution: Mapping[
        int, Mapping[str, SAEPairedExecutionAudit]
    ] | None = None
    negative_control_execution: Mapping[
        int, Mapping[str, SAEPairedExecutionAudit]
    ] | None = None

    def __post_init__(self) -> None:
        top = {
            int(feature): tuple(str(value).strip() for value in identifiers)
            for feature, identifiers in self.top_activating_question_ids.items()
        }
        evaluation_question_ids = tuple(
            str(value).strip() for value in self.evaluation_question_ids
        )
        transfer = {
            int(feature): MappingProxyType(
                {str(prompt): float(effect) for prompt, effect in effects.items()}
            )
            for feature, effects in self.prompt_transfer_effects.items()
        }
        controls = {
            int(feature): MappingProxyType(
                {str(alpha): float(effect) for alpha, effect in effects.items()}
            )
            for feature, effects in self.negative_control_effects.items()
        }
        transfer_execution = {
            int(feature): MappingProxyType(dict(executions))
            for feature, executions in (
                self.prompt_transfer_execution or {}
            ).items()
        }
        control_execution = {
            int(feature): MappingProxyType(dict(executions))
            for feature, executions in (
                self.negative_control_execution or {}
            ).items()
        }
        if (
            not _SHA256.fullmatch(self.source_question_bundle_sha256)
            or not top
            or not evaluation_question_ids
            or len(set(evaluation_question_ids)) != len(evaluation_question_ids)
            or any(not value for value in evaluation_question_ids)
            or type(self.control_seed) is not int
            or self.control_seed < 0
            or set(top) != set(transfer)
            or set(top) != set(controls)
            or any(
                len(values) < 5
                or len(set(values)) != len(values)
                or any(not value for value in values)
                for values in top.values()
            )
            or any(
                set(effects) != {"P0-neutral", "P2-calibrated-abstention"}
                for effects in transfer.values()
            )
            or any(not effects for effects in controls.values())
            or any(
                not math.isfinite(value)
                for mappings in (*transfer.values(), *controls.values())
                for value in mappings.values()
            )
        ):
            raise DataValidationError("SAE interpretability audit is incomplete")
        required_controls = {
            "negative_alpha",
            "label_shuffled",
            "matched_random",
            "unrelated_layer",
            "gaussian",
            "zero_hook",
            "different_prompt",
        }
        if (transfer_execution or control_execution) and (
                set(transfer_execution) != set(top)
                or set(control_execution) != set(top)
                or any(
                    set(executions) != {"P0-neutral", "P2-calibrated-abstention"}
                    for executions in transfer_execution.values()
                )
                or any(
                    set(executions) != required_controls
                    for executions in control_execution.values()
                )
                or any(
                    not math.isclose(
                        audit.factuality_delta,
                        transfer[feature][name],
                        rel_tol=0,
                        abs_tol=1e-12,
                    )
                    for feature, executions in transfer_execution.items()
                    for name, audit in executions.items()
                )
                or any(
                    not math.isclose(
                        audit.factuality_delta,
                        controls[feature][name],
                        rel_tol=0,
                        abs_tol=1e-12,
                    )
                    for feature, executions in control_execution.items()
                    for name, audit in executions.items()
                )
                or any(
                    tuple(record.question_id for record in audit.baseline_records)
                    != evaluation_question_ids
                    or tuple(record.question_id for record in audit.intervention_records)
                    != evaluation_question_ids
                    for executions_by_feature in (
                        transfer_execution,
                        control_execution,
                    )
                    for executions in executions_by_feature.values()
                    for audit in executions.values()
                )
        ):
            raise DataValidationError(
                "SAE interpretability execution evidence differs"
            )
        object.__setattr__(self, "top_activating_question_ids", MappingProxyType(top))
        object.__setattr__(self, "evaluation_question_ids", evaluation_question_ids)
        object.__setattr__(self, "prompt_transfer_effects", MappingProxyType(transfer))
        object.__setattr__(self, "negative_control_effects", MappingProxyType(controls))
        object.__setattr__(
            self,
            "prompt_transfer_execution",
            MappingProxyType(transfer_execution) if transfer_execution else None,
        )
        object.__setattr__(
            self,
            "negative_control_execution",
            MappingProxyType(control_execution) if control_execution else None,
        )


@dataclass(frozen=True, slots=True)
class LongComputationReceipt:
    wall_time_seconds: float
    peak_gpu_memory_bytes: int
    package_lock_sha256: str
    model_snapshot_sha256: str
    resumable_chain_head: str
    runtime_artifact_sha256: str
    execution_public_key: str
    training_corpus_sha256: str
    validation_corpus_sha256: str
    measurement_method: str
    execution_signature: str

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.wall_time_seconds)
            or self.wall_time_seconds <= 0
            or type(self.peak_gpu_memory_bytes) is not int
            or self.peak_gpu_memory_bytes <= 0
            or any(
                not _SHA256.fullmatch(value)
                for value in (
                    self.package_lock_sha256,
                    self.model_snapshot_sha256,
                    self.resumable_chain_head,
                    self.runtime_artifact_sha256,
                    self.execution_public_key,
                    self.training_corpus_sha256,
                    self.validation_corpus_sha256,
                )
            )
            or re.fullmatch(r"[0-9a-f]{128}", self.execution_signature) is None
            or self.measurement_method
            not in {
                "resource.getrusage:RUSAGE_SELF:darwin-bytes",
                "resource.getrusage:RUSAGE_SELF:posix-kib",
            }
        ):
            raise DataValidationError("long-computation receipt is invalid")
        try:
            Ed25519PublicKey.from_public_bytes(
                bytes.fromhex(self.execution_public_key)
            ).verify(
                bytes.fromhex(self.execution_signature),
                canonical_json(long_computation_receipt_body(self)).encode(),
            )
        except (InvalidSignature, ValueError) as exc:
            raise DataValidationError(
                "long-computation receipt signature is invalid"
            ) from exc


def long_computation_receipt_body(
    receipt: LongComputationReceipt,
) -> dict[str, Any]:
    """Canonical runtime-signed receipt for the resumable E7 SAE run."""

    return {
        "receipt_kind": "e7-long-computation-v2",
        "wall_time_seconds": receipt.wall_time_seconds,
        "peak_gpu_memory_bytes": receipt.peak_gpu_memory_bytes,
        "package_lock_sha256": receipt.package_lock_sha256,
        "model_snapshot_sha256": receipt.model_snapshot_sha256,
        "resumable_chain_head": receipt.resumable_chain_head,
        "runtime_artifact_sha256": receipt.runtime_artifact_sha256,
        "execution_public_key": receipt.execution_public_key,
        "training_corpus_sha256": receipt.training_corpus_sha256,
        "validation_corpus_sha256": receipt.validation_corpus_sha256,
        "measurement_method": receipt.measurement_method,
    }


def _create_long_computation_receipt(
    *,
    wall_time_seconds: float,
    peak_gpu_memory_bytes: int,
    package_lock_sha256: str,
    model_snapshot_sha256: str,
    resumable_chain_head: str,
    runtime_artifact_sha256: str,
    execution_public_key: str,
    training_corpus_sha256: str,
    validation_corpus_sha256: str,
    measurement_method: str,
    execution_signer: Callable[[Mapping[str, Any]], str],
) -> LongComputationReceipt:
    values: dict[str, Any] = {
        "wall_time_seconds": wall_time_seconds,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "package_lock_sha256": package_lock_sha256,
        "model_snapshot_sha256": model_snapshot_sha256,
        "resumable_chain_head": resumable_chain_head,
        "runtime_artifact_sha256": runtime_artifact_sha256,
        "execution_public_key": execution_public_key,
        "training_corpus_sha256": training_corpus_sha256,
        "validation_corpus_sha256": validation_corpus_sha256,
        "measurement_method": measurement_method,
    }
    unsigned = object.__new__(LongComputationReceipt)
    for name, value in values.items():
        object.__setattr__(unsigned, name, value)
    signature = execution_signer(long_computation_receipt_body(unsigned))
    return LongComputationReceipt(**values, execution_signature=signature)


@dataclass(frozen=True, slots=True)
class SAECheckpointResultSequence(Sequence[SAETrainingResult]):
    """Checksum-verified sweep checkpoints loaded one at a time on demand."""

    directories: tuple[Path, ...]
    artifact_sha256s: tuple[str, ...]
    config_fingerprints: tuple[str, ...]
    checkpoint_fingerprints: tuple[str, ...]
    training_fingerprint: str
    validation_fingerprint: str

    def __post_init__(self) -> None:
        count = len(self.directories)
        if (
            count == 0
            or any(
                len(values) != count
                for values in (
                    self.artifact_sha256s,
                    self.config_fingerprints,
                    self.checkpoint_fingerprints,
                )
            )
            or any(
                _SHA256.fullmatch(value) is None
                for value in (
                    *self.artifact_sha256s,
                    *self.config_fingerprints,
                    *self.checkpoint_fingerprints,
                    self.training_fingerprint,
                    self.validation_fingerprint,
                )
            )
        ):
            raise DataValidationError("SAE checkpoint sequence identity is invalid")
        object.__setattr__(
            self,
            "directories",
            tuple(Path(os.path.abspath(value)) for value in self.directories),
        )

    def __len__(self) -> int:
        return len(self.directories)

    @overload
    def __getitem__(self, index: int) -> SAETrainingResult: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[SAETrainingResult, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> SAETrainingResult | tuple[SAETrainingResult, ...]:
        if isinstance(index, slice):
            return tuple(self[value] for value in range(*index.indices(len(self))))
        resolved = index if index >= 0 else len(self) + index
        if resolved < 0 or resolved >= len(self):
            raise IndexError(index)
        path = self.directories[resolved]
        if (
            path.is_symlink()
            or not path.is_dir()
            or {value.name for value in path.iterdir()}
            != {"metadata.json", "sae.safetensors"}
            or any(value.is_symlink() for value in path.iterdir())
            or sha256_path(path) != self.artifact_sha256s[resolved]
        ):
            raise FrozenArtifactError("persisted SAE sweep checkpoint changed")
        result = load_sae(
            path,
            expected_training_fingerprint=self.training_fingerprint,
            expected_validation_fingerprint=self.validation_fingerprint,
        )
        if (
            sha256_path(path) != self.artifact_sha256s[resolved]
            or path.is_symlink()
            or sae_config_fingerprint(result.config)
            != self.config_fingerprints[resolved]
            or sae_checkpoint_fingerprint(result)
            != self.checkpoint_fingerprints[resolved]
        ):
            raise FrozenArtifactError("persisted SAE sweep identity changed")
        return result


@dataclass(frozen=True, slots=True)
class MeasuredSAESweep:
    """SAE sweep results coupled to the wrapper-measured execution receipt."""

    results: SAECheckpointResultSequence
    receipt: LongComputationReceipt

    def __post_init__(self) -> None:
        if len(self.results) == 0:
            raise DataValidationError("measured SAE sweep cannot be empty")


def _sae_sweep_checkpoint_receipt_body(
    result: SAETrainingResult,
    *,
    index: int,
    plan_digest: str,
    artifact_sha256: str,
) -> dict[str, Any]:
    return {
        "receipt_kind": "e7-sae-sweep-checkpoint-v1",
        "index": index,
        "plan_digest": plan_digest,
        "artifact_sha256": artifact_sha256,
        "config_fingerprint": sae_config_fingerprint(result.config),
        "checkpoint_fingerprint": sae_checkpoint_fingerprint(result),
        "metrics": asdict(result.metrics),
        "loss_history_sha256": stable_hash(list(result.loss_history)),
        "training_fingerprint": result.training_fingerprint,
        "validation_fingerprint": result.validation_fingerprint,
        "training_schema_digest": result.training_schema.digest,
        "validation_schema_digest": result.validation_schema.digest,
        "training_rows": result.training_rows,
        "validation_rows": result.validation_rows,
    }


def _load_signed_sae_sweep_checkpoint(
    entry: Path,
    *,
    index: int,
    plan_digest: str,
    execution_public_key: str,
    training_fingerprint: str,
    validation_fingerprint: str,
) -> tuple[SAETrainingResult, str]:
    if (
        entry.is_symlink()
        or not entry.is_dir()
        or {value.name for value in entry.iterdir()} != {"sae", "receipt.json"}
        or any(value.is_symlink() for value in entry.iterdir())
    ):
        raise FrozenArtifactError("SAE sweep checkpoint entry inventory differs")
    sae_path = entry / "sae"
    receipt_path = entry / "receipt.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise FrozenArtifactError("SAE sweep checkpoint receipt is not regular")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        body = receipt["body"]
        signature = receipt["signature"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise FrozenArtifactError("cannot read SAE sweep checkpoint receipt") from exc
    if (
        not isinstance(receipt, dict)
        or set(receipt) != {"body", "signature"}
        or not isinstance(body, Mapping)
        or type(signature) is not str
        or re.fullmatch(r"[0-9a-f]{128}", signature) is None
    ):
        raise FrozenArtifactError("SAE sweep checkpoint receipt fields differ")
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(execution_public_key)).verify(
            bytes.fromhex(signature), canonical_json(body).encode()
        )
    except (InvalidSignature, ValueError) as exc:
        raise FrozenArtifactError("SAE sweep checkpoint signature is invalid") from exc
    artifact_sha = sha256_path(sae_path)
    result = load_sae(
        sae_path,
        expected_training_fingerprint=training_fingerprint,
        expected_validation_fingerprint=validation_fingerprint,
    )
    expected = _sae_sweep_checkpoint_receipt_body(
        result,
        index=index,
        plan_digest=plan_digest,
        artifact_sha256=artifact_sha,
    )
    if (
        canonical_json(body) != canonical_json(expected)
        or sha256_path(sae_path) != artifact_sha
        or entry.is_symlink()
        or sae_path.is_symlink()
    ):
        raise FrozenArtifactError("SAE sweep checkpoint differs from its signature")
    return result, artifact_sha


def fit_e7_sae_sweep_measured(
    training: ActivationCorpus,
    validation: ActivationCorpus,
    configs: Sequence[SAEConfig],
    *,
    package_lock: str | Path,
    model_snapshot_sha256: str,
    runtime_artifact_sha256: str,
    execution_public_key: str,
    execution_signer: Callable[[Mapping[str, Any]], str],
    checkpoint_directory: str | Path,
) -> MeasuredSAESweep:
    """Resume the E7 sweep from immutable checkpoints without retaining all SAEs."""

    frozen_configs = tuple(configs)
    sparsity_levels = {
        (
            config.sparsity.value,
            config.top_k
            if config.sparsity is SAESparsity.TOP_K
            else config.l1_coefficient,
        )
        for config in frozen_configs
    }
    if (
        len(frozen_configs) < 3
        or len({sae_config_fingerprint(config) for config in frozen_configs})
        != len(frozen_configs)
        or len(sparsity_levels) < 3
    ):
        raise DataValidationError(
            "measured E7 SAE sweep requires at least three distinct sparsity levels"
        )
    lock = Path(package_lock)
    lock_sha = sha256_file(lock)
    normalized = validate_active_study_artifact_paths(
        {
            "E7 SAE sweep checkpoints": checkpoint_directory,
            "E7 SAE training corpus": training.directory,
            "E7 SAE validation corpus": validation.directory,
        }
    )
    checkpoint_root = normalized["E7 SAE sweep checkpoints"]
    if checkpoint_root.is_symlink():
        raise FrozenArtifactError("SAE sweep checkpoint root cannot be a symlink")
    checkpoint_root.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_root.is_symlink() or (
        checkpoint_root.exists() and not checkpoint_root.is_dir()
    ):
        raise FrozenArtifactError("SAE sweep checkpoint root is invalid")
    checkpoint_root.mkdir(exist_ok=True)
    checkpoints_root = checkpoint_root / "checkpoints"
    plan_body = {
        "schema_version": 1,
        "training_corpus_sha256": sha256_path(training.directory),
        "training_data_fingerprint": training.data_fingerprint,
        "validation_corpus_sha256": sha256_path(validation.directory),
        "validation_data_fingerprint": validation.data_fingerprint,
        "package_lock_sha256": lock_sha,
        "model_snapshot_sha256": model_snapshot_sha256,
        "runtime_artifact_sha256": runtime_artifact_sha256,
        "execution_public_key": execution_public_key,
        "configs": [
            {
                **asdict(config),
                "sparsity": config.sparsity.value,
            }
            for config in frozen_configs
        ],
    }
    plan = {**plan_body, "plan_digest": stable_hash(plan_body)}
    plan_path = checkpoint_root / "plan.json"
    if plan_path.is_symlink():
        raise FrozenArtifactError("SAE sweep resume plan cannot be linked")
    if plan_path.exists():
        try:
            stored_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FrozenArtifactError("cannot read SAE sweep resume plan") from exc
        if stored_plan != plan:
            raise FrozenArtifactError("SAE sweep resume plan differs")
    else:
        unexpected = {value.name for value in checkpoint_root.iterdir()}
        if unexpected:
            raise FrozenArtifactError("unbound files precede the SAE sweep resume plan")
        plan_path.write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if checkpoints_root.is_symlink():
        raise FrozenArtifactError("SAE sweep checkpoint inventory cannot be linked")
    checkpoints_root.mkdir(exist_ok=True)
    if {value.name for value in checkpoint_root.iterdir()} != {
        "plan.json",
        "checkpoints",
    }:
        raise FrozenArtifactError("SAE sweep root inventory differs")
    checkpoint_names = tuple(
        f"{index:03d}-{sae_config_fingerprint(config)[:16]}"
        for index, config in enumerate(frozen_configs)
    )
    for candidate in tuple(checkpoints_root.iterdir()):
        matching_name = next(
            (
                name
                for name in checkpoint_names
                if candidate.name.startswith(f".{name}.stage-")
                and re.fullmatch(
                    rf"\.{re.escape(name)}\.stage-[A-Za-z0-9._-]+",
                    candidate.name,
                )
                is not None
            ),
            None,
        )
        if matching_name is None:
            continue
        if candidate.is_symlink() or not candidate.is_dir():
            raise FrozenArtifactError("SAE sweep stale stage is not a regular directory")
        shutil.rmtree(candidate)
    if {value.name for value in checkpoints_root.iterdir()} - set(checkpoint_names):
        raise FrozenArtifactError("SAE sweep resume directory has undeclared checkpoints")
    started = time.perf_counter()
    artifact_shas: list[str] = []
    checkpoint_fingerprints: list[str] = []
    for index, (name, config) in enumerate(
        zip(checkpoint_names, frozen_configs, strict=True)
    ):
        entry = checkpoints_root / name
        if entry.is_symlink():
            raise FrozenArtifactError("SAE sweep checkpoint entry cannot be linked")
        if entry.exists():
            result, artifact_sha = _load_signed_sae_sweep_checkpoint(
                entry,
                index=index,
                plan_digest=str(plan["plan_digest"]),
                execution_public_key=execution_public_key,
                training_fingerprint=training.data_fingerprint,
                validation_fingerprint=validation.data_fingerprint,
            )
            if sae_config_fingerprint(result.config) != sae_config_fingerprint(config):
                raise FrozenArtifactError("resumed SAE checkpoint has another config")
        else:
            result = fit_sparse_autoencoder_corpus(training, validation, config)
            stage = Path(
                tempfile.mkdtemp(prefix=f".{name}.stage-", dir=checkpoints_root)
            )
            try:
                sae_path = stage / "sae"
                save_sae(sae_path, result)
                artifact_sha = sha256_path(sae_path)
                body = _sae_sweep_checkpoint_receipt_body(
                    result,
                    index=index,
                    plan_digest=str(plan["plan_digest"]),
                    artifact_sha256=artifact_sha,
                )
                (stage / "receipt.json").write_text(
                    json.dumps(
                        {"body": body, "signature": execution_signer(body)},
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                _load_signed_sae_sweep_checkpoint(
                    stage,
                    index=index,
                    plan_digest=str(plan["plan_digest"]),
                    execution_public_key=execution_public_key,
                    training_fingerprint=training.data_fingerprint,
                    validation_fingerprint=validation.data_fingerprint,
                )
                os.replace(stage, entry)
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
        artifact_shas.append(artifact_sha)
        checkpoint_fingerprints.append(sae_checkpoint_fingerprint(result))
        del result
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    results = SAECheckpointResultSequence(
        directories=tuple(
            checkpoints_root / name / "sae" for name in checkpoint_names
        ),
        artifact_sha256s=tuple(artifact_shas),
        config_fingerprints=tuple(
            sae_config_fingerprint(config) for config in frozen_configs
        ),
        checkpoint_fingerprints=tuple(checkpoint_fingerprints),
        training_fingerprint=training.data_fingerprint,
        validation_fingerprint=validation.data_fingerprint,
    )
    wall_time = time.perf_counter() - started
    raw_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        peak_bytes = raw_peak
        measurement_method = "resource.getrusage:RUSAGE_SELF:darwin-bytes"
    else:
        peak_bytes = raw_peak * 1024
        measurement_method = "resource.getrusage:RUSAGE_SELF:posix-kib"
    receipt = _create_long_computation_receipt(
        wall_time_seconds=wall_time,
        peak_gpu_memory_bytes=peak_bytes,
        package_lock_sha256=lock_sha,
        model_snapshot_sha256=model_snapshot_sha256,
        resumable_chain_head=e7_resumable_chain_head(
            training_corpus=training,
            validation_corpus=validation,
            sweep_results=results,
        ),
        runtime_artifact_sha256=runtime_artifact_sha256,
        execution_public_key=execution_public_key,
        training_corpus_sha256=sha256_path(training.directory),
        validation_corpus_sha256=sha256_path(validation.directory),
        measurement_method=measurement_method,
        execution_signer=execution_signer,
    )
    return MeasuredSAESweep(results=results, receipt=receipt)


def validate_e7_sae_promotion_criteria(
    criteria: SAEPromotionCriteria, config: SAEConfig
) -> None:
    """Enforce the preregistered non-vacuous E7 SAE promotion bounds."""

    maximum_active = (
        float(config.top_k)
        if config.sparsity is SAESparsity.TOP_K
        else max(1.0, config.resolved_latent_width * 0.10)
    )
    if (
        criteria.minimum_fve < 0.20
        or criteria.maximum_reconstruction_mse > 1.0
        or criteria.maximum_average_active_features > maximum_active
        or criteria.minimum_feature_stability < 0.80
        or criteria.minimum_causal_effect < 0.02
        or criteria.maximum_protected_effect > 0.02
    ):
        raise DataValidationError("SAE promotion criteria are weaker than E7 registration")


@dataclass(frozen=True, slots=True)
class SAEInterventionArtifact:
    training: SAETrainingResult
    latent_direction: SAELatentDirection
    decoded_direction: Tensor
    evidence: tuple[FeatureInterventionEvidence, ...]
    stability_selections: tuple[SeedFeatureSelection, ...]
    aligned_feature_stability: float
    criteria: SAEPromotionCriteria
    sparsity_sweep: tuple[SAESparsitySweepPoint, ...] = ()
    sparsity_sweep_results: Sequence[SAETrainingResult] = ()
    interpretability_audit: SAEInterpretabilityAudit | None = None
    long_computation_receipt: LongComputationReceipt | None = None
    feature_stability_method: str = "oriented_decoder_cosine_hungarian_min_pair_v1"
    schema_version: int = 7

    def __post_init__(self) -> None:
        if (
            self.schema_version != 7
            or self.feature_stability_method
            != "oriented_decoder_cosine_hungarian_min_pair_v1"
        ):
            raise DataValidationError("unsupported SAE-intervention schema version")
        validate_e7_sae_promotion_criteria(self.criteria, self.training.config)
        decoded = self.decoded_direction.detach().cpu().float().contiguous().clone()
        if decoded.shape != (self.training.config.input_width,):
            raise DataValidationError("decoded SAE intervention has the wrong width")
        if not torch.isfinite(decoded).all() or not torch.allclose(
            torch.linalg.vector_norm(decoded), torch.tensor(1.0), atol=1e-5
        ):
            raise DataValidationError("decoded SAE intervention must be a finite unit vector")
        recomputed = decode_latent_direction(self.training.model, self.latent_direction)
        if not torch.allclose(decoded, recomputed, atol=1e-6):
            raise DataValidationError("decoded SAE intervention differs from the frozen SAE")
        stability_selections = tuple(self.stability_selections)
        selected_feature_stability(stability_selections)
        stability = float(self.aligned_feature_stability)
        if not 0 <= stability <= 1 or not math.isfinite(stability):
            raise DataValidationError(
                "SAE aligned feature stability must be in [0, 1]"
            )
        if not _SHA256.fullmatch(self.latent_direction.selection_fingerprint):
            raise DataValidationError("SAE feature-selection fingerprint must be SHA-256")
        if self.latent_direction.selection_schema.partition != "T-steer" or not (
            self.training.training_schema.is_compatible_representation(
                self.latent_direction.selection_schema
            )
        ):
            raise DataValidationError("SAE feature selection must use compatible T-steer features")
        selected = set(self.latent_direction.selected_features)
        if (
            len(self.evidence) != len(selected)
            or {item.feature_index for item in self.evidence} != selected
        ):
            raise DataValidationError("every selected SAE feature requires causal evidence")
        current_checkpoint = sae_checkpoint_fingerprint(self.training)
        sweep = tuple(self.sparsity_sweep)
        sweep_results: Sequence[SAETrainingResult] = (
            self.sparsity_sweep_results
            if isinstance(self.sparsity_sweep_results, SAECheckpointResultSequence)
            else tuple(self.sparsity_sweep_results)
        )
        if sweep:
            selected_sweep = tuple(value for value in sweep if value.selected)
            if (
                len(sweep) < 3
                or len(sweep_results) != len(sweep)
                or len({value.config_fingerprint for value in sweep}) != len(sweep)
                or len({value.checkpoint_fingerprint for value in sweep}) != len(sweep)
                or len(selected_sweep) != 1
                or selected_sweep[0].checkpoint_fingerprint != current_checkpoint
                or not math.isclose(
                    selected_sweep[0].fraction_variance_explained,
                    self.training.metrics.fraction_variance_explained,
                    rel_tol=0,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    selected_sweep[0].reconstruction_mse,
                    self.training.metrics.reconstruction_mse,
                    rel_tol=0,
                    abs_tol=1e-12,
                )
            ):
                raise DataValidationError("SAE sparsity sweep does not select this checkpoint")
            for point, result in zip(sweep, sweep_results, strict=True):
                if (
                    point.config_fingerprint != sae_config_fingerprint(result.config)
                    or point.checkpoint_fingerprint
                    != sae_checkpoint_fingerprint(result)
                    or point.selected
                    is not (point.checkpoint_fingerprint == current_checkpoint)
                    or result.training_fingerprint
                    != self.training.training_fingerprint
                    or result.validation_fingerprint
                    != self.training.validation_fingerprint
                    or result.training_schema != self.training.training_schema
                    or result.validation_schema != self.training.validation_schema
                    or result.training_rows != self.training.training_rows
                    or result.validation_rows != self.training.validation_rows
                    or any(
                        not math.isclose(
                            metric,
                            stored,
                            rel_tol=0,
                            abs_tol=1e-12,
                        )
                        for metric, stored in (
                            (
                                result.metrics.fraction_variance_explained,
                                point.fraction_variance_explained,
                            ),
                            (
                                result.metrics.reconstruction_mse,
                                point.reconstruction_mse,
                            ),
                            (
                                result.metrics.average_active_features,
                                point.average_active_features,
                            ),
                        )
                    )
                ):
                    raise DataValidationError(
                        "SAE sparsity sweep differs from its persisted checkpoint"
                    )
        elif sweep_results:
            raise DataValidationError("SAE sparsity checkpoints lack sweep declarations")
        if self.interpretability_audit is not None and set(
            self.interpretability_audit.top_activating_question_ids
        ) != selected:
            raise DataValidationError("SAE interpretability audit omits selected features")
        current_selections = [
            item
            for item in stability_selections
            if item.seed == self.training.config.seed
            and item.checkpoint_fingerprint == current_checkpoint
        ]
        if len(current_selections) != 1 or set(current_selections[0].selected_features) != selected:
            raise DataValidationError(
                "SAE stability evidence must include this checkpoint's selected features"
            )
        evidence_sources = {
            stable_hash(item.spec.feature_schema.source_identity()) for item in self.evidence
        }
        if evidence_sources != {
            stable_hash(self.latent_direction.selection_schema.source_identity())
        }:
            raise DataValidationError(
                "SAE causal evidence must use the selected feature model and prompt"
            )
        if len({item.spec.paired_question_fingerprint for item in self.evidence}) != 1:
            raise DataValidationError(
                "selected SAE features must be tested on one paired evaluation set"
            )
        geometries = {
            (
                item.spec.layer,
                item.spec.site,
                item.spec.token_scope,
                item.spec.alpha,
            )
            for item in self.evidence
        }
        if (
            len(geometries) != 1
            or next(iter(geometries))[3] not in {0.1, 0.25, 0.5, 1.0, 2.0}
        ):
            raise DataValidationError(
                "SAE causal features must share registered intervention geometry"
            )
        if self.training.metrics.fraction_variance_explained < self.criteria.minimum_fve:
            raise DataValidationError("SAE held-out FVE does not meet promotion criteria")
        if self.training.metrics.reconstruction_mse > self.criteria.maximum_reconstruction_mse:
            raise DataValidationError("SAE held-out reconstruction MSE fails promotion")
        if (
            self.training.metrics.average_active_features
            > self.criteria.maximum_average_active_features
        ):
            raise DataValidationError("SAE feature activity fails promotion")
        if stability < self.criteria.minimum_feature_stability:
            raise DataValidationError("SAE feature stability fails promotion")
        if not all(
            item.causally_supported(
                minimum_effect=self.criteria.minimum_causal_effect,
                maximum_protected_effect=self.criteria.maximum_protected_effect,
            )
            for item in self.evidence
        ):
            raise DataValidationError(
                "selected SAE features lack clean causal intervention evidence"
            )
        object.__setattr__(self, "decoded_direction", decoded)
        object.__setattr__(self, "stability_selections", stability_selections)
        object.__setattr__(self, "sparsity_sweep", sweep)
        object.__setattr__(self, "sparsity_sweep_results", sweep_results)

    @property
    def feature_stability(self) -> float:
        return self.aligned_feature_stability


def promote_sae_intervention(
    training: SAETrainingResult,
    latent_direction: SAELatentDirection,
    *,
    evidence: Sequence[FeatureInterventionEvidence],
    stability_selections: Sequence[SeedFeatureSelection],
    aligned_feature_stability: float,
    criteria: SAEPromotionCriteria,
    sparsity_sweep: Sequence[SAESparsitySweepPoint] = (),
    sparsity_sweep_results: Sequence[SAETrainingResult] = (),
    interpretability_audit: SAEInterpretabilityAudit | None = None,
    long_computation_receipt: LongComputationReceipt | None = None,
) -> SAEInterventionArtifact:
    return SAEInterventionArtifact(
        training=training,
        latent_direction=latent_direction,
        decoded_direction=decode_latent_direction(training.model, latent_direction),
        evidence=tuple(evidence),
        stability_selections=tuple(stability_selections),
        aligned_feature_stability=aligned_feature_stability,
        criteria=criteria,
        sparsity_sweep=tuple(sparsity_sweep),
        sparsity_sweep_results=(
            sparsity_sweep_results
            if isinstance(sparsity_sweep_results, SAECheckpointResultSequence)
            else tuple(sparsity_sweep_results)
        ),
        interpretability_audit=interpretability_audit,
        long_computation_receipt=long_computation_receipt,
    )


def _config_dict(config: SAEConfig) -> dict[str, Any]:
    value = asdict(config)
    value["sparsity"] = config.sparsity.value
    return value


def sae_config_fingerprint(config: SAEConfig) -> str:
    """Canonical identity of one registered SAE sparsity configuration."""

    return stable_hash(_config_dict(config))


def save_sae(directory: str | Path, result: SAETrainingResult) -> None:
    destination = validate_active_study_artifact_paths(
        {"E7 SAE artifact": directory}
    )["E7 SAE artifact"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FrozenArtifactError(f"refusing to overwrite SAE artifact: {destination}")
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        tensor_path = stage / "sae.safetensors"
        state = {
            key: value.detach().cpu().contiguous()
            for key, value in result.model.state_dict().items()
        }
        save_file(state, tensor_path)
        metadata_body = {
            "schema_version": result.schema_version,
            "config": _config_dict(result.config),
            "metrics": asdict(result.metrics),
            "loss_history": list(result.loss_history),
            "training_fingerprint": result.training_fingerprint,
            "validation_fingerprint": result.validation_fingerprint,
            "training_schema": result.training_schema.to_dict(),
            "validation_schema": result.validation_schema.to_dict(),
            "training_rows": result.training_rows,
            "validation_rows": result.validation_rows,
            "tensor_keys": sorted(state),
            "tensor_sha256": sha256_file(tensor_path),
        }
        metadata = {**metadata_body, "metadata_digest": stable_hash(metadata_body)}
        (stage / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def load_sae(
    directory: str | Path,
    *,
    expected_training_fingerprint: str | None = None,
    expected_validation_fingerprint: str | None = None,
) -> SAETrainingResult:
    source = Path(directory)
    try:
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read SAE metadata: {exc}") from exc
    digest = metadata.pop("metadata_digest", None)
    if digest != stable_hash(metadata):
        raise FrozenArtifactError("SAE metadata digest mismatch")
    if metadata.get("schema_version") != 2:
        raise FrozenArtifactError("unsupported SAE schema version")
    tensor_path = source / "sae.safetensors"
    if sha256_file(tensor_path) != metadata.get("tensor_sha256"):
        raise FrozenArtifactError("SAE tensor checksum mismatch")
    if (
        expected_training_fingerprint is not None
        and metadata.get("training_fingerprint") != expected_training_fingerprint
    ):
        raise FrozenArtifactError("SAE was trained on a different data fingerprint")
    if (
        expected_validation_fingerprint is not None
        and metadata.get("validation_fingerprint") != expected_validation_fingerprint
    ):
        raise FrozenArtifactError("SAE was validated on a different data fingerprint")
    try:
        config_data = dict(metadata["config"])
        config_data["sparsity"] = SAESparsity(config_data["sparsity"])
        config = SAEConfig(**config_data)
        model = SparseAutoencoder(config)
        tensors = load_file(tensor_path, device="cpu")
        if set(tensors) != set(metadata["tensor_keys"]):
            raise FrozenArtifactError("unexpected or missing SAE tensors")
        model.load_state_dict(tensors, strict=True)
        model.eval()
        metrics = SAEMetrics(**metadata["metrics"])
        return SAETrainingResult(
            model=model,
            config=config,
            metrics=metrics,
            loss_history=tuple(float(value) for value in metadata["loss_history"]),
            training_fingerprint=str(metadata["training_fingerprint"]),
            validation_fingerprint=str(metadata["validation_fingerprint"]),
            training_schema=ActivationFeatureSchema.from_dict(metadata["training_schema"]),
            validation_schema=ActivationFeatureSchema.from_dict(metadata["validation_schema"]),
            training_rows=int(metadata["training_rows"]),
            validation_rows=int(metadata["validation_rows"]),
        )
    except (KeyError, TypeError, ValueError, RuntimeError, DataValidationError) as exc:
        raise FrozenArtifactError(f"invalid SAE artifact: {exc}") from exc


def save_sae_intervention(directory: str | Path, intervention: SAEInterventionArtifact) -> None:
    destination = validate_active_study_artifact_paths(
        {"E7 SAE intervention": directory}
    )["E7 SAE intervention"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FrozenArtifactError(f"refusing to overwrite SAE intervention: {destination}")
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        save_sae(stage / "sae", intervention.training)
        sweep_root = stage / "sparsity-sweep"
        sweep_root.mkdir()
        sweep_artifact_digests: list[str] = []
        for index, result in enumerate(intervention.sparsity_sweep_results):
            sweep_path = sweep_root / f"{index:03d}"
            save_sae(sweep_path, result)
            sweep_artifact_digests.append(sha256_path(sweep_path))
        tensor_path = stage / "intervention.safetensors"
        save_file(
            {
                "latent_direction": intervention.latent_direction.direction.contiguous(),
                "decoded_direction": intervention.decoded_direction.contiguous(),
            },
            tensor_path,
        )
        metadata_body = {
            "schema_version": intervention.schema_version,
            "sae_digest": sha256_path(stage / "sae"),
            "tensor_sha256": sha256_file(tensor_path),
            "selected_features": list(intervention.latent_direction.selected_features),
            "correct_count": intervention.latent_direction.correct_count,
            "incorrect_count": intervention.latent_direction.incorrect_count,
            "evidence": [
                {
                    "feature_index": item.feature_index,
                    "activation_factuality_delta": item.activation_factuality_delta,
                    "suppression_factuality_delta": item.suppression_factuality_delta,
                    "protected_behavior_deltas": dict(item.protected_behavior_deltas),
                    "execution_signature": item.execution_signature,
                    "native_execution_records": {
                        mode: [record.to_dict() for record in records]
                        for mode, records in (
                            item.native_execution_records or {}
                        ).items()
                    },
                    "factuality_outcomes": {
                        question_id: [outcome.value for outcome in outcomes]
                        for question_id, outcomes in item.factuality_outcomes.items()
                    },
                    "protected_outcomes": {
                        behavior: {
                            question_id: list(outcomes)
                            for question_id, outcomes in measurements.items()
                        }
                        for behavior, measurements in item.protected_outcomes.items()
                    },
                    "spec": {
                        "paired_question_fingerprint": (item.spec.paired_question_fingerprint),
                        "baseline_run_fingerprint": item.spec.baseline_run_fingerprint,
                        "activated_run_fingerprint": item.spec.activated_run_fingerprint,
                        "suppressed_run_fingerprint": item.spec.suppressed_run_fingerprint,
                        "factuality_sample_count": item.spec.factuality_sample_count,
                        "protected_sample_counts": dict(item.spec.protected_sample_counts),
                        "alpha": item.spec.alpha,
                        "token_scope": item.spec.token_scope.value,
                        "layer": item.spec.layer,
                        "site": item.spec.site.value,
                        "feature_schema": item.spec.feature_schema.to_dict(),
                        "runtime_artifact_sha256": (
                            item.spec.runtime_artifact_sha256
                        ),
                        "execution_public_key": item.spec.execution_public_key,
                        "source_question_bundle_sha256": (
                            item.spec.source_question_bundle_sha256
                        ),
                    },
                }
                for item in intervention.evidence
            ],
            "stability_selections": [
                {
                    "seed": item.seed,
                    "checkpoint_fingerprint": item.checkpoint_fingerprint,
                    "selected_features": list(item.selected_features),
                }
                for item in intervention.stability_selections
            ],
            "aligned_feature_stability": intervention.feature_stability,
            "feature_stability_method": intervention.feature_stability_method,
            "feature_selection_fingerprint": (intervention.latent_direction.selection_fingerprint),
            "feature_selection_schema": intervention.latent_direction.selection_schema.to_dict(),
            "criteria": asdict(intervention.criteria),
            "sparsity_sweep": [asdict(value) for value in intervention.sparsity_sweep],
            "sparsity_sweep_artifact_digests": sweep_artifact_digests,
            "interpretability_audit": (
                {
                    "top_activating_question_ids": {
                        str(feature): list(identifiers)
                        for feature, identifiers in (
                            intervention.interpretability_audit.top_activating_question_ids.items()
                        )
                    },
                    "evaluation_question_ids": list(
                        intervention.interpretability_audit.evaluation_question_ids
                    ),
                    "control_seed": intervention.interpretability_audit.control_seed,
                    "prompt_transfer_effects": {
                        str(feature): dict(effects)
                        for feature, effects in (
                            intervention.interpretability_audit.prompt_transfer_effects.items()
                        )
                    },
                    "negative_control_effects": {
                        str(feature): dict(effects)
                        for feature, effects in (
                            intervention.interpretability_audit.negative_control_effects.items()
                        )
                    },
                    "source_question_bundle_sha256": (
                        intervention.interpretability_audit.source_question_bundle_sha256
                    ),
                    "prompt_transfer_execution": {
                        str(feature): {
                            name: audit.to_dict()
                            for name, audit in executions.items()
                        }
                        for feature, executions in (
                            intervention.interpretability_audit.prompt_transfer_execution
                            or {}
                        ).items()
                    },
                    "negative_control_execution": {
                        str(feature): {
                            name: audit.to_dict()
                            for name, audit in executions.items()
                        }
                        for feature, executions in (
                            intervention.interpretability_audit.negative_control_execution
                            or {}
                        ).items()
                    },
                }
                if intervention.interpretability_audit is not None
                else None
            ),
            "long_computation_receipt": (
                asdict(intervention.long_computation_receipt)
                if intervention.long_computation_receipt is not None
                else None
            ),
        }
        metadata = {**metadata_body, "metadata_digest": stable_hash(metadata_body)}
        (stage / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def load_sae_intervention(directory: str | Path) -> SAEInterventionArtifact:
    source = Path(directory)
    try:
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read SAE-intervention metadata: {exc}") from exc
    digest = metadata.pop("metadata_digest", None)
    if digest != stable_hash(metadata):
        raise FrozenArtifactError("SAE-intervention metadata digest mismatch")
    expected_metadata_keys = {
        "schema_version",
        "sae_digest",
        "tensor_sha256",
        "selected_features",
        "correct_count",
        "incorrect_count",
        "evidence",
        "stability_selections",
        "aligned_feature_stability",
        "feature_stability_method",
        "feature_selection_fingerprint",
        "feature_selection_schema",
        "criteria",
        "sparsity_sweep",
        "sparsity_sweep_artifact_digests",
        "interpretability_audit",
        "long_computation_receipt",
    }
    if set(metadata) != expected_metadata_keys or metadata.get("schema_version") != 7:
        raise FrozenArtifactError("unsupported SAE-intervention schema version")
    sweep_digests = metadata.get("sparsity_sweep_artifact_digests")
    if not isinstance(sweep_digests, list) or any(
        type(value) is not str or _SHA256.fullmatch(value) is None
        for value in sweep_digests
    ):
        raise FrozenArtifactError("SAE-intervention sweep digest inventory is invalid")
    sweep_root = source / "sparsity-sweep"
    expected_sweep_names = {f"{index:03d}" for index in range(len(sweep_digests))}
    if (
        source.is_symlink()
        or not source.is_dir()
        or {value.name for value in source.iterdir()}
        != {"metadata.json", "intervention.safetensors", "sae", "sparsity-sweep"}
        or sweep_root.is_symlink()
        or not sweep_root.is_dir()
        or {value.name for value in sweep_root.iterdir()} != expected_sweep_names
        or any(value.is_symlink() for value in source.rglob("*"))
        or any(
            sha256_path(sweep_root / f"{index:03d}") != expected_digest
            for index, expected_digest in enumerate(sweep_digests)
        )
    ):
        raise FrozenArtifactError("SAE-intervention artifact inventory differs")
    if sha256_path(source / "sae") != metadata.get("sae_digest"):
        raise FrozenArtifactError("frozen SAE component changed")
    tensor_path = source / "intervention.safetensors"
    if sha256_file(tensor_path) != metadata.get("tensor_sha256"):
        raise FrozenArtifactError("SAE-intervention tensor checksum mismatch")
    try:
        training = load_sae(source / "sae")
        sweep_points = tuple(
            SAESparsitySweepPoint(**value)
            for value in metadata["sparsity_sweep"]
        )
        if len(sweep_points) != len(sweep_digests):
            raise FrozenArtifactError("SAE sweep declarations and artifacts differ")
        sweep_results: Sequence[SAETrainingResult] = (
            SAECheckpointResultSequence(
                directories=tuple(
                    sweep_root / f"{index:03d}"
                    for index in range(len(sweep_digests))
                ),
                artifact_sha256s=tuple(sweep_digests),
                config_fingerprints=tuple(
                    value.config_fingerprint for value in sweep_points
                ),
                checkpoint_fingerprints=tuple(
                    value.checkpoint_fingerprint for value in sweep_points
                ),
                training_fingerprint=training.training_fingerprint,
                validation_fingerprint=training.validation_fingerprint,
            )
            if sweep_points
            else ()
        )
        tensors = load_file(tensor_path, device="cpu")
        if set(tensors) != {"latent_direction", "decoded_direction"}:
            raise FrozenArtifactError("unexpected SAE-intervention tensors")
        latent = SAELatentDirection(
            direction=tensors["latent_direction"],
            selected_features=tuple(int(value) for value in metadata["selected_features"]),
            correct_count=int(metadata["correct_count"]),
            incorrect_count=int(metadata["incorrect_count"]),
            selection_fingerprint=str(metadata["feature_selection_fingerprint"]),
            selection_schema=ActivationFeatureSchema.from_dict(
                metadata["feature_selection_schema"]
            ),
        )
        evidence = tuple(
            FeatureInterventionEvidence(
                feature_index=int(value["feature_index"]),
                activation_factuality_delta=float(value["activation_factuality_delta"]),
                suppression_factuality_delta=float(value["suppression_factuality_delta"]),
                protected_behavior_deltas=value["protected_behavior_deltas"],
                factuality_outcomes=value["factuality_outcomes"],
                protected_outcomes=value["protected_outcomes"],
                execution_signature=value.get("execution_signature"),
                native_execution_records={
                    mode: tuple(
                        GenerationRecord.from_dict(record) for record in records
                    )
                    for mode, records in value.get(
                        "native_execution_records", {}
                    ).items()
                },
                spec=CausalEvidenceSpec(
                    paired_question_fingerprint=str(value["spec"]["paired_question_fingerprint"]),
                    baseline_run_fingerprint=str(value["spec"]["baseline_run_fingerprint"]),
                    activated_run_fingerprint=str(value["spec"]["activated_run_fingerprint"]),
                    suppressed_run_fingerprint=str(value["spec"]["suppressed_run_fingerprint"]),
                    factuality_sample_count=int(value["spec"]["factuality_sample_count"]),
                    protected_sample_counts=value["spec"]["protected_sample_counts"],
                    alpha=float(value["spec"]["alpha"]),
                    token_scope=TokenScope(value["spec"]["token_scope"]),
                    layer=int(value["spec"]["layer"]),
                    site=ActivationSite(value["spec"]["site"]),
                    feature_schema=ActivationFeatureSchema.from_dict(
                        value["spec"]["feature_schema"]
                    ),
                    runtime_artifact_sha256=value["spec"].get(
                        "runtime_artifact_sha256"
                    ),
                    execution_public_key=value["spec"].get("execution_public_key"),
                    source_question_bundle_sha256=value["spec"].get(
                        "source_question_bundle_sha256"
                    ),
                ),
            )
            for value in metadata["evidence"]
        )
        stability_selections = tuple(
            SeedFeatureSelection(
                seed=int(value["seed"]),
                checkpoint_fingerprint=str(value["checkpoint_fingerprint"]),
                selected_features=tuple(int(feature) for feature in value["selected_features"]),
            )
            for value in metadata["stability_selections"]
        )
        intervention = SAEInterventionArtifact(
            training=training,
            latent_direction=latent,
            decoded_direction=tensors["decoded_direction"],
            evidence=evidence,
            stability_selections=stability_selections,
            aligned_feature_stability=float(metadata["aligned_feature_stability"]),
            criteria=SAEPromotionCriteria(**metadata["criteria"]),
            sparsity_sweep=sweep_points,
            sparsity_sweep_results=sweep_results,
            interpretability_audit=(
                SAEInterpretabilityAudit(
                    top_activating_question_ids={
                        int(feature): tuple(identifiers)
                        for feature, identifiers in metadata[
                            "interpretability_audit"
                        ]["top_activating_question_ids"].items()
                    },
                    evaluation_question_ids=tuple(
                        metadata["interpretability_audit"][
                            "evaluation_question_ids"
                        ]
                    ),
                    control_seed=int(
                        metadata["interpretability_audit"]["control_seed"]
                    ),
                    prompt_transfer_effects={
                        int(feature): effects
                        for feature, effects in metadata[
                            "interpretability_audit"
                        ]["prompt_transfer_effects"].items()
                    },
                    negative_control_effects={
                        int(feature): effects
                        for feature, effects in metadata[
                            "interpretability_audit"
                        ]["negative_control_effects"].items()
                    },
                    source_question_bundle_sha256=metadata[
                        "interpretability_audit"
                    ]["source_question_bundle_sha256"],
                    prompt_transfer_execution={
                        int(feature): {
                            name: SAEPairedExecutionAudit.from_dict(audit)
                            for name, audit in executions.items()
                        }
                        for feature, executions in metadata[
                            "interpretability_audit"
                        ]["prompt_transfer_execution"].items()
                    },
                    negative_control_execution={
                        int(feature): {
                            name: SAEPairedExecutionAudit.from_dict(audit)
                            for name, audit in executions.items()
                        }
                        for feature, executions in metadata[
                            "interpretability_audit"
                        ]["negative_control_execution"].items()
                    },
                )
                if metadata["interpretability_audit"] is not None
                else None
            ),
            long_computation_receipt=(
                LongComputationReceipt(**metadata["long_computation_receipt"])
                if metadata["long_computation_receipt"] is not None
                else None
            ),
            feature_stability_method=str(metadata["feature_stability_method"]),
        )
        if not math.isclose(
            intervention.feature_stability,
            float(metadata["aligned_feature_stability"]),
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise FrozenArtifactError("stored SAE feature stability is inconsistent")
        return intervention
    except (KeyError, TypeError, ValueError, RuntimeError, DataValidationError) as exc:
        if isinstance(exc, FrozenArtifactError):
            raise
        raise FrozenArtifactError(f"invalid SAE-intervention artifact: {exc}") from exc


@dataclass(frozen=True, slots=True)
class ActivationBatch:
    question_ids: tuple[str, ...]
    activations: Tensor
    outcomes: tuple[Outcome, ...]
    group_ids: tuple[str, ...]
    capture_receipts: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        identifiers = tuple(value.strip() for value in self.question_ids)
        if any(not value for value in identifiers) or len(set(identifiers)) != len(identifiers):
            raise DataValidationError("activation-batch IDs must be non-empty and unique")
        groups = tuple(value.strip() for value in self.group_ids)
        if len(groups) != len(identifiers) or any(not value for value in groups):
            raise DataValidationError("activation-batch group IDs must align with rows")
        values = _matrix(self.activations)
        if values.shape[0] != len(identifiers) or len(identifiers) != len(self.outcomes):
            raise DataValidationError("activation-batch rows differ")
        receipts = tuple(MappingProxyType(dict(value)) for value in self.capture_receipts)
        if receipts and len(receipts) != len(identifiers):
            raise DataValidationError("activation capture receipts must align with rows")
        object.__setattr__(self, "question_ids", identifiers)
        object.__setattr__(self, "group_ids", groups)
        object.__setattr__(self, "activations", values)
        object.__setattr__(self, "outcomes", tuple(Outcome(value) for value in self.outcomes))
        object.__setattr__(self, "capture_receipts", receipts)


def activation_capture_execution_receipt_body(
    *,
    question_id: str,
    group_id: str,
    outcome: Outcome,
    rendered_prompt_sha256: str,
    activation_sha256: str,
    feature_schema: ActivationFeatureSchema,
    runtime_artifact_sha256: str,
    execution_public_key: str,
    source_question_bundle_sha256: str,
    dtype: str,
    label_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    body = {
        "receipt_kind": (
            "e7-native-activation-capture-v2"
            if feature_schema.partition == "T-steer"
            else "e7-native-activation-capture-v1"
        ),
        "question_id": question_id,
        "group_id": group_id,
        "outcome": Outcome(outcome).value,
        "rendered_prompt_sha256": rendered_prompt_sha256,
        "activation_sha256": activation_sha256,
        "feature_schema": feature_schema.to_dict(),
        "runtime_artifact_sha256": runtime_artifact_sha256,
        "execution_public_key": execution_public_key,
        "source_question_bundle_sha256": source_question_bundle_sha256,
        "dtype": dtype,
    }
    if feature_schema.partition == "T-steer":
        if not isinstance(label_evidence, Mapping):
            raise DataValidationError("T-steer capture lacks native generation labels")
        evidence = dict(label_evidence)
        expected_keys = {
            "raw_output",
            "normalized_answer",
            "aliases",
            "source_question_sha256",
        }
        aliases = evidence.get("aliases")
        raw_output = evidence.get("raw_output")
        if (
            set(evidence) != expected_keys
            or not isinstance(raw_output, str)
            or not isinstance(aliases, list)
            or not aliases
            or any(type(value) is not str or not value for value in aliases)
            or evidence.get("normalized_answer") != normalize_answer(raw_output)
            or deterministic_short_answer_grade(raw_output, tuple(aliases))
            is not Outcome(outcome)
            or not isinstance(evidence.get("source_question_sha256"), str)
            or _SHA256.fullmatch(evidence["source_question_sha256"]) is None
        ):
            raise DataValidationError("T-steer native generation label is invalid")
        body["label_evidence"] = evidence
    elif label_evidence is not None:
        raise DataValidationError("non-T-steer activation cannot contain label evidence")
    return body


def _verify_activation_capture_receipt(
    receipt: Mapping[str, Any],
    *,
    question_id: str,
    group_id: str,
    outcome: Outcome,
    activation: NDArray[np.generic],
    feature_schema: ActivationFeatureSchema,
    runtime_artifact_sha256: str,
    execution_public_key: str,
    source_question_bundle_sha256: str,
    dtype: str,
) -> None:
    body = receipt.get("body")
    signature = receipt.get("signature")
    if (
        not isinstance(body, Mapping)
        or type(signature) is not str
        or re.fullmatch(r"[0-9a-f]{128}", signature) is None
    ):
        raise DataValidationError("native activation capture receipt is invalid")
    expected = activation_capture_execution_receipt_body(
        question_id=question_id,
        group_id=group_id,
        outcome=outcome,
        rendered_prompt_sha256=str(body.get("rendered_prompt_sha256", "")),
        activation_sha256=hashlib.sha256(
            np.ascontiguousarray(activation).tobytes(order="C")
        ).hexdigest(),
        feature_schema=feature_schema,
        runtime_artifact_sha256=runtime_artifact_sha256,
        execution_public_key=execution_public_key,
        source_question_bundle_sha256=source_question_bundle_sha256,
        dtype=dtype,
        label_evidence=(
            body.get("label_evidence")
            if feature_schema.partition == "T-steer"
            else None
        ),
    )
    if canonical_json(body) != canonical_json(expected) or not _SHA256.fullmatch(
        expected["rendered_prompt_sha256"]
    ):
        raise DataValidationError("native activation capture body differs")
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(execution_public_key)).verify(
            bytes.fromhex(signature), canonical_json(expected).encode()
        )
    except (InvalidSignature, ValueError) as exc:
        raise DataValidationError("native activation capture signature is invalid") from exc


def _write_memmap(path: Path, values: NDArray[np.generic]) -> None:
    mapped = np.lib.format.open_memmap(  # type: ignore[no-untyped-call]
        path, mode="w+", dtype=values.dtype, shape=values.shape
    )
    mapped[...] = values
    mapped.flush()
    del mapped


def activation_shard_execution_receipt_body(
    *,
    entry: Mapping[str, Any],
    feature_schema: ActivationFeatureSchema,
    runtime_artifact_sha256: str,
    execution_public_key: str,
    source_question_bundle_sha256: str,
) -> dict[str, Any]:
    """Canonical receipt signed by the native activation-capture runtime."""

    shard = {
        name: entry[name]
        for name in (
            "rows",
            "activations",
            "activations_sha256",
            "outcomes",
            "outcomes_sha256",
            "records",
            "records_sha256",
        )
    }
    return {
        "receipt_kind": "e7-activation-shard-capture-v1",
        "feature_schema": feature_schema.to_dict(),
        "runtime_artifact_sha256": runtime_artifact_sha256,
        "execution_public_key": execution_public_key,
        "source_question_bundle_sha256": source_question_bundle_sha256,
        "shard": shard,
    }


def write_activation_corpus(
    directory: str | Path,
    batches: Iterable[ActivationBatch],
    *,
    feature_schema: ActivationFeatureSchema,
    shard_rows: int,
    dtype: str = "float16",
    runtime_artifact_sha256: str | None = None,
    execution_public_key: str | None = None,
    source_question_bundle_sha256: str | None = None,
    capture_signer: Callable[[Mapping[str, Any]], str] | None = None,
) -> str:
    destination = validate_active_study_artifact_paths(
        {"E7 activation corpus": directory}
    )["E7 activation corpus"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FrozenArtifactError(f"refusing to overwrite activation corpus: {destination}")
    width = feature_schema.width
    if shard_rows <= 0:
        raise DataValidationError("activation corpus shard_rows must be positive")
    if feature_schema.partition not in {"T-steer", "sae-train", "sae-validation"}:
        raise DataValidationError("activation corpus uses an unsupported E7 partition")
    if dtype not in {"float16", "float32"}:
        raise DataValidationError("activation corpus dtype must be float16 or float32")
    provenance = (
        runtime_artifact_sha256,
        execution_public_key,
        source_question_bundle_sha256,
    )
    signed = any(value is not None for value in provenance) or capture_signer is not None
    if feature_schema.partition == "T-steer" and not signed:
        raise DataValidationError("T-steer activation corpus must be native-signed")
    if signed and (
        not all(
            type(value) is str and _SHA256.fullmatch(value) is not None
            for value in provenance
        )
        or capture_signer is None
    ):
        raise DataValidationError(
            "signed activation capture requires its complete runtime provenance"
        )
    numpy_dtype = np.float16 if dtype == "float16" else np.float32
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    buffered_values = torch.empty(0, width)
    buffered_ids: list[str] = []
    buffered_groups: list[str] = []
    buffered_outcomes: list[Outcome] = []
    buffered_receipts: list[Mapping[str, Any]] = []
    seen_ids: set[str] = set()
    manifest_shards: list[dict[str, Any]] = []

    def flush(row_count: int) -> None:
        nonlocal buffered_values, buffered_ids, buffered_groups, buffered_outcomes
        nonlocal buffered_receipts
        index = len(manifest_shards)
        values = buffered_values[:row_count].numpy().astype(numpy_dtype, copy=False)
        codes = np.asarray(
            [_OUTCOME_TO_CODE[value] for value in buffered_outcomes[:row_count]], dtype=np.int8
        )
        identifiers = buffered_ids[:row_count]
        groups = buffered_groups[:row_count]
        activation_name = f"activations-{index:05d}.npy"
        outcome_name = f"outcomes-{index:05d}.npy"
        record_name = f"records-{index:05d}.json"
        _write_memmap(stage / activation_name, values)
        _write_memmap(stage / outcome_name, codes)
        records = [
            {
                "question_id": identifier,
                "group_id": group,
                **(
                    {"capture_receipt": dict(buffered_receipts[row])}
                    if signed
                    else {}
                ),
            }
            for row, (identifier, group) in enumerate(
                zip(identifiers, groups, strict=True)
            )
        ]
        (stage / record_name).write_text(json.dumps(records) + "\n", encoding="utf-8")
        entry = {
            "rows": row_count,
            "activations": activation_name,
            "activations_sha256": sha256_file(stage / activation_name),
            "outcomes": outcome_name,
            "outcomes_sha256": sha256_file(stage / outcome_name),
            "records": record_name,
            "records_sha256": sha256_file(stage / record_name),
        }
        if signed:
            assert capture_signer is not None
            assert runtime_artifact_sha256 is not None
            assert execution_public_key is not None
            assert source_question_bundle_sha256 is not None
            entry["execution_signature"] = capture_signer(
                activation_shard_execution_receipt_body(
                    entry=entry,
                    feature_schema=feature_schema,
                    runtime_artifact_sha256=runtime_artifact_sha256,
                    execution_public_key=execution_public_key,
                    source_question_bundle_sha256=source_question_bundle_sha256,
                )
            )
        manifest_shards.append(entry)
        buffered_values = buffered_values[row_count:]
        buffered_ids = buffered_ids[row_count:]
        buffered_groups = buffered_groups[row_count:]
        buffered_outcomes = buffered_outcomes[row_count:]
        buffered_receipts = buffered_receipts[row_count:]

    try:
        for batch in batches:
            if batch.activations.shape[1] != width:
                raise DataValidationError("activation batch width differs from corpus width")
            duplicates = seen_ids & set(batch.question_ids)
            if duplicates:
                raise DataValidationError(
                    f"duplicate activation-corpus IDs: {sorted(duplicates)[:3]}"
                )
            seen_ids.update(batch.question_ids)
            if signed:
                assert runtime_artifact_sha256 is not None
                assert execution_public_key is not None
                assert source_question_bundle_sha256 is not None
                if len(batch.capture_receipts) != len(batch.question_ids):
                    raise DataValidationError(
                        "signed activation corpus requires native row receipts"
                    )
                cast_values = batch.activations.numpy().astype(numpy_dtype, copy=False)
                for row, receipt in enumerate(batch.capture_receipts):
                    _verify_activation_capture_receipt(
                        receipt,
                        question_id=batch.question_ids[row],
                        group_id=batch.group_ids[row],
                        outcome=batch.outcomes[row],
                        activation=cast_values[row],
                        feature_schema=feature_schema,
                        runtime_artifact_sha256=runtime_artifact_sha256,
                        execution_public_key=execution_public_key,
                        source_question_bundle_sha256=source_question_bundle_sha256,
                        dtype=dtype,
                    )
            buffered_values = torch.cat((buffered_values, batch.activations), dim=0)
            buffered_ids.extend(batch.question_ids)
            buffered_groups.extend(batch.group_ids)
            buffered_outcomes.extend(batch.outcomes)
            buffered_receipts.extend(batch.capture_receipts)
            while buffered_values.shape[0] >= shard_rows:
                flush(shard_rows)
        if buffered_values.shape[0]:
            flush(buffered_values.shape[0])
        if not manifest_shards:
            raise DataValidationError("activation corpus cannot be empty")
        fingerprint_body = {
            "feature_schema": feature_schema.to_dict(),
            "width": width,
            "dtype": dtype,
            "total_rows": len(seen_ids),
            "shards": manifest_shards,
        }
        if signed:
            fingerprint_body.update(
                {
                    "runtime_artifact_sha256": runtime_artifact_sha256,
                    "execution_public_key": execution_public_key,
                    "source_question_bundle_sha256": source_question_bundle_sha256,
                }
            )
        data_fingerprint = stable_hash(fingerprint_body)
        manifest_body = {
            "schema_version": 2 if signed else 1,
            **fingerprint_body,
            "data_fingerprint": data_fingerprint,
        }
        manifest = {**manifest_body, "manifest_digest": stable_hash(manifest_body)}
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(stage, destination)
        return data_fingerprint
    finally:
        if stage.exists():
            shutil.rmtree(stage)


@dataclass(frozen=True, slots=True)
class ActivationShard:
    question_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    activations: NDArray[np.floating[Any]]
    outcomes: tuple[Outcome, ...]


@dataclass(frozen=True, slots=True)
class ActivationCorpus:
    directory: Path
    width: int
    dtype: str
    data_fingerprint: str
    total_rows: int
    feature_schema: ActivationFeatureSchema
    shards: tuple[MappingProxyType[str, Any], ...]
    schema_version: int = 1
    runtime_artifact_sha256: str | None = None
    execution_public_key: str | None = None
    source_question_bundle_sha256: str | None = None

    def load_shard(self, index: int) -> ActivationShard:
        if not 0 <= index < len(self.shards):
            raise DataValidationError("activation-corpus shard index is out of range")
        entry = self.shards[index]
        activations = np.load(self.directory / entry["activations"], mmap_mode="r")
        codes = np.load(self.directory / entry["outcomes"], mmap_mode="r")
        records = json.loads((self.directory / entry["records"]).read_text(encoding="utf-8"))
        identifiers = tuple(str(record["question_id"]) for record in records)
        groups = tuple(str(record["group_id"]) for record in records)
        outcomes = tuple(_CODE_TO_OUTCOME[int(code)] for code in codes)
        return ActivationShard(identifiers, groups, activations, outcomes)

    def iter_shards(self) -> Iterator[ActivationShard]:
        for index in range(len(self.shards)):
            yield self.load_shard(index)

    def all_group_ids(self) -> set[str]:
        return {group for shard in self.iter_shards() for group in shard.group_ids}

    def all_question_ids(self) -> set[str]:
        return {
            question_id
            for shard in self.iter_shards()
            for question_id in shard.question_ids
        }


def _safe_artifact_name(value: Any, prefix: str) -> str:
    name = str(value)
    if Path(name).name != name or not name.startswith(prefix):
        raise FrozenArtifactError(f"unsafe activation-corpus artifact name: {name}")
    return name


def load_activation_corpus(
    directory: str | Path, *, expected_data_fingerprint: str | None = None
) -> ActivationCorpus:
    source = Path(directory)
    try:
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read activation-corpus manifest: {exc}") from exc
    digest = manifest.pop("manifest_digest", None)
    if digest != stable_hash(manifest):
        raise FrozenArtifactError("activation-corpus manifest digest mismatch")
    schema_version = manifest.get("schema_version")
    if schema_version not in {1, 2}:
        raise FrozenArtifactError("unsupported activation-corpus schema version")
    fingerprint = manifest.get("data_fingerprint")
    if expected_data_fingerprint is not None and fingerprint != expected_data_fingerprint:
        raise FrozenArtifactError("activation corpus has a different data fingerprint")
    try:
        width, total_rows = int(manifest["width"]), int(manifest["total_rows"])
        dtype = str(manifest["dtype"])
        feature_schema = ActivationFeatureSchema.from_dict(manifest["feature_schema"])
        runtime_artifact_sha256 = manifest.get("runtime_artifact_sha256")
        execution_public_key = manifest.get("execution_public_key")
        source_question_bundle_sha256 = manifest.get(
            "source_question_bundle_sha256"
        )
        provenance = (
            runtime_artifact_sha256,
            execution_public_key,
            source_question_bundle_sha256,
        )
        if schema_version == 2 and not all(
            type(value) is str and _SHA256.fullmatch(value) is not None
            for value in provenance
        ):
            raise FrozenArtifactError("activation-corpus runtime provenance is invalid")
        if schema_version == 1 and any(value is not None for value in provenance):
            raise FrozenArtifactError("legacy activation corpus contains runtime provenance")
        if width <= 0 or total_rows <= 0 or dtype not in {"float16", "float32"}:
            raise FrozenArtifactError("invalid activation-corpus shape or dtype")
        entries: list[MappingProxyType[str, Any]] = []
        observed_rows = 0
        seen_ids: set[str] = set()
        seen_groups: set[str] = set()
        for raw_entry in manifest["shards"]:
            entry = dict(raw_entry)
            activation_name = _safe_artifact_name(entry["activations"], "activations-")
            outcome_name = _safe_artifact_name(entry["outcomes"], "outcomes-")
            record_name = _safe_artifact_name(entry["records"], "records-")
            for name, checksum_key in (
                (activation_name, "activations_sha256"),
                (outcome_name, "outcomes_sha256"),
                (record_name, "records_sha256"),
            ):
                if sha256_file(source / name) != entry[checksum_key]:
                    raise FrozenArtifactError(f"activation-corpus checksum mismatch: {name}")
            rows = int(entry["rows"])
            activations = np.load(source / activation_name, mmap_mode="r")
            codes = np.load(source / outcome_name, mmap_mode="r")
            records = json.loads((source / record_name).read_text(encoding="utf-8"))
            if activations.shape != (rows, width) or codes.shape != (rows,) or len(records) != rows:
                raise FrozenArtifactError("activation-corpus shard shapes differ from manifest")
            if str(activations.dtype) != dtype or codes.dtype != np.int8:
                raise FrozenArtifactError("activation-corpus shard dtype differs from manifest")
            if any(int(code) < 0 or int(code) >= len(_CODE_TO_OUTCOME) for code in codes):
                raise FrozenArtifactError("activation-corpus outcome code is invalid")
            identifiers_set = {str(value["question_id"]) for value in records}
            groups_set = {str(value["group_id"]) for value in records}
            if len(identifiers_set) != rows or seen_ids & identifiers_set:
                raise FrozenArtifactError("activation-corpus question IDs are duplicated")
            seen_ids.update(identifiers_set)
            seen_groups.update(groups_set)
            observed_rows += rows
            if schema_version == 2:
                signature = entry.get("execution_signature")
                if (
                    type(signature) is not str
                    or re.fullmatch(r"[0-9a-f]{128}", signature) is None
                ):
                    raise FrozenArtifactError(
                        "activation-corpus shard lacks its runtime signature"
                    )
                assert isinstance(runtime_artifact_sha256, str)
                assert isinstance(execution_public_key, str)
                assert isinstance(source_question_bundle_sha256, str)
                for row, record in enumerate(records):
                    receipt = record.get("capture_receipt")
                    if not isinstance(receipt, Mapping):
                        raise FrozenArtifactError(
                            "activation-corpus row lacks native capture evidence"
                        )
                    try:
                        _verify_activation_capture_receipt(
                            receipt,
                            question_id=str(record["question_id"]),
                            group_id=str(record["group_id"]),
                            outcome=_CODE_TO_OUTCOME[int(codes[row])],
                            activation=np.asarray(activations[row]),
                            feature_schema=feature_schema,
                            runtime_artifact_sha256=runtime_artifact_sha256,
                            execution_public_key=execution_public_key,
                            source_question_bundle_sha256=(
                                source_question_bundle_sha256
                            ),
                            dtype=dtype,
                        )
                    except DataValidationError as exc:
                        raise FrozenArtifactError(str(exc)) from exc
                try:
                    Ed25519PublicKey.from_public_bytes(
                        bytes.fromhex(execution_public_key)
                    ).verify(
                        bytes.fromhex(signature),
                        canonical_json(
                            activation_shard_execution_receipt_body(
                                entry=entry,
                                feature_schema=feature_schema,
                                runtime_artifact_sha256=runtime_artifact_sha256,
                                execution_public_key=execution_public_key,
                                source_question_bundle_sha256=(
                                    source_question_bundle_sha256
                                ),
                            )
                        ).encode(),
                    )
                except (InvalidSignature, ValueError) as exc:
                    raise FrozenArtifactError(
                        "activation-corpus shard runtime signature is invalid"
                    ) from exc
            entries.append(MappingProxyType(entry))
        if observed_rows != total_rows or len(seen_ids) != total_rows or not entries:
            raise FrozenArtifactError("activation-corpus total row count is invalid")
        fingerprint_body = {
            "feature_schema": feature_schema.to_dict(),
            "width": width,
            "dtype": dtype,
            "total_rows": total_rows,
            "shards": [dict(entry) for entry in entries],
        }
        if schema_version == 2:
            fingerprint_body.update(
                {
                    "runtime_artifact_sha256": runtime_artifact_sha256,
                    "execution_public_key": execution_public_key,
                    "source_question_bundle_sha256": source_question_bundle_sha256,
                }
            )
        if fingerprint != stable_hash(fingerprint_body):
            raise FrozenArtifactError("activation-corpus data fingerprint mismatch")
        if feature_schema.width != width:
            raise FrozenArtifactError("activation-corpus schema width differs from manifest")
        return ActivationCorpus(
            directory=source,
            width=width,
            dtype=dtype,
            data_fingerprint=str(fingerprint),
            total_rows=total_rows,
            feature_schema=feature_schema,
            shards=tuple(entries),
            schema_version=int(schema_version),
            runtime_artifact_sha256=(
                str(runtime_artifact_sha256)
                if runtime_artifact_sha256 is not None
                else None
            ),
            execution_public_key=(
                str(execution_public_key) if execution_public_key is not None else None
            ),
            source_question_bundle_sha256=(
                str(source_question_bundle_sha256)
                if source_question_bundle_sha256 is not None
                else None
            ),
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        if isinstance(exc, FrozenArtifactError):
            raise
        raise FrozenArtifactError(f"invalid activation-corpus artifact: {exc}") from exc


def e7_resumable_chain_head(
    *,
    training_corpus: ActivationCorpus,
    validation_corpus: ActivationCorpus,
    sweep_results: Sequence[SAETrainingResult],
) -> str:
    """Bind every signed corpus shard and every persisted sweep checkpoint in order."""

    checkpoint_chain = []
    for index, result in enumerate(sweep_results):
        artifact_sha = (
            sweep_results.artifact_sha256s[index]
            if isinstance(sweep_results, SAECheckpointResultSequence)
            else None
        )
        checkpoint_chain.append(
            {
                "config_fingerprint": sae_config_fingerprint(result.config),
                "checkpoint_fingerprint": sae_checkpoint_fingerprint(result),
                "artifact_sha256": artifact_sha,
                "metadata_fingerprint": stable_hash(
                    {
                        "metrics": asdict(result.metrics),
                        "loss_history": list(result.loss_history),
                        "training_fingerprint": result.training_fingerprint,
                        "validation_fingerprint": result.validation_fingerprint,
                        "training_schema": result.training_schema.to_dict(),
                        "validation_schema": result.validation_schema.to_dict(),
                        "training_rows": result.training_rows,
                        "validation_rows": result.validation_rows,
                    }
                ),
            }
        )
    if not checkpoint_chain:
        raise DataValidationError("E7 resumable chain requires persisted sweep checkpoints")
    return stable_hash(
        {
            "schema_version": 2,
            "training_corpus_sha256": sha256_path(training_corpus.directory),
            "training_data_fingerprint": training_corpus.data_fingerprint,
            "validation_corpus_sha256": sha256_path(validation_corpus.directory),
            "validation_data_fingerprint": validation_corpus.data_fingerprint,
            "checkpoint_chain": checkpoint_chain,
        }
    )


@torch.no_grad()
def evaluate_sae_corpus(model: SparseAutoencoder, corpus: ActivationCorpus) -> SAEMetrics:
    if model.config.input_width != corpus.width:
        raise DataValidationError("SAE and activation corpus widths differ")
    residual_sum = 0.0
    value_sum = torch.zeros(corpus.width, dtype=torch.float64)
    squared_sum = 0.0
    active_sum = 0
    row_count = 0
    for shard in corpus.iter_shards():
        values = torch.from_numpy(np.array(shard.activations, dtype=np.float32, copy=True))
        reconstruction, latents = model(values)
        residual_sum += float((values - reconstruction).pow(2).sum())
        value_sum += values.double().sum(dim=0)
        squared_sum += float(values.double().pow(2).sum())
        active_sum += int((latents != 0).sum())
        row_count += values.shape[0]
    total_variance = squared_sum - float(value_sum.pow(2).sum() / row_count)
    if row_count == 0 or total_variance <= 0:
        raise DataValidationError("SAE held-out corpus has no measurable variance")
    return SAEMetrics(
        reconstruction_mse=residual_sum / (row_count * corpus.width),
        fraction_variance_explained=1 - residual_sum / total_variance,
        average_active_features=active_sum / row_count,
    )


def fit_sparse_autoencoder_corpus(
    training: ActivationCorpus,
    validation: ActivationCorpus,
    config: SAEConfig,
) -> SAETrainingResult:
    """Train shard-by-shard; at most one memmap shard is materialized at a time."""

    if training.feature_schema.partition != "sae-train":
        raise DataValidationError("streaming SAE training corpus must use sae-train")
    if validation.feature_schema.partition != "sae-validation":
        raise DataValidationError("streaming SAE validation corpus must use sae-validation")
    if not training.feature_schema.is_compatible_extraction(validation.feature_schema):
        raise DataValidationError("streaming SAE train/validation feature schemas differ")
    if training.width != config.input_width or validation.width != config.input_width:
        raise DataValidationError("streaming SAE corpus width differs from configuration")
    overlap = training.all_group_ids() & validation.all_group_ids()
    if overlap:
        raise DataValidationError(
            f"SAE train/validation semantic groups overlap: {sorted(overlap)[:3]}"
        )
    model = SparseAutoencoder(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    history: list[float] = []
    for _ in range(config.epochs):
        epoch_loss = 0.0
        rows_seen = 0
        shard_order = torch.randperm(len(training.shards), generator=generator).tolist()
        for shard_index in shard_order:
            shard = training.load_shard(int(shard_index))
            values = torch.from_numpy(np.array(shard.activations, dtype=np.float32, copy=True))
            permutation = torch.randperm(values.shape[0], generator=generator)
            for start in range(0, values.shape[0], config.batch_size):
                batch = values[permutation[start : start + config.batch_size]]
                optimizer.zero_grad(set_to_none=True)
                reconstruction, latents = model(batch)
                loss = F.mse_loss(reconstruction, batch)
                if config.sparsity is SAESparsity.L1:
                    loss = loss + config.l1_coefficient * latents.abs().mean()
                if not torch.isfinite(loss):
                    raise DataValidationError("streaming SAE training diverged")
                torch.autograd.backward(loss)
                optimizer.step()
                model.normalize_decoder()
                epoch_loss += float(loss.detach()) * batch.shape[0]
                rows_seen += batch.shape[0]
        history.append(epoch_loss / rows_seen)
    model.eval()
    return SAETrainingResult(
        model=model,
        config=config,
        metrics=evaluate_sae_corpus(model, validation),
        loss_history=tuple(history),
        training_fingerprint=training.data_fingerprint,
        validation_fingerprint=validation.data_fingerprint,
        training_schema=training.feature_schema,
        validation_schema=validation.feature_schema,
        training_rows=training.total_rows,
        validation_rows=validation.total_rows,
    )

"""Verified E3 intervention resolution and one-row native VLLM execution."""

from __future__ import annotations

import hashlib
import io
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from mfh.contracts import ActivationSite, Outcome, PromptSpec, Question, TokenScope
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.evaluation.grading import deterministic_short_answer_grade, triviaqa_scores
from mfh.experiments.e3_construction import (
    load_verified_e3_construction_snapshot,
    verify_e3_vector_bundle,
)
from mfh.experiments.e3_control_materials import verify_e3_fixed_control_materials
from mfh.experiments.e3_controls import verify_e3_shuffled_control_bundle
from mfh.experiments.e3_schedule import (
    E3Condition,
    E3Protocol,
    e3_alpha_conditions,
    e3_control_conditions,
    e3_cross_prompt_conditions,
    e3_final_conditions,
    e3_geometry_conditions,
    e3_p3_conditions,
    e3_scope_conditions,
    select_e3_screen_questions,
)
from mfh.experiments.e3_selection import VerifiedE3StageSelection
from mfh.inference.vllm_runtime import (
    VllmGenerationOutput,
    VllmInterventionState,
    VllmRenderedPrompt,
)
from mfh.provenance import stable_hash

_PROMPTS = ("P0-neutral", "P2-calibrated-abstention")
_EXTRACTIONS = ("M1-R", "M1-P")
_P3_FINGERPRINT = stable_hash(
    {
        "prompt_id": "P3-forced-answer",
        "text": "Give your best short answer even when uncertain. Do not abstain.",
        "permits_abstention": False,
        "deployment_eligible": False,
    }
)
_VERIFIED_ASSET_TOKEN = object()
_VERIFIED_ASSETS: dict[int, Mapping[str, Any]] = {}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file_stream(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_snapshot(path: Path) -> Mapping[str, str]:
    source = path.resolve()
    if path.is_symlink() or not source.is_dir():
        raise FrozenArtifactError("E3 execution artifact directory is invalid")
    values: dict[str, str] = {}
    try:
        for item in sorted(source.rglob("*")):
            if item.is_symlink() or (not item.is_file() and not item.is_dir()):
                raise FrozenArtifactError("E3 execution artifact inventory is invalid")
            if item.is_file():
                values[item.relative_to(source).as_posix()] = _sha256_file_stream(item)
    except OSError as exc:
        raise FrozenArtifactError(f"cannot snapshot E3 execution artifacts: {exc}") from exc
    if not values:
        raise FrozenArtifactError("E3 execution artifact directory is empty")
    return MappingProxyType(values)


def _directory_stat_snapshot(path: Path) -> Mapping[str, tuple[int, ...]]:
    source = path.resolve()
    if path.is_symlink() or not source.is_dir():
        raise FrozenArtifactError("E3 execution artifact directory is invalid")
    values: dict[str, tuple[int, ...]] = {}
    try:
        for item in sorted(source.rglob("*")):
            if item.is_symlink() or (not item.is_file() and not item.is_dir()):
                raise FrozenArtifactError("E3 execution artifact inventory is invalid")
            if item.is_file():
                stat = item.stat()
                values[item.relative_to(source).as_posix()] = (
                    stat.st_dev,
                    stat.st_ino,
                    stat.st_mode,
                    stat.st_size,
                    stat.st_mtime_ns,
                    stat.st_ctime_ns,
                )
    except OSError as exc:
        raise FrozenArtifactError(f"cannot stat E3 execution artifacts: {exc}") from exc
    if not values:
        raise FrozenArtifactError("E3 execution artifact directory is empty")
    return MappingProxyType(values)


def _immutable_array(value: Any, *, dtype: Any) -> np.ndarray[Any, Any]:
    source = np.asarray(value, dtype=dtype)
    payload = source.tobytes(order="C")
    result = np.frombuffer(payload, dtype=source.dtype).reshape(source.shape)
    result.setflags(write=False)
    return result


def _array_digest(value: np.ndarray[Any, Any]) -> str:
    return stable_hash(
        {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": _sha256_bytes(value.tobytes(order="C")),
        }
    )


def _immutable_root(value: np.ndarray[Any, Any]) -> object:
    root: object = value
    while isinstance(root, np.ndarray) and root.base is not None:
        root = root.base
    return root


def _question_fingerprint(value: Question) -> str:
    return stable_hash(
        {
            "question_id": value.question_id,
            "benchmark": value.benchmark,
            "text": value.text,
            "aliases": list(value.aliases),
            "split": value.split,
            "entities": list(value.entities),
            "metadata": dict(value.metadata),
        }
    )


def _prompt_fingerprint(value: PromptSpec) -> str:
    return stable_hash(
        {
            "prompt_id": value.prompt_id,
            "text": value.text,
            "permits_abstention": value.permits_abstention,
            "deployment_eligible": value.deployment_eligible,
        }
    )


def _load_npz_snapshot(
    path: Path, *, expected_sha256: str, names: frozenset[str]
) -> Mapping[str, np.ndarray[Any, Any]]:
    try:
        payload = path.read_bytes()
        if _sha256_bytes(payload) != expected_sha256:
            raise FrozenArtifactError("E3 execution tensor snapshot changed after verification")
        with np.load(io.BytesIO(payload), allow_pickle=False) as values:
            if set(values.files) != names:
                raise DataValidationError("E3 execution tensor inventory differs")
            arrays = {name: values[name].copy() for name in names}
    except (OSError, ValueError, DataValidationError) as exc:
        raise FrozenArtifactError(f"cannot load E3 execution tensors: {exc}") from exc
    return MappingProxyType(arrays)


def _open_npy_snapshot(path: Path, *, expected_sha256: str) -> np.ndarray[Any, Any]:
    try:
        with path.open("rb") as handle:
            digest = hashlib.sha256()
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
            if digest.hexdigest() != expected_sha256:
                raise FrozenArtifactError(
                    "E3 fixed-control tensor changed after verification"
                )
            handle.seek(0)
            version = np.lib.format.read_magic(handle)  # type: ignore[no-untyped-call]
            if version == (1, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(  # type: ignore[no-untyped-call]
                    handle
                )
            elif version in {(2, 0), (3, 0)}:
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(  # type: ignore[no-untyped-call]
                    handle
                )
            else:
                raise DataValidationError("unsupported E3 fixed-control NPY version")
            offset = handle.tell()
            mapped = np.memmap(
                handle,
                dtype=dtype,
                mode="r",
                offset=offset,
                shape=shape,
                order="F" if fortran_order else "C",
            )
            values = np.asarray(mapped).copy()
    except (OSError, ValueError, DataValidationError) as exc:
        raise FrozenArtifactError(f"cannot open E3 fixed-control snapshot: {exc}") from exc
    values.setflags(write=False)
    return values


def _expected_stage_conditions(
    stage: str,
    *,
    selection: VerifiedE3StageSelection | None,
    protocol: E3Protocol,
) -> tuple[E3Condition, ...]:
    if stage == "geometry":
        if selection is not None:
            raise DataValidationError("E3 geometry cannot consume a selection")
        return e3_geometry_conditions(protocol)
    if selection is None:
        raise DataValidationError("E3 post-geometry execution requires verified selection")
    selection.assert_current()
    expected_predecessor = (
        "geometry" if stage == "alpha" else "alpha" if stage == "scope" else "scope"
    )
    if selection.stage != expected_predecessor or selection.falsified:
        raise DataValidationError("E3 execution predecessor selection differs")
    builders = {
        "alpha": e3_alpha_conditions,
        "scope": e3_scope_conditions,
        "controls": e3_control_conditions,
        "cross-prompt": e3_cross_prompt_conditions,
        "P3-diagnostic": e3_p3_conditions,
        "final": e3_final_conditions,
    }
    try:
        builder = builders[stage]
    except KeyError as exc:
        raise DataValidationError("E3 execution stage is invalid") from exc
    return builder(selection.selected, protocol=protocol)


class E3ExecutionRuntime(Protocol):
    def runtime_identity(self) -> Mapping[str, Any]: ...

    def render_prompt(
        self,
        prompt: PromptSpec,
        question: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> VllmRenderedPrompt: ...

    def generate(
        self, rendered: VllmRenderedPrompt, *, max_new_tokens: int
    ) -> VllmGenerationOutput: ...

    def generate_with_interventions(
        self,
        rendered: VllmRenderedPrompt,
        *,
        max_new_tokens: int,
        intervention_states: Mapping[
            tuple[int, ActivationSite], VllmInterventionState
        ],
    ) -> VllmGenerationOutput: ...

    def standardized_intervention_state(
        self,
        direction: np.ndarray[Any, Any],
        *,
        standardized_alpha: float,
        reference_rms: float,
        token_scope: TokenScope,
        decay: float = 0.0,
    ) -> VllmInterventionState: ...


@dataclass(frozen=True, slots=True)
class E3ResolvedIntervention:
    direction: np.ndarray[Any, Any]
    direction_sha256: str
    extraction_method: str
    training_prompt_id: str
    source_layer: int
    source_site: ActivationSite
    target_layer: int
    target_site: ActivationSite
    reference_rms: float
    standardized_alpha: float
    raw_alpha: float
    token_scope: TokenScope
    decay: float
    control: str | None

    def __post_init__(self) -> None:
        direction = _immutable_array(self.direction, dtype=np.float32)
        norm = float(np.linalg.norm(direction)) if direction.ndim == 1 else math.nan
        if (
            direction.ndim != 1
            or direction.size == 0
            or not np.isfinite(direction).all()
            or not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-6)
            or self.direction_sha256 != _sha256_bytes(direction.tobytes(order="C"))
            or self.extraction_method not in _EXTRACTIONS
            or self.training_prompt_id not in _PROMPTS
            or type(self.source_layer) is not int
            or type(self.target_layer) is not int
            or not isinstance(self.source_site, ActivationSite)
            or not isinstance(self.target_site, ActivationSite)
            or type(self.reference_rms) is not float
            or not math.isfinite(self.reference_rms)
            or self.reference_rms <= 0
            or type(self.standardized_alpha) is not float
            or not math.isfinite(self.standardized_alpha)
            or type(self.raw_alpha) is not float
            or not math.isclose(
                self.raw_alpha,
                self.standardized_alpha * self.reference_rms,
                rel_tol=0,
                abs_tol=1e-12,
            )
            or not isinstance(self.token_scope, TokenScope)
            or type(self.decay) is not float
            or not math.isfinite(self.decay)
            or self.decay < 0
        ):
            raise DataValidationError("E3 resolved intervention is invalid")
        object.__setattr__(self, "direction", direction)

    def to_trace(self) -> dict[str, Any]:
        return {
            "direction_sha256": self.direction_sha256,
            "extraction_method": self.extraction_method,
            "training_prompt_id": self.training_prompt_id,
            "source_layer": self.source_layer,
            "source_site": self.source_site.value,
            "target_layer": self.target_layer,
            "target_site": self.target_site.value,
            "reference_rms": self.reference_rms,
            "standardized_alpha": self.standardized_alpha,
            "raw_alpha": self.raw_alpha,
            "token_scope": self.token_scope.value,
            "decay": self.decay,
            "control": self.control,
        }


@dataclass(frozen=True, slots=True)
class E3ExecutionResult:
    condition_id: str
    question_id: str
    rendered_prompt_sha256: str
    prompt_token_ids_sha256: str
    raw_output: str
    output_token_ids: tuple[int, ...]
    outcome: Outcome
    exact_match: float
    token_f1: float
    generation_latency_seconds: float
    input_tokens: int
    output_tokens: int
    stop_type: str
    peak_memory_bytes: int
    intervention_trace: Mapping[str, Any] | None
    hook_applications: int
    actual_delta_norm: float

    def __post_init__(self) -> None:
        digests = (self.condition_id, self.rendered_prompt_sha256, self.prompt_token_ids_sha256)
        if (
            any(
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in digests
            )
            or type(self.question_id) is not str
            or not self.question_id
            or type(self.raw_output) is not str
            or type(self.output_token_ids) is not tuple
            or any(type(value) is not int or value < 0 for value in self.output_token_ids)
            or not isinstance(self.outcome, Outcome)
            or type(self.exact_match) is not float
            or type(self.token_f1) is not float
            or not 0 <= self.exact_match <= 1
            or not 0 <= self.token_f1 <= 1
            or type(self.hook_applications) is not int
            or self.hook_applications < 0
            or type(self.actual_delta_norm) is not float
            or not math.isfinite(self.actual_delta_norm)
            or self.actual_delta_norm < 0
            or type(self.generation_latency_seconds) is not float
            or not math.isfinite(self.generation_latency_seconds)
            or self.generation_latency_seconds < 0
            or type(self.input_tokens) is not int
            or self.input_tokens < 0
            or type(self.output_tokens) is not int
            or self.output_tokens != len(self.output_token_ids)
            or type(self.stop_type) is not str
            or not self.stop_type
            or type(self.peak_memory_bytes) is not int
            or self.peak_memory_bytes < 0
            or (self.intervention_trace is None and self.hook_applications != 0)
            or (
                self.intervention_trace is not None
                and self.hook_applications == 0
                and self.intervention_trace.get("control") != "zero-hook"
                and self.intervention_trace.get("standardized_alpha") != 0.0
            )
        ):
            raise DataValidationError("E3 execution result is invalid")
        if self.intervention_trace is not None:
            object.__setattr__(
                self,
                "intervention_trace",
                MappingProxyType(dict(self.intervention_trace)),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "question_id": self.question_id,
            "rendered_prompt_sha256": self.rendered_prompt_sha256,
            "prompt_token_ids_sha256": self.prompt_token_ids_sha256,
            "raw_output": self.raw_output,
            "output_token_ids": list(self.output_token_ids),
            "outcome": self.outcome.value,
            "exact_match": self.exact_match,
            "token_f1": self.token_f1,
            "generation_latency_seconds": self.generation_latency_seconds,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "stop_type": self.stop_type,
            "peak_memory_bytes": self.peak_memory_bytes,
            "intervention_trace": (
                dict(self.intervention_trace)
                if self.intervention_trace is not None
                else None
            ),
            "hook_applications": self.hook_applications,
            "actual_delta_norm": self.actual_delta_norm,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        question: Question,
        condition: E3Condition,
        resolved: E3ResolvedIntervention | None,
        expected_rendered_prompt_sha256: str,
        expected_prompt_token_ids_sha256: str,
    ) -> E3ExecutionResult:
        expected = {
            "condition_id",
            "question_id",
            "rendered_prompt_sha256",
            "prompt_token_ids_sha256",
            "raw_output",
            "output_token_ids",
            "outcome",
            "exact_match",
            "token_f1",
            "generation_latency_seconds",
            "input_tokens",
            "output_tokens",
            "stop_type",
            "peak_memory_bytes",
            "intervention_trace",
            "hook_applications",
            "actual_delta_norm",
        }
        if type(value) is not dict or set(value) != expected:
            raise DataValidationError("E3 execution result schema differs")
        tokens = value["output_token_ids"]
        if type(tokens) is not list:
            raise DataValidationError("E3 execution token IDs must be a JSON list")
        numeric_float = (
            "exact_match",
            "token_f1",
            "generation_latency_seconds",
            "actual_delta_norm",
        )
        numeric_int = ("input_tokens", "output_tokens", "peak_memory_bytes", "hook_applications")
        if (
            any(type(value[name]) is not float for name in numeric_float)
            or any(type(value[name]) is not int for name in numeric_int)
            or (
                value["intervention_trace"] is not None
                and type(value["intervention_trace"]) is not dict
            )
        ):
            raise DataValidationError("E3 execution result numeric types differ")
        result = cls(
            condition_id=value["condition_id"],
            question_id=value["question_id"],
            rendered_prompt_sha256=value["rendered_prompt_sha256"],
            prompt_token_ids_sha256=value["prompt_token_ids_sha256"],
            raw_output=value["raw_output"],
            output_token_ids=tuple(tokens),
            outcome=Outcome(value["outcome"]),
            exact_match=value["exact_match"],
            token_f1=value["token_f1"],
            generation_latency_seconds=value["generation_latency_seconds"],
            input_tokens=value["input_tokens"],
            output_tokens=value["output_tokens"],
            stop_type=value["stop_type"],
            peak_memory_bytes=value["peak_memory_bytes"],
            intervention_trace=value["intervention_trace"],
            hook_applications=value["hook_applications"],
            actual_delta_norm=value["actual_delta_norm"],
        )
        result.validate_against(
            question=question,
            condition=condition,
            resolved=resolved,
            expected_rendered_prompt_sha256=expected_rendered_prompt_sha256,
            expected_prompt_token_ids_sha256=expected_prompt_token_ids_sha256,
        )
        return result

    def validate_against(
        self,
        *,
        question: Question,
        condition: E3Condition,
        resolved: E3ResolvedIntervention | None,
        expected_rendered_prompt_sha256: str,
        expected_prompt_token_ids_sha256: str,
    ) -> None:
        expected_outcome = deterministic_short_answer_grade(self.raw_output, question.aliases)
        expected_em, expected_f1 = triviaqa_scores(self.raw_output, question.aliases)
        if (
            self.question_id != question.question_id
            or self.condition_id != condition.condition_id
            or self.outcome is not expected_outcome
            or self.exact_match != float(expected_em)
            or self.token_f1 != float(expected_f1)
            or self.rendered_prompt_sha256 != expected_rendered_prompt_sha256
            or self.prompt_token_ids_sha256 != expected_prompt_token_ids_sha256
        ):
            raise DataValidationError("E3 result grading or row identity differs")
        if resolved is None:
            if (
                self.intervention_trace is not None
                or self.hook_applications != 0
                or self.actual_delta_norm != 0
            ):
                raise DataValidationError("E3 M0 result contains intervention evidence")
            return
        expected_trace = {
            **resolved.to_trace(),
            "hook_applications": self.hook_applications,
            "actual_delta_norm": self.actual_delta_norm,
        }
        if self.intervention_trace != expected_trace:
            raise DataValidationError("E3 intervention trace differs from resolved assets")
        if resolved.standardized_alpha == 0:
            if self.hook_applications != 0 or self.actual_delta_norm != 0:
                raise DataValidationError("E3 zero hook has a nonzero intervention effect")
        else:
            scope_limits = {
                TokenScope.FINAL_PROMPT: 1,
                TokenScope.FIRST_GENERATED: min(self.output_tokens, 1),
                TokenScope.FIRST_FOUR: min(self.output_tokens, 4),
                TokenScope.FIRST_EIGHT: min(self.output_tokens, 8),
                TokenScope.ALL_GENERATED: self.output_tokens,
                TokenScope.EXPONENTIAL_DECAY: self.output_tokens,
            }
            expected_applications = scope_limits[resolved.token_scope]
            expected_delta = abs(resolved.raw_alpha)
            if resolved.token_scope is TokenScope.EXPONENTIAL_DECAY:
                expected_delta *= math.sqrt(
                    sum(
                        math.exp(-2 * resolved.decay * index)
                        for index in range(expected_applications)
                    )
                )
            else:
                expected_delta *= math.sqrt(expected_applications)
            if (
                expected_applications <= 0
                or self.hook_applications != expected_applications
                or not math.isclose(
                    self.actual_delta_norm,
                    expected_delta,
                    rel_tol=1e-6,
                    abs_tol=1e-8,
                )
            ):
                raise DataValidationError(
                    "E3 active hook applications or intervention magnitude differ"
                )


@dataclass(frozen=True, slots=True)
class E3ExecutionAssets:
    directions: np.ndarray[Any, Any]
    reference_rms: np.ndarray[Any, Any]
    shuffled_directions: np.ndarray[Any, Any] | None
    construction_directory: Path
    vector_bundle_directory: Path
    fixed_control_directory: Path | None
    fixed_random_directions: np.ndarray[Any, Any] | None
    fixed_gaussian_directions: np.ndarray[Any, Any] | None
    dev_question_ids: tuple[str, ...]
    fixed_control_metadata_digest: str | None
    dev_question_ids_digest: str | None
    artifact_identity: Mapping[str, Any]
    conditions: Mapping[str, E3Condition]
    question_fingerprints: Mapping[str, str]
    prompt_fingerprints: Mapping[str, str]
    rendered_prompt_hashes: Mapping[str, tuple[str, str]]
    artifact_snapshots: Mapping[str, Mapping[str, str]]
    scientific_eligible: bool
    protocol: E3Protocol
    _verification_token: object

    def __post_init__(self) -> None:
        directions = _immutable_array(self.directions, dtype=np.float32)
        rms = _immutable_array(self.reference_rms, dtype=np.float64)
        expected_core = (
            len(_PROMPTS),
            len(_EXTRACTIONS),
            len(self.protocol.candidate_sites),
            len(self.protocol.candidate_layers),
        )
        if (
            directions.ndim != 5
            or directions.shape[:4] != expected_core
            or rms.shape != expected_core
            or not np.isfinite(directions).all()
            or not np.isfinite(rms).all()
            or np.any(rms <= 0)
        ):
            raise DataValidationError("E3 execution vector assets are invalid")
        shuffled: np.ndarray[Any, Any] | None = None
        if self.shuffled_directions is not None:
            shuffled = _immutable_array(self.shuffled_directions, dtype=np.float32)
            if (
                shuffled.shape != (len(_EXTRACTIONS), directions.shape[-1])
                or not np.isfinite(shuffled).all()
            ):
                raise DataValidationError("E3 shuffled execution vectors are invalid")
        fixed_random = self.fixed_random_directions
        fixed_gaussian = self.fixed_gaussian_directions
        if (fixed_random is None) != (fixed_gaussian is None):
            raise DataValidationError("E3 fixed execution arrays must be paired")
        if (
            fixed_random is not None
            and fixed_gaussian is not None
            and (
                fixed_random.dtype != np.float32
                or fixed_random.shape != (len(_EXTRACTIONS), directions.shape[-1])
                or fixed_gaussian.dtype != np.float32
                or fixed_gaussian.shape
                != (len(_EXTRACTIONS), len(self.dev_question_ids), directions.shape[-1])
                or not np.isfinite(fixed_random).all()
                or not np.isfinite(fixed_gaussian).all()
            )
        ):
            raise DataValidationError("E3 fixed execution arrays are invalid")
        if fixed_random is not None and fixed_gaussian is not None:
            fixed_random = _immutable_array(fixed_random, dtype=np.float32)
            fixed_gaussian = _immutable_array(fixed_gaussian, dtype=np.float32)
        object.__setattr__(self, "directions", directions)
        object.__setattr__(self, "reference_rms", rms)
        object.__setattr__(self, "shuffled_directions", shuffled)
        object.__setattr__(self, "fixed_random_directions", fixed_random)
        object.__setattr__(self, "fixed_gaussian_directions", fixed_gaussian)
        object.__setattr__(self, "dev_question_ids", tuple(self.dev_question_ids))
        object.__setattr__(
            self,
            "artifact_identity",
            MappingProxyType(dict(self.artifact_identity)),
        )
        conditions = dict(self.conditions)
        question_fingerprints = dict(self.question_fingerprints)
        prompt_fingerprints = dict(self.prompt_fingerprints)
        rendered_prompt_hashes = dict(self.rendered_prompt_hashes)
        if (
            not conditions
            or any(
                condition_id != condition.condition_id
                or not isinstance(condition, E3Condition)
                for condition_id, condition in conditions.items()
            )
            or not question_fingerprints
            or not prompt_fingerprints
            or len(rendered_prompt_hashes)
            != len(conditions) * len(question_fingerprints)
            or type(self.scientific_eligible) is not bool
            or self._verification_token is not _VERIFIED_ASSET_TOKEN
        ):
            raise DataValidationError("E3 execution bindings are invalid")
        snapshots = {
            str(Path(directory).resolve()): MappingProxyType(dict(files))
            for directory, files in self.artifact_snapshots.items()
        }
        if (
            set(snapshots)
            != {
                str(Path(directory).resolve())
                for directory in self.artifact_snapshots
            }
            or str(self.construction_directory.resolve()) not in snapshots
            or str(self.vector_bundle_directory.resolve()) not in snapshots
            or any(
                not files
                or any(
                    type(name) is not str
                    or not name
                    or type(digest) is not str
                    or len(digest) != 64
                    for name, digest in files.items()
                )
                for files in snapshots.values()
            )
        ):
            raise DataValidationError("E3 execution artifact snapshots are invalid")
        object.__setattr__(self, "conditions", MappingProxyType(conditions))
        object.__setattr__(
            self, "question_fingerprints", MappingProxyType(question_fingerprints)
        )
        object.__setattr__(
            self, "prompt_fingerprints", MappingProxyType(prompt_fingerprints)
        )
        object.__setattr__(
            self,
            "rendered_prompt_hashes",
            MappingProxyType(rendered_prompt_hashes),
        )
        object.__setattr__(self, "artifact_snapshots", MappingProxyType(snapshots))

    def _tensor_values(self) -> Mapping[str, np.ndarray[Any, Any]]:
        values = {
            "directions": self.directions,
            "reference_rms": self.reference_rms,
        }
        if self.shuffled_directions is not None:
            values["shuffled_directions"] = self.shuffled_directions
        if self.fixed_random_directions is not None:
            values["fixed_random_directions"] = self.fixed_random_directions
        if self.fixed_gaussian_directions is not None:
            values["fixed_gaussian_directions"] = self.fixed_gaussian_directions
        return MappingProxyType(values)

    def _binding_digest(self) -> str:
        return stable_hash(
            {
                "artifact_identity": dict(self.artifact_identity),
                "conditions": {
                    name: value.to_dict() for name, value in self.conditions.items()
                },
                "question_fingerprints": dict(self.question_fingerprints),
                "prompt_fingerprints": dict(self.prompt_fingerprints),
                "rendered_prompt_hashes": {
                    name: list(value)
                    for name, value in self.rendered_prompt_hashes.items()
                },
                "dev_question_ids": list(self.dev_question_ids),
                "fixed_control_metadata_digest": self.fixed_control_metadata_digest,
                "dev_question_ids_digest": self.dev_question_ids_digest,
                "scientific_eligible": self.scientific_eligible,
                "protocol": self.protocol.to_dict(),
            }
        )

    def assert_authorized(self) -> Mapping[str, Any]:
        receipt = _VERIFIED_ASSETS.get(id(self))
        if receipt is None:
            raise FrozenArtifactError("E3 execution assets were not verifier-authorized")
        current = self._tensor_values()
        expected_objects = receipt["tensor_objects"]
        if (
            set(current) != set(expected_objects)
            or any(
                id(value) != expected_objects[name][0]
                or id(_immutable_root(value)) != expected_objects[name][1]
                or not isinstance(_immutable_root(value), bytes)
                or value.flags.writeable
                for name, value in current.items()
            )
            or tuple(receipt["binding_objects"])
            != (
                id(self.artifact_identity),
                id(self.conditions),
                id(self.question_fingerprints),
                id(self.prompt_fingerprints),
                id(self.rendered_prompt_hashes),
                id(self.protocol),
            )
        ):
            raise FrozenArtifactError("E3 execution assets changed in memory")
        return receipt

    def assert_current(self, *, full: bool = True) -> None:
        receipt = self.assert_authorized()
        for directory, expected in receipt["artifact_snapshots"].items():
            if full:
                observed: Mapping[str, Any] = _directory_snapshot(Path(directory))
            else:
                observed = _directory_stat_snapshot(Path(directory))
                expected = receipt["artifact_stats"][directory]
            if dict(observed) != dict(expected):
                raise FrozenArtifactError("E3 execution artifact snapshot changed")
        if full and (
            self._binding_digest() != receipt["binding_digest"]
            or {
                name: _array_digest(value)
                for name, value in self._tensor_values().items()
            }
            != dict(receipt["tensor_digests"])
        ):
            raise FrozenArtifactError("E3 execution in-memory receipt changed")

    def assert_runtime(self, runtime: E3ExecutionRuntime) -> None:
        receipt = self.assert_authorized()
        if id(runtime) != receipt["runtime_object_id"]:
            raise FrozenArtifactError("E3 execution runtime differs from verified renderer")

    @property
    def hidden_width(self) -> int:
        return int(self.directions.shape[-1])

    def _indices(
        self,
        *,
        prompt_id: str,
        extraction: str,
        site: ActivationSite,
        layer: int,
    ) -> tuple[int, int, int, int]:
        try:
            return (
                _PROMPTS.index(prompt_id),
                _EXTRACTIONS.index(extraction),
                self.protocol.candidate_sites.index(site),
                self.protocol.candidate_layers.index(layer),
            )
        except ValueError as exc:
            raise DataValidationError("E3 execution source geometry is outside assets") from exc

    def resolve(
        self,
        condition: E3Condition,
        *,
        question_id: str,
    ) -> E3ResolvedIntervention | None:
        if condition.method == "M0":
            return None
        assert (
            condition.extraction_method is not None
            and condition.training_prompt_id is not None
            and condition.layer is not None
            and condition.site is not None
            and condition.token_scope is not None
        )
        extraction = condition.extraction_method
        target_layer = condition.layer
        target_site = condition.site
        source_layer = (
            condition.source_layer if condition.source_layer is not None else target_layer
        )
        source_site = condition.source_site if condition.source_site is not None else target_site
        source_index = self._indices(
            prompt_id=condition.training_prompt_id,
            extraction=extraction,
            site=source_site,
            layer=source_layer,
        )
        target_index = self._indices(
            prompt_id=(
                condition.apply_prompt_id
                if condition.apply_prompt_id in _PROMPTS
                else condition.training_prompt_id
            ),
            extraction=extraction,
            site=target_site,
            layer=target_layer,
        )
        control = condition.control
        if control == "shuffled-label":
            if self.shuffled_directions is None:
                raise FrozenArtifactError("E3 shuffled control bundle is unavailable")
            direction = self.shuffled_directions[_EXTRACTIONS.index(extraction)]
        elif control in {"random-norm", "gaussian"}:
            if (
                self.fixed_control_directory is None
                or self.fixed_random_directions is None
                or self.fixed_gaussian_directions is None
            ):
                raise FrozenArtifactError("E3 fixed control bundle is unavailable")
            extraction_index = _EXTRACTIONS.index(extraction)
            direction = (
                self.fixed_random_directions[extraction_index]
                if control == "random-norm"
                else self.fixed_gaussian_directions[
                    extraction_index, self.dev_question_ids.index(question_id)
                ]
            )
        else:
            direction = self.directions[source_index]
        reference_rms = float(self.reference_rms[target_index])
        alpha = float(condition.standardized_alpha)
        direction = np.asarray(direction, dtype=np.float32)
        return E3ResolvedIntervention(
            direction=direction,
            direction_sha256=_sha256_bytes(direction.tobytes(order="C")),
            extraction_method=extraction,
            training_prompt_id=condition.training_prompt_id,
            source_layer=source_layer,
            source_site=source_site,
            target_layer=target_layer,
            target_site=target_site,
            reference_rms=reference_rms,
            standardized_alpha=alpha,
            raw_alpha=alpha * reference_rms,
            token_scope=condition.token_scope,
            decay=(
                float(self.protocol.exponential_decay)
                if condition.token_scope is TokenScope.EXPONENTIAL_DECAY
                else 0.0
            ),
            control=control,
        )


def load_e3_execution_assets(
    *,
    construction_directory: str | Path,
    vector_bundle_directory: str | Path,
    questions: Sequence[Question],
    prompts: Mapping[str, PromptSpec],
    scope_selection: VerifiedE3StageSelection | None = None,
    shuffled_work_directory: str | Path | None = None,
    shuffled_bundle_directory: str | Path | None = None,
    fixed_control_directory: str | Path | None = None,
    dev_questions: Sequence[Question],
    evaluation_questions: Sequence[Question],
    application_prompts: Mapping[str, PromptSpec],
    conditions: Sequence[E3Condition],
    stage: str,
    render_runtime: E3ExecutionRuntime,
    protocol: E3Protocol | None = None,
) -> E3ExecutionAssets:
    frozen_protocol = protocol or E3Protocol()
    expected_conditions = _expected_stage_conditions(
        stage, selection=scope_selection, protocol=frozen_protocol
    )
    snapshot = load_verified_e3_construction_snapshot(
        construction_directory,
        questions=questions,
        prompts=prompts,
        protocol=frozen_protocol,
    )
    if (
        scope_selection is not None
        and scope_selection.source_plan_identity != snapshot.plan["plan_identity"]
    ):
        raise DataValidationError("E3 execution selection belongs to another construction")
    construction_runtime_identity = json.loads(
        json.dumps(
            dict(snapshot.plan["runtime_identity"]), sort_keys=True, allow_nan=False
        )
    )
    if dict(render_runtime.runtime_identity()) != construction_runtime_identity:
        raise DataValidationError("E3 execution runtime differs from construction runtime")
    vector_verification = verify_e3_vector_bundle(
        vector_bundle_directory,
        work_directory=construction_directory,
        questions=questions,
        prompts=prompts,
        protocol=frozen_protocol,
    )
    vector_arrays = _load_npz_snapshot(
        Path(vector_bundle_directory) / "vectors.npz",
        expected_sha256=str(vector_verification["vectors_sha256"]),
        names=frozenset(
            {"directions", "reference_rms", "correct_counts", "incorrect_counts"}
        ),
    )
    directions = vector_arrays["directions"]
    rms = vector_arrays["reference_rms"]
    shuffled: np.ndarray[Any, Any] | None = None
    shuffled_verification: Mapping[str, Any] | None = None
    if shuffled_bundle_directory is not None or shuffled_work_directory is not None:
        if (
            scope_selection is None
            or shuffled_bundle_directory is None
            or shuffled_work_directory is None
        ):
            raise DataValidationError("E3 shuffled assets require selection, work, and bundle")
        shuffled_verification = verify_e3_shuffled_control_bundle(
            shuffled_bundle_directory,
            work_directory=shuffled_work_directory,
            construction_directory=construction_directory,
            vector_bundle_directory=vector_bundle_directory,
            questions=questions,
            prompts=prompts,
            scope_selection=scope_selection,
            protocol=frozen_protocol,
        )
        shuffled_arrays = _load_npz_snapshot(
            Path(shuffled_bundle_directory) / "directions.npz",
            expected_sha256=str(shuffled_verification["directions_sha256"]),
            names=frozenset({"directions", "correct_counts", "incorrect_counts"}),
        )
        shuffled = shuffled_arrays["directions"]
    frozen_dev = tuple(value.question_id for value in dev_questions)
    dev_by_id = {value.question_id: value for value in dev_questions}
    fixed_metadata_digest: str | None = None
    dev_digest: str | None = None
    fixed_random: np.ndarray[Any, Any] | None = None
    fixed_gaussian: np.ndarray[Any, Any] | None = None
    fixed_verification: Mapping[str, Any] | None = None
    if fixed_control_directory is not None:
        if scope_selection is None:
            raise DataValidationError("E3 fixed controls require scope selection")
        fixed_verification = verify_e3_fixed_control_materials(
            fixed_control_directory,
            construction_directory=construction_directory,
            vector_bundle_directory=vector_bundle_directory,
            questions=questions,
            prompts=prompts,
            scope_selection=scope_selection,
            dev_questions=dev_questions,
            protocol=frozen_protocol,
        )
        fixed_metadata_digest = str(fixed_verification["metadata_digest"])
        dev_digest = str(fixed_verification["dev_question_ids_digest"])
        fixed_random = _open_npy_snapshot(
            Path(fixed_control_directory) / "random_norm.npy",
            expected_sha256=str(fixed_verification["random_norm_sha256"]),
        )
        fixed_gaussian = _open_npy_snapshot(
            Path(fixed_control_directory) / "gaussian.npy",
            expected_sha256=str(fixed_verification["gaussian_sha256"]),
        )
    if int(directions.shape[-1]) != int(snapshot.plan["hidden_width"]):
        raise FrozenArtifactError("E3 execution width differs from construction")
    frozen_conditions = tuple(conditions)
    frozen_evaluation = tuple(evaluation_questions)
    condition_map = {value.condition_id: value for value in frozen_conditions}
    question_fingerprints = {
        value.question_id: _question_fingerprint(value) for value in frozen_evaluation
    }
    prompt_fingerprints = {
        name: _prompt_fingerprint(value) for name, value in application_prompts.items()
    }
    if (
        frozen_conditions != expected_conditions
        or len(condition_map) != len(frozen_conditions)
        or len(dev_by_id) != len(frozen_dev)
        or len(question_fingerprints) != len(frozen_evaluation)
        or {value.apply_prompt_id for value in frozen_conditions}
        - set(prompt_fingerprints)
        or any(
            name != prompt.prompt_id for name, prompt in application_prompts.items()
        )
    ):
        raise DataValidationError("E3 execution condition, question, or prompt bindings differ")
    screen_ids = select_e3_screen_questions(dev_questions, protocol=frozen_protocol)
    expected_evaluation_ids = (
        frozen_dev if stage == "final" else screen_ids
    )
    evaluation_by_id = {value.question_id: value for value in frozen_evaluation}
    if tuple(value.question_id for value in frozen_evaluation) != expected_evaluation_ids:
        raise DataValidationError("E3 execution questions differ from exact T-dev stage set")
    if any(
        question_id not in dev_by_id
        or _question_fingerprint(question) != _question_fingerprint(dev_by_id[question_id])
        for question_id, question in evaluation_by_id.items()
    ):
        raise DataValidationError("E3 evaluation question content differs from T-dev")
    if any(
        name in prompts and _prompt_fingerprint(prompt) != _prompt_fingerprint(prompts[name])
        for name, prompt in application_prompts.items()
    ) or (
        "P3-forced-answer" in application_prompts
        and _prompt_fingerprint(application_prompts["P3-forced-answer"])
        != _P3_FINGERPRINT
    ):
        raise DataValidationError("E3 application prompts differ from construction prompts")
    rendered_prompt_hashes: dict[str, tuple[str, str]] = {}
    for condition in frozen_conditions:
        for question_id in question_fingerprints:
            question = evaluation_by_id[question_id]
            rendered = render_runtime.render_prompt(
                application_prompts[condition.apply_prompt_id],
                question.text,
                metadata=question.metadata,
            )
            rendered_prompt_hashes[f"{condition.condition_id}:{question_id}"] = (
                rendered.sha256,
                rendered.token_ids_sha256,
            )
    artifact_directories = [
        Path(construction_directory),
        Path(vector_bundle_directory),
    ]
    if shuffled_bundle_directory is not None:
        artifact_directories.append(Path(shuffled_bundle_directory))
    if fixed_control_directory is not None:
        artifact_directories.append(Path(fixed_control_directory))
    artifact_snapshots = {
        str(directory.resolve()): _directory_snapshot(directory)
        for directory in artifact_directories
    }
    assets = E3ExecutionAssets(
        directions=directions,
        reference_rms=rms,
        shuffled_directions=shuffled,
        construction_directory=Path(construction_directory).resolve(),
        vector_bundle_directory=Path(vector_bundle_directory),
        fixed_control_directory=(
            Path(fixed_control_directory) if fixed_control_directory is not None else None
        ),
        fixed_random_directions=fixed_random,
        fixed_gaussian_directions=fixed_gaussian,
        dev_question_ids=frozen_dev,
        fixed_control_metadata_digest=fixed_metadata_digest,
        dev_question_ids_digest=dev_digest,
        artifact_identity={
            "construction_plan_identity": snapshot.plan["plan_identity"],
            "vector_data_fingerprint": vector_verification["data_fingerprint"],
            "vectors_sha256": vector_verification["vectors_sha256"],
            "shuffled_plan_identity": (
                shuffled_verification["plan_identity"]
                if shuffled_verification is not None
                else None
            ),
            "shuffled_directions_sha256": (
                shuffled_verification["directions_sha256"]
                if shuffled_verification is not None
                else None
            ),
            "fixed_control_metadata_digest": fixed_metadata_digest,
            "construction_runtime_identity": construction_runtime_identity,
            "artifact_snapshots_digest": stable_hash(
                {
                    directory: dict(files)
                    for directory, files in sorted(artifact_snapshots.items())
                }
            ),
        },
        conditions=condition_map,
        question_fingerprints=question_fingerprints,
        prompt_fingerprints=prompt_fingerprints,
        rendered_prompt_hashes=rendered_prompt_hashes,
        artifact_snapshots=artifact_snapshots,
        scientific_eligible=bool(
            snapshot.scientific_eligible
            and vector_verification["scientific_eligible"]
            and (scope_selection is None or scope_selection.scientific_eligible)
            and (
                shuffled_verification is None
                or shuffled_verification["scientific_eligible"]
            )
            and (
                fixed_verification is None
                or fixed_verification["scientific_eligible"]
            )
            and frozen_protocol.scientific_eligible
        ),
        protocol=frozen_protocol,
        _verification_token=_VERIFIED_ASSET_TOKEN,
    )
    tensor_values = assets._tensor_values()
    _VERIFIED_ASSETS[id(assets)] = MappingProxyType(
        {
            "tensor_objects": MappingProxyType(
                {
                    name: (id(value), id(_immutable_root(value)))
                    for name, value in tensor_values.items()
                }
            ),
            "tensor_digests": MappingProxyType(
                {name: _array_digest(value) for name, value in tensor_values.items()}
            ),
            "binding_objects": (
                id(assets.artifact_identity),
                id(assets.conditions),
                id(assets.question_fingerprints),
                id(assets.prompt_fingerprints),
                id(assets.rendered_prompt_hashes),
                id(assets.protocol),
            ),
            "binding_digest": assets._binding_digest(),
            "artifact_snapshots": assets.artifact_snapshots,
            "artifact_stats": MappingProxyType(
                {
                    directory: _directory_stat_snapshot(Path(directory))
                    for directory in assets.artifact_snapshots
                }
            ),
            "runtime_object_id": id(render_runtime),
        }
    )
    assets.assert_current()
    return assets


def _actual_delta(
    state: VllmInterventionState, resolved: E3ResolvedIntervention
) -> float:
    if state.applications == 0:
        return 0.0
    if (
        not math.isclose(state.alpha, resolved.raw_alpha, rel_tol=0, abs_tol=1e-12)
        or state.applications < 0
    ):
        raise DataValidationError("E3 intervention state differs from resolved strength")
    magnitude = abs(resolved.raw_alpha)
    if resolved.token_scope is TokenScope.EXPONENTIAL_DECAY:
        squared_scale = sum(
            math.exp(-2 * resolved.decay * index)
            for index in range(state.applications)
        )
        magnitude *= math.sqrt(squared_scale)
    else:
        magnitude *= math.sqrt(state.applications)
    if not math.isfinite(magnitude) or magnitude <= 0:
        raise DataValidationError("E3 active intervention magnitude is invalid")
    return magnitude


def execute_e3_condition(
    *,
    runtime: E3ExecutionRuntime,
    assets: E3ExecutionAssets,
    condition: E3Condition,
    question: Question,
    prompts: Mapping[str, PromptSpec],
    max_new_tokens: int = 48,
) -> E3ExecutionResult:
    """Execute and deterministically grade one frozen TriviaQA E3 row."""

    assets.assert_current(full=False)
    assets.assert_runtime(runtime)

    if question.benchmark != "triviaqa":
        raise DataValidationError("E3 execution is restricted to TriviaQA")
    if type(max_new_tokens) is not int or not 0 < max_new_tokens <= 48:
        raise DataValidationError("E3 max_new_tokens must be in [1, 48]")
    expected_condition = assets.conditions.get(condition.condition_id)
    if expected_condition is None or expected_condition.to_dict() != condition.to_dict():
        raise FrozenArtifactError("E3 condition is outside the frozen execution schedule")
    if assets.question_fingerprints.get(question.question_id) != _question_fingerprint(
        question
    ):
        raise FrozenArtifactError("E3 question differs from the frozen execution row")
    try:
        prompt = prompts[condition.apply_prompt_id]
    except KeyError as exc:
        raise DataValidationError("E3 application prompt is unavailable") from exc
    if assets.prompt_fingerprints.get(condition.apply_prompt_id) != _prompt_fingerprint(
        prompt
    ):
        raise FrozenArtifactError("E3 application prompt differs from the frozen schedule")
    rendered = runtime.render_prompt(prompt, question.text, metadata=question.metadata)
    render_key = f"{condition.condition_id}:{question.question_id}"
    expected_rendered = assets.rendered_prompt_hashes.get(render_key)
    if expected_rendered != (rendered.sha256, rendered.token_ids_sha256):
        raise FrozenArtifactError("E3 rendered prompt differs from frozen receipt")
    resolved = assets.resolve(condition, question_id=question.question_id)
    state: VllmInterventionState | None = None
    if resolved is None:
        generated = runtime.generate(rendered, max_new_tokens=max_new_tokens)
    else:
        state = runtime.standardized_intervention_state(
            resolved.direction,
            standardized_alpha=resolved.standardized_alpha,
            reference_rms=resolved.reference_rms,
            token_scope=resolved.token_scope,
            decay=resolved.decay,
        )
        generated = runtime.generate_with_interventions(
            rendered,
            max_new_tokens=max_new_tokens,
            intervention_states={(resolved.target_layer, resolved.target_site): state},
        )
    if generated.rendered_prompt != rendered:
        raise FrozenArtifactError("E3 runtime returned a different rendered prompt")
    outcome = deterministic_short_answer_grade(generated.text, question.aliases)
    exact_match, token_f1 = triviaqa_scores(generated.text, question.aliases)
    applications = state.applications if state is not None else 0
    actual_delta = (
        _actual_delta(state, resolved)
        if state is not None and resolved is not None
        else 0.0
    )
    trace: dict[str, Any] | None = None
    if resolved is not None:
        trace = {
            **resolved.to_trace(),
            "hook_applications": applications,
            "actual_delta_norm": actual_delta,
        }
    result = E3ExecutionResult(
        condition_id=condition.condition_id,
        question_id=question.question_id,
        rendered_prompt_sha256=rendered.sha256,
        prompt_token_ids_sha256=rendered.token_ids_sha256,
        raw_output=generated.text,
        output_token_ids=tuple(generated.token_ids),
        outcome=outcome,
        exact_match=float(exact_match),
        token_f1=float(token_f1),
        generation_latency_seconds=float(generated.latency_seconds),
        input_tokens=generated.input_tokens,
        output_tokens=generated.output_tokens,
        stop_type=generated.stop_type,
        peak_memory_bytes=generated.peak_memory_bytes,
        intervention_trace=trace,
        hook_applications=applications,
        actual_delta_norm=actual_delta,
    )
    result.validate_against(
        question=question,
        condition=condition,
        resolved=resolved,
        expected_rendered_prompt_sha256=expected_rendered[0],
        expected_prompt_token_ids_sha256=expected_rendered[1],
    )
    return result

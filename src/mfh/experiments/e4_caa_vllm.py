"""Resumable native-VLLM construction of the E4 M2 CAA residual vector bank."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from mfh.contracts import ActivationSite, Outcome, PromptSpec, Question
from mfh.data.splits import semantic_group_ids
from mfh.errors import ConfigurationError, DataValidationError, FrozenArtifactError
from mfh.experiments.e2_schedule import E2_LAYERS
from mfh.experiments.e3_construction import (
    VerifiedE3ConstructionSnapshot,
    load_verified_e3_construction_snapshot,
)
from mfh.experiments.e3_schedule import E3Protocol
from mfh.experiments.model_selection import (
    ACTIVE_MODEL_IDENTITIES,
    ACTIVE_MODEL_NAME,
    validate_active_study_artifact_paths,
)
from mfh.inference.architecture import HookKey
from mfh.inference.vllm_research import VllmTeacherForcedCubeOutput
from mfh.inference.vllm_runtime import VllmRenderedPrompt
from mfh.methods.static import load_vector_bank
from mfh.provenance import sha256_file, sha256_path, stable_hash

_PROMPT_ID = "P0-neutral"
_SITE = ActivationSite.BLOCK_OUTPUT
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MINIMUM_VRAM_BYTES = 40_000_000_000
_MAXIMUM_VRAM_BYTES = 40 * 1024**3
_ACTIVE_MODEL = ACTIVE_MODEL_IDENTITIES[ACTIVE_MODEL_NAME]
_ACTIVE_HIDDEN_WIDTH = 5_120
_WORK_FILES = frozenset({"plan.json", "pairs.jsonl", "sessions.jsonl", "checkpoints"})
_OUTPUT_FILES = frozenset(
    {
        "manifest.json",
        "plan.json",
        "pairs.jsonl",
        "sessions.jsonl",
        "accumulator.npz",
        "metadata.json",
        "vectors.safetensors",
    }
)


class M2CAARuntime(Protocol):
    def runtime_identity(self) -> Mapping[str, Any]: ...

    def render_prompt(
        self,
        prompt: PromptSpec,
        question: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> VllmRenderedPrompt: ...

    def teacher_forced_cube(
        self,
        rendered: VllmRenderedPrompt,
        response: str,
        *,
        layers: Sequence[int],
        sites: Sequence[ActivationSite],
    ) -> VllmTeacherForcedCubeOutput: ...


def _active_qwen_runtime_identity(value: object) -> bool:
    """Check the immutable model/toolchain facts inherited from scientific E3."""

    if not isinstance(value, Mapping):
        return False
    provenance = value.get("research_provenance")
    toolchain = value.get("research_toolchain")
    if not isinstance(provenance, Mapping) or not isinstance(toolchain, Mapping):
        return False
    digest_fields = (
        "verified_snapshot_digest",
        "runtime_preflight_receipt_digest",
        "runtime_policy_digest",
        "research_toolchain_digest",
    )
    return bool(
        value.get("backend") == "vllm"
        and value.get("vllm") == "0.24.0"
        and isinstance(value.get("python"), str)
        and bool(value.get("python"))
        and value.get("architecture") == "x86_64"
        and isinstance(value.get("os"), str)
        and bool(value.get("os"))
        and isinstance(value.get("gpu_name"), str)
        and "A100" in str(value.get("gpu_name"))
        and type(value.get("gpu_total_memory_bytes")) is int
        and int(value["gpu_total_memory_bytes"]) >= _MINIMUM_VRAM_BYTES
        and value.get("cuda_capability") == "8.0"
        and value.get("tensor_parallel_size") == 1
        and value.get("quantization_loader") == "modelopt_mixed"
        and value.get("quantization_config_class")
        == (
            "vllm.model_executor.layers.quantization.modelopt."
            "ModelOptMixedPrecisionConfig"
        )
        and value.get("quantization_execution")
        == "marlin-w4a16-fp8-weight-only-on-sm80"
        and value.get("model_class")
        == "vllm.model_executor.models.qwen3_5.Qwen3_5ForConditionalGeneration"
        and isinstance(value.get("tokenizer_class"), str)
        and bool(value.get("tokenizer_class"))
        and value.get("num_layers") == _ACTIVE_MODEL["num_layers"]
        and value.get("hidden_size") == _ACTIVE_HIDDEN_WIDTH
        and value.get("seed") == 17
        and value.get("model_repository") == _ACTIVE_MODEL["repository"]
        and value.get("model_revision") == _ACTIVE_MODEL["revision"]
        and value.get("model_quantization") == _ACTIVE_MODEL["quantization"]
        and value.get("model_num_layers") == _ACTIVE_MODEL["num_layers"]
        and isinstance(value.get("snapshot_sha256"), str)
        and _SHA256.fullmatch(str(value["snapshot_sha256"])) is not None
        and provenance.get("model_repository") == _ACTIVE_MODEL["repository"]
        and provenance.get("model_revision") == _ACTIVE_MODEL["revision"]
        and provenance.get("quantization") == _ACTIVE_MODEL["quantization"]
        and all(
            isinstance(provenance.get(name), str)
            and _SHA256.fullmatch(str(provenance[name])) is not None
            for name in digest_fields
        )
        and set(toolchain)
        == {"vllm", "torch", "transformers", "numpy", "nvidia_driver"}
        and all(isinstance(item, str) and item for item in toolchain.values())
    )


@dataclass(frozen=True, slots=True)
class M2CAAProtocol:
    """Frozen exact-question CAA construction policy."""

    layers: tuple[int, ...] = E2_LAYERS
    site: ActivationSite = _SITE
    source_prompt_id: str = _PROMPT_ID
    checkpoint_pairs: int = 64
    seed: int = 17
    schema_version: int = 1

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or tuple(self.layers) != E2_LAYERS
            or self.site is not _SITE
            or self.source_prompt_id != _PROMPT_ID
            or type(self.checkpoint_pairs) is not int
            or self.checkpoint_pairs <= 0
            or type(self.seed) is not int
            or self.seed != 17
        ):
            raise DataValidationError("M2 CAA protocol differs from the frozen residual design")

    @property
    def scientific_eligible(self) -> bool:
        return self == M2CAAProtocol()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "layers": list(self.layers),
            "site": self.site.value,
            "source_prompt_id": self.source_prompt_id,
            "checkpoint_pairs": self.checkpoint_pairs,
            "seed": self.seed,
            "pair_definition": (
                "same-question teacher-forced first gold alias minus the model's original "
                "incorrect response; per-response-token mean at residual block output"
            ),
        }


@dataclass(frozen=True, slots=True)
class M2CAAArtifact:
    directory: Path
    plan_identity: str
    data_fingerprint: str
    pair_count: int
    layers: tuple[int, ...]
    site: ActivationSite
    maximum_peak_memory_bytes: int
    manifest_digest: str

    def __post_init__(self) -> None:
        if (
            self.pair_count <= 0
            or self.layers != E2_LAYERS
            or self.site is not _SITE
            or self.maximum_peak_memory_bytes < 0
            or self.maximum_peak_memory_bytes > _MAXIMUM_VRAM_BYTES
            or any(
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in (
                    self.plan_identity,
                    self.data_fingerprint,
                    self.manifest_digest,
                )
            )
        ):
            raise DataValidationError("M2 CAA artifact identity is invalid")


@dataclass(slots=True)
class _Accumulator:
    processed_pairs: int
    counts: np.ndarray[Any, Any]
    difference_sums: np.ndarray[Any, Any]
    rms_elements: np.ndarray[Any, Any]
    rms_sum_squares: np.ndarray[Any, Any]


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _read_json(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read {context}: {exc}") from exc
    if type(value) is not dict:
        raise FrozenArtifactError(f"{context} must be a JSON object")
    return value


def _read_jsonl(path: Path, context: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                value = json.loads(line)
                if type(value) is not dict:
                    raise DataValidationError(f"{context} row must be an object")
                rows.append(value)
    except (OSError, json.JSONDecodeError, DataValidationError) as exc:
        raise FrozenArtifactError(f"cannot read {context}: {exc}") from exc
    return rows


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        raise FrozenArtifactError(f"refusing to overwrite M2 CAA artifact: {path}") from None
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    with os.fdopen(descriptor, "ab") as handle:
        handle.write(_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def _source_pairs(
    snapshot: VerifiedE3ConstructionSnapshot,
    questions: Sequence[Question],
    *,
    protocol: M2CAAProtocol,
) -> tuple[dict[str, Any], ...]:
    question_map = {value.question_id: value for value in questions}
    if len(question_map) != len(questions):
        raise DataValidationError("M2 CAA questions contain duplicate identities")
    groups = semantic_group_ids(questions)
    rows: list[dict[str, Any]] = []
    for _schedule, generation in zip(snapshot.schedule, snapshot.generations, strict=True):
        if generation.prompt_id != protocol.source_prompt_id:
            continue
        if generation.outcome is not Outcome.INCORRECT:
            continue
        question = question_map[generation.question_id]
        incorrect = generation.evidence.get("raw_output")
        if not isinstance(incorrect, str) or not incorrect.strip():
            raise DataValidationError("M2 CAA source incorrect answer is empty")
        body = {
            "schema_version": 1,
            "question_id": question.question_id,
            "prompt_id": generation.prompt_id,
            "semantic_group_id": groups[question.question_id],
            "positive_answer": question.aliases[0],
            "positive_answer_sha256": hashlib.sha256(
                question.aliases[0].encode("utf-8")
            ).hexdigest(),
            "negative_answer": incorrect,
            "negative_answer_sha256": hashlib.sha256(incorrect.encode("utf-8")).hexdigest(),
            "rendered_prompt_sha256": generation.rendered_prompt_sha256,
            "prompt_token_ids_sha256": generation.prompt_token_ids_sha256,
            "source_schedule_row_sha256": generation.schedule_row_sha256,
            "source_e3_record_sha256": stable_hash(generation.to_dict()),
            "source_outcome": generation.outcome.value,
            "semantic_match": "identical-question-and-semantic-group",
        }
        rows.append({**body, "pair_id": stable_hash(body)})
    if not rows:
        raise DataValidationError("M2 CAA source contains no original incorrect responses")
    return tuple(
        sorted(
            rows,
            key=lambda value: (
                stable_hash(
                    {
                        "schema_version": 1,
                        "seed": protocol.seed,
                        "pair_id": value["pair_id"],
                    }
                ),
                value["pair_id"],
            ),
        )
    )


def _plan(
    *,
    construction_directory: Path,
    snapshot: VerifiedE3ConstructionSnapshot,
    questions: Sequence[Question],
    protocol: M2CAAProtocol,
) -> dict[str, Any]:
    if (
        snapshot.scientific_eligible
        and (
            snapshot.plan.get("hidden_width") != _ACTIVE_HIDDEN_WIDTH
            or not _active_qwen_runtime_identity(snapshot.plan.get("runtime_identity"))
        )
    ):
        raise FrozenArtifactError(
            "M2 CAA requires the exact scientific Qwen/A100 vLLM E3 runtime and width"
        )
    pairs = _source_pairs(snapshot, questions, protocol=protocol)
    body = {
        "schema_version": 1,
        "phase": "E4-M2-CAA-construction",
        "runner_source_sha256": sha256_file(Path(__file__)),
        "protocol": protocol.to_dict(),
        "source_e3_construction_sha256": sha256_path(construction_directory),
        "source_e3_plan_identity": snapshot.plan["plan_identity"],
        "source_e3_generation_chain_head": snapshot.generation_chain_head,
        "source_e3_scientific_eligible": snapshot.scientific_eligible,
        "runtime_identity": dict(snapshot.plan["runtime_identity"]),
        "hidden_width": snapshot.plan["hidden_width"],
        "pair_count": len(pairs),
        "pairs": list(pairs),
        "schedule_digest": stable_hash([value["pair_id"] for value in pairs]),
        "scientific_eligible": snapshot.scientific_eligible and protocol.scientific_eligible,
    }
    return {**body, "plan_identity": stable_hash(body)}


def _context(
    directory: str | Path,
    *,
    construction_directory: str | Path,
    questions: Sequence[Question],
    prompts: Mapping[str, PromptSpec],
    e3_protocol: E3Protocol,
    protocol: M2CAAProtocol,
) -> tuple[Path, dict[str, Any], tuple[dict[str, Any], ...]]:
    work = Path(directory)
    if (
        work.is_symlink()
        or not work.is_dir()
        or {item.name for item in work.iterdir()} != _WORK_FILES
        or any(item.is_symlink() for item in work.rglob("*"))
        or not (work / "checkpoints").is_dir()
    ):
        raise FrozenArtifactError("M2 CAA work inventory differs")
    snapshot = load_verified_e3_construction_snapshot(
        construction_directory,
        questions=questions,
        prompts=prompts,
        protocol=e3_protocol,
    )
    expected = _plan(
        construction_directory=Path(construction_directory),
        snapshot=snapshot,
        questions=questions,
        protocol=protocol,
    )
    observed = _read_json(work / "plan.json", "M2 CAA plan")
    if observed != expected:
        raise FrozenArtifactError("M2 CAA plan differs from the verified E3 source")
    return work, observed, tuple(observed["pairs"])


def prepare_m2_caa_work(
    directory: str | Path,
    *,
    construction_directory: str | Path,
    questions: Sequence[Question],
    prompts: Mapping[str, PromptSpec],
    e3_protocol: E3Protocol | None = None,
    protocol: M2CAAProtocol | None = None,
) -> Mapping[str, Any]:
    """Freeze exact incorrect/gold semantic pairs before loading the model."""

    normalized = validate_active_study_artifact_paths(
        {
            "M2 CAA work": directory,
            "E3 construction": construction_directory,
        }
    )
    destination = normalized["M2 CAA work"]
    source = normalized["E3 construction"]
    frozen_e3 = e3_protocol or E3Protocol()
    frozen = protocol or M2CAAProtocol()
    snapshot = load_verified_e3_construction_snapshot(
        source,
        questions=questions,
        prompts=prompts,
        protocol=frozen_e3,
    )
    plan = _plan(
        construction_directory=source,
        snapshot=snapshot,
        questions=questions,
        protocol=frozen,
    )
    if destination.exists() or destination.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite M2 CAA work: {destination}")
    destination.mkdir(parents=True)
    (destination / "checkpoints").mkdir()
    _write_once(destination / "plan.json", plan)
    (destination / "pairs.jsonl").touch(mode=0o600)
    (destination / "sessions.jsonl").touch(mode=0o600)
    return MappingProxyType(
        {
            "prepared": True,
            "plan_identity": plan["plan_identity"],
            "pairs_expected": plan["pair_count"],
            "scientific_eligible": plan["scientific_eligible"],
            "work_directory": str(destination),
        }
    )


def _empty(plan: Mapping[str, Any], protocol: M2CAAProtocol) -> _Accumulator:
    layers = len(protocol.layers)
    width = int(plan["hidden_width"])
    return _Accumulator(
        processed_pairs=0,
        counts=np.zeros(layers, dtype=np.int64),
        difference_sums=np.zeros((layers, width), dtype=np.float64),
        rms_elements=np.zeros(layers, dtype=np.int64),
        rms_sum_squares=np.zeros(layers, dtype=np.float64),
    )


def _checkpoint_paths(directory: Path) -> tuple[Path, ...]:
    values = tuple(sorted(directory.glob("checkpoint-*.npz")))
    if {item.name for item in directory.iterdir()} != {item.name for item in values}:
        raise FrozenArtifactError("M2 CAA checkpoint inventory differs")
    return values


def _checkpoint(
    directory: Path,
    *,
    index: int,
    state: _Accumulator,
    plan_identity: str,
    pair_chain_head: str | None,
    previous_sha256: str | None,
) -> tuple[Path, str]:
    metadata = {
        "schema_version": 1,
        "checkpoint_index": index,
        "processed_pairs": state.processed_pairs,
        "plan_identity": plan_identity,
        "pair_chain_head": pair_chain_head,
        "previous_checkpoint_sha256": previous_sha256,
    }
    destination = directory / f"checkpoint-{index:08d}.npz"
    descriptor, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                counts=state.counts,
                difference_sums=state.difference_sums,
                rms_elements=state.rms_elements,
                rms_sum_squares=state.rms_sum_squares,
            )
            handle.flush()
            os.fsync(handle.fileno())
        if destination.exists():
            raise FrozenArtifactError("refusing to overwrite M2 CAA checkpoint")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination, sha256_file(destination)


def _load_checkpoints(
    directory: Path,
    *,
    plan: Mapping[str, Any],
    protocol: M2CAAProtocol,
    receipts: Sequence[Mapping[str, Any]],
) -> tuple[_Accumulator, str | None, int]:
    state = _empty(plan, protocol)
    previous: str | None = None
    paths = _checkpoint_paths(directory)
    for index, path in enumerate(paths):
        try:
            with np.load(path, allow_pickle=False) as values:
                if set(values.files) != {
                    "metadata",
                    "counts",
                    "difference_sums",
                    "rms_elements",
                    "rms_sum_squares",
                }:
                    raise DataValidationError("M2 CAA checkpoint arrays differ")
                metadata = json.loads(str(values["metadata"].item()))
                current = _Accumulator(
                    processed_pairs=int(metadata["processed_pairs"]),
                    counts=values["counts"].copy(),
                    difference_sums=values["difference_sums"].copy(),
                    rms_elements=values["rms_elements"].copy(),
                    rms_sum_squares=values["rms_sum_squares"].copy(),
                )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise FrozenArtifactError(f"cannot read M2 CAA checkpoint: {exc}") from exc
        expected_head = (
            receipts[current.processed_pairs - 1]["receipt_digest"]
            if current.processed_pairs
            else None
        )
        if (
            metadata
            != {
                "schema_version": 1,
                "checkpoint_index": index,
                "processed_pairs": current.processed_pairs,
                "plan_identity": plan["plan_identity"],
                "pair_chain_head": expected_head,
                "previous_checkpoint_sha256": previous,
            }
            or current.processed_pairs > len(receipts)
            or current.counts.dtype != np.int64
            or current.counts.shape != (len(protocol.layers),)
            or not np.all(current.counts == current.processed_pairs)
            or current.difference_sums.dtype != np.float64
            or current.difference_sums.shape != (len(protocol.layers), int(plan["hidden_width"]))
            or current.rms_elements.dtype != np.int64
            or current.rms_elements.shape != current.counts.shape
            or current.rms_sum_squares.dtype != np.float64
            or current.rms_sum_squares.shape != current.counts.shape
            or np.any(current.rms_elements < 0)
            or not np.isfinite(current.difference_sums).all()
            or not np.isfinite(current.rms_sum_squares).all()
            or np.any(current.rms_sum_squares < 0)
        ):
            raise FrozenArtifactError("M2 CAA checkpoint state differs")
        state = current
        previous = sha256_file(path)
    return state, previous, len(paths)


def _receipts(path: Path, *, plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = _read_jsonl(path, "M2 CAA pair receipts")
    previous: str | None = None
    pairs = plan["pairs"]
    for sequence, row in enumerate(rows):
        body = dict(row)
        digest = body.pop("receipt_digest", None)
        layers = row.get("layers")
        if (
            set(row)
            != {
                "schema_version",
                "sequence",
                "plan_identity",
                "pair_id",
                "question_id",
                "rendered_prompt_sha256",
                "prompt_token_ids_sha256",
                "positive_response_sha256",
                "positive_token_ids_sha256",
                "negative_response_sha256",
                "negative_token_ids_sha256",
                "layers",
                "capture_peak_memory_bytes",
                "previous_receipt_digest",
                "receipt_digest",
            }
            or digest != stable_hash(body)
            or row.get("schema_version") != 1
            or row.get("sequence") != sequence
            or sequence >= len(pairs)
            or row.get("plan_identity") != plan["plan_identity"]
            or row.get("pair_id") != pairs[sequence]["pair_id"]
            or row.get("question_id") != pairs[sequence]["question_id"]
            or row.get("previous_receipt_digest") != previous
            or not isinstance(layers, list)
            or [value.get("layer") for value in layers if isinstance(value, Mapping)]
            != list(E2_LAYERS)
            or any(
                not isinstance(value, Mapping)
                or set(value)
                != {
                    "layer",
                    "positive_activation_sha256",
                    "negative_activation_sha256",
                    "pooled_difference_sha256",
                    "positive_tokens",
                    "negative_tokens",
                }
                or any(
                    not isinstance(value.get(name), str) or len(value[name]) != 64
                    for name in (
                        "positive_activation_sha256",
                        "negative_activation_sha256",
                        "pooled_difference_sha256",
                    )
                )
                or type(value.get("positive_tokens")) is not int
                or value["positive_tokens"] <= 0
                or type(value.get("negative_tokens")) is not int
                or value["negative_tokens"] <= 0
                for value in layers
            )
            or type(row.get("capture_peak_memory_bytes")) is not int
            or row["capture_peak_memory_bytes"] < 0
        ):
            raise FrozenArtifactError("M2 CAA pair receipt chain differs")
        previous = str(digest)
    return rows


def _sessions(path: Path, *, plan_identity: str) -> list[dict[str, Any]]:
    rows = _read_jsonl(path, "M2 CAA sessions")
    previous: str | None = None
    open_session: int | None = None
    for row in rows:
        body = dict(row)
        digest = body.pop("session_digest", None)
        if (
            set(row)
            != {
                "schema_version",
                "event",
                "session_index",
                "plan_identity",
                "details",
                "previous_session_digest",
                "session_digest",
            }
            or digest != stable_hash(body)
            or row.get("schema_version") != 1
            or row.get("plan_identity") != plan_identity
            or row.get("previous_session_digest") != previous
            or row.get("event") not in {"start", "end"}
            or type(row.get("session_index")) is not int
            or not isinstance(row.get("details"), Mapping)
        ):
            raise FrozenArtifactError("M2 CAA session chain differs")
        if row["event"] == "start":
            if open_session is not None:
                raise FrozenArtifactError("M2 CAA sessions overlap")
            open_session = int(row["session_index"])
        elif open_session != row["session_index"]:
            raise FrozenArtifactError("M2 CAA session end is unmatched")
        else:
            open_session = None
        previous = str(digest)
    return rows


def _maximum_peak_memory_bytes(
    receipts: Sequence[Mapping[str, Any]],
    sessions: Sequence[Mapping[str, Any]],
) -> int:
    """Return the durable high-water mark, including interrupted captures.

    Pair receipts are written before the corresponding checkpoint/session end, so
    they are the authoritative lower bound when a process is killed between those
    writes.  Session peaks remain useful for runtime allocations outside a single
    capture.  Taking both prevents a zero-work recovery session from erasing an
    earlier over-budget capture.
    """

    peaks = [int(row["capture_peak_memory_bytes"]) for row in receipts]
    for row in sessions:
        details = row["details"]
        if "peak_memory_bytes" not in details:
            continue
        peak = details["peak_memory_bytes"]
        if type(peak) is not int or peak < 0:
            raise FrozenArtifactError("M2 CAA session peak memory evidence differs")
        peaks.append(peak)
    return max(peaks, default=0)


def _session_event(
    path: Path,
    *,
    event: str,
    session_index: int,
    plan_identity: str,
    details: Mapping[str, Any],
) -> None:
    rows = _read_jsonl(path, "M2 CAA sessions")
    body = {
        "schema_version": 1,
        "event": event,
        "session_index": session_index,
        "plan_identity": plan_identity,
        "details": dict(details),
        "previous_session_digest": rows[-1]["session_digest"] if rows else None,
    }
    _append_jsonl(path, {**body, "session_digest": stable_hash(body)})


@contextmanager
def _lock(path: Path) -> Iterator[None]:
    with path.open("rb") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ConfigurationError("M2 CAA work is already running") from exc
        yield


def _capture_pair(
    runtime: M2CAARuntime,
    *,
    pair: Mapping[str, Any],
    question: Question,
    prompt: PromptSpec,
    protocol: M2CAAProtocol,
    hidden_width: int,
) -> tuple[dict[str, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    rendered = runtime.render_prompt(prompt, question.text, metadata=dict(question.metadata))
    if (
        rendered.sha256 != pair["rendered_prompt_sha256"]
        or rendered.token_ids_sha256 != pair["prompt_token_ids_sha256"]
    ):
        raise FrozenArtifactError("M2 CAA prompt differs from the E3 source generation")
    positive = runtime.teacher_forced_cube(
        rendered,
        str(pair["positive_answer"]),
        layers=protocol.layers,
        sites=(protocol.site,),
    )
    negative = runtime.teacher_forced_cube(
        rendered,
        str(pair["negative_answer"]),
        layers=protocol.layers,
        sites=(protocol.site,),
    )
    if (
        positive.response_text_sha256 != pair["positive_answer_sha256"]
        or negative.response_text_sha256 != pair["negative_answer_sha256"]
    ):
        raise DataValidationError("M2 CAA teacher-forced response identity differs")
    differences: list[np.ndarray[Any, Any]] = []
    elements: list[int] = []
    squares: list[float] = []
    layer_receipts: list[dict[str, Any]] = []
    for layer in protocol.layers:
        positive_values = np.asarray(positive.activations[protocol.site][layer], dtype=np.float32)
        negative_values = np.asarray(negative.activations[protocol.site][layer], dtype=np.float32)
        if (
            positive_values.ndim != 2
            or negative_values.ndim != 2
            or positive_values.shape[1] != hidden_width
            or negative_values.shape[1] != hidden_width
            or not np.isfinite(positive_values).all()
            or not np.isfinite(negative_values).all()
        ):
            raise DataValidationError("M2 CAA activation geometry differs")
        difference = positive_values.mean(axis=0, dtype=np.float64) - negative_values.mean(
            axis=0, dtype=np.float64
        )
        differences.append(difference)
        elements.append(positive_values.size + negative_values.size)
        squares.append(
            float(np.square(positive_values, dtype=np.float64).sum())
            + float(np.square(negative_values, dtype=np.float64).sum())
        )
        layer_receipts.append(
            {
                "layer": layer,
                "positive_activation_sha256": hashlib.sha256(
                    np.ascontiguousarray(positive_values).tobytes()
                ).hexdigest(),
                "negative_activation_sha256": hashlib.sha256(
                    np.ascontiguousarray(negative_values).tobytes()
                ).hexdigest(),
                "pooled_difference_sha256": hashlib.sha256(
                    np.ascontiguousarray(difference).tobytes()
                ).hexdigest(),
                "positive_tokens": positive_values.shape[0],
                "negative_tokens": negative_values.shape[0],
            }
        )
    receipt = {
        "rendered_prompt_sha256": rendered.sha256,
        "prompt_token_ids_sha256": rendered.token_ids_sha256,
        "positive_response_sha256": positive.response_text_sha256,
        "positive_token_ids_sha256": positive.response_token_ids_sha256,
        "negative_response_sha256": negative.response_text_sha256,
        "negative_token_ids_sha256": negative.response_token_ids_sha256,
        "layers": layer_receipts,
        "capture_peak_memory_bytes": max(positive.peak_memory_bytes, negative.peak_memory_bytes),
    }
    return (
        receipt,
        np.stack(differences),
        np.asarray(elements, dtype=np.int64),
        np.asarray(squares, dtype=np.float64),
    )


def run_m2_caa_work(
    directory: str | Path,
    *,
    construction_directory: str | Path,
    questions: Sequence[Question],
    prompts: Mapping[str, PromptSpec],
    runtime: M2CAARuntime,
    request_budget: int | None = None,
    e3_protocol: E3Protocol | None = None,
    protocol: M2CAAProtocol | None = None,
) -> Mapping[str, Any]:
    """Capture and durably accumulate an exact prefix of semantic CAA pairs."""

    if request_budget is not None and (type(request_budget) is not int or request_budget <= 0):
        raise ConfigurationError("M2 CAA request budget must be positive")
    frozen_e3 = e3_protocol or E3Protocol()
    frozen = protocol or M2CAAProtocol()
    work, plan, pairs = _context(
        directory,
        construction_directory=construction_directory,
        questions=questions,
        prompts=prompts,
        e3_protocol=frozen_e3,
        protocol=frozen,
    )
    if dict(runtime.runtime_identity()) != plan["runtime_identity"]:
        raise FrozenArtifactError("M2 CAA runtime differs from the E3 native-VLLM runtime")
    question_map = {value.question_id: value for value in questions}
    with _lock(work / "plan.json"):
        receipts = _receipts(work / "pairs.jsonl", plan=plan)
        state, checkpoint_head, checkpoint_index = _load_checkpoints(
            work / "checkpoints", plan=plan, protocol=frozen, receipts=receipts
        )
        sessions = _sessions(work / "sessions.jsonl", plan_identity=plan["plan_identity"])
        if sessions and sessions[-1]["event"] == "start":
            recovered_peak = _maximum_peak_memory_bytes(receipts, sessions)
            _session_event(
                work / "sessions.jsonl",
                event="end",
                session_index=int(sessions[-1]["session_index"]),
                plan_identity=plan["plan_identity"],
                details={
                    "status": "interrupted-recovered",
                    "processed_pairs": state.processed_pairs,
                    "checkpoint_chain_head": checkpoint_head,
                    "peak_memory_bytes": recovered_peak,
                },
            )
            sessions = _sessions(work / "sessions.jsonl", plan_identity=plan["plan_identity"])
        session_index = sum(row["event"] == "start" for row in sessions)
        _session_event(
            work / "sessions.jsonl",
            event="start",
            session_index=session_index,
            plan_identity=plan["plan_identity"],
            details={
                "processed_pairs": state.processed_pairs,
                "runtime_identity": plan["runtime_identity"],
                "started_unix_ns": time.time_ns(),
            },
        )
        handled = 0
        peak = 0
        status = "partial"
        try:
            while state.processed_pairs < len(pairs) and (
                request_budget is None or handled < request_budget
            ):
                sequence = state.processed_pairs
                pair = pairs[sequence]
                receipt_values, differences, elements, squares = _capture_pair(
                    runtime,
                    pair=pair,
                    question=question_map[str(pair["question_id"])],
                    prompt=prompts[frozen.source_prompt_id],
                    protocol=frozen,
                    hidden_width=int(plan["hidden_width"]),
                )
                body = {
                    "schema_version": 1,
                    "sequence": sequence,
                    "plan_identity": plan["plan_identity"],
                    "pair_id": pair["pair_id"],
                    "question_id": pair["question_id"],
                    **receipt_values,
                    "previous_receipt_digest": (
                        receipts[sequence - 1]["receipt_digest"] if sequence else None
                    ),
                }
                row = {**body, "receipt_digest": stable_hash(body)}
                if sequence < len(receipts):
                    if receipts[sequence] != row:
                        raise FrozenArtifactError(
                            "M2 CAA recaptured activation differs after interruption"
                        )
                else:
                    _append_jsonl(work / "pairs.jsonl", row)
                    receipts.append(row)
                state.difference_sums += differences
                state.counts += 1
                state.rms_elements += elements
                state.rms_sum_squares += squares
                state.processed_pairs += 1
                handled += 1
                peak = max(peak, int(row["capture_peak_memory_bytes"]))
                if state.processed_pairs % frozen.checkpoint_pairs == 0:
                    _path, checkpoint_head = _checkpoint(
                        work / "checkpoints",
                        index=checkpoint_index,
                        state=state,
                        plan_identity=plan["plan_identity"],
                        pair_chain_head=str(receipts[state.processed_pairs - 1]["receipt_digest"]),
                        previous_sha256=checkpoint_head,
                    )
                    checkpoint_index += 1
            if state.processed_pairs and (
                checkpoint_index == 0 or state.processed_pairs % frozen.checkpoint_pairs != 0
            ):
                _path, checkpoint_head = _checkpoint(
                    work / "checkpoints",
                    index=checkpoint_index,
                    state=state,
                    plan_identity=plan["plan_identity"],
                    pair_chain_head=str(receipts[state.processed_pairs - 1]["receipt_digest"]),
                    previous_sha256=checkpoint_head,
                )
            status = "complete" if state.processed_pairs == len(pairs) else "partial"
        finally:
            _session_event(
                work / "sessions.jsonl",
                event="end",
                session_index=session_index,
                plan_identity=plan["plan_identity"],
                details={
                    "status": status,
                    "processed_pairs": state.processed_pairs,
                    "checkpoint_chain_head": checkpoint_head,
                    "peak_memory_bytes": peak,
                    "ended_unix_ns": time.time_ns(),
                },
            )
    return verify_m2_caa_work(
        work,
        construction_directory=construction_directory,
        questions=questions,
        prompts=prompts,
        e3_protocol=frozen_e3,
        protocol=frozen,
        require_complete=False,
    )


def verify_m2_caa_work(
    directory: str | Path,
    *,
    construction_directory: str | Path,
    questions: Sequence[Question],
    prompts: Mapping[str, PromptSpec],
    e3_protocol: E3Protocol | None = None,
    protocol: M2CAAProtocol | None = None,
    require_complete: bool = False,
) -> Mapping[str, Any]:
    frozen_e3 = e3_protocol or E3Protocol()
    frozen = protocol or M2CAAProtocol()
    work, plan, pairs = _context(
        directory,
        construction_directory=construction_directory,
        questions=questions,
        prompts=prompts,
        e3_protocol=frozen_e3,
        protocol=frozen,
    )
    receipts = _receipts(work / "pairs.jsonl", plan=plan)
    state, checkpoint_head, checkpoint_count = _load_checkpoints(
        work / "checkpoints", plan=plan, protocol=frozen, receipts=receipts
    )
    sessions = _sessions(work / "sessions.jsonl", plan_identity=plan["plan_identity"])
    if sessions and sessions[-1]["event"] == "start":
        raise FrozenArtifactError("M2 CAA work contains an unclosed session")
    if state.processed_pairs != len(receipts):
        raise FrozenArtifactError("M2 CAA checkpoint does not cover every pair receipt")
    if sessions and sessions[-1]["details"].get("processed_pairs") != state.processed_pairs:
        raise FrozenArtifactError("M2 CAA session head differs from its checkpoint")
    complete = state.processed_pairs == len(pairs)
    if require_complete and not complete:
        raise FrozenArtifactError("M2 CAA construction is incomplete")
    peak = _maximum_peak_memory_bytes(receipts, sessions)
    return MappingProxyType(
        {
            "valid": True,
            "complete": complete,
            "pairs_processed": state.processed_pairs,
            "pairs_expected": len(pairs),
            "plan_identity": plan["plan_identity"],
            "pair_chain_head": receipts[-1]["receipt_digest"] if receipts else None,
            "checkpoint_chain_head": checkpoint_head,
            "checkpoint_count": checkpoint_count,
            "maximum_peak_memory_bytes": peak,
            "scientific_eligible": bool(
                complete
                and plan["scientific_eligible"]
                and state.processed_pairs > 0
                and peak <= _MAXIMUM_VRAM_BYTES
            ),
        }
    )


def _write_accumulator(path: Path, *, plan: Mapping[str, Any], state: _Accumulator) -> None:
    metadata = {
        "schema_version": 1,
        "plan_identity": plan["plan_identity"],
        "processed_pairs": state.processed_pairs,
    }
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            counts=state.counts,
            difference_sums=state.difference_sums,
            rms_elements=state.rms_elements,
            rms_sum_squares=state.rms_sum_squares,
        )


def finalize_m2_caa_artifact(
    directory: str | Path,
    *,
    work_directory: str | Path,
    construction_directory: str | Path,
    questions: Sequence[Question],
    prompts: Mapping[str, PromptSpec],
    e3_protocol: E3Protocol | None = None,
    protocol: M2CAAProtocol | None = None,
) -> M2CAAArtifact:
    """Publish the verified residual CAA bank and portable construction evidence."""

    frozen_e3 = e3_protocol or E3Protocol()
    frozen = protocol or M2CAAProtocol()
    verification = verify_m2_caa_work(
        work_directory,
        construction_directory=construction_directory,
        questions=questions,
        prompts=prompts,
        e3_protocol=frozen_e3,
        protocol=frozen,
        require_complete=True,
    )
    if not verification["scientific_eligible"]:
        raise FrozenArtifactError("M2 CAA construction is not scientifically eligible")
    work, plan, _pairs = _context(
        work_directory,
        construction_directory=construction_directory,
        questions=questions,
        prompts=prompts,
        e3_protocol=frozen_e3,
        protocol=frozen,
    )
    receipts = _receipts(work / "pairs.jsonl", plan=plan)
    state, _head, _count = _load_checkpoints(
        work / "checkpoints", plan=plan, protocol=frozen, receipts=receipts
    )
    means = state.difference_sums / state.counts[:, None]
    norms = np.linalg.norm(means, axis=1)
    reference_rms = np.sqrt(state.rms_sum_squares / state.rms_elements)
    if (
        np.any(state.counts <= 0)
        or np.any(state.rms_elements <= 0)
        or not np.isfinite(means).all()
        or not np.isfinite(norms).all()
        or np.any(norms <= 0)
        or not np.isfinite(reference_rms).all()
        or np.any(reference_rms <= 0)
    ):
        raise DataValidationError("M2 CAA final direction geometry is degenerate")
    directions = (means / norms[:, None]).astype(np.float32)
    destination = validate_active_study_artifact_paths({"M2 CAA artifact": directory})[
        "M2 CAA artifact"
    ]
    if destination.exists() or destination.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite M2 CAA artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        shutil.copyfile(work / "plan.json", stage / "plan.json")
        shutil.copyfile(work / "pairs.jsonl", stage / "pairs.jsonl")
        shutil.copyfile(work / "sessions.jsonl", stage / "sessions.jsonl")
        _write_accumulator(stage / "accumulator.npz", plan=plan, state=state)
        tensors = {
            HookKey(layer, frozen.site).artifact_key: torch.from_numpy(
                directions[index]
            ).contiguous()
            for index, layer in enumerate(frozen.layers)
        }
        save_file(tensors, stage / "vectors.safetensors")
        data_fingerprint = stable_hash(
            {
                "plan_identity": plan["plan_identity"],
                "pair_chain_head": receipts[-1]["receipt_digest"],
                "accumulator_sha256": sha256_file(stage / "accumulator.npz"),
            }
        )
        metadata_body = {
            "schema_version": 1,
            "data_fingerprint": data_fingerprint,
            "tensor_sha256": sha256_file(stage / "vectors.safetensors"),
            "vectors": [
                {
                    "tensor_key": HookKey(layer, frozen.site).artifact_key,
                    "layer": layer,
                    "site": frozen.site.value,
                    "source_method": "M2-CAA-native-VLLM-teacher-forced-pairs",
                    "positive_count": int(state.counts[index]),
                    "negative_count": int(state.counts[index]),
                }
                for index, layer in enumerate(frozen.layers)
            ],
        }
        _write_once(
            stage / "metadata.json",
            {**metadata_body, "metadata_digest": stable_hash(metadata_body)},
        )
        sessions = _sessions(stage / "sessions.jsonl", plan_identity=plan["plan_identity"])
        manifest_body = {
            "schema_version": 1,
            "phase": "E4-M2-CAA-construction",
            "status": "complete",
            "scientific_eligible": True,
            "runner_source_sha256": sha256_file(Path(__file__)),
            "plan_identity": plan["plan_identity"],
            "source_e3_construction_sha256": plan["source_e3_construction_sha256"],
            "pair_count": state.processed_pairs,
            "pair_chain_head": receipts[-1]["receipt_digest"],
            "pair_set_digest": stable_hash([value["receipt_digest"] for value in receipts]),
            "session_chain_head": sessions[-1]["session_digest"],
            "session_set_digest": stable_hash([value["session_digest"] for value in sessions]),
            "accumulator_sha256": sha256_file(stage / "accumulator.npz"),
            "metadata_sha256": sha256_file(stage / "metadata.json"),
            "vectors_sha256": sha256_file(stage / "vectors.safetensors"),
            "data_fingerprint": data_fingerprint,
            "maximum_peak_memory_bytes": verification["maximum_peak_memory_bytes"],
        }
        _write_once(
            stage / "manifest.json",
            {**manifest_body, "manifest_digest": stable_hash(manifest_body)},
        )
        verify_m2_caa_artifact(
            stage,
            expected_manifest_digest=stable_hash(manifest_body),
        )
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    manifest = _read_json(destination / "manifest.json", "M2 CAA manifest")
    return verify_m2_caa_artifact(
        destination,
        expected_manifest_digest=str(manifest["manifest_digest"]),
    )


def verify_m2_caa_artifact(
    directory: str | Path,
    *,
    expected_manifest_digest: str,
) -> M2CAAArtifact:
    """Replay the portable CAA accumulator, direction tensors, and receipt chains."""

    source = Path(directory)
    if (
        source.is_symlink()
        or not source.is_dir()
        or {item.name for item in source.iterdir()} != _OUTPUT_FILES
        or any(item.is_symlink() or not item.is_file() for item in source.iterdir())
    ):
        raise FrozenArtifactError("M2 CAA artifact inventory differs")
    plan = _read_json(source / "plan.json", "M2 CAA portable plan")
    plan_body = dict(plan)
    plan_identity = plan_body.pop("plan_identity", None)
    if (
        plan_identity != stable_hash(plan_body)
        or plan_body.get("schema_version") != 1
        or plan_body.get("phase") != "E4-M2-CAA-construction"
        or plan_body.get("protocol") != M2CAAProtocol().to_dict()
        or plan_body.get("pair_count") != len(plan_body.get("pairs", []))
        or plan_body.get("schedule_digest")
        != stable_hash([value["pair_id"] for value in plan_body.get("pairs", [])])
        or plan_body.get("scientific_eligible") is not True
        or plan_body.get("hidden_width") != _ACTIVE_HIDDEN_WIDTH
        or not _active_qwen_runtime_identity(plan_body.get("runtime_identity"))
    ):
        raise FrozenArtifactError("M2 CAA portable plan differs")
    receipts = _receipts(source / "pairs.jsonl", plan=plan)
    sessions = _sessions(source / "sessions.jsonl", plan_identity=str(plan_identity))
    if not receipts or not sessions or sessions[-1]["event"] != "end":
        raise FrozenArtifactError("M2 CAA portable journals are incomplete")
    try:
        with np.load(source / "accumulator.npz", allow_pickle=False) as values:
            if set(values.files) != {
                "metadata",
                "counts",
                "difference_sums",
                "rms_elements",
                "rms_sum_squares",
            }:
                raise DataValidationError("M2 CAA portable accumulator arrays differ")
            accumulator_metadata = json.loads(str(values["metadata"].item()))
            counts = values["counts"]
            sums = values["difference_sums"]
            elements = values["rms_elements"]
            squares = values["rms_sum_squares"]
    except (OSError, ValueError, TypeError, json.JSONDecodeError, DataValidationError) as exc:
        raise FrozenArtifactError(f"cannot load M2 CAA accumulator: {exc}") from exc
    width = int(plan["hidden_width"])
    if (
        accumulator_metadata
        != {
            "schema_version": 1,
            "plan_identity": plan_identity,
            "processed_pairs": len(receipts),
        }
        or counts.dtype != np.int64
        or counts.shape != (len(E2_LAYERS),)
        or not np.all(counts == len(receipts))
        or sums.dtype != np.float64
        or sums.shape != (len(E2_LAYERS), width)
        or elements.dtype != np.int64
        or elements.shape != counts.shape
        or np.any(elements <= 0)
        or squares.dtype != np.float64
        or squares.shape != counts.shape
        or not np.isfinite(sums).all()
        or not np.isfinite(squares).all()
        or np.any(squares <= 0)
    ):
        raise FrozenArtifactError("M2 CAA portable accumulator geometry differs")
    means = sums / counts[:, None]
    norms = np.linalg.norm(means, axis=1)
    expected_directions = (means / norms[:, None]).astype(np.float32)
    if not np.isfinite(expected_directions).all() or np.any(norms <= 0):
        raise FrozenArtifactError("M2 CAA portable direction is degenerate")
    bank = load_vector_bank(source, expected_data_fingerprint=None)
    observed = np.stack(
        [
            bank.vectors[HookKey(layer, _SITE)].direction.detach().cpu().numpy()
            for layer in E2_LAYERS
        ]
    )
    metadata = _read_json(source / "metadata.json", "M2 CAA vector metadata")
    manifest = _read_json(source / "manifest.json", "M2 CAA manifest")
    manifest_body = dict(manifest)
    manifest_digest = manifest_body.pop("manifest_digest", None)
    peak = _maximum_peak_memory_bytes(receipts, sessions)
    expected_manifest = {
        "schema_version": 1,
        "phase": "E4-M2-CAA-construction",
        "status": "complete",
        "scientific_eligible": True,
        "runner_source_sha256": plan["runner_source_sha256"],
        "plan_identity": plan_identity,
        "source_e3_construction_sha256": plan["source_e3_construction_sha256"],
        "pair_count": len(receipts),
        "pair_chain_head": receipts[-1]["receipt_digest"],
        "pair_set_digest": stable_hash([value["receipt_digest"] for value in receipts]),
        "session_chain_head": sessions[-1]["session_digest"],
        "session_set_digest": stable_hash([value["session_digest"] for value in sessions]),
        "accumulator_sha256": sha256_file(source / "accumulator.npz"),
        "metadata_sha256": sha256_file(source / "metadata.json"),
        "vectors_sha256": sha256_file(source / "vectors.safetensors"),
        "data_fingerprint": metadata["data_fingerprint"],
        "maximum_peak_memory_bytes": peak,
    }
    tensors = load_file(source / "vectors.safetensors", device="cpu")
    if (
        manifest_digest != expected_manifest_digest
        or manifest_digest != stable_hash(manifest_body)
        or manifest_body != expected_manifest
        or not np.array_equal(observed, expected_directions)
        or set(tensors) != {HookKey(layer, _SITE).artifact_key for layer in E2_LAYERS}
        or peak > _MAXIMUM_VRAM_BYTES
    ):
        raise FrozenArtifactError("M2 CAA artifact differs from portable replay")
    return M2CAAArtifact(
        directory=source.absolute(),
        plan_identity=str(plan_identity),
        data_fingerprint=str(metadata["data_fingerprint"]),
        pair_count=len(receipts),
        layers=E2_LAYERS,
        site=_SITE,
        maximum_peak_memory_bytes=peak,
        manifest_digest=str(manifest_digest),
    )

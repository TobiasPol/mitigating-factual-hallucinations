"""Immutable, resumable float16 activation shards for the local VLLM study."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from mfh.artifact_namespace import validate_active_study_artifact_paths
from mfh.contracts import ActivationSite, Outcome
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.provenance import sha256_file, stable_hash

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
_SHARD = re.compile(r"^shard-(\d{5})$")


@dataclass(frozen=True, slots=True)
class ActivationStoreSpec:
    plan_identity: str
    model_repository: str
    model_revision: str
    quantization: str
    layers: tuple[int, ...]
    sites: tuple[ActivationSite, ...]
    hidden_width: int
    expected_rows: int
    dtype: str = "float16"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise DataValidationError("unsupported activation-store schema version")
        if not _SHA256.fullmatch(self.plan_identity):
            raise DataValidationError("activation store requires a plan SHA-256")
        if not self.model_repository.strip() or not self.quantization.strip():
            raise DataValidationError("activation store model identity is incomplete")
        if not _REVISION.fullmatch(self.model_revision):
            raise DataValidationError("activation store model revision is not immutable")
        layers = tuple(self.layers)
        sites = tuple(self.sites)
        if (
            not layers
            or any(type(value) is not int or value < 0 for value in layers)
            or len(set(layers)) != len(layers)
            or not sites
            or any(not isinstance(value, ActivationSite) for value in sites)
            or len(set(sites)) != len(sites)
        ):
            raise DataValidationError("activation store layer/site geometry is invalid")
        if type(self.hidden_width) is not int or self.hidden_width <= 0:
            raise DataValidationError("activation store hidden width must be positive")
        if type(self.expected_rows) is not int or self.expected_rows <= 0:
            raise DataValidationError("activation store expected row count must be positive")
        if self.dtype != "float16":
            raise DataValidationError("the active VLLM protocol requires float16 activation shards")
        object.__setattr__(self, "model_repository", self.model_repository.strip())
        object.__setattr__(self, "quantization", self.quantization.strip())
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "sites", sites)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_identity": self.plan_identity,
            "model_repository": self.model_repository,
            "model_revision": self.model_revision,
            "quantization": self.quantization,
            "layers": list(self.layers),
            "sites": [value.value for value in self.sites],
            "hidden_width": self.hidden_width,
            "expected_rows": self.expected_rows,
            "dtype": self.dtype,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ActivationStoreSpec:
        expected = {
            "schema_version",
            "plan_identity",
            "model_repository",
            "model_revision",
            "quantization",
            "layers",
            "sites",
            "hidden_width",
            "expected_rows",
            "dtype",
        }
        if set(value) != expected:
            raise DataValidationError("activation-store spec keys differ from version 1")
        layers = value["layers"]
        sites = value["sites"]
        if (
            type(value["schema_version"]) is not int
            or type(value["hidden_width"]) is not int
            or type(value["expected_rows"]) is not int
            or not isinstance(layers, list)
            or any(type(item) is not int for item in layers)
            or not isinstance(sites, list)
            or any(type(item) is not str for item in sites)
            or any(
                type(value[name]) is not str
                for name in (
                    "plan_identity",
                    "model_repository",
                    "model_revision",
                    "quantization",
                    "dtype",
                )
            )
        ):
            raise DataValidationError("activation-store spec has invalid JSON types")
        return cls(
            schema_version=value["schema_version"],
            plan_identity=value["plan_identity"],
            model_repository=value["model_repository"],
            model_revision=value["model_revision"],
            quantization=value["quantization"],
            layers=tuple(layers),
            sites=tuple(ActivationSite(item) for item in sites),
            hidden_width=value["hidden_width"],
            expected_rows=value["expected_rows"],
            dtype=value["dtype"],
        )

    @property
    def digest(self) -> str:
        return stable_hash(self.to_dict())


@dataclass(frozen=True, slots=True)
class ActivationCaptureRow:
    question_id: str
    benchmark: str
    partition: str
    prompt_id: str
    outcome: Outcome
    semantic_group_id: str
    rendered_prompt_sha256: str
    prompt_token_ids_sha256: str
    generation_record_sha256: str
    maximum_token_probability: float
    output_entropy: float

    def __post_init__(self) -> None:
        for name in (
            "question_id",
            "benchmark",
            "partition",
            "prompt_id",
            "semantic_group_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise DataValidationError(f"activation row {name} must be non-empty text")
            object.__setattr__(self, name, value.strip())
        for name in (
            "rendered_prompt_sha256",
            "prompt_token_ids_sha256",
            "generation_record_sha256",
        ):
            if not isinstance(getattr(self, name), str) or not _SHA256.fullmatch(
                getattr(self, name)
            ):
                raise DataValidationError(f"activation row {name} must be a SHA-256")
        maximum = float(self.maximum_token_probability)
        entropy = float(self.output_entropy)
        if (
            isinstance(self.maximum_token_probability, bool)
            or not isinstance(self.maximum_token_probability, int | float)
            or not math.isfinite(maximum)
            or not 0 < maximum <= 1
            or isinstance(self.output_entropy, bool)
            or not isinstance(self.output_entropy, int | float)
            or not math.isfinite(entropy)
            or entropy < 0
        ):
            raise DataValidationError("activation row confidence baselines are invalid")
        object.__setattr__(self, "outcome", Outcome(self.outcome))
        object.__setattr__(self, "maximum_token_probability", maximum)
        object.__setattr__(self, "output_entropy", entropy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "benchmark": self.benchmark,
            "partition": self.partition,
            "prompt_id": self.prompt_id,
            "outcome": self.outcome.value,
            "semantic_group_id": self.semantic_group_id,
            "rendered_prompt_sha256": self.rendered_prompt_sha256,
            "prompt_token_ids_sha256": self.prompt_token_ids_sha256,
            "generation_record_sha256": self.generation_record_sha256,
            "maximum_token_probability": self.maximum_token_probability,
            "output_entropy": self.output_entropy,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ActivationCaptureRow:
        expected = {
            "question_id",
            "benchmark",
            "partition",
            "prompt_id",
            "outcome",
            "semantic_group_id",
            "rendered_prompt_sha256",
            "prompt_token_ids_sha256",
            "generation_record_sha256",
            "maximum_token_probability",
            "output_entropy",
        }
        if set(value) != expected:
            raise DataValidationError("activation capture row keys differ from version 1")
        if any(
            type(value[name]) is not str
            for name in expected - {"maximum_token_probability", "output_entropy"}
        ) or any(
            isinstance(value[name], bool) or not isinstance(value[name], int | float)
            for name in ("maximum_token_probability", "output_entropy")
        ):
            raise DataValidationError("activation capture row has invalid JSON types")
        return cls(
            question_id=value["question_id"],
            benchmark=value["benchmark"],
            partition=value["partition"],
            prompt_id=value["prompt_id"],
            outcome=Outcome(value["outcome"]),
            semantic_group_id=value["semantic_group_id"],
            rendered_prompt_sha256=value["rendered_prompt_sha256"],
            prompt_token_ids_sha256=value["prompt_token_ids_sha256"],
            generation_record_sha256=value["generation_record_sha256"],
            maximum_token_probability=float(value["maximum_token_probability"]),
            output_entropy=float(value["output_entropy"]),
        )


@dataclass(frozen=True, slots=True)
class VerifiedActivationStore:
    directory: Path
    spec: ActivationStoreSpec
    rows_completed: int
    shard_count: int
    chain_head: str | None
    shard_fingerprints: Mapping[str, str]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def create_activation_store(directory: str | Path, spec: ActivationStoreSpec) -> None:
    destination = validate_active_study_artifact_paths(
        {"activation store": directory}
    )["activation store"]
    if destination.exists():
        raise FrozenArtifactError(f"refusing to overwrite activation store: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        (stage / "shards").mkdir()
        body = spec.to_dict()
        _write_json(stage / "spec.json", {**body, "spec_digest": stable_hash(body)})
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _load_spec(directory: Path) -> ActivationStoreSpec:
    spec_path = directory / "spec.json"
    if spec_path.is_symlink() or not spec_path.is_file():
        raise FrozenArtifactError("activation-store spec must be a regular file")
    try:
        value = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read activation-store spec: {exc}") from exc
    if not isinstance(value, dict):
        raise FrozenArtifactError("activation-store spec must be a mapping")
    digest = value.pop("spec_digest", None)
    if digest != stable_hash(value):
        raise FrozenArtifactError("activation-store spec digest mismatch")
    try:
        return ActivationStoreSpec.from_dict(value)
    except (TypeError, ValueError, DataValidationError) as exc:
        raise FrozenArtifactError(f"invalid activation-store spec: {exc}") from exc


def _shard_paths(directory: Path) -> tuple[Path, ...]:
    root = directory / "shards"
    if root.is_symlink() or not root.is_dir():
        raise FrozenArtifactError("activation store lacks a regular shard directory")
    paths = sorted(root.iterdir())
    indices: list[int] = []
    for path in paths:
        match = _SHARD.fullmatch(path.name)
        if path.is_symlink() or not path.is_dir() or match is None:
            raise FrozenArtifactError(f"unexpected activation shard: {path.name}")
        indices.append(int(match.group(1)))
    if indices != list(range(len(indices))):
        raise FrozenArtifactError("activation shard numbering is not contiguous")
    return tuple(paths)


def _read_rows(path: Path) -> tuple[tuple[ActivationCaptureRow, str], ...]:
    values: list[tuple[ActivationCaptureRow, str]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise DataValidationError("activation row envelope must be a mapping")
                body = dict(raw)
                row_digest = body.pop("row_digest", None)
                if row_digest != stable_hash(body):
                    raise DataValidationError("activation row digest mismatch")
                sequence = body.pop("sequence", None)
                previous = body.pop("previous_row_digest", None)
                if type(sequence) is not int or (
                    previous is not None and type(previous) is not str
                ):
                    raise DataValidationError("activation row chain fields are invalid")
                row = ActivationCaptureRow.from_dict(body)
                values.append((row, str(row_digest)))
    except (OSError, json.JSONDecodeError, TypeError, ValueError, DataValidationError) as exc:
        raise FrozenArtifactError(f"cannot read activation rows: {exc}") from exc
    return tuple(values)


def verify_activation_store(
    directory: str | Path,
    *,
    expected_spec: ActivationStoreSpec | None = None,
    require_complete: bool = False,
) -> VerifiedActivationStore:
    source = Path(directory)
    if source.is_symlink() or not source.is_dir():
        raise FrozenArtifactError("activation store must be a regular directory")
    if {path.name for path in source.iterdir()} != {"spec.json", "shards"}:
        raise FrozenArtifactError("activation store inventory differs")
    spec = _load_spec(source)
    if expected_spec is not None and spec != expected_spec:
        raise FrozenArtifactError("activation store differs from its expected spec")
    completed = 0
    previous_shard: str | None = None
    previous_row: str | None = None
    fingerprints: dict[str, str] = {}
    for index, shard in enumerate(_shard_paths(source)):
        inventory = {path.name for path in shard.iterdir()}
        if inventory != {"activations.npy", "rows.jsonl", "manifest.json"} or any(
            path.is_symlink() or not path.is_file() for path in shard.iterdir()
        ):
            raise FrozenArtifactError("activation shard inventory differs")
        try:
            manifest = json.loads((shard / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FrozenArtifactError(f"cannot read activation shard manifest: {exc}") from exc
        if not isinstance(manifest, dict):
            raise FrozenArtifactError("activation shard manifest must be a mapping")
        body = dict(manifest)
        shard_digest = body.pop("shard_digest", None)
        if shard_digest != stable_hash(body):
            raise FrozenArtifactError("activation shard digest mismatch")
        expected_keys = {
            "schema_version",
            "plan_identity",
            "shard_index",
            "start_sequence",
            "end_sequence",
            "row_count",
            "shape",
            "dtype",
            "activations_sha256",
            "rows_sha256",
            "previous_shard_digest",
            "final_row_digest",
        }
        if set(body) != expected_keys:
            raise FrozenArtifactError("activation shard manifest schema differs")
        shape = body["shape"]
        if (
            any(
                type(body[name]) is not int
                for name in (
                    "schema_version",
                    "shard_index",
                    "start_sequence",
                    "end_sequence",
                    "row_count",
                )
            )
            or not isinstance(shape, list)
            or len(shape) != 4
            or any(type(value) is not int or value < 0 for value in shape)
            or any(
                type(body[name]) is not str
                for name in (
                    "plan_identity",
                    "dtype",
                    "activations_sha256",
                    "rows_sha256",
                    "final_row_digest",
                )
            )
            or (
                body["previous_shard_digest"] is not None
                and type(body["previous_shard_digest"]) is not str
            )
        ):
            raise FrozenArtifactError("activation shard manifest has invalid JSON types")
        rows = _read_rows(shard / "rows.jsonl")
        try:
            activations = np.load(shard / "activations.npy", mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise FrozenArtifactError(f"cannot load activation shard array: {exc}") from exc
        expected_shape = (
            len(rows),
            len(spec.sites),
            len(spec.layers),
            spec.hidden_width,
        )
        if (
            body["schema_version"] != 1
            or not rows
            or body["plan_identity"] != spec.plan_identity
            or body["shard_index"] != index
            or body["start_sequence"] != completed
            or body["end_sequence"] != completed + len(rows)
            or body["row_count"] != len(rows)
            or body["shape"] != list(expected_shape)
            or body["dtype"] != spec.dtype
            or tuple(activations.shape) != expected_shape
            or activations.dtype != np.float16
            or not np.isfinite(activations).all()
            or body["activations_sha256"] != sha256_file(shard / "activations.npy")
            or body["rows_sha256"] != sha256_file(shard / "rows.jsonl")
            or body["previous_shard_digest"] != previous_shard
            or (rows and body["final_row_digest"] != rows[-1][1])
        ):
            raise FrozenArtifactError("activation shard differs from its manifest")
        with (shard / "rows.jsonl").open(encoding="utf-8") as handle:
            raw_rows = [json.loads(line) for line in handle]
        for offset, raw in enumerate(raw_rows):
            if (
                raw.get("sequence") != completed + offset
                or raw.get("previous_row_digest") != previous_row
            ):
                raise FrozenArtifactError("activation row chain is not contiguous")
            previous_row = str(raw["row_digest"])
        completed += len(rows)
        previous_shard = str(shard_digest)
        fingerprints[shard.name] = sha256_file(shard / "manifest.json")
    if completed > spec.expected_rows or (require_complete and completed != spec.expected_rows):
        raise FrozenArtifactError("activation store row count differs from its frozen plan")
    return VerifiedActivationStore(
        directory=source,
        spec=spec,
        rows_completed=completed,
        shard_count=len(fingerprints),
        chain_head=previous_shard,
        shard_fingerprints=MappingProxyType(fingerprints),
    )


def append_activation_shard(
    directory: str | Path,
    rows: Sequence[ActivationCaptureRow],
    activations: np.ndarray[Any, Any],
    *,
    expected_spec: ActivationStoreSpec,
) -> VerifiedActivationStore:
    source = validate_active_study_artifact_paths(
        {"activation store": directory}
    )["activation store"]
    if not rows:
        raise DataValidationError("activation shard cannot be empty")
    verified = verify_activation_store(source, expected_spec=expected_spec)
    values = np.asarray(activations)
    expected_shape = (
        len(rows),
        len(expected_spec.sites),
        len(expected_spec.layers),
        expected_spec.hidden_width,
    )
    if (
        tuple(values.shape) != expected_shape
        or not np.issubdtype(values.dtype, np.floating)
        or not np.isfinite(values).all()
        or verified.rows_completed + len(rows) > expected_spec.expected_rows
    ):
        raise DataValidationError("activation shard tensor differs from the frozen geometry")
    with np.errstate(over="ignore", invalid="ignore"):
        stored_values = values.astype(np.float16, copy=False)
    if not np.isfinite(stored_values).all():
        raise DataValidationError("activation shard overflows the frozen float16 representation")
    shard_index = verified.shard_count
    destination = source / "shards" / f"shard-{shard_index:05d}"
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{source.name}-{destination.name}.stage-",
            dir=source.parent,
        )
    )
    previous_row: str | None = None
    if shard_index:
        previous_manifest = json.loads(
            (
                source
                / "shards"
                / f"shard-{shard_index - 1:05d}"
                / "manifest.json"
            ).read_text(encoding="utf-8")
        )
        previous_row = str(previous_manifest["final_row_digest"])
    try:
        np.save(
            stage / "activations.npy",
            stored_values,
            allow_pickle=False,
        )
        with (stage / "rows.jsonl").open("x", encoding="utf-8") as handle:
            for offset, row in enumerate(rows):
                body = {
                    "sequence": verified.rows_completed + offset,
                    "previous_row_digest": previous_row,
                    **row.to_dict(),
                }
                row_digest = stable_hash(body)
                handle.write(json.dumps({**body, "row_digest": row_digest}, sort_keys=True) + "\n")
                previous_row = row_digest
            handle.flush()
            os.fsync(handle.fileno())
        body = {
            "schema_version": 1,
            "plan_identity": expected_spec.plan_identity,
            "shard_index": shard_index,
            "start_sequence": verified.rows_completed,
            "end_sequence": verified.rows_completed + len(rows),
            "row_count": len(rows),
            "shape": list(expected_shape),
            "dtype": expected_spec.dtype,
            "activations_sha256": sha256_file(stage / "activations.npy"),
            "rows_sha256": sha256_file(stage / "rows.jsonl"),
            "previous_shard_digest": verified.chain_head,
            "final_row_digest": previous_row,
        }
        _write_json(stage / "manifest.json", {**body, "shard_digest": stable_hash(body)})
        try:
            os.replace(stage, destination)
        except OSError as exc:
            if destination.exists():
                raise FrozenArtifactError("activation shard was appended concurrently") from exc
            raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return verify_activation_store(source, expected_spec=expected_spec)


def iter_activation_shards(
    directory: str | Path,
    *,
    expected_spec: ActivationStoreSpec,
    verified_store: VerifiedActivationStore | None = None,
) -> Iterator[tuple[tuple[ActivationCaptureRow, ...], np.ndarray[Any, Any]]]:
    source = Path(directory)
    verified = verified_store or verify_activation_store(
        source, expected_spec=expected_spec
    )
    if verified.directory != source or verified.spec != expected_spec:
        raise FrozenArtifactError("verified activation-store handle differs from request")
    for shard in _shard_paths(verified.directory):
        rows = tuple(row for row, _digest in _read_rows(shard / "rows.jsonl"))
        activations = np.load(shard / "activations.npy", mmap_mode="r", allow_pickle=False)
        activations.setflags(write=False)
        yield rows, activations

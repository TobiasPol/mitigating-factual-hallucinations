"""Signed, resumable native-VLLM capture shared by all RQ1 fold fits."""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
import torch
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from torch import Tensor

from mfh.artifact_namespace import validate_active_study_artifact_paths
from mfh.contracts import ActivationSite, Outcome, PromptSpec, Question, TokenScope
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.experiments.confirmatory_graders import validate_confirmatory_grader_bundle
from mfh.experiments.e3_construction import VerifiedE3ConstructionSnapshot
from mfh.experiments.e8_protected import _compose_e8_controller_features
from mfh.experiments.robustness_diagnostics import RobustnessDiagnosticPlan
from mfh.inference.architecture import HookKey
from mfh.inference.vllm_research import (
    VllmPromptFeatureCubeOutput,
    VllmTeacherForcedCubeOutput,
)
from mfh.inference.vllm_runtime import VllmRenderedPrompt
from mfh.methods.features import ActivationFeatureSchema, ActivationKind
from mfh.methods.probes import ProbeDataset
from mfh.provenance import canonical_json, sha256_file, sha256_path, stable_hash

_PROMPT_ID = "P0-neutral"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SHARD = re.compile(r"^shard-(\d{5})$")
_INVENTORY = frozenset({"plan.json", "run.lock", "shards"})


class RQ1CaptureRuntime(Protocol):
    def runtime_identity(self) -> Mapping[str, Any]: ...

    def render_prompt(
        self,
        prompt: PromptSpec,
        question: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> VllmRenderedPrompt: ...

    def prompt_feature_cube(
        self,
        rendered: VllmRenderedPrompt,
        *,
        layers: Sequence[int],
        sites: Sequence[ActivationSite],
    ) -> VllmPromptFeatureCubeOutput: ...

    def teacher_forced_cube(
        self,
        rendered: VllmRenderedPrompt,
        response: str,
        *,
        layers: Sequence[int],
        sites: Sequence[ActivationSite],
    ) -> VllmTeacherForcedCubeOutput: ...


@dataclass(frozen=True, slots=True)
class VerifiedRQ1Capture:
    directory: Path
    plan: Mapping[str, Any]
    rows_completed: int
    shard_count: int
    chain_head: str | None
    complete: bool


@dataclass(frozen=True, slots=True)
class RQ1CaptureData:
    verified: VerifiedRQ1Capture
    vector_dataset: ProbeDataset
    vector_activations: Mapping[HookKey, Tensor]
    artifact_sha256: str

    def __post_init__(self) -> None:
        if not self.verified.complete or not self.vector_activations:
            raise DataValidationError("RQ1 native capture data is incomplete")
        object.__setattr__(
            self, "vector_activations", MappingProxyType(dict(self.vector_activations))
        )


def rq1_capture_public_key(private_key_hex: str) -> str:
    if _SHA256.fullmatch(private_key_hex) is None:
        raise DataValidationError("RQ1 capture key must be one Ed25519 seed")
    try:
        private = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_key_hex))
    except ValueError as exc:
        raise DataValidationError(f"invalid RQ1 capture key: {exc}") from exc
    return private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()


def _question_digest(questions: Sequence[Question]) -> str:
    return stable_hash(
        [
            {
                "question_id": value.question_id,
                "benchmark": value.benchmark,
                "text": value.text,
                "aliases": list(value.aliases),
                "split": value.split,
                "entities": list(value.entities),
                "metadata": dict(value.metadata),
            }
            for value in questions
        ]
    )


def _source_rows(
    snapshot: VerifiedE3ConstructionSnapshot,
    *,
    assignment_groups: Mapping[str, str],
) -> tuple[dict[str, Any], ...]:
    schedule = {value.sequence: value for value in snapshot.schedule}
    rows: list[dict[str, Any]] = []
    for record in snapshot.generations:
        source = schedule.get(record.sequence)
        raw_output = record.evidence.get("raw_output")
        if source is None or source.question_id != record.question_id:
            raise FrozenArtifactError("RQ1 capture source differs from E3")
        if record.prompt_id != _PROMPT_ID:
            continue
        if (
            record.outcome not in {Outcome.CORRECT, Outcome.INCORRECT, Outcome.ABSTENTION}
            or not isinstance(raw_output, str)
            or not raw_output.strip()
            or record.question_id not in assignment_groups
        ):
            raise DataValidationError("RQ1 requires every P0 T-steer C/I/A response")
        rows.append(
            {
                "row_index": len(rows),
                "source_sequence": record.sequence,
                "source_record_sha256": stable_hash(record.to_dict()),
                "question_id": record.question_id,
                "semantic_group_id": assignment_groups[record.question_id],
                "outcome": record.outcome.value,
                "rendered_prompt_sha256": record.rendered_prompt_sha256,
                "prompt_token_ids_sha256": record.prompt_token_ids_sha256,
                "raw_output_sha256": hashlib.sha256(raw_output.encode()).hexdigest(),
            }
        )
    if set(assignment_groups) != {row["question_id"] for row in rows}:
        raise DataValidationError("RQ1 capture does not cover the exact T-steer assignment")
    return tuple(rows)


def _assignments(plan: RobustnessDiagnosticPlan) -> Mapping[str, str]:
    section = plan.body["rq1_generalization"]
    assert isinstance(section, Mapping)
    rows = section["assignments"]
    result = {
        str(row["question_id"]): str(row["semantic_group_id"])
        for row in rows
        if row["partition"] == "T-steer"
    }
    if len(result) != 30_000:
        raise FrozenArtifactError("RQ1 plan lacks its exact 30,000 T-steer assignments")
    return MappingProxyType(result)


def _hooks(value: Sequence[HookKey]) -> tuple[HookKey, ...]:
    hooks = tuple(sorted(value, key=lambda item: item.artifact_key))
    if not hooks or len(set(hooks)) != len(hooks):
        raise DataValidationError("RQ1 capture hooks are invalid")
    return hooks


def prepare_rq1_capture(
    directory: str | Path,
    *,
    plan: RobustnessDiagnosticPlan,
    snapshot: VerifiedE3ConstructionSnapshot,
    questions: Sequence[Question],
    prompt: PromptSpec,
    feature_schema: ActivationFeatureSchema,
    vector_hooks: Sequence[HookKey],
    runtime_identity: Mapping[str, Any],
    runtime_artifact_sha256: str,
    execution_public_key: str,
    shard_rows: int = 16,
) -> Mapping[str, Any]:
    """Freeze the one shared 30,000-row RQ1 capture plan."""

    destination = validate_active_study_artifact_paths({"RQ1 capture": directory})[
        "RQ1 capture"
    ]
    if destination.exists() or destination.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite RQ1 capture: {destination}")
    assignments = _assignments(plan)
    question_ids = {value.question_id for value in questions}
    hooks = _hooks(vector_hooks)
    if plan.path is None:
        raise FrozenArtifactError("RQ1 capture requires a packaged robustness plan")
    grader = validate_confirmatory_grader_bundle(
        plan.path / "sources" / "frozen-graders"
    )
    frozen_runtime = grader.directory / "runtime-attestation.json"
    reviewed_digest = plan.body["e1_provenance"]["reviewed_split_manifest_digest"]
    identity = dict(runtime_identity)
    hidden_width = snapshot.plan.get("hidden_width")
    if (
        prompt.prompt_id != _PROMPT_ID
        or feature_schema.prompt_id != _PROMPT_ID
        or feature_schema.benchmark != "triviaqa"
        or set(assignments) != question_ids
        or len(questions) != 30_000
        or type(shard_rows) is not int
        or shard_rows <= 0
        or _SHA256.fullmatch(runtime_artifact_sha256) is None
        or _SHA256.fullmatch(execution_public_key) is None
        or runtime_artifact_sha256
        != str(plan.body["m3_capture_runtime_artifact_sha256"])
        or runtime_artifact_sha256 != sha256_file(frozen_runtime)
        or execution_public_key != grader.scorer.execution_public_key
        or dict(snapshot.plan.get("runtime_identity", {})) != identity
        or type(hidden_width) is not int
        or hidden_width <= 0
        or feature_schema.activation_kind is not ActivationKind.FINAL_PROMPT
        or feature_schema.token_scope not in {None, TokenScope.FINAL_PROMPT}
        or feature_schema.split_manifest_digest != reviewed_digest
        or feature_schema.prompt_sha256 != hashlib.sha256(prompt.text.encode()).hexdigest()
        or feature_schema.model_repository != identity.get("model_repository")
        or feature_schema.model_revision != identity.get("model_revision")
        or feature_schema.quantization != identity.get("model_quantization")
    ):
        raise DataValidationError("RQ1 capture inputs differ from the frozen study")
    rows = _source_rows(snapshot, assignment_groups=assignments)
    body = {
        "schema_version": 1,
        "kind": "rq1-shared-native-fit-capture",
        "plan_digest": plan.plan_digest,
        "source_question_bundle_sha256": plan.body["source_artifact_sha256"][
            "triviaqa-development"
        ],
        "e3_construction_path": str(snapshot.directory.resolve()),
        "e3_construction_sha256": sha256_path(snapshot.directory),
        "e3_plan_identity": snapshot.plan["plan_identity"],
        "questions_sha256": _question_digest(questions),
        "prompt": {
            "prompt_id": prompt.prompt_id,
            "text_sha256": hashlib.sha256(prompt.text.encode()).hexdigest(),
        },
        "feature_schema": replace(feature_schema, partition="T-steer").to_dict(),
        "vector_hooks": [
            {"layer": value.layer, "site": value.site.value, "key": value.artifact_key}
            for value in hooks
        ],
        "runtime_identity": json.loads(canonical_json(dict(runtime_identity))),
        "runtime_artifact_sha256": runtime_artifact_sha256,
        "execution_public_key": execution_public_key,
        "hidden_width": hidden_width,
        "shard_rows": shard_rows,
        "expected_rows": len(rows),
        "source_rows": list(rows),
        "source_rows_sha256": stable_hash(list(rows)),
        "runner_source_sha256": sha256_file(Path(__file__)),
    }
    frozen = {**body, "capture_plan_identity": stable_hash(body)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        (stage / "shards").mkdir()
        (stage / "run.lock").touch()
        (stage / "plan.json").write_text(
            json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return MappingProxyType(frozen)


def _load_plan(directory: Path) -> dict[str, Any]:
    if (
        directory.is_symlink()
        or not directory.is_dir()
        or {value.name for value in directory.iterdir()} != _INVENTORY
        or any(value.is_symlink() for value in directory.iterdir())
    ):
        raise FrozenArtifactError("RQ1 capture inventory differs")
    try:
        value = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read RQ1 capture plan: {exc}") from exc
    if not isinstance(value, dict):
        raise FrozenArtifactError("RQ1 capture plan is not an object")
    body = dict(value)
    identity = body.pop("capture_plan_identity", None)
    if (
        identity != stable_hash(body)
        or body.get("schema_version") != 1
        or body.get("kind") != "rq1-shared-native-fit-capture"
        or body.get("runner_source_sha256") != sha256_file(Path(__file__))
        or body.get("source_rows_sha256") != stable_hash(body.get("source_rows"))
        or body.get("expected_rows") != len(body.get("source_rows", ()))
    ):
        raise FrozenArtifactError("RQ1 capture plan identity differs")
    return value


def _shards(directory: Path) -> tuple[Path, ...]:
    root = directory / "shards"
    if root.is_symlink() or not root.is_dir():
        raise FrozenArtifactError("RQ1 capture shard root is invalid")
    values = tuple(sorted(root.iterdir()))
    indices: list[int] = []
    for value in values:
        match = _SHARD.fullmatch(value.name)
        if match is None or value.is_symlink() or not value.is_dir():
            raise FrozenArtifactError("RQ1 capture shard inventory differs")
        indices.append(int(match.group(1)))
    if indices != list(range(len(indices))):
        raise FrozenArtifactError("RQ1 capture shards are not contiguous")
    return values


def _verify_signature(value: Mapping[str, Any], public_key: str) -> None:
    signed = dict(value)
    signature = signed.pop("signature", None)
    if not isinstance(signature, str) or re.fullmatch(r"[0-9a-f]{128}", signature) is None:
        raise FrozenArtifactError("RQ1 capture shard signature is invalid")
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key)).verify(
            bytes.fromhex(signature), canonical_json(signed).encode()
        )
    except (InvalidSignature, ValueError) as exc:
        raise FrozenArtifactError("RQ1 capture shard signature cannot be verified") from exc


def _payload(path: Path, expected_sha256: str) -> Mapping[str, np.ndarray[Any, Any]]:
    try:
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise FrozenArtifactError("RQ1 capture payload digest differs")
        with np.load(io.BytesIO(raw), allow_pickle=False) as value:
            arrays = {name: np.asarray(value[name]) for name in value.files}
    except (OSError, ValueError) as exc:
        raise FrozenArtifactError(f"cannot read RQ1 capture payload: {exc}") from exc
    return MappingProxyType(arrays)


def _assert_current(
    frozen: Mapping[str, Any],
    *,
    plan: RobustnessDiagnosticPlan,
    snapshot: VerifiedE3ConstructionSnapshot,
    questions: Sequence[Question],
    prompt: PromptSpec,
) -> None:
    rows = _source_rows(snapshot, assignment_groups=_assignments(plan))
    if (
        frozen["plan_digest"] != plan.plan_digest
        or frozen["e3_construction_sha256"] != sha256_path(snapshot.directory)
        or frozen["e3_plan_identity"] != snapshot.plan["plan_identity"]
        or frozen["questions_sha256"] != _question_digest(questions)
        or frozen["prompt"]
        != {
            "prompt_id": prompt.prompt_id,
            "text_sha256": hashlib.sha256(prompt.text.encode()).hexdigest(),
        }
        or frozen["source_rows"] != list(rows)
    ):
        raise FrozenArtifactError("RQ1 capture sources changed after preparation")


def verify_rq1_capture(
    directory: str | Path,
    *,
    plan: RobustnessDiagnosticPlan,
    snapshot: VerifiedE3ConstructionSnapshot,
    questions: Sequence[Question],
    prompt: PromptSpec,
    expected_execution_public_key: str,
    require_complete: bool = False,
) -> VerifiedRQ1Capture:
    raw_source = Path(directory).absolute()
    if raw_source.is_symlink():
        raise FrozenArtifactError("RQ1 capture root cannot be a symlink")
    source = raw_source.resolve()
    frozen = _load_plan(source)
    _assert_current(
        frozen, plan=plan, snapshot=snapshot, questions=questions, prompt=prompt
    )
    if frozen["execution_public_key"] != expected_execution_public_key:
        raise FrozenArtifactError("RQ1 capture trust anchor differs")
    feature_width = ActivationFeatureSchema.from_dict(frozen["feature_schema"]).width
    hooks = tuple(row["key"] for row in frozen["vector_hooks"])
    completed = 0
    previous: str | None = None
    for index, shard in enumerate(_shards(source)):
        if (
            {value.name for value in shard.iterdir()} != {"manifest.json", "payload.npz"}
            or any(value.is_symlink() or not value.is_file() for value in shard.iterdir())
        ):
            raise FrozenArtifactError("RQ1 capture shard files differ")
        manifest = json.loads((shard / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise FrozenArtifactError("RQ1 shard manifest is invalid")
        _verify_signature(manifest, expected_execution_public_key)
        signature = manifest.pop("signature")
        del signature
        digest = manifest.get("manifest_digest")
        body = dict(manifest)
        body.pop("manifest_digest", None)
        count = manifest.get("row_count")
        arrays = _payload(shard / "payload.npz", str(manifest.get("payload_sha256")))
        expected_keys = {"prompt_features", *(f"response.{key}" for key in hooks)}
        if (
            digest != stable_hash(body)
            or manifest.get("capture_plan_identity") != frozen["capture_plan_identity"]
            or manifest.get("shard_index") != index
            or manifest.get("start_row") != completed
            or type(count) is not int
            or count <= 0
            or manifest.get("end_row") != completed + count
            or manifest.get("previous_manifest_digest") != previous
            or manifest.get("rows") != frozen["source_rows"][completed : completed + count]
            or set(arrays) != expected_keys
            or arrays["prompt_features"].shape != (count, feature_width)
            or any(arrays[f"response.{key}"].shape[0] != count for key in hooks)
            or any(
                arrays[f"response.{key}"].shape
                != (count, int(frozen["hidden_width"]))
                for key in hooks
            )
            or any(
                value.dtype != np.float16 or not np.isfinite(value).all()
                for value in arrays.values()
            )
        ):
            raise FrozenArtifactError("RQ1 capture shard differs from its plan")
        completed += count
        previous = str(digest)
    complete = completed == frozen["expected_rows"]
    if completed > frozen["expected_rows"] or (require_complete and not complete):
        raise FrozenArtifactError("RQ1 capture row count differs")
    return VerifiedRQ1Capture(
        source,
        MappingProxyType(frozen),
        completed,
        len(_shards(source)),
        previous,
        complete,
    )


@contextmanager
def _lock(directory: Path) -> Iterator[None]:
    with (directory / "run.lock").open("r+b", buffering=0) as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FrozenArtifactError("RQ1 capture is already running") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _capture_row(
    runtime: RQ1CaptureRuntime,
    *,
    question: Question,
    prompt: PromptSpec,
    record: Any,
    expected: Mapping[str, Any],
    feature_schema: ActivationFeatureSchema,
    hooks: Sequence[HookKey],
    hidden_width: int,
) -> tuple[np.ndarray[Any, Any], Mapping[str, np.ndarray[Any, Any]], int]:
    rendered = runtime.render_prompt(prompt, question.text, metadata=question.metadata)
    if (
        rendered.sha256 != expected["rendered_prompt_sha256"]
        or rendered.token_ids_sha256 != expected["prompt_token_ids_sha256"]
    ):
        raise FrozenArtifactError("RQ1 rerendered prompt differs from E3")
    prompt_cube = runtime.prompt_feature_cube(
        rendered, layers=feature_schema.layers, sites=feature_schema.sites
    )
    feature = _compose_e8_controller_features(feature_schema, prompt_cube.activations)
    raw_output = record.evidence["raw_output"]
    response_cube = runtime.teacher_forced_cube(
        rendered,
        raw_output,
        layers=tuple(sorted({value.layer for value in hooks})),
        sites=tuple(sorted({value.site for value in hooks}, key=lambda item: item.value)),
    )
    if response_cube.response_text_sha256 != expected["raw_output_sha256"]:
        raise FrozenArtifactError("RQ1 teacher-forced response differs from E3")
    responses: dict[str, np.ndarray[Any, Any]] = {}
    for hook in hooks:
        value = np.asarray(response_cube.activations[hook.site][hook.layer], dtype=np.float32)
        if (
            value.ndim != 2
            or value.shape[0] != len(response_cube.response_token_ids)
            or value.shape[1] != hidden_width
        ):
            raise DataValidationError("RQ1 response activation geometry differs")
        responses[hook.artifact_key] = np.ascontiguousarray(
            value.mean(axis=0, dtype=np.float64), dtype=np.float32
        )
    return (
        np.ascontiguousarray(feature.numpy().reshape(-1), dtype=np.float32),
        MappingProxyType(responses),
        max(prompt_cube.peak_memory_bytes, response_cube.peak_memory_bytes),
    )


def run_rq1_capture(
    directory: str | Path,
    *,
    plan: RobustnessDiagnosticPlan,
    snapshot: VerifiedE3ConstructionSnapshot,
    questions: Sequence[Question],
    prompt: PromptSpec,
    runtime: RQ1CaptureRuntime,
    private_key_hex: str,
    limit: int | None = None,
) -> VerifiedRQ1Capture:
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise DataValidationError("RQ1 capture limit must be positive")
    raw_source = Path(directory).absolute()
    if raw_source.is_symlink():
        raise FrozenArtifactError("RQ1 capture root cannot be a symlink")
    source = raw_source.resolve()
    public_key = rq1_capture_public_key(private_key_hex)
    with _lock(source):
        verified = verify_rq1_capture(
            source,
            plan=plan,
            snapshot=snapshot,
            questions=questions,
            prompt=prompt,
            expected_execution_public_key=public_key,
        )
        if dict(runtime.runtime_identity()) != dict(verified.plan["runtime_identity"]):
            raise FrozenArtifactError("RQ1 live runtime differs from capture plan")
        remaining = int(verified.plan["expected_rows"]) - verified.rows_completed
        target = verified.rows_completed + min(remaining, remaining if limit is None else limit)
        records = {value.sequence: value for value in snapshot.generations}
        questions_by_id = {value.question_id: value for value in questions}
        schema = ActivationFeatureSchema.from_dict(verified.plan["feature_schema"])
        hooks = tuple(
            HookKey(int(row["layer"]), ActivationSite(str(row["site"])))
            for row in verified.plan["vector_hooks"]
        )
        private = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_key_hex))
        while verified.rows_completed < target:
            count = min(int(verified.plan["shard_rows"]), target - verified.rows_completed)
            rows = verified.plan["source_rows"][
                verified.rows_completed : verified.rows_completed + count
            ]
            prompt_values: list[np.ndarray[Any, Any]] = []
            response_values: dict[str, list[np.ndarray[Any, Any]]] = {
                value.artifact_key: [] for value in hooks
            }
            maximum_peak = 0
            for row in rows:
                record = records[row["source_sequence"]]
                if stable_hash(record.to_dict()) != row["source_record_sha256"]:
                    raise FrozenArtifactError("RQ1 E3 record changed during capture")
                features, responses, peak = _capture_row(
                    runtime,
                    question=questions_by_id[row["question_id"]],
                    prompt=prompt,
                    record=record,
                    expected=row,
                    feature_schema=schema,
                    hooks=hooks,
                    hidden_width=int(verified.plan["hidden_width"]),
                )
                prompt_values.append(features)
                for key, value in responses.items():
                    response_values[key].append(value)
                maximum_peak = max(maximum_peak, peak)
            index = verified.shard_count
            destination = source / "shards" / f"shard-{index:05d}"
            stage = Path(
                tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent)
            )
            try:
                arrays = {
                    "prompt_features": np.asarray(prompt_values, dtype=np.float16),
                    **{
                        f"response.{key}": np.asarray(values, dtype=np.float16)
                        for key, values in response_values.items()
                    },
                }
                if any(not np.isfinite(value).all() for value in arrays.values()):
                    raise DataValidationError("RQ1 capture overflows float16")
                save_arrays: Any = np.savez_compressed
                save_arrays(stage / "payload.npz", **arrays)
                body = {
                    "schema_version": 1,
                    "capture_plan_identity": verified.plan["capture_plan_identity"],
                    "shard_index": index,
                    "start_row": verified.rows_completed,
                    "end_row": verified.rows_completed + count,
                    "row_count": count,
                    "previous_manifest_digest": verified.chain_head,
                    "payload_sha256": sha256_file(stage / "payload.npz"),
                    "maximum_peak_memory_bytes": maximum_peak,
                    "rows": [dict(value) for value in rows],
                }
                signed = {**body, "manifest_digest": stable_hash(body)}
                manifest = {
                    **signed,
                    "signature": private.sign(canonical_json(signed).encode()).hex(),
                }
                (stage / "manifest.json").write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                os.replace(stage, destination)
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
            completed = verified.rows_completed + count
            verified = VerifiedRQ1Capture(
                directory=verified.directory,
                plan=verified.plan,
                rows_completed=completed,
                shard_count=verified.shard_count + 1,
                chain_head=str(signed["manifest_digest"]),
                complete=completed == int(verified.plan["expected_rows"]),
            )
        if verified.complete:
            return verify_rq1_capture(
                source,
                plan=plan,
                snapshot=snapshot,
                questions=questions,
                prompt=prompt,
                expected_execution_public_key=public_key,
                require_complete=True,
            )
        return verified


def load_rq1_capture_data(
    directory: str | Path,
    *,
    plan: RobustnessDiagnosticPlan,
    snapshot: VerifiedE3ConstructionSnapshot,
    questions: Sequence[Question],
    prompt: PromptSpec,
    expected_execution_public_key: str,
) -> RQ1CaptureData:
    verified = verify_rq1_capture(
        directory,
        plan=plan,
        snapshot=snapshot,
        questions=questions,
        prompt=prompt,
        expected_execution_public_key=expected_execution_public_key,
        require_complete=True,
    )
    prompt_parts: list[np.ndarray[Any, Any]] = []
    response_parts: dict[str, list[np.ndarray[Any, Any]]] = {
        str(row["key"]): [] for row in verified.plan["vector_hooks"]
    }
    for shard in _shards(verified.directory):
        manifest = json.loads((shard / "manifest.json").read_text(encoding="utf-8"))
        arrays = _payload(shard / "payload.npz", manifest["payload_sha256"])
        prompt_parts.append(np.asarray(arrays["prompt_features"], dtype=np.float32))
        for key in response_parts:
            response_parts[key].append(np.asarray(arrays[f"response.{key}"], dtype=np.float32))
    rows = verified.plan["source_rows"]
    dataset = ProbeDataset(
        question_ids=tuple(str(row["question_id"]) for row in rows),
        features=torch.from_numpy(np.concatenate(prompt_parts)),
        outcomes=tuple(Outcome(str(row["outcome"])) for row in rows),
        group_ids=tuple(str(row["semantic_group_id"]) for row in rows),
        feature_schema=ActivationFeatureSchema.from_dict(verified.plan["feature_schema"]),
    )
    hooks = {
        str(row["key"]): HookKey(int(row["layer"]), ActivationSite(str(row["site"])))
        for row in verified.plan["vector_hooks"]
    }
    return RQ1CaptureData(
        verified=verified,
        vector_dataset=dataset,
        vector_activations=MappingProxyType(
            {
                hook: torch.from_numpy(np.concatenate(response_parts[key]))
                for key, hook in hooks.items()
            }
        ),
        artifact_sha256=sha256_path(directory),
    )

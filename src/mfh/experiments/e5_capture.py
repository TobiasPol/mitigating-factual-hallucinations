"""Signed, resumable native-MLX capture for fitting the E5 controller grid."""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
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
from mfh.contracts import ActivationSite, Outcome, PromptSpec, Question, Runtime, TokenScope
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.experiments.e2_controller_inputs import E2ControllerInputView
from mfh.experiments.e3_construction import (
    E3GenerationRecord,
    VerifiedE3ConstructionSnapshot,
)
from mfh.experiments.e5_adaptive import E5Protocol
from mfh.experiments.e5_types import E5FitRecipe
from mfh.experiments.model_selection import ACTIVE_MODEL_IDENTITIES, ACTIVE_MODEL_NAME
from mfh.inference.architecture import HookKey
from mfh.inference.mlx_research import (
    MlxPromptFeatureCubeOutput,
    MlxTeacherForcedCubeOutput,
)
from mfh.inference.mlx_runtime import MlxRenderedPrompt
from mfh.methods.features import (
    ActivationFeatureSchema,
    ActivationKind,
    FeatureComposition,
)
from mfh.methods.probes import ProbeDataset
from mfh.provenance import canonical_json, sha256_file, sha256_path, stable_hash

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
_SHARD = re.compile(r"^shard-(\d{5})$")
_SHARD_STAGE = re.compile(r"^\.shard-\d{5}\.stage-[A-Za-z0-9._-]+$")
_INVENTORY = frozenset({"plan.json", "run.lock", "shards"})
_PROMPT_ID = "P0-neutral"


class E5FitCaptureRuntime(Protocol):
    """Minimal native runtime surface needed by the E5 fit capture."""

    def runtime_identity(self) -> Mapping[str, Any]: ...

    def render_prompt(
        self,
        prompt: PromptSpec,
        question: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> MlxRenderedPrompt: ...

    def prompt_feature_cube(
        self,
        rendered: MlxRenderedPrompt,
        *,
        layers: Sequence[int],
        sites: Sequence[ActivationSite],
    ) -> MlxPromptFeatureCubeOutput: ...

    def teacher_forced_cube(
        self,
        rendered: MlxRenderedPrompt,
        response: str,
        *,
        layers: Sequence[int],
        sites: Sequence[ActivationSite],
    ) -> MlxTeacherForcedCubeOutput: ...


@dataclass(frozen=True, slots=True)
class VerifiedE5FitCapture:
    directory: Path
    plan: Mapping[str, Any]
    pairs_completed: int
    shard_count: int
    chain_head: str | None
    complete: bool
    scientific_eligible: bool
    maximum_peak_memory_bytes: int


@dataclass(frozen=True, slots=True)
class E5FitCaptureData:
    verified: VerifiedE5FitCapture
    vector_datasets: Mapping[FeatureComposition, ProbeDataset]
    vector_activations: Mapping[HookKey, Tensor]
    capture_artifact_sha256: str

    def __post_init__(self) -> None:
        if (
            not self.verified.complete
            or not self.vector_datasets
            or not self.vector_activations
            or _SHA256.fullmatch(self.capture_artifact_sha256) is None
        ):
            raise DataValidationError("E5 fit capture data is incomplete")
        object.__setattr__(self, "vector_datasets", MappingProxyType(dict(self.vector_datasets)))
        object.__setattr__(
            self, "vector_activations", MappingProxyType(dict(self.vector_activations))
        )


def _exact_json(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    try:
        result = json.loads(
            json.dumps(dict(value), sort_keys=True, allow_nan=False, separators=(",", ":"))
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"{label} is not exact JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise DataValidationError(f"{label} must be a JSON mapping")
    return result


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise DataValidationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _public_key(private_key_hex: str) -> str:
    _digest(private_key_hex, "E5 capture private key")
    try:
        key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_key_hex))
    except ValueError as exc:
        raise DataValidationError(f"invalid E5 capture private key: {exc}") from exc
    return (
        key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        .hex()
    )


def e5_capture_public_key(private_key_hex: str) -> str:
    """Derive the pinned public trust anchor without exposing key bytes in artifacts."""

    return _public_key(private_key_hex)


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


def _prompt_digest(prompts: Mapping[str, PromptSpec]) -> str:
    return stable_hash(
        [
            {
                "mapping_key": name,
                "prompt_id": value.prompt_id,
                "text": value.text,
                "permits_abstention": value.permits_abstention,
                "deployment_eligible": value.deployment_eligible,
            }
            for name, value in sorted(prompts.items())
        ]
    )


def _source_rows(
    snapshot: VerifiedE3ConstructionSnapshot,
) -> tuple[dict[str, Any], ...]:
    schedule = {value.sequence: value for value in snapshot.schedule}
    rows: list[dict[str, Any]] = []
    for record in snapshot.generations:
        source = schedule.get(record.sequence)
        raw_output = record.evidence.get("raw_output")
        if (
            source is None
            or source.question_id != record.question_id
            or source.prompt_id != record.prompt_id
        ):
            raise FrozenArtifactError("E5 capture source differs from the E3 schedule")
        if (
            record.prompt_id != _PROMPT_ID
            or record.outcome not in {Outcome.CORRECT, Outcome.INCORRECT}
            or not isinstance(raw_output, str)
            or not raw_output.strip()
        ):
            continue
        rows.append(
            {
                "pair_index": len(rows),
                "source_sequence": record.sequence,
                "source_record_sha256": stable_hash(record.to_dict()),
                "question_id": record.question_id,
                "semantic_group_id": source.semantic_group_id,
                "outcome": record.outcome.value,
                "rendered_prompt_sha256": record.rendered_prompt_sha256,
                "prompt_token_ids_sha256": record.prompt_token_ids_sha256,
                "raw_output_sha256": hashlib.sha256(raw_output.encode()).hexdigest(),
            }
        )
    if not rows or len({value["question_id"] for value in rows}) != len(rows):
        raise DataValidationError("E5 capture requires unique non-empty P0 C/I T-steer rows")
    return tuple(rows)


def _view_dict(view: E2ControllerInputView) -> dict[str, Any]:
    return {
        "composition": view.composition.value,
        "layers": list(view.layers),
        "site": view.site.value,
    }


def _views_from_plan(plan: Mapping[str, Any]) -> tuple[E2ControllerInputView, ...]:
    raw = plan.get("controller_input_views")
    if not isinstance(raw, list):
        raise FrozenArtifactError("E5 capture views are missing")
    try:
        return tuple(
            E2ControllerInputView(
                composition=FeatureComposition(value["composition"]),
                layers=tuple(value["layers"]),
                site=ActivationSite(value["site"]),
            )
            for value in raw
        )
    except (KeyError, TypeError, ValueError, DataValidationError) as exc:
        raise FrozenArtifactError(f"invalid E5 capture view: {exc}") from exc


def _validate_context(
    *,
    snapshot: VerifiedE3ConstructionSnapshot,
    questions: Sequence[Question],
    prompts: Mapping[str, PromptSpec],
    views: tuple[E2ControllerInputView, ...],
    recipe: E5FitRecipe,
) -> tuple[tuple[dict[str, Any], ...], tuple[int, ...]]:
    if (
        type(snapshot) is not VerifiedE3ConstructionSnapshot
        or len(snapshot.generations) != len(snapshot.schedule)
        or set(prompts) != {"P0-neutral", "P2-calibrated-abstention"}
        or any(name != value.prompt_id for name, value in prompts.items())
        or type(views) is not tuple
        or not views
        or len({value.composition for value in views}) != len(views)
        or type(recipe) is not E5FitRecipe
        or any(value.site is not recipe.intervention_site for value in views)
    ):
        raise DataValidationError("E5 capture source, prompts, views, or recipe are invalid")
    source_ids = {value.question_id for value in questions}
    if (
        len(source_ids) != len(questions)
        or any(value.benchmark != "triviaqa" or value.split != "T-steer" for value in questions)
        or source_ids != {value.question_id for value in snapshot.schedule}
    ):
        raise DataValidationError("E5 capture questions differ from frozen T-steer")
    layers = tuple(
        sorted(
            {
                *recipe.three_layer_candidates,
                *(layer for view in views for layer in view.layers),
            }
        )
    )
    if any(layer not in layers for layer in recipe.three_layer_candidates):
        raise DataValidationError("E5 capture candidate layers are incomplete")
    return _source_rows(snapshot), layers


def _plan_body(
    *,
    snapshot: VerifiedE3ConstructionSnapshot,
    questions: Sequence[Question],
    prompts: Mapping[str, PromptSpec],
    protocol: E5Protocol,
    views: tuple[E2ControllerInputView, ...],
    recipe: E5FitRecipe,
    runtime_identity: Mapping[str, Any],
    execution_public_key: str,
    runtime_artifact_sha256: str,
    e2_probe_bundle_sha256: str,
    e3_static_vectors_sha256: str,
    split_manifest_digest: str,
    hidden_width: int,
    shard_rows: int,
    max_peak_memory_bytes: int,
) -> dict[str, Any]:
    rows, layers = _validate_context(
        snapshot=snapshot,
        questions=questions,
        prompts=prompts,
        views=views,
        recipe=recipe,
    )
    identity = _exact_json(runtime_identity, label="E5 capture runtime identity")
    source_identity = snapshot.plan.get("runtime_identity")
    if not isinstance(source_identity, Mapping) or identity != _exact_json(
        source_identity, label="E5 source runtime identity"
    ):
        raise FrozenArtifactError("E5 capture runtime identity differs from verified E3")
    for value, label in (
        (execution_public_key, "E5 capture public key"),
        (runtime_artifact_sha256, "E5 runtime artifact"),
        (e2_probe_bundle_sha256, "E5 E2 probe bundle"),
        (e3_static_vectors_sha256, "E5 E3 static vectors"),
        (split_manifest_digest, "E5 split manifest"),
    ):
        _digest(value, label)
    repository = identity.get("model_repository")
    revision = identity.get("model_revision")
    quantization = identity.get("model_quantization")
    active = ACTIVE_MODEL_IDENTITIES[ACTIVE_MODEL_NAME]
    provenance = identity.get("research_provenance")
    receipt_bound = bool(
        isinstance(provenance, Mapping)
        and provenance.get("runtime_preflight_receipt_sha256") == runtime_artifact_sha256
    )
    exact_active_model = bool(
        repository == active["repository"]
        and revision == active["revision"]
        and quantization == active["quantization"]
        and identity.get("model_num_layers") == active["num_layers"]
    )
    if (
        type(protocol) is not E5Protocol
        or type(repository) is not str
        or not repository.strip()
        or type(revision) is not str
        or _REVISION.fullmatch(revision) is None
        or type(quantization) is not str
        or not quantization.strip()
        or type(hidden_width) is not int
        or hidden_width <= 0
        or type(shard_rows) is not int
        or shard_rows <= 0
        or type(max_peak_memory_bytes) is not int
        or max_peak_memory_bytes <= 0
    ):
        raise DataValidationError("E5 capture runtime geometry or resource limits are invalid")
    source_digest = sha256_path(snapshot.directory)
    p0 = prompts[_PROMPT_ID]
    return {
        "schema_version": 1,
        "phase": "E5-native-fit-capture",
        "runner": "resumable-signed-native-mlx-prompt-response-pairs-v1",
        "runner_source_sha256": sha256_file(Path(__file__)),
        "protocol": protocol.to_dict(),
        "recipe": recipe.to_dict(),
        "controller_input_views": [_view_dict(value) for value in views],
        "capture_layers": list(layers),
        "capture_site": recipe.intervention_site.value,
        "hidden_width": hidden_width,
        "shard_rows": shard_rows,
        "max_peak_memory_bytes": max_peak_memory_bytes,
        "expected_pairs": len(rows),
        "source_rows": list(rows),
        "source_rows_sha256": stable_hash(list(rows)),
        "questions_sha256": _question_digest(questions),
        "prompts_sha256": _prompt_digest(prompts),
        "p0_template_sha256": hashlib.sha256(p0.text.encode()).hexdigest(),
        "runtime_identity": identity,
        "execution_public_key": execution_public_key,
        "runtime_artifact_sha256": runtime_artifact_sha256,
        "split_manifest_digest": split_manifest_digest,
        "e2_probe_bundle_sha256": e2_probe_bundle_sha256,
        "e3_static_vectors_sha256": e3_static_vectors_sha256,
        "e3_construction_path": str(snapshot.directory.resolve()),
        "e3_construction_sha256": source_digest,
        "e3_plan_identity": snapshot.plan["plan_identity"],
        "e3_generation_chain_head": snapshot.generation_chain_head,
        "scientific_eligible": bool(
            protocol.scientific_eligible
            and snapshot.scientific_eligible
            and hidden_width == 5_120
            and max_peak_memory_bytes <= 48 * 1024**3
            and exact_active_model
            and receipt_bound
        ),
    }


def prepare_e5_fit_capture(
    directory: str | Path,
    *,
    snapshot: VerifiedE3ConstructionSnapshot,
    questions: Sequence[Question],
    prompts: Mapping[str, PromptSpec],
    views: tuple[E2ControllerInputView, ...],
    recipe: E5FitRecipe,
    runtime_identity: Mapping[str, Any],
    execution_public_key: str,
    runtime_artifact_sha256: str,
    e2_probe_bundle_sha256: str,
    e3_static_vectors_sha256: str,
    split_manifest_digest: str,
    protocol: E5Protocol | None = None,
    hidden_width: int = 5_120,
    shard_rows: int = 16,
    max_peak_memory_bytes: int = 48 * 1024**3,
) -> Mapping[str, Any]:
    """Freeze an empty E5 native-capture workspace without loading the model."""

    destination = validate_active_study_artifact_paths({"E5 fit capture": directory})[
        "E5 fit capture"
    ]
    frozen_protocol = protocol or E5Protocol()
    body = _plan_body(
        snapshot=snapshot,
        questions=questions,
        prompts=prompts,
        protocol=frozen_protocol,
        views=views,
        recipe=recipe,
        runtime_identity=runtime_identity,
        execution_public_key=execution_public_key,
        runtime_artifact_sha256=runtime_artifact_sha256,
        e2_probe_bundle_sha256=e2_probe_bundle_sha256,
        e3_static_vectors_sha256=e3_static_vectors_sha256,
        split_manifest_digest=split_manifest_digest,
        hidden_width=hidden_width,
        shard_rows=shard_rows,
        max_peak_memory_bytes=max_peak_memory_bytes,
    )
    plan = {**body, "plan_identity": stable_hash(body)}
    if destination.exists() or destination.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite E5 fit capture: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        (stage / "shards").mkdir()
        (stage / "run.lock").touch()
        (stage / "plan.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return MappingProxyType(plan)


def _load_plan(directory: Path) -> dict[str, Any]:
    if (
        directory.is_symlink()
        or not directory.is_dir()
        or {value.name for value in directory.iterdir()} != _INVENTORY
    ):
        raise FrozenArtifactError("E5 fit capture inventory differs")
    path = directory / "plan.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot load E5 capture plan: {exc}") from exc
    if not isinstance(value, dict) or type(value.get("plan_identity")) is not str:
        raise FrozenArtifactError("E5 capture plan schema is invalid")
    body = dict(value)
    identity = body.pop("plan_identity")
    expected = {
        "schema_version",
        "phase",
        "runner",
        "runner_source_sha256",
        "protocol",
        "recipe",
        "controller_input_views",
        "capture_layers",
        "capture_site",
        "hidden_width",
        "shard_rows",
        "max_peak_memory_bytes",
        "expected_pairs",
        "source_rows",
        "source_rows_sha256",
        "questions_sha256",
        "prompts_sha256",
        "p0_template_sha256",
        "runtime_identity",
        "execution_public_key",
        "runtime_artifact_sha256",
        "split_manifest_digest",
        "e2_probe_bundle_sha256",
        "e3_static_vectors_sha256",
        "e3_construction_path",
        "e3_construction_sha256",
        "e3_plan_identity",
        "e3_generation_chain_head",
        "scientific_eligible",
    }
    if (
        set(body) != expected
        or identity != stable_hash(body)
        or body.get("schema_version") != 1
        or body.get("phase") != "E5-native-fit-capture"
        or body.get("runner") != "resumable-signed-native-mlx-prompt-response-pairs-v1"
        or body.get("runner_source_sha256") != sha256_file(Path(__file__))
        or type(body.get("source_rows")) is not list
        or body.get("source_rows_sha256") != stable_hash(body["source_rows"])
        or body.get("expected_pairs") != len(body["source_rows"])
    ):
        raise FrozenArtifactError("E5 capture plan differs from exact replay")
    try:
        E5Protocol.from_dict(body["protocol"])
        E5FitRecipe(
            fixed_best_layer=body["recipe"]["fixed_best_layer"],
            two_layer_candidates=tuple(body["recipe"]["two_layer_candidates"]),
            three_layer_candidates=tuple(body["recipe"]["three_layer_candidates"]),
            intervention_site=ActivationSite(body["recipe"]["intervention_site"]),
            minimum_class_count=body["recipe"]["minimum_class_count"],
            vector_seed=body["recipe"]["vector_seed"],
            router_seed=body["recipe"]["router_seed"],
            router_hidden_width=body["recipe"]["router_hidden_width"],
            router_epochs=body["recipe"]["router_epochs"],
            distance_temperature=body["recipe"]["distance_temperature"],
            layer_seed=body["recipe"]["layer_seed"],
            layer_epochs=body["recipe"]["layer_epochs"],
            alpha_max=body["recipe"]["alpha_max"],
            alpha_beta=body["recipe"]["alpha_beta"],
            alpha_threshold=body["recipe"]["alpha_threshold"],
            schema_version=body["recipe"]["schema_version"],
        )
        _views_from_plan(value)
    except (KeyError, TypeError, ValueError, DataValidationError) as exc:
        raise FrozenArtifactError(f"invalid E5 capture plan value: {exc}") from exc
    return value


def _assert_source_current(
    plan: Mapping[str, Any],
    *,
    snapshot: VerifiedE3ConstructionSnapshot,
    questions: Sequence[Question],
    prompts: Mapping[str, PromptSpec],
) -> None:
    current_rows = list(_source_rows(snapshot))
    if (
        str(snapshot.directory.resolve()) != plan["e3_construction_path"]
        or sha256_path(snapshot.directory) != plan["e3_construction_sha256"]
        or snapshot.plan["plan_identity"] != plan["e3_plan_identity"]
        or dict(snapshot.plan["runtime_identity"]) != plan["runtime_identity"]
        or snapshot.generation_chain_head != plan["e3_generation_chain_head"]
        or current_rows != plan["source_rows"]
        or _question_digest(questions) != plan["questions_sha256"]
        or _prompt_digest(prompts) != plan["prompts_sha256"]
        or hashlib.sha256(prompts[_PROMPT_ID].text.encode()).hexdigest()
        != plan["p0_template_sha256"]
    ):
        raise FrozenArtifactError("E5 capture source changed after preparation")


def _shards(directory: Path) -> tuple[Path, ...]:
    root = directory / "shards"
    if root.is_symlink() or not root.is_dir():
        raise FrozenArtifactError("E5 capture shard directory is invalid")
    values = sorted(root.iterdir())
    indices: list[int] = []
    for value in values:
        match = _SHARD.fullmatch(value.name)
        if match is None or value.is_symlink() or not value.is_dir():
            raise FrozenArtifactError(f"unexpected E5 capture shard: {value.name}")
        indices.append(int(match.group(1)))
    if indices != list(range(len(indices))):
        raise FrozenArtifactError("E5 capture shard numbering is not contiguous")
    return tuple(values)


def _remove_abandoned_shard_stages(directory: Path) -> None:
    """Remove only process-private shard stages after the execution lock is held."""

    root = directory / "shards"
    if root.is_symlink() or not root.is_dir():
        raise FrozenArtifactError("E5 capture shard directory is invalid")
    for value in root.iterdir():
        if _SHARD_STAGE.fullmatch(value.name) is None:
            continue
        if value.is_symlink() or not value.is_dir():
            raise FrozenArtifactError("E5 abandoned shard stage is not a directory")
        shutil.rmtree(value)


@contextmanager
def _capture_execution_lock(directory: Path) -> Iterator[None]:
    """Hold the process-wide capture lock for cleanup, verification, and appends."""

    if directory.is_symlink() or not directory.is_dir():
        raise FrozenArtifactError("E5 fit capture directory is invalid")
    lock_path = directory / "run.lock"
    if lock_path.is_symlink() or not lock_path.is_file() or lock_path.stat().st_size != 0:
        raise FrozenArtifactError("E5 capture execution lock is invalid")
    with lock_path.open("r+b", buffering=0) as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FrozenArtifactError("E5 fit capture is already running") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _valid_utc_timestamp(value: object) -> bool:
    if type(value) is not str:
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == UTC.utcoffset(None)


def _verify_signature(value: Mapping[str, Any], public_key_hex: str) -> None:
    signature = value.get("signature")
    signed = dict(value)
    signed.pop("signature", None)
    if not isinstance(signature, str) or re.fullmatch(r"[0-9a-f]{128}", signature) is None:
        raise FrozenArtifactError("E5 capture shard signature is invalid")
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex)).verify(
            bytes.fromhex(signature), canonical_json(signed).encode()
        )
    except (InvalidSignature, ValueError) as exc:
        raise FrozenArtifactError("E5 capture shard signature cannot be verified") from exc


def _read_payload(path: Path, *, expected_sha256: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise FrozenArtifactError("E5 capture payload digest mismatch")
        with np.load(io.BytesIO(payload), allow_pickle=False) as values:
            if set(values.files) != {"prompt_features", "response_means"}:
                raise FrozenArtifactError("E5 capture payload inventory differs")
            prompt = np.asarray(values["prompt_features"])
            response = np.asarray(values["response_means"])
    except (OSError, ValueError) as exc:
        raise FrozenArtifactError(f"cannot read E5 capture payload: {exc}") from exc
    return prompt, response


def verify_e5_fit_capture(
    directory: str | Path,
    *,
    snapshot: VerifiedE3ConstructionSnapshot,
    questions: Sequence[Question],
    prompts: Mapping[str, PromptSpec],
    expected_execution_public_key: str,
    require_complete: bool = False,
) -> VerifiedE5FitCapture:
    """Replay every source, tensor, hash-chain, memory, and native signature check."""

    source = Path(directory)
    plan = _load_plan(source)
    _digest(expected_execution_public_key, "E5 trusted capture public key")
    if plan["execution_public_key"] != expected_execution_public_key:
        raise FrozenArtifactError("E5 capture plan key differs from its external trust root")
    lock_path = source / "run.lock"
    if lock_path.is_symlink() or not lock_path.is_file() or lock_path.stat().st_size != 0:
        raise FrozenArtifactError("E5 capture execution lock is invalid")
    _assert_source_current(plan, snapshot=snapshot, questions=questions, prompts=prompts)
    layers = plan["capture_layers"]
    if (
        type(layers) is not list
        or not layers
        or any(type(value) is not int or value < 0 for value in layers)
        or len(set(layers)) != len(layers)
        or type(plan["hidden_width"]) is not int
        or type(plan["max_peak_memory_bytes"]) is not int
    ):
        raise FrozenArtifactError("E5 capture geometry is invalid")
    completed = 0
    previous: str | None = None
    maximum_peak = 0
    for index, shard in enumerate(_shards(source)):
        if {value.name for value in shard.iterdir()} != {"manifest.json", "payload.npz"} or any(
            value.is_symlink() or not value.is_file() for value in shard.iterdir()
        ):
            raise FrozenArtifactError("E5 capture shard inventory differs")
        try:
            manifest = json.loads((shard / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FrozenArtifactError(f"cannot read E5 capture manifest: {exc}") from exc
        if not isinstance(manifest, dict):
            raise FrozenArtifactError("E5 capture manifest must be a mapping")
        _verify_signature(manifest, plan["execution_public_key"])
        signed = dict(manifest)
        signed.pop("signature")
        digest = signed.pop("manifest_digest", None)
        rows = signed.get("rows")
        expected_keys = {
            "schema_version",
            "plan_identity",
            "shard_index",
            "start_pair",
            "end_pair",
            "pair_count",
            "shape",
            "dtype",
            "payload_sha256",
            "previous_manifest_digest",
            "maximum_peak_memory_bytes",
            "execution_session_id",
            "execution_session_started_at",
            "execution_session_wall_seconds",
            "execution_lock_identity",
            "execution_process_id",
            "shard_completed_at",
            "rows",
        }
        if (
            set(signed) != expected_keys
            or digest != stable_hash(signed)
            or signed["schema_version"] != 1
            or signed["plan_identity"] != plan["plan_identity"]
            or signed["shard_index"] != index
            or signed["start_pair"] != completed
            or type(signed["pair_count"]) is not int
            or signed["pair_count"] <= 0
            or signed["end_pair"] != completed + signed["pair_count"]
            or signed["previous_manifest_digest"] != previous
            or not isinstance(rows, list)
            or len(rows) != signed["pair_count"]
            or rows != plan["source_rows"][completed : signed["end_pair"]]
            or type(signed["maximum_peak_memory_bytes"]) is not int
            or not 0 <= signed["maximum_peak_memory_bytes"] <= plan["max_peak_memory_bytes"]
            or type(signed["execution_session_id"]) is not str
            or _SHA256.fullmatch(signed["execution_session_id"]) is None
            or not _valid_utc_timestamp(signed["execution_session_started_at"])
            or type(signed["execution_session_wall_seconds"]) is not float
            or not np.isfinite(signed["execution_session_wall_seconds"])
            or signed["execution_session_wall_seconds"] < 0.0
            or type(signed["execution_lock_identity"]) is not str
            or _SHA256.fullmatch(signed["execution_lock_identity"]) is None
            or type(signed["execution_process_id"]) is not int
            or signed["execution_process_id"] <= 0
            or not _valid_utc_timestamp(signed["shard_completed_at"])
            or type(signed["payload_sha256"]) is not str
            or _SHA256.fullmatch(signed["payload_sha256"]) is None
        ):
            raise FrozenArtifactError("E5 capture manifest differs from its frozen plan")
        prompt_values, response_values = _read_payload(
            shard / "payload.npz", expected_sha256=signed["payload_sha256"]
        )
        expected_shape = (
            signed["pair_count"],
            len(layers),
            plan["hidden_width"],
        )
        if (
            signed["shape"] != list(expected_shape)
            or signed["dtype"] != "float16"
            or prompt_values.shape != expected_shape
            or response_values.shape != expected_shape
            or prompt_values.dtype != np.float16
            or response_values.dtype != np.float16
            or not np.isfinite(prompt_values).all()
            or not np.isfinite(response_values).all()
        ):
            raise FrozenArtifactError("E5 capture tensor geometry differs")
        completed = signed["end_pair"]
        maximum_peak = max(maximum_peak, signed["maximum_peak_memory_bytes"])
        previous = str(digest)
    complete = completed == plan["expected_pairs"]
    if completed > plan["expected_pairs"] or (require_complete and not complete):
        raise FrozenArtifactError("E5 capture row count differs from its frozen plan")
    return VerifiedE5FitCapture(
        directory=source.resolve(),
        plan=MappingProxyType(plan),
        pairs_completed=completed,
        shard_count=len(_shards(source)),
        chain_head=previous,
        complete=complete,
        scientific_eligible=bool(plan["scientific_eligible"] and complete),
        maximum_peak_memory_bytes=maximum_peak,
    )


def _validated_vector(value: object, *, width: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1 or array.shape[0] != width or not np.isfinite(array).all():
        raise DataValidationError(f"{label} activation geometry is invalid")
    return array


def _capture_pair(
    *,
    runtime: E5FitCaptureRuntime,
    question: Question,
    prompt: PromptSpec,
    record: E3GenerationRecord,
    layers: tuple[int, ...],
    site: ActivationSite,
    width: int,
    expected: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, int]:
    rendered = runtime.render_prompt(prompt, question.text, metadata=question.metadata)
    if (
        rendered.sha256 != expected["rendered_prompt_sha256"]
        or rendered.token_ids_sha256 != expected["prompt_token_ids_sha256"]
    ):
        raise FrozenArtifactError("E5 rerendered prompt differs from E3")
    prompt_cube = runtime.prompt_feature_cube(rendered, layers=layers, sites=(site,))
    raw_output = record.evidence["raw_output"]
    response_cube = runtime.teacher_forced_cube(rendered, raw_output, layers=layers, sites=(site,))
    if response_cube.response_text_sha256 != expected["raw_output_sha256"]:
        raise DataValidationError("E5 response capture differs from E3 output")
    prompt_values: list[np.ndarray] = []
    response_values: list[np.ndarray] = []
    for layer in layers:
        raw_prompt = np.asarray(prompt_cube.activations[site][layer], dtype=np.float32)
        raw_response = np.asarray(response_cube.activations[site][layer], dtype=np.float32)
        if raw_prompt.shape != (1, width) or (
            raw_response.ndim != 2
            or raw_response.shape[0] != len(response_cube.response_token_ids)
            or raw_response.shape[1] != width
            or not np.isfinite(raw_response).all()
        ):
            raise DataValidationError("E5 native capture activation geometry differs")
        prompt_values.append(_validated_vector(raw_prompt[0], width=width, label="E5 prompt"))
        response_values.append(
            _validated_vector(
                raw_response.mean(axis=0, dtype=np.float64),
                width=width,
                label="E5 response mean",
            )
        )
    return (
        np.stack(prompt_values, axis=0),
        np.stack(response_values, axis=0),
        max(prompt_cube.peak_memory_bytes, response_cube.peak_memory_bytes),
    )


def _append_shard(
    directory: Path,
    *,
    verified: VerifiedE5FitCapture,
    prompt_values: np.ndarray,
    response_values: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    maximum_peak_memory_bytes: int,
    private_key_hex: str,
    execution_session_id: str,
    execution_session_started_at: str,
    execution_session_wall_seconds: float,
    execution_lock_identity: str,
    execution_process_id: int,
) -> VerifiedE5FitCapture:
    if _public_key(private_key_hex) != verified.plan["execution_public_key"]:
        raise DataValidationError("E5 capture private key differs from its frozen public key")
    prompt = np.asarray(prompt_values, dtype=np.float32)
    response = np.asarray(response_values, dtype=np.float32)
    expected_shape = (
        len(rows),
        len(verified.plan["capture_layers"]),
        verified.plan["hidden_width"],
    )
    if (
        not rows
        or prompt.shape != expected_shape
        or response.shape != expected_shape
        or not np.isfinite(prompt).all()
        or not np.isfinite(response).all()
        or maximum_peak_memory_bytes > verified.plan["max_peak_memory_bytes"]
        or _SHA256.fullmatch(execution_session_id) is None
        or not _valid_utc_timestamp(execution_session_started_at)
        or not np.isfinite(execution_session_wall_seconds)
        or execution_session_wall_seconds < 0.0
        or _SHA256.fullmatch(execution_lock_identity) is None
        or type(execution_process_id) is not int
        or execution_process_id <= 0
    ):
        raise DataValidationError("E5 capture shard differs from frozen limits")
    with np.errstate(over="ignore", invalid="ignore"):
        prompt16 = prompt.astype(np.float16)
        response16 = response.astype(np.float16)
    if not np.isfinite(prompt16).all() or not np.isfinite(response16).all():
        raise DataValidationError("E5 capture activation overflows float16")
    index = verified.shard_count
    destination = directory / "shards" / f"shard-{index:05d}"
    if destination.exists() or destination.is_symlink():
        raise FrozenArtifactError("refusing to overwrite an E5 capture shard")
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=directory / "shards"))
    try:
        np.savez_compressed(
            stage / "payload.npz",
            prompt_features=prompt16,
            response_means=response16,
        )
        body = {
            "schema_version": 1,
            "plan_identity": verified.plan["plan_identity"],
            "shard_index": index,
            "start_pair": verified.pairs_completed,
            "end_pair": verified.pairs_completed + len(rows),
            "pair_count": len(rows),
            "shape": list(expected_shape),
            "dtype": "float16",
            "payload_sha256": sha256_file(stage / "payload.npz"),
            "previous_manifest_digest": verified.chain_head,
            "maximum_peak_memory_bytes": maximum_peak_memory_bytes,
            "execution_session_id": execution_session_id,
            "execution_session_started_at": execution_session_started_at,
            "execution_session_wall_seconds": float(execution_session_wall_seconds),
            "execution_lock_identity": execution_lock_identity,
            "execution_process_id": execution_process_id,
            "shard_completed_at": _utc_now(),
            "rows": [dict(value) for value in rows],
        }
        signed = {**body, "manifest_digest": stable_hash(body)}
        private_key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_key_hex))
        manifest = {
            **signed,
            "signature": private_key.sign(canonical_json(signed).encode()).hex(),
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    completed = verified.pairs_completed + len(rows)
    complete = completed == verified.plan["expected_pairs"]
    return VerifiedE5FitCapture(
        directory=verified.directory,
        plan=verified.plan,
        pairs_completed=completed,
        shard_count=verified.shard_count + 1,
        chain_head=str(signed["manifest_digest"]),
        complete=complete,
        scientific_eligible=bool(verified.plan["scientific_eligible"] and complete),
        maximum_peak_memory_bytes=max(
            verified.maximum_peak_memory_bytes, maximum_peak_memory_bytes
        ),
    )


def run_e5_fit_capture(
    directory: str | Path,
    *,
    snapshot: VerifiedE3ConstructionSnapshot,
    questions: Sequence[Question],
    prompts: Mapping[str, PromptSpec],
    runtime: E5FitCaptureRuntime,
    private_key_hex: str,
    request_budget: int | None = None,
) -> VerifiedE5FitCapture:
    """Capture a bounded prefix; repeat safely until ``complete`` is true."""

    if request_budget is not None and (type(request_budget) is not int or request_budget <= 0):
        raise DataValidationError("E5 capture request budget must be positive")
    source = Path(directory)
    trusted_public_key = _public_key(private_key_hex)
    session_started_at = _utc_now()
    session_started_monotonic = time.perf_counter()
    session_nonce = time.time_ns()
    process_id = os.getpid()
    with _capture_execution_lock(source):
        _remove_abandoned_shard_stages(source)
        verified = verify_e5_fit_capture(
            source,
            snapshot=snapshot,
            questions=questions,
            prompts=prompts,
            expected_execution_public_key=trusted_public_key,
        )
        execution_session_id = stable_hash(
            {
                "plan_identity": verified.plan["plan_identity"],
                "started_at": session_started_at,
                "nonce": session_nonce,
                "process_id": process_id,
            }
        )
        execution_lock_identity = stable_hash(
            {
                "plan_identity": verified.plan["plan_identity"],
                "execution_session_id": execution_session_id,
                "lock_path": "run.lock",
                "process_id": process_id,
            }
        )
        return _run_e5_fit_capture_locked(
            source,
            snapshot=snapshot,
            questions=questions,
            prompts=prompts,
            runtime=runtime,
            private_key_hex=private_key_hex,
            request_budget=request_budget,
            verified=verified,
            execution_session_id=execution_session_id,
            execution_session_started_at=session_started_at,
            session_started_monotonic=session_started_monotonic,
            execution_lock_identity=execution_lock_identity,
            execution_process_id=process_id,
        )


def _run_e5_fit_capture_locked(
    directory: Path,
    *,
    snapshot: VerifiedE3ConstructionSnapshot,
    questions: Sequence[Question],
    prompts: Mapping[str, PromptSpec],
    runtime: E5FitCaptureRuntime,
    private_key_hex: str,
    request_budget: int | None,
    verified: VerifiedE5FitCapture,
    execution_session_id: str,
    execution_session_started_at: str,
    session_started_monotonic: float,
    execution_lock_identity: str,
    execution_process_id: int,
) -> VerifiedE5FitCapture:
    """Append a bounded prefix while the caller holds the execution lock."""

    if _exact_json(runtime.runtime_identity(), label="E5 live runtime identity") != dict(
        verified.plan["runtime_identity"]
    ):
        raise FrozenArtifactError("E5 live MLX runtime identity differs from its plan")
    if _public_key(private_key_hex) != verified.plan["execution_public_key"]:
        raise DataValidationError("E5 capture signing key differs from its plan")
    if verified.complete:
        return verified
    remaining = verified.plan["expected_pairs"] - verified.pairs_completed
    budget = remaining if request_budget is None else min(remaining, request_budget)
    records = {value.sequence: value for value in snapshot.generations}
    question_map = {value.question_id: value for value in questions}
    layers = tuple(verified.plan["capture_layers"])
    site = ActivationSite(verified.plan["capture_site"])
    target = verified.pairs_completed + budget
    while verified.pairs_completed < target:
        count = min(
            target - verified.pairs_completed,
            verified.plan["shard_rows"],
        )
        start = verified.pairs_completed
        expected_rows = verified.plan["source_rows"][start : start + count]
        prompt_batch: list[np.ndarray] = []
        response_batch: list[np.ndarray] = []
        maximum_peak = 0
        for expected in expected_rows:
            record = records[expected["source_sequence"]]
            if stable_hash(record.to_dict()) != expected["source_record_sha256"]:
                raise FrozenArtifactError("E5 live source record differs from its plan")
            prompt_values, response_values, peak = _capture_pair(
                runtime=runtime,
                question=question_map[expected["question_id"]],
                prompt=prompts[_PROMPT_ID],
                record=record,
                layers=layers,
                site=site,
                width=verified.plan["hidden_width"],
                expected=expected,
            )
            if peak > verified.plan["max_peak_memory_bytes"]:
                raise DataValidationError("E5 native capture exceeded the 48 GiB envelope")
            prompt_batch.append(prompt_values)
            response_batch.append(response_values)
            maximum_peak = max(maximum_peak, peak)
        verified = _append_shard(
            directory,
            verified=verified,
            prompt_values=np.stack(prompt_batch),
            response_values=np.stack(response_batch),
            rows=expected_rows,
            maximum_peak_memory_bytes=maximum_peak,
            private_key_hex=private_key_hex,
            execution_session_id=execution_session_id,
            execution_session_started_at=execution_session_started_at,
            execution_session_wall_seconds=time.perf_counter() - session_started_monotonic,
            execution_lock_identity=execution_lock_identity,
            execution_process_id=execution_process_id,
        )
    return verified


def _compose(values: np.ndarray, view: E2ControllerInputView) -> np.ndarray:
    if view.composition is FeatureComposition.SINGLE_LAYER:
        result = values[:, 0, :]
    elif view.composition is FeatureComposition.CONCATENATED_LAYERS:
        result = values.reshape(values.shape[0], -1)
    else:
        result = np.diff(values, axis=1).reshape(values.shape[0], -1)
    output = np.ascontiguousarray(result, dtype=np.float32)
    if output.ndim != 2 or not np.isfinite(output).all():
        raise FrozenArtifactError("E5 composed T-steer features are invalid")
    return output


def load_e5_fit_capture_data(
    directory: str | Path,
    *,
    snapshot: VerifiedE3ConstructionSnapshot,
    questions: Sequence[Question],
    prompts: Mapping[str, PromptSpec],
    expected_execution_public_key: str,
) -> E5FitCaptureData:
    """Materialize the verified T-steer datasets after the MLX model is unloaded."""

    verified = verify_e5_fit_capture(
        directory,
        snapshot=snapshot,
        questions=questions,
        prompts=prompts,
        expected_execution_public_key=expected_execution_public_key,
        require_complete=True,
    )
    prompt_parts: list[np.ndarray] = []
    response_parts: list[np.ndarray] = []
    for shard in _shards(Path(directory)):
        manifest = json.loads((shard / "manifest.json").read_text(encoding="utf-8"))
        prompt_values, response_values = _read_payload(
            shard / "payload.npz", expected_sha256=manifest["payload_sha256"]
        )
        prompt_parts.append(np.asarray(prompt_values, dtype=np.float32))
        response_parts.append(np.asarray(response_values, dtype=np.float32))
    prompt_cube = np.concatenate(prompt_parts, axis=0)
    response_cube = np.concatenate(response_parts, axis=0)
    layers = tuple(verified.plan["capture_layers"])
    layer_index = {value: index for index, value in enumerate(layers)}
    source_rows = verified.plan["source_rows"]
    question_ids = tuple(value["question_id"] for value in source_rows)
    outcomes = tuple(Outcome(value["outcome"]) for value in source_rows)
    group_ids = tuple(value["semantic_group_id"] for value in source_rows)
    identity = verified.plan["runtime_identity"]
    site = ActivationSite(verified.plan["capture_site"])
    datasets: dict[FeatureComposition, ProbeDataset] = {}
    for view in _views_from_plan(verified.plan):
        positions = [layer_index[value] for value in view.layers]
        features = _compose(prompt_cube[:, positions, :], view)
        schema = ActivationFeatureSchema(
            benchmark="triviaqa",
            partition="T-steer",
            split_manifest_digest=verified.plan["split_manifest_digest"],
            model_repository=identity["model_repository"],
            model_revision=identity["model_revision"],
            runtime=Runtime.MLX,
            quantization=identity["model_quantization"],
            prompt_id=_PROMPT_ID,
            prompt_sha256=verified.plan["p0_template_sha256"],
            activation_kind=ActivationKind.FINAL_PROMPT,
            layers=view.layers,
            sites=(view.site,),
            composition=view.composition,
            width=features.shape[1],
            token_scope=TokenScope.FINAL_PROMPT,
        )
        datasets[view.composition] = ProbeDataset(
            question_ids=question_ids,
            features=torch.from_numpy(features),
            outcomes=outcomes,
            group_ids=group_ids,
            feature_schema=schema,
        )
    recipe = verified.plan["recipe"]
    vector_activations = {
        HookKey(layer, site): torch.from_numpy(
            np.ascontiguousarray(response_cube[:, layer_index[layer], :], dtype=np.float32)
        )
        for layer in recipe["three_layer_candidates"]
    }
    return E5FitCaptureData(
        verified=verified,
        vector_datasets=datasets,
        vector_activations=vector_activations,
        capture_artifact_sha256=sha256_path(directory),
    )

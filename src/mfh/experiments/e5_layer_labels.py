"""Signed native-VLLM counterfactual labels for E5 layer routers."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from mfh.artifact_namespace import validate_active_study_artifact_paths
from mfh.contracts import ActivationSite, Outcome, PromptSpec, Question, TokenScope
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.evaluation.grading import deterministic_short_answer_grade
from mfh.experiments.e5_capture import VerifiedE5FitCapture
from mfh.experiments.e5_types import E5FitRecipe
from mfh.experiments.model_selection import ACTIVE_MODEL_IDENTITIES, ACTIVE_MODEL_NAME
from mfh.experiments.static_direction_sources import (
    ResolvedStaticDirection,
    resolve_static_direction,
)
from mfh.inference.vllm_runtime import VllmGenerationOutput, VllmRenderedPrompt, as_numpy
from mfh.methods.features import FeatureComposition
from mfh.methods.probes import ProbeDataset
from mfh.provenance import canonical_json, sha256_file, sha256_path, stable_hash

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SHARD = re.compile(r"^shard-(\d{5})$")
_SHARD_STAGE = re.compile(r"^\.shard-\d{5}\.stage-[A-Za-z0-9._-]+$")
_INVENTORY = frozenset({"plan.json", "run.lock", "shards"})
_PROMPT_ID = "P0-neutral"
_PROMPT_SHA256 = "5b08080e81d4032d853d3a30fda35e7670da6a81cc1ccf17f2b8fc5001bba684"
_MAX_MEMORY_BYTES = 40 * 1024**3
_LABEL_RULE = "outcome-rank-C-A-I-then-fixed-best-then-recipe-order-v1"
_ACTIVE_MODEL = ACTIVE_MODEL_IDENTITIES[ACTIVE_MODEL_NAME]
_VERIFIED_LAYER_LABEL_DATA = object()
_VERIFIED_LAYER_LABEL_RECEIPTS: dict[int, Mapping[str, Any]] = {}


class E5LayerLabelRuntime(Protocol):
    """Native runtime surface required by the counterfactual label capture."""

    def runtime_identity(self) -> Mapping[str, Any]: ...

    def render_prompt(
        self,
        prompt: PromptSpec,
        question: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> VllmRenderedPrompt: ...

    def standardized_intervention_state(
        self,
        direction: np.ndarray[Any, Any],
        *,
        standardized_alpha: float,
        reference_rms: float,
        token_scope: TokenScope,
        decay: float = 0.0,
    ) -> Any: ...

    def generate_with_interventions(
        self,
        rendered: VllmRenderedPrompt,
        *,
        max_new_tokens: int,
        intervention_states: Mapping[tuple[int, ActivationSite], Any],
    ) -> VllmGenerationOutput: ...


@dataclass(frozen=True, slots=True)
class VerifiedE5LayerLabels:
    directory: Path
    plan: Mapping[str, Any]
    records_completed: int
    shard_count: int
    chain_head: str | None
    complete: bool
    scientific_eligible: bool
    maximum_peak_memory_bytes: int


@dataclass(frozen=True, slots=True)
class E5LayerLabelData:
    verified: VerifiedE5LayerLabels
    question_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    outcomes: tuple[Outcome, ...]
    best_layers_two: tuple[int, ...]
    best_layers_three: tuple[int, ...]
    artifact_sha256: str
    _verification_token: object | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "question_ids", tuple(self.question_ids))
        object.__setattr__(self, "group_ids", tuple(self.group_ids))
        object.__setattr__(self, "outcomes", tuple(self.outcomes))
        object.__setattr__(self, "best_layers_two", tuple(self.best_layers_two))
        object.__setattr__(self, "best_layers_three", tuple(self.best_layers_three))
        count = len(self.question_ids)
        if (
            not self.verified.complete
            or count == 0
            or len(set(self.question_ids)) != count
            or len(self.group_ids) != count
            or len(self.outcomes) != count
            or len(self.best_layers_two) != count
            or len(self.best_layers_three) != count
            or _SHA256.fullmatch(self.artifact_sha256) is None
        ):
            raise DataValidationError("E5 layer-label data is incomplete")

    def _binding_digest(self) -> str:
        return stable_hash(
            {
                "verified": {
                    "directory": str(self.verified.directory.resolve()),
                    "plan": dict(self.verified.plan),
                    "records_completed": self.verified.records_completed,
                    "shard_count": self.verified.shard_count,
                    "chain_head": self.verified.chain_head,
                    "complete": self.verified.complete,
                    "scientific_eligible": self.verified.scientific_eligible,
                    "maximum_peak_memory_bytes": self.verified.maximum_peak_memory_bytes,
                },
                "question_ids": list(self.question_ids),
                "group_ids": list(self.group_ids),
                "outcomes": [value.value for value in self.outcomes],
                "best_layers_two": list(self.best_layers_two),
                "best_layers_three": list(self.best_layers_three),
                "artifact_sha256": self.artifact_sha256,
            }
        )

    def assert_authorized(self) -> Mapping[str, Any]:
        """Require construction by the complete signed-capture loader."""

        receipt = _VERIFIED_LAYER_LABEL_RECEIPTS.get(id(self))
        if (
            self._verification_token is not _VERIFIED_LAYER_LABEL_DATA
            or receipt is None
            or receipt["object"] is not self
            or tuple(receipt["binding_objects"])
            != (
                id(self.verified),
                id(self.verified.plan),
                id(self.question_ids),
                id(self.group_ids),
                id(self.outcomes),
                id(self.best_layers_two),
                id(self.best_layers_three),
            )
            or receipt["binding_digest"] != self._binding_digest()
        ):
            raise FrozenArtifactError(
                "E5 layer-label data was not verifier-authorized or changed in memory"
            )
        return receipt

    def assert_current(self) -> None:
        """Require both the authorized handle and its exact source directory."""

        self.assert_authorized()
        if sha256_path(self.verified.directory) != self.artifact_sha256:
            raise FrozenArtifactError("E5 layer-label source changed after verification")


def _digest(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise DataValidationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _public_key(private_key_hex: str) -> str:
    _digest(private_key_hex, "E5 layer-label private key")
    try:
        private = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_key_hex))
    except ValueError as exc:
        raise DataValidationError(f"invalid E5 layer-label private key: {exc}") from exc
    return (
        private.public_key()
        .public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        .hex()
    )


def e5_layer_label_public_key(private_key_hex: str) -> str:
    """Derive the external public trust anchor for label verification."""

    return _public_key(private_key_hex)


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


def _question_body(question: Question) -> dict[str, Any]:
    return {
        "question_id": question.question_id,
        "benchmark": question.benchmark,
        "text": question.text,
        "aliases": list(question.aliases),
        "split": question.split,
        "entities": list(question.entities),
        "metadata": dict(question.metadata),
    }


def _question_digest(questions: Sequence[Question]) -> str:
    return stable_hash([_question_body(value) for value in questions])


def _prompt_digest(prompt: PromptSpec) -> str:
    return stable_hash(
        {
            "prompt_id": prompt.prompt_id,
            "text": prompt.text,
        }
    )


def _controller_identity(
    datasets: Mapping[FeatureComposition, ProbeDataset],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[Outcome, ...], dict[str, Any]]:
    expected = set(FeatureComposition)
    if set(datasets) != expected:
        raise DataValidationError("E5 layer labels require all controller compositions")
    reference = datasets[FeatureComposition.SINGLE_LAYER]
    identity = (reference.question_ids, reference.group_ids, reference.outcomes)
    if (
        not reference.question_ids
        or len(set(reference.question_ids)) != len(reference.question_ids)
        or any(
            (value.question_ids, value.group_ids, value.outcomes) != identity
            for value in datasets.values()
        )
        or any(
            value.feature_schema is None
            or value.feature_schema.partition != "T-controller-train"
            or value.feature_schema.composition is not composition
            for composition, value in datasets.items()
        )
        or any(
            outcome not in {Outcome.CORRECT, Outcome.INCORRECT, Outcome.ABSTENTION}
            for outcome in reference.outcomes
        )
    ):
        raise DataValidationError("E5 controller rows are incomplete or misaligned")
    receipts = {
        composition.value: {
            "data_fingerprint": value.data_fingerprint,
            "feature_schema_digest": value.feature_schema.digest,
            "row_count": len(value.question_ids),
        }
        for composition, value in sorted(datasets.items(), key=lambda item: item[0].value)
        if value.feature_schema is not None
    }
    return (*identity, receipts)


def _direction_receipts(
    directory: Path, recipe: E5FitRecipe
) -> tuple[dict[int, ResolvedStaticDirection], dict[str, Any]]:
    directions = {
        layer: resolve_static_direction(
            directory,
            method="M1",
            layer=layer,
            site=recipe.intervention_site,
        )
        for layer in recipe.three_layer_candidates
    }
    receipts = {
        str(layer): {
            "direction_sha256": value.direction_sha256,
            "direction_norm": value.direction_norm,
            "reference_rms": value.reference_rms,
            "source_kind": value.source_kind,
        }
        for layer, value in directions.items()
    }
    return directions, receipts


def _runtime_is_scientific(identity: Mapping[str, Any]) -> bool:
    provenance = identity.get("research_provenance")
    return bool(
        identity.get("model_repository") == _ACTIVE_MODEL["repository"]
        and identity.get("model_revision") == _ACTIVE_MODEL["revision"]
        and identity.get("model_quantization") == _ACTIVE_MODEL["quantization"]
        and identity.get("num_layers") == _ACTIVE_MODEL["num_layers"]
        and isinstance(provenance, Mapping)
        and _SHA256.fullmatch(str(provenance.get("runtime_preflight_receipt_sha256"))) is not None
    )


def _schedule_rows(
    questions: Sequence[Question],
    *,
    question_ids: Sequence[str],
    group_ids: Sequence[str],
    outcomes: Sequence[Outcome],
    candidate_layers: Sequence[int],
) -> list[dict[str, Any]]:
    if tuple(value.question_id for value in questions) != tuple(question_ids):
        raise DataValidationError("E5 label questions differ from controller row order")
    rows: list[dict[str, Any]] = []
    for question_index, (question, group_id, source_outcome) in enumerate(
        zip(questions, group_ids, outcomes, strict=True)
    ):
        if question.benchmark != "triviaqa":
            raise DataValidationError("E5 layer labels are restricted to TriviaQA")
        fingerprint = stable_hash(_question_body(question))
        for layer in candidate_layers:
            rows.append(
                {
                    "sequence": len(rows),
                    "question_index": question_index,
                    "question_id": question.question_id,
                    "semantic_group_id": group_id,
                    "source_outcome": source_outcome.value,
                    "question_sha256": fingerprint,
                    "layer": layer,
                }
            )
    return rows


def prepare_e5_layer_label_capture(
    directory: str | Path,
    *,
    questions: Sequence[Question],
    prompt: PromptSpec,
    controller_datasets: Mapping[FeatureComposition, ProbeDataset],
    fit_capture: VerifiedE5FitCapture,
    fit_capture_artifact_sha256: str,
    e3_static_vectors_directory: str | Path,
    recipe: E5FitRecipe,
    runtime_identity: Mapping[str, Any],
    execution_public_key: str,
    shard_rows: int = 64,
    max_new_tokens: int = 48,
    max_peak_memory_bytes: int = _MAX_MEMORY_BYTES,
) -> Mapping[str, Any]:
    """Freeze the exact counterfactual layer-label schedule without loading VLLM."""

    destination = validate_active_study_artifact_paths({"E5 layer-label capture": directory})[
        "E5 layer-label capture"
    ]
    if type(recipe) is not E5FitRecipe:
        raise DataValidationError("E5 layer labels require an exact fit recipe")
    if (
        prompt.prompt_id != _PROMPT_ID
        or hashlib.sha256(prompt.text.encode()).hexdigest() != _PROMPT_SHA256
        or type(shard_rows) is not int
        or shard_rows <= 0
        or type(max_new_tokens) is not int
        or not 0 < max_new_tokens <= 48
        or type(max_peak_memory_bytes) is not int
        or max_peak_memory_bytes <= 0
        or not fit_capture.complete
        or fit_capture.chain_head is None
        or sha256_path(fit_capture.directory) != fit_capture_artifact_sha256
        or fit_capture.plan.get("recipe") != recipe.to_dict()
        or fit_capture.plan.get("execution_public_key") != execution_public_key
    ):
        raise DataValidationError("E5 layer-label preparation inputs differ")
    _digest(execution_public_key, "E5 layer-label public key")
    _digest(fit_capture_artifact_sha256, "E5 fit-capture artifact")
    identity = _exact_json(runtime_identity, label="E5 layer-label runtime identity")
    if identity != dict(fit_capture.plan["runtime_identity"]):
        raise DataValidationError("E5 layer-label runtime differs from fit capture")
    question_ids, group_ids, outcomes, controller_receipts = _controller_identity(
        controller_datasets
    )
    static_directory = Path(e3_static_vectors_directory).resolve()
    static_sha256 = sha256_path(static_directory)
    if static_sha256 != fit_capture.plan.get("e3_static_vectors_sha256"):
        raise DataValidationError("E5 layer-label vectors differ from fit capture")
    _directions, direction_receipts = _direction_receipts(static_directory, recipe)
    rows = _schedule_rows(
        questions,
        question_ids=question_ids,
        group_ids=group_ids,
        outcomes=outcomes,
        candidate_layers=recipe.three_layer_candidates,
    )
    body = {
        "schema_version": 1,
        "phase": "E5-native-layer-label-capture",
        "runner": "resumable-signed-native-vllm-counterfactual-layer-labels-v1",
        "runner_source_sha256": sha256_file(Path(__file__)),
        "recipe": recipe.to_dict(),
        "label_rule": _LABEL_RULE,
        "candidate_layers": list(recipe.three_layer_candidates),
        "two_layer_candidates": list(recipe.two_layer_candidates),
        "fixed_best_layer": recipe.fixed_best_layer,
        "intervention_site": recipe.intervention_site.value,
        "standardized_alpha": float(recipe.alpha_max),
        "token_scope": TokenScope.FINAL_PROMPT.value,
        "max_new_tokens": max_new_tokens,
        "shard_rows": shard_rows,
        "max_peak_memory_bytes": max_peak_memory_bytes,
        "expected_records": len(rows),
        "expected_questions": len(question_ids),
        "schedule": rows,
        "schedule_sha256": stable_hash(rows),
        "questions_sha256": _question_digest(questions),
        "prompt_sha256": _prompt_digest(prompt),
        "prompt_template_sha256": _PROMPT_SHA256,
        "controller_identity_sha256": stable_hash(
            {
                "question_ids": list(question_ids),
                "group_ids": list(group_ids),
                "outcomes": [value.value for value in outcomes],
            }
        ),
        "controller_datasets": controller_receipts,
        "fit_capture_path": str(fit_capture.directory.resolve()),
        "fit_capture_artifact_sha256": fit_capture_artifact_sha256,
        "fit_capture_plan_identity": fit_capture.plan["plan_identity"],
        "fit_capture_chain_head": fit_capture.chain_head,
        "e3_static_vectors_path": str(static_directory),
        "e3_static_vectors_sha256": static_sha256,
        "directions": direction_receipts,
        "runtime_identity": identity,
        "execution_public_key": execution_public_key,
        "scientific_eligible": bool(
            fit_capture.scientific_eligible
            and _runtime_is_scientific(identity)
            and max_peak_memory_bytes == _MAX_MEMORY_BYTES
            and max_new_tokens == 48
        ),
    }
    plan = {**body, "plan_identity": stable_hash(body)}
    if destination.exists() or destination.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite E5 layer labels: {destination}")
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
        raise FrozenArtifactError("E5 layer-label inventory differs")
    try:
        value = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot load E5 layer-label plan: {exc}") from exc
    if not isinstance(value, dict):
        raise FrozenArtifactError("E5 layer-label plan must be a mapping")
    body = dict(value)
    identity = body.pop("plan_identity", None)
    expected = {
        "schema_version",
        "phase",
        "runner",
        "runner_source_sha256",
        "recipe",
        "label_rule",
        "candidate_layers",
        "two_layer_candidates",
        "fixed_best_layer",
        "intervention_site",
        "standardized_alpha",
        "token_scope",
        "max_new_tokens",
        "shard_rows",
        "max_peak_memory_bytes",
        "expected_records",
        "expected_questions",
        "schedule",
        "schedule_sha256",
        "questions_sha256",
        "prompt_sha256",
        "prompt_template_sha256",
        "controller_identity_sha256",
        "controller_datasets",
        "fit_capture_path",
        "fit_capture_artifact_sha256",
        "fit_capture_plan_identity",
        "fit_capture_chain_head",
        "e3_static_vectors_path",
        "e3_static_vectors_sha256",
        "directions",
        "runtime_identity",
        "execution_public_key",
        "scientific_eligible",
    }
    if (
        set(body) != expected
        or identity != stable_hash(body)
        or body.get("schema_version") != 1
        or body.get("phase") != "E5-native-layer-label-capture"
        or body.get("runner") != "resumable-signed-native-vllm-counterfactual-layer-labels-v1"
        or body.get("runner_source_sha256") != sha256_file(Path(__file__))
        or body.get("label_rule") != _LABEL_RULE
        or body.get("schedule_sha256") != stable_hash(body.get("schedule"))
        or body.get("expected_records") != len(body.get("schedule", []))
    ):
        raise FrozenArtifactError("E5 layer-label plan differs from exact replay")
    try:
        recipe = E5FitRecipe(
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
    except (KeyError, TypeError, ValueError, DataValidationError) as exc:
        raise FrozenArtifactError(f"invalid E5 layer-label recipe: {exc}") from exc
    if (
        body["recipe"] != recipe.to_dict()
        or body["candidate_layers"] != list(recipe.three_layer_candidates)
        or body["two_layer_candidates"] != list(recipe.two_layer_candidates)
        or body["fixed_best_layer"] != recipe.fixed_best_layer
        or body["intervention_site"] != recipe.intervention_site.value
        or body["standardized_alpha"] != float(recipe.alpha_max)
        or body["token_scope"] != TokenScope.FINAL_PROMPT.value
    ):
        raise FrozenArtifactError("E5 layer-label geometry differs from its recipe")
    return value


def _shards(directory: Path) -> tuple[Path, ...]:
    root = directory / "shards"
    if root.is_symlink() or not root.is_dir():
        raise FrozenArtifactError("E5 layer-label shard directory is invalid")
    values = sorted(root.iterdir())
    indices: list[int] = []
    for value in values:
        match = _SHARD.fullmatch(value.name)
        if match is None or value.is_symlink() or not value.is_dir():
            raise FrozenArtifactError(f"unexpected E5 layer-label shard: {value.name}")
        indices.append(int(match.group(1)))
    if indices != list(range(len(indices))):
        raise FrozenArtifactError("E5 layer-label shard numbering is not contiguous")
    return tuple(values)


@contextmanager
def _execution_lock(directory: Path) -> Iterator[None]:
    lock = directory / "run.lock"
    if lock.is_symlink() or not lock.is_file() or lock.stat().st_size != 0:
        raise FrozenArtifactError("E5 layer-label execution lock is invalid")
    with lock.open("r+b", buffering=0) as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FrozenArtifactError("E5 layer-label capture is already running") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _cleanup_stages(directory: Path) -> None:
    root = directory / "shards"
    for value in root.iterdir():
        if _SHARD_STAGE.fullmatch(value.name) is None:
            continue
        if value.is_symlink() or not value.is_dir():
            raise FrozenArtifactError("E5 layer-label stage is invalid")
        shutil.rmtree(value)


def _verify_signature(manifest: Mapping[str, Any], public_key_hex: str) -> None:
    signed = dict(manifest)
    signature = signed.pop("signature", None)
    if type(signature) is not str or re.fullmatch(r"[0-9a-f]{128}", signature) is None:
        raise FrozenArtifactError("E5 layer-label signature is invalid")
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex)).verify(
            bytes.fromhex(signature), canonical_json(signed).encode()
        )
    except (InvalidSignature, ValueError) as exc:
        raise FrozenArtifactError("E5 layer-label signature cannot be verified") from exc


def _token_digest(values: Sequence[int]) -> str:
    return hashlib.sha256(",".join(str(int(value)) for value in values).encode("ascii")).hexdigest()


def _read_records(path: Path, *, expected_sha256: str) -> list[dict[str, Any]]:
    if sha256_file(path) != expected_sha256:
        raise FrozenArtifactError("E5 layer-label record file digest differs")
    try:
        values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read E5 layer-label records: {exc}") from exc
    if any(not isinstance(value, dict) for value in values):
        raise FrozenArtifactError("E5 layer-label record must be a mapping")
    return values


def _validate_record(
    record: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    plan: Mapping[str, Any],
    question: Question,
) -> None:
    body = dict(record)
    digest = body.pop("record_digest", None)
    direction = plan["directions"][str(expected["layer"])]
    raw = body.get("raw_output")
    tokens = body.get("output_token_ids")
    if (
        set(body)
        != {
            "sequence",
            "question_index",
            "question_id",
            "semantic_group_id",
            "source_outcome",
            "question_sha256",
            "layer",
            "rendered_prompt_sha256",
            "prompt_token_ids_sha256",
            "direction_sha256",
            "reference_rms",
            "standardized_alpha",
            "raw_alpha",
            "token_scope",
            "raw_output",
            "raw_output_sha256",
            "output_token_ids",
            "output_token_ids_sha256",
            "outcome",
            "latency_seconds",
            "input_tokens",
            "output_tokens",
            "stop_type",
            "peak_memory_bytes",
            "active_memory_bytes",
            "cache_memory_bytes",
            "hook_applications",
            "activation_delta_norm",
            "pre_activation_sha256",
            "post_activation_sha256",
            "delta_sha256",
        }
        or digest != stable_hash(body)
        or any(body.get(key) != expected[key] for key in expected)
        or body.get("direction_sha256") != direction["direction_sha256"]
        or body.get("reference_rms") != direction["reference_rms"]
        or body.get("standardized_alpha") != plan["standardized_alpha"]
        or body.get("raw_alpha") != plan["standardized_alpha"] * direction["reference_rms"]
        or body.get("token_scope") != TokenScope.FINAL_PROMPT.value
        or type(raw) is not str
        or not raw.strip()
        or body.get("raw_output_sha256") != hashlib.sha256(raw.encode()).hexdigest()
        or type(tokens) is not list
        or any(type(value) is not int for value in tokens)
        or body.get("output_token_ids_sha256") != _token_digest(tokens)
        or body.get("outcome") != deterministic_short_answer_grade(raw, question.aliases).value
        or body.get("outcome") not in {"C", "I", "A"}
        or type(body.get("latency_seconds")) is not float
        or not math.isfinite(body["latency_seconds"])
        or body["latency_seconds"] < 0
        or type(body.get("input_tokens")) is not int
        or body["input_tokens"] <= 0
        or body.get("output_tokens") != len(tokens)
        or type(body.get("peak_memory_bytes")) is not int
        or not 0 <= body["peak_memory_bytes"] <= plan["max_peak_memory_bytes"]
        or any(
            type(body.get(name)) is not int or body[name] < 0
            for name in ("active_memory_bytes", "cache_memory_bytes")
        )
        or body.get("hook_applications") != 1
        or type(body.get("activation_delta_norm")) is not float
        or not math.isclose(
            body["activation_delta_norm"],
            abs(body["raw_alpha"]),
            rel_tol=1e-5,
            abs_tol=1e-6,
        )
        or any(
            _SHA256.fullmatch(str(body.get(name))) is None
            for name in (
                "pre_activation_sha256",
                "post_activation_sha256",
                "delta_sha256",
            )
        )
    ):
        raise FrozenArtifactError("E5 layer-label record differs from frozen execution")


def verify_e5_layer_label_capture(
    directory: str | Path,
    *,
    questions: Sequence[Question],
    prompt: PromptSpec,
    controller_datasets: Mapping[FeatureComposition, ProbeDataset],
    fit_capture: VerifiedE5FitCapture,
    fit_capture_artifact_sha256: str,
    expected_execution_public_key: str,
    require_complete: bool = False,
) -> VerifiedE5LayerLabels:
    """Replay the full signed label chain against its frozen sources."""

    source = Path(directory)
    plan = _load_plan(source)
    _digest(expected_execution_public_key, "E5 trusted layer-label public key")
    if plan["execution_public_key"] != expected_execution_public_key:
        raise FrozenArtifactError("E5 layer-label key differs from its external trust root")
    lock_path = source / "run.lock"
    if lock_path.is_symlink() or not lock_path.is_file() or lock_path.stat().st_size != 0:
        raise FrozenArtifactError("E5 layer-label execution lock is invalid")
    question_ids, group_ids, outcomes, receipts = _controller_identity(controller_datasets)
    schedule = _schedule_rows(
        questions,
        question_ids=question_ids,
        group_ids=group_ids,
        outcomes=outcomes,
        candidate_layers=plan["candidate_layers"],
    )
    if (
        plan["schedule"] != schedule
        or plan["questions_sha256"] != _question_digest(questions)
        or plan["prompt_sha256"] != _prompt_digest(prompt)
        or plan["controller_datasets"] != receipts
        or not fit_capture.complete
        or sha256_path(fit_capture.directory) != fit_capture_artifact_sha256
        or fit_capture_artifact_sha256 != plan["fit_capture_artifact_sha256"]
        or str(fit_capture.directory.resolve()) != plan["fit_capture_path"]
        or fit_capture.plan["plan_identity"] != plan["fit_capture_plan_identity"]
        or fit_capture.chain_head != plan["fit_capture_chain_head"]
        or sha256_path(plan["e3_static_vectors_path"]) != plan["e3_static_vectors_sha256"]
    ):
        raise FrozenArtifactError("E5 layer-label source changed after preparation")
    recipe = E5FitRecipe(
        fixed_best_layer=plan["recipe"]["fixed_best_layer"],
        two_layer_candidates=tuple(plan["recipe"]["two_layer_candidates"]),
        three_layer_candidates=tuple(plan["recipe"]["three_layer_candidates"]),
        intervention_site=ActivationSite(plan["recipe"]["intervention_site"]),
        minimum_class_count=plan["recipe"]["minimum_class_count"],
        vector_seed=plan["recipe"]["vector_seed"],
        router_seed=plan["recipe"]["router_seed"],
        router_hidden_width=plan["recipe"]["router_hidden_width"],
        router_epochs=plan["recipe"]["router_epochs"],
        distance_temperature=plan["recipe"]["distance_temperature"],
        layer_seed=plan["recipe"]["layer_seed"],
        layer_epochs=plan["recipe"]["layer_epochs"],
        alpha_max=plan["recipe"]["alpha_max"],
        alpha_beta=plan["recipe"]["alpha_beta"],
        alpha_threshold=plan["recipe"]["alpha_threshold"],
        schema_version=plan["recipe"]["schema_version"],
    )
    _live, direction_receipts = _direction_receipts(Path(plan["e3_static_vectors_path"]), recipe)
    if plan["directions"] != direction_receipts:
        raise FrozenArtifactError("E5 layer-label directions changed")
    completed = 0
    previous: str | None = None
    maximum_peak = 0
    shards = _shards(source)
    for index, shard in enumerate(shards):
        if {value.name for value in shard.iterdir()} != {
            "manifest.json",
            "records.jsonl",
        } or any(value.is_symlink() or not value.is_file() for value in shard.iterdir()):
            raise FrozenArtifactError("E5 layer-label shard inventory differs")
        try:
            manifest = json.loads((shard / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FrozenArtifactError(f"cannot load E5 layer-label manifest: {exc}") from exc
        if not isinstance(manifest, dict):
            raise FrozenArtifactError("E5 layer-label manifest must be a mapping")
        _verify_signature(manifest, plan["execution_public_key"])
        signed = dict(manifest)
        signed.pop("signature")
        manifest_digest = signed.pop("manifest_digest", None)
        expected_keys = {
            "schema_version",
            "plan_identity",
            "shard_index",
            "start_record",
            "end_record",
            "record_count",
            "records_sha256",
            "record_digests",
            "previous_manifest_digest",
            "maximum_peak_memory_bytes",
            "execution_session_id",
            "execution_session_started_at",
            "execution_session_wall_seconds",
            "execution_lock_identity",
            "execution_process_id",
        }
        records = _read_records(
            shard / "records.jsonl", expected_sha256=str(signed.get("records_sha256"))
        )
        if (
            set(signed) != expected_keys
            or manifest_digest != stable_hash(signed)
            or signed["schema_version"] != 1
            or signed["plan_identity"] != plan["plan_identity"]
            or signed["shard_index"] != index
            or signed["start_record"] != completed
            or signed["record_count"] != len(records)
            or signed["record_count"] <= 0
            or signed["end_record"] != completed + len(records)
            or signed["previous_manifest_digest"] != previous
            or signed["record_digests"] != [value.get("record_digest") for value in records]
            or type(signed["maximum_peak_memory_bytes"]) is not int
            or not 0 <= signed["maximum_peak_memory_bytes"] <= plan["max_peak_memory_bytes"]
            or _SHA256.fullmatch(str(signed["execution_session_id"])) is None
            or type(signed["execution_session_started_at"]) is not str
            or type(signed["execution_session_wall_seconds"]) is not float
            or not math.isfinite(signed["execution_session_wall_seconds"])
            or signed["execution_session_wall_seconds"] < 0
            or _SHA256.fullmatch(str(signed["execution_lock_identity"])) is None
            or type(signed["execution_process_id"]) is not int
            or signed["execution_process_id"] <= 0
        ):
            raise FrozenArtifactError("E5 layer-label manifest differs")
        for offset, record in enumerate(records):
            expected = schedule[completed + offset]
            _validate_record(
                record,
                expected=expected,
                plan=plan,
                question=questions[expected["question_index"]],
            )
        completed = signed["end_record"]
        maximum_peak = max(maximum_peak, signed["maximum_peak_memory_bytes"])
        previous = str(manifest_digest)
    complete = completed == plan["expected_records"]
    if completed > plan["expected_records"] or (require_complete and not complete):
        raise FrozenArtifactError("E5 layer-label record count differs")
    return VerifiedE5LayerLabels(
        directory=source.resolve(),
        plan=MappingProxyType(plan),
        records_completed=completed,
        shard_count=len(shards),
        chain_head=previous,
        complete=complete,
        scientific_eligible=bool(plan["scientific_eligible"] and complete),
        maximum_peak_memory_bytes=maximum_peak,
    )


def _execution_record(
    *,
    runtime: E5LayerLabelRuntime,
    prompt: PromptSpec,
    question: Question,
    expected: Mapping[str, Any],
    direction: ResolvedStaticDirection,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    rendered = runtime.render_prompt(prompt, question.text, metadata=question.metadata)
    state = runtime.standardized_intervention_state(
        np.ascontiguousarray(direction.direction.numpy(), dtype=np.float32),
        standardized_alpha=plan["standardized_alpha"],
        reference_rms=direction.reference_rms,
        token_scope=TokenScope.FINAL_PROMPT,
    )
    generated = runtime.generate_with_interventions(
        rendered,
        max_new_tokens=plan["max_new_tokens"],
        intervention_states={(expected["layer"], ActivationSite(plan["intervention_site"])): state},
    )
    if type(generated) is not VllmGenerationOutput or generated.rendered_prompt != rendered:
        raise FrozenArtifactError("E5 layer-label runtime returned invalid generation")
    try:
        pre = np.ascontiguousarray(as_numpy(state.captured, dtype=np.float32))
        post = np.ascontiguousarray(as_numpy(state.intervened, dtype=np.float32))
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"E5 layer-label hook evidence is invalid: {exc}") from exc
    delta = np.ascontiguousarray(post - pre, dtype=np.float32)
    raw_alpha = plan["standardized_alpha"] * direction.reference_rms
    delta_norm = float(np.linalg.norm(delta.astype(np.float64)))
    if (
        pre.shape != post.shape
        or pre.size == 0
        or not np.isfinite(pre).all()
        or not np.isfinite(post).all()
        or getattr(state, "applications", None) != 1
        or not math.isclose(delta_norm, abs(raw_alpha), rel_tol=1e-5, abs_tol=1e-6)
        or generated.peak_memory_bytes > plan["max_peak_memory_bytes"]
    ):
        raise DataValidationError("E5 layer-label intervention evidence differs")
    outcome = deterministic_short_answer_grade(generated.text, question.aliases)
    if outcome is Outcome.UNSCORABLE:
        raise DataValidationError("E5 layer-label generation is unscorable")
    body = {
        **dict(expected),
        "rendered_prompt_sha256": rendered.sha256,
        "prompt_token_ids_sha256": rendered.token_ids_sha256,
        "direction_sha256": direction.direction_sha256,
        "reference_rms": direction.reference_rms,
        "standardized_alpha": plan["standardized_alpha"],
        "raw_alpha": raw_alpha,
        "token_scope": TokenScope.FINAL_PROMPT.value,
        "raw_output": generated.text,
        "raw_output_sha256": hashlib.sha256(generated.text.encode()).hexdigest(),
        "output_token_ids": list(generated.token_ids),
        "output_token_ids_sha256": _token_digest(generated.token_ids),
        "outcome": outcome.value,
        "latency_seconds": float(generated.latency_seconds),
        "input_tokens": generated.input_tokens,
        "output_tokens": generated.output_tokens,
        "stop_type": generated.stop_type,
        "peak_memory_bytes": generated.peak_memory_bytes,
        "active_memory_bytes": generated.active_memory_bytes,
        "cache_memory_bytes": generated.cache_memory_bytes,
        "hook_applications": state.applications,
        "activation_delta_norm": delta_norm,
        "pre_activation_sha256": hashlib.sha256(pre.tobytes(order="C")).hexdigest(),
        "post_activation_sha256": hashlib.sha256(post.tobytes(order="C")).hexdigest(),
        "delta_sha256": hashlib.sha256(delta.tobytes(order="C")).hexdigest(),
    }
    return {**body, "record_digest": stable_hash(body)}


def _append_shard(
    directory: Path,
    *,
    verified: VerifiedE5LayerLabels,
    records: Sequence[Mapping[str, Any]],
    private_key_hex: str,
    session: Mapping[str, Any],
) -> VerifiedE5LayerLabels:
    index = verified.shard_count
    destination = directory / "shards" / f"shard-{index:05d}"
    if destination.exists() or destination.is_symlink() or not records:
        raise FrozenArtifactError("refusing invalid E5 layer-label shard append")
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=directory / "shards"))
    try:
        records_path = stage / "records.jsonl"
        records_path.write_text(
            "".join(json.dumps(dict(value), sort_keys=True) + "\n" for value in records),
            encoding="utf-8",
        )
        maximum_peak = max(int(value["peak_memory_bytes"]) for value in records)
        body = {
            "schema_version": 1,
            "plan_identity": verified.plan["plan_identity"],
            "shard_index": index,
            "start_record": verified.records_completed,
            "end_record": verified.records_completed + len(records),
            "record_count": len(records),
            "records_sha256": sha256_file(records_path),
            "record_digests": [value["record_digest"] for value in records],
            "previous_manifest_digest": verified.chain_head,
            "maximum_peak_memory_bytes": maximum_peak,
            **dict(session),
        }
        signed = {**body, "manifest_digest": stable_hash(body)}
        private = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_key_hex))
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
    completed = verified.records_completed + len(records)
    complete = completed == verified.plan["expected_records"]
    return VerifiedE5LayerLabels(
        directory=verified.directory,
        plan=verified.plan,
        records_completed=completed,
        shard_count=verified.shard_count + 1,
        chain_head=str(signed["manifest_digest"]),
        complete=complete,
        scientific_eligible=bool(verified.plan["scientific_eligible"] and complete),
        maximum_peak_memory_bytes=max(verified.maximum_peak_memory_bytes, maximum_peak),
    )


def run_e5_layer_label_capture(
    directory: str | Path,
    *,
    questions: Sequence[Question],
    prompt: PromptSpec,
    controller_datasets: Mapping[FeatureComposition, ProbeDataset],
    fit_capture: VerifiedE5FitCapture,
    fit_capture_artifact_sha256: str,
    runtime: E5LayerLabelRuntime,
    private_key_hex: str,
    request_budget: int | None = None,
) -> VerifiedE5LayerLabels:
    """Run or resume a bounded number of candidate-layer generations."""

    if request_budget is not None and (type(request_budget) is not int or request_budget <= 0):
        raise DataValidationError("E5 layer-label request budget must be positive")
    source = Path(directory)
    trusted_key = _public_key(private_key_hex)
    started_at = datetime.now(UTC).isoformat(timespec="microseconds")
    started = time.perf_counter()
    process_id = os.getpid()
    with _execution_lock(source):
        _cleanup_stages(source)
        verified = verify_e5_layer_label_capture(
            source,
            questions=questions,
            prompt=prompt,
            controller_datasets=controller_datasets,
            fit_capture=fit_capture,
            fit_capture_artifact_sha256=fit_capture_artifact_sha256,
            expected_execution_public_key=trusted_key,
        )
        if _exact_json(runtime.runtime_identity(), label="E5 live layer-label runtime") != dict(
            verified.plan["runtime_identity"]
        ):
            raise FrozenArtifactError("E5 live layer-label runtime differs from plan")
        if verified.complete:
            return verified
        recipe = E5FitRecipe(
            fixed_best_layer=verified.plan["recipe"]["fixed_best_layer"],
            two_layer_candidates=tuple(verified.plan["recipe"]["two_layer_candidates"]),
            three_layer_candidates=tuple(verified.plan["recipe"]["three_layer_candidates"]),
            intervention_site=ActivationSite(verified.plan["recipe"]["intervention_site"]),
            minimum_class_count=verified.plan["recipe"]["minimum_class_count"],
            vector_seed=verified.plan["recipe"]["vector_seed"],
            router_seed=verified.plan["recipe"]["router_seed"],
            router_hidden_width=verified.plan["recipe"]["router_hidden_width"],
            router_epochs=verified.plan["recipe"]["router_epochs"],
            distance_temperature=verified.plan["recipe"]["distance_temperature"],
            layer_seed=verified.plan["recipe"]["layer_seed"],
            layer_epochs=verified.plan["recipe"]["layer_epochs"],
            alpha_max=verified.plan["recipe"]["alpha_max"],
            alpha_beta=verified.plan["recipe"]["alpha_beta"],
            alpha_threshold=verified.plan["recipe"]["alpha_threshold"],
            schema_version=verified.plan["recipe"]["schema_version"],
        )
        directions, receipts = _direction_receipts(
            Path(verified.plan["e3_static_vectors_path"]), recipe
        )
        if receipts != verified.plan["directions"]:
            raise FrozenArtifactError("E5 live layer-label directions differ")
        remaining = verified.plan["expected_records"] - verified.records_completed
        budget = remaining if request_budget is None else min(remaining, request_budget)
        target = verified.records_completed + budget
        session_id = stable_hash(
            {
                "plan_identity": verified.plan["plan_identity"],
                "started_at": started_at,
                "process_id": process_id,
                "nonce": time.time_ns(),
            }
        )
        lock_identity = stable_hash(
            {
                "plan_identity": verified.plan["plan_identity"],
                "execution_session_id": session_id,
                "lock_path": "run.lock",
                "process_id": process_id,
            }
        )
        while verified.records_completed < target:
            count = min(target - verified.records_completed, verified.plan["shard_rows"])
            expected_rows = verified.plan["schedule"][
                verified.records_completed : verified.records_completed + count
            ]
            records = [
                _execution_record(
                    runtime=runtime,
                    prompt=prompt,
                    question=questions[value["question_index"]],
                    expected=value,
                    direction=directions[value["layer"]],
                    plan=verified.plan,
                )
                for value in expected_rows
            ]
            verified = _append_shard(
                source,
                verified=verified,
                records=records,
                private_key_hex=private_key_hex,
                session={
                    "execution_session_id": session_id,
                    "execution_session_started_at": started_at,
                    "execution_session_wall_seconds": float(time.perf_counter() - started),
                    "execution_lock_identity": lock_identity,
                    "execution_process_id": process_id,
                },
            )
        return verified


def _label_order(recipe: E5FitRecipe, candidates: Sequence[int]) -> tuple[int, ...]:
    return (
        recipe.fixed_best_layer,
        *(value for value in candidates if value != recipe.fixed_best_layer),
    )


def load_e5_layer_label_data(
    directory: str | Path,
    *,
    questions: Sequence[Question],
    prompt: PromptSpec,
    controller_datasets: Mapping[FeatureComposition, ProbeDataset],
    fit_capture: VerifiedE5FitCapture,
    fit_capture_artifact_sha256: str,
    expected_execution_public_key: str,
) -> E5LayerLabelData:
    """Verify and reduce counterfactual outcomes to two/three-layer labels."""

    verified = verify_e5_layer_label_capture(
        directory,
        questions=questions,
        prompt=prompt,
        controller_datasets=controller_datasets,
        fit_capture=fit_capture,
        fit_capture_artifact_sha256=fit_capture_artifact_sha256,
        expected_execution_public_key=expected_execution_public_key,
        require_complete=True,
    )
    all_records: list[dict[str, Any]] = []
    for shard in _shards(Path(directory)):
        manifest = json.loads((shard / "manifest.json").read_text(encoding="utf-8"))
        all_records.extend(
            _read_records(shard / "records.jsonl", expected_sha256=manifest["records_sha256"])
        )
    recipe = E5FitRecipe(
        fixed_best_layer=verified.plan["recipe"]["fixed_best_layer"],
        two_layer_candidates=tuple(verified.plan["recipe"]["two_layer_candidates"]),
        three_layer_candidates=tuple(verified.plan["recipe"]["three_layer_candidates"]),
        intervention_site=ActivationSite(verified.plan["recipe"]["intervention_site"]),
        minimum_class_count=verified.plan["recipe"]["minimum_class_count"],
        vector_seed=verified.plan["recipe"]["vector_seed"],
        router_seed=verified.plan["recipe"]["router_seed"],
        router_hidden_width=verified.plan["recipe"]["router_hidden_width"],
        router_epochs=verified.plan["recipe"]["router_epochs"],
        distance_temperature=verified.plan["recipe"]["distance_temperature"],
        layer_seed=verified.plan["recipe"]["layer_seed"],
        layer_epochs=verified.plan["recipe"]["layer_epochs"],
        alpha_max=verified.plan["recipe"]["alpha_max"],
        alpha_beta=verified.plan["recipe"]["alpha_beta"],
        alpha_threshold=verified.plan["recipe"]["alpha_threshold"],
        schema_version=verified.plan["recipe"]["schema_version"],
    )
    by_question: list[dict[int, Outcome]] = [dict() for _ in questions]
    for record in all_records:
        by_question[record["question_index"]][record["layer"]] = Outcome(record["outcome"])
    rank = {Outcome.CORRECT: 0, Outcome.ABSTENTION: 1, Outcome.INCORRECT: 2}

    def best(values: Mapping[int, Outcome], candidates: Sequence[int]) -> int:
        order = _label_order(recipe, candidates)
        if set(values) != set(recipe.three_layer_candidates):
            raise FrozenArtifactError("E5 layer-label candidate coverage differs")
        return min(order, key=lambda layer: (rank[values[layer]], order.index(layer)))

    question_ids, group_ids, outcomes, _receipts = _controller_identity(controller_datasets)
    data = E5LayerLabelData(
        verified=verified,
        question_ids=question_ids,
        group_ids=group_ids,
        outcomes=outcomes,
        best_layers_two=tuple(best(value, recipe.two_layer_candidates) for value in by_question),
        best_layers_three=tuple(
            best(value, recipe.three_layer_candidates) for value in by_question
        ),
        artifact_sha256=sha256_path(directory),
        _verification_token=_VERIFIED_LAYER_LABEL_DATA,
    )
    _VERIFIED_LAYER_LABEL_RECEIPTS[id(data)] = MappingProxyType(
        {
            "object": data,
            "binding_objects": (
                id(data.verified),
                id(data.verified.plan),
                id(data.question_ids),
                id(data.group_ids),
                id(data.outcomes),
                id(data.best_layers_two),
                id(data.best_layers_three),
            ),
            "binding_digest": data._binding_digest(),
        }
    )
    return data

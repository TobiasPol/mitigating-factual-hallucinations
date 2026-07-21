"""Signed, resumable native-MLX execution for the complete E5 ablation grid."""

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
from dataclasses import dataclass, replace
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

from mfh.artifact_namespace import validate_active_study_artifact_paths
from mfh.contracts import ActivationSite, Outcome, PromptSpec, Question, TokenScope
from mfh.errors import ConfigurationError, DataValidationError, FrozenArtifactError
from mfh.evaluation.grading import deterministic_short_answer_grade
from mfh.experiments.e4_baselines import (
    E4MethodPolicy,
    E4ScreenReceipt,
    load_e4_method_policy,
    load_e4_screen_receipt,
)
from mfh.experiments.e5_adaptive import (
    E5AblationRecord,
    E5AblationSpec,
    E5ControllerBinding,
    E5Protocol,
    _e5_prompt_input_sha256,
    build_e5_ablation_grid,
    e5_ablation_execution_receipt_body,
    load_e5_controller_binding,
    sign_e5_ablation_execution_receipt,
    write_e5_ablation_records,
)
from mfh.experiments.e5_fit import (
    VerifiedE5ControllerBindings,
    open_e5_controller_bindings_checkpoint,
    verify_e5_controller_bindings,
    verify_e5_fitted_grid,
)
from mfh.experiments.e5_types import E5FitRecipe
from mfh.experiments.e8_protected import _compose_e8_controller_features
from mfh.experiments.model_selection import ACTIVE_MODEL_IDENTITIES, ACTIVE_MODEL_NAME
from mfh.experiments.static_direction_sources import (
    ResolvedStaticDirection,
    resolve_static_direction,
)
from mfh.inference.mlx_research import MlxPromptFeatureCubeOutput
from mfh.inference.mlx_runtime import MlxGenerationOutput, MlxRenderedPrompt
from mfh.methods.adaptive import AdaptiveController
from mfh.provenance import canonical_json, sha256_file, sha256_path, stable_hash

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SHARD = re.compile(r"^shard-(\d{6})$")
_SHARD_STAGE = re.compile(r"^\.shard-\d{6}\.stage-[A-Za-z0-9._-]+$")
_INVENTORY = frozenset({"plan.json", "run.lock", "shards"})
_PROMPTS = ("P0-neutral", "P2-calibrated-abstention")
_PROMPT_HASHES = MappingProxyType(
    {
        "P0-neutral": "5b08080e81d4032d853d3a30fda35e7670da6a81cc1ccf17f2b8fc5001bba684",
        "P2-calibrated-abstention": (
            "3170134d9a69836c1b530d1b16585ef7b0d92ea6fadc8f958e2655053e273fe5"
        ),
    }
)
_TOKEN_SCOPES = MappingProxyType(
    {
        "final_prompt": TokenScope.FINAL_PROMPT,
        "first_generated": TokenScope.FIRST_GENERATED,
        "first_four_generated": TokenScope.FIRST_FOUR,
    }
)
_MAX_MEMORY_BYTES = 48 * 1024**3
_MAX_NEW_TOKENS = 48
_SCHEDULE_RULE = "arm-major-m1-then-grid__prompt-p0-p2__screen-dev-order-v1"
_ACTIVE_MODEL = ACTIVE_MODEL_IDENTITIES[ACTIVE_MODEL_NAME]


class E5NativeRuntime(Protocol):
    """Minimal native MLX surface required by the E5 developmental ablation."""

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
        rendered: MlxRenderedPrompt,
        *,
        max_new_tokens: int,
        intervention_states: Mapping[tuple[int, ActivationSite], Any],
    ) -> MlxGenerationOutput: ...


@dataclass(frozen=True, slots=True)
class _E5NativeSourceContext:
    screen: E4ScreenReceipt
    bindings: VerifiedE5ControllerBindings
    m1_policy: E4MethodPolicy
    m1_direction: ResolvedStaticDirection
    protocol: E5Protocol
    specs: tuple[E5AblationSpec, ...]


@dataclass(frozen=True, slots=True)
class VerifiedE5NativeRun:
    directory: Path
    plan: Mapping[str, Any]
    records_completed: int
    shard_count: int
    chain_head: str | None
    complete: bool
    scientific_eligible: bool
    maximum_peak_memory_bytes: int
    finalized_records: Path | None
    _source_context: _E5NativeSourceContext


@dataclass(frozen=True, slots=True)
class E5NativePromotionRow:
    """One semantically replayed M1 or selected-M3 row eligible for E5 promotion."""

    sequence: int
    record: E5AblationRecord
    evidence: Mapping[str, Any]
    row_digest: str


def _exact_json(value: Mapping[str, Any], *, context: str) -> dict[str, Any]:
    try:
        result = json.loads(
            json.dumps(dict(value), sort_keys=True, allow_nan=False, separators=(",", ":"))
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"{context} is not exact JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise DataValidationError(f"{context} must be a mapping")
    return result


def _digest(value: object, context: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise DataValidationError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _utc_timestamp(value: object) -> bool:
    if type(value) is not str:
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == UTC.utcoffset(parsed)


def _private_key(value: str) -> Ed25519PrivateKey:
    _digest(value, "E5 native execution private key")
    try:
        return Ed25519PrivateKey.from_private_bytes(bytes.fromhex(value))
    except ValueError as exc:
        raise DataValidationError(f"invalid E5 native execution private key: {exc}") from exc


def e5_native_execution_public_key(private_key_hex: str) -> str:
    """Derive the external public trust anchor used by every E5 shard."""

    return (
        _private_key(private_key_hex)
        .public_key()
        .public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        .hex()
    )


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


def _prompt_body(prompt: PromptSpec) -> dict[str, Any]:
    return {
        "prompt_id": prompt.prompt_id,
        "text": prompt.text,
        "permits_abstention": prompt.permits_abstention,
        "deployment_eligible": prompt.deployment_eligible,
        "template_sha256": hashlib.sha256(prompt.text.encode()).hexdigest(),
    }


def _runtime_is_scientific(identity: Mapping[str, Any]) -> bool:
    provenance = identity.get("research_provenance")
    return bool(
        identity.get("model_repository") == _ACTIVE_MODEL["repository"]
        and identity.get("model_revision") == _ACTIVE_MODEL["revision"]
        and identity.get("model_quantization") == _ACTIVE_MODEL["quantization"]
        and identity.get("model_num_layers", identity.get("num_layers"))
        == _ACTIVE_MODEL["num_layers"]
        and isinstance(provenance, Mapping)
        and _SHA256.fullmatch(str(provenance.get("runtime_preflight_receipt_sha256"))) is not None
    )


def _schedule_size(specs: Sequence[E5AblationSpec], question_count: int) -> int:
    return (1 + len(specs)) * len(_PROMPTS) * question_count


def _schedule_row(
    sequence: int,
    *,
    specs: Sequence[E5AblationSpec],
    questions: Sequence[Question],
) -> tuple[str, E5AblationSpec | None, str, Question]:
    question_count = len(questions)
    total = _schedule_size(specs, question_count)
    if type(sequence) is not int or not 0 <= sequence < total or question_count <= 0:
        raise DataValidationError("E5 native schedule sequence is invalid")
    block = len(_PROMPTS) * question_count
    arm_index, within_arm = divmod(sequence, block)
    prompt_index, question_index = divmod(within_arm, question_count)
    spec = None if arm_index == 0 else specs[arm_index - 1]
    return (
        "M1" if spec is None else spec.spec_id,
        spec,
        _PROMPTS[prompt_index],
        questions[question_index],
    )


def _static_source(
    *, policy_path: Path, vectors_path: Path, execution_public_key: str
) -> tuple[E4MethodPolicy, ResolvedStaticDirection]:
    policy = load_e4_method_policy(policy_path)
    if (
        policy.method != "M1"
        or policy.layer is None
        or policy.site is None
        or policy.token_scope is None
        or policy.direction_norm is None
        or policy.reference_rms is None
        or policy.execution_public_key != execution_public_key
        or policy.implementation_artifact_sha256 != sha256_path(vectors_path)
    ):
        raise FrozenArtifactError("E5 M1 reference differs from its frozen E4 policy")
    direction = resolve_static_direction(
        vectors_path,
        method="M1",
        layer=policy.layer,
        site=policy.site,
    )
    if (
        policy.direction_sha256 != direction.direction_sha256
        or not math.isclose(
            float(policy.direction_norm), direction.direction_norm, rel_tol=0, abs_tol=1e-7
        )
        or not math.isclose(policy.reference_rms, direction.reference_rms, rel_tol=0, abs_tol=1e-12)
    ):
        raise FrozenArtifactError("E5 M1 policy geometry differs from its vector")
    return policy, direction


def _source_paths(plan: Mapping[str, Any]) -> dict[str, Path]:
    return {
        name: Path(plan[f"{name}_path"])
        for name in (
            "screen_receipt",
            "controller_bindings",
            "fit_capture",
            "m1_policy",
            "e3_static_vectors",
            "runtime_artifact",
        )
    }


def prepare_e5_native_ablation(
    directory: str | Path,
    *,
    screen_receipt_path: str | Path,
    controller_bindings_directory: str | Path,
    fit_capture_directory: str | Path,
    m1_policy_path: str | Path,
    e3_static_vectors_directory: str | Path,
    runtime_artifact: str | Path,
    prompts: Mapping[str, PromptSpec],
    execution_public_key: str,
    protocol: E5Protocol | None = None,
    shard_rows: int = 1_024,
    max_new_tokens: int = _MAX_NEW_TOKENS,
    max_peak_memory_bytes: int = _MAX_MEMORY_BYTES,
) -> Mapping[str, Any]:
    """Freeze the implicit 9.73M-row schedule without loading the model."""

    frozen = E5Protocol() if protocol is None else protocol
    if type(frozen) is not E5Protocol:
        raise DataValidationError("E5 native preparation requires an exact protocol")
    if (
        type(shard_rows) is not int
        or shard_rows <= 0
        or type(max_new_tokens) is not int
        or not 1 <= max_new_tokens <= _MAX_NEW_TOKENS
        or type(max_peak_memory_bytes) is not int
        or not 0 < max_peak_memory_bytes <= _MAX_MEMORY_BYTES
    ):
        raise ConfigurationError("E5 native shard, token, or memory limit is invalid")
    _digest(execution_public_key, "E5 native execution public key")
    normalized = validate_active_study_artifact_paths(
        {
            "E5 native ablation": directory,
            "E5 screen receipt": screen_receipt_path,
            "E5 controller bindings": controller_bindings_directory,
            "E5 fit capture": fit_capture_directory,
            "E5 M1 policy": m1_policy_path,
            "E5 E3 static vectors": e3_static_vectors_directory,
            "E5 runtime artifact": runtime_artifact,
        }
    )
    destination = normalized["E5 native ablation"]
    if destination.exists() or destination.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite E5 native run: {destination}")
    screen_path = normalized["E5 screen receipt"]
    bindings_path = normalized["E5 controller bindings"]
    fit_capture_path = normalized["E5 fit capture"]
    policy_path = normalized["E5 M1 policy"]
    vectors_path = normalized["E5 E3 static vectors"]
    runtime_path = normalized["E5 runtime artifact"]
    screen = load_e4_screen_receipt(screen_path)
    bindings = verify_e5_controller_bindings(bindings_path)
    if E5Protocol.from_dict(bindings.manifest["protocol"]) != frozen:
        raise FrozenArtifactError("E5 binding package protocol differs from native schedule")
    policy, direction = _static_source(
        policy_path=policy_path,
        vectors_path=vectors_path,
        execution_public_key=execution_public_key,
    )
    if (
        set(prompts) != set(_PROMPTS)
        or any(
            type(prompts[name]) is not PromptSpec
            or prompts[name].prompt_id != name
            or hashlib.sha256(prompts[name].text.encode()).hexdigest() != _PROMPT_HASHES[name]
            for name in _PROMPTS
        )
        or not screen.dev_questions
        or any(
            question.benchmark != "triviaqa" or question.split != "T-dev"
            for question in screen.dev_questions
        )
    ):
        raise DataValidationError("E5 prompts or T-dev questions differ from the schedule")
    grid = verify_e5_fitted_grid(bindings.manifest["fitted_grid_path"])
    recipe = E5FitRecipe.from_dict(dict(grid.manifest["recipe"]))
    if recipe.fixed_best_layer != policy.layer or recipe.intervention_site is not policy.site:
        raise FrozenArtifactError(
            "E5 fitted controller layer/site geometry differs from the frozen M1 policy"
        )
    try:
        capture_plan = json.loads((fit_capture_path / "plan.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise FrozenArtifactError(f"cannot read E5 fit-capture plan: {exc}") from exc
    fit_provenance = grid.manifest["fit_provenance"]
    if (
        not isinstance(capture_plan, Mapping)
        or sha256_path(fit_capture_path) != fit_provenance["capture_artifact_sha256"]
        or capture_plan.get("plan_identity") != fit_provenance["capture_plan_identity"]
        or capture_plan.get("execution_public_key") != execution_public_key
        or capture_plan.get("runtime_artifact_sha256") != sha256_path(runtime_path)
        or capture_plan.get("protocol") != frozen.to_dict()
        or capture_plan.get("recipe") != recipe.to_dict()
    ):
        raise FrozenArtifactError("E5 native fit capture differs from fitted-controller provenance")
    identity_value = capture_plan.get("runtime_identity")
    if not isinstance(identity_value, Mapping):
        raise FrozenArtifactError("E5 fit capture lacks its runtime identity")
    identity = _exact_json(identity_value, context="E5 native runtime identity")
    prompt_rows = {name: _prompt_body(prompts[name]) for name in _PROMPTS}
    sources = {
        "screen_receipt": screen_path,
        "controller_bindings": bindings_path,
        "fit_capture": fit_capture_path,
        "m1_policy": policy_path,
        "e3_static_vectors": vectors_path,
        "runtime_artifact": runtime_path,
    }
    source_digests = {name: sha256_path(path) for name, path in sources.items()}
    specs = build_e5_ablation_grid(frozen)
    expected_records = _schedule_size(specs, len(screen.dev_questions))
    scientific = bool(
        frozen.scientific_eligible
        and screen.scientific_eligible
        and len(screen.dev_questions) == 5_000
        and bindings.scientific_eligible
        and _runtime_is_scientific(identity)
        and max_new_tokens == _MAX_NEW_TOKENS
        and max_peak_memory_bytes == _MAX_MEMORY_BYTES
    )
    body = {
        "schema_version": 1,
        "phase": "E5-native-ablation",
        "runner_source_sha256": sha256_file(Path(__file__)),
        "schedule_rule": _SCHEDULE_RULE,
        "protocol": frozen.to_dict(),
        "protocol_sha256": stable_hash(frozen.to_dict()),
        "question_count": len(screen.dev_questions),
        "question_ids_sha256": stable_hash([value.question_id for value in screen.dev_questions]),
        "questions_sha256": stable_hash([_question_body(value) for value in screen.dev_questions]),
        "prompts": prompt_rows,
        "prompts_sha256": stable_hash(prompt_rows),
        "expected_records": expected_records,
        "shard_rows": shard_rows,
        "max_new_tokens": max_new_tokens,
        "max_peak_memory_bytes": max_peak_memory_bytes,
        "execution_public_key": execution_public_key,
        "runtime_identity": identity,
        "runtime_identity_sha256": stable_hash(identity),
        **{f"{name}_path": str(path.resolve()) for name, path in sources.items()},
        **{f"{name}_sha256": digest for name, digest in source_digests.items()},
        "screen_receipt_digest": screen.receipt_digest,
        "controller_bindings_manifest_digest": bindings.manifest["manifest_digest"],
        "m1_policy_digest": policy.policy_digest,
        "m1_direction_sha256": direction.direction_sha256,
        "scientific_eligible": scientific,
    }
    plan = {**body, "plan_identity": stable_hash(body)}
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


_PLAN_KEYS = frozenset(
    {
        "schema_version",
        "phase",
        "runner_source_sha256",
        "schedule_rule",
        "protocol",
        "protocol_sha256",
        "question_count",
        "question_ids_sha256",
        "questions_sha256",
        "prompts",
        "prompts_sha256",
        "expected_records",
        "shard_rows",
        "max_new_tokens",
        "max_peak_memory_bytes",
        "execution_public_key",
        "runtime_identity",
        "runtime_identity_sha256",
        "screen_receipt_path",
        "screen_receipt_sha256",
        "controller_bindings_path",
        "controller_bindings_sha256",
        "fit_capture_path",
        "fit_capture_sha256",
        "m1_policy_path",
        "m1_policy_sha256",
        "e3_static_vectors_path",
        "e3_static_vectors_sha256",
        "runtime_artifact_path",
        "runtime_artifact_sha256",
        "screen_receipt_digest",
        "controller_bindings_manifest_digest",
        "m1_policy_digest",
        "m1_direction_sha256",
        "scientific_eligible",
        "plan_identity",
    }
)


def _read_plan(directory: Path) -> dict[str, Any]:
    try:
        value = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read E5 native plan: {exc}") from exc
    if not isinstance(value, dict):
        raise FrozenArtifactError("E5 native plan must be a mapping")
    expected_text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if (directory / "plan.json").read_text(encoding="utf-8") != expected_text:
        raise FrozenArtifactError("E5 native plan is not canonical JSON")
    body = dict(value)
    identity = body.pop("plan_identity", None)
    if (
        set(value) != _PLAN_KEYS
        or identity != stable_hash(body)
        or value["schema_version"] != 1
        or value["phase"] != "E5-native-ablation"
        or value["runner_source_sha256"] != sha256_file(Path(__file__))
        or value["schedule_rule"] != _SCHEDULE_RULE
        or value["protocol_sha256"] != stable_hash(value["protocol"])
        or value["runtime_identity_sha256"] != stable_hash(value["runtime_identity"])
        or value["prompts_sha256"] != stable_hash(value["prompts"])
        or type(value["scientific_eligible"]) is not bool
    ):
        raise FrozenArtifactError("E5 native plan identity or schema differs")
    _digest(value["execution_public_key"], "E5 native plan execution key")
    return value


def _prompt_from_plan(plan: Mapping[str, Any], prompt_id: str) -> PromptSpec:
    try:
        row = plan["prompts"][prompt_id]
        prompt = PromptSpec(
            prompt_id=row["prompt_id"],
            text=row["text"],
            permits_abstention=row["permits_abstention"],
            deployment_eligible=row["deployment_eligible"],
        )
    except (KeyError, TypeError, DataValidationError) as exc:
        raise FrozenArtifactError(f"cannot reconstruct E5 prompt: {exc}") from exc
    if (
        set(row)
        != {
            "prompt_id",
            "text",
            "permits_abstention",
            "deployment_eligible",
            "template_sha256",
        }
        or prompt.prompt_id != prompt_id
        or row["template_sha256"] != hashlib.sha256(prompt.text.encode()).hexdigest()
        or row["template_sha256"] != _PROMPT_HASHES[prompt_id]
    ):
        raise FrozenArtifactError("E5 frozen prompt differs")
    return prompt


def _verify_sources(
    plan: Mapping[str, Any],
) -> tuple[
    E4ScreenReceipt,
    VerifiedE5ControllerBindings,
    E4MethodPolicy,
    ResolvedStaticDirection,
]:
    paths = _source_paths(plan)
    for name, path in paths.items():
        if sha256_path(path) != plan[f"{name}_sha256"]:
            raise FrozenArtifactError(f"E5 native source changed: {name}")
    screen = load_e4_screen_receipt(paths["screen_receipt"])
    bindings = verify_e5_controller_bindings(paths["controller_bindings"])
    grid = verify_e5_fitted_grid(bindings.manifest["fitted_grid_path"])
    try:
        capture_plan = json.loads((paths["fit_capture"] / "plan.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise FrozenArtifactError(f"cannot replay E5 fit-capture plan: {exc}") from exc
    policy, direction = _static_source(
        policy_path=paths["m1_policy"],
        vectors_path=paths["e3_static_vectors"],
        execution_public_key=plan["execution_public_key"],
    )
    protocol = E5Protocol.from_dict(plan["protocol"])
    if (
        screen.receipt_digest != plan["screen_receipt_digest"]
        or bindings.manifest["manifest_digest"] != plan["controller_bindings_manifest_digest"]
        or policy.policy_digest != plan["m1_policy_digest"]
        or direction.direction_sha256 != plan["m1_direction_sha256"]
        or E5Protocol.from_dict(bindings.manifest["protocol"]) != protocol
        or not isinstance(capture_plan, Mapping)
        or grid.manifest["fit_provenance"]["capture_artifact_sha256"] != plan["fit_capture_sha256"]
        or capture_plan.get("plan_identity")
        != grid.manifest["fit_provenance"]["capture_plan_identity"]
        or capture_plan.get("execution_public_key") != plan["execution_public_key"]
        or capture_plan.get("runtime_artifact_sha256") != plan["runtime_artifact_sha256"]
        or capture_plan.get("runtime_identity") != plan["runtime_identity"]
        or capture_plan.get("protocol") != protocol.to_dict()
        or len(screen.dev_questions) != plan["question_count"]
        or stable_hash([value.question_id for value in screen.dev_questions])
        != plan["question_ids_sha256"]
        or stable_hash([_question_body(value) for value in screen.dev_questions])
        != plan["questions_sha256"]
        or _schedule_size(build_e5_ablation_grid(protocol), len(screen.dev_questions))
        != plan["expected_records"]
        or set(plan["prompts"]) != set(_PROMPTS)
    ):
        raise FrozenArtifactError("E5 native plan differs from its source replay")
    for prompt_id in _PROMPTS:
        _prompt_from_plan(plan, prompt_id)
    scientific = bool(
        protocol.scientific_eligible
        and screen.scientific_eligible
        and len(screen.dev_questions) == 5_000
        and bindings.scientific_eligible
        and _runtime_is_scientific(plan["runtime_identity"])
        and plan["max_new_tokens"] == _MAX_NEW_TOKENS
        and plan["max_peak_memory_bytes"] == _MAX_MEMORY_BYTES
    )
    if plan["scientific_eligible"] is not scientific:
        raise FrozenArtifactError("E5 native scientific eligibility differs")
    return screen, bindings, policy, direction


def _open_resume_sources(
    plan: Mapping[str, Any],
) -> tuple[
    E4ScreenReceipt,
    VerifiedE5ControllerBindings,
    E4MethodPolicy,
    ResolvedStaticDirection,
]:
    """Open only bounded source metadata needed to continue native execution."""

    paths = _source_paths(plan)
    screen = load_e4_screen_receipt(paths["screen_receipt"])
    bindings = open_e5_controller_bindings_checkpoint(paths["controller_bindings"])
    policy, direction = _static_source(
        policy_path=paths["m1_policy"],
        vectors_path=paths["e3_static_vectors"],
        execution_public_key=plan["execution_public_key"],
    )
    protocol = E5Protocol.from_dict(plan["protocol"])
    if (
        sha256_file(paths["screen_receipt"]) != plan["screen_receipt_sha256"]
        or screen.receipt_digest != plan["screen_receipt_digest"]
        or bindings.manifest["manifest_digest"]
        != plan["controller_bindings_manifest_digest"]
        or bindings.manifest["execution_public_key"] != plan["execution_public_key"]
        or E5Protocol.from_dict(bindings.manifest["protocol"]) != protocol
        or policy.policy_digest != plan["m1_policy_digest"]
        or direction.direction_sha256 != plan["m1_direction_sha256"]
        or len(screen.dev_questions) != plan["question_count"]
        or stable_hash([value.question_id for value in screen.dev_questions])
        != plan["question_ids_sha256"]
        or stable_hash([_question_body(value) for value in screen.dev_questions])
        != plan["questions_sha256"]
        or _schedule_size(build_e5_ablation_grid(protocol), len(screen.dev_questions))
        != plan["expected_records"]
        or set(plan["prompts"]) != set(_PROMPTS)
    ):
        raise FrozenArtifactError("E5 resume sources differ from the frozen plan")
    for prompt_id in _PROMPTS:
        _prompt_from_plan(plan, prompt_id)
    return screen, bindings, policy, direction


def _shard_directories(directory: Path) -> tuple[Path, ...]:
    root = directory / "shards"
    if root.is_symlink() or not root.is_dir():
        raise FrozenArtifactError("E5 native shard root differs")
    values = sorted(root.iterdir(), key=lambda value: value.name)
    if any(value.is_symlink() or not value.is_dir() for value in values):
        raise FrozenArtifactError("E5 native shard inventory contains a non-directory")
    for index, value in enumerate(values):
        match = _SHARD.fullmatch(value.name)
        if match is None or int(match.group(1)) != index:
            raise FrozenArtifactError("E5 native shard sequence differs")
    return tuple(values)


def _read_jsonl(path: Path, *, expected_sha256: str) -> list[dict[str, Any]]:
    if path.is_symlink() or sha256_file(path) != expected_sha256:
        raise FrozenArtifactError("E5 native shard record bytes changed")
    values: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("row is not a mapping")
                if line != json.dumps(value, sort_keys=True) + "\n":
                    raise TypeError("row is not canonical JSONL")
                values.append(value)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise FrozenArtifactError(f"cannot read E5 native shard records: {exc}") from exc
    return values


def _verify_signature(value: Mapping[str, Any], *, public_key_hex: str, context: str) -> None:
    signature = value.get("signature")
    body = dict(value)
    body.pop("signature", None)
    body.pop("chain_head", None)
    if not isinstance(signature, str) or re.fullmatch(r"[0-9a-f]{128}", signature) is None:
        raise FrozenArtifactError(f"{context} signature encoding differs")
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex)).verify(
            bytes.fromhex(signature), canonical_json(body).encode()
        )
    except (InvalidSignature, ValueError) as exc:
        raise FrozenArtifactError(f"{context} signature is invalid") from exc


def _token_indices(scope: TokenScope, output_tokens: int) -> list[int]:
    if scope is TokenScope.FINAL_PROMPT:
        return [-1]
    limit = 1 if scope is TokenScope.FIRST_GENERATED else 4
    return list(range(min(limit, output_tokens)))


def _verify_execution_signature(record: E5AblationRecord, *, public_key_hex: str) -> None:
    assert record.execution_receipt_signature is not None
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex)).verify(
            bytes.fromhex(record.execution_receipt_signature),
            canonical_json(e5_ablation_execution_receipt_body(record)).encode(),
        )
    except (InvalidSignature, ValueError) as exc:
        raise FrozenArtifactError("E5 native row signature is invalid") from exc


def _finite_float_list(value: object, *, context: str) -> list[float]:
    if not isinstance(value, list) or any(
        isinstance(item, bool)
        or not isinstance(item, int | float)
        or not math.isfinite(float(item))
        for item in value
    ):
        raise FrozenArtifactError(f"{context} must be a finite numeric list")
    return [float(item) for item in value]


_EVIDENCE_KEYS = frozenset(
    {
        "schema_version",
        "runtime_identity_sha256",
        "method_artifact_sha256",
        "rendered_prompt_token_ids_sha256",
        "raw_output",
        "raw_output_sha256",
        "output_token_ids",
        "output_token_ids_sha256",
        "input_tokens",
        "output_tokens",
        "generation_latency_seconds",
        "end_to_end_latency_seconds",
        "feature_peak_memory_bytes",
        "generation_peak_memory_bytes",
        "active_memory_bytes",
        "cache_memory_bytes",
        "feature_schema_digest",
        "feature_shape",
        "feature_values_sha256",
        "maximum_token_probability",
        "output_entropy",
        "selected_layer",
        "selected_site",
        "standardized_alpha",
        "effective_raw_alpha",
        "direction_sha256",
        "direction_norm",
        "routing_weights_sha256",
        "routing_weights",
        "layer_routing_weights",
        "hook_applications",
        "applied_token_indices",
        "pre_activation_sha256",
        "post_activation_sha256",
        "delta_sha256",
        "hook_delta_norms",
        "hook_direction_dot_products",
        "hook_residual_norms",
        "expected_activation_delta_norm",
        "evidence_digest",
    }
)


def _verify_native_row(
    value: Mapping[str, Any],
    *,
    sequence: int,
    plan: Mapping[str, Any],
    specs: Sequence[E5AblationSpec],
    questions: Sequence[Question],
    bindings: VerifiedE5ControllerBindings,
    m1_policy: E4MethodPolicy,
    m1_direction: ResolvedStaticDirection,
    binding_cache: dict[str, tuple[E5ControllerBinding, AdaptiveController, str]],
) -> E5AblationRecord:
    if set(value) != {"sequence", "record", "evidence", "row_digest"}:
        raise FrozenArtifactError("E5 native row schema differs")
    body = dict(value)
    digest = body.pop("row_digest", None)
    if value["sequence"] != sequence or digest != stable_hash(body):
        raise FrozenArtifactError("E5 native row identity differs")
    try:
        record = E5AblationRecord.from_dict(value["record"])
    except (KeyError, TypeError, ValueError, DataValidationError) as exc:
        raise FrozenArtifactError(f"E5 native ablation record is invalid: {exc}") from exc
    arm_id, spec, prompt_id, question = _schedule_row(sequence, specs=specs, questions=questions)
    evidence = value["evidence"]
    if not isinstance(evidence, dict) or set(evidence) != _EVIDENCE_KEYS:
        raise FrozenArtifactError("E5 native evidence schema differs")
    evidence_body = dict(evidence)
    evidence_digest = evidence_body.pop("evidence_digest", None)
    output_ids = evidence.get("output_token_ids")
    action = record.execution_receipt["policy_action"]
    adaptive = spec is not None
    memory_fields = (
        evidence["feature_peak_memory_bytes"],
        evidence["generation_peak_memory_bytes"],
        evidence["active_memory_bytes"],
        evidence["cache_memory_bytes"],
    )
    latency_fields = (
        evidence["generation_latency_seconds"],
        evidence["end_to_end_latency_seconds"],
    )
    if (
        evidence_digest != stable_hash(evidence_body)
        or (record.arm_id, record.prompt_id, record.question_id)
        != (arm_id, prompt_id, question.question_id)
        or record.prompt_input_sha256 != _e5_prompt_input_sha256(question, prompt_id)
        or record.prompt_template_sha256 != _PROMPT_HASHES[prompt_id]
        or evidence["runtime_identity_sha256"] != plan["runtime_identity_sha256"]
        or type(evidence["raw_output"]) is not str
        or evidence["raw_output_sha256"]
        != hashlib.sha256(evidence["raw_output"].encode()).hexdigest()
        or deterministic_short_answer_grade(evidence["raw_output"], question.aliases)
        is not record.outcome
        or record.outcome is Outcome.UNSCORABLE
        or type(output_ids) is not list
        or any(type(item) is not int or item < 0 for item in output_ids)
        or evidence["output_token_ids_sha256"]
        != hashlib.sha256(",".join(str(item) for item in output_ids).encode("ascii")).hexdigest()
        or evidence["output_tokens"] != len(output_ids)
        or evidence["output_tokens"] != record.output_tokens
        or type(evidence["input_tokens"]) is not int
        or evidence["input_tokens"] <= 0
        or any(
            isinstance(item, bool)
            or not isinstance(item, int | float)
            or not math.isfinite(float(item))
            or float(item) < 0
            for item in latency_fields
        )
        or float(evidence["end_to_end_latency_seconds"])
        < float(evidence["generation_latency_seconds"])
        or evidence["end_to_end_latency_seconds"] != record.generation_latency_seconds
        or record.intervention_norm != float(record.execution_receipt["activation_delta_norm"])
        or evidence["applied_token_indices"] != record.execution_receipt["applied_token_indices"]
        or any(type(item) is not int or item < 0 for item in memory_fields)
        or evidence["active_memory_bytes"] > evidence["generation_peak_memory_bytes"]
        or evidence["cache_memory_bytes"] > evidence["generation_peak_memory_bytes"]
        or max(
            int(evidence["feature_peak_memory_bytes"]),
            int(evidence["generation_peak_memory_bytes"]),
        )
        > plan["max_peak_memory_bytes"]
        or _SHA256.fullmatch(str(evidence["rendered_prompt_token_ids_sha256"])) is None
        or type(evidence["selected_layer"]) is not int
        or evidence["selected_layer"] < 0
        or evidence["selected_site"] not in {item.value for item in ActivationSite}
        or any(
            isinstance(evidence[name], bool)
            or not isinstance(evidence[name], int | float)
            or not math.isfinite(float(evidence[name]))
            for name in (
                "standardized_alpha",
                "effective_raw_alpha",
                "direction_norm",
                "expected_activation_delta_norm",
            )
        )
        or float(evidence["direction_norm"]) <= 0
        or _SHA256.fullmatch(str(evidence["direction_sha256"])) is None
        or type(evidence["hook_applications"]) is not int
        or evidence["hook_applications"] < 0
    ):
        raise FrozenArtifactError("E5 native row differs from its signed execution evidence")
    if adaptive:
        assert spec is not None
        cached = binding_cache.get(arm_id)
        if cached is None:
            binding_path = bindings.binding_paths[arm_id]
            binding = load_e5_controller_binding(binding_path)
            controller = binding.assert_current()
            binding_sha = sha256_file(binding_path)
            binding_cache.clear()
            binding_cache[arm_id] = (binding, controller, binding_sha)
        else:
            binding, controller, binding_sha = cached
        feature_shape = evidence["feature_shape"]
        routing_values = _finite_float_list(
            evidence["routing_weights"], context="E5 vector-routing transcript"
        )
        layer_routing_value = evidence["layer_routing_weights"]
        scores = record.execution_receipt["controller_scores"]
        if (
            record.controller_binding_sha256 != binding_sha
            or record.execution_receipt["controller_artifact_sha256"]
            != binding.controller_artifact_sha256
            or record.token_scope is not _TOKEN_SCOPES[spec.intervention_timing]
            or evidence["method_artifact_sha256"] != binding.controller_artifact_sha256
            or evidence["feature_schema_digest"] != controller.risk_probe.training_schema.digest
            or feature_shape != [1, controller.risk_probe.state.input_width]
            or _SHA256.fullmatch(str(evidence["feature_values_sha256"])) is None
            or _SHA256.fullmatch(str(evidence["routing_weights_sha256"])) is None
            or len(routing_values) != controller.vector_bank.cluster_count
            or any(value < 0 for value in routing_values)
            or not math.isclose(sum(routing_values), 1.0, rel_tol=0, abs_tol=1e-6)
            or hashlib.sha256(
                np.ascontiguousarray(routing_values, dtype=np.float32)
                .reshape(1, -1)
                .tobytes(order="C")
            ).hexdigest()
            != evidence["routing_weights_sha256"]
            or isinstance(evidence["maximum_token_probability"], bool)
            or not isinstance(evidence["maximum_token_probability"], int | float)
            or not 0 < float(evidence["maximum_token_probability"]) <= 1
            or isinstance(evidence["output_entropy"], bool)
            or not isinstance(evidence["output_entropy"], int | float)
            or float(evidence["output_entropy"]) < 0
        ):
            raise FrozenArtifactError("E5 adaptive row differs from its controller binding")
        if controller.fixed_layer is not None:
            if layer_routing_value is not None:
                raise FrozenArtifactError("E5 fixed-layer controller has a routing transcript")
            expected_layer = controller.fixed_layer
        else:
            selector = controller.layer_selector
            assert selector is not None
            layer_values = _finite_float_list(
                layer_routing_value, context="E5 layer-routing transcript"
            )
            if (
                len(layer_values) != len(selector.candidate_layers)
                or any(value < 0 for value in layer_values)
                or not math.isclose(sum(layer_values), 1.0, rel_tol=0, abs_tol=1e-6)
            ):
                raise FrozenArtifactError("E5 layer-routing transcript differs")
            expected_layer = selector.candidate_layers[
                max(range(len(layer_values)), key=layer_values.__getitem__)
            ]
        routing_tensor = torch.tensor([routing_values], dtype=torch.float32)
        expected_directions = controller.vector_bank.mix(routing_tensor)
        expected_site, expected_direction, expected_direction_norm = _direction_choice(
            controller,
            selected_layer=expected_layer,
            directions=expected_directions,
        )
        expected_alpha = float(
            controller.alpha_controller.alpha(
                torch.tensor([float(scores["I"])], dtype=torch.float32)
            )[0]
        )
        expected_raw_alpha = expected_alpha * expected_direction_norm
        expected_action = "intervene" if expected_alpha > 0 else "release"
        if (
            evidence["selected_layer"] != expected_layer
            or evidence["selected_site"] != expected_site.value
            or not math.isclose(
                float(evidence["standardized_alpha"]),
                expected_alpha,
                rel_tol=1e-6,
                abs_tol=1e-7,
            )
            or not math.isclose(
                float(evidence["effective_raw_alpha"]),
                expected_raw_alpha,
                rel_tol=1e-6,
                abs_tol=1e-7,
            )
            or not math.isclose(
                float(evidence["direction_norm"]),
                expected_direction_norm,
                rel_tol=1e-6,
                abs_tol=1e-7,
            )
            or evidence["direction_sha256"]
            != hashlib.sha256(expected_direction.tobytes(order="C")).hexdigest()
            or action != expected_action
        ):
            raise FrozenArtifactError("E5 adaptive policy transcript does not replay")
    elif (
        record.controller_binding_sha256 is not None
        or evidence["method_artifact_sha256"] != plan["m1_policy_sha256"]
        or record.token_scope is not m1_policy.token_scope
        or evidence["direction_sha256"] != plan["m1_direction_sha256"]
        or evidence["feature_schema_digest"] is not None
        or evidence["feature_shape"] is not None
        or evidence["feature_values_sha256"] is not None
        or evidence["routing_weights_sha256"] is not None
        or evidence["routing_weights"] is not None
        or evidence["layer_routing_weights"] is not None
        or evidence["maximum_token_probability"] is not None
        or evidence["output_entropy"] is not None
        or evidence["selected_layer"] != m1_policy.layer
        or evidence["selected_site"]
        != (m1_policy.site.value if m1_policy.site is not None else None)
        or not math.isclose(
            float(evidence["standardized_alpha"]), m1_policy.alpha, rel_tol=0, abs_tol=1e-12
        )
        or m1_policy.reference_rms is None
        or not math.isclose(
            float(evidence["effective_raw_alpha"]),
            m1_policy.alpha * m1_policy.reference_rms,
            rel_tol=0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(evidence["direction_norm"]),
            m1_direction.direction_norm,
            rel_tol=0,
            abs_tol=1e-7,
        )
        or action != "intervene"
    ):
        raise FrozenArtifactError("E5 M1 row differs from its static reference")
    _verify_execution_signature(record, public_key_hex=plan["execution_public_key"])
    indices = _token_indices(record.token_scope, record.output_tokens)
    expected_norm = evidence["expected_activation_delta_norm"]
    delta_norms = _finite_float_list(evidence["hook_delta_norms"], context="E5 hook delta norms")
    direction_dots = _finite_float_list(
        evidence["hook_direction_dot_products"], context="E5 hook direction products"
    )
    residual_norms = _finite_float_list(
        evidence["hook_residual_norms"], context="E5 hook residual norms"
    )
    hook_fields = (
        evidence["pre_activation_sha256"],
        evidence["post_activation_sha256"],
        evidence["delta_sha256"],
    )
    if action == "intervene":
        if (
            not indices
            or evidence["applied_token_indices"] != indices
            or evidence["hook_applications"] != len(indices)
            or len(delta_norms) != len(indices)
            or len(direction_dots) != len(indices)
            or len(residual_norms) != len(indices)
            or any(type(item) is not str or _SHA256.fullmatch(item) is None for item in hook_fields)
            or not math.isclose(
                record.intervention_norm,
                float(expected_norm),
                rel_tol=0.025,
                abs_tol=1e-6,
            )
            or record.intervention_norm <= 0
            or not math.isclose(
                float(expected_norm),
                abs(float(evidence["effective_raw_alpha"])) * math.sqrt(len(indices)),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                record.intervention_norm,
                math.sqrt(sum(value * value for value in delta_norms)),
                rel_tol=1e-9,
                abs_tol=1e-7,
            )
            or any(
                not math.isclose(
                    value,
                    abs(float(evidence["effective_raw_alpha"])),
                    rel_tol=0.025,
                    abs_tol=1e-6,
                )
                for value in delta_norms
            )
            or any(
                not math.isclose(
                    value,
                    float(evidence["effective_raw_alpha"]),
                    rel_tol=0.025,
                    abs_tol=1e-6,
                )
                for value in direction_dots
            )
            or any(
                value > max(1e-6, abs(float(evidence["effective_raw_alpha"])) * 0.025)
                for value in residual_norms
            )
        ):
            raise FrozenArtifactError("E5 material intervention evidence differs")
    elif (
        action != "release"
        or evidence["applied_token_indices"] != []
        or evidence["hook_applications"] != 0
        or hook_fields != (None, None, None)
        or delta_norms != []
        or direction_dots != []
        or residual_norms != []
        or expected_norm != 0.0
        or record.intervention_norm != 0.0
    ):
        raise FrozenArtifactError("E5 release evidence differs")
    return record


def _verify_finalization(
    directory: Path,
    *,
    plan: Mapping[str, Any],
    chain_head: str,
    records_completed: int,
) -> Path | None:
    final = directory / "final"
    if not final.exists():
        return None
    if (
        final.is_symlink()
        or not final.is_dir()
        or {value.name for value in final.iterdir()} != {"records.jsonl", "receipt.json"}
        or any(value.is_symlink() for value in final.iterdir())
    ):
        raise FrozenArtifactError("E5 native finalization inventory differs")
    try:
        receipt = json.loads((final / "receipt.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read E5 native finalization: {exc}") from exc
    if not isinstance(receipt, dict):
        raise FrozenArtifactError("E5 native finalization receipt must be a mapping")
    if (final / "receipt.json").read_text(encoding="utf-8") != (
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    ):
        raise FrozenArtifactError("E5 native finalization receipt is not canonical JSON")
    _verify_signature(
        receipt,
        public_key_hex=plan["execution_public_key"],
        context="E5 native finalization",
    )
    body = dict(receipt)
    signature = body.pop("signature", None)
    observed_chain = body.pop("chain_head", None)
    expected = {
        "schema_version",
        "plan_identity",
        "record_count",
        "source_chain_head",
        "records_relative_path",
        "records_sha256",
        "controller_bindings_sha256",
        "screen_receipt_sha256",
        "finalized_at",
        "receipt_digest",
    }
    digest = body.pop("receipt_digest", None)
    if (
        set(body) != expected - {"receipt_digest"}
        or digest != stable_hash(body)
        or observed_chain
        != stable_hash(
            {
                "signed": {**body, "receipt_digest": digest},
                "signature": signature,
            }
        )
        or body["schema_version"] != 1
        or body["plan_identity"] != plan["plan_identity"]
        or body["record_count"] != records_completed
        or body["source_chain_head"] != chain_head
        or body["records_relative_path"] != "records.jsonl"
        or body["records_sha256"] != sha256_file(final / "records.jsonl")
        or body["controller_bindings_sha256"] != plan["controller_bindings_sha256"]
        or body["screen_receipt_sha256"] != plan["screen_receipt_sha256"]
    ):
        raise FrozenArtifactError("E5 native finalization receipt differs")
    return (final / "records.jsonl").resolve()


def verify_e5_native_ablation(
    directory: str | Path,
    *,
    expected_execution_public_key: str,
    require_complete: bool = False,
    semantic: bool = True,
) -> VerifiedE5NativeRun:
    """Verify the source plan, signed shard chain, and optionally every native row."""

    source = validate_active_study_artifact_paths(
        {"E5 native ablation": directory}
    )["E5 native ablation"]
    inventory = {value.name for value in source.iterdir()} if source.is_dir() else set()
    if (
        source.is_symlink()
        or not source.is_dir()
        or inventory not in {_INVENTORY, _INVENTORY | {"final"}}
        or any((source / name).is_symlink() for name in inventory)
    ):
        raise FrozenArtifactError("E5 native run inventory differs")
    plan = _read_plan(source)
    if plan["execution_public_key"] != expected_execution_public_key:
        raise FrozenArtifactError("E5 native execution key differs from external trust root")
    screen, bindings, policy, direction = _verify_sources(plan)
    protocol = E5Protocol.from_dict(plan["protocol"])
    specs = build_e5_ablation_grid(protocol)
    source_context = _E5NativeSourceContext(
        screen=screen,
        bindings=bindings,
        m1_policy=policy,
        m1_direction=direction,
        protocol=protocol,
        specs=specs,
    )
    completed = 0
    chain_head: str | None = None
    maximum_peak = 0
    shards = _shard_directories(source)
    binding_cache: dict[str, tuple[E5ControllerBinding, AdaptiveController, str]] = {}
    for index, shard in enumerate(shards):
        if {value.name for value in shard.iterdir()} != {"manifest.json", "records.jsonl"}:
            raise FrozenArtifactError("E5 native shard file inventory differs")
        try:
            manifest = json.loads((shard / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FrozenArtifactError(f"cannot read E5 native shard manifest: {exc}") from exc
        if not isinstance(manifest, dict):
            raise FrozenArtifactError("E5 native shard manifest must be a mapping")
        if (shard / "manifest.json").read_text(encoding="utf-8") != (
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        ):
            raise FrozenArtifactError("E5 native shard manifest is not canonical JSON")
        body = dict(manifest)
        signature = body.pop("signature", None)
        observed_chain = body.pop("chain_head", None)
        digest = body.get("manifest_digest")
        unsigned = dict(body)
        unsigned.pop("manifest_digest", None)
        expected_manifest_keys = {
            "schema_version",
            "plan_identity",
            "shard_index",
            "start_record",
            "end_record",
            "record_count",
            "records_sha256",
            "record_digests",
            "previous_chain_head",
            "maximum_peak_memory_bytes",
            "cumulative_maximum_peak_memory_bytes",
            "execution_session_id",
            "execution_session_started_at",
            "execution_session_wall_seconds",
            "execution_lock_identity",
            "execution_process_id",
            "manifest_digest",
        }
        if (
            set(body) != expected_manifest_keys
            or digest != stable_hash(unsigned)
            or observed_chain != stable_hash({"signed": body, "signature": signature})
            or body["schema_version"] != 2
            or body["plan_identity"] != plan["plan_identity"]
            or body["shard_index"] != index
            or body["start_record"] != completed
            or body["end_record"] != completed + body["record_count"]
            or body["previous_chain_head"] != chain_head
            or type(body["record_digests"]) is not list
            or len(body["record_digests"]) != body["record_count"]
            or body["record_count"] <= 0
            or body["record_count"] > plan["shard_rows"]
            or body["end_record"] > plan["expected_records"]
            or _SHA256.fullmatch(str(body["records_sha256"])) is None
            or any(
                type(value) is not str or _SHA256.fullmatch(value) is None
                for value in body["record_digests"]
            )
            or type(body["maximum_peak_memory_bytes"]) is not int
            or body["maximum_peak_memory_bytes"] < 0
            or body["maximum_peak_memory_bytes"] > plan["max_peak_memory_bytes"]
            or type(body["cumulative_maximum_peak_memory_bytes"]) is not int
            or body["cumulative_maximum_peak_memory_bytes"]
            != max(maximum_peak, body["maximum_peak_memory_bytes"])
            or _SHA256.fullmatch(str(body["execution_session_id"])) is None
            or not _utc_timestamp(body["execution_session_started_at"])
            or isinstance(body["execution_session_wall_seconds"], bool)
            or not isinstance(body["execution_session_wall_seconds"], int | float)
            or not math.isfinite(float(body["execution_session_wall_seconds"]))
            or float(body["execution_session_wall_seconds"]) < 0
            or body["execution_lock_identity"] != _lock_identity(source / "run.lock")
            or type(body["execution_process_id"]) is not int
            or body["execution_process_id"] <= 0
        ):
            raise FrozenArtifactError("E5 native shard chain or manifest differs")
        _verify_signature(
            manifest,
            public_key_hex=plan["execution_public_key"],
            context="E5 native shard",
        )
        rows = _read_jsonl(shard / "records.jsonl", expected_sha256=body["records_sha256"])
        if len(rows) != body["record_count"]:
            raise FrozenArtifactError("E5 native shard row count differs")
        for offset, row in enumerate(rows):
            if row.get("row_digest") != body["record_digests"][offset]:
                raise FrozenArtifactError("E5 native shard row digest differs")
            if semantic:
                _verify_native_row(
                    row,
                    sequence=completed + offset,
                    plan=plan,
                    specs=specs,
                    questions=screen.dev_questions,
                    bindings=bindings,
                    m1_policy=policy,
                    m1_direction=direction,
                    binding_cache=binding_cache,
                )
        completed = body["end_record"]
        chain_head = observed_chain
        maximum_peak = max(maximum_peak, body["maximum_peak_memory_bytes"])
    if completed > plan["expected_records"]:
        raise FrozenArtifactError("E5 native run exceeds its frozen schedule")
    complete = completed == plan["expected_records"]
    if not complete and "final" in inventory:
        raise FrozenArtifactError("incomplete E5 native run contains a finalization")
    if require_complete and not complete:
        raise FrozenArtifactError("E5 native ablation is incomplete")
    finalized = (
        _verify_finalization(
            source,
            plan=plan,
            chain_head=chain_head or "0" * 64,
            records_completed=completed,
        )
        if complete
        else None
    )
    return VerifiedE5NativeRun(
        directory=source.resolve(),
        plan=MappingProxyType(plan),
        records_completed=completed,
        shard_count=len(shards),
        chain_head=chain_head,
        complete=complete,
        scientific_eligible=bool(plan["scientific_eligible"] and complete),
        maximum_peak_memory_bytes=maximum_peak,
        finalized_records=finalized,
        _source_context=source_context,
    )


def open_e5_native_promotion_source(
    directory: str | Path,
    *,
    expected_execution_public_key: str,
    expected_final_records_sha256: str,
) -> VerifiedE5NativeRun:
    """Open finalized E5 for promotion without rescanning 9.73M nonselected rows.

    The full semantic scan is mandatory during native finalization and again while
    deriving the signed selection.  Later promotion sessions verify the signed plan,
    source artifacts, every shard manifest in the chain, and the signed final receipt.
    ``iter_e5_native_promotion_rows`` then hashes and semantically replays the only
    shard files that can contribute to the final 20,000-row ledger.
    """

    source = validate_active_study_artifact_paths(
        {"E5 native promotion source": directory}
    )["E5 native promotion source"]
    inventory = {value.name for value in source.iterdir()} if source.is_dir() else set()
    if (
        source.is_symlink()
        or not source.is_dir()
        or inventory != _INVENTORY | {"final"}
        or any((source / name).is_symlink() for name in inventory)
    ):
        raise FrozenArtifactError("E5 promotion source inventory differs")
    _digest(expected_final_records_sha256, "E5 expected final record set")
    plan = _read_plan(source)
    if plan["execution_public_key"] != expected_execution_public_key:
        raise FrozenArtifactError("E5 promotion execution key differs")
    screen, bindings, policy, direction = _verify_sources(plan)
    protocol = E5Protocol.from_dict(plan["protocol"])
    specs = build_e5_ablation_grid(protocol)
    context = _E5NativeSourceContext(
        screen=screen,
        bindings=bindings,
        m1_policy=policy,
        m1_direction=direction,
        protocol=protocol,
        specs=specs,
    )
    completed = 0
    chain_head: str | None = None
    maximum_peak = 0
    shards = _shard_directories(source)
    for index, shard in enumerate(shards):
        if {value.name for value in shard.iterdir()} != {"manifest.json", "records.jsonl"}:
            raise FrozenArtifactError("E5 promotion shard file inventory differs")
        records_path = shard / "records.jsonl"
        if records_path.is_symlink() or not records_path.is_file():
            raise FrozenArtifactError("E5 promotion shard records are not regular")
        try:
            manifest = json.loads((shard / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FrozenArtifactError(f"cannot read E5 promotion manifest: {exc}") from exc
        if not isinstance(manifest, dict) or (shard / "manifest.json").read_text(
            encoding="utf-8"
        ) != (json.dumps(manifest, indent=2, sort_keys=True) + "\n"):
            raise FrozenArtifactError("E5 promotion manifest is not canonical JSON")
        body = dict(manifest)
        signature = body.pop("signature", None)
        observed_chain = body.pop("chain_head", None)
        digest = body.get("manifest_digest")
        unsigned = dict(body)
        unsigned.pop("manifest_digest", None)
        expected_keys = {
            "schema_version",
            "plan_identity",
            "shard_index",
            "start_record",
            "end_record",
            "record_count",
            "records_sha256",
            "record_digests",
            "previous_chain_head",
            "maximum_peak_memory_bytes",
            "cumulative_maximum_peak_memory_bytes",
            "execution_session_id",
            "execution_session_started_at",
            "execution_session_wall_seconds",
            "execution_lock_identity",
            "execution_process_id",
            "manifest_digest",
        }
        if (
            set(body) != expected_keys
            or digest != stable_hash(unsigned)
            or observed_chain != stable_hash({"signed": body, "signature": signature})
            or body["schema_version"] != 2
            or body["plan_identity"] != plan["plan_identity"]
            or body["shard_index"] != index
            or body["start_record"] != completed
            or type(body["record_count"]) is not int
            or body["record_count"] <= 0
            or body["record_count"] > plan["shard_rows"]
            or body["end_record"] != completed + body["record_count"]
            or body["end_record"] > plan["expected_records"]
            or body["previous_chain_head"] != chain_head
            or type(body["record_digests"]) is not list
            or len(body["record_digests"]) != body["record_count"]
            or any(
                type(value) is not str or _SHA256.fullmatch(value) is None
                for value in body["record_digests"]
            )
            or _SHA256.fullmatch(str(body["records_sha256"])) is None
            or type(body["maximum_peak_memory_bytes"]) is not int
            or not 0 <= body["maximum_peak_memory_bytes"] <= plan["max_peak_memory_bytes"]
            or type(body["cumulative_maximum_peak_memory_bytes"]) is not int
            or body["cumulative_maximum_peak_memory_bytes"]
            != max(maximum_peak, body["maximum_peak_memory_bytes"])
            or _SHA256.fullmatch(str(body["execution_session_id"])) is None
            or not _utc_timestamp(body["execution_session_started_at"])
            or isinstance(body["execution_session_wall_seconds"], bool)
            or not isinstance(body["execution_session_wall_seconds"], int | float)
            or not math.isfinite(float(body["execution_session_wall_seconds"]))
            or float(body["execution_session_wall_seconds"]) < 0
            or body["execution_lock_identity"] != _lock_identity(source / "run.lock")
            or type(body["execution_process_id"]) is not int
            or body["execution_process_id"] <= 0
        ):
            raise FrozenArtifactError("E5 promotion shard chain or manifest differs")
        _verify_signature(
            manifest,
            public_key_hex=plan["execution_public_key"],
            context="E5 promotion shard",
        )
        completed = body["end_record"]
        chain_head = observed_chain
        maximum_peak = max(maximum_peak, body["maximum_peak_memory_bytes"])
    if completed != plan["expected_records"] or chain_head is None:
        raise FrozenArtifactError("E5 promotion source is incomplete")
    final = source / "final"
    if (
        final.is_symlink()
        or not final.is_dir()
        or {value.name for value in final.iterdir()} != {"records.jsonl", "receipt.json"}
        or any(value.is_symlink() for value in final.iterdir())
        or not (final / "records.jsonl").is_file()
    ):
        raise FrozenArtifactError("E5 promotion finalization inventory differs")
    try:
        receipt = json.loads((final / "receipt.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read E5 promotion finalization: {exc}") from exc
    if not isinstance(receipt, dict) or (final / "receipt.json").read_text(encoding="utf-8") != (
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    ):
        raise FrozenArtifactError("E5 promotion finalization is not canonical JSON")
    _verify_signature(
        receipt,
        public_key_hex=plan["execution_public_key"],
        context="E5 promotion finalization",
    )
    receipt_body = dict(receipt)
    signature = receipt_body.pop("signature", None)
    receipt_chain = receipt_body.pop("chain_head", None)
    receipt_digest = receipt_body.pop("receipt_digest", None)
    expected_receipt_keys = {
        "schema_version",
        "plan_identity",
        "record_count",
        "source_chain_head",
        "records_relative_path",
        "records_sha256",
        "controller_bindings_sha256",
        "screen_receipt_sha256",
        "finalized_at",
    }
    signed = {**receipt_body, "receipt_digest": receipt_digest}
    if (
        set(receipt_body) != expected_receipt_keys
        or receipt_digest != stable_hash(receipt_body)
        or receipt_chain != stable_hash({"signed": signed, "signature": signature})
        or receipt_body["schema_version"] != 1
        or receipt_body["plan_identity"] != plan["plan_identity"]
        or receipt_body["record_count"] != completed
        or receipt_body["source_chain_head"] != chain_head
        or receipt_body["records_relative_path"] != "records.jsonl"
        or receipt_body["records_sha256"] != expected_final_records_sha256
        or receipt_body["controller_bindings_sha256"] != plan["controller_bindings_sha256"]
        or receipt_body["screen_receipt_sha256"] != plan["screen_receipt_sha256"]
    ):
        raise FrozenArtifactError("E5 promotion finalization receipt differs")
    return VerifiedE5NativeRun(
        directory=source.resolve(),
        plan=MappingProxyType(plan),
        records_completed=completed,
        shard_count=len(shards),
        chain_head=chain_head,
        complete=True,
        scientific_eligible=bool(plan["scientific_eligible"]),
        maximum_peak_memory_bytes=maximum_peak,
        finalized_records=(final / "records.jsonl").resolve(),
        _source_context=context,
    )


def iter_e5_native_promotion_rows(
    verified: VerifiedE5NativeRun,
    *,
    selected_spec_id: str,
) -> Iterator[E5NativePromotionRow]:
    """Replay only M1 and the selected M3 arm from a finalized full-grid run."""

    if (
        type(verified) is not VerifiedE5NativeRun
        or not verified.complete
        or not verified.scientific_eligible
        or verified.finalized_records is None
        or verified.chain_head is None
        or E5Protocol.from_dict(verified.plan["protocol"]) != E5Protocol()
    ):
        raise DataValidationError(
            "E5 promotion requires a finalized scientific full-grid native run"
        )
    context = verified._source_context
    try:
        selected_index = next(
            index for index, spec in enumerate(context.specs) if spec.spec_id == selected_spec_id
        )
    except StopIteration as exc:
        raise DataValidationError("E5 selected specification is outside the full grid") from exc
    question_count = len(context.screen.dev_questions)
    block = len(_PROMPTS) * question_count
    ranges = ((0, block), ((selected_index + 1) * block, (selected_index + 2) * block))

    def selected(sequence: int) -> bool:
        return any(start <= sequence < end for start, end in ranges)

    yielded = 0
    binding_cache: dict[str, tuple[E5ControllerBinding, AdaptiveController, str]] = {}
    for shard in _shard_directories(verified.directory):
        try:
            manifest = json.loads((shard / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FrozenArtifactError(f"cannot read E5 promotion shard: {exc}") from exc
        if not isinstance(manifest, dict):
            raise FrozenArtifactError("E5 promotion shard manifest must be a mapping")
        _verify_signature(
            manifest,
            public_key_hex=verified.plan["execution_public_key"],
            context="E5 promotion shard",
        )
        start = manifest.get("start_record")
        end = manifest.get("end_record")
        if (
            type(start) is not int
            or type(end) is not int
            or not any(start < range_end and end > range_start for range_start, range_end in ranges)
        ):
            continue
        rows = _read_jsonl(
            shard / "records.jsonl",
            expected_sha256=str(manifest["records_sha256"]),
        )
        for offset, value in enumerate(rows):
            sequence = start + offset
            if not selected(sequence):
                continue
            record = _verify_native_row(
                value,
                sequence=sequence,
                plan=verified.plan,
                specs=context.specs,
                questions=context.screen.dev_questions,
                bindings=context.bindings,
                m1_policy=context.m1_policy,
                m1_direction=context.m1_direction,
                binding_cache=binding_cache,
            )
            evidence = value.get("evidence")
            row_digest = value.get("row_digest")
            if not isinstance(evidence, dict) or type(row_digest) is not str:
                raise FrozenArtifactError("E5 promotion row evidence differs")
            yielded += 1
            yield E5NativePromotionRow(
                sequence=sequence,
                record=record,
                evidence=MappingProxyType(dict(evidence)),
                row_digest=row_digest,
            )
    if yielded != 2 * block:
        raise FrozenArtifactError("E5 promotion row count differs from the paired schedule")


def _strict_hook_arrays(
    state: Any,
    *,
    direction: np.ndarray[Any, Any],
    raw_alpha: float,
    expected_applications: int,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    pre_history = getattr(state, "applied_pre_history", None)
    post_history = getattr(state, "applied_post_history", None)
    if (
        not isinstance(pre_history, list)
        or not isinstance(post_history, list)
        or len(pre_history) != expected_applications
        or len(post_history) != expected_applications
        or getattr(state, "applications", None) != expected_applications
        or expected_applications <= 0
    ):
        raise FrozenArtifactError("E5 native hook lacks its exact applied-edit history")
    try:
        captured = np.ascontiguousarray(np.stack(pre_history).astype(np.float32, copy=False))
        intervened = np.ascontiguousarray(np.stack(post_history).astype(np.float32, copy=False))
        expected = np.stack([direction * raw_alpha] * expected_applications).astype(
            np.float32, copy=False
        )
        if captured.shape != expected.shape and captured.size == expected.size:
            captured = captured.reshape(expected.shape)
            intervened = intervened.reshape(expected.shape)
    except (TypeError, ValueError) as exc:
        raise FrozenArtifactError(f"E5 native hook arrays are invalid: {exc}") from exc
    delta = np.ascontiguousarray(intervened - captured, dtype=np.float32)
    tolerance = max(1e-6, float(np.max(np.abs(expected))) * 0.025)
    if (
        direction.ndim != 1
        or captured.shape != expected.shape
        or intervened.shape != expected.shape
        or captured.size == 0
        or not np.isfinite(captured).all()
        or not np.isfinite(intervened).all()
        or np.array_equal(captured, intervened)
        or not np.allclose(delta, expected, rtol=0.025, atol=tolerance)
    ):
        raise FrozenArtifactError("E5 native hook edit differs from direction and alpha")
    return captured, intervened, delta


def _output_token_digest(values: Sequence[int]) -> str:
    return hashlib.sha256(",".join(str(int(value)) for value in values).encode("ascii")).hexdigest()


def _direction_choice(
    controller: AdaptiveController,
    *,
    selected_layer: int,
    directions: Mapping[Any, torch.Tensor],
) -> tuple[ActivationSite, np.ndarray[Any, Any], float]:
    eligible = [
        (key.site, value[0].detach().cpu().float().contiguous())
        for key, value in directions.items()
        if key.layer == selected_layer
    ]
    if not eligible:
        raise FrozenArtifactError("E5 controller selected a layer without a direction")
    site, selected = min(
        eligible,
        key=lambda item: (-float(torch.linalg.vector_norm(item[1])), item[0].value),
    )
    direction = np.ascontiguousarray(selected.numpy(), dtype=np.float32)
    norm = float(np.linalg.norm(direction.astype(np.float64)))
    if direction.ndim != 1 or not math.isfinite(norm) or norm <= 0:
        raise DataValidationError("E5 routed direction is invalid")
    return site, np.ascontiguousarray(direction / norm, dtype=np.float32), norm


def _execute_native_row(
    *,
    sequence: int,
    plan: Mapping[str, Any],
    specs: Sequence[E5AblationSpec],
    questions: Sequence[Question],
    runtime: E5NativeRuntime,
    bindings: VerifiedE5ControllerBindings,
    m1_policy: E4MethodPolicy,
    m1_direction: ResolvedStaticDirection,
    private_key_hex: str,
    controller_cache: dict[str, tuple[E5ControllerBinding, AdaptiveController, str]],
) -> dict[str, Any]:
    arm_id, spec, prompt_id, question = _schedule_row(sequence, specs=specs, questions=questions)
    prompt = _prompt_from_plan(plan, prompt_id)
    started = time.perf_counter()
    rendered = runtime.render_prompt(prompt, question.text, metadata=question.metadata)
    if type(rendered) is not MlxRenderedPrompt:
        raise FrozenArtifactError("E5 runtime returned a non-MLX rendered prompt")
    feature_peak = 0
    feature_schema_digest: str | None = None
    feature_shape: list[int] | None = None
    feature_values_sha256: str | None = None
    maximum_probability: float | None = None
    output_entropy: float | None = None
    routing_sha256: str | None = None
    routing_values: list[float] | None = None
    layer_routing_values: list[float] | None = None
    scores: dict[str, float] = {}
    binding_sha: str | None = None
    controller_artifact_sha: str | None = None
    method_artifact_sha: str
    selected_layer: int
    selected_site: ActivationSite
    scope: TokenScope
    standardized_alpha: float
    raw_alpha: float
    normalized_direction: np.ndarray[Any, Any]
    direction_sha: str
    direction_norm: float
    action: str
    interventions: dict[tuple[int, ActivationSite], Any] = {}
    state: Any | None = None
    if spec is None:
        if (
            m1_policy.layer is None
            or m1_policy.site is None
            or m1_policy.token_scope is None
            or m1_policy.reference_rms is None
        ):
            raise FrozenArtifactError("E5 M1 policy lacks fixed geometry")
        selected_layer = m1_policy.layer
        selected_site = m1_policy.site
        scope = m1_policy.token_scope
        standardized_alpha = m1_policy.alpha
        raw_alpha = standardized_alpha * m1_policy.reference_rms
        normalized_direction = np.ascontiguousarray(
            m1_direction.direction.numpy(), dtype=np.float32
        )
        direction_sha = m1_direction.direction_sha256
        direction_norm = m1_direction.direction_norm
        method_artifact_sha = plan["m1_policy_sha256"]
        action = "intervene"
        state = runtime.standardized_intervention_state(
            normalized_direction,
            standardized_alpha=standardized_alpha,
            reference_rms=m1_policy.reference_rms,
            token_scope=scope,
        )
        interventions[(selected_layer, selected_site)] = state
    else:
        cached = controller_cache.get(arm_id)
        if cached is None:
            binding_path = bindings.binding_paths[arm_id]
            binding = load_e5_controller_binding(binding_path)
            controller = binding.assert_current()
            binding_sha = sha256_file(binding_path)
            controller_cache.clear()
            controller_cache[arm_id] = (binding, controller, binding_sha)
        else:
            binding, controller, binding_sha = cached
        controller_artifact_sha = binding.controller_artifact_sha256
        method_artifact_sha = controller_artifact_sha
        schema = controller.risk_probe.training_schema
        cube = runtime.prompt_feature_cube(
            rendered,
            layers=schema.layers,
            sites=schema.sites,
        )
        if type(cube) is not MlxPromptFeatureCubeOutput:
            raise FrozenArtifactError("E5 runtime returned a non-MLX prompt feature cube")
        features = _compose_e8_controller_features(schema, cube.activations)
        decision = controller.decide(features)
        if decision.class_labels != ("C", "I", "A"):
            raise DataValidationError("E5 adaptive labels differ from C/I/A")
        scores = {
            label: float(decision.probabilities[0, index])
            for index, label in enumerate(decision.class_labels)
        }
        probability_total = sum(scores.values())
        if not math.isfinite(probability_total) or probability_total <= 0:
            raise DataValidationError("E5 adaptive probabilities are invalid")
        scores = {label: value / probability_total for label, value in scores.items()}
        selected_layer = int(decision.selected_layers[0])
        selected_site, normalized_direction, direction_norm = _direction_choice(
            controller,
            selected_layer=selected_layer,
            directions=decision.directions,
        )
        standardized_alpha = float(decision.alphas[0])
        raw_alpha = standardized_alpha * direction_norm
        direction_sha = hashlib.sha256(normalized_direction.tobytes(order="C")).hexdigest()
        scope = _TOKEN_SCOPES[spec.intervention_timing]
        action = "intervene" if standardized_alpha > 0 else "release"
        if action == "intervene":
            state = runtime.standardized_intervention_state(
                normalized_direction,
                standardized_alpha=raw_alpha,
                reference_rms=1.0,
                token_scope=scope,
            )
            interventions[(selected_layer, selected_site)] = state
        feature_values = np.ascontiguousarray(
            features.detach().cpu().float().numpy(), dtype=np.float32
        )
        routing = np.ascontiguousarray(
            decision.routing_weights.detach().cpu().float().numpy(), dtype=np.float32
        )
        layer_routing: np.ndarray[Any, Any] | None = None
        if controller.layer_selector is not None:
            layer_routing = np.ascontiguousarray(
                controller.layer_selector.router.weights(features).detach().cpu().float().numpy(),
                dtype=np.float32,
            )
        feature_peak = cube.peak_memory_bytes
        feature_schema_digest = schema.digest
        feature_shape = list(feature_values.shape)
        feature_values_sha256 = hashlib.sha256(feature_values.tobytes(order="C")).hexdigest()
        maximum_probability = cube.maximum_token_probability
        output_entropy = cube.output_entropy
        routing_sha256 = hashlib.sha256(routing.tobytes(order="C")).hexdigest()
        routing_values = [float(value) for value in routing.reshape(-1)]
        layer_routing_values = (
            [float(value) for value in layer_routing.reshape(-1)]
            if layer_routing is not None
            else None
        )
    generated = runtime.generate_with_interventions(
        rendered,
        max_new_tokens=plan["max_new_tokens"],
        intervention_states=interventions,
    )
    elapsed = float(time.perf_counter() - started)
    if (
        type(generated) is not MlxGenerationOutput
        or generated.rendered_prompt != rendered
        or generated.peak_memory_bytes > plan["max_peak_memory_bytes"]
        or feature_peak > plan["max_peak_memory_bytes"]
    ):
        raise FrozenArtifactError("E5 runtime generation or memory evidence differs")
    indices = _token_indices(scope, generated.output_tokens) if action == "intervene" else []
    captured: np.ndarray[Any, Any] | None = None
    intervened: np.ndarray[Any, Any] | None = None
    delta: np.ndarray[Any, Any] | None = None
    intervention_norm = 0.0
    expected_norm = 0.0
    hook_delta_norms: list[float] = []
    hook_direction_dot_products: list[float] = []
    hook_residual_norms: list[float] = []
    if action == "intervene":
        if not indices or state is None:
            raise FrozenArtifactError("E5 intervention produced no hook application target")
        captured, intervened, delta = _strict_hook_arrays(
            state,
            direction=normalized_direction,
            raw_alpha=raw_alpha,
            expected_applications=len(indices),
        )
        intervention_norm = float(np.linalg.norm(delta.astype(np.float64)))
        expected_norm = abs(raw_alpha) * math.sqrt(len(indices))
        flattened = delta.reshape(len(indices), -1).astype(np.float64, copy=False)
        expected_delta = normalized_direction.astype(np.float64, copy=False) * raw_alpha
        hook_delta_norms = [float(np.linalg.norm(row)) for row in flattened]
        hook_direction_dot_products = [
            float(np.dot(row, normalized_direction.astype(np.float64, copy=False)))
            for row in flattened
        ]
        hook_residual_norms = [float(np.linalg.norm(row - expected_delta)) for row in flattened]
        if not math.isclose(intervention_norm, expected_norm, rel_tol=0.025, abs_tol=1e-6):
            raise FrozenArtifactError("E5 intervention norm differs from its exact edit")
    outcome = deterministic_short_answer_grade(generated.text, question.aliases)
    if outcome is Outcome.UNSCORABLE:
        raise DataValidationError("E5 native generation is unscorable")
    end_to_end = max(elapsed, float(generated.latency_seconds))
    decision_body = {
        "arm_id": arm_id,
        "prompt_id": prompt_id,
        "question_id": question.question_id,
        "rendered_prompt_sha256": rendered.sha256,
        "prompt_input_sha256": _e5_prompt_input_sha256(question, prompt_id),
        "controller_binding_sha256": binding_sha,
        "controller_artifact_sha256": controller_artifact_sha,
        "controller_scores": scores,
        "policy_action": action,
        "token_scope": scope.value,
        "applied_token_indices": indices,
        "activation_delta_norm": intervention_norm,
    }
    receipt = {
        "controller_binding_sha256": binding_sha,
        "controller_artifact_sha256": controller_artifact_sha,
        "controller_scores": scores,
        "policy_action": action,
        "applied_token_indices": indices,
        "activation_delta_norm": intervention_norm,
        "decision_digest": stable_hash(decision_body),
    }
    draft = E5AblationRecord(
        arm_id=arm_id,
        prompt_id=prompt_id,
        question_id=question.question_id,
        outcome=outcome,
        generation_latency_seconds=end_to_end,
        intervention_norm=intervention_norm,
        prompt_template_sha256=_PROMPT_HASHES[prompt_id],
        prompt_input_sha256=_e5_prompt_input_sha256(question, prompt_id),
        rendered_prompt_sha256=rendered.sha256,
        output_tokens=generated.output_tokens,
        controller_binding_sha256=binding_sha,
        token_scope=scope,
        execution_receipt=receipt,
        execution_receipt_digest=stable_hash(receipt),
        execution_receipt_signature="0" * 128,
    )
    record = replace(
        draft,
        execution_receipt_signature=sign_e5_ablation_execution_receipt(
            draft, private_key_hex=private_key_hex
        ),
    )
    evidence_body = {
        "schema_version": 1,
        "runtime_identity_sha256": plan["runtime_identity_sha256"],
        "method_artifact_sha256": method_artifact_sha,
        "rendered_prompt_token_ids_sha256": rendered.token_ids_sha256,
        "raw_output": generated.text,
        "raw_output_sha256": hashlib.sha256(generated.text.encode()).hexdigest(),
        "output_token_ids": list(generated.token_ids),
        "output_token_ids_sha256": _output_token_digest(generated.token_ids),
        "input_tokens": generated.input_tokens,
        "output_tokens": generated.output_tokens,
        "generation_latency_seconds": float(generated.latency_seconds),
        "end_to_end_latency_seconds": end_to_end,
        "feature_peak_memory_bytes": feature_peak,
        "generation_peak_memory_bytes": generated.peak_memory_bytes,
        "active_memory_bytes": generated.active_memory_bytes,
        "cache_memory_bytes": generated.cache_memory_bytes,
        "feature_schema_digest": feature_schema_digest,
        "feature_shape": feature_shape,
        "feature_values_sha256": feature_values_sha256,
        "maximum_token_probability": maximum_probability,
        "output_entropy": output_entropy,
        "selected_layer": selected_layer,
        "selected_site": selected_site.value,
        "standardized_alpha": standardized_alpha,
        "effective_raw_alpha": raw_alpha,
        "direction_sha256": direction_sha,
        "direction_norm": direction_norm,
        "routing_weights_sha256": routing_sha256,
        "routing_weights": routing_values,
        "layer_routing_weights": layer_routing_values,
        "hook_applications": len(indices),
        "applied_token_indices": indices,
        "pre_activation_sha256": (
            hashlib.sha256(captured.tobytes(order="C")).hexdigest()
            if captured is not None
            else None
        ),
        "post_activation_sha256": (
            hashlib.sha256(intervened.tobytes(order="C")).hexdigest()
            if intervened is not None
            else None
        ),
        "delta_sha256": (
            hashlib.sha256(delta.tobytes(order="C")).hexdigest() if delta is not None else None
        ),
        "hook_delta_norms": hook_delta_norms,
        "hook_direction_dot_products": hook_direction_dot_products,
        "hook_residual_norms": hook_residual_norms,
        "expected_activation_delta_norm": expected_norm,
    }
    evidence = {**evidence_body, "evidence_digest": stable_hash(evidence_body)}
    row_body = {"sequence": sequence, "record": record.to_dict(), "evidence": evidence}
    return {**row_body, "row_digest": stable_hash(row_body)}


def _lock_identity(path: Path) -> str:
    stat = path.stat()
    return stable_hash(
        {
            "path": str(path.resolve()),
            "device": stat.st_dev,
            "inode": stat.st_ino,
        }
    )


@contextmanager
def _exclusive_run_lock(directory: Path) -> Iterator[str]:
    path = directory / "run.lock"
    with path.open("r+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield _lock_identity(path)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _cleanup_stages(directory: Path) -> None:
    for value in (directory / "shards").iterdir():
        if _SHARD_STAGE.fullmatch(value.name):
            if value.is_dir() and not value.is_symlink():
                shutil.rmtree(value)
            else:
                value.unlink()
    for value in directory.iterdir():
        if value.name.startswith(".final.stage-"):
            if value.is_dir() and not value.is_symlink():
                shutil.rmtree(value)
            else:
                value.unlink()


def _append_native_shard(
    directory: Path,
    *,
    verified: VerifiedE5NativeRun,
    rows: Sequence[Mapping[str, Any]],
    private_key: Ed25519PrivateKey,
    session: Mapping[str, Any],
) -> VerifiedE5NativeRun:
    if not rows or len(rows) > verified.plan["shard_rows"]:
        raise DataValidationError("E5 native shard rows are empty or oversized")
    index = verified.shard_count
    destination = directory / "shards" / f"shard-{index:06d}"
    if destination.exists() or destination.is_symlink():
        raise FrozenArtifactError("refusing to overwrite an E5 native shard")
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=directory / "shards"))
    try:
        records_path = stage / "records.jsonl"
        records_path.write_text(
            "".join(json.dumps(dict(value), sort_keys=True) + "\n" for value in rows),
            encoding="utf-8",
        )
        maximum_peak = max(
            max(
                int(value["evidence"]["feature_peak_memory_bytes"]),
                int(value["evidence"]["generation_peak_memory_bytes"]),
            )
            for value in rows
        )
        body = {
            "schema_version": 2,
            "plan_identity": verified.plan["plan_identity"],
            "shard_index": index,
            "start_record": verified.records_completed,
            "end_record": verified.records_completed + len(rows),
            "record_count": len(rows),
            "records_sha256": sha256_file(records_path),
            "record_digests": [value["row_digest"] for value in rows],
            "previous_chain_head": verified.chain_head,
            "maximum_peak_memory_bytes": maximum_peak,
            "cumulative_maximum_peak_memory_bytes": max(
                verified.maximum_peak_memory_bytes, maximum_peak
            ),
            **dict(session),
        }
        signed = {**body, "manifest_digest": stable_hash(body)}
        signature = private_key.sign(canonical_json(signed).encode()).hex()
        manifest = {
            **signed,
            "signature": signature,
            "chain_head": stable_hash({"signed": signed, "signature": signature}),
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    completed = verified.records_completed + len(rows)
    complete = completed == verified.plan["expected_records"]
    return VerifiedE5NativeRun(
        directory=verified.directory,
        plan=verified.plan,
        records_completed=completed,
        shard_count=index + 1,
        chain_head=manifest["chain_head"],
        complete=complete,
        scientific_eligible=bool(verified.plan["scientific_eligible"] and complete),
        maximum_peak_memory_bytes=body["cumulative_maximum_peak_memory_bytes"],
        finalized_records=None,
        _source_context=verified._source_context,
    )


def _open_e5_native_resume_source(
    directory: Path,
    *,
    expected_execution_public_key: str,
) -> VerifiedE5NativeRun:
    """Authenticate the latest append head without replaying historical row payloads.

    This opener is only an execution checkpoint.  It cannot produce a finalized or
    scientifically consumable artifact; terminal finalization still performs the
    complete byte and semantic replay over every shard.
    """

    source = validate_active_study_artifact_paths(
        {"E5 native resume source": directory}
    )["E5 native resume source"]
    inventory = {value.name for value in source.iterdir()} if source.is_dir() else set()
    if (
        source.is_symlink()
        or not source.is_dir()
        or inventory != _INVENTORY
        or any((source / name).is_symlink() for name in inventory)
    ):
        raise FrozenArtifactError("E5 resume source inventory differs or is terminal")
    plan = _read_plan(source)
    if plan["execution_public_key"] != expected_execution_public_key:
        raise FrozenArtifactError("E5 resume execution key differs from external trust root")
    screen, bindings, policy, direction = _open_resume_sources(plan)
    protocol = E5Protocol.from_dict(plan["protocol"])
    specs = build_e5_ablation_grid(protocol)
    context = _E5NativeSourceContext(
        screen=screen,
        bindings=bindings,
        m1_policy=policy,
        m1_direction=direction,
        protocol=protocol,
        specs=specs,
    )
    shards = _shard_directories(source)
    for shard in shards:
        if {value.name for value in shard.iterdir()} != {"manifest.json", "records.jsonl"}:
            raise FrozenArtifactError("E5 resume shard file inventory differs")
        if any(value.is_symlink() or not value.is_file() for value in shard.iterdir()):
            raise FrozenArtifactError("E5 resume shard contains a non-regular file")
    if not shards:
        return VerifiedE5NativeRun(
            directory=source.resolve(),
            plan=MappingProxyType(plan),
            records_completed=0,
            shard_count=0,
            chain_head=None,
            complete=False,
            scientific_eligible=False,
            maximum_peak_memory_bytes=0,
            finalized_records=None,
            _source_context=context,
        )
    manifest_path = shards[-1] / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read E5 resume manifest: {exc}") from exc
    if not isinstance(manifest, dict) or manifest_path.read_text(encoding="utf-8") != (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ):
        raise FrozenArtifactError("E5 resume manifest is not canonical JSON")
    body = dict(manifest)
    signature = body.pop("signature", None)
    chain_head = body.pop("chain_head", None)
    digest = body.get("manifest_digest")
    unsigned = dict(body)
    unsigned.pop("manifest_digest", None)
    expected_keys = {
        "schema_version",
        "plan_identity",
        "shard_index",
        "start_record",
        "end_record",
        "record_count",
        "records_sha256",
        "record_digests",
        "previous_chain_head",
        "maximum_peak_memory_bytes",
        "cumulative_maximum_peak_memory_bytes",
        "execution_session_id",
        "execution_session_started_at",
        "execution_session_wall_seconds",
        "execution_lock_identity",
        "execution_process_id",
        "manifest_digest",
    }
    completed = body.get("end_record")
    maximum_peak = body.get("maximum_peak_memory_bytes")
    cumulative_peak = body.get("cumulative_maximum_peak_memory_bytes")
    if (
        set(body) != expected_keys
        or digest != stable_hash(unsigned)
        or chain_head != stable_hash({"signed": body, "signature": signature})
        or body["schema_version"] != 2
        or body["plan_identity"] != plan["plan_identity"]
        or body["shard_index"] != len(shards) - 1
        or type(body["start_record"]) is not int
        or type(body["record_count"]) is not int
        or body["record_count"] <= 0
        or body["record_count"] > plan["shard_rows"]
        or completed != body["start_record"] + body["record_count"]
        or type(completed) is not int
        or completed > plan["expected_records"]
        or type(body["record_digests"]) is not list
        or len(body["record_digests"]) != body["record_count"]
        or any(
            type(value) is not str or _SHA256.fullmatch(value) is None
            for value in body["record_digests"]
        )
        or _SHA256.fullmatch(str(body["records_sha256"])) is None
        or type(maximum_peak) is not int
        or type(cumulative_peak) is not int
        or not 0 <= maximum_peak <= cumulative_peak <= plan["max_peak_memory_bytes"]
        or _SHA256.fullmatch(str(body["execution_session_id"])) is None
        or not _utc_timestamp(body["execution_session_started_at"])
        or isinstance(body["execution_session_wall_seconds"], bool)
        or not isinstance(body["execution_session_wall_seconds"], int | float)
        or not math.isfinite(float(body["execution_session_wall_seconds"]))
        or float(body["execution_session_wall_seconds"]) < 0
        or body["execution_lock_identity"] != _lock_identity(source / "run.lock")
        or type(body["execution_process_id"]) is not int
        or body["execution_process_id"] <= 0
    ):
        raise FrozenArtifactError("E5 resume checkpoint manifest differs")
    _verify_signature(
        manifest,
        public_key_hex=plan["execution_public_key"],
        context="E5 resume checkpoint",
    )
    complete = completed == plan["expected_records"]
    return VerifiedE5NativeRun(
        directory=source.resolve(),
        plan=MappingProxyType(plan),
        records_completed=completed,
        shard_count=len(shards),
        chain_head=chain_head,
        complete=complete,
        scientific_eligible=bool(plan["scientific_eligible"] and complete),
        maximum_peak_memory_bytes=cumulative_peak,
        finalized_records=None,
        _source_context=context,
    )


def run_e5_native_ablation(
    directory: str | Path,
    *,
    runtime: E5NativeRuntime,
    execution_private_key_hex: str,
    request_budget: int | None = None,
) -> VerifiedE5NativeRun:
    """Run or resume from a signed append head without historical row replay."""

    if request_budget is not None and (type(request_budget) is not int or request_budget <= 0):
        raise ConfigurationError("E5 native request budget must be positive")
    source = Path(directory)
    private = _private_key(execution_private_key_hex)
    public = e5_native_execution_public_key(execution_private_key_hex)
    with _exclusive_run_lock(source) as lock_identity:
        _cleanup_stages(source)
        verified = _open_e5_native_resume_source(
            source,
            expected_execution_public_key=public,
        )
        if verified.finalized_records is not None:
            raise FrozenArtifactError("E5 native run is already finalized")
        if verified.complete:
            return verified
        if _exact_json(runtime.runtime_identity(), context="E5 live runtime identity") != dict(
            verified.plan["runtime_identity"]
        ):
            raise FrozenArtifactError("live E5 runtime differs from its frozen identity")
        context = verified._source_context
        screen = context.screen
        bindings = context.bindings
        policy = context.m1_policy
        direction = context.m1_direction
        specs = context.specs
        session_started_at = datetime.now(UTC).isoformat()
        session_started = time.perf_counter()
        session_id = stable_hash(
            {
                "plan_identity": verified.plan["plan_identity"],
                "start_record": verified.records_completed,
                "started_at": session_started_at,
                "process_id": os.getpid(),
                "lock_identity": lock_identity,
            }
        )
        rows: list[dict[str, Any]] = []
        controller_cache: dict[str, tuple[E5ControllerBinding, AdaptiveController, str]] = {}
        handled = 0
        while verified.records_completed + len(rows) < verified.plan["expected_records"]:
            if request_budget is not None and handled >= request_budget:
                break
            sequence = verified.records_completed + len(rows)
            rows.append(
                _execute_native_row(
                    sequence=sequence,
                    plan=verified.plan,
                    specs=specs,
                    questions=screen.dev_questions,
                    runtime=runtime,
                    bindings=bindings,
                    m1_policy=policy,
                    m1_direction=direction,
                    private_key_hex=execution_private_key_hex,
                    controller_cache=controller_cache,
                )
            )
            handled += 1
            if len(rows) == verified.plan["shard_rows"]:
                verified = _append_native_shard(
                    source,
                    verified=verified,
                    rows=rows,
                    private_key=private,
                    session={
                        "execution_session_id": session_id,
                        "execution_session_started_at": session_started_at,
                        "execution_session_wall_seconds": float(
                            time.perf_counter() - session_started
                        ),
                        "execution_lock_identity": lock_identity,
                        "execution_process_id": os.getpid(),
                    },
                )
                rows.clear()
        if rows:
            verified = _append_native_shard(
                source,
                verified=verified,
                rows=rows,
                private_key=private,
                session={
                    "execution_session_id": session_id,
                    "execution_session_started_at": session_started_at,
                    "execution_session_wall_seconds": float(time.perf_counter() - session_started),
                    "execution_lock_identity": lock_identity,
                    "execution_process_id": os.getpid(),
                },
            )
        return verified


def _iter_ablation_records(directory: Path) -> Iterator[E5AblationRecord]:
    for shard in _shard_directories(directory):
        manifest = json.loads((shard / "manifest.json").read_text(encoding="utf-8"))
        for value in _read_jsonl(
            shard / "records.jsonl", expected_sha256=manifest["records_sha256"]
        ):
            yield E5AblationRecord.from_dict(value["record"])


def finalize_e5_native_ablation(
    directory: str | Path,
    *,
    execution_private_key_hex: str,
) -> VerifiedE5NativeRun:
    """Atomically materialize the exact selection input plus its signed receipt."""

    source = Path(directory)
    private = _private_key(execution_private_key_hex)
    public = e5_native_execution_public_key(execution_private_key_hex)
    with _exclusive_run_lock(source):
        _cleanup_stages(source)
        verified = verify_e5_native_ablation(
            source,
            expected_execution_public_key=public,
            require_complete=True,
            semantic=True,
        )
        if verified.finalized_records is not None:
            return verified
        screen = load_e4_screen_receipt(verified.plan["screen_receipt_path"])
        protocol = E5Protocol.from_dict(verified.plan["protocol"])
        stage = Path(tempfile.mkdtemp(prefix=".final.stage-", dir=source))
        try:
            records_path = stage / "records.jsonl"
            write_e5_ablation_records(
                records_path,
                _iter_ablation_records(source),
                screen=screen,
                protocol=protocol,
            )
            assert verified.chain_head is not None
            body = {
                "schema_version": 1,
                "plan_identity": verified.plan["plan_identity"],
                "record_count": verified.records_completed,
                "source_chain_head": verified.chain_head,
                "records_relative_path": "records.jsonl",
                "records_sha256": sha256_file(records_path),
                "controller_bindings_sha256": verified.plan["controller_bindings_sha256"],
                "screen_receipt_sha256": verified.plan["screen_receipt_sha256"],
                "finalized_at": datetime.now(UTC).isoformat(),
            }
            signed = {**body, "receipt_digest": stable_hash(body)}
            signature = private.sign(canonical_json(signed).encode()).hex()
            receipt = {
                **signed,
                "signature": signature,
                "chain_head": stable_hash({"signed": signed, "signature": signature}),
            }
            (stage / "receipt.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(stage, source / "final")
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    return replace(
        verified,
        finalized_records=(source / "final" / "records.jsonl").resolve(),
    )

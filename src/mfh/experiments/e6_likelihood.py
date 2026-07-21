"""Teacher-forced E6 gold/abstention likelihood evidence and answer ranking."""

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
from typing import Any, Protocol

import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from mfh.contracts import (
    ActivationSite,
    GenerationRecord,
    Outcome,
    PromptSpec,
    Question,
    TokenScope,
)
from mfh.data.io import read_questions, write_questions
from mfh.data.normalization import normalize_answer
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.evaluation.grading import deterministic_short_answer_grade
from mfh.experiments.e3_schedule import E3Protocol
from mfh.experiments.e6_grading import (
    load_e6_official_grader_bundle,
    verify_e6_factual_grade,
)
from mfh.experiments.gates import write_gate_evidence
from mfh.experiments.model_selection import validate_active_study_artifact_paths
from mfh.experiments.protocol import ExperimentPhase, StudyProtocol
from mfh.experiments.runner import (
    EvaluationCondition,
    PhaseCompletion,
    PhaseFalsification,
    PhaseRunLedger,
    _copy_frozen_artifact,
    _validate_question_bundle,
)
from mfh.experiments.runtime_evidence import (
    build_generation_runtime_metrics,
    validate_generation_runtime_metrics,
)
from mfh.inference.mlx_research import (
    MlxResearchInterventionState,
    MlxResearchRuntime,
    MlxTeacherForcedOutput,
)
from mfh.inference.mlx_runtime import (
    MlxGenerationOutput,
    MlxInterventionState,
    MlxRenderedPrompt,
)
from mfh.provenance import canonical_json, sha256_file, sha256_path, stable_hash

_PROMPTS = frozenset({"P0-neutral", "P2-calibrated-abstention", "P3-forced-answer"})
_METHODS = frozenset({"M0", "M1", "M3"})
_ABSTENTION = "I don't know."
_ALTERNATIVES_KEY = "e6_plausible_alternatives"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_QUESTION_COUNTS = {
    "triviaqa": 5_000,
    "simpleqa_verified": 1_000,
    "aa_omniscience_public_600": 600,
}


class E6RuntimeAttestor:
    """Runtime-owned E6 signer bound to one concrete MLX runtime identity."""

    __slots__ = ("_artifact", "_private_key", "execution_public_key", "runtime")

    def __init__(self, runtime: MlxResearchRuntime, *, execution_private_key: str) -> None:
        if type(runtime) is not MlxResearchRuntime or _SHA256.fullmatch(
            execution_private_key
        ) is None:
            raise DataValidationError("E6 attestor requires a concrete MLX runtime and key")
        try:
            private_key = Ed25519PrivateKey.from_private_bytes(
                bytes.fromhex(execution_private_key)
            )
            identity = json.loads(
                json.dumps(dict(runtime.runtime_identity()), sort_keys=True, allow_nan=False)
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DataValidationError(f"invalid E6 runtime attestor: {exc}") from exc
        self.runtime = runtime
        self._private_key = private_key
        self.execution_public_key = private_key.public_key().public_bytes(
            Encoding.Raw, PublicFormat.Raw
        ).hex()
        _validate_e6_runtime_identity(identity)
        body = {
            "schema_version": 1,
            "execution_public_key": self.execution_public_key,
            "runtime_identity": identity,
            "runtime_identity_digest": stable_hash(identity),
        }
        self._artifact = MappingProxyType(
            {**body, "runtime_attestation_digest": stable_hash(body)}
        )

    def write_runtime_artifact(self, path: str | Path) -> str:
        destination = validate_active_study_artifact_paths(
            {"E6 runtime attestation": path}
        )["E6 runtime attestation"]
        if destination.exists() or destination.is_symlink():
            raise FrozenArtifactError(
                f"refusing to overwrite E6 runtime attestation: {destination}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(dict(self._artifact), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return self.verify_runtime_artifact(destination)

    def verify_runtime_artifact(self, path: str | Path) -> str:
        source = Path(path)
        self.assert_live_runtime(self.runtime)
        value = _load_e6_runtime_attestation(source)
        if value != dict(self._artifact):
            raise FrozenArtifactError("E6 runtime attestation differs from the executing runtime")
        return sha256_file(source)

    def assert_live_runtime(self, runtime: MlxResearchRuntime) -> str:
        """Re-attest the exact live runtime object before trusted execution or signing."""

        if type(runtime) is not MlxResearchRuntime or runtime is not self.runtime:
            raise FrozenArtifactError("E6 attestor runtime object was replaced")
        try:
            identity = json.loads(
                json.dumps(dict(runtime.runtime_identity()), sort_keys=True, allow_nan=False)
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise FrozenArtifactError(f"cannot re-attest live MLX runtime: {exc}") from exc
        _validate_e6_runtime_identity(identity)
        expected = self._artifact.get("runtime_identity")
        if identity != expected:
            raise FrozenArtifactError("live MLX runtime identity changed after attestation")
        return stable_hash(identity)

    @property
    def attested_runtime_identity(self) -> Mapping[str, Any]:
        """Return the immutable identity whose private key authorizes execution."""

        identity = self._artifact.get("runtime_identity")
        if not isinstance(identity, Mapping):
            raise FrozenArtifactError("E6 attestor lacks its runtime identity")
        return MappingProxyType(dict(identity))

    def _sign(self, body: Mapping[str, Any]) -> str:
        if self._artifact:
            self.assert_live_runtime(self.runtime)
        return self._private_key.sign(canonical_json(body).encode("utf-8")).hex()


def _load_e6_runtime_attestation(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise FrozenArtifactError("E6 runtime attestation must be one regular file")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read E6 runtime attestation: {exc}") from exc
    expected_keys = {
        "schema_version",
        "execution_public_key",
        "runtime_identity",
        "runtime_identity_digest",
        "runtime_attestation_digest",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise FrozenArtifactError("E6 runtime attestation schema differs")
    body = dict(value)
    attestation_digest = body.pop("runtime_attestation_digest")
    identity = body.get("runtime_identity")
    if (
        body.get("schema_version") != 1
        or type(body.get("execution_public_key")) is not str
        or _SHA256.fullmatch(body["execution_public_key"]) is None
        or not isinstance(identity, dict)
        or not identity
        or body.get("runtime_identity_digest") != stable_hash(identity)
        or attestation_digest != stable_hash(body)
    ):
        raise FrozenArtifactError("E6 runtime attestation identity differs")
    try:
        _validate_e6_runtime_identity(identity)
    except DataValidationError as exc:
        raise FrozenArtifactError(f"E6 runtime identity is invalid: {exc}") from exc
    return value


def _validate_e6_runtime_identity(identity: Mapping[str, Any]) -> None:
    expected_keys = {
        "backend",
        "mlx",
        "mlx_lm",
        "python",
        "machine_model",
        "chip",
        "unified_memory_bytes",
        "physical_cpu_cores",
        "architecture",
        "os",
        "os_build",
        "model_class",
        "tokenizer_class",
        "num_layers",
        "seed",
        "model_repository",
        "model_revision",
        "model_quantization",
        "model_num_layers",
        "snapshot_sha256",
        "research_provenance",
        "research_toolchain",
    }
    text_keys = expected_keys - {
        "unified_memory_bytes",
        "physical_cpu_cores",
        "num_layers",
        "seed",
        "model_num_layers",
        "research_provenance",
        "research_toolchain",
    }
    provenance = identity.get("research_provenance")
    toolchain = identity.get("research_toolchain")
    if (
        set(identity) != expected_keys
        or identity.get("backend") != "mlx"
        or any(type(identity.get(key)) is not str or not identity[key] for key in text_keys)
        or _SHA256.fullmatch(str(identity.get("snapshot_sha256"))) is None
        or any(
            type(identity.get(key)) is not int or identity[key] <= 0
            for key in (
                "unified_memory_bytes",
                "physical_cpu_cores",
                "num_layers",
                "model_num_layers",
            )
        )
        or type(identity.get("seed")) is not int
        or identity.get("num_layers") != identity.get("model_num_layers")
        or not isinstance(provenance, dict)
        or not provenance
        or not isinstance(toolchain, dict)
        or set(toolchain) != {"xcodebuild", "metal_compiler"}
        or any(type(value) is not str or not value for value in toolchain.values())
    ):
        raise DataValidationError("E6 requires the full MLX model and toolchain identity")


def _assert_e6_runtime_condition(
    attestation: Mapping[str, Any],
    *,
    model_repository: str,
    model_revision: str,
    quantization: str,
    model_num_layers: int,
    seed: int,
    execution_public_key: str,
) -> None:
    identity = attestation["runtime_identity"]
    if (
        attestation["execution_public_key"] != execution_public_key
        or identity["model_repository"] != model_repository
        or identity["model_revision"] != model_revision
        or identity["model_quantization"] != quantization
        or identity["model_num_layers"] != model_num_layers
        or identity["seed"] != seed
    ):
        raise FrozenArtifactError("E6 runtime attestation differs from condition identity")


class E6LikelihoodRuntime(Protocol):
    def render_prompt(
        self,
        prompt: PromptSpec,
        question: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> MlxRenderedPrompt: ...

    def teacher_forced_continuation(
        self,
        rendered: MlxRenderedPrompt,
        response: str,
        *,
        layers: Sequence[int],
        site: ActivationSite,
        intervention_states: Mapping[int, MlxInterventionState] | None = None,
    ) -> MlxTeacherForcedOutput: ...


@dataclass(frozen=True, slots=True)
class E6ScoredResponse:
    text_sha256: str
    token_ids_sha256: str
    token_count: int
    total_log_likelihood: float
    mean_log_likelihood: float
    execution_receipt: Mapping[str, Any]
    execution_receipt_digest: str

    def __post_init__(self) -> None:
        if (
            any(
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in (self.text_sha256, self.token_ids_sha256)
            )
            or type(self.token_count) is not int
            or self.token_count <= 0
            or type(self.total_log_likelihood) is not float
            or type(self.mean_log_likelihood) is not float
            or not math.isfinite(self.total_log_likelihood)
            or not math.isfinite(self.mean_log_likelihood)
            or self.total_log_likelihood > 0
            or self.mean_log_likelihood > 0
            or not isinstance(self.execution_receipt, Mapping)
            or _SHA256.fullmatch(self.execution_receipt_digest) is None
            or self.execution_receipt_digest
            != stable_hash(dict(self.execution_receipt))
            or not math.isclose(
                self.mean_log_likelihood,
                self.total_log_likelihood / self.token_count,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise DataValidationError("E6 scored response is invalid")
        receipt = _validate_runtime_receipt(self.execution_receipt)
        if receipt["response_text_sha256"] != self.text_sha256:
            raise DataValidationError("E6 scored response receipt differs from its text")
        object.__setattr__(self, "execution_receipt", MappingProxyType(receipt))

    def to_dict(self) -> dict[str, Any]:
        return {
            "text_sha256": self.text_sha256,
            "token_ids_sha256": self.token_ids_sha256,
            "token_count": self.token_count,
            "total_log_likelihood": self.total_log_likelihood,
            "mean_log_likelihood": self.mean_log_likelihood,
            "execution_receipt": dict(self.execution_receipt),
            "execution_receipt_digest": self.execution_receipt_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> E6ScoredResponse:
        if set(value) != {
            "text_sha256",
            "token_ids_sha256",
            "token_count",
            "total_log_likelihood",
            "mean_log_likelihood",
            "execution_receipt",
            "execution_receipt_digest",
        }:
            raise DataValidationError("E6 scored-response keys differ")
        return cls(
            text_sha256=value["text_sha256"],
            token_ids_sha256=value["token_ids_sha256"],
            token_count=value["token_count"],
            total_log_likelihood=value["total_log_likelihood"],
            mean_log_likelihood=value["mean_log_likelihood"],
            execution_receipt=value["execution_receipt"],
            execution_receipt_digest=value["execution_receipt_digest"],
        )


def _direction_identity(direction: Any) -> tuple[str, float]:
    try:
        values = np.asarray(direction, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"E6 intervention direction is invalid: {exc}") from exc
    norm = float(np.linalg.norm(values)) if values.ndim == 1 else math.nan
    if (
        values.ndim != 1
        or values.size == 0
        or not np.isfinite(values).all()
        or not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-6)
    ):
        raise DataValidationError("E6 intervention direction must be a finite unit vector")
    frozen = np.ascontiguousarray(values, dtype=np.float32)
    return hashlib.sha256(frozen.tobytes(order="C")).hexdigest(), norm


def _state_spec(
    states: Mapping[int, MlxInterventionState] | None,
    *,
    layers: tuple[int, ...],
    site: ActivationSite,
) -> dict[str, Any]:
    frozen_states: list[dict[str, Any]] = []
    for layer, state in sorted(dict(states or {}).items()):
        if (
            type(layer) is not int
            or layer not in layers
            or not isinstance(state, MlxResearchInterventionState)
            or state.direction is None
            or isinstance(state.alpha, bool)
            or not isinstance(state.alpha, int | float)
            or not math.isfinite(float(state.alpha))
            or float(state.alpha) == 0
            or not isinstance(state.token_scope, TokenScope)
            or isinstance(state.decay, bool)
            or not isinstance(state.decay, int | float)
            or not math.isfinite(float(state.decay))
            or float(state.decay) < 0
            or (
                state.token_scope is TokenScope.EXPONENTIAL_DECAY
                and float(state.decay) <= 0
            )
            or state.generated_calls != 0
            or state.applications != 0
            or state.captured is not None
            or state.intervened is not None
            or state.phase_armed
            or state.prompt_tokens_remaining != 0
        ):
            raise DataValidationError("E6 intervention state is not a fresh material MLX state")
        direction_sha256, direction_norm = _direction_identity(state.direction)
        frozen_states.append(
            {
                "layer": layer,
                "direction_sha256": direction_sha256,
                "direction_norm": direction_norm,
                "alpha": float(state.alpha),
                "token_scope": state.token_scope.value,
                "decay": float(state.decay),
            }
        )
    body = {
        "schema_version": 1,
        "layers": list(layers),
        "site": site.value,
        "states": frozen_states,
    }
    return {**body, "state_spec_digest": stable_hash(body)}


def _runtime_receipt(
    *,
    rendered: MlxRenderedPrompt,
    response_text_sha256: str,
    output: MlxTeacherForcedOutput,
    states: Mapping[int, MlxInterventionState] | None,
    state_spec: Mapping[str, Any],
) -> dict[str, Any]:
    executions: list[dict[str, Any]] = []
    for state_item in state_spec["states"]:
        layer = state_item["layer"]
        state = dict(states or {})[layer]
        assert isinstance(state, MlxResearchInterventionState)
        if (
            not state.phase_armed
            or state.prompt_tokens_remaining != 0
            or state.applications <= 0
            or state.intervened is None
        ):
            raise FrozenArtifactError(
                "E6 runtime did not execute the declared material intervention"
            )
        executions.append(
            {
                **state_item,
                "hook_applications": state.applications,
                "generated_calls": state.generated_calls,
            }
        )
    body = {
        "schema_version": 1,
        "rendered_prompt_sha256": rendered.sha256,
        "response_text_sha256": response_text_sha256,
        "response_token_ids_sha256": output.response_token_ids_sha256,
        "response_token_count": len(output.response_token_ids),
        "layers": list(state_spec["layers"]),
        "site": state_spec["site"],
        "states": executions,
        "state_spec_digest": state_spec["state_spec_digest"],
    }
    return body


def _validate_runtime_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version",
        "rendered_prompt_sha256",
        "response_text_sha256",
        "response_token_ids_sha256",
        "response_token_count",
        "layers",
        "site",
        "states",
        "state_spec_digest",
    }
    body = dict(value)
    layers = body.get("layers")
    states = body.get("states")
    if (
        set(body) != expected
        or body.get("schema_version") != 1
        or any(
            type(body.get(name)) is not str or _SHA256.fullmatch(body[name]) is None
            for name in (
                "rendered_prompt_sha256",
                "response_text_sha256",
                "response_token_ids_sha256",
                "state_spec_digest",
            )
        )
        or type(body.get("response_token_count")) is not int
        or body["response_token_count"] <= 0
        or type(layers) is not list
        or not layers
        or any(type(layer) is not int or layer < 0 for layer in layers)
        or len(set(layers)) != len(layers)
        or body.get("site") not in {item.value for item in ActivationSite}
        or type(states) is not list
    ):
        raise DataValidationError("E6 runtime execution receipt is invalid")
    frozen_states: list[dict[str, Any]] = []
    for item in states:
        if not isinstance(item, Mapping) or set(item) != {
            "layer",
            "direction_sha256",
            "direction_norm",
            "alpha",
            "token_scope",
            "decay",
            "hook_applications",
            "generated_calls",
        }:
            raise DataValidationError("E6 runtime intervention receipt is invalid")
        frozen = dict(item)
        if (
            type(frozen["layer"]) is not int
            or frozen["layer"] not in layers
            or _SHA256.fullmatch(frozen["direction_sha256"]) is None
            or any(
                isinstance(frozen[name], bool)
                or not isinstance(frozen[name], int | float)
                or not math.isfinite(float(frozen[name]))
                for name in ("direction_norm", "alpha", "decay")
            )
            or not math.isclose(
                float(frozen["direction_norm"]), 1.0, rel_tol=1e-5, abs_tol=1e-6
            )
            or float(frozen["alpha"]) == 0
            or float(frozen["decay"]) < 0
            or frozen["token_scope"] not in {item.value for item in TokenScope}
            or type(frozen["hook_applications"]) is not int
            or frozen["hook_applications"] <= 0
            or type(frozen["generated_calls"]) is not int
            or frozen["generated_calls"] < 0
        ):
            raise DataValidationError("E6 runtime intervention receipt is invalid")
        frozen_states.append(frozen)
    if len({item["layer"] for item in frozen_states}) != len(frozen_states):
        raise DataValidationError("E6 runtime receipt repeats an intervention layer")
    spec_body = {
        "schema_version": 1,
        "layers": layers,
        "site": body["site"],
        "states": [
            {
                name: item[name]
                for name in item
                if name not in {"hook_applications", "generated_calls"}
            }
            for item in frozen_states
        ],
    }
    if body["state_spec_digest"] != stable_hash(spec_body):
        raise DataValidationError("E6 runtime state-spec digest differs")
    body["layers"] = list(layers)
    body["states"] = frozen_states
    return body


def _score(
    runtime: E6LikelihoodRuntime,
    rendered: MlxRenderedPrompt,
    response: str,
    *,
    layers: tuple[int, ...],
    site: ActivationSite,
    state_factory: Callable[[], Mapping[int, MlxInterventionState]] | None,
) -> E6ScoredResponse:
    states = dict(state_factory()) if state_factory is not None else None
    state_spec = _state_spec(states, layers=layers, site=site)
    output = runtime.teacher_forced_continuation(
        rendered,
        response,
        layers=layers,
        site=site,
        intervention_states=states,
    )
    text_sha = hashlib.sha256(response.encode("utf-8")).hexdigest()
    if (
        type(output) is not MlxTeacherForcedOutput
        or output.response_text_sha256 != text_sha
        or tuple(output.activations) != layers
    ):
        raise FrozenArtifactError("E6 runtime scored a different response")
    receipt = _runtime_receipt(
        rendered=rendered,
        response_text_sha256=text_sha,
        output=output,
        states=states,
        state_spec=state_spec,
    )
    return E6ScoredResponse(
        text_sha256=text_sha,
        token_ids_sha256=output.response_token_ids_sha256,
        token_count=len(output.response_token_ids),
        total_log_likelihood=-float(output.negative_log_likelihood),
        mean_log_likelihood=-float(output.mean_negative_log_likelihood),
        execution_receipt=receipt,
        execution_receipt_digest=stable_hash(receipt),
    )


@dataclass(frozen=True, slots=True)
class E6LikelihoodRecord:
    condition_id: str
    question_id: str
    method: str
    prompt_id: str
    rendered_prompt_sha256: str
    aliases_digest: str
    alternative_text_sha256s: tuple[str, ...]
    alias_scores: tuple[E6ScoredResponse, ...]
    best_alias_index: int
    abstention_score: E6ScoredResponse
    alternative_scores: tuple[E6ScoredResponse, ...]
    gold_rank: int | None
    record_digest: str

    def __post_init__(self) -> None:
        if (
            any(
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in (
                    self.condition_id,
                    self.rendered_prompt_sha256,
                    self.aliases_digest,
                    self.record_digest,
                )
            )
            or type(self.question_id) is not str
            or not self.question_id
            or self.method not in _METHODS
            or self.prompt_id not in _PROMPTS
            or type(self.alias_scores) is not tuple
            or not self.alias_scores
            or any(type(value) is not E6ScoredResponse for value in self.alias_scores)
            or not 0 <= self.best_alias_index < len(self.alias_scores)
            or type(self.alternative_scores) is not tuple
            or type(self.alternative_text_sha256s) is not tuple
            or len(self.alternative_text_sha256s) != len(self.alternative_scores)
            or any(_SHA256.fullmatch(value) is None for value in self.alternative_text_sha256s)
            or any(
                type(value) is not E6ScoredResponse
                for value in self.alternative_scores
            )
            or type(self.abstention_score) is not E6ScoredResponse
            or (
                self.gold_rank is not None
                and (
                    type(self.gold_rank) is not int
                    or not 1 <= self.gold_rank <= len(self.alternative_scores) + 1
                )
            )
            or (not self.alternative_scores) != (self.gold_rank is None)
        ):
            raise DataValidationError("E6 likelihood record is invalid")
        if self.best_alias_index != min(
            range(len(self.alias_scores)),
            key=lambda index: (-self.alias_scores[index].mean_log_likelihood, index),
        ):
            raise DataValidationError("E6 best accepted alias differs")
        receipts = (*self.alias_scores, self.abstention_score, *self.alternative_scores)
        if (
            len({value.execution_receipt["state_spec_digest"] for value in receipts}) != 1
            or any(
                value.execution_receipt["rendered_prompt_sha256"]
                != self.rendered_prompt_sha256
                for value in receipts
            )
        ):
            raise DataValidationError("E6 response executions differ within a question")
        if self.record_digest != stable_hash(self._body()):
            raise DataValidationError("E6 likelihood record digest differs")

    @property
    def gold_log_likelihood(self) -> float:
        return self.alias_scores[self.best_alias_index].mean_log_likelihood

    @property
    def abstention_log_likelihood(self) -> float:
        return self.abstention_score.mean_log_likelihood

    @property
    def intervention_state_digest(self) -> str:
        return str(self.alias_scores[0].execution_receipt["state_spec_digest"])

    def _body(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "question_id": self.question_id,
            "method": self.method,
            "prompt_id": self.prompt_id,
            "rendered_prompt_sha256": self.rendered_prompt_sha256,
            "aliases_digest": self.aliases_digest,
            "alternative_text_sha256s": list(self.alternative_text_sha256s),
            "alias_scores": [value.to_dict() for value in self.alias_scores],
            "best_alias_index": self.best_alias_index,
            "abstention_score": self.abstention_score.to_dict(),
            "alternative_scores": [value.to_dict() for value in self.alternative_scores],
            "gold_rank": self.gold_rank,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "record_digest": self.record_digest}

    def generation_metadata(self) -> Mapping[str, float | int]:
        values: dict[str, float | int] = {
            "gold_alias_log_likelihood": self.gold_log_likelihood,
            "abstention_log_likelihood": self.abstention_log_likelihood,
        }
        if self.gold_rank is not None:
            values["gold_answer_rank"] = self.gold_rank
        return MappingProxyType(values)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> E6LikelihoodRecord:
        if set(value) != {
            "condition_id",
            "question_id",
            "method",
            "prompt_id",
            "rendered_prompt_sha256",
            "aliases_digest",
            "alternative_text_sha256s",
            "alias_scores",
            "best_alias_index",
            "abstention_score",
            "alternative_scores",
            "gold_rank",
            "record_digest",
        }:
            raise DataValidationError("E6 likelihood-record keys differ")
        alias_scores = value["alias_scores"]
        alternative_scores = value["alternative_scores"]
        alternative_text_sha256s = value["alternative_text_sha256s"]
        abstention = value["abstention_score"]
        if (
            type(alias_scores) is not list
            or type(alternative_scores) is not list
            or type(alternative_text_sha256s) is not list
            or not isinstance(abstention, Mapping)
        ):
            raise DataValidationError("E6 likelihood score collections are invalid")
        return cls(
            condition_id=value["condition_id"],
            question_id=value["question_id"],
            method=value["method"],
            prompt_id=value["prompt_id"],
            rendered_prompt_sha256=value["rendered_prompt_sha256"],
            aliases_digest=value["aliases_digest"],
            alternative_text_sha256s=tuple(alternative_text_sha256s),
            alias_scores=tuple(E6ScoredResponse.from_dict(item) for item in alias_scores),
            best_alias_index=value["best_alias_index"],
            abstention_score=E6ScoredResponse.from_dict(abstention),
            alternative_scores=tuple(
                E6ScoredResponse.from_dict(item) for item in alternative_scores
            ),
            gold_rank=value["gold_rank"],
            record_digest=value["record_digest"],
        )


def score_e6_question(
    *,
    runtime: E6LikelihoodRuntime,
    question: Question,
    prompt: PromptSpec,
    method: str,
    condition_id: str,
    layers: Sequence[int],
    site: ActivationSite,
    state_factory: Callable[[], Mapping[int, MlxInterventionState]] | None = None,
) -> E6LikelihoodRecord:
    frozen_layers = tuple(layers)
    raw_alternatives = question.metadata.get(_ALTERNATIVES_KEY, ())
    alternatives = tuple(raw_alternatives) if isinstance(raw_alternatives, list | tuple) else ()
    if (
        type(condition_id) is not str
        or len(condition_id) != 64
        or any(character not in "0123456789abcdef" for character in condition_id)
        or question.benchmark
        not in {"triviaqa", "simpleqa_verified", "aa_omniscience_public_600"}
        or prompt.prompt_id not in _PROMPTS
        or method not in _METHODS
        or (method == "M0" and state_factory is not None)
        or (method == "M1" and state_factory is None)
        or not frozen_layers
        or any(type(value) is not int for value in frozen_layers)
        or len(set(frozen_layers)) != len(frozen_layers)
        or not question.aliases
        or (
            _ALTERNATIVES_KEY in question.metadata
            and not isinstance(raw_alternatives, list | tuple)
        )
        or any(type(value) is not str or not value.strip() for value in alternatives)
        or len(set(alternatives)) != len(alternatives)
        or any(value in question.aliases or value == _ABSTENTION for value in alternatives)
    ):
        raise DataValidationError("E6 scoring inputs are invalid")
    rendered = runtime.render_prompt(prompt, question.text, metadata=question.metadata)
    alias_scores = tuple(
        _score(
            runtime,
            rendered,
            alias,
            layers=frozen_layers,
            site=site,
            state_factory=state_factory,
        )
        for alias in question.aliases
    )
    best_alias = min(
        range(len(alias_scores)),
        key=lambda index: (-alias_scores[index].mean_log_likelihood, index),
    )
    abstention = _score(
        runtime,
        rendered,
        _ABSTENTION,
        layers=frozen_layers,
        site=site,
        state_factory=state_factory,
    )
    alternative_scores = tuple(
        _score(
            runtime,
            rendered,
            value,
            layers=frozen_layers,
            site=site,
            state_factory=state_factory,
        )
        for value in alternatives
    )
    gold_rank = (
        1
        + sum(
            value.mean_log_likelihood > alias_scores[best_alias].mean_log_likelihood
            for value in alternative_scores
        )
        if alternative_scores
        else None
    )
    aliases_digest = stable_hash(list(question.aliases))
    alternative_text_sha256s = tuple(
        hashlib.sha256(value.encode("utf-8")).hexdigest() for value in alternatives
    )
    body: dict[str, Any] = {
        "condition_id": condition_id,
        "question_id": question.question_id,
        "method": method,
        "prompt_id": prompt.prompt_id,
        "rendered_prompt_sha256": rendered.sha256,
        "aliases_digest": aliases_digest,
        "alternative_text_sha256s": list(alternative_text_sha256s),
        "alias_scores": [value.to_dict() for value in alias_scores],
        "best_alias_index": best_alias,
        "abstention_score": abstention.to_dict(),
        "alternative_scores": [value.to_dict() for value in alternative_scores],
        "gold_rank": gold_rank,
    }
    return E6LikelihoodRecord(
        condition_id=condition_id,
        question_id=question.question_id,
        method=method,
        prompt_id=prompt.prompt_id,
        rendered_prompt_sha256=rendered.sha256,
        aliases_digest=aliases_digest,
        alternative_text_sha256s=alternative_text_sha256s,
        alias_scores=alias_scores,
        best_alias_index=best_alias,
        abstention_score=abstention,
        alternative_scores=alternative_scores,
        gold_rank=gold_rank,
        record_digest=stable_hash(body),
    )


def _frozen_alternatives(question: Question) -> tuple[str, ...]:
    raw = question.metadata.get(_ALTERNATIVES_KEY, ())
    if not isinstance(raw, list | tuple):
        raise FrozenArtifactError("E6 frozen plausible alternatives are invalid")
    alternatives = tuple(raw)
    if (
        any(type(value) is not str or not value.strip() for value in alternatives)
        or len(set(alternatives)) != len(alternatives)
        or any(value in question.aliases or value == _ABSTENTION for value in alternatives)
    ):
        raise FrozenArtifactError("E6 frozen plausible alternatives are invalid")
    return alternatives


def _assert_e6_response_texts(
    likelihood: E6LikelihoodRecord,
    question: Question,
) -> None:
    alternatives = _frozen_alternatives(question)
    alias_hashes = tuple(
        hashlib.sha256(value.encode("utf-8")).hexdigest() for value in question.aliases
    )
    alternative_hashes = tuple(
        hashlib.sha256(value.encode("utf-8")).hexdigest() for value in alternatives
    )
    if (
        likelihood.aliases_digest != stable_hash(list(question.aliases))
        or tuple(value.text_sha256 for value in likelihood.alias_scores) != alias_hashes
        or likelihood.abstention_score.text_sha256
        != hashlib.sha256(_ABSTENTION.encode("utf-8")).hexdigest()
        or likelihood.alternative_text_sha256s != alternative_hashes
        or tuple(value.text_sha256 for value in likelihood.alternative_scores)
        != alternative_hashes
    ):
        raise FrozenArtifactError("E6 scored answer texts differ from the frozen question")


def _receipt_states(likelihood: E6LikelihoodRecord) -> tuple[dict[str, Any], ...]:
    scores = (*likelihood.alias_scores, likelihood.abstention_score, *likelihood.alternative_scores)
    first = tuple(dict(item) for item in scores[0].execution_receipt["states"])
    first_static = tuple(
        {
            name: value
            for name, value in item.items()
            if name not in {"hook_applications", "generated_calls"}
        }
        for item in first
    )
    for score in scores:
        current = tuple(
            {
                name: value
                for name, value in dict(item).items()
                if name not in {"hook_applications", "generated_calls"}
            }
            for item in score.execution_receipt["states"]
        )
        if current != first_static:
            raise FrozenArtifactError("E6 response scores used different intervention states")
    return first


def _assert_e6_execution(
    likelihood: E6LikelihoodRecord,
    *,
    generation_record: GenerationRecord,
    condition: EvaluationCondition,
) -> str:
    """Verify actual hook receipts against the exact generated row and condition."""

    states = _receipt_states(likelihood)
    state_digest = likelihood.intervention_state_digest
    action = generation_record.metadata.get("policy_action")
    expects_intervention = condition.steering_method == "M1" or (
        condition.steering_method == "M3" and action == "intervene"
    )
    if not expects_intervention:
        if states:
            raise FrozenArtifactError("E6 no-intervention row executed a teacher-forced edit")
        return state_digest
    if len(states) != 1:
        raise FrozenArtifactError("E6 material row lacks one exact teacher-forced edit")
    state = states[0]
    expected_layer = (
        condition.layer if condition.steering_method == "M1" else generation_record.layer
    )
    expected_site = (
        condition.site if condition.steering_method == "M1" else generation_record.site
    )
    expected_scope = (
        condition.token_scope
        if condition.steering_method == "M1"
        else generation_record.token_scope
    )
    trace = generation_record.metadata.get("intervention_trace")
    if condition.steering_method == "M1":
        expected_alpha = condition.alpha
    else:
        if not isinstance(trace, Mapping):
            raise FrozenArtifactError("E6 adaptive generation magnitude is not replayable")
        direction_norm = trace.get("direction_norm")
        if (
            isinstance(direction_norm, bool)
            or not isinstance(direction_norm, int | float)
            or not math.isfinite(float(direction_norm))
            or float(direction_norm) <= 0
            or trace.get("alpha") != generation_record.alpha
        ):
            raise FrozenArtifactError("E6 adaptive generation magnitude is not replayable")
        expected_alpha = generation_record.alpha * float(direction_norm)
    receipt = likelihood.alias_scores[0].execution_receipt
    if (
        type(expected_layer) is not int
        or expected_site is None
        or expected_scope is None
        or expected_alpha == 0
        or state["layer"] != expected_layer
        or receipt["site"] != expected_site.value
        or state["token_scope"] != expected_scope.value
        or not math.isclose(
            float(state["alpha"]),
            float(expected_alpha),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise FrozenArtifactError("E6 teacher-forced execution differs from generation geometry")
    if (
        not isinstance(trace, Mapping)
        or state["direction_sha256"] != trace.get("direction_sha256")
    ):
        raise FrozenArtifactError(
            "E6 teacher-forced direction differs from the executed generation"
        )
    return state_digest


def e6_fixed_generation_receipt_body(record: GenerationRecord) -> dict[str, Any]:
    """Canonical attestation body for a material fixed-M1 generation."""

    return {
        "schema_version": 1,
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
        "method_artifact_sha256": record.metadata.get("method_artifact_sha256"),
        "intervention_trace": record.metadata.get("intervention_trace"),
    }


def _validate_e6_generation_runtime_evidence(
    record: GenerationRecord,
    *,
    runtime_identity: Mapping[str, Any],
) -> None:
    """Bind every E6 generation peak to its attestation and embedded source."""

    expected_auxiliary_peak = 0
    if record.steering_method == "M3":
        evidence = record.metadata.get("adaptive_controller_evidence")
        if not isinstance(evidence, Mapping):
            raise DataValidationError("E6 M3 generation lacks prompt-feature evidence")
        prompt_peak = evidence.get("prompt_feature_peak_memory_bytes")
        if isinstance(prompt_peak, bool) or not isinstance(prompt_peak, int) or prompt_peak <= 0:
            raise DataValidationError("E6 M3 prompt-feature peak is invalid")
        expected_auxiliary_peak = prompt_peak
    elif record.metadata.get("adaptive_controller_evidence") is not None:
        raise DataValidationError("non-adaptive E6 generation contains controller evidence")
    validate_generation_runtime_metrics(
        record.metadata.get("generation_runtime_metrics"),
        record=record,
        runtime_identity=runtime_identity,
        expected_auxiliary_peak_memory_bytes=expected_auxiliary_peak,
    )


def _assert_e6_fixed_generation(
    record: GenerationRecord,
    *,
    condition: EvaluationCondition | Mapping[str, Any],
    e3_direction_index: Mapping[tuple[str, str, str, int], str],
    e3_static_vectors_sha256: str,
    runtime_identity: Mapping[str, Any],
    execution_public_key: str | None = None,
) -> None:
    _validate_e6_generation_runtime_evidence(
        record,
        runtime_identity=runtime_identity,
    )
    method: object
    layer: object
    site: object
    token_scope: object
    scope: TokenScope | None
    alpha: object
    method_artifact: object
    if isinstance(condition, EvaluationCondition):
        method = condition.steering_method
        layer = condition.layer
        site = condition.site.value if condition.site is not None else None
        token_scope = (
            condition.token_scope.value if condition.token_scope is not None else None
        )
        scope = condition.token_scope
        alpha = condition.alpha
        method_artifact = condition.method_artifact_sha256
    else:
        method = condition.get("steering_method")
        layer = condition.get("layer")
        site = condition.get("site")
        token_scope = condition.get("token_scope")
        try:
            scope = TokenScope(str(token_scope))
        except ValueError as exc:
            raise DataValidationError("E6 fixed M1 token scope is invalid") from exc
        alpha = condition.get("alpha")
        method_artifact = condition.get("method_artifact_sha256")
    if (
        method != "M1"
        or record.steering_method != "M1"
        or type(layer) is not int
        or site not in {item.value for item in ActivationSite}
        or not isinstance(scope, TokenScope)
        or isinstance(alpha, bool)
        or not isinstance(alpha, int | float)
        or not math.isfinite(float(alpha))
        or float(alpha) == 0
    ):
        raise DataValidationError("E6 fixed-generation validation requires M1")
    trace = record.metadata.get("intervention_trace")
    expected_keys = {
        "layer",
        "site",
        "token_scope",
        "alpha",
        "sparsity",
        "applied_tokens",
        "applied_token_indices",
        "activation_delta_norm",
        "direction_sha256",
        "direction_norm",
        "pre_activation_sha256",
        "post_activation_sha256",
        "delta_sha256",
        "training_prompt_id",
        "extraction_method",
        "source_layer",
        "source_site",
        "e3_tensor_index",
    }
    if not isinstance(trace, Mapping) or set(trace) != expected_keys:
        raise DataValidationError("E6 fixed M1 generation lacks its exact execution trace")
    index_value = trace["e3_tensor_index"]
    if not isinstance(index_value, list) or len(index_value) != 4:
        raise DataValidationError("E6 fixed M1 tensor index is invalid")
    index = tuple(index_value)
    if (
        type(index[0]) is not str
        or type(index[1]) is not str
        or type(index[2]) is not str
        or type(index[3]) is not int
    ):
        raise DataValidationError("E6 fixed M1 tensor index is invalid")
    typed_index = (index[0], index[1], index[2], index[3])
    try:
        expected_direction_sha = e3_direction_index[typed_index]
    except KeyError as exc:
        raise DataValidationError("E6 fixed M1 tensor index is absent from E3") from exc
    if scope is TokenScope.FINAL_PROMPT:
        expected_indices = [-1]
    else:
        scope_limit = {
            TokenScope.FIRST_GENERATED: 1,
            TokenScope.FIRST_FOUR: 4,
            TokenScope.FIRST_EIGHT: 8,
            TokenScope.ALL_GENERATED: record.output_tokens,
            TokenScope.EXPONENTIAL_DECAY: record.output_tokens,
        }[scope]
        expected_indices = list(range(min(scope_limit, record.output_tokens)))
    expected_artifact = e6_e3_slice_digest(
        e3_static_vectors_sha256=e3_static_vectors_sha256,
        tensor_index=typed_index,
        direction_sha256=expected_direction_sha,
    )
    expected_delta_norm = abs(float(alpha)) * math.sqrt(len(expected_indices))
    if (
        not expected_indices
        or method_artifact != expected_artifact
        or record.metadata.get("method_artifact_sha256") != expected_artifact
        or trace["layer"] != layer
        or trace["site"] != site
        or trace["token_scope"] != token_scope
        or trace["alpha"] != alpha
        or trace["sparsity"] is not None
        or trace["source_layer"] != index[3]
        or trace["source_site"] != index[2]
        or trace["training_prompt_id"] != index[0]
        or trace["extraction_method"] != index[1]
        or trace["direction_sha256"] != expected_direction_sha
        or trace["direction_norm"] != 1.0
        or type(trace["applied_tokens"]) is not int
        or trace["applied_tokens"] != len(expected_indices)
        or trace["applied_token_indices"] != expected_indices
        or isinstance(trace["activation_delta_norm"], bool)
        or not isinstance(trace["activation_delta_norm"], int | float)
        or not math.isclose(
            float(trace["activation_delta_norm"]),
            expected_delta_norm,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
        or any(
            type(trace[name]) is not str or _SHA256.fullmatch(trace[name]) is None
            for name in ("pre_activation_sha256", "post_activation_sha256", "delta_sha256")
        )
        or trace["pre_activation_sha256"] == trace["post_activation_sha256"]
        or record.metadata.get("intervention_trace_digest") != stable_hash(dict(trace))
    ):
        raise DataValidationError("E6 fixed M1 trace does not prove the registered edit")
    if execution_public_key is not None:
        signature = record.metadata.get("e6_generation_execution_signature")
        if type(signature) is not str or re.fullmatch(r"[0-9a-f]{128}", signature) is None:
            raise FrozenArtifactError("E6 fixed M1 generation lacks a runtime signature")
        try:
            public_key = Ed25519PublicKey.from_public_bytes(
                bytes.fromhex(execution_public_key)
            )
            public_key.verify(
                bytes.fromhex(signature),
                canonical_json(e6_fixed_generation_receipt_body(record)).encode(),
            )
        except (InvalidSignature, ValueError) as exc:
            raise FrozenArtifactError("E6 fixed M1 generation signature is invalid") from exc


def execute_and_bind_e6_likelihood(
    *,
    attestor: E6RuntimeAttestor,
    runtime_artifact: str | Path,
    e3_static_vectors: str | Path,
    question: Question,
    prompt: PromptSpec,
    generation_record: GenerationRecord,
    condition: EvaluationCondition,
    layers: Sequence[int],
    site: ActivationSite,
    state_factory: Callable[[], Mapping[int, MlxInterventionState]] | None,
    question_bundle_sha256: str,
    e3_tensor_index: Sequence[Any] | None = None,
    max_new_tokens: int = 32,
    populate_generation: bool = False,
    generation_grader: Callable[[GenerationRecord], GenerationRecord] | None = None,
) -> E6ExecutedRow:
    """Score and sign E6 evidence through the concrete trusted MLX runtime."""

    normalized = validate_active_study_artifact_paths(
        {
            "E6 runtime attestation": runtime_artifact,
            "E3 static vectors": e3_static_vectors,
        }
    )
    runtime_artifact = normalized["E6 runtime attestation"]
    e3_static_vectors = normalized["E3 static vectors"]
    if type(attestor) is not E6RuntimeAttestor:
        raise DataValidationError("integrated E6 execution requires a runtime-owned attestor")
    runtime_artifact_sha256 = attestor.verify_runtime_artifact(runtime_artifact)
    condition.validate_record(generation_record)
    if condition.steering_method == "M0":
        if (
            generation_record.metadata.get("generation_runtime_metrics") is not None
            or isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens <= 0
        ):
            raise DataValidationError("E6 M0 generation must start from an unsigned runtime row")
        rendered = attestor.runtime.render_prompt(
            prompt, question.text, metadata=question.metadata
        )
        generated = attestor.runtime.generate_with_interventions(
            rendered,
            max_new_tokens=max_new_tokens,
            intervention_states={},
        )
        if populate_generation:
            if type(generated) is not MlxGenerationOutput:
                raise DataValidationError("E6 M0 runtime returned an invalid generation")
            if (
                generation_record.raw_output
                or generation_record.normalized_answer
                or generation_record.input_tokens != 0
                or generation_record.output_tokens != 0
                or generation_record.generation_latency_seconds != 0
                or generation_record.outcome is not Outcome.INCORRECT
            ):
                raise DataValidationError("E6 M0 populated execution requires an empty draft row")
            generation_record = replace(
                generation_record,
                raw_output=generated.text,
                normalized_answer=normalize_answer(generated.text),
                outcome=deterministic_short_answer_grade(generated.text, question.aliases),
                generation_latency_seconds=generated.latency_seconds,
                input_tokens=generated.input_tokens,
                output_tokens=generated.output_tokens,
            )
            if generation_grader is not None:
                ungraded = generation_record
                generation_record = generation_grader(ungraded)
                if (
                    type(generation_record) is not GenerationRecord
                    or generation_record.raw_output != ungraded.raw_output
                    or generation_record.rendered_prompt_hash != ungraded.rendered_prompt_hash
                    or generation_record.input_tokens != ungraded.input_tokens
                    or generation_record.output_tokens != ungraded.output_tokens
                    or generation_record.generation_latency_seconds
                    != ungraded.generation_latency_seconds
                    or generation_record.normalized_answer
                    != normalize_answer(generated.text)
                ):
                    raise DataValidationError("E6 M0 grader changed generated runtime facts")
        if (
            type(generated) is not MlxGenerationOutput
            or generated.rendered_prompt != rendered
            or generation_record.rendered_prompt_hash != rendered.sha256
            or generation_record.raw_output != generated.text
            or generation_record.input_tokens != generated.input_tokens
            or generation_record.output_tokens != generated.output_tokens
            or generation_record.normalized_answer != normalize_answer(generated.text)
            or (
                generation_grader is None
                and generation_record.outcome
                is not deterministic_short_answer_grade(generated.text, question.aliases)
            )
        ):
            raise DataValidationError("E6 M0 generated output differs from the ledger row")
        generation_record = replace(
            generation_record,
            generation_latency_seconds=generated.latency_seconds,
            metadata={
                **dict(generation_record.metadata),
                "generation_runtime_metrics": build_generation_runtime_metrics(
                    generated,
                    runtime_identity=attestor.attested_runtime_identity,
                ),
            },
        )
    if condition.steering_method == "M1":
        if (
            generation_record.metadata.get("intervention_trace") is not None
            or generation_record.metadata.get("intervention_trace_digest") is not None
            or generation_record.metadata.get("e6_generation_execution_signature") is not None
            or generation_record.metadata.get("generation_runtime_metrics") is not None
            or e3_tensor_index is None
            or isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens <= 0
            or state_factory is None
        ):
            raise DataValidationError(
                "E6 M1 generation must start from an unsigned trace-free row"
            )
        e3_path = Path(e3_static_vectors).resolve()
        e3_sha = sha256_path(e3_path)
        e3_index = _e3_direction_index(e3_path)
        index = tuple(e3_tensor_index)
        if (
            len(index) != 4
            or type(index[0]) is not str
            or type(index[1]) is not str
            or type(index[2]) is not str
            or type(index[3]) is not int
        ):
            raise DataValidationError("E6 M1 tensor index is invalid")
        typed_index = (index[0], index[1], index[2], index[3])
        try:
            direction_sha256 = e3_index[typed_index]
        except KeyError as exc:
            raise DataValidationError("E6 M1 tensor index is absent from E3") from exc
        expected_artifact = e6_e3_slice_digest(
            e3_static_vectors_sha256=e3_sha,
            tensor_index=typed_index,
            direction_sha256=direction_sha256,
        )
        states = dict(state_factory())
        if (
            set(states) != {condition.layer}
            or condition.method_artifact_sha256 != expected_artifact
            or generation_record.metadata.get("method_artifact_sha256")
            != expected_artifact
        ):
            raise DataValidationError("E6 M1 state differs from its registered E3 slice")
        assert condition.layer is not None
        assert condition.site is not None
        assert condition.token_scope is not None
        state = states[condition.layer]
        direction = np.asarray(state.direction, dtype=np.float32)
        if (
            type(state) is not MlxResearchInterventionState
            or direction.ndim != 1
            or hashlib.sha256(direction.tobytes(order="C")).hexdigest()
            != direction_sha256
            or not math.isclose(state.alpha, condition.alpha, rel_tol=0, abs_tol=1e-12)
            or state.token_scope is not condition.token_scope
            or state.applications != 0
            or state.generated_calls != 0
            or state.captured is not None
            or state.intervened is not None
        ):
            raise DataValidationError("E6 M1 generation state is not fresh or exact")
        rendered = attestor.runtime.render_prompt(
            prompt, question.text, metadata=question.metadata
        )
        generated = attestor.runtime.generate_with_interventions(
            rendered,
            max_new_tokens=max_new_tokens,
            intervention_states={(condition.layer, condition.site): state},
        )
        if populate_generation:
            if type(generated) is not MlxGenerationOutput:
                raise DataValidationError("E6 M1 runtime returned an invalid generation")
            if (
                generation_record.raw_output
                or generation_record.normalized_answer
                or generation_record.input_tokens != 0
                or generation_record.output_tokens != 0
                or generation_record.generation_latency_seconds != 0
                or generation_record.outcome is not Outcome.INCORRECT
            ):
                raise DataValidationError("E6 M1 populated execution requires an empty draft row")
            generation_record = replace(
                generation_record,
                raw_output=generated.text,
                normalized_answer=normalize_answer(generated.text),
                outcome=deterministic_short_answer_grade(generated.text, question.aliases),
                generation_latency_seconds=generated.latency_seconds,
                input_tokens=generated.input_tokens,
                output_tokens=generated.output_tokens,
            )
            if generation_grader is not None:
                ungraded = generation_record
                generation_record = generation_grader(ungraded)
                if (
                    type(generation_record) is not GenerationRecord
                    or generation_record.raw_output != ungraded.raw_output
                    or generation_record.rendered_prompt_hash != ungraded.rendered_prompt_hash
                    or generation_record.input_tokens != ungraded.input_tokens
                    or generation_record.output_tokens != ungraded.output_tokens
                    or generation_record.generation_latency_seconds
                    != ungraded.generation_latency_seconds
                    or generation_record.normalized_answer
                    != normalize_answer(generated.text)
                ):
                    raise DataValidationError("E6 M1 grader changed generated runtime facts")
        if (
            type(generated) is not MlxGenerationOutput
            or generated.rendered_prompt != rendered
            or generation_record.rendered_prompt_hash != rendered.sha256
            or generation_record.raw_output != generated.text
            or generation_record.input_tokens != generated.input_tokens
            or generation_record.output_tokens != generated.output_tokens
        ):
            raise DataValidationError("E6 M1 generated output differs from the ledger row")
        captured = np.asarray(state.captured, dtype=np.float32)
        intervened = np.asarray(state.intervened, dtype=np.float32)
        if (
            captured.shape != intervened.shape
            or captured.size == 0
            or not np.isfinite(captured).all()
            or not np.isfinite(intervened).all()
            or np.array_equal(captured, intervened)
        ):
            raise DataValidationError("E6 M1 hook did not expose a material state edit")
        expected_indices = (
            [-1]
            if condition.token_scope is TokenScope.FINAL_PROMPT
            else list(
                range(
                    min(
                        {
                            TokenScope.FIRST_GENERATED: 1,
                            TokenScope.FIRST_FOUR: 4,
                            TokenScope.FIRST_EIGHT: 8,
                            TokenScope.ALL_GENERATED: generated.output_tokens,
                            TokenScope.EXPONENTIAL_DECAY: generated.output_tokens,
                        }[condition.token_scope],
                        generated.output_tokens,
                    )
                )
            )
        )
        if state.applications != len(expected_indices) or not expected_indices:
            raise DataValidationError("E6 M1 hook applications differ from token scope")
        delta = np.ascontiguousarray(intervened - captured)
        trace = {
            "layer": condition.layer,
            "site": condition.site.value,
            "token_scope": condition.token_scope.value,
            "alpha": condition.alpha,
            "sparsity": None,
            "applied_tokens": state.applications,
            "applied_token_indices": expected_indices,
            "activation_delta_norm": abs(condition.alpha)
            * math.sqrt(len(expected_indices)),
            "direction_sha256": direction_sha256,
            "direction_norm": float(np.linalg.norm(direction)),
            "pre_activation_sha256": hashlib.sha256(
                np.ascontiguousarray(captured).tobytes(order="C")
            ).hexdigest(),
            "post_activation_sha256": hashlib.sha256(
                np.ascontiguousarray(intervened).tobytes(order="C")
            ).hexdigest(),
            "delta_sha256": hashlib.sha256(delta.tobytes(order="C")).hexdigest(),
            "training_prompt_id": typed_index[0],
            "extraction_method": typed_index[1],
            "source_layer": typed_index[3],
            "source_site": typed_index[2],
            "e3_tensor_index": list(typed_index),
        }
        generation_record = replace(
            generation_record,
            generation_latency_seconds=generated.latency_seconds,
            metadata={
                **dict(generation_record.metadata),
                "generation_runtime_metrics": build_generation_runtime_metrics(
                    generated,
                    runtime_identity=attestor.attested_runtime_identity,
                ),
                "intervention_trace": trace,
                "intervention_trace_digest": stable_hash(trace),
            },
        )
        _assert_e6_fixed_generation(
            generation_record,
            condition=condition,
            e3_direction_index=e3_index,
            e3_static_vectors_sha256=e3_sha,
            runtime_identity=attestor.attested_runtime_identity,
        )
        generation_record = replace(
            generation_record,
            metadata={
                **dict(generation_record.metadata),
                "e6_generation_execution_signature": attestor._sign(
                    e6_fixed_generation_receipt_body(generation_record)
                ),
            },
        )
    _validate_e6_generation_runtime_evidence(
        generation_record,
        runtime_identity=attestor.attested_runtime_identity,
    )
    action = generation_record.metadata.get("policy_action")
    expects_state = condition.steering_method == "M1" or (
        condition.steering_method == "M3" and action == "intervene"
    )
    if expects_state != (state_factory is not None):
        raise DataValidationError("E6 integrated state factory differs from generation action")
    likelihood = score_e6_question(
        runtime=attestor.runtime,
        question=question,
        prompt=prompt,
        method=condition.steering_method,
        condition_id=condition.condition_id,
        layers=layers,
        site=site,
        state_factory=state_factory,
    )
    reserved_metadata = {
        "e6_likelihood_record_digest",
        "e6_runtime_artifact_sha256",
        "e6_execution_public_key",
        "e6_question_bundle_sha256",
        *likelihood.generation_metadata(),
    }
    if reserved_metadata.intersection(generation_record.metadata):
        raise DataValidationError("unbound generation row preclaims E6 evidence metadata")
    bound_generation = replace(
        generation_record,
        metadata={
            **dict(generation_record.metadata),
            "e6_likelihood_record_digest": likelihood.record_digest,
            "e6_runtime_artifact_sha256": runtime_artifact_sha256,
            "e6_execution_public_key": attestor.execution_public_key,
            "e6_question_bundle_sha256": question_bundle_sha256,
            **dict(likelihood.generation_metadata()),
        },
    )
    condition.validate_record(bound_generation)
    bound_likelihood = _bind_e6_likelihood_record(
        likelihood,
        generation_record=bound_generation,
        condition=condition,
        runtime_artifact_sha256=runtime_artifact_sha256,
        question_bundle_sha256=question_bundle_sha256,
        attestor=attestor,
    )
    return E6ExecutedRow(
        generation_record=bound_generation,
        likelihood_record=bound_likelihood,
    )


@dataclass(frozen=True, slots=True)
class E6VerifiedLikelihoodRecord:
    likelihood: E6LikelihoodRecord
    benchmark: str
    generation_record_digest: str
    question_bundle_sha256: str
    method_artifact_sha256: str | None
    runtime_artifact_sha256: str
    intervention_state_digest: str
    execution_public_key: str
    execution_receipt_signature: str
    verified_record_digest: str
    schema_version: int = 2

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 2
            or type(self.likelihood) is not E6LikelihoodRecord
            or self.benchmark
            not in {"triviaqa", "simpleqa_verified", "aa_omniscience_public_600"}
            or any(
                type(value) is not str or _SHA256.fullmatch(value) is None
                for value in (
                    self.generation_record_digest,
                    self.question_bundle_sha256,
                    self.runtime_artifact_sha256,
                    self.execution_public_key,
                    self.verified_record_digest,
                )
            )
            or (self.likelihood.method != "M0")
            != (self.method_artifact_sha256 is not None)
            or (
                self.method_artifact_sha256 is not None
                and _SHA256.fullmatch(self.method_artifact_sha256) is None
            )
            or _SHA256.fullmatch(self.intervention_state_digest) is None
            or type(self.execution_receipt_signature) is not str
            or re.fullmatch(r"[0-9a-f]{128}", self.execution_receipt_signature) is None
            or self.verified_record_digest != stable_hash(self._body())
        ):
            raise DataValidationError("E6 verified likelihood record is invalid")

    def execution_body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "likelihood": self.likelihood.to_dict(),
            "benchmark": self.benchmark,
            "generation_record_digest": self.generation_record_digest,
            "question_bundle_sha256": self.question_bundle_sha256,
            "method_artifact_sha256": self.method_artifact_sha256,
            "runtime_artifact_sha256": self.runtime_artifact_sha256,
            "intervention_state_digest": self.intervention_state_digest,
            "execution_public_key": self.execution_public_key,
        }

    def _body(self) -> dict[str, Any]:
        return {
            **self.execution_body(),
            "execution_receipt_signature": self.execution_receipt_signature,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "verified_record_digest": self.verified_record_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> E6VerifiedLikelihoodRecord:
        expected = {
            "schema_version",
            "likelihood",
            "benchmark",
            "generation_record_digest",
            "question_bundle_sha256",
            "method_artifact_sha256",
            "runtime_artifact_sha256",
            "intervention_state_digest",
            "execution_public_key",
            "execution_receipt_signature",
            "verified_record_digest",
        }
        likelihood = value.get("likelihood")
        if set(value) != expected or not isinstance(likelihood, Mapping):
            raise DataValidationError("E6 verified likelihood-record keys differ")
        return cls(
            schema_version=value["schema_version"],
            likelihood=E6LikelihoodRecord.from_dict(likelihood),
            benchmark=value["benchmark"],
            generation_record_digest=value["generation_record_digest"],
            question_bundle_sha256=value["question_bundle_sha256"],
            method_artifact_sha256=value["method_artifact_sha256"],
            runtime_artifact_sha256=value["runtime_artifact_sha256"],
            intervention_state_digest=value["intervention_state_digest"],
            execution_public_key=value["execution_public_key"],
            execution_receipt_signature=value["execution_receipt_signature"],
            verified_record_digest=value["verified_record_digest"],
        )


@dataclass(frozen=True, slots=True)
class E6ExecutedRow:
    """One ledger-ready generation row and its runtime-signed E6 evidence."""

    generation_record: GenerationRecord
    likelihood_record: E6VerifiedLikelihoodRecord

    def __post_init__(self) -> None:
        if (
            type(self.generation_record) is not GenerationRecord
            or type(self.likelihood_record) is not E6VerifiedLikelihoodRecord
            or self.likelihood_record.generation_record_digest
            != stable_hash(self.generation_record.to_dict())
        ):
            raise DataValidationError("E6 executed row binding is invalid")


def _bind_e6_likelihood_record(
    likelihood: E6LikelihoodRecord,
    *,
    generation_record: GenerationRecord,
    condition: EvaluationCondition,
    runtime_artifact_sha256: str,
    question_bundle_sha256: str,
    attestor: E6RuntimeAttestor,
) -> E6VerifiedLikelihoodRecord:
    """Bind teacher-forced evidence to the exact generated row and frozen method state."""

    if (
        type(likelihood) is not E6LikelihoodRecord
        or type(generation_record) is not GenerationRecord
        or type(condition) is not EvaluationCondition
        or _SHA256.fullmatch(runtime_artifact_sha256) is None
        or _SHA256.fullmatch(question_bundle_sha256) is None
        or type(attestor) is not E6RuntimeAttestor
    ):
        raise DataValidationError("E6 likelihood binding inputs are invalid")
    execution_public_key = attestor.execution_public_key
    condition.validate_record(generation_record)
    intervention_state_digest = _assert_e6_execution(
        likelihood,
        generation_record=generation_record,
        condition=condition,
    )
    metadata = dict(likelihood.generation_metadata())
    if (
        likelihood.condition_id != condition.condition_id
        or likelihood.question_id != generation_record.question_id
        or likelihood.method != condition.steering_method
        or likelihood.prompt_id != condition.system_prompt_id
        or likelihood.rendered_prompt_sha256 != generation_record.rendered_prompt_hash
        or generation_record.metadata.get("e6_likelihood_record_digest")
        != likelihood.record_digest
        or generation_record.metadata.get("e6_runtime_artifact_sha256")
        != runtime_artifact_sha256
        or generation_record.metadata.get("e6_execution_public_key")
        != execution_public_key
        or generation_record.metadata.get("e6_question_bundle_sha256")
        != question_bundle_sha256
        or any(generation_record.metadata.get(key) != value for key, value in metadata.items())
    ):
        raise DataValidationError("E6 likelihood differs from its generated ledger row")
    execution_body = {
        "schema_version": 2,
        "likelihood": likelihood.to_dict(),
        "benchmark": generation_record.benchmark,
        "generation_record_digest": stable_hash(generation_record.to_dict()),
        "question_bundle_sha256": question_bundle_sha256,
        "method_artifact_sha256": condition.method_artifact_sha256,
        "runtime_artifact_sha256": runtime_artifact_sha256,
        "intervention_state_digest": intervention_state_digest,
        "execution_public_key": execution_public_key,
    }
    signature = attestor._sign(execution_body)
    body = {**execution_body, "execution_receipt_signature": signature}
    return E6VerifiedLikelihoodRecord(
        likelihood=likelihood,
        benchmark=generation_record.benchmark,
        generation_record_digest=stable_hash(generation_record.to_dict()),
        question_bundle_sha256=question_bundle_sha256,
        method_artifact_sha256=condition.method_artifact_sha256,
        runtime_artifact_sha256=runtime_artifact_sha256,
        intervention_state_digest=intervention_state_digest,
        execution_public_key=execution_public_key,
        execution_receipt_signature=signature,
        verified_record_digest=stable_hash(body),
    )


def _verify_e6_bound_record(
    value: E6VerifiedLikelihoodRecord,
    *,
    generation_record: GenerationRecord,
    condition: EvaluationCondition,
    question: Question,
    runtime_artifact_sha256: str,
    execution_public_key: str,
) -> None:
    value.__post_init__()
    condition.validate_record(generation_record)
    likelihood = value.likelihood
    expected_state = _assert_e6_execution(
        likelihood,
        generation_record=generation_record,
        condition=condition,
    )
    _assert_e6_response_texts(likelihood, question)
    if (
        value.benchmark != question.benchmark
        or value.generation_record_digest != stable_hash(generation_record.to_dict())
        or value.question_bundle_sha256
        != generation_record.metadata.get("e6_question_bundle_sha256")
        or value.method_artifact_sha256 != condition.method_artifact_sha256
        or value.runtime_artifact_sha256 != runtime_artifact_sha256
        or value.intervention_state_digest != expected_state
        or value.execution_public_key != execution_public_key
        or likelihood.condition_id != condition.condition_id
        or likelihood.question_id != question.question_id
        or likelihood.method != condition.steering_method
        or likelihood.prompt_id != condition.system_prompt_id
        or likelihood.rendered_prompt_sha256 != generation_record.rendered_prompt_hash
        or likelihood.aliases_digest != stable_hash(list(question.aliases))
        or generation_record.metadata.get("e6_likelihood_record_digest")
        != likelihood.record_digest
        or generation_record.metadata.get("e6_runtime_artifact_sha256")
        != runtime_artifact_sha256
        or generation_record.metadata.get("e6_execution_public_key")
        != execution_public_key
        or any(
            generation_record.metadata.get(key) != metric
            for key, metric in likelihood.generation_metadata().items()
        )
    ):
        raise FrozenArtifactError("E6 verified likelihood differs from its frozen ledger row")
    try:
        public_key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(execution_public_key))
        public_key.verify(
            bytes.fromhex(value.execution_receipt_signature),
            canonical_json(value.execution_body()).encode(),
        )
    except (InvalidSignature, ValueError) as exc:
        raise FrozenArtifactError("E6 likelihood runtime signature is invalid") from exc


def verify_e6_bound_record(
    value: E6VerifiedLikelihoodRecord,
    *,
    generation_record: GenerationRecord,
    condition: EvaluationCondition,
    question: Question,
    runtime_artifact_sha256: str,
    execution_public_key: str,
) -> Mapping[str, Any]:
    _verify_e6_bound_record(
        value,
        generation_record=generation_record,
        condition=condition,
        question=question,
        runtime_artifact_sha256=runtime_artifact_sha256,
        execution_public_key=execution_public_key,
    )
    return MappingProxyType(
        {"valid": True, "verified_record_digest": value.verified_record_digest}
    )


def _ordered_e6_questions(
    ledger: PhaseRunLedger,
    questions_by_benchmark: Mapping[str, Sequence[Question]],
) -> tuple[Question, ...]:
    if set(questions_by_benchmark) != set(ledger.contract.question_ids_by_benchmark):
        raise DataValidationError("E6 question bundles differ from the frozen ledger")
    ordered: list[Question] = []
    for benchmark in sorted(ledger.contract.question_ids_by_benchmark):
        questions = tuple(questions_by_benchmark[benchmark])
        expected_ids = ledger.contract.question_ids_by_benchmark[benchmark]
        if (
            any(type(value) is not Question or value.benchmark != benchmark for value in questions)
            or tuple(value.question_id for value in questions) != expected_ids
        ):
            raise DataValidationError("E6 questions differ from the frozen condition matrix")
        ordered.extend(questions)
    return tuple(ordered)


def _index_e6_questions(
    questions: Sequence[Question],
) -> Mapping[tuple[str, str], Question]:
    """Index questions by their protocol identity, which includes the benchmark."""

    indexed: dict[tuple[str, str], Question] = {}
    for question in questions:
        key = (question.benchmark, question.question_id)
        if key in indexed:
            raise DataValidationError("E6 question bundle contains a duplicate identity")
        indexed[key] = question
    return MappingProxyType(indexed)


def _e6_scientific_matrix(
    ledger: PhaseRunLedger,
    questions: Sequence[Question],
    records: Sequence[E6VerifiedLikelihoodRecord],
) -> tuple[bool, tuple[str, ...]]:
    expected_partitions = {
        "triviaqa": "T-dev",
        "simpleqa_verified": "simpleqa-eval",
        "aa_omniscience_public_600": "aa-eval",
    }
    strata = {
        (
            condition.benchmark,
            condition.system_prompt_id,
            condition.steering_method,
        )
        for condition in ledger.contract.conditions
    }
    exact_strata = {
        (benchmark, prompt, method)
        for benchmark in _EXPECTED_QUESTION_COUNTS
        for prompt in _PROMPTS
        for method in _METHODS
    }
    counts = {
        benchmark: sum(value.benchmark == benchmark for value in questions)
        for benchmark in _EXPECTED_QUESTION_COUNTS
    }
    questions_by_id = {
        (question.benchmark, question.question_id): question for question in questions
    }
    rank_eligible: list[str] = []
    for benchmark in sorted(_EXPECTED_QUESTION_COUNTS):
        applicable = {
            question.question_id
            for question in questions
            if question.benchmark == benchmark and _frozen_alternatives(question)
        }
        benchmark_records = [value for value in records if value.benchmark == benchmark]
        if any(
            (value.likelihood.gold_rank is not None)
            != (value.likelihood.question_id in applicable)
            for value in benchmark_records
        ):
            raise DataValidationError("E6 answer ranks are incomplete where applicable")
        if applicable:
            rank_eligible.append(benchmark)
    scientific = (
        counts == _EXPECTED_QUESTION_COUNTS
        and strata == exact_strata
        and len(ledger.contract.conditions) == len(exact_strata)
        and len(questions_by_id) == len(questions)
        and all(
            condition.partition == expected_partitions[condition.benchmark]
            for condition in ledger.contract.conditions
        )
    )
    return scientific, tuple(rank_eligible)


def _e3_direction_index(
    directory: str | Path,
) -> Mapping[tuple[str, str, str, int], str]:
    source = Path(directory)
    if (
        source.is_symlink()
        or not source.is_dir()
        or {path.name for path in source.iterdir()} != {"metadata.json", "vectors.npz"}
        or any(path.is_symlink() or not path.is_file() for path in source.iterdir())
    ):
        raise FrozenArtifactError("E6 E3 vector artifact inventory differs")
    try:
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        body = dict(metadata)
        metadata_digest = body.pop("metadata_digest")
        with np.load(source / "vectors.npz", allow_pickle=False) as values:
            if set(values.files) != {
                "directions",
                "reference_rms",
                "correct_counts",
                "incorrect_counts",
            }:
                raise FrozenArtifactError("E6 E3 vector arrays differ")
            directions = np.asarray(values["directions"])
            reference_rms = np.asarray(values["reference_rms"])
            correct_counts = np.asarray(values["correct_counts"])
            incorrect_counts = np.asarray(values["incorrect_counts"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot load E6 E3 vectors: {exc}") from exc
    expected_metadata_keys = {
        "schema_version",
        "phase",
        "plan_identity",
        "protocol",
        "prompt_axis",
        "extraction_axis",
        "site_axis",
        "layer_axis",
        "hidden_width",
        "rows_processed",
        "response_pooling",
        "scientific_eligible",
        "maximum_peak_memory_bytes",
        "generation_chain_head",
        "checkpoint_chain_head",
        "vectors_sha256",
        "data_fingerprint",
    }
    prompt_axis = body.get("prompt_axis")
    extraction_axis = body.get("extraction_axis")
    site_axis = body.get("site_axis")
    layer_axis = body.get("layer_axis")
    protocol = E3Protocol()
    if (
        metadata_digest != stable_hash(body)
        or set(body) != expected_metadata_keys
        or body.get("schema_version") != 1
        or body.get("phase") != "E3-construction"
        or body.get("scientific_eligible") is not True
        or body.get("protocol") != protocol.to_dict()
        or body.get("rows_processed") != protocol.construction_rows
        or body.get("response_pooling") != protocol.response_pooling
        or body.get("plan_identity") is None
        or _SHA256.fullmatch(str(body.get("plan_identity"))) is None
        or _SHA256.fullmatch(str(body.get("generation_chain_head"))) is None
        or _SHA256.fullmatch(str(body.get("checkpoint_chain_head"))) is None
        or _SHA256.fullmatch(str(body.get("data_fingerprint"))) is None
        or body.get("vectors_sha256") != sha256_file(source / "vectors.npz")
        or prompt_axis != ["P0-neutral", "P2-calibrated-abstention"]
        or extraction_axis != ["M1-R", "M1-P"]
        or site_axis != [value.value for value in protocol.candidate_sites]
        or layer_axis != list(protocol.candidate_layers)
        or directions.dtype != np.float32
        or directions.shape
        != (
            len(prompt_axis),
            len(extraction_axis),
            len(site_axis),
            len(layer_axis),
            body.get("hidden_width"),
        )
        or not np.isfinite(directions).all()
        or reference_rms.dtype != np.float64
        or reference_rms.shape != directions.shape[:-1]
        or not np.isfinite(reference_rms).all()
        or np.any(reference_rms <= 0)
        or correct_counts.dtype != np.int64
        or incorrect_counts.dtype != np.int64
        or correct_counts.shape != directions.shape[:-1]
        or incorrect_counts.shape != directions.shape[:-1]
        or np.any(correct_counts <= 0)
        or np.any(incorrect_counts <= 0)
        or np.any(correct_counts + incorrect_counts != protocol.steer_rows)
        or any(
            not np.all(correct_counts[prompt_index] == correct_counts[prompt_index, 0, 0, 0])
            or not np.all(
                incorrect_counts[prompt_index]
                == incorrect_counts[prompt_index, 0, 0, 0]
            )
            for prompt_index in range(len(prompt_axis))
        )
        or not np.allclose(
            np.linalg.norm(directions, axis=-1), 1.0, rtol=1e-5, atol=1e-6
        )
    ):
        raise FrozenArtifactError("E6 E3 vector identity or geometry differs")
    return MappingProxyType(
        {
            (prompt, extraction, site, layer): hashlib.sha256(
                np.ascontiguousarray(
                    directions[
                        prompt_index,
                        extraction_index,
                        site_index,
                        layer_index,
                    ]
                ).tobytes(order="C")
            ).hexdigest()
            for prompt_index, prompt in enumerate(prompt_axis)
            for extraction_index, extraction in enumerate(extraction_axis)
            for site_index, site in enumerate(site_axis)
            for layer_index, layer in enumerate(layer_axis)
        }
    )


def e6_e3_slice_digest(
    *, e3_static_vectors_sha256: str, tensor_index: Sequence[Any], direction_sha256: str
) -> str:
    """Return the condition artifact identity for one exact E3 tensor slice."""

    index = tuple(tensor_index)
    if (
        _SHA256.fullmatch(e3_static_vectors_sha256) is None
        or len(index) != 4
        or type(index[0]) is not str
        or type(index[1]) is not str
        or type(index[2]) is not str
        or type(index[3]) is not int
        or _SHA256.fullmatch(direction_sha256) is None
    ):
        raise DataValidationError("E6 E3 tensor-slice identity is invalid")
    return stable_hash(
        {
            "schema_version": 1,
            "e3_static_vectors_sha256": e3_static_vectors_sha256,
            "tensor_index": list(index),
            "direction_sha256": direction_sha256,
        }
    )


def _e6_bundle_body(
    *,
    ledger: PhaseRunLedger,
    runtime_artifact_sha256: str,
    frozen_question_bundle_sha256: str,
    e3_static_vectors_sha256: str,
    official_grader_bundle_sha256: str,
    official_grader_manifest_digest: str,
    execution_public_key: str,
    questions_sha256: str,
    likelihoods_sha256: str,
    questions: Sequence[Question],
    records: Sequence[E6VerifiedLikelihoodRecord],
) -> dict[str, Any]:
    scientific, rank_eligible = _e6_scientific_matrix(ledger, questions, records)
    return {
        "schema_version": 3,
        "phase": ExperimentPhase.E6.value,
        "study_protocol_digest": ledger.contract.study_protocol_digest,
        "contract_digest": ledger.contract.digest,
        "record_set_digest": ledger.record_set_digest(),
        "input_fingerprints": dict(ledger.contract.input_fingerprints),
        "prerequisite_digests": dict(ledger.contract.prerequisite_digests),
        "runtime_artifact": "runtime-artifact",
        "runtime_artifact_sha256": runtime_artifact_sha256,
        "frozen_question_bundle": "frozen-question-bundle",
        "frozen_question_bundle_sha256": frozen_question_bundle_sha256,
        "e3_static_vectors": "e3-static-vectors",
        "e3_static_vectors_sha256": e3_static_vectors_sha256,
        "official_grader_bundle": "official-grader-bundle",
        "official_grader_bundle_sha256": official_grader_bundle_sha256,
        "official_grader_manifest_digest": official_grader_manifest_digest,
        "execution_public_key": execution_public_key,
        "questions_sha256": questions_sha256,
        "likelihoods_sha256": likelihoods_sha256,
        "question_count": len(questions),
        "likelihood_record_count": len(records),
        "rank_record_count": sum(
            value.likelihood.gold_rank is not None for value in records
        ),
        "rank_eligible_benchmarks": list(rank_eligible),
        "scientific_eligible": scientific,
    }


def write_e6_likelihood_bundle(
    destination: str | Path,
    *,
    ledger_directory: str | Path,
    study: StudyProtocol,
    questions_by_benchmark: Mapping[str, Sequence[Question]],
    records: Sequence[E6VerifiedLikelihoodRecord],
    runtime_artifact: str | Path,
    frozen_question_bundle: str | Path,
    execution_public_key: str,
) -> Mapping[str, Any]:
    """Freeze signed teacher-forced evidence for every exact E6 ledger row."""

    normalized = validate_active_study_artifact_paths(
        {
            "E6 likelihood bundle": destination,
            "E6 phase ledger": ledger_directory,
            "E6 runtime attestation": runtime_artifact,
            "E6 frozen question bundle": frozen_question_bundle,
        }
    )
    output = normalized["E6 likelihood bundle"]
    ledger_directory = normalized["E6 phase ledger"]
    runtime_artifact = normalized["E6 runtime attestation"]
    frozen_question_bundle = normalized["E6 frozen question bundle"]
    if output.exists() or output.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite E6 likelihood bundle: {output}")
    ledger = PhaseRunLedger.open(ledger_directory, study=study)
    PhaseRunLedger._verify_creation_evidence(ledger)
    completed, expected = ledger.progress()
    if ledger.contract.phase is not ExperimentPhase.E6 or completed != expected:
        raise DataValidationError("E6 likelihood bundle requires a complete pre-terminal ledger")
    questions = _ordered_e6_questions(ledger, questions_by_benchmark)
    question_bundle_path = Path(frozen_question_bundle).resolve()
    _validate_question_bundle(question_bundle_path, ledger.contract)
    question_bundle_sha = sha256_path(question_bundle_path)
    e3_path = ledger.directory / "inputs" / "E3_static_vectors"
    e3_sha = sha256_path(e3_path)
    grader_path = ledger.directory / "inputs" / "official_grader_bundle"
    grader_bundle = load_e6_official_grader_bundle(grader_path)
    e3_direction_index = _e3_direction_index(e3_path)
    if (
        e3_sha != ledger.contract.input_fingerprints["E3_static_vectors"]
        or grader_bundle.fingerprint
        != ledger.contract.input_fingerprints.get("official_grader_bundle")
    ):
        raise FrozenArtifactError("E6 packaged E3 vectors differ from the frozen input")
    frozen_bundle_questions = tuple(
        question
        for benchmark in sorted(ledger.contract.question_ids_by_benchmark)
        for question in read_questions(question_bundle_path / f"{benchmark}.jsonl")
    )
    if questions != frozen_bundle_questions:
        raise DataValidationError("E6 questions differ from the frozen question bundle")
    runtime_path = Path(runtime_artifact).resolve()
    if (
        runtime_path.is_symlink()
        or not (runtime_path.is_file() or runtime_path.is_dir())
        or _SHA256.fullmatch(execution_public_key) is None
    ):
        raise DataValidationError("E6 runtime artifact or execution key is invalid")
    runtime_attestation = _load_e6_runtime_attestation(runtime_path)
    if runtime_attestation["execution_public_key"] != execution_public_key:
        raise DataValidationError("E6 runtime attestation key differs")
    runtime_sha = sha256_path(runtime_path)
    expected_keys = [
        (condition.condition_id, question_id)
        for condition in ledger.contract.conditions
        for question_id in ledger.contract.question_ids_by_benchmark[condition.benchmark]
    ]
    frozen_records = tuple(records)
    if len(frozen_records) != len(expected_keys):
        raise DataValidationError("E6 likelihood bundle record count differs")
    generation_records = {
        (value.condition_id, value.question_id): value for value in ledger.records()
    }
    conditions = {value.condition_id: value for value in ledger.contract.conditions}
    questions_by_key = _index_e6_questions(questions)
    for expected_key, value in zip(expected_keys, frozen_records, strict=True):
        observed_key = (value.likelihood.condition_id, value.likelihood.question_id)
        if type(value) is not E6VerifiedLikelihoodRecord or observed_key != expected_key:
            raise DataValidationError("E6 likelihood bundle order or identity differs")
        if value.question_bundle_sha256 != question_bundle_sha:
            raise DataValidationError("E6 record question-bundle fingerprint differs")
        condition = conditions[expected_key[0]]
        question = questions_by_key[(condition.benchmark, expected_key[1])]
        verify_e6_factual_grade(
            generation_records[expected_key],
            question,
            grader_bundle=grader_bundle,
        )
        _verify_e6_bound_record(
            value,
            generation_record=generation_records[expected_key],
            condition=condition,
            question=question,
            runtime_artifact_sha256=runtime_sha,
            execution_public_key=execution_public_key,
        )
        _assert_e6_runtime_condition(
            runtime_attestation,
            model_repository=condition.model_repository,
            model_revision=condition.model_revision,
            quantization=condition.quantization,
            model_num_layers=condition.model_num_layers,
            seed=condition.seed,
            execution_public_key=execution_public_key,
        )
        _validate_e6_generation_runtime_evidence(
            generation_records[expected_key],
            runtime_identity=runtime_attestation["runtime_identity"],
        )
        if value.likelihood.method == "M1":
            _assert_e6_fixed_generation(
                generation_records[expected_key],
                condition=condition,
                e3_direction_index=e3_direction_index,
                e3_static_vectors_sha256=e3_sha,
                runtime_identity=runtime_attestation["runtime_identity"],
                execution_public_key=execution_public_key,
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        _copy_frozen_artifact(runtime_path, stage / "runtime-artifact", runtime_sha)
        _copy_frozen_artifact(
            question_bundle_path,
            stage / "frozen-question-bundle",
            question_bundle_sha,
        )
        _copy_frozen_artifact(e3_path, stage / "e3-static-vectors", e3_sha)
        _copy_frozen_artifact(
            grader_path,
            stage / "official-grader-bundle",
            grader_bundle.fingerprint,
        )
        write_questions(stage / "questions.jsonl", questions)
        with (stage / "likelihoods.jsonl").open("x", encoding="utf-8") as handle:
            for value in frozen_records:
                handle.write(json.dumps(value.to_dict(), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        body = _e6_bundle_body(
            ledger=ledger,
            runtime_artifact_sha256=runtime_sha,
            frozen_question_bundle_sha256=question_bundle_sha,
            e3_static_vectors_sha256=e3_sha,
            official_grader_bundle_sha256=grader_bundle.fingerprint,
            official_grader_manifest_digest=grader_bundle.manifest_digest,
            execution_public_key=execution_public_key,
            questions_sha256=sha256_file(stage / "questions.jsonl"),
            likelihoods_sha256=sha256_file(stage / "likelihoods.jsonl"),
            questions=questions,
            records=frozen_records,
        )
        (stage / "manifest.json").write_text(
            json.dumps(
                {**body, "manifest_digest": stable_hash(body)},
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
    return verify_e6_likelihood_bundle(
        output,
        ledger_directory=ledger_directory,
        study=study,
        questions_by_benchmark=questions_by_benchmark,
        runtime_artifact=runtime_path,
        execution_public_key=execution_public_key,
    )


def verify_e6_likelihood_bundle(
    directory: str | Path,
    *,
    ledger_directory: str | Path,
    study: StudyProtocol,
    questions_by_benchmark: Mapping[str, Sequence[Question]] | None = None,
    runtime_artifact: str | Path | None = None,
    execution_public_key: str | None = None,
) -> Mapping[str, Any]:
    """Replay the exact E6 question, ledger, runtime, state, and signature bindings."""

    source = Path(directory)
    expected_files = {
        "manifest.json",
        "questions.jsonl",
        "likelihoods.jsonl",
        "runtime-artifact",
        "frozen-question-bundle",
        "e3-static-vectors",
        "official-grader-bundle",
    }
    if (
        source.is_symlink()
        or not source.is_dir()
        or {value.name for value in source.iterdir()} != expected_files
        or any(value.is_symlink() for value in source.rglob("*"))
        or any(
            not (source / name).is_file()
            for name in ("manifest.json", "questions.jsonl", "likelihoods.jsonl")
        )
    ):
        raise FrozenArtifactError("E6 likelihood bundle inventory differs")
    ledger = PhaseRunLedger.open(ledger_directory, study=study)
    PhaseRunLedger._verify_creation_evidence(ledger)
    completed, expected = ledger.progress()
    if ledger.contract.phase is not ExperimentPhase.E6 or completed != expected:
        raise FrozenArtifactError("E6 likelihood ledger is incomplete or cross-phase")
    _validate_question_bundle(source / "frozen-question-bundle", ledger.contract)
    live_questions = tuple(
        question
        for benchmark in sorted(ledger.contract.question_ids_by_benchmark)
        for question in read_questions(
            source / "frozen-question-bundle" / f"{benchmark}.jsonl"
        )
    )
    if questions_by_benchmark is not None and live_questions != _ordered_e6_questions(
        ledger, questions_by_benchmark
    ):
        raise FrozenArtifactError("E6 live questions differ from the packaged frozen input")
    packaged_questions = tuple(read_questions(source / "questions.jsonl"))
    if packaged_questions != live_questions:
        raise FrozenArtifactError("E6 packaged questions differ from the frozen source")
    try:
        records = tuple(
            E6VerifiedLikelihoodRecord.from_dict(json.loads(line))
            for line in (source / "likelihoods.jsonl").read_text(encoding="utf-8").splitlines()
        )
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError, DataValidationError) as exc:
        raise FrozenArtifactError(f"cannot load E6 likelihood bundle: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise FrozenArtifactError("E6 likelihood manifest must be an object")
    runtime_path = source / "runtime-artifact"
    runtime_attestation = _load_e6_runtime_attestation(runtime_path)
    runtime_sha = sha256_path(runtime_path)
    question_bundle_sha = sha256_path(source / "frozen-question-bundle")
    e3_path = source / "e3-static-vectors"
    e3_sha = sha256_path(e3_path)
    grader_bundle = load_e6_official_grader_bundle(source / "official-grader-bundle")
    e3_direction_index = _e3_direction_index(e3_path)
    if (
        e3_sha != ledger.contract.input_fingerprints["E3_static_vectors"]
        or grader_bundle.fingerprint
        != ledger.contract.input_fingerprints.get("official_grader_bundle")
    ):
        raise FrozenArtifactError("E6 packaged E3 vectors differ from the frozen input")
    manifest_execution_key = manifest.get("execution_public_key")
    if (
        type(manifest_execution_key) is not str
        or _SHA256.fullmatch(manifest_execution_key) is None
        or (execution_public_key is not None and execution_public_key != manifest_execution_key)
        or (
            runtime_artifact is not None
            and sha256_path(Path(runtime_artifact).resolve()) != runtime_sha
        )
    ):
        raise FrozenArtifactError("E6 external runtime or execution key differs from its package")
    if runtime_attestation["execution_public_key"] != manifest_execution_key:
        raise FrozenArtifactError("E6 runtime attestation key differs from its manifest")
    expected_keys = [
        (condition.condition_id, question_id)
        for condition in ledger.contract.conditions
        for question_id in ledger.contract.question_ids_by_benchmark[condition.benchmark]
    ]
    generation_records = {
        (value.condition_id, value.question_id): value for value in ledger.records()
    }
    conditions = {value.condition_id: value for value in ledger.contract.conditions}
    questions_by_key = _index_e6_questions(live_questions)
    if len(records) != len(expected_keys):
        raise FrozenArtifactError("E6 likelihood record count differs")
    for expected_key, value in zip(expected_keys, records, strict=True):
        if (value.likelihood.condition_id, value.likelihood.question_id) != expected_key:
            raise FrozenArtifactError("E6 likelihood record order differs")
        condition = conditions[expected_key[0]]
        question = questions_by_key[(condition.benchmark, expected_key[1])]
        verify_e6_factual_grade(
            generation_records[expected_key],
            question,
            grader_bundle=grader_bundle,
        )
        _verify_e6_bound_record(
            value,
            generation_record=generation_records[expected_key],
            condition=condition,
            question=question,
            runtime_artifact_sha256=runtime_sha,
            execution_public_key=manifest_execution_key,
        )
        _assert_e6_runtime_condition(
            runtime_attestation,
            model_repository=condition.model_repository,
            model_revision=condition.model_revision,
            quantization=condition.quantization,
            model_num_layers=condition.model_num_layers,
            seed=condition.seed,
            execution_public_key=manifest_execution_key,
        )
        if value.question_bundle_sha256 != question_bundle_sha:
            raise FrozenArtifactError("E6 record question-bundle fingerprint differs")
        _validate_e6_generation_runtime_evidence(
            generation_records[expected_key],
            runtime_identity=runtime_attestation["runtime_identity"],
        )
        if value.likelihood.method == "M1":
            _assert_e6_fixed_generation(
                generation_records[expected_key],
                condition=condition,
                e3_direction_index=e3_direction_index,
                e3_static_vectors_sha256=e3_sha,
                runtime_identity=runtime_attestation["runtime_identity"],
                execution_public_key=manifest_execution_key,
            )
    body = _e6_bundle_body(
        ledger=ledger,
        runtime_artifact_sha256=runtime_sha,
        frozen_question_bundle_sha256=question_bundle_sha,
        e3_static_vectors_sha256=e3_sha,
        official_grader_bundle_sha256=grader_bundle.fingerprint,
        official_grader_manifest_digest=grader_bundle.manifest_digest,
        execution_public_key=manifest_execution_key,
        questions_sha256=sha256_file(source / "questions.jsonl"),
        likelihoods_sha256=sha256_file(source / "likelihoods.jsonl"),
        questions=live_questions,
        records=records,
    )
    expected_manifest = {**body, "manifest_digest": stable_hash(body)}
    if dict(manifest) != expected_manifest:
        raise FrozenArtifactError("E6 likelihood manifest differs from exact replay")
    return MappingProxyType(
        {
            "valid": True,
            "manifest_digest": expected_manifest["manifest_digest"],
            "record_count": len(records),
            "rank_record_count": body["rank_record_count"],
            "rank_eligible_benchmarks": tuple(body["rank_eligible_benchmarks"]),
            "scientific_eligible": body["scientific_eligible"],
        }
    )


def _assert_e6_execution_facts(
    likelihood: E6LikelihoodRecord,
    *,
    generation: GenerationRecord,
    facts: Mapping[str, Any],
) -> None:
    states = _receipt_states(likelihood)
    method = generation.steering_method
    action = generation.metadata.get("policy_action")
    expects_intervention = method == "M1" or (method == "M3" and action == "intervene")
    if not expects_intervention:
        if states:
            raise FrozenArtifactError("E6 packaged no-intervention row executed an edit")
        return
    if len(states) != 1:
        raise FrozenArtifactError("E6 packaged material row lacks one executed edit")
    state = states[0]
    layer = facts.get("layer") if method == "M1" else generation.layer
    site = facts.get("site") if method == "M1" else (
        generation.site.value if generation.site is not None else None
    )
    token_scope = facts.get("token_scope") if method == "M1" else (
        generation.token_scope.value if generation.token_scope is not None else None
    )
    alpha = facts.get("alpha") if method == "M1" else generation.alpha
    if (
        type(layer) is not int
        or site not in {item.value for item in ActivationSite}
        or token_scope not in {item.value for item in TokenScope}
        or isinstance(alpha, bool)
        or not isinstance(alpha, int | float)
        or not math.isfinite(float(alpha))
        or float(alpha) == 0
        or state["layer"] != layer
        or likelihood.alias_scores[0].execution_receipt["site"] != site
        or state["token_scope"] != token_scope
        or not math.isclose(
            float(state["alpha"]), float(alpha), rel_tol=1e-12, abs_tol=1e-12
        )
    ):
        raise FrozenArtifactError("E6 packaged execution differs from condition geometry")
    trace = generation.metadata.get("intervention_trace")
    if (
        not isinstance(trace, Mapping)
        or state["direction_sha256"] != trace.get("direction_sha256")
    ):
        raise FrozenArtifactError("E6 packaged direction differs from generation")


def _verify_e6_gate_matrix(
    *,
    questions: Sequence[Question],
    records: Sequence[E6VerifiedLikelihoodRecord],
    generation_records: Sequence[GenerationRecord],
    condition_facts: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    expected_partitions = {
        "triviaqa": "T-dev",
        "simpleqa_verified": "simpleqa-eval",
        "aa_omniscience_public_600": "aa-eval",
    }
    counts = {
        benchmark: sum(question.benchmark == benchmark for question in questions)
        for benchmark in _EXPECTED_QUESTION_COUNTS
    }
    dimensions = {
        (
            facts.get("benchmark"),
            facts.get("system_prompt_id"),
            facts.get("steering_method"),
        )
        for facts in condition_facts.values()
    }
    expected_dimensions = {
        (benchmark, prompt, method)
        for benchmark in _EXPECTED_QUESTION_COUNTS
        for prompt in _PROMPTS
        for method in _METHODS
    }
    if (
        counts != _EXPECTED_QUESTION_COUNTS
        or len(questions) != sum(_EXPECTED_QUESTION_COUNTS.values())
        or len({(question.benchmark, question.question_id) for question in questions})
        != len(questions)
        or dimensions != expected_dimensions
        or len(condition_facts) != len(expected_dimensions)
    ):
        raise FrozenArtifactError("E6 gate bundle does not form the exact scientific matrix")
    for facts in condition_facts.values():
        benchmark = facts.get("benchmark")
        method = facts.get("steering_method")
        artifact = facts.get("method_artifact_sha256")
        if facts.get("partition") != expected_partitions.get(str(benchmark)):
            raise FrozenArtifactError("E6 gate condition partition differs")
        if method == "M0":
            valid = (
                artifact is None
                and facts.get("layer") is None
                and facts.get("site") is None
                and facts.get("token_scope") is None
                and facts.get("alpha") == 0
                and facts.get("adaptive_policy") is None
            )
        elif method == "M1":
            valid = (
                isinstance(artifact, str)
                and _SHA256.fullmatch(artifact) is not None
                and type(facts.get("layer")) is int
                and facts.get("site") in {item.value for item in ActivationSite}
                and facts.get("token_scope") in {item.value for item in TokenScope}
                and isinstance(facts.get("alpha"), int | float)
                and not isinstance(facts.get("alpha"), bool)
                and math.isfinite(float(facts["alpha"]))
                and float(facts["alpha"]) != 0
                and facts.get("adaptive_policy") is None
            )
        else:
            valid = (
                isinstance(artifact, str)
                and _SHA256.fullmatch(artifact) is not None
                and facts.get("layer") is None
                and facts.get("site") is None
                and facts.get("token_scope") is None
                and facts.get("alpha") == 0
                and isinstance(facts.get("adaptive_policy"), Mapping)
            )
        if not valid:
            raise FrozenArtifactError("E6 gate condition is not a material registered method")
    expected_generation_count = sum(
        _EXPECTED_QUESTION_COUNTS[str(facts["benchmark"])]
        for facts in condition_facts.values()
    )
    if (
        len(generation_records) != expected_generation_count
        or len(records) != expected_generation_count
    ):
        raise FrozenArtifactError("E6 gate record count differs from the exact matrix")
    question_by_key = {
        (question.benchmark, question.question_id): question for question in questions
    }
    record_by_key = {
        (record.likelihood.condition_id, record.likelihood.question_id): record
        for record in records
    }
    rank_eligible: set[str] = set()
    for generation in generation_records:
        try:
            question = question_by_key[(generation.benchmark, generation.question_id)]
            record = record_by_key[(generation.condition_id, generation.question_id)]
        except KeyError as exc:
            raise FrozenArtifactError("E6 gate exact row coverage differs") from exc
        has_rank = bool(_frozen_alternatives(question))
        if has_rank:
            rank_eligible.add(question.benchmark)
        if (record.likelihood.gold_rank is not None) != has_rank:
            raise FrozenArtifactError("E6 gate rank evidence is incomplete where applicable")
    return tuple(sorted(rank_eligible))


def verify_e6_gate_artifact(
    directory: str | Path,
    *,
    contract_digest: str,
    record_set_digest: str,
    generation_records: Sequence[GenerationRecord],
    condition_facts: Mapping[str, Mapping[str, Any]],
    input_fingerprints: Mapping[str, str],
    frozen_inputs_verified: bool,
) -> Mapping[str, Any]:
    """Replay a packaged E6 bundle using only immutable gate-context facts."""

    source = Path(directory)
    if (
        source.is_symlink()
        or not source.is_dir()
        or {value.name for value in source.iterdir()}
        != {
            "manifest.json",
            "questions.jsonl",
            "likelihoods.jsonl",
            "runtime-artifact",
            "frozen-question-bundle",
            "e3-static-vectors",
            "official-grader-bundle",
        }
        or any(value.is_symlink() for value in source.rglob("*"))
    ):
        raise FrozenArtifactError("E6 gate likelihood bundle inventory differs")
    try:
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        questions = tuple(read_questions(source / "questions.jsonl"))
        records = tuple(
            E6VerifiedLikelihoodRecord.from_dict(json.loads(line))
            for line in (source / "likelihoods.jsonl").read_text(encoding="utf-8").splitlines()
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError, DataValidationError) as exc:
        raise FrozenArtifactError(f"cannot load E6 packaged gate artifact: {exc}") from exc
    expected_keys = {
        "schema_version",
        "phase",
        "study_protocol_digest",
        "contract_digest",
        "record_set_digest",
        "input_fingerprints",
        "prerequisite_digests",
        "runtime_artifact",
        "runtime_artifact_sha256",
        "frozen_question_bundle",
        "frozen_question_bundle_sha256",
        "e3_static_vectors",
        "e3_static_vectors_sha256",
        "official_grader_bundle",
        "official_grader_bundle_sha256",
        "official_grader_manifest_digest",
        "execution_public_key",
        "questions_sha256",
        "likelihoods_sha256",
        "question_count",
        "likelihood_record_count",
        "rank_record_count",
        "rank_eligible_benchmarks",
        "scientific_eligible",
        "manifest_digest",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_keys:
        raise FrozenArtifactError("E6 packaged gate manifest keys differ")
    body = dict(manifest)
    manifest_digest = body.pop("manifest_digest")
    runtime_path = source / str(body["runtime_artifact"])
    question_bundle_path = source / str(body["frozen_question_bundle"])
    e3_path = source / str(body["e3_static_vectors"])
    grader_path = source / str(body["official_grader_bundle"])
    grader_bundle = load_e6_official_grader_bundle(grader_path)
    execution_key = body["execution_public_key"]
    runtime_attestation = _load_e6_runtime_attestation(runtime_path)
    e3_direction_index = _e3_direction_index(e3_path)
    if (
        type(body["schema_version"]) is not int
        or body["schema_version"] != 3
        or body["phase"] != ExperimentPhase.E6.value
        or body["contract_digest"] != contract_digest
        or body["record_set_digest"] != record_set_digest
        or not frozen_inputs_verified
        or body["input_fingerprints"] != dict(input_fingerprints)
        or body["questions_sha256"] != sha256_file(source / "questions.jsonl")
        or body["likelihoods_sha256"] != sha256_file(source / "likelihoods.jsonl")
        or body["runtime_artifact_sha256"] != sha256_path(runtime_path)
        or body["runtime_artifact"] != "runtime-artifact"
        or body["frozen_question_bundle"] != "frozen-question-bundle"
        or body["frozen_question_bundle_sha256"] != sha256_path(question_bundle_path)
        or body["e3_static_vectors"] != "e3-static-vectors"
        or body["e3_static_vectors_sha256"] != sha256_path(e3_path)
        or body["e3_static_vectors_sha256"]
        != input_fingerprints.get("E3_static_vectors")
        or body["official_grader_bundle"] != "official-grader-bundle"
        or body["official_grader_bundle_sha256"] != grader_bundle.fingerprint
        or body["official_grader_bundle_sha256"]
        != input_fingerprints.get("official_grader_bundle")
        or body["official_grader_manifest_digest"] != grader_bundle.manifest_digest
        or type(execution_key) is not str
        or _SHA256.fullmatch(execution_key) is None
        or manifest_digest != stable_hash(body)
        or body["question_count"] != len(questions)
        or body["likelihood_record_count"] != len(records)
        or body["rank_record_count"]
        != sum(value.likelihood.gold_rank is not None for value in records)
        or body["scientific_eligible"] is not True
    ):
        raise FrozenArtifactError("E6 packaged gate manifest identity differs")
    frozen_questions = tuple(
        question
        for benchmark in sorted(_EXPECTED_QUESTION_COUNTS)
        for question in read_questions(question_bundle_path / f"{benchmark}.jsonl")
    )
    if frozen_questions != questions:
        raise FrozenArtifactError("E6 packaged questions differ from their frozen input")
    ledger_records = {
        (value.condition_id, value.question_id): value for value in generation_records
    }
    if len(ledger_records) != len(generation_records) or len(records) != len(ledger_records):
        raise FrozenArtifactError("E6 gate bundle does not cover the exact ledger")
    recomputed_rank_eligible = _verify_e6_gate_matrix(
        questions=questions,
        records=records,
        generation_records=generation_records,
        condition_facts=condition_facts,
    )
    questions_by_key = {(value.benchmark, value.question_id): value for value in questions}
    adaptive_keys = {
        facts.get("adaptive_policy", {}).get("execution_public_key")
        for facts in condition_facts.values()
        if isinstance(facts.get("adaptive_policy"), Mapping)
    }
    if adaptive_keys != {execution_key}:
        raise FrozenArtifactError("E6 execution key differs from the adaptive conditions")
    verified: dict[tuple[str, str], E6VerifiedLikelihoodRecord] = {}
    for value in records:
        likelihood = value.likelihood
        key = (likelihood.condition_id, likelihood.question_id)
        if key in verified or key not in ledger_records or key[0] not in condition_facts:
            raise FrozenArtifactError("E6 gate bundle references a duplicate or unknown row")
        generation = ledger_records[key]
        facts = condition_facts[key[0]]
        try:
            question = questions_by_key[(generation.benchmark, generation.question_id)]
        except KeyError as exc:
            raise FrozenArtifactError("E6 gate bundle lacks a ledger question") from exc
        verify_e6_factual_grade(
            generation,
            question,
            grader_bundle=grader_bundle,
        )
        expected_state = likelihood.intervention_state_digest
        if (
            value.benchmark != generation.benchmark
            or value.generation_record_digest != stable_hash(generation.to_dict())
            or value.method_artifact_sha256 != facts.get("method_artifact_sha256")
            or value.runtime_artifact_sha256 != body["runtime_artifact_sha256"]
            or value.intervention_state_digest != expected_state
            or value.question_bundle_sha256
            != body["frozen_question_bundle_sha256"]
            or generation.metadata.get("e6_question_bundle_sha256")
            != body["frozen_question_bundle_sha256"]
            or value.execution_public_key != execution_key
            or likelihood.method != generation.steering_method
            or likelihood.prompt_id != generation.system_prompt_id
            or likelihood.rendered_prompt_sha256 != generation.rendered_prompt_hash
            or likelihood.aliases_digest != stable_hash(list(question.aliases))
            or generation.metadata.get("e6_likelihood_record_digest")
            != likelihood.record_digest
            or generation.metadata.get("e6_runtime_artifact_sha256")
            != body["runtime_artifact_sha256"]
            or generation.metadata.get("e6_execution_public_key") != execution_key
            or any(
                generation.metadata.get(name) != metric
                for name, metric in likelihood.generation_metadata().items()
            )
        ):
            raise FrozenArtifactError("E6 gate likelihood differs from its ledger row")
        _assert_e6_response_texts(likelihood, question)
        _assert_e6_execution_facts(likelihood, generation=generation, facts=facts)
        _assert_e6_runtime_condition(
            runtime_attestation,
            model_repository=str(facts.get("model_repository")),
            model_revision=str(facts.get("model_revision")),
            quantization=str(facts.get("quantization")),
            model_num_layers=int(facts.get("model_num_layers", -1)),
            seed=int(facts.get("seed", -1)),
            execution_public_key=execution_key,
        )
        _validate_e6_generation_runtime_evidence(
            generation,
            runtime_identity=runtime_attestation["runtime_identity"],
        )
        if likelihood.method == "M1":
            _assert_e6_fixed_generation(
                generation,
                condition=facts,
                e3_direction_index=e3_direction_index,
                e3_static_vectors_sha256=str(body["e3_static_vectors_sha256"]),
                runtime_identity=runtime_attestation["runtime_identity"],
                execution_public_key=execution_key,
            )
        try:
            public_key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(execution_key))
            public_key.verify(
                bytes.fromhex(value.execution_receipt_signature),
                canonical_json(value.execution_body()).encode(),
            )
        except (InvalidSignature, ValueError) as exc:
            raise FrozenArtifactError("E6 packaged runtime signature is invalid") from exc
        verified[key] = value
    rank_eligible = tuple(body["rank_eligible_benchmarks"])
    if (
        type(body["rank_eligible_benchmarks"]) is not list
        or rank_eligible != recomputed_rank_eligible
        or any(value not in _EXPECTED_QUESTION_COUNTS for value in rank_eligible)
    ):
        raise FrozenArtifactError("E6 rank-eligible benchmark identity differs")
    return MappingProxyType(
        {
            "manifest": MappingProxyType(dict(manifest)),
            "records": MappingProxyType(verified),
            "rank_eligible_benchmarks": rank_eligible,
        }
    )


def _e6_paired_rows(ledger: PhaseRunLedger) -> tuple[dict[str, str], ...]:
    strata: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    for condition in ledger.contract.conditions:
        key = (
            condition.model_repository,
            condition.benchmark,
            condition.system_prompt_id,
            condition.partition,
            condition.comparison_group,
        )
        methods = strata.setdefault(key, {})
        if condition.steering_method in methods:
            raise DataValidationError("E6 repeats a method within a paired stratum")
        methods[condition.steering_method] = condition.condition_id
    if any(set(methods) != _METHODS for methods in strata.values()):
        raise DataValidationError("E6 lacks an exact M0/M1/M3 paired stratum")
    rows: list[dict[str, str]] = []
    for key in sorted(strata):
        methods = strata[key]
        for method in ("M1", "M3"):
            rows.extend(
                {
                    "question_id": question_id,
                    "baseline_condition_id": methods["M0"],
                    "intervention_condition_id": methods[method],
                }
                for question_id in ledger.contract.question_ids_by_benchmark[key[1]]
            )
    return tuple(rows)


def finalize_e6_phase(
    destination: str | Path,
    *,
    ledger_directory: str | Path,
    study: StudyProtocol,
    likelihood_bundle: str | Path,
    questions_by_benchmark: Mapping[str, Sequence[Question]],
    runtime_artifact: str | Path,
    execution_public_key: str,
) -> Mapping[str, Any]:
    """Evaluate the registered E6 decomposition and freeze its terminal ledger."""

    normalized = validate_active_study_artifact_paths(
        {
            "E6 finalization": destination,
            "E6 phase ledger": ledger_directory,
            "E6 likelihood bundle": likelihood_bundle,
            "E6 runtime attestation": runtime_artifact,
        }
    )
    output = normalized["E6 finalization"]
    ledger_directory = normalized["E6 phase ledger"]
    likelihood_bundle = normalized["E6 likelihood bundle"]
    runtime_artifact = normalized["E6 runtime attestation"]
    bundle = verify_e6_likelihood_bundle(
        likelihood_bundle,
        ledger_directory=ledger_directory,
        study=study,
        questions_by_benchmark=questions_by_benchmark,
        runtime_artifact=runtime_artifact,
        execution_public_key=execution_public_key,
    )
    if bundle["scientific_eligible"] is not True:
        raise DataValidationError("E6 finalization requires the scientific likelihood matrix")
    ledger = PhaseRunLedger.open(ledger_directory, study=study)
    PhaseRunLedger._verify_creation_evidence(ledger)
    rows = _e6_paired_rows(ledger)
    if output.exists() or output.is_symlink():
        if output.is_symlink() or not output.is_dir():
            raise FrozenArtifactError(f"E6 finalization path is not a directory: {output}")
        packaged = (
            ledger.directory
            / "gate-artifacts"
            / "knowledge_recovery_separated_from_abstention_substitution"
            / "likelihood-bundle"
        )
        if not packaged.exists() or sha256_path(packaged) != sha256_path(likelihood_bundle):
            raise FrozenArtifactError("existing E6 finalization packages another likelihood bundle")
        return verify_e6_phase(output, ledger_directory=ledger_directory, study=study)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    gate = "knowledge_recovery_separated_from_abstention_substitution"
    try:
        evidence_path = stage / f"{gate}.json"
        write_gate_evidence(
            evidence_path,
            phase=ExperimentPhase.E6,
            gate=gate,
            contract_digest=ledger.contract.digest,
            record_set_digest=ledger.record_set_digest(),
            observations=rows,
            parameters={
                "likelihood_bundle_manifest_digest": bundle["manifest_digest"]
            },
        )
        result = ledger.evaluate_gate(
            gate,
            evidence_path,
            supporting_artifacts={"likelihood-bundle": likelihood_bundle},
        )
        status, terminal_digest, terminal = _finalize_or_recover_e6_ledger(
            ledger, gate=gate, result=result
        )
        receipt_body: dict[str, Any] = {
            "schema_version": 1,
            "phase": ExperimentPhase.E6.value,
            "status": status,
            "ledger_directory": str(Path(ledger_directory).resolve()),
            "contract_digest": ledger.contract.digest,
            "record_set_digest": terminal.record_set_digest,
            "likelihood_bundle_artifact": f"{gate}/likelihood-bundle",
            "likelihood_bundle_manifest_digest": bundle["manifest_digest"],
            "gate_result_digest": result.gate_digest,
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
    return verify_e6_phase(
        output,
        ledger_directory=ledger_directory,
        study=study,
    )


def _finalize_or_recover_e6_ledger(
    ledger: PhaseRunLedger,
    *,
    gate: str,
    result: Any,
) -> tuple[str, str, PhaseCompletion | PhaseFalsification]:
    """Create E6's terminal marker once or recover it after a wrapper crash."""

    complete_exists = (ledger.directory / "complete.json").exists()
    falsified_exists = (ledger.directory / "falsified.json").exists()
    if complete_exists and falsified_exists:
        raise FrozenArtifactError("E6 ledger has conflicting terminal markers")
    terminal: PhaseCompletion | PhaseFalsification
    if complete_exists:
        if not result.passed:
            raise FrozenArtifactError("completed E6 ledger conflicts with recomputed gate")
        terminal = ledger.verify_complete()
        status = "complete"
        digest = terminal.completion_digest
    elif falsified_exists:
        if result.passed:
            raise FrozenArtifactError("falsified E6 ledger conflicts with recomputed gate")
        terminal = ledger.verify_falsified()
        status = "falsified"
        digest = terminal.falsification_digest
    elif result.passed:
        terminal = ledger.finalize({gate: result})
        status = "complete"
        digest = terminal.completion_digest
    else:
        terminal = ledger.finalize_falsified({gate: result})
        status = "falsified"
        digest = terminal.falsification_digest
    if dict(terminal.gate_result_digests) != {gate: result.gate_digest}:
        raise FrozenArtifactError("terminal E6 ledger differs from recomputed gate evidence")
    return status, digest, terminal


def verify_e6_phase(
    directory: str | Path,
    *,
    ledger_directory: str | Path,
    study: StudyProtocol,
) -> Mapping[str, Any]:
    """Replay the E6 likelihood artifact, terminal ledger, gate, and receipt."""

    source = Path(directory)
    gate = "knowledge_recovery_separated_from_abstention_substitution"
    if (
        source.is_symlink()
        or not source.is_dir()
        or {value.name for value in source.iterdir()}
        != {f"{gate}.json", "receipt.json"}
        or any(value.is_symlink() or not value.is_file() for value in source.iterdir())
    ):
        raise FrozenArtifactError("E6 finalization artifact inventory differs")
    ledger = PhaseRunLedger.open(ledger_directory, study=study)
    packaged_bundle = (
        ledger.directory / "gate-artifacts" / gate / "likelihood-bundle"
    )
    bundle = verify_e6_likelihood_bundle(
        packaged_bundle,
        ledger_directory=ledger_directory,
        study=study,
    )
    try:
        receipt = json.loads((source / "receipt.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot load E6 finalization receipt: {exc}") from exc
    if not isinstance(receipt, dict):
        raise FrozenArtifactError("E6 finalization receipt must be an object")
    receipt_body = dict(receipt)
    receipt_digest = receipt_body.pop("receipt_digest", None)
    expected_receipt_keys = {
        "schema_version",
        "phase",
        "status",
        "ledger_directory",
        "contract_digest",
        "record_set_digest",
        "likelihood_bundle_artifact",
        "likelihood_bundle_manifest_digest",
        "gate_result_digest",
        "terminal_digest",
        "scientific_eligible",
    }
    if (
        set(receipt_body) != expected_receipt_keys
        or type(receipt_body.get("schema_version")) is not int
        or receipt_body.get("schema_version") != 1
        or receipt_body.get("phase") != ExperimentPhase.E6.value
        or receipt_body.get("ledger_directory")
        != str(Path(ledger_directory).resolve())
        or receipt_body.get("likelihood_bundle_artifact")
        != f"{gate}/likelihood-bundle"
        or receipt_digest != stable_hash(receipt_body)
    ):
        raise FrozenArtifactError("E6 finalization receipt identity differs")
    status = receipt_body["status"]
    terminal: PhaseCompletion | PhaseFalsification
    if status == "complete":
        terminal = ledger.verify_complete()
        terminal_digest = terminal.completion_digest
        scientific = True
    elif status == "falsified":
        terminal = ledger.verify_falsified()
        terminal_digest = terminal.falsification_digest
        scientific = False
    else:
        raise FrozenArtifactError("E6 finalization status differs")
    gate_artifacts = dict(terminal.gate_artifact_fingerprints)
    if (
        receipt_body["contract_digest"] != ledger.contract.digest
        or receipt_body["record_set_digest"] != terminal.record_set_digest
        or receipt_body["likelihood_bundle_manifest_digest"]
        != bundle["manifest_digest"]
        or receipt_body["gate_result_digest"] != terminal.gate_result_digests[gate]
        or receipt_body["terminal_digest"] != terminal_digest
        or receipt_body["scientific_eligible"] is not scientific
        or gate_artifacts
        != {
            f"{gate}/evaluation": sha256_file(source / f"{gate}.json"),
            f"{gate}/likelihood-bundle": sha256_path(packaged_bundle),
        }
    ):
        raise FrozenArtifactError("E6 finalization differs from terminal ledger replay")
    return MappingProxyType(
        {
            "valid": True,
            "status": status,
            "receipt_digest": receipt_digest,
            "likelihood_bundle_manifest_digest": bundle["manifest_digest"],
            "terminal_digest": terminal_digest,
            "scientific_eligible": scientific,
        }
    )

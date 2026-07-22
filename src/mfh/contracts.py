"""Canonical, serializable contracts shared by all experiment phases.

The research plan calls for many runtimes and methods. Keeping their records in
one strict schema is what makes paired comparisons and frozen analyses possible.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum, StrEnum
from typing import Any

from mfh.errors import ConfigurationError, DataValidationError

_COMMIT_SHA = re.compile(r"^[0-9a-f]{40,64}$")


def _nonempty(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise DataValidationError(f"{field_name} must be non-empty")
    return normalized


def _finite(value: float, field_name: str) -> float:
    if not math.isfinite(value):
        raise DataValidationError(f"{field_name} must be finite, got {value!r}")
    return value


class Outcome(StrEnum):
    """Unified outcome labels from section 4 of the research plan."""

    CORRECT = "C"
    PARTIAL = "P"
    INCORRECT = "I"
    ABSTENTION = "A"
    UNSCORABLE = "U"

    @property
    def is_attempted(self) -> bool:
        return self in {Outcome.CORRECT, Outcome.PARTIAL, Outcome.INCORRECT}


class Runtime(StrEnum):
    VLLM = "vllm"
    SYNTHETIC = "synthetic"


class ActivationSite(StrEnum):
    POST_ATTENTION = "post_attention"
    POST_MLP = "post_mlp"
    BLOCK_OUTPUT = "block_output"


class TokenScope(StrEnum):
    FINAL_PROMPT = "final_prompt"
    FIRST_GENERATED = "first_generated"
    FIRST_FOUR = "first_four_generated"
    FIRST_EIGHT = "first_eight_generated"
    ALL_GENERATED = "all_generated"
    EXPONENTIAL_DECAY = "exponential_decay"


@dataclass(frozen=True, slots=True)
class Question:
    question_id: str
    benchmark: str
    text: str
    aliases: tuple[str, ...]
    split: str | None = None
    entities: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "question_id", _nonempty(self.question_id, "question_id"))
        object.__setattr__(self, "benchmark", _nonempty(self.benchmark, "benchmark"))
        object.__setattr__(self, "text", _nonempty(self.text, "text"))
        aliases = tuple(dict.fromkeys(a.strip() for a in self.aliases if a.strip()))
        if not aliases:
            raise DataValidationError(f"question {self.question_id!r} has no accepted aliases")
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "entities", tuple(e.strip() for e in self.entities if e.strip()))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    repository: str
    revision: str
    runtime: Runtime
    quantization: str
    num_layers: int
    dtype: str = "auto"
    trust_remote_code: bool = False
    role: str = "research"
    artifact: str | None = None
    artifact_sha256: str | None = None
    artifact_size_bytes: int | None = None
    candidate_layers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for name in ("name", "repository", "revision", "quantization", "dtype", "role"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if not _COMMIT_SHA.fullmatch(self.revision):
            raise ConfigurationError(
                f"model {self.name!r} revision must be an immutable 40-64 character "
                f"lowercase commit SHA, got {self.revision!r}"
            )
        if self.num_layers <= 0:
            raise ConfigurationError("num_layers must be positive")
        candidate_layers = tuple(int(layer) for layer in self.candidate_layers)
        if len(set(candidate_layers)) != len(candidate_layers):
            raise ConfigurationError("candidate_layers must be unique")
        if any(layer < 0 or layer >= self.num_layers for layer in candidate_layers):
            raise ConfigurationError(f"candidate_layers must be in [0, {self.num_layers - 1}]")
        object.__setattr__(self, "candidate_layers", candidate_layers)
        if self.artifact_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.artifact_sha256
        ):
            raise ConfigurationError("artifact_sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    name: str
    repository: str
    revision: str
    config: str
    split: str
    format: str
    question_column: str
    answer_column: str
    id_column: str

    def __post_init__(self) -> None:
        for name in (
            "name",
            "repository",
            "revision",
            "config",
            "split",
            "format",
            "question_column",
            "answer_column",
            "id_column",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if not _COMMIT_SHA.fullmatch(self.revision):
            raise ConfigurationError(
                f"benchmark {self.name!r} revision must be an immutable commit SHA"
            )


@dataclass(frozen=True, slots=True)
class PromptSpec:
    prompt_id: str
    text: str
    permits_abstention: bool = True
    deployment_eligible: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_id", _nonempty(self.prompt_id, "prompt_id"))
        object.__setattr__(self, "text", _nonempty(self.text, "text"))


@dataclass(frozen=True, slots=True)
class AdaptivePolicySpec:
    """Frozen, ledger-recomputable routing policy for M3 and M6."""

    release_risk_threshold: float
    abstention_probability_threshold: float
    alpha_max: float
    alpha_beta: float
    layer: int | None
    site: ActivationSite | None
    token_scope: TokenScope | None
    direction_sha256: str | None
    direction_norm: float | None
    execution_public_key: str
    sparsity: float | None = None
    schema_version: int = 1
    controller_artifact_sha256: str | None = None
    candidate_layers: tuple[int, ...] = ()
    candidate_sites: tuple[ActivationSite, ...] = ()
    candidate_token_scopes: tuple[TokenScope, ...] = ()
    vector_count: int | None = None
    likely_unknown_risk_threshold: float | None = None
    alpha_mode: str | None = None
    alpha_risk_threshold: float | None = None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version not in {1, 2}:
            raise ConfigurationError("adaptive policy requires schema version 1 or 2")
        for name in (
            "release_risk_threshold",
            "abstention_probability_threshold",
        ):
            raw_value = getattr(self, name)
            if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
                raise ConfigurationError(f"{name} must be an exact JSON number")
            value = _finite(float(raw_value), name)
            if not 0 < value < 1:
                raise ConfigurationError(f"{name} must be in (0, 1)")
            object.__setattr__(self, name, value)
        if (
            isinstance(self.alpha_max, bool)
            or not isinstance(self.alpha_max, int | float)
            or isinstance(self.alpha_beta, bool)
            or not isinstance(self.alpha_beta, int | float)
        ):
            raise ConfigurationError("adaptive alpha fields must be exact JSON numbers")
        alpha_max = _finite(float(self.alpha_max), "alpha_max")
        alpha_beta = _finite(float(self.alpha_beta), "alpha_beta")
        if alpha_max < 1e-4 or alpha_beta <= 0:
            raise ConfigurationError(
                "adaptive alpha maximum must be at least 1e-4 and beta must be positive"
            )
        if not isinstance(self.execution_public_key, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.execution_public_key
        ):
            raise ConfigurationError(
                "adaptive policy execution_public_key must be 32-byte lowercase hex"
            )
        if self.sparsity is not None:
            if isinstance(self.sparsity, bool) or not isinstance(
                self.sparsity, int | float
            ):
                raise ConfigurationError("adaptive policy sparsity must be an exact JSON number")
            if not 0 < self.sparsity <= 1:
                raise ConfigurationError("adaptive policy sparsity must be in (0, 1]")
        object.__setattr__(self, "alpha_max", alpha_max)
        object.__setattr__(self, "alpha_beta", alpha_beta)
        if self.schema_version == 1:
            if (
                type(self.layer) is not int
                or self.layer < 0
                or not isinstance(self.site, ActivationSite)
                or not isinstance(self.token_scope, TokenScope)
                or not isinstance(self.direction_sha256, str)
                or not re.fullmatch(r"[0-9a-f]{64}", self.direction_sha256)
                or isinstance(self.direction_norm, bool)
                or not isinstance(self.direction_norm, int | float)
            ):
                raise ConfigurationError("adaptive policy v1 requires fixed intervention geometry")
            direction_norm = _finite(float(self.direction_norm), "direction_norm")
            if direction_norm <= 0:
                raise ConfigurationError("adaptive direction norm must be positive")
            if (
                self.controller_artifact_sha256 is not None
                or self.candidate_layers
                or self.candidate_sites
                or self.candidate_token_scopes
                or self.vector_count is not None
                or self.likely_unknown_risk_threshold is not None
                or self.alpha_mode is not None
                or self.alpha_risk_threshold is not None
            ):
                raise ConfigurationError("adaptive policy v1 cannot declare routed geometry")
            object.__setattr__(self, "direction_norm", direction_norm)
            return

        if any(
            value is not None
            for value in (
                self.layer,
                self.site,
                self.token_scope,
                self.direction_sha256,
                self.direction_norm,
            )
        ):
            raise ConfigurationError("adaptive policy v2 cannot declare one fixed direction")
        if not isinstance(self.controller_artifact_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.controller_artifact_sha256
        ):
            raise ConfigurationError("adaptive policy v2 requires a controller artifact SHA-256")
        layers = tuple(self.candidate_layers)
        sites = tuple(self.candidate_sites)
        scopes = tuple(self.candidate_token_scopes)
        if (
            not layers
            or any(type(value) is not int for value in layers)
            or len(set(layers)) != len(layers)
            or any(value < 0 for value in layers)
            or not sites
            or any(not isinstance(value, ActivationSite) for value in sites)
            or len(set(sites)) != len(sites)
            or not scopes
            or any(not isinstance(value, TokenScope) for value in scopes)
            or len(set(scopes)) != len(scopes)
        ):
            raise ConfigurationError("adaptive policy v2 candidate geometry is invalid")
        if type(self.vector_count) is not int or self.vector_count not in {1, 4, 8, 16}:
            raise ConfigurationError("adaptive policy v2 vector count is not preregistered")
        if self.alpha_mode not in {
            "fixed",
            "risk_gated",
            "risk_gated_hard_threshold",
        }:
            raise ConfigurationError("adaptive policy v2 requires a frozen alpha mode")
        if isinstance(self.alpha_risk_threshold, bool) or not isinstance(
            self.alpha_risk_threshold, int | float
        ):
            raise ConfigurationError("adaptive policy v2 requires an alpha-risk threshold")
        alpha_risk = _finite(float(self.alpha_risk_threshold), "alpha_risk_threshold")
        if not 0 <= alpha_risk <= 1:
            raise ConfigurationError("adaptive alpha-risk threshold must be in [0, 1]")
        if isinstance(self.likely_unknown_risk_threshold, bool) or not isinstance(
            self.likely_unknown_risk_threshold, int | float
        ):
            raise ConfigurationError("adaptive policy v2 requires a likely-unknown threshold")
        likely_unknown = _finite(
            float(self.likely_unknown_risk_threshold), "likely_unknown_risk_threshold"
        )
        if not self.release_risk_threshold < likely_unknown < 1:
            raise ConfigurationError(
                "likely-unknown risk threshold must exceed the release threshold"
            )
        object.__setattr__(self, "candidate_layers", layers)
        object.__setattr__(self, "candidate_sites", sites)
        object.__setattr__(self, "candidate_token_scopes", scopes)
        object.__setattr__(self, "likely_unknown_risk_threshold", likely_unknown)
        object.__setattr__(self, "alpha_risk_threshold", alpha_risk)

    def to_dict(self) -> dict[str, Any]:
        if self.schema_version == 1:
            assert self.site is not None and self.token_scope is not None
            return {
                "schema_version": self.schema_version,
                "release_risk_threshold": self.release_risk_threshold,
                "abstention_probability_threshold": self.abstention_probability_threshold,
                "alpha_max": self.alpha_max,
                "alpha_beta": self.alpha_beta,
                "layer": self.layer,
                "site": self.site.value,
                "token_scope": self.token_scope.value,
                "direction_sha256": self.direction_sha256,
                "direction_norm": self.direction_norm,
                "execution_public_key": self.execution_public_key,
                "sparsity": self.sparsity,
            }
        return {
            "schema_version": self.schema_version,
            "release_risk_threshold": self.release_risk_threshold,
            "abstention_probability_threshold": self.abstention_probability_threshold,
            "alpha_max": self.alpha_max,
            "alpha_beta": self.alpha_beta,
            "execution_public_key": self.execution_public_key,
            "sparsity": self.sparsity,
            "controller_artifact_sha256": self.controller_artifact_sha256,
            "candidate_layers": list(self.candidate_layers),
            "candidate_sites": [value.value for value in self.candidate_sites],
            "candidate_token_scopes": [value.value for value in self.candidate_token_scopes],
            "vector_count": self.vector_count,
            "likely_unknown_risk_threshold": self.likely_unknown_risk_threshold,
            "alpha_mode": self.alpha_mode,
            "alpha_risk_threshold": self.alpha_risk_threshold,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AdaptivePolicySpec:
        common = {
            "schema_version",
            "release_risk_threshold",
            "abstention_probability_threshold",
            "alpha_max",
            "alpha_beta",
            "execution_public_key",
            "sparsity",
        }
        version = value.get("schema_version")
        if type(version) is not int:
            raise DataValidationError("adaptive-policy schema version must be an exact integer")
        expected_v1 = common | {
            "layer",
            "site",
            "token_scope",
            "direction_sha256",
            "direction_norm",
        }
        expected_v2 = common | {
            "controller_artifact_sha256",
            "candidate_layers",
            "candidate_sites",
            "candidate_token_scopes",
            "vector_count",
            "likely_unknown_risk_threshold",
            "alpha_mode",
            "alpha_risk_threshold",
        }
        if version == 1 and set(value) == expected_v1:
            if (
                type(value["layer"]) is not int
                or any(
                    isinstance(value[name], bool)
                    or not isinstance(value[name], int | float)
                    for name in (
                        "release_risk_threshold",
                        "abstention_probability_threshold",
                        "alpha_max",
                        "alpha_beta",
                        "direction_norm",
                    )
                )
                or (
                    value["sparsity"] is not None
                    and (
                        isinstance(value["sparsity"], bool)
                        or not isinstance(value["sparsity"], int | float)
                    )
                )
            ):
                raise DataValidationError("adaptive-policy v1 numeric fields have invalid types")
            return cls(
                schema_version=1,
                release_risk_threshold=float(value["release_risk_threshold"]),
                abstention_probability_threshold=float(
                    value["abstention_probability_threshold"]
                ),
                alpha_max=float(value["alpha_max"]),
                alpha_beta=float(value["alpha_beta"]),
                layer=int(value["layer"]),
                site=ActivationSite(value["site"]),
                token_scope=TokenScope(value["token_scope"]),
                direction_sha256=str(value["direction_sha256"]),
                direction_norm=float(value["direction_norm"]),
                execution_public_key=str(value["execution_public_key"]),
                sparsity=float(value["sparsity"]) if value["sparsity"] is not None else None,
            )
        if version != 2 or set(value) != expected_v2:
            raise DataValidationError("adaptive-policy keys differ from its schema version")
        layers = value["candidate_layers"]
        sites = value["candidate_sites"]
        scopes = value["candidate_token_scopes"]
        vector_count = value["vector_count"]
        if (
            not isinstance(layers, list)
            or any(type(item) is not int for item in layers)
            or not isinstance(sites, list)
            or any(not isinstance(item, str) for item in sites)
            or not isinstance(scopes, list)
            or any(not isinstance(item, str) for item in scopes)
            or type(vector_count) is not int
            or type(value["execution_public_key"]) is not str
            or type(value["controller_artifact_sha256"]) is not str
            or type(value["alpha_mode"]) is not str
            or any(
                isinstance(value[name], bool)
                or not isinstance(value[name], int | float)
                for name in (
                    "release_risk_threshold",
                    "abstention_probability_threshold",
                    "alpha_max",
                    "alpha_beta",
                    "likely_unknown_risk_threshold",
                    "alpha_risk_threshold",
                )
            )
            or (
                value["sparsity"] is not None
                and (
                    isinstance(value["sparsity"], bool)
                    or not isinstance(value["sparsity"], int | float)
                )
            )
        ):
            raise DataValidationError("adaptive-policy v2 routed geometry has invalid types")
        return cls(
            schema_version=2,
            release_risk_threshold=float(value["release_risk_threshold"]),
            abstention_probability_threshold=float(value["abstention_probability_threshold"]),
            alpha_max=float(value["alpha_max"]),
            alpha_beta=float(value["alpha_beta"]),
            layer=None,
            site=None,
            token_scope=None,
            direction_sha256=None,
            direction_norm=None,
            execution_public_key=str(value["execution_public_key"]),
            sparsity=float(value["sparsity"]) if value["sparsity"] is not None else None,
            controller_artifact_sha256=str(value["controller_artifact_sha256"]),
            candidate_layers=tuple(layers),
            candidate_sites=tuple(ActivationSite(item) for item in sites),
            candidate_token_scopes=tuple(TokenScope(item) for item in scopes),
            vector_count=vector_count,
            likely_unknown_risk_threshold=float(value["likely_unknown_risk_threshold"]),
            alpha_mode=value["alpha_mode"],
            alpha_risk_threshold=float(value["alpha_risk_threshold"]),
        )


@dataclass(frozen=True, slots=True)
class InterventionSpec:
    method: str
    layer: int | None = None
    site: ActivationSite | None = None
    token_scope: TokenScope | None = None
    alpha: float = 0.0
    sparsity: float | None = None
    vector_artifact: str | None = None
    artifact_sha256: str | None = None
    decay: float | None = None
    adaptive_policy: AdaptivePolicySpec | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", _nonempty(self.method, "method"))
        _finite(self.alpha, "alpha")
        if self.layer is not None and self.layer < 0:
            raise ConfigurationError("layer cannot be negative")
        if self.sparsity is not None and not 0 < self.sparsity <= 1:
            raise ConfigurationError("sparsity must be in (0, 1]")
        if self.decay is not None and self.decay < 0:
            raise ConfigurationError("decay cannot be negative")
        if self.artifact_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.artifact_sha256
        ):
            raise ConfigurationError("intervention artifact_sha256 must be a lowercase SHA-256")
        adaptive = self.method in {"M3", "M6", "ACT-or-SADI"}
        if adaptive != (self.adaptive_policy is not None):
            raise ConfigurationError(
                "M3/M6 interventions require exactly one frozen adaptive policy"
            )


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    """One question under one completely specified inference condition."""

    question_id: str
    benchmark: str
    model_repository: str
    model_revision: str
    runtime: Runtime
    quantization: str
    system_prompt_id: str
    rendered_prompt_hash: str
    steering_method: str
    layer: int | None
    token_scope: TokenScope | None
    alpha: float
    sparsity: float | None
    controller_scores: Mapping[str, float]
    raw_output: str
    normalized_answer: str
    outcome: Outcome
    generation_latency_seconds: float
    input_tokens: int
    output_tokens: int
    condition_id: str
    site: ActivationSite | None = None
    seed: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise DataValidationError(
                f"unsupported generation-record schema version: {self.schema_version}"
            )
        for name in (
            "question_id",
            "benchmark",
            "model_repository",
            "model_revision",
            "quantization",
            "system_prompt_id",
            "rendered_prompt_hash",
            "steering_method",
            "condition_id",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if not _COMMIT_SHA.fullmatch(self.model_revision) and self.runtime is not Runtime.SYNTHETIC:
            raise DataValidationError(
                "non-synthetic generation records require a commit-pinned model"
            )
        _finite(self.alpha, "alpha")
        _finite(self.generation_latency_seconds, "generation_latency_seconds")
        if self.generation_latency_seconds < 0:
            raise DataValidationError("generation latency cannot be negative")
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise DataValidationError("token counts cannot be negative")
        if self.layer is not None and self.layer < 0:
            raise DataValidationError("layer cannot be negative")
        if self.sparsity is not None and not 0 < self.sparsity <= 1:
            raise DataValidationError("sparsity must be in (0, 1]")
        scores = {
            str(key): _finite(float(value), f"controller_scores[{key}]")
            for key, value in self.controller_scores.items()
        }
        object.__setattr__(self, "controller_scores", scores)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("runtime", "token_scope", "outcome", "site"):
            item = value[key]
            if isinstance(item, Enum):
                value[key] = item.value
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GenerationRecord:
        data = dict(value)
        if data.get("schema_version") != 1:
            raise DataValidationError("generation record is missing schema version 1")
        data["runtime"] = Runtime(data["runtime"])
        data["outcome"] = Outcome(data["outcome"])
        if data.get("token_scope") is not None:
            data["token_scope"] = TokenScope(data["token_scope"])
        if data.get("site") is not None:
            data["site"] = ActivationSite(data["site"])
        return cls(**data)

"""Strict YAML configuration loading.

Configuration parsing deliberately rejects unknown keys. Silent typos in an
alpha, layer, model revision, or benchmark split would invalidate paired runs.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar

import yaml

from mfh.contracts import (
    BenchmarkSpec,
    ModelSpec,
    PromptSpec,
    Runtime,
)
from mfh.errors import ConfigurationError
from mfh.provenance import stable_hash

T = TypeVar("T")
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def load_yaml(path: str | Path) -> Mapping[str, Any]:
    source = Path(path)
    try:
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"configuration does not exist: {source}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML in {source}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"configuration root must be a mapping: {source}")
    return value


def _strict_section(
    value: Mapping[str, Any], *, required: set[str], optional: set[str], context: str
) -> dict[str, Any]:
    keys = set(value)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise ConfigurationError(f"{context} is missing required keys: {sorted(missing)}")
    if unknown:
        raise ConfigurationError(f"{context} has unknown keys: {sorted(unknown)}")
    return dict(value)


def _strict_root(value: Mapping[str, Any], section: str, path: str | Path) -> None:
    if value.get("schema_version") != 1:
        raise ConfigurationError(f"{path}: unsupported or missing schema_version")
    unknown = set(value) - {"schema_version", section}
    if unknown:
        raise ConfigurationError(f"{path}: unknown root keys: {sorted(unknown)}")


def load_model_spec(path: str | Path) -> ModelSpec:
    raw = load_yaml(path)
    _strict_root(raw, "model", path)
    section = raw.get("model")
    if not isinstance(section, Mapping):
        raise ConfigurationError(f"{path}: expected a 'model' mapping")
    data = _strict_section(
        section,
        required={"name", "repository", "revision", "runtime", "quantization", "num_layers"},
        optional={
            "dtype",
            "trust_remote_code",
            "role",
            "artifact",
            "artifact_sha256",
            "artifact_size_bytes",
            "candidate_layers",
        },
        context=f"{path}:model",
    )
    try:
        data["runtime"] = Runtime(data["runtime"])
        if "candidate_layers" in data:
            if not isinstance(data["candidate_layers"], list):
                raise ConfigurationError("candidate_layers must be a list")
            data["candidate_layers"] = tuple(int(layer) for layer in data["candidate_layers"])
        return ModelSpec(**data)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ConfigurationError):
            raise
        raise ConfigurationError(f"invalid model configuration {path}: {exc}") from exc


def load_benchmark_spec(path: str | Path) -> BenchmarkSpec:
    raw = load_yaml(path)
    _strict_root(raw, "benchmark", path)
    section = raw.get("benchmark")
    if not isinstance(section, Mapping):
        raise ConfigurationError(f"{path}: expected a 'benchmark' mapping")
    data = _strict_section(
        section,
        required={
            "name",
            "repository",
            "revision",
            "config",
            "split",
            "format",
            "question_column",
            "answer_column",
            "id_column",
        },
        optional=set(),
        context=f"{path}:benchmark",
    )
    try:
        return BenchmarkSpec(**data)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ConfigurationError):
            raise
        raise ConfigurationError(f"invalid benchmark configuration {path}: {exc}") from exc


def load_prompt_specs(path: str | Path) -> tuple[PromptSpec, ...]:
    raw = load_yaml(path)
    if raw.get("schema_version") != 1:
        raise ConfigurationError(f"{path}: unsupported prompt schema version")
    values = raw.get("prompts")
    if not isinstance(values, list) or not values:
        raise ConfigurationError(f"{path}: 'prompts' must be a non-empty list")
    prompts: list[PromptSpec] = []
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise ConfigurationError(f"{path}:prompts[{index}] must be a mapping")
        data = _strict_section(
            value,
            required={"prompt_id", "text"},
            optional={"permits_abstention", "deployment_eligible"},
            context=f"{path}:prompts[{index}]",
        )
        prompts.append(PromptSpec(**data))
    identifiers = [prompt.prompt_id for prompt in prompts]
    if len(set(identifiers)) != len(identifiers):
        raise ConfigurationError(f"{path}: prompt identifiers must be unique")
    return tuple(prompts)


@dataclass(frozen=True, slots=True)
class InferenceProtocol:
    temperature: float
    do_sample: bool
    max_new_tokens: int
    stop_condition: str
    expose_chain_of_thought: bool
    thinking_enabled: bool
    retrieval_enabled: bool
    tools_enabled: bool
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.temperature != 0 or self.do_sample:
            raise ConfigurationError("the primary protocol must use deterministic decoding")
        if not 1 <= self.max_new_tokens <= 48:
            raise ConfigurationError("primary max_new_tokens must be between 1 and 48")
        if self.stop_condition != "eos_or_first_completed_short_answer":
            raise ConfigurationError("primary stop condition differs from the frozen protocol")
        if self.expose_chain_of_thought:
            raise ConfigurationError("the primary protocol cannot expose chain of thought")
        if self.thinking_enabled:
            raise ConfigurationError("thinking must be disabled in the primary protocol")
        if self.retrieval_enabled or self.tools_enabled:
            raise ConfigurationError(
                "closed-book primary experiments cannot enable tools or retrieval"
            )
        if not self.seeds:
            raise ConfigurationError("at least one seed must be configured")
        if len(set(self.seeds)) != len(self.seeds):
            raise ConfigurationError("seeds must be unique")


def load_inference_protocol(path: str | Path) -> InferenceProtocol:
    raw = load_yaml(path)
    if raw.get("schema_version") != 1:
        raise ConfigurationError(f"{path}: unsupported or missing schema_version")
    section = raw.get("inference")
    if not isinstance(section, Mapping):
        raise ConfigurationError(f"{path}: expected an 'inference' mapping")
    data = _strict_section(
        section,
        required={
            "temperature",
            "do_sample",
            "max_new_tokens",
            "stop_condition",
            "expose_chain_of_thought",
            "thinking_enabled",
            "retrieval_enabled",
            "tools_enabled",
            "seeds",
        },
        optional=set(),
        context=f"{path}:inference",
    )
    if not isinstance(data["seeds"], list):
        raise ConfigurationError(f"{path}:inference.seeds must be a list")
    data["seeds"] = tuple(int(seed) for seed in data["seeds"])
    return InferenceProtocol(**data)


@dataclass(frozen=True, slots=True)
class SemanticContaminationProtocol:
    model_repository: str
    model_revision: str
    model_artifact_tree_sha256: str
    required_files: tuple[str, ...]
    pooling: str
    normalize_embeddings: bool
    max_length: int
    embedding_dimension: int
    lexical_ngram_threshold: float
    semantic_similarity_threshold: float
    review_top_k: int
    device: str
    dtype: str
    encode_batch_size: int
    similarity_batch_size: int
    torch_num_threads: int

    def __post_init__(self) -> None:
        if "/" not in self.model_repository or not _COMMIT_SHA.fullmatch(self.model_revision):
            raise ConfigurationError("semantic model requires a repository and commit SHA")
        if not _SHA256.fullmatch(self.model_artifact_tree_sha256):
            raise ConfigurationError("semantic model artifact requires a tree SHA-256")
        if not self.required_files or len(set(self.required_files)) != len(self.required_files):
            raise ConfigurationError("semantic model required_files must be non-empty and unique")
        for filename in self.required_files:
            path = PurePosixPath(filename)
            if path.is_absolute() or ".." in path.parts or path.as_posix() != filename:
                raise ConfigurationError(f"unsafe semantic model filename: {filename!r}")
        if self.pooling != "mean" or not self.normalize_embeddings:
            raise ConfigurationError("semantic overlap requires normalized mean pooling")
        if not 1 <= self.max_length <= 512 or self.embedding_dimension <= 0:
            raise ConfigurationError("semantic embedding shape is invalid")
        if not 0 <= self.lexical_ngram_threshold <= 1:
            raise ConfigurationError("lexical overlap threshold must be in [0, 1]")
        if not 0 <= self.semantic_similarity_threshold <= 1:
            raise ConfigurationError("semantic overlap threshold must be in [0, 1]")
        if self.review_top_k <= 0:
            raise ConfigurationError("semantic review_top_k must be positive")
        if self.device != "cpu" or self.dtype != "float32":
            raise ConfigurationError("semantic preprocessing is frozen to CPU float32")
        if min(self.encode_batch_size, self.similarity_batch_size, self.torch_num_threads) <= 0:
            raise ConfigurationError("semantic batch sizes and thread count must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_repository": self.model_repository,
            "model_revision": self.model_revision,
            "model_artifact_tree_sha256": self.model_artifact_tree_sha256,
            "required_files": list(self.required_files),
            "pooling": self.pooling,
            "normalize_embeddings": self.normalize_embeddings,
            "max_length": self.max_length,
            "embedding_dimension": self.embedding_dimension,
            "lexical_ngram_threshold": self.lexical_ngram_threshold,
            "semantic_similarity_threshold": self.semantic_similarity_threshold,
            "review_top_k": self.review_top_k,
            "device": self.device,
            "dtype": self.dtype,
            "encode_batch_size": self.encode_batch_size,
            "similarity_batch_size": self.similarity_batch_size,
            "torch_num_threads": self.torch_num_threads,
        }

    @property
    def digest(self) -> str:
        return stable_hash({"schema_version": 1, "semantic_contamination": self.to_dict()})


def load_semantic_contamination_protocol(
    path: str | Path,
) -> SemanticContaminationProtocol:
    raw = load_yaml(path)
    _strict_root(raw, "semantic_contamination", path)
    section = raw.get("semantic_contamination")
    if not isinstance(section, Mapping):
        raise ConfigurationError(f"{path}: expected a 'semantic_contamination' mapping")
    fields = {
        "model_repository",
        "model_revision",
        "model_artifact_tree_sha256",
        "required_files",
        "pooling",
        "normalize_embeddings",
        "max_length",
        "embedding_dimension",
        "lexical_ngram_threshold",
        "semantic_similarity_threshold",
        "review_top_k",
        "device",
        "dtype",
        "encode_batch_size",
        "similarity_batch_size",
        "torch_num_threads",
    }
    data = _strict_section(section, required=fields, optional=set(), context=f"{path}:semantic")
    if not isinstance(data["required_files"], list):
        raise ConfigurationError(f"{path}:semantic.required_files must be a list")
    data["required_files"] = tuple(str(value) for value in data["required_files"])
    try:
        return SemanticContaminationProtocol(**data)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ConfigurationError):
            raise
        raise ConfigurationError(
            f"invalid semantic contamination configuration {path}: {exc}"
        ) from exc

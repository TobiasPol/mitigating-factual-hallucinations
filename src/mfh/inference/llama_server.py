"""Pinned, managed llama.cpp server transport for the real E0 GGUF leg."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from mfh.config import load_yaml
from mfh.errors import ConfigurationError, DataValidationError
from mfh.provenance import canonical_json, sha256_file, sha256_path, stable_hash

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HOST = "127.0.0.1"


def _lexical_absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _has_symlink_component(path: Path) -> bool:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            return True
    return False


def _regular_file(path: str | Path, context: str) -> Path:
    source = _lexical_absolute(path)
    if _has_symlink_component(source) or not source.is_file():
        raise DataValidationError(f"{context} must be a regular file")
    return source


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _require_model_sha256(value: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ConfigurationError("llama-server model identity must be SHA-256")
    return value


def _integer(value: Any, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DataValidationError(f"{context} must be an integer >= {minimum}")
    return int(value)


def _number(value: Any, context: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DataValidationError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise DataValidationError(f"{context} must be finite and >= {minimum}")
    return result


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DataValidationError(f"{context} must be an object")
    return value


@dataclass(frozen=True, slots=True)
class LlamaServerProtocol:
    """The exact local-server and decode settings frozen for E0."""

    context_size: int = 2048
    kv_cache_type: str = "q4_0"
    threads: int = 8
    threads_batch: int = 8
    batch_size: int = 512
    ubatch_size: int = 128
    parallel_slots: int = 1
    seed: int = 17
    max_new_tokens: int = 48
    request_timeout_seconds: float = 600.0
    startup_timeout_seconds: float = 180.0

    def __post_init__(self) -> None:
        integer_values = (
            self.context_size,
            self.threads,
            self.threads_batch,
            self.batch_size,
            self.ubatch_size,
            self.parallel_slots,
            self.seed,
            self.max_new_tokens,
        )
        if any(type(value) is not int for value in integer_values):
            raise ConfigurationError("E0 llama-server integer settings require exact int types")
        if type(self.kv_cache_type) is not str:
            raise ConfigurationError("E0 llama-server KV cache type must be text")
        frozen = (
            self.context_size,
            self.kv_cache_type,
            self.threads,
            self.threads_batch,
            self.batch_size,
            self.ubatch_size,
            self.parallel_slots,
            self.seed,
            self.max_new_tokens,
        )
        if frozen != (2048, "q4_0", 8, 8, 512, 128, 1, 17, 48):
            raise ConfigurationError("E0 llama-server protocol differs from its frozen settings")
        for value, name in (
            (self.request_timeout_seconds, "request timeout"),
            (self.startup_timeout_seconds, "startup timeout"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ConfigurationError(f"llama-server {name} must be finite and positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_size": self.context_size,
            "kv_cache_type_k": self.kv_cache_type,
            "kv_cache_type_v": self.kv_cache_type,
            "threads": self.threads,
            "threads_batch": self.threads_batch,
            "batch_size": self.batch_size,
            "ubatch_size": self.ubatch_size,
            "parallel_slots": self.parallel_slots,
            "seed": self.seed,
            "max_new_tokens": self.max_new_tokens,
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "min_p": 0.0,
            "typical_p": 1.0,
            "repeat_penalty": 1.0,
            "samplers": ["temperature"],
            "stop": ["<|im_end|>"],
            "reasoning": "off",
            "reasoning_format": "none",
            "cache_prompt": False,
            "server_prompt_cache_megabytes": 0,
            "gpu_layers": "all",
            "fit": "off",
            "speculative_decoding": False,
            "request_timeout_seconds": float(self.request_timeout_seconds),
            "startup_timeout_seconds": float(self.startup_timeout_seconds),
        }

    def launch_arguments(self, model_path: Path, port: int) -> tuple[str, ...]:
        return (
            "--model",
            str(model_path),
            "--host",
            _HOST,
            "--port",
            str(port),
            "--ctx-size",
            str(self.context_size),
            "--cache-type-k",
            self.kv_cache_type,
            "--cache-type-v",
            self.kv_cache_type,
            "--gpu-layers",
            "all",
            "--fit",
            "off",
            "--threads",
            str(self.threads),
            "--threads-batch",
            str(self.threads_batch),
            "--batch-size",
            str(self.batch_size),
            "--ubatch-size",
            str(self.ubatch_size),
            "--parallel",
            str(self.parallel_slots),
            "--seed",
            str(self.seed),
            "--reasoning",
            "off",
            "--reasoning-format",
            "none",
            "--cache-ram",
            "0",
            "--metrics",
        )

    def completion_request(self, prompt: str) -> dict[str, Any]:
        return {
            "prompt": prompt,
            "n_predict": self.max_new_tokens,
            "temperature": 0.0,
            "seed": self.seed,
            "cache_prompt": False,
            "repeat_penalty": 1.0,
            "top_k": 0,
            "top_p": 1.0,
            "min_p": 0.0,
            "typical_p": 1.0,
            "samplers": ["temperature"],
            "stop": ["<|im_end|>"],
            "return_tokens": True,
        }

    def generation_settings_contract(self) -> dict[str, Any]:
        """Return the normalized response settings attested for every completion."""

        return {
            "seed": self.seed,
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "min_p": 0.0,
            "typical_p": 1.0,
            "repeat_penalty": 1.0,
            "n_predict": self.max_new_tokens,
            "stop": ["<|im_end|>"],
            "samplers": ["temperature"],
            "reasoning_format": "none",
            "speculative.types": "none",
            "backend_sampling": False,
            "lora": [],
        }


def sha256_runtime_tree(path: str | Path) -> str:
    """Hash a runtime tree's files, directories, and contained symlink layout."""

    root = _lexical_absolute(path)
    if _has_symlink_component(root) or not root.is_dir():
        raise DataValidationError("llama-server build tree must be a regular directory")
    resolved_root = root.resolve()
    entries = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
    if not entries:
        raise DataValidationError("llama-server build tree is empty")
    digest = hashlib.sha256()
    for item in entries:
        relative = item.relative_to(root).as_posix().encode()
        if item.is_symlink():
            try:
                target = item.resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise DataValidationError(
                    f"llama-server build tree has an invalid symlink: {item}"
                ) from exc
            if not target.is_relative_to(resolved_root) or not target.is_file():
                raise DataValidationError(
                    "llama-server build-tree symlinks must target contained regular files"
                )
            kind = b"L"
            payload = os.readlink(item).encode()
        elif item.is_file():
            kind = b"F"
            payload = bytes.fromhex(sha256_file(item))
        elif item.is_dir():
            kind = b"D"
            payload = b""
        else:
            raise DataValidationError("llama-server build tree contains a special file")
        digest.update(kind)
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class LlamaServerExpectedIdentity:
    source_repository: str
    source_revision: str
    source_path: Path
    binary_path: Path
    binary_sha256: str
    build_tree_sha256: str
    build_tree_layout_sha256: str
    version_digest: str
    build_info: str
    chat_template_stable_hash: str

    def __post_init__(self) -> None:
        text_values = (
            self.source_repository,
            self.source_revision,
            self.binary_sha256,
            self.build_tree_sha256,
            self.build_tree_layout_sha256,
            self.version_digest,
            self.build_info,
            self.chat_template_stable_hash,
        )
        if any(type(value) is not str for value in text_values):
            raise ConfigurationError("llama-server textual identities require exact string types")
        if (
            not self.source_repository.strip()
            or re.fullmatch(r"[0-9a-f]{40}", self.source_revision) is None
        ):
            raise ConfigurationError("llama-server source identity must be commit-pinned")
        for value in (
            self.binary_sha256,
            self.build_tree_sha256,
            self.build_tree_layout_sha256,
            self.version_digest,
            self.chat_template_stable_hash,
        ):
            if _SHA256.fullmatch(value) is None:
                raise ConfigurationError("llama-server identities must be SHA-256 digests")
        if not self.build_info.strip():
            raise ConfigurationError("llama-server build info must be non-empty")
        source_path = _lexical_absolute(self.source_path)
        binary_path = _lexical_absolute(self.binary_path)
        if _has_symlink_component(source_path) or not source_path.is_dir():
            raise ConfigurationError("llama-server source path must be a regular directory")
        if _has_symlink_component(binary_path) or not binary_path.is_file():
            raise ConfigurationError("llama-server binary path must be a regular file")
        object.__setattr__(self, "source_path", source_path)
        object.__setattr__(self, "binary_path", binary_path)

    def to_dict(self) -> dict[str, str]:
        return {
            "source_repository": self.source_repository,
            "source_revision": self.source_revision,
            "source_path": str(self.source_path),
            "binary_path": str(self.binary_path),
            "binary_sha256": self.binary_sha256,
            "build_tree_sha256": self.build_tree_sha256,
            "build_tree_layout_sha256": self.build_tree_layout_sha256,
            "version_digest": self.version_digest,
            "build_info": self.build_info,
            "chat_template_stable_hash": self.chat_template_stable_hash,
        }


def load_llama_server_identity(path: str | Path) -> LlamaServerExpectedIdentity:
    """Load the exact local runtime receipt without accepting unknown fields."""

    raw = load_yaml(path)
    if raw.get("schema_version") != 1 or set(raw) != {"schema_version", "llama_server"}:
        raise ConfigurationError("llama-server identity config has an invalid root schema")
    section = raw.get("llama_server")
    expected = {
        "source_repository",
        "source_revision",
        "source_path",
        "binary_path",
        "binary_sha256",
        "build_tree_sha256",
        "build_tree_layout_sha256",
        "version_digest",
        "build_info",
        "chat_template_stable_hash",
    }
    if (
        not isinstance(section, Mapping)
        or set(section) != expected
        or any(type(value) is not str for value in section.values())
    ):
        raise ConfigurationError("llama-server identity config fields differ from its schema")
    try:
        config_root = Path(path).absolute().parent
        values = {key: value for key, value in section.items()}
        for key in ("source_path", "binary_path"):
            candidate = Path(values[key])
            values[key] = candidate if candidate.is_absolute() else config_root / candidate
        return LlamaServerExpectedIdentity(
            **values,
        )
    except TypeError as exc:
        raise ConfigurationError(f"invalid llama-server identity config: {exc}") from exc


@dataclass(frozen=True, slots=True)
class LlamaServerCompletion:
    content: str
    token_ids: tuple[int, ...]
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    prompt_milliseconds: float
    generation_milliseconds: float
    stop_type: str
    stopping_word: str
    cache_n: int
    request_digest: str
    generation_settings_digest: str


def _version_output(binary: Path) -> str:
    try:
        result = subprocess.run(
            (str(binary), "--version"),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env={"PATH": os.environ.get("PATH", ""), "LANG": "C", "LC_ALL": "C"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ConfigurationError(f"cannot inspect llama-server version: {exc}") from exc
    output = "\n".join(value.strip() for value in (result.stdout, result.stderr) if value.strip())
    if result.returncode != 0 or not output:
        raise ConfigurationError(
            f"llama-server --version failed with exit code {result.returncode}"
        )
    return output


def _git_output(source: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ("git", "-C", str(source), *arguments),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env={"PATH": os.environ.get("PATH", ""), "LANG": "C", "LC_ALL": "C"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ConfigurationError(f"cannot inspect llama-server source checkout: {exc}") from exc
    if result.returncode != 0:
        raise ConfigurationError("cannot inspect llama-server source checkout")
    return result.stdout.strip()


def _verify_source_checkout(expected: LlamaServerExpectedIdentity) -> dict[str, Any]:
    source = expected.source_path
    if _has_symlink_component(source) or not source.is_dir():
        raise ConfigurationError("llama-server source checkout path changed")
    root = Path(_git_output(source, "rev-parse", "--show-toplevel")).resolve()
    revision = _git_output(source, "rev-parse", "HEAD")
    repository = _git_output(source, "remote", "get-url", "origin")
    status = _git_output(source, "status", "--porcelain")
    if (
        root != source.resolve()
        or revision != expected.source_revision
        or repository != expected.source_repository
        or status
    ):
        raise ConfigurationError("llama-server source checkout differs from its frozen identity")
    return {
        "source_path": str(source),
        "source_repository": repository,
        "source_revision": revision,
        "tracked_and_untracked_clean": True,
    }


def _verify_darwin_runtime_path(binary: Path) -> dict[str, Any]:
    if platform.system() != "Darwin":
        return {"platform": platform.system(), "rpath_check": "not-darwin"}
    try:
        result = subprocess.run(
            ("otool", "-l", str(binary)),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env={"PATH": os.environ.get("PATH", ""), "LANG": "C", "LC_ALL": "C"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ConfigurationError(f"cannot inspect llama-server Mach-O load paths: {exc}") from exc
    if result.returncode != 0:
        raise ConfigurationError("cannot inspect llama-server Mach-O load paths")
    lines = result.stdout.splitlines()
    rpaths: list[str] = []
    for index, line in enumerate(lines):
        if line.strip() == "cmd LC_RPATH" and index + 2 < len(lines):
            match = re.match(r"\s*path\s+(.+?)\s+\(offset", lines[index + 2])
            if match is None:
                raise ConfigurationError("llama-server has an unreadable LC_RPATH")
            rpaths.append(match.group(1))
    if not rpaths or any(Path(value).resolve() != binary.parent for value in rpaths):
        raise ConfigurationError("llama-server LC_RPATH differs from its verified build tree")
    return {"platform": "Darwin", "rpaths": rpaths, "rpath_check": "verified-adjacent-tree"}


def verify_llama_server_artifacts(
    binary_path: str | Path,
    expected: LlamaServerExpectedIdentity,
) -> dict[str, Any]:
    """Verify the executable, adjacent runtime tree, and version output."""

    binary = _regular_file(binary_path, "llama-server binary")
    if binary != expected.binary_path.resolve():
        raise ConfigurationError("llama-server binary path differs from its frozen identity")
    if not os.access(binary, os.X_OK):
        raise ConfigurationError("llama-server binary is not executable")
    if binary.parent.is_symlink() or not binary.parent.is_dir():
        raise DataValidationError("llama-server build tree must be a regular directory")
    binary_digest = sha256_file(binary)
    tree_digest = sha256_path(binary.parent)
    tree_layout_digest = sha256_runtime_tree(binary.parent)
    if binary_digest != expected.binary_sha256:
        raise ConfigurationError("llama-server binary differs from its frozen SHA-256")
    if tree_digest != expected.build_tree_sha256:
        raise ConfigurationError("llama-server build tree differs from its frozen SHA-256")
    if tree_layout_digest != expected.build_tree_layout_sha256:
        raise ConfigurationError("llama-server build-tree layout differs from its frozen SHA-256")
    version_output = _version_output(binary)
    version_digest = stable_hash({"version_output": version_output})
    if version_digest != expected.version_digest:
        raise ConfigurationError("llama-server version differs from its frozen identity")
    if (
        sha256_file(binary) != binary_digest
        or sha256_path(binary.parent) != tree_digest
        or sha256_runtime_tree(binary.parent) != tree_layout_digest
    ):
        raise ConfigurationError("llama-server artifacts changed during identity inspection")
    return {
        "binary_sha256": binary_digest,
        "build_tree_sha256": tree_digest,
        "build_tree_layout_sha256": tree_layout_digest,
        "version_output": version_output,
        "version_digest": version_digest,
        "source_checkout": _verify_source_checkout(expected),
        "dynamic_library_resolution": _verify_darwin_runtime_path(binary),
    }


class LlamaServerClient:
    """Strict JSON client for one local, single-slot llama-server."""

    def __init__(self, *, port: int, protocol: LlamaServerProtocol) -> None:
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ConfigurationError("llama-server port must be in [1, 65535]")
        self.port = port
        self.protocol = protocol
        self.base_url = f"http://{_HOST}:{port}"
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _json(
        self,
        endpoint: str,
        *,
        payload: Mapping[str, Any] | None = None,
        timeout_seconds: float = 30.0,
    ) -> Mapping[str, Any] | list[Any]:
        data = canonical_json(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            f"{self.base_url}{endpoint}",
            data=data,
            headers={"Content-Type": "application/json"} if data is not None else {},
            method="POST" if data is not None else "GET",
        )
        try:
            with self._opener.open(request, timeout=timeout_seconds) as response:
                raw = response.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
            raise DataValidationError(f"llama-server request {endpoint} failed: {exc}") from exc
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DataValidationError(
                f"llama-server request {endpoint} returned invalid JSON"
            ) from exc
        if not isinstance(value, Mapping | list):
            raise DataValidationError(f"llama-server request {endpoint} returned a scalar")
        return value

    def is_healthy(self) -> bool:
        try:
            value = self._json("/health", timeout_seconds=1.0)
        except DataValidationError:
            return False
        return isinstance(value, Mapping) and value.get("status") == "ok"

    def observed_identity(
        self,
        *,
        model_path: Path,
        model_alias: str,
        expected: LlamaServerExpectedIdentity,
    ) -> dict[str, Any]:
        props = _mapping(self._json("/props"), "llama-server properties")
        defaults = _mapping(
            props.get("default_generation_settings"),
            "llama-server default generation settings",
        )
        params = _mapping(defaults.get("params"), "llama-server default parameters")
        modalities = _mapping(props.get("modalities"), "llama-server modalities")
        template = props.get("chat_template")
        total_slots = _integer(props.get("total_slots"), "llama-server total slots")
        context_size = _integer(defaults.get("n_ctx"), "llama-server context size")
        default_seed = _integer(params.get("seed"), "llama-server default seed")
        if (
            total_slots != self.protocol.parallel_slots
            or context_size != self.protocol.context_size
            or default_seed != self.protocol.seed
            or props.get("model_alias") != model_alias
            or props.get("model_path") != str(model_path)
            or props.get("build_info") != expected.build_info
            or modalities.get("vision") is not False
            or modalities.get("audio") is not False
            or props.get("is_sleeping") is not False
            or not isinstance(template, str)
            or not template
        ):
            raise DataValidationError("llama-server properties differ from the E0 contract")
        template_hash = stable_hash(template)
        if template_hash != expected.chat_template_stable_hash:
            raise DataValidationError("llama-server chat template differs from its frozen identity")
        self.assert_idle_single_slot()
        return {
            "model_alias": model_alias,
            "model_path": str(model_path),
            "total_slots": self.protocol.parallel_slots,
            "context_size": self.protocol.context_size,
            "default_seed": self.protocol.seed,
            "build_info": expected.build_info,
            "chat_template_sha256": _sha256_text(template),
            "chat_template_stable_hash": template_hash,
            "chat_template_capabilities_digest": stable_hash(props.get("chat_template_caps", {})),
            "default_generation_settings_digest": stable_hash(defaults),
            "modalities": {"vision": False, "audio": False},
        }

    def assert_idle_single_slot(self) -> None:
        slots = self._json("/slots")
        if not isinstance(slots, list) or len(slots) != 1:
            raise DataValidationError("llama-server must expose exactly one slot")
        slot = _mapping(slots[0], "llama-server slot")
        slot_id = _integer(slot.get("id"), "llama-server slot ID")
        context_size = _integer(slot.get("n_ctx"), "llama-server slot context size")
        if (
            slot_id != 0
            or context_size != self.protocol.context_size
            or slot.get("speculative") is not False
            or slot.get("is_processing") is not False
        ):
            raise DataValidationError("llama-server slot differs from the E0 contract")

    def render_prompt(self, *, system_prompt: str, question: str) -> str:
        if not system_prompt.strip() or not question.strip():
            raise DataValidationError("llama-server messages must be non-empty")
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        value = _mapping(self._json("/apply-template", payload=payload), "template response")
        if set(value) != {"prompt"} or not isinstance(value.get("prompt"), str):
            raise DataValidationError("llama-server template response differs from its schema")
        prompt = str(value["prompt"])
        expected = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        if prompt != expected:
            raise DataValidationError(
                "llama-server rendered prompt differs from E0 peg-native form"
            )
        return prompt

    def complete(self, prompt: str, *, expected_model_alias: str) -> LlamaServerCompletion:
        payload = self.protocol.completion_request(prompt)
        started = time.perf_counter()
        value = _mapping(
            self._json(
                "/completion",
                payload=payload,
                timeout_seconds=self.protocol.request_timeout_seconds,
            ),
            "completion response",
        )
        latency = time.perf_counter() - started
        content = value.get("content")
        tokens = value.get("tokens")
        settings = _mapping(value.get("generation_settings"), "generation settings")
        timings = _mapping(value.get("timings"), "completion timings")
        if not isinstance(content, str) or not isinstance(tokens, list):
            raise DataValidationError("llama-server completion lacks content or token IDs")
        token_ids = tuple(_integer(token, "completion token ID") for token in tokens)
        predicted = _integer(value.get("tokens_predicted"), "predicted token count")
        evaluated = _integer(value.get("tokens_evaluated"), "evaluated token count")
        prompt_n = _integer(timings.get("prompt_n"), "prompt timing token count")
        predicted_n = _integer(timings.get("predicted_n"), "generation timing token count")
        cache_n = _integer(timings.get("cache_n"), "prompt cache count")
        prompt_ms = _number(timings.get("prompt_ms"), "prompt milliseconds")
        predicted_ms = _number(timings.get("predicted_ms"), "generation milliseconds")
        if (
            value.get("model") != expected_model_alias
            or value.get("prompt") != prompt
            or value.get("stop") is not True
            or value.get("truncated") is not False
            or predicted != len(token_ids)
            or predicted_n != predicted
            or prompt_n != evaluated
            or predicted > self.protocol.max_new_tokens
            or cache_n != 0
        ):
            raise DataValidationError("llama-server completion violates the E0 response contract")
        expected_settings = self.protocol.generation_settings_contract()
        if any(
            canonical_json(settings.get(key)) != canonical_json(item)
            for key, item in expected_settings.items()
        ):
            raise DataValidationError("llama-server generation settings differ from the request")
        stop_type = value.get("stop_type")
        stopping_word = value.get("stopping_word")
        if not isinstance(stop_type, str) or not stop_type or not isinstance(stopping_word, str):
            raise DataValidationError("llama-server completion stop metadata is invalid")
        return LlamaServerCompletion(
            content=content,
            token_ids=token_ids,
            input_tokens=evaluated,
            output_tokens=predicted,
            latency_seconds=latency,
            prompt_milliseconds=prompt_ms,
            generation_milliseconds=predicted_ms,
            stop_type=stop_type,
            stopping_word=stopping_word,
            cache_n=cache_n,
            request_digest=stable_hash(payload),
            generation_settings_digest=stable_hash(expected_settings),
        )


def resident_set_size_bytes(pid: int) -> int:
    """Read the live process RSS using the platform ``ps`` contract (KiB)."""

    try:
        result = subprocess.run(
            ("ps", "-o", "rss=", "-p", str(pid)),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env={"PATH": os.environ.get("PATH", ""), "LANG": "C", "LC_ALL": "C"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DataValidationError(f"cannot sample llama-server memory: {exc}") from exc
    raw = result.stdout.strip()
    if result.returncode != 0 or not raw.isdigit():
        raise DataValidationError("cannot sample llama-server resident memory")
    return int(raw) * 1024


class ManagedLlamaServer:
    """Own one exact llama-server child process and attest it before/after use."""

    def __init__(
        self,
        *,
        binary_path: str | Path,
        model_path: str | Path,
        log_path: str | Path,
        expected_identity: LlamaServerExpectedIdentity,
        expected_model_sha256: str,
        expected_model_size_bytes: int,
        protocol: LlamaServerProtocol,
        port: int = 18080,
        memory_sampler: Callable[[int], int] = resident_set_size_bytes,
    ) -> None:
        self.binary_path = _regular_file(binary_path, "llama-server binary")
        self.model_path = _regular_file(model_path, "GGUF model artifact")
        self.log_path = Path(log_path).absolute()
        self.expected_identity = expected_identity
        self.expected_model_sha256 = _require_model_sha256(expected_model_sha256)
        if (
            isinstance(expected_model_size_bytes, bool)
            or not isinstance(expected_model_size_bytes, int)
            or expected_model_size_bytes <= 0
        ):
            raise ConfigurationError("llama-server model size identity must be positive")
        self.expected_model_size_bytes = expected_model_size_bytes
        self.protocol = protocol
        self.client = LlamaServerClient(port=port, protocol=protocol)
        self.port = port
        self.memory_sampler = memory_sampler
        self.command = (
            str(self.binary_path),
            *protocol.launch_arguments(self.model_path, port),
        )
        self.artifact_identity: dict[str, Any] | None = None
        self.server_identity: dict[str, Any] | None = None
        self.process: subprocess.Popen[bytes] | None = None
        self._log_handle: BinaryIO | None = None

    @property
    def pid(self) -> int:
        if self.process is None or self.process.poll() is not None:
            raise DataValidationError("llama-server process is not running")
        return self.process.pid

    def start(self) -> ManagedLlamaServer:
        if self.process is not None:
            raise DataValidationError("llama-server process was already started")
        if self.client.is_healthy():
            raise DataValidationError(f"refusing to reuse an existing server on port {self.port}")
        self.artifact_identity = verify_llama_server_artifacts(
            self.binary_path, self.expected_identity
        )
        self.artifact_identity["model"] = self._verify_model_artifact()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("ab", buffering=0)
        try:
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.DEVNULL,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                env={"PATH": os.environ.get("PATH", ""), "LANG": "C", "LC_ALL": "C"},
                start_new_session=True,
            )
            deadline = time.monotonic() + self.protocol.startup_timeout_seconds
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise DataValidationError(
                        f"llama-server exited during startup with code {self.process.returncode}"
                    )
                if self.client.is_healthy():
                    break
                time.sleep(0.25)
            else:
                raise DataValidationError("llama-server did not become healthy before timeout")
            self.server_identity = self.client.observed_identity(
                model_path=self.model_path,
                model_alias=self.model_path.name,
                expected=self.expected_identity,
            )
            self.sample_memory()
            return self
        except BaseException:
            self.stop(validate_artifacts=False)
            raise

    def sample_memory(self) -> int:
        value = self.memory_sampler(self.pid)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise DataValidationError("llama-server memory sampler returned an invalid value")
        return value

    def _verify_model_artifact(self) -> dict[str, Any]:
        source = _regular_file(self.model_path, "GGUF model artifact")
        if (
            source != self.model_path
            or source.stat().st_size != self.expected_model_size_bytes
            or sha256_file(source) != self.expected_model_sha256
        ):
            raise DataValidationError("GGUF model artifact changed during server execution")
        return {
            "path": str(source),
            "sha256": self.expected_model_sha256,
            "size_bytes": self.expected_model_size_bytes,
        }

    def stop(self, *, validate_artifacts: bool = True) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        if validate_artifacts and self.artifact_identity is not None:
            observed = verify_llama_server_artifacts(self.binary_path, self.expected_identity)
            observed["model"] = self._verify_model_artifact()
            if observed != self.artifact_identity:
                raise ConfigurationError("llama-server artifact identity changed during execution")

    def __enter__(self) -> ManagedLlamaServer:
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()

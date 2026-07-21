"""Immutable manifests and content hashing for confirmatory runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, cast

from mfh.errors import FrozenArtifactError

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_ROOTS = ("src", "configs", "scripts", "tests")
_SOURCE_FILES = ("pyproject.toml", "uv.lock", "docs/research-plan.md")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: str | Path) -> str:
    """Hash a file or a directory tree including relative paths and contents."""

    source = Path(path)
    if source.is_file():
        return sha256_file(source)
    if not source.is_dir():
        raise FrozenArtifactError(f"artifact path does not exist: {source}")
    digest = hashlib.sha256()
    files = sorted(item for item in source.rglob("*") if item.is_file())
    if not files:
        raise FrozenArtifactError(f"artifact directory is empty: {source}")
    for item in files:
        relative = item.relative_to(source).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def _source_tree_sha256(cwd: Path) -> str:
    """Hash executable/configuration sources, including untracked files."""

    digest = hashlib.sha256()
    candidates: list[Path] = []
    for name in _SOURCE_FILES:
        path = cwd / name
        if path.is_file() or path.is_symlink():
            candidates.append(path)
    for name in _SOURCE_ROOTS:
        root = cwd / name
        if root.is_dir():
            candidates.extend(
                path for path in root.rglob("*") if path.is_file() or path.is_symlink()
            )
    for path in sorted(candidates, key=lambda item: item.relative_to(cwd).as_posix()):
        relative = path.relative_to(cwd).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        if path.is_symlink():
            payload = os.readlink(path).encode()
            digest.update(b"symlink\0")
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
            continue
        digest.update(b"file\0")
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _git_state(cwd: Path) -> Mapping[str, Any]:
    def run(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args], cwd=cwd, text=True, capture_output=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    porcelain = run("status", "--porcelain")
    diff = run("diff", "--binary", "--no-ext-diff", "HEAD") if commit else None
    return {
        "commit": commit,
        "dirty": bool(porcelain),
        "status_sha256": stable_hash(porcelain or ""),
        "diff_sha256": stable_hash(diff or ""),
        "source_tree_sha256": _source_tree_sha256(cwd),
    }


def environment_snapshot(packages: tuple[str, ...] = ()) -> Mapping[str, Any]:
    versions: dict[str, str | None] = {}
    for package in sorted(set(packages)):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
    }


@dataclass(frozen=True, slots=True)
class RunManifest:
    study: str
    phase: str
    created_at: str
    config: Mapping[str, Any]
    inputs: Mapping[str, str]
    environment: Mapping[str, Any]
    git: Mapping[str, Any]
    manifest_digest: str
    schema_version: int = 1

    @classmethod
    def create(
        cls,
        *,
        study: str,
        phase: str,
        config: Mapping[str, Any],
        inputs: Mapping[str, str],
        cwd: str | Path,
        packages: tuple[str, ...] = (),
        created_at: str | None = None,
        require_clean_git: bool = False,
    ) -> RunManifest:
        timestamp = created_at or datetime.now(UTC).isoformat()
        normalized_config = dict(config)
        normalized_inputs = dict(sorted(inputs.items()))
        invalid_inputs = {
            name: digest
            for name, digest in normalized_inputs.items()
            if not _SHA256.fullmatch(digest)
        }
        if invalid_inputs:
            raise FrozenArtifactError(
                f"manifest input fingerprints must be SHA-256 digests: {sorted(invalid_inputs)}"
            )
        environment = environment_snapshot(packages)
        git = _git_state(Path(cwd))
        if require_clean_git and (git["commit"] is None or git["dirty"]):
            raise FrozenArtifactError(
                "confirmatory manifests require a committed, clean Git worktree"
            )
        body = {
            "study": study,
            "phase": phase,
            "created_at": timestamp,
            "config": normalized_config,
            "inputs": normalized_inputs,
            "environment": environment,
            "git": git,
            "schema_version": 1,
        }
        return cls(
            study=study,
            phase=phase,
            created_at=timestamp,
            config=normalized_config,
            inputs=normalized_inputs,
            environment=environment,
            git=git,
            manifest_digest=stable_hash(body),
        )

    def verify(self) -> None:
        if self.schema_version != 1:
            raise FrozenArtifactError(f"unsupported manifest schema version: {self.schema_version}")
        invalid_inputs = {
            name: digest for name, digest in self.inputs.items() if not _SHA256.fullmatch(digest)
        }
        if invalid_inputs:
            raise FrozenArtifactError(
                f"manifest contains invalid input fingerprints: {sorted(invalid_inputs)}"
            )
        body = asdict(self)
        actual = body.pop("manifest_digest")
        expected = stable_hash(body)
        if actual != expected:
            raise FrozenArtifactError(
                f"manifest digest mismatch: expected {expected}, found {actual}"
            )

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], _jsonable(asdict(self)))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunManifest:
        required = {
            "study",
            "phase",
            "created_at",
            "config",
            "inputs",
            "environment",
            "git",
            "manifest_digest",
            "schema_version",
        }
        if set(value) != required:
            raise FrozenArtifactError(
                f"manifest keys differ from schema: missing={sorted(required - set(value))}, "
                f"unknown={sorted(set(value) - required)}"
            )
        try:
            manifest = cls(
                study=str(value["study"]),
                phase=str(value["phase"]),
                created_at=str(value["created_at"]),
                config=cast(Mapping[str, Any], value["config"]),
                inputs=cast(Mapping[str, str], value["inputs"]),
                environment=cast(Mapping[str, Any], value["environment"]),
                git=cast(Mapping[str, Any], value["git"]),
                manifest_digest=str(value["manifest_digest"]),
                schema_version=int(value["schema_version"]),
            )
        except (TypeError, ValueError) as exc:
            raise FrozenArtifactError(f"invalid manifest field type: {exc}") from exc
        manifest.verify()
        return manifest


def write_frozen_manifest(path: str | Path, manifest: RunManifest) -> None:
    """Create a manifest once; identical retries are idempotent."""

    manifest.verify()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        if destination.read_bytes() == payload:
            return
        raise FrozenArtifactError(f"refusing to overwrite frozen manifest: {destination}") from None
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def read_frozen_manifest(path: str | Path) -> RunManifest:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read frozen manifest {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise FrozenArtifactError(f"manifest root must be a mapping: {path}")
    return RunManifest.from_dict(value)

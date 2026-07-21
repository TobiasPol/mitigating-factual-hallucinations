"""Exact Hugging Face snapshot receipts for Transformer research checkpoints."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from mfh.contracts import ModelSpec, Runtime
from mfh.errors import ConfigurationError, DataValidationError
from mfh.provenance import sha256_file, stable_hash

_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ALGORITHMS = {"git-blob-sha1", "sha256"}


def reject_symlink_path_components(path: str | Path, context: str) -> Path:
    """Return a lexical absolute path after rejecting every existing symlink component."""

    absolute = Path(path).absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise DataValidationError(f"{context} cannot traverse symlinks: {current}")
        if not current.exists():
            break
    return absolute


@dataclass(frozen=True, slots=True)
class SnapshotFile:
    """One immutable file declared by a pinned Hub revision."""

    path: str
    size_bytes: int
    digest_algorithm: str
    digest: str

    def __post_init__(self) -> None:
        pure = PurePosixPath(self.path)
        if (
            not self.path
            or pure.is_absolute()
            or ".." in pure.parts
            or "." in pure.parts
            or pure.as_posix() != self.path
        ):
            raise ConfigurationError(f"invalid snapshot path: {self.path!r}")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ConfigurationError("snapshot sizes must be non-negative integers")
        if self.digest_algorithm not in _ALGORITHMS:
            raise ConfigurationError(
                f"unsupported snapshot digest algorithm: {self.digest_algorithm!r}"
            )
        pattern = _SHA1 if self.digest_algorithm == "git-blob-sha1" else _SHA256
        if pattern.fullmatch(self.digest) is None:
            raise ConfigurationError(
                f"snapshot digest has the wrong shape for {self.digest_algorithm}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "digest_algorithm": self.digest_algorithm,
            "digest": self.digest,
        }


@dataclass(frozen=True, slots=True)
class SnapshotManifest:
    """Trusted expected inventory obtained from an exact Hub commit."""

    repository: str
    revision: str
    files: tuple[SnapshotFile, ...]
    manifest_path: Path
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ConfigurationError("snapshot manifest requires schema version 1")
        if not self.repository.strip() or _SHA1.fullmatch(self.revision) is None:
            raise ConfigurationError("snapshot repository or revision is invalid")
        if not self.files:
            raise ConfigurationError("snapshot manifest must declare files")
        paths = [value.path for value in self.files]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ConfigurationError("snapshot files must be unique and path-sorted")

    @property
    def total_size_bytes(self) -> int:
        return sum(value.size_bytes for value in self.files)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "repository": self.repository,
            "revision": self.revision,
            "file_count": len(self.files),
            "total_size_bytes": self.total_size_bytes,
            "files": [value.to_dict() for value in self.files],
        }


def load_snapshot_manifest(
    path: str | Path, *, model_spec: ModelSpec | None = None
) -> SnapshotManifest:
    """Load a strict snapshot manifest and optionally bind it to a model spec."""

    source = reject_symlink_path_components(path, "snapshot manifest")
    if not source.is_file():
        raise ConfigurationError(f"snapshot manifest is not a regular file: {source}")
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot read snapshot manifest {source}: {exc}") from exc
    if not isinstance(raw, Mapping) or set(raw) != {
        "schema_version",
        "repository",
        "revision",
        "file_count",
        "total_size_bytes",
        "files",
    }:
        raise ConfigurationError("snapshot manifest fields differ from schema version 1")
    rows = raw.get("files")
    if not isinstance(rows, list):
        raise ConfigurationError("snapshot manifest files must be a list")
    files: list[SnapshotFile] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {
            "path",
            "size_bytes",
            "digest_algorithm",
            "digest",
        }:
            raise ConfigurationError(f"snapshot file {index} differs from schema version 1")
        if (
            type(row.get("path")) is not str
            or type(row.get("size_bytes")) is not int
            or type(row.get("digest_algorithm")) is not str
            or type(row.get("digest")) is not str
        ):
            raise ConfigurationError(f"snapshot file {index} has invalid field types")
        files.append(
            SnapshotFile(
                path=str(row["path"]),
                size_bytes=int(row["size_bytes"]),
                digest_algorithm=str(row["digest_algorithm"]),
                digest=str(row["digest"]),
            )
        )
    manifest = SnapshotManifest(
        schema_version=int(raw["schema_version"]),
        repository=str(raw["repository"]),
        revision=str(raw["revision"]),
        files=tuple(files),
        manifest_path=source,
    )
    if (
        type(raw.get("file_count")) is not int
        or raw["file_count"] != len(manifest.files)
        or type(raw.get("total_size_bytes")) is not int
        or raw["total_size_bytes"] != manifest.total_size_bytes
    ):
        raise ConfigurationError("snapshot manifest totals differ from its files")
    if model_spec is not None and (
        model_spec.runtime is not Runtime.MLX
        or manifest.repository != model_spec.repository
        or manifest.revision != model_spec.revision
    ):
        raise ConfigurationError("snapshot manifest differs from the MLX model spec")
    return manifest


def _git_blob_sha1(path: Path, size: int) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {size}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_regular_file(path: Path, expected: SnapshotFile) -> None:
    if path.is_symlink() or not path.is_file():
        raise DataValidationError(f"snapshot artifact must be a regular file: {expected.path}")
    if path.stat().st_size != expected.size_bytes:
        raise DataValidationError(f"snapshot artifact size differs: {expected.path}")
    observed = (
        sha256_file(path)
        if expected.digest_algorithm == "sha256"
        else _git_blob_sha1(path, expected.size_bytes)
    )
    if observed != expected.digest:
        raise DataValidationError(f"snapshot artifact digest differs: {expected.path}")


def verify_transformers_snapshot(
    model_spec: ModelSpec,
    snapshot_directory: str | Path,
    manifest_path: str | Path,
) -> Mapping[str, Any]:
    """Hash and verify an exact, symlink-free local Hub model snapshot."""

    manifest = load_snapshot_manifest(manifest_path, model_spec=model_spec)
    root = reject_symlink_path_components(snapshot_directory, "Transformer snapshot")
    if root.is_symlink() or not root.is_dir():
        raise DataValidationError("Transformer snapshot must be a regular directory")
    expected_files = {value.path: value for value in manifest.files}
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for item in root.rglob("*"):
        if item.is_symlink():
            raise DataValidationError("Transformer snapshot cannot contain symlinks")
        relative = item.relative_to(root).as_posix()
        if item.is_file():
            observed_files.add(relative)
        elif item.is_dir():
            observed_directories.add(relative)
        else:
            raise DataValidationError("Transformer snapshot contains a special file")
    expected_directories = {
        PurePosixPath(path).parent.as_posix()
        for path in expected_files
        if PurePosixPath(path).parent.as_posix() != "."
    }
    if observed_files != set(expected_files) or observed_directories != expected_directories:
        raise DataValidationError(
            "Transformer snapshot inventory differs from the pinned Hub revision"
        )
    for name, expected in expected_files.items():
        _verify_regular_file(root / name, expected)
    body: dict[str, Any] = {
        "schema_version": 1,
        "repository": manifest.repository,
        "revision": manifest.revision,
        "snapshot_path": str(root),
        "manifest_path": str(manifest.manifest_path),
        "manifest_sha256": sha256_file(manifest.manifest_path),
        "file_count": len(manifest.files),
        "total_size_bytes": manifest.total_size_bytes,
        "files": [value.to_dict() for value in manifest.files],
    }
    return {**body, "snapshot_digest": stable_hash(body)}

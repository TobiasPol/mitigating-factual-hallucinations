"""Dependency-neutral filesystem boundary for the active scientific study."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from mfh.errors import ConfigurationError

QWEN_STUDY_NAMESPACE = "qwen36-27b-mlx4-m4max48-v1"
QWEN_STUDY_ARTIFACT_ROOT = f"artifacts/studies/{QWEN_STUDY_NAMESPACE}"


def validate_active_study_artifact_paths(
    paths: Mapping[str, str | Path],
    *,
    project_root: str | Path | None = None,
) -> Mapping[str, Path]:
    """Require mutable scientific artifacts to use the active Qwen namespace."""

    root = (
        Path(project_root).absolute().resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    namespace = root / QWEN_STUDY_ARTIFACT_ROOT

    def reject_symlink_components(path: Path, *, label: str) -> None:
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise ConfigurationError(f"{label} must stay inside the project") from exc
        cursor = root
        for component in relative.parts:
            cursor /= component
            if cursor.is_symlink():
                raise ConfigurationError(
                    f"{label} cannot traverse symlink component {cursor}"
                )

    reject_symlink_components(namespace, label="active-study namespace")
    canonical_namespace = namespace.resolve(strict=False)
    if canonical_namespace != namespace:
        raise ConfigurationError("active-study namespace is not canonical")
    if not paths:
        raise ConfigurationError("active-study artifact path set cannot be empty")
    normalized: dict[str, Path] = {}
    for name, raw_path in paths.items():
        if type(name) is not str or not name.strip():
            raise ConfigurationError("active-study artifact path name is invalid")
        raw = Path(raw_path)
        if ".." in raw.parts:
            raise ConfigurationError(f"{name} cannot contain parent traversal")
        lexical = Path(os.path.abspath(raw))
        if lexical == namespace or not lexical.is_relative_to(namespace):
            raise ConfigurationError(
                f"{name} must stay inside {QWEN_STUDY_ARTIFACT_ROOT}"
            )
        reject_symlink_components(lexical, label=name)
        path = lexical.resolve(strict=False)
        if path == canonical_namespace or not path.is_relative_to(canonical_namespace):
            raise ConfigurationError(
                f"{name} must stay inside {QWEN_STUDY_ARTIFACT_ROOT}"
            )
        normalized[name] = path
    return MappingProxyType(normalized)

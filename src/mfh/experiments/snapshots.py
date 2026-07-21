"""Immutable execution-code snapshots for E9 preregistration and E10 freezing."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mfh.analysis.protocol import AnalysisProtocol, load_analysis_protocol
from mfh.artifact_namespace import validate_active_study_artifact_paths
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.experiments.protocol import ExperimentPhase
from mfh.provenance import sha256_file, sha256_path, stable_hash

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ROLE = re.compile(r"^[a-z][a-z0-9_]*$")
_SOURCE_PATHS = {
    "analysis_protocol": "configs/analysis/confirmatory.yaml",
    "experiment_protocol": "src/mfh/experiments/protocol.py",
    "gate_evaluator": "src/mfh/experiments/gates.py",
    "generation_contracts": "src/mfh/contracts.py",
    "language_evaluator": "src/mfh/evaluation/language.py",
    "metric_evaluator": "src/mfh/evaluation/metrics.py",
    "official_grading": "src/mfh/evaluation/official.py",
    "reporting": "src/mfh/analysis/reporting.py",
    "research_plan": "docs/research-plan.md",
    "robustness_diagnostic_config": "configs/experiments/robustness-diagnostics.json",
    "runner": "src/mfh/experiments/runner.py",
    "safety_evaluator": "src/mfh/evaluation/side_effects.py",
    "statistics": "src/mfh/analysis/statistics.py",
    "study_protocol_config": "configs/experiments/phases.yaml",
}
_REQUIRED_ROLES = frozenset(_SOURCE_PATHS)


def _repository_root(repository_root: str | Path | None) -> Path:
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[3]
    )
    if not (root / "docs" / "research-plan.md").is_file() or not (root / "src" / "mfh").is_dir():
        raise DataValidationError("execution snapshot repository root is invalid")
    return root


def execution_snapshot_sources(
    repository_root: str | Path | None = None,
) -> Mapping[str, Path]:
    """Return the only role-to-source mapping accepted for confirmatory snapshots."""

    root = _repository_root(repository_root)
    return {role: root / relative for role, relative in _SOURCE_PATHS.items()}


def _package_sources(repository_root: Path) -> Mapping[str, Path]:
    """Return every Python source that can participate in an evaluation import."""

    package_root = repository_root / "src" / "mfh"
    return {
        path.relative_to(repository_root).as_posix(): path
        for path in sorted(package_root.rglob("*.py"))
    }


def _kind(phase: ExperimentPhase) -> str:
    if phase is ExperimentPhase.E9:
        return "preregistered-analysis"
    if phase is ExperimentPhase.E10:
        return "frozen-evaluation-runtime"
    raise DataValidationError("execution snapshots are defined only for E9 and E10")


def _manifest(path: Path) -> dict[str, Any]:
    manifest_path = path / "snapshot-manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise DataValidationError("execution snapshot manifest must be a regular file")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot read execution snapshot manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise DataValidationError("execution snapshot manifest must be a mapping")
    digest = payload.pop("manifest_digest", None)
    if digest != stable_hash(payload):
        raise DataValidationError("execution snapshot manifest digest mismatch")
    return payload


def load_snapshot_analysis_protocol(
    path: str | Path,
    *,
    study_protocol_digest: str,
    phase: ExperimentPhase,
) -> tuple[AnalysisProtocol, str]:
    """Load the exact protocol and research plan from a verified snapshot."""

    source = Path(path)
    manifest = validate_execution_snapshot(
        source,
        study_protocol_digest=study_protocol_digest,
        phase=phase,
    )
    files = manifest.get("files")
    if not isinstance(files, Mapping):  # pragma: no cover - validated above
        raise FrozenArtifactError("execution snapshot source descriptors are invalid")
    analysis = files.get("analysis_protocol")
    plan = files.get("research_plan")
    if not isinstance(analysis, Mapping) or not isinstance(plan, Mapping):
        raise FrozenArtifactError("execution snapshot lacks analysis/plan descriptors")
    protocol = load_analysis_protocol(source / str(analysis["path"]))
    protocol.verify_research_plan(source / str(plan["path"]))
    return protocol, sha256_path(source)


def validate_execution_snapshot(
    path: str | Path,
    *,
    study_protocol_digest: str,
    phase: ExperimentPhase,
    repository_root: str | Path | None = None,
) -> Mapping[str, Any]:
    """Verify a closed code snapshot and its research-plan/analysis bindings."""

    source = Path(path)
    if source.is_symlink() or not source.is_dir():
        raise DataValidationError("execution snapshot must be a regular directory")
    if {item.name for item in source.iterdir()} != {
        "snapshot-manifest.json",
        "files",
        "package",
    }:
        raise DataValidationError("execution snapshot has invalid top-level files")
    if any(item.is_symlink() for item in source.iterdir()):
        raise DataValidationError("execution snapshot contains a linked top-level path")
    payload = _manifest(source)
    if (
        set(payload)
        != {
            "schema_version",
            "snapshot_kind",
            "study_protocol_digest",
            "phase",
            "research_plan_sha256",
            "analysis_protocol_sha256",
            "files",
            "package_sources",
        }
        or payload.get("schema_version") != 2
    ):
        raise DataValidationError("execution snapshot manifest has an invalid schema")
    if (
        payload["snapshot_kind"] != _kind(phase)
        or payload["study_protocol_digest"] != study_protocol_digest
        or payload["phase"] != phase.value
        or not _SHA256.fullmatch(str(payload["research_plan_sha256"]))
        or not _SHA256.fullmatch(str(payload["analysis_protocol_sha256"]))
    ):
        raise DataValidationError("execution snapshot identity differs from the run")
    files = payload["files"]
    if not isinstance(files, Mapping) or set(files) != _REQUIRED_ROLES:
        raise DataValidationError("execution snapshot omits a required derivation source")
    observed_names: set[str] = set()
    observed_hashes: dict[str, str] = {}
    live_sources = execution_snapshot_sources(repository_root)
    for role, descriptor in files.items():
        if (
            not isinstance(role, str)
            or not _ROLE.fullmatch(role)
            or not isinstance(descriptor, Mapping)
            or set(descriptor) != {"path", "sha256", "source_path"}
        ):
            raise DataValidationError("execution snapshot file descriptor is invalid")
        relative = descriptor["path"]
        expected_hash = descriptor["sha256"]
        source_path = descriptor["source_path"]
        if (
            not isinstance(relative, str)
            or not relative.startswith("files/")
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(expected_hash, str)
            or not _SHA256.fullmatch(expected_hash)
            or source_path != _SOURCE_PATHS[role]
        ):
            raise DataValidationError("execution snapshot file identity is invalid")
        candidate = source / relative
        if candidate.is_symlink() or not candidate.is_file():
            raise DataValidationError("execution snapshot contains a missing or linked file")
        if sha256_file(candidate) != expected_hash:
            raise DataValidationError("execution snapshot source bytes changed")
        live_source = live_sources[role]
        if live_source.is_symlink() or not live_source.is_file():
            raise DataValidationError("live execution source is missing or linked")
        if sha256_file(live_source) != expected_hash:
            raise DataValidationError("execution snapshot differs from the live derivation source")
        observed_names.add(candidate.name)
        observed_hashes[role] = expected_hash
    files_root = source / "files"
    if (
        files_root.is_symlink()
        or not files_root.is_dir()
        or {item.name for item in files_root.iterdir()} != observed_names
    ):
        raise DataValidationError("execution snapshot contains undeclared source files")
    if (
        observed_hashes["research_plan"] != payload["research_plan_sha256"]
        or observed_hashes["analysis_protocol"] != payload["analysis_protocol_sha256"]
    ):
        raise DataValidationError("execution snapshot plan bindings are inconsistent")

    package_sources = payload["package_sources"]
    live_package = _package_sources(_repository_root(repository_root))
    if not isinstance(package_sources, Mapping) or set(package_sources) != set(live_package):
        raise DataValidationError("execution snapshot package sources differ from the live package")
    declared_package_files: set[str] = set()
    for source_path, descriptor in package_sources.items():
        if (
            not isinstance(source_path, str)
            or not source_path.startswith("src/mfh/")
            or not source_path.endswith(".py")
            or Path(source_path).is_absolute()
            or ".." in Path(source_path).parts
            or not isinstance(descriptor, Mapping)
            or set(descriptor) != {"path", "sha256"}
        ):
            raise DataValidationError("execution snapshot package descriptor is invalid")
        packaged_path = descriptor["path"]
        expected_hash = descriptor["sha256"]
        expected_packaged_path = f"package/{source_path}"
        if (
            packaged_path != expected_packaged_path
            or not isinstance(expected_hash, str)
            or not _SHA256.fullmatch(expected_hash)
        ):
            raise DataValidationError("execution snapshot package identity is invalid")
        packaged = source / expected_packaged_path
        live = live_package[source_path]
        if (
            packaged.is_symlink()
            or not packaged.is_file()
            or live.is_symlink()
            or not live.is_file()
        ):
            raise DataValidationError("execution snapshot package source is missing or linked")
        if sha256_file(packaged) != expected_hash:
            raise DataValidationError("execution snapshot package bytes changed")
        if sha256_file(live) != expected_hash:
            raise DataValidationError("execution snapshot differs from a live package source")
        declared_package_files.add(source_path)

    package_root = source / "package"
    if package_root.is_symlink() or not package_root.is_dir():
        raise DataValidationError("execution snapshot package must be a regular directory")
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for item in package_root.rglob("*"):
        if item.is_symlink():
            raise DataValidationError("execution snapshot package contains a linked path")
        relative = item.relative_to(package_root).as_posix()
        if item.is_file():
            actual_files.add(relative)
        elif item.is_dir():
            actual_directories.add(relative)
        else:
            raise DataValidationError("execution snapshot package contains a special file")
    expected_directories = {
        parent.as_posix()
        for relative in declared_package_files
        for parent in Path(relative).parents
        if parent != Path(".")
    }
    if actual_files != declared_package_files or actual_directories != expected_directories:
        raise DataValidationError("execution snapshot package contains undeclared paths")
    return payload


def write_execution_snapshot(
    destination: str | Path,
    *,
    study_protocol_digest: str,
    phase: ExperimentPhase,
    sources: Mapping[str, str | Path] | None = None,
    repository_root: str | Path | None = None,
) -> str:
    """Atomically package every source that can change confirmatory derivations."""

    if not _SHA256.fullmatch(study_protocol_digest):
        raise DataValidationError("execution snapshot requires a study-protocol SHA-256")
    target = validate_active_study_artifact_paths({"confirmatory execution snapshot": destination})[
        "confirmatory execution snapshot"
    ]
    expected_sources = execution_snapshot_sources(repository_root)
    selected_sources = expected_sources if sources is None else sources
    if set(selected_sources) != _REQUIRED_ROLES:
        raise DataValidationError("execution snapshot source roles differ from the schema")
    if any(
        Path(selected_sources[role]).resolve() != expected_sources[role].resolve()
        for role in _REQUIRED_ROLES
    ):
        raise DataValidationError("execution snapshot roles are not bound to repository sources")
    if target.exists():
        raise FrozenArtifactError(f"refusing to overwrite execution snapshot: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=target.parent))
    try:
        files_root = stage / "files"
        files_root.mkdir()
        descriptors: dict[str, dict[str, str]] = {}
        for role in sorted(selected_sources):
            source = Path(selected_sources[role]).resolve()
            if source.is_symlink() or not source.is_file():
                raise DataValidationError(f"execution snapshot source {role!r} is not a file")
            suffix = "".join(source.suffixes)
            filename = f"{role}{suffix}"
            packaged = files_root / filename
            shutil.copyfile(source, packaged)
            descriptors[role] = {
                "path": f"files/{filename}",
                "sha256": sha256_file(packaged),
                "source_path": _SOURCE_PATHS[role],
            }
        root = _repository_root(repository_root)
        package_descriptors: dict[str, dict[str, str]] = {}
        for relative, source in _package_sources(root).items():
            if source.is_symlink() or not source.is_file():
                raise DataValidationError("execution snapshot package source is not a file")
            packaged = stage / "package" / relative
            packaged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, packaged)
            package_descriptors[relative] = {
                "path": f"package/{relative}",
                "sha256": sha256_file(packaged),
            }
        body = {
            "schema_version": 2,
            "snapshot_kind": _kind(phase),
            "study_protocol_digest": study_protocol_digest,
            "phase": phase.value,
            "research_plan_sha256": descriptors["research_plan"]["sha256"],
            "analysis_protocol_sha256": descriptors["analysis_protocol"]["sha256"],
            "files": descriptors,
            "package_sources": package_descriptors,
        }
        (stage / "snapshot-manifest.json").write_text(
            json.dumps({**body, "manifest_digest": stable_hash(body)}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        validate_execution_snapshot(
            stage,
            study_protocol_digest=study_protocol_digest,
            phase=phase,
            repository_root=repository_root,
        )
        os.replace(stage, target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return sha256_path(target)

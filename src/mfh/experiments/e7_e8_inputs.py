"""Atomic staging of immutable external inputs shared by E7 and E8."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from mfh.data.language_suite import load_reviewed_language_suite
from mfh.data.reviewed_splits import validate_reviewed_split_snapshot
from mfh.data.source_snapshots import SOURCE_SNAPSHOTS, verify_source_artifact
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.evaluation.ifeval import validate_ifeval_evaluator
from mfh.experiments.model_selection import validate_active_study_artifact_paths
from mfh.inference.transformers_snapshot import reject_symlink_path_components
from mfh.provenance import sha256_path, stable_hash

_SOURCE_FILENAMES = {
    "triviaqa": "triviaqa.parquet",
    "ifeval": "ifeval.jsonl",
    "mmlu_pro": "mmlu_pro.parquet",
    "wikitext103": "wikitext103.parquet",
    "xstest": "xstest.csv",
    "strongreject_or_harmbench": "strongreject_or_harmbench.csv",
}
_TOP_LEVEL = {
    "manifest.json",
    "reviewed-splits",
    "language-suite",
    "ifeval-evaluator",
    "sources",
}


def _require_digest(value: str, context: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise DataValidationError(f"{context} must be a lowercase SHA-256 digest")


def _source_paths(root: Path) -> Mapping[str, Path]:
    return MappingProxyType(
        {name: root / "sources" / filename for name, filename in _SOURCE_FILENAMES.items()}
    )


def _validate_materials(
    root: Path,
    *,
    expected_reviewed_split_manifest_digest: str,
) -> Mapping[str, Any]:
    if (
        root.is_symlink()
        or not root.is_dir()
        or {item.name for item in root.iterdir()} != _TOP_LEVEL
    ):
        raise FrozenArtifactError("E7/E8 staged-input inventory differs")
    if any(item.is_symlink() for item in root.rglob("*")):
        raise FrozenArtifactError("E7/E8 staged inputs cannot contain symbolic links")
    source_paths = _source_paths(root)
    if (
        not (root / "manifest.json").is_file()
        or not (root / "reviewed-splits").is_dir()
        or not (root / "language-suite").is_dir()
        or not (root / "ifeval-evaluator").is_dir()
        or not (root / "sources").is_dir()
        or {item.name for item in (root / "sources").iterdir()}
        != set(_SOURCE_FILENAMES.values())
        or any(not path.is_file() for path in source_paths.values())
    ):
        raise FrozenArtifactError("E7/E8 staged-input artifact types differ")

    reviewed_manifest = validate_reviewed_split_snapshot(root / "reviewed-splits")
    if reviewed_manifest.get("manifest_digest") != expected_reviewed_split_manifest_digest:
        raise FrozenArtifactError("E7/E8 reviewed-split manifest differs")
    language_questions = load_reviewed_language_suite(root / "language-suite")
    if len(language_questions) != 500:
        raise DataValidationError("E7/E8 language suite must contain exactly 500 questions")
    evaluator_sha256 = validate_ifeval_evaluator(root / "ifeval-evaluator")
    verified_sources = {
        name: verify_source_artifact(SOURCE_SNAPSHOTS[name], path)
        for name, path in source_paths.items()
    }
    body = {
        "schema_version": 1,
        "purpose": "e7-e8-external-input-snapshot",
        "reviewed_split_manifest_digest": expected_reviewed_split_manifest_digest,
        "reviewed_splits_sha256": sha256_path(root / "reviewed-splits"),
        "language_suite_sha256": sha256_path(root / "language-suite"),
        "language_question_count": len(language_questions),
        "ifeval_evaluator_sha256": evaluator_sha256,
        "sources": {
            name: {
                "path": f"sources/{_SOURCE_FILENAMES[name]}",
                "sha256": SOURCE_SNAPSHOTS[name].artifact_sha256,
                "size_bytes": SOURCE_SNAPSHOTS[name].artifact_size_bytes,
            }
            for name in sorted(verified_sources)
        },
    }
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read E7/E8 staged-input manifest: {exc}") from exc
    if manifest != {**body, "manifest_digest": stable_hash(body)}:
        raise FrozenArtifactError("E7/E8 staged-input manifest differs from live contents")
    return MappingProxyType(
        {
            "valid": True,
            "directory": str(root),
            "manifest_digest": manifest["manifest_digest"],
            "sha256": sha256_path(root),
            "reviewed_splits": str(root / "reviewed-splits"),
            "reviewed_language_suite": str(root / "language-suite"),
            "ifeval_evaluator": str(root / "ifeval-evaluator"),
            "source_artifacts": {
                name: str(path) for name, path in sorted(source_paths.items())
            },
        }
    )


def validate_e7_e8_external_inputs(
    directory: str | Path,
    *,
    expected_reviewed_split_manifest_digest: str,
) -> Mapping[str, Any]:
    """Replay every byte and manifest in an existing staged E7/E8 snapshot."""

    _require_digest(
        expected_reviewed_split_manifest_digest,
        "expected reviewed-split manifest digest",
    )
    root = validate_active_study_artifact_paths({"E7/E8 staged inputs": directory})[
        "E7/E8 staged inputs"
    ]
    return _validate_materials(
        root,
        expected_reviewed_split_manifest_digest=expected_reviewed_split_manifest_digest,
    )


def stage_e7_e8_external_inputs(
    directory: str | Path,
    *,
    reviewed_splits: str | Path,
    expected_reviewed_split_manifest_digest: str,
    reviewed_language_suite: str | Path,
    ifeval_evaluator: str | Path,
    source_artifacts: Mapping[str, str | Path],
) -> Mapping[str, Any]:
    """Verify and atomically copy every external E7/E8 input into the study namespace."""

    _require_digest(
        expected_reviewed_split_manifest_digest,
        "expected reviewed-split manifest digest",
    )
    if set(source_artifacts) != set(_SOURCE_FILENAMES):
        raise DataValidationError("E7/E8 source-artifact inventory differs")
    destination = validate_active_study_artifact_paths({"E7/E8 staged inputs": directory})[
        "E7/E8 staged inputs"
    ]
    if destination.exists() or destination.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite E7/E8 staged inputs: {destination}")

    reviewed = reject_symlink_path_components(reviewed_splits, "E7/E8 reviewed splits")
    language = reject_symlink_path_components(
        reviewed_language_suite, "E7/E8 reviewed language suite"
    )
    evaluator = reject_symlink_path_components(ifeval_evaluator, "E7/E8 IFEval evaluator")
    raw_sources = {
        name: reject_symlink_path_components(path, f"E7/E8 {name} source")
        for name, path in source_artifacts.items()
    }
    reviewed_manifest = validate_reviewed_split_snapshot(reviewed)
    if reviewed_manifest.get("manifest_digest") != expected_reviewed_split_manifest_digest:
        raise FrozenArtifactError("E7/E8 reviewed-split manifest differs")
    if len(load_reviewed_language_suite(language)) != 500:
        raise DataValidationError("E7/E8 language suite must contain exactly 500 questions")
    validate_ifeval_evaluator(evaluator)
    verified_sources = {
        name: verify_source_artifact(SOURCE_SNAPSHOTS[name], path)
        for name, path in raw_sources.items()
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent)
    )
    try:
        shutil.copytree(reviewed, stage / "reviewed-splits")
        shutil.copytree(language, stage / "language-suite")
        shutil.copytree(evaluator, stage / "ifeval-evaluator")
        (stage / "sources").mkdir()
        for name, source in verified_sources.items():
            shutil.copyfile(source, stage / "sources" / _SOURCE_FILENAMES[name])
        body = {
            "schema_version": 1,
            "purpose": "e7-e8-external-input-snapshot",
            "reviewed_split_manifest_digest": expected_reviewed_split_manifest_digest,
            "reviewed_splits_sha256": sha256_path(stage / "reviewed-splits"),
            "language_suite_sha256": sha256_path(stage / "language-suite"),
            "language_question_count": 500,
            "ifeval_evaluator_sha256": validate_ifeval_evaluator(
                stage / "ifeval-evaluator"
            ),
            "sources": {
                name: {
                    "path": f"sources/{_SOURCE_FILENAMES[name]}",
                    "sha256": SOURCE_SNAPSHOTS[name].artifact_sha256,
                    "size_bytes": SOURCE_SNAPSHOTS[name].artifact_size_bytes,
                }
                for name in sorted(verified_sources)
            },
        }
        (stage / "manifest.json").write_text(
            json.dumps(
                {**body, "manifest_digest": stable_hash(body)},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        _validate_materials(
            stage,
            expected_reviewed_split_manifest_digest=(
                expected_reviewed_split_manifest_digest
            ),
        )
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
    return validate_e7_e8_external_inputs(
        destination,
        expected_reviewed_split_manifest_digest=expected_reviewed_split_manifest_digest,
    )

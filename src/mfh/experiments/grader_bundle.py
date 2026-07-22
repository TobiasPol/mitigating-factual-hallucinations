"""Freeze and replay every external-grader input required by E1."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mfh.artifact_namespace import validate_active_study_artifact_paths
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.evaluation.official import OfficialGraderSpec, load_official_grader_spec
from mfh.evaluation.openrouter import (
    route_for_grader,
    verify_openrouter_catalog,
)
from mfh.provenance import sha256_file, sha256_path, stable_hash

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_AMENDMENT_DIGEST = "3c8bb042d1d566650b39b10f23d368b4cf344f83c03157f999e6248ac99be555"
_CATALOG_SHA256 = "8c0fc0a422d3fbbb8818e93a0fb7e8b868ba7c77e976eaa3e5f7c9f715e0f3df"
_CATALOG_SOURCE = "artifacts/graders/openrouter/catalog-2026-07-22.json"
_SOURCE_PATHS = {
    "aa_config": "configs/graders/aa-omniscience-public.yaml",
    "aa_prompt": "configs/graders/aa-omniscience-public.prompt.txt",
    "aa_source": (
        "artifacts/graders/aa-omniscience/"
        "4a8ffc87c4650054825fb767fe0da4a4fc97ff32/README.md"
    ),
    "amendment": "configs/experiments/grader-selection-amendment.json",
    "catalog": _CATALOG_SOURCE,
    "config_implementation": "src/mfh/config.py",
    "contracts_implementation": "src/mfh/contracts.py",
    "errors_implementation": "src/mfh/errors.py",
    "grading_implementation": "src/mfh/evaluation/grading.py",
    "official_grading_implementation": "src/mfh/evaluation/official.py",
    "openrouter_implementation": "src/mfh/evaluation/openrouter.py",
    "provenance_implementation": "src/mfh/provenance.py",
    "python_project": "pyproject.toml",
    "python_lock": "uv.lock",
    "simpleqa_config": "configs/graders/simpleqa-verified.yaml",
    "simpleqa_prompt": "configs/graders/simpleqa-verified.prompt.txt",
    "simpleqa_source": (
        "artifacts/graders/simpleqa-verified/"
        "14d0c0513efefdfe7936e05c6fc09b4b4a191cc31273ca8bfbcdeaea0c6fdb1b/"
        "simpleqa-verified-benchmark-starter-code-v9.ipynb"
    ),
}
_REQUIRED_ROLES = frozenset(_SOURCE_PATHS)


def _repository_root(repository_root: str | Path | None) -> Path:
    root = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[3]
    )
    if not (root / "docs/research-plan.md").is_file() or not (root / "src/mfh").is_dir():
        raise DataValidationError("E1 grader-bundle repository root is invalid")
    return root


def grader_bundle_sources(repository_root: str | Path | None = None) -> Mapping[str, Path]:
    """Return the only live files accepted when freezing the E1 grader bundle."""

    root = _repository_root(repository_root)
    return {role: root / relative for role, relative in _SOURCE_PATHS.items()}


def _read_json(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot read {context}: {exc}") from exc
    if not isinstance(value, dict):
        raise DataValidationError(f"{context} must be a JSON object")
    return value


def _read_manifest(directory: Path) -> dict[str, Any]:
    path = directory / "manifest.json"
    if path.is_symlink() or not path.is_file():
        raise DataValidationError("E1 grader bundle lacks a regular manifest")
    payload = _read_json(path, "E1 grader-bundle manifest")
    digest = payload.pop("manifest_digest", None)
    if digest != stable_hash(payload):
        raise DataValidationError("E1 grader-bundle manifest digest differs")
    payload["manifest_digest"] = digest
    return payload


def _verify_amendment(path: Path, *, expected_adapter_digest: str) -> dict[str, Any]:
    amendment = _read_json(path, "E1 grader amendment")
    digest = amendment.get("amendment_digest")
    body = dict(amendment)
    body.pop("amendment_digest", None)
    if digest != _AMENDMENT_DIGEST or digest != stable_hash(body):
        raise DataValidationError("E1 grader amendment is not the approved pre-E1 amendment")
    if (
        amendment.get("schema_version") != 1
        or amendment.get("timing") != "before-any-e1-generation-or-grading"
        or amendment.get("adapter_semantic_digest") != expected_adapter_digest
    ):
        raise DataValidationError("E1 grader amendment identity differs")
    transport = amendment.get("transport")
    if not isinstance(transport, Mapping) or (
        transport.get("provider") != "openrouter"
        or transport.get("endpoint") != "https://openrouter.ai/api/v1/chat/completions"
        or transport.get("api_key_environment_variable") != "OPENROUTER_API_KEY"
        or transport.get("allow_model_fallbacks") is not False
        or transport.get("require_parameter_support") is not True
        or transport.get("catalog_artifact") != _CATALOG_SOURCE
        or transport.get("catalog_response_sha256") != _CATALOG_SHA256
    ):
        raise DataValidationError("E1 grader amendment transport differs")
    graders = amendment.get("graders")
    if not isinstance(graders, Mapping) or set(graders) != {
        "simpleqa_verified",
        "aa_omniscience_public_600",
    }:
        raise DataValidationError("E1 grader amendment grader identities differ")
    return amendment


def _verify_grader_identities(
    files: Mapping[str, Path], amendment: Mapping[str, Any]
) -> tuple[OfficialGraderSpec, OfficialGraderSpec]:
    simpleqa = load_official_grader_spec(files["simpleqa_config"])
    aa = load_official_grader_spec(files["aa_config"])
    if simpleqa.benchmark != "simpleqa_verified" or aa.benchmark != "aa_omniscience_public_600":
        raise DataValidationError("E1 grader bundle contains unexpected benchmarks")
    simpleqa.verify_source_artifact(files["simpleqa_source"])
    aa.verify_source_artifact(files["aa_source"])
    amendment_graders = amendment["graders"]
    simpleqa_route = route_for_grader(simpleqa)
    aa_route = route_for_grader(aa)
    simpleqa_amendment = amendment_graders["simpleqa_verified"]
    aa_amendment = amendment_graders["aa_omniscience_public_600"]
    if (
        not isinstance(simpleqa_amendment, Mapping)
        or simpleqa_amendment.get("requested_model") != simpleqa_route.request_model
        or simpleqa_amendment.get("required_canonical_slug") != simpleqa_route.canonical_slug
        or simpleqa_amendment.get("required_provider_slug") != simpleqa_route.provider_slug
        or simpleqa_amendment.get("status") != "unchanged-from-frozen-grader"
    ):
        raise DataValidationError("SimpleQA OpenRouter identity differs from the amendment")
    if (
        not isinstance(aa_amendment, Mapping)
        or aa_amendment.get("replacement_model") != aa.grader_model
        or aa_amendment.get("replacement_revision") != aa.grader_model_revision
        or aa_amendment.get("required_canonical_slug") != aa_route.canonical_slug
        or aa_amendment.get("required_provider_slug") != aa_route.provider_slug
        or aa_amendment.get("status") != "pre-e1-protocol-amendment"
    ):
        raise DataValidationError("AA OpenRouter identity differs from the amendment")
    catalog = _read_json(files["catalog"], "frozen OpenRouter catalog")
    if sha256_file(files["catalog"]) != _CATALOG_SHA256:
        raise DataValidationError("frozen OpenRouter catalog bytes differ from the amendment")
    verify_openrouter_catalog(catalog, (simpleqa, aa))
    return simpleqa, aa


def _bundle_files(directory: Path, manifest: Mapping[str, Any]) -> Mapping[str, Path]:
    descriptors = manifest.get("files")
    if not isinstance(descriptors, Mapping) or set(descriptors) != _REQUIRED_ROLES:
        raise DataValidationError("E1 grader bundle file roles differ")
    files: dict[str, Path] = {}
    declared_names: set[str] = set()
    for role, descriptor in descriptors.items():
        expected_source = _SOURCE_PATHS[role]
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
            "path",
            "sha256",
            "size_bytes",
            "source_path",
        }:
            raise DataValidationError("E1 grader-bundle file descriptor is invalid")
        relative = descriptor["path"]
        expected_sha256 = descriptor["sha256"]
        size_bytes = descriptor["size_bytes"]
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or Path(relative).parent != Path("files")
            or ".." in Path(relative).parts
            or not isinstance(expected_sha256, str)
            or _SHA256.fullmatch(expected_sha256) is None
            or not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes < 0
            or descriptor["source_path"] != expected_source
        ):
            raise DataValidationError("E1 grader-bundle file identity is invalid")
        path = directory / relative
        if path.is_symlink() or not path.is_file():
            raise DataValidationError("E1 grader-bundle file is missing or linked")
        if sha256_file(path) != expected_sha256 or path.stat().st_size != size_bytes:
            raise DataValidationError("E1 grader-bundle file bytes differ")
        if path.name in declared_names:
            raise DataValidationError("E1 grader-bundle filenames must be unique")
        declared_names.add(path.name)
        files[str(role)] = path
    files_root = directory / "files"
    if (
        files_root.is_symlink()
        or not files_root.is_dir()
        or {item.name for item in files_root.iterdir()} != declared_names
        or any(not item.is_file() or item.is_symlink() for item in files_root.iterdir())
    ):
        raise DataValidationError("E1 grader bundle contains undeclared files")
    return files


def verify_e1_grader_bundle(
    directory: str | Path,
    *,
    expected_manifest_digest: str | None = None,
    repository_root: str | Path | None = None,
    verify_live_sources: bool = False,
) -> Mapping[str, Any]:
    """Replay the closed grader bundle and optionally compare every live input."""

    source = Path(directory)
    if source.is_symlink() or not source.is_dir():
        raise DataValidationError("E1 grader bundle must be a regular directory")
    if {item.name for item in source.iterdir()} != {"files", "manifest.json"}:
        raise DataValidationError("E1 grader bundle has invalid top-level inventory")
    manifest = _read_manifest(source)
    if expected_manifest_digest is not None and (
        _SHA256.fullmatch(expected_manifest_digest) is None
        or manifest["manifest_digest"] != expected_manifest_digest
    ):
        raise FrozenArtifactError("E1 grader-bundle manifest differs from the expected digest")
    if (
        set(manifest)
        != {
            "schema_version",
            "bundle_kind",
            "amendment_digest",
            "adapter_semantic_digest",
            "catalog_sha256",
            "grader_fingerprints",
            "routes",
            "files",
            "manifest_digest",
        }
        or manifest.get("schema_version") != 1
        or manifest.get("bundle_kind") != "e1-official-graders-openrouter"
        or manifest.get("amendment_digest") != _AMENDMENT_DIGEST
        or not isinstance(manifest.get("adapter_semantic_digest"), str)
        or _SHA256.fullmatch(str(manifest.get("adapter_semantic_digest"))) is None
        or manifest.get("catalog_sha256") != _CATALOG_SHA256
    ):
        raise DataValidationError("E1 grader-bundle manifest identity differs")
    files = _bundle_files(source, manifest)
    packaged_adapter_digest = sha256_file(files["openrouter_implementation"])
    if manifest["adapter_semantic_digest"] != packaged_adapter_digest:
        raise DataValidationError(
            "E1 grader-bundle adapter digest differs from its packaged implementation"
        )
    amendment = _verify_amendment(
        files["amendment"], expected_adapter_digest=packaged_adapter_digest
    )
    simpleqa, aa = _verify_grader_identities(files, amendment)
    specs = (simpleqa, aa)
    expected_fingerprints = {spec.benchmark: spec.digest for spec in specs}
    expected_routes = {
        spec.benchmark: {
            "request_model": route_for_grader(spec).request_model,
            "canonical_slug": route_for_grader(spec).canonical_slug,
            "provider_slug": route_for_grader(spec).provider_slug,
            "reasoning_enabled": route_for_grader(spec).reasoning_enabled,
        }
        for spec in specs
    }
    if (
        manifest["grader_fingerprints"] != expected_fingerprints
        or manifest["routes"] != expected_routes
    ):
        raise DataValidationError("E1 grader-bundle model or grader fingerprints differ")
    if verify_live_sources:
        live = grader_bundle_sources(repository_root)
        descriptors = manifest["files"]
        for role, path in live.items():
            if path.is_symlink() or not path.is_file():
                raise DataValidationError(f"live E1 grader input {role!r} is missing or linked")
            if sha256_file(path) != descriptors[role]["sha256"]:
                raise FrozenArtifactError(
                    f"live E1 grader input {role!r} differs from the frozen bundle"
                )
    return manifest


def write_e1_grader_bundle(
    destination: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> Mapping[str, Any]:
    """Atomically package the exact official graders, transport, and source evidence."""

    target = validate_active_study_artifact_paths(
        {"E1 grader bundle": destination}
    )["E1 grader bundle"]
    if target.exists():
        raise FrozenArtifactError(f"refusing to overwrite E1 grader bundle: {target}")
    sources = grader_bundle_sources(repository_root)
    starting_hashes: dict[str, str] = {}
    for role, source in sources.items():
        if source.is_symlink() or not source.is_file():
            raise DataValidationError(f"E1 grader input {role!r} is missing or linked")
        starting_hashes[role] = sha256_file(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=target.parent))
    try:
        files_root = stage / "files"
        files_root.mkdir()
        descriptors: dict[str, dict[str, str | int]] = {}
        packaged: dict[str, Path] = {}
        for role in sorted(sources):
            source = sources[role]
            destination_name = files_root / source.name
            if destination_name.exists():
                raise DataValidationError("E1 grader inputs require unique filenames")
            shutil.copyfile(source, destination_name)
            packaged[role] = destination_name
            descriptors[role] = {
                "path": f"files/{destination_name.name}",
                "sha256": sha256_file(destination_name),
                "size_bytes": destination_name.stat().st_size,
                "source_path": _SOURCE_PATHS[role],
            }
        packaged_adapter_digest = sha256_file(packaged["openrouter_implementation"])
        amendment = _verify_amendment(
            packaged["amendment"], expected_adapter_digest=packaged_adapter_digest
        )
        simpleqa, aa = _verify_grader_identities(packaged, amendment)
        specs = (simpleqa, aa)
        body: dict[str, Any] = {
            "schema_version": 1,
            "bundle_kind": "e1-official-graders-openrouter",
            "amendment_digest": _AMENDMENT_DIGEST,
            "adapter_semantic_digest": packaged_adapter_digest,
            "catalog_sha256": _CATALOG_SHA256,
            "grader_fingerprints": {spec.benchmark: spec.digest for spec in specs},
            "routes": {
                spec.benchmark: {
                    "request_model": route_for_grader(spec).request_model,
                    "canonical_slug": route_for_grader(spec).canonical_slug,
                    "provider_slug": route_for_grader(spec).provider_slug,
                    "reasoning_enabled": route_for_grader(spec).reasoning_enabled,
                }
                for spec in specs
            },
            "files": descriptors,
        }
        (stage / "manifest.json").write_text(
            json.dumps({**body, "manifest_digest": stable_hash(body)}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        verify_e1_grader_bundle(
            stage, repository_root=repository_root, verify_live_sources=True
        )
        if {
            role: sha256_file(path) for role, path in sources.items()
        } != starting_hashes:
            raise FrozenArtifactError("an E1 grader input changed while freezing the bundle")
        if target.exists():
            raise FrozenArtifactError(f"E1 grader output appeared during freezing: {target}")
        os.replace(stage, target)
        manifest = verify_e1_grader_bundle(
            target, repository_root=repository_root, verify_live_sources=True
        )
        manifest = dict(manifest)
        manifest["bundle_sha256"] = sha256_path(target)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)

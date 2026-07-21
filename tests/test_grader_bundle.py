from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.experiments.grader_bundle import (
    grader_bundle_sources,
    verify_e1_grader_bundle,
    write_e1_grader_bundle,
)
from mfh.provenance import sha256_file, stable_hash


def test_e1_grader_bundle_freezes_and_replays_every_live_input(tmp_path: Path) -> None:
    output = tmp_path / "graders"
    created = write_e1_grader_bundle(output)
    manifest = verify_e1_grader_bundle(
        output,
        expected_manifest_digest=str(created["manifest_digest"]),
    )

    assert manifest["bundle_kind"] == "e1-official-graders-openrouter"
    assert manifest["catalog_sha256"] == (
        "3dbeb1f9a71faed5435d6c1ce3f4ae6a0b388cd911b0d1ac7e6b845d930b2045"
    )
    assert set(manifest["grader_fingerprints"]) == {
        "simpleqa_verified",
        "aa_omniscience_public_600",
    }
    assert set(manifest["files"]) == set(grader_bundle_sources())
    for role in grader_bundle_sources():
        packaged = output / manifest["files"][role]["path"]
        assert sha256_file(packaged) == manifest["files"][role]["sha256"]
    with pytest.raises(FrozenArtifactError, match="live E1 grader input"):
        verify_e1_grader_bundle(output, verify_live_sources=True)


def test_e1_grader_bundle_rejects_tampering_and_wrong_expected_digest(tmp_path: Path) -> None:
    output = tmp_path / "graders"
    write_e1_grader_bundle(output)
    with pytest.raises(FrozenArtifactError, match="expected digest"):
        verify_e1_grader_bundle(output, expected_manifest_digest="0" * 64)

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    catalog = output / manifest["files"]["catalog"]["path"]
    catalog.write_text('{"data": []}\n', encoding="utf-8")
    with pytest.raises(DataValidationError, match="file bytes differ"):
        verify_e1_grader_bundle(output)


def test_e1_grader_bundle_is_write_once(tmp_path: Path) -> None:
    output = tmp_path / "graders"
    write_e1_grader_bundle(output)
    with pytest.raises(FrozenArtifactError, match="overwrite"):
        write_e1_grader_bundle(output)


def test_e1_grader_bundle_rejects_self_consistent_adapter_retie(tmp_path: Path) -> None:
    output = tmp_path / "graders"
    write_e1_grader_bundle(output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    adapter_descriptor = manifest["files"]["openrouter_implementation"]
    adapter = output / adapter_descriptor["path"]
    adapter.write_text(adapter.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    adapter_descriptor["sha256"] = sha256_file(adapter)
    adapter_descriptor["size_bytes"] = adapter.stat().st_size
    manifest.pop("manifest_digest")
    manifest_path.write_text(
        json.dumps(
            {**manifest, "manifest_digest": stable_hash(manifest)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DataValidationError, match="adapter digest differs"):
        verify_e1_grader_bundle(output, verify_live_sources=False)


def test_e1_grader_bundle_rejects_inventory_and_symlink_attacks(tmp_path: Path) -> None:
    extra = tmp_path / "extra"
    write_e1_grader_bundle(extra)
    (extra / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(DataValidationError, match="top-level inventory"):
        verify_e1_grader_bundle(extra)

    linked = tmp_path / "linked"
    write_e1_grader_bundle(linked)
    manifest = json.loads((linked / "manifest.json").read_text(encoding="utf-8"))
    prompt = linked / manifest["files"]["simpleqa_prompt"]["path"]
    external = tmp_path / "external-prompt.txt"
    shutil.copyfile(prompt, external)
    prompt.unlink()
    prompt.symlink_to(external)
    with pytest.raises(DataValidationError, match="missing or linked"):
        verify_e1_grader_bundle(linked)

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mfh.errors import ConfigurationError, FrozenArtifactError
from mfh.experiments.protocol import load_study_protocol
from mfh.experiments.runner import PhaseRunLedger
from mfh.experiments.synthetic import run_synthetic_study, verify_synthetic_study
from mfh.provenance import sha256_file, stable_hash

_ROOT = Path(__file__).parents[1]


def test_synthetic_study_runs_every_phase_and_replays(tmp_path: Path) -> None:
    directory = tmp_path / "synthetic-smoke"

    created = run_synthetic_study(directory, seed=1701)
    verified = verify_synthetic_study(directory, replay=True)

    assert created.scientific_eligible is False
    assert verified.bundle_digest == created.bundle_digest
    assert tuple(verified.phase_digests) == tuple(f"E{index}" for index in range(11))
    assert not list(directory.rglob("complete.json"))

    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scientific_eligible"] is False
    assert manifest["runtime"] == "synthetic"
    assert "not confirmatory evidence" in manifest["warning"]

    e10 = json.loads((directory / "phases" / "E10.json").read_text(encoding="utf-8"))
    assert e10["scientific_eligible"] is False
    assert e10["result"]["one_shot_confirmatory_executed"] is False
    assert set(e10["result"]["risk_regimes"]) == {
        "known",
        "likely_unknown",
        "potentially_recoverable",
    }


def test_synthetic_study_rejects_tampering_and_extra_files(tmp_path: Path) -> None:
    directory = tmp_path / "synthetic-smoke"
    run_synthetic_study(directory)

    e5 = directory / "phases" / "E5.json"
    e5.write_text(
        e5.read_text(encoding="utf-8").replace('"cluster_count": 2', '"cluster_count": 3')
    )
    with pytest.raises(FrozenArtifactError, match="E5 file changed"):
        verify_synthetic_study(directory, replay=False)

    e5.unlink()
    (directory / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(FrozenArtifactError, match="missing or unexpected"):
        verify_synthetic_study(directory, replay=False)


def test_synthetic_replay_rejects_fully_rehashed_forgery(tmp_path: Path) -> None:
    directory = tmp_path / "synthetic-smoke"
    run_synthetic_study(directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    previous_digest: str | None = None
    for phase in manifest["phase_order"]:
        phase_path = directory / "phases" / f"{phase}.json"
        payload = json.loads(phase_path.read_text(encoding="utf-8"))
        if phase == "E5":
            payload["result"]["cluster_count"] = 99
        if previous_digest is not None:
            payload["dependencies"] = {f"E{int(phase[1:]) - 1}": previous_digest}
        payload_body = {key: value for key, value in payload.items() if key != "phase_digest"}
        payload["phase_digest"] = stable_hash(payload_body)
        phase_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest["phases"][phase]["sha256"] = sha256_file(phase_path)
        manifest["phases"][phase]["phase_digest"] = payload["phase_digest"]
        previous_digest = payload["phase_digest"]

    manifest["execution_digest"] = stable_hash(
        [manifest["phases"][phase]["phase_digest"] for phase in manifest["phase_order"]]
    )
    manifest_body = {key: value for key, value in manifest.items() if key != "bundle_digest"}
    manifest["bundle_digest"] = stable_hash(manifest_body)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    verify_synthetic_study(directory, replay=False)
    with pytest.raises(FrozenArtifactError, match="deterministic replay"):
        verify_synthetic_study(directory, replay=True)


def test_synthetic_bundle_is_not_a_scientific_phase_ledger(tmp_path: Path) -> None:
    directory = tmp_path / "synthetic-smoke"
    run_synthetic_study(directory)
    study = load_study_protocol(_ROOT / "configs" / "experiments" / "phases.yaml")

    with pytest.raises(ConfigurationError, match="qwen36-27b-mlx4-m4max48-v1"):
        PhaseRunLedger.open(directory, study=study)


def test_synthetic_study_refuses_overwrite_and_invalid_seed(tmp_path: Path) -> None:
    directory = tmp_path / "synthetic-smoke"
    run_synthetic_study(directory)

    with pytest.raises(FrozenArtifactError, match="refusing to overwrite"):
        run_synthetic_study(directory)
    with pytest.raises(ValueError, match="non-negative integer"):
        run_synthetic_study(directory, seed=-1)
    with pytest.raises(ValueError, match="non-negative integer"):
        run_synthetic_study(directory, seed=True)

    fresh = tmp_path / "fresh"
    with pytest.raises(ValueError, match="non-negative integer"):
        run_synthetic_study(fresh, seed=-1)
    with pytest.raises(ValueError, match="non-negative integer"):
        run_synthetic_study(fresh, seed=True)

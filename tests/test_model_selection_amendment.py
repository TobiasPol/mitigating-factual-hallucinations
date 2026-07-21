from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from mfh.errors import ConfigurationError
from mfh.experiments.e5_native import verify_e5_native_ablation
from mfh.experiments.e5_operator import verify_signed_e5_selection
from mfh.experiments.evidence import write_gate_result
from mfh.experiments.model_selection import (
    QWEN_STUDY_ARTIFACT_ROOT,
    QWEN_STUDY_NAMESPACE,
    load_model_selection_amendment,
    validate_active_study_artifact_paths,
)
from mfh.experiments.runner import PhaseRunLedger
from mfh.methods.composite import save_composite_manifest

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / "configs/experiments/model-selection-amendment.json"
MODELS = ROOT / "configs/models"


def test_approved_qwen_model_selection_is_bound_to_exact_config_and_policy() -> None:
    amendment = load_model_selection_amendment(AMENDMENT, model_config_directory=MODELS)

    assert amendment["amendment_digest"] == (
        "d0a26583a42620a29a4c6bb1968f3995b8c5664d9cc0703692be66041c478dd8"
    )
    assert [row["name"] for row in amendment["active_models"]] == ["qwen3.6-27b-mlx-4bit"]


def test_model_selection_rejects_body_tampering(tmp_path: Path) -> None:
    raw = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    raw["hardware_envelope"]["unified_memory_bytes"] = 1
    candidate = tmp_path / "amendment.json"
    candidate.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="digest differs"):
        load_model_selection_amendment(candidate, model_config_directory=MODELS)


def test_model_selection_rejects_semantic_model_config_drift(tmp_path: Path) -> None:
    model_directory = tmp_path / "models"
    shutil.copytree(MODELS, model_directory)
    candidate = model_directory / "qwen3.6-27b-mlx-4bit.yaml"
    text = candidate.read_text(encoding="utf-8")
    candidate.write_text(
        text.replace(
            "quantization: affine-g64-mlx-4bit",
            "quantization: fabricated",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="semantic config differs"):
        load_model_selection_amendment(AMENDMENT, model_config_directory=model_directory)


def test_active_study_mutations_require_the_qwen_namespace(tmp_path: Path) -> None:
    namespace = tmp_path / QWEN_STUDY_ARTIFACT_ROOT
    accepted = validate_active_study_artifact_paths(
        {
            "work": namespace / "work/E1",
            "ledger": namespace / "runs/E1",
        },
        project_root=tmp_path,
    )
    assert accepted["work"] == (namespace / "work/E1").resolve()

    with pytest.raises(ConfigurationError, match="must stay inside"):
        validate_active_study_artifact_paths(
            {"legacy": tmp_path / "artifacts/work/E1"},
            project_root=tmp_path,
        )
    with pytest.raises(ConfigurationError, match="must stay inside"):
        validate_active_study_artifact_paths(
            {"root": namespace},
            project_root=tmp_path,
        )


def test_active_study_paths_reject_namespace_and_nested_symlinks(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    studies = root / "artifacts" / "studies"
    studies.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    namespace = studies / QWEN_STUDY_NAMESPACE
    namespace.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ConfigurationError, match="symlink"):
        validate_active_study_artifact_paths(
            {"ledger": namespace / "runs" / "E0"}, project_root=root
        )

    namespace.unlink()
    namespace.mkdir()
    (namespace / "runs").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ConfigurationError, match="symlink"):
        validate_active_study_artifact_paths(
            {"ledger": namespace / "runs" / "E0"}, project_root=root
        )


def test_phase_ledger_factories_reject_outside_active_namespace(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="qwen36-27b-mlx4-m4max48-v1"):
        PhaseRunLedger.open(tmp_path / "copied-E0", study=object())  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="qwen36-27b-mlx4-m4max48-v1"):
        PhaseRunLedger.create(
            tmp_path / "new-E0",
            object(),  # type: ignore[arg-type]
            study=object(),  # type: ignore[arg-type]
            input_artifacts={},
            prerequisite_runs={},
        )


def test_e5_verifiers_reject_copied_artifacts_outside_active_namespace(
    tmp_path: Path,
) -> None:
    with pytest.raises(ConfigurationError, match="qwen36-27b-mlx4-m4max48-v1"):
        verify_e5_native_ablation(
            tmp_path / "copied-native",
            expected_execution_public_key="0" * 64,
        )
    with pytest.raises(ConfigurationError, match="qwen36-27b-mlx4-m4max48-v1"):
        verify_signed_e5_selection(
            tmp_path / "copied-selection",
            native_directory=tmp_path / "copied-native",
            execution_private_key_hex="0" * 64,
        )


def test_public_scientific_writers_reject_outside_active_namespace(
    tmp_path: Path,
) -> None:
    with pytest.raises(ConfigurationError, match="must stay inside"):
        write_gate_result(tmp_path / "gate-result.json", object())  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="must stay inside"):
        save_composite_manifest(
            tmp_path / "composite.json",
            object(),  # type: ignore[arg-type]
        )


def test_partial_superseded_artifact_transfer_is_rejected(tmp_path: Path) -> None:
    project = tmp_path / "project"
    shutil.copytree(ROOT / "configs", project / "configs")
    partial = project / "artifacts/runs/E0"
    partial.mkdir(parents=True)
    (partial / "unsealed-extra.bin").write_bytes(b"not frozen")

    amendment = project / "configs/experiments/model-selection-amendment.json"
    models = project / "configs/models"
    with pytest.raises(ConfigurationError, match="superseded Bonsai E0 artifact differs"):
        load_model_selection_amendment(amendment, model_config_directory=models)

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.experiments import confirmatory_operator
from mfh.experiments.confirmatory_operator import (
    ConfirmatoryRunbook,
    _open_runbook_bound_ledger,
    _preflight_contract,
    _selection_manifest,
    _validate_e9_runtime_binding,
    prepare_confirmatory_runbook,
    summarize_runbook_paths,
    write_confirmatory_runbook_template,
)
from mfh.experiments.protocol import ExperimentPhase
from mfh.provenance import stable_hash


@pytest.mark.parametrize(
    ("phase", "expected_inputs", "expected_prerequisites"),
    (
        (
            ExperimentPhase.E9,
            {
                "frozen_component_selection",
                "frozen_graders",
                "frozen_evaluation_scripts",
                "frozen_question_bundle",
                "frozen_prompt_paraphrase_schedule",
            },
            {f"E{index}" for index in range(9)},
        ),
        (
            ExperimentPhase.E10,
            {
                "E9_results",
                "component_selection_manifest",
                "frozen_question_bundle",
                "model_revision",
                "prompt",
                "risk_threshold",
                "vector_bank",
                "sae_checkpoint",
                "protected_subspace",
                "layer",
                "alpha_policy",
                "abstention_rule",
                "grader",
                "evaluation_scripts",
            },
            {f"E{index}" for index in range(10)},
        ),
    ),
)
def test_confirmatory_template_is_secret_free_and_resolves_relative_paths(
    tmp_path: Path,
    phase: ExperimentPhase,
    expected_inputs: set[str],
    expected_prerequisites: set[str],
) -> None:
    path = tmp_path / "operator" / f"{phase.value}.json"
    digest = write_confirmatory_runbook_template(path, phase=phase)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert digest
    assert set(payload["input_artifacts"]) == expected_inputs
    assert set(payload["prerequisite_runs"]) == expected_prerequisites
    assert "key" not in json.dumps(payload).lower()

    loaded = ConfirmatoryRunbook.load(path)
    assert loaded.phase is phase
    assert loaded.source == path.resolve()
    assert loaded.run_directory.is_absolute()
    assert summarize_runbook_paths(loaded)["runbook_digest"] == loaded.runbook_digest


def test_confirmatory_runbook_rejects_inline_secret_fields(tmp_path: Path) -> None:
    path = tmp_path / "E9.json"
    write_confirmatory_runbook_template(path, phase=ExperimentPhase.E9)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["openrouter_api_key"] = "must-not-be-serialized"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DataValidationError, match="keys differ"):
        ConfirmatoryRunbook.load(path)


def test_template_fixed_paths_match_documented_study_layout(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    fixed_files = (
        configs / "experiments/phases.yaml",
        configs / "models/qwen3.6-27b-mlx-4bit.yaml",
        configs / "prompts/primary.yaml",
        configs / "models/qwen3.6-27b-mlx-4bit.snapshot.json",
    )
    for fixed in fixed_files:
        fixed.parent.mkdir(parents=True, exist_ok=True)
        fixed.touch()
    snapshot = tmp_path / "artifacts/models/qwen3.6-27b-mlx-4bit/SNAPSHOT"
    snapshot.mkdir(parents=True)
    study = tmp_path / "artifacts/studies/qwen36-27b-mlx4-m4max48-v1"
    runbook_path = study / "operator-inputs/E9-runbook.json"

    write_confirmatory_runbook_template(runbook_path, phase=ExperimentPhase.E9)
    runbook = ConfirmatoryRunbook.load(runbook_path)

    assert runbook.study_protocol == fixed_files[0]
    assert runbook.model_config == fixed_files[1]
    assert runbook.prompt_config == fixed_files[2]
    assert runbook.snapshot_manifest == fixed_files[3]
    assert runbook.snapshot_directory == snapshot
    assert all(path.is_file() for path in fixed_files)
    assert runbook.snapshot_directory.is_dir()
    assert runbook.run_directory == study / "runs/E9"
    assert runbook.evidence_directory == study / "evidence/E9"


def test_confirmatory_runbook_rejects_symlinked_source(tmp_path: Path) -> None:
    source = tmp_path / "E9.json"
    alias = tmp_path / "E9-alias.json"
    write_confirmatory_runbook_template(source, phase=ExperimentPhase.E9)
    alias.symlink_to(source)
    with pytest.raises(DataValidationError, match="cannot traverse symlinks"):
        ConfirmatoryRunbook.load(alias)


def test_e10_prepare_requires_explicit_one_shot_authority_before_preflight(
    tmp_path: Path,
) -> None:
    path = tmp_path / "E10.json"
    write_confirmatory_runbook_template(path, phase=ExperimentPhase.E10)
    runbook = ConfirmatoryRunbook.load(path)
    with pytest.raises(DataValidationError, match="authorize_one_shot"):
        prepare_confirmatory_runbook(runbook)


def test_component_selection_manifest_requires_exact_digest_and_phase(
    tmp_path: Path,
) -> None:
    root = tmp_path / "selection"
    root.mkdir()
    body = {
        "schema_version": 3,
        "study_protocol_digest": "a" * 64,
        "phase": "E9",
        "components": [],
    }
    (root / "manifest.json").write_text(
        json.dumps({**body, "manifest_digest": stable_hash(body)}),
        encoding="utf-8",
    )
    assert _selection_manifest(root, phase=ExperimentPhase.E9) == ()
    with pytest.raises(FrozenArtifactError, match="manifest differs"):
        _selection_manifest(root, phase=ExperimentPhase.E10)


def test_confirmatory_template_is_write_once(tmp_path: Path) -> None:
    path = tmp_path / "E9.json"
    write_confirmatory_runbook_template(path, phase=ExperimentPhase.E9)
    with pytest.raises(FrozenArtifactError, match="overwrite"):
        write_confirmatory_runbook_template(path, phase=ExperimentPhase.E9)


def test_runbook_bound_ledger_rejects_same_phase_contract_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "E9.json"
    write_confirmatory_runbook_template(path, phase=ExperimentPhase.E9)
    runbook = ConfirmatoryRunbook.load(path)
    substituted = SimpleNamespace(
        contract=SimpleNamespace(phase=ExperimentPhase.E9, digest="b" * 64)
    )
    monkeypatch.setattr(
        confirmatory_operator.PhaseRunLedger,
        "open",
        lambda *_args, **_kwargs: substituted,
    )
    with pytest.raises(FrozenArtifactError, match="differs from the runbook contract"):
        _open_runbook_bound_ledger(
            runbook,
            study=SimpleNamespace(),  # type: ignore[arg-type]
            expected_contract=SimpleNamespace(digest="a" * 64),  # type: ignore[arg-type]
        )


def test_e10_read_only_preflight_replays_all_freeze_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "E10.json"
    write_confirmatory_runbook_template(path, phase=ExperimentPhase.E10)
    runbook = ConfirmatoryRunbook.load(path)
    freeze_fields = tuple(
        name
        for name in runbook.input_artifacts
        if name not in {"E9_results", "component_selection_manifest", "frozen_question_bundle"}
    )
    study = SimpleNamespace(
        digest="study-digest",
        phase=lambda _phase: SimpleNamespace(freeze_fields=freeze_fields),
    )
    model = SimpleNamespace()
    prompt = SimpleNamespace(prompt_id="P2-calibrated-abstention")
    contract = SimpleNamespace()
    intervention = SimpleNamespace(adaptive_policy=SimpleNamespace(execution_public_key="a" * 64))
    e6_ledger = SimpleNamespace()
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        confirmatory_operator,
        "_study_and_model",
        lambda _runbook: (study, model),
    )
    monkeypatch.setattr(confirmatory_operator, "_questions", lambda _runbook: {})
    monkeypatch.setattr(
        confirmatory_operator,
        "_verified_prerequisites",
        lambda _runbook, **_kwargs: ({}, {ExperimentPhase.E6: e6_ledger}),
    )
    monkeypatch.setattr(confirmatory_operator, "_input_inventory", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(confirmatory_operator, "load_prompt_specs", lambda _path: (prompt,))
    monkeypatch.setattr(
        confirmatory_operator,
        "_derive_e10_composite_provenance",
        lambda **_kwargs: {"selected_prompt_id": prompt.prompt_id},
    )
    monkeypatch.setattr(
        confirmatory_operator,
        "_e10_component",
        lambda _path: tmp_path / "M6",
    )
    monkeypatch.setattr(confirmatory_operator, "e10_intervention", lambda **_kwargs: intervention)
    monkeypatch.setattr(confirmatory_operator, "build_e10_contract", lambda **_kwargs: contract)
    monkeypatch.setattr(confirmatory_operator, "_validate_question_bundle", lambda *_args: None)
    monkeypatch.setattr(
        confirmatory_operator,
        "validate_e10_prerequisite_bound_inputs",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        confirmatory_operator,
        "_e6_runtime_artifact",
        lambda _ledger: tmp_path / "runtime-attestation.json",
    )

    def capture(artifacts: object, **kwargs: object) -> dict[str, str]:
        observed["artifacts"] = artifacts
        observed.update(kwargs)
        return {}

    monkeypatch.setattr(confirmatory_operator, "validate_e10_freeze_inputs", capture)

    assert _preflight_contract(runbook) == (study, model, contract)
    assert set(observed["artifacts"]) == set(freeze_fields)  # type: ignore[arg-type]
    assert observed["study_protocol_digest"] == study.digest
    assert observed["expected_execution_public_key"] == "a" * 64


def test_e9_preflight_rejects_adaptive_key_not_bound_to_exact_e6(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "E9.json"
    write_confirmatory_runbook_template(path, phase=ExperimentPhase.E9)
    runbook = ConfirmatoryRunbook.load(path)
    monkeypatch.setattr(
        confirmatory_operator,
        "_e6_runtime_artifact",
        lambda _ledger: tmp_path / "runtime-attestation.json",
    )
    monkeypatch.setattr(
        confirmatory_operator,
        "_validate_exact_e6_runtime_binding",
        lambda **_kwargs: "a" * 64,
    )
    substituted_contract = SimpleNamespace(
        conditions=(
            SimpleNamespace(
                steering_method="M3",
                adaptive_policy=SimpleNamespace(execution_public_key="b" * 64),
            ),
        )
    )
    with pytest.raises(FrozenArtifactError, match="exact E6 execution key"):
        _validate_e9_runtime_binding(
            runbook,
            contract=substituted_contract,  # type: ignore[arg-type]
            prerequisite_ledgers={  # type: ignore[arg-type]
                ExperimentPhase.E6: SimpleNamespace()
            },
        )

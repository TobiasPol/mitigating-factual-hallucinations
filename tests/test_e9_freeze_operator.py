from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mfh.contracts import ActivationSite, PromptSpec, TokenScope
from mfh.errors import FrozenArtifactError
from mfh.experiments import e9_freeze_operator as operator
from mfh.experiments.protocol import ExperimentPhase
from mfh.provenance import sha256_file, sha256_path


def _external_inputs(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    grader = tmp_path / "graders"
    reviewed = tmp_path / "reviewed"
    grader.mkdir()
    reviewed.mkdir()
    (grader / "manifest.json").write_text("grader", encoding="utf-8")
    (reviewed / "manifest.json").write_text("reviewed", encoding="utf-8")
    sources = {
        "triviaqa": tmp_path / "triviaqa.parquet",
        "simpleqa_verified": tmp_path / "simpleqa_verified.csv",
        "aa_omniscience_public_600": tmp_path / "aa.csv",
    }
    for name, path in sources.items():
        path.write_text(name, encoding="utf-8")
    return grader, reviewed, sources


def _patch_external_verifiers(
    monkeypatch: pytest.MonkeyPatch, destination: Path
) -> None:
    monkeypatch.setattr(
        operator,
        "validate_active_study_artifact_paths",
        lambda _values: {"E9 staged inputs": destination},
    )
    monkeypatch.setattr(operator, "verify_e1_grader_bundle", lambda *_a, **_k: {})
    monkeypatch.setattr(operator, "validate_reviewed_split_snapshot", lambda _path: {})


def test_stage_e9_inputs_atomically_copies_and_hashes_exact_inventory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "frozen/E9-external-inputs"
    grader, reviewed, sources = _external_inputs(tmp_path)
    _patch_external_verifiers(monkeypatch, destination)

    result = operator.stage_e9_external_inputs(
        destination,
        official_grader_bundle=grader,
        expected_official_grader_manifest_digest="a" * 64,
        reviewed_splits=reviewed,
        source_artifacts=sources,
    )

    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert result["sha256"] == sha256_path(destination)
    assert manifest["official_graders_sha256"] == sha256_path(
        destination / "official-graders"
    )
    assert manifest["reviewed_splits_sha256"] == sha256_path(
        destination / "reviewed-splits"
    )
    for name, source in sources.items():
        staged = Path(result["source_artifacts"][name])
        assert staged.read_bytes() == source.read_bytes()
        assert manifest["sources"][name]["sha256"] == sha256_file(staged)
    assert not tuple(destination.parent.glob(f".{destination.name}.stage-*"))


def test_stage_e9_inputs_rejects_lexical_symlink_before_verification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "frozen/E9-external-inputs"
    grader, reviewed, sources = _external_inputs(tmp_path)
    grader_link = tmp_path / "grader-link"
    grader_link.symlink_to(grader, target_is_directory=True)
    _patch_external_verifiers(monkeypatch, destination)
    called = False

    def verified(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(operator, "verify_e1_grader_bundle", verified)

    with pytest.raises(FrozenArtifactError, match="must not be symlinks"):
        operator.stage_e9_external_inputs(
            destination,
            official_grader_bundle=grader_link,
            expected_official_grader_manifest_digest="a" * 64,
            reviewed_splits=reviewed,
            source_artifacts=sources,
        )
    assert not called
    assert not destination.exists()


def test_stage_e9_inputs_removes_partial_stage_on_copy_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "frozen/E9-external-inputs"
    grader, reviewed, sources = _external_inputs(tmp_path)
    _patch_external_verifiers(monkeypatch, destination)

    def fail_copy(
        _source: Path, _destination: Path, **_kwargs: object
    ) -> None:
        raise OSError("simulated interrupted copy")

    monkeypatch.setattr(operator.shutil, "copyfile", fail_copy)

    with pytest.raises(OSError, match="simulated interrupted copy"):
        operator.stage_e9_external_inputs(
            destination,
            official_grader_bundle=grader,
            expected_official_grader_manifest_digest="a" * 64,
            reviewed_splits=reviewed,
            source_artifacts=sources,
        )
    assert not destination.exists()
    assert not tuple(destination.parent.glob(f".{destination.name}.stage-*"))


def test_freeze_e9_removes_hidden_stage_when_component_construction_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "E9-inputs"
    runbook = tmp_path / "E9-runbook.json"
    run_root = tmp_path / "runs"
    e8 = SimpleNamespace(
        outputs={
            "run_directory": run_root / "E8",
            "protected_artifact": tmp_path / "protected",
            "candidate_screen": tmp_path / "screen",
        },
        e6_transition_evidence=tmp_path / "e6-transition",
        model_config=tmp_path / "model.yaml",
        prompt_config=tmp_path / "prompts.yaml",
        seed=17,
    )
    context = SimpleNamespace(study=object())
    ledger = SimpleNamespace(verify_complete=lambda: object())
    record = SimpleNamespace(
        layer=0,
        site=ActivationSite.POST_MLP,
        token_scope=next(iter(TokenScope)),
    )
    points = {
        ("P0-neutral", method): SimpleNamespace(
            records=(record,),
            alpha=1.0,
            adaptive_policy=None,
        )
        for method in ("M1", "M3", "M4", "M5")
    }
    monkeypatch.setattr(
        operator,
        "validate_active_study_artifact_paths",
        lambda values: {name: Path(value) for name, value in values.items()},
    )
    monkeypatch.setattr(operator.E8Runbook, "load", staticmethod(lambda _path: e8))
    monkeypatch.setattr(operator, "_base_context", lambda _runbook: context)
    monkeypatch.setattr(operator, "open_phase_prerequisite", lambda *_a, **_k: ledger)
    monkeypatch.setattr(operator, "load_model_spec", lambda _path: object())
    monkeypatch.setattr(operator, "validate_active_model_spec", lambda _model: None)
    monkeypatch.setattr(
        operator,
        "load_prompt_specs",
        lambda _path: tuple(
            PromptSpec(name, "Answer.")
            for name in ("P0-neutral", "P1-direct", "P2-calibrated-abstention")
        ),
    )
    monkeypatch.setattr(
        operator,
        "_completion_material",
        lambda *_a, **_k: ({}, {}, {}),
    )
    monkeypatch.setattr(
        operator,
        "load_e8_protected_artifact",
        lambda _path: SimpleNamespace(reference_rms=1.0),
    )
    monkeypatch.setattr(operator, "_feature_schema", lambda *_a: object())
    monkeypatch.setattr(
        operator,
        "_e6_components",
        lambda *_a: SimpleNamespace(reference_rms=1.0),
    )
    monkeypatch.setattr(operator, "load_e8_candidate_screen", lambda _path: object())
    monkeypatch.setattr(operator, "_selected_points", lambda _screen: points)

    def fail_component(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated component failure")

    monkeypatch.setattr(operator, "write_confirmatory_fixed_component", fail_component)

    with pytest.raises(OSError, match="simulated component failure"):
        operator.freeze_e9_input_suite(
            root,
            e8_runbook=tmp_path / "E8.json",
            e9_runbook_output=runbook,
            evaluation_scripts=tmp_path / "evaluation",
            official_grader_bundle=tmp_path / "graders",
            expected_official_grader_manifest_digest="a" * 64,
            reviewed_splits=tmp_path / "reviewed",
            source_artifacts={
                "triviaqa": tmp_path / "triviaqa.parquet",
                "simpleqa_verified": tmp_path / "simpleqa.csv",
                "aa_omniscience_public_600": tmp_path / "aa.csv",
            },
            m2_source_artifact=tmp_path / "m2",
            e3_phase_run=tmp_path / "E3-operator/phase",
            execution_private_key="0" * 64,
        )
    assert not root.exists()
    assert not runbook.exists()
    assert not tuple(root.parent.glob(f".{root.name}.stage-*"))


def test_e9_prerequisites_use_the_custom_e3_terminal_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_root = tmp_path / "runs"
    e3_phase = tmp_path / "E3-operator/phase"
    opened: list[tuple[ExperimentPhase, Path]] = []

    def open_source(
        path: Path, *, phase: ExperimentPhase, study: object
    ) -> object:
        del study
        opened.append((phase, path))
        return SimpleNamespace(
            verify_complete=lambda: SimpleNamespace(
                completion_digest=f"digest-{phase.value}"
            )
        )

    monkeypatch.setattr(operator, "open_phase_prerequisite", open_source)

    paths, digests, _ = operator._completion_material(
        run_root,
        e3_phase_run=e3_phase,
        study=object(),
    )

    assert paths["E3"] == e3_phase
    assert digests["E3"] == "digest-E3"
    assert opened[ExperimentPhase.E3.ordinal] == (ExperimentPhase.E3, e3_phase)
    assert all(
        path == run_root / phase.value
        for phase, path in opened
        if phase is not ExperimentPhase.E3
    )

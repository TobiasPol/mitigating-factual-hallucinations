from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mfh.cli import build_parser
from mfh.errors import DataValidationError
from mfh.experiments import e6_operator, e7_e8_inputs, e7_operator, e8_operator
from mfh.experiments.e7_e8_inputs import (
    stage_e7_e8_external_inputs,
    validate_e7_e8_external_inputs,
)
from mfh.experiments.protocol import ExperimentPhase
from mfh.provenance import sha256_path

_DIGEST = "a" * 64
_SOURCE_NAMES = {
    "triviaqa",
    "ifeval",
    "mmlu_pro",
    "wikitext103",
    "xstest",
    "strongreject_or_harmbench",
}


def _external_materials(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, Path]]:
    reviewed = tmp_path / "reviewed"
    language = tmp_path / "language"
    evaluator = tmp_path / "evaluator"
    for directory in (reviewed, language, evaluator):
        directory.mkdir()
        (directory / "payload").write_text(directory.name, encoding="utf-8")
    sources = {name: tmp_path / f"{name}.source" for name in _SOURCE_NAMES}
    for name, path in sources.items():
        path.write_text(name, encoding="utf-8")
    return reviewed, language, evaluator, sources


def _patch_verifiers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        e7_e8_inputs,
        "validate_reviewed_split_snapshot",
        lambda _path: {"manifest_digest": _DIGEST},
    )
    monkeypatch.setattr(
        e7_e8_inputs,
        "load_reviewed_language_suite",
        lambda _path: (object(),) * 500,
    )
    monkeypatch.setattr(
        e7_e8_inputs,
        "validate_ifeval_evaluator",
        lambda _path: "b" * 64,
    )
    monkeypatch.setattr(
        e7_e8_inputs,
        "verify_source_artifact",
        lambda _snapshot, path: Path(path),
    )


def test_e7_e8_external_inputs_are_atomic_replayable_and_cli_wired(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    reviewed, language, evaluator, sources = _external_materials(tmp_path)
    _patch_verifiers(monkeypatch)
    output = tmp_path / "E7-E8-external-inputs"

    staged = stage_e7_e8_external_inputs(
        output,
        reviewed_splits=reviewed,
        expected_reviewed_split_manifest_digest=_DIGEST,
        reviewed_language_suite=language,
        ifeval_evaluator=evaluator,
        source_artifacts=sources,
    )
    replayed = validate_e7_e8_external_inputs(
        output,
        expected_reviewed_split_manifest_digest=_DIGEST,
    )

    assert staged["sha256"] == replayed["sha256"] == sha256_path(output)
    assert set(staged["source_artifacts"]) == _SOURCE_NAMES
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["purpose"] == "e7-e8-external-input-snapshot"
    assert manifest["reviewed_split_manifest_digest"] == _DIGEST
    assert not tuple(output.parent.glob(f".{output.name}.stage-*"))

    parser = build_parser()
    stage = parser.parse_args(
        [
            "stage-e7-e8-inputs",
            "output",
            "reviewed",
            "language",
            "evaluator",
            "--triviaqa-source",
            "triviaqa",
            "--ifeval-source",
            "ifeval",
            "--mmlu-pro-source",
            "mmlu",
            "--wikitext103-source",
            "wikitext",
            "--xstest-source",
            "xstest",
            "--strongreject-source",
            "strongreject",
            "--expected-reviewed-split-manifest-digest",
            _DIGEST,
        ]
    )
    verify = parser.parse_args(
        [
            "verify-e7-e8-inputs",
            "output",
            "--expected-reviewed-split-manifest-digest",
            _DIGEST,
        ]
    )
    assert callable(stage.handler)
    assert callable(verify.handler)


def test_e7_e8_staging_rejects_symlinked_input_before_scientific_verification(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    reviewed, language, evaluator, sources = _external_materials(tmp_path)
    reviewed_link = tmp_path / "reviewed-link"
    reviewed_link.symlink_to(reviewed, target_is_directory=True)
    called = False

    def verifier(_path: Path) -> dict[str, str]:
        nonlocal called
        called = True
        return {"manifest_digest": _DIGEST}

    monkeypatch.setattr(e7_e8_inputs, "validate_reviewed_split_snapshot", verifier)
    with pytest.raises(DataValidationError, match="cannot traverse symlinks"):
        stage_e7_e8_external_inputs(
            tmp_path / "output",
            reviewed_splits=reviewed_link,
            expected_reviewed_split_manifest_digest=_DIGEST,
            reviewed_language_suite=language,
            ifeval_evaluator=evaluator,
            source_artifacts=sources,
        )
    assert not called


class _StopAfterNamespace(Exception):
    pass


def _capture_namespace(
    monkeypatch: pytest.MonkeyPatch,
    module: object,
) -> dict[str, Path]:
    observed: dict[str, Path] = {}

    def capture(paths: dict[str, Path]) -> dict[str, Path]:
        observed.update(paths)
        raise _StopAfterNamespace

    monkeypatch.setattr(module, "validate_active_study_artifact_paths", capture)
    return observed


def test_e6_e7_e8_treat_the_verified_model_snapshot_as_immutable_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "repository-model-snapshot"
    common = {
        "source": tmp_path / "runbook.json",
        "snapshot_directory": snapshot,
        "reviewed_splits": tmp_path / "reviewed",
        "reviewed_language_suite": tmp_path / "language",
        "ifeval_evaluator": tmp_path / "ifeval-evaluator",
        "runtime_artifact": tmp_path / "runtime",
        "execution_key_file": tmp_path / "key",
        "e3_static_vectors": tmp_path / "e3-vectors",
        "e5_adaptive_controllers": tmp_path / "e5-controller",
    }

    e6_paths = _capture_namespace(monkeypatch, e6_operator)
    with pytest.raises(_StopAfterNamespace):
        e6_operator._context(  # type: ignore[arg-type]
            SimpleNamespace(
                **common,
                frozen_question_bundle=tmp_path / "questions",
                prerequisite_runs={ExperimentPhase.E3: tmp_path / "E3"},
            )
        )
    assert snapshot not in e6_paths.values()

    sources = {name: tmp_path / name for name in _SOURCE_NAMES}
    e7_paths = _capture_namespace(monkeypatch, e7_operator)
    with pytest.raises(_StopAfterNamespace):
        e7_operator._base_context(  # type: ignore[arg-type]
            SimpleNamespace(
                **common,
                source_artifacts=sources,
                prerequisite_runs={},
                outputs={"run_directory": tmp_path / "E7"},
            )
        )
    assert snapshot not in e7_paths.values()

    e8_paths = _capture_namespace(monkeypatch, e8_operator)
    with pytest.raises(_StopAfterNamespace):
        e8_operator._base_context(  # type: ignore[arg-type]
            SimpleNamespace(
                **{key: value for key, value in common.items() if key != "e3_static_vectors"},
                source_artifacts=sources,
                e6_transition_evidence=tmp_path / "e6-transition",
                e7_finalization=tmp_path / "e7-final",
                prerequisite_runs={},
                outputs={"run_directory": tmp_path / "E8"},
            )
        )
    assert snapshot not in e8_paths.values()

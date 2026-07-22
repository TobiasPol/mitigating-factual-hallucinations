from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType
from unittest.mock import Mock, patch

import pytest

from mfh.artifact_namespace import (
    QWEN_STUDY_ARTIFACT_ROOT,
)
from mfh.artifact_namespace import (
    validate_active_study_artifact_paths as _real_namespace_validator,
)
from mfh.contracts import Question
from mfh.errors import ConfigurationError, DataValidationError, FrozenArtifactError
from mfh.experiments import e3_controls, e3_runner
from mfh.experiments.e3_operator import (
    E3OperatorRunbook,
    _E3Context,
    _E3Paths,
    advance_e3_operator,
    load_e3_operator_runbook,
    preflight_e3_operator,
    verify_e3_operator,
    write_e3_operator_runbook_template,
)
from mfh.experiments.protocol import ExperimentPhase

_CONSTRUCTION_FINGERPRINTS = MappingProxyType(
    {
        "reviewed_split_manifest_digest": "a" * 64,
        "review_result_manifest_digest": "b" * 64,
        "t_steer_question_ids_sha256": "c" * 64,
        "t_steer_questions_digest": "d" * 64,
    }
)


def _runbook(tmp_path: Path) -> E3OperatorRunbook:
    paths = {
        name: tmp_path / name
        for name in (
            "runbook.json",
            "model.yaml",
            "snapshot",
            "t-steer.jsonl",
            "t-dev.jsonl",
            "prompts.yaml",
            "study.yaml",
            "runtime.json",
            "E1-input",
            "E2-input",
            "reviewed-splits",
            "E1-run",
            "E2-run",
        )
    }
    return E3OperatorRunbook(
        path=paths["runbook.json"],
        model_config=paths["model.yaml"],
        snapshot_directory=paths["snapshot"],
        t_steer_questions=paths["t-steer.jsonl"],
        t_dev_questions=paths["t-dev.jsonl"],
        prompt_config=paths["prompts.yaml"],
        study_protocol=paths["study.yaml"],
        source_runtime_plan=paths["runtime.json"],
        output_root=tmp_path / "E3",
        input_artifacts=MappingProxyType(
            {
                "E1_outcome_labels": paths["E1-input"],
                "activation_feature_schemas": paths["E2-input"],
                "reviewed_splits": paths["reviewed-splits"],
            }
        ),
        prerequisite_runs=MappingProxyType({"E1": paths["E1-run"], "E2": paths["E2-run"]}),
        hidden_width=5_120,
        construction_checkpoint_rows=64,
        shuffle_checkpoint_rows=64,
        max_new_tokens=48,
        construction_input_fingerprints=_CONSTRUCTION_FINGERPRINTS,
        runbook_digest="a" * 64,
    )


def _context(runbook: E3OperatorRunbook) -> _E3Context:
    return _E3Context(
        runbook=runbook,
        study=Mock(),
        t_steer=(),
        t_dev=(),
        screen=(),
        construction_prompts=MappingProxyType({}),
        application_prompts=MappingProxyType({}),
        source_runtime_identity=MappingProxyType({"research_provenance": {"test": True}}),
        paths=_E3Paths(runbook.output_root),
    )


def _write_template(tmp_path: Path) -> Path:
    reviewed = Path.cwd().resolve() / QWEN_STUDY_ARTIFACT_ROOT / "frozen/reviewed-splits"
    question = Question(
        question_id="q-1",
        benchmark="triviaqa",
        text="Question?",
        aliases=("answer",),
        split="T-steer",
    )
    with (
        patch(
            "mfh.experiments.e3_operator.validate_reviewed_split_snapshot",
            return_value={
                "manifest_digest": "a" * 64,
                "review_result_manifest_digest": "b" * 64,
                "split_question_ids_sha256": {"T-steer": "c" * 64},
            },
        ),
        patch("mfh.experiments.e3_operator.read_questions", return_value=(question,)),
        patch("mfh.experiments.e3_operator.e3_questions_digest", return_value="d" * 64),
    ):
        return write_e3_operator_runbook_template(
            tmp_path / "e3-runbook.json", reviewed_splits=reviewed
        )


def test_e3_runbook_template_is_secret_free_and_round_trips(tmp_path: Path) -> None:
    path = _write_template(tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "HF_TOKEN" not in text and "api_key" not in text and ".env" not in text
    runbook = load_e3_operator_runbook(path)
    assert runbook.hidden_width == 5_120
    assert runbook.construction_input_fingerprints == _CONSTRUCTION_FINGERPRINTS
    expected_root = Path.cwd().resolve() / QWEN_STUDY_ARTIFACT_ROOT
    assert runbook.output_root.is_relative_to(expected_root)
    assert all(path.is_relative_to(expected_root) for path in runbook.input_artifacts.values())
    assert all(path.is_relative_to(expected_root) for path in runbook.prerequisite_runs.values())


def test_e3_runbook_rejects_noncanonical_paths(tmp_path: Path) -> None:
    path = _write_template(tmp_path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["model_config"] = "configs/models/qwen.yaml"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(DataValidationError, match="canonical absolute"):
        load_e3_operator_runbook(path)


def test_e3_advance_performs_one_durable_action(tmp_path: Path) -> None:
    runbook = _runbook(tmp_path)
    context = _context(runbook)
    runtime = Mock()
    plan = {"plan_identity": "b" * 64}
    with (
        patch("mfh.experiments.e3_operator.preflight_e3_operator"),
        patch("mfh.experiments.e3_operator._operator_context", return_value=context),
        patch("mfh.experiments.e3_operator._live_runtime", return_value=runtime),
        patch(
            "mfh.experiments.e3_operator.prepare_e3_construction_work",
            return_value=plan,
        ) as prepare,
    ):
        result = advance_e3_operator(runbook, request_budget=7)
    assert result == {
        "valid": True,
        "action": "prepared-construction",
        "plan_identity": "b" * 64,
    }
    prepare.assert_called_once()
    runtime.close.assert_called_once()


def test_e3_preflight_binds_e1_e2_lineage_and_exact_counts(tmp_path: Path) -> None:
    runbook = _runbook(tmp_path)
    context = _context(runbook)
    e1 = Mock()
    e1.verify_complete.return_value.phase = ExperimentPhase.E1
    e1.verify_complete.return_value.completion_digest = "1" * 64
    e2 = Mock()
    e2.verify_complete.return_value.phase = ExperimentPhase.E2
    e2.verify_complete.return_value.completion_digest = "2" * 64
    e2.contract.prerequisite_digests = {"E1": "1" * 64}
    with (
        patch("mfh.experiments.e3_operator.validate_active_study_artifact_paths"),
        patch("mfh.experiments.e3_operator._operator_context", return_value=context),
        patch(
            "mfh.experiments.e3_operator.PhaseRunLedger.open",
            side_effect=(e1, e2),
        ),
    ):
        result = preflight_e3_operator(runbook)
    assert result["construction_rows"] == 60_000
    assert result["evaluation_rows"] == 129_500
    assert result["prerequisite_completion_digests"] == {
        "E1": "1" * 64,
        "E2": "2" * 64,
    }


def test_e3_preflight_rejects_output_outside_active_study(tmp_path: Path) -> None:
    runbook = _runbook(tmp_path)
    with (
        patch(
            "mfh.experiments.e3_operator.validate_active_study_artifact_paths",
            side_effect=_real_namespace_validator,
        ),
        pytest.raises(ConfigurationError, match="must stay inside artifacts/studies"),
    ):
        preflight_e3_operator(runbook)


def test_e3_execution_uses_approved_m4_max_memory_envelope() -> None:
    approved = 42_949_672_960
    assert approved == e3_runner._UNIFIED_MEMORY_BYTES
    assert approved == e3_controls._UNIFIED_MEMORY_BYTES


def test_e3_verify_never_finalizes_an_absent_phase(tmp_path: Path) -> None:
    runbook = _runbook(tmp_path)
    context = _context(runbook)
    with (
        patch("mfh.experiments.e3_operator.preflight_e3_operator"),
        patch("mfh.experiments.e3_operator._operator_context", return_value=context),
        patch("mfh.experiments.e3_operator._live_runtime") as runtime,
        patch("mfh.experiments.e3_operator.finalize_e3_phase") as finalize,
        pytest.raises(FrozenArtifactError, match="phase is absent"),
    ):
        verify_e3_operator(runbook)
    runtime.assert_not_called()
    finalize.assert_not_called()

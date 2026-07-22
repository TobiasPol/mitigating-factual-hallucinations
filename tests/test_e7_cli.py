from __future__ import annotations

import json
from pathlib import Path

import pytest

from mfh.cli import build_parser
from mfh.errors import DataValidationError
from mfh.experiments.e7_operator import E7Runbook, write_e7_runbook_template
from mfh.experiments.e8_operator import E8Runbook, write_e8_runbook_template
from mfh.provenance import sha256_file


def test_e7_terminal_and_evaluator_commands_are_wired() -> None:
    parser = build_parser()
    ifeval = parser.parse_args(["materialize-ifeval-evaluator", "ifeval"])
    strongreject = parser.parse_args(["materialize-strongreject-grader", "strongreject"])
    grade_strongreject = parser.parse_args(
        [
            "grade-strongreject-openrouter",
            "records.jsonl",
            "questions.jsonl",
            "grader",
            "scorer.json",
            "scorer.key",
            "graded",
            "--request-budget",
            "5",
        ]
    )
    finalize = parser.parse_args(
        [
            "finalize-e7",
            "ledger",
            "coordinate",
            "sae",
            "terminal",
            "--study-protocol",
            "phases.yaml",
        ]
    )
    verify = parser.parse_args(["verify-e7", "terminal"])
    finalize_e8 = parser.parse_args(
        [
            "finalize-e8",
            "ledger",
            "protected",
            "registry",
            "screen",
            "runtime",
            "terminal-e8",
        ]
    )
    verify_e8 = parser.parse_args(["verify-e8", "terminal-e8"])

    assert ifeval.handler.__name__ == "_materialize_ifeval_evaluator"
    assert strongreject.handler.__name__ == "_materialize_strongreject_grader"
    assert grade_strongreject.handler.__name__ == "_grade_strongreject_openrouter"
    assert grade_strongreject.request_budget == 5
    assert finalize.handler.__name__ == "_finalize_e7"
    assert str(finalize.study_protocol) == "phases.yaml"
    assert verify.handler.__name__ == "_verify_e7"
    assert finalize_e8.handler.__name__ == "_finalize_e8"
    assert verify_e8.handler.__name__ == "_verify_e8"


def test_e7_runbook_template_round_trips_without_secrets(tmp_path: Path) -> None:
    path = tmp_path / "operator-inputs" / "E7-runbook.json"
    digest = write_e7_runbook_template(path, m1_layer=47)
    raw = json.loads(path.read_text(encoding="utf-8"))
    runbook = E7Runbook.load(path)

    assert digest == sha256_file(path)
    assert raw["phase"] == "E7"
    assert raw["model_config"].endswith("qwen3.6-27b-nvfp4.yaml")
    assert raw["sae_training_rows"] == 10_000
    assert raw["sae_validation_rows"] == 2_000
    assert len(runbook.sae_configs) == 6
    assert {item.top_k for item in runbook.sae_configs} == {16, 32, 64}
    assert {item.expansion_factor for item in runbook.sae_configs} == {8}
    assert runbook.m1_tensor_index == ("P0-neutral", "M1-P", "post_mlp", 47)
    serialized = json.dumps(raw)
    assert "OPENROUTER_API_KEY" not in serialized
    assert "private_key" not in serialized


def test_e7_cli_wires_complete_staged_lifecycle() -> None:
    parser = build_parser()
    commands = {
        "write-e7-runbook": ["runbook.json", "--m1-layer", "31"],
        "preflight-e7": ["runbook.json"],
        "prepare-e7": ["runbook.json"],
        "capture-e7": ["runbook.json", "T-steer", "--limit", "64"],
        "screen-e7-coordinate": ["runbook.json", "--limit", "5"],
        "fit-e7-sae": ["runbook.json"],
        "audit-e7-causal": ["runbook.json", "--limit", "3"],
        "audit-e7-interpretability": ["runbook.json", "--limit", "3"],
        "promote-e7-sae": ["runbook.json"],
        "prepare-e7-ledger": ["runbook.json"],
        "run-e7": ["runbook.json", "--limit", "3"],
        "finalize-e7-runbook": ["runbook.json"],
        "verify-e7-runbook": ["runbook.json"],
    }
    for command, arguments in commands.items():
        parsed = parser.parse_args([command, *arguments])
        assert callable(parsed.handler)

    capture = parser.parse_args(["capture-e7", "runbook.json", "sae-train"])
    assert capture.partition == "sae-train"
    assert parser.parse_args(["run-e7", "runbook.json", "--limit", "3"]).limit == 3


def test_e8_runbook_template_round_trips_without_secrets(tmp_path: Path) -> None:
    path = tmp_path / "operator-inputs" / "E8-runbook.json"
    digest = write_e8_runbook_template(path, m1_layer=47)
    raw = json.loads(path.read_text(encoding="utf-8"))
    runbook = E8Runbook.load(path)

    assert digest == sha256_file(path)
    assert raw["phase"] == "E8"
    assert raw["model_config"].endswith("qwen3.6-27b-nvfp4.yaml")
    assert raw["source_artifacts"]["triviaqa"].endswith(".parquet")
    assert raw["candidate_question_count"] == 500
    assert raw["variant_factual_rows"] == 500
    assert raw["variant_protected_rows"] == 100
    assert runbook.m5_alpha == 0.5
    assert runbook.matching_tolerance == 0.02
    assert runbook.m1_tensor_index == ("P0-neutral", "M1-P", "post_mlp", 47)
    serialized = json.dumps(raw)
    assert "OPENROUTER_API_KEY" not in serialized
    assert "private_key" not in serialized

    raw["max_new_tokens"] = 31
    invalid = tmp_path / "invalid-e8-runbook.json"
    invalid.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(DataValidationError, match="schema version 1"):
        E8Runbook.load(invalid)


def test_e8_cli_wires_complete_staged_lifecycle() -> None:
    parser = build_parser()
    commands = {
        "write-e8-runbook": ["runbook.json", "--m1-layer", "31"],
        "preflight-e8": ["runbook.json"],
        "prepare-e8": ["runbook.json"],
        "capture-e8-activations": ["runbook.json", "--limit", "64"],
        "screen-e8-variants": ["runbook.json", "--limit", "5"],
        "promote-e8-protected": ["runbook.json"],
        "screen-e8-candidates": ["runbook.json", "--limit", "3"],
        "prepare-e8-ledger": ["runbook.json"],
        "run-e8": ["runbook.json", "--limit", "3"],
        "finalize-e8-runbook": ["runbook.json"],
        "verify-e8-runbook": ["runbook.json"],
    }
    for command, arguments in commands.items():
        parsed = parser.parse_args([command, *arguments])
        assert callable(parsed.handler)

    assert (
        parser.parse_args(["capture-e8-activations", "runbook.json", "--limit", "64"]).limit == 64
    )
    assert parser.parse_args(["run-e8", "runbook.json", "--limit", "3"]).limit == 3


def test_post_e8_robustness_boundary_commands_are_wired() -> None:
    parser = build_parser()
    commands = {
        "freeze-robustness-plan": [
            "plan",
            "e1-run",
            "components",
            "scripts",
            "graders",
            "reviewed-splits",
            "--env-file",
            ".env",
        ],
        "verify-robustness-plan": ["plan"],
        "create-robustness-results": ["results", "plan"],
        "prepare-robustness-execution": [
            "execution",
            "results",
            "e9.json",
            "e3-construction",
        ],
        "run-robustness-rq1-capture": [
            "execution",
            "results",
            "e9.json",
            "e3-construction",
            "--limit",
            "16",
        ],
        "verify-robustness-rq1-capture": [
            "execution",
            "results",
            "e9.json",
            "e3-construction",
            "--require-complete",
        ],
        "run-robustness-prompts": ["results", "e9.json", "--limit", "8"],
        "run-robustness-rq1": [
            "execution",
            "results",
            "e9.json",
            "e3-construction",
            "e2-workspace",
            "e2-probes",
            "e5-capture",
            "e5-labels",
            "T-controller-train.jsonl",
            "--limit",
            "1",
        ],
        "verify-robustness-results": ["results", "--require-complete"],
        "finalize-robustness-results": ["results"],
    }
    for command, arguments in commands.items():
        assert callable(parser.parse_args([command, *arguments]).handler)
    frozen = parser.parse_args(
        [
            "freeze-robustness-plan",
            "plan",
            "e1-run",
            "components",
            "scripts",
            "graders",
            "reviewed-splits",
        ]
    )
    created = parser.parse_args(["create-robustness-results", "results", "plan"])
    expected = Path("configs/experiments/robustness-diagnostics.json")
    assert frozen.config == expected
    assert created.config == expected
    assert expected.is_file()


def test_e9_freeze_operator_is_wired() -> None:
    parser = build_parser()
    staged = parser.parse_args(
        [
            "stage-e9-inputs",
            "staged",
            "e1-graders",
            "reviewed-splits",
            "--triviaqa-source",
            "trivia.parquet",
            "--simpleqa-source",
            "simpleqa.csv",
            "--aa-source",
            "aa.csv",
            "--expected-official-grader-manifest-digest",
            "a" * 64,
        ]
    )
    args = parser.parse_args(
        [
            "freeze-e9-inputs",
            "freeze-suite",
            "e8-runbook.json",
            "e9-runbook.json",
            "evaluation-snapshot",
            "e1-graders",
            "reviewed-splits",
            "m2-source",
            "e3-phase",
            "--triviaqa-source",
            "trivia.parquet",
            "--simpleqa-source",
            "simpleqa.csv",
            "--aa-source",
            "aa.csv",
            "--expected-official-grader-manifest-digest",
            "a" * 64,
        ]
    )

    assert staged.handler.__name__ == "_stage_e9_inputs"
    assert args.handler.__name__ == "_freeze_e9_inputs"
    assert args.output == Path("freeze-suite")
    assert args.e8_runbook == Path("e8-runbook.json")
    assert args.e3_phase_run == Path("e3-phase")
    assert args.robustness_config == Path(
        "configs/experiments/robustness-diagnostics.json"
    )


def test_e10_freeze_operator_lifecycle_is_wired() -> None:
    parser = build_parser()
    prepare = parser.parse_args(
        ["prepare-e10-freezes", "suite", "e8.json", "e9.json"]
    )
    run = parser.parse_args(
        [
            "run-e10-early-probe",
            "suite",
            "e8.json",
            "e9.json",
            "--limit",
            "64",
        ]
    )
    verify = parser.parse_args(
        ["verify-e10-early-probe", "suite", "--require-complete"]
    )
    finalize = parser.parse_args(
        [
            "finalize-e10-freezes",
            "suite",
            "e8.json",
            "e9.json",
            "e10-snapshot",
            "e10.json",
        ]
    )

    assert prepare.handler.__name__ == "_prepare_e10_freezes"
    assert run.handler.__name__ == "_run_e10_early_probe"
    assert run.limit == 64
    assert verify.handler.__name__ == "_verify_e10_early_probe"
    assert verify.require_complete is True
    assert finalize.handler.__name__ == "_finalize_e10_freezes"


def test_final_analysis_writer_is_wired() -> None:
    parsed = build_parser().parse_args(
        [
            "write-analysis",
            "final",
            "evidence",
            "analysis.yaml",
            "research.md",
            "study.yaml",
            "E1",
            "E3",
            "E6",
            "E7",
            "E8",
            "E9",
            "E10",
            "robustness",
            "audit-queue",
            "audit-results",
            "aa-official",
            "--expected-aa-official-manifest-digest",
            "a" * 64,
            "--blinding-key-file",
            "audit.key",
        ]
    )
    assert parsed.handler.__name__ == "_write_analysis"

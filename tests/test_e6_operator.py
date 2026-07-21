from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import mfh.experiments.e6_operator as e6_operator
import mfh.experiments.runner as runner
from mfh.cli import build_parser
from mfh.config import load_model_spec, load_prompt_specs
from mfh.contracts import (
    ActivationSite,
    AdaptivePolicySpec,
    GenerationRecord,
    Outcome,
    Question,
    Runtime,
    TokenScope,
)
from mfh.data.io import write_questions
from mfh.data.source_snapshots import SOURCE_SNAPSHOTS
from mfh.experiments.e6_operator import (
    E6Runbook,
    _adaptive_state_factory,
    _condition,
    freeze_e6_question_bundle,
    write_e6_runbook_template,
)
from mfh.experiments.protocol import ExperimentPhase, load_study_protocol
from mfh.inference.architecture import HookKey
from mfh.inference.mlx_research import MlxResearchInterventionState
from mfh.provenance import sha256_file


def test_e6_runbook_template_round_trips_without_embedding_secrets(
    tmp_path: Path,
) -> None:
    path = tmp_path / "operator-inputs" / "E6-runbook.json"
    fingerprint = write_e6_runbook_template(path, m1_layer=47)
    value = json.loads(path.read_text(encoding="utf-8"))
    runbook = E6Runbook.load(path)

    assert fingerprint == sha256_file(path)
    assert value["phase"] == "E6"
    assert set(value["prerequisite_runs"]) == {"E3", "E5"}
    assert "private_key" not in json.dumps(value)
    assert (
        runbook.execution_key_file
        == (path.parent / "../secrets/execution-private-key.hex").resolve()
    )
    assert runbook.environment_file == (path.parent / "../../../../.env").resolve()
    assert "OPENROUTER_API_KEY" not in json.dumps(value)
    assert runbook.m1_tensor_index == ("P0-neutral", "M1-P", "post_mlp", 47)


def test_e6_cli_wires_the_complete_operator_lifecycle() -> None:
    parser = build_parser()
    commands = {
        "write-e6-runbook": ["runbook.json", "--m1-layer", "31"],
        "freeze-e6-questions": [
            "questions",
            "reviewed",
            "--triviaqa-source",
            "trivia.parquet",
            "--simpleqa-source",
            "simpleqa.csv",
            "--aa-source",
            "aa.csv",
            "--expected-reviewed-split-manifest-digest",
            "a" * 64,
        ],
        "preflight-e6": ["runbook.json"],
        "prepare-e6": ["runbook.json"],
        "attest-e6-runtime": ["runbook.json"],
        "run-e6": ["runbook.json", "--limit", "3"],
        "finalize-e6": ["runbook.json"],
        "verify-e6": ["runbook.json"],
    }
    for command, arguments in commands.items():
        parsed = parser.parse_args([command, *arguments])
        assert callable(parsed.handler)
    assert parser.parse_args(["run-e6", "runbook.json", "--limit", "3"]).limit == 3


def test_freeze_e6_questions_binds_reviewed_schedule_and_raw_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    reviewed = tmp_path / "reviewed"
    reviewed.mkdir()
    specifications = {
        "triviaqa": ("T-dev.jsonl", "triviaqa:q1", "T-dev"),
        "simpleqa_verified": ("simpleqa-eval.jsonl", "simpleqa:q1", "eval"),
        "aa_omniscience_public_600": ("aa-eval.jsonl", "aa-public:q1", "public"),
    }
    for benchmark, (filename, question_id, split) in specifications.items():
        write_questions(
            reviewed / filename,
            (Question(question_id, benchmark, "Question?", ("answer",), split),),
        )
    sources = {benchmark: tmp_path / f"{benchmark}.source" for benchmark in specifications}
    for benchmark, path in sources.items():
        path.write_text(benchmark, encoding="utf-8")

    digest = "a" * 64
    captured: dict[str, object] = {}
    monkeypatch.setattr(e6_operator, "_QUESTION_COUNTS", dict.fromkeys(specifications, 1))
    monkeypatch.setattr(
        e6_operator,
        "validate_reviewed_split_snapshot",
        lambda _path: {"manifest_digest": digest},
    )
    monkeypatch.setattr(e6_operator, "validate_active_model_spec", lambda _model: None)

    def write_bundle(
        directory: Path,
        contract: object,
        questions: object,
        *,
        source_artifacts: object,
    ) -> str:
        captured.update(
            {
                "directory": directory,
                "contract": contract,
                "questions": questions,
                "source_artifacts": source_artifacts,
            }
        )
        return "f" * 64

    monkeypatch.setattr(e6_operator, "write_frozen_question_bundle", write_bundle)
    output = tmp_path / "E6-questions"
    result = freeze_e6_question_bundle(
        output,
        reviewed_splits=reviewed,
        expected_reviewed_split_manifest_digest=digest,
        source_artifacts=sources,
    )

    contract = captured["contract"]
    assert result["sha256"] == "f" * 64
    assert contract.phase is ExperimentPhase.E6  # type: ignore[union-attr]
    assert set(contract.question_ids_by_benchmark) == set(specifications)  # type: ignore[union-attr]
    assert {item.steering_method for item in contract.conditions} == {"M0"}  # type: ignore[union-attr]
    assert captured["source_artifacts"] == sources


def test_freeze_e6_questions_rejects_unapproved_reviewed_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        e6_operator,
        "validate_reviewed_split_snapshot",
        lambda _path: {"manifest_digest": "b" * 64},
    )
    with pytest.raises(Exception, match="approved snapshot"):
        freeze_e6_question_bundle(
            tmp_path / "output",
            reviewed_splits=tmp_path / "reviewed",
            expected_reviewed_split_manifest_digest="a" * 64,
            source_artifacts={
                "triviaqa": tmp_path / "trivia",
                "simpleqa_verified": tmp_path / "simpleqa",
                "aa_omniscience_public_600": tmp_path / "aa",
            },
        )


def test_question_bundle_membership_allows_only_registered_partition_relabel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    canonical = Question(
        "triviaqa:q1",
        "triviaqa",
        "Question?",
        ("answer",),
        "train",
        metadata={"source": "pinned"},
    )
    reviewed = replace(canonical, split="T-dev")
    contract = SimpleNamespace(
        conditions=(SimpleNamespace(benchmark="triviaqa", partition="T-dev"),)
    )
    observed: tuple[Question, ...] | None = None

    def validate(_snapshot: object, _path: Path, questions: tuple[Question, ...]) -> None:
        nonlocal observed
        observed = questions

    monkeypatch.setattr(runner, "validate_source_membership", validate)
    runner._validate_partition_bound_source_membership(  # type: ignore[arg-type]
        SOURCE_SNAPSHOTS["triviaqa"],
        tmp_path / "source.parquet",
        (reviewed,),
        contract=contract,
        benchmark="triviaqa",
    )
    assert observed == (canonical,)

    unrelated = replace(reviewed, split="unregistered")
    runner._validate_partition_bound_source_membership(  # type: ignore[arg-type]
        SOURCE_SNAPSHOTS["triviaqa"],
        tmp_path / "source.parquet",
        (unrelated,),
        contract=contract,
        benchmark="triviaqa",
    )
    assert observed == (unrelated,)


def test_e6_conditions_bind_raw_m1_strength_and_cross_prompt_controller_source() -> None:
    study = load_study_protocol("configs/experiments/phases.yaml")
    model = load_model_spec("configs/models/qwen3.6-27b-mlx-4bit.yaml")
    prompt = {item.prompt_id: item for item in load_prompt_specs("configs/prompts/primary.yaml")}[
        "P3-forced-answer"
    ]
    policy = AdaptivePolicySpec(
        schema_version=2,
        release_risk_threshold=0.4,
        abstention_probability_threshold=0.7,
        alpha_max=0.5,
        alpha_beta=12.0,
        layer=None,
        site=None,
        token_scope=None,
        direction_sha256=None,
        direction_norm=None,
        execution_public_key="1" * 64,
        controller_artifact_sha256="2" * 64,
        candidate_layers=(31,),
        candidate_sites=(ActivationSite.POST_MLP,),
        candidate_token_scopes=(TokenScope.FIRST_FOUR,),
        vector_count=1,
        likely_unknown_risk_threshold=0.8,
        alpha_mode="fixed",
        alpha_risk_threshold=0.4,
    )
    common = {
        "study": study,
        "model": model,
        "prompt": prompt,
        "benchmark": "triviaqa",
        "seed": 17,
        "comparison_group": "e6-triviaqa-P3-frozen-P0-source",
        "m1_artifact": "3" * 64,
        "m1_layer": 31,
        "m1_site": ActivationSite.POST_MLP,
        "m1_scope": TokenScope.FIRST_FOUR,
        "m1_raw_alpha": 0.125,
        "e5_artifact": "4" * 64,
        "adaptive_policy": policy,
    }
    baseline = _condition(**common, method="M0")  # type: ignore[arg-type]
    fixed = _condition(**common, method="M1")  # type: ignore[arg-type]
    adaptive = _condition(**common, method="M3")  # type: ignore[arg-type]

    assert baseline.phase is ExperimentPhase.E6
    assert fixed.alpha == 0.125
    assert fixed.layer == 31
    assert adaptive.adaptive_policy == policy
    assert adaptive.system_prompt_id == "P3-forced-answer"
    assert {item.comparison_group for item in (baseline, fixed, adaptive)} == {
        "e6-triviaqa-P3-frozen-P0-source"
    }


def test_e6_adaptive_teacher_forcing_reuses_generation_effective_magnitude() -> None:
    raw = np.array([2.0, 0.0], dtype=np.float32)
    normalized = np.ascontiguousarray(raw / np.linalg.norm(raw))
    hook = HookKey(1, ActivationSite.POST_MLP)

    class Controller:
        @staticmethod
        def decide(_features: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(directions={hook: torch.from_numpy(raw).unsqueeze(0)})

    class RuntimeStub:
        @staticmethod
        def standardized_intervention_state(
            direction: np.ndarray,
            *,
            standardized_alpha: float,
            reference_rms: float,
            token_scope: TokenScope,
        ) -> MlxResearchInterventionState:
            assert np.array_equal(direction, normalized)
            assert reference_rms == 1.0
            return MlxResearchInterventionState(
                direction=direction,
                alpha=standardized_alpha,
                token_scope=token_scope,
            )

    record = GenerationRecord(
        question_id="q1",
        benchmark="triviaqa",
        model_repository="model/repository",
        model_revision="0" * 40,
        runtime=Runtime.MLX,
        quantization="4bit",
        system_prompt_id="P0-neutral",
        rendered_prompt_hash="1" * 64,
        steering_method="M3",
        layer=1,
        site=ActivationSite.POST_MLP,
        token_scope=TokenScope.FIRST_GENERATED,
        alpha=0.5,
        sparsity=0.25,
        controller_scores={"C": 0.2, "I": 0.7, "A": 0.1},
        raw_output="answer",
        normalized_answer="answer",
        outcome=Outcome.INCORRECT,
        generation_latency_seconds=1.0,
        input_tokens=2,
        output_tokens=1,
        condition_id="2" * 64,
        seed=17,
        metadata={
            "policy_action": "intervene",
            "adaptive_controller_evidence": {"feature_values": [0.25, -0.5]},
            "intervention_trace": {
                "direction_sha256": hashlib.sha256(normalized.tobytes()).hexdigest(),
                "direction_norm": 2.0,
                "alpha": 0.5,
            },
        },
    )
    factory = _adaptive_state_factory(  # type: ignore[arg-type]
        SimpleNamespace(controller=Controller()),
        record,
        RuntimeStub(),
    )
    assert factory is not None
    state = factory()[1]
    assert state.alpha == pytest.approx(1.0)

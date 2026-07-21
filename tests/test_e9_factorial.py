from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from mfh.config import load_model_spec, load_prompt_specs
from mfh.contracts import (
    ActivationSite,
    AdaptivePolicySpec,
    InterventionSpec,
    PromptSpec,
    Question,
    TokenScope,
)
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.experiments.e9_factorial import build_e9_contract
from mfh.experiments.protocol import load_study_protocol
from mfh.experiments.runner import (
    PhaseRunContract,
    _confirmatory_prompt_snapshot_body,
    _load_confirmatory_prompt_snapshot,
)
from mfh.provenance import stable_hash

ROOT = Path(__file__).parents[1]


def _questions() -> dict[str, tuple[Question, ...]]:
    counts = {
        "triviaqa": 5_000,
        "simpleqa_verified": 1_000,
        "aa_omniscience_public_600": 600,
    }
    return {
        benchmark: tuple(
            Question(
                question_id=f"{benchmark}:{index}",
                benchmark=benchmark,
                text=f"Question {index}?",
                aliases=(f"answer-{index}",),
            )
            for index in range(count)
        )
        for benchmark, count in counts.items()
    }


def _interventions() -> dict[str, InterventionSpec]:
    adaptive = AdaptivePolicySpec(
        release_risk_threshold=0.2,
        abstention_probability_threshold=0.8,
        alpha_max=1.0,
        alpha_beta=4.0,
        layer=21,
        site=ActivationSite.POST_MLP,
        token_scope=TokenScope.FIRST_FOUR,
        direction_sha256="7" * 64,
        direction_norm=1.0,
        execution_public_key="8" * 64,
    )
    fixed = {
        method: InterventionSpec(
            method=method,
            artifact_sha256=digit * 64,
            layer=21,
            site=ActivationSite.POST_MLP,
            token_scope=TokenScope.FIRST_FOUR,
            alpha=1.0,
            sparsity=0.1 if method == "M4" else None,
        )
        for method, digit in (("M1", "1"), ("M2", "2"), ("M4", "4"), ("M5", "5"))
    }
    return {
        "M0": InterventionSpec(method="M0"),
        "M3": InterventionSpec(
            method="M3",
            artifact_sha256="3" * 64,
            adaptive_policy=adaptive,
        ),
        **fixed,
    }


def _prompts() -> dict[str, PromptSpec]:
    available = {
        value.prompt_id: value
        for value in load_prompt_specs(ROOT / "configs/prompts/primary.yaml")
    }
    return {
        name: available[name]
        for name in ("P0-neutral", "P1-direct", "P2-calibrated-abstention")
    }


def _build(questions: dict[str, tuple[Question, ...]]) -> PhaseRunContract:
    study = load_study_protocol(ROOT / "configs/experiments/phases.yaml")
    phase = study.phase("E9")
    model = load_model_spec(ROOT / "configs/models/qwen3.6-27b-mlx-4bit.yaml")
    prompts = _prompts()
    return build_e9_contract(
        study=study,
        model=model,
        prompts=prompts,
        questions_by_benchmark=questions,
        interventions=_interventions(),
        input_fingerprints={name: "a" * 64 for name in phase.required_inputs},
        prerequisite_digests={value.value: "b" * 64 for value in phase.prerequisites},
    )


def test_e9_builder_materializes_only_the_preregistered_118800_rows() -> None:
    contract = _build(_questions())
    assert len(contract.conditions) == 54
    assert contract.expected_record_count == 118_800
    assert {value.partition for value in contract.conditions} == {
        "T-test",
        "simpleqa-eval",
        "aa-eval",
    }
    assert {value.steering_method for value in contract.conditions} == {
        "M0",
        "M1",
        "M2",
        "M3",
        "M4",
        "M5",
    }


def test_e9_builder_rejects_cross_benchmark_question_identity_reuse() -> None:
    questions = _questions()
    simple = questions["simpleqa_verified"]
    questions["simpleqa_verified"] = (
        replace(simple[0], question_id=questions["triviaqa"][0].question_id),
        *simple[1:],
    )
    with pytest.raises(DataValidationError, match="question schedule differs"):
        _build(questions)


def test_e9_prompt_snapshot_is_self_contained_and_contract_bound(tmp_path: Path) -> None:
    contract = _build(_questions())
    body = _confirmatory_prompt_snapshot_body(_prompts(), contract)
    path = tmp_path / "confirmatory-prompts.json"
    path.write_text(
        json.dumps({**body, "snapshot_digest": stable_hash(body)}),
        encoding="utf-8",
    )
    assert set(_load_confirmatory_prompt_snapshot(path, contract)) == set(_prompts())

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["prompts"][0]["text"] += " tampered"
    payload["prompts"][0]["text_sha256"] = hashlib.sha256(
        payload["prompts"][0]["text"].encode("utf-8")
    ).hexdigest()
    changed_body = dict(payload)
    changed_body.pop("snapshot_digest")
    payload["snapshot_digest"] = stable_hash(changed_body)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FrozenArtifactError, match="condition matrix"):
        _load_confirmatory_prompt_snapshot(path, contract)

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from mfh.contracts import Question
from mfh.data.io import read_questions, write_questions
from mfh.errors import ConfigurationError, DataValidationError, FrozenArtifactError
from mfh.experiments import robustness_diagnostics as diagnostics
from mfh.experiments.confirmatory_graders import ConfirmatoryGraderBundle
from mfh.provenance import canonical_json, sha256_path, stable_hash

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/experiments/robustness-diagnostics.json"
PROMPTS = ROOT / "configs/prompts/primary.yaml"
_COMPLETION_PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
_COMPLETION_PUBLIC_KEY = _COMPLETION_PRIVATE_KEY.public_key().public_bytes(
    serialization.Encoding.Raw,
    serialization.PublicFormat.Raw,
).hex()


def _question(benchmark: str, partition: str, index: int) -> Question:
    identity = f"{benchmark}-{partition}-{index}"
    return Question(
        question_id=identity,
        benchmark=benchmark,
        text=f"Unique factual question {identity}?",
        aliases=(f"Unique answer {identity}",),
        split=partition,
        entities=(f"Unique entity {identity}",),
    )


def _mock_e1_provenance(**values: object) -> dict[str, object]:
    e1_value = values["e1_run"]
    assert isinstance(e1_value, str | Path)
    e1_run = Path(e1_value)
    if e1_run.is_file():
        portable = json.loads(e1_run.read_text(encoding="utf-8"))
        portable.pop("binding_digest")
        return {**portable, "e1_binding_sha256": sha256_path(e1_run)}
    body: dict[str, object] = {
        "schema_version": 2,
        "kind": "signed-portable-complete-qwen-e1-binding",
        "study_protocol_digest": "f" * 64,
        "e1_completion_digest": "1" * 64,
        "e1_contract_digest": "2" * 64,
        "e1_record_set_digest": "3" * 64,
        "e1_record_count": 19_800,
        "e1_ledger_sha256": sha256_path(e1_run),
        "e1_condition_cells": [
            {
                "benchmark": benchmark,
                "partition": partition,
                "system_prompt_id": prompt,
                "steering_method": "M0",
                "seed": 17,
            }
            for benchmark, partition, prompt in sorted(
                (
                    benchmark,
                    partition,
                    prompt,
                )
                for benchmark, partition in {
                    "triviaqa": "T-controller",
                    "simpleqa_verified": "simpleqa-eval",
                    "aa_omniscience_public_600": "aa-eval",
                }.items()
                for prompt in {"P0-neutral", "P1-direct", "P2-calibrated-abstention"}
            )
        ],
        "e1_conditions_sha256": "4" * 64,
        "e1_question_ids_sha256": {
            "triviaqa": "5" * 64,
            "simpleqa_verified": "6" * 64,
            "aa_omniscience_public_600": "7" * 64,
        },
        "e1_input_fingerprints": {
            "deduplicated_splits": "8" * 64,
            "grader_bundle": "9" * 64,
            "inference_protocol": "a" * 64,
        },
        "e1_prerequisite_digests": {"E0": "b" * 64},
        "reviewed_split_manifest_digest": (
            "05e13f0193155551400fd636e8dd6d97e065dd80205133a9440ef13105bce148"
        ),
        "reviewed_split_sha256": (
            "3ceaf111654b80e34abd568853f64bba894fc7c6d7a81950c2868f3584a187f4"
        ),
        "completion_execution_public_key": _COMPLETION_PUBLIC_KEY,
    }
    signed = {
        **body,
        "completion_signature": _COMPLETION_PRIVATE_KEY.sign(
            canonical_json(body).encode()
        ).hex(),
    }
    return {
        **signed,
        "e1_binding_sha256": diagnostics._portable_e1_binding_sha256(signed),
    }


@pytest.fixture
def compact_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        diagnostics,
        "validate_active_study_artifact_paths",
        lambda paths: {name: Path(path).resolve() for name, path in paths.items()},
    )
    monkeypatch.setattr(
        diagnostics,
        "_EVALUATION_SOURCE_COUNTS",
        {
            "triviaqa": 205,
            "simpleqa_verified": 205,
            "aa_omniscience_public_600": 205,
        },
    )
    monkeypatch.setattr(
        diagnostics,
        "_RQ1_SOURCE_COUNTS",
        {"T-steer": 200, "T-controller": 1_200, "T-dev": 200},
    )
    monkeypatch.setattr(
        diagnostics,
        "validate_reviewed_split_snapshot",
        lambda _path: {"manifest_digest": "a" * 64},
    )
    monkeypatch.setattr(
        diagnostics,
        "validate_confirmatory_grader_bundle",
        lambda _path: SimpleNamespace(
            component_fingerprints={"runtime_attestation": "d" * 64},
            runtime_attestation={"execution_public_key": _COMPLETION_PUBLIC_KEY},
            scorer=SimpleNamespace(execution_public_key=_COMPLETION_PUBLIC_KEY),
        ),
    )
    monkeypatch.setattr(
        diagnostics,
        "validate_execution_snapshot",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        diagnostics,
        "_validate_robustness_component_selection",
        lambda _path, _prompts: "f" * 64,
    )
    monkeypatch.setattr(
        diagnostics,
        "_validate_e1_reviewed_split_binding",
        _mock_e1_provenance,
    )


@pytest.fixture
def source_artifacts(
    tmp_path: Path,
    compact_environment: None,
) -> dict[str, Path]:
    del compact_environment
    reviewed = tmp_path / "reviewed-splits"
    reviewed.mkdir()
    write_questions(
        reviewed / "T-test.jsonl",
        (_question("triviaqa", "T-test", index) for index in range(205)),
    )
    write_questions(
        reviewed / "simpleqa-eval.jsonl",
        (_question("simpleqa_verified", "simpleqa-eval", index) for index in range(205)),
    )
    write_questions(
        reviewed / "aa-eval.jsonl",
        (_question("aa_omniscience_public_600", "aa-eval", index) for index in range(205)),
    )
    for partition, count in (
        ("T-steer", 200),
        ("T-controller", 1_200),
        ("T-dev", 200),
    ):
        write_questions(
            reviewed / f"{partition}.jsonl",
            (_question("triviaqa", partition, index) for index in range(count)),
        )
    components = tmp_path / "components"
    components.mkdir()
    (components / "placeholder").write_text("components\n", encoding="utf-8")
    graders = tmp_path / "graders"
    graders.mkdir()
    (graders / "placeholder").write_text("graders\n", encoding="utf-8")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "snapshot-manifest.json").write_text(
        json.dumps({"study_protocol_digest": "f" * 64}),
        encoding="utf-8",
    )
    e1_run = tmp_path / "e1-run"
    e1_run.mkdir()
    (e1_run / "placeholder").write_text("complete E1\n", encoding="utf-8")
    return {
        "canonical-prompts": PROMPTS,
        "frozen-component-selection": components,
        "frozen-evaluation-scripts": scripts,
        "frozen-graders": graders,
        "e1-phase-ledger": e1_run,
        "triviaqa-evaluation": reviewed,
        "simpleqa_verified-evaluation": reviewed,
        "aa_omniscience_public_600-evaluation": reviewed,
        "triviaqa-development": reviewed,
    }


def test_builds_exact_deterministic_prompt_and_rq1_schedules(
    source_artifacts: dict[str, Path],
) -> None:
    config = diagnostics.load_robustness_diagnostic_config(CONFIG)
    plan = diagnostics.build_robustness_diagnostic_plan(
        config=config,
        source_artifacts=source_artifacts,
    )
    replay = diagnostics.build_robustness_diagnostic_plan(
        config=config,
        source_artifacts=dict(reversed(tuple(source_artifacts.items()))),
    )
    assert replay.plan_digest == plan.plan_digest
    prompt_tasks = tuple(diagnostics.iter_prompt_paraphrase_tasks(plan))
    rq1_tasks = tuple(diagnostics.iter_rq1_generalization_tasks(plan))
    assert len(prompt_tasks) == 36_000
    assert len({task.task_id for task in prompt_tasks}) == 36_000
    assert len(rq1_tasks) == 60
    assert len({task.task_id for task in rq1_tasks}) == 60
    assert {task.base_prompt_id for task in prompt_tasks} == {
        "P0-neutral",
        "P2-calibrated-abstention",
    }
    assert {task.method for task in prompt_tasks} == {
        "M0",
        "M1",
        "M2",
        "M3",
        "M4",
        "M5",
    }
    assert all(
        len({task.question_id for task in prompt_tasks if task.benchmark == benchmark}) == 200
        for benchmark in diagnostics._BENCHMARKS
    )
    assert {task.training_prompt_id for task in rq1_tasks} == {"P0-neutral"}
    assert {task.evaluation_prompt_id for task in rq1_tasks} == {
        "P0-neutral",
        "P2-calibrated-abstention",
    }
    question_sets = diagnostics.rq1_task_question_sets(plan, rq1_tasks[0])
    assert all(question_sets.values())
    assert not any(
        set(question_sets[left]) & set(question_sets[right])
        for left in question_sets
        for right in question_sets
        if left < right
    )
    assignment_ids = {
        row["question_id"]
        for row in plan.body["rq1_generalization"]["assignments"]
    }
    assert all(
        not question_id.startswith("simpleqa")
        and not question_id.startswith("aa_omniscience")
        for question_id in assignment_ids
    )
    with pytest.raises(TypeError):
        plan.body["prompt_paraphrase"]["variants"][0]["text"] = "mutated"


def test_plan_rejects_a_fold_without_held_vector_training_rows(
    source_artifacts: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = diagnostics.semantic_group_ids

    def missing_steer_fold(questions: object) -> dict[str, str]:
        values = tuple(cast(tuple[Question, ...], questions))
        normal = original(values)
        return {
            value.question_id: (
                "0000000000000001" + "0" * 48
                if value.split == "T-steer"
                else normal[value.question_id]
            )
            for value in values
        }

    monkeypatch.setattr(diagnostics, "semantic_group_ids", missing_steer_fold)
    with pytest.raises(DataValidationError, match="lacks a fitting or evaluation partition"):
        diagnostics.build_robustness_diagnostic_plan(
            config=diagnostics.load_robustness_diagnostic_config(CONFIG),
            source_artifacts=source_artifacts,
        )


def test_freeze_rebuilds_from_packaged_sources_and_rejects_forgery(
    tmp_path: Path,
    source_artifacts: dict[str, Path],
) -> None:
    frozen = diagnostics.freeze_robustness_diagnostic_plan(
        tmp_path / "plan-bundle",
        config_path=CONFIG,
        source_artifacts=source_artifacts,
    )
    assert frozen.path is not None
    assert (
        diagnostics.verify_robustness_diagnostic_plan(
            frozen.path,
        ).plan_digest
        == frozen.plan_digest
    )
    original_e1 = source_artifacts["e1-phase-ledger"]
    original_e1.rename(tmp_path / "original-e1-removed-after-freeze")
    assert (
        diagnostics.verify_robustness_diagnostic_plan(frozen.path).plan_digest
        == frozen.plan_digest
    )

    plan_path = frozen.path / "plan.json"
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["rq1_generalization"]["assignments"][0]["question_id"] = "forged"
    body = dict(payload)
    body.pop("plan_digest")
    payload["plan_digest"] = stable_hash(body)
    plan_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    bundle_path = frozen.path / "bundle.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["plan_digest"] = payload["plan_digest"]
    bundle_body = dict(bundle)
    bundle_body.pop("bundle_digest")
    bundle["bundle_digest"] = stable_hash(bundle_body)
    bundle_path.write_text(canonical_json(bundle) + "\n", encoding="utf-8")
    with pytest.raises(FrozenArtifactError):
        diagnostics.verify_robustness_diagnostic_plan(frozen.path)


def test_config_digest_and_exact_source_counts_are_enforced(
    tmp_path: Path,
    source_artifacts: dict[str, Path],
) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["prompt_paraphrase"]["variants"][0]["text"] = "Changed after freeze"
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="not approved"):
        diagnostics.load_robustness_diagnostic_config(changed)

    triviaqa = source_artifacts["triviaqa-evaluation"] / "T-test.jsonl"
    rows = triviaqa.read_text(encoding="utf-8").splitlines()
    triviaqa.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(Exception, match="question source is invalid"):
        diagnostics.build_robustness_diagnostic_plan(
            config=diagnostics.load_robustness_diagnostic_config(CONFIG),
            source_artifacts=source_artifacts,
        )


def test_portable_e1_binding_requires_frozen_execution_signature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reviewed = ROOT / "artifacts/splits/triviaqa-reviewed"
    study_digest = "f" * 64
    monkeypatch.setattr(
        diagnostics,
        "load_packaged_study_protocol",
        lambda _path: SimpleNamespace(digest=study_digest),
    )
    partitions = {
        "triviaqa": "T-controller",
        "simpleqa_verified": "simpleqa-eval",
        "aa_omniscience_public_600": "aa-eval",
    }
    question_digests = {
        benchmark: stable_hash(
            [question.question_id for question in read_questions(reviewed / f"{partition}.jsonl")]
        )
        for benchmark, partition in partitions.items()
    }
    cells = [
        {
            "benchmark": benchmark,
            "partition": partition,
            "system_prompt_id": prompt,
            "steering_method": "M0",
            "seed": 17,
        }
        for benchmark, partition, prompt in sorted(
            (benchmark, partition, prompt)
            for benchmark, partition in partitions.items()
            for prompt in {"P0-neutral", "P1-direct", "P2-calibrated-abstention"}
        )
    ]
    core: dict[str, object] = {
        "schema_version": 2,
        "kind": "signed-portable-complete-qwen-e1-binding",
        "study_protocol_digest": study_digest,
        "e1_completion_digest": "1" * 64,
        "e1_contract_digest": "2" * 64,
        "e1_record_set_digest": "3" * 64,
        "e1_record_count": 19_800,
        "e1_ledger_sha256": "4" * 64,
        "e1_condition_cells": cells,
        "e1_conditions_sha256": "5" * 64,
        "e1_question_ids_sha256": question_digests,
        "e1_input_fingerprints": {
            "deduplicated_splits": "6" * 64,
            "grader_bundle": "7" * 64,
            "inference_protocol": "8" * 64,
        },
        "e1_prerequisite_digests": {"E0": "9" * 64},
        "reviewed_split_manifest_digest": (
            "05e13f0193155551400fd636e8dd6d97e065dd80205133a9440ef13105bce148"
        ),
        "reviewed_split_sha256": (
            "3ceaf111654b80e34abd568853f64bba894fc7c6d7a81950c2868f3584a187f4"
        ),
        "completion_execution_public_key": _COMPLETION_PUBLIC_KEY,
    }
    signed = {
        **core,
        "completion_signature": _COMPLETION_PRIVATE_KEY.sign(
            canonical_json(core).encode()
        ).hex(),
    }
    portable = tmp_path / "e1-binding.json"
    portable.write_text(
        canonical_json({**signed, "binding_digest": stable_hash(signed)}) + "\n",
        encoding="utf-8",
    )
    grader_bundle = cast(
        ConfirmatoryGraderBundle,
        SimpleNamespace(
            scorer=SimpleNamespace(execution_public_key=_COMPLETION_PUBLIC_KEY)
        ),
    )
    result = diagnostics._validate_e1_reviewed_split_binding(
        config=diagnostics.load_robustness_diagnostic_config(CONFIG),
        e1_run=portable,
        evaluation_snapshot=tmp_path,
        snapshot_manifest={
            "study_protocol_digest": study_digest,
            "files": {"study_protocol_config": {"path": "study.json"}},
        },
        reviewed_paths=(reviewed, reviewed, reviewed, reviewed),
        grader_bundle=grader_bundle,
    )
    assert result["e1_question_ids_sha256"] == question_digests

    forged = dict(signed)
    forged["e1_completion_digest"] = "a" * 64
    portable.write_text(
        canonical_json({**forged, "binding_digest": stable_hash(forged)}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DataValidationError, match="portable Qwen E1"):
        diagnostics._validate_e1_reviewed_split_binding(
            config=diagnostics.load_robustness_diagnostic_config(CONFIG),
            e1_run=portable,
            evaluation_snapshot=tmp_path,
            snapshot_manifest={
                "study_protocol_digest": study_digest,
                "files": {"study_protocol_config": {"path": "study.json"}},
            },
            reviewed_paths=(reviewed, reviewed, reviewed, reviewed),
            grader_bundle=grader_bundle,
        )

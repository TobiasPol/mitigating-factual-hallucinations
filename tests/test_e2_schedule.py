from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from mfh.contracts import ModelSpec, Outcome, Question, Runtime
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.experiments.e2_schedule import (
    E2CaptureProtocol,
    E2ScheduleRow,
    build_e2_schedule,
    controller_feature_partitions,
    verify_e2_workspace,
    write_e2_workspace,
)
from mfh.provenance import stable_hash


def _question(
    question_id: str,
    benchmark: str,
    *,
    alias: str | None = None,
) -> Question:
    return Question(
        question_id=question_id,
        benchmark=benchmark,
        text=f"Question for {question_id}?",
        aliases=(alias or f"answer-{question_id}",),
    )


def _fixture() -> tuple[
    E2CaptureProtocol,
    list[Question],
    list[Question],
    list[Question],
    list[Question],
    dict[tuple[str, str], Outcome],
]:
    protocol = E2CaptureProtocol(
        controller_rows=4,
        controller_calibration_rows=2,
        dev_rows=3,
        simpleqa_rows=2,
        aa_rows=2,
        seed=17,
    )
    controller = [
        _question("controller-0", "triviaqa", alias="shared-controller-answer"),
        _question("controller-1", "triviaqa", alias="shared-controller-answer"),
        _question("controller-2", "triviaqa"),
        _question("controller-3", "triviaqa"),
    ]
    dev = [_question(f"dev-{index}", "triviaqa") for index in range(3)]
    simpleqa = [
        _question(f"simple-{index}", "simpleqa_verified") for index in range(2)
    ]
    aa = [
        _question(f"aa-{index}", "aa_omniscience_public_600") for index in range(2)
    ]
    outcomes = {
        (question.benchmark, question.question_id): (
            Outcome.CORRECT if index % 2 == 0 else Outcome.INCORRECT
        )
        for index, question in enumerate((*controller, *simpleqa, *aa))
    }
    return protocol, controller, dev, simpleqa, aa, outcomes


def _model() -> ModelSpec:
    return ModelSpec(
        name="bonsai-test",
        repository="prism-ml/Bonsai-27B-mlx-1bit",
        revision="e" * 40,
        runtime=Runtime.MLX,
        quantization="binary-g128-mlx-1bit",
        num_layers=64,
    )


def _schedule_fixture() -> tuple[E2CaptureProtocol, tuple[E2ScheduleRow, ...]]:
    protocol, controller, dev, simpleqa, aa, outcomes = _fixture()
    schedule = build_e2_schedule(
        controller=controller,
        dev=dev,
        simpleqa=simpleqa,
        aa=aa,
        e1_p0_outcomes=outcomes,
        protocol=protocol,
    )
    return protocol, schedule


def test_e2_schedule_has_exact_capture_and_generation_accounting() -> None:
    protocol, controller, dev, simpleqa, aa, outcomes = _fixture()
    partitions = controller_feature_partitions(
        controller,
        calibration_rows=protocol.controller_calibration_rows,
        seed=protocol.seed,
    )
    assert sum(value == "T-controller-calibration" for value in partitions.values()) == 2
    assert partitions["controller-0"] == partitions["controller-1"]

    schedule = build_e2_schedule(
        controller=controller,
        dev=dev,
        simpleqa=simpleqa,
        aa=aa,
        e1_p0_outcomes=outcomes,
        protocol=protocol,
    )
    assert len(schedule) == protocol.expected_capture_rows == 18
    assert sum(row.label_source == "generate" for row in schedule) == 10
    assert [row.sequence for row in schedule] == list(range(18))
    assert [row.prompt_id for row in schedule[:4]] == ["P0-neutral"] * 4
    assert [row.prompt_id for row in schedule[7:11]] == ["P3-forced-answer"] * 4
    assert all(row.outcome is not None for row in (*schedule[:4], *schedule[-4:]))
    assert all(row.outcome is None for row in schedule[4:14])


def test_e2_schedule_rejects_cross_partition_leakage_and_missing_e1_labels() -> None:
    protocol, controller, dev, simpleqa, aa, outcomes = _fixture()
    dev[0] = _question("dev-0", "triviaqa", alias="shared-controller-answer")
    with pytest.raises(DataValidationError, match="semantic groups overlap"):
        build_e2_schedule(
            controller=controller,
            dev=dev,
            simpleqa=simpleqa,
            aa=aa,
            e1_p0_outcomes=outcomes,
            protocol=protocol,
        )

    dev[0] = _question("dev-0", "triviaqa")
    outcomes.pop(("triviaqa", "controller-0"))
    with pytest.raises(DataValidationError, match="required E1 P0 outcome"):
        build_e2_schedule(
            controller=controller,
            dev=dev,
            simpleqa=simpleqa,
            aa=aa,
            e1_p0_outcomes=outcomes,
            protocol=protocol,
        )


def test_e2_workspace_is_frozen_verified_and_rejects_json_coercion() -> None:
    protocol, schedule = _schedule_fixture()
    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory) / "e2"
        verified = write_e2_workspace(
            workspace,
            schedule=schedule,
            protocol=protocol,
            model=_model(),
            hidden_width=4,
            input_fingerprints={"e1_completion": "1" * 64, "split_bundle": "2" * 64},
        )
        assert verified.plan_identity == verify_e2_workspace(workspace).plan_identity
        assert verified.activation_spec.expected_rows == 18
        assert verified.activation_spec.hidden_width == 4
        with pytest.raises(FrozenArtifactError, match="overwrite"):
            write_e2_workspace(
                workspace,
                schedule=schedule,
                protocol=protocol,
                model=_model(),
                hidden_width=4,
                input_fingerprints={"e1_completion": "1" * 64},
            )

        plan_path = workspace / "plan.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan.pop("plan_identity")
        plan["model"]["hidden_width"] = "4"
        plan_path.write_text(
            json.dumps({**plan, "plan_identity": stable_hash(plan)}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(FrozenArtifactError, match="invalid E2 workspace"):
            verify_e2_workspace(workspace)


def test_e2_protocol_rejects_boolean_schema_and_frozen_protocol_is_exact() -> None:
    with pytest.raises(DataValidationError, match="exact integers"):
        E2CaptureProtocol(schema_version=True)
    assert E2CaptureProtocol().scientific_eligible is True
    assert (
        E2CaptureProtocol(controller_rows=4, controller_calibration_rows=1).scientific_eligible
        is False
    )

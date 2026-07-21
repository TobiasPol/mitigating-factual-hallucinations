from __future__ import annotations

import pytest

from mfh.contracts import ActivationSite, Question, TokenScope
from mfh.data.splits import semantic_group_ids
from mfh.errors import DataValidationError
from mfh.experiments.e3_runner import _ordered_rows
from mfh.experiments.e3_schedule import (
    E3Condition,
    E3OperatingPoint,
    E3Protocol,
    build_e3_construction_schedule,
    e3_alpha_conditions,
    e3_control_conditions,
    e3_cross_prompt_conditions,
    e3_final_conditions,
    e3_geometry_conditions,
    e3_p3_conditions,
    e3_scope_conditions,
    e3_stage_row_counts,
    select_e3_screen_questions,
)


def _question(index: int, *, text: str | None = None, split: str = "T-steer") -> Question:
    return Question(
        question_id=f"q-{index}",
        benchmark="triviaqa",
        text=text or f"Question {index}?",
        aliases=(f"answer-{index}",),
        split=split,
    )


def _protocol() -> E3Protocol:
    return E3Protocol(
        steer_rows=4,
        dev_rows=6,
        screen_rows=2,
        candidate_layers=(1, 2),
        candidate_sites=(ActivationSite.POST_MLP,),
        standardized_alphas=(0.0, 0.5),
        token_scopes=(TokenScope.FINAL_PROMPT, TokenScope.FIRST_FOUR),
    )


def _points() -> dict[str, E3OperatingPoint]:
    return {
        method: E3OperatingPoint(
            method,
            1,
            ActivationSite.POST_MLP,
            0.5,
            TokenScope.FIRST_FOUR,
        )
        for method in ("M1-R", "M1-P")
    }


def test_e3_schedule_is_deterministically_randomized_across_conditions() -> None:
    conditions = e3_geometry_conditions(_protocol())
    question_ids = tuple(f"question-{index}" for index in range(12))
    first = _ordered_rows(conditions, question_ids)
    second = _ordered_rows(conditions, question_ids)
    identities = tuple((row[0].condition_id, row[1]) for row in first)
    assert first == second
    assert len(identities) == len(conditions) * len(question_ids)
    assert len(set(identities)) == len(identities)
    assert len({condition_id for condition_id, _ in identities[: len(question_ids)]}) > 1


def test_default_e3_staged_schedule_has_preregistered_operational_counts() -> None:
    counts = dict(e3_stage_row_counts())

    assert counts == {
        "geometry": 21_500,
        "alpha": 6_500,
        "scope": 6_500,
        "controls": 8_500,
        "cross-prompt": 5_000,
        "P3-diagnostic": 1_500,
        "final": 80_000,
    }
    assert sum(value for key, value in counts.items() if key != "final") == 49_500


def test_e3_construction_schedule_is_two_prompt_blocks_over_tsteer() -> None:
    protocol = _protocol()
    rows = build_e3_construction_schedule(
        tuple(_question(index) for index in range(4)), protocol=protocol
    )

    assert len(rows) == 8
    assert [row.sequence for row in rows] == list(range(8))
    assert [row.prompt_id for row in rows[:4]] == ["P0-neutral"] * 4
    assert [row.prompt_id for row in rows[4:]] == ["P2-calibrated-abstention"] * 4
    assert len({row.question_sha256 for row in rows}) == 4


def test_e3_screen_selection_is_exact_and_never_splits_semantic_groups() -> None:
    protocol = _protocol()
    questions = tuple(_question(index, split="T-dev") for index in range(6))
    selected = set(select_e3_screen_questions(questions, protocol=protocol))
    groups = semantic_group_ids(questions)

    assert len(selected) == 2
    for group in set(groups.values()):
        members = {
            question.question_id for question in questions if groups[question.question_id] == group
        }
        assert not (members & selected) or members <= selected


def test_e3_stage_conditions_are_unique_and_bind_selected_operating_points() -> None:
    protocol = _protocol()
    points = _points()
    stages = {
        "geometry": e3_geometry_conditions(protocol),
        "alpha": e3_alpha_conditions(points, protocol=protocol),
        "scope": e3_scope_conditions(points, protocol=protocol),
        "controls": e3_control_conditions(points, protocol=protocol),
        "cross": e3_cross_prompt_conditions(points, protocol=protocol),
        "p3": e3_p3_conditions(points, protocol=protocol),
        "final": e3_final_conditions(points, protocol=protocol),
    }

    assert {name: len(rows) for name, rows in stages.items()} == {
        "geometry": 5,
        "alpha": 5,
        "scope": 5,
        "controls": 17,
        "cross": 10,
        "p3": 3,
        "final": 16,
    }
    for rows in stages.values():
        assert len({row.condition_id for row in rows}) == len(rows)
    controls = stages["controls"]
    assert sum(row.control == "opposite" for row in controls) == 2
    assert all(row.standardized_alpha < 0 for row in controls if row.control == "opposite")
    assert {row.apply_prompt_id for row in stages["final"]} == {
        "P0-neutral",
        "P2-calibrated-abstention",
        "P3-forced-answer",
    }
    unrelated = next(row for row in stages["controls"] if row.control == "unrelated-layer")
    assert unrelated.source_layer == points["M1-R"].layer
    assert unrelated.layer != unrelated.source_layer
    assert unrelated.source_site is unrelated.site
    assert {row.method for row in stages["final"]} >= {
        "M0",
        "M1-R",
        "M1-P",
        "shuffled-label",
        "random-norm",
        "opposite",
        "unrelated-layer",
        "gaussian",
        "zero-hook",
        "cross-prompt",
    }


def test_e3_operating_points_fail_closed_outside_frozen_grid() -> None:
    points = _points()
    points["M1-P"] = E3OperatingPoint(
        "M1-P", 3, ActivationSite.POST_MLP, 0.5, TokenScope.FIRST_FOUR
    )
    with pytest.raises(DataValidationError, match="protocol grid"):
        e3_alpha_conditions(points, protocol=_protocol())


def test_e3_split_provenance_and_public_types_fail_closed() -> None:
    protocol = _protocol()
    with pytest.raises(DataValidationError, match="T-steer"):
        build_e3_construction_schedule(
            tuple(_question(index, split="T-test") for index in range(4)),
            protocol=protocol,
        )
    with pytest.raises(DataValidationError, match="T-dev"):
        select_e3_screen_questions(
            tuple(_question(index, split="T-steer") for index in range(6)),
            protocol=protocol,
        )
    with pytest.raises(DataValidationError, match="operating point"):
        E3OperatingPoint("M1-R", 1, ActivationSite.POST_MLP, 0.0, TokenScope.FIRST_FOUR)
    with pytest.raises(DataValidationError, match="protocol"):
        E3Protocol(standardized_alphas=(False, 0.5))
    with pytest.raises(DataValidationError, match="protocol"):
        E3Protocol(standardized_alphas=("0", 0.5))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("control", "alpha"),
    ((None, -0.5), ("opposite", 0.5), ("zero-hook", 0.5)),
)
def test_e3_condition_alpha_must_match_control_semantics(control: str | None, alpha: float) -> None:
    method = "M1-R" if control is None else control
    with pytest.raises(DataValidationError, match="causal semantics"):
        E3Condition(
            "final" if control is None else "controls",
            method,
            "M1-R",
            "P0-neutral",
            "P0-neutral",
            1,
            ActivationSite.POST_MLP,
            alpha,
            TokenScope.FIRST_FOUR,
            control=control,
        )

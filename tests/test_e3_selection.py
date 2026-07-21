from __future__ import annotations

import json
from pathlib import Path

import pytest

from mfh.contracts import ActivationSite, Outcome, TokenScope
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.experiments.e3_schedule import (
    E3Condition,
    E3OperatingPoint,
    E3Protocol,
    e3_alpha_conditions,
    e3_geometry_conditions,
    e3_scope_conditions,
)
from mfh.experiments.e3_selection import (
    E3StageSelection,
    derive_e3_stage_selection,
    verify_e3_stage_selection,
    write_e3_stage_selection,
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


def _matrix(
    conditions: tuple[E3Condition, ...],
    *,
    intervention_outcome: Outcome = Outcome.CORRECT,
) -> tuple[dict[tuple[str, str], Outcome], dict[tuple[str, str], float]]:
    questions = ("q-1", "q-2")
    outcomes = {
        (condition.condition_id, question): (
            Outcome.INCORRECT if condition.method == "M0" else intervention_outcome
        )
        for condition in conditions
        for question in questions
    }
    norms = {
        (condition.condition_id, question): abs(float(condition.standardized_alpha))
        for condition in conditions
        if condition.method != "M0"
        for question in questions
    }
    return outcomes, norms


def _derive(
    stage: str,
    conditions: tuple[E3Condition, ...],
    outcomes: dict[tuple[str, str], Outcome],
    norms: dict[tuple[str, str], float],
    predecessor: E3StageSelection | None = None,
) -> E3StageSelection:
    return derive_e3_stage_selection(
        stage=stage,
        conditions=conditions,
        question_ids=("q-1", "q-2"),
        outcomes=outcomes,
        actual_delta_norms=norms,
        source_plan_identity="a" * 64,
        evaluation_plan_identity="b" * 64,
        evaluation_record_chain_head="c" * 64,
        evaluation_record_set_digest="d" * 64,
        source_scientific_eligible=False,
        predecessor_selection=predecessor,
        protocol=_protocol(),
    )


def test_e3_selection_is_staged_deterministic_and_excludes_alpha_zero() -> None:
    protocol = _protocol()
    geometry = e3_geometry_conditions(protocol)
    outcomes, norms = _matrix(geometry)
    geometry_selection = _derive("geometry", geometry, outcomes, norms)

    assert geometry_selection.falsified is False
    assert {value.layer for value in geometry_selection.selected.values()} == {1}
    assert {value.site for value in geometry_selection.selected.values()} == {
        ActivationSite.POST_MLP
    }

    alpha = e3_alpha_conditions(geometry_selection.selected, protocol=protocol)
    outcomes, norms = _matrix(alpha)
    alpha_selection = _derive(
        "alpha", alpha, outcomes, norms, predecessor=geometry_selection
    )
    assert {value.standardized_alpha for value in alpha_selection.selected.values()} == {
        0.5
    }
    assert all(
        not candidate.promotion_eligible
        for candidate in alpha_selection.candidates
        if candidate.standardized_alpha == 0
    )

    scope = e3_scope_conditions(alpha_selection.selected, protocol=protocol)
    outcomes, norms = _matrix(scope)
    scope_selection = _derive(
        "scope", scope, outcomes, norms, predecessor=alpha_selection
    )
    assert {value.token_scope for value in scope_selection.selected.values()} == {
        TokenScope.FINAL_PROMPT
    }
    assert scope_selection.scientific_eligible is False


def test_e3_selection_freezes_falsification_when_coverage_is_ineligible() -> None:
    conditions = e3_geometry_conditions(_protocol())
    outcomes, norms = _matrix(conditions, intervention_outcome=Outcome.ABSTENTION)

    selection = _derive("geometry", conditions, outcomes, norms)

    assert selection.falsified is True
    assert not selection.selected
    assert selection.falsification_reason is not None


def test_e3_selection_rejects_unscorable_or_incomplete_evidence() -> None:
    conditions = e3_geometry_conditions(_protocol())
    outcomes, norms = _matrix(conditions)
    outcomes[(conditions[0].condition_id, "q-1")] = Outcome.UNSCORABLE
    with pytest.raises(DataValidationError, match="unscorable"):
        _derive("geometry", conditions, outcomes, norms)
    outcomes[(conditions[0].condition_id, "q-1")] = Outcome.INCORRECT
    norms.pop(next(iter(norms)))
    with pytest.raises(DataValidationError, match="delta norms"):
        _derive("geometry", conditions, outcomes, norms)


def test_e3_selection_artifact_is_exactly_replayed(tmp_path: Path) -> None:
    conditions = e3_geometry_conditions(_protocol())
    outcomes, norms = _matrix(conditions)
    inputs = {
        "stage": "geometry",
        "conditions": conditions,
        "question_ids": ("q-1", "q-2"),
        "outcomes": outcomes,
        "actual_delta_norms": norms,
        "source_plan_identity": "a" * 64,
        "evaluation_plan_identity": "b" * 64,
        "evaluation_record_chain_head": "c" * 64,
        "evaluation_record_set_digest": "d" * 64,
        "source_scientific_eligible": False,
        "protocol": _protocol(),
    }
    path = tmp_path / "selection.json"
    written = write_e3_stage_selection(path, **inputs)

    assert verify_e3_stage_selection(path, **inputs) == written
    value = json.loads(path.read_text(encoding="utf-8"))
    value["baseline_coverage"] = 0.0
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(FrozenArtifactError, match="differs"):
        verify_e3_stage_selection(path, **inputs)


def test_e3_selection_requires_exact_grid_and_predecessor_chain() -> None:
    protocol = _protocol()
    geometry = e3_geometry_conditions(protocol)
    subset = (geometry[0], geometry[1], geometry[3])
    outcomes, norms = _matrix(subset)
    with pytest.raises(DataValidationError, match="frozen stage grid"):
        _derive("geometry", subset, outcomes, norms)

    points = {
        name: E3OperatingPoint(
            name, 1, ActivationSite.POST_MLP, 0.5, TokenScope.FIRST_FOUR
        )
        for name in ("M1-R", "M1-P")
    }
    alpha = e3_alpha_conditions(points, protocol=protocol)
    outcomes, norms = _matrix(alpha)
    with pytest.raises(DataValidationError, match="successful geometry"):
        _derive("alpha", alpha, outcomes, norms)


def test_e3_selection_rejects_impossible_delta_norms_and_duplicate_json(
    tmp_path: Path,
) -> None:
    conditions = e3_geometry_conditions(_protocol())
    outcomes, norms = _matrix(conditions)
    key = next(iter(norms))
    norms[key] = 0.0
    with pytest.raises(DataValidationError, match="contradicts"):
        _derive("geometry", conditions, outcomes, norms)

    outcomes, norms = _matrix(conditions)
    inputs = {
        "stage": "geometry",
        "conditions": conditions,
        "question_ids": ("q-1", "q-2"),
        "outcomes": outcomes,
        "actual_delta_norms": norms,
        "source_plan_identity": "a" * 64,
        "evaluation_plan_identity": "b" * 64,
        "evaluation_record_chain_head": "c" * 64,
        "evaluation_record_set_digest": "d" * 64,
        "source_scientific_eligible": False,
        "protocol": _protocol(),
    }
    path = tmp_path / "selection.json"
    selection = write_e3_stage_selection(path, **inputs)
    valid = json.dumps(selection.to_dict(), sort_keys=True)
    path.write_text('{"stage":"malicious",' + valid[1:], encoding="utf-8")
    with pytest.raises(FrozenArtifactError, match="duplicate JSON key"):
        verify_e3_stage_selection(path, **inputs)


def test_e3_operating_point_source_types_remain_strict() -> None:
    with pytest.raises(DataValidationError):
        E3OperatingPoint(
            "M1-R", 1, ActivationSite.POST_MLP, 0.5, "final_prompt"  # type: ignore[arg-type]
        )

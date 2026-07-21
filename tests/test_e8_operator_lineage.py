from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mfh.errors import FrozenArtifactError
from mfh.experiments import e8_operator
from mfh.provenance import stable_hash


def test_intermediate_activation_bundle_must_match_live_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runbook = SimpleNamespace(
        outputs={"protected_behavior_activations": SimpleNamespace()},
        e7_finalization=tmp_path,
    )
    context = SimpleNamespace(
        runtime_artifact_sha256="a" * 64,
        execution_public_key="b" * 64,
    )
    copied = SimpleNamespace(
        runtime_artifact_sha256="c" * 64,
        execution_public_key="b" * 64,
        source_question_bundle_sha256="d" * 64,
        feature_schema=SimpleNamespace(),
        evidence=(),
    )
    monkeypatch.setattr(
        e8_operator,
        "load_e8_behavior_activation_bundle",
        lambda _path: copied,
    )
    monkeypatch.setattr(e8_operator, "sha256_path", lambda _path: "d" * 64)
    monkeypatch.setattr(
        e8_operator, "_feature_schema", lambda _runbook, _context: copied.feature_schema
    )
    monkeypatch.setattr(
        e8_operator,
        "_e7_label_material",
        lambda _runbook, _schema: ({}, {}, tmp_path),
    )
    stages = {
        "activations": True,
        "variant_screens": False,
        "protected_artifact": False,
        "candidate_screen": False,
    }

    with pytest.raises(FrozenArtifactError, match="another runbook"):
        e8_operator._verify_e8_intermediate_stages(runbook, context, stages)  # type: ignore[arg-type]


def test_candidate_screen_cannot_verify_without_protected_sources() -> None:
    runbook = SimpleNamespace(outputs={})
    context = SimpleNamespace()
    stages = {
        "activations": False,
        "variant_screens": False,
        "protected_artifact": False,
        "candidate_screen": True,
    }

    with pytest.raises(FrozenArtifactError, match="lacks its protected source"):
        e8_operator._verify_e8_intermediate_stages(runbook, context, stages)  # type: ignore[arg-type]


def _candidate_fixture() -> tuple[SimpleNamespace, SimpleNamespace]:
    points = []
    for prompt in ("P0-neutral", "P2-calibrated-abstention"):
        for method in ("M1", "M3", "M4", "M5"):
            for alpha in (0.1, 0.25, 0.5, 1.0, 2.0):
                condition_id = stable_hash([prompt, method, alpha])
                points.append(
                    SimpleNamespace(
                        prompt_id=prompt,
                        method=method,
                        alpha=alpha,
                        hallucination_risk=alpha,
                        coverage=alpha,
                        candidate_condition_id=condition_id,
                        selected_condition_id=(condition_id if alpha == 0.5 else None),
                    )
                )
    runbook = SimpleNamespace(
        matching_dimension="hallucination_risk",
        matching_tolerance=0.0,
        m5_alpha=0.5,
    )
    candidate = SimpleNamespace(
        matching_dimension="hallucination_risk",
        tolerance=0.0,
        target=0.5,
        points=tuple(points),
    )
    return runbook, candidate


def test_candidate_matching_policy_and_selected_ids_are_replayed() -> None:
    runbook, candidate = _candidate_fixture()
    mapping = e8_operator._candidate_selection_mapping(  # type: ignore[arg-type]
        runbook, candidate
    )
    assert set(mapping) == {"P0-neutral", "P2-calibrated-abstention"}
    assert all(set(values) == {"M1", "M3", "M4", "M5"} for values in mapping.values())

    candidate.target = 0.4
    with pytest.raises(FrozenArtifactError, match="matching policy"):
        e8_operator._candidate_selection_mapping(runbook, candidate)  # type: ignore[arg-type]


def test_candidate_wrong_selected_condition_is_rejected() -> None:
    runbook, candidate = _candidate_fixture()
    candidate.points[0].selected_condition_id = candidate.points[0].candidate_condition_id
    with pytest.raises(FrozenArtifactError, match="matching policy"):
        e8_operator._candidate_selection_mapping(runbook, candidate)  # type: ignore[arg-type]

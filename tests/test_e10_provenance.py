from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest

from mfh.contracts import (
    ActivationSite,
    AdaptivePolicySpec,
    InterventionSpec,
    Question,
    TokenScope,
)
from mfh.data.io import write_questions
from mfh.errors import FrozenArtifactError
from mfh.experiments.e10_composite import (
    _validate_e10_triviaqa_source,
    _validate_exact_e6_runtime_binding,
)
from mfh.experiments.runner import PhaseRunLedger


def _runtime(key: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "execution_public_key": key,
        "runtime_identity_digest": "a" * 64,
    }


def _intervention(key: str) -> InterventionSpec:
    return InterventionSpec(
        method="M6",
        artifact_sha256="c" * 64,
        adaptive_policy=AdaptivePolicySpec(
            release_risk_threshold=0.1,
            abstention_probability_threshold=0.8,
            alpha_max=1.0,
            alpha_beta=5.0,
            layer=1,
            site=ActivationSite.BLOCK_OUTPUT,
            token_scope=TokenScope.FIRST_FOUR,
            direction_sha256="d" * 64,
            direction_norm=1.0,
            execution_public_key=key,
        ),
    )


def test_e10_exact_e6_runtime_rejects_alternate_grader_or_key(tmp_path: Path) -> None:
    exact_key = "1" * 64
    exact_runtime = tmp_path / "e6-runtime.json"
    exact_runtime.write_text("exact runtime\n", encoding="utf-8")
    grader = tmp_path / "grader"
    grader.mkdir()
    packaged_runtime = grader / "runtime-attestation.json"
    packaged_runtime.write_bytes(exact_runtime.read_bytes())
    bundle = SimpleNamespace(
        directory=grader,
        runtime_attestation=_runtime(exact_key),
        scorer=SimpleNamespace(execution_public_key=exact_key),
    )
    with (
        patch(
            "mfh.experiments.e10_composite._load_e6_runtime_attestation",
            return_value=_runtime(exact_key),
        ),
        patch(
            "mfh.experiments.confirmatory_graders.validate_confirmatory_grader_bundle",
            return_value=bundle,
        ),
    ):
        assert (
            _validate_exact_e6_runtime_binding(
                runtime_artifact=exact_runtime,
                grader=grader,
                intervention=_intervention(exact_key),
            )
            == exact_key
        )
        with pytest.raises(FrozenArtifactError, match="exact E6"):
            _validate_exact_e6_runtime_binding(
                runtime_artifact=exact_runtime,
                grader=grader,
                intervention=_intervention("2" * 64),
            )
        packaged_runtime.write_text("alternate runtime\n", encoding="utf-8")
        with pytest.raises(FrozenArtifactError, match="exact E6"):
            _validate_exact_e6_runtime_binding(
                runtime_artifact=exact_runtime,
                grader=grader,
                intervention=_intervention(exact_key),
            )


def test_e10_triviaqa_binding_rejects_alias_and_entity_tamper(tmp_path: Path) -> None:
    trusted = (
        Question(
            question_id="t-1",
            benchmark="triviaqa",
            text="Who wrote the work?",
            aliases=("Trusted Alias",),
            split="T-test",
            entities=("Trusted Entity",),
        ),
        Question(
            question_id="t-2",
            benchmark="triviaqa",
            text="Where was it written?",
            aliases=("Trusted Place",),
            split="T-test",
            entities=("Trusted Entity 2",),
        ),
    )
    bundle = tmp_path / "questions"
    bundle.mkdir()
    write_questions(bundle / "triviaqa.jsonl", trusted)
    e1 = cast(PhaseRunLedger, SimpleNamespace())
    with (
        patch.dict(
            "mfh.experiments.e10_composite._COUNTS",
            {"triviaqa": len(trusted)},
        ),
        patch(
            "mfh.experiments.e10_composite._reviewed_questions_from_e1",
            return_value={"T-test": trusted},
        ),
    ):
        _validate_e10_triviaqa_source(
            e1=e1,
            supplied=trusted,
            frozen_question_bundle=bundle,
        )
        for tampered in (
            replace(trusted[0], aliases=("Altered Alias",)),
            replace(trusted[0], entities=("Altered Entity",)),
        ):
            with pytest.raises(FrozenArtifactError, match="exact E1 reviewed"):
                _validate_e10_triviaqa_source(
                    e1=e1,
                    supplied=(tampered, trusted[1]),
                    frozen_question_bundle=bundle,
                )

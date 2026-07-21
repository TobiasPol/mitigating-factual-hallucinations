from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from mfh.contracts import ActivationSite, Outcome, PromptSpec, Question
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.experiments.rq1_capture import (
    _capture_row,
    _source_rows,
    run_rq1_capture,
    verify_rq1_capture,
)
from mfh.inference.architecture import HookKey
from mfh.methods.features import ActivationFeatureSchema


class _Record(SimpleNamespace):
    def to_dict(self) -> dict[str, Any]:
        return dict(self.serialized)


def _record(
    sequence: int,
    question_id: str,
    outcome: Outcome,
    prompt_id: str = "P0-neutral",
) -> _Record:
    raw = "answer" if outcome is not Outcome.ABSTENTION else "I don't know"
    return _Record(
        sequence=sequence,
        question_id=question_id,
        prompt_id=prompt_id,
        outcome=outcome,
        rendered_prompt_sha256="a" * 64,
        prompt_token_ids_sha256="b" * 64,
        evidence={"raw_output": raw},
        serialized={"sequence": sequence, "question_id": question_id, "outcome": outcome.value},
    )


def test_source_rows_retain_exact_p0_abstentions() -> None:
    snapshot = SimpleNamespace(
        schedule=(
            SimpleNamespace(sequence=0, question_id="q-c", prompt_id="P0-neutral"),
            SimpleNamespace(sequence=1, question_id="q-a", prompt_id="P0-neutral"),
            SimpleNamespace(sequence=2, question_id="q-c", prompt_id="P2-calibrated-abstention"),
        ),
        generations=(
            _record(0, "q-c", Outcome.CORRECT),
            _record(1, "q-a", Outcome.ABSTENTION),
            _record(2, "q-c", Outcome.CORRECT, "P2-calibrated-abstention"),
        ),
    )

    rows = _source_rows(snapshot, assignment_groups={"q-c": "g1", "q-a": "g2"})

    assert [row["question_id"] for row in rows] == ["q-c", "q-a"]
    assert [row["outcome"] for row in rows] == ["C", "A"]


class _Runtime:
    def __init__(self, *, response_width: int) -> None:
        self.response_width = response_width

    def render_prompt(self, *_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(sha256="a" * 64, token_ids_sha256="b" * 64)

    def prompt_feature_cube(self, *_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(
            activations={ActivationSite.POST_MLP: {0: np.asarray([[1.0, 2.0]])}},
            peak_memory_bytes=10,
        )

    def teacher_forced_cube(self, *_args: Any, **_kwargs: Any) -> Any:
        raw = "answer"
        return SimpleNamespace(
            response_text_sha256=hashlib.sha256(raw.encode()).hexdigest(),
            response_token_ids=(1, 2),
            activations={
                ActivationSite.POST_MLP: {
                    0: np.ones((2, self.response_width), dtype=np.float32)
                }
            },
            peak_memory_bytes=20,
        )


def test_capture_row_rejects_response_width_drift() -> None:
    question = Question(
        "q",
        "triviaqa",
        "Question?",
        ("answer",),
        split="T-steer",
        entities=("entity",),
    )
    prompt = PromptSpec("P0-neutral", "Answer.")
    record = _record(0, "q", Outcome.CORRECT)
    expected = {
        "rendered_prompt_sha256": "a" * 64,
        "prompt_token_ids_sha256": "b" * 64,
        "raw_output_sha256": hashlib.sha256(b"answer").hexdigest(),
    }
    schema = ActivationFeatureSchema.synthetic(partition="T-steer", width=2)

    with pytest.raises(DataValidationError, match="response activation geometry"):
        _capture_row(
            _Runtime(response_width=3),
            question=question,
            prompt=prompt,
            record=record,
            expected=expected,
            feature_schema=schema,
            hooks=(HookKey(0, ActivationSite.POST_MLP),),
            hidden_width=2,
        )


def test_verifier_rejects_symlinked_capture_root(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "capture"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(FrozenArtifactError, match="root cannot be a symlink"):
        verify_rq1_capture(
            link,
            plan=None,  # type: ignore[arg-type]
            snapshot=None,  # type: ignore[arg-type]
            questions=(),
            prompt=None,  # type: ignore[arg-type]
            expected_execution_public_key="0" * 64,
        )

    with pytest.raises(FrozenArtifactError, match="root cannot be a symlink"):
        run_rq1_capture(
            link,
            plan=None,  # type: ignore[arg-type]
            snapshot=None,  # type: ignore[arg-type]
            questions=(),
            prompt=None,  # type: ignore[arg-type]
            runtime=None,  # type: ignore[arg-type]
            private_key_hex="0" * 64,
        )

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from urllib.request import Request

import pytest

from mfh.contracts import GenerationRecord, Outcome, Question, Runtime
from mfh.errors import FrozenArtifactError
from mfh.evaluation.openrouter import OpenRouterTransport
from mfh.experiments import e6_grading as e6_grading_module
from mfh.experiments.e6_grading import (
    E6FactualGrader,
    load_e6_official_grader_bundle,
    load_env_secret,
    verify_e6_factual_grade,
)

ROOT = Path(__file__).parents[1]
MANIFEST_DIGEST = "b" * 64


def _record(benchmark: str, question_id: str) -> GenerationRecord:
    return GenerationRecord(
        question_id=question_id,
        benchmark=benchmark,
        model_repository="nvidia/Qwen3.6-27B-NVFP4",
        model_revision="0" * 40,
        runtime=Runtime.VLLM,
        quantization="4bit",
        system_prompt_id="P0-neutral",
        rendered_prompt_hash="1" * 64,
        steering_method="M0",
        layer=None,
        token_scope=None,
        alpha=0.0,
        sparsity=None,
        controller_scores={},
        raw_output="candidate answer",
        normalized_answer="candidate answer",
        outcome=Outcome.INCORRECT,
        generation_latency_seconds=1.0,
        input_tokens=2,
        output_tokens=2,
        condition_id="2" * 64,
        seed=17,
        metadata={},
    )


@pytest.mark.parametrize(
    ("benchmark", "provider", "label", "outcome"),
    (
        ("simpleqa_verified", "OpenAI", "A", Outcome.CORRECT),
        ("aa_omniscience_public_600", "Google AI Studio", "C", Outcome.PARTIAL),
    ),
)
def test_e6_official_grades_replay_from_portable_bundle(
    tmp_path: Path,
    benchmark: str,
    provider: str,
    label: str,
    outcome: Outcome,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path = tmp_path / "official-graders"
    bundle_path.mkdir()
    for name in (
        "simpleqa-verified.yaml",
        "simpleqa-verified.prompt.txt",
        "aa-omniscience-public.yaml",
        "aa-omniscience-public.prompt.txt",
    ):
        shutil.copy2(ROOT / "configs/graders" / name, bundle_path / name)
    monkeypatch.setattr(
        e6_grading_module,
        "verify_e1_grader_bundle",
        lambda *_args, **_kwargs: {
            "manifest_digest": MANIFEST_DIGEST,
            "files": {
                "simpleqa_config": {"path": "simpleqa-verified.yaml"},
                "aa_config": {"path": "aa-omniscience-public.yaml"},
            },
        },
    )
    bundle = load_e6_official_grader_bundle(
        bundle_path, expected_manifest_digest=MANIFEST_DIGEST
    )

    def sender(request: Request, _timeout: float) -> tuple[int, bytes]:
        payload = json.loads(request.data or b"{}")
        return 200, json.dumps(
            {
                "id": "grade-e6-test",
                "model": payload["model"],
                "provider": provider,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": label},
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "total_tokens": 11,
                },
            }
        ).encode()

    env = tmp_path / ".env"
    env.write_text("OPENROUTER_API_KEY=test-key\n", encoding="utf-8")
    grader = E6FactualGrader(
        bundle,
        environment_file=env,
        transport=OpenRouterTransport(api_key="test-key", sender=sender),
    )
    question = Question("q1", benchmark, "Question?", ("reference answer",))
    graded = grader(_record(benchmark, question.question_id), question)

    assert graded.outcome is outcome
    verify_e6_factual_grade(graded, question, grader_bundle=bundle)
    with pytest.raises(FrozenArtifactError, match="does not replay"):
        verify_e6_factual_grade(
            replace(graded, outcome=Outcome.ABSTENTION),
            question,
            grader_bundle=bundle,
        )


def test_e6_env_secret_loads_exact_local_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    path = tmp_path / ".env"
    path.write_text("# local only\nOPENROUTER_API_KEY='stored-key'\n", encoding="utf-8")
    assert load_env_secret(path, "OPENROUTER_API_KEY") == "stored-key"

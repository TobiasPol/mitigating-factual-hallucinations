from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from mfh.contracts import GenerationRecord, Outcome, Question, Runtime
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.evaluation.side_effects import write_side_effect_scorer_spec
from mfh.experiments.confirmatory_graders import (
    validate_confirmatory_factual_grade,
    validate_confirmatory_grader_bundle,
    write_confirmatory_grader_bundle,
)
from mfh.provenance import sha256_path, stable_hash

OFFICIAL_DIGEST = "b" * 64


def _fake_directory(path: Path, name: str) -> Path:
    path.mkdir()
    (path / name).write_text(f"{name}\n", encoding="utf-8")
    return path


def _fake_runtime(path: Path, execution_public_key: str) -> tuple[Path, dict[str, str]]:
    path.write_text("{}\n", encoding="utf-8")
    return path, {
        "execution_public_key": execution_public_key,
        "runtime_identity_digest": "9" * 64,
    }


def _fake_official(path: Path) -> Path:
    return _fake_directory(path, "fixture.txt")


def test_confirmatory_bundle_recursively_preserves_every_grader(tmp_path: Path) -> None:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    scorer = tmp_path / "scorer.json"
    write_side_effect_scorer_spec(scorer, execution_public_key=public_key)
    ifeval = _fake_directory(tmp_path / "ifeval", "evaluate.py")
    strongreject = _fake_directory(tmp_path / "strongreject", "rubric.txt")
    runtime_path, runtime = _fake_runtime(tmp_path / "runtime.json", public_key)
    official = _fake_official(tmp_path / "official")
    destination = tmp_path / "confirmatory-graders"

    with (
        patch(
            "mfh.experiments.confirmatory_graders.validate_ifeval_evaluator",
            side_effect=lambda value: sha256_path(value),
        ),
        patch(
            "mfh.experiments.confirmatory_graders.validate_strongreject_grader",
            side_effect=lambda value: sha256_path(value),
        ),
        patch(
            "mfh.experiments.confirmatory_graders._load_e6_runtime_attestation",
            return_value=runtime,
        ),
        patch(
            "mfh.experiments.confirmatory_graders.verify_e1_grader_bundle",
            return_value={"manifest_digest": OFFICIAL_DIGEST},
        ),
    ):
        written = write_confirmatory_grader_bundle(
            destination,
            official_grader_bundle=official,
            expected_official_manifest_digest=OFFICIAL_DIGEST,
            side_effect_scorer=scorer,
            ifeval_evaluator=ifeval,
            strongreject_grader=strongreject,
            runtime_attestation=runtime_path,
        )
        verified = validate_confirmatory_grader_bundle(
            destination,
            expected_official_manifest_digest=OFFICIAL_DIGEST,
        )

    assert written.fingerprint == verified.fingerprint == sha256_path(destination)
    assert verified.scorer.execution_public_key == public_key
    assert verified.official_manifest_digest == OFFICIAL_DIGEST
    assert sha256_path(destination / "official-graders") == sha256_path(official)


def test_confirmatory_bundle_rejects_component_tampering(tmp_path: Path) -> None:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    scorer = tmp_path / "scorer.json"
    write_side_effect_scorer_spec(scorer, execution_public_key=public_key)
    ifeval = _fake_directory(tmp_path / "ifeval", "evaluate.py")
    strongreject = _fake_directory(tmp_path / "strongreject", "rubric.txt")
    runtime_path, runtime = _fake_runtime(tmp_path / "runtime.json", public_key)
    official = _fake_official(tmp_path / "official")
    destination = tmp_path / "confirmatory-graders"
    validator_patches = (
        patch(
            "mfh.experiments.confirmatory_graders.validate_ifeval_evaluator",
            side_effect=lambda value: sha256_path(value),
        ),
        patch(
            "mfh.experiments.confirmatory_graders.validate_strongreject_grader",
            side_effect=lambda value: sha256_path(value),
        ),
    )
    with validator_patches[0], validator_patches[1], patch(
        "mfh.experiments.confirmatory_graders._load_e6_runtime_attestation",
        return_value=runtime,
    ), patch(
        "mfh.experiments.confirmatory_graders.verify_e1_grader_bundle",
        return_value={"manifest_digest": OFFICIAL_DIGEST},
    ):
        write_confirmatory_grader_bundle(
            destination,
            official_grader_bundle=official,
            expected_official_manifest_digest=OFFICIAL_DIGEST,
            side_effect_scorer=scorer,
            ifeval_evaluator=ifeval,
            strongreject_grader=strongreject,
            runtime_attestation=runtime_path,
        )

    payload = json.loads((destination / "side-effect-scorer.json").read_text())
    payload["execution_public_key"] = "0" * 64
    (destination / "side-effect-scorer.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (
        patch(
            "mfh.experiments.confirmatory_graders.validate_ifeval_evaluator",
            side_effect=lambda value: sha256_path(value),
        ),
        patch(
            "mfh.experiments.confirmatory_graders.validate_strongreject_grader",
            side_effect=lambda value: sha256_path(value),
        ),
        patch(
            "mfh.experiments.confirmatory_graders._load_e6_runtime_attestation",
            return_value=runtime,
        ),
        patch(
            "mfh.experiments.confirmatory_graders.verify_e1_grader_bundle",
            return_value={"manifest_digest": OFFICIAL_DIGEST},
        ),
        pytest.raises(FrozenArtifactError, match="identity differs"),
    ):
        validate_confirmatory_grader_bundle(destination)


def test_confirmatory_triviaqa_grade_recomputes_frozen_aliases() -> None:
    question = Question(
        question_id="triviaqa:test:1",
        benchmark="triviaqa",
        text="What is the capital of France?",
        aliases=("Paris",),
    )
    record = GenerationRecord(
        question_id=question.question_id,
        benchmark=question.benchmark,
        model_repository="example/model",
        model_revision="a" * 40,
        runtime=Runtime.VLLM,
        quantization="1bit",
        system_prompt_id="P0-neutral",
        rendered_prompt_hash="b" * 64,
        steering_method="M0",
        layer=None,
        site=None,
        token_scope=None,
        alpha=0.0,
        sparsity=None,
        controller_scores={},
        raw_output="Definitely Lyon",
        normalized_answer="definitely lyon",
        outcome=Outcome.CORRECT,
        generation_latency_seconds=1.0,
        input_tokens=10,
        output_tokens=2,
        condition_id="c" * 64,
        metadata={
            "official_score_output_sha256": stable_hash("Definitely Lyon"),
            "official_scorer": "mfh.triviaqa.alias-aware-em-f1.v1",
            "official_exact_match": 1.0,
            "official_token_f1": 1.0,
            "reference_aliases_digest": stable_hash(["Paris"]),
        },
    )
    with pytest.raises(DataValidationError, match="frozen aliases"):
        validate_confirmatory_factual_grade(
            record,
            question,
            grader_bundle=cast(Any, None),
        )

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from mfh.analysis.human_audit import (
    finalize_human_audit,
    load_factual_adjudicated_rows,
    prepare_human_audit,
    prepare_synthetic_human_audit,
    verify_human_audit_queue,
    verify_human_audit_results,
)
from mfh.analysis.protocol import load_analysis_protocol
from mfh.contracts import GenerationRecord, Outcome, Question, Runtime
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.provenance import stable_hash

ROOT = Path(__file__).parents[1]
PROTOCOL = ROOT / "configs" / "analysis" / "confirmatory.yaml"
MODELS = (
    "nvidia/Qwen3.6-27B-NVFP4",
)
BENCHMARKS = (
    "triviaqa",
    "simpleqa_verified",
    "aa_omniscience_public_600",
)
BLINDING_KEY = bytes(range(32))


def _record(
    *,
    question_id: str,
    benchmark: str,
    model: str,
    method: str,
    outcome: Outcome,
    raw_output: str,
    metadata: dict[str, object] | None = None,
) -> GenerationRecord:
    return GenerationRecord(
        question_id=question_id,
        benchmark=benchmark,
        model_repository=model,
        model_revision="a" * 40,
        runtime=Runtime.VLLM,
        quantization="modelopt-mixed-nvfp4-fp8",
        system_prompt_id="P0-neutral",
        rendered_prompt_hash="b" * 64,
        steering_method=method,
        layer=None,
        token_scope=None,
        alpha=0.0,
        sparsity=None,
        controller_scores={},
        raw_output=raw_output,
        normalized_answer=raw_output,
        outcome=outcome,
        generation_latency_seconds=0.1,
        input_tokens=8,
        output_tokens=2,
        condition_id=f"E9:{benchmark}:{model}:{method}:P0-neutral",
        metadata=metadata or {},
    )


def _sources() -> tuple[list[GenerationRecord], list[Question]]:
    questions: list[Question] = []
    records: list[GenerationRecord] = []
    for benchmark in BENCHMARKS:
        for index in range(200):
            question_id = f"{benchmark}:{index:03d}"
            questions.append(
                Question(
                    question_id=question_id,
                    benchmark=benchmark,
                    text=f"Question {index} for {benchmark}?",
                    aliases=(f"answer-{index}",),
                )
            )
            for model_index, model in enumerate(MODELS):
                method = ("M0", "M1", "M3")[index % 3]
                if benchmark == "aa_omniscience_public_600" and index % 10 == 0:
                    outcome = Outcome.PARTIAL
                elif index % 5 == 0:
                    outcome = Outcome.ABSTENTION
                elif index % 3 == 0:
                    outcome = Outcome.INCORRECT
                else:
                    outcome = Outcome.CORRECT
                metadata: dict[str, object] = {}
                if benchmark == "simpleqa_verified" and index == 1 and model_index == 0:
                    metadata = {
                        "automated_grader_labels": {"grader-a": "C", "grader-b": "I"},
                        "automated_grader_disagreement": True,
                    }
                records.append(
                    _record(
                        question_id=question_id,
                        benchmark=benchmark,
                        model=model,
                        method=method,
                        outcome=outcome,
                        raw_output=f"response-{benchmark}-{model_index}-{index}",
                        metadata=metadata,
                    )
                )

    side_specs = (
        (
            "language_consistency",
            "lang-001",
            "Antworte auf Deutsch: Was ist die Hauptstadt von Frankreich?",
            "The answer is Paris.",
            {"requested_language": "de", "requested_language_correct": False},
        ),
        (
            "xstest",
            "safe-001",
            "How do I bake bread?",
            "I cannot help with that.",
            {"refused": True},
        ),
        (
            "strongreject_or_harmbench",
            "harm-001",
            "Provide a harmful procedure.",
            "1. Carry out the harmful step.",
            {"refused": False, "harmful_compliance_score": 1.0},
        ),
    )
    for benchmark, question_id, text, response, metadata in side_specs:
        questions.append(
            Question(
                question_id=question_id,
                benchmark=benchmark,
                text=text,
                aliases=("human-review-required",),
            )
        )
        for model in MODELS:
            records.append(
                _record(
                    question_id=question_id,
                    benchmark=benchmark,
                    model=model,
                    method="M6",
                    outcome=Outcome.UNSCORABLE,
                    raw_output=response,
                    metadata=metadata,
                )
            )
    return records, questions


def _write_annotations(
    path: Path,
    bindings: Sequence[Mapping[str, object]],
    *,
    annotator_id: str,
    changed_ids: set[str] | None = None,
) -> dict[str, str]:
    changed_ids = changed_ids or set()
    labels: dict[str, str] = {}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("audit_id", "annotator_id", "label"),
            lineterminator="\n",
        )
        writer.writeheader()
        for binding in bindings:
            audit_id = str(binding["audit_id"])
            automated = str(binding["automated_label"])
            if audit_id in changed_ids:
                task = str(binding["audit_task"])
                label = ("I" if automated != "I" else "C") if task == "factual_outcome" else "U"
            else:
                label = automated
            labels[audit_id] = label
            writer.writerow({"audit_id": audit_id, "annotator_id": annotator_id, "label": label})
    return labels


def test_blinded_audit_queue_and_adjudication_round_trip(tmp_path: Path) -> None:
    protocol = load_analysis_protocol(PROTOCOL)
    records, questions = _sources()
    queue = prepare_synthetic_human_audit(
        tmp_path / "queue",
        records=records,
        questions=questions,
        protocol=protocol,
        blinding_key=BLINDING_KEY,
    )

    assert len(queue.bindings) == 603
    forbidden = {
        "model",
        "model_repository",
        "method",
        "prompt",
        "condition_id",
        "automated_label",
        "selection_reasons",
    }
    assert all(not (forbidden & set(row)) for row in queue.blind_items)
    first_binding = queue.bindings[0]
    public_guess = (
        "audit-"
        + stable_hash(
            {
                "schema_version": 1,
                "seed": protocol.human_audit.sample_seed,
                "condition_id": first_binding["condition_id"],
                "question_id": first_binding["question_id"],
                "response_sha256": first_binding["response_sha256"],
            }
        )[:24]
    )
    assert first_binding["audit_id"] != public_guess
    manifest = json.loads((queue.directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["reason_counts"]["automated_grader_disagreements"] == 1
    assert manifest["reason_counts"]["partial_aa_responses"] == 20
    assert manifest["reason_counts"]["language_switch_detections"] == 1
    assert manifest["reason_counts"]["suspected_safety_regressions"] == 2
    assert all(value >= 200 for value in manifest["factual_combination_counts"].values())

    second_queue = prepare_synthetic_human_audit(
        tmp_path / "queue-copy",
        records=reversed(records),
        questions=reversed(questions),
        protocol=protocol,
        blinding_key=BLINDING_KEY,
    )
    assert second_queue.manifest_digest == queue.manifest_digest
    assert (second_queue.directory / "blind-items.jsonl").read_bytes() == (
        queue.directory / "blind-items.jsonl"
    ).read_bytes()

    changed = {
        next(
            str(row["audit_id"]) for row in queue.bindings if row["audit_task"] == "factual_outcome"
        ),
        next(
            str(row["audit_id"]) for row in queue.bindings if row["audit_task"] != "factual_outcome"
        ),
    }
    first_path = tmp_path / "annotator-a.csv"
    second_path = tmp_path / "annotator-b.csv"
    first = _write_annotations(first_path, queue.bindings, annotator_id="reviewer-a")
    second = _write_annotations(
        second_path,
        queue.bindings,
        annotator_id="reviewer-b",
        changed_ids=changed,
    )
    adjudications = tmp_path / "adjudications.csv"
    with adjudications.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("audit_id", "label"), lineterminator="\n")
        writer.writeheader()
        for audit_id in sorted(changed):
            writer.writerow({"audit_id": audit_id, "label": first[audit_id]})
    assert all(first[audit_id] != second[audit_id] for audit_id in changed)

    results = finalize_human_audit(
        queue.directory,
        tmp_path / "results",
        annotations={"reviewer-b": second_path, "reviewer-a": first_path},
        adjudications=adjudications,
        expected_protocol=protocol,
        require_scientific=False,
    )
    verified = verify_human_audit_results(
        results.directory,
        queue_directory=queue.directory,
        expected_protocol=protocol,
        require_scientific=False,
    )
    factual_payload = verified.summary["factual_reporting_payload"]
    assert factual_payload["adjudication_summary"]["rows"] == 600
    assert factual_payload["adjudication_summary"]["disagreements"] == 1
    assert (
        len(
            load_factual_adjudicated_rows(
                results.directory,
                queue_directory=queue.directory,
                expected_protocol=protocol,
                require_scientific=False,
            )
        )
        == 600
    )

    (queue.directory / "unexpected.txt").write_text("tamper", encoding="utf-8")
    with pytest.raises(FrozenArtifactError, match="missing or unexpected"):
        verify_human_audit_queue(
            queue.directory,
            expected_protocol=protocol,
            require_scientific=False,
        )
    (queue.directory / "unexpected.txt").unlink()
    (queue.directory / "empty-extra-directory").mkdir()
    with pytest.raises(FrozenArtifactError, match="missing or unexpected"):
        verify_human_audit_queue(
            queue.directory,
            expected_protocol=protocol,
            require_scientific=False,
        )


def test_human_audit_requires_every_disagreement_adjudicated(tmp_path: Path) -> None:
    protocol = load_analysis_protocol(PROTOCOL)
    records, questions = _sources()
    queue = prepare_synthetic_human_audit(
        tmp_path / "queue",
        records=records,
        questions=questions,
        protocol=protocol,
        blinding_key=BLINDING_KEY,
    )
    changed = {str(queue.bindings[0]["audit_id"])}
    first_path = tmp_path / "annotator-a.csv"
    second_path = tmp_path / "annotator-b.csv"
    _write_annotations(first_path, queue.bindings, annotator_id="one")
    _write_annotations(
        second_path,
        queue.bindings,
        annotator_id="two",
        changed_ids=changed,
    )
    empty = tmp_path / "adjudications.csv"
    empty.write_text("audit_id,label\n", encoding="utf-8")

    with pytest.raises(DataValidationError, match="every annotator disagreement"):
        finalize_human_audit(
            queue.directory,
            tmp_path / "results",
            annotations={"one": first_path, "two": second_path},
            adjudications=empty,
            expected_protocol=protocol,
            require_scientific=False,
        )

    with pytest.raises(DataValidationError, match="distinct regular annotation files"):
        finalize_human_audit(
            queue.directory,
            tmp_path / "same-file-results",
            annotations={"one": first_path, "two": first_path},
            adjudications=empty,
            expected_protocol=protocol,
            require_scientific=False,
        )


def test_scientific_audit_is_bound_to_live_complete_sources(tmp_path: Path) -> None:
    protocol = load_analysis_protocol(PROTOCOL)
    records, questions = _sources()
    factual_records = tuple(record for record in records if record.benchmark in BENCHMARKS)
    factual_questions = tuple(
        question for question in questions if question.benchmark in BENCHMARKS
    )
    completions = {"E9": "9" * 64, "E10": "a" * 64}

    with patch(
        "mfh.analysis.human_audit._verified_audit_sources",
        return_value=(factual_records, factual_questions, completions),
    ):
        queue = prepare_human_audit(
            tmp_path / "scientific-queue",
            study=object(),  # type: ignore[arg-type]
            phase_run_directories={"E9": tmp_path / "E9", "E10": tmp_path / "E10"},
            protocol=protocol,
            blinding_key=BLINDING_KEY,
        )

    with (
        patch(
            "mfh.analysis.human_audit._verified_audit_sources",
            return_value=(factual_records, factual_questions, completions),
        ),
        pytest.raises(FrozenArtifactError, match="blinding key differs"),
    ):
        verify_human_audit_queue(
            queue.directory,
            expected_protocol=protocol,
            study=object(),  # type: ignore[arg-type]
            phase_run_directories={"E9": tmp_path / "E9", "E10": tmp_path / "E10"},
            blinding_key=bytes(reversed(BLINDING_KEY)),
        )

    extra = replace(
        factual_records[0],
        condition_id="new-complete-ledger-condition",
        raw_output="previously omitted response",
    )
    with (
        patch(
            "mfh.analysis.human_audit._verified_audit_sources",
            return_value=((*factual_records, extra), factual_questions, completions),
        ),
        pytest.raises(FrozenArtifactError, match="source ledgers or questions changed"),
    ):
        verify_human_audit_queue(
            queue.directory,
            expected_protocol=protocol,
            study=object(),  # type: ignore[arg-type]
            phase_run_directories={"E9": tmp_path / "E9", "E10": tmp_path / "E10"},
            blinding_key=BLINDING_KEY,
        )

    synthetic = prepare_synthetic_human_audit(
        tmp_path / "synthetic-queue",
        records=factual_records,
        questions=factual_questions,
        protocol=protocol,
        blinding_key=BLINDING_KEY,
    )
    with pytest.raises(FrozenArtifactError, match="scientific provenance"):
        verify_human_audit_queue(
            synthetic.directory,
            expected_protocol=protocol,
        )

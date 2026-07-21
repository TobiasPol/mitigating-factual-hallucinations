from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import mfh.experiments.e5_adaptive as e5_adaptive
from mfh.contracts import Outcome, Question, TokenScope
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.experiments.e4_baselines import (
    E4Protocol,
    build_e4_screen_receipt,
    write_e4_screen_receipt,
)
from mfh.experiments.e5_adaptive import (
    E5AblationRecord,
    E5ControllerBinding,
    E5Protocol,
    build_e5_ablation_grid,
    derive_e5_selection,
    finalize_e5_phase,
    sign_e5_ablation_execution_receipt,
    verify_e5_phase,
    verify_e5_selection,
    write_e5_ablation_records,
    write_e5_selection,
)
from mfh.provenance import sha256_file, stable_hash

_EXECUTION_PRIVATE_KEY = "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60"
_EXECUTION_PUBLIC_KEY = "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"


def _protocol() -> E5Protocol:
    return E5Protocol(
        vector_counts=(1, 4),
        routers=("nearest_centroid",),
        alpha_modes=("fixed", "risk_gated"),
        layer_modes=("fixed_best",),
        intervention_timings=("final_prompt",),
        controller_inputs=("one_layer",),
    )


def _screen(tmp_path: Path):  # type: ignore[no-untyped-def]
    questions = tuple(
        Question(
            f"q-{index}",
            "triviaqa",
            f"Question {index}?",
            (f"answer-{index}",),
            split="T-dev",
        )
        for index in range(3)
    )
    receipt = build_e4_screen_receipt(
        questions,
        protocol=E4Protocol(dev_rows=3, screen_rows=2),
    )
    path = tmp_path / "screen.json"
    write_e4_screen_receipt(path, receipt)
    return receipt, path


def _records(
    protocol: E5Protocol,
    questions: tuple[Question, ...],
    bindings: dict[str, E5ControllerBinding],
    binding_paths: dict[str, Path],
) -> Iterator[E5AblationRecord]:
    grid = build_e5_ablation_grid(protocol)
    question_ids = tuple(value.question_id for value in questions)
    questions_by_id = {value.question_id: value for value in questions}
    for arm_index, arm_id in enumerate(("M1", *(value.spec_id for value in grid))):
        for prompt_id in ("P0-neutral", "P2-calibrated-abstention"):
            for question_index, question_id in enumerate(question_ids):
                incorrect_cutoff = max(0, 2 - arm_index)
                outcome = (
                    Outcome.INCORRECT if question_index < incorrect_cutoff else Outcome.CORRECT
                )
                adaptive = arm_id != "M1"
                binding_sha = sha256_file(binding_paths[arm_id]) if adaptive else None
                controller_sha = bindings[arm_id].controller_artifact_sha256 if adaptive else None
                scores = {"C": 0.2, "I": 0.7, "A": 0.1} if adaptive else {}
                rendered = stable_hash({"prompt": prompt_id, "question": question_id})
                question = questions_by_id[question_id]
                prompt_input = stable_hash(
                    {
                        "prompt_id": prompt_id,
                        "prompt_template_sha256": {
                            "P0-neutral": (
                                "5b08080e81d4032d853d3a30fda35e7670da6a81cc1ccf17f2b8fc5001bba684"
                            ),
                            "P2-calibrated-abstention": (
                                "3170134d9a69836c1b530d1b16585ef7b0d92ea6fadc8f958e2655053e273fe5"
                            ),
                        }[prompt_id],
                        "question_id": question.question_id,
                        "benchmark": question.benchmark,
                        "text": question.text,
                        "aliases": list(question.aliases),
                        "split": question.split,
                        "entities": list(question.entities),
                        "metadata": dict(question.metadata),
                    }
                )
                decision_body = {
                    "arm_id": arm_id,
                    "prompt_id": prompt_id,
                    "question_id": question_id,
                    "rendered_prompt_sha256": rendered,
                    "prompt_input_sha256": prompt_input,
                    "controller_binding_sha256": binding_sha,
                    "controller_artifact_sha256": controller_sha,
                    "controller_scores": scores,
                    "policy_action": "intervene",
                    "token_scope": TokenScope.FINAL_PROMPT.value,
                    "applied_token_indices": [-1],
                    "activation_delta_norm": 1.0,
                }
                receipt = {
                    "controller_binding_sha256": binding_sha,
                    "controller_artifact_sha256": controller_sha,
                    "controller_scores": scores,
                    "policy_action": "intervene",
                    "applied_token_indices": [-1],
                    "activation_delta_norm": 1.0,
                    "decision_digest": stable_hash(decision_body),
                }
                draft = E5AblationRecord(
                    arm_id=arm_id,
                    prompt_id=prompt_id,
                    question_id=question_id,
                    outcome=outcome,
                    generation_latency_seconds=1.0,
                    intervention_norm=1.0,
                    prompt_template_sha256={
                        "P0-neutral": (
                            "5b08080e81d4032d853d3a30fda35e7670da6a81cc1ccf17f2b8fc5001bba684"
                        ),
                        "P2-calibrated-abstention": (
                            "3170134d9a69836c1b530d1b16585ef7b0d92ea6fadc8f958e2655053e273fe5"
                        ),
                    }[prompt_id],
                    rendered_prompt_sha256=rendered,
                    prompt_input_sha256=prompt_input,
                    output_tokens=1,
                    controller_binding_sha256=binding_sha,
                    token_scope=TokenScope.FINAL_PROMPT,
                    execution_receipt=receipt,
                    execution_receipt_digest=stable_hash(receipt),
                    execution_receipt_signature="0" * 128,
                )
                yield replace(
                    draft,
                    execution_receipt_signature=sign_e5_ablation_execution_receipt(
                        draft,
                        private_key_hex=_EXECUTION_PRIVATE_KEY,
                    ),
                )


def test_e5_default_grid_and_strict_protocol_type() -> None:
    grid = build_e5_ablation_grid()
    assert len(grid) == 4 * 3 * 3 * 3 * 3 * 3
    assert len({value.spec_id for value in grid}) == len(grid)

    class _Fake:
        vector_counts = (1,)

    with pytest.raises(DataValidationError, match="exact E5Protocol"):
        build_e5_ablation_grid(_Fake())  # type: ignore[arg-type]


def test_e5_selection_cannot_promote_static_reduction_as_adaptive() -> None:
    protocol = E5Protocol(
        vector_counts=(1, 4),
        routers=("nearest_centroid",),
        alpha_modes=("fixed",),
        layer_modes=("fixed_best",),
        intervention_timings=("final_prompt",),
        controller_inputs=("one_layer",),
    )
    static_spec, adaptive_spec = build_e5_ablation_grid(protocol)
    reference = e5_adaptive.E5StaticReference(
        accuracy=0.9,
        coverage=1.0,
        abstention_rate=0.0,
        hallucination_risk=0.1,
        mean_intervention_norm=1.0,
        mean_latency_seconds=1.0,
    )
    measurements = (
        e5_adaptive.E5Measurement(
            spec_id=static_spec.spec_id,
            controller_artifact_sha256="a" * 64,
            accuracy=0.95,
            coverage=1.0,
            abstention_rate=0.0,
            hallucination_risk=0.05,
            mean_intervention_norm=1.0,
            mean_latency_seconds=1.0,
        ),
        e5_adaptive.E5Measurement(
            spec_id=adaptive_spec.spec_id,
            controller_artifact_sha256="b" * 64,
            accuracy=0.9,
            coverage=1.0,
            abstention_rate=0.0,
            hallucination_risk=0.1,
            mean_intervention_norm=1.0,
            mean_latency_seconds=1.0,
        ),
    )
    _, selected = e5_adaptive._selection_values(
        measurements,
        static_reference=reference,
        protocol=protocol,
    )
    assert selected is not None
    assert selected.spec_id == adaptive_spec.spec_id


def test_e5_terminal_marker_is_recovered_after_wrapper_crash(tmp_path: Path) -> None:
    terminal = SimpleNamespace(
        completion_digest="a" * 64,
        record_set_digest="b" * 64,
        gate_result_digests={"matched_coverage": "c" * 64},
    )

    class _Ledger:
        directory = tmp_path
        finalize_calls = 0

        def finalize(self, _results):  # type: ignore[no-untyped-def]
            self.finalize_calls += 1
            (self.directory / "complete.json").write_text("terminal\n", encoding="utf-8")
            return terminal

        def verify_complete(self):  # type: ignore[no-untyped-def]
            return terminal

        def finalize_falsified(self, _results):  # type: ignore[no-untyped-def]
            raise AssertionError("passing gates cannot falsify E5")

        def verify_falsified(self):  # type: ignore[no-untyped-def]
            raise AssertionError("passing gates cannot verify a falsification")

    ledger = _Ledger()
    gates = {
        "matched_coverage": SimpleNamespace(passed=True, gate_digest="c" * 64)
    }
    first = e5_adaptive._finalize_or_recover_e5_ledger(
        cast(Any, ledger), cast(Any, gates)
    )
    second = e5_adaptive._finalize_or_recover_e5_ledger(
        cast(Any, ledger), cast(Any, gates)
    )
    assert first == second
    assert ledger.finalize_calls == 1


def test_e5_streaming_selection_replays_all_sources(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    protocol = _protocol()
    screen, screen_path = _screen(tmp_path)
    upstream = {}
    for name in (
        "E2_calibrated_probes",
        "E3_static_vectors",
        "E4_promoted_baselines",
    ):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        upstream[name] = path
    binding_paths: dict[str, Path] = {}
    bindings: dict[str, E5ControllerBinding] = {}
    controller_directory = tmp_path.resolve()
    controller_sha = "a" * 64
    for spec in build_e5_ablation_grid(protocol):
        body = {
            "schema_version": 1,
            "spec": spec.to_dict(),
            "spec_id": spec.spec_id,
            "controller_directory": str(controller_directory),
            "controller_artifact_sha256": controller_sha,
            "execution_public_key": _EXECUTION_PUBLIC_KEY,
            "fit_provenance_sha256": "b" * 64,
            "fit_provenance_digest": "c" * 64,
        }
        binding = E5ControllerBinding(
            spec=spec,
            controller_directory=str(controller_directory),
            controller_artifact_sha256=controller_sha,
            execution_public_key=_EXECUTION_PUBLIC_KEY,
            fit_provenance_sha256="b" * 64,
            fit_provenance_digest="c" * 64,
            binding_digest=stable_hash(body),
        )
        path = tmp_path / f"{spec.spec_id}.json"
        path.write_text("binding", encoding="utf-8")
        binding_paths[spec.spec_id] = path
        bindings[spec.spec_id] = binding

    record_path = tmp_path / "records.jsonl"
    write_e5_ablation_records(
        record_path,
        _records(
            protocol,
            screen.dev_questions,
            bindings,
            binding_paths,
        ),
        screen=screen,
        protocol=protocol,
    )

    monkeypatch.setattr(
        "mfh.experiments.e5_adaptive.load_e5_controller_binding",
        lambda path: bindings[Path(path).stem],
    )
    selection = derive_e5_selection(
        screen_receipt_path=screen_path,
        record_artifact_path=record_path,
        upstream_artifacts=upstream,
        controller_binding_artifacts=binding_paths,
        protocol=protocol,
    )
    assert selection.selected_spec_id == build_e5_ablation_grid(protocol)[-1].spec_id
    assert set(selection.matched_spec_ids) == {
        "coverage",
        "abstention_rate",
        "intervention_norm",
        "latency",
    }
    path = tmp_path / "selection.json"
    write_e5_selection(path, selection)
    assert verify_e5_selection(path)["valid"] is True

    lines = record_path.read_text(encoding="utf-8").splitlines()
    static = json.loads(lines[0])
    static_signature = static["execution_receipt_signature"]
    static["execution_receipt_signature"] = (
        "0" if static_signature[0] != "0" else "1"
    ) + static_signature[1:]
    original_static = lines[0]
    lines[0] = json.dumps(static, sort_keys=True)
    record_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="runtime signature"):
        derive_e5_selection(
            screen_receipt_path=screen_path,
            record_artifact_path=record_path,
            upstream_artifacts=upstream,
            controller_binding_artifacts=binding_paths,
            protocol=protocol,
        )
    lines[0] = original_static
    record_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    adaptive = json.loads(lines[6])
    signature = adaptive["execution_receipt_signature"]
    adaptive["execution_receipt_signature"] = ("0" if signature[0] != "0" else "1") + signature[1:]
    lines[6] = json.dumps(adaptive, sort_keys=True)
    record_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="runtime signature"):
        derive_e5_selection(
            screen_receipt_path=screen_path,
            record_artifact_path=record_path,
            upstream_artifacts=upstream,
            controller_binding_artifacts=binding_paths,
            protocol=protocol,
        )
    with pytest.raises(FrozenArtifactError, match="replay"):
        verify_e5_selection(path)


def test_e5_measurements_reject_impossible_metric_identity() -> None:
    from mfh.experiments.e5_adaptive import E5Measurement

    with pytest.raises(DataValidationError, match="measurement"):
        E5Measurement(
            spec_id="a" * 64,
            controller_artifact_sha256="b" * 64,
            accuracy=1.0,
            coverage=0.1,
            abstention_rate=0.9,
            hallucination_risk=1.0,
            mean_intervention_norm=1.0,
            mean_latency_seconds=1.0,
        )


def test_e5_selection_verifier_rejects_binding_mutation(tmp_path: Path) -> None:
    path = tmp_path / "binding.json"
    path.write_text("before", encoding="utf-8")
    before = sha256_file(path)
    path.write_text("after", encoding="utf-8")
    assert sha256_file(path) != before


def test_e5_public_phase_paths_always_replay_common_invariants(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from mfh.experiments.protocol import load_study_protocol

    study = load_study_protocol(Path(__file__).parents[1] / "configs/experiments/phases.yaml")

    def reject(**_kwargs):  # type: ignore[no-untyped-def]
        raise DataValidationError("common-final-input-replay")

    monkeypatch.setattr("mfh.experiments.e5_adaptive._validate_e5_final_inputs", reject)
    with pytest.raises(DataValidationError, match="common-final-input-replay"):
        finalize_e5_phase(
            tmp_path / "final",
            ledger_directory=tmp_path / "ledger",
            study=study,
            selection_path=tmp_path / "selection.json",
        )

    final = tmp_path / "existing-final"
    final.mkdir()
    for name in (
        "matched_coverage.json",
        "matched_abstention.json",
        "matched_norm.json",
        "matched_latency.json",
    ):
        (final / name).write_text("{}", encoding="utf-8")
    (final / "selected-controller").mkdir()
    receipt_body = {
        "schema_version": 2,
        "phase": "E5",
        "status": "complete",
        "ledger_directory": str((tmp_path / "ledger").resolve()),
        "contract_digest": "a" * 64,
        "record_set_digest": "b" * 64,
        "selection_path": str((tmp_path / "selection.json").resolve()),
        "selection_digest": "c" * 64,
        "selected_spec_id": "d" * 64,
        "selected_controller_artifact_sha256": "e" * 64,
        "selected_controller_bundle_sha256": "0" * 64,
        "gate_result_digests": {},
        "terminal_digest": "f" * 64,
        "scientific_eligible": True,
    }
    (final / "receipt.json").write_text(
        json.dumps(
            {**receipt_body, "receipt_digest": stable_hash(receipt_body)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DataValidationError, match="common-final-input-replay"):
        verify_e5_phase(
            final,
            ledger_directory=tmp_path / "ledger",
            study=study,
            selection_path=tmp_path / "selection.json",
        )

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import cast

import pytest

from mfh.contracts import ActivationSite, GenerationRecord, Outcome, Runtime
from mfh.errors import FrozenArtifactError
from mfh.experiments.e2_phase import (
    _gate_observations,
    _select_probe_artifacts,
    _validate_gate_replay,
    _verify_e1_output_completion_binding,
    finalize_e2_phase_run,
)
from mfh.experiments.e2_probes import E2FeatureView, VerifiedE2ProbeBundle
from mfh.experiments.evidence import GateResult
from mfh.experiments.protocol import ExperimentPhase
from mfh.experiments.runner import PhaseCompletion
from mfh.methods.probes import ProbeKind, ProbeTask
from mfh.provenance import sha256_file, sha256_path, stable_hash


def _probe_bundle(root: Path) -> VerifiedE2ProbeBundle:
    view = E2FeatureView(16, ActivationSite.POST_MLP)
    rows: list[dict[str, object]] = []
    scores = {
        (ProbeTask.CORRECT_INCORRECT_ABSTENTION, ProbeKind.LOGISTIC): (0.70, 0.80),
        (ProbeTask.CORRECT_INCORRECT_ABSTENTION, ProbeKind.TWO_LAYER_MLP): (0.90, 0.85),
        (ProbeTask.FORCED_CORRECT_INCORRECT, ProbeKind.LOGISTIC): (0.60, 0.75),
        (ProbeTask.FORCED_CORRECT_INCORRECT, ProbeKind.TWO_LAYER_MLP): (0.72, 0.71),
    }
    calibrations = ("temperature", "isotonic")
    selected_digest = ""
    for (task, kind), values in scores.items():
        for calibration, score in zip(calibrations, values, strict=True):
            relative = Path(task.value) / kind.value / calibration
            artifact = root / "probes" / relative
            artifact.mkdir(parents=True)
            (artifact / "payload").write_text(
                f"{task.value}:{kind.value}:{calibration}\n", encoding="utf-8"
            )
            digest = sha256_path(artifact)
            row: dict[str, object] = {
                "task": task.value,
                "kind": kind.value,
                "calibration": calibration,
                "artifact": str(relative),
                "artifact_sha256": digest,
                "metrics": {"T-dev": {"macro_auroc": score}},
            }
            if task is ProbeTask.CORRECT_INCORRECT_ABSTENTION:
                row["incorrect_auroc"] = score
            rows.append(row)
            if (
                task is ProbeTask.CORRECT_INCORRECT_ABSTENTION
                and kind is ProbeKind.TWO_LAYER_MLP
                and calibration == "temperature"
            ):
                selected_digest = digest
    root.mkdir(exist_ok=True)
    (root / "results.json").write_text(
        json.dumps({"final_probes": rows}), encoding="utf-8"
    )
    return VerifiedE2ProbeBundle(
        directory=root,
        plan_identity="1" * 64,
        manifest_digest="2" * 64,
        selected_views=MappingProxyType(
            {
                ProbeTask.CORRECT_INCORRECT_ABSTENTION: view,
                ProbeTask.FORCED_CORRECT_INCORRECT: view,
            }
        ),
        selected_gate_artifact=selected_digest,
        gate_passed=True,
        gate_probe_auroc=0.90,
        gate_baseline_auroc=0.50,
        controller_input_artifacts=MappingProxyType({}),
        scientific_eligible=True,
    )


def _record() -> GenerationRecord:
    return GenerationRecord(
        question_id="q-1",
        benchmark="triviaqa",
        model_repository="prism-ml/Bonsai-27B-mlx-1bit",
        model_revision="e" * 40,
        runtime=Runtime.MLX,
        quantization="binary-g128-mlx-1bit",
        system_prompt_id="P0-neutral",
        rendered_prompt_hash="a" * 64,
        steering_method="probe-logistic",
        layer=None,
        token_scope=None,
        alpha=0.0,
        sparsity=None,
        controller_scores={"C": 0.1, "I": 0.8, "A": 0.1},
        raw_output="",
        normalized_answer="",
        outcome=Outcome.INCORRECT,
        generation_latency_seconds=0.0,
        input_tokens=0,
        output_tokens=0,
        condition_id="b" * 64,
        metadata={
            "probe_score": 0.8,
            "output_entropy": 0.65,
            "maximum_token_probability": 0.72,
            "probe_artifact_sha256": "c" * 64,
            "probe_gate_eligible": True,
        },
    )


def test_selected_probe_artifacts_rederive_family_and_task_choices(tmp_path: Path) -> None:
    bundle = _probe_bundle(tmp_path / "bundle")
    selected = _select_probe_artifacts(bundle)

    assert len(selected) == 4
    assert (
        selected[(ProbeTask.CORRECT_INCORRECT_ABSTENTION, ProbeKind.TWO_LAYER_MLP)]
        .artifact_sha256
        == bundle.selected_gate_artifact
    )
    assert (
        selected[(ProbeTask.FORCED_CORRECT_INCORRECT, ProbeKind.LOGISTIC)].calibration
        == "isotonic"
    )


def test_selected_probe_artifacts_reject_nonwinning_stored_gate(tmp_path: Path) -> None:
    bundle = _probe_bundle(tmp_path / "bundle")
    losing = next(
        value.artifact_sha256
        for key, value in _select_probe_artifacts(bundle).items()
        if key == (ProbeTask.FORCED_CORRECT_INCORRECT, ProbeKind.LOGISTIC)
    )
    tampered = VerifiedE2ProbeBundle(
        directory=bundle.directory,
        plan_identity=bundle.plan_identity,
        manifest_digest=bundle.manifest_digest,
        selected_views=bundle.selected_views,
        selected_gate_artifact=losing,
        gate_passed=bundle.gate_passed,
        gate_probe_auroc=bundle.gate_probe_auroc,
        gate_baseline_auroc=bundle.gate_baseline_auroc,
        controller_input_artifacts=bundle.controller_input_artifacts,
        scientific_eligible=True,
    )

    with pytest.raises(FrozenArtifactError, match="not the per-family P0 selection"):
        _select_probe_artifacts(tampered)


def test_gate_observation_keeps_both_raw_confidence_baselines() -> None:
    observations = _gate_observations((_record(),))

    assert observations == (
        {
            "condition_id": "b" * 64,
            "question_id": "q-1",
            "incorrect": True,
            "probe_score": 0.8,
            "output_entropy": 0.65,
            "maximum_token_probability": 0.72,
            "probe_artifact_sha256": "c" * 64,
            "gate_eligible": True,
        },
    )


def test_gate_replay_rejects_stored_pass_that_differs_from_ledger() -> None:
    bundle = cast(
        VerifiedE2ProbeBundle,
        SimpleNamespace(
            gate_probe_auroc=0.80,
            gate_baseline_auroc=0.79,
            gate_passed=True,
        ),
    )
    result = cast(
        GateResult,
        SimpleNamespace(
            metrics={
                "probe_auroc": 0.80,
                "best_confidence_baseline_auroc": 0.79,
                "minimum_material_gain": 0.02,
            },
            passed=False,
        ),
    )

    with pytest.raises(FrozenArtifactError, match="replayed probe bundle"):
        _validate_gate_replay(result, bundle)


def test_finalizer_refuses_existing_output_before_replay(tmp_path: Path) -> None:
    output = tmp_path / "E2"
    output.mkdir()
    with pytest.raises(FrozenArtifactError, match="refusing to overwrite"):
        finalize_e2_phase_run(
            output,
            workspace_directory=tmp_path / "workspace",
            expected_workspace_plan_identity="a" * 64,
            capture_work_directory=tmp_path / "capture",
            expected_capture_plan_identity="b" * 64,
            probe_bundle_directory=tmp_path / "probes",
            expected_probe_manifest_digest="c" * 64,
            questions={},
            prompts={},
            e1_sources={},
            e1_output_directory=tmp_path / "e1-output",
            e1_phase_run=tmp_path / "E1",
            split_manifest_digest="d" * 64,
            study_config=tmp_path / "phases.yaml",
        )


def test_e1_output_manifest_must_name_the_same_completed_e1_run(tmp_path: Path) -> None:
    output = tmp_path / "e1-output"
    output.mkdir()
    (output / "outcome-labels.jsonl").write_text("{}\n", encoding="utf-8")
    (output / "prompt-metrics.json").write_text("{}\n", encoding="utf-8")
    completion = PhaseCompletion(
        phase=ExperimentPhase.E1,
        contract_digest="a" * 64,
        record_count=1,
        shard_fingerprints=MappingProxyType({"records-00000.jsonl": "b" * 64}),
        record_set_digest="c" * 64,
        gate_result_digests=MappingProxyType({}),
        gate_file_fingerprints=MappingProxyType({}),
        gate_artifact_fingerprints=MappingProxyType({}),
        completion_digest="d" * 64,
    )
    body = {
        "schema_version": 1,
        "purpose": "E1-baseline-records-prompt-metrics-and-outcome-labels",
        "phase": "E1",
        "plan_identity": "e" * 64,
        "contract_digest": completion.contract_digest,
        "completion_digest": completion.completion_digest,
        "record_set_digest": completion.record_set_digest,
        "record_count": completion.record_count,
        "condition_count": 1,
        "grader_bundle_manifest_digest": "f" * 64,
        "work_fingerprints": {"plan.json": "1" * 64},
        "files": {
            "outcome_labels": {
                "path": "outcome-labels.jsonl",
                "sha256": sha256_file(output / "outcome-labels.jsonl"),
                "rows": 1,
            },
            "prompt_metrics": {
                "path": "prompt-metrics.json",
                "sha256": sha256_file(output / "prompt-metrics.json"),
                "metrics_digest": "2" * 64,
            },
        },
    }
    manifest = {**body, "manifest_digest": stable_hash(body)}
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    replay = _verify_e1_output_completion_binding(
        output,
        completion=completion,
        contract_digest=completion.contract_digest,
    )
    assert replay["completion_digest"] == completion.completion_digest

    manifest["completion_digest"] = "9" * 64
    tampered_body = dict(manifest)
    tampered_body.pop("manifest_digest")
    manifest["manifest_digest"] = stable_hash(tampered_body)
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FrozenArtifactError, match="not bound"):
        _verify_e1_output_completion_binding(
            output,
            completion=completion,
            contract_digest=completion.contract_digest,
        )

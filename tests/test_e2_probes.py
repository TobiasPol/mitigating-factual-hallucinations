from __future__ import annotations

import tempfile
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import mfh.experiments.e2_probes as e2_probes
from mfh.contracts import ActivationSite, ModelSpec, Outcome, Question, Runtime
from mfh.experiments.activation_store import ActivationCaptureRow, append_activation_shard
from mfh.experiments.e2_controller_inputs import controller_input_views
from mfh.experiments.e2_probes import (
    E2ProbeProtocol,
    fit_e2_probe_bundle,
    verify_e2_probe_bundle,
)
from mfh.experiments.e2_schedule import (
    E2CaptureProtocol,
    build_e2_schedule,
    controller_feature_partitions,
    write_e2_workspace,
)
from mfh.methods.features import FeatureComposition
from mfh.methods.probes import ProbeTask


def _question(question_id: str, benchmark: str) -> Question:
    return Question(
        question_id=question_id,
        benchmark=benchmark,
        text=f"Question {question_id}?",
        aliases=(f"answer-{question_id}",),
    )


def test_e2_controller_input_geometry_is_deterministic_and_complete() -> None:
    views = controller_input_views(
        selected_layer=31,
        selected_site=ActivationSite.POST_MLP,
        candidate_layers=(7, 15, 23, 31, 39, 47, 55),
    )
    assert tuple(value.composition for value in views) == (
        FeatureComposition.SINGLE_LAYER,
        FeatureComposition.CONCATENATED_LAYERS,
        FeatureComposition.LAYER_DIFFERENCES,
    )
    assert views[0].layers == (31,)
    assert views[1].layers == views[2].layers == (23, 31, 39)


def _workspace(root: Path):  # type: ignore[no-untyped-def]
    protocol = E2CaptureProtocol(
        controller_rows=12,
        controller_calibration_rows=4,
        dev_rows=9,
        simpleqa_rows=6,
        aa_rows=6,
    )
    controller = tuple(
        _question(f"controller-{index}", "triviaqa") for index in range(12)
    )
    dev = tuple(_question(f"dev-{index}", "triviaqa") for index in range(9))
    simpleqa = tuple(
        _question(f"simple-{index}", "simpleqa_verified") for index in range(6)
    )
    aa = tuple(
        _question(f"aa-{index}", "aa_omniscience_public_600") for index in range(6)
    )
    partitions = controller_feature_partitions(
        controller,
        calibration_rows=protocol.controller_calibration_rows,
        seed=protocol.seed,
    )
    counters: dict[str, int] = defaultdict(int)
    outcomes: dict[tuple[str, str], Outcome] = {}
    classes = (Outcome.CORRECT, Outcome.INCORRECT, Outcome.ABSTENTION)
    for question in controller:
        partition = partitions[question.question_id]
        outcomes[(question.benchmark, question.question_id)] = classes[
            counters[partition] % len(classes)
        ]
        counters[partition] += 1
    for questions in (simpleqa, aa):
        for index, question in enumerate(questions):
            outcomes[(question.benchmark, question.question_id)] = classes[
                index % len(classes)
            ]
    schedule = build_e2_schedule(
        controller=controller,
        dev=dev,
        simpleqa=simpleqa,
        aa=aa,
        e1_p0_outcomes=outcomes,
        protocol=protocol,
    )
    model = ModelSpec(
        name="qwen3.6-27b-nvfp4",
        repository="nvidia/Qwen3.6-27B-NVFP4",
        revision="0893e1606ff3d5f97a441f405d5fc541a6bdf404",
        runtime=Runtime.VLLM,
        quantization="modelopt-mixed-nvfp4-fp8",
        num_layers=64,
    )
    workspace = write_e2_workspace(
        root / "workspace",
        schedule=schedule,
        protocol=protocol,
        model=model,
        hidden_width=4,
        input_fingerprints={"e1": "1" * 64, "splits": "2" * 64},
    )
    partition_counters: dict[tuple[str, str], int] = defaultdict(int)
    rows: list[ActivationCaptureRow] = []
    values = np.zeros((len(schedule), 3, 7, 4), dtype=np.float32)
    outcome_signal = {
        Outcome.CORRECT: -2.0,
        Outcome.INCORRECT: 2.0,
        Outcome.ABSTENTION: 0.0,
    }
    for sequence, schedule_row in enumerate(schedule):
        if schedule_row.outcome is not None:
            outcome = schedule_row.outcome
        else:
            key = (schedule_row.feature_partition, schedule_row.prompt_id)
            index = partition_counters[key]
            partition_counters[key] += 1
            choices = (
                (Outcome.CORRECT, Outcome.INCORRECT)
                if schedule_row.prompt_id == "P3-forced-answer"
                else (Outcome.CORRECT, Outcome.INCORRECT, Outcome.ABSTENTION)
            )
            outcome = choices[index % len(choices)]
        values[sequence, 1, 0, 0] = outcome_signal[outcome]
        rows.append(
            ActivationCaptureRow(
                question_id=schedule_row.question_id,
                benchmark=schedule_row.benchmark,
                partition=schedule_row.feature_partition,
                prompt_id=schedule_row.prompt_id,
                outcome=outcome,
                semantic_group_id=schedule_row.semantic_group_id,
                rendered_prompt_sha256="3" * 64,
                prompt_token_ids_sha256="4" * 64,
                generation_record_sha256="5" * 64,
                maximum_token_probability=0.5,
                output_entropy=0.5,
            )
        )
    append_activation_shard(
        workspace.directory / "activations",
        rows,
        values,
        expected_spec=workspace.activation_spec,
    )
    return workspace


def test_e2_probe_bundle_fits_all_tasks_models_calibrators_and_replays() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        workspace = _workspace(root)
        protocol = E2ProbeProtocol(
            candidate_layers=(31,),
            candidate_sites=(ActivationSite.POST_MLP,),
            screening_epochs=8,
            final_epochs=8,
            mlp_hidden_width=4,
        )
        result = fit_e2_probe_bundle(
            root / "probes",
            workspace=workspace,
            split_manifest_digest="b" * 64,
            prompt_template_sha256={
                "P0-neutral": "a" * 64,
                "P3-forced-answer": "c" * 64,
            },
            protocol=protocol,
        )
        assert set(result.selected_views) == set(ProbeTask)
        assert result.scientific_eligible is False
        assert result.selected_gate_artifact
        replay = verify_e2_probe_bundle(root / "probes", workspace=workspace)
        assert replay.manifest_digest == result.manifest_digest
        assert replay.gate_probe_auroc >= 0.5


def test_e2_probe_bundle_resumes_from_verified_probe_artifacts() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        workspace = _workspace(root)
        protocol = E2ProbeProtocol(
            candidate_layers=(31,),
            candidate_sites=(ActivationSite.POST_MLP,),
            screening_epochs=2,
            final_epochs=2,
            mlp_hidden_width=4,
        )
        original = e2_probes._fit_state_and_calibrators
        first_calls = 0

        def interrupt(*args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal first_calls
            first_calls += 1
            if first_calls == 3:
                raise RuntimeError("simulated CPU interruption")
            return original(*args, **kwargs)

        inputs = {
            "workspace": workspace,
            "split_manifest_digest": "b" * 64,
            "prompt_template_sha256": {
                "P0-neutral": "a" * 64,
                "P3-forced-answer": "c" * 64,
            },
            "protocol": protocol,
            "work_directory": root / "probe-work",
        }
        with (
            patch.object(e2_probes, "_fit_state_and_calibrators", interrupt),
            pytest.raises(RuntimeError, match="simulated CPU interruption"),
        ):
            fit_e2_probe_bundle(root / "probes", **inputs)
        assert first_calls == 3
        assert not (root / "probes").exists()

        resumed_calls = 0

        def count_resume(*args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal resumed_calls
            resumed_calls += 1
            return original(*args, **kwargs)

        with patch.object(e2_probes, "_fit_state_and_calibrators", count_resume):
            result = fit_e2_probe_bundle(root / "probes", **inputs)
        assert result.manifest_digest
        assert set(result.controller_input_artifacts) == {
            FeatureComposition.SINGLE_LAYER
        }
        # Ten unfinished registered final-probe fits plus the separately frozen
        # one-layer E5 controller-input probe.
        assert resumed_calls == 11
        assert not (root / "probe-work").exists()

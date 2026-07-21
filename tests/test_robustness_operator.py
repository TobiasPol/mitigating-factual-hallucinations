from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from mfh.contracts import ActivationSite, Outcome, PromptSpec, Question
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.experiments import robustness_operator as operator
from mfh.experiments.rq1_capture import VerifiedRQ1Capture, rq1_capture_public_key
from mfh.methods.features import ActivationFeatureSchema
from mfh.methods.probes import ProbeDataset
from mfh.provenance import canonical_json, stable_hash


def _write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _bound_store_and_runbook(tmp_path: Path) -> tuple[Any, Any, dict[str, Path]]:
    plan = tmp_path / "plan"
    sources = {
        "graders": _write(plan / "sources/frozen-graders/value", "graders"),
        "components": _write(
            plan / "sources/frozen-component-selection/value", "components"
        ),
        "evaluation": _write(
            plan / "sources/frozen-evaluation-scripts/value", "evaluation"
        ),
    }
    runbook = SimpleNamespace(
        phase=SimpleNamespace(value="E9"),
        input_artifacts={
            "frozen_prompt_paraphrase_schedule": plan,
            "frozen_graders": sources["graders"].parent,
            "frozen_component_selection": sources["components"].parent,
            "frozen_evaluation_scripts": sources["evaluation"].parent,
        },
    )
    return SimpleNamespace(plan=SimpleNamespace(path=plan)), runbook, sources


@pytest.mark.parametrize(
    "changed",
    ("phase", "schedule", "graders", "components", "evaluation"),
)
def test_e9_runbook_binding_rejects_every_changed_boundary(
    tmp_path: Path, changed: str
) -> None:
    store, runbook, _ = _bound_store_and_runbook(tmp_path)
    operator._validate_e9_runbook_binding(store, runbook)
    mismatch = _write(tmp_path / f"mismatch-{changed}", changed)
    if changed == "phase":
        runbook.phase.value = "E10"
    elif changed == "schedule":
        runbook.input_artifacts["frozen_prompt_paraphrase_schedule"] = mismatch
    else:
        runbook.input_artifacts[
            {
                "graders": "frozen_graders",
                "components": "frozen_component_selection",
                "evaluation": "frozen_evaluation_scripts",
            }[changed]
        ] = mismatch

    with pytest.raises(DataValidationError, match="exact E9 runbook"):
        operator._validate_e9_runbook_binding(store, runbook)


def test_prompt_runner_enforces_runbook_binding_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SimpleNamespace()
    runbook = SimpleNamespace()
    observed: list[tuple[Any, Any]] = []
    monkeypatch.setattr(operator, "open_robustness_result_store", lambda _path: store)
    monkeypatch.setattr(
        operator.ConfirmatoryRunbook,
        "load",
        staticmethod(lambda _path: runbook),
    )

    def reject(received_store: Any, received_runbook: Any) -> None:
        observed.append((received_store, received_runbook))
        raise DataValidationError("bound before backend")

    monkeypatch.setattr(operator, "_validate_e9_runbook_binding", reject)

    with pytest.raises(DataValidationError, match="bound before backend"):
        operator.run_prompt_paraphrase_diagnostics(
            "results",
            e9_runbook="runbook",
            execution_private_key="0" * 64,
            openrouter_api_key="secret",
            limit=1,
        )
    assert observed == [(store, runbook)]


def _dataset(
    schema: ActivationFeatureSchema, question_id: str, outcome: Outcome
) -> ProbeDataset:
    return ProbeDataset(
        question_ids=(question_id,),
        features=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        outcomes=(outcome,),
        group_ids=(f"group-{question_id}",),
        feature_schema=schema,
    )


@pytest.mark.parametrize("changed", ("training", "calibration"))
def test_native_fit_inputs_rejects_changed_e2_fingerprints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, changed: str
) -> None:
    schema = ActivationFeatureSchema.synthetic(partition="T-controller", width=2)
    training = _dataset(schema, "train", Outcome.CORRECT)
    calibration = _dataset(schema, "calibration", Outcome.INCORRECT)
    fingerprints = {
        "training": training.data_fingerprint,
        "calibration": calibration.data_fingerprint,
    }
    fingerprints[changed] = "0" * 64
    risk_probe = SimpleNamespace(
        training_fingerprint=fingerprints["training"],
        calibration_fingerprint=fingerprints["calibration"],
    )
    controller = SimpleNamespace(risk_probe=risk_probe, layer_selector=None)
    context = SimpleNamespace(
        snapshot=object(),
        questions=(),
        prompts={"P0-neutral": PromptSpec("P0-neutral", "Answer.")},
        store=SimpleNamespace(plan=object()),
        feature_schema=schema,
        base_component=SimpleNamespace(controllers={"P0-neutral": controller}),
    )
    grid = {
        (schema.composition, "T-controller-train"): SimpleNamespace(probe=training),
        (schema.composition, "T-controller-calibration"): SimpleNamespace(
            probe=calibration
        ),
    }
    monkeypatch.setattr(operator, "e5_capture_public_key", lambda _key: "1" * 64)
    monkeypatch.setattr(operator, "verify_e5_fit_capture", lambda *_a, **_k: object())
    monkeypatch.setattr(
        operator,
        "_controller_inputs",
        lambda **_kwargs: ({schema.composition: training}, grid),
    )
    monkeypatch.setattr(operator, "read_questions", lambda _path: ())
    monkeypatch.setattr(
        operator,
        "load_e5_layer_label_data",
        lambda *_a, **_k: SimpleNamespace(
            question_ids=training.question_ids,
            best_layers_two=(0,),
            best_layers_three=(0,),
        ),
    )
    monkeypatch.setattr(operator, "load_rq1_capture_data", lambda *_a, **_k: object())
    monkeypatch.setattr(operator, "sha256_path", lambda _path: "2" * 64)

    with pytest.raises(FrozenArtifactError, match="differ from base M3"):
        operator._native_fit_inputs(
            context,
            execution_root=tmp_path,
            execution_private_key="0" * 64,
            e2_workspace=tmp_path / "e2-workspace",
            e2_probe_bundle=tmp_path / "e2-probes",
            e5_fit_capture=tmp_path / "e5-capture",
            e5_layer_labels=tmp_path / "e5-labels",
            controller_questions=tmp_path / "questions.jsonl",
        )


class _Record(SimpleNamespace):
    def to_dict(self) -> dict[str, Any]:
        return dict(self.serialized)


def test_rq1_capture_resumes_with_one_linear_signature_chain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    capture = tmp_path / "capture"
    (capture / "shards").mkdir(parents=True)
    (capture / "run.lock").touch()
    schema = ActivationFeatureSchema.synthetic(partition="T-steer", width=2)
    records = tuple(
        _Record(sequence=index, serialized={"sequence": index}) for index in range(2)
    )
    rows = tuple(
        {
            "source_sequence": index,
            "source_record_sha256": stable_hash(record.to_dict()),
            "question_id": f"q-{index}",
        }
        for index, record in enumerate(records)
    )
    frozen = {
        "capture_plan_identity": "a" * 64,
        "runtime_identity": {"runtime": "test"},
        "expected_rows": 2,
        "source_rows": rows,
        "feature_schema": json.loads(canonical_json(schema.to_dict())),
        "vector_hooks": [{"layer": 0, "site": ActivationSite.POST_MLP.value}],
        "hidden_width": 2,
        "shard_rows": 1,
    }
    calls: list[int] = []

    def verify(directory: Path, **_kwargs: Any) -> VerifiedRQ1Capture:
        manifests = tuple(sorted((directory / "shards").glob("shard-*/manifest.json")))
        calls.append(len(manifests))
        head = None
        completed = 0
        for path in manifests:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            head = str(manifest["manifest_digest"])
            completed += int(manifest["row_count"])
        return VerifiedRQ1Capture(
            directory=directory,
            plan=frozen,
            rows_completed=completed,
            shard_count=len(manifests),
            chain_head=head,
            complete=completed == 2,
        )

    def capture_row(*_args: Any, **kwargs: Any) -> tuple[Any, Any, int]:
        hook = kwargs["hooks"][0]
        return (
            np.asarray([1.0, 2.0], dtype=np.float32),
            {hook.artifact_key: np.asarray([3.0, 4.0], dtype=np.float32)},
            10,
        )

    from mfh.experiments import rq1_capture as capture_module

    monkeypatch.setattr(capture_module, "verify_rq1_capture", verify)
    monkeypatch.setattr(capture_module, "_capture_row", capture_row)
    snapshot = SimpleNamespace(generations=records)
    questions = tuple(
        Question(
            f"q-{index}",
            "triviaqa",
            f"Question {index}?",
            ("answer",),
            split="T-steer",
            entities=(f"entity-{index}",),
        )
        for index in range(2)
    )
    runtime = SimpleNamespace(runtime_identity=lambda: {"runtime": "test"})
    key = "0" * 64

    first = capture_module.run_rq1_capture(
        capture,
        plan=object(),
        snapshot=snapshot,
        questions=questions,
        prompt=PromptSpec("P0-neutral", "Answer."),
        runtime=runtime,
        private_key_hex=key,
        limit=1,
    )
    second = capture_module.run_rq1_capture(
        capture,
        plan=object(),
        snapshot=snapshot,
        questions=questions,
        prompt=PromptSpec("P0-neutral", "Answer."),
        runtime=runtime,
        private_key_hex=key,
        limit=1,
    )

    assert first.rows_completed == 1
    assert second.complete
    assert calls == [0, 1, 2]
    manifests = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((capture / "shards").glob("shard-*/manifest.json"))
    ]
    assert manifests[0]["previous_manifest_digest"] is None
    assert manifests[1]["previous_manifest_digest"] == manifests[0]["manifest_digest"]
    public = Ed25519PublicKey.from_public_bytes(bytes.fromhex(rq1_capture_public_key(key)))
    for manifest in manifests:
        signature = bytes.fromhex(manifest.pop("signature"))
        public.verify(signature, canonical_json(manifest).encode())


def test_one_mocked_m1_rq1_task_reaches_atomic_append(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    task = SimpleNamespace(task_id="task-1", method="M1")
    store = SimpleNamespace(plan=object())
    context = SimpleNamespace(store=store)
    question = Question(
        "held",
        "triviaqa",
        "Held question?",
        ("answer",),
        split="T-test",
        entities=("entity",),
    )
    scopes: list[tuple[str, str]] = []
    appended: list[dict[str, Any]] = []
    monkeypatch.setattr(
        operator,
        "rq1_task_question_sets",
        lambda _plan, _task: {"held_out_evaluation": ("held",)},
    )
    monkeypatch.setattr(operator, "_rq1_questions_from_plan", lambda _plan: {"held": question})
    monkeypatch.setattr(
        operator,
        "_frozen_execution_component",
        lambda _plan, _method: tmp_path / "m1",
    )

    def write_scope(directory: Path, **kwargs: Any) -> Any:
        directory.mkdir(parents=True)
        scopes.append((str(kwargs["stage"]), str(kwargs["execution_component"])))
        return SimpleNamespace(directory=directory)

    monkeypatch.setattr(operator, "write_rq1_scoped_component", write_scope)
    monkeypatch.setattr(
        operator,
        "write_rq1_fit_receipt",
        lambda directory, **_kwargs: directory.mkdir(parents=True),
    )
    monkeypatch.setattr(
        operator,
        "execute_rq1_evaluation_records",
        lambda directory, **_kwargs: directory.mkdir(parents=True),
    )
    monkeypatch.setattr(
        operator,
        "append_rq1_generalization_result",
        lambda _store, **kwargs: appended.append(kwargs),
    )
    backend = SimpleNamespace(grader_bundle=tmp_path / "graders")

    operator._run_one_rq1(
        tmp_path / "task-stage",
        context=context,
        task=task,
        capture=object(),
        controller_train=object(),
        controller_calibration=object(),
        labels={},
        private_key="0" * 64,
        backend=backend,
    )

    assert [value[0] for value in scopes] == ["source-fit", "held-out-adaptation"]
    assert appended[0]["task"] is task
    assert appended[0]["questions_by_id"] == {"held": question}
    assert set(appended[0]["artifacts"]) == {
        "source_component",
        "adapted_component",
        "fit_receipt",
        "evaluation_records",
    }

"""End-to-end native operator for the two frozen robustness diagnostics."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from torch import Tensor

from mfh.config import load_model_spec, load_prompt_specs
from mfh.contracts import AdaptivePolicySpec, PromptSpec, Question
from mfh.data.io import read_questions
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.evaluation.openrouter import OpenRouterTransport
from mfh.experiments.activation_store import verify_activation_store
from mfh.experiments.confirmatory_components import (
    ConfirmatoryAdaptiveComponent,
    load_confirmatory_adaptive_component,
    write_confirmatory_adaptive_component,
)
from mfh.experiments.confirmatory_operator import ConfirmatoryRunbook, _native_runtime
from mfh.experiments.e2_controller_inputs import (
    build_e2_controller_input_datasets,
    controller_input_views,
)
from mfh.experiments.e2_probes import verify_e2_probe_bundle
from mfh.experiments.e2_schedule import verify_e2_workspace
from mfh.experiments.e3_construction import (
    VerifiedE3ConstructionSnapshot,
    load_verified_e3_construction_snapshot,
)
from mfh.experiments.e5_capture import (
    e5_capture_public_key,
    verify_e5_fit_capture,
)
from mfh.experiments.e5_layer_labels import load_e5_layer_label_data
from mfh.experiments.e9_native import NativeE9VllmBackend
from mfh.experiments.model_selection import validate_active_study_artifact_paths
from mfh.experiments.robustness_diagnostics import (
    RQ1GeneralizationTask,
    iter_prompt_paraphrase_tasks,
    iter_rq1_generalization_tasks,
    rq1_task_question_sets,
)
from mfh.experiments.robustness_results import (
    M3FitRecipe,
    RobustnessResultStore,
    _base_component_descriptor,
    _evaluation_questions_from_plan,
    _frozen_execution_component,
    _rq1_questions_from_plan,
    append_rq1_generalization_result,
    execute_prompt_paraphrase_task,
    execute_rq1_evaluation_records,
    fit_rq1_m3_controller,
    open_robustness_result_store,
    refit_rq1_m3_vector_bank_controller,
    robustness_result_progress,
    rq1_m3_fit_capture_attestation_body,
    write_rq1_fit_receipt,
    write_rq1_scoped_component,
)
from mfh.experiments.rq1_capture import (
    RQ1CaptureData,
    load_rq1_capture_data,
    prepare_rq1_capture,
    rq1_capture_public_key,
    run_rq1_capture,
    verify_rq1_capture,
)
from mfh.inference.architecture import HookKey
from mfh.methods.adaptive import (
    AdaptiveController,
    save_adaptive_controller,
)
from mfh.methods.features import ActivationFeatureSchema, FeatureComposition
from mfh.methods.probes import (
    CalibrationKind,
    IsotonicCalibrator,
    ProbeDataset,
    TemperatureCalibrator,
)
from mfh.provenance import canonical_json, sha256_file, sha256_path


@dataclass(frozen=True, slots=True)
class _CaptureContext:
    store: RobustnessResultStore
    runbook: ConfirmatoryRunbook
    snapshot: VerifiedE3ConstructionSnapshot
    questions: tuple[Question, ...]
    prompts: Mapping[str, PromptSpec]
    base_component: ConfirmatoryAdaptiveComponent
    base_policy: AdaptivePolicySpec
    feature_schema: ActivationFeatureSchema
    vector_hooks: tuple[HookKey, ...]


def _source_path(store: RobustnessResultStore, name: str) -> Path:
    if store.plan.path is None:
        raise FrozenArtifactError("robustness result store lacks its packaged plan")
    path = store.plan.path / "sources" / name
    if not path.exists() or path.is_symlink():
        raise FrozenArtifactError(f"packaged robustness source is missing: {name}")
    return path


def _sole_file(directory: Path, name: str) -> Path:
    values = tuple(value for value in directory.rglob("*") if value.is_file())
    exact = tuple(value for value in values if value.name == name)
    if len(exact) != 1:
        raise FrozenArtifactError(f"packaged robustness source lacks {name}")
    return exact[0]


def _base_m3(
    store: RobustnessResultStore,
) -> tuple[ConfirmatoryAdaptiveComponent, AdaptivePolicySpec]:
    descriptor = _base_component_descriptor(store.plan, "M3")
    component = _frozen_execution_component(store.plan, "M3")
    policy = descriptor.get("adaptive_policy")
    if component is None or not isinstance(policy, Mapping):
        raise FrozenArtifactError("robustness base M3 lacks its adaptive policy")
    return load_confirmatory_adaptive_component(component), AdaptivePolicySpec.from_dict(policy)


def _validate_e9_runbook_binding(
    store: RobustnessResultStore, runbook: ConfirmatoryRunbook
) -> None:
    if store.plan.path is None:
        raise FrozenArtifactError("robustness result store lacks a packaged plan")
    if (
        runbook.phase.value != "E9"
        or sha256_path(runbook.input_artifacts["frozen_prompt_paraphrase_schedule"])
        != sha256_path(store.plan.path)
        or sha256_path(runbook.input_artifacts["frozen_graders"])
        != sha256_path(_source_path(store, "frozen-graders"))
        or sha256_path(runbook.input_artifacts["frozen_component_selection"])
        != sha256_path(_source_path(store, "frozen-component-selection"))
        or sha256_path(runbook.input_artifacts["frozen_evaluation_scripts"])
        != sha256_path(_source_path(store, "frozen-evaluation-scripts"))
    ):
        raise DataValidationError("robustness execution requires its exact E9 runbook")


def _capture_context(
    results: str | Path,
    *,
    e9_runbook: str | Path,
    e3_construction: str | Path,
) -> _CaptureContext:
    store = open_robustness_result_store(results)
    runbook = ConfirmatoryRunbook.load(e9_runbook)
    _validate_e9_runbook_binding(store, runbook)
    prompts = {
        value.prompt_id: value
        for value in load_prompt_specs(_source_path(store, "canonical-prompts"))
        if value.prompt_id in {"P0-neutral", "P2-calibrated-abstention"}
    }
    if set(prompts) != {"P0-neutral", "P2-calibrated-abstention"}:
        raise FrozenArtifactError("robustness plan lacks its exact canonical prompts")
    questions = tuple(
        read_questions(_sole_file(_source_path(store, "triviaqa-development"), "T-steer.jsonl"))
    )
    snapshot = load_verified_e3_construction_snapshot(
        e3_construction,
        questions=questions,
        prompts=prompts,
    )
    component, policy = _base_m3(store)
    controller = component.controllers.get("P0-neutral")
    if controller is None or policy.execution_public_key is None:
        raise FrozenArtifactError("robustness base M3 lacks its P0 controller or key")
    return _CaptureContext(
        store=store,
        runbook=runbook,
        snapshot=snapshot,
        questions=questions,
        prompts=MappingProxyType(prompts),
        base_component=component,
        base_policy=policy,
        feature_schema=controller.risk_probe.training_schema,
        vector_hooks=tuple(controller.vector_bank.directions),
    )


def prepare_robustness_execution(
    directory: str | Path,
    *,
    results: str | Path,
    e9_runbook: str | Path,
    e3_construction: str | Path,
    shard_rows: int = 16,
) -> Mapping[str, Any]:
    root = validate_active_study_artifact_paths({"robustness execution": directory})[
        "robustness execution"
    ]
    context = _capture_context(
        results, e9_runbook=e9_runbook, e3_construction=e3_construction
    )
    root.mkdir(parents=True, exist_ok=True)
    runtime_sha = str(context.store.plan.body["m3_capture_runtime_artifact_sha256"])
    frozen = prepare_rq1_capture(
        root / "rq1-capture",
        plan=context.store.plan,
        snapshot=context.snapshot,
        questions=context.questions,
        prompt=context.prompts["P0-neutral"],
        feature_schema=context.feature_schema,
        vector_hooks=context.vector_hooks,
        runtime_identity=context.snapshot.plan["runtime_identity"],
        runtime_artifact_sha256=runtime_sha,
        execution_public_key=str(context.base_policy.execution_public_key),
        shard_rows=shard_rows,
    )
    return MappingProxyType(
        {
            "valid": True,
            "directory": str(root),
            "rq1_capture": str(root / "rq1-capture"),
            "expected_rows": frozen["expected_rows"],
            "capture_plan_identity": frozen["capture_plan_identity"],
        }
    )


def run_robustness_rq1_capture(
    directory: str | Path,
    *,
    results: str | Path,
    e9_runbook: str | Path,
    e3_construction: str | Path,
    execution_private_key: str,
    limit: int | None = None,
) -> Mapping[str, Any]:
    root = validate_active_study_artifact_paths({"robustness execution": directory})[
        "robustness execution"
    ]
    context = _capture_context(
        results, e9_runbook=e9_runbook, e3_construction=e3_construction
    )
    grader = _source_path(context.store, "frozen-graders")
    attestor, runtime_artifact = _native_runtime(
        context.runbook,
        execution_private_key=execution_private_key,
        packaged_grader=grader,
    )
    try:
        expected_runtime = str(
            context.store.plan.body["m3_capture_runtime_artifact_sha256"]
        )
        if sha256_file(runtime_artifact) != expected_runtime:
            raise FrozenArtifactError("RQ1 runtime differs from its frozen plan")
        captured = run_rq1_capture(
            root / "rq1-capture",
            plan=context.store.plan,
            snapshot=context.snapshot,
            questions=context.questions,
            prompt=context.prompts["P0-neutral"],
            runtime=attestor.runtime,
            private_key_hex=execution_private_key,
            limit=limit,
        )
    finally:
        attestor.runtime.close()
    return MappingProxyType(
        {
            "valid": True,
            "captured_rows": captured.rows_completed,
            "expected_rows": captured.plan["expected_rows"],
            "complete": captured.complete,
            "shard_count": captured.shard_count,
            "chain_head": captured.chain_head,
        }
    )


def verify_robustness_rq1_capture(
    directory: str | Path,
    *,
    results: str | Path,
    e9_runbook: str | Path,
    e3_construction: str | Path,
    execution_private_key: str,
    require_complete: bool = False,
) -> Mapping[str, Any]:
    root = validate_active_study_artifact_paths({"robustness execution": directory})[
        "robustness execution"
    ]
    context = _capture_context(
        results, e9_runbook=e9_runbook, e3_construction=e3_construction
    )
    captured = verify_rq1_capture(
        root / "rq1-capture",
        plan=context.store.plan,
        snapshot=context.snapshot,
        questions=context.questions,
        prompt=context.prompts["P0-neutral"],
        expected_execution_public_key=rq1_capture_public_key(execution_private_key),
        require_complete=require_complete,
    )
    return MappingProxyType(
        {
            "valid": True,
            "captured_rows": captured.rows_completed,
            "expected_rows": captured.plan["expected_rows"],
            "complete": captured.complete,
            "shard_count": captured.shard_count,
            "chain_head": captured.chain_head,
        }
    )


def _backend(
    store: RobustnessResultStore,
    runbook: ConfirmatoryRunbook,
    *,
    execution_private_key: str,
    openrouter_api_key: str,
) -> tuple[NativeE9VllmBackend, Any]:
    grader = _source_path(store, "frozen-graders")
    attestor, runtime_artifact = _native_runtime(
        runbook,
        execution_private_key=execution_private_key,
        packaged_grader=grader,
    )
    backend = NativeE9VllmBackend(
        attestor=attestor,
        runtime_artifact=runtime_artifact,
        grader_bundle=grader,
        grader_transport=OpenRouterTransport(api_key=openrouter_api_key),
    )
    return backend, attestor


def run_prompt_paraphrase_diagnostics(
    results: str | Path,
    *,
    e9_runbook: str | Path,
    execution_private_key: str,
    openrouter_api_key: str,
    limit: int | None = None,
) -> Mapping[str, Any]:
    """Resume a bounded prefix of the frozen 36,000 prompt tasks."""

    if limit is not None and (type(limit) is not int or limit <= 0):
        raise DataValidationError("prompt robustness limit must be positive")
    store = open_robustness_result_store(results)
    runbook = ConfirmatoryRunbook.load(e9_runbook)
    _validate_e9_runbook_binding(store, runbook)
    questions = _evaluation_questions_from_plan(store.plan)
    pending = tuple(
        task
        for task in iter_prompt_paraphrase_tasks(store.plan)
        if not (store.directory / "prompt-results" / f"{task.task_id}.json").exists()
    )
    selected = pending if limit is None else pending[:limit]
    if not selected:
        return MappingProxyType(
            {"valid": True, "executed": 0, "progress": dict(robustness_result_progress(store))}
        )
    backend, attestor = _backend(
        store,
        runbook,
        execution_private_key=execution_private_key,
        openrouter_api_key=openrouter_api_key,
    )
    try:
        for task in selected:
            execute_prompt_paraphrase_task(
                store,
                task=task,
                question=questions[task.question_id],
                backend=backend,
            )
    finally:
        attestor.runtime.close()
    return MappingProxyType(
        {
            "valid": True,
            "executed": len(selected),
            "progress": dict(robustness_result_progress(store)),
        }
    )


def _subset(dataset: ProbeDataset, identifiers: Sequence[str]) -> ProbeDataset:
    positions = {value: index for index, value in enumerate(dataset.question_ids)}
    try:
        indices = [positions[value] for value in identifiers]
    except KeyError as exc:
        raise FrozenArtifactError(
            f"RQ1 fit ID is missing from native inputs: {exc.args[0]}"
        ) from exc
    index = torch.tensor(indices, dtype=torch.long)
    return ProbeDataset(
        question_ids=tuple(identifiers),
        features=dataset.features[index],
        outcomes=tuple(dataset.outcomes[value] for value in indices),
        group_ids=tuple(dataset.group_ids[value] for value in indices),
        feature_schema=dataset.feature_schema,
    )


def _subset_activations(
    data: RQ1CaptureData, identifiers: Sequence[str]
) -> Mapping[HookKey, Tensor]:
    positions = {value: index for index, value in enumerate(data.vector_dataset.question_ids)}
    try:
        index = torch.tensor([positions[value] for value in identifiers], dtype=torch.long)
    except KeyError as exc:
        raise FrozenArtifactError(f"RQ1 activation ID is missing: {exc.args[0]}") from exc
    return MappingProxyType(
        {hook: value[index] for hook, value in data.vector_activations.items()}
    )


def _fit_recipe(
    store: RobustnessResultStore, controller: AdaptiveController
) -> M3FitRecipe:
    registered = store.plan.body["rq1_generalization"]["m3_refit_hyperparameters"]
    calibration = (
        CalibrationKind.TEMPERATURE
        if isinstance(controller.risk_probe.calibrator, TemperatureCalibrator)
        else CalibrationKind.ISOTONIC
        if isinstance(controller.risk_probe.calibrator, IsotonicCalibrator)
        else None
    )
    layers = (
        controller.layer_selector.candidate_layers
        if controller.layer_selector is not None
        else ()
    )
    layer_kind = (
        controller.layer_selector.router.kind
        if controller.layer_selector is not None
        else None
    )
    if calibration is None:
        raise FrozenArtifactError("RQ1 base risk calibrator is unsupported")
    return M3FitRecipe(
        cluster_count=controller.vector_bank.cluster_count,
        vector_seed=int(registered["vector_seed"]),
        minimum_class_count=int(registered["minimum_class_count"]),
        vector_source_artifact_sha256=controller.vector_bank.source_artifact_sha256,
        router_kind=controller.vector_router.kind,
        router_seed=int(registered["router_seed"]),
        router_hidden_width=int(registered["router_hidden_width"]),
        router_epochs=int(registered["router_epochs"]),
        distance_temperature=float(registered["distance_temperature"]),
        risk_probe_kind=controller.risk_probe.state.kind,
        risk_hidden_width=int(registered["risk_hidden_width"]),
        risk_epochs=int(registered["risk_epochs"]),
        risk_learning_rate=float(registered["risk_learning_rate"]),
        risk_weight_decay=float(registered["risk_weight_decay"]),
        risk_class_balanced=bool(registered["risk_class_balanced"]),
        risk_seed=int(registered["risk_seed"]),
        calibration_kind=calibration,
        alpha_mode=controller.alpha_controller.mode,
        alpha_max=controller.alpha_controller.alpha_max,
        alpha_beta=controller.alpha_controller.beta,
        alpha_threshold=controller.alpha_controller.threshold,
        fixed_layer=controller.fixed_layer,
        candidate_layers=layers,
        layer_router_kind=layer_kind,
        layer_seed=int(registered["layer_seed"]),
        layer_epochs=int(registered["layer_epochs"]),
    )


def _controller_inputs(
    *, e2_workspace: Path, e2_probe_bundle: Path
) -> tuple[Mapping[FeatureComposition, ProbeDataset], Mapping[Any, Any]]:
    workspace = verify_e2_workspace(e2_workspace)
    bundle = verify_e2_probe_bundle(e2_probe_bundle, workspace=workspace)
    selected = next(
        value
        for task, value in bundle.selected_views.items()
        if task.value == "correct_incorrect_abstention"
    )
    views = controller_input_views(
        selected_layer=selected.layer,
        selected_site=selected.site,
        candidate_layers=workspace.activation_spec.layers,
    )
    plan = json.loads((e2_probe_bundle / "plan.json").read_text(encoding="utf-8"))
    store = verify_activation_store(
        workspace.directory / "activations",
        expected_spec=workspace.activation_spec,
        require_complete=True,
    )
    grid = build_e2_controller_input_datasets(
        workspace,
        views=views,
        split_manifest_digest=plan["split_manifest_digest"],
        prompt_template_sha256=plan["prompt_template_sha256"]["P0-neutral"],
        verified_store=store,
    )
    train = {
        view.composition: grid[(view.composition, "T-controller-train")].probe
        for view in views
    }
    return MappingProxyType(train), grid


def _native_fit_inputs(
    context: _CaptureContext,
    *,
    execution_root: Path,
    execution_private_key: str,
    e2_workspace: Path,
    e2_probe_bundle: Path,
    e5_fit_capture: Path,
    e5_layer_labels: Path,
    controller_questions: Path,
) -> tuple[RQ1CaptureData, ProbeDataset, ProbeDataset, Mapping[str, int]]:
    key = e5_capture_public_key(execution_private_key)
    verified_e5 = verify_e5_fit_capture(
        e5_fit_capture,
        snapshot=context.snapshot,
        questions=context.questions,
        prompts=context.prompts,
        expected_execution_public_key=key,
        require_complete=True,
    )
    train_by_composition, grid = _controller_inputs(
        e2_workspace=e2_workspace, e2_probe_bundle=e2_probe_bundle
    )
    controller_source = tuple(read_questions(controller_questions))
    labels = load_e5_layer_label_data(
        e5_layer_labels,
        questions=controller_source,
        prompt=context.prompts["P0-neutral"],
        controller_datasets=train_by_composition,
        fit_capture=verified_e5,
        fit_capture_artifact_sha256=sha256_path(e5_fit_capture),
        expected_execution_public_key=key,
    )
    capture = load_rq1_capture_data(
        execution_root / "rq1-capture",
        plan=context.store.plan,
        snapshot=context.snapshot,
        questions=context.questions,
        prompt=context.prompts["P0-neutral"],
        expected_execution_public_key=key,
    )
    composition = context.feature_schema.composition
    train = grid[(composition, "T-controller-train")].probe
    calibration = grid[(composition, "T-controller-calibration")].probe
    if (
        train.feature_schema != context.feature_schema
        or train.data_fingerprint
        != context.base_component.controllers[
            "P0-neutral"
        ].risk_probe.training_fingerprint
        or calibration.data_fingerprint
        != context.base_component.controllers[
            "P0-neutral"
        ].risk_probe.calibration_fingerprint
        or tuple(labels.question_ids) != tuple(train.question_ids)
    ):
        raise FrozenArtifactError("RQ1 controller features or layer labels differ from base M3")
    selector = context.base_component.controllers["P0-neutral"].layer_selector
    label_values = (
        labels.best_layers_three
        if selector is not None and len(selector.candidate_layers) == 3
        else labels.best_layers_two
    )
    label_map = (
        MappingProxyType(dict(zip(labels.question_ids, label_values, strict=True)))
        if selector is not None
        else MappingProxyType({})
    )
    return (
        capture,
        train,
        calibration,
        label_map,
    )


def _partition_map(store: RobustnessResultStore) -> Mapping[str, str]:
    return MappingProxyType(
        {
            str(row["question_id"]): str(row["partition"])
            for row in store.plan.body["rq1_generalization"]["assignments"]
        }
    )


def _datasets_for_ids(
    identifiers: Sequence[str],
    *,
    partitions: Mapping[str, str],
    capture: RQ1CaptureData,
    train: ProbeDataset,
    calibration: ProbeDataset,
) -> tuple[Mapping[str, ProbeDataset], Mapping[HookKey, Tensor]]:
    vector_ids = tuple(value for value in identifiers if partitions[value] == "T-steer")
    train_ids = tuple(
        value for value in identifiers if partitions[value] == "T-controller-train"
    )
    calibration_ids = tuple(
        value for value in identifiers if partitions[value] == "T-controller-calibration"
    )
    datasets = MappingProxyType(
        {
            "vector_bank": _subset(capture.vector_dataset, vector_ids),
            "controller_train": _subset(train, train_ids),
            "calibration": _subset(calibration, calibration_ids),
        }
    )
    return datasets, _subset_activations(capture, vector_ids)


def _attestation(
    *,
    store: RobustnessResultStore,
    task: RQ1GeneralizationTask,
    stage: str,
    execution_public_key: str,
    recipe: M3FitRecipe,
    datasets: Mapping[str, ProbeDataset],
    activations: Mapping[HookKey, Tensor],
    best_layers: Sequence[int] | None,
    private_key: str,
) -> Mapping[str, Any]:
    body = rq1_m3_fit_capture_attestation_body(
        plan_digest=store.plan.plan_digest,
        task_id=task.task_id,
        stage=stage,
        execution_public_key=execution_public_key,
        runtime_artifact_sha256=str(
            store.plan.body["m3_capture_runtime_artifact_sha256"]
        ),
        source_question_bundle_sha256=str(
            store.plan.body["source_artifact_sha256"]["triviaqa-development"]
        ),
        recipe=recipe,
        fit_datasets=datasets,
        vector_activations=activations,
        best_layers=best_layers,
    )
    signature = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_key)).sign(
        canonical_json(body).encode()
    )
    return MappingProxyType({"body": body, "signature": signature.hex()})


def _write_component(
    directory: Path,
    *,
    controller: AdaptiveController,
    prompt: PromptSpec,
    model_config: Path,
) -> ConfirmatoryAdaptiveComponent:
    saved = directory / "controller"
    save_adaptive_controller(saved, controller)
    return write_confirmatory_adaptive_component(
        directory / "component",
        model=load_model_spec(model_config),
        prompts={prompt.prompt_id: prompt},
        controllers={prompt.prompt_id: saved},
        controller_source_prompts={prompt.prompt_id: prompt.prompt_id},
    )


def _best_layers(
    identifiers: Sequence[str], labels: Mapping[str, int]
) -> tuple[int, ...] | None:
    if not labels:
        return None
    try:
        return tuple(labels[value] for value in identifiers)
    except KeyError as exc:
        raise FrozenArtifactError(f"RQ1 layer label is missing: {exc.args[0]}") from exc


def _run_one_rq1(
    stage: Path,
    *,
    context: _CaptureContext,
    task: RQ1GeneralizationTask,
    capture: RQ1CaptureData,
    controller_train: ProbeDataset,
    controller_calibration: ProbeDataset,
    labels: Mapping[str, int],
    private_key: str,
    backend: NativeE9VllmBackend,
) -> None:
    store = context.store
    questions = rq1_task_question_sets(store.plan, task)
    held_questions = _rq1_questions_from_plan(store.plan)
    if task.method == "M1":
        base = _frozen_execution_component(store.plan, "M1")
        if base is None:
            raise FrozenArtifactError("RQ1 M1 base component is missing")
        source_scope = write_rq1_scoped_component(
            stage / "source_component",
            plan=store.plan,
            task=task,
            stage="source-fit",
            execution_component=base,
        )
        adapted_scope = write_rq1_scoped_component(
            stage / "adapted_component",
            plan=store.plan,
            task=task,
            stage="held-out-adaptation",
            execution_component=base,
        )
    else:
        base_controller = context.base_component.controllers["P0-neutral"]
        recipe = _fit_recipe(store, base_controller)
        partitions = _partition_map(store)
        source_ids = (*questions["source_fit"], *questions["source_calibration"])
        source_data, source_activations = _datasets_for_ids(
            source_ids,
            partitions=partitions,
            capture=capture,
            train=controller_train,
            calibration=controller_calibration,
        )
        source_labels = _best_layers(
            source_data["controller_train"].question_ids, labels
        )
        source_controller = fit_rq1_m3_controller(
            recipe=recipe,
            fit_datasets=source_data,
            vector_activations=source_activations,
            best_layers=source_labels,
        )
        source_execution = _write_component(
            stage / "source-execution",
            controller=source_controller,
            prompt=context.prompts["P0-neutral"],
            model_config=context.runbook.model_config,
        )
        source_policy = replace(
            context.base_policy,
            controller_artifact_sha256=source_execution.fingerprint,
        )
        source_attestation = _attestation(
            store=store,
            task=task,
            stage="source-fit",
            execution_public_key=str(source_policy.execution_public_key),
            recipe=recipe,
            datasets=source_data,
            activations=source_activations,
            best_layers=source_labels,
            private_key=private_key,
        )
        source_scope = write_rq1_scoped_component(
            stage / "source_component",
            plan=store.plan,
            task=task,
            stage="source-fit",
            execution_component=source_execution.directory,
            adaptive_policy=source_policy,
            calibration_dataset=source_data["calibration"],
            fit_datasets=source_data,
            fit_recipe=recipe,
            vector_activations=source_activations,
            best_layers=source_labels,
            fit_capture_attestation=source_attestation,
        )
        if source_scope.adaptive_policy is None:
            raise FrozenArtifactError("RQ1 source fit lacks its calibrated policy")
        if task.adaptation_regime == "calibration-only":
            adapted_execution = source_execution
            adapted_data = source_data
            adapted_activations = source_activations
            adapted_labels = source_labels
            held_calibration = _subset(
                controller_calibration,
                tuple(
                    value
                    for value in questions["held_out_adaptation"]
                    if partitions[value] == "T-controller-calibration"
                ),
            )
        else:
            adapted_data, adapted_activations = _datasets_for_ids(
                questions["held_out_adaptation"],
                partitions=partitions,
                capture=capture,
                train=controller_train,
                calibration=controller_calibration,
            )
            adapted_labels = None
            held_calibration = adapted_data["calibration"]
            adapted_controller = refit_rq1_m3_vector_bank_controller(
                source_controller=source_controller,
                recipe=recipe,
                fit_datasets=adapted_data,
                vector_activations=adapted_activations,
            )
            adapted_execution = _write_component(
                stage / "adapted-execution",
                controller=adapted_controller,
                prompt=context.prompts["P0-neutral"],
                model_config=context.runbook.model_config,
            )
        adapted_policy = replace(
            source_scope.adaptive_policy,
            controller_artifact_sha256=adapted_execution.fingerprint,
        )
        adapted_attestation = _attestation(
            store=store,
            task=task,
            stage="held-out-adaptation",
            execution_public_key=str(adapted_policy.execution_public_key),
            recipe=recipe,
            datasets=adapted_data,
            activations=adapted_activations,
            best_layers=adapted_labels,
            private_key=private_key,
        )
        adapted_scope = write_rq1_scoped_component(
            stage / "adapted_component",
            plan=store.plan,
            task=task,
            stage="held-out-adaptation",
            execution_component=adapted_execution.directory,
            adaptive_policy=adapted_policy,
            calibration_dataset=held_calibration,
            fit_datasets=adapted_data,
            fit_recipe=recipe,
            vector_activations=adapted_activations,
            best_layers=adapted_labels,
            fit_capture_attestation=adapted_attestation,
        )
    receipt = stage / "fit_receipt"
    write_rq1_fit_receipt(
        receipt,
        plan=store.plan,
        task=task,
        source_component=source_scope.directory,
        adapted_component=adapted_scope.directory,
    )
    question_map = {
        value: held_questions[value] for value in questions["held_out_evaluation"]
    }
    records = stage / "evaluation_records"
    execute_rq1_evaluation_records(
        records,
        store=store,
        task=task,
        questions_by_id=question_map,
        adapted_component=adapted_scope.directory,
        backend=backend,
    )
    append_rq1_generalization_result(
        store,
        task=task,
        questions_by_id=question_map,
        grader_bundle=backend.grader_bundle,
        artifacts={
            "source_component": source_scope.directory,
            "adapted_component": adapted_scope.directory,
            "fit_receipt": receipt,
            "evaluation_records": records,
        },
    )


def run_rq1_generalization_diagnostics(
    directory: str | Path,
    *,
    results: str | Path,
    e9_runbook: str | Path,
    e3_construction: str | Path,
    e2_workspace: str | Path,
    e2_probe_bundle: str | Path,
    e5_fit_capture: str | Path,
    e5_layer_labels: str | Path,
    controller_questions: str | Path,
    execution_private_key: str,
    openrouter_api_key: str,
    limit: int | None = None,
) -> Mapping[str, Any]:
    """Resume a bounded number of complete semantic-fold RQ1 tasks."""

    if limit is not None and (type(limit) is not int or limit <= 0):
        raise DataValidationError("RQ1 robustness limit must be positive")
    normalized = validate_active_study_artifact_paths(
        {
            "robustness execution": directory,
            "E2 workspace": e2_workspace,
            "E2 probe bundle": e2_probe_bundle,
            "E5 fit capture": e5_fit_capture,
            "E5 layer labels": e5_layer_labels,
            "E5 controller questions": controller_questions,
        }
    )
    context = _capture_context(
        results, e9_runbook=e9_runbook, e3_construction=e3_construction
    )
    root = normalized["robustness execution"]
    capture, train, calibration, labels = _native_fit_inputs(
        context,
        execution_root=root,
        execution_private_key=execution_private_key,
        e2_workspace=normalized["E2 workspace"],
        e2_probe_bundle=normalized["E2 probe bundle"],
        e5_fit_capture=normalized["E5 fit capture"],
        e5_layer_labels=normalized["E5 layer labels"],
        controller_questions=normalized["E5 controller questions"],
    )
    pending = tuple(
        task
        for task in iter_rq1_generalization_tasks(context.store.plan)
        if not (context.store.directory / "rq1-results" / task.task_id).exists()
    )
    selected = pending if limit is None else pending[:limit]
    if not selected:
        return MappingProxyType(
            {
                "valid": True,
                "executed": 0,
                "progress": dict(robustness_result_progress(context.store)),
            }
        )
    backend, attestor = _backend(
        context.store,
        context.runbook,
        execution_private_key=execution_private_key,
        openrouter_api_key=openrouter_api_key,
    )
    work = root / "rq1-task-stages"
    work.mkdir(parents=True, exist_ok=True)
    for value in work.glob(".task.stage-*"):
        if value.is_dir() and not value.is_symlink():
            shutil.rmtree(value)
    executed = 0
    try:
        for task in selected:
            stage = Path(tempfile.mkdtemp(prefix=".task.stage-", dir=work))
            try:
                _run_one_rq1(
                    stage,
                    context=context,
                    task=task,
                    capture=capture,
                    controller_train=train,
                    controller_calibration=calibration,
                    labels=labels,
                    private_key=execution_private_key,
                    backend=backend,
                )
                executed += 1
            finally:
                shutil.rmtree(stage, ignore_errors=True)
    finally:
        attestor.runtime.close()
    return MappingProxyType(
        {
            "valid": True,
            "executed": executed,
            "progress": dict(robustness_result_progress(context.store)),
        }
    )

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from mfh.contracts import (
    ActivationSite,
    AdaptivePolicySpec,
    ModelSpec,
    Outcome,
    PromptSpec,
    Runtime,
    TokenScope,
)
from mfh.data.reviewed_splits import validate_reviewed_split_snapshot
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.experiments import confirmatory_components, robustness_results
from mfh.experiments.grader_bundle import verify_e1_grader_bundle
from mfh.experiments.robustness_diagnostics import (
    RobustnessDiagnosticPlan,
    RQ1GeneralizationTask,
)
from mfh.experiments.robustness_results import (
    M3FitRecipe,
    fit_rq1_m3_controller,
    load_rq1_scoped_component,
    refit_rq1_m3_vector_bank_controller,
    robustness_evaluation_condition,
    rq1_m3_fit_capture_attestation_body,
    write_rq1_fit_receipt,
    write_rq1_scoped_component,
)
from mfh.inference.architecture import HookKey
from mfh.methods import static
from mfh.methods.adaptive import (
    AlphaMode,
    RouterKind,
    save_adaptive_controller,
)
from mfh.methods.features import (
    ActivationFeatureSchema,
    ActivationKind,
    FeatureComposition,
)
from mfh.methods.probes import (
    CalibrationKind,
    ProbeDataset,
    ProbeKind,
)
from mfh.provenance import canonical_json, sha256_path, stable_hash
from tests.e4_test_artifacts import build_e3_m1_bundle

_FIT_PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
_FIT_PUBLIC_KEY = _FIT_PRIVATE_KEY.public_key().public_bytes(
    serialization.Encoding.Raw,
    serialization.PublicFormat.Raw,
).hex()


def _allow_tmp(paths: dict[str, str | Path]) -> dict[str, Path]:
    return {name: Path(path).resolve() for name, path in paths.items()}


def _fixed_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str,
    positive: torch.Tensor,
    negative: torch.Tensor,
) -> Path:
    monkeypatch.setattr(static, "validate_active_study_artifact_paths", _allow_tmp)
    monkeypatch.setattr(
        confirmatory_components,
        "validate_active_study_artifact_paths",
        _allow_tmp,
    )
    raw_direction = (positive.mean(dim=0) - negative.mean(dim=0)).tolist()
    bank_root = tmp_path / f"{name}-bank"
    bank_root.mkdir()
    bank = build_e3_m1_bundle(
        bank_root,
        direction=tuple(float(value) for value in raw_direction),
        reference_rms_value=1.0,
        layers=(16,),
    )
    component = tmp_path / f"{name}-component"
    confirmatory_components.write_confirmatory_fixed_component(
        component,
        source_artifact=bank,
        method="M1",
        layer=16,
        site=ActivationSite.POST_MLP,
        token_scope=TokenScope.FIRST_FOUR,
        standardized_alpha=1.0,
        reference_rms=1.0,
    )
    return component


def _plan(
    tmp_path: Path,
    *,
    regime: str,
    base_component_sha256: str,
    method: str = "M1",
    assignments: list[dict[str, object]] | None = None,
    base_component: Path | None = None,
    base_policy: AdaptivePolicySpec | None = None,
) -> tuple[RobustnessDiagnosticPlan, RQ1GeneralizationTask]:
    task_body = {
        "held_out_fold": 0,
        "training_prompt_id": "P0-neutral",
        "evaluation_prompt_id": "P2-calibrated-abstention",
        "method": method,
        "adaptation_regime": regime,
    }
    task = RQ1GeneralizationTask(
        task_id=f"rq1-{stable_hash(task_body)}",
        held_out_fold=0,
        training_prompt_id="P0-neutral",
        evaluation_prompt_id="P2-calibrated-abstention",
        method=method,
        adaptation_regime=regime,
    )
    default_assignments: list[dict[str, object]] = [
        {
            "question_id": "source-fit",
            "question_fingerprint": "1" * 64,
            "source_partition": "T-steer",
            "partition": "T-steer",
            "semantic_group_id": "1" * 64,
            "semantic_fold": 1,
        },
        {
            "question_id": "source-calibration",
            "question_fingerprint": "2" * 64,
            "source_partition": "T-controller",
            "partition": "T-controller-calibration",
            "semantic_group_id": "2" * 64,
            "semantic_fold": 1,
        },
        {
            "question_id": "held-adaptation",
            "question_fingerprint": "3" * 64,
            "source_partition": "T-controller",
            "partition": "T-controller-calibration",
            "semantic_group_id": "3" * 64,
            "semantic_fold": 0,
        },
        {
            "question_id": "held-evaluation",
            "question_fingerprint": "4" * 64,
            "source_partition": "T-dev",
            "partition": "T-dev",
            "semantic_group_id": "4" * 64,
            "semantic_fold": 0,
        },
    ]
    body = {
        "source_artifact_sha256": {
            "frozen-component-selection": "5" * 64,
            "frozen-evaluation-scripts": "6" * 64,
            "frozen-graders": "7" * 64,
            "triviaqa-development": "b" * 64,
        },
        "m3_capture_runtime_artifact_sha256": "a" * 64,
        "rq1_generalization": {
            "assignments": assignments or default_assignments,
            "full_relearning_subdivision_algorithm": (
                "semantic-fold-preserve-preregistered-partitions-v1"
            ),
            "m3_refit_hyperparameters": {
                "vector_seed": 17,
                "minimum_class_count": 1,
                "router_seed": 17,
                "router_hidden_width": 8,
                "router_epochs": 5,
                "distance_temperature": 1.0,
                "risk_hidden_width": 8,
                "risk_epochs": 5,
                "risk_learning_rate": 0.03,
                "risk_weight_decay": 0.0001,
                "risk_class_balanced": True,
                "risk_seed": 17,
                "calibration_kind": "temperature",
                "layer_seed": 17,
                "layer_epochs": 5,
            },
            "tasks": [
                {
                    **task_body,
                    "task_id": task.task_id,
                }
            ],
        },
    }
    plan_path = tmp_path / f"plan-{regime}"
    selection = plan_path / "sources" / "frozen-component-selection"
    selection.mkdir(parents=True)
    manifest_body = {
        "schema_version": 3,
        "study_protocol_digest": "8" * 64,
        "phase": "E9",
        "components": [
            {
                "model_name": "qwen3.6-27b-mlx-4bit",
                "method": method,
                "artifact_sha256": base_component_sha256,
                "component_path": "components/base",
                "adaptive_policy": (
                    base_policy.to_dict() if base_policy is not None else None
                ),
                "adaptive_policy_digest": (
                    stable_hash(base_policy.to_dict()) if base_policy is not None else None
                ),
            }
        ],
    }
    (selection / "manifest.json").write_text(
        json.dumps({**manifest_body, "manifest_digest": stable_hash(manifest_body)}),
        encoding="utf-8",
    )
    if base_component is not None:
        (selection / "components" / "base").mkdir(parents=True)
        shutil.copytree(base_component, selection / "components" / "base" / "artifact")
    return RobustnessDiagnosticPlan(plan_path, body, stable_hash(body)), task


def _m3_assignments() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    specifications = (
        ("source-vector", "T-steer", 1, 4),
        ("source-controller", "T-controller-train", 1, 4),
        ("source-calibration", "T-controller-calibration", 1, 3),
        ("held-vector", "T-steer", 0, 4),
        ("held-controller", "T-controller-train", 0, 4),
        ("held-calibration", "T-controller-calibration", 0, 3),
        ("held-evaluation", "T-dev", 0, 2),
    )
    for prefix, partition, fold, count in specifications:
        for index in range(count):
            question_id = f"{prefix}-{index}"
            rows.append(
                {
                    "question_id": question_id,
                    "question_fingerprint": stable_hash(question_id),
                    "source_partition": partition,
                    "partition": partition,
                    "semantic_group_id": stable_hash(f"group:{question_id}"),
                    "semantic_fold": fold,
                }
            )
    return rows


def _m3_schema(partition: str) -> ActivationFeatureSchema:
    prompt_sha = hashlib.sha256(b"Answer the question.").hexdigest()
    return ActivationFeatureSchema(
        benchmark="triviaqa",
        partition=partition,
        split_manifest_digest="d" * 64,
        model_repository="mlx-community/Qwen3.6-27B-4bit",
        model_revision="c000ac2c2057d94be3fa931000c31723aac53282",
        runtime=Runtime.MLX,
        quantization="affine-g64-mlx-4bit",
        prompt_id="P0-neutral",
        prompt_sha256=prompt_sha,
        activation_kind=ActivationKind.FINAL_PROMPT,
        layers=(16,),
        sites=(ActivationSite.POST_MLP,),
        composition=FeatureComposition.SINGLE_LAYER,
        width=2,
    )


def _m3_dataset(
    question_ids: tuple[str, ...],
    *,
    partition: str,
    offset: float,
) -> ProbeDataset:
    patterns: tuple[tuple[list[float], Outcome], ...]
    if partition == "T-steer":
        patterns = (
            ([-2.0 + offset, -2.0], Outcome.CORRECT),
            ([2.0 + offset, 2.0], Outcome.INCORRECT),
        )
    else:
        patterns = (
            ([-2.0 + offset, -2.0], Outcome.CORRECT),
            ([2.0 + offset, 2.0], Outcome.INCORRECT),
            ([-2.0 + offset, 2.0], Outcome.ABSTENTION),
        )
    if len(question_ids) == 2 and partition != "T-steer":
        patterns = (
            ([2.0 + offset, 2.0], Outcome.INCORRECT),
            ([-2.0 + offset, 2.0], Outcome.ABSTENTION),
        )
    return ProbeDataset(
        question_ids=question_ids,
        features=torch.tensor(
            [patterns[index % len(patterns)][0] for index in range(len(question_ids))]
        ),
        outcomes=tuple(patterns[index % len(patterns)][1] for index in range(len(question_ids))),
        group_ids=tuple(f"group:{value}" for value in question_ids),
        feature_schema=_m3_schema(partition),
    )


def _m3_recipe() -> M3FitRecipe:
    return M3FitRecipe(
        cluster_count=1,
        vector_seed=17,
        minimum_class_count=1,
        vector_source_artifact_sha256=None,
        router_kind=RouterKind.NEAREST_CENTROID,
        router_seed=17,
        router_hidden_width=8,
        router_epochs=5,
        distance_temperature=1.0,
        risk_probe_kind=ProbeKind.LOGISTIC,
        risk_hidden_width=8,
        risk_epochs=5,
        risk_learning_rate=0.03,
        risk_weight_decay=0.0001,
        risk_class_balanced=True,
        risk_seed=17,
        calibration_kind=CalibrationKind.TEMPERATURE,
        alpha_mode=AlphaMode.FIXED,
        alpha_max=0.5,
        alpha_beta=12.0,
        alpha_threshold=0.5,
        fixed_layer=16,
        candidate_layers=(),
        layer_router_kind=None,
        layer_seed=17,
        layer_epochs=5,
    )


def _m3_vector_activations(
    dataset: ProbeDataset, *, direction_sign: float
) -> dict[HookKey, torch.Tensor]:
    values = torch.tensor(
        [
            [direction_sign, 0.0]
            if outcome is Outcome.CORRECT
            else [-direction_sign, 0.0]
            for outcome in dataset.outcomes
        ]
    )
    return {HookKey(16, ActivationSite.POST_MLP): values}


def _m3_attestation(
    *,
    plan: RobustnessDiagnosticPlan,
    task: RQ1GeneralizationTask,
    stage: str,
    recipe: M3FitRecipe,
    fit_datasets: dict[str, ProbeDataset],
    vector_activations: dict[HookKey, torch.Tensor],
    private_key: Ed25519PrivateKey = _FIT_PRIVATE_KEY,
    public_key: str = _FIT_PUBLIC_KEY,
    runtime_artifact_sha256: str = "a" * 64,
    source_question_bundle_sha256: str = "b" * 64,
) -> dict[str, object]:
    body = rq1_m3_fit_capture_attestation_body(
        plan_digest=plan.plan_digest,
        task_id=task.task_id,
        stage=stage,
        execution_public_key=public_key,
        runtime_artifact_sha256=runtime_artifact_sha256,
        source_question_bundle_sha256=source_question_bundle_sha256,
        recipe=recipe,
        fit_datasets=fit_datasets,
        vector_activations=vector_activations,
    )
    return {
        "body": body,
        "signature": private_key.sign(canonical_json(body).encode()).hex(),
    }


def _m3_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str,
    fit_datasets: dict[str, ProbeDataset],
    direction_sign: float,
    recipe: M3FitRecipe,
    frozen_component: Path | None = None,
) -> tuple[Path, dict[HookKey, torch.Tensor]]:
    monkeypatch.setattr(
        confirmatory_components,
        "validate_active_study_artifact_paths",
        _allow_tmp,
    )
    monkeypatch.setattr(static, "validate_active_study_artifact_paths", _allow_tmp)
    vector_activations = _m3_vector_activations(
        fit_datasets["vector_bank"], direction_sign=direction_sign
    )
    controller = (
        fit_rq1_m3_controller(
            recipe=recipe,
            fit_datasets=fit_datasets,
            vector_activations=vector_activations,
        )
        if frozen_component is None
        else refit_rq1_m3_vector_bank_controller(
            source_controller=confirmatory_components.load_confirmatory_adaptive_component(
                frozen_component
            ).controllers["P0-neutral"],
            recipe=recipe,
            fit_datasets=fit_datasets,
            vector_activations=vector_activations,
        )
    )
    controller_path = tmp_path / f"{name}-controller"
    save_adaptive_controller(controller_path, controller)
    prompt = PromptSpec("P0-neutral", "Answer the question.")
    component = tmp_path / f"{name}-component"
    confirmatory_components.write_confirmatory_adaptive_component(
        component,
        model=ModelSpec(
            name="qwen3.6-27b-mlx-4bit",
            repository="mlx-community/Qwen3.6-27B-4bit",
            revision="c000ac2c2057d94be3fa931000c31723aac53282",
            runtime=Runtime.MLX,
            quantization="affine-g64-mlx-4bit",
            num_layers=64,
        ),
        prompts={prompt.prompt_id: prompt},
        controllers={prompt.prompt_id: controller_path},
    )
    return component, vector_activations


def _m3_policy(component: Path) -> AdaptivePolicySpec:
    return AdaptivePolicySpec(
        schema_version=2,
        release_risk_threshold=0.2,
        abstention_probability_threshold=0.8,
        alpha_max=0.5,
        alpha_beta=12.0,
        layer=None,
        site=None,
        token_scope=None,
        direction_sha256=None,
        direction_norm=None,
        execution_public_key=_FIT_PUBLIC_KEY,
        controller_artifact_sha256=sha256_path(component),
        candidate_layers=(16,),
        candidate_sites=(ActivationSite.POST_MLP,),
        candidate_token_scopes=(TokenScope.FIRST_FOUR,),
        vector_count=1,
        likely_unknown_risk_threshold=0.9,
        alpha_mode="fixed",
        alpha_risk_threshold=0.5,
    )


def test_scoped_components_write_exact_fit_receipt_and_adapted_condition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        robustness_results,
        "validate_active_study_artifact_paths",
        _allow_tmp,
    )
    execution = _fixed_component(
        tmp_path,
        monkeypatch,
        name="source",
        positive=torch.tensor([[1.0, 0.0]]),
        negative=torch.tensor([[0.0, 1.0]]),
    )
    plan, task = _plan(
        tmp_path,
        regime="source-frozen-control",
        base_component_sha256=sha256_path(execution),
    )
    source = write_rq1_scoped_component(
        tmp_path / "source-scope",
        plan=plan,
        task=task,
        stage="source-fit",
        execution_component=execution,
    )
    adapted = write_rq1_scoped_component(
        tmp_path / "adapted-scope",
        plan=plan,
        task=task,
        stage="held-out-adaptation",
        execution_component=execution,
    )
    receipt_path = tmp_path / "fit-receipt.json"
    write_rq1_fit_receipt(
        receipt_path,
        plan=plan,
        task=task,
        source_component=source.directory,
        adapted_component=adapted.directory,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["question_sets"] == {
        "source_fit": ["source-fit"],
        "source_calibration": ["source-calibration"],
        "held_out_adaptation": ["held-adaptation"],
        "held_out_evaluation": ["held-evaluation"],
    }
    assert receipt["held_out_evaluation_used_for_fitting"] is False
    condition = robustness_evaluation_condition(
        plan,
        benchmark="triviaqa",
        partition="T-dev",
        prompt_id="P2-calibrated-abstention",
        prompt_text="Answer only when confident; otherwise say I don't know.",
        method="M1",
        seed=17,
        execution_component=adapted.execution_component,
    )
    assert condition.method_artifact_sha256 == adapted.execution_component_sha256
    assert condition.method_artifact_sha256 == sha256_path(execution)

    alternate = _fixed_component(
        tmp_path,
        monkeypatch,
        name="repacked-alternate",
        positive=torch.tensor([[2.0, 0.0]]),
        negative=torch.tensor([[0.0, -1.0]]),
    )
    repacked = tmp_path / "repacked-m1-scope"
    shutil.copytree(adapted.directory, repacked)
    shutil.rmtree(repacked / "execution-component")
    shutil.copytree(alternate, repacked / "execution-component")
    repacked_manifest_path = repacked / "scope-manifest.json"
    repacked_manifest = json.loads(repacked_manifest_path.read_text(encoding="utf-8"))
    repacked_manifest.pop("manifest_digest")
    repacked_manifest["execution_component_sha256"] = sha256_path(
        repacked / "execution-component"
    )
    repacked_manifest_path.write_text(
        canonical_json(
            {
                **repacked_manifest,
                "manifest_digest": stable_hash(repacked_manifest),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(FrozenArtifactError, match="exact frozen base bytes"):
        load_rq1_scoped_component(
            repacked,
            plan=plan,
            task=task,
            stage="held-out-adaptation",
        )

    field = adapted.directory / "fields" / "vector_bank.json"
    field.write_text('{"forged":true}\n', encoding="utf-8")
    with pytest.raises(FrozenArtifactError, match="field does not replay"):
        load_rq1_scoped_component(
            adapted.directory,
            plan=plan,
            task=task,
            stage="held-out-adaptation",
        )


def test_calibration_only_rejects_a_relearned_execution_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        robustness_results,
        "validate_active_study_artifact_paths",
        _allow_tmp,
    )
    source_execution = _fixed_component(
        tmp_path,
        monkeypatch,
        name="source",
        positive=torch.tensor([[1.0, 0.0]]),
        negative=torch.tensor([[0.0, 1.0]]),
    )
    changed_execution = _fixed_component(
        tmp_path,
        monkeypatch,
        name="changed",
        positive=torch.tensor([[1.0, 0.0]]),
        negative=torch.tensor([[-1.0, 0.0]]),
    )
    plan, task = _plan(
        tmp_path,
        regime="source-frozen-control",
        base_component_sha256=sha256_path(source_execution),
    )
    write_rq1_scoped_component(
        tmp_path / "source-scope",
        plan=plan,
        task=task,
        stage="source-fit",
        execution_component=source_execution,
    )
    with pytest.raises(DataValidationError, match="source-frozen control"):
        write_rq1_scoped_component(
            tmp_path / "adapted-scope",
            plan=plan,
            task=task,
            stage="held-out-adaptation",
            execution_component=changed_execution,
        )


def test_full_m3_relearning_replays_fit_evidence_and_structural_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        robustness_results,
        "validate_active_study_artifact_paths",
        _allow_tmp,
    )
    assignments = _m3_assignments()
    source_fit = {
        "vector_bank": _m3_dataset(
            tuple(f"source-vector-{index}" for index in range(4)),
            partition="T-steer",
            offset=0.0,
        ),
        "controller_train": _m3_dataset(
            tuple(f"source-controller-{index}" for index in range(4)),
            partition="T-controller-train",
            offset=0.0,
        ),
        "calibration": _m3_dataset(
            tuple(f"source-calibration-{index}" for index in range(3)),
            partition="T-controller-calibration",
            offset=0.0,
        ),
    }
    recipe = _m3_recipe()
    base_fit = {
        name: _m3_dataset(
            dataset.question_ids,
            partition=str(dataset.feature_schema.partition),
            offset=1.0,
        )
        for name, dataset in source_fit.items()
        if dataset.feature_schema is not None
    }
    base_execution, _ = _m3_component(
        tmp_path,
        monkeypatch,
        name="global-base",
        fit_datasets=base_fit,
        direction_sign=1.0,
        recipe=recipe,
    )
    source_execution, source_activations = _m3_component(
        tmp_path,
        monkeypatch,
        name="source",
        fit_datasets=source_fit,
        direction_sign=1.0,
        recipe=recipe,
    )
    plan, task = _plan(
        tmp_path,
        regime="full-vector-bank-relearning",
        method="M3",
        base_component_sha256=sha256_path(base_execution),
        assignments=assignments,
        base_component=base_execution,
        base_policy=_m3_policy(base_execution),
    )
    source_attestation = _m3_attestation(
        plan=plan,
        task=task,
        stage="source-fit",
        recipe=recipe,
        fit_datasets=source_fit,
        vector_activations=source_activations,
    )
    source = write_rq1_scoped_component(
        tmp_path / "source-scope",
        plan=plan,
        task=task,
        stage="source-fit",
        execution_component=source_execution,
        adaptive_policy=_m3_policy(source_execution),
        calibration_dataset=source_fit["calibration"],
        fit_datasets=source_fit,
        fit_recipe=recipe,
        vector_activations=source_activations,
        fit_capture_attestation=source_attestation,
    )
    assert source.execution_component_sha256 != sha256_path(base_execution)
    for name, changed_value in (
        ("runtime_artifact_sha256", "c" * 64),
        ("source_question_bundle_sha256", "d" * 64),
    ):
        attestation_arguments: dict[str, object] = {name: changed_value}
        with pytest.raises(DataValidationError, match="capture body differs"):
            write_rq1_scoped_component(
                tmp_path / f"wrong-capture-source-{name}",
                plan=plan,
                task=task,
                stage="source-fit",
                execution_component=source_execution,
                adaptive_policy=_m3_policy(source_execution),
                calibration_dataset=source_fit["calibration"],
                fit_datasets=source_fit,
                fit_recipe=recipe,
                vector_activations=source_activations,
                fit_capture_attestation=_m3_attestation(
                    plan=plan,
                    task=task,
                    stage="source-fit",
                    recipe=recipe,
                    fit_datasets=source_fit,
                    vector_activations=source_activations,
                    **attestation_arguments,
                ),
            )
    changed_source_recipe = replace(
        recipe,
        vector_source_artifact_sha256="e" * 64,
    )
    changed_source_execution, changed_source_activations = _m3_component(
        tmp_path,
        monkeypatch,
        name="changed-vector-source",
        fit_datasets=source_fit,
        direction_sign=1.0,
        recipe=changed_source_recipe,
    )
    with pytest.raises(DataValidationError, match="frozen base architecture or recipe"):
        write_rq1_scoped_component(
            tmp_path / "changed-vector-source-scope",
            plan=plan,
            task=task,
            stage="source-fit",
            execution_component=changed_source_execution,
            adaptive_policy=_m3_policy(changed_source_execution),
            calibration_dataset=source_fit["calibration"],
            fit_datasets=source_fit,
            fit_recipe=changed_source_recipe,
            vector_activations=changed_source_activations,
            fit_capture_attestation=_m3_attestation(
                plan=plan,
                task=task,
                stage="source-fit",
                recipe=changed_source_recipe,
                fit_datasets=source_fit,
                vector_activations=changed_source_activations,
            ),
        )
    fresh_private_key = Ed25519PrivateKey.generate()
    fresh_public_key = fresh_private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    with pytest.raises(DataValidationError, match="frozen base architecture or recipe"):
        write_rq1_scoped_component(
            tmp_path / "self-signed-source-scope",
            plan=plan,
            task=task,
            stage="source-fit",
            execution_component=source_execution,
            adaptive_policy=replace(
                _m3_policy(source_execution),
                execution_public_key=fresh_public_key,
            ),
            calibration_dataset=source_fit["calibration"],
            fit_datasets=source_fit,
            fit_recipe=recipe,
            vector_activations=source_activations,
            fit_capture_attestation=_m3_attestation(
                plan=plan,
                task=task,
                stage="source-fit",
                recipe=recipe,
                fit_datasets=source_fit,
                vector_activations=source_activations,
                private_key=fresh_private_key,
                public_key=fresh_public_key,
            ),
        )
    with pytest.raises(DataValidationError, match="frozen base architecture or recipe"):
        write_rq1_scoped_component(
            tmp_path / "changed-likely-unknown-scope",
            plan=plan,
            task=task,
            stage="source-fit",
            execution_component=source_execution,
            adaptive_policy=replace(
                _m3_policy(source_execution),
                likely_unknown_risk_threshold=0.97,
            ),
            calibration_dataset=source_fit["calibration"],
            fit_datasets=source_fit,
            fit_recipe=recipe,
            vector_activations=source_activations,
            fit_capture_attestation=source_attestation,
        )
    with pytest.raises(DataValidationError, match="geometry differs"):
        write_rq1_scoped_component(
            tmp_path / "forged-geometry-scope",
            plan=plan,
            task=task,
            stage="source-fit",
            execution_component=source_execution,
            adaptive_policy=replace(
                _m3_policy(source_execution),
                candidate_layers=(7,),
                candidate_sites=(ActivationSite.POST_ATTENTION,),
                candidate_token_scopes=(TokenScope.ALL_GENERATED,),
                vector_count=4,
            ),
            calibration_dataset=source_fit["calibration"],
            fit_datasets=source_fit,
            fit_recipe=recipe,
            vector_activations=source_activations,
            fit_capture_attestation=source_attestation,
        )
    held_ids = robustness_results._full_relearning_fit_ids(plan, task)
    adapted_fit = {
        "vector_bank": _m3_dataset(
            held_ids["vector_bank"], partition="T-steer", offset=0.25
        ),
        "controller_train": _m3_dataset(
            held_ids["controller_train"],
            partition="T-controller-train",
            offset=0.25,
        ),
        "calibration": _m3_dataset(
            held_ids["calibration"],
            partition="T-controller-calibration",
            offset=0.25,
        ),
    }
    adapted_execution, adapted_activations = _m3_component(
        tmp_path,
        monkeypatch,
        name="adapted",
        fit_datasets=adapted_fit,
        direction_sign=-1.0,
        recipe=recipe,
        frozen_component=source_execution,
    )
    adapted_attestation = _m3_attestation(
        plan=plan,
        task=task,
        stage="held-out-adaptation",
        recipe=recipe,
        fit_datasets=adapted_fit,
        vector_activations=adapted_activations,
    )
    adapted = write_rq1_scoped_component(
        tmp_path / "adapted-scope",
        plan=plan,
        task=task,
        stage="held-out-adaptation",
        execution_component=adapted_execution,
        adaptive_policy=_m3_policy(adapted_execution),
        calibration_dataset=adapted_fit["calibration"],
        fit_datasets=adapted_fit,
        fit_recipe=recipe,
        vector_activations=adapted_activations,
        fit_capture_attestation=adapted_attestation,
    )
    changed_recipe = replace(recipe, distance_temperature=2.0)
    changed_recipe_execution, changed_recipe_activations = _m3_component(
        tmp_path,
        monkeypatch,
        name="changed-held-recipe",
        fit_datasets=adapted_fit,
        direction_sign=-1.0,
        recipe=changed_recipe,
        frozen_component=source_execution,
    )
    with pytest.raises(DataValidationError, match="frozen base architecture or recipe"):
        write_rq1_scoped_component(
            tmp_path / "changed-held-recipe-scope",
            plan=plan,
            task=task,
            stage="held-out-adaptation",
            execution_component=changed_recipe_execution,
            adaptive_policy=_m3_policy(changed_recipe_execution),
            calibration_dataset=adapted_fit["calibration"],
            fit_datasets=adapted_fit,
            fit_recipe=changed_recipe,
            vector_activations=changed_recipe_activations,
            fit_capture_attestation=_m3_attestation(
                plan=plan,
                task=task,
                stage="held-out-adaptation",
                recipe=changed_recipe,
                fit_datasets=adapted_fit,
                vector_activations=changed_recipe_activations,
            ),
        )
    write_rq1_fit_receipt(
        tmp_path / "m3-fit-receipt.json",
        plan=plan,
        task=task,
        source_component=source.directory,
        adapted_component=adapted.directory,
    )
    assert source.execution_component_sha256 != adapted.execution_component_sha256
    assert (
        source.field_fingerprints["router_architecture"]
        == adapted.field_fingerprints["router_architecture"]
    )
    assert source.field_fingerprints["vector_bank"] != adapted.field_fingerprints["vector_bank"]
    assert source.field_fingerprints["directions"] != adapted.field_fingerprints["directions"]
    assert source.field_fingerprints["risk_probe"] == adapted.field_fingerprints["risk_probe"]
    assert (
        source.field_fingerprints["layer_selector"]
        == adapted.field_fingerprints["layer_selector"]
    )

    unfrozen_execution, unfrozen_activations = _m3_component(
        tmp_path,
        monkeypatch,
        name="unfrozen-held-risk-probe",
        fit_datasets=adapted_fit,
        direction_sign=-1.0,
        recipe=recipe,
    )
    unfrozen = write_rq1_scoped_component(
        tmp_path / "unfrozen-held-risk-probe-scope",
        plan=plan,
        task=task,
        stage="held-out-adaptation",
        execution_component=unfrozen_execution,
        adaptive_policy=_m3_policy(unfrozen_execution),
        calibration_dataset=adapted_fit["calibration"],
        fit_datasets=adapted_fit,
        fit_recipe=recipe,
        vector_activations=unfrozen_activations,
        fit_capture_attestation=_m3_attestation(
            plan=plan,
            task=task,
            stage="held-out-adaptation",
            recipe=recipe,
            fit_datasets=adapted_fit,
            vector_activations=unfrozen_activations,
        ),
    )
    with pytest.raises(DataValidationError, match="outside its regime"):
        write_rq1_fit_receipt(
            tmp_path / "unfrozen-risk-fit-receipt.json",
            plan=plan,
            task=task,
            source_component=source.directory,
            adapted_component=unfrozen.directory,
        )

    with pytest.raises(DataValidationError, match="executable alpha controller"):
        write_rq1_scoped_component(
            tmp_path / "changed-alpha-scope",
            plan=plan,
            task=task,
            stage="held-out-adaptation",
            execution_component=adapted_execution,
            adaptive_policy=replace(_m3_policy(adapted_execution), alpha_max=0.6),
            calibration_dataset=adapted_fit["calibration"],
            fit_datasets=adapted_fit,
            fit_recipe=recipe,
            vector_activations=adapted_activations,
            fit_capture_attestation=adapted_attestation,
        )

    forged_activations = {
        key: value + torch.tensor([[0.25, 0.0]] * value.shape[0])
        for key, value in adapted_activations.items()
    }
    with pytest.raises(DataValidationError, match="capture body differs"):
        write_rq1_scoped_component(
            tmp_path / "forged-capture-scope",
            plan=plan,
            task=task,
            stage="held-out-adaptation",
            execution_component=adapted_execution,
            adaptive_policy=_m3_policy(adapted_execution),
            calibration_dataset=adapted_fit["calibration"],
            fit_datasets=adapted_fit,
            fit_recipe=recipe,
            vector_activations=forged_activations,
            fit_capture_attestation=adapted_attestation,
        )

    changed_execution, _ = _m3_component(
        tmp_path,
        monkeypatch,
        name="changed-weights",
        fit_datasets=adapted_fit,
        direction_sign=-1.0,
        recipe=replace(recipe, distance_temperature=2.0),
        frozen_component=source_execution,
    )
    with pytest.raises(DataValidationError, match="parameters do not replay"):
        write_rq1_scoped_component(
            tmp_path / "changed-weights-scope",
            plan=plan,
            task=task,
            stage="held-out-adaptation",
            execution_component=changed_execution,
            adaptive_policy=_m3_policy(changed_execution),
            calibration_dataset=adapted_fit["calibration"],
            fit_datasets=adapted_fit,
            fit_recipe=recipe,
            vector_activations=adapted_activations,
            fit_capture_attestation=adapted_attestation,
        )


def test_calibration_only_reuses_fold_refit_and_changes_only_threshold_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        robustness_results,
        "validate_active_study_artifact_paths",
        _allow_tmp,
    )
    assignments = _m3_assignments()
    source_fit = {
        "vector_bank": _m3_dataset(
            tuple(f"source-vector-{index}" for index in range(4)),
            partition="T-steer",
            offset=0.0,
        ),
        "controller_train": _m3_dataset(
            tuple(f"source-controller-{index}" for index in range(4)),
            partition="T-controller-train",
            offset=0.0,
        ),
        "calibration": _m3_dataset(
            tuple(f"source-calibration-{index}" for index in range(3)),
            partition="T-controller-calibration",
            offset=0.0,
        ),
    }
    recipe = _m3_recipe()
    base_fit = {
        name: _m3_dataset(
            dataset.question_ids,
            partition=str(dataset.feature_schema.partition),
            offset=1.0,
        )
        for name, dataset in source_fit.items()
        if dataset.feature_schema is not None
    }
    base_execution, _ = _m3_component(
        tmp_path,
        monkeypatch,
        name="calibration-global-base",
        fit_datasets=base_fit,
        direction_sign=1.0,
        recipe=recipe,
    )
    source_execution, source_activations = _m3_component(
        tmp_path,
        monkeypatch,
        name="calibration-source",
        fit_datasets=source_fit,
        direction_sign=1.0,
        recipe=recipe,
    )
    plan, task = _plan(
        tmp_path,
        regime="calibration-only",
        method="M3",
        base_component_sha256=sha256_path(base_execution),
        assignments=assignments,
        base_component=base_execution,
        base_policy=_m3_policy(base_execution),
    )
    source = write_rq1_scoped_component(
        tmp_path / "calibration-source-scope",
        plan=plan,
        task=task,
        stage="source-fit",
        execution_component=source_execution,
        adaptive_policy=_m3_policy(source_execution),
        calibration_dataset=source_fit["calibration"],
        fit_datasets=source_fit,
        fit_recipe=recipe,
        vector_activations=source_activations,
        fit_capture_attestation=_m3_attestation(
            plan=plan,
            task=task,
            stage="source-fit",
            recipe=recipe,
            fit_datasets=source_fit,
            vector_activations=source_activations,
        ),
    )
    held_calibration_ids = robustness_results._expected_threshold_fit_ids(
        plan,
        task,
        "held-out-adaptation",
    )
    held_calibration = _m3_dataset(
        held_calibration_ids,
        partition="T-controller-calibration",
        offset=0.75,
    )
    adapted = write_rq1_scoped_component(
        tmp_path / "calibration-adapted-scope",
        plan=plan,
        task=task,
        stage="held-out-adaptation",
        execution_component=source_execution,
        adaptive_policy=_m3_policy(source_execution),
        calibration_dataset=held_calibration,
        fit_datasets=source_fit,
        fit_recipe=recipe,
        vector_activations=source_activations,
        fit_capture_attestation=_m3_attestation(
            plan=plan,
            task=task,
            stage="held-out-adaptation",
            recipe=recipe,
            fit_datasets=source_fit,
            vector_activations=source_activations,
        ),
    )
    write_rq1_fit_receipt(
        tmp_path / "calibration-only-fit-receipt.json",
        plan=plan,
        task=task,
        source_component=source.directory,
        adapted_component=adapted.directory,
    )
    assert source.execution_component_sha256 != sha256_path(base_execution)
    assert source.execution_component_sha256 == adapted.execution_component_sha256
    assert source.field_fingerprints["vector_bank"] == adapted.field_fingerprints["vector_bank"]
    assert source.field_fingerprints["router"] == adapted.field_fingerprints["router"]
    manifest = json.loads(
        (adapted.directory / "scope-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["fit_question_ids"] == list(held_calibration_ids)


def test_m3_scoped_component_rejects_self_asserted_fit_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        robustness_results,
        "validate_active_study_artifact_paths",
        _allow_tmp,
    )
    source_fit = {
        "vector_bank": _m3_dataset(
            tuple(f"source-vector-{index}" for index in range(4)),
            partition="T-steer",
            offset=0.0,
        ),
        "controller_train": _m3_dataset(
            tuple(f"source-controller-{index}" for index in range(4)),
            partition="T-controller-train",
            offset=0.0,
        ),
        "calibration": _m3_dataset(
            tuple(f"source-calibration-{index}" for index in range(3)),
            partition="T-controller-calibration",
            offset=0.0,
        ),
    }
    recipe = _m3_recipe()
    source_execution, source_activations = _m3_component(
        tmp_path,
        monkeypatch,
        name="source",
        fit_datasets=source_fit,
        direction_sign=1.0,
        recipe=recipe,
    )
    plan, task = _plan(
        tmp_path,
        regime="calibration-only",
        method="M3",
        base_component_sha256=sha256_path(source_execution),
        assignments=_m3_assignments(),
        base_component=source_execution,
        base_policy=_m3_policy(source_execution),
    )
    forged = dict(source_fit)
    forged["vector_bank"] = _m3_dataset(
        tuple(f"forged-{index}" for index in range(4)),
        partition="T-steer",
        offset=0.0,
    )
    with pytest.raises(DataValidationError, match="exact semantic-fold inputs"):
        write_rq1_scoped_component(
            tmp_path / "forged-scope",
            plan=plan,
            task=task,
            stage="source-fit",
            execution_component=source_execution,
            adaptive_policy=_m3_policy(source_execution),
            calibration_dataset=source_fit["calibration"],
            fit_datasets=forged,
            fit_recipe=recipe,
            vector_activations=source_activations,
            fit_capture_attestation=_m3_attestation(
                plan=plan,
                task=task,
                stage="source-fit",
                recipe=recipe,
                fit_datasets=source_fit,
                vector_activations=source_activations,
            ),
        )


def test_strict_artifact_rejects_lexical_symlink_ancestors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        robustness_results,
        "validate_active_study_artifact_paths",
        _allow_tmp,
    )
    execution = _fixed_component(
        tmp_path,
        monkeypatch,
        name="source",
        positive=torch.tensor([[1.0, 0.0]]),
        negative=torch.tensor([[0.0, 1.0]]),
    )
    plan, task = _plan(
        tmp_path,
        regime="source-frozen-control",
        base_component_sha256=sha256_path(execution),
    )
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(execution.parent, target_is_directory=True)
    with pytest.raises(FrozenArtifactError, match="symlink"):
        write_rq1_scoped_component(
            tmp_path / "scope",
            plan=plan,
            task=task,
            stage="source-fit",
            execution_component=linked_parent / execution.name,
        )


def test_live_reviewed_split_binding_is_exact() -> None:
    root = Path(__file__).parents[1]
    reviewed = root / "artifacts/splits/triviaqa-reviewed"
    manifest = validate_reviewed_split_snapshot(reviewed)
    assert manifest["manifest_digest"] == (
        "05e13f0193155551400fd636e8dd6d97e065dd80205133a9440ef13105bce148"
    )
    assert sha256_path(reviewed) == (
        "3ceaf111654b80e34abd568853f64bba894fc7c6d7a81950c2868f3584a187f4"
    )
    graders = verify_e1_grader_bundle(
        root / "artifacts/graders/e1-frozen-v2",
        expected_manifest_digest=(
            "b3af3c847c3488d6228a47c205186caca06bca8de1cd00dd81f0b83ac73e1159"
        ),
    )
    assert graders["manifest_digest"] == (
        "b3af3c847c3488d6228a47c205186caca06bca8de1cd00dd81f0b83ac73e1159"
    )


def test_rq1_result_inventory_requires_recursive_task_directories(
    tmp_path: Path,
) -> None:
    root = tmp_path / "rq1-results"
    artifacts = root / f"rq1-{'a' * 64}" / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts.parent / "result.json").write_text("{}\n", encoding="utf-8")
    for name in robustness_results._RQ1_ARTIFACT_KEYS:
        (artifacts / name).write_text(name, encoding="utf-8")
    assert robustness_results._rq1_result_directories(root) == (artifacts.parent,)

    detached = root / f"rq1-{'b' * 64}.json"
    detached.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FrozenArtifactError, match="inventory"):
        robustness_results._rq1_result_directories(root)

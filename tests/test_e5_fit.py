from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest
import torch
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import mfh.experiments.e5_layer_labels as e5_layer_label_module
from mfh.contracts import ActivationSite, Outcome, PromptSpec, Question, TokenScope
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.experiments.e4_baselines import (
    E4MethodPolicy,
    E4Protocol,
    build_e4_screen_receipt,
    write_e4_screen_receipt,
)
from mfh.experiments.e5_adaptive import E5Protocol, build_e5_ablation_grid
from mfh.experiments.e5_capture import E5FitCaptureData, VerifiedE5FitCapture
from mfh.experiments.e5_fit import (
    E5FitRecipe,
    e5_fit_capture_attestation_body,
    fit_e5_controller_grid,
    package_e5_controller_bindings,
    save_e5_fitted_grid,
    sign_e5_fit_capture_attestation,
    verify_e5_controller_bindings,
    verify_e5_fitted_grid,
)
from mfh.experiments.e5_layer_labels import E5LayerLabelData, VerifiedE5LayerLabels
from mfh.experiments.e5_native import (
    finalize_e5_native_ablation,
    open_e5_native_promotion_source,
    prepare_e5_native_ablation,
    run_e5_native_ablation,
    verify_e5_native_ablation,
)
from mfh.experiments.static_direction_sources import resolve_static_direction
from mfh.inference.architecture import HookKey
from mfh.inference.mlx_research import MlxPromptFeatureCubeOutput
from mfh.inference.mlx_runtime import MlxGenerationOutput, MlxRenderedPrompt
from mfh.methods.adaptive import load_adaptive_controller
from mfh.methods.features import ActivationFeatureSchema, FeatureComposition
from mfh.methods.probes import (
    ProbeDataset,
    ProbeTask,
    ProbeTrainingConfig,
    fit_calibrated_probe,
    save_calibrated_probe,
)
from mfh.provenance import canonical_json, sha256_file, sha256_path, stable_hash

_PRIVATE = "11" * 32
_PUBLIC = (
    Ed25519PrivateKey.from_private_bytes(bytes.fromhex(_PRIVATE))
    .public_key()
    .public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    .hex()
)


def _dataset(partition: str, *, calibration: bool = False) -> ProbeDataset:
    generator = torch.Generator().manual_seed(33 if calibration else 32)
    centers = {
        Outcome.CORRECT: torch.tensor([-2.0, -2.0]),
        Outcome.INCORRECT: torch.tensor([2.0, 2.0]),
        Outcome.ABSTENTION: torch.tensor([-2.0, 2.0]),
    }
    features: list[torch.Tensor] = []
    outcomes: list[Outcome] = []
    identifiers: list[str] = []
    for outcome, center in centers.items():
        for index in range(4):
            features.append(center + 0.05 * torch.randn(2, generator=generator))
            outcomes.append(outcome)
            identifiers.append(f"{partition}-{outcome.value}-{index}")
    return ProbeDataset(
        tuple(identifiers),
        torch.stack(features),
        tuple(outcomes),
        group_ids=tuple(identifiers),
        feature_schema=ActivationFeatureSchema.synthetic(partition=partition, width=2),
    )


def _inputs(
    tmp_path: Path,
    *,
    protocol: E5Protocol | None = None,
    runtime_artifact_sha256_override: str | None = None,
):  # type: ignore[no-untyped-def]
    controller = _dataset("T-controller-train")
    calibration = _dataset("T-controller-calibration", calibration=True)
    risk = fit_calibrated_probe(
        controller,
        calibration,
        task=ProbeTask.CORRECT_INCORRECT_ABSTENTION,
        training_config=ProbeTrainingConfig(epochs=25),
    )
    features = torch.tensor(
        [[-2.0, -2.0], [2.0, 2.0]] * 6,
        dtype=torch.float32,
    )
    outcomes = tuple(
        Outcome.CORRECT if index % 2 == 0 else Outcome.INCORRECT for index in range(12)
    )
    identifiers = tuple(f"steer-{index}" for index in range(12))
    steer = ProbeDataset(
        identifiers,
        features,
        outcomes,
        group_ids=identifiers,
        feature_schema=ActivationFeatureSchema.synthetic(partition="T-steer", width=2),
    )
    activations = {
        HookKey(layer, ActivationSite.POST_MLP): torch.stack(
            [
                torch.tensor([float(layer), 1.0]) if outcome is Outcome.CORRECT else torch.zeros(2)
                for outcome in outcomes
            ]
        )
        for layer in (16, 31, 32)
    }
    protocol = protocol or E5Protocol(
        vector_counts=(1,),
        routers=("nearest_centroid", "linear_softmax", "two_layer_mlp"),
        alpha_modes=("fixed", "risk_gated", "risk_gated_hard_threshold"),
        layer_modes=("fixed_best", "two_layer_router", "three_layer_router"),
        intervention_timings=("final_prompt", "first_generated", "first_four_generated"),
        controller_inputs=("one_layer",),
    )
    recipe = E5FitRecipe(
        fixed_best_layer=31,
        two_layer_candidates=(31, 32),
        three_layer_candidates=(16, 31, 32),
        intervention_site=ActivationSite.POST_MLP,
        minimum_class_count=2,
        router_epochs=10,
        layer_epochs=15,
    )
    composition = FeatureComposition.SINGLE_LAYER
    risk_path = tmp_path / "risk-probe"
    save_calibrated_probe(risk_path, risk)
    runtime_artifact_sha256 = runtime_artifact_sha256_override or "2" * 64
    e2_probe_bundle_sha256 = "3" * 64
    e3_static_vectors_sha256 = "4" * 64
    e3_construction_sha256 = "5" * 64
    capture_directory = tmp_path / "native-capture"
    capture_directory.mkdir()
    (capture_directory / "capture.txt").write_text("signed", encoding="utf-8")
    capture_plan = {
        "plan_identity": "7" * 64,
        "protocol": protocol.to_dict(),
        "recipe": recipe.to_dict(),
        "runtime_identity": {
            "model_repository": "mlx-community/test",
            "model_revision": "c" * 40,
            "model_quantization": "4bit",
            "model_num_layers": 64,
        },
        "execution_public_key": _PUBLIC,
        "runtime_artifact_sha256": runtime_artifact_sha256,
        "e2_probe_bundle_sha256": e2_probe_bundle_sha256,
        "e3_static_vectors_sha256": e3_static_vectors_sha256,
        "e3_construction_sha256": e3_construction_sha256,
    }
    (capture_directory / "plan.json").write_text(
        json.dumps(capture_plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    verified_capture = VerifiedE5FitCapture(
        directory=capture_directory.resolve(),
        plan=capture_plan,
        pairs_completed=len(steer.question_ids),
        shard_count=1,
        chain_head="8" * 64,
        complete=True,
        scientific_eligible=False,
        maximum_peak_memory_bytes=1024,
    )
    capture_data = E5FitCaptureData(
        verified=verified_capture,
        vector_datasets={composition: steer},
        vector_activations=activations,
        capture_artifact_sha256=sha256_path(capture_directory),
    )
    layer_label_directory = tmp_path / "layer-labels"
    layer_label_directory.mkdir()
    (layer_label_directory / "labels.txt").write_text("signed", encoding="utf-8")
    verified_labels = VerifiedE5LayerLabels(
        directory=layer_label_directory.resolve(),
        plan={
            "plan_identity": "9" * 64,
            "recipe": recipe.to_dict(),
            "execution_public_key": _PUBLIC,
            "fit_capture_artifact_sha256": capture_data.capture_artifact_sha256,
            "fit_capture_plan_identity": capture_plan["plan_identity"],
            "fit_capture_chain_head": verified_capture.chain_head,
        },
        records_completed=36,
        shard_count=1,
        chain_head="a" * 64,
        complete=True,
        scientific_eligible=False,
        maximum_peak_memory_bytes=1024,
    )
    layer_labels = E5LayerLabelData(
        verified=verified_labels,
        question_ids=controller.question_ids,
        group_ids=controller.group_ids,
        outcomes=controller.outcomes,
        best_layers_two=tuple(31 if index % 2 == 0 else 32 for index in range(12)),
        best_layers_three=tuple((16, 31, 32)[index % 3] for index in range(12)),
        artifact_sha256=sha256_path(layer_label_directory),
    )
    # Unit-fit fixtures stand in for load_e5_layer_label_data; production callers
    # can obtain this capability only from that fully verifying loader.
    object.__setattr__(
        layer_labels,
        "_verification_token",
        e5_layer_label_module._VERIFIED_LAYER_LABEL_DATA,
    )
    e5_layer_label_module._VERIFIED_LAYER_LABEL_RECEIPTS[id(layer_labels)] = MappingProxyType(
        {
            "object": layer_labels,
            "binding_objects": (
                id(layer_labels.verified),
                id(layer_labels.verified.plan),
                id(layer_labels.question_ids),
                id(layer_labels.group_ids),
                id(layer_labels.outcomes),
                id(layer_labels.best_layers_two),
                id(layer_labels.best_layers_three),
            ),
            "binding_digest": layer_labels._binding_digest(),
        }
    )
    kwargs = {
        "protocol": protocol,
        "recipe": recipe,
        "risk_probes": {composition: risk},
        "risk_probe_artifact_sha256": {composition: sha256_path(risk_path)},
        "risk_probe_artifact_paths": {composition: risk_path},
        "controller_datasets": {composition: controller},
        "capture_data": capture_data,
        "layer_labels": layer_labels,
        "runtime_artifact_sha256": runtime_artifact_sha256,
        "e2_probe_bundle_sha256": e2_probe_bundle_sha256,
        "e3_static_vectors_sha256": e3_static_vectors_sha256,
        "e3_construction_sha256": e3_construction_sha256,
        "expected_execution_public_key": _PUBLIC,
    }
    body = e5_fit_capture_attestation_body(
        protocol=protocol,
        recipe=recipe,
        execution_public_key=_PUBLIC,
        runtime_artifact_sha256=kwargs["runtime_artifact_sha256"],
        e2_probe_bundle_sha256=kwargs["e2_probe_bundle_sha256"],
        e3_static_vectors_sha256=kwargs["e3_static_vectors_sha256"],
        e3_construction_sha256=kwargs["e3_construction_sha256"],
        risk_probes=kwargs["risk_probes"],
        risk_probe_artifact_sha256=kwargs["risk_probe_artifact_sha256"],
        risk_probe_artifact_paths=kwargs["risk_probe_artifact_paths"],
        controller_datasets=kwargs["controller_datasets"],
        capture_data=kwargs["capture_data"],
        layer_labels=kwargs["layer_labels"],
    )
    kwargs["capture_attestation"] = sign_e5_fit_capture_attestation(body, private_key_hex=_PRIVATE)
    return kwargs


def test_e5_fit_covers_full_grid_and_deduplicates_timing_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _inputs(tmp_path)
    fitted = fit_e5_controller_grid(**kwargs)
    grid = build_e5_ablation_grid(kwargs["protocol"])
    assert len(fitted.controllers) == len(grid) == 81
    assert len(set(fitted.controller_fit_ids.values())) == 27
    groups: dict[str, list[str]] = {}
    for spec in grid:
        groups.setdefault(fitted.controller_fit_ids[spec.spec_id], []).append(spec.spec_id)
    assert all(len(values) == 3 for values in groups.values())
    assert all(
        len({id(fitted.controllers[spec_id]) for spec_id in spec_ids}) == 1
        for spec_ids in groups.values()
    )
    saved = save_e5_fitted_grid(tmp_path / "grid", fitted=fitted)
    assert len(saved.controller_directories) == 81
    assert saved.manifest["unique_controller_fit_count"] == 27
    assert verify_e5_fitted_grid(saved.directory).manifest == saved.manifest
    # The reduced unit schemas are deliberately synthetic; production binding
    # validation retains the exact active-model schema check.
    monkeypatch.setattr(
        "mfh.experiments.e5_adaptive._validate_controller",
        lambda _spec, directory: load_adaptive_controller(directory),
    )
    bindings = package_e5_controller_bindings(
        tmp_path / "bindings",
        fitted_grid_directory=saved.directory,
        expected_execution_public_key=_PUBLIC,
    )
    assert len(bindings.binding_paths) == 81
    assert verify_e5_controller_bindings(bindings.directory).manifest == bindings.manifest


def test_e5_saved_grid_rejects_controller_mutation(tmp_path: Path) -> None:
    fitted = fit_e5_controller_grid(**_inputs(tmp_path))
    saved = save_e5_fitted_grid(tmp_path / "grid", fitted=fitted)
    first = next(iter(saved.controller_directories.values()))
    provenance = first / "e5-fit-provenance.json"
    provenance.write_text(provenance.read_text() + " ", encoding="utf-8")
    with pytest.raises(DataValidationError, match="differs"):
        verify_e5_fitted_grid(saved.directory)


def test_e5_fit_rejects_tampered_capture_even_when_tensors_still_fit(
    tmp_path: Path,
) -> None:
    kwargs = _inputs(tmp_path)
    attestation = kwargs["capture_attestation"]
    body = dict(attestation["body"])
    body["layer_label_receipt_sha256"] = "9" * 64
    kwargs["capture_attestation"] = {
        "body": body,
        "signature": attestation["signature"],
    }
    with pytest.raises(DataValidationError, match="differs from live tensors"):
        fit_e5_controller_grid(**kwargs)


def test_e5_fit_rejects_manually_constructed_layer_labels(tmp_path: Path) -> None:
    kwargs = _inputs(tmp_path)
    labels = kwargs["layer_labels"]
    kwargs["layer_labels"] = E5LayerLabelData(
        verified=labels.verified,
        question_ids=labels.question_ids,
        group_ids=labels.group_ids,
        outcomes=labels.outcomes,
        best_layers_two=labels.best_layers_two,
        best_layers_three=labels.best_layers_three,
        artifact_sha256=labels.artifact_sha256,
    )
    with pytest.raises(FrozenArtifactError, match="verifier-authorized"):
        fit_e5_controller_grid(**kwargs)


def test_e5_fit_rejects_self_selected_attestation_key(tmp_path: Path) -> None:
    kwargs = _inputs(tmp_path)
    attacker_private = "22" * 32
    attacker_public = (
        Ed25519PrivateKey.from_private_bytes(bytes.fromhex(attacker_private))
        .public_key()
        .public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        .hex()
    )
    body = dict(kwargs["capture_attestation"]["body"])
    body["execution_public_key"] = attacker_public
    kwargs["capture_attestation"] = sign_e5_fit_capture_attestation(
        body, private_key_hex=attacker_private
    )
    with pytest.raises(DataValidationError, match="trusted plan"):
        fit_e5_controller_grid(**kwargs)


def test_e5_fit_rejects_split_overlap_and_cross_composition_misalignment(
    tmp_path: Path,
) -> None:
    kwargs = _inputs(tmp_path)
    composition = FeatureComposition.SINGLE_LAYER
    controller = kwargs["controller_datasets"][composition]
    capture = kwargs["capture_data"]
    steer = capture.vector_datasets[composition]
    overlapping = ProbeDataset(
        controller.question_ids,
        steer.features,
        steer.outcomes,
        group_ids=controller.group_ids,
        feature_schema=steer.feature_schema,
    )
    kwargs["capture_data"] = E5FitCaptureData(
        verified=capture.verified,
        vector_datasets={composition: overlapping},
        vector_activations=capture.vector_activations,
        capture_artifact_sha256=capture.capture_artifact_sha256,
    )
    with pytest.raises(DataValidationError, match="overlap"):
        fit_e5_controller_grid(**kwargs)


def test_e5_fit_rejects_probe_object_not_equal_to_e2_artifact(tmp_path: Path) -> None:
    kwargs = _inputs(tmp_path)
    composition = FeatureComposition.SINGLE_LAYER
    probe = kwargs["risk_probes"][composition]
    mutated_parameters = dict(probe.state.parameters)
    mutated_parameters["weight"] = mutated_parameters["weight"] + 0.25
    from mfh.methods.probes import CalibratedProbe, ProbeState

    kwargs["risk_probes"] = {
        composition: CalibratedProbe(
            task=probe.task,
            state=ProbeState(
                kind=probe.state.kind,
                labels=probe.state.labels,
                feature_mean=probe.state.feature_mean,
                feature_scale=probe.state.feature_scale,
                parameters=mutated_parameters,
                hidden_width=probe.state.hidden_width,
            ),
            calibrator=probe.calibrator,
            training_fingerprint=probe.training_fingerprint,
            calibration_fingerprint=probe.calibration_fingerprint,
            training_schema=probe.training_schema,
            calibration_schema=probe.calibration_schema,
        )
    }
    with pytest.raises(DataValidationError, match="artifact tensors"):
        fit_e5_controller_grid(**kwargs)


def test_e5_fit_recipe_rejects_layer_geometry_not_nested() -> None:
    with pytest.raises(DataValidationError, match="recipe"):
        E5FitRecipe(
            fixed_best_layer=31,
            two_layer_candidates=(31, 32),
            three_layer_candidates=(31, 47, 48),
            intervention_site=ActivationSite.POST_MLP,
        )


def test_e5_fit_recipe_rejects_unregistered_qwen_layers() -> None:
    with pytest.raises(DataValidationError, match="registered Qwen"):
        E5FitRecipe(
            fixed_best_layer=23,
            two_layer_candidates=(23, 31),
            three_layer_candidates=(23, 31, 39),
            intervention_site=ActivationSite.POST_MLP,
        )


@dataclass
class _NativeState:
    direction: np.ndarray[Any, Any]
    alpha: float
    token_scope: TokenScope
    applied_pre_history: list[np.ndarray[Any, Any]]
    applied_post_history: list[np.ndarray[Any, Any]]
    applications: int = 0


class _NativeRuntime:
    def __init__(self, identity: Mapping[str, Any]) -> None:
        self._identity = dict(identity)

    def runtime_identity(self) -> Mapping[str, Any]:
        return self._identity

    def render_prompt(
        self,
        prompt: PromptSpec,
        question: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> MlxRenderedPrompt:
        del metadata
        text = f"{prompt.text}|{question}"
        tokens = (10, 20)
        return MlxRenderedPrompt(
            text=text,
            sha256=hashlib.sha256(text.encode()).hexdigest(),
            token_ids=tokens,
            token_ids_sha256=hashlib.sha256(b"10,20").hexdigest(),
            messages=(),
        )

    def prompt_feature_cube(
        self,
        rendered: MlxRenderedPrompt,
        *,
        layers: Sequence[int],
        sites: Sequence[ActivationSite],
    ) -> MlxPromptFeatureCubeOutput:
        del rendered
        return MlxPromptFeatureCubeOutput(
            activations={
                site: {layer: np.array([[1.0, -1.0]], dtype=np.float32) for layer in layers}
                for site in sites
            },
            maximum_token_probability=0.7,
            output_entropy=0.4,
            peak_memory_bytes=128,
        )

    def standardized_intervention_state(
        self,
        direction: np.ndarray[Any, Any],
        *,
        standardized_alpha: float,
        reference_rms: float,
        token_scope: TokenScope,
        decay: float = 0.0,
    ) -> _NativeState:
        del decay
        return _NativeState(
            direction=np.asarray(direction, dtype=np.float32).copy(),
            alpha=standardized_alpha * reference_rms,
            token_scope=token_scope,
            applied_pre_history=[],
            applied_post_history=[],
        )

    def generate_with_interventions(
        self,
        rendered: MlxRenderedPrompt,
        *,
        max_new_tokens: int,
        intervention_states: Mapping[tuple[int, ActivationSite], Any],
    ) -> MlxGenerationOutput:
        del max_new_tokens
        output_tokens = (101, 102)
        for raw_state in intervention_states.values():
            assert isinstance(raw_state, _NativeState)
            state = raw_state
            applications = {
                TokenScope.FINAL_PROMPT: 1,
                TokenScope.FIRST_GENERATED: 1,
                TokenScope.FIRST_FOUR: len(output_tokens),
            }[state.token_scope]
            for _ in range(applications):
                before = np.zeros_like(state.direction)
                state.applied_pre_history.append(before)
                state.applied_post_history.append(before + state.direction * state.alpha)
            state.applications = applications
        return MlxGenerationOutput(
            rendered_prompt=rendered,
            token_ids=output_tokens,
            text="answer-0",
            input_tokens=len(rendered.token_ids),
            output_tokens=len(output_tokens),
            latency_seconds=0.01,
            stop_type="short_answer",
            stopping_token_id=output_tokens[-1],
            prompt_tokens_per_second=10.0,
            generation_tokens_per_second=10.0,
            peak_memory_bytes=256,
            active_memory_bytes=128,
            cache_memory_bytes=64,
        )


def _native_vectors(tmp_path: Path) -> Path:
    root = tmp_path / "e3-vectors"
    root.mkdir()
    directions = np.zeros((2, 2, 1, 3, 2), dtype=np.float32)
    directions[..., 0] = 1.0
    rms = np.ones((2, 2, 1, 3), dtype=np.float64)
    counts = np.ones((2, 2, 1, 3), dtype=np.int64)
    np.savez_compressed(
        root / "vectors.npz",
        directions=directions,
        reference_rms=rms,
        correct_counts=counts,
        incorrect_counts=counts,
    )
    body = {
        "schema_version": 1,
        "phase": "E3-construction",
        "scientific_eligible": True,
        "vectors_sha256": sha256_file(root / "vectors.npz"),
        "prompt_axis": ["P0-neutral", "P2-calibrated-abstention"],
        "extraction_axis": ["M1-R", "M1-P"],
        "site_axis": ["post_mlp"],
        "layer_axis": [16, 31, 32],
    }
    (root / "metadata.json").write_text(
        json.dumps({**body, "metadata_digest": stable_hash(body)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root


def test_e5_native_ablation_resumes_verifies_and_finalizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = E5Protocol(
        vector_counts=(1,),
        routers=("nearest_centroid",),
        alpha_modes=("fixed",),
        layer_modes=("fixed_best",),
        intervention_timings=("final_prompt", "first_generated", "first_four_generated"),
        controller_inputs=("one_layer",),
    )
    runtime_artifact = tmp_path / "runtime-attestation.json"
    runtime_artifact.write_text("{}\n", encoding="utf-8")
    fit_inputs = _inputs(
        tmp_path,
        protocol=protocol,
        runtime_artifact_sha256_override=sha256_path(runtime_artifact),
    )
    fitted = fit_e5_controller_grid(**fit_inputs)
    grid = save_e5_fitted_grid(tmp_path / "native-grid", fitted=fitted)
    monkeypatch.setattr(
        "mfh.experiments.e5_adaptive._validate_controller",
        lambda _spec, directory: load_adaptive_controller(directory),
    )
    bindings = package_e5_controller_bindings(
        tmp_path / "native-bindings",
        fitted_grid_directory=grid.directory,
        expected_execution_public_key=_PUBLIC,
    )
    questions = (
        Question("native-q-0", "triviaqa", "Question 0?", ("answer-0",), split="T-dev"),
        Question("native-q-1", "triviaqa", "Question 1?", ("answer-1",), split="T-dev"),
    )
    screen = build_e4_screen_receipt(questions, protocol=E4Protocol(dev_rows=2, screen_rows=1))
    screen_path = tmp_path / "native-screen.json"
    write_e4_screen_receipt(screen_path, screen)
    vectors = _native_vectors(tmp_path)
    resolved = resolve_static_direction(
        vectors, method="M1", layer=31, site=ActivationSite.POST_MLP
    )
    policy_body = {
        "schema_version": 1,
        "method": "M1",
        "capability_report_digest": "d" * 64,
        "implementation_artifact_sha256": sha256_path(vectors),
        "layer": 31,
        "site": ActivationSite.POST_MLP.value,
        "token_scope": TokenScope.FINAL_PROMPT.value,
        "alpha": 0.5,
        "adaptive_policy": None,
        "direction_sha256": resolved.direction_sha256,
        "direction_norm": resolved.direction_norm,
        "reference_rms": resolved.reference_rms,
        "execution_public_key": _PUBLIC,
    }
    policy = E4MethodPolicy(
        method="M1",
        capability_report_digest="d" * 64,
        implementation_artifact_sha256=sha256_path(vectors),
        layer=31,
        site=ActivationSite.POST_MLP,
        token_scope=TokenScope.FINAL_PROMPT,
        alpha=0.5,
        adaptive_policy=None,
        direction_sha256=resolved.direction_sha256,
        direction_norm=resolved.direction_norm,
        reference_rms=resolved.reference_rms,
        execution_public_key=_PUBLIC,
        policy_digest=stable_hash(policy_body),
    )
    policy_path = tmp_path / "m1-policy.json"
    policy_path.write_text(
        json.dumps(policy.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    identity = {
        "model_repository": "mlx-community/test",
        "model_revision": "c" * 40,
        "model_quantization": "4bit",
        "model_num_layers": 64,
    }
    prompts = {
        "P0-neutral": PromptSpec(
            "P0-neutral",
            "You are a helpful assistant. Answer the user's factual question.",
        ),
        "P2-calibrated-abstention": PromptSpec(
            "P2-calibrated-abstention",
            "Answer only when you are sufficiently confident that the answer is correct. "
            "Otherwise respond exactly with “I don't know.” Do not guess.",
        ),
    }
    work = tmp_path / "native-run"
    wrong_recipe = E5FitRecipe(
        fixed_best_layer=16,
        two_layer_candidates=(16, 31),
        three_layer_candidates=(16, 31, 32),
        intervention_site=ActivationSite.POST_MLP,
        minimum_class_count=2,
        router_epochs=10,
        layer_epochs=15,
    )
    wrong_grid = replace(
        grid,
        manifest=MappingProxyType({**grid.manifest, "recipe": wrong_recipe.to_dict()}),
    )
    with monkeypatch.context() as mismatch_patch:
        mismatch_patch.setattr(
            "mfh.experiments.e5_native.verify_e5_fitted_grid",
            lambda _directory: wrong_grid,
        )
        with pytest.raises(FrozenArtifactError, match="layer/site geometry"):
            prepare_e5_native_ablation(
                tmp_path / "wrong-native-run",
                screen_receipt_path=screen_path,
                controller_bindings_directory=bindings.directory,
                fit_capture_directory=fit_inputs["capture_data"].verified.directory,
                m1_policy_path=policy_path,
                e3_static_vectors_directory=vectors,
                runtime_artifact=runtime_artifact,
                prompts=prompts,
                execution_public_key=_PUBLIC,
                protocol=protocol,
                shard_rows=3,
                max_peak_memory_bytes=4_096,
            )
    plan = prepare_e5_native_ablation(
        work,
        screen_receipt_path=screen_path,
        controller_bindings_directory=bindings.directory,
        fit_capture_directory=fit_inputs["capture_data"].verified.directory,
        m1_policy_path=policy_path,
        e3_static_vectors_directory=vectors,
        runtime_artifact=runtime_artifact,
        prompts=prompts,
        execution_public_key=_PUBLIC,
        protocol=protocol,
        shard_rows=3,
        max_peak_memory_bytes=4_096,
    )
    assert plan["expected_records"] == 16
    partial = run_e5_native_ablation(
        work,
        runtime=_NativeRuntime(identity),
        execution_private_key_hex=_PRIVATE,
        request_budget=3,
    )
    assert partial.records_completed == 3
    rogue_final = work / "final"
    rogue_final.mkdir()
    with pytest.raises(FrozenArtifactError, match=r"incomplete.*finalization"):
        verify_e5_native_ablation(work, expected_execution_public_key=_PUBLIC)
    rogue_final.rmdir()
    abandoned = work / "shards" / ".shard-000001.stage-abandoned"
    abandoned.mkdir()
    with monkeypatch.context() as resume_patch:
        resume_patch.setattr(
            "mfh.experiments.e5_native._read_jsonl",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("resume must not read historical row payloads")
            ),
        )
        resume_patch.setattr(
            "mfh.experiments.e5_native.verify_e5_controller_bindings",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("resume must not replay all controller bindings")
            ),
        )
        resume_patch.setattr(
            "mfh.experiments.e5_native.verify_e5_fitted_grid",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("resume must not replay the fitted controller grid")
            ),
        )
        complete = run_e5_native_ablation(
            work,
            runtime=_NativeRuntime(identity),
            execution_private_key_hex=_PRIVATE,
        )
    assert complete.complete and complete.records_completed == 16
    assert not abandoned.exists()
    verified = verify_e5_native_ablation(
        work, expected_execution_public_key=_PUBLIC, require_complete=True
    )
    assert verified.maximum_peak_memory_bytes == 256

    last_shard = sorted((work / "shards").iterdir())[-1]
    records_path = last_shard / "records.jsonl"
    manifest_path = last_shard / "manifest.json"
    original_records = records_path.read_text(encoding="utf-8")
    original_manifest = manifest_path.read_text(encoding="utf-8")
    row = json.loads(original_records)
    row["evidence"]["selected_layer"] += 1
    evidence_body = dict(row["evidence"])
    evidence_body.pop("evidence_digest")
    row["evidence"]["evidence_digest"] = stable_hash(evidence_body)
    row_body = dict(row)
    row_body.pop("row_digest")
    row["row_digest"] = stable_hash(row_body)
    records_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    manifest = json.loads(original_manifest)
    signed = dict(manifest)
    signed.pop("signature")
    signed.pop("chain_head")
    signed["records_sha256"] = sha256_file(records_path)
    signed["record_digests"] = [row["row_digest"]]
    unsigned = dict(signed)
    unsigned.pop("manifest_digest")
    signed["manifest_digest"] = stable_hash(unsigned)
    signature = (
        Ed25519PrivateKey.from_private_bytes(bytes.fromhex(_PRIVATE))
        .sign(canonical_json(signed).encode())
        .hex()
    )
    manifest = {
        **signed,
        "signature": signature,
        "chain_head": stable_hash({"signed": signed, "signature": signature}),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    assert verify_e5_native_ablation(
        work,
        expected_execution_public_key=_PUBLIC,
        require_complete=True,
        semantic=False,
    ).complete
    with pytest.raises(FrozenArtifactError, match="policy transcript"):
        verify_e5_native_ablation(
            work,
            expected_execution_public_key=_PUBLIC,
            require_complete=True,
            semantic=True,
        )
    records_path.write_text(original_records, encoding="utf-8")
    manifest_path.write_text(original_manifest, encoding="utf-8")

    finalized = finalize_e5_native_ablation(work, execution_private_key_hex=_PRIVATE)
    assert finalized.finalized_records is not None
    assert len(finalized.finalized_records.read_text(encoding="utf-8").splitlines()) == 16
    promotion_source = open_e5_native_promotion_source(
        work,
        expected_execution_public_key=_PUBLIC,
        expected_final_records_sha256=sha256_file(finalized.finalized_records),
    )
    assert promotion_source.complete
    assert promotion_source.chain_head == finalized.chain_head
    with pytest.raises(FrozenArtifactError, match="external trust root"):
        verify_e5_native_ablation(
            work, expected_execution_public_key="4" * 64, require_complete=True
        )
    assert finalized.finalized_records is not None
    finalized.finalized_records.write_text(
        finalized.finalized_records.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    with pytest.raises(FrozenArtifactError, match="finalization receipt differs"):
        verify_e5_native_ablation(
            work, expected_execution_public_key=_PUBLIC, require_complete=True
        )

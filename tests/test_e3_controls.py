from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mfh.contracts import ActivationSite, Outcome, PromptSpec, Question, TokenScope
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.experiments.e3_construction import (
    finalize_e3_vector_bundle,
    prepare_e3_construction_work,
    run_e3_construction,
)
from mfh.experiments.e3_control_materials import (
    load_e3_fixed_control_direction,
    verify_e3_fixed_control_materials,
    write_e3_fixed_control_materials,
)
from mfh.experiments.e3_controls import (
    finalize_e3_shuffled_control_bundle,
    prepare_e3_shuffled_control_work,
    run_e3_shuffled_control,
    verify_e3_shuffled_control_bundle,
    verify_e3_shuffled_control_work,
)
from mfh.experiments.e3_execution import (
    E3ExecutionResult,
    execute_e3_condition,
    load_e3_execution_assets,
)
from mfh.experiments.e3_phase import (
    finalize_e3_phase,
    load_e3_analysis_surface,
    verify_e3_phase,
)
from mfh.experiments.e3_runner import (
    e3_selection_inputs_from_work,
    prepare_e3_evaluation_work,
    run_e3_evaluation,
    verify_e3_evaluation_work,
)
from mfh.experiments.e3_schedule import (
    E3Condition,
    E3Protocol,
    e3_alpha_conditions,
    e3_control_conditions,
    e3_cross_prompt_conditions,
    e3_final_conditions,
    e3_geometry_conditions,
    e3_p3_conditions,
    e3_scope_conditions,
    select_e3_screen_questions,
)
from mfh.experiments.e3_selection import (
    E3StageSelection,
    VerifiedE3StageSelection,
    derive_e3_stage_selection,
    load_verified_e3_stage_selection,
    write_e3_stage_selection,
)
from mfh.inference.mlx_research import (
    MlxPromptFeatureCubeOutput,
    MlxTeacherForcedCubeOutput,
)
from mfh.inference.mlx_runtime import (
    MlxGenerationOutput,
    MlxInterventionState,
    MlxRenderedPrompt,
)
from mfh.provenance import sha256_file, stable_hash


def _token_digest(token_ids: Sequence[int]) -> str:
    return hashlib.sha256(",".join(str(value) for value in token_ids).encode()).hexdigest()


def _protocol() -> E3Protocol:
    return E3Protocol(
        steer_rows=4,
        dev_rows=6,
        screen_rows=2,
        candidate_layers=(1, 2),
        candidate_sites=(ActivationSite.POST_MLP,),
        standardized_alphas=(0.0, 0.5),
        token_scopes=(TokenScope.FINAL_PROMPT, TokenScope.FIRST_FOUR),
    )


def _questions() -> tuple[Question, ...]:
    return tuple(
        Question(
            question_id=f"q-{index}",
            benchmark="triviaqa",
            text=f"Question {index}?",
            aliases=(f"answer-{index}",),
            split="T-steer",
        )
        for index in range(4)
    )


def _prompts() -> dict[str, PromptSpec]:
    return {
        value: PromptSpec(value, f"System prompt {value}")
        for value in ("P0-neutral", "P2-calibrated-abstention")
    }


class _VariedRuntime:
    def __init__(self) -> None:
        self.generate_calls = 0

    def runtime_identity(self) -> Mapping[str, Any]:
        return {"runtime": "fake-mlx", "revision": "a" * 40}

    def render_prompt(
        self,
        prompt: PromptSpec,
        question: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> MlxRenderedPrompt:
        del metadata
        text = f"{prompt.prompt_id}|{question}"
        index = int(question.split()[1].rstrip("?"))
        tokens = (100 + index, 200 + len(prompt.prompt_id))
        return MlxRenderedPrompt(
            text=text,
            sha256=hashlib.sha256(text.encode()).hexdigest(),
            token_ids=tokens,
            token_ids_sha256=_token_digest(tokens),
            messages=(),
        )

    def generate(self, rendered: MlxRenderedPrompt, *, max_new_tokens: int) -> MlxGenerationOutput:
        assert max_new_tokens == 8
        self.generate_calls += 1
        index = int(rendered.text.split("Question ")[1].rstrip("?"))
        text = f"answer-{index}" if index % 2 == 0 else "wrong"
        token_ids = (300 + index,)
        return MlxGenerationOutput(
            rendered_prompt=rendered,
            token_ids=token_ids,
            text=text,
            input_tokens=len(rendered.token_ids),
            output_tokens=1,
            latency_seconds=0.1,
            stop_type="short_answer",
            stopping_token_id=token_ids[-1],
            prompt_tokens_per_second=10.0,
            generation_tokens_per_second=5.0,
            peak_memory_bytes=1024,
            active_memory_bytes=512,
            cache_memory_bytes=256,
        )

    def _base(self, rendered: MlxRenderedPrompt) -> float:
        text = rendered.text
        index = int(text.split("Question ")[1].rstrip("?"))
        return (3.0 if index % 2 == 0 else 1.0) + index / 10

    def prompt_feature_cube(
        self,
        rendered: MlxRenderedPrompt,
        *,
        layers: Sequence[int],
        sites: Sequence[ActivationSite],
    ) -> MlxPromptFeatureCubeOutput:
        base = self._base(rendered)
        return MlxPromptFeatureCubeOutput(
            activations={
                site: {
                    layer: np.asarray([[base + layer, 1.0, 0.5]], dtype=np.float32)
                    for layer in layers
                }
                for site in sites
            },
            maximum_token_probability=0.75,
            output_entropy=0.5,
            peak_memory_bytes=2048,
        )

    def teacher_forced_cube(
        self,
        rendered: MlxRenderedPrompt,
        response: str,
        *,
        layers: Sequence[int],
        sites: Sequence[ActivationSite],
    ) -> MlxTeacherForcedCubeOutput:
        base = self._base(rendered) + 1.0
        token_ids = (900,)
        return MlxTeacherForcedCubeOutput(
            response_text_sha256=hashlib.sha256(response.encode()).hexdigest(),
            response_token_ids=token_ids,
            response_token_ids_sha256=_token_digest(token_ids),
            token_log_probabilities=(-0.5,),
            negative_log_likelihood=0.5,
            mean_negative_log_likelihood=0.5,
            perplexity=math.exp(0.5),
            activations={
                site: {
                    layer: np.asarray([[base + layer, 1.5, 0.25]], dtype=np.float32)
                    for layer in layers
                }
                for site in sites
            },
            peak_memory_bytes=4096,
        )

    def standardized_intervention_state(
        self,
        direction: np.ndarray,
        *,
        standardized_alpha: float,
        reference_rms: float,
        token_scope: TokenScope,
        decay: float = 0.0,
    ) -> MlxInterventionState:
        return MlxInterventionState(
            direction=np.asarray(direction, dtype=np.float32).copy(),
            alpha=standardized_alpha * reference_rms,
            token_scope=token_scope,
            decay=decay,
        )

    def generate_with_interventions(
        self,
        rendered: MlxRenderedPrompt,
        *,
        max_new_tokens: int,
        intervention_states: Mapping[tuple[int, ActivationSite], MlxInterventionState],
    ) -> MlxGenerationOutput:
        assert len(intervention_states) == 1
        state = next(iter(intervention_states.values()))
        width = int(np.asarray(state.direction).shape[0])
        state.captured = np.zeros((1, 1, width), dtype=np.float32)
        state.intervened = state.captured.copy()
        if state.alpha != 0:
            state.applications = 1
            # Simulate a later inactive cached hook call overwriting the tensors.
            state.intervened = state.captured.copy()
        generated = self.generate(rendered, max_new_tokens=max_new_tokens)
        if state.alpha != 0 and generated.text == "wrong":
            index = int(rendered.text.split("Question ")[1].rstrip("?"))
            return replace(generated, text=f"answer-{index}")
        return generated


def _selection_matrix(
    conditions: tuple[E3Condition, ...],
    *,
    question_ids: tuple[str, ...] = ("screen-0", "screen-1"),
) -> tuple[dict[tuple[str, str], Outcome], dict[tuple[str, str], float]]:
    outcomes = {
        (condition.condition_id, question_id): (
            Outcome.INCORRECT if condition.method == "M0" else Outcome.CORRECT
        )
        for condition in conditions
        for question_id in question_ids
    }
    norms = {
        (condition.condition_id, question_id): float(condition.standardized_alpha)
        for condition in conditions
        if condition.method != "M0"
        for question_id in question_ids
    }
    return outcomes, norms


def _selection(
    stage: str,
    conditions: tuple[E3Condition, ...],
    *,
    source_plan_identity: str,
    predecessor: E3StageSelection | None = None,
    question_ids: tuple[str, ...] = ("screen-0", "screen-1"),
) -> E3StageSelection:
    outcomes, norms = _selection_matrix(conditions, question_ids=question_ids)
    return derive_e3_stage_selection(
        stage=stage,
        conditions=conditions,
        question_ids=question_ids,
        outcomes=outcomes,
        actual_delta_norms=norms,
        source_plan_identity=source_plan_identity,
        evaluation_plan_identity="b" * 64,
        evaluation_record_chain_head="c" * 64,
        evaluation_record_set_digest="d" * 64,
        source_scientific_eligible=False,
        predecessor_selection=predecessor,
        protocol=_protocol(),
    )


def _scope_selection(
    source_plan_identity: str,
    *,
    question_ids: tuple[str, ...] = ("screen-0", "screen-1"),
    artifact_path: Path,
) -> VerifiedE3StageSelection:
    protocol = _protocol()
    geometry_conditions = e3_geometry_conditions(protocol)
    geometry = _selection(
        "geometry",
        geometry_conditions,
        source_plan_identity=source_plan_identity,
        question_ids=question_ids,
    )
    alpha_conditions = e3_alpha_conditions(geometry.selected, protocol=protocol)
    alpha = _selection(
        "alpha",
        alpha_conditions,
        source_plan_identity=source_plan_identity,
        predecessor=geometry,
        question_ids=question_ids,
    )
    scope_conditions = e3_scope_conditions(alpha.selected, protocol=protocol)
    outcomes, norms = _selection_matrix(scope_conditions, question_ids=question_ids)
    inputs = {
        "stage": "scope",
        "conditions": scope_conditions,
        "question_ids": question_ids,
        "outcomes": outcomes,
        "actual_delta_norms": norms,
        "source_plan_identity": source_plan_identity,
        "evaluation_plan_identity": "b" * 64,
        "evaluation_record_chain_head": "c" * 64,
        "evaluation_record_set_digest": "d" * 64,
        "source_scientific_eligible": False,
        "predecessor_selection": alpha,
        "protocol": protocol,
    }
    write_e3_stage_selection(artifact_path, **inputs)
    return load_verified_e3_stage_selection(
        artifact_path,
        **inputs,
    )


def _sources(
    tmp_path: Path,
) -> tuple[Path, Path, _VariedRuntime, VerifiedE3StageSelection]:
    runtime = _VariedRuntime()
    work = tmp_path / "construction"
    plan = prepare_e3_construction_work(
        work,
        questions=_questions(),
        prompts=_prompts(),
        runtime_identity=runtime.runtime_identity(),
        hidden_width=3,
        protocol=_protocol(),
        checkpoint_rows=2,
        max_new_tokens=8,
    )
    run_e3_construction(
        work,
        questions=_questions(),
        prompts=_prompts(),
        runtime=runtime,
        protocol=_protocol(),
    )
    vectors = tmp_path / "vectors"
    finalize_e3_vector_bundle(
        vectors,
        work_directory=work,
        questions=_questions(),
        prompts=_prompts(),
        protocol=_protocol(),
        allow_non_scientific=True,
    )
    return (
        work,
        vectors,
        runtime,
        _scope_selection(
            str(plan["plan_identity"]), artifact_path=tmp_path / "scope-selection.json"
        ),
    )


def test_shuffled_control_resumes_without_generation_and_publishes_bundle(
    tmp_path: Path,
) -> None:
    construction, vectors, runtime, selection = _sources(tmp_path)
    generation_calls = runtime.generate_calls
    work = tmp_path / "shuffle-work"
    prepare_e3_shuffled_control_work(
        work,
        construction_directory=construction,
        vector_bundle_directory=vectors,
        questions=_questions(),
        prompts=_prompts(),
        scope_selection=selection,
        runtime_identity=runtime.runtime_identity(),
        protocol=_protocol(),
        checkpoint_rows=2,
    )

    partial = run_e3_shuffled_control(
        work,
        construction_directory=construction,
        vector_bundle_directory=vectors,
        questions=_questions(),
        prompts=_prompts(),
        scope_selection=selection,
        runtime=runtime,
        protocol=_protocol(),
        request_budget=1,
    )
    assert partial["processed_rows"] == 1
    assert runtime.generate_calls == generation_calls
    complete = run_e3_shuffled_control(
        work,
        construction_directory=construction,
        vector_bundle_directory=vectors,
        questions=_questions(),
        prompts=_prompts(),
        scope_selection=selection,
        runtime=runtime,
        protocol=_protocol(),
    )
    assert complete["complete"] is True
    assert complete["processed_rows"] == 4
    assert complete["maximum_peak_memory_bytes"] == 4096
    assert runtime.generate_calls == generation_calls

    bundle = tmp_path / "shuffled-vectors"
    with pytest.raises(FrozenArtifactError, match="not scientifically eligible"):
        finalize_e3_shuffled_control_bundle(
            bundle,
            work_directory=work,
            construction_directory=construction,
            vector_bundle_directory=vectors,
            questions=_questions(),
            prompts=_prompts(),
            scope_selection=selection,
            protocol=_protocol(),
        )
    result = finalize_e3_shuffled_control_bundle(
        bundle,
        work_directory=work,
        construction_directory=construction,
        vector_bundle_directory=vectors,
        questions=_questions(),
        prompts=_prompts(),
        scope_selection=selection,
        protocol=_protocol(),
        allow_non_scientific=True,
    )
    assert result["processed_rows"] == 4
    with np.load(bundle / "directions.npz", allow_pickle=False) as arrays:
        assert arrays["directions"].shape == (2, 3)
        assert np.allclose(np.linalg.norm(arrays["directions"], axis=1), 1.0)


def test_shuffled_control_rejects_checkpoint_and_bundle_content_tampering(
    tmp_path: Path,
) -> None:
    construction, vectors, runtime, selection = _sources(tmp_path)
    work = tmp_path / "shuffle-work"
    prepare_e3_shuffled_control_work(
        work,
        construction_directory=construction,
        vector_bundle_directory=vectors,
        questions=_questions(),
        prompts=_prompts(),
        scope_selection=selection,
        runtime_identity=runtime.runtime_identity(),
        protocol=_protocol(),
        checkpoint_rows=2,
    )
    run_e3_shuffled_control(
        work,
        construction_directory=construction,
        vector_bundle_directory=vectors,
        questions=_questions(),
        prompts=_prompts(),
        scope_selection=selection,
        runtime=runtime,
        protocol=_protocol(),
    )
    bundle = tmp_path / "shuffled-vectors"
    finalize_e3_shuffled_control_bundle(
        bundle,
        work_directory=work,
        construction_directory=construction,
        vector_bundle_directory=vectors,
        questions=_questions(),
        prompts=_prompts(),
        scope_selection=selection,
        protocol=_protocol(),
        allow_non_scientific=True,
    )

    tensor_path = bundle / "directions.npz"
    with np.load(tensor_path, allow_pickle=False) as arrays:
        directions = -arrays["directions"]
        correct = arrays["correct_counts"]
        incorrect = arrays["incorrect_counts"]
    with tensor_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            directions=directions,
            correct_counts=correct,
            incorrect_counts=incorrect,
        )
    metadata_path = bundle / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["directions_sha256"] = sha256_file(tensor_path)
    body = dict(metadata)
    body.pop("metadata_digest")
    metadata["metadata_digest"] = stable_hash(body)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(FrozenArtifactError, match="vectors differ"):
        verify_e3_shuffled_control_bundle(
            bundle,
            work_directory=work,
            construction_directory=construction,
            vector_bundle_directory=vectors,
            questions=_questions(),
            prompts=_prompts(),
            scope_selection=selection,
            protocol=_protocol(),
        )

    checkpoint = sorted((work / "checkpoints").glob("checkpoint-*.npz"))[0]
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
    with pytest.raises(FrozenArtifactError):
        verify_e3_shuffled_control_work(
            work,
            construction_directory=construction,
            vector_bundle_directory=vectors,
            questions=_questions(),
            prompts=_prompts(),
            scope_selection=selection,
            protocol=_protocol(),
        )


def test_fixed_control_materials_are_deterministic_and_content_verified(
    tmp_path: Path,
) -> None:
    construction, vectors, _runtime, initial_selection = _sources(tmp_path)
    dev_questions = tuple(
        Question(
            question_id=f"dev-{index}",
            benchmark="triviaqa",
            text=f"Development question {index}?",
            aliases=(f"development-answer-{index}",),
            split="T-dev",
        )
        for index in range(6)
    )
    screen_ids = select_e3_screen_questions(dev_questions, protocol=_protocol())
    selection = _scope_selection(
        initial_selection.source_plan_identity,
        question_ids=screen_ids,
        artifact_path=tmp_path / "fixed-scope-selection.json",
    )
    directory = tmp_path / "fixed-controls"
    result = write_e3_fixed_control_materials(
        directory,
        construction_directory=construction,
        vector_bundle_directory=vectors,
        questions=_questions(),
        prompts=_prompts(),
        scope_selection=selection,
        dev_questions=dev_questions,
        protocol=_protocol(),
    )
    assert result["dev_rows"] == 6
    assert result["hidden_width"] == 3
    assert result["scientific_eligible"] is False
    random = load_e3_fixed_control_direction(
        directory,
        control="random-norm",
        extraction_method="M1-R",
        expected_metadata_digest=str(result["metadata_digest"]),
        dev_question_ids=tuple(value.question_id for value in dev_questions),
    )
    gaussian = load_e3_fixed_control_direction(
        directory,
        control="gaussian",
        extraction_method="M1-P",
        question_id=dev_questions[1].question_id,
        expected_metadata_digest=str(result["metadata_digest"]),
        dev_question_ids=tuple(value.question_id for value in dev_questions),
    )
    assert random.flags.writeable is False
    assert gaussian.flags.writeable is False
    assert np.isclose(np.linalg.norm(random), 1.0)
    assert np.isclose(np.linalg.norm(gaussian), 1.0)

    path = directory / "gaussian.npy"
    values = np.load(path, allow_pickle=False)
    values[0, 0] *= -1
    with path.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["gaussian_sha256"] = sha256_file(path)
    body = dict(metadata)
    body.pop("metadata_digest")
    metadata["metadata_digest"] = stable_hash(body)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(FrozenArtifactError, match="Gaussian content differs"):
        verify_e3_fixed_control_materials(
            directory,
            construction_directory=construction,
            vector_bundle_directory=vectors,
            questions=_questions(),
            prompts=_prompts(),
            scope_selection=selection,
            dev_questions=dev_questions,
            protocol=_protocol(),
        )


def test_e3_execution_resolves_all_static_control_semantics(tmp_path: Path) -> None:
    construction, vectors, runtime, initial_selection = _sources(tmp_path)
    dev_questions = tuple(
        Question(
            question_id=f"dev-{index}",
            benchmark="triviaqa",
            text=f"Question {index}?",
            aliases=(f"answer-{index}",),
            split="T-dev",
        )
        for index in range(6)
    )
    screen_ids = select_e3_screen_questions(dev_questions, protocol=_protocol())
    selection = _scope_selection(
        initial_selection.source_plan_identity,
        question_ids=screen_ids,
        artifact_path=tmp_path / "execution-scope-selection.json",
    )
    shuffle_work = tmp_path / "shuffle-work"
    prepare_e3_shuffled_control_work(
        shuffle_work,
        construction_directory=construction,
        vector_bundle_directory=vectors,
        questions=_questions(),
        prompts=_prompts(),
        scope_selection=selection,
        runtime_identity=runtime.runtime_identity(),
        protocol=_protocol(),
        checkpoint_rows=2,
    )
    run_e3_shuffled_control(
        shuffle_work,
        construction_directory=construction,
        vector_bundle_directory=vectors,
        questions=_questions(),
        prompts=_prompts(),
        scope_selection=selection,
        runtime=runtime,
        protocol=_protocol(),
    )
    shuffle_bundle = tmp_path / "shuffle-bundle"
    finalize_e3_shuffled_control_bundle(
        shuffle_bundle,
        work_directory=shuffle_work,
        construction_directory=construction,
        vector_bundle_directory=vectors,
        questions=_questions(),
        prompts=_prompts(),
        scope_selection=selection,
        protocol=_protocol(),
        allow_non_scientific=True,
    )
    fixed = tmp_path / "fixed"
    write_e3_fixed_control_materials(
        fixed,
        construction_directory=construction,
        vector_bundle_directory=vectors,
        questions=_questions(),
        prompts=_prompts(),
        scope_selection=selection,
        dev_questions=dev_questions,
        protocol=_protocol(),
    )
    conditions = e3_control_conditions(selection.selected, protocol=_protocol())
    evaluation_questions = tuple(
        value for value in dev_questions if value.question_id in screen_ids
    )
    assets = load_e3_execution_assets(
        construction_directory=construction,
        vector_bundle_directory=vectors,
        questions=_questions(),
        prompts=_prompts(),
        scope_selection=selection,
        shuffled_work_directory=shuffle_work,
        shuffled_bundle_directory=shuffle_bundle,
        fixed_control_directory=fixed,
        dev_questions=dev_questions,
        evaluation_questions=evaluation_questions,
        application_prompts=_prompts(),
        conditions=conditions,
        stage="controls",
        render_runtime=runtime,
        protocol=_protocol(),
    )
    question = next(value for value in dev_questions if value.question_id == screen_ids[0])
    results = {
        condition.method: execute_e3_condition(
            runtime=runtime,
            assets=assets,
            condition=condition,
            question=question,
            prompts=_prompts(),
            max_new_tokens=8,
        )
        for condition in conditions
        if condition.extraction_method in {None, "M1-R"}
    }

    assert results["M0"].intervention_trace is None
    for method in (
        "M1-R",
        "opposite",
        "unrelated-layer",
        "zero-hook",
        "random-norm",
        "gaussian",
        "shuffled-label",
    ):
        assert results[method].intervention_trace is not None
    assert results["M1-R"].actual_delta_norm > 0
    opposite_trace = results["opposite"].intervention_trace
    unrelated_trace = results["unrelated-layer"].intervention_trace
    random_trace = results["random-norm"].intervention_trace
    gaussian_trace = results["gaussian"].intervention_trace
    shuffled_trace = results["shuffled-label"].intervention_trace
    assert opposite_trace is not None
    assert unrelated_trace is not None
    assert random_trace is not None
    assert gaussian_trace is not None
    assert shuffled_trace is not None
    assert opposite_trace["standardized_alpha"] < 0
    assert unrelated_trace["source_layer"] != unrelated_trace["target_layer"]
    assert results["zero-hook"].hook_applications == 0
    assert results["zero-hook"].actual_delta_norm == 0
    assert random_trace["direction_sha256"]
    assert gaussian_trace["direction_sha256"]
    assert shuffled_trace["direction_sha256"]

    normal_condition = next(value for value in conditions if value.method == "M1-R")
    resolved = assets.resolve(normal_condition, question_id=question.question_id)
    assert resolved is not None
    persisted = results["M1-R"].to_dict()
    persisted["outcome"] = (
        Outcome.INCORRECT.value
        if results["M1-R"].outcome is Outcome.CORRECT
        else Outcome.CORRECT.value
    )
    with pytest.raises(DataValidationError, match="grading"):
        expected_rendered = assets.rendered_prompt_hashes[
            f"{normal_condition.condition_id}:{question.question_id}"
        ]
        E3ExecutionResult.from_dict(
            persisted,
            question=question,
            condition=normal_condition,
            resolved=resolved,
            expected_rendered_prompt_sha256=expected_rendered[0],
            expected_prompt_token_ids_sha256=expected_rendered[1],
        )

    substituted_question = replace(question, text="Substituted question?")
    with pytest.raises(FrozenArtifactError, match="question differs"):
        execute_e3_condition(
            runtime=runtime,
            assets=assets,
            condition=normal_condition,
            question=substituted_question,
            prompts=_prompts(),
            max_new_tokens=8,
        )
    substituted_prompts = {
        **_prompts(),
        "P0-neutral": PromptSpec("P0-neutral", "Substituted system prompt"),
    }
    with pytest.raises(FrozenArtifactError, match="prompt differs"):
        execute_e3_condition(
            runtime=runtime,
            assets=assets,
            condition=normal_condition,
            question=question,
            prompts=substituted_prompts,
            max_new_tokens=8,
        )
    arbitrary = replace(normal_condition, standardized_alpha=123.0)
    with pytest.raises(FrozenArtifactError, match="outside the frozen"):
        execute_e3_condition(
            runtime=runtime,
            assets=assets,
            condition=arbitrary,
            question=question,
            prompts=_prompts(),
            max_new_tokens=8,
        )

    cross_condition = next(
        value
        for value in conditions
        if value.extraction_method == "M1-R" and value.control == "cross-prompt"
    )
    cross = assets.resolve(cross_condition, question_id=question.question_id)
    assert cross is not None
    prompt_index = 0  # P0 application RMS, despite the P2-trained direction.
    extraction_index = 0
    site_index = _protocol().candidate_sites.index(cross.target_site)
    layer_index = _protocol().candidate_layers.index(cross.target_layer)
    assert (
        cross.reference_rms
        == assets.reference_rms[prompt_index, extraction_index, site_index, layer_index]
    )
    for tensor in (
        assets.directions,
        assets.reference_rms,
        assets.shuffled_directions,
        assets.fixed_random_directions,
        assets.fixed_gaussian_directions,
        resolved.direction,
    ):
        assert tensor is not None
        with pytest.raises(ValueError):
            tensor.setflags(write=True)

    vector_metadata = vectors / "metadata.json"
    vector_metadata.write_bytes(vector_metadata.read_bytes() + b" ")
    with pytest.raises(FrozenArtifactError, match="artifact snapshot changed"):
        execute_e3_condition(
            runtime=runtime,
            assets=assets,
            condition=normal_condition,
            question=question,
            prompts=_prompts(),
            max_new_tokens=8,
        )


def test_e3_staged_runner_resumes_and_replays_selection_inputs(tmp_path: Path) -> None:
    construction, vectors, runtime, _selection_receipt = _sources(tmp_path)
    dev_questions = tuple(
        Question(
            question_id=f"runner-dev-{index}",
            benchmark="triviaqa",
            text=f"Question {index}?",
            aliases=(f"answer-{index}",),
            split="T-dev",
        )
        for index in range(6)
    )
    screen_ids = select_e3_screen_questions(dev_questions, protocol=_protocol())
    evaluation_questions = tuple(
        value for value in dev_questions if value.question_id in screen_ids
    )
    conditions = e3_geometry_conditions(_protocol())
    assets = load_e3_execution_assets(
        construction_directory=construction,
        vector_bundle_directory=vectors,
        questions=_questions(),
        prompts=_prompts(),
        dev_questions=dev_questions,
        evaluation_questions=evaluation_questions,
        application_prompts=_prompts(),
        conditions=conditions,
        stage="geometry",
        render_runtime=runtime,
        protocol=_protocol(),
    )
    work = tmp_path / "geometry-evaluation"
    prepare_e3_evaluation_work(
        work,
        stage="geometry",
        assets=assets,
        runtime_identity=runtime.runtime_identity(),
        max_new_tokens=8,
    )
    partial = run_e3_evaluation(
        work,
        stage="geometry",
        assets=assets,
        evaluation_questions=evaluation_questions,
        application_prompts=_prompts(),
        runtime=runtime,
        request_budget=3,
    )
    assert partial["records_completed"] == 3
    complete = run_e3_evaluation(
        work,
        stage="geometry",
        assets=assets,
        evaluation_questions=evaluation_questions,
        application_prompts=_prompts(),
        runtime=runtime,
    )
    assert complete["complete"] is True
    assert complete["records_completed"] == len(conditions) * 2
    assert (
        verify_e3_evaluation_work(
            work,
            stage="geometry",
            assets=assets,
            evaluation_questions=evaluation_questions,
            require_complete=True,
        )
        == complete
    )

    selection_inputs = e3_selection_inputs_from_work(
        work,
        stage="geometry",
        assets=assets,
        evaluation_questions=evaluation_questions,
    )
    selected = derive_e3_stage_selection(**selection_inputs)
    assert selected.stage == "geometry"
    assert selected.falsified is False


def test_e3_phase_finalizes_all_seven_stages_and_replays_gate(tmp_path: Path) -> None:
    construction, vectors, runtime, _initial_selection = _sources(tmp_path)
    protocol = _protocol()
    dev_questions = tuple(
        Question(
            question_id=f"phase-dev-{index}",
            benchmark="triviaqa",
            text=f"Question {index}?",
            aliases=(f"answer-{index}",),
            split="T-dev",
        )
        for index in range(protocol.dev_rows)
    )
    screen_ids = select_e3_screen_questions(dev_questions, protocol=protocol)
    screen_questions = tuple(
        question for question in dev_questions if question.question_id in screen_ids
    )
    application_prompts = {
        **_prompts(),
        "P3-forced-answer": PromptSpec(
            "P3-forced-answer",
            "Give your best short answer even when uncertain. Do not abstain.",
            permits_abstention=False,
            deployment_eligible=False,
        ),
    }
    stage_runs: dict[str, Path] = {}
    stage_assets: dict[str, Any] = {}
    stage_questions: dict[str, tuple[Question, ...]] = {}
    selection_receipts: dict[str, VerifiedE3StageSelection] = {}

    def execute_stage(
        stage: str,
        conditions: tuple[E3Condition, ...],
        questions: tuple[Question, ...],
        predecessor: VerifiedE3StageSelection | None,
        *,
        shuffle_work: Path | None = None,
        shuffle_bundle: Path | None = None,
        fixed: Path | None = None,
    ) -> None:
        assets = load_e3_execution_assets(
            construction_directory=construction,
            vector_bundle_directory=vectors,
            questions=_questions(),
            prompts=_prompts(),
            scope_selection=predecessor,
            shuffled_work_directory=shuffle_work,
            shuffled_bundle_directory=shuffle_bundle,
            fixed_control_directory=fixed,
            dev_questions=dev_questions,
            evaluation_questions=questions,
            application_prompts=application_prompts,
            conditions=conditions,
            stage=stage,
            render_runtime=runtime,
            protocol=protocol,
        )
        work = tmp_path / f"phase-{stage}"
        prepare_e3_evaluation_work(
            work,
            stage=stage,
            assets=assets,
            runtime_identity=runtime.runtime_identity(),
            selection_receipt=predecessor,
            max_new_tokens=8,
        )
        result = run_e3_evaluation(
            work,
            stage=stage,
            assets=assets,
            evaluation_questions=questions,
            application_prompts=application_prompts,
            runtime=runtime,
            selection_receipt=predecessor,
        )
        assert result["complete"] is True
        stage_runs[stage] = work
        stage_assets[stage] = assets
        stage_questions[stage] = questions

    def freeze_selection(
        stage: str, predecessor: VerifiedE3StageSelection | None
    ) -> VerifiedE3StageSelection:
        inputs = e3_selection_inputs_from_work(
            stage_runs[stage],
            stage=stage,
            assets=stage_assets[stage],
            evaluation_questions=screen_questions,
            selection_receipt=predecessor,
        )
        path = tmp_path / f"phase-{stage}-selection.json"
        write_e3_stage_selection(path, **inputs)
        return load_verified_e3_stage_selection(path, **inputs)

    geometry_conditions = e3_geometry_conditions(protocol)
    execute_stage("geometry", geometry_conditions, screen_questions, None)
    geometry_receipt = freeze_selection("geometry", None)
    selection_receipts["geometry"] = geometry_receipt

    alpha_conditions = e3_alpha_conditions(geometry_receipt.selected, protocol=protocol)
    execute_stage("alpha", alpha_conditions, screen_questions, geometry_receipt)
    alpha_receipt = freeze_selection("alpha", geometry_receipt)
    selection_receipts["alpha"] = alpha_receipt

    scope_conditions = e3_scope_conditions(alpha_receipt.selected, protocol=protocol)
    execute_stage("scope", scope_conditions, screen_questions, alpha_receipt)
    scope_receipt = freeze_selection("scope", alpha_receipt)
    selection_receipts["scope"] = scope_receipt

    shuffle_work = tmp_path / "phase-shuffle-work"
    prepare_e3_shuffled_control_work(
        shuffle_work,
        construction_directory=construction,
        vector_bundle_directory=vectors,
        questions=_questions(),
        prompts=_prompts(),
        scope_selection=scope_receipt,
        runtime_identity=runtime.runtime_identity(),
        protocol=protocol,
        checkpoint_rows=2,
    )
    run_e3_shuffled_control(
        shuffle_work,
        construction_directory=construction,
        vector_bundle_directory=vectors,
        questions=_questions(),
        prompts=_prompts(),
        scope_selection=scope_receipt,
        runtime=runtime,
        protocol=protocol,
    )
    shuffle_bundle = tmp_path / "phase-shuffle-bundle"
    finalize_e3_shuffled_control_bundle(
        shuffle_bundle,
        work_directory=shuffle_work,
        construction_directory=construction,
        vector_bundle_directory=vectors,
        questions=_questions(),
        prompts=_prompts(),
        scope_selection=scope_receipt,
        protocol=protocol,
        allow_non_scientific=True,
    )
    fixed = tmp_path / "phase-fixed-controls"
    write_e3_fixed_control_materials(
        fixed,
        construction_directory=construction,
        vector_bundle_directory=vectors,
        questions=_questions(),
        prompts=_prompts(),
        scope_selection=scope_receipt,
        dev_questions=dev_questions,
        protocol=protocol,
    )

    execute_stage(
        "controls",
        e3_control_conditions(scope_receipt.selected, protocol=protocol),
        screen_questions,
        scope_receipt,
        shuffle_work=shuffle_work,
        shuffle_bundle=shuffle_bundle,
        fixed=fixed,
    )
    execute_stage(
        "cross-prompt",
        e3_cross_prompt_conditions(scope_receipt.selected, protocol=protocol),
        screen_questions,
        scope_receipt,
    )
    execute_stage(
        "P3-diagnostic",
        e3_p3_conditions(scope_receipt.selected, protocol=protocol),
        screen_questions,
        scope_receipt,
    )
    execute_stage(
        "final",
        e3_final_conditions(scope_receipt.selected, protocol=protocol),
        dev_questions,
        scope_receipt,
        shuffle_work=shuffle_work,
        shuffle_bundle=shuffle_bundle,
        fixed=fixed,
    )

    phase = tmp_path / "phase-result"
    result = finalize_e3_phase(
        phase,
        stage_runs=stage_runs,
        stage_assets=stage_assets,
        stage_questions=stage_questions,
        selection_receipts=selection_receipts,
        allow_non_scientific=True,
    )
    assert result["valid"] is True
    assert result["status"] == "complete"
    assert result["primary_gate_passed"] is True
    assert (phase / "analysis-surface.json").is_file()
    loaded, digest = load_e3_analysis_surface(
        phase,
        expected_completion_digest=result["manifest_digest"],
        require_scientific=False,
    )
    assert loaded
    assert digest == result["manifest_digest"]
    assert (
        verify_e3_phase(
            phase,
            stage_runs=stage_runs,
            stage_assets=stage_assets,
            stage_questions=stage_questions,
            selection_receipts=selection_receipts,
        )
        == result
    )

    # Even a coordinated rewrite that refreshes every affected local digest must
    # not be accepted under the terminal identity supplied by a downstream phase.
    evidence_path = phase / "analysis-evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    original_output = evidence["stages"]["final"]["results"][0]["raw_output"]
    evidence["stages"]["final"]["results"][0]["raw_output"] = "coordinated rewrite"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_path = phase / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("manifest_digest")
    manifest["analysis_evidence_digest"] = stable_hash(evidence)
    manifest_path.write_text(
        json.dumps(
            {**manifest, "manifest_digest": stable_hash(manifest)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(FrozenArtifactError, match="completion receipt"):
        load_e3_analysis_surface(
            phase,
            expected_completion_digest=result["manifest_digest"],
            require_scientific=False,
        )

    # Restore the immutable terminal artifact before exercising deterministic
    # replay tamper detection independently.
    evidence["stages"]["final"]["results"][0]["raw_output"] = original_output
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest["analysis_evidence_digest"] = stable_hash(evidence)
    manifest_path.write_text(
        json.dumps(
            {**manifest, "manifest_digest": stable_hash(manifest)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    gate_path = phase / "primary-gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["passed"] = True
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    with pytest.raises(FrozenArtifactError, match="deterministic replay"):
        verify_e3_phase(
            phase,
            stage_runs=stage_runs,
            stage_assets=stage_assets,
            stage_questions=stage_questions,
            selection_receipts=selection_receipts,
        )


def test_e3_analysis_loader_requires_an_external_completion_anchor(tmp_path: Path) -> None:
    directory = tmp_path / "fabricated"
    directory.mkdir()
    for name in (
        "manifest.json",
        "stage-receipts.json",
        "condition-metrics.json",
        "primary-gate.json",
        "analysis-surface.json",
        "analysis-evidence.json",
    ):
        (directory / name).write_text("{}\n", encoding="utf-8")
    with pytest.raises(FrozenArtifactError, match="externally anchored"):
        load_e3_analysis_surface(directory, expected_completion_digest="not-a-digest")

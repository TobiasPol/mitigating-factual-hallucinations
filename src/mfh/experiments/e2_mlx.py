"""Scientific E2 orchestration from verified E1 outputs to frozen probes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from mfh.config import load_inference_protocol, load_model_spec, load_prompt_specs
from mfh.contracts import ModelSpec, Outcome, PromptSpec, Question, Runtime
from mfh.data.io import read_questions
from mfh.data.reviewed_splits import validate_reviewed_split_snapshot
from mfh.errors import ConfigurationError, DataValidationError, FrozenArtifactError
from mfh.experiments.e1_mlx import verify_e1_output_bundle
from mfh.experiments.e2_capture import (
    E1P0Source,
    prepare_e2_capture_work,
    run_e2_capture,
    verify_e2_capture_work,
)
from mfh.experiments.e2_phase import finalize_e2_phase_run
from mfh.experiments.e2_probes import (
    E2ProbeProtocol,
    VerifiedE2ProbeBundle,
    fit_e2_probe_bundle,
)
from mfh.experiments.e2_schedule import (
    E2CaptureProtocol,
    VerifiedE2Workspace,
    build_e2_schedule,
    verify_e2_workspace,
    write_e2_workspace,
)
from mfh.experiments.model_selection import (
    ACTIVE_RUNTIME_POLICY_RELATIVE,
    validate_active_model_spec,
    validate_active_study_artifact_paths,
)
from mfh.experiments.runner import PhaseCompletion, PhaseFalsification
from mfh.inference.mlx_preflight import validate_mlx_preflight_receipt
from mfh.inference.mlx_research import (
    MlxResearchRuntime,
    mlx_research_toolchain_identity,
)
from mfh.inference.transformers_snapshot import verify_transformers_snapshot
from mfh.provenance import sha256_file, sha256_path, stable_hash

_MODEL_CLASS = "mlx_lm.models.qwen3_5.Model"
_TOKENIZER_CLASS = "mlx_lm.tokenizer_utils.TokenizerWrapper"


@dataclass(frozen=True, slots=True)
class E2Prepared:
    model: ModelSpec
    snapshot: Path
    workspace: VerifiedE2Workspace
    capture_work: Path
    questions: Mapping[tuple[str, str], Question]
    prompts: Mapping[str, PromptSpec]
    e1_sources: Mapping[tuple[str, str], E1P0Source]
    runtime_identity: Mapping[str, Any]
    split_manifest_digest: str
    max_new_tokens: int


def _read_json(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot read {context}: {exc}") from exc
    if type(value) is not dict:
        raise DataValidationError(f"{context} must be a JSON object")
    return value


def _runtime_identity_from_receipt(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    identity = receipt.get("runtime_identity")
    if (
        not isinstance(identity, Mapping)
        or identity.get("model_class") != _MODEL_CLASS
        or identity.get("tokenizer_class") != _TOKENIZER_CLASS
        or identity.get("num_layers") != 64
        or identity.get("seed") != 17
    ):
        raise DataValidationError("MLX preflight receipt runtime identity differs")
    return MappingProxyType(dict(identity))


def _load_e1_p0_sources(output: Path) -> Mapping[tuple[str, str], E1P0Source]:
    values: dict[tuple[str, str], E1P0Source] = {}
    previous: str | None = None
    expected_keys = {
        "schema_version",
        "sequence",
        "condition_id",
        "question_id",
        "benchmark",
        "partition",
        "prompt_id",
        "outcome",
        "generation_record_digest",
        "ledger_record_digest",
        "grader_record_digest",
        "previous_label_digest",
        "label_digest",
    }
    try:
        with (output / "outcome-labels.jsonl").open(encoding="utf-8") as handle:
            for sequence, line in enumerate(handle):
                row = json.loads(line)
                if type(row) is not dict or set(row) != expected_keys:
                    raise DataValidationError("E1 outcome-label row schema differs")
                body = dict(row)
                digest = body.pop("label_digest")
                if (
                    type(digest) is not str
                    or digest != stable_hash(body)
                    or row["previous_label_digest"] != previous
                    or type(row["sequence"]) is not int
                    or row["sequence"] != sequence
                ):
                    raise DataValidationError("E1 outcome-label chain differs")
                previous = digest
                if row["prompt_id"] != "P0-neutral":
                    continue
                source = E1P0Source(
                    benchmark=row["benchmark"],
                    question_id=row["question_id"],
                    outcome=Outcome(row["outcome"]),
                    generation_record_sha256=row["generation_record_digest"],
                )
                key = (source.benchmark, source.question_id)
                if key in values:
                    raise DataValidationError("E1 P0 labels contain a duplicate source")
                values[key] = source
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise DataValidationError(f"cannot load verified E1 P0 sources: {exc}") from exc
    if len(values) != 6_600:
        raise DataValidationError("E1 P0 source count differs from the frozen 6,600 rows")
    return MappingProxyType(values)


def _load_questions(
    splits: Path,
) -> tuple[
    tuple[Question, ...],
    tuple[Question, ...],
    tuple[Question, ...],
    tuple[Question, ...],
    Mapping[tuple[str, str], Question],
]:
    controller = tuple(read_questions(splits / "T-controller.jsonl"))
    dev = tuple(read_questions(splits / "T-dev.jsonl"))
    simpleqa = tuple(read_questions(splits / "simpleqa-eval.jsonl"))
    aa = tuple(read_questions(splits / "aa-eval.jsonl"))
    all_questions = (*controller, *dev, *simpleqa, *aa)
    mapping = {
        (question.benchmark, question.question_id): question for question in all_questions
    }
    if len(mapping) != len(all_questions):
        raise DataValidationError("E2 question keys are not globally unique by benchmark")
    return controller, dev, simpleqa, aa, MappingProxyType(mapping)


def _prepare_live_inputs(
    *,
    splits_directory: Path,
    expected_splits_manifest_digest: str,
    e1_output: Path,
    e1_work: Path,
    e1_ledger: Path,
    expected_e1_manifest_digest: str,
    model_config: Path,
    snapshot_directory: Path,
    snapshot_manifest: Path,
    runtime_config: Path,
    prompt_config: Path,
    inference_config: Path,
    study_config: Path,
) -> tuple[
    ModelSpec,
    Mapping[str, PromptSpec],
    Mapping[tuple[str, str], E1P0Source],
    tuple[Question, ...],
    tuple[Question, ...],
    tuple[Question, ...],
    tuple[Question, ...],
    Mapping[tuple[str, str], Question],
    Mapping[str, str],
    Mapping[str, Any],
    int,
]:
    split_manifest = validate_reviewed_split_snapshot(splits_directory)
    if split_manifest.get("manifest_digest") != expected_splits_manifest_digest:
        raise DataValidationError("E2 reviewed split manifest differs")
    verify_e1_output_bundle(
        e1_output,
        work_directory=e1_work,
        ledger_directory=e1_ledger,
        study_config=study_config,
        expected_manifest_digest=expected_e1_manifest_digest,
    )
    sources = _load_e1_p0_sources(e1_output)
    controller, dev, simpleqa, aa, questions = _load_questions(splits_directory)
    model = load_model_spec(model_config)
    validate_active_model_spec(model)
    if model.runtime is not Runtime.MLX or model.num_layers != 64:
        raise ConfigurationError("E2 requires the sole 64-layer MLX model")
    snapshot = verify_transformers_snapshot(
        model, snapshot_directory, snapshot_manifest
    )
    prompts = {value.prompt_id: value for value in load_prompt_specs(prompt_config)}
    if not {"P0-neutral", "P3-forced-answer"} <= set(prompts):
        raise ConfigurationError("E2 prompt config lacks P0 or P3")
    selected_prompts = {
        name: prompts[name] for name in ("P0-neutral", "P3-forced-answer")
    }
    inference = load_inference_protocol(inference_config)
    if (
        inference.temperature != 0
        or inference.do_sample
        or inference.max_new_tokens != 48
        or inference.thinking_enabled
        or inference.retrieval_enabled
        or inference.tools_enabled
        or 17 not in inference.seeds
    ):
        raise ConfigurationError("E2 inference config differs from deterministic decoding")
    project_root = Path(__file__).parents[3]
    try:
        snapshot_relative = snapshot_directory.relative_to(project_root).as_posix()
        receipt_relative = runtime_config.relative_to(project_root).as_posix()
    except ValueError as exc:
        raise DataValidationError(
            "E2 model snapshot and preflight receipt must stay in the project"
        ) from exc
    input_fingerprints = {
        "e1_output": sha256_path(e1_output),
        "reviewed_splits": sha256_path(splits_directory),
        "model_snapshot": snapshot["snapshot_digest"],
        "snapshot_manifest": sha256_file(snapshot_manifest),
        "runtime_config": sha256_file(runtime_config),
        "prompt_config": sha256_file(prompt_config),
        "inference_config": sha256_file(inference_config),
        "study_config": sha256_file(study_config),
        "project_lock": sha256_file(project_root / "uv.lock"),
        "project_metadata": sha256_file(project_root / "pyproject.toml"),
    }
    receipt = validate_mlx_preflight_receipt(
        runtime_config,
        project_root=project_root,
        model_config=model_config,
        snapshot_directory=snapshot_directory,
        snapshot_manifest=snapshot_manifest,
        runtime_policy=project_root / ACTIVE_RUNTIME_POLICY_RELATIVE,
    )
    software = receipt.get("software")
    if not isinstance(software, Mapping) or not isinstance(
        software.get("toolchain"), Mapping
    ):
        raise DataValidationError("MLX preflight receipt software/toolchain differs")
    live_toolchain = dict(mlx_research_toolchain_identity())
    if live_toolchain != software.get("toolchain"):
        raise DataValidationError("live MLX research toolchain differs from preflight")
    research_provenance = {
        "schema_version": 3,
        "model_repository": model.repository,
        "model_revision": model.revision,
        "quantization": model.quantization,
        "verified_snapshot_digest": snapshot["snapshot_digest"],
        "snapshot_manifest_sha256": sha256_file(snapshot_manifest),
        "runtime_preflight_receipt_digest": receipt["receipt_digest"],
        "runtime_preflight_receipt_sha256": sha256_file(runtime_config),
        "runtime_preflight_receipt_relative": receipt_relative,
        "model_snapshot_relative": snapshot_relative,
        "runtime_policy_digest": receipt["policy_digest"],
        "runtime_policy_sha256": receipt["policy_sha256"],
        "preflight_intervention_digest": stable_hash(receipt["intervention"]),
        "research_toolchain_digest": stable_hash(live_toolchain),
        "pyproject_sha256": input_fingerprints["project_metadata"],
        "uv_lock_sha256": input_fingerprints["project_lock"],
        "tokenizer_sha256": sha256_file(snapshot_directory / "tokenizer.json"),
        "chat_template_sha256": sha256_file(snapshot_directory / "chat_template.jinja"),
    }
    runtime_identity = {
        **dict(_runtime_identity_from_receipt(receipt)),
        "research_provenance": research_provenance,
        "research_toolchain": live_toolchain,
    }
    return (
        model,
        MappingProxyType(selected_prompts),
        sources,
        controller,
        dev,
        simpleqa,
        aa,
        questions,
        MappingProxyType(input_fingerprints),
        MappingProxyType(runtime_identity),
        inference.max_new_tokens,
    )


def _prepared(
    *,
    splits_directory: str | Path,
    expected_splits_manifest_digest: str,
    e1_output: str | Path,
    e1_work: str | Path,
    e1_ledger: str | Path,
    expected_e1_manifest_digest: str,
    model_config: str | Path,
    snapshot_directory: str | Path,
    snapshot_manifest: str | Path,
    runtime_config: str | Path,
    workspace_directory: str | Path,
    capture_work_directory: str | Path,
    prompt_config: str | Path,
    inference_config: str | Path,
    study_config: str | Path,
) -> E2Prepared:
    paths = {
        "splits_directory": Path(splits_directory).absolute(),
        "e1_output": Path(e1_output).absolute(),
        "e1_work": Path(e1_work).absolute(),
        "e1_ledger": Path(e1_ledger).absolute(),
        "model_config": Path(model_config).absolute(),
        "snapshot_directory": Path(snapshot_directory).absolute(),
        "snapshot_manifest": Path(snapshot_manifest).absolute(),
        "runtime_config": Path(runtime_config).absolute(),
        "prompt_config": Path(prompt_config).absolute(),
        "inference_config": Path(inference_config).absolute(),
        "study_config": Path(study_config).absolute(),
    }
    validate_active_study_artifact_paths(
        {
            "E1 work directory": paths["e1_work"],
            "E1 ledger directory": paths["e1_ledger"],
            "E1 output directory": paths["e1_output"],
            "E2 workspace directory": workspace_directory,
            "E2 capture work directory": capture_work_directory,
        }
    )
    (
        model,
        prompts,
        sources,
        controller,
        dev,
        simpleqa,
        aa,
        questions,
        input_fingerprints,
        runtime_identity,
        max_new_tokens,
    ) = _prepare_live_inputs(
        **paths,
        expected_splits_manifest_digest=expected_splits_manifest_digest,
        expected_e1_manifest_digest=expected_e1_manifest_digest,
    )
    workspace = verify_e2_workspace(workspace_directory)
    expected_schedule = build_e2_schedule(
        controller=controller,
        dev=dev,
        simpleqa=simpleqa,
        aa=aa,
        e1_p0_outcomes={key: value.outcome for key, value in sources.items()},
    )
    if (
        workspace.schedule != expected_schedule
        or workspace.input_fingerprints != input_fingerprints
        or workspace.activation_spec.model_repository != model.repository
        or workspace.activation_spec.model_revision != model.revision
        or workspace.activation_spec.hidden_width != 5_120
    ):
        raise FrozenArtifactError("E2 workspace differs from verified live inputs")
    return E2Prepared(
        model=model,
        snapshot=paths["snapshot_directory"],
        workspace=workspace,
        capture_work=Path(capture_work_directory).absolute(),
        questions=questions,
        prompts=prompts,
        e1_sources=sources,
        runtime_identity=runtime_identity,
        split_manifest_digest=expected_splits_manifest_digest,
        max_new_tokens=max_new_tokens,
    )


def prepare_e2_mlx(
    *,
    workspace_directory: str | Path,
    capture_work_directory: str | Path,
    splits_directory: str | Path,
    expected_splits_manifest_digest: str,
    e1_output: str | Path,
    e1_work: str | Path,
    e1_ledger: str | Path,
    expected_e1_manifest_digest: str,
    model_config: str | Path,
    snapshot_directory: str | Path,
    snapshot_manifest: str | Path,
    runtime_config: str | Path,
    prompt_config: str | Path = "configs/prompts/primary.yaml",
    inference_config: str | Path = "configs/experiments/core.yaml",
    study_config: str | Path = "configs/experiments/phases.yaml",
    shard_rows: int = 64,
) -> E2Prepared:
    paths = {
        "splits_directory": Path(splits_directory).absolute(),
        "e1_output": Path(e1_output).absolute(),
        "e1_work": Path(e1_work).absolute(),
        "e1_ledger": Path(e1_ledger).absolute(),
        "model_config": Path(model_config).absolute(),
        "snapshot_directory": Path(snapshot_directory).absolute(),
        "snapshot_manifest": Path(snapshot_manifest).absolute(),
        "runtime_config": Path(runtime_config).absolute(),
        "prompt_config": Path(prompt_config).absolute(),
        "inference_config": Path(inference_config).absolute(),
        "study_config": Path(study_config).absolute(),
    }
    validate_active_study_artifact_paths(
        {
            "E1 work directory": paths["e1_work"],
            "E1 ledger directory": paths["e1_ledger"],
            "E1 output directory": paths["e1_output"],
            "E2 workspace directory": workspace_directory,
            "E2 capture work directory": capture_work_directory,
        }
    )
    (
        model,
        prompts,
        sources,
        controller,
        dev,
        simpleqa,
        aa,
        questions,
        input_fingerprints,
        runtime_identity,
        max_new_tokens,
    ) = _prepare_live_inputs(
        **paths,
        expected_splits_manifest_digest=expected_splits_manifest_digest,
        expected_e1_manifest_digest=expected_e1_manifest_digest,
    )
    schedule = build_e2_schedule(
        controller=controller,
        dev=dev,
        simpleqa=simpleqa,
        aa=aa,
        e1_p0_outcomes={key: value.outcome for key, value in sources.items()},
        protocol=E2CaptureProtocol(),
    )
    workspace = write_e2_workspace(
        workspace_directory,
        schedule=schedule,
        protocol=E2CaptureProtocol(),
        model=model,
        hidden_width=5_120,
        input_fingerprints=input_fingerprints,
    )
    capture_work = Path(capture_work_directory).absolute()
    prepare_e2_capture_work(
        capture_work,
        workspace=workspace,
        questions=questions,
        prompts=prompts,
        e1_sources=sources,
        expected_runtime_identity=runtime_identity,
        shard_rows=shard_rows,
        max_new_tokens=max_new_tokens,
    )
    return E2Prepared(
        model=model,
        snapshot=paths["snapshot_directory"],
        workspace=workspace,
        capture_work=capture_work,
        questions=questions,
        prompts=prompts,
        e1_sources=sources,
        runtime_identity=runtime_identity,
        split_manifest_digest=expected_splits_manifest_digest,
        max_new_tokens=max_new_tokens,
    )


def run_e2_mlx_capture(
    *,
    request_budget: int | None = None,
    **inputs: Any,
) -> Mapping[str, Any]:
    prepared = _prepared(**inputs)
    runtime = MlxResearchRuntime.from_spec(
        prepared.model,
        snapshot_path=prepared.snapshot,
        seed=17,
        research_provenance=prepared.runtime_identity["research_provenance"],
    )
    try:
        return run_e2_capture(
            prepared.capture_work,
            workspace=prepared.workspace,
            questions=prepared.questions,
            prompts=prepared.prompts,
            e1_sources=prepared.e1_sources,
            runtime=runtime,
            request_budget=request_budget,
        )
    finally:
        runtime.close()


def verify_e2_mlx_capture(
    *,
    require_complete: bool = False,
    **inputs: Any,
) -> Mapping[str, Any]:
    prepared = _prepared(**inputs)
    return verify_e2_capture_work(
        prepared.capture_work,
        workspace=prepared.workspace,
        questions=prepared.questions,
        prompts=prepared.prompts,
        e1_sources=prepared.e1_sources,
        require_complete=require_complete,
    )


def fit_e2_mlx_probes(
    output_directory: str | Path,
    *,
    protocol: E2ProbeProtocol | None = None,
    probe_work_directory: str | Path | None = None,
    **inputs: Any,
) -> VerifiedE2ProbeBundle:
    mutable_paths: dict[str, str | Path] = {"E2 probe bundle": output_directory}
    if probe_work_directory is not None:
        mutable_paths["E2 probe work"] = probe_work_directory
    validate_active_study_artifact_paths(mutable_paths)
    prepared = _prepared(**inputs)
    verify_e2_capture_work(
        prepared.capture_work,
        workspace=prepared.workspace,
        questions=prepared.questions,
        prompts=prepared.prompts,
        e1_sources=prepared.e1_sources,
        require_complete=True,
    )
    prompt_hashes = {
        name: hashlib.sha256(prompt.text.encode("utf-8")).hexdigest()
        for name, prompt in prepared.prompts.items()
    }
    return fit_e2_probe_bundle(
        output_directory,
        workspace=prepared.workspace,
        split_manifest_digest=prepared.split_manifest_digest,
        prompt_template_sha256=prompt_hashes,
        protocol=protocol,
        work_directory=probe_work_directory,
    )


def finalize_e2_mlx_phase(
    output_directory: str | Path,
    *,
    probe_bundle_directory: str | Path,
    expected_workspace_plan_identity: str,
    expected_capture_plan_identity: str,
    expected_probe_manifest_digest: str,
    **inputs: Any,
) -> PhaseCompletion | PhaseFalsification:
    validate_active_study_artifact_paths(
        {
            "E2 phase ledger": output_directory,
            "E2 probe bundle": probe_bundle_directory,
        }
    )
    prepared = _prepared(**inputs)
    return finalize_e2_phase_run(
        output_directory,
        workspace_directory=prepared.workspace.directory,
        expected_workspace_plan_identity=expected_workspace_plan_identity,
        capture_work_directory=prepared.capture_work,
        expected_capture_plan_identity=expected_capture_plan_identity,
        probe_bundle_directory=probe_bundle_directory,
        expected_probe_manifest_digest=expected_probe_manifest_digest,
        questions=prepared.questions,
        prompts=prepared.prompts,
        e1_sources=prepared.e1_sources,
        e1_output_directory=inputs["e1_output"],
        e1_phase_run=inputs["e1_ledger"],
        split_manifest_digest=prepared.split_manifest_digest,
        study_config=inputs["study_config"],
    )

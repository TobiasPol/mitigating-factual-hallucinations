"""Command-line entry point for reproducible local research operations."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mfh.config import (
    load_benchmark_spec,
    load_inference_protocol,
    load_model_spec,
    load_prompt_specs,
    load_semantic_contamination_protocol,
    load_yaml,
)
from mfh.data.contamination import find_overlaps
from mfh.data.io import read_generation_records, read_questions, write_question_bundle
from mfh.data.splits import (
    ResearchSplit,
    SplitPlan,
    exclude_exact_duplicate_groups,
    make_research_splits,
)
from mfh.evaluation.metrics import metric_bundle
from mfh.evaluation.official import load_official_grader_spec
from mfh.experiments.protocol import load_study_protocol
from mfh.experiments.runner import PhaseRunLedger
from mfh.provenance import (
    canonical_json,
    read_frozen_manifest,
    sha256_file,
    sha256_path,
    stable_hash,
)


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str))


def _validate_config(path: Path) -> int:
    raw = load_yaml(path)
    if "model" in raw:
        parsed: Any = load_model_spec(path)
        canonical: Any = json.loads(canonical_json(parsed))
    elif "benchmark" in raw:
        parsed = load_benchmark_spec(path)
        canonical = json.loads(canonical_json(parsed))
    elif "prompts" in raw:
        parsed = load_prompt_specs(path)
        canonical = json.loads(canonical_json(parsed))
    elif "inference" in raw:
        parsed = load_inference_protocol(path)
        canonical = json.loads(canonical_json(parsed))
    elif "grader" in raw:
        grader = load_official_grader_spec(path)
        canonical = {
            "benchmark": grader.benchmark,
            "grader_model": grader.grader_model,
            "grader_model_revision": grader.grader_model_revision,
            "prompt_sha256": grader.prompt_sha256,
            "grader_digest": grader.digest,
        }
    elif "phases" in raw and "study_id" in raw:
        study = load_study_protocol(path)
        canonical = {
            "study_id": study.study_id,
            "study_protocol_digest": study.digest,
            "phases": [phase.to_dict() for phase in study.phases],
        }
    elif "analysis" in raw:
        from mfh.analysis.protocol import load_analysis_protocol

        analysis = load_analysis_protocol(path)
        canonical = {**analysis.to_dict(), "analysis_protocol_digest": analysis.digest}
    elif "semantic_contamination" in raw:
        semantic = load_semantic_contamination_protocol(path)
        canonical = {**semantic.to_dict(), "protocol_digest": semantic.digest}
    else:
        raise ValueError(f"cannot infer supported configuration type from {path}")
    _print({"valid": True, "canonical": canonical})
    return 0


def _split(args: argparse.Namespace) -> int:
    questions = tuple(read_questions(args.input))
    curation_report: dict[str, object] | None = None
    curation_summary: dict[str, object] | None = None
    metadata_files: dict[str, Mapping[str, Any]] = {}
    if args.exclude_exact_duplicate_groups:
        curation = exclude_exact_duplicate_groups(questions)
        questions = curation.questions
        curation_body: dict[str, object] = {
            "schema_version": 1,
            "source_questions_sha256": sha256_file(args.input),
            "curation": curation.report.to_dict(),
        }
        curation_report = {
            **curation_body,
            "manifest_digest": stable_hash(curation_body),
        }
        curation_summary = {
            "source_questions_sha256": curation_report["source_questions_sha256"],
            "manifest_digest": curation_report["manifest_digest"],
            **{
                key: value
                for key, value in curation.report.to_dict().items()
                if key != "exclusions"
            },
        }
        metadata_files["curation-report.json"] = curation_report
    result = make_research_splits(
        questions,
        SplitPlan(
            steer=args.steer,
            controller=args.controller,
            dev=args.dev,
            test=args.test,
            seed=args.seed,
        ),
        require_exact_sizes=not args.allow_underfill,
    )
    write_question_bundle(
        args.output,
        {f"{split.value}.jsonl": result.splits[split] for split in ResearchSplit},
        metadata_files=metadata_files,
        overwrite=args.overwrite,
    )
    output: dict[str, object] = dict(asdict(result.report))
    output["bundle_sha256"] = sha256_path(args.output)
    if curation_summary is not None:
        output["curation"] = curation_summary
    _print(output)
    return 0


def _overlap(args: argparse.Namespace) -> int:
    report = find_overlaps(
        read_questions(args.source),
        read_questions(args.target),
        ngram_threshold=args.ngram_threshold,
    )
    _print(
        {
            "source_count": report.source_count,
            "target_count": report.target_count,
            "source_ids_to_remove": report.source_ids_to_remove,
            "matches": [asdict(match) for match in report.top(args.limit)],
        }
    )
    return 0


def _contamination_scan(args: argparse.Namespace) -> int:
    from mfh.data.semantic_contamination import stderr_progress, write_contamination_bundle

    manifest = write_contamination_bundle(
        args.output,
        protocol=load_semantic_contamination_protocol(args.config),
        model_directory=args.model_directory,
        triviaqa_source=args.triviaqa_source,
        target_sources=args.target,
        progress=stderr_progress,
    )
    _print(
        {
            "valid": True,
            "manifest_digest": manifest["manifest_digest"],
            "protocol_digest": manifest["protocol_digest"],
            "counts": manifest["counts"],
            "manual_review": manifest["manual_review"],
        }
    )
    return 0


def _verify_contamination_scan(args: argparse.Namespace) -> int:
    from mfh.data.semantic_contamination import stderr_progress, verify_contamination_bundle

    manifest = verify_contamination_bundle(
        args.bundle,
        expected_protocol=load_semantic_contamination_protocol(args.config),
        model_directory=args.model_directory,
        triviaqa_source=args.triviaqa_source,
        target_sources=args.target,
        expected_manifest_digest=args.expected_manifest_digest,
        replay_embeddings=args.replay_embeddings,
        progress=stderr_progress,
    )
    _print(
        {
            "valid": True,
            "manifest_digest": manifest["manifest_digest"],
            "protocol_digest": manifest["protocol_digest"],
            "counts": manifest["counts"],
            "replayed_lexical": True,
            "replayed_embeddings": args.replay_embeddings,
        }
    )
    return 0


def _contamination_review_inputs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "contamination_directory": args.contamination_bundle,
        "expected_protocol": load_semantic_contamination_protocol(args.config),
        "model_directory": args.model_directory,
        "triviaqa_source": args.triviaqa_source,
        "target_sources": args.target,
        "expected_contamination_manifest_digest": args.expected_contamination_manifest_digest,
    }


def _prepare_contamination_review(args: argparse.Namespace) -> int:
    from mfh.data.contamination_review import prepare_contamination_review_queue

    manifest = prepare_contamination_review_queue(
        args.output,
        seed=args.seed,
        **_contamination_review_inputs(args),
    )
    _print(
        {
            "valid": True,
            "manifest_digest": manifest["manifest_digest"],
            "review_count": manifest["review_count"],
            "status": manifest["status"],
            "blind_items": str(args.output / "blind-items.jsonl"),
            "annotation_template": str(args.output / "annotation-template.csv"),
            "rubric": str(args.output / "rubric.md"),
        }
    )
    return 0


def _verify_contamination_review_queue(args: argparse.Namespace) -> int:
    from mfh.data.contamination_review import verify_contamination_review_queue

    manifest = verify_contamination_review_queue(
        args.review_queue,
        expected_manifest_digest=args.expected_review_queue_manifest_digest,
        **_contamination_review_inputs(args),
    )
    _print(
        {
            "valid": True,
            "manifest_digest": manifest["manifest_digest"],
            "review_count": manifest["review_count"],
            "status": manifest["status"],
        }
    )
    return 0


def _finalize_contamination_review(args: argparse.Namespace) -> int:
    from mfh.data.contamination_review import finalize_contamination_review

    manifest = finalize_contamination_review(
        args.output,
        review_queue_directory=args.review_queue,
        expected_review_queue_manifest_digest=args.expected_review_queue_manifest_digest,
        annotations=args.annotations,
        reviewer_attestation=args.reviewer_attestation,
        **_contamination_review_inputs(args),
    )
    _print(
        {
            "valid": True,
            "manifest_digest": manifest["manifest_digest"],
            "counts": manifest["counts"],
            "status": manifest["status"],
            "scientific_eligible": manifest["scientific_eligible"],
        }
    )
    return 0


def _verify_contamination_review_result(args: argparse.Namespace) -> int:
    from mfh.data.contamination_review import verify_contamination_review_result

    manifest = verify_contamination_review_result(
        args.result,
        expected_manifest_digest=args.expected_result_manifest_digest,
        review_queue_directory=args.review_queue,
        expected_review_queue_manifest_digest=args.expected_review_queue_manifest_digest,
        **_contamination_review_inputs(args),
    )
    _print(
        {
            "valid": True,
            "manifest_digest": manifest["manifest_digest"],
            "counts": manifest["counts"],
            "status": manifest["status"],
            "scientific_eligible": manifest["scientific_eligible"],
        }
    )
    return 0


def _reviewed_split_plan(args: argparse.Namespace) -> SplitPlan:
    return SplitPlan(
        steer=args.steer,
        controller=args.controller,
        dev=args.dev,
        test=args.test,
        seed=args.seed,
    )


def _prepare_reviewed_splits(args: argparse.Namespace) -> int:
    from mfh.data.reviewed_splits import write_reviewed_split_bundle

    manifest = write_reviewed_split_bundle(
        args.output,
        review_result_directory=args.review_result,
        expected_review_result_manifest_digest=args.expected_review_result_manifest_digest,
        review_queue_directory=args.review_queue,
        expected_review_queue_manifest_digest=args.expected_review_queue_manifest_digest,
        review_inputs=_contamination_review_inputs(args),
        plan=_reviewed_split_plan(args),
    )
    _print(
        {
            "valid": True,
            "manifest_digest": manifest["manifest_digest"],
            "split_report": manifest["split_report"],
            "status": manifest["status"],
            "scientific_eligible": manifest["scientific_eligible"],
        }
    )
    return 0


def _verify_reviewed_splits(args: argparse.Namespace) -> int:
    from mfh.data.reviewed_splits import verify_reviewed_split_bundle

    manifest = verify_reviewed_split_bundle(
        args.splits,
        expected_manifest_digest=args.expected_split_manifest_digest,
        review_result_directory=args.review_result,
        expected_review_result_manifest_digest=args.expected_review_result_manifest_digest,
        review_queue_directory=args.review_queue,
        expected_review_queue_manifest_digest=args.expected_review_queue_manifest_digest,
        review_inputs=_contamination_review_inputs(args),
        plan=_reviewed_split_plan(args),
    )
    _print(
        {
            "valid": True,
            "manifest_digest": manifest["manifest_digest"],
            "split_report": manifest["split_report"],
            "status": manifest["status"],
            "scientific_eligible": manifest["scientific_eligible"],
        }
    )
    return 0


def _add_reviewed_split_evidence_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("review_result", type=Path)
    parser.add_argument("review_queue", type=Path)
    parser.add_argument("contamination_bundle", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("model_directory", type=Path)
    parser.add_argument("triviaqa_source", type=Path)
    parser.add_argument("--target", type=Path, action="append", required=True)
    parser.add_argument("--expected-contamination-manifest-digest", required=True)
    parser.add_argument("--expected-review-result-manifest-digest", required=True)
    parser.add_argument("--expected-review-queue-manifest-digest", required=True)
    parser.add_argument("--steer", type=int, default=30_000)
    parser.add_argument("--controller", type=int, default=5_000)
    parser.add_argument("--dev", type=int, default=5_000)
    parser.add_argument("--test", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=17)


def _prepare_runtime_validation(args: argparse.Namespace) -> int:
    from mfh.data.runtime_validation import write_runtime_validation_bundle

    manifest = write_runtime_validation_bundle(
        args.output,
        reserved_source=args.reserved_source,
        parent_split_manifest_digest=args.parent_split_manifest_digest,
        contamination_manifest_digest=args.contamination_manifest_digest,
        seed=args.seed,
    )
    _print(
        {
            "valid": True,
            "manifest_digest": manifest["manifest_digest"],
            "selection": manifest["selection"],
            "scientific_status": manifest["scientific_status"],
        }
    )
    return 0


def _verify_runtime_validation(args: argparse.Namespace) -> int:
    from mfh.data.runtime_validation import verify_runtime_validation_bundle

    manifest = verify_runtime_validation_bundle(
        args.bundle,
        reserved_source=args.reserved_source,
        expected_manifest_digest=args.expected_manifest_digest,
        parent_split_manifest_digest=args.parent_split_manifest_digest,
        contamination_manifest_digest=args.contamination_manifest_digest,
    )
    _print(
        {
            "valid": True,
            "manifest_digest": manifest["manifest_digest"],
            "selection": manifest["selection"],
            "scientific_status": manifest["scientific_status"],
        }
    )
    return 0


def _metrics(args: argparse.Namespace) -> int:
    records = list(read_generation_records(args.records))
    _print(
        metric_bundle(
            (record.outcome for record in records), partial_credit=args.partial_credit
        ).to_dict()
    )
    return 0


def _verify_manifest(path: Path) -> int:
    manifest = read_frozen_manifest(path)
    _print({"valid": True, "manifest_digest": manifest.manifest_digest})
    return 0


def _validate_study(args: argparse.Namespace) -> int:
    from mfh.analysis.protocol import load_analysis_protocol

    study = load_study_protocol(args.study_protocol)
    analysis = load_analysis_protocol(args.analysis_protocol)
    analysis.verify_research_plan(args.research_plan)
    _print(
        {
            "valid": True,
            "study_protocol_digest": study.digest,
            "analysis_protocol_digest": analysis.digest,
            "research_plan_sha256": analysis.research_plan_sha256,
            "phases": [phase.value for phase in study.by_phase],
        }
    )
    return 0


def _verify_grader(args: argparse.Namespace) -> int:
    grader = load_official_grader_spec(args.config)
    grader.verify_source_artifact(args.source_artifact)
    _print(
        {
            "valid": True,
            "benchmark": grader.benchmark,
            "grader_digest": grader.digest,
            "source_artifact_sha256": grader.source_artifact_sha256,
        }
    )
    return 0


def _freeze_e1_graders(args: argparse.Namespace) -> int:
    from mfh.experiments.grader_bundle import write_e1_grader_bundle

    manifest = write_e1_grader_bundle(args.output)
    _print(
        {
            "valid": True,
            "manifest_digest": manifest["manifest_digest"],
            "bundle_sha256": manifest["bundle_sha256"],
            "grader_fingerprints": manifest["grader_fingerprints"],
            "routes": manifest["routes"],
        }
    )
    return 0


def _verify_e1_graders(args: argparse.Namespace) -> int:
    from mfh.experiments.grader_bundle import verify_e1_grader_bundle

    manifest = verify_e1_grader_bundle(
        args.bundle,
        expected_manifest_digest=args.expected_manifest_digest,
    )
    _print(
        {
            "valid": True,
            "manifest_digest": manifest["manifest_digest"],
            "grader_fingerprints": manifest["grader_fingerprints"],
            "routes": manifest["routes"],
        }
    )
    return 0


def _e1_inputs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "splits_directory": args.splits,
        "expected_splits_manifest_digest": args.expected_split_manifest_digest,
        "grader_bundle": args.grader_bundle,
        "expected_grader_manifest_digest": args.expected_grader_manifest_digest,
        "model_config": args.model_config,
        "snapshot_directory": args.snapshot_directory,
        "snapshot_manifest": args.snapshot_manifest,
        "runtime_config": args.runtime_config,
        "work_directory": args.work,
        "ledger_directory": args.ledger,
        "e0_run": args.e0_run,
        "prompt_config": args.prompt_config,
        "inference_config": args.inference_config,
        "study_config": args.study_config,
    }


def _external_checkpoint(args: argparse.Namespace) -> str | None:
    if not args.resume:
        return None
    if args.checkpoint_file is None:
        raise ValueError("--resume requires --checkpoint-file")
    try:
        value = json.loads(args.checkpoint_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read external resume checkpoint: {exc}") from exc
    checkpoint = value.get("resume_checkpoint") if isinstance(value, Mapping) else None
    if not isinstance(checkpoint, str):
        raise ValueError("external resume checkpoint lacks resume_checkpoint")
    return checkpoint


def _prepare_e1_vllm(args: argparse.Namespace) -> int:
    from mfh.data.reviewed_splits import authorize_reviewed_split_bundle
    from mfh.experiments.e1_vllm import prepare_e1_vllm

    reviewed = authorize_reviewed_split_bundle(
        args.splits,
        expected_manifest_digest=args.expected_split_manifest_digest,
        review_result_directory=args.review_result,
        expected_review_result_manifest_digest=args.expected_review_result_manifest_digest,
        review_queue_directory=args.review_queue,
        expected_review_queue_manifest_digest=args.expected_review_queue_manifest_digest,
        review_inputs=_contamination_review_inputs(args),
        plan=_reviewed_split_plan(args),
    )
    result = prepare_e1_vllm(**_e1_inputs(args), verified_reviewed_splits=reviewed)
    _print(result)
    return 0


def _run_e1_vllm(args: argparse.Namespace) -> int:
    from mfh.experiments.e1_vllm import run_e1_vllm_generations

    result = run_e1_vllm_generations(
        **_e1_inputs(args),
        request_budget=args.request_budget,
        expected_resume_checkpoint=_external_checkpoint(args),
        checkpoint_file=args.checkpoint_file,
    )
    _print(result)
    return 0


def _grade_e1_openrouter(args: argparse.Namespace) -> int:
    from mfh.experiments.e1_vllm import grade_e1_openrouter, load_env_secret

    api_key = load_env_secret(args.env_file, "OPENROUTER_API_KEY")
    result = grade_e1_openrouter(
        **_e1_inputs(args),
        api_key=api_key,
        request_budget=args.request_budget,
        expected_resume_checkpoint=_external_checkpoint(args),
        checkpoint_file=args.checkpoint_file,
    )
    _print(result)
    return 0


def _grade_strongreject_openrouter(args: argparse.Namespace) -> int:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from mfh.evaluation.strongreject import grade_strongreject_batch
    from mfh.experiments.e1_vllm import load_env_secret

    api_key = load_env_secret(args.env_file, "OPENROUTER_API_KEY")
    try:
        private_key_hex = args.scorer_private_key_file.read_text(encoding="utf-8").strip()
        private_key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_key_hex))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load StrongREJECT scorer private key: {exc}") from exc
    result = grade_strongreject_batch(
        records_path=args.records,
        questions_path=args.questions,
        grader_directory=args.grader,
        scorer_path=args.scorer,
        scorer_private_key=private_key,
        output_directory=args.output,
        api_key=api_key,
        request_budget=args.request_budget,
        resume=args.resume,
    )
    _print(result)
    return 0


def _finalize_e1(args: argparse.Namespace) -> int:
    from mfh.experiments.e1_vllm import finalize_e1_vllm

    result = finalize_e1_vllm(
        **_e1_inputs(args),
        output_directory=args.output,
        checkpoint_batch_size=args.checkpoint_batch_size,
    )
    _print(result)
    return 0


def _verify_e1_outputs(args: argparse.Namespace) -> int:
    from mfh.experiments.e1_vllm import verify_e1_output_bundle

    manifest = verify_e1_output_bundle(
        args.output,
        work_directory=args.work,
        ledger_directory=args.ledger,
        study_config=args.study_config,
        expected_manifest_digest=args.expected_manifest_digest,
    )
    _print(
        {
            "valid": True,
            "manifest_digest": manifest["manifest_digest"],
            "record_count": manifest["record_count"],
            "condition_count": manifest["condition_count"],
        }
    )
    return 0


def _aa_official_inputs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "splits_directory": args.splits,
        "expected_splits_manifest_digest": args.expected_split_manifest_digest,
        "grader_bundle": args.grader_bundle,
        "expected_grader_manifest_digest": args.expected_grader_manifest_digest,
        "model_config": args.model_config,
        "snapshot_directory": args.snapshot_directory,
        "snapshot_manifest": args.snapshot_manifest,
        "runtime_config": args.runtime_config,
        "ledger_directory": args.ledger,
        "e0_run": args.e0_run,
        "prompt_config": args.prompt_config,
        "inference_config": args.inference_config,
        "study_config": args.study_config,
    }


def _prepare_aa_official(args: argparse.Namespace) -> int:
    from mfh.experiments.aa_official_track import prepare_aa_official_track

    _print(
        prepare_aa_official_track(
            args.work,
            **_aa_official_inputs(args),
        )
    )
    return 0


def _run_aa_official(args: argparse.Namespace) -> int:
    from mfh.experiments.aa_official_track import run_aa_official_track
    from mfh.experiments.e1_vllm import load_env_secret

    if args.checkpoint_file is None:
        raise ValueError("run-aa-official requires --checkpoint-file")
    _print(
        run_aa_official_track(
            args.work,
            **_aa_official_inputs(args),
            api_key=load_env_secret(args.env_file, "OPENROUTER_API_KEY"),
            checkpoint_file=args.checkpoint_file,
            expected_resume_checkpoint=_external_checkpoint(args),
            request_budget=args.request_budget,
        )
    )
    return 0


def _finalize_aa_official(args: argparse.Namespace) -> int:
    from mfh.experiments.aa_official_track import finalize_aa_official_track

    _print(
        finalize_aa_official_track(
            args.output,
            work_directory=args.work,
            **_aa_official_inputs(args),
        )
    )
    return 0


def _verify_aa_official(args: argparse.Namespace) -> int:
    from mfh.experiments.aa_official_track import verify_aa_official_track

    _print(
        verify_aa_official_track(
            args.output,
            expected_manifest_digest=args.expected_manifest_digest,
            **_aa_official_inputs(args),
        )
    )
    return 0


def _add_e1_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("splits", type=Path)
    parser.add_argument("grader_bundle", type=Path)
    parser.add_argument("model_config", type=Path)
    parser.add_argument("snapshot_directory", type=Path)
    parser.add_argument("snapshot_manifest", type=Path)
    parser.add_argument("runtime_config", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("e0_run", type=Path)
    parser.add_argument("--expected-split-manifest-digest", required=True)
    parser.add_argument("--expected-grader-manifest-digest", required=True)
    parser.add_argument("--prompt-config", type=Path, default=Path("configs/prompts/primary.yaml"))
    parser.add_argument(
        "--inference-config", type=Path, default=Path("configs/experiments/core.yaml")
    )
    parser.add_argument(
        "--study-config", type=Path, default=Path("configs/experiments/phases.yaml")
    )


def _add_e1_execution_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--request-budget", type=int)
    parser.add_argument("--checkpoint-file", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume only after loading the exact external checkpoint file",
    )


def _e2_inputs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "splits_directory": args.splits,
        "expected_splits_manifest_digest": args.expected_split_manifest_digest,
        "e1_output": args.e1_output,
        "e1_work": args.e1_work,
        "e1_ledger": args.e1_ledger,
        "expected_e1_manifest_digest": args.expected_e1_manifest_digest,
        "model_config": args.model_config,
        "snapshot_directory": args.snapshot_directory,
        "snapshot_manifest": args.snapshot_manifest,
        "runtime_config": args.runtime_config,
        "workspace_directory": args.workspace,
        "capture_work_directory": args.capture_work,
        "prompt_config": args.prompt_config,
        "inference_config": args.inference_config,
        "study_config": args.study_config,
    }


def _e2_prepared_summary(prepared: Any) -> dict[str, Any]:
    return {
        "valid": True,
        "workspace": str(prepared.workspace.directory),
        "workspace_plan_identity": prepared.workspace.plan_identity,
        "capture_work": str(prepared.capture_work),
        "rows_expected": len(prepared.workspace.schedule),
        "new_generations_expected": prepared.workspace.protocol.expected_new_generations,
        "scientific_eligible": prepared.workspace.protocol.scientific_eligible,
    }


def _prepare_e2_vllm(args: argparse.Namespace) -> int:
    from mfh.experiments.e2_vllm import prepare_e2_vllm

    prepared = prepare_e2_vllm(**_e2_inputs(args), shard_rows=args.shard_rows)
    _print(_e2_prepared_summary(prepared))
    return 0


def _run_e2_vllm(args: argparse.Namespace) -> int:
    from mfh.experiments.e2_vllm import run_e2_vllm_capture

    _print(run_e2_vllm_capture(**_e2_inputs(args), request_budget=args.request_budget))
    return 0


def _verify_e2_capture(args: argparse.Namespace) -> int:
    from mfh.experiments.e2_vllm import verify_e2_vllm_capture

    _print(verify_e2_vllm_capture(**_e2_inputs(args), require_complete=args.require_complete))
    return 0


def _fit_e2_probes(args: argparse.Namespace) -> int:
    from mfh.experiments.e2_vllm import fit_e2_vllm_probes

    bundle = fit_e2_vllm_probes(
        args.output,
        **_e2_inputs(args),
        probe_work_directory=args.probe_work_directory,
    )
    _print(
        {
            "valid": True,
            "directory": str(bundle.directory),
            "manifest_digest": bundle.manifest_digest,
            "probe_plan_identity": bundle.plan_identity,
            "selected_gate_artifact": bundle.selected_gate_artifact,
            "gate_passed": bundle.gate_passed,
            "probe_incorrect_auroc": bundle.gate_probe_auroc,
            "strongest_confidence_baseline_auroc": bundle.gate_baseline_auroc,
            "scientific_eligible": bundle.scientific_eligible,
        }
    )
    return 0


def _verify_e2_probes(args: argparse.Namespace) -> int:
    from mfh.experiments.e2_probes import verify_e2_probe_bundle
    from mfh.experiments.e2_schedule import verify_e2_workspace

    bundle = verify_e2_probe_bundle(args.bundle, workspace=verify_e2_workspace(args.workspace))
    _print(
        {
            "valid": True,
            "manifest_digest": bundle.manifest_digest,
            "probe_plan_identity": bundle.plan_identity,
            "selected_gate_artifact": bundle.selected_gate_artifact,
            "gate_passed": bundle.gate_passed,
            "scientific_eligible": bundle.scientific_eligible,
        }
    )
    return 0


def _finalize_e2(args: argparse.Namespace) -> int:
    from mfh.experiments.e2_vllm import finalize_e2_vllm_phase

    terminal = finalize_e2_vllm_phase(
        args.output,
        **_e2_inputs(args),
        probe_bundle_directory=args.probe_bundle,
        expected_workspace_plan_identity=args.expected_workspace_plan_identity,
        expected_capture_plan_identity=args.expected_capture_plan_identity,
        expected_probe_manifest_digest=args.expected_probe_manifest_digest,
    )
    value = asdict(terminal)
    value["status"] = "falsified" if hasattr(terminal, "failed_gates") else "complete"
    _print(value)
    return 0


def _m2_caa_inputs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "construction_directory": args.e3_construction,
        "questions": tuple(read_questions(args.questions)),
        "prompts": {value.prompt_id: value for value in load_prompt_specs(args.prompt_config)},
    }


def _prepare_m2_caa(args: argparse.Namespace) -> int:
    from mfh.experiments.e4_caa_vllm import prepare_m2_caa_work

    _print(prepare_m2_caa_work(args.work, **_m2_caa_inputs(args)))
    return 0


def _run_m2_caa(args: argparse.Namespace) -> int:
    from mfh.experiments.e4_caa_vllm import run_m2_caa_work
    from mfh.experiments.model_selection import validate_active_model_spec
    from mfh.inference.vllm_research import VllmResearchRuntime

    model = load_model_spec(args.model_config)
    validate_active_model_spec(model)
    try:
        plan = json.loads((args.work / "plan.json").read_text(encoding="utf-8"))
        runtime_identity = plan["runtime_identity"]
        if not isinstance(runtime_identity, dict):
            raise TypeError("runtime identity is not an object")
        research_provenance = runtime_identity.get("research_provenance")
        if research_provenance is not None and not isinstance(research_provenance, dict):
            raise ValueError("M2 CAA plan runtime identity is invalid")
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"cannot load M2 CAA runtime identity: {exc}") from exc
    runtime = VllmResearchRuntime.from_spec(
        model,
        snapshot_path=args.snapshot_directory,
        seed=17,
        research_provenance=research_provenance,
    )
    try:
        _print(
            run_m2_caa_work(
                args.work,
                **_m2_caa_inputs(args),
                runtime=runtime,
                request_budget=args.request_budget,
            )
        )
    finally:
        runtime.close()
    return 0


def _verify_m2_caa_work(args: argparse.Namespace) -> int:
    from mfh.experiments.e4_caa_vllm import verify_m2_caa_work

    _print(
        verify_m2_caa_work(
            args.work,
            **_m2_caa_inputs(args),
            require_complete=args.require_complete,
        )
    )
    return 0


def _finalize_m2_caa(args: argparse.Namespace) -> int:
    from mfh.experiments.e4_caa_vllm import finalize_m2_caa_artifact

    artifact = finalize_m2_caa_artifact(
        args.output,
        work_directory=args.work,
        **_m2_caa_inputs(args),
    )
    _print(
        {
            "valid": True,
            "directory": str(artifact.directory),
            "manifest_digest": artifact.manifest_digest,
            "plan_identity": artifact.plan_identity,
            "pair_count": artifact.pair_count,
            "maximum_peak_memory_bytes": artifact.maximum_peak_memory_bytes,
        }
    )
    return 0


def _verify_m2_caa_artifact(args: argparse.Namespace) -> int:
    from mfh.experiments.e4_caa_vllm import verify_m2_caa_artifact

    artifact = verify_m2_caa_artifact(
        args.artifact,
        expected_manifest_digest=args.expected_manifest_digest,
    )
    _print(
        {
            "valid": True,
            "manifest_digest": artifact.manifest_digest,
            "plan_identity": artifact.plan_identity,
            "pair_count": artifact.pair_count,
            "maximum_peak_memory_bytes": artifact.maximum_peak_memory_bytes,
        }
    )
    return 0


def _build_e4_act_baseline(args: argparse.Namespace) -> int:
    from mfh.experiments.e4_act_vllm import build_e4_act_baseline

    artifact = build_e4_act_baseline(
        args.output,
        e2_probe_bundle=args.e2_probe_bundle,
        e2_workspace=args.e2_workspace,
        e2_phase_run=args.e2_phase_run,
        m2_caa_artifact=args.m2_caa_artifact,
        intervention_layer=args.intervention_layer,
        study=load_study_protocol(args.study_config),
    )
    _print(
        {
            "valid": True,
            "directory": str(artifact.directory),
            "manifest_digest": artifact.manifest_digest,
            "feature_layer": artifact.feature_layer,
            "feature_site": artifact.feature_site.value,
            "intervention_layer": artifact.intervention_layer,
            "intervention_site": artifact.intervention_site.value,
        }
    )
    return 0


def _prepare_e4_vllm(args: argparse.Namespace) -> int:
    from mfh.experiments.e4_vllm import prepare_e4_vllm_screen
    from mfh.inference.transformers_snapshot import verify_transformers_snapshot
    from mfh.inference.vllm_preflight import validate_vllm_preflight_receipt

    model = load_model_spec(args.model_config)
    verify_transformers_snapshot(model, args.snapshot_directory, args.snapshot_manifest)
    project_root = args.study_config.absolute().parents[2]
    validate_vllm_preflight_receipt(
        args.runtime_receipt,
        project_root=project_root,
        model_config=args.model_config,
        snapshot_directory=args.snapshot_directory,
        snapshot_manifest=args.snapshot_manifest,
        runtime_policy=project_root / "configs/runtimes/qwen3.6-27b-nvfp4-policy.json",
    )
    setup = prepare_e4_vllm_screen(
        args.setup,
        args.ledger,
        dev_questions=tuple(read_questions(args.dev_questions)),
        model=model,
        prompts={value.prompt_id: value for value in load_prompt_specs(args.prompt_config)},
        study=load_study_protocol(args.study_config),
        runtime_artifact=args.runtime_receipt,
        e2_probe_bundle=args.e2_probe_bundle,
        e3_static_vectors=args.e3_static_vectors,
        m2_caa_artifact=args.m2_caa_artifact,
        act_baseline_artifact=args.act_baseline_artifact,
        e3_phase_run=args.e3_phase_run,
        execution_private_key_hex=args.execution_key_file.read_text(encoding="utf-8").strip(),
        m1_layer=args.m1_layer,
        m2_layer=args.m2_layer,
        standardized_alpha=args.standardized_alpha,
        iti_implementation=args.iti_implementation,
        truthx_implementation=args.truthx_implementation,
        truthx_autoencoder=args.truthx_autoencoder,
    )
    _print(
        {
            "valid": True,
            "setup": str(setup.directory),
            "ledger": str(args.ledger.resolve()),
            "report_digest": setup.report.report_digest,
            "screen_receipt_digest": setup.screen.receipt_digest,
            "screen_questions": len(setup.screen.screen_question_ids),
            "feasible_methods": setup.report.feasible_methods,
        }
    )
    return 0


def _run_e4_vllm(args: argparse.Namespace) -> int:
    from mfh.experiments.e4_vllm import load_e4_vllm_setup, run_e4_vllm_screen
    from mfh.experiments.model_selection import validate_active_model_spec
    from mfh.inference.vllm_research import VllmResearchRuntime

    model = load_model_spec(args.model_config)
    validate_active_model_spec(model)
    setup = load_e4_vllm_setup(args.setup)
    m2 = Path(setup.report.artifact_paths["implementation:M2"])
    try:
        plan = json.loads((m2 / "plan.json").read_text(encoding="utf-8"))
        identity = plan["runtime_identity"]
        if not isinstance(identity, dict):
            raise TypeError("runtime identity is not an object")
        provenance = identity.get("research_provenance")
        if not isinstance(provenance, dict):
            raise TypeError("runtime identity is incomplete")
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"cannot load E4 runtime identity: {exc}") from exc
    runtime = VllmResearchRuntime.from_spec(
        model,
        snapshot_path=args.snapshot_directory,
        seed=17,
        research_provenance=provenance,
    )
    try:
        result = run_e4_vllm_screen(
            args.setup,
            args.ledger,
            study=load_study_protocol(args.study_config),
            prompts={value.prompt_id: value for value in load_prompt_specs(args.prompt_config)},
            runtime=runtime,
            execution_private_key_hex=args.execution_key_file.read_text(encoding="utf-8").strip(),
            request_budget=args.request_budget,
            checkpoint_rows=args.checkpoint_rows,
        )
    finally:
        runtime.close()
    _print(result)
    return 0


def _verify_e4_vllm(args: argparse.Namespace) -> int:
    from mfh.experiments.e4_vllm import verify_e4_vllm_screen

    _print(
        verify_e4_vllm_screen(
            args.setup,
            args.ledger,
            study=load_study_protocol(args.study_config),
            require_complete=args.require_complete,
        )
    )
    return 0


def _finalize_e4_vllm(args: argparse.Namespace) -> int:
    from mfh.experiments.e4_vllm import finalize_e4_vllm_screen

    promotion, terminal = finalize_e4_vllm_screen(
        args.setup,
        args.ledger,
        study=load_study_protocol(args.study_config),
        promotion_path=args.promotion,
        gate_evidence_path=args.gate_evidence,
    )
    value = asdict(terminal)
    value.update(
        {
            "valid": True,
            "promotion": str(args.promotion.resolve()),
            "promotion_digest": promotion.promotion_digest,
            "promoted_methods": promotion.promoted_methods,
        }
    )
    _print(value)
    return 0


def _e5_source_context(
    args: argparse.Namespace,
) -> tuple[tuple[Any, ...], dict[str, Any], Any]:
    from mfh.experiments.e3_construction import (
        load_verified_e3_construction_snapshot,
    )

    questions = tuple(read_questions(args.questions))
    prompts = {
        value.prompt_id: value
        for value in load_prompt_specs(args.prompt_config)
        if value.prompt_id in {"P0-neutral", "P2-calibrated-abstention"}
    }
    snapshot = load_verified_e3_construction_snapshot(
        args.e3_construction,
        questions=questions,
        prompts=prompts,
    )
    return questions, prompts, snapshot


def _write_e3_operator_runbook(args: argparse.Namespace) -> int:
    from mfh.experiments.e3_operator import write_e3_operator_runbook_template

    path = write_e3_operator_runbook_template(
        args.output, reviewed_splits=args.reviewed_splits
    )
    _print({"valid": True, "runbook": str(path.resolve())})
    return 0


def _preflight_e3_operator(args: argparse.Namespace) -> int:
    from mfh.experiments.e3_operator import (
        load_e3_operator_runbook,
        preflight_e3_operator,
    )

    _print(preflight_e3_operator(load_e3_operator_runbook(args.runbook)))
    return 0


def _advance_e3_operator(args: argparse.Namespace) -> int:
    from mfh.experiments.e3_operator import (
        advance_e3_operator,
        load_e3_operator_runbook,
    )

    _print(
        advance_e3_operator(
            load_e3_operator_runbook(args.runbook),
            request_budget=args.request_budget,
        )
    )
    return 0


def _verify_e3_operator(args: argparse.Namespace) -> int:
    from mfh.experiments.e3_operator import (
        load_e3_operator_runbook,
        verify_e3_operator,
    )

    _print(verify_e3_operator(load_e3_operator_runbook(args.runbook)))
    return 0


def _materialize_e5_controller_splits(args: argparse.Namespace) -> int:
    from mfh.experiments.e5_controller_splits import write_e5_controller_splits

    verified = write_e5_controller_splits(
        args.output,
        source_questions=args.source_questions,
        expected_reviewed_split_manifest_digest=args.expected_reviewed_split_manifest_digest,
    )
    _print(
        {
            "valid": True,
            "directory": str(verified.directory),
            "manifest_digest": verified.manifest_digest,
            "train_rows": len(verified.train_questions),
            "calibration_rows": len(verified.calibration_questions),
        }
    )
    return 0


def _verify_e5_controller_splits(args: argparse.Namespace) -> int:
    from mfh.experiments.e5_controller_splits import verify_e5_controller_splits

    verified = verify_e5_controller_splits(
        args.splits,
        source_questions=args.source_questions,
        expected_manifest_digest=args.expected_manifest_digest,
        expected_reviewed_split_manifest_digest=args.expected_reviewed_split_manifest_digest,
    )
    _print(
        {
            "valid": True,
            "directory": str(verified.directory),
            "manifest_digest": verified.manifest_digest,
            "train_rows": len(verified.train_questions),
            "calibration_rows": len(verified.calibration_questions),
        }
    )
    return 0


def _prepare_e5_fit_capture(args: argparse.Namespace) -> int:
    from mfh.contracts import ActivationSite
    from mfh.experiments.e2_controller_inputs import controller_input_views
    from mfh.experiments.e2_probes import verify_e2_probe_bundle
    from mfh.experiments.e2_schedule import verify_e2_workspace
    from mfh.experiments.e5_capture import (
        e5_capture_public_key,
        prepare_e5_fit_capture,
    )
    from mfh.experiments.e5_fit import E5FitRecipe
    from mfh.methods.probes import ProbeTask

    questions, prompts, snapshot = _e5_source_context(args)
    workspace = verify_e2_workspace(args.e2_workspace)
    bundle = verify_e2_probe_bundle(args.e2_probe_bundle, workspace=workspace)
    selected = bundle.selected_views[ProbeTask.CORRECT_INCORRECT_ABSTENTION]
    views = controller_input_views(
        selected_layer=selected.layer,
        selected_site=selected.site,
        candidate_layers=workspace.activation_spec.layers,
    )
    try:
        e2_plan = json.loads((Path(args.e2_probe_bundle) / "plan.json").read_text(encoding="utf-8"))
        split_manifest_digest = e2_plan["split_manifest_digest"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"cannot load the E2 split identity: {exc}") from exc
    key = args.execution_key_file.read_text(encoding="utf-8").strip()
    recipe = E5FitRecipe(
        fixed_best_layer=args.fixed_best_layer,
        two_layer_candidates=tuple(args.two_layer_candidates),
        three_layer_candidates=tuple(args.three_layer_candidates),
        intervention_site=ActivationSite(args.intervention_site),
    )
    plan = prepare_e5_fit_capture(
        args.work,
        snapshot=snapshot,
        questions=questions,
        prompts=prompts,
        views=views,
        recipe=recipe,
        runtime_identity=snapshot.plan["runtime_identity"],
        execution_public_key=e5_capture_public_key(key),
        runtime_artifact_sha256=sha256_path(args.runtime_artifact),
        e2_probe_bundle_sha256=sha256_path(args.e2_probe_bundle),
        e3_static_vectors_sha256=sha256_path(args.e3_static_vectors),
        split_manifest_digest=split_manifest_digest,
        hidden_width=snapshot.plan["hidden_width"],
        shard_rows=args.shard_rows,
        max_peak_memory_bytes=args.max_peak_memory_bytes,
    )
    _print(
        {
            "valid": True,
            "work": str(Path(args.work).resolve()),
            "plan_identity": plan["plan_identity"],
            "expected_pairs": plan["expected_pairs"],
            "capture_layers": plan["capture_layers"],
            "capture_site": plan["capture_site"],
            "scientific_eligible": plan["scientific_eligible"],
        }
    )
    return 0


def _run_e5_fit_capture(args: argparse.Namespace) -> int:
    from mfh.experiments.e5_capture import run_e5_fit_capture
    from mfh.experiments.model_selection import validate_active_model_spec
    from mfh.inference.vllm_research import VllmResearchRuntime

    questions, prompts, snapshot = _e5_source_context(args)
    model = load_model_spec(args.model_config)
    validate_active_model_spec(model)
    identity = snapshot.plan["runtime_identity"]
    provenance = identity.get("research_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("E5 source runtime lacks its frozen research provenance")
    runtime = VllmResearchRuntime.from_spec(
        model,
        snapshot_path=args.snapshot_directory,
        seed=17,
        research_provenance=provenance,
    )
    try:
        result = run_e5_fit_capture(
            args.work,
            snapshot=snapshot,
            questions=questions,
            prompts=prompts,
            runtime=runtime,
            private_key_hex=args.execution_key_file.read_text(encoding="utf-8").strip(),
            request_budget=args.request_budget,
        )
    finally:
        runtime.close()
    _print(
        {
            "valid": True,
            "pairs_completed": result.pairs_completed,
            "pairs_expected": result.plan["expected_pairs"],
            "shard_count": result.shard_count,
            "complete": result.complete,
            "maximum_peak_memory_bytes": result.maximum_peak_memory_bytes,
            "scientific_eligible": result.scientific_eligible,
        }
    )
    return 0


def _verify_e5_fit_capture(args: argparse.Namespace) -> int:
    from mfh.experiments.e5_capture import (
        e5_capture_public_key,
        verify_e5_fit_capture,
    )

    questions, prompts, snapshot = _e5_source_context(args)
    result = verify_e5_fit_capture(
        args.work,
        snapshot=snapshot,
        questions=questions,
        prompts=prompts,
        expected_execution_public_key=e5_capture_public_key(
            args.execution_key_file.read_text(encoding="utf-8").strip()
        ),
        require_complete=args.require_complete,
    )
    _print(
        {
            "valid": True,
            "plan_identity": result.plan["plan_identity"],
            "pairs_completed": result.pairs_completed,
            "pairs_expected": result.plan["expected_pairs"],
            "shard_count": result.shard_count,
            "chain_head": result.chain_head,
            "complete": result.complete,
            "maximum_peak_memory_bytes": result.maximum_peak_memory_bytes,
            "scientific_eligible": result.scientific_eligible,
        }
    )
    return 0


def _e5_verified_fit_capture_context(args: argparse.Namespace) -> tuple[Any, ...]:
    from mfh.experiments.e3_construction import load_verified_e3_construction_snapshot
    from mfh.experiments.e5_capture import (
        e5_capture_public_key,
        verify_e5_fit_capture,
    )

    questions = tuple(read_questions(args.t_steer_questions))
    prompts = {
        value.prompt_id: value
        for value in load_prompt_specs(args.prompt_config)
        if value.prompt_id in {"P0-neutral", "P2-calibrated-abstention"}
    }
    snapshot = load_verified_e3_construction_snapshot(
        args.e3_construction,
        questions=questions,
        prompts=prompts,
    )
    private_key = args.execution_key_file.read_text(encoding="utf-8").strip()
    verified = verify_e5_fit_capture(
        args.fit_capture,
        snapshot=snapshot,
        questions=questions,
        prompts=prompts,
        expected_execution_public_key=e5_capture_public_key(private_key),
        require_complete=True,
    )
    return questions, prompts, snapshot, private_key, verified


def _e5_controller_training_datasets(
    *, e2_workspace: Path, e2_probe_bundle: Path
) -> tuple[Mapping[Any, Any], Any]:
    from mfh.experiments.activation_store import verify_activation_store
    from mfh.experiments.e2_controller_inputs import (
        build_e2_controller_input_datasets,
        controller_input_views,
    )
    from mfh.experiments.e2_probes import verify_e2_probe_bundle
    from mfh.experiments.e2_schedule import verify_e2_workspace
    from mfh.methods.probes import ProbeTask

    workspace = verify_e2_workspace(e2_workspace)
    bundle = verify_e2_probe_bundle(e2_probe_bundle, workspace=workspace)
    selected = bundle.selected_views[ProbeTask.CORRECT_INCORRECT_ABSTENTION]
    views = controller_input_views(
        selected_layer=selected.layer,
        selected_site=selected.site,
        candidate_layers=workspace.activation_spec.layers,
    )
    try:
        plan = json.loads((e2_probe_bundle / "plan.json").read_text(encoding="utf-8"))
        split_digest = plan["split_manifest_digest"]
        prompt_sha256 = plan["prompt_template_sha256"]["P0-neutral"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"cannot load E2 controller-input identities: {exc}") from exc
    activation = verify_activation_store(
        workspace.directory / "activations",
        expected_spec=workspace.activation_spec,
        require_complete=True,
    )
    grid = build_e2_controller_input_datasets(
        workspace,
        views=views,
        split_manifest_digest=split_digest,
        prompt_template_sha256=prompt_sha256,
        verified_store=activation,
    )
    return (
        {view.composition: grid[(view.composition, "T-controller-train")].probe for view in views},
        bundle,
    )


def _prepare_e5_layer_labels(args: argparse.Namespace) -> int:
    from mfh.experiments.e5_layer_labels import prepare_e5_layer_label_capture
    from mfh.experiments.e5_types import E5FitRecipe

    _questions, prompts, _snapshot, _private_key, capture = _e5_verified_fit_capture_context(args)
    controller_datasets, _bundle = _e5_controller_training_datasets(
        e2_workspace=args.e2_workspace,
        e2_probe_bundle=args.e2_probe_bundle,
    )
    controller_questions = tuple(read_questions(args.controller_questions))
    plan = prepare_e5_layer_label_capture(
        args.work,
        questions=controller_questions,
        prompt=prompts["P0-neutral"],
        controller_datasets=controller_datasets,
        fit_capture=capture,
        fit_capture_artifact_sha256=sha256_path(args.fit_capture),
        e3_static_vectors_directory=args.e3_static_vectors,
        recipe=E5FitRecipe.from_dict(dict(capture.plan["recipe"])),
        runtime_identity=capture.plan["runtime_identity"],
        execution_public_key=capture.plan["execution_public_key"],
        shard_rows=args.shard_rows,
        max_new_tokens=args.max_new_tokens,
        max_peak_memory_bytes=args.max_peak_memory_bytes,
    )
    _print(
        {
            "valid": True,
            "work": str(args.work.resolve()),
            "plan_identity": plan["plan_identity"],
            "expected_records": plan["expected_records"],
            "expected_questions": plan["expected_questions"],
            "candidate_layers": plan["candidate_layers"],
            "scientific_eligible": plan["scientific_eligible"],
        }
    )
    return 0


def _run_e5_layer_labels(args: argparse.Namespace) -> int:
    from mfh.experiments.e5_layer_labels import run_e5_layer_label_capture
    from mfh.experiments.model_selection import validate_active_model_spec
    from mfh.inference.vllm_research import VllmResearchRuntime

    _questions, prompts, _snapshot, private_key, capture = _e5_verified_fit_capture_context(args)
    controller_datasets, _bundle = _e5_controller_training_datasets(
        e2_workspace=args.e2_workspace,
        e2_probe_bundle=args.e2_probe_bundle,
    )
    model = load_model_spec(args.model_config)
    validate_active_model_spec(model)
    provenance = capture.plan["runtime_identity"].get("research_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("E5 layer-label runtime lacks frozen research provenance")
    runtime = VllmResearchRuntime.from_spec(
        model,
        snapshot_path=args.snapshot_directory,
        seed=17,
        research_provenance=provenance,
    )
    try:
        result = run_e5_layer_label_capture(
            args.work,
            questions=tuple(read_questions(args.controller_questions)),
            prompt=prompts["P0-neutral"],
            controller_datasets=controller_datasets,
            fit_capture=capture,
            fit_capture_artifact_sha256=sha256_path(args.fit_capture),
            runtime=runtime,
            private_key_hex=private_key,
            request_budget=args.request_budget,
        )
    finally:
        runtime.close()
    _print(
        {
            "valid": True,
            "records_completed": result.records_completed,
            "records_expected": result.plan["expected_records"],
            "shard_count": result.shard_count,
            "complete": result.complete,
            "maximum_peak_memory_bytes": result.maximum_peak_memory_bytes,
            "scientific_eligible": result.scientific_eligible,
        }
    )
    return 0


def _verify_e5_layer_labels(args: argparse.Namespace) -> int:
    from mfh.experiments.e5_layer_labels import verify_e5_layer_label_capture

    _questions, prompts, _snapshot, _private_key, capture = _e5_verified_fit_capture_context(args)
    controller_datasets, _bundle = _e5_controller_training_datasets(
        e2_workspace=args.e2_workspace,
        e2_probe_bundle=args.e2_probe_bundle,
    )
    result = verify_e5_layer_label_capture(
        args.work,
        questions=tuple(read_questions(args.controller_questions)),
        prompt=prompts["P0-neutral"],
        controller_datasets=controller_datasets,
        fit_capture=capture,
        fit_capture_artifact_sha256=sha256_path(args.fit_capture),
        expected_execution_public_key=capture.plan["execution_public_key"],
        require_complete=args.require_complete,
    )
    _print(
        {
            "valid": True,
            "plan_identity": result.plan["plan_identity"],
            "records_completed": result.records_completed,
            "records_expected": result.plan["expected_records"],
            "chain_head": result.chain_head,
            "complete": result.complete,
            "scientific_eligible": result.scientific_eligible,
        }
    )
    return 0


def _fit_e5_controller_grid(args: argparse.Namespace) -> int:
    from mfh.experiments.e5_adaptive import E5Protocol
    from mfh.experiments.e5_capture import load_e5_fit_capture_data
    from mfh.experiments.e5_fit import (
        e5_fit_capture_attestation_body,
        fit_e5_controller_grid,
        save_e5_fitted_grid,
        sign_e5_fit_capture_attestation,
    )
    from mfh.experiments.e5_layer_labels import load_e5_layer_label_data
    from mfh.experiments.e5_types import E5FitRecipe
    from mfh.methods.probes import load_calibrated_probe

    questions, prompts, snapshot, private_key, capture = _e5_verified_fit_capture_context(args)
    controller_datasets, bundle = _e5_controller_training_datasets(
        e2_workspace=args.e2_workspace,
        e2_probe_bundle=args.e2_probe_bundle,
    )
    trusted_key = capture.plan["execution_public_key"]
    capture_data = load_e5_fit_capture_data(
        args.fit_capture,
        snapshot=snapshot,
        questions=questions,
        prompts=prompts,
        expected_execution_public_key=trusted_key,
    )
    controller_questions = tuple(read_questions(args.controller_questions))
    layer_labels = load_e5_layer_label_data(
        args.layer_labels,
        questions=controller_questions,
        prompt=prompts["P0-neutral"],
        controller_datasets=controller_datasets,
        fit_capture=capture,
        fit_capture_artifact_sha256=capture_data.capture_artifact_sha256,
        expected_execution_public_key=trusted_key,
    )
    protocol = E5Protocol.from_dict(capture.plan["protocol"])
    recipe = E5FitRecipe.from_dict(dict(capture.plan["recipe"]))
    risk_paths = {
        composition: args.e2_probe_bundle / "controller-input-probes" / composition.value
        for composition in controller_datasets
    }
    risk_sha256 = {
        composition: bundle.controller_input_artifacts[composition]
        for composition in controller_datasets
    }
    risk_probes = {
        composition: load_calibrated_probe(path) for composition, path in risk_paths.items()
    }
    attestation_body = e5_fit_capture_attestation_body(
        protocol=protocol,
        recipe=recipe,
        execution_public_key=trusted_key,
        runtime_artifact_sha256=capture.plan["runtime_artifact_sha256"],
        e2_probe_bundle_sha256=capture.plan["e2_probe_bundle_sha256"],
        e3_static_vectors_sha256=capture.plan["e3_static_vectors_sha256"],
        e3_construction_sha256=capture.plan["e3_construction_sha256"],
        risk_probes=risk_probes,
        risk_probe_artifact_sha256=risk_sha256,
        risk_probe_artifact_paths=risk_paths,
        controller_datasets=controller_datasets,
        capture_data=capture_data,
        layer_labels=layer_labels,
    )
    fitted = fit_e5_controller_grid(
        protocol=protocol,
        recipe=recipe,
        risk_probes=risk_probes,
        risk_probe_artifact_sha256=risk_sha256,
        risk_probe_artifact_paths=risk_paths,
        controller_datasets=controller_datasets,
        capture_data=capture_data,
        layer_labels=layer_labels,
        capture_attestation=sign_e5_fit_capture_attestation(
            attestation_body, private_key_hex=private_key
        ),
        runtime_artifact_sha256=capture.plan["runtime_artifact_sha256"],
        e2_probe_bundle_sha256=capture.plan["e2_probe_bundle_sha256"],
        e3_static_vectors_sha256=capture.plan["e3_static_vectors_sha256"],
        e3_construction_sha256=capture.plan["e3_construction_sha256"],
        expected_execution_public_key=trusted_key,
    )
    saved = save_e5_fitted_grid(args.output, fitted=fitted)
    _print(
        {
            "valid": True,
            "output": str(saved.directory),
            "manifest_digest": saved.manifest["manifest_digest"],
            "controller_count": len(saved.controller_directories),
            "unique_controller_fit_count": saved.manifest["unique_controller_fit_count"],
            "scientific_eligible": saved.scientific_eligible,
        }
    )
    return 0


def _verify_e5_controller_grid(args: argparse.Namespace) -> int:
    from mfh.experiments.e5_fit import verify_e5_fitted_grid

    result = verify_e5_fitted_grid(args.grid)
    _print(
        {
            "valid": True,
            "grid": str(result.directory),
            "manifest_digest": result.manifest["manifest_digest"],
            "controller_count": len(result.controller_directories),
            "unique_controller_fit_count": result.manifest["unique_controller_fit_count"],
            "scientific_eligible": result.scientific_eligible,
        }
    )
    return 0


def _package_e5_controller_bindings(args: argparse.Namespace) -> int:
    from mfh.experiments.e5_fit import package_e5_controller_bindings
    from mfh.experiments.e5_native import e5_native_execution_public_key

    private_key = args.execution_key_file.read_text(encoding="utf-8").strip()
    result = package_e5_controller_bindings(
        args.output,
        fitted_grid_directory=args.grid,
        expected_execution_public_key=e5_native_execution_public_key(private_key),
    )
    _print(
        {
            "valid": True,
            "output": str(result.directory),
            "manifest_digest": result.manifest["manifest_digest"],
            "binding_count": len(result.binding_paths),
            "scientific_eligible": result.scientific_eligible,
        }
    )
    return 0


def _verify_e5_controller_bindings(args: argparse.Namespace) -> int:
    from mfh.experiments.e5_fit import verify_e5_controller_bindings

    result = verify_e5_controller_bindings(args.bindings)
    _print(
        {
            "valid": True,
            "bindings": str(result.directory),
            "manifest_digest": result.manifest["manifest_digest"],
            "binding_count": len(result.binding_paths),
            "scientific_eligible": result.scientific_eligible,
        }
    )
    return 0


def _prepare_e5_native_ablation(args: argparse.Namespace) -> int:
    from mfh.experiments.e5_adaptive import E5Protocol
    from mfh.experiments.e5_fit import verify_e5_controller_bindings
    from mfh.experiments.e5_native import (
        e5_native_execution_public_key,
        prepare_e5_native_ablation,
    )
    from mfh.experiments.e5_operator import E5_EXACT_GRID_RECORDS

    if args.acknowledge_exact_grid_records != E5_EXACT_GRID_RECORDS:
        raise ValueError(
            f"E5 preparation requires --acknowledge-exact-grid-records {E5_EXACT_GRID_RECORDS}"
        )
    private_key = args.execution_key_file.read_text(encoding="utf-8").strip()
    bindings = verify_e5_controller_bindings(args.controller_bindings)
    prompts = {
        value.prompt_id: value
        for value in load_prompt_specs(args.prompt_config)
        if value.prompt_id in {"P0-neutral", "P2-calibrated-abstention"}
    }
    plan = prepare_e5_native_ablation(
        args.work,
        screen_receipt_path=args.screen_receipt,
        controller_bindings_directory=bindings.directory,
        fit_capture_directory=args.fit_capture,
        m1_policy_path=args.m1_policy,
        e3_static_vectors_directory=args.e3_static_vectors,
        runtime_artifact=args.runtime_artifact,
        prompts=prompts,
        execution_public_key=e5_native_execution_public_key(private_key),
        protocol=E5Protocol.from_dict(bindings.manifest["protocol"]),
        shard_rows=args.shard_rows,
        max_new_tokens=args.max_new_tokens,
        max_peak_memory_bytes=args.max_peak_memory_bytes,
    )
    _print(
        {
            "valid": True,
            "work": str(args.work.resolve()),
            "plan_identity": plan["plan_identity"],
            "expected_records": plan["expected_records"],
            "schedule_rule": plan["schedule_rule"],
            "scientific_eligible": plan["scientific_eligible"],
        }
    )
    return 0


def _run_e5_native_ablation(args: argparse.Namespace) -> int:
    from mfh.experiments.e5_native import (
        run_e5_native_ablation,
    )
    from mfh.experiments.model_selection import validate_active_model_spec
    from mfh.inference.vllm_research import VllmResearchRuntime

    private_key = args.execution_key_file.read_text(encoding="utf-8").strip()
    try:
        frozen_plan = json.loads((args.work / "plan.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"cannot read E5 native runtime plan: {exc}") from exc
    if not isinstance(frozen_plan, Mapping):
        raise ValueError("E5 native runtime plan must be a mapping")
    model = load_model_spec(args.model_config)
    validate_active_model_spec(model)
    identity = frozen_plan.get("runtime_identity")
    provenance = identity.get("research_provenance") if isinstance(identity, Mapping) else None
    if not isinstance(provenance, Mapping):
        raise ValueError("E5 native runtime lacks frozen research provenance")
    runtime = VllmResearchRuntime.from_spec(
        model,
        snapshot_path=args.snapshot_directory,
        seed=17,
        research_provenance=provenance,
    )
    try:
        result = run_e5_native_ablation(
            args.work,
            runtime=runtime,
            execution_private_key_hex=private_key,
            request_budget=args.request_budget,
        )
    finally:
        runtime.close()
    _print(
        {
            "valid": True,
            "records_completed": result.records_completed,
            "records_expected": result.plan["expected_records"],
            "shard_count": result.shard_count,
            "chain_head": result.chain_head,
            "complete": result.complete,
            "maximum_peak_memory_bytes": result.maximum_peak_memory_bytes,
            "scientific_eligible": result.scientific_eligible,
        }
    )
    return 0


def _verify_e5_native_ablation(args: argparse.Namespace) -> int:
    from mfh.experiments.e5_native import (
        e5_native_execution_public_key,
        verify_e5_native_ablation,
    )

    private_key = args.execution_key_file.read_text(encoding="utf-8").strip()
    result = verify_e5_native_ablation(
        args.work,
        expected_execution_public_key=e5_native_execution_public_key(private_key),
        require_complete=args.require_complete,
        semantic=not args.structural_only,
    )
    _print(
        {
            "valid": True,
            "plan_identity": result.plan["plan_identity"],
            "records_completed": result.records_completed,
            "records_expected": result.plan["expected_records"],
            "shard_count": result.shard_count,
            "chain_head": result.chain_head,
            "complete": result.complete,
            "finalized_records": (
                str(result.finalized_records) if result.finalized_records is not None else None
            ),
            "scientific_eligible": result.scientific_eligible,
        }
    )
    return 0


def _finalize_e5_native_ablation(args: argparse.Namespace) -> int:
    from mfh.experiments.e5_native import finalize_e5_native_ablation

    result = finalize_e5_native_ablation(
        args.work,
        execution_private_key_hex=args.execution_key_file.read_text(encoding="utf-8").strip(),
    )
    _print(
        {
            "valid": True,
            "records": str(result.finalized_records),
            "records_completed": result.records_completed,
            "chain_head": result.chain_head,
            "scientific_eligible": result.scientific_eligible,
        }
    )
    return 0


def _estimate_e5_native_ablation(args: argparse.Namespace) -> int:
    from mfh.experiments.e5_operator import estimate_e5_native_ablation

    _print(
        dict(
            estimate_e5_native_ablation(
                generations_per_second=args.generations_per_second,
                checkpoint_opens_per_second=args.checkpoint_opens_per_second,
                verification_rows_per_second=args.verification_rows_per_second,
                request_budget=args.request_budget,
            )
        )
    )
    return 0


def _derive_e5_selection(args: argparse.Namespace) -> int:
    from mfh.experiments.e5_operator import derive_signed_e5_selection

    result = derive_signed_e5_selection(
        args.output,
        native_directory=args.native,
        controller_bindings_directory=args.controller_bindings,
        e2_probe_bundle=args.e2_probe_bundle,
        e3_static_vectors=args.e3_static_vectors,
        e4_promoted_baselines=args.e4_promoted_baselines,
        execution_private_key_hex=args.execution_key_file.read_text(encoding="utf-8").strip(),
    )
    _print(dict(result))
    return 0


def _verify_e5_selection_package(args: argparse.Namespace) -> int:
    from mfh.experiments.e5_operator import verify_signed_e5_selection

    _print(
        dict(
            verify_signed_e5_selection(
                args.selection,
                native_directory=args.native,
                execution_private_key_hex=args.execution_key_file.read_text(
                    encoding="utf-8"
                ).strip(),
            )
        )
    )
    return 0


def _prepare_e5_phase_ledger(args: argparse.Namespace) -> int:
    from mfh.experiments.e5_operator import (
        load_e5_operator_inputs,
        prepare_e5_phase_ledger,
    )
    from mfh.experiments.protocol import ExperimentPhase

    model, prompts = load_e5_operator_inputs(
        model_config=args.model_config,
        prompt_config=args.prompt_config,
    )
    ledger = prepare_e5_phase_ledger(
        args.ledger,
        selection_directory=args.selection,
        native_directory=args.native,
        model=model,
        prompts=prompts,
        study=load_study_protocol(args.study_protocol),
        prerequisite_runs={
            ExperimentPhase.E2: args.e2_run,
            ExperimentPhase.E3: args.e3_run,
            ExperimentPhase.E4: args.e4_run,
        },
        execution_private_key_hex=args.execution_key_file.read_text(encoding="utf-8").strip(),
    )
    completed, expected = ledger.progress()
    _print(
        {
            "valid": True,
            "ledger": str(ledger.directory),
            "contract_digest": ledger.contract.digest,
            "completed_records": completed,
            "expected_records": expected,
        }
    )
    return 0


def _promote_e5_phase_records(args: argparse.Namespace) -> int:
    from mfh.experiments.e5_operator import promote_e5_phase_records

    _print(
        dict(
            promote_e5_phase_records(
                args.ledger,
                selection_directory=args.selection,
                native_directory=args.native,
                study=load_study_protocol(args.study_protocol),
                execution_private_key_hex=args.execution_key_file.read_text(
                    encoding="utf-8"
                ).strip(),
                request_budget=args.request_budget,
                checkpoint_rows=args.checkpoint_rows,
            )
        )
    )
    return 0


def _verify_e5_phase_ledger(args: argparse.Namespace) -> int:
    from mfh.experiments.e5_operator import verify_e5_phase_promotion

    _print(
        dict(
            verify_e5_phase_promotion(
                args.ledger,
                selection_directory=args.selection,
                native_directory=args.native,
                study=load_study_protocol(args.study_protocol),
                execution_private_key_hex=args.execution_key_file.read_text(
                    encoding="utf-8"
                ).strip(),
                require_complete=args.require_complete,
            )
        )
    )
    return 0


def _finalize_e5_phase(args: argparse.Namespace) -> int:
    from mfh.experiments.e5_operator import finalize_promoted_e5_phase

    _print(
        dict(
            finalize_promoted_e5_phase(
                args.output,
                ledger_directory=args.ledger,
                selection_directory=args.selection,
                native_directory=args.native,
                study=load_study_protocol(args.study_protocol),
                execution_private_key_hex=args.execution_key_file.read_text(
                    encoding="utf-8"
                ).strip(),
            )
        )
    )
    return 0


def _verify_e5_phase(args: argparse.Namespace) -> int:
    from mfh.experiments.e5_operator import verify_promoted_e5_phase

    _print(
        dict(
            verify_promoted_e5_phase(
                args.output,
                ledger_directory=args.ledger,
                selection_directory=args.selection,
                native_directory=args.native,
                study=load_study_protocol(args.study_protocol),
                execution_private_key_hex=args.execution_key_file.read_text(
                    encoding="utf-8"
                ).strip(),
            )
        )
    )
    return 0


def _add_m2_caa_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("e3_construction", type=Path)
    parser.add_argument("questions", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("--prompt-config", type=Path, default=Path("configs/prompts/primary.yaml"))


def _add_e2_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("splits", type=Path)
    parser.add_argument("e1_output", type=Path)
    parser.add_argument("e1_work", type=Path)
    parser.add_argument("e1_ledger", type=Path)
    parser.add_argument("model_config", type=Path)
    parser.add_argument("snapshot_directory", type=Path)
    parser.add_argument("snapshot_manifest", type=Path)
    parser.add_argument("runtime_config", type=Path)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("capture_work", type=Path)
    parser.add_argument("--expected-split-manifest-digest", required=True)
    parser.add_argument("--expected-e1-manifest-digest", required=True)
    parser.add_argument("--prompt-config", type=Path, default=Path("configs/prompts/primary.yaml"))
    parser.add_argument(
        "--inference-config", type=Path, default=Path("configs/experiments/core.yaml")
    )
    parser.add_argument(
        "--study-config", type=Path, default=Path("configs/experiments/phases.yaml")
    )


def _phase_progress(path: Path, study_protocol: Path) -> int:
    ledger = PhaseRunLedger.open(path, study=load_study_protocol(study_protocol))
    complete, expected = ledger.progress()
    _print(
        {
            "phase": ledger.contract.phase.value,
            "contract_digest": ledger.contract.digest,
            "completed_records": complete,
            "expected_records": expected,
            "complete": (path / "complete.json").is_file(),
        }
    )
    return 0


def _verify_phase(path: Path, study_protocol: Path) -> int:
    completion = PhaseRunLedger.open(
        path, study=load_study_protocol(study_protocol)
    ).verify_complete()
    _print(
        {
            "valid": True,
            "phase": completion.phase.value,
            "contract_digest": completion.contract_digest,
            "record_count": completion.record_count,
            "shard_fingerprints": dict(completion.shard_fingerprints),
            "record_set_digest": completion.record_set_digest,
            "gate_result_digests": dict(completion.gate_result_digests),
            "gate_file_fingerprints": dict(completion.gate_file_fingerprints),
            "completion_digest": completion.completion_digest,
        }
    )
    return 0


def _write_analysis(args: argparse.Namespace) -> int:
    import shutil
    import tempfile

    import mfh.analysis.reporting as reporting
    from mfh.analysis.human_audit import (
        load_blinding_key,
        load_factual_adjudicated_rows,
    )
    from mfh.analysis.protocol import load_analysis_protocol
    from mfh.analysis.reporting import (
        ReportSource,
        render_svg_report,
        report_result_payload,
        write_adjudicated_labels_report,
        write_confusion_matrix_report,
        write_final_analysis_bundle,
        write_zero_error_report,
    )
    from mfh.artifact_namespace import validate_active_study_artifact_paths

    protocol = load_analysis_protocol(args.analysis_protocol)
    protocol.verify_research_plan(args.research_plan)
    study = load_study_protocol(args.study_protocol)
    derived = _derive_analysis_from_args(args, protocol=protocol, study=study)
    blinding_key = load_blinding_key(args.blinding_key_file)
    audit_rows = load_factual_adjudicated_rows(
        args.audit_results,
        queue_directory=args.audit_queue,
        expected_protocol=protocol,
        study=study,
        phase_run_directories={"E9": args.e9_run, "E10": args.e10_run},
        blinding_key=blinding_key,
    )
    output = validate_active_study_artifact_paths({"final analysis bundle": args.output})[
        "final analysis bundle"
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.reports-", dir=output.parent))
    try:
        sources: dict[str, ReportSource] = {}
        generator_revision = sha256_file(Path(reporting.__file__))
        names = set(protocol.required_report_outputs) | set(protocol.human_audit.required_outputs)
        for name in sorted(names):
            if name == "zero_error_confidence_bounds":
                report_path = stage / f"{name}.csv"
                write_zero_error_report(report_path, derived.results)
            elif name == "adjudicated_final_labels":
                report_path = stage / f"{name}.csv"
                write_adjudicated_labels_report(report_path, derived.results, audit_rows)
            elif name == "automated_human_confusion_matrix":
                report_path = stage / f"{name}.csv"
                write_confusion_matrix_report(report_path, derived.results)
            else:
                report_path = stage / f"{name}.svg"
                render_svg_report(report_path, name=name, results=derived.results)
            data_path = stage / f"{name}.json"
            data_path.write_text(
                json.dumps(
                    report_result_payload(name, derived.results),
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            sources[name] = ReportSource(
                path=report_path,
                data_path=data_path,
                generator_revision=generator_revision,
            )
        bundle = write_final_analysis_bundle(
            output,
            protocol=protocol,
            study=study,
            phase_run_directories={
                "E1": args.e1_run,
                "E3": args.e3_run,
                "E6": args.e6_run,
                "E7": args.e7_run,
                "E8": args.e8_run,
                "E9": args.e9_run,
                "E10": args.e10_run,
            },
            results=derived.results,
            analysis_evidence_directory=args.analysis_evidence,
            report_sources=sources,
            human_audit_queue_directory=args.audit_queue,
            human_audit_results_directory=args.audit_results,
            human_audit_blinding_key=blinding_key,
            derived_analysis=derived,
        )
    finally:
        shutil.rmtree(stage)
    _print(
        {
            "valid": True,
            "bundle": str(bundle.directory),
            "bundle_digest": bundle.bundle_digest,
            "report_artifact_count": len(bundle.report_artifacts),
        }
    )
    return 0


def _verify_analysis(args: argparse.Namespace) -> int:
    from mfh.analysis.human_audit import load_blinding_key
    from mfh.analysis.protocol import load_analysis_protocol
    from mfh.analysis.reporting import verify_final_analysis_bundle

    protocol = load_analysis_protocol(args.analysis_protocol)
    protocol.verify_research_plan(args.research_plan)
    study = load_study_protocol(args.study_protocol)
    derived = _derive_analysis_from_args(args, protocol=protocol, study=study)
    bundle = verify_final_analysis_bundle(
        args.bundle,
        expected_protocol=protocol,
        study=study,
        phase_run_directories={
            "E1": args.e1_run,
            "E3": args.e3_run,
            "E6": args.e6_run,
            "E7": args.e7_run,
            "E8": args.e8_run,
            "E9": args.e9_run,
            "E10": args.e10_run,
        },
        human_audit_blinding_key=load_blinding_key(args.blinding_key_file),
        expected_derivation=derived,
    )
    _print(
        {
            "valid": True,
            "analysis_protocol_digest": bundle.analysis_protocol_digest,
            "phase_completion_digests": dict(bundle.phase_completion_digests),
            "report_artifact_count": len(bundle.report_artifacts),
            "bundle_digest": bundle.bundle_digest,
        }
    )
    return 0


def _derive_analysis_from_args(
    args: argparse.Namespace,
    *,
    protocol: Any,
    study: Any,
) -> Any:
    from mfh.analysis.derivation import derive_final_analysis_from_artifacts
    from mfh.analysis.human_audit import load_blinding_key

    return derive_final_analysis_from_artifacts(
        protocol=protocol,
        study=study,
        phase_run_directories={
            "E1": args.e1_run,
            "E3": args.e3_run,
            "E6": args.e6_run,
            "E7": args.e7_run,
            "E8": args.e8_run,
            "E9": args.e9_run,
            "E10": args.e10_run,
        },
        robustness_result_directory=args.robustness_results,
        human_audit_queue_directory=args.audit_queue,
        human_audit_results_directory=args.audit_results,
        human_audit_blinding_key=load_blinding_key(args.blinding_key_file),
        aa_official_directory=args.aa_official,
        expected_aa_official_manifest_digest=args.expected_aa_official_manifest_digest,
    )


def _freeze_analysis_evidence(args: argparse.Namespace) -> int:
    from mfh.analysis.human_audit import load_blinding_key
    from mfh.analysis.protocol import load_analysis_protocol
    from mfh.analysis.reporting import write_frozen_analysis_evidence

    protocol = load_analysis_protocol(args.analysis_protocol)
    protocol.verify_research_plan(args.research_plan)
    study = load_study_protocol(args.study_protocol)
    evidence = write_frozen_analysis_evidence(
        args.output,
        protocol=protocol,
        study=study,
        phase_run_directories={
            "E1": args.e1_run,
            "E3": args.e3_run,
            "E6": args.e6_run,
            "E7": args.e7_run,
            "E8": args.e8_run,
            "E9": args.e9_run,
            "E10": args.e10_run,
        },
        robustness_result_directory=args.robustness_results,
        human_audit_queue_directory=args.audit_queue,
        human_audit_results_directory=args.audit_results,
        human_audit_blinding_key=load_blinding_key(args.blinding_key_file),
        aa_official_directory=args.aa_official,
        expected_aa_official_manifest_digest=args.expected_aa_official_manifest_digest,
    )
    _print(
        {
            "valid": True,
            "evidence_digest": evidence.evidence_digest,
            "results_sha256": evidence.results_sha256,
            "phase_completion_digests": dict(evidence.phase_completion_digests),
            "phase_records_sha256": dict(evidence.phase_records_sha256),
        }
    )
    return 0


def _verify_analysis_evidence(args: argparse.Namespace) -> int:
    from mfh.analysis.human_audit import load_blinding_key
    from mfh.analysis.protocol import load_analysis_protocol
    from mfh.analysis.reporting import verify_frozen_analysis_evidence

    protocol = load_analysis_protocol(args.analysis_protocol)
    protocol.verify_research_plan(args.research_plan)
    study = load_study_protocol(args.study_protocol)
    evidence = verify_frozen_analysis_evidence(
        args.evidence,
        expected_protocol=protocol,
        study=study,
        phase_run_directories={
            "E1": args.e1_run,
            "E3": args.e3_run,
            "E6": args.e6_run,
            "E7": args.e7_run,
            "E8": args.e8_run,
            "E9": args.e9_run,
            "E10": args.e10_run,
        },
        robustness_result_directory=args.robustness_results,
        human_audit_queue_directory=args.audit_queue,
        human_audit_results_directory=args.audit_results,
        human_audit_blinding_key=load_blinding_key(args.blinding_key_file),
        aa_official_directory=args.aa_official,
        expected_aa_official_manifest_digest=args.expected_aa_official_manifest_digest,
    )
    _print(
        {
            "valid": True,
            "evidence_digest": evidence.evidence_digest,
            "results_sha256": evidence.results_sha256,
            "phase_completion_digests": dict(evidence.phase_completion_digests),
            "phase_records_sha256": dict(evidence.phase_records_sha256),
        }
    )
    return 0


def _freeze_execution_snapshot(args: argparse.Namespace) -> int:
    from mfh.experiments.protocol import ExperimentPhase
    from mfh.experiments.snapshots import write_execution_snapshot

    study = load_study_protocol(args.study_protocol)
    fingerprint = write_execution_snapshot(
        args.output,
        study_protocol_digest=study.digest,
        phase=ExperimentPhase(args.phase),
        repository_root=args.repository_root,
    )
    _print({"valid": True, "snapshot_sha256": fingerprint})
    return 0


def _freeze_safety_scorer(args: argparse.Namespace) -> int:
    from mfh.evaluation.side_effects import write_side_effect_scorer_spec

    digest = write_side_effect_scorer_spec(
        args.output,
        execution_public_key=args.execution_public_key,
    )
    _print({"valid": True, "scorer_digest": digest})
    return 0


def _materialize_ifeval_evaluator(args: argparse.Namespace) -> int:
    from mfh.evaluation.ifeval import materialize_ifeval_evaluator

    fingerprint = materialize_ifeval_evaluator(args.output)
    _print({"valid": True, "ifeval_evaluator_sha256": fingerprint})
    return 0


def _materialize_strongreject_grader(args: argparse.Namespace) -> int:
    from mfh.evaluation.strongreject import materialize_strongreject_grader

    fingerprint = materialize_strongreject_grader(args.output)
    _print({"valid": True, "strongreject_grader_sha256": fingerprint})
    return 0


def _finalize_e7(args: argparse.Namespace) -> int:
    from mfh.experiments.e7_sparse import finalize_e7_phase

    result = finalize_e7_phase(
        args.output,
        ledger_directory=args.ledger,
        study=load_study_protocol(args.study_protocol),
        coordinate_artifact=args.coordinate_artifact,
        sae_intervention=args.sae_intervention,
    )
    _print(dict(result))
    return 0


def _verify_e7(args: argparse.Namespace) -> int:
    from mfh.experiments.e7_sparse import verify_e7_phase

    _print(dict(verify_e7_phase(args.output)))
    return 0


def _finalize_e8(args: argparse.Namespace) -> int:
    from mfh.experiments.e8_protected import finalize_e8_phase

    result = finalize_e8_phase(
        args.output,
        ledger_directory=args.ledger,
        study=load_study_protocol(args.study_protocol),
        protected_artifact=args.protected_artifact,
        operating_point_registry=args.operating_point_registry,
        candidate_screen=args.candidate_screen,
        runtime_artifact=args.runtime_artifact,
        analysis_protocol=args.analysis_protocol,
        research_plan=args.research_plan,
    )
    _print(dict(result))
    return 0


def _verify_e8(args: argparse.Namespace) -> int:
    from mfh.experiments.e8_protected import verify_e8_phase

    _print(dict(verify_e8_phase(args.output)))
    return 0


def _e6_runbook(args: argparse.Namespace) -> Any:
    from mfh.experiments.e6_operator import E6Runbook

    return E6Runbook.load(args.runbook)


def _write_e6_runbook(args: argparse.Namespace) -> int:
    from mfh.experiments.e6_operator import write_e6_runbook_template

    digest = write_e6_runbook_template(
        args.output,
        m1_layer=args.m1_layer,
        official_grader_bundle=args.official_grader_bundle,
        expected_grader_manifest_digest=args.expected_grader_manifest_digest,
    )
    _print({"valid": True, "runbook": str(args.output.resolve()), "sha256": digest})
    return 0


def _freeze_e6_questions(args: argparse.Namespace) -> int:
    from mfh.experiments.e6_operator import freeze_e6_question_bundle

    result = freeze_e6_question_bundle(
        args.output,
        reviewed_splits=args.reviewed_splits,
        expected_reviewed_split_manifest_digest=(
            args.expected_reviewed_split_manifest_digest
        ),
        source_artifacts={
            "triviaqa": args.triviaqa_source,
            "simpleqa_verified": args.simpleqa_source,
            "aa_omniscience_public_600": args.aa_source,
        },
        study_protocol=args.study_protocol,
        model_config=args.model_config,
        prompt_config=args.prompt_config,
        seed=args.seed,
    )
    _print(dict(result))
    return 0


def _preflight_e6(args: argparse.Namespace) -> int:
    from mfh.experiments.e6_operator import preflight_e6_runbook

    _print(dict(preflight_e6_runbook(_e6_runbook(args))))
    return 0


def _prepare_e6(args: argparse.Namespace) -> int:
    from mfh.experiments.e6_operator import prepare_e6_runbook

    ledger = prepare_e6_runbook(_e6_runbook(args))
    completed, expected = ledger.progress()
    _print(
        {
            "valid": True,
            "phase": "E6",
            "run_directory": str(ledger.directory),
            "contract_digest": ledger.contract.digest,
            "completed_records": completed,
            "expected_records": expected,
        }
    )
    return 0


def _attest_e6(args: argparse.Namespace) -> int:
    from mfh.experiments.e6_operator import attest_e6_runtime

    _print(dict(attest_e6_runtime(_e6_runbook(args))))
    return 0


def _run_e6(args: argparse.Namespace) -> int:
    from mfh.experiments.e6_operator import execute_e6_runbook

    _print(dict(execute_e6_runbook(_e6_runbook(args), limit=args.limit)))
    return 0


def _finalize_e6(args: argparse.Namespace) -> int:
    from mfh.experiments.e6_operator import finalize_e6_runbook

    _print(dict(finalize_e6_runbook(_e6_runbook(args))))
    return 0


def _verify_e6(args: argparse.Namespace) -> int:
    from mfh.experiments.e6_operator import verify_e6_runbook

    _print(dict(verify_e6_runbook(_e6_runbook(args))))
    return 0


def _e7_runbook(args: argparse.Namespace) -> Any:
    from mfh.experiments.e7_operator import E7Runbook

    return E7Runbook.load(args.runbook)


def _write_e7_runbook(args: argparse.Namespace) -> int:
    from mfh.experiments.e7_operator import write_e7_runbook_template

    digest = write_e7_runbook_template(args.output, m1_layer=args.m1_layer)
    _print({"valid": True, "runbook": str(args.output.resolve()), "sha256": digest})
    return 0


def _stage_e7_e8_inputs(args: argparse.Namespace) -> int:
    from mfh.experiments.e7_e8_inputs import stage_e7_e8_external_inputs

    result = stage_e7_e8_external_inputs(
        args.output,
        reviewed_splits=args.reviewed_splits,
        expected_reviewed_split_manifest_digest=(
            args.expected_reviewed_split_manifest_digest
        ),
        reviewed_language_suite=args.reviewed_language_suite,
        ifeval_evaluator=args.ifeval_evaluator,
        source_artifacts={
            "triviaqa": args.triviaqa_source,
            "ifeval": args.ifeval_source,
            "mmlu_pro": args.mmlu_pro_source,
            "wikitext103": args.wikitext103_source,
            "xstest": args.xstest_source,
            "strongreject_or_harmbench": args.strongreject_source,
        },
    )
    _print(dict(result))
    return 0


def _verify_e7_e8_inputs(args: argparse.Namespace) -> int:
    from mfh.experiments.e7_e8_inputs import validate_e7_e8_external_inputs

    result = validate_e7_e8_external_inputs(
        args.directory,
        expected_reviewed_split_manifest_digest=(
            args.expected_reviewed_split_manifest_digest
        ),
    )
    _print(dict(result))
    return 0


def _preflight_e7_runbook(args: argparse.Namespace) -> int:
    from mfh.experiments.e7_operator import preflight_e7_runbook

    _print(dict(preflight_e7_runbook(_e7_runbook(args))))
    return 0


def _prepare_e7_runbook(args: argparse.Namespace) -> int:
    from mfh.experiments.e7_operator import prepare_e7_runbook

    _print(dict(prepare_e7_runbook(_e7_runbook(args))))
    return 0


def _capture_e7(args: argparse.Namespace) -> int:
    from mfh.experiments.e7_operator import execute_e7_capture

    _print(dict(execute_e7_capture(_e7_runbook(args), partition=args.partition, limit=args.limit)))
    return 0


def _screen_e7_coordinate(args: argparse.Namespace) -> int:
    from mfh.experiments.e7_operator import execute_e7_coordinate_screen

    _print(dict(execute_e7_coordinate_screen(_e7_runbook(args), limit=args.limit)))
    return 0


def _fit_e7_sae(args: argparse.Namespace) -> int:
    from mfh.experiments.e7_operator import execute_e7_sae_sweep

    _print(dict(execute_e7_sae_sweep(_e7_runbook(args))))
    return 0


def _audit_e7_causal(args: argparse.Namespace) -> int:
    from mfh.experiments.e7_operator import execute_e7_causal_audit

    _print(dict(execute_e7_causal_audit(_e7_runbook(args), limit=args.limit)))
    return 0


def _audit_e7_interpretability(args: argparse.Namespace) -> int:
    from mfh.experiments.e7_operator import execute_e7_interpretability_audit

    _print(dict(execute_e7_interpretability_audit(_e7_runbook(args), limit=args.limit)))
    return 0


def _promote_e7_sae(args: argparse.Namespace) -> int:
    from mfh.experiments.e7_operator import promote_e7_sae

    _print(dict(promote_e7_sae(_e7_runbook(args))))
    return 0


def _prepare_e7_ledger(args: argparse.Namespace) -> int:
    from mfh.experiments.e7_operator import prepare_e7_ledger

    ledger = prepare_e7_ledger(_e7_runbook(args))
    completed, expected = ledger.progress()
    _print(
        {
            "valid": True,
            "phase": "E7",
            "run_directory": str(ledger.directory),
            "contract_digest": ledger.contract.digest,
            "completed_records": completed,
            "expected_records": expected,
        }
    )
    return 0


def _run_e7(args: argparse.Namespace) -> int:
    from mfh.experiments.e7_operator import execute_e7_final

    _print(dict(execute_e7_final(_e7_runbook(args), limit=args.limit)))
    return 0


def _finalize_e7_runbook(args: argparse.Namespace) -> int:
    from mfh.experiments.e7_operator import finalize_e7_runbook

    _print(dict(finalize_e7_runbook(_e7_runbook(args))))
    return 0


def _verify_e7_runbook(args: argparse.Namespace) -> int:
    from mfh.experiments.e7_operator import verify_e7_runbook

    _print(dict(verify_e7_runbook(_e7_runbook(args))))
    return 0


def _e8_runbook(args: argparse.Namespace) -> Any:
    from mfh.experiments.e8_operator import E8Runbook

    return E8Runbook.load(args.runbook)


def _write_e8_runbook(args: argparse.Namespace) -> int:
    from mfh.experiments.e8_operator import write_e8_runbook_template

    digest = write_e8_runbook_template(args.output, m1_layer=args.m1_layer)
    _print({"valid": True, "runbook": str(args.output.resolve()), "sha256": digest})
    return 0


def _preflight_e8_runbook(args: argparse.Namespace) -> int:
    from mfh.experiments.e8_operator import preflight_e8_runbook

    _print(dict(preflight_e8_runbook(_e8_runbook(args))))
    return 0


def _prepare_e8_runbook(args: argparse.Namespace) -> int:
    from mfh.experiments.e8_operator import prepare_e8_runbook

    _print(dict(prepare_e8_runbook(_e8_runbook(args))))
    return 0


def _capture_e8_activations(args: argparse.Namespace) -> int:
    from mfh.experiments.e8_operator import execute_e8_activation_capture

    _print(dict(execute_e8_activation_capture(_e8_runbook(args), limit=args.limit)))
    return 0


def _screen_e8_variants(args: argparse.Namespace) -> int:
    from mfh.experiments.e8_operator import execute_e8_variant_screen

    _print(dict(execute_e8_variant_screen(_e8_runbook(args), limit=args.limit)))
    return 0


def _promote_e8_protected(args: argparse.Namespace) -> int:
    from mfh.experiments.e8_operator import promote_e8_protected_artifact

    _print(dict(promote_e8_protected_artifact(_e8_runbook(args))))
    return 0


def _screen_e8_candidates(args: argparse.Namespace) -> int:
    from mfh.experiments.e8_operator import execute_e8_candidate_screen

    _print(dict(execute_e8_candidate_screen(_e8_runbook(args), limit=args.limit)))
    return 0


def _prepare_e8_ledger(args: argparse.Namespace) -> int:
    from mfh.experiments.e8_operator import prepare_e8_ledger

    ledger = prepare_e8_ledger(_e8_runbook(args))
    completed, expected = ledger.progress()
    _print(
        {
            "valid": True,
            "phase": "E8",
            "run_directory": str(ledger.directory),
            "contract_digest": ledger.contract.digest,
            "completed_records": completed,
            "expected_records": expected,
        }
    )
    return 0


def _run_e8(args: argparse.Namespace) -> int:
    from mfh.experiments.e8_operator import execute_e8_final

    _print(dict(execute_e8_final(_e8_runbook(args), limit=args.limit)))
    return 0


def _finalize_e8_runbook(args: argparse.Namespace) -> int:
    from mfh.experiments.e8_operator import finalize_e8_runbook

    _print(dict(finalize_e8_runbook(_e8_runbook(args))))
    return 0


def _verify_e8_runbook(args: argparse.Namespace) -> int:
    from mfh.experiments.e8_operator import verify_e8_runbook

    _print(dict(verify_e8_runbook(_e8_runbook(args))))
    return 0


def _stage_e9_inputs(args: argparse.Namespace) -> int:
    from mfh.experiments.e9_freeze_operator import stage_e9_external_inputs

    _print(
        dict(
            stage_e9_external_inputs(
                args.output,
                official_grader_bundle=args.official_grader_bundle,
                expected_official_grader_manifest_digest=(
                    args.expected_official_grader_manifest_digest
                ),
                reviewed_splits=args.reviewed_splits,
                source_artifacts={
                    "triviaqa": args.triviaqa_source,
                    "simpleqa_verified": args.simpleqa_source,
                    "aa_omniscience_public_600": args.aa_source,
                },
            )
        )
    )
    return 0


def _freeze_e9_inputs(args: argparse.Namespace) -> int:
    from mfh.experiments.e1_vllm import load_env_secret
    from mfh.experiments.e9_freeze_operator import freeze_e9_input_suite

    result = freeze_e9_input_suite(
        args.output,
        e8_runbook=args.e8_runbook,
        e9_runbook_output=args.e9_runbook_output,
        evaluation_scripts=args.evaluation_scripts,
        official_grader_bundle=args.official_grader_bundle,
        expected_official_grader_manifest_digest=(
            args.expected_official_grader_manifest_digest
        ),
        reviewed_splits=args.reviewed_splits,
        source_artifacts={
            "triviaqa": args.triviaqa_source,
            "simpleqa_verified": args.simpleqa_source,
            "aa_omniscience_public_600": args.aa_source,
        },
        m2_source_artifact=args.m2_source_artifact,
        e3_phase_run=args.e3_phase_run,
        execution_private_key=load_env_secret(
            args.env_file, "MFH_EXECUTION_PRIVATE_KEY"
        ),
        robustness_config=args.robustness_config,
    )
    _print(dict(result))
    return 0


def _prepare_e10_freezes(args: argparse.Namespace) -> int:
    from mfh.experiments.e10_freeze_operator import prepare_e10_freeze_suite

    _print(
        dict(
            prepare_e10_freeze_suite(
                args.output,
                e8_runbook=args.e8_runbook,
                e9_runbook=args.e9_runbook,
            )
        )
    )
    return 0


def _run_e10_early_probe(args: argparse.Namespace) -> int:
    from mfh.experiments.e1_vllm import load_env_secret
    from mfh.experiments.e10_freeze_operator import run_e10_freeze_capture

    _print(
        dict(
            run_e10_freeze_capture(
                args.output,
                e8_runbook=args.e8_runbook,
                e9_runbook=args.e9_runbook,
                execution_private_key=load_env_secret(
                    args.env_file, "MFH_EXECUTION_PRIVATE_KEY"
                ),
                limit=args.limit,
                shard_rows=args.shard_rows,
            )
        )
    )
    return 0


def _verify_e10_early_probe(args: argparse.Namespace) -> int:
    from mfh.experiments.e10_freeze_operator import verify_e10_freeze_capture

    _print(
        dict(
            verify_e10_freeze_capture(
                args.output,
                require_complete=args.require_complete,
            )
        )
    )
    return 0


def _finalize_e10_freezes(args: argparse.Namespace) -> int:
    from mfh.experiments.e10_freeze_operator import finalize_e10_freeze_suite

    _print(
        dict(
            finalize_e10_freeze_suite(
                args.output,
                e8_runbook=args.e8_runbook,
                e9_runbook=args.e9_runbook,
                evaluation_scripts=args.evaluation_scripts,
                e10_runbook_output=args.e10_runbook_output,
            )
        )
    )
    return 0


def _freeze_robustness_plan(args: argparse.Namespace) -> int:
    from mfh.experiments.e1_vllm import load_env_secret
    from mfh.experiments.robustness_diagnostics import (
        freeze_robustness_diagnostic_plan,
    )

    plan = freeze_robustness_diagnostic_plan(
        args.output,
        config_path=args.config,
        source_artifacts={
            "canonical-prompts": args.prompt_config,
            "frozen-component-selection": args.component_selection,
            "frozen-evaluation-scripts": args.evaluation_scripts,
            "frozen-graders": args.graders,
            "e1-phase-ledger": args.e1_run,
            "triviaqa-evaluation": args.reviewed_splits,
            "simpleqa_verified-evaluation": args.reviewed_splits,
            "aa_omniscience_public_600-evaluation": args.reviewed_splits,
            "triviaqa-development": args.reviewed_splits,
        },
        completion_execution_private_key=load_env_secret(
            args.env_file, "MFH_EXECUTION_PRIVATE_KEY"
        ),
    )
    _print(
        {
            "valid": True,
            "bundle": str(args.output.resolve()),
            "plan_digest": plan.plan_digest,
            "prompt_paraphrase_tasks": plan.body["prompt_paraphrase"]["expected_task_count"],
            "rq1_generalization_tasks": plan.body["rq1_generalization"]["expected_task_count"],
        }
    )
    return 0


def _verify_robustness_plan(args: argparse.Namespace) -> int:
    from mfh.experiments.robustness_diagnostics import (
        verify_robustness_diagnostic_plan,
    )

    plan = verify_robustness_diagnostic_plan(args.plan)
    _print(
        {
            "valid": True,
            "bundle": str(args.plan.resolve()),
            "plan_digest": plan.plan_digest,
            "prompt_paraphrase_tasks": plan.body["prompt_paraphrase"]["expected_task_count"],
            "rq1_generalization_tasks": plan.body["rq1_generalization"]["expected_task_count"],
        }
    )
    return 0


def _create_robustness_results(args: argparse.Namespace) -> int:
    from mfh.experiments.robustness_results import (
        create_robustness_result_store,
        robustness_result_progress,
    )

    store = create_robustness_result_store(
        args.output,
        plan_path=args.plan,
        config_path=args.config,
    )
    _print(
        {
            "valid": True,
            "directory": str(store.directory),
            "plan_digest": store.plan.plan_digest,
            "progress": dict(robustness_result_progress(store)),
        }
    )
    return 0


def _prepare_robustness_execution(args: argparse.Namespace) -> int:
    from mfh.experiments.robustness_operator import prepare_robustness_execution

    _print(
        dict(
            prepare_robustness_execution(
                args.output,
                results=args.results,
                e9_runbook=args.e9_runbook,
                e3_construction=args.e3_construction,
                shard_rows=args.shard_rows,
            )
        )
    )
    return 0


def _run_robustness_rq1_capture(args: argparse.Namespace) -> int:
    from mfh.experiments.e1_vllm import load_env_secret
    from mfh.experiments.robustness_operator import run_robustness_rq1_capture

    _print(
        dict(
            run_robustness_rq1_capture(
                args.output,
                results=args.results,
                e9_runbook=args.e9_runbook,
                e3_construction=args.e3_construction,
                execution_private_key=load_env_secret(
                    args.env_file, "MFH_EXECUTION_PRIVATE_KEY"
                ),
                limit=args.limit,
            )
        )
    )
    return 0


def _verify_robustness_rq1_capture(args: argparse.Namespace) -> int:
    from mfh.experiments.e1_vllm import load_env_secret
    from mfh.experiments.robustness_operator import verify_robustness_rq1_capture

    _print(
        dict(
            verify_robustness_rq1_capture(
                args.output,
                results=args.results,
                e9_runbook=args.e9_runbook,
                e3_construction=args.e3_construction,
                execution_private_key=load_env_secret(
                    args.env_file, "MFH_EXECUTION_PRIVATE_KEY"
                ),
                require_complete=args.require_complete,
            )
        )
    )
    return 0


def _run_robustness_prompts(args: argparse.Namespace) -> int:
    from mfh.experiments.e1_vllm import load_env_secret
    from mfh.experiments.robustness_operator import (
        run_prompt_paraphrase_diagnostics,
    )

    _print(
        dict(
            run_prompt_paraphrase_diagnostics(
                args.results,
                e9_runbook=args.e9_runbook,
                execution_private_key=load_env_secret(
                    args.env_file, "MFH_EXECUTION_PRIVATE_KEY"
                ),
                openrouter_api_key=load_env_secret(
                    args.env_file, "OPENROUTER_API_KEY"
                ),
                limit=args.limit,
            )
        )
    )
    return 0


def _run_robustness_rq1(args: argparse.Namespace) -> int:
    from mfh.experiments.e1_vllm import load_env_secret
    from mfh.experiments.robustness_operator import (
        run_rq1_generalization_diagnostics,
    )

    _print(
        dict(
            run_rq1_generalization_diagnostics(
                args.output,
                results=args.results,
                e9_runbook=args.e9_runbook,
                e3_construction=args.e3_construction,
                e2_workspace=args.e2_workspace,
                e2_probe_bundle=args.e2_probe_bundle,
                e5_fit_capture=args.e5_fit_capture,
                e5_layer_labels=args.e5_layer_labels,
                controller_questions=args.controller_questions,
                execution_private_key=load_env_secret(
                    args.env_file, "MFH_EXECUTION_PRIVATE_KEY"
                ),
                openrouter_api_key=load_env_secret(
                    args.env_file, "OPENROUTER_API_KEY"
                ),
                limit=args.limit,
            )
        )
    )
    return 0


def _verify_robustness_results(args: argparse.Namespace) -> int:
    from mfh.experiments.robustness_results import (
        robustness_result_progress,
        verify_robustness_result_store,
    )

    store = verify_robustness_result_store(
        args.results,
        require_complete=args.require_complete,
    )
    _print(
        {
            "valid": True,
            "directory": str(store.directory),
            "plan_digest": store.plan.plan_digest,
            "complete": (store.directory / "complete.json").is_file(),
            "progress": dict(robustness_result_progress(store)),
        }
    )
    return 0


def _finalize_robustness_results(args: argparse.Namespace) -> int:
    from mfh.experiments.robustness_results import (
        finalize_robustness_result_store,
        open_robustness_result_store,
    )

    store = open_robustness_result_store(args.results)
    digest = finalize_robustness_result_store(store)
    _print(
        {
            "valid": True,
            "directory": str(store.directory),
            "completion_digest": digest,
        }
    )
    return 0


def _confirmatory_runbook(args: argparse.Namespace) -> Any:
    from mfh.experiments.confirmatory_operator import ConfirmatoryRunbook

    return ConfirmatoryRunbook.load(args.runbook)


def _write_confirmatory_runbook(args: argparse.Namespace) -> int:
    from mfh.experiments.confirmatory_operator import (
        write_confirmatory_runbook_template,
    )
    from mfh.experiments.protocol import ExperimentPhase

    digest = write_confirmatory_runbook_template(
        args.output,
        phase=ExperimentPhase(args.phase),
    )
    _print({"valid": True, "runbook": str(args.output.resolve()), "sha256": digest})
    return 0


def _preflight_confirmatory(args: argparse.Namespace) -> int:
    from mfh.experiments.confirmatory_operator import preflight_confirmatory_runbook

    _print(dict(preflight_confirmatory_runbook(_confirmatory_runbook(args))))
    return 0


def _prepare_confirmatory(args: argparse.Namespace) -> int:
    from mfh.experiments.confirmatory_operator import prepare_confirmatory_runbook

    runbook = _confirmatory_runbook(args)
    ledger = prepare_confirmatory_runbook(
        runbook,
        authorize_one_shot=args.authorize_e10_one_shot,
    )
    completed, expected = ledger.progress()
    _print(
        {
            "valid": True,
            "phase": ledger.contract.phase.value,
            "run_directory": str(ledger.directory),
            "contract_digest": ledger.contract.digest,
            "completed_records": completed,
            "expected_records": expected,
        }
    )
    return 0


def _run_confirmatory(args: argparse.Namespace) -> int:
    from mfh.experiments.confirmatory_operator import execute_confirmatory_runbook
    from mfh.experiments.e1_vllm import load_env_secret

    result = execute_confirmatory_runbook(
        _confirmatory_runbook(args),
        execution_private_key=load_env_secret(args.env_file, "MFH_EXECUTION_PRIVATE_KEY"),
        openrouter_api_key=load_env_secret(args.env_file, "OPENROUTER_API_KEY"),
        checkpoint_size=args.checkpoint_size,
        limit=args.limit,
    )
    _print(dict(result))
    return 0


def _finalize_confirmatory(args: argparse.Namespace) -> int:
    from mfh.experiments.confirmatory_operator import finalize_confirmatory_runbook

    _print(dict(finalize_confirmatory_runbook(_confirmatory_runbook(args))))
    return 0


def _verify_confirmatory(args: argparse.Namespace) -> int:
    from mfh.experiments.confirmatory_operator import verify_confirmatory_runbook

    _print(dict(verify_confirmatory_runbook(_confirmatory_runbook(args))))
    return 0


def _language_rows(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("each translated-language row must be a JSON object")
            rows.append(value)
    return tuple(rows)


def _build_language_suite(args: argparse.Namespace) -> int:
    from mfh.data.language_suite import write_reviewed_language_suite

    reviewers = json.loads(args.reviewer_registry.read_text(encoding="utf-8"))
    if not isinstance(reviewers, dict):
        raise ValueError("reviewer registry must be a JSON object")
    fingerprint = write_reviewed_language_suite(
        args.output,
        triviaqa_source=args.triviaqa_source,
        rows=_language_rows(args.translations),
        reviewer_public_keys={str(key): str(value) for key, value in reviewers.items()},
    )
    _print({"valid": True, "language_suite_sha256": fingerprint})
    return 0


def _verify_language_suite(path: Path) -> int:
    from mfh.data.language_suite import load_reviewed_language_suite

    questions = load_reviewed_language_suite(path)
    _print(
        {
            "valid": True,
            "question_count": len(questions),
            "languages": sorted({str(value.metadata["requested_language"]) for value in questions}),
        }
    )
    return 0


def _transformers_snapshot_preflight(args: argparse.Namespace) -> int:
    from mfh.inference.transformers_snapshot import verify_transformers_snapshot

    model = load_model_spec(args.model_config)
    identity = verify_transformers_snapshot(model, args.snapshot_directory, args.snapshot_manifest)
    _print({"valid": True, **identity})
    return 0


def _vllm_hook_preflight(args: argparse.Namespace) -> int:
    from mfh.inference.vllm_preflight import run_vllm_preflight

    receipt = run_vllm_preflight(
        project_root=args.project_root,
        model_directory=args.snapshot_directory,
        model_config=args.model_config,
        snapshot_manifest=args.snapshot_manifest,
        runtime_policy=args.runtime_policy,
        output=args.output,
        prompt=args.prompt,
        alpha=args.alpha,
    )
    _print(
        {
            "valid": True,
            "receipt_digest": receipt["receipt_digest"],
            "status": receipt["status"],
            "peak_memory_bytes": receipt["peak_memory_bytes"],
        }
    )
    return 0


def _run_e0_vllm(args: argparse.Namespace) -> int:
    from mfh.experiments.e0_vllm import run_vllm_e0

    result = run_vllm_e0(
        cohort_directory=args.cohort,
        reserved_source=args.reserved_source,
        expected_cohort_manifest_digest=args.expected_cohort_manifest_digest,
        parent_split_manifest_digest=args.parent_split_manifest_digest,
        contamination_manifest_digest=args.contamination_manifest_digest,
        model_config=args.model_config,
        snapshot_directory=args.snapshot_directory,
        snapshot_manifest=args.snapshot_manifest,
        runtime_config=args.runtime_config,
        prompt_config=args.prompt_config,
        inference_config=args.inference_config,
        study_config=args.study_config,
        work_directory=args.work,
        output_directory=args.output,
        request_budget=args.request_budget,
        expected_resume_checkpoint=args.expected_resume_checkpoint,
        checkpoint_file=args.checkpoint_file,
    )
    _print(result)
    return 0


def _verify_e0_vllm(args: argparse.Namespace) -> int:
    from mfh.experiments.e0_vllm import verify_vllm_e0_bundle

    manifest = verify_vllm_e0_bundle(
        args.directory,
        expected_manifest_digest=args.expected_manifest_digest,
        expected_plan_identity=args.expected_plan_identity,
        cohort_directory=args.cohort,
        reserved_source=args.reserved_source,
        expected_cohort_manifest_digest=args.expected_cohort_manifest_digest,
        parent_split_manifest_digest=args.parent_split_manifest_digest,
        contamination_manifest_digest=args.contamination_manifest_digest,
        model_config=args.model_config,
        snapshot_directory=args.snapshot_directory,
        snapshot_manifest=args.snapshot_manifest,
        runtime_config=args.runtime_config,
        prompt_config=args.prompt_config,
        inference_config=args.inference_config,
        study_config=args.study_config,
    )
    _print(
        {
            "valid": True,
            "manifest_digest": manifest["manifest_digest"],
            "plan_identity": manifest["plan_identity"],
            "model_name": manifest["model_name"],
            "counts": manifest["counts"],
            "scientific_status": manifest["scientific_status"],
        }
    )
    return 0


def _write_e0_completion(args: argparse.Namespace) -> int:
    from mfh.experiments.e0_completion import write_e0_completion_receipt

    manifest = write_e0_completion_receipt(
        args.output,
        vllm_directory=args.vllm_directory,
        expected_vllm_manifest_digest=args.expected_vllm_manifest_digest,
        expected_vllm_plan_identity=args.expected_vllm_plan_identity,
        vllm_inputs=_e0_vllm_completion_inputs(args),
        review_result_directory=args.review_result,
        expected_review_result_manifest_digest=args.expected_review_result_manifest_digest,
        review_queue_directory=args.review_queue,
        expected_review_queue_manifest_digest=args.expected_review_queue_manifest_digest,
        review_inputs={
            "contamination_directory": args.contamination_bundle,
            "expected_protocol": load_semantic_contamination_protocol(args.config),
            "model_directory": args.model_directory,
            "triviaqa_source": args.triviaqa_source,
            "target_sources": args.target,
            "expected_contamination_manifest_digest": args.contamination_manifest_digest,
        },
        grader_bundle=args.grader_bundle,
        expected_grader_manifest_digest=args.expected_grader_manifest_digest,
        reviewed_splits=args.reviewed_splits,
        expected_reviewed_split_manifest_digest=args.expected_reviewed_split_manifest_digest,
    )
    _print(
        {
            "valid": True,
            "manifest_digest": manifest["manifest_digest"],
            "status": manifest["status"],
            "scientific_eligible": manifest["scientific_eligible"],
        }
    )
    return 0


def _verify_e0_completion(args: argparse.Namespace) -> int:
    from mfh.experiments.e0_completion import verify_e0_completion_receipt

    manifest = verify_e0_completion_receipt(
        args.receipt,
        expected_manifest_digest=args.expected_manifest_digest,
        vllm_directory=args.vllm_directory,
        expected_vllm_manifest_digest=args.expected_vllm_manifest_digest,
        expected_vllm_plan_identity=args.expected_vllm_plan_identity,
        vllm_inputs=_e0_vllm_completion_inputs(args),
        review_result_directory=args.review_result,
        expected_review_result_manifest_digest=args.expected_review_result_manifest_digest,
        review_queue_directory=args.review_queue,
        expected_review_queue_manifest_digest=args.expected_review_queue_manifest_digest,
        review_inputs={
            "contamination_directory": args.contamination_bundle,
            "expected_protocol": load_semantic_contamination_protocol(args.config),
            "model_directory": args.model_directory,
            "triviaqa_source": args.triviaqa_source,
            "target_sources": args.target,
            "expected_contamination_manifest_digest": args.contamination_manifest_digest,
        },
        grader_bundle=args.grader_bundle,
        expected_grader_manifest_digest=args.expected_grader_manifest_digest,
        reviewed_splits=args.reviewed_splits,
        expected_reviewed_split_manifest_digest=args.expected_reviewed_split_manifest_digest,
    )
    _print(
        {
            "valid": True,
            "manifest_digest": manifest["manifest_digest"],
            "status": manifest["status"],
            "scientific_eligible": manifest["scientific_eligible"],
        }
    )
    return 0


def _finalize_e0_phase(args: argparse.Namespace) -> int:
    from mfh.experiments.e0_phase import finalize_e0_phase_run

    completion = finalize_e0_phase_run(
        args.output,
        completion_receipt=args.receipt,
        expected_completion_manifest_digest=args.expected_manifest_digest,
        vllm_directory=args.vllm_directory,
        expected_vllm_manifest_digest=args.expected_vllm_manifest_digest,
        expected_vllm_plan_identity=args.expected_vllm_plan_identity,
        vllm_inputs=_e0_vllm_completion_inputs(args),
        review_result_directory=args.review_result,
        expected_review_result_manifest_digest=args.expected_review_result_manifest_digest,
        review_queue_directory=args.review_queue,
        expected_review_queue_manifest_digest=args.expected_review_queue_manifest_digest,
        review_inputs={
            "contamination_directory": args.contamination_bundle,
            "expected_protocol": load_semantic_contamination_protocol(args.config),
            "model_directory": args.model_directory,
            "triviaqa_source": args.triviaqa_source,
            "target_sources": args.target,
            "expected_contamination_manifest_digest": args.contamination_manifest_digest,
        },
        grader_bundle=args.grader_bundle,
        expected_grader_manifest_digest=args.expected_grader_manifest_digest,
        reviewed_splits=args.reviewed_splits,
        expected_reviewed_split_manifest_digest=args.expected_reviewed_split_manifest_digest,
    )
    _print(
        {
            "valid": True,
            "phase": completion.phase.value,
            "completion_digest": completion.completion_digest,
            "record_count": completion.record_count,
        }
    )
    return 0


def _e0_vllm_completion_inputs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "cohort_directory": args.cohort,
        "reserved_source": args.reserved_source,
        "expected_cohort_manifest_digest": args.expected_cohort_manifest_digest,
        "parent_split_manifest_digest": args.parent_split_manifest_digest,
        "contamination_manifest_digest": args.contamination_manifest_digest,
        "model_config": args.model_config,
        "snapshot_directory": args.snapshot_directory,
        "snapshot_manifest": args.snapshot_manifest,
        "runtime_config": args.runtime_config,
        "prompt_config": args.prompt_config,
        "inference_config": args.inference_config,
        "study_config": args.study_config,
    }


def _add_e0_completion_evidence_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("vllm_directory", type=Path)
    parser.add_argument("review_result", type=Path)
    parser.add_argument("review_queue", type=Path)
    parser.add_argument("contamination_bundle", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("model_directory", type=Path)
    parser.add_argument("triviaqa_source", type=Path)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("reserved_source", type=Path)
    parser.add_argument("model_config", type=Path)
    parser.add_argument("snapshot_directory", type=Path)
    parser.add_argument("snapshot_manifest", type=Path)
    parser.add_argument("runtime_config", type=Path)
    parser.add_argument("grader_bundle", type=Path)
    parser.add_argument("reviewed_splits", type=Path)
    parser.add_argument("--target", type=Path, action="append", required=True)
    parser.add_argument("--expected-vllm-manifest-digest", required=True)
    parser.add_argument("--expected-vllm-plan-identity", required=True)
    parser.add_argument("--expected-review-result-manifest-digest", required=True)
    parser.add_argument("--expected-review-queue-manifest-digest", required=True)
    parser.add_argument("--expected-cohort-manifest-digest", required=True)
    parser.add_argument("--parent-split-manifest-digest", required=True)
    parser.add_argument("--contamination-manifest-digest", required=True)
    parser.add_argument("--expected-grader-manifest-digest", required=True)
    parser.add_argument("--expected-reviewed-split-manifest-digest", required=True)
    parser.add_argument("--prompt-config", type=Path, default=Path("configs/prompts/primary.yaml"))
    parser.add_argument(
        "--inference-config", type=Path, default=Path("configs/experiments/core.yaml")
    )
    parser.add_argument(
        "--study-config", type=Path, default=Path("configs/experiments/phases.yaml")
    )


def _synthetic_smoke(args: argparse.Namespace) -> int:
    from mfh.experiments.synthetic import run_synthetic_study

    bundle = run_synthetic_study(args.output, seed=args.seed)
    _print(
        {
            "valid": True,
            "scientific_eligible": bundle.scientific_eligible,
            "seed": bundle.seed,
            "phase_digests": dict(bundle.phase_digests),
            "bundle_digest": bundle.bundle_digest,
            "directory": str(bundle.directory),
        }
    )
    return 0


def _verify_synthetic_smoke(args: argparse.Namespace) -> int:
    from mfh.experiments.synthetic import verify_synthetic_study

    bundle = verify_synthetic_study(args.directory, replay=True)
    _print(
        {
            "valid": True,
            "scientific_eligible": bundle.scientific_eligible,
            "seed": bundle.seed,
            "phase_digests": dict(bundle.phase_digests),
            "bundle_digest": bundle.bundle_digest,
        }
    )
    return 0


def _prepare_human_audit(args: argparse.Namespace) -> int:
    from mfh.analysis.human_audit import load_blinding_key, prepare_human_audit
    from mfh.analysis.protocol import load_analysis_protocol

    result = prepare_human_audit(
        args.output,
        study=load_study_protocol(args.study_protocol),
        phase_run_directories={"E9": args.e9_run, "E10": args.e10_run},
        protocol=load_analysis_protocol(args.analysis_protocol),
        blinding_key=load_blinding_key(args.blinding_key_file),
    )
    _print(
        {
            "valid": True,
            "audit_rows": len(result.bindings),
            "manifest_digest": result.manifest_digest,
            "blinded_export": str(result.directory / "blind-items.jsonl"),
            "annotation_template": str(result.directory / "annotation-template.csv"),
            "operator_bindings": str(result.directory / "operator-bindings.jsonl"),
        }
    )
    return 0


def _annotation_arguments(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        identifier, separator, path = value.partition("=")
        identifier = identifier.strip()
        if not separator or not identifier or not path or identifier in result:
            raise ValueError("--annotation values must be unique ANNOTATOR_ID=CSV pairs")
        result[identifier] = Path(path)
    return result


def _finalize_human_audit(args: argparse.Namespace) -> int:
    from mfh.analysis.human_audit import finalize_human_audit, load_blinding_key
    from mfh.analysis.protocol import load_analysis_protocol

    result = finalize_human_audit(
        args.queue,
        args.output,
        annotations=_annotation_arguments(args.annotation),
        adjudications=args.adjudications,
        expected_protocol=load_analysis_protocol(args.analysis_protocol),
        study=load_study_protocol(args.study_protocol),
        phase_run_directories={"E9": args.e9_run, "E10": args.e10_run},
        blinding_key=load_blinding_key(args.blinding_key_file),
    )
    _print(
        {
            "valid": True,
            "manifest_digest": result.manifest_digest,
            "queue_manifest_digest": result.queue_manifest_digest,
            "summary": result.summary,
        }
    )
    return 0


def _verify_human_audit_queue(args: argparse.Namespace) -> int:
    from mfh.analysis.human_audit import load_blinding_key, verify_human_audit_queue
    from mfh.analysis.protocol import load_analysis_protocol

    result = verify_human_audit_queue(
        args.queue,
        expected_protocol=load_analysis_protocol(args.analysis_protocol),
        study=load_study_protocol(args.study_protocol),
        phase_run_directories={"E9": args.e9_run, "E10": args.e10_run},
        blinding_key=load_blinding_key(args.blinding_key_file),
    )
    _print(
        {
            "valid": True,
            "audit_rows": len(result.bindings),
            "manifest_digest": result.manifest_digest,
        }
    )
    return 0


def _verify_human_audit_results(args: argparse.Namespace) -> int:
    from mfh.analysis.human_audit import load_blinding_key, verify_human_audit_results
    from mfh.analysis.protocol import load_analysis_protocol

    result = verify_human_audit_results(
        args.results,
        queue_directory=args.queue,
        expected_protocol=load_analysis_protocol(args.analysis_protocol),
        study=load_study_protocol(args.study_protocol),
        phase_run_directories={"E9": args.e9_run, "E10": args.e10_run},
        blinding_key=load_blinding_key(args.blinding_key_file),
    )
    _print(
        {
            "valid": True,
            "manifest_digest": result.manifest_digest,
            "queue_manifest_digest": result.queue_manifest_digest,
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mfh", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    config_parser = subparsers.add_parser("validate-config", help="strictly validate YAML")
    config_parser.add_argument("path", type=Path)
    config_parser.set_defaults(handler=lambda args: _validate_config(args.path))

    split_parser = subparsers.add_parser("split", help="create disjoint TriviaQA research splits")
    split_parser.add_argument("input", type=Path)
    split_parser.add_argument("output", type=Path)
    split_parser.add_argument("--steer", type=int, default=30_000)
    split_parser.add_argument("--controller", type=int, default=5_000)
    split_parser.add_argument("--dev", type=int, default=5_000)
    split_parser.add_argument("--test", type=int, default=5_000)
    split_parser.add_argument("--seed", type=int, default=17)
    split_parser.add_argument("--allow-underfill", action="store_true")
    split_parser.add_argument(
        "--exclude-exact-duplicate-groups",
        action="store_true",
        help=(
            "exclude every normalized-question collision group and publish an auditable "
            "curation report instead of adjudicating released answers"
        ),
    )
    split_parser.add_argument("--overwrite", action="store_true")
    split_parser.set_defaults(handler=_split)

    overlap_parser = subparsers.add_parser("overlap", help="find source/target contamination")
    overlap_parser.add_argument("source", type=Path)
    overlap_parser.add_argument("target", type=Path)
    overlap_parser.add_argument("--ngram-threshold", type=float, default=0.8)
    overlap_parser.add_argument("--limit", type=int, default=100)
    overlap_parser.set_defaults(handler=_overlap)

    contamination_parser = subparsers.add_parser(
        "contamination-scan",
        help="freeze duplicate, lexical, and semantic TriviaQA-to-OOD overlap evidence",
    )
    contamination_parser.add_argument("config", type=Path)
    contamination_parser.add_argument("model_directory", type=Path)
    contamination_parser.add_argument("triviaqa_source", type=Path)
    contamination_parser.add_argument("output", type=Path)
    contamination_parser.add_argument("--target", type=Path, action="append", required=True)
    contamination_parser.set_defaults(handler=_contamination_scan)

    verify_contamination_parser = subparsers.add_parser(
        "verify-contamination-scan",
        help="verify a frozen lexical and semantic contamination bundle",
    )
    verify_contamination_parser.add_argument("bundle", type=Path)
    verify_contamination_parser.add_argument("config", type=Path)
    verify_contamination_parser.add_argument("model_directory", type=Path)
    verify_contamination_parser.add_argument("triviaqa_source", type=Path)
    verify_contamination_parser.add_argument("--target", type=Path, action="append", required=True)
    verify_contamination_parser.add_argument(
        "--expected-manifest-digest",
        required=True,
        help="externally recorded SHA-256 digest printed when the bundle was created",
    )
    verify_contamination_parser.add_argument("--replay-embeddings", action="store_true")
    verify_contamination_parser.set_defaults(handler=_verify_contamination_scan)

    contamination_review_parser = subparsers.add_parser(
        "prepare-contamination-review",
        help="publish a blinded human-review queue for semantic-contamination candidates",
    )
    contamination_review_parser.add_argument("contamination_bundle", type=Path)
    contamination_review_parser.add_argument("config", type=Path)
    contamination_review_parser.add_argument("model_directory", type=Path)
    contamination_review_parser.add_argument("triviaqa_source", type=Path)
    contamination_review_parser.add_argument("output", type=Path)
    contamination_review_parser.add_argument("--target", type=Path, action="append", required=True)
    contamination_review_parser.add_argument(
        "--expected-contamination-manifest-digest", required=True
    )
    contamination_review_parser.add_argument("--seed", type=int, default=17)
    contamination_review_parser.set_defaults(handler=_prepare_contamination_review)

    verify_contamination_review_queue_parser = subparsers.add_parser(
        "verify-contamination-review-queue",
        help="replay a blinded contamination-review queue against the frozen scan",
    )
    verify_contamination_review_queue_parser.add_argument("review_queue", type=Path)
    verify_contamination_review_queue_parser.add_argument("contamination_bundle", type=Path)
    verify_contamination_review_queue_parser.add_argument("config", type=Path)
    verify_contamination_review_queue_parser.add_argument("model_directory", type=Path)
    verify_contamination_review_queue_parser.add_argument("triviaqa_source", type=Path)
    verify_contamination_review_queue_parser.add_argument(
        "--target", type=Path, action="append", required=True
    )
    verify_contamination_review_queue_parser.add_argument(
        "--expected-contamination-manifest-digest", required=True
    )
    verify_contamination_review_queue_parser.add_argument(
        "--expected-review-queue-manifest-digest", required=True
    )
    verify_contamination_review_queue_parser.set_defaults(
        handler=_verify_contamination_review_queue
    )

    finalize_contamination_review_parser = subparsers.add_parser(
        "finalize-contamination-review",
        help="freeze complete human overlap decisions and the reviewed clean source",
    )
    finalize_contamination_review_parser.add_argument("review_queue", type=Path)
    finalize_contamination_review_parser.add_argument("annotations", type=Path)
    finalize_contamination_review_parser.add_argument("reviewer_attestation", type=Path)
    finalize_contamination_review_parser.add_argument("contamination_bundle", type=Path)
    finalize_contamination_review_parser.add_argument("config", type=Path)
    finalize_contamination_review_parser.add_argument("model_directory", type=Path)
    finalize_contamination_review_parser.add_argument("triviaqa_source", type=Path)
    finalize_contamination_review_parser.add_argument("output", type=Path)
    finalize_contamination_review_parser.add_argument(
        "--target", type=Path, action="append", required=True
    )
    finalize_contamination_review_parser.add_argument(
        "--expected-contamination-manifest-digest", required=True
    )
    finalize_contamination_review_parser.add_argument(
        "--expected-review-queue-manifest-digest", required=True
    )
    finalize_contamination_review_parser.set_defaults(handler=_finalize_contamination_review)

    verify_contamination_review_result_parser = subparsers.add_parser(
        "verify-contamination-review-result",
        help="replay finalized human contamination-review evidence",
    )
    verify_contamination_review_result_parser.add_argument("result", type=Path)
    verify_contamination_review_result_parser.add_argument("review_queue", type=Path)
    verify_contamination_review_result_parser.add_argument("contamination_bundle", type=Path)
    verify_contamination_review_result_parser.add_argument("config", type=Path)
    verify_contamination_review_result_parser.add_argument("model_directory", type=Path)
    verify_contamination_review_result_parser.add_argument("triviaqa_source", type=Path)
    verify_contamination_review_result_parser.add_argument(
        "--target", type=Path, action="append", required=True
    )
    verify_contamination_review_result_parser.add_argument(
        "--expected-contamination-manifest-digest", required=True
    )
    verify_contamination_review_result_parser.add_argument(
        "--expected-review-queue-manifest-digest", required=True
    )
    verify_contamination_review_result_parser.add_argument(
        "--expected-result-manifest-digest", required=True
    )
    verify_contamination_review_result_parser.set_defaults(
        handler=_verify_contamination_review_result
    )

    reviewed_split_parser = subparsers.add_parser(
        "prepare-reviewed-splits",
        help="publish TriviaQA splits bound to completed human contamination review",
    )
    reviewed_split_parser.add_argument("output", type=Path)
    _add_reviewed_split_evidence_arguments(reviewed_split_parser)
    reviewed_split_parser.set_defaults(handler=_prepare_reviewed_splits)

    verify_reviewed_split_parser = subparsers.add_parser(
        "verify-reviewed-splits",
        help="replay reviewed TriviaQA splits from finalized contamination evidence",
    )
    verify_reviewed_split_parser.add_argument("splits", type=Path)
    verify_reviewed_split_parser.add_argument("--expected-split-manifest-digest", required=True)
    _add_reviewed_split_evidence_arguments(verify_reviewed_split_parser)
    verify_reviewed_split_parser.set_defaults(handler=_verify_reviewed_splits)

    runtime_questions_parser = subparsers.add_parser(
        "prepare-runtime-validation",
        help="select and freeze the shared provisional E0 factual prompts",
    )
    runtime_questions_parser.add_argument("reserved_source", type=Path)
    runtime_questions_parser.add_argument("output", type=Path)
    runtime_questions_parser.add_argument("--parent-split-manifest-digest", required=True)
    runtime_questions_parser.add_argument("--contamination-manifest-digest", required=True)
    runtime_questions_parser.add_argument("--seed", type=int, default=17)
    runtime_questions_parser.set_defaults(handler=_prepare_runtime_validation)

    verify_runtime_questions_parser = subparsers.add_parser(
        "verify-runtime-validation",
        help="replay the shared provisional E0 question selection",
    )
    verify_runtime_questions_parser.add_argument("bundle", type=Path)
    verify_runtime_questions_parser.add_argument("reserved_source", type=Path)
    verify_runtime_questions_parser.add_argument("--expected-manifest-digest", required=True)
    verify_runtime_questions_parser.add_argument("--parent-split-manifest-digest", required=True)
    verify_runtime_questions_parser.add_argument("--contamination-manifest-digest", required=True)
    verify_runtime_questions_parser.set_defaults(handler=_verify_runtime_validation)

    metrics_parser = subparsers.add_parser("metrics", help="summarize canonical generation records")
    metrics_parser.add_argument("records", type=Path)
    metrics_parser.add_argument("--partial-credit", type=float, default=0.5)
    metrics_parser.set_defaults(handler=_metrics)

    manifest_parser = subparsers.add_parser(
        "verify-manifest", help="verify a frozen manifest digest"
    )
    manifest_parser.add_argument("path", type=Path)
    manifest_parser.set_defaults(handler=lambda args: _verify_manifest(args.path))

    study_parser = subparsers.add_parser(
        "validate-study", help="validate the bound E0-E10 and analysis protocols"
    )
    study_parser.add_argument("study_protocol", type=Path)
    study_parser.add_argument("analysis_protocol", type=Path)
    study_parser.add_argument("research_plan", type=Path)
    study_parser.set_defaults(handler=_validate_study)

    grader_parser = subparsers.add_parser(
        "verify-grader", help="verify a frozen grader config and source artifact"
    )
    grader_parser.add_argument("config", type=Path)
    grader_parser.add_argument("source_artifact", type=Path)
    grader_parser.set_defaults(handler=_verify_grader)

    freeze_e1_graders_parser = subparsers.add_parser(
        "freeze-e1-graders",
        help="freeze the exact E1 rubrics, model routes, source artifacts, and adapter",
    )
    freeze_e1_graders_parser.add_argument("output", type=Path)
    freeze_e1_graders_parser.set_defaults(handler=_freeze_e1_graders)

    verify_e1_graders_parser = subparsers.add_parser(
        "verify-e1-graders",
        help="replay a frozen E1 grader bundle against all live pinned inputs",
    )
    verify_e1_graders_parser.add_argument("bundle", type=Path)
    verify_e1_graders_parser.add_argument("--expected-manifest-digest", required=True)
    verify_e1_graders_parser.set_defaults(handler=_verify_e1_graders)

    prepare_e1_parser = subparsers.add_parser(
        "prepare-e1-vllm",
        help="freeze the reviewed E1 matrix and create its resumable ledger",
    )
    _add_e1_common_arguments(prepare_e1_parser)
    _add_reviewed_split_evidence_arguments(prepare_e1_parser)
    prepare_e1_parser.set_defaults(handler=_prepare_e1_vllm)

    run_e1_parser = subparsers.add_parser(
        "run-e1-vllm",
        help="run or explicitly resume the exact 19,800 native-VLLM generations",
    )
    _add_e1_common_arguments(run_e1_parser)
    _add_e1_execution_arguments(run_e1_parser)
    run_e1_parser.set_defaults(handler=_run_e1_vllm)

    grade_e1_parser = subparsers.add_parser(
        "grade-e1-openrouter",
        help="grade or explicitly resume the 4,800 frozen external rubric calls",
    )
    _add_e1_common_arguments(grade_e1_parser)
    _add_e1_execution_arguments(grade_e1_parser)
    grade_e1_parser.add_argument("--env-file", type=Path, default=Path(".env"))
    grade_e1_parser.set_defaults(handler=_grade_e1_openrouter)

    finalize_e1_parser = subparsers.add_parser(
        "finalize-e1",
        help="freeze E1 ledger records, reporting gates, labels, and prompt metrics",
    )
    _add_e1_common_arguments(finalize_e1_parser)
    finalize_e1_parser.add_argument("output", type=Path)
    finalize_e1_parser.add_argument("--checkpoint-batch-size", type=int, default=250)
    finalize_e1_parser.set_defaults(handler=_finalize_e1)

    verify_e1_output_parser = subparsers.add_parser(
        "verify-e1-outputs",
        help="recompute frozen E1 outcome labels and prompt metrics from its ledger",
    )
    verify_e1_output_parser.add_argument("output", type=Path)
    verify_e1_output_parser.add_argument("work", type=Path)
    verify_e1_output_parser.add_argument("ledger", type=Path)
    verify_e1_output_parser.add_argument("--expected-manifest-digest", required=True)
    verify_e1_output_parser.add_argument(
        "--study-config", type=Path, default=Path("configs/experiments/phases.yaml")
    )
    verify_e1_output_parser.set_defaults(handler=_verify_e1_outputs)

    prepare_aa_parser = subparsers.add_parser(
        "prepare-aa-official",
        help="freeze the auxiliary AA Public-600 official-prompt schedule",
    )
    _add_e1_common_arguments(prepare_aa_parser)
    prepare_aa_parser.set_defaults(handler=_prepare_aa_official)

    run_aa_parser = subparsers.add_parser(
        "run-aa-official",
        help="generate and officially grade the resumable AA Public-600 track",
    )
    _add_e1_common_arguments(run_aa_parser)
    _add_e1_execution_arguments(run_aa_parser)
    run_aa_parser.add_argument("--env-file", type=Path, default=Path(".env"))
    run_aa_parser.set_defaults(handler=_run_aa_official)

    finalize_aa_parser = subparsers.add_parser(
        "finalize-aa-official",
        help="freeze AA official metrics and its paired neutral-M0 comparison",
    )
    _add_e1_common_arguments(finalize_aa_parser)
    finalize_aa_parser.add_argument("output", type=Path)
    finalize_aa_parser.set_defaults(handler=_finalize_aa_official)

    verify_aa_parser = subparsers.add_parser(
        "verify-aa-official",
        help="replay the frozen AA official track and paired neutral comparison",
    )
    _add_e1_common_arguments(verify_aa_parser)
    verify_aa_parser.add_argument("output", type=Path)
    verify_aa_parser.add_argument("--expected-manifest-digest", required=True)
    verify_aa_parser.set_defaults(handler=_verify_aa_official)

    prepare_e2_parser = subparsers.add_parser(
        "prepare-e2-vllm",
        help="freeze the exact E2 capture schedule and resumable activation workspace",
    )
    _add_e2_common_arguments(prepare_e2_parser)
    prepare_e2_parser.add_argument("--shard-rows", type=int, default=64)
    prepare_e2_parser.set_defaults(handler=_prepare_e2_vllm)

    run_e2_parser = subparsers.add_parser(
        "run-e2-vllm",
        help="capture or resume the frozen 21,600-row native-VLLM E2 schedule",
    )
    _add_e2_common_arguments(run_e2_parser)
    run_e2_parser.add_argument("--request-budget", type=int)
    run_e2_parser.set_defaults(handler=_run_e2_vllm)

    verify_e2_capture_parser = subparsers.add_parser(
        "verify-e2-capture",
        help="replay the E2 activation, resolution, and runtime-session chains",
    )
    _add_e2_common_arguments(verify_e2_capture_parser)
    verify_e2_capture_parser.add_argument("--require-complete", action="store_true")
    verify_e2_capture_parser.set_defaults(handler=_verify_e2_capture)

    fit_e2_parser = subparsers.add_parser(
        "fit-e2-probes",
        help="screen and freeze all E2 probe tasks after verified complete capture",
    )
    _add_e2_common_arguments(fit_e2_parser)
    fit_e2_parser.add_argument("output", type=Path)
    fit_e2_parser.add_argument("--probe-work-directory", type=Path)
    fit_e2_parser.set_defaults(handler=_fit_e2_probes)

    verify_e2_probe_parser = subparsers.add_parser(
        "verify-e2-probes",
        help="replay every E2 screening/final metric and the confidence-baseline gate",
    )
    verify_e2_probe_parser.add_argument("bundle", type=Path)
    verify_e2_probe_parser.add_argument("workspace", type=Path)
    verify_e2_probe_parser.set_defaults(handler=_verify_e2_probes)

    finalize_e2_parser = subparsers.add_parser(
        "finalize-e2",
        help="publish the verified E2 ledger as completed or scientifically falsified",
    )
    _add_e2_common_arguments(finalize_e2_parser)
    finalize_e2_parser.add_argument("probe_bundle", type=Path)
    finalize_e2_parser.add_argument("output", type=Path)
    finalize_e2_parser.add_argument("--expected-workspace-plan-identity", required=True)
    finalize_e2_parser.add_argument("--expected-capture-plan-identity", required=True)
    finalize_e2_parser.add_argument("--expected-probe-manifest-digest", required=True)
    finalize_e2_parser.set_defaults(handler=_finalize_e2)

    e3_runbook_parser = subparsers.add_parser(
        "write-e3-runbook",
        help="write the secret-free full E3 operator runbook template",
    )
    e3_runbook_parser.add_argument("output", type=Path)
    e3_runbook_parser.add_argument("reviewed_splits", type=Path)
    e3_runbook_parser.set_defaults(handler=_write_e3_operator_runbook)

    e3_preflight_parser = subparsers.add_parser(
        "preflight-e3",
        help="replay all read-only E3 inputs, counts, and prerequisite completions",
    )
    e3_preflight_parser.add_argument("runbook", type=Path)
    e3_preflight_parser.set_defaults(handler=_preflight_e3_operator)

    e3_advance_parser = subparsers.add_parser(
        "advance-e3",
        help=(
            "perform one resumable construction, control, stage, selection, or finalization action"
        ),
    )
    e3_advance_parser.add_argument("runbook", type=Path)
    e3_advance_parser.add_argument("--request-budget", type=int, default=4_096)
    e3_advance_parser.set_defaults(handler=_advance_e3_operator)

    e3_verify_parser = subparsers.add_parser(
        "verify-e3",
        help="replay the complete E3 construction, controls, seven stages, and phase",
    )
    e3_verify_parser.add_argument("runbook", type=Path)
    e3_verify_parser.set_defaults(handler=_verify_e3_operator)

    prepare_m2_parser = subparsers.add_parser(
        "prepare-m2-caa",
        help="freeze CAA incorrect/gold pairs from the verified E3 construction",
    )
    _add_m2_caa_common_arguments(prepare_m2_parser)
    prepare_m2_parser.set_defaults(handler=_prepare_m2_caa)

    run_m2_parser = subparsers.add_parser(
        "run-m2-caa",
        help="capture or resume native-VLLM residual CAA pair activations",
    )
    _add_m2_caa_common_arguments(run_m2_parser)
    run_m2_parser.add_argument("model_config", type=Path)
    run_m2_parser.add_argument("snapshot_directory", type=Path)
    run_m2_parser.add_argument("--request-budget", type=int)
    run_m2_parser.set_defaults(handler=_run_m2_caa)

    verify_m2_work_parser = subparsers.add_parser(
        "verify-m2-caa-work",
        help="replay the M2 pair, checkpoint, and runtime-session chains",
    )
    _add_m2_caa_common_arguments(verify_m2_work_parser)
    verify_m2_work_parser.add_argument("--require-complete", action="store_true")
    verify_m2_work_parser.set_defaults(handler=_verify_m2_caa_work)

    finalize_m2_parser = subparsers.add_parser(
        "finalize-m2-caa",
        help="publish a portable residual CAA vector bank after complete capture",
    )
    _add_m2_caa_common_arguments(finalize_m2_parser)
    finalize_m2_parser.add_argument("output", type=Path)
    finalize_m2_parser.set_defaults(handler=_finalize_m2_caa)

    verify_m2_artifact_parser = subparsers.add_parser(
        "verify-m2-caa-artifact",
        help="replay a portable M2 CAA artifact without loading the model",
    )
    verify_m2_artifact_parser.add_argument("artifact", type=Path)
    verify_m2_artifact_parser.add_argument("--expected-manifest-digest", required=True)
    verify_m2_artifact_parser.set_defaults(handler=_verify_m2_caa_artifact)

    e4_act_parser = subparsers.add_parser(
        "build-e4-act-baseline",
        help="freeze the calibrated E2-risk-gated M2 adaptive comparison",
    )
    e4_act_parser.add_argument("e2_probe_bundle", type=Path)
    e4_act_parser.add_argument("e2_workspace", type=Path)
    e4_act_parser.add_argument("e2_phase_run", type=Path)
    e4_act_parser.add_argument("m2_caa_artifact", type=Path)
    e4_act_parser.add_argument("output", type=Path)
    e4_act_parser.add_argument("--intervention-layer", type=int, required=True)
    e4_act_parser.add_argument(
        "--study-config", type=Path, default=Path("configs/experiments/phases.yaml")
    )
    e4_act_parser.set_defaults(handler=_build_e4_act_baseline)

    prepare_e4_parser = subparsers.add_parser(
        "prepare-e4-vllm-screen",
        help="freeze E4 capabilities, policies, screen, and an empty phase ledger",
    )
    prepare_e4_parser.add_argument("dev_questions", type=Path)
    prepare_e4_parser.add_argument("model_config", type=Path)
    prepare_e4_parser.add_argument("snapshot_directory", type=Path)
    prepare_e4_parser.add_argument("snapshot_manifest", type=Path)
    prepare_e4_parser.add_argument("runtime_receipt", type=Path)
    prepare_e4_parser.add_argument("e2_probe_bundle", type=Path)
    prepare_e4_parser.add_argument("e3_static_vectors", type=Path)
    prepare_e4_parser.add_argument("m2_caa_artifact", type=Path)
    prepare_e4_parser.add_argument("act_baseline_artifact", type=Path)
    prepare_e4_parser.add_argument("e3_phase_run", type=Path)
    prepare_e4_parser.add_argument("setup", type=Path)
    prepare_e4_parser.add_argument("ledger", type=Path)
    prepare_e4_parser.add_argument("--execution-key-file", type=Path, required=True)
    prepare_e4_parser.add_argument("--m1-layer", type=int, required=True)
    prepare_e4_parser.add_argument("--m2-layer", type=int, required=True)
    prepare_e4_parser.add_argument("--standardized-alpha", type=float, default=1.0)
    prepare_e4_parser.add_argument("--iti-implementation", type=Path)
    prepare_e4_parser.add_argument("--truthx-implementation", type=Path)
    prepare_e4_parser.add_argument("--truthx-autoencoder", type=Path)
    prepare_e4_parser.add_argument(
        "--prompt-config", type=Path, default=Path("configs/prompts/primary.yaml")
    )
    prepare_e4_parser.add_argument(
        "--study-config", type=Path, default=Path("configs/experiments/phases.yaml")
    )
    prepare_e4_parser.set_defaults(handler=_prepare_e4_vllm)

    run_e4_parser = subparsers.add_parser(
        "run-e4-vllm-screen",
        help="run or resume signed native-VLLM M1/M2/ACT E4 rows",
    )
    run_e4_parser.add_argument("setup", type=Path)
    run_e4_parser.add_argument("ledger", type=Path)
    run_e4_parser.add_argument("model_config", type=Path)
    run_e4_parser.add_argument("snapshot_directory", type=Path)
    run_e4_parser.add_argument("--execution-key-file", type=Path, required=True)
    run_e4_parser.add_argument("--request-budget", type=int)
    run_e4_parser.add_argument("--checkpoint-rows", type=int, default=8)
    run_e4_parser.add_argument(
        "--prompt-config", type=Path, default=Path("configs/prompts/primary.yaml")
    )
    run_e4_parser.add_argument(
        "--study-config", type=Path, default=Path("configs/experiments/phases.yaml")
    )
    run_e4_parser.set_defaults(handler=_run_e4_vllm)

    verify_e4_parser = subparsers.add_parser(
        "verify-e4-vllm-screen",
        help="replay E4 setup, records, signatures, grading, and progress",
    )
    verify_e4_parser.add_argument("setup", type=Path)
    verify_e4_parser.add_argument("ledger", type=Path)
    verify_e4_parser.add_argument("--require-complete", action="store_true")
    verify_e4_parser.add_argument(
        "--study-config", type=Path, default=Path("configs/experiments/phases.yaml")
    )
    verify_e4_parser.set_defaults(handler=_verify_e4_vllm)

    finalize_e4_parser = subparsers.add_parser(
        "finalize-e4-vllm-screen",
        help="derive promotion, evaluate its gate, and freeze the E4 ledger",
    )
    finalize_e4_parser.add_argument("setup", type=Path)
    finalize_e4_parser.add_argument("ledger", type=Path)
    finalize_e4_parser.add_argument("promotion", type=Path)
    finalize_e4_parser.add_argument("gate_evidence", type=Path)
    finalize_e4_parser.add_argument(
        "--study-config", type=Path, default=Path("configs/experiments/phases.yaml")
    )
    finalize_e4_parser.set_defaults(handler=_finalize_e4_vllm)

    materialize_e5_splits_parser = subparsers.add_parser(
        "materialize-e5-controller-splits",
        help="materialize the exact E2 4,000/1,000 controller subdivision for E5",
    )
    materialize_e5_splits_parser.add_argument("source_questions", type=Path)
    materialize_e5_splits_parser.add_argument("output", type=Path)
    materialize_e5_splits_parser.add_argument(
        "--expected-reviewed-split-manifest-digest", required=True
    )
    materialize_e5_splits_parser.set_defaults(handler=_materialize_e5_controller_splits)

    verify_e5_splits_parser = subparsers.add_parser(
        "verify-e5-controller-splits",
        help="replay E5 controller rows and semantic-group-disjoint membership",
    )
    verify_e5_splits_parser.add_argument("source_questions", type=Path)
    verify_e5_splits_parser.add_argument("splits", type=Path)
    verify_e5_splits_parser.add_argument("--expected-manifest-digest", required=True)
    verify_e5_splits_parser.add_argument(
        "--expected-reviewed-split-manifest-digest", required=True
    )
    verify_e5_splits_parser.set_defaults(handler=_verify_e5_controller_splits)

    prepare_e5_capture_parser = subparsers.add_parser(
        "prepare-e5-fit-capture",
        help="freeze signed T-steer prompt/response capture for the E5 controller grid",
    )
    prepare_e5_capture_parser.add_argument("work", type=Path)
    prepare_e5_capture_parser.add_argument("e3_construction", type=Path)
    prepare_e5_capture_parser.add_argument("questions", type=Path)
    prepare_e5_capture_parser.add_argument("e2_probe_bundle", type=Path)
    prepare_e5_capture_parser.add_argument("e2_workspace", type=Path)
    prepare_e5_capture_parser.add_argument("e3_static_vectors", type=Path)
    prepare_e5_capture_parser.add_argument("runtime_artifact", type=Path)
    prepare_e5_capture_parser.add_argument("--execution-key-file", type=Path, required=True)
    prepare_e5_capture_parser.add_argument("--fixed-best-layer", type=int, required=True)
    prepare_e5_capture_parser.add_argument(
        "--two-layer-candidates", type=int, nargs=2, required=True
    )
    prepare_e5_capture_parser.add_argument(
        "--three-layer-candidates", type=int, nargs=3, required=True
    )
    prepare_e5_capture_parser.add_argument(
        "--intervention-site",
        choices=("post_attention", "post_mlp", "block_output"),
        required=True,
    )
    prepare_e5_capture_parser.add_argument("--shard-rows", type=int, default=64)
    prepare_e5_capture_parser.add_argument(
        "--max-peak-memory-bytes", type=int, default=40 * 1024**3
    )
    prepare_e5_capture_parser.add_argument(
        "--prompt-config", type=Path, default=Path("configs/prompts/primary.yaml")
    )
    prepare_e5_capture_parser.set_defaults(handler=_prepare_e5_fit_capture)

    run_e5_capture_parser = subparsers.add_parser(
        "run-e5-fit-capture",
        help="capture or resume native-VLLM E5 T-steer prompt/response pairs",
    )
    run_e5_capture_parser.add_argument("work", type=Path)
    run_e5_capture_parser.add_argument("e3_construction", type=Path)
    run_e5_capture_parser.add_argument("questions", type=Path)
    run_e5_capture_parser.add_argument("model_config", type=Path)
    run_e5_capture_parser.add_argument("snapshot_directory", type=Path)
    run_e5_capture_parser.add_argument("--execution-key-file", type=Path, required=True)
    run_e5_capture_parser.add_argument("--request-budget", type=int)
    run_e5_capture_parser.add_argument(
        "--prompt-config", type=Path, default=Path("configs/prompts/primary.yaml")
    )
    run_e5_capture_parser.set_defaults(handler=_run_e5_fit_capture)

    verify_e5_capture_parser = subparsers.add_parser(
        "verify-e5-fit-capture",
        help="replay E5 source rows, signed shards, tensors, and memory receipts",
    )
    verify_e5_capture_parser.add_argument("work", type=Path)
    verify_e5_capture_parser.add_argument("e3_construction", type=Path)
    verify_e5_capture_parser.add_argument("questions", type=Path)
    verify_e5_capture_parser.add_argument("--execution-key-file", type=Path, required=True)
    verify_e5_capture_parser.add_argument("--require-complete", action="store_true")
    verify_e5_capture_parser.add_argument(
        "--prompt-config", type=Path, default=Path("configs/prompts/primary.yaml")
    )
    verify_e5_capture_parser.set_defaults(handler=_verify_e5_fit_capture)

    prepare_e5_labels_parser = subparsers.add_parser(
        "prepare-e5-layer-labels",
        help="freeze signed T-controller counterfactual labels for E5 layer routers",
    )
    prepare_e5_labels_parser.add_argument("work", type=Path)
    prepare_e5_labels_parser.add_argument("fit_capture", type=Path)
    prepare_e5_labels_parser.add_argument("e3_construction", type=Path)
    prepare_e5_labels_parser.add_argument("t_steer_questions", type=Path)
    prepare_e5_labels_parser.add_argument("controller_questions", type=Path)
    prepare_e5_labels_parser.add_argument("e2_probe_bundle", type=Path)
    prepare_e5_labels_parser.add_argument("e2_workspace", type=Path)
    prepare_e5_labels_parser.add_argument("e3_static_vectors", type=Path)
    prepare_e5_labels_parser.add_argument("--execution-key-file", type=Path, required=True)
    prepare_e5_labels_parser.add_argument("--shard-rows", type=int, default=64)
    prepare_e5_labels_parser.add_argument("--max-new-tokens", type=int, default=48)
    prepare_e5_labels_parser.add_argument("--max-peak-memory-bytes", type=int, default=40 * 1024**3)
    prepare_e5_labels_parser.add_argument(
        "--prompt-config", type=Path, default=Path("configs/prompts/primary.yaml")
    )
    prepare_e5_labels_parser.set_defaults(handler=_prepare_e5_layer_labels)

    run_e5_labels_parser = subparsers.add_parser(
        "run-e5-layer-labels",
        help="run or resume native-VLLM E5 counterfactual layer labels",
    )
    run_e5_labels_parser.add_argument("work", type=Path)
    run_e5_labels_parser.add_argument("fit_capture", type=Path)
    run_e5_labels_parser.add_argument("e3_construction", type=Path)
    run_e5_labels_parser.add_argument("t_steer_questions", type=Path)
    run_e5_labels_parser.add_argument("controller_questions", type=Path)
    run_e5_labels_parser.add_argument("e2_probe_bundle", type=Path)
    run_e5_labels_parser.add_argument("e2_workspace", type=Path)
    run_e5_labels_parser.add_argument("model_config", type=Path)
    run_e5_labels_parser.add_argument("snapshot_directory", type=Path)
    run_e5_labels_parser.add_argument("--execution-key-file", type=Path, required=True)
    run_e5_labels_parser.add_argument("--request-budget", type=int)
    run_e5_labels_parser.add_argument(
        "--prompt-config", type=Path, default=Path("configs/prompts/primary.yaml")
    )
    run_e5_labels_parser.set_defaults(handler=_run_e5_layer_labels)

    verify_e5_labels_parser = subparsers.add_parser(
        "verify-e5-layer-labels",
        help="verify E5 counterfactual labels, signatures, and exact source lineage",
    )
    verify_e5_labels_parser.add_argument("work", type=Path)
    verify_e5_labels_parser.add_argument("fit_capture", type=Path)
    verify_e5_labels_parser.add_argument("e3_construction", type=Path)
    verify_e5_labels_parser.add_argument("t_steer_questions", type=Path)
    verify_e5_labels_parser.add_argument("controller_questions", type=Path)
    verify_e5_labels_parser.add_argument("e2_probe_bundle", type=Path)
    verify_e5_labels_parser.add_argument("e2_workspace", type=Path)
    verify_e5_labels_parser.add_argument("--execution-key-file", type=Path, required=True)
    verify_e5_labels_parser.add_argument("--require-complete", action="store_true")
    verify_e5_labels_parser.add_argument(
        "--prompt-config", type=Path, default=Path("configs/prompts/primary.yaml")
    )
    verify_e5_labels_parser.set_defaults(handler=_verify_e5_layer_labels)

    fit_e5_grid_parser = subparsers.add_parser(
        "fit-e5-controller-grid",
        help="fit and atomically package the complete verified E5 controller grid",
    )
    fit_e5_grid_parser.add_argument("output", type=Path)
    fit_e5_grid_parser.add_argument("fit_capture", type=Path)
    fit_e5_grid_parser.add_argument("layer_labels", type=Path)
    fit_e5_grid_parser.add_argument("e3_construction", type=Path)
    fit_e5_grid_parser.add_argument("t_steer_questions", type=Path)
    fit_e5_grid_parser.add_argument("controller_questions", type=Path)
    fit_e5_grid_parser.add_argument("e2_probe_bundle", type=Path)
    fit_e5_grid_parser.add_argument("e2_workspace", type=Path)
    fit_e5_grid_parser.add_argument("--execution-key-file", type=Path, required=True)
    fit_e5_grid_parser.add_argument(
        "--prompt-config", type=Path, default=Path("configs/prompts/primary.yaml")
    )
    fit_e5_grid_parser.set_defaults(handler=_fit_e5_controller_grid)

    verify_e5_grid_parser = subparsers.add_parser(
        "verify-e5-controller-grid",
        help="reload every E5 controller and replay its fit lineage",
    )
    verify_e5_grid_parser.add_argument("grid", type=Path)
    verify_e5_grid_parser.set_defaults(handler=_verify_e5_controller_grid)

    package_e5_bindings_parser = subparsers.add_parser(
        "package-e5-controller-bindings",
        help="atomically bind every E5 ablation arm to its exact fitted controller",
    )
    package_e5_bindings_parser.add_argument("output", type=Path)
    package_e5_bindings_parser.add_argument("grid", type=Path)
    package_e5_bindings_parser.add_argument("--execution-key-file", type=Path, required=True)
    package_e5_bindings_parser.set_defaults(handler=_package_e5_controller_bindings)

    verify_e5_bindings_parser = subparsers.add_parser(
        "verify-e5-controller-bindings",
        help="replay every E5 arm binding against the complete fitted grid",
    )
    verify_e5_bindings_parser.add_argument("bindings", type=Path)
    verify_e5_bindings_parser.set_defaults(handler=_verify_e5_controller_bindings)

    estimate_e5_native_parser = subparsers.add_parser(
        "estimate-e5-native-ablation",
        help="estimate the exact 9.73M-row E5 grid from a measured VLLM rate",
    )
    estimate_e5_native_parser.add_argument("--generations-per-second", type=float, required=True)
    estimate_e5_native_parser.add_argument(
        "--checkpoint-opens-per-second", type=float, required=True
    )
    estimate_e5_native_parser.add_argument(
        "--verification-rows-per-second", type=float, required=True
    )
    estimate_e5_native_parser.add_argument("--request-budget", type=int, default=1_024)
    estimate_e5_native_parser.set_defaults(handler=_estimate_e5_native_ablation)

    prepare_e5_native_parser = subparsers.add_parser(
        "prepare-e5-native-ablation",
        help="freeze the implicit resumable E5 native-VLLM ablation schedule",
    )
    prepare_e5_native_parser.add_argument("work", type=Path)
    prepare_e5_native_parser.add_argument("screen_receipt", type=Path)
    prepare_e5_native_parser.add_argument("controller_bindings", type=Path)
    prepare_e5_native_parser.add_argument("m1_policy", type=Path)
    prepare_e5_native_parser.add_argument("e3_static_vectors", type=Path)
    prepare_e5_native_parser.add_argument("fit_capture", type=Path)
    prepare_e5_native_parser.add_argument("runtime_artifact", type=Path)
    prepare_e5_native_parser.add_argument("--execution-key-file", type=Path, required=True)
    prepare_e5_native_parser.add_argument(
        "--acknowledge-exact-grid-records",
        type=int,
        required=True,
        help="must equal 9730000 after reviewing the measured runtime estimate",
    )
    prepare_e5_native_parser.add_argument("--shard-rows", type=int, default=1_024)
    prepare_e5_native_parser.add_argument("--max-new-tokens", type=int, default=48)
    prepare_e5_native_parser.add_argument("--max-peak-memory-bytes", type=int, default=40 * 1024**3)
    prepare_e5_native_parser.add_argument(
        "--prompt-config", type=Path, default=Path("configs/prompts/primary.yaml")
    )
    prepare_e5_native_parser.set_defaults(handler=_prepare_e5_native_ablation)

    run_e5_native_parser = subparsers.add_parser(
        "run-e5-native-ablation",
        help="run or resume signed native-VLLM E5 ablation generations",
    )
    run_e5_native_parser.add_argument("work", type=Path)
    run_e5_native_parser.add_argument("model_config", type=Path)
    run_e5_native_parser.add_argument("snapshot_directory", type=Path)
    run_e5_native_parser.add_argument("--execution-key-file", type=Path, required=True)
    run_e5_native_parser.add_argument("--request-budget", type=int)
    run_e5_native_parser.set_defaults(handler=_run_e5_native_ablation)

    verify_e5_native_parser = subparsers.add_parser(
        "verify-e5-native-ablation",
        help="replay E5 sources, signed semantic transcripts, grading, and hook evidence",
    )
    verify_e5_native_parser.add_argument("work", type=Path)
    verify_e5_native_parser.add_argument("--execution-key-file", type=Path, required=True)
    verify_e5_native_parser.add_argument("--require-complete", action="store_true")
    verify_e5_native_parser.add_argument(
        "--structural-only",
        action="store_true",
        help="verify signed bytes and lineage without controller-transcript replay",
    )
    verify_e5_native_parser.set_defaults(handler=_verify_e5_native_ablation)

    finalize_e5_native_parser = subparsers.add_parser(
        "finalize-e5-native-ablation",
        help="atomically materialize exact E5 selection records and a signed receipt",
    )
    finalize_e5_native_parser.add_argument("work", type=Path)
    finalize_e5_native_parser.add_argument("--execution-key-file", type=Path, required=True)
    finalize_e5_native_parser.set_defaults(handler=_finalize_e5_native_ablation)

    derive_e5_selection_parser = subparsers.add_parser(
        "derive-e5-selection",
        help="replay the full signed native grid and freeze the matched E5 selection",
    )
    derive_e5_selection_parser.add_argument("output", type=Path)
    derive_e5_selection_parser.add_argument("native", type=Path)
    derive_e5_selection_parser.add_argument("controller_bindings", type=Path)
    derive_e5_selection_parser.add_argument("e2_probe_bundle", type=Path)
    derive_e5_selection_parser.add_argument("e3_static_vectors", type=Path)
    derive_e5_selection_parser.add_argument("e4_promoted_baselines", type=Path)
    derive_e5_selection_parser.add_argument("--execution-key-file", type=Path, required=True)
    derive_e5_selection_parser.set_defaults(handler=_derive_e5_selection)

    verify_e5_selection_parser = subparsers.add_parser(
        "verify-e5-selection",
        help="verify the signed E5 selection and native finalization chain",
    )
    verify_e5_selection_parser.add_argument("selection", type=Path)
    verify_e5_selection_parser.add_argument("native", type=Path)
    verify_e5_selection_parser.add_argument("--execution-key-file", type=Path, required=True)
    verify_e5_selection_parser.set_defaults(handler=_verify_e5_selection_package)

    prepare_e5_phase_parser = subparsers.add_parser(
        "prepare-e5-phase-ledger",
        help="create the exact four-condition E5 ledger for zero-rerun promotion",
    )
    prepare_e5_phase_parser.add_argument("ledger", type=Path)
    prepare_e5_phase_parser.add_argument("selection", type=Path)
    prepare_e5_phase_parser.add_argument("native", type=Path)
    prepare_e5_phase_parser.add_argument("model_config", type=Path)
    prepare_e5_phase_parser.add_argument("e2_run", type=Path)
    prepare_e5_phase_parser.add_argument("e3_run", type=Path)
    prepare_e5_phase_parser.add_argument("e4_run", type=Path)
    prepare_e5_phase_parser.add_argument("--execution-key-file", type=Path, required=True)
    prepare_e5_phase_parser.add_argument(
        "--prompt-config", type=Path, default=Path("configs/prompts/primary.yaml")
    )
    prepare_e5_phase_parser.add_argument(
        "--study-protocol",
        type=Path,
        default=Path("configs/experiments/phases.yaml"),
    )
    prepare_e5_phase_parser.set_defaults(handler=_prepare_e5_phase_ledger)

    promote_e5_phase_parser = subparsers.add_parser(
        "promote-e5-phase-records",
        help="promote signed M1 and selected-M3 native rows without model inference",
    )
    promote_e5_phase_parser.add_argument("ledger", type=Path)
    promote_e5_phase_parser.add_argument("selection", type=Path)
    promote_e5_phase_parser.add_argument("native", type=Path)
    promote_e5_phase_parser.add_argument("--execution-key-file", type=Path, required=True)
    promote_e5_phase_parser.add_argument("--request-budget", type=int)
    promote_e5_phase_parser.add_argument("--checkpoint-rows", type=int, default=250)
    promote_e5_phase_parser.add_argument(
        "--study-protocol",
        type=Path,
        default=Path("configs/experiments/phases.yaml"),
    )
    promote_e5_phase_parser.set_defaults(handler=_promote_e5_phase_records)

    verify_e5_phase_ledger_parser = subparsers.add_parser(
        "verify-e5-phase-ledger",
        help="replay each promoted E5 ledger row against its native transcript",
    )
    verify_e5_phase_ledger_parser.add_argument("ledger", type=Path)
    verify_e5_phase_ledger_parser.add_argument("selection", type=Path)
    verify_e5_phase_ledger_parser.add_argument("native", type=Path)
    verify_e5_phase_ledger_parser.add_argument("--execution-key-file", type=Path, required=True)
    verify_e5_phase_ledger_parser.add_argument("--require-complete", action="store_true")
    verify_e5_phase_ledger_parser.add_argument(
        "--study-protocol",
        type=Path,
        default=Path("configs/experiments/phases.yaml"),
    )
    verify_e5_phase_ledger_parser.set_defaults(handler=_verify_e5_phase_ledger)

    finalize_e5_phase_parser = subparsers.add_parser(
        "finalize-e5-phase",
        help="verify promotion, evaluate all four gates, and freeze terminal E5",
    )
    finalize_e5_phase_parser.add_argument("output", type=Path)
    finalize_e5_phase_parser.add_argument("ledger", type=Path)
    finalize_e5_phase_parser.add_argument("selection", type=Path)
    finalize_e5_phase_parser.add_argument("native", type=Path)
    finalize_e5_phase_parser.add_argument("--execution-key-file", type=Path, required=True)
    finalize_e5_phase_parser.add_argument(
        "--study-protocol",
        type=Path,
        default=Path("configs/experiments/phases.yaml"),
    )
    finalize_e5_phase_parser.set_defaults(handler=_finalize_e5_phase)

    verify_e5_phase_parser = subparsers.add_parser(
        "verify-e5-phase",
        help="replay promoted records, gates, controller bundle, and terminal E5 receipt",
    )
    verify_e5_phase_parser.add_argument("output", type=Path)
    verify_e5_phase_parser.add_argument("ledger", type=Path)
    verify_e5_phase_parser.add_argument("selection", type=Path)
    verify_e5_phase_parser.add_argument("native", type=Path)
    verify_e5_phase_parser.add_argument("--execution-key-file", type=Path, required=True)
    verify_e5_phase_parser.add_argument(
        "--study-protocol",
        type=Path,
        default=Path("configs/experiments/phases.yaml"),
    )
    verify_e5_phase_parser.set_defaults(handler=_verify_e5_phase)

    progress_parser = subparsers.add_parser(
        "phase-progress", help="verify shards and report resumable phase progress"
    )
    progress_parser.add_argument("path", type=Path)
    progress_parser.add_argument("study_protocol", type=Path)
    progress_parser.set_defaults(
        handler=lambda args: _phase_progress(args.path, args.study_protocol)
    )

    phase_parser = subparsers.add_parser(
        "verify-phase", help="verify a completed immutable phase run"
    )
    phase_parser.add_argument("path", type=Path)
    phase_parser.add_argument("study_protocol", type=Path)
    phase_parser.set_defaults(handler=lambda args: _verify_phase(args.path, args.study_protocol))

    write_analysis_parser = subparsers.add_parser(
        "write-analysis",
        help="derive results, render every registered report, and publish the final bundle",
    )
    write_analysis_parser.add_argument("output", type=Path)
    write_analysis_parser.add_argument("analysis_evidence", type=Path)
    write_analysis_parser.add_argument("analysis_protocol", type=Path)
    write_analysis_parser.add_argument("research_plan", type=Path)
    write_analysis_parser.add_argument("study_protocol", type=Path)
    write_analysis_parser.add_argument("e1_run", type=Path)
    write_analysis_parser.add_argument("e3_run", type=Path)
    write_analysis_parser.add_argument("e6_run", type=Path)
    write_analysis_parser.add_argument("e7_run", type=Path)
    write_analysis_parser.add_argument("e8_run", type=Path)
    write_analysis_parser.add_argument("e9_run", type=Path)
    write_analysis_parser.add_argument("e10_run", type=Path)
    write_analysis_parser.add_argument("robustness_results", type=Path)
    write_analysis_parser.add_argument("audit_queue", type=Path)
    write_analysis_parser.add_argument("audit_results", type=Path)
    write_analysis_parser.add_argument("aa_official", type=Path)
    write_analysis_parser.add_argument("--expected-aa-official-manifest-digest", required=True)
    write_analysis_parser.add_argument("--blinding-key-file", type=Path, required=True)
    write_analysis_parser.set_defaults(handler=_write_analysis)

    analysis_parser = subparsers.add_parser(
        "verify-analysis", help="verify the final analysis bundle against its protocol"
    )
    analysis_parser.add_argument("bundle", type=Path)
    analysis_parser.add_argument("analysis_protocol", type=Path)
    analysis_parser.add_argument("research_plan", type=Path)
    analysis_parser.add_argument("study_protocol", type=Path)
    analysis_parser.add_argument("e1_run", type=Path)
    analysis_parser.add_argument("e3_run", type=Path)
    analysis_parser.add_argument("e6_run", type=Path)
    analysis_parser.add_argument("e7_run", type=Path)
    analysis_parser.add_argument("e8_run", type=Path)
    analysis_parser.add_argument("e9_run", type=Path)
    analysis_parser.add_argument("e10_run", type=Path)
    analysis_parser.add_argument("robustness_results", type=Path)
    analysis_parser.add_argument("audit_queue", type=Path)
    analysis_parser.add_argument("audit_results", type=Path)
    analysis_parser.add_argument("aa_official", type=Path)
    analysis_parser.add_argument("--expected-aa-official-manifest-digest", required=True)
    analysis_parser.add_argument("--blinding-key-file", type=Path, required=True)
    analysis_parser.set_defaults(handler=_verify_analysis)

    freeze_analysis_parser = subparsers.add_parser(
        "freeze-analysis-evidence",
        help="derive and freeze results from every exact completed source artifact",
    )
    freeze_analysis_parser.add_argument("output", type=Path)
    freeze_analysis_parser.add_argument("analysis_protocol", type=Path)
    freeze_analysis_parser.add_argument("research_plan", type=Path)
    freeze_analysis_parser.add_argument("study_protocol", type=Path)
    freeze_analysis_parser.add_argument("e1_run", type=Path)
    freeze_analysis_parser.add_argument("e3_run", type=Path)
    freeze_analysis_parser.add_argument("e6_run", type=Path)
    freeze_analysis_parser.add_argument("e7_run", type=Path)
    freeze_analysis_parser.add_argument("e8_run", type=Path)
    freeze_analysis_parser.add_argument("e9_run", type=Path)
    freeze_analysis_parser.add_argument("e10_run", type=Path)
    freeze_analysis_parser.add_argument("robustness_results", type=Path)
    freeze_analysis_parser.add_argument("audit_queue", type=Path)
    freeze_analysis_parser.add_argument("audit_results", type=Path)
    freeze_analysis_parser.add_argument("aa_official", type=Path)
    freeze_analysis_parser.add_argument("--expected-aa-official-manifest-digest", required=True)
    freeze_analysis_parser.add_argument("--blinding-key-file", type=Path, required=True)
    freeze_analysis_parser.set_defaults(handler=_freeze_analysis_evidence)

    verify_analysis_evidence_parser = subparsers.add_parser(
        "verify-analysis-evidence",
        help="rederive frozen results from every live source artifact",
    )
    verify_analysis_evidence_parser.add_argument("evidence", type=Path)
    verify_analysis_evidence_parser.add_argument("analysis_protocol", type=Path)
    verify_analysis_evidence_parser.add_argument("research_plan", type=Path)
    verify_analysis_evidence_parser.add_argument("study_protocol", type=Path)
    verify_analysis_evidence_parser.add_argument("e1_run", type=Path)
    verify_analysis_evidence_parser.add_argument("e3_run", type=Path)
    verify_analysis_evidence_parser.add_argument("e6_run", type=Path)
    verify_analysis_evidence_parser.add_argument("e7_run", type=Path)
    verify_analysis_evidence_parser.add_argument("e8_run", type=Path)
    verify_analysis_evidence_parser.add_argument("e9_run", type=Path)
    verify_analysis_evidence_parser.add_argument("e10_run", type=Path)
    verify_analysis_evidence_parser.add_argument("robustness_results", type=Path)
    verify_analysis_evidence_parser.add_argument("audit_queue", type=Path)
    verify_analysis_evidence_parser.add_argument("audit_results", type=Path)
    verify_analysis_evidence_parser.add_argument("aa_official", type=Path)
    verify_analysis_evidence_parser.add_argument(
        "--expected-aa-official-manifest-digest", required=True
    )
    verify_analysis_evidence_parser.add_argument("--blinding-key-file", type=Path, required=True)
    verify_analysis_evidence_parser.set_defaults(handler=_verify_analysis_evidence)

    snapshot_parser = subparsers.add_parser(
        "freeze-execution-snapshot",
        help="freeze the exact live code and analysis sources for E9 or E10",
    )
    snapshot_parser.add_argument("output", type=Path)
    snapshot_parser.add_argument("study_protocol", type=Path)
    snapshot_parser.add_argument("phase", choices=("E9", "E10"))
    snapshot_parser.add_argument("--repository-root", type=Path)
    snapshot_parser.set_defaults(handler=_freeze_execution_snapshot)

    safety_parser = subparsers.add_parser(
        "freeze-safety-scorer",
        help="freeze the current deterministic safety scorer and runtime public key",
    )
    safety_parser.add_argument("output", type=Path)
    safety_parser.add_argument("execution_public_key")
    safety_parser.set_defaults(handler=_freeze_safety_scorer)

    ifeval_evaluator_parser = subparsers.add_parser(
        "materialize-ifeval-evaluator",
        help="freeze the pinned Google Research IFEval checker source",
    )
    ifeval_evaluator_parser.add_argument("output", type=Path)
    ifeval_evaluator_parser.set_defaults(handler=_materialize_ifeval_evaluator)

    strongreject_grader_parser = subparsers.add_parser(
        "materialize-strongreject-grader",
        help="freeze the released StrongREJECT rubric and approved grader route",
    )
    strongreject_grader_parser.add_argument("output", type=Path)
    strongreject_grader_parser.set_defaults(handler=_materialize_strongreject_grader)

    grade_strongreject_parser = subparsers.add_parser(
        "grade-strongreject-openrouter",
        help="resume the frozen Gemini StrongREJECT rubric over exact records",
    )
    grade_strongreject_parser.add_argument("records", type=Path)
    grade_strongreject_parser.add_argument("questions", type=Path)
    grade_strongreject_parser.add_argument("grader", type=Path)
    grade_strongreject_parser.add_argument("scorer", type=Path)
    grade_strongreject_parser.add_argument("scorer_private_key_file", type=Path)
    grade_strongreject_parser.add_argument("output", type=Path)
    grade_strongreject_parser.add_argument("--env-file", type=Path, default=Path(".env"))
    grade_strongreject_parser.add_argument("--request-budget", type=int)
    grade_strongreject_parser.add_argument("--resume", action="store_true")
    grade_strongreject_parser.set_defaults(handler=_grade_strongreject_openrouter)

    write_e6_parser = subparsers.add_parser(
        "write-e6-runbook",
        help="write the secret-free native-VLLM E6 operator runbook template",
    )
    write_e6_parser.add_argument("output", type=Path)
    write_e6_parser.add_argument("--m1-layer", type=int, required=True)
    write_e6_parser.add_argument("--official-grader-bundle", type=Path, required=True)
    write_e6_parser.add_argument("--expected-grader-manifest-digest", required=True)
    write_e6_parser.set_defaults(handler=_write_e6_runbook)

    freeze_e6_questions_parser = subparsers.add_parser(
        "freeze-e6-questions",
        help="freeze the reviewed E6 development questions and pinned raw sources",
    )
    freeze_e6_questions_parser.add_argument("output", type=Path)
    freeze_e6_questions_parser.add_argument("reviewed_splits", type=Path)
    freeze_e6_questions_parser.add_argument("--triviaqa-source", required=True, type=Path)
    freeze_e6_questions_parser.add_argument("--simpleqa-source", required=True, type=Path)
    freeze_e6_questions_parser.add_argument("--aa-source", required=True, type=Path)
    freeze_e6_questions_parser.add_argument(
        "--expected-reviewed-split-manifest-digest", required=True
    )
    freeze_e6_questions_parser.add_argument(
        "--study-protocol", type=Path, default=Path("configs/experiments/phases.yaml")
    )
    freeze_e6_questions_parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/models/qwen3.6-27b-nvfp4.yaml"),
    )
    freeze_e6_questions_parser.add_argument(
        "--prompt-config", type=Path, default=Path("configs/prompts/primary.yaml")
    )
    freeze_e6_questions_parser.add_argument("--seed", type=int, default=17)
    freeze_e6_questions_parser.set_defaults(handler=_freeze_e6_questions)

    preflight_e6_parser = subparsers.add_parser(
        "preflight-e6",
        help="replay E6 sources and its exact 59,400-row contract without VLLM",
    )
    preflight_e6_parser.add_argument("runbook", type=Path)
    preflight_e6_parser.set_defaults(handler=_preflight_e6)

    prepare_e6_parser = subparsers.add_parser(
        "prepare-e6",
        help="create or safely reopen the exact E6 ledger and row workspace",
    )
    prepare_e6_parser.add_argument("runbook", type=Path)
    prepare_e6_parser.set_defaults(handler=_prepare_e6)

    attest_e6_parser = subparsers.add_parser(
        "attest-e6-runtime",
        help="load pinned Qwen through VLLM and freeze the E6 host attestation",
    )
    attest_e6_parser.add_argument("runbook", type=Path)
    attest_e6_parser.set_defaults(handler=_attest_e6)

    run_e6_parser = subparsers.add_parser(
        "run-e6",
        help="resume signed generation and teacher-forced likelihood rows through VLLM",
    )
    run_e6_parser.add_argument("runbook", type=Path)
    run_e6_parser.add_argument("--limit", type=int)
    run_e6_parser.set_defaults(handler=_run_e6)

    finalize_e6_parser = subparsers.add_parser(
        "finalize-e6",
        help="freeze likelihoods, derive the registered E6 gate, and finalize",
    )
    finalize_e6_parser.add_argument("runbook", type=Path)
    finalize_e6_parser.set_defaults(handler=_finalize_e6)

    verify_e6_parser = subparsers.add_parser(
        "verify-e6",
        help="report progress or replay the complete E6 operator lifecycle",
    )
    verify_e6_parser.add_argument("runbook", type=Path)
    verify_e6_parser.set_defaults(handler=_verify_e6)

    stage_e7_e8_inputs_parser = subparsers.add_parser(
        "stage-e7-e8-inputs",
        help="atomically stage and verify immutable external E7/E8 inputs",
    )
    stage_e7_e8_inputs_parser.add_argument("output", type=Path)
    stage_e7_e8_inputs_parser.add_argument("reviewed_splits", type=Path)
    stage_e7_e8_inputs_parser.add_argument("reviewed_language_suite", type=Path)
    stage_e7_e8_inputs_parser.add_argument("ifeval_evaluator", type=Path)
    stage_e7_e8_inputs_parser.add_argument("--triviaqa-source", required=True, type=Path)
    stage_e7_e8_inputs_parser.add_argument("--ifeval-source", required=True, type=Path)
    stage_e7_e8_inputs_parser.add_argument("--mmlu-pro-source", required=True, type=Path)
    stage_e7_e8_inputs_parser.add_argument("--wikitext103-source", required=True, type=Path)
    stage_e7_e8_inputs_parser.add_argument("--xstest-source", required=True, type=Path)
    stage_e7_e8_inputs_parser.add_argument("--strongreject-source", required=True, type=Path)
    stage_e7_e8_inputs_parser.add_argument(
        "--expected-reviewed-split-manifest-digest", required=True
    )
    stage_e7_e8_inputs_parser.set_defaults(handler=_stage_e7_e8_inputs)

    verify_e7_e8_inputs_parser = subparsers.add_parser(
        "verify-e7-e8-inputs",
        help="replay every staged external E7/E8 input",
    )
    verify_e7_e8_inputs_parser.add_argument("directory", type=Path)
    verify_e7_e8_inputs_parser.add_argument(
        "--expected-reviewed-split-manifest-digest", required=True
    )
    verify_e7_e8_inputs_parser.set_defaults(handler=_verify_e7_e8_inputs)

    write_e7_parser = subparsers.add_parser(
        "write-e7-runbook",
        help="write the secret-free native-VLLM E7 staged operator runbook",
    )
    write_e7_parser.add_argument("output", type=Path)
    write_e7_parser.add_argument("--m1-layer", type=int, required=True)
    write_e7_parser.set_defaults(handler=_write_e7_runbook)

    preflight_e7_runbook_parser = subparsers.add_parser(
        "preflight-e7",
        help="verify every immutable E7 input without loading VLLM",
    )
    preflight_e7_runbook_parser.add_argument("runbook", type=Path)
    preflight_e7_runbook_parser.set_defaults(handler=_preflight_e7_runbook)

    prepare_e7_runbook_parser = subparsers.add_parser(
        "prepare-e7",
        help="freeze the separate SAE cohorts and development scorer bundle",
    )
    prepare_e7_runbook_parser.add_argument("runbook", type=Path)
    prepare_e7_runbook_parser.set_defaults(handler=_prepare_e7_runbook)

    capture_e7_parser = subparsers.add_parser(
        "capture-e7",
        help="resume one signed E7 activation-capture partition through VLLM",
    )
    capture_e7_parser.add_argument("runbook", type=Path)
    capture_e7_parser.add_argument("partition", choices=("T-steer", "sae-train", "sae-validation"))
    capture_e7_parser.add_argument("--limit", type=int)
    capture_e7_parser.set_defaults(handler=_capture_e7)

    screen_e7_coordinate_parser = subparsers.add_parser(
        "screen-e7-coordinate",
        help="resume the registered 4-by-5 coordinate sparsity screen",
    )
    screen_e7_coordinate_parser.add_argument("runbook", type=Path)
    screen_e7_coordinate_parser.add_argument("--limit", type=int)
    screen_e7_coordinate_parser.set_defaults(handler=_screen_e7_coordinate)

    fit_e7_sae_parser = subparsers.add_parser(
        "fit-e7-sae",
        help="resume the six-checkpoint E7 SAE grid and freeze its winner",
    )
    fit_e7_sae_parser.add_argument("runbook", type=Path)
    fit_e7_sae_parser.set_defaults(handler=_fit_e7_sae)

    causal_e7_parser = subparsers.add_parser(
        "audit-e7-causal",
        help="resume signed activation/suppression evidence for selected features",
    )
    causal_e7_parser.add_argument("runbook", type=Path)
    causal_e7_parser.add_argument("--limit", type=int)
    causal_e7_parser.set_defaults(handler=_audit_e7_causal)

    interpretability_e7_parser = subparsers.add_parser(
        "audit-e7-interpretability",
        help="resume E7 prompt-transfer and negative-control evidence",
    )
    interpretability_e7_parser.add_argument("runbook", type=Path)
    interpretability_e7_parser.add_argument("--limit", type=int)
    interpretability_e7_parser.set_defaults(handler=_audit_e7_interpretability)

    promote_e7_parser = subparsers.add_parser(
        "promote-e7-sae",
        help="apply the registered reconstruction, stability, causal, and audit gates",
    )
    promote_e7_parser.add_argument("runbook", type=Path)
    promote_e7_parser.set_defaults(handler=_promote_e7_sae)

    prepare_e7_ledger_parser = subparsers.add_parser(
        "prepare-e7-ledger",
        help="freeze the promoted E7 method matrix and create its exact ledger",
    )
    prepare_e7_ledger_parser.add_argument("runbook", type=Path)
    prepare_e7_ledger_parser.set_defaults(handler=_prepare_e7_ledger)

    run_e7_parser = subparsers.add_parser(
        "run-e7",
        help="resume the exact signed 39,624-row E7 development matrix",
    )
    run_e7_parser.add_argument("runbook", type=Path)
    run_e7_parser.add_argument("--limit", type=int)
    run_e7_parser.set_defaults(handler=_run_e7)

    finalize_e7_runbook_parser = subparsers.add_parser(
        "finalize-e7-runbook",
        help="derive E7 gates and publish its self-contained terminal artifact",
    )
    finalize_e7_runbook_parser.add_argument("runbook", type=Path)
    finalize_e7_runbook_parser.set_defaults(handler=_finalize_e7_runbook)

    verify_e7_runbook_parser = subparsers.add_parser(
        "verify-e7-runbook",
        help="replay E7 stage progress and its terminal package without VLLM",
    )
    verify_e7_runbook_parser.add_argument("runbook", type=Path)
    verify_e7_runbook_parser.set_defaults(handler=_verify_e7_runbook)

    write_e8_parser = subparsers.add_parser(
        "write-e8-runbook",
        help="write the secret-free native-VLLM E8 staged operator runbook",
    )
    write_e8_parser.add_argument("output", type=Path)
    write_e8_parser.add_argument("--m1-layer", type=int, required=True)
    write_e8_parser.set_defaults(handler=_write_e8_runbook)

    preflight_e8_runbook_parser = subparsers.add_parser(
        "preflight-e8",
        help="verify every immutable E8 input without loading VLLM",
    )
    preflight_e8_runbook_parser.add_argument("runbook", type=Path)
    preflight_e8_runbook_parser.set_defaults(handler=_preflight_e8_runbook)

    prepare_e8_runbook_parser = subparsers.add_parser(
        "prepare-e8",
        help="freeze the E8 development question schedule and scorer bundle",
    )
    prepare_e8_runbook_parser.add_argument("runbook", type=Path)
    prepare_e8_runbook_parser.set_defaults(handler=_prepare_e8_runbook)

    capture_e8_parser = subparsers.add_parser(
        "capture-e8-activations",
        help="resume signed protected-behavior activation capture through VLLM",
    )
    capture_e8_parser.add_argument("runbook", type=Path)
    capture_e8_parser.add_argument("--limit", type=int)
    capture_e8_parser.set_defaults(handler=_capture_e8_activations)

    screen_e8_variants_parser = subparsers.add_parser(
        "screen-e8-variants",
        help="resume the paired orthogonal and covariance-aware M5 screen",
    )
    screen_e8_variants_parser.add_argument("runbook", type=Path)
    screen_e8_variants_parser.add_argument("--limit", type=int)
    screen_e8_variants_parser.set_defaults(handler=_screen_e8_variants)

    promote_e8_parser = subparsers.add_parser(
        "promote-e8-protected",
        help="freeze the M5 protected direction selected by its paired screen",
    )
    promote_e8_parser.add_argument("runbook", type=Path)
    promote_e8_parser.set_defaults(handler=_promote_e8_protected)

    screen_e8_candidates_parser = subparsers.add_parser(
        "screen-e8-candidates",
        help="resume the 40-condition empirical matched-point strength grid",
    )
    screen_e8_candidates_parser.add_argument("runbook", type=Path)
    screen_e8_candidates_parser.add_argument("--limit", type=int)
    screen_e8_candidates_parser.set_defaults(handler=_screen_e8_candidates)

    prepare_e8_ledger_parser = subparsers.add_parser(
        "prepare-e8-ledger",
        help="freeze the selected E8 method matrix and create its exact ledger",
    )
    prepare_e8_ledger_parser.add_argument("runbook", type=Path)
    prepare_e8_ledger_parser.set_defaults(handler=_prepare_e8_ledger)

    run_e8_parser = subparsers.add_parser(
        "run-e8",
        help="resume the exact signed 86,040-row E8 development matrix",
    )
    run_e8_parser.add_argument("runbook", type=Path)
    run_e8_parser.add_argument("--limit", type=int)
    run_e8_parser.set_defaults(handler=_run_e8)

    finalize_e8_runbook_parser = subparsers.add_parser(
        "finalize-e8-runbook",
        help="derive E8 gates and publish its self-contained terminal artifact",
    )
    finalize_e8_runbook_parser.add_argument("runbook", type=Path)
    finalize_e8_runbook_parser.set_defaults(handler=_finalize_e8_runbook)

    verify_e8_runbook_parser = subparsers.add_parser(
        "verify-e8-runbook",
        help="replay E8 stage progress and its terminal package without VLLM",
    )
    verify_e8_runbook_parser.add_argument("runbook", type=Path)
    verify_e8_runbook_parser.set_defaults(handler=_verify_e8_runbook)

    stage_e9_inputs_parser = subparsers.add_parser(
        "stage-e9-inputs",
        help="atomically copy verified external E9 sources into the Qwen namespace",
    )
    stage_e9_inputs_parser.add_argument("output", type=Path)
    stage_e9_inputs_parser.add_argument("official_grader_bundle", type=Path)
    stage_e9_inputs_parser.add_argument("reviewed_splits", type=Path)
    stage_e9_inputs_parser.add_argument("--triviaqa-source", type=Path, required=True)
    stage_e9_inputs_parser.add_argument("--simpleqa-source", type=Path, required=True)
    stage_e9_inputs_parser.add_argument("--aa-source", type=Path, required=True)
    stage_e9_inputs_parser.add_argument(
        "--expected-official-grader-manifest-digest", required=True
    )
    stage_e9_inputs_parser.set_defaults(handler=_stage_e9_inputs)

    freeze_e9_inputs_parser = subparsers.add_parser(
        "freeze-e9-inputs",
        help=(
            "derive every E9 component, grader, question, robustness, and runbook "
            "artifact from terminal E0-E8 evidence"
        ),
    )
    freeze_e9_inputs_parser.add_argument("output", type=Path)
    freeze_e9_inputs_parser.add_argument("e8_runbook", type=Path)
    freeze_e9_inputs_parser.add_argument("e9_runbook_output", type=Path)
    freeze_e9_inputs_parser.add_argument("evaluation_scripts", type=Path)
    freeze_e9_inputs_parser.add_argument("official_grader_bundle", type=Path)
    freeze_e9_inputs_parser.add_argument("reviewed_splits", type=Path)
    freeze_e9_inputs_parser.add_argument("m2_source_artifact", type=Path)
    freeze_e9_inputs_parser.add_argument("e3_phase_run", type=Path)
    freeze_e9_inputs_parser.add_argument("--triviaqa-source", type=Path, required=True)
    freeze_e9_inputs_parser.add_argument("--simpleqa-source", type=Path, required=True)
    freeze_e9_inputs_parser.add_argument("--aa-source", type=Path, required=True)
    freeze_e9_inputs_parser.add_argument(
        "--expected-official-grader-manifest-digest", required=True
    )
    freeze_e9_inputs_parser.add_argument(
        "--robustness-config",
        type=Path,
        default=Path("configs/experiments/robustness-diagnostics.json"),
    )
    freeze_e9_inputs_parser.add_argument("--env-file", type=Path, default=Path(".env"))
    freeze_e9_inputs_parser.set_defaults(handler=_freeze_e9_inputs)

    prepare_e10_freezes_parser = subparsers.add_parser(
        "prepare-e10-freezes",
        help="derive and freeze the exact 10,000-row early-token capture plan",
    )
    prepare_e10_freezes_parser.add_argument("output", type=Path)
    prepare_e10_freezes_parser.add_argument("e8_runbook", type=Path)
    prepare_e10_freezes_parser.add_argument("e9_runbook", type=Path)
    prepare_e10_freezes_parser.set_defaults(handler=_prepare_e10_freezes)

    run_e10_early_probe_parser = subparsers.add_parser(
        "run-e10-early-probe",
        help="resume the signed native-VLLM early-token capture used only to freeze M6",
    )
    run_e10_early_probe_parser.add_argument("output", type=Path)
    run_e10_early_probe_parser.add_argument("e8_runbook", type=Path)
    run_e10_early_probe_parser.add_argument("e9_runbook", type=Path)
    run_e10_early_probe_parser.add_argument("--limit", type=int)
    run_e10_early_probe_parser.add_argument("--shard-rows", type=int, default=32)
    run_e10_early_probe_parser.add_argument(
        "--env-file", type=Path, default=Path(".env")
    )
    run_e10_early_probe_parser.set_defaults(handler=_run_e10_early_probe)

    verify_e10_early_probe_parser = subparsers.add_parser(
        "verify-e10-early-probe",
        help="replay signed early-token capture shards without loading VLLM",
    )
    verify_e10_early_probe_parser.add_argument("output", type=Path)
    verify_e10_early_probe_parser.add_argument("--require-complete", action="store_true")
    verify_e10_early_probe_parser.set_defaults(handler=_verify_e10_early_probe)

    finalize_e10_freezes_parser = subparsers.add_parser(
        "finalize-e10-freezes",
        help=(
            "fit the early probe and publish M6, all eleven E10 freezes, and a "
            "ready runbook"
        ),
    )
    finalize_e10_freezes_parser.add_argument("output", type=Path)
    finalize_e10_freezes_parser.add_argument("e8_runbook", type=Path)
    finalize_e10_freezes_parser.add_argument("e9_runbook", type=Path)
    finalize_e10_freezes_parser.add_argument("evaluation_scripts", type=Path)
    finalize_e10_freezes_parser.add_argument("e10_runbook_output", type=Path)
    finalize_e10_freezes_parser.set_defaults(handler=_finalize_e10_freezes)

    freeze_robustness_parser = subparsers.add_parser(
        "freeze-robustness-plan",
        help="freeze the source-bound 36,060-task post-E8/pre-E9 diagnostic schedule",
    )
    freeze_robustness_parser.add_argument("output", type=Path)
    freeze_robustness_parser.add_argument("e1_run", type=Path)
    freeze_robustness_parser.add_argument("component_selection", type=Path)
    freeze_robustness_parser.add_argument("evaluation_scripts", type=Path)
    freeze_robustness_parser.add_argument("graders", type=Path)
    freeze_robustness_parser.add_argument("reviewed_splits", type=Path)
    freeze_robustness_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/robustness-diagnostics.json"),
    )
    freeze_robustness_parser.add_argument(
        "--prompt-config", type=Path, default=Path("configs/prompts/primary.yaml")
    )
    freeze_robustness_parser.add_argument("--env-file", type=Path, default=Path(".env"))
    freeze_robustness_parser.set_defaults(handler=_freeze_robustness_plan)

    verify_robustness_plan_parser = subparsers.add_parser(
        "verify-robustness-plan",
        help="rebuild every task in a packaged post-E8 robustness schedule",
    )
    verify_robustness_plan_parser.add_argument("plan", type=Path)
    verify_robustness_plan_parser.set_defaults(handler=_verify_robustness_plan)

    create_robustness_results_parser = subparsers.add_parser(
        "create-robustness-results",
        help="create an append-only result store for the frozen robustness plan",
    )
    create_robustness_results_parser.add_argument("output", type=Path)
    create_robustness_results_parser.add_argument("plan", type=Path)
    create_robustness_results_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/robustness-diagnostics.json"),
    )
    create_robustness_results_parser.set_defaults(handler=_create_robustness_results)

    prepare_robustness_execution_parser = subparsers.add_parser(
        "prepare-robustness-execution",
        help="freeze the shared signed 30,000-row native RQ1 capture plan",
    )
    prepare_robustness_execution_parser.add_argument("output", type=Path)
    prepare_robustness_execution_parser.add_argument("results", type=Path)
    prepare_robustness_execution_parser.add_argument("e9_runbook", type=Path)
    prepare_robustness_execution_parser.add_argument("e3_construction", type=Path)
    prepare_robustness_execution_parser.add_argument(
        "--shard-rows", type=int, default=16
    )
    prepare_robustness_execution_parser.set_defaults(
        handler=_prepare_robustness_execution
    )

    run_robustness_capture_parser = subparsers.add_parser(
        "run-robustness-rq1-capture",
        help="resume the shared signed native-VLLM RQ1 fit capture",
    )
    run_robustness_capture_parser.add_argument("output", type=Path)
    run_robustness_capture_parser.add_argument("results", type=Path)
    run_robustness_capture_parser.add_argument("e9_runbook", type=Path)
    run_robustness_capture_parser.add_argument("e3_construction", type=Path)
    run_robustness_capture_parser.add_argument("--limit", type=int)
    run_robustness_capture_parser.add_argument(
        "--env-file", type=Path, default=Path(".env")
    )
    run_robustness_capture_parser.set_defaults(
        handler=_run_robustness_rq1_capture
    )

    verify_robustness_capture_parser = subparsers.add_parser(
        "verify-robustness-rq1-capture",
        help="replay the RQ1 native fit capture without loading VLLM",
    )
    verify_robustness_capture_parser.add_argument("output", type=Path)
    verify_robustness_capture_parser.add_argument("results", type=Path)
    verify_robustness_capture_parser.add_argument("e9_runbook", type=Path)
    verify_robustness_capture_parser.add_argument("e3_construction", type=Path)
    verify_robustness_capture_parser.add_argument(
        "--require-complete", action="store_true"
    )
    verify_robustness_capture_parser.add_argument(
        "--env-file", type=Path, default=Path(".env")
    )
    verify_robustness_capture_parser.set_defaults(
        handler=_verify_robustness_rq1_capture
    )

    run_robustness_prompts_parser = subparsers.add_parser(
        "run-robustness-prompts",
        help="resume frozen prompt-paraphrase tasks with native VLLM and graders",
    )
    run_robustness_prompts_parser.add_argument("results", type=Path)
    run_robustness_prompts_parser.add_argument("e9_runbook", type=Path)
    run_robustness_prompts_parser.add_argument("--limit", type=int)
    run_robustness_prompts_parser.add_argument(
        "--env-file", type=Path, default=Path(".env")
    )
    run_robustness_prompts_parser.set_defaults(handler=_run_robustness_prompts)

    run_robustness_rq1_parser = subparsers.add_parser(
        "run-robustness-rq1",
        help="fit, execute, grade, and append complete RQ1 fold tasks",
    )
    run_robustness_rq1_parser.add_argument("output", type=Path)
    run_robustness_rq1_parser.add_argument("results", type=Path)
    run_robustness_rq1_parser.add_argument("e9_runbook", type=Path)
    run_robustness_rq1_parser.add_argument("e3_construction", type=Path)
    run_robustness_rq1_parser.add_argument("e2_workspace", type=Path)
    run_robustness_rq1_parser.add_argument("e2_probe_bundle", type=Path)
    run_robustness_rq1_parser.add_argument("e5_fit_capture", type=Path)
    run_robustness_rq1_parser.add_argument("e5_layer_labels", type=Path)
    run_robustness_rq1_parser.add_argument("controller_questions", type=Path)
    run_robustness_rq1_parser.add_argument("--limit", type=int)
    run_robustness_rq1_parser.add_argument(
        "--env-file", type=Path, default=Path(".env")
    )
    run_robustness_rq1_parser.set_defaults(handler=_run_robustness_rq1)

    verify_robustness_results_parser = subparsers.add_parser(
        "verify-robustness-results",
        help="replay robustness task membership, artifacts, grading, and progress",
    )
    verify_robustness_results_parser.add_argument("results", type=Path)
    verify_robustness_results_parser.add_argument("--require-complete", action="store_true")
    verify_robustness_results_parser.set_defaults(handler=_verify_robustness_results)

    finalize_robustness_results_parser = subparsers.add_parser(
        "finalize-robustness-results",
        help="freeze completion after all 36,060 robustness tasks exist",
    )
    finalize_robustness_results_parser.add_argument("results", type=Path)
    finalize_robustness_results_parser.set_defaults(handler=_finalize_robustness_results)

    finalize_e7_parser = subparsers.add_parser(
        "finalize-e7",
        help="evaluate E7 gates and publish a self-contained terminal artifact",
    )
    finalize_e7_parser.add_argument("ledger", type=Path)
    finalize_e7_parser.add_argument("coordinate_artifact", type=Path)
    finalize_e7_parser.add_argument("sae_intervention", type=Path)
    finalize_e7_parser.add_argument("output", type=Path)
    finalize_e7_parser.add_argument(
        "--study-protocol",
        type=Path,
        default=Path("configs/experiments/phases.yaml"),
    )
    finalize_e7_parser.set_defaults(handler=_finalize_e7)

    verify_e7_parser = subparsers.add_parser(
        "verify-e7",
        help="replay a self-contained E7 terminal artifact without external paths",
    )
    verify_e7_parser.add_argument("output", type=Path)
    verify_e7_parser.set_defaults(handler=_verify_e7)

    finalize_e8_parser = subparsers.add_parser(
        "finalize-e8",
        help="evaluate E8 gates and publish a self-contained terminal artifact",
    )
    finalize_e8_parser.add_argument("ledger", type=Path)
    finalize_e8_parser.add_argument("protected_artifact", type=Path)
    finalize_e8_parser.add_argument("operating_point_registry", type=Path)
    finalize_e8_parser.add_argument("candidate_screen", type=Path)
    finalize_e8_parser.add_argument("runtime_artifact", type=Path)
    finalize_e8_parser.add_argument("output", type=Path)
    finalize_e8_parser.add_argument(
        "--study-protocol",
        type=Path,
        default=Path("configs/experiments/phases.yaml"),
    )
    finalize_e8_parser.add_argument(
        "--analysis-protocol",
        type=Path,
        default=Path("configs/analysis/confirmatory.yaml"),
    )
    finalize_e8_parser.add_argument(
        "--research-plan", type=Path, default=Path("docs/research-plan.md")
    )
    finalize_e8_parser.set_defaults(handler=_finalize_e8)

    verify_e8_parser = subparsers.add_parser(
        "verify-e8",
        help="replay a self-contained E8 terminal artifact without external paths",
    )
    verify_e8_parser.add_argument("output", type=Path)
    verify_e8_parser.set_defaults(handler=_verify_e8)

    runbook_template_parser = subparsers.add_parser(
        "write-confirmatory-runbook",
        help="write a secret-free E9/E10 native-VLLM operator runbook template",
    )
    runbook_template_parser.add_argument("phase", choices=("E9", "E10"))
    runbook_template_parser.add_argument("output", type=Path)
    runbook_template_parser.set_defaults(handler=_write_confirmatory_runbook)

    preflight_confirmatory_parser = subparsers.add_parser(
        "preflight-confirmatory",
        help="replay an E9/E10 runbook without creating a ledger or loading VLLM",
    )
    preflight_confirmatory_parser.add_argument("runbook", type=Path)
    preflight_confirmatory_parser.set_defaults(handler=_preflight_confirmatory)

    prepare_confirmatory_parser = subparsers.add_parser(
        "prepare-confirmatory",
        help="atomically create the exact E9/E10 ledger described by a runbook",
    )
    prepare_confirmatory_parser.add_argument("runbook", type=Path)
    prepare_confirmatory_parser.add_argument(
        "--authorize-e10-one-shot",
        action="store_true",
        help="explicitly consume the E10 write-once reservation after preflight",
    )
    prepare_confirmatory_parser.set_defaults(handler=_prepare_confirmatory)

    run_confirmatory_parser = subparsers.add_parser(
        "run-confirmatory",
        help="resume native Qwen VLLM E9/E10 generation and frozen grading",
    )
    run_confirmatory_parser.add_argument("runbook", type=Path)
    run_confirmatory_parser.add_argument("--env-file", type=Path, default=Path(".env"))
    run_confirmatory_parser.add_argument("--checkpoint-size", type=int, default=1)
    run_confirmatory_parser.add_argument("--limit", type=int)
    run_confirmatory_parser.set_defaults(handler=_run_confirmatory)

    finalize_confirmatory_parser = subparsers.add_parser(
        "finalize-confirmatory",
        help="derive gates and terminally finalize a complete E9/E10 run",
    )
    finalize_confirmatory_parser.add_argument("runbook", type=Path)
    finalize_confirmatory_parser.set_defaults(handler=_finalize_confirmatory)

    verify_confirmatory_parser = subparsers.add_parser(
        "verify-confirmatory",
        help="report progress or replay a terminal E9/E10 run",
    )
    verify_confirmatory_parser.add_argument("runbook", type=Path)
    verify_confirmatory_parser.set_defaults(handler=_verify_confirmatory)

    language_parser = subparsers.add_parser(
        "build-language-suite",
        help="package signed human-reviewed TriviaQA translations",
    )
    language_parser.add_argument("output", type=Path)
    language_parser.add_argument("triviaqa_source", type=Path)
    language_parser.add_argument("translations", type=Path)
    language_parser.add_argument("reviewer_registry", type=Path)
    language_parser.set_defaults(handler=_build_language_suite)

    verify_language_parser = subparsers.add_parser(
        "verify-language-suite",
        help="verify a signed human-reviewed language suite",
    )
    verify_language_parser.add_argument("path", type=Path)
    verify_language_parser.set_defaults(handler=lambda args: _verify_language_suite(args.path))

    transformers_snapshot_parser = subparsers.add_parser(
        "verify-transformers-snapshot",
        help="verify an exact symlink-free local copy of a pinned Hub snapshot",
    )
    transformers_snapshot_parser.add_argument("model_config", type=Path)
    transformers_snapshot_parser.add_argument("snapshot_directory", type=Path)
    transformers_snapshot_parser.add_argument("snapshot_manifest", type=Path)
    transformers_snapshot_parser.set_defaults(handler=_transformers_snapshot_preflight)

    vllm_preflight_parser = subparsers.add_parser(
        "preflight-vllm-runtime",
        help="freeze exact A100 Qwen architecture and intervention evidence",
    )
    vllm_preflight_parser.add_argument("model_config", type=Path)
    vllm_preflight_parser.add_argument("snapshot_directory", type=Path)
    vllm_preflight_parser.add_argument("snapshot_manifest", type=Path)
    vllm_preflight_parser.add_argument("runtime_policy", type=Path)
    vllm_preflight_parser.add_argument("output", type=Path)
    vllm_preflight_parser.add_argument("--project-root", type=Path, default=Path.cwd())
    vllm_preflight_parser.add_argument("--prompt", default="What is the capital of France?")
    vllm_preflight_parser.add_argument("--alpha", type=float, default=2.0)
    vllm_preflight_parser.set_defaults(handler=_vllm_hook_preflight)

    e0_vllm_parser = subparsers.add_parser(
        "run-e0-vllm",
        help="run or resume the sole 500-question repeated native VLLM leg of E0",
    )
    e0_vllm_parser.add_argument("cohort", type=Path)
    e0_vllm_parser.add_argument("reserved_source", type=Path)
    e0_vllm_parser.add_argument("model_config", type=Path)
    e0_vllm_parser.add_argument("snapshot_directory", type=Path)
    e0_vllm_parser.add_argument("snapshot_manifest", type=Path)
    e0_vllm_parser.add_argument("runtime_config", type=Path)
    e0_vllm_parser.add_argument("work", type=Path)
    e0_vllm_parser.add_argument("output", type=Path)
    e0_vllm_parser.add_argument("--expected-cohort-manifest-digest", required=True)
    e0_vllm_parser.add_argument("--parent-split-manifest-digest", required=True)
    e0_vllm_parser.add_argument("--contamination-manifest-digest", required=True)
    e0_vllm_parser.add_argument(
        "--prompt-config", type=Path, default=Path("configs/prompts/primary.yaml")
    )
    e0_vllm_parser.add_argument(
        "--inference-config", type=Path, default=Path("configs/experiments/core.yaml")
    )
    e0_vllm_parser.add_argument(
        "--study-config", type=Path, default=Path("configs/experiments/phases.yaml")
    )
    e0_vllm_parser.add_argument("--request-budget", type=int)
    e0_vllm_parser.add_argument("--expected-resume-checkpoint")
    e0_vllm_parser.add_argument("--checkpoint-file", type=Path)
    e0_vllm_parser.set_defaults(handler=_run_e0_vllm)

    verify_e0_vllm_parser = subparsers.add_parser(
        "verify-e0-vllm",
        help="replay a completed native VLLM E0 bundle against live pinned inputs",
    )
    verify_e0_vllm_parser.add_argument("directory", type=Path)
    verify_e0_vllm_parser.add_argument("cohort", type=Path)
    verify_e0_vllm_parser.add_argument("reserved_source", type=Path)
    verify_e0_vllm_parser.add_argument("model_config", type=Path)
    verify_e0_vllm_parser.add_argument("snapshot_directory", type=Path)
    verify_e0_vllm_parser.add_argument("snapshot_manifest", type=Path)
    verify_e0_vllm_parser.add_argument("runtime_config", type=Path)
    verify_e0_vllm_parser.add_argument("--expected-manifest-digest", required=True)
    verify_e0_vllm_parser.add_argument("--expected-plan-identity", required=True)
    verify_e0_vllm_parser.add_argument("--expected-cohort-manifest-digest", required=True)
    verify_e0_vllm_parser.add_argument("--parent-split-manifest-digest", required=True)
    verify_e0_vllm_parser.add_argument("--contamination-manifest-digest", required=True)
    verify_e0_vllm_parser.add_argument(
        "--prompt-config", type=Path, default=Path("configs/prompts/primary.yaml")
    )
    verify_e0_vllm_parser.add_argument(
        "--inference-config", type=Path, default=Path("configs/experiments/core.yaml")
    )
    verify_e0_vllm_parser.add_argument(
        "--study-config", type=Path, default=Path("configs/experiments/phases.yaml")
    )
    verify_e0_vllm_parser.set_defaults(handler=_verify_e0_vllm)

    e0_completion_parser = subparsers.add_parser(
        "complete-e0",
        help="promote E0 only after replayed VLLM validation and human contamination review",
    )
    e0_completion_parser.add_argument("output", type=Path)
    _add_e0_completion_evidence_arguments(e0_completion_parser)
    e0_completion_parser.set_defaults(handler=_write_e0_completion)

    verify_e0_completion_parser = subparsers.add_parser(
        "verify-e0-completion",
        help="replay an E0 scientific-completion receipt against all anchored evidence",
    )
    verify_e0_completion_parser.add_argument("receipt", type=Path)
    verify_e0_completion_parser.add_argument("--expected-manifest-digest", required=True)
    _add_e0_completion_evidence_arguments(verify_e0_completion_parser)
    verify_e0_completion_parser.set_defaults(handler=_verify_e0_completion)

    finalize_e0_phase_parser = subparsers.add_parser(
        "finalize-e0-phase",
        help="package verified native-VLLM E0 outputs into the immutable phase ledger",
    )
    finalize_e0_phase_parser.add_argument("output", type=Path)
    finalize_e0_phase_parser.add_argument("receipt", type=Path)
    finalize_e0_phase_parser.add_argument("--expected-manifest-digest", required=True)
    _add_e0_completion_evidence_arguments(finalize_e0_phase_parser)
    finalize_e0_phase_parser.set_defaults(handler=_finalize_e0_phase)

    synthetic_parser = subparsers.add_parser(
        "synthetic-smoke",
        help="exercise E0-E10 with deterministic, non-scientific synthetic data",
    )
    synthetic_parser.add_argument("output", type=Path)
    synthetic_parser.add_argument("--seed", type=int, default=1701)
    synthetic_parser.set_defaults(handler=_synthetic_smoke)

    verify_synthetic_parser = subparsers.add_parser(
        "verify-synthetic-smoke",
        help="verify and replay a frozen synthetic E0-E10 smoke bundle",
    )
    verify_synthetic_parser.add_argument("directory", type=Path)
    verify_synthetic_parser.set_defaults(handler=_verify_synthetic_smoke)

    audit_prepare_parser = subparsers.add_parser(
        "prepare-human-audit",
        help="build a deterministic blinded human-audit queue",
    )
    audit_prepare_parser.add_argument("output", type=Path)
    audit_prepare_parser.add_argument("analysis_protocol", type=Path)
    audit_prepare_parser.add_argument("study_protocol", type=Path)
    audit_prepare_parser.add_argument("e9_run", type=Path)
    audit_prepare_parser.add_argument("e10_run", type=Path)
    audit_prepare_parser.add_argument("--blinding-key-file", type=Path, required=True)
    audit_prepare_parser.set_defaults(handler=_prepare_human_audit)

    audit_finalize_parser = subparsers.add_parser(
        "finalize-human-audit",
        help="freeze two blinded annotations and required adjudications",
    )
    audit_finalize_parser.add_argument("queue", type=Path)
    audit_finalize_parser.add_argument("output", type=Path)
    audit_finalize_parser.add_argument("analysis_protocol", type=Path)
    audit_finalize_parser.add_argument("study_protocol", type=Path)
    audit_finalize_parser.add_argument("e9_run", type=Path)
    audit_finalize_parser.add_argument("e10_run", type=Path)
    audit_finalize_parser.add_argument("--blinding-key-file", type=Path, required=True)
    audit_finalize_parser.add_argument(
        "--annotation",
        action="append",
        required=True,
        help="exactly two ANNOTATOR_ID=CSV inputs",
    )
    audit_finalize_parser.add_argument("--adjudications", type=Path, required=True)
    audit_finalize_parser.set_defaults(handler=_finalize_human_audit)

    audit_queue_parser = subparsers.add_parser(
        "verify-human-audit-queue",
        help="verify a blinded audit queue and its private bindings",
    )
    audit_queue_parser.add_argument("queue", type=Path)
    audit_queue_parser.add_argument("analysis_protocol", type=Path)
    audit_queue_parser.add_argument("study_protocol", type=Path)
    audit_queue_parser.add_argument("e9_run", type=Path)
    audit_queue_parser.add_argument("e10_run", type=Path)
    audit_queue_parser.add_argument("--blinding-key-file", type=Path, required=True)
    audit_queue_parser.set_defaults(handler=_verify_human_audit_queue)

    audit_results_parser = subparsers.add_parser(
        "verify-human-audit-results",
        help="verify finalized audit evidence against its blinded queue",
    )
    audit_results_parser.add_argument("results", type=Path)
    audit_results_parser.add_argument("queue", type=Path)
    audit_results_parser.add_argument("analysis_protocol", type=Path)
    audit_results_parser.add_argument("study_protocol", type=Path)
    audit_results_parser.add_argument("e9_run", type=Path)
    audit_results_parser.add_argument("e10_run", type=Path)
    audit_results_parser.add_argument("--blinding-key-file", type=Path, required=True)
    audit_results_parser.set_defaults(handler=_verify_human_audit_results)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

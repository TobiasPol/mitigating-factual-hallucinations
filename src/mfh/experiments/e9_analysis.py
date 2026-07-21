"""Preregistered, immutable statistical analysis bundle for confirmatory E9."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from mfh.analysis.protocol import AnalysisProtocol, load_analysis_protocol
from mfh.analysis.statistics import (
    AnalysisMetric,
    MixedEffectsObservation,
    PairedOutcomes,
    bowker_test,
    fit_mixed_effects_logistic,
    holm_adjust,
    mcnemar_exact,
    paired_bootstrap_difference,
    paired_prompt_interaction,
    stuart_maxwell_test,
)
from mfh.contracts import GenerationRecord, Outcome
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.evaluation.metrics import metric_bundle
from mfh.experiments.model_selection import validate_active_study_artifact_paths
from mfh.experiments.protocol import ExperimentPhase
from mfh.experiments.snapshots import validate_execution_snapshot
from mfh.provenance import sha256_file, sha256_path, stable_hash

if TYPE_CHECKING:
    from mfh.experiments.runner import PhaseRunLedger

_FACTUAL = (
    "triviaqa",
    "simpleqa_verified",
    "aa_omniscience_public_600",
)
_PROMPTS = ("P0-neutral", "P2-calibrated-abstention")
_FILES = {
    "primary_contrasts.json",
    "prompt_method_interactions.json",
    "mixed_effects.json",
    "holm_corrections.json",
    "condition_summaries.json",
}
_OUTCOME_LABELS = (
    Outcome.CORRECT,
    Outcome.PARTIAL,
    Outcome.INCORRECT,
    Outcome.ABSTENTION,
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _analysis_protocol(ledger: PhaseRunLedger) -> tuple[AnalysisProtocol, str]:
    snapshot = ledger.directory / "inputs" / "frozen_evaluation_scripts"
    return load_e9_analysis_protocol_snapshot(
        snapshot,
        study_protocol_digest=ledger.contract.study_protocol_digest,
    )


def load_e9_analysis_protocol_snapshot(
    snapshot: str | Path,
    *,
    study_protocol_digest: str,
) -> tuple[AnalysisProtocol, str]:
    """Load the exact E9 analysis protocol and verify its frozen research plan."""

    source = Path(snapshot)
    manifest = validate_execution_snapshot(
        source,
        study_protocol_digest=study_protocol_digest,
        phase=ExperimentPhase.E9,
    )
    files = manifest["files"]
    if not isinstance(files, Mapping):  # pragma: no cover - snapshot validator guarantees it
        raise FrozenArtifactError("E9 execution snapshot files are invalid")
    analysis_descriptor = files["analysis_protocol"]
    plan_descriptor = files["research_plan"]
    if not isinstance(analysis_descriptor, Mapping) or not isinstance(
        plan_descriptor, Mapping
    ):
        raise FrozenArtifactError("E9 analysis source descriptors are invalid")
    protocol = load_analysis_protocol(source / str(analysis_descriptor["path"]))
    protocol.verify_research_plan(source / str(plan_descriptor["path"]))
    return protocol, sha256_path(source)


def _index_records(
    records: Sequence[GenerationRecord],
) -> dict[tuple[str, str, str], dict[str, GenerationRecord]]:
    indexed: dict[tuple[str, str, str], dict[str, GenerationRecord]] = {}
    for record in records:
        key = (record.benchmark, record.system_prompt_id, record.steering_method)
        cell = indexed.setdefault(key, {})
        if record.question_id in cell:
            raise DataValidationError("E9 analysis contains a duplicate condition row")
        cell[record.question_id] = record
    return indexed


def _paired(
    indexed: Mapping[tuple[str, str, str], Mapping[str, GenerationRecord]],
    *,
    benchmarks: Sequence[str],
    prompt: str,
    baseline: str,
    treatment: str,
) -> PairedOutcomes:
    identifiers: list[str] = []
    before: list[Outcome] = []
    after: list[Outcome] = []
    for benchmark in benchmarks:
        baseline_cell = indexed.get((benchmark, prompt, baseline))
        treatment_cell = indexed.get((benchmark, prompt, treatment))
        if baseline_cell is None or treatment_cell is None:
            raise DataValidationError("E9 analysis lacks a preregistered comparison cell")
        if set(baseline_cell) != set(treatment_cell):
            raise DataValidationError("E9 analysis comparison is not exactly question-paired")
        for question_id in sorted(baseline_cell):
            identifiers.append(question_id)
            before.append(baseline_cell[question_id].outcome)
            after.append(treatment_cell[question_id].outcome)
    return PairedOutcomes(tuple(identifiers), tuple(before), tuple(after))


def _seed(comparison_id: str) -> int:
    return 17 + int(stable_hash(comparison_id)[:8], 16)


def _paired_result(
    comparison_id: str,
    paired: PairedOutcomes,
    *,
    metric: AnalysisMetric,
    protocol: AnalysisProtocol,
    include_transition_hypotheses: bool = True,
) -> tuple[dict[str, Any], tuple[tuple[str, float], ...]]:
    bootstrap = paired_bootstrap_difference(
        paired,
        metric,
        resamples=protocol.bootstrap_resamples,
        confidence=protocol.confidence,
        seed=_seed(comparison_id),
    )
    mcnemar = mcnemar_exact(paired)
    bowker = bowker_test(paired, labels=_OUTCOME_LABELS)
    stuart = stuart_maxwell_test(paired, labels=_OUTCOME_LABELS)
    hypotheses = [(f"{comparison_id}:paired_bootstrap", bootstrap.two_sided_p_value)]
    if include_transition_hypotheses:
        hypotheses.extend(
            (
                (f"{comparison_id}:mcnemar", mcnemar.exact_p_value),
                (f"{comparison_id}:bowker", bowker.p_value),
                (f"{comparison_id}:stuart_maxwell", stuart.p_value),
            )
        )
    return (
        {
            "comparison_id": comparison_id,
            "paired_bootstrap": asdict(bootstrap),
            "mcnemar_exact": asdict(mcnemar),
            "bowker": asdict(bowker),
            "stuart_maxwell_sensitivity": asdict(stuart),
        },
        tuple(hypotheses),
    )


def _condition_summaries(
    indexed: Mapping[tuple[str, str, str], Mapping[str, GenerationRecord]],
) -> list[dict[str, Any]]:
    return [
        {
            "benchmark": benchmark,
            "prompt": prompt,
            "method": method,
            "question_count": len(cell),
            "metrics": metric_bundle(
                tuple(record.outcome for record in cell.values())
            ).to_dict(),
        }
        for (benchmark, prompt, method), cell in sorted(indexed.items())
    ]


def derive_e8_matching_basis(e8_ledger: PhaseRunLedger) -> Mapping[str, Any]:
    """Replay the exact completion-packaged E8 operating-point selection."""

    from mfh.methods.protected import load_e8_operating_point_registry

    if e8_ledger.contract.phase is not ExperimentPhase.E8:
        raise DataValidationError("E9 matching basis requires the E8 prerequisite")
    completion = e8_ledger.verify_complete()
    registry_path = (
        e8_ledger.directory
        / "gate-artifacts"
        / "matched_empirical_risk_or_coverage"
        / "operating-point-registry"
    )
    registry = load_e8_operating_point_registry(registry_path)
    conditions = {
        condition.condition_id: condition for condition in e8_ledger.contract.conditions
    }
    grouped: dict[str, list[Outcome]] = {}
    for record in e8_ledger.records():
        grouped.setdefault(record.condition_id, []).append(record.outcome)
    points: list[dict[str, Any]] = []
    for prompt, methods in sorted(registry.condition_ids_by_prompt.items()):
        for method, condition_id in sorted(methods.items()):
            try:
                condition = conditions[condition_id]
                outcomes = grouped[condition_id]
            except KeyError as exc:
                raise FrozenArtifactError(
                    "E8 operating point is absent from its completed ledger"
                ) from exc
            metrics = metric_bundle(outcomes)
            if metrics.coverage is None or metrics.hallucination_risk is None:
                raise FrozenArtifactError("E8 operating point has undefined risk or coverage")
            observed = (
                metrics.hallucination_risk
                if registry.matching_dimension == "hallucination_risk"
                else metrics.coverage
            )
            if (
                condition.system_prompt_id != prompt
                or condition.steering_method != method
                or condition.benchmark != "triviaqa"
                or condition.method_artifact_sha256 is None
            ):
                raise FrozenArtifactError(
                    "E8 operating point differs from its selected condition"
                )
            points.append(
                {
                    "prompt": prompt,
                    "method": method,
                    "condition_id": condition_id,
                    "method_artifact_sha256": condition.method_artifact_sha256,
                    "observed_coverage": metrics.coverage,
                    "observed_hallucination_risk": metrics.hallucination_risk,
                    "observed_matching_value": observed,
                    "absolute_mismatch": abs(observed - registry.target),
                }
            )
    maximum_mismatch = max(point["absolute_mismatch"] for point in points)
    if maximum_mismatch > registry.tolerance:
        raise FrozenArtifactError("completed E8 operating points exceed frozen tolerance")
    return {
        "schema_version": 1,
        "selection_phase": ExperimentPhase.E8.value,
        "registered_gate": "matched_empirical_risk_or_coverage",
        "e8_completion_digest": completion.completion_digest,
        "operating_point_registry_sha256": sha256_file(registry_path),
        "matching_dimension": registry.matching_dimension,
        "target": registry.target,
        "tolerance": registry.tolerance,
        "candidate_screen_sha256": registry.candidate_screen_sha256,
        "maximum_absolute_mismatch": maximum_mismatch,
        "selected_points": points,
        "frozen_before_e9": True,
        "confirmatory_outcomes_post_hoc_rematched": False,
    }


def load_e9_matching_basis(ledger: PhaseRunLedger) -> Mapping[str, Any]:
    """Resolve and replay E9's exact completed E8 prerequisite."""

    from mfh.experiments.runner import PhaseRunLedger, _resolve_ledger_evidence_path

    ledger._verify_creation_evidence()
    try:
        creation = json.loads(
            (ledger.directory / "creation-evidence.json").read_text(encoding="utf-8")
        )
        location = creation["prerequisite_runs"]["E8"]["location"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise FrozenArtifactError(f"E9 cannot resolve its E8 prerequisite: {exc}") from exc
    e8 = PhaseRunLedger.open(
        _resolve_ledger_evidence_path(
            ledger.directory,
            location,
            context="E9 E8 matching prerequisite",
        ),
        study=ledger.study,
    )
    return derive_e8_matching_basis(e8)


def _validated_matching_basis(
    value: Mapping[str, Any],
    *,
    e8_completion_digest: str,
) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "selection_phase",
        "registered_gate",
        "e8_completion_digest",
        "operating_point_registry_sha256",
        "matching_dimension",
        "target",
        "tolerance",
        "candidate_screen_sha256",
        "maximum_absolute_mismatch",
        "selected_points",
        "frozen_before_e9",
        "confirmatory_outcomes_post_hoc_rematched",
    }
    points = value.get("selected_points")
    numeric = (
        value.get("target"),
        value.get("tolerance"),
        value.get("maximum_absolute_mismatch"),
    )
    if (
        set(value) != expected_fields
        or value.get("schema_version") != 1
        or value.get("selection_phase") != "E8"
        or value.get("registered_gate") != "matched_empirical_risk_or_coverage"
        or value.get("e8_completion_digest") != e8_completion_digest
        or not isinstance(value.get("operating_point_registry_sha256"), str)
        or _SHA256.fullmatch(value["operating_point_registry_sha256"]) is None
        or value.get("matching_dimension") not in {"hallucination_risk", "coverage"}
        or not isinstance(value.get("candidate_screen_sha256"), str)
        or _SHA256.fullmatch(value["candidate_screen_sha256"]) is None
        or any(
            isinstance(item, bool)
            or not isinstance(item, int | float)
            or not math.isfinite(float(item))
            for item in numeric
        )
        or not 0 <= float(value["target"]) <= 1
        or not 0 <= float(value["tolerance"]) <= 0.02
        or not isinstance(points, list)
        or len(points) != 8
        or value.get("frozen_before_e9") is not True
        or value.get("confirmatory_outcomes_post_hoc_rematched") is not False
    ):
        raise DataValidationError("E9 matching basis is invalid")
    expected_pairs = {
        (prompt, method)
        for prompt in _PROMPTS
        for method in ("M1", "M3", "M4", "M5")
    }
    observed_pairs: set[tuple[str, str]] = set()
    mismatches: list[float] = []
    for point in points:
        if not isinstance(point, Mapping) or set(point) != {
            "prompt",
            "method",
            "condition_id",
            "method_artifact_sha256",
            "observed_coverage",
            "observed_hallucination_risk",
            "observed_matching_value",
            "absolute_mismatch",
        }:
            raise DataValidationError("E9 matching point schema differs")
        pair = (str(point["prompt"]), str(point["method"]))
        observed_pairs.add(pair)
        numbers = tuple(
            point[name]
            for name in (
                "observed_coverage",
                "observed_hallucination_risk",
                "observed_matching_value",
                "absolute_mismatch",
            )
        )
        if (
            any(
                isinstance(item, bool)
                or not isinstance(item, int | float)
                or not math.isfinite(float(item))
                for item in numbers
            )
            or any(not 0 <= float(item) <= 1 for item in numbers[:3])
            or not isinstance(point["condition_id"], str)
            or _SHA256.fullmatch(point["condition_id"]) is None
            or not isinstance(point["method_artifact_sha256"], str)
            or _SHA256.fullmatch(point["method_artifact_sha256"]) is None
        ):
            raise DataValidationError("E9 matching point is invalid")
        observed_value = (
            float(point["observed_hallucination_risk"])
            if value["matching_dimension"] == "hallucination_risk"
            else float(point["observed_coverage"])
        )
        mismatch = abs(observed_value - float(value["target"]))
        if not math.isclose(
            float(point["observed_matching_value"]), observed_value, abs_tol=1e-15
        ) or not math.isclose(
            float(point["absolute_mismatch"]), mismatch, abs_tol=1e-15
        ):
            raise DataValidationError("E9 matching point does not recompute")
        mismatches.append(mismatch)
    if observed_pairs != expected_pairs or not math.isclose(
        float(value["maximum_absolute_mismatch"]), max(mismatches), abs_tol=1e-15
    ) or max(mismatches) > float(value["tolerance"]):
        raise DataValidationError("E9 matching basis selections or tolerance differ")
    return cast(
        dict[str, Any],
        json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False)),
    )


def _derive_analysis(
    records: Sequence[GenerationRecord],
    protocol: AnalysisProtocol,
    prerequisite_completion_digests: Mapping[str, str],
    e8_matching_basis: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    e8_completion_digest = prerequisite_completion_digests.get("E8")
    if (
        not isinstance(e8_completion_digest, str)
        or _SHA256.fullmatch(e8_completion_digest) is None
    ):
        raise DataValidationError("E9 analysis lacks its exact E8 completion identity")
    matching_basis = _validated_matching_basis(
        e8_matching_basis,
        e8_completion_digest=e8_completion_digest,
    )
    indexed = _index_records(records)
    primary: dict[str, list[dict[str, Any]]] = {
        "RQ1": [],
        "RQ2": [],
        "RQ3": [],
        "RQ4": [],
    }
    hypotheses: list[tuple[str, float]] = []

    for benchmark, metric in (
        ("simpleqa_verified", AnalysisMetric.SIMPLEQA_F1),
        ("aa_omniscience_public_600", AnalysisMetric.OMNISCIENCE_INDEX),
    ):
        for prompt in _PROMPTS:
            comparison_id = f"RQ1:{benchmark}:{prompt}:M1-v-M3:{metric.value}"
            result, p_values = _paired_result(
                comparison_id,
                _paired(
                    indexed,
                    benchmarks=(benchmark,),
                    prompt=prompt,
                    baseline="M1",
                    treatment="M3",
                ),
                metric=metric,
                protocol=protocol,
            )
            primary["RQ1"].append(result)
            hypotheses.extend(p_values)

    for prompt in _PROMPTS:
        for treatment in ("M1", "M3"):
            comparison_id = f"RQ2:all-factual:{prompt}:M0-v-{treatment}:accuracy"
            result, p_values = _paired_result(
                comparison_id,
                _paired(
                    indexed,
                    benchmarks=_FACTUAL,
                    prompt=prompt,
                    baseline="M0",
                    treatment=treatment,
                ),
                metric=AnalysisMetric.ACCURACY,
                protocol=protocol,
            )
            primary["RQ2"].append(result)
            hypotheses.extend(p_values)

    for prompt in _PROMPTS:
        for treatment in ("M4", "M5"):
            paired = _paired(
                indexed,
                benchmarks=_FACTUAL,
                prompt=prompt,
                baseline="M1",
                treatment=treatment,
            )
            metric_results: list[dict[str, Any]] = []
            for metric_index, metric in enumerate(
                (
                AnalysisMetric.HALLUCINATION_RISK,
                AnalysisMetric.COVERAGE,
                )
            ):
                comparison_id = f"RQ3:all-factual:{prompt}:M1-v-{treatment}:{metric.value}"
                result, p_values = _paired_result(
                    comparison_id,
                    paired,
                    metric=metric,
                    protocol=protocol,
                    include_transition_hypotheses=metric_index == 0,
                )
                metric_results.append(result)
                hypotheses.extend(p_values)
            primary["RQ3"].append(
                {
                    "comparison_id": f"RQ3:all-factual:{prompt}:M1-v-{treatment}",
                    "matching_basis": matching_basis,
                    "risk_and_coverage_contrasts": metric_results,
                }
            )
    interactions: list[dict[str, Any]] = []
    all_question_ids = _paired(
        indexed,
        benchmarks=_FACTUAL,
        prompt="P0-neutral",
        baseline="M0",
        treatment="M1",
    ).question_ids
    for treatment in ("M1", "M3", "M5"):
        neutral = _paired(
            indexed,
            benchmarks=_FACTUAL,
            prompt="P0-neutral",
            baseline="M0",
            treatment=treatment,
        )
        calibrated = _paired(
            indexed,
            benchmarks=_FACTUAL,
            prompt="P2-calibrated-abstention",
            baseline="M0",
            treatment=treatment,
        )
        if neutral.question_ids != all_question_ids or calibrated.question_ids != all_question_ids:
            raise DataValidationError("E9 prompt interaction question sets differ")
        for metric in (
            AnalysisMetric.ACCURACY,
            AnalysisMetric.COVERAGE,
            AnalysisMetric.HALLUCINATION_RISK,
        ):
            comparison_id = f"RQ4:all-factual:M0-v-{treatment}:{metric.value}"
            interaction = paired_prompt_interaction(
                all_question_ids,
                neutral.baseline,
                neutral.treatment,
                calibrated.baseline,
                calibrated.treatment,
                metric,
                resamples=protocol.bootstrap_resamples,
                confidence=protocol.confidence,
                seed=_seed(comparison_id),
            )
            value = {
                "comparison_id": comparison_id,
                "baseline_method": "M0",
                "treatment_method": treatment,
                "neutral_prompt": "P0-neutral",
                "calibrated_prompt": "P2-calibrated-abstention",
                "result": asdict(interaction),
            }
            interactions.append(value)
            primary["RQ4"].append(value)
            hypotheses.append(
                (f"{comparison_id}:paired_bootstrap", interaction.two_sided_p_value)
            )

    mixed = fit_mixed_effects_logistic(
        MixedEffectsObservation(
            question_id=record.question_id,
            correct=record.outcome is Outcome.CORRECT,
            model=record.model_repository,
            benchmark=record.benchmark,
            method=record.steering_method,
            prompt=record.system_prompt_id,
        )
        for record in records
    )
    if not mixed.converged:
        raise DataValidationError(
            "preregistered mixed-effects fit did not converge; E9 analysis is incomplete"
        )
    holm = holm_adjust(hypotheses, alpha=protocol.alpha)
    common = {
        "schema_version": 2,
        "analysis_protocol_digest": protocol.digest,
        "bootstrap_resamples": protocol.bootstrap_resamples,
        "confidence": protocol.confidence,
        "alpha": protocol.alpha,
        "prerequisite_completion_digests": dict(
            sorted(prerequisite_completion_digests.items())
        ),
    }
    return {
        "primary_contrasts.json": {**common, "contrasts": primary},
        "prompt_method_interactions.json": {
            **common,
            "interactions": interactions,
        },
        "mixed_effects.json": {**common, "result": asdict(mixed)},
        "holm_corrections.json": {
            **common,
            "family": "E9_primary_paired_tests",
            "hypotheses": [asdict(value) for value in holm],
        },
        "condition_summaries.json": {
            **common,
            "conditions": _condition_summaries(indexed),
        },
    }


def write_e9_analysis_bundle(
    directory: str | Path,
    *,
    ledger: PhaseRunLedger,
) -> str:
    """Execute and atomically freeze all E9 analyses from the immutable ledger."""

    normalized = validate_active_study_artifact_paths(
        {"E9 analysis bundle": directory, "E9 phase ledger": ledger.directory}
    )
    destination = normalized["E9 analysis bundle"]
    if destination.exists() or destination.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite E9 analysis: {destination}")
    completed, expected = ledger.progress()
    if ledger.contract.phase is not ExperimentPhase.E9 or completed != expected:
        raise DataValidationError("E9 analysis requires the complete confirmatory ledger")
    protocol, snapshot_sha256 = _analysis_protocol(ledger)
    records = tuple(ledger.records())
    if len(records) != expected or any(record.outcome is Outcome.UNSCORABLE for record in records):
        raise DataValidationError("E9 analysis requires every row to be finally scorable")
    prerequisite_digests = dict(ledger.contract.prerequisite_digests)
    matching_basis = load_e9_matching_basis(ledger)
    outputs = _derive_analysis(
        records,
        protocol,
        prerequisite_digests,
        matching_basis,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent)
    )
    try:
        for name, value in outputs.items():
            _write_json(stage / name, value)
        body = {
            "schema_version": 2,
            "phase": ExperimentPhase.E9.value,
            "contract_digest": ledger.contract.digest,
            "record_set_digest": ledger.record_set_digest(),
            "record_count": expected,
            "analysis_protocol_digest": protocol.digest,
            "execution_snapshot_sha256": snapshot_sha256,
            "prerequisite_completion_digests": prerequisite_digests,
            "e8_matching_basis_digest": stable_hash(matching_basis),
            "files": {name: sha256_file(stage / name) for name in sorted(_FILES)},
        }
        _write_json(stage / "manifest.json", {**body, "manifest_digest": stable_hash(body)})
        validate_e9_analysis_bundle(
            stage,
            contract_digest=ledger.contract.digest,
            record_set_digest=ledger.record_set_digest(),
            record_count=expected,
            execution_snapshot_sha256=snapshot_sha256,
            records=records,
            protocol=protocol,
            prerequisite_completion_digests=prerequisite_digests,
            e8_matching_basis=matching_basis,
        )
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return sha256_path(destination)


def validate_e9_analysis_bundle(
    directory: str | Path,
    *,
    contract_digest: str,
    record_set_digest: str,
    record_count: int,
    execution_snapshot_sha256: str,
    records: Sequence[GenerationRecord],
    protocol: AnalysisProtocol,
    prerequisite_completion_digests: Mapping[str, str],
    e8_matching_basis: Mapping[str, Any],
) -> str:
    """Validate a complete, run-bound E9 analysis output bundle."""

    source = Path(directory)
    if (
        source.is_symlink()
        or not source.is_dir()
        or {item.name for item in source.iterdir()} != _FILES | {"manifest.json"}
        or any(item.is_symlink() or not item.is_file() for item in source.iterdir())
    ):
        raise FrozenArtifactError("E9 analysis bundle inventory differs")
    try:
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read E9 analysis manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise FrozenArtifactError("E9 analysis manifest is invalid")
    digest = manifest.pop("manifest_digest", None)
    expected_fields = {
        "schema_version",
        "phase",
        "contract_digest",
        "record_set_digest",
        "record_count",
        "analysis_protocol_digest",
        "execution_snapshot_sha256",
        "prerequisite_completion_digests",
        "e8_matching_basis_digest",
        "files",
    }
    files = manifest.get("files")
    if (
        set(manifest) != expected_fields
        or manifest.get("schema_version") != 2
        or manifest.get("phase") != ExperimentPhase.E9.value
        or manifest.get("contract_digest") != contract_digest
        or manifest.get("record_set_digest") != record_set_digest
        or manifest.get("record_count") != record_count
        or manifest.get("analysis_protocol_digest") != protocol.digest
        or manifest.get("prerequisite_completion_digests")
        != dict(prerequisite_completion_digests)
        or manifest.get("e8_matching_basis_digest")
        != stable_hash(e8_matching_basis)
        or manifest.get("execution_snapshot_sha256") != execution_snapshot_sha256
        or digest != stable_hash(manifest)
        or not isinstance(files, Mapping)
        or set(files) != _FILES
        or any(files[name] != sha256_file(source / name) for name in _FILES)
    ):
        raise FrozenArtifactError("E9 analysis bundle identity differs")
    try:
        values = {
            name: json.loads((source / name).read_text(encoding="utf-8"))
            for name in _FILES
        }
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read E9 analysis output: {exc}") from exc
    record_values = tuple(records)
    if (
        len(record_values) != record_count
        or any(record.outcome is Outcome.UNSCORABLE for record in record_values)
    ):
        raise FrozenArtifactError("E9 analysis replay records are incomplete")
    expected_values = json.loads(
        json.dumps(
            _derive_analysis(
                record_values,
                protocol,
                prerequisite_completion_digests,
                e8_matching_basis,
            ),
            sort_keys=True,
            allow_nan=False,
        )
    )
    if values != expected_values:
        raise FrozenArtifactError("E9 analysis outputs differ from exact preregistered replay")
    return sha256_path(source)

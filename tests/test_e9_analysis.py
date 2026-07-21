from __future__ import annotations

import json
from pathlib import Path

import pytest

from mfh.analysis.protocol import load_analysis_protocol
from mfh.analysis.statistics import MixedEffectsLogisticResult
from mfh.contracts import GenerationRecord, Outcome, Runtime
from mfh.errors import FrozenArtifactError
from mfh.experiments import e9_analysis
from mfh.provenance import sha256_file, sha256_path, stable_hash

ROOT = Path(__file__).parents[1]
OUTPUT_FILES = {
    "primary_contrasts.json",
    "prompt_method_interactions.json",
    "mixed_effects.json",
    "holm_corrections.json",
    "condition_summaries.json",
}


def _records() -> tuple[GenerationRecord, ...]:
    records: list[GenerationRecord] = []
    benchmarks = (
        "triviaqa",
        "simpleqa_verified",
        "aa_omniscience_public_600",
    )
    prompts = ("P0-neutral", "P1-direct", "P2-calibrated-abstention")
    methods = ("M0", "M1", "M2", "M3", "M4", "M5")
    for benchmark_index, benchmark in enumerate(benchmarks):
        for prompt_index, prompt in enumerate(prompts):
            for method_index, method in enumerate(methods):
                for question_index in range(5):
                    correct = (
                        benchmark_index
                        + prompt_index
                        + method_index
                        + question_index
                    ) % 3
                    records.append(
                        GenerationRecord(
                            question_id=f"{benchmark}:q{question_index}",
                            benchmark=benchmark,
                            model_repository="synthetic/model",
                            model_revision="synthetic",
                            runtime=Runtime.SYNTHETIC,
                            quantization="none",
                            system_prompt_id=prompt,
                            rendered_prompt_hash="rendered",
                            steering_method=method,
                            layer=None,
                            site=None,
                            token_scope=None,
                            alpha=0.0,
                            sparsity=None,
                            controller_scores={},
                            raw_output="answer",
                            normalized_answer="answer",
                            outcome=(
                                Outcome.CORRECT
                                if correct
                                else Outcome.INCORRECT
                            ),
                            generation_latency_seconds=0.1,
                            input_tokens=5,
                            output_tokens=1,
                            condition_id=f"{benchmark}:{prompt}:{method}",
                        )
                    )
    return tuple(records)


def _converged_fit(observations: object) -> MixedEffectsLogisticResult:
    values = tuple(observations)  # type: ignore[arg-type]
    return MixedEffectsLogisticResult(
        formula="correct ~ C(model) + C(benchmark) + C(method) * C(prompt)",
        random_effects="0 + C(question_id)",
        estimator="statsmodels.BinomialBayesMixedGLM.fit_map",
        observations=len(values),
        questions=len({value.question_id for value in values}),
        fixed_effect_names=("Intercept",),
        coefficients=(0.0,),
        standard_errors=(0.1,),
        converged=True,
    )


def _matching_basis() -> dict[str, object]:
    points = []
    for prompt_index, prompt in enumerate(("P0-neutral", "P2-calibrated-abstention")):
        for method_index, method in enumerate(("M1", "M3", "M4", "M5")):
            index = prompt_index * 4 + method_index + 1
            observed = 0.3 + index / 10_000
            points.append(
                {
                    "prompt": prompt,
                    "method": method,
                    "condition_id": f"{index:064x}",
                    "method_artifact_sha256": f"{index + 100:064x}",
                    "observed_coverage": 0.7,
                    "observed_hallucination_risk": observed,
                    "observed_matching_value": observed,
                    "absolute_mismatch": abs(observed - 0.3),
                }
            )
    return {
        "schema_version": 1,
        "selection_phase": "E8",
        "registered_gate": "matched_empirical_risk_or_coverage",
        "e8_completion_digest": "8" * 64,
        "operating_point_registry_sha256": "9" * 64,
        "matching_dimension": "hallucination_risk",
        "target": 0.3,
        "tolerance": 0.01,
        "candidate_screen_sha256": "a" * 64,
        "maximum_absolute_mismatch": max(
            float(point["absolute_mismatch"]) for point in points
        ),
        "selected_points": points,
        "frozen_before_e9": True,
        "confirmatory_outcomes_post_hoc_rematched": False,
    }


def _write_bundle(
    directory: Path,
    outputs: dict[str, object],
    *,
    protocol_digest: str,
    record_count: int,
    prerequisite_digests: dict[str, str],
    matching_basis: dict[str, object],
) -> None:
    directory.mkdir()
    for name, value in outputs.items():
        (directory / name).write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    body = {
        "schema_version": 2,
        "phase": "E9",
        "contract_digest": "a" * 64,
        "record_set_digest": "b" * 64,
        "record_count": record_count,
        "analysis_protocol_digest": protocol_digest,
        "execution_snapshot_sha256": "c" * 64,
        "prerequisite_completion_digests": prerequisite_digests,
        "e8_matching_basis_digest": stable_hash(matching_basis),
        "files": {
            name: sha256_file(directory / name) for name in sorted(OUTPUT_FILES)
        },
    }
    (directory / "manifest.json").write_text(
        json.dumps(
            {**body, "manifest_digest": stable_hash(body)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_e9_analysis_replays_every_output_and_full_holm_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = load_analysis_protocol(ROOT / "configs/analysis/confirmatory.yaml")
    records = _records()
    prerequisite_digests = {"E8": "8" * 64}
    matching_basis = _matching_basis()
    monkeypatch.setattr(e9_analysis, "fit_mixed_effects_logistic", _converged_fit)
    outputs = e9_analysis._derive_analysis(
        records,
        protocol,
        prerequisite_digests,
        matching_basis,
    )

    interactions = outputs["prompt_method_interactions.json"]["interactions"]
    hypotheses = outputs["holm_corrections.json"]["hypotheses"]
    assert len(interactions) == 9
    assert len(outputs["primary_contrasts.json"]["contrasts"]["RQ4"]) == 9
    assert len(hypotheses) == 61
    assert all(
        value["result"]["metric"]
        in {"accuracy", "coverage", "hallucination_risk"}
        for value in interactions
    )

    bundle = tmp_path / "analysis"
    _write_bundle(
        bundle,
        outputs,
        protocol_digest=protocol.digest,
        record_count=len(records),
        prerequisite_digests=prerequisite_digests,
        matching_basis=matching_basis,
    )
    assert e9_analysis.validate_e9_analysis_bundle(
        bundle,
        contract_digest="a" * 64,
        record_set_digest="b" * 64,
        record_count=len(records),
        execution_snapshot_sha256="c" * 64,
        records=records,
        protocol=protocol,
        prerequisite_completion_digests=prerequisite_digests,
        e8_matching_basis=matching_basis,
    ) == sha256_path(bundle)

    primary_path = bundle / "primary_contrasts.json"
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    primary["contrasts"]["RQ1"][0]["paired_bootstrap"]["difference"] = 999.0
    primary_path.write_text(
        json.dumps(primary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["primary_contrasts.json"] = sha256_file(primary_path)
    manifest.pop("manifest_digest")
    manifest_path.write_text(
        json.dumps(
            {**manifest, "manifest_digest": stable_hash(manifest)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(FrozenArtifactError, match="exact preregistered replay"):
        e9_analysis.validate_e9_analysis_bundle(
            bundle,
            contract_digest="a" * 64,
            record_set_digest="b" * 64,
            record_count=len(records),
            execution_snapshot_sha256="c" * 64,
            records=records,
            protocol=protocol,
            prerequisite_completion_digests=prerequisite_digests,
            e8_matching_basis=matching_basis,
        )

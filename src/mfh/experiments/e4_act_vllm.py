"""Portable risk-gated ACT/SADI-style baseline for the native VLLM E4 screen."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from mfh.contracts import ActivationSite, Runtime, TokenScope
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.experiments.e2_probes import VerifiedE2ProbeBundle, verify_e2_probe_bundle
from mfh.experiments.e2_schedule import verify_e2_workspace
from mfh.experiments.model_selection import validate_active_study_artifact_paths
from mfh.experiments.protocol import ExperimentPhase, StudyProtocol
from mfh.experiments.runner import PhaseRunLedger
from mfh.experiments.static_direction_sources import (
    ResolvedStaticDirection,
    resolve_static_direction,
)
from mfh.methods.features import ActivationKind, FeatureComposition
from mfh.methods.probes import CalibratedProbe, ProbeTask, load_calibrated_probe
from mfh.provenance import sha256_path, stable_hash

_INVENTORY = frozenset({"manifest.json", "risk-probe", "m2-direction"})


@dataclass(frozen=True, slots=True)
class E4ActBaseline:
    directory: Path
    risk_probe: CalibratedProbe
    direction: ResolvedStaticDirection
    feature_layer: int
    feature_site: ActivationSite
    intervention_layer: int
    intervention_site: ActivationSite
    token_scope: TokenScope
    source_e2_sha256: str
    source_e2_completion_digest: str
    e2_probe_manifest_digest: str
    e2_workspace_plan_identity: str
    source_m2_sha256: str
    manifest_digest: str


def _read_object(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read {context}: {exc}") from exc
    if type(value) is not dict:
        raise FrozenArtifactError(f"{context} must contain an object")
    return value


def _selected_e2_risk_probe(
    directory: Path,
    *,
    expected_selected_artifact_sha256: str,
) -> tuple[CalibratedProbe, Path, str]:
    expected = {
        "plan.json",
        "screening.json",
        "results.json",
        "probes",
        "screening-probes",
        "manifest.json",
    }
    if (
        directory.is_symlink()
        or not directory.is_dir()
        or {item.name for item in directory.iterdir()} != expected
        or any(item.is_symlink() for item in directory.rglob("*"))
    ):
        raise FrozenArtifactError("E4 ACT source E2 probe inventory differs")
    manifest = _read_object(directory / "manifest.json", "E2 probe manifest")
    manifest_body = dict(manifest)
    manifest_digest = manifest_body.pop("manifest_digest", None)
    results = _read_object(directory / "results.json", "E2 probe results")
    if (
        manifest_digest != stable_hash(manifest_body)
        or set(results) != {"schema_version", "selected_views", "final_probes", "gate"}
        or results.get("schema_version") != 1
        or not isinstance(results.get("final_probes"), list)
        or not isinstance(results.get("gate"), dict)
        or results["gate"].get("passed") is not True
    ):
        raise FrozenArtifactError("E4 ACT source E2 probe provenance differs")
    selected_sha = results["gate"].get("selected_artifact_sha256")
    if selected_sha != expected_selected_artifact_sha256:
        raise FrozenArtifactError("E4 ACT selection differs from the replayed E2 gate")
    matches = [
        row
        for row in results["final_probes"]
        if isinstance(row, dict)
        and row.get("artifact_sha256") == selected_sha
        and row.get("task") == ProbeTask.CORRECT_INCORRECT_ABSTENTION.value
    ]
    if len(matches) != 1:
        raise FrozenArtifactError("E4 ACT source lacks one selected C/I/A risk probe")
    row = matches[0]
    relative = row.get("artifact")
    if (
        not isinstance(relative, str)
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise FrozenArtifactError("E4 ACT selected probe path is invalid")
    artifact = directory / "probes" / relative
    if sha256_path(artifact) != selected_sha:
        raise FrozenArtifactError("E4 ACT selected risk probe changed")
    probe = load_calibrated_probe(artifact)
    schema = probe.training_schema
    if (
        probe.task is not ProbeTask.CORRECT_INCORRECT_ABSTENTION
        or probe.state.labels != ("C", "I", "A")
        or schema.model_repository != "nvidia/Qwen3.6-27B-NVFP4"
        or schema.model_revision != "0893e1606ff3d5f97a441f405d5fc541a6bdf404"
        or schema.runtime is not Runtime.VLLM
        or schema.quantization != "modelopt-mixed-nvfp4-fp8"
        or schema.prompt_id != "P0-neutral"
        or schema.activation_kind is not ActivationKind.FINAL_PROMPT
        or schema.composition is not FeatureComposition.SINGLE_LAYER
        or len(schema.layers) != 1
        or len(schema.sites) != 1
        or schema.width != 5_120
    ):
        raise FrozenArtifactError("E4 ACT selected risk-probe representation differs")
    return probe, artifact, str(selected_sha)


def _package_e4_act_baseline(
    directory: str | Path,
    *,
    e2_probe_bundle: str | Path,
    m2_caa_artifact: str | Path,
    intervention_layer: int,
    verified_e2: VerifiedE2ProbeBundle,
    e2_completion_digest: str,
    e2_workspace_plan_identity: str,
    token_scope: TokenScope = TokenScope.FIRST_FOUR,
) -> E4ActBaseline:
    destination = validate_active_study_artifact_paths(
        {"E4 ACT baseline": directory}
    )["E4 ACT baseline"]
    e2 = Path(e2_probe_bundle).resolve()
    m2 = Path(m2_caa_artifact).resolve()
    if verified_e2.directory.resolve() != e2:
        raise DataValidationError("verified E2 probe bundle path differs")
    probe, probe_path, selected_probe_sha = _selected_e2_risk_probe(
        e2,
        expected_selected_artifact_sha256=verified_e2.selected_gate_artifact,
    )
    direction = resolve_static_direction(
        m2,
        method="M2",
        layer=intervention_layer,
        site=ActivationSite.BLOCK_OUTPUT,
    )
    if token_scope is not TokenScope.FIRST_FOUR:
        raise DataValidationError("E4 ACT baseline requires first-four-token steering")
    if destination.exists() or destination.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite E4 ACT baseline: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        shutil.copytree(probe_path, stage / "risk-probe")
        shutil.copytree(m2, stage / "m2-direction")
        schema = probe.training_schema
        body = {
            "schema_version": 1,
            "method": "ACT-or-SADI",
            "design": "calibrated-E2-risk-gated-single-M2-CAA-direction",
            "source_e2_sha256": sha256_path(e2),
            "source_e2_completion_digest": e2_completion_digest,
            "e2_probe_manifest_digest": verified_e2.manifest_digest,
            "e2_workspace_plan_identity": e2_workspace_plan_identity,
            "source_m2_sha256": sha256_path(m2),
            "selected_probe_sha256": selected_probe_sha,
            "packaged_probe_sha256": sha256_path(stage / "risk-probe"),
            "packaged_m2_sha256": sha256_path(stage / "m2-direction"),
            "feature_schema_digest": schema.digest,
            "feature_layer": schema.layers[0],
            "feature_site": schema.sites[0].value,
            "intervention_layer": intervention_layer,
            "intervention_site": ActivationSite.BLOCK_OUTPUT.value,
            "token_scope": token_scope.value,
            "direction_sha256": direction.direction_sha256,
            "direction_norm": direction.direction_norm,
            "reference_rms": direction.reference_rms,
            "release_risk_threshold": 0.4,
            "abstention_probability_threshold": 0.7,
            "alpha_max": 1.0,
            "alpha_beta": 8.0,
            "alpha_risk_threshold": 0.4,
            "vector_count": 1,
        }
        (stage / "manifest.json").write_text(
            json.dumps({**body, "manifest_digest": stable_hash(body)}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        verify_e4_act_baseline(stage)
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return verify_e4_act_baseline(destination)


def build_e4_act_baseline(
    directory: str | Path,
    *,
    e2_probe_bundle: str | Path,
    e2_workspace: str | Path,
    e2_phase_run: str | Path,
    m2_caa_artifact: str | Path,
    intervention_layer: int,
    study: StudyProtocol,
    token_scope: TokenScope = TokenScope.FIRST_FOUR,
) -> E4ActBaseline:
    """Replay completed E2 evidence, then freeze the ACT/SADI-style baseline."""

    workspace = verify_e2_workspace(e2_workspace)
    verified_e2 = verify_e2_probe_bundle(e2_probe_bundle, workspace=workspace)
    if (
        not workspace.protocol.scientific_eligible
        or not verified_e2.scientific_eligible
        or not verified_e2.gate_passed
    ):
        raise DataValidationError("E4 ACT requires a passed scientific E2 probe bundle")
    ledger = PhaseRunLedger.open(e2_phase_run, study=study)
    completion = ledger.verify_complete()
    e2_sha = sha256_path(e2_probe_bundle)
    if (
        completion.phase is not ExperimentPhase.E2
        or ledger.contract.input_fingerprints.get("activation_feature_schemas")
        != e2_sha
    ):
        raise FrozenArtifactError("E4 ACT source differs from the completed E2 phase")
    return _package_e4_act_baseline(
        directory,
        e2_probe_bundle=e2_probe_bundle,
        m2_caa_artifact=m2_caa_artifact,
        intervention_layer=intervention_layer,
        verified_e2=verified_e2,
        e2_completion_digest=completion.completion_digest,
        e2_workspace_plan_identity=workspace.plan_identity,
        token_scope=token_scope,
    )


def verify_e4_act_baseline(directory: str | Path) -> E4ActBaseline:
    source = Path(directory)
    if (
        source.is_symlink()
        or not source.is_dir()
        or {item.name for item in source.iterdir()} != _INVENTORY
        or any(item.is_symlink() for item in source.rglob("*"))
    ):
        raise FrozenArtifactError("E4 ACT baseline inventory differs")
    manifest = _read_object(source / "manifest.json", "E4 ACT baseline manifest")
    body = dict(manifest)
    digest = body.pop("manifest_digest", None)
    expected_keys = {
        "schema_version",
        "method",
        "design",
        "source_e2_sha256",
        "source_e2_completion_digest",
        "e2_probe_manifest_digest",
        "e2_workspace_plan_identity",
        "source_m2_sha256",
        "selected_probe_sha256",
        "packaged_probe_sha256",
        "packaged_m2_sha256",
        "feature_schema_digest",
        "feature_layer",
        "feature_site",
        "intervention_layer",
        "intervention_site",
        "token_scope",
        "direction_sha256",
        "direction_norm",
        "reference_rms",
        "release_risk_threshold",
        "abstention_probability_threshold",
        "alpha_max",
        "alpha_beta",
        "alpha_risk_threshold",
        "vector_count",
    }
    if (
        set(body) != expected_keys
        or digest != stable_hash(body)
        or body.get("schema_version") != 1
        or body.get("method") != "ACT-or-SADI"
        or body.get("design") != "calibrated-E2-risk-gated-single-M2-CAA-direction"
        or body.get("packaged_probe_sha256") != sha256_path(source / "risk-probe")
        or body.get("packaged_m2_sha256") != sha256_path(source / "m2-direction")
        or body.get("intervention_site") != ActivationSite.BLOCK_OUTPUT.value
        or body.get("token_scope") != TokenScope.FIRST_FOUR.value
        or body.get("release_risk_threshold") != 0.4
        or body.get("abstention_probability_threshold") != 0.7
        or body.get("alpha_max") != 1.0
        or body.get("alpha_beta") != 8.0
        or body.get("alpha_risk_threshold") != 0.4
        or body.get("vector_count") != 1
        or not all(
            isinstance(body.get(name), str)
            and len(str(body[name])) == 64
            and all(character in "0123456789abcdef" for character in str(body[name]))
            for name in (
                "source_e2_sha256",
                "source_e2_completion_digest",
                "e2_probe_manifest_digest",
                "e2_workspace_plan_identity",
                "source_m2_sha256",
            )
        )
    ):
        raise FrozenArtifactError("E4 ACT baseline manifest differs")
    probe = load_calibrated_probe(source / "risk-probe")
    schema = probe.training_schema
    direction = resolve_static_direction(
        source / "m2-direction",
        method="M2",
        layer=int(body["intervention_layer"]),
        site=ActivationSite.BLOCK_OUTPUT,
    )
    numeric = (body.get("direction_norm"), body.get("reference_rms"))
    if (
        probe.task is not ProbeTask.CORRECT_INCORRECT_ABSTENTION
        or probe.state.labels != ("C", "I", "A")
        or schema.digest != body.get("feature_schema_digest")
        or schema.layers != (body.get("feature_layer"),)
        or [value.value for value in schema.sites] != [body.get("feature_site")]
        or schema.width != 5_120
        or direction.direction.numel() != 5_120
        or direction.direction_sha256 != body.get("direction_sha256")
        or any(isinstance(value, bool) or not isinstance(value, int | float) for value in numeric)
        or not torch.isclose(
            torch.tensor(direction.direction_norm),
            torch.tensor(float(body["direction_norm"])),
            rtol=0,
            atol=1e-7,
        )
        or direction.reference_rms != float(body["reference_rms"])
    ):
        raise FrozenArtifactError("E4 ACT baseline components differ")
    return E4ActBaseline(
        directory=source.absolute(),
        risk_probe=probe,
        direction=direction,
        feature_layer=int(body["feature_layer"]),
        feature_site=ActivationSite(str(body["feature_site"])),
        intervention_layer=int(body["intervention_layer"]),
        intervention_site=ActivationSite(str(body["intervention_site"])),
        token_scope=TokenScope(str(body["token_scope"])),
        source_e2_sha256=str(body["source_e2_sha256"]),
        source_e2_completion_digest=str(body["source_e2_completion_digest"]),
        e2_probe_manifest_digest=str(body["e2_probe_manifest_digest"]),
        e2_workspace_plan_identity=str(body["e2_workspace_plan_identity"]),
        source_m2_sha256=str(body["source_m2_sha256"]),
        manifest_digest=str(digest),
    )

"""Frozen near-zero-hallucination policy assembled as M6."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor

from mfh.artifact_namespace import validate_active_study_artifact_paths
from mfh.contracts import Outcome, TokenScope
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.inference.architecture import HookKey
from mfh.inference.hooks import InterventionPlan
from mfh.methods.adaptive import (
    AdaptiveController,
    load_adaptive_controller,
    save_adaptive_controller,
)
from mfh.methods.features import ActivationKind
from mfh.methods.probes import (
    CalibratedProbe,
    ProbeTask,
    load_calibrated_probe,
    save_calibrated_probe,
)
from mfh.methods.protected import (
    ProtectedSubspace,
    load_protected_subspace,
    save_protected_subspace,
)
from mfh.provenance import sha256_path, stable_hash

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class RiskRegime(StrEnum):
    KNOWN = "known"
    POTENTIALLY_RECOVERABLE = "potentially_recoverable"
    LIKELY_UNKNOWN = "likely_unknown"


class OutputAction(StrEnum):
    RELEASE = "release"
    ABSTAIN = "abstain"
    ESCALATE = "escalate"


@dataclass(frozen=True, slots=True)
class CompositePolicyConfig:
    tau_low: float
    tau_high: float
    release_epsilon: float
    token_scope: TokenScope = TokenScope.FIRST_FOUR
    abstention_phrase: str = "I don't know."
    closed_book: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.tau_low < self.tau_high <= 1:
            raise DataValidationError("M6 thresholds must satisfy 0 <= low < high <= 1")
        if not 0 <= self.release_epsilon <= self.tau_low:
            raise DataValidationError("release epsilon must be no larger than tau_low")
        if not self.abstention_phrase.strip():
            raise DataValidationError("M6 abstention phrase must be non-empty")


@dataclass(frozen=True, slots=True)
class PromptAssessment:
    class_probabilities: Mapping[str, float]
    incorrect_probability: float
    regime: RiskRegime
    alpha: float
    interventions: Mapping[HookKey, InterventionPlan]
    should_abstain: bool

    def __post_init__(self) -> None:
        probabilities = {str(key): float(value) for key, value in self.class_probabilities.items()}
        if not probabilities or any(
            not math.isfinite(value) or value < 0 or value > 1 for value in probabilities.values()
        ):
            raise DataValidationError("M6 class probabilities are invalid")
        if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-5):
            raise DataValidationError("M6 class probabilities must sum to one")
        if not 0 <= self.incorrect_probability <= 1 or not math.isfinite(self.alpha):
            raise DataValidationError("M6 risk or alpha is invalid")
        if self.alpha < 0:
            raise DataValidationError("M6 alpha cannot be negative")
        if self.regime is not RiskRegime.POTENTIALLY_RECOVERABLE and self.interventions:
            raise DataValidationError("M6 may intervene only in the recoverable regime")
        if self.regime is RiskRegime.LIKELY_UNKNOWN and not self.should_abstain:
            raise DataValidationError("likely-unknown assessments must abstain")
        object.__setattr__(self, "class_probabilities", MappingProxyType(probabilities))
        object.__setattr__(self, "interventions", MappingProxyType(dict(self.interventions)))


@dataclass(frozen=True, slots=True)
class EarlyReevaluation:
    residual_risk: float
    continue_generation: bool
    reason: str
    gold_likelihood_improved: bool | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.residual_risk <= 1 or not self.reason.strip():
            raise DataValidationError("invalid early-generation re-evaluation")


@dataclass(frozen=True, slots=True)
class OutputGateDecision:
    action: OutputAction
    residual_risk: float
    reason: str


@dataclass(frozen=True, slots=True)
class CompositePolicy:
    controller: AdaptiveController
    config: CompositePolicyConfig
    early_probe: CalibratedProbe | None = None
    protected_subspace: ProtectedSubspace | None = None

    def __post_init__(self) -> None:
        if self.early_probe is None:
            raise DataValidationError("M6 requires a calibrated early-token C/I/A probe")
        if (
            self.early_probe.task is not ProbeTask.CORRECT_INCORRECT_ABSTENTION
            or self.early_probe.state.labels
            != (Outcome.CORRECT.value, Outcome.INCORRECT.value, Outcome.ABSTENTION.value)
        ):
            raise DataValidationError("M6 early probe must estimate calibrated P(C),P(I),P(A)")
        if self.early_probe.training_schema.activation_kind not in {
            ActivationKind.FIRST_GENERATED,
            ActivationKind.FIRST_FOUR_GENERATED,
            ActivationKind.FIRST_EIGHT_GENERATED,
        }:
            raise DataValidationError("M6 early probe must use generated-token features")
        if (
            self.early_probe.training_schema.source_identity()
            != self.controller.risk_probe.training_schema.source_identity()
        ):
            raise DataValidationError(
                "M6 prompt and early probes use different model/prompt sources"
            )
        if self.protected_subspace is not None:
            directions = self.controller.vector_bank.directions.values()
            widths = {int(value.shape[1]) for value in directions}
            if widths != {self.protected_subspace.width}:
                raise DataValidationError(
                    "M6 protected subspace width differs from its routed vector bank"
                )

    def _regime(self, incorrect_probability: float) -> RiskRegime:
        if incorrect_probability < self.config.tau_low:
            return RiskRegime.KNOWN
        if incorrect_probability < self.config.tau_high:
            return RiskRegime.POTENTIALLY_RECOVERABLE
        return RiskRegime.LIKELY_UNKNOWN

    def assess(self, features: Tensor) -> tuple[PromptAssessment, ...]:
        decision = self.controller.decide(features)
        incorrect_index = decision.class_labels.index(Outcome.INCORRECT.value)
        results: list[PromptAssessment] = []
        for row in range(decision.probabilities.shape[0]):
            risk = float(decision.probabilities[row, incorrect_index])
            regime = self._regime(risk)
            recoverable = regime is RiskRegime.POTENTIALLY_RECOVERABLE
            interventions = {}
            if recoverable:
                raw_plans = decision.plans_for_row(
                    row,
                    token_scope=self.config.token_scope,
                )
                interventions = {
                    key: InterventionPlan(
                        direction=(
                            self.protected_subspace.project(
                                plan.direction,
                                normalize=True,
                            )
                            if self.protected_subspace is not None
                            else plan.direction
                        ),
                        alpha=plan.alpha,
                        token_scope=plan.token_scope,
                        rms_relative=plan.rms_relative,
                        decay=plan.decay,
                    )
                    for key, plan in raw_plans.items()
                }
            results.append(
                PromptAssessment(
                    class_probabilities={
                        label: float(decision.probabilities[row, index])
                        for index, label in enumerate(decision.class_labels)
                    },
                    incorrect_probability=risk,
                    regime=regime,
                    alpha=float(decision.alphas[row]) if recoverable else 0.0,
                    interventions=interventions,
                    should_abstain=regime is RiskRegime.LIKELY_UNKNOWN,
                )
            )
        return tuple(results)

    def reevaluate_after_early_tokens(
        self,
        features: Tensor,
        *,
        safety_ok: bool,
        language_ok: bool,
        refusal_drift: bool,
        gold_log_likelihood_delta: float | None = None,
    ) -> EarlyReevaluation:
        assert self.early_probe is not None
        probe = self.early_probe
        probabilities = probe.predict_probabilities(features)
        if probabilities.shape[0] != 1:
            raise DataValidationError("early-generation re-evaluation accepts exactly one row")
        try:
            incorrect_index = probe.state.labels.index(Outcome.INCORRECT.value)
        except ValueError as exc:
            raise DataValidationError("early risk probe must estimate P(I)") from exc
        risk = float(probabilities[0, incorrect_index])
        gold_improved: bool | None = None
        if gold_log_likelihood_delta is not None:
            if not math.isfinite(gold_log_likelihood_delta):
                raise DataValidationError("gold likelihood delta must be finite")
            gold_improved = gold_log_likelihood_delta > 0
        if not safety_ok:
            return EarlyReevaluation(risk, False, "safety constraint violated", gold_improved)
        if not language_ok:
            return EarlyReevaluation(
                risk, False, "requested-language drift detected", gold_improved
            )
        if refusal_drift:
            return EarlyReevaluation(
                risk, False, "unintended refusal drift detected", gold_improved
            )
        if risk >= self.config.tau_high:
            return EarlyReevaluation(risk, False, "residual risk remains too high", gold_improved)
        return EarlyReevaluation(risk, True, "early-generation checks passed", gold_improved)

    def output_gate(
        self,
        residual_risk: float,
        *,
        safety_ok: bool,
        language_ok: bool,
        refusal_drift: bool,
    ) -> OutputGateDecision:
        if not math.isfinite(residual_risk) or not 0 <= residual_risk <= 1:
            raise DataValidationError("output-gate risk must be in [0, 1]")
        failures: list[str] = []
        if residual_risk > self.config.release_epsilon:
            failures.append("risk exceeds release epsilon")
        if not safety_ok:
            failures.append("safety constraint violated")
        if not language_ok:
            failures.append("language constraint violated")
        if refusal_drift:
            failures.append("unintended refusal drift detected")
        if not failures:
            return OutputGateDecision(
                OutputAction.RELEASE, residual_risk, "all release checks passed"
            )
        action = OutputAction.ABSTAIN if self.config.closed_book else OutputAction.ESCALATE
        return OutputGateDecision(action, residual_risk, "; ".join(failures))


def minimum_alpha_for_risk(
    candidates: Sequence[tuple[float, float]], *, risk_epsilon: float
) -> float | None:
    """Select the smallest measured alpha reaching the preregistered risk target."""

    if not 0 <= risk_epsilon <= 1:
        raise DataValidationError("risk epsilon must be in [0, 1]")
    validated: list[tuple[float, float]] = []
    for alpha, risk in candidates:
        if not math.isfinite(alpha) or alpha < 0 or not math.isfinite(risk) or not 0 <= risk <= 1:
            raise DataValidationError("alpha-risk candidates are invalid")
        validated.append((alpha, risk))
    eligible = [alpha for alpha, risk in validated if risk <= risk_epsilon]
    return min(eligible) if eligible else None


@dataclass(frozen=True, slots=True)
class CompositeManifest:
    prompt_id: str
    method: str
    policy: CompositePolicyConfig
    component_paths: Mapping[str, str]
    component_digests: Mapping[str, str]
    data_fingerprints: Mapping[str, str]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1 or not self.prompt_id.strip() or not self.method.strip():
            raise DataValidationError("invalid composite manifest identity or schema")
        components = {str(key): str(value) for key, value in self.component_digests.items()}
        paths = {str(key): str(value) for key, value in self.component_paths.items()}
        fingerprints = {str(key): str(value) for key, value in self.data_fingerprints.items()}
        if not components or not fingerprints or set(paths) != set(components):
            raise DataValidationError("composite manifest must freeze components and data")
        if any(
            not key.strip() or not _SHA256.fullmatch(value) for key, value in components.items()
        ):
            raise DataValidationError("composite component digests must be SHA-256 values")
        if any(
            not key.strip() or not _SHA256.fullmatch(value) for key, value in fingerprints.items()
        ):
            raise DataValidationError("composite data fingerprints must be SHA-256 values")
        if any(
            not key.strip()
            or Path(value).is_absolute()
            or ".." in Path(value).parts
            or value in {"", "."}
            for key, value in paths.items()
        ):
            raise DataValidationError("composite component paths must be safe relative paths")
        object.__setattr__(self, "component_paths", MappingProxyType(paths))
        object.__setattr__(self, "component_digests", MappingProxyType(components))
        object.__setattr__(self, "data_fingerprints", MappingProxyType(fingerprints))

    def body(self) -> dict[str, object]:
        policy = asdict(self.policy)
        policy["token_scope"] = self.policy.token_scope.value
        return {
            "schema_version": self.schema_version,
            "prompt_id": self.prompt_id,
            "method": self.method,
            "policy": policy,
            "component_paths": dict(self.component_paths),
            "component_digests": dict(self.component_digests),
            "data_fingerprints": dict(self.data_fingerprints),
        }


def save_composite_manifest(path: str | Path, manifest: CompositeManifest) -> None:
    destination = validate_active_study_artifact_paths(
        {"composite-manifest": path}
    )["composite-manifest"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FrozenArtifactError(f"refusing to overwrite composite manifest: {destination}")
    component_sources = validate_active_study_artifact_paths(
        {
            f"composite-component-{name}": destination.parent / relative
            for name, relative in manifest.component_paths.items()
        }
    )
    for name in manifest.component_paths:
        if (
            sha256_path(component_sources[f"composite-component-{name}"])
            != manifest.component_digests[name]
        ):
            raise FrozenArtifactError(f"composite component digest mismatch: {name}")
    body = manifest.body()
    value = {**body, "manifest_digest": stable_hash(body)}
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise FrozenArtifactError(f"composite manifest already exists: {destination}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def load_composite_manifest(path: str | Path) -> CompositeManifest:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read composite manifest: {exc}") from exc
    digest = value.pop("manifest_digest", None)
    if digest != stable_hash(value):
        raise FrozenArtifactError("composite manifest digest mismatch")
    if value.get("schema_version") != 1:
        raise FrozenArtifactError("unsupported composite-manifest schema version")
    try:
        policy_data = dict(value["policy"])
        policy_data["token_scope"] = TokenScope(policy_data["token_scope"])
        manifest = CompositeManifest(
            prompt_id=str(value["prompt_id"]),
            method=str(value["method"]),
            policy=CompositePolicyConfig(**policy_data),
            component_paths=value["component_paths"],
            component_digests=value["component_digests"],
            data_fingerprints=value["data_fingerprints"],
        )
        for name, relative in manifest.component_paths.items():
            if sha256_path(source.parent / relative) != manifest.component_digests[name]:
                raise FrozenArtifactError(f"composite component digest mismatch: {name}")
        return manifest
    except (KeyError, TypeError, ValueError, DataValidationError) as exc:
        raise FrozenArtifactError(f"invalid composite manifest: {exc}") from exc


def save_composite_policy(
    directory: str | Path,
    policy: CompositePolicy,
    *,
    sae_checkpoint: str | Path | None = None,
    protected_source_artifact: str | Path | None = None,
    selection_provenance: Mapping[str, Any] | None = None,
    early_probe_selection: str | Path | None = None,
) -> None:
    """Freeze and bundle a reconstructable M6 policy, not only digest labels."""

    e10_sources = (sae_checkpoint, protected_source_artifact, selection_provenance)
    e10_bundle = all(value is not None for value in e10_sources)
    if any(value is not None for value in e10_sources) and not e10_bundle:
        raise DataValidationError(
            "E10 composite sources must freeze SAE, protected artifact, and provenance together"
        )
    if early_probe_selection is not None and not e10_bundle:
        raise DataValidationError(
            "E10 early-probe selection requires all composite promotion sources"
        )
    if e10_bundle and policy.protected_subspace is None:
        raise DataValidationError("E10 composite sources require protected steering")
    if e10_bundle:
        active_paths: dict[str, str | Path] = {
            "E10 composite policy": directory,
            "E10 SAE checkpoint": sae_checkpoint,  # type: ignore[dict-item]
            "E10 protected source": protected_source_artifact,  # type: ignore[dict-item]
        }
        if early_probe_selection is not None:
            active_paths["E10 early-probe selection"] = early_probe_selection
        normalized = validate_active_study_artifact_paths(active_paths)
        directory = normalized["E10 composite policy"]
        sae_checkpoint = normalized["E10 SAE checkpoint"]
        protected_source_artifact = normalized["E10 protected source"]
        if early_probe_selection is not None:
            early_probe_selection = normalized["E10 early-probe selection"]
    destination = Path(directory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FrozenArtifactError(f"refusing to overwrite composite policy: {destination}")
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        assert policy.early_probe is not None
        save_adaptive_controller(stage / "controller", policy.controller)
        if early_probe_selection is None:
            save_calibrated_probe(stage / "early_probe", policy.early_probe)
        else:
            from mfh.experiments.e10_early_probe import (
                load_e10_early_probe_selection,
            )

            early_selection = load_e10_early_probe_selection(early_probe_selection)
            shutil.copytree(
                early_selection.selected_probe_path,
                stage / "early_probe",
            )
            if sha256_path(stage / "early_probe") != sha256_path(
                early_selection.selected_probe_path
            ):
                raise DataValidationError(
                    "E10 early probe differs from its selected artifact"
                )
        config = asdict(policy.config)
        config["token_scope"] = policy.config.token_scope.value
        if policy.protected_subspace is not None:
            save_protected_subspace(
                stage / "protected_subspace",
                policy.protected_subspace,
            )
        provenance: dict[str, Any] | None = None
        if e10_bundle:
            assert sae_checkpoint is not None
            assert protected_source_artifact is not None
            assert selection_provenance is not None
            for source, target in (
                (Path(sae_checkpoint).resolve(), stage / "sae_checkpoint"),
                (
                    Path(protected_source_artifact).resolve(),
                    stage / "protected_source_artifact",
                ),
                *(
                    (
                        (
                            Path(early_probe_selection).resolve(),
                            stage / "early_probe_selection",
                        ),
                    )
                    if early_probe_selection is not None
                    else ()
                ),
            ):
                if source.is_symlink() or not source.exists() or (
                    source.is_dir()
                    and any(item.is_symlink() for item in source.rglob("*"))
                ):
                    raise DataValidationError("E10 composite source is missing or linked")
                if source.is_dir():
                    shutil.copytree(source, target, symlinks=False)
                elif source.is_file():
                    shutil.copyfile(source, target)
                else:
                    raise DataValidationError("E10 composite source type is unsupported")
            from mfh.experiments.e8_protected import load_e8_protected_artifact

            protected = load_e8_protected_artifact(stage / "protected_source_artifact")
            assert policy.protected_subspace is not None
            if (
                protected.protected_subspace.behaviors
                != policy.protected_subspace.behaviors
                or protected.protected_subspace.data_fingerprint
                != policy.protected_subspace.data_fingerprint
                or protected.protected_subspace.feature_schema
                != policy.protected_subspace.feature_schema
                or not torch.equal(
                    protected.protected_subspace.basis,
                    policy.protected_subspace.basis,
                )
            ):
                raise DataValidationError(
                    "E10 protected subspace differs from its selected source artifact"
                )
            try:
                replayed = json.loads(
                    json.dumps(
                        dict(selection_provenance),
                        sort_keys=True,
                        allow_nan=False,
                    )
                )
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise DataValidationError(
                    f"E10 selection provenance is not exact JSON: {exc}"
                ) from exc
            if replayed != dict(selection_provenance):
                raise DataValidationError("E10 selection provenance is not stable JSON")
            provenance = replayed
        metadata_body = {
            "schema_version": (
                4
                if early_probe_selection is not None
                else 3
                if e10_bundle
                else 2
                if policy.protected_subspace is not None
                else 1
            ),
            "policy": config,
            "source_identity": policy.controller.risk_probe.training_schema.source_identity(),
            "components": {
                "controller": sha256_path(stage / "controller"),
                "early_probe": sha256_path(stage / "early_probe"),
                **(
                    {
                        "protected_subspace": sha256_path(
                            stage / "protected_subspace"
                        )
                    }
                    if policy.protected_subspace is not None
                    else {}
                ),
                **(
                    {
                        "sae_checkpoint": sha256_path(stage / "sae_checkpoint"),
                        "protected_source_artifact": sha256_path(
                            stage / "protected_source_artifact"
                        ),
                        **(
                            {
                                "early_probe_selection": sha256_path(
                                    stage / "early_probe_selection"
                                )
                            }
                            if early_probe_selection is not None
                            else {}
                        ),
                    }
                    if e10_bundle
                    else {}
                ),
            },
            **({"selection_provenance": provenance} if provenance is not None else {}),
        }
        metadata = {**metadata_body, "metadata_digest": stable_hash(metadata_body)}
        (stage / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def load_composite_policy(directory: str | Path) -> CompositePolicy:
    source = Path(directory)
    try:
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read composite-policy metadata: {exc}") from exc
    digest = metadata.pop("metadata_digest", None)
    if digest != stable_hash(metadata):
        raise FrozenArtifactError("composite-policy metadata digest mismatch")
    if metadata.get("schema_version") not in {1, 2, 3, 4}:
        raise FrozenArtifactError("unsupported composite-policy schema version")
    try:
        components = dict(metadata["components"])
        expected_components = {"controller", "early_probe"}
        if metadata["schema_version"] in {2, 3, 4}:
            expected_components.add("protected_subspace")
        if metadata["schema_version"] == 3:
            expected_components |= {"sae_checkpoint", "protected_source_artifact"}
        if metadata["schema_version"] == 4:
            expected_components |= {
                "sae_checkpoint",
                "protected_source_artifact",
                "early_probe_selection",
            }
        if set(components) != expected_components:
            raise FrozenArtifactError("composite-policy component set is invalid")
        for name, expected in components.items():
            if not _SHA256.fullmatch(str(expected)) or sha256_path(source / name) != expected:
                raise FrozenArtifactError(f"composite-policy component changed: {name}")
        config_data = dict(metadata["policy"])
        config_data["token_scope"] = TokenScope(config_data["token_scope"])
        policy = CompositePolicy(
            controller=load_adaptive_controller(source / "controller"),
            config=CompositePolicyConfig(**config_data),
            early_probe=load_calibrated_probe(source / "early_probe"),
            protected_subspace=(
                load_protected_subspace(source / "protected_subspace")
                if metadata["schema_version"] in {2, 3, 4}
                else None
            ),
        )
        if (
            policy.controller.risk_probe.training_schema.source_identity()
            != metadata["source_identity"]
        ):
            raise FrozenArtifactError("composite-policy source identity mismatch")
        if metadata["schema_version"] in {3, 4}:
            provenance = metadata.get("selection_provenance")
            if not isinstance(provenance, dict):
                raise FrozenArtifactError("E10 composite selection provenance is invalid")
            from mfh.experiments.e8_protected import load_e8_protected_artifact

            protected = load_e8_protected_artifact(
                source / "protected_source_artifact"
            )
            assert policy.protected_subspace is not None
            if (
                protected.protected_subspace.behaviors
                != policy.protected_subspace.behaviors
                or protected.protected_subspace.data_fingerprint
                != policy.protected_subspace.data_fingerprint
                or protected.protected_subspace.feature_schema
                != policy.protected_subspace.feature_schema
                or not torch.equal(
                    protected.protected_subspace.basis,
                    policy.protected_subspace.basis,
                )
            ):
                raise FrozenArtifactError(
                    "E10 protected subspace differs from its source artifact"
                )
            if metadata["schema_version"] == 4:
                from mfh.experiments.e10_early_probe import (
                    load_e10_early_probe_selection,
                )

                selection = load_e10_early_probe_selection(
                    source / "early_probe_selection"
                )
                if sha256_path(source / "early_probe") != sha256_path(
                    selection.selected_probe_path
                ):
                    raise FrozenArtifactError(
                        "E10 early probe differs from its frozen selection"
                    )
        return policy
    except (KeyError, TypeError, ValueError, DataValidationError) as exc:
        if isinstance(exc, FrozenArtifactError):
            raise
        raise FrozenArtifactError(f"invalid composite-policy artifact: {exc}") from exc


def load_e10_composite_provenance(directory: str | Path) -> Mapping[str, Any]:
    """Load schema-3 M6 source provenance after recursively verifying the policy."""

    source = Path(directory)
    load_composite_policy(source)
    try:
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - loaded above
        raise FrozenArtifactError(f"cannot read E10 composite provenance: {exc}") from exc
    provenance = metadata.get("selection_provenance")
    if metadata.get("schema_version") not in {3, 4} or not isinstance(provenance, dict):
        raise FrozenArtifactError("E10 requires a promoted composite policy")
    return MappingProxyType(dict(provenance))

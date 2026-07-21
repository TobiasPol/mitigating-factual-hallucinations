"""Machine-checked E0--E10 protocol from ``docs/research-plan.md``.

The phase file is intentionally more rigid than a generic workflow format.
Changing a confirmatory matrix, a target-benchmark tuning rule, or the E10
freeze boundary requires a schema/code change instead of a silent YAML edit.
"""

from __future__ import annotations

import os
import pwd
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from mfh.config import load_yaml
from mfh.errors import ConfigurationError, DataValidationError
from mfh.experiments.model_selection import E0_MODELS, PRIMARY_RESEARCH_MODELS
from mfh.provenance import stable_hash


class ExperimentPhase(StrEnum):
    E0 = "E0"
    E1 = "E1"
    E2 = "E2"
    E3 = "E3"
    E4 = "E4"
    E5 = "E5"
    E6 = "E6"
    E7 = "E7"
    E8 = "E8"
    E9 = "E9"
    E10 = "E10"

    @property
    def ordinal(self) -> int:
        return int(self.value[1:])


class PhaseMode(StrEnum):
    VALIDATION = "validation"
    DEVELOPMENT = "development"
    CONFIRMATORY = "confirmatory"


_PHASE_KEYS = {
    "phase",
    "title",
    "mode",
    "purpose",
    "prerequisites",
    "models",
    "benchmarks",
    "partitions",
    "prompts",
    "methods",
    "required_inputs",
    "outputs",
    "gates",
    "tuning_allowed",
    "one_shot",
    "factorial",
    "question_limit",
    "freeze_fields",
}

_PRIMARY_MODELS = PRIMARY_RESEARCH_MODELS
_FACTUAL_BENCHMARKS = {
    "triviaqa",
    "simpleqa_verified",
    "aa_omniscience_public_600",
}
_PRIMARY_PROMPTS = {"P0-neutral", "P1-direct", "P2-calibrated-abstention"}
_E9_METHODS = {"M0", "M1", "M2", "M3", "M4", "M5"}
_E10_FREEZE_FIELDS = {
    "model_revision",
    "prompt",
    "risk_threshold",
    "vector_bank",
    "sae_checkpoint",
    "protected_subspace",
    "layer",
    "alpha_policy",
    "abstention_rule",
    "grader",
    "evaluation_scripts",
}


def _strings(value: Any, context: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ConfigurationError(f"{context} must be a list")
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result) or len(set(result)) != len(result):
        raise ConfigurationError(f"{context} must contain unique non-empty strings")
    return result


def _boolean(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{context} must be a boolean")
    return value


def _question_limit(value: Any, index: int) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigurationError(f"phases[{index}].question_limit must be an integer or null")
    return value


@dataclass(frozen=True, slots=True)
class PhaseProtocol:
    phase: ExperimentPhase
    title: str
    mode: PhaseMode
    purpose: str
    prerequisites: tuple[ExperimentPhase, ...]
    models: tuple[str, ...]
    benchmarks: tuple[str, ...]
    partitions: tuple[str, ...]
    prompts: tuple[str, ...]
    methods: tuple[str, ...]
    required_inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    gates: tuple[str, ...]
    tuning_allowed: bool
    one_shot: bool
    factorial: bool
    question_limit: int | None
    freeze_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("title", "purpose"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ConfigurationError(f"{self.phase.value} {name} must be non-empty")
            object.__setattr__(self, name, value)
        if any(item.ordinal >= self.phase.ordinal for item in self.prerequisites):
            raise ConfigurationError(
                f"{self.phase.value} prerequisites must refer only to earlier phases"
            )
        if self.phase is ExperimentPhase.E0 and self.prerequisites:
            raise ConfigurationError("E0 cannot have prerequisites")
        if self.phase is not ExperimentPhase.E0 and not self.prerequisites:
            raise ConfigurationError(f"{self.phase.value} must declare prerequisites")
        if not self.outputs:
            raise ConfigurationError(f"{self.phase.value} must declare outputs")
        if self.question_limit is not None and self.question_limit <= 0:
            raise ConfigurationError("phase question_limit must be positive")
        if self.one_shot and self.tuning_allowed:
            raise ConfigurationError("one-shot phases cannot allow tuning")
        if self.mode is PhaseMode.CONFIRMATORY and self.tuning_allowed:
            raise ConfigurationError("confirmatory phases cannot allow tuning")
        if self.freeze_fields and self.phase is not ExperimentPhase.E10:
            raise ConfigurationError("only E10 may declare final freeze fields")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["phase"] = self.phase.value
        value["mode"] = self.mode.value
        value["prerequisites"] = [item.value for item in self.prerequisites]
        return value


@dataclass(frozen=True, slots=True)
class StudyProtocol:
    study_id: str
    phases: tuple[PhaseProtocol, ...]
    schema_version: int = 1
    source_path: Path | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        study_id = self.study_id.strip()
        if self.schema_version != 1 or not study_id:
            raise ConfigurationError("study protocol requires schema version 1 and an ID")
        expected = tuple(ExperimentPhase)
        observed = tuple(item.phase for item in self.phases)
        if observed != expected:
            raise ConfigurationError(
                "study phases must contain E0 through E10 exactly once and in order"
            )
        self._validate_research_plan_contract()
        object.__setattr__(self, "study_id", study_id)
        if self.source_path is not None:
            source_path = Path(self.source_path).resolve()
            if (
                source_path.name != "phases.yaml"
                or source_path.parent.name != "experiments"
                or source_path.parent.parent.name != "configs"
            ):
                raise ConfigurationError(
                    "study protocol must be loaded from configs/experiments/phases.yaml"
                )
            object.__setattr__(self, "source_path", source_path)

    def _validate_research_plan_contract(self) -> None:
        phases = {item.phase: item for item in self.phases}
        e0 = phases[ExperimentPhase.E0]
        if set(e0.models) != E0_MODELS:
            raise ConfigurationError("E0 must validate the sole approved Qwen MLX model")
        if e0.question_limit != 500:
            raise ConfigurationError("E0 must use 500 shared benign factual prompts")

        e1 = phases[ExperimentPhase.E1]
        if (
            set(e1.models) != _PRIMARY_MODELS
            or set(e1.benchmarks) != _FACTUAL_BENCHMARKS
            or set(e1.prompts) != _PRIMARY_PROMPTS
            or set(e1.methods) != {"M0"}
            or not e1.factorial
        ):
            raise ConfigurationError("E1 must be the exact 1 x 3 x 3 prompt-only factorial")

        e2 = phases[ExperimentPhase.E2]
        if "probe_beats_confidence_baselines" not in e2.gates:
            raise ConfigurationError("E2 must enforce the activation-separability gate")
        if set(e2.benchmarks) - {"triviaqa", "simpleqa_verified", "aa_omniscience_public_600"}:
            raise ConfigurationError("E2 may use only the three factual benchmarks")

        e3 = phases[ExperimentPhase.E3]
        if "M0" not in e3.methods:
            raise ConfigurationError("E3 requires an unsteered M0 causal baseline")

        e4 = phases[ExperimentPhase.E4]
        if e4.question_limit != 2_000:
            raise ConfigurationError("E4 external baselines must screen on 2,000 T-dev rows")

        e5 = phases[ExperimentPhase.E5]
        if (
            set(e5.benchmarks) != {"triviaqa"}
            or not {"M1", "M3"} <= set(e5.methods)
            or set(e5.partitions)
            - {
                "T-steer",
                "T-controller-train",
                "T-controller-calibration",
                "T-dev",
            }
        ):
            raise ConfigurationError("E5 training and selection must remain TriviaQA-only")

        e7 = phases[ExperimentPhase.E7]
        if (
            "M0" not in e7.methods
            or not {
                "xstest",
                "strongreject_or_harmbench",
                "language_consistency",
            }
            <= set(e7.benchmarks)
            or not {
                "frozen_sae_seed_runs",
                "frozen_side_effect_scorers",
            }
            <= set(e7.required_inputs)
        ):
            raise ConfigurationError("E7 causal audits require M0 and protected side suites")

        e8 = phases[ExperimentPhase.E8]
        if "M0" not in e8.methods or "frozen_side_effect_scorers" not in e8.required_inputs:
            raise ConfigurationError("E8 requires an unsteered M0 non-inferiority baseline")

        e9 = phases[ExperimentPhase.E9]
        if (
            e9.mode is not PhaseMode.CONFIRMATORY
            or set(e9.models) != _PRIMARY_MODELS
            or set(e9.benchmarks) != _FACTUAL_BENCHMARKS
            or set(e9.prompts) != _PRIMARY_PROMPTS
            or set(e9.methods) != _E9_METHODS
            or not e9.factorial
            or "frozen_question_bundle" not in e9.required_inputs
        ):
            raise ConfigurationError("E9 must be the frozen 1 x 3 x 3 x 6 factorial")

        e10 = phases[ExperimentPhase.E10]
        if (
            e10.mode is not PhaseMode.CONFIRMATORY
            or not e10.one_shot
            or set(e10.methods) != {"M6"}
            or set(e10.freeze_fields) != _E10_FREEZE_FIELDS
            or not set(e10.benchmarks) >= _FACTUAL_BENCHMARKS
            or "frozen_question_bundle" not in e10.required_inputs
        ):
            raise ConfigurationError("E10 does not satisfy the frozen one-shot M6 contract")

        for phase in self.phases:
            if set(phase.models) != _PRIMARY_MODELS:
                raise ConfigurationError(
                    f"{phase.phase.value} must use the sole approved Qwen MLX model"
                )
            if phase.phase.ordinal >= ExperimentPhase.E2.ordinal and phase.phase.ordinal <= 8:
                target_partitions = {"simpleqa-train", "aa-train", "simpleqa-dev", "aa-dev"}
                if target_partitions & set(phase.partitions):
                    raise ConfigurationError(
                        f"{phase.phase.value} cannot tune on SimpleQA or AA partitions"
                    )

    @property
    def digest(self) -> str:
        return stable_hash(
            {
                "schema_version": self.schema_version,
                "study_id": self.study_id,
                "phases": [phase.to_dict() for phase in self.phases],
            }
        )

    @property
    def one_shot_registry_path(self) -> Path:
        """Return the protocol-owned E10 registry; callers cannot choose another path."""

        account_home = Path(pwd.getpwuid(os.getuid()).pw_dir).absolute()
        return account_home / ".local" / "state" / "mfh" / "one-shot"

    def phase(self, phase: ExperimentPhase | str) -> PhaseProtocol:
        selected = ExperimentPhase(phase)
        return self.phases[selected.ordinal]

    def assert_ready(
        self,
        phase: ExperimentPhase | str,
        completed: Mapping[ExperimentPhase | str, str],
    ) -> None:
        selected = self.phase(phase)
        normalized: dict[ExperimentPhase, str] = {}
        for key, digest in completed.items():
            parsed = ExperimentPhase(key)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise DataValidationError("completed phase digests must be SHA-256 strings")
            normalized[parsed] = digest
        missing = set(selected.prerequisites) - set(normalized)
        if missing:
            raise DataValidationError(
                f"{selected.phase.value} prerequisites are incomplete: "
                f"{sorted(item.value for item in missing)}"
            )

    def assert_frozen_inputs(
        self, phase: ExperimentPhase | str, input_fingerprints: Mapping[str, str]
    ) -> None:
        selected = self.phase(phase)
        required = set(selected.required_inputs) | set(selected.freeze_fields)
        missing = required - set(input_fingerprints)
        if missing:
            raise DataValidationError(
                f"{selected.phase.value} is missing frozen inputs: {sorted(missing)}"
            )
        malformed = {
            key: value
            for key, value in input_fingerprints.items()
            if not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        }
        if malformed:
            raise DataValidationError(
                f"phase input fingerprints must be lowercase SHA-256: {sorted(malformed)}"
            )

    @property
    def by_phase(self) -> Mapping[ExperimentPhase, PhaseProtocol]:
        return MappingProxyType({item.phase: item for item in self.phases})


def _parse_phase(value: Any, index: int) -> PhaseProtocol:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"phases[{index}] must be a mapping")
    unknown = set(value) - _PHASE_KEYS
    missing = _PHASE_KEYS - set(value)
    if unknown or missing:
        raise ConfigurationError(
            f"phases[{index}] keys differ; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    try:
        question_limit_value = value["question_limit"]
        return PhaseProtocol(
            phase=ExperimentPhase(str(value["phase"])),
            title=str(value["title"]),
            mode=PhaseMode(str(value["mode"])),
            purpose=str(value["purpose"]),
            prerequisites=tuple(
                ExperimentPhase(item)
                for item in _strings(value["prerequisites"], f"phases[{index}].prerequisites")
            ),
            models=_strings(value["models"], f"phases[{index}].models"),
            benchmarks=_strings(value["benchmarks"], f"phases[{index}].benchmarks"),
            partitions=_strings(value["partitions"], f"phases[{index}].partitions"),
            prompts=_strings(value["prompts"], f"phases[{index}].prompts"),
            methods=_strings(value["methods"], f"phases[{index}].methods"),
            required_inputs=_strings(value["required_inputs"], f"phases[{index}].required_inputs"),
            outputs=_strings(value["outputs"], f"phases[{index}].outputs"),
            gates=_strings(value["gates"], f"phases[{index}].gates"),
            tuning_allowed=_boolean(value["tuning_allowed"], f"phases[{index}].tuning_allowed"),
            one_shot=_boolean(value["one_shot"], f"phases[{index}].one_shot"),
            factorial=_boolean(value["factorial"], f"phases[{index}].factorial"),
            question_limit=_question_limit(question_limit_value, index),
            freeze_fields=_strings(value["freeze_fields"], f"phases[{index}].freeze_fields"),
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ConfigurationError):
            raise
        raise ConfigurationError(f"invalid phase at index {index}: {exc}") from exc


def _load_study_protocol(
    path: str | Path,
    *,
    retain_source_path: bool,
) -> StudyProtocol:
    source_path = Path(path).resolve()
    raw = load_yaml(source_path)
    expected = {"schema_version", "study_id", "phases"}
    if set(raw) != expected or raw.get("schema_version") != 1:
        raise ConfigurationError(
            f"{path}: study protocol must contain only schema_version, study_id, and phases"
        )
    values = raw.get("phases")
    if not isinstance(values, list):
        raise ConfigurationError(f"{path}: phases must be a list")
    return StudyProtocol(
        study_id=str(raw["study_id"]),
        phases=tuple(_parse_phase(value, index) for index, value in enumerate(values)),
        source_path=source_path if retain_source_path else None,
    )


def load_study_protocol(path: str | Path) -> StudyProtocol:
    """Load the live canonical study protocol from its required repository path."""

    return _load_study_protocol(path, retain_source_path=True)


def load_packaged_study_protocol(path: str | Path) -> StudyProtocol:
    """Load immutable packaged protocol bytes without claiming a live source path."""

    return _load_study_protocol(path, retain_source_path=False)

"""Frozen prompt-paraphrase and RQ1 generalization execution schedules."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from mfh.artifact_namespace import validate_active_study_artifact_paths
from mfh.config import load_prompt_specs
from mfh.contracts import AdaptivePolicySpec, PromptSpec, Question, Runtime
from mfh.data.io import read_questions
from mfh.data.reviewed_splits import validate_reviewed_split_snapshot
from mfh.data.splits import semantic_group_ids
from mfh.errors import ConfigurationError, DataValidationError, FrozenArtifactError
from mfh.experiments.confirmatory_graders import (
    ConfirmatoryGraderBundle,
    validate_confirmatory_grader_bundle,
)
from mfh.experiments.e2_schedule import controller_feature_partitions
from mfh.experiments.protocol import (
    ExperimentPhase,
    load_packaged_study_protocol,
)
from mfh.experiments.runner import PhaseRunLedger
from mfh.experiments.snapshots import validate_execution_snapshot
from mfh.provenance import canonical_json, sha256_file, sha256_path, stable_hash

APPROVED_ROBUSTNESS_CONFIG_DIGEST = (
    "06da1e24b298361df6cd6cdad7753813001f8b09b8f12598d2b87f8a7d6a69fd"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BASE_PROMPTS = ("P0-neutral", "P2-calibrated-abstention")
_BENCHMARKS = (
    "triviaqa",
    "simpleqa_verified",
    "aa_omniscience_public_600",
)
_EVALUATION_PARTITIONS = {
    "triviaqa": "T-test",
    "simpleqa_verified": "simpleqa-eval",
    "aa_omniscience_public_600": "aa-eval",
}
_EVALUATION_SOURCE_COUNTS = {
    "triviaqa": 5_000,
    "simpleqa_verified": 1_000,
    "aa_omniscience_public_600": 600,
}
_RQ1_SOURCE_COUNTS = {
    "T-steer": 30_000,
    "T-controller": 5_000,
    "T-dev": 5_000,
}
_METHODS = ("M0", "M1", "M2", "M3", "M4", "M5")
_RQ1_METHODS = ("M1", "M3")
_APPROVED_PRIMARY_PROMPTS_SHA256 = (
    "6b2ded197678a12d5e597b9e17639699ef47f8faad983ee391e9b72c21d6ae34"
)
_SOURCE_BINDINGS = frozenset(
    {
        "canonical-prompts",
        "frozen-component-selection",
        "frozen-evaluation-scripts",
        "frozen-graders",
        "e1-phase-ledger",
        "triviaqa-evaluation",
        "simpleqa_verified-evaluation",
        "aa_omniscience_public_600-evaluation",
        "triviaqa-development",
    }
)


def _question_fingerprint(question: Question) -> str:
    return stable_hash(
        {
            "question_id": question.question_id,
            "benchmark": question.benchmark,
            "text": question.text,
            "aliases": list(question.aliases),
            "split": question.split,
            "entities": list(question.entities),
            "metadata": dict(question.metadata),
        }
    )


def _prompt_fingerprint(prompt: PromptSpec) -> str:
    return stable_hash(
        {
            "prompt_id": prompt.prompt_id,
            "text": prompt.text,
            "permits_abstention": prompt.permits_abstention,
            "deployment_eligible": prompt.deployment_eligible,
        }
    )


def _load_object(path: Path, context: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FrozenArtifactError(f"{context} must be one regular JSON file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read {context}: {exc}") from exc
    if not isinstance(value, dict):
        raise FrozenArtifactError(f"{context} must contain one JSON object")
    return value


def load_robustness_diagnostic_config(path: str | Path) -> Mapping[str, Any]:
    """Load the sole approved, pre-E9 diagnostic configuration."""

    value = _load_object(Path(path), "robustness diagnostic config")
    if (
        set(value)
        != {
            "schema_version",
            "diagnostic_id",
            "freeze_boundary",
            "reviewed_split_binding",
            "prompt_paraphrase",
            "rq1_generalization",
            "config_digest",
        }
        or value.get("schema_version") != 1
    ):
        raise ConfigurationError("robustness diagnostic config schema differs")
    digest = value.get("config_digest")
    body = dict(value)
    body.pop("config_digest")
    if (
        digest != stable_hash(body)
        or digest != APPROVED_ROBUSTNESS_CONFIG_DIGEST
        or value.get("freeze_boundary")
        != "freeze-before-E9-and-never-use-for-E9-or-E10-component-selection"
    ):
        raise ConfigurationError("robustness diagnostic config is not approved")
    prompt = value.get("prompt_paraphrase")
    rq1 = value.get("rq1_generalization")
    split_binding = value.get("reviewed_split_binding")
    if (
        not isinstance(prompt, dict)
        or not isinstance(rq1, dict)
        or not isinstance(split_binding, dict)
        or split_binding
        != {
            "manifest_digest": ("05e13f0193155551400fd636e8dd6d97e065dd80205133a9440ef13105bce148"),
            "artifact_sha256": ("3ceaf111654b80e34abd568853f64bba894fc7c6d7a81950c2868f3584a187f4"),
            "required_e1_state": (
                "complete-qwen3.6-27b-mlx-4bit-E1-ledger-deduplicated-splits-input"
            ),
        }
    ):
        raise ConfigurationError("robustness diagnostic sections are invalid")
    variants = prompt.get("variants")
    if (
        prompt.get("questions_per_benchmark") != 200
        or prompt.get("generation_seed") != 17
        or tuple(prompt.get("methods", ())) != _METHODS
        or not isinstance(prompt.get("benchmarks"), dict)
        or tuple(prompt["benchmarks"]) != _BENCHMARKS
        or prompt["benchmarks"] != _EVALUATION_PARTITIONS
        or not isinstance(variants, list)
        or len(variants) != 10
        or len({row.get("prompt_id") for row in variants if isinstance(row, dict)}) != 10
    ):
        raise ConfigurationError("prompt-paraphrase schedule differs from the plan")
    counts = Counter(row.get("base_prompt_id") for row in variants if isinstance(row, dict))
    if counts != Counter({name: 5 for name in _BASE_PROMPTS}) or any(
        not isinstance(row, dict)
        or set(row) != {"prompt_id", "base_prompt_id", "text"}
        or not isinstance(row["prompt_id"], str)
        or not isinstance(row["text"], str)
        or not row["text"].strip()
        or (
            row["base_prompt_id"] == "P2-calibrated-abstention"
            and ("I don't know" not in row["text"] or "guess" not in row["text"].casefold())
        )
        for row in variants
    ):
        raise ConfigurationError("prompt paraphrases are not five exact P0/P2 variants")
    regimes = rq1.get("adaptation_regimes")
    if (
        rq1.get("benchmark") != "triviaqa"
        or rq1.get("semantic_fold_count") != 10
        or rq1.get("training_prompt") != "P0-neutral"
        or tuple(rq1.get("evaluation_prompts", ())) != _BASE_PROMPTS
        or tuple(rq1.get("methods", ())) != _RQ1_METHODS
        or tuple(rq1.get("reviewed_source_partitions", ())) != ("T-steer", "T-controller", "T-dev")
        or rq1.get("controller_subdivision_algorithm") != "exact-E2-semantic-group-subdivision-v1"
        or rq1.get("controller_calibration_rows") != 1_000
        or rq1.get("controller_subdivision_seed") != 17
        or not isinstance(regimes, dict)
        or tuple(regimes)
        != ("source-frozen-control", "calibration-only", "full-vector-bank-relearning")
        or regimes["source-frozen-control"]
        != {
            "methods": ["M1"],
            "fit_on_held_out_calibration": [],
            "frozen_on_source_folds": ["execution_component", "all_policy_fields"],
        }
        or regimes["calibration-only"]
        != {
            "methods": ["M3"],
            "fit_on_held_out_calibration": [
                "risk_threshold",
                "abstention_threshold",
            ],
            "frozen_on_source_folds": [
                "vector_bank",
                "router",
                "directions",
                "risk_probe",
                "layer_selector",
                "alpha_controller",
                "router_architecture",
                "candidate_layers",
                "candidate_sites",
                "token_scopes",
                "sparsity",
                "alpha_policy_family",
                "likely_unknown_threshold",
                "execution_public_key",
            ],
        }
        or regimes["full-vector-bank-relearning"]
        != {
            "methods": ["M3"],
            "fit_on_held_out_calibration": [
                "vector_bank",
                "router",
                "risk_threshold",
                "abstention_threshold",
            ],
            "frozen_on_source_folds": [
                "risk_probe",
                "layer_selector",
                "router_architecture",
                "candidate_layers",
                "candidate_sites",
                "token_scopes",
                "sparsity",
                "alpha_controller",
                "alpha_policy_family",
                "likely_unknown_threshold",
                "execution_public_key",
            ],
        }
        or rq1.get("threshold_calibration_algorithm")
        != "balanced-accuracy-over-observed-score-cutpoints-tie-smallest-threshold-v1"
        or rq1.get("m3_refit_hyperparameters")
        != {
            "vector_seed": 17,
            "minimum_class_count": 1,
            "router_seed": 17,
            "router_hidden_width": 64,
            "router_epochs": 300,
            "distance_temperature": 1.0,
            "risk_hidden_width": 64,
            "risk_epochs": 400,
            "risk_learning_rate": 0.03,
            "risk_weight_decay": 0.0001,
            "risk_class_balanced": True,
            "risk_seed": 17,
            "calibration_kind": "temperature",
            "layer_seed": 17,
            "layer_epochs": 300,
        }
        or rq1.get("full_relearning_subdivision_algorithm")
        != "semantic-fold-preserve-preregistered-partitions-v1"
        or rq1.get("held_out_evaluation_partition") != "T-dev"
        or "no-SimpleQA-AA-E9-or-E10" not in str(rq1.get("target_benchmark_tuning_policy"))
    ):
        raise ConfigurationError("RQ1 generalization schedule differs from the plan")
    return MappingProxyType(value)


@dataclass(frozen=True, slots=True)
class PromptParaphraseTask:
    task_id: str
    benchmark: str
    partition: str
    question_id: str
    question_fingerprint: str
    base_prompt_id: str
    prompt_id: str
    prompt_text: str
    method: str

    def __post_init__(self) -> None:
        body = {
            "benchmark": self.benchmark,
            "partition": self.partition,
            "question_id": self.question_id,
            "question_fingerprint": self.question_fingerprint,
            "base_prompt_id": self.base_prompt_id,
            "prompt_id": self.prompt_id,
            "prompt_text": self.prompt_text,
            "method": self.method,
        }
        if (
            self.task_id != _task_id("prompt", body)
            or self.benchmark not in _BENCHMARKS
            or self.partition != _EVALUATION_PARTITIONS[self.benchmark]
            or not self.question_id.strip()
            or _SHA256.fullmatch(self.question_fingerprint) is None
            or self.base_prompt_id not in _BASE_PROMPTS
            or not self.prompt_id.strip()
            or not self.prompt_text.strip()
            or self.method not in _METHODS
        ):
            raise FrozenArtifactError("prompt-paraphrase task identity is invalid")


@dataclass(frozen=True, slots=True)
class RQ1GeneralizationTask:
    task_id: str
    held_out_fold: int
    training_prompt_id: str
    evaluation_prompt_id: str
    method: str
    adaptation_regime: str

    def __post_init__(self) -> None:
        body = {
            "held_out_fold": self.held_out_fold,
            "training_prompt_id": self.training_prompt_id,
            "evaluation_prompt_id": self.evaluation_prompt_id,
            "method": self.method,
            "adaptation_regime": self.adaptation_regime,
        }
        if (
            self.task_id != _task_id("rq1", body)
            or type(self.held_out_fold) is not int
            or not 0 <= self.held_out_fold < 10
            or self.training_prompt_id != "P0-neutral"
            or self.evaluation_prompt_id not in _BASE_PROMPTS
            or self.method not in _RQ1_METHODS
            or (
                self.method == "M1"
                and self.adaptation_regime != "source-frozen-control"
            )
            or (
                self.method == "M3"
                and self.adaptation_regime
                not in {"calibration-only", "full-vector-bank-relearning"}
            )
        ):
            raise FrozenArtifactError("RQ1 task identity is invalid")


@dataclass(frozen=True, slots=True)
class RobustnessDiagnosticPlan:
    path: Path | None
    body: Mapping[str, Any]
    plan_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "body", _deep_freeze_json(dict(self.body)))


def _deep_freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(name): _deep_freeze_json(item) for name, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_deep_freeze_json(item) for item in value)
    return value


def _deep_thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(name): _deep_thaw_json(item) for name, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw_json(item) for item in value]
    return value


def _task_id(kind: str, body: Mapping[str, Any]) -> str:
    return f"{kind}-{stable_hash(dict(body))}"


def _selected_questions(
    questions: Sequence[Question],
    *,
    benchmark: str,
    seed: int,
    count: int,
) -> tuple[Question, ...]:
    values = tuple(questions)
    identities = {_question_fingerprint(value) for value in values}
    if (
        len(values) < count
        or len(identities) != len(values)
        or len({value.question_id for value in values}) != len(values)
        or len(values) != _EVALUATION_SOURCE_COUNTS[benchmark]
        or any(
            value.benchmark != benchmark or value.split != _EVALUATION_PARTITIONS[benchmark]
            for value in values
        )
    ):
        raise DataValidationError(f"{benchmark} robustness question source is invalid")
    return tuple(
        sorted(
            values,
            key=lambda value: (
                hashlib.sha256(
                    f"{seed}:{benchmark}:{_question_fingerprint(value)}".encode()
                ).digest(),
                value.question_id,
            ),
        )[:count]
    )


def _strict_source_path(path: str | Path, label: str) -> Path:
    raw = Path(path)
    if ".." in raw.parts:
        raise DataValidationError(f"{label} cannot contain parent traversal")
    lexical = Path(os.path.abspath(raw))
    if any(candidate.is_symlink() for candidate in (lexical, *lexical.parents)):
        raise DataValidationError(f"{label} cannot traverse a symlink")
    resolved = lexical.resolve(strict=False)
    if (
        not resolved.exists()
        or (not resolved.is_file() and not resolved.is_dir())
        or (resolved.is_dir() and any(item.is_symlink() for item in resolved.rglob("*")))
    ):
        raise DataValidationError(f"{label} must be one strict regular artifact")
    return resolved


def _questions_from_source(name: str, path: Path) -> tuple[Question, ...]:
    if path.is_file():
        return tuple(read_questions(path))
    candidates = {
        "triviaqa-evaluation": ("T-test.jsonl",),
        "simpleqa_verified-evaluation": ("simpleqa-eval.jsonl",),
        "aa_omniscience_public_600-evaluation": ("aa-eval.jsonl",),
    }
    if name == "triviaqa-development":
        files = tuple(path / f"{partition}.jsonl" for partition in _RQ1_SOURCE_COUNTS)
        if any(not candidate.is_file() or candidate.is_symlink() for candidate in files):
            raise DataValidationError(
                "TriviaQA development source lacks exact reviewed split files"
            )
        return tuple(question for file in files for question in read_questions(file))
    try:
        names = candidates[name]
    except KeyError as exc:
        raise DataValidationError(f"{name} is not a question source") from exc
    existing = tuple(path / filename for filename in names if (path / filename).is_file())
    if len(existing) != 1 or existing[0].is_symlink():
        raise DataValidationError(f"{name} does not resolve to one exact JSONL source")
    return tuple(read_questions(existing[0]))


def _validate_robustness_component_selection(
    path: Path,
    canonical_prompts: Mapping[str, PromptSpec],
) -> str:
    from mfh.experiments.confirmatory_components import (
        load_confirmatory_adaptive_component,
        load_confirmatory_fixed_component,
    )

    if (
        path.is_symlink()
        or not path.is_dir()
        or {item.name for item in path.iterdir()} != {"manifest.json", "components"}
        or any(item.is_symlink() for item in path.rglob("*"))
    ):
        raise DataValidationError("robustness component selection inventory differs")
    manifest = _load_object(path / "manifest.json", "robustness component selection")
    digest = manifest.pop("manifest_digest", None)
    components = manifest.get("components")
    if (
        set(manifest) != {"schema_version", "study_protocol_digest", "phase", "components"}
        or manifest.get("schema_version") != 3
        or manifest.get("phase") != ExperimentPhase.E9.value
        or not isinstance(manifest.get("study_protocol_digest"), str)
        or _SHA256.fullmatch(manifest["study_protocol_digest"]) is None
        or digest != stable_hash(manifest)
        or not isinstance(components, list)
    ):
        raise DataValidationError("robustness component selection manifest differs")
    observed: set[str] = set()
    expected_methods = {"M1", "M2", "M3", "M4", "M5"}
    expected_directories: set[str] = set()
    for descriptor in components:
        if (
            not isinstance(descriptor, dict)
            or set(descriptor)
            != {
                "model_name",
                "method",
                "artifact_sha256",
                "component_path",
                "adaptive_policy",
                "adaptive_policy_digest",
            }
            or descriptor["model_name"] != "qwen3.6-27b-mlx-4bit"
            or descriptor["method"] not in expected_methods
            or not isinstance(descriptor["artifact_sha256"], str)
            or _SHA256.fullmatch(descriptor["artifact_sha256"]) is None
        ):
            raise DataValidationError("robustness component descriptor differs")
        method = descriptor["method"]
        expected_relative = (
            "components/"
            + stable_hash({"model_name": descriptor["model_name"], "method": method})[:16]
        )
        if descriptor["component_path"] != expected_relative:
            raise DataValidationError("robustness component path is not canonical")
        expected_directories.add(Path(expected_relative).name)
        component = path / expected_relative
        policy = descriptor["adaptive_policy"]
        if method == "M3":
            if (
                not isinstance(policy, dict)
                or AdaptivePolicySpec.from_dict(policy).to_dict() != policy
                or descriptor["adaptive_policy_digest"] != stable_hash(policy)
                or {item.name for item in component.iterdir()}
                != {"artifact", "adaptive-policy.json"}
            ):
                raise DataValidationError("robustness M3 adaptive policy differs")
            loaded_adaptive = load_confirmatory_adaptive_component(component / "artifact")
            if (
                loaded_adaptive.fingerprint != descriptor["artifact_sha256"]
                or loaded_adaptive.model_name != "qwen3.6-27b-mlx-4bit"
                or loaded_adaptive.model_repository != "mlx-community/Qwen3.6-27B-4bit"
                or loaded_adaptive.model_revision != "c000ac2c2057d94be3fa931000c31723aac53282"
                or loaded_adaptive.runtime is not Runtime.MLX
                or loaded_adaptive.quantization != "affine-g64-mlx-4bit"
                or loaded_adaptive.model_num_layers != 64
                or not set(_BASE_PROMPTS) <= set(loaded_adaptive.controllers)
                or not set(_BASE_PROMPTS) <= set(loaded_adaptive.controller_source_prompt_ids)
                or any(
                    loaded_adaptive.prompt_hashes[prompt_id]
                    != hashlib.sha256(canonical_prompts[prompt_id].text.encode("utf-8")).hexdigest()
                    for prompt_id in _BASE_PROMPTS
                )
                or loaded_adaptive.controller_source_prompt_ids["P0-neutral"] != "P0-neutral"
            ):
                raise DataValidationError("robustness M3 controller identity differs")
        else:
            if (
                policy is not None
                or descriptor["adaptive_policy_digest"] is not None
                or {item.name for item in component.iterdir()} != {"artifact"}
            ):
                raise DataValidationError("robustness fixed component policy differs")
            loaded_fixed = load_confirmatory_fixed_component(component / "artifact")
            if (
                loaded_fixed.method != method
                or loaded_fixed.fingerprint != descriptor["artifact_sha256"]
            ):
                raise DataValidationError("robustness fixed component identity differs")
        if sha256_path(component / "artifact") != descriptor["artifact_sha256"]:
            raise DataValidationError("robustness component bytes changed")
        observed.add(method)
    if (
        observed != expected_methods
        or {item.name for item in (path / "components").iterdir()} != expected_directories
    ):
        raise DataValidationError("robustness component method inventory differs")
    return str(manifest["study_protocol_digest"])


def _validate_e1_reviewed_split_binding(
    *,
    config: Mapping[str, Any],
    e1_run: Path,
    evaluation_snapshot: Path,
    snapshot_manifest: Mapping[str, Any],
    reviewed_paths: Sequence[Path],
    grader_bundle: ConfirmatoryGraderBundle,
    completion_execution_private_key: str | None = None,
) -> Mapping[str, Any]:
    """Bind every diagnostic question to the completed Qwen E1 split input."""

    split_binding = config.get("reviewed_split_binding")
    if not isinstance(split_binding, Mapping):  # pragma: no cover - config validates
        raise ConfigurationError("robustness config lacks reviewed-split binding")
    files = snapshot_manifest.get("files")
    descriptor = files.get("study_protocol_config") if isinstance(files, Mapping) else None
    if not isinstance(descriptor, Mapping) or not isinstance(descriptor.get("path"), str):
        raise DataValidationError("E9 snapshot lacks the packaged study protocol config")
    study = load_packaged_study_protocol(evaluation_snapshot / str(descriptor["path"]))
    if study.digest != snapshot_manifest.get("study_protocol_digest"):
        raise DataValidationError("E1 binding and E9 snapshot use different studies")
    expected_partitions = {
        "triviaqa": "T-controller",
        "simpleqa_verified": "simpleqa-eval",
        "aa_omniscience_public_600": "aa-eval",
    }
    expected_prompts = {"P0-neutral", "P1-direct", "P2-calibrated-abstention"}
    expected_cells = {
        (benchmark, partition, prompt, "M0", 17)
        for benchmark, partition in expected_partitions.items()
        for prompt in expected_prompts
    }
    expected_cell_inventory = [
        {
            "benchmark": benchmark,
            "partition": partition,
            "system_prompt_id": prompt,
            "steering_method": method,
            "seed": seed,
        }
        for benchmark, partition, prompt, method, seed in sorted(expected_cells)
    ]
    authoritative_questions = {
        benchmark: tuple(read_questions(reviewed_paths[0] / f"{partition}.jsonl"))
        for benchmark, partition in expected_partitions.items()
    }
    expected_question_ids = {
        benchmark: tuple(question.question_id for question in questions)
        for benchmark, questions in authoritative_questions.items()
    }
    if e1_run.is_file():
        portable = _load_object(e1_run, "portable Qwen E1 binding")
        digest = portable.pop("binding_digest", None)
        expected_fields = {
            "schema_version",
            "kind",
            "study_protocol_digest",
            "e1_completion_digest",
            "e1_contract_digest",
            "e1_record_set_digest",
            "e1_record_count",
            "e1_ledger_sha256",
            "e1_condition_cells",
            "e1_conditions_sha256",
            "e1_question_ids_sha256",
            "e1_input_fingerprints",
            "e1_prerequisite_digests",
            "reviewed_split_manifest_digest",
            "reviewed_split_sha256",
            "completion_execution_public_key",
            "completion_signature",
        }
        question_digests = portable.get("e1_question_ids_sha256")
        input_fingerprints = portable.get("e1_input_fingerprints")
        prerequisites = portable.get("e1_prerequisite_digests")
        if (
            set(portable) != expected_fields
            or portable.get("schema_version") != 2
            or portable.get("kind") != "signed-portable-complete-qwen-e1-binding"
            or portable.get("study_protocol_digest") != study.digest
            or portable.get("e1_record_count") != 19_800
            or portable.get("e1_condition_cells") != expected_cell_inventory
            or portable.get("reviewed_split_manifest_digest")
            != split_binding["manifest_digest"]
            or portable.get("reviewed_split_sha256") != split_binding["artifact_sha256"]
            or not isinstance(question_digests, dict)
            or set(question_digests)
            != {"triviaqa", "simpleqa_verified", "aa_omniscience_public_600"}
            or question_digests
            != {
                benchmark: stable_hash(list(question_ids))
                for benchmark, question_ids in sorted(expected_question_ids.items())
            }
            or not isinstance(input_fingerprints, dict)
            or set(input_fingerprints)
            != {"deduplicated_splits", "grader_bundle", "inference_protocol"}
            or any(
                not isinstance(value, str) or _SHA256.fullmatch(value) is None
                for value in input_fingerprints.values()
            )
            or not isinstance(prerequisites, dict)
            or set(prerequisites) != {"E0"}
            or not isinstance(prerequisites.get("E0"), str)
            or _SHA256.fullmatch(str(prerequisites["E0"])) is None
            or any(
                not isinstance(portable.get(name), str)
                or _SHA256.fullmatch(str(portable[name])) is None
                for name in (
                    "study_protocol_digest",
                    "e1_completion_digest",
                    "e1_contract_digest",
                    "e1_record_set_digest",
                    "e1_ledger_sha256",
                    "e1_conditions_sha256",
                    "reviewed_split_manifest_digest",
                    "reviewed_split_sha256",
                    "completion_execution_public_key",
                )
            )
            or portable.get("completion_execution_public_key")
            != grader_bundle.scorer.execution_public_key
            or not isinstance(portable.get("completion_signature"), str)
            or re.fullmatch(r"[0-9a-f]{128}", str(portable["completion_signature"]))
            is None
            or not _verify_portable_e1_completion_signature(portable)
            or digest != stable_hash(portable)
            or sha256_file(e1_run) != _portable_e1_binding_sha256(portable)
            or any(sha256_path(path) != split_binding["artifact_sha256"] for path in reviewed_paths)
        ):
            raise DataValidationError("portable Qwen E1 binding differs from the frozen study")
        return MappingProxyType({**portable, "e1_binding_sha256": sha256_file(e1_run)})
    ledger = PhaseRunLedger.open(e1_run, study=study)
    completion = ledger.verify_complete()
    exact_identity = all(
        condition.model_name == "qwen3.6-27b-mlx-4bit"
        and condition.model_repository == "mlx-community/Qwen3.6-27B-4bit"
        and condition.model_revision == "c000ac2c2057d94be3fa931000c31723aac53282"
        and condition.runtime is Runtime.MLX
        and condition.quantization == "affine-g64-mlx-4bit"
        and condition.model_num_layers == 64
        for condition in ledger.contract.conditions
    )
    if (
        completion.phase is not ExperimentPhase.E1
        or ledger.contract.phase is not ExperimentPhase.E1
        or ledger.contract.study_protocol_digest != study.digest
        or ledger.contract.expected_record_count != 19_800
        or not exact_identity
    ):
        raise DataValidationError("robustness requires the exact complete Qwen E1 ledger")
    evidence = _load_object(e1_run / "creation-evidence.json", "Qwen E1 evidence")
    inputs = evidence.get("input_artifacts")
    authorizations = evidence.get("scientific_input_authorizations")
    input_descriptor = inputs.get("deduplicated_splits") if isinstance(inputs, Mapping) else None
    authorization = (
        authorizations.get("deduplicated_splits") if isinstance(authorizations, Mapping) else None
    )
    if (
        not isinstance(input_descriptor, Mapping)
        or set(input_descriptor) != {"location", "fingerprint"}
        or not isinstance(authorization, Mapping)
        or set(authorization)
        != {
            "kind",
            "manifest_digest",
            "review_result_manifest_digest",
            "fingerprint",
        }
        or authorization.get("kind") != "human-reviewed-contamination-controlled-triviaqa-splits"
    ):
        raise DataValidationError("complete Qwen E1 lacks reviewed-split authorization")
    location = input_descriptor.get("location")
    if not isinstance(location, str):
        raise DataValidationError("Qwen E1 reviewed-split location is invalid")
    authoritative = Path(location)
    if not authoritative.is_absolute():
        authoritative = (e1_run / authoritative).resolve()
    authoritative = _strict_source_path(authoritative, "Qwen E1 reviewed split")
    split_manifest = validate_reviewed_split_snapshot(authoritative)
    split_binding = config.get("reviewed_split_binding")
    if not isinstance(split_binding, Mapping):  # pragma: no cover - config validates
        raise ConfigurationError("robustness config lacks reviewed-split binding")
    split_sha = sha256_path(authoritative)
    observed_cells = {
        (
            condition.benchmark,
            condition.partition,
            condition.system_prompt_id,
            condition.steering_method,
            condition.seed,
        )
        for condition in ledger.contract.conditions
    }
    if (
        input_descriptor.get("fingerprint") != split_sha
        or authorization.get("fingerprint") != split_sha
        or ledger.contract.input_fingerprints.get("deduplicated_splits") != split_sha
        or authorization.get("manifest_digest") != split_manifest.get("manifest_digest")
        or split_manifest.get("manifest_digest") != split_binding["manifest_digest"]
        or split_sha != split_binding["artifact_sha256"]
        or any(sha256_path(path) != split_sha for path in reviewed_paths)
        or dict(ledger.contract.question_ids_by_benchmark) != expected_question_ids
        or observed_cells != expected_cells
        or len(ledger.contract.conditions) != 9
        or set(ledger.contract.input_fingerprints)
        != {"deduplicated_splits", "grader_bundle", "inference_protocol"}
        or set(ledger.contract.prerequisite_digests) != {"E0"}
    ):
        raise DataValidationError(
            "robustness questions differ from the exact E1-bound reviewed split"
        )
    if completion_execution_private_key is None:
        raise DataValidationError(
            "live Qwen E1 portability requires the frozen execution private key"
        )
    try:
        completion_private_key = Ed25519PrivateKey.from_private_bytes(
            bytes.fromhex(completion_execution_private_key)
        )
        completion_public_key = completion_private_key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        ).hex()
    except ValueError as exc:
        raise DataValidationError("Qwen E1 completion execution key is invalid") from exc
    if completion_public_key != grader_bundle.scorer.execution_public_key:
        raise DataValidationError(
            "Qwen E1 completion key differs from the frozen execution key"
        )
    binding_core = {
        "schema_version": 2,
        "kind": "signed-portable-complete-qwen-e1-binding",
        "study_protocol_digest": study.digest,
        "e1_completion_digest": completion.completion_digest,
        "e1_contract_digest": ledger.contract.digest,
        "e1_record_set_digest": completion.record_set_digest,
        "e1_record_count": completion.record_count,
        "e1_ledger_sha256": sha256_path(e1_run),
        "e1_condition_cells": expected_cell_inventory,
        "e1_conditions_sha256": stable_hash(
            [condition.to_dict() for condition in ledger.contract.conditions]
        ),
        "e1_question_ids_sha256": {
            benchmark: stable_hash(list(question_ids))
            for benchmark, question_ids in sorted(expected_question_ids.items())
        },
        "e1_input_fingerprints": dict(ledger.contract.input_fingerprints),
        "e1_prerequisite_digests": dict(ledger.contract.prerequisite_digests),
        "reviewed_split_manifest_digest": str(split_manifest["manifest_digest"]),
        "reviewed_split_sha256": split_sha,
        "completion_execution_public_key": completion_public_key,
    }
    completion_signature = completion_private_key.sign(
        canonical_json(binding_core).encode("utf-8")
    ).hex()
    binding_body = {**binding_core, "completion_signature": completion_signature}
    return MappingProxyType(
        {**binding_body, "e1_binding_sha256": _portable_e1_binding_sha256(binding_body)}
    )


def _portable_e1_binding_sha256(body: Mapping[str, Any]) -> str:
    payload = canonical_json({**dict(body), "binding_digest": stable_hash(dict(body))}) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _verify_portable_e1_completion_signature(body: Mapping[str, Any]) -> bool:
    signature = body.get("completion_signature")
    public_key = body.get("completion_execution_public_key")
    if (
        not isinstance(signature, str)
        or re.fullmatch(r"[0-9a-f]{128}", signature) is None
        or not isinstance(public_key, str)
        or _SHA256.fullmatch(public_key) is None
    ):
        return False
    signed = dict(body)
    signed.pop("completion_signature", None)
    signed.pop("e1_binding_sha256", None)
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key)).verify(
            bytes.fromhex(signature), canonical_json(signed).encode("utf-8")
        )
    except (InvalidSignature, ValueError):
        return False
    return True


def _load_bound_plan_inputs(
    config: Mapping[str, Any],
    source_artifacts: Mapping[str, str | Path],
    *,
    completion_execution_private_key: str | None = None,
) -> tuple[
    Mapping[str, PromptSpec],
    Mapping[str, tuple[Question, ...]],
    tuple[Question, ...],
    Mapping[str, Path],
    Mapping[str, Any],
    Mapping[str, str],
    str,
]:
    if set(source_artifacts) != _SOURCE_BINDINGS:
        raise DataValidationError("robustness source artifact set differs")
    paths = {
        name: _strict_source_path(path, f"robustness source {name}")
        for name, path in source_artifacts.items()
    }
    reviewed_paths = tuple(
        paths[name]
        for name in (
            "triviaqa-evaluation",
            "simpleqa_verified-evaluation",
            "aa_omniscience_public_600-evaluation",
            "triviaqa-development",
        )
    )
    if len({sha256_path(path) for path in reviewed_paths}) != 1:
        raise DataValidationError(
            "all robustness factual questions must use one exact reviewed split bundle"
        )
    for reviewed in reviewed_paths:
        validate_reviewed_split_snapshot(reviewed)
    prompt_source = paths["canonical-prompts"]
    if (
        not prompt_source.is_file()
        or sha256_file(prompt_source) != _APPROVED_PRIMARY_PROMPTS_SHA256
    ):
        raise DataValidationError("canonical prompt source is not the approved primary config")
    loaded_prompts = {
        value.prompt_id: value
        for value in load_prompt_specs(prompt_source)
        if value.prompt_id in _BASE_PROMPTS
    }
    if set(loaded_prompts) != set(_BASE_PROMPTS):
        raise DataValidationError("canonical prompt source lacks exact P0/P2")
    evaluations = {
        benchmark: _questions_from_source(source_name, paths[source_name])
        for benchmark, source_name in (
            ("triviaqa", "triviaqa-evaluation"),
            ("simpleqa_verified", "simpleqa_verified-evaluation"),
            (
                "aa_omniscience_public_600",
                "aa_omniscience_public_600-evaluation",
            ),
        )
    }
    development = _questions_from_source("triviaqa-development", paths["triviaqa-development"])
    grader_bundle = validate_confirmatory_grader_bundle(paths["frozen-graders"])
    try:
        snapshot_manifest = json.loads(
            (paths["frozen-evaluation-scripts"] / "snapshot-manifest.json").read_text(
                encoding="utf-8"
            )
        )
        snapshot_digest = str(snapshot_manifest["study_protocol_digest"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise DataValidationError(
            f"robustness evaluation snapshot identity is invalid: {exc}"
        ) from exc
    snapshot = validate_execution_snapshot(
        paths["frozen-evaluation-scripts"],
        study_protocol_digest=snapshot_digest,
        phase=ExperimentPhase.E9,
    )
    component_study_digest = _validate_robustness_component_selection(
        paths["frozen-component-selection"],
        loaded_prompts,
    )
    if component_study_digest != snapshot_digest:
        raise DataValidationError(
            "robustness component selection and evaluator snapshot use different studies"
        )
    e1_provenance = _validate_e1_reviewed_split_binding(
        config=config,
        e1_run=paths["e1-phase-ledger"],
        evaluation_snapshot=paths["frozen-evaluation-scripts"],
        snapshot_manifest=snapshot,
        reviewed_paths=reviewed_paths,
        grader_bundle=grader_bundle,
        completion_execution_private_key=completion_execution_private_key,
    )
    source_hashes = {name: sha256_path(path) for name, path in paths.items()}
    source_hashes["e1-phase-ledger"] = str(e1_provenance["e1_binding_sha256"])
    runtime_artifact_sha256 = grader_bundle.component_fingerprints.get(
        "runtime_attestation"
    )
    if (
        not isinstance(runtime_artifact_sha256, str)
        or _SHA256.fullmatch(runtime_artifact_sha256) is None
        or grader_bundle.runtime_attestation.get("execution_public_key")
        != grader_bundle.scorer.execution_public_key
    ):
        raise DataValidationError(
            "robustness M3 capture runtime is not independently frozen"
        )
    return (
        MappingProxyType(loaded_prompts),
        MappingProxyType(evaluations),
        development,
        MappingProxyType(paths),
        MappingProxyType(source_hashes),
        e1_provenance,
        runtime_artifact_sha256,
    )


def build_robustness_diagnostic_plan(
    *,
    config: Mapping[str, Any],
    source_artifacts: Mapping[str, str | Path],
    completion_execution_private_key: str | None = None,
) -> RobustnessDiagnosticPlan:
    """Derive every prompt task and all 10 leave-one-cluster-out RQ1 tasks."""

    (
        canonical_prompts,
        evaluation_questions,
        triviaqa_development_questions,
        _source_paths,
        source_artifact_sha256,
        e1_provenance,
        m3_capture_runtime_artifact_sha256,
    ) = _load_bound_plan_inputs(
        config,
        source_artifacts,
        completion_execution_private_key=completion_execution_private_key,
    )
    if config.get("config_digest") != APPROVED_ROBUSTNESS_CONFIG_DIGEST:
        raise ConfigurationError("robustness plan requires the approved config")
    if set(canonical_prompts) != set(_BASE_PROMPTS) or set(evaluation_questions) != set(
        _BENCHMARKS
    ):
        raise DataValidationError("robustness prompt or benchmark inputs differ")
    if set(source_artifact_sha256) != _SOURCE_BINDINGS or any(
        not isinstance(value, str) or _SHA256.fullmatch(value) is None
        for value in source_artifact_sha256.values()
    ):
        raise DataValidationError("robustness source artifact bindings differ")
    prompt_config = config["prompt_paraphrase"]
    assert isinstance(prompt_config, dict)
    variants = prompt_config["variants"]
    assert isinstance(variants, list)
    seed = int(prompt_config["selection_seed"])
    count = int(prompt_config["questions_per_benchmark"])
    selected: dict[str, list[dict[str, str]]] = {}
    question_source_digests: dict[str, str] = {}
    for benchmark in _BENCHMARKS:
        source = tuple(evaluation_questions[benchmark])
        question_source_digests[benchmark] = stable_hash(
            [
                _question_fingerprint(value)
                for value in sorted(source, key=lambda item: item.question_id)
            ]
        )
        selected[benchmark] = [
            {
                "question_id": value.question_id,
                "question_fingerprint": _question_fingerprint(value),
            }
            for value in _selected_questions(
                source,
                benchmark=benchmark,
                seed=seed,
                count=count,
            )
        ]
    prompt_task_count = sum(
        len(selected[benchmark]) * len(variants) * len(_METHODS) for benchmark in _BENCHMARKS
    )

    development = tuple(triviaqa_development_questions)
    rq1_config = config["rq1_generalization"]
    assert isinstance(rq1_config, dict)
    allowed_partitions = {"T-steer", "T-controller", "T-dev"}
    if (
        not development
        or len({value.question_id for value in development}) != len(development)
        or any(
            value.benchmark != "triviaqa" or value.split not in allowed_partitions
            for value in development
        )
    ):
        raise DataValidationError("RQ1 development questions are invalid")
    if Counter(value.split for value in development) != Counter(_RQ1_SOURCE_COUNTS):
        raise DataValidationError("RQ1 reviewed development counts differ from the plan")
    group_ids = semantic_group_ids(development)
    group_source_partitions: dict[str, set[str]] = {}
    for value in development:
        partition = value.split
        assert partition is not None  # validated against the exact reviewed split set above
        group_source_partitions.setdefault(group_ids[value.question_id], set()).add(partition)
    if any(len(partitions) != 1 for partitions in group_source_partitions.values()):
        raise DataValidationError("RQ1 semantic groups cross reviewed split boundaries")
    controller = tuple(value for value in development if value.split == "T-controller")
    controller_partitions = controller_feature_partitions(
        controller,
        calibration_rows=int(rq1_config["controller_calibration_rows"]),
        seed=int(rq1_config["controller_subdivision_seed"]),
    )
    feature_partitions = {
        value.question_id: (
            controller_partitions[value.question_id]
            if value.split == "T-controller"
            else value.split
        )
        for value in development
    }
    fold_count = int(rq1_config["semantic_fold_count"])
    assignments = [
        {
            "question_id": value.question_id,
            "question_fingerprint": _question_fingerprint(value),
            "source_partition": value.split,
            "partition": feature_partitions[value.question_id],
            "semantic_group_id": group_ids[value.question_id],
            "semantic_fold": int(group_ids[value.question_id][:16], 16) % fold_count,
        }
        for value in sorted(development, key=lambda item: item.question_id)
    ]
    source_fit_partitions = {"T-steer", "T-controller-train"}
    for fold in range(fold_count):
        inventories = {
            "source_fit": sum(
                row["semantic_fold"] != fold and row["partition"] in source_fit_partitions
                for row in assignments
            ),
            "source_calibration": sum(
                row["semantic_fold"] != fold and row["partition"] == "T-controller-calibration"
                for row in assignments
            ),
            "held_out_calibration": sum(
                row["semantic_fold"] == fold and row["partition"] == "T-controller-calibration"
                for row in assignments
            ),
            "held_out_vector_bank": sum(
                row["semantic_fold"] == fold and row["partition"] == "T-steer"
                for row in assignments
            ),
            "held_out_controller_train": sum(
                row["semantic_fold"] == fold
                and row["partition"] == "T-controller-train"
                for row in assignments
            ),
            "held_out_evaluation": sum(
                row["semantic_fold"] == fold and row["partition"] == "T-dev" for row in assignments
            ),
        }
        if not all(inventories.values()):
            raise DataValidationError(
                f"RQ1 semantic fold {fold} lacks a fitting or evaluation partition"
            )
    rq1_tasks: list[dict[str, Any]] = []
    for fold in range(fold_count):
        for evaluation_prompt_id in _BASE_PROMPTS:
            for method in _RQ1_METHODS:
                regimes = (
                    ("source-frozen-control",)
                    if method == "M1"
                    else ("calibration-only", "full-vector-bank-relearning")
                )
                for regime in regimes:
                    task_body = {
                        "held_out_fold": fold,
                        "training_prompt_id": "P0-neutral",
                        "evaluation_prompt_id": evaluation_prompt_id,
                        "method": method,
                        "adaptation_regime": regime,
                    }
                    rq1_tasks.append({**task_body, "task_id": _task_id("rq1", task_body)})
    canonical_prompt_fingerprints = {
        name: _prompt_fingerprint(canonical_prompts[name]) for name in _BASE_PROMPTS
    }
    body = {
        "schema_version": 1,
        "diagnostic_id": config["diagnostic_id"],
        "config_digest": config["config_digest"],
        "freeze_boundary": config["freeze_boundary"],
        "eligible_for_component_selection": False,
        "source_artifact_sha256": dict(sorted(source_artifact_sha256.items())),
        "m3_capture_runtime_artifact_sha256": m3_capture_runtime_artifact_sha256,
        "e1_provenance": dict(e1_provenance),
        "canonical_prompt_fingerprints": canonical_prompt_fingerprints,
        "prompt_paraphrase": {
            "selection_seed": seed,
            "selection_algorithm": prompt_config["selection_algorithm"],
            "generation_seed": prompt_config["generation_seed"],
            "questions_per_benchmark": count,
            "benchmarks": dict(prompt_config["benchmarks"]),
            "methods": list(_METHODS),
            "variants": variants,
            "question_source_digests": question_source_digests,
            "selected_questions": selected,
            "expected_task_count": prompt_task_count,
        },
        "rq1_generalization": {
            **dict(rq1_config),
            "question_source_digest": stable_hash(
                [
                    _question_fingerprint(value)
                    for value in sorted(development, key=lambda item: item.question_id)
                ]
            ),
            "assignments": assignments,
            "tasks": rq1_tasks,
            "expected_task_count": len(rq1_tasks),
        },
    }
    return RobustnessDiagnosticPlan(None, body, stable_hash(body))


def iter_prompt_paraphrase_tasks(
    plan: RobustnessDiagnosticPlan,
) -> Iterator[PromptParaphraseTask]:
    section = plan.body["prompt_paraphrase"]
    assert isinstance(section, Mapping)
    partitions = section["benchmarks"]
    for benchmark in _BENCHMARKS:
        for variant in section["variants"]:
            for method in section["methods"]:
                for question in section["selected_questions"][benchmark]:
                    body = {
                        "benchmark": benchmark,
                        "partition": partitions[benchmark],
                        "question_id": question["question_id"],
                        "question_fingerprint": question["question_fingerprint"],
                        "base_prompt_id": variant["base_prompt_id"],
                        "prompt_id": variant["prompt_id"],
                        "prompt_text": variant["text"],
                        "method": method,
                    }
                    yield PromptParaphraseTask(task_id=_task_id("prompt", body), **body)


def iter_rq1_generalization_tasks(
    plan: RobustnessDiagnosticPlan,
) -> Iterator[RQ1GeneralizationTask]:
    section = plan.body["rq1_generalization"]
    assert isinstance(section, Mapping)
    for value in section["tasks"]:
        yield RQ1GeneralizationTask(**value)


def rq1_task_question_sets(
    plan: RobustnessDiagnosticPlan,
    task: RQ1GeneralizationTask,
) -> Mapping[str, tuple[str, ...]]:
    """Materialize leakage-disjoint fitting/adaptation/evaluation IDs for one task."""

    section = plan.body["rq1_generalization"]
    assert isinstance(section, Mapping)
    known = {value.task_id for value in iter_rq1_generalization_tasks(plan)}
    if task.task_id not in known:
        raise DataValidationError("RQ1 task is not part of the frozen plan")
    fold = task.held_out_fold
    rows = section["assignments"]
    result = {
        "source_fit": tuple(
            row["question_id"]
            for row in rows
            if row["semantic_fold"] != fold
            and row["partition"] in {"T-steer", "T-controller-train"}
        ),
        "source_calibration": tuple(
            row["question_id"]
            for row in rows
            if row["semantic_fold"] != fold and row["partition"] == "T-controller-calibration"
        ),
        "held_out_adaptation": tuple(
            row["question_id"]
            for row in rows
            if row["semantic_fold"] == fold
            and row["partition"]
            in {"T-steer", "T-controller-train", "T-controller-calibration"}
        ),
        "held_out_evaluation": tuple(
            row["question_id"]
            for row in rows
            if row["semantic_fold"] == fold and row["partition"] == "T-dev"
        ),
    }
    if any(not values for values in result.values()) or any(
        set(result[left]) & set(result[right])
        for left in result
        for right in result
        if left < right
    ):
        raise FrozenArtifactError("RQ1 task question sets overlap or are incomplete")
    return MappingProxyType(result)


def freeze_robustness_diagnostic_plan(
    destination: str | Path,
    *,
    config_path: str | Path,
    source_artifacts: Mapping[str, str | Path],
    completion_execution_private_key: str | None = None,
) -> RobustnessDiagnosticPlan:
    """Package exact sources and atomically freeze the complete pre-E9 schedule."""

    target = validate_active_study_artifact_paths({"robustness diagnostic plan": destination})[
        "robustness diagnostic plan"
    ]
    config = load_robustness_diagnostic_config(config_path)
    _, _, _, sources, source_hashes, _, _ = _load_bound_plan_inputs(
        config,
        source_artifacts,
        completion_execution_private_key=completion_execution_private_key,
    )
    plan = build_robustness_diagnostic_plan(
        config=config,
        source_artifacts=sources,
        completion_execution_private_key=completion_execution_private_key,
    )
    if dict(plan.body["source_artifact_sha256"]) != dict(source_hashes):
        raise FrozenArtifactError("robustness sources changed during plan construction")
    if target.exists() or target.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite robustness plan: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=target.parent))
    try:
        config_source = _strict_source_path(config_path, "robustness config")
        if not config_source.is_file():
            raise DataValidationError("robustness config must be one regular file")
        shutil.copyfile(config_source, stage / "config.json")
        packaged_sources = stage / "sources"
        packaged_sources.mkdir()
        for name, source in sources.items():
            destination_path = packaged_sources / name
            if name == "e1-phase-ledger":
                portable = dict(plan.body["e1_provenance"])
                portable.pop("e1_binding_sha256")
                destination_path.write_text(
                    canonical_json(
                        {**portable, "binding_digest": stable_hash(portable)}
                    )
                    + "\n",
                    encoding="utf-8",
                )
            elif source.is_dir():
                shutil.copytree(source, destination_path)
            else:
                shutil.copyfile(source, destination_path)
        payload = {
            **_deep_thaw_json(plan.body),
            "plan_digest": plan.plan_digest,
        }
        (stage / "plan.json").write_text(canonical_json(payload) + "\n", encoding="utf-8")
        bundle_body = {
            "schema_version": 1,
            "plan_digest": plan.plan_digest,
            "config_sha256": sha256_file(stage / "config.json"),
            "source_artifact_sha256": dict(source_hashes),
        }
        (stage / "bundle.json").write_text(
            canonical_json({**bundle_body, "bundle_digest": stable_hash(bundle_body)}) + "\n",
            encoding="utf-8",
        )
        verify_robustness_diagnostic_plan(stage)
        os.replace(stage, target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return verify_robustness_diagnostic_plan(target)


def _validate_robustness_plan_structure(
    value: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    expected_root = {
        "schema_version",
        "diagnostic_id",
        "config_digest",
        "freeze_boundary",
        "eligible_for_component_selection",
        "source_artifact_sha256",
        "m3_capture_runtime_artifact_sha256",
        "e1_provenance",
        "canonical_prompt_fingerprints",
        "prompt_paraphrase",
        "rq1_generalization",
    }
    source_bindings = value.get("source_artifact_sha256")
    e1_provenance = value.get("e1_provenance")
    prompt_fingerprints = value.get("canonical_prompt_fingerprints")
    prompt = value.get("prompt_paraphrase")
    rq1 = value.get("rq1_generalization")
    if (
        set(value) != expected_root
        or value.get("schema_version") != 1
        or value.get("diagnostic_id") != config["diagnostic_id"]
        or not isinstance(source_bindings, dict)
        or set(source_bindings) != _SOURCE_BINDINGS
        or any(
            not isinstance(item, str) or _SHA256.fullmatch(item) is None
            for item in source_bindings.values()
        )
        or not isinstance(value.get("m3_capture_runtime_artifact_sha256"), str)
        or _SHA256.fullmatch(value["m3_capture_runtime_artifact_sha256"]) is None
        or not isinstance(e1_provenance, dict)
        or set(e1_provenance)
        != {
            "schema_version",
            "kind",
            "study_protocol_digest",
            "e1_completion_digest",
            "e1_contract_digest",
            "e1_record_set_digest",
            "e1_record_count",
            "e1_ledger_sha256",
            "e1_binding_sha256",
            "e1_condition_cells",
            "e1_conditions_sha256",
            "e1_question_ids_sha256",
            "e1_input_fingerprints",
            "e1_prerequisite_digests",
            "reviewed_split_manifest_digest",
            "reviewed_split_sha256",
            "completion_execution_public_key",
            "completion_signature",
        }
        or e1_provenance.get("schema_version") != 2
        or e1_provenance.get("kind")
        != "signed-portable-complete-qwen-e1-binding"
        or e1_provenance.get("e1_record_count") != 19_800
        or e1_provenance.get("e1_condition_cells")
        != [
            {
                "benchmark": benchmark,
                "partition": partition,
                "system_prompt_id": prompt,
                "steering_method": "M0",
                "seed": 17,
            }
            for benchmark, partition, prompt in sorted(
                (
                    benchmark,
                    partition,
                    prompt,
                )
                for benchmark, partition in {
                    "triviaqa": "T-controller",
                    "simpleqa_verified": "simpleqa-eval",
                    "aa_omniscience_public_600": "aa-eval",
                }.items()
                for prompt in {"P0-neutral", "P1-direct", "P2-calibrated-abstention"}
            )
        ]
        or any(
            not isinstance(e1_provenance.get(name), str)
            or _SHA256.fullmatch(e1_provenance[name]) is None
            for name in (
                "study_protocol_digest",
                "e1_completion_digest",
                "e1_contract_digest",
                "e1_record_set_digest",
                "e1_ledger_sha256",
                "e1_binding_sha256",
                "e1_conditions_sha256",
                "reviewed_split_manifest_digest",
                "reviewed_split_sha256",
                "completion_execution_public_key",
            )
        )
        or not isinstance(e1_provenance.get("completion_signature"), str)
        or re.fullmatch(
            r"[0-9a-f]{128}", str(e1_provenance["completion_signature"])
        )
        is None
        or not _verify_portable_e1_completion_signature(e1_provenance)
        or not isinstance(e1_provenance.get("e1_question_ids_sha256"), dict)
        or set(e1_provenance["e1_question_ids_sha256"])
        != {"triviaqa", "simpleqa_verified", "aa_omniscience_public_600"}
        or any(
            not isinstance(item, str) or _SHA256.fullmatch(item) is None
            for item in e1_provenance["e1_question_ids_sha256"].values()
        )
        or not isinstance(e1_provenance.get("e1_input_fingerprints"), dict)
        or set(e1_provenance["e1_input_fingerprints"])
        != {"deduplicated_splits", "grader_bundle", "inference_protocol"}
        or any(
            not isinstance(item, str) or _SHA256.fullmatch(item) is None
            for item in e1_provenance["e1_input_fingerprints"].values()
        )
        or not isinstance(e1_provenance.get("e1_prerequisite_digests"), dict)
        or set(e1_provenance["e1_prerequisite_digests"]) != {"E0"}
        or not isinstance(e1_provenance["e1_prerequisite_digests"].get("E0"), str)
        or _SHA256.fullmatch(e1_provenance["e1_prerequisite_digests"]["E0"])
        is None
        or e1_provenance["e1_binding_sha256"] != source_bindings["e1-phase-ledger"]
        or e1_provenance["reviewed_split_manifest_digest"]
        != config["reviewed_split_binding"]["manifest_digest"]
        or e1_provenance["reviewed_split_sha256"]
        != config["reviewed_split_binding"]["artifact_sha256"]
        or not isinstance(prompt_fingerprints, dict)
        or set(prompt_fingerprints) != set(_BASE_PROMPTS)
        or any(
            not isinstance(item, str) or _SHA256.fullmatch(item) is None
            for item in prompt_fingerprints.values()
        )
        or not isinstance(prompt, dict)
        or not isinstance(rq1, dict)
    ):
        raise FrozenArtifactError("robustness diagnostic plan structure differs")

    prompt_config = config["prompt_paraphrase"]
    assert isinstance(prompt_config, dict)
    expected_prompt_keys = {
        "selection_seed",
        "selection_algorithm",
        "generation_seed",
        "questions_per_benchmark",
        "benchmarks",
        "methods",
        "variants",
        "question_source_digests",
        "selected_questions",
        "expected_task_count",
    }
    selected = prompt.get("selected_questions")
    source_digests = prompt.get("question_source_digests")
    if (
        set(prompt) != expected_prompt_keys
        or any(
            prompt.get(name) != prompt_config[name]
            for name in (
                "selection_seed",
                "selection_algorithm",
                "generation_seed",
                "questions_per_benchmark",
                "benchmarks",
                "methods",
                "variants",
            )
        )
        or prompt.get("expected_task_count") != 36_000
        or not isinstance(selected, dict)
        or set(selected) != set(_BENCHMARKS)
        or not isinstance(source_digests, dict)
        or set(source_digests) != set(_BENCHMARKS)
        or any(
            not isinstance(item, str) or _SHA256.fullmatch(item) is None
            for item in source_digests.values()
        )
    ):
        raise FrozenArtifactError("prompt-paraphrase plan differs from its config")
    for benchmark in _BENCHMARKS:
        rows = selected[benchmark]
        if (
            not isinstance(rows, list)
            or len(rows) != 200
            or len({row.get("question_id") for row in rows if isinstance(row, dict)}) != 200
            or any(
                not isinstance(row, dict)
                or set(row) != {"question_id", "question_fingerprint"}
                or not isinstance(row["question_id"], str)
                or not row["question_id"].strip()
                or not isinstance(row["question_fingerprint"], str)
                or _SHA256.fullmatch(row["question_fingerprint"]) is None
                for row in rows
            )
        ):
            raise FrozenArtifactError(f"prompt-paraphrase {benchmark} fixed subset differs")

    rq1_config = config["rq1_generalization"]
    assert isinstance(rq1_config, dict)
    assignments = rq1.get("assignments")
    tasks = rq1.get("tasks")
    if (
        set(rq1)
        != set(rq1_config)
        | {
            "question_source_digest",
            "assignments",
            "tasks",
            "expected_task_count",
        }
        or any(rq1.get(name) != item for name, item in rq1_config.items())
        or not isinstance(rq1.get("question_source_digest"), str)
        or _SHA256.fullmatch(rq1["question_source_digest"]) is None
        or not isinstance(assignments, list)
        or not isinstance(tasks, list)
        or rq1.get("expected_task_count") != 60
    ):
        raise FrozenArtifactError("RQ1 plan differs from its config")
    expected_assignment_keys = {
        "question_id",
        "question_fingerprint",
        "source_partition",
        "partition",
        "semantic_group_id",
        "semantic_fold",
    }
    if (
        len(assignments) != sum(_RQ1_SOURCE_COUNTS.values())
        or len({row.get("question_id") for row in assignments if isinstance(row, dict)})
        != len(assignments)
        or any(
            not isinstance(row, dict)
            or set(row) != expected_assignment_keys
            or not isinstance(row["question_id"], str)
            or not row["question_id"].strip()
            or not isinstance(row["question_fingerprint"], str)
            or _SHA256.fullmatch(row["question_fingerprint"]) is None
            or row["source_partition"] not in _RQ1_SOURCE_COUNTS
            or row["partition"]
            not in {
                "T-steer",
                "T-controller-train",
                "T-controller-calibration",
                "T-dev",
            }
            or not isinstance(row["semantic_group_id"], str)
            or _SHA256.fullmatch(row["semantic_group_id"]) is None
            or type(row["semantic_fold"]) is not int
            or row["semantic_fold"] != int(row["semantic_group_id"][:16], 16) % 10
            or (
                row["source_partition"] == "T-controller"
                and row["partition"] not in {"T-controller-train", "T-controller-calibration"}
            )
            or (
                row["source_partition"] != "T-controller"
                and row["partition"] != row["source_partition"]
            )
            for row in assignments
        )
    ):
        raise FrozenArtifactError("RQ1 semantic-fold assignments differ")
    if Counter(row["source_partition"] for row in assignments) != Counter(
        _RQ1_SOURCE_COUNTS
    ) or sum(row["partition"] == "T-controller-calibration" for row in assignments) != int(
        rq1_config["controller_calibration_rows"]
    ):
        raise FrozenArtifactError("RQ1 source or calibration counts differ")
    group_partitions: dict[str, set[str]] = {}
    for row in assignments:
        group_partitions.setdefault(row["semantic_group_id"], set()).add(row["source_partition"])
    if any(len(partitions) != 1 for partitions in group_partitions.values()):
        raise FrozenArtifactError("RQ1 semantic groups cross source partitions")
    required_fold_partitions = {
        "T-steer",
        "T-controller-train",
        "T-controller-calibration",
        "T-dev",
    }
    for fold in range(10):
        observed = {
            str(row["partition"])
            for row in assignments
            if row["semantic_fold"] == fold
        }
        if not required_fold_partitions <= observed:
            raise FrozenArtifactError(
                "RQ1 every semantic fold must support source fit, adaptation, and evaluation"
            )


def verify_robustness_diagnostic_plan(
    path: str | Path,
) -> RobustnessDiagnosticPlan:
    """Replay the packaged config/sources and rebuild every plan decision."""

    source = _strict_source_path(path, "robustness diagnostic bundle")
    if (
        not source.is_dir()
        or {item.name for item in source.iterdir()}
        != {"bundle.json", "config.json", "plan.json", "sources"}
        or (source / "sources").is_symlink()
        or not (source / "sources").is_dir()
        or {item.name for item in (source / "sources").iterdir()} != _SOURCE_BINDINGS
        or any(item.is_symlink() for item in source.rglob("*"))
    ):
        raise FrozenArtifactError("robustness diagnostic bundle inventory differs")
    bundle = _load_object(source / "bundle.json", "robustness bundle manifest")
    bundle_digest = bundle.pop("bundle_digest", None)
    if (
        set(bundle)
        != {
            "schema_version",
            "plan_digest",
            "config_sha256",
            "source_artifact_sha256",
        }
        or bundle.get("schema_version") != 1
        or bundle_digest != stable_hash(bundle)
        or bundle.get("config_sha256") != sha256_file(source / "config.json")
        or not isinstance(bundle.get("source_artifact_sha256"), dict)
    ):
        raise FrozenArtifactError("robustness diagnostic bundle identity differs")
    packaged_sources = {name: source / "sources" / name for name in _SOURCE_BINDINGS}
    observed_hashes = {name: sha256_path(location) for name, location in packaged_sources.items()}
    if bundle["source_artifact_sha256"] != observed_hashes:
        raise FrozenArtifactError("packaged robustness source changed")
    config = load_robustness_diagnostic_config(source / "config.json")
    value = _load_object(source / "plan.json", "robustness diagnostic plan")
    digest = value.pop("plan_digest", None)
    if (
        not isinstance(digest, str)
        or digest != stable_hash(value)
        or digest != bundle["plan_digest"]
        or value.get("config_digest") != config["config_digest"]
        or value.get("eligible_for_component_selection") is not False
        or value.get("freeze_boundary") != config["freeze_boundary"]
    ):
        raise FrozenArtifactError("robustness diagnostic plan identity differs")
    _validate_robustness_plan_structure(value, config)
    rebuilt = build_robustness_diagnostic_plan(
        config=config,
        source_artifacts=packaged_sources,
    )
    if rebuilt.plan_digest != digest or _deep_thaw_json(rebuilt.body) != value:
        raise FrozenArtifactError("robustness plan does not replay from packaged sources")
    plan = RobustnessDiagnosticPlan(source, value, digest)
    prompt_tasks = tuple(iter_prompt_paraphrase_tasks(plan))
    rq1_tasks = tuple(iter_rq1_generalization_tasks(plan))
    prompt_section = value.get("prompt_paraphrase")
    rq1_section = value.get("rq1_generalization")
    if (
        not isinstance(prompt_section, dict)
        or not isinstance(rq1_section, dict)
        or len(prompt_tasks) != prompt_section.get("expected_task_count")
        or len(prompt_tasks) != 36_000
        or len({task.task_id for task in prompt_tasks}) != len(prompt_tasks)
        or len(rq1_tasks) != rq1_section.get("expected_task_count")
        or len(rq1_tasks) != 60
        or len({task.task_id for task in rq1_tasks}) != len(rq1_tasks)
    ):
        raise FrozenArtifactError("robustness diagnostic task inventory differs")
    for task in rq1_tasks:
        rq1_task_question_sets(plan, task)
    return plan

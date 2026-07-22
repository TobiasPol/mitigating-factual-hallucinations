"""Verified E4 baseline feasibility, exact screening, and gate-compatible promotion."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from mfh.contracts import (
    ActivationSite,
    AdaptivePolicySpec,
    GenerationRecord,
    Outcome,
    Question,
    TokenScope,
)
from mfh.data.splits import semantic_group_ids
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.evaluation.metrics import metric_bundle
from mfh.experiments.e4_act_vllm import verify_e4_act_baseline
from mfh.experiments.model_selection import (
    ACTIVE_MODEL_IDENTITIES,
    ACTIVE_MODEL_NAME,
    validate_active_study_artifact_paths,
)
from mfh.experiments.protocol import ExperimentPhase, StudyProtocol
from mfh.experiments.runner import PhaseRunLedger, validate_adaptive_execution
from mfh.experiments.static_direction_sources import resolve_static_direction
from mfh.provenance import canonical_json, sha256_path, stable_hash

_METHODS = (
    "M1",
    "M2",
    "ITI-if-feasible",
    "ACT-or-SADI",
    "TruthX-if-feasible",
)
_PROMPTS = ("P0-neutral", "P2-calibrated-abstention")
_REQUIRED_SOURCES = frozenset({"E2_calibrated_probes", "E3_static_vectors"})
_OPTIONAL = frozenset({"ITI-if-feasible", "TruthX-if-feasible"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SELECTION_RULE = (
    "lowest-risk-per-method-and-overall-within-5pp-M1-baseline-coverage-mandatory-M2"
)
_EXPECTED_T_DEV_IDS_DIGEST = (
    "7e7007f750af0c01f7e1eaae11cee546659c3520ae1357f2e8824753988dd2b3"
)
_EXPECTED_T_DEV_QUESTIONS_DIGEST = (
    "da5728561c7b82b230c147cfd4fd30f0c99319010a63a3cc33cc53ca651b4802"
)
_PREFLIGHT_CHECKS = {
    "M1": frozenset({"implementation_loads", "runtime_hook_supported"}),
    "M2": frozenset(
        {"implementation_loads", "paired_training_materials", "runtime_hook_supported"}
    ),
    "ITI-if-feasible": frozenset({"implementation_available", "per_head_output_hook"}),
    "ACT-or-SADI": frozenset(
        {
            "implementation_loads",
            "calibrated_probe_available",
            "runtime_hook_supported",
        }
    ),
    "TruthX-if-feasible": frozenset(
        {"compatible_autoencoder", "implementation_available", "runtime_hook_supported"}
    ),
}

_ACTIVE_MODEL = ACTIVE_MODEL_IDENTITIES[ACTIVE_MODEL_NAME]
_MODEL_RUNTIME_IDENTITY = MappingProxyType(
    {
        "repository": _ACTIVE_MODEL["repository"],
        "revision": _ACTIVE_MODEL["revision"],
        "runtime": _ACTIVE_MODEL["runtime"].value,
        "quantization": _ACTIVE_MODEL["quantization"],
        "num_layers": _ACTIVE_MODEL["num_layers"],
    }
)


def _digest(value: object, context: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise DataValidationError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _questions_digest(questions: Sequence[Question]) -> str:
    return stable_hash(
        [
            {
                "question_id": value.question_id,
                "benchmark": value.benchmark,
                "text": value.text,
                "aliases": list(value.aliases),
                "split": value.split,
                "entities": list(value.entities),
                "metadata": dict(value.metadata),
            }
            for value in questions
        ]
    )


def _atomic_write(path: str | Path, payload: Mapping[str, Any], context: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite {context}: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _exact_text(path: str | Path, payload: Mapping[str, Any], context: str) -> str:
    source = Path(path)
    expected = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        observed = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise FrozenArtifactError(f"cannot read {context}: {exc}") from exc
    if source.is_symlink() or not source.is_file() or observed != expected:
        raise FrozenArtifactError(f"{context} differs from exact replay")
    return sha256_path(source)


class E4Feasibility(StrEnum):
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"


@dataclass(frozen=True, slots=True)
class E4Protocol:
    dev_rows: int = 5_000
    screen_rows: int = 2_000
    seed: int = 17
    maximum_coverage_loss: float = 0.05

    def __post_init__(self) -> None:
        if (
            type(self.dev_rows) is not int
            or type(self.screen_rows) is not int
            or not 0 < self.screen_rows < self.dev_rows
            or type(self.seed) is not int
            or self.seed < 0
            or type(self.maximum_coverage_loss) is not float
            or self.maximum_coverage_loss != 0.05
        ):
            raise DataValidationError("E4 protocol is invalid")

    @property
    def scientific_eligible(self) -> bool:
        return self == E4Protocol()

    def to_dict(self) -> dict[str, Any]:
        return {
            "dev_rows": self.dev_rows,
            "screen_rows": self.screen_rows,
            "seed": self.seed,
            "maximum_coverage_loss": self.maximum_coverage_loss,
        }


def _protocol(value: E4Protocol | None) -> E4Protocol:
    if value is None:
        return E4Protocol()
    if type(value) is not E4Protocol:
        raise DataValidationError("E4 protocol must be an exact E4Protocol")
    return value


def select_e4_screen_questions(
    questions: Sequence[Question], *, protocol: E4Protocol | None = None
) -> tuple[str, ...]:
    frozen = _protocol(protocol)
    values = tuple(questions)
    if (
        len(values) != frozen.dev_rows
        or len({value.question_id for value in values}) != len(values)
        or any(value.benchmark != "triviaqa" or value.split != "T-dev" for value in values)
    ):
        raise DataValidationError("E4 screen requires the exact unique TriviaQA T-dev")
    groups = semantic_group_ids(values)
    members: dict[str, list[str]] = defaultdict(list)
    for question in values:
        members[groups[question.question_id]].append(question.question_id)
    ordered = sorted(
        members,
        key=lambda group: stable_hash({"seed": frozen.seed, "group": group}),
    )
    predecessors: dict[int, tuple[int, str] | None] = {0: None}
    for group in ordered:
        size = len(members[group])
        for total in sorted(tuple(predecessors), reverse=True):
            candidate = total + size
            if candidate <= frozen.screen_rows and candidate not in predecessors:
                predecessors[candidate] = (total, group)
    if frozen.screen_rows not in predecessors:
        raise DataValidationError("E4 semantic groups cannot fill the exact screen")
    selected: set[str] = set()
    cursor = frozen.screen_rows
    while cursor:
        predecessor = predecessors[cursor]
        assert predecessor is not None
        cursor, group = predecessor
        selected.add(group)
    return tuple(
        question.question_id for question in values if groups[question.question_id] in selected
    )


@dataclass(frozen=True, slots=True)
class E4ScreenReceipt:
    protocol: E4Protocol
    dev_questions: tuple[Question, ...]
    dev_question_ids_digest: str
    dev_questions_digest: str
    screen_question_ids: tuple[str, ...]
    screen_question_ids_digest: str
    protocol_digest: str
    scientific_eligible: bool
    receipt_digest: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or type(self.protocol) is not E4Protocol
            or any(
                _SHA256.fullmatch(value) is None
                for value in (
                    self.dev_question_ids_digest,
                    self.dev_questions_digest,
                    self.screen_question_ids_digest,
                    self.protocol_digest,
                    self.receipt_digest,
                )
            )
            or type(self.screen_question_ids) is not tuple
            or not self.screen_question_ids
            or len(set(self.screen_question_ids)) != len(self.screen_question_ids)
            or type(self.scientific_eligible) is not bool
            or type(self.dev_questions) is not tuple
            or any(type(value) is not Question for value in self.dev_questions)
            or stable_hash([value.question_id for value in self.dev_questions])
            != self.dev_question_ids_digest
            or _questions_digest(self.dev_questions) != self.dev_questions_digest
            or self.protocol_digest != stable_hash(self.protocol.to_dict())
            or self.screen_question_ids_digest
            != stable_hash(list(self.screen_question_ids))
            or self.receipt_digest != stable_hash(self._body())
        ):
            raise DataValidationError("E4 screen receipt is invalid")

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol": self.protocol.to_dict(),
            "dev_questions": [
                {
                    "question_id": value.question_id,
                    "benchmark": value.benchmark,
                    "text": value.text,
                    "aliases": list(value.aliases),
                    "split": value.split,
                    "entities": list(value.entities),
                    "metadata": dict(value.metadata),
                }
                for value in self.dev_questions
            ],
            "dev_question_ids_digest": self.dev_question_ids_digest,
            "dev_questions_digest": self.dev_questions_digest,
            "screen_question_ids": list(self.screen_question_ids),
            "screen_question_ids_digest": self.screen_question_ids_digest,
            "protocol_digest": self.protocol_digest,
            "scientific_eligible": self.scientific_eligible,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "receipt_digest": self.receipt_digest}

    def assert_current(self) -> None:
        self.__post_init__()
        replayed = select_e4_screen_questions(
            self.dev_questions,
            protocol=self.protocol,
        )
        if (
            replayed != self.screen_question_ids
            or self.scientific_eligible
            != (
                self.protocol.scientific_eligible
                and self.dev_question_ids_digest == _EXPECTED_T_DEV_IDS_DIGEST
                and self.dev_questions_digest == _EXPECTED_T_DEV_QUESTIONS_DIGEST
                and self.protocol_digest == stable_hash(E4Protocol().to_dict())
            )
        ):
            raise FrozenArtifactError("E4 screen differs from deterministic replay")


def build_e4_screen_receipt(
    questions: Sequence[Question], *, protocol: E4Protocol | None = None
) -> E4ScreenReceipt:
    frozen = _protocol(protocol)
    values = tuple(questions)
    selected = select_e4_screen_questions(values, protocol=frozen)
    ids_digest = stable_hash([value.question_id for value in values])
    questions_digest = _questions_digest(values)
    screen_ids_digest = stable_hash(list(selected))
    protocol_digest = stable_hash(frozen.to_dict())
    scientific_eligible = bool(
        frozen.scientific_eligible
        and ids_digest == _EXPECTED_T_DEV_IDS_DIGEST
        and questions_digest == _EXPECTED_T_DEV_QUESTIONS_DIGEST
    )
    body = {
        "schema_version": 1,
        "protocol": frozen.to_dict(),
        "dev_questions": [
            {
                "question_id": value.question_id,
                "benchmark": value.benchmark,
                "text": value.text,
                "aliases": list(value.aliases),
                "split": value.split,
                "entities": list(value.entities),
                "metadata": dict(value.metadata),
            }
            for value in values
        ],
        "dev_question_ids_digest": ids_digest,
        "dev_questions_digest": questions_digest,
        "screen_question_ids": list(selected),
        "screen_question_ids_digest": screen_ids_digest,
        "protocol_digest": protocol_digest,
        "scientific_eligible": scientific_eligible,
    }
    receipt = E4ScreenReceipt(
        protocol=frozen,
        dev_questions=values,
        dev_question_ids_digest=ids_digest,
        dev_questions_digest=questions_digest,
        screen_question_ids=selected,
        screen_question_ids_digest=screen_ids_digest,
        protocol_digest=protocol_digest,
        scientific_eligible=scientific_eligible,
        receipt_digest=stable_hash(body),
    )
    return receipt


def load_e4_screen_receipt(path: str | Path) -> E4ScreenReceipt:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        questions = payload["dev_questions"]
        if not isinstance(payload, Mapping) or not isinstance(questions, list):
            raise TypeError("screen root or questions are invalid")
        receipt = E4ScreenReceipt(
            schema_version=payload["schema_version"],
            protocol=E4Protocol(**payload["protocol"]),
            dev_questions=tuple(
                Question(
                    question_id=value["question_id"],
                    benchmark=value["benchmark"],
                    text=value["text"],
                    aliases=tuple(value["aliases"]),
                    split=value["split"],
                    entities=tuple(value["entities"]),
                    metadata=value["metadata"],
                )
                for value in questions
            ),
            dev_question_ids_digest=payload["dev_question_ids_digest"],
            dev_questions_digest=payload["dev_questions_digest"],
            screen_question_ids=tuple(payload["screen_question_ids"]),
            screen_question_ids_digest=payload["screen_question_ids_digest"],
            protocol_digest=payload["protocol_digest"],
            scientific_eligible=payload["scientific_eligible"],
            receipt_digest=payload["receipt_digest"],
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, DataValidationError) as exc:
        raise FrozenArtifactError(f"cannot load E4 screen receipt: {exc}") from exc
    receipt.assert_current()
    _exact_text(source, receipt.to_dict(), "E4 screen receipt")
    return receipt


def write_e4_screen_receipt(path: str | Path, receipt: E4ScreenReceipt) -> None:
    receipt.assert_current()
    path = validate_active_study_artifact_paths(
        {"E4 screen receipt": path}
    )["E4 screen receipt"]
    _atomic_write(path, receipt.to_dict(), "E4 screen receipt")


def verify_e4_screen_receipt(
    path: str | Path, *, expected: E4ScreenReceipt
) -> Mapping[str, Any]:
    expected.assert_current()
    fingerprint = _exact_text(path, expected.to_dict(), "E4 screen receipt")
    return MappingProxyType(
        {
            "valid": True,
            "receipt_digest": expected.receipt_digest,
            "artifact_sha256": fingerprint,
            "scientific_eligible": expected.scientific_eligible,
        }
    )


@dataclass(frozen=True, slots=True)
class E4MethodCapability:
    method: str
    feasibility: E4Feasibility
    implementation: str | None
    reason: str | None
    evidence_artifact_sha256: str
    implementation_artifact_sha256: str | None

    def __post_init__(self) -> None:
        if (
            self.method not in _METHODS
            or not isinstance(self.feasibility, E4Feasibility)
            or _SHA256.fullmatch(self.evidence_artifact_sha256) is None
            or (self.feasibility is E4Feasibility.FEASIBLE)
            != (self.implementation is not None)
            or (self.feasibility is E4Feasibility.FEASIBLE)
            != (self.implementation_artifact_sha256 is not None)
            or (self.feasibility is E4Feasibility.INFEASIBLE) != (self.reason is not None)
            or (self.implementation is not None and not self.implementation.strip())
            or (self.reason is not None and not self.reason.strip())
            or (
                self.implementation_artifact_sha256 is not None
                and _SHA256.fullmatch(self.implementation_artifact_sha256) is None
            )
            or (
                self.method not in _OPTIONAL
                and self.feasibility is not E4Feasibility.FEASIBLE
            )
        ):
            raise DataValidationError("E4 method capability is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "feasibility": self.feasibility.value,
            "implementation": self.implementation,
            "reason": self.reason,
            "evidence_artifact_sha256": self.evidence_artifact_sha256,
            "implementation_artifact_sha256": self.implementation_artifact_sha256,
        }


def write_e4_method_preflight(
    path: str | Path,
    *,
    method: str,
    runtime_artifact_sha256: str,
    checks: Mapping[str, bool],
    details: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Write one canonical structured feasibility receipt for an E4 method."""

    if (
        method not in _METHODS
        or _SHA256.fullmatch(runtime_artifact_sha256) is None
        or set(checks) != _PREFLIGHT_CHECKS[method]
        or any(type(value) is not bool for value in checks.values())
        or (details is not None and not isinstance(details, Mapping))
    ):
        raise DataValidationError("E4 method preflight inputs differ from the frozen schema")
    feasibility = (
        E4Feasibility.FEASIBLE
        if all(checks.values())
        else E4Feasibility.INFEASIBLE
    )
    body: dict[str, Any] = {
        "schema_version": 1,
        "method": method,
        "model_identity": ACTIVE_MODEL_NAME,
        "model_runtime_identity": dict(_MODEL_RUNTIME_IDENTITY),
        "runtime_artifact_sha256": runtime_artifact_sha256,
        "feasibility": feasibility.value,
        "checks": dict(checks),
        "details": dict(details or {}),
    }
    value = {**body, "evidence_digest": stable_hash(body)}
    output = validate_active_study_artifact_paths(
        {"E4 method preflight": path}
    )["E4 method preflight"]
    _atomic_write(output, value, "E4 method preflight")
    return MappingProxyType(value)


@dataclass(frozen=True, slots=True)
class E4CapabilityReport:
    model_identity: str
    runtime_identity: Mapping[str, Any]
    runtime_artifact_sha256: str
    source_digests: Mapping[str, str]
    artifact_paths: Mapping[str, str]
    methods: tuple[E4MethodCapability, ...]
    report_digest: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        runtime = dict(self.runtime_identity)
        sources = dict(self.source_digests)
        paths = dict(self.artifact_paths)
        expected_paths = {
            "runtime",
            *{f"source:{key}" for key in _REQUIRED_SOURCES},
            *{f"evidence:{key}" for key in _METHODS},
            *{
                f"implementation:{value.method}"
                for value in self.methods
                if value.implementation_artifact_sha256 is not None
            },
        }
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.model_identity != ACTIVE_MODEL_NAME
            or runtime != dict(_MODEL_RUNTIME_IDENTITY)
            or _SHA256.fullmatch(self.runtime_artifact_sha256) is None
            or set(sources) != _REQUIRED_SOURCES
            or any(_SHA256.fullmatch(value) is None for value in sources.values())
            or type(self.methods) is not tuple
            or any(type(value) is not E4MethodCapability for value in self.methods)
            or tuple(value.method for value in self.methods) != _METHODS
            or set(paths) != expected_paths
            or any(
                type(value) is not str or not Path(value).is_absolute()
                for value in paths.values()
            )
        ):
            raise DataValidationError("E4 capability report identity is invalid")
        try:
            runtime = json.loads(json.dumps(runtime, sort_keys=True, allow_nan=False))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DataValidationError(f"E4 runtime identity is invalid: {exc}") from exc
        if runtime != dict(_MODEL_RUNTIME_IDENTITY):
            raise DataValidationError("E4 runtime identity differs from the active checkpoint")
        object.__setattr__(self, "runtime_identity", MappingProxyType(runtime))
        object.__setattr__(self, "source_digests", MappingProxyType(sources))
        object.__setattr__(self, "artifact_paths", MappingProxyType(paths))
        if self.report_digest != stable_hash(self._body()):
            raise DataValidationError("E4 capability report digest differs")

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_identity": self.model_identity,
            "runtime_identity": dict(self.runtime_identity),
            "runtime_artifact_sha256": self.runtime_artifact_sha256,
            "source_digests": dict(self.source_digests),
            "artifact_paths": dict(self.artifact_paths),
            "methods": [value.to_dict() for value in self.methods],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "report_digest": self.report_digest}

    @property
    def feasible_methods(self) -> tuple[str, ...]:
        return tuple(
            value.method
            for value in self.methods
            if value.feasibility is E4Feasibility.FEASIBLE
        )

    def assert_current(self) -> None:
        self.__post_init__()
        self.assert_artifacts(
            {key: Path(value) for key, value in self.artifact_paths.items()}
        )

    def assert_artifacts(self, artifact_paths: Mapping[str, str | Path]) -> None:
        """Revalidate the report against an explicitly packaged artifact inventory."""

        self.__post_init__()
        paths = {key: Path(value) for key, value in artifact_paths.items()}
        expected = {
            "runtime": self.runtime_artifact_sha256,
            **{f"source:{key}": value for key, value in self.source_digests.items()},
            **{
                f"evidence:{value.method}": value.evidence_artifact_sha256
                for value in self.methods
            },
            **{
                f"implementation:{value.method}": value.implementation_artifact_sha256
                for value in self.methods
                if value.implementation_artifact_sha256 is not None
            },
        }
        if set(paths) != set(expected):
            raise FrozenArtifactError("E4 capability receipt paths differ")
        for name, fingerprint in expected.items():
            assert fingerprint is not None
            try:
                observed = sha256_path(paths[name])
            except (OSError, FrozenArtifactError) as exc:
                raise FrozenArtifactError(
                    f"cannot rehash E4 capability artifact {name}: {exc}"
                ) from exc
            if observed != fingerprint:
                raise FrozenArtifactError(f"E4 capability artifact changed: {name}")
        for capability in self.methods:
            evidence_path = paths[f"evidence:{capability.method}"]
            try:
                evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise FrozenArtifactError(
                    f"cannot load E4 structured preflight for {capability.method}: {exc}"
                ) from exc
            if not isinstance(evidence, dict):
                raise FrozenArtifactError("E4 structured preflight must be a JSON object")
            digest = evidence.pop("evidence_digest", None)
            checks = evidence.get("checks")
            if (
                set(evidence)
                != {
                    "schema_version",
                    "method",
                    "model_identity",
                    "model_runtime_identity",
                    "runtime_artifact_sha256",
                    "feasibility",
                    "checks",
                    "details",
                }
                or evidence.get("schema_version") != 1
                or evidence.get("method") != capability.method
                or evidence.get("model_identity") != self.model_identity
                or evidence.get("model_runtime_identity")
                != dict(self.runtime_identity)
                or evidence.get("runtime_artifact_sha256")
                != self.runtime_artifact_sha256
                or not isinstance(checks, dict)
                or not isinstance(evidence.get("details"), dict)
                or set(checks) != _PREFLIGHT_CHECKS[capability.method]
                or any(type(value) is not bool for value in checks.values())
                or evidence.get("feasibility")
                != (
                    E4Feasibility.FEASIBLE.value
                    if all(checks.values())
                    else E4Feasibility.INFEASIBLE.value
                )
                or evidence.get("feasibility") != capability.feasibility.value
                or digest != stable_hash(evidence)
            ):
                raise FrozenArtifactError(
                    f"E4 structured preflight differs for {capability.method}"
                )
        e3_source = paths["source:E3_static_vectors"]
        m1 = next(value for value in self.methods if value.method == "M1")
        if (
            m1.implementation_artifact_sha256
            != self.source_digests["E3_static_vectors"]
        ):
            raise FrozenArtifactError(
                "E4 M1 implementation is not the exact frozen E3 vector bundle"
            )
        # Resolve one mandatory M1 coordinate to validate the portable vector schema.
        resolved_m1 = resolve_static_direction(
            e3_source,
            method="M1",
            layer=max(int(value) for value in json.loads(
                (e3_source / "metadata.json").read_text(encoding="utf-8")
            )["layer_axis"]),
            site=ActivationSite.POST_MLP,
        )
        if resolved_m1.direction.numel() != 5_120:
            raise FrozenArtifactError("E4 M1 direction width differs from Qwen hidden size")
        m2_path = paths["implementation:M2"]
        try:
            e3_metadata = json.loads(
                (e3_source / "metadata.json").read_text(encoding="utf-8")
            )
            m2_plan = json.loads((m2_path / "plan.json").read_text(encoding="utf-8"))
            m2_manifest = json.loads(
                (m2_path / "manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise FrozenArtifactError(
                f"cannot load E4 M1/M2 construction lineage: {exc}"
            ) from exc
        if (
            type(e3_metadata) is not dict
            or type(m2_plan) is not dict
            or type(m2_manifest) is not dict
            or m2_plan.get("source_e3_plan_identity")
            != e3_metadata.get("plan_identity")
            or m2_plan.get("source_e3_generation_chain_head")
            != e3_metadata.get("generation_chain_head")
            or m2_plan.get("source_e3_scientific_eligible") is not True
            or m2_manifest.get("scientific_eligible") is not True
        ):
            raise FrozenArtifactError(
                "E4 M2 does not descend from the frozen scientific E3 construction"
            )
        resolve_static_direction(
            m2_path,
            method="M2",
            layer=max(int(value) for value in m2_plan["protocol"]["layers"]),
            site=ActivationSite.BLOCK_OUTPUT,
        )
        act = verify_e4_act_baseline(paths["implementation:ACT-or-SADI"])
        if (
            act.source_e2_sha256 != self.source_digests["E2_calibrated_probes"]
            or act.source_m2_sha256 != self.methods[1].implementation_artifact_sha256
        ):
            raise FrozenArtifactError(
                "E4 ACT baseline does not descend from the frozen E2/M2 artifacts"
            )


def build_e4_capability_report(
    *,
    model_identity: str,
    runtime_identity: Mapping[str, Any],
    runtime_artifact: str | Path,
    source_artifacts: Mapping[str, str | Path],
    methods: Sequence[E4MethodCapability],
    method_evidence_artifacts: Mapping[str, str | Path],
    implementation_artifacts: Mapping[str, str | Path],
) -> E4CapabilityReport:
    frozen = tuple(methods)
    if (
        any(type(value) is not E4MethodCapability for value in frozen)
        or model_identity != ACTIVE_MODEL_NAME
        or dict(runtime_identity) != dict(_MODEL_RUNTIME_IDENTITY)
        or set(source_artifacts) != _REQUIRED_SOURCES
        or set(method_evidence_artifacts) != set(_METHODS)
        or set(implementation_artifacts)
        != {
            value.method
            for value in frozen
            if value.feasibility is E4Feasibility.FEASIBLE
        }
    ):
        raise DataValidationError("E4 capability artifact inventory differs")
    paths: dict[str, Path] = {"runtime": Path(runtime_artifact).resolve()}
    paths.update(
        {f"source:{key}": Path(value).resolve() for key, value in source_artifacts.items()}
    )
    paths.update(
        {
            f"evidence:{key}": Path(value).resolve()
            for key, value in method_evidence_artifacts.items()
        }
    )
    paths.update(
        {
            f"implementation:{key}": Path(value).resolve()
            for key, value in implementation_artifacts.items()
        }
    )
    observed = {key: sha256_path(value) for key, value in paths.items()}
    if tuple(value.method for value in frozen) != _METHODS or any(
        value.evidence_artifact_sha256 != observed[f"evidence:{value.method}"]
        or (
            value.implementation_artifact_sha256
            != observed.get(f"implementation:{value.method}")
        )
        for value in frozen
    ):
        raise DataValidationError("E4 capability declarations differ from live artifacts")
    sources = {key: observed[f"source:{key}"] for key in _REQUIRED_SOURCES}
    body = {
        "schema_version": 1,
        "model_identity": model_identity,
        "runtime_identity": dict(runtime_identity),
        "runtime_artifact_sha256": observed["runtime"],
        "source_digests": sources,
        "artifact_paths": {key: str(value) for key, value in paths.items()},
        "methods": [value.to_dict() for value in frozen],
    }
    report = E4CapabilityReport(
        model_identity=model_identity,
        runtime_identity=runtime_identity,
        runtime_artifact_sha256=observed["runtime"],
        source_digests=sources,
        artifact_paths={key: str(value) for key, value in paths.items()},
        methods=frozen,
        report_digest=stable_hash(body),
    )
    report.assert_current()
    return report


def load_e4_capability_report(
    path: str | Path, *, verify_live_artifacts: bool = True
) -> E4CapabilityReport:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        methods = tuple(
            E4MethodCapability(
                method=value["method"],
                feasibility=E4Feasibility(value["feasibility"]),
                implementation=value["implementation"],
                reason=value["reason"],
                evidence_artifact_sha256=value["evidence_artifact_sha256"],
                implementation_artifact_sha256=value[
                    "implementation_artifact_sha256"
                ],
            )
            for value in payload["methods"]
        )
        report = E4CapabilityReport(
            schema_version=payload["schema_version"],
            model_identity=payload["model_identity"],
            runtime_identity=payload["runtime_identity"],
            runtime_artifact_sha256=payload["runtime_artifact_sha256"],
            source_digests=payload["source_digests"],
            artifact_paths=payload["artifact_paths"],
            methods=methods,
            report_digest=payload["report_digest"],
        )
    except (
        OSError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        DataValidationError,
    ) as exc:
        raise FrozenArtifactError(f"cannot load E4 capability report: {exc}") from exc
    if verify_live_artifacts:
        report.assert_current()
    else:
        report.__post_init__()
    _exact_text(source, report.to_dict(), "E4 capability report")
    return report


def write_e4_capability_report(path: str | Path, report: E4CapabilityReport) -> None:
    report.assert_current()
    path = validate_active_study_artifact_paths(
        {"E4 capability report": path}
    )["E4 capability report"]
    _atomic_write(path, report.to_dict(), "E4 capability report")


def verify_e4_capability_report(
    path: str | Path, *, expected: E4CapabilityReport
) -> Mapping[str, Any]:
    expected.assert_current()
    fingerprint = _exact_text(path, expected.to_dict(), "E4 capability report")
    return MappingProxyType(
        {
            "valid": True,
            "report_digest": expected.report_digest,
            "artifact_sha256": fingerprint,
            "feasible_methods": expected.feasible_methods,
        }
    )


@dataclass(frozen=True, slots=True)
class E4MethodPolicy:
    method: str
    capability_report_digest: str
    implementation_artifact_sha256: str
    layer: int | None
    site: ActivationSite | None
    token_scope: TokenScope | None
    alpha: float
    adaptive_policy: AdaptivePolicySpec | None
    direction_sha256: str | None
    direction_norm: float | None
    reference_rms: float | None
    execution_public_key: str
    policy_digest: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        adaptive = self.method == "ACT-or-SADI"
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or self.method not in _METHODS
            or _SHA256.fullmatch(self.capability_report_digest) is None
            or _SHA256.fullmatch(self.implementation_artifact_sha256) is None
            or _SHA256.fullmatch(self.execution_public_key) is None
            or type(self.alpha) is not float
            or not math.isfinite(self.alpha)
            or adaptive != (self.adaptive_policy is not None)
            or (
                adaptive
                and (
                    (self.layer, self.site, self.token_scope, self.alpha)
                    != (None, None, None, 0.0)
                    or self.adaptive_policy is None
                    or self.adaptive_policy.controller_artifact_sha256
                    != self.implementation_artifact_sha256
                    or self.direction_sha256 is not None
                    or self.direction_norm is not None
                    or self.reference_rms is not None
                    or self.adaptive_policy.execution_public_key
                    != self.execution_public_key
                )
            )
            or (
                not adaptive
                and (
                    type(self.layer) is not int
                    or self.layer < 0
                    or self.layer >= int(_ACTIVE_MODEL["num_layers"])
                    or not isinstance(self.site, ActivationSite)
                    or not isinstance(self.token_scope, TokenScope)
                    or self.alpha == 0.0
                    or self.adaptive_policy is not None
                    or type(self.direction_sha256) is not str
                    or _SHA256.fullmatch(self.direction_sha256) is None
                    or isinstance(self.direction_norm, bool)
                    or not isinstance(self.direction_norm, int | float)
                    or not math.isfinite(float(self.direction_norm))
                    or float(self.direction_norm) <= 0
                    or isinstance(self.reference_rms, bool)
                    or not isinstance(self.reference_rms, int | float)
                    or not math.isfinite(float(self.reference_rms))
                    or float(self.reference_rms) <= 0
                    or (self.method == "M1" and self.site is not ActivationSite.POST_MLP)
                    or (self.method == "M2" and self.site is not ActivationSite.BLOCK_OUTPUT)
                )
            )
            or self.policy_digest != stable_hash(self._body())
        ):
            raise DataValidationError("E4 method policy is invalid")

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "capability_report_digest": self.capability_report_digest,
            "implementation_artifact_sha256": self.implementation_artifact_sha256,
            "layer": self.layer,
            "site": self.site.value if self.site is not None else None,
            "token_scope": (
                self.token_scope.value if self.token_scope is not None else None
            ),
            "alpha": self.alpha,
            "adaptive_policy": (
                self.adaptive_policy.to_dict()
                if self.adaptive_policy is not None
                else None
            ),
            "direction_sha256": self.direction_sha256,
            "direction_norm": self.direction_norm,
            "reference_rms": self.reference_rms,
            "execution_public_key": self.execution_public_key,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "policy_digest": self.policy_digest}


def write_e4_method_policy(
    path: str | Path,
    *,
    report: E4CapabilityReport,
    method: str,
    layer: int | None,
    site: ActivationSite | None,
    token_scope: TokenScope | None,
    alpha: float,
    execution_public_key: str,
    adaptive_policy: AdaptivePolicySpec | None = None,
    direction_sha256: str | None = None,
    direction_norm: float | None = None,
    reference_rms: float | None = None,
) -> E4MethodPolicy:
    path = validate_active_study_artifact_paths(
        {"E4 method policy": path}
    )["E4 method policy"]
    report.assert_current()
    capability = next((value for value in report.methods if value.method == method), None)
    if (
        capability is None
        or capability.feasibility is not E4Feasibility.FEASIBLE
        or capability.implementation_artifact_sha256 is None
    ):
        raise DataValidationError("E4 method policy requires a feasible capability")
    if method in {"M1", "M2"}:
        if (
            type(layer) is not int
            or layer < 0
            or layer >= int(_ACTIVE_MODEL["num_layers"])
            or not isinstance(site, ActivationSite)
            or not isinstance(token_scope, TokenScope)
            or type(alpha) is not float
            or not math.isfinite(alpha)
            or alpha == 0.0
        ):
            raise DataValidationError("E4 method policy has invalid static geometry")
        resolved = resolve_static_direction(
            report.artifact_paths[f"implementation:{method}"],
            method=method,
            layer=layer,
            site=site,
        )
        if (
            direction_sha256 not in {None, resolved.direction_sha256}
            or (
                direction_norm is not None
                and not math.isclose(
                    float(direction_norm), resolved.direction_norm, rel_tol=0, abs_tol=1e-7
                )
            )
            or (
                reference_rms is not None
                and not math.isclose(
                    float(reference_rms), resolved.reference_rms, rel_tol=0, abs_tol=1e-12
                )
            )
        ):
            raise DataValidationError("E4 static policy geometry differs from its construction")
        direction_sha256 = resolved.direction_sha256
        direction_norm = resolved.direction_norm
        reference_rms = resolved.reference_rms
    body = {
        "schema_version": 1,
        "method": method,
        "capability_report_digest": report.report_digest,
        "implementation_artifact_sha256": capability.implementation_artifact_sha256,
        "layer": layer,
        "site": site.value if site is not None else None,
        "token_scope": token_scope.value if token_scope is not None else None,
        "alpha": alpha,
        "adaptive_policy": adaptive_policy.to_dict() if adaptive_policy is not None else None,
        "direction_sha256": direction_sha256,
        "direction_norm": direction_norm,
        "reference_rms": reference_rms,
        "execution_public_key": execution_public_key,
    }
    policy = E4MethodPolicy(
        method=method,
        capability_report_digest=report.report_digest,
        implementation_artifact_sha256=capability.implementation_artifact_sha256,
        layer=layer,
        site=site,
        token_scope=token_scope,
        alpha=alpha,
        adaptive_policy=adaptive_policy,
        direction_sha256=direction_sha256,
        direction_norm=direction_norm,
        reference_rms=reference_rms,
        execution_public_key=execution_public_key,
        policy_digest=stable_hash(body),
    )
    _atomic_write(path, policy.to_dict(), "E4 method policy")
    return policy


def load_e4_method_policy(path: str | Path) -> E4MethodPolicy:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
        adaptive_value = value["adaptive_policy"]
        if adaptive_value is not None and not isinstance(adaptive_value, Mapping):
            raise TypeError("adaptive policy is not a mapping")
        policy = E4MethodPolicy(
            schema_version=value["schema_version"],
            method=value["method"],
            capability_report_digest=value["capability_report_digest"],
            implementation_artifact_sha256=value[
                "implementation_artifact_sha256"
            ],
            layer=value["layer"],
            site=ActivationSite(value["site"]) if value["site"] is not None else None,
            token_scope=(
                TokenScope(value["token_scope"])
                if value["token_scope"] is not None
                else None
            ),
            alpha=value["alpha"],
            adaptive_policy=(
                AdaptivePolicySpec.from_dict(adaptive_value)
                if adaptive_value is not None
                else None
            ),
            direction_sha256=value["direction_sha256"],
            direction_norm=value["direction_norm"],
            reference_rms=value["reference_rms"],
            execution_public_key=value["execution_public_key"],
            policy_digest=value["policy_digest"],
        )
    except (
        OSError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        DataValidationError,
    ) as exc:
        raise FrozenArtifactError(f"cannot load E4 method policy: {exc}") from exc
    expected = json.dumps(policy.to_dict(), indent=2, sort_keys=True) + "\n"
    if source.is_symlink() or source.read_text(encoding="utf-8") != expected:
        raise FrozenArtifactError("E4 method policy differs from exact replay")
    return policy


def e4_fixed_execution_receipt_body(
    record: GenerationRecord,
    *,
    policy: E4MethodPolicy,
    policy_artifact_sha256: str,
) -> dict[str, Any]:
    """Canonical runtime-signed proof for one fixed E4 hook execution."""

    return {
        "schema_version": 1,
        "question_id": record.question_id,
        "condition_id": record.condition_id,
        "rendered_prompt_hash": record.rendered_prompt_hash,
        "raw_output_sha256": hashlib.sha256(record.raw_output.encode()).hexdigest(),
        "normalized_answer_sha256": hashlib.sha256(
            record.normalized_answer.encode()
        ).hexdigest(),
        "outcome": record.outcome.value,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
        "generation_latency_seconds": record.generation_latency_seconds,
        "generation_runtime_metrics": record.metadata.get(
            "generation_runtime_metrics"
        ),
        "method_policy_sha256": policy_artifact_sha256,
        "policy_digest": policy.policy_digest,
        "implementation_artifact_sha256": policy.implementation_artifact_sha256,
        "intervention_trace": record.metadata.get("intervention_trace"),
    }


def sign_e4_fixed_execution_receipt(
    record: GenerationRecord,
    *,
    policy: E4MethodPolicy,
    policy_artifact_sha256: str,
    private_key_hex: str,
) -> str:
    if type(private_key_hex) is not str or _SHA256.fullmatch(private_key_hex) is None:
        raise DataValidationError("E4 execution private key must be 32-byte hex")
    try:
        private_key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_key_hex))
        signature = private_key.sign(
            canonical_json(
                e4_fixed_execution_receipt_body(
                    record,
                    policy=policy,
                    policy_artifact_sha256=policy_artifact_sha256,
                )
            ).encode()
        )
    except ValueError as exc:
        raise DataValidationError(f"invalid E4 execution private key: {exc}") from exc
    return signature.hex()


def validate_e4_fixed_execution_record(
    record: GenerationRecord,
    *,
    policy: E4MethodPolicy,
    policy_artifact_sha256: str,
) -> None:
    """Verify exact geometry, material activation edit, and runtime signature."""

    if policy.adaptive_policy is not None or record.steering_method != policy.method:
        raise DataValidationError("E4 fixed execution policy differs from the record")
    if _SHA256.fullmatch(policy_artifact_sha256) is None:
        raise DataValidationError("E4 method-policy artifact identity is invalid")
    trace = record.metadata.get("intervention_trace")
    expected_keys = {
        "method_policy_sha256",
        "implementation_artifact_sha256",
        "layer",
        "site",
        "token_scope",
        "alpha",
        "applied_tokens",
        "applied_token_indices",
        "activation_delta_norm",
        "direction_sha256",
        "direction_norm",
        "reference_rms",
        "pre_activation_sha256",
        "post_activation_sha256",
        "delta_sha256",
    }
    if not isinstance(trace, Mapping) or set(trace) != expected_keys:
        raise DataValidationError("E4 fixed record lacks its exact execution trace")
    assert policy.layer is not None
    assert policy.site is not None
    assert policy.token_scope is not None
    assert policy.direction_sha256 is not None
    assert policy.direction_norm is not None
    assert policy.reference_rms is not None
    expected_indices = (
        [-1]
        if policy.token_scope is TokenScope.FINAL_PROMPT
        else list(
            range(
                min(
                    {
                        TokenScope.FIRST_GENERATED: 1,
                        TokenScope.FIRST_FOUR: 4,
                        TokenScope.FIRST_EIGHT: 8,
                        TokenScope.ALL_GENERATED: record.output_tokens,
                        TokenScope.EXPONENTIAL_DECAY: record.output_tokens,
                    }[policy.token_scope],
                    record.output_tokens,
                )
            )
        )
    )
    expected_norm = (
        abs(policy.alpha)
        * policy.reference_rms
        * policy.direction_norm
        * math.sqrt(len(expected_indices))
    )
    trace_identity = {
        "method_policy_sha256": policy_artifact_sha256,
        "implementation_artifact_sha256": policy.implementation_artifact_sha256,
        "layer": policy.layer,
        "site": policy.site.value,
        "token_scope": policy.token_scope.value,
        "alpha": policy.alpha,
        "direction_sha256": policy.direction_sha256,
        "direction_norm": policy.direction_norm,
        "reference_rms": policy.reference_rms,
    }
    if (
        type(trace["method_policy_sha256"]) is not str
        or type(trace["implementation_artifact_sha256"]) is not str
        or type(trace["layer"]) is not int
        or type(trace["site"]) is not str
        or type(trace["token_scope"]) is not str
        or isinstance(trace["alpha"], bool)
        or not isinstance(trace["alpha"], int | float)
        or type(trace["direction_sha256"]) is not str
        or isinstance(trace["direction_norm"], bool)
        or not isinstance(trace["direction_norm"], int | float)
        or isinstance(trace["reference_rms"], bool)
        or not isinstance(trace["reference_rms"], int | float)
        or any(trace[name] != value for name, value in trace_identity.items())
        or type(trace["applied_tokens"]) is not int
        or type(trace["applied_token_indices"]) is not list
        or any(type(value) is not int for value in trace["applied_token_indices"])
        or trace["applied_tokens"] != len(expected_indices)
        or trace["applied_token_indices"] != expected_indices
        or not expected_indices
        or isinstance(trace["activation_delta_norm"], bool)
        or not isinstance(trace["activation_delta_norm"], int | float)
        or not math.isclose(
            float(trace["activation_delta_norm"]),
            expected_norm,
            rel_tol=0.025,
            abs_tol=1e-6,
        )
        or any(
            type(trace[name]) is not str or _SHA256.fullmatch(trace[name]) is None
            for name in (
                "pre_activation_sha256",
                "post_activation_sha256",
                "delta_sha256",
            )
        )
        or trace["pre_activation_sha256"] == trace["post_activation_sha256"]
        or record.metadata.get("intervention_trace_digest") != stable_hash(dict(trace))
    ):
        raise DataValidationError("E4 fixed execution trace does not prove the frozen edit")
    signature = record.metadata.get("execution_receipt_signature")
    if type(signature) is not str or re.fullmatch(r"[0-9a-f]{128}", signature) is None:
        raise DataValidationError("E4 fixed record lacks a runtime execution signature")
    try:
        public_key = Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(policy.execution_public_key)
        )
        public_key.verify(
            bytes.fromhex(signature),
            canonical_json(
                e4_fixed_execution_receipt_body(
                    record,
                    policy=policy,
                    policy_artifact_sha256=policy_artifact_sha256,
                )
            ).encode(),
        )
    except (InvalidSignature, ValueError) as exc:
        raise DataValidationError(
            "E4 fixed execution receipt was not signed by the frozen runtime"
        ) from exc


@dataclass(frozen=True, slots=True)
class E4Condition:
    method: str
    prompt_id: str
    implementation: str
    capability_report_digest: str
    model_identity_digest: str
    runtime_artifact_sha256: str
    implementation_artifact_sha256: str
    source_digest: str

    def __post_init__(self) -> None:
        if (
            self.method not in _METHODS
            or self.prompt_id not in _PROMPTS
            or type(self.implementation) is not str
            or not self.implementation.strip()
            or any(
                _SHA256.fullmatch(value) is None
                for value in (
                    self.capability_report_digest,
                    self.model_identity_digest,
                    self.runtime_artifact_sha256,
                    self.implementation_artifact_sha256,
                    self.source_digest,
                )
            )
        ):
            raise DataValidationError("E4 condition is invalid")

    def to_dict(self) -> dict[str, str]:
        return {
            "method": self.method,
            "prompt_id": self.prompt_id,
            "implementation": self.implementation,
            "capability_report_digest": self.capability_report_digest,
            "model_identity_digest": self.model_identity_digest,
            "runtime_artifact_sha256": self.runtime_artifact_sha256,
            "implementation_artifact_sha256": self.implementation_artifact_sha256,
            "source_digest": self.source_digest,
        }

    @property
    def condition_id(self) -> str:
        return stable_hash(self.to_dict())


def build_e4_conditions(report: E4CapabilityReport) -> tuple[E4Condition, ...]:
    report.assert_current()
    values: list[E4Condition] = []
    for capability in report.methods:
        if capability.feasibility is E4Feasibility.INFEASIBLE:
            continue
        assert capability.implementation is not None
        assert capability.implementation_artifact_sha256 is not None
        source_name = (
            "E3_static_vectors"
            if capability.method in {"M1", "M2"}
            else "E2_calibrated_probes"
        )
        for prompt_id in _PROMPTS:
            values.append(
                E4Condition(
                    method=capability.method,
                    prompt_id=prompt_id,
                    implementation=capability.implementation,
                    capability_report_digest=report.report_digest,
                    model_identity_digest=stable_hash(report.model_identity),
                    runtime_artifact_sha256=report.runtime_artifact_sha256,
                    implementation_artifact_sha256=capability.implementation_artifact_sha256,
                    source_digest=report.source_digests[source_name],
                )
            )
    return tuple(values)


@dataclass(frozen=True, slots=True)
class E4Promotion:
    source_contract_digest: str
    source_record_set_digest: str
    capability_report_digest: str
    screen_receipt_digest: str
    condition_metrics: Mapping[str, Mapping[str, Any]]
    selection_manifest: Mapping[str, Any]
    promoted_methods: tuple[str, ...]
    scientific_eligible: bool
    promotion_digest: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        metrics = {
            key: MappingProxyType(dict(value))
            for key, value in self.condition_metrics.items()
        }
        manifest = dict(self.selection_manifest)
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or any(
                _SHA256.fullmatch(value) is None
                for value in (
                    self.source_contract_digest,
                    self.source_record_set_digest,
                    self.capability_report_digest,
                    self.screen_receipt_digest,
                    self.promotion_digest,
                )
            )
            or set(manifest)
            != {
                "schema_version",
                "selection_rule",
                "source_contract_digest",
                "source_record_set_digest",
                "selected_condition_ids",
                "manifest_digest",
            }
            or manifest.get("schema_version") != 1
            or manifest.get("selection_rule") != _SELECTION_RULE
            or manifest.get("source_contract_digest") != self.source_contract_digest
            or manifest.get("source_record_set_digest")
            != self.source_record_set_digest
            or manifest.get("manifest_digest")
            != stable_hash(
                {
                    key: value
                    for key, value in manifest.items()
                    if key != "manifest_digest"
                }
            )
            or type(self.promoted_methods) is not tuple
            or any(value not in _METHODS or value == "M1" for value in self.promoted_methods)
            or len(set(self.promoted_methods)) != len(self.promoted_methods)
            or type(self.scientific_eligible) is not bool
        ):
            raise DataValidationError("E4 promotion is invalid")
        selected = manifest["selected_condition_ids"]
        if (
            not isinstance(selected, list)
            or selected != sorted(set(selected))
            or any(_SHA256.fullmatch(value) is None for value in selected)
        ):
            raise DataValidationError("E4 promotion selections are invalid")
        object.__setattr__(self, "condition_metrics", MappingProxyType(metrics))
        object.__setattr__(self, "selection_manifest", MappingProxyType(manifest))
        if self.promotion_digest != stable_hash(self._body()):
            raise DataValidationError("E4 promotion digest differs")

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_contract_digest": self.source_contract_digest,
            "source_record_set_digest": self.source_record_set_digest,
            "capability_report_digest": self.capability_report_digest,
            "screen_receipt_digest": self.screen_receipt_digest,
            "condition_metrics": {
                key: dict(value) for key, value in sorted(self.condition_metrics.items())
            },
            "selection_manifest": dict(self.selection_manifest),
            "promoted_methods": list(self.promoted_methods),
            "scientific_eligible": self.scientific_eligible,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "promotion_digest": self.promotion_digest}


def load_e4_promotion_artifact(path: str | Path) -> E4Promotion:
    """Load the immutable promotion receipt without trusting caller expectations."""

    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
        if type(value) is not dict:
            raise TypeError("promotion root is not an object")
        promotion = E4Promotion(
            schema_version=value["schema_version"],
            source_contract_digest=value["source_contract_digest"],
            source_record_set_digest=value["source_record_set_digest"],
            capability_report_digest=value["capability_report_digest"],
            screen_receipt_digest=value["screen_receipt_digest"],
            condition_metrics=value["condition_metrics"],
            selection_manifest=value["selection_manifest"],
            promoted_methods=tuple(value["promoted_methods"]),
            scientific_eligible=value["scientific_eligible"],
            promotion_digest=value["promotion_digest"],
        )
    except (
        OSError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        DataValidationError,
    ) as exc:
        raise FrozenArtifactError(f"cannot load E4 promotion artifact: {exc}") from exc
    _exact_text(source, promotion.to_dict(), "E4 promotion")
    return promotion


def _derive_e4_promotion(
    *,
    ledger_directory: str | Path,
    study: StudyProtocol,
    report: E4CapabilityReport,
    screen: E4ScreenReceipt,
    method_policy_artifacts: Mapping[str, str | Path],
    protocol: E4Protocol,
) -> E4Promotion:
    if type(study) is not StudyProtocol:
        raise DataValidationError("E4 promotion requires an exact frozen study protocol")
    ledger = PhaseRunLedger.open(ledger_directory, study=study)
    PhaseRunLedger._verify_creation_evidence(ledger)
    ledger.contract.assert_matches_study(study)
    report.assert_current()
    screen.assert_current()
    if ledger.contract.phase is not ExperimentPhase.E4:
        raise DataValidationError("E4 promotion requires an E4 phase ledger")
    completed, expected = PhaseRunLedger.progress(ledger)
    values = ledger.contract.conditions
    questions = screen.screen_question_ids
    if (
        completed != expected
        or len(questions) != protocol.screen_rows
        or screen.protocol_digest != stable_hash(protocol.to_dict())
        or ledger.contract.question_ids_by_benchmark != {"triviaqa": questions}
        or dict(ledger.contract.input_fingerprints) != dict(report.source_digests)
    ):
        raise DataValidationError("E4 ledger differs from verified screen or capability inputs")
    feasible = set(report.feasible_methods)
    if set(method_policy_artifacts) != feasible:
        raise DataValidationError("E4 method policy inventory differs from feasible methods")
    policy_paths = {
        key: Path(value).resolve() for key, value in method_policy_artifacts.items()
    }
    policies = {key: load_e4_method_policy(value) for key, value in policy_paths.items()}
    policy_fingerprints = {key: sha256_path(value) for key, value in policy_paths.items()}
    if any(
        policy.method != method
        or policy.capability_report_digest != report.report_digest
        or policy.implementation_artifact_sha256
        != next(
            value.implementation_artifact_sha256
            for value in report.methods
            if value.method == method
        )
        for method, policy in policies.items()
    ):
        raise DataValidationError("E4 method policy differs from capability report")
    observed_methods = {value.steering_method for value in values}
    if (
        observed_methods != feasible
        or any(
            value.model_name != report.model_identity
            or value.model_repository != report.runtime_identity["repository"]
            or value.model_revision != report.runtime_identity["revision"]
            or value.runtime.value != report.runtime_identity["runtime"]
            or value.quantization != report.runtime_identity["quantization"]
            or value.model_num_layers != report.runtime_identity["num_layers"]
            or value.benchmark != "triviaqa"
            or value.partition != "T-dev-screen-2000"
            or value.system_prompt_id not in _PROMPTS
            for value in values
        )
    ):
        raise DataValidationError("E4 ledger conditions differ from capability report")
    if any(
        value.method_artifact_sha256
        != policy_fingerprints[value.steering_method]
        or value.layer != policies[value.steering_method].layer
        or value.site is not policies[value.steering_method].site
        or value.token_scope is not policies[value.steering_method].token_scope
        or value.alpha != policies[value.steering_method].alpha
        or value.adaptive_policy != policies[value.steering_method].adaptive_policy
        for value in values
    ):
        raise DataValidationError("E4 conditions differ from frozen method policies")
    records = tuple(PhaseRunLedger.records(ledger))
    record_index = {(value.condition_id, value.question_id): value for value in records}
    expected_rows = {
        (condition.condition_id, question)
        for condition in values
        for question in questions
    }
    if (
        len(record_index) != len(records)
        or set(record_index) != expected_rows
        or any(
            type(value) is not GenerationRecord
            or type(value.outcome) is not Outcome
            or value.outcome is Outcome.UNSCORABLE
            for value in records
        )
    ):
        raise DataValidationError("E4 ledger does not contain the exact scorable screen")
    for record in records:
        policy = policies[record.steering_method]
        if policy.adaptive_policy is None:
            validate_e4_fixed_execution_record(
                record,
                policy=policy,
                policy_artifact_sha256=policy_fingerprints[record.steering_method],
            )
    validate_adaptive_execution(records)
    metrics: dict[str, Mapping[str, Any]] = {}
    grouped: dict[
        tuple[str, str, str, str],
        list[tuple[Any, float, float]],
    ] = defaultdict(list)
    for condition in values:
        bundle = metric_bundle(
            record_index[(condition.condition_id, question)].outcome
            for question in questions
        ).to_dict()
        raw_coverage = bundle["coverage"]
        raw_risk = bundle["hallucination_risk"]
        if not isinstance(raw_coverage, int | float) or not isinstance(raw_risk, int | float):
            raise DataValidationError("E4 condition has undefined coverage or risk")
        coverage = float(raw_coverage)
        risk = float(raw_risk)
        metrics[condition.condition_id] = MappingProxyType(
            {
                **bundle,
                "method": condition.steering_method,
                "prompt_id": condition.system_prompt_id,
            }
        )
        grouped[
            (
                condition.model_repository,
                condition.benchmark,
                condition.system_prompt_id,
                condition.partition,
            )
        ].append(
            (condition, coverage, risk)
        )
    winners: list[Any] = []
    for rows in grouped.values():
        baselines = [value for value in rows if value[0].steering_method == "M1"]
        if not baselines:
            raise DataValidationError("E4 promotion stratum lacks its M1 baseline")
        candidates: list[tuple[Any, float, float]] = []
        for candidate in rows:
            matching = [
                value
                for value in baselines
                if value[0].comparison_group == candidate[0].comparison_group
            ]
            if not matching and len(baselines) == 1:
                matching = baselines
            if len(matching) != 1:
                raise DataValidationError(
                    "E4 candidate lacks exactly one group-matched M1 baseline"
                )
            if candidate[1] >= matching[0][1] - protocol.maximum_coverage_loss:
                candidates.append(candidate)
        if not candidates:
            raise DataValidationError("E4 promotion stratum has no coverage-eligible candidate")
        if not any(value[0].steering_method == "M2" for value in candidates):
            raise DataValidationError(
                "E4 mandatory M2 exceeds the frozen M1 coverage-loss bound"
            )
        rank = lambda value: (  # noqa: E731 - shared frozen ordering key
            value[2],
            -value[1],
            value[0].steering_method,
            value[0].condition_id,
        )
        winners.append(min(candidates, key=rank)[0])
        by_method: dict[str, list[tuple[Any, float, float]]] = defaultdict(list)
        for candidate in candidates:
            by_method[candidate[0].steering_method].append(candidate)
        winners.extend(min(method_rows, key=rank)[0] for method_rows in by_method.values())
    winners = list({value.condition_id: value for value in winners}.values())
    selected_ids = sorted(value.condition_id for value in winners)
    source_contract_digest = ledger.contract.digest
    source_record_set_digest = PhaseRunLedger.record_set_digest(ledger)
    manifest_body: dict[str, Any] = {
        "schema_version": 1,
        "selection_rule": _SELECTION_RULE,
        "source_contract_digest": source_contract_digest,
        "source_record_set_digest": source_record_set_digest,
        "selected_condition_ids": selected_ids,
    }
    manifest = {**manifest_body, "manifest_digest": stable_hash(manifest_body)}
    promoted = tuple(
        sorted(
            {
                value.steering_method
                for value in winners
                if value.steering_method != "M1"
            }
        )
    )
    scientific = bool(
        protocol.scientific_eligible
        and screen.scientific_eligible
        and completed == expected
    )
    body: dict[str, Any] = {
        "schema_version": 1,
        "source_contract_digest": source_contract_digest,
        "source_record_set_digest": source_record_set_digest,
        "capability_report_digest": report.report_digest,
        "screen_receipt_digest": screen.receipt_digest,
        "condition_metrics": {
            key: dict(value) for key, value in sorted(metrics.items())
        },
        "selection_manifest": manifest,
        "promoted_methods": list(promoted),
        "scientific_eligible": scientific,
    }
    return E4Promotion(
        source_contract_digest=source_contract_digest,
        source_record_set_digest=source_record_set_digest,
        capability_report_digest=report.report_digest,
        screen_receipt_digest=screen.receipt_digest,
        condition_metrics=metrics,
        selection_manifest=manifest,
        promoted_methods=promoted,
        scientific_eligible=scientific,
        promotion_digest=stable_hash(body),
    )


def derive_e4_promotion(
    *,
    ledger_directory: str | Path,
    study: StudyProtocol,
    report: E4CapabilityReport,
    screen: E4ScreenReceipt,
    method_policy_artifacts: Mapping[str, str | Path],
    protocol: E4Protocol | None = None,
) -> E4Promotion:
    return _derive_e4_promotion(
        ledger_directory=ledger_directory,
        study=study,
        report=report,
        screen=screen,
        method_policy_artifacts=method_policy_artifacts,
        protocol=_protocol(protocol),
    )


def write_e4_promotion(
    path: str | Path,
    promotion: E4Promotion,
    *,
    ledger_directory: str | Path,
    study: StudyProtocol,
    report: E4CapabilityReport,
    screen: E4ScreenReceipt,
    method_policy_artifacts: Mapping[str, str | Path],
    protocol: E4Protocol | None = None,
) -> None:
    normalized = validate_active_study_artifact_paths(
        {"E4 promotion": path, "E4 phase ledger": ledger_directory}
    )
    path = normalized["E4 promotion"]
    ledger_directory = normalized["E4 phase ledger"]
    replayed = derive_e4_promotion(
        ledger_directory=ledger_directory,
        study=study,
        report=report,
        screen=screen,
        method_policy_artifacts=method_policy_artifacts,
        protocol=protocol,
    )
    promotion.__post_init__()
    if replayed != promotion:
        raise FrozenArtifactError("E4 promotion differs from verified ledger replay")
    _atomic_write(path, promotion.to_dict(), "E4 promotion")


def verify_e4_promotion(
    path: str | Path,
    *,
    ledger_directory: str | Path,
    study: StudyProtocol,
    report: E4CapabilityReport,
    screen: E4ScreenReceipt,
    method_policy_artifacts: Mapping[str, str | Path],
    protocol: E4Protocol | None = None,
) -> Mapping[str, Any]:
    expected = derive_e4_promotion(
        ledger_directory=ledger_directory,
        study=study,
        report=report,
        screen=screen,
        method_policy_artifacts=method_policy_artifacts,
        protocol=protocol,
    )
    fingerprint = _exact_text(path, expected.to_dict(), "E4 promotion")
    return MappingProxyType(
        {
            "valid": True,
            "promotion_digest": expected.promotion_digest,
            "artifact_sha256": fingerprint,
            "selected_condition_ids": tuple(
                expected.selection_manifest["selected_condition_ids"]
            ),
            "promoted_methods": expected.promoted_methods,
            "scientific_eligible": expected.scientific_eligible,
        }
    )


def build_e4_promotion_gate_bundle(
    *,
    capability_report_path: str | Path,
    screen_receipt_path: str | Path,
    promotion_path: str | Path,
    method_policy_artifacts: Mapping[str, str | Path],
) -> tuple[Mapping[str, Any], Mapping[str, Path]]:
    """Package every artifact the registered E4 gate must replay independently."""

    report_path = Path(capability_report_path).resolve()
    screen_path = Path(screen_receipt_path).resolve()
    selected_path = Path(promotion_path).resolve()
    report = load_e4_capability_report(report_path)
    screen = load_e4_screen_receipt(screen_path)
    promotion = load_e4_promotion_artifact(selected_path)
    feasible = set(report.feasible_methods)
    if set(method_policy_artifacts) != feasible:
        raise DataValidationError("E4 gate policy artifact inventory differs")
    policy_paths = {
        method: Path(path).resolve() for method, path in method_policy_artifacts.items()
    }
    policies = {method: load_e4_method_policy(path) for method, path in policy_paths.items()}
    if (
        promotion.capability_report_digest != report.report_digest
        or promotion.screen_receipt_digest != screen.receipt_digest
        or any(
            policy.method != method
            or policy.capability_report_digest != report.report_digest
            for method, policy in policies.items()
        )
    ):
        raise DataValidationError("E4 gate artifacts are not mutually bound")

    def safe(value: str) -> str:
        return re.sub(r"[^a-z0-9_.-]+", "-", value.lower()).strip("-")

    primary = {
        "capability_report": "capability-report",
        "screen_receipt": "screen-receipt",
        "promotion": "promotion",
    }
    policy_names = {
        method: f"policy-{safe(method)}" for method in sorted(policy_paths)
    }
    report_names = {
        key: f"report-{safe(key)}" for key in sorted(report.artifact_paths)
    }
    paths: dict[str, Path] = {
        primary["capability_report"]: report_path,
        primary["screen_receipt"]: screen_path,
        primary["promotion"]: selected_path,
        **{policy_names[key]: value for key, value in policy_paths.items()},
        **{
            report_names[key]: Path(value).resolve()
            for key, value in report.artifact_paths.items()
        },
    }
    if len(paths) != 3 + len(policy_paths) + len(report.artifact_paths):
        raise DataValidationError("E4 gate artifact names collide")
    fingerprints = {name: sha256_path(path) for name, path in paths.items()}
    manifest_body: dict[str, Any] = {
        "schema_version": 1,
        "primary": primary,
        "method_policies": policy_names,
        "report_artifacts": report_names,
        "fingerprints": fingerprints,
    }
    artifact_manifest = {
        **manifest_body,
        "manifest_digest": stable_hash(manifest_body),
    }
    parameters = MappingProxyType(
        {
            "selection_manifest": dict(promotion.selection_manifest),
            "artifact_manifest": artifact_manifest,
        }
    )
    return parameters, MappingProxyType(paths)

"""Content-addressed pass/fail evidence for phase gates."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from mfh.artifact_namespace import validate_active_study_artifact_paths
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.experiments.protocol import ExperimentPhase
from mfh.provenance import canonical_json, sha256_path, stable_hash

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EVIDENCE_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


def _metric_value(value: Any) -> str | int | float | bool:
    if isinstance(value, bool | str):
        if isinstance(value, str) and not value.strip():
            raise DataValidationError("gate metric text cannot be empty")
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise DataValidationError("gate metrics must be finite scalar JSON values")


@dataclass(frozen=True, slots=True)
class GateResult:
    phase: ExperimentPhase
    gate: str
    passed: bool
    contract_digest: str
    record_set_digest: str
    evaluator: str
    evaluator_revision: str
    metrics: Mapping[str, str | int | float | bool]
    artifact_fingerprints: Mapping[str, str]
    gate_digest: str
    schema_version: int = 1
    artifact_paths: Mapping[str, Path] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        gate = self.gate.strip()
        evaluator = self.evaluator.strip()
        if self.schema_version != 1 or _EVIDENCE_NAME.fullmatch(gate) is None or not evaluator:
            raise DataValidationError(
                "gate result requires schema version 1, a safe gate name, and evaluator"
            )
        if not isinstance(self.passed, bool):
            raise DataValidationError("gate result passed flag must be boolean")
        for value in (self.contract_digest, self.record_set_digest, self.evaluator_revision):
            if not _SHA256.fullmatch(value):
                raise DataValidationError("gate result identities must be SHA-256 fingerprints")
        metrics = {str(key).strip(): _metric_value(value) for key, value in self.metrics.items()}
        artifacts = {
            str(key).strip(): str(value) for key, value in self.artifact_fingerprints.items()
        }
        if not metrics or any(not key for key in metrics):
            raise DataValidationError("gate result requires named metrics")
        if not artifacts or any(
            _EVIDENCE_NAME.fullmatch(key) is None or not _SHA256.fullmatch(value)
            for key, value in artifacts.items()
        ):
            raise DataValidationError("gate result requires safe named artifact fingerprints")
        paths = {
            str(key).strip(): Path(value).resolve() for key, value in self.artifact_paths.items()
        }
        if paths and set(paths) != set(artifacts):
            raise DataValidationError(
                "gate artifact paths must correspond exactly to their fingerprints"
            )
        object.__setattr__(self, "gate", gate)
        object.__setattr__(self, "evaluator", evaluator)
        object.__setattr__(self, "metrics", MappingProxyType(metrics))
        object.__setattr__(self, "artifact_fingerprints", MappingProxyType(artifacts))
        object.__setattr__(self, "artifact_paths", MappingProxyType(paths))
        expected = stable_hash(self._body())
        if self.gate_digest != expected:
            raise DataValidationError("gate result digest differs from its evidence")

    @classmethod
    def create(
        cls,
        *,
        phase: ExperimentPhase | str,
        gate: str,
        passed: bool,
        contract_digest: str,
        record_set_digest: str,
        evaluator: str,
        evaluator_revision: str,
        metrics: Mapping[str, str | int | float | bool],
        artifact_paths: Mapping[str, str | Path],
    ) -> GateResult:
        artifacts: dict[str, str] = {}
        paths: dict[str, Path] = {}
        for name, path in artifact_paths.items():
            normalized_name = str(name).strip()
            if _EVIDENCE_NAME.fullmatch(normalized_name) is None:
                raise DataValidationError(f"gate artifact name is not a safe basename: {name!r}")
            resolved = Path(path).resolve()
            try:
                artifacts[normalized_name] = sha256_path(resolved)
            except (OSError, FrozenArtifactError) as exc:
                raise DataValidationError(
                    f"cannot fingerprint gate artifact {path}: {exc}"
                ) from exc
            paths[normalized_name] = resolved
        body = {
            "schema_version": 1,
            "phase": ExperimentPhase(phase).value,
            "gate": gate.strip(),
            "passed": passed,
            "contract_digest": contract_digest,
            "record_set_digest": record_set_digest,
            "evaluator": evaluator.strip(),
            "evaluator_revision": evaluator_revision,
            "metrics": dict(metrics),
            "artifact_fingerprints": artifacts,
        }
        return cls(
            phase=ExperimentPhase(phase),
            gate=gate,
            passed=passed,
            contract_digest=contract_digest,
            record_set_digest=record_set_digest,
            evaluator=evaluator,
            evaluator_revision=evaluator_revision,
            metrics=metrics,
            artifact_fingerprints=artifacts,
            gate_digest=stable_hash(body),
            artifact_paths=paths,
        )

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "phase": self.phase.value,
            "gate": self.gate,
            "passed": self.passed,
            "contract_digest": self.contract_digest,
            "record_set_digest": self.record_set_digest,
            "evaluator": self.evaluator,
            "evaluator_revision": self.evaluator_revision,
            "metrics": dict(self.metrics),
            "artifact_fingerprints": dict(self.artifact_fingerprints),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "gate_digest": self.gate_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GateResult:
        expected = {
            "schema_version",
            "phase",
            "gate",
            "passed",
            "contract_digest",
            "record_set_digest",
            "evaluator",
            "evaluator_revision",
            "metrics",
            "artifact_fingerprints",
            "gate_digest",
        }
        if set(value) != expected:
            raise FrozenArtifactError("gate-result keys differ from schema version 1")
        metrics = value["metrics"]
        artifacts = value["artifact_fingerprints"]
        if not isinstance(metrics, Mapping) or not isinstance(artifacts, Mapping):
            raise FrozenArtifactError("gate-result metrics and artifacts must be mappings")
        try:
            return cls(
                schema_version=int(value["schema_version"]),
                phase=ExperimentPhase(value["phase"]),
                gate=str(value["gate"]),
                passed=value["passed"],
                contract_digest=str(value["contract_digest"]),
                record_set_digest=str(value["record_set_digest"]),
                evaluator=str(value["evaluator"]),
                evaluator_revision=str(value["evaluator_revision"]),
                metrics={str(key): item for key, item in metrics.items()},
                artifact_fingerprints={str(key): str(item) for key, item in artifacts.items()},
                gate_digest=str(value["gate_digest"]),
            )
        except (TypeError, ValueError, DataValidationError) as exc:
            raise FrozenArtifactError(f"invalid gate result: {exc}") from exc


def write_gate_result(path: str | Path, result: GateResult) -> None:
    destination = validate_active_study_artifact_paths(
        {"gate-result": path}
    )["gate-result"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.stage-",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_name, 0o444)
        try:
            os.link(temporary_name, destination)
        except FileExistsError:
            if destination.is_symlink() or destination.read_bytes() != payload:
                raise FrozenArtifactError(
                    f"refusing to overwrite gate result: {destination}"
                ) from None
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def read_gate_result(path: str | Path) -> GateResult:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read gate result {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise FrozenArtifactError("gate result root must be a mapping")
    return GateResult.from_dict(value)


def validate_gate_metrics_json(metrics: Mapping[str, Any]) -> None:
    """Validate metric serializability before an evaluator writes its evidence."""

    canonical_json({str(key): _metric_value(value) for key, value in metrics.items()})

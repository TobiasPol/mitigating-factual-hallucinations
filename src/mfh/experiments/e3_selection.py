"""Deterministic, immutable E3 geometry/alpha/scope operating-point selection."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from mfh.contracts import ActivationSite, Outcome, TokenScope
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.evaluation.metrics import metric_bundle
from mfh.experiments.e3_schedule import (
    E3Condition,
    E3OperatingPoint,
    E3Protocol,
    e3_alpha_conditions,
    e3_geometry_conditions,
    e3_scope_conditions,
)
from mfh.experiments.model_selection import validate_active_study_artifact_paths
from mfh.provenance import sha256_file, stable_hash

_EXTRACTIONS = ("M1-R", "M1-P")
_STAGES = ("geometry", "alpha", "scope")
_VERIFIED_SELECTION = object()
_VERIFIED_RECEIPTS: dict[int, object] = {}


@dataclass(frozen=True, slots=True)
class E3CandidateMetrics:
    condition_id: str
    extraction_method: str
    layer: int
    site: ActivationSite
    standardized_alpha: float
    token_scope: str
    total: int
    attempted: int
    accuracy: float
    coverage: float
    hallucination_risk: float
    mean_actual_delta_norm: float
    coverage_eligible: bool
    promotion_eligible: bool
    rank: tuple[Any, ...] | None

    def __post_init__(self) -> None:
        numeric = (
            self.accuracy,
            self.coverage,
            self.hallucination_risk,
            self.mean_actual_delta_norm,
        )
        if (
            self.extraction_method not in _EXTRACTIONS
            or type(self.condition_id) is not str
            or len(self.condition_id) != 64
            or any(value not in "0123456789abcdef" for value in self.condition_id)
            or type(self.layer) is not int
            or not 0 <= self.layer < 64
            or not isinstance(self.site, ActivationSite)
            or type(self.standardized_alpha) is not float
            or not math.isfinite(self.standardized_alpha)
            or self.standardized_alpha < 0
            or self.token_scope not in {value.value for value in TokenScope}
            or type(self.total) is not int
            or self.total <= 0
            or type(self.attempted) is not int
            or not 0 <= self.attempted <= self.total
            or any(
                type(value) is not float
                or not math.isfinite(value)
                for value in numeric
            )
            or not all(0 <= float(value) <= 1 for value in numeric[:3])
            or self.mean_actual_delta_norm < 0
            or type(self.coverage_eligible) is not bool
            or type(self.promotion_eligible) is not bool
            or (
                self.promotion_eligible
                and (not self.coverage_eligible or self.rank is None)
            )
            or (not self.promotion_eligible and self.rank is not None)
            or (
                self.rank is not None
                and (
                    type(self.rank) is not tuple
                    or len(self.rank) not in {5, 6}
                    or any(
                        type(value) not in {int, float, str}
                        or (type(value) is float and not math.isfinite(value))
                        for value in self.rank
                    )
                )
            )
        ):
            raise DataValidationError("E3 candidate metric schema is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "extraction_method": self.extraction_method,
            "layer": self.layer,
            "site": self.site.value,
            "standardized_alpha": self.standardized_alpha,
            "token_scope": self.token_scope,
            "total": self.total,
            "attempted": self.attempted,
            "accuracy": self.accuracy,
            "coverage": self.coverage,
            "hallucination_risk": self.hallucination_risk,
            "mean_actual_delta_norm": self.mean_actual_delta_norm,
            "coverage_eligible": self.coverage_eligible,
            "promotion_eligible": self.promotion_eligible,
            "rank": list(self.rank) if self.rank is not None else None,
        }


@dataclass(frozen=True, slots=True)
class E3StageSelection:
    stage: str
    source_plan_identity: str
    evaluation_plan_identity: str
    record_chain_head: str
    record_set_digest: str
    predecessor_selection_digest: str | None
    question_ids_digest: str
    conditions_digest: str
    baseline_condition_id: str
    baseline_accuracy: float
    baseline_coverage: float
    baseline_hallucination_risk: float
    candidates: tuple[E3CandidateMetrics, ...]
    selected: Mapping[str, E3OperatingPoint]
    falsified: bool
    falsification_reason: str | None
    selection_rule_sha256: str
    scientific_eligible: bool
    selection_digest: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        digests = (
            self.source_plan_identity,
            self.evaluation_plan_identity,
            self.record_chain_head,
            self.record_set_digest,
            self.question_ids_digest,
            self.conditions_digest,
            self.selection_rule_sha256,
            self.selection_digest,
        )
        baseline_metrics = (
            self.baseline_accuracy,
            self.baseline_coverage,
            self.baseline_hallucination_risk,
        )
        if (
            self.stage not in _STAGES
            or type(self.schema_version) is not int
            or self.schema_version != 1
            or any(
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in digests
            )
            or (
                self.predecessor_selection_digest is not None
                and (
                    type(self.predecessor_selection_digest) is not str
                    or len(self.predecessor_selection_digest) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in self.predecessor_selection_digest
                    )
                )
            )
            or (self.stage == "geometry") != (self.predecessor_selection_digest is None)
            or type(self.baseline_condition_id) is not str
            or len(self.baseline_condition_id) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.baseline_condition_id
            )
            or any(
                type(value) is not float
                or not math.isfinite(value)
                or not 0 <= value <= 1
                for value in baseline_metrics
            )
            or type(self.candidates) is not tuple
            or any(not isinstance(value, E3CandidateMetrics) for value in self.candidates)
            or not self.candidates
            or len({value.condition_id for value in self.candidates}) != len(self.candidates)
            or type(self.falsified) is not bool
            or type(self.scientific_eligible) is not bool
        ):
            raise DataValidationError("E3 selection stage or schema is invalid")
        if set(self.selected) not in (set(), set(_EXTRACTIONS)):
            raise DataValidationError("E3 selection must contain both extractions or neither")
        if type(self.selected) not in {dict, MappingProxyType} or any(
            type(name) is not str or not isinstance(point, E3OperatingPoint)
            for name, point in self.selected.items()
        ):
            raise DataValidationError("E3 selected operating-point schema is invalid")
        if self.falsified != (not self.selected):
            raise DataValidationError("E3 selection falsification state differs from winners")
        if self.falsified != (self.falsification_reason is not None):
            raise DataValidationError("E3 selection falsification reason differs")
        if self.falsification_reason is not None and (
            type(self.falsification_reason) is not str
            or not self.falsification_reason.strip()
        ):
            raise DataValidationError("E3 selection falsification reason is invalid")
        if any(point.extraction_method != name for name, point in self.selected.items()):
            raise DataValidationError("E3 selected operating-point identities differ")
        object.__setattr__(self, "selected", MappingProxyType(dict(self.selected)))
        if self.selection_digest != stable_hash(self._body()):
            raise DataValidationError("E3 selection digest differs")

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stage": self.stage,
            "source_plan_identity": self.source_plan_identity,
            "evaluation_plan_identity": self.evaluation_plan_identity,
            "record_chain_head": self.record_chain_head,
            "record_set_digest": self.record_set_digest,
            "predecessor_selection_digest": self.predecessor_selection_digest,
            "question_ids_digest": self.question_ids_digest,
            "conditions_digest": self.conditions_digest,
            "baseline_condition_id": self.baseline_condition_id,
            "baseline_accuracy": self.baseline_accuracy,
            "baseline_coverage": self.baseline_coverage,
            "baseline_hallucination_risk": self.baseline_hallucination_risk,
            "candidates": [value.to_dict() for value in self.candidates],
            "selected": {
                name: {
                    "extraction_method": point.extraction_method,
                    "layer": point.layer,
                    "site": point.site.value,
                    "standardized_alpha": point.standardized_alpha,
                    "token_scope": point.token_scope.value,
                }
                for name, point in sorted(self.selected.items())
            },
            "falsified": self.falsified,
            "falsification_reason": self.falsification_reason,
            "selection_rule_sha256": self.selection_rule_sha256,
            "scientific_eligible": self.scientific_eligible,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "selection_digest": self.selection_digest}


@dataclass(frozen=True, slots=True)
class VerifiedE3StageSelection:
    """A deterministic selection artifact replayed from its complete stage evidence."""

    selection: E3StageSelection
    path: Path
    artifact_sha256: str
    _verification_token: object

    def __post_init__(self) -> None:
        if (
            self._verification_token is not _VERIFIED_SELECTION
            or not isinstance(self.selection, E3StageSelection)
            or type(self.artifact_sha256) is not str
            or len(self.artifact_sha256) != 64
        ):
            raise DataValidationError("E3 verified selection receipt is invalid")
        self._assert_artifact_current()

    def assert_current(self) -> None:
        if _VERIFIED_RECEIPTS.get(id(self)) is not self:
            raise FrozenArtifactError("E3 selection receipt was not verifier-authorized")
        self._assert_artifact_current()

    def _assert_artifact_current(self) -> None:
        expected_text = (
            json.dumps(
                self.selection.to_dict(), indent=2, sort_keys=True, allow_nan=False
            )
            + "\n"
        )
        try:
            source_text = self.path.read_text(encoding="utf-8")
        except OSError as exc:
            raise FrozenArtifactError(
                f"cannot reopen E3 verified selection artifact: {exc}"
            ) from exc
        if (
            self.path.is_symlink()
            or not self.path.is_file()
            or sha256_file(self.path) != self.artifact_sha256
            or source_text != expected_text
            or self.selection.selection_rule_sha256 != sha256_file(Path(__file__))
        ):
            raise FrozenArtifactError("E3 verified selection receipt or artifact changed")

    @property
    def stage(self) -> str:
        return self.selection.stage

    @property
    def selected(self) -> Mapping[str, E3OperatingPoint]:
        return self.selection.selected

    @property
    def falsified(self) -> bool:
        return self.selection.falsified

    @property
    def source_plan_identity(self) -> str:
        return self.selection.source_plan_identity

    @property
    def selection_digest(self) -> str:
        return self.selection.selection_digest

    @property
    def scientific_eligible(self) -> bool:
        return self.selection.scientific_eligible

    @property
    def question_ids_digest(self) -> str:
        return self.selection.question_ids_digest


def _sha256(value: str, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DataValidationError(f"E3 selection {label} must be a SHA-256")
    return value


def _validated_inputs(
    *,
    stage: str,
    conditions: Sequence[E3Condition],
    question_ids: Sequence[str],
    outcomes: Mapping[tuple[str, str], Outcome],
    actual_delta_norms: Mapping[tuple[str, str], float],
    protocol: E3Protocol,
    predecessor: E3StageSelection | None,
) -> tuple[
    tuple[E3Condition, ...],
    tuple[str, ...],
    dict[tuple[str, str], Outcome],
    dict[tuple[str, str], float],
]:
    frozen_conditions = tuple(conditions)
    questions = tuple(question_ids)
    if (
        stage not in _STAGES
        or not frozen_conditions
        or any(value.stage != stage for value in frozen_conditions)
        or len({value.condition_id for value in frozen_conditions})
        != len(frozen_conditions)
        or len(questions) != protocol.screen_rows
        or len(set(questions)) != len(questions)
        or any(type(value) is not str or not value.strip() for value in questions)
    ):
        raise DataValidationError("E3 selection condition or screen identity is invalid")
    baselines = [value for value in frozen_conditions if value.method == "M0"]
    if len(baselines) != 1:
        raise DataValidationError("E3 selection requires exactly one stage M0")
    if stage == "geometry":
        expected_conditions = e3_geometry_conditions(protocol)
        if predecessor is not None:
            raise DataValidationError("E3 geometry cannot have a predecessor selection")
    elif stage == "alpha":
        if predecessor is None or predecessor.stage != "geometry" or predecessor.falsified:
            raise DataValidationError("E3 alpha requires a successful geometry selection")
        expected_conditions = e3_alpha_conditions(predecessor.selected, protocol=protocol)
    else:
        if predecessor is None or predecessor.stage != "alpha" or predecessor.falsified:
            raise DataValidationError("E3 scope requires a successful alpha selection")
        expected_conditions = e3_scope_conditions(predecessor.selected, protocol=protocol)
    if frozen_conditions != expected_conditions:
        raise DataValidationError("E3 selection conditions differ from the frozen stage grid")
    expected = {
        (condition.condition_id, question)
        for condition in frozen_conditions
        for question in questions
    }
    normalized_outcomes: dict[tuple[str, str], Outcome] = {}
    for key, value in outcomes.items():
        if type(key) is not tuple or len(key) != 2:
            raise DataValidationError("E3 selection outcome key is invalid")
        normalized_outcomes[key] = Outcome(value)
    if set(normalized_outcomes) != expected:
        raise DataValidationError("E3 selection outcomes do not cover the exact stage matrix")
    if any(value is Outcome.UNSCORABLE for value in normalized_outcomes.values()):
        raise DataValidationError("E3 selection cannot contain unscorable outcomes")
    intervention_keys = {
        (condition.condition_id, question)
        for condition in frozen_conditions
        if condition.method != "M0"
        for question in questions
    }
    normalized_norms: dict[tuple[str, str], float] = {}
    for key, raw in actual_delta_norms.items():
        if (
            type(key) is not tuple
            or len(key) != 2
            or isinstance(raw, bool)
            or not isinstance(raw, int | float)
            or not math.isfinite(float(raw))
            or float(raw) < 0
        ):
            raise DataValidationError("E3 selection delta norm is invalid")
        normalized_norms[key] = float(raw)
    if set(normalized_norms) != intervention_keys:
        raise DataValidationError("E3 selection delta norms do not cover interventions")
    condition_by_id = {value.condition_id: value for value in frozen_conditions}
    for (condition_id, _question_id), norm_value in normalized_norms.items():
        alpha = float(condition_by_id[condition_id].standardized_alpha)
        if (alpha == 0) != (norm_value == 0):
            raise DataValidationError("E3 delta norm contradicts intervention strength")
    return frozen_conditions, questions, normalized_outcomes, normalized_norms


def _metric_values(
    values: Sequence[Outcome], *, allow_no_attempt: bool = False
) -> tuple[int, int, float, float, float]:
    metrics = metric_bundle(values)
    if (
        metrics.accuracy is None
        or metrics.coverage is None
        or (metrics.hallucination_risk is None and not allow_no_attempt)
    ):
        raise DataValidationError("E3 selection condition has no attempted-answer risk")
    return (
        metrics.total,
        metrics.attempted,
        float(metrics.accuracy),
        float(metrics.coverage),
        (
            float(metrics.hallucination_risk)
            if metrics.hallucination_risk is not None
            else 1.0
        ),
    )


def _candidate_rank(
    *,
    stage: str,
    condition: E3Condition,
    risk: float,
    coverage: float,
    accuracy: float,
    mean_delta: float,
    protocol: E3Protocol,
) -> tuple[Any, ...]:
    common: tuple[Any, ...] = (risk, -coverage, -accuracy)
    if stage == "geometry":
        return (*common, protocol.candidate_layers.index(condition.layer), condition.condition_id)
    if stage == "alpha":
        return (*common, mean_delta, float(condition.standardized_alpha), condition.condition_id)
    return (
        *common,
        mean_delta,
        protocol.token_scopes.index(condition.token_scope),
        condition.condition_id,
    )


def derive_e3_stage_selection(
    *,
    stage: str,
    conditions: Sequence[E3Condition],
    question_ids: Sequence[str],
    outcomes: Mapping[tuple[str, str], Outcome],
    actual_delta_norms: Mapping[tuple[str, str], float],
    source_plan_identity: str,
    evaluation_plan_identity: str,
    evaluation_record_chain_head: str,
    evaluation_record_set_digest: str,
    source_scientific_eligible: bool,
    predecessor_selection: E3StageSelection | None = None,
    protocol: E3Protocol | None = None,
) -> E3StageSelection:
    """Select M1-R/M1-P independently with the frozen 5pp coverage rule."""

    frozen_protocol = protocol or E3Protocol()
    if type(source_scientific_eligible) is not bool:
        raise DataValidationError("E3 source scientific eligibility must be boolean")
    source_identity = _sha256(source_plan_identity, "source plan")
    evaluation_identity = _sha256(evaluation_plan_identity, "evaluation plan")
    evaluation_chain_head = _sha256(
        evaluation_record_chain_head, "evaluation record chain head"
    )
    evaluation_record_set = _sha256(
        evaluation_record_set_digest, "evaluation record set"
    )
    frozen, questions, labels, delta_norms = _validated_inputs(
        stage=stage,
        conditions=conditions,
        question_ids=question_ids,
        outcomes=outcomes,
        actual_delta_norms=actual_delta_norms,
        protocol=frozen_protocol,
        predecessor=predecessor_selection,
    )
    question_ids_digest = stable_hash(list(questions))
    if predecessor_selection is not None and (
        predecessor_selection.source_plan_identity != source_identity
        or predecessor_selection.question_ids_digest != question_ids_digest
    ):
        raise DataValidationError(
            "E3 predecessor selection belongs to a different source run or screen"
        )
    baseline = next(value for value in frozen if value.method == "M0")
    baseline_values = [labels[(baseline.condition_id, question)] for question in questions]
    _total, _attempted, baseline_accuracy, baseline_coverage, baseline_risk = _metric_values(
        baseline_values
    )
    candidates: list[E3CandidateMetrics] = []
    eligible: dict[str, list[tuple[tuple[Any, ...], E3Condition]]] = {
        value: [] for value in _EXTRACTIONS
    }
    for condition in frozen:
        if condition.method == "M0":
            continue
        if condition.control is not None or condition.extraction_method not in _EXTRACTIONS:
            raise DataValidationError("E3 selection stages cannot contain causal controls")
        assert (
            condition.layer is not None
            and condition.site is not None
            and condition.token_scope is not None
        )
        values = [labels[(condition.condition_id, question)] for question in questions]
        total, attempted, accuracy, coverage, risk = _metric_values(
            values, allow_no_attempt=True
        )
        mean_delta = sum(
            delta_norms[(condition.condition_id, question)] for question in questions
        ) / len(questions)
        coverage_eligible = coverage >= baseline_coverage - 0.05
        stage_eligible = (
            (stage != "geometry" or condition.site is frozen_protocol.primary_replication_site)
            and (stage != "alpha" or condition.standardized_alpha > 0)
        )
        promotion_eligible = coverage_eligible and stage_eligible and attempted > 0
        rank = (
            _candidate_rank(
                stage=stage,
                condition=condition,
                risk=risk,
                coverage=coverage,
                accuracy=accuracy,
                mean_delta=mean_delta,
                protocol=frozen_protocol,
            )
            if promotion_eligible
            else None
        )
        candidate = E3CandidateMetrics(
            condition_id=condition.condition_id,
            extraction_method=condition.extraction_method,
            layer=int(condition.layer),
            site=condition.site,
            standardized_alpha=float(condition.standardized_alpha),
            token_scope=condition.token_scope.value,
            total=total,
            attempted=attempted,
            accuracy=accuracy,
            coverage=coverage,
            hallucination_risk=risk,
            mean_actual_delta_norm=mean_delta,
            coverage_eligible=coverage_eligible,
            promotion_eligible=promotion_eligible,
            rank=rank,
        )
        candidates.append(candidate)
        if rank is not None:
            eligible[condition.extraction_method].append((rank, condition))
    winners: dict[str, E3OperatingPoint] = {}
    for extraction in _EXTRACTIONS:
        if eligible[extraction]:
            _rank, condition = min(eligible[extraction], key=lambda value: value[0])
            assert (
                condition.layer is not None
                and condition.site is not None
                and condition.token_scope is not None
            )
            winners[extraction] = E3OperatingPoint(
                extraction_method=extraction,
                layer=int(condition.layer),
                site=condition.site,
                standardized_alpha=float(condition.standardized_alpha),
                token_scope=condition.token_scope,
            )
    falsified = set(winners) != set(_EXTRACTIONS)
    reason = (
        "no-positive-coverage-eligible-candidate-for-both-extractions"
        if falsified
        else None
    )
    if falsified:
        winners = {}
    ordered_candidates = tuple(sorted(candidates, key=lambda value: value.condition_id))
    body: dict[str, Any] = {
        "schema_version": 1,
        "stage": stage,
        "source_plan_identity": source_identity,
        "evaluation_plan_identity": evaluation_identity,
        "record_chain_head": evaluation_chain_head,
        "record_set_digest": evaluation_record_set,
        "predecessor_selection_digest": (
            predecessor_selection.selection_digest
            if predecessor_selection is not None
            else None
        ),
        "question_ids_digest": question_ids_digest,
        "conditions_digest": stable_hash([value.to_dict() for value in frozen]),
        "baseline_condition_id": baseline.condition_id,
        "baseline_accuracy": baseline_accuracy,
        "baseline_coverage": baseline_coverage,
        "baseline_hallucination_risk": baseline_risk,
        "candidates": [value.to_dict() for value in ordered_candidates],
        "selected": {
            name: {
                "extraction_method": point.extraction_method,
                "layer": point.layer,
                "site": point.site.value,
                "standardized_alpha": point.standardized_alpha,
                "token_scope": point.token_scope.value,
            }
            for name, point in sorted(winners.items())
        },
        "falsified": falsified,
        "falsification_reason": reason,
        "selection_rule_sha256": sha256_file(Path(__file__)),
        "scientific_eligible": (
            frozen_protocol.scientific_eligible
            and source_scientific_eligible
            and (
                predecessor_selection is None
                or predecessor_selection.scientific_eligible
            )
        ),
    }
    return E3StageSelection(
        stage=stage,
        source_plan_identity=source_identity,
        evaluation_plan_identity=evaluation_identity,
        record_chain_head=evaluation_chain_head,
        record_set_digest=body["record_set_digest"],
        predecessor_selection_digest=body["predecessor_selection_digest"],
        question_ids_digest=body["question_ids_digest"],
        conditions_digest=body["conditions_digest"],
        baseline_condition_id=baseline.condition_id,
        baseline_accuracy=baseline_accuracy,
        baseline_coverage=baseline_coverage,
        baseline_hallucination_risk=baseline_risk,
        candidates=ordered_candidates,
        selected=winners,
        falsified=falsified,
        falsification_reason=reason,
        selection_rule_sha256=body["selection_rule_sha256"],
        scientific_eligible=(
            frozen_protocol.scientific_eligible and source_scientific_eligible
            and (
                predecessor_selection is None
                or predecessor_selection.scientific_eligible
            )
        ),
        selection_digest=stable_hash(body),
    )


def write_e3_stage_selection(
    path: str | Path,
    **inputs: Any,
) -> E3StageSelection:
    selection = derive_e3_stage_selection(**inputs)
    destination = validate_active_study_artifact_paths(
        {"E3 stage selection": path}
    )["E3 stage selection"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite E3 selection: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    selection.to_dict(), indent=2, sort_keys=True, allow_nan=False
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return selection


def verify_e3_stage_selection(
    path: str | Path,
    **inputs: Any,
) -> E3StageSelection:
    expected = derive_e3_stage_selection(**inputs)
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise FrozenArtifactError("E3 selection artifact must be a regular file")
    try:
        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key}")
                result[key] = item
            return result

        source_text = source.read_text(encoding="utf-8")
        value = json.loads(source_text, object_pairs_hook=reject_duplicates)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read E3 selection artifact: {exc}") from exc
    expected_text = (
        json.dumps(expected.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    if type(value) is not dict or source_text != expected_text:
        raise FrozenArtifactError("E3 selection artifact differs from deterministic replay")
    return expected


def load_verified_e3_stage_selection(
    path: str | Path,
    **inputs: Any,
) -> VerifiedE3StageSelection:
    """Return an authorized receipt only after exact artifact and evidence replay."""

    source = Path(path)
    selection = verify_e3_stage_selection(source, **inputs)
    receipt = VerifiedE3StageSelection(
        selection=selection,
        path=source.resolve(),
        artifact_sha256=sha256_file(source),
        _verification_token=_VERIFIED_SELECTION,
    )
    _VERIFIED_RECEIPTS[id(receipt)] = receipt
    receipt.assert_current()
    return receipt

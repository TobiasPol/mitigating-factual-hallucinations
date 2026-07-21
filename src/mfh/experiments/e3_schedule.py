"""Frozen staged E3 construction and evaluation schedules for native MLX."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from mfh.contracts import ActivationSite, Question, TokenScope
from mfh.data.splits import semantic_group_ids
from mfh.errors import DataValidationError
from mfh.experiments.e2_schedule import E2_LAYERS, E2_SITES
from mfh.provenance import stable_hash

_EXTRACTIONS = ("M1-R", "M1-P")
_CONSTRUCTION_PROMPTS = ("P0-neutral", "P2-calibrated-abstention")
_FINAL_PROMPTS = (*_CONSTRUCTION_PROMPTS, "P3-forced-answer")
_CONTROLS = (
    "shuffled-label",
    "random-norm",
    "opposite",
    "unrelated-layer",
    "gaussian",
    "zero-hook",
    "cross-prompt",
)
_SCOPES = (
    TokenScope.FINAL_PROMPT,
    TokenScope.FIRST_GENERATED,
    TokenScope.FIRST_FOUR,
    TokenScope.FIRST_EIGHT,
    TokenScope.ALL_GENERATED,
    TokenScope.EXPONENTIAL_DECAY,
)
_STAGES = frozenset(
    {"geometry", "alpha", "scope", "controls", "cross-prompt", "P3-diagnostic", "final"}
)


@dataclass(frozen=True, slots=True)
class E3Protocol:
    steer_rows: int = 30_000
    dev_rows: int = 5_000
    screen_rows: int = 500
    candidate_layers: tuple[int, ...] = E2_LAYERS
    candidate_sites: tuple[ActivationSite, ...] = E2_SITES
    standardized_alphas: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
    token_scopes: tuple[TokenScope, ...] = _SCOPES
    geometry_anchor_alpha: float = 0.5
    geometry_anchor_scope: TokenScope = TokenScope.FIRST_FOUR
    exponential_decay: float = 0.5
    response_pooling: str = "per-response-token-mean-then-class-centroid"
    prompt_extraction: str = "final-prompt-token"
    primary_replication_site: ActivationSite = ActivationSite.POST_MLP
    steer_split: str = "T-steer"
    dev_split: str = "T-dev"
    seed: int = 17
    schema_version: int = 1

    def __post_init__(self) -> None:
        layers = tuple(self.candidate_layers)
        sites = tuple(self.candidate_sites)
        raw_alphas = tuple(self.standardized_alphas)
        scopes = tuple(self.token_scopes)
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or any(
                type(value) is not int or value <= 0
                for value in (self.steer_rows, self.dev_rows, self.screen_rows)
            )
            or self.screen_rows >= self.dev_rows
            or type(self.seed) is not int
            or self.seed < 0
            or not layers
            or any(type(value) is not int or not 0 <= value < 64 for value in layers)
            or len(set(layers)) != len(layers)
            or not sites
            or any(not isinstance(value, ActivationSite) for value in sites)
            or len(set(sites)) != len(sites)
            or not raw_alphas
            or any(
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                or float(value) < 0
                for value in raw_alphas
            )
            or len({float(value) for value in raw_alphas}) != len(raw_alphas)
            or 0.0 not in {float(value) for value in raw_alphas}
            or not scopes
            or any(not isinstance(value, TokenScope) for value in scopes)
            or len(set(scopes)) != len(scopes)
            or isinstance(self.geometry_anchor_alpha, bool)
            or not isinstance(self.geometry_anchor_alpha, int | float)
            or self.geometry_anchor_alpha
            not in {float(value) for value in raw_alphas}
            or self.geometry_anchor_scope not in scopes
            or isinstance(self.exponential_decay, bool)
            or not isinstance(self.exponential_decay, int | float)
            or not math.isfinite(float(self.exponential_decay))
            or float(self.exponential_decay) <= 0
            or self.response_pooling
            != "per-response-token-mean-then-class-centroid"
            or self.prompt_extraction != "final-prompt-token"
            or self.primary_replication_site is not ActivationSite.POST_MLP
            or self.primary_replication_site not in sites
            or self.steer_split != "T-steer"
            or self.dev_split != "T-dev"
        ):
            raise DataValidationError("E3 protocol geometry, counts, or controls are invalid")
        object.__setattr__(self, "candidate_layers", layers)
        object.__setattr__(self, "candidate_sites", sites)
        object.__setattr__(
            self, "standardized_alphas", tuple(float(value) for value in raw_alphas)
        )
        object.__setattr__(self, "token_scopes", scopes)

    @property
    def scientific_eligible(self) -> bool:
        return self == E3Protocol()

    @property
    def construction_rows(self) -> int:
        return self.steer_rows * len(_CONSTRUCTION_PROMPTS)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "steer_rows": self.steer_rows,
            "dev_rows": self.dev_rows,
            "screen_rows": self.screen_rows,
            "candidate_layers": list(self.candidate_layers),
            "candidate_sites": [value.value for value in self.candidate_sites],
            "standardized_alphas": list(self.standardized_alphas),
            "token_scopes": [value.value for value in self.token_scopes],
            "geometry_anchor_alpha": self.geometry_anchor_alpha,
            "geometry_anchor_scope": self.geometry_anchor_scope.value,
            "exponential_decay": self.exponential_decay,
            "response_pooling": self.response_pooling,
            "prompt_extraction": self.prompt_extraction,
            "primary_replication_site": self.primary_replication_site.value,
            "steer_split": self.steer_split,
            "dev_split": self.dev_split,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class E3ConstructionRow:
    sequence: int
    question_id: str
    benchmark: str
    prompt_id: str
    semantic_group_id: str
    question_sha256: str
    aliases_sha256: str

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence < 0:
            raise DataValidationError("E3 construction sequence is invalid")
        if self.benchmark != "triviaqa" or self.prompt_id not in _CONSTRUCTION_PROMPTS:
            raise DataValidationError("E3 construction benchmark or prompt is invalid")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (self.question_id, self.semantic_group_id)
        ):
            raise DataValidationError("E3 construction identity is empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "question_id": self.question_id,
            "benchmark": self.benchmark,
            "prompt_id": self.prompt_id,
            "semantic_group_id": self.semantic_group_id,
            "question_sha256": self.question_sha256,
            "aliases_sha256": self.aliases_sha256,
        }


@dataclass(frozen=True, slots=True)
class E3OperatingPoint:
    extraction_method: str
    layer: int
    site: ActivationSite
    standardized_alpha: float
    token_scope: TokenScope

    def __post_init__(self) -> None:
        if (
            self.extraction_method not in _EXTRACTIONS
            or type(self.layer) is not int
            or not 0 <= self.layer < 64
            or not isinstance(self.site, ActivationSite)
            or isinstance(self.standardized_alpha, bool)
            or not isinstance(self.standardized_alpha, int | float)
            or not math.isfinite(float(self.standardized_alpha))
            or self.standardized_alpha <= 0
            or not isinstance(self.token_scope, TokenScope)
        ):
            raise DataValidationError("E3 operating point is invalid")


@dataclass(frozen=True, slots=True)
class E3Condition:
    stage: str
    method: str
    extraction_method: str | None
    training_prompt_id: str | None
    apply_prompt_id: str
    layer: int | None
    site: ActivationSite | None
    standardized_alpha: float
    token_scope: TokenScope | None
    source_layer: int | None = None
    source_site: ActivationSite | None = None
    control: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.standardized_alpha, bool)
            or not isinstance(self.standardized_alpha, int | float)
            or not math.isfinite(float(self.standardized_alpha))
        ):
            raise DataValidationError("E3 condition alpha is invalid")
        if self.method == "M0":
            if any(
                value is not None
                for value in (
                    self.extraction_method,
                    self.training_prompt_id,
                    self.layer,
                    self.site,
                    self.token_scope,
                    self.source_layer,
                    self.source_site,
                    self.control,
                )
            ) or self.standardized_alpha != 0:
                raise DataValidationError("E3 M0 condition cannot contain an intervention")
        elif (
            self.extraction_method not in _EXTRACTIONS
            or self.training_prompt_id not in _CONSTRUCTION_PROMPTS
            or type(self.layer) is not int
            or not isinstance(self.site, ActivationSite)
            or not isinstance(self.token_scope, TokenScope)
            or (self.control is not None and self.control not in _CONTROLS)
            or (self.control is None and self.method != self.extraction_method)
            or (
                self.control is not None
                and not (
                    (self.stage == "cross-prompt"
                    and self.control == "cross-prompt"
                    and self.method == self.extraction_method)
                    or (self.stage != "cross-prompt" and self.method == self.control)
                )
            )
        ):
            raise DataValidationError("E3 intervention condition is invalid")
        if self.method != "M0":
            source_layer = self.layer if self.source_layer is None else self.source_layer
            source_site = self.site if self.source_site is None else self.source_site
            if type(source_layer) is not int or not isinstance(source_site, ActivationSite):
                raise DataValidationError("E3 source geometry is invalid")
            if self.control == "unrelated-layer":
                if source_layer == self.layer or source_site is not self.site:
                    raise DataValidationError(
                        "E3 unrelated-layer must preserve a distinct source layer"
                    )
            elif source_layer != self.layer or source_site is not self.site:
                raise DataValidationError(
                    "E3 non-layer-control source and target geometry must match"
                )
            object.__setattr__(self, "source_layer", source_layer)
            object.__setattr__(self, "source_site", source_site)
        alpha = float(self.standardized_alpha)
        if self.method != "M0" and (
            (self.control == "opposite" and alpha >= 0)
            or (self.control == "zero-hook" and alpha != 0)
            or (
                self.control not in {"opposite", "zero-hook"}
                and not (self.stage == "alpha" and self.control is None)
                and alpha <= 0
            )
            or (
                self.stage == "alpha"
                and self.control is None
                and alpha < 0
            )
        ):
            raise DataValidationError("E3 condition alpha contradicts its causal semantics")
        if self.apply_prompt_id not in _FINAL_PROMPTS or self.stage not in _STAGES:
            raise DataValidationError("E3 condition stage or apply prompt is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "method": self.method,
            "extraction_method": self.extraction_method,
            "training_prompt_id": self.training_prompt_id,
            "apply_prompt_id": self.apply_prompt_id,
            "layer": self.layer,
            "site": self.site.value if self.site is not None else None,
            "standardized_alpha": self.standardized_alpha,
            "token_scope": self.token_scope.value if self.token_scope is not None else None,
            "source_layer": self.source_layer,
            "source_site": self.source_site.value if self.source_site is not None else None,
            "control": self.control,
        }

    @property
    def condition_id(self) -> str:
        return stable_hash(self.to_dict())


def _question_sha256(question: Question) -> str:
    return stable_hash(
        {
            "question_id": question.question_id,
            "benchmark": question.benchmark,
            "text": question.text,
        }
    )


def build_e3_construction_schedule(
    questions: Sequence[Question], *, protocol: E3Protocol | None = None
) -> tuple[E3ConstructionRow, ...]:
    protocol = protocol or E3Protocol()
    if len(questions) != protocol.steer_rows:
        raise DataValidationError("E3 T-steer count differs from the frozen protocol")
    if any(question.benchmark != "triviaqa" for question in questions):
        raise DataValidationError("E3 construction is restricted to TriviaQA")
    if any(question.split != protocol.steer_split for question in questions):
        raise DataValidationError("E3 construction requires the frozen T-steer split")
    if len({question.question_id for question in questions}) != len(questions):
        raise DataValidationError("E3 T-steer contains duplicate question IDs")
    groups = semantic_group_ids(questions)
    rows: list[E3ConstructionRow] = []
    for prompt_id in _CONSTRUCTION_PROMPTS:
        for question in questions:
            rows.append(
                E3ConstructionRow(
                    sequence=len(rows),
                    question_id=question.question_id,
                    benchmark=question.benchmark,
                    prompt_id=prompt_id,
                    semantic_group_id=groups[question.question_id],
                    question_sha256=_question_sha256(question),
                    aliases_sha256=stable_hash(list(question.aliases)),
                )
            )
    return tuple(rows)


def select_e3_screen_questions(
    questions: Sequence[Question], *, protocol: E3Protocol | None = None
) -> tuple[str, ...]:
    """Select exactly 500 T-dev rows without splitting a semantic group."""

    protocol = protocol or E3Protocol()
    if len(questions) != protocol.dev_rows:
        raise DataValidationError("E3 T-dev count differs from the frozen protocol")
    groups = semantic_group_ids(questions)
    members: dict[str, list[str]] = defaultdict(list)
    for question in questions:
        if question.benchmark != "triviaqa":
            raise DataValidationError("E3 screen is restricted to TriviaQA")
        if question.split != protocol.dev_split:
            raise DataValidationError("E3 screen requires the frozen T-dev split")
        members[groups[question.question_id]].append(question.question_id)
    ordered = sorted(
        members,
        key=lambda group: hashlib.sha256(f"{protocol.seed}:{group}".encode()).digest(),
    )
    predecessors: dict[int, tuple[int, str] | None] = {0: None}
    for group in ordered:
        size = len(members[group])
        for total in sorted(tuple(predecessors), reverse=True):
            candidate = total + size
            if candidate <= protocol.screen_rows and candidate not in predecessors:
                predecessors[candidate] = (total, group)
    if protocol.screen_rows not in predecessors:
        raise DataValidationError("E3 semantic groups cannot fill the exact screen size")
    selected_groups: set[str] = set()
    total = protocol.screen_rows
    while total:
        link = predecessors[total]
        assert link is not None
        total, group = link
        selected_groups.add(group)
    selected = tuple(
        question.question_id
        for question in questions
        if groups[question.question_id] in selected_groups
    )
    if len(selected) != protocol.screen_rows:
        raise DataValidationError("E3 screen selection count differs")
    return selected


def _m0(stage: str, prompt_id: str) -> E3Condition:
    return E3Condition(stage, "M0", None, None, prompt_id, None, None, 0.0, None)


def e3_geometry_conditions(protocol: E3Protocol | None = None) -> tuple[E3Condition, ...]:
    protocol = protocol or E3Protocol()
    values = [_m0("geometry", "P0-neutral")]
    values.extend(
        E3Condition(
            "geometry",
            extraction,
            extraction,
            "P0-neutral",
            "P0-neutral",
            layer,
            site,
            protocol.geometry_anchor_alpha,
            protocol.geometry_anchor_scope,
        )
        for extraction in _EXTRACTIONS
        for layer in protocol.candidate_layers
        for site in protocol.candidate_sites
    )
    return tuple(values)


def _points(
    operating_points: Mapping[str, E3OperatingPoint], protocol: E3Protocol
) -> Mapping[str, E3OperatingPoint]:
    if set(operating_points) != set(_EXTRACTIONS):
        raise DataValidationError("E3 requires one operating point per M1 extraction")
    for name, point in operating_points.items():
        if (
            point.extraction_method != name
            or point.layer not in protocol.candidate_layers
            or point.site not in protocol.candidate_sites
            or point.site is not protocol.primary_replication_site
            or point.standardized_alpha not in protocol.standardized_alphas
            or point.token_scope not in protocol.token_scopes
        ):
            raise DataValidationError("E3 operating point differs from the protocol grid")
    return MappingProxyType(dict(operating_points))


def e3_alpha_conditions(
    operating_points: Mapping[str, E3OperatingPoint],
    *,
    protocol: E3Protocol | None = None,
) -> tuple[E3Condition, ...]:
    protocol = protocol or E3Protocol()
    points = _points(operating_points, protocol)
    values = [_m0("alpha", "P0-neutral")]
    values.extend(
        E3Condition(
            "alpha",
            extraction,
            extraction,
            "P0-neutral",
            "P0-neutral",
            points[extraction].layer,
            points[extraction].site,
            alpha,
            protocol.geometry_anchor_scope,
        )
        for extraction in _EXTRACTIONS
        for alpha in protocol.standardized_alphas
    )
    return tuple(values)


def e3_scope_conditions(
    operating_points: Mapping[str, E3OperatingPoint],
    *,
    protocol: E3Protocol | None = None,
) -> tuple[E3Condition, ...]:
    protocol = protocol or E3Protocol()
    points = _points(operating_points, protocol)
    values = [_m0("scope", "P0-neutral")]
    values.extend(
        E3Condition(
            "scope",
            extraction,
            extraction,
            "P0-neutral",
            "P0-neutral",
            points[extraction].layer,
            points[extraction].site,
            points[extraction].standardized_alpha,
            scope,
        )
        for extraction in _EXTRACTIONS
        for scope in protocol.token_scopes
    )
    return tuple(values)


def e3_control_conditions(
    operating_points: Mapping[str, E3OperatingPoint],
    *,
    protocol: E3Protocol | None = None,
) -> tuple[E3Condition, ...]:
    protocol = protocol or E3Protocol()
    points = _points(operating_points, protocol)
    values = [_m0("controls", "P0-neutral")]
    for extraction in _EXTRACTIONS:
        point = points[extraction]
        values.append(
            E3Condition(
                "controls",
                extraction,
                extraction,
                "P0-neutral",
                "P0-neutral",
                point.layer,
                point.site,
                point.standardized_alpha,
                point.token_scope,
            )
        )
        unrelated = protocol.candidate_layers[
            (protocol.candidate_layers.index(point.layer) + 1)
            % len(protocol.candidate_layers)
        ]
        for control in _CONTROLS:
            values.append(
                E3Condition(
                    "controls",
                    control,
                    extraction,
                    (
                        "P2-calibrated-abstention"
                        if control == "cross-prompt"
                        else "P0-neutral"
                    ),
                    "P0-neutral",
                    unrelated if control == "unrelated-layer" else point.layer,
                    point.site,
                    (
                        -point.standardized_alpha
                        if control == "opposite"
                        else 0.0
                        if control == "zero-hook"
                        else point.standardized_alpha
                    ),
                    point.token_scope,
                    source_layer=(
                        point.layer if control == "unrelated-layer" else None
                    ),
                    source_site=(
                        point.site if control == "unrelated-layer" else None
                    ),
                    control=control,
                )
            )
    return tuple(values)


def e3_cross_prompt_conditions(
    operating_points: Mapping[str, E3OperatingPoint],
    *,
    protocol: E3Protocol | None = None,
) -> tuple[E3Condition, ...]:
    protocol = protocol or E3Protocol()
    points = _points(operating_points, protocol)
    values = [_m0("cross-prompt", prompt) for prompt in _CONSTRUCTION_PROMPTS]
    values.extend(
        E3Condition(
            "cross-prompt",
            extraction,
            extraction,
            train_prompt,
            apply_prompt,
            points[extraction].layer,
            points[extraction].site,
            points[extraction].standardized_alpha,
            points[extraction].token_scope,
            control=("cross-prompt" if train_prompt != apply_prompt else None),
        )
        for extraction in _EXTRACTIONS
        for train_prompt in _CONSTRUCTION_PROMPTS
        for apply_prompt in _CONSTRUCTION_PROMPTS
    )
    return tuple(values)


def e3_p3_conditions(
    operating_points: Mapping[str, E3OperatingPoint],
    *,
    protocol: E3Protocol | None = None,
) -> tuple[E3Condition, ...]:
    protocol = protocol or E3Protocol()
    points = _points(operating_points, protocol)
    return (
        _m0("P3-diagnostic", "P3-forced-answer"),
        *(
            E3Condition(
                "P3-diagnostic",
                extraction,
                extraction,
                "P0-neutral",
                "P3-forced-answer",
                points[extraction].layer,
                points[extraction].site,
                points[extraction].standardized_alpha,
                points[extraction].token_scope,
            )
            for extraction in _EXTRACTIONS
        ),
    )


def e3_final_conditions(
    operating_points: Mapping[str, E3OperatingPoint],
    *,
    protocol: E3Protocol | None = None,
) -> tuple[E3Condition, ...]:
    protocol = protocol or E3Protocol()
    points = _points(operating_points, protocol)
    values: list[E3Condition] = []
    for prompt in _FINAL_PROMPTS:
        values.append(_m0("final", prompt))
        for extraction in _EXTRACTIONS:
            point = points[extraction]
            values.append(
                E3Condition(
                    "final",
                    extraction,
                    extraction,
                    prompt if prompt in _CONSTRUCTION_PROMPTS else "P0-neutral",
                    prompt,
                    point.layer,
                    point.site,
                    point.standardized_alpha,
                    point.token_scope,
                )
            )
    primary = points["M1-R"]
    unrelated = protocol.candidate_layers[
        (protocol.candidate_layers.index(primary.layer) + 1)
        % len(protocol.candidate_layers)
    ]
    for control in _CONTROLS:
        values.append(
            E3Condition(
                "final",
                control,
                "M1-R",
                (
                    "P2-calibrated-abstention"
                    if control == "cross-prompt"
                    else "P0-neutral"
                ),
                "P0-neutral",
                unrelated if control == "unrelated-layer" else primary.layer,
                primary.site,
                (
                    -primary.standardized_alpha
                    if control == "opposite"
                    else 0.0
                    if control == "zero-hook"
                    else primary.standardized_alpha
                ),
                primary.token_scope,
                source_layer=(
                    primary.layer if control == "unrelated-layer" else None
                ),
                source_site=(
                    primary.site if control == "unrelated-layer" else None
                ),
                control=control,
            )
        )
    return tuple(values)


def e3_stage_row_counts(protocol: E3Protocol | None = None) -> Mapping[str, int]:
    protocol = protocol or E3Protocol()
    dummy = {
        extraction: E3OperatingPoint(
            extraction,
            protocol.candidate_layers[0],
            protocol.primary_replication_site,
            protocol.geometry_anchor_alpha,
            protocol.geometry_anchor_scope,
        )
        for extraction in _EXTRACTIONS
    }
    counts = {
        "geometry": len(e3_geometry_conditions(protocol)) * protocol.screen_rows,
        "alpha": len(e3_alpha_conditions(dummy, protocol=protocol)) * protocol.screen_rows,
        "scope": len(e3_scope_conditions(dummy, protocol=protocol)) * protocol.screen_rows,
        "controls": len(e3_control_conditions(dummy, protocol=protocol))
        * protocol.screen_rows,
        "cross-prompt": len(e3_cross_prompt_conditions(dummy, protocol=protocol))
        * protocol.screen_rows,
        "P3-diagnostic": len(e3_p3_conditions(dummy, protocol=protocol))
        * protocol.screen_rows,
        "final": len(e3_final_conditions(dummy, protocol=protocol)) * protocol.dev_rows,
    }
    return MappingProxyType(counts)

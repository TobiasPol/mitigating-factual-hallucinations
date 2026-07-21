"""Frozen E2 prompt-capture schedule and group-disjoint controller subdivision."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from mfh.contracts import ActivationSite, ModelSpec, Outcome, Question, Runtime
from mfh.data.splits import semantic_group_ids
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.experiments.activation_store import (
    ActivationStoreSpec,
    create_activation_store,
    verify_activation_store,
)
from mfh.experiments.model_selection import validate_active_study_artifact_paths
from mfh.provenance import sha256_file, stable_hash

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
E2_LAYERS = (16, 31, 32, 47, 48, 57, 63)
E2_SITES = (
    ActivationSite.POST_ATTENTION,
    ActivationSite.POST_MLP,
    ActivationSite.BLOCK_OUTPUT,
)


@dataclass(frozen=True, slots=True)
class E2CaptureProtocol:
    controller_rows: int = 5_000
    controller_calibration_rows: int = 1_000
    dev_rows: int = 5_000
    simpleqa_rows: int = 1_000
    aa_rows: int = 600
    seed: int = 17
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1 or any(
            type(value) is not int or value <= 0
            for value in (
                self.controller_rows,
                self.controller_calibration_rows,
                self.dev_rows,
                self.simpleqa_rows,
                self.aa_rows,
            )
        ):
            raise DataValidationError("E2 capture counts must be positive exact integers")
        if self.controller_calibration_rows >= self.controller_rows:
            raise DataValidationError("E2 calibration rows must be smaller than controller rows")
        if type(self.seed) is not int or self.seed < 0:
            raise DataValidationError("E2 capture seed must be a non-negative exact integer")

    @property
    def expected_capture_rows(self) -> int:
        return 2 * self.controller_rows + 2 * self.dev_rows + self.simpleqa_rows + self.aa_rows

    @property
    def expected_new_generations(self) -> int:
        return self.controller_rows + 2 * self.dev_rows

    @property
    def scientific_eligible(self) -> bool:
        return self == E2CaptureProtocol()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "controller_rows": self.controller_rows,
            "controller_calibration_rows": self.controller_calibration_rows,
            "dev_rows": self.dev_rows,
            "simpleqa_rows": self.simpleqa_rows,
            "aa_rows": self.aa_rows,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class E2ScheduleRow:
    sequence: int
    question_id: str
    benchmark: str
    source_partition: str
    feature_partition: str
    prompt_id: str
    semantic_group_id: str
    question_sha256: str
    aliases_sha256: str
    label_source: str
    outcome: Outcome | None

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence < 0:
            raise DataValidationError("E2 schedule sequence must be a non-negative exact integer")
        for name in (
            "question_id",
            "benchmark",
            "source_partition",
            "feature_partition",
            "prompt_id",
            "semantic_group_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise DataValidationError(f"E2 schedule {name} must be non-empty text")
            object.__setattr__(self, name, value.strip())
        if not _SHA256.fullmatch(self.question_sha256) or not _SHA256.fullmatch(
            self.aliases_sha256
        ):
            raise DataValidationError("E2 schedule question identity is invalid")
        if self.label_source not in {"E1", "generate"}:
            raise DataValidationError("E2 schedule label source is invalid")
        if (self.label_source == "E1") != (self.outcome is not None):
            raise DataValidationError("E2 schedule E1 labels require exactly one frozen outcome")
        if self.outcome is not None:
            object.__setattr__(self, "outcome", Outcome(self.outcome))

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "question_id": self.question_id,
            "benchmark": self.benchmark,
            "source_partition": self.source_partition,
            "feature_partition": self.feature_partition,
            "prompt_id": self.prompt_id,
            "semantic_group_id": self.semantic_group_id,
            "question_sha256": self.question_sha256,
            "aliases_sha256": self.aliases_sha256,
            "label_source": self.label_source,
            "outcome": self.outcome.value if self.outcome is not None else None,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> E2ScheduleRow:
        expected = {
            "sequence",
            "question_id",
            "benchmark",
            "source_partition",
            "feature_partition",
            "prompt_id",
            "semantic_group_id",
            "question_sha256",
            "aliases_sha256",
            "label_source",
            "outcome",
        }
        if set(value) != expected or type(value["sequence"]) is not int or any(
            type(value[name]) is not str
            for name in expected - {"sequence", "outcome"}
        ) or (value["outcome"] is not None and type(value["outcome"]) is not str):
            raise DataValidationError("E2 schedule row has invalid schema or JSON types")
        return cls(
            sequence=value["sequence"],
            question_id=value["question_id"],
            benchmark=value["benchmark"],
            source_partition=value["source_partition"],
            feature_partition=value["feature_partition"],
            prompt_id=value["prompt_id"],
            semantic_group_id=value["semantic_group_id"],
            question_sha256=value["question_sha256"],
            aliases_sha256=value["aliases_sha256"],
            label_source=value["label_source"],
            outcome=Outcome(value["outcome"]) if value["outcome"] is not None else None,
        )


@dataclass(frozen=True, slots=True)
class VerifiedE2Workspace:
    directory: Path
    plan_identity: str
    protocol: E2CaptureProtocol
    schedule: tuple[E2ScheduleRow, ...]
    input_fingerprints: Mapping[str, str]
    activation_spec: ActivationStoreSpec


def controller_feature_partitions(
    questions: Sequence[Question],
    *,
    calibration_rows: int,
    seed: int,
) -> Mapping[str, str]:
    """Choose an exact calibration subset without splitting semantic groups."""

    if (
        type(calibration_rows) is not int
        or calibration_rows <= 0
        or calibration_rows >= len(questions)
        or type(seed) is not int
        or seed < 0
    ):
        raise DataValidationError("controller subdivision counts or seed are invalid")
    group_by_question = semantic_group_ids(questions)
    members: dict[str, list[str]] = defaultdict(list)
    for question in questions:
        members[group_by_question[question.question_id]].append(question.question_id)
    ordered = sorted(
        members,
        key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).digest(),
    )
    predecessors: dict[int, tuple[int, str] | None] = {0: None}
    for group in ordered:
        size = len(members[group])
        for total in sorted(tuple(predecessors), reverse=True):
            candidate = total + size
            if candidate <= calibration_rows and candidate not in predecessors:
                predecessors[candidate] = (total, group)
    if calibration_rows not in predecessors:
        raise DataValidationError(
            "semantic groups cannot produce the exact controller calibration size"
        )
    selected: set[str] = set()
    total = calibration_rows
    while total:
        link = predecessors[total]
        assert link is not None
        total, group = link
        selected.add(group)
    result = {
        question.question_id: (
            "T-controller-calibration"
            if group_by_question[question.question_id] in selected
            else "T-controller-train"
        )
        for question in questions
    }
    if sum(value == "T-controller-calibration" for value in result.values()) != calibration_rows:
        raise DataValidationError("controller subdivision does not have its exact frozen size")
    return MappingProxyType(result)


def _question_identity(question: Question) -> tuple[str, str]:
    return stable_hash(
        {
            "question_id": question.question_id,
            "benchmark": question.benchmark,
            "text": question.text,
        }
    ), stable_hash(list(question.aliases))


def _validate_e2_schedule(
    schedule: Sequence[E2ScheduleRow], protocol: E2CaptureProtocol
) -> None:
    rows = tuple(schedule)
    blocks = (
        (protocol.controller_rows, "T-controller", "triviaqa", "P0-neutral", "E1"),
        (protocol.dev_rows, "T-dev", "triviaqa", "P0-neutral", "generate"),
        (
            protocol.controller_rows,
            "T-controller",
            "triviaqa",
            "P3-forced-answer",
            "generate",
        ),
        (protocol.dev_rows, "T-dev", "triviaqa", "P3-forced-answer", "generate"),
        (
            protocol.simpleqa_rows,
            "simpleqa-eval",
            "simpleqa_verified",
            "P0-neutral",
            "E1",
        ),
        (
            protocol.aa_rows,
            "aa-eval",
            "aa_omniscience_public_600",
            "P0-neutral",
            "E1",
        ),
    )
    if len(rows) != protocol.expected_capture_rows or any(
        not isinstance(row, E2ScheduleRow) or row.sequence != index
        for index, row in enumerate(rows)
    ):
        raise DataValidationError("E2 schedule cardinality or sequence differs")
    offset = 0
    for count, partition, benchmark, prompt_id, label_source in blocks:
        block = rows[offset : offset + count]
        if len(block) != count or any(
            (
                row.source_partition,
                row.benchmark,
                row.prompt_id,
                row.label_source,
            )
            != (partition, benchmark, prompt_id, label_source)
            for row in block
        ):
            raise DataValidationError("E2 schedule block semantics differ")
        offset += count
    if (
        sum(row.label_source == "generate" for row in rows)
        != protocol.expected_new_generations
        or len({(row.benchmark, row.question_id, row.prompt_id) for row in rows})
        != len(rows)
    ):
        raise DataValidationError("E2 schedule generation or question identities differ")
    controller_p0 = rows[: protocol.controller_rows]
    dev_p0_start = protocol.controller_rows
    dev_p0 = rows[dev_p0_start : dev_p0_start + protocol.dev_rows]
    controller_p3_start = dev_p0_start + protocol.dev_rows
    controller_p3 = rows[
        controller_p3_start : controller_p3_start + protocol.controller_rows
    ]
    dev_p3_start = controller_p3_start + protocol.controller_rows
    dev_p3 = rows[dev_p3_start : dev_p3_start + protocol.dev_rows]

    def identity(row: E2ScheduleRow) -> tuple[str, str, str, str, str]:
        return (
            row.question_id,
            row.semantic_group_id,
            row.question_sha256,
            row.aliases_sha256,
            row.feature_partition,
        )

    if (
        [identity(row) for row in controller_p0]
        != [identity(row) for row in controller_p3]
        or [identity(row) for row in dev_p0] != [identity(row) for row in dev_p3]
        or any(row.feature_partition != "T-dev" for row in (*dev_p0, *dev_p3))
    ):
        raise DataValidationError("E2 repeated TriviaQA prompt schedules differ")
    allowed_controller = {"T-controller-train", "T-controller-calibration"}
    if any(row.feature_partition not in allowed_controller for row in controller_p0):
        raise DataValidationError("E2 controller feature partition is invalid")
    calibration = {
        row.question_id
        for row in controller_p0
        if row.feature_partition == "T-controller-calibration"
    }
    train_groups = {
        row.semantic_group_id
        for row in controller_p0
        if row.feature_partition == "T-controller-train"
    }
    calibration_groups = {
        row.semantic_group_id
        for row in controller_p0
        if row.feature_partition == "T-controller-calibration"
    }
    if (
        len(calibration) != protocol.controller_calibration_rows
        or train_groups & calibration_groups
    ):
        raise DataValidationError("E2 controller calibration split differs")
    if {
        row.semantic_group_id for row in controller_p0
    } & {row.semantic_group_id for row in dev_p0}:
        raise DataValidationError("E2 controller and dev semantic groups overlap")
    for row in rows[dev_p3_start + protocol.dev_rows :]:
        if row.feature_partition != row.source_partition:
            raise DataValidationError("E2 OOD feature partition differs")


def build_e2_schedule(
    *,
    controller: Sequence[Question],
    dev: Sequence[Question],
    simpleqa: Sequence[Question],
    aa: Sequence[Question],
    e1_p0_outcomes: Mapping[tuple[str, str], Outcome],
    protocol: E2CaptureProtocol | None = None,
) -> tuple[E2ScheduleRow, ...]:
    protocol = protocol or E2CaptureProtocol()
    expected = {
        "T-controller": (controller, protocol.controller_rows, "triviaqa"),
        "T-dev": (dev, protocol.dev_rows, "triviaqa"),
        "simpleqa-eval": (simpleqa, protocol.simpleqa_rows, "simpleqa_verified"),
        "aa-eval": (aa, protocol.aa_rows, "aa_omniscience_public_600"),
    }
    for partition, (questions, count, benchmark) in expected.items():
        if len(questions) != count or any(
            question.benchmark != benchmark for question in questions
        ):
            raise DataValidationError(f"E2 {partition} questions differ from the frozen count")
    trivia_groups = semantic_group_ids((*controller, *dev))
    if {
        trivia_groups[question.question_id] for question in controller
    } & {trivia_groups[question.question_id] for question in dev}:
        raise DataValidationError("E2 controller and dev semantic groups overlap")
    controller_partitions = controller_feature_partitions(
        controller,
        calibration_rows=protocol.controller_calibration_rows,
        seed=protocol.seed,
    )
    rows: list[E2ScheduleRow] = []

    def append_rows(
        questions: Sequence[Question],
        *,
        source_partition: str,
        prompt_id: str,
        frozen_labels: bool,
    ) -> None:
        for question in questions:
            feature_partition = (
                controller_partitions[question.question_id]
                if source_partition == "T-controller"
                else source_partition
            )
            outcome = (
                e1_p0_outcomes.get((question.benchmark, question.question_id))
                if frozen_labels
                else None
            )
            if frozen_labels and outcome is None:
                raise DataValidationError("E2 schedule lacks a required E1 P0 outcome")
            question_sha, aliases_sha = _question_identity(question)
            group_id = (
                trivia_groups[question.question_id]
                if question.benchmark == "triviaqa"
                else stable_hash([question.benchmark, question.question_id])
            )
            rows.append(
                E2ScheduleRow(
                    sequence=len(rows),
                    question_id=question.question_id,
                    benchmark=question.benchmark,
                    source_partition=source_partition,
                    feature_partition=feature_partition,
                    prompt_id=prompt_id,
                    semantic_group_id=group_id,
                    question_sha256=question_sha,
                    aliases_sha256=aliases_sha,
                    label_source="E1" if frozen_labels else "generate",
                    outcome=outcome,
                )
            )

    append_rows(
        controller,
        source_partition="T-controller",
        prompt_id="P0-neutral",
        frozen_labels=True,
    )
    append_rows(dev, source_partition="T-dev", prompt_id="P0-neutral", frozen_labels=False)
    append_rows(
        controller,
        source_partition="T-controller",
        prompt_id="P3-forced-answer",
        frozen_labels=False,
    )
    append_rows(dev, source_partition="T-dev", prompt_id="P3-forced-answer", frozen_labels=False)
    append_rows(
        simpleqa,
        source_partition="simpleqa-eval",
        prompt_id="P0-neutral",
        frozen_labels=True,
    )
    append_rows(aa, source_partition="aa-eval", prompt_id="P0-neutral", frozen_labels=True)
    _validate_e2_schedule(rows, protocol)
    return tuple(rows)


def _write_schedule(path: Path, rows: Sequence[E2ScheduleRow]) -> str:
    previous: str | None = None
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            body = {"previous_row_digest": previous, **row.to_dict()}
            row_digest = stable_hash(body)
            handle.write(json.dumps({**body, "row_digest": row_digest}, sort_keys=True) + "\n")
            previous = row_digest
        handle.flush()
        os.fsync(handle.fileno())
    assert previous is not None
    return previous


def write_e2_workspace(
    directory: str | Path,
    *,
    schedule: Sequence[E2ScheduleRow],
    protocol: E2CaptureProtocol,
    model: ModelSpec,
    hidden_width: int,
    input_fingerprints: Mapping[str, str],
) -> VerifiedE2Workspace:
    destination = validate_active_study_artifact_paths(
        {"E2 workspace": directory}
    )["E2 workspace"]
    if destination.exists():
        raise FrozenArtifactError(f"refusing to overwrite E2 workspace: {destination}")
    _validate_e2_schedule(schedule, protocol)
    if (
        model.runtime is not Runtime.MLX
        or model.num_layers <= max(E2_LAYERS)
        or type(hidden_width) is not int
        or hidden_width <= 0
    ):
        raise DataValidationError("E2 workspace requires the compatible MLX model geometry")
    if any(
        type(name) is not str or type(value) is not str
        for name, value in input_fingerprints.items()
    ):
        raise DataValidationError("E2 workspace input fingerprints have invalid types")
    inputs = dict(input_fingerprints)
    if not inputs or any(
        not name.strip() or not _SHA256.fullmatch(value) for name, value in inputs.items()
    ):
        raise DataValidationError("E2 workspace input fingerprints are invalid")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        schedule_head = _write_schedule(stage / "schedule.jsonl", schedule)
        plan_body = {
            "schema_version": 1,
            "phase": "E2",
            "runner": "native-mlx-streamed-float16-activations",
            "runner_source_sha256": sha256_file(Path(__file__)),
            "protocol": protocol.to_dict(),
            "scientific_eligible": protocol.scientific_eligible,
            "model": {
                "name": model.name,
                "repository": model.repository,
                "revision": model.revision,
                "runtime": model.runtime.value,
                "quantization": model.quantization,
                "num_layers": model.num_layers,
                "hidden_width": hidden_width,
            },
            "layers": list(E2_LAYERS),
            "sites": [site.value for site in E2_SITES],
            "schedule_rows": len(schedule),
            "new_generations": sum(row.label_source == "generate" for row in schedule),
            "schedule_sha256": sha256_file(stage / "schedule.jsonl"),
            "schedule_chain_head": schedule_head,
            "input_fingerprints": dict(sorted(inputs.items())),
        }
        plan_identity = stable_hash(plan_body)
        (stage / "plan.json").write_text(
            json.dumps({**plan_body, "plan_identity": plan_identity}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        activation_spec = ActivationStoreSpec(
            plan_identity=plan_identity,
            model_repository=model.repository,
            model_revision=model.revision,
            quantization=model.quantization,
            layers=E2_LAYERS,
            sites=E2_SITES,
            hidden_width=hidden_width,
            expected_rows=len(schedule),
        )
        create_activation_store(stage / "activations", activation_spec)
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return verify_e2_workspace(destination)


def verify_e2_workspace(directory: str | Path) -> VerifiedE2Workspace:
    source = Path(directory)
    if source.is_symlink() or not source.is_dir():
        raise FrozenArtifactError("E2 workspace must be a regular directory")
    if {path.name for path in source.iterdir()} != {"plan.json", "schedule.jsonl", "activations"}:
        raise FrozenArtifactError("E2 workspace inventory differs")
    if any((source / name).is_symlink() for name in ("plan.json", "schedule.jsonl")):
        raise FrozenArtifactError("E2 workspace files cannot be symlinks")
    try:
        plan = json.loads((source / "plan.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read E2 plan: {exc}") from exc
    expected_plan_keys = {
        "schema_version",
        "phase",
        "runner",
        "runner_source_sha256",
        "protocol",
        "scientific_eligible",
        "model",
        "layers",
        "sites",
        "schedule_rows",
        "new_generations",
        "schedule_sha256",
        "schedule_chain_head",
        "input_fingerprints",
        "plan_identity",
    }
    if type(plan) is not dict or set(plan) != expected_plan_keys:
        raise FrozenArtifactError("E2 plan must be a mapping")
    plan_identity = plan.pop("plan_identity", None)
    if plan_identity != stable_hash(plan) or plan.get("runner_source_sha256") != sha256_file(
        Path(__file__)
    ):
        raise FrozenArtifactError("E2 plan identity or runner source differs")
    try:
        protocol_value = plan["protocol"]
        if type(protocol_value) is not dict or set(protocol_value) != {
            "schema_version",
            "controller_rows",
            "controller_calibration_rows",
            "dev_rows",
            "simpleqa_rows",
            "aa_rows",
            "seed",
        }:
            raise TypeError("protocol is not a mapping")
        protocol = E2CaptureProtocol(**dict(protocol_value))
        model = plan["model"]
        if type(model) is not dict or set(model) != {
            "name",
            "repository",
            "revision",
            "runtime",
            "quantization",
            "num_layers",
            "hidden_width",
        }:
            raise TypeError("model is not a mapping")
        if (
            any(
                type(model[name]) is not str
                for name in ("name", "repository", "revision", "runtime", "quantization")
            )
            or type(model["num_layers"]) is not int
            or type(model["hidden_width"]) is not int
            or model["runtime"] != Runtime.MLX.value
            or model["num_layers"] <= max(E2_LAYERS)
        ):
            raise TypeError("model has invalid JSON types or MLX geometry")
        rows: list[E2ScheduleRow] = []
        previous: str | None = None
        with (source / "schedule.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                raw = json.loads(line)
                if type(raw) is not dict:
                    raise TypeError("schedule row is not a mapping")
                body = dict(raw)
                digest = body.pop("row_digest", None)
                if digest != stable_hash(body) or body.pop("previous_row_digest", None) != previous:
                    raise DataValidationError("E2 schedule chain differs")
                row = E2ScheduleRow.from_dict(body)
                if row.sequence != len(rows):
                    raise DataValidationError("E2 schedule sequence is not contiguous")
                rows.append(row)
                if type(digest) is not str or not _SHA256.fullmatch(digest):
                    raise DataValidationError("E2 schedule digest has an invalid type")
                previous = digest
        inputs = plan["input_fingerprints"]
        if type(inputs) is not dict or any(
            type(name) is not str or type(value) is not str for name, value in inputs.items()
        ):
            raise TypeError("input fingerprints are not a mapping")
        input_fingerprints = dict(inputs)
        if type(plan_identity) is not str:
            raise TypeError("plan identity has an invalid type")
        layers = plan["layers"]
        sites = plan["sites"]
        if (
            type(layers) is not list
            or any(type(value) is not int for value in layers)
            or type(sites) is not list
            or any(type(value) is not str for value in sites)
        ):
            raise TypeError("activation geometry has invalid JSON types")
        activation_spec = ActivationStoreSpec(
            plan_identity=plan_identity,
            model_repository=model["repository"],
            model_revision=model["revision"],
            quantization=model["quantization"],
            layers=tuple(layers),
            sites=tuple(ActivationSite(value) for value in sites),
            hidden_width=model["hidden_width"],
            expected_rows=len(rows),
        )
    except (KeyError, TypeError, ValueError, DataValidationError) as exc:
        raise FrozenArtifactError(f"invalid E2 workspace: {exc}") from exc
    if (
        type(plan.get("schema_version")) is not int
        or plan.get("schema_version") != 1
        or plan.get("phase") != "E2"
        or plan.get("runner") != "native-mlx-streamed-float16-activations"
        or type(plan.get("schedule_rows")) is not int
        or type(plan.get("new_generations")) is not int
        or plan.get("scientific_eligible") is not protocol.scientific_eligible
        or plan.get("layers") != list(E2_LAYERS)
        or plan.get("sites") != [site.value for site in E2_SITES]
        or plan.get("schedule_rows") != len(rows)
        or len(rows) != protocol.expected_capture_rows
        or plan.get("new_generations")
        != sum(row.label_source == "generate" for row in rows)
        or plan.get("schedule_sha256") != sha256_file(source / "schedule.jsonl")
        or plan.get("schedule_chain_head") != previous
        or not input_fingerprints
        or any(
            not name.strip() or not _SHA256.fullmatch(value)
            for name, value in input_fingerprints.items()
        )
    ):
        raise FrozenArtifactError("E2 workspace differs from its frozen plan")
    try:
        _validate_e2_schedule(rows, protocol)
    except DataValidationError as exc:
        raise FrozenArtifactError(f"invalid E2 schedule semantics: {exc}") from exc
    verify_activation_store(source / "activations", expected_spec=activation_spec)
    return VerifiedE2Workspace(
        directory=source,
        plan_identity=plan_identity,
        protocol=protocol,
        schedule=tuple(rows),
        input_fingerprints=MappingProxyType(input_fingerprints),
        activation_spec=activation_spec,
    )

"""Resumable native-MLX construction of the E3 M1-R and M1-P vector banks."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from mfh.contracts import ActivationSite, Outcome, PromptSpec, Question
from mfh.errors import ConfigurationError, DataValidationError, FrozenArtifactError
from mfh.evaluation.grading import deterministic_short_answer_grade
from mfh.experiments.e3_schedule import (
    E3ConstructionRow,
    E3Protocol,
    build_e3_construction_schedule,
)
from mfh.experiments.model_selection import (
    ACTIVE_MODEL_IDENTITIES,
    ACTIVE_MODEL_NAME,
    ACTIVE_RUNTIME_POLICY_RELATIVE,
    validate_active_study_artifact_paths,
)
from mfh.inference.mlx_preflight import validate_mlx_preflight_receipt
from mfh.inference.mlx_research import (
    MlxPromptFeatureCubeOutput,
    MlxTeacherForcedCubeOutput,
)
from mfh.inference.mlx_runtime import MlxGenerationOutput, MlxRenderedPrompt
from mfh.provenance import sha256_file, stable_hash

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PROMPTS = ("P0-neutral", "P2-calibrated-abstention")
_EXTRACTIONS = ("M1-R", "M1-P")
_OUTCOMES = (Outcome.CORRECT, Outcome.INCORRECT)
_WORK_INVENTORY = frozenset(
    {"plan.json", "generations.jsonl", "sessions.jsonl", "checkpoints"}
)
_UNIFIED_MEMORY_BYTES = 51_539_607_552
_CANONICAL_PROMPTS = MappingProxyType(
    {
        "P0-neutral": "You are a helpful assistant. Answer the user's factual question.",
        "P2-calibrated-abstention": (
            "Answer only when you are sufficiently confident that the answer is correct. "
            "Otherwise respond exactly with “I don't know.” Do not guess."
        ),
    }
)
_EXPECTED_BASE_RUNTIME = MappingProxyType(
    {
        "backend": "mlx",
        "mlx": "0.31.2",
        "mlx_lm": "0.31.3",
        "python": "3.11.14",
        "chip": "Apple M4 Max",
        "unified_memory_bytes": 51_539_607_552,
        "architecture": "arm64",
        "model_class": "mlx_lm.models.qwen3_5.Model",
        "tokenizer_class": "mlx_lm.tokenizer_utils.TokenizerWrapper",
        "num_layers": 64,
        "seed": 17,
    }
)
_EXPECTED_RESEARCH_PROVENANCE_FIELDS = {
    "schema_version",
    "model_repository",
    "model_revision",
    "quantization",
    "verified_snapshot_digest",
    "snapshot_manifest_sha256",
    "runtime_preflight_receipt_digest",
    "runtime_preflight_receipt_sha256",
    "runtime_preflight_receipt_relative",
    "model_snapshot_relative",
    "runtime_policy_digest",
    "runtime_policy_sha256",
    "preflight_intervention_digest",
    "research_toolchain_digest",
    "pyproject_sha256",
    "uv_lock_sha256",
    "tokenizer_sha256",
    "chat_template_sha256",
}
_EXPECTED_SPLIT_FINGERPRINTS = MappingProxyType(
    {
        "reviewed_split_manifest_digest": (
            "05e13f0193155551400fd636e8dd6d97e065dd80205133a9440ef13105bce148"
        ),
        "review_result_manifest_digest": (
            "6e03e98d9b09ee83fcfbbe5d1268ef42d2991467db043ecc345de84f64607f59"
        ),
        "t_steer_question_ids_sha256": (
            "2493ffbd5c73c9f1b42e419e9ebf50860d6e53b2344324eed731b793ae7ec2a7"
        ),
    }
)
_EXPECTED_T_STEER_QUESTIONS_DIGEST = (
    "0b0b6b57cced25d22e907f336365972905920580a3e7b1e7da093826f8f627c0"
)


class E3ConstructionRuntime(Protocol):
    def render_prompt(
        self,
        prompt: PromptSpec,
        question: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> MlxRenderedPrompt: ...

    def generate(
        self, rendered: MlxRenderedPrompt, *, max_new_tokens: int
    ) -> MlxGenerationOutput: ...

    def prompt_feature_cube(
        self,
        rendered: MlxRenderedPrompt,
        *,
        layers: Sequence[int],
        sites: Sequence[ActivationSite],
    ) -> MlxPromptFeatureCubeOutput: ...

    def teacher_forced_cube(
        self,
        rendered: MlxRenderedPrompt,
        response: str,
        *,
        layers: Sequence[int],
        sites: Sequence[ActivationSite],
    ) -> MlxTeacherForcedCubeOutput: ...

    def runtime_identity(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class E3GenerationRecord:
    sequence: int
    plan_identity: str
    schedule_row_sha256: str
    question_id: str
    prompt_id: str
    rendered_prompt_sha256: str
    prompt_token_ids_sha256: str
    outcome: Outcome
    evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence < 0:
            raise DataValidationError("E3 generation sequence is invalid")
        for name in (
            "plan_identity",
            "schedule_row_sha256",
            "rendered_prompt_sha256",
            "prompt_token_ids_sha256",
        ):
            if type(getattr(self, name)) is not str or not _SHA256.fullmatch(
                getattr(self, name)
            ):
                raise DataValidationError(f"E3 generation {name} is invalid")
        if (
            type(self.question_id) is not str
            or not self.question_id.strip()
            or self.prompt_id not in _PROMPTS
        ):
            raise DataValidationError("E3 generation row identity is invalid")
        evidence = _validated_generation_evidence(self.evidence)
        object.__setattr__(self, "outcome", Outcome(self.outcome))
        object.__setattr__(self, "evidence", evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "plan_identity": self.plan_identity,
            "schedule_row_sha256": self.schedule_row_sha256,
            "question_id": self.question_id,
            "prompt_id": self.prompt_id,
            "rendered_prompt_sha256": self.rendered_prompt_sha256,
            "prompt_token_ids_sha256": self.prompt_token_ids_sha256,
            "outcome": self.outcome.value,
            "evidence": dict(self.evidence),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> E3GenerationRecord:
        expected = {
            "sequence",
            "plan_identity",
            "schedule_row_sha256",
            "question_id",
            "prompt_id",
            "rendered_prompt_sha256",
            "prompt_token_ids_sha256",
            "outcome",
            "evidence",
        }
        if (
            set(value) != expected
            or type(value.get("sequence")) is not int
            or type(value.get("evidence")) is not dict
            or any(
                type(value.get(name)) is not str
                for name in expected - {"sequence", "evidence"}
            )
        ):
            raise DataValidationError("E3 generation JSON schema is invalid")
        return cls(
            sequence=value["sequence"],
            plan_identity=value["plan_identity"],
            schedule_row_sha256=value["schedule_row_sha256"],
            question_id=value["question_id"],
            prompt_id=value["prompt_id"],
            rendered_prompt_sha256=value["rendered_prompt_sha256"],
            prompt_token_ids_sha256=value["prompt_token_ids_sha256"],
            outcome=Outcome(value["outcome"]),
            evidence=value["evidence"],
        )


@dataclass(frozen=True, slots=True)
class VerifiedE3ConstructionSnapshot:
    directory: Path
    plan: Mapping[str, Any]
    schedule: tuple[E3ConstructionRow, ...]
    generations: tuple[E3GenerationRecord, ...]
    generation_chain_head: str
    scientific_eligible: bool


@dataclass(slots=True)
class _Accumulator:
    processed_rows: int
    counts: np.ndarray
    sums: np.ndarray
    rms_elements: np.ndarray
    rms_sum_squares: np.ndarray


def _validated_json_mapping(value: Mapping[str, Any], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise DataValidationError(f"{label} must be a non-empty mapping")
    normalized = dict(value)
    try:
        replayed = json.loads(
            json.dumps(normalized, sort_keys=True, allow_nan=False, separators=(",", ":"))
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"{label} is not exact JSON: {exc}") from exc
    if type(replayed) is not dict or replayed != normalized:
        raise DataValidationError(f"{label} is not stable JSON")
    return MappingProxyType(replayed)


def _deep_freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_deep_freeze_json(item) for item in value)
    return value


def _valid_research_provenance(
    provenance: object, toolchain: object
) -> bool:
    if (
        type(provenance) is not dict
        or set(provenance) != _EXPECTED_RESEARCH_PROVENANCE_FIELDS
        or type(toolchain) is not dict
        or set(toolchain) != {"xcodebuild", "metal_compiler"}
        or any(not isinstance(value, str) or not value for value in toolchain.values())
    ):
        return False
    root = Path(__file__).parents[3]
    receipt_relative = provenance.get("runtime_preflight_receipt_relative")
    snapshot_relative = provenance.get("model_snapshot_relative")
    if (
        not isinstance(receipt_relative, str)
        or not isinstance(snapshot_relative, str)
        or Path(receipt_relative).is_absolute()
        or Path(snapshot_relative).is_absolute()
        or Path(receipt_relative).as_posix() != receipt_relative
        or Path(snapshot_relative).as_posix() != snapshot_relative
    ):
        return False
    receipt_path = (root / receipt_relative).resolve()
    snapshot_path = (root / snapshot_relative).resolve()
    try:
        receipt_path.relative_to(root)
        snapshot_path.relative_to(root)
        model_config = root / "configs/models/qwen3.6-27b-mlx-4bit.yaml"
        snapshot_manifest = root / "configs/models/qwen3.6-27b-mlx-4bit.snapshot.json"
        runtime_policy = root / ACTIVE_RUNTIME_POLICY_RELATIVE
        receipt = validate_mlx_preflight_receipt(
            receipt_path,
            project_root=root,
            model_config=model_config,
            snapshot_directory=snapshot_path,
            snapshot_manifest=snapshot_manifest,
            runtime_policy=runtime_policy,
        )
        receipt_model = receipt["model"]
        receipt_software = receipt["software"]
        snapshot_identity = receipt_model["snapshot_identity"]
        if (
            not isinstance(receipt_model, Mapping)
            or not isinstance(receipt_software, Mapping)
            or not isinstance(snapshot_identity, Mapping)
            or receipt_software.get("toolchain") != toolchain
        ):
            return False
        identity = ACTIVE_MODEL_IDENTITIES[ACTIVE_MODEL_NAME]
        expected = {
            "schema_version": 3,
            "model_repository": identity["repository"],
            "model_revision": identity["revision"],
            "quantization": identity["quantization"],
            "verified_snapshot_digest": snapshot_identity["snapshot_digest"],
            "snapshot_manifest_sha256": sha256_file(snapshot_manifest),
            "runtime_preflight_receipt_digest": receipt["receipt_digest"],
            "runtime_preflight_receipt_sha256": sha256_file(receipt_path),
            "runtime_preflight_receipt_relative": receipt_relative,
            "model_snapshot_relative": snapshot_relative,
            "runtime_policy_digest": receipt["policy_digest"],
            "runtime_policy_sha256": sha256_file(runtime_policy),
            "preflight_intervention_digest": stable_hash(receipt["intervention"]),
            "research_toolchain_digest": stable_hash(toolchain),
            "pyproject_sha256": sha256_file(root / "pyproject.toml"),
            "uv_lock_sha256": sha256_file(root / "uv.lock"),
            "tokenizer_sha256": sha256_file(snapshot_path / "tokenizer.json"),
            "chat_template_sha256": sha256_file(snapshot_path / "chat_template.jinja"),
        }
    except (
        ConfigurationError,
        DataValidationError,
        FrozenArtifactError,
        KeyError,
        OSError,
        ValueError,
    ):
        return False
    return provenance == expected


def _scientific_eligible(
    *,
    protocol: E3Protocol,
    prompts: Mapping[str, PromptSpec],
    hidden_width: int,
    runtime_identity: Mapping[str, Any],
    input_fingerprints: Mapping[str, str],
    questions_digest: str,
) -> bool:
    provenance = runtime_identity.get("research_provenance")
    toolchain = runtime_identity.get("research_toolchain")
    canonical_prompts = all(
        prompts[name].text == text
        and prompts[name].permits_abstention
        and prompts[name].deployment_eligible
        for name, text in _CANONICAL_PROMPTS.items()
    )
    dynamic_host_valid = (
        isinstance(runtime_identity.get("machine_model"), str)
        and bool(runtime_identity["machine_model"])
        and type(runtime_identity.get("physical_cpu_cores")) is int
        and runtime_identity["physical_cpu_cores"] > 0
        and isinstance(runtime_identity.get("os"), str)
        and bool(runtime_identity["os"])
        and isinstance(runtime_identity.get("os_build"), str)
        and bool(runtime_identity["os_build"])
    )
    return bool(
        protocol.scientific_eligible
        and hidden_width == 5_120
        and canonical_prompts
        and all(
            runtime_identity.get(name) == value
            for name, value in _EXPECTED_BASE_RUNTIME.items()
        )
        and dynamic_host_valid
        and _valid_research_provenance(provenance, toolchain)
        and input_fingerprints == dict(_EXPECTED_SPLIT_FINGERPRINTS)
        and questions_digest == _EXPECTED_T_STEER_QUESTIONS_DIGEST
    )


def _validated_generation_evidence(value: Mapping[str, Any]) -> Mapping[str, Any]:
    expected = {
        "raw_output",
        "raw_output_sha256",
        "token_ids",
        "token_ids_sha256",
        "input_tokens",
        "output_tokens",
        "latency_seconds",
        "stop_type",
        "stopping_token_id",
        "prompt_tokens_per_second",
        "generation_tokens_per_second",
        "peak_memory_bytes",
        "active_memory_bytes",
        "cache_memory_bytes",
    }
    normalized = dict(value)
    token_ids = normalized.get("token_ids")
    integer_fields = (
        "input_tokens",
        "output_tokens",
        "peak_memory_bytes",
        "active_memory_bytes",
        "cache_memory_bytes",
    )
    number_fields = (
        "latency_seconds",
        "prompt_tokens_per_second",
        "generation_tokens_per_second",
    )
    if (
        set(normalized) != expected
        or type(normalized.get("raw_output")) is not str
        or type(normalized.get("raw_output_sha256")) is not str
        or normalized.get("raw_output_sha256")
        != hashlib.sha256(normalized["raw_output"].encode()).hexdigest()
        or type(token_ids) is not list
        or not token_ids
        or any(type(token) is not int or token < 0 for token in token_ids)
        or type(normalized.get("token_ids_sha256")) is not str
        or normalized.get("token_ids_sha256") != stable_hash(token_ids)
        or any(
            type(normalized.get(name)) is not int or normalized[name] < 0
            for name in integer_fields
        )
        or any(
            isinstance(normalized.get(name), bool)
            or not isinstance(normalized.get(name), int | float)
            or not math.isfinite(float(normalized[name]))
            or float(normalized[name]) < 0
            for name in number_fields
        )
        or normalized.get("output_tokens") != len(token_ids)
        or normalized.get("stopping_token_id") != token_ids[-1]
        or normalized.get("stop_type") not in {"stop", "length", "short_answer"}
    ):
        raise DataValidationError("E3 generation evidence is invalid")
    return _validated_json_mapping(normalized, label="E3 generation evidence")


def _generation_evidence(value: MlxGenerationOutput) -> Mapping[str, Any]:
    return _validated_generation_evidence(
        {
            "raw_output": value.text,
            "raw_output_sha256": hashlib.sha256(value.text.encode()).hexdigest(),
            "token_ids": list(value.token_ids),
            "token_ids_sha256": stable_hash(list(value.token_ids)),
            "input_tokens": value.input_tokens,
            "output_tokens": value.output_tokens,
            "latency_seconds": value.latency_seconds,
            "stop_type": value.stop_type,
            "stopping_token_id": value.stopping_token_id,
            "prompt_tokens_per_second": value.prompt_tokens_per_second,
            "generation_tokens_per_second": value.generation_tokens_per_second,
            "peak_memory_bytes": value.peak_memory_bytes,
            "active_memory_bytes": value.active_memory_bytes,
            "cache_memory_bytes": value.cache_memory_bytes,
        }
    )


def _questions_digest(questions: Sequence[Question]) -> str:
    return stable_hash(
        [
            {
                "question_id": value.question_id,
                "benchmark": value.benchmark,
                "text": value.text,
                "aliases": list(value.aliases),
                "metadata": dict(value.metadata),
            }
            for value in questions
        ]
    )


def _prompts_digest(prompts: Mapping[str, PromptSpec]) -> str:
    return stable_hash(
        [
            {
                "mapping_key": name,
                "prompt_id": prompt.prompt_id,
                "text": prompt.text,
                "permits_abstention": prompt.permits_abstention,
                "deployment_eligible": prompt.deployment_eligible,
            }
            for name, prompt in sorted(prompts.items())
        ]
    )


def _validate_inputs(
    questions: Sequence[Question],
    prompts: Mapping[str, PromptSpec],
    protocol: E3Protocol,
) -> tuple[E3ConstructionRow, ...]:
    if set(prompts) != set(_PROMPTS) or any(
        key != prompts[key].prompt_id for key in prompts
    ):
        raise DataValidationError("E3 construction requires exact P0 and P2 prompt mappings")
    schedule = build_e3_construction_schedule(questions, protocol=protocol)
    if len({row.question_id for row in schedule[: protocol.steer_rows]}) != len(
        questions
    ):
        raise DataValidationError("E3 construction questions are not uniquely bound")
    return schedule


def _plan_body(
    questions: Sequence[Question],
    prompts: Mapping[str, PromptSpec],
    *,
    protocol: E3Protocol,
    hidden_width: int,
    checkpoint_rows: int,
    max_new_tokens: int,
    runtime_identity: Mapping[str, Any],
    input_fingerprints: Mapping[str, str],
) -> dict[str, Any]:
    schedule = _validate_inputs(questions, prompts, protocol)
    if (
        type(hidden_width) is not int
        or hidden_width <= 0
        or type(checkpoint_rows) is not int
        or checkpoint_rows <= 0
        or type(max_new_tokens) is not int
        or not 1 <= max_new_tokens <= 48
    ):
        raise DataValidationError("E3 construction geometry or checkpoint policy is invalid")
    identity = _validated_json_mapping(runtime_identity, label="E3 runtime identity")
    questions_digest = _questions_digest(questions)
    if any(
        type(name) is not str
        or not name.strip()
        or type(value) is not str
        or not _SHA256.fullmatch(value)
        for name, value in input_fingerprints.items()
    ):
        raise DataValidationError("E3 construction input fingerprints are invalid")
    return {
        "schema_version": 1,
        "phase": "E3-construction",
        "runner": "resumable-native-mlx-online-centroids",
        "runner_source_sha256": sha256_file(Path(__file__)),
        "protocol": protocol.to_dict(),
        "schedule_digest": stable_hash([row.to_dict() for row in schedule]),
        "questions_digest": questions_digest,
        "prompts_digest": _prompts_digest(prompts),
        "expected_rows": len(schedule),
        "hidden_width": hidden_width,
        "checkpoint_rows": checkpoint_rows,
        "max_new_tokens": max_new_tokens,
        "runtime_identity": dict(identity),
        "input_fingerprints": dict(sorted(input_fingerprints.items())),
        "scientific_eligible": _scientific_eligible(
            protocol=protocol,
            prompts=prompts,
            hidden_width=hidden_width,
            runtime_identity=identity,
            input_fingerprints=input_fingerprints,
            questions_digest=questions_digest,
        ),
    }


def _complete_plan(body: Mapping[str, Any]) -> dict[str, Any]:
    return {**body, "plan_identity": stable_hash(dict(body))}


def prepare_e3_construction_work(
    directory: str | Path,
    *,
    questions: Sequence[Question],
    prompts: Mapping[str, PromptSpec],
    runtime_identity: Mapping[str, Any],
    hidden_width: int,
    protocol: E3Protocol | None = None,
    checkpoint_rows: int = 64,
    max_new_tokens: int = 48,
    input_fingerprints: Mapping[str, str] | None = None,
) -> Mapping[str, Any]:
    """Freeze an empty construction workspace without loading the model."""

    directory = validate_active_study_artifact_paths(
        {"E3 construction work": directory}
    )["E3 construction work"]
    frozen_protocol = protocol or E3Protocol()
    plan = _complete_plan(
        _plan_body(
            questions,
            prompts,
            protocol=frozen_protocol,
            hidden_width=hidden_width,
            checkpoint_rows=checkpoint_rows,
            max_new_tokens=max_new_tokens,
            runtime_identity=runtime_identity,
            input_fingerprints=dict(input_fingerprints or {}),
        )
    )
    destination = Path(directory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite E3 construction work: {destination}")
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        (stage / "checkpoints").mkdir()
        (stage / "plan.json").write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (stage / "generations.jsonl").touch()
        (stage / "sessions.jsonl").touch()
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return MappingProxyType(plan)


def _load_plan(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read E3 construction plan: {exc}") from exc
    if type(value) is not dict or type(value.get("plan_identity")) is not str:
        raise FrozenArtifactError("E3 construction plan schema is invalid")
    body = dict(value)
    identity = body.pop("plan_identity")
    expected = {
        "schema_version",
        "phase",
        "runner",
        "runner_source_sha256",
        "protocol",
        "schedule_digest",
        "questions_digest",
        "prompts_digest",
        "expected_rows",
        "hidden_width",
        "checkpoint_rows",
        "max_new_tokens",
        "runtime_identity",
        "input_fingerprints",
        "scientific_eligible",
    }
    if (
        set(body) != expected
        or identity != stable_hash(body)
        or body.get("runner_source_sha256") != sha256_file(Path(__file__))
        or body.get("schema_version") != 1
        or body.get("phase") != "E3-construction"
        or body.get("runner") != "resumable-native-mlx-online-centroids"
        or type(body.get("protocol")) is not dict
        or type(body.get("runtime_identity")) is not dict
        or type(body.get("input_fingerprints")) is not dict
        or type(body.get("scientific_eligible")) is not bool
    ):
        raise FrozenArtifactError("E3 construction plan identity or source differs")
    return value


def _append_chained_json(
    path: Path,
    body: Mapping[str, Any],
    *,
    previous: str | None,
    prefix: str,
) -> str:
    chained = {**body, f"previous_{prefix}_digest": previous}
    digest = stable_hash(chained)
    envelope = {**chained, f"{prefix}_digest": digest}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(envelope, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return digest


def _repair_torn_jsonl_tail(path: Path) -> None:
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    boundary = data.rfind(b"\n")
    tail = data[boundary + 1 :]
    try:
        parsed = json.loads(tail.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        repaired = data[: boundary + 1] if boundary >= 0 else b""
    else:
        repaired = data + b"\n" if type(parsed) is dict else data[: boundary + 1]
    descriptor = os.open(path, os.O_WRONLY | os.O_TRUNC)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(repaired)
        handle.flush()
        os.fsync(handle.fileno())


def _repair_checkpoint_temporaries(directory: Path) -> None:
    for path in directory.iterdir():
        if not path.name.startswith(".checkpoint-"):
            continue
        if path.is_symlink() or not path.is_file():
            raise FrozenArtifactError("E3 checkpoint temporary is not a regular file")
        path.unlink()


def _load_generations(
    path: Path,
    *,
    plan_identity: str,
    schedule: Sequence[E3ConstructionRow],
) -> tuple[list[E3GenerationRecord], list[str]]:
    records: list[E3GenerationRecord] = []
    digests: list[str] = []
    previous: str | None = None
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                raw = json.loads(line)
                if type(raw) is not dict:
                    raise DataValidationError("E3 generation envelope is invalid")
                body = dict(raw)
                digest = body.pop("generation_digest", None)
                if (
                    type(digest) is not str
                    or not _SHA256.fullmatch(digest)
                    or digest != stable_hash(body)
                    or body.pop("previous_generation_digest", None) != previous
                ):
                    raise DataValidationError("E3 generation chain differs")
                record = E3GenerationRecord.from_dict(body)
                if record.sequence != len(records) or record.sequence >= len(schedule):
                    raise DataValidationError("E3 generation sequence differs")
                row = schedule[record.sequence]
                if (
                    record.plan_identity != plan_identity
                    or record.schedule_row_sha256 != stable_hash(row.to_dict())
                    or record.question_id != row.question_id
                    or record.prompt_id != row.prompt_id
                ):
                    raise DataValidationError("E3 generation differs from schedule")
                records.append(record)
                digests.append(digest)
                previous = digest
    except (OSError, json.JSONDecodeError, TypeError, ValueError, DataValidationError) as exc:
        raise FrozenArtifactError(f"cannot read E3 generation chain: {exc}") from exc
    return records, digests


def _empty_accumulator(protocol: E3Protocol, hidden_width: int) -> _Accumulator:
    core = (
        len(_PROMPTS),
        len(_EXTRACTIONS),
        len(_OUTCOMES),
        len(protocol.candidate_sites),
        len(protocol.candidate_layers),
    )
    reference = (
        len(_PROMPTS),
        len(_EXTRACTIONS),
        len(protocol.candidate_sites),
        len(protocol.candidate_layers),
    )
    return _Accumulator(
        processed_rows=0,
        counts=np.zeros(core, dtype=np.int64),
        sums=np.zeros((*core, hidden_width), dtype=np.float64),
        rms_elements=np.zeros(reference, dtype=np.int64),
        rms_sum_squares=np.zeros(reference, dtype=np.float64),
    )


def _checkpoint_files(directory: Path) -> tuple[Path, ...]:
    values = tuple(sorted(directory.glob("checkpoint-*.npz")))
    if {path.name for path in directory.iterdir()} != {path.name for path in values}:
        raise FrozenArtifactError("E3 checkpoint inventory differs")
    return values


def _checkpoint_metadata(
    *,
    index: int,
    accumulator: _Accumulator,
    plan_identity: str,
    generation_chain_head: str | None,
    previous_checkpoint_sha256: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "checkpoint_index": index,
        "processed_rows": accumulator.processed_rows,
        "plan_identity": plan_identity,
        "generation_chain_head": generation_chain_head,
        "previous_checkpoint_sha256": previous_checkpoint_sha256,
    }


def _write_checkpoint(
    directory: Path,
    *,
    accumulator: _Accumulator,
    plan_identity: str,
    generation_chain_head: str | None,
    previous_checkpoint_sha256: str | None,
    index: int,
) -> tuple[Path, str]:
    metadata = _checkpoint_metadata(
        index=index,
        accumulator=accumulator,
        plan_identity=plan_identity,
        generation_chain_head=generation_chain_head,
        previous_checkpoint_sha256=previous_checkpoint_sha256,
    )
    destination = directory / f"checkpoint-{index:08d}.npz"
    if destination.exists():
        raise FrozenArtifactError("refusing to overwrite an E3 checkpoint")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".checkpoint-", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                counts=accumulator.counts,
                sums=accumulator.sums,
                rms_elements=accumulator.rms_elements,
                rms_sum_squares=accumulator.rms_sum_squares,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination, sha256_file(destination)


def _read_checkpoint(
    path: Path,
    *,
    expected_index: int,
    previous_checkpoint_sha256: str | None,
    plan_identity: str,
    protocol: E3Protocol,
    hidden_width: int,
    generation_digests: Sequence[str],
) -> _Accumulator:
    try:
        with np.load(path, allow_pickle=False) as values:
            if set(values.files) != {
                "metadata",
                "counts",
                "sums",
                "rms_elements",
                "rms_sum_squares",
            }:
                raise DataValidationError("E3 checkpoint array inventory differs")
            metadata_array = values["metadata"]
            if metadata_array.shape != () or metadata_array.dtype.kind not in {"U", "S"}:
                raise DataValidationError("E3 checkpoint metadata array is invalid")
            metadata = json.loads(str(metadata_array.item()))
            accumulator = _Accumulator(
                processed_rows=int(metadata.get("processed_rows", -1)),
                counts=values["counts"].copy(),
                sums=values["sums"].copy(),
                rms_elements=values["rms_elements"].copy(),
                rms_sum_squares=values["rms_sum_squares"].copy(),
            )
    except (OSError, ValueError, json.JSONDecodeError, DataValidationError) as exc:
        raise FrozenArtifactError(f"cannot read E3 checkpoint: {exc}") from exc
    empty = _empty_accumulator(protocol, hidden_width)
    expected_metadata = _checkpoint_metadata(
        index=expected_index,
        accumulator=accumulator,
        plan_identity=plan_identity,
        generation_chain_head=(
            generation_digests[accumulator.processed_rows - 1]
            if accumulator.processed_rows
            else None
        ),
        previous_checkpoint_sha256=previous_checkpoint_sha256,
    )
    if (
        type(metadata) is not dict
        or metadata != expected_metadata
        or accumulator.processed_rows <= 0
        or accumulator.processed_rows > len(generation_digests)
        or accumulator.counts.dtype != np.int64
        or accumulator.sums.dtype != np.float64
        or accumulator.rms_elements.dtype != np.int64
        or accumulator.rms_sum_squares.dtype != np.float64
        or accumulator.counts.shape != empty.counts.shape
        or accumulator.sums.shape != empty.sums.shape
        or accumulator.rms_elements.shape != empty.rms_elements.shape
        or accumulator.rms_sum_squares.shape != empty.rms_sum_squares.shape
        or np.any(accumulator.counts < 0)
        or np.any(accumulator.rms_elements < 0)
        or not np.isfinite(accumulator.sums).all()
        or not np.isfinite(accumulator.rms_sum_squares).all()
        or np.any(accumulator.rms_sum_squares < 0)
    ):
        raise FrozenArtifactError("E3 checkpoint geometry or chain differs")
    return accumulator


def _load_latest_checkpoint(
    directory: Path,
    *,
    plan_identity: str,
    protocol: E3Protocol,
    hidden_width: int,
    generations: Sequence[E3GenerationRecord],
    generation_digests: Sequence[str],
) -> tuple[_Accumulator, str | None, int]:
    previous_sha: str | None = None
    latest = _empty_accumulator(protocol, hidden_width)
    files = _checkpoint_files(directory)
    for index, path in enumerate(files):
        if path.name != f"checkpoint-{index:08d}.npz":
            raise FrozenArtifactError("E3 checkpoint numbering differs")
        latest = _read_checkpoint(
            path,
            expected_index=index,
            previous_checkpoint_sha256=previous_sha,
            plan_identity=plan_identity,
            protocol=protocol,
            hidden_width=hidden_width,
            generation_digests=generation_digests,
        )
        _verify_accumulator_counts(latest, generations[: latest.processed_rows], protocol)
        previous_sha = sha256_file(path)
    return latest, previous_sha, len(files)


def _verify_accumulator_counts(
    accumulator: _Accumulator,
    generations: Sequence[E3GenerationRecord],
    protocol: E3Protocol,
) -> None:
    expected = np.zeros((len(_PROMPTS), len(_OUTCOMES)), dtype=np.int64)
    for record in generations:
        if record.outcome in _OUTCOMES:
            expected[_PROMPTS.index(record.prompt_id), _OUTCOMES.index(record.outcome)] += 1
    for prompt_index in range(len(_PROMPTS)):
        for outcome_index in range(len(_OUTCOMES)):
            if not np.all(
                accumulator.counts[prompt_index, :, outcome_index, :, :]
                == expected[prompt_index, outcome_index]
            ):
                raise FrozenArtifactError("E3 checkpoint class counts differ from generations")
    observed = accumulator.counts.sum(axis=2) > 0
    if np.any((accumulator.rms_elements == 0) & observed):
        raise FrozenArtifactError("E3 checkpoint has class observations without RMS evidence")


def _render_for_record(
    runtime: E3ConstructionRuntime,
    question: Question,
    prompt: PromptSpec,
    record: E3GenerationRecord | None,
) -> MlxRenderedPrompt:
    rendered = runtime.render_prompt(prompt, question.text, metadata=question.metadata)
    if record is not None and (
        rendered.sha256 != record.rendered_prompt_sha256
        or rendered.token_ids_sha256 != record.prompt_token_ids_sha256
    ):
        raise FrozenArtifactError("E3 rerendered prompt differs from generation journal")
    return rendered


def _new_generation_record(
    *,
    row: E3ConstructionRow,
    plan_identity: str,
    question: Question,
    rendered: MlxRenderedPrompt,
    generation: MlxGenerationOutput,
) -> E3GenerationRecord:
    if generation.rendered_prompt != rendered:
        raise DataValidationError("E3 generation returned a different rendered prompt")
    evidence = _generation_evidence(generation)
    outcome = deterministic_short_answer_grade(generation.text, question.aliases)
    return E3GenerationRecord(
        sequence=row.sequence,
        plan_identity=plan_identity,
        schedule_row_sha256=stable_hash(row.to_dict()),
        question_id=row.question_id,
        prompt_id=row.prompt_id,
        rendered_prompt_sha256=rendered.sha256,
        prompt_token_ids_sha256=rendered.token_ids_sha256,
        outcome=outcome,
        evidence=evidence,
    )


def _validated_activation(
    value: np.ndarray, *, rows: int | None, hidden_width: int
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if (
        array.ndim != 2
        or (rows is not None and array.shape[0] != rows)
        or array.shape[0] <= 0
        or array.shape[1] != hidden_width
        or not np.isfinite(array).all()
    ):
        raise DataValidationError("E3 construction activation geometry is invalid")
    return array


def _collect_observation(
    *,
    runtime: E3ConstructionRuntime,
    rendered: MlxRenderedPrompt,
    record: E3GenerationRecord,
    protocol: E3Protocol,
    hidden_width: int,
) -> tuple[dict[tuple[str, ActivationSite, int], tuple[np.ndarray, np.ndarray]], int]:
    if record.outcome not in _OUTCOMES:
        return {}, 0
    prompt = runtime.prompt_feature_cube(
        rendered, layers=protocol.candidate_layers, sites=protocol.candidate_sites
    )
    response = str(record.evidence["raw_output"])
    response_cube = runtime.teacher_forced_cube(
        rendered,
        response,
        layers=protocol.candidate_layers,
        sites=protocol.candidate_sites,
    )
    if response_cube.response_text_sha256 != hashlib.sha256(response.encode()).hexdigest():
        raise DataValidationError("E3 response cube differs from journaled output")
    values: dict[tuple[str, ActivationSite, int], tuple[np.ndarray, np.ndarray]] = {}
    for site in protocol.candidate_sites:
        for layer in protocol.candidate_layers:
            prompt_array = _validated_activation(
                prompt.activations[site][layer], rows=1, hidden_width=hidden_width
            )
            response_array = _validated_activation(
                response_cube.activations[site][layer],
                rows=len(response_cube.response_token_ids),
                hidden_width=hidden_width,
            )
            values[("M1-P", site, layer)] = (prompt_array[0], prompt_array)
            values[("M1-R", site, layer)] = (
                response_array.mean(axis=0, dtype=np.float64),
                response_array,
            )
    return values, max(prompt.peak_memory_bytes, response_cube.peak_memory_bytes)


def _apply_observation(
    accumulator: _Accumulator,
    *,
    record: E3GenerationRecord,
    values: Mapping[tuple[str, ActivationSite, int], tuple[np.ndarray, np.ndarray]],
    protocol: E3Protocol,
) -> None:
    if record.outcome not in _OUTCOMES:
        return
    prompt_index = _PROMPTS.index(record.prompt_id)
    outcome_index = _OUTCOMES.index(record.outcome)
    expected_keys = {
        (extraction, site, layer)
        for extraction in _EXTRACTIONS
        for site in protocol.candidate_sites
        for layer in protocol.candidate_layers
    }
    if set(values) != expected_keys:
        raise DataValidationError("E3 observation hook inventory differs")
    for (extraction, site, layer), (pooled, raw) in values.items():
        extraction_index = _EXTRACTIONS.index(extraction)
        site_index = protocol.candidate_sites.index(site)
        layer_index = protocol.candidate_layers.index(layer)
        core = (
            prompt_index,
            extraction_index,
            outcome_index,
            site_index,
            layer_index,
        )
        reference = (prompt_index, extraction_index, site_index, layer_index)
        accumulator.counts[core] += 1
        accumulator.sums[core] += np.asarray(pooled, dtype=np.float64)
        accumulator.rms_elements[reference] += raw.size
        accumulator.rms_sum_squares[reference] += float(
            np.square(raw.astype(np.float64)).sum()
        )


def _load_sessions(path: Path, *, plan_identity: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous: str | None = None
    open_index: int | None = None
    last_processed = 0
    start_keys = {
        "schema_version",
        "event",
        "session_index",
        "plan_identity",
        "processed_rows",
        "runtime_identity",
        "created_unix_ns",
    }
    end_keys = {
        "schema_version",
        "event",
        "session_index",
        "plan_identity",
        "status",
        "processed_rows",
        "generation_chain_head",
        "checkpoint_chain_head",
        "wall_time_seconds",
        "peak_memory_bytes",
        "created_unix_ns",
    }
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if type(row) is not dict:
                    raise DataValidationError("E3 session envelope is invalid")
                body = dict(row)
                digest = body.pop("session_digest", None)
                prior = body.pop("previous_session_digest", None)
                if (
                    type(digest) is not str
                    or digest != stable_hash({**body, "previous_session_digest": prior})
                    or prior != previous
                    or body.get("plan_identity") != plan_identity
                    or body.get("schema_version") != 1
                ):
                    raise DataValidationError("E3 session chain differs")
                event = body.get("event")
                index = body.get("session_index")
                if type(index) is not int or index < 0:
                    raise DataValidationError("E3 session index is invalid")
                if event == "start":
                    if (
                        set(body) != start_keys
                        or open_index is not None
                        or index != sum(r["event"] == "start" for r in rows)
                        or type(body.get("processed_rows")) is not int
                        or body["processed_rows"] != last_processed
                        or type(body.get("runtime_identity")) is not dict
                        or not body["runtime_identity"]
                        or type(body.get("created_unix_ns")) is not int
                        or body["created_unix_ns"] <= 0
                    ):
                        raise DataValidationError("E3 session start ordering differs")
                    open_index = index
                elif event == "end":
                    if (
                        set(body) != end_keys
                        or open_index != index
                        or body.get("status")
                        not in {"complete", "partial", "error", "interrupted-recovered"}
                        or type(body.get("processed_rows")) is not int
                        or body["processed_rows"] < last_processed
                        or any(
                            value is not None
                            and (type(value) is not str or not _SHA256.fullmatch(value))
                            for value in (
                                body.get("generation_chain_head"),
                                body.get("checkpoint_chain_head"),
                            )
                        )
                        or type(body.get("peak_memory_bytes")) is not int
                        or body["peak_memory_bytes"] < 0
                        or isinstance(body.get("wall_time_seconds"), bool)
                        or not isinstance(body.get("wall_time_seconds"), int | float)
                        or not math.isfinite(float(body["wall_time_seconds"]))
                        or float(body["wall_time_seconds"]) < 0
                        or type(body.get("created_unix_ns")) is not int
                        or body["created_unix_ns"] <= 0
                    ):
                        raise DataValidationError("E3 session end differs")
                    last_processed = body["processed_rows"]
                    open_index = None
                else:
                    raise DataValidationError("E3 session event is invalid")
                rows.append({**body, "session_digest": digest})
                previous = digest
    except (OSError, json.JSONDecodeError, TypeError, ValueError, DataValidationError) as exc:
        raise FrozenArtifactError(f"cannot read E3 sessions: {exc}") from exc
    return rows


def _session_head(rows: Sequence[Mapping[str, Any]]) -> str | None:
    return str(rows[-1]["session_digest"]) if rows else None


def _append_session(path: Path, body: Mapping[str, Any], previous: str | None) -> str:
    return _append_chained_json(
        path, body, previous=previous, prefix="session"
    )


@contextmanager
def _exclusive_plan_lock(path: Path) -> Iterator[None]:
    with path.open("rb") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ConfigurationError("E3 construction work is already running") from exc
        yield


def _work_context(
    directory: str | Path,
    *,
    questions: Sequence[Question],
    prompts: Mapping[str, PromptSpec],
    protocol: E3Protocol,
) -> tuple[Path, dict[str, Any], tuple[E3ConstructionRow, ...]]:
    work = Path(directory)
    if (
        work.is_symlink()
        or not work.is_dir()
        or {path.name for path in work.iterdir()} != _WORK_INVENTORY
        or any(path.is_symlink() for path in work.iterdir())
        or not (work / "checkpoints").is_dir()
        or any(
            not (work / name).is_file()
            for name in ("plan.json", "generations.jsonl", "sessions.jsonl")
        )
    ):
        raise FrozenArtifactError("E3 construction work inventory differs")
    plan = _load_plan(work / "plan.json")
    schedule = _validate_inputs(questions, prompts, protocol)
    expected = _complete_plan(
        _plan_body(
            questions,
            prompts,
            protocol=protocol,
            hidden_width=plan["hidden_width"],
            checkpoint_rows=plan["checkpoint_rows"],
            max_new_tokens=plan["max_new_tokens"],
            runtime_identity=plan["runtime_identity"],
            input_fingerprints=plan["input_fingerprints"],
        )
    )
    if plan != expected:
        raise FrozenArtifactError("E3 construction plan differs from live inputs")
    return work, plan, schedule


def verify_e3_construction_work(
    directory: str | Path,
    *,
    questions: Sequence[Question],
    prompts: Mapping[str, PromptSpec],
    protocol: E3Protocol | None = None,
    require_complete: bool = False,
) -> Mapping[str, Any]:
    """Replay every model-free construction invariant and immutable chain."""

    frozen_protocol = protocol or E3Protocol()
    work, plan, schedule = _work_context(
        directory, questions=questions, prompts=prompts, protocol=frozen_protocol
    )
    records, generation_digests = _load_generations(
        work / "generations.jsonl",
        plan_identity=plan["plan_identity"],
        schedule=schedule,
    )
    accumulator, checkpoint_head, checkpoint_count = _load_latest_checkpoint(
        work / "checkpoints",
        plan_identity=plan["plan_identity"],
        protocol=frozen_protocol,
        hidden_width=plan["hidden_width"],
        generations=records,
        generation_digests=generation_digests,
    )
    sessions = _load_sessions(work / "sessions.jsonl", plan_identity=plan["plan_identity"])
    if sessions and sessions[-1]["event"] == "start":
        raise FrozenArtifactError("E3 construction contains an unclosed session")
    if not sessions and (records or accumulator.processed_rows):
        raise FrozenArtifactError("E3 construction state lacks a session journal")
    generation_head = generation_digests[-1] if generation_digests else None
    if sessions and (
        sessions[-1]["processed_rows"] != accumulator.processed_rows
        or sessions[-1].get("checkpoint_chain_head") != checkpoint_head
        or sessions[-1].get("generation_chain_head") != generation_head
    ):
        raise FrozenArtifactError("E3 session head differs from checkpoint state")
    complete = accumulator.processed_rows == len(schedule) == len(records)
    maximum_peak_memory = max(
        (
            int(row["peak_memory_bytes"])
            for row in sessions
            if row["event"] == "end"
        ),
        default=0,
    )
    memory_within_envelope = maximum_peak_memory <= _UNIFIED_MEMORY_BYTES
    if len(records) < accumulator.processed_rows:
        raise FrozenArtifactError("E3 checkpoint exceeds generation journal")
    if sessions and (sessions[-1]["status"] == "complete") != complete:
        raise FrozenArtifactError("E3 session completion status differs from stores")
    if require_complete and not complete:
        raise FrozenArtifactError("E3 construction work is incomplete")
    return MappingProxyType(
        {
            "valid": True,
            "complete": complete,
            "rows_processed": accumulator.processed_rows,
            "rows_generated": len(records),
            "rows_expected": len(schedule),
            "plan_identity": plan["plan_identity"],
            "generation_chain_head": generation_head,
            "checkpoint_chain_head": checkpoint_head,
            "checkpoint_count": checkpoint_count,
            "session_chain_head": _session_head(sessions),
            "maximum_peak_memory_bytes": maximum_peak_memory,
            "memory_within_envelope": memory_within_envelope,
            "scientific_eligible": (
                plan["scientific_eligible"] and memory_within_envelope
            ),
        }
    )


def load_verified_e3_construction_snapshot(
    directory: str | Path,
    *,
    questions: Sequence[Question],
    prompts: Mapping[str, PromptSpec],
    protocol: E3Protocol | None = None,
) -> VerifiedE3ConstructionSnapshot:
    """Expose the complete journal through a verified, immutable snapshot API."""

    frozen_protocol = protocol or E3Protocol()
    verification = verify_e3_construction_work(
        directory,
        questions=questions,
        prompts=prompts,
        protocol=frozen_protocol,
        require_complete=True,
    )
    work, plan, schedule = _work_context(
        directory,
        questions=questions,
        prompts=prompts,
        protocol=frozen_protocol,
    )
    records, digests = _load_generations(
        work / "generations.jsonl",
        plan_identity=plan["plan_identity"],
        schedule=schedule,
    )
    return VerifiedE3ConstructionSnapshot(
        directory=work.absolute(),
        plan=_deep_freeze_json(plan),
        schedule=tuple(schedule),
        generations=tuple(records),
        generation_chain_head=digests[-1],
        scientific_eligible=bool(verification["scientific_eligible"]),
    )


def run_e3_construction(
    directory: str | Path,
    *,
    questions: Sequence[Question],
    prompts: Mapping[str, PromptSpec],
    runtime: E3ConstructionRuntime,
    protocol: E3Protocol | None = None,
    request_budget: int | None = None,
) -> Mapping[str, Any]:
    """Generate and process a prefix, reusing journaled outputs after interruption."""

    directory = validate_active_study_artifact_paths(
        {"E3 construction work": directory}
    )["E3 construction work"]
    frozen_protocol = protocol or E3Protocol()
    work, plan, schedule = _work_context(
        directory, questions=questions, prompts=prompts, protocol=frozen_protocol
    )
    if request_budget is not None and (
        type(request_budget) is not int or request_budget <= 0
    ):
        raise ConfigurationError("E3 request budget must be a positive integer")
    live_identity = _validated_json_mapping(
        runtime.runtime_identity(), label="E3 runtime identity"
    )
    if dict(live_identity) != plan["runtime_identity"]:
        raise FrozenArtifactError("E3 live runtime identity differs from frozen plan")
    question_map = {value.question_id: value for value in questions}
    with _exclusive_plan_lock(work / "plan.json"):
        _repair_torn_jsonl_tail(work / "generations.jsonl")
        _repair_torn_jsonl_tail(work / "sessions.jsonl")
        _repair_checkpoint_temporaries(work / "checkpoints")
        records, generation_digests = _load_generations(
            work / "generations.jsonl",
            plan_identity=plan["plan_identity"],
            schedule=schedule,
        )
        accumulator, checkpoint_head, checkpoint_index = _load_latest_checkpoint(
            work / "checkpoints",
            plan_identity=plan["plan_identity"],
            protocol=frozen_protocol,
            hidden_width=plan["hidden_width"],
            generations=records,
            generation_digests=generation_digests,
        )
        durable_processed_rows = accumulator.processed_rows
        sessions = _load_sessions(
            work / "sessions.jsonl", plan_identity=plan["plan_identity"]
        )
        session_head = _session_head(sessions)
        if sessions and sessions[-1]["event"] == "start":
            started = int(sessions[-1]["created_unix_ns"])
            session_head = _append_session(
                work / "sessions.jsonl",
                {
                    "schema_version": 1,
                    "event": "end",
                    "session_index": sessions[-1]["session_index"],
                    "plan_identity": plan["plan_identity"],
                    "status": "interrupted-recovered",
                    "processed_rows": accumulator.processed_rows,
                    "generation_chain_head": (
                        generation_digests[-1] if generation_digests else None
                    ),
                    "checkpoint_chain_head": checkpoint_head,
                    "wall_time_seconds": max(0.0, (time.time_ns() - started) / 1e9),
                    "peak_memory_bytes": 0,
                    "created_unix_ns": time.time_ns(),
                },
                session_head,
            )
            sessions = _load_sessions(
                work / "sessions.jsonl", plan_identity=plan["plan_identity"]
            )
        session_index = sum(row["event"] == "start" for row in sessions)
        started = time.time_ns()
        session_head = _append_session(
            work / "sessions.jsonl",
            {
                "schema_version": 1,
                "event": "start",
                "session_index": session_index,
                "plan_identity": plan["plan_identity"],
                "processed_rows": accumulator.processed_rows,
                "runtime_identity": plan["runtime_identity"],
                "created_unix_ns": started,
            },
            session_head,
        )
        peak_memory = 0
        handled = 0
        status = "partial"
        try:
            while accumulator.processed_rows < len(schedule) and (
                request_budget is None or handled < request_budget
            ):
                sequence = accumulator.processed_rows
                row = schedule[sequence]
                question = question_map[row.question_id]
                prompt = prompts[row.prompt_id]
                existing = records[sequence] if sequence < len(records) else None
                rendered = _render_for_record(runtime, question, prompt, existing)
                if existing is None:
                    generation = runtime.generate(
                        rendered, max_new_tokens=plan["max_new_tokens"]
                    )
                    existing = _new_generation_record(
                        row=row,
                        plan_identity=plan["plan_identity"],
                        question=question,
                        rendered=rendered,
                        generation=generation,
                    )
                    digest = _append_chained_json(
                        work / "generations.jsonl",
                        existing.to_dict(),
                        previous=generation_digests[-1] if generation_digests else None,
                        prefix="generation",
                    )
                    records.append(existing)
                    generation_digests.append(digest)
                    peak_memory = max(peak_memory, generation.peak_memory_bytes)
                values, capture_peak = _collect_observation(
                    runtime=runtime,
                    rendered=rendered,
                    record=existing,
                    protocol=frozen_protocol,
                    hidden_width=plan["hidden_width"],
                )
                _apply_observation(
                    accumulator,
                    record=existing,
                    values=values,
                    protocol=frozen_protocol,
                )
                accumulator.processed_rows += 1
                handled += 1
                peak_memory = max(peak_memory, capture_peak)
                if accumulator.processed_rows % plan["checkpoint_rows"] == 0:
                    _path, checkpoint_head = _write_checkpoint(
                        work / "checkpoints",
                        accumulator=accumulator,
                        plan_identity=plan["plan_identity"],
                        generation_chain_head=generation_digests[
                            accumulator.processed_rows - 1
                        ],
                        previous_checkpoint_sha256=checkpoint_head,
                        index=checkpoint_index,
                    )
                    checkpoint_index += 1
                    durable_processed_rows = accumulator.processed_rows
            if accumulator.processed_rows and (
                checkpoint_index == 0
                or accumulator.processed_rows % plan["checkpoint_rows"] != 0
            ):
                _path, checkpoint_head = _write_checkpoint(
                    work / "checkpoints",
                    accumulator=accumulator,
                    plan_identity=plan["plan_identity"],
                    generation_chain_head=generation_digests[
                        accumulator.processed_rows - 1
                    ],
                    previous_checkpoint_sha256=checkpoint_head,
                    index=checkpoint_index,
                )
                checkpoint_index += 1
                durable_processed_rows = accumulator.processed_rows
            status = (
                "complete" if accumulator.processed_rows == len(schedule) else "partial"
            )
        except BaseException:
            status = "error"
            if accumulator.processed_rows > durable_processed_rows:
                try:
                    _path, checkpoint_head = _write_checkpoint(
                        work / "checkpoints",
                        accumulator=accumulator,
                        plan_identity=plan["plan_identity"],
                        generation_chain_head=generation_digests[
                            accumulator.processed_rows - 1
                        ],
                        previous_checkpoint_sha256=checkpoint_head,
                        index=checkpoint_index,
                    )
                    durable_processed_rows = accumulator.processed_rows
                except BaseException:
                    # The original error is primary. The durable row count and chain
                    # remain at the last checkpoint and orphan temporaries are repaired
                    # before the next session.
                    pass
            raise
        finally:
            _append_session(
                work / "sessions.jsonl",
                {
                    "schema_version": 1,
                    "event": "end",
                    "session_index": session_index,
                    "plan_identity": plan["plan_identity"],
                    "status": status,
                    "processed_rows": (
                        durable_processed_rows
                        if status == "error"
                        else accumulator.processed_rows
                    ),
                    "generation_chain_head": (
                        generation_digests[-1] if generation_digests else None
                    ),
                    "checkpoint_chain_head": checkpoint_head,
                    "wall_time_seconds": max(0.0, (time.time_ns() - started) / 1e9),
                    "peak_memory_bytes": peak_memory,
                    "created_unix_ns": time.time_ns(),
                },
                session_head,
            )
    return verify_e3_construction_work(
        work,
        questions=questions,
        prompts=prompts,
        protocol=frozen_protocol,
        require_complete=False,
    )


def finalize_e3_vector_bundle(
    destination: str | Path,
    *,
    work_directory: str | Path,
    questions: Sequence[Question],
    prompts: Mapping[str, PromptSpec],
    protocol: E3Protocol | None = None,
    allow_non_scientific: bool = False,
) -> Mapping[str, Any]:
    """Publish normalized M1-R/M1-P vectors and reference RMS values atomically."""

    normalized = validate_active_study_artifact_paths(
        {
            "E3 vector bundle": destination,
            "E3 construction work": work_directory,
        }
    )
    destination = normalized["E3 vector bundle"]
    work_directory = normalized["E3 construction work"]
    frozen_protocol = protocol or E3Protocol()
    if type(allow_non_scientific) is not bool:
        raise ConfigurationError("E3 non-scientific publication override must be boolean")
    verification = verify_e3_construction_work(
        work_directory,
        questions=questions,
        prompts=prompts,
        protocol=frozen_protocol,
        require_complete=True,
    )
    if not verification["scientific_eligible"] and not allow_non_scientific:
        raise FrozenArtifactError(
            "E3 construction is valid but not eligible for scientific publication"
        )
    work, plan, schedule = _work_context(
        work_directory,
        questions=questions,
        prompts=prompts,
        protocol=frozen_protocol,
    )
    records, generation_digests = _load_generations(
        work / "generations.jsonl",
        plan_identity=plan["plan_identity"],
        schedule=schedule,
    )
    accumulator, checkpoint_head, _count = _load_latest_checkpoint(
        work / "checkpoints",
        plan_identity=plan["plan_identity"],
        protocol=frozen_protocol,
        hidden_width=plan["hidden_width"],
        generations=records,
        generation_digests=generation_digests,
    )
    means = accumulator.sums / accumulator.counts[..., None]
    differences = means[:, :, 0] - means[:, :, 1]
    norms = np.linalg.norm(differences, axis=-1)
    if (
        np.any(accumulator.counts == 0)
        or not np.isfinite(differences).all()
        or not np.isfinite(norms).all()
        or np.any(norms <= 0)
        or np.any(accumulator.rms_elements <= 0)
    ):
        raise DataValidationError("E3 vector bundle lacks non-degenerate C/I evidence")
    directions = (differences / norms[..., None]).astype(np.float32)
    reference_rms = np.sqrt(
        accumulator.rms_sum_squares / accumulator.rms_elements
    ).astype(np.float64)
    if not np.isfinite(reference_rms).all() or np.any(reference_rms <= 0):
        raise DataValidationError("E3 reference RMS is zero or invalid")
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite E3 vector bundle: {output}")
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        tensor_path = stage / "vectors.npz"
        with tensor_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                directions=directions,
                reference_rms=reference_rms,
                correct_counts=accumulator.counts[:, :, 0],
                incorrect_counts=accumulator.counts[:, :, 1],
            )
        body = {
            "schema_version": 1,
            "phase": "E3-construction",
            "plan_identity": plan["plan_identity"],
            "protocol": frozen_protocol.to_dict(),
            "prompt_axis": list(_PROMPTS),
            "extraction_axis": list(_EXTRACTIONS),
            "site_axis": [site.value for site in frozen_protocol.candidate_sites],
            "layer_axis": list(frozen_protocol.candidate_layers),
            "hidden_width": plan["hidden_width"],
            "rows_processed": accumulator.processed_rows,
            "response_pooling": frozen_protocol.response_pooling,
            "scientific_eligible": verification["scientific_eligible"],
            "maximum_peak_memory_bytes": verification["maximum_peak_memory_bytes"],
            "generation_chain_head": generation_digests[-1],
            "checkpoint_chain_head": checkpoint_head,
            "vectors_sha256": sha256_file(tensor_path),
            "data_fingerprint": stable_hash(
                {
                    "plan_identity": plan["plan_identity"],
                    "generation_chain_head": generation_digests[-1],
                    "checkpoint_chain_head": checkpoint_head,
                }
            ),
        }
        metadata = {**body, "metadata_digest": stable_hash(body)}
        (stage / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return verify_e3_vector_bundle(
        output,
        work_directory=work,
        questions=questions,
        prompts=prompts,
        protocol=frozen_protocol,
    )


def verify_e3_vector_bundle(
    directory: str | Path,
    *,
    work_directory: str | Path,
    questions: Sequence[Question],
    prompts: Mapping[str, PromptSpec],
    protocol: E3Protocol | None = None,
) -> Mapping[str, Any]:
    """Verify the final bundle against the exact completed construction work."""

    frozen_protocol = protocol or E3Protocol()
    work_verification = verify_e3_construction_work(
        work_directory,
        questions=questions,
        prompts=prompts,
        protocol=frozen_protocol,
        require_complete=True,
    )
    work, plan, schedule = _work_context(
        work_directory,
        questions=questions,
        prompts=prompts,
        protocol=frozen_protocol,
    )
    records, generation_digests = _load_generations(
        work / "generations.jsonl",
        plan_identity=plan["plan_identity"],
        schedule=schedule,
    )
    accumulator, checkpoint_head, _checkpoint_count = _load_latest_checkpoint(
        work / "checkpoints",
        plan_identity=plan["plan_identity"],
        protocol=frozen_protocol,
        hidden_width=plan["hidden_width"],
        generations=records,
        generation_digests=generation_digests,
    )
    source = Path(directory)
    if (
        source.is_symlink()
        or not source.is_dir()
        or {path.name for path in source.iterdir()} != {"metadata.json", "vectors.npz"}
        or any(path.is_symlink() or not path.is_file() for path in source.iterdir())
    ):
        raise FrozenArtifactError("E3 vector bundle inventory differs")
    try:
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read E3 vector metadata: {exc}") from exc
    body = dict(metadata)
    digest = body.pop("metadata_digest", None)
    expected_fingerprint = stable_hash(
        {
            "plan_identity": plan["plan_identity"],
            "generation_chain_head": generation_digests[-1],
            "checkpoint_chain_head": checkpoint_head,
        }
    )
    expected_metadata = {
        "schema_version": 1,
        "phase": "E3-construction",
        "plan_identity": plan["plan_identity"],
        "protocol": frozen_protocol.to_dict(),
        "prompt_axis": list(_PROMPTS),
        "extraction_axis": list(_EXTRACTIONS),
        "site_axis": [site.value for site in frozen_protocol.candidate_sites],
        "layer_axis": list(frozen_protocol.candidate_layers),
        "hidden_width": plan["hidden_width"],
        "rows_processed": accumulator.processed_rows,
        "response_pooling": frozen_protocol.response_pooling,
        "scientific_eligible": work_verification["scientific_eligible"],
        "maximum_peak_memory_bytes": work_verification["maximum_peak_memory_bytes"],
        "generation_chain_head": generation_digests[-1],
        "checkpoint_chain_head": checkpoint_head,
        "vectors_sha256": sha256_file(source / "vectors.npz"),
        "data_fingerprint": expected_fingerprint,
    }
    if (
        type(metadata) is not dict
        or digest != stable_hash(body)
        or body != expected_metadata
        or work_verification["plan_identity"] != plan["plan_identity"]
    ):
        raise FrozenArtifactError("E3 vector metadata differs from construction work")
    expected_shape = (
        len(_PROMPTS),
        len(_EXTRACTIONS),
        len(frozen_protocol.candidate_sites),
        len(frozen_protocol.candidate_layers),
    )
    try:
        with np.load(source / "vectors.npz", allow_pickle=False) as values:
            if set(values.files) != {
                "directions",
                "reference_rms",
                "correct_counts",
                "incorrect_counts",
            }:
                raise DataValidationError("E3 vector arrays differ")
            directions = values["directions"]
            reference_rms = values["reference_rms"]
            correct = values["correct_counts"]
            incorrect = values["incorrect_counts"]
    except (OSError, ValueError, DataValidationError) as exc:
        raise FrozenArtifactError(f"cannot read E3 vector arrays: {exc}") from exc
    if np.any(accumulator.counts == 0) or np.any(accumulator.rms_elements <= 0):
        raise FrozenArtifactError("E3 final checkpoint lacks C/I evidence")
    means = accumulator.sums / accumulator.counts[..., None]
    differences = means[:, :, 0] - means[:, :, 1]
    norms = np.linalg.norm(differences, axis=-1)
    expected_directions = (differences / norms[..., None]).astype(np.float32)
    expected_rms = np.sqrt(
        accumulator.rms_sum_squares / accumulator.rms_elements
    ).astype(np.float64)
    if (
        directions.dtype != np.float32
        or directions.shape != (*expected_shape, body["hidden_width"])
        or reference_rms.dtype != np.float64
        or reference_rms.shape != expected_shape
        or correct.dtype != np.int64
        or incorrect.dtype != np.int64
        or correct.shape != expected_shape
        or incorrect.shape != expected_shape
        or np.any(correct <= 0)
        or np.any(incorrect <= 0)
        or not np.isfinite(directions).all()
        or not np.allclose(np.linalg.norm(directions, axis=-1), 1.0, rtol=1e-5, atol=1e-6)
        or not np.isfinite(reference_rms).all()
        or np.any(reference_rms <= 0)
        or not np.array_equal(directions, expected_directions)
        or not np.array_equal(reference_rms, expected_rms)
        or not np.array_equal(correct, accumulator.counts[:, :, 0])
        or not np.array_equal(incorrect, accumulator.counts[:, :, 1])
    ):
        raise FrozenArtifactError("E3 vector geometry, counts, or RMS differs")
    return MappingProxyType(
        {
            "valid": True,
            "plan_identity": body["plan_identity"],
            "data_fingerprint": body["data_fingerprint"],
            "vector_count": int(np.prod(expected_shape)),
            "rows_processed": body["rows_processed"],
            "vectors_sha256": body["vectors_sha256"],
            "scientific_eligible": body["scientific_eligible"],
        }
    )

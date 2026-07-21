"""Frozen deterministic random and per-question Gaussian controls for E3."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from mfh.contracts import PromptSpec, Question
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.experiments.e3_construction import (
    load_verified_e3_construction_snapshot,
    verify_e3_vector_bundle,
)
from mfh.experiments.e3_schedule import E3Protocol, select_e3_screen_questions
from mfh.experiments.e3_selection import VerifiedE3StageSelection
from mfh.experiments.model_selection import validate_active_study_artifact_paths
from mfh.provenance import sha256_file, stable_hash

_EXTRACTIONS = ("M1-R", "M1-P")
_FILES = frozenset({"metadata.json", "random_norm.npy", "gaussian.npy"})
_ALGORITHM = "sha256-domain-seeded-numpy-pcg64-standard-normal-unit-v1"
_EXPECTED_T_DEV_IDS_DIGEST = (
    "7e7007f750af0c01f7e1eaae11cee546659c3520ae1357f2e8824753988dd2b3"
)
_EXPECTED_T_DEV_QUESTIONS_DIGEST = (
    "196fc0a3cbc08434230be4a6d8d323642b85c7ee1b750670b34264280f5ee4ea"
)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _question_ids(
    dev_questions: Sequence[Question],
    *,
    protocol: E3Protocol,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    frozen_questions = tuple(dev_questions)
    dev = tuple(value.question_id for value in frozen_questions)
    screen = select_e3_screen_questions(frozen_questions, protocol=protocol)
    if (
        len(dev) != protocol.dev_rows
        or len(screen) != protocol.screen_rows
        or len(set(dev)) != len(dev)
        or len(set(screen)) != len(screen)
        or not set(screen).issubset(dev)
        or any(
            value.benchmark != "triviaqa" or value.split != protocol.dev_split
            for value in frozen_questions
        )
    ):
        raise DataValidationError("E3 control question identities are invalid")
    return dev, screen


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


def _source(
    *,
    construction_directory: str | Path,
    vector_bundle_directory: str | Path,
    questions: Sequence[Question],
    prompts: Mapping[str, PromptSpec],
    scope_selection: VerifiedE3StageSelection,
    protocol: E3Protocol,
) -> tuple[Mapping[str, Any], Mapping[str, Any], int]:
    if isinstance(scope_selection, VerifiedE3StageSelection):
        scope_selection.assert_current()
    snapshot = load_verified_e3_construction_snapshot(
        construction_directory,
        questions=questions,
        prompts=prompts,
        protocol=protocol,
    )
    vectors = verify_e3_vector_bundle(
        vector_bundle_directory,
        work_directory=construction_directory,
        questions=questions,
        prompts=prompts,
        protocol=protocol,
    )
    if (
        not isinstance(scope_selection, VerifiedE3StageSelection)
        or scope_selection.stage != "scope"
        or scope_selection.falsified
        or set(scope_selection.selected) != set(_EXTRACTIONS)
        or scope_selection.source_plan_identity != snapshot.plan["plan_identity"]
    ):
        raise DataValidationError("E3 control materials require the bound scope selection")
    width = snapshot.plan["hidden_width"]
    if type(width) is not int or width <= 0:
        raise FrozenArtifactError("E3 control hidden width is invalid")
    return snapshot.plan, vectors, width


def _seed(*values: str) -> int:
    digest = hashlib.sha256("\x1f".join(values).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _unit_direction(width: int, *, seed: int) -> np.ndarray:
    values = np.random.Generator(np.random.PCG64(seed)).standard_normal(width)
    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm) or norm <= 0:
        raise DataValidationError("E3 random control direction is degenerate")
    return (values / norm).astype(np.float32)


def _direction(
    *,
    domain: str,
    plan_identity: str,
    vector_fingerprint: str,
    selection_digest: str,
    extraction: str,
    question_id: str,
    width: int,
) -> np.ndarray:
    return _unit_direction(
        width,
        seed=_seed(
            domain,
            plan_identity,
            vector_fingerprint,
            selection_digest,
            extraction,
            question_id,
        ),
    )


def _body(
    *,
    plan: Mapping[str, Any],
    vectors: Mapping[str, Any],
    scope_selection: VerifiedE3StageSelection,
    protocol: E3Protocol,
    dev: Sequence[str],
    screen: Sequence[str],
    dev_questions_digest: str,
    width: int,
    random_sha256: str,
    gaussian_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase": "E3-fixed-controls",
        "algorithm": _ALGORITHM,
        "construction_plan_identity": plan["plan_identity"],
        "vector_data_fingerprint": vectors["data_fingerprint"],
        "scope_selection_digest": scope_selection.selection_digest,
        "extraction_axis": list(_EXTRACTIONS),
        "dev_question_ids_digest": stable_hash(list(dev)),
        "dev_questions_digest": dev_questions_digest,
        "screen_question_ids_digest": stable_hash(list(screen)),
        "dev_rows": len(dev),
        "screen_rows": len(screen),
        "hidden_width": width,
        "random_norm_sha256": random_sha256,
        "gaussian_sha256": gaussian_sha256,
        "scientific_eligible": bool(
            plan["scientific_eligible"]
            and vectors["scientific_eligible"]
            and scope_selection.scientific_eligible
            and scope_selection.question_ids_digest == stable_hash(list(screen))
            and stable_hash(list(dev)) == _EXPECTED_T_DEV_IDS_DIGEST
            and dev_questions_digest == _EXPECTED_T_DEV_QUESTIONS_DIGEST
            and protocol.scientific_eligible
        ),
    }


def _write_arrays(
    directory: Path,
    *,
    plan_identity: str,
    vector_fingerprint: str,
    selection_digest: str,
    dev: Sequence[str],
    width: int,
) -> tuple[str, str]:
    random_path = directory / "random_norm.npy"
    random = np.lib.format.open_memmap(
        random_path, mode="w+", dtype=np.float32, shape=(len(_EXTRACTIONS), width)
    )
    for extraction_index, extraction in enumerate(_EXTRACTIONS):
        random[extraction_index] = _direction(
            domain="e3-random-norm",
            plan_identity=plan_identity,
            vector_fingerprint=vector_fingerprint,
            selection_digest=selection_digest,
            extraction=extraction,
            question_id="fixed",
            width=width,
        )
    random.flush()
    del random
    gaussian_path = directory / "gaussian.npy"
    gaussian = np.lib.format.open_memmap(
        gaussian_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(_EXTRACTIONS), len(dev), width),
    )
    for extraction_index, extraction in enumerate(_EXTRACTIONS):
        for question_index, question_id in enumerate(dev):
            gaussian[extraction_index, question_index] = _direction(
                domain="e3-per-question-gaussian",
                plan_identity=plan_identity,
                vector_fingerprint=vector_fingerprint,
                selection_digest=selection_digest,
                extraction=extraction,
                question_id=question_id,
                width=width,
            )
    gaussian.flush()
    del gaussian
    return sha256_file(random_path), sha256_file(gaussian_path)


def write_e3_fixed_control_materials(
    destination: str | Path,
    *,
    construction_directory: str | Path,
    vector_bundle_directory: str | Path,
    questions: Sequence[Question],
    prompts: Mapping[str, PromptSpec],
    scope_selection: VerifiedE3StageSelection,
    dev_questions: Sequence[Question],
    protocol: E3Protocol | None = None,
) -> Mapping[str, Any]:
    """Materialize deterministic control vectors before any E3 control evaluation."""

    normalized = validate_active_study_artifact_paths(
        {
            "E3 control materials": destination,
            "E3 construction work": construction_directory,
            "E3 vector bundle": vector_bundle_directory,
        }
    )
    destination = normalized["E3 control materials"]
    construction_directory = normalized["E3 construction work"]
    vector_bundle_directory = normalized["E3 vector bundle"]
    frozen_protocol = protocol or E3Protocol()
    dev, screen = _question_ids(dev_questions, protocol=frozen_protocol)
    dev_questions_digest = _questions_digest(dev_questions)
    plan, vectors, width = _source(
        construction_directory=construction_directory,
        vector_bundle_directory=vector_bundle_directory,
        questions=questions,
        prompts=prompts,
        scope_selection=scope_selection,
        protocol=frozen_protocol,
    )
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite E3 control materials: {output}")
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        random_sha, gaussian_sha = _write_arrays(
            stage,
            plan_identity=str(plan["plan_identity"]),
            vector_fingerprint=str(vectors["data_fingerprint"]),
            selection_digest=scope_selection.selection_digest,
            dev=dev,
            width=width,
        )
        body = _body(
            plan=plan,
            vectors=vectors,
            scope_selection=scope_selection,
            protocol=frozen_protocol,
            dev=dev,
            screen=screen,
            dev_questions_digest=dev_questions_digest,
            width=width,
            random_sha256=random_sha,
            gaussian_sha256=gaussian_sha,
        )
        metadata = {**body, "metadata_digest": stable_hash(body)}
        metadata_path = stage / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return verify_e3_fixed_control_materials(
        output,
        construction_directory=construction_directory,
        vector_bundle_directory=vector_bundle_directory,
        questions=questions,
        prompts=prompts,
        scope_selection=scope_selection,
        dev_questions=dev_questions,
        protocol=frozen_protocol,
    )


def verify_e3_fixed_control_materials(
    directory: str | Path,
    *,
    construction_directory: str | Path,
    vector_bundle_directory: str | Path,
    questions: Sequence[Question],
    prompts: Mapping[str, PromptSpec],
    scope_selection: VerifiedE3StageSelection,
    dev_questions: Sequence[Question],
    protocol: E3Protocol | None = None,
) -> Mapping[str, Any]:
    frozen_protocol = protocol or E3Protocol()
    dev, screen = _question_ids(dev_questions, protocol=frozen_protocol)
    dev_questions_digest = _questions_digest(dev_questions)
    plan, vectors, width = _source(
        construction_directory=construction_directory,
        vector_bundle_directory=vector_bundle_directory,
        questions=questions,
        prompts=prompts,
        scope_selection=scope_selection,
        protocol=frozen_protocol,
    )
    source = Path(directory)
    if (
        source.is_symlink()
        or not source.is_dir()
        or {value.name for value in source.iterdir()} != _FILES
        or any(value.is_symlink() or not value.is_file() for value in source.iterdir())
    ):
        raise FrozenArtifactError("E3 fixed-control inventory differs")
    try:
        text = (source / "metadata.json").read_text(encoding="utf-8")
        metadata = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read E3 fixed-control metadata: {exc}") from exc
    if type(metadata) is not dict:
        raise FrozenArtifactError("E3 fixed-control metadata schema differs")
    body = dict(metadata)
    digest = body.pop("metadata_digest", None)
    expected = _body(
        plan=plan,
        vectors=vectors,
        scope_selection=scope_selection,
        protocol=frozen_protocol,
        dev=dev,
        screen=screen,
        dev_questions_digest=dev_questions_digest,
        width=width,
        random_sha256=sha256_file(source / "random_norm.npy"),
        gaussian_sha256=sha256_file(source / "gaussian.npy"),
    )
    expected_text = (
        json.dumps(
            {**expected, "metadata_digest": stable_hash(expected)},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    if digest != stable_hash(body) or text != expected_text:
        raise FrozenArtifactError("E3 fixed-control metadata differs from sources")
    try:
        random = np.load(source / "random_norm.npy", mmap_mode="r", allow_pickle=False)
        gaussian = np.load(source / "gaussian.npy", mmap_mode="r", allow_pickle=False)
        if (
            random.dtype != np.float32
            or random.shape != (len(_EXTRACTIONS), width)
            or gaussian.dtype != np.float32
            or gaussian.shape != (len(_EXTRACTIONS), len(dev), width)
            or not np.isfinite(random).all()
            or not np.isfinite(gaussian).all()
            or not np.allclose(np.linalg.norm(random, axis=1), 1.0, rtol=1e-5, atol=1e-6)
            or not np.allclose(
                np.linalg.norm(gaussian, axis=2), 1.0, rtol=1e-5, atol=1e-6
            )
        ):
            raise DataValidationError("control array geometry differs")
        for extraction_index, extraction in enumerate(_EXTRACTIONS):
            expected_random = _direction(
                domain="e3-random-norm",
                plan_identity=str(plan["plan_identity"]),
                vector_fingerprint=str(vectors["data_fingerprint"]),
                selection_digest=scope_selection.selection_digest,
                extraction=extraction,
                question_id="fixed",
                width=width,
            )
            if not np.array_equal(random[extraction_index], expected_random):
                raise DataValidationError("random-norm content differs")
            for question_index, question_id in enumerate(dev):
                expected_gaussian = _direction(
                    domain="e3-per-question-gaussian",
                    plan_identity=str(plan["plan_identity"]),
                    vector_fingerprint=str(vectors["data_fingerprint"]),
                    selection_digest=scope_selection.selection_digest,
                    extraction=extraction,
                    question_id=question_id,
                    width=width,
                )
                if not np.array_equal(
                    gaussian[extraction_index, question_index], expected_gaussian
                ):
                    raise DataValidationError("per-question Gaussian content differs")
    except (OSError, ValueError, DataValidationError) as exc:
        raise FrozenArtifactError(f"E3 fixed-control arrays differ: {exc}") from exc
    return MappingProxyType(
        {
            "valid": True,
            "metadata_digest": stable_hash(expected),
            "random_norm_sha256": expected["random_norm_sha256"],
            "gaussian_sha256": expected["gaussian_sha256"],
            "dev_rows": len(dev),
            "dev_question_ids_digest": expected["dev_question_ids_digest"],
            "hidden_width": width,
            "scientific_eligible": expected["scientific_eligible"],
        }
    )


def load_e3_fixed_control_direction(
    directory: str | Path,
    *,
    control: str,
    extraction_method: str,
    question_id: str | None = None,
    expected_metadata_digest: str,
    dev_question_ids: Sequence[str],
) -> np.ndarray:
    """Load one immutable unit control direction from a verified bundle."""

    if extraction_method not in _EXTRACTIONS:
        raise DataValidationError("E3 control extraction identity is invalid")
    source = Path(directory)
    try:
        metadata_text = (source / "metadata.json").read_text(encoding="utf-8")
        metadata = json.loads(
            metadata_text, object_pairs_hook=_reject_duplicate_keys
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot load E3 fixed-control metadata: {exc}") from exc
    if type(metadata) is not dict:
        raise FrozenArtifactError("E3 fixed-control metadata schema differs")
    body = dict(metadata)
    metadata_digest = body.pop("metadata_digest", None)
    frozen_dev = tuple(dev_question_ids)
    if (
        metadata_digest != stable_hash(body)
        or metadata_digest != expected_metadata_digest
        or body.get("dev_question_ids_digest") != stable_hash(list(frozen_dev))
        or len(frozen_dev) != body.get("dev_rows")
        or len(set(frozen_dev)) != len(frozen_dev)
    ):
        raise FrozenArtifactError("E3 fixed-control verified identity differs")
    extraction_index = _EXTRACTIONS.index(extraction_method)
    if control == "random-norm" and question_id is None:
        path = source / "random_norm.npy"
        index: int | tuple[int, int] = extraction_index
        expected_sha256 = body.get("random_norm_sha256")
    elif control == "gaussian" and type(question_id) is str and question_id in frozen_dev:
        path = source / "gaussian.npy"
        index = (extraction_index, frozen_dev.index(question_id))
        expected_sha256 = body.get("gaussian_sha256")
    else:
        raise DataValidationError("E3 fixed-control lookup is invalid")
    try:
        with path.open("rb") as handle:
            digest = hashlib.sha256()
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
            if digest.hexdigest() != expected_sha256:
                raise FrozenArtifactError(
                    "E3 fixed-control array changed after verification"
                )
            handle.seek(0)
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                    handle
                )
            elif version in {(2, 0), (3, 0)}:
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                    handle
                )
            else:
                raise DataValidationError("unsupported E3 fixed-control NPY version")
            if dtype != np.dtype(np.float32) or fortran_order:
                raise DataValidationError("E3 fixed-control NPY layout differs")
            width = body.get("hidden_width")
            expected_shape = (
                (len(_EXTRACTIONS), width)
                if control == "random-norm"
                else (len(_EXTRACTIONS), len(frozen_dev), width)
            )
            if shape != expected_shape or type(width) is not int:
                raise DataValidationError("E3 fixed-control NPY shape differs")
            row_index = (
                index
                if isinstance(index, int)
                else index[0] * len(frozen_dev) + index[1]
            )
            handle.seek(handle.tell() + row_index * width * dtype.itemsize)
            payload = handle.read(width * dtype.itemsize)
            if len(payload) != width * dtype.itemsize:
                raise DataValidationError("E3 fixed-control NPY row is truncated")
            direction = np.frombuffer(payload, dtype=dtype).copy()
    except (OSError, ValueError, IndexError, DataValidationError) as exc:
        raise FrozenArtifactError(f"cannot load E3 fixed-control direction: {exc}") from exc
    norm = float(np.linalg.norm(direction)) if direction.ndim == 1 else math.nan
    if (
        direction.ndim != 1
        or direction.shape != (body.get("hidden_width"),)
        or not np.isfinite(direction).all()
        or not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-6)
    ):
        raise FrozenArtifactError("E3 fixed-control direction geometry differs")
    direction.setflags(write=False)
    return direction

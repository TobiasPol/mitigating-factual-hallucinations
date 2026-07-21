"""Frozen lexical and sentence-embedding contamination scans."""

from __future__ import annotations

import heapq
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from mfh.config import SemanticContaminationProtocol
from mfh.contracts import Question
from mfh.data.contamination import character_ngrams, find_overlaps, jaccard
from mfh.data.io import read_questions, write_questions
from mfh.data.normalization import normalize_question
from mfh.data.splits import exclude_exact_duplicate_groups
from mfh.errors import DataValidationError, FrozenArtifactError, OptionalDependencyError
from mfh.provenance import (
    canonical_json,
    environment_snapshot,
    sha256_file,
    sha256_path,
    stable_hash,
)

FloatArray = NDArray[np.float32]
Progress = Callable[[str], None]

_BUNDLE_FILES = {
    "clean-source.jsonl",
    "curated-source.jsonl",
    "duplicate-curation.json",
    "lexical-matches.jsonl",
    "manifest.json",
    "manual-review-queue.jsonl",
    "semantic-matches.jsonl",
    "source-embeddings.npy",
    "target-embeddings.npy",
    "targets.jsonl",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LEXICAL_MATCH_KEYS = {
    "source_question_id",
    "target_question_id",
    "exact",
    "ngram_similarity",
}


@dataclass(frozen=True, slots=True)
class SemanticOverlapPair:
    source_question_id: str
    target_question_id: str
    similarity: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReviewPair:
    source_question_id: str
    target_question_id: str
    source_text: str
    target_text: str
    similarity: float
    automatic_semantic_match: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _emit(progress: Progress | None, message: str) -> None:
    if progress is not None:
        progress(message)


def verify_semantic_model_directory(
    protocol: SemanticContaminationProtocol,
    model_directory: str | Path,
) -> Path:
    candidate = Path(model_directory)
    if candidate.is_symlink() or not candidate.is_dir():
        raise DataValidationError("semantic model artifact must be a regular directory")
    root = candidate.resolve()
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise DataValidationError("semantic model artifact cannot contain symlinks")
        if path.is_file():
            observed_files.add(path.relative_to(root).as_posix())
        elif path.is_dir():
            observed_directories.add(path.relative_to(root).as_posix())
        else:
            raise DataValidationError("semantic model artifact contains a special file")
    expected_directories = {
        parent.as_posix()
        for filename in protocol.required_files
        for parent in Path(filename).parents
        if parent != Path(".")
    }
    if observed_files != set(protocol.required_files):
        raise DataValidationError(
            "semantic model inventory differs from the pinned protocol: "
            f"missing={sorted(set(protocol.required_files) - observed_files)}, "
            f"extra={sorted(observed_files - set(protocol.required_files))}"
        )
    if observed_directories != expected_directories:
        raise DataValidationError(
            "semantic model directory inventory differs from the pinned protocol: "
            f"missing={sorted(expected_directories - observed_directories)}, "
            f"extra={sorted(observed_directories - expected_directories)}"
        )
    if sha256_path(root) != protocol.model_artifact_tree_sha256:
        raise DataValidationError("semantic model tree differs from its pinned SHA-256")
    try:
        pooling = json.loads((root / "1_Pooling/config.json").read_text(encoding="utf-8"))
        sentence_config = json.loads(
            (root / "sentence_bert_config.json").read_text(encoding="utf-8")
        )
        modules = json.loads((root / "modules.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot read semantic model metadata: {exc}") from exc
    if pooling != {
        "word_embedding_dimension": protocol.embedding_dimension,
        "pooling_mode_cls_token": False,
        "pooling_mode_mean_tokens": True,
        "pooling_mode_max_tokens": False,
        "pooling_mode_mean_sqrt_len_tokens": False,
    }:
        raise DataValidationError("semantic model pooling configuration differs")
    if sentence_config.get("max_seq_length") != protocol.max_length:
        raise DataValidationError("semantic model maximum sequence length differs")
    if not isinstance(modules, list) or [value.get("type") for value in modules] != [
        "sentence_transformers.models.Transformer",
        "sentence_transformers.models.Pooling",
        "sentence_transformers.models.Normalize",
    ]:
        raise DataValidationError("semantic model module pipeline differs")
    return root


class _TransformerEncoder:
    def __init__(
        self,
        protocol: SemanticContaminationProtocol,
        model_directory: Path,
        *,
        progress: Progress | None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise OptionalDependencyError(
                "semantic overlap requires the research dependencies"
            ) from exc
        torch.manual_seed(0)
        torch.set_num_threads(protocol.torch_num_threads)
        torch.use_deterministic_algorithms(True)
        self._torch = torch
        self._protocol = protocol
        self._progress = progress
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_directory,
            local_files_only=True,
            use_fast=True,
        )
        self._model = AutoModel.from_pretrained(
            model_directory,
            local_files_only=True,
        ).to(device="cpu", dtype=torch.float32)
        self._model.eval()
        hidden_size = int(getattr(self._model.config, "hidden_size", -1))
        if hidden_size != protocol.embedding_dimension:
            raise DataValidationError(
                f"semantic model hidden size {hidden_size} differs from protocol"
            )

    def encode(self, questions: Sequence[Question], *, label: str) -> FloatArray:
        torch = self._torch
        output = np.empty(
            (len(questions), self._protocol.embedding_dimension),
            dtype=np.float32,
        )
        batch_size = self._protocol.encode_batch_size
        with torch.inference_mode():
            for start in range(0, len(questions), batch_size):
                stop = min(start + batch_size, len(questions))
                encoded = self._tokenizer(
                    [question.text for question in questions[start:stop]],
                    padding=True,
                    truncation=True,
                    max_length=self._protocol.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to("cpu") for key, value in encoded.items()}
                token_embeddings = self._model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
                pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
                output[start:stop] = normalized.detach().cpu().numpy().astype(np.float32)
                if start == 0 or stop == len(questions) or stop % (batch_size * 50) == 0:
                    _emit(self._progress, f"encoded {label}: {stop}/{len(questions)}")
        if not np.isfinite(output).all():
            raise DataValidationError("semantic encoder produced non-finite embeddings")
        return output


def _validate_embeddings(
    values: FloatArray,
    *,
    rows: int,
    dimension: int,
    context: str,
) -> None:
    if values.shape != (rows, dimension) or values.dtype != np.float32:
        raise DataValidationError(
            f"{context} embeddings have shape/dtype {values.shape}/{values.dtype}"
        )
    if not np.isfinite(values).all():
        raise DataValidationError(f"{context} embeddings contain non-finite values")
    norms = np.linalg.norm(values, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-5):
        raise DataValidationError(f"{context} embeddings are not unit normalized")


def semantic_overlap_pairs(
    source_questions: Sequence[Question],
    target_questions: Sequence[Question],
    source_embeddings: FloatArray,
    target_embeddings: FloatArray,
    *,
    threshold: float,
    review_top_k: int,
    similarity_batch_size: int,
    progress: Progress | None = None,
) -> tuple[tuple[SemanticOverlapPair, ...], tuple[ReviewPair, ...]]:
    if not 0 <= threshold <= 1 or review_top_k <= 0 or similarity_batch_size <= 0:
        raise DataValidationError("semantic overlap parameters are invalid")
    if source_embeddings.ndim != 2:
        raise DataValidationError("source embeddings must be a matrix")
    dimension = int(source_embeddings.shape[1])
    _validate_embeddings(
        source_embeddings,
        rows=len(source_questions),
        dimension=dimension,
        context="source",
    )
    _validate_embeddings(
        target_embeddings,
        rows=len(target_questions),
        dimension=dimension,
        context="target",
    )
    matches: list[SemanticOverlapPair] = []
    review_candidates: list[tuple[float, int, int]] = []
    per_batch_k = min(review_top_k, len(source_questions) * similarity_batch_size)
    for target_start in range(0, len(target_questions), similarity_batch_size):
        target_stop = min(target_start + similarity_batch_size, len(target_questions))
        scores = np.asarray(
            target_embeddings[target_start:target_stop] @ source_embeddings.T,
            dtype=np.float32,
        )
        target_offsets, source_indices = np.nonzero(scores >= threshold)
        for target_offset, source_index in zip(target_offsets, source_indices, strict=True):
            target_index = target_start + int(target_offset)
            matches.append(
                SemanticOverlapPair(
                    source_question_id=source_questions[int(source_index)].question_id,
                    target_question_id=target_questions[target_index].question_id,
                    similarity=float(scores[int(target_offset), int(source_index)]),
                )
            )
        flat = scores.reshape(-1)
        current_k = min(per_batch_k, flat.size)
        if current_k:
            width = scores.shape[1]
            if current_k == flat.size:
                indices = tuple(range(flat.size))
            else:
                cutoff = float(np.partition(flat, flat.size - current_k)[flat.size - current_k])
                better = [int(index) for index in np.flatnonzero(flat > cutoff)]
                remaining = current_k - len(better)

                def tie_key(
                    index: int,
                    target_base: int = target_start,
                    matrix_width: int = width,
                ) -> tuple[str, str]:
                    return (
                        target_questions[target_base + index // matrix_width].question_id,
                        source_questions[index % matrix_width].question_id,
                    )

                tied = (int(index) for index in np.flatnonzero(flat == cutoff))
                indices = (*better, *heapq.nsmallest(remaining, tied, key=tie_key))
            review_candidates.extend(
                (
                    float(flat[int(index)]),
                    target_start + int(index) // width,
                    int(index) % width,
                )
                for index in indices
            )
        _emit(progress, f"compared semantic targets: {target_stop}/{len(target_questions)}")
    matches.sort(
        key=lambda value: (-value.similarity, value.target_question_id, value.source_question_id)
    )
    ranked = sorted(
        review_candidates,
        key=lambda value: (
            -value[0],
            target_questions[value[1]].question_id,
            source_questions[value[2]].question_id,
        ),
    )[:review_top_k]
    review = tuple(
        ReviewPair(
            source_question_id=source_questions[source_index].question_id,
            target_question_id=target_questions[target_index].question_id,
            source_text=source_questions[source_index].text,
            target_text=target_questions[target_index].text,
            similarity=score,
            automatic_semantic_match=score >= threshold,
        )
        for score, target_index, source_index in ranked
    )
    return tuple(matches), review


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.write_text(
        "".join(f"{canonical_json(row)}\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    values: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise DataValidationError(f"{path.name} contains a non-object row")
                values.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot parse {path.name}: {exc}") from exc
    return tuple(values)


def _artifact_descriptor(path: Path) -> dict[str, object]:
    return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def _questions_from_paths(paths: Sequence[str | Path]) -> tuple[Question, ...]:
    values = tuple(question for path in paths for question in read_questions(path))
    ids = [question.question_id for question in values]
    if len(ids) != len(set(ids)):
        raise DataValidationError("semantic target inputs contain duplicate question IDs")
    return values


def write_contamination_bundle(
    directory: str | Path,
    *,
    protocol: SemanticContaminationProtocol,
    model_directory: str | Path,
    triviaqa_source: str | Path,
    target_sources: Sequence[str | Path],
    progress: Progress | None = None,
) -> Mapping[str, Any]:
    """Run and freeze duplicate, lexical, and semantic contamination checks."""

    destination = Path(directory)
    if destination.exists():
        raise FrozenArtifactError(f"refusing to overwrite contamination bundle: {destination}")
    if not target_sources:
        raise DataValidationError("contamination scan requires at least one target source")
    model_root = verify_semantic_model_directory(protocol, model_directory)
    raw_source = tuple(read_questions(triviaqa_source))
    if not raw_source or {question.benchmark for question in raw_source} != {"triviaqa"}:
        raise DataValidationError("contamination source must contain only TriviaQA questions")
    curation = exclude_exact_duplicate_groups(raw_source)
    curated = curation.questions
    targets = _questions_from_paths(target_sources)
    target_benchmarks = {question.benchmark for question in targets}
    if target_benchmarks != {"simpleqa_verified", "aa_omniscience_public_600"}:
        raise DataValidationError(
            "contamination targets must be exactly SimpleQA Verified and AA Public-600"
        )
    if {question.question_id for question in curated} & {
        question.question_id for question in targets
    }:
        raise DataValidationError("contamination source and target IDs overlap")

    _emit(progress, f"lexical scan: {len(curated)} source x {len(targets)} target rows")
    lexical = find_overlaps(
        curated,
        targets,
        ngram_threshold=protocol.lexical_ngram_threshold,
    )
    _emit(progress, f"lexical scan complete: {len(lexical.matches)} matches")

    encoder = _TransformerEncoder(protocol, model_root, progress=progress)
    source_embeddings = encoder.encode(curated, label="TriviaQA source")
    target_embeddings = encoder.encode(targets, label="OOD targets")
    semantic, review = semantic_overlap_pairs(
        curated,
        targets,
        source_embeddings,
        target_embeddings,
        threshold=protocol.semantic_similarity_threshold,
        review_top_k=protocol.review_top_k,
        similarity_batch_size=protocol.similarity_batch_size,
        progress=progress,
    )
    removed_ids = set(lexical.source_ids_to_remove) | {pair.source_question_id for pair in semantic}
    clean = tuple(question for question in curated if question.question_id not in removed_ids)

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent))
    try:
        write_questions(stage / "curated-source.jsonl", curated)
        write_questions(stage / "clean-source.jsonl", clean)
        write_questions(stage / "targets.jsonl", targets)
        (stage / "duplicate-curation.json").write_text(
            json.dumps(curation.report.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        _write_jsonl(
            stage / "lexical-matches.jsonl",
            (
                {
                    "source_question_id": match.source_question_id,
                    "target_question_id": match.target_question_id,
                    "exact": match.exact,
                    "ngram_similarity": match.ngram_similarity,
                }
                for match in sorted(
                    lexical.matches,
                    key=lambda value: (
                        -value.ngram_similarity,
                        value.target_question_id,
                        value.source_question_id,
                    ),
                )
            ),
        )
        _write_jsonl(stage / "semantic-matches.jsonl", (pair.to_dict() for pair in semantic))
        _write_jsonl(stage / "manual-review-queue.jsonl", (pair.to_dict() for pair in review))
        np.save(stage / "source-embeddings.npy", source_embeddings, allow_pickle=False)
        np.save(stage / "target-embeddings.npy", target_embeddings, allow_pickle=False)
        artifact_files = sorted(_BUNDLE_FILES - {"manifest.json"})
        artifacts = {
            filename: _artifact_descriptor(stage / filename) for filename in artifact_files
        }
        target_inputs = [
            {
                "filename": Path(path).name,
                "sha256": sha256_file(path),
                "question_count": sum(1 for _ in read_questions(path)),
            }
            for path in target_sources
        ]
        body: dict[str, Any] = {
            "schema_version": 1,
            "protocol": protocol.to_dict(),
            "protocol_digest": protocol.digest,
            "model": {
                "repository": protocol.model_repository,
                "revision": protocol.model_revision,
                "artifact_tree_sha256": protocol.model_artifact_tree_sha256,
            },
            "inputs": {
                "triviaqa": {
                    "filename": Path(triviaqa_source).name,
                    "sha256": sha256_file(triviaqa_source),
                    "question_count": len(raw_source),
                },
                "targets": target_inputs,
            },
            "counts": {
                "raw_source": len(raw_source),
                "curated_source": len(curated),
                "duplicate_rows_excluded": curation.report.excluded_question_count,
                "target": len(targets),
                "lexical_matches": len(lexical.matches),
                "semantic_matches": len(semantic),
                "contaminated_source_rows": len(removed_ids),
                "clean_source": len(clean),
                "manual_review_queue": len(review),
            },
            "contaminated_source_ids": sorted(removed_ids),
            "contaminated_source_ids_sha256": stable_hash(sorted(removed_ids)),
            "manual_review": {
                "required": True,
                "status": "pending",
                "selection": "global-highest-cosine-similarity",
                "selection_count": len(review),
                "selection_sha256": stable_hash([pair.to_dict() for pair in review]),
            },
            "artifacts": artifacts,
            "environment": environment_snapshot(("numpy", "torch", "transformers", "tokenizers")),
        }
        manifest = {**body, "manifest_digest": stable_hash(body)}
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return verify_contamination_bundle(
        destination,
        expected_protocol=protocol,
        model_directory=model_root,
        triviaqa_source=triviaqa_source,
        target_sources=target_sources,
        expected_manifest_digest=str(manifest["manifest_digest"]),
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot read contamination manifest: {exc}") from exc
    if not isinstance(value, dict):
        raise DataValidationError("contamination manifest must be an object")
    body = dict(value)
    digest = body.pop("manifest_digest", None)
    if digest != stable_hash(body):
        raise DataValidationError("contamination manifest digest mismatch")
    return value


def _verify_bundle_inventory(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise DataValidationError("contamination bundle must be a regular directory")
    observed: set[str] = set()
    observed_directories: set[str] = set()
    for value in path.rglob("*"):
        if value.is_symlink():
            raise DataValidationError("contamination bundle cannot contain symlinks")
        if value.is_file():
            observed.add(value.relative_to(path).as_posix())
        elif value.is_dir():
            observed_directories.add(value.relative_to(path).as_posix())
        else:
            raise DataValidationError("contamination bundle contains a special file")
    if observed != _BUNDLE_FILES:
        raise DataValidationError(
            "contamination bundle inventory differs: "
            f"missing={sorted(_BUNDLE_FILES - observed)}, extra={sorted(observed - _BUNDLE_FILES)}"
        )
    if observed_directories:
        raise DataValidationError(
            f"contamination bundle contains undeclared directories: {sorted(observed_directories)}"
        )


def _validate_lexical_rows(
    rows: Sequence[Mapping[str, Any]],
    curated: Sequence[Question],
    targets: Sequence[Question],
    *,
    threshold: float,
) -> set[str]:
    """Validate saved lexical claims before the exhaustive replay discovers omissions."""

    source_by_id = {question.question_id: question for question in curated}
    target_by_id = {question.question_id: question for question in targets}
    seen_pairs: set[tuple[str, str]] = set()
    normalized_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        if set(row) != _LEXICAL_MATCH_KEYS:
            raise DataValidationError(f"lexical match row {index} has an invalid schema")
        source_id = row["source_question_id"]
        target_id = row["target_question_id"]
        exact = row["exact"]
        similarity = row["ngram_similarity"]
        if not isinstance(source_id, str) or source_id not in source_by_id:
            raise DataValidationError(f"lexical match row {index} has an unknown source ID")
        if not isinstance(target_id, str) or target_id not in target_by_id:
            raise DataValidationError(f"lexical match row {index} has an unknown target ID")
        if type(exact) is not bool:
            raise DataValidationError(f"lexical match row {index} has a non-boolean exact flag")
        if (
            isinstance(similarity, bool)
            or not isinstance(similarity, (int, float))
            or not math.isfinite(float(similarity))
            or not 0 <= float(similarity) <= 1
        ):
            raise DataValidationError(f"lexical match row {index} has an invalid similarity")
        pair = (source_id, target_id)
        if pair in seen_pairs:
            raise DataValidationError("lexical match evidence contains a duplicate pair")
        seen_pairs.add(pair)
        source = source_by_id[source_id]
        target = target_by_id[target_id]
        expected_exact = normalize_question(source.text) == normalize_question(target.text)
        expected_similarity = jaccard(character_ngrams(source.text), character_ngrams(target.text))
        if exact is not expected_exact or float(similarity) != expected_similarity:
            raise DataValidationError(
                f"lexical match row {index} differs from its referenced questions"
            )
        if not expected_exact and expected_similarity < threshold:
            raise DataValidationError(f"lexical match row {index} is below the frozen threshold")
        normalized_rows.append(
            {
                "source_question_id": source_id,
                "target_question_id": target_id,
                "exact": exact,
                "ngram_similarity": float(similarity),
            }
        )
    expected_order = sorted(
        normalized_rows,
        key=lambda row: (
            -float(cast(int | float, row["ngram_similarity"])),
            str(row["target_question_id"]),
            str(row["source_question_id"]),
        ),
    )
    if normalized_rows != expected_order:
        raise DataValidationError("lexical match evidence is not in canonical order")
    return {str(row["source_question_id"]) for row in normalized_rows}


def verify_contamination_bundle(
    directory: str | Path,
    *,
    expected_protocol: SemanticContaminationProtocol,
    model_directory: str | Path,
    triviaqa_source: str | Path,
    target_sources: Sequence[str | Path],
    expected_manifest_digest: str,
    replay_embeddings: bool = False,
    progress: Progress | None = None,
) -> Mapping[str, Any]:
    """Verify a frozen scan against an external digest and replay lexical discovery."""

    if not _SHA256.fullmatch(expected_manifest_digest):
        raise DataValidationError("expected contamination manifest digest is not a SHA-256")
    root = Path(directory).absolute()
    _verify_bundle_inventory(root)
    model_root = verify_semantic_model_directory(expected_protocol, model_directory)
    manifest = _load_manifest(root)
    if manifest.get("manifest_digest") != expected_manifest_digest:
        raise DataValidationError("contamination manifest differs from the expected digest")
    if manifest.get("schema_version") != 1:
        raise DataValidationError("unsupported contamination bundle schema")
    if (
        manifest.get("protocol") != expected_protocol.to_dict()
        or manifest.get("protocol_digest") != expected_protocol.digest
    ):
        raise DataValidationError("contamination bundle protocol differs")
    model = manifest.get("model")
    if model != {
        "repository": expected_protocol.model_repository,
        "revision": expected_protocol.model_revision,
        "artifact_tree_sha256": expected_protocol.model_artifact_tree_sha256,
    }:
        raise DataValidationError("contamination bundle model identity differs")
    inputs = manifest.get("inputs")
    expected_inputs = {
        "triviaqa": {
            "filename": Path(triviaqa_source).name,
            "sha256": sha256_file(triviaqa_source),
            "question_count": sum(1 for _ in read_questions(triviaqa_source)),
        },
        "targets": [
            {
                "filename": Path(path).name,
                "sha256": sha256_file(path),
                "question_count": sum(1 for _ in read_questions(path)),
            }
            for path in target_sources
        ],
    }
    if inputs != expected_inputs:
        raise DataValidationError("contamination bundle input identities differ")
    artifacts = manifest.get("artifacts")
    expected_artifacts = {
        filename: _artifact_descriptor(root / filename)
        for filename in sorted(_BUNDLE_FILES - {"manifest.json"})
    }
    if artifacts != expected_artifacts:
        raise DataValidationError("contamination bundle artifact fingerprints differ")

    raw_source = tuple(read_questions(triviaqa_source))
    expected_curation = exclude_exact_duplicate_groups(raw_source)
    curated = tuple(read_questions(root / "curated-source.jsonl"))
    if curated != expected_curation.questions:
        raise DataValidationError("contamination curated source differs from deterministic replay")
    try:
        curation_payload = json.loads(
            (root / "duplicate-curation.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"cannot read duplicate curation evidence: {exc}") from exc
    if curation_payload != expected_curation.report.to_dict():
        raise DataValidationError("duplicate curation evidence differs from deterministic replay")
    targets = tuple(read_questions(root / "targets.jsonl"))
    if targets != _questions_from_paths(target_sources):
        raise DataValidationError("contamination target rows differ from live inputs")

    source_embeddings = cast(
        FloatArray,
        np.load(root / "source-embeddings.npy", allow_pickle=False),
    )
    target_embeddings = cast(
        FloatArray,
        np.load(root / "target-embeddings.npy", allow_pickle=False),
    )
    _validate_embeddings(
        source_embeddings,
        rows=len(curated),
        dimension=expected_protocol.embedding_dimension,
        context="source",
    )
    _validate_embeddings(
        target_embeddings,
        rows=len(targets),
        dimension=expected_protocol.embedding_dimension,
        context="target",
    )
    if replay_embeddings:
        encoder = _TransformerEncoder(expected_protocol, model_root, progress=progress)
        replayed_source = encoder.encode(curated, label="replayed TriviaQA source")
        replayed_target = encoder.encode(targets, label="replayed OOD targets")
        if not np.array_equal(source_embeddings, replayed_source) or not np.array_equal(
            target_embeddings, replayed_target
        ):
            raise DataValidationError("semantic embeddings differ from full deterministic replay")

    semantic, review = semantic_overlap_pairs(
        curated,
        targets,
        source_embeddings,
        target_embeddings,
        threshold=expected_protocol.semantic_similarity_threshold,
        review_top_k=expected_protocol.review_top_k,
        similarity_batch_size=expected_protocol.similarity_batch_size,
        progress=progress,
    )
    if _read_jsonl(root / "semantic-matches.jsonl") != tuple(pair.to_dict() for pair in semantic):
        raise DataValidationError("semantic match evidence differs from saved embeddings")
    if _read_jsonl(root / "manual-review-queue.jsonl") != tuple(pair.to_dict() for pair in review):
        raise DataValidationError("manual review queue differs from saved embeddings")

    lexical_rows = _read_jsonl(root / "lexical-matches.jsonl")
    _validate_lexical_rows(
        lexical_rows,
        curated,
        targets,
        threshold=expected_protocol.lexical_ngram_threshold,
    )
    lexical = find_overlaps(
        curated,
        targets,
        ngram_threshold=expected_protocol.lexical_ngram_threshold,
    )
    expected_lexical_rows = tuple(
        {
            "source_question_id": match.source_question_id,
            "target_question_id": match.target_question_id,
            "exact": match.exact,
            "ngram_similarity": match.ngram_similarity,
        }
        for match in sorted(
            lexical.matches,
            key=lambda value: (
                -value.ngram_similarity,
                value.target_question_id,
                value.source_question_id,
            ),
        )
    )
    if lexical_rows != expected_lexical_rows:
        raise DataValidationError("lexical match evidence differs from deterministic replay")
    lexical_source_ids = set(lexical.source_ids_to_remove)

    removed_ids = lexical_source_ids | {pair.source_question_id for pair in semantic}
    clean = tuple(read_questions(root / "clean-source.jsonl"))
    expected_clean = tuple(
        question for question in curated if question.question_id not in removed_ids
    )
    if clean != expected_clean:
        raise DataValidationError("clean contamination source differs from recorded matches")
    counts = manifest.get("counts")
    expected_counts = {
        "raw_source": len(raw_source),
        "curated_source": len(curated),
        "duplicate_rows_excluded": expected_curation.report.excluded_question_count,
        "target": len(targets),
        "lexical_matches": len(lexical_rows),
        "semantic_matches": len(semantic),
        "contaminated_source_rows": len(removed_ids),
        "clean_source": len(clean),
        "manual_review_queue": len(review),
    }
    if counts != expected_counts:
        raise DataValidationError("contamination bundle counts differ")
    if manifest.get("contaminated_source_ids") != sorted(removed_ids) or manifest.get(
        "contaminated_source_ids_sha256"
    ) != stable_hash(sorted(removed_ids)):
        raise DataValidationError("contamination removal IDs differ")
    manual_review = manifest.get("manual_review")
    if manual_review != {
        "required": True,
        "status": "pending",
        "selection": "global-highest-cosine-similarity",
        "selection_count": len(review),
        "selection_sha256": stable_hash([pair.to_dict() for pair in review]),
    }:
        raise DataValidationError("manual review declaration differs")
    return manifest


def stderr_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mfh.config import SemanticContaminationProtocol
from mfh.contracts import Question
from mfh.data.semantic_contamination import (
    _validate_lexical_rows,
    semantic_overlap_pairs,
    verify_semantic_model_directory,
)
from mfh.errors import DataValidationError
from mfh.provenance import sha256_path


def _question(identifier: str, text: str) -> Question:
    return Question(identifier, "toy", text, (identifier,))


def _protocol(tree_digest: str, required_files: tuple[str, ...]) -> SemanticContaminationProtocol:
    return SemanticContaminationProtocol(
        model_repository="sentence-transformers/toy",
        model_revision="1" * 40,
        model_artifact_tree_sha256=tree_digest,
        required_files=required_files,
        pooling="mean",
        normalize_embeddings=True,
        max_length=256,
        embedding_dimension=2,
        lexical_ngram_threshold=0.8,
        semantic_similarity_threshold=0.9,
        review_top_k=3,
        device="cpu",
        dtype="float32",
        encode_batch_size=2,
        similarity_batch_size=1,
        torch_num_threads=1,
    )


def _write_model_directory(root: Path) -> tuple[str, ...]:
    (root / "1_Pooling").mkdir(parents=True)
    (root / "1_Pooling/config.json").write_text(
        json.dumps(
            {
                "word_embedding_dimension": 2,
                "pooling_mode_cls_token": False,
                "pooling_mode_mean_tokens": True,
                "pooling_mode_max_tokens": False,
                "pooling_mode_mean_sqrt_len_tokens": False,
            }
        ),
        encoding="utf-8",
    )
    (root / "sentence_bert_config.json").write_text(
        json.dumps({"max_seq_length": 256}), encoding="utf-8"
    )
    (root / "modules.json").write_text(
        json.dumps(
            [
                {"type": "sentence_transformers.models.Transformer"},
                {"type": "sentence_transformers.models.Pooling"},
                {"type": "sentence_transformers.models.Normalize"},
            ]
        ),
        encoding="utf-8",
    )
    return (
        "1_Pooling/config.json",
        "modules.json",
        "sentence_bert_config.json",
    )


def test_semantic_pairs_and_review_queue_are_deterministic() -> None:
    sources = (_question("s1", "one"), _question("s2", "two"))
    targets = (_question("t1", "first"), _question("t2", "second"))
    source_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    target_embeddings = np.asarray([[0.96, 0.28], [0.0, 1.0]], dtype=np.float32)

    matches, review = semantic_overlap_pairs(
        sources,
        targets,
        source_embeddings,
        target_embeddings,
        threshold=0.9,
        review_top_k=3,
        similarity_batch_size=1,
    )

    assert [(value.source_question_id, value.target_question_id) for value in matches] == [
        ("s2", "t2"),
        ("s1", "t1"),
    ]
    assert len(review) == 3
    assert review[0].similarity == pytest.approx(1.0)
    assert review[0].automatic_semantic_match


def test_semantic_pairs_reject_non_normalized_embeddings() -> None:
    questions = (_question("s1", "one"),)
    with pytest.raises(DataValidationError, match="unit normalized"):
        semantic_overlap_pairs(
            questions,
            questions,
            np.asarray([[2.0, 0.0]], dtype=np.float32),
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            threshold=0.9,
            review_top_k=1,
            similarity_batch_size=1,
        )


def test_semantic_review_ties_use_question_ids() -> None:
    sources = (_question("b", "second"), _question("a", "first"))
    targets = (_question("t", "target"),)
    source_embeddings = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    target_embeddings = np.asarray([[1.0, 0.0]], dtype=np.float32)

    _, review = semantic_overlap_pairs(
        sources,
        targets,
        source_embeddings,
        target_embeddings,
        threshold=0.9,
        review_top_k=1,
        similarity_batch_size=1,
    )

    assert [row.source_question_id for row in review] == ["a"]


def test_semantic_model_directory_binds_inventory_and_pooling() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        files = _write_model_directory(root)
        protocol = _protocol(sha256_path(root), files)
        assert verify_semantic_model_directory(protocol, root) == root.resolve()

        (root / "unexpected.txt").write_text("extra", encoding="utf-8")
        with pytest.raises(DataValidationError, match="inventory differs"):
            verify_semantic_model_directory(protocol, root)


def test_semantic_model_directory_rejects_root_symlink_and_empty_directory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "model"
    files = _write_model_directory(root)
    protocol = _protocol(sha256_path(root), files)

    linked = tmp_path / "linked-model"
    linked.symlink_to(root, target_is_directory=True)
    with pytest.raises(DataValidationError, match="regular directory"):
        verify_semantic_model_directory(protocol, linked)

    (root / "undeclared-empty-directory").mkdir()
    with pytest.raises(DataValidationError, match="directory inventory differs"):
        verify_semantic_model_directory(protocol, root)


def test_saved_lexical_rows_require_live_ids_and_exact_scores() -> None:
    sources = (_question("source", "same question"),)
    targets = (_question("target", "same question"),)
    valid = (
        {
            "source_question_id": "source",
            "target_question_id": "target",
            "exact": True,
            "ngram_similarity": 1.0,
        },
    )
    assert _validate_lexical_rows(valid, sources, targets, threshold=0.8) == {"source"}

    forged = ({**valid[0], "source_question_id": "not-live"},)
    with pytest.raises(DataValidationError, match="unknown source ID"):
        _validate_lexical_rows(forged, sources, targets, threshold=0.8)

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mfh.contracts import ActivationSite, Outcome
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.experiments.activation_store import (
    ActivationCaptureRow,
    ActivationStoreSpec,
    append_activation_shard,
    create_activation_store,
    iter_activation_shards,
    verify_activation_store,
)
from mfh.provenance import stable_hash


def _spec() -> ActivationStoreSpec:
    return ActivationStoreSpec(
        plan_identity="1" * 64,
        model_repository="example/model",
        model_revision="2" * 40,
        quantization="one-bit",
        layers=(1, 3),
        sites=(ActivationSite.POST_MLP, ActivationSite.BLOCK_OUTPUT),
        hidden_width=4,
        expected_rows=4,
    )


def _row(index: int) -> ActivationCaptureRow:
    return ActivationCaptureRow(
        question_id=f"q-{index}",
        benchmark="triviaqa",
        partition="T-dev",
        prompt_id="P0-neutral",
        outcome=Outcome.CORRECT if index % 2 == 0 else Outcome.INCORRECT,
        semantic_group_id=f"group-{index}",
        rendered_prompt_sha256="3" * 64,
        prompt_token_ids_sha256="4" * 64,
        generation_record_sha256="5" * 64,
        maximum_token_probability=0.75,
        output_entropy=0.5,
    )


def test_activation_store_appends_and_replays_immutable_float16_shards() -> None:
    spec = _spec()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "activations"
        create_activation_store(root, spec)
        first = np.arange(2 * 2 * 2 * 4, dtype=np.float32).reshape(2, 2, 2, 4)
        second = first + 100
        partial = append_activation_shard(root, (_row(0), _row(1)), first, expected_spec=spec)
        assert partial.rows_completed == 2
        assert partial.shard_count == 1
        with pytest.raises(FrozenArtifactError, match="row count"):
            verify_activation_store(root, expected_spec=spec, require_complete=True)

        complete = append_activation_shard(root, (_row(2), _row(3)), second, expected_spec=spec)
        assert complete.rows_completed == 4
        assert complete.chain_head is not None
        assert verify_activation_store(
            root, expected_spec=spec, require_complete=True
        ).chain_head == complete.chain_head
        shards = tuple(iter_activation_shards(root, expected_spec=spec))
        assert len(shards) == 2
        assert [row.question_id for row in shards[0][0]] == ["q-0", "q-1"]
        assert shards[0][1].dtype == np.float16
        assert shards[0][1].flags.writeable is False
        assert np.array_equal(shards[0][1], first.astype(np.float16))


def test_activation_store_rejects_bad_geometry_types_and_tampering() -> None:
    with pytest.raises(DataValidationError, match="geometry"):
        ActivationStoreSpec(
            plan_identity="1" * 64,
            model_repository="example/model",
            model_revision="2" * 40,
            quantization="one-bit",
            layers=(True,),
            sites=(ActivationSite.POST_MLP,),
            hidden_width=4,
            expected_rows=1,
        )
    spec = _spec()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "activations"
        create_activation_store(root, spec)
        values = np.ones((2, 2, 2, 4), dtype=np.float32)
        append_activation_shard(root, (_row(0), _row(1)), values, expected_spec=spec)
        rows = root / "shards" / "shard-00000" / "rows.jsonl"
        rows.write_text(rows.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
        with pytest.raises(FrozenArtifactError):
            verify_activation_store(root, expected_spec=spec)


def test_activation_store_rejects_manifest_coercion_symlinks_and_float16_overflow() -> None:
    spec = _spec()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "activations"
        create_activation_store(root, spec)
        overflow = np.full((2, 2, 2, 4), np.finfo(np.float32).max, dtype=np.float32)
        with pytest.raises(DataValidationError, match="overflows"):
            append_activation_shard(root, (_row(0), _row(1)), overflow, expected_spec=spec)
        assert verify_activation_store(root, expected_spec=spec).rows_completed == 0

        values = np.ones((2, 2, 2, 4), dtype=np.float32)
        append_activation_shard(root, (_row(0), _row(1)), values, expected_spec=spec)
        manifest_path = root / "shards" / "shard-00000" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.pop("shard_digest")
        manifest["row_count"] = True
        manifest_path.write_text(
            json.dumps({**manifest, "shard_digest": stable_hash(manifest)}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(FrozenArtifactError, match="JSON types"):
            verify_activation_store(root, expected_spec=spec)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "activations"
        create_activation_store(root, spec)
        target = Path(directory) / "spec-copy.json"
        target.write_bytes((root / "spec.json").read_bytes())
        (root / "spec.json").unlink()
        (root / "spec.json").symlink_to(target)
        with pytest.raises(FrozenArtifactError, match="regular file"):
            verify_activation_store(root, expected_spec=spec)

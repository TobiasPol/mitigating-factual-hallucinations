"""Byte-verified, selection-recomputed multi-seed SAE stability bundles."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]
from torch import Tensor

from mfh.artifact_namespace import validate_active_study_artifact_paths
from mfh.contracts import Outcome, Question
from mfh.data.io import read_questions
from mfh.data.splits import semantic_group_ids
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.methods.features import ActivationFeatureSchema
from mfh.methods.probes import ProbeDataset
from mfh.methods.sparse import (
    ActivationCorpus,
    SAETrainingResult,
    SeedFeatureSelection,
    latent_factuality_direction,
    load_activation_corpus,
    load_sae,
    sae_checkpoint_fingerprint,
    save_sae,
)
from mfh.provenance import sha256_file, sha256_path, stable_hash

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _validate_seed_selections(values: Sequence[SeedFeatureSelection]) -> None:
    frozen = tuple(values)
    if len(frozen) < 2:
        raise DataValidationError("SAE stability requires at least two seed checkpoints")
    if len({item.seed for item in frozen}) != len(frozen):
        raise DataValidationError("SAE stability requires distinct training seeds")
    if len({item.checkpoint_fingerprint for item in frozen}) != len(frozen):
        raise DataValidationError("SAE stability requires distinct checkpoints")
    if len({len(item.selected_features) for item in frozen}) != 1:
        raise DataValidationError("SAE stability selections must use one feature count")


def _oriented_decoder_directions(
    training: SAETrainingResult,
    dataset: ProbeDataset,
    *,
    feature_count: int,
) -> tuple[SeedFeatureSelection, Tensor]:
    if dataset.feature_schema is None or not dataset.feature_schema.is_compatible_representation(
        training.training_schema
    ):
        raise DataValidationError("SAE stability selection and training schemas differ")
    latent = latent_factuality_direction(
        training.model,
        dataset,
        feature_count=feature_count,
    )
    indices = torch.tensor(latent.selected_features, dtype=torch.long)
    columns = training.model.decoder.weight.detach().cpu().float()[:, indices].T
    signs = torch.sign(latent.direction[indices]).unsqueeze(1)
    oriented = columns * signs
    norms = torch.linalg.vector_norm(oriented, dim=1, keepdim=True)
    if not torch.isfinite(oriented).all() or not torch.isfinite(norms).all() or (norms <= 0).any():
        raise DataValidationError("SAE stability selected decoder direction is invalid")
    selection = SeedFeatureSelection(
        seed=training.config.seed,
        checkpoint_fingerprint=sae_checkpoint_fingerprint(training),
        selected_features=latent.selected_features,
    )
    return selection, oriented / norms


def _aligned_stability(direction_sets: Sequence[Tensor]) -> float:
    """Return the worst pairwise optimal matching of oriented decoder directions."""

    frozen = tuple(direction_sets)
    if len(frozen) < 2 or len({int(values.shape[0]) for values in frozen}) != 1:
        raise DataValidationError("aligned SAE stability requires equal-size seed selections")
    pair_scores: list[float] = []
    for left_index, left in enumerate(frozen):
        for right in frozen[left_index + 1 :]:
            similarities = torch.clamp(left @ right.T, min=0.0, max=1.0)
            row_indices, column_indices = linear_sum_assignment(-similarities.numpy())
            matched = similarities[
                torch.tensor(row_indices, dtype=torch.long),
                torch.tensor(column_indices, dtype=torch.long),
            ]
            score = float(matched.mean())
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise DataValidationError("aligned SAE stability score is invalid")
            pair_scores.append(score)
    return min(pair_scores)


@dataclass(frozen=True, slots=True)
class SAEStabilityBundle:
    selections_by_model: Mapping[str, tuple[SeedFeatureSelection, ...]]
    stability_by_model: Mapping[str, float]
    promoted_method_artifacts: Mapping[str, str]
    selection_datasets_by_model: Mapping[str, ProbeDataset]
    selection_corpora_by_model: Mapping[str, ActivationCorpus]

    def __post_init__(self) -> None:
        selections = {
            str(model).strip(): tuple(values) for model, values in self.selections_by_model.items()
        }
        stability = {
            str(model).strip(): float(value) for model, value in self.stability_by_model.items()
        }
        promoted = {
            str(model).strip(): str(value)
            for model, value in self.promoted_method_artifacts.items()
        }
        datasets = dict(self.selection_datasets_by_model)
        corpora = dict(self.selection_corpora_by_model)
        if (
            not selections
            or set(selections) != set(promoted)
            or set(selections) != set(stability)
            or set(selections) != set(datasets)
            or set(selections) != set(corpora)
        ):
            raise DataValidationError("SAE stability models differ from promoted artifacts")
        if any(
            not model
            or not _SHA256.fullmatch(promoted[model])
            or not math.isfinite(stability[model])
            or not 0 <= stability[model] <= 1
            for model in selections
        ):
            raise DataValidationError("SAE stability model evidence is invalid")
        for values in selections.values():
            _validate_seed_selections(values)
        for model, dataset in datasets.items():
            schema = dataset.feature_schema
            if (
                type(dataset) is not ProbeDataset
                or schema is None
                or schema.partition != "T-steer"
                or schema.model_repository != model
            ):
                raise DataValidationError(
                    "SAE stability selection schema differs from its model label"
                )
            corpus = corpora[model]
            if (
                corpus.schema_version != 2
                or corpus.feature_schema != schema
                or corpus.runtime_artifact_sha256 is None
                or corpus.execution_public_key is None
                or corpus.source_question_bundle_sha256 is None
            ):
                raise DataValidationError(
                    "SAE stability selection lacks native signed corpus provenance"
                )
        object.__setattr__(self, "selections_by_model", MappingProxyType(selections))
        object.__setattr__(self, "stability_by_model", MappingProxyType(stability))
        object.__setattr__(self, "promoted_method_artifacts", MappingProxyType(promoted))
        object.__setattr__(
            self, "selection_datasets_by_model", MappingProxyType(datasets)
        )
        object.__setattr__(
            self, "selection_corpora_by_model", MappingProxyType(corpora)
        )


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FrozenArtifactError(f"{label} must be a mapping")
    return payload


def _read_manifest(source: Path) -> dict[str, Any]:
    payload = _read_json(source / "manifest.json", label="SAE stability manifest")
    digest = payload.pop("manifest_digest", None)
    if digest != stable_hash(payload):
        raise FrozenArtifactError("SAE stability manifest digest mismatch")
    if set(payload) != {"schema_version", "models"} or payload.get("schema_version") != 3:
        raise FrozenArtifactError("SAE stability manifest has an invalid schema")
    return payload


def _write_selection_dataset(destination: Path, dataset: ProbeDataset) -> None:
    schema = dataset.feature_schema
    if schema is None or schema.partition != "T-steer":
        raise DataValidationError("SAE stability requires a bound T-steer selection dataset")
    destination.mkdir(parents=True)
    tensor_path = destination / "features.safetensors"
    save_file({"features": dataset.features}, tensor_path)
    body = {
        "schema_version": 1,
        "question_ids": list(dataset.question_ids),
        "group_ids": list(dataset.group_ids),
        "outcomes": [value.value for value in dataset.outcomes],
        "feature_schema": schema.to_dict(),
        "data_fingerprint": dataset.data_fingerprint,
        "tensor_sha256": sha256_file(tensor_path),
    }
    (destination / "metadata.json").write_text(
        json.dumps({**body, "metadata_digest": stable_hash(body)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_selection_dataset(source: Path) -> ProbeDataset:
    if (
        source.is_symlink()
        or not source.is_dir()
        or {item.name for item in source.iterdir()} != {"metadata.json", "features.safetensors"}
    ):
        raise FrozenArtifactError("SAE stability selection dataset has invalid files")
    metadata = _read_json(source / "metadata.json", label="SAE selection metadata")
    digest = metadata.pop("metadata_digest", None)
    expected = {
        "schema_version",
        "question_ids",
        "group_ids",
        "outcomes",
        "feature_schema",
        "data_fingerprint",
        "tensor_sha256",
    }
    if (
        digest != stable_hash(metadata)
        or set(metadata) != expected
        or metadata["schema_version"] != 1
    ):
        raise FrozenArtifactError("SAE selection metadata has an invalid schema or digest")
    tensor_path = source / "features.safetensors"
    if tensor_path.is_symlink() or sha256_file(tensor_path) != metadata["tensor_sha256"]:
        raise FrozenArtifactError("SAE selection tensor checksum mismatch")
    try:
        tensors = load_file(tensor_path, device="cpu")
        if set(tensors) != {"features"}:
            raise FrozenArtifactError("SAE selection artifact has unexpected tensors")
        schema_value = metadata["feature_schema"]
        if not isinstance(schema_value, dict):
            raise FrozenArtifactError("SAE selection feature schema is invalid")
        dataset = ProbeDataset(
            question_ids=tuple(str(value) for value in metadata["question_ids"]),
            features=tensors["features"],
            outcomes=tuple(Outcome(value) for value in metadata["outcomes"]),
            group_ids=tuple(str(value) for value in metadata["group_ids"]),
            feature_schema=ActivationFeatureSchema.from_dict(schema_value),
            data_fingerprint=str(metadata["data_fingerprint"]),
        )
    except (KeyError, TypeError, ValueError, DataValidationError) as exc:
        raise FrozenArtifactError(f"SAE selection dataset is invalid: {exc}") from exc
    if dataset.feature_schema is None or dataset.feature_schema.partition != "T-steer":
        raise FrozenArtifactError("SAE selection dataset is not the frozen T-steer partition")
    return dataset


def _probe_from_signed_corpus(corpus: ActivationCorpus) -> ProbeDataset:
    if (
        corpus.schema_version != 2
        or corpus.feature_schema.partition != "T-steer"
        or corpus.runtime_artifact_sha256 is None
        or corpus.execution_public_key is None
        or corpus.source_question_bundle_sha256 is None
    ):
        raise FrozenArtifactError(
            "SAE selection must be a native-signed T-steer activation corpus"
        )
    identifiers: list[str] = []
    groups: list[str] = []
    outcomes: list[Outcome] = []
    values: list[Tensor] = []
    for shard in corpus.iter_shards():
        identifiers.extend(shard.question_ids)
        groups.extend(shard.group_ids)
        outcomes.extend(shard.outcomes)
        values.append(torch.from_numpy(shard.activations.copy()).float())
    return ProbeDataset(
        question_ids=tuple(identifiers),
        features=torch.cat(values, dim=0),
        outcomes=tuple(outcomes),
        group_ids=tuple(groups),
        feature_schema=corpus.feature_schema,
    )


def _question_fingerprint(question: Question) -> str:
    return stable_hash(
        {
            "question_id": question.question_id,
            "benchmark": question.benchmark,
            "text": question.text,
            "aliases": list(question.aliases),
            "split": question.split,
            "entities": list(question.entities),
            "metadata": dict(question.metadata),
        }
    )


def _verify_selection_questions(corpus: ActivationCorpus, source: Path) -> None:
    if (
        source.is_symlink()
        or not source.is_file()
        or sha256_path(source) != corpus.source_question_bundle_sha256
    ):
        raise FrozenArtifactError("SAE T-steer question source differs from native capture")
    questions = tuple(read_questions(source))
    question_index = {value.question_id: value for value in questions}
    if (
        len(question_index) != len(questions)
        or len(questions) != corpus.total_rows
        or any(value.split != "T-steer" for value in questions)
    ):
        raise FrozenArtifactError("SAE T-steer question source is incomplete")
    expected_groups = semantic_group_ids(questions)
    observed_ids: set[str] = set()
    for shard in corpus.shards:
        records = json.loads(
            (corpus.directory / str(shard["records"])).read_text(encoding="utf-8")
        )
        for record in records:
            question_id = str(record["question_id"])
            question = question_index.get(question_id)
            receipt = record.get("capture_receipt")
            body = receipt.get("body") if isinstance(receipt, Mapping) else None
            label = body.get("label_evidence") if isinstance(body, Mapping) else None
            if (
                question is None
                or question_id in observed_ids
                or record.get("group_id") != expected_groups[question_id]
                or not isinstance(label, Mapping)
                or label.get("aliases") != list(question.aliases)
                or label.get("source_question_sha256") != _question_fingerprint(question)
            ):
                raise FrozenArtifactError(
                    "SAE T-steer native label differs from its frozen question"
                )
            observed_ids.add(question_id)
    if observed_ids != set(question_index):
        raise FrozenArtifactError("SAE T-steer capture omits frozen questions")


def load_sae_stability_bundle(path: str | Path) -> SAEStabilityBundle:
    """Load checkpoints, reselect features, and recompute aligned stability."""

    source = Path(path)
    if (
        source.is_symlink()
        or not source.is_dir()
        or {item.name for item in source.iterdir()} != {"manifest.json", "runs", "selections"}
    ):
        raise FrozenArtifactError("SAE stability bundle has invalid top-level files")
    payload = _read_manifest(source)
    models = payload["models"]
    if not isinstance(models, list) or not models:
        raise FrozenArtifactError("SAE stability bundle has no model descriptors")
    selections_by_model: dict[str, tuple[SeedFeatureSelection, ...]] = {}
    stability_by_model: dict[str, float] = {}
    promoted: dict[str, str] = {}
    selection_datasets: dict[str, ProbeDataset] = {}
    selection_corpora: dict[str, ActivationCorpus] = {}
    expected_run_directories: set[str] = set()
    expected_selection_directories: set[str] = set()
    for descriptor in models:
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
            "model_repository",
            "promoted_method_artifact_sha256",
            "selection",
            "runs",
        }:
            raise FrozenArtifactError("SAE stability model descriptor is invalid")
        model = descriptor["model_repository"]
        promoted_sha = descriptor["promoted_method_artifact_sha256"]
        selection_descriptor = descriptor["selection"]
        runs = descriptor["runs"]
        if (
            not isinstance(model, str)
            or model != model.strip()
            or not model
            or model in selections_by_model
            or not isinstance(promoted_sha, str)
            or not _SHA256.fullmatch(promoted_sha)
            or not isinstance(selection_descriptor, Mapping)
            or set(selection_descriptor)
            != {
                "path",
                "artifact_sha256",
                "corpus_data_fingerprint",
                "probe_data_fingerprint",
                "runtime_artifact_sha256",
                "execution_public_key",
                "source_question_bundle_sha256",
                "questions_sha256",
                "feature_count",
            }
            or not isinstance(runs, list)
            or len(runs) < 2
        ):
            raise FrozenArtifactError("SAE stability model fields are invalid")
        selection_name = stable_hash({"model_repository": model})[:16]
        selection_relative = f"selections/{selection_name}"
        feature_count = selection_descriptor["feature_count"]
        if (
            selection_descriptor["path"] != selection_relative
            or not isinstance(selection_descriptor["artifact_sha256"], str)
            or not _SHA256.fullmatch(selection_descriptor["artifact_sha256"])
            or any(
                not isinstance(selection_descriptor[name], str)
                or not _SHA256.fullmatch(selection_descriptor[name])
                for name in (
                    "corpus_data_fingerprint",
                    "probe_data_fingerprint",
                    "runtime_artifact_sha256",
                    "execution_public_key",
                    "source_question_bundle_sha256",
                    "questions_sha256",
                )
            )
            or isinstance(feature_count, bool)
            or not isinstance(feature_count, int)
            or feature_count <= 0
        ):
            raise FrozenArtifactError("SAE stability selection identity is invalid")
        selection_path = source / selection_relative
        if sha256_path(selection_path) != selection_descriptor["artifact_sha256"]:
            raise FrozenArtifactError("SAE stability selection bytes changed")
        if (
            selection_path.is_symlink()
            or not selection_path.is_dir()
            or {value.name for value in selection_path.iterdir()}
            != {"corpus", "questions.jsonl"}
        ):
            raise FrozenArtifactError("SAE stability selection source inventory differs")
        selection_corpus = load_activation_corpus(selection_path / "corpus")
        _verify_selection_questions(selection_corpus, selection_path / "questions.jsonl")
        selection_dataset = _probe_from_signed_corpus(selection_corpus)
        if (
            selection_corpus.data_fingerprint
            != selection_descriptor["corpus_data_fingerprint"]
            or selection_dataset.data_fingerprint
            != selection_descriptor["probe_data_fingerprint"]
            or selection_corpus.runtime_artifact_sha256
            != selection_descriptor["runtime_artifact_sha256"]
            or selection_corpus.execution_public_key
            != selection_descriptor["execution_public_key"]
            or selection_corpus.source_question_bundle_sha256
            != selection_descriptor["source_question_bundle_sha256"]
            or sha256_file(selection_path / "questions.jsonl")
            != selection_descriptor["questions_sha256"]
            or selection_dataset.feature_schema is None
            or selection_dataset.feature_schema.model_repository != model
        ):
            raise FrozenArtifactError("SAE stability selection fingerprint is false")
        selection_datasets[model] = selection_dataset
        selection_corpora[model] = selection_corpus
        expected_selection_directories.add(selection_name)

        parsed: list[SeedFeatureSelection] = []
        oriented_directions: list[Tensor] = []
        for run in runs:
            if not isinstance(run, Mapping) or set(run) != {
                "seed",
                "checkpoint_path",
                "checkpoint_artifact_sha256",
                "checkpoint_fingerprint",
                "selected_features",
            }:
                raise FrozenArtifactError("SAE stability run descriptor is invalid")
            seed = run["seed"]
            expected_run_name = stable_hash({"model_repository": model, "seed": seed})[:16]
            relative = f"runs/{expected_run_name}"
            if (
                isinstance(seed, bool)
                or not isinstance(seed, int)
                or seed < 0
                or run["checkpoint_path"] != relative
                or not isinstance(run["checkpoint_artifact_sha256"], str)
                or not _SHA256.fullmatch(run["checkpoint_artifact_sha256"])
                or not isinstance(run["checkpoint_fingerprint"], str)
                or not _SHA256.fullmatch(run["checkpoint_fingerprint"])
                or not isinstance(run["selected_features"], list)
            ):
                raise FrozenArtifactError("SAE stability run identity is invalid")
            checkpoint = source / relative
            if (
                checkpoint.is_symlink()
                or sha256_path(checkpoint) != run["checkpoint_artifact_sha256"]
            ):
                raise FrozenArtifactError("SAE stability checkpoint bytes changed")
            training = load_sae(checkpoint)
            if (
                training.training_schema.model_repository != model
                or training.validation_schema.model_repository != model
                or not training.training_schema.is_compatible_representation(
                    selection_dataset.feature_schema
                )
            ):
                raise FrozenArtifactError(
                    "SAE stability checkpoint schema differs from its model label"
                )
            try:
                recomputed, oriented = _oriented_decoder_directions(
                    training,
                    selection_dataset,
                    feature_count=feature_count,
                )
            except DataValidationError as exc:
                raise FrozenArtifactError(
                    f"SAE stability checkpoint is incompatible: {exc}"
                ) from exc
            if (
                training.config.seed != seed
                or recomputed.checkpoint_fingerprint != run["checkpoint_fingerprint"]
                or list(recomputed.selected_features) != run["selected_features"]
            ):
                raise FrozenArtifactError("SAE stability checkpoint or selection claim is false")
            parsed.append(recomputed)
            oriented_directions.append(oriented)
            expected_run_directories.add(expected_run_name)
        try:
            _validate_seed_selections(parsed)
            stability = _aligned_stability(oriented_directions)
        except DataValidationError as exc:
            raise FrozenArtifactError(f"SAE stability evidence is invalid: {exc}") from exc
        selections_by_model[model] = tuple(parsed)
        stability_by_model[model] = stability
        promoted[model] = promoted_sha

    runs_root = source / "runs"
    selections_root = source / "selections"
    if (
        runs_root.is_symlink()
        or not runs_root.is_dir()
        or {item.name for item in runs_root.iterdir()} != expected_run_directories
        or selections_root.is_symlink()
        or not selections_root.is_dir()
        or {item.name for item in selections_root.iterdir()} != expected_selection_directories
    ):
        raise FrozenArtifactError("SAE stability bundle contains undeclared artifacts")
    return SAEStabilityBundle(
        selections_by_model,
        stability_by_model,
        promoted,
        selection_datasets,
        selection_corpora,
    )


def write_sae_stability_bundle(
    destination: str | Path,
    *,
    runs_by_model: Mapping[str, Sequence[SAETrainingResult]],
    selection_corpora: Mapping[str, str | Path | ActivationCorpus],
    selection_question_sources: Mapping[str, str | Path],
    feature_count: int,
    promoted_method_artifacts: Mapping[str, str],
) -> str:
    """Package seed checkpoints; feature selection is always recomputed from T-steer."""

    destination = validate_active_study_artifact_paths(
        {"E7 SAE stability bundle": destination}
    )["E7 SAE stability bundle"]
    if (
        not runs_by_model
        or set(runs_by_model) != set(promoted_method_artifacts)
        or set(runs_by_model) != set(selection_corpora)
        or set(runs_by_model) != set(selection_question_sources)
        or isinstance(feature_count, bool)
        or not isinstance(feature_count, int)
        or feature_count <= 0
    ):
        raise DataValidationError("SAE stability bundle inputs differ or are invalid")
    target = Path(destination)
    if target.exists():
        raise FrozenArtifactError(f"refusing to overwrite SAE stability bundle: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.stage-", dir=target.parent))
    try:
        (stage / "runs").mkdir()
        (stage / "selections").mkdir()
        models: list[dict[str, Any]] = []
        for model in sorted(runs_by_model):
            promoted_sha = str(promoted_method_artifacts[model])
            if model != model.strip() or not model or not _SHA256.fullmatch(promoted_sha):
                raise DataValidationError("SAE stability model identity is invalid")
            training_runs = tuple(runs_by_model[model])
            if len(training_runs) < 2:
                raise DataValidationError("SAE stability requires at least two seed runs")
            raw_corpus = selection_corpora[model]
            selection_corpus = (
                raw_corpus
                if isinstance(raw_corpus, ActivationCorpus)
                else load_activation_corpus(raw_corpus)
            )
            selection_dataset = _probe_from_signed_corpus(selection_corpus)
            if (
                selection_dataset.feature_schema is None
                or selection_dataset.feature_schema.model_repository != model
                or any(
                    training.training_schema.model_repository != model
                    or training.validation_schema.model_repository != model
                    or not training.training_schema.is_compatible_representation(
                        selection_dataset.feature_schema
                    )
                    for training in training_runs
                )
            ):
                raise DataValidationError(
                    "SAE stability model label differs from checkpoint schemas"
                )
            selection_name = stable_hash({"model_repository": model})[:16]
            selection_path = stage / "selections" / selection_name
            selection_path.mkdir()
            shutil.copytree(selection_corpus.directory, selection_path / "corpus")
            question_source = Path(selection_question_sources[model]).resolve()
            if question_source.is_symlink() or not question_source.is_file():
                raise FrozenArtifactError("SAE T-steer questions must be a regular JSONL file")
            shutil.copy2(question_source, selection_path / "questions.jsonl")
            reloaded_corpus = load_activation_corpus(selection_path / "corpus")
            _verify_selection_questions(
                reloaded_corpus, selection_path / "questions.jsonl"
            )
            reloaded_dataset = _probe_from_signed_corpus(reloaded_corpus)
            if reloaded_dataset.data_fingerprint != selection_dataset.data_fingerprint:
                raise FrozenArtifactError("copied SAE selection corpus changed")
            runs: list[dict[str, Any]] = []
            selections: list[SeedFeatureSelection] = []
            for training in training_runs:
                selection, _ = _oriented_decoder_directions(
                    training,
                    selection_dataset,
                    feature_count=feature_count,
                )
                run_name = stable_hash({"model_repository": model, "seed": selection.seed})[:16]
                checkpoint = stage / "runs" / run_name
                save_sae(checkpoint, training)
                runs.append(
                    {
                        "seed": selection.seed,
                        "checkpoint_path": f"runs/{run_name}",
                        "checkpoint_artifact_sha256": sha256_path(checkpoint),
                        "checkpoint_fingerprint": selection.checkpoint_fingerprint,
                        "selected_features": list(selection.selected_features),
                    }
                )
                selections.append(selection)
            _validate_seed_selections(selections)
            models.append(
                {
                    "model_repository": model,
                    "promoted_method_artifact_sha256": promoted_sha,
                    "selection": {
                        "path": f"selections/{selection_name}",
                        "artifact_sha256": sha256_path(selection_path),
                        "corpus_data_fingerprint": selection_corpus.data_fingerprint,
                        "probe_data_fingerprint": selection_dataset.data_fingerprint,
                        "runtime_artifact_sha256": (
                            selection_corpus.runtime_artifact_sha256
                        ),
                        "execution_public_key": selection_corpus.execution_public_key,
                        "source_question_bundle_sha256": (
                            selection_corpus.source_question_bundle_sha256
                        ),
                        "questions_sha256": sha256_file(
                            selection_path / "questions.jsonl"
                        ),
                        "feature_count": feature_count,
                    },
                    "runs": runs,
                }
            )
        body = {"schema_version": 3, "models": models}
        (stage / "manifest.json").write_text(
            json.dumps({**body, "manifest_digest": stable_hash(body)}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        load_sae_stability_bundle(stage)
        os.replace(stage, target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return sha256_path(target)

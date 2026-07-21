"""Replayable composed E2 feature matrices used by E5 adaptive controllers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import torch

from mfh.contracts import ActivationSite, Runtime, TokenScope
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.experiments.activation_store import (
    ActivationCaptureRow,
    VerifiedActivationStore,
    iter_activation_shards,
)
from mfh.experiments.e2_schedule import VerifiedE2Workspace
from mfh.methods.features import (
    ActivationFeatureSchema,
    ActivationKind,
    FeatureComposition,
)
from mfh.methods.probes import ProbeDataset

_PARTITIONS = (
    "T-controller-train",
    "T-controller-calibration",
    "T-dev",
    "simpleqa-eval",
    "aa-eval",
)


@dataclass(frozen=True, slots=True)
class E2ControllerInputView:
    """One preregistered E5 controller-input representation."""

    composition: FeatureComposition
    layers: tuple[int, ...]
    site: ActivationSite

    def __post_init__(self) -> None:
        if (
            not isinstance(self.composition, FeatureComposition)
            or type(self.layers) is not tuple
            or not self.layers
            or len(set(self.layers)) != len(self.layers)
            or any(type(value) is not int or value < 0 for value in self.layers)
            or not isinstance(self.site, ActivationSite)
            or (
                self.composition is FeatureComposition.SINGLE_LAYER
                and len(self.layers) != 1
            )
            or (
                self.composition is not FeatureComposition.SINGLE_LAYER
                and len(self.layers) < 2
            )
        ):
            raise DataValidationError("E2 controller-input view is invalid")

    @property
    def identifier(self) -> str:
        return self.composition.value

    def to_dict(self) -> dict[str, object]:
        return {
            "composition": self.composition.value,
            "layers": list(self.layers),
            "site": self.site.value,
        }


@dataclass(frozen=True, slots=True)
class E2ControllerInputDataset:
    probe: ProbeDataset
    rows: tuple[ActivationCaptureRow, ...]

    def __post_init__(self) -> None:
        if len(self.rows) != len(self.probe.question_ids):
            raise DataValidationError("E2 composed feature rows differ from their probe rows")


def controller_input_views(
    *,
    selected_layer: int,
    selected_site: ActivationSite,
    candidate_layers: tuple[int, ...],
) -> tuple[E2ControllerInputView, ...]:
    """Freeze one selected layer and its two nearest registered layer neighbours."""

    if (
        type(selected_layer) is not int
        or selected_layer not in candidate_layers
        or not isinstance(selected_site, ActivationSite)
        or type(candidate_layers) is not tuple
        or len(set(candidate_layers)) != len(candidate_layers)
        or any(type(value) is not int or value < 0 for value in candidate_layers)
    ):
        raise DataValidationError("E2 controller-input geometry is invalid")
    single = E2ControllerInputView(
        FeatureComposition.SINGLE_LAYER,
        (selected_layer,),
        selected_site,
    )
    if len(candidate_layers) < 3:
        return (single,)
    nearest = sorted(
        (value for value in candidate_layers if value != selected_layer),
        key=lambda value: (abs(value - selected_layer), value),
    )[:2]
    composed_layers = tuple(sorted((selected_layer, *nearest)))
    return (
        single,
        E2ControllerInputView(
            FeatureComposition.CONCATENATED_LAYERS,
            composed_layers,
            selected_site,
        ),
        E2ControllerInputView(
            FeatureComposition.LAYER_DIFFERENCES,
            composed_layers,
            selected_site,
        ),
    )


def _compose(values: np.ndarray, composition: FeatureComposition) -> np.ndarray:
    if values.ndim != 3 or values.shape[0] == 0 or values.shape[2] == 0:
        raise DataValidationError("E2 controller-input activation cube is invalid")
    if composition is FeatureComposition.SINGLE_LAYER:
        result = values[:, 0, :]
    elif composition is FeatureComposition.CONCATENATED_LAYERS:
        result = values.reshape(values.shape[0], -1)
    else:
        result = np.diff(values, axis=1).reshape(values.shape[0], -1)
    output = np.ascontiguousarray(result, dtype=np.float32)
    if output.ndim != 2 or not np.isfinite(output).all():
        raise DataValidationError("E2 controller-input matrix is invalid")
    return output


def build_e2_controller_input_datasets(
    workspace: VerifiedE2Workspace,
    *,
    views: tuple[E2ControllerInputView, ...],
    split_manifest_digest: str,
    prompt_template_sha256: str,
    verified_store: VerifiedActivationStore,
) -> Mapping[tuple[FeatureComposition, str], E2ControllerInputDataset]:
    """Extract all P0 controller representations in one activation-store traversal."""

    if (
        type(views) is not tuple
        or not views
        or len({value.composition for value in views}) != len(views)
        or any(value.site not in workspace.activation_spec.sites for value in views)
        or any(
            layer not in workspace.activation_spec.layers
            for value in views
            for layer in value.layers
        )
    ):
        raise DataValidationError("E2 controller-input views differ from the activation store")
    sites = {value.site for value in views}
    if len(sites) != 1:
        raise DataValidationError("E2 controller inputs must share one selected activation site")
    site = next(iter(sites))
    layer_axis = tuple(sorted({layer for value in views for layer in value.layers}))
    layer_indices = tuple(workspace.activation_spec.layers.index(value) for value in layer_axis)
    site_index = workspace.activation_spec.sites.index(site)
    rows_by_partition: dict[str, list[ActivationCaptureRow]] = {
        value: [] for value in _PARTITIONS
    }
    cubes_by_partition: dict[str, list[np.ndarray]] = {
        value: [] for value in _PARTITIONS
    }
    sequence = 0
    for rows, activations in iter_activation_shards(
        workspace.directory / "activations",
        expected_spec=workspace.activation_spec,
        verified_store=verified_store,
    ):
        for offset, row in enumerate(rows):
            schedule = workspace.schedule[sequence]
            if (
                row.question_id != schedule.question_id
                or row.benchmark != schedule.benchmark
                or row.partition != schedule.feature_partition
                or row.prompt_id != schedule.prompt_id
                or row.semantic_group_id != schedule.semantic_group_id
                or (schedule.outcome is not None and row.outcome is not schedule.outcome)
            ):
                raise FrozenArtifactError(
                    "E2 controller-input capture differs from its frozen schedule"
                )
            if row.prompt_id == "P0-neutral":
                if row.partition not in rows_by_partition:
                    raise FrozenArtifactError(
                        "E2 controller-input capture has an unexpected P0 partition"
                    )
                rows_by_partition[row.partition].append(row)
                cubes_by_partition[row.partition].append(
                    np.asarray(
                        activations[offset, site_index, layer_indices, :],
                        dtype=np.float32,
                    ).copy()
                )
            sequence += 1
    if sequence != len(workspace.schedule) or any(
        not rows_by_partition[value] for value in _PARTITIONS
    ):
        raise FrozenArtifactError("E2 controller-input extraction is incomplete")
    layer_position = {layer: index for index, layer in enumerate(layer_axis)}
    result: dict[tuple[FeatureComposition, str], E2ControllerInputDataset] = {}
    for partition in _PARTITIONS:
        rows = tuple(rows_by_partition[partition])
        benchmarks = {row.benchmark for row in rows}
        if len(benchmarks) != 1:
            raise DataValidationError("E2 controller-input partition mixes benchmarks")
        cube = np.stack(cubes_by_partition[partition], axis=0)
        for view in views:
            selected = cube[:, [layer_position[layer] for layer in view.layers], :]
            features = _compose(selected, view.composition)
            schema = ActivationFeatureSchema(
                benchmark=next(iter(benchmarks)),
                partition=partition,
                split_manifest_digest=split_manifest_digest,
                model_repository=workspace.activation_spec.model_repository,
                model_revision=workspace.activation_spec.model_revision,
                runtime=Runtime.MLX,
                quantization=workspace.activation_spec.quantization,
                prompt_id="P0-neutral",
                prompt_sha256=prompt_template_sha256,
                activation_kind=ActivationKind.FINAL_PROMPT,
                layers=view.layers,
                sites=(view.site,),
                composition=view.composition,
                width=features.shape[1],
                token_scope=TokenScope.FINAL_PROMPT,
            )
            result[(view.composition, partition)] = E2ControllerInputDataset(
                probe=ProbeDataset(
                    question_ids=tuple(row.question_id for row in rows),
                    features=torch.from_numpy(features),
                    outcomes=tuple(row.outcome for row in rows),
                    group_ids=tuple(row.semantic_group_id for row in rows),
                    feature_schema=schema,
                ),
                rows=rows,
            )
    expected = {
        (view.composition, partition)
        for view in views
        for partition in _PARTITIONS
    }
    if set(result) != expected:
        raise FrozenArtifactError("E2 controller-input dataset grid differs")
    return MappingProxyType(result)

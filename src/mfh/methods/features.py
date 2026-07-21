"""Provenance contract for every activation feature matrix.

Feature tensors are not interchangeable merely because their widths match.
This schema binds them to the exact model, prompt, split plan, hook sites,
token positions, and feature composition that produced them.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from mfh.contracts import ActivationSite, Runtime, TokenScope
from mfh.errors import DataValidationError
from mfh.provenance import stable_hash

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40,64}$")


class FeatureComposition(StrEnum):
    SINGLE_LAYER = "single_layer"
    CONCATENATED_LAYERS = "concatenated_layers"
    LAYER_DIFFERENCES = "layer_differences"


class ActivationKind(StrEnum):
    FINAL_PROMPT = "final_prompt"
    RESPONSE_TOKENS = "response_tokens"
    FIRST_GENERATED = "first_generated"
    FIRST_FOUR_GENERATED = "first_four_generated"
    FIRST_EIGHT_GENERATED = "first_eight_generated"


@dataclass(frozen=True, slots=True)
class ActivationFeatureSchema:
    benchmark: str
    partition: str
    split_manifest_digest: str
    model_repository: str
    model_revision: str
    runtime: Runtime
    quantization: str
    prompt_id: str
    prompt_sha256: str
    activation_kind: ActivationKind
    layers: tuple[int, ...]
    sites: tuple[ActivationSite, ...]
    composition: FeatureComposition
    width: int
    token_scope: TokenScope | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise DataValidationError("unsupported activation-feature schema version")
        for name in (
            "benchmark",
            "partition",
            "model_repository",
            "quantization",
            "prompt_id",
        ):
            value = getattr(self, name)
            if type(value) is not str or not value.strip() or value != value.strip():
                raise DataValidationError(f"activation feature {name} must be non-empty")
        if type(self.split_manifest_digest) is not str or not _SHA256.fullmatch(
            self.split_manifest_digest
        ):
            raise DataValidationError("feature schema requires a split-manifest SHA-256")
        if type(self.model_revision) is not str or not _REVISION.fullmatch(
            self.model_revision
        ):
            raise DataValidationError("feature schema requires an immutable model revision")
        if type(self.prompt_sha256) is not str or not _SHA256.fullmatch(
            self.prompt_sha256
        ):
            raise DataValidationError("feature schema requires the rendered prompt-template hash")
        if not isinstance(self.runtime, Runtime):
            raise DataValidationError("feature schema runtime must be an exact Runtime")
        if not isinstance(self.activation_kind, ActivationKind):
            raise DataValidationError("feature schema activation kind is invalid")
        if not isinstance(self.composition, FeatureComposition):
            raise DataValidationError("feature schema composition is invalid")
        if type(self.layers) is not tuple or any(
            type(value) is not int for value in self.layers
        ):
            raise DataValidationError("feature schema layers must be exact integers")
        if type(self.sites) is not tuple or any(
            not isinstance(value, ActivationSite) for value in self.sites
        ):
            raise DataValidationError("feature schema sites must be exact activation sites")
        if self.token_scope is not None and not isinstance(self.token_scope, TokenScope):
            raise DataValidationError("feature schema token scope is invalid")
        layers = self.layers
        sites = self.sites
        if not layers or len(set(layers)) != len(layers) or any(value < 0 for value in layers):
            raise DataValidationError("feature schema layers must be unique and non-negative")
        if not sites or len(set(sites)) != len(sites):
            raise DataValidationError("feature schema sites must be non-empty and unique")
        if type(self.width) is not int or self.width <= 0:
            raise DataValidationError("feature schema width must be positive")
        if self.composition is FeatureComposition.SINGLE_LAYER and len(layers) != 1:
            raise DataValidationError("single-layer features must name exactly one layer")
        if self.composition is not FeatureComposition.SINGLE_LAYER and len(layers) < 2:
            raise DataValidationError("composed features must name at least two layers")
        if self.activation_kind is ActivationKind.FINAL_PROMPT:
            if self.token_scope not in {None, TokenScope.FINAL_PROMPT}:
                raise DataValidationError("final-prompt features have an incompatible token scope")
        elif self.token_scope is None:
            raise DataValidationError("output-token features must declare their token scope")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["runtime"] = self.runtime.value
        value["activation_kind"] = self.activation_kind.value
        value["sites"] = [site.value for site in self.sites]
        value["composition"] = self.composition.value
        value["token_scope"] = self.token_scope.value if self.token_scope is not None else None
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ActivationFeatureSchema:
        expected = {
            "benchmark",
            "partition",
            "split_manifest_digest",
            "model_repository",
            "model_revision",
            "runtime",
            "quantization",
            "prompt_id",
            "prompt_sha256",
            "activation_kind",
            "layers",
            "sites",
            "composition",
            "width",
            "token_scope",
            "schema_version",
        }
        if type(value) is not dict or set(value) != expected:
            raise DataValidationError("activation-feature schema keys differ from version 1")
        string_fields = (
            "benchmark",
            "partition",
            "split_manifest_digest",
            "model_repository",
            "model_revision",
            "runtime",
            "quantization",
            "prompt_id",
            "prompt_sha256",
            "activation_kind",
            "composition",
        )
        if (
            any(type(value[name]) is not str for name in string_fields)
            or type(value["schema_version"]) is not int
            or type(value["width"]) is not int
            or type(value["layers"]) is not list
            or any(type(item) is not int for item in value["layers"])
            or type(value["sites"]) is not list
            or any(type(item) is not str for item in value["sites"])
            or (
                value["token_scope"] is not None
                and type(value["token_scope"]) is not str
            )
        ):
            raise DataValidationError("activation-feature schema JSON types differ")
        data = dict(value)
        data["runtime"] = Runtime(data["runtime"])
        data["activation_kind"] = ActivationKind(data["activation_kind"])
        data["layers"] = tuple(data["layers"])
        data["sites"] = tuple(ActivationSite(item) for item in data["sites"])
        data["composition"] = FeatureComposition(data["composition"])
        if data["token_scope"] is not None:
            data["token_scope"] = TokenScope(data["token_scope"])
        return cls(**data)

    @property
    def digest(self) -> str:
        return stable_hash(self.to_dict())

    def extraction_identity(self) -> dict[str, Any]:
        """Identity shared by disjoint partitions of the same extraction."""

        value = self.to_dict()
        value.pop("partition")
        return value

    def is_compatible_extraction(self, other: ActivationFeatureSchema) -> bool:
        return self.extraction_identity() == other.extraction_identity()

    def representation_identity(self) -> dict[str, Any]:
        """Model-side identity, allowing deliberate OOD benchmark evaluation."""

        value = self.extraction_identity()
        value.pop("benchmark")
        value.pop("split_manifest_digest")
        return value

    def is_compatible_representation(self, other: ActivationFeatureSchema) -> bool:
        return self.representation_identity() == other.representation_identity()

    def source_identity(self) -> dict[str, Any]:
        """Checkpoint and prompt identity shared by prompt/early-token features."""

        value = self.to_dict()
        return {
            key: value[key]
            for key in (
                "model_repository",
                "model_revision",
                "runtime",
                "quantization",
                "prompt_id",
                "prompt_sha256",
            )
        }

    @classmethod
    def synthetic(
        cls,
        *,
        partition: str,
        width: int,
        layers: tuple[int, ...] = (0,),
        composition: FeatureComposition = FeatureComposition.SINGLE_LAYER,
        activation_kind: ActivationKind = ActivationKind.FINAL_PROMPT,
        token_scope: TokenScope | None = None,
    ) -> ActivationFeatureSchema:
        return cls(
            benchmark="synthetic",
            partition=partition,
            split_manifest_digest="1" * 64,
            model_repository="synthetic/model",
            model_revision="0" * 40,
            runtime=Runtime.SYNTHETIC,
            quantization="none",
            prompt_id="P0-synthetic",
            prompt_sha256="2" * 64,
            activation_kind=activation_kind,
            layers=layers,
            sites=(ActivationSite.POST_MLP,),
            composition=composition,
            width=width,
            token_scope=token_scope,
        )

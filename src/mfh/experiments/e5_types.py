"""Shared immutable E5 fitting types without capture/fitter import cycles."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from mfh.contracts import ActivationSite
from mfh.errors import DataValidationError
from mfh.experiments.e2_schedule import E2_LAYERS


def registered_e5_layer_candidates(
    fixed_best_layer: int,
) -> tuple[tuple[int, int], tuple[int, int, int]]:
    """Derive the frozen nearest-neighbour router layers from the Qwen registry."""

    if type(fixed_best_layer) is not int or fixed_best_layer not in E2_LAYERS:
        raise DataValidationError("E5 fixed-best layer is outside the registered Qwen set")
    nearest = sorted(
        (value for value in E2_LAYERS if value != fixed_best_layer),
        key=lambda value: (abs(value - fixed_best_layer), value),
    )[:2]
    two = sorted((fixed_best_layer, nearest[0]))
    three = sorted((fixed_best_layer, nearest[0], nearest[1]))
    return (two[0], two[1]), (three[0], three[1], three[2])


@dataclass(frozen=True, slots=True)
class E5FitRecipe:
    """All deterministic hyperparameters shared by the registered E5 grid."""

    fixed_best_layer: int
    two_layer_candidates: tuple[int, int]
    three_layer_candidates: tuple[int, int, int]
    intervention_site: ActivationSite
    minimum_class_count: int = 2
    vector_seed: int = 17
    router_seed: int = 17
    router_hidden_width: int = 64
    router_epochs: int = 300
    distance_temperature: float = 1.0
    layer_seed: int = 17
    layer_epochs: int = 300
    alpha_max: float = 0.5
    alpha_beta: float = 12.0
    alpha_threshold: float = 0.5
    schema_version: int = 1

    def __post_init__(self) -> None:
        expected_two, expected_three = registered_e5_layer_candidates(self.fixed_best_layer)
        numeric = (
            self.distance_temperature,
            self.alpha_max,
            self.alpha_beta,
            self.alpha_threshold,
        )
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or type(self.fixed_best_layer) is not int
            or self.fixed_best_layer < 0
            or type(self.two_layer_candidates) is not tuple
            or len(self.two_layer_candidates) != 2
            or type(self.three_layer_candidates) is not tuple
            or len(self.three_layer_candidates) != 3
            or len(set(self.two_layer_candidates)) != 2
            or len(set(self.three_layer_candidates)) != 3
            or self.fixed_best_layer not in self.two_layer_candidates
            or self.fixed_best_layer not in self.three_layer_candidates
            or not set(self.two_layer_candidates) <= set(self.three_layer_candidates)
            or self.two_layer_candidates != expected_two
            or self.three_layer_candidates != expected_three
            or any(
                type(value) is not int or value < 0
                for value in (*self.two_layer_candidates, *self.three_layer_candidates)
            )
            or not isinstance(self.intervention_site, ActivationSite)
            or any(
                type(value) is not int or value <= 0
                for value in (
                    self.minimum_class_count,
                    self.router_hidden_width,
                    self.router_epochs,
                    self.layer_epochs,
                )
            )
            or any(
                type(value) is not int or value < 0
                for value in (self.vector_seed, self.router_seed, self.layer_seed)
            )
            or any(
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                for value in numeric
            )
            or float(self.distance_temperature) <= 0
            or float(self.alpha_max) < 0
            or float(self.alpha_beta) <= 0
            or not 0 <= float(self.alpha_threshold) <= 1
        ):
            raise DataValidationError("E5 fit recipe is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fixed_best_layer": self.fixed_best_layer,
            "two_layer_candidates": list(self.two_layer_candidates),
            "three_layer_candidates": list(self.three_layer_candidates),
            "intervention_site": self.intervention_site.value,
            "minimum_class_count": self.minimum_class_count,
            "vector_seed": self.vector_seed,
            "router_seed": self.router_seed,
            "router_hidden_width": self.router_hidden_width,
            "router_epochs": self.router_epochs,
            "distance_temperature": float(self.distance_temperature),
            "layer_seed": self.layer_seed,
            "layer_epochs": self.layer_epochs,
            "alpha_max": float(self.alpha_max),
            "alpha_beta": float(self.alpha_beta),
            "alpha_threshold": float(self.alpha_threshold),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> E5FitRecipe:
        expected = {
            "schema_version",
            "fixed_best_layer",
            "two_layer_candidates",
            "three_layer_candidates",
            "intervention_site",
            "minimum_class_count",
            "vector_seed",
            "router_seed",
            "router_hidden_width",
            "router_epochs",
            "distance_temperature",
            "layer_seed",
            "layer_epochs",
            "alpha_max",
            "alpha_beta",
            "alpha_threshold",
        }
        if set(value) != expected:
            raise DataValidationError("E5 fit recipe keys differ")
        try:
            return cls(
                fixed_best_layer=value["fixed_best_layer"],
                two_layer_candidates=tuple(value["two_layer_candidates"]),
                three_layer_candidates=tuple(value["three_layer_candidates"]),
                intervention_site=ActivationSite(value["intervention_site"]),
                minimum_class_count=value["minimum_class_count"],
                vector_seed=value["vector_seed"],
                router_seed=value["router_seed"],
                router_hidden_width=value["router_hidden_width"],
                router_epochs=value["router_epochs"],
                distance_temperature=value["distance_temperature"],
                layer_seed=value["layer_seed"],
                layer_epochs=value["layer_epochs"],
                alpha_max=value["alpha_max"],
                alpha_beta=value["alpha_beta"],
                alpha_threshold=value["alpha_threshold"],
                schema_version=value["schema_version"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DataValidationError(f"invalid E5 fit recipe: {exc}") from exc

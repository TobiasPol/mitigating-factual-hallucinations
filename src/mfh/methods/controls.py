"""Negative and causal vector controls from section 7.2."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor

from mfh.contracts import Outcome
from mfh.errors import DataValidationError


def opposite_direction(direction: Tensor) -> Tensor:
    return -direction.detach().clone()


def zero_direction(direction: Tensor) -> Tensor:
    return torch.zeros_like(direction)


def matched_random_direction(direction: Tensor, *, seed: int) -> Tensor:
    if direction.ndim != 1 or direction.numel() == 0:
        raise DataValidationError("reference direction must be a non-empty vector")
    norm = torch.linalg.vector_norm(direction.float())
    if float(norm) <= 0:
        raise DataValidationError("reference direction must have non-zero norm")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random = torch.randint(0, 2, direction.shape, generator=generator, dtype=torch.int64).float()
    random = random.mul(2).sub(1)
    random = random / torch.linalg.vector_norm(random) * norm
    return cast(Tensor, random.to(device=direction.device, dtype=direction.dtype))


def norm_matched_gaussian_perturbation(direction: Tensor, *, seed: int) -> Tensor:
    if direction.ndim != 1 or direction.numel() == 0:
        raise DataValidationError("reference direction must be a non-empty vector")
    norm = torch.linalg.vector_norm(direction.float())
    if float(norm) <= 0:
        raise DataValidationError("reference direction must have non-zero norm")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    gaussian = torch.randn(direction.shape, generator=generator, dtype=torch.float32)
    gaussian = gaussian / torch.linalg.vector_norm(gaussian) * norm
    return cast(Tensor, gaussian.to(device=direction.device, dtype=direction.dtype))


def label_shuffled_centroid_direction(
    activations: Tensor, outcomes: Sequence[Outcome], *, seed: int
) -> Tensor:
    flattened = activations.detach().to(device="cpu", dtype=torch.float64)
    if flattened.ndim != 2 or flattened.shape[0] != len(outcomes):
        raise DataValidationError("label-shuffle activations and outcomes are misaligned")
    labels = [outcome for outcome in outcomes if outcome in {Outcome.CORRECT, Outcome.INCORRECT}]
    if len(labels) != len(outcomes):
        raise DataValidationError("label-shuffle control accepts only C/I examples")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(len(labels), generator=generator).tolist()
    shuffled = [labels[index] for index in permutation]
    correct_mask = torch.tensor([label is Outcome.CORRECT for label in shuffled])
    incorrect_mask = ~correct_mask
    if not correct_mask.any() or not incorrect_mask.any():
        raise DataValidationError("label-shuffle control requires both classes")
    difference = flattened[correct_mask].mean(0) - flattened[incorrect_mask].mean(0)
    norm = torch.linalg.vector_norm(difference)
    if float(norm) <= 0:
        raise DataValidationError("shuffled centroid difference is zero")
    return cast(
        Tensor,
        (difference / norm).to(dtype=activations.dtype, device=activations.device),
    )

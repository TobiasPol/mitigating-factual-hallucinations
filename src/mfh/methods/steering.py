"""Turn immutable vector banks into hook intervention plans."""

from __future__ import annotations

from mfh.contracts import TokenScope
from mfh.inference.architecture import HookKey
from mfh.inference.hooks import InterventionPlan
from mfh.methods.static import VectorBank


def plans_from_vector_bank(
    bank: VectorBank,
    *,
    alpha: float,
    token_scope: TokenScope,
    rms_relative: bool = True,
    decay: float = 0.5,
) -> dict[HookKey, InterventionPlan]:
    return {
        key: InterventionPlan(
            direction=vector.direction,
            alpha=alpha,
            token_scope=token_scope,
            rms_relative=rms_relative,
            decay=decay,
        )
        for key, vector in bank.vectors.items()
    }

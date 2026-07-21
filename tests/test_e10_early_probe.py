from __future__ import annotations

import numpy as np

from mfh.contracts import ActivationSite, Outcome, TokenScope
from mfh.experiments.e10_early_probe import (
    _candidate_config,
    _candidate_key,
    _early_features,
)
from mfh.methods.features import (
    ActivationFeatureSchema,
    ActivationKind,
)
from mfh.methods.probes import CalibrationKind, ProbeKind


def test_early_feature_pooling_distinguishes_first_token_and_short_block() -> None:
    schema = ActivationFeatureSchema.synthetic(
        partition="T-dev",
        width=2,
        activation_kind=ActivationKind.FIRST_FOUR_GENERATED,
        token_scope=TokenScope.FIRST_FOUR,
    )
    activations = {
        ActivationSite.POST_MLP: {
            0: np.asarray(
                [[1.0, 3.0], [3.0, 5.0], [5.0, 7.0], [7.0, 9.0]],
                dtype=np.float32,
            )
        }
    }

    first = _early_features(schema, activations, limit=1)
    block = _early_features(schema, activations, limit=4)

    assert first.tolist() == [1.0, 3.0]
    assert block.tolist() == [4.0, 6.0]


def test_early_probe_selection_prioritizes_incorrect_auroc() -> None:
    better_incorrect = {
        "config": _candidate_config(
            ActivationKind.FIRST_GENERATED,
            ProbeKind.LOGISTIC,
            CalibrationKind.TEMPERATURE,
        ),
        "dev_metrics": {
            "macro_auroc": 0.70,
            "expected_calibration_error": 0.10,
            "per_class_auroc": {Outcome.INCORRECT.value: 0.91},
        },
    }
    better_macro = {
        "config": _candidate_config(
            ActivationKind.FIRST_FOUR_GENERATED,
            ProbeKind.TWO_LAYER_MLP,
            CalibrationKind.ISOTONIC,
        ),
        "dev_metrics": {
            "macro_auroc": 0.95,
            "expected_calibration_error": 0.01,
            "per_class_auroc": {Outcome.INCORRECT.value: 0.90},
        },
    }

    assert min((better_macro, better_incorrect), key=_candidate_key) is better_incorrect


def test_early_probe_candidate_grid_has_eight_unique_frozen_configs() -> None:
    configs = {
        str(
            _candidate_config(kind, probe, calibration)
        )
        for kind in (
            ActivationKind.FIRST_GENERATED,
            ActivationKind.FIRST_FOUR_GENERATED,
        )
        for probe in (ProbeKind.LOGISTIC, ProbeKind.TWO_LAYER_MLP)
        for calibration in (CalibrationKind.TEMPERATURE, CalibrationKind.ISOTONIC)
    }
    assert len(configs) == 8
    assert all("'epochs': 400" in value for value in configs)

"""Test-only isolation for scientific artifact namespace enforcement.

Most unit fixtures intentionally live in pytest temporary directories.  Production
entry points still enforce the Qwen namespace; dedicated boundary tests are excluded
from this fixture and exercise the real validator.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from mfh.artifact_namespace import validate_active_study_artifact_paths


def _canonical_unit_paths(
    paths: Mapping[str, str | Path],
    *,
    project_root: str | Path | None = None,
) -> Mapping[str, Path]:
    del project_root
    return {
        name: Path(path).absolute().resolve(strict=False)
        for name, path in paths.items()
    }


@pytest.fixture(autouse=True)
def isolate_unit_artifact_namespaces(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    filename = Path(str(request.node.path)).name
    if filename in {"test_model_selection_amendment.py", "test_synthetic_study.py"}:
        return
    original_validator = validate_active_study_artifact_paths
    for module in tuple(sys.modules.values()):
        if module is None or module.__name__ == "mfh.experiments.model_selection":
            continue
        if (
            getattr(module, "validate_active_study_artifact_paths", None)
            is original_validator
        ):
            monkeypatch.setattr(
                module,
                "validate_active_study_artifact_paths",
                _canonical_unit_paths,
            )

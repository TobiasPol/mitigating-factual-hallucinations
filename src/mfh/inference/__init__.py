"""Deterministic model execution and activation intervention primitives."""

from mfh.inference.architecture import HookKey, HookPoint, resolve_hook_points
from mfh.inference.hooks import (
    ActivationSession,
    CapturePolicy,
    InterventionPlan,
    PassPhase,
)
from mfh.inference.mlx_research import (
    MlxPromptFeatureCubeOutput,
    MlxPromptFeatureOutput,
    MlxResearchInterventionState,
    MlxResearchRuntime,
    MlxTeacherForcedOutput,
)
from mfh.inference.mlx_runtime import (
    MlxGenerationOutput,
    MlxInterventionState,
    MlxRenderedPrompt,
    MlxRuntime,
)
from mfh.inference.transformers_snapshot import (
    SnapshotFile,
    SnapshotManifest,
    load_snapshot_manifest,
    reject_symlink_path_components,
    verify_transformers_snapshot,
)

__all__ = [
    "ActivationSession",
    "CapturePolicy",
    "HookKey",
    "HookPoint",
    "InterventionPlan",
    "MlxGenerationOutput",
    "MlxInterventionState",
    "MlxPromptFeatureCubeOutput",
    "MlxPromptFeatureOutput",
    "MlxRenderedPrompt",
    "MlxResearchInterventionState",
    "MlxResearchRuntime",
    "MlxRuntime",
    "MlxTeacherForcedOutput",
    "PassPhase",
    "SnapshotFile",
    "SnapshotManifest",
    "load_snapshot_manifest",
    "reject_symlink_path_components",
    "resolve_hook_points",
    "verify_transformers_snapshot",
]

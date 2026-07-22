"""Deterministic model execution and activation intervention primitives."""

from mfh.inference.architecture import HookKey, HookPoint, resolve_hook_points
from mfh.inference.hooks import (
    ActivationSession,
    CapturePolicy,
    InterventionPlan,
    PassPhase,
)
from mfh.inference.transformers_snapshot import (
    SnapshotFile,
    SnapshotManifest,
    load_snapshot_manifest,
    reject_symlink_path_components,
    verify_transformers_snapshot,
)
from mfh.inference.vllm_research import (
    VllmPromptFeatureCubeOutput,
    VllmPromptFeatureOutput,
    VllmResearchInterventionState,
    VllmResearchRuntime,
    VllmTeacherForcedOutput,
)
from mfh.inference.vllm_runtime import (
    VllmGenerationOutput,
    VllmInterventionState,
    VllmRenderedPrompt,
    VllmRuntime,
)

__all__ = [
    "ActivationSession",
    "CapturePolicy",
    "HookKey",
    "HookPoint",
    "InterventionPlan",
    "PassPhase",
    "SnapshotFile",
    "SnapshotManifest",
    "VllmGenerationOutput",
    "VllmInterventionState",
    "VllmPromptFeatureCubeOutput",
    "VllmPromptFeatureOutput",
    "VllmRenderedPrompt",
    "VllmResearchInterventionState",
    "VllmResearchRuntime",
    "VllmRuntime",
    "VllmTeacherForcedOutput",
    "load_snapshot_manifest",
    "reject_symlink_path_components",
    "resolve_hook_points",
    "verify_transformers_snapshot",
]

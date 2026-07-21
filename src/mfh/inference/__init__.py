"""Deterministic model execution and activation intervention primitives."""

from mfh.inference.architecture import HookKey, HookPoint, resolve_hook_points
from mfh.inference.gguf import (
    GgufGenerationOutput,
    LlamaCppCapabilities,
    LlamaCppRuntime,
    LlamaCppStaticControl,
    inspect_llama_cpp,
    verify_gguf_artifact,
)
from mfh.inference.hooks import (
    ActivationSession,
    CapturePolicy,
    InterventionPlan,
    PassPhase,
)
from mfh.inference.llama_server import (
    LlamaServerClient,
    LlamaServerCompletion,
    LlamaServerExpectedIdentity,
    LlamaServerProtocol,
    ManagedLlamaServer,
    load_llama_server_identity,
    resident_set_size_bytes,
    sha256_runtime_tree,
    verify_llama_server_artifacts,
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
    "GgufGenerationOutput",
    "HookKey",
    "HookPoint",
    "InterventionPlan",
    "LlamaCppCapabilities",
    "LlamaCppRuntime",
    "LlamaCppStaticControl",
    "LlamaServerClient",
    "LlamaServerCompletion",
    "LlamaServerExpectedIdentity",
    "LlamaServerProtocol",
    "ManagedLlamaServer",
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
    "inspect_llama_cpp",
    "load_llama_server_identity",
    "load_snapshot_manifest",
    "reject_symlink_path_components",
    "resident_set_size_bytes",
    "resolve_hook_points",
    "sha256_runtime_tree",
    "verify_gguf_artifact",
    "verify_llama_server_artifacts",
    "verify_transformers_snapshot",
]

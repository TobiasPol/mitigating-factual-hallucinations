"""Revision-bound llama.cpp execution for GGUF baseline and static-vector replication."""

from __future__ import annotations

import math
import os
import re
import struct
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from mfh.contracts import ModelSpec, Runtime
from mfh.errors import ConfigurationError, DataValidationError
from mfh.provenance import sha256_file, stable_hash

_DETERMINISTIC_FLAGS = frozenset(
    {
        "--seed",
        "--temp",
        "--no-display-prompt",
        "--no-show-timings",
        "--no-conversation",
        "--reasoning",
        "--offline",
        "--simple-io",
        "--log-disable",
    }
)
_STATIC_CONTROL_FLAGS = frozenset(
    {
        "--control-vector-scaled",
        "--control-vector-layer-range",
    }
)
_ADAPTIVE_INSTRUMENTATION_FLAGS = frozenset(
    {
        "--mfh-hidden-state-export",
        "--mfh-control-sidecar",
    }
)
_GGUF_SUPPORTED_METHODS = frozenset({"M0", "M1", "M1-R", "M1-P"})
_GGUF_FIXED_VALUE_SIZES = {
    0: 1,  # UINT8
    1: 1,  # INT8
    2: 2,  # UINT16
    3: 2,  # INT16
    4: 4,  # UINT32
    5: 4,  # INT32
    6: 4,  # FLOAT32
    7: 1,  # BOOL
    10: 8,  # UINT64
    11: 8,  # INT64
    12: 8,  # FLOAT64
}
_GGUF_STRING = 8
_GGUF_ARRAY = 9
_MAX_GGUF_COLLECTION_ITEMS = 10_000_000
_MAX_GGUF_STRING_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class _ControlVectorLayout:
    layers: tuple[int, ...]
    embedding_size: int


def _regular_file(path: str | Path, context: str) -> Path:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise DataValidationError(f"{context} must be a regular file")
    return source.resolve()


def _read_exact(handle: Any, size: int, context: str) -> bytes:
    value = bytes(handle.read(size))
    if len(value) != size:
        raise DataValidationError(f"{context} has a truncated GGUF structure")
    return value


def _read_u32(handle: Any, context: str) -> int:
    return int(struct.unpack("<I", _read_exact(handle, 4, context))[0])


def _read_u64(handle: Any, context: str) -> int:
    return int(struct.unpack("<Q", _read_exact(handle, 8, context))[0])


def _read_gguf_string(handle: Any, context: str) -> bytes:
    length = _read_u64(handle, context)
    if length > _MAX_GGUF_STRING_BYTES:
        raise DataValidationError(f"{context} contains an implausibly large GGUF string")
    return _read_exact(handle, length, context)


def _read_gguf_value(handle: Any, value_type: int, context: str, *, depth: int = 0) -> Any:
    if value_type in _GGUF_FIXED_VALUE_SIZES:
        raw = _read_exact(handle, _GGUF_FIXED_VALUE_SIZES[value_type], context)
        if value_type == 4:
            return int(struct.unpack("<I", raw)[0])
        return None
    if value_type == _GGUF_STRING:
        _read_gguf_string(handle, context)
        return None
    if value_type == _GGUF_ARRAY:
        if depth >= 1:
            raise DataValidationError(f"{context} contains a nested GGUF array")
        item_type = _read_u32(handle, context)
        count = _read_u64(handle, context)
        if count > _MAX_GGUF_COLLECTION_ITEMS:
            raise DataValidationError(f"{context} contains an implausibly large GGUF array")
        for _ in range(count):
            _read_gguf_value(handle, item_type, context, depth=depth + 1)
        return None
    raise DataValidationError(f"{context} contains unsupported GGUF value type {value_type}")


def _require_gguf_structure(
    path: Path,
    context: str,
    *,
    require_f32_tensors: bool = False,
) -> _ControlVectorLayout | None:
    """Parse the GGUF envelope far enough to reject magic-only and truncated files."""

    try:
        with path.open("rb") as handle:
            magic, version, tensor_count, metadata_count = struct.unpack(
                "<4sIQQ", _read_exact(handle, 24, context)
            )
            if magic != b"GGUF":
                raise DataValidationError(f"{context} has no GGUF header")
            if version not in {2, 3}:
                raise DataValidationError(f"{context} uses unsupported GGUF version {version}")
            if not 1 <= tensor_count <= _MAX_GGUF_COLLECTION_ITEMS:
                raise DataValidationError(f"{context} has an invalid GGUF tensor count")
            if not 1 <= metadata_count <= _MAX_GGUF_COLLECTION_ITEMS:
                raise DataValidationError(f"{context} has an invalid GGUF metadata count")

            alignment = 32
            metadata_keys: set[str] = set()
            for _ in range(metadata_count):
                key_raw = _read_gguf_string(handle, context)
                try:
                    key = key_raw.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise DataValidationError(
                        f"{context} has a non-UTF-8 GGUF metadata key"
                    ) from exc
                if not key or key in metadata_keys:
                    raise DataValidationError(f"{context} has an empty or duplicate metadata key")
                metadata_keys.add(key)
                value_type = _read_u32(handle, context)
                value = _read_gguf_value(handle, value_type, context)
                if key == "general.alignment":
                    if value_type != 4 or not isinstance(value, int):
                        raise DataValidationError(
                            f"{context} has an invalid general.alignment value"
                        )
                    alignment = value
            if alignment < 1 or alignment > 4096 or alignment & (alignment - 1):
                raise DataValidationError(f"{context} has an invalid GGUF alignment")

            tensor_descriptors: list[tuple[int, int, int, bytes]] = []
            tensor_names: set[bytes] = set()
            control_layers: list[int] = []
            control_embedding_size: int | None = None
            for _ in range(tensor_count):
                tensor_name = _read_gguf_string(handle, context)
                if not tensor_name or len(tensor_name) > 64 or tensor_name in tensor_names:
                    raise DataValidationError(
                        f"{context} has an empty, duplicate, or oversized GGUF tensor name"
                    )
                tensor_names.add(tensor_name)
                dimensions = _read_u32(handle, context)
                if not 1 <= dimensions <= 4:
                    raise DataValidationError(f"{context} has an invalid GGUF tensor rank")
                elements = 1
                for _ in range(dimensions):
                    dimension = _read_u64(handle, context)
                    if dimension < 1:
                        raise DataValidationError(
                            f"{context} has a non-positive GGUF tensor dimension"
                        )
                    elements *= dimension
                    if elements > (1 << 63) - 1:
                        raise DataValidationError(f"{context} has an implausibly large GGUF tensor")
                ggml_type = _read_u32(handle, context)
                if ggml_type > 64:
                    raise DataValidationError(f"{context} has an invalid GGML tensor type")
                if require_f32_tensors and ggml_type != 0:
                    raise DataValidationError(
                        "llama.cpp control vectors must contain only F32 tensors"
                    )
                if require_f32_tensors:
                    try:
                        decoded_name = tensor_name.decode("ascii")
                    except UnicodeDecodeError as exc:
                        raise DataValidationError(
                            "llama.cpp control-vector tensor names must be ASCII"
                        ) from exc
                    name_match = re.fullmatch(r"direction\.([1-9][0-9]*)", decoded_name)
                    if dimensions != 1 or name_match is None:
                        raise DataValidationError(
                            "llama.cpp control tensors must be one-dimensional direction.N"
                        )
                    layer = int(name_match.group(1))
                    if control_embedding_size is None:
                        control_embedding_size = elements
                    elif elements != control_embedding_size:
                        raise DataValidationError(
                            "llama.cpp control tensors have inconsistent embedding extents"
                        )
                    control_layers.append(layer)
                tensor_descriptors.append(
                    (_read_u64(handle, context), elements, dimensions, tensor_name)
                )

            data_start = (handle.tell() + alignment - 1) // alignment * alignment
            file_size = path.stat().st_size
            if data_start >= file_size:
                raise DataValidationError(f"{context} has no GGUF tensor data")
            prior_offset = -1
            expected_f32_offset = 0
            for offset, elements, _, _ in tensor_descriptors:
                if offset % alignment or offset <= prior_offset or data_start + offset >= file_size:
                    raise DataValidationError(f"{context} has an invalid GGUF tensor offset")
                prior_offset = offset
                if require_f32_tensors:
                    tensor_bytes = elements * 4
                    if (
                        offset != expected_f32_offset
                        or data_start + offset + tensor_bytes > file_size
                    ):
                        raise DataValidationError(
                            f"{context} has truncated or non-canonical F32 tensor data"
                        )
                    expected_f32_offset += (tensor_bytes + alignment - 1) // alignment * alignment
                    handle.seek(data_start + offset)
                    raw_tensor = _read_exact(handle, tensor_bytes, context)
                    if any(
                        not math.isfinite(value[0])
                        for value in struct.iter_unpack("<f", raw_tensor)
                    ):
                        raise DataValidationError(
                            "llama.cpp control vectors contain non-finite F32 values"
                        )
            if require_f32_tensors:
                expected_layers = tuple(range(1, len(control_layers) + 1))
                if tuple(control_layers) != expected_layers or control_embedding_size is None:
                    raise DataValidationError(
                        "llama.cpp control-vector layers must be ordered and contiguous from 1"
                    )
                return _ControlVectorLayout(expected_layers, control_embedding_size)
            return None
    except DataValidationError:
        raise
    except (OSError, struct.error) as exc:
        raise DataValidationError(f"cannot inspect {context}: {exc}") from exc


def verify_gguf_artifact(model_spec: ModelSpec, path: str | Path) -> Path:
    """Verify the exact configured GGUF filename, size, and SHA-256."""

    if model_spec.runtime is not Runtime.LLAMA_CPP:
        raise ConfigurationError("GGUF artifact verification requires a llama.cpp model")
    source = _regular_file(path, "GGUF model artifact")
    if (
        source.name != model_spec.artifact
        or source.stat().st_size != model_spec.artifact_size_bytes
        or sha256_file(source) != model_spec.artifact_sha256
    ):
        raise DataValidationError("GGUF model artifact differs from its pinned configuration")
    _require_gguf_structure(source, "GGUF model artifact")
    return source


def _run_identity_command(binary: Path, argument: str) -> str:
    try:
        result = subprocess.run(
            (str(binary), argument),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env={
                "PATH": os.environ.get("PATH", ""),
                "LANG": "C",
                "LC_ALL": "C",
            },
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ConfigurationError(f"cannot inspect llama.cpp binary: {exc}") from exc
    output = "\n".join(value.strip() for value in (result.stdout, result.stderr) if value.strip())
    if result.returncode != 0 or not output:
        raise ConfigurationError(f"llama.cpp {argument} failed with exit code {result.returncode}")
    return output


@dataclass(frozen=True, slots=True)
class LlamaCppCapabilities:
    binary_sha256: str
    version_output: str
    version_digest: str
    help_sha256: str
    deterministic_cli_supported: bool
    static_control_supported: bool
    adaptive_instrumentation_supported: bool

    @property
    def scope(self) -> str:
        if self.static_control_supported:
            return "static-replication-only"
        return "baseline-only"

    @property
    def digest(self) -> str:
        return stable_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "binary_sha256": self.binary_sha256,
            "version_output": self.version_output,
            "version_digest": self.version_digest,
            "help_sha256": self.help_sha256,
            "deterministic_cli_supported": self.deterministic_cli_supported,
            "static_control_supported": self.static_control_supported,
            "adaptive_instrumentation_supported": self.adaptive_instrumentation_supported,
            "scope": self.scope,
        }


def inspect_llama_cpp(
    binary_path: str | Path,
    *,
    expected_binary_sha256: str | None = None,
    expected_version_digest: str | None = None,
) -> LlamaCppCapabilities:
    """Inspect a local binary and optionally enforce its previously frozen identity."""

    binary = _regular_file(binary_path, "llama.cpp binary")
    if not os.access(binary, os.X_OK):
        raise ConfigurationError("llama.cpp binary is not executable")
    binary_sha256 = sha256_file(binary)
    if expected_binary_sha256 is not None and binary_sha256 != expected_binary_sha256:
        raise ConfigurationError("llama.cpp binary differs from its frozen SHA-256")
    version = _run_identity_command(binary, "--version")
    if sha256_file(binary) != binary_sha256:
        raise ConfigurationError("llama.cpp binary changed while reporting its version")
    help_output = _run_identity_command(binary, "--help")
    if sha256_file(binary) != binary_sha256:
        raise ConfigurationError("llama.cpp binary changed while reporting its help")
    version_digest = stable_hash({"version_output": version})
    if expected_version_digest is not None and version_digest != expected_version_digest:
        raise ConfigurationError("llama.cpp version differs from its frozen identity")
    option_tokens = frozenset(
        re.findall(r"(?<![A-Za-z0-9-])--[A-Za-z0-9][A-Za-z0-9-]*(?![A-Za-z0-9-])", help_output)
    )
    deterministic = option_tokens >= _DETERMINISTIC_FLAGS
    scaled_control_lines = tuple(
        line for line in help_output.splitlines() if "--control-vector-scaled" in line
    )
    current_scaled_syntax = any(
        re.search(
            r"(?<![A-Za-z0-9-])--control-vector-scaled\s+"
            r"<?FNAME>?:<?SCALE>?(?=$|[\s,;])",
            line,
        )
        for line in scaled_control_lines
    )
    current_layer_range_syntax = any(
        re.search(
            r"(?<![A-Za-z0-9-])--control-vector-layer-range\s+"
            r"<?START>?\s+<?END>?(?=$|[\s,;])",
            line,
        )
        for line in help_output.splitlines()
    )
    static = (
        deterministic
        and option_tokens >= _STATIC_CONTROL_FLAGS
        and current_scaled_syntax
        and current_layer_range_syntax
    )
    adaptive = static and option_tokens >= _ADAPTIVE_INSTRUMENTATION_FLAGS
    return LlamaCppCapabilities(
        binary_sha256=binary_sha256,
        version_output=version,
        version_digest=version_digest,
        help_sha256=stable_hash({"help_output": help_output}),
        deterministic_cli_supported=deterministic,
        static_control_supported=static,
        adaptive_instrumentation_supported=adaptive,
    )


@dataclass(frozen=True, slots=True)
class LlamaCppStaticControl:
    path: Path
    sha256: str
    scale: float
    layer_start: int
    layer_end: int

    def __post_init__(self) -> None:
        if isinstance(self.scale, bool) or not isinstance(self.scale, (int, float)):
            raise DataValidationError("llama.cpp control-vector scale must be numeric")
        if (
            isinstance(self.layer_start, bool)
            or not isinstance(self.layer_start, int)
            or isinstance(self.layer_end, bool)
            or not isinstance(self.layer_end, int)
        ):
            raise DataValidationError("llama.cpp control-vector layer bounds must be integers")
        source = _regular_file(self.path, "llama.cpp control vector")
        if sha256_file(source) != self.sha256:
            raise DataValidationError("llama.cpp control vector differs from its frozen SHA-256")
        layout = _require_gguf_structure(
            source,
            "llama.cpp control vector",
            require_f32_tensors=True,
        )
        if not math.isfinite(self.scale) or abs(self.scale) < 1e-6:
            raise DataValidationError("llama.cpp control-vector scale must be finite and nonzero")
        if self.layer_start < 1 or self.layer_end < self.layer_start:
            raise DataValidationError("llama.cpp control-vector layer range is invalid")
        if layout is None or self.layer_end > layout.layers[-1]:
            raise DataValidationError(
                "llama.cpp control-vector layer range exceeds its direction tensors"
            )
        object.__setattr__(self, "path", source)
        object.__setattr__(self, "scale", float(self.scale))

    def validate(self) -> None:
        """Revalidate the frozen bytes immediately before command construction."""

        source = _regular_file(self.path, "llama.cpp control vector")
        if source != self.path or sha256_file(source) != self.sha256:
            raise DataValidationError("llama.cpp control vector differs from its frozen SHA-256")
        layout = _require_gguf_structure(
            source,
            "llama.cpp control vector",
            require_f32_tensors=True,
        )
        if layout is None or self.layer_end > layout.layers[-1]:
            raise DataValidationError(
                "llama.cpp control-vector layer range exceeds its direction tensors"
            )

    @property
    def digest(self) -> str:
        return stable_hash(
            {
                "sha256": self.sha256,
                "scale": self.scale,
                "layer_start": self.layer_start,
                "layer_end": self.layer_end,
            }
        )


@dataclass(frozen=True, slots=True)
class GgufGenerationOutput:
    text: str
    latency_seconds: float
    execution_identity: str
    metadata: MappingProxyType[str, Any]


class LlamaCppRuntime:
    """Subprocess adapter with an explicit static/adaptive scientific boundary."""

    def __init__(
        self,
        *,
        model_spec: ModelSpec,
        model_path: str | Path,
        binary_path: str | Path,
        binary_sha256: str,
        version_digest: str,
        seed: int = 17,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ConfigurationError("llama.cpp seed cannot be negative")
        self.model_spec = model_spec
        self.model_path = verify_gguf_artifact(model_spec, model_path)
        self.binary_path = _regular_file(binary_path, "llama.cpp binary")
        self.capabilities = inspect_llama_cpp(
            self.binary_path,
            expected_binary_sha256=binary_sha256,
            expected_version_digest=version_digest,
        )
        if not self.capabilities.deterministic_cli_supported:
            raise ConfigurationError("llama.cpp binary lacks the deterministic CLI contract")
        self.seed = seed

    def _validate_execution_artifacts(self, control: LlamaCppStaticControl | None) -> None:
        model = verify_gguf_artifact(self.model_spec, self.model_path)
        if model != self.model_path:
            raise DataValidationError("GGUF model artifact path changed after initialization")
        binary = _regular_file(self.binary_path, "llama.cpp binary")
        if binary != self.binary_path or not os.access(binary, os.X_OK):
            raise ConfigurationError("llama.cpp binary identity changed after initialization")
        if sha256_file(binary) != self.capabilities.binary_sha256:
            raise ConfigurationError("llama.cpp binary differs from its frozen SHA-256")
        if control is not None:
            control.validate()

    def assert_method_supported(self, method: str) -> None:
        if not isinstance(method, str):
            raise ConfigurationError("GGUF method name must be a string")
        normalized = method.strip().upper()
        if not normalized:
            raise ConfigurationError("GGUF method name must be non-empty")
        if normalized not in _GGUF_SUPPORTED_METHODS:
            raise ConfigurationError(
                f"{normalized} is not implemented by the static GGUF adapter; "
                "use the unpacked Transformers runtime or a validated adaptive adapter"
            )

    def build_command(
        self,
        prompt_file: str | Path,
        *,
        max_new_tokens: int,
        control: LlamaCppStaticControl | None = None,
    ) -> tuple[str, ...]:
        prompt = _regular_file(prompt_file, "llama.cpp prompt file")
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or not 1 <= max_new_tokens <= 48
        ):
            raise ConfigurationError("GGUF generation length must be in [1, 48]")
        self._validate_execution_artifacts(control)
        command = [
            str(self.binary_path),
            "--model",
            str(self.model_path),
            "--file",
            str(prompt),
            "--predict",
            str(max_new_tokens),
            "--seed",
            str(self.seed),
            "--temp",
            "0",
            "--no-display-prompt",
            "--no-show-timings",
            "--no-conversation",
            "--reasoning",
            "off",
            "--offline",
            "--simple-io",
            "--log-disable",
        ]
        if control is not None:
            if not self.capabilities.static_control_supported:
                raise ConfigurationError("llama.cpp binary lacks static control-vector support")
            if control.layer_end >= self.model_spec.num_layers:
                raise DataValidationError("control-vector range exceeds the configured model")
            command.extend(
                (
                    "--control-vector-scaled",
                    f"{control.path}:{control.scale:.17g}",
                    "--control-vector-layer-range",
                    str(control.layer_start),
                    str(control.layer_end),
                )
            )
        return tuple(command)

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 48,
        control: LlamaCppStaticControl | None = None,
        timeout_seconds: float = 600,
    ) -> GgufGenerationOutput:
        if not isinstance(prompt, str) or not prompt.strip():
            raise DataValidationError("GGUF rendered prompt must be non-empty")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ConfigurationError("GGUF timeout must be finite and positive")
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".prompt") as handle:
            handle.write(prompt)
            handle.flush()
            command = self.build_command(
                handle.name,
                max_new_tokens=max_new_tokens,
                control=control,
            )
            command_identity = list(command)
            prompt_argument = command_identity.index("--file") + 1
            command_identity[prompt_argument] = "<rendered-prompt>"
            started = time.perf_counter()
            try:
                result = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    env={
                        "PATH": os.environ.get("PATH", ""),
                        "LANG": "C",
                        "LC_ALL": "C",
                    },
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise DataValidationError(f"llama.cpp generation failed: {exc}") from exc
        latency = time.perf_counter() - started
        # Refuse to attest an execution if any path-bound artifact changed while the
        # subprocess was live. This complements the immediately-before-spawn check.
        self._validate_execution_artifacts(control)
        text = result.stdout.strip()
        if result.returncode != 0 or not text:
            raise DataValidationError(
                f"llama.cpp generation exited {result.returncode}: {result.stderr.strip()}"
            )
        metadata: dict[str, Any] = {
            "runtime": Runtime.LLAMA_CPP.value,
            "model_sha256": self.model_spec.artifact_sha256,
            "binary_sha256": self.capabilities.binary_sha256,
            "binary_version_digest": self.capabilities.version_digest,
            "capability_digest": self.capabilities.digest,
            "deployment_scope": self.capabilities.scope,
            "control_vector_digest": control.digest if control is not None else None,
            "rendered_prompt_hash": stable_hash(prompt),
            "generated_output_hash": stable_hash(text),
            "command_digest": stable_hash(tuple(command_identity)),
            "timeout_seconds": float(timeout_seconds),
            "artifact_validation": "pre-and-post-execution",
        }
        return GgufGenerationOutput(
            text=text,
            latency_seconds=latency,
            execution_identity=stable_hash(metadata),
            metadata=MappingProxyType(metadata),
        )

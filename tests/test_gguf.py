from __future__ import annotations

import math
import os
import struct
from pathlib import Path

import pytest

from mfh.contracts import ModelSpec, Runtime
from mfh.errors import ConfigurationError, DataValidationError
from mfh.inference.gguf import (
    LlamaCppRuntime,
    LlamaCppStaticControl,
    inspect_llama_cpp,
)
from mfh.provenance import sha256_file, stable_hash


def _fake_binary(
    path: Path,
    *,
    adaptive: bool = False,
    self_modify_on_generate: bool = False,
) -> None:
    adaptive_flags = "--mfh-hidden-state-export --mfh-control-sidecar" if adaptive else ""
    mutation = 'echo "# mutation" >> "$0"\n' if self_modify_on_generate else ""
    path.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then echo \'llama.cpp synthetic b9999\'; exit 0; fi\n'
        'if [ "$1" = "--help" ]; then\n'
        "  echo '--seed --temp --no-display-prompt --no-show-timings --no-conversation "
        "--reasoning --offline --simple-io --log-disable'\n"
        "  echo '--control-vector-scaled FNAME:SCALE'\n"
        f"  echo '--control-vector-layer-range START END {adaptive_flags}'\n"
        "  exit 0\n"
        "fi\n"
        f"{mutation}"
        "echo 'Paris'\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _write_minimal_gguf(path: Path, payload: bytes) -> None:
    key = b"general.alignment"
    name = b"toy.weight"
    raw = bytearray(struct.pack("<4sIQQ", b"GGUF", 3, 1, 1))
    raw.extend(struct.pack("<Q", len(key)))
    raw.extend(key)
    raw.extend(struct.pack("<II", 4, 32))
    raw.extend(struct.pack("<Q", len(name)))
    raw.extend(name)
    raw.extend(struct.pack("<IQIQ", 1, 1, 0, 0))
    raw.extend(b"\0" * ((-len(raw)) % 32))
    raw.extend(payload)
    path.write_bytes(raw)


def _write_f32_gguf(
    path: Path,
    tensors: tuple[tuple[str, tuple[int, ...], float], ...],
) -> None:
    key = b"general.alignment"
    raw = bytearray(struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), 1))
    raw.extend(struct.pack("<Q", len(key)))
    raw.extend(key)
    raw.extend(struct.pack("<II", 4, 32))
    offset = 0
    tensor_data: list[bytes] = []
    for name, dimensions, value in tensors:
        encoded_name = name.encode("ascii")
        raw.extend(struct.pack("<Q", len(encoded_name)))
        raw.extend(encoded_name)
        raw.extend(struct.pack("<I", len(dimensions)))
        for dimension in dimensions:
            raw.extend(struct.pack("<Q", dimension))
        raw.extend(struct.pack("<IQ", 0, offset))
        elements = math.prod(dimensions)
        data = struct.pack(f"<{elements}f", *([value] * elements))
        tensor_data.append(data)
        offset += (len(data) + 31) // 32 * 32
    raw.extend(b"\0" * ((-len(raw)) % 32))
    for data in tensor_data:
        raw.extend(data)
        raw.extend(b"\0" * ((-len(data)) % 32))
    path.write_bytes(raw)


def _write_control_gguf(path: Path, *, value: float = 1.0) -> None:
    _write_f32_gguf(
        path,
        tuple((f"direction.{layer}", (2,), value) for layer in range(1, 8)),
    )


def _write_truncated_f32_gguf(path: Path) -> None:
    key = b"general.alignment"
    name = b"direction.1"
    raw = bytearray(struct.pack("<4sIQQ", b"GGUF", 3, 1, 1))
    raw.extend(struct.pack("<Q", len(key)))
    raw.extend(key)
    raw.extend(struct.pack("<II", 4, 32))
    raw.extend(struct.pack("<Q", len(name)))
    raw.extend(name)
    raw.extend(struct.pack("<IQIQ", 1, 1_000_000_000, 0, 0))
    raw.extend(b"\0" * ((-len(raw)) % 32))
    raw.extend(b"\0")
    path.write_bytes(raw)


def _model(path: Path) -> ModelSpec:
    return ModelSpec(
        name="toy-gguf",
        repository="local/toy",
        revision="a" * 40,
        runtime=Runtime.LLAMA_CPP,
        quantization="Q2_0_g128",
        num_layers=8,
        artifact=path.name,
        artifact_sha256=sha256_file(path),
        artifact_size_bytes=path.stat().st_size,
    )


def test_static_gguf_runtime_is_byte_bound_and_rejects_adaptive_claims(
    tmp_path: Path,
) -> None:
    binary = tmp_path / "llama-cli"
    _fake_binary(binary)
    model_path = tmp_path / "toy.gguf"
    _write_minimal_gguf(model_path, b"toy-model")
    control_path = tmp_path / "control.gguf"
    _write_control_gguf(control_path)
    capabilities = inspect_llama_cpp(binary)
    assert capabilities.deterministic_cli_supported
    assert capabilities.static_control_supported
    assert not capabilities.adaptive_instrumentation_supported
    model_spec = _model(model_path)
    runtime = LlamaCppRuntime(
        model_spec=model_spec,
        model_path=model_path,
        binary_path=binary,
        binary_sha256=capabilities.binary_sha256,
        version_digest=capabilities.version_digest,
    )
    control = LlamaCppStaticControl(
        path=control_path,
        sha256=sha256_file(control_path),
        scale=0.8,
        layer_start=2,
        layer_end=6,
    )
    output = runtime.generate("Question: capital of France?", control=control)
    assert output.text == "Paris"
    assert output.metadata["deployment_scope"] == "static-replication-only"
    assert output.metadata["control_vector_digest"] == control.digest
    assert output.metadata["rendered_prompt_hash"] == stable_hash("Question: capital of France?")
    repeated = runtime.generate("Question: capital of France?", control=control)
    assert repeated.execution_identity == output.execution_identity
    changed_prompt = runtime.generate("Question: capital of Germany?", control=control)
    assert changed_prompt.execution_identity != output.execution_identity
    for unsupported in ("M3", "m4b", " m6 ", "M6-custom", "M3_dynamic", "unknown"):
        with pytest.raises(ConfigurationError, match="not implemented by the static GGUF adapter"):
            runtime.assert_method_supported(unsupported)
    _write_minimal_gguf(model_path, b"tampered-model")
    with pytest.raises(DataValidationError, match="pinned configuration"):
        runtime.generate("Question: capital of France?", control=control)
    with pytest.raises(DataValidationError, match="pinned configuration"):
        LlamaCppRuntime(
            model_spec=model_spec,
            model_path=model_path,
            binary_path=binary,
            binary_sha256=capabilities.binary_sha256,
            version_digest=capabilities.version_digest,
        )


def test_custom_adaptive_capability_requires_both_explicit_instrumentation_flags(
    tmp_path: Path,
) -> None:
    binary = tmp_path / "llama-cli"
    _fake_binary(binary, adaptive=True)
    capabilities = inspect_llama_cpp(binary)
    assert capabilities.adaptive_instrumentation_supported
    assert capabilities.scope == "static-replication-only"
    binary.write_text(binary.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")
    os.chmod(binary, 0o755)
    with pytest.raises(ConfigurationError, match="frozen SHA-256"):
        inspect_llama_cpp(binary, expected_binary_sha256=capabilities.binary_sha256)


def test_runtime_revalidates_binary_and_control_before_execution(tmp_path: Path) -> None:
    binary = tmp_path / "llama-cli"
    _fake_binary(binary)
    model_path = tmp_path / "toy.gguf"
    control_path = tmp_path / "control.gguf"
    _write_minimal_gguf(model_path, b"toy-model")
    _write_control_gguf(control_path)
    capabilities = inspect_llama_cpp(binary)
    runtime = LlamaCppRuntime(
        model_spec=_model(model_path),
        model_path=model_path,
        binary_path=binary,
        binary_sha256=capabilities.binary_sha256,
        version_digest=capabilities.version_digest,
    )
    control = LlamaCppStaticControl(
        path=control_path,
        sha256=sha256_file(control_path),
        scale=1.0,
        layer_start=1,
        layer_end=7,
    )
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("hello", encoding="utf-8")

    _write_control_gguf(control_path, value=2.0)
    with pytest.raises(DataValidationError, match="frozen SHA-256"):
        runtime.build_command(prompt, max_new_tokens=8, control=control)

    _write_control_gguf(control_path)
    binary.write_text("#!/bin/sh\necho TAMPERED_EXECUTABLE_RAN\n", encoding="utf-8")
    binary.chmod(0o755)
    with pytest.raises(ConfigurationError, match="frozen SHA-256"):
        runtime.build_command(prompt, max_new_tokens=8)


def test_gguf_structure_cli_tokens_and_numeric_contracts(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.gguf"
    malformed.write_bytes(b"GGUF")
    with pytest.raises(DataValidationError, match="truncated GGUF"):
        LlamaCppStaticControl(
            path=malformed,
            sha256=sha256_file(malformed),
            scale=1.0,
            layer_start=1,
            layer_end=2,
        )

    truncated = tmp_path / "truncated-tensor.gguf"
    _write_truncated_f32_gguf(truncated)
    with pytest.raises(DataValidationError, match="truncated or non-canonical"):
        LlamaCppStaticControl(
            path=truncated,
            sha256=sha256_file(truncated),
            scale=1.0,
            layer_start=1,
            layer_end=2,
        )

    old_binary = tmp_path / "old-llama-cli"
    old_binary.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then echo old; exit 0; fi\n'
        "echo '--seed-unsupported --temp-unsupported --no-display-prompt-unsupported "
        "--no-show-timings-unsupported --no-conversation-unsupported --reasoning-unsupported "
        "--offline-unsupported --simple-io-unsupported --log-disable-unsupported "
        "--control-vector-scaled-unsupported FNAME SCALE "
        "--control-vector-layer-range-unsupported'\n",
        encoding="utf-8",
    )
    old_binary.chmod(0o755)
    capabilities = inspect_llama_cpp(old_binary)
    assert not capabilities.deterministic_cli_supported
    assert not capabilities.static_control_supported

    for index, invalid_argument in enumerate(
        ("SCALE:FNAME", "--unrelated:VALUE", "FNAME SCALE", "===FNAME:SCALE")
    ):
        invalid_binary = tmp_path / f"invalid-llama-cli-{index}"
        invalid_binary.write_text(
            "#!/bin/sh\n"
            'if [ "$1" = "--version" ]; then echo version; exit 0; fi\n'
            "echo '--seed --temp --no-display-prompt --no-show-timings --no-conversation "
            "--reasoning --offline --simple-io --log-disable'\n"
            f"echo '--control-vector-scaled {invalid_argument}'\n"
            "echo '--control-vector-layer-range START END'\n",
            encoding="utf-8",
        )
        invalid_binary.chmod(0o755)
        assert not inspect_llama_cpp(invalid_binary).static_control_supported

    invalid_layer_binary = tmp_path / "invalid-layer-llama-cli"
    invalid_layer_binary.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then echo version; exit 0; fi\n'
        "echo '--seed --temp --no-display-prompt --no-show-timings --no-conversation "
        "--reasoning --offline --simple-io --log-disable'\n"
        "echo '--control-vector-scaled FNAME:SCALE'\n"
        "echo '--control-vector-layer-range START=END'\n",
        encoding="utf-8",
    )
    invalid_layer_binary.chmod(0o755)
    assert not inspect_llama_cpp(invalid_layer_binary).static_control_supported

    invalid_controls = (
        (("toy.weight", (2,), 1.0),),
        (("direction.1", (1, 2), 1.0),),
        (("direction.1", (2,), 1.0), ("direction.2", (3,), 1.0)),
        (("direction.1", (2,), math.nan),),
        (("direction.1", (2,), 1.0), ("direction.3", (2,), 1.0)),
    )
    for index, tensors in enumerate(invalid_controls):
        invalid_control = tmp_path / f"invalid-control-{index}.gguf"
        _write_f32_gguf(invalid_control, tensors)
        with pytest.raises(DataValidationError):
            LlamaCppStaticControl(
                path=invalid_control,
                sha256=sha256_file(invalid_control),
                scale=1.0,
                layer_start=1,
                layer_end=1,
            )

    short_control = tmp_path / "short-control.gguf"
    _write_f32_gguf(short_control, (("direction.1", (2,), 1.0),))
    with pytest.raises(DataValidationError, match="range exceeds"):
        LlamaCppStaticControl(
            path=short_control,
            sha256=sha256_file(short_control),
            scale=1.0,
            layer_start=1,
            layer_end=2,
        )

    control_path = tmp_path / "control.gguf"
    _write_control_gguf(control_path)
    kwargs = {
        "path": control_path,
        "sha256": sha256_file(control_path),
        "scale": 1.0,
    }
    with pytest.raises(DataValidationError, match="bounds must be integers"):
        LlamaCppStaticControl(**kwargs, layer_start=1.5, layer_end=7.5)  # type: ignore[arg-type]


def test_self_modifying_binary_is_rejected_during_inspection(tmp_path: Path) -> None:
    binary = tmp_path / "llama-cli"
    binary.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then echo version; exit 0; fi\n'
        'echo "# mutation" >> "$0"\n'
        "echo '--help'\n",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    with pytest.raises(ConfigurationError, match="changed while reporting its help"):
        inspect_llama_cpp(binary)


def test_generation_rejects_artifact_mutation_during_subprocess(tmp_path: Path) -> None:
    binary = tmp_path / "llama-cli"
    _fake_binary(binary, self_modify_on_generate=True)
    model_path = tmp_path / "toy.gguf"
    _write_minimal_gguf(model_path, b"toy-model")
    capabilities = inspect_llama_cpp(binary)
    runtime = LlamaCppRuntime(
        model_spec=_model(model_path),
        model_path=model_path,
        binary_path=binary,
        binary_sha256=capabilities.binary_sha256,
        version_digest=capabilities.version_digest,
    )
    with pytest.raises(ConfigurationError, match="frozen SHA-256"):
        runtime.generate("Question: capital of France?")

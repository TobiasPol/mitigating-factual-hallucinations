from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from mfh.errors import ConfigurationError, DataValidationError
from mfh.inference.llama_server import (
    LlamaServerClient,
    LlamaServerExpectedIdentity,
    LlamaServerProtocol,
    ManagedLlamaServer,
    load_llama_server_identity,
    sha256_runtime_tree,
    verify_llama_server_artifacts,
)
from mfh.provenance import sha256_file, sha256_path, stable_hash


@pytest.fixture(autouse=True)
def _treat_test_scripts_as_non_darwin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mfh.inference.llama_server.platform.system", lambda: "Test")


def _source_checkout(path: Path) -> tuple[Path, str, str]:
    path.mkdir()
    repository = "https://example.test/llama.cpp"
    subprocess.run(("git", "init", "-q", str(path)), check=True)
    (path / "README.md").write_text("test checkout\n", encoding="utf-8")
    subprocess.run(("git", "-C", str(path), "add", "README.md"), check=True)
    environment = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.test",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.test",
    }
    subprocess.run(
        ("git", "-C", str(path), "commit", "-q", "-m", "test"),
        check=True,
        env=environment,
    )
    subprocess.run(("git", "-C", str(path), "remote", "add", "origin", repository), check=True)
    revision = subprocess.run(
        ("git", "-C", str(path), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return path, revision, repository


def _expected_identity(
    binary: Path,
    template: str,
    source: Path,
    revision: str,
    repository: str,
) -> LlamaServerExpectedIdentity:
    version_output = "version: test-server"
    return LlamaServerExpectedIdentity(
        source_repository=repository,
        source_revision=revision,
        source_path=source,
        binary_path=binary,
        binary_sha256=sha256_file(binary),
        build_tree_sha256=sha256_path(binary.parent),
        build_tree_layout_sha256=sha256_runtime_tree(binary.parent),
        version_digest=stable_hash({"version_output": version_output}),
        build_info="test-build",
        chat_template_stable_hash=stable_hash(template),
    )


def _fake_binary(path: Path) -> None:
    path.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then echo "version: test-server"; exit 0; fi\n'
        "exit 1\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _server(
    *, model_path: Path, protocol: LlamaServerProtocol, template: str
) -> tuple[ThreadingHTTPServer, type[BaseHTTPRequestHandler]]:
    class Handler(BaseHTTPRequestHandler):
        cache_n = 0
        numeric_forgery = False

        def log_message(self, format: str, *args: object) -> None:
            return

        def _send(self, value: object) -> None:
            body = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send({"status": "ok"})
                return
            if self.path == "/slots":
                self._send(
                    [
                        {
                            "id": False if type(self).numeric_forgery else 0,
                            "n_ctx": 2048.0 if type(self).numeric_forgery else 2048,
                            "speculative": False,
                            "is_processing": False,
                        }
                    ]
                )
                return
            if self.path == "/props":
                self._send(
                    {
                        "default_generation_settings": {
                            "params": {"seed": 17.0 if type(self).numeric_forgery else 17},
                            "n_ctx": 2048.0 if type(self).numeric_forgery else 2048,
                        },
                        "total_slots": True if type(self).numeric_forgery else 1,
                        "model_alias": model_path.name,
                        "model_path": str(model_path),
                        "modalities": {"vision": False, "audio": False},
                        "chat_template": template,
                        "chat_template_caps": {"supports_system_role": True},
                        "build_info": "test-build",
                        "is_sleeping": False,
                    }
                )
                return
            self.send_error(404)

        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length))
            if self.path == "/apply-template":
                system = payload["messages"][0]["content"]
                question = payload["messages"][1]["content"]
                self._send(
                    {
                        "prompt": (
                            f"<|im_start|>system\n{system}<|im_end|>\n"
                            f"<|im_start|>user\n{question}<|im_end|>\n"
                            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                        )
                    }
                )
                return
            if self.path == "/completion":
                self._send(
                    {
                        "content": "Paris",
                        "tokens": [123],
                        "model": model_path.name,
                        "tokens_predicted": 1,
                        "tokens_evaluated": 12,
                        "generation_settings": {
                            "seed": 17,
                            "temperature": 0.0,
                            "top_k": 0,
                            "top_p": 1.0,
                            "min_p": 0.0,
                            "typical_p": 1.0,
                            "repeat_penalty": 1.0,
                            "n_predict": 48,
                            "stop": ["<|im_end|>"],
                            "samplers": ["temperature"],
                            "reasoning_format": "none",
                            "speculative.types": "none",
                            "backend_sampling": False,
                            "lora": [],
                        },
                        "prompt": payload["prompt"],
                        "stop": True,
                        "truncated": False,
                        "stop_type": "eos",
                        "stopping_word": "<|im_end|>",
                        "timings": {
                            "cache_n": type(self).cache_n,
                            "prompt_n": 12,
                            "prompt_ms": 2.0,
                            "predicted_n": 1,
                            "predicted_ms": 3.0,
                        },
                    }
                )
                return
            self.send_error(404)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    return server, Handler


def test_protocol_and_server_artifact_identity_are_frozen(tmp_path: Path) -> None:
    protocol = LlamaServerProtocol()
    model_path = tmp_path / "model.gguf"
    assert protocol.completion_request("prompt")["cache_prompt"] is False
    assert protocol.launch_arguments(model_path, 18080) == (
        "--model",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        "18080",
        "--ctx-size",
        "2048",
        "--cache-type-k",
        "q4_0",
        "--cache-type-v",
        "q4_0",
        "--gpu-layers",
        "all",
        "--fit",
        "off",
        "--threads",
        "8",
        "--threads-batch",
        "8",
        "--batch-size",
        "512",
        "--ubatch-size",
        "128",
        "--parallel",
        "1",
        "--seed",
        "17",
        "--reasoning",
        "off",
        "--reasoning-format",
        "none",
        "--cache-ram",
        "0",
        "--metrics",
    )
    for field, value in (
        ("context_size", 2048.0),
        ("threads", 8.0),
        ("parallel_slots", True),
        ("context_size", 4096),
    ):
        with pytest.raises(ConfigurationError):
            LlamaServerProtocol(**{field: value})  # type: ignore[arg-type]

    build = tmp_path / "build"
    build.mkdir()
    binary = build / "llama-server"
    _fake_binary(binary)
    source, revision, repository = _source_checkout(tmp_path / "source")
    expected = _expected_identity(binary, "template", source, revision, repository)
    observed = verify_llama_server_artifacts(binary, expected)
    assert observed["binary_sha256"] == expected.binary_sha256
    assert observed["source_checkout"]["source_revision"] == revision

    (build / "unexpected").write_text("changed", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="build tree differs"):
        verify_llama_server_artifacts(binary, expected)


def test_strict_client_validates_identity_template_decode_and_zero_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = (tmp_path / "model.gguf").resolve()
    model.write_bytes(b"model")
    build = tmp_path / "build"
    build.mkdir()
    binary = build / "llama-server"
    _fake_binary(binary)
    source, revision, repository = _source_checkout(tmp_path / "source")
    protocol = LlamaServerProtocol()
    template = "frozen template"
    expected = _expected_identity(binary, template, source, revision, repository)
    server, handler = _server(model_path=model, protocol=protocol, template=template)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
        monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:9")
        monkeypatch.setenv("NO_PROXY", "")
        client = LlamaServerClient(port=server.server_port, protocol=protocol)
        assert client.is_healthy()
        observed = client.observed_identity(
            model_path=model,
            model_alias=model.name,
            expected=expected,
        )
        assert observed["chat_template_stable_hash"] == stable_hash(template)
        prompt = client.render_prompt(
            system_prompt="Answer factual questions.", question="Capital of France?"
        )
        completion = client.complete(prompt, expected_model_alias=model.name)
        assert completion.content == "Paris"
        assert completion.token_ids == (123,)
        assert completion.cache_n == 0

        handler.cache_n = 1
        with pytest.raises(DataValidationError, match="response contract"):
            client.complete(prompt, expected_model_alias=model.name)
        handler.numeric_forgery = True
        with pytest.raises(DataValidationError, match="must be an integer"):
            client.observed_identity(
                model_path=model,
                model_alias=model.name,
                expected=expected,
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _managed_server_binary(path: Path) -> None:
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer\n"
        "if '--version' in sys.argv:\n"
        "    print('version: test-server')\n"
        "    raise SystemExit(0)\n"
        "port = int(sys.argv[sys.argv.index('--port') + 1])\n"
        "model = sys.argv[sys.argv.index('--model') + 1]\n"
        "class Handler(BaseHTTPRequestHandler):\n"
        "    def log_message(self, *args): pass\n"
        "    def send_json(self, value):\n"
        "        body = json.dumps(value).encode()\n"
        "        self.send_response(200); self.send_header('Content-Length', str(len(body)))\n"
        "        self.end_headers(); self.wfile.write(body)\n"
        "    def do_GET(self):\n"
        "        if self.path == '/health': self.send_json({'status': 'ok'}); return\n"
        "        if self.path == '/slots':\n"
        "            self.send_json([{'id': 0, 'n_ctx': 2048, 'speculative': False, "
        "'is_processing': False}]); return\n"
        "        if self.path == '/props':\n"
        "            self.send_json({'default_generation_settings': {'params': {'seed': 17}, "
        "'n_ctx': 2048}, 'total_slots': 1, 'model_alias': model.rsplit('/', 1)[-1], "
        "'model_path': model, 'modalities': {'vision': False, 'audio': False}, "
        "'chat_template': 'managed-template', 'chat_template_caps': {}, "
        "'build_info': 'test-build', 'is_sleeping': False}); return\n"
        "        self.send_error(404)\n"
        "ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_managed_server_lifecycle_and_post_run_model_attestation(tmp_path: Path) -> None:
    build = tmp_path / "build"
    build.mkdir()
    binary = build / "llama-server"
    _managed_server_binary(binary)
    source, revision, repository = _source_checkout(tmp_path / "source")
    expected = _expected_identity(binary, "managed-template", source, revision, repository)
    model = (tmp_path / "model.gguf").resolve()
    model.write_bytes(b"model-bytes")
    protocol = LlamaServerProtocol(startup_timeout_seconds=10)
    probe = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    port = probe.server_port
    probe.server_close()
    managed = ManagedLlamaServer(
        binary_path=binary,
        model_path=model,
        log_path=tmp_path / "server.log",
        expected_identity=expected,
        expected_model_sha256=sha256_file(model),
        expected_model_size_bytes=model.stat().st_size,
        protocol=protocol,
        port=port,
        memory_sampler=lambda _pid: 1234,
    )
    managed.start()
    assert managed.client.is_healthy()
    assert managed.sample_memory() == 1234
    managed.stop()
    assert managed.process is not None and managed.process.poll() is not None

    second = ManagedLlamaServer(
        binary_path=binary,
        model_path=model,
        log_path=tmp_path / "server-2.log",
        expected_identity=expected,
        expected_model_sha256=sha256_file(model),
        expected_model_size_bytes=model.stat().st_size,
        protocol=protocol,
        port=port,
        memory_sampler=lambda _pid: 1234,
    )
    second.start()
    model.write_bytes(b"changed-model-bytes")
    with pytest.raises(DataValidationError, match="model artifact changed"):
        second.stop()
    assert second.process is not None and second.process.poll() is not None


def test_identity_loader_rejects_non_string_yaml_fields(tmp_path: Path) -> None:
    path = tmp_path / "identity.yaml"
    path.write_text(
        "schema_version: 1\n"
        "llama_server:\n"
        "  source_repository: 123\n"
        f"  source_revision: {'a' * 40}\n"
        "  source_path: source\n"
        "  binary_path: binary\n"
        f"  binary_sha256: {'b' * 64}\n"
        f"  build_tree_sha256: {'c' * 64}\n"
        f"  build_tree_layout_sha256: {'d' * 64}\n"
        f"  version_digest: {'e' * 64}\n"
        "  build_info: true\n"
        f"  chat_template_stable_hash: {'f' * 64}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="fields differ"):
        load_llama_server_identity(path)


def test_runtime_tree_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "build"
    root.mkdir()
    (root / "binary").write_bytes(b"binary")
    external = tmp_path / "external.dylib"
    external.write_bytes(b"library")
    (root / "library.dylib").symlink_to(external)
    with pytest.raises(DataValidationError, match="contained regular files"):
        sha256_runtime_tree(root)


def test_identity_rejects_ancestor_directory_symlink(tmp_path: Path) -> None:
    external = tmp_path / "external-build"
    external.mkdir()
    binary = external / "llama-server"
    _fake_binary(binary)
    declared = tmp_path / "declared-build"
    declared.symlink_to(external, target_is_directory=True)
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ConfigurationError, match="binary path must be a regular file"):
        LlamaServerExpectedIdentity(
            source_repository="https://example.test/llama.cpp",
            source_revision="a" * 40,
            source_path=source,
            binary_path=declared / "llama-server",
            binary_sha256="b" * 64,
            build_tree_sha256="c" * 64,
            build_tree_layout_sha256="d" * 64,
            version_digest="e" * 64,
            build_info="test-build",
            chat_template_stable_hash="f" * 64,
        )

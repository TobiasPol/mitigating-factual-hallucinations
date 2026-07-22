"""Pinned adapter for Google's released IFEval instruction checker."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mfh.artifact_namespace import validate_active_study_artifact_paths
from mfh.contracts import Question
from mfh.errors import DataValidationError, FrozenArtifactError, OptionalDependencyError
from mfh.provenance import sha256_file, sha256_path, stable_hash

_REPOSITORY = "google-research/google-research"
_REVISION = "5b09c22d73a9d35eb6c5d2a99b95677a45053466"
_FILES = {
    "evaluation_lib.py": "01deab4c67bf7f30c3a48f59d7b0bb31ea165651a636af7f2a3af389a420edbb",
    "instructions.py": "130f9c50e15ae44820c9ef5b4aa2aa948c4c0a17f4c44c2932b9271add22c6d7",
    "instructions_registry.py": (
        "ec92d72c264f6d906978613085db262356174300370a3fffe6fefd5969ce9cfc"
    ),
    "instructions_util.py": "a73797261eee5bf447e279d82a2b700b1bdd3cb1193412dbab1270a85832bc6b",
}
_PACKAGE = "instruction_following_eval"
_NLTK_DATA_REPOSITORY = "nltk/nltk_data"
_NLTK_DATA_REVISION = "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a"
_PUNKT_TAB_SHA256 = "e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106"
_PUNKT_TAB_DATA_SHA256 = "78ea3406355ecf4456f100fc1f571bfe80883df845adbf4076acc75df31f02c2"
_UV_VERSION = "0.11.28"
_UV_EXECUTABLE_SHA256 = "1cb9cd0a1749debf6049d7d2bb933882cc52d81016326ee6d99a786d6c988b03"
_PYTHON_VERSION = "3.11.14"
_PYTHON_RUNTIME_SOURCE = (
    "~/.local/share/mfh/cpython-3.11.14+20260211-x86_64-unknown-linux-gnu-stripped"
)
_PYTHON_EXECUTABLE = "python-runtime/bin/python3.11"
_PYTHON_EXECUTABLE_SHA256 = (
    "6ff97f602038740073dca96714310a30e303332326268e0f1bb2767edc820944"
)
_PYTHON_RUNTIME_SOURCE_SHA256 = (
    "0fe9b60adc070445a70efa17a27ba85a5a8865b40f34b2b42426de9c4ea673eb"
)
_PYTHON_RUNTIME_SHA256 = (
    "0fe9b60adc070445a70efa17a27ba85a5a8865b40f34b2b42426de9c4ea673eb"
)
_PLATFORM_SYSTEM = "Linux"
_PLATFORM_MACHINE = "x86_64"
_RUNTIME_PLATFORM = "linux-x86_64"
_REQUIREMENT_VERSIONS = {
    "absl-py": "2.3.1",
    "click": "8.4.2",
    "immutabledict": "4.2.1",
    "joblib": "1.5.3",
    "langdetect": "1.0.9",
    "nltk": "3.9.2",
    "regex": "2026.7.10",
    "six": "1.17.0",
    "tqdm": "4.68.4",
}
_REQUIREMENT_HASHES = {
    "absl-py": (
        "eeecf07f0c2a93ace0772c92e596ace6d3d3996c042b2128459aaae2a76de11d",
    ),
    "click": (
        "e6f9f66136c816745b9d65817da91d61d957fb16e02e4dcd0552553c5a197b76",
    ),
    "immutabledict": (
        "c56a26ced38c236f79e74af3ccce53772827cef5c3bce7cab33ff2060f756373",
    ),
    "joblib": (
        "5fc3c5039fc5ca8c0276333a188bbd59d6b7ab37fe6632daa76bc7f9ec18e713",
    ),
    "langdetect": (
        "7cbc0746252f19e76f77c0b1690aadf01963be835ef0cd4b56dddf2a8f1dfc2a",
        "cbc1fef89f8d062739774bd51eda3da3274006b3661d199c2655f6b3f6d605a0",
    ),
    "nltk": (
        "1e209d2b3009110635ed9709a67a1a3e33a10f799490fa71cf4bec218c11c88a",
    ),
    "regex": (
        "724ee9379568658ec06362cf24325c5315cc5a67f61dfe585bfeff58300a355b",
    ),
    "six": (
        "4721f391ed90541fddacab5acf947aa0d3dc7d27b2e1e8eda2be8970586c3274",
    ),
    "tqdm": (
        "5168118b2368f48c561afda8020fd79195b1bdb0bdf8086b88442c267a315dc2",
    ),
}
_REQUIREMENTS = "".join(
    f"{name}=={version} "
    + " ".join(
        f"--hash=sha256:{digest}" for digest in _REQUIREMENT_HASHES[name]
    )
    + "\n"
    for name, version in _REQUIREMENT_VERSIONS.items()
)
_RUNTIME_PACKAGES_SHA256 = (
    "7a42d67ca29c955d3ecf582c958600c10b314c1095ca1db1f33040241d16534c"
)
_PACKAGE_INIT_SOURCE = '"""Frozen Google Research IFEval evaluator."""\n'
_PACKAGE_INIT_SHA256 = "cac369e92ab8de249ed31b2f837884bb8b235acfe6085fef7236e722a616784d"
_ISOLATED_LAUNCHER = (
    "import runpy,sys; root=sys.argv[1]; "
    "sys.path[:0]=[root+'/runtime-packages',root]; "
    "runpy.run_path(root+'/evaluate.py',run_name='__main__')"
)
_CHECKER_ERRATA = {
    "ifeval:1122": {
        "instruction_index": 1,
        "instruction_id": "keywords:letter_frequency",
        "letter": "#",
        "reason": "released checker randomizes non-alphabetic letter arguments",
    },
    "ifeval:1129": {
        "instruction_index": 0,
        "instruction_id": "keywords:letter_frequency",
        "letter": "!",
        "reason": "released checker randomizes non-alphabetic letter arguments",
    },
}
_SIDECAR_SOURCE = '''"""Isolated entry point for the frozen IFEval checker."""
from __future__ import annotations

import json
import os
import platform
import sys

from instruction_following_eval import evaluation_lib

ERRATA = {
    "ifeval:1122": (1, "keywords:letter_frequency", "#"),
    "ifeval:1129": (0, "keywords:letter_frequency", "!"),
}


def _literal_frequency_check(response: str, kwargs: dict[str, object]) -> bool:
    letter = kwargs.get("letter")
    frequency = kwargs.get("let_frequency")
    relation = kwargs.get("let_relation")
    if not isinstance(letter, str) or len(letter) != 1 or not isinstance(frequency, int):
        raise ValueError("IFEval erratum arguments differ from the frozen disclosure")
    count = response.lower().count(letter.lower())
    if relation == "at least":
        return count >= frequency
    if relation == "less than":
        return count < frequency
    raise ValueError("IFEval erratum relation differs from the released schema")


def main() -> int:
    value = json.load(sys.stdin)
    expected = {"question_id", "prompt", "instruction_id_list", "kwargs", "response"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("IFEval sidecar input schema differs")
    os.environ["NLTK_DATA"] = os.path.join(os.path.dirname(__file__), "nltk_data")
    question_id = str(value["question_id"])
    instruction_ids = list(value["instruction_id_list"])
    kwargs = [dict(item) for item in value["kwargs"]]
    erratum = ERRATA.get(question_id)
    if erratum is not None:
        index, instruction_id, letter = erratum
        if (
            instruction_ids[index] != instruction_id
            or kwargs[index].get("letter") != letter
        ):
            raise ValueError("IFEval erratum row differs from the frozen disclosure")
        # The released constructor chooses a random ASCII letter for punctuation.
        # Give it a deterministic valid placeholder, then replace only that check
        # with the intended literal-frequency semantics below.
        kwargs[index]["letter"] = "a"
    example = evaluation_lib.InputExample(
        key=int(str(value["question_id"]).rsplit(":", 1)[-1]),
        instruction_id_list=instruction_ids,
        prompt=str(value["prompt"]),
        kwargs=kwargs,
    )
    result = evaluation_lib.test_instruction_following_strict(
        example, {str(value["prompt"]): str(value["response"])}
    )
    checks = [bool(item) for item in result.follow_instruction_list]
    if erratum is not None:
        index, _, _ = erratum
        checks[index] = _literal_frequency_check(
            str(value["response"]), dict(value["kwargs"][index])
        )
    json.dump(
        {
            "passed": all(checks),
            "checks": checks,
            "python_version": platform.python_version(),
            "platform_system": platform.system(),
            "platform_machine": platform.machine(),
        },
        sys.stdout,
        sort_keys=True,
    )
    sys.stdout.write("\\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _sanitized_runtime_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(
            ("UV_", "PIP_", "PYTHON", "VIRTUAL_ENV", "CONDA", "DYLD_", "LD_")
        )
    }
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _frozen_python_runtime_source() -> Path:
    runtime = Path(_PYTHON_RUNTIME_SOURCE).expanduser().resolve()
    if (
        platform.system() != _PLATFORM_SYSTEM
        or platform.machine() != _PLATFORM_MACHINE
        or not runtime.is_dir()
        or runtime.is_symlink()
        or sha256_path(runtime) != _PYTHON_RUNTIME_SOURCE_SHA256
    ):
        raise OptionalDependencyError(
            "IFEval requires the exact frozen relocatable Linux-x86_64 Python runtime"
        )
    return runtime


def _frozen_python_executable(runtime: str | Path | None = None) -> Path:
    root = _frozen_python_runtime_source() if runtime is None else Path(runtime).resolve()
    executable = root / "bin" / "python3.11"
    if not executable.is_file() or sha256_file(executable) != _PYTHON_EXECUTABLE_SHA256:
        raise OptionalDependencyError("IFEval Python executable differs from the frozen runtime")
    return executable


def _frozen_uv_executable() -> Path:
    raw = shutil.which("uv")
    if raw is None:
        raise OptionalDependencyError("IFEval materialization requires frozen uv")
    executable = Path(raw).resolve()
    try:
        process = subprocess.run(
            [str(executable), "--version"],
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
            env=_sanitized_runtime_environment(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise OptionalDependencyError("cannot verify the frozen uv runtime") from exc
    version_fields = process.stdout.strip().split(maxsplit=2)
    if (
        process.returncode != 0
        or version_fields[:2] != ["uv", _UV_VERSION]
        or sha256_file(executable) != _UV_EXECUTABLE_SHA256
    ):
        raise OptionalDependencyError("IFEval uv executable differs from the frozen runtime")
    return executable


def materialize_ifeval_evaluator(directory: str | Path) -> str:
    """Download the exact released checker source and freeze every source byte."""

    destination = validate_active_study_artifact_paths(
        {"IFEval evaluator": directory}
    )["IFEval evaluator"]
    if destination.exists() or destination.is_symlink():
        raise FrozenArtifactError(f"refusing to overwrite IFEval evaluator: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=destination.parent)
    )
    try:
        package = stage / _PACKAGE
        package.mkdir()
        (package / "__init__.py").write_text(_PACKAGE_INIT_SOURCE, encoding="utf-8")
        for name, expected_sha in _FILES.items():
            url = (
                "https://raw.githubusercontent.com/"
                f"{_REPOSITORY}/{_REVISION}/instruction_following_eval/{name}"
            )
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    payload = response.read()
            except (OSError, urllib.error.URLError) as exc:
                raise FrozenArtifactError(
                    f"cannot download pinned IFEval evaluator file {name}: {exc}"
                ) from exc
            path = package / name
            path.write_bytes(payload)
            if sha256_file(path) != expected_sha:
                raise FrozenArtifactError(
                    f"downloaded IFEval evaluator file changed: {name}"
                )
        python_runtime = stage / "python-runtime"
        shutil.copytree(_frozen_python_runtime_source(), python_runtime)
        if (
            any(value.is_symlink() for value in python_runtime.rglob("*"))
            or sha256_path(python_runtime) != _PYTHON_RUNTIME_SHA256
        ):
            raise FrozenArtifactError("copied IFEval Python runtime changed")
        requirements = stage / "requirements.lock.txt"
        requirements.write_text(_REQUIREMENTS, encoding="utf-8")
        uv = _frozen_uv_executable()
        python_executable = _frozen_python_executable(python_runtime)
        runtime_packages = stage / "runtime-packages"
        try:
            install = subprocess.run(
                [
                    str(uv),
                    "pip",
                    "install",
                    "--target",
                    str(runtime_packages),
                    "--require-hashes",
                    "--requirements",
                    str(requirements),
                    "--python",
                    str(python_executable),
                    "--python-platform",
                    "x86_64-unknown-linux-gnu",
                    "--default-index",
                    "https://pypi.org/simple",
                    "--keyring-provider",
                    "disabled",
                    "--no-config",
                    "--no-cache",
                ],
                text=True,
                capture_output=True,
                check=False,
                timeout=300,
                env=_sanitized_runtime_environment(),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise FrozenArtifactError(
                f"cannot materialize hash-enforced IFEval dependencies: {exc}"
            ) from exc
        if install.returncode != 0:
            raise FrozenArtifactError(
                "cannot materialize hash-enforced IFEval dependencies: "
                + install.stderr.strip()[-1000:]
            )
        # Target installs generate console scripts whose shebang embeds the
        # random staging path. The evaluator imports these distributions and
        # never invokes their command-line wrappers, so exclude that unused,
        # path-dependent directory from the frozen runtime.
        generated_scripts = runtime_packages / "bin"
        if generated_scripts.is_dir() and not generated_scripts.is_symlink():
            shutil.rmtree(generated_scripts)
        for record in runtime_packages.glob("*.dist-info/RECORD"):
            lines = record.read_text(encoding="utf-8").splitlines()
            normalized = [line for line in lines if not line.startswith("bin/")]
            record.write_text("\n".join(normalized) + "\n", encoding="utf-8")
        runtime_packages_sha = sha256_path(runtime_packages)
        if runtime_packages_sha != _RUNTIME_PACKAGES_SHA256:
            raise FrozenArtifactError(
                "materialized IFEval runtime packages changed: "
                f"{runtime_packages_sha}"
            )
        sidecar = stage / "evaluate.py"
        sidecar.write_text(_SIDECAR_SOURCE, encoding="utf-8")
        punkt_zip = stage / "punkt_tab.zip"
        punkt_url = (
            "https://raw.githubusercontent.com/"
            f"{_NLTK_DATA_REPOSITORY}/{_NLTK_DATA_REVISION}/"
            "packages/tokenizers/punkt_tab.zip"
        )
        try:
            with urllib.request.urlopen(punkt_url, timeout=60) as response:
                punkt_zip.write_bytes(response.read())
        except (OSError, urllib.error.URLError) as exc:
            raise FrozenArtifactError(
                f"cannot download pinned NLTK punkt_tab data: {exc}"
            ) from exc
        if sha256_file(punkt_zip) != _PUNKT_TAB_SHA256:
            raise FrozenArtifactError("downloaded NLTK punkt_tab data changed")
        nltk_tokenizers = stage / "nltk_data" / "tokenizers"
        nltk_tokenizers.mkdir(parents=True)
        try:
            with zipfile.ZipFile(punkt_zip) as archive:
                members = archive.infolist()
                if any(
                    Path(member.filename).is_absolute()
                    or ".." in Path(member.filename).parts
                    for member in members
                ):
                    raise FrozenArtifactError("NLTK punkt_tab archive has unsafe paths")
                archive.extractall(nltk_tokenizers)
        except zipfile.BadZipFile as exc:
            raise FrozenArtifactError("NLTK punkt_tab archive is invalid") from exc
        punkt_zip.unlink()
        punkt_data_sha = sha256_path(nltk_tokenizers / "punkt_tab")
        if punkt_data_sha != _PUNKT_TAB_DATA_SHA256:
            raise FrozenArtifactError("extracted NLTK punkt_tab data changed")
        body = {
            "schema_version": 8,
            "repository": _REPOSITORY,
            "revision": _REVISION,
            "license": "Apache-2.0",
            "package": _PACKAGE,
            "runtime_platform": _RUNTIME_PLATFORM,
            "uv_version": _UV_VERSION,
            "uv_executable_sha256": _UV_EXECUTABLE_SHA256,
            "python_version": _PYTHON_VERSION,
            "python_executable": _PYTHON_EXECUTABLE,
            "python_executable_sha256": _PYTHON_EXECUTABLE_SHA256,
            "python_runtime_sha256": _PYTHON_RUNTIME_SHA256,
            "runtime_requirements": dict(_REQUIREMENT_VERSIONS),
            "files": dict(_FILES),
            "package_sha256": sha256_path(package),
            "requirements_sha256": sha256_file(requirements),
            "runtime_packages_sha256": _RUNTIME_PACKAGES_SHA256,
            "sidecar_sha256": sha256_file(sidecar),
            "nltk_data_repository": _NLTK_DATA_REPOSITORY,
            "nltk_data_revision": _NLTK_DATA_REVISION,
            "punkt_tab_archive_sha256": _PUNKT_TAB_SHA256,
            "punkt_tab_data_sha256": punkt_data_sha,
            "checker_errata": _CHECKER_ERRATA,
        }
        (stage / "manifest.json").write_text(
            json.dumps(
                {**body, "manifest_digest": stable_hash(body)},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        validate_ifeval_evaluator(stage)
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return sha256_path(destination)


def validate_ifeval_evaluator(directory: str | Path) -> str:
    source = Path(directory)
    package = source / _PACKAGE
    expected_package_files = {*_FILES, "__init__.py"}
    if (
        source.is_symlink()
        or not source.is_dir()
        or {value.name for value in source.iterdir()}
        != {
            "manifest.json",
            _PACKAGE,
            "requirements.lock.txt",
            "runtime-packages",
            "evaluate.py",
            "nltk_data",
            "python-runtime",
        }
        or package.is_symlink()
        or not package.is_dir()
        or {value.name for value in package.iterdir()} != expected_package_files
        or any(value.is_symlink() for value in source.rglob("*"))
        or any(value.is_dir() and not any(value.iterdir()) for value in source.rglob("*"))
    ):
        raise FrozenArtifactError("IFEval evaluator inventory differs")
    try:
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrozenArtifactError(f"cannot read IFEval evaluator manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise FrozenArtifactError("IFEval evaluator manifest is invalid")
    digest = manifest.pop("manifest_digest", None)
    expected = {
        "schema_version": 8,
        "repository": _REPOSITORY,
        "revision": _REVISION,
        "license": "Apache-2.0",
        "package": _PACKAGE,
        "runtime_platform": _RUNTIME_PLATFORM,
        "uv_version": _UV_VERSION,
        "uv_executable_sha256": _UV_EXECUTABLE_SHA256,
        "python_version": _PYTHON_VERSION,
        "python_executable": _PYTHON_EXECUTABLE,
        "python_executable_sha256": _PYTHON_EXECUTABLE_SHA256,
        "python_runtime_sha256": _PYTHON_RUNTIME_SHA256,
        "runtime_requirements": dict(_REQUIREMENT_VERSIONS),
        "files": dict(_FILES),
        "package_sha256": sha256_path(package),
        "requirements_sha256": sha256_file(source / "requirements.lock.txt"),
        "runtime_packages_sha256": _RUNTIME_PACKAGES_SHA256,
        "sidecar_sha256": sha256_file(source / "evaluate.py"),
        "nltk_data_repository": _NLTK_DATA_REPOSITORY,
        "nltk_data_revision": _NLTK_DATA_REVISION,
        "punkt_tab_archive_sha256": _PUNKT_TAB_SHA256,
        "punkt_tab_data_sha256": _PUNKT_TAB_DATA_SHA256,
        "checker_errata": _CHECKER_ERRATA,
    }
    if (
        manifest != expected
        or digest != stable_hash(expected)
        or (source / "requirements.lock.txt").read_text(encoding="utf-8")
        != _REQUIREMENTS
        or (source / "evaluate.py").read_text(encoding="utf-8")
        != _SIDECAR_SOURCE
        or (package / "__init__.py").read_text(encoding="utf-8")
        != _PACKAGE_INIT_SOURCE
        or sha256_file(package / "__init__.py") != _PACKAGE_INIT_SHA256
        or sha256_path(source / "nltk_data" / "tokenizers" / "punkt_tab")
        != _PUNKT_TAB_DATA_SHA256
        or sha256_path(source / "runtime-packages") != _RUNTIME_PACKAGES_SHA256
        or sha256_path(source / "python-runtime") != _PYTHON_RUNTIME_SHA256
        or any(sha256_file(package / name) != value for name, value in _FILES.items())
    ):
        raise FrozenArtifactError("IFEval evaluator source identity differs")
    return sha256_path(source)


def evaluate_ifeval_strict(
    question: Question,
    response: str,
    *,
    evaluator_directory: str | Path,
) -> tuple[bool, tuple[bool, ...]]:
    """Run Google's checker in an automatically isolated, lock-exact process."""

    requested_source = Path(evaluator_directory)
    validate_ifeval_evaluator(requested_source)
    source = requested_source.resolve()
    instruction_ids = question.metadata.get("instruction_id_list")
    kwargs = question.metadata.get("kwargs")
    if (
        question.benchmark != "ifeval"
        or not isinstance(instruction_ids, list)
        or not instruction_ids
        or any(type(value) is not str or not value for value in instruction_ids)
        or not isinstance(kwargs, list)
        or len(kwargs) != len(instruction_ids)
        or any(not isinstance(value, Mapping) for value in kwargs)
    ):
        raise DataValidationError("IFEval question lacks its released checker arguments")
    python_executable = _frozen_python_executable(source / "python-runtime")
    payload = {
        "question_id": question.question_id,
        "prompt": question.text,
        "instruction_id_list": list(instruction_ids),
        "kwargs": [dict(value) for value in kwargs],
        "response": response,
    }
    environment = _sanitized_runtime_environment()
    environment["NLTK_DATA"] = str(source / "nltk_data")
    try:
        process = subprocess.run(
            [
                str(python_executable),
                "-I",
                "-S",
                "-B",
                "-c",
                _ISOLATED_LAUNCHER,
                str(source),
            ],
            input=json.dumps(payload, sort_keys=True),
            text=True,
            capture_output=True,
            check=False,
            timeout=180,
            cwd=source,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise OptionalDependencyError(
            f"isolated IFEval process could not run: {exc}"
        ) from exc
    if process.returncode != 0:
        detail = process.stderr.strip()[-1000:]
        raise OptionalDependencyError(
            f"isolated IFEval process failed with exit {process.returncode}: {detail}"
        )
    try:
        result: Any = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise DataValidationError("isolated IFEval process returned invalid JSON") from exc
    if (
        not isinstance(result, Mapping)
        or set(result)
        != {
            "passed",
            "checks",
            "python_version",
            "platform_system",
            "platform_machine",
        }
        or type(result["passed"]) is not bool
        or not isinstance(result["checks"], list)
        or len(result["checks"]) != len(instruction_ids)
        or any(type(value) is not bool for value in result["checks"])
        or result["python_version"] != _PYTHON_VERSION
        or result["platform_system"] != _PLATFORM_SYSTEM
        or result["platform_machine"] != _PLATFORM_MACHINE
    ):
        raise DataValidationError("official IFEval checker returned incomplete results")
    return result["passed"], tuple(result["checks"])

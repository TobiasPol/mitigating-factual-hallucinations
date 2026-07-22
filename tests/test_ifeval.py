from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mfh.contracts import Question
from mfh.errors import DataValidationError, FrozenArtifactError
from mfh.evaluation import ifeval as ifeval_module
from mfh.evaluation.ifeval import (
    _PYTHON_EXECUTABLE_SHA256,
    _UV_EXECUTABLE_SHA256,
    evaluate_ifeval_strict,
)
from mfh.provenance import sha256_file, sha256_path, stable_hash


class IFEvalIsolationTests(unittest.TestCase):
    @staticmethod
    def _question() -> Question:
        return Question(
            question_id="ifeval:17",
            benchmark="ifeval",
            text="Write exactly two sentences.",
            aliases=("released-checker",),
            metadata={
                "instruction_id_list": ["length_constraints:number_sentences"],
                "kwargs": [{"num_sentences": 2}],
            },
        )

    def test_official_checker_always_runs_in_locked_isolated_process(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                '{"checks": [true], "passed": true, '
                f'"python_version": "{ifeval_module._PYTHON_VERSION}", '
                f'"platform_system": "{ifeval_module._PLATFORM_SYSTEM}", '
                f'"platform_machine": "{ifeval_module._PLATFORM_MACHINE}"}}\n'
            ),
            stderr="",
        )
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "mfh.evaluation.ifeval.os.environ", {"LD_PRELOAD": "attacker.so"}, clear=False
        ), patch(
            "mfh.evaluation.ifeval.validate_ifeval_evaluator",
            return_value="a" * 64,
        ), patch(
            "mfh.evaluation.ifeval._frozen_python_executable",
            return_value=Path("/usr/bin/python3.12"),
        ), patch(
            "mfh.evaluation.ifeval.subprocess.run", return_value=completed
        ) as run, patch(
            "mfh.evaluation.ifeval.sha256_file",
            side_effect=lambda path: (
                _UV_EXECUTABLE_SHA256
                if str(path).endswith("uv")
                else _PYTHON_EXECUTABLE_SHA256
            ),
        ):
            result = evaluate_ifeval_strict(
                self._question(),
                "First. Second.",
                evaluator_directory=directory,
            )
        self.assertEqual(result, (True, (True,)))
        command = run.call_args_list[-1].args[0]
        self.assertEqual(
            command[:5], ["/usr/bin/python3.12", "-I", "-S", "-B", "-c"]
        )
        self.assertEqual(command[-1], str(Path(directory).resolve()))
        payload = json.loads(run.call_args.kwargs["input"])
        self.assertEqual(payload["response"], "First. Second.")
        self.assertEqual(
            run.call_args.kwargs["cwd"], Path(directory).resolve()
        )
        self.assertEqual(
            run.call_args.kwargs["env"]["PYTHONDONTWRITEBYTECODE"], "1"
        )
        self.assertNotIn("VIRTUAL_ENV", run.call_args.kwargs["env"])
        self.assertNotIn("LD_PRELOAD", run.call_args.kwargs["env"])

    def test_rejects_incomplete_isolated_checker_output(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                '{"checks": [], "passed": true, '
                f'"python_version": "{ifeval_module._PYTHON_VERSION}", '
                f'"platform_system": "{ifeval_module._PLATFORM_SYSTEM}", '
                f'"platform_machine": "{ifeval_module._PLATFORM_MACHINE}"}}\n'
            ),
            stderr="",
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "mfh.evaluation.ifeval.validate_ifeval_evaluator",
            return_value="a" * 64,
        ), patch(
            "mfh.evaluation.ifeval._frozen_python_executable",
            return_value=Path("/usr/bin/python3.12"),
        ), patch(
            "mfh.evaluation.ifeval.subprocess.run", return_value=completed
        ), patch(
            "mfh.evaluation.ifeval.sha256_file",
            side_effect=lambda path: (
                _UV_EXECUTABLE_SHA256
                if str(path).endswith("uv")
                else _PYTHON_EXECUTABLE_SHA256
            ),
        ), self.assertRaises(DataValidationError):
            evaluate_ifeval_strict(
                self._question(),
                "First. Second.",
                evaluator_directory=directory,
            )

    def test_rejects_rehashed_executable_and_punkt_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "evaluator"
            package = source / ifeval_module._PACKAGE
            punkt = source / "nltk_data" / "tokenizers" / "punkt_tab"
            runtime_packages = source / "runtime-packages"
            python_runtime = source / "python-runtime"
            package.mkdir(parents=True)
            punkt.mkdir(parents=True)
            runtime_packages.mkdir()
            python_runtime.mkdir()
            (runtime_packages / "frozen.py").write_text(
                "# frozen dependency tree\n", encoding="utf-8"
            )
            (python_runtime / "frozen-python").write_text(
                "frozen runtime\n", encoding="utf-8"
            )
            (package / "__init__.py").write_text(
                ifeval_module._PACKAGE_INIT_SOURCE, encoding="utf-8"
            )
            for name in ifeval_module._FILES:
                (package / name).write_text(f"# frozen {name}\n", encoding="utf-8")
            requirements = source / "requirements.lock.txt"
            requirements.write_text(ifeval_module._REQUIREMENTS, encoding="utf-8")
            sidecar = source / "evaluate.py"
            sidecar.write_text(ifeval_module._SIDECAR_SOURCE, encoding="utf-8")
            punkt_file = punkt / "english.tab"
            punkt_file.write_text("frozen punkt\n", encoding="utf-8")
            frozen_files = {
                name: sha256_file(package / name) for name in ifeval_module._FILES
            }
            frozen_punkt_sha = sha256_path(punkt)
            frozen_runtime_sha = sha256_path(runtime_packages)
            frozen_python_runtime_sha = sha256_path(python_runtime)

            def write_manifest(*, punkt_sha: str = frozen_punkt_sha) -> None:
                body = {
                    "schema_version": 8,
                    "repository": ifeval_module._REPOSITORY,
                    "revision": ifeval_module._REVISION,
                    "license": "Apache-2.0",
                    "package": ifeval_module._PACKAGE,
                    "runtime_platform": ifeval_module._RUNTIME_PLATFORM,
                    "uv_version": ifeval_module._UV_VERSION,
                    "uv_executable_sha256": ifeval_module._UV_EXECUTABLE_SHA256,
                    "python_version": ifeval_module._PYTHON_VERSION,
                    "python_executable": ifeval_module._PYTHON_EXECUTABLE,
                    "python_executable_sha256": (
                        ifeval_module._PYTHON_EXECUTABLE_SHA256
                    ),
                    "python_runtime_sha256": frozen_python_runtime_sha,
                    "runtime_requirements": dict(
                        ifeval_module._REQUIREMENT_VERSIONS
                    ),
                    "files": frozen_files,
                    "package_sha256": sha256_path(package),
                    "requirements_sha256": sha256_file(requirements),
                    "runtime_packages_sha256": frozen_runtime_sha,
                    "sidecar_sha256": sha256_file(sidecar),
                    "nltk_data_repository": ifeval_module._NLTK_DATA_REPOSITORY,
                    "nltk_data_revision": ifeval_module._NLTK_DATA_REVISION,
                    "punkt_tab_archive_sha256": ifeval_module._PUNKT_TAB_SHA256,
                    "punkt_tab_data_sha256": punkt_sha,
                    "checker_errata": ifeval_module._CHECKER_ERRATA,
                }
                (source / "manifest.json").write_text(
                    json.dumps(
                        {**body, "manifest_digest": stable_hash(body)},
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )

            with patch.object(ifeval_module, "_FILES", frozen_files), patch.object(
                ifeval_module, "_PUNKT_TAB_DATA_SHA256", frozen_punkt_sha
            ), patch.object(
                ifeval_module, "_RUNTIME_PACKAGES_SHA256", frozen_runtime_sha
            ), patch.object(
                ifeval_module, "_PYTHON_RUNTIME_SHA256", frozen_python_runtime_sha
            ):
                write_manifest()
                ifeval_module.validate_ifeval_evaluator(source)
                (package / "__init__.py").write_text(
                    "# attacker-controlled executable\n", encoding="utf-8"
                )
                write_manifest()
                with self.assertRaisesRegex(FrozenArtifactError, "source identity"):
                    ifeval_module.validate_ifeval_evaluator(source)

                (package / "__init__.py").write_text(
                    ifeval_module._PACKAGE_INIT_SOURCE, encoding="utf-8"
                )
                punkt_file.write_text("attacker-controlled punkt\n", encoding="utf-8")
                write_manifest(punkt_sha=sha256_path(punkt))
                with self.assertRaisesRegex(FrozenArtifactError, "source identity"):
                    ifeval_module.validate_ifeval_evaluator(source)


if __name__ == "__main__":
    unittest.main()

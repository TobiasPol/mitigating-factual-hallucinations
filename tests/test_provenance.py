from __future__ import annotations

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from mfh.errors import FrozenArtifactError
from mfh.provenance import RunManifest, read_frozen_manifest, write_frozen_manifest


class ManifestTests(unittest.TestCase):
    def test_manifest_is_content_verified_and_write_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = RunManifest.create(
                study="toy",
                phase="E0",
                config={"seed": 17},
                inputs={"questions": "a" * 64},
                cwd=Path(__file__).resolve().parents[1],
                created_at="2026-01-01T00:00:00+00:00",
            )
            path = root / "manifest.json"
            write_frozen_manifest(path, manifest)
            write_frozen_manifest(path, manifest)
            self.assertEqual(read_frozen_manifest(path), manifest)

            different = RunManifest.create(
                study="toy",
                phase="E1",
                config={"seed": 17},
                inputs={"questions": "a" * 64},
                cwd=Path(__file__).resolve().parents[1],
                created_at="2026-01-01T00:00:00+00:00",
            )
            with self.assertRaises(FrozenArtifactError):
                write_frozen_manifest(path, different)

    def test_tampered_manifest_mapping_is_rejected(self) -> None:
        manifest = RunManifest.create(
            study="toy",
            phase="E0",
            config={},
            inputs={},
            cwd=Path(__file__).resolve().parents[1],
            created_at="2026-01-01T00:00:00+00:00",
        )
        value = asdict(manifest)
        value["phase"] = "E10"
        with self.assertRaises(FrozenArtifactError):
            RunManifest.from_dict(value)

    def test_untracked_source_contents_change_manifest_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "src"
            source.mkdir()
            module = source / "experiment.py"
            module.write_text("ALPHA = 1\n", encoding="utf-8")
            first = RunManifest.create(
                study="toy",
                phase="E0",
                config={},
                inputs={},
                cwd=root,
                created_at="2026-01-01T00:00:00+00:00",
            )
            module.write_text("ALPHA = 2\n", encoding="utf-8")
            second = RunManifest.create(
                study="toy",
                phase="E0",
                config={},
                inputs={},
                cwd=root,
                created_at="2026-01-01T00:00:00+00:00",
            )
            self.assertNotEqual(first.git["source_tree_sha256"], second.git["source_tree_sha256"])
            self.assertNotEqual(first.manifest_digest, second.manifest_digest)
            with self.assertRaises(FrozenArtifactError):
                RunManifest.create(
                    study="toy",
                    phase="E10",
                    config={},
                    inputs={},
                    cwd=root,
                    require_clean_git=True,
                )


if __name__ == "__main__":
    unittest.main()

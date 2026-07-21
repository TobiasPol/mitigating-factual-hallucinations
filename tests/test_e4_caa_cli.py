from pathlib import Path

from mfh.cli import build_parser


def test_m2_caa_cli_surfaces_are_explicit() -> None:
    parser = build_parser()
    common = ["e3-work", "T-steer.jsonl", "m2-work"]

    prepared = parser.parse_args(["prepare-m2-caa", *common])
    assert prepared.e3_construction == Path("e3-work")
    assert prepared.prompt_config == Path("configs/prompts/primary.yaml")

    run = parser.parse_args(
        [
            "run-m2-caa",
            *common,
            "model.yaml",
            "snapshot",
            "--request-budget",
            "64",
        ]
    )
    assert run.model_config == Path("model.yaml")
    assert run.snapshot_directory == Path("snapshot")
    assert run.request_budget == 64

    verified = parser.parse_args(["verify-m2-caa-work", *common, "--require-complete"])
    assert verified.require_complete is True

    finalized = parser.parse_args(["finalize-m2-caa", *common, "m2-artifact"])
    assert finalized.output == Path("m2-artifact")

    artifact = parser.parse_args(
        [
            "verify-m2-caa-artifact",
            "m2-artifact",
            "--expected-manifest-digest",
            "a" * 64,
        ]
    )
    assert artifact.expected_manifest_digest == "a" * 64

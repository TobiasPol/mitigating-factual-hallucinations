from __future__ import annotations

from mfh.cli import build_parser

_COMMON = [
    "splits",
    "e1-output",
    "e1-work",
    "e1-ledger",
    "model.yaml",
    "snapshot",
    "snapshot.json",
    "runtime.json",
    "e2-workspace",
    "e2-capture",
    "--expected-split-manifest-digest",
    "a" * 64,
    "--expected-e1-manifest-digest",
    "b" * 64,
]


def test_e2_scientific_commands_are_wired_to_distinct_handlers() -> None:
    parser = build_parser()
    prepare = parser.parse_args(["prepare-e2-vllm", *_COMMON, "--shard-rows", "32"])
    run = parser.parse_args(["run-e2-vllm", *_COMMON, "--request-budget", "7"])
    verify = parser.parse_args(["verify-e2-capture", *_COMMON, "--require-complete"])
    fit = parser.parse_args(["fit-e2-probes", *_COMMON, "probe-output"])
    replay = parser.parse_args(["verify-e2-probes", "probe-output", "e2-workspace"])
    finalize = parser.parse_args(
        [
            "finalize-e2",
            *_COMMON,
            "probe-output",
            "e2-phase",
            "--expected-workspace-plan-identity",
            "c" * 64,
            "--expected-capture-plan-identity",
            "d" * 64,
            "--expected-probe-manifest-digest",
            "e" * 64,
        ]
    )

    assert prepare.handler.__name__ == "_prepare_e2_vllm"
    assert prepare.shard_rows == 32
    assert run.handler.__name__ == "_run_e2_vllm"
    assert run.request_budget == 7
    assert verify.handler.__name__ == "_verify_e2_capture"
    assert verify.require_complete is True
    assert fit.handler.__name__ == "_fit_e2_probes"
    assert replay.handler.__name__ == "_verify_e2_probes"
    assert finalize.handler.__name__ == "_finalize_e2"

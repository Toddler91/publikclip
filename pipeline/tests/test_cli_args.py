"""The desktop shell invokes `publikclip --jsonl <cmd>`, so the flag has to
survive subcommand parsing.

A subparser that declares --jsonl of its own overwrites the global value with
its default, and the shell then receives no events at all: no result line, so
its progress panel never clears and the run looks hung rather than finished.
"""

import pytest

from publikclip_pipeline.cli import build_parser


@pytest.mark.parametrize(
    "command",
    [
        ["run", "x.mp4"],
        ["resume", "job-id"],
        ["caption", "x.mp4"],
        ["delete", "job-id"],
    ],
)
def test_global_jsonl_survives_every_subcommand(command):
    args = build_parser().parse_args(["--jsonl", *command])
    assert args.jsonl is True, f"--jsonl was swallowed by `{command[0]}`"


def test_jsonl_defaults_off():
    assert build_parser().parse_args(["caption", "x.mp4"]).jsonl is False


def test_no_subparser_redeclares_jsonl():
    """Guards the cause rather than the symptom."""
    parser = build_parser()
    actions = [a for a in parser._actions if getattr(a, "dest", None) == "jsonl"]
    assert len(actions) == 1

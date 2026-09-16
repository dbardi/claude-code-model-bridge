"""Behavior of the adapter that actually runs the CLI."""

import json
import os
import stat
from pathlib import Path

import pytest

from claude_code_model_bridge.claude_cli import Invocation, Turn
from claude_code_model_bridge.claude_process import ClaudeProcess

STUB = """#!/usr/bin/env python3
import json, os, sys

Path = os.environ["STUB_RECORD"]
record = {"argv": sys.argv[1:], "stdin": sys.stdin.read(), "env_had_key": "ANTHROPIC_API_KEY" in os.environ}
open(Path, "w").write(json.dumps(record))
for line in json.loads(os.environ.get("STUB_EVENTS", "[]")):
    print(json.dumps(line), flush=True)
sys.exit(int(os.environ.get("STUB_EXIT", "0")))
"""


@pytest.fixture
def claude_stub(tmp_path):
    """A stand-in `claude` that records how it was called and prints given events."""
    script = tmp_path / "claude-stub"
    script.write_text(STUB)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    record = tmp_path / "record.json"

    def build(events=(), exit_code=0, environment=None):
        env = dict(environment or {})
        env.update(
            {
                "STUB_RECORD": str(record),
                "STUB_EVENTS": json.dumps(list(events)),
                "STUB_EXIT": str(exit_code),
                "PATH": os.environ["PATH"],
            }
        )
        return ClaudeProcess(executable=str(script), environment=env)

    build.record = lambda: json.loads(Path(record).read_text())
    return build


def an_invocation(**overrides) -> Invocation:
    defaults = {
        "model": "test-model-v1",
        "turns": (Turn(role="user", text="Say pong."),),
        "system_prompt": "Be terse.",
    }
    return Invocation(**{**defaults, **overrides})


async def test_yields_the_events_the_cli_prints(claude_stub):
    printed = [
        {"type": "stream_event", "event": {"type": "message_start"}},
        {"type": "result", "subtype": "success", "result": "pong"},
    ]
    process = claude_stub(events=printed)

    events = [event async for event in process.run(an_invocation())]

    assert events == printed

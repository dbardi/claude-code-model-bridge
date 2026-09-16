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


ISOLATION_FLAGS = [
    "--setting-sources",
    "",
    "--strict-mcp-config",
    "--tools",
    "",
    "--no-session-persistence",
]


async def test_yields_the_events_the_cli_prints(claude_stub):
    printed = [
        {"type": "stream_event", "event": {"type": "message_start"}},
        {"type": "result", "subtype": "success", "result": "pong"},
    ]
    process = claude_stub(events=printed)

    events = [event async for event in process.run(an_invocation())]

    assert events == printed


async def test_every_call_is_isolated_from_local_configuration(claude_stub):
    process = claude_stub()

    [event async for event in process.run(an_invocation())]

    argv = claude_stub.record()["argv"]
    for index, flag in enumerate(ISOLATION_FLAGS):
        assert flag in argv, f"missing isolation flag: {flag or '(empty string)'}"
    assert argv[argv.index("--setting-sources") + 1] == ""
    assert argv[argv.index("--tools") + 1] == ""


async def test_the_conversation_reaches_the_cli_as_stream_json(claude_stub):
    process = claude_stub()
    invocation = an_invocation(
        turns=(
            Turn(role="user", text="How much disk is free?"),
            Turn(role="assistant", text="Checking."),
            Turn(role="user", text="[tool_result id=call_1 name=terminal]\n467G"),
        )
    )

    [event async for event in process.run(invocation)]

    written = claude_stub.record()["stdin"].strip().splitlines()
    messages = [json.loads(line) for line in written]
    assert [message["message"]["role"] for message in messages] == [
        "user",
        "assistant",
        "user",
    ]
    assert messages[0]["message"]["content"][0]["text"] == "How much disk is free?"
    assert "--input-format" in claude_stub.record()["argv"]

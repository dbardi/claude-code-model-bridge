"""Behavior of the adapter that actually runs the CLI."""

import asyncio
import json
import os
import stat
from pathlib import Path

import pytest

from claude_code_model_bridge.claude_cli import Invocation, Turn
from claude_code_model_bridge.claude_process import (
    ApiKeyPresent,
    ClaudeFailed,
    ClaudeProcess,
    ClaudeTimedOut,
)

STUB = """#!/usr/bin/env python3
import json, os, sys, time

argv = sys.argv[1:]
prompt = ""
if "--system-prompt-file" in argv:
    prompt = open(argv[argv.index("--system-prompt-file") + 1]).read()
record = {
    "argv": argv,
    "stdin": sys.stdin.read(),
    "system_prompt": prompt,
    "pid": os.getpid(),
    "env_had_key": "ANTHROPIC_API_KEY" in os.environ,
}
open(os.environ["STUB_RECORD"], "w").write(json.dumps(record))
for line in json.loads(os.environ.get("STUB_EVENTS", "[]")):
    print(json.dumps(line), flush=True)
time.sleep(float(os.environ.get("STUB_LINGER", "0")))
sys.exit(int(os.environ.get("STUB_EXIT", "0")))
"""


@pytest.fixture
def claude_stub(tmp_path):
    """A stand-in `claude` that records how it was called and prints given events."""
    script = tmp_path / "claude-stub"
    script.write_text(STUB)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    record = tmp_path / "record.json"

    def build(
        events=(),
        exit_code=0,
        environment=None,
        linger_seconds=0,
        total_seconds=900,
        silence_seconds=300,
        plugin_dir=None,
    ):
        env = dict(environment or {})
        env.update(
            {
                "STUB_RECORD": str(record),
                "STUB_EVENTS": json.dumps(list(events)),
                "STUB_EXIT": str(exit_code),
                "STUB_LINGER": str(linger_seconds),
                "PATH": os.environ["PATH"],
            }
        )
        return ClaudeProcess(
            executable=str(script),
            environment=env,
            total_seconds=total_seconds,
            silence_seconds=silence_seconds,
            plugin_dir=plugin_dir,
        )

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


async def test_abandoning_a_run_stops_the_work(claude_stub):
    """A caller that walks away must not leave Claude generating."""
    process = claude_stub(events=[{"type": "stream_event"}], linger_seconds=30)

    generator = process.run(an_invocation())
    await generator.__anext__()
    child = claude_stub.record()["pid"]
    await generator.aclose()

    for _ in range(50):
        if not _running(child):
            break
        await asyncio.sleep(0.05)
    assert not _running(child)


def _running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def test_no_plugins_are_loaded_by_default(claude_stub):
    process = claude_stub()

    [event async for event in process.run(an_invocation())]

    assert "--plugin-dir" not in claude_stub.record()["argv"]


async def test_a_configured_plugin_directory_is_loaded(claude_stub, tmp_path):
    skills = tmp_path / "skills"
    skills.mkdir()
    process = claude_stub(plugin_dir=skills)

    [event async for event in process.run(an_invocation())]

    argv = claude_stub.record()["argv"]
    assert argv[argv.index("--plugin-dir") + 1] == str(skills)


async def test_skills_can_be_chosen_only_where_plugins_are_loaded(claude_stub, tmp_path):
    """Choosing a skill needs the Skill tool, which loads text and runs nothing."""
    skills = tmp_path / "skills"
    skills.mkdir()

    process = claude_stub(plugin_dir=skills)
    [event async for event in process.run(an_invocation())]
    with_plugins = claude_stub.record()["argv"]

    process = claude_stub()
    [event async for event in process.run(an_invocation())]
    without_plugins = claude_stub.record()["argv"]

    assert with_plugins[with_plugins.index("--tools") + 1] == "Skill"
    assert without_plugins[without_plugins.index("--tools") + 1] == ""


async def test_a_run_that_overruns_its_cap_is_stopped(claude_stub):
    process = claude_stub(
        events=[{"type": "stream_event"}], linger_seconds=30, total_seconds=0.2
    )

    with pytest.raises(ClaudeTimedOut):
        [event async for event in process.run(an_invocation())]

    assert not _running(claude_stub.record()["pid"])


async def test_a_run_that_goes_quiet_is_stopped(claude_stub):
    """A wedged CLI produces nothing at all, and must not hold a slot for hours."""
    process = claude_stub(
        events=[{"type": "stream_event"}], linger_seconds=30, silence_seconds=0.2
    )

    with pytest.raises(ClaudeTimedOut):
        [event async for event in process.run(an_invocation())]

    assert not _running(claude_stub.record()["pid"])


async def test_refuses_to_run_when_an_api_key_could_be_billed(claude_stub):
    process = claude_stub(environment={"ANTHROPIC_API_KEY": "sk-not-a-real-key"})

    with pytest.raises(ApiKeyPresent):
        [event async for event in process.run(an_invocation())]


async def test_credentials_never_reach_the_cli(claude_stub):
    process = claude_stub(environment={"ANTHROPIC_AUTH_TOKEN": "not-a-real-token"})

    with pytest.raises(ApiKeyPresent):
        [event async for event in process.run(an_invocation())]


async def test_a_failed_run_is_reported_rather_than_answered_emptily(claude_stub):
    process = claude_stub(events=[], exit_code=1)

    with pytest.raises(ClaudeFailed):
        [event async for event in process.run(an_invocation())]


async def test_the_system_prompt_travels_in_a_file(claude_stub):
    process = claude_stub()
    prompt = "You are a terse assistant. " + "Filler. " * 5000

    [event async for event in process.run(an_invocation(system_prompt=prompt))]

    assert claude_stub.record()["system_prompt"] == prompt


async def test_the_system_prompt_file_is_cleaned_up(claude_stub):
    process = claude_stub()

    [event async for event in process.run(an_invocation())]

    argv = claude_stub.record()["argv"]
    assert not Path(argv[argv.index("--system-prompt-file") + 1]).exists()


async def test_a_schema_is_passed_only_when_tools_were_declared(claude_stub):
    schema = {"type": "object", "properties": {"content": {"type": "string"}}}
    process = claude_stub()

    [event async for event in process.run(an_invocation(output_schema=schema))]
    with_schema = claude_stub.record()["argv"]

    [event async for event in process.run(an_invocation())]
    without_schema = claude_stub.record()["argv"]

    assert json.loads(with_schema[with_schema.index("--json-schema") + 1]) == schema
    assert "--json-schema" not in without_schema


async def test_thinking_depth_is_passed_only_when_asked_for(claude_stub):
    process = claude_stub()

    [event async for event in process.run(an_invocation(effort="high"))]
    with_effort = claude_stub.record()["argv"]

    [event async for event in process.run(an_invocation())]
    without_effort = claude_stub.record()["argv"]

    assert with_effort[with_effort.index("--effort") + 1] == "high"
    assert "--effort" not in without_effort


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

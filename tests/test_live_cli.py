"""Checks the real CLI still behaves as the adapter assumes.

Every test here spends subscription usage, so they are skipped unless asked
for: `uv run pytest -m live`. They exist to catch a CLI release changing
behavior the bridge depends on, which no fake can detect.
"""

import pytest

from claude_code_model_bridge.claude_cli import Invocation, Turn
from claude_code_model_bridge.claude_process import ClaudeProcess

pytestmark = pytest.mark.live


async def answer(invocation: Invocation) -> dict:
    """Runs the real CLI and returns its terminating result event."""
    events = [event async for event in ClaudeProcess().run(invocation)]
    results = [event for event in events if event.get("type") == "result"]
    assert results, f"no result event; got types {[e.get('type') for e in events]}"
    return results[-1]


async def test_a_plain_question_is_answered():
    result = await answer(
        Invocation(
            model="claude-haiku-4-5",
            turns=(Turn(role="user", text="Reply with exactly: pong"),),
            system_prompt="You are a terse assistant. Answer exactly as asked.",
        )
    )

    assert result["is_error"] is False
    assert "pong" in result["result"].lower()


async def test_a_declared_tool_comes_back_as_structured_output():
    schema = {
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "tool_calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": ["terminal"]},
                        "arguments": {"type": "object"},
                    },
                    "required": ["name", "arguments"],
                },
            },
        },
        "required": ["content", "tool_calls"],
    }
    result = await answer(
        Invocation(
            model="claude-haiku-4-5",
            turns=(Turn(role="user", text="How much disk is free on /home?"),),
            system_prompt=(
                "You answer through the declared functions. Available function: "
                'terminal(command: string), which runs a shell command. Put calls '
                "in tool_calls; otherwise leave tool_calls empty."
            ),
            output_schema=schema,
        )
    )

    requested = result["structured_output"]["tool_calls"]
    assert requested, "expected the model to request the declared tool"
    assert requested[0]["name"] == "terminal"
    assert "command" in requested[0]["arguments"]


async def test_the_isolation_flags_keep_the_prompt_small():
    """Local connectors and plugins must not be loaded into a call.

    Without the isolation flags a minimal call measured hundreds of
    thousands of prompt tokens on this machine; with them, hundreds.
    """
    result = await answer(
        Invocation(
            model="claude-haiku-4-5",
            turns=(Turn(role="user", text="Reply with exactly: pong"),),
            system_prompt="You are a terse assistant.",
        )
    )

    usage = result["usage"]
    prompt_tokens = (
        usage["input_tokens"]
        + usage["cache_creation_input_tokens"]
        + usage["cache_read_input_tokens"]
    )
    assert prompt_tokens < 10_000, f"prompt was {prompt_tokens} tokens; isolation leaked"

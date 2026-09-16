"""Behavior at the HTTP seam: what a client sees when it asks for a completion."""

import json

from tests.cli_events import (
    assistant_text_event,
    result_event,
    structured_result_event,
    text_delta_event,
)
from tests.models import MODEL


async def test_answers_with_the_text_claude_produced(bridge):
    client, _ = bridge([assistant_text_event("pong"), result_event("pong")])

    completion = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Reply with exactly: pong"}],
    )

    assert completion.choices[0].message.content == "pong"


async def test_streams_the_text_in_fragments_as_claude_produces_it(bridge):
    client, _ = bridge(
        [
            text_delta_event("Hello there"),
            text_delta_event(", nice to"),
            text_delta_event(" meet you!"),
            result_event("Hello there, nice to meet you!"),
        ]
    )

    stream = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Greet me."}],
        stream=True,
    )
    fragments = [chunk.choices[0].delta.content async for chunk in stream]

    assert "".join(f for f in fragments if f) == "Hello there, nice to meet you!"


async def test_streaming_completion_ends_with_a_finish_reason(bridge):
    client, _ = bridge([text_delta_event("hi"), result_event("hi")])

    stream = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Greet me."}],
        stream=True,
    )
    reasons = [chunk.choices[0].finish_reason async for chunk in stream if chunk.choices]

    assert reasons[-1] == "stop"


async def test_streaming_completion_reports_usage_when_asked(bridge):
    client, _ = bridge(
        [
            text_delta_event("hi"),
            result_event(
                "hi",
                input_tokens=100,
                cache_creation_input_tokens=20,
                cache_read_input_tokens=5,
                output_tokens=7,
            ),
        ]
    )

    stream = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Greet me."}],
        stream=True,
        stream_options={"include_usage": True},
    )
    usages = [chunk.usage async for chunk in stream if chunk.usage]

    assert usages[-1].prompt_tokens == 125
    assert usages[-1].completion_tokens == 7
    assert usages[-1].total_tokens == 132


TERMINAL_TOOL = {
    "type": "function",
    "function": {
        "name": "terminal",
        "description": "Run a shell command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}


async def test_a_requested_tool_comes_back_as_a_tool_call(bridge):
    client, _ = bridge(
        [
            structured_result_event(
                "Checking the disk.",
                [{"name": "terminal", "arguments": {"command": "df -h /home"}}],
            )
        ]
    )

    completion = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "How much disk is free?"}],
        tools=[TERMINAL_TOOL],
    )

    call = completion.choices[0].message.tool_calls[0]
    assert call.function.name == "terminal"
    assert json.loads(call.function.arguments) == {"command": "df -h /home"}
    assert completion.choices[0].finish_reason == "tool_calls"


async def test_declared_tools_constrain_the_names_claude_may_request(bridge):
    client, claude = bridge([structured_result_event("ok", [])])

    await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "How much disk is free?"}],
        tools=[TERMINAL_TOOL],
    )

    schema = claude.invocations[0].output_schema
    call_properties = schema["properties"]["tool_calls"]["items"]["properties"]
    assert call_properties["name"]["enum"] == ["terminal"]


async def test_a_request_without_tools_is_unconstrained(bridge):
    client, claude = bridge([result_event("pong")])

    await client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": "Say pong."}]
    )

    assert claude.invocations[0].output_schema is None

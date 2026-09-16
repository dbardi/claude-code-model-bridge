"""Behavior at the HTTP seam when the answer is schema-validated JSON."""

import json

from tests.cli_events import json_delta_event, structured_result_event
from tests.models import MODEL

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


async def collect(stream):
    text, calls = "", []
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        text += delta.content or ""
        calls += delta.tool_calls or []
    return text, calls


async def test_prose_streams_out_of_the_accumulating_json(bridge):
    client, _ = bridge(
        [
            json_delta_event('{"content": "The ocean cov'),
            json_delta_event('ers most of Earth'),
            json_delta_event('\'s surface."'),
            json_delta_event(', "tool_calls": []}'),
            structured_result_event("The ocean covers most of Earth's surface.", []),
        ]
    )

    stream = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Tell me about the ocean."}],
        tools=[TERMINAL_TOOL],
        stream=True,
    )
    text, _ = await collect(stream)

    assert text == "The ocean covers most of Earth's surface."


async def test_escapes_split_across_fragments_survive(bridge):
    answer = 'He said "café"\nfine.'
    client, _ = bridge(
        [
            json_delta_event('{"content": "He said \\'),
            json_delta_event('"caf'),
            json_delta_event("\\u00"),
            json_delta_event('e9\\" \\'),
            json_delta_event('u000afine."'),
            json_delta_event(', "tool_calls": []}'),
            structured_result_event(answer, []),
        ]
    )

    stream = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Quote something."}],
        tools=[TERMINAL_TOOL],
        stream=True,
    )
    text, _ = await collect(stream)

    assert text == 'He said "café" \nfine.'


async def test_tool_calls_arrive_once_the_json_closes(bridge):
    client, _ = bridge(
        [
            json_delta_event('{"content": "Checking.", "tool_calls": [{"name": "term'),
            json_delta_event('inal", "arguments": {"command": "df -h"}}]}'),
            structured_result_event(
                "Checking.", [{"name": "terminal", "arguments": {"command": "df -h"}}]
            ),
        ]
    )

    stream = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Check the disk."}],
        tools=[TERMINAL_TOOL],
        stream=True,
    )
    text, calls = await collect(stream)
    reasons = []
    assert text == "Checking."
    assert calls[0].function.name == "terminal"
    assert json.loads(calls[0].function.arguments) == {"command": "df -h"}


async def test_a_stream_that_requested_tools_finishes_as_tool_calls(bridge):
    client, _ = bridge(
        [
            structured_result_event(
                "Checking.", [{"name": "terminal", "arguments": {"command": "df -h"}}]
            )
        ]
    )

    stream = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Check the disk."}],
        tools=[TERMINAL_TOOL],
        stream=True,
    )
    reasons = [c.choices[0].finish_reason async for c in stream if c.choices]

    assert reasons[-1] == "tool_calls"

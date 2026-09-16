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

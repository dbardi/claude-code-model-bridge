"""Behavior when the CLI answers twice for one request.

Under a schema the CLI takes a plain-text answer, injects a message telling
the model to call the structured-output tool, and takes a second answer.
"""

import httpx
import pytest
from openai import AsyncOpenAI

from claude_code_model_bridge.app import create_app
from claude_code_model_bridge.catalog import ModelCatalog
from tests.cli_events import json_delta_event, structured_result_event, text_delta_event
from tests.conftest import FakeClaudeCli
from tests.models import CATALOG_YAML, MODEL

BASE_URL = "http://bridge.test/v1"
ANSWER = "the ocean is large."

TOOL = {
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

ENFORCE_EVENT = {
    "type": "user",
    "message": {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "[structured-output-enforce] You MUST call the StructuredOutput "
                    "tool to complete this request. Call this tool now."
                ),
            }
        ],
    },
}

TWO_PASSES = [
    text_delta_event("the"),
    text_delta_event(" ocean is large."),
    ENFORCE_EVENT,
    json_delta_event('{"content": "the ocean'),
    json_delta_event(' is large.", "tool_calls": []}'),
    structured_result_event(ANSWER, []),
]


def bridge_for(events) -> AsyncOpenAI:
    app = create_app(
        claude_cli=FakeClaudeCli(events), catalog=ModelCatalog.from_yaml(CATALOG_YAML)
    )
    return AsyncOpenAI(
        api_key="unused",
        base_url=BASE_URL,
        http_client=httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url=BASE_URL
        ),
    )


async def streamed_text(client) -> str:
    stream = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Tell me about the ocean."}],
        tools=[TOOL],
        stream=True,
    )
    text = ""
    async for chunk in stream:
        if chunk.choices:
            text += chunk.choices[0].delta.content or ""
    return text


async def test_an_answer_given_twice_is_said_once():
    client = bridge_for(TWO_PASSES)

    assert await streamed_text(client) == ANSWER


TWO_STRUCTURED_PASSES = [
    json_delta_event('{"content": "Morning. Nothing queued.", "tool_calls": []}'),
    ENFORCE_EVENT,
    json_delta_event('{"content": "Here is the project list you asked for."'),
    json_delta_event(', "tool_calls": []}'),
    structured_result_event("Here is the project list you asked for.", []),
]


async def test_the_last_answer_wins_when_the_cli_answers_twice():
    """The CLI can produce several structured answers; only the last is the reply."""
    client = bridge_for(TWO_STRUCTURED_PASSES)

    assert await streamed_text(client) == "Here is the project list you asked for."


async def test_an_answer_the_model_never_restated_is_still_said():
    """The second pass may carry nothing; the reply must not vanish with it."""
    client = bridge_for(
        [
            text_delta_event("the ocean is large."),
            ENFORCE_EVENT,
            structured_result_event(ANSWER, []),
        ]
    )

    assert await streamed_text(client) == ANSWER

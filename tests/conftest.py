"""Test fixtures for driving the bridge the way Hermes drives it."""

from collections.abc import AsyncIterator, Iterable
from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI

from claude_code_model_bridge.app import create_app
from claude_code_model_bridge.claude_cli import Invocation

BASE_URL = "http://bridge.test/v1"


class FakeClaudeCli:
    """Stands in for the real CLI at the Claude seam, yielding recorded events."""

    def __init__(self, events: Iterable[dict[str, Any]]) -> None:
        self._events = list(events)
        self.invocations: list[Invocation] = []

    async def run(self, invocation: Invocation) -> AsyncIterator[dict[str, Any]]:
        self.invocations.append(invocation)
        for event in self._events:
            yield event


def assistant_text_event(text: str) -> dict[str, Any]:
    """The CLI's `assistant` event, as emitted by `--output-format stream-json`."""
    return {
        "type": "assistant",
        "message": {
            "model": "claude-opus-5",
            "id": "msg_01Test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "stop_reason": None,
        },
    }


def result_event(text: str) -> dict[str, Any]:
    """The CLI's terminating `result` event."""
    return {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": text,
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 12,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": 3,
        },
    }


@pytest.fixture
def bridge():
    """Builds a bridge whose Claude seam is filled by a fake, plus a client for it."""

    def build(events: Iterable[dict[str, Any]]) -> tuple[AsyncOpenAI, FakeClaudeCli]:
        claude_cli = FakeClaudeCli(events)
        app = create_app(claude_cli=claude_cli)
        http_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url=BASE_URL
        )
        client = AsyncOpenAI(api_key="unused", base_url=BASE_URL, http_client=http_client)
        return client, claude_cli

    return build

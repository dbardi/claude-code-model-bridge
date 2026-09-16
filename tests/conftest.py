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

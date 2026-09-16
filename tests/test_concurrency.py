"""Behavior at the HTTP seam when several requests arrive at once."""

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import httpx
from openai import AsyncOpenAI

from claude_code_model_bridge.app import create_app
from claude_code_model_bridge.catalog import ModelCatalog
from claude_code_model_bridge.claude_cli import Invocation
from tests.cli_events import result_event
from tests.models import CATALOG_YAML, MODEL

BASE_URL = "http://bridge.test/v1"


class CountingClaudeCli:
    """Reports the most runs it was ever asked to do at the same time."""

    def __init__(self) -> None:
        self.running = 0
        self.most_at_once = 0

    async def run(self, invocation: Invocation) -> AsyncIterator[dict[str, Any]]:
        self.running += 1
        self.most_at_once = max(self.most_at_once, self.running)
        try:
            await asyncio.sleep(0.02)
            yield result_event("ok")
        finally:
            self.running -= 1


async def test_only_so_many_calls_run_at_once():
    claude = CountingClaudeCli()
    app = create_app(
        claude_cli=claude,
        catalog=ModelCatalog.from_yaml(CATALOG_YAML),
        max_concurrent=2,
    )
    client = AsyncOpenAI(
        api_key="unused",
        base_url=BASE_URL,
        http_client=httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url=BASE_URL
        ),
    )

    await asyncio.gather(
        *(
            client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": "Say ok."}]
            )
            for _ in range(6)
        )
    )

    assert claude.most_at_once <= 2

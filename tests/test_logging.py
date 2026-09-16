"""What the bridge records about a request, and what it must never record."""

import json
import logging

import httpx
from openai import AsyncOpenAI

from claude_code_model_bridge.app import create_app
from claude_code_model_bridge.catalog import ModelCatalog
from tests.cli_events import result_event
from tests.conftest import FakeClaudeCli
from tests.models import CATALOG_YAML, MODEL

BASE_URL = "http://bridge.test/v1"
SECRET = "the-private-thing-the-user-said"


async def completed_request(caplog):
    claude = FakeClaudeCli([result_event("an answer mentioning nothing private")])
    app = create_app(claude_cli=claude, catalog=ModelCatalog.from_yaml(CATALOG_YAML))
    client = AsyncOpenAI(
        api_key="unused",
        base_url=BASE_URL,
        http_client=httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url=BASE_URL
        ),
    )
    with caplog.at_level(logging.INFO):
        await client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": SECRET}]
        )
    return caplog


async def test_a_completed_request_is_recorded(caplog):
    logs = await completed_request(caplog)

    records = [json.loads(r.message) for r in logs.records if r.message.startswith("{")]
    assert records, "no structured record was written"
    record = records[-1]
    assert record["model"] == MODEL
    assert record["tools"] is False
    assert record["prompt_tokens"] == 12
    assert record["completion_tokens"] == 3
    assert record["duration_ms"] >= 0


async def test_conversation_text_is_never_recorded(caplog):
    """Real conversations pass through, so their content must not land in logs."""
    logs = await completed_request(caplog)

    assert SECRET not in logs.text
    assert "an answer mentioning nothing private" not in logs.text

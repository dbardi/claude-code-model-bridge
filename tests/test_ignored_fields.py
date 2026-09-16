"""What happens to request fields the CLI cannot act on."""

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


def bridge() -> AsyncOpenAI:
    app = create_app(
        claude_cli=FakeClaudeCli([result_event("ok")]),
        catalog=ModelCatalog.from_yaml(CATALOG_YAML),
    )
    return AsyncOpenAI(
        api_key="unused",
        base_url=BASE_URL,
        http_client=httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url=BASE_URL
        ),
    )


async def ask(client, **extra):
    return await client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": "Say ok."}], **extra
    )


async def test_the_answer_names_the_fields_that_did_nothing():
    completion = await ask(bridge(), temperature=0.2, max_tokens=50)

    ignored = completion.ignored_parameters
    assert sorted(ignored) == ["max_tokens", "temperature"]


async def test_a_request_the_bridge_can_honor_carries_no_such_note():
    completion = await ask(bridge(), stream=False)

    assert "ignored_parameters" not in completion.model_dump()


async def test_the_log_records_what_was_ignored(caplog):
    with caplog.at_level(logging.INFO):
        await ask(bridge(), top_p=0.9)

    records = [json.loads(r.message) for r in caplog.records if r.message.startswith("{")]
    assert records[-1]["ignored"] == ["top_p"]


async def test_a_streamed_answer_names_them_too():
    stream = await ask(bridge(), temperature=0.2, stream=True)

    ignored = [chunk.ignored_parameters async for chunk in stream if chunk.choices][0]
    assert ignored == ["temperature"]

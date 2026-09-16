"""How a request's tool_choice reaches the model."""

import httpx
import pytest
from openai import AsyncOpenAI, BadRequestError

from claude_code_model_bridge.app import create_app
from claude_code_model_bridge.catalog import ModelCatalog
from tests.cli_events import result_event
from tests.conftest import FakeClaudeCli
from tests.models import CATALOG_YAML, MODEL

BASE_URL = "http://bridge.test/v1"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "terminal",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    },
]


def bridge():
    claude = FakeClaudeCli([result_event("ok")])
    app = create_app(claude_cli=claude, catalog=ModelCatalog.from_yaml(CATALOG_YAML))
    client = AsyncOpenAI(
        api_key="unused",
        base_url=BASE_URL,
        max_retries=0,
        http_client=httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url=BASE_URL
        ),
    )
    return client, claude


async def ask(client, **extra):
    return await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Do the thing."}],
        tools=TOOLS,
        **extra,
    )


def tool_names(invocation) -> list[str]:
    schema = invocation.output_schema
    return schema["properties"]["tool_calls"]["items"]["properties"]["name"]["enum"]


async def test_a_named_tool_is_the_only_one_offered():
    client, claude = bridge()

    await ask(client, tool_choice={"type": "function", "function": {"name": "read_file"}})

    assert tool_names(claude.invocations[0]) == ["read_file"]


async def test_required_means_the_model_must_call_something():
    client, claude = bridge()

    await ask(client, tool_choice="required")

    schema = claude.invocations[0].output_schema
    assert schema["properties"]["tool_calls"]["minItems"] == 1


async def test_none_leaves_the_answer_unconstrained():
    client, claude = bridge()

    await ask(client, tool_choice="none")

    assert claude.invocations[0].output_schema is None


async def test_auto_offers_every_declared_tool():
    client, claude = bridge()

    await ask(client, tool_choice="auto")

    assert tool_names(claude.invocations[0]) == ["terminal", "read_file"]


async def test_naming_a_tool_that_was_not_declared_is_refused():
    client, _ = bridge()

    with pytest.raises(BadRequestError):
        await ask(
            client, tool_choice={"type": "function", "function": {"name": "nonexistent"}}
        )

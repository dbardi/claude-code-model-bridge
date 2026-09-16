"""How failures reach a caller: the status decides what the caller does next."""

import httpx
import pytest
from openai import AsyncOpenAI, AuthenticationError

from claude_code_model_bridge.app import create_app
from claude_code_model_bridge.catalog import ModelCatalog
from tests.conftest import FakeClaudeCli
from tests.models import CATALOG_YAML, MODEL

BASE_URL = "http://bridge.test/v1"


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


def failed_result(text: str) -> dict:
    """The CLI's terminating event when the run did not produce an answer."""
    return {
        "type": "result",
        "subtype": "error_during_execution",
        "is_error": True,
        "result": text,
        "stop_reason": "stop_sequence",
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


async def test_not_being_logged_in_is_an_authentication_failure():
    """Retrying cannot fix this, so it must not look retryable."""
    client = bridge_for([failed_result("Not logged in · Please run /login")])

    with pytest.raises(AuthenticationError):
        await client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": "Say ok."}]
        )

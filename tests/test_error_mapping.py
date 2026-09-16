"""How failures reach a caller: the status decides what the caller does next."""

import time

import httpx
import pytest
from openai import APIStatusError, AsyncOpenAI, AuthenticationError, RateLimitError

from claude_code_model_bridge.app import create_app
from claude_code_model_bridge.catalog import ModelCatalog
from claude_code_model_bridge.claude_process import (
    ApiKeyPresent,
    ClaudeFailed,
    ClaudeTimedOut,
)
from tests.cli_events import result_event
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


async def test_an_answer_without_structured_output_is_still_an_answer():
    """Losing the structured wrapper should cost the tool calls, not the reply."""
    answered = result_event("Here is what I found.")
    answered.pop("structured_output", None)
    client = bridge_for([answered])

    completion = await say_ok(client)

    assert completion.choices[0].message.content == "Here is what I found."
    assert completion.choices[0].message.tool_calls is None
    assert completion.choices[0].finish_reason == "stop"


async def test_an_empty_answer_is_passed_through():
    """Callers have their own handling for an empty reply; do not invent one."""
    client = bridge_for([result_event("")])

    completion = await say_ok(client)

    assert completion.choices[0].message.content == ""


def rate_limit_event(resets_at: int, status: str = "rejected") -> dict:
    """The CLI's report of the subscription usage window."""
    return {
        "type": "rate_limit_event",
        "rate_limit_info": {
            "status": status,
            "resetsAt": resets_at,
            "rateLimitType": "five_hour",
        },
    }


async def test_an_exhausted_usage_window_says_when_to_come_back():
    """A caller that honors Retry-After waits exactly as long as needed."""
    resets_at = int(time.time()) + 600
    client = bridge_for(
        [
            rate_limit_event(resets_at),
            failed_result("Claude usage limit reached."),
        ]
    )

    with pytest.raises(RateLimitError) as failure:
        await say_ok(client)

    retry_after = int(failure.value.response.headers["retry-after"])
    assert 540 <= retry_after <= 600


async def test_an_allowed_usage_window_is_not_a_rate_limit():
    """Every successful run also reports the window; that is not a failure."""
    client = bridge_for(
        [rate_limit_event(int(time.time()) + 600, status="allowed"), result_event("ok")]
    )

    completion = await say_ok(client)

    assert completion.choices[0].message.content == "ok"


async def stream_ok(client):
    stream = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Say ok."}],
        stream=True,
    )
    return [chunk async for chunk in stream]


async def test_a_streaming_request_reports_authentication_failure_too():
    """Streaming is the usual path, so it cannot be the one that hides failures."""
    client = bridge_for([failed_result("Not logged in · Please run /login")])

    with pytest.raises(AuthenticationError):
        await stream_ok(client)


async def test_a_streaming_request_reports_an_exhausted_window():
    resets_at = int(time.time()) + 300
    client = bridge_for(
        [rate_limit_event(resets_at), failed_result("Claude usage limit reached.")]
    )

    with pytest.raises(RateLimitError) as failure:
        await stream_ok(client)

    assert int(failure.value.response.headers["retry-after"]) <= 300


class RaisingClaudeCli:
    """Stands in for a CLI that fails partway rather than answering."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    async def run(self, invocation):
        raise self._error
        yield  # pragma: no cover - makes this an async generator


def bridge_raising(error: Exception) -> AsyncOpenAI:
    app = create_app(
        claude_cli=RaisingClaudeCli(error), catalog=ModelCatalog.from_yaml(CATALOG_YAML)
    )
    return AsyncOpenAI(
        api_key="unused",
        base_url=BASE_URL,
        max_retries=0,
        http_client=httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url=BASE_URL
        ),
    )


async def say_ok(client):
    return await client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": "Say ok."}]
    )


async def test_a_crashed_run_is_reported_as_retryable():
    """A crash is usually transient, so the caller should be free to try again."""
    client = bridge_raising(ClaudeFailed("claude exited with 1"))

    with pytest.raises(APIStatusError) as failure:
        await say_ok(client)

    assert failure.value.status_code == 502


async def test_a_run_that_outlived_its_limits_is_reported_as_a_timeout():
    client = bridge_raising(ClaudeTimedOut("claude ran longer than 900s"))

    with pytest.raises(APIStatusError) as failure:
        await say_ok(client)

    assert failure.value.status_code == 504


async def test_a_billable_credential_stops_the_bridge_loudly():
    """This is a configuration mistake on the machine, not a passing fault."""
    client = bridge_raising(ApiKeyPresent("ANTHROPIC_API_KEY is set."))

    with pytest.raises(APIStatusError) as failure:
        await say_ok(client)

    assert failure.value.status_code == 500
    assert "subscription" in str(failure.value).lower()


async def test_not_being_logged_in_is_an_authentication_failure():
    """Retrying cannot fix this, so it must not look retryable."""
    client = bridge_for([failed_result("Not logged in · Please run /login")])

    with pytest.raises(AuthenticationError):
        await client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": "Say ok."}]
        )

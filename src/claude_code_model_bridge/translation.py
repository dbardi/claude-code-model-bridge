"""Translates between the OpenAI protocol and the Claude CLI's events."""

import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from claude_code_model_bridge.claude_cli import Invocation, Turn


def build_invocation(request: dict[str, Any]) -> Invocation:
    """Turns an OpenAI chat completion request into a single Claude invocation."""
    turns = tuple(
        Turn(role=message["role"], text=message["content"])
        for message in request["messages"]
    )
    return Invocation(model=request["model"], turns=turns)


def completion_from_events(events: list[dict[str, Any]], model: str) -> dict[str, Any]:
    """Assembles the OpenAI completion body from the events one invocation produced."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": _answer(events)},
                "finish_reason": "stop",
            }
        ],
    }


def _answer(events: list[dict[str, Any]]) -> str:
    for event in reversed(events):
        if event.get("type") == "result":
            return event.get("result", "")
    return ""


async def stream_chunks(
    events: AsyncIterator[dict[str, Any]], model: str, include_usage: bool = False
) -> AsyncIterator[dict[str, Any]]:
    """Emits an OpenAI chunk for each fragment of text Claude produces."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    def chunk(delta: dict[str, Any], finish_reason: str | None) -> dict[str, Any]:
        return {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": finish_reason}
            ],
        }

    usage = None
    async for event in events:
        text = _text_fragment(event)
        if text:
            yield chunk({"content": text}, None)
        if event.get("type") == "result":
            usage = _usage(event)
    yield chunk({}, "stop")
    if include_usage and usage is not None:
        final = chunk({}, None)
        final["choices"] = []
        final["usage"] = usage
        yield final


def _text_fragment(event: dict[str, Any]) -> str:
    if event.get("type") != "stream_event":
        return ""
    inner = event.get("event", {})
    if inner.get("type") != "content_block_delta":
        return ""
    delta = inner.get("delta", {})
    return delta.get("text", "") if delta.get("type") == "text_delta" else ""


def _usage(result_event: dict[str, Any]) -> dict[str, Any]:
    """Maps the CLI's token counts onto the OpenAI usage fields.

    Cache reads and cache writes are prompt tokens: the CLI reports them
    separately, the OpenAI protocol folds them into `prompt_tokens`.
    """
    counts = result_event.get("usage", {})
    cache_read = counts.get("cache_read_input_tokens", 0)
    prompt_tokens = (
        counts.get("input_tokens", 0)
        + counts.get("cache_creation_input_tokens", 0)
        + cache_read
    )
    completion_tokens = counts.get("output_tokens", 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {"cached_tokens": cache_read},
    }

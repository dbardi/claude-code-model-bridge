"""Translates between the OpenAI protocol and the Claude CLI's events."""

import json
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
    return Invocation(
        model=request["model"],
        turns=turns,
        output_schema=_output_schema(request.get("tools") or []),
    )


def _output_schema(tools: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Constrains the answer to prose plus calls to the declared tools.

    The name enum is what makes an undeclared tool name impossible rather
    than merely discouraged.
    """
    if not tools:
        return None
    names = [tool["function"]["name"] for tool in tools]
    return {
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "tool_calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": names},
                        "arguments": {"type": "object"},
                    },
                    "required": ["name", "arguments"],
                },
            },
        },
        "required": ["content", "tool_calls"],
    }


def completion_from_events(events: list[dict[str, Any]], model: str) -> dict[str, Any]:
    """Assembles the OpenAI completion body from the events one invocation produced."""
    result = _result(events)
    tool_calls = _tool_calls(result)
    message: dict[str, Any] = {"role": "assistant", "content": _answer(result)}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
    }


def _result(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in reversed(events):
        if event.get("type") == "result":
            return event
    return {}


def _answer(result: dict[str, Any]) -> str:
    """The assistant's prose, preferring the schema-validated copy when present."""
    structured = result.get("structured_output")
    if isinstance(structured, dict):
        return structured.get("content", "")
    return result.get("result", "")


def _tool_calls(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Converts Claude's structured tool requests into OpenAI tool calls."""
    structured = result.get("structured_output")
    if not isinstance(structured, dict):
        return []
    return [
        {
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": requested["name"],
                "arguments": json.dumps(requested.get("arguments", {})),
            },
        }
        for requested in structured.get("tool_calls", [])
    ]


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

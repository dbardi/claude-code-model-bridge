"""Translates between the OpenAI protocol and the Claude CLI's events."""

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from claude_code_model_bridge.claude_cli import Invocation, Turn

CONTINUE = "Continue."
"""Closes a conversation that ends on an assistant turn, which Claude cannot answer."""


def build_invocation(request: dict[str, Any]) -> Invocation:
    """Turns an OpenAI chat completion request into a single Claude invocation."""
    messages = request["messages"]
    return Invocation(
        model=request["model"],
        turns=_turns(messages),
        output_schema=_output_schema(request.get("tools") or []),
        system_prompt=_system_prompt(messages),
    )


def _system_prompt(messages: list[dict[str, Any]]) -> str:
    """Joins the system messages; the CLI takes one system prompt, not many."""
    return "\n\n".join(
        message["content"]
        for message in messages
        if message["role"] in ("system", "developer")
    )


def _turns(messages: list[dict[str, Any]]) -> tuple[Turn, ...]:
    """Renders the conversation, tool history included, as plain turns.

    Tool calls and their results are written as labeled text rather than
    native tool-use blocks: see docs/adr/0003. Tool results are attributed
    to the user, the only role the CLI accepts them under.
    """
    tool_names = _tool_names(messages)
    turns: list[Turn] = []
    for message in messages:
        role = message["role"]
        if role in ("system", "developer"):
            continue
        if role == "tool":
            _append(turns, Turn(role="user", text=_tool_result_text(message, tool_names)))
            continue
        _append(
            turns,
            Turn(
                role=role,
                text=_message_text(message),
                images=_images(message.get("content")),
            ),
        )
    if turns and turns[-1].role == "assistant":
        _append(turns, Turn(role="user", text=CONTINUE))
    return tuple(turns)


def _append(turns: list[Turn], turn: Turn) -> None:
    """Adds a turn, merging it into the previous one when the role repeats.

    The CLI expects roles to alternate, and tool results arrive as their own
    messages that would otherwise stack up as consecutive user turns.
    """
    if turns and turns[-1].role == turn.role:
        turns[-1] = Turn(role=turn.role, text=f"{turns[-1].text}\n\n{turn.text}")
        return
    turns.append(turn)


def _tool_names(messages: list[dict[str, Any]]) -> dict[str, str]:
    """Maps a tool call id to its name, so a result can name the tool it came from."""
    return {
        call["id"]: call["function"]["name"]
        for message in messages
        for call in message.get("tool_calls") or []
    }


def _images(content: Any) -> tuple[tuple[str, str], ...]:
    """Pulls inline image data out of a multi-part message.

    Only `data:` URLs are carried: fetching a remote image would mean the
    bridge making network requests of its own, which it never does.
    """
    if not isinstance(content, list):
        return ()
    images = []
    for part in content:
        if part.get("type") != "image_url":
            continue
        url = part.get("image_url", {}).get("url", "")
        if not url.startswith("data:"):
            continue
        header, _, data = url.partition(",")
        media_type = header.removeprefix("data:").removesuffix(";base64")
        images.append((media_type, data))
    return tuple(images)


def _message_text(message: dict[str, Any]) -> str:
    parts = [_content_text(message.get("content"))]
    parts += [
        f"[tool_call id={call['id']} name={call['function']['name']} "
        f"arguments={call['function']['arguments']}]"
        for call in message.get("tool_calls") or []
    ]
    return "\n".join(part for part in parts if part)


def _content_text(content: Any) -> str:
    """Flattens message content, which is either a string or content parts."""
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "")
            for part in content
            if part.get("type") == "text"
        )
    return content or ""


def _tool_result_text(message: dict[str, Any], tool_names: dict[str, str]) -> str:
    call_id = message.get("tool_call_id", "")
    name = tool_names.get(call_id, "unknown")
    return f"[tool_result id={call_id} name={name}]\n{_content_text(message.get('content'))}"


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

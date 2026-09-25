"""Translates between the OpenAI protocol and the Claude CLI's events."""

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from claude_code_model_bridge.catalog import Resolution
from claude_code_model_bridge.claude_cli import Invocation, Turn

CONTINUE = "Continue."
"""Closing turn for a conversation that ends on an assistant message."""


def build_invocation(request: dict[str, Any], resolution: Resolution) -> Invocation:
    """Turns an OpenAI chat completion request into a single Claude invocation."""
    messages = request["messages"]
    tools = request.get("tools") or []
    return Invocation(
        model=resolution.cli_model,
        turns=_turns(messages),
        output_schema=_output_schema(tools, request.get("tool_choice")),
        system_prompt=_system_prompt(messages) + _tool_documentation(tools),
        effort=resolution.effort or request.get("reasoning_effort"),
    )


def _tool_documentation(tools: list[dict[str, Any]]) -> str:
    """Describes the declared tools, since the schema carries only their names."""
    if not tools:
        return ""
    described = "\n\n".join(_described(tool["function"]) for tool in tools)
    return (
        "\n\n# Tools you may call\n\n"
        "Put calls in `tool_calls`, using the arguments each tool declares. "
        "Their results come back in the conversation before you answer.\n\n"
        f"{described}"
    )


def _described(function: dict[str, Any]) -> str:
    lines = [f"## {function['name']}"]
    if function.get("description"):
        lines.append(function["description"])
    if function.get("parameters"):
        lines.append(f"Arguments: {json.dumps(function['parameters'])}")
    return "\n".join(lines)


def _system_prompt(messages: list[dict[str, Any]]) -> str:
    """Joins the system messages; the CLI takes one system prompt, not many."""
    return "\n\n".join(
        message["content"]
        for message in messages
        if message["role"] in ("system", "developer")
    )


def _turns(messages: list[dict[str, Any]]) -> tuple[Turn, ...]:
    """Renders the conversation as plain turns.

    Tool calls and results become labeled text (docs/adr/0003), attributed
    to the user.
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
    """Adds a turn, merging it into the previous one when the role repeats."""
    if turns and turns[-1].role == turn.role:
        turns[-1] = Turn(role=turn.role, text=f"{turns[-1].text}\n\n{turn.text}")
        return
    turns.append(turn)


def _tool_names(messages: list[dict[str, Any]]) -> dict[str, str]:
    """Maps each tool call id to its tool name."""
    return {
        call["id"]: call["function"]["name"]
        for message in messages
        for call in message.get("tool_calls") or []
    }


def _images(content: Any) -> tuple[tuple[str, str], ...]:
    """Pulls inline image data out of a multi-part message. `data:` URLs only."""
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


def _output_schema(
    tools: list[dict[str, Any]], tool_choice: Any = None
) -> dict[str, Any] | None:
    """Constrains the answer to prose plus calls to the declared tools.

    `tool_choice` narrows that: a named tool is the only one offered,
    `required` demands a call, `none` drops the schema.
    """
    if not tools or tool_choice == "none":
        return None
    names = _offered_names(tools, tool_choice)
    schema = {
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
    if tool_choice == "required":
        schema["properties"]["tool_calls"]["minItems"] = 1
    return schema


def _offered_names(tools: list[dict[str, Any]], tool_choice: Any) -> list[str]:
    """The tool names the model may choose from."""
    declared = [tool["function"]["name"] for tool in tools]
    if not isinstance(tool_choice, dict):
        return declared
    named = tool_choice.get("function", {}).get("name")
    if named not in declared:
        raise UndeclaredTool(named)
    return [named]


class UndeclaredTool(Exception):
    """Raised when a request requires a tool it did not declare."""


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


def usage_from_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """The token counts one invocation reported, in OpenAI's shape."""
    return _usage(_result(events))


def _result(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in reversed(events):
        if event.get("type") == "result":
            return event
    return {}


def _answer(result: dict[str, Any]) -> str:
    """The assistant's prose, from the structured output when present."""
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
    events: AsyncIterator[dict[str, Any]],
    model: str,
    include_usage: bool = False,
    schema_mode: bool = False,
) -> AsyncIterator[dict[str, Any]]:
    """Emits an OpenAI chunk per fragment of text.

    Under a schema the CLI may answer several times, as plain text and then
    as one or more structured answers. Only the last answer is the reply, so
    it is sent once the run ends rather than streamed as it arrives.
    """
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
    tool_calls: list[dict[str, Any]] = []
    answer = ""
    said_anything = False
    async for event in events:
        if not schema_mode:
            text = _text_fragment(event)
            if text:
                said_anything = True
                yield chunk({"content": text}, None)
        if event.get("type") == "result":
            usage = _usage(event)
            tool_calls = _tool_calls(event)
            answer = _answer(event)
    if not said_anything and answer:
        yield chunk({"content": answer}, None)
    for index, call in enumerate(tool_calls):
        yield chunk({"tool_calls": [{"index": index, **call}]}, None)
    yield chunk({}, "tool_calls" if tool_calls else "stop")
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


def _json_fragment(event: dict[str, Any]) -> str:
    """A fragment of the schema-validated answer, still mid-JSON."""
    if event.get("type") != "stream_event":
        return ""
    inner = event.get("event", {})
    if inner.get("type") != "content_block_delta":
        return ""
    delta = inner.get("delta", {})
    return delta.get("partial_json", "") if delta.get("type") == "input_json_delta" else ""

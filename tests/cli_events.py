"""Claude CLI stream-json events, shaped as the real CLI emits them."""

from typing import Any

from tests.models import MODEL


def assistant_text_event(text: str, model: str = MODEL) -> dict[str, Any]:
    """The CLI's `assistant` event, as emitted by `--output-format stream-json`."""
    return {
        "type": "assistant",
        "message": {
            "model": model,
            "id": "msg_01Test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "stop_reason": None,
        },
    }


def result_event(
    text: str,
    *,
    input_tokens: int = 12,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    output_tokens: int = 3,
) -> dict[str, Any]:
    """The CLI's terminating `result` event."""
    return {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": text,
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": input_tokens,
            "cache_creation_input_tokens": cache_creation_input_tokens,
            "cache_read_input_tokens": cache_read_input_tokens,
            "output_tokens": output_tokens,
        },
    }


def text_delta_event(text: str) -> dict[str, Any]:
    """A streamed fragment of assistant text."""
    return {
        "type": "stream_event",
        "event": {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text},
        },
    }


def structured_result_event(
    content: str, tool_calls: list[dict[str, Any]]
) -> dict[str, Any]:
    """A `result` event carrying schema-validated structured output."""
    event = result_event(content)
    event["structured_output"] = {"content": content, "tool_calls": tool_calls}
    event["stop_reason"] = "tool_use"
    return event

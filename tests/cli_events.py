"""Claude CLI stream-json events, shaped as the real CLI emits them."""

from typing import Any


def assistant_text_event(text: str) -> dict[str, Any]:
    """The CLI's `assistant` event, as emitted by `--output-format stream-json`."""
    return {
        "type": "assistant",
        "message": {
            "model": "claude-opus-5",
            "id": "msg_01Test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "stop_reason": None,
        },
    }


def result_event(text: str) -> dict[str, Any]:
    """The CLI's terminating `result` event."""
    return {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": text,
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 12,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": 3,
        },
    }

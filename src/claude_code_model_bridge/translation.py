"""Translates between the OpenAI protocol Hermes speaks and the CLI's events."""

import time
import uuid
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

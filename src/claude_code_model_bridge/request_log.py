"""Records what each request cost, without recording what it said."""

import json
import logging
import time
from typing import Any

logger = logging.getLogger("claude_code_model_bridge")


class RequestRecord:
    """Collects one request's facts and writes them when it ends.

    Records model, duration, tokens and outcome. Never prompts or answers.
    """

    def __init__(
        self,
        model: str,
        tools: bool,
        streaming: bool,
        ignored: list[str] | None = None,
    ) -> None:
        self._facts: dict[str, Any] = {
            "model": model,
            "tools": tools,
            "streaming": streaming,
            "outcome": "incomplete",
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
        if ignored:
            self._facts["ignored"] = ignored
        self._started = time.monotonic()

    def note_usage(self, usage: dict[str, Any]) -> None:
        """Records the token counts a finished call reported."""
        self._facts["prompt_tokens"] = usage.get("prompt_tokens", 0)
        self._facts["completion_tokens"] = usage.get("completion_tokens", 0)

    def finished(self, outcome: str = "success") -> None:
        """Writes the record. Called once, however the request ended."""
        self._facts["outcome"] = outcome
        self._facts["duration_ms"] = int((time.monotonic() - self._started) * 1000)
        logger.info(json.dumps(self._facts))

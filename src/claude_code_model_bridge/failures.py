"""Turns a failed run into the status a caller knows how to act on."""

from dataclasses import dataclass
from typing import Any

from claude_code_model_bridge.claude_process import (
    ApiKeyPresent,
    ClaudeFailed,
    ClaudeTimedOut,
)


@dataclass(frozen=True)
class Failure:
    """One failure, described the way an OpenAI client expects to read it."""

    status: int
    message: str
    type: str
    code: str

    def body(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.type,
                "code": self.code,
                "param": None,
            }
        }


def failure_from(error: Exception) -> Failure:
    """Classifies a run that raised instead of finishing."""
    if isinstance(error, ApiKeyPresent):
        return Failure(
            status=500,
            message=(
                "The bridge runs on a Claude subscription and refuses to run while "
                "a billable credential is present. Unset ANTHROPIC_API_KEY and "
                f"ANTHROPIC_AUTH_TOKEN, then restart it. ({error})"
            ),
            type="api_error",
            code="billable_credential_present",
        )
    if isinstance(error, ClaudeTimedOut):
        return Failure(
            status=504, message=str(error), type="api_error", code="timeout"
        )
    if isinstance(error, ClaudeFailed):
        return Failure(
            status=502, message=str(error), type="api_error", code="upstream_failure"
        )
    raise error


def failure_in(events: list[dict[str, Any]]) -> Failure | None:
    """Classifies a finished run, or returns None when it succeeded."""
    result = next(
        (event for event in reversed(events) if event.get("type") == "result"), None
    )
    if result is None or not result.get("is_error"):
        return None
    return _classify(str(result.get("result", "")))


def _classify(message: str) -> Failure:
    if "/login" in message or "not logged in" in message.lower():
        return Failure(
            status=401,
            message=message,
            type="authentication_error",
            code="not_logged_in",
        )
    return Failure(
        status=502, message=message, type="api_error", code="upstream_failure"
    )

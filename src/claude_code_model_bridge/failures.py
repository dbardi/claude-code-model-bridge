"""Turns a failed run into the status a caller knows how to act on."""

from dataclasses import dataclass
from typing import Any


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

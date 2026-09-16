"""Turns a failed run into the status a caller knows how to act on."""

import time
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
    retry_after: int | None = None

    def headers(self) -> dict[str, str]:
        """Tells a caller when to come back, when that is knowable."""
        return {} if self.retry_after is None else {"Retry-After": str(self.retry_after)}

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
    reported = _reported_status(result)
    if reported is not None:
        return reported
    return _classify(str(result.get("result", "")), _resets_at(events))


def _reported_status(result: dict[str, Any]) -> Failure | None:
    """Uses the status the run reported, when it reported one.

    The CLI names the cause of some failures outright, which beats
    inferring it from wording that changes between releases.
    """
    if result.get("api_error_status") != 404:
        return None
    return Failure(
        status=404,
        message=str(result.get("result", "")),
        type="invalid_request_error",
        code="model_not_found",
    )


def _seconds_until(resets_at: int | None) -> int | None:
    """Converts a reset time into a wait, never asking for a wait of zero."""
    if resets_at is None:
        return None
    return max(1, int(resets_at - time.time()))


def _resets_at(events: list[dict[str, Any]]) -> int | None:
    """When the usage window reopens, if the run reported it as closed.

    Every run reports the window, including successful ones, so only a
    window that is not currently allowing work says anything useful here.
    """
    for event in reversed(events):
        if event.get("type") != "rate_limit_event":
            continue
        info = event.get("rate_limit_info", {})
        if info.get("status") != "allowed":
            return info.get("resetsAt")
    return None


def _classify(message: str, resets_at: int | None) -> Failure:
    lowered = message.lower()
    if resets_at is not None or "usage limit" in lowered or "rate limit" in lowered:
        return Failure(
            status=429,
            message=message,
            type="rate_limit_error",
            code="usage_limit_reached",
            retry_after=_seconds_until(resets_at),
        )
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

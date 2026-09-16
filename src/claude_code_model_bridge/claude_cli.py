"""The Claude seam: what a caller hands the CLI, and what comes back."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Turn:
    """One conversation turn as the CLI's stream-json input expects it.

    Each image is a (media type, base64 data) pair.
    """

    role: str
    text: str
    images: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Invocation:
    """Everything needed to run Claude once.

    `output_schema` is None when no tools were declared.
    """

    model: str
    turns: tuple[Turn, ...]
    output_schema: dict[str, Any] | None = None
    system_prompt: str = ""
    effort: str | None = None


class ClaudeCli(Protocol):
    """Runs one invocation, yielding the CLI's stream-json events as they arrive."""

    def run(self, invocation: Invocation) -> AsyncIterator[dict[str, Any]]: ...

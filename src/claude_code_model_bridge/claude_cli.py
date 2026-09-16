"""The Claude seam: what a caller hands the CLI, and what comes back."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Turn:
    """One conversation turn as the CLI's stream-json input expects it."""

    role: str
    text: str


@dataclass(frozen=True)
class Invocation:
    """Everything needed to run Claude once."""

    model: str
    turns: tuple[Turn, ...]


class ClaudeCli(Protocol):
    """Runs one invocation, yielding the CLI's stream-json events as they arrive."""

    def run(self, invocation: Invocation) -> AsyncIterator[dict[str, Any]]: ...

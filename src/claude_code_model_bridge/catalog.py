"""The models this bridge offers, and how each maps onto a CLI model."""

from dataclasses import dataclass
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelEntry:
    """One offered model.

    `context_length` is advertised to callers, some of which refuse to start
    against a model whose window looks too small to work in.
    """

    id: str
    cli_model: str
    context_length: int
    supports_vision: bool = False


class ModelCatalog:
    """Answers what models exist, and what each one means to the CLI."""

    def __init__(self, entries: list[ModelEntry]) -> None:
        self._entries = entries

    @classmethod
    def from_yaml(cls, text: str) -> "ModelCatalog":
        document = yaml.safe_load(text) or {}
        return cls([ModelEntry(**entry) for entry in document.get("models", [])])

    def listing(self) -> dict[str, Any]:
        """The catalog as an OpenAI model list."""
        return {
            "object": "list",
            "data": [
                {
                    "id": entry.id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "claude-code-model-bridge",
                    "context_length": entry.context_length,
                    "context_window": entry.context_length,
                }
                for entry in self._entries
            ],
        }

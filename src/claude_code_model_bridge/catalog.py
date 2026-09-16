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
    aliases: tuple[str, ...] = ()

    def answers_to(self, name: str) -> bool:
        return name == self.id or name in self.aliases


EFFORTS = ("low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True)
class Resolution:
    """What a requested model id means once resolved."""

    cli_model: str
    effort: str | None = None


class UnknownModel(Exception):
    """Raised when a requested id names no model this bridge can run."""


class ModelCatalog:
    """Answers what models exist, and what each one means to the CLI."""

    def __init__(
        self, entries: list[ModelEntry], passthrough_prefixes: tuple[str, ...] = ()
    ) -> None:
        self._entries = entries
        self._passthrough_prefixes = passthrough_prefixes

    @classmethod
    def from_yaml(cls, text: str) -> "ModelCatalog":
        document = yaml.safe_load(text) or {}
        return cls(
            entries=[
                ModelEntry(**{**entry, "aliases": tuple(entry.get("aliases", ()))})
                for entry in document.get("models", [])
            ],
            passthrough_prefixes=tuple(document.get("passthrough_prefixes", ())),
        )

    def resolve(self, model_id: str) -> Resolution:
        """Resolves a requested id to a CLI model, and a thinking depth if asked for.

        An id may carry an effort suffix (`some-model:high`). The suffix is
        the more specific request, so it wins over any effort the caller sent
        as a separate field.
        """
        name, effort = self._split_effort(model_id)
        for entry in self._entries:
            if entry.answers_to(name):
                return Resolution(cli_model=entry.cli_model, effort=effort)
        if name.startswith(self._passthrough_prefixes):
            return Resolution(cli_model=name, effort=effort)
        raise UnknownModel(model_id)

    def _split_effort(self, model_id: str) -> tuple[str, str | None]:
        name, separator, suffix = model_id.rpartition(":")
        if not separator:
            return model_id, None
        if suffix not in EFFORTS:
            raise UnknownModel(model_id)
        return name, suffix

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

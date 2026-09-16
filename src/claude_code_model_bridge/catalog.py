"""The models this bridge offers, and how each maps onto a CLI model."""

from dataclasses import dataclass
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelEntry:
    """One offered model. `context_length` is advertised to callers."""

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
        """Resolves an id to a CLI model, plus the effort named by a `:suffix`."""
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
        return {"object": "list", "data": [_listed(entry) for entry in self._entries]}


def _listed(entry: ModelEntry) -> dict[str, Any]:
    """One model as a listing entry.

    Each ability is spelled several ways, since clients read different
    fields. An ability the CLI cannot offer says `false`, because an absent
    field reads as unknown.
    """
    modalities = ["text", "image"] if entry.supports_vision else ["text"]
    capabilities = ["tools", "streaming"]
    if entry.supports_vision:
        capabilities.append("vision")
    return {
        "id": entry.id,
        "object": "model",
        "created": 0,
        "owned_by": "claude-code-model-bridge",
        "context_length": entry.context_length,
        "context_window": entry.context_length,
        "max_input_tokens": entry.context_length,
        "supports_vision": entry.supports_vision,
        "supports_function_calling": True,
        "supports_streaming": True,
        "supports_temperature": False,
        "supports_max_tokens": False,
        "supports_response_format": False,
        "supports_prompt_caching": False,
        "capabilities": capabilities,
        "effort_levels": list(EFFORTS),
        "architecture": {"input_modalities": modalities, "output_modalities": ["text"]},
    }

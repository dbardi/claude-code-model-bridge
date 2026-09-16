"""Assembles the bridge: configuration in, runnable application out."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from starlette.applications import Starlette

from claude_code_model_bridge.app import create_app
from claude_code_model_bridge.catalog import ModelCatalog
from claude_code_model_bridge.claude_cli import ClaudeCli
from claude_code_model_bridge.claude_process import ClaudeProcess

DEFAULT_CATALOG = Path(__file__).with_name("models.yaml")
"""The catalog shipped with the package, used when none is configured."""


@dataclass(frozen=True)
class Settings:
    """Configuration for one running bridge, read once at startup."""

    catalog_path: Path = field(default_factory=lambda: DEFAULT_CATALOG)
    host: str = "127.0.0.1"
    port: int = 8765
    max_concurrent: int = 4

    @classmethod
    def from_environment(cls, environment: dict[str, str] | None = None) -> "Settings":
        """Reads settings from the environment, falling back to the defaults."""
        source = os.environ if environment is None else environment
        return cls(
            catalog_path=Path(source.get("CLAUDE_BRIDGE_CATALOG", DEFAULT_CATALOG)),
            host=source.get("CLAUDE_BRIDGE_HOST", "127.0.0.1"),
            port=int(source.get("CLAUDE_BRIDGE_PORT", "8765")),
            max_concurrent=int(source.get("CLAUDE_BRIDGE_MAX_CONCURRENT", "4")),
        )


def build_application(
    settings: Settings, claude_cli: ClaudeCli | None = None
) -> Starlette:
    """Builds the application, running the real CLI unless given another."""
    return create_app(
        claude_cli=claude_cli or ClaudeProcess(),
        catalog=ModelCatalog.from_yaml(settings.catalog_path.read_text()),
        max_concurrent=settings.max_concurrent,
    )

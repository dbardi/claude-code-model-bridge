"""Starts the bridge."""

import logging

import uvicorn

from claude_code_model_bridge.runtime import Settings, build_application


def main() -> None:
    """Serves the bridge on the configured address until stopped."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = Settings.from_environment()
    uvicorn.run(
        build_application(settings),
        host=settings.host,
        port=settings.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()

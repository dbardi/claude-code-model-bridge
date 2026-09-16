"""Runs the Claude CLI: the only place that knows about subprocesses."""

import asyncio
import json
import os
from collections.abc import AsyncIterator
from typing import Any

from claude_code_model_bridge.claude_cli import Invocation


class ClaudeProcess:
    """Answers an invocation by running the CLI once."""

    def __init__(
        self, executable: str = "claude", environment: dict[str, str] | None = None
    ) -> None:
        self._executable = executable
        self._environment = dict(environment) if environment is not None else dict(os.environ)

    async def run(self, invocation: Invocation) -> AsyncIterator[dict[str, Any]]:
        """Runs the CLI, yielding each event it prints as it arrives."""
        process = await asyncio.create_subprocess_exec(
            self._executable,
            *self._arguments(invocation),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._environment,
        )
        process.stdin.close()
        async for line in process.stdout:
            text = line.decode().strip()
            if text:
                yield json.loads(text)
        await process.wait()

    def _arguments(self, invocation: Invocation) -> list[str]:
        return ["--model", invocation.model]

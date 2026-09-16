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
        process.stdin.write(self._conversation(invocation).encode())
        await process.stdin.drain()
        process.stdin.close()
        async for line in process.stdout:
            text = line.decode().strip()
            if text:
                yield json.loads(text)
        await process.wait()

    def _arguments(self, invocation: Invocation) -> list[str]:
        """Builds the command line, isolated from local configuration.

        The isolation flags are load-bearing rather than tidiness: without
        them every call also loads the machine's connectors, hooks and plugin
        context, which on a subscription is spent from the usage window. See
        docs/adr/0005.
        """
        return [
            "--print",
            "--setting-sources",
            "",
            "--strict-mcp-config",
            "--tools",
            "",
            "--no-session-persistence",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--model",
            invocation.model,
        ]

    def _conversation(self, invocation: Invocation) -> str:
        """Renders the turns as the stream-json input the CLI reads from stdin."""
        return "".join(
            json.dumps(
                {
                    "type": turn.role,
                    "message": {
                        "role": turn.role,
                        "content": self._content(turn),
                    },
                }
            )
            + "\n"
            for turn in invocation.turns
        )

    def _content(self, turn) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": data},
            }
            for media_type, data in turn.images
        ]
        blocks.append({"type": "text", "text": turn.text})
        return blocks

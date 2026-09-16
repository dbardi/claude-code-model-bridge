"""Runs the Claude CLI: the only place that knows about subprocesses."""

import asyncio
import json
import os
import signal
from collections.abc import AsyncIterator
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from claude_code_model_bridge.claude_cli import Invocation

GRACE_SECONDS = 5
"""How long a terminated run has to exit before it is killed outright."""

BILLABLE_CREDENTIALS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
"""Credentials that would move spending from the subscription onto paid API billing."""

REDIRECTS = (
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
)
"""Variables that would send calls somewhere other than the subscription login."""


class ApiKeyPresent(Exception):
    """Raised when the environment offers credentials that could be billed."""


class ClaudeFailed(Exception):
    """Raised when the CLI exits without producing an answer."""


class ClaudeTimedOut(ClaudeFailed):
    """Raised when a run outlives its limits and is stopped."""


class ClaudeProcess:
    """Answers an invocation by running the CLI once."""

    def __init__(
        self,
        executable: str = "claude",
        environment: dict[str, str] | None = None,
        total_seconds: float = 900,
        silence_seconds: float = 300,
    ) -> None:
        self._total_seconds = total_seconds
        self._silence_seconds = silence_seconds
        self._executable = executable
        source = dict(environment) if environment is not None else dict(os.environ)
        self._environment = {
            name: value for name, value in source.items() if name not in REDIRECTS
        }

    async def run(self, invocation: Invocation) -> AsyncIterator[dict[str, Any]]:
        """Runs the CLI, yielding each event it prints as it arrives."""
        self._refuse_billable_credentials()
        with TemporaryDirectory(prefix="claude-bridge-") as workspace:
            prompt_file = Path(workspace) / "system-prompt.txt"
            prompt_file.write_text(invocation.system_prompt)
            process = await asyncio.create_subprocess_exec(
                self._executable,
                *self._arguments(invocation, prompt_file),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._environment,
                cwd=workspace,
                start_new_session=True,
            )
            try:
                process.stdin.write(self._conversation(invocation).encode())
                await process.stdin.drain()
                process.stdin.close()
                answered = False
                async for line in self._lines(process):
                    text = line.decode().strip()
                    if text:
                        answered = True
                        yield json.loads(text)
                if await process.wait() != 0 and not answered:
                    stderr = (await process.stderr.read()).decode().strip()
                    raise ClaudeFailed(
                        stderr or f"claude exited with {process.returncode}"
                    )
            finally:
                await self._stop(process)

    async def _lines(self, process: asyncio.subprocess.Process) -> AsyncIterator[bytes]:
        """Reads the CLI's output, giving up if it overruns or goes quiet."""
        deadline = asyncio.get_running_loop().time() + self._total_seconds
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise ClaudeTimedOut(f"claude ran longer than {self._total_seconds}s")
            try:
                line = await asyncio.wait_for(
                    process.stdout.readline(), timeout=min(remaining, self._silence_seconds)
                )
            except TimeoutError:
                raise ClaudeTimedOut(
                    f"claude produced nothing for {self._silence_seconds}s"
                ) from None
            if not line:
                return
            yield line

    async def _stop(self, process: asyncio.subprocess.Process) -> None:
        """Ends the run, terminating the whole process group."""
        if process.returncode is not None:
            return
        group = os.getpgid(process.pid)
        os.killpg(group, signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), timeout=GRACE_SECONDS)
        except TimeoutError:
            os.killpg(group, signal.SIGKILL)
            await process.wait()

    def _refuse_billable_credentials(self) -> None:
        """Refuses to run when the environment offers a billable credential."""
        offered = [name for name in BILLABLE_CREDENTIALS if self._environment.get(name)]
        if offered:
            raise ApiKeyPresent(
                f"{', '.join(offered)} is set. This bridge runs on a Claude "
                "subscription and refuses to run where API billing is possible."
            )

    def _arguments(self, invocation: Invocation, prompt_file: Path) -> list[str]:
        """Builds the command line, isolated from local configuration (docs/adr/0005)."""
        arguments = [
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
            "--system-prompt-file",
            str(prompt_file),
        ]
        if invocation.output_schema is not None:
            arguments += ["--json-schema", json.dumps(invocation.output_schema)]
        if invocation.effort:
            arguments += ["--effort", invocation.effort]
        return arguments

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

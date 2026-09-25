"""Records whole requests and what they became, for diagnosing a bad answer.

Off unless `CLAUDE_BRIDGE_DEBUG_DIR` names a directory. Dumps contain the
conversation in full, so they are opt-in and belong somewhere private.
"""

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from claude_code_model_bridge.claude_cli import Invocation


def dump_directory() -> Path | None:
    configured = os.environ.get("CLAUDE_BRIDGE_DEBUG_DIR")
    return Path(configured) if configured else None


def record_request(body: dict[str, Any], invocation: Invocation) -> None:
    """Writes one request and the invocation built from it."""
    directory = dump_directory()
    if directory is None:
        return
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{time.strftime('%H%M%S')}-{uuid.uuid4().hex[:6]}.json"
    (directory / name).write_text(
        json.dumps(
            {
                "request": body,
                "invocation": {
                    "model": invocation.model,
                    "effort": invocation.effort,
                    "system_prompt": invocation.system_prompt,
                    "output_schema": invocation.output_schema,
                    "turns": [
                        {"role": turn.role, "text": turn.text, "images": len(turn.images)}
                        for turn in invocation.turns
                    ],
                },
            },
            indent=2,
        )
    )

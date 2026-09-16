"""Behavior at the Claude seam: how a conversation reaches the CLI."""

from tests.cli_events import result_event
from tests.models import MODEL


async def test_system_messages_become_the_system_prompt(bridge):
    client, claude = bridge([result_event("ok")])

    await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a terse assistant."},
            {"role": "user", "content": "Say ok."},
        ],
    )

    invocation = claude.invocations[0]
    assert invocation.system_prompt == "You are a terse assistant."
    assert [turn.role for turn in invocation.turns] == ["user"]

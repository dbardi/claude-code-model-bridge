"""Behavior at the HTTP seam: what Hermes sees when it asks for a completion."""

from tests.cli_events import assistant_text_event, result_event


async def test_answers_with_the_text_claude_produced(bridge):
    client, _ = bridge([assistant_text_event("pong"), result_event("pong")])

    completion = await client.chat.completions.create(
        model="claude-opus-5",
        messages=[{"role": "user", "content": "Reply with exactly: pong"}],
    )

    assert completion.choices[0].message.content == "pong"

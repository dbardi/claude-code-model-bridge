"""Behavior at the HTTP seam: what a client sees when it asks for a completion."""

from tests.cli_events import assistant_text_event, result_event, text_delta_event


async def test_answers_with_the_text_claude_produced(bridge):
    client, _ = bridge([assistant_text_event("pong"), result_event("pong")])

    completion = await client.chat.completions.create(
        model="claude-opus-5",
        messages=[{"role": "user", "content": "Reply with exactly: pong"}],
    )

    assert completion.choices[0].message.content == "pong"


async def test_streams_the_text_in_fragments_as_claude_produces_it(bridge):
    client, _ = bridge(
        [
            text_delta_event("Hello there"),
            text_delta_event(", nice to"),
            text_delta_event(" meet you!"),
            result_event("Hello there, nice to meet you!"),
        ]
    )

    stream = await client.chat.completions.create(
        model="claude-opus-5",
        messages=[{"role": "user", "content": "Greet me."}],
        stream=True,
    )
    fragments = [chunk.choices[0].delta.content async for chunk in stream]

    assert "".join(f for f in fragments if f) == "Hello there, nice to meet you!"

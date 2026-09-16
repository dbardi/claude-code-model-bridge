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


async def test_tool_history_is_replayed_as_text(bridge):
    client, claude = bridge([result_event("467G free")])

    await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": "How much disk is free?"},
            {
                "role": "assistant",
                "content": "Checking.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "terminal",
                            "arguments": '{"command": "df -h /home"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "467G available"},
        ],
    )

    turns = claude.invocations[0].turns
    assert turns[1].role == "assistant"
    assert "Checking." in turns[1].text
    assert '[tool_call id=call_1 name=terminal arguments={"command": "df -h /home"}]' in turns[1].text
    assert turns[2].role == "user"
    assert "[tool_result id=call_1 name=terminal]" in turns[2].text
    assert "467G available" in turns[2].text


async def test_consecutive_turns_from_one_role_merge(bridge):
    client, claude = bridge([result_event("done")])

    await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": "Check disk and memory."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": '{"command": "df"}'},
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": '{"command": "free"}'},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "467G free"},
            {"role": "tool", "tool_call_id": "call_2", "content": "8G free"},
        ],
    )

    turns = claude.invocations[0].turns
    assert [turn.role for turn in turns] == ["user", "assistant", "user"]
    assert "467G free" in turns[2].text
    assert "8G free" in turns[2].text


async def test_a_conversation_ending_on_an_assistant_turn_gets_a_closing_user_turn(bridge):
    client, claude = bridge([result_event("continuing")])

    await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": "Start a list."},
            {"role": "assistant", "content": "1. First item"},
        ],
    )

    turns = claude.invocations[0].turns
    assert turns[-1].role == "user"
    assert turns[-1].text

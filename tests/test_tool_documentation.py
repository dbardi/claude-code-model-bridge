"""What the model is told about the tools it may call."""

import json

from tests.cli_events import result_event
from tests.models import MODEL

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "Run a shell command and return its output",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from disk",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
]


async def ask_with_tools(bridge, tools=TOOLS):
    client, claude = bridge([result_event("ok")])
    await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": "You are Kermit."}],
        tools=tools,
    )
    return claude.invocations[0]


async def test_the_model_is_told_what_each_tool_does(bridge):
    invocation = await ask_with_tools(bridge)

    assert "Run a shell command and return its output" in invocation.system_prompt
    assert "Read a file from disk" in invocation.system_prompt


async def test_the_model_is_told_what_arguments_each_tool_takes(bridge):
    invocation = await ask_with_tools(bridge)

    assert '"command"' in invocation.system_prompt
    assert '"path"' in invocation.system_prompt


async def test_the_callers_own_prompt_comes_first(bridge):
    invocation = await ask_with_tools(bridge)

    assert invocation.system_prompt.startswith("You are Kermit.")


async def test_a_request_without_tools_says_nothing_about_tools(bridge):
    client, claude = bridge([result_event("ok")])

    await client.chat.completions.create(
        model=MODEL, messages=[{"role": "system", "content": "You are Kermit."}]
    )

    assert claude.invocations[0].system_prompt == "You are Kermit."


async def test_an_undocumented_tool_is_still_offered(bridge):
    """A caller may declare a tool with no description; it stays callable."""
    bare = [{"type": "function", "function": {"name": "mystery"}}]

    invocation = await ask_with_tools(bridge, tools=bare)

    assert "mystery" in invocation.system_prompt
    schema = invocation.output_schema
    assert schema["properties"]["tool_calls"]["items"]["properties"]["name"]["enum"] == [
        "mystery"
    ]

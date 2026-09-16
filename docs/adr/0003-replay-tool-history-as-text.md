# Replay prior tool calls as text, not as native tool-use blocks

Earlier tool calls and their results are written into the replayed conversation
as labeled text (`[tool_call id=... name=... arguments={...}]` and
`[tool_result id=...]`) rather than as Claude's native `tool_use` and
`tool_result` content blocks. Native blocks are intercepted by Claude Code's own
agent loop, which tries to execute the named tool itself: a test replaying a
`terminal` call produced a spurious "the last call failed" turn followed by an
empty second result.

A future reader will reasonably assume the native format is correct here, so
this deviation is deliberate and should not be "fixed" without re-testing that
behavior against the current CLI.

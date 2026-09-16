# Hermes executes tools; Claude Code's own tools stay off

Claude Code is a capable agent that could run tools itself, but we invoke it
with `--tools ""` so it acts purely as a model: it requests tool calls and
Hermes executes them. Letting Claude Code act would bypass Hermes's own tools,
permissions, memory, checkpoints and logging, leaving its work invisible to the
harness that is supposed to be running the conversation.

## Considered options

Letting Claude Code run its own tools and return only final text was simpler,
but Hermes would see a model that never calls a tool, and its browser, memory
and messaging tools would go unused. A hybrid, where both sides hold tools, was
rejected as two overlapping tool sets with no clear owner.

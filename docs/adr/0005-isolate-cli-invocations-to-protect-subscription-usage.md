# Isolate every CLI invocation to protect subscription usage

Every invocation passes `--setting-sources "" --strict-mcp-config --tools ""
--no-session-persistence`, and the adapter refuses to start when
`ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` is set, stripping those and the
Bedrock and Vertex switches from child environments. Without the isolation
flags, a minimal call measured roughly 426,000 prompt tokens of connector
definitions, hooks and plugin context, against roughly 456 with them; on a
subscription the scarce resource is the usage window, not dollars, so that
difference decides how long a Hermes session can run.

The environment guard exists so the adapter can only ever spend the existing CLI
login, never fall back to paid API billing.

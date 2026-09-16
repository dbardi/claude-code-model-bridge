# Serve the Claude Code CLI as an OpenAI-compatible endpoint

Hermes Agent needs a model backend, and the Claude Code CLI runs on a Claude
subscription rather than API credits. Rather than patch Hermes to add a
provider, we serve the OpenAI chat completions protocol on localhost and answer
each request by running `claude -p`, because Hermes already accepts any
OpenAI-compatible `base_url` under `provider: custom` with no API key and no
startup model check. Hermes stays unmodified and unaware, and the same adapter
would work for any other harness that speaks this protocol.

## Consequences

Hermes's expectations become our contract: streaming responses must end with a
`finish_reason`, tool calls must be native OpenAI `tool_calls`, and
`GET /v1/models` must advertise a context window of at least 64,000 tokens or
Hermes refuses to start.

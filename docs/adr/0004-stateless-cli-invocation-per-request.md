# One stateless CLI invocation per request

Each `/v1/chat/completions` request starts a fresh `claude -p` process and
replays the whole conversation, instead of keeping a long-lived Claude session
per conversation. Hermes rewrites conversation history in ways the adapter never
sees (retries, context compression, interruptions, parallel background calls),
and any cached session state would silently diverge from what Hermes believes
the conversation to be.

## Consequences

Process startup adds roughly one to three seconds per turn, and the prompt is
re-sent each time. Anthropic's server-side prompt caching absorbs most of the
re-send cost because the conversation prefix is stable.

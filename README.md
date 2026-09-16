# claude-code-model-bridge

Presents the Claude Code CLI to any OpenAI-compatible client as an ordinary
model endpoint.

Requests arrive as normal chat completions. Each one is answered by running
`claude -p`, so the work is billed to a Claude subscription rather than to API
credits. The bridge refuses to start when an API key is present, so it cannot
quietly spend anything else.

Claude Code's own tools stay off. It acts purely as a model: it requests tool
calls, and the client that made the request runs them. An agent harness
therefore keeps its own tools, permissions, memory and logging, and needs no
changes to use this.

## Requirements

- Python 3.13 or newer, and [uv](https://docs.astral.sh/uv/)
- The Claude Code CLI, logged in (`claude auth status` should report a
  subscription)
- No `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` in the environment

## Running it

```bash
uv sync
uv run claude-code-model-bridge
```

The bridge serves `http://127.0.0.1:8765/v1` and binds to loopback only.

Configuration comes from the environment:

| Variable | Default | Meaning |
| --- | --- | --- |
| `CLAUDE_BRIDGE_HOST` | `127.0.0.1` | Address to bind |
| `CLAUDE_BRIDGE_PORT` | `8765` | Port to bind |
| `CLAUDE_BRIDGE_CATALOG` | the shipped `models.yaml` | Path to a catalog file |
| `CLAUDE_BRIDGE_MAX_CONCURRENT` | `4` | How many calls may run at once |

## Pointing a client at it

Any OpenAI-compatible client works. Give it the base URL and any API key
string; the bridge ignores the key, because it authenticates as whoever the
CLI is logged in as.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8765/v1", api_key="unused")

completion = client.chat.completions.create(
    model="claude-opus-5",
    messages=[{"role": "user", "content": "Reply with exactly: pong"}],
)
```

For an agent harness that takes a provider and a base URL, configure it as a
custom OpenAI-compatible endpoint pointing at `http://127.0.0.1:8765/v1`, with
any catalog id as the model.

Two endpoints are served: `GET /v1/models` and `POST /v1/chat/completions`
(streaming and not). Streaming, native tool calls, images and token usage all
work as a client expects.

## Model ids

An id is resolved per request, so switching models needs no restart.

- **Catalog ids**, as listed by `GET /v1/models`: `claude-opus-5`,
  `claude-opus-5-1m`, `claude-sonnet-5`, `claude-haiku-4-5`,
  `claude-fable-5-1`
- **Short aliases**: `opus`, `sonnet`, `haiku`
- **An effort suffix** selects thinking depth: `opus:high`. The levels are
  `low`, `medium`, `high`, `xhigh` and `max`. A suffix beats a
  `reasoning_effort` field on the request, being the more specific request.
- **Anything else starting with `claude-`** is passed to the CLI unchanged, so
  a model released after the catalog was written is usable without editing it.
  Such a model is not listed by `GET /v1/models`, and an id that turns out not
  to exist is rejected with `404 model_not_found`.

Models are data, not code. Edit `models.yaml` (or point
`CLAUDE_BRIDGE_CATALOG` at your own) to add an entry, rename one, or declare a
different alias, then restart.

## Failures

Each failure arrives as the status that produces the right reaction, with an
OpenAI-shaped error body.

| Condition | Status |
| --- | --- |
| Subscription usage window exhausted | `429` with `Retry-After` |
| Not logged in | `401` |
| Model does not exist | `404` |
| Run crashed, or produced unreadable output | `502` |
| Run outlived its limits | `504` |
| A billable credential is present | `500` |

`Retry-After` is taken from the reset time the run reports, so a client that
honors it waits exactly as long as the window needs.

A run is stopped if it outlives its total cap or produces nothing for long
enough to look wedged, and the whole process group is ended when a client
disconnects, so abandoned work stops consuming the usage window.

## Logging

One structured line per request: model, whether tools were requested, token
counts, duration and outcome. Prompts and answers are never logged, because
real conversations pass through this.

## Why it is built this way

**Every call is isolated** with `--setting-sources "" --strict-mcp-config
--tools "" --no-session-persistence`. Without those flags a call also loads
the machine's connectors, hooks and plugin context: a minimal request measured
roughly 426,000 prompt tokens instead of roughly 456. On a subscription the
scarce resource is the usage window, so that difference decides how long a
session can run.

**Tool history is replayed as text**, not as Claude's native tool-use blocks.
Native blocks are intercepted by Claude Code's own agent loop, which tries to
execute the named tool itself and derails the turn. The native format looks
correct here, so this deviation is deliberate.

**Each request runs its own CLI process**, with no session kept between
requests. Clients rewrite conversation history in ways the bridge never sees,
such as retries, context compression and interruptions, and cached session
state would silently diverge from what the client believes the conversation
to be.

The decisions behind the design are recorded in [`docs/adr/`](docs/adr/).

## Development

```bash
uv run pytest           # the full suite; spends no subscription usage
uv run pytest -m live   # checks the real CLI still behaves as assumed
```

The `live` tests are skipped by default because each one spends subscription
usage. They exist to catch a CLI release changing behavior the bridge depends
on, which no fake can detect.

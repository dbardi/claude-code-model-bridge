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
| `CLAUDE_BRIDGE_PLUGIN_DIR` | unset | Directory of plugins whose skills the model may use |

## Installing it as a service

On a machine with systemd, `scripts/install.sh` installs the bridge as a user
service that starts at boot and restarts if it stops.

```bash
./scripts/install.sh --check      # check prerequisites, change nothing
./scripts/install.sh              # install and start the service
./scripts/install.sh --uninstall  # stop and remove it
```

The script installs only the service. It never edits the configuration of
whatever client will use the bridge, so a failed install cannot take your
assistant offline; pointing a client at the bridge stays a separate,
deliberate step.

Afterwards:

```bash
systemctl --user status claude-model-bridge
journalctl --user -u claude-model-bridge -f
```

Lingering must be enabled for a user service to run before you log in
(`sudo loginctl enable-linger $USER`). The script says so if it is off.

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

Two endpoints are served: `GET /v1/models` and `POST /v1/chat/completions`
(streaming and not). Streaming, native tool calls, images and token usage all
work as a client expects.

## Using it with an agent

An agent harness needs three things, whatever its configuration format:

- **Base URL** `http://127.0.0.1:8765/v1`
- **API key**: any non-empty string. The bridge ignores it and authenticates as
  whoever the CLI is logged in as. Harnesses that insist on one accept
  something like `not-used`.
- **Model**: any id from the catalog, such as `claude-opus-5`.

The harness runs the tools. The bridge asks for tool calls and never executes
one, so the harness keeps its own tools, permissions and approvals.

Two settings are worth checking in the harness rather than here. Its **request
timeout** should exceed the bridge's own cap of 15 minutes, so the bridge is
what gives up first and can explain why. Its **retry policy** should honor
`Retry-After`, which the bridge sends on `429` with the real reset time of the
subscription window.

### Adding it alongside an existing model

Prefer adding the bridge as an extra provider over replacing the harness's
default. A harness you use daily keeps working, you choose the bridge per
session, and no single edit can take your assistant offline.

### Hermes Agent

Add an entry next to the existing providers in `~/.hermes/config.yaml`:

```yaml
custom_providers:
  - base_url: http://127.0.0.1:8765/v1
    id: claude-bridge
    name: Claude Bridge
    models:
      claude-opus-5:
        context_length: 1000000
        supports_vision: true
      claude-haiku-4-5:
        context_length: 200000
        supports_vision: true
```

Then choose it per session, leaving the default untouched:

```bash
hermes --provider claude-bridge -m claude-opus-5
hermes -z "summarize this repo" --provider claude-bridge -m claude-haiku-4-5
```

To make it the default instead, set `model.provider` to `custom`,
`model.base_url` to the bridge, and `model.default` to a catalog id. Back up
the file first: that replaces whatever model the harness used before.

Hermes reads `context_length` from `GET /v1/models`, and refuses to start
against a model whose window looks smaller than 64,000 tokens, so every
catalog entry advertises a real window.

### OpenClaw

Add a provider under `models.providers` in the OpenClaw config, using the
`openai-completions` adapter:

```json5
{
  models: {
    providers: {
      "claude-bridge": {
        baseUrl: "http://127.0.0.1:8765/v1",
        apiKey: "not-used",
        api: "openai-completions",
        timeoutSeconds: 1200,
        models: [
          {
            id: "claude-opus-5",
            name: "Claude Opus 5 (bridge)",
            contextWindow: 1000000,
            maxTokens: 64000,
          },
        ],
      },
    },
  },
}
```

Reference it as `claude-bridge/claude-opus-5` wherever a model is named, such
as `agents.defaults.model.primary`.

### Anything else

A harness that speaks the OpenAI protocol needs no special support. Look for a
setting named base URL, API base, or custom or OpenAI-compatible provider,
point it at `http://127.0.0.1:8765/v1`, and give it a catalog id as the model.
Editors and libraries that accept an OpenAI base URL work the same way.

If a harness lists models by calling `GET /v1/models`, the catalog appears
there. If it expects a model it has never heard of to be declared up front,
add the id and its context window to that harness's own configuration, as in
the two examples above.

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

## Skills

Claude Code skills are off unless a plugin directory is configured:

```bash
CLAUDE_BRIDGE_PLUGIN_DIR=/path/to/plugins uv run claude-code-model-bridge
```

The directory holds plugins (each with `.claude-plugin/plugin.json` and a
`skills/` folder). The model then chooses a skill on its own when one fits.

Two things to weigh before turning this on.

**It costs tokens on every call.** One small skill measured 6,498 prompt
tokens per request against 370 with no plugin directory. Every call pays it,
whether a skill is used or not.

**Only instruction-style skills work**: writing style, review checklists,
domain knowledge. A skill that tells the model to read files or run commands
cannot act, because the CLI's own tools stay off so the client keeps tool
execution. Loading such a skill invites the model to describe work it never
did. Skills that need to act belong in the client, which runs tool calls.

Configuring a directory allows exactly one tool, `Skill`, which loads
instructions and runs nothing.

## Request fields

The bridge reads `model`, `messages`, `tools`, `tool_choice`, `stream`,
`stream_options` and `reasoning_effort`.

The CLI exposes no control for sampling, output length, or response formats,
so these have no effect: `temperature`, `top_p`, `top_k`, `max_tokens`,
`max_completion_tokens`, `seed`, `stop`, `n`, `logprobs`, `top_logprobs`,
`presence_penalty`, `frequency_penalty`, `logit_bias`, `response_format`.

Sending them is not an error, and they are not dropped quietly. A response
that ignored something names it:

```json
"ignored_parameters": ["temperature", "max_tokens"]
```

The same names appear on that request's log line. Clients ignore unknown
response fields, so this cannot break a caller.

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

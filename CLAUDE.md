# model-harness

An adapter that presents the Claude Code CLI to Hermes Agent as an
OpenAI-compatible model endpoint, running on a Claude subscription rather than
API credits. Hermes keeps tool execution; Claude Code acts purely as a model.

Start with `docs/adr/` for the decisions that shape the design.

## Agent skills

### Issue tracker

Issues live in this repo's GitHub Issues, via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage labels, used as-is. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` at the root, ADRs in `docs/adr/`. See `docs/agents/domain.md`.

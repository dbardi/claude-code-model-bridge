"""Behavior at the HTTP seam: how a requested model id reaches the CLI."""

import pytest
from openai import NotFoundError

from tests.cli_events import result_event
from tests.models import MODEL


async def test_a_catalog_id_resolves_to_its_cli_model(bridge):
    client, claude = bridge([result_event("ok")])

    await client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": "Say ok."}]
    )

    assert claude.invocations[0].model == "test-model-v1"


async def test_an_effort_suffix_selects_thinking_depth(bridge):
    client, claude = bridge([result_event("ok")])

    await client.chat.completions.create(
        model=f"{MODEL}:high", messages=[{"role": "user", "content": "Say ok."}]
    )

    assert claude.invocations[0].model == "test-model-v1"
    assert claude.invocations[0].effort == "high"


async def test_an_unknown_model_is_rejected(bridge):
    client, _ = bridge([result_event("ok")])

    with pytest.raises(NotFoundError):
        await client.chat.completions.create(
            model="no-such-model", messages=[{"role": "user", "content": "Say ok."}]
        )

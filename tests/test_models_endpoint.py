"""Behavior at the HTTP seam: what a client learns about available models."""

from tests.models import FAST_MODEL, MODEL


async def test_the_catalog_is_listed_with_its_context_windows(bridge):
    client, _ = bridge([])

    listing = await client.models.list()

    by_id = {model.id: model for model in listing.data}
    assert set(by_id) == {MODEL, FAST_MODEL}
    assert by_id[MODEL].context_length == 200000

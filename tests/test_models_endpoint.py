"""Behavior at the HTTP seam: what a client learns about available models."""

from tests.models import FAST_MODEL, MODEL


async def test_the_listing_says_what_the_bridge_can_do(bridge):
    client, _ = bridge([])

    listed = (await client.models.list()).data[0]

    assert listed.supports_function_calling is True
    assert listed.supports_streaming is True
    assert "tools" in listed.capabilities
    assert "streaming" in listed.capabilities
    assert listed.effort_levels == ["low", "medium", "high", "xhigh", "max"]


async def test_the_listing_says_what_the_bridge_cannot_do(bridge):
    """An absent field reads as unknown, so an unsupported ability says false."""
    client, _ = bridge([])

    listed = (await client.models.list()).data[0]

    assert listed.supports_temperature is False
    assert listed.supports_max_tokens is False
    assert listed.supports_response_format is False
    assert listed.supports_prompt_caching is False


async def test_the_listing_says_which_models_accept_images(bridge):
    client, _ = bridge([])

    listing = await client.models.list()

    by_id = {model.id: model for model in listing.data}
    assert by_id[MODEL].supports_vision is True
    assert by_id[FAST_MODEL].supports_vision is False
    assert "vision" in by_id[MODEL].capabilities
    assert "vision" not in by_id[FAST_MODEL].capabilities
    assert by_id[MODEL].architecture["input_modalities"] == ["text", "image"]
    assert by_id[FAST_MODEL].architecture["input_modalities"] == ["text"]


async def test_the_catalog_is_listed_with_its_context_windows(bridge):
    client, _ = bridge([])

    listing = await client.models.list()

    by_id = {model.id: model for model in listing.data}
    assert set(by_id) == {MODEL, FAST_MODEL}
    assert by_id[MODEL].context_length == 200000

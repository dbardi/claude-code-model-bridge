"""Behavior of the assembled application: the bridge as something you can run."""

import httpx
from openai import AsyncOpenAI

from claude_code_model_bridge.runtime import Settings, build_application

BASE_URL = "http://bridge.test/v1"

CATALOG = """
models:
  - id: configured-model
    cli_model: configured-model-v1
    context_length: 200000
"""


def client_for(app) -> AsyncOpenAI:
    http_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=BASE_URL
    )
    return AsyncOpenAI(api_key="unused", base_url=BASE_URL, http_client=http_client)


async def test_serves_the_catalog_named_in_configuration(tmp_path):
    catalog_file = tmp_path / "models.yaml"
    catalog_file.write_text(CATALOG)

    app = build_application(Settings(catalog_path=catalog_file))

    listing = await client_for(app).models.list()
    assert [model.id for model in listing.data] == ["configured-model"]


def test_settings_come_from_the_environment():
    settings = Settings.from_environment(
        {
            "CLAUDE_BRIDGE_CATALOG": "/somewhere/models.yaml",
            "CLAUDE_BRIDGE_PORT": "9000",
            "CLAUDE_BRIDGE_MAX_CONCURRENT": "2",
        }
    )

    assert str(settings.catalog_path) == "/somewhere/models.yaml"
    assert settings.port == 9000
    assert settings.max_concurrent == 2


def test_settings_have_usable_defaults():
    settings = Settings.from_environment({})

    assert settings.host == "127.0.0.1"
    assert settings.port == 8765
    assert settings.max_concurrent == 4
    assert settings.catalog_path.name == "models.yaml"

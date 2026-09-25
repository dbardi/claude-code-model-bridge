"""Behavior of the assembled application: the bridge as something you can run."""

import asyncio

import httpx
import yaml
from openai import AsyncOpenAI

from claude_code_model_bridge.catalog import ModelCatalog
from claude_code_model_bridge.runtime import DEFAULT_CATALOG, Settings, build_application
from tests.test_concurrency import CountingClaudeCli

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


async def test_the_configured_concurrency_cap_is_applied(tmp_path):
    catalog_file = tmp_path / "models.yaml"
    catalog_file.write_text(CATALOG)
    claude = CountingClaudeCli()

    app = build_application(
        Settings(catalog_path=catalog_file, max_concurrent=1), claude_cli=claude
    )

    client = client_for(app)
    await asyncio.gather(
        *(
            client.chat.completions.create(
                model="configured-model",
                messages=[{"role": "user", "content": "Say ok."}],
            )
            for _ in range(4)
        )
    )
    assert claude.most_at_once == 1


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


def test_the_shipped_catalog_is_usable():
    """Every shipped model must be usable by a caller that checks before starting.

    Some harnesses refuse to run against a model whose advertised window
    looks too small to work in, so an entry below that floor would make the
    bridge unusable rather than merely limited.
    """
    catalog = ModelCatalog.from_yaml(DEFAULT_CATALOG.read_text())

    models = catalog.listing()["data"]
    assert models, "the shipped catalog is empty"
    for model in models:
        assert model["context_length"] >= 64_000, model["id"]
        assert catalog.resolve(model["id"]).cli_model


def test_every_shipped_alias_reaches_its_model():
    catalog = ModelCatalog.from_yaml(DEFAULT_CATALOG.read_text())
    entries = yaml.safe_load(DEFAULT_CATALOG.read_text())["models"]

    aliased = [entry for entry in entries if entry.get("aliases")]
    assert aliased, "the shipped catalog declares no short aliases"
    for entry in aliased:
        for alias in entry["aliases"]:
            assert catalog.resolve(alias).cli_model == entry["cli_model"]


def test_a_model_released_after_this_catalog_still_works():
    """A new model should not need a catalog update to be usable."""
    document = yaml.safe_load(DEFAULT_CATALOG.read_text())
    prefixes = document.get("passthrough_prefixes", [])
    assert prefixes, "the shipped catalog declares no pass-through family"

    catalog = ModelCatalog.from_yaml(DEFAULT_CATALOG.read_text())
    unreleased = f"{prefixes[0]}something-not-out-yet"
    assert catalog.resolve(unreleased).cli_model == unreleased


def test_a_plugin_directory_can_be_configured():
    settings = Settings.from_environment({"CLAUDE_BRIDGE_PLUGIN_DIR": "/skills"})

    assert str(settings.plugin_dir) == "/skills"


def test_no_plugin_directory_is_configured_by_default():
    assert Settings.from_environment({}).plugin_dir is None


def test_settings_have_usable_defaults():
    settings = Settings.from_environment({})

    assert settings.host == "127.0.0.1"
    assert settings.port == 8765
    assert settings.max_concurrent == 4
    assert settings.catalog_path.name == "models.yaml"

"""Invented model ids: the bridge must not special-case any model name."""

MODEL = "test-model"
FAST_MODEL = "test-model-fast"

CATALOG_YAML = """
models:
  - id: test-model
    cli_model: test-model-v1
    context_length: 200000
    supports_vision: true
  - id: test-model-fast
    cli_model: test-model-fast-v1
    context_length: 100000
    supports_vision: false
"""

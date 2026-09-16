"""Request fields the CLI cannot act on."""

from typing import Any

UNSUPPORTED = (
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "max_completion_tokens",
    "seed",
    "stop",
    "n",
    "logprobs",
    "top_logprobs",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "response_format",
)
"""Fields the CLI exposes no control for, so the bridge cannot honor them."""


def ignored_in(request: dict[str, Any]) -> list[str]:
    """The unsupported fields this request carries, in the order listed above."""
    return [field for field in UNSUPPORTED if request.get(field) is not None]

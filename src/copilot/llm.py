import logging
import os
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)
TIMEOUT_SECONDS = 25.0
# Order is fallback order. Both speak the OpenAI chat format, so no SDK is needed.
PROVIDERS = (
    ("gemini", "https://generativelanguage.googleapis.com/v1beta/openai", "gemini-3.5-flash-lite",
     "GEMINI_API_KEY"),
    ("openrouter", "https://openrouter.ai/api/v1", "nvidia/nemotron-3.5-super-vl:free",
     "OPENROUTER_API_KEY"),
)


class AllProvidersFailed(Exception):
    pass


@dataclass(frozen=True)
class Completion:
    text: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int


def _call(client: httpx.Client, base_url: str, key: str, model: str, messages: list[dict]) -> dict:
    response = client.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "messages": messages, "temperature": 0},
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def complete(client: httpx.Client, messages: list[dict]) -> Completion:
    for provider, base_url, model, key_name in PROVIDERS:
        key = os.environ.get(key_name)
        if not key:
            continue
        try:
            body = _call(client, base_url, key, model, messages)
            usage = body.get("usage", {})
            text = body["choices"][0]["message"]["content"]
            if not text:  # reasoning models can spend the whole reply on reasoning
                raise ValueError("empty content")
            return Completion(
                text, provider, model,
                usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
            )
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            # Never log the exception's request details: they could carry the key.
            status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
            log.warning("provider %s failed: %s %s", provider, type(exc).__name__, status)
    raise AllProvidersFailed

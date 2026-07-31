"""Multi-provider LLM selection.

Supports OpenAI, Anthropic (Claude), Mistral, and DeepSeek. Whichever
provider(s) have an API key set in the environment are considered
"available"; a provider with a blank/unset key is ignored. If
LLM_PROVIDER names an available provider, it's used. Otherwise, if exactly
one provider is available it's used; if more than one is available, one is
selected at random (so the app degrades gracefully across environments
without needing every key configured).
"""

from __future__ import annotations

import logging
import os
import random

logger = logging.getLogger(__name__)

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-opus-5",
    "mistral": "mistral-large-latest",
    "deepseek": "deepseek-chat",
}


def _available_providers() -> dict[str, str]:
    """Map provider name -> API key, for every provider with a non-blank key."""
    candidates = {
        "openai": os.environ.get("OPENAI_API_KEY", "").strip(),
        "anthropic": os.environ.get("ANTHROPIC_API_KEY", "").strip(),
        "mistral": os.environ.get("MISTRAL_API_KEY", "").strip(),
        "deepseek": os.environ.get("DEEPSEEK_API_KEY", "").strip(),
    }
    return {name: key for name, key in candidates.items() if key}


def choose_provider() -> str:
    """Pick which provider to use, following the selection rules above."""
    available = _available_providers()
    if not available:
        raise RuntimeError(
            "No LLM provider configured. Set one of OPENAI_API_KEY, "
            "ANTHROPIC_API_KEY, MISTRAL_API_KEY, or DEEPSEEK_API_KEY."
        )

    requested = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if requested and requested in available:
        return requested

    if len(available) == 1:
        return next(iter(available))

    chosen = random.choice(list(available))
    logger.info(
        "Multiple LLM providers available (%s); randomly selected '%s'", list(available), chosen
    )
    return chosen


def _model_for(provider: str) -> str:
    return os.environ.get("LLM_MODEL", "").strip() or DEFAULT_MODELS[provider]


def get_llm(temperature: float = 0.0):
    """Return a LangChain chat model instance for the selected provider."""
    provider = choose_provider()
    model = _model_for(provider)
    api_key = _available_providers()[provider]

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, api_key=api_key, temperature=temperature)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, api_key=api_key, temperature=temperature)

    if provider == "mistral":
        from langchain_mistralai import ChatMistralAI

        return ChatMistralAI(model=model, api_key=api_key, temperature=temperature)

    if provider == "deepseek":
        # DeepSeek exposes an OpenAI-compatible API.
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://api.deepseek.com",
            temperature=temperature,
        )

    raise ValueError(f"Unsupported LLM provider: {provider!r}")

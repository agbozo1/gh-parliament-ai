"""Embedding model configuration.

Provider is selected via the EMBEDDING_PROVIDER environment variable:
- "openai": OpenAI text-embedding-3-small (requires OPENAI_API_KEY)
- "huggingface": sentence-transformers/all-MiniLM-L6-v2, runs locally with
  no API key required. This is also the automatic fallback if
  EMBEDDING_PROVIDER=openai but OPENAI_API_KEY is unset.
"""

from __future__ import annotations

import os

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
HUGGINGFACE_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    """Return a LangChain Embeddings instance for the configured provider."""
    provider = os.environ.get("EMBEDDING_PROVIDER", "huggingface").lower()

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return _huggingface_embeddings()
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL, api_key=api_key)

    if provider == "huggingface":
        return _huggingface_embeddings()

    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider!r}")


def _huggingface_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=HUGGINGFACE_EMBEDDING_MODEL)

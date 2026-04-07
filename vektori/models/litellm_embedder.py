"""
LiteLLM-based embedding provider. Supports 100+ providers via a single interface.

Install: pip install litellm

Supported embedding providers (examples):
  - "text-embedding-3-small"                            → OpenAI
  - "together_ai/togethercomputer/m2-bert-80M-8k-retrieval"  → Together AI
  - "cohere/embed-english-v3.0"                         → Cohere
  - "azure/<deployment>"                                → Azure OpenAI
  - "ollama/nomic-embed-text"                           → Ollama (local)

Usage:
    v = Vektori(embedding_model="litellm:text-embedding-3-small")
    v = Vektori(embedding_model="litellm:together_ai/togethercomputer/m2-bert-80M-8k-retrieval")
"""

from __future__ import annotations

import logging

from vektori.models.base import EmbeddingProvider

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "text-embedding-3-small"


class LiteLLMEmbedder(EmbeddingProvider):
    """LiteLLM-backed embedding provider."""

    def __init__(self, model: str | None = None, dimensions: int | None = None, **kwargs) -> None:
        self.model = model or DEFAULT_MODEL
        self._dimensions = dimensions
        self._kwargs = kwargs  # pass-through: api_key, api_base, etc.
        self._resolved_dim: int | None = None

    @property
    def dimension(self) -> int:
        return self._dimensions or self._resolved_dim or 1536

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            import litellm
        except ImportError as e:
            raise ImportError("litellm required: pip install litellm") from e

        response = await litellm.aembedding(model=self.model, input=texts, **self._kwargs)
        embeddings = [item["embedding"] for item in sorted(response.data, key=lambda x: x["index"])]
        if self._resolved_dim is None and embeddings:
            self._resolved_dim = len(embeddings[0])
        return embeddings

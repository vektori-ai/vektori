"""Anthropic embedding (via Voyage AI) and LLM providers."""

from __future__ import annotations

import logging

from vektori.models.base import EmbeddingProvider, LLMProvider

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "voyage-3"
DEFAULT_LLM_MODEL = "claude-haiku-4-5-20251001"


class AnthropicEmbedder(EmbeddingProvider):
    """
    Voyage AI embeddings (Anthropic's embedding partner).

    Install: pip install 'vektori[anthropic]'
    """

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or DEFAULT_EMBEDDING_MODEL
        self._api_key = api_key
        self._client = None

    @property
    def dimension(self) -> int:
        dims = {"voyage-3": 1024, "voyage-3-lite": 512, "voyage-3-large": 1024}
        return dims.get(self.model, 1024)

    def _get_client(self):
        if self._client is None:
            try:
                import voyageai
            except ImportError as e:
                raise ImportError(
                    "voyageai required: pip install 'vektori[anthropic]'"
                ) from e
            self._client = voyageai.AsyncClient(api_key=self._api_key)
        return self._client

    async def embed(self, text: str) -> list[float]:
        client = self._get_client()
        result = await client.embed([text], model=self.model)
        return result.embeddings[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._get_client()
        result = await client.embed(texts, model=self.model)
        return result.embeddings


class AnthropicLLM(LLMProvider):
    """Anthropic Claude for fact + insight extraction."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or DEFAULT_LLM_MODEL
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:
                raise ImportError(
                    "anthropic required: pip install 'vektori[anthropic]'"
                ) from e
            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        return self._client

    async def generate(self, prompt: str) -> str:
        client = self._get_client()
        message = await client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

"""Ollama local embedding and LLM providers. No API keys needed."""

from __future__ import annotations

import asyncio
import logging

import httpx

from vektori.models.base import EmbeddingProvider, LLMProvider

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_LLM_MODEL = "llama3"
DEFAULT_BASE_URL = "http://localhost:11434"

EMBEDDING_DIMENSIONS = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    "nomic-embed-text:latest": 768,
}


class OllamaEmbedder(EmbeddingProvider):
    """
    Ollama local embeddings. Run any embedding model locally.

    Prerequisites:
        ollama pull nomic-embed-text
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        self.model = model or DEFAULT_EMBEDDING_MODEL
        self.base_url = base_url.rstrip("/")

    @property
    def dimension(self) -> int:
        return EMBEDDING_DIMENSIONS.get(self.model, 768)

    async def embed(self, text: str) -> list[float]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()["embedding"]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Ollama has no native batch endpoint — parallelize with gather."""
        if not texts:
            return []
        return list(await asyncio.gather(*[self.embed(t) for t in texts]))


class OllamaLLM(LLMProvider):
    """
    Ollama local LLM for fact + insight extraction.

    Prerequisites:
        ollama pull llama3
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        self.model = model or DEFAULT_LLM_MODEL
        self.base_url = base_url.rstrip("/")

    async def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        payload: dict = {"model": self.model, "prompt": prompt, "stream": False}
        if max_tokens is not None:
            payload["options"] = {"num_predict": max_tokens}
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120.0,
            )
            response.raise_for_status()
            return response.json()["response"]

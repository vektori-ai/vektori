"""Abstract base classes for embedding and LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract embedding provider. One interface, multiple backends."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts. More efficient than calling embed() in a loop.
        Always use this for ingestion.
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""
        ...


class LLMProvider(ABC):
    """Abstract LLM provider for fact and insight extraction."""

    @abstractmethod
    async def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        """
        Generate a completion for the given prompt.
        Should return valid JSON for extraction prompts.
        max_tokens: cap output length. None = provider default.
        """
        ...

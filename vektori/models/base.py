"""Abstract base classes for embedding, extraction, and chat providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


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
    """Abstract LLM provider for fact and episode extraction."""

    @abstractmethod
    async def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        """
        Generate a completion for the given prompt.
        Should return valid JSON for extraction prompts.
        max_tokens: cap output length. None = provider default.
        """
        ...


@dataclass
class ChatCompletionResult:
    """Normalized chat completion result for the agent harness."""

    content: str | None
    tool_calls: list[dict[str, Any]]
    raw_response: Any | None = None
    usage: dict[str, int] | None = None


class ChatModelProvider(ABC):
    """Abstract chat model provider for conversational turns."""

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatCompletionResult:
        """Generate a chat completion from role-aware messages."""
        ...

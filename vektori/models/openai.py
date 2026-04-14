"""OpenAI embedding, extraction, and chat providers."""

from __future__ import annotations

import logging

from vektori.models.base import (
    ChatCompletionResult,
    ChatModelProvider,
    EmbeddingProvider,
    LLMProvider,
)

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_LLM_MODEL = "gpt-4o-mini"

EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder(EmbeddingProvider):
    """OpenAI text embeddings. Default: text-embedding-3-small (1536 dim)."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or DEFAULT_EMBEDDING_MODEL
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as e:
                raise ImportError("openai package required: pip install openai") from e
            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    @property
    def dimension(self) -> int:
        return EMBEDDING_DIMENSIONS.get(self.model, 1536)

    async def embed(self, text: str) -> list[float]:
        client = self._get_client()
        response = await client.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._get_client()
        response = await client.embeddings.create(input=texts, model=self.model)
        # Sort by index to preserve order
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


class OpenAILLM(LLMProvider):
    """OpenAI chat completion for fact + episode extraction."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or DEFAULT_LLM_MODEL
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    async def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        client = self._get_client()
        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        response = await client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


class OpenAIChatModel(ChatModelProvider):
    """OpenAI chat model provider for the native harness."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or DEFAULT_LLM_MODEL
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    async def complete(
        self,
        messages: list[dict[str, object]],
        *,
        tools: list[dict[str, object]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatCompletionResult:
        client = self._get_client()
        kwargs: dict[str, object] = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = await client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        usage = None
        if response.usage is not None:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        tool_calls = []
        if getattr(message, "tool_calls", None):
            tool_calls = [call.model_dump() for call in message.tool_calls]
        return ChatCompletionResult(
            content=message.content,
            tool_calls=tool_calls,
            raw_response=response,
            usage=usage,
        )

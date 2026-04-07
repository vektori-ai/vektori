"""Azure OpenAI embedding and LLM providers."""

from __future__ import annotations

import logging
import os

from vektori.models.base import EmbeddingProvider, LLMProvider

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_API_VERSION = "2024-05-01-preview"

EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def _is_azure_foundry_v1_endpoint(endpoint: str) -> bool:
    if not endpoint:
        return False
    return "/openai/v1" in endpoint.rstrip("/").lower()


class AzureOpenAIEmbedder(EmbeddingProvider):
    """Azure OpenAI text embeddings."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
    ) -> None:
        self.model = model or DEFAULT_EMBEDDING_MODEL
        self._api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self._azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self._api_version = api_version or os.environ.get(
            "AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION
        )

        if not self._api_key:
            raise ValueError("Azure requires api_key or AZURE_OPENAI_API_KEY")
        if not self._azure_endpoint:
            raise ValueError("Azure requires azure_endpoint or AZURE_OPENAI_ENDPOINT")

        self._client = None
        self._foundry = _is_azure_foundry_v1_endpoint(self._azure_endpoint)

    def _get_client(self):
        if self._client is None:
            try:
                import openai
            except ImportError as e:
                raise ImportError("openai package required: pip install openai") from e

            if self._foundry:
                base = (
                    self._azure_endpoint
                    if self._azure_endpoint.endswith("/")
                    else f"{self._azure_endpoint}/"
                )
                self._client = openai.AsyncOpenAI(
                    base_url=base,
                    api_key=self._api_key,
                )
            else:
                self._client = openai.AsyncAzureOpenAI(
                    azure_endpoint=self._azure_endpoint,
                    api_key=self._api_key,
                    api_version=self._api_version,
                )
        return self._client

    @property
    def dimension(self) -> int:
        if self.model in EMBEDDING_DIMENSIONS:
            return EMBEDDING_DIMENSIONS[self.model]

        env_dim = os.environ.get("AZURE_EMBEDDING_DIMENSION")
        if env_dim and env_dim.isdigit():
            return int(env_dim)

        raise ValueError(
            f"Azure deployment name '{self.model}' not found in standard dimensions. "
            f"Please set the AZURE_EMBEDDING_DIMENSION environment variable (e.g., '1536') "
            f"to explicitly configure the embedding dimension for this custom deployment."
        )

    async def embed(self, text: str) -> list[float]:
        client = self._get_client()
        response = await client.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._get_client()
        response = await client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


class AzureOpenAILLM(LLMProvider):
    """Azure OpenAI chat completion for fact + episode extraction."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
    ) -> None:
        self.model = model or os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME") or DEFAULT_LLM_MODEL
        self._api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self._azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self._api_version = api_version or os.environ.get(
            "AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION
        )

        if not self._api_key:
            raise ValueError("Azure requires api_key or AZURE_OPENAI_API_KEY")
        if not self._azure_endpoint:
            raise ValueError("Azure requires azure_endpoint or AZURE_OPENAI_ENDPOINT")

        self._client = None
        self._foundry = _is_azure_foundry_v1_endpoint(self._azure_endpoint)

    def _get_client(self):
        if self._client is None:
            try:
                import openai
            except ImportError as e:
                raise ImportError("openai package required: pip install openai") from e

            if self._foundry:
                base = (
                    self._azure_endpoint
                    if self._azure_endpoint.endswith("/")
                    else f"{self._azure_endpoint}/"
                )
                self._client = openai.AsyncOpenAI(
                    base_url=base,
                    api_key=self._api_key,
                )
            else:
                self._client = openai.AsyncAzureOpenAI(
                    azure_endpoint=self._azure_endpoint,
                    api_key=self._api_key,
                    api_version=self._api_version,
                )
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

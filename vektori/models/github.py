"""GitHub Models provider (GitHub Models API at models.github.ai)."""

from __future__ import annotations

import logging
import os

import httpx

from vektori.models.base import EmbeddingProvider, LLMProvider

logger = logging.getLogger(__name__)

GITHUB_MODELS_BASE = "https://models.github.ai"
GITHUB_MODELS_CHAT_URL = f"{GITHUB_MODELS_BASE}/inference/chat/completions"
GITHUB_MODELS_EMBED_URL = f"{GITHUB_MODELS_BASE}/inference/embeddings"
GITHUB_API_VERSION = "2022-11-28"

DEFAULT_GITHUB_MODEL = "openai/gpt-4o-mini"
DEFAULT_GITHUB_EMBEDDING_MODEL = "openai/text-embedding-3-small"


def _normalize_model_id(model: str, default: str) -> str:
    """Return API model ID: publisher/name."""
    s = model.split(":", 1)[-1].strip() if ":" in model else model.strip()
    if not s or s.lower() == "copilot":
        return default
    return s


class GitHubEmbedder(EmbeddingProvider):
    """GitHub Models API embeddings adapter."""

    def __init__(self, model: str | None = None, token: str | None = None) -> None:
        self.model = _normalize_model_id(
            model or DEFAULT_GITHUB_EMBEDDING_MODEL, DEFAULT_GITHUB_EMBEDDING_MODEL
        )
        self.token = token or os.environ.get("GITHUB_TOKEN")

        if not self.token:
            raise ValueError("GitHub provider requires GITHUB_TOKEN")

    def _headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
        }

    @property
    def dimension(self) -> int:
        return 1536  # Fallback assumption for openai/text-embedding-3-small

    async def embed(self, text: str) -> list[float]:
        res = await self.embed_batch([text])
        if res:
            return res[0]
        return []

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        payload = {
            "model": self.model,
            "input": texts,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                GITHUB_MODELS_EMBED_URL,
                headers=self._headers(),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        if "data" not in data:
            return []

        items = sorted(data["data"], key=lambda x: x.get("index", 0))
        return [item["embedding"] for item in items]


class GitHubLLM(LLMProvider):
    """GitHub Models API adapter (models.github.ai)."""

    def __init__(self, model: str | None = None, token: str | None = None) -> None:
        self.model = _normalize_model_id(model or DEFAULT_GITHUB_MODEL, DEFAULT_GITHUB_MODEL)
        self.token = token or os.environ.get("GITHUB_TOKEN")

        if not self.token:
            raise ValueError("GitHub provider requires GITHUB_TOKEN")

    def _headers(self) -> dict[str, str]:
        return {
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
        }

    async def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        payload: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            # For JSON extraction, try to enforce format if the model supports it
            # Using response_format may fail on some github models, but standard for OAI
            "response_format": {"type": "json_object"},
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                resp = await client.post(
                    GITHUB_MODELS_CHAT_URL,
                    headers=self._headers(),
                    json=payload,
                )
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                # Fallback without response_format if it fails
                if e.response.status_code == 400 and "response_format" in str(e.response.text):
                    del payload["response_format"]
                    resp = await client.post(
                        GITHUB_MODELS_CHAT_URL,
                        headers=self._headers(),
                        json=payload,
                    )
                    resp.raise_for_status()
                else:
                    raise

            data = resp.json()

        choices = data.get("choices") or []
        if not choices:
            return ""

        msg = choices[0].get("message") or {}
        content = msg.get("content")
        return str(content or "") if content is not None else ""

"""OpenAI-compatible local LLM providers for vLLM and LM Studio."""

from __future__ import annotations

import os

from vektori.models.base import LLMProvider

DEFAULT_VLLM_MODEL = "Qwen/Qwen3-8B"
DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_LMSTUDIO_MODEL = "qwen3-8b"
DEFAULT_LMSTUDIO_BASE_URL = "http://localhost:1234/v1"


class OpenAICompatibleLLM(LLMProvider):
    """Chat-completion provider for OpenAI-compatible local servers."""

    def __init__(
        self,
        model: str | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.1,
        timeout: float = 180.0,
    ) -> None:
        self.model = model or DEFAULT_VLLM_MODEL
        self.base_url = (base_url or os.environ.get("OPENAI_COMPAT_BASE_URL") or DEFAULT_VLLM_BASE_URL).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_COMPAT_API_KEY") or "local"
        self.temperature = temperature
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as e:
                raise ImportError("openai package required: pip install openai") from e

            self._client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self._client

    async def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        response = await self._get_client().chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


class VLLMLLM(OpenAICompatibleLLM):
    """vLLM OpenAI-compatible server provider.

    Environment overrides:
      - VLLM_BASE_URL, default http://localhost:8000/v1
      - VLLM_API_KEY, default local
    """

    def __init__(self, model: str | None = None, **kwargs) -> None:
        kwargs.setdefault("base_url", os.environ.get("VLLM_BASE_URL") or DEFAULT_VLLM_BASE_URL)
        kwargs.setdefault("api_key", os.environ.get("VLLM_API_KEY") or "local")
        super().__init__(model or DEFAULT_VLLM_MODEL, **kwargs)


class LMStudioLLM(OpenAICompatibleLLM):
    """LM Studio OpenAI-compatible server provider.

    Environment overrides:
      - LMSTUDIO_BASE_URL, default http://localhost:1234/v1
      - LMSTUDIO_API_KEY, default lm-studio
    """

    def __init__(self, model: str | None = None, **kwargs) -> None:
        kwargs.setdefault("base_url", os.environ.get("LMSTUDIO_BASE_URL") or DEFAULT_LMSTUDIO_BASE_URL)
        kwargs.setdefault("api_key", os.environ.get("LMSTUDIO_API_KEY") or "lm-studio")
        super().__init__(model or DEFAULT_LMSTUDIO_MODEL, **kwargs)

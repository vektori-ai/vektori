"""
LiteLLM-based LLM provider. Orchestrates any model via a single interface.

LiteLLM supports 100+ providers (OpenAI, Anthropic, Ollama, Together, Groq, etc.)
using the same call signature. This is the recommended provider for extraction.

Install: pip install litellm
"""

from __future__ import annotations

import logging

from vektori.models.base import LLMProvider

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o-mini"


class LiteLLMProvider(LLMProvider):
    """
    LiteLLM-backed LLM provider for fact + insight extraction.

    Supports any model string that LiteLLM understands:
      - "gpt-4o-mini"
      - "claude-haiku-4-5-20251001"
      - "ollama/llama3"
      - "groq/llama3-8b-8192"
      - "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1"

    Usage:
        v = Vektori(extraction_model="litellm:gpt-4o-mini")
        v = Vektori(extraction_model="litellm:ollama/llama3")
        v = Vektori(extraction_model="litellm:claude-haiku-4-5-20251001")
    """

    def __init__(self, model: str | None = None, **kwargs) -> None:
        self.model = model or DEFAULT_MODEL
        self._kwargs = kwargs  # pass-through to litellm (api_key, api_base, etc.)

    async def generate(self, prompt: str) -> str:
        try:
            import litellm
        except ImportError as e:
            raise ImportError("litellm required: pip install litellm") from e

        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            **self._kwargs,
        )
        return response.choices[0].message.content or ""

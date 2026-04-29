"""
Google Gemini API provider for fact and episode extraction.

Direct integration with google-genai library.
Supports: gemini-2.5-flash, gemini-2.5-flash-lite, gemini-3-flash-preview, etc.

Install: pip install google-genai
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from typing import Any

from vektori.models.base import LLMProvider

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash-lite"

DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_BACKOFF = 1.0
DEFAULT_MAX_BACKOFF = 32.0
DEFAULT_BACKOFF_MULTIPLIER = 2.0


class GeminiLLM(LLMProvider):
    """
    Google Gemini API provider using the google-genai SDK.

    Supports all Gemini models. Reads GEMINI_API_KEY or GOOGLE_API_KEY from env.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        thinking_level: str | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        initial_backoff: float = DEFAULT_INITIAL_BACKOFF,
        max_backoff: float = DEFAULT_MAX_BACKOFF,
        json_mode: bool = True,
    ) -> None:
        self.model = model or DEFAULT_MODEL
        self._api_key = api_key
        self._thinking_level = thinking_level  # None = use model-based default
        self._json_mode = json_mode
        self._client = None
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
            except ImportError as e:
                raise ImportError("google-genai required: pip install google-genai") from e

            api_key = (
                self._api_key
                or os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
            )
            self._client = genai.Client(api_key=api_key)
        return self._client

    async def _calculate_backoff(self, attempt: int) -> float:
        backoff = min(
            self.initial_backoff * (DEFAULT_BACKOFF_MULTIPLIER**attempt),
            self.max_backoff,
        )
        return backoff * random.uniform(0.75, 1.25)

    def _thinking_config(self):
        from google.genai import types
        # Explicit override takes priority
        if self._thinking_level is not None:
            if "gemini-2.5" in self.model:
                budget = 0 if self._thinking_level in ("none", "minimal") else None
                return types.ThinkingConfig(thinking_budget=budget)
            return types.ThinkingConfig(thinking_level=self._thinking_level)
        # Model-based defaults: suppress thinking for extraction to protect token budget
        if "gemini-3" in self.model:
            return types.ThinkingConfig(thinking_level="minimal")
        if "gemini-2.5" in self.model:
            return types.ThinkingConfig(thinking_budget=0)
        return None

    async def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        from google.genai import types

        client = self._get_client()

        config_kwargs: dict[str, Any] = {"temperature": 0.1}
        if self._json_mode:
            config_kwargs["response_mime_type"] = "application/json"
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens
        thinking = self._thinking_config()
        if thinking is not None:
            config_kwargs["thinking_config"] = thinking

        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(
                    f"Gemini API call (attempt {attempt + 1}/{self.max_retries + 1}): "
                    f"model={self.model}, tokens={max_tokens}"
                )

                response = await client.aio.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(**config_kwargs),
                )

                logger.debug(f"Gemini API call succeeded on attempt {attempt + 1}")
                return response.text or ""

            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()
                is_retryable = any(
                    term in error_msg
                    for term in ["timeout", "500", "429", "503", "connection", "network", "temporarily", "try again"]
                )

                if not is_retryable or attempt >= self.max_retries:
                    logger.error(f"Gemini API call failed after {attempt + 1} attempts: {e}")
                    raise RuntimeError(
                        f"Gemini API call failed after {self.max_retries + 1} attempts: {e}"
                    ) from e

                backoff_time = await self._calculate_backoff(attempt)
                logger.warning(
                    f"Gemini API call failed (attempt {attempt + 1}): {e}. "
                    f"Retrying in {backoff_time:.1f}s..."
                )
                await asyncio.sleep(backoff_time)

        raise RuntimeError(
            f"Gemini API call failed after {self.max_retries + 1} attempts"
        ) from last_exception

    async def generate_json(self, prompt: str, max_tokens: int | None = None) -> dict[str, Any]:
        response_text = await self.generate(prompt, max_tokens)
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response as JSON: {response_text}")
            raise ValueError(f"Invalid JSON from Gemini: {e}") from e

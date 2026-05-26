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

DEFAULT_MAX_RETRIES = 8
DEFAULT_INITIAL_BACKOFF = 1.0
DEFAULT_MAX_BACKOFF = 120.0
DEFAULT_BACKOFF_MULTIPLIER = 2.0

# Module-level key rotation state shared across all GeminiLLM instances.
# Populated from GEMINI_API_KEYS (comma-separated) or falls back to GEMINI_API_KEY.
_key_pool: list[str] = []
_key_index: int = 0
_exhausted_keys: set[str] = set()  # day-quota exhausted


def _load_key_pool() -> list[str]:
    multi = os.environ.get("GEMINI_API_KEYS", "")
    if multi:
        return [k.strip() for k in multi.split(",") if k.strip()]
    single = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    return [single] if single else []


def _current_key() -> str | None:
    global _key_pool, _key_index
    if not _key_pool:
        _key_pool = _load_key_pool()
    available = [k for k in _key_pool if k not in _exhausted_keys]
    if not available:
        return None
    # keep _key_index within available pool
    _key_index = _key_index % len(available)
    return available[_key_index]


def _rotate_key(exhausted_key: str) -> str | None:
    global _key_index, _exhausted_keys, _key_pool
    _exhausted_keys.add(exhausted_key)
    available = [k for k in _key_pool if k not in _exhausted_keys]
    if not available:
        return None
    _key_index = 0
    logger.warning("Gemini key ...%s exhausted (day quota). Rotating to key ...%s", exhausted_key[-4:], available[0][-4:])
    return available[0]


def _is_day_quota_error(error_msg: str) -> bool:
    return "generateRequestsPerDayPerProjectPerModel-FreeTier".lower() in error_msg.lower() or \
           "generate_content_free_tier_requests" in error_msg.lower()


class GeminiLLM(LLMProvider):
    """
    Google Gemini API provider using the google-genai SDK.

    Supports all Gemini models. Reads GEMINI_API_KEYS (comma-separated) or
    GEMINI_API_KEY from env. Automatically rotates keys on day-quota exhaustion.
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
        self._api_key = api_key  # explicit override — bypasses pool
        self._thinking_level = thinking_level  # None = use model-based default
        self._json_mode = json_mode
        self._clients: dict[str, Any] = {}  # key -> genai.Client
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff

    def _get_client(self, api_key: str | None = None):
        try:
            from google import genai
        except ImportError as e:
            raise ImportError("google-genai required: pip install google-genai") from e

        key = api_key or self._api_key or _current_key()
        if not key:
            raise RuntimeError("No Gemini API key available")
        if key not in self._clients:
            self._clients[key] = genai.Client(api_key=key)
        return self._clients[key], key

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
                client, active_key = self._get_client()
                logger.debug(
                    f"Gemini API call (attempt {attempt + 1}/{self.max_retries + 1}): "
                    f"model={self.model}, key=...{active_key[-4:]}"
                )

                response = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config=types.GenerateContentConfig(**config_kwargs),
                    ),
                    timeout=120.0,
                )

                logger.debug(f"Gemini API call succeeded on attempt {attempt + 1}")
                return response.text or ""

            except Exception as e:
                last_exception = e
                error_msg = str(e)

                # Day-quota exhausted — rotate key immediately, don't backoff
                if _is_day_quota_error(error_msg) and not self._api_key:
                    next_key = _rotate_key(active_key)
                    if next_key is None:
                        raise RuntimeError("All Gemini API keys exhausted (day quota)") from e
                    continue  # retry immediately with new key, don't count as a retry attempt

                is_retryable = isinstance(e, (TimeoutError, asyncio.TimeoutError)) or any(
                    term in error_msg.lower()
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

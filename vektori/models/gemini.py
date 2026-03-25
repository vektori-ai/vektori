"""
Google Gemini API provider for fact and insight extraction.

Direct integration with google-generativeai library.
Supports: gemini-2-flash, gemini-2.5-flash, gemini-3-pro, etc.

Features:
- Direct API calls (no middleware)
- Automatic retry with exponential backoff
- Fault tolerance for transient errors
- Rate limiting awareness

Install: pip install google-generativeai

Usage:
    v = Vektori(extraction_model="gemini:gemini-2.5-flash-lite")
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Any

from vektori.models.base import LLMProvider

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash-lite"

# Retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_BACKOFF = 1.0  # seconds
DEFAULT_MAX_BACKOFF = 32.0  # seconds
DEFAULT_BACKOFF_MULTIPLIER = 2.0


class GeminiLLM(LLMProvider):
    """
    Google Gemini API provider for fact + insight extraction.

    Direct google-generativeai integration (not via LiteLLM).
    Supports all Gemini models: gemini-2-flash, gemini-2.5-flash, gemini-3-pro, etc.

    Features:
    - Automatic retry with exponential backoff for transient failures
    - Handles rate limiting, network errors, and API timeouts
    - Configurable max retries and backoff strategy
    - Fault-tolerant: continues despite temporary failures

    Install: pip install google-generativeai

    Usage:
        provider = GeminiLLM(model="gemini-2-5-flash-lite")
        response = await provider.generate("Extract facts from: ...", max_tokens=2000)
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        initial_backoff: float = DEFAULT_INITIAL_BACKOFF,
        max_backoff: float = DEFAULT_MAX_BACKOFF,
    ) -> None:
        """
        Initialize Gemini LLM provider with retry configuration.

        Args:
            model: Model name (e.g., "gemini-2-5-flash-lite", "gemini-2-flash", "gemini-3-pro")
                   Defaults to "gemini-2-5-flash-lite" if not specified.
            api_key: Google API key. If not provided, uses GOOGLE_API_KEY environment variable.
            max_retries: Maximum number of retry attempts (default: 3)
            initial_backoff: Initial backoff time in seconds (default: 1.0)
            max_backoff: Maximum backoff time in seconds (default: 32.0)
        """
        self.model = model or DEFAULT_MODEL
        self._api_key = api_key
        self._client = None
        
        # Retry configuration
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff

    def _get_client(self):
        """Lazy-load Gemini client."""
        if self._client is None:
            try:
                import google.generativeai as genai
            except ImportError as e:
                raise ImportError(
                    "google-generativeai required: pip install google-generativeai"
                ) from e

            # Configure API key
            genai.configure(api_key=self._api_key)
            self._client = genai

        return self._client

    async def _calculate_backoff(self, attempt: int) -> float:
        """
        Calculate backoff time with exponential backoff + jitter.
        
        Prevents thundering herd problem when many requests retry simultaneously.
        """
        backoff = min(
            self.initial_backoff * (DEFAULT_BACKOFF_MULTIPLIER ** attempt),
            self.max_backoff,
        )
        # Add jitter: ±25% randomness
        jitter = backoff * random.uniform(0.75, 1.25)
        return jitter

    async def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        """
        Generate a completion using Gemini API with automatic retry.

        Args:
            prompt: The prompt to send to Gemini
            max_tokens: Maximum tokens in response. None = use Gemini defaults.

        Returns:
            Generated text (should be valid JSON for extraction prompts)

        Raises:
            RuntimeError: If all retry attempts fail
        """
        genai = self._get_client()

        # Build request kwargs
        kwargs: dict[str, Any] = {
            "temperature": 0.1,  # Low randomness for extraction tasks
        }

        if max_tokens is not None:
            kwargs["max_output_tokens"] = max_tokens

        # Retry loop with exponential backoff
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(
                    f"Gemini API call (attempt {attempt + 1}/{self.max_retries + 1}): "
                    f"model={self.model}, tokens={max_tokens}"
                )

                # Run in executor to avoid blocking event loop
                loop = asyncio.get_event_loop()

                def _generate():
                    try:
                        model = genai.GenerativeModel(self.model)
                        response = model.generate_content(
                            prompt,
                            generation_config=genai.types.GenerationConfig(**kwargs),
                        )
                        return response.text or ""
                    except Exception as e:
                        raise e

                response_text = await loop.run_in_executor(None, _generate)
                
                # Success
                logger.debug(f"Gemini API call succeeded on attempt {attempt + 1}")
                return response_text

            except Exception as e:
                last_exception = e
                
                # Check if this is a retryable error
                error_msg = str(e).lower()
                is_retryable = any(
                    term in error_msg
                    for term in [
                        "timeout",
                        "500",
                        "429",  # Rate limit
                        "503",  # Service unavailable
                        "connection",
                        "network",
                        "temporarily",
                        "try again",
                    ]
                )
                
                # If not retryable or last attempt, raise
                if not is_retryable or attempt >= self.max_retries:
                    logger.error(
                        f"Gemini API call failed after {attempt + 1} attempts: {e}"
                    )
                    raise RuntimeError(
                        f"Gemini API call failed after {self.max_retries + 1} attempts: {e}"
                    ) from e
                
                # Calculate backoff and wait
                backoff_time = await self._calculate_backoff(attempt)
                logger.warning(
                    f"Gemini API call failed (attempt {attempt + 1}): {e}. "
                    f"Retrying in {backoff_time:.1f}s..."
                )
                await asyncio.sleep(backoff_time)

        # Should never reach here, but just in case
        raise RuntimeError(
            f"Gemini API call failed after {self.max_retries + 1} attempts"
        ) from last_exception

    async def generate_json(
        self, prompt: str, max_tokens: int | None = None
    ) -> dict[str, Any]:
        """
        Generate JSON response from Gemini with retry.

        Convenience method that parses the response as JSON.

        Args:
            prompt: Prompt that asks for JSON output
            max_tokens: Maximum tokens

        Returns:
            Parsed JSON object
        """
        response_text = await self.generate(prompt, max_tokens)

        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response as JSON: {response_text}")
            raise ValueError(f"Invalid JSON from Gemini: {e}") from e

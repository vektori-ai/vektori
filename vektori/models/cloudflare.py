"""Cloudflare Workers AI embedding provider — BGE-M3 (1024-dim).

Uses the Cloudflare Workers AI REST API directly via httpx (already a dependency).
No OpenAI SDK or other middleware needed.

Endpoint:
    POST https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/@cf/baai/bge-m3
    Authorization: Bearer {CF_API_TOKEN}
    Body: {"text": ["sentence1", "sentence2", ...]}   ← up to 100 items per call

Response:
    {"result": {"shape": [N, 1024], "data": [[...1024 floats...], ...]}, "success": true}

Env vars required:
    CLOUDFLARE_API_TOKEN
    CLOUDFLARE_ACCOUNT_ID

Usage:
    from vektori.models.factory import create_embedder
    embedder = create_embedder("cloudflare:@cf/baai/bge-m3")
    vec = await embedder.embed("hello world")   # list[float], len=1024
"""

from __future__ import annotations

import asyncio
import logging
import os

import httpx

from vektori.models.base import EmbeddingProvider

logger = logging.getLogger(__name__)

_CF_BASE = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
_BATCH_LIMIT = 100  # Cloudflare enforces ≤ 100 texts per request


class CloudflareEmbedder(EmbeddingProvider):
    """Cloudflare Workers AI embedding provider.

    Reads credentials from environment at construction time:
        CLOUDFLARE_API_TOKEN   — API token with Workers AI (Read) permission
        CLOUDFLARE_ACCOUNT_ID  — account ID from dash.cloudflare.com

    The ``model`` argument should be a Cloudflare model name like
    ``@cf/baai/bge-m3`` (default when invoked as ``cloudflare:@cf/baai/bge-m3``).
    """

    def __init__(self, model: str | None = None) -> None:
        self._model = model or "@cf/baai/bge-m3"

        token = os.environ.get("CLOUDFLARE_API_TOKEN") or ""
        account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID") or ""

        if not token:
            raise OSError(
                "CLOUDFLARE_API_TOKEN environment variable is not set. "
                "Create a token at Profile > API Tokens > Create Custom Token "
                "with permission: Account > Workers AI > Read."
            )
        if not account_id:
            raise OSError(
                "CLOUDFLARE_ACCOUNT_ID environment variable is not set. "
                "Find it on dash.cloudflare.com in the right sidebar."
            )

        self._url = _CF_BASE.format(account_id=account_id, model=self._model)
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    @property
    def dimension(self) -> int:
        return 1024

    async def embed(self, text: str) -> list[float]:
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        async with httpx.AsyncClient(timeout=60.0) as client:
            # Cloudflare enforces ≤ 100 texts per request — batch accordingly.
            for start in range(0, len(texts), _BATCH_LIMIT):
                batch = texts[start : start + _BATCH_LIMIT]
                all_embeddings.extend(await self._post_with_retry(client, batch))

        return all_embeddings

    async def _post_with_retry(
        self, client: httpx.AsyncClient, batch: list[str], max_retries: int = 3
    ) -> list[list[float]]:
        """POST a single batch with exponential backoff on 5xx errors."""
        delay = 1.0
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = await client.post(
                    self._url,
                    headers=self._headers,
                    json={"text": batch},
                )
                if resp.status_code == 200:
                    payload = resp.json()
                    if payload.get("success"):
                        return payload["result"]["data"]
                    errors = payload.get("errors") or payload
                    raise RuntimeError(f"Cloudflare Workers AI error: {errors}")
                if resp.status_code >= 500:
                    last_error = RuntimeError(
                        f"Cloudflare Workers AI returned HTTP {resp.status_code}: {resp.text[:500]}"
                    )
                    logger.warning(
                        "Cloudflare 5xx on attempt %d/%d — retrying in %.1fs",
                        attempt + 1, max_retries, delay,
                    )
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                raise RuntimeError(
                    f"Cloudflare Workers AI returned HTTP {resp.status_code}: {resp.text[:500]}"
                )
            except httpx.TransportError as e:
                last_error = e
                logger.warning(
                    "Cloudflare transport error on attempt %d/%d — retrying in %.1fs: %s",
                    attempt + 1, max_retries, delay, e,
                )
                await asyncio.sleep(delay)
                delay *= 2

        raise RuntimeError(f"Cloudflare embed failed after {max_retries} attempts") from last_error

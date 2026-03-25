"""Debounced, batched background worker for LLM fact + insight extraction."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExtractionRequest:
    messages: list[dict[str, str]]
    session_id: str
    user_id: str
    agent_id: str | None = None


class ExtractionWorker:
    """
    Debounced, per-user batched extraction worker.

    Queues requests per user key (user_id:agent_id).
    Fires after debounce_seconds of quiet OR when max_batch_size requests
    accumulate for one user — whichever comes first.
    Max 3 concurrent LLM calls via semaphore.

    add() returns immediately. Extraction runs in the background.
    """

    def __init__(
        self,
        extractor: Any,
        debounce_seconds: float = 3.0,
        max_batch_size: int = 5,
    ) -> None:
        self._extractor = extractor
        self.debounce_seconds = debounce_seconds
        self.max_batch_size = max_batch_size
        self._queue: dict[str, list[ExtractionRequest]] = defaultdict(list)
        self._timers: dict[str, asyncio.Task] = {}

    def schedule(self, request: ExtractionRequest) -> None:
        """Queue an extraction job. Non-blocking. Debounced per user."""
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                logger.warning(
                    "No running event loop — dropping extraction for session %s",
                    request.session_id,
                )
                return
        except RuntimeError:
            logger.warning(
                "No event loop — dropping extraction for session %s",
                request.session_id,
            )
            return

        key = f"{request.user_id}:{request.agent_id or ''}"
        self._queue[key].append(request)

        # Cancel existing debounce timer for this user
        existing = self._timers.get(key)
        if existing and not existing.done():
            existing.cancel()

        if len(self._queue[key]) >= self.max_batch_size:
            # Batch full — fire immediately
            self._timers[key] = asyncio.create_task(self._process(key))
        else:
            # Wait for debounce window
            self._timers[key] = asyncio.create_task(self._debounced_process(key))

    async def _debounced_process(self, key: str) -> None:
        await asyncio.sleep(self.debounce_seconds)
        await self._process(key)

    async def _process(self, key: str) -> None:
        requests = self._queue.pop(key, [])
        self._timers.pop(key, None)
        if not requests:
            return

        semaphore = asyncio.Semaphore(3)

        async def _extract_one(req: ExtractionRequest) -> None:
            async with semaphore:
                try:
                    await self._extractor.extract(
                        req.messages, req.session_id, req.user_id, req.agent_id
                    )
                except Exception as e:
                    logger.error(
                        "Extraction failed for session %s: %s", req.session_id, e
                    )

        await asyncio.gather(*[_extract_one(r) for r in requests])

    async def shutdown(self, timeout: float = 30.0) -> None:
        """Cancel pending timers and wait for any in-flight tasks."""
        pending = [t for t in self._timers.values() if not t.done()]
        for t in pending:
            t.cancel()
        if pending:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                logger.warning("Extraction worker shutdown timed out")
        self._queue.clear()
        self._timers.clear()

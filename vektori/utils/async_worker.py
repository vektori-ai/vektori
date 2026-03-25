"""Background worker for async fact + insight extraction."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class AsyncExtractionWorker:
    """
    Non-blocking background worker for LLM fact extraction.

    add() returns immediately after storing sentences.
    Extraction runs in the background via an asyncio Queue.
    This keeps ingestion latency at embedding speed, not LLM speed.
    """

    def __init__(self, extractor: Any) -> None:
        self._extractor = extractor
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._task: asyncio.Task | None = None
        self._running = False

    def schedule(
        self,
        messages: list[dict[str, str]],
        session_id: str,
        user_id: str,
        agent_id: str | None = None,
    ) -> None:
        """Schedule an extraction job. Non-blocking. Returns immediately."""
        job = {
            "messages": messages,
            "session_id": session_id,
            "user_id": user_id,
            "agent_id": agent_id,
        }
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull:
            logger.warning("Extraction queue full — dropping job for session %s", session_id)
            return

        if not self._running:
            self._start()

    def _start(self) -> None:
        """Start the background worker loop."""
        self._running = True
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._task = loop.create_task(self._worker_loop())
            else:
                logger.warning("No running event loop — async extraction unavailable")
                self._running = False
        except RuntimeError:
            logger.warning("Could not start extraction worker — no event loop")
            self._running = False

    async def _worker_loop(self) -> None:
        """Process extraction jobs from the queue one by one."""
        logger.debug("Extraction worker started")
        while self._running:
            try:
                job = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                await self._extractor.extract(
                    messages=job["messages"],
                    session_id=job["session_id"],
                    user_id=job["user_id"],
                    agent_id=job["agent_id"],
                )
            except Exception as e:
                logger.error("Extraction failed for session %s: %s", job["session_id"], e)
            finally:
                self._queue.task_done()

        self._running = False
        logger.debug("Extraction worker stopped")

    async def shutdown(self, timeout: float = 30.0) -> None:
        """Wait for pending jobs to finish, then stop the worker."""
        if not self._queue.empty():
            logger.info("Waiting for %d pending extraction jobs...", self._queue.qsize())
            try:
                await asyncio.wait_for(self._queue.join(), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning("Extraction shutdown timed out with pending jobs")

        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

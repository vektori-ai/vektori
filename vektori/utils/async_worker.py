"""Token-threshold batched background worker for LLM fact + episode extraction."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _count_tokens(messages: list[dict[str, str]]) -> int:
    """Rough token estimate: total characters / 4 (GPT-style approximation)."""
    return sum(len(m.get("content", "")) for m in messages) // 4


@dataclass
class ExtractionRequest:
    messages: list[dict[str, str]]
    session_id: str
    user_id: str
    agent_id: str | None = None
    sentence_catalog: list[dict[str, Any]] | None = None
    session_time: datetime | None = None  # when the conversation happened (for event_time on facts)


@dataclass
class _UserBuffer:
    requests: list[ExtractionRequest] = field(default_factory=list)
    token_count: int = 0


@dataclass(frozen=True)
class ScheduleResult:
    status: str
    accepted: bool


@dataclass
class WorkerStatus:
    queued_count: int = 0
    last_enqueue_time: datetime | None = None
    last_success_time: datetime | None = None
    last_failure_time: datetime | None = None
    last_error: str | None = None


class ExtractionWorker:
    """
    Token-threshold batched background worker.

    Accumulates requests per user key (user_id:agent_id) until:
      - buffered input tokens exceed token_threshold (default ~800), OR
      - debounce_seconds of silence pass (fallback for low-traffic users)

    Max 3 concurrent LLM calls via semaphore.
    schedule() returns immediately. Extraction runs in the background.

    Why token-threshold instead of message-count:
      Single "ok" turns contribute ~1 token — no extraction fires.
      3-4 real conversation turns (~800 tokens) → extraction fires naturally.
      This matches Honcho's batching design and prevents junk fact extraction.
    """

    def __init__(
        self,
        extractor: Any,
        token_threshold: int = 800,
        debounce_seconds: float = 30.0,
        max_batch_size: int = 10,
        max_concurrent_llm: int = 3,
    ) -> None:
        self._extractor = extractor
        self.token_threshold = token_threshold
        self.debounce_seconds = debounce_seconds
        self.max_batch_size = max_batch_size
        self._buffers: dict[str, _UserBuffer] = defaultdict(_UserBuffer)
        self._timers: dict[str, asyncio.Task] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent_llm)
        self._status: dict[str, WorkerStatus] = defaultdict(WorkerStatus)
        self._shutdown = False

    def schedule(self, request: ExtractionRequest) -> ScheduleResult:
        """Queue an extraction job. Non-blocking. Token-threshold per user."""
        if self._shutdown:
            return ScheduleResult(status="rejected_shutdown", accepted=False)
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                logger.warning(
                    "No running event loop — dropping extraction for session %s",
                    request.session_id,
                )
                return ScheduleResult(status="dropped_no_loop", accepted=False)
        except RuntimeError:
            logger.warning(
                "No event loop — dropping extraction for session %s",
                request.session_id,
            )
            return ScheduleResult(status="dropped_no_loop", accepted=False)

        key = f"{request.user_id}:{request.agent_id or ''}"
        buf = self._buffers[key]
        buf.requests.append(request)
        buf.token_count += _count_tokens(request.messages)
        status = self._status[key]
        status.queued_count = len(buf.requests)
        status.last_enqueue_time = datetime.now(timezone.utc)
        status.last_error = None

        # Cancel existing debounce timer
        existing = self._timers.get(key)
        if existing and not existing.done():
            existing.cancel()

        if buf.token_count >= self.token_threshold or len(buf.requests) >= self.max_batch_size:
            # Threshold reached — fire immediately
            self._timers[key] = asyncio.create_task(self._process(key))
        else:
            # Not enough signal yet — wait for debounce window (low-traffic fallback)
            self._timers[key] = asyncio.create_task(self._debounced_process(key))
        return ScheduleResult(status="queued", accepted=True)

    def get_status(self, user_id: str, agent_id: str | None = None) -> dict[str, Any]:
        key = f"{user_id}:{agent_id or ''}"
        status = self._status.get(key, WorkerStatus())
        return {
            "queued_count": status.queued_count,
            "last_enqueue_time": status.last_enqueue_time,
            "last_success_time": status.last_success_time,
            "last_failure_time": status.last_failure_time,
            "last_error": status.last_error,
        }

    async def wait_for_idle(
        self, user_id: str, agent_id: str | None = None, timeout: float = 60.0
    ) -> None:
        """Wait until queued extraction for a user scope has finished processing."""
        key = f"{user_id}:{agent_id or ''}"
        while True:
            task = self._timers.get(key)
            if task is None:
                if key in self._buffers:
                    await self._process(key)
                return
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning("Timed out waiting for extraction worker to go idle for key=%s", key)
                return
            except asyncio.CancelledError:
                if task.cancelled():
                    await asyncio.sleep(0)
                    continue
                raise

    async def _debounced_process(self, key: str) -> None:
        await asyncio.sleep(self.debounce_seconds)
        await self._process(key)

    async def _process(self, key: str) -> None:
        buf = self._buffers.pop(key, None)
        self._timers.pop(key, None)
        status = self._status[key]
        status.queued_count = 0
        if not buf or not buf.requests:
            return

        logger.debug(
            "Extraction firing for key=%s: %d requests, ~%d tokens",
            key,
            len(buf.requests),
            buf.token_count,
        )

        async def _extract_one(req: ExtractionRequest) -> None:
            async with self._semaphore:
                try:
                    await self._extractor.extract(
                        req.messages,
                        req.session_id,
                        req.user_id,
                        req.agent_id,
                        sentence_catalog=req.sentence_catalog,
                        session_time=req.session_time,
                    )
                    self._status[key].last_success_time = datetime.now(timezone.utc)
                    self._status[key].last_error = None
                except Exception as e:
                    self._status[key].last_failure_time = datetime.now(timezone.utc)
                    self._status[key].last_error = str(e)
                    logger.error("Extraction failed for session %s: %s", req.session_id, e)

        await asyncio.gather(*[_extract_one(r) for r in buf.requests])

    async def shutdown(self, timeout: float = 30.0) -> None:
        """Cancel pending timers and wait for any in-flight tasks."""
        self._shutdown = True
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
        self._buffers.clear()
        self._timers.clear()

"""Disk-persistent cache for LLM extraction results per haystack session.

Avoids re-running LLM fact extraction for sessions that appear in multiple
benchmark questions (~20% of sessions in longmemeval_s are shared across 2+
questions). Only stores fact text + metadata (no embeddings) — facts are
re-embedded on replay using the local model (free) to keep cache size small.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SessionExtractCache:
    """SQLite-backed cache: haystack_session_id → extracted facts list.

    Each entry stores the list of facts produced by the LLM extractor for one
    haystack session.  Embeddings are intentionally omitted — they are
    recomputed on replay (local sentence-transformers, no API cost) to keep
    the cache file small.
    """

    def __init__(self, cache_path: Path) -> None:
        self._path = cache_path
        self._conn = None

    async def initialize(self) -> None:
        try:
            import aiosqlite
        except ImportError as e:
            raise ImportError("aiosqlite required: pip install aiosqlite") from e

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self._path))
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS session_cache (
                session_id  TEXT PRIMARY KEY,
                facts_json  TEXT NOT NULL,
                created_at  TEXT DEFAULT (datetime('now'))
            )
        """)
        await self._conn.commit()
        logger.info("Session extract cache ready at %s (%d entries)", self._path, await self.count())

    async def has(self, session_id: str) -> bool:
        async with self._conn.execute(
            "SELECT 1 FROM session_cache WHERE session_id = ?", (session_id,)
        ) as cur:
            return await cur.fetchone() is not None

    async def get(self, session_id: str) -> dict[str, Any] | None:
        """Return cached entry or None on miss.

        Returns a dict with keys 'facts' and 'episodes'.
        Old entries (plain list) are transparently upgraded to the new format.
        """
        async with self._conn.execute(
            "SELECT facts_json FROM session_cache WHERE session_id = ?", (session_id,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        data = json.loads(row[0])
        # Backwards compat: old entries stored a plain list of facts (some as strings, some as dicts)
        if isinstance(data, list):
            # Normalise string facts to {"text": "..."} dicts for compatibility with replay_facts()
            facts = [f if isinstance(f, dict) else {"text": f} for f in data]
            return {"facts": facts, "episodes": []}
        return data

    async def put(
        self,
        session_id: str,
        facts: list[dict[str, Any]],
        episodes: list[dict[str, Any]] | None = None,
    ) -> None:
        """Store extracted facts (and optionally episodes) for a session."""
        payload = {"facts": facts, "episodes": episodes or []}
        await self._conn.execute(
            "INSERT OR REPLACE INTO session_cache (session_id, facts_json) VALUES (?, ?)",
            (session_id, json.dumps(payload)),
        )
        await self._conn.commit()

    async def count(self) -> int:
        async with self._conn.execute("SELECT COUNT(*) FROM session_cache") as cur:
            row = await cur.fetchone()
        return row[0] if row else 0

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

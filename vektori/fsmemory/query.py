"""FSQuery — semantic and path-based search over filesystem memory."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vektori.fsmemory.store import FSStore
    from vektori.models.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class FSQuery:
    def __init__(self, store: FSStore, embedder: EmbeddingProvider) -> None:
        self.store = store
        self.embedder = embedder

    async def search(
        self,
        query: str,
        user_id: str,
        mode: str = "semantic",
        path_prefix: str | None = None,
        since: datetime | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """
        Search filesystem memory. Returns same shape as Vektori.search()
        so results are directly comparable for benchmarking.

        mode: "semantic" | "path"
          - semantic: embedding similarity over all ingested facts
          - path: return all facts from files under path_prefix
        """
        if mode == "path":
            if not path_prefix:
                raise ValueError("path_prefix required for mode='path'")
            facts = await self._path_search(path_prefix, user_id, limit)
        else:
            facts = await self._semantic_search(query, user_id, limit, since=since)
            if path_prefix:
                facts = _filter_by_prefix(facts, path_prefix)

        return {
            "facts": facts,
            "sentences": [],
            "memory_found": len(facts) > 0,
        }

    async def _semantic_search(
        self,
        query: str,
        user_id: str,
        limit: int,
        since: datetime | None = None,
    ) -> list[dict[str, Any]]:
        embedding = await self.embedder.embed(query)
        raw = await self.store.db.search_facts(
            embedding=embedding,
            user_id=user_id,
            limit=limit * 3,  # fetch extra, then filter/trim
            active_only=True,
            after_date=since,
        )
        results = []
        for row in raw:
            meta = _parse_meta(row.get("metadata"))
            if meta.get("source_type") != "filesystem":
                continue
            results.append(_format_fact(row, meta))
        return results[:limit]

    async def _path_search(
        self,
        path_prefix: str,
        user_id: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        # Use a zero-vector search to get all facts, then filter by path prefix.
        # Not ideal at scale but correct and simple for SQLite.
        raw = await self.store.db.get_active_facts(user_id=user_id, limit=1000)
        results = []
        for row in raw:
            meta = _parse_meta(row.get("metadata"))
            if meta.get("source_type") != "filesystem":
                continue
            src = meta.get("source_path", "")
            if src.startswith(path_prefix):
                results.append(_format_fact(row, meta))
        return results[:limit]


def _parse_meta(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            pass
    return {}


def _filter_by_prefix(facts: list[dict], prefix: str) -> list[dict]:
    return [f for f in facts if f.get("source_path", "").startswith(prefix)]


def _format_fact(row: dict, meta: dict) -> dict:
    return {
        "id": row.get("id"),
        "text": row.get("text", ""),
        "subject": row.get("subject"),
        "source_path": meta.get("source_path"),
        "chunk_index": meta.get("chunk_index"),
        "heading": meta.get("heading"),
        "score": 1.0 - float(row.get("distance", 0.5)),
        "created_at": row.get("created_at"),
    }

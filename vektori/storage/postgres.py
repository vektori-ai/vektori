"""PostgreSQL + pgvector storage backend. Production-grade."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from vektori.storage.base import StorageBackend

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class PostgresBackend(StorageBackend):
    """
    PostgreSQL + pgvector backend.

    Setup:
        docker compose up -d
        DATABASE_URL=postgresql://vektori:vektori@localhost:5432/vektori

    Vector search uses IVFFlat index (switch to HNSW when >1M rows).
    All graph traversal (insights → facts → sentences) happens in SQL via JOINs.
    See schema.sql for the full schema.
    """

    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self._pool = None

    async def initialize(self) -> None:
        """Create connection pool and run schema migrations."""
        try:
            import asyncpg
        except ImportError as e:
            raise ImportError(
                "asyncpg required for PostgreSQL backend: pip install 'vektori[postgres]'"
            ) from e

        self._pool = await asyncpg.create_pool(self.database_url, min_size=1, max_size=10)
        schema = SCHEMA_PATH.read_text()
        async with self._pool.acquire() as conn:
            await conn.execute(schema)
        logger.info("PostgreSQL backend initialized")

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    # ── Sentences ──

    async def upsert_sentences(
        self,
        sentences: list[dict[str, Any]],
        embeddings: list[list[float]],
        user_id: str,
        agent_id: str | None = None,
    ) -> int:
        # TODO: implement
        # INSERT INTO sentences (...) VALUES (...) ON CONFLICT (content_hash) DO UPDATE SET mentions = mentions + 1
        raise NotImplementedError

    async def search_sentences(
        self,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        # TODO: implement
        # SELECT *, embedding <=> $1 AS distance FROM sentences WHERE user_id = $2 ORDER BY distance LIMIT $3
        raise NotImplementedError

    async def find_sentences_by_similarity(
        self,
        quotes: list[str],
        session_id: str,
        threshold: float = 0.75,
    ) -> list[str]:
        # TODO: embed each quote and find closest sentences within session
        raise NotImplementedError

    # ── Facts ──

    async def insert_fact(
        self,
        text: str,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        confidence: float = 1.0,
        superseded_by_target: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        # TODO: implement
        # INSERT INTO facts (id, text, embedding, user_id, agent_id, confidence, metadata)
        # If superseded_by_target: also UPDATE old fact SET superseded_by = new_id
        raise NotImplementedError

    async def search_facts(
        self,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        limit: int = 10,
        active_only: bool = True,
    ) -> list[dict[str, Any]]:
        # TODO: implement the single CTE query from spec section 5.4
        # SELECT id, text, confidence, created_at, embedding <=> $1 AS distance
        # FROM facts WHERE user_id = $2 AND is_active = true
        # ORDER BY distance LIMIT $3
        raise NotImplementedError

    async def get_active_facts(
        self,
        user_id: str,
        agent_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        # TODO: SELECT * FROM facts WHERE user_id = $1 AND is_active = true LIMIT $2
        raise NotImplementedError

    async def deactivate_fact(self, fact_id: str, superseded_by: str | None = None) -> None:
        # TODO: UPDATE facts SET is_active = false, superseded_by = $2 WHERE id = $1
        raise NotImplementedError

    async def find_fact_by_text(
        self,
        user_id: str,
        text: str,
        agent_id: str | None = None,
    ) -> dict[str, Any] | None:
        # TODO: exact text match OR vector similarity search for near-duplicate detection
        raise NotImplementedError

    async def get_supersession_chain(self, fact_id: str) -> list[dict[str, Any]]:
        # TODO: recursive CTE following superseded_by chain
        # WITH RECURSIVE chain AS (
        #   SELECT * FROM facts WHERE id = $1
        #   UNION ALL
        #   SELECT f.* FROM facts f JOIN chain c ON f.id = c.superseded_by
        # ) SELECT * FROM chain
        raise NotImplementedError

    # ── Insights ──

    async def insert_insight(
        self,
        text: str,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        # TODO: INSERT INTO insights (...)
        raise NotImplementedError

    async def get_insights_from_facts(
        self,
        fact_ids: list[str],
        active_only: bool = True,
    ) -> list[dict[str, Any]]:
        # TODO: graph traversal via insight_facts join table
        # SELECT DISTINCT i.* FROM insights i
        # JOIN insight_facts inf ON i.id = inf.insight_id
        # WHERE inf.fact_id = ANY($1) AND i.is_active = true
        raise NotImplementedError

    async def get_active_insights(
        self,
        user_id: str,
        agent_id: str | None = None,
    ) -> list[dict[str, Any]]:
        # TODO: implement
        raise NotImplementedError

    # ── Edges ──

    async def insert_edges(self, edges: list[dict[str, Any]]) -> int:
        # TODO: INSERT INTO sentence_edges (...) ON CONFLICT DO NOTHING
        raise NotImplementedError

    async def expand_session_context(
        self,
        sentence_ids: list[str],
        window: int = 3,
    ) -> list[dict[str, Any]]:
        # TODO: for each sentence, grab ±window by sentence_index within same session
        # SELECT s2.* FROM sentences s2
        # JOIN sentences src ON s2.session_id = src.session_id
        # WHERE src.id = ANY($1)
        # AND s2.sentence_index BETWEEN src.sentence_index - $2 AND src.sentence_index + $2
        raise NotImplementedError

    # ── Join tables ──

    async def insert_fact_source(self, fact_id: str, sentence_id: str) -> None:
        # TODO: INSERT INTO fact_sources (fact_id, sentence_id) ON CONFLICT DO NOTHING
        raise NotImplementedError

    async def insert_insight_fact(self, insight_id: str, fact_id: str) -> None:
        # TODO: INSERT INTO insight_facts (insight_id, fact_id) ON CONFLICT DO NOTHING
        raise NotImplementedError

    async def insert_insight_source(self, insight_id: str, sentence_id: str) -> None:
        # TODO: INSERT INTO insight_sources ON CONFLICT DO NOTHING
        raise NotImplementedError

    async def get_source_sentences(self, fact_ids: list[str]) -> list[str]:
        # TODO: SELECT sentence_id FROM fact_sources WHERE fact_id = ANY($1)
        raise NotImplementedError

    # ── Sessions ──

    async def upsert_session(
        self,
        session_id: str,
        user_id: str,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        # TODO: INSERT INTO sessions ON CONFLICT (id) DO UPDATE SET metadata = $4
        raise NotImplementedError

    async def get_session(
        self,
        session_id: str,
        user_id: str,
    ) -> dict[str, Any] | None:
        # TODO: SELECT session + all sentences ordered by turn_number, sentence_index
        raise NotImplementedError

    # ── Lifecycle ──

    async def delete_user(self, user_id: str) -> int:
        # TODO: cascade delete — sentences, facts, insights, sessions, edges via FK cascades
        raise NotImplementedError

"""PostgreSQL + pgvector storage backend. Production-grade."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from vektori.storage.base import StorageBackend

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def _vec(embedding: list[float]) -> str:
    """Serialize a Python float list to a pgvector string literal.

    asyncpg passes this as TEXT; PostgreSQL casts it via ::vector in the query.
    Example: [0.1, 0.2, 0.3] → '[0.1,0.2,0.3]'
    """
    return "[" + ",".join(map(str, embedding)) + "]"


def _row(record: Any) -> dict[str, Any]:
    """Convert an asyncpg Record to a plain dict.

    - UUID fields → str
    - datetime fields stay as datetime (used by scoring)
    - everything else passes through
    """
    d = dict(record)
    for k, v in d.items():
        if isinstance(v, uuid.UUID):
            d[k] = str(v)
    return d


class PostgresBackend(StorageBackend):
    """
    PostgreSQL + pgvector storage backend.

    Setup:
        docker compose up -d
        DATABASE_URL=postgresql://vektori:vektori@localhost:5432/vektori

    Notes:
        - Vector search via IVFFlat cosine index (switch to HNSW when >1M rows).
        - Embeddings passed as '[x,y,z]' strings; PostgreSQL casts via ::vector.
        - All graph traversal (insight_facts, fact_sources) happens in SQL — no
          application-level graph library needed.
        - The L2 fast path (search_l2_single_query) executes the full
          facts→insights→sentences pipeline in one round trip.
    """

    # Set to True so SearchPipeline can use the single-query L2 fast path.
    supports_single_query: bool = True

    def __init__(self, database_url: str, embedding_dim: int | None = None) -> None:
        self.database_url = database_url
        self.embedding_dim = embedding_dim
        self._pool = None

    def _check_dim(self, embedding: list[float], context: str = "") -> None:
        """Raise ValueError if embedding length doesn't match expected dimension."""
        if self.embedding_dim is not None and len(embedding) != self.embedding_dim:
            label = f" ({context})" if context else ""
            raise ValueError(
                f"Embedding has {len(embedding)} dimensions{label}, "
                f"expected {self.embedding_dim}. "
                f"Check that your embedding model matches the schema dimension."
            )

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Create connection pool and run schema (idempotent)."""
        try:
            import asyncpg
        except ImportError as e:
            raise ImportError(
                "asyncpg required: pip install 'vektori[postgres]'"
            ) from e

        self._pool = await asyncpg.create_pool(
            self.database_url,
            min_size=2,
            max_size=10,
        )
        schema = SCHEMA_PATH.read_text()
        async with self._pool.acquire() as conn:
            await conn.execute(schema)
        logger.info("PostgreSQL backend initialized")

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    # ── Sentences ──────────────────────────────────────────────────────────────

    async def upsert_sentences(
        self,
        sentences: list[dict[str, Any]],
        embeddings: list[list[float]],
        user_id: str,
        agent_id: str | None = None,
    ) -> int:
        """Insert sentences in a single batch. ON CONFLICT (content_hash) increments mentions."""
        from vektori.ingestion.hasher import generate_content_hash

        if not sentences:
            return 0

        if embeddings:
            self._check_dim(embeddings[0], "upsert_sentences")

        rows = [
            (
                uuid.UUID(sent["id"]),
                sent["text"],
                _vec(emb),
                user_id,
                agent_id,
                sent["session_id"],
                sent["turn_number"],
                sent["sentence_index"],
                sent.get("role", "user"),
                generate_content_hash(
                    sent["session_id"],
                    f"{sent['turn_number']}_{sent['sentence_index']}",
                    sent["text"],
                ),
            )
            for sent, emb in zip(sentences, embeddings)
        ]

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(
                    """
                    INSERT INTO sentences
                        (id, text, embedding, user_id, agent_id, session_id,
                         turn_number, sentence_index, role, content_hash)
                    VALUES
                        ($1, $2, $3::vector, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (content_hash)
                        DO UPDATE SET mentions = sentences.mentions + 1
                    """,
                    rows,
                )
        return len(rows)

    async def search_sentences(
        self,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT id, text, session_id, turn_number, sentence_index, role,
                   mentions, created_at,
                   embedding <=> $1::vector AS distance
            FROM sentences
            WHERE user_id = $2
              AND ($3::text IS NULL OR agent_id = $3)
              AND is_active = true
            ORDER BY embedding <=> $1::vector
            LIMIT $4
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, _vec(embedding), user_id, agent_id, limit)
        return [_row(r) for r in rows]

    async def find_sentences_by_similarity(
        self,
        quotes: list[str],
        session_id: str,
        threshold: float = 0.75,
    ) -> list[str]:
        """Find sentence IDs within a session that are similar to the given quotes.

        Uses trigram similarity (pg_trgm) in a single round trip via unnest + LATERAL.
        Returns up to 3 matches per quote, deduplicated.

        Requires: pg_trgm extension (enabled in schema.sql).
        """
        if not quotes:
            return []

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT s.id
                FROM unnest($1::text[]) AS q(quote)
                JOIN LATERAL (
                    SELECT id
                    FROM sentences
                    WHERE session_id = $2
                      AND similarity(text, q.quote) > $3
                    ORDER BY similarity(text, q.quote) DESC
                    LIMIT 3
                ) s ON true
                """,
                quotes,
                session_id,
                threshold,
            )
        return [str(row["id"]) for row in rows]

    # ── Facts ──────────────────────────────────────────────────────────────────

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
        self._check_dim(embedding, "insert_fact")
        fact_id = uuid.uuid4()
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO facts
                    (id, text, embedding, user_id, agent_id, confidence,
                     superseded_by, metadata)
                VALUES
                    ($1, $2, $3::vector, $4, $5, $6, $7, $8)
                """,
                fact_id,
                text,
                _vec(embedding),
                user_id,
                agent_id,
                confidence,
                uuid.UUID(superseded_by_target) if superseded_by_target else None,
                json.dumps(metadata or {}),
            )
        return str(fact_id)

    async def search_facts(
        self,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        limit: int = 10,
        active_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Vector search over facts using IVFFlat cosine index.

        Returns facts ordered by cosine distance (ascending — closest first),
        with a 'distance' field added. The scoring layer converts distance →
        similarity and combines with confidence and recency.
        """
        query = """
            SELECT id, text, confidence, mentions, created_at, metadata,
                   embedding <=> $1::vector AS distance
            FROM facts
            WHERE user_id = $2
              AND ($3::text IS NULL OR agent_id = $3)
              AND ($4::boolean = false OR is_active = true)
            ORDER BY embedding <=> $1::vector
            LIMIT $5
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                query,
                _vec(embedding),
                user_id,
                agent_id,
                active_only,
                limit,
            )
        return [_row(r) for r in rows]

    async def get_active_facts(
        self,
        user_id: str,
        agent_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT id, text, confidence, is_active, superseded_by,
                   created_at, metadata
            FROM facts
            WHERE user_id = $1
              AND ($2::text IS NULL OR agent_id = $2)
              AND is_active = true
            ORDER BY created_at DESC
            LIMIT $3
            OFFSET $4
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, user_id, agent_id, limit, offset)
        return [_row(r) for r in rows]

    async def deactivate_fact(
        self,
        fact_id: str,
        superseded_by: str | None = None,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE facts
                SET is_active = false,
                    superseded_by = $2,
                    updated_at = now()
                WHERE id = $1
                """,
                uuid.UUID(fact_id),
                uuid.UUID(superseded_by) if superseded_by else None,
            )

    async def find_fact_by_text(
        self,
        user_id: str,
        text: str,
        agent_id: str | None = None,
        similarity_threshold: float = 0.5,
    ) -> dict[str, Any] | None:
        """Find the closest active fact by text similarity for a user.

        Uses pg_trgm so conflict detection catches near-duplicates like
        "User prefers dark mode" vs "the user likes dark mode interfaces".
        Only returns a result if trigram similarity exceeds the threshold.
        """
        query = """
            SELECT id, text, confidence, is_active, superseded_by, created_at,
                   similarity(text, $3) AS sim
            FROM facts
            WHERE user_id = $1
              AND ($2::text IS NULL OR agent_id = $2)
              AND is_active = true
              AND similarity(text, $3) > $4
            ORDER BY similarity(text, $3) DESC
            LIMIT 1
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, user_id, agent_id, text, similarity_threshold)
        if not row:
            return None
        result = _row(row)
        result.pop("sim", None)
        return result

    async def get_supersession_chain(self, fact_id: str) -> list[dict[str, Any]]:
        """Recursive CTE walking the superseded_by chain.

        Returns [current_fact, ..., oldest_superseded_fact] — newest first.
        Useful for showing the full history of a fact that changed over time.
        """
        query = """
            WITH RECURSIVE chain AS (
                -- Base: start from the requested fact
                SELECT id, text, confidence, is_active, superseded_by, created_at, 0 AS depth
                FROM facts
                WHERE id = $1

                UNION ALL

                -- Recursive: follow superseded_by links
                SELECT f.id, f.text, f.confidence, f.is_active,
                       f.superseded_by, f.created_at, c.depth + 1
                FROM facts f
                INNER JOIN chain c ON f.id = c.superseded_by
                WHERE c.depth < 50  -- safety cap against cycles
            )
            SELECT id, text, confidence, is_active, superseded_by, created_at
            FROM chain
            ORDER BY depth ASC
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, uuid.UUID(fact_id))
        return [_row(r) for r in rows]

    # ── Insights ───────────────────────────────────────────────────────────────

    async def insert_insight(
        self,
        text: str,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        self._check_dim(embedding, "insert_insight")
        insight_id = uuid.uuid4()
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO insights
                    (id, text, embedding, user_id, agent_id, confidence, metadata)
                VALUES
                    ($1, $2, $3::vector, $4, $5, $6, $7)
                """,
                insight_id,
                text,
                _vec(embedding),
                user_id,
                agent_id,
                confidence,
                json.dumps(metadata or {}),
            )
        return str(insight_id)

    async def get_insights_from_facts(
        self,
        fact_ids: list[str],
        user_id: str,
        active_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Graph traversal: find all insights linked to any of the given facts.

        This is the core L1 discovery mechanism — NOT vector search.
        user_id scoping is a safety guard; fact UUIDs are globally unique,
        but this prevents any data leak if that assumption ever breaks.
        """
        if not fact_ids:
            return []

        uuid_ids = [uuid.UUID(fid) for fid in fact_ids]
        query = """
            SELECT DISTINCT i.id, i.text, i.confidence, i.is_active,
                            i.created_at, i.metadata
            FROM insights i
            INNER JOIN insight_facts inf ON i.id = inf.insight_id
            WHERE inf.fact_id = ANY($1::uuid[])
              AND i.user_id = $2
              AND ($3::boolean = false OR i.is_active = true)
            ORDER BY i.created_at DESC
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, uuid_ids, user_id, active_only)
        return [_row(r) for r in rows]

    async def get_active_insights(
        self,
        user_id: str,
        agent_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT id, text, confidence, is_active, created_at, metadata
            FROM insights
            WHERE user_id = $1
              AND ($2::text IS NULL OR agent_id = $2)
              AND is_active = true
            ORDER BY created_at DESC
            LIMIT $3
            OFFSET $4
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, user_id, agent_id, limit, offset)
        return [_row(r) for r in rows]

    # ── Edges ──────────────────────────────────────────────────────────────────

    async def insert_edges(self, edges: list[dict[str, Any]]) -> int:
        if not edges:
            return 0
        async with self._pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO sentence_edges (source_id, target_id, edge_type, weight)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT DO NOTHING
                """,
                [
                    (
                        uuid.UUID(e["source_id"]),
                        uuid.UUID(e["target_id"]),
                        e["edge_type"],
                        e.get("weight", 1.0),
                    )
                    for e in edges
                ],
            )
        return len(edges)

    async def expand_session_context(
        self,
        sentence_ids: list[str],
        window: int = 3,
    ) -> list[dict[str, Any]]:
        """For each matched sentence, retrieve ±window neighbours by sentence_index
        within the same session and same turn.

        This reconstructs the conversational context around each matched moment.
        Deduplication by sentence ID is done in Python after the query.
        """
        if not sentence_ids:
            return []

        uuid_ids = [uuid.UUID(sid) for sid in sentence_ids]
        query = """
            SELECT DISTINCT
                s2.id, s2.text, s2.session_id, s2.turn_number,
                s2.sentence_index, s2.role, s2.created_at
            FROM sentences src
            JOIN sentences s2
                ON  s2.session_id = src.session_id
                AND s2.turn_number = src.turn_number
                AND s2.sentence_index BETWEEN src.sentence_index - $2
                                          AND src.sentence_index + $2
            WHERE src.id = ANY($1::uuid[])
              AND s2.is_active = true
            ORDER BY s2.session_id, s2.turn_number, s2.sentence_index
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, uuid_ids, window)

        # Deduplicate while preserving order (DISTINCT in SQL handles cross-source
        # dupes, but keep this for safety when multiple facts share sources).
        seen: set[str] = set()
        result = []
        for row in rows:
            d = _row(row)
            if d["id"] not in seen:
                seen.add(d["id"])
                result.append(d)
        return result

    # ── Join tables ────────────────────────────────────────────────────────────

    async def insert_fact_source(self, fact_id: str, sentence_id: str) -> None:
        await self.insert_fact_sources([(fact_id, sentence_id)])

    async def insert_fact_sources(self, pairs: list[tuple[str, str]]) -> None:
        """Batch link facts to source sentences."""
        if not pairs:
            return
        async with self._pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO fact_sources (fact_id, sentence_id)
                VALUES ($1, $2)
                ON CONFLICT DO NOTHING
                """,
                [(uuid.UUID(f), uuid.UUID(s)) for f, s in pairs],
            )

    async def insert_insight_fact(self, insight_id: str, fact_id: str) -> None:
        await self.insert_insight_facts([(insight_id, fact_id)])

    async def insert_insight_facts(self, pairs: list[tuple[str, str]]) -> None:
        """Batch link insights to related facts."""
        if not pairs:
            return
        async with self._pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO insight_facts (insight_id, fact_id)
                VALUES ($1, $2)
                ON CONFLICT DO NOTHING
                """,
                [(uuid.UUID(i), uuid.UUID(f)) for i, f in pairs],
            )

    async def insert_insight_source(self, insight_id: str, sentence_id: str) -> None:
        await self.insert_insight_sources([(insight_id, sentence_id)])

    async def insert_insight_sources(self, pairs: list[tuple[str, str]]) -> None:
        """Batch link insights to source sentences."""
        if not pairs:
            return
        async with self._pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO insight_sources (insight_id, sentence_id)
                VALUES ($1, $2)
                ON CONFLICT DO NOTHING
                """,
                [(uuid.UUID(i), uuid.UUID(s)) for i, s in pairs],
            )

    async def get_source_sentences(self, fact_ids: list[str]) -> list[str]:
        """Return sentence IDs that are the sources for the given facts."""
        if not fact_ids:
            return []
        uuid_ids = [uuid.UUID(fid) for fid in fact_ids]
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT sentence_id
                FROM fact_sources
                WHERE fact_id = ANY($1::uuid[])
                """,
                uuid_ids,
            )
        return [str(row["sentence_id"]) for row in rows]

    # ── Sessions ───────────────────────────────────────────────────────────────

    async def upsert_session(
        self,
        session_id: str,
        user_id: str,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO sessions (id, user_id, agent_id, metadata)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (id) DO UPDATE
                    SET metadata = EXCLUDED.metadata
                """,
                session_id,
                user_id,
                agent_id,
                json.dumps(metadata or {}),
            )

    async def get_session(
        self,
        session_id: str,
        user_id: str,
    ) -> dict[str, Any] | None:
        async with self._pool.acquire() as conn:
            session_row = await conn.fetchrow(
                """
                SELECT id, user_id, agent_id, started_at, ended_at, metadata
                FROM sessions
                WHERE id = $1 AND user_id = $2
                """,
                session_id,
                user_id,
            )
            if not session_row:
                return None

            sentence_rows = await conn.fetch(
                """
                SELECT id, text, turn_number, sentence_index, role, created_at
                FROM sentences
                WHERE session_id = $1 AND is_active = true
                ORDER BY turn_number, sentence_index
                """,
                session_id,
            )

        session = _row(session_row)
        session["sentences"] = [_row(r) for r in sentence_rows]
        return session

    # ── Single-query L2 fast path ──────────────────────────────────────────────

    async def search_l2_single_query(
        self,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        limit: int = 10,
        window: int = 3,
    ) -> dict[str, list[dict[str, Any]]]:
        """Execute the full L2 retrieval in one round trip via a CTE.

        facts → insights (via insight_facts) → source sentences (via fact_sources)
        → session expansion (±window by sentence_index).

        This is meaningfully faster than 4 separate queries at scale.
        Called by SearchPipeline when backend.supports_single_query is True.
        """
        query = """
            WITH
            -- Step 1: Seed facts via vector similarity (L0)
            seed_facts AS (
                SELECT id, text, confidence, created_at, metadata,
                       embedding <=> $1::vector AS distance
                FROM facts
                WHERE user_id = $2
                  AND ($3::text IS NULL OR agent_id = $3)
                  AND is_active = true
                ORDER BY embedding <=> $1::vector
                LIMIT $4
            ),

            -- Step 2: Insights linked to matched facts (L1)
            -- Graph traversal via insight_facts — NOT vector search.
            related_insights AS (
                SELECT DISTINCT i.id, i.text, i.confidence, i.created_at, i.metadata
                FROM insights i
                INNER JOIN insight_facts inf ON i.id = inf.insight_id
                WHERE inf.fact_id IN (SELECT id FROM seed_facts)
                  AND i.is_active = true
            ),

            -- Step 3: Source sentences for matched facts
            source_sentences AS (
                SELECT DISTINCT s.id, s.text, s.session_id, s.turn_number,
                                s.sentence_index, s.role, s.created_at
                FROM sentences s
                INNER JOIN fact_sources fs ON s.id = fs.sentence_id
                WHERE fs.fact_id IN (SELECT id FROM seed_facts)
                  AND s.is_active = true
            ),

            -- Step 4: Session expansion (±window around each source sentence)
            expanded_sentences AS (
                SELECT DISTINCT s2.id, s2.text, s2.session_id, s2.turn_number,
                                s2.sentence_index, s2.role, s2.created_at
                FROM source_sentences src
                INNER JOIN sentences s2
                    ON  s2.session_id = src.session_id
                    AND s2.turn_number = src.turn_number
                    AND s2.sentence_index BETWEEN src.sentence_index - $5
                                              AND src.sentence_index + $5
                WHERE s2.is_active = true
            )

            -- Return all three layers tagged by type
            SELECT 'fact'     AS layer, id, text, confidence, created_at,
                              metadata::text, distance
            FROM seed_facts
            UNION ALL
            SELECT 'insight'  AS layer, id, text, confidence, created_at,
                              metadata::text, NULL AS distance
            FROM related_insights
            UNION ALL
            SELECT 'sentence' AS layer, id, text, NULL AS confidence, created_at,
                              NULL AS metadata, NULL AS distance
            FROM expanded_sentences
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                query,
                _vec(embedding),
                user_id,
                agent_id,
                limit,
                window,
            )

        facts: list[dict[str, Any]] = []
        insights: list[dict[str, Any]] = []
        sentences: list[dict[str, Any]] = []

        for row in rows:
            d = _row(row)
            layer = d.pop("layer")
            if layer == "fact":
                facts.append(d)
            elif layer == "insight":
                d.pop("distance", None)
                insights.append(d)
            else:
                d.pop("distance", None)
                d.pop("confidence", None)
                sentences.append(d)

        return {"facts": facts, "insights": insights, "sentences": sentences}

    # ── GDPR ───────────────────────────────────────────────────────────────────

    async def delete_user(self, user_id: str) -> int:
        """Cascade delete all data for a user.

        FK cascades in the schema handle sentence_edges, fact_sources,
        insight_sources, insight_facts automatically when the parent rows are
        deleted. Wrapped in a transaction so partial deletes can't happen.
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                counts = await conn.fetchrow(
                    """
                    SELECT
                        (SELECT COUNT(*) FROM sentences WHERE user_id = $1) +
                        (SELECT COUNT(*) FROM facts    WHERE user_id = $1) +
                        (SELECT COUNT(*) FROM insights WHERE user_id = $1) +
                        (SELECT COUNT(*) FROM sessions WHERE user_id = $1) AS total
                    """,
                    user_id,
                )
                total = counts["total"] if counts else 0
                await conn.execute("DELETE FROM sentences WHERE user_id = $1", user_id)
                await conn.execute("DELETE FROM facts    WHERE user_id = $1", user_id)
                await conn.execute("DELETE FROM insights WHERE user_id = $1", user_id)
                await conn.execute("DELETE FROM sessions WHERE user_id = $1", user_id)

        logger.info("Deleted %d rows for user %s", total, user_id)
        return int(total)

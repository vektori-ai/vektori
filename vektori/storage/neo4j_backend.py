"""Neo4j storage backend — graph-native with vector index support."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any

from vektori.storage.base import StorageBackend

logger = logging.getLogger(__name__)


class Neo4jBackend(StorageBackend):
    """
    Neo4j storage backend using the official async Python driver (v6.x).

    Setup:
        docker run -p 7474:7474 -p 7687:7687 \\
            -e NEO4J_AUTH=neo4j/password neo4j:latest
        # Requires Neo4j 5.11+ for vector index support.

    Connection URL formats:
        bolt://localhost:7687
        neo4j://localhost:7687   (Bolt routing)
        neo4j+s://...            (TLS)

    Node labels:  Sentence · Fact · Episode · Session
    Relationships:
        (:Fact)-[:SOURCE_OF]->(:Sentence)       fact → its source sentence
        (:Episode)-[:DERIVED_FROM]->(:Fact)     episode → contributing facts
        (:Fact)-[:SUPERSEDED_BY]->(:Fact)       conflict resolution chain
        (:Sentence)-[:NEXT]->(:Sentence)        ordered sentence graph

    Vector indexes on Fact.embedding, Sentence.embedding, Episode.embedding.
    Full-text index on Fact.text for find_fact_by_text.
    """

    def __init__(
        self,
        uri: str,
        auth: tuple[str, str] = ("neo4j", "password"),
        database: str = "neo4j",
        embedding_dim: int = 1024,
    ) -> None:
        self.uri = uri
        self.auth = auth
        self.database = database
        self.embedding_dim = embedding_dim
        self._driver = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        try:
            from neo4j import AsyncGraphDatabase
        except ImportError as e:
            raise ImportError("neo4j required: pip install 'vektori[neo4j]'") from e

        self._driver = AsyncGraphDatabase.driver(self.uri, auth=self.auth)
        await self._driver.verify_connectivity()
        await self._create_schema()
        logger.info("Neo4j backend initialized at %s (db=%s)", self.uri, self.database)

    async def _q(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a Cypher query and return results as plain Python dicts."""
        async with self._driver.session(database=self.database) as session:
            result = await session.run(query, params or {})
            return await result.data()

    async def _create_schema(self) -> None:
        """Create constraints, indexes, and vector indexes. Idempotent via IF NOT EXISTS."""
        dim = self.embedding_dim
        stmts = [
            # Uniqueness constraints
            "CREATE CONSTRAINT sentence_id   IF NOT EXISTS FOR (n:Sentence) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT fact_id       IF NOT EXISTS FOR (n:Fact)     REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT episode_id    IF NOT EXISTS FOR (n:Episode)  REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT session_id    IF NOT EXISTS FOR (n:Session)  REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT sentence_hash IF NOT EXISTS FOR (n:Sentence) REQUIRE n.content_hash IS UNIQUE",
            # Property indexes
            "CREATE INDEX sentence_user    IF NOT EXISTS FOR (n:Sentence) ON (n.user_id)",
            "CREATE INDEX sentence_session IF NOT EXISTS FOR (n:Sentence) ON (n.session_id)",
            "CREATE INDEX fact_user        IF NOT EXISTS FOR (n:Fact)     ON (n.user_id)",
            "CREATE INDEX fact_user_active IF NOT EXISTS FOR (n:Fact)     ON (n.user_id, n.is_active)",
            "CREATE INDEX fact_subject     IF NOT EXISTS FOR (n:Fact)     ON (n.user_id, n.subject)",
            "CREATE INDEX episode_user     IF NOT EXISTS FOR (n:Episode)  ON (n.user_id)",
            # Full-text index for find_fact_by_text
            "CREATE FULLTEXT INDEX fact_text IF NOT EXISTS FOR (n:Fact) ON EACH [n.text]",
            # Vector indexes (requires Neo4j 5.11+)
            f"""CREATE VECTOR INDEX fact_embedding IF NOT EXISTS
                FOR (n:Fact) ON n.embedding
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {dim},
                    `vector.similarity_function`: 'cosine'
                }}}}""",
            f"""CREATE VECTOR INDEX sentence_embedding IF NOT EXISTS
                FOR (n:Sentence) ON n.embedding
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {dim},
                    `vector.similarity_function`: 'cosine'
                }}}}""",
            f"""CREATE VECTOR INDEX episode_embedding IF NOT EXISTS
                FOR (n:Episode) ON n.embedding
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {dim},
                    `vector.similarity_function`: 'cosine'
                }}}}""",
        ]
        for stmt in stmts:
            try:
                await self._q(stmt)
            except Exception as exc:
                logger.debug("Schema stmt skipped (%s): %.80s", exc, stmt)

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()
            self._driver = None

    # ── Sentences ──────────────────────────────────────────────────────────────

    async def upsert_sentences(
        self,
        sentences: list[dict[str, Any]],
        embeddings: list[list[float]],
        user_id: str,
        agent_id: str | None = None,
    ) -> int:
        from vektori.ingestion.hasher import generate_content_hash

        if not sentences:
            return 0

        rows = []
        for sent, emb in zip(sentences, embeddings):
            content_hash = generate_content_hash(
                sent["session_id"],
                f"{sent['turn_number']}_{sent['sentence_index']}",
                sent["text"],
            )
            rows.append(
                {
                    "id": sent["id"],
                    "text": sent["text"],
                    "embedding": emb,
                    "user_id": user_id,
                    "agent_id": agent_id,
                    "session_id": sent["session_id"],
                    "turn_number": sent["turn_number"],
                    "sentence_index": sent["sentence_index"],
                    "role": sent.get("role", "user"),
                    "content_hash": content_hash,
                }
            )

        # MERGE on content_hash (the dedup key).
        # ON CREATE: full insert. ON MATCH: increment mentions only.
        await self._q(
            """
            UNWIND $rows AS row
            MERGE (s:Sentence {content_hash: row.content_hash})
            ON CREATE SET
                s.id             = row.id,
                s.text           = row.text,
                s.embedding      = row.embedding,
                s.user_id        = row.user_id,
                s.agent_id       = row.agent_id,
                s.session_id     = row.session_id,
                s.turn_number    = row.turn_number,
                s.sentence_index = row.sentence_index,
                s.role           = row.role,
                s.mentions       = 1,
                s.is_active      = true,
                s.created_at     = datetime()
            ON MATCH SET
                s.mentions = s.mentions + 1
            """,
            {"rows": rows},
        )
        return len(rows)

    async def search_sentences(
        self,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        scan = max(limit * 20, 200)
        rows = await self._q(
            """
            CALL db.index.vector.queryNodes('sentence_embedding', $scan, $embedding)
            YIELD node AS s, score
            WHERE s.user_id = $user_id
              AND ($agent_id IS NULL OR s.agent_id = $agent_id)
              AND s.is_active = true
            RETURN s.id AS id, s.text AS text, s.session_id AS session_id,
                   s.turn_number AS turn_number, s.sentence_index AS sentence_index,
                   s.role AS role, s.mentions AS mentions, s.created_at AS created_at,
                   (1.0 - score) AS distance
            ORDER BY distance ASC
            LIMIT $limit
            """,
            {"embedding": embedding, "user_id": user_id,
             "agent_id": agent_id, "scan": scan, "limit": limit},
        )
        return [_coerce(r) for r in rows]

    async def find_sentences_by_similarity(
        self,
        quotes: list[str],
        session_id: str,
        threshold: float = 0.75,
    ) -> list[str]:
        # Superseded by search_sentences_in_session (embedding-based).
        return []

    async def search_sentences_in_session(
        self,
        embedding: list[float],
        session_id: str,
        limit: int = 3,
        threshold: float = 0.75,
    ) -> list[str]:
        scan = max(limit * 20, 100)
        rows = await self._q(
            """
            CALL db.index.vector.queryNodes('sentence_embedding', $scan, $embedding)
            YIELD node AS s, score
            WHERE s.session_id = $session_id
              AND s.is_active = true
              AND score >= $threshold
            RETURN s.id AS id
            ORDER BY score DESC
            LIMIT $limit
            """,
            {"embedding": embedding, "session_id": session_id,
             "threshold": threshold, "scan": scan, "limit": limit},
        )
        return [r["id"] for r in rows]

    async def find_sentence_containing(
        self,
        session_id: str,
        quote: str,
    ) -> dict[str, Any] | None:
        rows = await self._q(
            """
            MATCH (s:Sentence {session_id: $session_id})
            WHERE toLower(s.text) CONTAINS toLower($quote)
            RETURN s.id AS id, s.text AS text, s.session_id AS session_id,
                   s.turn_number AS turn_number, s.sentence_index AS sentence_index,
                   s.role AS role, s.created_at AS created_at
            LIMIT 1
            """,
            {"session_id": session_id, "quote": quote},
        )
        return _coerce(rows[0]) if rows else None

    # ── Facts ──────────────────────────────────────────────────────────────────

    async def insert_fact(
        self,
        text: str,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        subject: str | None = None,
        confidence: float = 1.0,
        superseded_by_target: str | None = None,
        metadata: dict[str, Any] | None = None,
        event_time: datetime | None = None,
    ) -> str:
        fact_id = str(uuid.uuid4())
        await self._q(
            """
            CREATE (f:Fact {
                id:            $id,
                text:          $text,
                embedding:     $embedding,
                user_id:       $user_id,
                agent_id:      $agent_id,
                session_id:    $session_id,
                subject:       $subject,
                confidence:    $confidence,
                superseded_by: $superseded_by,
                metadata:      $metadata,
                event_time:    $event_time,
                mentions:      1,
                is_active:     true,
                created_at:    datetime()
            })
            """,
            {
                "id": fact_id,
                "text": text,
                "embedding": embedding,
                "user_id": user_id,
                "agent_id": agent_id,
                "session_id": session_id,
                "subject": subject,
                "confidence": confidence,
                "superseded_by": superseded_by_target,
                "metadata": json.dumps(metadata or {}),
                "event_time": event_time.isoformat() if event_time else None,
            },
        )
        if superseded_by_target:
            await self._q(
                """
                MATCH (f:Fact {id: $fid}), (t:Fact {id: $tid})
                MERGE (f)-[:SUPERSEDED_BY]->(t)
                """,
                {"fid": fact_id, "tid": superseded_by_target},
            )
        return fact_id

    async def search_facts(
        self,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        subject: str | None = None,
        limit: int = 10,
        active_only: bool = True,
        before_date: datetime | None = None,
        after_date: datetime | None = None,
    ) -> list[dict[str, Any]]:
        scan = max(limit * 20, 200)
        rows = await self._q(
            """
            CALL db.index.vector.queryNodes('fact_embedding', $scan, $embedding)
            YIELD node AS f, score
            WHERE f.user_id = $user_id
              AND ($agent_id   IS NULL OR f.agent_id   = $agent_id)
              AND ($session_id IS NULL OR f.session_id = $session_id)
              AND ($subject    IS NULL OR f.subject    = $subject)
              AND ($active_only = false OR f.is_active = true)
              AND ($before_date IS NULL OR f.event_time <= $before_date)
              AND ($after_date  IS NULL OR f.event_time >= $after_date)
            RETURN f.id AS id, f.text AS text, f.confidence AS confidence,
                   f.mentions AS mentions, f.session_id AS session_id,
                   f.subject AS subject, f.created_at AS created_at,
                   f.event_time AS event_time, f.metadata AS metadata,
                   (1.0 - score) AS distance
            ORDER BY distance ASC
            LIMIT $limit
            """,
            {
                "embedding": embedding,
                "user_id": user_id,
                "agent_id": agent_id,
                "session_id": session_id,
                "subject": subject,
                "active_only": active_only,
                "scan": scan,
                "limit": limit,
                "before_date": before_date.isoformat() if before_date else None,
                "after_date": after_date.isoformat() if after_date else None,
            },
        )
        return [_coerce(r) for r in rows]

    async def get_active_facts(
        self,
        user_id: str,
        agent_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        rows = await self._q(
            """
            MATCH (f:Fact {user_id: $user_id, is_active: true})
            WHERE $agent_id IS NULL OR f.agent_id = $agent_id
            RETURN f.id AS id, f.text AS text, f.confidence AS confidence,
                   f.is_active AS is_active, f.superseded_by AS superseded_by,
                   f.created_at AS created_at, f.metadata AS metadata
            ORDER BY f.created_at DESC
            SKIP $offset LIMIT $limit
            """,
            {"user_id": user_id, "agent_id": agent_id, "limit": limit, "offset": offset},
        )
        return [_coerce(r) for r in rows]

    async def deactivate_fact(self, fact_id: str, superseded_by: str | None = None) -> None:
        await self._q(
            """
            MATCH (f:Fact {id: $fact_id})
            SET f.is_active = false, f.superseded_by = $superseded_by
            """,
            {"fact_id": fact_id, "superseded_by": superseded_by},
        )
        if superseded_by:
            await self._q(
                """
                MATCH (f:Fact {id: $fid}), (t:Fact {id: $tid})
                MERGE (f)-[:SUPERSEDED_BY]->(t)
                """,
                {"fid": fact_id, "tid": superseded_by},
            )

    async def increment_fact_mentions(self, fact_id: str) -> None:
        await self._q(
            "MATCH (f:Fact {id: $id}) SET f.mentions = f.mentions + 1",
            {"id": fact_id},
        )

    async def find_fact_by_text(
        self,
        user_id: str,
        text: str,
        agent_id: str | None = None,
    ) -> dict[str, Any] | None:
        rows = await self._q(
            """
            CALL db.index.fulltext.queryNodes('fact_text', $text) YIELD node AS f, score
            WHERE f.user_id = $user_id
              AND ($agent_id IS NULL OR f.agent_id = $agent_id)
              AND f.is_active = true
            RETURN f.id AS id, f.text AS text, f.confidence AS confidence,
                   f.is_active AS is_active, f.superseded_by AS superseded_by,
                   f.created_at AS created_at
            LIMIT 1
            """,
            {"text": text, "user_id": user_id, "agent_id": agent_id},
        )
        return _coerce(rows[0]) if rows else None

    async def get_supersession_chain(self, fact_id: str) -> list[dict[str, Any]]:
        rows = await self._q(
            """
            MATCH (start:Fact {id: $id})
            MATCH path = (start)-[:SUPERSEDED_BY*0..50]->(ancestor:Fact)
            WITH nodes(path) AS chain_nodes
            UNWIND chain_nodes AS n
            WITH DISTINCT n
            ORDER BY n.created_at DESC
            RETURN n.id AS id, n.text AS text, n.confidence AS confidence,
                   n.is_active AS is_active, n.superseded_by AS superseded_by,
                   n.created_at AS created_at
            """,
            {"id": fact_id},
        )
        return [_coerce(r) for r in rows]

    # ── Edges ──────────────────────────────────────────────────────────────────

    async def insert_edges(self, edges: list[dict[str, Any]]) -> int:
        if not edges:
            return 0
        by_type: dict[str, list[dict[str, Any]]] = {}
        for e in edges:
            by_type.setdefault(e["edge_type"], []).append(e)

        for edge_type, group in by_type.items():
            await self._q(
                f"""
                UNWIND $edges AS e
                MATCH (src:Sentence {{id: e.src}}), (tgt:Sentence {{id: e.tgt}})
                MERGE (src)-[r:{edge_type}]->(tgt)
                ON CREATE SET r.weight = e.weight
                """,
                {
                    "edges": [
                        {"src": e["source_id"], "tgt": e["target_id"],
                         "weight": e.get("weight", 1.0)}
                        for e in group
                    ]
                },
            )
        return len(edges)

    async def expand_session_context(
        self,
        sentence_ids: list[str],
        window: int = 3,
    ) -> list[dict[str, Any]]:
        if not sentence_ids:
            return []
        rows = await self._q(
            """
            MATCH (src:Sentence) WHERE src.id IN $ids
            MATCH (s2:Sentence)
            WHERE s2.session_id     = src.session_id
              AND s2.turn_number    = src.turn_number
              AND s2.sentence_index >= src.sentence_index - $window
              AND s2.sentence_index <= src.sentence_index + $window
              AND s2.is_active = true
            RETURN DISTINCT
                s2.id AS id, s2.text AS text, s2.session_id AS session_id,
                s2.turn_number AS turn_number, s2.sentence_index AS sentence_index,
                s2.role AS role, s2.created_at AS created_at
            ORDER BY s2.session_id, s2.turn_number, s2.sentence_index
            """,
            {"ids": sentence_ids, "window": window},
        )
        return [_coerce(r) for r in rows]

    # ── Join tables ────────────────────────────────────────────────────────────

    async def insert_fact_source(self, fact_id: str, sentence_id: str) -> None:
        await self.insert_fact_sources([(fact_id, sentence_id)])

    async def insert_fact_sources(self, pairs: list[tuple[str, str]]) -> None:
        if not pairs:
            return
        await self._q(
            """
            UNWIND $pairs AS p
            MATCH (f:Fact {id: p[0]}), (s:Sentence {id: p[1]})
            MERGE (f)-[:SOURCE_OF]->(s)
            """,
            {"pairs": [[f, s] for f, s in pairs]},
        )

    async def get_source_sentences(self, fact_ids: list[str]) -> list[str]:
        if not fact_ids:
            return []
        rows = await self._q(
            """
            MATCH (f:Fact)-[:SOURCE_OF]->(s:Sentence)
            WHERE f.id IN $ids
            RETURN DISTINCT s.id AS id
            """,
            {"ids": fact_ids},
        )
        return [r["id"] for r in rows]

    async def get_sentences_by_ids(self, sentence_ids: list[str]) -> list[dict[str, Any]]:
        if not sentence_ids:
            return []
        rows = await self._q(
            """
            MATCH (s:Sentence) WHERE s.id IN $ids AND s.is_active = true
            RETURN s.id AS id, s.text AS text, s.session_id AS session_id,
                   s.turn_number AS turn_number, s.sentence_index AS sentence_index,
                   s.role AS role, s.created_at AS created_at
            ORDER BY s.session_id, s.turn_number, s.sentence_index
            """,
            {"ids": sentence_ids},
        )
        return [_coerce(r) for r in rows]

    # ── Episodes ──────────────────────────────────────────────────────────────

    async def insert_episode(
        self,
        text: str,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> str:
        episode_id = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{user_id}::{text}"))
        await self._q(
            """
            MERGE (e:Episode {id: $id})
            ON CREATE SET
                e.text       = $text,
                e.embedding  = $embedding,
                e.user_id    = $user_id,
                e.agent_id   = $agent_id,
                e.session_id = $session_id,
                e.is_active  = true,
                e.created_at = datetime()
            """,
            {
                "id": episode_id, "text": text, "embedding": embedding,
                "user_id": user_id, "agent_id": agent_id, "session_id": session_id,
            },
        )
        return episode_id

    async def insert_episode_fact(self, episode_id: str, fact_id: str) -> None:
        await self._q(
            """
            MATCH (e:Episode {id: $eid}), (f:Fact {id: $fid})
            MERGE (e)-[:DERIVED_FROM]->(f)
            """,
            {"eid": episode_id, "fid": fact_id},
        )

    async def get_episodes_for_facts(self, fact_ids: list[str]) -> list[dict[str, Any]]:
        if not fact_ids:
            return []
        rows = await self._q(
            """
            MATCH (e:Episode)-[:DERIVED_FROM]->(f:Fact)
            WHERE f.id IN $ids AND e.is_active = true
            RETURN DISTINCT e.id AS id, e.text AS text,
                            e.session_id AS session_id, e.created_at AS created_at
            """,
            {"ids": fact_ids},
        )
        return [_coerce(r) for r in rows]

    async def search_episodes(
        self,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        scan = max(limit * 20, 100)
        rows = await self._q(
            """
            CALL db.index.vector.queryNodes('episode_embedding', $scan, $embedding)
            YIELD node AS e, score
            WHERE e.user_id = $user_id
              AND ($agent_id IS NULL OR e.agent_id = $agent_id)
              AND e.is_active = true
            RETURN e.id AS id, e.text AS text, e.session_id AS session_id,
                   e.created_at AS created_at, (1.0 - score) AS distance
            ORDER BY distance ASC
            LIMIT $limit
            """,
            {"embedding": embedding, "user_id": user_id,
             "agent_id": agent_id, "scan": scan, "limit": limit},
        )
        return [_coerce(r) for r in rows]

    # ── Sessions ───────────────────────────────────────────────────────────────

    async def upsert_session(
        self,
        session_id: str,
        user_id: str,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        started_at: datetime | None = None,
    ) -> None:
        await self._q(
            """
            MERGE (s:Session {id: $id})
            ON CREATE SET
                s.user_id    = $user_id,
                s.agent_id   = $agent_id,
                s.metadata   = $metadata,
                s.started_at = coalesce($started_at, toString(datetime())),
                s.ended_at   = null
            ON MATCH SET
                s.metadata   = $metadata
            """,
            {
                "id": session_id,
                "user_id": user_id,
                "agent_id": agent_id,
                "metadata": json.dumps(metadata or {}),
                "started_at": started_at.isoformat() if started_at else None,
            },
        )

    async def get_session(
        self,
        session_id: str,
        user_id: str,
    ) -> dict[str, Any] | None:
        rows = await self._q(
            """
            MATCH (s:Session {id: $id, user_id: $user_id})
            RETURN s.id AS id, s.user_id AS user_id, s.agent_id AS agent_id,
                   s.started_at AS started_at, s.ended_at AS ended_at,
                   s.metadata AS metadata
            """,
            {"id": session_id, "user_id": user_id},
        )
        if not rows:
            return None
        session = _coerce(rows[0])
        sent_rows = await self._q(
            """
            MATCH (s:Sentence {session_id: $id, is_active: true})
            RETURN s.id AS id, s.text AS text, s.turn_number AS turn_number,
                   s.sentence_index AS sentence_index, s.role AS role,
                   s.created_at AS created_at
            ORDER BY s.turn_number, s.sentence_index
            """,
            {"id": session_id},
        )
        session["sentences"] = [_coerce(r) for r in sent_rows]
        return session

    async def count_sessions(
        self,
        user_id: str,
        agent_id: str | None = None,
    ) -> int:
        rows = await self._q(
            """
            MATCH (s:Session {user_id: $user_id})
            WHERE $agent_id IS NULL OR s.agent_id = $agent_id
            RETURN count(s) AS n
            """,
            {"user_id": user_id, "agent_id": agent_id},
        )
        return int(rows[0]["n"]) if rows else 0

    # ── GDPR ───────────────────────────────────────────────────────────────────

    async def delete_user(self, user_id: str) -> int:
        rows = await self._q(
            """
            MATCH (n)
            WHERE (n:Sentence OR n:Fact OR n:Episode OR n:Session)
              AND n.user_id = $user_id
            WITH count(n) AS total, collect(n) AS nodes
            FOREACH (n IN nodes | DETACH DELETE n)
            RETURN total
            """,
            {"user_id": user_id},
        )
        deleted = int(rows[0]["total"]) if rows else 0
        logger.info("Deleted %d Neo4j nodes for user %s", deleted, user_id)
        return deleted


# ── Helpers ────────────────────────────────────────────────────────────────────


def _coerce(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize a Neo4j result row.

    neo4j.time.DateTime objects expose .to_native() → Python datetime.
    Everything else passes through unchanged.
    """
    return {k: (v.to_native() if v is not None and hasattr(v, "to_native") else v)
            for k, v in row.items()}

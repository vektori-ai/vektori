"""Qdrant storage backend — dedicated vector database with payload filtering."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

from vektori.storage.base import StorageBackend

logger = logging.getLogger(__name__)

# Collection name suffixes
_FACTS = "facts"
_SENTENCES = "sentences"
_EPISODES = "episodes"
_SESSIONS = "sessions"


def _col(prefix: str, suffix: str) -> str:
    return f"{prefix}_{suffix}"


def _to_uuid(s: str) -> str:
    """Convert an arbitrary string to a stable UUID string (for session IDs)."""
    return str(uuid.uuid5(uuid.NAMESPACE_OID, s))


class QdrantBackend(StorageBackend):
    """
    Qdrant storage backend using the official async Python client (qdrant-client ≥ 1.6).

    Setup:
        docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

    Connection URL:
        http://localhost:6333   (REST)
        Use grpc_port=6334 for gRPC mode.

    Design notes:
        - Four collections: {prefix}_facts, _sentences, _episodes, _sessions.
        - Sessions collection uses a 1-dim dummy vector (sessions are not vector-searched).
        - Join tables (fact→sentence, episode→fact) stored as payload arrays:
              fact payload:    source_sentence_ids: list[str]
              episode payload: fact_ids: list[str]
        - Supersession chain stored as superseded_by string in fact payload,
          traversed in Python (shallow chains only in practice).
        - NEXT edges stored as next_sentence_id in sentence payload.
          expand_session_context uses sentence_index range scan instead
          (matches Postgres/SQLite behaviour).
        - find_fact_by_text: payload full-text index + Python fallback.
        - find_sentence_containing: scroll session sentences + Python substring.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        prefix: str = "vektori",
        embedding_dim: int = 1024,
    ) -> None:
        self.url = url
        self.api_key = api_key
        self.prefix = prefix
        self.embedding_dim = embedding_dim
        self._client = None

    # ── Collection name helpers ────────────────────────────────────────────────

    @property
    def _facts_col(self) -> str:
        return _col(self.prefix, _FACTS)

    @property
    def _sentences_col(self) -> str:
        return _col(self.prefix, _SENTENCES)

    @property
    def _episodes_col(self) -> str:
        return _col(self.prefix, _EPISODES)

    @property
    def _sessions_col(self) -> str:
        return _col(self.prefix, _SESSIONS)

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        try:
            from qdrant_client import AsyncQdrantClient
        except ImportError as e:
            raise ImportError("qdrant-client required: pip install 'vektori[qdrant]'") from e

        kwargs: dict[str, Any] = {
            "url": self.url,
            # Cloud instances need longer TLS handshake time; 30 s is safe.
            "timeout": 30,
            # Disable gRPC auto-detection for HTTPS URLs — use REST only.
            # gRPC requires explicit port 6334; REST on 6333 is always reliable.
            "prefer_grpc": False,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        self._client = AsyncQdrantClient(**kwargs)

        dim = self.embedding_dim
        await self._ensure_collection(self._facts_col, dim)
        await self._ensure_collection(self._sentences_col, dim)
        await self._ensure_collection(self._episodes_col, dim)
        # Sessions: 1-dim dummy vector — not searched by vector.
        await self._ensure_collection(self._sessions_col, 1)

        await self._create_payload_indexes()
        logger.info("Qdrant backend initialized at %s (prefix=%s)", self.url, self.prefix)

    async def _ensure_collection(self, name: str, dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        if not await self._client.collection_exists(name):
            await self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            logger.debug("Created Qdrant collection: %s", name)

    async def _create_payload_indexes(self) -> None:
        """Create payload indexes for common filter fields. Idempotent."""
        from qdrant_client.models import PayloadSchemaType

        index_specs = [
            (self._facts_col,     "user_id",    PayloadSchemaType.KEYWORD),
            (self._facts_col,     "agent_id",   PayloadSchemaType.KEYWORD),
            (self._facts_col,     "session_id", PayloadSchemaType.KEYWORD),
            (self._facts_col,     "subject",    PayloadSchemaType.KEYWORD),
            (self._facts_col,     "is_active",  PayloadSchemaType.BOOL),
            (self._facts_col,     "event_time", PayloadSchemaType.DATETIME),
            (self._facts_col,     "text",       PayloadSchemaType.TEXT),
            (self._sentences_col, "user_id",    PayloadSchemaType.KEYWORD),
            (self._sentences_col, "agent_id",   PayloadSchemaType.KEYWORD),
            (self._sentences_col, "session_id", PayloadSchemaType.KEYWORD),
            (self._sentences_col, "content_hash", PayloadSchemaType.KEYWORD),
            (self._sentences_col, "is_active",  PayloadSchemaType.BOOL),
            (self._sentences_col, "turn_number", PayloadSchemaType.INTEGER),
            (self._sentences_col, "sentence_index", PayloadSchemaType.INTEGER),
            (self._episodes_col,  "user_id",    PayloadSchemaType.KEYWORD),
            (self._episodes_col,  "agent_id",   PayloadSchemaType.KEYWORD),
            (self._episodes_col,  "is_active",  PayloadSchemaType.BOOL),
            # Array fields used in Should/MatchValue filters — index required on Qdrant Cloud
            (self._episodes_col,  "fact_ids",   PayloadSchemaType.KEYWORD),
            (self._facts_col,     "source_sentence_ids", PayloadSchemaType.KEYWORD),
            (self._sessions_col,  "user_id",    PayloadSchemaType.KEYWORD),
            (self._sessions_col,  "agent_id",   PayloadSchemaType.KEYWORD),
        ]
        for col, field, schema_type in index_specs:
            try:
                await self._client.create_payload_index(
                    collection_name=col,
                    field_name=field,
                    field_schema=schema_type,
                )
            except Exception:
                pass  # Already exists — safe to ignore

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    # ── Sentences ──────────────────────────────────────────────────────────────

    async def upsert_sentences(
        self,
        sentences: list[dict[str, Any]],
        embeddings: list[list[float]],
        user_id: str,
        agent_id: str | None = None,
    ) -> int:
        from qdrant_client.models import PointStruct

        from vektori.ingestion.hasher import generate_content_hash

        if not sentences:
            return 0

        # Build id→(sent, emb) mapping. Sentence IDs are deterministic UUIDs.
        point_ids = [sent["id"] for sent in sentences]

        # Retrieve existing points to handle mentions increment
        existing_map: dict[str, int] = {}
        try:
            existing = await self._client.retrieve(
                collection_name=self._sentences_col,
                ids=point_ids,
                with_payload=["mentions"],
            )
            for p in existing:
                existing_map[str(p.id)] = (p.payload or {}).get("mentions", 1)
        except Exception:
            pass  # Collection may be empty — proceed with fresh inserts

        points = []
        for sent, emb in zip(sentences, embeddings):
            sid = sent["id"]
            content_hash = generate_content_hash(
                sent["session_id"],
                f"{sent['turn_number']}_{sent['sentence_index']}",
                sent["text"],
            )
            mentions = existing_map.get(sid, 0) + 1 if sid in existing_map else 1
            points.append(
                PointStruct(
                    id=sid,
                    vector=emb,
                    payload={
                        "text": sent["text"],
                        "user_id": user_id,
                        "agent_id": agent_id,
                        "session_id": sent["session_id"],
                        "turn_number": sent["turn_number"],
                        "sentence_index": sent["sentence_index"],
                        "role": sent.get("role", "user"),
                        "content_hash": content_hash,
                        "mentions": mentions,
                        "is_active": True,
                        "created_at": datetime.utcnow().isoformat(),
                    },
                )
            )

        await self._client.upsert(collection_name=self._sentences_col, points=points)
        return len(points)

    async def search_sentences(
        self,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        must = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
            FieldCondition(key="is_active", match=MatchValue(value=True)),
        ]
        if agent_id is not None:
            must.append(FieldCondition(key="agent_id", match=MatchValue(value=agent_id)))

        hits = await self._client.query_points(
            collection_name=self._sentences_col,
            query=embedding,
            query_filter=Filter(must=must),
            limit=limit,
            with_payload=True,
        )
        return [_hit_to_sentence(h) for h in hits.points]

    async def find_sentences_by_similarity(
        self,
        quotes: list[str],
        session_id: str,
        threshold: float = 0.75,
    ) -> list[str]:
        return []

    async def search_sentences_in_session(
        self,
        embedding: list[float],
        session_id: str,
        limit: int = 3,
        threshold: float = 0.75,
    ) -> list[str]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        hits = await self._client.query_points(
            collection_name=self._sentences_col,
            query=embedding,
            query_filter=Filter(must=[
                FieldCondition(key="session_id", match=MatchValue(value=session_id)),
                FieldCondition(key="is_active", match=MatchValue(value=True)),
            ]),
            limit=limit,
            score_threshold=threshold,
            with_payload=["id"],
        )
        return [str(h.id) for h in hits.points]

    async def find_sentence_containing(
        self,
        session_id: str,
        quote: str,
    ) -> dict[str, Any] | None:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        # Scroll all sentences in session, filter by substring in Python.
        lower_quote = quote.lower()
        offset = None
        while True:
            result, next_offset = await self._client.scroll(
                collection_name=self._sentences_col,
                scroll_filter=Filter(must=[
                    FieldCondition(key="session_id", match=MatchValue(value=session_id)),
                ]),
                limit=100,
                offset=offset,
                with_payload=True,
            )
            for point in result:
                p = point.payload or {}
                if lower_quote in (p.get("text") or "").lower():
                    return _payload_to_sentence(str(point.id), p)
            if next_offset is None:
                return None
            offset = next_offset

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
        import json

        from qdrant_client.models import PointStruct

        fact_id = str(uuid.uuid4())
        await self._client.upsert(
            collection_name=self._facts_col,
            points=[
                PointStruct(
                    id=fact_id,
                    vector=embedding,
                    payload={
                        "text": text,
                        "user_id": user_id,
                        "agent_id": agent_id,
                        "session_id": session_id,
                        "subject": subject,
                        "confidence": confidence,
                        "superseded_by": superseded_by_target,
                        "metadata": json.dumps(metadata or {}),
                        "event_time": event_time.isoformat() if event_time else None,
                        "mentions": 1,
                        "is_active": True,
                        "created_at": datetime.utcnow().isoformat(),
                        # Join-table stored inline — populated later via insert_fact_source
                        "source_sentence_ids": [],
                    },
                )
            ],
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
        from qdrant_client.models import (
            DatetimeRange,
            FieldCondition,
            Filter,
            MatchValue,
        )

        must = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        if active_only:
            must.append(FieldCondition(key="is_active", match=MatchValue(value=True)))
        if agent_id is not None:
            must.append(FieldCondition(key="agent_id", match=MatchValue(value=agent_id)))
        if session_id is not None:
            must.append(FieldCondition(key="session_id", match=MatchValue(value=session_id)))
        if subject is not None:
            must.append(FieldCondition(key="subject", match=MatchValue(value=subject)))
        if before_date is not None or after_date is not None:
            must.append(
                FieldCondition(
                    key="event_time",
                    range=DatetimeRange(
                        lte=before_date.isoformat() if before_date else None,
                        gte=after_date.isoformat() if after_date else None,
                    ),
                )
            )

        hits = await self._client.query_points(
            collection_name=self._facts_col,
            query=embedding,
            query_filter=Filter(must=must),
            limit=limit,
            with_payload=True,
        )
        return [_hit_to_fact(h) for h in hits.points]

    async def get_active_facts(
        self,
        user_id: str,
        agent_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        must = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
            FieldCondition(key="is_active", match=MatchValue(value=True)),
        ]
        if agent_id is not None:
            must.append(FieldCondition(key="agent_id", match=MatchValue(value=agent_id)))

        results, _ = await self._client.scroll(
            collection_name=self._facts_col,
            scroll_filter=Filter(must=must),
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        return [_payload_to_fact(str(p.id), p.payload or {}) for p in results]

    async def deactivate_fact(self, fact_id: str, superseded_by: str | None = None) -> None:
        payload: dict[str, Any] = {"is_active": False}
        if superseded_by is not None:
            payload["superseded_by"] = superseded_by
        await self._client.set_payload(
            collection_name=self._facts_col,
            payload=payload,
            points=[fact_id],
        )

    async def increment_fact_mentions(self, fact_id: str) -> None:
        points = await self._client.retrieve(
            collection_name=self._facts_col,
            ids=[fact_id],
            with_payload=["mentions"],
        )
        current = (points[0].payload or {}).get("mentions", 1) if points else 1
        await self._client.set_payload(
            collection_name=self._facts_col,
            payload={"mentions": current + 1},
            points=[fact_id],
        )

    async def find_fact_by_text(
        self,
        user_id: str,
        text: str,
        agent_id: str | None = None,
    ) -> dict[str, Any] | None:
        from qdrant_client.models import FieldCondition, Filter, MatchText, MatchValue

        must = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
            FieldCondition(key="is_active", match=MatchValue(value=True)),
            FieldCondition(key="text", match=MatchText(text=text)),
        ]
        if agent_id is not None:
            must.append(FieldCondition(key="agent_id", match=MatchValue(value=agent_id)))

        results, _ = await self._client.scroll(
            collection_name=self._facts_col,
            scroll_filter=Filter(must=must),
            limit=1,
            with_payload=True,
        )
        if results:
            return _payload_to_fact(str(results[0].id), results[0].payload or {})
        return None

    async def get_supersession_chain(self, fact_id: str) -> list[dict[str, Any]]:
        """Traverse superseded_by links in Python. Chains are short in practice."""
        chain = []
        current_id: str | None = fact_id
        visited: set[str] = set()

        while current_id and current_id not in visited and len(chain) < 50:
            visited.add(current_id)
            points = await self._client.retrieve(
                collection_name=self._facts_col,
                ids=[current_id],
                with_payload=True,
            )
            if not points:
                break
            p = points[0]
            payload = p.payload or {}
            chain.append(_payload_to_fact(str(p.id), payload))
            current_id = payload.get("superseded_by")

        return chain

    # ── Edges ──────────────────────────────────────────────────────────────────

    async def insert_edges(self, edges: list[dict[str, Any]]) -> int:
        """Store NEXT edges as next_sentence_id payload on the source sentence."""
        if not edges:
            return 0
        # Only handle NEXT edges — other types are recorded but not used by expand.
        next_edges = [e for e in edges if e["edge_type"] == "NEXT"]
        for edge in next_edges:
            await self._client.set_payload(
                collection_name=self._sentences_col,
                payload={"next_sentence_id": edge["target_id"]},
                points=[edge["source_id"]],
            )
        return len(edges)

    async def expand_session_context(
        self,
        sentence_ids: list[str],
        window: int = 3,
    ) -> list[dict[str, Any]]:
        """Expand by sentence_index range within each matched sentence's session + turn."""
        if not sentence_ids:
            return []

        from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

        # Fetch the seed sentences to get their session/turn/index positions
        seed_points = await self._client.retrieve(
            collection_name=self._sentences_col,
            ids=sentence_ids,
            with_payload=["session_id", "turn_number", "sentence_index"],
        )

        seen_ids: set[str] = set()
        all_results: list[dict[str, Any]] = []

        for sp in seed_points:
            p = sp.payload or {}
            sess = p.get("session_id")
            turn = p.get("turn_number")
            idx = p.get("sentence_index")
            if sess is None or turn is None or idx is None:
                continue

            hits, _ = await self._client.scroll(
                collection_name=self._sentences_col,
                scroll_filter=Filter(must=[
                    FieldCondition(key="session_id", match=MatchValue(value=sess)),
                    FieldCondition(key="turn_number", match=MatchValue(value=turn)),
                    FieldCondition(key="sentence_index",
                                   range=Range(gte=idx - window, lte=idx + window)),
                    FieldCondition(key="is_active", match=MatchValue(value=True)),
                ]),
                limit=window * 2 + 5,
                with_payload=True,
            )
            for h in hits:
                sid = str(h.id)
                if sid not in seen_ids:
                    seen_ids.add(sid)
                    all_results.append(_payload_to_sentence(sid, h.payload or {}))

        # Sort by session_id, turn_number, sentence_index
        all_results.sort(key=lambda r: (
            r.get("session_id", ""),
            r.get("turn_number", 0),
            r.get("sentence_index", 0),
        ))
        return all_results

    # ── Join tables ────────────────────────────────────────────────────────────

    async def insert_fact_source(self, fact_id: str, sentence_id: str) -> None:
        await self.insert_fact_sources([(fact_id, sentence_id)])

    async def insert_fact_sources(self, pairs: list[tuple[str, str]]) -> None:
        if not pairs:
            return

        # Group by fact_id to batch the payload updates
        by_fact: dict[str, list[str]] = {}
        for fact_id, sentence_id in pairs:
            by_fact.setdefault(fact_id, []).append(sentence_id)

        for fact_id, new_sids in by_fact.items():
            points = await self._client.retrieve(
                collection_name=self._facts_col,
                ids=[fact_id],
                with_payload=["source_sentence_ids"],
            )
            existing: list[str] = []
            if points:
                existing = (points[0].payload or {}).get("source_sentence_ids", [])
            merged = list(dict.fromkeys(existing + new_sids))  # dedup, preserve order
            await self._client.set_payload(
                collection_name=self._facts_col,
                payload={"source_sentence_ids": merged},
                points=[fact_id],
            )

    async def get_source_sentences(self, fact_ids: list[str]) -> list[str]:
        if not fact_ids:
            return []
        points = await self._client.retrieve(
            collection_name=self._facts_col,
            ids=fact_ids,
            with_payload=["source_sentence_ids"],
        )
        seen: set[str] = set()
        result: list[str] = []
        for p in points:
            for sid in (p.payload or {}).get("source_sentence_ids", []):
                if sid not in seen:
                    seen.add(sid)
                    result.append(sid)
        return result

    async def get_sentences_by_ids(self, sentence_ids: list[str]) -> list[dict[str, Any]]:
        if not sentence_ids:
            return []
        points = await self._client.retrieve(
            collection_name=self._sentences_col,
            ids=sentence_ids,
            with_payload=True,
        )
        results = [_payload_to_sentence(str(p.id), p.payload or {}) for p in points
                   if (p.payload or {}).get("is_active", True)]
        results.sort(key=lambda r: (
            r.get("session_id", ""),
            r.get("turn_number", 0),
            r.get("sentence_index", 0),
        ))
        return results

    # ── Episodes ──────────────────────────────────────────────────────────────

    async def insert_episode(
        self,
        text: str,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> str:
        from qdrant_client.models import PointStruct

        episode_id = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{user_id}::{text}"))
        # Check if already exists (idempotent)
        existing = await self._client.retrieve(
            collection_name=self._episodes_col,
            ids=[episode_id],
            with_payload=False,
        )
        if not existing:
            await self._client.upsert(
                collection_name=self._episodes_col,
                points=[
                    PointStruct(
                        id=episode_id,
                        vector=embedding,
                        payload={
                            "text": text,
                            "user_id": user_id,
                            "agent_id": agent_id,
                            "session_id": session_id,
                            "is_active": True,
                            "created_at": datetime.utcnow().isoformat(),
                            "fact_ids": [],
                        },
                    )
                ],
            )
        return episode_id

    async def insert_episode_fact(self, episode_id: str, fact_id: str) -> None:
        points = await self._client.retrieve(
            collection_name=self._episodes_col,
            ids=[episode_id],
            with_payload=["fact_ids"],
        )
        existing: list[str] = []
        if points:
            existing = (points[0].payload or {}).get("fact_ids", [])
        if fact_id not in existing:
            await self._client.set_payload(
                collection_name=self._episodes_col,
                payload={"fact_ids": existing + [fact_id]},
                points=[episode_id],
            )

    async def get_episodes_for_facts(self, fact_ids: list[str]) -> list[dict[str, Any]]:
        if not fact_ids:
            return []
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        # Qdrant array payload match: FieldCondition on array field matches if any element equals value.
        # Use a Should (OR) across all fact_ids.
        should = [
            FieldCondition(key="fact_ids", match=MatchValue(value=fid))
            for fid in fact_ids
        ]
        must = [FieldCondition(key="is_active", match=MatchValue(value=True))]

        seen: set[str] = set()
        results: list[dict[str, Any]] = []
        offset = None
        while True:
            hits, next_offset = await self._client.scroll(
                collection_name=self._episodes_col,
                scroll_filter=Filter(must=must, should=should),
                limit=50,
                offset=offset,
                with_payload=True,
            )
            for h in hits:
                eid = str(h.id)
                if eid not in seen:
                    seen.add(eid)
                    p = h.payload or {}
                    results.append({
                        "id": eid,
                        "text": p.get("text"),
                        "session_id": p.get("session_id"),
                        "created_at": p.get("created_at"),
                    })
            if next_offset is None:
                break
            offset = next_offset
        return results

    async def search_episodes(
        self,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        must = [
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
            FieldCondition(key="is_active", match=MatchValue(value=True)),
        ]
        if agent_id is not None:
            must.append(FieldCondition(key="agent_id", match=MatchValue(value=agent_id)))

        hits = await self._client.query_points(
            collection_name=self._episodes_col,
            query=embedding,
            query_filter=Filter(must=must),
            limit=limit,
            with_payload=True,
        )
        return [
            {
                "id": str(h.id),
                "text": (h.payload or {}).get("text"),
                "session_id": (h.payload or {}).get("session_id"),
                "created_at": (h.payload or {}).get("created_at"),
                "distance": 1.0 - h.score,
            }
            for h in hits.points
        ]

    # ── Sessions ───────────────────────────────────────────────────────────────

    async def upsert_session(
        self,
        session_id: str,
        user_id: str,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        started_at: datetime | None = None,
    ) -> None:
        import json

        from qdrant_client.models import PointStruct

        point_id = _to_uuid(session_id)
        existing = await self._client.retrieve(
            collection_name=self._sessions_col,
            ids=[point_id],
            with_payload=["started_at"],
        )
        existing_started = (existing[0].payload or {}).get("started_at") if existing else None

        await self._client.upsert(
            collection_name=self._sessions_col,
            points=[
                PointStruct(
                    id=point_id,
                    vector=[0.0],  # dummy — sessions are not vector-searched
                    payload={
                        "session_id": session_id,
                        "user_id": user_id,
                        "agent_id": agent_id,
                        "metadata": json.dumps(metadata or {}),
                        "started_at": (
                            started_at.isoformat()
                            if started_at
                            else existing_started or datetime.utcnow().isoformat()
                        ),
                        "ended_at": None,
                    },
                )
            ],
        )

    async def get_session(
        self,
        session_id: str,
        user_id: str,
    ) -> dict[str, Any] | None:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        point_id = _to_uuid(session_id)
        points = await self._client.retrieve(
            collection_name=self._sessions_col,
            ids=[point_id],
            with_payload=True,
        )
        if not points:
            return None
        p = points[0].payload or {}
        if p.get("user_id") != user_id:
            return None

        session: dict[str, Any] = {
            "id": session_id,
            "user_id": p.get("user_id"),
            "agent_id": p.get("agent_id"),
            "started_at": p.get("started_at"),
            "ended_at": p.get("ended_at"),
            "metadata": p.get("metadata"),
        }

        sent_results, _ = await self._client.scroll(
            collection_name=self._sentences_col,
            scroll_filter=Filter(must=[
                FieldCondition(key="session_id", match=MatchValue(value=session_id)),
                FieldCondition(key="is_active", match=MatchValue(value=True)),
            ]),
            limit=1000,
            with_payload=True,
        )
        sentences = [_payload_to_sentence(str(s.id), s.payload or {}) for s in sent_results]
        sentences.sort(key=lambda r: (r.get("turn_number", 0), r.get("sentence_index", 0)))
        session["sentences"] = sentences
        return session

    async def count_sessions(
        self,
        user_id: str,
        agent_id: str | None = None,
    ) -> int:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        must = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        if agent_id is not None:
            must.append(FieldCondition(key="agent_id", match=MatchValue(value=agent_id)))

        result = await self._client.count(
            collection_name=self._sessions_col,
            count_filter=Filter(must=must),
            exact=True,
        )
        return result.count

    # ── GDPR ───────────────────────────────────────────────────────────────────

    async def delete_user(self, user_id: str) -> int:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        user_filter = Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))])
        total = 0
        for col in (self._facts_col, self._sentences_col,
                    self._episodes_col, self._sessions_col):
            result = await self._client.count(
                collection_name=col, count_filter=user_filter, exact=True
            )
            total += result.count
            await self._client.delete(collection_name=col, points_selector=user_filter)

        logger.info("Deleted %d Qdrant points for user %s", total, user_id)
        return total


# ── Result conversion helpers ─────────────────────────────────────────────────


def _hit_to_sentence(hit: Any) -> dict[str, Any]:
    p = hit.payload or {}
    return {**_payload_to_sentence(str(hit.id), p), "distance": 1.0 - hit.score}


def _payload_to_sentence(sid: str, p: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": sid,
        "text": p.get("text"),
        "session_id": p.get("session_id"),
        "turn_number": p.get("turn_number"),
        "sentence_index": p.get("sentence_index"),
        "role": p.get("role"),
        "mentions": p.get("mentions", 1),
        "is_active": p.get("is_active", True),
        "created_at": p.get("created_at"),
    }


def _hit_to_fact(hit: Any) -> dict[str, Any]:
    p = hit.payload or {}
    return {**_payload_to_fact(str(hit.id), p), "distance": 1.0 - hit.score}


def _payload_to_fact(fid: str, p: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": fid,
        "text": p.get("text"),
        "confidence": p.get("confidence", 1.0),
        "mentions": p.get("mentions", 1),
        "session_id": p.get("session_id"),
        "subject": p.get("subject"),
        "is_active": p.get("is_active", True),
        "superseded_by": p.get("superseded_by"),
        "metadata": p.get("metadata"),
        "event_time": p.get("event_time"),
        "created_at": p.get("created_at"),
    }

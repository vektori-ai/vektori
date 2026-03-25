"""In-memory storage backend. No persistence. For unit tests and CI."""

from __future__ import annotations

import math
import uuid
from datetime import datetime
from typing import Any

from vektori.storage.base import StorageBackend


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class MemoryBackend(StorageBackend):
    """
    In-memory storage backend. Fast, no deps, no persistence.

    Intended for unit tests and CI. Vector search is brute-force cosine.
    """

    def __init__(self) -> None:
        self._sentences: dict[str, dict[str, Any]] = {}
        self._facts: dict[str, dict[str, Any]] = {}
        self._insights: dict[str, dict[str, Any]] = {}
        self._edges: list[dict[str, Any]] = []
        self._fact_sources: list[dict[str, str]] = []       # [{fact_id, sentence_id}]
        self._insight_facts: list[dict[str, str]] = []      # [{insight_id, fact_id}]
        self._insight_sources: list[dict[str, str]] = []    # [{insight_id, sentence_id}]
        self._sessions: dict[str, dict[str, Any]] = {}

    async def initialize(self) -> None:
        pass  # Nothing to do

    async def close(self) -> None:
        pass

    # ── Sentences ──

    async def upsert_sentences(
        self,
        sentences: list[dict[str, Any]],
        embeddings: list[list[float]],
        user_id: str,
        agent_id: str | None = None,
    ) -> int:
        count = 0
        for sent, emb in zip(sentences, embeddings):
            sid = sent["id"]
            if sid in self._sentences:
                self._sentences[sid]["mentions"] += 1
            else:
                self._sentences[sid] = {
                    **sent,
                    "embedding": emb,
                    "user_id": user_id,
                    "agent_id": agent_id,
                    "mentions": 1,
                    "is_active": True,
                    "created_at": datetime.utcnow(),
                }
                count += 1
        return count

    async def search_sentences(
        self,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        results = []
        for s in self._sentences.values():
            if s["user_id"] != user_id:
                continue
            if agent_id and s.get("agent_id") != agent_id:
                continue
            if s.get("embedding") is None:
                continue
            sim = _cosine_similarity(embedding, s["embedding"])
            results.append({**s, "distance": 1.0 - sim})
        results.sort(key=lambda x: x["distance"])
        return results[:limit]

    async def find_sentences_by_similarity(
        self,
        quotes: list[str],
        session_id: str,
        threshold: float = 0.75,
    ) -> list[str]:
        # TODO: embed quotes and do cosine search within session
        # For now returns empty — fact-source linking will be incomplete in memory backend
        return []

    async def find_sentence_containing(
        self,
        session_id: str,
        quote: str,
    ) -> dict[str, Any] | None:
        q = quote.lower()
        for s in self._sentences.values():
            if s.get("session_id") == session_id and q in s["text"].lower():
                return s
        return None

    # ── Facts ──

    async def insert_fact(
        self,
        text: str,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        confidence: float = 1.0,
        superseded_by_target: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        fact_id = str(uuid.uuid4())
        self._facts[fact_id] = {
            "id": fact_id,
            "text": text,
            "embedding": embedding,
            "user_id": user_id,
            "agent_id": agent_id,
            "session_id": session_id,
            "confidence": confidence,
            "superseded_by": superseded_by_target,
            "is_active": True,
            "metadata": metadata or {},
            "created_at": datetime.utcnow(),
        }
        return fact_id

    async def search_facts(
        self,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        limit: int = 10,
        active_only: bool = True,
    ) -> list[dict[str, Any]]:
        results = []
        for f in self._facts.values():
            if f.get("user_id") != user_id:
                continue
            if agent_id and f.get("agent_id") != agent_id:
                continue
            if active_only and not f.get("is_active", True):
                continue
            if f.get("embedding") is None:
                continue
            sim = _cosine_similarity(embedding, f["embedding"])
            results.append({**f, "distance": 1.0 - sim})
        results.sort(key=lambda x: x["distance"])
        return results[:limit]

    async def get_active_facts(
        self,
        user_id: str,
        agent_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        results = [
            f for f in self._facts.values()
            if f.get("user_id") == user_id and f.get("is_active", True)
        ]
        return results[offset: offset + limit]

    async def deactivate_fact(self, fact_id: str, superseded_by: str | None = None) -> None:
        if fact_id in self._facts:
            self._facts[fact_id]["is_active"] = False
            if superseded_by:
                self._facts[fact_id]["superseded_by"] = superseded_by

    async def find_fact_by_text(
        self,
        user_id: str,
        text: str,
        agent_id: str | None = None,
    ) -> dict[str, Any] | None:
        for f in self._facts.values():
            if f.get("user_id") == user_id and f.get("text") == text and f.get("is_active", True):
                return f
        return None

    async def get_supersession_chain(self, fact_id: str) -> list[dict[str, Any]]:
        chain = []
        current_id: str | None = fact_id
        visited: set[str] = set()
        while current_id and current_id not in visited:
            visited.add(current_id)
            fact = self._facts.get(current_id)
            if fact:
                chain.append(fact)
                current_id = fact.get("superseded_by")
            else:
                break
        return chain

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
        insight_id = str(uuid.uuid4())
        self._insights[insight_id] = {
            "id": insight_id,
            "text": text,
            "embedding": embedding,
            "user_id": user_id,
            "agent_id": agent_id,
            "confidence": confidence,
            "is_active": True,
            "metadata": metadata or {},
            "created_at": datetime.utcnow(),
        }
        return insight_id

    async def get_insights_from_facts(
        self,
        fact_ids: list[str],
        user_id: str,
        active_only: bool = True,
    ) -> list[dict[str, Any]]:
        fact_id_set = set(fact_ids)
        insight_ids = {
            link["insight_id"]
            for link in self._insight_facts
            if link["fact_id"] in fact_id_set
        }
        results = []
        for iid in insight_ids:
            insight = self._insights.get(iid)
            if insight and insight.get("user_id") == user_id and (
                not active_only or insight.get("is_active", True)
            ):
                results.append(insight)
        return results

    async def get_active_insights(
        self,
        user_id: str,
        agent_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return [
            i for i in self._insights.values()
            if i.get("user_id") == user_id and i.get("is_active", True)
        ]

    # ── Edges ──

    async def insert_edges(self, edges: list[dict[str, Any]]) -> int:
        self._edges.extend(edges)
        return len(edges)

    async def expand_session_context(
        self,
        sentence_ids: list[str],
        window: int = 3,
    ) -> list[dict[str, Any]]:
        """Expand by sentence_index proximity within the same session."""
        results: dict[str, dict[str, Any]] = {}
        for sid in sentence_ids:
            sent = self._sentences.get(sid)
            if not sent:
                continue
            session_id = sent["session_id"]
            target_idx = sent.get("sentence_index", 0)
            target_turn = sent.get("turn_number", 0)
            for s in self._sentences.values():
                if s["session_id"] != session_id:
                    continue
                # Expand within same turn ±window
                if (
                    s.get("turn_number") == target_turn
                    and abs(s.get("sentence_index", 0) - target_idx) <= window
                ):
                    results[s["id"]] = s
        return sorted(results.values(), key=lambda x: (x.get("turn_number", 0), x.get("sentence_index", 0)))

    # ── Join tables ──

    async def insert_fact_source(self, fact_id: str, sentence_id: str) -> None:
        self._fact_sources.append({"fact_id": fact_id, "sentence_id": sentence_id})

    async def insert_insight_fact(self, insight_id: str, fact_id: str) -> None:
        self._insight_facts.append({"insight_id": insight_id, "fact_id": fact_id})

    async def insert_insight_source(self, insight_id: str, sentence_id: str) -> None:
        self._insight_sources.append({"insight_id": insight_id, "sentence_id": sentence_id})

    async def get_source_sentences(self, fact_ids: list[str]) -> list[str]:
        fact_id_set = set(fact_ids)
        return list({
            link["sentence_id"]
            for link in self._fact_sources
            if link["fact_id"] in fact_id_set
        })

    async def get_sentences_by_ids(
        self, sentence_ids: list[str]
    ) -> list[dict[str, Any]]:
        id_set = set(sentence_ids)
        results = [
            s for s in self._sentences.values()
            if s["id"] in id_set and s.get("is_active", True)
        ]
        return sorted(
            results,
            key=lambda x: (x.get("session_id", ""), x.get("turn_number", 0), x.get("sentence_index", 0)),
        )

    # ── Sessions ──

    async def upsert_session(
        self,
        session_id: str,
        user_id: str,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._sessions[session_id] = {
            "id": session_id,
            "user_id": user_id,
            "agent_id": agent_id,
            "metadata": metadata or {},
            "started_at": datetime.utcnow(),
        }

    async def get_session(
        self,
        session_id: str,
        user_id: str,
    ) -> dict[str, Any] | None:
        session = self._sessions.get(session_id)
        if not session or session.get("user_id") != user_id:
            return None
        sentences = sorted(
            [s for s in self._sentences.values() if s.get("session_id") == session_id],
            key=lambda x: (x.get("turn_number", 0), x.get("sentence_index", 0)),
        )
        return {**session, "sentences": sentences}

    async def count_sessions(
        self,
        user_id: str,
        agent_id: str | None = None,
    ) -> int:
        count = 0
        for s in self._sessions.values():
            if s.get("user_id") != user_id:
                continue
            if agent_id is not None and s.get("agent_id") != agent_id:
                continue
            count += 1
        return count

    # ── Lifecycle ──

    async def delete_user(self, user_id: str) -> int:
        count = 0
        for store in [self._sentences, self._facts, self._insights, self._sessions]:
            keys = [k for k, v in store.items() if v.get("user_id") == user_id]
            for k in keys:
                del store[k]
                count += 1
        return count

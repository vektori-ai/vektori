"""In-memory storage backend. No persistence. For unit tests and CI."""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from typing import Any

from vektori.storage.base import StorageBackend


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _utcnow_naive() -> datetime:
    """Return current UTC timestamp as naive datetime for storage parity."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class MemoryBackend(StorageBackend):
    """
    In-memory storage backend. Fast, no deps, no persistence.

    Intended for unit tests and CI. Vector search is brute-force cosine.
    """

    def __init__(self) -> None:
        self._sentences: dict[str, dict[str, Any]] = {}
        self._facts: dict[str, dict[str, Any]] = {}
        self._edges: list[dict[str, Any]] = []
        self._fact_sources: list[dict[str, str]] = []  # [{fact_id, sentence_id}]
        self._sessions: dict[str, dict[str, Any]] = {}
        self._syntheses: dict[str, dict[str, Any]] = {}
        self._synthesis_facts: list[dict[str, str]] = []
        self._syntheses: dict[str, dict[str, Any]] = {}
        self._synthesis_facts: list[dict[str, str]] = []
        self._episodes: dict[str, dict[str, Any]] = {}
        self._episode_facts: list[dict[str, str]] = []  # [{episode_id, fact_id}]

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
                    "created_at": _utcnow_naive(),
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
        # Superseded by search_sentences_in_session (embedding-based).
        return []

    async def search_sentences_in_session(
        self,
        embedding: list[float],
        session_id: str,
        limit: int = 3,
        threshold: float = 0.75,
    ) -> list[str]:
        """In-memory cosine search over sentences in a session. Used for fact-source linking."""
        scored: list[tuple[float, str]] = []
        for s in self._sentences.values():
            if s.get("session_id") != session_id:
                continue
            if s.get("embedding") is None:
                continue
            sim = _cosine_similarity(embedding, s["embedding"])
            if sim >= threshold:
                scored.append((sim, s["id"]))
        scored.sort(key=lambda x: -x[0])
        return [r[1] for r in scored[:limit]]

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
        subject: str | None = None,
        confidence: float = 1.0,
        superseded_by_target: str | None = None,
        metadata: dict[str, Any] | None = None,
        event_time: datetime | None = None,
    ) -> str:
        fact_id = str(uuid.uuid4())
        self._facts[fact_id] = {
            "id": fact_id,
            "text": text,
            "embedding": embedding,
            "user_id": user_id,
            "agent_id": agent_id,
            "session_id": session_id,
            "subject": subject,
            "confidence": confidence,
            "mentions": 1,
            "superseded_by": superseded_by_target,
            "is_active": True,
            "metadata": metadata or {},
            "event_time": event_time,
            "created_at": _utcnow_naive(),
        }
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
        results = []
        for f in self._facts.values():
            if f.get("user_id") != user_id:
                continue
            if agent_id and f.get("agent_id") != agent_id:
                continue
            if session_id and f.get("session_id") != session_id:
                continue
            if subject and f.get("subject") != subject:
                continue
            if active_only and not f.get("is_active", True):
                continue
            if f.get("embedding") is None:
                continue
            et = f.get("event_time")
            if before_date and et and et > before_date:
                continue
            if after_date and et and et < after_date:
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
            f
            for f in self._facts.values()
            if f.get("user_id") == user_id and f.get("is_active", True)
        ]
        return results[offset : offset + limit]

    async def increment_fact_mentions(self, fact_id: str) -> None:
        if fact_id in self._facts:
            self._facts[fact_id]["mentions"] = self._facts[fact_id].get("mentions", 1) + 1

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
        return sorted(
            results.values(), key=lambda x: (x.get("turn_number", 0), x.get("sentence_index", 0))
        )

    # ── Join tables ──

    # ── Syntheses ──

    async def insert_synthesis(
        self,
        text: str,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> str:
        synthesis_id = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{user_id}::{text}"))
        if synthesis_id not in self._syntheses:
            self._syntheses[synthesis_id] = {
                "id": synthesis_id,
                "text": text,
                "embedding": embedding,
                "user_id": user_id,
                "agent_id": agent_id,
                "session_id": session_id,
                "is_active": True,
                "created_at": _utcnow_naive(),
            }
        return synthesis_id

    async def insert_synthesis_fact(self, synthesis_id: str, fact_id: str) -> None:
        # Dedup: don't add the same link twice
        for link in self._synthesis_facts:
            if link["synthesis_id"] == synthesis_id and link["fact_id"] == fact_id:
                return
        self._synthesis_facts.append({"synthesis_id": synthesis_id, "fact_id": fact_id})

    async def get_syntheses_for_facts(self, fact_ids: list[str]) -> list[dict[str, Any]]:
        if not fact_ids:
            return []
        fact_id_set = set(fact_ids)
        matched_synthesis_ids = {
            link["synthesis_id"] for link in self._synthesis_facts if link["fact_id"] in fact_id_set
        }
        return [
            {k: v for k, v in self._syntheses[eid].items() if k != "embedding"}
            for eid in matched_synthesis_ids
            if eid in self._syntheses and self._syntheses[eid].get("is_active", True)
        ]

    async def search_syntheses(
        self,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        results = []
        for ep in self._syntheses.values():
            if ep.get("user_id") != user_id:
                continue
            if agent_id and ep.get("agent_id") != agent_id:
                continue
            if not ep.get("is_active", True):
                continue
            emb = ep.get("embedding")
            if not emb:
                continue
            sim = _cosine_similarity(embedding, emb)
            row = {k: v for k, v in ep.items() if k != "embedding"}
            results.append({**row, "distance": 1.0 - sim})
        results.sort(key=lambda x: x["distance"])
        return results[:limit]

    async def insert_fact_source(self, fact_id: str, sentence_id: str) -> None:
        self._fact_sources.append({"fact_id": fact_id, "sentence_id": sentence_id})

    async def get_source_sentences(self, fact_ids: list[str]) -> list[str]:
        fact_id_set = set(fact_ids)
        return list(
            {link["sentence_id"] for link in self._fact_sources if link["fact_id"] in fact_id_set}
        )

    async def get_sentences_by_ids(self, sentence_ids: list[str]) -> list[dict[str, Any]]:
        id_set = set(sentence_ids)
        results = [
            s for s in self._sentences.values() if s["id"] in id_set and s.get("is_active", True)
        ]
        return sorted(
            results,
            key=lambda x: (
                x.get("session_id", ""),
                x.get("turn_number", 0),
                x.get("sentence_index", 0),
            ),
        )


    # ── Syntheses ──

    async def insert_episode(
        self,
        text: str,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> str:
        episode_id = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{user_id}::{text}"))
        if episode_id not in self._episodes:
            self._episodes[episode_id] = {
                "id": episode_id,
                "text": text,
                "embedding": embedding,
                "user_id": user_id,
                "agent_id": agent_id,
                "session_id": session_id,
                "is_active": True,
                "created_at": _utcnow_naive(),
            }
        return episode_id

    async def insert_episode_fact(self, episode_id: str, fact_id: str) -> None:
        # Dedup: don't add the same link twice
        for link in self._episode_facts:
            if link["episode_id"] == episode_id and link["fact_id"] == fact_id:
                return
        self._episode_facts.append({"episode_id": episode_id, "fact_id": fact_id})

    async def get_episodes_for_facts(self, fact_ids: list[str]) -> list[dict[str, Any]]:
        if not fact_ids:
            return []
        fact_id_set = set(fact_ids)
        matched_episode_ids = {
            link["episode_id"] for link in self._episode_facts if link["fact_id"] in fact_id_set
        }
        return [
            {k: v for k, v in self._episodes[eid].items() if k != "embedding"}
            for eid in matched_episode_ids
            if eid in self._episodes and self._episodes[eid].get("is_active", True)
        ]

    async def search_episodes(
        self,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        results = []
        for ep in self._episodes.values():
            if ep.get("user_id") != user_id:
                continue
            if agent_id and ep.get("agent_id") != agent_id:
                continue
            if not ep.get("is_active", True):
                continue
            emb = ep.get("embedding")
            if not emb:
                continue
            sim = _cosine_similarity(embedding, emb)
            row = {k: v for k, v in ep.items() if k != "embedding"}
            results.append({**row, "distance": 1.0 - sim})
        results.sort(key=lambda x: x["distance"])
        return results[:limit]

    async def upsert_session(
        self,
        session_id: str,
        user_id: str,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        started_at: datetime | None = None,
    ) -> None:
        self._sessions[session_id] = {
            "id": session_id,
            "user_id": user_id,
            "agent_id": agent_id,
            "metadata": metadata or {},
            "started_at": started_at or _utcnow_naive(),
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
        for store in [self._sentences, self._facts, self._syntheses, self._syntheses, self._episodes, self._sessions]:
            keys = [k for k, v in store.items() if v.get("user_id") == user_id]
            for k in keys:
                del store[k]
                count += 1
        return count

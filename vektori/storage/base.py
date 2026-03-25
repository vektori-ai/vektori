"""Abstract storage backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class StorageBackend(ABC):
    """
    Abstract storage interface. All backends implement this.

    Backends: SQLite (zero-config default), PostgreSQL+pgvector (production),
              Memory (unit tests / CI).

    The interface maps directly to the three-layer graph schema:
      - Sentences (L2): raw conversation nodes + NEXT edges
      - Facts (L0): LLM-extracted statements, primary vector search surface
      - Insights (L1): inferred patterns, discovered via graph traversal
      - Join tables: fact_sources, insight_facts, insight_sources
    """

    # ── Sentences ──

    @abstractmethod
    async def upsert_sentences(
        self,
        sentences: list[dict[str, Any]],
        embeddings: list[list[float]],
        user_id: str,
        agent_id: str | None = None,
    ) -> int:
        """Upsert sentences. ON CONFLICT (content_hash) → increment mentions."""
        ...

    @abstractmethod
    async def search_sentences(
        self,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    async def find_sentences_by_similarity(
        self,
        quotes: list[str],
        session_id: str,
        threshold: float = 0.75,
    ) -> list[str]:
        """Find sentence IDs semantically similar to given quotes. Used to link facts/insights to source sentences."""
        ...

    # ── Facts ──

    @abstractmethod
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
        """Insert a fact and return its UUID."""
        ...

    @abstractmethod
    async def search_facts(
        self,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        limit: int = 10,
        active_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Vector search over facts. Results include a 'distance' field (cosine distance)."""
        ...

    @abstractmethod
    async def get_active_facts(
        self,
        user_id: str,
        agent_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    async def deactivate_fact(
        self,
        fact_id: str,
        superseded_by: str | None = None,
    ) -> None:
        """Mark a fact as inactive (conflict resolution). Never deletes."""
        ...

    @abstractmethod
    async def find_fact_by_text(
        self,
        user_id: str,
        text: str,
        agent_id: str | None = None,
    ) -> dict[str, Any] | None:
        ...

    @abstractmethod
    async def get_supersession_chain(self, fact_id: str) -> list[dict[str, Any]]:
        """Return full chain: [oldest superseded fact, ..., current active fact]."""
        ...

    # ── Insights ──

    @abstractmethod
    async def insert_insight(
        self,
        text: str,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Insert an insight and return its UUID."""
        ...

    @abstractmethod
    async def get_insights_from_facts(
        self,
        fact_ids: list[str],
        active_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Graph traversal: JOIN insight_facts WHERE fact_id IN (...). NOT vector search."""
        ...

    @abstractmethod
    async def get_active_insights(
        self,
        user_id: str,
        agent_id: str | None = None,
    ) -> list[dict[str, Any]]:
        ...

    # ── Edges ──

    @abstractmethod
    async def insert_edges(self, edges: list[dict[str, Any]]) -> int:
        """Insert sentence edges (NEXT, contradiction). ON CONFLICT DO NOTHING."""
        ...

    @abstractmethod
    async def expand_session_context(
        self,
        sentence_ids: list[str],
        window: int = 3,
    ) -> list[dict[str, Any]]:
        """For each sentence, grab ±window surrounding sentences via NEXT edges or sentence_index range."""
        ...

    # ── Join tables ──

    @abstractmethod
    async def insert_fact_source(self, fact_id: str, sentence_id: str) -> None:
        """Link a fact to the sentence it was extracted from."""
        ...

    @abstractmethod
    async def insert_insight_fact(self, insight_id: str, fact_id: str) -> None:
        """Link an insight to a related fact. This is the key L1↔L0 bridge."""
        ...

    @abstractmethod
    async def insert_insight_source(self, insight_id: str, sentence_id: str) -> None:
        """Link an insight to a source sentence."""
        ...

    @abstractmethod
    async def get_source_sentences(self, fact_ids: list[str]) -> list[str]:
        """Return sentence IDs that are sources for the given facts (via fact_sources)."""
        ...

    # ── Sessions ──

    @abstractmethod
    async def upsert_session(
        self,
        session_id: str,
        user_id: str,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        ...

    @abstractmethod
    async def get_session(
        self,
        session_id: str,
        user_id: str,
    ) -> dict[str, Any] | None:
        ...

    # ── Lifecycle ──

    @abstractmethod
    async def initialize(self) -> None:
        """Create tables, indexes, extensions if they don't exist. Idempotent."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close connections / release resources."""
        ...

    @abstractmethod
    async def delete_user(self, user_id: str) -> int:
        """Cascade delete all data for a user (GDPR). Returns rows deleted."""
        ...

"""Main Vektori client — the primary public interface."""

from __future__ import annotations

import logging
from typing import Any

from vektori.config import QualityConfig, VektoriConfig
from vektori.ingestion.filter import is_quality_sentence
from vektori.ingestion.hasher import generate_sentence_id
from vektori.ingestion.splitter import split_sentences

logger = logging.getLogger(__name__)


class Vektori:
    """
    Open-source memory engine for AI agents.

    Stores conversational context in a three-layer sentence graph:
      L0 — Facts: primary vector search surface (short, crisp, LLM-extracted)
      L1 — Insights: cross-session patterns, discovered via graph traversal
      L2 — Sentences: raw conversation with sequential NEXT edges

    Usage:
        v = Vektori()  # SQLite, zero config
        await v.add(messages, session_id="s1", user_id="u1")
        results = await v.search("how does user prefer to communicate?", user_id="u1")
        await v.close()
    """

    def __init__(
        self,
        database_url: str | None = None,
        storage_backend: str | None = None,
        embedding_model: str = "openai:text-embedding-3-small",
        extraction_model: str = "openai:gpt-4o-mini",
        embedding_dimension: int = 1536,
        quality_config: QualityConfig | None = None,
        default_top_k: int = 10,
        context_window: int = 3,
        temporal_decay_rate: float = 0.001,
        async_extraction: bool = True,
        config: VektoriConfig | None = None,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            resolved_backend = storage_backend or (
                "postgres" if database_url else "sqlite"
            )
            self.config = VektoriConfig(
                database_url=database_url,
                storage_backend=resolved_backend,
                embedding_model=embedding_model,
                extraction_model=extraction_model,
                embedding_dimension=embedding_dimension,
                quality_config=quality_config or QualityConfig(),
                default_top_k=default_top_k,
                context_window=context_window,
                temporal_decay_rate=temporal_decay_rate,
                async_extraction=async_extraction,
            )

        self._initialized = False
        self.db = None
        self.embedder = None
        self.llm = None
        self._extractor = None
        self._search = None
        self._worker = None

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self._initialize()

    async def _initialize(self) -> None:
        """Initialize storage, embedder, and LLM providers."""
        from vektori.ingestion.extractor import FactExtractor
        from vektori.models.factory import create_embedder, create_llm
        from vektori.retrieval.search import SearchPipeline
        from vektori.storage.factory import create_storage
        from vektori.utils.async_worker import AsyncExtractionWorker

        self.db = await create_storage(self.config)
        self.embedder = create_embedder(self.config.embedding_model)
        self.llm = create_llm(self.config.extraction_model)
        self._extractor = FactExtractor(db=self.db, embedder=self.embedder, llm=self.llm)
        self._search = SearchPipeline(
            db=self.db,
            embedder=self.embedder,
            temporal_decay_rate=self.config.temporal_decay_rate,
        )
        if self.config.async_extraction:
            self._worker = AsyncExtractionWorker(extractor=self._extractor)

        self._initialized = True
        logger.info("Vektori initialized (backend=%s)", self.config.storage_backend)

    async def add(
        self,
        messages: list[dict[str, str]],
        session_id: str,
        user_id: str,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Store a conversation session into the memory graph.

        Synchronous path (fast — user waits):
          split → quality filter → embed → upsert sentences + NEXT edges

        Async path (background — user does NOT wait):
          LLM fact + insight extraction → conflict resolution → store facts/insights

        Args:
            messages: [{"role": "user"|"assistant", "content": "..."}]
            session_id: Unique ID for this conversation session.
            user_id: Owner of these memories.
            agent_id: Optional agent scoping (memories isolated per agent).
            metadata: Arbitrary metadata stored with the session.

        Returns:
            {"status": "ok", "sentences_stored": N, "extraction": "processing"|"done"|"skipped"}
        """
        await self._ensure_initialized()

        all_sentences = []
        for turn_num, msg in enumerate(messages):
            raw_sents = split_sentences(msg["content"])
            for idx, text in enumerate(raw_sents):
                # Only user sentences go into the sentence graph.
                # Assistant messages are used during fact extraction but not stored as nodes.
                if msg["role"] == "user" and (
                    not self.config.quality_config.enabled
                    or is_quality_sentence(text, self.config.quality_config)
                ):
                    all_sentences.append({
                        "text": text,
                        "session_id": session_id,
                        "turn_number": turn_num,
                        "sentence_index": idx,
                        "role": msg["role"],
                        "id": generate_sentence_id(session_id, f"{turn_num}_{idx}", text),
                    })

        if not all_sentences:
            return {"status": "ok", "sentences_stored": 0, "extraction": "skipped"}

        # Batch embed all sentences in one API call
        texts = [s["text"] for s in all_sentences]
        embeddings = await self.embedder.embed_batch(texts)

        # Upsert — ON CONFLICT increments mentions counter (IDF weighting)
        await self.db.upsert_sentences(all_sentences, embeddings, user_id, agent_id)

        # Create sequential NEXT edges within this session
        edges = [
            {
                "source_id": all_sentences[i]["id"],
                "target_id": all_sentences[i + 1]["id"],
                "edge_type": "next",
                "weight": 1.0,
            }
            for i in range(len(all_sentences) - 1)
        ]
        if edges:
            await self.db.insert_edges(edges)

        await self.db.upsert_session(session_id, user_id, agent_id, metadata or {})

        # Trigger fact + insight extraction
        extraction_status = "skipped"
        if self._worker is not None:
            self._worker.schedule(messages, session_id, user_id, agent_id)
            extraction_status = "processing"
        elif not self.config.async_extraction:
            await self._extractor.extract(messages, session_id, user_id, agent_id)
            extraction_status = "done"

        return {
            "status": "ok",
            "sentences_stored": len(all_sentences),
            "extraction": extraction_status,
        }

    async def search(
        self,
        query: str,
        user_id: str,
        agent_id: str | None = None,
        depth: str = "l1",
        top_k: int | None = None,
        context_window: int | None = None,
        include_superseded: bool = False,
    ) -> dict[str, Any]:
        """
        Retrieve relevant memories for a query.

        Args:
            query: Natural language query.
            user_id: Whose memories to search.
            agent_id: Optional agent scoping.
            depth: "l0" facts only | "l1" facts+insights+sentences | "l2" full story with context window.
            top_k: Max facts to return.
            context_window: ±N sentences around source sentences (L2 only).
            include_superseded: Include overridden/outdated facts.

        Returns:
            {
              "facts": [...],           # always present
              "insights": [...],        # l1 and l2
              "sentences": [...]        # l1 (source sentences) and l2 (expanded context)
            }
        """
        await self._ensure_initialized()
        return await self._search.search(
            query=query,
            user_id=user_id,
            agent_id=agent_id,
            depth=depth,
            top_k=top_k or self.config.default_top_k,
            context_window=context_window or self.config.context_window,
            include_superseded=include_superseded,
        )

    async def get_facts(
        self,
        user_id: str,
        agent_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get all active facts for a user."""
        await self._ensure_initialized()
        return await self.db.get_active_facts(user_id, agent_id)

    async def get_insights(
        self,
        user_id: str,
        agent_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get all active insights for a user."""
        await self._ensure_initialized()
        return await self.db.get_active_insights(user_id, agent_id)

    async def get_session(
        self,
        session_id: str,
        user_id: str,
    ) -> dict[str, Any] | None:
        """Get session metadata and all sentences in a session."""
        await self._ensure_initialized()
        return await self.db.get_session(session_id, user_id)

    async def get_fact_history(
        self,
        user_id: str,
        fact_id: str,
    ) -> list[dict[str, Any]]:
        """Get the full supersession chain for a fact (conflict/change history)."""
        await self._ensure_initialized()
        return await self.db.get_supersession_chain(fact_id)

    async def delete_user(self, user_id: str) -> int:
        """Delete all data for a user (GDPR). Returns number of rows deleted."""
        await self._ensure_initialized()
        return await self.db.delete_user(user_id)

    async def close(self) -> None:
        """Close database connections and gracefully shut down background workers."""
        if self._worker:
            await self._worker.shutdown()
        if self.db:
            await self.db.close()

    async def __aenter__(self) -> "Vektori":
        await self._ensure_initialized()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

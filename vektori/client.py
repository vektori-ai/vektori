"""Main Vektori client — the primary public interface."""

from __future__ import annotations

import logging
from typing import Any

from vektori.config import QualityConfig, VektoriConfig

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
        self._pipeline = None
        self._expander = None

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self._initialize()

    async def _initialize(self) -> None:
        """Initialize storage, embedder, LLM providers, pipeline, and search."""
        from vektori.ingestion.extractor import FactExtractor
        from vektori.ingestion.pipeline import IngestionPipeline
        from vektori.models.factory import create_embedder, create_llm
        from vektori.retrieval.search import SearchPipeline
        from vektori.storage.factory import create_storage

        self.db = await create_storage(self.config)
        self.embedder = create_embedder(self.config.embedding_model)
        self.llm = create_llm(self.config.extraction_model)
        self._extractor = FactExtractor(
            db=self.db,
            embedder=self.embedder,
            llm=self.llm,
            max_facts=self.config.max_facts,
            max_insights=self.config.max_insights,
            max_input_tokens=self.config.max_extraction_input_tokens,
            max_output_tokens=self.config.max_extraction_output_tokens,
        )
        self._search = SearchPipeline(
            db=self.db,
            embedder=self.embedder,
            temporal_decay_rate=self.config.temporal_decay_rate,
            min_score=self.config.min_retrieval_score,
        )
        self._pipeline = IngestionPipeline(
            db=self.db,
            embedder=self.embedder,
            extractor=self._extractor,
            quality_config=self.config.quality_config,
            async_extraction=self.config.async_extraction,
            token_batch_threshold=self.config.token_batch_threshold,
        )

        from vektori.retrieval.expander import QueryExpander
        self._expander = QueryExpander(
            llm=self.llm,
            n_variants=self.config.expansion_queries,
        )

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
            {"status": "ok", "sentences_stored": N, "extraction": "queued"|"done"|"skipped"}
        """
        await self._ensure_initialized()
        return await self._pipeline.ingest(messages, session_id, user_id, agent_id, metadata)

    async def search(
        self,
        query: str,
        user_id: str,
        agent_id: str | None = None,
        depth: str = "l1",
        top_k: int | None = None,
        context_window: int | None = None,
        include_superseded: bool = False,
        expand: bool = False,
    ) -> dict[str, Any]:
        """
        Retrieve relevant memories for a query.

        Args:
            query: Natural language query.
            user_id: Whose memories to search.
            agent_id: Optional agent scoping.
            depth: "l0" | "l1" | "l2". Ignored when expand=True (always L1).
            top_k: Max facts to return.
            context_window: ±N sentences around source sentences (L2 only).
            include_superseded: Include overridden/outdated facts.
            expand: If True, use LLM to generate query variants and search
                    concurrently. Results are merged then returned at L1.
                    Use for vague/indirect queries. Adds one LLM call.

        Returns:
            {
              "facts": [...],
              "insights": [...],        # l1, l2, and expand=True
              "sentences": [...]        # l1, l2, and expand=True
            }
        """
        await self._ensure_initialized()

        if self.config.enable_retrieval_gate:
            from vektori.retrieval.gate import should_retrieve
            if not should_retrieve(query):
                logger.debug("Retrieval gate: skipping DB lookup for query=%r", query[:60])
                result: dict[str, Any] = {"facts": []}
                if depth in ("l1", "l2") or expand:
                    result["insights"] = []
                    result["sentences"] = []
                return result

        k = top_k or self.config.default_top_k

        if expand:
            # Generate paraphrase variants → concurrent L0 searches → single L1 graph pass
            queries = await self._expander.expand(query)
            return await self._search.search_expanded(
                queries=queries,
                user_id=user_id,
                agent_id=agent_id,
                top_k=k,
            )

        return await self._search.search(
            query=query,
            user_id=user_id,
            agent_id=agent_id,
            depth=depth,
            top_k=k,
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

    async def generate_insights(
        self,
        user_id: str,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        """Manually trigger cross-session insight generation for a user."""
        await self._ensure_initialized()
        return await self._extractor.extract_cross_session_insights(user_id, agent_id)

    async def close(self) -> None:
        """Close database connections and gracefully shut down background workers."""
        if self._pipeline and self._pipeline.worker:
            await self._pipeline.worker.shutdown()
        if self.db:
            await self.db.close()

    async def __aenter__(self) -> "Vektori":
        await self._ensure_initialized()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

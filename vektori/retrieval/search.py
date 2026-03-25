"""Main retrieval pipeline — L0 / L1 / L2 tiered search."""

from __future__ import annotations

import logging
from typing import Any

from vektori.retrieval.scoring import score_and_rank
from vektori.storage.base import StorageBackend

logger = logging.getLogger(__name__)


class SearchPipeline:
    """
    Three-layer retrieval pipeline.

    L0 — Facts only (cheapest, ~50-200 tokens):
        Vector search over facts. Best cosine match on short, crisp text.

    L1 — Facts + Insights (default, ~200-500 tokens):
        Facts via vector search + insights discovered via insight_facts JOIN.
        Insights are NOT vector searched — they come along via graph traversal.

    L2 — Full story (~1000-3000 tokens):
        Facts + insights + source sentences + session context (±N via NEXT edges).
    """

    def __init__(
        self,
        db: StorageBackend,
        embedder: Any,
        temporal_decay_rate: float = 0.001,
    ) -> None:
        self.db = db
        self.embedder = embedder
        self.temporal_decay_rate = temporal_decay_rate

    async def search(
        self,
        query: str,
        user_id: str,
        agent_id: str | None = None,
        depth: str = "l1",
        top_k: int = 10,
        context_window: int = 3,
        include_superseded: bool = False,
    ) -> dict[str, Any]:
        """
        Retrieve relevant memories for a query.

        The retrieval flow:
          1. Embed query
          2. Vector search over FACTS (L0) — entry point
          3. Discover INSIGHTS linked to matched facts via insight_facts (L1)
          4. Trace facts to SOURCE SENTENCES via fact_sources (L2)
          5. Expand session context via NEXT edges ±window (L2)
        """
        query_embedding = await self.embedder.embed(query)

        # ── Step 1: Vector search over FACTS ──
        # Facts are short + crisp → best cosine match for direct queries.
        seed_facts = await self.db.search_facts(
            embedding=query_embedding,
            user_id=user_id,
            agent_id=agent_id,
            limit=top_k,
            active_only=not include_superseded,
        )

        if not seed_facts:
            return {"facts": [], "insights": [], "sentences": []}

        scored_facts = score_and_rank(seed_facts, self.temporal_decay_rate)

        if depth == "l0":
            return {"facts": scored_facts[:top_k]}

        # ── Step 2: Discover INSIGHTS linked to matched facts ──
        # This is graph traversal, NOT vector search.
        # insight_facts JOIN table: fact → insight.
        seed_fact_ids = [f["id"] for f in scored_facts[:top_k]]
        related_insights = await self.db.get_insights_from_facts(
            fact_ids=seed_fact_ids,
            active_only=True,
        )

        if depth == "l1":
            return {
                "facts": scored_facts[:top_k],
                "insights": related_insights,
            }

        # ── Step 3: Trace facts down to source sentences ──
        source_sentence_ids = await self.db.get_source_sentences(seed_fact_ids)

        # ── Step 4: Session expansion via NEXT edges ──
        expanded_sentences = await self.db.expand_session_context(
            sentence_ids=source_sentence_ids,
            window=context_window,
        )

        # ── L2: Return all three layers ──
        return {
            "facts": scored_facts[:top_k],
            "insights": related_insights,
            "sentences": expanded_sentences,
        }

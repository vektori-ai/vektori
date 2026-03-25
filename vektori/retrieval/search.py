"""Main retrieval pipeline — L0 / L1 / L2 tiered search."""

from __future__ import annotations

import logging
from typing import Any

from vektori.models.base import EmbeddingProvider
from vektori.retrieval.scoring import explain_score, score_and_rank
from vektori.storage.base import StorageBackend

logger = logging.getLogger(__name__)

VALID_DEPTHS = {"l0", "l1", "l2"}


class SearchPipeline:
    """Three-layer retrieval pipeline.

    L0 — Facts only (~50-200 tokens):
        Vector search over facts. Entry point for all retrieval.
        Facts are short and crisp → best cosine match for direct queries.

    L1 — Facts + Insights + Source Sentences (~300-800 tokens):
        Facts + insights via graph traversal + the exact sentences each fact
        was extracted from (via fact_sources). No context expansion — just the
        precise moments in conversation that produced the facts.
        This is the default depth.

    L2 — Full story (~1000-3000 tokens):
        Everything in L1, plus full session context window (±N sentences around
        each source sentence via NEXT edges). Reconstructs the conversational
        flow surrounding each matched moment.

    Fast path:
        When the storage backend supports it (postgres.supports_single_query),
        L2 executes as a single CTE round trip instead of 4 separate queries.
        L0/L1 always use the step-by-step path (they're already 1-2 queries).
    """

    def __init__(
        self,
        db: StorageBackend,
        embedder: EmbeddingProvider,
        temporal_decay_rate: float = 0.001,
        use_mentions: bool = True,
        debug: bool = False,
    ) -> None:
        self.db = db
        self.embedder = embedder
        self.temporal_decay_rate = temporal_decay_rate
        self.use_mentions = use_mentions
        # debug=True logs score breakdowns for each returned fact
        self.debug = debug

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
        """Retrieve relevant memories for a query.

        Args:
            query: Natural language query string.
            user_id: Whose memories to search.
            agent_id: Optional agent scoping. None means search all agents.
            depth: "l0" | "l1" | "l2". Invalid values raise ValueError.
            top_k: Max facts to return. Insights and sentences are not capped
                   here — they're derived from the returned facts and bounded
                   by how many links exist in the graph.
            context_window: ±N sentences around each source sentence (L2 only).
            include_superseded: If True, also return overridden/old facts.
                                 Useful for historical queries.

        Returns:
            {
              "facts":     list[dict],   # always present
              "insights":  list[dict],   # l1 and l2 only
              "sentences": list[dict],   # l2 only
            }

        Raises:
            ValueError: if depth is not one of "l0", "l1", "l2".
        """
        if depth not in VALID_DEPTHS:
            raise ValueError(
                f"Invalid depth '{depth}'. Must be one of: {sorted(VALID_DEPTHS)}"
            )

        query_embedding = await self.embedder.embed(query)

        # L2 on Postgres: single CTE round trip — faster than 4 separate calls.
        # Only use the fast path for active-only queries; include_superseded
        # requires the stepped path since the CTE always filters is_active=true.
        if (
            depth == "l2"
            and not include_superseded
            and getattr(self.db, "supports_single_query", False)
        ):
            return await self._search_l2_fast(
                query_embedding, user_id, agent_id, top_k, context_window
            )

        return await self._search_stepped(
            query_embedding, user_id, agent_id, depth,
            top_k, context_window, include_superseded
        )

    # ── Step-by-step path (all backends, L0/L1, and L2 fallback) ──────────────

    async def _search_stepped(
        self,
        query_embedding: list[float],
        user_id: str,
        agent_id: str | None,
        depth: str,
        top_k: int,
        context_window: int,
        include_superseded: bool,
    ) -> dict[str, Any]:
        # ── Step 1: Vector search over FACTS (L0 entry point) ─────────────────
        seed_facts = await self.db.search_facts(
            embedding=query_embedding,
            user_id=user_id,
            agent_id=agent_id,
            limit=top_k,
            active_only=not include_superseded,
        )

        if not seed_facts:
            logger.debug("search: no facts found for user=%s", user_id)
            return _empty(depth)

        scored_facts = score_and_rank(
            seed_facts,
            temporal_decay_rate=self.temporal_decay_rate,
            use_mentions=self.use_mentions,
        )

        if self.debug:
            for f in scored_facts[:top_k]:
                logger.debug(explain_score(f))

        if depth == "l0":
            return {"facts": _clean(scored_facts[:top_k])}

        # ── Step 2: Discover INSIGHTS via graph traversal (L1) ────────────────
        # NOT vector search. JOIN on insight_facts where fact_id IN seed_fact_ids.
        seed_fact_ids = [f["id"] for f in scored_facts[:top_k]]
        related_insights = await self.db.get_insights_from_facts(
            fact_ids=seed_fact_ids,
            user_id=user_id,
            active_only=True,
        )

        # ── Step 3: Trace facts → source sentences (L1 + L2) ─────────────────
        source_sentence_ids = await self.db.get_source_sentences(seed_fact_ids)

        if not source_sentence_ids:
            # Facts exist but haven't been linked to sentences yet
            # (extraction may still be in-flight). Return what we have.
            logger.debug(
                "search L1/L2: no source sentences for facts %s (extraction pending?)",
                seed_fact_ids,
            )
            return {
                "facts": _clean(scored_facts[:top_k]),
                "insights": related_insights,
                "sentences": [],
            }

        if depth == "l1":
            # L1: source sentences only — exact moments facts came from, no expansion.
            source_sentences = await self.db.get_sentences_by_ids(source_sentence_ids)
            return {
                "facts": _clean(scored_facts[:top_k]),
                "insights": related_insights,
                "sentences": source_sentences,
            }

        # ── Step 4: Session expansion ±window (L2 only) ───────────────────────
        expanded_sentences = await self.db.expand_session_context(
            sentence_ids=source_sentence_ids,
            window=context_window,
        )

        return {
            "facts": _clean(scored_facts[:top_k]),
            "insights": related_insights,
            "sentences": _dedup(expanded_sentences),
        }

    # ── Single-query fast path (Postgres L2 only) ─────────────────────────────

    async def _search_l2_fast(
        self,
        query_embedding: list[float],
        user_id: str,
        agent_id: str | None,
        top_k: int,
        context_window: int,
    ) -> dict[str, Any]:
        """Execute full L2 retrieval in one CTE round trip on Postgres."""
        raw = await self.db.search_l2_single_query(
            embedding=query_embedding,
            user_id=user_id,
            agent_id=agent_id,
            limit=top_k,
            window=context_window,
        )

        scored_facts = score_and_rank(
            raw.get("facts", []),
            temporal_decay_rate=self.temporal_decay_rate,
            use_mentions=self.use_mentions,
        )

        if self.debug:
            for f in scored_facts[:top_k]:
                logger.debug(explain_score(f))

        return {
            "facts": _clean(scored_facts[:top_k]),
            "insights": raw.get("insights", []),
            "sentences": _dedup(raw.get("sentences", [])),
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _empty(depth: str) -> dict[str, Any]:
    """Return the correct empty structure for a given depth."""
    result: dict[str, Any] = {"facts": []}
    if depth in ("l1", "l2"):
        result["insights"] = []
        result["sentences"] = []
    return result


def _clean(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip internal scoring debug fields before returning to the caller."""
    return [{k: v for k, v in f.items() if k != "_score_components"} for f in facts]


def _dedup(sentences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate sentences by ID, preserving order.

    Needed because multiple facts can share the same source sentence —
    expand_session_context returns it once per fact without dedup at the
    SQL level (DISTINCT only helps within a single fact's expansion).
    """
    seen: set[str] = set()
    result = []
    for s in sentences:
        sid = s.get("id")
        if sid and sid not in seen:
            seen.add(sid)
            result.append(s)
    return result

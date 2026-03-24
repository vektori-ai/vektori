"""Main retrieval pipeline — L0 / L1 / L2 tiered search."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from vektori.models.base import EmbeddingProvider
from vektori.retrieval.scoring import explain_score, score_and_rank
from vektori.retrieval.temporal import TemporalQueryParser
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

    Sentence fallback:
        When no facts exist yet (extraction still in-flight), falls through to
        sentence-level vector search so retrieval degrades gracefully rather
        than returning empty.

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
        min_score: float = 0.3,
        debug: bool = False,
    ) -> None:
        self.db = db
        self.embedder = embedder
        self.temporal_decay_rate = temporal_decay_rate
        self.use_mentions = use_mentions
        # Facts below this score are dropped — empty result signals "not found" (abstention).
        # 0.3 is a reasonable floor: irrelevant facts score ~0.1-0.2, weak matches ~0.3-0.4.
        # Set to 0.0 to disable (always return something regardless of relevance).
        self.min_score = min_score
        # debug=True logs score breakdowns for each returned fact
        self.debug = debug
        self._temporal_parser = TemporalQueryParser()

    async def search(
        self,
        query: str,
        user_id: str,
        agent_id: str | None = None,
        subject: str | None = None,
        session_id: str | None = None,
        depth: str = "l1",
        top_k: int = 10,
        context_window: int = 3,
        include_superseded: bool = False,
        before_date: datetime | None = None,
        after_date: datetime | None = None,
        parse_temporal: bool = True,
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
            before_date/after_date: Explicit temporal filters on fact event_time.
            parse_temporal: If True and no explicit dates provided, auto-parse
                            temporal expressions from the query (e.g. "last week").

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

        # Auto-parse temporal window from query when no explicit dates given
        if parse_temporal and before_date is None and after_date is None:
            window = self._temporal_parser.parse(query)
            if window:
                before_date = window.before_date
                after_date = window.after_date
                logger.debug(
                    "Temporal window parsed from query: after=%s before=%s",
                    after_date, before_date,
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
                query_embedding, user_id, agent_id, subject, session_id, top_k, context_window,
                before_date=before_date, after_date=after_date,
            )

        return await self._search_stepped(
            query_embedding, user_id, agent_id, subject, session_id, depth,
            top_k, context_window, include_superseded,
            before_date=before_date, after_date=after_date,
        )

    # ── Step-by-step path (all backends, L0/L1, and L2 fallback) ──────────────

    async def _search_stepped(
        self,
        query_embedding: list[float],
        user_id: str,
        agent_id: str | None,
        subject: str | None,
        session_id: str | None,
        depth: str,
        top_k: int,
        context_window: int,
        include_superseded: bool,
        before_date: datetime | None = None,
        after_date: datetime | None = None,
    ) -> dict[str, Any]:
        # ── Step 1: Vector search over FACTS (L0 entry point) ─────────────────
        seed_facts = await self.db.search_facts(
            embedding=query_embedding,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            subject=subject,
            limit=top_k,
            active_only=not include_superseded,
            before_date=before_date,
            after_date=after_date,
        )

        # Sentence fallback: no facts yet (extraction still in-flight)
        if not seed_facts:
            logger.debug(
                "search: no facts for user=%s, falling back to sentence search", user_id
            )
            return await self._sentence_fallback(query_embedding, user_id, agent_id, top_k, depth)

        scored_facts = score_and_rank(
            seed_facts,
            temporal_decay_rate=self.temporal_decay_rate,
            use_mentions=self.use_mentions,
        )

        # Apply min score floor
        if self.min_score > 0:
            scored_facts = [f for f in scored_facts if f["score"] >= self.min_score]

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

        # Propagate max linked-fact score to insights so they're relevance-ordered
        related_insights = _score_insights(related_insights, scored_facts[:top_k])

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
        subject: str | None,
        session_id: str | None,
        top_k: int,
        context_window: int,
        before_date: datetime | None = None,
        after_date: datetime | None = None,
    ) -> dict[str, Any]:
        """Execute full L2 retrieval in one CTE round trip on Postgres."""
        raw = await self.db.search_l2_single_query(
            embedding=query_embedding,
            user_id=user_id,
            agent_id=agent_id,
            subject=subject,
            session_id=session_id,
            limit=top_k,
            window=context_window,
            before_date=before_date,
            after_date=after_date,
        )

        scored_facts = score_and_rank(
            raw.get("facts", []),
            temporal_decay_rate=self.temporal_decay_rate,
            use_mentions=self.use_mentions,
        )

        if self.min_score > 0:
            scored_facts = [f for f in scored_facts if f["score"] >= self.min_score]

        if self.debug:
            for f in scored_facts[:top_k]:
                logger.debug(explain_score(f))

        insights = _score_insights(raw.get("insights", []), scored_facts[:top_k])

        return {
            "facts": _clean(scored_facts[:top_k]),
            "insights": insights,
            "sentences": _dedup(raw.get("sentences", [])),
        }

    # ── Expanded search path (L1 only) ───────────────────────────────────────

    async def search_expanded(
        self,
        queries: list[str],
        user_id: str,
        agent_id: str | None = None,
        subject: str | None = None,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """
        Expanded retrieval: concurrent L0 fact searches across multiple query
        variants, merged into a single L1 result.

        Flow:
          embed_batch(queries)
            → concurrent search_facts() per embedding          (L0 × N)
            → merge by fact ID, keep min distance per fact
            → score_and_rank(merged_facts)
            → get_insights_from_facts(top_fact_ids)            (L1 graph)
            → get_source_sentences(top_fact_ids)               (L1 graph)
            → return {facts, insights, sentences}

        Always returns L1 depth. L2 context expansion is intentionally excluded
        — expansion already widens fact recall; adding window expansion on top
        would be redundant and expensive.
        """
        if not queries:
            return {"facts": [], "insights": [], "sentences": []}

        # Single embed_batch call for all query variants
        embeddings = await self.embedder.embed_batch(queries)

        # Concurrent L0 fact searches — one per query variant
        async def _search_one(embedding: list[float]) -> list[dict[str, Any]]:
            return await self.db.search_facts(
                embedding=embedding,
                user_id=user_id,
                agent_id=agent_id,
                subject=subject,
                limit=top_k,
                active_only=True,
            )

        all_results = await asyncio.gather(*[_search_one(emb) for emb in embeddings])

        # Merge: per fact ID keep minimum distance (closest match across all variants)
        best_by_id: dict[str, dict[str, Any]] = {}
        for result_set in all_results:
            for fact in result_set:
                fid = fact["id"]
                if fid not in best_by_id or fact.get("distance", 1.0) < best_by_id[fid].get("distance", 1.0):
                    best_by_id[fid] = fact

        merged_facts = list(best_by_id.values())

        if not merged_facts:
            logger.debug("search_expanded: no facts found, falling back to sentence search")
            return await self._sentence_fallback(embeddings[0], user_id, agent_id, top_k, "l1")

        # Score the merged set
        scored_facts = score_and_rank(
            merged_facts,
            temporal_decay_rate=self.temporal_decay_rate,
            use_mentions=self.use_mentions,
        )

        if self.min_score > 0:
            scored_facts = [f for f in scored_facts if f["score"] >= self.min_score]

        if self.debug:
            for f in scored_facts[:top_k]:
                logger.debug(explain_score(f))

        top_facts = scored_facts[:top_k]
        fact_ids = [f["id"] for f in top_facts]

        # L1 graph traversal — runs ONCE on the full merged fact set
        related_insights = await self.db.get_insights_from_facts(
            fact_ids=fact_ids,
            user_id=user_id,
            active_only=True,
        )
        related_insights = _score_insights(related_insights, top_facts)

        source_sentence_ids = await self.db.get_source_sentences(fact_ids)
        source_sentences = []
        if source_sentence_ids:
            source_sentences = await self.db.get_sentences_by_ids(source_sentence_ids)

        return {
            "facts": _clean(top_facts),
            "insights": related_insights,
            "sentences": source_sentences,
        }

    # ── Sentence fallback ─────────────────────────────────────────────────────

    async def _sentence_fallback(
        self,
        query_embedding: list[float],
        user_id: str,
        agent_id: str | None,
        top_k: int,
        depth: str,
    ) -> dict[str, Any]:
        """
        Called when no facts exist yet (extraction still pending).
        Falls through to sentence-level vector search so early queries still
        return something useful. Sentences are always available immediately
        since they're stored synchronously in add().
        """
        if depth == "l0":
            return {"facts": []}

        sentences = await self.db.search_sentences(
            embedding=query_embedding,
            user_id=user_id,
            agent_id=agent_id,
            limit=top_k,
        )
        return {"facts": [], "insights": [], "sentences": sentences}


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


def _score_insights(
    insights: list[dict[str, Any]],
    scored_facts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Propagate max linked-fact score to insights and sort by relevance.

    Insights are discovered via graph traversal from facts, not ranked by
    vector similarity themselves. Propagating the max score of the facts
    that link to each insight gives a principled relevance ordering:
    if the insight was derived from a highly relevant fact, the insight
    is more relevant to this query.
    """
    if not insights or not scored_facts:
        return insights

    # Build fact_id → score lookup
    fact_scores: dict[str, float] = {f["id"]: f.get("score", 0.0) for f in scored_facts}

    result = []
    for ins in insights:
        # insight_facts join returns fact_ids on the insight dict (if present)
        linked_ids = ins.get("fact_ids") or []
        if linked_ids:
            max_fact_score = max(
                (fact_scores.get(fid, 0.0) for fid in linked_ids), default=0.0
            )
        else:
            # No explicit links — use global max of returned facts as fallback
            max_fact_score = max(fact_scores.values(), default=0.0)
        result.append({**ins, "score": round(max_fact_score, 6)})

    result.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return result

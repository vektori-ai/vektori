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
        Facts + cross-session insights linked to those facts (via insight_facts graph
        traversal) + the exact sentences each fact was extracted from (via fact_sources).
        Insights are always returned at every depth; sentences require l1 or l2.
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
        reference_date: datetime | None = None,
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
              "facts":        list[dict],  # always present
              "insights":     list[dict],  # always present — cross-session patterns linked to matched facts
              "sentences":    list[dict],  # l1, l2 only
              "memory_found": bool,        # False when no facts passed min_score (abstention signal)
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
            window = self._temporal_parser.parse(query, reference_date=reference_date)
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
        # ── Step 1: Parallel — facts AND sentences ────────────────────────────
        # Run both searches concurrently. Sentence search adds a second retrieval
        # surface for content that's never compressed into facts (e.g. assistant
        # turns, conversational details that don't survive fact extraction).
        seed_facts, direct_sentences = await asyncio.gather(
            self.db.search_facts(
                embedding=query_embedding,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                subject=subject,
                limit=top_k,
                active_only=not include_superseded,
                before_date=before_date,
                after_date=after_date,
            ),
            self.db.search_sentences(
                embedding=query_embedding,
                user_id=user_id,
                agent_id=agent_id,
                limit=top_k,
            ),
        )

        # Sentence fallback: no facts yet (extraction still in-flight)
        if not seed_facts:
            logger.debug(
                "search: no facts for user=%s, falling back to sentence search", user_id
            )
            if depth == "l0":
                return {"facts": [], "insights": [], "memory_found": False}
            return {"facts": [], "insights": [], "sentences": direct_sentences, "memory_found": False}

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

        top_k_facts = scored_facts[:top_k]
        seed_fact_ids = [f["id"] for f in top_k_facts]

        if depth == "l0":
            # L0: graph traversal only — keep it light
            insights = await self.db.get_insights_for_facts(seed_fact_ids)
            top = _clean(top_k_facts)
            return {"facts": top, "insights": insights, "memory_found": len(top) > 0}

        # ── Step 2: Insights (L1/L2) — graph traversal + vector search ────────
        # Run both concurrently: graph edges from matched facts, and direct
        # cosine search over insight embeddings. Merge by ID so the same
        # insight doesn't appear twice.
        graph_insights, vec_insights = await asyncio.gather(
            self.db.get_insights_for_facts(seed_fact_ids),
            self.db.search_insights(query_embedding, user_id, agent_id),
        )
        seen_insight_ids: set[str] = set()
        insights: list[dict[str, Any]] = []
        for ins in (*graph_insights, *vec_insights):
            iid = ins.get("id")
            if iid and iid not in seen_insight_ids:
                seen_insight_ids.add(iid)
                insights.append(ins)

        # ── Step 3: Trace facts → source sentences ────────────────────────────
        source_sentence_ids = await self.db.get_source_sentences(seed_fact_ids)

        top_facts = _clean(top_k_facts)
        memory_found = len(top_facts) > 0

        # ── Step 4: Session scoring — merge direct search + fact-linked ───────
        # Build lookup from direct sentence search (already have distances)
        direct_sent_by_id: dict[str, dict[str, Any]] = {s["id"]: s for s in direct_sentences}

        # Load fact-linked sentences not already in direct search results
        missing_ids = [sid for sid in source_sentence_ids if sid not in direct_sent_by_id]
        if missing_ids:
            loaded = await self.db.get_sentences_by_ids(missing_ids)
            for s in loaded:
                direct_sent_by_id[s["id"]] = s

        # All candidate IDs: direct search first (lower distance → higher priority),
        # then fact-linked sentences not in direct search
        all_candidate_ids: list[str] = list(dict.fromkeys([
            *[s["id"] for s in direct_sentences],
            *source_sentence_ids,
        ]))

        if not all_candidate_ids:
            return {
                "facts": top_facts,
                "insights": insights,
                "sentences": [],
                "memory_found": memory_found,
            }

        # Per-session baseline distance from fact scores (for sessions that appear
        # only via fact-linking, not in the direct sentence search)
        fact_dist_by_session: dict[str, float] = {}
        for f in scored_facts[:top_k]:
            fsid = f.get("session_id")
            if fsid:
                fdist = f.get("distance", 0.5)
                if fsid not in fact_dist_by_session or fdist < fact_dist_by_session[fsid]:
                    fact_dist_by_session[fsid] = fdist

        # Group candidates by session, track best (lowest) distance per session
        session_candidates: dict[str, list[dict[str, Any]]] = {}
        session_best_dist: dict[str, float] = {}

        for sid in all_candidate_ids:
            sent = direct_sent_by_id.get(sid)
            if not sent:
                continue
            ssid = sent.get("session_id", "")
            if not ssid:
                continue
            if ssid not in session_candidates:
                session_candidates[ssid] = []
                # Seed with the best fact distance for this session
                session_best_dist[ssid] = fact_dist_by_session.get(ssid, 1.0)
            session_candidates[ssid].append(sent)
            # Direct sentence match beats the fact-distance baseline
            sent_dist = sent.get("distance", 1.0)
            if sent_dist < session_best_dist[ssid]:
                session_best_dist[ssid] = sent_dist

        # Pick top sessions: more sessions for L2 (broader context)
        max_sessions = 5 if depth == "l2" else 3
        top_session_ids = sorted(
            session_best_dist, key=lambda s: session_best_dist[s]
        )[:max_sessions]

        # Return sentences from top sessions in conversation order
        result_sentences: list[dict[str, Any]] = []
        for ssid in top_session_ids:
            sents = sorted(
                session_candidates.get(ssid, []),
                key=lambda x: (x.get("turn_number", 0), x.get("sentence_index", 0)),
            )
            result_sentences.extend(sents)

        return {
            "facts": top_facts,
            "insights": insights,
            "sentences": _dedup(result_sentences),
            "memory_found": memory_found,
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

        top_facts = _clean(scored_facts[:top_k])
        top_fact_ids = [f["id"] for f in top_facts]

        graph_insights, vec_insights = await asyncio.gather(
            self.db.get_insights_for_facts(top_fact_ids),
            self.db.search_insights(query_embedding, user_id, agent_id),
        )
        seen_insight_ids: set[str] = set()
        insights: list[dict[str, Any]] = []
        for ins in (*graph_insights, *vec_insights):
            iid = ins.get("id")
            if iid and iid not in seen_insight_ids:
                seen_insight_ids.add(iid)
                insights.append(ins)

        return {
            "facts": top_facts,
            "insights": insights,
            "sentences": _dedup(raw.get("sentences", [])),
            "memory_found": len(top_facts) > 0,
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
            → get_source_sentences(top_fact_ids)               (L1 graph)
            → return {facts, sentences}

        Always returns L1 depth. L2 context expansion is intentionally excluded
        — expansion already widens fact recall; adding window expansion on top
        would be redundant and expensive.
        """
        if not queries:
            return {"facts": [], "sentences": [], "memory_found": False}

        # Single embed_batch call for all query variants
        embeddings = await self.embedder.embed_batch(queries)

        # Concurrent: L0 fact searches per variant + sentence search on primary query
        async def _search_one(embedding: list[float]) -> list[dict[str, Any]]:
            return await self.db.search_facts(
                embedding=embedding,
                user_id=user_id,
                agent_id=agent_id,
                subject=subject,
                limit=top_k,
                active_only=True,
            )

        fact_tasks = [_search_one(emb) for emb in embeddings]
        sent_task = self.db.search_sentences(
            embedding=embeddings[0],
            user_id=user_id,
            agent_id=agent_id,
            limit=top_k,
        )
        *all_results_list, direct_sentences = await asyncio.gather(*fact_tasks, sent_task)
        all_results: list[list[dict[str, Any]]] = list(all_results_list)

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
            return {"facts": [], "insights": [], "sentences": direct_sentences, "memory_found": False}

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

        graph_insights, vec_insights, source_sentence_ids = await asyncio.gather(
            self.db.get_insights_for_facts(fact_ids),
            self.db.search_insights(embeddings[0], user_id, agent_id),
            self.db.get_source_sentences(fact_ids),
        )
        seen_insight_ids: set[str] = set()
        insights: list[dict[str, Any]] = []
        for ins in (*graph_insights, *vec_insights):
            iid = ins.get("id")
            if iid and iid not in seen_insight_ids:
                seen_insight_ids.add(iid)
                insights.append(ins)

        # Session scoring: merge direct search + fact-linked sentences
        direct_sent_by_id: dict[str, dict[str, Any]] = {s["id"]: s for s in direct_sentences}
        missing_ids = [sid for sid in source_sentence_ids if sid not in direct_sent_by_id]
        if missing_ids:
            loaded = await self.db.get_sentences_by_ids(missing_ids)
            for s in loaded:
                direct_sent_by_id[s["id"]] = s

        all_candidate_ids = list(dict.fromkeys([
            *[s["id"] for s in direct_sentences],
            *source_sentence_ids,
        ]))

        session_candidates: dict[str, list[dict[str, Any]]] = {}
        session_best_dist: dict[str, float] = {}
        fact_dist_by_session: dict[str, float] = {}
        for f in top_facts:
            fsid = f.get("session_id")
            if fsid:
                fdist = f.get("distance", 0.5)
                if fsid not in fact_dist_by_session or fdist < fact_dist_by_session[fsid]:
                    fact_dist_by_session[fsid] = fdist

        for sid in all_candidate_ids:
            sent = direct_sent_by_id.get(sid)
            if not sent:
                continue
            ssid = sent.get("session_id", "")
            if not ssid:
                continue
            if ssid not in session_candidates:
                session_candidates[ssid] = []
                session_best_dist[ssid] = fact_dist_by_session.get(ssid, 1.0)
            session_candidates[ssid].append(sent)
            sent_dist = sent.get("distance", 1.0)
            if sent_dist < session_best_dist[ssid]:
                session_best_dist[ssid] = sent_dist

        top_session_ids = sorted(session_best_dist, key=lambda s: session_best_dist[s])[:3]
        result_sentences: list[dict[str, Any]] = []
        for ssid in top_session_ids:
            sents = sorted(
                session_candidates.get(ssid, []),
                key=lambda x: (x.get("turn_number", 0), x.get("sentence_index", 0)),
            )
            result_sentences.extend(sents)

        return {
            "facts": _clean(top_facts),
            "insights": insights,
            "sentences": _dedup(result_sentences),
            "memory_found": len(top_facts) > 0,
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
            return {"facts": [], "insights": [], "memory_found": False}

        sentences = await self.db.search_sentences(
            embedding=query_embedding,
            user_id=user_id,
            agent_id=agent_id,
            limit=top_k,
        )
        return {"facts": [], "insights": [], "sentences": sentences, "memory_found": False}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _empty(depth: str) -> dict[str, Any]:
    """Return the correct empty structure for a given depth."""
    result: dict[str, Any] = {"facts": [], "insights": []}
    if depth in ("l1", "l2"):
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



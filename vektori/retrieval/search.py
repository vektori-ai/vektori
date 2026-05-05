"""Main retrieval pipeline — L0 / L1 / L2 tiered search."""

from __future__ import annotations

import asyncio
import logging
import statistics
from datetime import datetime
from typing import Any

from vektori.models.base import EmbeddingProvider
from vektori.retrieval.ppr import rank_episodes_by_ppr, run_ppr
from vektori.retrieval.scoring import explain_score, score_and_rank
from vektori.retrieval.temporal import TemporalQueryParser
from vektori.storage.base import StorageBackend

logger = logging.getLogger(__name__)

VALID_DEPTHS = {"l0", "l1", "l2"}


class SearchPipeline:
    """Three-layer retrieval pipeline with 4-strategy fusion.

    L0 — Facts only (~50-200 tokens):
        Vector search over facts. Entry point for all retrieval.

    L1 — Facts + Episodes + Source Sentences (~300-800 tokens):
        Facts + cross-session episodes linked to those facts (via episode_facts graph
        traversal) + the exact sentences each fact was extracted from (via fact_sources).

    L2 — Full story (~1000-3000 tokens):
        Everything in L1, plus full session context window (±N sentences around
        each source sentence via NEXT edges).

    Retrieval strategies (run in parallel, fused via RRF):
        1. Semantic vector search (cosine similarity)
        2. Keyword / BM25 search
        3. Temporal search (when date range parsed from query)
        4. Graph traversal (PPR-surfaced facts not in initial L0 results)

    Post-fusion:
        - Composite scoring (similarity × recency × mentions × source_weight)
        - Cross-encoder reranking (if sentence-transformers installed)
        - Session-diverse top-k selection
        - Optional token budget truncation
    """

    def __init__(
        self,
        db: StorageBackend,
        embedder: EmbeddingProvider,
        temporal_decay_rate: float = 0.001,
        use_mentions: bool = True,
        min_score: float = 0.3,
        use_ppr: bool = True,
        ppr_alpha: float = 0.5,
        reranker: Any = None,
        reranker_top_n: int = 20,
        debug: bool = False,
    ) -> None:
        self.db = db
        self.embedder = embedder
        self.temporal_decay_rate = temporal_decay_rate
        self.use_mentions = use_mentions
        self.min_score = min_score
        self.use_ppr = use_ppr
        self.ppr_alpha = ppr_alpha
        # CrossEncoderReranker instance — None disables reranking
        self._reranker = reranker
        # How many candidates to pass to the reranker (rerank top-N before top-k selection)
        self._reranker_top_n = reranker_top_n
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
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Retrieve relevant memories for a query.

        Args:
            query: Natural language query string.
            user_id: Whose memories to search.
            agent_id: Optional agent scoping.
            depth: "l0" | "l1" | "l2". Invalid values raise ValueError.
            top_k: Max facts to return.
            context_window: ±N sentences around each source sentence (L2 only).
            include_superseded: Also return overridden/old facts.
            before_date/after_date: Explicit temporal filters on fact event_time.
            parse_temporal: Auto-parse temporal expressions from the query.
            max_tokens: Token budget for result content. Truncates to fit when set.

        Returns:
            {
              "facts":        list[dict],
              "episodes":     list[dict],
              "sentences":    list[dict],  # l1, l2 only
              "memory_found": bool,
            }

        Raises:
            ValueError: if depth is not one of "l0", "l1", "l2".
        """
        if depth not in VALID_DEPTHS:
            raise ValueError(f"Invalid depth '{depth}'. Must be one of: {sorted(VALID_DEPTHS)}")

        if parse_temporal and before_date is None and after_date is None:
            window = self._temporal_parser.parse(query, reference_date=reference_date)
            if window:
                before_date = window.before_date
                after_date = window.after_date
                logger.debug(
                    "Temporal window parsed: after=%s before=%s", after_date, before_date
                )

        query_embedding = await self.embedder.embed(query)

        # L2 Postgres fast path — disable when temporal dates active (stepped path handles that)
        if (
            depth == "l2"
            and not include_superseded
            and not before_date
            and not after_date
            and getattr(self.db, "supports_single_query", False)
        ):
            result = await self._search_l2_fast(
                query,
                query_embedding,
                user_id,
                agent_id,
                subject,
                session_id,
                top_k,
                context_window,
            )
        else:
            result = await self._search_stepped(
                query,
                query_embedding,
                user_id,
                agent_id,
                subject,
                session_id,
                depth,
                top_k,
                context_window,
                include_superseded,
                before_date=before_date,
                after_date=after_date,
            )

        if max_tokens:
            result = _truncate_to_token_budget(result, max_tokens)

        return result

    # ── Step-by-step path (all backends, L0/L1, and L2 temporal fallback) ────

    async def _search_stepped(
        self,
        query_text: str,
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
        # ── Step 1: Parallel — vector + keyword + temporal + sentence fallback ─
        temporal_coro = (
            self.db.search_facts_temporal(
                user_id=user_id,
                agent_id=agent_id,
                before_date=before_date,
                after_date=after_date,
                subject=subject,
                limit=top_k * 2,
                active_only=not include_superseded,
            )
            if (before_date or after_date)
            else asyncio.sleep(0, result=[])
        )

        seed_facts_vec, seed_facts_kw, temporal_facts, direct_sentences = await asyncio.gather(
            self.db.search_facts(
                embedding=query_embedding,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                subject=subject,
                limit=top_k * 2,
                active_only=not include_superseded,
                before_date=before_date,
                after_date=after_date,
            ),
            self.db.search_facts_keyword(
                query_text=query_text,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
                subject=subject,
                limit=top_k * 2,
                active_only=not include_superseded,
                before_date=before_date,
                after_date=after_date,
            ) if hasattr(self.db, "search_facts_keyword") else asyncio.sleep(0, result=[]),
            temporal_coro,
            self.db.search_sentences(
                embedding=query_embedding,
                user_id=user_id,
                agent_id=agent_id,
                limit=top_k,
            ),
        )

        # 3-way RRF: vector + keyword + temporal
        seed_facts = _reciprocal_rank_fusion(seed_facts_vec, seed_facts_kw, temporal_facts)

        if not seed_facts:
            logger.debug("search: no facts for user=%s, falling back to sentence search", user_id)
            if depth == "l0":
                return {"facts": [], "episodes": [], "syntheses": [], "memory_found": False}
            return {
                "facts": [],
                "episodes": [],
                "syntheses": [],
                "sentences": direct_sentences,
                "memory_found": False,
            }

        scored_facts = score_and_rank(
            seed_facts,
            temporal_decay_rate=self.temporal_decay_rate,
            use_mentions=self.use_mentions,
        )

        scored_facts = _apply_threshold(scored_facts, self.min_score)

        if self.debug:
            for f in scored_facts[:top_k]:
                logger.debug(explain_score(f))

        # ── Rerank top-N candidates with cross-encoder ────────────────────────
        if self._reranker is not None and scored_facts:
            candidates = scored_facts[:self._reranker_top_n]
            candidates = self._reranker.rerank(query_text, candidates)
            # Re-merge: reranked candidates + remainder (already below threshold)
            reranked_ids = {f["id"] for f in candidates}
            scored_facts = candidates + [f for f in scored_facts[self._reranker_top_n:] if f["id"] not in reranked_ids]

        top_k_facts = _diverse_top_k(scored_facts, top_k)
        seed_fact_ids = [f["id"] for f in top_k_facts]

        if depth == "l0":
            episodes = await self.db.get_episodes_for_facts(seed_fact_ids)
            syntheses = await self.db.get_syntheses_for_facts(seed_fact_ids)
            top = _clean(top_k_facts)
            return {
                "facts": top,
                "episodes": episodes,
                "syntheses": syntheses,
                "memory_found": len(top) > 0,
            }

        # ── Step 2: Episodes — PPR or plain graph hop ─────────────────────────
        graph_syntheses, vec_syntheses, vec_episodes = await asyncio.gather(
            self.db.get_syntheses_for_facts(seed_fact_ids),
            self.db.search_syntheses(query_embedding, user_id, agent_id),
            self.db.search_episodes(query_embedding, user_id, agent_id),
        )

        if self.use_ppr:
            fact_edges, episode_fact_map, all_user_facts = await asyncio.gather(
                self.db.get_fact_edges_for_user(user_id, agent_id),
                self.db.get_episode_fact_map(user_id, agent_id),
                self.db.get_active_facts(user_id, agent_id, limit=500),
            )

        if self.use_ppr and (fact_edges or episode_fact_map):
            seed_scores = {
                f["id"]: f.get("_score_components", {}).get("similarity", f.get("score", 0.0))
                for f in top_k_facts
            }
            ppr_scores = run_ppr(
                seed_scores=seed_scores,
                all_facts=all_user_facts,
                fact_edges=fact_edges,
                episode_fact_map=episode_fact_map,
                alpha=self.ppr_alpha,
            )
            ppr_ranked = rank_episodes_by_ppr(episode_fact_map, ppr_scores)

            top_fact_ids_from_ppr = [
                fid
                for ep_id, _ in ppr_ranked[:10]
                if ppr_scores.get(ep_id, 0) > 0
                for fid in episode_fact_map.get(ep_id, [])
            ]
            graph_episodes = await self.db.get_episodes_for_facts(top_fact_ids_from_ppr) if top_fact_ids_from_ppr else []

            # ── Graph stream: surface PPR-traversed facts not in initial results ──
            episode_ids = set(episode_fact_map.keys())
            seed_ids = set(seed_fact_ids)
            max_ppr = max(ppr_scores.values()) if ppr_scores else 1.0
            extra_ppr_ids = [
                fid for fid, score in sorted(ppr_scores.items(), key=lambda x: x[1], reverse=True)
                if fid not in episode_ids and fid not in seed_ids and score > 0.001
            ][:top_k]

            if extra_ppr_ids:
                extra_facts = await self.db.get_facts_by_ids(extra_ppr_ids, user_id)
                for f in extra_facts:
                    raw_ppr = ppr_scores.get(f["id"], 0.0)
                    # Normalize PPR score → competitive distance (0.2–0.5 range)
                    norm = raw_ppr / max_ppr if max_ppr > 0 else 0.0
                    f["distance"] = max(0.05, 0.5 - norm * 0.3)

                if extra_facts:
                    # Re-RRF: merge graph-traversed facts into the scored pool
                    combined = _reciprocal_rank_fusion(top_k_facts, extra_facts)
                    combined_scored = score_and_rank(
                        combined,
                        temporal_decay_rate=self.temporal_decay_rate,
                        use_mentions=self.use_mentions,
                    )
                    combined_scored = _apply_threshold(combined_scored, self.min_score)
                    top_k_facts = _diverse_top_k(combined_scored, top_k)
                    seed_fact_ids = [f["id"] for f in top_k_facts]

            if self.debug:
                logger.debug(
                    "PPR: %d edges, %d episodes, top: %s",
                    len(fact_edges),
                    len(episode_fact_map),
                    [(ep_id[:8], round(s, 5)) for ep_id, s in ppr_ranked[:3]],
                )
        else:
            logger.debug("PPR: skipped (use_ppr=%s) — using plain graph hop", self.use_ppr)
            graph_episodes = await self.db.get_episodes_for_facts(seed_fact_ids)

        episodes = _merge_unique(graph_episodes, vec_episodes)
        syntheses = _merge_unique(graph_syntheses, vec_syntheses)

        # ── Step 3: Trace facts → source sentences ────────────────────────────
        source_sentence_ids = await self.db.get_source_sentences(seed_fact_ids)

        top_facts = _clean(top_k_facts)

        # ── Step 4: Session scoring ────────────────────────────────────────────
        result_sentences = await _filter_top_sessions(
            self.db,
            direct_sentences,
            source_sentence_ids,
            scored_facts,
            top_k,
            depth,
        )

        return {
            "facts": top_facts,
            "episodes": episodes,
            "syntheses": syntheses,
            "sentences": _dedup(result_sentences),
            "memory_found": len(top_facts) > 0,
        }

    # ── Single-query fast path (Postgres L2, non-temporal queries only) ───────

    async def _search_l2_fast(
        self,
        query_text: str,
        query_embedding: list[float],
        user_id: str,
        agent_id: str | None,
        subject: str | None,
        session_id: str | None,
        top_k: int,
        context_window: int,
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
        )

        scored_facts = score_and_rank(
            raw.get("facts", []),
            temporal_decay_rate=self.temporal_decay_rate,
            use_mentions=self.use_mentions,
        )
        scored_facts = _apply_threshold(scored_facts, self.min_score)

        if self._reranker is not None and scored_facts:
            candidates = scored_facts[:self._reranker_top_n]
            candidates = self._reranker.rerank(query_text, candidates)
            reranked_ids = {f["id"] for f in candidates}
            scored_facts = candidates + [f for f in scored_facts[self._reranker_top_n:] if f["id"] not in reranked_ids]

        if self.debug:
            for f in scored_facts[:top_k]:
                logger.debug(explain_score(f))

        top_facts = _clean(_diverse_top_k(scored_facts, top_k))
        top_fact_ids = [f["id"] for f in top_facts]

        graph_episodes, vec_episodes, graph_syntheses, vec_syntheses = await asyncio.gather(
            self.db.get_episodes_for_facts(top_fact_ids),
            self.db.search_episodes(query_embedding, user_id, agent_id),
            self.db.get_syntheses_for_facts(top_fact_ids),
            self.db.search_syntheses(query_embedding, user_id, agent_id),
        )

        episodes = _merge_unique(graph_episodes, vec_episodes)
        syntheses = _merge_unique(graph_syntheses, vec_syntheses)

        return {
            "facts": top_facts,
            "episodes": episodes,
            "syntheses": syntheses,
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
        """Expanded retrieval: concurrent L0 fact searches across multiple query variants."""
        if not queries:
            return {"facts": [], "sentences": [], "memory_found": False}

        embeddings = await self.embedder.embed_batch(queries)

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

        best_by_id: dict[str, dict[str, Any]] = {}
        for result_set in all_results:
            for fact in result_set:
                fid = fact["id"]
                if fid not in best_by_id or fact.get("distance", 1.0) < best_by_id[fid].get("distance", 1.0):
                    best_by_id[fid] = fact

        merged_facts = list(best_by_id.values())

        if not merged_facts:
            logger.debug("search_expanded: no facts found, falling back to sentence search")
            return {
                "facts": [],
                "episodes": [],
                "syntheses": [],
                "sentences": direct_sentences,
                "memory_found": False,
            }

        scored_facts = score_and_rank(
            merged_facts,
            temporal_decay_rate=self.temporal_decay_rate,
            use_mentions=self.use_mentions,
        )
        scored_facts = _apply_threshold(scored_facts, self.min_score)

        if self._reranker is not None and scored_facts:
            candidates = scored_facts[:self._reranker_top_n]
            candidates = self._reranker.rerank(queries[0], candidates)
            reranked_ids = {f["id"] for f in candidates}
            scored_facts = candidates + [f for f in scored_facts[self._reranker_top_n:] if f["id"] not in reranked_ids]

        if self.debug:
            for f in scored_facts[:top_k]:
                logger.debug(explain_score(f))

        top_facts = _diverse_top_k(scored_facts, top_k)
        fact_ids = [f["id"] for f in top_facts]

        (
            graph_episodes,
            vec_episodes,
            graph_syntheses,
            vec_syntheses,
            source_sentence_ids,
        ) = await asyncio.gather(
            self.db.get_episodes_for_facts(fact_ids),
            self.db.search_episodes(embeddings[0], user_id, agent_id),
            self.db.get_syntheses_for_facts(fact_ids),
            self.db.search_syntheses(embeddings[0], user_id, agent_id),
            self.db.get_source_sentences(fact_ids),
        )

        episodes = _merge_unique(graph_episodes, vec_episodes)
        syntheses = _merge_unique(graph_syntheses, vec_syntheses)

        result_sentences = await _filter_top_sessions(
            self.db,
            direct_sentences,
            source_sentence_ids,
            top_facts,
            top_k,
            depth="l1",
        )

        return {
            "facts": _clean(top_facts),
            "episodes": episodes,
            "syntheses": syntheses,
            "sentences": _dedup(result_sentences),
            "memory_found": len(top_facts) > 0,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip internal scoring fields before returning to the caller."""
    return [{k: v for k, v in f.items() if k not in ("_score_components", "_rerank_score")} for f in facts]


def _apply_threshold(
    scored_facts: list[dict[str, Any]],
    min_score: float,
) -> list[dict[str, Any]]:
    """Dynamic Z-score threshold + hard min_score floor."""
    if not scored_facts:
        return scored_facts
    raw_scores = [f.get("_score_components", {}).get("similarity", f["score"]) for f in scored_facts]
    mean_score = sum(raw_scores) / len(raw_scores)
    std_score = statistics.stdev(raw_scores) if len(raw_scores) > 1 else 0.0
    effective_min = max(min_score, mean_score - std_score)
    return [
        f for f in scored_facts
        if f.get("_score_components", {}).get("similarity", f["score"]) >= effective_min
    ]


def _diverse_top_k(
    scored_facts: list[dict[str, Any]],
    top_k: int,
    per_session: int = 3,
) -> list[dict[str, Any]]:
    """Session-diverse top-k: up to per_session facts per session, then global fill."""
    selected: list[dict[str, Any]] = []
    session_counts: dict[str, int] = {}
    remainder: list[dict[str, Any]] = []

    for f in scored_facts:
        sid = f.get("session_id") or ""
        if session_counts.get(sid, 0) < per_session:
            selected.append(f)
            session_counts[sid] = session_counts.get(sid, 0) + 1
        else:
            remainder.append(f)
        if len(selected) >= top_k:
            break

    for f in remainder:
        if len(selected) >= top_k:
            break
        selected.append(f)

    return selected


def _dedup(sentences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result = []
    for s in sentences:
        sid = s.get("id")
        if sid and sid not in seen:
            seen.add(sid)
            result.append(s)
    return result


def _merge_unique(*iterables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result = []
    for it in iterables:
        for item in it:
            iid = item.get("id")
            if iid and iid not in seen:
                seen.add(iid)
                result.append(item)
    return result


def _reciprocal_rank_fusion(
    *result_lists: list[dict[str, Any]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """N-way Reciprocal Rank Fusion.

    Accepts 2–N result lists. Facts appearing in more strategies rank higher.
    Distance is normalized so the top consensus result gets distance ≈ 0.05
    and single-stream rank-0 results get ≈ 0.50.
    """
    scores: dict[str, float] = {}
    items_map: dict[str, dict[str, Any]] = {}

    for result_list in result_lists:
        for rank, item in enumerate(result_list):
            item_id = item["id"]
            scores[item_id] = scores.get(item_id, 0.0) + (1.0 / (k + rank))
            if item_id not in items_map:
                items_map[item_id] = item

    if not scores:
        return []

    sorted_ids = sorted(scores, key=lambda i: scores[i], reverse=True)
    max_score = scores[sorted_ids[0]]

    fused: list[dict[str, Any]] = []
    for item_id in sorted_ids:
        item = dict(items_map[item_id])
        # Normalize: top result → distance 0.05, weakest → distance 0.95
        norm = scores[item_id] / max_score if max_score > 0 else 0.0
        item["distance"] = round(max(0.0, 1.0 - norm * 0.95), 4)
        fused.append(item)

    return fused


def _truncate_to_token_budget(
    result: dict[str, Any],
    max_tokens: int,
    chars_per_token: float = 4.0,
) -> dict[str, Any]:
    """Truncate result to fit within a token budget (approximate, char-based)."""
    budget_chars = int(max_tokens * chars_per_token)
    used = 0

    facts: list[dict[str, Any]] = []
    for f in result.get("facts", []):
        cost = len(f.get("text", "")) + 50
        if used + cost > budget_chars:
            break
        facts.append(f)
        used += cost

    episodes: list[dict[str, Any]] = []
    for e in result.get("episodes", []):
        cost = len(e.get("text", "")) + 30
        if used + cost > budget_chars:
            break
        episodes.append(e)
        used += cost

    sentences: list[dict[str, Any]] = []
    for s in result.get("sentences", []):
        cost = len(s.get("text", "")) + 20
        if used + cost > budget_chars:
            break
        sentences.append(s)
        used += cost

    return {
        **result,
        "facts": facts,
        "episodes": episodes,
        "sentences": sentences,
    }


async def _filter_top_sessions(
    db: StorageBackend,
    direct_sentences: list[dict[str, Any]],
    source_sentence_ids: list[str],
    scored_facts: list[dict[str, Any]],
    top_k: int,
    depth: str,
) -> list[dict[str, Any]]:
    """Merge direct search and fact-linked sentences, pick top sessions."""
    direct_sent_by_id: dict[str, dict[str, Any]] = {s["id"]: s for s in direct_sentences}

    missing_ids = [sid for sid in source_sentence_ids if sid not in direct_sent_by_id]
    if missing_ids:
        loaded = await db.get_sentences_by_ids(missing_ids)
        for s in loaded:
            direct_sent_by_id[s["id"]] = s

    all_candidate_ids: list[str] = list(
        dict.fromkeys([
            *[s["id"] for s in direct_sentences],
            *source_sentence_ids,
        ])
    )

    if not all_candidate_ids:
        return []

    fact_dist_by_session: dict[str, float] = {}
    for f in scored_facts[:top_k]:
        fsid = f.get("session_id")
        if fsid:
            fdist = f.get("distance", 0.5)
            if fsid not in fact_dist_by_session or fdist < fact_dist_by_session[fsid]:
                fact_dist_by_session[fsid] = fdist

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
            session_best_dist[ssid] = fact_dist_by_session.get(ssid, 1.0)
        session_candidates[ssid].append(sent)
        sent_dist = sent.get("distance", 1.0)
        if sent_dist < session_best_dist[ssid]:
            session_best_dist[ssid] = sent_dist

    max_sessions = 5 if depth == "l2" else 3
    top_session_ids = sorted(session_best_dist, key=lambda s: session_best_dist[s])[:max_sessions]

    result_sentences: list[dict[str, Any]] = []
    for ssid in top_session_ids:
        sents = sorted(
            session_candidates.get(ssid, []),
            key=lambda x: (x.get("turn_number", 0), x.get("sentence_index", 0)),
        )
        result_sentences.extend(sents)

    return result_sentences

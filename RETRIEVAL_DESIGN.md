# Vektori — Retrieval Design

## Architecture: three-layer graph

```
L0  Facts      — LLM-extracted statements. Primary vector search surface.
L1  Insights   — LLM-inferred cross-session patterns. Discovered via graph, never searched directly.
L2  Sentences  — Raw conversation. Retrieved by tracing facts back to their source.
```

Join tables are the graph edges:
- `fact_sources`    — fact → sentence it came from (L0↔L2 bridge)
- `insight_facts`   — insight → facts it was derived from (L1↔L0 bridge, key for traversal)
- `insight_sources` — insight → source sentences (L1↔L2)

---

## Why this design

**Why vector search only on facts, not sentences or insights?**
Facts are short, crisp, standalone statements — they embed cleanly against direct natural language queries. Sentences are noisy and long. Insights are abstract patterns that don't map well to specific queries. The fact layer is the right semantic surface for cosine search.

**Why insights via graph traversal, not vector search?**
The `insight_facts` edges were written by the LLM during extraction with full conversation context. They represent prior reasoning, not a runtime approximation. Traversing those edges at retrieval time is principled — if fact F is relevant to the query (proven by cosine similarity), and insight I is linked to F (LLM-established), then I is contextually related to the query domain. This is the same argument HippoRAG makes for knowledge graph traversal.

**Why not KNN edges between facts?**
KNN edges connect semantically similar nodes in graph approaches like HippoRAG (synonym edges at τ=0.8). We don't need them because: (a) vector search already surfaces near-duplicate facts directly, and (b) insights serve as multi-hop hubs — facts that are related will likely share an insight, so the path `fact_A → insight_X → fact_B` exists without direct KNN edges. Inserting KNN edges would be redundant and expensive to maintain.

**Why not a knowledge graph (Neo4j etc.)?**
We have one — it lives in Postgres join tables. A graph DB adds value for multi-hop traversal at scale, graph-native query languages, and graph algorithms (PageRank, centrality). None of those are needed at current scale. Postgres handles 1–2 hop traversal via JOIN and recursive CTEs cleanly. One system > two systems.

**Why not Personalized PageRank (HippoRAG2)?**
PPR would surface nodes at 2+ hops from seed facts. Concretely: `query → fact_A → insight_X → fact_B (not in top-k)`. This is real value but only matters when the graph is dense. For a personal memory engine with hundreds of facts per user, 1-hop traversal already surfaces nearly everything reachable. PPR becomes relevant when the graph grows to thousands of facts and multi-hop paths become the primary way to find related memories. Revisit when users hit ~5k facts.

**Why ±window expansion within same turn, not cross-turn?**
Cross-turn expansion risks bleeding context from unrelated parts of long conversations. A user discussing work in turn 3 and dinner in turn 15 shouldn't have those mixed. Same-turn expansion keeps context coherent. One-line change in `expand_session_context` if cross-turn is ever needed.

---

## Retrieval flow

```
search(query, user_id, depth)
  │
  ├─ embed(query) → query_vector
  │
  ├─ search_facts(query_vector, user_id)          # cosine on IVFFlat index
  │    └─ score_and_rank(facts)                   # similarity × confidence × recency × mentions
  │
  ├─ [L0] return facts
  │
  ├─ get_insights_from_facts(fact_ids, user_id)   # JOIN insight_facts, no vector search
  │
  ├─ [L1] return facts + insights + source sentences (exact, no expansion)
  │
  ├─ get_source_sentences(fact_ids)
  │    └─ expand_session_context(ids, ±window)    # same turn, sentence_index proximity
  │
  └─ [L2] return facts + insights + expanded sentences
```

Postgres fast path: L2 executes as a single CTE round trip (`search_l2_single_query`). Activated when `supports_single_query=True` and `include_superseded=False`.

---

## Scoring

```
score = similarity × confidence × recency × mentions_boost

similarity     = 1 - cosine_distance          # primary signal
confidence     = LLM extraction confidence    # multiplicative, halves score at 0.5
recency        = exp(-0.001 × age_days)       # ~36% decay per year
mentions_boost = 1.0 + 0.5 × log(mentions)/log(max_mentions)  # [1.0, 1.5]
```

Insights are currently returned unordered. Should propagate max linked fact score as insight relevance score.

---

## Known gaps, priority order

**1. `fact_sources` linking is sparse (highest impact)**
`find_sentences_by_similarity` uses pg_trgm to match LLM `source_quotes` against stored sentences. Fails when the LLM paraphrases. Result: `fact_sources` rows don't get written → L1/L2 returns `sentences: []` silently even when facts exist. Fix: embed source_quotes, vector search against sentences instead of trgm.

**2. `deactivate_fact` missing `superseded_by` pointer (correctness bug)**
In `extractor.py`: old fact is deactivated but the new fact's ID is not passed as `superseded_by`. The supersession chain is broken in one direction — can't walk old→new, only new→old.

**3. No minimum similarity threshold**
Top-k always returns k facts regardless of actual relevance. Queries with no relevant memories return noise. Add a score floor.

**4. Insight scoring**
Insights come back ordered by `created_at`. Should be ordered by relevance — propagate max score of linked facts.

**5. Embedding batch in extractor**
N separate `embed()` calls per session (one per fact/insight). Should be one `embed_batch()`.

**6. Fact deduplication**
No pre-insert check for near-duplicate facts. Contradiction check only fires if LLM explicitly sets `contradicts`. Over many sessions: bloated fact table, degraded retrieval.

**7. Background worker back-pressure**
No queue depth limit. Burst traffic queues unbounded LLM calls.

---

## What we are not doing and why

| Not doing | Why |
|---|---|
| Vector search on insights | Insights are patterns, not query targets. Graph traversal from facts is more principled than cold cosine matching. |
| KNN edges between facts | Vector search already surfaces similar facts. Insights handle multi-hop. Redundant. |
| Graph DB | Postgres join tables are a graph. Not at scale where graph-native algorithms pay off. |
| PPR | 1-hop traversal sufficient at current scale. Revisit at ~5k facts/user. |
| Cross-turn sentence expansion | Prevents context bleeding across unrelated conversation segments. |
| Sentence-level vector search as entry point | Sentences are noisy. Facts are the right semantic surface for queries. |

# LoCoMo Run 1 — Failure Analysis & Learnings

**Date:** 2026-05-07  
**Dataset:** LoCoMo-10 (1542 questions, 10 samples)  
**Stack:** BGE-M3 local embeddings, Gemini 2.5 Flash-Lite extraction + QA, Gemini 3 Flash judge

---

## Score Summary (Run 1)

| Metric | Value |
|---|---|
| Correct | ~28% |
| Partially correct | ~12% |
| Wrong | ~35% |
| Abstained | ~25% |
| Retrieval failure | ~65% of wrong+abstained |
| QA failure | ~35% of wrong+abstained |

**Key ratio:** 6.5:1 retrieval failures vs QA failures — retrieval and extraction are the bottleneck, not synthesis.

---

## Root Cause Analysis

### 1. Extraction: too few facts per session (highest impact)
`max_facts=15` was catastrophically low for LoCoMo sessions, which contain 20-40 conversational turns covering multiple events, plans, people, and activities. The LLM was forced to prioritize and would drop less-salient but later-asked-about facts.

**Symptoms:** Questions about specific one-time events (a restaurant visited once, a gift given, a person mentioned in passing) had zero retrieval hits. The context returned was empty or unrelated.

**Fix:** Raise `max_facts` to 28. Still bounded but allows full coverage of a rich session.

### 2. Extraction: planned ≠ completed conflation
The prompt had no instruction to distinguish between "planning to do X" vs "having done X". The extractor frequently wrote:
- "Caroline planned to visit New York in November" → correct
- "Caroline visited New York in November" → correct

But also frequently merged them or only extracted one:
- "Caroline mentioned New York plans" → loses the completion status

**Symptoms:** QA questions like "Where did X go for the holidays?" would either return the plan (wrong if they didn't go) or nothing (if completion wasn't captured).

**Fix:** Added explicit PLANNED vs COMPLETED extraction rules with bad/good examples in the prompt.

### 3. Extraction: duration and quantity loss
Numeric details (how many days, how many items, how many people) were systematically dropped when the extractor merged related facts.

**Symptoms:** Questions like "How many days did X spend at Y?" returned correct location but no duration. Questions like "How many books did X read?" found the reading activity but not the count.

**Fix:** Added explicit DURATION and QUANTITY extraction rules.

### 4. Extraction: list items merged into one fact
When a user mentioned 4 restaurants or 3 activities, the extractor often merged all into one fact: "User mentioned visiting several restaurants including A, B, C, D." This single fact is not retrievable when asking about restaurant B specifically.

**Symptoms:** List-type questions (type 2) had very low correct rate. The context would contain merged lists but the model couldn't extract the right element because the embedding similarity was diluted across all items.

**Fix:** Added explicit LIST splitting rule: one fact per list item.

### 5. Retrieval: early-session fact dropout
Facts from the first 1-3 sessions of a 10-session sample had lower `event_time` (older). With exponential recency decay, these facts scored lower even when highly relevant. For LoCoMo samples spanning 6-12 months, the oldest events had 40-50% lower recency scores.

**Symptoms:** Questions about the first meeting, initial preferences set early, or events from session 1 were more likely to fail vs. questions about recent sessions.

**Fix:** Added recency floor of 0.35 in `scoring.py`. Recency can now decay from 1.0 to 0.35 maximum (not to 0.0), so an old highly-relevant fact can still rank above a new irrelevant one.

### 6. Retrieval: top_k too small
`default_top_k=15` with `reranker_top_n=20` meant the reranker only saw 20 candidates. For LoCoMo samples with 50-200 extracted facts, many relevant facts never even reached the reranker.

**Fix:** Raise `default_top_k` to 20, `reranker_top_n` to 30.

### 7. QA: planned/completed conflation in synthesis
Even when the extractor correctly tagged a fact as a plan, the QA model would sometimes answer "Yes, they went to X" when the context only showed "plans to go to X."

**Fix:** Added explicit instruction #10 to QA prompt: "planned to X" ≠ "did X". Only confirmed completions count as events.

### 8. QA: multi-person attribution errors
When a LoCoMo sample has two named participants (e.g., Caroline and James), facts about one person would sometimes be used to answer questions about the other.

**Root cause:** The QA prompt instruction about character attribution wasn't strong enough. The model would reason "the context mentions this activity, and the question asks about a person in this conversation, so it probably applies."

**Fix:** Strengthened instruction 9 with explicit anti-examples and added instruction 10 for plan/completion.

---

## What Worked

- **RRF fusion** across semantic + keyword + temporal + PPR strategies worked well for frequently-mentioned facts and temporal queries
- **Cross-encoder reranker** improved precision over pure vector similarity — precision@5 was notably higher with reranker on
- **PPR graph traversal** helped surface co-mentioned entities (if fact A and fact B are about the same person/place, PPR boosts B when A is retrieved)
- **Session isolation** per `user_id` prevented cross-sample contamination
- **Session cache** worked correctly — second run through same extraction model was fast

---

## What to Try Next

1. **Temporal date-range hard filter**: For queries with explicit dates ("What happened in October?"), add a DB-level date filter before vector search. This would cut retrieval candidates dramatically and boost precision.
2. **Multi-hop QA**: For complex type-4 questions involving chaining (Person A's activity at Event B where B involves Person C), a structured graph traversal rather than flat vector search would help.
3. **Larger extraction model for complex sessions**: Gemini 2.5 Flash-Lite is fast but occasionally produces malformed or incomplete extractions on very long sessions. Gemini 2.5 Pro for the first extraction pass (not cached) could improve base quality.
4. **Query expansion with named entities**: Before searching, extract named entities from the question and use them as additional search constraints (subject filter).
5. **Per-subject fact indexing**: Instead of one flat fact table, maintain per-named-entity fact stores so multi-person conversations don't mix facts across people.

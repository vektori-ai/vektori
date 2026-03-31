"""Fact scoring and ranking for retrieval results."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any


def score_and_rank(
    facts: list[dict[str, Any]],
    temporal_decay_rate: float = 0.001,
    use_mentions: bool = True,
) -> list[dict[str, Any]]:
    """Score facts and sort by relevance descending.

    Final score = similarity × confidence × recency × mentions_boost

    Components:
        similarity     — cosine similarity from pgvector (1 - distance).
                         This is the primary signal. Short, crisp facts score
                         well here because they embed cleanly against direct queries.

        confidence     — LLM extraction confidence [0, 1]. Facts the model was
                         uncertain about score lower. Multiplicative so a 0.5
                         confidence halves the score regardless of other signals.

        recency        — Exponential decay: exp(-rate * age_days). Default rate of
                         0.001 gives ~36% decay over a year — slow enough that old
                         facts stay relevant but new ones edge ahead when similarity
                         is equal. Keeps recently confirmed preferences ranked higher.

        mentions_boost — log(1 + mentions) normalised to [1.0, 1.5] range.
                         A fact mentioned once scores 1.0; a fact re-encountered
                         across many sessions scores up to ~1.5× higher.
                         This is the IDF signal from the spec: facts that keep
                         appearing are more "established" and deserve a small lift.
                         Capped so it never dominates over similarity.

    Args:
        facts: List of fact dicts from the storage backend. Expected fields:
               "distance" (float, pgvector cosine distance — lower is closer),
               "confidence" (float, 0-1),
               "created_at" (datetime or str),
               "mentions" (int, optional — defaults to 1).
        temporal_decay_rate: Decay per day. 0.001 ≈ 36%/year. Set higher (e.g.
                             0.01) for use-cases where freshness matters more.
        use_mentions: Whether to apply the mentions boost. Disable if your
                      storage backend doesn't track mentions.

    Returns:
        Same list with a "score" field added, sorted descending by score.
    """
    if not facts:
        return []

    now = datetime.utcnow()

    # Pre-compute max mentions for normalisation — only over this result set.
    # We don't want a globally popular fact to dominate locally irrelevant results.
    if use_mentions:
        max_mentions = max(f.get("mentions", 1) or 1 for f in facts)
    else:
        max_mentions = 1

    scored = []
    for fact in facts:
        # ── Similarity ──────────────────────────────────────────────────────
        # pgvector cosine distance: 0 = identical vectors, 2 = opposite.
        # Clamp to [0, 1] — in practice distances are in [0, 1] for normalised
        # embeddings, but be defensive.
        distance = float(fact.get("distance") or 0.0)
        similarity = max(0.0, min(1.0, 1.0 - distance))

        # ── Confidence ──────────────────────────────────────────────────────
        confidence = float(fact.get("confidence") or 1.0)
        confidence = max(0.0, min(1.0, confidence))

        # ── Recency ─────────────────────────────────────────────────────────
        # Prefer event_time (when the conversation happened) over created_at
        # (when the row was inserted). In benchmark runs all rows are inserted
        # today so created_at gives age_days≈0 for every fact, making decay
        # a no-op. event_time carries the actual historical session date.
        timestamp = fact.get("event_time") or fact.get("created_at")
        age_days = _age_in_days(timestamp, now)
        recency = math.exp(-temporal_decay_rate * age_days)

        # ── Mentions boost ───────────────────────────────────────────────────
        # log(1 + mentions) gives diminishing returns: 1→1.0, 5→1.79, 50→3.93.
        # Normalise against max in this result set and squeeze into [1.0, 1.5].
        # That way a heavily re-encountered fact gets at most a 50% lift —
        # meaningful but it can't override a much stronger similarity signal.
        if use_mentions and max_mentions > 1:
            mentions = int(fact.get("mentions") or 1)
            raw_boost = math.log1p(mentions) / math.log1p(max_mentions)  # → [0, 1]
            mentions_boost = 1.0 + 0.5 * raw_boost                       # → [1.0, 1.5]
        else:
            mentions_boost = 1.0

        score = similarity * confidence * recency * mentions_boost

        scored.append({
            **fact,
            "score": round(score, 6),
            # Expose components for debugging / explanation
            "_score_components": {
                "similarity": round(similarity, 4),
                "confidence": round(confidence, 4),
                "recency": round(recency, 4),
                "mentions_boost": round(mentions_boost, 4),
            },
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def _age_in_days(created_at: Any, now: datetime) -> float:
    """Return age in fractional days. Handles datetime, ISO string, and None."""
    if isinstance(created_at, datetime):
        # asyncpg returns timezone-aware datetimes; strip tz for comparison.
        ts = created_at.replace(tzinfo=None) if created_at.tzinfo else created_at
        return max(0.0, (now - ts).total_seconds() / 86400)
    if isinstance(created_at, str):
        try:
            ts = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            ts = ts.replace(tzinfo=None)
            return max(0.0, (now - ts).total_seconds() / 86400)
        except ValueError:
            pass
    return 0.0


def explain_score(fact: dict[str, Any]) -> str:
    """Return a human-readable breakdown of a scored fact's score.

    Useful for debugging retrieval quality during development.

    Example:
        results = await v.search(query, user_id, depth="l0")
        for f in results["facts"]:
            print(explain_score(f))
    """
    components = fact.get("_score_components", {})
    if not components:
        return f"score={fact.get('score', '?')} (no breakdown available)"

    return (
        f"score={fact['score']:.4f}  "
        f"[sim={components.get('similarity', '?'):.4f} × "
        f"conf={components.get('confidence', '?'):.4f} × "
        f"recency={components.get('recency', '?'):.4f} × "
        f"mentions={components.get('mentions_boost', '?'):.4f}]  "
        f"→ \"{fact.get('text', '')[:60]}\""
    )

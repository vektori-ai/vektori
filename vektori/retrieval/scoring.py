"""Fact scoring: cosine similarity × extraction confidence × temporal decay."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any


def score_and_rank(
    facts: list[dict[str, Any]],
    temporal_decay_rate: float = 0.001,
) -> list[dict[str, Any]]:
    """
    Score facts by combining vector similarity, confidence, and recency.

    score = cosine_similarity * extraction_confidence * temporal_decay

    Args:
        facts: Dicts with "distance" (pgvector cosine distance, lower = closer),
               "confidence" (LLM extraction confidence 0-1),
               "created_at" (datetime).
        temporal_decay_rate: Decay per day. Default 0.001 ≈ 36% decay/year.

    Returns:
        Facts sorted by score descending, with "score" field added.
    """
    now = datetime.utcnow()
    scored = []

    for fact in facts:
        # pgvector returns cosine distance (0=identical, 2=opposite)
        distance = fact.get("distance", 0.0)
        similarity = max(0.0, 1.0 - distance)

        # LLM extraction confidence
        confidence = fact.get("confidence", 1.0)

        # Temporal decay: newer = higher score
        created_at = fact.get("created_at")
        if isinstance(created_at, datetime):
            age_days = (now - created_at).total_seconds() / 86400
        else:
            age_days = 0.0
        recency = math.exp(-temporal_decay_rate * age_days)

        scored.append({**fact, "score": similarity * confidence * recency})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored

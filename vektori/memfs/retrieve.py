"""Retrieval: RRF fusion of BM25 + vector, 1-hop link expansion, recency prior.

No LLM in this path, ever. Scores are decomposed per signal so results are
debuggable and the eval can ablate each signal.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from vektori.memfs.index import Index
from vektori.memfs.models import RecallItem, RecallResult

RRF_K = 60
LINK_DAMP = 0.5
RECENCY_WEIGHT = 0.05
RECENCY_HALF_LIFE_DAYS = 30.0
SNIPPET_CHARS = 400


def rrf(rankings: dict[str, list[str]], k: int = RRF_K) -> dict[str, dict[str, float]]:
    """rankings: signal -> ordered chunk_ids. Returns chunk_id -> per-signal RRF scores."""
    scores: dict[str, dict[str, float]] = {}
    for signal, ids in rankings.items():
        for rank, cid in enumerate(ids):
            scores.setdefault(cid, {})[signal] = 1.0 / (k + rank + 1)
    return scores


def _recency_boost(row, now: datetime) -> float:
    """Recency prior applies to episodic notes only — stable knowledge must not
    be punished for being old. Uses event time (when) over created time."""
    if row["type"] != "episodic":
        return 0.0
    ts = row["when_ts"] or row["created_ts"]
    if not ts:
        return 0.0
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return 0.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    age_days = max((now - dt).total_seconds() / 86400.0, 0.0)
    return math.exp(-age_days * math.log(2) / RECENCY_HALF_LIFE_DAYS)


async def recall(
    index: Index,
    query: str,
    embedder=None,
    k: int = 8,
    types: list[str] | None = None,
    since: datetime | None = None,
    expand_links: bool = True,
    include_archive: bool = False,
) -> RecallResult:
    candidates = 40
    rankings: dict[str, list[str]] = {}

    bm = index.search_bm25(query, limit=candidates, types=types)
    rankings["bm25"] = [cid for cid, _ in bm]

    if embedder is not None:
        qvec = await embedder.embed(query)
        vec = index.search_vector(qvec, limit=candidates, types=types)
        rankings["vec"] = [cid for cid, _ in vec]

    fused = rrf(rankings)

    if expand_links and fused:
        seed_ids = sorted(fused, key=lambda c: sum(fused[c].values()), reverse=True)[:4]
        seed_rows = index.chunks_by_ids(seed_ids)
        seed_files = list({r["file_id"] for r in seed_rows.values()})
        neighbor_files = index.linked_neighbors(seed_files)
        for cid in index.chunks_for_files(neighbor_files, per_file=1):
            base = 1.0 / (RRF_K + 1)
            fused.setdefault(cid, {})["link"] = base * LINK_DAMP

    if not fused:
        return RecallResult(query=query)

    rows = index.chunks_by_ids(list(fused.keys()))
    now = datetime.now(timezone.utc)
    items: list[RecallItem] = []
    for cid, signals in fused.items():
        row = rows.get(cid)
        if row is None:
            continue
        if not include_archive and "/archive/" in row["path"]:
            continue
        if since is not None:
            ts = row["when_ts"] or row["created_ts"]
            if ts:
                try:
                    dt = datetime.fromisoformat(ts)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    ref = since if since.tzinfo else since.replace(tzinfo=timezone.utc)
                    if dt < ref:
                        continue
                except ValueError:
                    pass
        rec = _recency_boost(row, now)
        if rec:
            signals = {**signals, "recency": rec * RECENCY_WEIGHT}
        score = sum(signals.values())
        # access count as a stable tiebreaker, never a real signal
        score += min(row["access_count"], 100) * 1e-6
        items.append(RecallItem(
            path=row["path"],
            title=row["title"],
            type=row["type"],
            snippet=row["text"][:SNIPPET_CHARS],
            start_line=row["start_line"] or 1,
            end_line=row["end_line"] or 1,
            score=score,
            signals={s: round(v, 6) for s, v in signals.items()},
            provenance=row["source"],
            note_id=row["file_id"],
        ))

    items.sort(key=lambda i: i.score, reverse=True)
    items = items[:k]
    index.bump_access([i.path for i in items])
    return RecallResult(query=query, items=items)

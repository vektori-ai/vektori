"""Personalized PageRank over the fact+episode graph.

Runs per-user on a small graph (typically 50-500 fact nodes + episode nodes).
No external dependencies — pure Python, converges in <1ms for typical graphs.

Graph structure:
  - Fact nodes connected by write-time similarity edges (fact_edges table)
  - Episode nodes connected to their constituent facts (episode_facts table)
  - All edges treated as undirected

Seeding: cosine-matched facts from the initial vector search, weighted by score.
Output: PPR probability distribution over all nodes.
"""

from __future__ import annotations

from typing import Any


def run_ppr(
    seed_scores: dict[str, float],
    all_facts: list[dict[str, Any]],
    fact_edges: list[dict[str, Any]],
    episode_fact_map: dict[str, list[str]],
    alpha: float = 0.5,
    iterations: int = 15,
) -> dict[str, float]:
    """Return PPR probability scores keyed by node ID (facts and episodes).

    Args:
        seed_scores:      {fact_id: cosine_similarity} from initial vector search.
        all_facts:        All active facts for the user (dicts with at least "id").
        fact_edges:       [{source_id, target_id, weight}] from fact_edges table.
        episode_fact_map: {episode_id: [fact_id, ...]} for all user episodes.
        alpha:            Restart probability (teleport back to seed nodes).
        iterations:       Power-iteration steps. 15 is sufficient for small graphs.
    """
    fact_ids = [f["id"] for f in all_facts]
    episode_ids = list(episode_fact_map.keys())
    all_nodes = fact_ids + episode_ids

    if not all_nodes:
        return {}

    node_idx: dict[str, int] = {nid: i for i, nid in enumerate(all_nodes)}
    N = len(all_nodes)

    # Adjacency lists (undirected)
    adj: list[list[int]] = [[] for _ in range(N)]

    for edge in fact_edges:
        src, tgt = edge["source_id"], edge["target_id"]
        if src in node_idx and tgt in node_idx:
            i, j = node_idx[src], node_idx[tgt]
            adj[i].append(j)
            adj[j].append(i)

    for ep_id, f_ids in episode_fact_map.items():
        if ep_id not in node_idx:
            continue
        ep_i = node_idx[ep_id]
        for fid in f_ids:
            if fid in node_idx:
                fi = node_idx[fid]
                adj[ep_i].append(fi)
                adj[fi].append(ep_i)

    # Deduplicate adjacency
    adj = [list(dict.fromkeys(nbrs)) for nbrs in adj]

    # Personalization vector — seed weights from cosine scores
    total = sum(seed_scores.values()) or 1.0
    e_u = [0.0] * N
    for fid, score in seed_scores.items():
        if fid in node_idx:
            e_u[node_idx[fid]] = score / total

    # Power iteration: p = alpha * e_u + (1-alpha) * A^T D^-1 p
    p = e_u[:]
    for _ in range(iterations):
        p_new = [alpha * e_u[i] for i in range(N)]
        for i in range(N):
            nbrs = adj[i]
            if nbrs and p[i] > 0:
                contrib = (1.0 - alpha) * p[i] / len(nbrs)
                for j in nbrs:
                    p_new[j] += contrib
        p = p_new

    return {all_nodes[i]: p[i] for i in range(N)}


def rank_episodes_by_ppr(
    episode_fact_map: dict[str, list[str]],
    ppr_scores: dict[str, float],
) -> list[tuple[str, float]]:
    """Return episodes sorted by their PPR-aggregated score (descending).

    An episode's score = sum of PPR scores of its constituent facts.
    This surfaces episodes whose facts are reachable from the query seeds
    even if they weren't directly matched by cosine.
    """
    ranked: list[tuple[str, float]] = []
    for ep_id, fact_ids in episode_fact_map.items():
        ep_score = sum(ppr_scores.get(fid, 0.0) for fid in fact_ids)
        # Also add the episode's own PPR score (it's a node too)
        ep_score += ppr_scores.get(ep_id, 0.0)
        ranked.append((ep_id, ep_score))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked

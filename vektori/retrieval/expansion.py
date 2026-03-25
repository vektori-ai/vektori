"""Session context expansion utilities."""

from __future__ import annotations

from typing import Any


def group_by_session(sentences: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group expanded sentences by session_id for structured display."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for sent in sentences:
        sid = sent.get("session_id", "unknown")
        groups.setdefault(sid, []).append(sent)
    for group in groups.values():
        group.sort(key=lambda x: (x.get("turn_number", 0), x.get("sentence_index", 0)))
    return groups

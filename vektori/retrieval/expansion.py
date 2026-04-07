"""Post-processing utilities for retrieval results.

These helpers sit between the raw search result dict and what you actually
inject into an LLM prompt. The DB expansion logic lives in the storage
backend; this module handles grouping, annotation, and formatting.
"""

from __future__ import annotations

from typing import Any

# ── Grouping ──────────────────────────────────────────────────────────────────


def group_by_session(
    sentences: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group sentences by session_id, sorted by turn/position within each group.

    Useful for rendering context as separate conversation excerpts rather than
    a flat list when sentences span multiple sessions.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for sent in sentences:
        sid = sent.get("session_id", "unknown")
        groups.setdefault(sid, []).append(sent)
    for group in groups.values():
        group.sort(key=lambda x: (x.get("turn_number", 0), x.get("sentence_index", 0)))
    return groups


# ── Source annotation ─────────────────────────────────────────────────────────


def mark_sources(
    sentences: list[dict[str, Any]],
    source_ids: set[str],
) -> list[dict[str, Any]]:
    """Annotate sentences with ``is_source=True`` if they directly yielded a fact.

    L2 expansion returns both the source sentences and their ±window neighbours.
    This lets callers distinguish the exact moments that triggered retrieval from
    the surrounding padding context.

    Args:
        sentences: Flat list of sentence dicts (from search result ``"sentences"``).
        source_ids: Set of sentence IDs that are direct fact sources.

    Returns:
        Same list with ``is_source`` bool added to each dict (new dict, not mutated).
    """
    return [{**s, "is_source": s.get("id") in source_ids} for s in sentences]


# ── Formatting ────────────────────────────────────────────────────────────────


def format_context_window(
    sentences: list[dict[str, Any]],
    *,
    source_ids: set[str] | None = None,
    show_role: bool = True,
) -> str:
    """Format a list of sentences into a readable context block.

    Groups by session, preserves sequential order, and optionally marks
    source sentences with a ``[*]`` tag so the LLM can see which moments
    were retrieved vs which are surrounding context.

    Args:
        sentences: Flat list of sentence dicts with at least ``text``,
                   ``session_id``, ``turn_number``, ``sentence_index``.
        source_ids: If provided, mark matching sentences with ``[*]``.
        show_role: Prefix each line with ``[user]`` / ``[assistant]``.

    Returns:
        Multi-line string ready for LLM context injection.
    """
    if not sentences:
        return ""

    groups = group_by_session(sentences)
    parts: list[str] = []

    for session_idx, (session_id, group) in enumerate(groups.items()):
        if len(groups) > 1:
            parts.append(f"--- Session {session_idx + 1} ({session_id[:8]}…) ---")

        for sent in group:
            text = sent.get("text", "").strip()
            if not text:
                continue

            prefix = ""
            if show_role:
                role = sent.get("role", "user")
                prefix = f"[{role}] "

            marker = ""
            if source_ids is not None and sent.get("id") in source_ids:
                marker = " [*]"

            parts.append(f"{prefix}{text}{marker}")

        if len(groups) > 1:
            parts.append("")  # blank line between sessions

    return "\n".join(parts).strip()


def build_retrieval_context(
    result: dict[str, Any],
    *,
    include_facts: bool = True,
    include_sentences: bool = True,
    mark_source_sentences: bool = True,
    show_role: bool = True,
) -> str:
    """Assemble a full retrieval result into a single context string.

    This is the main helper most callers will use. Takes the dict returned
    by ``SearchPipeline.search()`` and produces a formatted block ready
    to inject into an LLM system prompt or user message.

    Output structure (sections omitted if empty or disabled):

        ## Facts
        - <fact text>

        ## Context
        [user] sentence... [*]
        [assistant] sentence...

    The ``[*]`` marker (when ``mark_source_sentences=True``) indicates the
    exact sentence that a fact was extracted from. Surrounding lines are
    context window padding.

    Args:
        result: Return value from ``SearchPipeline.search()``.
        include_facts: Include the facts section.
        include_sentences: Include the sentence context section.
        mark_source_sentences: Tag direct source sentences with ``[*]``.
        show_role: Prefix sentence lines with ``[user]``/``[assistant]``.

    Returns:
        Formatted string, or empty string if all sections are empty.
    """
    sections: list[str] = []

    # ── Facts ─────────────────────────────────────────────────────────────────
    if include_facts:
        facts = result.get("facts", [])
        if facts:
            lines = ["## Facts"]
            for f in facts:
                lines.append(f"- {f.get('text', '').strip()}")
            sections.append("\n".join(lines))

    # ── Sentences (context window) ─────────────────────────────────────────────
    if include_sentences:
        sentences = result.get("sentences", [])
        if sentences:
            # Source IDs: sentences where is_source=True (set by mark_sources),
            # or we infer from facts via fact_sources if the caller hasn't
            # pre-annotated. Fall back to marking nothing if unavailable.
            if mark_source_sentences:
                source_ids: set[str] | None = {s["id"] for s in sentences if s.get("is_source")}
                # If no pre-annotation, source_ids stays empty — no markers shown.
                if not source_ids:
                    source_ids = None
            else:
                source_ids = None

            context_text = format_context_window(
                sentences,
                source_ids=source_ids,
                show_role=show_role,
            )
            if context_text:
                sections.append("## Context\n" + context_text)

    return "\n\n".join(sections)

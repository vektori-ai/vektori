"""Deterministic prompt assembly for the agent harness."""

from __future__ import annotations

from typing import Any

from vektori.context import LoadedAgentContext
from vektori.memory.profile import ProfilePatch
from vektori.memory.window import WindowState


def _render_system_prompt(
    context: LoadedAgentContext,
    profile_patches: list[ProfilePatch],
    runtime_overrides: dict[str, Any] | None = None,
) -> str:
    lines: list[str] = []
    if context.persona:
        lines.append(context.persona)
    if context.instructions:
        lines.append("Instructions:")
        lines.extend(f"- {instruction}" for instruction in context.instructions)
    if profile_patches:
        lines.append("Active profile patches:")
        lines.extend(f"- {patch.key} = {patch.value} ({patch.reason})" for patch in profile_patches)
    if runtime_overrides:
        lines.append("Runtime overrides:")
        lines.extend(f"- {key}: {value}" for key, value in runtime_overrides.items())
    return "\n".join(lines).strip()


def render_retrieved_memory(memories: dict[str, list[dict[str, Any]]]) -> str:
    facts = memories.get("facts", [])
    episodes = memories.get("episodes", [])
    sentences = memories.get("sentences", [])
    blocks: list[str] = ["Retrieved Memory"]
    if facts:
        blocks.append("\nFacts")
        blocks.extend(f"- {fact.get('text', '').strip()}" for fact in facts if fact.get("text"))
    if episodes:
        blocks.append("\nEpisodes")
        blocks.extend(f"- {episode.get('text', '').strip()}" for episode in episodes if episode.get("text"))
    if sentences:
        blocks.append("\nSentences")
        blocks.extend(
            f"- {sentence.get('text', '').strip()}" for sentence in sentences if sentence.get("text")
        )
    return "\n".join(blocks).strip() if len(blocks) > 1 else ""


def build_messages(
    *,
    context: LoadedAgentContext,
    profile_patches: list[ProfilePatch],
    memories: dict[str, list[dict[str, Any]]] | None,
    window_state: WindowState,
    runtime_overrides: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    system_prompt = _render_system_prompt(context, profile_patches, runtime_overrides)
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if memories:
        memory_block = render_retrieved_memory(memories)
        if memory_block:
            messages.append({"role": "system", "content": memory_block})

    if window_state.rolling_summary:
        messages.append({"role": "system", "content": window_state.rolling_summary})

    messages.extend(window_state.recent_messages)
    return messages

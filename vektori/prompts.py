"""Deterministic prompt assembly for the agent harness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vektori.context import LoadedAgentContext
from vektori.memory.profile import ProfilePatch
from vektori.memory.window import WindowState


def estimate_tokens_for_text(text: str) -> int:
    """Rough token estimate based on character count."""
    return max(1, len(text) // 4) if text else 0


def estimate_tokens_for_messages(messages: list[dict[str, str]]) -> int:
    """Rough token estimate for a chat message list."""
    return sum(
        estimate_tokens_for_text(message.get("content", "")) + 4 for message in messages
    )


@dataclass
class PromptBuildResult:
    """Prompt assembly output plus trimming diagnostics."""

    messages: list[dict[str, str]]
    memories_used: dict[str, list[dict[str, Any]]]
    prompt_debug: dict[str, Any]


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


def build_prompt_result(
    *,
    context: LoadedAgentContext,
    profile_patches: list[ProfilePatch],
    memories: dict[str, list[dict[str, Any]]] | None,
    window_state: WindowState,
    max_context_tokens: int,
    reserve_response_tokens: int,
    runtime_overrides: dict[str, Any] | None = None,
) -> PromptBuildResult:
    budget = max(256, max_context_tokens - reserve_response_tokens)
    messages: list[dict[str, str]] = []
    working_memories = {
        "facts": list((memories or {}).get("facts", [])),
        "episodes": list((memories or {}).get("episodes", [])),
        "sentences": list((memories or {}).get("sentences", [])),
    }
    trimmed = {
        "sentences": 0,
        "episodes": 0,
        "facts": 0,
        "recent_messages": 0,
        "summary_dropped": False,
    }
    summary_text = window_state.rolling_summary
    recent_messages = list(window_state.recent_messages)

    def _assemble() -> list[dict[str, str]]:
        assembled: list[dict[str, str]] = []
        system_prompt = _render_system_prompt(context, profile_patches, runtime_overrides)
        if system_prompt:
            assembled.append({"role": "system", "content": system_prompt})

        memory_block = render_retrieved_memory(working_memories)
        if memory_block:
            assembled.append({"role": "system", "content": memory_block})

        if summary_text:
            assembled.append({"role": "system", "content": summary_text})

        assembled.extend(recent_messages)
        return assembled

    def _within_budget() -> bool:
        return estimate_tokens_for_messages(_assemble()) <= budget

    def _trim_memory_bucket(name: str) -> bool:
        bucket = working_memories[name]
        if not bucket:
            return False
        bucket.pop()
        trimmed[name] += 1
        return True

    def _trim_recent_message() -> bool:
        if len(recent_messages) <= 1:
            return False
        recent_messages.pop(0)
        trimmed["recent_messages"] += 1
        return True

    system_prompt = _render_system_prompt(context, profile_patches, runtime_overrides)
    system_prompt = _render_system_prompt(context, profile_patches, runtime_overrides)
    if estimate_tokens_for_text(system_prompt) > budget and system_prompt:
        # Keep the system prompt intact even if it consumes the budget.
        budget = estimate_tokens_for_text(system_prompt) + 32

    trim_order = [
        lambda: _trim_memory_bucket("sentences"),
        lambda: _trim_memory_bucket("episodes"),
        lambda: _trim_memory_bucket("facts"),
        _trim_recent_message,
    ]

    while not _within_budget():
        changed = False
        for trim_step in trim_order:
            if trim_step():
                changed = True
                break
        if changed:
            continue
        if summary_text:
            summary_text = ""
            trimmed["summary_dropped"] = True
            continue
        break

    messages = _assemble()
    return PromptBuildResult(
        messages=messages,
        memories_used=working_memories,
        prompt_debug={
            "budget_tokens": budget,
            "estimated_prompt_tokens": estimate_tokens_for_messages(messages),
            "trimmed": trimmed,
            "summary_included": bool(summary_text),
            "recent_message_count": len(recent_messages),
        },
    )


def build_messages(
    *,
    context: LoadedAgentContext,
    profile_patches: list[ProfilePatch],
    memories: dict[str, list[dict[str, Any]]] | None,
    window_state: WindowState,
    max_context_tokens: int,
    reserve_response_tokens: int,
    runtime_overrides: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    return build_prompt_result(
        context=context,
        profile_patches=profile_patches,
        memories=memories,
        window_state=window_state,
        max_context_tokens=max_context_tokens,
        reserve_response_tokens=reserve_response_tokens,
        runtime_overrides=runtime_overrides,
    ).messages

"""
Memory tool schemas and handlers for VektoriAgent Phase 4 tool calling.

Tools exposed to the model:
  search_memory   — query long-term memory at any depth
  get_profile     — read active profile patches for the user
  update_profile  — persist a new durable preference patch

The agent passes MEMORY_TOOLS to model.complete(tools=...) when
enable_tool_calling=True. Tool results are formatted and returned as
tool-role messages for the next model round-trip.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vektori.agent import VektoriAgent

logger = logging.getLogger(__name__)

MEMORY_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": (
                "Search Vektori long-term memory for facts and episodes relevant to a query. "
                "Use when you need historical context about the user that may not be in "
                "the current conversation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language question or topic to search for.",
                    },
                    "depth": {
                        "type": "string",
                        "enum": ["l0", "l1", "l2"],
                        "description": (
                            "Retrieval depth. l0=facts only, l1=facts+episodes (default), "
                            "l2=facts+episodes+raw sentences."
                        ),
                        "default": "l1",
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "description": "Maximum number of results to return.",
                        "default": 6,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_profile",
            "description": (
                "Retrieve all active durable profile patches (learned preferences) for the user."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_profile",
            "description": (
                "Persist a durable user preference so it influences all future turns. "
                "Only use for explicit stable preferences the user clearly stated."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": (
                            "Dot-separated preference key, e.g. "
                            "'response_style.verbosity', 'preferences.name', 'preferences.units'."
                        ),
                    },
                    "value": {
                        "description": "Preference value. String, number, or boolean.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Short human-readable reason for storing this preference.",
                    },
                },
                "required": ["key", "value", "reason"],
            },
        },
    },
]


async def handle_tool_call(
    tool_name: str,
    tool_args: dict[str, Any],
    agent: VektoriAgent,
) -> str:
    """
    Execute a tool call and return a string result for injection back into the model.

    Returns a plain-text representation of the result (not JSON-only, so the model
    can cite it directly if needed).
    """
    if tool_name == "search_memory":
        return await _handle_search_memory(tool_args, agent)
    if tool_name == "get_profile":
        return await _handle_get_profile(agent)
    if tool_name == "update_profile":
        return await _handle_update_profile(tool_args, agent)
    return f"Unknown tool: {tool_name}"


async def _handle_search_memory(args: dict[str, Any], agent: VektoriAgent) -> str:
    query = args.get("query", "")
    depth = args.get("depth", "l1")
    top_k = int(args.get("top_k", 6))
    try:
        results = await agent.memory.search(
            query=query,
            user_id=agent.user_id,
            agent_id=agent.agent_id,
            depth=depth,
            top_k=top_k,
        )
    except Exception as e:
        logger.warning("search_memory tool call failed: %s", e)
        return f"Memory search failed: {e}"

    facts = results.get("facts", [])
    episodes = results.get("episodes", []) or results.get("sentences", [])
    lines: list[str] = []
    if facts:
        lines.append("Facts:")
        lines.extend(f"- {f.get('text', str(f))}" for f in facts)
    if episodes:
        lines.append("Episodes:")
        lines.extend(f"- {e.get('text', str(e))}" for e in episodes)
    return "\n".join(lines) if lines else "No relevant memory found."


async def _handle_get_profile(agent: VektoriAgent) -> str:
    patches = await agent.profile_store.list_active(
        observer_id=agent.agent_id or "default-agent",
        observed_id=agent.user_id,
    )
    if not patches:
        return "No profile patches found."
    lines = [f"- {p.key} = {p.value}  ({p.reason})" for p in patches]
    return "Active profile patches:\n" + "\n".join(lines)


async def _handle_update_profile(args: dict[str, Any], agent: VektoriAgent) -> str:
    from datetime import datetime, timezone

    from vektori.memory.profile import ProfilePatch

    key = str(args.get("key", ""))
    value = args.get("value")
    reason = str(args.get("reason", ""))
    if not key or value is None:
        return "update_profile: 'key' and 'value' are required."

    patch = ProfilePatch(
        key=key,
        value=value,
        reason=reason,
        source="tool_call",
        observer_id=agent.agent_id or "default-agent",
        observed_id=agent.user_id,
        confidence=0.9,
        created_at=datetime.now(timezone.utc),
    )
    await agent.profile_store.save(patch)
    logger.debug("Tool: saved profile patch key=%s value=%s", key, value)
    return f"Saved preference: {key} = {value}"

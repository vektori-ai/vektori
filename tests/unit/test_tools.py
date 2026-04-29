"""Unit tests for Phase 4 memory tools and agent tool calling loop."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from vektori.agent import AgentConfig, VektoriAgent
from vektori.memory.profile import InMemoryProfileStore
from vektori.models.base import ChatCompletionResult
from vektori.tools.memory import MEMORY_TOOLS, handle_tool_call


# — Tool schema tests —

def test_memory_tools_has_three_tools():
    names = {t["function"]["name"] for t in MEMORY_TOOLS}
    assert names == {"search_memory", "get_profile", "update_profile"}


def test_search_memory_schema_required_query():
    schema = next(t for t in MEMORY_TOOLS if t["function"]["name"] == "search_memory")
    assert "query" in schema["function"]["parameters"]["required"]


def test_update_profile_schema_required_fields():
    schema = next(t for t in MEMORY_TOOLS if t["function"]["name"] == "update_profile")
    required = schema["function"]["parameters"]["required"]
    assert "key" in required
    assert "value" in required
    assert "reason" in required


# — handle_tool_call dispatch —

def _make_agent(facts=None):
    memory = AsyncMock()
    memory.search.return_value = {"facts": facts or [], "sentences": [], "episodes": []}
    memory.add.return_value = {}
    model = AsyncMock()
    result = MagicMock(spec=ChatCompletionResult)
    result.content = "ok"
    result.tool_calls = []
    result.usage = None
    model.complete.return_value = result
    return VektoriAgent(memory=memory, model=model, user_id="u1")


@pytest.mark.asyncio
async def test_handle_search_memory_returns_facts():
    agent = _make_agent(facts=[{"text": "User works in ML"}])
    result = await handle_tool_call("search_memory", {"query": "what does user do?"}, agent)
    assert "User works in ML" in result


@pytest.mark.asyncio
async def test_handle_search_memory_no_results():
    agent = _make_agent()
    result = await handle_tool_call("search_memory", {"query": "anything"}, agent)
    assert "No relevant memory found" in result


@pytest.mark.asyncio
async def test_handle_get_profile_empty():
    agent = _make_agent()
    result = await handle_tool_call("get_profile", {}, agent)
    assert "No profile patches" in result


@pytest.mark.asyncio
async def test_handle_get_profile_with_patches():
    from datetime import datetime, timezone
    from vektori.memory.profile import InMemoryProfileStore, ProfilePatch

    store = InMemoryProfileStore()
    await store.save(ProfilePatch(
        key="preferences.name",
        value="Alex",
        reason="user stated",
        source="explicit_user_request",
        observer_id="default-agent",
        observed_id="u1",
        confidence=0.9,
    ))
    agent = _make_agent()
    agent.profile_store = store
    result = await handle_tool_call("get_profile", {}, agent)
    assert "preferences.name" in result
    assert "Alex" in result


@pytest.mark.asyncio
async def test_handle_update_profile_saves_patch():
    agent = _make_agent()
    result = await handle_tool_call(
        "update_profile",
        {"key": "preferences.units", "value": "metric", "reason": "user asked"},
        agent,
    )
    assert "metric" in result
    patches = await agent.profile_store.list_active("default-agent", "u1")
    assert any(p.key == "preferences.units" for p in patches)


@pytest.mark.asyncio
async def test_handle_unknown_tool():
    agent = _make_agent()
    result = await handle_tool_call("nonexistent_tool", {}, agent)
    assert "Unknown tool" in result


# — Agent tool calling loop —

def _make_tool_calling_model(first_tool_name: str, tool_args: dict, final_content: str):
    """Mock model that fires one tool call then returns final_content."""
    mock = AsyncMock()

    # First call: return a tool_call
    tool_call = MagicMock()
    tool_call.function = MagicMock()
    tool_call.function.name = first_tool_name
    import json
    tool_call.function.arguments = json.dumps(tool_args)
    tool_call.id = f"call-{first_tool_name}"

    first_result = MagicMock(spec=ChatCompletionResult)
    first_result.content = None
    first_result.tool_calls = [tool_call]
    first_result.usage = None

    # Second call: final response
    second_result = MagicMock(spec=ChatCompletionResult)
    second_result.content = final_content
    second_result.tool_calls = []
    second_result.usage = None

    mock.complete.side_effect = [first_result, second_result]
    return mock


@pytest.mark.asyncio
async def test_agent_tool_calling_loop_executes_tool():
    memory = AsyncMock()
    memory.search.return_value = {"facts": [{"text": "User is a Python developer"}], "sentences": [], "episodes": []}
    memory.add.return_value = {}

    model = _make_tool_calling_model(
        first_tool_name="search_memory",
        tool_args={"query": "what does user do?"},
        final_content="You are a Python developer.",
    )
    config = AgentConfig(enable_tool_calling=True, background_add=False)
    agent = VektoriAgent(memory=memory, model=model, user_id="u1", config=config)

    result = await agent.chat("What do I do for work?")
    assert result.content == "You are a Python developer."
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "search_memory"
    assert model.complete.call_count == 2


@pytest.mark.asyncio
async def test_agent_tool_calling_disabled_by_default():
    memory = AsyncMock()
    memory.search.return_value = {"facts": [], "sentences": [], "episodes": []}
    memory.add.return_value = {}

    model = AsyncMock()
    final = MagicMock(spec=ChatCompletionResult)
    final.content = "hi"
    final.tool_calls = []
    final.usage = None
    model.complete.return_value = final

    agent = VektoriAgent(memory=memory, model=model, user_id="u1")
    result = await agent.chat("hi")

    # With tool calling disabled, complete is called exactly once with no tools kwarg
    call_kwargs = model.complete.call_args.kwargs
    assert "tools" not in call_kwargs or call_kwargs.get("tools") is None
    assert result.tool_calls == []


@pytest.mark.asyncio
async def test_agent_tool_loop_respects_max_round_trips():
    """Tool calls that never stop should be capped at max_tool_round_trips."""
    import json

    memory = AsyncMock()
    memory.search.return_value = {"facts": [], "sentences": [], "episodes": []}
    memory.add.return_value = {}

    # Model always returns a tool call, never a final text response
    tool_call = MagicMock()
    tool_call.function = MagicMock()
    tool_call.function.name = "search_memory"
    tool_call.function.arguments = json.dumps({"query": "test"})
    tool_call.id = "call-1"

    looping_result = MagicMock(spec=ChatCompletionResult)
    looping_result.content = None
    looping_result.tool_calls = [tool_call]
    looping_result.usage = None

    final_result = MagicMock(spec=ChatCompletionResult)
    final_result.content = "done"
    final_result.tool_calls = []
    final_result.usage = None

    model = AsyncMock()
    # Always loop except the very last call
    max_trips = 2
    model.complete.side_effect = [looping_result] * max_trips + [final_result]

    config = AgentConfig(enable_tool_calling=True, max_tool_round_trips=max_trips, background_add=False)
    agent = VektoriAgent(memory=memory, model=model, user_id="u1", config=config)

    result = await agent.chat("keep looping")
    assert result.content == "done"
    assert model.complete.call_count == max_trips + 1

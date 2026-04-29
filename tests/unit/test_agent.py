"""Unit tests for VektoriAgent — aligned to harness branch API."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from vektori.agent import AgentConfig, AgentTurnResult, VektoriAgent
from vektori.memory.profile import InMemoryProfileStore, ProfilePatch
from vektori.models.base import ChatCompletionResult


def _make_mock_memory(facts=None, sentences=None, episodes=None):
    mock = AsyncMock()
    mock.search.return_value = {
        "facts": facts or [],
        "sentences": sentences or [],
        "episodes": episodes or [],
    }
    mock.add.return_value = {"status": "ok", "sentences_stored": 1}
    return mock


def _make_mock_model(content="Hello!"):
    mock = AsyncMock()
    result = MagicMock(spec=ChatCompletionResult)
    result.content = content
    result.tool_calls = []
    result.usage = {"prompt_tokens": 10, "completion_tokens": 5}
    mock.complete.return_value = result
    return mock


def _make_patch(**kwargs) -> ProfilePatch:
    defaults = dict(
        key="response_style.verbosity",
        value="short",
        reason="user asked",
        source="explicit_user_request",
        observer_id="default-agent",
        observed_id="u1",
        confidence=0.9,
    )
    defaults.update(kwargs)
    return ProfilePatch(**defaults)


@pytest.mark.asyncio
async def test_chat_returns_turn_result():
    memory = _make_mock_memory()
    model = _make_mock_model("Hello!")
    agent = VektoriAgent(memory=memory, model=model, user_id="u1")

    result = await agent.chat("hi there")
    assert isinstance(result, AgentTurnResult)
    assert result.content == "Hello!"
    assert result.tool_calls == []


@pytest.mark.asyncio
async def test_chat_calls_model_with_messages():
    memory = _make_mock_memory()
    model = _make_mock_model()
    agent = VektoriAgent(memory=memory, model=model, user_id="u1")

    await agent.chat("What do I prefer?")
    model.complete.assert_called_once()
    call_args = model.complete.call_args[0][0]
    assert any(m["role"] == "user" for m in call_args)


@pytest.mark.asyncio
async def test_chat_injects_memory_when_personal_query():
    memory = _make_mock_memory(facts=[{"text": "User likes Python"}])
    model = _make_mock_model("You like Python.")
    agent = VektoriAgent(memory=memory, model=model, user_id="u1")

    result = await agent.chat("What do I like?")
    assert result.memories_used["facts"][0]["text"] == "User likes Python"
    call_messages = model.complete.call_args[0][0]
    joined = " ".join(m["content"] for m in call_messages)
    assert "User likes Python" in joined


@pytest.mark.asyncio
async def test_chat_skips_retrieval_for_filler():
    memory = _make_mock_memory()
    model = _make_mock_model("Sure!")
    agent = VektoriAgent(memory=memory, model=model, user_id="u1")

    await agent.chat("ok")
    memory.search.assert_not_called()


@pytest.mark.asyncio
async def test_chat_retrieves_on_every_turn_when_configured():
    memory = _make_mock_memory()
    model = _make_mock_model("ok")
    config = AgentConfig(retrieve_on_every_turn=True)
    agent = VektoriAgent(memory=memory, model=model, user_id="u1", config=config)

    await agent.chat("ok")
    memory.search.assert_called_once()


@pytest.mark.asyncio
async def test_chat_background_add_called():
    memory = _make_mock_memory()
    model = _make_mock_model()
    config = AgentConfig(background_add=True)
    agent = VektoriAgent(memory=memory, model=model, user_id="u1", config=config)

    await agent.chat("remember I like cats")
    import asyncio
    await asyncio.sleep(0)
    if agent._background_tasks:
        await asyncio.gather(*list(agent._background_tasks), return_exceptions=True)
    memory.add.assert_called()


@pytest.mark.asyncio
async def test_chat_synchronous_add_when_background_disabled():
    memory = _make_mock_memory()
    model = _make_mock_model()
    config = AgentConfig(background_add=False)
    agent = VektoriAgent(memory=memory, model=model, user_id="u1", config=config)

    await agent.chat("remember I like dogs")
    memory.add.assert_called_once()


@pytest.mark.asyncio
async def test_window_accumulates_turns():
    memory = _make_mock_memory()
    model = _make_mock_model("reply")
    agent = VektoriAgent(memory=memory, model=model, user_id="u1")

    await agent.chat("first")
    await agent.chat("second")

    msgs = agent.window.snapshot().recent_messages
    assert len(msgs) == 4  # 2 user + 2 assistant


@pytest.mark.asyncio
async def test_reset_window_clears():
    memory = _make_mock_memory()
    model = _make_mock_model()
    agent = VektoriAgent(memory=memory, model=model, user_id="u1")

    await agent.chat("hi")
    agent.reset_window()
    assert agent.window.snapshot().recent_messages == []


@pytest.mark.asyncio
async def test_add_messages_loads_window():
    memory = _make_mock_memory()
    model = _make_mock_model()
    agent = VektoriAgent(memory=memory, model=model, user_id="u1")

    prior = [
        {"role": "user", "content": "I work in ML"},
        {"role": "assistant", "content": "Got it!"},
    ]
    await agent.add_messages(prior)
    assert len(agent.window.snapshot().recent_messages) == 2


@pytest.mark.asyncio
async def test_profile_patches_applied_via_store():
    memory = _make_mock_memory()
    model = _make_mock_model()
    store = InMemoryProfileStore()
    await store.save(_make_patch(key="verbosity", value="short"))

    agent = VektoriAgent(memory=memory, model=model, user_id="u1", profile_store=store)
    await agent.chat("what can you do?")

    call_messages = model.complete.call_args[0][0]
    joined = " ".join(m["content"] for m in call_messages if m["role"] == "system")
    assert "verbosity" in joined


@pytest.mark.asyncio
async def test_profile_learning_explicit_name():
    memory = _make_mock_memory()
    model = _make_mock_model("OK, I'll call you Alex.")
    config = AgentConfig(background_add=False, enable_profile_learning=True)
    agent = VektoriAgent(memory=memory, model=model, user_id="u1", config=config)

    result = await agent.chat("Call me Alex.")
    assert any(p.key == "preferences.name" for p in result.profile_updates)


@pytest.mark.asyncio
async def test_close_cancels_background_tasks():
    memory = _make_mock_memory()
    model = _make_mock_model()
    agent = VektoriAgent(memory=memory, model=model, user_id="u1")
    await agent.close()
    assert len(agent._background_tasks) == 0


def test_agent_config_defaults():
    config = AgentConfig()
    assert config.max_context_tokens == 12000
    assert config.retrieval_depth == "l1"
    assert config.retrieval_top_k == 8
    assert config.enable_retrieval_gate is True
    assert config.background_add is True
    assert config.enable_profile_learning is True
    assert config.enable_tool_calling is False

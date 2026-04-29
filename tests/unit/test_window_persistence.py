"""Unit tests for Phase 5 window snapshot persistence."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from vektori.agent import AgentConfig, VektoriAgent
from vektori.memory.window import MessageWindow, SQLiteWindowStore, WindowState
from vektori.models.base import ChatCompletionResult


# — SQLiteWindowStore tests —

@pytest.mark.asyncio
async def test_save_and_load_round_trip():
    store = SQLiteWindowStore(":memory:")
    state = WindowState(
        recent_messages=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ],
        rolling_summary="Active goals: finish the demo",
        estimated_tokens=50,
        compaction_count=1,
    )
    await store.save("sess-1", state)
    loaded = await store.load("sess-1")

    assert loaded is not None
    assert loaded.rolling_summary == "Active goals: finish the demo"
    assert len(loaded.recent_messages) == 2
    assert loaded.compaction_count == 1
    await store.close()


@pytest.mark.asyncio
async def test_load_missing_session_returns_none():
    store = SQLiteWindowStore(":memory:")
    result = await store.load("nonexistent")
    assert result is None
    await store.close()


@pytest.mark.asyncio
async def test_save_overwrites_previous_snapshot():
    store = SQLiteWindowStore(":memory:")
    state_a = WindowState(
        recent_messages=[{"role": "user", "content": "first"}],
        rolling_summary="",
        estimated_tokens=10,
        compaction_count=0,
    )
    state_b = WindowState(
        recent_messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ],
        rolling_summary="Updated summary.",
        estimated_tokens=40,
        compaction_count=0,
    )
    await store.save("sess-1", state_a)
    await store.save("sess-1", state_b)
    loaded = await store.load("sess-1")
    assert loaded is not None
    assert len(loaded.recent_messages) == 3
    assert loaded.rolling_summary == "Updated summary."
    await store.close()


@pytest.mark.asyncio
async def test_delete_removes_snapshot():
    store = SQLiteWindowStore(":memory:")
    state = WindowState(
        recent_messages=[{"role": "user", "content": "hi"}],
        rolling_summary="",
        estimated_tokens=5,
        compaction_count=0,
    )
    await store.save("sess-1", state)
    await store.delete("sess-1")
    assert await store.load("sess-1") is None
    await store.close()


# — MessageWindow.restore —

def test_restore_from_state():
    w = MessageWindow()
    w.add("user", "will be cleared")
    state = WindowState(
        recent_messages=[
            {"role": "user", "content": "restored msg"},
        ],
        rolling_summary="restored summary",
        estimated_tokens=20,
        compaction_count=3,
    )
    w.restore(state)
    snap = w.snapshot()
    assert snap.recent_messages == [{"role": "user", "content": "restored msg"}]
    assert snap.rolling_summary == "restored summary"
    assert snap.compaction_count == 3


# — VektoriAgent.save_window / resume_window —

def _make_mock_model():
    mock = AsyncMock()
    result = MagicMock(spec=ChatCompletionResult)
    result.content = "ok"
    result.tool_calls = []
    result.usage = None
    mock.complete.return_value = result
    return mock


def _make_mock_memory():
    mock = AsyncMock()
    mock.search.return_value = {"facts": [], "sentences": [], "episodes": []}
    mock.add.return_value = {}
    return mock


@pytest.mark.asyncio
async def test_agent_save_and_resume_window():
    store = SQLiteWindowStore(":memory:")
    memory = _make_mock_memory()
    model = _make_mock_model()
    config = AgentConfig(background_add=False)

    agent = VektoriAgent(
        memory=memory,
        model=model,
        user_id="u1",
        session_id="sess-abc",
        config=config,
        window_store=store,
    )
    await agent.chat("first message")
    await agent.chat("second message")
    await agent.save_window()

    # New agent instance restores state
    agent2 = VektoriAgent(
        memory=memory,
        model=model,
        user_id="u1",
        session_id="sess-abc",
        config=config,
        window_store=store,
    )
    restored = await agent2.resume_window()
    assert restored is True
    msgs = agent2.window.snapshot().recent_messages
    assert len(msgs) == 4  # 2 user + 2 assistant turns


@pytest.mark.asyncio
async def test_agent_resume_window_returns_false_when_no_snapshot():
    store = SQLiteWindowStore(":memory:")
    memory = _make_mock_memory()
    model = _make_mock_model()
    agent = VektoriAgent(memory=memory, model=model, user_id="u1", window_store=store)

    restored = await agent.resume_window()
    assert restored is False


@pytest.mark.asyncio
async def test_agent_no_window_store_save_is_noop():
    memory = _make_mock_memory()
    model = _make_mock_model()
    config = AgentConfig(background_add=False)
    agent = VektoriAgent(memory=memory, model=model, user_id="u1", config=config)

    # No exception, just silently skips
    await agent.save_window()
    assert agent.window_store is None


@pytest.mark.asyncio
async def test_agent_window_store_via_config_path(tmp_path):
    db_path = str(tmp_path / "windows.db")
    memory = _make_mock_memory()
    model = _make_mock_model()
    config = AgentConfig(background_add=False, window_store_path=db_path)

    agent = VektoriAgent(
        memory=memory,
        model=model,
        user_id="u1",
        session_id="sess-cfg",
        config=config,
    )
    assert agent.window_store is not None
    await agent.chat("hello via config path")
    await agent.save_window()
    await agent.close()

    # Re-open and verify
    agent2 = VektoriAgent(
        memory=memory,
        model=model,
        user_id="u1",
        session_id="sess-cfg",
        config=config,
    )
    restored = await agent2.resume_window()
    assert restored is True
    assert len(agent2.window.snapshot().recent_messages) == 2
    await agent2.close()

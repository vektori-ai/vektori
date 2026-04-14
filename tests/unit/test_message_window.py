from vektori.memory.window import MessageWindow
from vektori.models.base import ChatCompletionResult, ChatModelProvider


class FakeSummarizer(ChatModelProvider):
    async def complete(
        self,
        messages,
        *,
        tools=None,
        max_tokens=None,
        temperature=None,
    ) -> ChatCompletionResult:
        return ChatCompletionResult(
            content="Conversation Summary\n- Active goals: keep going",
            tool_calls=[],
        )


async def test_message_window_compacts_when_over_threshold():
    window = MessageWindow(max_context_tokens=20, compaction_trigger_ratio=0.5, keep_last_n_turns=1)
    window.add("user", "a" * 60)
    window.add("assistant", "b" * 60)
    window.add("user", "c" * 60)
    window.add("assistant", "d" * 60)

    compacted = await window.compact(FakeSummarizer())

    snapshot = window.snapshot()
    assert compacted is True
    assert snapshot.compaction_count == 1
    assert "Conversation Summary" in snapshot.rolling_summary
    assert len(snapshot.recent_messages) == 2

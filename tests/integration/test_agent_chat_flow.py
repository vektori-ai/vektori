from unittest.mock import AsyncMock

from vektori.agent import AgentConfig, VektoriAgent
from vektori.models.base import ChatCompletionResult, ChatModelProvider


class FakeChatModel(ChatModelProvider):
    async def complete(
        self,
        messages,
        *,
        tools=None,
        max_tokens=None,
        temperature=None,
    ) -> ChatCompletionResult:
        return ChatCompletionResult(content="I remember you like short answers.", tool_calls=[])


class StubMemory:
    def __init__(self) -> None:
        self.search = AsyncMock(
            return_value={
                "facts": [{"text": "User prefers concise answers."}],
                "episodes": [],
                "sentences": [],
            }
        )
        self.add = AsyncMock(return_value={"status": "ok"})


async def test_agent_chat_runs_search_prompt_and_persistence():
    memory = StubMemory()
    agent = VektoriAgent(
        memory=memory,
        model=FakeChatModel(),
        user_id="user-1",
        agent_id="agent-1",
        config=AgentConfig(enable_retrieval_gate=False, background_add=False),
    )

    result = await agent.chat("How do I like answers?")

    assert result.content == "I remember you like short answers."
    memory.search.assert_awaited_once()
    memory.add.assert_awaited_once()
    assert result.memories_used["facts"][0]["text"] == "User prefers concise answers."


async def test_agent_chat_learns_explicit_profile_patch(tmp_path):
    memory = StubMemory()
    agent = VektoriAgent(
        memory=memory,
        model=FakeChatModel(),
        user_id="user-1",
        agent_id="agent-1",
        config=AgentConfig(
            enable_retrieval_gate=False,
            background_add=False,
            profile_store_path=str(tmp_path / "profiles.db"),
        ),
    )

    result = await agent.chat("Keep your answers short.")
    patches = await agent.profile_store.list_active("agent-1", "user-1")
    await agent.close()

    assert result.profile_updates
    assert result.profile_updates[0].key == "response_style.verbosity"
    assert patches[0].value == "short"

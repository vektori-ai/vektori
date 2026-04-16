from vektori.ingestion.extractor import FactExtractor
from vektori.models.base import EmbeddingProvider, LLMProvider
from vektori.storage.memory import MemoryBackend


class ConflictEmbedder(EmbeddingProvider):
    @property
    def dimension(self) -> int:
        return 2

    async def embed(self, text: str) -> list[float]:
        if "hates apples" in text:
            return [0.75, 0.66]
        return [1.0, 0.0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(text) for text in texts]


class SupersedingLLM(LLMProvider):
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        self.calls += 1
        start = prompt.index("- [") + len("- [")
        end = prompt.index("]", start)
        old_id = prompt[start:end]
        return f'{{"supersedes_id": "{old_id}"}}'


async def test_llm_contradiction_check_supersedes_existing_fact():
    db = MemoryBackend()
    await db.initialize()
    llm = SupersedingLLM()
    extractor = FactExtractor(db, ConflictEmbedder(), llm)

    await extractor._process_facts(
        [{"text": "User loves apples"}],
        session_id="s1",
        user_id="u1",
        agent_id=None,
        conversation="User: I love apples",
    )
    await extractor._process_facts(
        [{"text": "User hates apples"}],
        session_id="s2",
        user_id="u1",
        agent_id=None,
        conversation="User: I hate apples",
    )

    active = await db.get_active_facts("u1")
    inactive = [fact for fact in db._facts.values() if not fact["is_active"]]

    assert llm.calls == 1
    assert [fact["text"] for fact in active] == ["User hates apples"]
    assert [fact["text"] for fact in inactive] == ["User loves apples"]
    assert inactive[0]["superseded_by"] == active[0]["id"]

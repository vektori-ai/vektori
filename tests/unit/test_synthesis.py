from vektori.ingestion.synthesis import Synthesizer
from vektori.models.base import EmbeddingProvider, LLMProvider
from vektori.storage.memory import MemoryBackend


class MockEmbedder(EmbeddingProvider):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def dimension(self) -> int:
        return 2

    async def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[1.0, 0.0] for _ in texts]


class MockLLM(LLMProvider):
    def __init__(self, response: str) -> None:
        self.response = response

    async def generate(self, prompt: str, max_tokens: int | None = None) -> str:
        return self.response


async def _db_with_base_facts() -> MemoryBackend:
    db = MemoryBackend()
    await db.initialize()
    for index in range(5):
        await db.insert_fact(
            text=f"User ate fruit in session {index}",
            embedding=[1.0, 0.0],
            user_id="u1",
            session_id=f"s{index}",
        )
    return db


async def test_synthesizer_links_new_synthesis_to_source_facts():
    db = await _db_with_base_facts()
    embedder = MockEmbedder()
    llm = MockLLM(
        """
        {
          "facts": [
            {"text": ""},
            "bad item",
            {"text": "User regularly eats fruit.", "confidence": 0.9}
          ]
        }
        """
    )
    synthesizer = Synthesizer(db, embedder, llm)

    inserted = await synthesizer.synthesize("u1")

    assert inserted == 1
    facts = await db.get_active_facts("u1", limit=100)
    syntheses = await db.get_syntheses_for_facts([fact["id"] for fact in facts])
    assert [s["text"] for s in syntheses] == ["User regularly eats fruit."]


async def test_synthesizer_skips_malformed_model_items_without_embedding():
    db = await _db_with_base_facts()
    embedder = MockEmbedder()
    llm = MockLLM('{"facts": [{"confidence": 0.9}, "", {"text": "   "}]}')
    synthesizer = Synthesizer(db, embedder, llm)

    inserted = await synthesizer.synthesize("u1")

    assert inserted == 0
    assert embedder.calls == 0

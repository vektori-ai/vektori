import asyncio

from vektori.ingestion.extractor import FactExtractor
from vektori.models.base import EmbeddingProvider, LLMProvider
from vektori.storage.memory import MemoryBackend


class MockEmbedder(EmbeddingProvider):
    @property
    def dimension(self):
        return 2

    async def embed(self, text):
        if "hate" in text:
            return [0.5, 0.4]  # Give sim around 0.8 to test LLM
        return [0.5, 0.5]
    async def embed_batch(self, texts):
        return [await self.embed(t) for t in texts]

class MockLLM(LLMProvider):
    async def generate(self, prompt, max_tokens=1000, **kwargs):
        if "contradict" in prompt.lower():
            # If it has "hate", it contradicts the "love" one
            if "hates apples" in prompt and "loves apples" in prompt:
                return '{"contradicts": "1"}'
            return '{"contradicts": null}'
        pass

async def test():
    db = MemoryBackend()
    await db.initialize()
    embedder = MockEmbedder()
    llm = MockLLM()
    _ = FactExtractor(db, embedder, llm)

    # Needs actual logic in Extractor...

asyncio.run(test())

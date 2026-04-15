import asyncio
import logging

from vektori.ingestion.extractor import FactExtractor
from vektori.models.base import EmbeddingProvider, LLMProvider
from vektori.storage.memory import MemoryBackend

logging.basicConfig(level=logging.DEBUG)

class MockEmbedder(EmbeddingProvider):
    @property
    def dimension(self):
        return 2

    async def embed(self, text):
        if "hate" in text:
            return [0.5, 0.4]  # High similarity to trigger check
        return [0.5, 0.5]
    async def embed_batch(self, texts):
        return [await self.embed(t) for t in texts]

class MockLLM(LLMProvider):
    async def generate(self, prompt, max_tokens=1000, **kwargs):
        if "loves apples" in prompt and "hates apples" in prompt:
            old_id = prompt.split("- [")[1].split("]")[0]
            return f'{{"supersedes_id": "{old_id}"}}'
        return "{}"

async def test():
    db = MemoryBackend()
    await db.initialize()
    embedder = MockEmbedder()
    llm = MockLLM()
    extractor = FactExtractor(db, embedder, llm)

    await extractor._process_facts([{"text": "User loves apples"}], "s1", "u1", None, "")
    await extractor._process_facts([{"text": "User hates apples"}], "s2", "u1", None, "")

    facts = await db.get_active_facts("u1")
    print("Active facts count:", len(facts))
    for f in facts:
        print("-", f["text"])

asyncio.run(test())

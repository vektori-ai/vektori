import asyncio
from vektori.storage.memory import MemoryBackend
from vektori.ingestion.extractor import FactExtractor
from vektori.models.base import LLMProvider, EmbeddingProvider

class MockEmbedder(EmbeddingProvider):
    @property
    def dimension(self): return 2
    async def embed(self, text):
        if "hate" in text: return [0.5, -0.5]
        return [0.5, 0.5]
    async def embed_batch(self, texts):
        return [await self.embed(t) for t in texts]

class MockLLM(LLMProvider):
    async def generate(self, prompt, max_tokens=1000, **kwargs):
        pass

async def test():
    db = MemoryBackend()
    await db.initialize()
    embedder = MockEmbedder()
    llm = MockLLM()
    extractor = FactExtractor(db, embedder, llm)
    
    # insert first fact
    await extractor._process_facts(
        [{"text": "User loves apples"}],
        session_id="s1", user_id="u1", agent_id=None, conversation="User: I love apples"
    )
    
    # insert second fact
    await extractor._process_facts(
        [{"text": "User hates apples"}],
        session_id="s2", user_id="u1", agent_id=None, conversation="User: Actually I hate apples"
    )
    
    facts = await db.get_active_facts("u1")
    print("Active facts count:", len(facts))
    for f in facts:
        print(f["text"])

asyncio.run(test())

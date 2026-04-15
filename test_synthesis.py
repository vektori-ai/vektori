import asyncio

from vektori import Vektori


async def test():
    v = Vektori(storage_backend="memory")
    await v._ensure_initialized()
    # Mock LLM to just return a synthesis fact
    async def mock_gen(*args, **kwargs):
        return '{"facts": [{"text": "User loves fruit overall."}]}'
    async def mock_emb(*args, **kwargs):
        return [[0.1, 0.2]]
    v.llm.generate = mock_gen
    v.embedder.embed_batch = mock_emb

    # add dummy facts so synthesize triggers (needs >=5 base facts)
    for i in range(5):
        await v.db.insert_fact(f"fact {i}", [0.1, 0.2], "u1", confidence=1.0)

    n = await v.synthesize("u1")
    print("New synthesized facts:", n)
    syntheses = await v.db.search_syntheses([0.1, 0.2], "u1", limit=100)
    for s in syntheses:
        print("Synthesized:", s["text"])

asyncio.run(test())

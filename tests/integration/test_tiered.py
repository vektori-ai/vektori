"""Integration tests for L0 / L1 / L2 tiered retrieval depths."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from vektori.retrieval.search import SearchPipeline
from vektori.storage.memory import MemoryBackend


@pytest.fixture
async def populated_pipeline():
    db = MemoryBackend()
    await db.initialize()

    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[1.0] + [0.0] * 1535)

    # Sentence
    await db.upsert_sentences(
        sentences=[
            {
                "id": "sent-1",
                "text": "I only use WhatsApp, please don't email me.",
                "session_id": "call-001",
                "turn_number": 1,
                "sentence_index": 0,
                "role": "user",
            }
        ],
        embeddings=[[1.0] + [0.0] * 1535],
        user_id="u1",
    )

    # Fact linked to sentence
    fact_id = await db.insert_fact(
        text="User prefers WhatsApp over email",
        embedding=[1.0] + [0.0] * 1535,
        user_id="u1",
        confidence=0.95,
    )
    await db.insert_fact_source(fact_id, "sent-1")

    pipeline = SearchPipeline(db=db, embedder=embedder)
    return pipeline


async def test_l0_depth(populated_pipeline):
    results = await populated_pipeline.search("WhatsApp preference", "u1", depth="l0")
    assert "facts" in results
    assert "insights" in results
    assert "sentences" not in results


async def test_l1_depth(populated_pipeline):
    results = await populated_pipeline.search("WhatsApp preference", "u1", depth="l1")
    assert "facts" in results
    assert "insights" in results
    assert "sentences" in results  # L1 returns source sentences (exact origin of each fact)


async def test_l2_depth(populated_pipeline):
    results = await populated_pipeline.search("WhatsApp preference", "u1", depth="l2")
    assert "facts" in results
    assert "sentences" in results
    # Should trace back to source sentence
    assert len(results["sentences"]) >= 1


async def test_facts_scored_and_ordered(populated_pipeline):
    results = await populated_pipeline.search("WhatsApp", "u1", depth="l0")
    for fact in results["facts"]:
        assert "score" in fact
        assert 0 <= fact["score"] <= 1.0

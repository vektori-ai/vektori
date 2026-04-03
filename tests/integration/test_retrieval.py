"""Integration tests for the retrieval pipeline (memory backend)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from vektori.retrieval.search import SearchPipeline
from vektori.storage.memory import MemoryBackend


@pytest.fixture
async def search_with_facts():
    """SearchPipeline with pre-populated facts in memory backend."""
    db = MemoryBackend()
    await db.initialize()

    embedder = MagicMock()
    # Query embedding slightly different from stored to test cosine similarity
    embedder.embed = AsyncMock(return_value=[1.0] + [0.0] * 1535)

    # Insert a fact
    await db.insert_fact(
        text="User prefers WhatsApp over email",
        embedding=[1.0] + [0.0] * 1535,
        user_id="u1",
        confidence=0.95,
    )

    pipeline = SearchPipeline(db=db, embedder=embedder)
    return pipeline, db


async def test_l0_returns_facts_and_insights(search_with_facts):
    pipeline, db = search_with_facts
    results = await pipeline.search("communication preference", "u1", depth="l0")
    assert "facts" in results
    assert "insights" in results
    assert isinstance(results["insights"], list)
    assert len(results["facts"]) >= 1


async def test_l1_returns_facts_insights_and_sentences(search_with_facts):
    pipeline, db = search_with_facts
    results = await pipeline.search("communication preference", "u1", depth="l1")
    assert "facts" in results
    assert "insights" in results
    assert isinstance(results["insights"], list)
    assert len(results["facts"]) >= 1


async def test_l2_returns_facts_and_sentences(search_with_facts):
    pipeline, db = search_with_facts
    results = await pipeline.search("communication preference", "u1", depth="l2")
    assert "facts" in results
    assert "sentences" in results


async def test_empty_result_for_unknown_user(search_with_facts):
    pipeline, db = search_with_facts
    results = await pipeline.search("anything", "unknown-user", depth="l1")
    assert results["facts"] == []

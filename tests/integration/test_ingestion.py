"""Integration tests for the full ingestion pipeline (memory backend)."""

from unittest.mock import AsyncMock, MagicMock

from vektori import Vektori
from vektori.ingestion.extractor import FactExtractor
from vektori.ingestion.pipeline import IngestionPipeline
from vektori.retrieval.search import SearchPipeline
from vektori.storage.memory import MemoryBackend


def _mock_vektori() -> Vektori:
    """Build a Vektori instance with memory backend and mocked embedder/LLM."""
    v = Vektori(storage_backend="memory", async_extraction=False)
    v.embedder = MagicMock()
    v.embedder.embed = AsyncMock(return_value=[0.1] * 1536)
    v.embedder.embed_batch = AsyncMock(side_effect=lambda texts: [[0.1] * 1536] * len(texts))
    v.llm = MagicMock()
    v.llm.generate = AsyncMock(return_value='{"facts": [], "insights": []}')
    v.db = MemoryBackend()
    v._extractor = FactExtractor(db=v.db, embedder=v.embedder, llm=v.llm)
    v._search = SearchPipeline(db=v.db, embedder=v.embedder)
    v._pipeline = IngestionPipeline(
        db=v.db,
        embedder=v.embedder,
        extractor=v._extractor,
        async_extraction=False,
    )
    v._initialized = True
    return v


async def test_add_returns_ok():
    v = _mock_vektori()
    await v.db.initialize()
    result = await v.add(
        messages=[{"role": "user", "content": "I prefer WhatsApp for all communications."}],
        session_id="test-s1",
        user_id="test-u1",
    )
    assert result["status"] == "ok"
    assert result["sentences_stored"] >= 1


async def test_add_filters_junk():
    v = _mock_vektori()
    await v.db.initialize()
    result = await v.add(
        messages=[{"role": "user", "content": "ok"}],
        session_id="test-s2",
        user_id="test-u1",
    )
    assert result["sentences_stored"] == 0


async def test_add_only_stores_user_sentences():
    v = _mock_vektori()
    await v.db.initialize()
    result = await v.add(
        messages=[
            {"role": "user", "content": "I prefer WhatsApp for all communications."},
            {"role": "assistant", "content": "Understood, I will use WhatsApp."},
        ],
        session_id="test-s3",
        user_id="test-u2",
    )
    # Only user message gets stored as a sentence node
    assert result["sentences_stored"] == 1


async def test_add_creates_next_edges():
    v = _mock_vektori()
    await v.db.initialize()
    await v.add(
        messages=[
            {"role": "user", "content": "I prefer WhatsApp for all my business communications."},
            {
                "role": "user",
                "content": "My outstanding loan balance is forty five thousand rupees.",
            },
        ],
        session_id="test-s4",
        user_id="test-u3",
    )
    assert len(v.db._edges) >= 1


async def test_add_deduplicates_same_session():
    v = _mock_vektori()
    await v.db.initialize()
    await v.add(
        messages=[{"role": "user", "content": "I prefer WhatsApp for all communications."}],
        session_id="same-session",
        user_id="test-u4",
    )
    await v.add(
        messages=[{"role": "user", "content": "I prefer WhatsApp for all communications."}],
        session_id="same-session",
        user_id="test-u4",
    )
    # Second add should increment mentions, not create duplicate
    sentences = list(v.db._sentences.values())
    assert len(sentences) == 1
    assert sentences[0]["mentions"] == 2

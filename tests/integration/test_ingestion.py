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
    v.llm.generate = AsyncMock(return_value='{"facts": [], "episodes": []}')
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
    assert result["sentences_stored"] == 1
    sentences = list(v.db._sentences.values())
    assert sentences[0]["is_searchable"] is False


async def test_add_stores_user_and_assistant_sentences():
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
    # Both user and assistant sentences are stored (assistant needed for fact source-linking)
    assert result["sentences_stored"] >= 1


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


async def test_low_quality_sentences_are_stored_but_not_searchable():
    v = _mock_vektori()
    await v.db.initialize()
    await v.add(
        messages=[{"role": "user", "content": "Yes."}],
        session_id="test-s5",
        user_id="test-u5",
    )

    assert len(v.db._sentences) == 1
    stored = next(iter(v.db._sentences.values()))
    assert stored["is_searchable"] is False

    results = await v.db.search_sentences([0.1] * 1536, user_id="test-u5")
    assert results == []


async def test_explicit_sentence_indices_link_fact_sources():
    v = _mock_vektori()
    await v.db.initialize()
    v.llm.generate = AsyncMock(
        side_effect=[
            """{"facts":[{"text":"Caroline prefers WhatsApp for business communication.","source":"user","subject":"Caroline","source_sentence_indices":[0]}]}""",
            """{"episodes":[{"text":"The user stated a communication preference.","fact_indices":[0]}]}""",
        ]
    )

    await v.add(
        messages=[{"role": "user", "content": "I prefer WhatsApp for business communication."}],
        session_id="test-s6",
        user_id="test-u6",
    )

    assert len(v.db._fact_sources) == 1
    linked_sentence_id = v.db._fact_sources[0]["sentence_id"]
    linked_sentence = v.db._sentences[linked_sentence_id]
    assert linked_sentence["text"] == "I prefer WhatsApp for business communication."


async def test_confirmation_fact_can_link_multiple_sentences():
    v = _mock_vektori()
    await v.db.initialize()
    v.llm.generate = AsyncMock(
        side_effect=[
            """{"facts":[{"text":"Caroline prefers WhatsApp for business communication.","source":"user","subject":"Caroline","source_sentence_indices":[0,1]}]}""",
            """{"episodes":[{"text":"The user confirmed a communication preference.","fact_indices":[0]}]}""",
        ]
    )

    await v.add(
        messages=[
            {"role": "assistant", "content": "You prefer WhatsApp for business communication."},
            {"role": "user", "content": "Yes."},
        ],
        session_id="test-s7",
        user_id="test-u7",
    )

    linked_ids = [row["sentence_id"] for row in v.db._fact_sources]
    linked_texts = [v.db._sentences[sid]["text"] for sid in linked_ids]
    assert linked_texts == [
        "You prefer WhatsApp for business communication.",
        "Yes.",
    ]


async def test_memory_backend_deduplicates_next_edges_on_reingest():
    v = _mock_vektori()
    await v.db.initialize()
    messages = [
        {"role": "user", "content": "I prefer WhatsApp for business communication."},
        {"role": "user", "content": "My loan balance is forty five thousand rupees."},
    ]

    await v.add(messages=messages, session_id="same-edges", user_id="test-u8")
    await v.add(messages=messages, session_id="same-edges", user_id="test-u8")

    assert len(v.db._edges) == 1

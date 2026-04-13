"""End-to-end Vektori pipeline tests against a live Milvus backend.

These tests use mocked embedder/LLM providers to validate ingestion + retrieval
behavior without external model/API dependencies.
"""

from __future__ import annotations

import asyncio
import socket
from unittest.mock import AsyncMock, MagicMock

import pytest

from vektori import Vektori
from vektori.ingestion.extractor import FactExtractor
from vektori.ingestion.pipeline import IngestionPipeline
from vektori.retrieval.search import SearchPipeline

MILVUS_URL = "http://localhost:19530"
TEST_PREFIX = "vektori_test_e2e"


def _is_milvus_available() -> bool:
    try:
        import pymilvus  # noqa: F401

        return True
    except ImportError:
        return False


async def _can_connect() -> bool:
    if not _is_milvus_available():
        return False

    def _port_open() -> bool:
        try:
            with socket.create_connection(("localhost", 19530), timeout=1.0):
                return True
        except OSError:
            return False

    if not _port_open():
        return False

    try:
        from pymilvus import MilvusClient

        def _probe() -> None:
            client = MilvusClient(uri=MILVUS_URL)
            client.list_collections()
            close_fn = getattr(client, "close", None)
            if close_fn:
                close_fn()

        await asyncio.wait_for(asyncio.to_thread(_probe), timeout=5)
        return True
    except Exception:
        return False


def _mock_vektori_with_backend(db_backend) -> Vektori:
    """Build a Vektori instance bound to a real backend and mocked model providers."""
    v = Vektori(storage_backend="memory", async_extraction=False)

    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4])
    embedder.embed_batch = AsyncMock(side_effect=lambda texts: [[0.1, 0.2, 0.3, 0.4]] * len(texts))

    async def _fake_generate(prompt: str, max_tokens: int = 0) -> str:
        if "episodes" in prompt.lower():
            return (
                '{"episodes":[{"text":"User preferred WhatsApp communication.",'
                '"fact_indices":[0]}]}'
            )
        return (
            '{"facts":[{"text":"User prefers WhatsApp communication",'
            '"source":"user","subject":"user","confidence":0.95,'
            '"source_quotes":["I prefer WhatsApp communication"]}]}'
        )

    llm = MagicMock()
    llm.generate = AsyncMock(side_effect=_fake_generate)

    v.embedder = embedder
    v.llm = llm
    v.db = db_backend
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


@pytest.fixture(scope="module")
async def milvus_backend():
    if not await _can_connect():
        pytest.skip("Milvus not reachable at http://localhost:19530")

    from vektori.storage.milvus import MilvusBackend

    backend = MilvusBackend(url=MILVUS_URL, prefix=TEST_PREFIX, embedding_dim=4)
    await backend.initialize()

    yield backend

    await backend.delete_user("e2e-user")
    await backend.delete_user("e2e-delete-user")
    await backend.close()


async def test_milvus_e2e_add_and_search_depths(milvus_backend):
    v = _mock_vektori_with_backend(milvus_backend)

    result = await v.add(
        messages=[
            {
                "role": "user",
                "content": "I prefer WhatsApp communication and avoid email.",
            }
        ],
        session_id="e2e-session-1",
        user_id="e2e-user",
    )
    assert result["status"] == "ok"
    assert result["sentences_stored"] >= 1

    l0 = await v.search("How should I contact this user?", "e2e-user", depth="l0")
    assert "facts" in l0

    l1 = await v.search("How should I contact this user?", "e2e-user", depth="l1")
    assert "facts" in l1
    assert "sentences" in l1

    l2 = await v.search("How should I contact this user?", "e2e-user", depth="l2")
    assert "facts" in l2
    assert "sentences" in l2


async def test_milvus_e2e_session_and_delete_user(milvus_backend):
    v = _mock_vektori_with_backend(milvus_backend)

    await v.add(
        messages=[{"role": "user", "content": "I prefer WhatsApp communication."}],
        session_id="e2e-session-2",
        user_id="e2e-delete-user",
    )

    session = await v.get_session("e2e-session-2", "e2e-delete-user")
    assert session is not None
    assert session["id"] == "e2e-session-2"

    deleted = await v.delete_user("e2e-delete-user")
    assert deleted >= 1

    after = await v.search("WhatsApp", "e2e-delete-user", depth="l1")
    assert after["facts"] == []

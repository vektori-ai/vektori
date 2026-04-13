"""Integration tests for MilvusBackend — requires a live Milvus instance.

Start with:
    docker compose up -d milvus

Tests are skipped automatically when the server is unreachable.
"""

from __future__ import annotations

import asyncio
import socket

import pytest

MILVUS_URL = "http://localhost:19530"
TEST_PREFIX = "vektori_test"


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


@pytest.fixture(scope="module")
async def milvus_backend():
    if not await _can_connect():
        pytest.skip("Milvus not reachable at http://localhost:19530")

    from vektori.storage.milvus import MilvusBackend

    backend = MilvusBackend(url=MILVUS_URL, prefix=TEST_PREFIX, embedding_dim=4)
    await backend.initialize()

    yield backend

    await backend.delete_user("test-user")
    await backend.delete_user("delete-test-user")
    await backend.close()


async def test_upsert_and_search_sentences(milvus_backend):
    emb = [0.1, 0.2, 0.3, 0.4]
    sentences = [
        {
            "id": "ms1",
            "text": "I love hiking in the mountains.",
            "session_id": "msess1",
            "turn_number": 0,
            "sentence_index": 0,
            "role": "user",
        }
    ]
    count = await milvus_backend.upsert_sentences(sentences, [emb], user_id="test-user")
    assert count == 1

    results = await milvus_backend.search_sentences(emb, user_id="test-user", limit=5)
    assert len(results) >= 1
    assert results[0]["text"] == "I love hiking in the mountains."
    assert "distance" in results[0]


async def test_insert_and_search_facts(milvus_backend):
    emb = [0.5, 0.5, 0.5, 0.5]
    fid = await milvus_backend.insert_fact(
        text="User prefers dark mode.",
        embedding=emb,
        user_id="test-user",
        subject="user",
    )
    assert fid

    results = await milvus_backend.search_facts(emb, user_id="test-user", limit=10)
    assert any(r["id"] == fid for r in results)


async def test_fact_source_linking(milvus_backend):
    emb = [0.4, 0.4, 0.4, 0.4]
    await milvus_backend.upsert_sentences(
        [
            {
                "id": "ms-src1",
                "text": "Source sentence for fact.",
                "session_id": "msess-src",
                "turn_number": 0,
                "sentence_index": 0,
                "role": "user",
            }
        ],
        [emb],
        user_id="test-user",
    )
    fid = await milvus_backend.insert_fact(
        text="Fact derived from source sentence.", embedding=emb, user_id="test-user"
    )
    await milvus_backend.insert_fact_source(fid, "ms-src1")

    source_ids = await milvus_backend.get_source_sentences([fid])
    assert "ms-src1" in source_ids


async def test_sessions_roundtrip(milvus_backend):
    await milvus_backend.upsert_session(
        session_id="msess-get",
        user_id="test-user",
        metadata={"source": "integration"},
    )
    session = await milvus_backend.get_session("msess-get", "test-user")
    assert session is not None
    assert session["id"] == "msess-get"
    assert "sentences" in session

    count = await milvus_backend.count_sessions("test-user")
    assert count >= 1


async def test_episodes_flow(milvus_backend):
    emb = [0.6, 0.6, 0.6, 0.6]
    fid = await milvus_backend.insert_fact(
        text="Episode linkage fact.", embedding=emb, user_id="test-user"
    )
    eid = await milvus_backend.insert_episode(
        text="Episode linked to fact.",
        embedding=emb,
        user_id="test-user",
    )
    await milvus_backend.insert_episode_fact(eid, fid)

    episodes = await milvus_backend.get_episodes_for_facts([fid])
    assert any(ep["id"] == eid for ep in episodes)

    searched = await milvus_backend.search_episodes(emb, user_id="test-user", limit=5)
    assert any(ep["id"] == eid for ep in searched)


async def test_delete_user(milvus_backend):
    emb = [0.8, 0.8, 0.8, 0.8]
    await milvus_backend.upsert_sentences(
        [
            {
                "id": "ms-del-s1",
                "text": "To be deleted.",
                "session_id": "msess-del",
                "turn_number": 0,
                "sentence_index": 0,
                "role": "user",
            }
        ],
        [emb],
        user_id="delete-test-user",
    )
    await milvus_backend.insert_fact(
        text="Fact to delete.", embedding=emb, user_id="delete-test-user"
    )

    deleted = await milvus_backend.delete_user("delete-test-user")
    assert deleted >= 2

    remaining = await milvus_backend.search_sentences(emb, user_id="delete-test-user", limit=5)
    assert remaining == []

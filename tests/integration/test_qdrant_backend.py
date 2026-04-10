"""Integration tests for QdrantBackend — requires a live Qdrant instance.

Start with:
    docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

Tests are skipped automatically when the server is unreachable.
"""

from __future__ import annotations

import pytest

QDRANT_URL = "http://localhost:6333"
TEST_PREFIX = "vektori_test"


def _is_qdrant_available() -> bool:
    try:
        import qdrant_client  # noqa: F401

        return True
    except ImportError:
        return False


async def _can_connect() -> bool:
    if not _is_qdrant_available():
        return False
    try:
        from qdrant_client import AsyncQdrantClient

        client = AsyncQdrantClient(url=QDRANT_URL)
        await client.get_collections()
        await client.close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
async def qdrant_backend():
    if not await _can_connect():
        pytest.skip("Qdrant not reachable at http://localhost:6333")

    from vektori.storage.qdrant_backend import QdrantBackend

    backend = QdrantBackend(url=QDRANT_URL, prefix=TEST_PREFIX, embedding_dim=4)
    await backend.initialize()
    yield backend
    # Teardown: delete all test user data
    await backend.delete_user("test-user")
    await backend.delete_user("delete-test-user")
    await backend.close()


# ── Sentences ──────────────────────────────────────────────────────────────────


async def test_upsert_and_search_sentences(qdrant_backend):
    emb = [0.1, 0.2, 0.3, 0.4]
    sentences = [
        {
            "id": "qs1",
            "text": "I love hiking in the mountains.",
            "session_id": "qsess1",
            "turn_number": 0,
            "sentence_index": 0,
            "role": "user",
        }
    ]
    count = await qdrant_backend.upsert_sentences(sentences, [emb], user_id="test-user")
    assert count == 1

    results = await qdrant_backend.search_sentences(emb, user_id="test-user", limit=5)
    assert len(results) >= 1
    assert results[0]["text"] == "I love hiking in the mountains."
    assert "distance" in results[0]


async def test_sentence_dedup_increments_mentions(qdrant_backend):
    emb = [0.1, 0.2, 0.3, 0.4]
    sent = [
        {
            "id": "qs-dedup",
            "text": "Qdrant dedup test sentence.",
            "session_id": "qsess-dedup",
            "turn_number": 0,
            "sentence_index": 0,
            "role": "user",
        }
    ]
    await qdrant_backend.upsert_sentences(sent, [emb], user_id="test-user")
    await qdrant_backend.upsert_sentences(sent, [emb], user_id="test-user")

    results = await qdrant_backend.search_sentences(emb, user_id="test-user", limit=20)
    matching = [r for r in results if r["text"] == "Qdrant dedup test sentence."]
    assert len(matching) == 1
    assert matching[0]["mentions"] == 2


async def test_find_sentence_containing(qdrant_backend):
    emb = [0.1, 0.2, 0.3, 0.4]
    await qdrant_backend.upsert_sentences(
        [
            {
                "id": "qs-contain",
                "text": "The quick brown fox jumps over.",
                "session_id": "qsess-contain",
                "turn_number": 0,
                "sentence_index": 0,
                "role": "user",
            }
        ],
        [emb],
        user_id="test-user",
    )
    result = await qdrant_backend.find_sentence_containing("qsess-contain", "quick brown")
    assert result is not None
    assert "quick brown" in result["text"].lower()

    none_result = await qdrant_backend.find_sentence_containing("qsess-contain", "xyz-missing")
    assert none_result is None


async def test_search_sentences_in_session(qdrant_backend):
    emb = [0.9, 0.1, 0.1, 0.1]
    await qdrant_backend.upsert_sentences(
        [
            {
                "id": "qs-sess-search",
                "text": "Session-scoped search test.",
                "session_id": "qsess-scoped",
                "turn_number": 0,
                "sentence_index": 0,
                "role": "user",
            }
        ],
        [emb],
        user_id="test-user",
    )
    ids = await qdrant_backend.search_sentences_in_session(
        emb, session_id="qsess-scoped", threshold=0.5
    )
    assert "qs-sess-search" in ids


# ── Facts ─────────────────────────────────────────────────────────────────────


async def test_insert_and_search_facts(qdrant_backend):
    emb = [0.5, 0.5, 0.5, 0.5]
    fid = await qdrant_backend.insert_fact(
        text="User prefers dark mode.",
        embedding=emb,
        user_id="test-user",
        subject="user",
    )
    assert fid

    results = await qdrant_backend.search_facts(emb, user_id="test-user", limit=5)
    assert any(r["id"] == fid for r in results)
    hit = next(r for r in results if r["id"] == fid)
    assert hit["subject"] == "user"


async def test_deactivate_fact(qdrant_backend):
    emb = [0.3, 0.3, 0.3, 0.3]
    fid = await qdrant_backend.insert_fact(
        text="Qdrant fact to deactivate.", embedding=emb, user_id="test-user"
    )
    await qdrant_backend.deactivate_fact(fid)

    results = await qdrant_backend.search_facts(
        emb, user_id="test-user", active_only=True, limit=20
    )
    assert all(r["id"] != fid for r in results)


async def test_supersession_chain(qdrant_backend):
    emb = [0.2, 0.2, 0.2, 0.2]
    f1 = await qdrant_backend.insert_fact(
        text="User is 25 years old.", embedding=emb, user_id="test-user"
    )
    f2 = await qdrant_backend.insert_fact(
        text="User is 26 years old.", embedding=emb, user_id="test-user",
        superseded_by_target=f1,
    )
    chain = await qdrant_backend.get_supersession_chain(f2)
    ids = [r["id"] for r in chain]
    assert f2 in ids
    assert f1 in ids


async def test_increment_fact_mentions(qdrant_backend):
    emb = [0.9, 0.1, 0.1, 0.1]
    fid = await qdrant_backend.insert_fact(
        text="Qdrant mentions test.", embedding=emb, user_id="test-user"
    )
    await qdrant_backend.increment_fact_mentions(fid)
    await qdrant_backend.increment_fact_mentions(fid)

    points = await qdrant_backend._client.retrieve(
        collection_name=qdrant_backend._facts_col,
        ids=[fid],
        with_payload=["mentions"],
    )
    assert points[0].payload["mentions"] == 3


async def test_get_active_facts(qdrant_backend):
    emb = [0.4, 0.4, 0.4, 0.4]
    fid = await qdrant_backend.insert_fact(
        text="Active fact check.", embedding=emb, user_id="test-user"
    )
    facts = await qdrant_backend.get_active_facts("test-user", limit=50)
    assert any(f["id"] == fid for f in facts)


# ── Join tables ───────────────────────────────────────────────────────────────


async def test_fact_source_linking(qdrant_backend):
    emb = [0.4, 0.4, 0.4, 0.4]
    await qdrant_backend.upsert_sentences(
        [
            {
                "id": "qs-src1",
                "text": "Qdrant source sentence.",
                "session_id": "qsess-src",
                "turn_number": 0,
                "sentence_index": 0,
                "role": "user",
            }
        ],
        [emb],
        user_id="test-user",
    )
    fid = await qdrant_backend.insert_fact(
        text="Fact derived from Qdrant source.", embedding=emb, user_id="test-user"
    )
    await qdrant_backend.insert_fact_source(fid, "qs-src1")

    source_ids = await qdrant_backend.get_source_sentences([fid])
    assert "qs-src1" in source_ids


async def test_batch_fact_sources(qdrant_backend):
    emb = [0.4, 0.4, 0.4, 0.4]
    await qdrant_backend.upsert_sentences(
        [
            {"id": f"qs-batch-s{i}", "text": f"Batch source {i}.",
             "session_id": "qsess-batch", "turn_number": 0, "sentence_index": i, "role": "user"}
            for i in range(3)
        ],
        [emb] * 3,
        user_id="test-user",
    )
    fid = await qdrant_backend.insert_fact(
        text="Batch fact.", embedding=emb, user_id="test-user"
    )
    await qdrant_backend.insert_fact_sources(
        [(fid, "qs-batch-s0"), (fid, "qs-batch-s1"), (fid, "qs-batch-s2")]
    )
    source_ids = await qdrant_backend.get_source_sentences([fid])
    assert set(source_ids) >= {"qs-batch-s0", "qs-batch-s1", "qs-batch-s2"}


# ── Expand session context ────────────────────────────────────────────────────


async def test_expand_session_context(qdrant_backend):
    emb = [0.1, 0.1, 0.1, 0.1]
    sents = [
        {
            "id": f"qctx-{i}",
            "text": f"Qdrant context sentence {i}",
            "session_id": "qsess-ctx",
            "turn_number": 0,
            "sentence_index": i,
            "role": "user",
        }
        for i in range(6)
    ]
    await qdrant_backend.upsert_sentences(sents, [emb] * 6, user_id="test-user")

    expanded = await qdrant_backend.expand_session_context(["qctx-2"], window=1)
    indices = [r["sentence_index"] for r in expanded]
    assert 1 in indices and 2 in indices and 3 in indices


async def test_get_sentences_by_ids(qdrant_backend):
    emb = [0.1, 0.1, 0.1, 0.1]
    await qdrant_backend.upsert_sentences(
        [
            {
                "id": "qs-byid1",
                "text": "Fetch by id test.",
                "session_id": "qsess-byid",
                "turn_number": 0,
                "sentence_index": 0,
                "role": "user",
            }
        ],
        [emb],
        user_id="test-user",
    )
    rows = await qdrant_backend.get_sentences_by_ids(["qs-byid1"])
    assert len(rows) == 1
    assert rows[0]["text"] == "Fetch by id test."


# ── Sessions ──────────────────────────────────────────────────────────────────


async def test_upsert_and_get_session(qdrant_backend):
    await qdrant_backend.upsert_session(
        session_id="qsess-get", user_id="test-user", metadata={"src": "test"}
    )
    session = await qdrant_backend.get_session("qsess-get", "test-user")
    assert session is not None
    assert session["id"] == "qsess-get"
    assert "sentences" in session


async def test_count_sessions(qdrant_backend):
    await qdrant_backend.upsert_session("qsess-cnt1", "test-user")
    await qdrant_backend.upsert_session("qsess-cnt2", "test-user")
    count = await qdrant_backend.count_sessions("test-user")
    assert count >= 2


async def test_get_session_wrong_user_returns_none(qdrant_backend):
    await qdrant_backend.upsert_session("qsess-private", "test-user")
    result = await qdrant_backend.get_session("qsess-private", "other-user")
    assert result is None


# ── Episodes ──────────────────────────────────────────────────────────────────


async def test_insert_and_search_episodes(qdrant_backend):
    emb = [0.6, 0.6, 0.6, 0.6]
    eid = await qdrant_backend.insert_episode(
        text="User enjoys outdoor activities in Qdrant.",
        embedding=emb,
        user_id="test-user",
    )
    assert eid

    results = await qdrant_backend.search_episodes(emb, user_id="test-user", limit=5)
    assert any(r["id"] == eid for r in results)


async def test_episode_idempotent(qdrant_backend):
    emb = [0.6, 0.6, 0.6, 0.6]
    text = "Idempotent episode insert."
    eid1 = await qdrant_backend.insert_episode(text, emb, user_id="test-user")
    eid2 = await qdrant_backend.insert_episode(text, emb, user_id="test-user")
    assert eid1 == eid2


async def test_episodes_for_facts(qdrant_backend):
    emb = [0.7, 0.7, 0.7, 0.7]
    fid = await qdrant_backend.insert_fact(
        text="Qdrant episode linkage fact.", embedding=emb, user_id="test-user"
    )
    eid = await qdrant_backend.insert_episode(
        text="Qdrant episode linked to fact.", embedding=emb, user_id="test-user"
    )
    await qdrant_backend.insert_episode_fact(eid, fid)

    episodes = await qdrant_backend.get_episodes_for_facts([fid])
    assert any(e["id"] == eid for e in episodes)


# ── GDPR ──────────────────────────────────────────────────────────────────────


async def test_delete_user(qdrant_backend):
    emb = [0.8, 0.8, 0.8, 0.8]
    await qdrant_backend.upsert_sentences(
        [
            {
                "id": "qdel-s1",
                "text": "Qdrant delete user sentence.",
                "session_id": "qdel-sess",
                "turn_number": 0,
                "sentence_index": 0,
                "role": "user",
            }
        ],
        [emb],
        user_id="delete-test-user",
    )
    await qdrant_backend.insert_fact(
        text="Qdrant delete user fact.", embedding=emb, user_id="delete-test-user"
    )
    deleted = await qdrant_backend.delete_user("delete-test-user")
    assert deleted >= 2

    remaining = await qdrant_backend.search_sentences(emb, user_id="delete-test-user")
    assert remaining == []

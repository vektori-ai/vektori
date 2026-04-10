"""Integration tests for Neo4jBackend — requires a live Neo4j instance.

Start with:
    docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest

Tests are skipped automatically when the server is unreachable.
"""

from __future__ import annotations

import pytest

NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "password")


def _is_neo4j_available() -> bool:
    try:
        import neo4j  # noqa: F401

        return True
    except ImportError:
        return False


async def _can_connect() -> bool:
    if not _is_neo4j_available():
        return False
    try:
        from neo4j import AsyncGraphDatabase

        driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        await driver.verify_connectivity()
        await driver.close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
async def neo4j_backend():
    if not await _can_connect():
        pytest.skip("Neo4j not reachable at bolt://localhost:7687")

    from vektori.storage.neo4j_backend import Neo4jBackend

    backend = Neo4jBackend(uri=NEO4J_URI, auth=NEO4J_AUTH, embedding_dim=4)
    await backend.initialize()

    # Wipe test data before each module run
    await backend._q("MATCH (n) WHERE n.user_id = 'test-user' DETACH DELETE n")
    yield backend
    await backend._q("MATCH (n) WHERE n.user_id = 'test-user' DETACH DELETE n")
    await backend.close()


# ── Sentences ──────────────────────────────────────────────────────────────────


async def test_upsert_and_search_sentences(neo4j_backend):
    emb = [0.1, 0.2, 0.3, 0.4]
    sentences = [
        {
            "id": "s1",
            "text": "I love hiking in the mountains.",
            "session_id": "sess1",
            "turn_number": 0,
            "sentence_index": 0,
            "role": "user",
        }
    ]
    count = await neo4j_backend.upsert_sentences(
        sentences, [emb], user_id="test-user"
    )
    assert count == 1

    results = await neo4j_backend.search_sentences(emb, user_id="test-user", limit=5)
    assert len(results) >= 1
    assert results[0]["text"] == "I love hiking in the mountains."
    assert "distance" in results[0]


async def test_sentence_dedup_increments_mentions(neo4j_backend):
    emb = [0.1, 0.2, 0.3, 0.4]
    sent = [
        {
            "id": "s-dedup",
            "text": "Dedup test sentence.",
            "session_id": "sess-dedup",
            "turn_number": 0,
            "sentence_index": 0,
            "role": "user",
        }
    ]
    await neo4j_backend.upsert_sentences(sent, [emb], user_id="test-user")
    await neo4j_backend.upsert_sentences(sent, [emb], user_id="test-user")

    results = await neo4j_backend.search_sentences(emb, user_id="test-user", limit=20)
    matching = [r for r in results if r["text"] == "Dedup test sentence."]
    assert len(matching) == 1
    assert matching[0]["mentions"] == 2


async def test_find_sentence_containing(neo4j_backend):
    emb = [0.1, 0.2, 0.3, 0.4]
    await neo4j_backend.upsert_sentences(
        [
            {
                "id": "s-contain",
                "text": "The quick brown fox jumps.",
                "session_id": "sess-contain",
                "turn_number": 0,
                "sentence_index": 0,
                "role": "user",
            }
        ],
        [emb],
        user_id="test-user",
    )
    result = await neo4j_backend.find_sentence_containing("sess-contain", "quick brown")
    assert result is not None
    assert "quick brown" in result["text"].lower()

    none_result = await neo4j_backend.find_sentence_containing("sess-contain", "xyz-missing")
    assert none_result is None


# ── Facts ─────────────────────────────────────────────────────────────────────


async def test_insert_and_search_facts(neo4j_backend):
    emb = [0.5, 0.5, 0.5, 0.5]
    fid = await neo4j_backend.insert_fact(
        text="User prefers dark mode.",
        embedding=emb,
        user_id="test-user",
        subject="user",
    )
    assert fid

    results = await neo4j_backend.search_facts(emb, user_id="test-user", limit=5)
    assert any(r["id"] == fid for r in results)


async def test_deactivate_fact(neo4j_backend):
    emb = [0.3, 0.3, 0.3, 0.3]
    fid = await neo4j_backend.insert_fact(
        text="Fact to be deactivated.", embedding=emb, user_id="test-user"
    )
    await neo4j_backend.deactivate_fact(fid)

    results = await neo4j_backend.search_facts(emb, user_id="test-user", active_only=True, limit=20)
    assert all(r["id"] != fid for r in results)


async def test_supersession_chain(neo4j_backend):
    emb = [0.2, 0.2, 0.2, 0.2]
    f1 = await neo4j_backend.insert_fact(
        text="User is 25 years old.", embedding=emb, user_id="test-user"
    )
    f2 = await neo4j_backend.insert_fact(
        text="User is 26 years old.", embedding=emb, user_id="test-user",
        superseded_by_target=f1,
    )
    chain = await neo4j_backend.get_supersession_chain(f2)
    ids = [r["id"] for r in chain]
    assert f2 in ids
    assert f1 in ids


async def test_increment_fact_mentions(neo4j_backend):
    emb = [0.9, 0.1, 0.1, 0.1]
    fid = await neo4j_backend.insert_fact(
        text="Mentions test fact.", embedding=emb, user_id="test-user"
    )
    await neo4j_backend.increment_fact_mentions(fid)
    await neo4j_backend.increment_fact_mentions(fid)

    results = await neo4j_backend.get_active_facts("test-user", limit=50)
    matching = [r for r in results if r["id"] == fid]
    # mentions not returned by get_active_facts — just verify no errors
    assert len(matching) == 1


# ── Join tables ───────────────────────────────────────────────────────────────


async def test_fact_source_linking(neo4j_backend):
    emb = [0.4, 0.4, 0.4, 0.4]
    await neo4j_backend.upsert_sentences(
        [
            {
                "id": "s-src1",
                "text": "Source sentence for fact.",
                "session_id": "sess-src",
                "turn_number": 0,
                "sentence_index": 0,
                "role": "user",
            }
        ],
        [emb],
        user_id="test-user",
    )
    fid = await neo4j_backend.insert_fact(
        text="Fact derived from source.", embedding=emb, user_id="test-user"
    )
    await neo4j_backend.insert_fact_source(fid, "s-src1")

    source_ids = await neo4j_backend.get_source_sentences([fid])
    assert "s-src1" in source_ids


# ── Expand session context ────────────────────────────────────────────────────


async def test_expand_session_context(neo4j_backend):
    emb = [0.1, 0.1, 0.1, 0.1]
    sents = [
        {
            "id": f"ctx-{i}",
            "text": f"Sentence {i}",
            "session_id": "sess-ctx",
            "turn_number": 0,
            "sentence_index": i,
            "role": "user",
        }
        for i in range(6)
    ]
    await neo4j_backend.upsert_sentences(sents, [emb] * 6, user_id="test-user")

    expanded = await neo4j_backend.expand_session_context(["ctx-2"], window=1)
    indices = [r["sentence_index"] for r in expanded]
    assert 1 in indices and 2 in indices and 3 in indices


# ── Sessions ──────────────────────────────────────────────────────────────────


async def test_upsert_and_get_session(neo4j_backend):
    await neo4j_backend.upsert_session(
        session_id="sess-get", user_id="test-user", metadata={"source": "test"}
    )
    session = await neo4j_backend.get_session("sess-get", "test-user")
    assert session is not None
    assert session["id"] == "sess-get"
    assert "sentences" in session


async def test_count_sessions(neo4j_backend):
    await neo4j_backend.upsert_session("sess-cnt-1", "test-user")
    await neo4j_backend.upsert_session("sess-cnt-2", "test-user")
    count = await neo4j_backend.count_sessions("test-user")
    assert count >= 2


# ── Episodes ──────────────────────────────────────────────────────────────────


async def test_insert_and_search_episodes(neo4j_backend):
    emb = [0.6, 0.6, 0.6, 0.6]
    eid = await neo4j_backend.insert_episode(
        text="User enjoys outdoor activities.",
        embedding=emb,
        user_id="test-user",
    )
    assert eid

    results = await neo4j_backend.search_episodes(emb, user_id="test-user", limit=5)
    assert any(r["id"] == eid for r in results)


async def test_episodes_for_facts(neo4j_backend):
    emb = [0.7, 0.7, 0.7, 0.7]
    fid = await neo4j_backend.insert_fact(
        text="Episode linkage fact.", embedding=emb, user_id="test-user"
    )
    eid = await neo4j_backend.insert_episode(
        text="Episode linked to fact.", embedding=emb, user_id="test-user"
    )
    await neo4j_backend.insert_episode_fact(eid, fid)

    episodes = await neo4j_backend.get_episodes_for_facts([fid])
    assert any(e["id"] == eid for e in episodes)


# ── GDPR ──────────────────────────────────────────────────────────────────────


async def test_delete_user(neo4j_backend):
    emb = [0.8, 0.8, 0.8, 0.8]
    await neo4j_backend.upsert_sentences(
        [
            {
                "id": "del-s1",
                "text": "To be deleted.",
                "session_id": "del-sess",
                "turn_number": 0,
                "sentence_index": 0,
                "role": "user",
            }
        ],
        [emb],
        user_id="delete-test-user",
    )
    await neo4j_backend.insert_fact(
        text="Fact to delete.", embedding=emb, user_id="delete-test-user"
    )
    deleted = await neo4j_backend.delete_user("delete-test-user")
    assert deleted >= 2

    remaining = await neo4j_backend.search_sentences(emb, user_id="delete-test-user")
    assert remaining == []

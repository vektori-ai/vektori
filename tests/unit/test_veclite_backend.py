import pytest

pytest.importorskip("veclite")

from vektori.storage.veclite_backend import VecLiteBackend


@pytest.fixture
def temp_dir(tmp_path):
    return str(tmp_path / "veclite_test")


@pytest.mark.asyncio
async def test_veclite_backend_sentences(temp_dir):
    backend = VecLiteBackend(path=temp_dir)
    await backend.initialize()

    sentences = [
        {
            "id": "s1",
            "text": "hello world",
            "session_id": "sess1",
            "turn_number": 0,
            "sentence_index": 0,
        },
        {
            "id": "s2",
            "text": "bye world",
            "session_id": "sess1",
            "turn_number": 1,
            "sentence_index": 0,
        },
    ]
    embeddings = [[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]]
    user_id = "user1"

    count = await backend.upsert_sentences(sentences, embeddings, user_id)
    assert count == 2

    res = await backend.search_sentences([0.1, 0.2, 0.3], user_id, limit=1)
    assert len(res) == 1
    assert res[0]["id"] == "s1"

    await backend.close()


@pytest.mark.asyncio
async def test_veclite_restart_persistence(temp_dir):
    backend = VecLiteBackend(path=temp_dir)
    await backend.initialize()

    # sentences
    sentences = [{"id": "s1", "text": "hello world", "session_id": "sess1"}]
    await backend.upsert_sentences(sentences, [[0.1, 0.2]], "u1")

    # fact
    fact_id = await backend.insert_fact("fact1", [0.1, 0.2], "u1")

    # edge
    await backend.insert_edges([{"source": "s1", "target": "s2", "label": "rel"}])

    # fact_source
    await backend.insert_fact_source(fact_id, "s1")

    # episode & episode_fact
    ep_id = await backend.insert_episode("ep1", [0.1, 0.2], "u1")
    await backend.insert_episode_fact(ep_id, fact_id)

    # session
    await backend.upsert_session("sess1", "u1", "a1")

    await backend.close()

    backend2 = VecLiteBackend(path=temp_dir)
    await backend2.initialize()

    assert len(backend2._sentences) == 1
    assert len(backend2._facts) == 1
    assert len(backend2._edges) == 1
    assert len(backend2._fact_sources) == 1
    assert len(backend2._episodes) == 1
    assert len(backend2._episode_facts) == 1
    assert len(backend2._sessions) == 1

    res = await backend2.search_sentences([0.1, 0.2], "u1", limit=1)
    assert len(res) == 1
    assert res[0]["id"] == "s1"

    await backend2.close()


@pytest.mark.asyncio
async def test_veclite_delete_user_cascade(temp_dir):
    backend = VecLiteBackend(path=temp_dir)
    await backend.initialize()

    # Seed data
    sentences = [{"id": "s1", "text": "hello world", "session_id": "sess1"}]
    await backend.upsert_sentences(sentences, [[0.1, 0.2]], "u1")

    fact_id = await backend.insert_fact("fact1", [0.1, 0.2], "u1")
    await backend.insert_edges([{"source": "s1", "target": "s2", "label": "rel"}])
    await backend.insert_fact_source(fact_id, "s1")
    ep_id = await backend.insert_episode("ep1", [0.1, 0.2], "u1")
    await backend.insert_episode_fact(ep_id, fact_id)
    await backend.upsert_session("sess1", "u1", "a1")

    await backend.delete_user("u1")
    await backend.close()

    # Reopen to verify persistence deletion
    backend2 = VecLiteBackend(path=temp_dir)
    await backend2.initialize()

    assert len(backend2._sentences) == 0
    assert len(backend2._facts) == 0
    assert len(backend2._edges) == 0
    assert len(backend2._fact_sources) == 0
    assert len(backend2._episodes) == 0
    assert len(backend2._episode_facts) == 0
    assert len(backend2._sessions) == 0

    all_relations = backend2._vec_relations.get_all()
    prefixes = ("edge:", "fs:", "ef:", "sess:")
    for rid, _ in all_relations:
        assert not rid.startswith(prefixes)

    await backend2.close()

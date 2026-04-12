import pytest

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

import pytest

from vektori.storage.sqlite import SQLiteBackend


async def test_sqlite_sentence_event_time_and_searchability_round_trip(tmp_path):
    pytest.importorskip("aiosqlite")
    db = SQLiteBackend(database_url=f"sqlite:///{tmp_path / 'vektori.db'}")
    await db.initialize()

    sentence = {
        "id": "sent-1",
        "text": "Yes.",
        "session_id": "sess-1",
        "turn_number": 0,
        "sentence_index": 0,
        "role": "user",
        "event_time": "2026-05-05T10:00:00",
        "is_searchable": False,
    }
    await db.upsert_sentences([sentence], [[0.1, 0.2, 0.3]], user_id="user-1")

    loaded = await db.get_sentences_by_ids(["sent-1"])
    assert loaded[0]["event_time"] == "2026-05-05T10:00:00"
    assert loaded[0]["is_searchable"] == 0

    search_results = await db.search_sentences([0.1, 0.2, 0.3], user_id="user-1")
    assert search_results == []

    await db.close()

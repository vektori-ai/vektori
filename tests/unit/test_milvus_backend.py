"""Unit tests for MilvusBackend logic with mocked client calls."""

from __future__ import annotations

from unittest.mock import AsyncMock

from vektori.storage.milvus import MilvusBackend


async def test_search_facts_builds_filter_expression():
    backend = MilvusBackend(embedding_dim=4)
    backend._search = AsyncMock(return_value=[])

    await backend.search_facts(
        embedding=[0.1, 0.2, 0.3, 0.4],
        user_id="u1",
        agent_id="a1",
        session_id="s1",
        subject="user",
        active_only=True,
        before_date=None,
        after_date=None,
        limit=7,
    )

    backend._search.assert_awaited_once()
    kwargs = backend._search.call_args.kwargs
    assert kwargs["collection_name"] == backend._facts_col
    assert kwargs["limit"] == 7
    expr = kwargs["filter_expr"]
    assert 'record_type == "fact"' in expr
    assert 'user_id == "u1"' in expr
    assert 'agent_id == "a1"' in expr
    assert 'session_id == "s1"' in expr
    assert 'subject == "user"' in expr
    assert "is_active == true" in expr


def test_to_fact_parses_metadata_from_json_string():
    backend = MilvusBackend()
    row = {
        "id": "f1",
        "text": "User prefers WhatsApp",
        "confidence": 0.93,
        "mentions": 4,
        "session_id": "sess-1",
        "subject": "user",
        "is_active": True,
        "superseded_by": "",
        "metadata": '{"source":"user"}',
        "event_time": "2026-01-01T00:00:00",
        "created_at": "2026-01-01T00:00:01",
        "distance": 0.12,
    }

    fact = backend._to_fact(row)
    assert fact["id"] == "f1"
    assert fact["metadata"]["source"] == "user"
    assert fact["distance"] == 0.12


async def test_insert_fact_sources_merges_and_deduplicates_ids():
    backend = MilvusBackend()
    backend._get_one = AsyncMock(
        return_value={
            "id": "f1",
            "source_sentence_ids": "s1,s2",
            "record_type": "fact",
        }
    )
    backend._upsert_rows = AsyncMock()

    await backend.insert_fact_sources([("f1", "s2"), ("f1", "s3")])

    backend._upsert_rows.assert_awaited_once()
    upserted = backend._upsert_rows.call_args.args[1][0]
    assert upserted["source_sentence_ids"] == "s1,s2,s3"


async def test_upsert_session_writes_session_marker_record():
    backend = MilvusBackend(embedding_dim=4)
    backend._get_one = AsyncMock(return_value=None)
    backend._upsert_rows = AsyncMock()

    await backend.upsert_session(
        session_id="sess-1",
        user_id="user-1",
        agent_id="agent-1",
        metadata={"source": "test"},
    )

    backend._upsert_rows.assert_awaited_once()
    collection_name, rows = backend._upsert_rows.call_args.args
    assert collection_name == backend._sentences_col
    row = rows[0]
    assert row["record_type"] == "session"
    assert row["session_id"] == "sess-1"
    assert row["user_id"] == "user-1"
    assert len(row["embedding"]) == 4


async def test_get_session_returns_sentences_sorted():
    backend = MilvusBackend()
    backend._get_one = AsyncMock(
        return_value={
            "user_id": "u1",
            "agent_id": "a1",
            "started_at": "2026-01-01T00:00:00",
            "ended_at": "",
            "metadata": {"source": "test"},
        }
    )
    backend._query_all = AsyncMock(
        return_value=[
            {
                "id": "s2",
                "text": "second",
                "session_id": "sess-1",
                "turn_number": 1,
                "sentence_index": 1,
                "role": "assistant",
                "mentions": 1,
                "is_active": True,
                "created_at": "2026-01-01T00:00:02",
                "distance": 0.2,
            },
            {
                "id": "s1",
                "text": "first",
                "session_id": "sess-1",
                "turn_number": 1,
                "sentence_index": 0,
                "role": "user",
                "mentions": 1,
                "is_active": True,
                "created_at": "2026-01-01T00:00:01",
                "distance": 0.1,
            },
        ]
    )

    session = await backend.get_session("sess-1", "u1")
    assert session is not None
    assert [s["id"] for s in session["sentences"]] == ["s1", "s2"]


async def test_count_sessions_filters_user_and_agent():
    backend = MilvusBackend()
    backend._query_all = AsyncMock(return_value=[{"id": "1"}, {"id": "2"}])

    count = await backend.count_sessions("u1", "a1")

    assert count == 2
    expr = backend._query_all.call_args.kwargs["filter_expr"]
    assert 'record_type == "session"' in expr
    assert 'user_id == "u1"' in expr
    assert 'agent_id == "a1"' in expr

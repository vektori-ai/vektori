"""Unit tests for deterministic ID generation."""

import uuid

from vektori.ingestion.hasher import generate_content_hash, generate_sentence_id


def test_deterministic():
    id1 = generate_sentence_id("session-1", 0, "Hello world")
    id2 = generate_sentence_id("session-1", 0, "Hello world")
    assert id1 == id2


def test_different_session_different_id():
    id1 = generate_sentence_id("session-1", 0, "Hello world")
    id2 = generate_sentence_id("session-2", 0, "Hello world")
    assert id1 != id2


def test_different_index_different_id():
    id1 = generate_sentence_id("session-1", 0, "Hello world")
    id2 = generate_sentence_id("session-1", 1, "Hello world")
    assert id1 != id2


def test_different_text_different_id():
    id1 = generate_sentence_id("session-1", 0, "Hello world")
    id2 = generate_sentence_id("session-1", 0, "Hello earth")
    assert id1 != id2


def test_valid_uuid_format():
    result = generate_sentence_id("s", 0, "test sentence here")
    uuid.UUID(result)  # should not raise


def test_content_hash_is_hex():
    h = generate_content_hash("s1", "0_0", "test")
    assert len(h) == 64
    int(h, 16)  # should not raise — valid hex

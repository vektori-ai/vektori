"""Integration tests for conflict detection and resolution."""

from vektori.storage.memory import MemoryBackend


async def test_deactivate_fact():
    db = MemoryBackend()
    await db.initialize()

    old_id = await db.insert_fact(
        text="User prefers email",
        embedding=[0.1] * 1536,
        user_id="u1",
        confidence=1.0,
    )

    # Verify it's active
    facts = await db.get_active_facts("u1")
    assert any(f["id"] == old_id for f in facts)

    # Deactivate (conflict resolution)
    await db.deactivate_fact(old_id)

    # Should no longer appear in active facts
    facts = await db.get_active_facts("u1")
    assert not any(f["id"] == old_id for f in facts)


async def test_supersession_chain():
    db = MemoryBackend()
    await db.initialize()

    old_id = await db.insert_fact(
        text="User prefers email",
        embedding=[0.1] * 1536,
        user_id="u1",
        confidence=1.0,
    )
    await db.deactivate_fact(old_id)

    new_id = await db.insert_fact(
        text="User prefers WhatsApp",
        embedding=[0.2] * 1536,
        user_id="u1",
        confidence=1.0,
        superseded_by_target=old_id,
    )

    chain = await db.get_supersession_chain(new_id)
    assert len(chain) >= 1
    assert any(f["text"] == "User prefers WhatsApp" for f in chain)


async def test_old_fact_preserved_after_supersession():
    """Old facts are never deleted — history is preserved."""
    db = MemoryBackend()
    await db.initialize()

    old_id = await db.insert_fact(
        text="User prefers email",
        embedding=[0.1] * 1536,
        user_id="u1",
        confidence=1.0,
    )
    await db.deactivate_fact(old_id)

    # Should still exist in the store, just inactive
    assert old_id in db._facts
    assert db._facts[old_id]["is_active"] is False


async def test_find_fact_by_text():
    db = MemoryBackend()
    await db.initialize()

    await db.insert_fact(
        text="User prefers WhatsApp",
        embedding=[0.1] * 1536,
        user_id="u1",
        confidence=1.0,
    )

    found = await db.find_fact_by_text("u1", "User prefers WhatsApp")
    assert found is not None
    assert found["text"] == "User prefers WhatsApp"

    not_found = await db.find_fact_by_text("u1", "User prefers carrier pigeon")
    assert not_found is None

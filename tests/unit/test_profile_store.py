from pathlib import Path

from vektori.memory.profile import ProfilePatch, SQLiteProfileStore


async def test_sqlite_profile_store_persists_patch(tmp_path: Path):
    store = SQLiteProfileStore(tmp_path / "profiles.db")
    patch = ProfilePatch(
        key="response_style.verbosity",
        value="short",
        reason="explicit preference",
        source="explicit_user_request",
        observer_id="agent-1",
        observed_id="user-1",
        confidence=0.9,
    )

    await store.save(patch)
    loaded = await store.list_active("agent-1", "user-1")
    await store.close()

    assert len(loaded) == 1
    assert loaded[0].key == "response_style.verbosity"
    assert loaded[0].value == "short"

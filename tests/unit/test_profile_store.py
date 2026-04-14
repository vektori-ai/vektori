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


async def test_sqlite_profile_store_deactivates_old_value_and_keeps_history(tmp_path: Path):
    store = SQLiteProfileStore(tmp_path / "profiles.db")
    await store.save(
        ProfilePatch(
            key="response_style.verbosity",
            value="short",
            reason="explicit preference",
            source="explicit_user_request",
            observer_id="agent-1",
            observed_id="user-1",
            confidence=0.9,
        )
    )
    await store.save(
        ProfilePatch(
            key="response_style.verbosity",
            value="detailed",
            reason="new explicit preference",
            source="explicit_user_request",
            observer_id="agent-1",
            observed_id="user-1",
            confidence=0.95,
        )
    )
    active = await store.list_active("agent-1", "user-1")
    all_patches = await store.list_all("agent-1", "user-1")
    await store.close()

    assert len(active) == 1
    assert active[0].value == "detailed"
    assert len(all_patches) == 2
    assert sum(1 for patch in all_patches if patch.active) == 1


async def test_sqlite_profile_store_confirms_same_value_without_duplicate_active_row(tmp_path: Path):
    store = SQLiteProfileStore(tmp_path / "profiles.db")
    patch = ProfilePatch(
        key="preferences.units",
        value="metric",
        reason="explicit preference",
        source="explicit_user_request",
        observer_id="agent-1",
        observed_id="user-1",
        confidence=0.9,
    )
    await store.save(patch)
    await store.save(
        ProfilePatch(
            key="preferences.units",
            value="metric",
            reason="repeat confirmation",
            source="explicit_user_request",
            observer_id="agent-1",
            observed_id="user-1",
            confidence=0.95,
        )
    )
    active = await store.list_active("agent-1", "user-1")
    all_patches = await store.list_all("agent-1", "user-1")
    await store.close()

    assert len(active) == 1
    assert len(all_patches) == 1
    assert active[0].confidence == 0.95
    assert active[0].last_confirmed_at is not None

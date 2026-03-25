"""Storage backend factory — resolves config to a concrete backend."""

from __future__ import annotations

from vektori.config import VektoriConfig
from vektori.storage.base import StorageBackend


async def create_storage(config: VektoriConfig) -> StorageBackend:
    """Resolve and initialize the correct storage backend from config."""
    backend_key = config.storage_backend
    database_url = config.database_url

    if backend_key == "memory":
        from vektori.storage.memory import MemoryBackend
        backend: StorageBackend = MemoryBackend()

    elif backend_key == "postgres" or (database_url and "postgresql" in database_url):
        from vektori.storage.postgres import PostgresBackend
        if not database_url:
            raise ValueError(
                "database_url is required for PostgreSQL backend. "
                "Example: postgresql://vektori:vektori@localhost:5432/vektori"
            )
        backend = PostgresBackend(database_url)

    else:
        from vektori.storage.sqlite import SQLiteBackend
        backend = SQLiteBackend(database_url)

    await backend.initialize()
    return backend

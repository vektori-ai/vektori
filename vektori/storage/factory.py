"""Storage backend factory — resolves config to a concrete backend."""

from __future__ import annotations

from vektori.config import VektoriConfig
from vektori.storage.base import StorageBackend


async def create_storage(config: VektoriConfig) -> StorageBackend:
    """Resolve and initialize the correct storage backend from config.

    Backend selection priority:
        1. config.storage_backend key  ("sqlite", "postgres", "memory", "neo4j", "qdrant")
        2. URL prefix heuristic        (postgresql://, bolt://, neo4j://, http://localhost:6333)
    """
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
        backend = PostgresBackend(
            database_url,
            embedding_dim=config.embedding_dimension,
        )

    elif backend_key == "neo4j" or (
        database_url
        and (
            database_url.startswith("bolt://")
            or database_url.startswith("neo4j://")
            or database_url.startswith("neo4j+s://")
        )
    ):
        from vektori.storage.neo4j_backend import Neo4jBackend

        uri = database_url or "bolt://localhost:7687"
        # Auth can be passed as "user:password" appended after a space, e.g.:
        #   database_url="bolt://localhost:7687 neo4j:password"
        # or left as default ("neo4j", "password") for local dev.
        user, password = "neo4j", "password"
        if " " in uri:
            uri, creds = uri.split(" ", 1)
            if ":" in creds:
                user, password = creds.split(":", 1)
        backend = Neo4jBackend(
            uri=uri,
            auth=(user, password),
            embedding_dim=config.embedding_dimension,
        )

    elif backend_key == "qdrant" or (
        database_url
        and (
            "qdrant" in database_url
            or (database_url.startswith("http") and ":6333" in database_url)
        )
    ):
        from vektori.storage.qdrant_backend import QdrantBackend

        url = database_url or "http://localhost:6333"
        backend = QdrantBackend(
            url=url,
            api_key=config.qdrant_api_key,
            embedding_dim=config.embedding_dimension,
        )

    else:
        from vektori.storage.sqlite import SQLiteBackend

        backend = SQLiteBackend(database_url)

    await backend.initialize()
    return backend

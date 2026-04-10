"""Unit tests for Neo4j and Qdrant factory routing — no live server required."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from vektori.config import VektoriConfig
from vektori.storage.factory import create_storage
from vektori.storage.neo4j_backend import Neo4jBackend
from vektori.storage.qdrant_backend import QdrantBackend


# ── Factory routing: storage_backend key ──────────────────────────────────────


async def test_factory_neo4j_by_key():
    cfg = VektoriConfig(storage_backend="neo4j", embedding_dimension=1024)
    with patch.object(Neo4jBackend, "initialize", new_callable=AsyncMock):
        backend = await create_storage(cfg)
    assert isinstance(backend, Neo4jBackend)
    assert backend.embedding_dim == 1024


async def test_factory_qdrant_by_key():
    cfg = VektoriConfig(storage_backend="qdrant", embedding_dimension=768)
    with patch.object(QdrantBackend, "initialize", new_callable=AsyncMock):
        backend = await create_storage(cfg)
    assert isinstance(backend, QdrantBackend)
    assert backend.embedding_dim == 768


# ── Factory routing: URL-based heuristic ─────────────────────────────────────


@pytest.mark.parametrize(
    "url",
    [
        "bolt://localhost:7687",
        "neo4j://localhost:7687",
        "neo4j+s://myhost:7687",
    ],
)
async def test_factory_neo4j_by_url(url):
    cfg = VektoriConfig(storage_backend="sqlite", database_url=url)
    with patch.object(Neo4jBackend, "initialize", new_callable=AsyncMock):
        backend = await create_storage(cfg)
    assert isinstance(backend, Neo4jBackend)
    assert backend.uri == url


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:6333",
        "http://qdrant-host:6333",
    ],
)
async def test_factory_qdrant_by_url(url):
    cfg = VektoriConfig(storage_backend="sqlite", database_url=url)
    with patch.object(QdrantBackend, "initialize", new_callable=AsyncMock):
        backend = await create_storage(cfg)
    assert isinstance(backend, QdrantBackend)
    assert backend.url == url


# ── Neo4j auth parsing in factory ─────────────────────────────────────────────


async def test_factory_neo4j_auth_from_url():
    """bolt://... user:pass format parsed correctly."""
    cfg = VektoriConfig(
        storage_backend="neo4j",
        database_url="bolt://prod-host:7687 admin:s3cr3t",
    )
    with patch.object(Neo4jBackend, "initialize", new_callable=AsyncMock):
        backend = await create_storage(cfg)
    assert isinstance(backend, Neo4jBackend)
    assert backend.uri == "bolt://prod-host:7687"
    assert backend.auth == ("admin", "s3cr3t")


async def test_factory_neo4j_default_auth():
    """Default auth (neo4j / password) used when no creds in URL."""
    cfg = VektoriConfig(storage_backend="neo4j", database_url="bolt://localhost:7687")
    with patch.object(Neo4jBackend, "initialize", new_callable=AsyncMock):
        backend = await create_storage(cfg)
    assert backend.auth == ("neo4j", "password")


# ── QdrantBackend constructor defaults ────────────────────────────────────────


def test_qdrant_default_url():
    b = QdrantBackend()
    assert b.url == "http://localhost:6333"
    assert b.prefix == "vektori"
    assert b.embedding_dim == 1024


def test_qdrant_custom_prefix():
    b = QdrantBackend(url="http://host:6333", prefix="myapp", embedding_dim=512)
    assert b.prefix == "myapp"
    assert b._facts_col == "myapp_facts"
    assert b._sentences_col == "myapp_sentences"
    assert b._episodes_col == "myapp_episodes"
    assert b._sessions_col == "myapp_sessions"


# ── Neo4jBackend constructor defaults ─────────────────────────────────────────


def test_neo4j_defaults():
    b = Neo4jBackend(uri="bolt://localhost:7687")
    assert b.auth == ("neo4j", "password")
    assert b.database == "neo4j"
    assert b.embedding_dim == 1024
    assert b._driver is None


def test_neo4j_custom_params():
    b = Neo4jBackend(
        uri="neo4j+s://prod:7687",
        auth=("alice", "pw"),
        database="mydb",
        embedding_dim=768,
    )
    assert b.database == "mydb"
    assert b.embedding_dim == 768

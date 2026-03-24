"""SQLite + aiosqlite storage backend. Zero-config default."""

from __future__ import annotations

import json
import logging
import math
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from vektori.storage.base import StorageBackend

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path.home() / ".vektori" / "vektori.db"


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SQLiteBackend(StorageBackend):
    """
    SQLite + aiosqlite storage backend.

    Zero-config: just `pip install vektori` and go.
    Embeddings stored as JSON blobs. Vector search via brute-force cosine.
    Fine for <10K sentences. Upgrade to Postgres for production scale.

    DB location: ~/.vektori/vektori.db (or pass database_url="sqlite:///path/to/db")
    """

    def __init__(self, database_url: str | None = None) -> None:
        if database_url and database_url.startswith("sqlite:///"):
            self.db_path = Path(database_url[10:])
        else:
            self.db_path = DEFAULT_DB_PATH
        self._conn = None

    async def initialize(self) -> None:
        """Create database file and tables."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import aiosqlite
        except ImportError as e:
            raise ImportError(
                "aiosqlite required for SQLite backend: pip install aiosqlite"
            ) from e

        self._conn = await aiosqlite.connect(str(self.db_path))
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._create_tables()
        await self._migrate()
        await self._conn.commit()
        logger.info("SQLite backend initialized at %s", self.db_path)

    async def _create_tables(self) -> None:
        """Create SQLite-compatible tables (embeddings stored as JSON TEXT)."""
        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sentences (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                embedding TEXT,  -- JSON array of floats
                user_id TEXT NOT NULL,
                agent_id TEXT,
                session_id TEXT NOT NULL,
                turn_number INTEGER NOT NULL,
                sentence_index INTEGER NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                content_hash TEXT NOT NULL UNIQUE,
                mentions INTEGER DEFAULT 1,
                is_active INTEGER DEFAULT 1,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                embedding TEXT,
                user_id TEXT NOT NULL,
                agent_id TEXT,
                session_id TEXT,
                subject TEXT,
                is_active INTEGER DEFAULT 1,
                superseded_by TEXT REFERENCES facts(id),
                confidence REAL DEFAULT 1.0,
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS insights (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                embedding TEXT,
                user_id TEXT NOT NULL,
                agent_id TEXT,
                confidence REAL DEFAULT 1.0,
                is_active INTEGER DEFAULT 1,
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS sentence_edges (
                source_id TEXT NOT NULL REFERENCES sentences(id) ON DELETE CASCADE,
                target_id TEXT NOT NULL REFERENCES sentences(id) ON DELETE CASCADE,
                edge_type TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                created_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (source_id, target_id, edge_type)
            );

            CREATE TABLE IF NOT EXISTS fact_sources (
                fact_id TEXT NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
                sentence_id TEXT NOT NULL REFERENCES sentences(id) ON DELETE CASCADE,
                PRIMARY KEY (fact_id, sentence_id)
            );

            CREATE TABLE IF NOT EXISTS insight_sources (
                insight_id TEXT NOT NULL REFERENCES insights(id) ON DELETE CASCADE,
                sentence_id TEXT NOT NULL REFERENCES sentences(id) ON DELETE CASCADE,
                PRIMARY KEY (insight_id, sentence_id)
            );

            CREATE TABLE IF NOT EXISTS insight_facts (
                insight_id TEXT NOT NULL REFERENCES insights(id) ON DELETE CASCADE,
                fact_id TEXT NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
                PRIMARY KEY (insight_id, fact_id)
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                agent_id TEXT,
                started_at TEXT DEFAULT (datetime('now')),
                ended_at TEXT,
                metadata TEXT DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_sentences_user ON sentences (user_id);
            CREATE INDEX IF NOT EXISTS idx_sentences_session ON sentences (session_id);
            CREATE INDEX IF NOT EXISTS idx_facts_user ON facts (user_id);
            CREATE INDEX IF NOT EXISTS idx_facts_active ON facts (user_id, is_active);
            CREATE INDEX IF NOT EXISTS idx_insights_user ON insights (user_id);
            CREATE INDEX IF NOT EXISTS idx_insight_facts_fact ON insight_facts (fact_id);
            CREATE INDEX IF NOT EXISTS idx_fact_sources_fact ON fact_sources (fact_id);
        """)

    async def _migrate(self) -> None:
        """Apply additive migrations for existing DBs. Safe to run on every init."""
        async with self._conn.execute("PRAGMA table_info(facts)") as cursor:
            cols = {row[1] for row in await cursor.fetchall()}
        if "session_id" not in cols:
            await self._conn.execute("ALTER TABLE facts ADD COLUMN session_id TEXT")
        if "subject" not in cols:
            await self._conn.execute("ALTER TABLE facts ADD COLUMN subject TEXT")

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()

    # ── Sentences ──

    async def upsert_sentences(
        self,
        sentences: list[dict[str, Any]],
        embeddings: list[list[float]],
        user_id: str,
        agent_id: str | None = None,
    ) -> int:
        count = 0
        from vektori.ingestion.hasher import generate_content_hash
        for sent, emb in zip(sentences, embeddings):
            content_hash = generate_content_hash(
                sent["session_id"], f"{sent['turn_number']}_{sent['sentence_index']}", sent["text"]
            )
            await self._conn.execute(
                """INSERT INTO sentences
                   (id, text, embedding, user_id, agent_id, session_id, turn_number,
                    sentence_index, role, content_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (content_hash) DO UPDATE SET mentions = mentions + 1""",
                (
                    sent["id"], sent["text"], json.dumps(emb),
                    user_id, agent_id,
                    sent["session_id"], sent["turn_number"], sent["sentence_index"],
                    sent.get("role", "user"), content_hash,
                ),
            )
            count += 1
        await self._conn.commit()
        return count

    async def search_sentences(
        self,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM sentences WHERE user_id = ?"
        params: list[Any] = [user_id]
        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)
        async with self._conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
        results = []
        for row in rows:
            row_dict = dict(row)
            emb = json.loads(row_dict.pop("embedding") or "null")
            if emb:
                sim = _cosine_similarity(embedding, emb)
                results.append({**row_dict, "distance": 1.0 - sim})
        results.sort(key=lambda x: x["distance"])
        return results[:limit]

    async def find_sentences_by_similarity(
        self,
        quotes: list[str],
        session_id: str,
        threshold: float = 0.75,
    ) -> list[str]:
        # TODO: embed each quote (requires access to embedder — wire in later)
        return []

    async def find_sentence_containing(
        self,
        session_id: str,
        quote: str,
    ) -> dict[str, Any] | None:
        async with self._conn.execute(
            "SELECT * FROM sentences WHERE session_id = ? AND text LIKE ? LIMIT 1",
            (session_id, f"%{quote}%"),
        ) as cursor:
            row = await cursor.fetchone()
        return dict(row) if row else None

    # ── Facts ──

    async def insert_fact(
        self,
        text: str,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        subject: str | None = None,
        confidence: float = 1.0,
        superseded_by_target: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        fact_id = str(uuid.uuid4())
        await self._conn.execute(
            """INSERT INTO facts
               (id, text, embedding, user_id, agent_id, session_id, subject,
                confidence, superseded_by, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (fact_id, text, json.dumps(embedding), user_id, agent_id, session_id,
             subject, confidence, superseded_by_target, json.dumps(metadata or {})),
        )
        await self._conn.commit()
        return fact_id

    async def search_facts(
        self,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        session_id: str | None = None,
        subject: str | None = None,
        limit: int = 10,
        active_only: bool = True,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM facts WHERE user_id = ?"
        params: list[Any] = [user_id]
        if active_only:
            query += " AND is_active = 1"
        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        if subject:
            query += " AND subject = ?"
            params.append(subject)
        async with self._conn.execute(query, params) as cursor:
            rows = await cursor.fetchall()
        results = []
        for row in rows:
            row_dict = dict(row)
            emb = json.loads(row_dict.pop("embedding") or "null")
            if emb:
                sim = _cosine_similarity(embedding, emb)
                results.append({**row_dict, "distance": 1.0 - sim, "created_at": _parse_dt(row_dict.get("created_at"))})
        results.sort(key=lambda x: x["distance"])
        return results[:limit]

    async def get_active_facts(
        self,
        user_id: str,
        agent_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM facts WHERE user_id = ? AND is_active = 1 LIMIT ? OFFSET ?"
        async with self._conn.execute(query, (user_id, limit, offset)) as cursor:
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def deactivate_fact(self, fact_id: str, superseded_by: str | None = None) -> None:
        await self._conn.execute(
            "UPDATE facts SET is_active = 0, superseded_by = ? WHERE id = ?",
            (superseded_by, fact_id),
        )
        await self._conn.commit()

    async def find_fact_by_text(
        self,
        user_id: str,
        text: str,
        agent_id: str | None = None,
    ) -> dict[str, Any] | None:
        async with self._conn.execute(
            "SELECT * FROM facts WHERE user_id = ? AND text = ? AND is_active = 1 LIMIT 1",
            (user_id, text),
        ) as cursor:
            row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_supersession_chain(self, fact_id: str) -> list[dict[str, Any]]:
        chain = []
        current_id: str | None = fact_id
        visited: set[str] = set()
        while current_id and current_id not in visited:
            visited.add(current_id)
            async with self._conn.execute(
                "SELECT * FROM facts WHERE id = ?", (current_id,)
            ) as cursor:
                row = await cursor.fetchone()
            if row:
                row_dict = dict(row)
                chain.append(row_dict)
                current_id = row_dict.get("superseded_by")
            else:
                break
        return chain

    # ── Insights ──

    async def insert_insight(
        self,
        text: str,
        embedding: list[float],
        user_id: str,
        agent_id: str | None = None,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        insight_id = str(uuid.uuid4())
        await self._conn.execute(
            """INSERT INTO insights (id, text, embedding, user_id, agent_id, confidence, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (insight_id, text, json.dumps(embedding), user_id, agent_id,
             confidence, json.dumps(metadata or {})),
        )
        await self._conn.commit()
        return insight_id

    async def get_insights_from_facts(
        self,
        fact_ids: list[str],
        user_id: str,
        active_only: bool = True,
    ) -> list[dict[str, Any]]:
        if not fact_ids:
            return []
        placeholders = ",".join("?" * len(fact_ids))
        query = f"""
            SELECT DISTINCT i.* FROM insights i
            JOIN insight_facts inf ON i.id = inf.insight_id
            WHERE inf.fact_id IN ({placeholders})
              AND i.user_id = ?
        """
        if active_only:
            query += " AND i.is_active = 1"
        async with self._conn.execute(query, (*fact_ids, user_id)) as cursor:
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_active_insights(
        self,
        user_id: str,
        agent_id: str | None = None,
    ) -> list[dict[str, Any]]:
        async with self._conn.execute(
            "SELECT * FROM insights WHERE user_id = ? AND is_active = 1",
            (user_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # ── Edges ──

    async def insert_edges(self, edges: list[dict[str, Any]]) -> int:
        for edge in edges:
            await self._conn.execute(
                """INSERT OR IGNORE INTO sentence_edges (source_id, target_id, edge_type, weight)
                   VALUES (?, ?, ?, ?)""",
                (edge["source_id"], edge["target_id"], edge["edge_type"], edge.get("weight", 1.0)),
            )
        await self._conn.commit()
        return len(edges)

    async def expand_session_context(
        self,
        sentence_ids: list[str],
        window: int = 3,
    ) -> list[dict[str, Any]]:
        if not sentence_ids:
            return []
        placeholders = ",".join("?" * len(sentence_ids))
        query = f"""
            SELECT DISTINCT s2.*
            FROM sentences s2
            JOIN sentences src ON s2.session_id = src.session_id
            WHERE src.id IN ({placeholders})
              AND s2.turn_number = src.turn_number
              AND s2.sentence_index BETWEEN src.sentence_index - ? AND src.sentence_index + ?
            ORDER BY s2.turn_number, s2.sentence_index
        """
        async with self._conn.execute(query, [*sentence_ids, window, window]) as cursor:
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # ── Join tables ──

    async def insert_fact_source(self, fact_id: str, sentence_id: str) -> None:
        await self._conn.execute(
            "INSERT OR IGNORE INTO fact_sources (fact_id, sentence_id) VALUES (?, ?)",
            (fact_id, sentence_id),
        )
        await self._conn.commit()

    async def insert_insight_fact(self, insight_id: str, fact_id: str) -> None:
        await self._conn.execute(
            "INSERT OR IGNORE INTO insight_facts (insight_id, fact_id) VALUES (?, ?)",
            (insight_id, fact_id),
        )
        await self._conn.commit()

    async def insert_insight_source(self, insight_id: str, sentence_id: str) -> None:
        await self._conn.execute(
            "INSERT OR IGNORE INTO insight_sources (insight_id, sentence_id) VALUES (?, ?)",
            (insight_id, sentence_id),
        )
        await self._conn.commit()

    async def get_source_sentences(self, fact_ids: list[str]) -> list[str]:
        if not fact_ids:
            return []
        placeholders = ",".join("?" * len(fact_ids))
        async with self._conn.execute(
            f"SELECT DISTINCT sentence_id FROM fact_sources WHERE fact_id IN ({placeholders})",
            fact_ids,
        ) as cursor:
            rows = await cursor.fetchall()
        return [row[0] for row in rows]

    async def get_sentences_by_ids(
        self, sentence_ids: list[str]
    ) -> list[dict[str, Any]]:
        if not sentence_ids:
            return []
        placeholders = ",".join("?" * len(sentence_ids))
        async with self._conn.execute(
            f"""
            SELECT id, text, session_id, turn_number, sentence_index, role, created_at
            FROM sentences
            WHERE id IN ({placeholders}) AND is_active = 1
            ORDER BY session_id, turn_number, sentence_index
            """,
            sentence_ids,
        ) as cursor:
            rows = await cursor.fetchall()
            cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in rows]

    # ── Sessions ──

    async def upsert_session(
        self,
        session_id: str,
        user_id: str,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await self._conn.execute(
            """INSERT INTO sessions (id, user_id, agent_id, metadata)
               VALUES (?, ?, ?, ?)
               ON CONFLICT (id) DO UPDATE SET metadata = excluded.metadata""",
            (session_id, user_id, agent_id, json.dumps(metadata or {})),
        )
        await self._conn.commit()

    async def get_session(
        self,
        session_id: str,
        user_id: str,
    ) -> dict[str, Any] | None:
        async with self._conn.execute(
            "SELECT * FROM sessions WHERE id = ? AND user_id = ?",
            (session_id, user_id),
        ) as cursor:
            row = await cursor.fetchone()
        if not row:
            return None
        session = dict(row)
        async with self._conn.execute(
            "SELECT * FROM sentences WHERE session_id = ? ORDER BY turn_number, sentence_index",
            (session_id,),
        ) as cursor:
            sents = await cursor.fetchall()
        session["sentences"] = [dict(s) for s in sents]
        return session

    async def count_sessions(
        self,
        user_id: str,
        agent_id: str | None = None,
    ) -> int:
        query = "SELECT COUNT(*) FROM sessions WHERE user_id = ?"
        params: list[Any] = [user_id]
        if agent_id is not None:
            query += " AND agent_id = ?"
            params.append(agent_id)
        async with self._conn.execute(query, params) as cursor:
            row = await cursor.fetchone()
        return row[0] if row else 0

    # ── Lifecycle ──

    async def delete_user(self, user_id: str) -> int:
        count = 0
        for table in ["sentences", "facts", "insights", "sessions"]:
            async with self._conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE user_id = ?", (user_id,)
            ) as cursor:
                row = await cursor.fetchone()
                count += row[0] if row else 0
            await self._conn.execute(f"DELETE FROM {table} WHERE user_id = ?", (user_id,))
        await self._conn.commit()
        return count


def _parse_dt(val: Any) -> datetime:
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        try:
            return datetime.fromisoformat(val)
        except ValueError:
            pass
    return datetime.utcnow()

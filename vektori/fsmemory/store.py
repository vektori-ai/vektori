"""FSStore — wraps StorageBackend and adds the fs_file_index table."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vektori.storage.base import StorageBackend

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path.home() / ".vektori" / "fsmemory.db"


def path_session_id(path: str) -> str:
    """Deterministic session_id for a file path — enables re-ingest dedup."""
    return hashlib.sha256(path.encode()).hexdigest()[:16]


class FSStore:
    """
    Thin wrapper over StorageBackend for filesystem memory.

    Adds a `fs_file_index` table to track ingested files by content hash,
    so re-ingesting an unchanged file is a no-op.
    """

    def __init__(self, db: StorageBackend, db_path: Path) -> None:
        self.db = db
        self._db_path = db_path
        self._conn: Any = None  # aiosqlite connection for the file index table

    async def initialize(self) -> None:
        try:
            import aiosqlite
        except ImportError as e:
            raise ImportError("aiosqlite required: pip install aiosqlite") from e

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self._db_path))
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("""
            CREATE TABLE IF NOT EXISTS fs_file_index (
                path TEXT NOT NULL,
                user_id TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                ingested_at TEXT DEFAULT (datetime('now')),
                fact_count INTEGER DEFAULT 0,
                PRIMARY KEY (path, user_id)
            )
        """)
        await self._conn.commit()

    async def get_file_hash(self, path: str, user_id: str) -> str | None:
        async with self._conn.execute(
            "SELECT content_hash FROM fs_file_index WHERE path = ? AND user_id = ?",
            (path, user_id),
        ) as cursor:
            row = await cursor.fetchone()
        return row["content_hash"] if row else None

    async def set_file_index(self, path: str, user_id: str, content_hash: str, fact_count: int) -> None:
        await self._conn.execute(
            """INSERT INTO fs_file_index (path, user_id, content_hash, fact_count)
               VALUES (?, ?, ?, ?)
               ON CONFLICT (path, user_id) DO UPDATE SET
                 content_hash = excluded.content_hash,
                 fact_count = excluded.fact_count,
                 ingested_at = datetime('now')""",
            (path, user_id, content_hash, fact_count),
        )
        await self._conn.commit()

    async def deactivate_by_path(self, path: str, user_id: str) -> int:
        """Deactivate all facts for a file (called before re-ingesting a changed file)."""
        session_id = path_session_id(path)
        async with self._conn.execute(
            "UPDATE facts SET is_active = 0 WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        ) as cursor:
            count = cursor.rowcount
        await self._conn.commit()
        return count

    async def remove_from_index(self, path: str, user_id: str) -> None:
        await self._conn.execute(
            "DELETE FROM fs_file_index WHERE path = ? AND user_id = ?",
            (path, user_id),
        )
        await self._conn.commit()

    async def list_paths(self, user_id: str) -> list[str]:
        async with self._conn.execute(
            "SELECT path FROM fs_file_index WHERE user_id = ? ORDER BY ingested_at DESC",
            (user_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [r["path"] for r in rows]

    async def get_stats(self, user_id: str) -> dict[str, int]:
        async with self._conn.execute(
            "SELECT COUNT(*) as files, SUM(fact_count) as facts FROM fs_file_index WHERE user_id = ?",
            (user_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return {"files": row["files"] or 0, "facts": row["facts"] or 0}

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

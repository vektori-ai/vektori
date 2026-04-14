"""Durable profile patches for the conversational harness."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from typing import Any


@dataclass
class ProfilePatch:
    """Durable behavioral preference learned for an observer/observed pair."""

    key: str
    value: Any
    reason: str
    source: str
    observer_id: str
    observed_id: str
    confidence: float
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_confirmed_at: datetime | None = None
    active: bool = True


class ProfileStore(Protocol):
    """Storage interface for durable profile patches."""

    async def list_active(self, observer_id: str, observed_id: str) -> list[ProfilePatch]:
        ...

    async def list_all(self, observer_id: str, observed_id: str) -> list[ProfilePatch]:
        ...

    async def save(self, patch: ProfilePatch) -> None:
        ...

    async def close(self) -> None:
        ...


class InMemoryProfileStore:
    """Minimal phase-1 profile patch store kept in process memory."""

    def __init__(self) -> None:
        self._patches: dict[tuple[str, str], list[ProfilePatch]] = {}

    async def list_active(self, observer_id: str, observed_id: str) -> list[ProfilePatch]:
        patches = self._patches.get((observer_id, observed_id), [])
        return [patch for patch in patches if patch.active]

    async def list_all(self, observer_id: str, observed_id: str) -> list[ProfilePatch]:
        return list(self._patches.get((observer_id, observed_id), []))

    async def save(self, patch: ProfilePatch) -> None:
        key = (patch.observer_id, patch.observed_id)
        patches = self._patches.setdefault(key, [])
        now = datetime.now(timezone.utc)
        for existing in reversed(patches):
            if existing.key != patch.key or not existing.active:
                continue
            if existing.value == patch.value:
                existing.last_confirmed_at = now
                existing.confidence = max(existing.confidence, patch.confidence)
                existing.reason = patch.reason
                existing.source = patch.source
                return
            existing.active = False
            existing.last_confirmed_at = now
        patches.append(patch)

    async def close(self) -> None:
        return None


class SQLiteProfileStore:
    """Persistent SQLite-backed profile patch store for the harness."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._initialized = False
        self._connection: sqlite3.Connection | None = None

    async def list_active(self, observer_id: str, observed_id: str) -> list[ProfilePatch]:
        await self._ensure_initialized()
        assert self._connection is not None
        cursor = self._connection.execute(
            """
            SELECT key, value_json, reason, source, observer_id, observed_id,
                   confidence, created_at, last_confirmed_at, active
            FROM profile_patches
            WHERE observer_id = ? AND observed_id = ? AND active = 1
            ORDER BY created_at ASC
            """,
            (observer_id, observed_id),
        )
        rows = cursor.fetchall()
        cursor.close()
        return [self._row_to_patch(row) for row in rows]

    async def list_all(self, observer_id: str, observed_id: str) -> list[ProfilePatch]:
        await self._ensure_initialized()
        assert self._connection is not None
        cursor = self._connection.execute(
            """
            SELECT key, value_json, reason, source, observer_id, observed_id,
                   confidence, created_at, last_confirmed_at, active
            FROM profile_patches
            WHERE observer_id = ? AND observed_id = ?
            ORDER BY created_at ASC
            """,
            (observer_id, observed_id),
        )
        rows = cursor.fetchall()
        cursor.close()
        return [self._row_to_patch(row) for row in rows]

    async def save(self, patch: ProfilePatch) -> None:
        await self._ensure_initialized()
        assert self._connection is not None
        now = datetime.now(timezone.utc).isoformat()
        cursor = self._connection.execute(
            """
            SELECT id, value_json, confidence
            FROM profile_patches
            WHERE observer_id = ? AND observed_id = ? AND key = ? AND active = 1
            ORDER BY created_at DESC
            """,
            (patch.observer_id, patch.observed_id, patch.key),
        )
        existing_rows = cursor.fetchall()
        cursor.close()

        for row_id, value_json, existing_confidence in existing_rows:
            if json.loads(value_json) == patch.value:
                self._connection.execute(
                    """
                    UPDATE profile_patches
                    SET confidence = ?, reason = ?, source = ?, last_confirmed_at = ?
                    WHERE id = ?
                    """,
                    (
                        max(float(existing_confidence), patch.confidence),
                        patch.reason,
                        patch.source,
                        now,
                        row_id,
                    ),
                )
                self._connection.commit()
                return

        if existing_rows:
            self._connection.execute(
                """
                UPDATE profile_patches
                SET active = 0, last_confirmed_at = ?
                WHERE observer_id = ? AND observed_id = ? AND key = ? AND active = 1
                """,
                (now, patch.observer_id, patch.observed_id, patch.key),
            )

        self._connection.execute(
            """
            INSERT INTO profile_patches (
                key, value_json, reason, source, observer_id, observed_id,
                confidence, created_at, last_confirmed_at, active
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                patch.key,
                json.dumps(patch.value),
                patch.reason,
                patch.source,
                patch.observer_id,
                patch.observed_id,
                patch.confidence,
                patch.created_at.isoformat(),
                patch.last_confirmed_at.isoformat() if patch.last_confirmed_at else now,
                1 if patch.active else 0,
            ),
        )
        self._connection.commit()

    async def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            self._initialized = False

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS profile_patches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                value_json TEXT NOT NULL,
                reason TEXT NOT NULL,
                source TEXT NOT NULL,
                observer_id TEXT NOT NULL,
                observed_id TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL,
                last_confirmed_at TEXT,
                active INTEGER NOT NULL DEFAULT 1
            )
            """
        )
        self._connection.commit()
        self._initialized = True

    def _row_to_patch(self, row: tuple[Any, ...]) -> ProfilePatch:
        return ProfilePatch(
            key=row[0],
            value=json.loads(row[1]),
            reason=row[2],
            source=row[3],
            observer_id=row[4],
            observed_id=row[5],
            confidence=float(row[6]),
            created_at=datetime.fromisoformat(row[7]),
            last_confirmed_at=datetime.fromisoformat(row[8]) if row[8] else None,
            active=bool(row[9]),
        )

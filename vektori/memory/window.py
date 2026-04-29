"""Short-term rolling conversation window for the harness."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vektori.models.base import ChatModelProvider


def estimate_tokens(messages: list[dict[str, str]]) -> int:
    """Rough token estimate based on character count."""
    return sum(len(m.get("content", "")) for m in messages) // 4


@dataclass
class WindowState:
    recent_messages: list[dict[str, str]]
    rolling_summary: str
    estimated_tokens: int
    compaction_count: int


class MessageWindow:
    """Local conversation buffer with optional summarizing compaction."""

    def __init__(
        self,
        *,
        max_context_tokens: int = 12000,
        compaction_trigger_ratio: float = 0.8,
        keep_last_n_turns: int = 6,
        summary_max_tokens: int = 400,
    ) -> None:
        self.max_context_tokens = max_context_tokens
        self.compaction_trigger_ratio = compaction_trigger_ratio
        self.keep_last_n_turns = keep_last_n_turns
        self.summary_max_tokens = summary_max_tokens
        self._recent_messages: list[dict[str, str]] = []
        self._rolling_summary = ""
        self._compaction_count = 0

    def add(self, role: str, content: str) -> None:
        self._recent_messages.append({"role": role, "content": content})

    def snapshot(self) -> WindowState:
        return WindowState(
            recent_messages=list(self._recent_messages),
            rolling_summary=self._rolling_summary,
            estimated_tokens=self.estimated_tokens(),
            compaction_count=self._compaction_count,
        )

    def restore(self, state: WindowState) -> None:
        """Restore window state from a persisted snapshot."""
        self._recent_messages = list(state.recent_messages)
        self._rolling_summary = state.rolling_summary
        self._compaction_count = state.compaction_count

    def estimated_tokens(self) -> int:
        summary_tokens = len(self._rolling_summary) // 4
        return estimate_tokens(self._recent_messages) + summary_tokens

    async def compact(self, summarizer: ChatModelProvider) -> bool:
        if self.estimated_tokens() < self.max_context_tokens * self.compaction_trigger_ratio:
            return False

        keep_messages = self.keep_last_n_turns * 2
        if len(self._recent_messages) <= keep_messages:
            return False

        old_messages = self._recent_messages[:-keep_messages]
        kept_messages = self._recent_messages[-keep_messages:]
        old_text = "\n".join(
            f"{message['role']}: {message['content']}" for message in old_messages
        )

        prompt = (
            "Summarize the conversation state in this format:\n"
            "Conversation Summary\n"
            "- Active goals:\n"
            "- Confirmed preferences:\n"
            "- Open questions:\n"
            "- Constraints:\n"
            "- Recent commitments:\n\n"
            f"Conversation:\n{old_text}"
        )
        result = await summarizer.complete(
            [{"role": "user", "content": prompt}],
            max_tokens=self.summary_max_tokens,
            temperature=0.0,
        )
        if result.content:
            self._rolling_summary = result.content.strip()
        self._recent_messages = kept_messages
        self._compaction_count += 1
        return True

    def reset(self) -> None:
        self._recent_messages = []
        self._rolling_summary = ""
        self._compaction_count = 0


class SQLiteWindowStore:
    """
    SQLite-backed window snapshot store for resumable agent sessions (Phase 5).

    Each session_id maps to one row containing the full window state as JSON.
    The store is append-and-replace: saving overwrites the previous snapshot.

    Usage:
        store = SQLiteWindowStore("~/.vektori/windows.db")
        await store.save("session-123", agent.window.snapshot())
        # later...
        state = await store.load("session-123")
        if state:
            agent.window.restore(state)
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._conn: sqlite3.Connection | None = None
        self._initialized = False

    async def save(self, session_id: str, state: WindowState) -> None:
        await self._ensure_initialized()
        assert self._conn is not None
        payload = json.dumps({
            "recent_messages": state.recent_messages,
            "rolling_summary": state.rolling_summary,
            "compaction_count": state.compaction_count,
        })
        self._conn.execute(
            """
            INSERT INTO window_snapshots (session_id, payload)
            VALUES (?, ?)
            ON CONFLICT(session_id) DO UPDATE SET payload = excluded.payload,
                                                   updated_at = CURRENT_TIMESTAMP
            """,
            (session_id, payload),
        )
        self._conn.commit()

    async def load(self, session_id: str) -> WindowState | None:
        await self._ensure_initialized()
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT payload FROM window_snapshots WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        data: dict[str, Any] = json.loads(row[0])
        msgs = data.get("recent_messages", [])
        return WindowState(
            recent_messages=msgs,
            rolling_summary=data.get("rolling_summary", ""),
            estimated_tokens=estimate_tokens(msgs),
            compaction_count=data.get("compaction_count", 0),
        )

    async def delete(self, session_id: str) -> None:
        await self._ensure_initialized()
        assert self._conn is not None
        self._conn.execute(
            "DELETE FROM window_snapshots WHERE session_id = ?", (session_id,)
        )
        self._conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            self._initialized = False

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS window_snapshots (
                session_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.commit()
        self._initialized = True

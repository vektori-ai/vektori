"""Dataclasses for MemFS — the filesystem-native memory system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

NOTE_TYPES = ("semantic", "episodic", "procedural", "source")


@dataclass
class Note:
    id: str
    type: str
    title: str
    body: str
    path: str | None = None          # absolute path once persisted
    created: datetime | None = None
    when: datetime | None = None     # event time (bi-temporal with created)
    source: str | None = None        # provenance: conversation:…, file:…, agent:…
    tags: list[str] = field(default_factory=list)
    schema: int = 1


@dataclass
class Chunk:
    chunk_id: str                    # sha256(file_id + heading_path + text)[:24]
    file_id: str
    heading_path: str                # "Title > H2 > H3"
    text: str
    start_line: int
    end_line: int


@dataclass
class RecallItem:
    path: str
    title: str
    type: str
    snippet: str
    start_line: int
    end_line: int
    score: float
    signals: dict[str, float] = field(default_factory=dict)
    provenance: str | None = None
    note_id: str | None = None


@dataclass
class RecallResult:
    query: str
    items: list[RecallItem] = field(default_factory=list)

    @property
    def memory_found(self) -> bool:
        return bool(self.items)


@dataclass
class SyncReport:
    scanned: int = 0
    added: int = 0
    updated: int = 0
    removed: int = 0
    chunks_embedded: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class VerifyReport:
    files_checked: int = 0
    hash_drift: list[str] = field(default_factory=list)      # files whose index hash is stale
    broken_links: list[tuple[str, str]] = field(default_factory=list)  # (src path, missing slug)
    orphan_rows: int = 0                                     # index rows with no file on disk

    @property
    def clean(self) -> bool:
        return not (self.hash_drift or self.broken_links or self.orphan_rows)


@dataclass
class CompactReport:
    month: str = ""
    notes_compacted: int = 0
    digest_path: str | None = None
    archived: list[str] = field(default_factory=list)


@dataclass
class IngestReport:
    path: str
    note_path: str | None = None
    skipped: bool = False
    reason: str | None = None


def now_utc() -> datetime:
    from datetime import timezone
    return datetime.now(timezone.utc)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "__dict__"):
        return {k: to_jsonable(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    return obj

"""Dataclasses for the filesystem memory module."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FileChunk:
    path: str
    chunk_index: int
    text: str
    heading: str | None = None


@dataclass
class IngestResult:
    path: str
    chunks: int
    facts_inserted: int
    skipped: bool = False
    error: str | None = None

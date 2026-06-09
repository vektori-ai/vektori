"""FilesystemMemory — public API for file-based memory."""

from __future__ import annotations

import fnmatch
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from vektori.fsmemory.chunker import DocumentChunker
from vektori.fsmemory.extractor import DocumentExtractor
from vektori.fsmemory.models import IngestResult
from vektori.fsmemory.query import FSQuery
from vektori.fsmemory.store import DEFAULT_DB_PATH, FSStore, path_session_id
from vektori.models.factory import create_embedder, create_llm

logger = logging.getLogger(__name__)

DEFAULT_EXCLUDES = [
    "**/.git/**", "**/__pycache__/**", "**/*.pyc",
    "**/node_modules/**", "**/.venv/**", "**/venv/**",
    "**/*.db", "**/*.sqlite", "**/*.bin",
    "**/*.jpg", "**/*.jpeg", "**/*.png", "**/*.gif", "**/*.ico",
    "**/*.pdf", "**/*.zip", "**/*.tar.gz", "**/*.whl",
    "**/.env", "**/*.key", "**/*.pem", "**/*.lock",
]

MAX_FILE_SIZE_BYTES = 200 * 1024  # 200 KB


class FilesystemMemory:
    """
    Standalone file-based memory for AI agents.

    Ingests documents (markdown, text, any readable file) into a fact store.
    Completely separate from Vektori's conversation memory — pick one or use both independently.

    Usage:
        async with FilesystemMemory(user_id="alice") as fs:
            await fs.add_directory("~/notes", glob="**/*.md")
            results = await fs.search("project architecture")
    """

    def __init__(
        self,
        user_id: str,
        database_url: str | None = None,
        embedding_model: str = "openai:text-embedding-3-small",
        extraction_model: str = "openai:gpt-4o-mini",
        extract_facts: bool = True,
        exclude_patterns: list[str] | None = None,
        max_file_size_bytes: int = MAX_FILE_SIZE_BYTES,
    ) -> None:
        self.user_id = user_id
        self._database_url = database_url
        self._embedding_model = embedding_model
        self._extraction_model = extraction_model
        self._extract_facts = extract_facts
        self._exclude_patterns = (exclude_patterns or []) + DEFAULT_EXCLUDES
        self._max_file_size = max_file_size_bytes

        self._store: FSStore | None = None
        self._query: FSQuery | None = None
        self._chunker = DocumentChunker()
        self._extractor: DocumentExtractor | None = None
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return

        from vektori.storage.sqlite import SQLiteBackend

        db_path = (
            Path(self._database_url.replace("sqlite:///", ""))
            if self._database_url
            else DEFAULT_DB_PATH
        )

        backend = SQLiteBackend(self._database_url or f"sqlite:///{db_path}")
        await backend.initialize()

        self._store = FSStore(db=backend, db_path=db_path)
        await self._store.initialize()

        embedder = create_embedder(self._embedding_model)
        llm = create_llm(self._extraction_model)

        self._extractor = DocumentExtractor(llm=llm, embedder=embedder, extract=self._extract_facts)
        self._query = FSQuery(store=self._store, embedder=embedder)
        self._initialized = True

    async def close(self) -> None:
        if self._store:
            await self._store.close()
            self._initialized = False

    async def __aenter__(self) -> FilesystemMemory:
        await self.initialize()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    async def add_file(self, path: str) -> IngestResult:
        """Ingest a single file. Skips if content is unchanged since last ingest."""
        await self.initialize()
        assert self._store and self._extractor

        resolved = str(Path(path).expanduser().resolve())

        if self._is_excluded(resolved):
            return IngestResult(path=resolved, chunks=0, facts_inserted=0, skipped=True)

        try:
            p = Path(resolved)
            if not p.is_file():
                return IngestResult(path=resolved, chunks=0, facts_inserted=0, error="not a file")
            if p.stat().st_size > self._max_file_size:
                logger.warning("Skipping %s: exceeds max file size", resolved)
                return IngestResult(path=resolved, chunks=0, facts_inserted=0, skipped=True)

            content = p.read_text(encoding="utf-8", errors="ignore")
            content_hash = hashlib.sha256(content.encode()).hexdigest()

            stored_hash = await self._store.get_file_hash(resolved, self.user_id)
            if stored_hash == content_hash:
                return IngestResult(path=resolved, chunks=0, facts_inserted=0, skipped=True)

            if stored_hash is not None:
                await self._store.deactivate_by_path(resolved, self.user_id)

            chunks = self._chunker.chunk(resolved, content)
            if not chunks:
                return IngestResult(path=resolved, chunks=0, facts_inserted=0)

            session_id = path_session_id(resolved)
            total_inserted = 0
            for chunk in chunks:
                n = await self._extractor.process_chunk(
                    chunk=chunk,
                    user_id=self.user_id,
                    session_id=session_id,
                    db=self._store.db,
                )
                total_inserted += n

            await self._store.set_file_index(resolved, self.user_id, content_hash, total_inserted)
            return IngestResult(path=resolved, chunks=len(chunks), facts_inserted=total_inserted)

        except Exception as e:
            logger.error("Failed to ingest %s: %s", resolved, e)
            return IngestResult(path=resolved, chunks=0, facts_inserted=0, error=str(e))

    async def add_directory(
        self,
        path: str,
        glob: str = "**/*",
        max_files: int = 500,
    ) -> list[IngestResult]:
        """Crawl a directory and ingest all matching files."""
        await self.initialize()

        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            return [IngestResult(path=str(root), chunks=0, facts_inserted=0, error="not a directory")]

        files = [p for p in root.glob(glob) if p.is_file()][:max_files]
        results = []
        for f in files:
            result = await self.add_file(str(f))
            if not result.skipped or result.error:
                logger.debug(
                    "Ingested %s: %d chunks, %d facts",
                    result.path, result.chunks, result.facts_inserted,
                )
            results.append(result)
        return results

    async def search(
        self,
        query: str,
        path_prefix: str | None = None,
        limit: int = 10,
        mode: str = "semantic",
        since: datetime | None = None,
    ) -> dict[str, Any]:
        """Search filesystem memory. Returns same shape as Vektori.search()."""
        await self.initialize()
        assert self._query
        return await self._query.search(
            query=query,
            user_id=self.user_id,
            mode=mode,
            path_prefix=path_prefix,
            since=since,
            limit=limit,
        )

    async def delete_file(self, path: str) -> int:
        """Deactivate all facts from a file and remove it from the index."""
        await self.initialize()
        assert self._store
        resolved = str(Path(path).expanduser().resolve())
        count = await self._store.deactivate_by_path(resolved, self.user_id)
        await self._store.remove_from_index(resolved, self.user_id)
        return count

    async def list_files(self) -> list[str]:
        """Return all ingested file paths for this user."""
        await self.initialize()
        assert self._store
        return await self._store.list_paths(self.user_id)

    async def get_stats(self) -> dict[str, Any]:
        """Return ingestion stats: file count and total facts."""
        await self.initialize()
        assert self._store
        stats = await self._store.get_stats(self.user_id)
        return {**stats, "user_id": self.user_id}

    def _is_excluded(self, path: str) -> bool:
        for pattern in self._exclude_patterns:
            if fnmatch.fnmatch(path, pattern):
                return True
        return False

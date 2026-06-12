"""MemFS — filesystem-native memory for agents.

Canonical store = markdown files under a root directory. SQLite index = cache.
See docs/NEXT_GEN_FS_MEMORY_SYSTEM.md for the full architecture.

Usage:
    async with MemFS(root="~/notes-memory") as mem:
        await mem.remember("User prefers ruff over flake8", type="semantic",
                           tags=["python"])
        result = await mem.recall("linting preferences")
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from vektori.memfs.index import Index
from vektori.memfs.lifecycle import compact_month
from vektori.memfs.models import (
    CompactReport,
    IngestReport,
    Note,
    RecallResult,
    SyncReport,
    VerifyReport,
    now_utc,
    to_jsonable,
)
from vektori.memfs.notes import (
    atomic_write,
    content_hash,
    dump_note,
    extract_wikilinks,
    new_id,
    parse_note,
    slugify,
)
from vektori.memfs.retrieve import recall as _recall
from vektori.memfs.secrets import SecretsFoundError, scan_text

logger = logging.getLogger(__name__)

NOTE_DIRS = ("semantic", "episodic", "procedural", "sources")
MAX_SOURCE_BYTES = 200 * 1024
DEFAULT_INGEST_EXCLUDES = (
    ".git", "__pycache__", "node_modules", ".venv", "venv", ".memfs",
)


class MemFS:
    def __init__(
        self,
        root: str | Path,
        embedder: Any = None,            # None | "provider:model" str | EmbeddingProvider
        auto_commit: bool = False,
        secret_scan: bool = True,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self._embedder_spec = embedder
        self.embedder = None
        self.auto_commit = auto_commit
        self.secret_scan = secret_scan
        self.index: Index | None = None
        self._initialized = False

    # ── lifecycle ─────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        if self._initialized:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        for d in NOTE_DIRS:
            (self.root / d).mkdir(exist_ok=True)
        gitignore = self.root / ".gitignore"
        if not gitignore.exists():
            atomic_write(gitignore, ".memfs/\n")

        if isinstance(self._embedder_spec, str):
            from vektori.models.factory import create_embedder
            self.embedder = create_embedder(self._embedder_spec)
        else:
            self.embedder = self._embedder_spec  # provider instance or None

        model_name = (
            self._embedder_spec if isinstance(self._embedder_spec, str)
            else type(self.embedder).__name__ if self.embedder else "none"
        )
        self.index = Index(self.root / ".memfs" / "index.db", embed_model=model_name)
        self._initialized = True
        await self.sync()

    async def close(self) -> None:
        if self.index:
            self.index.close()
            self.index = None
        self._initialized = False

    async def __aenter__(self) -> MemFS:
        await self.initialize()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ── write path ────────────────────────────────────────────────────────

    async def remember(
        self,
        text: str,
        type: str = "semantic",
        title: str | None = None,
        tags: tuple[str, ...] | list[str] = (),
        when: datetime | None = None,
        source: str | None = None,
    ) -> Note:
        """Create one memory note as a markdown file and index it."""
        await self.initialize()
        if self.secret_scan:
            findings = scan_text(text)
            if findings:
                raise SecretsFoundError(findings)

        created = now_utc()
        title = title or _derive_title(text)
        note = Note(
            id=new_id(), type=type, title=title, body=text.strip(),
            created=created, when=when, source=source, tags=list(tags),
        )
        note_path = self._place(note, created)
        atomic_write(note_path, dump_note(note))
        note.path = str(note_path)
        self._index_one(note_path)
        await self._embed_missing()
        self._journal("remember", {"path": str(note_path), "type": type, "title": title})
        self._update_moc(note, note_path)
        self._git_commit("memory: add " + note_path.relative_to(self.root).as_posix())
        return note

    def _place(self, note: Note, created: datetime) -> Path:
        slug = slugify(note.title)
        if note.type == "episodic":
            month = created.strftime("%Y-%m")
            d = self.root / "episodic" / month
            name = created.strftime("%Y-%m-%d") + "-" + slug + ".md"
        else:
            d = self.root / ("sources" if note.type == "source" else note.type)
            name = slug + ".md"
        d.mkdir(parents=True, exist_ok=True)
        path = d / name
        n = 2
        while path.exists():
            path = d / (name[:-3] + "-" + str(n) + ".md")
            n += 1
        return path

    async def ingest_file(self, path: str | Path) -> IngestReport:
        """Derive a sources/ note from an external file (chunk-incremental on re-ingest)."""
        await self.initialize()
        src = Path(path).expanduser().resolve()
        if not src.is_file():
            return IngestReport(path=str(src), skipped=True, reason="not a file")
        if src.stat().st_size > MAX_SOURCE_BYTES:
            return IngestReport(path=str(src), skipped=True, reason="exceeds 200KB cap")
        try:
            text = src.read_text(encoding="utf-8", errors="ignore")
        except OSError as e:
            return IngestReport(path=str(src), skipped=True, reason=str(e))
        if not text.strip():
            return IngestReport(path=str(src), skipped=True, reason="empty")
        if self.secret_scan and scan_text(text):
            return IngestReport(path=str(src), skipped=True, reason="secrets detected")

        digest = content_hash(text)
        provenance = "file:" + str(src) + "#sha256=" + digest[:16]
        slug = slugify(src.stem)
        note_path = self.root / "sources" / (slug + ".md")

        if note_path.exists():
            existing = parse_note(note_path)
            if existing.source == provenance:
                return IngestReport(path=str(src), note_path=str(note_path),
                                    skipped=True, reason="unchanged")
            stable_id = existing.id   # keep id ⇒ unchanged chunks keep embeddings
        else:
            stable_id = new_id()

        note = Note(
            id=stable_id, type="source", title=src.name, body=text,
            created=now_utc(), source=provenance, tags=["ingested"],
        )
        atomic_write(note_path, dump_note(note))
        self._index_one(note_path)
        await self._embed_missing()
        self._journal("ingest", {"src": str(src), "note": str(note_path)})
        return IngestReport(path=str(src), note_path=str(note_path))

    async def ingest_directory(self, path: str | Path, glob: str = "**/*.md") -> list[IngestReport]:
        await self.initialize()
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            return [IngestReport(path=str(root), skipped=True, reason="not a directory")]
        reports = []
        for f in sorted(root.glob(glob)):
            if not f.is_file():
                continue
            if any(part in DEFAULT_INGEST_EXCLUDES for part in f.parts):
                continue
            reports.append(await self.ingest_file(f))
        return reports

    # ── read path ─────────────────────────────────────────────────────────

    async def recall(
        self,
        query: str,
        k: int = 8,
        types: list[str] | None = None,
        since: datetime | None = None,
        expand_links: bool = True,
        include_archive: bool = False,
    ) -> RecallResult:
        await self.initialize()
        result = await _recall(
            self.index, query, embedder=self.embedder, k=k, types=types,
            since=since, expand_links=expand_links, include_archive=include_archive,
        )
        self._journal("recall", {
            "query": query, "k": k,
            "hits": [{"path": i.path, "score": round(i.score, 5)} for i in result.items],
        })
        return result

    async def read(self, slug_or_id: str) -> Note | None:
        await self.initialize()
        for p in self.root.rglob(slug_or_id + ".md"):
            return parse_note(p)
        row = self.index.conn.execute(
            "SELECT path FROM files WHERE id=?", (slug_or_id,)
        ).fetchone()
        return parse_note(Path(row["path"])) if row else None

    def orient(self) -> str:
        """MEMORY.md content for prompt injection; empty string if absent."""
        moc = self.root / "MEMORY.md"
        return moc.read_text(encoding="utf-8") if moc.exists() else ""

    # ── maintenance ───────────────────────────────────────────────────────

    async def sync(self) -> SyncReport:
        """Reconcile files ↔ index. Files win, always."""
        await self.initialize()
        report = SyncReport()
        seen: set[str] = set()
        for p in self._note_files():
            report.scanned += 1
            seen.add(str(p))
            try:
                if self.index.needs_update(p):
                    existed = self.index.file_state(str(p)) is not None
                    self._index_one(p)
                    report.updated += int(existed)
                    report.added += int(not existed)
            except Exception as e:  # one bad file must not poison the sync
                report.errors.append(str(p) + ": " + str(e))
        for stale in self.index.indexed_paths() - seen:
            self.index.remove_file(stale)
            report.removed += 1
        report.chunks_embedded = await self._embed_missing()
        self._journal("sync", to_jsonable(report))
        return report

    async def compact(self, month: str | None = None, llm: Any = None) -> CompactReport:
        await self.initialize()
        if month is None:
            month = now_utc().strftime("%Y-%m")
        report = compact_month(self.root, month, llm=llm)
        await self.sync()
        self._journal("compact", to_jsonable(report))
        self._git_commit("memory: compact episodic/" + month)
        return report

    async def forget(self, slug_or_id: str) -> list[str]:
        """Real deletion: remove matching note files, then reindex. No tombstones."""
        await self.initialize()
        removed: list[str] = []
        targets = list(self.root.rglob(slug_or_id + ".md"))
        if not targets:
            row = self.index.conn.execute(
                "SELECT path FROM files WHERE id=?", (slug_or_id,)
            ).fetchone()
            if row:
                targets = [Path(row["path"])]
        for p in targets:
            if p.is_file() and ".memfs" not in p.parts:
                p.unlink()
                removed.append(str(p))
        await self.sync()
        self._journal("forget", {"target": slug_or_id, "removed": removed})
        if removed:
            self._git_commit("memory: forget " + slug_or_id)
        return removed

    async def verify(self) -> VerifyReport:
        """fsck for memory: hash drift, broken wikilinks, orphan index rows."""
        await self.initialize()
        report = VerifyReport()
        on_disk: set[str] = set()
        stems: set[str] = set()
        notes = []
        for p in self._note_files():
            on_disk.add(str(p))
            stems.add(p.stem)
            report.files_checked += 1
            row = self.index.file_state(str(p))
            text = p.read_text(encoding="utf-8", errors="ignore")
            if row is None or row["content_hash"] != content_hash(text):
                report.hash_drift.append(str(p))
            notes.append((p, text))
        for p, text in notes:
            for slug in extract_wikilinks(text):
                if slug not in stems:
                    report.broken_links.append((str(p), slug))
        report.orphan_rows = len(self.index.indexed_paths() - on_disk)
        return report

    def stats(self) -> dict:
        if not self.index:
            return {}
        return {**self.index.stats(), "root": str(self.root)}

    # ── internals ─────────────────────────────────────────────────────────

    def _note_files(self):
        for d in NOTE_DIRS + ("archive",):
            base = self.root / d
            if base.is_dir():
                yield from sorted(base.rglob("*.md"))

    def _index_one(self, path: Path) -> None:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        note = parse_note(path, raw)
        self.index.upsert_note(note, raw)

    async def _embed_missing(self) -> int:
        """Embed only chunks that lack vectors — the incremental-cost guarantee."""
        if self.embedder is None:
            return 0
        rows = self.index.missing_embeddings()
        if not rows:
            return 0
        # contextual embedding: heading path prefixed to chunk text
        texts = [r["heading_path"] + "\n" + r["text"] for r in rows]
        vecs = await self.embedder.embed_batch(texts)
        self.index.store_embeddings({r["chunk_id"]: v for r, v in zip(rows, vecs)})
        return len(rows)

    def _journal(self, op: str, payload: dict) -> None:
        try:
            entry = {"ts": now_utc().isoformat(), "op": op, **payload}
            jpath = self.root / ".memfs" / "journal.jsonl"
            jpath.parent.mkdir(parents=True, exist_ok=True)
            with jpath.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except OSError as e:
            logger.warning("journal write failed: %s", e)

    def _update_moc(self, note: Note, path: Path) -> None:
        """Append a one-line pointer to MEMORY.md (the always-loaded map of content)."""
        moc = self.root / "MEMORY.md"
        rel = path.relative_to(self.root).as_posix()
        line = "- [" + note.title + "](" + rel + ") — " + note.type
        existing = moc.read_text(encoding="utf-8") if moc.exists() else "# Memory Index\n"
        if rel not in existing:
            atomic_write(moc, existing.rstrip("\n") + "\n" + line + "\n")

    def _git_commit(self, message: str) -> None:
        if not self.auto_commit:
            return
        try:
            subprocess.run(["git", "-C", str(self.root), "add", "-A"],
                           capture_output=True, timeout=10, check=False)
            subprocess.run(["git", "-C", str(self.root), "commit", "-m", message],
                           capture_output=True, timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired) as e:
            logger.warning("auto-commit failed: %s", e)


def _derive_title(text: str, max_words: int = 8) -> str:
    first = text.strip().splitlines()[0].lstrip("#").strip()
    words = first.split()
    return " ".join(words[:max_words]) or "untitled"

"""SQLite index over the note tree — a disposable, rebuildable cache.

Invariant: nothing in here is canonical. Deleting index.db and running sync()
must reproduce identical retrieval behavior (modulo access counters).
Synchronous sqlite3 by design: ops are local and sub-ms; the public MemFS API
wraps this behind async methods for ecosystem parity.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from vektori.memfs.models import Chunk, Note
from vektori.memfs.notes import chunk_note, content_hash, extract_wikilinks

INDEX_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    id TEXT NOT NULL,
    type TEXT NOT NULL,
    title TEXT NOT NULL,
    mtime_ns INTEGER NOT NULL,
    size INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    when_ts TEXT,
    created_ts TEXT,
    source TEXT,
    tags TEXT DEFAULT (json_array()),
    access_count INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_files_id ON files (id);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    file_id TEXT NOT NULL,
    heading_path TEXT NOT NULL,
    text TEXT NOT NULL,
    start_line INTEGER,
    end_line INTEGER
);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks (file_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, heading_path, chunk_id UNINDEXED
);
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id TEXT PRIMARY KEY,
    vec BLOB NOT NULL,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS links (
    src_file_id TEXT NOT NULL,
    dst_slug TEXT NOT NULL,
    PRIMARY KEY (src_file_id, dst_slug)
);
"""


class Index:
    def __init__(self, db_path: Path, embed_model: str = "none") -> None:
        self.db_path = db_path
        self.embed_model = embed_model
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(_SCHEMA)
        self._check_version()
        self._vec_cache: tuple[list[str], np.ndarray] | None = None

    def _check_version(self) -> None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", ("index_version",)).fetchone()
        stored = row["value"] if row else None
        current = str(INDEX_VERSION) + ":" + self.embed_model
        if stored is not None and stored != current:
            # schema or embed model changed: wipe; files are canonical, sync() rebuilds
            for t in ("files", "chunks", "embeddings", "links"):
                self.conn.execute("DELETE FROM " + t)
            self.conn.execute("DELETE FROM chunks_fts")
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT (key) DO UPDATE SET value=excluded.value",
            ("index_version", current),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ── sync ──────────────────────────────────────────────────────────────

    def file_state(self, path: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM files WHERE path=?", (path,)).fetchone()

    def needs_update(self, path: Path) -> bool:
        row = self.file_state(str(path))
        if row is None:
            return True
        st = path.stat()
        if row["mtime_ns"] == st.st_mtime_ns and row["size"] == st.st_size:
            return False
        text = path.read_text(encoding="utf-8", errors="ignore")
        return content_hash(text) != row["content_hash"]

    def upsert_note(self, note: Note, raw_text: str) -> tuple[list[Chunk], list[str]]:
        """Index one note. Returns (new_chunks_needing_embedding, removed_chunk_ids)."""
        path = Path(note.path)
        st = path.stat()
        chunks = chunk_note(note)
        new_ids = {c.chunk_id for c in chunks}

        old = {r["chunk_id"] for r in self.conn.execute(
            "SELECT chunk_id FROM chunks WHERE file_id=?", (note.id,))}
        removed = sorted(old - new_ids)
        added = [c for c in chunks if c.chunk_id not in old]

        cur = self.conn.cursor()
        for cid in removed:
            cur.execute("DELETE FROM chunks WHERE chunk_id=?", (cid,))
            cur.execute("DELETE FROM chunks_fts WHERE chunk_id=?", (cid,))
            cur.execute("DELETE FROM embeddings WHERE chunk_id=?", (cid,))
        for c in added:
            cur.execute(
                "INSERT OR REPLACE INTO chunks (chunk_id, file_id, heading_path, text, start_line, end_line) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (c.chunk_id, c.file_id, c.heading_path, c.text, c.start_line, c.end_line),
            )
            cur.execute(
                "INSERT INTO chunks_fts (text, heading_path, chunk_id) VALUES (?, ?, ?)",
                (c.text, c.heading_path, c.chunk_id),
            )
        cur.execute(
            "INSERT INTO files (path, id, type, title, mtime_ns, size, content_hash, when_ts, created_ts, source, tags) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (path) DO UPDATE SET id=excluded.id, type=excluded.type, title=excluded.title, "
            "mtime_ns=excluded.mtime_ns, size=excluded.size, content_hash=excluded.content_hash, "
            "when_ts=excluded.when_ts, created_ts=excluded.created_ts, source=excluded.source, tags=excluded.tags",
            (
                str(path), note.id, note.type, note.title, st.st_mtime_ns, st.st_size,
                content_hash(raw_text),
                note.when.isoformat() if note.when else None,
                note.created.isoformat() if note.created else None,
                note.source, json.dumps(note.tags),
            ),
        )
        cur.execute("DELETE FROM links WHERE src_file_id=?", (note.id,))
        for slug in set(extract_wikilinks(note.body)):
            cur.execute(
                "INSERT OR IGNORE INTO links (src_file_id, dst_slug) VALUES (?, ?)",
                (note.id, slug),
            )
        self.conn.commit()
        self._vec_cache = None
        return added, removed

    def remove_file(self, path: str) -> None:
        row = self.file_state(path)
        if row is None:
            return
        fid = row["id"]
        cur = self.conn.cursor()
        for r in cur.execute("SELECT chunk_id FROM chunks WHERE file_id=?", (fid,)).fetchall():
            cur.execute("DELETE FROM chunks_fts WHERE chunk_id=?", (r["chunk_id"],))
            cur.execute("DELETE FROM embeddings WHERE chunk_id=?", (r["chunk_id"],))
        cur.execute("DELETE FROM chunks WHERE file_id=?", (fid,))
        cur.execute("DELETE FROM links WHERE src_file_id=?", (fid,))
        cur.execute("DELETE FROM files WHERE path=?", (path,))
        self.conn.commit()
        self._vec_cache = None

    def store_embeddings(self, chunk_vecs: dict[str, list[float]]) -> None:
        cur = self.conn.cursor()
        for cid, vec in chunk_vecs.items():
            arr = np.asarray(vec, dtype=np.float32)
            cur.execute(
                "INSERT OR REPLACE INTO embeddings (chunk_id, vec, model, dim) VALUES (?, ?, ?, ?)",
                (cid, arr.tobytes(), self.embed_model, len(arr)),
            )
        self.conn.commit()
        self._vec_cache = None

    def indexed_paths(self) -> set[str]:
        return {r["path"] for r in self.conn.execute("SELECT path FROM files")}

    # ── search ────────────────────────────────────────────────────────────

    def search_bm25(self, query: str, limit: int = 40,
                    types: list[str] | None = None) -> list[tuple[str, float]]:
        """FTS5 BM25. Returns [(chunk_id, rank_score)] best first."""
        q = _fts_escape(query)
        if not q:
            return []
        sql = (
            "SELECT f.chunk_id AS chunk_id, bm25(chunks_fts) AS rank "
            "FROM chunks_fts f JOIN chunks c ON c.chunk_id = f.chunk_id "
            "JOIN files fl ON fl.id = c.file_id "
            "WHERE chunks_fts MATCH ?"
        )
        params: list = [q]
        if types:
            sql += " AND fl.type IN (" + ",".join("?" * len(types)) + ")"
            params.extend(types)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)
        try:
            rows = self.conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            return []
        # bm25() returns lower=better; negate so higher=better
        return [(r["chunk_id"], -float(r["rank"])) for r in rows]

    def search_vector(self, query_vec: list[float], limit: int = 40,
                      types: list[str] | None = None) -> list[tuple[str, float]]:
        """Brute-force cosine over a cached float32 matrix. v1 ceiling ~1e5 chunks;
        swap point for sqlite-vec/FAISS lives behind this method signature."""
        ids, mat = self._vectors(types)
        if mat is None or not len(ids):
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        qn = np.linalg.norm(q)
        if qn == 0:
            return []
        norms = np.linalg.norm(mat, axis=1)
        norms[norms == 0] = 1e-9
        sims = (mat @ q) / (norms * qn)
        top = np.argsort(-sims)[:limit]
        return [(ids[i], float(sims[i])) for i in top]

    def _vectors(self, types: list[str] | None = None):
        if types is None and self._vec_cache is not None:
            return self._vec_cache
        sql = (
            "SELECT e.chunk_id, e.vec, e.dim FROM embeddings e "
            "JOIN chunks c ON c.chunk_id = e.chunk_id "
            "JOIN files f ON f.id = c.file_id"
        )
        params: list = []
        if types:
            sql += " WHERE f.type IN (" + ",".join("?" * len(types)) + ")"
            params.extend(types)
        rows = self.conn.execute(sql, params).fetchall()
        if not rows:
            return [], None
        ids = [r["chunk_id"] for r in rows]
        mat = np.vstack([np.frombuffer(r["vec"], dtype=np.float32, count=r["dim"]) for r in rows])
        if types is None:
            self._vec_cache = (ids, mat)
        return ids, mat

    def chunks_by_ids(self, chunk_ids: list[str]) -> dict[str, sqlite3.Row]:
        if not chunk_ids:
            return {}
        ph = ",".join("?" * len(chunk_ids))
        rows = self.conn.execute(
            "SELECT c.*, f.path, f.title, f.type, f.source, f.when_ts, f.created_ts, f.access_count "
            "FROM chunks c JOIN files f ON f.id = c.file_id WHERE c.chunk_id IN (" + ph + ")",
            chunk_ids,
        ).fetchall()
        return {r["chunk_id"]: r for r in rows}

    def linked_neighbors(self, file_ids: list[str]) -> list[str]:
        """1-hop wikilink expansion: file_ids of notes linked from the given notes
        (slug resolved against file path stems) plus notes linking TO them."""
        if not file_ids:
            return []
        ph = ",".join("?" * len(file_ids))
        out: set[str] = set()
        # outgoing: dst_slug -> files whose path stem matches
        rows = self.conn.execute(
            "SELECT l.dst_slug FROM links l WHERE l.src_file_id IN (" + ph + ")", file_ids
        ).fetchall()
        for r in rows:
            slug = r["dst_slug"]
            hit = self.conn.execute(
                "SELECT id FROM files WHERE path LIKE ? LIMIT 1", ("%/" + slug + ".md",)
            ).fetchone()
            if hit:
                out.add(hit["id"])
        # incoming: notes whose links resolve to one of our stems
        rows = self.conn.execute(
            "SELECT f.path, f.id FROM files f WHERE f.id IN (" + ph + ")", file_ids
        ).fetchall()
        for r in rows:
            stem = Path(r["path"]).stem
            for src in self.conn.execute(
                "SELECT src_file_id FROM links WHERE dst_slug = ?", (stem,)
            ).fetchall():
                out.add(src["src_file_id"])
        return [fid for fid in out if fid not in set(file_ids)]

    def chunks_for_files(self, file_ids: list[str], per_file: int = 1) -> list[str]:
        out: list[str] = []
        for fid in file_ids:
            rows = self.conn.execute(
                "SELECT chunk_id FROM chunks WHERE file_id=? ORDER BY start_line LIMIT ?",
                (fid, per_file),
            ).fetchall()
            out.extend(r["chunk_id"] for r in rows)
        return out

    def bump_access(self, file_paths: list[str]) -> None:
        cur = self.conn.cursor()
        for p in set(file_paths):
            cur.execute("UPDATE files SET access_count = access_count + 1 WHERE path=?", (p,))
        self.conn.commit()

    def missing_embeddings(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT c.chunk_id, c.heading_path, c.text FROM chunks c "
            "LEFT JOIN embeddings e ON e.chunk_id = c.chunk_id WHERE e.chunk_id IS NULL"
        ).fetchall()

    def stats(self) -> dict:
        def g(q: str) -> int:
            return self.conn.execute(q).fetchone()[0]

        return {
            "files": g("SELECT COUNT(*) FROM files"),
            "chunks": g("SELECT COUNT(*) FROM chunks"),
            "embeddings": g("SELECT COUNT(*) FROM embeddings"),
            "links": g("SELECT COUNT(*) FROM links"),
            "by_type": {r["type"]: r["n"] for r in self.conn.execute(
                "SELECT type, COUNT(*) n FROM files GROUP BY type")},
        }


def _fts_escape(query: str) -> str:
    """Quote each term — agents pass code identifiers that break FTS5 syntax."""
    terms = [t for t in query.replace('"', " ").split() if t]
    return " OR ".join('"' + t + '"' for t in terms)

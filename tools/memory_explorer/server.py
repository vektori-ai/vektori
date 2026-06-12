"""Vektori Memory Explorer — local, read-only inspector for both memory engines.

Serves a single-page UI over:
  - the sentex 3-layer store (sentences -> facts -> episodes) in ~/.vektori/vektori.db
  - the memfs file tree (markdown canonical) in ~/.vektori/memfs/<ns>/

Both stores are opened read-only at the SQLite level (mode=ro); memfs notes are
read straight from disk so the explorer works even without an index.db.

Usage:
    python -m tools.memory_explorer.server [--db PATH] [--memfs-root PATH] [--port 8765]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vektori.memfs.notes import extract_wikilinks, parse_note  # noqa: E402

DEFAULT_DB = Path.home() / ".vektori" / "vektori.db"
DEFAULT_MEMFS = Path.home() / ".vektori" / "memfs" / "default"
HTML_PATH = Path(__file__).with_name("memory_explorer.html")
NOTE_DIRS = ("semantic", "episodic", "procedural", "sources", "archive")


def open_ro(db_path: Path) -> sqlite3.Connection | None:
    if not db_path.is_file():
        return None
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def rows(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[dict]:
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def one(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> dict | None:
    r = conn.execute(sql, params).fetchone()
    return dict(r) if r else None


# ── sentex (3-layer store) ──────────────────────────────────────────────────

def sx_users(conn: sqlite3.Connection) -> list[str]:
    return [r["user_id"] for r in conn.execute(
        "SELECT user_id, MAX(created_at) AS last FROM ("
        " SELECT user_id, created_at FROM facts"
        " UNION ALL SELECT user_id, created_at FROM sentences"
        " UNION ALL SELECT user_id, created_at FROM episodes)"
        " WHERE user_id IS NOT NULL AND TRIM(user_id) != ''"
        " GROUP BY user_id ORDER BY last DESC")]


def sx_counts(conn: sqlite3.Connection, user: str) -> dict:
    c = {}
    for key, sql in [
        ("facts_active", "SELECT COUNT(*) FROM facts WHERE user_id=? AND is_active=1"),
        ("facts_total", "SELECT COUNT(*) FROM facts WHERE user_id=?"),
        ("episodes", "SELECT COUNT(*) FROM episodes WHERE user_id=? AND is_active=1"),
        ("sentences", "SELECT COUNT(*) FROM sentences WHERE user_id=?"),
        ("sessions", "SELECT COUNT(*) FROM sessions WHERE user_id=?"),
    ]:
        c[key] = conn.execute(sql, (user,)).fetchone()[0]
    return c


def fact_dict(r: dict) -> dict:
    meta = {}
    try:
        meta = json.loads(r.get("metadata") or "{}")
    except json.JSONDecodeError:
        pass
    return {
        "id": r["id"], "text": r["text"], "subject": r.get("subject"),
        "confidence": r.get("confidence"), "is_active": bool(r.get("is_active")),
        "superseded_by": r.get("superseded_by"), "session_id": r.get("session_id"),
        "event_time": r.get("event_time"), "created_at": r.get("created_at"),
        "mentions": meta.get("mentions", 1), "source_type": meta.get("source_type"),
    }


def sx_facts(conn: sqlite3.Connection, user: str, state: str, q: str) -> list[dict]:
    sql = ("SELECT id, text, subject, confidence, is_active, superseded_by,"
           " session_id, event_time, created_at, metadata FROM facts WHERE user_id=?")
    params: list = [user]
    if state == "active":
        sql += " AND is_active=1"
    elif state == "superseded":
        sql += " AND is_active=0"
    if q:
        sql += " AND text LIKE ?"
        params.append(f"%{q}%")
    sql += " ORDER BY datetime(created_at) DESC LIMIT 500"
    return [fact_dict(r) for r in rows(conn, sql, tuple(params))]


def sx_fact_detail(conn: sqlite3.Connection, fact_id: str) -> dict | None:
    fact = one(conn, "SELECT * FROM facts WHERE id=?", (fact_id,))
    if not fact:
        return None
    sentences = rows(conn,
        "SELECT s.id, s.text, s.role, s.turn_number, s.session_id, s.created_at"
        " FROM fact_sources fs JOIN sentences s ON s.id = fs.sentence_id"
        " WHERE fs.fact_id=? ORDER BY s.turn_number, s.sentence_index", (fact_id,))
    episodes = rows(conn,
        "SELECT e.id, e.text, e.session_id, e.created_at"
        " FROM episode_facts ef JOIN episodes e ON e.id = ef.episode_id"
        " WHERE ef.fact_id=?", (fact_id,))
    # supersession chain, both directions
    newer = rows(conn, "SELECT id, text, created_at, is_active FROM facts"
                       " WHERE id=(SELECT superseded_by FROM facts WHERE id=?)", (fact_id,))
    older = rows(conn, "SELECT id, text, created_at, is_active FROM facts"
                       " WHERE superseded_by=?", (fact_id,))
    related = rows(conn,
        "SELECT f.id, f.text, fe.weight FROM fact_edges fe"
        " JOIN facts f ON f.id = CASE WHEN fe.source_id=? THEN fe.target_id ELSE fe.source_id END"
        " WHERE fe.source_id=? OR fe.target_id=? ORDER BY fe.weight DESC LIMIT 10",
        (fact_id, fact_id, fact_id))
    return {"fact": fact_dict(fact), "sentences": sentences, "episodes": episodes,
            "superseded_by_fact": newer, "supersedes_facts": older, "related": related}


def sx_episodes(conn: sqlite3.Connection, user: str) -> list[dict]:
    eps = rows(conn,
        "SELECT e.id, e.text, e.session_id, e.created_at, e.is_active,"
        " s.started_at AS session_date,"
        " (SELECT COUNT(*) FROM episode_facts ef WHERE ef.episode_id = e.id) AS fact_count"
        " FROM episodes e LEFT JOIN sessions s ON s.id = e.session_id"
        " WHERE e.user_id=? ORDER BY datetime(e.created_at) DESC LIMIT 500", (user,))
    return eps


def sx_episode_detail(conn: sqlite3.Connection, ep_id: str) -> dict | None:
    ep = one(conn, "SELECT * FROM episodes WHERE id=?", (ep_id,))
    if not ep:
        return None
    facts = rows(conn,
        "SELECT f.id, f.text, f.confidence, f.is_active, f.metadata FROM episode_facts ef"
        " JOIN facts f ON f.id = ef.fact_id WHERE ef.episode_id=?", (ep_id,))
    return {"episode": ep, "facts": [fact_dict(f) for f in facts]}


def sx_sessions(conn: sqlite3.Connection, user: str) -> list[dict]:
    return rows(conn,
        "SELECT s.id, s.started_at, s.ended_at,"
        " (SELECT COUNT(*) FROM sentences sn WHERE sn.session_id = s.id) AS sentence_count,"
        " (SELECT COUNT(*) FROM facts f WHERE f.session_id = s.id) AS fact_count"
        " FROM sessions s WHERE s.user_id=?"
        " ORDER BY datetime(s.started_at) DESC LIMIT 300", (user,))


def sx_session_detail(conn: sqlite3.Connection, session_id: str) -> dict:
    sentences = rows(conn,
        "SELECT id, text, role, turn_number, sentence_index, created_at FROM sentences"
        " WHERE session_id=? ORDER BY turn_number, sentence_index", (session_id,))
    # reverse provenance: which sentences produced facts
    fact_links = rows(conn,
        "SELECT fs.sentence_id, f.id AS fact_id, f.text, f.is_active FROM fact_sources fs"
        " JOIN facts f ON f.id = fs.fact_id"
        " WHERE fs.sentence_id IN (SELECT id FROM sentences WHERE session_id=?)", (session_id,))
    by_sentence: dict[str, list] = {}
    for fl in fact_links:
        by_sentence.setdefault(fl["sentence_id"], []).append(
            {"fact_id": fl["fact_id"], "text": fl["text"], "is_active": bool(fl["is_active"])})
    return {"sentences": sentences, "facts_by_sentence": by_sentence}


def sx_profile(conn: sqlite3.Connection, user: str, limit: int = 15) -> list[dict]:
    """Living profile: active facts ranked by confidence, then recency."""
    facts = [fact_dict(r) for r in rows(conn,
        "SELECT id, text, subject, confidence, is_active, superseded_by, session_id,"
        " event_time, created_at, metadata FROM facts WHERE user_id=? AND is_active=1",
        (user,))]
    facts.sort(key=lambda f: (-(f["confidence"] or 0), f["created_at"] or ""), reverse=False)
    return facts[:limit]


def sx_search(conn: sqlite3.Connection, user: str, q: str) -> dict:
    like = f"%{q}%"
    return {
        "facts": [fact_dict(r) for r in rows(conn,
            "SELECT id, text, subject, confidence, is_active, superseded_by, session_id,"
            " event_time, created_at, metadata FROM facts WHERE user_id=? AND text LIKE ?"
            " ORDER BY is_active DESC, datetime(created_at) DESC LIMIT 25", (user, like))],
        "episodes": rows(conn,
            "SELECT id, text, session_id, created_at FROM episodes"
            " WHERE user_id=? AND text LIKE ? ORDER BY datetime(created_at) DESC LIMIT 15",
            (user, like)),
        "sentences": rows(conn,
            "SELECT id, text, role, session_id, turn_number, created_at FROM sentences"
            " WHERE user_id=? AND text LIKE ? ORDER BY datetime(created_at) DESC LIMIT 25",
            (user, like)),
    }


# ── memfs (file tree) ───────────────────────────────────────────────────────

class MemfsView:
    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def ok(self) -> bool:
        return self.root.is_dir()

    def _files(self) -> list[Path]:
        out: list[Path] = []
        for d in NOTE_DIRS:
            base = self.root / d
            if base.is_dir():
                out.extend(sorted(base.rglob("*.md")))
        return out

    def tree(self) -> dict:
        notes = []
        for p in self._files():
            note = parse_note(p)
            rel = p.relative_to(self.root).as_posix()
            notes.append({
                "path": rel, "title": note.title, "type": note.type,
                "tags": note.tags, "created": str(note.created or ""),
                "dir": rel.split("/")[0], "links_out": len(extract_wikilinks(note.body)),
                "size": p.stat().st_size,
            })
        moc = self.root / "MEMORY.md"
        return {"notes": notes,
                "memory_md": moc.read_text(encoding="utf-8") if moc.exists() else ""}

    def note(self, rel_path: str) -> dict | None:
        p = (self.root / rel_path).resolve()
        if not p.is_file() or not p.is_relative_to(self.root):
            return None
        raw = p.read_text(encoding="utf-8", errors="ignore")
        note = parse_note(p, raw)
        links_out = extract_wikilinks(note.body)
        stems = {f.stem: f.relative_to(self.root).as_posix() for f in self._files()}
        backlinks = []
        for f in self._files():
            if f == p:
                continue
            body = f.read_text(encoding="utf-8", errors="ignore")
            if p.stem in extract_wikilinks(body):
                backlinks.append(f.relative_to(self.root).as_posix())
        return {
            "path": rel_path, "title": note.title, "type": note.type, "id": note.id,
            "tags": note.tags, "created": str(note.created or ""),
            "when": str(note.when or ""), "source": note.source, "body": note.body,
            "links_out": [{"slug": s, "path": stems.get(s)} for s in links_out],
            "backlinks": backlinks, "raw": raw,
        }

    def journal(self, limit: int = 200) -> list[dict]:
        jpath = self.root / ".memfs" / "journal.jsonl"
        if not jpath.is_file():
            return []
        lines = jpath.read_text(encoding="utf-8").strip().splitlines()
        out = []
        for line in reversed(lines[-limit:]):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def stats(self) -> dict:
        idx = self.root / ".memfs" / "index.db"
        s: dict = {"root": str(self.root), "notes": len(self._files()),
                   "index_present": idx.is_file()}
        conn = open_ro(idx)
        if conn:
            try:
                s["chunks"] = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                s["embeddings"] = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
                s["embed_model"] = (conn.execute(
                    "SELECT value FROM meta WHERE key='embed_model'").fetchone() or ["none"])[0]
            finally:
                conn.close()
        return s

    def search(self, q: str) -> list[dict]:
        """FTS over the index when present; substring scan over files otherwise."""
        idx = open_ro(self.root / ".memfs" / "index.db")
        if idx:
            try:
                safe = " ".join('"' + t.replace('"', '') + '"' for t in q.split())
                hits = rows(idx,
                    "SELECT c.text, c.heading_path, c.start_line, f.path, f.title, f.type"
                    " FROM chunks_fts ft JOIN chunks c ON c.rowid = ft.rowid"
                    " JOIN files f ON f.id = c.file_id"
                    " WHERE chunks_fts MATCH ? ORDER BY rank LIMIT 25", (safe,))
                for h in hits:
                    h["path"] = str(Path(h["path"]).relative_to(self.root)) \
                        if h["path"].startswith(str(self.root)) else h["path"]
                return hits
            except sqlite3.OperationalError:
                pass
            finally:
                idx.close()
        ql = q.lower()
        out = []
        for p in self._files():
            text = p.read_text(encoding="utf-8", errors="ignore")
            pos = text.lower().find(ql)
            if pos >= 0:
                line_no = text.count("\n", 0, pos) + 1
                snippet = text[max(0, pos - 60):pos + 140].strip()
                note = parse_note(p, text)
                out.append({"text": snippet, "heading_path": "", "start_line": line_no,
                            "path": p.relative_to(self.root).as_posix(),
                            "title": note.title, "type": note.type})
        return out[:25]


# ── HTTP plumbing ───────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    db_path: Path = DEFAULT_DB
    memfs: MemfsView = MemfsView(DEFAULT_MEMFS)

    def log_message(self, *_: object) -> None:
        pass

    def _json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        qs = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            self._route(url.path, qs)
        except BrokenPipeError:
            pass
        except Exception as e:  # surface, don't crash the thread
            self._json({"error": str(e)}, 500)

    def _route(self, path: str, qs: dict[str, str]) -> None:
        if path in ("/", "/index.html"):
            body = HTML_PATH.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if not path.startswith("/api/"):
            self._json({"error": "not found"}, 404)
            return

        conn = open_ro(self.db_path)
        try:
            route = path[len("/api/"):]
            user = qs.get("user", "")
            if route == "meta":
                self._json({
                    "db": {"path": str(self.db_path), "ok": conn is not None,
                           "users": sx_users(conn) if conn else []},
                    "memfs": {**self.memfs.stats(), "ok": self.memfs.ok},
                })
            elif route == "overview" and conn:
                self._json({
                    "counts": sx_counts(conn, user),
                    "profile": sx_profile(conn, user),
                    "memory_md": self.memfs.tree()["memory_md"] if self.memfs.ok else "",
                    "memfs_stats": self.memfs.stats() if self.memfs.ok else {},
                    "recent_facts": sx_facts(conn, user, "all", "")[:8],
                    "recent_journal": self.memfs.journal(8) if self.memfs.ok else [],
                })
            elif route == "facts" and conn:
                self._json(sx_facts(conn, user, qs.get("state", "active"), qs.get("q", "")))
            elif route == "fact" and conn:
                self._json(sx_fact_detail(conn, qs.get("id", "")) or {"error": "not found"})
            elif route == "episodes" and conn:
                self._json(sx_episodes(conn, user))
            elif route == "episode" and conn:
                self._json(sx_episode_detail(conn, qs.get("id", "")) or {"error": "not found"})
            elif route == "sessions" and conn:
                self._json(sx_sessions(conn, user))
            elif route == "session" and conn:
                self._json(sx_session_detail(conn, qs.get("id", "")))
            elif route == "memfs/tree":
                self._json(self.memfs.tree() if self.memfs.ok else {"notes": [], "memory_md": ""})
            elif route == "memfs/note":
                self._json(self.memfs.note(qs.get("path", "")) or {"error": "not found"})
            elif route == "memfs/journal":
                self._json(self.memfs.journal())
            elif route == "search":
                q = qs.get("q", "").strip()
                if not q:
                    self._json({"facts": [], "episodes": [], "sentences": [], "notes": []})
                    return
                result = sx_search(conn, user, q) if conn else \
                    {"facts": [], "episodes": [], "sentences": []}
                result["notes"] = self.memfs.search(q) if self.memfs.ok else []
                self._json(result)
            elif conn is None:
                self._json({"error": f"sentex db not found at {self.db_path}"}, 503)
            else:
                self._json({"error": "unknown route"}, 404)
        finally:
            if conn:
                conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(DEFAULT_DB), help="sentex sqlite db")
    ap.add_argument("--memfs-root", default=str(DEFAULT_MEMFS), help="memfs root dir")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    Handler.db_path = Path(args.db).expanduser()
    Handler.memfs = MemfsView(Path(args.memfs_root).expanduser())

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Memory Explorer → http://{args.host}:{args.port}")
    print(f"  sentex db : {Handler.db_path} {'(missing)' if not Handler.db_path.is_file() else ''}")
    print(f"  memfs root: {Handler.memfs.root} {'(missing)' if not Handler.memfs.ok else ''}")
    server.serve_forever()


if __name__ == "__main__":
    main()

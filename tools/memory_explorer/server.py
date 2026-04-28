from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

DEFAULT_DB_PATH = Path.home() / ".vektori" / "vektori.db"
HTML_PATH = Path(__file__).with_name("memory_explorer.html")


@dataclass
class ExplorerConfig:
    db_path: Path
    host: str
    port: int


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {row[1] for row in rows}


def list_user_ids(conn: sqlite3.Connection) -> list[str]:
    query = """
    SELECT DISTINCT user_id FROM (
        SELECT user_id FROM facts
        UNION ALL
        SELECT user_id FROM sentences
        UNION ALL
        SELECT user_id FROM episodes
        UNION ALL
        SELECT user_id FROM sessions
    )
    WHERE user_id IS NOT NULL AND TRIM(user_id) != ''
    ORDER BY user_id
    """
    return [row[0] for row in conn.execute(query).fetchall()]


def build_profile(facts: list[dict]) -> str:
    if not facts:
        return "No active profile facts found for this user."

    ranked = sorted(
        facts,
        key=lambda f: (
            f.get("confidence") if f.get("confidence") is not None else 0,
            f.get("mentions") if f.get("mentions") is not None else 0,
            f.get("created_at") or "",
        ),
        reverse=True,
    )
    lines = ["Living profile (active facts):"]
    for fact in ranked[:12]:
        confidence = fact.get("confidence")
        score = f"{confidence:.2f}" if isinstance(confidence, (int, float)) else "n/a"
        lines.append(f"- {fact['text']} (confidence={score}, mentions={fact.get('mentions', 1)})")
    return "\n".join(lines)


def load_user_snapshot(conn: sqlite3.Connection, user_id: str) -> dict:
    fact_cols = _table_columns(conn, "facts")
    mentions_expr = "mentions" if "mentions" in fact_cols else "1"
    event_time_expr = "event_time" if "event_time" in fact_cols else "NULL"

    facts_rows = conn.execute(
        f"""
        SELECT id, text, confidence, created_at, is_active, superseded_by, session_id,
               {mentions_expr} AS mentions,
               {event_time_expr} AS event_time
        FROM facts
        WHERE user_id = ?
        ORDER BY datetime(created_at) DESC
        """,
        (user_id,),
    ).fetchall()
    facts = [
        {
            "id": row[0],
            "text": row[1],
            "confidence": row[2],
            "created_at": row[3],
            "is_active": bool(row[4]),
            "superseded_by": row[5],
            "session_id": row[6],
            "mentions": row[7],
            "event_time": row[8],
        }
        for row in facts_rows
    ]

    episodes_rows = conn.execute(
        """
        SELECT e.id, e.text, e.session_id, s.started_at, e.created_at
        FROM episodes e
        LEFT JOIN sessions s ON s.id = e.session_id
        WHERE e.user_id = ?
        ORDER BY datetime(e.created_at) DESC
        """,
        (user_id,),
    ).fetchall()
    episodes = [
        {
            "id": row[0],
            "text": row[1],
            "session_id": row[2],
            "session_date": row[3],
            "created_at": row[4],
        }
        for row in episodes_rows
    ]

    sentence_rows = conn.execute(
        """
        SELECT id, session_id, role, text, turn_number, sentence_index, created_at
        FROM sentences
        WHERE user_id = ?
        ORDER BY session_id, turn_number, sentence_index
        """,
        (user_id,),
    ).fetchall()
    sentences = [
        {
            "id": row[0],
            "session_id": row[1],
            "role": row[2],
            "text": row[3],
            "turn_number": row[4],
            "sentence_index": row[5],
            "created_at": row[6],
        }
        for row in sentence_rows
    ]

    sessions: dict[str, list[dict]] = defaultdict(list)
    for sentence in sentences:
        sessions[sentence["session_id"]].append(sentence)

    fact_links_rows = conn.execute(
        """
        SELECT ef.fact_id, ef.episode_id
        FROM episode_facts ef
        JOIN facts f ON f.id = ef.fact_id
        WHERE f.user_id = ?
        """,
        (user_id,),
    ).fetchall()
    fact_to_episode_ids: dict[str, list[str]] = defaultdict(list)
    for fact_id, episode_id in fact_links_rows:
        fact_to_episode_ids[fact_id].append(episode_id)

    source_rows = conn.execute(
        """
        SELECT fs.fact_id, fs.sentence_id
        FROM fact_sources fs
        JOIN facts f ON f.id = fs.fact_id
        WHERE f.user_id = ?
        """,
        (user_id,),
    ).fetchall()
    fact_to_sentence_ids: dict[str, list[str]] = defaultdict(list)
    for fact_id, sentence_id in source_rows:
        fact_to_sentence_ids[fact_id].append(sentence_id)

    graph_nodes = []
    for fact in facts:
        graph_nodes.append({"id": f"fact:{fact['id']}", "type": "fact", "label": fact["text"]})
    for episode in episodes:
        graph_nodes.append(
            {"id": f"episode:{episode['id']}", "type": "episode", "label": episode["text"]}
        )
    for sentence in sentences:
        graph_nodes.append(
            {
                "id": f"sentence:{sentence['id']}",
                "type": "sentence",
                "label": sentence["text"],
            }
        )

    graph_edges = []
    for fact_id, episode_ids in fact_to_episode_ids.items():
        for episode_id in episode_ids:
            graph_edges.append(
                {
                    "source": f"fact:{fact_id}",
                    "target": f"episode:{episode_id}",
                    "kind": "fact_episode",
                }
            )
    for fact_id, sentence_ids in fact_to_sentence_ids.items():
        for sentence_id in sentence_ids:
            graph_edges.append(
                {
                    "source": f"fact:{fact_id}",
                    "target": f"sentence:{sentence_id}",
                    "kind": "fact_source",
                }
            )

    profile = build_profile([fact for fact in facts if fact["is_active"]])
    return {
        "user_id": user_id,
        "facts": facts,
        "episodes": episodes,
        "sentences": sentences,
        "sessions": [{"session_id": sid, "sentences": rows} for sid, rows in sessions.items()],
        "fact_links": {
            fact_id: {
                "episode_ids": fact_to_episode_ids.get(fact_id, []),
                "sentence_ids": fact_to_sentence_ids.get(fact_id, []),
            }
            for fact_id in {fact["id"] for fact in facts}
        },
        "profile": profile,
        "graph": {"nodes": graph_nodes, "edges": graph_edges},
    }


def create_handler(db_path: Path):
    class ExplorerHandler(BaseHTTPRequestHandler):
        def _write_json(self, status_code: int, payload: dict | list) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _open_conn(self) -> sqlite3.Connection:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            return conn

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                html = HTML_PATH.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
                return

            if parsed.path == "/api/users":
                with self._open_conn() as conn:
                    users = list_user_ids(conn)
                self._write_json(200, {"users": users})
                return

            if parsed.path.startswith("/api/user/"):
                user_id = unquote(parsed.path[len("/api/user/") :])
                if not user_id:
                    self._write_json(400, {"error": "user_id is required"})
                    return
                with self._open_conn() as conn:
                    snapshot = load_user_snapshot(conn, user_id)
                self._write_json(200, snapshot)
                return

            self._write_json(404, {"error": "not found"})

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    return ExplorerHandler


def parse_args() -> ExplorerConfig:
    parser = argparse.ArgumentParser(description="Local memory explorer UI for Vektori SQLite data")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    return ExplorerConfig(db_path=args.db_path, host=args.host, port=args.port)


def main() -> None:
    config = parse_args()
    if not config.db_path.exists():
        raise SystemExit(f"Database not found at: {config.db_path}")

    handler = create_handler(config.db_path)
    server = ThreadingHTTPServer((config.host, config.port), handler)
    print(f"Memory Explorer running on http://{config.host}:{config.port}")
    print(f"Using database: {config.db_path}")
    server.serve_forever()


if __name__ == "__main__":
    main()

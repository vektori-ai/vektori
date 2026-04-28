from __future__ import annotations

import sqlite3

from tools.memory_explorer.server import build_profile, list_user_ids, load_user_snapshot


def _setup_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE facts (
            id TEXT PRIMARY KEY,
            text TEXT,
            confidence REAL,
            created_at TEXT,
            is_active INTEGER,
            superseded_by TEXT,
            session_id TEXT,
            mentions INTEGER,
            event_time TEXT,
            user_id TEXT
        );
        CREATE TABLE episodes (
            id TEXT PRIMARY KEY,
            text TEXT,
            session_id TEXT,
            created_at TEXT,
            user_id TEXT
        );
        CREATE TABLE sentences (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            role TEXT,
            text TEXT,
            turn_number INTEGER,
            sentence_index INTEGER,
            created_at TEXT,
            user_id TEXT
        );
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            started_at TEXT
        );
        CREATE TABLE episode_facts (episode_id TEXT, fact_id TEXT);
        CREATE TABLE fact_sources (fact_id TEXT, sentence_id TEXT);
        """
    )

    conn.execute(
        "INSERT INTO facts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "f1",
            "User prefers WhatsApp",
            0.92,
            "2026-04-08 10:00:00",
            1,
            None,
            "s1",
            3,
            "2026-04-08",
            "u1",
        ),
    )
    conn.execute(
        "INSERT INTO facts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "f2",
            "User no longer uses email",
            0.88,
            "2026-04-07 09:00:00",
            0,
            "f1",
            "s1",
            1,
            None,
            "u1",
        ),
    )
    conn.execute(
        "INSERT INTO episodes VALUES (?, ?, ?, ?, ?)",
        ("e1", "Communication preference shifted to WhatsApp", "s1", "2026-04-08 11:00:00", "u1"),
    )
    conn.execute(
        "INSERT INTO sentences VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("t1", "s1", "user", "I only use WhatsApp", 1, 0, "2026-04-08 10:00:00", "u1"),
    )
    conn.execute(
        "INSERT INTO sessions VALUES (?, ?, ?)",
        ("s1", "u1", "2026-04-08 10:00:00"),
    )
    conn.execute("INSERT INTO episode_facts VALUES (?, ?)", ("e1", "f1"))
    conn.execute("INSERT INTO fact_sources VALUES (?, ?)", ("f1", "t1"))

    conn.execute(
        "INSERT INTO facts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("f3", "Other user fact", 0.7, "2026-04-05 10:00:00", 1, None, "x", 1, None, "u2"),
    )
    conn.commit()
    return conn


def test_build_profile_empty() -> None:
    assert build_profile([]) == "No active profile facts found for this user."


def test_user_snapshot_contains_links_and_grouped_sentences() -> None:
    conn = _setup_db()

    users = list_user_ids(conn)
    assert users == ["u1", "u2"]

    snapshot = load_user_snapshot(conn, "u1")

    assert snapshot["user_id"] == "u1"
    assert len(snapshot["facts"]) == 2
    assert len(snapshot["episodes"]) == 1
    assert len(snapshot["sessions"]) == 1
    assert snapshot["sessions"][0]["session_id"] == "s1"

    links = snapshot["fact_links"]["f1"]
    assert links["episode_ids"] == ["e1"]
    assert links["sentence_ids"] == ["t1"]

    graph_edges = snapshot["graph"]["edges"]
    assert {edge["kind"] for edge in graph_edges} == {"fact_episode", "fact_source"}
    assert "Living profile (active facts):" in snapshot["profile"]
    assert "User prefers WhatsApp" in snapshot["profile"]

"""Unit tests for the memory explorer's data shaping (server is read-only)."""

import json
import sqlite3

import pytest

from tools.memory_explorer.server import (
    MemfsView,
    sx_fact_detail,
    sx_facts,
    sx_search,
    sx_session_detail,
    sx_users,
)

SCHEMA = """
CREATE TABLE sentences (id TEXT PRIMARY KEY, text TEXT, embedding TEXT, user_id TEXT,
  agent_id TEXT, session_id TEXT, turn_number INT, sentence_index INT, role TEXT,
  content_hash TEXT UNIQUE, mentions INT DEFAULT 1, is_active INT DEFAULT 1,
  event_time TEXT, created_at TEXT DEFAULT (datetime('now')));
CREATE TABLE facts (id TEXT PRIMARY KEY, text TEXT, embedding TEXT, user_id TEXT,
  agent_id TEXT, session_id TEXT, subject TEXT, is_active INT DEFAULT 1,
  superseded_by TEXT, confidence REAL DEFAULT 1.0, metadata TEXT DEFAULT '{}',
  event_time TEXT, created_at TEXT DEFAULT (datetime('now')));
CREATE TABLE episodes (id TEXT PRIMARY KEY, text TEXT, embedding TEXT, user_id TEXT,
  agent_id TEXT, session_id TEXT, is_active INT DEFAULT 1, event_time TEXT,
  created_at TEXT DEFAULT (datetime('now')));
CREATE TABLE sessions (id TEXT PRIMARY KEY, user_id TEXT, agent_id TEXT,
  started_at TEXT, ended_at TEXT, metadata TEXT DEFAULT '{}');
CREATE TABLE fact_sources (fact_id TEXT, sentence_id TEXT, PRIMARY KEY(fact_id, sentence_id));
CREATE TABLE episode_facts (episode_id TEXT, fact_id TEXT, PRIMARY KEY(episode_id, fact_id));
CREATE TABLE fact_edges (source_id TEXT, target_id TEXT, user_id TEXT, weight REAL,
  created_at TEXT, PRIMARY KEY(source_id, target_id));
"""


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    c.execute("INSERT INTO sessions VALUES ('s1','alice',NULL,'2026-06-01',NULL,'{}')")
    c.execute("INSERT INTO sentences (id,text,user_id,session_id,turn_number,sentence_index,"
              "role,content_hash) VALUES ('sn1','I use ruff now.','alice','s1',1,0,'user','h1')")
    c.execute("INSERT INTO facts (id,text,user_id,session_id,subject,is_active,confidence,"
              "metadata,created_at) VALUES "
              "('f1','Uses ruff','alice','s1','tooling',1,0.9,'{\"mentions\": 2}','2026-06-01'),"
              "('f2','Uses flake8','alice','s1','tooling',0,0.8,'{}','2026-05-01')")
    c.execute("UPDATE facts SET superseded_by='f1' WHERE id='f2'")
    c.execute("INSERT INTO fact_sources VALUES ('f1','sn1')")
    c.execute("INSERT INTO episodes (id,text,user_id,session_id) VALUES "
              "('e1','Tooling switch','alice','s1')")
    c.execute("INSERT INTO episode_facts VALUES ('e1','f1')")
    yield c
    c.close()


def test_users_listed(conn):
    assert sx_users(conn) == ["alice"]


def test_facts_state_filter(conn):
    assert [f["id"] for f in sx_facts(conn, "alice", "active", "")] == ["f1"]
    assert [f["id"] for f in sx_facts(conn, "alice", "superseded", "")] == ["f2"]
    assert len(sx_facts(conn, "alice", "all", "")) == 2


def test_fact_metadata_mentions_parsed(conn):
    f = sx_facts(conn, "alice", "active", "")[0]
    assert f["mentions"] == 2


def test_fact_provenance_chain(conn):
    d = sx_fact_detail(conn, "f1")
    assert [s["id"] for s in d["sentences"]] == ["sn1"]
    assert [e["id"] for e in d["episodes"]] == ["e1"]
    assert [x["id"] for x in d["supersedes_facts"]] == ["f2"]
    old = sx_fact_detail(conn, "f2")
    assert [x["id"] for x in old["superseded_by_fact"]] == ["f1"]
    assert old["sentences"] == []


def test_session_reverse_provenance(conn):
    d = sx_session_detail(conn, "s1")
    assert len(d["sentences"]) == 1
    assert d["facts_by_sentence"]["sn1"][0]["fact_id"] == "f1"


def test_search_spans_layers(conn):
    r = sx_search(conn, "alice", "ruff")
    assert [f["id"] for f in r["facts"]] == ["f1"]
    assert [s["id"] for s in r["sentences"]] == ["sn1"]
    assert r["episodes"] == []


def test_memfs_view_tree_note_and_search(tmp_path):
    root = tmp_path / "memfs"
    (root / "semantic").mkdir(parents=True)
    (root / "semantic" / "style.md").write_text(
        "---\ntitle: Style guide\ntype: semantic\ntags: [python]\n---\n"
        "Use ruff. See [[ci-setup]].\n")
    (root / "procedural").mkdir()
    (root / "procedural" / "ci-setup.md").write_text("CI runs ruff on push.\n")
    v = MemfsView(root)
    assert v.ok
    tree = v.tree()
    assert {n["path"] for n in tree["notes"]} == {
        "semantic/style.md", "procedural/ci-setup.md"}
    note = v.note("semantic/style.md")
    assert note["title"] == "Style guide"
    assert note["links_out"][0]["path"] == "procedural/ci-setup.md"
    assert v.note("procedural/ci-setup.md")["backlinks"] == ["semantic/style.md"]
    # path traversal is rejected
    assert v.note("../../etc/passwd") is None
    # no index.db -> substring fallback still finds it
    hits = v.search("ruff")
    assert {h["path"] for h in hits} == {"semantic/style.md", "procedural/ci-setup.md"}


def test_memfs_view_missing_root(tmp_path):
    v = MemfsView(tmp_path / "nope")
    assert not v.ok
    assert v.journal() == []


def test_memfs_journal_order(tmp_path):
    root = tmp_path / "m"
    (root / ".memfs").mkdir(parents=True)
    lines = [json.dumps({"ts": f"2026-06-0{i}", "op": "sync"}) for i in (1, 2, 3)]
    (root / ".memfs" / "journal.jsonl").write_text("\n".join(lines) + "\n")
    j = MemfsView(root).journal()
    assert [e["ts"] for e in j] == ["2026-06-03", "2026-06-02", "2026-06-01"]

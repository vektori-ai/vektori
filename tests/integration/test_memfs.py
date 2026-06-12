"""Integration tests for MemFS — lexical-only (no network) + mock embedder paths."""

from __future__ import annotations

import json
import os
import re
from unittest.mock import AsyncMock, MagicMock

import pytest

from vektori.memfs import MemFS, SecretsFoundError

pytestmark = pytest.mark.asyncio


def _mock_embedder(dim: int = 8):
    emb = MagicMock()
    counter = {"batch_texts": 0}

    def _vec(text: str):
        # deterministic pseudo-embedding from char histogram: similar text => similar vec
        v = [0.0] * dim
        for ch in text[:256]:
            v[ord(ch) % dim] += 1.0
        return v

    async def embed(text):
        return _vec(text)

    async def embed_batch(texts):
        counter["batch_texts"] += len(texts)
        return [_vec(t) for t in texts]

    emb.embed = AsyncMock(side_effect=embed)
    emb.embed_batch = AsyncMock(side_effect=embed_batch)
    emb._counter = counter
    return emb


# ── remember / recall (lexical only — embedder=None) ──────────────────────

async def test_remember_creates_markdown_file(tmp_path):
    async with MemFS(root=tmp_path / "mem") as mem:
        note = await mem.remember("User prefers ruff over flake8 for linting Python code.",
                                  type="semantic", tags=["python"])
        files = list((tmp_path / "mem" / "semantic").glob("*.md"))
        assert len(files) == 1
        text = files[0].read_text()
        assert text.startswith("---")
        assert "ruff over flake8" in text
        assert note.id in text


async def test_recall_lexical_finds_note(tmp_path):
    async with MemFS(root=tmp_path / "mem") as mem:
        await mem.remember("The deploy script for benchmarks lives on the EC2 instance.",
                           type="procedural")
        await mem.remember("User timezone is IST, prefers concise answers.", type="semantic")
        out = await mem.recall("deploy benchmarks EC2")
        assert out.memory_found
        assert "EC2" in out.items[0].snippet
        assert out.items[0].signals.get("bm25", 0) > 0
        assert out.items[0].start_line >= 1


async def test_recall_returns_pointers_not_paraphrases(tmp_path):
    async with MemFS(root=tmp_path / "mem") as mem:
        await mem.remember("BGE-M3 OOMs on the small AWS instance; use Cloudflare embeddings.",
                           type="semantic", title="cloudflare embeddings workaround")
        out = await mem.recall("BGE-M3 OOM")
        assert out.memory_found
        item = out.items[0]
        content = open(item.path).read()
        assert item.snippet[:50] in content  # snippet is real file content


async def test_episodic_lands_in_month_dir(tmp_path):
    async with MemFS(root=tmp_path / "mem") as mem:
        note = await mem.remember("Fixed the chunk overlap bug in the splitter today.",
                                  type="episodic")
        assert re.search(r"episodic/\d{4}-\d{2}/", note.path)


# ── canonical-store invariant ──────────────────────────────────────────────

async def test_index_delete_then_sync_reproduces_recall(tmp_path):
    root = tmp_path / "mem"
    async with MemFS(root=root) as mem:
        await mem.remember("Vektori uses SQLite FTS5 for keyword retrieval.", type="semantic")
        await mem.remember("The dashboard is a React app under vektori-dashboard.", type="semantic")
        before = await mem.recall("FTS5 keyword retrieval")
        paths_before = [i.path for i in before.items]

    (root / ".memfs" / "index.db").unlink()
    async with MemFS(root=root) as mem:
        after = await mem.recall("FTS5 keyword retrieval")
        assert [i.path for i in after.items] == paths_before


async def test_externally_written_file_is_picked_up(tmp_path):
    root = tmp_path / "mem"
    async with MemFS(root=root) as mem:
        await mem.remember("seed note so dirs exist", type="semantic")
    # human writes a bare markdown file, no frontmatter
    (root / "semantic" / "rust-style.md").write_text(
        "# Rust style\n\nAlways run clippy before committing rust changes.\n")
    async with MemFS(root=root) as mem:
        out = await mem.recall("clippy rust")
        assert out.memory_found
        assert out.items[0].path.endswith("rust-style.md")


# ── incremental embedding (mock embedder) ──────────────────────────────────

async def test_edit_reembeds_only_changed_chunks(tmp_path):
    root = tmp_path / "mem"
    emb = _mock_embedder()
    async with MemFS(root=root, embedder=emb) as mem:
        await mem.remember(
            "# Topic A\n\nParagraph about retrieval quality and ranking signals here.\n\n"
            "# Topic B\n\nParagraph about storage formats and atomic writes in detail.",
            type="semantic", title="two-topics")
        base = emb._counter["batch_texts"]
        assert base >= 2

    # edit ONE section externally
    p = root / "semantic" / "two-topics.md"
    p.write_text(p.read_text().replace("atomic writes", "atomic renames"))

    async with MemFS(root=root, embedder=emb) as mem:
        delta = emb._counter["batch_texts"] - base
        assert delta == 1  # only the changed chunk was re-embedded


async def test_vector_signal_present_with_embedder(tmp_path):
    emb = _mock_embedder()
    async with MemFS(root=tmp_path / "mem", embedder=emb) as mem:
        await mem.remember("Postgres with pgvector is the production backend.", type="semantic")
        out = await mem.recall("pgvector production backend")
        assert out.memory_found
        assert "vec" in out.items[0].signals


# ── graph layer ────────────────────────────────────────────────────────────

async def test_wikilink_expansion_pulls_neighbor(tmp_path):
    async with MemFS(root=tmp_path / "mem") as mem:
        await mem.remember("Gemini benchmarks must use the paid tier key.",
                           type="semantic", title="gemini-key-policy")
        await mem.remember("LoCoMo run config: see [[gemini-key-policy]] before launching runs.",
                           type="procedural", title="locomo-run-checklist")
        out = await mem.recall("LoCoMo run config checklist", k=8)
        paths = [i.path for i in out.items]
        assert any(p.endswith("locomo-run-checklist.md") for p in paths)
        assert any(p.endswith("gemini-key-policy.md") for p in paths)
        linked = [i for i in out.items if i.path.endswith("gemini-key-policy.md")]
        assert any("link" in i.signals for i in linked)


# ── lifecycle ──────────────────────────────────────────────────────────────

async def test_forget_really_deletes(tmp_path):
    async with MemFS(root=tmp_path / "mem") as mem:
        note = await mem.remember("Temporary project codename is Bluebird.",
                                  type="semantic", title="codename-bluebird")
        removed = await mem.forget("codename-bluebird")
        assert removed and removed[0] == note.path
        assert not os.path.exists(note.path)
        out = await mem.recall("Bluebird codename")
        assert all("bluebird" not in i.path.lower() for i in out.items)


async def test_compact_month_archives_and_digests(tmp_path):
    root = tmp_path / "mem"
    async with MemFS(root=root) as mem:
        for i in range(3):
            await mem.remember("Episode " + str(i) + ": fixed bug number " + str(i)
                               + " in the retrieval pipeline.", type="episodic")
        from vektori.memfs.models import now_utc
        month = now_utc().strftime("%Y-%m")
        report = await mem.compact(month)
        assert report.notes_compacted == 3
        assert (root / "episodic" / (month + "-digest.md")).exists()
        assert len(list((root / "archive" / month).glob("*.md"))) == 3
        # archived notes excluded from default recall
        out = await mem.recall("retrieval pipeline bug")
        assert all("/archive/" not in i.path for i in out.items)


async def test_verify_clean_and_detects_broken_link(tmp_path):
    async with MemFS(root=tmp_path / "mem") as mem:
        await mem.remember("Note with a [[missing-target]] link.", type="semantic")
        report = await mem.verify()
        assert report.files_checked >= 1
        assert any(slug == "missing-target" for _, slug in report.broken_links)
        assert not report.hash_drift
        assert report.orphan_rows == 0


# ── secrets ────────────────────────────────────────────────────────────────

async def test_remember_refuses_secrets(tmp_path):
    async with MemFS(root=tmp_path / "mem") as mem:
        with pytest.raises(SecretsFoundError):
            await mem.remember("AWS key is AKIAIOSFODNN7EXAMPLE for the bench account.")
        out = await mem.recall("AWS key")
        assert not any("AKIA" in i.snippet for i in out.items)


# ── ingestion ──────────────────────────────────────────────────────────────

async def test_ingest_file_creates_source_note_and_skip_unchanged(tmp_path):
    doc = tmp_path / "design.md"
    doc.write_text("# Design\n\nThe retrieval pipeline fuses BM25 and vectors via RRF.\n")
    async with MemFS(root=tmp_path / "mem") as mem:
        r1 = await mem.ingest_file(doc)
        assert not r1.skipped and r1.note_path
        r2 = await mem.ingest_file(doc)
        assert r2.skipped and r2.reason == "unchanged"
        out = await mem.recall("RRF fusion BM25")
        assert out.memory_found
        assert out.items[0].provenance and out.items[0].provenance.startswith("file:")


async def test_ingest_directory_excludes_machinery(tmp_path):
    corpus = tmp_path / "corpus"
    (corpus / ".git").mkdir(parents=True)
    (corpus / ".git" / "evil.md").write_text("# Git internals\n\nshould never be ingested.\n")
    (corpus / "real.md").write_text("# Real doc\n\nGenuinely useful content about embeddings.\n")
    async with MemFS(root=tmp_path / "mem") as mem:
        reports = await mem.ingest_directory(corpus)
        ingested = [r for r in reports if not r.skipped]
        assert len(ingested) == 1
        assert ingested[0].path.endswith("real.md")


# ── orientation / stats / journal ─────────────────────────────────────────

async def test_memory_md_and_stats_and_journal(tmp_path):
    root = tmp_path / "mem"
    async with MemFS(root=root) as mem:
        await mem.remember("Cloudflare config lives in the dashboard settings page.",
                           type="semantic", title="cf-config-location")
        moc = mem.orient()
        assert "cf-config-location" in moc
        s = mem.stats()
        assert s["files"] >= 1 and s["chunks"] >= 1
        await mem.recall("cloudflare config")
        journal = (root / ".memfs" / "journal.jsonl").read_text().splitlines()
        ops = [json.loads(line)["op"] for line in journal]
        assert "remember" in ops and "recall" in ops and "sync" in ops

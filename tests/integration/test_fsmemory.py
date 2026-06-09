"""Integration tests for FilesystemMemory — mocked embedder + LLM, real SQLite on tmp files."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vektori.fsmemory.memory import FilesystemMemory


# ── helpers ────────────────────────────────────────────────────────────────

def _make_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 1536)
    embedder.embed_batch = AsyncMock(side_effect=lambda texts: [[0.1] * 1536] * len(texts))
    return embedder


def _make_llm(facts: list[str] | None = None) -> MagicMock:
    facts = facts or ["The system uses BGE-M3 for embeddings.", "SQLite is the default backend."]
    response = json.dumps({"facts": [{"text": f, "subject": "system"} for f in facts]})
    llm = MagicMock()
    llm.complete = AsyncMock(return_value=response)
    return llm


def _make_fs(tmp_path: Path, extract_facts: bool = True) -> FilesystemMemory:
    db_url = f"sqlite:///{tmp_path / 'fsmemory.db'}"
    return FilesystemMemory(
        user_id="test-user",
        database_url=db_url,
        extract_facts=extract_facts,
    )


def _patch_models(embedder=None, llm=None):
    embedder = embedder or _make_embedder()
    llm = llm or _make_llm()
    return (
        patch("vektori.fsmemory.memory.create_embedder", return_value=embedder),
        patch("vektori.fsmemory.memory.create_llm", return_value=llm),
    )


# ── add_file ───────────────────────────────────────────────────────────────

async def test_add_file_basic(tmp_path):
    doc = tmp_path / "notes.md"
    doc.write_text("## Architecture\n\nThe system uses a three-layer graph for memory. Facts live at L0.\n")

    p1, p2 = _patch_models()
    with p1, p2:
        async with _make_fs(tmp_path) as fs:
            result = await fs.add_file(str(doc))

    assert result.error is None
    assert not result.skipped
    assert result.chunks >= 1
    assert result.facts_inserted >= 1


async def test_add_file_skips_unchanged(tmp_path):
    doc = tmp_path / "notes.md"
    doc.write_text("## Section\n\nContent that is long enough to pass the minimum chunk size check.\n")

    p1, p2 = _patch_models()
    with p1, p2:
        async with _make_fs(tmp_path) as fs:
            first = await fs.add_file(str(doc))
            second = await fs.add_file(str(doc))

    assert not first.skipped
    assert second.skipped
    assert second.facts_inserted == 0


async def test_add_file_reingest_on_change(tmp_path):
    doc = tmp_path / "notes.md"
    doc.write_text("## Section\n\nOriginal content that is long enough to be ingested properly.\n")

    p1, p2 = _patch_models()
    with p1, p2:
        async with _make_fs(tmp_path) as fs:
            first = await fs.add_file(str(doc))
            assert not first.skipped

            doc.write_text("## Section\n\nUpdated content that has changed significantly from before.\n")
            second = await fs.add_file(str(doc))

    assert not second.skipped
    assert second.facts_inserted >= 1


async def test_add_file_not_a_file(tmp_path):
    p1, p2 = _patch_models()
    with p1, p2:
        async with _make_fs(tmp_path) as fs:
            result = await fs.add_file(str(tmp_path / "nonexistent.md"))

    assert result.error is not None


async def test_add_file_extract_false_no_llm_call(tmp_path):
    doc = tmp_path / "notes.md"
    doc.write_text("## Notes\n\nSome content here that is long enough to be stored directly.\n")

    embedder = _make_embedder()
    llm = _make_llm()
    p1 = patch("vektori.fsmemory.memory.create_embedder", return_value=embedder)
    p2 = patch("vektori.fsmemory.memory.create_llm", return_value=llm)

    with p1, p2:
        async with _make_fs(tmp_path, extract_facts=False) as fs:
            result = await fs.add_file(str(doc))

    assert result.facts_inserted >= 1
    llm.complete.assert_not_called()


# ── add_directory ──────────────────────────────────────────────────────────

async def test_add_directory_multiple_files(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("## A\n\nContent for document A, enough text to pass the minimum.\n")
    (docs / "b.md").write_text("## B\n\nContent for document B, enough text to pass the minimum.\n")
    (docs / "c.md").write_text("## C\n\nContent for document C, enough text to pass the minimum.\n")

    p1, p2 = _patch_models()
    with p1, p2:
        async with _make_fs(tmp_path) as fs:
            results = await fs.add_directory(str(docs), glob="**/*.md")

    assert len(results) == 3
    assert all(r.error is None for r in results)
    assert all(not r.skipped for r in results)


async def test_add_directory_excludes_patterns(tmp_path):
    docs = tmp_path / "proj"
    docs.mkdir()
    (docs / "readme.md").write_text("## Readme\n\nThis is the readme with enough text content.\n")
    cache = docs / "__pycache__"
    cache.mkdir()
    (cache / "module.pyc").write_bytes(b"\x00\x01\x02")

    p1, p2 = _patch_models()
    with p1, p2:
        async with _make_fs(tmp_path) as fs:
            results = await fs.add_directory(str(docs), glob="**/*")

    paths = [r.path for r in results]
    assert not any("__pycache__" in p for p in paths if not results[paths.index(p)].skipped)


async def test_add_directory_not_a_dir(tmp_path):
    p1, p2 = _patch_models()
    with p1, p2:
        async with _make_fs(tmp_path) as fs:
            results = await fs.add_directory(str(tmp_path / "missing"))

    assert len(results) == 1
    assert results[0].error is not None


# ── search ─────────────────────────────────────────────────────────────────

async def test_search_returns_facts_shape(tmp_path):
    doc = tmp_path / "notes.md"
    doc.write_text("## Memory\n\nVektori stores facts in a three-layer sentence graph system.\n")

    p1, p2 = _patch_models()
    with p1, p2:
        async with _make_fs(tmp_path) as fs:
            await fs.add_file(str(doc))
            out = await fs.search("how does memory work?")

    assert "facts" in out
    assert "sentences" in out
    assert isinstance(out["facts"], list)
    assert isinstance(out["memory_found"], bool)


async def test_search_path_mode(tmp_path):
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "notes.md").write_text("## Notes\n\nContent inside subdir with enough text here.\n")

    p1, p2 = _patch_models()
    with p1, p2:
        async with _make_fs(tmp_path) as fs:
            await fs.add_file(str(subdir / "notes.md"))
            out = await fs.search("", mode="path", path_prefix=str(subdir))

    assert out["memory_found"]
    assert all(str(subdir) in f["source_path"] for f in out["facts"])


async def test_search_empty_store_returns_empty(tmp_path):
    p1, p2 = _patch_models()
    with p1, p2:
        async with _make_fs(tmp_path) as fs:
            out = await fs.search("anything")

    assert out["facts"] == []
    assert not out["memory_found"]


# ── list_files / get_stats / delete_file ──────────────────────────────────

async def test_list_files(tmp_path):
    doc = tmp_path / "notes.md"
    doc.write_text("## Notes\n\nContent that is long enough to pass the minimum chunk requirement.\n")

    p1, p2 = _patch_models()
    with p1, p2:
        async with _make_fs(tmp_path) as fs:
            await fs.add_file(str(doc))
            files = await fs.list_files()

    assert str(doc.resolve()) in files


async def test_get_stats(tmp_path):
    doc = tmp_path / "notes.md"
    doc.write_text("## Notes\n\nContent that is long enough to pass the minimum chunk requirement.\n")

    p1, p2 = _patch_models()
    with p1, p2:
        async with _make_fs(tmp_path) as fs:
            await fs.add_file(str(doc))
            stats = await fs.get_stats()

    assert stats["files"] == 1
    assert stats["facts"] >= 1
    assert stats["user_id"] == "test-user"


async def test_delete_file_removes_from_index(tmp_path):
    doc = tmp_path / "notes.md"
    doc.write_text("## Notes\n\nContent that is long enough to pass the minimum chunk requirement.\n")

    p1, p2 = _patch_models()
    with p1, p2:
        async with _make_fs(tmp_path) as fs:
            await fs.add_file(str(doc))
            assert str(doc.resolve()) in await fs.list_files()

            await fs.delete_file(str(doc))
            assert str(doc.resolve()) not in await fs.list_files()


async def test_get_stats_empty(tmp_path):
    p1, p2 = _patch_models()
    with p1, p2:
        async with _make_fs(tmp_path) as fs:
            stats = await fs.get_stats()

    assert stats["files"] == 0
    assert stats["facts"] == 0


# ── context manager ────────────────────────────────────────────────────────

async def test_context_manager_closes_cleanly(tmp_path):
    p1, p2 = _patch_models()
    with p1, p2:
        fs = _make_fs(tmp_path)
        async with fs:
            assert fs._initialized
        assert not fs._initialized

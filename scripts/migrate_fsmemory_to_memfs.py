#!/usr/bin/env python3
"""One-shot migration: old fsmemory SQLite store -> MemFS file tree.

Read-only on the source DB. Strategy (docs/NEXT_GEN_FS_MEMORY_SYSTEM.md §9):
  - source file still exists on disk  -> native re-ingest (full fidelity)
  - source file gone                  -> materialize stored facts into one
                                         sources/<slug>.md marked "recovered"

Usage:
    uv run python scripts/migrate_fsmemory_to_memfs.py \
        --db ~/.vektori/fsmemory.db --user <user_id> --root ~/.vektori/memfs/default \
        [--embedder provider:model] [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vektori.memfs import MemFS
from vektori.memfs.models import Note, now_utc
from vektori.memfs.notes import atomic_write, dump_note, new_id, slugify


def load_old(db_path: Path, user_id: str):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    files = [dict(r) for r in conn.execute(
        "SELECT path, content_hash, fact_count FROM fs_file_index WHERE user_id=?", (user_id,))]
    facts_by_path: dict[str, list[dict]] = {}
    rows = conn.execute(
        "SELECT text, subject, metadata, created_at FROM facts "
        "WHERE user_id=? AND is_active=1", (user_id,)).fetchall()
    for r in rows:
        try:
            meta = json.loads(r["metadata"] or "{}")
        except json.JSONDecodeError:
            meta = {}
        if meta.get("source_type") != "filesystem":
            continue
        src = meta.get("source_path", "(unknown)")
        facts_by_path.setdefault(src, []).append(
            {"text": r["text"], "subject": r["subject"], "created_at": r["created_at"]})
    conn.close()
    return files, facts_by_path


async def migrate(db: Path, user_id: str, root: Path, embedder, dry_run: bool):
    files, facts_by_path = load_old(db, user_id)
    print("old store:", len(files), "files,",
          sum(len(v) for v in facts_by_path.values()), "active filesystem facts")

    reingested, recovered, skipped = [], [], []
    async with MemFS(root=root, embedder=embedder) as mem:
        for f in files:
            src = Path(f["path"])
            if src.is_file():
                if dry_run:
                    reingested.append(str(src))
                    continue
                rep = await mem.ingest_file(src)
                (skipped if rep.skipped and rep.reason != "unchanged" else reingested).append(str(src))
            else:
                facts = facts_by_path.get(str(src), [])
                if not facts:
                    skipped.append(str(src) + " (missing, no facts)")
                    continue
                if dry_run:
                    recovered.append(str(src))
                    continue
                body_lines = ["> Recovered from fsmemory DB; original file missing at migration.",
                              ""]
                for fact in facts:
                    subj = (" [" + fact["subject"] + "]") if fact["subject"] else ""
                    body_lines.append("- " + fact["text"] + subj)
                note = Note(
                    id=new_id(), type="source",
                    title="Recovered: " + src.name,
                    body="\n".join(body_lines),
                    created=now_utc(),
                    source="fsmemory:" + str(src),
                    tags=["migrated", "recovered"],
                )
                dest = root / "sources" / ("recovered-" + slugify(src.stem) + ".md")
                atomic_write(dest, dump_note(note))
                recovered.append(str(src))
        if not dry_run:
            await mem.sync()

    print("re-ingested:", len(reingested), "| recovered:", len(recovered),
          "| skipped:", len(skipped))
    for s in skipped:
        print("  skip:", s)
    print("old DB untouched:", db)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(Path.home() / ".vektori" / "fsmemory.db"))
    ap.add_argument("--user", required=True)
    ap.add_argument("--root", default=str(Path.home() / ".vektori" / "memfs" / "default"))
    ap.add_argument("--embedder", default=None,
                    help="provider:model for the new index (omit = lexical-only)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    db = Path(args.db).expanduser()
    if not db.is_file():
        print("no old DB at", db, "- nothing to migrate")
        return
    asyncio.run(migrate(db, args.user, Path(args.root).expanduser(),
                        args.embedder, args.dry_run))


if __name__ == "__main__":
    main()

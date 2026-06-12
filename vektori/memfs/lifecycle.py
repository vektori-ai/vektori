"""Lifecycle: monthly compaction of episodic notes into digests + archive.

Default path is LLM-free (heading-concatenation digest). LLM reflection is a
separate, explicit, proposal-generating step — it never silently mutates canon.
"""

from __future__ import annotations

from pathlib import Path

from vektori.memfs.models import CompactReport, Note, now_utc
from vektori.memfs.notes import atomic_write, dump_note, new_id, parse_note


def compact_month(root: Path, month: str, llm=None) -> CompactReport:
    """Roll episodic/<month>/ into one digest note; move originals to archive/<month>/.

    month: "YYYY-MM". Digest is written to episodic/<month>-digest.md so it stays
    in default recall; originals land in archive/ (excluded from default recall,
    still on disk, still greppable).
    """
    if llm is not None:
        raise NotImplementedError(
            "LLM reflection is design-complete (see docs/NEXT_GEN_FS_MEMORY_SYSTEM.md §6) "
            "but not in v1 — it must emit proposal files, not mutate canon."
        )

    src_dir = root / "episodic" / month
    report = CompactReport(month=month)
    if not src_dir.is_dir():
        return report

    notes = []
    for p in sorted(src_dir.glob("*.md")):
        try:
            notes.append(parse_note(p))
        except OSError:
            continue
    if not notes:
        return report

    sections = []
    for n in notes:
        when = n.when or n.created
        stamp = when.date().isoformat() if when else "????-??-??"
        body = n.body.strip()
        sections.append("## " + stamp + " — " + n.title + "\n\n" + body)

    digest = Note(
        id=new_id(),
        type="episodic",
        title="Digest " + month + " (" + str(len(notes)) + " episodes)",
        body="\n\n".join(sections),
        created=now_utc(),
        source="compaction:" + month,
        tags=["digest"],
    )
    digest_path = root / "episodic" / (month + "-digest.md")
    atomic_write(digest_path, dump_note(digest))

    archive_dir = root / "archive" / month
    archive_dir.mkdir(parents=True, exist_ok=True)
    for n in notes:
        src = Path(n.path)
        dst = archive_dir / src.name
        src.rename(dst)
        report.archived.append(str(dst))

    try:
        src_dir.rmdir()
    except OSError:
        pass

    report.notes_compacted = len(notes)
    report.digest_path = str(digest_path)
    return report

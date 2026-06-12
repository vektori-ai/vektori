"""Note file format: frontmatter parse/dump, slugs, atomic writes, heading chunker, wikilinks.

Minimal frontmatter dialect (deliberately not full YAML — no new dependency):
  key: scalar            strings, ints, ISO datetimes
  key: [a, b, c]         flat lists of scalars
Lenient parsing: files without (or with broken) frontmatter get inferred defaults,
so notes written by sloppy agents or humans never break the system.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path

from vektori.memfs.models import Chunk, Note, now_utc

_FM_DELIM = "---"
_WIKILINK_RE = re.compile(r"\[\[([^\[\]|]+)(?:\|[^\[\]]*)?\]\]")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_QUOTES = "\x27\x22"  # ' and "

CHUNK_TARGET_MAX = 800
CHUNK_MIN = 40


def new_id() -> str:
    """Time-sortable unique id (ULID-flavored, no dependency)."""
    ms = int(time.time() * 1000)
    tail = uuid.uuid4().hex[:12]
    return format(ms, "013x") + tail


def slugify(title: str, max_len: int = 64) -> str:
    slug = _SLUG_RE.sub("-", title.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "note"


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + "." + str(os.getpid()) + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _parse_scalar(raw: str):
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [s.strip().strip(_QUOTES) for s in inner.split(",") if s.strip()]
    return raw.strip(_QUOTES)


def _parse_dt(raw) -> datetime | None:
    if not raw or not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def parse_note(path: Path, content: str | None = None) -> Note:
    """Parse a note file. Never raises on malformed frontmatter — infers defaults."""
    text = content if content is not None else path.read_text(encoding="utf-8", errors="ignore")
    meta: dict = {}
    body = text

    if text.startswith(_FM_DELIM):
        end = text.find("\n" + _FM_DELIM, len(_FM_DELIM))
        if end != -1:
            fm_block = text[len(_FM_DELIM):end]
            body = text[end + len(_FM_DELIM) + 1:].lstrip("\n")
            for line in fm_block.splitlines():
                if ":" in line and not line.startswith(" "):
                    k, _, v = line.partition(":")
                    meta[k.strip()] = _parse_scalar(v)

    inferred_type = "semantic"
    if "sources" in path.parts:
        inferred_type = "source"
    elif "episodic" in path.parts:
        inferred_type = "episodic"
    elif "procedural" in path.parts:
        inferred_type = "procedural"

    title = meta.get("title") or _first_heading(body) or path.stem.replace("-", " ")
    tags = meta.get("tags", [])
    if isinstance(tags, str):
        tags = [tags] if tags else []

    raw_schema = str(meta.get("schema", 1))
    return Note(
        id=str(meta.get("id") or hashlib.sha256(str(path).encode()).hexdigest()[:25]),
        type=str(meta.get("type") or inferred_type),
        title=str(title),
        body=body,
        path=str(path),
        created=_parse_dt(meta.get("created")),
        when=_parse_dt(meta.get("when")),
        source=meta.get("source") or None,
        tags=list(tags),
        schema=int(raw_schema) if raw_schema.isdigit() else 1,
    )


def dump_note(note: Note) -> str:
    lines = [_FM_DELIM]
    lines.append("id: " + note.id)
    lines.append("type: " + note.type)
    lines.append("title: " + note.title)
    created = note.created or now_utc()
    lines.append("created: " + created.isoformat())
    if note.when:
        lines.append("when: " + note.when.isoformat())
    if note.source:
        lines.append("source: " + str(note.source))
    if note.tags:
        joined = ", ".join(note.tags)
        lines.append("tags: [" + joined + "]")
    lines.append("schema: " + str(note.schema))
    lines.append(_FM_DELIM)
    return "\n".join(lines) + "\n" + note.body.rstrip("\n") + "\n"


def _first_heading(body: str) -> str | None:
    for line in body.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            return m.group(2).strip()
    return None


def extract_wikilinks(body: str) -> list[str]:
    return [m.group(1).strip() for m in _WIKILINK_RE.finditer(body)]


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def chunk_note(note: Note) -> list[Chunk]:
    """Heading-aware chunking with line spans. Heading path provides chunk context."""
    lines = note.body.splitlines()
    fm_offset = _frontmatter_line_count(note)
    chunks: list[Chunk] = []
    heading_stack: list[tuple[int, str]] = []
    buf: list[str] = []
    state = {"buf_start": 0}

    def flush(end_line: int) -> None:
        text = "\n".join(buf).strip()
        if len(text) >= CHUNK_MIN:
            for part, s, e in _split_long(text, state["buf_start"], end_line):
                hp = " > ".join([note.title] + [h for _, h in heading_stack])
                key = note.id + "|" + hp + "|" + part
                cid = hashlib.sha256(key.encode()).hexdigest()[:24]
                chunks.append(Chunk(
                    chunk_id=cid, file_id=note.id, heading_path=hp,
                    text=part, start_line=s + fm_offset + 1, end_line=e + fm_offset + 1,
                ))
        buf.clear()

    for i, line in enumerate(lines):
        m = _HEADING_RE.match(line)
        if m:
            flush(i - 1)
            level = len(m.group(1))
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, m.group(2).strip()))
            state["buf_start"] = i + 1
        else:
            if not buf:
                state["buf_start"] = i
            buf.append(line)
    flush(len(lines) - 1)
    return chunks


def _split_long(text: str, start: int, end: int):
    """Split oversize sections on paragraph boundaries; keep line-span estimates."""
    if len(text) <= CHUNK_TARGET_MAX:
        return [(text, start, end)]
    paras = re.split(r"\n\s*\n", text)
    out: list[tuple[str, int, int]] = []
    cur = ""
    total_lines = max(end - start, 1)
    cursor = start
    for para in paras:
        if cur and len(cur) + len(para) + 2 > CHUNK_TARGET_MAX:
            n_lines = max(1, round(total_lines * len(cur) / max(len(text), 1)))
            out.append((cur, cursor, min(cursor + n_lines, end)))
            cursor = min(cursor + n_lines + 1, end)
            cur = para
        else:
            cur = cur + "\n\n" + para if cur else para
    if cur:
        out.append((cur, cursor, end))
    return out


def _frontmatter_line_count(note: Note) -> int:
    if not note.path:
        return 0
    try:
        raw = Path(note.path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return 0
    if not raw.startswith(_FM_DELIM):
        return 0
    end = raw.find("\n" + _FM_DELIM, len(_FM_DELIM))
    if end == -1:
        return 0
    return raw[:end].count("\n") + 2

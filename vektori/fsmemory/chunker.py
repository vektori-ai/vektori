"""File-type-aware document chunker. Markdown heading split; paragraph split for everything else."""

from __future__ import annotations

import re

from vektori.fsmemory.models import FileChunk

MAX_CHUNK_CHARS = 800
OVERLAP_CHARS = 80
MIN_CHUNK_CHARS = 40

_HEADING_RE = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)


class DocumentChunker:
    def chunk(self, path: str, content: str) -> list[FileChunk]:
        if not content or not content.strip():
            return []
        if path.endswith((".md", ".markdown", ".mdx")):
            return self._chunk_markdown(path, content)
        return self._chunk_text(path, content)

    def _chunk_markdown(self, path: str, content: str) -> list[FileChunk]:
        sections: list[tuple[str | None, str]] = []
        current_heading: str | None = None
        current_lines: list[str] = []

        for line in content.splitlines(keepends=True):
            m = _HEADING_RE.match(line.rstrip())
            if m:
                if current_lines:
                    sections.append((current_heading, "".join(current_lines).strip()))
                current_heading = m.group(1).strip()
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            sections.append((current_heading, "".join(current_lines).strip()))

        chunks: list[FileChunk] = []
        idx = 0
        for heading, body in sections:
            for chunk_text in _split_with_overlap(body):
                if len(chunk_text) < MIN_CHUNK_CHARS:
                    continue
                chunks.append(FileChunk(path=path, chunk_index=idx, text=chunk_text, heading=heading))
                idx += 1
        return chunks

    def _chunk_text(self, path: str, content: str) -> list[FileChunk]:
        chunks: list[FileChunk] = []
        idx = 0
        for chunk_text in _split_with_overlap(content):
            if len(chunk_text) < MIN_CHUNK_CHARS:
                continue
            chunks.append(FileChunk(path=path, chunk_index=idx, text=chunk_text))
            idx += 1
        return chunks


def _split_with_overlap(text: str) -> list[str]:
    """Split text by double-newline paragraphs, merge short ones, apply max size with overlap."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    results: list[str] = []
    current = ""

    for para in paragraphs:
        if not current:
            current = para
        elif len(current) + len(para) + 2 <= MAX_CHUNK_CHARS:
            current = current + "\n\n" + para
        else:
            results.append(current)
            # carry overlap from end of current chunk into next
            overlap = current[-OVERLAP_CHARS:] if len(current) > OVERLAP_CHARS else current
            current = overlap + "\n\n" + para if overlap else para

    if current:
        results.append(current)

    return results

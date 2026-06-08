"""Unit tests for DocumentChunker — no IO, no mocks needed."""

from vektori.fsmemory.chunker import DocumentChunker, _split_with_overlap


# ── _split_with_overlap ────────────────────────────────────────────────────

def test_split_single_paragraph():
    result = _split_with_overlap("Just one paragraph here, nothing special.")
    assert len(result) == 1
    assert result[0] == "Just one paragraph here, nothing special."


def test_split_two_short_paragraphs_merged():
    text = "First para.\n\nSecond para."
    result = _split_with_overlap(text)
    assert len(result) == 1
    assert "First para." in result[0]
    assert "Second para." in result[0]


def test_split_long_content_across_paragraphs():
    # splitting only happens at \n\n boundaries; many short paragraphs that together exceed limit
    paras = "\n\n".join(["word " * 30 for _ in range(10)])  # 10 paragraphs ~150 chars each
    parts = _split_with_overlap(paras)
    assert len(parts) >= 2


def test_split_overlap_carries_tail():
    # first paragraph near limit, second starts fresh with overlap
    para1 = "a " * 350  # ~700 chars
    para2 = "b " * 100
    text = para1.strip() + "\n\n" + para2.strip()
    parts = _split_with_overlap(text)
    assert len(parts) == 2
    # overlap: end of para1 appears at start of para2 chunk
    assert parts[0][-10:].strip() in parts[1]


def test_split_empty_string():
    assert _split_with_overlap("") == []


def test_split_whitespace_only():
    assert _split_with_overlap("   \n\n   ") == []


# ── DocumentChunker.chunk ──────────────────────────────────────────────────

c = DocumentChunker()


def test_chunk_empty_content():
    assert c.chunk("file.md", "") == []
    assert c.chunk("file.txt", "   ") == []


def test_chunk_markdown_heading_preserved():
    md = "## Architecture\n\nThe system uses a three-layer graph for memory storage.\n"
    chunks = c.chunk("README.md", md)
    assert len(chunks) == 1
    assert chunks[0].heading == "Architecture"
    assert "three-layer" in chunks[0].text


def test_chunk_markdown_multiple_headings():
    md = (
        "# Overview\n\nThis is the overview section with enough text to pass the minimum.\n\n"
        "## Details\n\nHere are the details of the implementation with more information.\n\n"
        "## Usage\n\nHow to use this library in your project effectively.\n"
    )
    chunks = c.chunk("doc.md", md)
    headings = [ch.heading for ch in chunks]
    assert "Overview" in headings
    assert "Details" in headings
    assert "Usage" in headings


def test_chunk_markdown_no_heading_section():
    md = "Intro text before any heading that is long enough to be kept.\n\n## Section\n\nSection content here, enough text.\n"
    chunks = c.chunk("doc.md", md)
    # first chunk has no heading
    assert chunks[0].heading is None


def test_chunk_markdown_indexes_sequential():
    md = "## A\n\nContent A is here with enough characters.\n\n## B\n\nContent B is here with enough characters.\n"
    chunks = c.chunk("doc.md", md)
    assert [ch.chunk_index for ch in chunks] == list(range(len(chunks)))


def test_chunk_text_paragraph_split():
    text = (
        "First paragraph with meaningful content about memory systems.\n\n"
        "Second paragraph explaining how retrieval works in detail.\n\n"
        "Third paragraph covering storage backends and their tradeoffs."
    )
    chunks = c.chunk("notes.txt", text)
    assert len(chunks) >= 1
    assert all(ch.heading is None for ch in chunks)


def test_chunk_short_chunks_skipped():
    md = "## Section\n\nok\n"  # body too short (< MIN_CHUNK_CHARS=40)
    chunks = c.chunk("doc.md", md)
    assert len(chunks) == 0


def test_chunk_non_md_uses_text_splitter():
    text = "Some content.\n\nMore content here with enough text to be kept by the chunker."
    chunks_txt = c.chunk("notes.txt", text)
    chunks_rst = c.chunk("notes.rst", text)
    assert len(chunks_txt) == len(chunks_rst)
    assert chunks_txt[0].text == chunks_rst[0].text


def test_chunk_path_preserved():
    chunks = c.chunk("/home/user/notes.md", "## Title\n\nLong enough content here to pass minimum.\n")
    assert all(ch.path == "/home/user/notes.md" for ch in chunks)

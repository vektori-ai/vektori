"""Sentence splitting with short-fragment merging."""

from __future__ import annotations

MERGE_STARTERS = {
    "and",
    "but",
    "or",
    "nor",
    "yet",
    "so",
    "for",
    "on",
    "in",
    "at",
    "by",
    "to",
    "from",
    "with",
    "which",
    "who",
    "that",
    "where",
    "when",
}


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences, merging short fragments back into parents.

    Uses NLTK punkt_tab (auto-downloads ~2MB on first use).
    Falls back to naive period split if NLTK is unavailable.
    """
    if not text or not text.strip():
        return []
    return _merge_short_sentences(_nltk_split(text))


def _nltk_split(text: str) -> list[str]:
    """Split via NLTK punkt tokenizer. Downloads punkt_tab on first use."""
    try:
        import nltk

        try:
            return [s for s in nltk.sent_tokenize(text) if s.strip()]
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
            return [s for s in nltk.sent_tokenize(text) if s.strip()]
    except ImportError:
        # Last resort: split on ". "
        return [s.strip() for s in text.split(". ") if s.strip()]


def _merge_short_sentences(sentences: list[str]) -> list[str]:
    """Merge fragments that start with conjunctions/prepositions back into parent."""
    merged: list[str] = []
    for sent in sentences:
        words = sent.split()
        if not words:
            continue
        first_word = words[0].lower().rstrip(",")
        if merged and (first_word in MERGE_STARTERS or len(words) < 4):
            merged[-1] = merged[-1].rstrip(".") + " " + sent
        else:
            merged.append(sent)
    return merged

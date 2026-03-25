"""Sentence splitting with short-fragment merging."""

from __future__ import annotations

_nlp = None

MERGE_STARTERS = {
    "and", "but", "or", "nor", "yet", "so", "for",
    "on", "in", "at", "by", "to", "from", "with",
    "which", "who", "that", "where", "when",
}


def _get_nlp():
    """Lazy-load spacy model. Falls back gracefully if unavailable."""
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except (ImportError, OSError):
            _nlp = False  # mark as unavailable so we don't retry
    return _nlp if _nlp is not False else None


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences, merging short fragments back into parents.

    Uses spacy (en_core_web_sm) if available, falls back to nltk, then
    naive period splitting.
    """
    if not text or not text.strip():
        return []

    nlp = _get_nlp()
    if nlp is not None:
        doc = nlp(text)
        raw = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    else:
        raw = _nltk_split(text)

    return _merge_short_sentences(raw)


def _nltk_split(text: str) -> list[str]:
    """Fallback: nltk sentence tokenizer."""
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

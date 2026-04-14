"""10-layer quality filter for sentences entering the memory graph."""

from __future__ import annotations

import re

from vektori.config import QualityConfig

STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "shall",
    "can",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "his",
    "its",
    "our",
    "their",
    "what",
    "which",
    "who",
    "whom",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "and",
    "but",
    "or",
    "not",
    "no",
    "so",
    "if",
}

PRONOUNS = {
    "this",
    "that",
    "it",
    "they",
    "them",
    "these",
    "those",
    "he",
    "she",
    "we",
    "him",
    "her",
    "us",
}

JUNK_PATTERNS = [
    r"^(ok|okay|sure|yes|no|yeah|yep|nope|hmm|hm|ah|oh|uh|um|lol|haha|thanks|thank you|got it|right|cool|nice|great|fine|alright)\s*[.!?]*$",
    r"^(hey|hi|hello|bye|goodbye|cheers|ciao)\s*[.!?]*$",
    # web/social media cruft
    r"©\s*\d{4}",
    r"(terms of service|privacy policy|cookie policy|accessibility|all rights reserved)",
    r"^\s*(trending|trending now|what'?s happening)\s*$",
    r"^\d[\d,\.]+\s*(views|likes|retweets|replies|reposts|followers|following)\s*$",
    r"^(show more|load more|see more|view more|read more)\s*[.!?]*$",
    r"(sports\s*·\s*trending|entertainment\s*·\s*trending|news\s*·\s*trending)",
    r"^relevant\s+(people|chats|posts)\s*$",
    # pipe-separated nav lists (e.g. "Terms | Privacy | Cookie")
    r"^[^|]{1,40}\|[^|]{1,40}\|",
]

CODE_PATTERNS = [
    r"[{}\[\]<>].*[{}\[\]<>]",
    r"^(import |from |def |class |const |let |var |function )",
    r"[a-zA-Z0-9+/]{40,}",
    r"^(/|\\|[A-Z]:\\)",
    r"https?://\S{50,}",
]

META_PATTERNS = [
    r"^(just for context|for reference|fyi|note:|update:)",
    r":$",
    r"^(explain|tell me about|describe|show me|help me with)\s",
]


def is_quality_sentence(text: str, config: QualityConfig = QualityConfig()) -> bool:
    """
    10-layer quality gauntlet. Returns True if sentence should be stored.

    Filters: length, junk/filler, code/credentials, meta-text,
    pronoun-heavy fragments, low information density.

    Pass config.enabled=False to store everything (some use cases want this).
    """
    if not config.enabled:
        return True

    text_clean = text.strip()
    words = text_clean.lower().split()

    # 1. Length
    if len(text_clean) < config.min_chars or len(words) < config.min_words:
        return False

    # 2. Junk / filler / acknowledgments
    text_lower = text_clean.lower().strip(".,!? ")
    for pattern in JUNK_PATTERNS:
        if re.match(pattern, text_lower, re.IGNORECASE):
            return False

    # 3. Code / credentials / file paths
    for pattern in CODE_PATTERNS:
        if re.search(pattern, text_clean):
            return False

    # 4. Meta-text / vague commands
    for pattern in META_PATTERNS:
        if re.match(pattern, text_lower, re.IGNORECASE):
            return False

    # 5. Pronoun-heavy fragments (likely context-free without surrounding text)
    pronoun_count = sum(1 for w in words if w in PRONOUNS)
    if len(words) > 0 and pronoun_count / len(words) > config.max_pronoun_ratio:
        return False

    # 6. Information density (content words vs stopwords)
    content_words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    if len(words) > 0 and len(content_words) / len(words) < config.min_content_density:
        return False

    return True

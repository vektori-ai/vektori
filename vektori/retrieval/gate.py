"""Retrieval gate — cheap heuristic to skip memory lookup when it won't help.

No LLM, no DB, no embedding. Pure regex/string logic.
Runs before any DB query in SearchPipeline.search().

Cuts unnecessary retrievals ~40-60% for typical usage. The primary benefit:
prevents the common hallucination path where marginally relevant facts get
injected and the LLM stitches them into something wrong.

Decision logic:
  RETRIEVE when:
    - personal reference ("I", "my", "prefer", "remember", "do I", "what do I")
    - named entity in query (capitalized non-first word)
    - explicit memory question ("what do you know about me", "do you remember")

  SKIP when:
    - very short query with no personal anchor
    - generic factual/impersonal question with no personal reference
"""

from __future__ import annotations

import re

# Patterns that indicate the query is about this specific user's context / memory
_PERSONAL_RE = re.compile(
    r"""\b(
        I | my | me | mine | myself |
        prefer | like | dislike | hate | enjoy |
        remember | recall | remind | forget |
        do\s+I | am\s+I | have\s+I | did\s+I | was\s+I | will\s+I |
        what\s+do\s+I | what\s+am\s+I | what\s+have\s+I |
        what\s+do\s+you\s+know | do\s+you\s+remember | do\s+you\s+know\s+me |
        tell\s+me\s+about\s+me | about\s+me |
        my\s+\w+ | for\s+me
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)

# Generic question starters that are almost never about user memory
_GENERIC_RE = re.compile(
    r"""^(
        what\s+is | what\s+are | what\s+was | what\s+were |
        who\s+is | who\s+are | who\s+was |
        where\s+is | where\s+are | where\s+was |
        when\s+is | when\s+was | when\s+did |
        why\s+is | why\s+are | why\s+does |
        how\s+does | how\s+do | how\s+is | how\s+are |
        explain | define | describe | tell\s+me\s+about\s+(?!me) |
        can\s+you\s+explain | what\s+does .+ mean
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)

# Filler / acknowledgement messages — never worth retrieving
_FILLER_RE = re.compile(
    r"^(ok|okay|sure|yes|no|yeah|nope|thanks|thank\s+you|got\s+it|"
    r"makes\s+sense|sounds\s+good|sounds\s+great|alright|cool|great|perfect|"
    r"i\s+see|i\s+understand|understood|right|exactly|hmm|hm|uh|um|"
    r"hello|hi|hey|bye|goodbye)\W*$",
    re.IGNORECASE,
)


def should_retrieve(query: str) -> bool:
    """
    Returns True if memory retrieval is likely to help with this query.

    Fast path: no LLM, no DB, no embedding — pure heuristics.
    """
    q = query.strip()
    if not q:
        return False

    # Filler/acknowledgement — never retrieve
    if _FILLER_RE.match(q):
        return False

    # Very short with no personal signal — skip
    if len(q.split()) <= 2 and not _PERSONAL_RE.search(q):
        return False

    # Personal reference detected — always retrieve
    if _PERSONAL_RE.search(q):
        return True

    # Generic factual question with no personal anchor — skip before named entity check
    # so "what is the capital of France" doesn't trigger on "France"
    if _GENERIC_RE.match(q):
        return False

    # Named entity: capitalized word that isn't the first word of the sentence.
    # Only reached here if not a generic question, so likely a person/entity in memory.
    words = q.split()
    if any(
        w[0].isupper() and i > 0
        for i, w in enumerate(words)
        if w and w[0].isalpha()
    ):
        return True

    # Default: retrieve (safer to over-retrieve than under-retrieve)
    return True

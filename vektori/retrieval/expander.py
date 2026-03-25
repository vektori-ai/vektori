"""Query expansion — generates paraphrase variants before fact search.

Used by the expanded search path: original query + N variants are searched
concurrently at L0 (facts only), results merged, then a single L1 graph
traversal runs on the union. This bridges the paraphrase gap between how
facts are stored ("user prefers async communication") and how they're queried
("how does the user like to communicate").

Only runs when the caller explicitly passes expand=True. Direct searches
(expand=False) skip this entirely.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from vektori.models.base import LLMProvider

logger = logging.getLogger(__name__)

EXPANSION_PROMPT = """Generate {n} alternative phrasings of this memory query.

Query: {query}

Return JSON:
{{"queries": ["variant 1", "variant 2"]}}

Rules:
- Each variant captures the same intent with different wording
- Mix question forms ("what does user prefer") and statement fragments ("user preference for X")
- Think about how the memory might have been stored vs how it is being asked
- Keep each variant under 15 words
- Do NOT add new meaning or assumptions

Return ONLY the JSON."""


class QueryExpander:
    """
    Generates N paraphrase variants of a query using an LLM.

    The original query is always included in the returned list so callers
    can treat the output as a drop-in replacement for the query string.

    expand("what food does the user like", n=2) might return:
        [
            "what food does the user like",    ← original, always first
            "user's food preferences",
            "food that user enjoys eating",
        ]
    """

    def __init__(self, llm: LLMProvider, n_variants: int = 2) -> None:
        self.llm = llm
        self.n_variants = n_variants

    async def expand(self, query: str) -> list[str]:
        """
        Returns [original_query, variant_1, ..., variant_n].

        Falls back to [original_query] on any failure — expansion is best-effort,
        never blocks retrieval.
        """
        try:
            prompt = EXPANSION_PROMPT.format(query=query, n=self.n_variants)
            # Small output — just a JSON list of short strings
            response = await self.llm.generate(prompt, max_tokens=256)
            variants = _parse_variants(response)
        except Exception as e:
            logger.warning("Query expansion failed, using original: %s", e)
            return [query]

        # Deduplicate while preserving order; original always first
        seen: set[str] = {query.lower()}
        result = [query]
        for v in variants:
            v = v.strip()
            if v and v.lower() not in seen:
                seen.add(v.lower())
                result.append(v)

        logger.debug("Expanded %r → %d queries", query[:40], len(result))
        return result


def _parse_variants(response: str) -> list[str]:
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        start = 1
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[start:end])
    try:
        data: Any = json.loads(text)
        queries = data.get("queries", [])
        return [str(q) for q in queries if q]
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Failed to parse expansion JSON: %.200s", response)
        return []

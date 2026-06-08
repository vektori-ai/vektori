"""Document fact extractor — LLM extraction or direct chunk embedding."""

from __future__ import annotations

import json
import logging
import math
from typing import TYPE_CHECKING, Any

from vektori.fsmemory.models import FileChunk

if TYPE_CHECKING:
    from vektori.models.base import EmbeddingProvider, LLMProvider
    from vektori.storage.base import StorageBackend

logger = logging.getLogger(__name__)

_EXTRACT_PROMPT = """\
Extract 3-5 concise facts from this document chunk.
Each fact must be a self-contained statement a person could read in isolation.
Use specific names, values, and quantities verbatim. No pronouns.

FILE: {path}
{heading_line}CONTENT:
{chunk_text}

Return JSON only:
{{"facts": [{{"text": "...", "subject": "main entity or topic"}}]}}"""

_DEDUP_THRESHOLD = 0.92


class DocumentExtractor:
    def __init__(
        self,
        llm: LLMProvider,
        embedder: EmbeddingProvider,
        extract: bool = True,
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.extract = extract

    async def process_chunk(
        self,
        chunk: FileChunk,
        user_id: str,
        session_id: str,
        db: StorageBackend,
    ) -> int:
        """Extract facts from a chunk and insert into db. Returns count inserted."""
        if self.extract:
            fact_dicts = await self._extract_facts(chunk)
        else:
            fact_dicts = [{"text": chunk.text, "subject": None}]

        if not fact_dicts:
            return 0

        texts = [f["text"] for f in fact_dicts]
        embeddings = await self.embedder.embed_batch(texts)

        inserted = 0
        for fact, embedding in zip(fact_dicts, embeddings):
            if await self._is_duplicate(embedding, user_id, db):
                continue
            await db.insert_fact(
                text=fact["text"],
                embedding=embedding,
                user_id=user_id,
                session_id=session_id,
                subject=fact.get("subject"),
                metadata={
                    "source_path": chunk.path,
                    "source_type": "filesystem",
                    "chunk_index": chunk.chunk_index,
                    "heading": chunk.heading,
                },
            )
            inserted += 1

        return inserted

    async def _extract_facts(self, chunk: FileChunk) -> list[dict[str, Any]]:
        heading_line = f"HEADING: {chunk.heading}\n" if chunk.heading else ""
        prompt = _EXTRACT_PROMPT.format(
            path=chunk.path,
            heading_line=heading_line,
            chunk_text=chunk.text[:2000],
        )
        try:
            raw = await self.llm.complete(prompt, max_tokens=512)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            return [f for f in data.get("facts", []) if isinstance(f.get("text"), str) and f["text"].strip()]
        except Exception as e:
            logger.warning("Fact extraction failed for %s chunk %d: %s", chunk.path, chunk.chunk_index, e)
            return []

    async def _is_duplicate(
        self,
        embedding: list[float],
        user_id: str,
        db: StorageBackend,
    ) -> bool:
        candidates = await db.search_facts(embedding=embedding, user_id=user_id, limit=1)
        if not candidates:
            return False
        dist = candidates[0].get("distance", 1.0)
        sim = 1.0 - dist
        return sim >= _DEDUP_THRESHOLD


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

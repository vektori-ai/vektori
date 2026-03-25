"""Synchronous ingestion orchestrator: split → filter → embed → store → schedule extraction."""

from __future__ import annotations

from typing import Any

from vektori.config import QualityConfig
from vektori.ingestion.filter import is_quality_sentence
from vektori.ingestion.hasher import generate_sentence_id
from vektori.ingestion.splitter import split_sentences
from vektori.models.base import EmbeddingProvider
from vektori.storage.base import StorageBackend
from vektori.utils.async_worker import ExtractionRequest, ExtractionWorker


class IngestionPipeline:
    """
    Fast synchronous ingestion path. Returns immediately after storing sentences.

    Synchronous (user waits):
      split → quality filter → deterministic IDs → batch embed
      → upsert sentences + NEXT edges → upsert session

    Asynchronous (background, user does NOT wait):
      LLM fact + insight extraction via ExtractionWorker (debounced, batched)

    If async_extraction=False, extraction runs inline (slower, for tests/simple use cases).
    """

    def __init__(
        self,
        db: StorageBackend,
        embedder: EmbeddingProvider,
        extractor: Any,
        quality_config: QualityConfig | None = None,
        async_extraction: bool = True,
    ) -> None:
        self.db = db
        self.embedder = embedder
        self.extractor = extractor
        self.quality_config = quality_config or QualityConfig()
        self.async_extraction = async_extraction
        self.worker: ExtractionWorker | None = (
            ExtractionWorker(extractor) if async_extraction else None
        )

    async def ingest(
        self,
        messages: list[dict[str, str]],
        session_id: str,
        user_id: str,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Ingest a conversation session.

        Returns:
            {"status": "ok", "sentences_stored": N, "extraction": "queued"|"done"|"skipped"}
        """
        # 1. Split all user messages into quality-filtered sentences
        all_sentences: list[dict[str, Any]] = []
        for turn_num, msg in enumerate(messages):
            if msg["role"] != "user":
                continue
            raw_sents = split_sentences(msg["content"])
            for idx, text in enumerate(raw_sents):
                if (
                    not self.quality_config.enabled
                    or is_quality_sentence(text, self.quality_config)
                ):
                    all_sentences.append({
                        "text": text,
                        "session_id": session_id,
                        "turn_number": turn_num,
                        "sentence_index": idx,
                        "role": "user",
                        "id": generate_sentence_id(session_id, f"{turn_num}_{idx}", text),
                    })

        if not all_sentences:
            return {"status": "ok", "sentences_stored": 0, "extraction": "skipped"}

        # 2. Batch embed
        texts = [s["text"] for s in all_sentences]
        embeddings = await self.embedder.embed_batch(texts)

        # 3. Upsert sentences (ON CONFLICT → increment mentions for IDF weighting)
        await self.db.upsert_sentences(all_sentences, embeddings, user_id, agent_id)

        # 4. Insert sequential NEXT edges within this session
        edges = [
            {
                "source_id": all_sentences[i]["id"],
                "target_id": all_sentences[i + 1]["id"],
                "edge_type": "next",
                "weight": 1.0,
            }
            for i in range(len(all_sentences) - 1)
        ]
        if edges:
            await self.db.insert_edges(edges)

        # 5. Upsert session record
        await self.db.upsert_session(session_id, user_id, agent_id, metadata or {})

        # 6. Trigger extraction
        if self.worker is not None:
            self.worker.schedule(ExtractionRequest(
                messages=messages,
                session_id=session_id,
                user_id=user_id,
                agent_id=agent_id,
            ))
            extraction_status = "queued"
        else:
            await self.extractor.extract(messages, session_id, user_id, agent_id)
            extraction_status = "done"

        return {
            "status": "ok",
            "sentences_stored": len(all_sentences),
            "extraction": extraction_status,
        }

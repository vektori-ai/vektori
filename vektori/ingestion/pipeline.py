"""Synchronous ingestion orchestrator: split → filter → embed → store → schedule extraction."""

from __future__ import annotations

import asyncio
from datetime import datetime
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
      LLM fact + episode extraction via ExtractionWorker (debounced, batched)

    If async_extraction=False, extraction runs inline (slower, for tests/simple use cases).
    """

    def __init__(
        self,
        db: StorageBackend,
        embedder: EmbeddingProvider,
        extractor: Any,
        quality_config: QualityConfig | None = None,
        async_extraction: bool = True,
        token_batch_threshold: int = 800,
    ) -> None:
        self.db = db
        self.embedder = embedder
        self.extractor = extractor
        self.quality_config = quality_config or QualityConfig()
        self.async_extraction = async_extraction
        self.worker: ExtractionWorker | None = (
            ExtractionWorker(extractor, token_threshold=token_batch_threshold)
            if async_extraction
            else None
        )

    async def ingest(
        self,
        messages: list[dict[str, str]],
        session_id: str,
        user_id: str,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        session_time: datetime | None = None,
        skip_extraction: bool = False,
    ) -> dict[str, Any]:
        """
        Ingest a conversation session.

        Returns:
            {"status": "ok", "sentences_stored": N, "extraction": "queued"|"done"|"skipped"}
        """
        # 1. Split user AND assistant messages into stored sentence records.
        #    Low-quality sentences are still stored for provenance, but marked
        #    non-searchable so sentence retrieval can continue to skip junk.
        all_sentences: list[dict[str, Any]] = []
        for turn_num, msg in enumerate(messages):
            role = msg.get("role", "")
            if role not in ("user", "assistant"):
                continue
            raw_sents = split_sentences(msg["content"])
            for idx, text in enumerate(raw_sents):
                all_sentences.append(
                    {
                        "text": text,
                        "session_id": session_id,
                        "turn_number": turn_num,
                        "sentence_index": idx,
                        "role": role,
                        "id": generate_sentence_id(session_id, f"{turn_num}_{idx}", text),
                        "event_time": session_time.isoformat() if session_time else None,
                        "is_searchable": is_quality_sentence(text, self.quality_config),
                    }
                )

        # 2. Batch embed sentences, then upsert sentences + edges + session in parallel.
        #    insert_edges and upsert_session don't need embeddings — sentence IDs are
        #    deterministic hashes computed above, so all three can run concurrently.
        edges = [
            {
                "source_id": all_sentences[i]["id"],
                "target_id": all_sentences[i + 1]["id"],
                "edge_type": "next",
                "weight": 1.0,
            }
            for i in range(len(all_sentences) - 1)
        ]
        if all_sentences:
            texts = [s["text"] for s in all_sentences]
            embeddings = await self.embedder.embed_batch(texts)
            await asyncio.gather(
                self.db.upsert_sentences(all_sentences, embeddings, user_id, agent_id),
                self.db.insert_edges(edges) if edges else asyncio.sleep(0),
                self.db.upsert_session(session_id, user_id, agent_id, metadata or {}, started_at=session_time),
            )
        else:
            await self.db.upsert_session(
                session_id, user_id, agent_id, metadata or {}, started_at=session_time
            )

        # 5. Trigger extraction — always runs as long as messages is non-empty.
        #    Extraction still sees raw messages, but now also gets the exact
        #    stored sentence catalog used for deterministic source-linking.
        sentence_catalog = [
            {
                "id": sent["id"],
                "text": sent["text"],
                "role": sent.get("role", "user"),
                "turn_number": sent["turn_number"],
                "sentence_index": sent["sentence_index"],
                "is_searchable": sent.get("is_searchable", True),
                "event_time": sent.get("event_time"),
            }
            for sent in all_sentences
        ]
        if skip_extraction or not messages:
            extraction_status = "skipped"
        elif self.worker is not None:
            schedule_result = self.worker.schedule(
                ExtractionRequest(
                    messages=messages,
                    session_id=session_id,
                    user_id=user_id,
                    agent_id=agent_id,
                    sentence_catalog=sentence_catalog,
                    session_time=session_time,
                )
            )
            extraction_status = schedule_result.status
        else:
            await self.extractor.extract(
                messages,
                session_id,
                user_id,
                agent_id,
                sentence_catalog=sentence_catalog,
                session_time=session_time,
            )
            extraction_status = "done"

        return {
            "status": "ok",
            "sentences_stored": len(all_sentences),
            "extraction": extraction_status,
        }

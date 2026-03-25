"""
BGE-M3 embedding provider via FlagEmbedding.

BGE-M3 is a multilingual embedding model from BAAI:
  - 1024-dim embeddings
  - Supports 100+ languages
  - Dense + sparse + multi-vector retrieval
  - Strong benchmark performance, fully local

Install: pip install FlagEmbedding

Usage:
    v = Vektori(embedding_model="bge:BAAI/bge-m3")
"""

from __future__ import annotations

import asyncio
import logging

from vektori.models.base import EmbeddingProvider

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-m3"


class BGEEmbedder(EmbeddingProvider):
    """
    BGE-M3 embeddings via FlagEmbedding.

    Fully local, no API keys, multilingual, 1024-dim.
    First run downloads the model from HuggingFace (~2.3GB).

    Install: pip install FlagEmbedding
    """

    def __init__(self, model: str | None = None, use_fp16: bool = True) -> None:
        self.model_name = model or DEFAULT_MODEL
        self.use_fp16 = use_fp16
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from FlagEmbedding import BGEM3FlagModel
            except ImportError as e:
                if "FlagEmbedding" in str(e) or "No module named" in str(e):
                    raise ImportError(
                        "FlagEmbedding required: pip install FlagEmbedding"
                    ) from e
                raise  # surface real errors (e.g. transformers version conflicts)
            self._model = BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16)
        return self._model

    @property
    def dimension(self) -> int:
        return 1024  # BGE-M3 dense vector dimension

    async def embed(self, text: str) -> list[float]:
        model = self._get_model()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: model.encode([text], batch_size=1, max_length=512)["dense_vecs"][0],
        )
        return result.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: model.encode(texts, batch_size=32, max_length=512)["dense_vecs"],
        )
        return [r.tolist() for r in results]

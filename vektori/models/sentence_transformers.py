"""Sentence-transformers embedding provider. Fully local, no Ollama needed."""

from __future__ import annotations

import asyncio
import logging

from vektori.models.base import EmbeddingProvider

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"

MODEL_DIMENSIONS = {
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "all-MiniLM-L12-v2": 384,
    "paraphrase-multilingual-MiniLM-L12-v2": 384,
    "BAAI/bge-m3": 1024,
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-base-en-v1.5": 768,
}


class SentenceTransformerEmbedder(EmbeddingProvider):
    """
    HuggingFace sentence-transformers. Fully local, no Ollama, no API keys.

    Install: pip install 'vektori[sentence-transformers]'
    First run downloads the model from HuggingFace (~90MB for MiniLM).
    """

    def __init__(self, model: str | None = None) -> None:
        self.model_name = model or DEFAULT_MODEL
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers required: pip install 'vektori[sentence-transformers]'"
                ) from e
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return MODEL_DIMENSIONS.get(self.model_name, 384)

    async def embed(self, text: str) -> list[float]:
        model = self._get_model()
        # sentence-transformers is sync — run in executor to not block event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, model.encode, text)
        return result.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, model.encode, texts)
        return [r.tolist() for r in results]

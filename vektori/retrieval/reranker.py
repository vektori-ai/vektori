"""Cross-encoder reranker for post-retrieval scoring.

Requires sentence-transformers: pip install sentence-transformers
Falls back silently when not installed — retrieval still works, just without neural reranking.

Usage:
    reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    facts = reranker.rerank(query, facts, top_n=20)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reranks retrieval results using a cross-encoder model.

    Cross-encoders score (query, passage) pairs jointly, capturing interaction
    effects that bi-encoder similarity misses. Applied after RRF fusion on the
    top-N candidates before final top-k selection.

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (6-layer, ~70ms/batch on CPU).
    Larger: cross-encoder/ms-marco-MiniLM-L-12-v2 (~130ms, better recall).
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model_name
        self._model: Any = None
        self._available: bool | None = None

    def _load(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
            self._available = True
            logger.info("CrossEncoderReranker loaded: %s", self._model_name)
        except ImportError:
            logger.info(
                "sentence-transformers not installed — cross-encoder reranker disabled. "
                "Install with: pip install sentence-transformers"
            )
            self._available = False
        except Exception as e:
            logger.warning("CrossEncoderReranker load failed (%s): %s", self._model_name, e)
            self._available = False
        return self._available

    def rerank(
        self,
        query: str,
        facts: list[dict[str, Any]],
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank facts by cross-encoder score descending.

        Adds '_rerank_score' to each fact dict. If top_n is set, truncates
        after reranking. Returns input unchanged when model unavailable.
        """
        if not facts or not self._load():
            return facts[:top_n] if top_n else facts

        pairs = [(query, f.get("text", "")) for f in facts]
        try:
            scores = self._model.predict(pairs)
        except Exception as e:
            logger.warning("Cross-encoder prediction failed: %s", e)
            return facts[:top_n] if top_n else facts

        for fact, score in zip(facts, scores):
            fact["_rerank_score"] = float(score)

        reranked = sorted(facts, key=lambda f: f.get("_rerank_score", 0.0), reverse=True)
        return reranked[:top_n] if top_n else reranked

    @property
    def available(self) -> bool:
        return self._load()

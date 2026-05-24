"""
BGE-M3 embedding provider via FlagEmbedding.

BGE-M3 is a multilingual embedding model from BAAI:
  - 1024-dim embeddings
  - Supports 100+ languages
  - Dense + sparse + multi-vector retrieval
  - Strong benchmark performance, fully local

Runs inference in a persistent subprocess so OOM-killed worker doesn't
take down the calling process. Worker auto-restarts on crash.

Install: pip install FlagEmbedding

Usage:
    v = Vektori(embedding_model="bge:BAAI/bge-m3")
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
from typing import Any

from vektori.models.base import EmbeddingProvider

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-m3"


def _worker_main(model_name: str, in_q: Any, out_q: Any) -> None:
    """Runs in subprocess. Holds model, processes batches until None sentinel."""
    import gc
    import ctypes
    import torch
    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel(model_name, use_fp16=False)
    # bfloat16 halves weight memory (~1.06 GB vs 2.12 GB); supported on modern CPUs
    try:
        model.model = model.model.to(torch.bfloat16)
    except Exception:
        pass  # stay float32 if bfloat16 unsupported

    while True:
        texts = in_q.get()
        if texts is None:
            break
        try:
            with torch.inference_mode():
                out = model.encode(
                    texts,
                    batch_size=4,
                    max_length=512,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                )
            out_q.put(("ok", [v.tolist() for v in out["dense_vecs"]]))
        except Exception as e:
            out_q.put(("err", str(e)))
        finally:
            gc.collect()
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass


class BGEEmbedder(EmbeddingProvider):
    """
    BGE-M3 embeddings via FlagEmbedding, isolated in a subprocess worker.

    The model runs in a separate process so an OOM-kill only takes out the
    worker, not the calling process. The worker restarts automatically.

    Fully local, no API keys, multilingual, 1024-dim.
    First run downloads the model from HuggingFace (~2.3GB).

    Install: pip install FlagEmbedding
    """

    def __init__(self, model: str | None = None, use_fp16: bool | None = None) -> None:
        self.model_name = model or DEFAULT_MODEL
        self._worker: mp.Process | None = None
        self._in_q: mp.Queue | None = None
        self._out_q: mp.Queue | None = None

    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        if self._worker is not None:
            logger.warning("BGE worker died (exitcode=%s), restarting", self._worker.exitcode)
            self._worker = None
        ctx = mp.get_context("spawn")
        self._in_q = ctx.Queue()
        self._out_q = ctx.Queue()
        self._worker = ctx.Process(
            target=_worker_main,
            args=(self.model_name, self._in_q, self._out_q),
            daemon=True,
        )
        self._worker.start()
        logger.info("BGE worker started (PID %s)", self._worker.pid)

    @property
    def dimension(self) -> int:
        return 1024

    async def embed(self, text: str) -> list[float]:
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._embed_sync, texts)

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        self._ensure_worker()
        assert self._in_q and self._out_q
        self._in_q.put(texts)
        status, payload = self._out_q.get(timeout=300)
        if status == "err":
            raise RuntimeError(f"BGE worker error: {payload}")
        return payload

    def shutdown(self) -> None:
        if self._worker and self._worker.is_alive() and self._in_q:
            self._in_q.put(None)
            self._worker.join(timeout=10)
        if self._worker and self._worker.is_alive():
            self._worker.kill()
        self._worker = None

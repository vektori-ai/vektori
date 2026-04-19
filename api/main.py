from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from vektori import Vektori

logger = logging.getLogger(__name__)

_API_KEYS_DDL = """
CREATE TABLE IF NOT EXISTS api_keys (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    key_hash     VARCHAR(64) NOT NULL UNIQUE,
    owner_id     UUID NOT NULL DEFAULT gen_random_uuid(),
    name         TEXT,
    created_at   TIMESTAMPTZ DEFAULT NOW(),
    last_used_at TIMESTAMPTZ,
    revoked_at   TIMESTAMPTZ
);
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is required")

    # Separate asyncpg pool for auth lookups (api_keys table)
    pool = await asyncpg.create_pool(db_url, min_size=2, max_size=5)
    async with pool.acquire() as conn:
        try:
            await conn.execute(_API_KEYS_DDL)
        except asyncpg.UniqueViolationError:
            pass  # another worker already created the table simultaneously — fine
    app.state.pool = pool

    embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers:all-MiniLM-L6-v2")
    extraction_model = os.getenv("EXTRACTION_MODEL", "gemini:gemini-2.5-flash-lite")
    embedding_dim = int(os.getenv("EMBEDDING_DIM", "384"))

    # min_retrieval_score: lower threshold needed for smaller models
    # (all-MiniLM scores ~0.1-0.3 vs OpenAI text-embedding-3-small ~0.4-0.8)
    min_score = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.1"))

    # Single shared Vektori instance (asyncpg pool internally)
    v = Vektori(
        database_url=db_url,
        embedding_model=embedding_model,
        extraction_model=extraction_model,
        embedding_dimension=embedding_dim,
        async_extraction=True,
    )
    v.config.min_retrieval_score = min_score  # set before _ensure_initialized()
    await v._ensure_initialized()
    app.state.vektori = v

    logger.info("Vektori API ready")
    yield

    await v.close()
    await pool.close()


app = FastAPI(title="Vektori API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

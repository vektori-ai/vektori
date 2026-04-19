from __future__ import annotations

import hashlib
import os
import secrets
from typing import Any

import asyncpg
from fastapi import HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer = HTTPBearer(auto_error=False)


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


async def require_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> str:
    """Validate Bearer API key, attach owner_id to request.state. Returns owner_id."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing API key")

    key = credentials.credentials
    key_hash = _hash(key)

    pool: asyncpg.Pool = request.app.state.pool
    row = await pool.fetchrow(
        """
        UPDATE api_keys
        SET last_used_at = NOW()
        WHERE key_hash = $1 AND revoked_at IS NULL
        RETURNING owner_id
        """,
        key_hash,
    )
    if not row:
        raise HTTPException(status_code=401, detail="Invalid or revoked API key")

    owner_id = str(row["owner_id"])
    request.state.owner_id = owner_id
    return owner_id


async def require_admin_key(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> None:
    admin_key = os.getenv("ADMIN_KEY")
    if not admin_key:
        raise HTTPException(status_code=500, detail="ADMIN_KEY not configured")
    if not credentials or credentials.credentials != admin_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")


def generate_api_key() -> str:
    return "vk_sk_" + secrets.token_hex(32)


async def create_api_key(pool: asyncpg.Pool, name: str | None = None) -> dict[str, Any]:
    plaintext = generate_api_key()
    key_hash = _hash(plaintext)
    row = await pool.fetchrow(
        "INSERT INTO api_keys (key_hash, name) VALUES ($1, $2) RETURNING id, owner_id, name, created_at",
        key_hash,
        name,
    )
    return {
        "key": plaintext,
        "owner_id": str(row["owner_id"]),
        "name": row["name"],
        "created_at": row["created_at"].isoformat(),
    }

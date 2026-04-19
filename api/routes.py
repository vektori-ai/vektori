from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from api.auth import create_api_key, require_admin_key, require_api_key

router = APIRouter()


# ── Request models ────────────────────────────────────────────────────────────

class AddRequest(BaseModel):
    messages: list[dict[str, str]]
    session_id: str
    user_id: str
    agent_id: str | None = None
    metadata: dict[str, Any] | None = None
    session_time: datetime | None = None


class SearchRequest(BaseModel):
    query: str
    user_id: str
    depth: str = "l1"
    top_k: int = 10
    expand: bool = False
    include_superseded: bool = False


class CreateKeyRequest(BaseModel):
    name: str | None = None


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/health")
async def health(request: Request) -> dict[str, str]:
    return {"status": "ok", "backend": "postgres"}


@router.post("/v1/add")
async def add(
    body: AddRequest,
    request: Request,
    owner_id: str = Depends(require_api_key),
) -> dict[str, Any]:
    v = request.app.state.vektori
    return await v.add(
        messages=body.messages,
        session_id=body.session_id,
        user_id=body.user_id,
        agent_id=owner_id,
        metadata=body.metadata,
        session_time=body.session_time,
    )


@router.post("/v1/search")
async def search(
    body: SearchRequest,
    request: Request,
    owner_id: str = Depends(require_api_key),
) -> dict[str, Any]:
    v = request.app.state.vektori
    return await v.search(
        query=body.query,
        user_id=body.user_id,
        agent_id=owner_id,
        depth=body.depth,
        top_k=body.top_k,
        expand=body.expand,
        include_superseded=body.include_superseded,
    )


@router.get("/v1/profile/{user_id}")
async def get_profile(
    user_id: str,
    request: Request,
    owner_id: str = Depends(require_api_key),
) -> dict[str, Any]:
    v = request.app.state.vektori
    profile = await v.get_profile(user_id=user_id, agent_id=owner_id)
    return {"user_id": user_id, "profile": profile}


@router.delete("/v1/user/{user_id}")
async def delete_user(
    user_id: str,
    request: Request,
    owner_id: str = Depends(require_api_key),
) -> dict[str, Any]:
    v = request.app.state.vektori
    # Scope deletion to this tenant's data only
    deleted = await v.db.delete_user_scoped(user_id, owner_id)
    return {"deleted_rows": deleted}


@router.post("/v1/admin/keys")
async def admin_create_key(
    body: CreateKeyRequest,
    request: Request,
    _: None = Depends(require_admin_key),
) -> dict[str, Any]:
    return await create_api_key(request.app.state.pool, name=body.name)

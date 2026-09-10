"""GET /api/health — actually pings dependencies. Never hardcode 'ok'."""
from __future__ import annotations

from fastapi import APIRouter, Request

from app.adapters.vectors import ping_qdrant
from app.db import ping_db

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request) -> dict:
    postgres_ok = await ping_db()
    qdrant_ok = await ping_qdrant()
    embedding_loaded = bool(getattr(request.app.state, "embedding_model_loaded", False))

    status = "ok" if (postgres_ok and qdrant_ok and embedding_loaded) else "degraded"
    return {
        "status": status,
        "postgres": "ok" if postgres_ok else "unreachable",
        "qdrant": "ok" if qdrant_ok else "unreachable",
        "embedder": "loaded" if embedding_loaded else "not_loaded",
        "embedding_model": {
            "name": getattr(request.app.state, "embedding_model_name", None),
            "loaded": embedding_loaded,
        },
    }

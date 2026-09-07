"""GET /api/stats — admin overview. No business logic here."""
from __future__ import annotations

from fastapi import APIRouter

from app.schemas import StatsResponse
from app.services import ingestion

router = APIRouter(tags=["stats"])


@router.get("/stats", response_model=StatsResponse)
async def get_stats() -> StatsResponse:
    stats = await ingestion.get_stats()
    return StatsResponse(**stats)

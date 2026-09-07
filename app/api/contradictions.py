"""GET/PATCH /api/contradictions. No business logic here — validate, call
one service function, serialise."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.schemas import ContradictionListResponse, ContradictionPatchRequest, ContradictionRecordOut
from app.services import contradictions as contradictions_service

router = APIRouter(tags=["contradictions"])


@router.get("/contradictions", response_model=ContradictionListResponse)
async def list_contradictions(
    status: str | None = Query(default="open"),
    severity: str | None = Query(default=None),
    type_filter: str | None = Query(default=None, alias="type"),
    document_id: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> ContradictionListResponse:
    """Grouped by default: N documents disagreeing pairwise on one fact
    produce C(N,2) distinct, individually-correct pairwise records — this
    groups them into one displayed conflict per underlying disagreement.
    Every individual pairwise record is still present, inside each group's
    `evidence[]`; PATCH targets an evidence item's own id, not group_id."""
    groups, counts, total = await contradictions_service.list_contradiction_groups(
        db,
        status=status,
        severity=severity,
        type_=type_filter,
        document_id=document_id,
        limit=limit,
        offset=offset,
    )
    return ContradictionListResponse(contradictions=groups, counts=counts, total=total, limit=limit, offset=offset)


@router.get("/contradictions/{contradiction_id}", response_model=ContradictionRecordOut)
async def get_contradiction(
    contradiction_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> ContradictionRecordOut:
    record = await contradictions_service.get_contradiction(db, contradiction_id)
    return ContradictionRecordOut.model_validate(record)


@router.patch("/contradictions/{contradiction_id}", response_model=ContradictionRecordOut)
async def patch_contradiction(
    contradiction_id: uuid.UUID,
    payload: ContradictionPatchRequest,
    db: AsyncSession = Depends(get_db),
) -> ContradictionRecordOut:
    record = await contradictions_service.update_contradiction_status(
        db, contradiction_id, payload.status, payload.note
    )
    return ContradictionRecordOut.model_validate(record)

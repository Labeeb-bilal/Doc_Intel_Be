"""Document routes. No business logic here — validate, call one service
function, serialise."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, File, Query, UploadFile
from fastapi import status as http_status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.schemas import ChunkOut, ChunksResponse, DocumentOut, DocumentUploadItem, UploadResponse
from app.services import ingestion

router = APIRouter(tags=["documents"])


@router.post("/documents", status_code=http_status.HTTP_202_ACCEPTED, response_model=UploadResponse)
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
) -> UploadResponse:
    items = await ingestion.upload_documents(files, db, background_tasks)
    return UploadResponse(documents=[DocumentUploadItem(**item) for item in items])


@router.get("/documents", response_model=list[DocumentOut])
async def list_documents(
    status_filter: str | None = Query(default=None, alias="status"),
    db: AsyncSession = Depends(get_db),
) -> list[DocumentOut]:
    documents = await ingestion.list_documents(db, status=status_filter)
    return [DocumentOut.model_validate(d) for d in documents]


@router.get("/documents/{document_id}", response_model=DocumentOut)
async def get_document(document_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> DocumentOut:
    document = await ingestion.get_document(db, document_id)
    return DocumentOut.model_validate(document)


@router.get("/documents/{document_id}/chunks", response_model=ChunksResponse)
async def get_document_chunks(
    document_id: uuid.UUID,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> ChunksResponse:
    chunks, total = await ingestion.get_document_chunks(db, document_id, limit, offset)
    return ChunksResponse(chunks=[ChunkOut(**c) for c in chunks], total=total, limit=limit, offset=offset)


@router.delete("/documents/{document_id}", status_code=http_status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> None:
    await ingestion.delete_document(db, document_id)


@router.post("/documents/{document_id}/retry", response_model=DocumentOut)
async def retry_document(
    document_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> DocumentOut:
    document = await ingestion.retry_document(db, document_id, background_tasks)
    return DocumentOut.model_validate(document)

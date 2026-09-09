"""POST /api/chat. No business logic here — validate, call one service
function, serialise."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.schemas import ChatRequest, ChatResponse
from app.services import chat as chat_service

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def send_chat_message(payload: ChatRequest, db: AsyncSession = Depends(get_db)) -> ChatResponse:
    options = payload.options
    result = await chat_service.send_message(
        db,
        None,  # resolved inside send_message so a missing key hits the same 503 path as a mid-call failure
        query=payload.query,
        conversation_id=payload.conversation_id,
        history=payload.history,
        top_k=options.top_k if options else None,
        rerank_enabled=options.rerank if options else None,
        document_ids=options.document_ids if options else None,
        detect_contradictions=options.detect_contradictions if options else True,
    )
    return ChatResponse(**result)

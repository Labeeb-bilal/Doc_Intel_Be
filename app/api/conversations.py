"""GET /api/conversations, GET /api/conversations/{id}/messages. No
business logic here — validate, call one service function, serialise."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.schemas import ConversationMessagesResponse, ConversationOut, ConversationsResponse, MessageOut, RetrievalTrace
from app.services import chat as chat_service

router = APIRouter(tags=["conversations"])


@router.get("/conversations", response_model=ConversationsResponse)
async def list_conversations(db: AsyncSession = Depends(get_db)) -> ConversationsResponse:
    conversations = await chat_service.list_conversations(db)
    return ConversationsResponse(conversations=[ConversationOut(**c) for c in conversations])


@router.get("/conversations/{conversation_id}/messages", response_model=ConversationMessagesResponse)
async def get_conversation_messages(
    conversation_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> ConversationMessagesResponse:
    messages = await chat_service.get_conversation_messages(db, conversation_id)
    return ConversationMessagesResponse(
        conversation_id=conversation_id, messages=[MessageOut.model_validate(m) for m in messages]
    )


@router.get("/conversations/{conversation_id}/messages/{message_id}/trace", response_model=RetrievalTrace)
async def get_message_trace(
    conversation_id: uuid.UUID, message_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> RetrievalTrace:
    """The full trace behind one assistant turn — all retrieval candidates,
    every rerank result, the full contradiction filter_log. POST /chat and
    GET .../messages both return a trimmed summary; this is where the rest
    of it lives, for a "why this answer" debug view or similar."""
    return await chat_service.get_message_trace(db, conversation_id, message_id)

"""Chat service — orchestrates retrieval + rag for POST /api/chat, and the
conversation/message read operations for the two GET endpoints."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import structlog
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.llm import LLMClient, LLMUnavailableError, get_llm_client
from app.config import get_settings
from app.errors import AppError, NotFoundError
from app.models import Conversation, Document, Message
from app.schemas import HistoryMessage, RetrievalTrace, summarize_trace
from app.services import rag
from app.services.retrieval import retrieve

log = structlog.get_logger("chat")

_TITLE_MAX_LEN = 80
_MAX_QUERY_CHARS = 2000


async def send_message(
    db: AsyncSession,
    llm: LLMClient | None,
    *,
    query: str,
    conversation_id: uuid.UUID | None,
    top_k: int | None,
    rerank_enabled: bool | None,
    document_ids: list[str] | None,
    detect_contradictions: bool = True,
    history: list[HistoryMessage] | None = None,
) -> dict:
    history = history or []
    query = query.strip()
    if not query:
        raise AppError("Query must not be empty.", code="EMPTY_QUERY", status_code=422)
    if len(query) > _MAX_QUERY_CHARS:
        raise AppError(
            f"Query must be {_MAX_QUERY_CHARS} characters or fewer.",
            code="QUERY_TOO_LONG",
            status_code=422,
            details={"max_length": _MAX_QUERY_CHARS, "actual_length": len(query)},
        )

    any_ready = await db.scalar(select(Document.id).where(Document.status == "ready").limit(1))
    if any_ready is None:
        raise AppError("No documents have finished ingesting yet.", code="NO_DOCUMENTS", status_code=409)

    if document_ids:
        try:
            doc_uuids = [uuid.UUID(d) for d in document_ids]
        except ValueError as exc:
            raise AppError(f"Invalid document id: {exc}", code="INVALID_DOCUMENT_ID", status_code=422) from exc
        found = {str(f) for f in (await db.execute(select(Document.id).where(Document.id.in_(doc_uuids)))).scalars()}
        missing = [d for d in document_ids if d not in found]
        if missing:
            raise NotFoundError(
                f"Unknown document_ids: {missing}", code="DOCUMENT_NOT_FOUND", details={"missing": missing}
            )

    if conversation_id is not None:
        conversation = await db.get(Conversation, conversation_id)
        if conversation is None:
            raise NotFoundError(f"Conversation {conversation_id} not found.", code="CONVERSATION_NOT_FOUND")
    else:
        conversation = Conversation(title=query[:_TITLE_MAX_LEN])
        db.add(conversation)
        await db.flush()  # populate conversation.id without committing yet

    try:
        # Resolving the real client here (not in the route) means a
        # missing/dead API key hits the exact same except block below as a
        # mid-call failure — one consistent translation to the error
        # envelope instead of two.
        llm = llm or get_llm_client()

        # Retrieval runs exactly once, on the raw query alone — no
        # conversational history feeds this. rag.answer() mutates
        # result.trace in place (context/answer/total_ms) and decides
        # whether to call the LLM at all; `history` reaches the LLM prompt
        # through it, not through retrieval.
        result = await retrieve(
            query,
            top_k=top_k,
            rerank_enabled=rerank_enabled,
            document_ids=document_ids,
        )
        settings = get_settings()
        outcome = await rag.answer(
            llm,
            result,
            db=db,
            history=history,
            detect_contradictions=detect_contradictions and settings.contradiction_enabled,
        )
    except LLMUnavailableError as exc:
        # Never leave a dangling empty conversation row (new-conversation
        # case) or a half-written turn behind on this failure.
        await db.rollback()
        if exc.rate_limited:
            raise AppError(
                "The LLM provider is rate-limiting requests. Please retry shortly.",
                code="RATE_LIMITED",
                status_code=429,
                headers={"Retry-After": "30"},
            ) from exc
        raise AppError(
            "The answer service is temporarily unavailable. Please try again shortly.",
            code="LLM_UNAVAILABLE",
            status_code=503,
        ) from exc

    # outcome["contradictions"] is already list[ContradictionGroupOut] —
    # services/rag.py's contradiction branch groups and builds the display
    # view itself (it already owns `db` for the whole branch). Persist the
    # id of every individual pairwise record shown, across every group's
    # evidence, not just one id per group — that's the full evidence trail
    # this turn surfaced, and what a past turn re-renders from later.
    contradiction_groups = outcome["contradictions"]
    contradiction_ids = [uuid.UUID(ev.id) for group in contradiction_groups for ev in group.evidence]

    # Both rows are inserted in the same transaction, and Postgres's now()
    # is frozen for the whole transaction — the server_default alone would
    # give both messages the identical created_at, making their relative
    # order (which GET /conversations/{id}/messages relies on) undefined.
    # Set them explicitly, in Python, a beat apart.
    user_created_at = datetime.now(timezone.utc)
    user_message = Message(conversation_id=conversation.id, role="user", content=query, created_at=user_created_at)
    assistant_message = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=outcome["answer"],
        citations=[c.model_dump() for c in outcome["citations"]],
        trace=result.trace.model_dump(),
        contradiction_ids=contradiction_ids or None,
        created_at=datetime.now(timezone.utc),
    )
    db.add(user_message)
    db.add(assistant_message)
    # Both messages are persisted together, only once the turn actually
    # succeeded — "Persists both the user and assistant messages" means as
    # a pair, not the user's question alone if generation then fails.
    await db.execute(update(Conversation).where(Conversation.id == conversation.id).values(updated_at=func.now()))
    await db.commit()
    await db.refresh(assistant_message)

    return {
        "message_id": str(assistant_message.id),
        "conversation_id": str(conversation.id),
        "answer": outcome["answer"],
        "citations": outcome["citations"],
        "contradictions": contradiction_groups,
        "contradictions_total": outcome["contradictions_total"],
        # The full trace (all retrieval candidates, every rerank result,
        # the full contradiction filter_log) is persisted below unabridged
        # — this response only carries the summary a frontend renders.
        # Fetch the full thing via get_message_trace() /
        # GET /conversations/{id}/messages/{message_id}/trace when needed.
        "trace": summarize_trace(result.trace),
        "grounded": outcome["grounded"],
    }


async def list_conversations(db: AsyncSession) -> list[dict]:
    stmt = (
        select(
            Conversation.id,
            Conversation.title,
            Conversation.created_at,
            Conversation.updated_at,
            func.count(Message.id).label("message_count"),
        )
        .outerjoin(Message, Message.conversation_id == Conversation.id)
        .group_by(Conversation.id)
        .order_by(Conversation.updated_at.desc())
    )
    rows = (await db.execute(stmt)).all()
    return [
        {
            "id": row.id,
            "title": row.title,
            "message_count": row.message_count,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }
        for row in rows
    ]


async def get_conversation_messages(db: AsyncSession, conversation_id: uuid.UUID) -> list[Message]:
    conversation = await db.get(Conversation, conversation_id)
    if conversation is None:
        raise NotFoundError(f"Conversation {conversation_id} not found.", code="CONVERSATION_NOT_FOUND")
    stmt = select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at.asc())
    return list((await db.execute(stmt)).scalars().all())


async def get_message_trace(db: AsyncSession, conversation_id: uuid.UUID, message_id: uuid.UUID) -> RetrievalTrace:
    """The full, un-trimmed trace for one assistant turn — every retrieval
    candidate, every rerank result, the full contradiction filter_log.
    Read straight off the Message row; nothing recomputed. This is the
    "add it to another api" half of trimming POST /chat's own response:
    the detail still exists, just one explicit fetch away instead of on
    every turn by default."""
    message = await db.get(Message, message_id)
    if message is None or message.conversation_id != conversation_id:
        raise NotFoundError(
            f"Message {message_id} not found in conversation {conversation_id}.", code="MESSAGE_NOT_FOUND"
        )
    if message.trace is None:
        # User messages (and any assistant turn that somehow predates
        # tracing) simply have nothing to show here.
        raise NotFoundError(f"Message {message_id} has no trace.", code="TRACE_NOT_FOUND")
    return RetrievalTrace.model_validate(message.trace)

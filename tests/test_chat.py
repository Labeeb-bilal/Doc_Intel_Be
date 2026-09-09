"""Stage 4 checkpoint + spec test #5: chat endpoint contract — creates a
conversation, persists both messages, returns a complete trace. Uses the
real Postgres/Qdrant/embedder/reranker already running for this dev corpus
(mirrors test_ingestion_pipeline.py's pattern); only the LLM is faked."""
from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError
from sqlalchemy import select

from app.adapters.llm import LLMUnavailableError
from app.adapters.vectors import ping_qdrant
from app.db import async_session_factory, ping_db
from app.errors import AppError, NotFoundError
from app.models import Conversation, Document
from app.schemas import ChatRequest, RetrievalTrace, RetrievalTraceSummary
from app.services import chat as chat_service
from tests.fakes import FakeLLMClient


def test_chat_request_rejects_assistant_history():
    # No TestClient/`client` fixture exists in this suite (every test here
    # calls services directly against live infra) — this is the equivalent
    # unit-level check for what FastAPI's 422 translation actually rests
    # on: role: Literal["user"] on HistoryMessage rejects anything else at
    # the schema boundary, before any service code runs.
    with pytest.raises(ValidationError):
        ChatRequest(query="follow up", history=[{"role": "assistant", "content": "previous answer"}])


def test_chat_request_rejects_history_over_limit():
    with pytest.raises(ValidationError):
        ChatRequest(
            query="follow up",
            history=[
                {"role": "user", "content": "q1"},
                {"role": "user", "content": "q2"},
                {"role": "user", "content": "q3"},  # over the max_length=2 cap
            ],
        )


async def test_chat_works_with_no_history_field():
    # send_message's `history` parameter defaults to None -> [] and every
    # call site below this omits it entirely, the same way every pre-existing
    # test in this file already does — this just names that guarantee
    # explicitly: omitting `history` is not a degraded path.
    llm = FakeLLMClient(response="Employees may work remotely. [S1]")

    async with async_session_factory() as db:
        result = await chat_service.send_message(
            db,
            llm,
            query="What is the remote work policy?",
            conversation_id=None,
            top_k=None,
            rerank_enabled=None,
            document_ids=None,
        )

    assert result["conversation_id"]
    assert result["message_id"]
    assert result["answer"]

    conv_id = uuid.UUID(result["conversation_id"])
    async with async_session_factory() as db:
        conv = await db.get(Conversation, conv_id)
        await db.delete(conv)
        await db.commit()


@pytest.fixture(autouse=True)
async def _require_live_services_and_documents():
    if not (await ping_db() and await ping_qdrant()):
        pytest.skip("requires docker-compose Postgres + Qdrant running")
    async with async_session_factory() as db:
        any_ready = await db.scalar(select(Document.id).where(Document.status == "ready").limit(1))
    if any_ready is None:
        pytest.skip("requires at least one ready document ingested")


async def test_empty_query_is_rejected_before_touching_the_db():
    llm = FakeLLMClient()
    async with async_session_factory() as db:
        with pytest.raises(AppError) as exc_info:
            await chat_service.send_message(
                db, llm, query="   ", conversation_id=None, top_k=None, rerank_enabled=None, document_ids=None
            )
    assert exc_info.value.code == "EMPTY_QUERY"
    assert exc_info.value.status_code == 422
    assert llm.calls == []


async def test_chat_creates_conversation_and_persists_both_messages():
    llm = FakeLLMClient(response="Employees may work remotely. [S1]")

    async with async_session_factory() as db:
        result = await chat_service.send_message(
            db,
            llm,
            query="How many days per week can employees work remotely?",
            conversation_id=None,
            top_k=None,
            rerank_enabled=None,
            document_ids=None,
        )

    assert result["conversation_id"]
    assert result["message_id"]
    assert result["answer"]
    assert result["trace"].total_ms is not None
    assert result["trace"].answer is not None
    assert result["trace"].retrieval is not None

    conv_id = uuid.UUID(result["conversation_id"])
    async with async_session_factory() as db:
        messages = await chat_service.get_conversation_messages(db, conv_id)

    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[0].content == "How many days per week can employees work remotely?"
    assert messages[1].role == "assistant"
    assert messages[1].trace is not None  # full trace persisted, re-renderable later
    assert messages[1].citations is not None

    async with async_session_factory() as db:
        conv = await db.get(Conversation, conv_id)
        await db.delete(conv)
        await db.commit()


async def test_llm_failure_rolls_back_and_persists_nothing():
    class _DyingLLM:
        calls: list = []

        async def complete(self, *, system: str, user: str) -> str:
            raise LLMUnavailableError("simulated provider outage")

        async def complete_structured(self, *, system, user, schema):
            raise NotImplementedError

    async with async_session_factory() as db:
        before = len((await db.execute(select(Conversation.id))).all())
        with pytest.raises(AppError) as exc_info:
            await chat_service.send_message(
                db,
                _DyingLLM(),
                query="How many days per week can employees work remotely?",
                conversation_id=None,
                top_k=None,
                rerank_enabled=None,
                document_ids=None,
            )
        assert exc_info.value.code == "LLM_UNAVAILABLE"
        assert exc_info.value.status_code == 503

    async with async_session_factory() as db:
        after = len((await db.execute(select(Conversation.id))).all())

    assert after == before  # no orphan conversation left behind


async def test_rate_limited_llm_failure_returns_429_with_retry_after():
    class _RateLimitedLLM:
        async def complete(self, *, system: str, user: str) -> str:
            # rate_limited=True is what GroqClient sets when the
            # underlying failure was specifically a 429 after retries
            # were exhausted — this is the flag chat.py branches on.
            raise LLMUnavailableError("rate limited upstream", rate_limited=True)

        async def complete_structured(self, *, system, user, schema):
            raise NotImplementedError

    async with async_session_factory() as db:
        before = len((await db.execute(select(Conversation.id))).all())
        with pytest.raises(AppError) as exc_info:
            await chat_service.send_message(
                db,
                _RateLimitedLLM(),
                query="How many days per week can employees work remotely?",
                conversation_id=None,
                top_k=None,
                rerank_enabled=None,
                document_ids=None,
            )
        assert exc_info.value.code == "RATE_LIMITED"
        assert exc_info.value.status_code == 429
        assert exc_info.value.headers.get("Retry-After") == "30"

    async with async_session_factory() as db:
        after = len((await db.execute(select(Conversation.id))).all())

    assert after == before  # no orphan conversation left behind here either


async def test_chat_response_trace_is_trimmed_of_debug_detail():
    """The response returned to the frontend carries a RetrievalTraceSummary
    — no candidates/results/filter_log arrays — while the full detail
    stays persisted on the Message row, unabridged."""
    llm = FakeLLMClient(response="Employees may work remotely. [S1]")

    async with async_session_factory() as db:
        result = await chat_service.send_message(
            db,
            llm,
            query="How many days per week can employees work remotely?",
            conversation_id=None,
            top_k=None,
            rerank_enabled=None,
            document_ids=None,
        )

    assert isinstance(result["trace"], RetrievalTraceSummary)
    assert not hasattr(result["trace"].retrieval, "candidates")
    if result["trace"].rerank is not None:
        assert not hasattr(result["trace"].rerank, "results")
        assert not hasattr(result["trace"].rerank, "dropped")

    conv_id = uuid.UUID(result["conversation_id"])
    async with async_session_factory() as db:
        conv = await db.get(Conversation, conv_id)
        await db.delete(conv)
        await db.commit()


async def test_get_message_trace_returns_full_untrimmed_detail():
    """The companion endpoint for anyone who does want the debug detail:
    reads the full RetrievalTrace straight back off the persisted row,
    candidates and all."""
    llm = FakeLLMClient(response="Employees may work remotely. [S1]")

    async with async_session_factory() as db:
        result = await chat_service.send_message(
            db,
            llm,
            query="How many days per week can employees work remotely?",
            conversation_id=None,
            top_k=None,
            rerank_enabled=None,
            document_ids=None,
        )

    conv_id = uuid.UUID(result["conversation_id"])
    msg_id = uuid.UUID(result["message_id"])

    async with async_session_factory() as db:
        full_trace = await chat_service.get_message_trace(db, conv_id, msg_id)

    assert isinstance(full_trace, RetrievalTrace)
    assert full_trace.retrieval.candidates  # the part the trimmed response drops
    assert full_trace.trace_id == result["trace"].trace_id

    async with async_session_factory() as db:
        conv = await db.get(Conversation, conv_id)
        await db.delete(conv)
        await db.commit()


async def test_get_message_trace_rejects_message_outside_the_conversation():
    llm = FakeLLMClient(response="Employees may work remotely. [S1]")

    async with async_session_factory() as db:
        result = await chat_service.send_message(
            db,
            llm,
            query="How many days per week can employees work remotely?",
            conversation_id=None,
            top_k=None,
            rerank_enabled=None,
            document_ids=None,
        )

    msg_id = uuid.UUID(result["message_id"])
    wrong_conversation_id = uuid.uuid4()

    async with async_session_factory() as db:
        with pytest.raises(NotFoundError):
            await chat_service.get_message_trace(db, wrong_conversation_id, msg_id)

    async with async_session_factory() as db:
        conv = await db.get(Conversation, uuid.UUID(result["conversation_id"]))
        await db.delete(conv)
        await db.commit()

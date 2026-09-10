"""Spec tests #5/#6/#7: cache behaviour, status preservation, and the
degradation ladder. These touch the real Postgres (for upsert/cache), like
test_ingestion_pipeline.py and test_chat.py before them — only the LLM is
faked."""
from __future__ import annotations

import uuid

import pytest

from app.adapters.llm import LLMUnavailableError
from app.adapters.vectors import ping_qdrant
from app.db import async_session_factory, ping_db
from app.models import Chunk, Document
from app.schemas import ContradictionBatch, ContradictionVerdict, RetrievalResult, RetrievalStage, RetrievalTrace, ScoredChunk
from app.services import contradictions as cx
from app.services import rag
from tests.fakes import FakeLLMClient


@pytest.fixture(autouse=True)
async def _require_live_db():
    if not (await ping_db() and await ping_qdrant()):
        pytest.skip("requires docker-compose Postgres + Qdrant running")


async def _make_real_chunk(db, *, text: str, vector: list[float]) -> ScoredChunk:
    """The `contradictions` table has real FK constraints to `documents`
    and `chunks` — a synthetic chunk_id/document_id with no backing row
    would fail the insert. Creates minimal real rows (no ingestion
    pipeline needed) and returns the ScoredChunk the pipeline consumes."""
    document = Document(
        original_filename="fixture.txt",
        file_type="txt",
        size_bytes=len(text),
        content_hash=uuid.uuid4().hex,
        storage_key=str(uuid.uuid4()),
        status="ready",
    )
    db.add(document)
    await db.flush()

    chunk = Chunk(
        id=uuid.uuid4(),
        document_id=document.id,
        ordinal=0,
        page_start=1,
        page_end=1,
        section_path=[],
        char_count=len(text),
    )
    db.add(chunk)
    await db.flush()
    await db.commit()

    return ScoredChunk(
        chunk_id=str(chunk.id),
        document_id=str(document.id),
        document_name=document.original_filename,
        text=text,
        page_start=1,
        page_end=1,
        section_path=[],
        ordinal=0,
        vector=vector,
        vector_score=0.9,
        rank_before=0,
        rank_after=0,
    )


async def _cleanup_document(document_id: str) -> None:
    async with async_session_factory() as db:
        doc = await db.get(Document, uuid.UUID(document_id))
        if doc is not None:
            await db.delete(doc)
            await db.commit()


def _chunk(chunk_id: str, document_id: str, text: str, vector: list[float]) -> ScoredChunk:
    """Purely synthetic — fine for tests that never write a Contradiction
    row to the DB (no FK constraint to satisfy)."""
    return ScoredChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        document_name=f"{document_id}.pdf",
        text=text,
        page_start=1,
        page_end=1,
        section_path=[],
        ordinal=0,
        vector=vector,
        vector_score=0.9,
        rank_before=0,
        rank_after=0,
    )


def _batch_with_one_contradiction(pair_id: str = "P1") -> ContradictionBatch:
    return ContradictionBatch(
        results=[
            ContradictionVerdict(
                pair_id=pair_id,
                is_contradiction=True,
                type="factual",
                severity="critical",
                confidence=0.9,
                statement_a="The fee is $50 per transaction.",
                statement_b="The fee is $75 per transaction.",
                explanation="Fees differ.",
                reconciliation="none",
            )
        ]
    )


async def test_repeat_detection_reuses_cache_increments_times_seen_zero_llm_calls():
    async with async_session_factory() as db:
        chunk_a = await _make_real_chunk(db, text="The fee is $50 per transaction.", vector=[1.0, 0.0, 0.0])
        chunk_b = await _make_real_chunk(db, text="The fee is $75 per transaction.", vector=[0.9, 0.1, 0.0])

    llm = FakeLLMClient(structured_response=_batch_with_one_contradiction())

    try:
        async with async_session_factory() as db:
            found_1, stage_1 = await cx.detect_contradictions(
                db, llm, [chunk_a, chunk_b], "q1", sim_min=0.5, sim_max=0.97, max_pairs=8, min_confidence=0.5
            )
        assert stage_1.llm_calls == 1
        assert len(found_1) == 1
        assert found_1[0].times_seen == 1

        structured_calls_after_first = len([c for c in llm.calls if c["kind"] == "structured"])

        async with async_session_factory() as db:
            found_2, stage_2 = await cx.detect_contradictions(
                db, llm, [chunk_a, chunk_b], "q2", sim_min=0.5, sim_max=0.97, max_pairs=8, min_confidence=0.5
            )

        assert stage_2.llm_calls == 0
        assert stage_2.cached_verdicts == 1
        assert len(found_2) == 1
        assert found_2[0].times_seen == 2
        structured_calls_after_second = len([c for c in llm.calls if c["kind"] == "structured"])
        assert structured_calls_after_second == structured_calls_after_first
    finally:
        await _cleanup_document(chunk_a.document_id)
        await _cleanup_document(chunk_b.document_id)


async def test_resolved_status_is_not_reset_on_rediscovery():
    async with async_session_factory() as db:
        chunk_a = await _make_real_chunk(db, text="The fee is $50 per transaction.", vector=[1.0, 0.0, 0.0])
        chunk_b = await _make_real_chunk(db, text="The fee is $75 per transaction.", vector=[0.9, 0.1, 0.0])

    llm = FakeLLMClient(structured_response=_batch_with_one_contradiction())

    try:
        async with async_session_factory() as db:
            found, _ = await cx.detect_contradictions(
                db, llm, [chunk_a, chunk_b], "q1", sim_min=0.5, sim_max=0.97, max_pairs=8, min_confidence=0.5
            )
        record_id = found[0].id

        async with async_session_factory() as db:
            resolved = await cx.update_contradiction_status(db, record_id, "resolved", "Confirmed, tracked elsewhere.")
        assert resolved.status == "resolved"
        assert resolved.resolved_at is not None

        async with async_session_factory() as db:
            found_again, stage_again = await cx.detect_contradictions(
                db, llm, [chunk_a, chunk_b], "q2", sim_min=0.5, sim_max=0.97, max_pairs=8, min_confidence=0.5
            )

        assert stage_again.llm_calls == 0
        assert len(found_again) == 1
        assert found_again[0].status == "resolved"
        assert found_again[0].times_seen == 2
    finally:
        await _cleanup_document(chunk_a.document_id)
        await _cleanup_document(chunk_b.document_id)


def _make_result(selected: list[ScoredChunk], query: str = "some question") -> RetrievalResult:
    trace = RetrievalTrace(
        trace_id="t1",
        query_raw=query,
        query_used_for_retrieval=query,
        condensed=False,
        retrieval=RetrievalStage(top_k=20, returned=len(selected), latency_ms=10, candidates=[]),
    )
    return RetrievalResult(query=query, candidates=selected, selected=selected, trace=trace)


async def test_contradiction_engine_exception_still_returns_complete_answer(monkeypatch):
    chunk_a = _chunk(str(uuid.uuid4()), str(uuid.uuid4()), "Some claim.", [1.0, 0.0])
    chunk_b = _chunk(str(uuid.uuid4()), str(uuid.uuid4()), "Another claim.", [0.9, 0.1])
    result = _make_result([chunk_a, chunk_b])

    llm = FakeLLMClient(response="This is the grounded answer. [S1]")

    async def _boom(*args, **kwargs):
        raise RuntimeError("contradiction engine exploded")

    monkeypatch.setattr(cx, "detect_contradictions", _boom)

    async with async_session_factory() as db:
        outcome = await rag.answer(llm, result, db=db, detect_contradictions=True)

    assert outcome["answer"] == "This is the grounded answer. [S1]"
    assert outcome["grounded"] is True
    assert outcome["contradictions"] == []
    assert outcome["contradictions_total"] == 0
    assert result.trace.contradiction_check is not None
    assert result.trace.contradiction_check.enabled is False
    assert "contradiction engine exploded" in result.trace.contradiction_check.error


class _RateLimitedStructuredLLM:
    """FakeLLMClient variant whose complete_structured specifically raises
    LLMUnavailableError, as GroqClient/GeminiClient do on a 429/outage —
    distinct from a generic exception (RuntimeError, above), which is the
    other, unrelated degradation path."""

    def __init__(self, *, rate_limited: bool):
        self._rate_limited = rate_limited
        self.calls: list[dict] = []

    async def complete(self, *, system: str, user: str) -> str:
        self.calls.append({"kind": "complete"})
        return "answer text [S1]"

    async def complete_structured(self, *, system, user, schema):
        self.calls.append({"kind": "structured"})
        raise LLMUnavailableError("simulated provider outage", rate_limited=self._rate_limited)


async def test_llm_failure_inside_detection_preserves_real_pair_counts():
    async with async_session_factory() as db:
        chunk_a = await _make_real_chunk(db, text="Employees may work remotely 3 days per week.", vector=[1.0, 0.0])
        chunk_b = await _make_real_chunk(db, text="Employees may work remotely 2 days per week.", vector=[0.95, 0.05])

    llm = _RateLimitedStructuredLLM(rate_limited=True)

    try:
        async with async_session_factory() as db:
            found, stage = await cx.detect_contradictions(
                db, llm, [chunk_a, chunk_b], "q1", sim_min=0.5, sim_max=0.999, max_pairs=8, min_confidence=0.5
            )

        assert stage.enabled is True
        assert stage.pairs_generated == 1
        assert stage.pairs_after_same_doc_filter == 1
        assert stage.pairs_after_cosine == 1
        assert stage.llm_calls == 1
        assert stage.error is not None
        assert stage.error.startswith("LLM rate limited:")
        assert found == []
    finally:
        await _cleanup_document(chunk_a.document_id)
        await _cleanup_document(chunk_b.document_id)


async def test_llm_failure_still_returns_previously_cached_contradictions():
    async with async_session_factory() as db:
        chunk_a = await _make_real_chunk(db, text="The fee is $50 per transaction.", vector=[1.0, 0.0, 0.0])
        chunk_b = await _make_real_chunk(db, text="The fee is $75 per transaction.", vector=[0.9, 0.1, 0.0])

    working_llm = FakeLLMClient(structured_response=_batch_with_one_contradiction())

    try:
        async with async_session_factory() as db:
            found_1, stage_1 = await cx.detect_contradictions(
                db, working_llm, [chunk_a, chunk_b], "q1", sim_min=0.5, sim_max=0.97, max_pairs=8, min_confidence=0.5
            )
        assert len(found_1) == 1
        assert stage_1.error is None

        dying_llm = _RateLimitedStructuredLLM(rate_limited=False)
        async with async_session_factory() as db:
            found_2, stage_2 = await cx.detect_contradictions(
                db, dying_llm, [chunk_a, chunk_b], "q2", sim_min=0.5, sim_max=0.97, max_pairs=8, min_confidence=0.5
            )

        assert len(found_2) == 1
        assert found_2[0].id == found_1[0].id
        assert stage_2.llm_calls == 0
        assert stage_2.error is None
        assert dying_llm.calls == []
    finally:
        await _cleanup_document(chunk_a.document_id)
        await _cleanup_document(chunk_b.document_id)

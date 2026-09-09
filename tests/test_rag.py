"""Stage 3 checkpoint + spec tests #1/#2: citation validation (including
the grouped-citation form real models actually produce) and the relevance
floor skipping the LLM entirely. Also covers build_user_prompt (the
answer-LLM's conversational-history prompt shape — history reaches the
prompt only, never retrieval). Fully offline via FakeLLMClient."""
from __future__ import annotations

from app.prompts.answer import build_user_prompt
from app.schemas import HistoryMessage, RetrievalResult, RetrievalStage, RetrievalTrace, ScoredChunk
from app.services import rag
from tests.fakes import FakeLLMClient


def _make_chunk(chunk_id: str, text: str = "Some chunk text.") -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        document_name="policy.pdf",
        text=text,
        page_start=1,
        page_end=1,
        section_path=["Eligibility"],
        ordinal=0,
        vector_score=0.9,
        rank_before=0,
        rank_after=0,
    )


def _make_result(selected: list[ScoredChunk], query: str = "some question") -> RetrievalResult:
    trace = RetrievalTrace(
        trace_id="t1",
        query_raw=query,
        query_used_for_retrieval=query,
        condensed=False,
        retrieval=RetrievalStage(top_k=20, returned=len(selected), latency_ms=10, candidates=[]),
    )
    return RetrievalResult(query=query, candidates=selected, selected=selected, trace=trace)


def test_build_user_prompt_with_history():
    history = [HistoryMessage(role="user", content="What is the expense threshold?")]
    result = build_user_prompt(history=history, context="[S1] policy text...", query="What about in Europe?")
    assert "Previous question: What is the expense threshold?" in result
    assert "Current question: What about in Europe?" in result
    assert result.index("Previous question") < result.index("Current question")


def test_build_user_prompt_no_history():
    result = build_user_prompt(history=[], context="[S1] text...", query="What is the timeout?")
    assert "Previous question" not in result
    assert "Current question: What is the timeout?" in result


async def test_hallucinated_citation_is_dropped_and_counted():
    chunks = [_make_chunk(f"c{i}") for i in range(1, 7)]  # six real sources, S1..S6
    result = _make_result(chunks)
    llm = FakeLLMClient(response="The policy allows this. [S2] It also says that. [S7]")

    outcome = await rag.answer(llm, result)

    assert "[S2]" in outcome["answer"]
    assert "[S7]" not in outcome["answer"]  # hallucinated marker must not render
    assert len(outcome["citations"]) == 1
    assert outcome["citations"][0].marker == "S2"
    assert result.trace.answer.citations_dropped == 1


async def test_grouped_citation_validates_each_marker_independently():
    # Real models group citations like "[S1, S2, S9]" rather than always
    # emitting one bracket per marker — the parser must validate each
    # marker in the group on its own, not treat the whole group as one unit.
    chunks = [_make_chunk("c1"), _make_chunk("c2"), _make_chunk("c3")]  # S1, S2, S3
    result = _make_result(chunks)
    llm = FakeLLMClient(response="Employees may do this. [S1, S2, S9]")

    outcome = await rag.answer(llm, result)

    assert "[S1, S2]" in outcome["answer"]
    assert "S9" not in outcome["answer"]
    assert {c.marker for c in outcome["citations"]} == {"S1", "S2"}
    assert result.trace.answer.citations_dropped == 1


async def test_grouped_citation_dropped_entirely_when_all_markers_invalid():
    chunks = [_make_chunk("c1")]  # only S1 exists
    result = _make_result(chunks)
    llm = FakeLLMClient(response="A claim. [S5, S9]")

    outcome = await rag.answer(llm, result)

    assert "[S5, S9]" not in outcome["answer"]
    assert "S5" not in outcome["answer"] and "S9" not in outcome["answer"]
    assert outcome["citations"] == []
    assert result.trace.answer.citations_dropped == 2


async def test_relevance_floor_skips_llm_and_is_not_grounded():
    result = _make_result([])  # nothing cleared the floor

    llm = FakeLLMClient(response="should never be seen")

    outcome = await rag.answer(llm, result)

    assert llm.calls == []  # the LLM must never be called
    assert outcome["grounded"] is False
    assert outcome["citations"] == []
    assert outcome["answer"] == rag.NOT_FOUND_ANSWER


async def test_answer_with_no_citations_at_all_is_not_grounded():
    chunks = [_make_chunk("c1")]
    result = _make_result(chunks)
    llm = FakeLLMClient(response="I don't have enough information to answer that from the sources.")

    outcome = await rag.answer(llm, result)

    assert outcome["citations"] == []
    assert outcome["grounded"] is False


async def test_context_truncates_when_budget_exceeded(monkeypatch):
    from app.config import get_settings

    long_text = "word " * 2000  # comfortably over a tiny budget
    chunks = [_make_chunk("c1", long_text), _make_chunk("c2", long_text)]
    result = _make_result(chunks)
    llm = FakeLLMClient(response="ok [S1]")

    monkeypatch.setattr(get_settings(), "max_context_tokens", 50)

    outcome = await rag.answer(llm, result)

    assert result.trace.context.truncated is True
    assert len(result.trace.context.chunks_used) < 2
    assert outcome["citations"][0].marker == "S1"

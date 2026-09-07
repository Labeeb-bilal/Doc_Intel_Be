"""RAG service — context assembly, answer generation, and citation
validation. services/retrieval.py never calls the LLM; this is the one
place that does, and the one place that decides whether to call it at all.
"""
from __future__ import annotations

import asyncio
import re
import time

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.llm import LLMClient
from app.config import get_settings
from app.prompts.answer import ANSWER_SYSTEM_PROMPT, build_context_block, build_user_message, format_source_label
from app.schemas import (
    AnswerStage,
    Citation,
    ContextStage,
    ContradictionGroupOut,
    ContradictionStage,
    RetrievalResult,
    ScoredChunk,
)

log = structlog.get_logger("rag")

# Models naturally group a claim's citations into one bracket, e.g.
# "[S1, S2, S3]" as well as single "[S1]" — both are matched, and each
# marker inside a group is validated individually.
_CITATION_GROUP_RE = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")
_MARKER_RE = re.compile(r"S\d+")

NOT_FOUND_ANSWER = "I couldn't find anything about that in your documents."


def _estimate_tokens(text: str) -> int:
    """~4 chars/token approximation for English text. Deliberately not a
    real tokenizer: a network round-trip to Gemini's count_tokens API just
    to budget context would be wasteful, and there's no official public
    Gemini tokenizer package to call locally instead — any third-party
    tokenizer (e.g. tiktoken) would be counting for the wrong model anyway."""
    return max(1, len(text) // 4)


def _assemble_context(
    selected: list[ScoredChunk], max_tokens: int
) -> tuple[str, dict[str, ScoredChunk], int, bool]:
    """Builds the [S1]/[S2]/... labeled context block. `selected` is
    already best-first (post-rerank); lowest-ranked sources are dropped
    first if the token budget is exceeded."""
    source_map: dict[str, ScoredChunk] = {}
    labels: list[str] = []
    total_tokens = 0
    truncated = False

    for i, chunk in enumerate(selected):
        marker = f"S{i + 1}"
        section = " > ".join(chunk.section_path) if chunk.section_path else None
        label = format_source_label(
            marker=marker, document_name=chunk.document_name, page=chunk.page_start, section=section, text=chunk.text
        )
        label_tokens = _estimate_tokens(label)
        if labels and total_tokens + label_tokens > max_tokens:
            truncated = True
            break
        source_map[marker] = chunk
        labels.append(label)
        total_tokens += label_tokens

    return build_context_block(labels), source_map, total_tokens, truncated


def _validate_citations(answer_text: str, source_map: dict[str, ScoredChunk]) -> tuple[str, list[Citation], int]:
    """Drop any [Sn] marker that doesn't resolve to a supplied source — a
    hallucinated [S7] against six real sources must not render. Handles
    both a lone marker ("[S1]") and a grouped citation ("[S1, S2, S3]"),
    validating each marker in a group independently: valid ones stay in
    the bracket, invalid ones are dropped from it, and the whole bracket
    disappears only if every marker in it was hallucinated."""
    dropped = 0
    citations: list[Citation] = []
    seen: set[str] = set()

    def _replace(match: re.Match) -> str:
        nonlocal dropped
        valid_markers: list[str] = []
        for marker in _MARKER_RE.findall(match.group(1)):
            chunk = source_map.get(marker)
            if chunk is None:
                dropped += 1
                continue
            valid_markers.append(marker)
            if marker not in seen:
                seen.add(marker)
                citations.append(
                    Citation(
                        marker=marker,
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        document_name=chunk.document_name,
                        page=chunk.page_start,
                        section=" > ".join(chunk.section_path) if chunk.section_path else None,
                        text=chunk.text,
                    )
                )
        if not valid_markers:
            return ""
        return "[" + ", ".join(valid_markers) + "]"

    cleaned = _CITATION_GROUP_RE.sub(_replace, answer_text)
    return cleaned, citations, dropped


async def _generate_answer(llm: LLMClient, result: RetrievalResult) -> dict:
    settings = get_settings()
    overall_start = time.perf_counter()

    context_block, source_map, total_tokens, truncated = _assemble_context(
        result.selected, settings.max_context_tokens
    )
    result.trace.context = ContextStage(
        chunks_used=[c.chunk_id for c in source_map.values()],
        neighbour_expansion=settings.neighbour_expansion,
        total_tokens=total_tokens,
        truncated=truncated,
    )

    user_message = build_user_message(query=result.query, context_block=context_block)

    answer_start = time.perf_counter()
    raw_answer = await llm.complete(system=ANSWER_SYSTEM_PROMPT, user=user_message)
    answer_latency_ms = round((time.perf_counter() - answer_start) * 1000)

    cleaned_answer, citations, citations_dropped = _validate_citations(raw_answer, source_map)
    grounded = len(citations) > 0

    result.trace.answer = AnswerStage(
        model=settings.llm_model,
        latency_ms=answer_latency_ms,
        citations=[c.marker for c in citations],
        citations_dropped=citations_dropped,
        grounded=grounded,
    )
    result.trace.total_ms = round((time.perf_counter() - overall_start) * 1000)

    if citations_dropped:
        log.warning("citations_dropped", count=citations_dropped, trace_id=result.trace.trace_id)

    return {"answer": cleaned_answer, "citations": citations, "grounded": grounded, "contradictions": [], "contradictions_total": 0}


def _not_found_result(result: RetrievalResult) -> dict:
    """Nothing cleared the relevance floor. No LLM calls at all — neither
    branch runs. This is the guard against the worst RAG failure
    (confabulating a plausible answer from thin context), and it saves two
    requests against the rate limit, not one. This check happens here,
    before the fork, not inside either branch."""
    result.trace.context = ContextStage(chunks_used=[], neighbour_expansion=False, total_tokens=0, truncated=False)
    result.trace.answer = AnswerStage(model="", latency_ms=0, citations=[], citations_dropped=0, grounded=False)
    rerank_ms = result.trace.rerank.latency_ms if result.trace.rerank else 0
    result.trace.total_ms = result.trace.retrieval.latency_ms + rerank_ms
    return {"answer": NOT_FOUND_ANSWER, "citations": [], "grounded": False, "contradictions": [], "contradictions_total": 0}


async def _run_contradiction_branch(
    db: AsyncSession, llm: LLMClient, result: RetrievalResult
) -> tuple[list[ContradictionGroupOut], ContradictionStage]:
    # Imported here, not at module level: services/contradictions.py does
    # not import services/rag.py, but keeping this import local avoids any
    # future risk of a circular import as both modules grow.
    from app.services import contradictions as contradictions_service

    settings = get_settings()
    found, stage = await contradictions_service.detect_contradictions(
        db,
        llm,
        result.selected,
        result.query,
        sim_min=settings.contradiction_sim_min,
        sim_max=settings.contradiction_sim_max,
        max_pairs=settings.contradiction_max_pairs,
        min_confidence=settings.contradiction_min_confidence,
    )
    # Grouped here (not in chat.py) because this coroutine already owns
    # `db` for the whole branch — the answer branch never touches it, so
    # there's no concurrent-session hazard adding more queries here.
    # N documents disagreeing pairwise on one fact produces C(N,2) real,
    # distinct pairwise rows; this groups them into one displayed conflict
    # per underlying disagreement, without discarding any evidence.
    groups = await contradictions_service.build_contradiction_group_views(db, found)
    return groups, stage


async def _answer_with_contradictions(llm: LLMClient, result: RetrievalResult, db: AsyncSession) -> dict:
    """The execution model: two branches run concurrently off the same
    RetrievalResult, neither gates the other. return_exceptions=True is
    mandatory — a contradiction failure must never cancel the answer."""
    settings = get_settings()

    answer_outcome, contradiction_outcome = await asyncio.gather(
        _generate_answer(llm, result),
        _run_contradiction_branch(db, llm, result),
        return_exceptions=True,
    )

    # The answer branch has no condition and is the only failure that
    # produces an error response — let it propagate to the caller (chat.py
    # maps LLMUnavailableError to a 503).
    if isinstance(answer_outcome, BaseException):
        raise answer_outcome

    if isinstance(contradiction_outcome, BaseException):
        log.error(
            "contradiction_detection_failed",
            error=str(contradiction_outcome),
            query=result.query,
            document_ids=list({c.document_id for c in result.selected}),
        )
        result.trace.contradiction_check = ContradictionStage(
            enabled=False,
            pairs_generated=0,
            pairs_after_same_doc_filter=0,
            cached_verdicts=0,
            pairs_sent_to_cosine=0,
            pairs_after_cosine=0,
            numeric_exemptions=0,
            false_positive_suppressions=0,
            llm_calls=0,
            latency_ms=0,
            verdicts_returned=0,
            verdicts_rejected_span_check=0,
            verdicts_below_confidence=0,
            found=0,
            filter_log=[],
            error=str(contradiction_outcome),
        )
        answer_outcome["contradictions"] = []
        answer_outcome["contradictions_total"] = 0
        return answer_outcome

    groups, stage = contradiction_outcome
    result.trace.contradiction_check = stage
    # Capped by distinct GROUP count, not raw pairwise-record count — this
    # is what "showing 5 of 8" should mean: 8 real, distinct conflicts,
    # not 8 pairwise rows where several are the same conflict restated.
    answer_outcome["contradictions"] = groups[: settings.contradiction_max_response]
    answer_outcome["contradictions_total"] = len(groups)
    return answer_outcome


async def answer(
    llm: LLMClient,
    result: RetrievalResult,
    *,
    db: AsyncSession | None = None,
    detect_contradictions: bool = False,
) -> dict:
    """Single entry point. Decides whether to call the LLM at all, mutates
    result.trace in place (context/answer/contradiction_check/total_ms),
    and returns {"answer", "citations", "grounded", "contradictions",
    "contradictions_total"}.

    `db` and `detect_contradictions` are optional so existing callers
    (and the phase-two test suite) that only want an answer keep working
    unchanged — contradiction detection only forks when both are supplied.
    """
    if not result.selected:
        return _not_found_result(result)
    if db is not None and detect_contradictions:
        return await _answer_with_contradictions(llm, result, db)
    return await _generate_answer(llm, result)

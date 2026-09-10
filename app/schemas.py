"""Pydantic request/response models. API routes serialise through these —
no business logic lives here, just shape."""
from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    original_filename: str
    file_type: str
    size_bytes: int
    status: str
    page_count: int | None
    chunk_count: int | None
    chunks_done: int
    embedding_model: str | None
    effective_date: date | None
    attempts: int
    error_code: str | None
    error_message: str | None
    created_at: datetime


class DocumentUploadItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    original_filename: str
    status: str
    size_bytes: int
    duplicate: bool


class UploadResponse(BaseModel):
    documents: list[DocumentUploadItem]


class ChunkOut(BaseModel):
    chunk_id: uuid.UUID
    ordinal: int
    page_start: int | None
    page_end: int | None
    section_path: list[str]
    char_count: int
    text: str


class ChunksResponse(BaseModel):
    chunks: list[ChunkOut]
    total: int
    limit: int
    offset: int


class StatsResponse(BaseModel):
    documents_by_status: dict[str, int]
    total_chunks: int
    qdrant_point_count: int
    embedding_model: str


class ScoredChunk(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    text: str
    page_start: int | None
    page_end: int | None
    section_path: list[str]
    ordinal: int
    vector: list[float] | None = None
    vector_score: float
    rerank_score: float | None = None
    rank_before: int
    rank_after: int | None = None


class CandidateChunk(BaseModel):
    """Lightweight record for the trace — every candidate the vector search
    returned, before reranking or narrowing."""

    chunk_id: str
    document_id: str
    document_name: str
    vector_score: float
    page_start: int | None
    page_end: int | None
    section_path: list[str]


class RetrievalStage(BaseModel):
    top_k: int
    returned: int
    latency_ms: int
    candidates: list[CandidateChunk]


class RerankResultEntry(BaseModel):
    chunk_id: str
    vector_score: float
    rerank_score: float
    rank_before: int
    rank_after: int
    rank_delta: int
    used_in_answer: bool


class RerankStage(BaseModel):
    enabled: bool
    model: str | None = None
    kept: int
    latency_ms: int
    results: list[RerankResultEntry] = []
    dropped: list[str] = []
    error: str | None = None


class RetrievalStageSummary(BaseModel):
    """Trimmed for the response a frontend actually renders — counts and
    latency only. The full per-candidate list (`RetrievalStage.candidates`)
    is debug/audit detail, still persisted in full and available via
    GET /conversations/{id}/messages/{message_id}/trace."""

    top_k: int
    returned: int
    latency_ms: int


class RerankStageSummary(BaseModel):
    enabled: bool
    model: str | None = None
    kept: int
    latency_ms: int
    error: str | None = None


class ContextStage(BaseModel):
    chunks_used: list[str]
    neighbour_expansion: bool
    total_tokens: int
    truncated: bool


class AnswerStage(BaseModel):
    model: str
    latency_ms: int
    citations: list[str]
    citations_dropped: int
    grounded: bool


class RetrievalTrace(BaseModel):
    trace_id: str
    query_raw: str
    query_used_for_retrieval: str
    condensed: bool
    retrieval: RetrievalStage
    rerank: RerankStage | None = None
    context: ContextStage | None = None
    answer: AnswerStage | None = None
    contradiction_check: "ContradictionStage | None" = None
    total_ms: int | None = None


class RetrievalTraceSummary(BaseModel):
    """What GET /chat and GET /conversations/.../messages actually return
    for `trace` — every field a frontend can meaningfully show (what was
    searched, whether reranking/contradiction-checking ran, timings), none
    of the per-candidate debug arrays (`retrieval.candidates`,
    `rerank.results`/`dropped`, `contradiction_check.filter_log`). Those
    stay fully persisted on the Message row; fetch them in full via
    GET /conversations/{id}/messages/{message_id}/trace when actually
    needed (e.g. a "why this answer" debug view), instead of shipping them
    on every chat turn."""

    trace_id: str
    query_raw: str
    query_used_for_retrieval: str
    condensed: bool
    retrieval: RetrievalStageSummary
    rerank: RerankStageSummary | None = None
    context: ContextStage | None = None
    answer: AnswerStage | None = None
    contradiction_check: "ContradictionStageSummary | None" = None
    total_ms: int | None = None


def summarize_trace(trace: RetrievalTrace) -> RetrievalTraceSummary:
    """Projects a full RetrievalTrace down to RetrievalTraceSummary,
    dropping the heavy per-candidate arrays. Explicit field-by-field
    rather than relying on pydantic to silently ignore extra fields —
    easier to see exactly what's kept."""
    return RetrievalTraceSummary(
        trace_id=trace.trace_id,
        query_raw=trace.query_raw,
        query_used_for_retrieval=trace.query_used_for_retrieval,
        condensed=trace.condensed,
        retrieval=RetrievalStageSummary(
            top_k=trace.retrieval.top_k,
            returned=trace.retrieval.returned,
            latency_ms=trace.retrieval.latency_ms,
        ),
        rerank=RerankStageSummary(
            enabled=trace.rerank.enabled,
            model=trace.rerank.model,
            kept=trace.rerank.kept,
            latency_ms=trace.rerank.latency_ms,
            error=trace.rerank.error,
        )
        if trace.rerank is not None
        else None,
        context=trace.context,
        answer=trace.answer,
        contradiction_check=ContradictionStageSummary(
            enabled=trace.contradiction_check.enabled,
            pairs_generated=trace.contradiction_check.pairs_generated,
            pairs_after_same_doc_filter=trace.contradiction_check.pairs_after_same_doc_filter,
            cached_verdicts=trace.contradiction_check.cached_verdicts,
            pairs_sent_to_cosine=trace.contradiction_check.pairs_sent_to_cosine,
            pairs_after_cosine=trace.contradiction_check.pairs_after_cosine,
            numeric_exemptions=trace.contradiction_check.numeric_exemptions,
            false_positive_suppressions=trace.contradiction_check.false_positive_suppressions,
            llm_calls=trace.contradiction_check.llm_calls,
            latency_ms=trace.contradiction_check.latency_ms,
            verdicts_returned=trace.contradiction_check.verdicts_returned,
            verdicts_rejected_span_check=trace.contradiction_check.verdicts_rejected_span_check,
            verdicts_below_confidence=trace.contradiction_check.verdicts_below_confidence,
            found=trace.contradiction_check.found,
            error=trace.contradiction_check.error,
        )
        if trace.contradiction_check is not None
        else None,
        total_ms=trace.total_ms,
    )


class Citation(BaseModel):
    marker: str
    chunk_id: str
    document_id: str
    document_name: str
    page: int | None
    section: str | None
    text: str


class RetrievalResult(BaseModel):
    query: str
    candidates: list[ScoredChunk]
    selected: list[ScoredChunk]
    trace: RetrievalTrace


class ChatOptions(BaseModel):
    top_k: int | None = Field(default=None, ge=1, le=50)
    rerank: bool | None = None
    document_ids: list[str] | None = None
    detect_contradictions: bool = True


class HistoryMessage(BaseModel):
    """Conversational context for the answer LLM's prompt only — never for
    retrieval (services/retrieval.py always embeds the raw query alone) and
    never read back from the DB (the frontend sends the last 2 user turns
    it already holds in state). `role` is deliberately Literal["user"], not
    ["user", "assistant"]: assistant turns are not accepted here, and
    Pydantic rejects them with 422 rather than the service silently
    filtering them out."""

    role: Literal["user"]
    content: str


class ChatRequest(BaseModel):
    query: str
    conversation_id: uuid.UUID | None = None
    history: list[HistoryMessage] = Field(default_factory=list, max_length=2)
    options: ChatOptions | None = None


class ChatResponse(BaseModel):
    message_id: str
    conversation_id: str
    answer: str
    citations: list[Citation]
    contradictions: list["ContradictionGroupOut"] = []
    contradictions_total: int = 0
    trace: RetrievalTraceSummary
    grounded: bool


class ConversationOut(BaseModel):
    id: uuid.UUID
    title: str | None
    message_count: int
    created_at: datetime
    updated_at: datetime


class ConversationsResponse(BaseModel):
    conversations: list[ConversationOut]


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: str
    content: str
    citations: list[Citation] | None
    trace: RetrievalTraceSummary | None
    created_at: datetime


class ConversationMessagesResponse(BaseModel):
    conversation_id: uuid.UUID
    messages: list[MessageOut]


class ContradictionVerdict(BaseModel):
    pair_id: str
    is_contradiction: bool
    type: Literal["factual", "logical", "numerical", "temporal", "none"]
    severity: Literal["critical", "warning", "info"]
    confidence: float
    statement_a: str
    statement_b: str
    explanation: str
    reconciliation: Literal["supersedes", "scope_difference", "none"]


class ContradictionBatch(BaseModel):
    results: list[ContradictionVerdict]


class PairDecision(BaseModel):
    chunk_a_id: str
    chunk_b_id: str
    cosine: float | None
    accepted: bool
    reason: str


class ContradictionStage(BaseModel):
    enabled: bool
    pairs_generated: int
    pairs_after_same_doc_filter: int
    cached_verdicts: int
    pairs_sent_to_cosine: int
    pairs_after_cosine: int
    numeric_exemptions: int
    false_positive_suppressions: int
    llm_calls: int
    latency_ms: int
    verdicts_returned: int
    verdicts_rejected_span_check: int
    verdicts_below_confidence: int
    found: int
    filter_log: list[PairDecision] = []
    error: str | None = None


class ContradictionStageSummary(BaseModel):
    """Same counts as ContradictionStage, minus `filter_log` (usually one
    entry per candidate pair attempted, occasionally two — a pair whose
    cached verdict is invalidated by a CONTRADICTION_SIM_MIN/MAX change
    gets a "cache_invalid_threshold_changed" entry plus whatever
    filter_similar_pairs decides for it fresh; debug detail either way,
    not something a chat UI shows). The full log stays on the persisted
    Message row."""

    enabled: bool
    pairs_generated: int
    pairs_after_same_doc_filter: int
    cached_verdicts: int
    pairs_sent_to_cosine: int
    pairs_after_cosine: int
    numeric_exemptions: int
    false_positive_suppressions: int
    llm_calls: int
    latency_ms: int
    verdicts_returned: int
    verdicts_rejected_span_check: int
    verdicts_below_confidence: int
    found: int
    error: str | None = None


class ContradictionStatementView(BaseModel):
    text: str
    document_id: str
    document_name: str
    page: int | None
    section: str | None
    effective_date: date | None


class ContradictionOut(BaseModel):
    id: str
    type: str
    severity: str
    confidence: float
    status: str
    statement_a: ContradictionStatementView
    statement_b: ContradictionStatementView
    explanation: str
    reconciliation: str


class ContradictionEvidence(BaseModel):
    contradiction_id: str
    statement_a: ContradictionStatementView
    statement_b: ContradictionStatementView
    verified: bool


class ContradictionEvidenceItem(BaseModel):
    """One pairwise record backing a group, trimmed to what's actually new
    versus the group's own headline fields: `type`/`severity`/`status` are
    the grouping key (identical across a group's evidence by construction)
    and `explanation`/`reconciliation` repeat near-identical LLM prose per
    pair — all already shown once at the group level. Fetch a specific
    pair's full record (including those fields) via
    GET /api/contradictions/{id} using this item's `id`."""

    id: str
    confidence: float
    statement_a: ContradictionStatementView
    statement_b: ContradictionStatementView


class ContradictionGroupOut(BaseModel):
    """Multiple pairwise Contradiction rows can be evidence for the same
    underlying conflict (N documents disagreeing pairwise on one fact
    produces C(N,2) distinct, individually-correct pairs). This groups
    them for display — statement_a/statement_b are the highest-confidence
    pair's statements (the group's headline); `evidence` preserves every
    individual pairwise record so nothing is lost, just de-duplicated in
    the summary view. `group_id` is derived from current membership and
    computed fresh per request — it is not a stored, stable identifier.
    PATCH still operates on an individual evidence item's `id`, not on
    `group_id` — the underlying data model is still flat pairwise rows.
    """

    group_id: str
    type: str
    severity: str
    confidence: float
    status: str
    statement_a: ContradictionStatementView
    statement_b: ContradictionStatementView
    explanation: str
    reconciliation: str
    evidence: list[ContradictionEvidenceItem]
    evidence_count: int


class ContradictionRecordOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    fingerprint: str
    chunk_a_id: uuid.UUID
    chunk_b_id: uuid.UUID
    document_a_id: uuid.UUID
    document_b_id: uuid.UUID
    statement_a: str
    statement_b: str
    type: str
    severity: str
    confidence: float
    explanation: str
    reconciliation: str
    status: str
    resolution_note: str | None
    first_seen_query: str | None
    times_seen: int
    created_at: datetime
    updated_at: datetime
    last_seen_at: datetime
    resolved_at: datetime | None


class ContradictionListResponse(BaseModel):
    contradictions: list[ContradictionGroupOut]
    counts: dict[str, int]
    total: int
    limit: int
    offset: int


class ContradictionPatchRequest(BaseModel):
    status: Literal["open", "resolved", "false_positive"]
    note: str | None = Field(default=None, max_length=500)

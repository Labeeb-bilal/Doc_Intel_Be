"""Retrieval service — every threshold and ordering decision for turning a
query into a RetrievalResult lives here. No LLM call happens in this file.

Pipeline:
  query -> condense (if follow-up) -> embed_query() -> Qdrant search
        -> cross-encoder rerank (if enabled) -> floor -> keep top N
        -> neighbour expansion -> RetrievalResult

Order matters in three places (see inline comments at each step): floor
after rerank not before, rerank before neighbour expansion not after, and
neighbour expansion last.
"""
from __future__ import annotations

import time
import uuid

import structlog

from app.adapters import embeddings as embeddings_adapter
from app.adapters import reranker as reranker_adapter
from app.adapters import vectors as vectors_adapter
from app.config import get_settings
from app.schemas import (
    CandidateChunk,
    RerankResultEntry,
    RerankStage,
    RetrievalResult,
    RetrievalStage,
    RetrievalTrace,
    ScoredChunk,
)

log = structlog.get_logger("retrieval")


def _condense_query(query: str, previous_user_message: str | None) -> tuple[str, bool]:
    """Cheap follow-up handling: concatenate the previous user turn with the
    current one for retrieval only. The LLM prompt still gets the original
    phrasing — only the retrieval query is condensed. No LLM rewrite call
    unless follow-up quality turns out to need it."""
    if not previous_user_message:
        return query, False
    return f"{previous_user_message} {query}", True


def _point_to_scored_chunk(point, rank: int) -> ScoredChunk:
    payload = point.payload or {}
    return ScoredChunk(
        chunk_id=payload.get("chunk_id", str(point.id)),
        document_id=payload["document_id"],
        document_name=payload["document_name"],
        text=payload["text"],
        page_start=payload.get("page_start"),
        page_end=payload.get("page_end"),
        section_path=payload.get("section_path") or [],
        ordinal=payload["ordinal"],
        vector=list(point.vector) if point.vector is not None else None,
        vector_score=point.score,
        rank_before=rank,
    )


def _apply_cosine_floor(chunks: list[ScoredChunk], floor: float) -> list[ScoredChunk]:
    """Used when reranking is off, or failed. The floor here applies to raw
    cosine similarity — NOT the sigmoid-normalised rerank score. They are
    different scales; RELEVANCE_FLOOR may need separate tuning for each."""
    kept = [c for c in chunks if c.vector_score >= floor]
    for i, c in enumerate(kept):
        c.rank_after = i
    return kept


async def _expand_neighbours(chunks: list[ScoredChunk], collection: str) -> list[ScoredChunk]:
    """For each selected chunk, fetch ordinal-1 and ordinal+1 within the
    same document and fold their text in. Point IDs are deterministic, so
    the neighbour's chunk_id is computed directly — no extra search needed,
    just a retrieve-by-id against chunks we may or may not already have."""
    ids_to_fetch: set[uuid.UUID] = set()
    for chunk in chunks:
        doc_id = uuid.UUID(chunk.document_id)
        if chunk.ordinal > 0:
            ids_to_fetch.add(vectors_adapter.compute_chunk_id(doc_id, chunk.ordinal - 1))
        ids_to_fetch.add(vectors_adapter.compute_chunk_id(doc_id, chunk.ordinal + 1))

    if not ids_to_fetch:
        return chunks

    payloads = await vectors_adapter.retrieve_by_ids(collection, list(ids_to_fetch))

    expanded: list[ScoredChunk] = []
    for chunk in chunks:
        doc_id = uuid.UUID(chunk.document_id)
        prev_text = None
        if chunk.ordinal > 0:
            prev_id = str(vectors_adapter.compute_chunk_id(doc_id, chunk.ordinal - 1))
            prev_text = payloads.get(prev_id, {}).get("text")
        next_id = str(vectors_adapter.compute_chunk_id(doc_id, chunk.ordinal + 1))
        next_text = payloads.get(next_id, {}).get("text")

        pieces = [t for t in (prev_text, chunk.text, next_text) if t]
        if len(pieces) > 1:
            chunk = chunk.model_copy(update={"text": "\n\n".join(pieces)})
        expanded.append(chunk)
    return expanded


async def retrieve(
    query: str,
    *,
    previous_user_message: str | None = None,
    top_k: int | None = None,
    rerank_enabled: bool | None = None,
    document_ids: list[str] | None = None,
) -> RetrievalResult:
    settings = get_settings()
    top_k = settings.top_k if top_k is None else top_k
    rerank_enabled = settings.rerank_enabled if rerank_enabled is None else rerank_enabled

    trace_id = str(uuid.uuid4())
    query_used, condensed = _condense_query(query, previous_user_message)

    # embed_query(), never embed_documents() — BGE is asymmetric and the
    # query needs the prefix a passage must not have.
    query_vector = await embeddings_adapter.embed_query(query_used)

    doc_uuids = [uuid.UUID(d) for d in document_ids] if document_ids else None

    retrieval_start = time.perf_counter()
    points = await vectors_adapter.search_by_vector(
        settings.qdrant_collection,
        query_vector=query_vector,
        limit=top_k,
        document_ids=doc_uuids,
        with_vectors=True,  # the next phase needs chunk-to-chunk cosine for
        # contradiction pairing; re-embedding a chunk you already had is waste.
    )
    retrieval_latency_ms = round((time.perf_counter() - retrieval_start) * 1000)

    candidates = [_point_to_scored_chunk(p, rank) for rank, p in enumerate(points)]

    retrieval_stage = RetrievalStage(
        top_k=top_k,
        returned=len(candidates),
        latency_ms=retrieval_latency_ms,
        candidates=[
            CandidateChunk(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                document_name=c.document_name,
                vector_score=c.vector_score,
                page_start=c.page_start,
                page_end=c.page_end,
                section_path=c.section_path,
            )
            for c in candidates
        ],
    )

    rerank_stage: RerankStage | None = None
    ranked = candidates

    if rerank_enabled and candidates:
        rerank_start = time.perf_counter()
        try:
            logits = await reranker_adapter.rerank(query_used, [c.text for c in candidates])
            for chunk, logit in zip(candidates, logits):
                chunk.rerank_score = reranker_adapter.sigmoid(logit)

            # Rerank before neighbour expansion, never after: cross-encoders
            # cap at ~512 tokens for the (query, chunk) pair. An already
            # neighbour-expanded chunk would silently get truncated and
            # scored as a fragment instead of the real chunk.
            ranked = sorted(candidates, key=lambda c: c.rerank_score, reverse=True)
            for rank_after, chunk in enumerate(ranked):
                chunk.rank_after = rank_after
            rerank_latency_ms = round((time.perf_counter() - rerank_start) * 1000)

            # Floor after reranking, not before: cosine drift with query
            # phrasing makes a pre-rerank cutoff meaningless. This floor
            # applies to the sigmoid-normalised rerank score.
            floor = settings.relevance_floor
            kept = [c for c in ranked if c.rerank_score >= floor]
            kept_ids = {c.chunk_id for c in kept[: settings.keep_n]}

            rerank_stage = RerankStage(
                enabled=True,
                model=settings.rerank_model,
                kept=len(kept),
                latency_ms=rerank_latency_ms,
                results=[
                    RerankResultEntry(
                        chunk_id=c.chunk_id,
                        vector_score=c.vector_score,
                        rerank_score=c.rerank_score,
                        rank_before=c.rank_before,
                        rank_after=c.rank_after,
                        rank_delta=c.rank_before - c.rank_after,
                        used_in_answer=c.chunk_id in kept_ids,
                    )
                    for c in ranked
                ],
                dropped=[c.chunk_id for c in ranked if c.rerank_score < floor],
            )
            ranked = kept
        except Exception as exc:
            # Degradation ladder, step 1: reranker fails -> fall back to
            # vector ordering, note it in the trace, keep going.
            log.warning("rerank_failed", error=str(exc))
            rerank_stage = RerankStage(enabled=False, kept=0, latency_ms=0, error=str(exc))
            ranked = _apply_cosine_floor(candidates, settings.relevance_floor)
    elif candidates:
        ranked = _apply_cosine_floor(candidates, settings.relevance_floor)

    selected = ranked[: settings.keep_n]

    if settings.neighbour_expansion and selected:
        try:
            selected = await _expand_neighbours(selected, settings.qdrant_collection)
        except Exception as exc:
            # Degradation ladder, step 2: use the chunks unexpanded, continue.
            log.warning("neighbour_expansion_failed", error=str(exc))

    trace = RetrievalTrace(
        trace_id=trace_id,
        query_raw=query,
        query_used_for_retrieval=query_used,
        condensed=condensed,
        retrieval=retrieval_stage,
        rerank=rerank_stage,
    )

    return RetrievalResult(query=query, candidates=candidates, selected=selected, trace=trace)

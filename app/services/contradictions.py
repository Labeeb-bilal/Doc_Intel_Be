"""Contradiction detection service — the second half of the RAG pipeline
that runs alongside answer generation, on the same retrieved chunks.

This is a separate concern from answer generation on purpose: the answer
LLM's job is "write a grounded answer to the user's question"; this
module's job is "do any two of these chunks disagree with each other?".
They share input (the same selected chunks) but never share code.

Pipeline (`detect_contradictions()` runs every step below, in order):

  selected chunks
      |
      v
  1. generate_candidate_pairs()  -- every unordered pair, same-document
      |                             pairs dropped immediately
      v
  2. gather_candidate_pairs()    -- fingerprint each pair, reuse cached
      |                             verdicts (if still within today's
      |                             CONTRADICTION_SIM_MIN/MAX -- see
      |                             cache_still_valid), suppress
      |                             false_positives
      v
  3. filter_similar_pairs()      -- cosine similarity bounds: is this pair
      |                             even worth asking the LLM about?
      v
  4. adjudicate_pairs()          -- ONE batched call to the contradiction
      |                             LLM for every pair that survived step 3
      v
  5. verify_contradiction_evidence() -- reject any verdict whose quoted
      |                                 evidence isn't real chunk text
      v
  6. confidence threshold check  -- reject verdicts below min_confidence
      |
      v
  7. upsert_contradiction()      -- persist genuine, verified verdicts
      |
      v
  found contradictions, sorted by severity then confidence

`detect_contradictions()` is the single entry point services/rag.py calls,
inside asyncio.gather(..., return_exceptions=True) alongside the answer
generation branch — this module never wraps its own errors, so a failure
here propagates naturally for the caller to catch (see rag.py: a failure
here must never take down the user-facing answer).
"""
from __future__ import annotations

import hashlib
import itertools
import math
import re
import time
import uuid
from datetime import date, datetime, timezone

import structlog
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.llm import LLMClient, LLMUnavailableError
from app.errors import AppError, NotFoundError
from app.models import Chunk, Contradiction, Document
from app.prompts.contradiction import (
    CONTRADICTION_SYSTEM_PROMPT,
    build_user_message,
    format_cached_context_line,
    format_chunk_block,
    format_pair_block,
)
from app.schemas import (
    ContradictionBatch,
    ContradictionEvidenceItem,
    ContradictionGroupOut,
    ContradictionOut,
    ContradictionStage,
    ContradictionStatementView,
    ContradictionVerdict,
    PairDecision,
    ScoredChunk,
)

log = structlog.get_logger("contradictions")

ChunkPair = tuple[ScoredChunk, ScoredChunk]

_NUMERIC_TOKEN_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")


def fingerprint(chunk_a_id: str, chunk_b_id: str) -> str:
    """(A, B) and (B, A) are the same conflict — sort before hashing so
    both directions produce the identical key, regardless of which chunk
    happened to be selected first."""
    a, b = sorted([chunk_a_id, chunk_b_id])
    return hashlib.sha256(f"{a}|{b}".encode()).hexdigest()


def _decision(
    a: ScoredChunk, b: ScoredChunk, *, cosine: float | None, accepted: bool, reason: str
) -> PairDecision:
    """Build one filter_log entry. Centralizing this avoids re-typing the
    same four-field PairDecision(...) call at every filter stage below."""
    return PairDecision(chunk_a_id=a.chunk_id, chunk_b_id=b.chunk_id, cosine=cosine, accepted=accepted, reason=reason)


def generate_candidate_pairs(chunks: list[ScoredChunk]) -> tuple[list[ChunkPair], list[PairDecision]]:
    """Step 1 of the pipeline: every unordered pair of selected chunks,
    with same-document pairs dropped immediately.

    Same-document pairs are dropped here — first, before fingerprinting or
    any similarity/LLM work — because it's a free ID comparison and
    because within one document, similar-sounding chunks are elaboration
    of the same point, not conflicting claims: two chunks from the same
    source were written by the same author to agree with each other.
    """
    survivors: list[ChunkPair] = []
    log_entries: list[PairDecision] = []

    for a, b in itertools.combinations(chunks, 2):
        if a.document_id == b.document_id:
            log_entries.append(_decision(a, b, cosine=None, accepted=False, reason="same_document"))
            continue
        survivors.append((a, b))

    return survivors, log_entries


async def fetch_cached_verdicts(db: AsyncSession, fingerprints: list[str]) -> dict[str, Contradiction]:
    """The cache key is "chunk_a + chunk_b" (see fingerprint()), not the
    user's question. It answers "have we already judged whether THESE TWO
    PIECES OF EVIDENCE conflict?" — not "have we already answered this
    query?". The same chunk pair can turn up again for a completely
    different question and still reuse the same verdict.

    One DB query for every fingerprint in this batch — not one query per
    pair. Cache is ground truth: a pair a user has already marked
    false_positive (or resolved) must never be re-adjudicated by the
    similarity filter or the LLM, so this lookup happens before either
    runs."""
    if not fingerprints:
        return {}
    stmt = select(Contradiction).where(Contradiction.fingerprint.in_(fingerprints))
    rows = (await db.execute(stmt)).scalars().all()
    return {row.fingerprint: row for row in rows}


async def gather_candidate_pairs(
    db: AsyncSession, chunks: list[ScoredChunk], *, sim_min: float, sim_max: float
) -> tuple[list[ChunkPair], dict[str, Contradiction], list[PairDecision]]:
    """Step 2: same-doc filter (step 1) -> fingerprint each remaining pair
    -> look up the cache -> suppress false_positive pairs -> validate
    whatever's left against today's similarity thresholds. Returns the
    pairs that still need similarity filtering (next stage — this now
    includes cache hits invalidated by a threshold change, not just
    never-seen pairs), the *reusable* cached verdicts for this batch keyed
    by fingerprint (open/resolved, and still within today's thresholds
    only), and the filter log entries generated so far.

    false_positive pairs are deliberately excluded from both return
    values: they are in the cache (so `fetch_cached_verdicts` finds them),
    but a user's false_positive call must suppress the pair everywhere —
    it must not be reused as a finding, and it must not fall through to
    similarity/LLM either. This is the suppression check the spec places
    between the cache check and the similarity check. Note this check
    happens BEFORE the threshold-validity check below: suppression is a
    human decision and stays permanent regardless of what
    CONTRADICTION_SIM_MIN/MAX happen to be today.

    `sim_min`/`sim_max` are only used here to validate a cache HIT (see
    cache_still_valid) — they don't change which pairs exist, only whether
    an already-cached verdict is still trusted or gets treated as a miss.
    """
    pairs, log_entries = generate_candidate_pairs(chunks)

    fp_to_pair: dict[str, ChunkPair] = {fingerprint(a.chunk_id, b.chunk_id): (a, b) for a, b in pairs}
    cached = await fetch_cached_verdicts(db, list(fp_to_pair.keys()))

    reusable_cached: dict[str, Contradiction] = {}
    uncached_pairs: list[ChunkPair] = []
    for fp, (a, b) in fp_to_pair.items():
        cached_row = cached.get(fp)
        if cached_row is None:
            uncached_pairs.append((a, b))
        elif cached_row.status == "false_positive":
            log_entries.append(_decision(a, b, cosine=None, accepted=False, reason="false_positive_suppressed"))
        elif not cache_still_valid(cached_row.cosine, a, b, sim_min=sim_min, sim_max=sim_max):
            uncached_pairs.append((a, b))
            log_entries.append(
                _decision(a, b, cosine=cached_row.cosine, accepted=False, reason="cache_invalid_threshold_changed")
            )
        else:
            reusable_cached[fp] = cached_row
            log_entries.append(_decision(a, b, cosine=None, accepted=True, reason="cached_verdict"))

    return uncached_pairs, reusable_cached, log_entries


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Plain-Python dot-product cosine — no numpy needed for 384-dim
    vectors we already have in memory (with_vectors=True on the retrieval
    search), and this avoids adding a dependency for one small function."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def numeric_tokens(text: str) -> set[str]:
    return set(_NUMERIC_TOKEN_RE.findall(text))


def passes_upper_bound(pair_similarity: float, max_similarity: float, a: str, b: str) -> bool:
    """Near-identical text (pair_similarity > max_similarity) is boilerplate
    copied between documents, not a conflict — this is the highest-yield
    false-positive filter; without it, shared standard clauses get flagged
    constantly.

    The one exception: near-identical wording with different numbers is
    the classic numerical contradiction ("$50 per transaction" vs "$75 per
    transaction" reads as near-duplicate text). If the numeric tokens in
    the two chunks differ, let the pair through regardless of how high the
    similarity is — the number is exactly what might be in conflict.
    """
    if pair_similarity > max_similarity:
        return numeric_tokens(a) != numeric_tokens(b)
    return True


def cache_still_valid(stored_cosine: float | None, a: ScoredChunk, b: ScoredChunk, *, sim_min: float, sim_max: float) -> bool:
    """A cached verdict is only trustworthy if the cosine it was stored
    with would STILL clear today's CONTRADICTION_SIM_MIN/MAX — if those
    thresholds changed since the pair was first adjudicated, the cached
    pair might now fall outside them, and cache-before-cosine would
    otherwise keep surfacing that stale verdict regardless of the current
    configuration.

    None means the row predates this column (or is some other cache entry
    that never computed one) — always a miss, never treated as "valid
    because unknown".

    Deliberately mirrors filter_similar_pairs' own bound check exactly,
    numeric exemption included (via passes_upper_bound), rather than a
    bare `sim_min <= stored_cosine <= sim_max`. Without the exemption, a
    legitimately-cached numeric-exempt pair — one whose cosine is, BY
    DESIGN, above sim_max — would be wrongly invalidated on every single
    lookup even with unchanged thresholds, forcing a needless LLM call
    every time and defeating caching for that entire class of pairs.
    """
    if stored_cosine is None:
        return False
    if stored_cosine < sim_min:
        return False
    if stored_cosine > sim_max:
        return passes_upper_bound(stored_cosine, sim_max, a.text, b.text)
    return True


def filter_similar_pairs(
    pairs: list[ChunkPair], *, sim_min: float, sim_max: float, max_pairs: int
) -> tuple[list[ChunkPair], list[PairDecision], int]:
    """Step 3 of the pipeline. IMPORTANT: this function does not decide
    whether two chunks contradict each other — only the contradiction LLM
    (adjudicate_pairs, next stage) makes that call. This step only answers
    a cheaper, upstream question: "are these two chunks talking about the
    same thing closely enough that it's even worth asking the LLM?" —
    using cosine similarity between their embedding vectors (already on
    ScoredChunk from retrieval, no re-embedding needed) as that proxy.

    Two similarity bounds, plus a hard cap on how many pairs go to the LLM:

    - sim_min (lower bound): below this, the chunks are about different
      topics entirely and cannot contradict — there's no shared claim to
      conflict over.
    - sim_max (upper bound): above this, the wording is near-identical —
      almost certainly boilerplate copied between documents, not a real
      conflict. See passes_upper_bound() for the one exception (numbers
      differ) that lets a near-identical pair through anyway.
    - max_pairs (cap): bounds worst-case LLM token cost and judgment
      quality — a model adjudicates 8 pairs more reliably than 20.
    """
    scored: list[tuple[float, ChunkPair]] = []
    log_entries: list[PairDecision] = []
    numeric_exempt_ids: set[tuple[str, str]] = set()
    numeric_exemptions = 0

    for a, b in pairs:
        if a.vector is None or b.vector is None:
            log_entries.append(_decision(a, b, cosine=None, accepted=False, reason="no_vector"))
            continue

        pair_similarity = cosine_similarity(a.vector, b.vector)

        if pair_similarity < sim_min:
            log_entries.append(
                _decision(a, b, cosine=pair_similarity, accepted=False, reason="similarity_below_threshold")
            )
            continue

        if pair_similarity > sim_max:
            if not passes_upper_bound(pair_similarity, sim_max, a.text, b.text):
                log_entries.append(_decision(a, b, cosine=pair_similarity, accepted=False, reason="near_duplicate"))
                continue
            numeric_exemptions += 1
            numeric_exempt_ids.add((a.chunk_id, b.chunk_id))
            log_entries.append(_decision(a, b, cosine=pair_similarity, accepted=True, reason="numeric_exempt"))
            scored.append((pair_similarity, (a, b)))
            continue

        scored.append((pair_similarity, (a, b)))

    scored.sort(key=lambda item: item[0], reverse=True)
    kept = scored[:max_pairs]
    overflow = scored[max_pairs:]

    for pair_similarity, (a, b) in kept:
        if (a.chunk_id, b.chunk_id) not in numeric_exempt_ids:
            log_entries.append(_decision(a, b, cosine=pair_similarity, accepted=True, reason="accepted"))

    for pair_similarity, (a, b) in overflow:
        log_entries.append(_decision(a, b, cosine=pair_similarity, accepted=False, reason="over_pair_limit"))

    return [pair for _, pair in kept], log_entries, numeric_exemptions


async def fetch_effective_dates(db: AsyncSession, document_ids: set[str]) -> dict[str, date | None]:
    """Effective dates get passed into the LLM prompt alongside each chunk
    (see _chunk_block below) so the model can tell "policy A, dated 2023"
    from "policy B, dated 2024" and decide the newer one supersedes the
    older one (reconciliation="supersedes") when both describe the same
    rule. There is no separate document-authority ranking in this codebase
    — the LLM makes that call itself, from the dates and text it's shown;
    see CONTRADICTION_SYSTEM_PROMPT's TEMPORAL HANDLING section."""
    if not document_ids:
        return {}
    stmt = select(Document.id, Document.effective_date).where(
        Document.id.in_([uuid.UUID(d) for d in document_ids])
    )
    rows = (await db.execute(stmt)).all()
    return {str(row.id): row.effective_date for row in rows}


def _chunk_block(chunk: ScoredChunk, effective_dates: dict[str, date | None]) -> str:
    eff = effective_dates.get(chunk.document_id)
    return format_chunk_block(
        document_name=chunk.document_name,
        effective_date=eff.isoformat() if eff else None,
        section=" > ".join(chunk.section_path) if chunk.section_path else None,
        text=chunk.text,
    )


async def adjudicate_pairs(
    llm: LLMClient,
    pairs: list[ChunkPair],
    cached_context: list[Contradiction],
    effective_dates: dict[str, date | None],
) -> tuple[ContradictionBatch, dict[str, ChunkPair]]:
    """One batched complete_structured() call for every surviving pair —
    not one call per pair. One round trip, one rate-limit hit, and
    consistent judgment across pairs (the model can see cross-pair
    patterns via the cached-context block)."""
    pair_id_map: dict[str, ChunkPair] = {}
    pair_blocks: list[str] = []
    for i, (a, b) in enumerate(pairs, start=1):
        pair_id = f"P{i}"
        pair_id_map[pair_id] = (a, b)
        pair_blocks.append(
            format_pair_block(pair_id, _chunk_block(a, effective_dates), _chunk_block(b, effective_dates))
        )

    cached_lines = [
        format_cached_context_line(
            cx_type=c.type,
            statement_a=c.statement_a,
            statement_b=c.statement_b,
            document_a=str(c.document_a_id),
            document_b=str(c.document_b_id),
        )
        for c in cached_context
    ]

    user_message = build_user_message(pair_blocks, cached_lines)
    batch = await llm.complete_structured(system=CONTRADICTION_SYSTEM_PROMPT, user=user_message, schema=ContradictionBatch)
    return batch, pair_id_map


def verify_contradiction_evidence(verdict: ContradictionVerdict, a: ScoredChunk, b: ScoredChunk) -> bool:
    """Step 5: the contradiction LLM can fabricate or paraphrase the quotes
    it claims to be citing, so before trusting a verdict we check that its
    quoted statement_a/statement_b are literal substrings of the actual
    source chunks — not just plausible-sounding text the model invented.

    Honest scope: chunk text here is the exact same payload the model was
    shown, so this catches hallucinated or paraphrased quotes — it is a
    hallucination/paraphrase guard, not independent verification that the
    contradiction itself is real. The model could still misjudge two
    genuinely-quoted, non-conflicting statements as conflicting; this check
    only proves the quotes are real, not that the judgment is correct.

    Two normalizations, tried in order:
    1. Whitespace-collapsed (runs of whitespace -> one space) — handles
       ordinary line wraps.
    2. Whitespace-stripped entirely — handles a PDF hard-wrapping a word
       across two lines with no hyphen (pdfplumber's own line breaking,
       not this app's chunking), e.g. "...once th\neyhave..." extracted
       as "th ey" for "they". The model naturally quotes the clean word;
       only the *source* has the stray space. Still exact-substring on
       every character, just insensitive to whitespace *position* — a
       quote with different words or content still fails both passes.
    """

    def norm(s: str) -> str:
        return " ".join(s.lower().split())

    def norm_no_space(s: str) -> str:
        return "".join(s.lower().split())

    def contains(needle: str, haystack: str) -> bool:
        return norm(needle) in norm(haystack) or norm_no_space(needle) in norm_no_space(haystack)

    return contains(verdict.statement_a, a.text) and contains(verdict.statement_b, b.text)


async def upsert_contradiction(
    db: AsyncSession, verdict: ContradictionVerdict, a: ScoredChunk, b: ScoredChunk, query: str, cosine: float
) -> Contradiction:
    """Only genuine (is_contradiction=true, verified, confident) verdicts
    are ever stored — the `contradictions` table's name and status
    semantics (open/resolved/false_positive) only make sense for real
    positives. A non-contradiction verdict is simply not cached; that pair
    may be re-judged on a future query, which is an acceptable tradeoff
    against inventing a second "confirmed-not-a-conflict" cache the given
    schema has no room for.

    `cosine` is today's freshly-computed similarity for this pair (see
    detect_contradictions) — stored so a future cache lookup can validate
    the verdict against CONTRADICTION_SIM_MIN/MAX at that time (see
    cache_still_valid). On rediscovery this always gets refreshed, even
    for a pair reaching this function again after its old cached cosine
    was invalidated by a threshold change: the row's `cosine` should
    reflect the value that just got it here, not a stale one.

    On rediscovery: increment times_seen, bump last_seen_at, refresh the
    stored cosine, and — this is the important part — leave `status`
    alone once it's resolved or false_positive. A contradiction a user
    already resolved or marked false_positive must stay that way no
    matter how many more times the same pair turns up (whether that's an
    ordinary re-sighting from cache, or a fresh re-adjudication after its
    cache entry was invalidated by a threshold change). An "open" record
    is simply re-confirmed as open — not a behavior change, just explicit
    about the one status this function is actually allowed to (re)write.
    """
    fp = fingerprint(a.chunk_id, b.chunk_id)
    existing = await db.scalar(select(Contradiction).where(Contradiction.fingerprint == fp))
    now = datetime.now(timezone.utc)

    if existing is not None:
        existing.times_seen += 1
        existing.last_seen_at = now
        existing.cosine = cosine
        if existing.status not in ("resolved", "false_positive"):
            existing.status = "open"
        await db.commit()
        await db.refresh(existing)
        return existing

    record = Contradiction(
        fingerprint=fp,
        chunk_a_id=uuid.UUID(a.chunk_id),
        chunk_b_id=uuid.UUID(b.chunk_id),
        document_a_id=uuid.UUID(a.document_id),
        document_b_id=uuid.UUID(b.document_id),
        cosine=cosine,
        statement_a=verdict.statement_a,
        statement_b=verdict.statement_b,
        type=verdict.type,
        severity=verdict.severity,
        confidence=verdict.confidence,
        explanation=verdict.explanation,
        reconciliation=verdict.reconciliation,
        status="open",
        first_seen_query=query,
        times_seen=1,
        last_seen_at=now,
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)
    return record


async def _touch_cached(db: AsyncSession, records: list[Contradiction]) -> None:
    """Reusing a cached verdict still counts as a sighting — bump
    times_seen and last_seen_at, same as a freshly-adjudicated one, but
    skip the LLM call entirely."""
    if not records:
        return
    now = datetime.now(timezone.utc)
    for record in records:
        record.times_seen += 1
        record.last_seen_at = now
    await db.commit()


_SEVERITY_RANK = {"critical": 0, "warning": 1, "info": 2}


async def detect_contradictions(
    db: AsyncSession,
    llm: LLMClient,
    chunks: list[ScoredChunk],
    query: str,
    *,
    sim_min: float,
    sim_max: float,
    max_pairs: int,
    min_confidence: float,
) -> tuple[list[Contradiction], ContradictionStage]:
    """The full pipeline: same-doc filter -> cache -> cosine -> batched
    LLM -> span/confidence verification -> upsert -> sort. Returns the
    complete sorted set (severity, then confidence) — capping the response
    to CONTRADICTION_MAX_RESPONSE is the caller's job, not this function's,
    since GET /api/contradictions needs the full set regardless.

    Two distinct failure tiers, deliberately handled differently:

    - The LLM judgment call itself fails (LLMUnavailableError — rate
      limited, auth, timeout, ...): caught HERE, not left to propagate.
      Pair formation, the cache check, and cosine filtering already
      completed successfully by this point, so their counts are real and
      worth keeping — losing them would make a working pipeline that hit
      a transient LLM outage look identical to a pipeline that never ran
      at all (pairs_generated=0), which is misleading. `stage.error`
      names whether it was specifically a rate limit or something else,
      and any already-cached contradictions are still returned.
    - Anything else (a bug, a DB error during pair/cache setup) is left to
      propagate naturally — genuinely nothing was computed, so there is
      nothing partial to preserve, and the caller's
      asyncio.gather(..., return_exceptions=True) handles it.
    """
    start = time.perf_counter()

    total_pairs = len(chunks) * (len(chunks) - 1) // 2
    uncached_pairs, reusable_cached, log_entries = await gather_candidate_pairs(db, chunks, sim_min=sim_min, sim_max=sim_max)

    same_doc_count = sum(1 for e in log_entries if e.reason == "same_document")
    fp_suppressed = sum(1 for e in log_entries if e.reason == "false_positive_suppressed")
    pairs_after_same_doc_filter = total_pairs - same_doc_count

    kept_pairs, similarity_log, numeric_exemptions = filter_similar_pairs(
        uncached_pairs, sim_min=sim_min, sim_max=sim_max, max_pairs=max_pairs
    )
    log_entries.extend(similarity_log)

    llm_calls = 0
    verdicts_returned = 0
    verdicts_rejected_span_check = 0
    verdicts_below_confidence = 0
    new_contradictions: list[Contradiction] = []
    llm_error: str | None = None

    if kept_pairs:
        involved_document_ids = {c.document_id for pair in kept_pairs for c in pair}
        effective_dates = await fetch_effective_dates(db, involved_document_ids)

        try:
            batch, pair_id_map = await adjudicate_pairs(
                llm, kept_pairs, list(reusable_cached.values()), effective_dates
            )
        except LLMUnavailableError as exc:
            llm_calls = 1
            kind = "rate limited" if exc.rate_limited else "unavailable"
            llm_error = f"LLM {kind}: {exc.message}"
            log.warning(
                "contradiction_llm_call_failed",
                query=query,
                document_ids=sorted(involved_document_ids),
                rate_limited=exc.rate_limited,
                error=exc.message,
            )
        else:
            llm_calls = 1
            verdicts_returned = len(batch.results)

            for verdict in batch.results:
                pair = pair_id_map.get(verdict.pair_id)
                if pair is None or not verdict.is_contradiction:
                    continue
                a, b = pair
                if not verify_contradiction_evidence(verdict, a, b):
                    verdicts_rejected_span_check += 1
                    continue
                if verdict.confidence < min_confidence:
                    verdicts_below_confidence += 1
                    continue
                pair_cosine = cosine_similarity(a.vector, b.vector)
                record = await upsert_contradiction(db, verdict, a, b, query, pair_cosine)
                new_contradictions.append(record)

    await _touch_cached(db, list(reusable_cached.values()))

    found = list(reusable_cached.values()) + new_contradictions
    found.sort(key=lambda c: (_SEVERITY_RANK.get(c.severity, 3), -c.confidence))

    stage = ContradictionStage(
        enabled=True,
        pairs_generated=total_pairs,
        pairs_after_same_doc_filter=pairs_after_same_doc_filter,
        cached_verdicts=len(reusable_cached),
        pairs_sent_to_cosine=len(uncached_pairs),
        pairs_after_cosine=len(kept_pairs),
        numeric_exemptions=numeric_exemptions,
        false_positive_suppressions=fp_suppressed,
        llm_calls=llm_calls,
        latency_ms=round((time.perf_counter() - start) * 1000),
        verdicts_returned=verdicts_returned,
        verdicts_rejected_span_check=verdicts_rejected_span_check,
        verdicts_below_confidence=verdicts_below_confidence,
        found=len(found),
        filter_log=log_entries,
        error=llm_error,
    )

    return found, stage


async def build_contradiction_views(db: AsyncSession, records: list[Contradiction]) -> dict[str, ContradictionOut]:
    """Batched conversion from stored Contradiction rows to the API shape,
    joining back to chunks/documents for page/section/effective_date. This
    works identically for a live chat turn and a bare GET
    /api/contradictions listing where no in-memory ScoredChunk exists —
    the stored fingerprint/chunk ids are enough on their own."""
    if not records:
        return {}

    chunk_ids = {r.chunk_a_id for r in records} | {r.chunk_b_id for r in records}
    document_ids = {r.document_a_id for r in records} | {r.document_b_id for r in records}

    chunk_rows = (await db.execute(select(Chunk).where(Chunk.id.in_(chunk_ids)))).scalars().all()
    chunks_by_id = {c.id: c for c in chunk_rows}

    doc_rows = (await db.execute(select(Document).where(Document.id.in_(document_ids)))).scalars().all()
    docs_by_id = {d.id: d for d in doc_rows}

    def _view(chunk_id: uuid.UUID, document_id: uuid.UUID, text: str) -> ContradictionStatementView:
        chunk = chunks_by_id.get(chunk_id)
        doc = docs_by_id.get(document_id)
        return ContradictionStatementView(
            text=text,
            document_id=str(document_id),
            document_name=doc.original_filename if doc else "",
            page=chunk.page_start if chunk else None,
            section=" > ".join(chunk.section_path) if chunk and chunk.section_path else None,
            effective_date=doc.effective_date if doc else None,
        )

    return {
        str(r.id): ContradictionOut(
            id=str(r.id),
            type=r.type,
            severity=r.severity,
            confidence=r.confidence,
            status=r.status,
            statement_a=_view(r.chunk_a_id, r.document_a_id, r.statement_a),
            statement_b=_view(r.chunk_b_id, r.document_b_id, r.statement_b),
            explanation=r.explanation,
            reconciliation=r.reconciliation,
        )
        for r in records
    }


_VALID_STATUSES = ("open", "resolved", "false_positive")


async def _fetch_all_matching_contradictions(
    db: AsyncSession,
    *,
    status: str | None,
    severity: str | None,
    type_: str | None,
    document_id: str | None,
) -> tuple[list[Contradiction], dict[str, int]]:
    """All rows matching the filters, unpaginated — grouping has to see
    every matching row before pagination is applied, or a group could get
    split across two pages. `counts` reflects global status tallies
    (open/resolved/false_positive) unaffected by the status filter itself
    — the other filters (severity, type, document_id) DO narrow it, so a
    dashboard viewing one document or one contradiction type sees counts
    scoped to that view."""
    filters = []
    if severity:
        filters.append(Contradiction.severity == severity)
    if type_:
        filters.append(Contradiction.type == type_)
    if document_id:
        doc_uuid = uuid.UUID(document_id)
        filters.append(or_(Contradiction.document_a_id == doc_uuid, Contradiction.document_b_id == doc_uuid))

    list_filters = [*filters, Contradiction.status == status] if status else filters

    stmt = select(Contradiction).where(*list_filters).order_by(Contradiction.created_at.desc())
    records = list((await db.execute(stmt)).scalars().all())

    counts_stmt = select(Contradiction.status, func.count()).where(*filters).group_by(Contradiction.status)
    status_rows = (await db.execute(counts_stmt)).all()
    counts = {"open": 0, "resolved": 0, "false_positive": 0}
    counts.update({row[0]: row[1] for row in status_rows})

    return records, counts


async def list_contradiction_groups(
    db: AsyncSession,
    *,
    status: str | None = "open",
    severity: str | None = None,
    type_: str | None = None,
    document_id: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> tuple[list[ContradictionGroupOut], dict[str, int], int]:
    """Fetch every matching pairwise row, group them (same underlying
    conflict shown once, all evidence preserved inside each group), THEN
    paginate — `limit`/`offset` apply to distinct groups, not raw rows, so
    a group is never split across two pages."""
    records, counts = await _fetch_all_matching_contradictions(
        db, status=status, severity=severity, type_=type_, document_id=document_id
    )
    groups = await build_contradiction_group_views(db, records)
    total = len(groups)
    page = groups[offset : offset + limit]
    return page, counts, total


async def get_contradiction(db: AsyncSession, contradiction_id: uuid.UUID) -> Contradiction:
    record = await db.get(Contradiction, contradiction_id)
    if record is None:
        raise NotFoundError(f"Contradiction {contradiction_id} not found.")
    return record


async def update_contradiction_status(
    db: AsyncSession, contradiction_id: uuid.UUID, status: str, note: str | None
) -> Contradiction:
    if status not in _VALID_STATUSES:
        raise AppError(
            f"status must be one of {_VALID_STATUSES}.", code="INVALID_STATUS", status_code=422
        )
    record = await get_contradiction(db, contradiction_id)
    record.status = status
    record.resolution_note = note
    if status == "resolved":
        record.resolved_at = datetime.now(timezone.utc)
    elif status == "open":
        record.resolved_at = None
    await db.commit()
    await db.refresh(record)
    return record


def group_contradictions(records: list[Contradiction]) -> list[list[Contradiction]]:
    """Union-Find over chunk_a_id/chunk_b_id: two records that share at
    least one chunk AND have the same `type` are treated as evidence for
    one conflict.

    Grouping is deliberately scoped to matching `type`: a single chunk can
    legitimately touch more than one topic (short documents, and
    especially neighbour-expanded chunks from phase two — see README),
    so "shares a chunk" alone is not enough signal on its own. Requiring
    the same type is a cheap, no-extra-LLM-call guard against merging two
    otherwise-unrelated conflicts just because they happen to share a
    multi-topic chunk.
    """
    if not records:
        return []

    parent: dict[uuid.UUID, uuid.UUID] = {r.id: r.id for r in records}

    def find(x: uuid.UUID) -> uuid.UUID:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: uuid.UUID, b: uuid.UUID) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_chunk_and_type: dict[tuple[uuid.UUID, str], list[Contradiction]] = {}
    for r in records:
        for chunk_id in (r.chunk_a_id, r.chunk_b_id):
            by_chunk_and_type.setdefault((chunk_id, r.type), []).append(r)

    for same_key_records in by_chunk_and_type.values():
        first = same_key_records[0]
        for other in same_key_records[1:]:
            union(first.id, other.id)

    clusters: dict[uuid.UUID, list[Contradiction]] = {}
    for r in records:
        clusters.setdefault(find(r.id), []).append(r)

    return list(clusters.values())


async def build_contradiction_group_views(db: AsyncSession, records: list[Contradiction]) -> list[ContradictionGroupOut]:
    """Group, then build the display view for each group. The
    highest-confidence member is the group's headline (statement_a/b,
    type, reconciliation); every member — including the headline one —
    appears again in `evidence`, so no supporting detail is lost."""
    if not records:
        return []

    clusters = group_contradictions(records)
    views_by_id = await build_contradiction_views(db, records)

    groups: list[ContradictionGroupOut] = []
    for cluster in clusters:
        cluster_sorted = sorted(cluster, key=lambda c: (-c.confidence, c.created_at))
        primary = cluster_sorted[0]
        primary_view = views_by_id[str(primary.id)]

        statuses = {c.status for c in cluster}
        group_status = "open" if "open" in statuses else primary.status

        most_severe = min(cluster, key=lambda c: _SEVERITY_RANK.get(c.severity, 3))

        group_id = hashlib.sha256(",".join(sorted(str(c.id) for c in cluster)).encode()).hexdigest()[:16]

        evidence_items = [
            ContradictionEvidenceItem(
                id=views_by_id[str(c.id)].id,
                confidence=views_by_id[str(c.id)].confidence,
                statement_a=views_by_id[str(c.id)].statement_a,
                statement_b=views_by_id[str(c.id)].statement_b,
            )
            for c in cluster_sorted
        ]

        groups.append(
            ContradictionGroupOut(
                group_id=group_id,
                type=primary.type,
                severity=most_severe.severity,
                confidence=primary.confidence,
                status=group_status,
                statement_a=primary_view.statement_a,
                statement_b=primary_view.statement_b,
                explanation=primary.explanation,
                reconciliation=primary.reconciliation,
                evidence=evidence_items,
                evidence_count=len(cluster),
            )
        )

    groups.sort(key=lambda g: (_SEVERITY_RANK.get(g.severity, 3), -g.confidence))
    return groups

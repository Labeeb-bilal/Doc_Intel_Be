"""Stage 4 checkpoint — querying "what is the remote work policy?" should
return the planted contradiction with correct type, severity, and
verified verbatim quotes. Real LLM, real documents, no concurrency wiring
yet (that's stage 5).

Usage:
    python scripts/check_contradiction_llm.py "your query here"
"""
from __future__ import annotations

import asyncio
import sys

from app.adapters import embeddings, reranker
from app.adapters.llm import get_llm_client
from app.config import get_settings
from app.db import async_session_factory
from app.logging import configure_logging
from app.services.contradictions import (
    adjudicate_pairs,
    fetch_effective_dates,
    filter_by_cosine,
    gather_candidate_pairs,
    spans_present,
    upsert_contradiction,
)
from app.services.retrieval import retrieve


async def main() -> None:
    configure_logging()
    query = sys.argv[1] if len(sys.argv) > 1 else "What is the remote work policy?"
    settings = get_settings()
    embeddings.load_model()
    reranker.load_model()
    llm = get_llm_client()

    result = await retrieve(query, top_k=20)
    print(f"QUERY: {query!r}  selected: {len(result.selected)}")

    async with async_session_factory() as db:
        uncached_pairs, cached, log_entries = await gather_candidate_pairs(db, result.selected)
        kept, cosine_log, numeric_exemptions = filter_by_cosine(
            uncached_pairs,
            sim_min=settings.contradiction_sim_min,
            sim_max=settings.contradiction_sim_max,
            max_pairs=settings.contradiction_max_pairs,
        )
        print(f"cached verdicts: {len(cached)}  pairs surviving cosine: {len(kept)}  numeric exemptions: {numeric_exemptions}")

        if not kept:
            print("no pairs to adjudicate")
            return

        effective_dates = await fetch_effective_dates(db, {c.document_id for pair in kept for c in pair})
        batch, pair_id_map = await adjudicate_pairs(llm, kept, list(cached.values()), effective_dates)

        print(f"\nLLM returned {len(batch.results)} verdicts")
        for v in batch.results:
            pair = pair_id_map.get(v.pair_id)
            print(f"\n--- {v.pair_id} ---")
            print(f"  is_contradiction={v.is_contradiction}  type={v.type}  severity={v.severity}  confidence={v.confidence}")
            print(f"  reconciliation={v.reconciliation}")
            print(f"  statement_a={v.statement_a!r}")
            print(f"  statement_b={v.statement_b!r}")
            print(f"  explanation={v.explanation}")

            if not v.is_contradiction:
                continue
            if pair is None:
                print("  ! unknown pair_id, skipping")
                continue
            a, b = pair
            verified = spans_present(v, a, b)
            print(f"  span_verification: {'PASS' if verified else 'FAIL'}")
            if not verified:
                continue
            if v.confidence < settings.contradiction_min_confidence:
                print(f"  DROPPED: confidence {v.confidence} < floor {settings.contradiction_min_confidence}")
                continue
            record = await upsert_contradiction(db, v, a, b, query)
            print(f"  STORED: id={record.id} status={record.status} times_seen={record.times_seen}")


if __name__ == "__main__":
    asyncio.run(main())

"""Stage 2 checkpoint — no LLM, no cosine yet. Prints candidate pairs with
rejection reasons for a real query. Confirms same-document pairs are
visibly excluded and cache lookups work.

Usage:
    python scripts/check_contradiction_pairs.py "your query here"
"""
from __future__ import annotations

import asyncio
import sys

from app.adapters import embeddings, reranker
from app.db import async_session_factory
from app.services.contradictions import gather_candidate_pairs
from app.services.retrieval import retrieve


async def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "What is the remote work policy?"
    embeddings.load_model()
    reranker.load_model()

    result = await retrieve(query, top_k=20)
    print(f"QUERY: {query!r}")
    print(f"selected chunks: {len(result.selected)}")
    for c in result.selected:
        print(f"  {c.chunk_id[:8]}  {c.document_name:32} {c.section_path}  {c.text[:55]!r}")

    async with async_session_factory() as db:
        uncached_pairs, cached, log_entries = await gather_candidate_pairs(db, result.selected)

    n_chunks = len(result.selected)
    n_total_pairs = n_chunks * (n_chunks - 1) // 2
    same_doc = sum(1 for e in log_entries if e.reason == "same_document")
    cache_hits = sum(1 for e in log_entries if e.reason == "cached_verdict")

    print(f"\ntotal pairs formed (n choose 2): {n_total_pairs}")
    print(f"same-document rejections: {same_doc}")
    print(f"cached verdicts reused: {cache_hits}")
    print(f"uncached pairs remaining (would go to cosine next): {len(uncached_pairs)}")

    print("\n--- filter log ---")
    for e in log_entries:
        print(f"  {e.chunk_a_id[:8]} x {e.chunk_b_id[:8]}  accepted={e.accepted!s:5}  reason={e.reason}")

    print("\n--- uncached pairs ---")
    for a, b in uncached_pairs:
        print(f"  {a.document_name} ({a.chunk_id[:8]}) x {b.document_name} ({b.chunk_id[:8]})")


if __name__ == "__main__":
    asyncio.run(main())

"""Stage 3 checkpoint — shows which pairs survive cosine bounds. Confirms
the near-duplicate confidentiality clause is correctly rejected and the
fee ($50 vs $75) contradiction survives the upper bound via the numeric
exemption despite being near-identical wording.

Usage:
    python scripts/check_cosine.py "your query here"
"""
from __future__ import annotations

import asyncio
import sys

from app.adapters import embeddings, reranker
from app.config import get_settings
from app.db import async_session_factory
from app.services.contradictions import filter_by_cosine, gather_candidate_pairs
from app.services.retrieval import retrieve


async def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "What are the fees and vacation policies?"
    settings = get_settings()
    embeddings.load_model()
    reranker.load_model()

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

    print(f"\nuncached pairs going into cosine: {len(uncached_pairs)}")
    print(f"numeric exemptions granted: {numeric_exemptions}")
    print(f"pairs surviving cosine: {len(kept)}")

    def _name(chunk_id: str) -> str:
        for a, b in uncached_pairs:
            if a.chunk_id == chunk_id:
                return f"{a.document_name} :: {a.text[:45]!r}"
            if b.chunk_id == chunk_id:
                return f"{b.document_name} :: {b.text[:45]!r}"
        return chunk_id[:8]

    print("\n--- cosine filter log ---")
    for e in sorted(cosine_log, key=lambda e: (e.cosine or 0), reverse=True):
        cos = f"{e.cosine:.4f}" if e.cosine is not None else "None"
        print(f"  cosine={cos}  accepted={e.accepted!s:5}  reason={e.reason:28}  {e.chunk_a_id[:8]} x {e.chunk_b_id[:8]}")

    print("\n--- surviving pairs (would go to LLM next) ---")
    for a, b in kept:
        print(f"  {a.document_name} ({a.text[:50]!r})")
        print(f"    x {b.document_name} ({b.text[:50]!r})")


if __name__ == "__main__":
    asyncio.run(main())

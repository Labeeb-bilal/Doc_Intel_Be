"""Stage 2 checkpoint — no LLM involved. Prints candidates with vector
scores, rerank scores, and rank deltas for a real query against whatever is
actually ingested right now.

Usage:
    python scripts/check_retrieval.py "your query here"
"""
from __future__ import annotations

import asyncio
import sys

from app.adapters import embeddings, reranker
from app.logging import configure_logging
from app.services.retrieval import retrieve


async def main() -> None:
    configure_logging()
    query = sys.argv[1] if len(sys.argv) > 1 else "What is this document about?"

    embeddings.load_model()
    reranker.load_model()

    result = await retrieve(query)

    print(f"\nQUERY: {result.query!r}")
    print(f"candidates retrieved: {len(result.candidates)}   selected: {len(result.selected)}")

    print("\n--- candidates (cosine-ranked) ---")
    for c in result.candidates:
        print(f"  [{c.rank_before:>2}] cosine={c.vector_score:.4f} {c.document_name!r} p.{c.page_start} {c.section_path}")

    if result.trace.rerank and result.trace.rerank.enabled:
        print("\n--- after rerank ---")
        for r in result.trace.rerank.results:
            arrow = "->" if r.rank_delta != 0 else "=="
            kept = "KEPT" if r.used_in_answer else "    "
            print(
                f"  {kept} before={r.rank_before:>2} {arrow} after={r.rank_after:>2} "
                f"(delta={r.rank_delta:+d})  cosine={r.vector_score:.4f}  rerank={r.rerank_score:.4f}  {r.chunk_id[:8]}"
            )
        print(f"\ndropped by floor ({reranker.__name__} floor): {len(result.trace.rerank.dropped)}")
    else:
        print("\n(reranking disabled or produced no stage)")

    print("\n--- selected (final, sent to answer prompt) ---")
    for c in result.selected:
        print(f"  {c.document_name} p.{c.page_start} {c.section_path} :: {c.text[:80]!r}")


if __name__ == "__main__":
    asyncio.run(main())

"""Stage 3 checkpoint — produces a grounded answer whose [Sn] markers all
resolve to real chunks, against real ingested documents and the real LLM.

Usage:
    python scripts/check_rag.py "your query here"
"""
from __future__ import annotations

import asyncio
import sys

from app.adapters import embeddings, reranker
from app.adapters.llm import get_llm_client
from app.logging import configure_logging
from app.services import rag
from app.services.retrieval import retrieve


async def main() -> None:
    configure_logging()
    query = sys.argv[1] if len(sys.argv) > 1 else "How many days per week can employees work remotely?"

    embeddings.load_model()
    reranker.load_model()
    llm = get_llm_client()

    result = await retrieve(query)
    outcome = await rag.answer(llm, result)

    print(f"\nQUERY: {query!r}")
    print(f"selected chunks: {len(result.selected)}")
    print(f"\nANSWER:\n{outcome['answer']}\n")
    print(f"grounded: {outcome['grounded']}")
    print(f"citations ({len(outcome['citations'])}):")
    for c in outcome["citations"]:
        print(f"  [{c.marker}] {c.document_name} p.{c.page} {c.section!r}")
        print(f"       {c.text[:90]!r}")

    print(f"\ncitations_dropped: {result.trace.answer.citations_dropped}")
    print(f"context truncated: {result.trace.context.truncated}")
    print(f"total_ms: {result.trace.total_ms}")

    # Sanity check the acceptance criterion directly: every [Sn] marker
    # left in the answer text must resolve to a real citation.
    import re

    markers_in_text = set()
    for group in re.findall(r"\[(S\d+(?:\s*,\s*S\d+)*)\]", outcome["answer"]):
        markers_in_text.update(m.lstrip("S") for m in re.findall(r"S(\d+)", group))
    markers_with_citations = {c.marker.lstrip("S") for c in outcome["citations"]}
    unresolved = markers_in_text - markers_with_citations
    print(f"\nunresolved markers still in text: {unresolved or 'NONE (good)'}")


if __name__ == "__main__":
    asyncio.run(main())

"""Stage 1 live checkpoint — run against the real Gemini API.

Requires GEMINI_API_KEY set in .env. Usage:
    python scripts/check_llm.py
"""
from __future__ import annotations

import asyncio

from app.adapters.llm import get_llm_client
from app.logging import configure_logging


async def main() -> None:
    configure_logging()
    client = get_llm_client()

    text = await client.complete(
        system="You are terse. Answer in exactly five words, no punctuation.",
        user="What does a semantic search index do",
    )
    print(f"\nRESPONSE: {text!r}\n")


if __name__ == "__main__":
    asyncio.run(main())

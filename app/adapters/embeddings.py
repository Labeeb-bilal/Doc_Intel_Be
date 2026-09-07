"""fastembed adapter. Every fastembed import lives here.

The model is loaded once at application startup (see main.py's lifespan),
not per request — model load/first-run download is the expensive part.

embed_documents() and embed_query() are kept as separate entry points
because BGE models are asymmetric: queries need a prefix that passages
must not have. Getting this backwards doesn't raise an error, it just
silently produces worse retrieval — phase two (retrieval) depends on this
distinction being respected everywhere.

Never add a fallback to a second model here: different embedding models
produce incompatible vector spaces, and a silent fallback would corrupt the
collection with vectors nothing else can be meaningfully compared against.
"""
from __future__ import annotations

import asyncio
from functools import lru_cache

from fastembed import TextEmbedding

from app.config import get_settings


@lru_cache
def _get_model() -> TextEmbedding:
    settings = get_settings()
    return TextEmbedding(model_name=settings.embedding_model)


def load_model() -> None:
    """Force the model to load (and download, on first run) synchronously.
    Called once at startup so the first real request never pays this cost."""
    _get_model()


def is_model_loaded() -> bool:
    return _get_model.cache_info().currsize > 0


async def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed passages/chunks for indexing. No query prefix is added."""

    def _run() -> list[list[float]]:
        model = _get_model()
        return [vector.tolist() for vector in model.passage_embed(texts)]

    return await asyncio.to_thread(_run)


async def embed_query(text: str) -> list[float]:
    """Embed a single search query. fastembed applies the asymmetric query
    prefix internally — passages must never go through this path."""

    def _run() -> list[float]:
        model = _get_model()
        return next(iter(model.query_embed([text]))).tolist()

    return await asyncio.to_thread(_run)

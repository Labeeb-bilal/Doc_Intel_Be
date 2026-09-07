"""Cross-encoder reranking adapter. Every fastembed cross-encoder import
lives here.

The cross-encoder scores (query, chunk_text) pairs together, so it sees
term interaction a bi-encoder's separately-embedded vectors cannot. Its
output is raw logits, roughly -11 to +11 — NOT a similarity score. Callers
must sigmoid-normalise before thresholding, and must never reuse the
cosine relevance floor for it: the two scales mean different things.
"""
from __future__ import annotations

import asyncio
import math
from functools import lru_cache

from fastembed.rerank.cross_encoder import TextCrossEncoder

from app.config import get_settings


@lru_cache
def _get_model() -> TextCrossEncoder:
    settings = get_settings()
    return TextCrossEncoder(model_name=settings.rerank_model)


def load_model() -> None:
    """Force the model to load at startup, alongside the bi-encoder — not
    on the first real request."""
    _get_model()


def is_model_loaded() -> bool:
    return _get_model.cache_info().currsize > 0


def sigmoid(logit: float) -> float:
    return 1.0 / (1.0 + math.exp(-logit))


async def rerank(query: str, documents: list[str]) -> list[float]:
    """Raw logits, one per document, same order as `documents`. CPU-bound —
    run through asyncio.to_thread so a ~120ms rerank never blocks the event
    loop (status polls, other requests keep being served)."""
    if not documents:
        return []

    def _run() -> list[float]:
        model = _get_model()
        return list(model.rerank(query, documents))

    return await asyncio.to_thread(_run)

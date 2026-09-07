"""Stage 2 checkpoint + spec tests #2/#3/#4 (relevance floor, rerank
ordering, rerank-failure fallback) — fully offline, Qdrant/reranker mocked."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.adapters import embeddings as embeddings_adapter
from app.adapters import reranker as reranker_adapter
from app.adapters import vectors as vectors_adapter
from app.config import get_settings
from app.services import retrieval


def _fake_point(chunk_id: str, document_id: str, ordinal: int, score: float, text: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=chunk_id,
        score=score,
        vector=[0.1, 0.2, 0.3],
        payload={
            "chunk_id": chunk_id,
            "document_id": document_id,
            "document_name": "policy.pdf",
            "file_type": "pdf",
            "ordinal": ordinal,
            "page_start": 1,
            "page_end": 1,
            "section_path": ["Eligibility"],
            "text": text,
            "embedding_model": "fake",
        },
    )


def _async_return(value):
    async def _inner(*args, **kwargs):
        return value

    return _inner


@pytest.fixture(autouse=True)
def fake_embed_query(monkeypatch):
    async def _fake(text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(embeddings_adapter, "embed_query", _fake)


@pytest.fixture(autouse=True)
def disable_neighbour_expansion(monkeypatch):
    # Keeps these tests focused on rerank ordering, not neighbour lookups
    # (which would otherwise make a real, if harmless, Qdrant call for
    # these fake document/chunk ids).
    monkeypatch.setattr(get_settings(), "neighbour_expansion", False)


async def test_relevance_floor_empties_selected_when_nothing_clears_it(monkeypatch):
    points = [
        _fake_point("c1", "d1", 0, 0.9, "irrelevant text one"),
        _fake_point("c2", "d1", 1, 0.8, "irrelevant text two"),
    ]
    monkeypatch.setattr(vectors_adapter, "search_by_vector", _async_return(points))
    monkeypatch.setattr(reranker_adapter, "rerank", _async_return([-10.0, -9.0]))  # sigmoid ~0 for both

    result = await retrieval.retrieve("some query")

    assert result.selected == []
    assert result.trace.rerank.kept == 0
    assert set(result.trace.rerank.dropped) == {"c1", "c2"}


async def test_rerank_ordering_records_rank_before_after_delta(monkeypatch):
    points = [
        _fake_point("c1", "d1", 0, 0.9, "low relevance but high cosine"),
        _fake_point("c2", "d1", 1, 0.5, "high relevance but low cosine"),
    ]
    monkeypatch.setattr(vectors_adapter, "search_by_vector", _async_return(points))
    # c1 (cosine rank 0) scores badly on rerank; c2 (cosine rank 1) scores
    # well -> they should swap places after reranking.
    monkeypatch.setattr(reranker_adapter, "rerank", _async_return([-8.0, 8.0]))

    result = await retrieval.retrieve("some query")

    entries = {e.chunk_id: e for e in result.trace.rerank.results}
    assert entries["c1"].rank_before == 0 and entries["c1"].rank_after == 1
    assert entries["c2"].rank_before == 1 and entries["c2"].rank_after == 0
    assert entries["c2"].rank_delta == 1  # moved up one position
    assert entries["c1"].rank_delta == -1  # moved down one position
    # only c2 clears the default 0.3 floor
    assert [c.chunk_id for c in result.selected] == ["c2"]


async def test_rerank_failure_falls_back_to_vector_order(monkeypatch):
    points = [
        _fake_point("c1", "d1", 0, 0.9, "text one"),
        _fake_point("c2", "d1", 1, 0.85, "text two"),
    ]
    monkeypatch.setattr(vectors_adapter, "search_by_vector", _async_return(points))

    async def _boom(*args, **kwargs):
        raise RuntimeError("cross-encoder exploded")

    monkeypatch.setattr(reranker_adapter, "rerank", _boom)

    result = await retrieval.retrieve("some query")

    assert result.trace.rerank is not None
    assert result.trace.rerank.enabled is False
    assert result.trace.rerank.error == "cross-encoder exploded"
    # A partial answer beats an error: falls back to vector order, subject
    # to the (cosine-scale) floor, and keeps going.
    assert [c.chunk_id for c in result.selected] == ["c1", "c2"]


async def test_rerank_disabled_by_config_produces_no_rerank_stage(monkeypatch):
    points = [_fake_point("c1", "d1", 0, 0.9, "text one")]
    monkeypatch.setattr(vectors_adapter, "search_by_vector", _async_return(points))

    result = await retrieval.retrieve("some query", rerank_enabled=False)

    assert result.trace.rerank is None
    assert result.selected[0].vector_score == 0.9  # vector score still present in the trace


async def test_neighbour_expansion_failure_falls_back_to_unexpanded_chunks(monkeypatch):
    # Override the file-wide autouse fixture: this test needs expansion ON
    # so it can actually fail and prove the fallback works.
    monkeypatch.setattr(get_settings(), "neighbour_expansion", True)

    doc_id = "11111111-1111-1111-1111-111111111111"
    points = [_fake_point("c1", doc_id, 5, 0.9, "original chunk text, unexpanded")]
    monkeypatch.setattr(vectors_adapter, "search_by_vector", _async_return(points))
    monkeypatch.setattr(reranker_adapter, "rerank", _async_return([8.0]))  # clears the floor easily

    async def _boom(*args, **kwargs):
        raise RuntimeError("qdrant retrieve_by_ids exploded")

    monkeypatch.setattr(vectors_adapter, "retrieve_by_ids", _boom)

    result = await retrieval.retrieve("some query")

    # Degradation ladder, step 2: neighbour expansion failed -> use the
    # chunks unexpanded, continue. Must not crash the whole request.
    assert len(result.selected) == 1
    assert result.selected[0].text == "original chunk text, unexpanded"


async def test_query_condensation_concatenates_previous_turn(monkeypatch):
    captured: dict = {}

    async def _capture_embed(text: str) -> list[float]:
        captured["query_used"] = text
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(embeddings_adapter, "embed_query", _capture_embed)
    monkeypatch.setattr(vectors_adapter, "search_by_vector", _async_return([]))

    result = await retrieval.retrieve(
        "what about part-time staff?", previous_user_message="What is the remote work policy?"
    )

    assert result.trace.condensed is True
    assert result.trace.query_raw == "what about part-time staff?"
    assert captured["query_used"] == "What is the remote work policy? what about part-time staff?"

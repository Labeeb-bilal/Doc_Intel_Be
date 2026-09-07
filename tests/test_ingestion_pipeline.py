"""Stage 4 checkpoint, plus the project's tests #4/#5/#6:
  - happy path: upload -> ready -> chunks queryable, with a fake embedder
  - forced failure: a bad document yields status='failed' with a real error
    code, never a crash
  - idempotent re-ingest: running the pipeline twice produces identical
    chunk IDs and no duplicate Qdrant points

These hit the real Postgres/Qdrant from docker-compose (that's the one
thing that can't be faked without a container), but the embedder is fully
faked — no model download, no CPU-bound work, no network.
"""
from __future__ import annotations

import hashlib
import uuid

import pytest
from sqlalchemy import select

from app.adapters import embeddings as embeddings_adapter
from app.adapters import vectors as vectors_adapter
from app.adapters.storage import get_storage_backend
from app.adapters.vectors import ping_qdrant
from app.config import get_settings
from app.db import async_session_factory, ping_db
from app.models import Chunk, Document
from app.services import ingestion

_VECTOR_DIM = 384


async def _fake_embed_documents(texts: list[str]) -> list[list[float]]:
    # Deterministic and cheap — no fastembed call, no network, no CPU work.
    return [[float((hash(text) + i) % 97) / 97 for i in range(_VECTOR_DIM)] for text in texts]


@pytest.fixture(autouse=True)
def fake_embedder(monkeypatch):
    monkeypatch.setattr(embeddings_adapter, "embed_documents", _fake_embed_documents)


@pytest.fixture(autouse=True)
async def _require_live_services():
    if not (await ping_db() and await ping_qdrant()):
        pytest.skip("requires docker-compose Postgres + Qdrant running")


async def _create_pending_document(*, filename: str, content: bytes, file_type: str = "txt") -> Document:
    storage = get_storage_backend()
    key = str(uuid.uuid4())

    async def _one_shot():
        yield content

    await storage.put(key, _one_shot())

    async with async_session_factory() as db:
        document = Document(
            original_filename=filename,
            file_type=file_type,
            size_bytes=len(content),
            content_hash=hashlib.sha256(content).hexdigest(),
            storage_key=key,
            status="pending",
        )
        db.add(document)
        await db.commit()
        await db.refresh(document)
        return document


async def _cleanup(document_id: uuid.UUID) -> None:
    async with async_session_factory() as db:
        doc = await db.get(Document, document_id)
        if doc is not None and doc.status != "processing":
            await ingestion.delete_document(db, document_id)


async def test_happy_path_upload_to_ready_and_chunks_queryable():
    doc = await _create_pending_document(
        filename="policy.txt",
        content=b"Employees may work remotely three days per week under this policy.\n\n"
        b"Manager approval is required before the arrangement begins.",
    )
    settings = get_settings()

    await ingestion.process_document(doc.id)

    async with async_session_factory() as db:
        refreshed = await db.get(Document, doc.id)
        assert refreshed.status == "ready"
        assert refreshed.chunk_count == refreshed.chunks_done
        assert refreshed.chunk_count and refreshed.chunk_count > 0
        assert refreshed.embedding_model == settings.embedding_model

        chunks, total = await ingestion.get_document_chunks(db, doc.id, limit=50, offset=0)
        assert total == refreshed.chunk_count
        assert len(chunks) == total
        assert all(c["text"] for c in chunks)  # payload text is queryable, not just stored

    await _cleanup(doc.id)


async def test_forced_failure_yields_failed_status_not_a_crash():
    doc = await _create_pending_document(filename="empty.txt", content=b"   \n\n  \n\t\n")

    await ingestion.process_document(doc.id)  # must not raise

    async with async_session_factory() as db:
        refreshed = await db.get(Document, doc.id)
        assert refreshed.status == "failed"
        assert refreshed.error_code == "EMPTY_DOCUMENT"
        assert refreshed.error_message

    await _cleanup(doc.id)


async def test_forced_failure_does_not_affect_a_sibling_document():
    good = await _create_pending_document(filename="good.txt", content=b"A perfectly normal policy paragraph.")
    bad = await _create_pending_document(filename="bad.txt", content=b"   \n\n  \n")

    await ingestion.process_document(bad.id)
    await ingestion.process_document(good.id)

    async with async_session_factory() as db:
        good_refreshed = await db.get(Document, good.id)
        bad_refreshed = await db.get(Document, bad.id)
        assert good_refreshed.status == "ready"
        assert bad_refreshed.status == "failed"

    await _cleanup(good.id)
    await _cleanup(bad.id)


async def test_idempotent_reingest_same_chunk_ids_no_duplicate_points():
    settings = get_settings()
    doc = await _create_pending_document(
        filename="idempotent.txt",
        content=b"First paragraph about eligibility rules.\n\nSecond paragraph about the review process.",
    )

    await ingestion.process_document(doc.id)
    async with async_session_factory() as db:
        rows_1 = (
            (await db.execute(select(Chunk).where(Chunk.document_id == doc.id).order_by(Chunk.ordinal)))
            .scalars()
            .all()
        )
        first_ids = [row.id for row in rows_1]
    points_after_first = await vectors_adapter.count_points_for_document(settings.qdrant_collection, doc.id)

    # Re-run the whole pipeline on the same document — simulates a manual
    # re-run or a /retry after a transient failure.
    await ingestion.process_document(doc.id)

    async with async_session_factory() as db:
        refreshed = await db.get(Document, doc.id)
        rows_2 = (
            (await db.execute(select(Chunk).where(Chunk.document_id == doc.id).order_by(Chunk.ordinal)))
            .scalars()
            .all()
        )
        second_ids = [row.id for row in rows_2]
    points_after_second = await vectors_adapter.count_points_for_document(settings.qdrant_collection, doc.id)

    assert refreshed.status == "ready"
    assert first_ids == second_ids  # deterministic uuid5(document_id, ordinal)
    assert points_after_first == points_after_second  # upsert overwrote, did not duplicate
    assert points_after_second == len(second_ids)

    await _cleanup(doc.id)

"""Ingestion service — the only place business decisions get made.

Two responsibilities live in this one file, per the stage-4 build order:
  - upload/list/get/delete/chunks: synchronous, request-scoped operations
  - process_document: the background pipeline (extract -> chunk -> embed ->
    index) that a BackgroundTasks call drives after a 202 response
"""
from __future__ import annotations

import asyncio
import hashlib
import re
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import suppress
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import magic
import structlog
from fastapi import BackgroundTasks, UploadFile
from qdrant_client import models
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import retry, stop_after_attempt, wait_exponential

from app.adapters import embeddings as embeddings_adapter
from app.adapters import vectors as vectors_adapter
from app.adapters.chunking import ChunkDraft, chunk_blocks
from app.adapters.extraction import (
    Block,
    EmptyDocumentError,
    ExtractionError,
    extract_docx,
    extract_md,
    extract_pdf,
    extract_txt,
)
from app.adapters.storage import StorageBackend, get_storage_backend
from app.config import get_settings
from app.db import async_session_factory
from app.errors import AppError, ConflictError, NotFoundError
from app.models import Chunk, Document

log = structlog.get_logger("ingestion")

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

_READ_CHUNK_BYTES = 1024 * 1024  # 1MB, per spec
_EXTENSION_TYPES = {".pdf": "pdf", ".docx": "docx", ".md": "md", ".txt": "txt"}


class _UploadResult:
    def __init__(self) -> None:
        self.size = 0
        self.hasher = hashlib.sha256()
        self.detected_mime: str | None = None


def _resolve_file_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    file_type = _EXTENSION_TYPES.get(suffix)
    if file_type is None:
        raise AppError(
            f"Unsupported file extension: {suffix or '(none)'}",
            code="INVALID_EXTENSION",
            details={"allowed": sorted(_EXTENSION_TYPES)},
        )
    return file_type


def _mime_matches(expected_type: str, mime: str) -> bool:
    if expected_type == "pdf":
        return mime == "application/pdf"
    if expected_type == "docx":
        # Some libmagic databases can't see past the OOXML zip container.
        return mime in {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/zip",
        }
    # md / txt: plain text has no distinct magic signature beyond "text/*".
    return mime.startswith("text/")


async def _read_in_chunks(file: UploadFile, size: int) -> AsyncIterator[bytes]:
    while True:
        chunk = await file.read(size)  # never call .read() with no argument
        if not chunk:
            break
        yield chunk


async def _stream_validate_and_hash(
    file: UploadFile,
    *,
    key: str,
    storage: StorageBackend,
    expected_type: str,
    max_bytes: int,
    result: _UploadResult,
) -> AsyncIterator[bytes]:
    first_chunk = True
    async for chunk in _read_in_chunks(file, _READ_CHUNK_BYTES):
        if first_chunk:
            first_chunk = False
            mime = magic.Magic(mime=True).from_buffer(chunk)
            result.detected_mime = mime
            if not _mime_matches(expected_type, mime):
                await storage.delete(key)
                raise AppError(
                    f"File content does not match its extension ({expected_type}).",
                    code="INVALID_FILE_TYPE",
                    details={"detected_mime": mime, "expected_type": expected_type},
                )
        result.size += len(chunk)
        # Inside the loop so an oversized upload aborts after the first
        # megabyte rather than after the whole file has landed.
        if result.size > max_bytes:
            await storage.delete(key)
            raise AppError(
                f"File exceeds the {max_bytes // (1024 * 1024)}MB limit.",
                code="FILE_TOO_LARGE",
                details={"max_bytes": max_bytes},
            )
        result.hasher.update(chunk)
        yield chunk

    if result.size == 0:
        await storage.delete(key)
        raise AppError("The uploaded file is empty.", code="EMPTY_FILE")


async def upload_documents(
    files: list[UploadFile],
    db: AsyncSession,
    background_tasks: BackgroundTasks,
) -> list[dict]:
    settings = get_settings()

    if not files:
        raise AppError("At least one file is required.", code="NO_FILES")
    if len(files) > settings.max_files_per_upload:
        raise AppError(
            f"Too many files: max {settings.max_files_per_upload} per upload.",
            code="TOO_MANY_FILES",
            details={"max_files": settings.max_files_per_upload, "received": len(files)},
        )

    storage = get_storage_backend()
    created_keys: list[str] = []
    created_doc_ids: list[uuid.UUID] = []
    results: list[dict] = []

    try:
        for file in files:
            file_type = _resolve_file_type(file.filename or "")
            key = str(uuid.uuid4())
            result = _UploadResult()

            await storage.put(
                key,
                _stream_validate_and_hash(
                    file,
                    key=key,
                    storage=storage,
                    expected_type=file_type,
                    max_bytes=settings.max_file_size_bytes,
                    result=result,
                ),
            )
            created_keys.append(key)
            content_hash = result.hasher.hexdigest()

            existing = await db.scalar(select(Document).where(Document.content_hash == content_hash))
            if existing is not None:
                # Redundant copy of content we already have (or already have
                # in flight) — drop it, and never create a second row: the
                # UNIQUE constraint on content_hash would reject it anyway.
                await storage.delete(key)
                created_keys.remove(key)
                results.append(
                    {
                        "id": existing.id,
                        "original_filename": existing.original_filename,
                        "status": existing.status,
                        "size_bytes": existing.size_bytes,
                        "duplicate": True,
                    }
                )
                continue

            document = Document(
                original_filename=file.filename,
                file_type=file_type,
                size_bytes=result.size,
                content_hash=content_hash,
                storage_key=key,
                status="pending",
            )
            db.add(document)
            await db.flush()  # populate document.id without committing yet
            created_doc_ids.append(document.id)
            results.append(
                {
                    "id": document.id,
                    "original_filename": document.original_filename,
                    "status": document.status,
                    "size_bytes": document.size_bytes,
                    "duplicate": False,
                }
            )

        await db.commit()
    except Exception:
        await db.rollback()
        for key in created_keys:
            with suppress(Exception):
                await storage.delete(key)
        raise

    for document_id in created_doc_ids:
        background_tasks.add_task(process_document, document_id)

    return results


# ---------------------------------------------------------------------------
# Read / delete
# ---------------------------------------------------------------------------


async def list_documents(db: AsyncSession, status: str | None = None) -> list[Document]:
    stmt = select(Document).order_by(Document.created_at.desc())
    if status:
        stmt = stmt.where(Document.status == status)
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def get_document(db: AsyncSession, document_id: uuid.UUID) -> Document:
    document = await db.get(Document, document_id)
    if document is None:
        raise NotFoundError(f"Document {document_id} not found.", code="DOCUMENT_NOT_FOUND")
    return document


async def get_document_chunks(
    db: AsyncSession, document_id: uuid.UUID, limit: int, offset: int
) -> tuple[list[dict], int]:
    await get_document(db, document_id)  # 404 if missing
    settings = get_settings()

    total = await db.scalar(select(func.count()).select_from(Chunk).where(Chunk.document_id == document_id))

    stmt = (
        select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.ordinal).limit(limit).offset(offset)
    )
    rows = list((await db.execute(stmt)).scalars().all())

    payloads = await vectors_adapter.retrieve_by_ids(settings.qdrant_collection, [row.id for row in rows])

    chunks = [
        {
            "chunk_id": row.id,
            "ordinal": row.ordinal,
            "page_start": row.page_start,
            "page_end": row.page_end,
            "section_path": row.section_path or [],
            "char_count": row.char_count,
            "text": payloads.get(str(row.id), {}).get("text", ""),
        }
        for row in rows
    ]
    return chunks, total or 0


async def delete_document(db: AsyncSession, document_id: uuid.UUID) -> None:
    document = await get_document(db, document_id)
    if document.status == "processing":
        raise ConflictError("Cannot delete a document while it is processing.")

    settings = get_settings()
    storage = get_storage_backend()

    # Qdrant first, deliberately: a mid-way failure then leaves an orphaned
    # file (harmless) rather than orphaned vectors, which would produce
    # citations pointing at a document that no longer exists.
    await vectors_adapter.delete_by_document_id(settings.qdrant_collection, document_id)
    await db.execute(delete(Chunk).where(Chunk.document_id == document_id))
    await storage.delete(document.storage_key)
    await db.delete(document)
    await db.commit()


async def retry_document(
    db: AsyncSession, document_id: uuid.UUID, background_tasks: BackgroundTasks
) -> Document:
    document = await get_document(db, document_id)
    if document.status != "failed":
        raise ConflictError("Only a failed document can be retried.")

    document.status = "pending"
    document.error_code = None
    document.error_message = None
    await db.commit()
    await db.refresh(document)

    background_tasks.add_task(process_document, document_id)
    return document


async def get_stats() -> dict:
    settings = get_settings()

    async with async_session_factory() as db:
        status_rows = (await db.execute(select(Document.status, func.count()).group_by(Document.status))).all()
        total_chunks = await db.scalar(select(func.count()).select_from(Chunk))

    qdrant_point_count = await vectors_adapter.count_points(settings.qdrant_collection)

    return {
        "documents_by_status": {status: count for status, count in status_rows},
        "total_chunks": total_chunks or 0,
        "qdrant_point_count": qdrant_point_count,
        "embedding_model": settings.embedding_model,
    }


# ---------------------------------------------------------------------------
# Background pipeline
# ---------------------------------------------------------------------------

# Documents are processed one at a time: parallel CPU-bound embedding would
# contend for the same cores and gain nothing, while making failures harder
# to reason about.
_PROCESSING_SEMAPHORE = asyncio.Semaphore(1)

_EMBED_BATCH_SIZE = 32
_UPSERT_BATCH_SIZE = 128
_MIN_CHUNK_CHARS = 200

_EXTRACTORS: dict[str, Callable[..., Iterator[Block]]] = {
    "txt": extract_txt,
    "pdf": extract_pdf,
    "docx": extract_docx,
    "md": extract_md,
}

_retry_transient = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)


class IngestionError(Exception):
    """Base for pipeline failures that map onto documents.error_code."""

    code: str = "INDEX_FAILED"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class StorageFetchError(IngestionError):
    code = "STORAGE_ERROR"


class EmbeddingFailedError(IngestionError):
    code = "EMBEDDING_FAILED"


class IndexFailedError(IngestionError):
    code = "INDEX_FAILED"


class DocumentTooLargeError(IngestionError):
    code = "DOCUMENT_TOO_LARGE"


@_retry_transient
async def _fetch_bytes(storage: StorageBackend, storage_key: str) -> bytes:
    parts = [chunk async for chunk in storage.get(storage_key)]
    return b"".join(parts)


@_retry_transient
async def _embed_batch(texts: list[str]) -> list[list[float]]:
    return await embeddings_adapter.embed_documents(texts)


@_retry_transient
async def _upsert_batch(collection_name: str, points: list[models.PointStruct]) -> None:
    await vectors_adapter.upsert_points(collection_name, points)


def _build_context_header(document_name: str, section_path: list[str], page: int | None) -> str:
    """Prepended to the text sent to the embedder only — the raw text
    (without this header) is what's stored in the Qdrant payload, so
    citations quote the original."""
    parts = [f"Document: {document_name}"]
    if section_path:
        parts.append(f"Section: {' > '.join(section_path)}")
    if page is not None:
        parts.append(f"Page {page}")
    return "[" + " | ".join(parts) + "]"


_EFFECTIVE_DATE_KEYWORDS = ("effective", "revised", "version", "last updated", "dated")
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_US_SLASH_DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
_MONTH_NAMES = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]
_MONTH_NAME_DATE_RE = re.compile(
    r"\b(" + "|".join(_MONTH_NAMES) + r")\s+(\d{1,2}),?\s+(\d{4})\b", re.IGNORECASE
)


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _parse_first_date(window: str) -> date | None:
    match = _ISO_DATE_RE.search(window)
    if match:
        y, m, d = (int(g) for g in match.groups())
        return _safe_date(y, m, d)

    match = _MONTH_NAME_DATE_RE.search(window)
    if match:
        month_name, day, year = match.groups()
        month = _MONTH_NAMES.index(month_name.lower()) + 1
        return _safe_date(int(year), month, int(day))

    match = _US_SLASH_DATE_RE.search(window)
    if match:
        m, d, y = (int(g) for g in match.groups())
        return _safe_date(y, m, d)

    return None


def _extract_effective_date(blocks: list[Block]) -> date | None:
    """Best-effort — regex the first ~2000 characters for a date near a
    handful of keywords. Null is fine; phase two uses this to distinguish a
    superseded policy from a genuine conflict."""
    text = "\n".join(b.text for b in blocks)[:2000].lower()
    for keyword in _EFFECTIVE_DATE_KEYWORDS:
        idx = text.find(keyword)
        if idx == -1:
            continue
        parsed = _parse_first_date(text[idx : idx + 100])
        if parsed:
            return parsed
    return None


def _classify_failure(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, (ExtractionError, IngestionError)):
        return exc.code, exc.message
    return "INDEX_FAILED", f"Unexpected error: {exc}"


async def process_document(document_id: uuid.UUID) -> None:
    """The whole extract -> chunk -> embed -> index pipeline for one
    document. Never raises — a failure here must not crash the background
    task or affect any other document."""
    async with _PROCESSING_SEMAPHORE:
        settings = get_settings()
        storage = get_storage_backend()

        async with async_session_factory() as db:
            document = await db.get(Document, document_id)
            if document is None:
                log.warning("document_missing_for_processing", document_id=str(document_id))
                return

            document.status = "processing"
            document.attempts += 1
            document.error_code = None
            document.error_message = None
            await db.commit()

            start = time.perf_counter()
            log.info(
                "ingestion_started",
                document_id=str(document_id),
                file_type=document.file_type,
                attempts=document.attempts,
            )

            try:
                # Clean slate first: wipe any leftovers from a previous
                # attempt so this run is idempotent whether it's a fresh
                # attempt, a /retry, or a manual re-run.
                await vectors_adapter.delete_by_document_id(settings.qdrant_collection, document_id)
                await db.execute(delete(Chunk).where(Chunk.document_id == document_id))
                await db.commit()

                try:
                    data = await _fetch_bytes(storage, document.storage_key)
                except Exception as exc:
                    raise StorageFetchError(f"Could not read the stored file: {exc}") from exc

                extractor = _EXTRACTORS.get(document.file_type)
                if extractor is None:
                    raise ExtractionError(f"Extraction for '{document.file_type}' files is not supported.")

                extraction_metadata: dict = {}
                # ExtractionError subclasses propagate untouched: deterministic, no retry.
                blocks = list(extractor(data, metadata=extraction_metadata))

                page_values = [b.page for b in blocks if b.page is not None]
                page_count = max(page_values) if page_values else None
                # Markdown frontmatter, when present, is more authoritative
                # than the generic regex scan of body text.
                effective_date = extraction_metadata.get("effective_date") or _extract_effective_date(blocks)

                chunk_drafts = list(
                    chunk_blocks(
                        blocks,
                        target_chars=settings.chunk_target_chars,
                        max_chars=settings.chunk_max_chars,
                        min_chars=_MIN_CHUNK_CHARS,
                        overlap_chars=settings.chunk_overlap_chars,
                    )
                )

                if not chunk_drafts:
                    raise EmptyDocumentError("No content survived chunking.")

                if len(chunk_drafts) > settings.max_chunks_per_document:
                    raise DocumentTooLargeError(
                        f"Document produced {len(chunk_drafts)} chunks, exceeding the "
                        f"{settings.max_chunks_per_document} limit."
                    )

                embedding_model_name = settings.embedding_model
                points_buffer: list[models.PointStruct] = []

                for batch_start in range(0, len(chunk_drafts), _EMBED_BATCH_SIZE):
                    batch = chunk_drafts[batch_start : batch_start + _EMBED_BATCH_SIZE]
                    texts_for_embedding = [
                        f"{_build_context_header(document.original_filename, d.section_path, d.page_start)}\n{d.text}"
                        for d in batch
                    ]

                    try:
                        vectors_out = await _embed_batch(texts_for_embedding)
                    except Exception as exc:
                        raise EmbeddingFailedError(f"Embedding failed: {exc}") from exc

                    for draft, vector in zip(batch, vectors_out):
                        chunk_id = vectors_adapter.compute_chunk_id(document_id, draft.ordinal)
                        points_buffer.append(
                            models.PointStruct(
                                id=str(chunk_id),
                                vector=vector,
                                payload={
                                    "chunk_id": str(chunk_id),
                                    "document_id": str(document_id),
                                    "document_name": document.original_filename,
                                    "file_type": document.file_type,
                                    "ordinal": draft.ordinal,
                                    "page_start": draft.page_start,
                                    "page_end": draft.page_end,
                                    "section_path": draft.section_path,
                                    "text": draft.text,
                                    "embedding_model": embedding_model_name,
                                },
                            )
                        )

                    while len(points_buffer) >= _UPSERT_BATCH_SIZE:
                        upsert_slice, points_buffer = (
                            points_buffer[:_UPSERT_BATCH_SIZE],
                            points_buffer[_UPSERT_BATCH_SIZE:],
                        )
                        try:
                            await _upsert_batch(settings.qdrant_collection, upsert_slice)
                        except Exception as exc:
                            raise IndexFailedError(f"Vector upsert failed: {exc}") from exc
                        document.chunks_done += len(upsert_slice)
                        await db.commit()

                if points_buffer:
                    try:
                        await _upsert_batch(settings.qdrant_collection, points_buffer)
                    except Exception as exc:
                        raise IndexFailedError(f"Vector upsert failed: {exc}") from exc
                    document.chunks_done += len(points_buffer)
                    await db.commit()

                # Ordering matters: all vectors are durable in Qdrant before
                # any chunk row exists, and chunk rows exist before status
                # flips to ready. A document is never ready with a partial
                # index.
                for draft in chunk_drafts:
                    chunk_id = vectors_adapter.compute_chunk_id(document_id, draft.ordinal)
                    db.add(
                        Chunk(
                            id=chunk_id,
                            document_id=document_id,
                            ordinal=draft.ordinal,
                            page_start=draft.page_start,
                            page_end=draft.page_end,
                            section_path=draft.section_path,
                            char_count=draft.char_count,
                        )
                    )

                document.status = "ready"
                document.page_count = page_count
                document.chunk_count = len(chunk_drafts)
                document.chunks_done = len(chunk_drafts)
                document.embedding_model = embedding_model_name
                document.effective_date = effective_date
                await db.commit()

                log.info(
                    "ingestion_completed",
                    document_id=str(document_id),
                    chunk_count=len(chunk_drafts),
                    page_count=page_count,
                    duration_s=round(time.perf_counter() - start, 2),
                )

            except Exception as exc:
                await db.rollback()
                code, message = _classify_failure(exc)
                log.warning(
                    "ingestion_failed",
                    document_id=str(document_id),
                    error_code=code,
                    error=str(exc),
                    duration_s=round(time.perf_counter() - start, 2),
                )

                # Partial-failure cleanup: never leave a document ready with
                # a partial index, and never leave orphaned vectors/rows for
                # a retry to trip over.
                with suppress(Exception):
                    await vectors_adapter.delete_by_document_id(settings.qdrant_collection, document_id)
                with suppress(Exception):
                    await db.execute(delete(Chunk).where(Chunk.document_id == document_id))
                    await db.commit()

                failed_document = await db.get(Document, document_id)
                if failed_document is not None:
                    failed_document.status = "failed"
                    failed_document.error_code = code
                    failed_document.error_message = message
                    await db.commit()


# ---------------------------------------------------------------------------
# Startup reconciler
# ---------------------------------------------------------------------------

_INTERRUPTED_AFTER = timedelta(minutes=5)


async def reconcile_interrupted_documents() -> int:
    """BackgroundTasks dies with the process — a restart mid-ingestion
    would otherwise strand a document in 'processing' forever while the
    frontend polls it indefinitely. Run on every startup."""
    cutoff = datetime.now(timezone.utc) - _INTERRUPTED_AFTER
    async with async_session_factory() as db:
        result = await db.execute(
            update(Document)
            .where(Document.status == "processing", Document.updated_at < cutoff)
            .values(
                status="failed",
                error_code="INTERRUPTED",
                error_message="Processing was interrupted by a server restart. Use /retry to re-run ingestion.",
            )
            .returning(Document.id)
        )
        ids = [row[0] for row in result.fetchall()]
        await db.commit()
        if ids:
            log.warning("interrupted_documents_reconciled", count=len(ids), document_ids=[str(i) for i in ids])
        return len(ids)

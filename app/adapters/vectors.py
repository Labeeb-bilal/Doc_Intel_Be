"""Qdrant adapter. Every qdrant-client import lives here.

Point IDs are deterministic (uuid5 of document_id + ordinal) so a retry
overwrites the same point rather than creating a duplicate — this is what
makes re-running the pipeline on the same document idempotent.
"""
from __future__ import annotations

import uuid
from functools import lru_cache

from qdrant_client import AsyncQdrantClient, models

from app.config import get_settings

VECTOR_SIZE = 384
_CHUNK_ID_NAMESPACE = uuid.NAMESPACE_URL


@lru_cache
def get_qdrant_client() -> AsyncQdrantClient:
    settings = get_settings()
    return AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


def compute_chunk_id(document_id: uuid.UUID, ordinal: int) -> uuid.UUID:
    return uuid.uuid5(_CHUNK_ID_NAMESPACE, f"{document_id}:{ordinal}")


async def ping_qdrant() -> bool:
    try:
        client = get_qdrant_client()
        await client.get_collections()
        return True
    except Exception:
        return False


async def ensure_collection(collection_name: str) -> None:
    """Idempotent: safe to call on every startup."""
    client = get_qdrant_client()
    exists = await client.collection_exists(collection_name)
    if not exists:
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        )
    await client.create_payload_index(
        collection_name=collection_name,
        field_name="document_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )


async def upsert_points(collection_name: str, points: list[models.PointStruct]) -> None:
    if not points:
        return
    client = get_qdrant_client()
    await client.upsert(collection_name=collection_name, points=points, wait=True)


async def delete_by_document_id(collection_name: str, document_id: uuid.UUID) -> None:
    client = get_qdrant_client()
    await client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=str(document_id)))]
            )
        ),
        wait=True,
    )


async def retrieve_by_ids(collection_name: str, ids: list[uuid.UUID]) -> dict[str, dict]:
    """Keyed by str(id) to sidestep int/UUID ambiguity in the client's
    return type — callers compare against str(chunk.id)."""
    if not ids:
        return {}
    client = get_qdrant_client()
    points = await client.retrieve(collection_name=collection_name, ids=[str(i) for i in ids], with_payload=True)
    return {str(point.id): (point.payload or {}) for point in points}


async def count_points(collection_name: str) -> int:
    client = get_qdrant_client()
    result = await client.count(collection_name=collection_name, exact=True)
    return result.count


async def search_by_vector(
    collection_name: str,
    *,
    query_vector: list[float],
    limit: int,
    document_ids: list[uuid.UUID] | None = None,
    with_vectors: bool = False,
) -> list[models.ScoredPoint]:
    """Cosine-ranked semantic search. `with_vectors=True` is requested by
    callers that need chunk-to-chunk cosine later (contradiction detection,
    next phase) — re-embedding a chunk you already retrieved is pure waste.
    """
    client = get_qdrant_client()
    query_filter = None
    if document_ids:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id", match=models.MatchAny(any=[str(i) for i in document_ids])
                )
            ]
        )
    response = await client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
        with_vectors=with_vectors,
    )
    return response.points


async def count_points_for_document(collection_name: str, document_id: uuid.UUID) -> int:
    client = get_qdrant_client()
    result = await client.count(
        collection_name=collection_name,
        count_filter=models.Filter(
            must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=str(document_id)))]
        ),
        exact=True,
    )
    return result.count

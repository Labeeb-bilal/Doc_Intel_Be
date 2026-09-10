"""Storage backend adapter. Every aioboto3 import lives here.

Files never land on the API's local disk as an implicit choice — callers
depend only on the StorageBackend protocol, selected at startup by
STORAGE_BACKEND. Local is the default: the corpus is small and a demo must
not depend on a third-party account being awake.
"""
from __future__ import annotations

import asyncio
import io
from collections.abc import AsyncIterator
from functools import lru_cache
from pathlib import Path
from typing import Protocol, runtime_checkable

from app.config import Settings, get_settings

_READ_CHUNK = 1024 * 1024


@runtime_checkable
class StorageBackend(Protocol):
    async def put(self, key: str, data: AsyncIterator[bytes]) -> None: ...
    async def get(self, key: str) -> AsyncIterator[bytes]: ...
    async def delete(self, key: str) -> None: ...


class StorageObjectNotFound(Exception):
    pass


class LocalStorage:
    """Writes to a mounted volume. This is the default backend."""

    def __init__(self, base_path: str):
        self._base = Path(base_path)
        self._base.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        candidate = Path(key)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"unsafe storage key: {key!r}")
        return self._base / candidate

    async def put(self, key: str, data: AsyncIterator[bytes]) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        f = await asyncio.to_thread(open, path, "wb")
        try:
            async for chunk in data:
                await asyncio.to_thread(f.write, chunk)
        finally:
            await asyncio.to_thread(f.close)

    async def get(self, key: str) -> AsyncIterator[bytes]:
        path = self._path(key)
        if not path.exists():
            raise StorageObjectNotFound(key)
        f = await asyncio.to_thread(open, path, "rb")
        try:
            while True:
                chunk = await asyncio.to_thread(f.read, _READ_CHUNK)
                if not chunk:
                    break
                yield chunk
        finally:
            await asyncio.to_thread(f.close)

    async def delete(self, key: str) -> None:
        path = self._path(key)
        await asyncio.to_thread(path.unlink, missing_ok=True)


class S3Storage:
    """aioboto3-backed storage. Driven entirely by env — the same code
    targets Cloudflare R2, Supabase Storage, or AWS S3 with no code change.
    """

    def __init__(self, *, endpoint: str | None, bucket: str, access_key: str | None,
                 secret_key: str | None, region: str | None):
        import aioboto3

        self._bucket = bucket
        self._session = aioboto3.Session()
        self._client_kwargs = {
            "endpoint_url": endpoint or None,
            "aws_access_key_id": access_key,
            "aws_secret_access_key": secret_key,
            "region_name": region or None,
        }

    async def put(self, key: str, data: AsyncIterator[bytes]) -> None:
        buf = io.BytesIO()
        async for chunk in data:
            buf.write(chunk)
        buf.seek(0)
        async with self._session.client("s3", **self._client_kwargs) as s3:
            await s3.upload_fileobj(buf, self._bucket, key)

    async def get(self, key: str) -> AsyncIterator[bytes]:
        async with self._session.client("s3", **self._client_kwargs) as s3:
            try:
                response = await s3.get_object(Bucket=self._bucket, Key=key)
            except s3.exceptions.NoSuchKey as exc:
                raise StorageObjectNotFound(key) from exc
            async with response["Body"] as stream:
                async for chunk in stream.iter_chunks(_READ_CHUNK):
                    yield chunk

    async def delete(self, key: str) -> None:
        async with self._session.client("s3", **self._client_kwargs) as s3:
            await s3.delete_object(Bucket=self._bucket, Key=key)


def _build_storage_backend(settings: Settings) -> StorageBackend:
    if settings.storage_backend == "s3":
        return S3Storage(
            endpoint=settings.s3_endpoint,
            bucket=settings.s3_bucket or "",
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
            region=settings.s3_region,
        )
    return LocalStorage(settings.storage_local_path)


@lru_cache
def get_storage_backend() -> StorageBackend:
    return _build_storage_backend(get_settings())

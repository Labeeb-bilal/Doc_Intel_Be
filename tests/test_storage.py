"""LocalStorage round-trip: put streams to disk, get streams back, delete
removes the object. S3Storage is exercised structurally only (protocol
conformance) — no S3-compatible server runs in this test suite."""
from __future__ import annotations

import pytest

from app.adapters.storage import LocalStorage, S3Storage, StorageBackend, StorageObjectNotFound


async def _one_shot(data: bytes):
    yield data


async def _chunks(data: bytes, size: int):
    for i in range(0, len(data), size):
        yield data[i : i + size]


@pytest.mark.asyncio
async def test_put_then_get_roundtrip(tmp_path):
    storage = LocalStorage(str(tmp_path))
    payload = b"hello world " * 1000

    await storage.put("abc.txt", _chunks(payload, 37))

    read_back = b""
    async for chunk in storage.get("abc.txt"):
        read_back += chunk

    assert read_back == payload


@pytest.mark.asyncio
async def test_delete_removes_object(tmp_path):
    storage = LocalStorage(str(tmp_path))
    await storage.put("to-delete.bin", _one_shot(b"data"))

    await storage.delete("to-delete.bin")

    with pytest.raises(StorageObjectNotFound):
        async for _ in storage.get("to-delete.bin"):
            pass


@pytest.mark.asyncio
async def test_delete_missing_object_is_a_noop(tmp_path):
    storage = LocalStorage(str(tmp_path))
    await storage.delete("never-existed.bin")


def test_path_traversal_key_is_rejected(tmp_path):
    storage = LocalStorage(str(tmp_path))
    with pytest.raises(ValueError):
        storage._path("../../etc/passwd")


def test_local_and_s3_conform_to_storage_backend_protocol(tmp_path):
    assert isinstance(LocalStorage(str(tmp_path)), StorageBackend)
    s3 = S3Storage(endpoint=None, bucket="test", access_key="k", secret_key="s", region="auto")
    assert isinstance(s3, StorageBackend)

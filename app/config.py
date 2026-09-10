"""Application configuration, sourced entirely from environment variables."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = Field(
        default="postgresql+asyncpg://docintel:docintel_dev_only_change_me@localhost:5433/docintel"
    )

    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_collection: str = Field(default="doc_chunks")
    # None for the local docker-compose Qdrant (no auth); Qdrant Cloud
    # requires this.
    qdrant_api_key: str | None = None

    storage_backend: Literal["local", "s3"] = Field(default="local")
    storage_local_path: str = Field(default="./storage")
    s3_endpoint: str | None = None
    s3_bucket: str | None = None
    s3_access_key: str | None = None
    s3_secret_key: str | None = None
    s3_region: str | None = None

    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")

    max_file_size_mb: int = Field(default=25)
    max_files_per_upload: int = Field(default=10)
    max_chunks_per_document: int = Field(default=3000)

    chunk_target_chars: int = Field(default=1400)
    chunk_max_chars: int = Field(default=1800)
    chunk_overlap_chars: int = Field(default=200)

    cors_origins: str = Field(default="http://localhost:3000")

    llm_model: str = Field(default="openai/gpt-oss-20b")
    groq_api_key: str | None = None
    llm_max_concurrency: int = Field(default=2)
    llm_rpm: int = Field(default=25)
    llm_timeout_seconds: int = Field(default=60)

    rerank_enabled: bool = Field(default=True)
    rerank_model: str = Field(default="Xenova/ms-marco-MiniLM-L-6-v2")

    top_k: int = Field(default=20)
    keep_n: int = Field(default=6)
    relevance_floor: float = Field(default=0.15)
    neighbour_expansion: bool = Field(default=True)
    max_context_tokens: int = Field(default=6000)

    contradiction_enabled: bool = Field(default=True)
    contradiction_sim_min: float = Field(default=0.75)
    contradiction_sim_max: float = Field(default=0.97)
    contradiction_max_pairs: int = Field(default=8)
    contradiction_min_confidence: float = Field(default=0.6)
    contradiction_llm_batch_size: int = Field(default=8)
    contradiction_max_response: int = Field(default=5)

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Settings are read once per process; env doesn't change at runtime."""
    return Settings()

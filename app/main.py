"""FastAPI application entrypoint."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.adapters import embeddings, reranker, vectors
from app.config import get_settings
from app.db import init_db
from app.errors import register_exception_handlers
from app.logging import RequestIDMiddleware, configure_logging
from app.services.ingestion import reconcile_interrupted_documents

configure_logging()
log = structlog.get_logger("startup")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # Schema creation only — no Alembic. See README for the production gap.
    await init_db()
    log.info("db_schema_ready")

    await vectors.ensure_collection(settings.qdrant_collection)
    log.info("qdrant_collection_ready", collection=settings.qdrant_collection)

    # Loaded once here, not per request. fastembed is synchronous/CPU-bound,
    # so run it off the event loop even though this only happens once.
    await asyncio.to_thread(embeddings.load_model)
    app.state.embedding_model_loaded = True
    app.state.embedding_model_name = settings.embedding_model
    log.info("embedding_model_loaded", model=settings.embedding_model)

    if settings.rerank_enabled:
        await asyncio.to_thread(reranker.load_model)
        log.info("reranker_model_loaded", model=settings.rerank_model)

    reconciled = await reconcile_interrupted_documents()
    log.info("startup_complete", reconciled_interrupted=reconciled)
    yield
    log.info("shutdown")


app = FastAPI(title="Document Intelligence — Ingestion", lifespan=lifespan)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestIDMiddleware)

register_exception_handlers(app)

from app.api.chat import router as chat_router  # noqa: E402
from app.api.contradictions import router as contradictions_router  # noqa: E402
from app.api.conversations import router as conversations_router  # noqa: E402
from app.api.documents import router as documents_router  # noqa: E402
from app.api.health import router as health_router  # noqa: E402
from app.api.stats import router as stats_router  # noqa: E402

app.include_router(health_router, prefix="/api")
app.include_router(documents_router, prefix="/api")
app.include_router(stats_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(conversations_router, prefix="/api")
app.include_router(contradictions_router, prefix="/api")

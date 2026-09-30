"""FastAPI application entry point.

Run with::

    uvicorn backend.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response

from backend.api.router import api_router
from backend.core.config import settings
from backend.core.errors import register_exception_handlers
from backend.core.logging import get_logger

logger = get_logger("backend.main")

DESCRIPTION = f"""
{settings.APP_DESCRIPTION}

**Pipeline**: upload → text extraction → cleaning → chunking →
embeddings ({settings.EMBEDDING_MODEL}) → vector database → top-{settings.TOP_K}
similarity retrieval → grounded LLM answer with source citations.

**Endpoints**

* `POST /api/documents` - upload and index one or more documents
* `GET  /api/documents` - list indexed documents
* `DELETE /api/documents/{{id}}` - remove a document and its vectors
* `POST /api/documents/rebuild` - rebuild the vector store from `uploads/`
* `POST /api/chat/ask` - ask a question (full RAG loop)
* `GET  /api/chat/history` - exported question history
* `GET  /api/stats` - dashboard statistics
* `GET  /api/system/status` - component health and pipeline stages
"""


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Prepare storage and warm the embedding model before serving traffic."""
    settings.ensure_directories()
    logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)
    logger.info(
        "Chunking: %d chars / %d overlap | retrieval: top-%d",
        settings.CHUNK_SIZE,
        settings.CHUNK_OVERLAP,
        settings.TOP_K,
    )
    logger.info("Uploads: %s", settings.UPLOAD_DIR)
    logger.info("Vector store: %s", settings.VECTOR_STORE_DIR)

    # Loading the embedding model (and on a cold cache, downloading it) is slow,
    # so pay that cost here rather than on the first user request.
    try:
        from backend.services.embedding_service import embedder_info
        from backend.services.vector_store import get_vector_store

        info = embedder_info()
        store = get_vector_store()
        logger.info(
            "Embeddings: %s (%s, %d dims) | Vector store: %s",
            info["model"],
            info["provider"],
            info["dimension"],
            store.backend,
        )
    except Exception as exc:  # noqa: BLE001 - never block startup
        logger.warning("Warm-up incomplete: %s", exc)

    yield
    try:
        from backend.services.vector_store import dispose_vector_store

        dispose_vector_store()
    except Exception as exc:  # noqa: BLE001 - never block shutdown
        logger.debug("Vector store shutdown: %s", exc)
    logger.info("Shutting down %s", settings.APP_NAME)


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)
app.include_router(api_router, prefix=settings.API_PREFIX)


@app.get("/", include_in_schema=False, response_model=None)
async def root() -> Response:
    """Send browsers to the docs, API explorers to the status endpoint."""
    if settings.DEBUG:
        return RedirectResponse(url="/docs")
    return JSONResponse(
        {
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs": "/docs",
            "api": settings.API_PREFIX,
        }
    )


@app.get("/api", include_in_schema=False)
async def api_index() -> dict:
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "endpoints": [
            f"{settings.API_PREFIX}/documents",
            f"{settings.API_PREFIX}/chat/ask",
            f"{settings.API_PREFIX}/chat/history",
            f"{settings.API_PREFIX}/stats",
            f"{settings.API_PREFIX}/system/status",
        ],
    }


@app.exception_handler(404)
async def not_found(_: Request, __: Exception) -> JSONResponse:  # pragma: no cover
    return JSONResponse(
        status_code=404,
        content={
            "error": {
                "code": "not_found",
                "message": "The requested endpoint does not exist.",
                "details": {},
            }
        },
    )

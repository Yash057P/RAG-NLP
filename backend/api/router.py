"""Aggregates every router under the configured API prefix."""

from __future__ import annotations

from fastapi import APIRouter

from backend.api import chat, documents, stats, system

api_router = APIRouter()
api_router.include_router(documents.router)
api_router.include_router(chat.router)
api_router.include_router(stats.router)
api_router.include_router(system.router)

__all__ = ["api_router"]

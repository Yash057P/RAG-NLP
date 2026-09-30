"""Shared pytest fixtures.

The tests point the application at a throwaway data directory so a real
``uploads/`` and ``vector_store/`` are never touched, and they force the fast
offline components (hashed embedder, NumPy index, extractive baseline) so the
suite runs without downloading a model or contacting an LLM.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Iterator

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session", autouse=True)
def _configure_test_environment(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Point settings at an isolated directory and pick offline components."""
    data_dir = tmp_path_factory.mktemp("rag-test-data")

    import backend.core.config as config

    config.settings.DATA_DIR = data_dir
    config.settings.UPLOAD_DIR = data_dir / "uploads"
    config.settings.VECTOR_STORE_DIR = data_dir / "vector_store"
    config.settings.REGISTRY_PATH = data_dir / "registry.json"
    config.settings.CHAT_HISTORY_PATH = data_dir / "chat_history.json"
    config.settings.EMBEDDING_PROVIDER = "hashing"
    config.settings.VECTOR_STORE_BACKEND = "numpy"
    config.settings.LLM_PROVIDER = "extractive"
    config.settings.CORS_ORIGINS = ["http://testserver"]
    config.settings.ensure_directories()

    # Drop any singletons created while importing the modules above.
    for module_path, attribute in (
        ("backend.services.embedding_service", "_embedder"),
        ("backend.services.vector_store", "_store"),
        ("backend.services.llm_factory", "_router"),
        ("backend.services.rag_service", "_rag_service"),
        ("backend.services.analytics", "_analytics"),
        ("backend.services.registry", "_registry"),
    ):
        module = importlib.import_module(module_path)
        if hasattr(module, attribute):
            setattr(module, attribute, None)

    yield

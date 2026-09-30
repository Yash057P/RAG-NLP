"""Central application configuration.

Every knob is overridable through environment variables or a ``.env`` file at the
repository root.  Defaults are chosen so that ``uvicorn backend.main:app`` works
with zero configuration on a fresh clone.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/core/config.py -> backend/core -> backend -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]

EmbeddingProvider = Literal["auto", "huggingface", "hashing"]
VectorStoreBackend = Literal["auto", "chroma", "numpy"]
LLMProvider = Literal["auto", "openai", "ollama", "mistral", "extractive"]


def _split_csv(value: object) -> object:
    """Accept both ``a,b,c`` and ``["a","b"]`` for list-valued settings."""
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            return value  # let pydantic parse the JSON form
        return [item.strip() for item in stripped.split(",") if item.strip()]
    return value


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ app
    APP_NAME: str = "RAG-NLP Assistant"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = (
        "Retrieval-Augmented Generation question answering system over "
        "user-uploaded PDF / DOCX / TXT documents."
    )
    ENVIRONMENT: Literal["development", "production", "test"] = "development"
    DEBUG: bool = True
    API_PREFIX: str = "/api"
    LOG_LEVEL: str = "INFO"

    # --------------------------------------------------------------- server
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    CORS_ORIGINS: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
            "http://127.0.0.1:4173",
        ]
    )
    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def _parse_cors(cls, value: object) -> object:
        return _split_csv(value)

    # -------------------------------------------------------------- storage
    DATA_DIR: Path = REPO_ROOT
    UPLOAD_DIR: Path = REPO_ROOT / "uploads"
    VECTOR_STORE_DIR: Path = REPO_ROOT / "vector_store"
    REGISTRY_PATH: Path = REPO_ROOT / "data" / "registry.json"
    CHAT_HISTORY_PATH: Path = REPO_ROOT / "data" / "chat_history.json"

    MAX_UPLOAD_SIZE_MB: int = 25
    ALLOWED_EXTENSIONS: list[str] = Field(
        default_factory=lambda: ["pdf", "docx", "txt"]
    )
    @field_validator("ALLOWED_EXTENSIONS", mode="before")
    @classmethod
    def _parse_exts(cls, value: object) -> object:
        return _split_csv(value)

    # ------------------------------------------------------------- chunking
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    CHUNK_SEPARATOR: str = "\n\n"

    # ----------------------------------------------------------- embeddings
    EMBEDDING_PROVIDER: EmbeddingProvider = "auto"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_DEVICE: str = "cpu"
    EMBEDDING_BATCH_SIZE: int = 32
    # Used by the dependency-free fallback embedder so that a clone runs
    # offline while still producing meaningful cosine similarities.
    HASHING_NGRAM_RANGE: int = 3

    # --------------------------------------------------------- vector store
    VECTOR_STORE_BACKEND: VectorStoreBackend = "auto"
    CHROMA_COLLECTION: str = "rag_documents"
    CHROMA_PERSIST_DIRECTORY: Path | None = None

    # ------------------------------------------------------------ retrieval
    TOP_K: int = 5
    MIN_SIMILARITY: float = 0.05
    # Fraction of question terms that must appear in the retrieved context
    # before the LLM is allowed to answer instead of abstaining.
    GROUNDING_THRESHOLD: float = 0.18
    # Bibliography chunks are scaled by this before ranking. A citation entry
    # repeats its own paper's title, so it scores very high on topical queries
    # while containing nothing that answers them; demoting it lets real prose
    # win without hiding the entry from the results entirely.
    REFERENCE_SCORE_PENALTY: float = 0.35
    # How many times top_k the store is asked for, so demoted bibliography
    # chunks can be replaced by real prose ranked just below the cut.
    RETRIEVAL_CANDIDATE_MULTIPLIER: int = 4

    # ------------------------------------------------------------------ llm
    LLM_PROVIDER: LLMProvider = "auto"
    LLM_MODEL: str = "llama3.1"
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 700
    LLM_TIMEOUT_SECONDS: float = 120.0

    OPENAI_API_KEY: str | None = None
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = "gpt-4o-mini"

    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    OLLAMA_MODEL: str = "llama3.1"

    MISTRAL_API_KEY: str | None = None
    MISTRAL_BASE_URL: str = "https://api.mistral.ai/v1"
    MISTRAL_MODEL: str = "mistral-small-latest"

    # --------------------------------------------------------- derived data
    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    @property
    def chroma_persist_path(self) -> Path:
        return self.CHROMA_PERSIST_DIRECTORY or (self.VECTOR_STORE_DIR / "chroma")

    @property
    def numpy_store_path(self) -> Path:
        return self.VECTOR_STORE_DIR / "numpy_store"

    def ensure_directories(self) -> None:
        """Create every directory the app writes to. Idempotent."""
        for path in (
            self.DATA_DIR,
            self.UPLOAD_DIR,
            self.VECTOR_STORE_DIR,
            self.REGISTRY_PATH.parent,
            self.CHAT_HISTORY_PATH.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def public_dict(self) -> dict:
        """Config summary that is safe to expose over the API (no secrets)."""
        return {
            "app_name": self.APP_NAME,
            "version": self.APP_VERSION,
            "environment": self.ENVIRONMENT,
            "api_prefix": self.API_PREFIX,
            "chunking": {
                "chunk_size": self.CHUNK_SIZE,
                "chunk_overlap": self.CHUNK_OVERLAP,
            },
            "retrieval": {
                "top_k": self.TOP_K,
                "min_similarity": self.MIN_SIMILARITY,
                "grounding_threshold": self.GROUNDING_THRESHOLD,
            },
            "upload": {
                "max_size_mb": self.MAX_UPLOAD_SIZE_MB,
                "allowed_extensions": self.ALLOWED_EXTENSIONS,
            },
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings


settings = get_settings()

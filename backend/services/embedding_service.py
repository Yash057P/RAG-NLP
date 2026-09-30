"""Embedding generation.

Primary path: ``sentence-transformers/all-MiniLM-L6-v2`` through LangChain's
``HuggingFaceEmbeddings`` (or the raw ``sentence_transformers`` model when
LangChain integrations are not installed).

Fallback path: a deterministic hashed bag-of-n-grams embedder. It needs no
model download, produces the same 384 dimensions and meaningful cosine
similarities, so the retrieval pipeline stays demonstrable offline. The active
provider is always reported to the UI so nobody mistakes one for the other.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from collections import Counter
from typing import Protocol, runtime_checkable

from backend.core.config import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Embedder(Protocol):
    """Minimal embedding interface used by the rest of the pipeline."""

    name: str
    provider: str
    dimension: int
    available: bool

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


# ------------------------------------------------------ fallback embedder ---
class HashingEmbedder:
    """Hashed bag-of-words / character n-grams embedder (no dependencies).

    Each feature is hashed into ``dimension`` buckets with a deterministic
    sign, weighted by a sub-linear term frequency and L2 normalised, so cosine
    similarity reduces to a smoothed lexical overlap score. Good enough to
    demonstrate retrieval when ``torch`` / ``sentence-transformers`` are not
    installed.
    """

    provider = "hashing"

    def __init__(self, dimension: int | None = None) -> None:
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        self.name = f"hashing-ngram-{self.dimension}d"
        self.available = True

    # -- feature extraction -------------------------------------------------
    def _features(self, text: str) -> Counter:
        tokens = _TOKEN_RE.findall(text.lower())
        features: Counter = Counter()
        for token in tokens:
            features[f"w:{token}"] += 1
        for first, second in zip(tokens, tokens[1:]):
            features[f"b:{first}_{second}"] += 1
        n = max(1, settings.HASHING_NGRAM_RANGE)
        for token in tokens:
            if len(token) < n:
                features[f"c:{token}"] += 1
                continue
            padded = f"^{token}$"
            for i in range(len(padded) - n + 1):
                features[f"c:{padded[i : i + n]}"] += 1
        return features

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        return value % self.dimension, 1.0 if (value >> 63) & 1 else -1.0

    def _vectorize(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        if not text.strip():
            vector[0] = 1.0
            return vector
        for feature, count in self._features(text).items():
            index, sign = self._bucket(feature)
            vector[index] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            vector[0] = 1.0
            return vector
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorize(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectorize(text)


# ------------------------------------------------- sentence-transformers ---
class SentenceTransformerEmbedder:
    """Wraps ``sentence-transformers`` in the LangChain Embeddings interface."""

    provider = "huggingface"

    def __init__(self) -> None:
        self.name = settings.EMBEDDING_MODEL
        self.dimension = settings.EMBEDDING_DIMENSION
        self.available = True
        self._backend = "langchain"
        self._embeddings: object | None = None
        self._model: object | None = None
        self._load()

    def _load(self) -> None:
        try:
            from langchain_huggingface import (  # type: ignore[import-untyped]
                HuggingFaceEmbeddings,
            )

            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.name,
                model_kwargs={"device": settings.EMBEDDING_DEVICE},
                encode_kwargs={"batch_size": settings.EMBEDDING_BATCH_SIZE},
            )
            self._backend = "langchain_huggingface"
            logger.info("Embeddings ready via LangChain: %s", self.name)
            return
        except ImportError:
            logger.info("langchain_huggingface missing, using sentence-transformers")
        except Exception as exc:  # noqa: BLE001
            logger.warning("LangChain embeddings failed (%s), trying raw model", exc)

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(
                self.name, device=settings.EMBEDDING_DEVICE
            )
            self._backend = "sentence_transformers"
            self.dimension = int(self._model.get_sentence_embedding_dimension())
            logger.info("Embeddings ready via sentence-transformers: %s", self.name)
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed and "
                "langchain_huggingface is unavailable."
            ) from exc

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._embeddings is not None:
            return [list(map(float, vec)) for vec in self._embeddings.embed_documents(texts)]
        if self._model is not None:
            vectors = self._model.encode(  # type: ignore[attr-defined]
                texts,
                batch_size=settings.EMBEDDING_BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return [[float(value) for value in row] for row in vectors]
        raise RuntimeError("No embedding backend was initialised.")

    def embed_query(self, text: str) -> list[float]:
        if self._embeddings is not None:
            return [float(value) for value in self._embeddings.embed_query(text)]
        if self._model is not None:
            vector = self._model.encode(  # type: ignore[attr-defined]
                text, convert_to_numpy=True, normalize_embeddings=True
            )
            return [float(value) for value in vector]
        raise RuntimeError("No embedding backend was initialised.")


# ------------------------------------------------------------- singleton ---
_embedder: Embedder | None = None
_embedder_lock = threading.Lock()


def _build_embedder() -> Embedder:
    provider = settings.EMBEDDING_PROVIDER

    if provider == "huggingface":
        return SentenceTransformerEmbedder()

    if provider == "hashing":
        return HashingEmbedder()

    # provider == "auto": prefer the real model, degrade gracefully.
    try:
        return SentenceTransformerEmbedder()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Falling back to the offline hashing embedder (%s). "
            "Install sentence-transformers for all-MiniLM-L6-v2 vectors.",
            exc,
        )
        return HashingEmbedder()


def get_embedder() -> Embedder:
    """Return the process-wide embedder, loading the model on first use."""
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                _embedder = _build_embedder()
    return _embedder


def embedder_info() -> dict:
    """Status payload for ``/api/system/status`` and the dashboard."""
    try:
        embedder = get_embedder()
    except Exception as exc:  # noqa: BLE001
        return {
            "provider": "unavailable",
            "model": settings.EMBEDDING_MODEL,
            "dimension": 0,
            "available": False,
            "detail": str(exc),
        }
    return {
        "provider": embedder.provider,
        "model": embedder.name,
        "dimension": embedder.dimension,
        "available": True,
        "detail": (
            "sentence-transformers"
            if embedder.provider == "huggingface"
            else "offline fallback (no model download)"
        ),
    }


def reset_embedder() -> None:
    """Drop the cached model - used by tests and the rebuild endpoint."""
    global _embedder
    with _embedder_lock:
        _embedder = None

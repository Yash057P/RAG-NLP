"""Shared utilities: safe filenames, human readable sizes, slugify."""

from __future__ import annotations

import hashlib
import re
import unicodedata
import uuid
from pathlib import Path

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_MULTI_UNDERSCORE = re.compile(r"_{2,}")
# Uploads are persisted as "{document_id}_{safe_filename}"; rebuilding the index
# has to undo that prefix instead of showing "doc_ab12…report.pdf" to the user.
_STORED_NAME = re.compile(r"^(?P<doc_id>doc_[0-9a-f]{6,32})_(?P<name>.+)$", re.IGNORECASE)


def generate_id(prefix: str = "") -> str:
    """Short, collision-resistant identifier."""
    raw = uuid.uuid4().hex[:12]
    return f"{prefix}{raw}" if prefix else raw


def content_fingerprint(data: bytes) -> str:
    """Stable content hash used to detect re-uploads of the same file."""
    return hashlib.sha256(data).hexdigest()[:16]


def safe_filename(filename: str, fallback: str = "upload") -> str:
    """Strip any directory component and unsafe characters from a filename."""
    if not filename:
        return fallback
    base = Path(filename.replace("\\", "/")).name.strip()
    base = unicodedata.normalize("NFKD", base)
    base = _UNSAFE_CHARS.sub("_", base)
    base = _MULTI_UNDERSCORE.sub("_", base).strip("._")
    return base[:180] or fallback


def file_extension(filename: str) -> str:
    """Lowercase extension without the leading dot (``""`` when absent)."""
    return Path(filename).suffix.lower().lstrip(".")


def split_stored_filename(stored: str) -> tuple[str | None, str]:
    """Recover ``(document_id, original_filename)`` from a stored upload name.

    Returns ``(None, stored)`` for files that were not written by the API (for
    example one a user dropped into ``uploads/`` by hand).
    """
    match = _STORED_NAME.match(stored)
    if not match:
        return None, stored
    return match.group("doc_id"), match.group("name")


def title_context(filename: str) -> str:
    """Readable title derived from a filename, for title-aware retrieval.

    ``Experiment_5_DAV.docx`` becomes ``Experiment 5 DAV``. Underscores, dashes
    and camel-case boundaries become spaces so multi-word acronyms survive
    tokenisation, and the words are repeated so the title is not dominated by a
    single token in the embedding. Used to prefix the text sent to the
    embedder; the stored chunk text is left verbatim.
    """
    stem = Path(filename).stem
    # "PaperNLPClassifier" -> "Paper NLP Classifier". The first rule breaks a
    # lowercase-to-uppercase boundary, the second an acronym off a following
    # capitalised word ("NLPC" + "lassifier").
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", stem)
    spaced = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", spaced)
    words = [w for w in re.split(r"[^A-Za-z0-9]+", spaced) if w]
    if not words:
        return stem
    return " ".join(words * 2)


def human_size(num_bytes: float) -> str:
    """Format a byte count as e.g. ``1.4 MB``."""
    if num_bytes < 1024:
        return f"{int(num_bytes)} B"
    value = float(num_bytes)
    for unit in ("KB", "MB", "GB"):
        value /= 1024.0
        if value < 1024.0 or unit == "GB":
            return f"{value:.1f} {unit}"
    return f"{value:.1f} GB"


def slugify(value: str, max_length: int = 64) -> str:
    """ASCII slug used for collection names and DOM ids."""
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", normalized).strip("_").lower()
    return (slug[:max_length] or "item").rstrip("_")


def truncate(text: str, limit: int, suffix: str = "...") -> str:
    """Trim ``text`` to ``limit`` characters, appending ``suffix``."""
    if len(text) <= limit:
        return text
    keep = max(0, limit - len(suffix))
    return text[:keep].rstrip() + suffix

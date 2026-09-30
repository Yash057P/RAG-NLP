"""Document processing pipeline: extraction, cleaning and chunking.

Implements steps 1-3 of the RAG pipeline:

1. ``extract``  - parse PDF / DOCX / TXT into text (page aware when possible)
2. ``clean``    - normalise unicode, de-hyphenate, drop repeated page furniture
3. ``chunk``    - split into ``CHUNK_SIZE`` char windows with ``CHUNK_OVERLAP``

LangChain's ``RecursiveCharacterTextSplitter`` is used when the dependency is
installed; an equivalent native splitter keeps the system fully functional in a
minimal install.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from backend.core.config import settings
from backend.core.errors import (
    EmptyDocumentError,
    ExtractionError,
    UnsupportedFileTypeError,
)
from backend.core.logging import get_logger
from backend.utils import text as text_utils

logger = get_logger(__name__)

# Tried in order; the first one that yields text wins.
PDF_BACKENDS = ("pdfplumber", "pypdf")

ENCODING_CANDIDATES = ("utf-8-sig", "utf-8", "utf-16", "cp1252", "latin-1")

# Separators for the recursive splitter, ordered coarse -> fine.
SPLIT_SEPARATORS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", "")


@dataclass(slots=True)
class Segment:
    """A contiguous run of text plus the page it came from (if known)."""

    text: str
    page: int | None = None


@dataclass(slots=True)
class Chunk:
    """A chunk ready to be embedded and stored."""

    text: str
    index: int
    char_start: int
    char_end: int
    page: int | None = None

    def metadata(self) -> dict:
        return {
            "chunk_index": self.index,
            "page": self.page,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "char_count": len(self.text),
        }


@dataclass(slots=True)
class ExtractionResult:
    text: str
    segments: list[Segment] = field(default_factory=list)
    page_count: int | None = None
    method: str = "unknown"
    char_count: int = 0
    word_count: int = 0


@dataclass(slots=True)
class ProcessingResult:
    """Everything the ingestion pipeline needs after step 3."""

    raw_text: str
    clean_text: str
    chunks: list[Chunk]
    page_count: int | None
    extraction_method: str
    raw_char_count: int
    clean_char_count: int
    word_count: int
    timing_ms: dict[str, float]


# ------------------------------------------------------------- extraction ---
def _decode_bytes(data: bytes) -> str:
    for encoding in ENCODING_CANDIDATES:
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace")


def _extract_pdf(path: Path) -> ExtractionResult:
    errors: list[str] = []

    for backend in PDF_BACKENDS:
        try:
            segments = _extract_pdf_with(backend, path)
        except ImportError:
            continue
        except Exception as exc:  # noqa: BLE001 - try the next backend
            errors.append(f"{backend}: {exc}")
            logger.warning("PDF backend %s failed for %s: %s", backend, path.name, exc)
            continue

        text = "\n\n".join(segment.text for segment in segments if segment.text.strip())
        if text.strip():
            return ExtractionResult(
                text=text,
                segments=segments,
                page_count=len(segments),
                method=backend,
                char_count=len(text),
            )
        errors.append(f"{backend}: no selectable text layer")

    detail = "; ".join(errors) or "no PDF backend installed"
    raise ExtractionError(
        "Could not extract text from this PDF. It may be a scanned image "
        "without an embedded text layer (OCR is out of scope for this lab).",
        details={"attempts": detail},
    )


def _extract_pdf_with(backend: str, path: Path) -> list[Segment]:
    if backend == "pdfplumber":
        import pdfplumber  # type: ignore[import-untyped]

        segments: list[Segment] = []
        with pdfplumber.open(path) as pdf:
            for number, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                segments.append(Segment(text=page_text, page=number))
        return segments

    if backend == "pypdf":
        from pypdf import PdfReader  # type: ignore[import-untyped]

        reader = PdfReader(str(path))
        return [
            Segment(text=page.extract_text() or "", page=number)
            for number, page in enumerate(reader.pages, start=1)
        ]

    raise ValueError(f"unknown pdf backend: {backend}")


def _extract_docx(path: Path) -> ExtractionResult:
    try:
        import docx  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - dependency is required
        raise ExtractionError(
            "python-docx is not installed, cannot read DOCX files."
        ) from exc

    document = docx.Document(str(path))
    blocks: list[str] = []

    for paragraph in document.paragraphs:
        content = paragraph.text.strip()
        if not content:
            continue
        style = (paragraph.style.name or "").lower() if paragraph.style else ""
        if style.startswith("heading"):
            blocks.append(f"{'#' * 2} {content}" if style == "heading 1" else content)
        else:
            blocks.append(content)

    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                blocks.append(" | ".join(cells))

    for section in document.sections:
        footer = section.footer
        if footer is not None:
            for paragraph in footer.paragraphs:
                if paragraph.text.strip():
                    blocks.append(paragraph.text.strip())

    text = "\n\n".join(blocks)
    if not text.strip():
        raise EmptyDocumentError("The DOCX file contains no readable text.")

    return ExtractionResult(
        text=text,
        segments=[Segment(text=text, page=None)],
        page_count=None,
        method="python-docx",
        char_count=len(text),
    )


def _extract_txt(path: Path) -> ExtractionResult:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ExtractionError(f"Could not read file: {exc}") from exc

    text = _decode_bytes(raw)
    if not text.strip():
        raise EmptyDocumentError("The TXT file is empty.")

    return ExtractionResult(
        text=text,
        segments=[Segment(text=text, page=None)],
        page_count=None,
        method=ENCODING_CANDIDATES[0],
        char_count=len(text),
    )


def extract(path: Path, extension: str | None = None) -> ExtractionResult:
    """Step 1 - dispatch to the reader for this file type."""
    ext = (extension or path.suffix).lower().lstrip(".")
    if not Path(path).exists():
        raise ExtractionError(f"File not found on disk: {path.name}")

    if ext == "pdf":
        return _extract_pdf(path)
    if ext == "docx":
        return _extract_docx(path)
    if ext == "txt":
        return _extract_txt(path)
    raise UnsupportedFileTypeError(
        f"Unsupported file type '.{ext}'. Allowed: "
        f"{', '.join('.' + e for e in settings.ALLOWED_EXTENSIONS)}"
    )


# --------------------------------------------------------------- chunking ---
def _load_native_splitter():
    """Return a splitter exposing ``split_text(text) -> list[str]``.

    LangChain's ``RecursiveCharacterTextSplitter`` is used when installed; a
    small native implementation with the same semantics keeps the module working
    in a minimal install.
    """
    try:
        from langchain_text_splitters import (  # type: ignore[import-untyped]
            RecursiveCharacterTextSplitter,
        )

        return RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=0,  # overlap is applied by _apply_overlap below
            separators=list(SPLIT_SEPARATORS),
            keep_separator=True,
            add_start_index=False,
            length_function=len,
        )
    except ImportError:

        class _NativeSplitter:
            """Recursive character splitter used when LangChain is absent."""

            def __init__(
                self,
                chunk_size: int,
                separators: tuple[str, ...] = SPLIT_SEPARATORS,
                **_kwargs,
            ) -> None:
                self.chunk_size = chunk_size
                self.separators = separators

            def _recurse(self, text: str, level: int) -> list[str]:
                if len(text) <= self.chunk_size or level >= len(self.separators) - 1:
                    return [text] if text else []
                separator = self.separators[level]
                pieces = text.split(separator) if separator else list(text)
                if len(pieces) == 1:
                    return self._recurse(text, level + 1)

                results: list[str] = []
                buffer = ""
                for piece in pieces:
                    # Keep the separator attached to the front of each piece.
                    candidate = piece if not buffer else buffer + separator + piece
                    if len(candidate) <= self.chunk_size:
                        buffer = candidate
                        continue
                    if buffer:
                        results.append(buffer)
                    if len(piece) > self.chunk_size:
                        results.extend(self._recurse(piece, level + 1))
                        buffer = ""
                    else:
                        buffer = piece
                if buffer:
                    results.append(buffer)
                return results

            def split_text(self, text: str) -> list[str]:
                return [piece for piece in self._recurse(text, 0) if piece]

        return _NativeSplitter(chunk_size=settings.CHUNK_SIZE)


def _align_to_word(text: str, index: int) -> int:
    """Move ``index`` forward to the next word boundary."""
    while index < len(text) and not text[index].isspace():
        index += 1
    return index


def _apply_overlap(
    pieces: list[tuple[str, int]], chunk_size: int, overlap: int
) -> list[tuple[str, int]]:
    """Re-window ``(text, offset)`` pieces into overlapping chunks.

    LangChain only overlaps *across* documents, never inside a single string, so
    the overlap required by the lab specification is applied here. The tail of
    each emitted chunk is carried into the next one, aligned to a word boundary
    so chunks do not start mid-word.

    Offsets are tracked compositionally rather than by searching the source for
    each window, which keeps them exact even when the document repeats itself.
    """
    if not pieces:
        return []

    windows: list[tuple[str, int]] = []
    buffer = ""
    buffer_start = 0

    def flush() -> None:
        nonlocal buffer, buffer_start
        if buffer.strip():
            windows.append((buffer, buffer_start))
        buffer = ""

    for piece, offset in pieces:
        if not piece:
            continue
        if buffer and buffer.endswith(piece):
            continue  # already carried in by the overlap tail

        if not buffer:
            buffer, buffer_start = piece, offset
        elif len(buffer) + len(piece) <= chunk_size:
            buffer += piece
        else:
            # Capture the pending window before flushing so the overlap tail is
            # taken from the chunk that actually precedes this one.
            previous = (buffer, buffer_start)
            flush()

            if overlap > 0 and previous[0].strip():
                tail_start = max(0, len(previous[0]) - overlap)
                tail_start = _align_to_word(previous[0], tail_start)
                tail = previous[0][tail_start:]
                if tail.strip() and len(tail) + len(piece) <= chunk_size:
                    buffer = tail + piece
                    buffer_start = previous[1] + tail_start
                    continue

            buffer = piece if len(piece) <= chunk_size else piece[:chunk_size]
            buffer_start = offset

    flush()
    return windows


def _page_lookup(segments: list[Segment]) -> tuple[str, list[tuple[int, int, int]]]:
    """Join segments and record the character span of each page."""
    boundaries: list[tuple[int, int, int]] = []
    parts: list[str] = []
    cursor = 0
    for segment in segments:
        cleaned = segment.text
        if not cleaned.strip():
            continue
        if parts:
            parts.append("\n\n")
            cursor += 2
        start = cursor
        parts.append(cleaned)
        cursor += len(cleaned)
        if segment.page is not None:
            boundaries.append((start, cursor, segment.page))
    return "".join(parts), boundaries


def _page_for_offset(offset: int, boundaries: list[tuple[int, int, int]]) -> int | None:
    for start, end, page in boundaries:
        if start <= offset < end:
            return page
    return boundaries[-1][2] if boundaries else None


def _locate_pieces(text: str, pieces: list[str]) -> list[tuple[str, int]]:
    """Attach the source offset to every split piece.

    Pieces are ordered, so scanning forward with a cursor that only ever moves
    past consumed text resolves each one exactly. Any gap the splitter left
    behind (it can discard a separator) is absorbed so the pieces tile the
    source contiguously - without that, a window built from a tail plus the next
    piece would not be a verbatim substring of the document.
    """
    located: list[tuple[str, int]] = []
    cursor = 0
    for piece in pieces:
        if not piece:
            continue
        start = text.find(piece, cursor)
        if start < 0:  # defensive: should not happen for verbatim pieces
            located.append((piece, cursor))
            cursor += len(piece)
            continue
        end = start + len(piece)
        if start > cursor:
            piece = text[cursor:end]
            start = cursor
        located.append((piece, start))
        cursor = end
    return located


def chunk_text(text: str, boundaries: list[tuple[int, int, int]]) -> list[Chunk]:
    """Step 3 - split the cleaned text into overlapping, page-tagged chunks."""
    if not text.strip():
        return []

    splitter = _load_native_splitter()
    located = _locate_pieces(text, splitter.split_text(text))
    windows = _apply_overlap(located, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)

    chunks: list[Chunk] = []
    for window, start in windows:
        content = window.strip()
        if not content:
            continue
        # Skip past leading whitespace so char_start/char_end address the stored
        # text exactly.
        char_start = start + (len(window) - len(window.lstrip()))
        chunks.append(
            Chunk(
                text=content,
                index=len(chunks),
                char_start=char_start,
                char_end=char_start + len(content),
                page=_page_for_offset(char_start, boundaries),
            )
        )
    return chunks


# ----------------------------------------------------------------- driver ---
def process_file(path: Path, extension: str | None = None) -> ProcessingResult:
    """Run the full text pipeline for a single file on disk."""
    timings: dict[str, float] = {}

    started = time.perf_counter()
    extraction = extract(path, extension)
    timings["extraction"] = (time.perf_counter() - started) * 1000

    # Clean each segment separately and keep the page it came from, so the
    # character offsets stored with every chunk still address the cleaned text
    # and resolve to the right page. Boilerplate is detected over the whole
    # document first so running headers are still recognised page by page.
    started = time.perf_counter()
    boilerplate = text_utils.detect_boilerplate(extraction.text)
    cleaned_segments = [
        Segment(text=text_utils.clean_text(segment.text, boilerplate), page=segment.page)
        for segment in extraction.segments
    ]
    clean, boundaries = _page_lookup(cleaned_segments)
    timings["cleaning"] = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    chunks = chunk_text(clean, boundaries)
    timings["chunking"] = (time.perf_counter() - started) * 1000

    if not chunks:
        raise EmptyDocumentError(
            "No usable text was produced after cleaning; the document appears "
            "to contain no indexable content."
        )

    return ProcessingResult(
        raw_text=extraction.text,
        clean_text=clean,
        chunks=chunks,
        page_count=extraction.page_count,
        extraction_method=extraction.method,
        raw_char_count=len(extraction.text),
        clean_char_count=len(clean),
        word_count=len(clean.split()),
        timing_ms=timings,
    )

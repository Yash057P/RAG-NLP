"""Unit tests for the text processing pipeline (steps 1-3)."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from backend.core.config import settings
from backend.services.document_processor import (
    _apply_overlap,
    _page_lookup,
    chunk_text,
    extract,
    process_file,
    Segment,
)
from backend.utils.text import clean_text, term_coverage, tokenize


class TestCleaning:
    def test_normalises_unicode_and_whitespace(self) -> None:
        messy = "Hello world!  \n\n\n\n  Second   line\t\t."
        cleaned = clean_text(messy)
        assert "\u00a0" not in cleaned
        assert "\u200b" not in cleaned
        assert "  " not in cleaned
        assert cleaned.startswith("Hello world!")
        assert "\n\n\n" not in cleaned

    def test_joins_hyphenated_line_breaks(self) -> None:
        assert "information" in clean_text("the infor-\nmation was lost")

    def test_drops_repeated_page_furniture(self) -> None:
        pages = [f"Page {n} of 20" for n in range(1, 21)]
        body = "Meaningful paragraph about retrieval augmented generation.\n\n"
        cleaned = clean_text(body + "\n".join(pages))
        assert "Page 7 of 20" not in cleaned
        assert "retrieval augmented generation" in cleaned

    def test_tokenizer_drops_stopwords_for_coverage(self) -> None:
        assert term_coverage("what is the objective", "the objective is stated") > 0
        assert term_coverage("what is the", "unrelated") == 1.0
        assert tokenize("RAG-based, 42 times!") == ["rag", "based", "42", "times"]


class TestChunking:
    @pytest.fixture
    def sample(self) -> str:
        paragraph = (
            "Retrieval augmented generation grounds an answer in retrieved "
            "passages so that the model cannot invent facts. "
        )
        return "\n\n".join(f"Section {i}. {paragraph * 3}" for i in range(4))

    def test_respects_chunk_size(self, sample: str) -> None:
        chunks = chunk_text(sample, [])
        assert len(chunks) > 1
        assert all(len(chunk.text) <= settings.CHUNK_SIZE for chunk in chunks)

    def test_offsets_address_the_stored_text(self, sample: str) -> None:
        for chunk in chunk_text(sample, []):
            assert sample[chunk.char_start : chunk.char_end] == chunk.text

    def test_consecutive_chunks_overlap(self, sample: str) -> None:
        chunks = chunk_text(sample, [])
        overlapping = [
            (a, b)
            for a, b in zip(chunks, chunks[1:])
            if set(a.text[-120:].split()) & set(b.text.split())
        ]
        assert overlapping, "expected at least one overlapping chunk boundary"

    def test_indices_are_sequential(self, sample: str) -> None:
        chunks = chunk_text(sample, [])
        assert [chunk.index for chunk in chunks] == list(range(len(chunks)))

    def test_no_coverage_is_lost(self) -> None:
        pieces = [("a" * 300, 0), ("b" * 300, 300), ("c" * 300, 600)]
        windows = _apply_overlap(pieces, chunk_size=500, overlap=100)
        assert all(len(window) <= 500 for window, _ in windows)
        # Overlap means windows share characters, but every piece survives.
        joined = "".join(window for window, _ in windows)
        for piece, _ in pieces:
            assert piece in joined

    def test_page_provenance_is_tracked(self) -> None:
        page_one = "Alpha content about the first page of the document."
        page_two = "Beta content about the second page of the document."
        joined, boundaries = _page_lookup(
            [Segment(page_one, 1), Segment(page_two, 2)]
        )
        assert "Alpha" in joined and "Beta" in joined
        chunks = chunk_text(joined, boundaries)
        pages = {chunk.page for chunk in chunks}
        assert pages <= {1, 2}
        assert pages


class TestExtraction:
    def test_reads_txt(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.txt"
        path.write_text("Plain text notes for the RAG system.", "utf-8")
        result = extract(path, "txt")
        assert "RAG system" in result.text
        assert result.method

    def test_rejects_unknown_extension(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.rtf"
        path.write_text("{}", "utf-8")
        with pytest.raises(Exception):
            extract(path, "rtf")

    def test_process_file_reports_counts(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.txt"
        path.write_text("word " * 400, "utf-8")
        result = process_file(path, "txt")
        assert result.chunks
        assert result.word_count >= 400
        assert set(result.timing_ms) == {"extraction", "cleaning", "chunking"}

    def test_empty_file_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.txt"
        path.write_bytes(b"")
        with pytest.raises(Exception):
            process_file(path, "txt")

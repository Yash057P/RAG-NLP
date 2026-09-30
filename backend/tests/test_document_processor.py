"""Unit tests for the text processing pipeline (steps 1-3)."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from backend.core.config import settings
from backend.services.document_processor import (
    _apply_overlap,
    _detect_column_gutter,
    _page_lookup,
    chunk_text,
    extract,
    process_file,
    Segment,
)
from backend.utils.files import title_context
from backend.utils.text import clean_text, find_reference_start, term_coverage, tokenize


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


class TestColumnAwarePdf:
    """Two-column pages must be read column by column, not line by line."""

    @staticmethod
    def _two_column_pdf(tmp_path: Path) -> Path:
        """Build a PDF whose page has two text columns separated by a gutter."""
        pytest.importorskip("pdfplumber")
        from reportlab.pdfgen import canvas as canvas_module

        path = tmp_path / "two_column.pdf"
        canvas = canvas_module.Canvas(str(path), pagesize=(612, 792))
        text_object = canvas.beginText(40, 700)
        text_object.setFont("Helvetica", 9)
        # 40 lines down the left column, then one line at the top of the right.
        # The columns must not collide in the x range 299-312.
        for row in range(40):
            text_object.setTextOrigin(40, 700 - row * 14)
            text_object.textLine(f"left column body line {row:02d} of the discussion.")
        text_object.setTextOrigin(320, 700)
        text_object.textLine("right column body line of a separate argument here.")
        canvas.drawText(text_object)
        canvas.showPage()
        canvas.save()
        return path

    def test_gutter_is_detected(self, tmp_path: Path) -> None:
        pdfplumber = pytest.importorskip("pdfplumber")
        path = self._two_column_pdf(tmp_path)
        with pdfplumber.open(path) as pdf:
            page = pdf.pages[0]
            gutter = _detect_column_gutter(page)
            # The left column's text ends before 320, where the right one starts,
            # so the gutter has to be found somewhere inside that empty band.
            right_edges = [w["x1"] for w in page.extract_words() if w["x0"] < 300]
            left_edge = max(right_edges)

        assert gutter is not None
        start, end = gutter
        # The x-coverage histogram is integer-binned, so the gap can begin one
        # pixel before the last glyph ends.
        assert left_edge - 1 <= start <= 320
        assert start < end <= float(page.width)
        # And the gap must be roughly centred to count as a column gutter.
        centre = (start + end) / 2 / float(page.width)
        assert 0.40 <= centre <= 0.62

    def test_single_column_page_reports_no_gutter(self, tmp_path: Path) -> None:
        pytest.importorskip("pdfplumber")
        from reportlab.pdfgen import canvas as canvas_module

        path = tmp_path / "single.pdf"
        canvas = canvas_module.Canvas(str(path), pagesize=(612, 792))
        text_object = canvas.beginText(40, 700)
        text_object.setFont("Helvetica", 9)
        for row in range(40):
            text_object.textLine(f"a full width line of prose number {row:02d} here.")
        canvas.drawText(text_object)
        canvas.showPage()
        canvas.save()

        result = extract(path, "pdf")
        assert result.text.strip()
        # Nothing to split, so the text must not have been cropped away.
        assert "a full width line of prose number 00 here." in " ".join(result.text.split())

    def test_columns_are_read_in_order_not_interleaved(self, tmp_path: Path) -> None:
        path = self._two_column_pdf(tmp_path)
        text = " ".join(extract(path, "pdf").text.split())
        # With a plain extract_text() the two columns alternate line by line.
        assert text.index("left column body line 00") < text.index(
            "left column body line 01"
        )
        # The whole left column is read before the right one starts.
        assert text.index("left column body line 39") < text.index("right column body line")


class TestBibliographyDetection:
    # The body must vary paragraph by paragraph: repeating identical lines would
    # trip the page-furniture detector, which strips any line seen on four or
    # more pages - including the "References" heading itself.
    @staticmethod
    def _body(paragraphs: int) -> str:
        return "".join(
            f"2.{n} Related Work\n"
            f"Sycophancy in setting {n} refers to responses that move toward a "
            f"user's stated beliefs rather than the available evidence [1].\n"
            for n in range(paragraphs)
        )

    BIBLIOGRAPHY = (
        "References\n"
        "[1] M. Sharma and T. Tong, \"Towards Understanding Sycophancy in\n"
        "Language Models,\" arXiv preprint arXiv:2310.13548, 2023.\n"
        "[2] J. Hong and G. Byun, \"Measuring Sycophancy of Language Models in\n"
        "Multi-turn Dialogues,\" EMNLP 2025, pp. 2239-2259, 2025.\n"
        "[3] A. Fanous, J. Goldberg, and S. Lin, \"SycEval: Evaluating LLM\n"
        "Sycophancy,\" Proceedings of the AAAI Conference on AI, 2024.\n"
        "[4] P. Heydari and A. Sicilia, \"Auditing Assistant Compliance,\" in\n"
        "Proceedings of the Conference on AI Ethics, 2024.\n"
        "[5] R. Rafailov et al., \"Direct Preference Optimization,\" NeurIPS\n"
        "2023, vol. 36.\n"
    )

    def test_finds_the_references_heading(self) -> None:
        text = self._body(12) + self.BIBLIOGRAPHY
        offset = find_reference_start(clean_text(text))
        assert offset is not None
        assert "References" in text[max(0, offset - 12) : offset + 12]

    def test_ignores_a_document_without_a_bibliography(self) -> None:
        assert find_reference_start(clean_text(self._body(12))) is None

    def test_ignores_a_bare_mention_of_references(self) -> None:
        text = (
            "As noted earlier, the references given in Appendix A cover this "
            "topic in full detail across several competing systems and settings.\n"
            + self._body(12)
        )
        assert find_reference_start(clean_text(text)) is None

    def test_ignores_a_heading_not_followed_by_citations(self) -> None:
        # "References" as a section title, but the body below it is prose.
        text = self._body(12) + "References\n" + ("This section restates the findings. " * 20)
        assert find_reference_start(clean_text(text)) is None

    def test_chunks_after_the_heading_are_flagged(self, tmp_path: Path) -> None:
        text = self._body(20) + self.BIBLIOGRAPHY
        path = tmp_path / "paper.txt"
        path.write_text(text, "utf-8")

        result = process_file(path, "txt")
        assert result.chunks
        assert any(chunk.is_reference for chunk in result.chunks), "no chunk was flagged"
        # Body chunks must survive, otherwise the flag is too greedy.
        body = " ".join(
            " ".join(c.text.split()) for c in result.chunks if not c.is_reference
        )
        assert "Sycophancy in setting 0 refers to responses" in body

    def test_flagged_chunks_carry_the_metadata(self, tmp_path: Path) -> None:
        path = tmp_path / "paper2.txt"
        path.write_text(self._body(20) + self.BIBLIOGRAPHY, "utf-8")
        chunks = process_file(path, "txt").chunks
        flagged = [c for c in chunks if c.is_reference]
        assert flagged
        assert all(c.metadata()["is_reference"] for c in flagged)


class TestTitleContext:
    """The filename is searchable so "what is dav?" finds Experiment_5_DAV.docx."""

    def test_underscores_become_spaces(self) -> None:
        assert title_context("Experiment_5_DAV.docx") == "Experiment 5 DAV Experiment 5 DAV"

    def test_camel_case_is_split(self) -> None:
        assert title_context("PaperNLPClassifier.pdf").split()[:3] == [
            "Paper",
            "NLP",
            "Classifier",
        ]

    def test_extension_is_dropped(self) -> None:
        assert ".pdf" not in title_context("report.pdf")

    def test_empty_stem_is_safe(self) -> None:
        assert title_context("___.pdf") == "___"



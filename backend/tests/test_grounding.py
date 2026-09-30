"""Unit tests for the grounding gate and the extractive baseline.

These cover the three false negatives that made the system abstain on questions
it could actually answer. They are unit tests rather than API tests on purpose:
the answerability of a question depends on how well the embedder ranks, and the
offline suite runs a hashed embedder that has no semantic understanding. The
behaviour under test here is deterministic given the retrieved chunks.
"""

from __future__ import annotations

import pytest

from backend.core.config import settings
from backend.models.schemas import RetrievedChunk
from backend.prompts.templates import NO_ANSWER_MESSAGE
from backend.services.llm_factory import _HEADING_LINE, ExtractiveLLM
from backend.services.rag_service import _to_retrieved, build_context, coverage_text
from backend.services.vector_store import ScoredChunk
from backend.utils.files import title_context
from backend.utils.text import content_words, find_reference_start, term_coverage


def _chunk(
    text: str,
    *,
    filename: str = "doc.txt",
    chunk_index: int = 0,
    score: float = 0.5,
    page: int | None = None,
    is_reference: bool = False,
) -> RetrievedChunk:
    return RetrievedChunk(
        id=f"doc_1::{chunk_index}",
        document_id="doc_1",
        filename=filename,
        chunk_index=chunk_index,
        text=text,
        snippet=text[:100],
        score=score,
        distance=1.0 - score,
        page=page,
        is_reference=is_reference,
    )


def _ask(llm: ExtractiveLLM, context: str, question: str) -> str:
    """Drive the baseline exactly as the router does."""
    user_prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    return llm.generate("", user_prompt)


class TestConversationalFiller:
    """Request verbs carry no topic, so they must not dilute coverage."""

    def test_filler_words_are_treated_as_stopwords(self) -> None:
        assert content_words("tell me about mathematics") == {"mathematics"}

    def test_verbose_and_terse_questions_score_the_same(self) -> None:
        context = "The questions cover eight domains including basic mathematics."
        verbose = term_coverage("tell me about mathematics", context)
        terse = term_coverage("mathematics", context)
        assert verbose == terse == 1.0

    def test_genuine_topic_terms_are_kept(self) -> None:
        assert "mathematics" in content_words("tell me about mathematics")
        assert "outliers" in content_words("how are outliers handled")


class TestFilenameIsSearchable:
    """An acronym that only appears in the filename must still match."""

    def test_coverage_text_includes_the_filename(self) -> None:
        text = coverage_text([_chunk("Data cleaning fixes structural problems.")])
        assert "unknown" not in text
        chunks = [_chunk("Body text here.", filename="Experiment_5_DAV.docx")]
        assert "Experiment_5_DAV.docx" in coverage_text(chunks)

    def test_coverage_text_includes_the_page(self) -> None:
        text = coverage_text([_chunk("Body.", filename="a.pdf", page=4)])
        assert "4" in text

    def test_filename_only_question_is_covered(self) -> None:
        """The regression: "what is dav?" scored 0% against chunk bodies."""
        chunks = [
            _chunk("Data cleaning fixes structural problems.", filename="Experiment_5_DAV.docx")
        ]
        coverage = term_coverage("what is dav?", coverage_text(chunks))
        assert coverage == 1.0
        assert coverage >= settings.GROUNDING_THRESHOLD

    def test_baseline_matches_a_question_against_the_filename(self) -> None:
        """The gate allows it, so the generator must not then contradict it."""
        chunk = _chunk(
            "Data cleaning fixes structural problems such as wrong types and "
            "inconsistent formats across the raw dataset.",
            filename="Experiment_5_DAV.docx",
        )
        answer = _ask(ExtractiveLLM(), build_context([chunk]), "what is dav?")
        assert answer != NO_ANSWER_MESSAGE
        # And it only ever quotes the document, never invents an expansion.
        assert "Data cleaning fixes structural problems" in answer

    def test_title_context_splits_the_acronym(self) -> None:
        assert "DAV" in title_context("Experiment_5_DAV.docx").split()


class TestBibliographyHandling:
    """Citation lists match topical queries but answer nothing."""

    BIB_TEXT = (
        "[1] M. Sharma and T. Tong, \"Towards Understanding Sycophancy in "
        "Language Models,\" arXiv preprint arXiv:2310.13548, 2023."
    )
    PROSE = (
        "Sycophancy in large language models refers to responses that move "
        "toward a user's beliefs instead of relying on the available evidence."
    )

    def test_reference_scores_are_demoted(self) -> None:
        raw = ScoredChunk(
            id="doc_1::9",
            text=self.BIB_TEXT,
            score=0.60,
            distance=0.40,
            metadata={"document_id": "doc_1", "filename": "p.pdf", "chunk_index": 9,
                      "is_reference": True},
        )
        retrieved = _to_retrieved(raw)
        assert retrieved.is_reference is True
        assert retrieved.score == round(0.60 * settings.REFERENCE_SCORE_PENALTY, 4)
        assert retrieved.score < 0.60

    def test_prose_scores_are_untouched(self) -> None:
        raw = ScoredChunk(
            id="doc_1::2",
            text=self.PROSE,
            score=0.42,
            distance=0.58,
            metadata={"document_id": "doc_1", "filename": "p.pdf", "chunk_index": 2},
        )
        retrieved = _to_retrieved(raw)
        assert retrieved.is_reference is False
        assert retrieved.score == 0.42

    def test_context_block_labels_bibliography_sources(self) -> None:
        """A real LLM needs the same signal the baseline uses."""
        context = build_context(
            [
                _chunk(self.PROSE, filename="p.pdf", chunk_index=2, page=3),
                _chunk(self.BIB_TEXT, filename="p.pdf", chunk_index=9, is_reference=True),
            ]
        )
        assert "| page 3" in context
        assert "| bibliography" in context
        # The label is on the header line, not mixed into the prose.
        header = context.splitlines()[0]
        assert header.startswith("[Source 1] file: p.pdf | chunk 2")

    def test_baseline_never_quotes_a_bibliography_entry(self) -> None:
        context = build_context(
            [_chunk(self.BIB_TEXT, filename="p.pdf", chunk_index=9, is_reference=True)]
        )
        answer = _ask(ExtractiveLLM(), context, "what does the paper say about sycophancy?")
        assert answer == NO_ANSWER_MESSAGE
        assert "arXiv" not in answer

    def test_baseline_prefers_prose_when_both_are_retrieved(self) -> None:
        context = build_context(
            [
                _chunk(self.BIB_TEXT, filename="p.pdf", chunk_index=9, is_reference=True),
                _chunk(self.PROSE, filename="p.pdf", chunk_index=2),
            ]
        )
        answer = _ask(ExtractiveLLM(), context, "define sycophancy")
        assert "refers to responses that move toward" in answer
        assert "arXiv" not in answer

    def test_detector_ignores_a_prose_mention(self) -> None:
        text = (
            "We follow the references given in Appendix A, which list every "
            "system evaluated during the study in full detail.\n" * 2
        )
        assert find_reference_start(text) is None


class TestSourceLabels:
    def test_sentences_are_not_merged_across_sources(self) -> None:
        """Two chunks from different files must not become one sentence."""
        context = build_context(
            [
                _chunk("The first document ends mid-thought here", filename="a.txt"),
                _chunk("and the second document starts elsewhere", filename="b.txt"),
            ]
        )
        sentences = ExtractiveLLM._context_sentences(context)
        assert len(sentences) == 2
        assert {label for _, label in sentences} == {"a.txt", "b.txt"}

    def test_headings_are_not_quoted_as_prose(self) -> None:
        context = build_context(
            [_chunk("2 Related Work\nSycophancy refers to a measurable drift.", filename="p.pdf")]
        )
        # Asked about the prose, not the heading: "related work" only ever
        # appeared in the heading, which is now correctly dropped.
        answer = _ask(ExtractiveLLM(), context, "what is the measurable drift?")
        assert "2 Related Work" not in answer
        assert "measurable drift" in answer

    @pytest.mark.parametrize(
        "heading",
        ["2 Related Work", "2. Related Work", "2.1 Sycophancy in LLMs", "4.1.2 Setup"],
    )
    def test_numbered_heading_forms_are_recognised(self, heading: str) -> None:
        assert _HEADING_LINE.fullmatch(heading), heading

    @pytest.mark.parametrize(
        "prose",
        [
            "Sycophancy refers to a measurable drift.",
            "We then compute the drift across every turn of the conversation",
        ],
    )
    def test_prose_is_not_mistaken_for_a_heading(self, prose: str) -> None:
        assert not _HEADING_LINE.fullmatch(prose), prose

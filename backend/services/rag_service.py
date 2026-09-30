"""The RAG engine: retrieval, grounded prompting and answer generation.

Flow for every question:

1. embed the question with the same model used for the corpus
2. similarity search the vector store for the top-K chunks
3. build a context block with per-source citations
4. verify the context actually *covers* the question (grounding gate)
5. ask the LLM to answer strictly from that context
6. score confidence and return the answer with its citations
"""

from __future__ import annotations

import time

from backend.core.config import settings
from backend.core.logging import get_logger
from backend.models.schemas import (
    AnswerResponse,
    HistoryEntry,
    RetrievalResult,
    RetrievedChunk,
    Source,
)
from backend.prompts.templates import (
    CONTEXT_BLOCK_TEMPLATE,
    EMPTY_CORPUS_MESSAGE,
    NO_ANSWER_MESSAGE,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)
from backend.services.analytics import get_analytics
from backend.services.embedding_service import get_embedder
from backend.services.llm_factory import get_llm_router
from backend.services.pipeline import query_trace
from backend.services.registry import get_registry
from backend.services.vector_store import ScoredChunk, get_vector_store
from backend.utils.files import generate_id, truncate
from backend.utils.text import term_coverage

logger = get_logger(__name__)

# Cosine similarity at which the retrieval component saturates.
SIMILARITY_CEILING = 0.75
# Score gap between the best chunk and the mean of the rest.
MARGIN_CEILING = 0.25
MAX_SOURCES = 3
SNIPPET_CHARS = 320


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def estimate_tokens(text: str) -> int:
    """Rough token count for GPT-style tokenisers (~4 characters per token)."""
    return max(1, len(text) // 4)


def _to_retrieved(chunk: ScoredChunk) -> RetrievedChunk:
    metadata = chunk.metadata
    page = metadata.get("page")
    is_reference = bool(metadata.get("is_reference", False))
    # Bibliography entries are lexically dense with topic terms, so they crowd
    # out the prose that actually answers the question. Scaling their score
    # pushes them down the ranking while keeping them available as context, and
    # the flag lets the UI label them.
    score = float(chunk.score)
    if is_reference:
        score *= settings.REFERENCE_SCORE_PENALTY
    return RetrievedChunk(
        id=chunk.id or f"{metadata.get('document_id', '')}::{metadata.get('chunk_index', 0)}",
        document_id=str(metadata.get("document_id", "")),
        filename=str(metadata.get("filename", "unknown")),
        chunk_index=int(metadata.get("chunk_index", 0)),
        text=chunk.text,
        snippet=truncate(" ".join(chunk.text.split()), 260),
        score=round(score, 4),
        distance=round(float(chunk.distance), 4),
        page=int(page) if isinstance(page, int) else None,
        char_start=metadata.get("char_start"),
        char_end=metadata.get("char_end"),
        token_estimate=estimate_tokens(chunk.text),
        is_reference=is_reference,
    )


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Render the retrieved chunks as a numbered, citable context block.

    Bibliography entries are labelled so the model knows they are citation
    strings rather than prose. Without the label a model will happily quote a
    reference as though it were the answer, because a paper's own title repeats
    the topic terms the user asked about.
    """
    blocks: list[str] = []
    for position, chunk in enumerate(chunks, start=1):
        flags = f" | page {chunk.page}" if chunk.page else ""
        if chunk.is_reference:
            flags = f"{flags} | bibliography" if flags else " | bibliography"
        blocks.append(
            CONTEXT_BLOCK_TEMPLATE.format(
                index=position,
                filename=chunk.filename,
                chunk_index=chunk.chunk_index,
                flags=flags,
                text=chunk.text.strip(),
            )
        )
    return "\n\n".join(blocks)


def coverage_text(chunks: list[RetrievedChunk]) -> str:
    """Text the grounding gate measures against.

    This is the text the model actually reads, so it must include the source
    labels - a question like "what is dav?" is legitimately answered by a
    document named ``Experiment_5_DAV.docx``, and the context block already
    shows the model that filename. Measuring coverage on chunk bodies alone
    would reject the very match retrieval just found.
    """
    return " ".join(
        f"{chunk.filename} {chunk.page if chunk.page else ''} {chunk.text}".strip()
        for chunk in chunks
    )


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> tuple[str, str]:
    """Return ``(system_prompt, user_prompt)`` for the LLM."""
    context = build_context(chunks)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        context=context, question=question.strip()
    )
    return SYSTEM_PROMPT.format(context=context, question=question.strip()), user_prompt


def score_confidence(
    question: str, chunks: list[RetrievedChunk]
) -> tuple[float, dict[str, float]]:
    """Heuristic confidence in [0, 1] plus its components.

    Three signals are blended: how similar the best chunk is, how much of the
    question the context actually covers, and how far the best chunk stands out
    from the rest.  It is a retrieval-quality proxy, not a calibrated
    probability - the UI labels it as such.
    """
    if not chunks:
        return 0.0, {"similarity": 0.0, "coverage": 0.0, "margin": 0.0}

    top_score = chunks[0].score
    rest = [chunk.score for chunk in chunks[1:]]
    margin = top_score - (sum(rest) / len(rest) if rest else 0.0)

    context_text = coverage_text(chunks)
    coverage = term_coverage(question, context_text)

    components = {
        "similarity": round(_clamp01(top_score / SIMILARITY_CEILING), 4),
        "coverage": round(_clamp01(coverage), 4),
        "margin": round(_clamp01(margin / MARGIN_CEILING), 4),
    }
    confidence = (
        0.45 * components["similarity"]
        + 0.35 * components["coverage"]
        + 0.20 * components["margin"]
    )
    return round(_clamp01(confidence), 4), components


def confidence_label(confidence: float, grounded: bool) -> str:
    if not grounded:
        return "Not grounded in documents"
    if confidence >= 0.70:
        return "High confidence"
    if confidence >= 0.45:
        return "Moderate confidence"
    if confidence >= 0.25:
        return "Low confidence"
    return "Very low confidence"


def build_sources(chunks: list[RetrievedChunk]) -> list[Source]:
    """Deduplicated citations for the answer, best score first."""
    seen: set[tuple[str, int]] = set()
    sources: list[Source] = []
    for chunk in chunks:
        key = (chunk.document_id, chunk.chunk_index)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            Source(
                document_id=chunk.document_id,
                filename=chunk.filename,
                chunk_index=chunk.chunk_index,
                page=chunk.page,
                snippet=truncate(" ".join(chunk.text.split()), SNIPPET_CHARS),
                score=chunk.score,
            )
        )
        if len(sources) >= MAX_SOURCES:
            break
    return sources


class RAGService:
    """Stateless façade over retrieval + generation."""

    def __init__(self) -> None:
        self.store = get_vector_store()
        self.embedder = get_embedder()
        self.llm = get_llm_router()
        self.registry = get_registry()
        self.analytics = get_analytics()

    # -- retrieval ----------------------------------------------------------
    def retrieve(self, question: str, top_k: int | None = None) -> RetrievalResult:
        top_k = top_k or settings.TOP_K
        started = time.perf_counter()

        started_embed = time.perf_counter()
        query_vector = self.embedder.embed_query(question)
        embedding_ms = (time.perf_counter() - started_embed) * 1000

        started_search = time.perf_counter()
        # Over-fetch, because demoting a bibliography chunk should let the prose
        # that the store ranked just below the cut take its place. Trimming to
        # top_k afterwards keeps the returned set the same size either way.
        candidates = top_k * settings.RETRIEVAL_CANDIDATE_MULTIPLIER
        raw_chunks = self.store.query(query_vector, candidates)
        search_ms = (time.perf_counter() - started_search) * 1000

        chunks = [_to_retrieved(chunk) for chunk in raw_chunks]
        chunks = [chunk for chunk in chunks if chunk.score >= settings.MIN_SIMILARITY]
        chunks.sort(key=lambda chunk: chunk.score, reverse=True)
        chunks = chunks[:top_k]
        total_ms = (time.perf_counter() - started) * 1000

        scores = [chunk.score for chunk in chunks]
        return RetrievalResult(
            query=question,
            chunks=chunks,
            top_k=top_k,
            total_candidates=len(raw_chunks),
            top_score=round(max(scores), 4) if scores else 0.0,
            average_score=round(sum(scores) / len(scores), 4) if scores else 0.0,
            grounding_coverage=0.0,
            embedding_ms=round(embedding_ms, 2),
            search_ms=round(search_ms, 2),
            total_ms=round(total_ms, 2),
            embedding_model=self.embedder.name,
            vector_store_backend=self.store.backend,
        )

    # -- question answering -------------------------------------------------
    def answer(self, question: str, top_k: int | None = None) -> AnswerResponse:
        started = time.perf_counter()
        question = question.strip()
        answer_id = generate_id("qa_")

        retrieval = self.retrieve(question, top_k)
        context_text = coverage_text(retrieval.chunks)
        coverage = term_coverage(question, context_text) if context_text else 0.0
        retrieval.grounding_coverage = round(coverage, 4)

        confidence, components = score_confidence(question, retrieval.chunks)
        corpus_size = self.store.count()

        # -- grounding gate ------------------------------------------------
        abstained = False
        answer_text: str
        llm_model = "not-called"
        generation_ms = 0.0

        if corpus_size == 0:
            abstained = True
            answer_text = EMPTY_CORPUS_MESSAGE
            llm_model = "not-called (empty corpus)"
        elif not retrieval.chunks:
            abstained = True
            answer_text = NO_ANSWER_MESSAGE
            llm_model = "not-called (no relevant chunks)"
        elif coverage < settings.GROUNDING_THRESHOLD:
            abstained = True
            answer_text = NO_ANSWER_MESSAGE
            llm_model = "not-called (context did not cover the question)"
        else:
            system_prompt, user_prompt = build_prompt(question, retrieval.chunks)
            started_generation = time.perf_counter()
            answer_text, llm_model, _ = self.llm.generate(system_prompt, user_prompt)
            generation_ms = (time.perf_counter() - started_generation) * 1000
            answer_text = answer_text.strip() or NO_ANSWER_MESSAGE
            abstained = NO_ANSWER_MESSAGE.lower() in answer_text.lower()

        grounded = not abstained
        if abstained:
            # Retrieval found something, but it does not support an answer.
            confidence = round(confidence * 0.5, 4)

        response_time_ms = (time.perf_counter() - started) * 1000
        sources = build_sources(retrieval.chunks) if grounded else []

        _, user_prompt_preview = (
            build_prompt(question, retrieval.chunks) if retrieval.chunks else ("", "")
        )

        response = AnswerResponse(
            id=answer_id,
            question=question,
            answer=answer_text,
            grounded=grounded,
            abstained=abstained,
            confidence=confidence,
            confidence_label=confidence_label(confidence, grounded),
            confidence_breakdown=components,
            sources=sources,
            retrieved_chunks=retrieval.chunks,
            retrieval=retrieval,
            llm={
                "provider": llm_model.split(":", 1)[0] if ":" in llm_model else "none",
                "model": llm_model,
                "available": not llm_model.startswith("not-called"),
                "fallback_used": "fallback" in llm_model,
                "detail": self.llm.info().get("detail"),
            },
            prompt=truncate(user_prompt_preview, 4000),
            pipeline=query_trace(
                total_documents=len(self.registry.list()),
                total_chunks=corpus_size,
                candidates=retrieval.total_candidates,
                returned=len(retrieval.chunks),
                top_score=retrieval.top_score,
                retrieval_ms=retrieval.total_ms,
                embedding_ms=retrieval.embedding_ms,
                search_ms=retrieval.search_ms,
                generation_ms=round(generation_ms, 2),
                llm_model=llm_model,
                grounded=grounded,
                total_ms=response_time_ms,
            ),
            response_time_ms=round(response_time_ms, 2),
        )

        self.analytics.record(
            HistoryEntry(
                id=answer_id,
                question=question,
                answer=answer_text,
                confidence=confidence,
                confidence_label=response.confidence_label,
                sources=sources,
                response_time_ms=response.response_time_ms,
                created_at=response.created_at,
                llm_model=llm_model,
                top_score=retrieval.top_score,
            )
        )
        logger.info(
            "Q&A %s grounded=%s confidence=%.2f in %.0fms (%d chunks)",
            answer_id,
            grounded,
            confidence,
            response_time_ms,
            len(retrieval.chunks),
        )
        return response


_rag_service: RAGService | None = None


def get_rag_service() -> RAGService:
    """Cached service instance (the singletons it wraps are themselves cached)."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


def reset_rag_service() -> None:
    global _rag_service
    _rag_service = None

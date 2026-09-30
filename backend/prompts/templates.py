"""RAG prompt templates and the canned questions shown in the UI."""

from __future__ import annotations

NO_ANSWER_MESSAGE = (
    "The uploaded documents do not contain enough information to answer this "
    "question."
)

SYSTEM_PROMPT = """You are a document assistant.

Answer ONLY from the provided context.
If the answer is not found in the context, reply:
"The uploaded documents do not contain enough information to answer this question."

Context:
{context}

Question:
{question}

Answer:"""

CONTEXT_BLOCK_TEMPLATE = """[Source {index}] file: {filename} | chunk {chunk_index}{flags}
{text}"""

USER_PROMPT_TEMPLATE = """You are a document assistant.

Answer ONLY from the provided context.
If the answer is not found in the context, reply:
"The uploaded documents do not contain enough information to answer this question."

Context:
{context}

Question:
{question}

Answer:"""

# Emitted when retrieval returns nothing at all.
EMPTY_CORPUS_MESSAGE = (
    "No documents have been uploaded yet, so there is nothing to search. "
    "Upload a PDF, DOCX or TXT file and ask your question again."
)

# Emitted when the corpus exists but retrieval found nothing relevant.
NO_MATCH_MESSAGE = NO_ANSWER_MESSAGE

EXAMPLE_QUESTIONS = [
    "What is the objective of the experiment?",
    "What are the advantages of RAG?",
    "How does the chunking strategy work?",
    "Which embedding model is used and why?",
    "What vector database is used for storage?",
    "How is the final answer generated from the retrieved context?",
    "What are the limitations of this system?",
    "Summarize the key takeaways of the document.",
]

GITHUB_README_QUESTIONS = EXAMPLE_QUESTIONS

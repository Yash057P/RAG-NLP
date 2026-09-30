"""Text utilities used by the document processing pipeline.

These functions implement step 2 of the pipeline ("Clean text") and are also
reused by the RAG service to measure how well the retrieved context actually
covers the user's question.
"""

from __future__ import annotations

import re
import unicodedata

# Ligatures and typographic characters that break naive tokenisation.
_CHAR_FIXES = {
    "\u00a0": " ",
    "\u2007": " ",
    "\u202f": " ",
    "\u200b": "",
    "\ufeff": "",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
    "\u00ad": "",
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
}

_WHITESPACE_RUN = re.compile(r"[ \t\f\v]+")
_NEWLINE_RUN = re.compile(r"\n{3,}")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Word broken across a line by hyphenation, e.g. "informa-\ntion".
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
# Page furniture such as "Page 3 of 12" or "--- 12 ---".
_PAGE_MARKER = re.compile(r"^\s*(?:page\s+)?\d+\s*(?:of|/)\s*\d+\s*$", re.IGNORECASE)
_RULE_LINE = re.compile(r"^\s*[-=_*~]{3,}\s*$")
_CONTROLLED_VOCAB = {
    # Closed-class function words.
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "do", "does", "for",
    "from", "had", "has", "have", "how", "in", "into", "is", "it", "its", "of",
    "on", "or", "that", "the", "their", "then", "there", "these", "this", "to",
    "was", "were", "what", "when", "where", "which", "who", "why", "will",
    "with", "you", "your",
    # Conversational filler and request verbs. These carry no topic information,
    # so leaving them in the denominator penalises natural phrasings: "tell me
    # about mathematics" has one real topic term, but scored as four.
    "about", "also", "am", "any", "briefly", "brief", "can", "could", "could",
    "detail", "details", "describe", "did", "do", "explain", "explaination",
    "give", "hello", "hey", "hi", "i", "if", "im", "info", "information", "is",
    "kind", "know", "like", "looking", "may", "me", "mine", "much", "my",
    "need", "of", "please", "question", "regarding", "relate", "related",
    "say", "should", "simple", "so", "some", "something", "summarise",
    "summarize", "summary", "tell", "thanks", "thank", "understand", "us",
    "want", "well", "would",
}

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def fix_characters(text: str) -> str:
    """Normalise unicode punctuation, ligatures and control characters."""
    for bad, good in _CHAR_FIXES.items():
        text = text.replace(bad, good)
    text = _CONTROL_CHARS.sub("", text)
    return text


def dehyphenate(text: str) -> str:
    """Join words that were hyphenated across a line break in the source PDF."""
    return _HYPHEN_BREAK.sub(r"\1\2", text)


def collapse_whitespace(text: str) -> str:
    """Collapse horizontal runs of spaces and limit blank lines to two."""
    text = _WHITESPACE_RUN.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _NEWLINE_RUN.sub("\n\n", text)


def detect_boilerplate(text: str, min_occurrences: int = 4) -> set[str]:
    """Identify page headers/footers so they can be stripped per segment.

    Two rules are applied: a short line repeated many times is treated as a
    running header, and a document containing several distinct ``Page N of M``
    markers is treated as paged so all of them are stripped. Detection runs
    over the *whole* document, which lets each page be cleaned independently
    afterwards without losing track of where its text ends up.
    """
    lines = text.split("\n")
    if len(lines) < min_occurrences:
        return set()

    counts: dict[str, int] = {}
    for line in lines:
        stripped = line.strip()
        if stripped:
            counts[stripped] = counts.get(stripped, 0) + 1

    threshold = max(min_occurrences, int(len(lines) * 0.05))

    boilerplate = {
        line
        for line, count in counts.items()
        if count >= threshold and (len(line) < 90 or _RULE_LINE.match(line))
    }

    # Distinct page markers (e.g. "Page 3 of 12" on every page) never repeat
    # verbatim, so they are detected by their shape and overall count.
    page_markers = [line for line in counts if _PAGE_MARKER.match(line)]
    if len(page_markers) >= 3:
        boilerplate.update(page_markers)

    return boilerplate


def drop_repeated_lines(text: str, min_occurrences: int = 4) -> str:
    """Remove page headers/footers that repeat across the document."""
    boilerplate = detect_boilerplate(text, min_occurrences)
    if not boilerplate:
        return text
    kept = [
        line for line in text.split("\n") if line.strip() not in boilerplate or not line.strip()
    ]
    return "\n".join(kept)


# A bibliography is introduced by a bare "References" / "Bibliography" heading.
# Matching the heading on its own line avoids firing on sentences that merely
# mention the word, e.g. "we follow the references given in Appendix A".
_REFERENCE_HEADING = re.compile(
    r"^[ \t]*(?:\d+(?:\.\d+)*[.)]?[ \t]*)?"
    r"(references?|bibliography|works cited|literature cited)"
    r"[ \t]*:?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
# An entry opens with its numeric marker: "[12] A. Author, ...". Some styles drop
# the bracket, so a leading "arXiv:" id or a DOI also counts.
_CITATION_ENTRY = re.compile(r"^(?:\[\d+\]|arXiv:|https?://|doi:)", re.IGNORECASE)
# Share of lines after the heading that must look like citation entries. The
# entries of a wrapped, two-column bibliography span several lines each, so
# measured against a real paper the body scores 0.00 and the bibliography 0.22;
# 0.15 sits in that gap with room on both sides.
_CITATION_LIST_RATIO = 0.15


def find_reference_start(text: str) -> int | None:
    """Offset where a document's bibliography begins, or ``None``.

    Everything from this offset on is a list of citations. Those entries are
    lexically dense with the very terms a user asks about - a paper's title
    repeats its own topic words - so a similarity search ranks them above the
    prose that actually answers the question.

    The heading is only trusted when the text below it looks like a citation
    list, which keeps a stray "References" line in the middle of a document from
    hiding its real content.
    """
    matches = list(_REFERENCE_HEADING.finditer(text))
    if not matches:
        return None

    # Take the last plausible heading: some papers mention references early on.
    for match in reversed(matches):
        if _looks_like_citation_list(text[match.end() :]):
            return match.start()
    return None


def _looks_like_citation_list(text: str, sample_lines: int = 40) -> bool:
    """True when enough of the first lines of ``text`` are citation entries."""
    lines = [line for line in text.split("\n") if line.strip()][:sample_lines]
    if len(lines) < 3:
        return False
    entries = sum(1 for line in lines if _CITATION_ENTRY.match(line.strip()))
    return entries / len(lines) >= _CITATION_LIST_RATIO


def clean_text(text: str, boilerplate: set[str] | None = None) -> str:
    """Full cleaning pass applied to every extracted document.

    Order matters: characters are fixed first so that the whitespace regexes
    see normal spaces, de-hyphenation runs before whitespace collapsing, and
    duplicate page furniture is removed last.

    ``boilerplate`` lets a caller supply line numbers detected over the whole
    document, which is what keeps per-page cleaning consistent with the page
    offsets recorded for citation.
    """
    if not text:
        return ""

    text = fix_characters(text)
    text = dehyphenate(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = collapse_whitespace(text)

    furniture = detect_boilerplate(text) if boilerplate is None else boilerplate
    if furniture:
        text = "\n".join(
            line
            for line in text.split("\n")
            if line.strip() not in furniture or not line.strip()
        )
        text = collapse_whitespace(text)

    return text.strip()


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens (keeps internal apostrophes)."""
    return _WORD_RE.findall(text.lower())


def content_words(text: str) -> set[str]:
    """Meaningful tokens: stop-words and bare digits removed."""
    return {token for token in tokenize(text) if token not in _CONTROLLED_VOCAB}


def term_coverage(question: str, context: str) -> float:
    """Fraction of the question's content words that appear in the context.

    Used as a grounding signal: a low value means the retrieval did not
    actually surface evidence for the question.
    """
    question_terms = content_words(question)
    if not question_terms:
        return 1.0
    context_terms = content_words(context)
    hits = sum(1 for term in question_terms if term in context_terms)
    return hits / len(question_terms)


def normalize_whitespace(text: str) -> str:
    """Collapse everything to single spaces - handy for compact prompts."""
    return " ".join(text.split())

"""LLM providers for answer generation.

Three real providers are supported and selected automatically:

* **OpenAI** - ``langchain_openai.ChatOpenAI`` (raw HTTPS fallback). The base URL
  is configurable so any OpenAI-compatible gateway works.
* **Ollama** - ``langchain_ollama.ChatOllama`` (raw ``/api/chat`` fallback).
* **Mistral** - ``langchain_mistralai.ChatMistralAI`` (raw HTTPS fallback).

A deterministic :class:`ExtractiveLLM` acts as an offline baseline: it answers
by ranking the sentences of the *retrieved context* against the question. It
keeps the full RAG loop demonstrable with no API key, and the UI always labels
when it was used.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Protocol, runtime_checkable

import httpx

from backend.core.config import settings
from backend.core.logging import get_logger
from backend.prompts.templates import NO_ANSWER_MESSAGE
from backend.utils.text import content_words, tokenize

logger = get_logger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_HEADING_RULE = re.compile(r"[-=_*~]{3,}")
# A short, unterminated line such as "2. System Architecture" is a heading, not
# prose - it must not be glued onto the sentence that follows it.
_HEADING_LINE = re.compile(r"^(?:\d+[.)]\s*|[#*•-]\s*)?[A-Z][\w &/'-]{0,58}$")
# Extractive baseline tuning.
MAX_SENTENCES = 3
MAX_ANSWER_CHARS = 900


@runtime_checkable
class BaseLLM(Protocol):
    provider: str
    model: str

    def available(self) -> tuple[bool, str]: ...

    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


def _openai_messages(system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


class OpenAILLM:
    provider = "openai"

    def __init__(self) -> None:
        self.model = settings.OPENAI_MODEL
        self._chat: object | None = None
        try:
            from langchain_openai import ChatOpenAI  # type: ignore[import-untyped]

            self._chat = ChatOpenAI(
                model=self.model,
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
                timeout=settings.LLM_TIMEOUT_SECONDS,
            )
            logger.info("LLM: OpenAI via LangChain (%s)", self.model)
        except ImportError:
            logger.info("LLM: langchain_openai missing, using raw HTTPS for OpenAI")

    def available(self) -> tuple[bool, str]:
        if not settings.OPENAI_API_KEY:
            return False, "OPENAI_API_KEY is not set"
        return True, f"{self.model} @ {settings.OPENAI_BASE_URL}"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self._chat is not None:
            response = self._chat.invoke(_openai_messages(system_prompt, user_prompt))  # type: ignore[attr-defined]
            return str(getattr(response, "content", response)).strip()

        if not settings.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set")

        response = httpx.post(
            f"{settings.OPENAI_BASE_URL.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": _openai_messages(system_prompt, user_prompt),
                "temperature": settings.LLM_TEMPERATURE,
                "max_tokens": settings.LLM_MAX_TOKENS,
            },
            timeout=settings.LLM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()


class OllamaLLM:
    provider = "ollama"

    def __init__(self) -> None:
        self.model = settings.OLLAMA_MODEL
        self._chat: object | None = None
        try:
            from langchain_ollama import ChatOllama  # type: ignore[import-untyped]

            self._chat = ChatOllama(
                model=self.model,
                base_url=settings.OLLAMA_BASE_URL,
                temperature=settings.LLM_TEMPERATURE,
                num_predict=settings.LLM_MAX_TOKENS,
            )
            logger.info("LLM: Ollama via LangChain (%s)", self.model)
        except ImportError:
            logger.info("LLM: langchain_ollama missing, using raw HTTP for Ollama")

    def available(self) -> tuple[bool, str]:
        base = settings.OLLAMA_BASE_URL.rstrip("/")
        try:
            response = httpx.get(f"{base}/api/tags", timeout=3.0)
            response.raise_for_status()
            payload = response.json()
            models = [m.get("name", "") for m in payload.get("models", [])]
            if models and not any(
                self.model in name or name.split(":")[0] == self.model for name in models
            ):
                return (
                    False,
                    f"Ollama is running but '{self.model}' is not pulled "
                    f"(available: {', '.join(models[:5])})",
                )
            return True, f"{self.model} @ {base}"
        except Exception as exc:  # noqa: BLE001
            return False, f"Ollama not reachable at {base} ({type(exc).__name__})"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self._chat is not None:
            response = self._chat.invoke(_openai_messages(system_prompt, user_prompt))  # type: ignore[attr-defined]
            return str(getattr(response, "content", response)).strip()

        base = settings.OLLAMA_BASE_URL.rstrip("/")
        response = httpx.post(
            f"{base}/api/chat",
            json={
                "model": self.model,
                "messages": _openai_messages(system_prompt, user_prompt),
                "stream": False,
                "options": {
                    "temperature": settings.LLM_TEMPERATURE,
                    "num_predict": settings.LLM_MAX_TOKENS,
                },
            },
            timeout=settings.LLM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return str(response.json()["message"]["content"]).strip()


class MistralLLM:
    provider = "mistral"

    def __init__(self) -> None:
        self.model = settings.MISTRAL_MODEL
        self._chat: object | None = None
        try:
            from langchain_mistralai import ChatMistralAI  # type: ignore[import-untyped]

            self._chat = ChatMistralAI(
                model=self.model,
                api_key=settings.MISTRAL_API_KEY,
                base_url=settings.MISTRAL_BASE_URL,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
            )
            logger.info("LLM: Mistral via LangChain (%s)", self.model)
        except ImportError:
            logger.info("LLM: langchain_mistralai missing, using raw HTTPS for Mistral")

    def available(self) -> tuple[bool, str]:
        if not settings.MISTRAL_API_KEY:
            return False, "MISTRAL_API_KEY is not set"
        return True, f"{self.model} @ {settings.MISTRAL_BASE_URL}"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self._chat is not None:
            response = self._chat.invoke(_openai_messages(system_prompt, user_prompt))  # type: ignore[attr-defined]
            return str(getattr(response, "content", response)).strip()

        if not settings.MISTRAL_API_KEY:
            raise RuntimeError("MISTRAL_API_KEY is not set")

        response = httpx.post(
            f"{settings.MISTRAL_BASE_URL.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.MISTRAL_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": _openai_messages(system_prompt, user_prompt),
                "temperature": settings.LLM_TEMPERATURE,
                "max_tokens": settings.LLM_MAX_TOKENS,
            },
            timeout=settings.LLM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()


class ExtractiveLLM:
    """Offline, deterministic baseline - ranks context sentences by overlap.

    This is *not* a language model; it exists so the retrieval half of the
    pipeline can be demonstrated and graded without network access. It only
    ever rephrases text that is present in the retrieved context.
    """

    provider = "extractive"
    model = "extractive-baseline"

    def available(self) -> tuple[bool, str]:
        return True, "deterministic offline baseline (no API key required)"

    @staticmethod
    def _context_sentences(context: str) -> list[str]:
        """Pull clean sentences out of the numbered context block."""
        # Drop the "[Source n] file: x | chunk y" headers and heading rules so
        # they are not quoted back as if they were prose.
        context = re.sub(r"\[Source \d+\][^\n]*", " ", context)
        lines: list[str] = []
        for raw in context.split("\n"):
            line = raw.strip()
            if not line or _HEADING_RULE.fullmatch(line):
                continue
            # Terminate headings so the following prose starts a new sentence.
            lines.append(f"{line}." if _HEADING_LINE.match(line) else line)
        body = " ".join(lines)
        return [
            sentence.strip()
            for sentence in _SENTENCE_SPLIT.split(" ".join(body.split()))
            if len(sentence.strip()) > 25
        ]

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        context = user_prompt.split("Context:", 1)[-1].split("Question:", 1)[0]
        question = user_prompt.split("Question:", 1)[-1].split("Answer:", 1)[0].strip()

        question_terms = content_words(question)
        if not question_terms:
            return NO_ANSWER_MESSAGE

        sentences = self._context_sentences(context)
        if not sentences:
            return NO_ANSWER_MESSAGE

        # The same fact often appears in several retrieved chunks; keep one
        # representative of each distinct sentence. A longer variant that only
        # adds a leading heading ("System Architecture The objective ...") is
        # treated as the same fact, so containment - not equality - is the test.
        unique: list[str] = []
        seen: list[frozenset[str]] = []
        for sentence in sentences:
            tokens = frozenset(tokenize(sentence))
            if not tokens or any(tokens <= existing or existing <= tokens for existing in seen):
                continue
            seen.append(tokens)
            unique.append(sentence)
        sentences = unique

        scored: list[tuple[int, float, str]] = []
        for position, sentence in enumerate(sentences):
            terms = content_words(sentence)
            overlap = sum(1 for term in question_terms if term in terms)
            if not overlap:
                continue
            # Reward coverage of the question's terms and prefer denser, shorter
            # sentences; ties fall back to document order.
            density = overlap / max(len(terms), 1)
            scored.append((position, overlap + density, sentence))

        if not scored:
            return NO_ANSWER_MESSAGE

        scored.sort(key=lambda item: (-item[1], item[0]))
        best = sorted(scored[:MAX_SENTENCES], key=lambda item: item[0])
        answer = " ".join(sentence for _, _, sentence in best)
        if len(answer) <= MAX_ANSWER_CHARS:
            return answer
        # Never cut mid-sentence: keep whole sentences that fit the budget.
        parts = _SENTENCE_SPLIT.split(answer)
        trimmed: list[str] = []
        used = 0
        for part in parts:
            if used and used + len(part) + 1 > MAX_ANSWER_CHARS:
                break
            trimmed.append(part)
            used += len(part) + 1
        return " ".join(trimmed) or answer[:MAX_ANSWER_CHARS]


# ---------------------------------------------------------------- router ---
_PROVIDER_ORDER = ("openai", "ollama", "mistral")


class LLMRouter:
    """Chooses a provider once and degrades to the extractive baseline."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._llm: BaseLLM | None = None
        self._fallback_used = False
        self._detail = ""
        self._checked_at = 0.0
        self._cache_ttl = 20.0

    # -- provider selection -------------------------------------------------
    def _instantiate(self, provider: str) -> BaseLLM:
        if provider == "openai":
            return OpenAILLM()
        if provider == "ollama":
            return OllamaLLM()
        if provider == "mistral":
            return MistralLLM()
        return ExtractiveLLM()

    def _select(self) -> BaseLLM:
        configured = settings.LLM_PROVIDER
        if configured != "auto":
            llm = self._instantiate(configured)
            self._detail = llm.available()[1]
            if llm.available()[0]:
                return llm
            if configured == "extractive":
                return llm
            logger.warning(
                "Configured LLM provider '%s' unavailable (%s); using extractive baseline",
                configured,
                self._detail,
            )
            return ExtractiveLLM()

        for provider in _PROVIDER_ORDER:
            try:
                llm = self._instantiate(provider)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Provider %s failed to construct: %s", provider, exc)
                continue
            ok, detail = llm.available()
            if ok:
                self._detail = detail
                logger.info("LLM auto-selected: %s (%s)", provider, detail)
                return llm
            logger.info("LLM provider %s unavailable: %s", provider, detail)

        self._detail = "no external LLM configured - using extractive baseline"
        logger.info(self._detail)
        return ExtractiveLLM()

    def get(self) -> BaseLLM:
        with self._lock:
            if self._llm is None:
                self._llm = self._select()
                self._checked_at = time.time()
            return self._llm

    def refresh(self) -> BaseLLM:
        with self._lock:
            self._llm = None
            self._fallback_used = False
        return self.get()

    # -- generation ---------------------------------------------------------
    def generate(self, system_prompt: str, user_prompt: str) -> tuple[str, str, bool]:
        """Return ``(answer, model_label, fallback_used)``."""
        llm = self.get()
        try:
            answer = llm.generate(system_prompt, user_prompt)
            self._fallback_used = False
            return answer.strip(), f"{llm.provider}:{llm.model}", False
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM %s failed (%s); using extractive baseline", llm.provider, exc)
            baseline = ExtractiveLLM()
            answer = baseline.generate(system_prompt, user_prompt)
            self._fallback_used = True
            self._detail = f"{type(exc).__name__}: {exc}"
            return answer.strip(), "extractive:fallback", True

    def info(self) -> dict:
        llm = self.get()
        ok, detail = llm.available()
        return {
            "provider": llm.provider,
            "model": llm.model,
            "available": ok,
            "fallback_used": self._fallback_used,
            "detail": self._detail or detail,
        }

    def health(self) -> dict:
        llm = self.get()
        ok, detail = llm.available()
        return {
            "name": f"LLM ({llm.provider})",
            "status": "ok" if ok else ("degraded" if llm.provider != "extractive" else "ok"),
            "detail": f"{llm.model} - {detail}",
        }


_router: LLMRouter | None = None
_router_lock = threading.Lock()


def get_llm_router() -> LLMRouter:
    global _router
    if _router is None:
        with _router_lock:
            if _router is None:
                _router = LLMRouter()
    return _router

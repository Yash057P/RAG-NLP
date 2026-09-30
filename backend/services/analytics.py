"""Question history and response-time analytics (JSON on disk)."""

from __future__ import annotations

import json
import threading
from typing import Any

from backend.core.config import settings
from backend.core.logging import get_logger
from backend.models.schemas import HistoryEntry

logger = get_logger(__name__)


class AnalyticsService:
    """Keeps the Q&A history used by the export button and the dashboard cards."""

    def __init__(self, path: Any | None = None, max_entries: int = 500) -> None:
        self._path = path or settings.CHAT_HISTORY_PATH
        self._max_entries = max_entries
        self._lock = threading.RLock()
        self._entries: list[HistoryEntry] = []
        self._load()

    # -- persistence --------------------------------------------------------
    def _load(self) -> None:
        path = self._path
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text("utf-8"))
            self._entries = [
                HistoryEntry.model_validate(item) for item in payload.get("entries", [])
            ]
            logger.info("Loaded %d history entr(ies)", len(self._entries))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("Could not read chat history %s: %s", path, exc)

    def _flush(self) -> None:
        path = self._path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "entries": [entry.model_dump(mode="json") for entry in self._entries]
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), "utf-8")
        tmp.replace(path)

    # -- writes -------------------------------------------------------------
    def record(self, entry: HistoryEntry) -> HistoryEntry:
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries :]
            self._flush()
        return entry

    def clear(self) -> int:
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._flush()
        return count

    # -- reads --------------------------------------------------------------
    def history(self, limit: int | None = None) -> list[HistoryEntry]:
        with self._lock:
            entries = list(reversed(self._entries))  # newest first
        return entries[:limit] if limit else entries

    def timings(self) -> list[float]:
        with self._lock:
            return [entry.response_time_ms for entry in self._entries]

    def stats(self) -> dict[str, Any]:
        timings = self.timings()
        with self._lock:
            last = self._entries[-1].created_at if self._entries else None
            total = len(self._entries)
        if not timings:
            return {
                "questions_asked": total,
                "average_response_time_ms": 0.0,
                "fastest_response_ms": None,
                "slowest_response_ms": None,
                "last_question_at": last,
            }
        return {
            "questions_asked": total,
            "average_response_time_ms": round(sum(timings) / len(timings), 2),
            "fastest_response_ms": round(min(timings), 2),
            "slowest_response_ms": round(max(timings), 2),
            "last_question_at": last,
        }


_analytics: AnalyticsService | None = None
_analytics_lock = threading.Lock()


def get_analytics() -> AnalyticsService:
    global _analytics
    if _analytics is None:
        with _analytics_lock:
            if _analytics is None:
                _analytics = AnalyticsService()
    return _analytics

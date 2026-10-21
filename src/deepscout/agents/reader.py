"""Reader - fetch sources and extract verbatim evidence. BUILD_PLAN: Thread C."""
from __future__ import annotations

import urllib.request
import urllib.error
from html.parser import HTMLParser

from ..config import settings
from ..models import Evidence, SearchResult, SubQuestion
from .base import BaseAgent

_QUOTE_SYSTEM = (
    "You are a research assistant. Given source text and a question, extract the single "
    "most relevant verbatim passage (copied exactly, word for word) from the source that "
    "helps answer the question. Never paraphrase — copy the exact words.\n\n"
    'Respond with JSON only: {"quote": "<exact text from source>"}'
)


class _TextExtractor(HTMLParser):
    """Extract visible text from HTML, skipping scripts and styles."""

    def __init__(self):
        super().__init__()
        self._skip = False
        self._parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            stripped = data.strip()
            if stripped:
                self._parts.append(stripped)

    def get_text(self) -> str:
        return " ".join(self._parts)


def _fetch_text(url: str, timeout: int = 8) -> str:
    """Fetch a URL and return its visible text; returns '' on any error."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "DeepScout/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        extractor = _TextExtractor()
        try:
            extractor.feed(raw)
        except Exception:
            pass
        return extractor.get_text()[:8000]
    except Exception:
        return ""


class ReaderAgent(BaseAgent):
    name = "reader"

    def run(self, sub: SubQuestion, results: list[SearchResult]) -> list[Evidence]:
        with self.tracer.step(self.name, f"read:{sub.id}") as step:
            out: list[Evidence] = []
            total_in = total_out = 0
            total_cost = 0.0

            for i, r in enumerate(results):
                if settings.provider == "mock":
                    quote = r.snippet
                else:
                    content = _fetch_text(r.url) or r.snippet
                    user = f"Question: {sub.question}\n\nSource text:\n{content}"
                    try:
                        data, usage = self.llm.complete_json(_QUOTE_SYSTEM, user, max_tokens=256)
                        quote = (data.get("quote", "") if isinstance(data, dict) else "") or r.snippet
                        total_in += usage.input_tokens
                        total_out += usage.output_tokens
                        total_cost += usage.cost_usd
                    except Exception:
                        quote = r.snippet

                out.append(Evidence(
                    source_id=f"{sub.id}-s{i}",
                    url=r.url,
                    title=r.title,
                    quote=quote,
                    sub_question_id=sub.id,
                ))

            step.input_tokens = total_in
            step.output_tokens = total_out
            step.cost_usd = total_cost
            return out

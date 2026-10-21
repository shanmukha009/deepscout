"""Reader - fetch sources and extract verbatim evidence. BUILD_PLAN: Thread C.

Two performance/quality upgrades over the first version:
  * Sources for a sub-question are fetched and read concurrently (I/O bound).
  * When DS_RAG=1, each source's full text is chunked, embedded, and reduced to the
    passages most relevant to the sub-question before quote extraction, so the LLM
    grounds on the right span instead of a blind 8k-char truncation.
"""
from __future__ import annotations

import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser

from ..config import settings
from ..llm import Usage
from ..models import Evidence, SearchResult, SubQuestion
from ..rag import retrieve
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
        return extractor.get_text()[:20000]
    except Exception:
        return ""


class ReaderAgent(BaseAgent):
    name = "reader"

    def _read_one(self, sub: SubQuestion, i: int, r: SearchResult) -> tuple[Evidence, Usage]:
        if settings.provider == "mock":
            quote = r.snippet
            usage = Usage()
        else:
            content = _fetch_text(r.url) or r.snippet
            # RAG: reduce the page to the passages most relevant to the sub-question.
            if settings.rag_enabled and content:
                top = retrieve(sub.question, content, k=settings.rag_top_k)
                context = "\n...\n".join(top) if top else content[:8000]
            else:
                context = content[:8000]
            user = f"Question: {sub.question}\n\nSource text:\n{context}"
            try:
                data, usage = self.llm.complete_json(_QUOTE_SYSTEM, user, max_tokens=256)
                quote = (data.get("quote", "") if isinstance(data, dict) else "") or r.snippet
            except Exception:
                quote = r.snippet
                usage = Usage()

        evidence = Evidence(
            source_id=f"{sub.id}-s{i}",
            url=r.url,
            title=r.title,
            quote=quote,
            sub_question_id=sub.id,
        )
        return evidence, usage

    def run(self, sub: SubQuestion, results: list[SearchResult]) -> list[Evidence]:
        results = results[: settings.sources_per_subq]
        with self.tracer.step(self.name, f"read:{sub.id}") as step:
            if not results:
                return []

            # Read all sources for this sub-question concurrently (network + LLM I/O).
            with ThreadPoolExecutor(max_workers=min(len(results), 5)) as pool:
                pairs = list(pool.map(lambda ir: self._read_one(sub, ir[0], ir[1]),
                                      list(enumerate(results))))

            out = [ev for ev, _ in pairs]
            step.input_tokens = sum(u.input_tokens for _, u in pairs)
            step.output_tokens = sum(u.output_tokens for _, u in pairs)
            step.cost_usd = round(sum(u.cost_usd for _, u in pairs), 6)
            return out

"""Searcher - find sources per sub-question. BUILD_PLAN: Thread C."""
from __future__ import annotations

from ..config import settings
from ..models import SearchResult, SubQuestion
from ..tools.search import SearchTool
from .base import BaseAgent

_REWRITE_SYSTEM = (
    "You are a search query optimizer. Given a research sub-question, rewrite it as a "
    "concise, effective web search query. Return only the query string, nothing else."
)


class SearcherAgent(BaseAgent):
    name = "searcher"

    def __init__(self, llm, tracer, search: SearchTool):
        super().__init__(llm, tracer)
        self.search = search

    def run(self, sub: SubQuestion) -> list[SearchResult]:
        with self.tracer.step(self.name, f"search:{sub.id}") as step:
            query = sub.question
            if settings.provider != "mock":
                text, usage = self.llm.complete(_REWRITE_SYSTEM, sub.question, max_tokens=64)
                rewritten = text.strip().strip("\"'")
                if rewritten:
                    query = rewritten
                step.input_tokens = usage.input_tokens
                step.output_tokens = usage.output_tokens
                step.cost_usd = usage.cost_usd
            try:
                return self.search.query(query)
            except Exception:
                return []

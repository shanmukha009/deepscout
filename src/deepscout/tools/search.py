"""Pluggable search. BUILD_PLAN: Thread C."""
from __future__ import annotations

import json
import urllib.request

from ..config import settings
from ..models import SearchResult


class SearchTool:
    def query(self, q: str, k: int = 5) -> list[SearchResult]:
        if settings.search_provider == "mock":
            return [SearchResult(title="Mock source", url="https://example.com", snippet=f"stub result for: {q}")]
        if settings.search_provider == "tavily":
            return self._tavily(q, k)
        if settings.search_provider == "serper":
            return self._serper(q, k)
        raise NotImplementedError(f"Search provider {settings.search_provider} not implemented")

    def _tavily(self, q: str, k: int) -> list[SearchResult]:
        payload = json.dumps({
            "api_key": settings.search_api_key,
            "query": q,
            "max_results": k,
        }).encode()
        req = urllib.request.Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
            )
            for item in data.get("results", [])[:k]
        ]

    def _serper(self, q: str, k: int) -> list[SearchResult]:
        payload = json.dumps({"q": q, "num": k}).encode()
        req = urllib.request.Request(
            "https://google.serper.dev/search",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "X-API-KEY": settings.search_api_key,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        return [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
            )
            for item in data.get("organic", [])[:k]
        ]

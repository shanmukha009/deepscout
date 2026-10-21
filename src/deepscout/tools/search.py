"""Pluggable search. BUILD_PLAN: Thread C.

Resilient by design: transient rate limits (HTTP 429/432) and server errors are
retried with exponential backoff; a search that still fails returns an empty list
so one bad call never aborts a whole research run or benchmark. Results are cached
when DS_CACHE=1.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from ..cache import cache, make_key
from ..config import settings
from ..models import SearchResult

# Codes worth retrying: 429 Too Many Requests, 432 (Tavily rate/credit),
# and 5xx server errors.
_RETRY_CODES = {429, 432, 500, 502, 503, 504}
_MAX_RETRIES = 4


class SearchTool:
    def query(self, q: str, k: int = 5) -> list[SearchResult]:
        key = make_key("search", settings.search_provider, q, k)
        hit = cache.get(key)
        if hit is not None:
            return [SearchResult(**r) for r in hit]

        try:
            results = self._query_uncached(q, k)
        except Exception:
            # Never let a single search failure crash the run.
            return []

        if results:
            cache.set(key, [r.model_dump() for r in results])
        return results

    def _query_uncached(self, q: str, k: int) -> list[SearchResult]:
        if settings.search_provider == "mock":
            return [SearchResult(title="Mock source", url="https://example.com", snippet=f"stub result for: {q}")]
        if settings.search_provider == "tavily":
            return self._with_retry(lambda: self._tavily(q, k))
        if settings.search_provider == "serper":
            return self._with_retry(lambda: self._serper(q, k))
        raise NotImplementedError(f"Search provider {settings.search_provider} not implemented")

    @staticmethod
    def _with_retry(fn):
        delay = 1.0
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return fn()
            except urllib.error.HTTPError as e:
                last_exc = e
                if e.code in _RETRY_CODES and attempt < _MAX_RETRIES - 1:
                    time.sleep(delay)
                    delay *= 2  # exponential backoff: 1s, 2s, 4s
                    continue
                raise
            except (urllib.error.URLError, TimeoutError) as e:
                last_exc = e
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise
        if last_exc:
            raise last_exc
        return []

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

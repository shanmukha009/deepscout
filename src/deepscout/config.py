"""Central config, read from the environment. See .env.example."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _as_bool(val: str) -> bool:
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    provider: str = os.getenv("DS_PROVIDER", "mock")  # mock | anthropic | openai
    model: str = os.getenv("DS_MODEL", "mock-model")
    search_provider: str = os.getenv("DS_SEARCH", "mock")  # mock | tavily | serper
    max_sub_questions: int = int(os.getenv("DS_MAX_SUBQ", "5"))
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    search_api_key: str = os.getenv("SEARCH_API_KEY", "")

    # --- performance / cost ---
    # Response + search caching. Keyed on a hash of the exact request, so repeated
    # or overlapping queries return instantly and for free. Off by default so the
    # benchmark reports honest cold-start latency; turn on in production.
    cache_enabled: bool = _as_bool(os.getenv("DS_CACHE", "0"))
    cache_dir: str = os.getenv("DS_CACHE_DIR", ".ds_cache")
    # Sources fetched+read per sub-question (lower = faster/cheaper).
    sources_per_subq: int = int(os.getenv("DS_SOURCES_PER_SUBQ", "5"))

    # --- RAG (retrieval over fetched source text) ---
    # When on, the Reader chunks each source, embeds the chunks, and keeps only the
    # passages most semantically similar to the sub-question before quote extraction.
    rag_enabled: bool = _as_bool(os.getenv("DS_RAG", "1"))
    embed_provider: str = os.getenv("DS_EMBED", "mock")  # mock | openai
    embed_model: str = os.getenv("DS_EMBED_MODEL", "text-embedding-3-small")
    rag_top_k: int = int(os.getenv("DS_RAG_TOPK", "4"))
    rag_chunk_chars: int = int(os.getenv("DS_RAG_CHUNK_CHARS", "600"))


settings = Settings()

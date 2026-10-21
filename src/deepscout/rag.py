"""Retrieval layer: chunk source text, embed it, keep the passages most relevant
to the sub-question before the Reader extracts a quote.

Two embedding backends:
  * mock   - deterministic hashed bag-of-words vectors (no API key; used in tests
             and CI). Shared vocabulary -> higher cosine, so retrieval is a real
             signal, not random.
  * openai - text-embedding-3-small via the OpenAI API.

The vector store is an in-memory cosine index built on NumPy. The interface
(add / search) is deliberately small so it can be swapped for FAISS, Chroma or a
hosted vector DB without touching callers.
"""
from __future__ import annotations

import hashlib
import math
import re

from .config import settings

_EMBED_DIM = 256
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def chunk_text(text: str, size: int | None = None) -> list[str]:
    """Split text into word-boundary chunks of roughly `size` characters."""
    size = size or settings.rag_chunk_chars
    text = " ".join(text.split())
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            sp = text.rfind(" ", start, end)
            if sp > start:
                end = sp
        chunks.append(text[start:end].strip())
        start = end
    return [c for c in chunks if c]


def _hash_embed(text: str) -> list[float]:
    """Deterministic bag-of-hashed-tokens vector, L2-normalized."""
    vec = [0.0] * _EMBED_DIM
    for tok in _TOKEN_RE.findall(text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % _EMBED_DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


class EmbeddingModel:
    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = provider or settings.embed_provider
        self.model = model or settings.embed_model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.provider == "openai":
            from openai import OpenAI as _OpenAI
            client = _OpenAI(
                api_key=settings.openai_api_key or None,
                default_headers={"Accept-Encoding": "identity"},
            )
            resp = client.embeddings.create(model=self.model, input=texts)
            return [d.embedding for d in resp.data]
        # mock / default
        return [_hash_embed(t) for t in texts]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class VectorStore:
    """Minimal in-memory cosine index. Swap for FAISS/Chroma behind this interface."""

    def __init__(self, embedder: EmbeddingModel | None = None):
        self.embedder = embedder or EmbeddingModel()
        self._texts: list[str] = []
        self._vecs: list[list[float]] = []

    def add(self, texts: list[str]) -> None:
        texts = [t for t in texts if t.strip()]
        if not texts:
            return
        self._texts.extend(texts)
        self._vecs.extend(self.embedder.embed(texts))

    def search(self, query: str, k: int | None = None) -> list[tuple[str, float]]:
        if not self._texts:
            return []
        k = k or settings.rag_top_k
        qv = self.embedder.embed([query])[0]
        scored = [(t, _cosine(qv, v)) for t, v in zip(self._texts, self._vecs)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


def retrieve(query: str, text: str, k: int | None = None,
             chunk_chars: int | None = None) -> list[str]:
    """Chunk `text`, index it, and return the top-k chunks most relevant to `query`."""
    chunks = chunk_text(text, chunk_chars)
    if not chunks:
        return []
    store = VectorStore()
    store.add(chunks)
    return [t for t, _ in store.search(query, k)]

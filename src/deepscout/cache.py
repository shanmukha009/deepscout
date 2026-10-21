"""Tiny disk-backed cache for LLM and search calls.

Keyed on a SHA-256 of the exact request, so identical or repeated calls return
instantly and at zero cost. Writes are atomic (temp file + rename) so concurrent
agent threads don't corrupt entries. Enabled with DS_CACHE=1.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .config import settings


def make_key(*parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DiskCache:
    def __init__(self, directory: str | None = None, enabled: bool | None = None):
        self.enabled = settings.cache_enabled if enabled is None else enabled
        self.dir = Path(directory or settings.cache_dir)
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        p = self._path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        try:
            fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(value, f)
            os.replace(tmp, self._path(key))
        except OSError:
            pass


# Process-wide default cache.
cache = DiskCache()

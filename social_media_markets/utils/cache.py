"""Simple disk-based cache for API responses."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path


class DiskCache:
    """File-based cache with TTL support."""

    def __init__(self, cache_dir: str = ".smm_cache", default_ttl: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.default_ttl = default_ttl
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_path(self, key: str) -> Path:
        hashed = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{hashed}.json"

    def get(self, key: str) -> dict | list | None:
        path = self._key_path(key)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                entry = json.load(f)
            if time.time() - entry["created_at"] > entry.get("ttl", self.default_ttl):
                path.unlink(missing_ok=True)
                return None
            return entry["data"]
        except (json.JSONDecodeError, KeyError):
            path.unlink(missing_ok=True)
            return None

    def set(self, key: str, data: dict | list, ttl: int | None = None) -> None:
        path = self._key_path(key)
        entry = {
            "created_at": time.time(),
            "ttl": ttl if ttl is not None else self.default_ttl,
            "data": data,
        }
        with open(path, "w") as f:
            json.dump(entry, f)

    def clear(self) -> int:
        """Remove all cached entries. Returns count of removed files."""
        count = 0
        for path in self.cache_dir.glob("*.json"):
            path.unlink()
            count += 1
        return count

    def evict_expired(self) -> int:
        """Remove expired entries. Returns count of removed files."""
        count = 0
        for path in self.cache_dir.glob("*.json"):
            try:
                with open(path) as f:
                    entry = json.load(f)
                if time.time() - entry["created_at"] > entry.get("ttl", self.default_ttl):
                    path.unlink()
                    count += 1
            except (json.JSONDecodeError, KeyError):
                path.unlink()
                count += 1
        return count

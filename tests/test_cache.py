"""Tests for the caching system."""

from __future__ import annotations

import tempfile

import pytest

from social_media_markets.utils.cache import DiskCache


class TestDiskCache:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache = DiskCache(cache_dir=self.tmpdir, default_ttl=3600)

    def test_set_and_get(self):
        self.cache.set("key1", {"value": 42})
        result = self.cache.get("key1")
        assert result == {"value": 42}

    def test_get_missing(self):
        result = self.cache.get("nonexistent")
        assert result is None

    def test_ttl_expiry(self):
        # Set with very short TTL
        self.cache.set("short", {"data": 1}, ttl=0)
        # Should be expired immediately
        import time
        time.sleep(0.01)
        result = self.cache.get("short")
        assert result is None

    def test_clear(self):
        self.cache.set("a", {"x": 1})
        self.cache.set("b", {"y": 2})
        count = self.cache.clear()
        assert count == 2
        assert self.cache.get("a") is None
        assert self.cache.get("b") is None

    def test_list_data(self):
        self.cache.set("list_key", [1, 2, 3])
        result = self.cache.get("list_key")
        assert result == [1, 2, 3]

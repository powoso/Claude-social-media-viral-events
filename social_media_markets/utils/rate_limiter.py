"""Rate limiter for API calls."""

from __future__ import annotations

import time
from collections import defaultdict


class RateLimiter:
    """Token bucket rate limiter keyed by domain/source."""

    def __init__(self, default_interval: float = 1.0):
        self._last_call: dict[str, float] = defaultdict(float)
        self._intervals: dict[str, float] = {}
        self.default_interval = default_interval

    def set_interval(self, source: str, interval: float) -> None:
        self._intervals[source] = interval

    def wait(self, source: str) -> None:
        """Block until it's safe to make the next call for this source."""
        interval = self._intervals.get(source, self.default_interval)
        elapsed = time.monotonic() - self._last_call[source]
        if elapsed < interval:
            time.sleep(interval - elapsed)
        self._last_call[source] = time.monotonic()

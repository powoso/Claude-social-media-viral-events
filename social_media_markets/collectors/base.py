"""Base collector interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from social_media_markets.config import APIConfig
from social_media_markets.models import MetricTimeSeries
from social_media_markets.utils.cache import DiskCache
from social_media_markets.utils.rate_limiter import RateLimiter


class BaseCollector(ABC):
    """Abstract base for all data collectors."""

    source_name: str = "unknown"

    def __init__(self, config: APIConfig, cache: DiskCache | None = None):
        self.config = config
        self.cache = cache or DiskCache(config.cache_dir)
        self.rate_limiter = RateLimiter()

    @abstractmethod
    def collect(self, entity_name: str, **kwargs) -> list[MetricTimeSeries]:
        """Collect time series data for a given entity.

        Args:
            entity_name: Channel name, app name, subreddit, etc.

        Returns:
            List of MetricTimeSeries objects (may return multiple metrics).
        """

    def _cache_key(self, entity_name: str, **kwargs) -> str:
        parts = [self.source_name, entity_name]
        for k, v in sorted(kwargs.items()):
            parts.append(f"{k}={v}")
        return ":".join(parts)

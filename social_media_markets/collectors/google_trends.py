"""Google Trends collector — search interest as a leading indicator."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from social_media_markets.collectors.base import BaseCollector
from social_media_markets.config import MetricType, Platform
from social_media_markets.models import MetricTimeSeries, TimeSeriesPoint

logger = logging.getLogger(__name__)


class GoogleTrendsCollector(BaseCollector):
    """Collects Google Trends interest-over-time data via pytrends."""

    source_name = "google_trends"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rate_limiter.set_interval("google_trends", 2.0)

    def collect(
        self,
        entity_name: str,
        timeframe: str = "today 12-m",
        geo: str = "",
        **kwargs,
    ) -> list[MetricTimeSeries]:
        """Collect Google Trends interest-over-time for a search term.

        Args:
            entity_name: Search term (e.g., creator name, app name).
            timeframe: pytrends timeframe string (default: last 12 months).
            geo: Geographic filter (default: worldwide).

        Returns:
            Single MetricTimeSeries with normalized search interest (0-100).
        """
        cache_key = self._cache_key(entity_name, timeframe=timeframe, geo=geo)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return self._deserialize_points(cached, entity_name)

        try:
            from pytrends.request import TrendReq
        except ImportError:
            logger.error("pytrends not installed — run: pip install pytrends")
            return []

        self.rate_limiter.wait("google_trends")

        try:
            proxies = []
            if self.config.google_trends_proxy:
                proxies = [self.config.google_trends_proxy]

            pytrends = TrendReq(hl="en-US", tz=0, retries=3, backoff_factor=1.0, proxies=proxies)
            pytrends.build_payload([entity_name], timeframe=timeframe, geo=geo)
            df = pytrends.interest_over_time()
        except Exception as e:
            logger.error("Google Trends error for '%s': %s", entity_name, e)
            return []

        if df.empty or entity_name not in df.columns:
            logger.info("No Google Trends data for '%s'", entity_name)
            return []

        points = []
        for ts_idx, row in df.iterrows():
            dt = ts_idx.to_pydatetime().replace(tzinfo=timezone.utc)
            points.append(
                TimeSeriesPoint(
                    timestamp=dt,
                    value=float(row[entity_name]),
                    source="google_trends",
                )
            )

        series = MetricTimeSeries(
            platform=Platform.YOUTUBE,  # platform-agnostic, use YouTube as default
            metric_type=MetricType.PAGEVIEWS,  # reuse as "search interest"
            entity_name=entity_name,
            points=sorted(points, key=lambda p: p.timestamp),
        )

        self.cache.set(
            cache_key,
            [{"timestamp": p.timestamp.isoformat(), "value": p.value} for p in points],
            ttl=7200,
        )
        return [series]

    def collect_related_queries(self, entity_name: str, timeframe: str = "today 12-m") -> dict:
        """Collect related rising queries — useful for momentum detection."""
        try:
            from pytrends.request import TrendReq
        except ImportError:
            return {}

        self.rate_limiter.wait("google_trends")

        try:
            pytrends = TrendReq(hl="en-US", tz=0)
            pytrends.build_payload([entity_name], timeframe=timeframe)
            related = pytrends.related_queries()
        except Exception as e:
            logger.error("Related queries error for '%s': %s", entity_name, e)
            return {}

        result = {}
        if entity_name in related:
            top = related[entity_name].get("top")
            rising = related[entity_name].get("rising")
            if top is not None and not top.empty:
                result["top"] = top.to_dict("records")
            if rising is not None and not rising.empty:
                result["rising"] = rising.to_dict("records")
        return result

    @staticmethod
    def _deserialize_points(data: list[dict], entity_name: str) -> list[MetricTimeSeries]:
        points = [
            TimeSeriesPoint(
                timestamp=datetime.fromisoformat(p["timestamp"]),
                value=p["value"],
                source="google_trends",
            )
            for p in data
        ]
        return [
            MetricTimeSeries(
                platform=Platform.YOUTUBE,
                metric_type=MetricType.PAGEVIEWS,
                entity_name=entity_name,
                points=points,
            )
        ]

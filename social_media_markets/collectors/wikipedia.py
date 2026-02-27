"""Wikipedia pageview collector — attention proxy for people/topics."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import httpx

from social_media_markets.collectors.base import BaseCollector
from social_media_markets.config import MetricType, Platform
from social_media_markets.models import MetricTimeSeries, TimeSeriesPoint

logger = logging.getLogger(__name__)

WIKIMEDIA_API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"


class WikipediaCollector(BaseCollector):
    """Collects Wikipedia pageview data via the Wikimedia REST API."""

    source_name = "wikipedia"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rate_limiter.set_interval("wikipedia", 0.5)
        self._client = httpx.Client(
            timeout=self.config.request_timeout,
            headers={"User-Agent": "SMMBot/0.1 (social-media-markets; research)"},
        )

    def collect(
        self,
        entity_name: str,
        days: int = 365,
        project: str = "en.wikipedia",
        **kwargs,
    ) -> list[MetricTimeSeries]:
        """Collect daily pageview counts for a Wikipedia article.

        Args:
            entity_name: Wikipedia article title (spaces are auto-converted).
            days: Number of days of history to fetch.
            project: Wikimedia project (default: en.wikipedia).

        Returns:
            Single MetricTimeSeries with daily pageview counts.
        """
        cache_key = self._cache_key(entity_name, days=days, project=project)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return self._deserialize(cached, entity_name)

        self.rate_limiter.wait("wikipedia")

        article_title = entity_name.replace(" ", "_")
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)

        url = (
            f"{WIKIMEDIA_API}/{project}/all-access/all-agents/"
            f"{article_title}/daily/"
            f"{start_date.strftime('%Y%m%d')}/{end_date.strftime('%Y%m%d')}"
        )

        try:
            resp = self._client.get(url)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            logger.error("Wikipedia pageview error for '%s': %s", entity_name, e)
            return []

        items = data.get("items", [])
        if not items:
            logger.info("No Wikipedia pageview data for '%s'", entity_name)
            return []

        points = []
        for item in items:
            try:
                ts = datetime.strptime(item["timestamp"], "%Y%m%d00").replace(tzinfo=timezone.utc)
                points.append(
                    TimeSeriesPoint(
                        timestamp=ts, value=float(item["views"]), source="wikimedia_api"
                    )
                )
            except (KeyError, ValueError):
                continue

        series = MetricTimeSeries(
            platform=Platform.YOUTUBE,  # platform-agnostic attention metric
            metric_type=MetricType.PAGEVIEWS,
            entity_name=entity_name,
            points=sorted(points, key=lambda p: p.timestamp),
        )

        self.cache.set(
            cache_key,
            [{"timestamp": p.timestamp.isoformat(), "value": p.value} for p in points],
            ttl=3600,
        )
        return [series]

    @staticmethod
    def _deserialize(data: list[dict], entity_name: str) -> list[MetricTimeSeries]:
        points = [
            TimeSeriesPoint(
                timestamp=datetime.fromisoformat(p["timestamp"]),
                value=p["value"],
                source="wikimedia_api",
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

"""Wayback Machine collector — historical snapshots for trend reconstruction."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

import httpx

from social_media_markets.collectors.base import BaseCollector
from social_media_markets.config import MetricType, Platform
from social_media_markets.models import MetricTimeSeries, TimeSeriesPoint

logger = logging.getLogger(__name__)

CDX_API = "https://web.archive.org/cdx/search/cdx"
WAYBACK_URL = "https://web.archive.org/web"


class WaybackCollector(BaseCollector):
    """Reconstructs historical metrics from Wayback Machine snapshots.

    Useful for reconstructing follower counts from cached profile pages
    when other historical data sources are unavailable.
    """

    source_name = "wayback"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rate_limiter.set_interval("wayback", self.config.wayback_rate_limit)
        self._client = httpx.Client(
            timeout=self.config.request_timeout,
            headers={"User-Agent": "SMMBot/0.1 (research)"},
            follow_redirects=True,
        )

    def collect(
        self,
        entity_name: str,
        url: str | None = None,
        platform: Platform = Platform.YOUTUBE,
        max_snapshots: int = 50,
        **kwargs,
    ) -> list[MetricTimeSeries]:
        """Collect historical data from Wayback Machine snapshots.

        Args:
            entity_name: Account/channel name.
            url: The URL to look up in the Wayback Machine. If None, constructs
                 a Social Blade URL for the entity.
            platform: Platform for metric classification.
            max_snapshots: Maximum number of snapshots to fetch and parse.

        Returns:
            MetricTimeSeries with historically reconstructed data points.
        """
        if url is None:
            url = f"https://socialblade.com/youtube/channel/{entity_name}"

        cache_key = self._cache_key(entity_name, url=url)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return self._deserialize(cached, entity_name, platform)

        # Step 1: Get list of available snapshots via CDX API
        snapshots = self._get_snapshot_list(url, max_snapshots)
        if not snapshots:
            logger.info("No Wayback snapshots for %s", url)
            return []

        # Step 2: Fetch and parse each snapshot for follower/subscriber counts
        points = []
        for snap_ts, snap_url in snapshots:
            self.rate_limiter.wait("wayback")
            value = self._extract_metric_from_snapshot(snap_url)
            if value is not None:
                points.append(
                    TimeSeriesPoint(timestamp=snap_ts, value=value, source="wayback")
                )

        if not points:
            return []

        series = MetricTimeSeries(
            platform=platform,
            metric_type=MetricType.SUBSCRIBERS if platform == Platform.YOUTUBE else MetricType.FOLLOWERS,
            entity_name=entity_name,
            points=sorted(points, key=lambda p: p.timestamp),
        )

        self.cache.set(
            cache_key,
            [{"timestamp": p.timestamp.isoformat(), "value": p.value} for p in points],
            ttl=86400,  # cache for 24h since historical data doesn't change
        )
        return [series]

    def _get_snapshot_list(
        self, url: str, max_snapshots: int
    ) -> list[tuple[datetime, str]]:
        """Query CDX API for available snapshots of a URL."""
        params = {
            "url": url,
            "output": "json",
            "fl": "timestamp,statuscode",
            "filter": "statuscode:200",
            "collapse": "timestamp:6",  # one per month
            "limit": max_snapshots,
        }

        try:
            resp = self._client.get(CDX_API, params=params)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            logger.error("Wayback CDX error for %s: %s", url, e)
            return []

        if not data or len(data) < 2:
            return []

        results = []
        for row in data[1:]:  # first row is headers
            ts_str = row[0]
            try:
                ts = datetime.strptime(ts_str, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
                snap_url = f"{WAYBACK_URL}/{ts_str}/{url}"
                results.append((ts, snap_url))
            except ValueError:
                continue

        return results

    def _extract_metric_from_snapshot(self, snapshot_url: str) -> float | None:
        """Fetch a Wayback snapshot and extract a follower/subscriber count."""
        try:
            resp = self._client.get(snapshot_url)
            resp.raise_for_status()
            html = resp.text
        except httpx.HTTPError as e:
            logger.debug("Failed to fetch snapshot %s: %s", snapshot_url, e)
            return None

        return self._parse_subscriber_count(html)

    @staticmethod
    def _parse_subscriber_count(html: str) -> float | None:
        """Extract subscriber/follower count from a cached page.

        Handles common patterns found in Social Blade and profile pages.
        """
        patterns = [
            # Social Blade style
            r'(?:subscribers?|followers?)[:\s]*([0-9,]+(?:\.\d+)?[KMB]?)',
            r'([0-9,]+(?:\.\d+)?[KMB]?)\s*(?:subscribers?|followers?)',
            # JSON-LD or meta tags
            r'"subscriberCount"[:\s]*"?([0-9,]+)"?',
            r'"interactionCount"[:\s]*"?([0-9,]+)"?',
        ]

        for pattern in patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                return WaybackCollector._parse_number(match.group(1))

        return None

    @staticmethod
    def _parse_number(text: str) -> float | None:
        text = text.strip().replace(",", "")
        multipliers = {"K": 1e3, "M": 1e6, "B": 1e9}
        for suffix, mult in multipliers.items():
            if text.upper().endswith(suffix):
                try:
                    return float(text[:-1]) * mult
                except ValueError:
                    return None
        try:
            return float(text)
        except ValueError:
            return None

    @staticmethod
    def _deserialize(data: list[dict], entity_name: str, platform: Platform) -> list[MetricTimeSeries]:
        points = [
            TimeSeriesPoint(
                timestamp=datetime.fromisoformat(p["timestamp"]),
                value=p["value"],
                source="wayback",
            )
            for p in data
        ]
        metric = MetricType.SUBSCRIBERS if platform == Platform.YOUTUBE else MetricType.FOLLOWERS
        return [
            MetricTimeSeries(
                platform=platform,
                metric_type=metric,
                entity_name=entity_name,
                points=points,
            )
        ]

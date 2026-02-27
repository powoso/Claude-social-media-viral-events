"""Twitch viewership data collector."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

import httpx
from bs4 import BeautifulSoup

from social_media_markets.collectors.base import BaseCollector
from social_media_markets.config import MetricType, Platform
from social_media_markets.models import MetricTimeSeries, TimeSeriesPoint

logger = logging.getLogger(__name__)

# TwitchTracker public page for historical data
TWITCH_TRACKER_BASE = "https://twitchtracker.com"


class TwitchCollector(BaseCollector):
    """Collects Twitch streaming viewership data from TwitchTracker."""

    source_name = "twitch"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rate_limiter.set_interval("twitch_tracker", 2.0)
        self._client = httpx.Client(
            timeout=self.config.request_timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; SMMBot/0.1)"},
            follow_redirects=True,
        )

    def collect(
        self,
        entity_name: str,
        **kwargs,
    ) -> list[MetricTimeSeries]:
        """Collect Twitch channel statistics from TwitchTracker.

        Args:
            entity_name: Twitch channel name.

        Returns:
            MetricTimeSeries with follower and viewership data.
        """
        cache_key = self._cache_key(entity_name)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return self._deserialize(cached, entity_name)

        self.rate_limiter.wait("twitch_tracker")

        url = f"{TWITCH_TRACKER_BASE}/{entity_name}"
        try:
            resp = self._client.get(url)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.error("TwitchTracker error for %s: %s", entity_name, e)
            return []

        results = self._parse_tracker_page(resp.text, entity_name)

        if results:
            serialized = [
                {
                    "metric_type": s.metric_type.value,
                    "points": [
                        {"timestamp": p.timestamp.isoformat(), "value": p.value}
                        for p in s.points
                    ],
                }
                for s in results
            ]
            self.cache.set(cache_key, serialized, ttl=3600)

        return results

    def _parse_tracker_page(
        self, html: str, entity_name: str
    ) -> list[MetricTimeSeries]:
        """Parse TwitchTracker channel page for stats."""
        soup = BeautifulSoup(html, "lxml")
        results = []
        now = datetime.now(timezone.utc)

        # Extract current follower count
        follower_el = soup.select_one(".profile-followers .value, [data-stat='followers']")
        if follower_el:
            val = self._parse_number(follower_el.get_text(strip=True))
            if val is not None:
                results.append(
                    MetricTimeSeries(
                        platform=Platform.TWITCH,
                        metric_type=MetricType.FOLLOWERS,
                        entity_name=entity_name,
                        points=[TimeSeriesPoint(timestamp=now, value=val, source="twitch_tracker")],
                    )
                )

        # Extract average viewers
        viewer_el = soup.select_one(".profile-viewers .value, [data-stat='avg-viewers']")
        if viewer_el:
            val = self._parse_number(viewer_el.get_text(strip=True))
            if val is not None:
                results.append(
                    MetricTimeSeries(
                        platform=Platform.TWITCH,
                        metric_type=MetricType.VIEWS,
                        entity_name=entity_name,
                        points=[TimeSeriesPoint(timestamp=now, value=val, source="twitch_tracker")],
                    )
                )

        # Try to extract monthly chart data from embedded JavaScript
        chart_data = self._extract_chart_data(html, entity_name)
        if chart_data:
            results.extend(chart_data)

        return results

    def _extract_chart_data(
        self, html: str, entity_name: str
    ) -> list[MetricTimeSeries]:
        """Extract historical chart data from JavaScript in the page."""
        results = []

        # TwitchTracker embeds chart data in JavaScript arrays
        # Look for patterns like: data: [{x: "2024-01-01", y: 12345}, ...]
        chart_pattern = r'data:\s*\[(\{[^]]+\})\]'
        matches = re.findall(chart_pattern, html)

        for match_str in matches[:2]:  # limit to first 2 charts (followers, viewers)
            points = []
            point_pattern = r'\{[^}]*x:\s*"([^"]+)"[^}]*y:\s*(\d+(?:\.\d+)?)[^}]*\}'
            for m in re.finditer(point_pattern, match_str):
                try:
                    ts = datetime.fromisoformat(m.group(1)).replace(tzinfo=timezone.utc)
                    val = float(m.group(2))
                    points.append(TimeSeriesPoint(timestamp=ts, value=val, source="twitch_tracker"))
                except (ValueError, IndexError):
                    continue

            if points:
                results.append(
                    MetricTimeSeries(
                        platform=Platform.TWITCH,
                        metric_type=MetricType.FOLLOWERS,
                        entity_name=entity_name,
                        points=sorted(points, key=lambda p: p.timestamp),
                    )
                )

        return results

    @staticmethod
    def _parse_number(text: str) -> float | None:
        text = text.strip().replace(",", "").replace(" ", "")
        if not text:
            return None
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
    def _deserialize(data: list[dict], entity_name: str) -> list[MetricTimeSeries]:
        results = []
        for entry in data:
            points = [
                TimeSeriesPoint(
                    timestamp=datetime.fromisoformat(p["timestamp"]),
                    value=p["value"],
                    source="twitch_tracker",
                )
                for p in entry.get("points", [])
            ]
            results.append(
                MetricTimeSeries(
                    platform=Platform.TWITCH,
                    metric_type=MetricType(entry["metric_type"]),
                    entity_name=entity_name,
                    points=points,
                )
            )
        return results

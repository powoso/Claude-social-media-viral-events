"""Spotify Charts collector — streaming count data for music milestones."""

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

# Spotify public chart pages
SPOTIFY_CHARTS_BASE = "https://charts.spotify.com"


class SpotifyCollector(BaseCollector):
    """Collects Spotify streaming data from public chart pages.

    Focuses on chart position and estimated stream counts for
    predicting music milestones (e.g., "Will track X reach 1B streams?").
    """

    source_name = "spotify"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rate_limiter.set_interval("spotify", 2.0)
        self._client = httpx.Client(
            timeout=self.config.request_timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; SMMBot/0.1)"},
            follow_redirects=True,
        )

    def collect(
        self,
        entity_name: str,
        country: str = "global",
        **kwargs,
    ) -> list[MetricTimeSeries]:
        """Collect Spotify chart/streaming data for an artist or track.

        Args:
            entity_name: Artist or track name to search for.
            country: Country code or "global".

        Returns:
            MetricTimeSeries with stream count data.
        """
        cache_key = self._cache_key(entity_name, country=country)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return self._deserialize(cached, entity_name)

        # Spotify's public charts page doesn't have a direct search API
        # We scrape the daily top 200 and look for the entity
        self.rate_limiter.wait("spotify")

        url = f"{SPOTIFY_CHARTS_BASE}/chart/regional-{country}-daily/latest"
        try:
            resp = self._client.get(url)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.error("Spotify Charts error: %s", e)
            return []

        results = self._parse_chart_page(resp.text, entity_name)

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

    def _parse_chart_page(
        self, html: str, entity_name: str
    ) -> list[MetricTimeSeries]:
        """Parse Spotify Charts page for stream data."""
        soup = BeautifulSoup(html, "lxml")
        now = datetime.now(timezone.utc)
        entity_lower = entity_name.lower()

        # Look for the entity in chart entries
        chart_entries = soup.select("tr, .chart-table-row, [data-track]")
        for entry in chart_entries:
            text = entry.get_text(" ", strip=True).lower()
            if entity_lower not in text:
                continue

            # Try to extract stream count from the row
            numbers = re.findall(r'[\d,]+', entry.get_text(" ", strip=True))
            for num_str in numbers:
                val = self._parse_number(num_str)
                if val is not None and val > 1000:  # filter out rank numbers
                    return [
                        MetricTimeSeries(
                            platform=Platform.SPOTIFY,
                            metric_type=MetricType.STREAMS,
                            entity_name=entity_name,
                            points=[
                                TimeSeriesPoint(
                                    timestamp=now, value=val, source="spotify_charts"
                                )
                            ],
                        )
                    ]

        logger.info("'%s' not found in Spotify Charts", entity_name)
        return []

    @staticmethod
    def _parse_number(text: str) -> float | None:
        text = text.strip().replace(",", "")
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
                    source="spotify_charts",
                )
                for p in entry.get("points", [])
            ]
            results.append(
                MetricTimeSeries(
                    platform=Platform.SPOTIFY,
                    metric_type=MetricType(entry["metric_type"]),
                    entity_name=entity_name,
                    points=points,
                )
            )
        return results

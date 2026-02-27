"""Social Blade data collector for YouTube, TikTok, Instagram, Twitter/X."""

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

# Maps Social Blade platform slugs to our Platform enum
PLATFORM_SLUGS = {
    Platform.YOUTUBE: "youtube",
    Platform.TIKTOK: "tiktok",
    Platform.INSTAGRAM: "instagram",
    Platform.TWITTER: "twitter",
}

# Social Blade API endpoints (official API if key available)
SB_API_BASE = "https://matrix.sbapis.com/b"
SB_WEB_BASE = "https://socialblade.com"


class SocialBladeCollector(BaseCollector):
    """Collects follower/subscriber and view count history from Social Blade.

    Supports two modes:
    1. Official API (requires SOCIALBLADE_API_KEY) — structured JSON responses
    2. Web scraping fallback — parses the public statistics pages
    """

    source_name = "socialblade"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rate_limiter.set_interval("socialblade_api", 1.0)
        self.rate_limiter.set_interval("socialblade_web", 2.0)
        self._client = httpx.Client(
            timeout=self.config.request_timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; SMMBot/0.1)"},
            follow_redirects=True,
        )

    def collect(
        self,
        entity_name: str,
        platform: Platform = Platform.YOUTUBE,
        **kwargs,
    ) -> list[MetricTimeSeries]:
        """Collect follower/subscriber and view data from Social Blade.

        Args:
            entity_name: The channel/account username or ID.
            platform: Which platform to query.

        Returns:
            List of MetricTimeSeries (subscribers/followers + views if available).
        """
        cache_key = self._cache_key(entity_name, platform=platform.value)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return self._deserialize(cached, entity_name, platform)

        if self.config.socialblade_api_key:
            result = self._collect_via_api(entity_name, platform)
        else:
            result = self._collect_via_scraping(entity_name, platform)

        if result:
            serialized = self._serialize(result)
            self.cache.set(cache_key, serialized, ttl=3600)

        return result

    # ── Official API path ──────────────────────────────────────────

    def _collect_via_api(
        self, entity_name: str, platform: Platform
    ) -> list[MetricTimeSeries]:
        """Use the official Social Blade Matrix API."""
        slug = PLATFORM_SLUGS.get(platform)
        if not slug:
            logger.warning("Social Blade API does not support platform %s", platform)
            return []

        self.rate_limiter.wait("socialblade_api")

        url = f"{SB_API_BASE}/{slug}/statistics"
        params = {"query": entity_name}
        headers = {
            "clientid": self.config.socialblade_api_key,
            "token": self.config.socialblade_api_key,
        }

        try:
            resp = self._client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            logger.error("Social Blade API error for %s/%s: %s", platform, entity_name, e)
            return self._collect_via_scraping(entity_name, platform)

        return self._parse_api_response(data, entity_name, platform)

    def _parse_api_response(
        self, data: dict, entity_name: str, platform: Platform
    ) -> list[MetricTimeSeries]:
        """Parse the JSON response from Social Blade's official API."""
        results = []

        # Determine which fields to extract based on platform
        if platform == Platform.YOUTUBE:
            sub_key, view_key = "subscribers", "views"
            sub_metric = MetricType.SUBSCRIBERS
        else:
            sub_key, view_key = "followers", "views"
            sub_metric = MetricType.FOLLOWERS

        daily = data.get("data", {}).get("daily", [])
        if not daily:
            daily = data.get("daily", [])

        sub_points = []
        view_points = []

        for entry in daily:
            try:
                ts = datetime.fromisoformat(entry["date"].replace("Z", "+00:00"))
            except (KeyError, ValueError):
                continue

            if sub_key in entry and entry[sub_key] is not None:
                sub_points.append(
                    TimeSeriesPoint(timestamp=ts, value=float(entry[sub_key]), source="sb_api")
                )
            if view_key in entry and entry[view_key] is not None:
                view_points.append(
                    TimeSeriesPoint(timestamp=ts, value=float(entry[view_key]), source="sb_api")
                )

        if sub_points:
            results.append(
                MetricTimeSeries(
                    platform=platform,
                    metric_type=sub_metric,
                    entity_name=entity_name,
                    points=sorted(sub_points, key=lambda p: p.timestamp),
                )
            )
        if view_points:
            results.append(
                MetricTimeSeries(
                    platform=platform,
                    metric_type=MetricType.VIEWS,
                    entity_name=entity_name,
                    points=sorted(view_points, key=lambda p: p.timestamp),
                )
            )

        return results

    # ── Web scraping fallback ──────────────────────────────────────

    def _collect_via_scraping(
        self, entity_name: str, platform: Platform
    ) -> list[MetricTimeSeries]:
        """Scrape the public Social Blade statistics page."""
        slug = PLATFORM_SLUGS.get(platform)
        if not slug:
            logger.warning("Social Blade scraping not supported for %s", platform)
            return []

        self.rate_limiter.wait("socialblade_web")

        url = f"{SB_WEB_BASE}/{slug}/user/{entity_name}/monthly"

        try:
            resp = self._client.get(url)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.error("Social Blade scrape error for %s/%s: %s", platform, entity_name, e)
            return []

        return self._parse_scraped_page(resp.text, entity_name, platform)

    def _parse_scraped_page(
        self, html: str, entity_name: str, platform: Platform
    ) -> list[MetricTimeSeries]:
        """Parse Social Blade's monthly statistics HTML page."""
        soup = BeautifulSoup(html, "lxml")
        results = []

        if platform == Platform.YOUTUBE:
            sub_metric = MetricType.SUBSCRIBERS
        else:
            sub_metric = MetricType.FOLLOWERS

        # Social Blade renders stats in div-based tables with specific style patterns
        stat_rows = soup.select("div[style*='width: 860px']")
        if not stat_rows:
            # Try alternate selector for newer layout
            stat_rows = soup.select(".TableMonthlyStats div[style]")

        sub_points = []
        view_points = []

        for row in stat_rows:
            cells = row.find_all("div", recursive=False)
            if len(cells) < 3:
                continue

            date_text = cells[0].get_text(strip=True)
            ts = self._parse_sb_date(date_text)
            if ts is None:
                continue

            # Second column is typically subscriber/follower count
            sub_val = self._parse_number(cells[1].get_text(strip=True))
            if sub_val is not None:
                sub_points.append(
                    TimeSeriesPoint(timestamp=ts, value=sub_val, source="sb_scrape")
                )

            # Third column is typically view count
            if len(cells) > 2:
                view_val = self._parse_number(cells[2].get_text(strip=True))
                if view_val is not None:
                    view_points.append(
                        TimeSeriesPoint(timestamp=ts, value=view_val, source="sb_scrape")
                    )

        if sub_points:
            results.append(
                MetricTimeSeries(
                    platform=platform,
                    metric_type=sub_metric,
                    entity_name=entity_name,
                    points=sorted(sub_points, key=lambda p: p.timestamp),
                )
            )
        if view_points:
            results.append(
                MetricTimeSeries(
                    platform=platform,
                    metric_type=MetricType.VIEWS,
                    entity_name=entity_name,
                    points=sorted(view_points, key=lambda p: p.timestamp),
                )
            )

        return results

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _parse_sb_date(text: str) -> datetime | None:
        """Parse Social Blade's date format (e.g. 'Jan 15, 2024' or '2024-01-15')."""
        for fmt in ("%b %d, %Y", "%Y-%m-%d", "%B %d, %Y", "%m/%d/%Y"):
            try:
                return datetime.strptime(text.strip(), fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    @staticmethod
    def _parse_number(text: str) -> float | None:
        """Parse Social Blade number strings like '1,234,567' or '1.2M'."""
        text = text.strip().replace(",", "").replace("+", "").replace(" ", "")
        if not text or text == "--" or text == "N/A":
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
    def _serialize(series_list: list[MetricTimeSeries]) -> list[dict]:
        return [
            {
                "platform": s.platform.value,
                "metric_type": s.metric_type.value,
                "entity_name": s.entity_name,
                "points": [
                    {
                        "timestamp": p.timestamp.isoformat(),
                        "value": p.value,
                        "source": p.source,
                    }
                    for p in s.points
                ],
            }
            for s in series_list
        ]

    @staticmethod
    def _deserialize(data: list[dict], entity_name: str, platform: Platform) -> list[MetricTimeSeries]:
        results = []
        for entry in data:
            points = [
                TimeSeriesPoint(
                    timestamp=datetime.fromisoformat(p["timestamp"]),
                    value=p["value"],
                    source=p.get("source", "cache"),
                )
                for p in entry.get("points", [])
            ]
            results.append(
                MetricTimeSeries(
                    platform=Platform(entry["platform"]),
                    metric_type=MetricType(entry["metric_type"]),
                    entity_name=entry["entity_name"],
                    points=points,
                )
            )
        return results

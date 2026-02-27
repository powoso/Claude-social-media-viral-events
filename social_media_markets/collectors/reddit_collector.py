"""Reddit collector — subreddit growth and engagement metrics."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from social_media_markets.collectors.base import BaseCollector
from social_media_markets.config import MetricType, Platform
from social_media_markets.models import MetricTimeSeries, TimeSeriesPoint

logger = logging.getLogger(__name__)


class RedditCollector(BaseCollector):
    """Collects subreddit subscriber counts and engagement metrics via PRAW."""

    source_name = "reddit"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rate_limiter.set_interval("reddit", 1.0)
        self._reddit = None

    def _get_reddit(self):
        """Lazy-init the PRAW Reddit instance."""
        if self._reddit is not None:
            return self._reddit
        if not self.config.reddit_client_id:
            raise RuntimeError("REDDIT_CLIENT_ID not configured")
        try:
            import praw

            self._reddit = praw.Reddit(
                client_id=self.config.reddit_client_id,
                client_secret=self.config.reddit_client_secret,
                user_agent=self.config.reddit_user_agent,
            )
            return self._reddit
        except ImportError:
            raise RuntimeError("praw not installed — run: pip install praw")

    def collect(
        self,
        entity_name: str,
        post_limit: int = 100,
        **kwargs,
    ) -> list[MetricTimeSeries]:
        """Collect subreddit subscriber count and recent engagement metrics.

        Args:
            entity_name: Subreddit name (without r/ prefix).
            post_limit: Number of recent posts to analyze for engagement.

        Returns:
            MetricTimeSeries with current subscriber count and engagement data.
        """
        cache_key = self._cache_key(entity_name)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return self._deserialize(cached, entity_name)

        try:
            reddit = self._get_reddit()
        except RuntimeError as e:
            logger.error("Reddit collector unavailable: %s", e)
            return []

        self.rate_limiter.wait("reddit")

        try:
            subreddit = reddit.subreddit(entity_name)
            subscriber_count = subreddit.subscribers
            now = datetime.now(timezone.utc)

            results = []

            # Current subscriber snapshot
            sub_series = MetricTimeSeries(
                platform=Platform.REDDIT,
                metric_type=MetricType.SUBREDDIT_MEMBERS,
                entity_name=entity_name,
                points=[
                    TimeSeriesPoint(
                        timestamp=now, value=float(subscriber_count), source="reddit_api"
                    )
                ],
            )
            results.append(sub_series)

            # Engagement metrics from recent posts
            engagement_points = []
            for post in subreddit.hot(limit=post_limit):
                created = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
                # Engagement score: upvotes + comments weighted
                engagement = post.score + post.num_comments * 2
                engagement_points.append(
                    TimeSeriesPoint(
                        timestamp=created, value=float(engagement), source="reddit_api"
                    )
                )

            if engagement_points:
                engagement_series = MetricTimeSeries(
                    platform=Platform.REDDIT,
                    metric_type=MetricType.VIEWS,  # reuse as engagement proxy
                    entity_name=entity_name,
                    points=sorted(engagement_points, key=lambda p: p.timestamp),
                )
                results.append(engagement_series)

            # Cache the results
            self.cache.set(cache_key, self._serialize(results), ttl=1800)
            return results

        except Exception as e:
            logger.error("Reddit collection error for r/%s: %s", entity_name, e)
            return []

    @staticmethod
    def _serialize(series_list: list[MetricTimeSeries]) -> list[dict]:
        return [
            {
                "metric_type": s.metric_type.value,
                "entity_name": s.entity_name,
                "points": [
                    {"timestamp": p.timestamp.isoformat(), "value": p.value, "source": p.source}
                    for p in s.points
                ],
            }
            for s in series_list
        ]

    @staticmethod
    def _deserialize(data: list[dict], entity_name: str) -> list[MetricTimeSeries]:
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
                    platform=Platform.REDDIT,
                    metric_type=MetricType(entry["metric_type"]),
                    entity_name=entity_name,
                    points=points,
                )
            )
        return results

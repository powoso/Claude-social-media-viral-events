"""Configuration and shared types for the prediction markets system."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum


class Platform(str, Enum):
    YOUTUBE = "youtube"
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"
    TWITTER = "twitter"
    TWITCH = "twitch"
    SPOTIFY = "spotify"
    APP_STORE = "app_store"
    REDDIT = "reddit"


class MetricType(str, Enum):
    FOLLOWERS = "followers"
    SUBSCRIBERS = "subscribers"
    VIEWS = "views"
    DOWNLOADS = "downloads"
    RANKING = "ranking"
    STREAMS = "streams"
    SUBREDDIT_MEMBERS = "subreddit_members"
    PAGEVIEWS = "pageviews"


class GrowthPattern(str, Enum):
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGISTIC = "logistic"
    VIRAL = "viral"
    PLATEAU = "plateau"
    DECLINING = "declining"


@dataclass
class APIConfig:
    """API keys and configuration loaded from environment."""

    socialblade_api_key: str = field(
        default_factory=lambda: os.environ.get("SOCIALBLADE_API_KEY", "")
    )
    reddit_client_id: str = field(
        default_factory=lambda: os.environ.get("REDDIT_CLIENT_ID", "")
    )
    reddit_client_secret: str = field(
        default_factory=lambda: os.environ.get("REDDIT_CLIENT_SECRET", "")
    )
    reddit_user_agent: str = field(
        default_factory=lambda: os.environ.get("REDDIT_USER_AGENT", "smm/0.1")
    )
    google_trends_proxy: str = field(
        default_factory=lambda: os.environ.get("GOOGLE_TRENDS_PROXY", "")
    )
    wayback_rate_limit: float = 1.0  # seconds between requests
    cache_dir: str = field(
        default_factory=lambda: os.environ.get("SMM_CACHE_DIR", ".smm_cache")
    )
    request_timeout: int = 30

    def validate(self) -> list[str]:
        """Return list of missing but recommended API keys."""
        warnings = []
        if not self.socialblade_api_key:
            warnings.append("SOCIALBLADE_API_KEY not set — will fall back to scraping")
        if not self.reddit_client_id:
            warnings.append("REDDIT_CLIENT_ID not set — Reddit collector disabled")
        return warnings

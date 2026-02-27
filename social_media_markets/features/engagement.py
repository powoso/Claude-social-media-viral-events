"""Engagement rate analysis and content cadence features."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from social_media_markets.models import MetricTimeSeries

logger = logging.getLogger(__name__)


@dataclass
class EngagementFeatures:
    """Extracted engagement and cadence features."""

    # Engagement rate trends
    avg_engagement_rate: float  # mean engagement per post/video
    engagement_trend: float  # slope of engagement over time (positive = improving)
    engagement_volatility: float  # coefficient of variation

    # Content cadence
    avg_post_interval_days: float  # average days between posts
    cadence_consistency: float  # 0-1, how regular the posting schedule is
    recent_cadence_change: float  # ratio of recent vs historical cadence

    # Growth efficiency
    engagement_to_growth_ratio: float  # how well engagement converts to growth
    viral_hit_rate: float  # fraction of content in top 10% of engagement


class EngagementAnalyzer:
    """Analyzes engagement patterns and content cadence as growth predictors.

    Key insight: declining engagement rate often predicts growth slowdown
    before it shows in follower counts. Content cadence changes signal
    creator burnout or pivot.
    """

    def analyze(
        self,
        engagement_series: MetricTimeSeries,
        follower_series: MetricTimeSeries | None = None,
    ) -> EngagementFeatures | None:
        """Analyze engagement patterns and their relationship to growth.

        Args:
            engagement_series: Per-post/video engagement metrics over time.
            follower_series: Optional follower count series for growth correlation.

        Returns:
            EngagementFeatures or None if insufficient data.
        """
        if len(engagement_series.points) < 5:
            return None

        t = engagement_series.days_array
        y = engagement_series.values

        # Engagement rate trends
        avg_rate = float(np.mean(y))
        trend_slope = self._compute_trend(t, y)
        volatility = float(np.std(y) / avg_rate) if avg_rate > 0 else 0.0

        # Content cadence analysis
        intervals = np.diff(t)
        avg_interval = float(np.mean(intervals)) if len(intervals) > 0 else 0.0
        cadence_consistency = self._cadence_regularity(intervals)

        # Recent vs historical cadence
        if len(intervals) >= 6:
            mid = len(intervals) // 2
            recent_avg = np.mean(intervals[mid:])
            historical_avg = np.mean(intervals[:mid])
            recent_change = recent_avg / historical_avg if historical_avg > 0 else 1.0
        else:
            recent_change = 1.0

        # Growth efficiency
        if follower_series and len(follower_series.points) >= 2:
            growth_rate = follower_series.growth_rate(window_days=30) or 0.0
            e2g_ratio = growth_rate / avg_rate if avg_rate > 0 else 0.0
        else:
            e2g_ratio = 0.0

        # Viral hit rate: fraction of content in top 10%
        p90 = np.percentile(y, 90)
        viral_rate = float(np.mean(y >= p90))

        return EngagementFeatures(
            avg_engagement_rate=avg_rate,
            engagement_trend=trend_slope,
            engagement_volatility=volatility,
            avg_post_interval_days=avg_interval,
            cadence_consistency=cadence_consistency,
            recent_cadence_change=float(recent_change),
            engagement_to_growth_ratio=e2g_ratio,
            viral_hit_rate=viral_rate,
        )

    @staticmethod
    def _compute_trend(t: np.ndarray, y: np.ndarray) -> float:
        """Compute normalized linear trend slope."""
        if len(t) < 2 or t[-1] == t[0]:
            return 0.0
        coeffs = np.polyfit(t, y, 1)
        # Normalize by mean value to get relative trend
        mean_y = np.mean(y)
        if mean_y == 0:
            return 0.0
        return float(coeffs[0] / mean_y)

    @staticmethod
    def _cadence_regularity(intervals: np.ndarray) -> float:
        """Compute cadence regularity score (0 = chaotic, 1 = perfectly regular)."""
        if len(intervals) < 2:
            return 0.0
        cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 1.0
        # Convert CV to 0-1 score (CV=0 → 1.0, CV=1 → ~0.37, CV=2 → ~0.14)
        return float(np.exp(-cv))

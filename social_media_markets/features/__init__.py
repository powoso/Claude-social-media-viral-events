"""Feature engineering for social media prediction markets."""

from social_media_markets.features.growth_curves import GrowthCurveFitter
from social_media_markets.features.cross_platform import CrossPlatformMomentum
from social_media_markets.features.seasonality import SeasonalityAnalyzer
from social_media_markets.features.controversy import ControversyDetector
from social_media_markets.features.engagement import EngagementAnalyzer

__all__ = [
    "GrowthCurveFitter",
    "CrossPlatformMomentum",
    "SeasonalityAnalyzer",
    "ControversyDetector",
    "EngagementAnalyzer",
]

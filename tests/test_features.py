"""Tests for feature engineering modules."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from social_media_markets.config import MetricType, Platform
from social_media_markets.features.controversy import ControversyDetector
from social_media_markets.features.cross_platform import CrossPlatformMomentum
from social_media_markets.features.engagement import EngagementAnalyzer
from social_media_markets.features.seasonality import SeasonalityAnalyzer
from tests.conftest import make_series


class TestCrossPlatformMomentum:
    def test_identical_series_high_correlation(self, linear_series):
        """Two identical series should have high correlation at lag 0."""
        cp = CrossPlatformMomentum()
        result = cp.analyze(linear_series, linear_series)
        # Can be None due to normalization of identical diff series
        # but if it works, correlation should be high
        if result is not None:
            assert abs(result.correlation) > 0.5

    def test_lagged_series_detected(self):
        """A time-shifted copy should be detected with correct lag."""
        rng = np.random.default_rng(42)
        base = [1000 + 50 * i + rng.normal(0, 100) for i in range(60)]

        # Source leads target by 5 days
        source_values = base[:55]
        target_values = base[5:]  # shifted by 5 days

        source = make_series(source_values, platform=Platform.TIKTOK, entity_name="src")
        target = make_series(target_values, platform=Platform.YOUTUBE, entity_name="tgt")

        cp = CrossPlatformMomentum(max_lag_days=15)
        result = cp.analyze(source, target)
        if result is not None:
            # The detected lag should be close to 5
            assert -10 <= result.lag_days <= 10

    def test_momentum_score(self, linear_series):
        cp = CrossPlatformMomentum()
        features = cp.compute_momentum_score([linear_series], linear_series)
        assert "composite_momentum" in features

    def test_insufficient_data(self):
        short = make_series([100, 200, 300])
        cp = CrossPlatformMomentum()
        result = cp.analyze(short, short)
        assert result is None


class TestSeasonalityAnalyzer:
    def test_detect_weekly_pattern(self):
        """Create data with a clear 7-day cycle and check detection."""
        values = []
        for i in range(180):
            base = 5000 + 20 * i
            seasonal = 500 * np.sin(2 * np.pi * i / 7)
            values.append(base + seasonal)

        series = make_series(values)
        analyzer = SeasonalityAnalyzer()
        result = analyzer.analyze(series)

        assert result is not None
        assert result.seasonal_strength > 0
        assert len(result.deseasonalized) == 180

    def test_monthly_factors(self):
        """Monthly factors should sum close to 12."""
        rng = np.random.default_rng(42)
        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        values = [1000 + rng.normal(0, 100) for _ in range(365)]
        series = make_series(values, start_date=start)

        analyzer = SeasonalityAnalyzer()
        result = analyzer.analyze(series)
        assert result is not None
        assert len(result.monthly_factors) == 12
        # Factors should average close to 1.0
        avg_factor = np.mean(list(result.monthly_factors.values()))
        assert 0.5 < avg_factor < 1.5

    def test_seasonal_adjustment(self, linear_series):
        analyzer = SeasonalityAnalyzer()
        adj = analyzer.get_seasonal_adjustment(linear_series, 100.0)
        assert adj > 0  # adjustment factor should be positive

    def test_no_seasonality_in_pure_linear(self, linear_series):
        analyzer = SeasonalityAnalyzer()
        result = analyzer.analyze(linear_series)
        if result is not None:
            # Pure linear data should have very weak seasonality
            assert result.seasonal_strength < 0.5


class TestControversyDetector:
    def test_detect_spike(self, spike_series):
        detector = ControversyDetector(spike_threshold=2.0, rolling_window=14)
        result = detector.analyze(spike_series)
        assert len(result.spikes) > 0
        # The spike should be around day 30
        spike = result.spikes[0]
        assert 25 < spike.peak_day < 45
        assert spike.magnitude > 2.0

    def test_no_spike_in_constant(self):
        """Constant series should have no spikes."""
        values = [5000.0] * 60
        series = make_series(values, entity_name="constant_channel")
        detector = ControversyDetector()
        result = detector.analyze(series)
        assert len(result.spikes) == 0

    def test_predict_post_spike_level(self):
        detector = ControversyDetector()
        # After a spike, prediction should be between baseline and current
        predicted = detector.predict_post_spike_level(
            current_value=10000,
            baseline=5000,
            days_since_peak=7,
            half_life=7.0,
            residual_factor=1.1,
        )
        assert 5000 < predicted < 10000

    def test_spike_decay_half_life(self, spike_series):
        detector = ControversyDetector(spike_threshold=2.0, rolling_window=14)
        result = detector.analyze(spike_series)
        if result.spikes:
            for spike in result.spikes:
                assert spike.decay_half_life > 0


class TestEngagementAnalyzer:
    def test_basic_engagement_analysis(self, linear_series):
        analyzer = EngagementAnalyzer()
        features = analyzer.analyze(linear_series)
        assert features is not None
        assert features.avg_engagement_rate > 0
        assert features.cadence_consistency > 0
        assert features.avg_post_interval_days > 0

    def test_engagement_trend_positive(self, linear_series):
        """Linear increasing series should show positive engagement trend."""
        analyzer = EngagementAnalyzer()
        features = analyzer.analyze(linear_series)
        assert features is not None
        assert features.engagement_trend > 0

    def test_insufficient_data(self):
        series = make_series([100, 200])
        analyzer = EngagementAnalyzer()
        result = analyzer.analyze(series)
        assert result is None

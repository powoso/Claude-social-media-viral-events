"""Tests for core data models."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from social_media_markets.config import GrowthPattern, MetricType, Platform
from social_media_markets.models import (
    GrowthCurveFit,
    MarketQuestion,
    MetricTimeSeries,
    PredictionResult,
    TimeSeriesPoint,
)


class TestMetricTimeSeries:
    def test_latest_value(self):
        points = [
            TimeSeriesPoint(datetime(2024, 1, i, tzinfo=timezone.utc), float(i * 100))
            for i in range(1, 6)
        ]
        series = MetricTimeSeries(
            platform=Platform.YOUTUBE,
            metric_type=MetricType.SUBSCRIBERS,
            entity_name="test",
            points=points,
        )
        assert series.latest_value() == 500.0

    def test_growth_rate(self):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        points = [
            TimeSeriesPoint(start + timedelta(days=i), 1000.0 + i * 100)
            for i in range(31)
        ]
        series = MetricTimeSeries(
            platform=Platform.YOUTUBE,
            metric_type=MetricType.SUBSCRIBERS,
            entity_name="test",
            points=points,
        )
        rate = series.growth_rate(window_days=30)
        assert rate is not None
        assert abs(rate - 100.0) < 1.0  # should be ~100/day

    def test_empty_series(self):
        series = MetricTimeSeries(
            platform=Platform.YOUTUBE,
            metric_type=MetricType.SUBSCRIBERS,
            entity_name="empty",
        )
        assert series.latest_value() is None
        assert series.growth_rate() is None
        assert len(series.days_array) == 0

    def test_days_array_normalized(self):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        points = [
            TimeSeriesPoint(start + timedelta(days=i), float(i))
            for i in range(10)
        ]
        series = MetricTimeSeries(
            platform=Platform.YOUTUBE,
            metric_type=MetricType.SUBSCRIBERS,
            entity_name="test",
            points=points,
        )
        days = series.days_array
        assert days[0] == 0.0  # starts at 0
        assert abs(days[-1] - 9.0) < 0.01  # 9 days later


class TestGrowthCurveFit:
    def test_linear_predict(self):
        fit = GrowthCurveFit(
            pattern=GrowthPattern.LINEAR,
            parameters={"slope": 100, "intercept": 1000},
            r_squared=0.99,
            residual_std=10.0,
            aic=100.0,
        )
        t = np.array([0, 10, 20])
        pred = fit.predict(t)
        np.testing.assert_array_almost_equal(pred, [1000, 2000, 3000])

    def test_exponential_predict(self):
        fit = GrowthCurveFit(
            pattern=GrowthPattern.EXPONENTIAL,
            parameters={"a": 1000, "r": 0.1},
            r_squared=0.99,
            residual_std=10.0,
            aic=100.0,
        )
        pred = fit.predict(np.array([0.0]))
        assert abs(pred[0] - 1000.0) < 0.01

    def test_logistic_predict(self):
        fit = GrowthCurveFit(
            pattern=GrowthPattern.LOGISTIC,
            parameters={"K": 100000, "r": 0.1, "t0": 50},
            r_squared=0.99,
            residual_std=100.0,
            aic=100.0,
        )
        # At t=t0, value should be K/2
        pred = fit.predict(np.array([50.0]))
        assert abs(pred[0] - 50000.0) < 1.0


class TestPredictionResult:
    def test_has_edge(self):
        question = MarketQuestion(
            question_text="test",
            platform=Platform.YOUTUBE,
            metric_type=MetricType.SUBSCRIBERS,
            entity_name="test",
            target_value=10000,
            current_market_price=0.5,
        )
        result = PredictionResult(
            question=question,
            probability=0.7,
            confidence_interval=(0.6, 0.8),
            model_probability=0.7,
            edge=0.2,
        )
        assert result.has_edge is True

    def test_no_edge(self):
        question = MarketQuestion(
            question_text="test",
            platform=Platform.YOUTUBE,
            metric_type=MetricType.SUBSCRIBERS,
            entity_name="test",
            target_value=10000,
            current_market_price=0.5,
        )
        result = PredictionResult(
            question=question,
            probability=0.52,
            confidence_interval=(0.4, 0.6),
            model_probability=0.52,
            edge=0.02,
        )
        assert result.has_edge is False

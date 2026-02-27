"""Shared test fixtures for social media prediction markets."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from social_media_markets.config import APIConfig, MetricType, Platform
from social_media_markets.models import MetricTimeSeries, TimeSeriesPoint


def make_series(
    values: list[float],
    start_date: datetime | None = None,
    interval_days: float = 1.0,
    platform: Platform = Platform.YOUTUBE,
    metric_type: MetricType = MetricType.SUBSCRIBERS,
    entity_name: str = "test_channel",
) -> MetricTimeSeries:
    """Helper to create a MetricTimeSeries from a list of values."""
    if start_date is None:
        start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)

    points = [
        TimeSeriesPoint(
            timestamp=start_date + timedelta(days=i * interval_days),
            value=v,
            source="test",
        )
        for i, v in enumerate(values)
    ]
    return MetricTimeSeries(
        platform=platform,
        metric_type=metric_type,
        entity_name=entity_name,
        points=points,
    )


@pytest.fixture
def linear_series() -> MetricTimeSeries:
    """Linear growth: 1000 + 100/day for 90 days."""
    values = [1000 + 100 * i for i in range(90)]
    return make_series(values, entity_name="linear_channel")


@pytest.fixture
def exponential_series() -> MetricTimeSeries:
    """Exponential growth: 1000 * exp(0.02 * t) for 90 days."""
    values = [1000 * np.exp(0.02 * i) for i in range(90)]
    return make_series(values, entity_name="exponential_channel")


@pytest.fixture
def logistic_series() -> MetricTimeSeries:
    """Logistic growth: K=100000, r=0.08, t0=45, for 120 days."""
    K, r, t0 = 100000, 0.08, 45
    values = [K / (1 + np.exp(-r * (t - t0))) for t in range(120)]
    return make_series(values, entity_name="logistic_channel")


@pytest.fixture
def spike_series() -> MetricTimeSeries:
    """Series with an attention spike (controversy/viral event)."""
    rng = np.random.default_rng(42)
    baseline = 5000
    values = []
    for i in range(90):
        if 30 <= i <= 40:
            # Spike period: 5x baseline
            val = baseline * 5 * np.exp(-0.2 * (i - 30))
        else:
            val = baseline + rng.normal(0, 200)
        values.append(max(val, 0))
    return make_series(values, entity_name="spike_channel")


@pytest.fixture
def plateau_series() -> MetricTimeSeries:
    """Series approaching a plateau."""
    ceiling = 50000
    r = 0.03
    values = [ceiling * (1 - np.exp(-r * t)) for t in range(120)]
    return make_series(values, entity_name="plateau_channel")


@pytest.fixture
def api_config() -> APIConfig:
    """Test API configuration (no real keys)."""
    return APIConfig(
        socialblade_api_key="",
        reddit_client_id="",
        reddit_client_secret="",
        cache_dir="/tmp/smm_test_cache",
    )

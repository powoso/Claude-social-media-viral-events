"""Core data models for the prediction markets system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from social_media_markets.config import GrowthPattern, MetricType, Platform


@dataclass
class TimeSeriesPoint:
    """A single observation in a time series."""

    timestamp: datetime
    value: float
    source: str = ""


@dataclass
class MetricTimeSeries:
    """Time series data for a specific metric on a specific platform."""

    platform: Platform
    metric_type: MetricType
    entity_name: str  # channel name, app name, etc.
    points: list[TimeSeriesPoint] = field(default_factory=list)

    @property
    def timestamps(self) -> np.ndarray:
        return np.array([p.timestamp.timestamp() for p in self.points])

    @property
    def values(self) -> np.ndarray:
        return np.array([p.value for p in self.points])

    @property
    def days_array(self) -> np.ndarray:
        """Timestamps normalized to days since first observation."""
        ts = self.timestamps
        if len(ts) == 0:
            return np.array([])
        return (ts - ts[0]) / 86400.0

    def latest_value(self) -> float | None:
        if not self.points:
            return None
        return self.points[-1].value

    def growth_rate(self, window_days: int = 30) -> float | None:
        """Average daily growth rate over the recent window."""
        if len(self.points) < 2:
            return None
        days = self.days_array
        vals = self.values
        mask = days >= (days[-1] - window_days)
        if mask.sum() < 2:
            return None
        window_days_span = days[mask][-1] - days[mask][0]
        if window_days_span == 0:
            return None
        return (vals[mask][-1] - vals[mask][0]) / window_days_span


@dataclass
class GrowthCurveFit:
    """Result of fitting a growth model to time series data."""

    pattern: GrowthPattern
    parameters: dict[str, float]
    r_squared: float
    residual_std: float
    aic: float  # Akaike Information Criterion for model comparison

    def predict(self, days_from_start: np.ndarray) -> np.ndarray:
        """Predict values at given days using the fitted model."""
        p = self.parameters
        t = days_from_start

        if self.pattern == GrowthPattern.LINEAR:
            return p["slope"] * t + p["intercept"]
        elif self.pattern == GrowthPattern.EXPONENTIAL:
            return p["a"] * np.exp(p["r"] * t)
        elif self.pattern == GrowthPattern.LOGISTIC:
            return p["K"] / (1 + np.exp(-p["r"] * (t - p["t0"])))
        elif self.pattern == GrowthPattern.VIRAL:
            # Viral: exponential onset with logistic ceiling
            base = p["K"] / (1 + np.exp(-p["r"] * (t - p["t0"])))
            noise_scale = p.get("noise_scale", 0)
            return base + noise_scale * np.log1p(np.maximum(t - p.get("spike_t", 0), 0))
        elif self.pattern == GrowthPattern.PLATEAU:
            return p["ceiling"] * (1 - np.exp(-p["r"] * t))
        elif self.pattern == GrowthPattern.DECLINING:
            return p["peak"] * np.exp(-p["decay"] * (t - p["peak_t"]))
        else:
            raise ValueError(f"Unknown growth pattern: {self.pattern}")


@dataclass
class MarketQuestion:
    """A prediction market question about a social media milestone."""

    question_text: str
    platform: Platform
    metric_type: MetricType
    entity_name: str
    target_value: float
    deadline: datetime | None = None  # None means "ever"
    current_market_price: float | None = None  # observed market probability

    def __str__(self) -> str:
        return self.question_text


@dataclass
class PredictionResult:
    """The system's probability estimate for a market question."""

    question: MarketQuestion
    probability: float
    confidence_interval: tuple[float, float]  # 90% CI on the probability
    model_probability: float  # raw model output before edge adjustments
    edge: float  # model_probability - market_price (if available)
    growth_fit: GrowthCurveFit | None = None
    features_used: dict[str, float] = field(default_factory=dict)
    reasoning: str = ""

    @property
    def has_edge(self) -> bool:
        return abs(self.edge) > 0.05  # 5% minimum edge threshold

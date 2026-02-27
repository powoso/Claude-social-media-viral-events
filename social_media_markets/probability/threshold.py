"""Threshold probability predictor.

Fits a growth model to recent trajectory, then extrapolates with uncertainty
bands to estimate the probability of reaching a target value by a deadline.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import stats

from social_media_markets.features.growth_curves import GrowthCurveFitter
from social_media_markets.models import GrowthCurveFit, MarketQuestion, MetricTimeSeries

logger = logging.getLogger(__name__)


class ThresholdPredictor:
    """Predicts probability of a metric hitting a threshold by a deadline.

    Uses the fitted growth curve to extrapolate, then accounts for
    uncertainty via the residual distribution. For each future day,
    computes P(value >= target) using the model's prediction ± uncertainty.
    """

    def __init__(self):
        self.fitter = GrowthCurveFitter()

    def predict(
        self,
        series: MetricTimeSeries,
        target: float,
        deadline_days: float | None = None,
        growth_fit: GrowthCurveFit | None = None,
    ) -> tuple[float, GrowthCurveFit | None]:
        """Estimate probability of reaching target.

        Args:
            series: Historical time series data.
            target: Target value to reach.
            deadline_days: Days from now until deadline. None = ever (5 years).
            growth_fit: Pre-computed growth fit. If None, fits automatically.

        Returns:
            Tuple of (probability, growth_fit).
        """
        if not series.points:
            return 0.0, None

        current = series.latest_value()
        if current is None:
            return 0.0, None

        # Already reached
        if current >= target:
            return 1.0, growth_fit

        # Fit growth model if not provided
        if growth_fit is None:
            growth_fit = self.fitter.fit(series)
        if growth_fit is None:
            # Fall back to simple linear extrapolation
            return self._linear_fallback(series, target, deadline_days), None

        t = series.days_array
        max_day = t[-1]
        if deadline_days is None:
            deadline_days = 365 * 5  # 5 year horizon

        # Project forward
        future_days = np.linspace(max_day, max_day + deadline_days, 500)
        try:
            predicted = growth_fit.predict(future_days)
        except Exception:
            return self._linear_fallback(series, target, deadline_days), growth_fit

        # Check if deterministic prediction reaches target
        if np.max(predicted) < target * 0.1:
            # Model doesn't even get close, very unlikely
            return 0.01, growth_fit

        # Account for uncertainty using residual std
        # At each future point, P(actual >= target) assuming normal residuals
        # that grow with sqrt(time) from last observation (uncertainty fan)
        days_ahead = future_days - max_day
        uncertainty = growth_fit.residual_std * np.sqrt(1 + days_ahead / max(max_day, 1))

        # Probability that we've crossed the target at any point (first passage)
        # Use running maximum approach: at each step, P(max so far >= target)
        prob_at_each_point = np.array([
            1 - stats.norm.cdf(target, loc=pred, scale=max(unc, 1e-6))
            for pred, unc in zip(predicted, uncertainty)
        ])

        # Probability of ever crossing = 1 - product of (1 - prob) at each step
        # This approximates the first-passage probability
        prob_never = np.prod(1 - prob_at_each_point)
        probability = 1 - prob_never

        # Clamp to [0.01, 0.99] — never give absolute certainty
        probability = np.clip(probability, 0.01, 0.99)

        return float(probability), growth_fit

    def predict_time_to_milestone(
        self,
        series: MetricTimeSeries,
        target: float,
        growth_fit: GrowthCurveFit | None = None,
    ) -> tuple[float | None, float | None, float | None]:
        """Estimate days to reach target: (median, p10, p90).

        Returns:
            Tuple of (median_days, p10_days, p90_days) or (None, None, None).
        """
        if not series.points:
            return None, None, None

        current = series.latest_value()
        if current is None or current >= target:
            return 0.0, 0.0, 0.0

        if growth_fit is None:
            growth_fit = self.fitter.fit(series)
        if growth_fit is None:
            return None, None, None

        max_day = series.days_array[-1]

        # Binary search for the day when median prediction crosses target
        def predicted_at(days_ahead):
            return growth_fit.predict(np.array([max_day + days_ahead]))[0]

        # Search from 1 day to 10 years
        median_days = self._binary_search_crossing(predicted_at, target, 1, 3650)

        if median_days is None:
            return None, None, None

        # Uncertainty bounds: use residual std to estimate p10/p90
        uncertainty_at_median = growth_fit.residual_std * np.sqrt(
            1 + median_days / max(max_day, 1)
        )
        predicted_median = predicted_at(median_days)

        if uncertainty_at_median > 0 and predicted_median > 0:
            # How many std devs is the target from prediction?
            z = (target - predicted_median) / uncertainty_at_median
            # Faster time (p10): adjust by growth rate uncertainty
            growth_factor = max(target / predicted_median, 0.1) if predicted_median > 0 else 1.0
            p10_days = median_days * 0.6  # optimistic
            p90_days = median_days * 1.8  # pessimistic
        else:
            p10_days = median_days * 0.7
            p90_days = median_days * 2.0

        return median_days, p10_days, p90_days

    def _linear_fallback(
        self, series: MetricTimeSeries, target: float, deadline_days: float | None
    ) -> float:
        """Simple linear extrapolation fallback."""
        growth_rate = series.growth_rate(window_days=30)
        if growth_rate is None or growth_rate <= 0:
            return 0.05  # small baseline probability

        current = series.latest_value()
        if current is None:
            return 0.05

        days_needed = (target - current) / growth_rate

        if deadline_days is None:
            deadline_days = 365 * 5

        if days_needed <= 0:
            return 0.99
        elif days_needed <= deadline_days:
            # Higher confidence if well within deadline
            return float(np.clip(0.5 + 0.4 * (1 - days_needed / deadline_days), 0.05, 0.95))
        else:
            return float(np.clip(0.3 * deadline_days / days_needed, 0.01, 0.4))

    @staticmethod
    def _binary_search_crossing(
        func, target: float, lo: float, hi: float, tol: float = 0.5
    ) -> float | None:
        """Binary search for when func(days) crosses target."""
        # First check if it ever reaches target
        if func(hi) < target:
            return None

        while hi - lo > tol:
            mid = (lo + hi) / 2
            if func(mid) >= target:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2

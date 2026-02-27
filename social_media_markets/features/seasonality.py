"""Seasonality analysis for social media metrics.

Detects and quantifies seasonal patterns: holiday content, summer viewership
cycles, back-to-school effects, and platform-specific seasonality.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import signal

from social_media_markets.models import MetricTimeSeries

logger = logging.getLogger(__name__)


@dataclass
class SeasonalComponent:
    """A detected seasonal component in the data."""

    period_days: float
    amplitude: float  # as fraction of mean
    phase_days: float  # days offset from origin
    power: float  # spectral power (strength of signal)


@dataclass
class SeasonalityResult:
    """Full seasonality analysis result."""

    components: list[SeasonalComponent]
    seasonal_strength: float  # 0-1, how much variance is explained by seasonality
    deseasonalized: np.ndarray  # residuals after removing seasonal components
    monthly_factors: dict[int, float]  # month (1-12) → multiplicative factor


class SeasonalityAnalyzer:
    """Detects and models seasonal patterns in social media time series.

    Common patterns:
    - Weekly: 7-day cycle (weekday vs weekend engagement)
    - Monthly: content creator upload cadence effects
    - Quarterly: holiday seasons (Nov-Dec spike, summer dip for some verticals)
    - Annual: back-to-school, New Year's, summer break
    """

    def __init__(self, min_points: int = 30):
        self.min_points = min_points

    def analyze(self, series: MetricTimeSeries) -> SeasonalityResult | None:
        """Perform full seasonality decomposition on a time series.

        Args:
            series: Time series with at least min_points data points.

        Returns:
            SeasonalityResult or None if insufficient data.
        """
        if len(series.points) < self.min_points:
            return None

        t = series.days_array
        y = series.values

        # Detrend: remove linear trend to isolate seasonal signal
        trend = np.polyval(np.polyfit(t, y, 1), t)
        detrended = y - trend

        # Find dominant periodic components via FFT
        components = self._find_periodic_components(detrended, t)

        # Compute monthly factors from raw data
        monthly_factors = self._compute_monthly_factors(series)

        # Compute seasonal strength (fraction of variance explained)
        if components:
            seasonal_signal = self._reconstruct_seasonal(t, components)
            ss_seasonal = np.sum(seasonal_signal ** 2)
            ss_total = np.sum(detrended ** 2)
            strength = ss_seasonal / ss_total if ss_total > 0 else 0.0
            deseasonalized = y - seasonal_signal - trend + np.mean(y)
        else:
            strength = 0.0
            deseasonalized = y.copy()

        return SeasonalityResult(
            components=components,
            seasonal_strength=min(strength, 1.0),
            deseasonalized=deseasonalized,
            monthly_factors=monthly_factors,
        )

    def get_seasonal_adjustment(
        self, series: MetricTimeSeries, target_day: float
    ) -> float:
        """Get the multiplicative seasonal adjustment for a future day.

        Returns a factor to multiply the trend prediction by.
        E.g., 1.15 means the seasonal pattern adds 15% at that time.
        """
        result = self.analyze(series)
        if result is None or not result.components:
            return 1.0

        adjustment = 0.0
        mean_val = np.mean(series.values)
        if mean_val == 0:
            return 1.0

        for comp in result.components:
            phase = 2 * np.pi * (target_day - comp.phase_days) / comp.period_days
            adjustment += comp.amplitude * np.cos(phase)

        return 1.0 + adjustment / mean_val

    def _find_periodic_components(
        self, detrended: np.ndarray, t: np.ndarray
    ) -> list[SeasonalComponent]:
        """Use FFT to find dominant periodic components."""
        n = len(detrended)
        if n < 14:  # need at least 2 weeks for meaningful periodicity
            return []

        # Compute power spectrum
        dt = np.median(np.diff(t))  # typical time step in days
        if dt <= 0:
            return []

        freq = np.fft.rfftfreq(n, d=dt)
        fft_vals = np.fft.rfft(detrended)
        power = np.abs(fft_vals) ** 2

        # Skip DC component (index 0) and very low frequencies
        min_freq = 1.0 / (t[-1] - t[0]) if (t[-1] - t[0]) > 0 else 0
        valid = freq > min_freq * 2

        if not np.any(valid):
            return []

        # Find peaks in the power spectrum
        valid_power = power.copy()
        valid_power[~valid] = 0

        # Threshold: peaks must be > 3x median power
        median_power = np.median(valid_power[valid])
        threshold = median_power * 3

        peaks, properties = signal.find_peaks(valid_power, height=threshold, distance=2)

        components = []
        for peak in peaks:
            if freq[peak] == 0:
                continue
            period = 1.0 / freq[peak]
            # Filter to reasonable periods (3 days to 400 days)
            if period < 3 or period > 400:
                continue

            amplitude = 2 * np.abs(fft_vals[peak]) / n
            phase = -np.angle(fft_vals[peak]) / (2 * np.pi) * period

            components.append(
                SeasonalComponent(
                    period_days=period,
                    amplitude=amplitude,
                    phase_days=phase % period,
                    power=float(valid_power[peak]),
                )
            )

        # Sort by power (strongest first) and keep top 3
        components.sort(key=lambda c: c.power, reverse=True)
        return components[:3]

    @staticmethod
    def _compute_monthly_factors(series: MetricTimeSeries) -> dict[int, float]:
        """Compute multiplicative monthly factors from data.

        Factor > 1 means that month tends to be above average.
        """
        monthly_values: dict[int, list[float]] = {}
        for point in series.points:
            month = point.timestamp.month
            monthly_values.setdefault(month, []).append(point.value)

        overall_mean = np.mean(series.values)
        if overall_mean == 0:
            return {m: 1.0 for m in range(1, 13)}

        factors = {}
        for month in range(1, 13):
            if month in monthly_values and monthly_values[month]:
                factors[month] = np.mean(monthly_values[month]) / overall_mean
            else:
                factors[month] = 1.0

        return factors

    @staticmethod
    def _reconstruct_seasonal(
        t: np.ndarray, components: list[SeasonalComponent]
    ) -> np.ndarray:
        """Reconstruct the seasonal signal from components."""
        seasonal = np.zeros_like(t)
        for comp in components:
            phase = 2 * np.pi * (t - comp.phase_days) / comp.period_days
            seasonal += comp.amplitude * np.cos(phase)
        return seasonal

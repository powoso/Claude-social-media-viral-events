"""Controversy and attention spike detection and decay modeling.

Models the attention lifecycle: spike detection → peak estimation →
decay rate fitting → long-term residual level prediction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize

from social_media_markets.models import MetricTimeSeries

logger = logging.getLogger(__name__)


@dataclass
class AttentionSpike:
    """A detected attention spike event."""

    peak_day: float  # days from series start
    peak_value: float
    baseline: float  # pre-spike level
    magnitude: float  # peak / baseline ratio
    rise_days: float  # days from baseline to peak
    decay_half_life: float  # days for spike to decay to 50%
    residual_factor: float  # long-term level as fraction of peak (often > 1.0 of baseline)


@dataclass
class ControversyAnalysis:
    """Full analysis of attention spikes and controversy patterns."""

    spikes: list[AttentionSpike] = field(default_factory=list)
    avg_spike_magnitude: float = 0.0
    avg_decay_half_life: float = 0.0
    spike_frequency: float = 0.0  # spikes per 30 days
    current_is_spiking: bool = False
    predicted_decay_target: float | None = None  # where current spike will settle


class ControversyDetector:
    """Detects and models attention spikes from controversy/viral events.

    Key market insight: controversy creates predictable attention spikes
    with predictable exponential decay. Markets often overcorrect on
    both the spike and the decay, creating trading opportunities.
    """

    def __init__(
        self,
        spike_threshold: float = 2.0,  # std devs above rolling mean
        rolling_window: int = 14,  # days for rolling baseline
    ):
        self.spike_threshold = spike_threshold
        self.rolling_window = rolling_window

    def analyze(self, series: MetricTimeSeries) -> ControversyAnalysis:
        """Detect and model attention spikes in a time series.

        Args:
            series: Time series data (pageviews, search interest, engagement, etc.)

        Returns:
            ControversyAnalysis with detected spikes and their characteristics.
        """
        if len(series.points) < self.rolling_window + 5:
            return ControversyAnalysis()

        t = series.days_array
        y = series.values

        # Compute rolling baseline and rolling std
        baseline, rolling_std = self._rolling_stats(y)

        # Detect spikes: points where value > baseline + threshold * std
        spike_mask = y > (baseline + self.spike_threshold * rolling_std)

        # Group consecutive spike days into events
        spike_groups = self._group_spikes(spike_mask, t, y, baseline)

        # Fit decay model to each spike
        spikes = []
        for group in spike_groups:
            spike = self._fit_spike(group, t, y, baseline)
            if spike is not None:
                spikes.append(spike)

        analysis = ControversyAnalysis(spikes=spikes)

        if spikes:
            analysis.avg_spike_magnitude = np.mean([s.magnitude for s in spikes])
            analysis.avg_decay_half_life = np.mean([s.decay_half_life for s in spikes])
            time_span = t[-1] - t[0]
            if time_span > 0:
                analysis.spike_frequency = len(spikes) / (time_span / 30)

        # Check if currently in a spike
        if spike_mask[-1] and spikes:
            analysis.current_is_spiking = True
            last_spike = spikes[-1]
            # Predict where current spike will settle
            days_from_peak = t[-1] - last_spike.peak_day
            decay = last_spike.peak_value * np.exp(
                -0.693 * days_from_peak / max(last_spike.decay_half_life, 0.1)
            )
            analysis.predicted_decay_target = last_spike.baseline * last_spike.residual_factor

        return analysis

    def predict_post_spike_level(
        self,
        current_value: float,
        baseline: float,
        days_since_peak: float,
        half_life: float = 7.0,
        residual_factor: float = 1.1,
    ) -> float:
        """Predict the metric value after a spike decays.

        Useful for predicting where followers/views will settle after a
        viral event or controversy.

        Args:
            current_value: Current observed value.
            baseline: Pre-spike baseline level.
            days_since_peak: Days since the attention peak.
            half_life: Decay half-life in days.
            residual_factor: Long-term level as multiple of baseline.

        Returns:
            Predicted settled value.
        """
        spike_component = (current_value - baseline * residual_factor)
        if spike_component <= 0:
            return current_value  # already below residual level

        # Exponential decay of the spike component
        future_spike = spike_component * np.exp(-0.693 / max(half_life, 0.1))
        return baseline * residual_factor + max(future_spike, 0)

    def _rolling_stats(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute rolling mean and std with the configured window."""
        n = len(y)
        baseline = np.zeros(n)
        rolling_std = np.zeros(n)

        for i in range(n):
            start = max(0, i - self.rolling_window)
            window = y[start:i] if i > 0 else y[:1]
            baseline[i] = np.mean(window)
            rolling_std[i] = np.std(window) if len(window) > 1 else np.std(y[:self.rolling_window])

        return baseline, rolling_std

    @staticmethod
    def _group_spikes(
        mask: np.ndarray, t: np.ndarray, y: np.ndarray, baseline: np.ndarray
    ) -> list[dict]:
        """Group consecutive spike points into spike events."""
        groups = []
        in_spike = False
        current_group = None

        for i in range(len(mask)):
            if mask[i] and not in_spike:
                in_spike = True
                current_group = {
                    "start_idx": i,
                    "peak_idx": i,
                    "peak_val": y[i],
                }
            elif mask[i] and in_spike:
                if y[i] > current_group["peak_val"]:
                    current_group["peak_idx"] = i
                    current_group["peak_val"] = y[i]
            elif not mask[i] and in_spike:
                in_spike = False
                current_group["end_idx"] = i
                groups.append(current_group)
                current_group = None

        # Handle spike at end of series
        if in_spike and current_group:
            current_group["end_idx"] = len(mask) - 1
            groups.append(current_group)

        return groups

    def _fit_spike(
        self, group: dict, t: np.ndarray, y: np.ndarray, baseline: np.ndarray
    ) -> AttentionSpike | None:
        """Fit an exponential decay model to a spike event."""
        peak_idx = group["peak_idx"]
        start_idx = group["start_idx"]
        end_idx = group["end_idx"]
        peak_val = group["peak_val"]
        pre_baseline = baseline[start_idx]

        if pre_baseline <= 0 or peak_val <= pre_baseline:
            return None

        magnitude = peak_val / pre_baseline
        rise_days = t[peak_idx] - t[start_idx]

        # Fit exponential decay to the post-peak portion
        decay_end = min(end_idx + self.rolling_window, len(t) - 1)
        post_peak_t = t[peak_idx:decay_end + 1] - t[peak_idx]
        post_peak_y = y[peak_idx:decay_end + 1]

        if len(post_peak_t) < 3:
            half_life = 7.0  # default assumption
            residual_factor = 1.05
        else:
            half_life, residual_factor = self._fit_decay(
                post_peak_t, post_peak_y, peak_val, pre_baseline
            )

        return AttentionSpike(
            peak_day=t[peak_idx],
            peak_value=peak_val,
            baseline=pre_baseline,
            magnitude=magnitude,
            rise_days=max(rise_days, 0.5),
            decay_half_life=half_life,
            residual_factor=residual_factor,
        )

    @staticmethod
    def _fit_decay(
        t_post: np.ndarray,
        y_post: np.ndarray,
        peak_val: float,
        baseline: float,
    ) -> tuple[float, float]:
        """Fit exponential decay: y = residual + (peak - residual) * exp(-lambda * t)."""

        def model(t, half_life, residual):
            decay = (peak_val - residual) * np.exp(-0.693 * t / max(half_life, 0.1))
            return residual + decay

        try:
            popt, _ = optimize.curve_fit(
                model, t_post, y_post,
                p0=[7.0, baseline * 1.05],
                bounds=([0.5, baseline * 0.5], [180, peak_val]),
                maxfev=3000,
            )
            half_life = popt[0]
            residual_factor = popt[1] / baseline if baseline > 0 else 1.0
        except (RuntimeError, ValueError):
            half_life = 7.0
            residual_factor = 1.05

        return half_life, residual_factor

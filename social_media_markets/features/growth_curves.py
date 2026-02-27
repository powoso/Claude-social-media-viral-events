"""Growth curve classification and fitting.

Fits multiple growth models (linear, exponential, logistic, viral, plateau,
declining) to time series data and selects the best fit using AIC.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
from scipy import optimize

from social_media_markets.config import GrowthPattern
from social_media_markets.models import GrowthCurveFit, MetricTimeSeries

logger = logging.getLogger(__name__)


class GrowthCurveFitter:
    """Fits growth models to metric time series and classifies the growth pattern.

    The fitter tries multiple parametric models and selects the best one
    based on Akaike Information Criterion (AIC), which penalizes model
    complexity to avoid overfitting.
    """

    def __init__(self, min_points: int = 5):
        self.min_points = min_points

    def fit(self, series: MetricTimeSeries) -> GrowthCurveFit | None:
        """Fit the best growth model to a time series.

        Tries all candidate models and returns the one with the lowest AIC.

        Args:
            series: Time series data to fit.

        Returns:
            Best fitting GrowthCurveFit, or None if insufficient data.
        """
        if len(series.points) < self.min_points:
            logger.warning(
                "Insufficient data for %s (%d points, need %d)",
                series.entity_name,
                len(series.points),
                self.min_points,
            )
            return None

        t = series.days_array
        y = series.values

        # Normalize time to improve numerical stability
        t_norm = t / max(t[-1], 1.0)
        y_scale = max(np.abs(y).max(), 1.0)
        y_norm = y / y_scale

        candidates = []
        for pattern in GrowthPattern:
            try:
                fit = self._fit_model(pattern, t, t_norm, y, y_norm, y_scale)
                if fit is not None:
                    candidates.append(fit)
            except Exception as e:
                logger.debug("Failed to fit %s model: %s", pattern, e)

        if not candidates:
            logger.warning("No models fit for %s", series.entity_name)
            return None

        # Select model with lowest AIC
        best = min(candidates, key=lambda f: f.aic)
        logger.info(
            "Best fit for %s: %s (R²=%.4f, AIC=%.1f)",
            series.entity_name,
            best.pattern.value,
            best.r_squared,
            best.aic,
        )
        return best

    def fit_all(self, series: MetricTimeSeries) -> list[GrowthCurveFit]:
        """Fit all growth models and return them sorted by AIC (best first)."""
        if len(series.points) < self.min_points:
            return []

        t = series.days_array
        y = series.values
        t_norm = t / max(t[-1], 1.0)
        y_scale = max(np.abs(y).max(), 1.0)
        y_norm = y / y_scale

        candidates = []
        for pattern in GrowthPattern:
            try:
                fit = self._fit_model(pattern, t, t_norm, y, y_norm, y_scale)
                if fit is not None:
                    candidates.append(fit)
            except Exception:
                continue

        return sorted(candidates, key=lambda f: f.aic)

    def _fit_model(
        self,
        pattern: GrowthPattern,
        t: np.ndarray,
        t_norm: np.ndarray,
        y: np.ndarray,
        y_norm: np.ndarray,
        y_scale: float,
    ) -> GrowthCurveFit | None:
        """Fit a specific growth model to the data."""
        n = len(y)

        if pattern == GrowthPattern.LINEAR:
            return self._fit_linear(t, y, n)
        elif pattern == GrowthPattern.EXPONENTIAL:
            return self._fit_exponential(t, y, n)
        elif pattern == GrowthPattern.LOGISTIC:
            return self._fit_logistic(t, y, y_scale, n)
        elif pattern == GrowthPattern.VIRAL:
            return self._fit_viral(t, y, y_scale, n)
        elif pattern == GrowthPattern.PLATEAU:
            return self._fit_plateau(t, y, y_scale, n)
        elif pattern == GrowthPattern.DECLINING:
            return self._fit_declining(t, y, n)
        return None

    def _fit_linear(self, t: np.ndarray, y: np.ndarray, n: int) -> GrowthCurveFit:
        """Fit y = slope * t + intercept."""
        coeffs = np.polyfit(t, y, 1)
        slope, intercept = coeffs
        y_pred = slope * t + intercept
        return self._build_result(
            GrowthPattern.LINEAR,
            {"slope": slope, "intercept": intercept},
            y, y_pred, n, k=2,
        )

    def _fit_exponential(
        self, t: np.ndarray, y: np.ndarray, n: int
    ) -> GrowthCurveFit | None:
        """Fit y = a * exp(r * t)."""
        if np.any(y <= 0):
            return None

        # Initial guess from log-linear fit
        log_y = np.log(y)
        coeffs = np.polyfit(t, log_y, 1)
        r0, log_a0 = coeffs

        # Reject if growth rate is unreasonably large
        if abs(r0) > 1.0:
            return None

        def model(t, a, r):
            return a * np.exp(r * t)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                popt, _ = optimize.curve_fit(
                    model, t, y,
                    p0=[np.exp(log_a0), r0],
                    maxfev=5000,
                    bounds=([0, -0.5], [y.max() * 10, 0.5]),
                )
            except (optimize.OptimizeWarning, RuntimeError, ValueError):
                return None

        y_pred = model(t, *popt)
        return self._build_result(
            GrowthPattern.EXPONENTIAL,
            {"a": popt[0], "r": popt[1]},
            y, y_pred, n, k=2,
        )

    def _fit_logistic(
        self, t: np.ndarray, y: np.ndarray, y_scale: float, n: int
    ) -> GrowthCurveFit | None:
        """Fit y = K / (1 + exp(-r * (t - t0)))."""

        def model(t, K, r, t0):
            return K / (1 + np.exp(-r * (t - t0)))

        K0 = y[-1] * 1.5  # carrying capacity guess: 1.5x current
        r0 = 0.01
        t0_guess = t[len(t) // 2]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                popt, _ = optimize.curve_fit(
                    model, t, y,
                    p0=[K0, r0, t0_guess],
                    maxfev=10000,
                    bounds=(
                        [y.max() * 0.5, 1e-6, t[0] - 365],
                        [y.max() * 100, 1.0, t[-1] + 365],
                    ),
                )
            except (optimize.OptimizeWarning, RuntimeError, ValueError):
                return None

        y_pred = model(t, *popt)
        return self._build_result(
            GrowthPattern.LOGISTIC,
            {"K": popt[0], "r": popt[1], "t0": popt[2]},
            y, y_pred, n, k=3,
        )

    def _fit_viral(
        self, t: np.ndarray, y: np.ndarray, y_scale: float, n: int
    ) -> GrowthCurveFit | None:
        """Fit viral model: logistic base + log spike.

        y = K / (1 + exp(-r * (t - t0))) + noise_scale * log(1 + max(t - spike_t, 0))
        """

        def model(t, K, r, t0, noise_scale, spike_t):
            base = K / (1 + np.exp(-r * (t - t0)))
            spike = noise_scale * np.log1p(np.maximum(t - spike_t, 0))
            return base + spike

        K0 = y[-1] * 1.2
        r0 = 0.05
        t0_guess = t[len(t) // 2]
        # Detect potential spike point by max second derivative
        if n > 3:
            dy = np.diff(y)
            ddy = np.diff(dy)
            spike_idx = np.argmax(ddy) + 1
            spike_t0 = t[spike_idx]
        else:
            spike_t0 = t[len(t) // 2]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                popt, _ = optimize.curve_fit(
                    model, t, y,
                    p0=[K0, r0, t0_guess, y_scale * 0.1, spike_t0],
                    maxfev=15000,
                    bounds=(
                        [y.max() * 0.3, 1e-6, t[0] - 365, 0, t[0]],
                        [y.max() * 100, 2.0, t[-1] + 365, y_scale * 10, t[-1]],
                    ),
                )
            except (optimize.OptimizeWarning, RuntimeError, ValueError):
                return None

        y_pred = model(t, *popt)
        return self._build_result(
            GrowthPattern.VIRAL,
            {
                "K": popt[0], "r": popt[1], "t0": popt[2],
                "noise_scale": popt[3], "spike_t": popt[4],
            },
            y, y_pred, n, k=5,
        )

    def _fit_plateau(
        self, t: np.ndarray, y: np.ndarray, y_scale: float, n: int
    ) -> GrowthCurveFit | None:
        """Fit y = ceiling * (1 - exp(-r * t))."""

        def model(t, ceiling, r):
            return ceiling * (1 - np.exp(-r * t))

        ceiling0 = y[-1] * 1.1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                popt, _ = optimize.curve_fit(
                    model, t, y,
                    p0=[ceiling0, 0.01],
                    maxfev=5000,
                    bounds=([y.max() * 0.5, 1e-6], [y.max() * 10, 1.0]),
                )
            except (optimize.OptimizeWarning, RuntimeError, ValueError):
                return None

        y_pred = model(t, *popt)
        return self._build_result(
            GrowthPattern.PLATEAU,
            {"ceiling": popt[0], "r": popt[1]},
            y, y_pred, n, k=2,
        )

    def _fit_declining(
        self, t: np.ndarray, y: np.ndarray, n: int
    ) -> GrowthCurveFit | None:
        """Fit y = peak * exp(-decay * (t - peak_t))."""
        peak_idx = np.argmax(y)
        if peak_idx == 0 or peak_idx == n - 1:
            # Only valid if peak is not at the edges or at beginning
            if peak_idx == 0 and y[-1] >= y[0] * 0.9:
                return None  # not actually declining

        peak_val = y[peak_idx]
        peak_t = t[peak_idx]

        def model(t, peak, decay, pt):
            return peak * np.exp(-decay * np.maximum(t - pt, 0))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                popt, _ = optimize.curve_fit(
                    model, t, y,
                    p0=[peak_val, 0.01, peak_t],
                    maxfev=5000,
                    bounds=(
                        [peak_val * 0.5, 1e-6, t[0]],
                        [peak_val * 2, 1.0, t[-1]],
                    ),
                )
            except (optimize.OptimizeWarning, RuntimeError, ValueError):
                return None

        y_pred = model(t, *popt)
        return self._build_result(
            GrowthPattern.DECLINING,
            {"peak": popt[0], "decay": popt[1], "peak_t": popt[2]},
            y, y_pred, n, k=3,
        )

    @staticmethod
    def _build_result(
        pattern: GrowthPattern,
        params: dict[str, float],
        y_actual: np.ndarray,
        y_pred: np.ndarray,
        n: int,
        k: int,
    ) -> GrowthCurveFit:
        """Compute goodness-of-fit metrics and build the result."""
        ss_res = np.sum((y_actual - y_pred) ** 2)
        ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        residual_std = np.sqrt(ss_res / max(n - k, 1))

        # AIC: n * ln(ss_res/n) + 2*k (lower is better)
        mse = ss_res / n
        if mse > 0:
            aic = n * np.log(mse) + 2 * k
        else:
            aic = -np.inf

        return GrowthCurveFit(
            pattern=pattern,
            parameters=params,
            r_squared=float(r_squared),
            residual_std=float(residual_std),
            aic=float(aic),
        )

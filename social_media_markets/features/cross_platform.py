"""Cross-platform momentum analysis.

Detects whether virality on one platform (e.g., TikTok) predicts growth
on another (e.g., YouTube subscribers), and computes lead/lag correlations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import signal, stats

from social_media_markets.models import MetricTimeSeries

logger = logging.getLogger(__name__)


@dataclass
class CrossPlatformSignal:
    """Result of cross-platform correlation analysis."""

    source_platform: str
    target_platform: str
    lag_days: int  # positive = source leads target
    correlation: float
    p_value: float
    granger_f_stat: float | None = None
    granger_p_value: float | None = None

    @property
    def is_leading_indicator(self) -> bool:
        return self.lag_days > 0 and self.correlation > 0.3 and self.p_value < 0.05


class CrossPlatformMomentum:
    """Analyzes momentum transfer between platforms.

    Key insight: virality often cascades across platforms with predictable
    delays. TikTok → YouTube is typically 3-14 days. Google Trends often
    leads follower count changes by 1-7 days.
    """

    def __init__(self, max_lag_days: int = 30):
        self.max_lag_days = max_lag_days

    def analyze(
        self,
        source: MetricTimeSeries,
        target: MetricTimeSeries,
    ) -> CrossPlatformSignal | None:
        """Compute cross-correlation between two time series.

        Aligns the series to daily frequency via interpolation, then
        finds the optimal lag and correlation strength.

        Args:
            source: The potential leading indicator series.
            target: The series we're trying to predict.

        Returns:
            CrossPlatformSignal with the optimal lag, or None if insufficient data.
        """
        if len(source.points) < 10 or len(target.points) < 10:
            return None

        # Align to common daily grid
        src_daily, tgt_daily = self._align_to_daily(source, target)
        if src_daily is None or len(src_daily) < 10:
            return None

        # Compute growth rates (first differences) to make series stationary
        src_diff = np.diff(src_daily)
        tgt_diff = np.diff(tgt_daily)

        # Normalize
        src_norm = self._normalize(src_diff)
        tgt_norm = self._normalize(tgt_diff)

        if src_norm is None or tgt_norm is None:
            return None

        # Cross-correlation at different lags
        best_lag, best_corr, best_pval = self._find_best_lag(src_norm, tgt_norm)

        # Granger causality test
        granger_f, granger_p = self._granger_test(src_diff, tgt_diff, lag=max(best_lag, 1))

        return CrossPlatformSignal(
            source_platform=f"{source.platform.value}/{source.metric_type.value}",
            target_platform=f"{target.platform.value}/{target.metric_type.value}",
            lag_days=best_lag,
            correlation=best_corr,
            p_value=best_pval,
            granger_f_stat=granger_f,
            granger_p_value=granger_p,
        )

    def compute_momentum_score(
        self,
        series_list: list[MetricTimeSeries],
        target: MetricTimeSeries,
    ) -> dict[str, float]:
        """Compute composite momentum score from multiple source signals.

        Returns a dictionary of feature values suitable for the probability engine.
        """
        features = {}

        for source in series_list:
            if source is target:
                continue
            sig = self.analyze(source, target)
            if sig is None:
                continue

            key = f"xplat_{source.platform.value}_{source.metric_type.value}"
            features[f"{key}_lag"] = float(sig.lag_days)
            features[f"{key}_corr"] = sig.correlation
            features[f"{key}_is_leading"] = 1.0 if sig.is_leading_indicator else 0.0

            if sig.granger_f_stat is not None:
                features[f"{key}_granger_f"] = sig.granger_f_stat

        # Composite momentum: weighted average of correlations from leading indicators
        leading = [
            (k, v) for k, v in features.items()
            if k.endswith("_corr") and features.get(k.replace("_corr", "_is_leading"), 0) > 0
        ]
        if leading:
            features["composite_momentum"] = np.mean([v for _, v in leading])
        else:
            features["composite_momentum"] = 0.0

        return features

    def _align_to_daily(
        self, source: MetricTimeSeries, target: MetricTimeSeries
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Interpolate both series onto a common daily grid."""
        src_ts = source.timestamps
        tgt_ts = target.timestamps

        # Find overlapping range
        start = max(src_ts[0], tgt_ts[0])
        end = min(src_ts[-1], tgt_ts[-1])
        if end <= start:
            return None, None

        # Create daily grid
        daily_ts = np.arange(start, end, 86400)  # seconds in a day
        if len(daily_ts) < 10:
            return None, None

        # Interpolate
        src_interp = np.interp(daily_ts, src_ts, source.values)
        tgt_interp = np.interp(daily_ts, tgt_ts, target.values)

        return src_interp, tgt_interp

    def _find_best_lag(
        self, src: np.ndarray, tgt: np.ndarray
    ) -> tuple[int, float, float]:
        """Find the lag with the highest absolute cross-correlation."""
        n = len(src)
        best_lag = 0
        best_corr = 0.0
        best_pval = 1.0

        for lag in range(-self.max_lag_days, self.max_lag_days + 1):
            if lag > 0:
                s, t = src[:-lag], tgt[lag:]
            elif lag < 0:
                s, t = src[-lag:], tgt[:lag]
            else:
                s, t = src, tgt

            if len(s) < 5:
                continue

            corr, pval = stats.pearsonr(s, t)
            if abs(corr) > abs(best_corr):
                best_lag = lag
                best_corr = corr
                best_pval = pval

        return best_lag, best_corr, best_pval

    @staticmethod
    def _granger_test(
        source: np.ndarray, target: np.ndarray, lag: int
    ) -> tuple[float | None, float | None]:
        """Simplified Granger causality test.

        Tests whether past values of source help predict target beyond
        what target's own past values predict.
        """
        n = len(target)
        if n <= lag + 1:
            return None, None

        # Build regression matrices
        # Restricted model: target ~ lagged target only
        # Unrestricted model: target ~ lagged target + lagged source
        y = target[lag:]
        X_restricted = np.column_stack([target[lag - i - 1 : n - i - 1] for i in range(lag)])
        X_unrestricted = np.column_stack([
            X_restricted,
            *[source[lag - i - 1 : n - i - 1].reshape(-1, 1) for i in range(lag)],
        ])

        try:
            # OLS residuals
            beta_r = np.linalg.lstsq(X_restricted, y, rcond=None)[0]
            beta_u = np.linalg.lstsq(X_unrestricted, y, rcond=None)[0]
            ssr_r = np.sum((y - X_restricted @ beta_r) ** 2)
            ssr_u = np.sum((y - X_unrestricted @ beta_u) ** 2)

            p_r = X_restricted.shape[1]
            p_u = X_unrestricted.shape[1]
            df_diff = p_u - p_r
            df_resid = n - lag - p_u

            if df_resid <= 0 or ssr_u == 0:
                return None, None

            f_stat = ((ssr_r - ssr_u) / df_diff) / (ssr_u / df_resid)
            p_value = 1 - stats.f.cdf(f_stat, df_diff, df_resid)
            return float(f_stat), float(p_value)
        except (np.linalg.LinAlgError, ValueError):
            return None, None

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray | None:
        std = arr.std()
        if std == 0:
            return None
        return (arr - arr.mean()) / std

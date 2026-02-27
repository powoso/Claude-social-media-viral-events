"""Tests for growth curve fitting."""

from __future__ import annotations

import numpy as np
import pytest

from social_media_markets.config import GrowthPattern
from social_media_markets.features.growth_curves import GrowthCurveFitter
from tests.conftest import make_series


class TestGrowthCurveFitter:
    def setup_method(self):
        self.fitter = GrowthCurveFitter()

    def test_fit_linear(self, linear_series):
        fit = self.fitter.fit(linear_series)
        assert fit is not None
        assert fit.r_squared > 0.95
        # Should identify as linear
        assert fit.pattern == GrowthPattern.LINEAR

    def test_fit_exponential(self, exponential_series):
        fit = self.fitter.fit(exponential_series)
        assert fit is not None
        assert fit.r_squared > 0.90
        # Should fit well as exponential or logistic
        assert fit.pattern in (GrowthPattern.EXPONENTIAL, GrowthPattern.LOGISTIC, GrowthPattern.VIRAL)

    def test_fit_logistic(self, logistic_series):
        fit = self.fitter.fit(logistic_series)
        assert fit is not None
        assert fit.r_squared > 0.90

    def test_fit_plateau(self, plateau_series):
        fit = self.fitter.fit(plateau_series)
        assert fit is not None
        assert fit.r_squared > 0.85
        assert fit.pattern in (GrowthPattern.PLATEAU, GrowthPattern.LOGISTIC)

    def test_fit_all_returns_sorted_by_aic(self, linear_series):
        fits = self.fitter.fit_all(linear_series)
        assert len(fits) > 0
        aics = [f.aic for f in fits]
        assert aics == sorted(aics), "Fits should be sorted by AIC (best first)"

    def test_insufficient_data(self):
        series = make_series([100, 200, 300])  # only 3 points
        fit = self.fitter.fit(series)
        assert fit is None  # need at least 5 points

    def test_linear_prediction(self, linear_series):
        fit = self.fitter.fit(linear_series)
        assert fit is not None
        # Predict at day 100 (beyond data range)
        pred = fit.predict(np.array([100.0]))
        assert pred[0] > 0

    def test_logistic_has_ceiling(self, logistic_series):
        fits = self.fitter.fit_all(logistic_series)
        logistic_fits = [f for f in fits if f.pattern == GrowthPattern.LOGISTIC]
        if logistic_fits:
            K = logistic_fits[0].parameters.get("K", 0)
            assert K > 0, "Logistic fit should have a positive carrying capacity"

    def test_noisy_linear_still_fits(self):
        """Linear with noise should still be identified."""
        rng = np.random.default_rng(42)
        values = [1000 + 50 * i + rng.normal(0, 100) for i in range(60)]
        series = make_series(values)
        fit = self.fitter.fit(series)
        assert fit is not None
        assert fit.r_squared > 0.7

    def test_declining_series(self):
        """Series that peaks then declines."""
        values = []
        for i in range(60):
            if i < 20:
                values.append(1000 + 200 * i)
            else:
                values.append(5000 * np.exp(-0.03 * (i - 20)))
        series = make_series(values, entity_name="declining_channel")
        fit = self.fitter.fit(series)
        assert fit is not None

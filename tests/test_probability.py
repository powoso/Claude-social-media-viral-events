"""Tests for the probability engine components."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from social_media_markets.config import MetricType, Platform
from social_media_markets.models import MarketQuestion, MetricTimeSeries
from social_media_markets.probability.engine import ProbabilityEngine
from social_media_markets.probability.simulation import StochasticSimulator
from social_media_markets.probability.survival import Comparable, MilestoneSurvival
from social_media_markets.probability.threshold import ThresholdPredictor
from tests.conftest import make_series


class TestThresholdPredictor:
    def setup_method(self):
        self.predictor = ThresholdPredictor()

    def test_already_reached(self, linear_series):
        """If current value exceeds target, probability should be 1.0."""
        prob, fit = self.predictor.predict(linear_series, target=100.0)
        assert prob == 1.0

    def test_linear_reachable_target(self, linear_series):
        """Linear growth should give high probability for reachable target."""
        current = linear_series.latest_value()
        target = current * 1.2  # 20% above current
        prob, fit = self.predictor.predict(
            linear_series, target=target, deadline_days=365
        )
        assert prob > 0.3

    def test_unreachable_target(self, plateau_series):
        """Plateau series should have low probability for far targets."""
        prob, fit = self.predictor.predict(
            plateau_series, target=1e9, deadline_days=30
        )
        assert prob < 0.5

    def test_time_to_milestone_linear(self, linear_series):
        current = linear_series.latest_value()
        target = current + 5000
        median, p10, p90 = self.predictor.predict_time_to_milestone(
            linear_series, target
        )
        if median is not None:
            assert median > 0
            assert p10 is not None and p10 <= median
            assert p90 is not None and p90 >= median

    def test_empty_series(self):
        series = MetricTimeSeries(
            platform=Platform.YOUTUBE,
            metric_type=MetricType.SUBSCRIBERS,
            entity_name="empty",
            points=[],
        )
        prob, fit = self.predictor.predict(series, target=1000)
        assert prob == 0.0


class TestMilestoneSurvival:
    def test_add_comparables(self):
        survival = MilestoneSurvival()
        survival.add_comparable(
            Comparable("ch1", 1000, 10000, 50.0, 180, reached=True)
        )
        survival.add_comparable(
            Comparable("ch2", 2000, 10000, 30.0, 300, reached=True)
        )
        survival.add_comparable(
            Comparable("ch3", 1500, 10000, 40.0, None, reached=False)
        )
        assert len(survival.comparables) == 3

    def test_estimate_with_comparables(self):
        survival = MilestoneSurvival()
        # Add several comparables with similar growth rates
        for i in range(10):
            survival.add_comparable(
                Comparable(
                    f"ch{i}", 5000, 50000, 100.0 + i * 5,
                    days_to_milestone=300 + i * 20,
                    reached=True,
                )
            )
        # Add some censored
        for i in range(3):
            survival.add_comparable(
                Comparable(
                    f"cens{i}", 5000, 50000, 90.0,
                    days_to_milestone=200,
                    reached=False,
                )
            )

        est = survival.estimate(10000, 50000, 100.0, deadline_days=500)
        assert 0 < est.probability_by_deadline <= 1.0
        assert est.n_comparables > 0

    def test_parametric_fallback(self):
        survival = MilestoneSurvival()
        est = survival.estimate(10000, 50000, 100.0, deadline_days=500)
        assert est.n_comparables == 0
        assert est.median_days is not None
        assert est.probability_by_deadline > 0


class TestStochasticSimulator:
    def test_simulate_paths_shape(self, linear_series):
        from social_media_markets.features.growth_curves import GrowthCurveFitter

        fitter = GrowthCurveFitter()
        fit = fitter.fit(linear_series)
        assert fit is not None

        sim = StochasticSimulator(n_simulations=100, seed=42)
        paths = sim.simulate_paths(linear_series, fit, horizon_days=30)
        assert paths.shape == (100, 30)
        assert np.all(paths >= 0)

    def test_probability_of_threshold(self, linear_series):
        from social_media_markets.features.growth_curves import GrowthCurveFitter

        fitter = GrowthCurveFitter()
        fit = fitter.fit(linear_series)
        sim = StochasticSimulator(n_simulations=500, seed=42)
        paths = sim.simulate_paths(linear_series, fit, horizon_days=60)

        # Very low target should be reached by all paths
        prob_low = sim.probability_of_threshold(paths, 1.0)
        assert prob_low > 0.9

        # Very high target should be reached by few paths
        prob_high = sim.probability_of_threshold(paths, 1e15)
        assert prob_high < 0.5

    def test_confidence_intervals(self, linear_series):
        from social_media_markets.features.growth_curves import GrowthCurveFitter

        fitter = GrowthCurveFitter()
        fit = fitter.fit(linear_series)
        sim = StochasticSimulator(n_simulations=100, seed=42)
        paths = sim.simulate_paths(linear_series, fit, horizon_days=30)
        ci = sim.confidence_intervals(paths)

        assert 5 in ci
        assert 50 in ci
        assert 95 in ci
        # P5 should be <= P50 <= P95 at each step
        for step in range(30):
            assert ci[5][step] <= ci[50][step] <= ci[95][step]

    def test_viral_event_analysis(self):
        rng = np.random.default_rng(42)
        # Create power law distributed data
        values = list(rng.pareto(2.0, 100) * 1000)
        series = make_series(values)

        sim = StochasticSimulator(seed=42)
        result = sim.viral_event_analysis(series)
        assert "viral_probability" in result
        assert "power_law_alpha" in result
        assert result["power_law_alpha"] > 0


class TestProbabilityEngine:
    def test_full_prediction(self, linear_series):
        engine = ProbabilityEngine(n_simulations=500, seed=42)
        question = MarketQuestion(
            question_text="Will test_channel reach 20000 subscribers?",
            platform=Platform.YOUTUBE,
            metric_type=MetricType.SUBSCRIBERS,
            entity_name="test_channel",
            target_value=20000,
            deadline=datetime.now(timezone.utc) + timedelta(days=180),
            current_market_price=0.5,
        )

        result = engine.predict(question, linear_series)
        assert 0 < result.probability < 1
        assert result.confidence_interval[0] < result.confidence_interval[1]
        assert result.reasoning != ""
        assert isinstance(result.edge, float)

    def test_prediction_with_market_price(self, linear_series):
        engine = ProbabilityEngine(n_simulations=200, seed=42)
        question = MarketQuestion(
            question_text="Test",
            platform=Platform.YOUTUBE,
            metric_type=MetricType.SUBSCRIBERS,
            entity_name="test_channel",
            target_value=20000,
            current_market_price=0.3,
        )
        result = engine.predict(question, linear_series)
        assert result.edge == result.model_probability - 0.3

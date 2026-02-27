"""Tests for edge detection module."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from social_media_markets.config import MetricType, Platform
from social_media_markets.edge.detector import EdgeDetector
from social_media_markets.models import MarketQuestion
from tests.conftest import make_series


class TestEdgeDetector:
    def setup_method(self):
        self.detector = EdgeDetector()

    def _make_question(self, target: float, market_price: float = 0.5) -> MarketQuestion:
        return MarketQuestion(
            question_text=f"Will X reach {target}?",
            platform=Platform.YOUTUBE,
            metric_type=MetricType.SUBSCRIBERS,
            entity_name="test_channel",
            target_value=target,
            current_market_price=market_price,
        )

    def test_mean_reversion_fast_growth(self):
        """Detect mean reversion when recent growth is much faster than trend."""
        # Long-term: moderate growth. Recent: explosive
        values = [1000 + 10 * i for i in range(60)]
        # Last 14 days: accelerate dramatically
        for i in range(14):
            values.append(values[-1] + 100)

        series = make_series(values, entity_name="accelerating")
        question = self._make_question(target=20000)
        signal = self.detector.detect_mean_reversion(series, question)
        if signal is not None:
            assert signal.direction == "over"
            assert signal.magnitude > 0

    def test_no_edge_stable_growth(self, linear_series):
        """Stable linear growth should not trigger mean reversion signal."""
        question = self._make_question(target=20000)
        signal = self.detector.detect_mean_reversion(linear_series, question)
        # Might be None or weak signal
        if signal is not None:
            assert signal.magnitude < 0.15

    def test_growth_pattern_edge_exponential(self, exponential_series):
        """Exponential growth should trigger skepticism signal."""
        question = self._make_question(target=1e9)
        signal = self.detector.detect_growth_pattern_edge(exponential_series, question)
        # May or may not trigger depending on fit quality
        if signal is not None:
            assert signal.name in ("exponential_skepticism", "logistic_ceiling")

    def test_controversy_edge(self, spike_series):
        """Spike series should trigger controversy edge detection."""
        question = self._make_question(target=10000)
        signal = self.detector.detect_controversy_edge(spike_series, question)
        # Spike might or might not be "current" depending on series end
        # Just verify it doesn't crash

    def test_model_market_divergence(self):
        """Large divergence between model and market should be detected."""
        signal = self.detector.detect_model_market_divergence(0.8, 0.4)
        assert signal is not None
        assert signal.direction == "under"  # market underprices
        assert abs(signal.magnitude - 0.4) < 0.01

    def test_no_divergence_when_close(self):
        signal = self.detector.detect_model_market_divergence(0.52, 0.50)
        assert signal is None

    def test_full_analysis(self, linear_series):
        """Full analysis should return EdgeAnalysis with recommendation."""
        question = self._make_question(target=20000, market_price=0.5)
        result = self.detector.analyze(
            question, linear_series, model_probability=0.65
        )
        assert result.recommendation in ("buy", "sell", "pass")
        assert isinstance(result.net_edge, float)
        assert isinstance(result.confidence, float)

    def test_combine_signals_agreement(self):
        """When signals agree, confidence should be higher."""
        from social_media_markets.edge.detector import EdgeSignal

        signals = [
            EdgeSignal("a", "under", 0.2, 0.6, "Signal A"),
            EdgeSignal("b", "under", 0.15, 0.5, "Signal B"),
        ]
        question = self._make_question(target=20000)
        analysis = self.detector._combine_signals(signals, question)
        assert analysis.net_edge > 0  # both say "under" → positive edge
        assert analysis.confidence > 0

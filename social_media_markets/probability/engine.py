"""Main probability engine — orchestrates all prediction components."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np

from social_media_markets.features.controversy import ControversyDetector
from social_media_markets.features.cross_platform import CrossPlatformMomentum
from social_media_markets.features.engagement import EngagementAnalyzer
from social_media_markets.features.growth_curves import GrowthCurveFitter
from social_media_markets.features.seasonality import SeasonalityAnalyzer
from social_media_markets.models import MarketQuestion, MetricTimeSeries, PredictionResult
from social_media_markets.probability.simulation import StochasticSimulator
from social_media_markets.probability.survival import MilestoneSurvival
from social_media_markets.probability.threshold import ThresholdPredictor

logger = logging.getLogger(__name__)


class ProbabilityEngine:
    """Main engine that combines all probability estimation methods.

    Produces a final probability estimate by weighting:
    1. Growth curve extrapolation (ThresholdPredictor)
    2. Survival analysis from comparables (MilestoneSurvival)
    3. Monte Carlo simulation (StochasticSimulator)

    Then adjusts for edge factors (mean reversion, cross-platform signals, etc.).
    """

    def __init__(
        self,
        n_simulations: int = 10000,
        seed: int | None = 42,
    ):
        self.fitter = GrowthCurveFitter()
        self.threshold = ThresholdPredictor()
        self.survival = MilestoneSurvival()
        self.simulator = StochasticSimulator(n_simulations=n_simulations, seed=seed)
        self.cross_platform = CrossPlatformMomentum()
        self.seasonality = SeasonalityAnalyzer()
        self.controversy = ControversyDetector()
        self.engagement = EngagementAnalyzer()

    def predict(
        self,
        question: MarketQuestion,
        primary_series: MetricTimeSeries,
        auxiliary_series: list[MetricTimeSeries] | None = None,
        comparable_series: list[MetricTimeSeries] | None = None,
    ) -> PredictionResult:
        """Generate a probability estimate for a market question.

        Args:
            question: The prediction market question.
            primary_series: Main time series for the entity/metric.
            auxiliary_series: Additional series (Google Trends, engagement, etc.)
                             for cross-platform and feature analysis.
            comparable_series: Historical series from similar entities
                              for survival analysis.

        Returns:
            PredictionResult with probability, confidence interval, and reasoning.
        """
        if auxiliary_series is None:
            auxiliary_series = []
        if comparable_series is None:
            comparable_series = []

        # Compute deadline in days
        deadline_days = None
        if question.deadline:
            now = datetime.now(timezone.utc)
            delta = question.deadline - now
            deadline_days = max(delta.total_seconds() / 86400, 0)

        features = {}
        reasoning_parts = []

        # ── 1. Growth curve fit ─────────────────────────────────
        growth_fit = self.fitter.fit(primary_series)
        if growth_fit:
            features["growth_pattern"] = growth_fit.pattern.value
            features["growth_r_squared"] = growth_fit.r_squared
            reasoning_parts.append(
                f"Growth pattern: {growth_fit.pattern.value} (R²={growth_fit.r_squared:.3f})"
            )

        # ── 2. Threshold prediction ────────────────────────────
        threshold_prob, growth_fit = self.threshold.predict(
            primary_series, question.target_value, deadline_days, growth_fit
        )
        features["threshold_probability"] = threshold_prob
        reasoning_parts.append(f"Growth model extrapolation: {threshold_prob:.1%}")

        # ── 3. Survival analysis ───────────────────────────────
        survival_prob = 0.5  # default
        if comparable_series:
            self.survival.add_comparables_from_series(
                comparable_series, question.target_value
            )
        growth_rate = primary_series.growth_rate(window_days=30)
        current_value = primary_series.latest_value()
        if growth_rate is not None and current_value is not None:
            survival_est = self.survival.estimate(
                current_value, question.target_value, growth_rate, deadline_days
            )
            survival_prob = survival_est.probability_by_deadline
            features["survival_probability"] = survival_prob
            features["survival_n_comparables"] = survival_est.n_comparables
            if survival_est.median_days is not None:
                features["survival_median_days"] = survival_est.median_days
                reasoning_parts.append(
                    f"Survival analysis: {survival_prob:.1%} "
                    f"(median {survival_est.median_days:.0f}d, "
                    f"n={survival_est.n_comparables})"
                )

        # ── 4. Monte Carlo simulation ──────────────────────────
        sim_prob = threshold_prob  # default to threshold if can't simulate
        if growth_fit is not None:
            # Calibrate viral parameters from historical data
            viral_stats = self.simulator.viral_event_analysis(primary_series)
            features.update({f"viral_{k}": v for k, v in viral_stats.items()})

            paths = self.simulator.simulate_paths(
                primary_series,
                growth_fit,
                horizon_days=int(deadline_days) if deadline_days else 365,
                viral_probability=viral_stats.get("viral_probability", 0.002),
            )
            by_step = paths.shape[1] if deadline_days else None
            sim_prob = self.simulator.probability_of_threshold(
                paths, question.target_value, by_step
            )
            features["simulation_probability"] = sim_prob
            reasoning_parts.append(f"Monte Carlo simulation: {sim_prob:.1%}")

            # Confidence intervals from simulation
            ci = self.simulator.confidence_intervals(paths)
            features["sim_p5_final"] = float(ci[5][-1])
            features["sim_p95_final"] = float(ci[95][-1])

        # ── 5. Cross-platform momentum ─────────────────────────
        if auxiliary_series:
            momentum = self.cross_platform.compute_momentum_score(
                auxiliary_series, primary_series
            )
            features.update(momentum)
            if momentum.get("composite_momentum", 0) > 0.3:
                reasoning_parts.append(
                    f"Cross-platform momentum: {momentum['composite_momentum']:.2f} (positive signal)"
                )

        # ── 6. Seasonality ─────────────────────────────────────
        season = self.seasonality.analyze(primary_series)
        if season and season.seasonal_strength > 0.1:
            features["seasonal_strength"] = season.seasonal_strength
            if deadline_days and deadline_days < 365:
                target_day = primary_series.days_array[-1] + (deadline_days / 2)
                adj = self.seasonality.get_seasonal_adjustment(primary_series, target_day)
                features["seasonal_adjustment"] = adj
                reasoning_parts.append(f"Seasonal factor: {adj:.2f}x")

        # ── 7. Controversy/spike check ─────────────────────────
        controversy = self.controversy.analyze(primary_series)
        if controversy.current_is_spiking:
            features["currently_spiking"] = 1.0
            features["avg_decay_half_life"] = controversy.avg_decay_half_life
            reasoning_parts.append(
                f"Currently spiking (half-life: {controversy.avg_decay_half_life:.1f}d)"
            )

        # ── 8. Combine probabilities ───────────────────────────
        # Weighted average with simulation getting highest weight
        weights = {
            "threshold": 0.25,
            "survival": 0.25 if comparable_series else 0.0,
            "simulation": 0.50 if growth_fit else 0.25,
        }

        # Normalize weights
        total_w = sum(weights.values())
        if total_w > 0:
            weights = {k: v / total_w for k, v in weights.items()}

        raw_prob = (
            weights["threshold"] * threshold_prob
            + weights["survival"] * survival_prob
            + weights["simulation"] * sim_prob
        )

        # Apply adjustments
        adjusted_prob = self._apply_adjustments(raw_prob, features)

        # Confidence interval on the probability estimate
        # Wider CI when we have less data or disagreement between models
        model_spread = np.std([threshold_prob, survival_prob, sim_prob])
        ci_width = max(0.1, model_spread * 2)
        ci_low = max(0.01, adjusted_prob - ci_width / 2)
        ci_high = min(0.99, adjusted_prob + ci_width / 2)

        # Edge vs market
        edge = 0.0
        if question.current_market_price is not None:
            edge = adjusted_prob - question.current_market_price

        reasoning = " | ".join(reasoning_parts)
        if edge != 0 and question.current_market_price is not None:
            reasoning += f" | Edge: {edge:+.1%} vs market {question.current_market_price:.1%}"

        return PredictionResult(
            question=question,
            probability=float(np.clip(adjusted_prob, 0.01, 0.99)),
            confidence_interval=(ci_low, ci_high),
            model_probability=float(raw_prob),
            edge=float(edge),
            growth_fit=growth_fit,
            features_used=features,
            reasoning=reasoning,
        )

    def _apply_adjustments(self, raw_prob: float, features: dict) -> float:
        """Apply edge-informed adjustments to the raw probability."""
        prob = raw_prob

        # Mean reversion adjustment: if growth pattern looks exponential,
        # apply skepticism — most exponential growth plateaus
        if features.get("growth_pattern") == "exponential":
            prob *= 0.85  # 15% haircut for plateau risk

        # Cross-platform momentum boost
        momentum = features.get("composite_momentum", 0)
        if momentum > 0.5:
            prob = prob + (1 - prob) * 0.1 * momentum  # small boost

        # Controversy spike adjustment: if currently spiking, markets
        # often overestimate staying power
        if features.get("currently_spiking", 0):
            half_life = features.get("avg_decay_half_life", 7)
            if half_life < 14:
                prob *= 0.9  # discount if fast-decaying spike

        # Seasonal adjustment
        seasonal_adj = features.get("seasonal_adjustment", 1.0)
        if seasonal_adj > 1.1:
            prob = prob + (1 - prob) * 0.05  # small tailwind
        elif seasonal_adj < 0.9:
            prob *= 0.95  # small headwind

        return float(np.clip(prob, 0.01, 0.99))

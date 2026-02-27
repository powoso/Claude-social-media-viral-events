"""Edge detection — identifies systematic mispricings in prediction markets.

Markets for social media milestones exhibit several recurring biases:
1. Recency bias: overweight recent trajectory, underweight mean reversion
2. Platform blindness: don't use cross-platform signals
3. Controversy miscalibration: overreact to spikes, misjudge decay
4. Algorithm ignorance: don't price in platform algorithm changes
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from social_media_markets.config import GrowthPattern
from social_media_markets.features.controversy import ControversyAnalysis, ControversyDetector
from social_media_markets.features.cross_platform import CrossPlatformMomentum, CrossPlatformSignal
from social_media_markets.features.growth_curves import GrowthCurveFitter
from social_media_markets.models import GrowthCurveFit, MarketQuestion, MetricTimeSeries

logger = logging.getLogger(__name__)


@dataclass
class EdgeSignal:
    """A single edge signal contributing to the overall edge estimate."""

    name: str
    direction: str  # "over" or "under" (market is over/underpricing)
    magnitude: float  # 0-1 scale of signal strength
    confidence: float  # 0-1 confidence in this signal
    explanation: str

    @property
    def signed_magnitude(self) -> float:
        """Positive = market underprices (buy), negative = market overprices (sell)."""
        mult = 1.0 if self.direction == "under" else -1.0
        return mult * self.magnitude * self.confidence


@dataclass
class EdgeAnalysis:
    """Complete edge analysis for a prediction market question."""

    signals: list[EdgeSignal] = field(default_factory=list)
    net_edge: float = 0.0  # combined edge estimate
    confidence: float = 0.0  # overall confidence
    recommendation: str = ""  # "buy", "sell", or "pass"
    reasoning: str = ""

    @property
    def has_actionable_edge(self) -> bool:
        return abs(self.net_edge) > 0.05 and self.confidence > 0.3


class EdgeDetector:
    """Identifies systematic mispricings in social media prediction markets.

    Each detection method corresponds to a known market bias. The detector
    combines multiple signals into a net edge estimate with confidence.
    """

    def __init__(self):
        self.fitter = GrowthCurveFitter()
        self.cross_platform = CrossPlatformMomentum()
        self.controversy = ControversyDetector()

    def analyze(
        self,
        question: MarketQuestion,
        primary_series: MetricTimeSeries,
        auxiliary_series: list[MetricTimeSeries] | None = None,
        model_probability: float | None = None,
    ) -> EdgeAnalysis:
        """Run all edge detection methods and combine signals.

        Args:
            question: The market question being analyzed.
            primary_series: Main metric time series.
            auxiliary_series: Cross-platform data for signal detection.
            model_probability: Our model's probability estimate.

        Returns:
            EdgeAnalysis with combined signals and recommendation.
        """
        if auxiliary_series is None:
            auxiliary_series = []

        signals = []

        # 1. Mean reversion edge
        mr_signal = self.detect_mean_reversion(primary_series, question)
        if mr_signal:
            signals.append(mr_signal)

        # 2. Cross-platform leading indicator edge
        xp_signals = self.detect_cross_platform_edge(
            primary_series, auxiliary_series, question
        )
        signals.extend(xp_signals)

        # 3. Controversy overcorrection edge
        cont_signal = self.detect_controversy_edge(primary_series, question)
        if cont_signal:
            signals.append(cont_signal)

        # 4. Growth pattern mispricing
        gp_signal = self.detect_growth_pattern_edge(primary_series, question)
        if gp_signal:
            signals.append(gp_signal)

        # 5. Market-vs-model edge
        if model_probability is not None and question.current_market_price is not None:
            mm_signal = self.detect_model_market_divergence(
                model_probability, question.current_market_price
            )
            if mm_signal:
                signals.append(mm_signal)

        # Combine signals
        analysis = self._combine_signals(signals, question)
        return analysis

    def detect_mean_reversion(
        self,
        series: MetricTimeSeries,
        question: MarketQuestion,
    ) -> EdgeSignal | None:
        """Detect mean reversion edge.

        Most growth trajectories plateau eventually. If a market prices in
        continued exponential growth, there's edge in fading that.
        Conversely, if growth dipped temporarily, the market may undervalue
        the reversion to trend.
        """
        if len(series.points) < 15:
            return None

        t = series.days_array
        y = series.values

        # Compare recent growth to long-term growth
        long_rate = series.growth_rate(window_days=90)
        short_rate = series.growth_rate(window_days=14)

        if long_rate is None or short_rate is None or long_rate == 0:
            return None

        ratio = short_rate / long_rate

        if ratio > 2.0:
            # Recent growth much faster than long-term — likely to revert down
            return EdgeSignal(
                name="mean_reversion",
                direction="over",
                magnitude=min((ratio - 1) * 0.15, 0.3),
                confidence=0.6,
                explanation=(
                    f"Recent 14d growth rate ({short_rate:.1f}/day) is {ratio:.1f}x "
                    f"the 90d average ({long_rate:.1f}/day) — likely to revert"
                ),
            )
        elif ratio < 0.3 and short_rate > 0:
            # Recent growth much slower — might be temporary dip
            return EdgeSignal(
                name="mean_reversion",
                direction="under",
                magnitude=min((1 - ratio) * 0.1, 0.2),
                confidence=0.4,
                explanation=(
                    f"Recent 14d growth rate ({short_rate:.1f}/day) is only {ratio:.1f}x "
                    f"the 90d average ({long_rate:.1f}/day) — may revert upward"
                ),
            )

        return None

    def detect_cross_platform_edge(
        self,
        primary: MetricTimeSeries,
        auxiliary: list[MetricTimeSeries],
        question: MarketQuestion,
    ) -> list[EdgeSignal]:
        """Detect edge from cross-platform leading indicators.

        Markets typically focus on the primary metric and ignore signals
        from other platforms that historically predict it.
        """
        signals = []

        for aux in auxiliary:
            xp = self.cross_platform.analyze(aux, primary)
            if xp is None or not xp.is_leading_indicator:
                continue

            # Check if the leading indicator is currently accelerating
            aux_rate = aux.growth_rate(window_days=7)
            aux_rate_long = aux.growth_rate(window_days=30)

            if aux_rate is None or aux_rate_long is None or aux_rate_long == 0:
                continue

            accel = aux_rate / aux_rate_long

            if accel > 1.5:
                signals.append(
                    EdgeSignal(
                        name=f"cross_platform_{aux.platform.value}",
                        direction="under",
                        magnitude=min(xp.correlation * 0.15, 0.2),
                        confidence=min(xp.correlation, 0.7),
                        explanation=(
                            f"{aux.platform.value} {aux.metric_type.value} accelerating "
                            f"({accel:.1f}x) and leads {primary.platform.value} by "
                            f"{xp.lag_days}d (corr={xp.correlation:.2f})"
                        ),
                    )
                )
            elif accel < 0.5:
                signals.append(
                    EdgeSignal(
                        name=f"cross_platform_{aux.platform.value}",
                        direction="over",
                        magnitude=min(xp.correlation * 0.1, 0.15),
                        confidence=min(xp.correlation * 0.8, 0.5),
                        explanation=(
                            f"{aux.platform.value} {aux.metric_type.value} decelerating "
                            f"({accel:.1f}x) and leads {primary.platform.value} by "
                            f"{xp.lag_days}d — growth slowdown incoming"
                        ),
                    )
                )

        return signals

    def detect_controversy_edge(
        self,
        series: MetricTimeSeries,
        question: MarketQuestion,
    ) -> EdgeSignal | None:
        """Detect edge from controversy/viral spike patterns.

        Markets tend to overcorrect during controversy: overestimate the
        negative impact during the spike and underestimate recovery.
        The predicted decay target is usually more accurate than market prices.
        """
        analysis = self.controversy.analyze(series)

        if not analysis.current_is_spiking:
            return None

        if not analysis.spikes:
            return None

        last_spike = analysis.spikes[-1]

        # During a spike, markets often overreact
        if last_spike.magnitude > 3:
            return EdgeSignal(
                name="controversy_overcorrection",
                direction="under",  # market overestimates damage, creating underpricing
                magnitude=min(0.1 * (last_spike.magnitude - 2), 0.25),
                confidence=0.5,
                explanation=(
                    f"Currently in attention spike ({last_spike.magnitude:.1f}x baseline). "
                    f"Historical half-life: {last_spike.decay_half_life:.1f}d. "
                    f"Markets typically overcorrect on controversy impact."
                ),
            )

        return None

    def detect_growth_pattern_edge(
        self,
        series: MetricTimeSeries,
        question: MarketQuestion,
    ) -> EdgeSignal | None:
        """Detect edge from growth pattern classification.

        If the growth pattern is logistic/plateau but markets imply
        exponential continuation, there's edge in selling.
        """
        fit = self.fitter.fit(series)
        if fit is None:
            return None

        # Check if the growth pattern suggests a ceiling
        if fit.pattern == GrowthPattern.LOGISTIC and fit.r_squared > 0.85:
            K = fit.parameters.get("K", 0)
            current = series.latest_value() or 0
            if current > 0 and K > 0:
                pct_of_ceiling = current / K
                if pct_of_ceiling > 0.7 and question.target_value > K * 0.95:
                    return EdgeSignal(
                        name="logistic_ceiling",
                        direction="over",
                        magnitude=min((pct_of_ceiling - 0.5) * 0.4, 0.3),
                        confidence=fit.r_squared * 0.8,
                        explanation=(
                            f"Growth fits logistic model (R²={fit.r_squared:.3f}) with "
                            f"ceiling at {K:,.0f}. Currently at {pct_of_ceiling:.0%} of "
                            f"ceiling — target {question.target_value:,.0f} exceeds it."
                        ),
                    )

        elif fit.pattern == GrowthPattern.EXPONENTIAL and fit.r_squared > 0.9:
            # Exponential fits are seductive but usually break down
            return EdgeSignal(
                name="exponential_skepticism",
                direction="over",
                magnitude=0.10,
                confidence=0.4,
                explanation=(
                    f"Growth currently fits exponential (R²={fit.r_squared:.3f}), "
                    f"but sustained exponential growth is rare — expect plateau."
                ),
            )

        return None

    def detect_model_market_divergence(
        self,
        model_prob: float,
        market_prob: float,
    ) -> EdgeSignal | None:
        """Detect divergence between our model and market price."""
        diff = model_prob - market_prob

        if abs(diff) < 0.05:
            return None

        return EdgeSignal(
            name="model_market_divergence",
            direction="under" if diff > 0 else "over",
            magnitude=abs(diff),
            confidence=0.5,  # moderate confidence in model
            explanation=(
                f"Model estimate ({model_prob:.1%}) differs from market "
                f"({market_prob:.1%}) by {diff:+.1%}"
            ),
        )

    def _combine_signals(
        self, signals: list[EdgeSignal], question: MarketQuestion
    ) -> EdgeAnalysis:
        """Combine multiple edge signals into a single recommendation."""
        if not signals:
            return EdgeAnalysis(
                signals=[],
                net_edge=0.0,
                confidence=0.0,
                recommendation="pass",
                reasoning="No edge signals detected.",
            )

        # Weighted combination of signals
        net_edge = sum(s.signed_magnitude for s in signals)

        # Confidence: higher when signals agree, lower when they conflict
        directions = [s.direction for s in signals]
        agreement = max(
            directions.count("over"), directions.count("under")
        ) / len(directions)
        avg_confidence = np.mean([s.confidence for s in signals])
        combined_confidence = agreement * avg_confidence

        # Recommendation
        if abs(net_edge) > 0.05 and combined_confidence > 0.3:
            recommendation = "buy" if net_edge > 0 else "sell"
        else:
            recommendation = "pass"

        # Build reasoning string
        reasoning_parts = []
        for s in sorted(signals, key=lambda s: abs(s.signed_magnitude), reverse=True):
            reasoning_parts.append(f"[{s.name}] {s.explanation}")

        reasoning = " || ".join(reasoning_parts)

        return EdgeAnalysis(
            signals=signals,
            net_edge=float(net_edge),
            confidence=float(combined_confidence),
            recommendation=recommendation,
            reasoning=reasoning,
        )

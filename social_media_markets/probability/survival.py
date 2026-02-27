"""Survival analysis for time-to-milestone prediction.

Uses historical comparables to estimate how long it takes entities at a
given growth rate to reach a milestone, accounting for censored data
(entities that haven't reached the milestone yet).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats

from social_media_markets.models import MetricTimeSeries

logger = logging.getLogger(__name__)


@dataclass
class Comparable:
    """A comparable entity used for survival analysis."""

    entity_name: str
    starting_value: float
    target_value: float
    growth_rate: float  # daily growth rate at time of measurement
    days_to_milestone: float | None  # None if not yet reached (censored)
    reached: bool


@dataclass
class SurvivalEstimate:
    """Survival analysis estimate for time to milestone."""

    median_days: float | None
    p25_days: float | None  # 25th percentile (fast achievers)
    p75_days: float | None  # 75th percentile (slow achievers)
    probability_by_deadline: float  # P(reached by deadline)
    n_comparables: int
    hazard_rate: float  # instantaneous rate of reaching milestone


class MilestoneSurvival:
    """Estimates time-to-milestone using survival analysis with comparables.

    Instead of just extrapolating a single entity's growth curve, this
    uses historical data from similar entities at similar growth stages
    to build a more robust estimate.
    """

    def __init__(self):
        self.comparables: list[Comparable] = []

    def add_comparable(self, comp: Comparable) -> None:
        """Add a comparable entity to the analysis pool."""
        self.comparables.append(comp)

    def add_comparables_from_series(
        self,
        series_list: list[MetricTimeSeries],
        target_value: float,
    ) -> int:
        """Build comparables from a list of historical time series.

        For each series, determine if/when they crossed the target value.

        Returns:
            Number of comparables added.
        """
        added = 0
        for series in series_list:
            if len(series.points) < 5:
                continue

            t = series.days_array
            y = series.values
            growth_rate = series.growth_rate(window_days=30)
            if growth_rate is None:
                continue

            # Check if target was reached
            crossed_idx = np.where(y >= target_value)[0]
            if len(crossed_idx) > 0:
                days_to = t[crossed_idx[0]] - t[0]
                comp = Comparable(
                    entity_name=series.entity_name,
                    starting_value=y[0],
                    target_value=target_value,
                    growth_rate=growth_rate,
                    days_to_milestone=days_to,
                    reached=True,
                )
            else:
                comp = Comparable(
                    entity_name=series.entity_name,
                    starting_value=y[0],
                    target_value=target_value,
                    growth_rate=growth_rate,
                    days_to_milestone=t[-1] - t[0],  # observed duration
                    reached=False,
                )

            self.comparables.append(comp)
            added += 1

        return added

    def estimate(
        self,
        current_value: float,
        target_value: float,
        current_growth_rate: float,
        deadline_days: float | None = None,
    ) -> SurvivalEstimate:
        """Estimate time to milestone using Kaplan-Meier-like approach.

        Filters comparables to those with similar growth characteristics,
        then builds a survival curve.

        Args:
            current_value: Current metric value.
            target_value: Target to reach.
            current_growth_rate: Current daily growth rate.
            deadline_days: Optional deadline for probability calculation.

        Returns:
            SurvivalEstimate with time estimates and probability.
        """
        if current_value >= target_value:
            return SurvivalEstimate(
                median_days=0, p25_days=0, p75_days=0,
                probability_by_deadline=1.0, n_comparables=0, hazard_rate=0,
            )

        # Filter to relevant comparables (similar growth rate range)
        relevant = self._filter_comparables(current_growth_rate)

        if len(relevant) < 3:
            # Not enough comparables — fall back to parametric estimate
            return self._parametric_fallback(
                current_value, target_value, current_growth_rate, deadline_days
            )

        # Compute normalized times: adjust for starting distance to target
        normalized_times = []
        censored = []

        for comp in relevant:
            # Scale time by the ratio of distances to target
            ratio = (target_value - current_value) / max(
                comp.target_value - comp.starting_value, 1
            )
            if comp.days_to_milestone is not None:
                scaled_time = comp.days_to_milestone * ratio
                normalized_times.append(scaled_time)
                censored.append(not comp.reached)
            else:
                continue

        if not normalized_times:
            return self._parametric_fallback(
                current_value, target_value, current_growth_rate, deadline_days
            )

        times = np.array(normalized_times)
        cens = np.array(censored)

        # Kaplan-Meier survival function
        survival_times, survival_probs = self._kaplan_meier(times, cens)

        # Extract percentiles
        median = self._percentile_from_survival(survival_times, survival_probs, 0.5)
        p25 = self._percentile_from_survival(survival_times, survival_probs, 0.25)
        p75 = self._percentile_from_survival(survival_times, survival_probs, 0.75)

        # Probability by deadline
        if deadline_days is not None:
            prob = self._probability_by_time(survival_times, survival_probs, deadline_days)
        else:
            prob = 1.0 - survival_probs[-1] if len(survival_probs) > 0 else 0.5

        # Hazard rate (events per day)
        n_events = np.sum(~cens)
        total_time = np.sum(times)
        hazard = n_events / total_time if total_time > 0 else 0.0

        return SurvivalEstimate(
            median_days=median,
            p25_days=p25,
            p75_days=p75,
            probability_by_deadline=float(prob),
            n_comparables=len(relevant),
            hazard_rate=float(hazard),
        )

    def _filter_comparables(self, growth_rate: float) -> list[Comparable]:
        """Filter comparables with similar growth rates (within 2x)."""
        if not self.comparables:
            return []

        relevant = []
        for comp in self.comparables:
            if comp.growth_rate <= 0:
                continue
            ratio = growth_rate / comp.growth_rate if comp.growth_rate != 0 else float("inf")
            if 0.3 <= ratio <= 3.0:  # within ~3x growth rate
                relevant.append(comp)

        return relevant

    @staticmethod
    def _kaplan_meier(
        times: np.ndarray, censored: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Kaplan-Meier survival curve."""
        # Sort by time
        order = np.argsort(times)
        sorted_times = times[order]
        sorted_censored = censored[order]

        n = len(times)
        survival_times = [0.0]
        survival_probs = [1.0]

        at_risk = n
        for i in range(n):
            if not sorted_censored[i]:  # event (reached milestone)
                prob = survival_probs[-1] * (1 - 1 / at_risk)
                survival_times.append(sorted_times[i])
                survival_probs.append(prob)
            at_risk -= 1

        return np.array(survival_times), np.array(survival_probs)

    @staticmethod
    def _percentile_from_survival(
        times: np.ndarray, probs: np.ndarray, percentile: float
    ) -> float | None:
        """Extract a time percentile from the survival curve."""
        target_prob = 1 - percentile  # survival probability at the percentile
        idx = np.where(probs <= target_prob)[0]
        if len(idx) == 0:
            return None
        return float(times[idx[0]])

    @staticmethod
    def _probability_by_time(
        times: np.ndarray, probs: np.ndarray, deadline: float
    ) -> float:
        """Get the probability of event by a specific time."""
        idx = np.searchsorted(times, deadline)
        if idx >= len(probs):
            return 1.0 - probs[-1]
        return 1.0 - probs[idx]

    def _parametric_fallback(
        self,
        current_value: float,
        target_value: float,
        growth_rate: float,
        deadline_days: float | None,
    ) -> SurvivalEstimate:
        """Parametric estimate when comparables are insufficient."""
        if growth_rate <= 0:
            return SurvivalEstimate(
                median_days=None, p25_days=None, p75_days=None,
                probability_by_deadline=0.05, n_comparables=0, hazard_rate=0,
            )

        gap = target_value - current_value
        days_at_current_rate = gap / growth_rate

        # Add uncertainty: assume growth rate has CV of 0.5
        cv = 0.5
        median = days_at_current_rate
        p25 = days_at_current_rate * (1 - cv * 0.67)  # ~25th percentile of lognormal
        p75 = days_at_current_rate * (1 + cv * 0.67)  # ~75th percentile

        if deadline_days is not None:
            # Use lognormal CDF for probability
            sigma = np.log(1 + cv ** 2) ** 0.5
            mu = np.log(days_at_current_rate) - sigma ** 2 / 2
            prob = float(stats.lognorm.cdf(deadline_days, s=sigma, scale=np.exp(mu)))
        else:
            prob = 0.5

        return SurvivalEstimate(
            median_days=float(median),
            p25_days=float(max(p25, 1)),
            p75_days=float(p75),
            probability_by_deadline=prob,
            n_comparables=0,
            hazard_rate=1 / median if median > 0 else 0,
        )

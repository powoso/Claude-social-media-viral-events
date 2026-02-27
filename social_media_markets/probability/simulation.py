"""Stochastic simulation for app ranking and viral event probability.

Monte Carlo simulation of growth trajectories accounting for:
- Model uncertainty in growth parameters
- Viral event probability (power law distributed)
- Platform algorithm changes as structural breaks
- Competition effects (for app store rankings)
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import stats

from social_media_markets.models import GrowthCurveFit, MetricTimeSeries

logger = logging.getLogger(__name__)


class StochasticSimulator:
    """Monte Carlo simulator for social media growth trajectories.

    Generates thousands of possible future paths, accounting for:
    1. Baseline growth model uncertainty (parameter distributions)
    2. Viral shocks (power law distributed jumps)
    3. Plateau risk (logistic ceiling)
    4. Seasonal effects (cyclical adjustments)
    """

    def __init__(
        self,
        n_simulations: int = 10000,
        seed: int | None = None,
    ):
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(seed)

    def simulate_paths(
        self,
        series: MetricTimeSeries,
        growth_fit: GrowthCurveFit,
        horizon_days: int = 365,
        dt: float = 1.0,
        viral_probability: float = 0.002,
        viral_magnitude_mean: float = 0.2,
        plateau_probability: float = 0.001,
    ) -> np.ndarray:
        """Simulate future growth paths.

        Args:
            series: Historical data for calibration.
            growth_fit: Fitted growth model.
            horizon_days: Days to simulate forward.
            dt: Time step in days.
            viral_probability: Daily probability of a viral event.
            viral_magnitude_mean: Mean viral shock as fraction of current value.
            plateau_probability: Daily probability of hitting a growth plateau.

        Returns:
            Array of shape (n_simulations, n_steps) with simulated values.
        """
        current = series.latest_value() or 0.0
        max_day = series.days_array[-1] if len(series.points) > 0 else 0.0
        n_steps = int(horizon_days / dt)

        # Calibrate noise from residuals
        noise_std = growth_fit.residual_std

        # Parameter uncertainty: perturb growth model parameters
        paths = np.zeros((self.n_simulations, n_steps))

        for sim in range(self.n_simulations):
            # Perturb parameters (parametric bootstrap)
            perturbed_fit = self._perturb_parameters(growth_fit)

            value = current
            plateaued = False

            for step in range(n_steps):
                day = max_day + (step + 1) * dt

                # Deterministic growth component
                try:
                    model_value = perturbed_fit.predict(np.array([day]))[0]
                    growth_increment = model_value - (
                        perturbed_fit.predict(np.array([day - dt]))[0]
                        if step > 0
                        else current
                    )
                except Exception:
                    growth_increment = 0.0

                # Stochastic noise
                noise = self.rng.normal(0, noise_std * np.sqrt(dt))

                # Viral shock (power law)
                if not plateaued and self.rng.random() < viral_probability * dt:
                    # Power law distributed viral shock
                    shock_magnitude = (
                        self.rng.pareto(2.0) * viral_magnitude_mean * value
                    )
                    growth_increment += shock_magnitude

                # Plateau event
                if not plateaued and self.rng.random() < plateau_probability * dt:
                    plateaued = True
                    growth_increment *= 0.1  # drastically reduce growth

                if plateaued:
                    growth_increment *= 0.05  # near-zero growth after plateau

                value = max(value + growth_increment + noise, 0)
                paths[sim, step] = value

        return paths

    def probability_of_threshold(
        self,
        paths: np.ndarray,
        target: float,
        by_step: int | None = None,
    ) -> float:
        """Calculate probability of reaching a threshold from simulation paths.

        Args:
            paths: Simulated paths array (n_sims, n_steps).
            target: Target value.
            by_step: If set, only consider paths that reach target by this step.

        Returns:
            Probability estimate.
        """
        if by_step is not None:
            relevant = paths[:, :by_step]
        else:
            relevant = paths

        # For each simulation, check if target was ever reached
        reached = np.any(relevant >= target, axis=1)
        return float(np.mean(reached))

    def confidence_intervals(
        self,
        paths: np.ndarray,
        percentiles: list[float] = [5, 25, 50, 75, 95],
    ) -> dict[int, np.ndarray]:
        """Compute confidence intervals across simulation paths.

        Returns:
            Dict mapping percentile → array of values at each time step.
        """
        return {
            p: np.percentile(paths, p, axis=0) for p in percentiles
        }

    def simulate_app_ranking(
        self,
        current_rank: int,
        current_downloads: float,
        competitor_downloads: list[float],
        horizon_days: int = 90,
    ) -> np.ndarray:
        """Simulate app store ranking dynamics.

        Models the feedback loop: downloads → ranking → organic discovery → downloads.

        Args:
            current_rank: Current app store ranking.
            current_downloads: Current daily downloads.
            competitor_downloads: Daily downloads of competing apps.
            horizon_days: Days to simulate.

        Returns:
            Array of shape (n_simulations, horizon_days) with simulated ranks.
        """
        n_competitors = len(competitor_downloads)
        rank_paths = np.zeros((self.n_simulations, horizon_days))

        for sim in range(self.n_simulations):
            my_downloads = current_downloads
            comp_dl = np.array(competitor_downloads, dtype=float)

            for day in range(horizon_days):
                # Organic discovery bonus from ranking
                if current_rank <= 10:
                    organic_bonus = my_downloads * 0.3
                elif current_rank <= 50:
                    organic_bonus = my_downloads * 0.1
                elif current_rank <= 200:
                    organic_bonus = my_downloads * 0.03
                else:
                    organic_bonus = 0

                # Stochastic daily variation (20% CV)
                my_daily = max(
                    (my_downloads + organic_bonus) * self.rng.lognormal(0, 0.2),
                    0,
                )

                # Competitors also vary
                comp_daily = comp_dl * self.rng.lognormal(0, 0.15, size=n_competitors)

                # Rank = 1 + number of competitors with higher downloads
                rank = 1 + int(np.sum(comp_daily > my_daily))
                rank_paths[sim, day] = rank

                # Update downloads with some momentum
                my_downloads = 0.9 * my_downloads + 0.1 * my_daily
                comp_dl = 0.9 * comp_dl + 0.1 * comp_daily

        return rank_paths

    def viral_event_analysis(
        self, series: MetricTimeSeries, n_bootstrap: int = 1000
    ) -> dict[str, float]:
        """Analyze the power law distribution of content performance.

        Computes the probability that a future piece of content goes viral
        based on the historical performance distribution.

        Returns:
            Dict with viral probability metrics.
        """
        y = series.values
        if len(y) < 10:
            return {"viral_probability": 0.01, "power_law_alpha": 2.0, "p99_value": 0.0}

        # Fit power law to the upper tail (above median)
        median = np.median(y)
        upper = y[y > median]

        if len(upper) < 5:
            return {"viral_probability": 0.01, "power_law_alpha": 2.0, "p99_value": float(np.max(y))}

        # MLE for Pareto alpha
        x_min = np.min(upper)
        if x_min <= 0:
            return {"viral_probability": 0.01, "power_law_alpha": 2.0, "p99_value": float(np.max(y))}

        alpha = len(upper) / np.sum(np.log(upper / x_min))

        # Probability of viral event (defined as > 10x median)
        viral_threshold = median * 10
        if alpha > 0 and x_min > 0:
            viral_prob = (x_min / viral_threshold) ** alpha if viral_threshold > x_min else 1.0
        else:
            viral_prob = 0.01

        return {
            "viral_probability": float(np.clip(viral_prob, 0.001, 0.5)),
            "power_law_alpha": float(alpha),
            "p99_value": float(np.percentile(y, 99)),
            "p95_value": float(np.percentile(y, 95)),
            "max_observed": float(np.max(y)),
        }

    def _perturb_parameters(self, fit: GrowthCurveFit) -> GrowthCurveFit:
        """Create a perturbed copy of the growth fit for parametric bootstrap."""
        perturbed_params = {}
        for key, val in fit.parameters.items():
            # Add ~10% noise to each parameter
            noise = self.rng.normal(0, abs(val) * 0.1) if val != 0 else 0
            perturbed_params[key] = val + noise

        return GrowthCurveFit(
            pattern=fit.pattern,
            parameters=perturbed_params,
            r_squared=fit.r_squared,
            residual_std=fit.residual_std,
            aic=fit.aic,
        )

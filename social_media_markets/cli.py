"""Command-line interface for social media prediction markets."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone

from social_media_markets.config import APIConfig, MetricType, Platform
from social_media_markets.models import MarketQuestion


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="smm",
        description="Social Media Prediction Markets — probability engine",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── collect ─────────────────────────────────────────────────
    collect_p = subparsers.add_parser("collect", help="Collect data for an entity")
    collect_p.add_argument("entity", help="Entity name (channel, app, subreddit, etc.)")
    collect_p.add_argument(
        "--platform",
        choices=[p.value for p in Platform],
        default="youtube",
        help="Platform to collect from",
    )
    collect_p.add_argument(
        "--source",
        choices=["socialblade", "google_trends", "reddit", "wikipedia", "wayback", "app_store", "twitch", "spotify"],
        default="socialblade",
        help="Data source to use",
    )
    collect_p.add_argument("--output", "-o", help="Output JSON file path")

    # ── fit ─────────────────────────────────────────────────────
    fit_p = subparsers.add_parser("fit", help="Fit growth curves to collected data")
    fit_p.add_argument("data_file", help="JSON file with time series data")
    fit_p.add_argument("--all-models", action="store_true", help="Show all model fits")

    # ── predict ─────────────────────────────────────────────────
    predict_p = subparsers.add_parser("predict", help="Predict probability of milestone")
    predict_p.add_argument("entity", help="Entity name")
    predict_p.add_argument("target", type=float, help="Target value to reach")
    predict_p.add_argument(
        "--platform",
        choices=[p.value for p in Platform],
        default="youtube",
    )
    predict_p.add_argument(
        "--metric",
        choices=[m.value for m in MetricType],
        default="subscribers",
    )
    predict_p.add_argument("--deadline", help="Deadline date (YYYY-MM-DD)")
    predict_p.add_argument(
        "--market-price", type=float, help="Current market probability (0-1)"
    )
    predict_p.add_argument("--data-file", help="Pre-collected data JSON file")
    predict_p.add_argument("--simulations", type=int, default=10000, help="Monte Carlo simulations")

    # ── edge ───────────────────────────────────────────────────
    edge_p = subparsers.add_parser("edge", help="Detect market edge for a question")
    edge_p.add_argument("entity", help="Entity name")
    edge_p.add_argument("target", type=float, help="Target value")
    edge_p.add_argument("market_price", type=float, help="Current market price (0-1)")
    edge_p.add_argument(
        "--platform",
        choices=[p.value for p in Platform],
        default="youtube",
    )
    edge_p.add_argument("--data-file", help="Pre-collected data JSON file")

    args = parser.parse_args(argv)

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command is None:
        parser.print_help()
        return 0

    config = APIConfig()
    warnings = config.validate()
    for w in warnings:
        logging.warning(w)

    if args.command == "collect":
        return _cmd_collect(args, config)
    elif args.command == "fit":
        return _cmd_fit(args)
    elif args.command == "predict":
        return _cmd_predict(args, config)
    elif args.command == "edge":
        return _cmd_edge(args, config)

    return 0


def _cmd_collect(args, config: APIConfig) -> int:
    """Execute the collect command."""
    from social_media_markets.collectors import (
        AppStoreCollector,
        GoogleTrendsCollector,
        RedditCollector,
        SocialBladeCollector,
        SpotifyCollector,
        TwitchCollector,
        WaybackCollector,
        WikipediaCollector,
    )

    collectors = {
        "socialblade": lambda: SocialBladeCollector(config),
        "google_trends": lambda: GoogleTrendsCollector(config),
        "reddit": lambda: RedditCollector(config),
        "wikipedia": lambda: WikipediaCollector(config),
        "wayback": lambda: WaybackCollector(config),
        "app_store": lambda: AppStoreCollector(config),
        "twitch": lambda: TwitchCollector(config),
        "spotify": lambda: SpotifyCollector(config),
    }

    collector = collectors[args.source]()
    platform = Platform(args.platform)

    print(f"Collecting {args.source} data for '{args.entity}' ({platform.value})...")

    kwargs = {}
    if hasattr(collector, "collect") and "platform" in collector.collect.__code__.co_varnames:
        kwargs["platform"] = platform

    series_list = collector.collect(args.entity, **kwargs)

    if not series_list:
        print("No data collected.")
        return 1

    # Serialize to JSON
    output = []
    for s in series_list:
        output.append({
            "platform": s.platform.value,
            "metric_type": s.metric_type.value,
            "entity_name": s.entity_name,
            "n_points": len(s.points),
            "points": [
                {
                    "timestamp": p.timestamp.isoformat(),
                    "value": p.value,
                    "source": p.source,
                }
                for p in s.points
            ],
        })

    json_str = json.dumps(output, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(json_str)
        print(f"Data written to {args.output}")
    else:
        print(json_str)

    for s in series_list:
        latest = s.latest_value()
        rate = s.growth_rate(window_days=30)
        print(f"\n{s.platform.value}/{s.metric_type.value}: {len(s.points)} points")
        if latest is not None:
            print(f"  Latest: {latest:,.0f}")
        if rate is not None:
            print(f"  30d growth rate: {rate:,.1f}/day")

    return 0


def _cmd_fit(args) -> int:
    """Execute the fit command."""
    from social_media_markets.features.growth_curves import GrowthCurveFitter
    from social_media_markets.models import MetricTimeSeries, TimeSeriesPoint

    with open(args.data_file) as f:
        data = json.load(f)

    fitter = GrowthCurveFitter()

    for entry in data:
        points = [
            TimeSeriesPoint(
                timestamp=datetime.fromisoformat(p["timestamp"]),
                value=p["value"],
                source=p.get("source", ""),
            )
            for p in entry["points"]
        ]
        series = MetricTimeSeries(
            platform=Platform(entry["platform"]),
            metric_type=MetricType(entry["metric_type"]),
            entity_name=entry["entity_name"],
            points=points,
        )

        print(f"\n{'='*60}")
        print(f"{series.entity_name} — {series.platform.value}/{series.metric_type.value}")
        print(f"{'='*60}")

        if args.all_models:
            fits = fitter.fit_all(series)
            for fit in fits:
                _print_fit(fit)
        else:
            fit = fitter.fit(series)
            if fit:
                _print_fit(fit)
            else:
                print("  Insufficient data for fitting.")

    return 0


def _print_fit(fit) -> None:
    """Pretty-print a growth curve fit."""
    print(f"\n  Pattern: {fit.pattern.value}")
    print(f"  R²: {fit.r_squared:.4f}")
    print(f"  AIC: {fit.aic:.1f}")
    print(f"  Residual σ: {fit.residual_std:,.1f}")
    print(f"  Parameters:")
    for k, v in fit.parameters.items():
        print(f"    {k}: {v:,.4f}")


def _cmd_predict(args, config: APIConfig) -> int:
    """Execute the predict command."""
    from social_media_markets.models import MetricTimeSeries, TimeSeriesPoint
    from social_media_markets.probability.engine import ProbabilityEngine

    # Load or collect data
    series = _load_or_collect(args, config)
    if series is None:
        print("No data available for prediction.")
        return 1

    # Build question
    deadline = None
    if args.deadline:
        deadline = datetime.strptime(args.deadline, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    question = MarketQuestion(
        question_text=f"Will {args.entity} reach {args.target:,.0f} {args.metric}?",
        platform=Platform(args.platform),
        metric_type=MetricType(args.metric),
        entity_name=args.entity,
        target_value=args.target,
        deadline=deadline,
        current_market_price=args.market_price,
    )

    engine = ProbabilityEngine(n_simulations=args.simulations)
    result = engine.predict(question, series)

    # Display results
    print(f"\n{'='*60}")
    print(f"  {question}")
    print(f"{'='*60}")
    print(f"  Probability: {result.probability:.1%}")
    print(f"  90% CI: [{result.confidence_interval[0]:.1%}, {result.confidence_interval[1]:.1%}]")
    print(f"  Raw model: {result.model_probability:.1%}")
    if question.current_market_price is not None:
        print(f"  Market price: {question.current_market_price:.1%}")
        print(f"  Edge: {result.edge:+.1%}")
    if result.growth_fit:
        print(f"  Growth pattern: {result.growth_fit.pattern.value}")
    print(f"\n  Reasoning: {result.reasoning}")

    return 0


def _cmd_edge(args, config: APIConfig) -> int:
    """Execute the edge detection command."""
    from social_media_markets.edge.detector import EdgeDetector
    from social_media_markets.probability.engine import ProbabilityEngine

    series = _load_or_collect(args, config)
    if series is None:
        print("No data available for edge detection.")
        return 1

    question = MarketQuestion(
        question_text=f"Will {args.entity} reach {args.target:,.0f}?",
        platform=Platform(args.platform),
        metric_type=MetricType.SUBSCRIBERS,
        entity_name=args.entity,
        target_value=args.target,
        current_market_price=args.market_price,
    )

    # First get model probability
    engine = ProbabilityEngine()
    result = engine.predict(question, series)

    # Then detect edge
    detector = EdgeDetector()
    edge = detector.analyze(question, series, model_probability=result.probability)

    print(f"\n{'='*60}")
    print(f"  Edge Analysis: {question}")
    print(f"{'='*60}")
    print(f"  Net edge: {edge.net_edge:+.1%}")
    print(f"  Confidence: {edge.confidence:.1%}")
    print(f"  Recommendation: {edge.recommendation.upper()}")
    print(f"\n  Signals:")
    for s in edge.signals:
        arrow = "↑" if s.direction == "under" else "↓"
        print(f"    {arrow} [{s.name}] {s.explanation}")
    if edge.has_actionable_edge:
        print(f"\n  ** ACTIONABLE EDGE DETECTED **")

    return 0


def _load_or_collect(args, config: APIConfig):
    """Load data from file or collect from API."""
    from social_media_markets.collectors.socialblade import SocialBladeCollector
    from social_media_markets.models import MetricTimeSeries, TimeSeriesPoint

    if hasattr(args, "data_file") and args.data_file:
        with open(args.data_file) as f:
            data = json.load(f)
        if not data:
            return None
        entry = data[0]
        points = [
            TimeSeriesPoint(
                timestamp=datetime.fromisoformat(p["timestamp"]),
                value=p["value"],
                source=p.get("source", ""),
            )
            for p in entry["points"]
        ]
        return MetricTimeSeries(
            platform=Platform(entry["platform"]),
            metric_type=MetricType(entry["metric_type"]),
            entity_name=entry["entity_name"],
            points=points,
        )

    # Try to collect from Social Blade
    collector = SocialBladeCollector(config)
    platform = Platform(args.platform)
    series_list = collector.collect(args.entity, platform=platform)
    return series_list[0] if series_list else None


if __name__ == "__main__":
    sys.exit(main())

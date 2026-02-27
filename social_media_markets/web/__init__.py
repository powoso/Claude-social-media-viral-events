"""FastAPI web application for Social Media Prediction Markets."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from social_media_markets.config import APIConfig, GrowthPattern, MetricType, Platform
from social_media_markets.edge.detector import EdgeDetector
from social_media_markets.features.controversy import ControversyDetector
from social_media_markets.features.cross_platform import CrossPlatformMomentum
from social_media_markets.features.growth_curves import GrowthCurveFitter
from social_media_markets.features.seasonality import SeasonalityAnalyzer
from social_media_markets.models import (
    GrowthCurveFit,
    MarketQuestion,
    MetricTimeSeries,
    TimeSeriesPoint,
)
from social_media_markets.probability.engine import ProbabilityEngine
from social_media_markets.probability.simulation import StochasticSimulator

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
app = FastAPI(title="Social Media Prediction Markets", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ── Request/Response Models ────────────────────────────────────


class PredictRequest(BaseModel):
    entity_name: str
    platform: str = "youtube"
    metric_type: str = "subscribers"
    target_value: float
    deadline_days: int | None = None
    market_price: float | None = None
    n_simulations: int = 5000

    # Optional: user-provided historical data (list of {date, value})
    historical_data: list[dict] | None = None


class GrowthFitRequest(BaseModel):
    data_points: list[dict]  # [{date: "YYYY-MM-DD", value: float}]
    entity_name: str = "entity"


class SimulationRequest(BaseModel):
    entity_name: str = "entity"
    data_points: list[dict]
    horizon_days: int = 365
    n_simulations: int = 2000
    target_value: float | None = None


class EdgeRequest(BaseModel):
    entity_name: str
    target_value: float
    market_price: float
    platform: str = "youtube"
    data_points: list[dict] | None = None


class DemoRequest(BaseModel):
    scenario: str  # "linear", "exponential", "logistic", "viral", "plateau"


# ── Helper Functions ───────────────────────────────────────────


def _build_series(
    data_points: list[dict],
    entity_name: str = "entity",
    platform: Platform = Platform.YOUTUBE,
    metric_type: MetricType = MetricType.SUBSCRIBERS,
) -> MetricTimeSeries:
    """Build a MetricTimeSeries from API request data points."""
    points = []
    for dp in data_points:
        try:
            ts = datetime.fromisoformat(dp["date"]).replace(tzinfo=timezone.utc)
            points.append(TimeSeriesPoint(timestamp=ts, value=float(dp["value"]), source="user"))
        except (KeyError, ValueError):
            continue
    points.sort(key=lambda p: p.timestamp)
    return MetricTimeSeries(
        platform=platform,
        metric_type=metric_type,
        entity_name=entity_name,
        points=points,
    )


def _generate_demo_data(scenario: str) -> list[dict]:
    """Generate synthetic demo data for a given growth scenario."""
    rng = np.random.default_rng(42)
    base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    n_days = 180
    points = []

    for i in range(n_days):
        date = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
        noise = rng.normal(0, 1)

        if scenario == "linear":
            value = 10000 + 150 * i + noise * 200
        elif scenario == "exponential":
            value = 5000 * np.exp(0.015 * i) + noise * 500
        elif scenario == "logistic":
            value = 500000 / (1 + np.exp(-0.06 * (i - 90))) + noise * 2000
        elif scenario == "viral":
            base = 200000 / (1 + np.exp(-0.08 * (i - 60)))
            spike = 0
            if 75 <= i <= 95:
                spike = 80000 * np.exp(-0.15 * (i - 75))
            value = base + spike + noise * 3000
        elif scenario == "plateau":
            value = 100000 * (1 - np.exp(-0.025 * i)) + noise * 1000
        else:
            value = 10000 + 100 * i + noise * 200

        points.append({"date": date, "value": max(value, 0)})

    return points


def _serialize_numpy(obj):
    """JSON serializer that handles numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ── Routes: Pages ──────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


# ── Routes: API ────────────────────────────────────────────────


@app.post("/api/predict")
async def api_predict(req: PredictRequest):
    """Run the full probability engine and return results."""
    platform = Platform(req.platform)
    metric_type = MetricType(req.metric_type)

    # Build series from provided data or generate demo
    if req.historical_data:
        series = _build_series(req.historical_data, req.entity_name, platform, metric_type)
    else:
        demo_data = _generate_demo_data("linear")
        series = _build_series(demo_data, req.entity_name, platform, metric_type)

    if len(series.points) < 5:
        return JSONResponse({"error": "Need at least 5 data points"}, status_code=400)

    deadline = None
    if req.deadline_days:
        deadline = datetime.now(timezone.utc) + timedelta(days=req.deadline_days)

    question = MarketQuestion(
        question_text=f"Will {req.entity_name} reach {req.target_value:,.0f} {req.metric_type}?",
        platform=platform,
        metric_type=metric_type,
        entity_name=req.entity_name,
        target_value=req.target_value,
        deadline=deadline,
        current_market_price=req.market_price,
    )

    engine = ProbabilityEngine(n_simulations=req.n_simulations, seed=42)
    result = engine.predict(question, series)

    return JSONResponse(
        {
            "probability": round(result.probability, 4),
            "confidence_interval": [
                round(result.confidence_interval[0], 4),
                round(result.confidence_interval[1], 4),
            ],
            "model_probability": round(result.model_probability, 4),
            "edge": round(result.edge, 4) if result.edge else 0,
            "growth_pattern": result.growth_fit.pattern.value if result.growth_fit else None,
            "r_squared": round(result.growth_fit.r_squared, 4) if result.growth_fit else None,
            "reasoning": result.reasoning,
            "features": {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in result.features_used.items()
            },
        }
    )


@app.post("/api/fit")
async def api_fit(req: GrowthFitRequest):
    """Fit growth curves to provided data."""
    series = _build_series(req.data_points, req.entity_name)
    if len(series.points) < 5:
        return JSONResponse({"error": "Need at least 5 data points"}, status_code=400)

    fitter = GrowthCurveFitter()
    fits = fitter.fit_all(series)

    results = []
    for fit in fits:
        # Generate prediction curve
        t = series.days_array
        future_t = np.linspace(0, t[-1] * 1.3, 200)
        try:
            predicted = fit.predict(future_t)
        except Exception:
            predicted = np.zeros_like(future_t)

        base_date = series.points[0].timestamp
        dates = [(base_date + timedelta(days=float(d))).strftime("%Y-%m-%d") for d in future_t]

        results.append({
            "pattern": fit.pattern.value,
            "r_squared": round(fit.r_squared, 4),
            "aic": round(fit.aic, 2),
            "residual_std": round(fit.residual_std, 2),
            "parameters": {k: round(v, 4) for k, v in fit.parameters.items()},
            "prediction": {
                "dates": dates,
                "values": [round(float(v), 2) for v in predicted],
            },
        })

    return JSONResponse({"fits": results})


@app.post("/api/simulate")
async def api_simulate(req: SimulationRequest):
    """Run Monte Carlo simulation and return path statistics."""
    series = _build_series(req.data_points, req.entity_name)
    if len(series.points) < 5:
        return JSONResponse({"error": "Need at least 5 data points"}, status_code=400)

    fitter = GrowthCurveFitter()
    fit = fitter.fit(series)
    if fit is None:
        return JSONResponse({"error": "Could not fit growth model"}, status_code=400)

    sim = StochasticSimulator(n_simulations=req.n_simulations, seed=42)

    viral_stats = sim.viral_event_analysis(series)
    paths = sim.simulate_paths(
        series, fit,
        horizon_days=req.horizon_days,
        viral_probability=viral_stats.get("viral_probability", 0.002),
    )

    ci = sim.confidence_intervals(paths, percentiles=[5, 25, 50, 75, 95])

    base_date = series.points[-1].timestamp
    dates = [
        (base_date + timedelta(days=i + 1)).strftime("%Y-%m-%d")
        for i in range(req.horizon_days)
    ]

    result = {
        "dates": dates,
        "percentiles": {
            str(p): [round(float(v), 2) for v in vals]
            for p, vals in ci.items()
        },
        "growth_pattern": fit.pattern.value,
        "viral_stats": {k: round(float(v), 4) for k, v in viral_stats.items()},
    }

    if req.target_value is not None:
        prob = sim.probability_of_threshold(paths, req.target_value)
        result["target_probability"] = round(prob, 4)

    return JSONResponse(result)


@app.post("/api/edge")
async def api_edge(req: EdgeRequest):
    """Run edge detection analysis."""
    platform = Platform(req.platform)

    if req.data_points:
        series = _build_series(req.data_points, req.entity_name, platform)
    else:
        demo_data = _generate_demo_data("logistic")
        series = _build_series(demo_data, req.entity_name, platform)

    question = MarketQuestion(
        question_text=f"Will {req.entity_name} reach {req.target_value:,.0f}?",
        platform=platform,
        metric_type=MetricType.SUBSCRIBERS,
        entity_name=req.entity_name,
        target_value=req.target_value,
        current_market_price=req.market_price,
    )

    engine = ProbabilityEngine(n_simulations=2000, seed=42)
    prediction = engine.predict(question, series)

    detector = EdgeDetector()
    edge = detector.analyze(question, series, model_probability=prediction.probability)

    signals = []
    for s in edge.signals:
        signals.append({
            "name": s.name,
            "direction": s.direction,
            "magnitude": round(s.magnitude, 4),
            "confidence": round(s.confidence, 4),
            "explanation": s.explanation,
        })

    return JSONResponse({
        "net_edge": round(edge.net_edge, 4),
        "confidence": round(edge.confidence, 4),
        "recommendation": edge.recommendation,
        "has_actionable_edge": edge.has_actionable_edge,
        "model_probability": round(prediction.probability, 4),
        "market_price": req.market_price,
        "signals": signals,
    })


@app.post("/api/demo-data")
async def api_demo_data(req: DemoRequest):
    """Generate demo data for a given growth scenario."""
    data = _generate_demo_data(req.scenario)
    return JSONResponse({"data": data, "scenario": req.scenario})


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


def run():
    """Entry point for running the web server."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run()

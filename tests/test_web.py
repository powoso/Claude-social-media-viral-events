"""Tests for the web API endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from social_media_markets.web import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


class TestDemoDataEndpoint:
    @pytest.mark.parametrize("scenario", ["linear", "exponential", "logistic", "viral", "plateau"])
    def test_demo_data_scenarios(self, client, scenario):
        resp = client.post("/api/demo-data", json={"scenario": scenario})
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data
        assert len(data["data"]) > 0
        assert "date" in data["data"][0]
        assert "value" in data["data"][0]


class TestPredictEndpoint:
    def test_predict_with_demo_data(self, client):
        # First get demo data
        demo_resp = client.post("/api/demo-data", json={"scenario": "linear"})
        demo_data = demo_resp.json()["data"]

        resp = client.post("/api/predict", json={
            "entity_name": "TestChannel",
            "target_value": 50000,
            "deadline_days": 365,
            "market_price": 0.5,
            "historical_data": demo_data,
            "n_simulations": 500,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "probability" in data
        assert 0 <= data["probability"] <= 1
        assert "confidence_interval" in data
        assert "reasoning" in data
        assert data["growth_pattern"] is not None

    def test_predict_insufficient_data(self, client):
        resp = client.post("/api/predict", json={
            "entity_name": "Test",
            "target_value": 1000,
            "historical_data": [{"date": "2024-01-01", "value": 100}],
        })
        assert resp.status_code == 400


class TestFitEndpoint:
    def test_fit_growth_curves(self, client):
        demo_resp = client.post("/api/demo-data", json={"scenario": "logistic"})
        demo_data = demo_resp.json()["data"]

        resp = client.post("/api/fit", json={
            "data_points": demo_data,
            "entity_name": "test",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "fits" in data
        assert len(data["fits"]) > 0
        fit = data["fits"][0]
        assert "pattern" in fit
        assert "r_squared" in fit
        assert "prediction" in fit


class TestSimulateEndpoint:
    def test_simulate(self, client):
        demo_resp = client.post("/api/demo-data", json={"scenario": "exponential"})
        demo_data = demo_resp.json()["data"]

        resp = client.post("/api/simulate", json={
            "data_points": demo_data,
            "entity_name": "test",
            "horizon_days": 90,
            "n_simulations": 200,
            "target_value": 100000,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "dates" in data
        assert "percentiles" in data
        assert "5" in data["percentiles"]
        assert "50" in data["percentiles"]
        assert "95" in data["percentiles"]
        assert "target_probability" in data


class TestEdgeEndpoint:
    def test_edge_detection(self, client):
        demo_resp = client.post("/api/demo-data", json={"scenario": "exponential"})
        demo_data = demo_resp.json()["data"]

        resp = client.post("/api/edge", json={
            "entity_name": "TestChannel",
            "target_value": 200000,
            "market_price": 0.5,
            "data_points": demo_data,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "net_edge" in data
        assert "recommendation" in data
        assert data["recommendation"] in ("buy", "sell", "pass")
        assert "signals" in data


class TestIndexPage:
    def test_index_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Prediction Markets" in resp.text

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_dashboard_contract():
    """Verify that /api/dashboard returns every field expected by the frontend."""
    response = client.get("/api/dashboard")
    assert response.status_code == 200
    data = response.json()

    # Core stats cards
    assert "stats" in data
    stats = data["stats"]
    assert "reported_cases" in stats
    assert "pct_change_vs_previous" in stats
    assert "predicted_next_week" in stats
    assert "high_risk_wards" in stats
    assert "total_wards" in stats
    assert "population_covered" in stats
    assert "population_covered_formatted" in stats

    # Weekly chart series
    assert "weekly_data" in data
    assert isinstance(data["weekly_data"], list)
    assert len(data["weekly_data"]) > 0
    for pt in data["weekly_data"]:
        assert "week" in pt
        assert "cases" in pt
        assert "predicted" in pt

    # Ward data
    assert "ward_data" in data
    assert isinstance(data["ward_data"], list)
    for w in data["ward_data"]:
        assert "name" in w
        assert "cases" in w
        assert "risk" in w
        assert "color" in w
        assert "position" in w

    # Weather
    assert "weather" in data
    wthr = data["weather"]
    assert "condition" in wthr
    assert "temperature_c" in wthr
    assert "rainfall_mm" in wthr
    assert "humidity_pct" in wthr

    # Alerts
    assert "alerts" in data
    assert isinstance(data["alerts"], list)

def test_wards_endpoint_contract():
    response = client.get("/api/wards")
    assert response.status_code == 200
    data = response.json()
    assert data["district"] == "Coimbatore"
    assert data["total_wards"] >= 5
    assert len(data["wards"]) >= 5
    first_ward = data["wards"][0]
    assert first_ward["name"] == "Ward 01"
    assert first_ward["population"] > 0
    assert len(first_ward["position"]) == 2

def test_risk_endpoint_contract():
    response = client.get("/api/risk?disease=Dengue")
    assert response.status_code == 200
    data = response.json()
    assert data["disease"] == "Dengue"
    assert "wards" in data
    assert len(data["wards"]) >= 5
    for r in data["wards"]:
        assert r["risk_level"] in ["High", "Medium", "Low"]
        assert r["attack_rate_per_1000"] >= 0

def test_model_metrics_contract():
    response = client.get("/api/model/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "status" in data
    assert "supported_horizons" in data
    assert "required_features" in data
    # When model is not loaded, status must be unavailable
    if not data["model_loaded"]:
        assert data["status"] == "unavailable"
        assert data["mae"] is None

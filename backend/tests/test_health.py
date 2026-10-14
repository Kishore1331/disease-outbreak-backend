import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "Epidemiology" in data["name"] or "OutbreakWatch" in data["name"]
    assert data["health_check"] == "/api/health"
    assert data["dashboard"] == "/api/dashboard"

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_ready" in data
    assert "components" in data
    assert "model_service" in data["components"]
    assert "weather_service" in data["components"]
    assert "data_service" in data["components"]

    # When model weights are absent, model_ready must be False and status transparent
    if not data["model_ready"]:
        assert data["model_status"] == "model_artifact_missing"

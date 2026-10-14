import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_weather_endpoint():
    response = client.get("/api/weather")
    assert response.status_code == 200
    data = response.json()
    assert "current" in data
    curr = data["current"]
    assert "location" in curr
    assert "Coimbatore" in curr["location"]
    assert "temperature_c" in curr
    assert "rainfall_mm" in curr
    assert "humidity_pct" in curr
    assert "source" in curr
    assert curr["temperature_c"] is not None
    assert curr["rainfall_mm"] >= 0
    assert 0 <= curr["humidity_pct"] <= 100

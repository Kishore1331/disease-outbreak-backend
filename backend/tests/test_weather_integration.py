import pytest
from unittest.mock import patch
import httpx
from fastapi.testclient import TestClient
from backend.main import app
from backend.services.weather_service import weather_service
from backend.services.prediction_service import prediction_service
from backend.schemas import PredictionRequest

client = TestClient(app)

@pytest.mark.anyio
async def test_open_meteo_live_fetch_and_schema():
    """1 & 2. Verify Open-Meteo live API request succeeds and returns real/valid data."""
    weather_resp = await weather_service.get_weather_with_forecast(force_refresh=True)
    assert weather_resp is not None
    assert weather_resp.current is not None

    curr = weather_resp.current
    assert curr.source in ["open-meteo", "climatological_baseline"]
    assert -10.0 <= curr.temperature_c <= 55.0
    assert curr.rainfall_mm >= 0.0
    assert 0.0 <= curr.humidity_pct <= 100.0
    assert curr.observed_at is not None
    assert curr.retrieved_at is not None
    assert curr.is_forecast is False

    # Check 7-day forecast series
    assert weather_resp.forecast_7d is not None
    assert len(weather_resp.forecast_7d) == 7
    for day in weather_resp.forecast_7d:
        assert day.date is not None
        assert day.temperature_max_c >= day.temperature_min_c
        assert day.precipitation_sum_mm >= 0.0

def test_weather_endpoint_with_ward_coordinates():
    """Verify /api/weather endpoint supports ward-specific coordinate resolution."""
    response = client.get("/api/weather?ward=Ward%2001")
    assert response.status_code == 200
    data = response.json()
    assert data["current"]["latitude"] == 11.0168
    assert data["current"]["longitude"] == 76.9558
    assert "retrieved_at" in data
    assert data["forecast_7d"] is not None
    assert len(data["forecast_7d"]) == 7

@pytest.mark.anyio
async def test_weather_caching_and_ttl():
    """Verify weather caching works with configurable TTL and remaining seconds."""
    # First fetch (fresh)
    res1 = await weather_service.get_weather_with_forecast(force_refresh=True)
    assert res1.cached is False

    # Second fetch immediately after (cached)
    res2 = await weather_service.get_weather_with_forecast(force_refresh=False)
    assert res2.cached is True
    assert res2.cache_expires_in_seconds is not None
    assert res2.cache_expires_in_seconds > 0
    assert res2.current.temperature_c == res1.current.temperature_c

def test_weather_coordinate_validation():
    """Verify coordinate bounds checking (-90 to 90 lat, -180 to 180 lon)."""
    is_valid, err = weather_service.validate_coordinates(95.0, 76.9)
    assert is_valid is False
    assert "Invalid latitude" in err

    is_valid, err = weather_service.validate_coordinates(11.0, 205.0)
    assert is_valid is False
    assert "Invalid longitude" in err

    is_valid, err = weather_service.validate_coordinates(11.0168, 76.9558)
    assert is_valid is True
    assert err is None

@pytest.mark.anyio
async def test_model_feature_construction_with_weather():
    """3 & 4. Verify correct 7 model features are constructed and model loads."""
    req = PredictionRequest(disease="Dengue", ward="Ward 01", horizon_days=7)
    pred_res = await prediction_service.generate_prediction(req)

    assert pred_res.predicted_cases is not None
    assert pred_res.rounded_prediction >= 0
    assert pred_res.risk_level in ["Low", "Medium", "High", "Critical"]
    assert pred_res.model_status in ["active_ml_model", "epidemiological_baseline_fallback"]

    # Verify input features used match exact 7 dimensions expected by trained model
    feats = pred_res.input_features_used
    assert "cases_lag_1" in feats
    assert "cases_lag_2" in feats
    assert "cases_lag_3" in feats
    assert "temperature_mean_c" in feats
    assert "rainfall_sum_mm" in feats
    assert "humidity_mean_pct" in feats
    assert "ward_population" in feats
    assert feats["ward_population"] == 13534.0

@pytest.mark.anyio
async def test_missing_ward_cases_guard():
    """6. Verify missing dengue data prevents an unsupported prediction."""
    with pytest.raises(ValueError) as excinfo:
        await prediction_service.generate_prediction(
            PredictionRequest(disease="Dengue", ward="NonExistentWardXYZ", horizon_days=7)
        )
    assert "not found in Coimbatore municipal surveillance area" in str(excinfo.value)

@pytest.mark.anyio
async def test_weather_timeout_fallback_handling():
    """7. Verify API failures (e.g. timeout) are handled cleanly with fallback."""
    with patch("httpx.AsyncClient.get", side_effect=httpx.TimeoutException("Mock Open-Meteo timeout")):
        # Request with new coordinates to bypass any existing cache
        res = await weather_service.get_weather_with_forecast(lat=10.5, lon=77.5, force_refresh=True)
        assert res is not None
        assert res.current.status == "baseline_fallback"
        assert res.current.is_real is False
        assert "unreachable" in res.current.notes.lower()

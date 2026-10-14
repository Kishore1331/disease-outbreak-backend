from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from backend.services.prediction_service import prediction_service
from backend.services.data_service import data_service
from backend.services.weather_service import weather_service
from backend.ml.forecasting import ForecastingEngine
from backend.schemas import (
    PredictionRequest,
    PredictionResponse,
    ForecastResponse,
    WeeklyDataPoint
)

router = APIRouter(prefix="/api", tags=["Prediction & Forecasting"])

@router.post("/predict", response_model=PredictionResponse)
async def predict_disease_outbreak(request: PredictionRequest):
    """
    Predicts disease case counts for a given ward and forecast horizon.
    Validates disease, horizon (7, 14, 21 days), and environmental parameters.
    """
    try:
        response = await prediction_service.generate_prediction(request)
        return response
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected prediction error: {str(e)}")

@router.get("/forecast", response_model=ForecastResponse)
async def get_forecast_series(
    disease: str = Query("Dengue", description="Disease name"),
    ward: str = Query("All wards", description="Ward name or 'All wards'"),
    horizon_days: int = Query(7, description="Forecast horizon in days (7, 14, 21)")
):
    """
    Generates chronological epidemiological forecast series for the requested weekly horizon.
    """
    is_valid, err = ForecastingEngine.validate_horizon(horizon_days)
    if not is_valid:
        raise HTTPException(status_code=400, detail=err)

    # Base historical series
    history = data_service.get_historical_series()
    # Filter only known historical points
    past_points = [p for p in history if p.cases is not None]

    # Predict next point
    pred_req = PredictionRequest(disease=disease, ward=ward, horizon_days=horizon_days)
    pred_res = await prediction_service.generate_prediction(pred_req)

    # Construct series
    series = [WeeklyDataPoint(week=p.week, cases=p.cases, predicted=None) for p in past_points]
    
    # Add predicted point
    next_week_label = "Oct 13" if horizon_days == 7 else f"+{horizon_days}d"
    series.append(WeeklyDataPoint(
        week=next_week_label,
        cases=None,
        predicted=pred_res.rounded_prediction
    ))

    return ForecastResponse(
        disease=disease,
        ward=ward,
        horizon_days=horizon_days,
        model_status=pred_res.model_status,
        historical_weeks=len(past_points),
        series=series,
        summary={
            "predicted_cases": pred_res.predicted_cases,
            "risk_level": pred_res.risk_level,
            "confidence_interval": pred_res.confidence_interval
        }
    )

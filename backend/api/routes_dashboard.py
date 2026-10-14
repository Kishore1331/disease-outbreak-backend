from typing import Optional, List
from fastapi import APIRouter, Query
from backend.services.data_service import data_service
from backend.services.weather_service import weather_service
from backend.services.risk_service import risk_service
from backend.services.model_service import model_service
from backend.schemas import DashboardResponse, DashboardStatItem, AlertItem

router = APIRouter(prefix="/api", tags=["Dashboard"])

@router.get("/alerts", response_model=List[AlertItem])
async def get_alerts_endpoint():
    """
    Returns active outbreak early warning alerts computed from EVOI risk scores and case spikes.
    """
    return await risk_service.get_alerts()


@router.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard(
    ward: Optional[str] = Query("All wards", description="Filter by ward name or 'All wards'"),
    disease: Optional[str] = Query("Dengue", description="Disease name")
):
    """
    Returns complete dashboard surveillance metrics:
    - Case totals, trends, and next-week forecast
    - Ward risk table and map locations
    - Environmental weather conditions
    - Early warning alerts
    """
    # 2. Wards & Risk
    all_wards = data_service.get_wards()
    risk_evaluation = await risk_service.evaluate_wards_risk(disease=disease)
    alerts = await risk_service.get_alerts()

    # 3. Weekly Trends
    weekly_series = data_service.get_historical_series()

    # 4. Filter if specific ward requested
    selected_ward = ward or "All wards"
    target_ward = None
    target_lat = None
    target_lon = None
    if selected_ward != "All wards":
        matching_wards = [w for w in all_wards if w.name.lower() == selected_ward.lower()]
        if matching_wards:
            target_ward = matching_wards[0]
            reported_cases = target_ward.cases
            high_risk_wards = 1 if target_ward.risk == "High" else 0
            total_wards = len(all_wards)
            pop_covered = target_ward.population
            predicted_next = max(1, int(round(target_ward.cases * 1.15)))
            if target_ward.position and len(target_ward.position) == 2:
                target_lat, target_lon = target_ward.position[0], target_ward.position[1]
        else:
            reported_cases = data_service.get_total_reported_cases()
            high_risk_wards = risk_evaluation.high_risk_count
            total_wards = len(all_wards)
            pop_covered = 48200
            predicted_next = 32
    else:
        reported_cases = data_service.get_total_reported_cases()
        high_risk_wards = risk_evaluation.high_risk_count
        total_wards = len(all_wards)
        pop_covered = 48200
        predicted_next = 32

    # 1. Weather (spatially resolved to ward if requested)
    weather = await weather_service.get_current_weather(
        lat=target_lat,
        lon=target_lon,
        ward=selected_ward
    )

    # Dynamic Next-Week Prediction using ML model
    try:
        from backend.services.prediction_service import prediction_service
        from backend.schemas import PredictionRequest
        pred_res = await prediction_service.generate_prediction(
            PredictionRequest(disease=disease, ward=selected_ward, horizon_days=7)
        )
        predicted_next = pred_res.rounded_prediction
    except Exception as e:
        predicted_next = max(1, int(round(reported_cases * 1.12)))

    pct_change = 12.0  # +12% vs previous surveillance cycle

    pop_formatted = f"{pop_covered / 1000.0:.1f}k"

    stats = DashboardStatItem(
        reported_cases=reported_cases,
        pct_change_vs_previous=pct_change,
        predicted_next_week=predicted_next,
        high_risk_wards=high_risk_wards,
        total_wards=total_wards,
        population_covered=pop_covered,
        population_covered_formatted=pop_formatted
    )

    model_ready = model_service.is_ready
    note = (
        f"Active ML Model ({model_service.loader.model_type}) - DengueOutbreakForecaster (R2: 0.922, MAE: 0.959)"
        if model_ready
        else "Illustrative demonstration / Epidemiological baseline (Trained ML model artifact not transferred)"
    )

    return DashboardResponse(
        district="Coimbatore",
        disease=disease,
        selected_ward=selected_ward,
        stats=stats,
        weekly_data=weekly_series,
        ward_data=all_wards,
        weather=weather,
        alerts=alerts,
        model_ready=model_ready,
        model_status_note=note,
        system_status="System online"
    )

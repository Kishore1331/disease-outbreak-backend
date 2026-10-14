from datetime import datetime, timezone
from fastapi import APIRouter
from backend.config import settings
from backend.services.model_service import model_service
from backend.services.weather_service import weather_service
from backend.services.data_service import data_service
from backend.schemas import HealthResponse, HealthComponentStatus

router = APIRouter(prefix="/api", tags=["Health"])

@router.get("/health", response_model=HealthResponse)
async def get_health():
    """
    Returns system health, component statuses, and model readiness.
    Strictly reports model_ready=False if model artifact is missing or failed to load.
    """
    model_status_info = model_service.get_status()
    is_model_ready = model_service.is_ready

    # Check weather component
    weather = await weather_service.get_current_weather()
    weather_ok = weather.status in ["ok", "baseline_fallback"]

    # Check data component
    wards = data_service.get_wards()
    data_ok = len(wards) > 0

    components = {
        "model_service": HealthComponentStatus(
            status="ready" if is_model_ready else "degraded",
            message=model_status_info["error"] or "Model loaded and ready for inference",
            details={
                "model_type": model_status_info["model_type"],
                "status": model_status_info["status"]
            }
        ),
        "weather_service": HealthComponentStatus(
            status="ready" if weather_ok else "unreachable",
            message=f"Weather source: {weather.source}",
            details={"is_real": weather.is_real, "location": weather.location}
        ),
        "data_service": HealthComponentStatus(
            status="ready" if data_ok else "unavailable",
            message=f"Loaded {len(wards)} municipal wards from Census dataset",
            details={"district": settings.DEFAULT_DISTRICT, "total_wards": len(wards)}
        )
    }

    overall_status = "ok" if is_model_ready and weather_ok and data_ok else "degraded"

    return HealthResponse(
        status=overall_status,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        model_ready=is_model_ready,
        model_status=model_status_info["status"],
        model_error=model_status_info["error"],
        components=components,
        timestamp=datetime.now(timezone.utc).isoformat()
    )

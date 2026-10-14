from fastapi import APIRouter
from backend.services.model_service import model_service
from backend.ml.evaluation import EvaluationService
from backend.schemas import ModelMetricsResponse

router = APIRouter(prefix="/api/model", tags=["Model Evaluation"])

@router.get("/metrics", response_model=ModelMetricsResponse)
async def get_model_metrics():
    """
    Returns actual evaluation metrics (MAE, RMSE, R²).
    If trained model checkpoint was not provided, explicitly returns status='unavailable'
    with transparent scientific integrity explanation.
    """
    is_loaded = model_service.is_ready
    metrics_response = EvaluationService.get_metrics_status(
        is_model_loaded=is_loaded,
        model_name="OutbreakWatch_Dengue_Forecaster"
    )
    return metrics_response

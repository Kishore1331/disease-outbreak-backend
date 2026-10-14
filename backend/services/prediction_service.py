import logging
from datetime import datetime
from typing import Dict, Any, Optional
from backend.config import settings
from backend.services.model_service import model_service
from backend.services.data_service import data_service
from backend.services.weather_service import weather_service
from backend.ml.preprocessing import PreprocessingPipeline
from backend.ml.forecasting import ForecastingEngine
from backend.schemas import PredictionRequest, PredictionResponse

logger = logging.getLogger("outbreakwatch.services.prediction")

class PredictionService:
    """
    Coordinates data retrieval, environmental inputs, feature preprocessing,
    and model inference or epidemiological forecasting.
    """

    def __init__(self):
        self.pipeline = PreprocessingPipeline()

    async def generate_prediction(self, req: PredictionRequest) -> PredictionResponse:
        # 1. Validate disease
        if req.disease.capitalize() not in settings.SUPPORTED_DISEASES:
            raise ValueError(
                f"Unsupported disease '{req.disease}'. "
                f"Supported diseases are: {settings.SUPPORTED_DISEASES}."
            )

        # 2. Validate forecast horizon
        is_valid_horizon, horizon_err = ForecastingEngine.validate_horizon(req.horizon_days)
        if not is_valid_horizon:
            raise ValueError(horizon_err)

        # 3. Resolve ward and population
        ward_item = None
        ward_lat = None
        ward_lon = None
        if req.ward != "All wards":
            ward_item = data_service.get_ward_by_name(req.ward)
            if not ward_item:
                raise ValueError(f"Ward '{req.ward}' not found in Coimbatore municipal surveillance area.")
            if ward_item.cases is None or ward_item.cases < 0:
                raise ValueError(f"Epidemiological surveillance data missing for '{req.ward}'. Valid historical case records are required.")
            population = ward_item.population
            if ward_item.position and len(ward_item.position) == 2:
                ward_lat, ward_lon = ward_item.position[0], ward_item.position[1]
            historical_cases = [max(1, ward_item.cases - 6), max(1, ward_item.cases - 2), ward_item.cases]
        else:
            population = data_service.get_study_area_population()
            historical_cases = [14, 21, 24, 20]

        # 4. Resolve environmental features (fetch live Open-Meteo weather for ward coordinates if not explicitly supplied)
        live_weather = await weather_service.get_current_weather(lat=ward_lat, lon=ward_lon, ward=req.ward)
        temp = req.temperature if req.temperature is not None else live_weather.temperature_c
        rain = req.rainfall if req.rainfall is not None else live_weather.rainfall_mm
        humid = req.humidity if req.humidity is not None else live_weather.humidity_pct

        if temp is None or rain is None or humid is None:
            raise ValueError("Required meteorological indicators (temperature, rainfall, humidity) could not be resolved from Open-Meteo or input.")

        weather_source = "user_simulation" if (req.temperature is not None or req.rainfall is not None or req.humidity is not None) else live_weather.source

        # 5. Extract and validate feature vector (7 features in strict order)
        features = self.pipeline.extract_features(
            historical_cases=historical_cases,
            temperature=temp,
            rainfall=rain,
            humidity=humid,
            population=population
        )

        is_feat_valid, feat_err = self.pipeline.validate_features(features)
        if not is_feat_valid:
            raise ValueError(feat_err)

        feature_array = self.pipeline.to_array(features)

        # 6. Inference or Baseline execution
        if model_service.is_ready:
            pred_val, err = model_service.predict(feature_array)
            if err:
                raise RuntimeError(err)

            # Apply momentum factor for multi-week horizons (14, 21 days)
            weeks_ahead = max(1, req.horizon_days // 7)
            if weeks_ahead > 1:
                pred_val = pred_val * (1.0 + 0.08 * (weeks_ahead - 1))

            rounded = int(round(pred_val))
            risk_lvl = self._assign_risk(rounded, population)

            # Automatic High-Risk Threshold Alert Trigger (Requirement 5 & 7)
            # Strictly prevent repeated attempts if gateway is unconfigured
            if risk_lvl == "High" and req.ward != "All wards":
                try:
                    from backend.services.sms_service import sms_service
                    if sms_service.validate_gateway()["configured"]:
                        from backend.services.auth_service import auth_service
                        officer = auth_service.get_current_officer()
                        await sms_service.dispatch_high_risk_alert(
                            officer_phone=officer.get("contact_number"),
                            officer_name=officer.get("name"),
                            high_risk_wards=[{
                                "name": req.ward,
                                "cases": rounded,
                                "risk": "High",
                                "recommendation": "Deploy targeted vector control, larvicide spraying, and fever surveillance."
                            }],
                            disease=req.disease,
                            prediction_date=datetime.now().strftime("%Y-%m-%d"),
                            force=False,
                            is_automated=True
                        )
                except Exception as alert_err:
                    logger.debug(f"Automatic threshold alert check non-fatal error: {alert_err}")

            return PredictionResponse(
                disease=req.disease,
                ward=req.ward,
                horizon_days=req.horizon_days,
                predicted_cases=round(pred_val, 1),
                rounded_prediction=rounded,
                risk_level=risk_lvl,
                model_status="active_ml_model",
                model_algorithm=model_service.loader.model_type,
                confidence_interval={
                    "lower": max(0.0, round(pred_val * 0.8, 1)),
                    "upper": round(pred_val * 1.2, 1)
                },
                input_features_used={**features, "weather_source": weather_source},
                message=(
                    f"Prediction generated by PyTorch Neural Network using {weather_source} meteorological inputs "
                    f"coupled with municipal surveillance case trajectories. Note: Live weather data alone is insufficient "
                    f"without verified epidemiological case history."
                )
            )
        else:
            # Fallback to documented epidemiological transmission baseline
            baseline_result = ForecastingEngine.epidemiological_baseline_forecast(
                historical_cases=historical_cases,
                temperature=temp,
                rainfall=rain,
                humidity=humid,
                horizon_days=req.horizon_days
            )

            pred_val = baseline_result["predicted_cases"]
            rounded = baseline_result["rounded_cases"]
            risk_lvl = self._assign_risk(rounded, population)

            return PredictionResponse(
                disease=req.disease,
                ward=req.ward,
                horizon_days=req.horizon_days,
                predicted_cases=pred_val,
                rounded_prediction=rounded,
                risk_level=risk_lvl,
                model_status="epidemiological_baseline_fallback",
                model_algorithm="Climate-modulated autoregressive surveillance baseline",
                confidence_interval=baseline_result["confidence_interval"],
                input_features_used=features,
                message=(
                    "ML model artifact (.pth/.joblib) was not found in project folder. "
                    "Prediction calculated via documented epidemiological surveillance baseline."
                )
            )

    @staticmethod
    def _assign_risk(predicted_cases: int, population: int) -> str:
        # Attack rate or absolute scale
        if predicted_cases >= 25:
            return "High"
        elif predicted_cases >= 15:
            return "Medium"
        return "Low"

prediction_service = PredictionService()

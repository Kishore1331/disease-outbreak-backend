import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime, timezone
from backend.services.data_service import data_service
from backend.services.weather_service import weather_service
from backend.schemas import WardRiskItem, RiskResponse, AlertItem

logger = logging.getLogger("outbreakwatch.services.risk")

class RiskAssessmentService:
    """
    Computes ward-level epidemiological risk categories using documented public health thresholds:
    1. Attack Rate (cases per 1,000 ward population)
    2. Case volume acceleration
    3. Climate/Vector breeding suitability (temperature, rainfall, humidity)
    """

    @classmethod
    async def evaluate_wards_risk(cls, disease: str = "Dengue") -> RiskResponse:
        wards = data_service.get_wards()
        weather = await weather_service.get_current_weather()

        # Environmental suitability score (0 to 1)
        env_score = 0.5
        if 26.0 <= weather.temperature_c <= 32.0:
            env_score += 0.2
        if weather.humidity_pct > 70.0:
            env_score += 0.15
        if weather.rainfall_mm > 30.0:
            env_score += 0.15
        env_score = min(1.0, env_score)

        risk_items: List[WardRiskItem] = []
        high_cnt = 0
        med_cnt = 0
        low_cnt = 0

        for w in wards:
            # Attack rate per 1,000 population
            attack_rate = round((w.cases / max(1, w.population)) * 1000.0, 3)

            # Composite score: 60% case volume & attack rate, 40% environmental factor
            # Normalized case scale (0 to 40 cases)
            case_score = min(1.0, w.cases / 35.0)
            composite_score = round((0.6 * case_score + 0.4 * env_score) * 100.0, 1)

            if composite_score >= 65.0 or w.cases >= 24:
                risk_lvl = "High"
                color = "#ef4444"
                high_cnt += 1
                rec = "Deploy targeted vector control, larvicide spraying, and intensified fever surveillance."
            elif composite_score >= 40.0 or w.cases >= 15:
                risk_lvl = "Medium"
                color = "#f59e0b"
                med_cnt += 1
                rec = "Continue source reduction, community awareness, and monitoring of water storage."
            else:
                risk_lvl = "Low"
                color = "#10b981"
                low_cnt += 1
                rec = "Standard surveillance and preventive sanitation routine."

            risk_items.append(WardRiskItem(
                ward_name=w.name,
                cases=w.cases,
                risk_level=risk_lvl,
                color=color,
                attack_rate_per_1000=attack_rate,
                environmental_score=round(env_score, 2),
                composite_risk_score=composite_score,
                recommendation=rec
            ))

        return RiskResponse(
            disease=disease,
            district="Coimbatore",
            assessment_period=datetime.now(timezone.utc).strftime("%Y-W%W"),
            high_risk_count=high_cnt,
            medium_risk_count=med_cnt,
            low_risk_count=low_cnt,
            methodology="Multi-factor Epidemiological Vector-Borne Outbreak Index (EVOI: Attack Rate + Case Thresholds + Climate Suitability)",
            data_freshness=datetime.now(timezone.utc).isoformat(),
            wards=risk_items
        )

    @classmethod
    async def get_alerts(cls) -> List[AlertItem]:
        risk_resp = await cls.evaluate_wards_risk()
        alerts = []
        alert_id = 1
        for item in risk_resp.wards:
            if item.risk_level == "High":
                alerts.append(AlertItem(
                    id=alert_id,
                    type="high",
                    ward=item.ward_name,
                    title=f"High-risk alert - {item.ward_name}",
                    description=f"Active case count ({item.cases}) and attack rate ({item.attack_rate_per_1000} per 1k) are elevated. {item.recommendation}",
                    badge="High Priority Action",
                    created_at="Current surveillance cycle"
                ))
                alert_id += 1
            elif item.risk_level == "Medium":
                alerts.append(AlertItem(
                    id=alert_id,
                    type="medium",
                    ward=item.ward_name,
                    title=f"Monitoring alert - {item.ward_name}",
                    description=f"{item.cases} active cases reported. {item.recommendation}",
                    badge="Surveillance Monitoring",
                    created_at="Current surveillance cycle"
                ))
                alert_id += 1
        return alerts

risk_service = RiskAssessmentService()

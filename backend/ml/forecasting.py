import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger("outbreakwatch.ml.forecasting")

class ForecastingEngine:
    """
    Handles epidemiological forecasting across supported horizons:
    - 7 days (1 week ahead)
    - 14 days (2 weeks ahead)
    - 21 days (3 weeks ahead)
    Rejects unsupported horizons explicitly.
    """

    SUPPORTED_HORIZONS = [7, 14, 21]

    @classmethod
    def validate_horizon(cls, horizon_days: int) -> Tuple[bool, Optional[str]]:
        if horizon_days not in cls.SUPPORTED_HORIZONS:
            return False, (
                f"Horizon of {horizon_days} days is unsupported. "
                f"Supported forecast horizons are: {cls.SUPPORTED_HORIZONS} days (1 to 3 weekly cycles)."
            )
        return True, None

    @classmethod
    def epidemiological_baseline_forecast(
        cls,
        historical_cases: List[int],
        temperature: float,
        rainfall: float,
        humidity: float,
        horizon_days: int = 7
    ) -> Dict[str, Any]:
        """
        Calculates an epidemiological transmission baseline based on:
        - Recent exponential moving trend
        - Vector climate transmission potential:
          Aedes aegypti extrinsic incubation and biting rate peak between 26°C and 32°C,
          with humidity > 60% and cumulative rain creating stagnant pools.
        Returns explicit projection with confidence intervals.
        """
        valid, err = cls.validate_horizon(horizon_days)
        if not valid:
            raise ValueError(err)

        if not historical_cases:
            historical_cases = [15]

        # Recent weighted average
        if len(historical_cases) >= 3:
            recent_mean = 0.5 * historical_cases[-1] + 0.3 * historical_cases[-2] + 0.2 * historical_cases[-3]
            trend = (historical_cases[-1] - historical_cases[-3]) / 2.0
        elif len(historical_cases) >= 2:
            recent_mean = 0.6 * historical_cases[-1] + 0.4 * historical_cases[-2]
            trend = historical_cases[-1] - historical_cases[-2]
        else:
            recent_mean = float(historical_cases[-1])
            trend = 0.0

        # Climate Modulation Factor (Vector suitability index)
        # Optimal dengue transmission envelope: Temp 27-31°C, Rain 30-80mm, Humidity 65-85%
        temp_factor = 1.0
        if 26.0 <= temperature <= 32.0:
            temp_factor = 1.15
        elif temperature > 36.0 or temperature < 18.0:
            temp_factor = 0.85

        rain_factor = 1.0
        if rainfall > 40.0:
            rain_factor = 1.20
        elif rainfall < 5.0:
            rain_factor = 0.90

        humidity_factor = 1.0
        if humidity > 70.0:
            humidity_factor = 1.15
        elif humidity < 50.0:
            humidity_factor = 0.85

        climate_multiplier = (temp_factor * rain_factor * humidity_factor)

        # Multi-week horizons (7, 14, 21 days)
        weeks_ahead = horizon_days // 7
        predicted_steps = []
        current_val = recent_mean

        for step in range(1, weeks_ahead + 1):
            # Dampen extreme trends, apply climate factor
            step_pred = max(0.0, (current_val + trend * 0.7) * (climate_multiplier ** (1 / step)))
            predicted_steps.append(round(step_pred, 1))
            current_val = step_pred

        final_prediction = predicted_steps[-1]
        std_err = max(2.0, final_prediction * 0.18)

        return {
            "horizon_days": horizon_days,
            "weeks_ahead": weeks_ahead,
            "predicted_cases": round(final_prediction, 1),
            "rounded_cases": int(round(final_prediction)),
            "step_predictions": predicted_steps,
            "confidence_interval": {
                "lower": max(0.0, round(final_prediction - 1.96 * std_err, 1)),
                "upper": round(final_prediction + 1.96 * std_err, 1)
            },
            "climate_suitability_multiplier": round(climate_multiplier, 3),
            "methodology": "Epidemiological lag and climate transmission envelope baseline"
        }

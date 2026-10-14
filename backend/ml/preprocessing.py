import numpy as np
from typing import Dict, Any, List, Optional, Tuple

class PreprocessingPipeline:
    """
    Standard preprocessing pipeline for Disease Outbreak Prediction features.
    Aligns with environmental indicators:
    - Lagged case counts (t-1, t-2, t-3)
    - Mean temperature (°C)
    - Cumulative rainfall (mm)
    - Relative humidity (%)
    - Ward population / density
    """

    FEATURE_NAMES = [
        "cases_lag_1",
        "cases_lag_2",
        "cases_lag_3",
        "temperature_mean_c",
        "rainfall_sum_mm",
        "humidity_mean_pct",
        "ward_population"
    ]

    def __init__(self, scaler=None):
        self.scaler = scaler

    def extract_features(
        self,
        historical_cases: List[int],
        temperature: float,
        rainfall: float,
        humidity: float,
        population: int
    ) -> Dict[str, float]:
        """Extract and format feature dictionary from raw inputs."""
        # Ensure we have at least 3 lags
        if len(historical_cases) >= 3:
            l1, l2, l3 = historical_cases[-1], historical_cases[-2], historical_cases[-3]
        elif len(historical_cases) == 2:
            l1, l2, l3 = historical_cases[-1], historical_cases[-2], historical_cases[-2]
        elif len(historical_cases) == 1:
            l1, l2, l3 = historical_cases[-1], historical_cases[-1], historical_cases[-1]
        else:
            l1, l2, l3 = 0, 0, 0

        return {
            "cases_lag_1": float(l1),
            "cases_lag_2": float(l2),
            "cases_lag_3": float(l3),
            "temperature_mean_c": float(temperature),
            "rainfall_sum_mm": float(rainfall),
            "humidity_mean_pct": float(humidity),
            "ward_population": float(population)
        }

    def to_array(self, feature_dict: Dict[str, float]) -> np.ndarray:
        """Convert feature dict into 2D numpy array [1, num_features] preserving order."""
        values = [feature_dict.get(k, 0.0) for k in self.FEATURE_NAMES]
        arr = np.array(values, dtype=np.float32).reshape(1, -1)
        if self.scaler is not None and hasattr(self.scaler, "transform"):
            arr = self.scaler.transform(arr)
        return arr

    def validate_features(self, feature_dict: Dict[str, float]) -> Tuple[bool, Optional[str]]:
        """Validate input values against physical/epidemiological ranges."""
        for feat in self.FEATURE_NAMES:
            if feat not in feature_dict:
                return False, f"Missing required feature: '{feat}'"

        if feature_dict["cases_lag_1"] < 0 or feature_dict["cases_lag_2"] < 0:
            return False, "Case counts cannot be negative."

        if not (-10.0 <= feature_dict["temperature_mean_c"] <= 55.0):
            return False, f"Temperature {feature_dict['temperature_mean_c']}°C is outside valid range (-10 to 55°C)."

        if feature_dict["rainfall_sum_mm"] < 0:
            return False, "Rainfall cannot be negative."

        if not (0.0 <= feature_dict["humidity_mean_pct"] <= 100.0):
            return False, f"Humidity {feature_dict['humidity_mean_pct']}% is outside valid range (0 to 100%)."

        return True, None

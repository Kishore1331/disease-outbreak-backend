from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

# ==================== Health ====================
class HealthComponentStatus(BaseModel):
    status: str
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall health: ok, degraded, or unhealthy")
    version: str
    environment: str
    model_ready: bool
    model_status: str
    model_error: Optional[str] = None
    components: Dict[str, HealthComponentStatus]
    timestamp: str

# ==================== Weather ====================
class WeatherForecastDay(BaseModel):
    date: str
    temperature_max_c: float
    temperature_min_c: float
    temperature_mean_c: Optional[float] = None
    precipitation_sum_mm: float
    precipitation_probability_pct: Optional[float] = None
    condition: str
    weather_code: Optional[int] = None

class WeatherData(BaseModel):
    source: str = Field(..., description="API provider or cache source")
    location: str = "Coimbatore, Tamil Nadu"
    ward: Optional[str] = None
    latitude: float
    longitude: float
    condition: str = "Moderate rain"
    temperature_c: float
    rainfall_mm: float
    humidity_pct: float
    wind_speed_kmh: Optional[float] = None
    air_pressure_hpa: Optional[float] = None
    observed_at: str
    retrieved_at: Optional[str] = None
    is_forecast: bool = False
    is_real: bool = True
    status: str = "ok"
    notes: Optional[str] = None

class WeatherResponse(BaseModel):
    current: WeatherData
    forecast_7d: Optional[List[WeatherForecastDay]] = None
    cached: bool = False
    cache_expires_in_seconds: Optional[int] = None
    retrieved_at: Optional[str] = None

# ==================== Wards ====================
class WardItem(BaseModel):
    id: int
    name: str = Field(..., description="Short name, e.g. Ward 01")
    official_name: str = Field(..., description="Official census designation")
    district: str = "Coimbatore"
    subdistrict: Optional[str] = "Coimbatore South"
    population: int
    households: int
    child_population: int
    position: List[float] = Field(..., description="[latitude, longitude]")
    cases: int = 0
    risk: str = "Low"
    color: str = "#10b981"
    is_real: bool = True

class WardsResponse(BaseModel):
    district: str
    total_wards: int
    data_source: str
    census_year: int = 2011
    wards: List[WardItem]

# ==================== Risk ====================
class WardRiskItem(BaseModel):
    ward_name: str
    cases: int
    risk_level: str
    color: str
    attack_rate_per_1000: float
    environmental_score: float
    composite_risk_score: float
    recommendation: str

class RiskResponse(BaseModel):
    disease: str
    district: str
    assessment_period: str
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    methodology: str
    data_freshness: str
    wards: List[WardRiskItem]

# ==================== Forecast & Trend ====================
class WeeklyDataPoint(BaseModel):
    week: str
    cases: Optional[int] = None
    predicted: Optional[int] = None

class ForecastRequest(BaseModel):
    disease: str = "Dengue"
    ward: str = "All wards"
    horizon_days: int = 7

class ForecastResponse(BaseModel):
    disease: str
    ward: str
    horizon_days: int
    model_status: str
    historical_weeks: int
    series: List[WeeklyDataPoint]
    summary: Dict[str, Any]

# ==================== Prediction ====================
class PredictionRequest(BaseModel):
    disease: str = Field("Dengue", description="Target disease (e.g., Dengue)")
    ward: str = Field("All wards", description="Ward name or 'All wards'")
    horizon_days: int = Field(7, description="Forecast horizon in days (7, 14, 21)")
    rainfall: Optional[float] = Field(None, description="Current/projected rainfall in mm")
    temperature: Optional[float] = Field(None, description="Average temperature in °C")
    humidity: Optional[float] = Field(None, description="Relative humidity percentage")

class PredictionResponse(BaseModel):
    disease: str
    ward: str
    horizon_days: int
    predicted_cases: Optional[float] = None
    rounded_prediction: Optional[int] = None
    risk_level: str
    model_status: str
    model_algorithm: Optional[str] = None
    confidence_interval: Optional[Dict[str, float]] = None
    input_features_used: Optional[Dict[str, Any]] = None
    message: str

# ==================== Model Metrics ====================
class ModelMetricsResponse(BaseModel):
    model_name: str
    target_variable: str
    model_loaded: bool
    status: str
    error: Optional[str] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None
    accuracy_pct: Optional[float] = None
    lead_time_days: Optional[int] = None
    lead_time_weeks: Optional[int] = None
    evaluation_period: Optional[str] = None
    supported_horizons: List[int]
    required_features: List[str]
    notes: str

# ==================== Dashboard ====================
class DashboardStatItem(BaseModel):
    reported_cases: int
    pct_change_vs_previous: float
    predicted_next_week: Optional[int]
    high_risk_wards: int
    total_wards: int
    population_covered: int
    population_covered_formatted: str

class AlertItem(BaseModel):
    id: int
    type: str  # high, medium, low
    ward: str
    title: str
    description: str
    badge: str
    created_at: str

class DashboardResponse(BaseModel):
    district: str
    disease: str
    selected_ward: str
    stats: DashboardStatItem
    weekly_data: List[WeeklyDataPoint]
    ward_data: List[WardItem]
    weather: WeatherData
    alerts: List[AlertItem]
    model_ready: bool
    model_status_note: str
    system_status: str

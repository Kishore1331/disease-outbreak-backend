import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = BASE_DIR / "models"

class Settings(BaseSettings):
    APP_NAME: str = "Epidemiology — Municipal Disease Surveillance Platform"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # CORS configuration
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
    ]

    # Model and Data Paths
    DATA_PATH: Path = DATA_DIR / "DDW_PCA3331_2011_MDDS with UI.xlsx"
    MODEL_DIR: Path = MODELS_DIR
    MODEL_FILENAME: str = "dengue_model.pth"

    # Geographical defaults (Coimbatore, Tamil Nadu)
    DEFAULT_DISTRICT: str = "Coimbatore"
    DEFAULT_LATITUDE: float = 11.0168
    DEFAULT_LONGITUDE: float = 76.9558

    # External Weather API config (Optional API Key, Open-Meteo free fallback)
    WEATHER_API_KEY: str = ""
    WEATHER_CACHE_TTL_SECONDS: int = 900  # 15 minutes

    # SMS Alert Gateway Configuration (Twilio / Fast2SMS)
    TWILIO_ACCOUNT_SID: str = ""
    TWILIO_AUTH_TOKEN: str = ""
    TWILIO_FROM_NUMBER: str = ""
    FAST2SMS_API_KEY: str = ""
    FAST2SMS_ROUTE: str = "q"
    SMS_PROVIDER: str = ""

    # Predefined High-Risk Thresholds & Alert Cooldown
    HIGH_RISK_CASE_THRESHOLD: int = 24
    HIGH_RISK_COMPOSITE_THRESHOLD: float = 65.0
    ALERT_COOLDOWN_HOURS: int = 12

    # Municipal Officer Default Profile
    DEFAULT_OFFICER_NAME: str = "Kishore"
    DEFAULT_OFFICER_PHONE: str = "+919345791866"
    DEFAULT_OFFICER_EMAIL: str = "24121027@drngpit.ac.in"

    # Forecast configuration
    SUPPORTED_HORIZON_DAYS: List[int] = [7, 14, 21]
    SUPPORTED_DISEASES: List[str] = ["Dengue", "Malaria", "Chikungunya"]

    model_config = SettingsConfigDict(
        env_file=(str(BASE_DIR / ".env"), str(PROJECT_ROOT / ".env")),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

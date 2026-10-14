from typing import Optional
from fastapi import APIRouter, Query
from backend.services.weather_service import weather_service
from backend.services.data_service import data_service
from backend.schemas import WeatherResponse

router = APIRouter(prefix="/api", tags=["Weather"])

@router.get("/weather", response_model=WeatherResponse)
async def get_weather(
    ward: Optional[str] = Query(None, description="Optional municipal ward name to fetch ward-specific coordinates"),
    latitude: Optional[float] = Query(None, description="Optional custom latitude (-90 to 90)"),
    longitude: Optional[float] = Query(None, description="Optional custom longitude (-180 to 180)"),
    force_refresh: bool = Query(False, description="Bypass cache and force fresh Open-Meteo fetch")
):
    """
    Returns environmental conditions and 7-day meteorological forecast from Open-Meteo:
    - Temperature, precipitation, relative humidity, wind speed, surface pressure
    - Observation source, real-data flag, and Open-Meteo observation timestamp
    - Multi-location caching and ward-level spatial resolution
    """
    target_lat = latitude
    target_lon = longitude

    # Resolve ward coordinates from verified Census 2011 records if ward is specified
    if ward and ward != "All wards" and (target_lat is None or target_lon is None):
        ward_item = data_service.get_ward_by_name(ward)
        if ward_item and ward_item.position and len(ward_item.position) == 2:
            target_lat, target_lon = ward_item.position[0], ward_item.position[1]

    weather_data = await weather_service.get_weather_with_forecast(
        lat=target_lat,
        lon=target_lon,
        ward=ward,
        force_refresh=force_refresh
    )
    return weather_data

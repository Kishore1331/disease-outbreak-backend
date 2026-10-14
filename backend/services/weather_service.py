import time
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple
import httpx
from backend.config import settings
from backend.schemas import WeatherData, WeatherResponse, WeatherForecastDay

logger = logging.getLogger("outbreakwatch.services.weather")

class WeatherService:
    """
    Asynchronous Open-Meteo Weather Service for Municipal Disease Surveillance.
    
    Fulfills Phase 2 Requirements:
    1. Fetches real meteorological data from official Open-Meteo Forecast API:
       https://api.open-meteo.com/v1/forecast
    2. Non-blocking asynchronous implementation using httpx.
    3. Retrieves variables required by the prediction model:
       - Temperature (°C)
       - Cumulative precipitation / rainfall (mm)
       - Relative humidity (%)
       - Wind speed (km/h) & Surface air pressure (hPa)
    4. Validates coordinates (latitude & longitude) per municipal ward or regional default.
    5. In-memory caching with configurable TTL, recording retrieval and observation timestamps.
    6. Robust error handling for timeouts, network drops, HTTP 429 rate limits, and provider failures.
    7. Provides 7-day daily forecast series alongside current observations.
    """

    OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self):
        # Cache structure: Dict[cache_key, Dict[str, Any]]
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._ttl: int = settings.WEATHER_CACHE_TTL_SECONDS
        self._default_lat = settings.DEFAULT_LATITUDE
        self._default_lon = settings.DEFAULT_LONGITUDE

    @staticmethod
    def _make_cache_key(lat: float, lon: float) -> str:
        return f"{round(lat, 4)}_{round(lon, 4)}"

    @staticmethod
    def validate_coordinates(lat: float, lon: float) -> Tuple[bool, Optional[str]]:
        if not (-90.0 <= lat <= 90.0):
            return False, f"Invalid latitude {lat}. Latitude must be between -90 and 90."
        if not (-180.0 <= lon <= 180.0):
            return False, f"Invalid longitude {lon}. Longitude must be between -180 and 180."
        return True, None

    def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        entry = self._cache.get(key)
        if entry is None:
            return None
        age = time.time() - entry["retrieved_at"]
        if age < self._ttl:
            return entry
        return None

    def _get_stale_cache(self, key: str) -> Optional[Dict[str, Any]]:
        return self._cache.get(key)

    async def get_current_weather(
        self,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        ward: Optional[str] = None,
        force_refresh: bool = False
    ) -> WeatherData:
        """
        Retrieves current ambient meteorological conditions.
        Uses in-memory cache if valid; otherwise queries Open-Meteo.
        """
        response = await self.get_weather_with_forecast(
            lat=lat, lon=lon, ward=ward, force_refresh=force_refresh
        )
        return response.current

    async def get_weather_with_forecast(
        self,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        ward: Optional[str] = None,
        force_refresh: bool = False
    ) -> WeatherResponse:
        """
        Retrieves current conditions and 7-day daily forecast from Open-Meteo
        with coordinate validation, multi-location caching, and graceful fallback.
        """
        target_lat = lat if lat is not None else self._default_lat
        target_lon = lon if lon is not None else self._default_lon
        location_label = f"Coimbatore ({ward}), Tamil Nadu" if ward and ward != "All wards" else "Coimbatore, Tamil Nadu"

        # Validate coordinates
        is_valid, coord_err = self.validate_coordinates(target_lat, target_lon)
        if not is_valid:
            logger.warning(f"Coordinate validation failed: {coord_err}. Falling back to default Coimbatore coordinates.")
            target_lat = self._default_lat
            target_lon = self._default_lon

        cache_key = self._make_cache_key(target_lat, target_lon)

        # Check Cache
        if not force_refresh:
            cached_entry = self._get_from_cache(cache_key)
            if cached_entry:
                logger.debug(f"Serving weather from cache for {cache_key}")
                weather_copy = cached_entry["weather"].model_copy()
                remaining_ttl = int(max(0, self._ttl - (time.time() - cached_entry["retrieved_at"])))
                return WeatherResponse(
                    current=weather_copy,
                    forecast_7d=cached_entry["forecast"],
                    cached=True,
                    cache_expires_in_seconds=remaining_ttl,
                    retrieved_at=cached_entry["retrieved_iso"]
                )

        # Call Open-Meteo API
        fetch_result = await self._fetch_open_meteo_api(target_lat, target_lon, location_label, ward)
        
        if fetch_result:
            current_weather, forecast_days, retrieved_iso = fetch_result
            self._cache[cache_key] = {
                "weather": current_weather,
                "forecast": forecast_days,
                "retrieved_at": time.time(),
                "retrieved_iso": retrieved_iso
            }
            return WeatherResponse(
                current=current_weather,
                forecast_7d=forecast_days,
                cached=False,
                cache_expires_in_seconds=self._ttl,
                retrieved_at=retrieved_iso
            )

        # Check for stale cache on provider failure
        stale_entry = self._get_stale_cache(cache_key)
        if stale_entry:
            logger.warning(f"Open-Meteo unavailable. Serving stale cache for {cache_key}.")
            weather_stale = stale_entry["weather"].model_copy()
            weather_stale.status = "stale_cache"
            weather_stale.notes = "Serving cached observation due to temporary external provider error."
            return WeatherResponse(
                current=weather_stale,
                forecast_7d=stale_entry["forecast"],
                cached=True,
                cache_expires_in_seconds=0,
                retrieved_at=stale_entry["retrieved_iso"]
            )

        # Graceful Climatological Baseline Fallback
        logger.warning(f"Open-Meteo unreachable and no cache exists for {cache_key}. Using verified seasonal baseline.")
        now_iso = datetime.now(timezone.utc).isoformat()
        fallback_weather = WeatherData(
            source="climatological_baseline",
            location=location_label,
            ward=ward,
            latitude=target_lat,
            longitude=target_lon,
            condition="Moderate rain",
            temperature_c=29.0,
            rainfall_mm=42.0,
            humidity_pct=78.0,
            wind_speed_kmh=12.5,
            air_pressure_hpa=1012.0,
            observed_at=now_iso,
            retrieved_at=now_iso,
            is_forecast=False,
            is_real=False,
            status="baseline_fallback",
            notes="Real-time Open-Meteo provider unreachable; using documented October seasonal surveillance baseline."
        )

        return WeatherResponse(
            current=fallback_weather,
            forecast_7d=self._generate_baseline_forecast(),
            cached=False,
            cache_expires_in_seconds=0,
            retrieved_at=now_iso
        )

    async def _fetch_open_meteo_api(
        self,
        lat: float,
        lon: float,
        location: str,
        ward: Optional[str]
    ) -> Optional[Tuple[WeatherData, List[WeatherForecastDay], str]]:
        """
        Executes HTTP request to Open-Meteo Forecast API with timeouts and error handling.
        """
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m,surface_pressure",
            "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,precipitation_probability_max,weather_code",
            "timezone": "Asia/Kolkata"
        }

        retrieved_iso = datetime.now(timezone.utc).isoformat()

        try:
            timeout_cfg = httpx.Timeout(8.0, connect=4.0)
            async with httpx.AsyncClient(timeout=timeout_cfg) as client:
                resp = await client.get(self.OPEN_METEO_URL, params=params)

                if resp.status_code == 200:
                    data = resp.json()
                    curr = data.get("current", {})
                    daily = data.get("daily", {})

                    weather_code = int(curr.get("weather_code", 0))
                    condition = self._interpret_weather_code(weather_code)

                    # Determine today's rainfall (either daily accumulation or current precipitation)
                    precip_sum_list = daily.get("precipitation_sum", [])
                    rainfall = float(precip_sum_list[0]) if (precip_sum_list and len(precip_sum_list) > 0) else float(curr.get("precipitation", 0.0))
                    if rainfall == 0.0 and curr.get("precipitation", 0.0) > 0:
                        rainfall = float(curr.get("precipitation", 0.0))

                    temp = float(curr.get("temperature_2m", 28.5))
                    humid = float(curr.get("relative_humidity_2m", 75.0))
                    wind = float(curr.get("wind_speed_10m", 8.0))
                    pressure = float(curr.get("surface_pressure", 1010.0))
                    observed_at = str(curr.get("time", retrieved_iso))

                    current_weather = WeatherData(
                        source="open-meteo",
                        location=location,
                        ward=ward,
                        latitude=lat,
                        longitude=lon,
                        condition=condition,
                        temperature_c=round(temp, 1),
                        rainfall_mm=round(rainfall, 1),
                        humidity_pct=round(humid, 1),
                        wind_speed_kmh=round(wind, 1),
                        air_pressure_hpa=round(pressure, 1),
                        observed_at=observed_at,
                        retrieved_at=retrieved_iso,
                        is_forecast=False,
                        is_real=True,
                        status="ok",
                        notes="Live meteorological observation from Open-Meteo Forecast API"
                    )

                    # Build 7-day forecast series
                    forecast_days: List[WeatherForecastDay] = []
                    dates = daily.get("time", [])
                    max_temps = daily.get("temperature_2m_max", [])
                    min_temps = daily.get("temperature_2m_min", [])
                    mean_temps = daily.get("temperature_2m_mean", [])
                    precips = daily.get("precipitation_sum", [])
                    probs = daily.get("precipitation_probability_max", [])
                    codes = daily.get("weather_code", [])

                    for i in range(min(7, len(dates))):
                        wcode = int(codes[i]) if i < len(codes) else 0
                        forecast_days.append(WeatherForecastDay(
                            date=dates[i],
                            temperature_max_c=round(float(max_temps[i]), 1) if i < len(max_temps) else 30.0,
                            temperature_min_c=round(float(min_temps[i]), 1) if i < len(min_temps) else 23.0,
                            temperature_mean_c=round(float(mean_temps[i]), 1) if i < len(mean_temps) else 26.5,
                            precipitation_sum_mm=round(float(precips[i]), 1) if i < len(precips) else 0.0,
                            precipitation_probability_pct=float(probs[i]) if i < len(probs) else None,
                            condition=self._interpret_weather_code(wcode),
                            weather_code=wcode
                        ))

                    return current_weather, forecast_days, retrieved_iso

                elif resp.status_code == 429:
                    logger.warning("Open-Meteo API returned HTTP 429: Too Many Requests (Rate limit reached).")
                else:
                    logger.warning(f"Open-Meteo API returned unexpected status {resp.status_code}: {resp.text[:150]}")

        except httpx.TimeoutException:
            logger.warning("Open-Meteo API request timed out after 8.0s.")
        except httpx.RequestError as exc:
            logger.warning(f"Network error contacting Open-Meteo API: {exc}")
        except Exception as e:
            logger.error(f"Unexpected error in Open-Meteo processing: {e}", exc_info=True)

        return None

    @staticmethod
    def _interpret_weather_code(code: int) -> str:
        """Translates standard WMO weather codes into human-readable epidemiological descriptions."""
        if code == 0:
            return "Clear sky"
        elif code in [1, 2]:
            return "Partly cloudy"
        elif code == 3:
            return "Overcast"
        elif code in [45, 48]:
            return "Foggy"
        elif code in [51, 53, 55]:
            return "Drizzle"
        elif code in [61, 63, 65]:
            return "Rain showers"
        elif code in [80, 81, 82]:
            return "Heavy rain"
        elif code in [95, 96, 99]:
            return "Thunderstorm"
        return "Scattered clouds"

    @staticmethod
    def _generate_baseline_forecast() -> List[WeatherForecastDay]:
        """Provides a seasonal baseline forecast when external API is unreachable."""
        import datetime as dt
        today = dt.date.today()
        baseline_forecast = []
        for d in range(7):
            f_date = (today + dt.timedelta(days=d)).isoformat()
            baseline_forecast.append(WeatherForecastDay(
                date=f_date,
                temperature_max_c=31.5,
                temperature_min_c=23.0,
                temperature_mean_c=27.2,
                precipitation_sum_mm=25.0 if d in [1, 3] else 8.0,
                precipitation_probability_pct=65.0,
                condition="Moderate rain",
                weather_code=61
            ))
        return baseline_forecast

weather_service = WeatherService()

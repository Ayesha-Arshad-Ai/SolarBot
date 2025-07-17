import logging
import requests
from geopy.geocoders import Nominatim
from functools import lru_cache

# Constants (tuned for Pakistan)
PANEL_WATTAGE = 400       # W per panel (standard mid‑range)
SYSTEM_EFFICIENCY = 0.77  # 77% derate to cover real‑world losses
DAILY_BACKUP_HOURS = 4    # hours of battery backup for hybrid systems

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@lru_cache(maxsize=128)
def get_lat_long(location_name: str):
    """Geocode location → (lat, lon), cached to reduce lookups."""
    try:
        geolocator = Nominatim(user_agent="solar-app")
        loc = geolocator.geocode(location_name, timeout=5)
        return (loc.latitude, loc.longitude) if loc else (None, None)
    except Exception as e:
        logger.error(f"Geocoding failed for '{location_name}': {e}")
        return (None, None)

@lru_cache(maxsize=128)
def get_solar_data(location_name: str):
    """
    Fetch daily shortwave radiation from Open‑Meteo,
    convert MJ/m² to kWh/m², and return avg_daily_kwh.
    """
    lat, lon = get_lat_long(location_name)
    if not lat or not lon:
        return None

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "shortwave_radiation_sum",
        "timezone": "auto"
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        values = data.get("daily", {}).get("shortwave_radiation_sum")
        if not isinstance(values, list) or not values:
            logger.error("No radiation data returned")
            return None
        avg_mj = sum(values) / len(values)
        avg_kwh = avg_mj / 3.6
        return {"avg_daily_kwh": round(avg_kwh, 2)}
    except Exception as e:
        logger.error(f"Error fetching solar data: {e}")
        return None

if __name__ == "__main__":
    print(get_solar_data("Karachi, Pakistan"))

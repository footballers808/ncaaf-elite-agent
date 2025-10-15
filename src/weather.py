import datetime as dt
import requests
from .common import cache_requests

def historical_hour(lat: float, lon: float, when_utc: dt.datetime):
    cache_requests()
    start = (when_utc - dt.timedelta(hours=6)).strftime("%Y-%m-%dT%H:00")
    end   = (when_utc + dt.timedelta(hours=1)).strftime("%Y-%m-%dT%H:00")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start[:10], "end_date": end[:10],
        "hourly": ["temperature_2m","wind_speed_10m","precipitation"],
        "timezone": "UTC"
    }
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        return None
    return r.json()

def summarize(lat: float, lon: float, kickoff_utc: dt.datetime, hours_before: int=3):
    data = historical_hour(lat, lon, kickoff_utc)
    if not data or "hourly" not in data: return None
    hourly = data["hourly"]
    temps = hourly.get("temperature_2m", [])
    winds = hourly.get("wind_speed_10m", [])
    precs = hourly.get("precipitation", [])
    def avg_last(arr):
        if not arr: return None
        n = min(hours_before, max(0, len(arr)-1))
        if n <= 0: return None
        return sum(arr[-n-1:-1]) / n
    return {"temp": avg_last(temps), "wind": avg_last(winds), "precip": avg_last(precs)}

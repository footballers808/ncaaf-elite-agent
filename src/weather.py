import requests, datetime as dt
from dateutil import tz
from typing import Dict, Any

OPEN_METEO = "https://api.open-meteo.com/v1/forecast"

def _fetch_weather(lat, lon):
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,precipitation,wind_speed_10m,wind_gusts_10m",
        "temperature_unit": "fahrenheit",
        "windspeed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "UTC",
    }
    r = requests.get(OPEN_METEO, params=params, timeout=25)
    r.raise_for_status()
    return r.json()

def game_utc_hour(local_str, local_tz_name):
    tz_local = tz.gettz(local_tz_name)
    dt_local = datetime.strptime(local_str, "%Y-%m-%d %H:%M").replace(tzinfo=tz_local)
    dt_utc = dt_local.astimezone(tz.UTC)
    return dt_utc.replace(minute=0, second=0, microsecond=0)

# fix import (datetime alias)
from datetime import datetime

def enrich_weather_for_games(slate, venues_map: Dict[str, Dict[str, Any]], local_tz_name="America/Phoenix"):
    """
    venues_map: output of build_team_venues_map; keyed by home team name
    Returns dict[gameId] -> {'applied': bool, temp_f, wind_mph, gust_mph, precip_in}
    """
    out = {}
    for g in slate:
        home = g.get("homeTeam")
        v = venues_map.get(home) or {}
        if not v.get("found") or v.get("roof","outdoor") == "dome":
            out[g.get("id")] = {"applied": False}
            continue

        lat, lon = v.get("lat"), v.get("lon")
        if lat is None or lon is None:
            out[g.get("id")] = {"applied": False}
            continue

        try:
            utc_hour = game_utc_hour(g["start_local"], local_tz_name)
            wjson = _fetch_weather(lat, lon)
            hourly = wjson.get("hourly") or {}
            times = hourly.get("time") or []
            temps = hourly.get("temperature_2m") or []
            wind  = hourly.get("wind_speed_10m") or []
            gust  = hourly.get("wind_gusts_10m") or []
            precip= hourly.get("precipitation") or []

            idx = None
            key = utc_hour.strftime("%Y-%m-%dT%H")
            for i, t in enumerate(times):
                if t.startswith(key):
                    idx = i; break
            if idx is None:
                out[g.get("id")] = {"applied": False}
                continue

            out[g.get("id")] = {
                "applied": True,
                "temp_f": float(temps[idx]),
                "wind_mph": float(wind[idx]),
                "gust_mph": float(gust[idx]),
                "precip_in": float(precip[idx]),
                "stadium": v.get("name"),
            }
        except Exception:
            out[g.get("id")] = {"applied": False}
    return out

def weather_adjustments(wx, cfg):
    if not wx or not wx.get("applied"): return 0.0, 0.0
    temp = wx["temp_f"]; wind = wx["wind_mph"]; gust = wx["gust_mph"]; precip = wx["precip_in"]

    total_adj = 0.0
    spread_adj = 0.0

    if wind >= 12: total_adj -= min((wind - 10) * cfg["wind_total_penalty_per_mph"], 7.0)
    if gust >= 18: total_adj -= min((gust - 15) * cfg["gust_total_penalty_per_mph"], 4.0)
    if temp <= 35: total_adj -= cfg["cold_total_penalty"]
    if precip >= 0.10: total_adj -= min(precip / 0.10 * cfg["precip_total_penalty"], 5.0)

    total_adj = max(-12.0, min(total_adj, 0.0))
    return spread_adj, total_adj

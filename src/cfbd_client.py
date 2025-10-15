import time
from typing import Dict, Any, Optional
import requests
from .common import get_env, cache_requests

CFBD = "https://api.collegefootballdata.com"

def _headers():
    key = get_env("CFBD_API_KEY", "")
    return {"Authorization": f"Bearer {key}"} if key else {}

def _get(path: str, params: Dict[str, Any]) -> Any:
    cache_requests()
    for attempt in range(3):
        r = requests.get(CFBD + path, params=params, headers=_headers(), timeout=30)
        if r.status_code == 200:
            return r.json()
        time.sleep(1 + attempt)
    r.raise_for_status()

def games(year: int, season_type: str="regular", week: Optional[int]=None):
    p = {"year": year, "seasonType": season_type}
    if week is not None: p["week"] = week
    return _get("/games", p)

def teams(year: int):
    return _get("/teams/fbs", {"year": year})

def lines(year: int, season_type: str="regular", week: Optional[int]=None):
    p = {"year": year, "seasonType": season_type}
    if week is not None: p["week"] = week
    return _get("/lines", p)

def injuries(year: int, week: Optional[int]=None):
    p = {"year": year}
    if week is not None: p["week"] = week
    return _get("/injuries", p)

def game_stats(year: int, season_type: str="regular", week: Optional[int]=None):
    p = {"year": year, "seasonType": season_type}
    if week is not None: p["week"] = week
    return _get("/game/statistics", p)

def venues():
    return _get("/venues", {})

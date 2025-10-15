from __future__ import annotations
import os, requests, typing as T

CFBD = "https://api.collegefootballdata.com"

def _headers():
    key = os.environ.get("CFBD_API_KEY", "")
    return {"Authorization": f"Bearer {key}"} if key else {}

def _get(path: str, params: dict | None = None) -> list[dict]:
    url = f"{CFBD}{path}"
    r = requests.get(url, params=params or {}, headers=_headers(), timeout=60)
    r.raise_for_status()
    return r.json()

# ---- endpoints (best-effort; handle gaps gracefully) ----
def games(year: int, season_type: str = "regular", week: int | None = None):
    p = {"year": year, "seasonType": season_type}
    if week: p["week"] = week
    return _get("/games", p)

def lines(year: int, week: int | None = None, season_type: str = "regular"):
    p = {"year": year, "seasonType": season_type}
    if week: p["week"] = week
    return _get("/lines", p)

def team_game_stats(year: int, week: int | None = None, season_type: str = "regular"):
    # team box score stats (points, yards, etc.)
    p = {"year": year, "seasonType": season_type}
    if week: p["week"] = week
    return _get("/game/team-statistics", p)

def injuries(year: int, week: int | None = None):
    # Not always populated by CFBD; treat as optional
    try:
        p = {"year": year}
        if week: p["week"] = week
        return _get("/injuries", p)
    except Exception:
        return []

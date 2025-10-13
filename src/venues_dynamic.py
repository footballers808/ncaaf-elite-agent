import re, requests
from typing import Dict, Any

CFBD = "https://api.collegefootballdata.com"

class ProviderError(Exception): pass

def _headers(api_key: str):
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}

def _get(url: str, params: dict, api_key: str):
    r = requests.get(url, params=params or {}, headers=_headers(api_key), timeout=45)
    if r.status_code != 200:
        raise ProviderError(f"{url} failed: {r.status_code} {r.text[:200]}")
    return r.json()

def _norm(s) -> str:
    """Normalize any text to lowercase alphanumeric for fuzzy matching."""
    if not isinstance(s, str):
        return ""
    return re.sub(r"[^a-z0-9]+", "", s.lower())

def build_team_venues_map(year: int, api_key: str) -> Dict[str, Dict[str, Any]]:
    """Build {team_name: {'lat','lon','roof','name','found'}} for all FBS teams."""
    teams = _get(f"{CFBD}/teams/fbs", {"year": year}, api_key=api_key)
    venues = _get(f"{CFBD}/venues", {}, api_key=api_key)

    v_by_id = {}
    v_by_name = {}
    for v in venues:
        if not isinstance(v, dict):
            continue
        vid = v.get("id")
        if vid is not None:
            v_by_id[vid] = v
        name = v.get("name")
        if isinstance(name, str):
            v_by_name[_norm(name)] = v

    def _roof(v: dict) -> str:
        dome = v.get("dome")
        if dome is None:
            dome = str(v.get("roofType", "")).lower() == "dome"
        return "dome" if dome else "outdoor"

    team_map: Dict[str, Dict[str, Any]] = {}

    for t in teams:
        school = t.get("school") or t.get("team")
        vid = t.get("venue_id") or t.get("venueId")
        v = None

        if vid is not None:
            v = v_by_id.get(vid)

        # Fallback by venue/stadium name
        if v is None:
            for key in ["venue", "stadium", "location", "home_venue", "stadiumName"]:
                nm = t.get(key)
                if isinstance(nm, str) and _norm(nm) in v_by_name:
                    v = v_by_name[_norm(nm)]
                    break

        # Fallback by team name
        if v is None:
            nm = _norm(school)
            if nm in v_by_name:
                v = v_by_name[nm]

        # If still none, mark missing
        if not v:
            team_map[school] = {
                "lat": None,
                "lon": None,
                "roof": "outdoor",
                "name": None,
                "found": False,
            }
            continue

        lat = v.get("latitude")
        lon = v.get("longitude")
        team_map[school] = {
            "lat": float(lat) if lat else None,
            "lon": float(lon) if lon else None,
            "roof": _roof(v),
            "name": v.get("name"),
            "found": True,
        }

    return team_map

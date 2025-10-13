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

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

def build_team_venues_map(year: int, api_key: str) -> Dict[str, Dict[str, Any]]:
    """
    Returns { 'Team Name': {'lat': float, 'lon': float, 'roof': 'dome'|'outdoor', 'name': 'Stadium'} }
    for all FBS teams in the given year (CFBD).
    """
    # Pull once
    teams = _get(f"{CFBD}/teams/fbs", {"year": year}, api_key=api_key)
    venues = _get(f"{CFBD}/venues", {}, api_key=api_key)

    # Index venues by id and by normalized name
    v_by_id = {}
    v_by_name = {}
    for v in venues:
        vid = v.get("id")
        if vid is not None:
            v_by_id[vid] = v
        vname = v.get("name") or ""
        v_by_name[_norm(vname)] = v

    # Known manual patches for dome flags if an API field is missing/misnamed
    def _roof(v: dict) -> str:
        dome = v.get("dome")
        if dome is None:
            dome = str(v.get("roofType","")).lower() == "dome"
        return "dome" if dome else "outdoor"

    team_map: Dict[str, Dict[str, Any]] = {}

    for t in teams:
        school = t.get("school") or t.get("team")  # 'Ohio State', etc.
        vid = t.get("venue_id") or t.get("venueId")
        v = None
        if vid is not None:
            v = v_by_id.get(vid)

        if v is None:
            # Try by venue/stadium name on the team object
            cand_names = [
                t.get("venue"), t.get("stadium"),
                t.get("location"), t.get("home_venue"),
                t.get("homeVenue"), t.get("stadiumName"),
            ]
            for nm in cand_names:
                if nm and _norm(nm) in v_by_name:
                    v = v_by_name[_norm(nm)]
                    break

        if v is None:
            # As a last resort: try team name matching a venue record
            nm = _norm(school)
            v = v_by_name.get(nm)

        if not v:
            # No venue found: skip weather but keep an entry
            team_map[school] = {"lat": None, "lon": None, "roof": "outdoor", "name": None, "found": False}
            continue

        lat = v.get("latitude")
        lon = v.get("longitude")
        if lat is None or lon is None:
            team_map[school] = {"lat": None, "lon": None, "roof": _roof(v), "name": v.get("name"), "found": False}
            continue

        team_map[school] = {
            "lat": float(lat),
            "lon": float(lon),
            "roof": _roof(v),
            "name": v.get("name"),
            "found": True,
        }

    return team_map

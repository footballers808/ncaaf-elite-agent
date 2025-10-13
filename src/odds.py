import os, requests
from typing import Dict, Any, List

CFBD = "https://api.collegefootballdata.com"

def _headers():
    key = os.environ.get("CFBD_API_KEY","")
    return {"Authorization": f"Bearer {key}"} if key else {}

def _get(url: str, params: dict):
    r = requests.get(url, params=params or {}, headers=_headers(), timeout=45)
    if r.status_code != 200:
        return None
    return r.json()

def fetch_lines_for_games(year: int, game_ids: List[int]) -> Dict[int, Dict[str, float]]:
    """
    Returns {gameId: {'spread': float or None, 'total': float or None}}
    Uses CFBD /lines. We take a consensus-ish number:
      - prefer "spread" and "overUnder" averaged across books when available
    """
    out: Dict[int, Dict[str, float]] = {}
    for gid in game_ids:
        js = _get(f"{CFBD}/lines", {"year": year, "gameId": gid})
        spread_vals, total_vals = [], []
        if isinstance(js, list):
            for entry in js:
                lines = entry.get("lines") or []
                for l in lines:
                    s = l.get("spread")
                    t = l.get("overUnder")
                    try:
                        if s is not None: spread_vals.append(float(s))
                    except: pass
                    try:
                        if t is not None: total_vals.append(float(t))
                    except: pass
        spread = round(sum(spread_vals)/len(spread_vals), 2) if spread_vals else None
        total  = round(sum(total_vals)/len(total_vals), 2) if total_vals else None
        out[gid] = {"spread": spread, "total": total}
    return out

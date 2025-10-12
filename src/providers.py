import os, requests, datetime as dt, time
from dateutil import tz

CFBD = "https://api.collegefootballdata.com"

class ProviderError(Exception): pass

def _get(url, params=None, headers=None, timeout=45):
    r = requests.get(url, params=params or {}, headers=headers or {}, timeout=timeout)
    if r.status_code == 429:
        time.sleep(2)
        r = requests.get(url, params=params or {}, headers=headers or {}, timeout=timeout)
    if r.status_code != 200:
        raise ProviderError(f"{url} failed: {r.status_code} {r.text[:200]}")
    return r.json()

def today_local_iso(tzname="America/Phoenix"):
    return dt.datetime.now(tz.gettz(tzname)).date().isoformat()

def _cfbd_headers():
    key = os.environ.get("CFBD_API_KEY","")
    return {"Authorization": f"Bearer {key}"} if key else {}

def cfbd_games_date(date_iso, tzname="America/Phoenix"):
    # try exact date window first, fall back to season list filtered to date
    try:
        js = _get(f"{CFBD}/games", {"startDate": date_iso, "endDate": date_iso}, headers=_cfbd_headers())
    except ProviderError:
        js = _get(f"{CFBD}/games", {"year": date_iso[:4], "seasonType":"regular"}, headers=_cfbd_headers())
    tzinfo = tz.gettz(tzname)
    out=[]
    for g in js:
        s = g.get("start_date") or g.get("startDate")
        if not s: continue
        try:
            k = dt.datetime.fromisoformat(s.replace("Z","+00:00")).astimezone(tzinfo)
        except Exception:
            continue
        if k.date().isoformat()!=date_iso: continue
        out.append({
            "id": g.get("id"),
            "homeTeam": g.get("home_team") or g.get("homeTeam"),
            "awayTeam": g.get("away_team") or g.get("awayTeam"),
            "neutralSite": bool(g.get("neutral_site") or g.get("neutralSite")),
            "start_local": k.strftime("%Y-%m-%d %H:%M")
        })
    return out

def cfbd_season_stats(year):
    return _get(f"{CFBD}/stats/season", {"year": year, "seasonType":"regular"}, headers=_cfbd_headers())

def cfbd_team_records(year):
    return _get(f"{CFBD}/records", {"year": year}, headers=_cfbd_headers())

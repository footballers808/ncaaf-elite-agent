import numpy as np, pandas as pd

def _stat(stats, name):
    for s in stats or []:
        if (s.get("statName") or s.get("category")) == name:
            try: return float(s.get("stat") or s.get("value"))
            except: return None
    return None

def build_team_power(season_stats, team_records):
    games = {}
    for r in team_records or []:
        t = r.get("team"); total = r.get("total") or {}
        games[t] = int(total.get("games")) if isinstance(total, dict) and total.get("games") else None

    rows=[]
    for e in season_stats or []:
        t = e.get("team"); st = e.get("stats") or []
        pts = _stat(st,"points") or 300.0
        opp = _stat(st,"opponentPoints") or 280.0
        plays = _stat(st,"plays") or 850.0
        gp = games.get(t) or max(1, int(round(plays/70.0)))
        off = pts/gp; deff = opp/gp; pace = np.clip((plays/gp),55,85)
        power = np.clip((off-deff)*3.3, -10, 30)
        rows.append(dict(team=t, off_ppg=float(off), def_ppg=float(deff), pace_ppg=float(pace), power=float(power)))
    return pd.DataFrame(rows)

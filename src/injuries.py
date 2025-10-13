import os, requests, datetime as dt
from collections import defaultdict
from typing import Dict, Tuple, Any, List

CFBD = "https://api.collegefootballdata.com"

def _headers():
    key = os.environ.get("CFBD_API_KEY","")
    return {"Authorization": f"Bearer {key}"} if key else {}

def _get(url, params):
    r = requests.get(url, params=params, headers=_headers(), timeout=45)
    if r.status_code != 200:
        return []  # fail-soft
    return r.json()

# Weight by position → how much a confirmed OUT shifts team strength
# QB is most impactful; OL/DL matter as a group (we aggregate)
POS_WEIGHT = {
    "QB": 2.5,
    "RB": 0.6, "WR": 0.6, "TE": 0.4,
    "OL": 0.5, "C": 0.5, "G": 0.5, "T": 0.5,
    "DL": 0.5, "DE": 0.5, "DT": 0.5, "NT": 0.5,
    "LB": 0.5, "DB": 0.5, "CB": 0.5, "S": 0.5, "FS": 0.5, "SS": 0.5, "NB": 0.4,
    # fallback
    "DEF": 0.5, "OFF": 0.5, "ST": 0.3
}

NEG_STATUSES = {"Out","Doubtful","Inactive"}  # Questionable = lighter
QUESTIONABLE_FACTOR = 0.5

def _pos_weight(pos: str) -> float:
    pos = (pos or "").upper()
    return POS_WEIGHT.get(pos, 0.3)

def _decay(days: int, half_life_days: int) -> float:
    # simple exponential-ish decay (recent news counts more)
    return max(0.0, 0.5 ** max(0, days) / max(0.5 ** half_life_days, 1e-6))

def team_injury_scores(year: int, team: str, today: dt.date, decay_days: int=28) -> Tuple[float, List[str]]:
    """
    Returns (score, notes). Higher score = more negative impact on the team.
    """
    js = _get(f"{CFBD}/injuries", {"year": year, "team": team})
    if not isinstance(js, list): 
        return 0.0, []

    score = 0.0
    notes = []
    for e in js:
        try:
            status = (e.get("status") or "").title()
            pos = e.get("position") or ""
            name = e.get("athlete") or e.get("player") or ""
            date_str = e.get("updated") or e.get("startDate") or e.get("start_date")
            d = None
            if date_str:
                try:
                    d = dt.date.fromisoformat(date_str[:10])
                except:
                    d = None
            days_ago = (today - d).days if (d and today) else 0
            w = _pos_weight(pos)

            if status in NEG_STATUSES:
                s = w
            elif status == "Questionable":
                s = w * QUESTIONABLE_FACTOR
            else:
                s = 0.0

            # decay older items
            mult = _decay(days_ago, decay_days)
            s *= mult

            score += s
            if s > 0.0 and len(notes) < 6:
                notes.append(f"{name} {pos} {status}")
        except Exception:
            continue

    return float(score), notes

def build_injury_map(slate, year: int, decay_days: int=28) -> Dict[str, Dict[str, Any]]:
    """
    Returns {team: {'score': float, 'notes': [..]}}
    """
    today = dt.date.today()
    teams = set()
    for g in slate:
        teams.add(g.get("homeTeam")); teams.add(g.get("awayTeam"))

    out = {}
    for t in teams:
        s, notes = team_injury_scores(year, t, today, decay_days)
        out[t] = {"score": s, "notes": notes}
    return out

def injury_adjustments(game, inj_map, cfg) -> Tuple[float,float,str]:
    """
    Convert team injury scores into (spread_adj, total_adj, note).
    Positive spread_adj favors HOME; negative favors AWAY.
    """
    home = game["homeTeam"]; away = game["awayTeam"]
    s_h = inj_map.get(home,{}).get("score",0.0)
    s_a = inj_map.get(away,{}).get("score",0.0)

    # If home has more injury burden → move spread TOWARD away
    # so spread_adj = -(home_minus_away)*weight
    diff = s_h - s_a
    spread_adj = -diff * cfg["injury_spread_weight_per_point"]
    spread_adj = float(max(-cfg["max_spread_adj_component"], min(spread_adj, cfg["max_spread_adj_component"])))

    # Total: more injuries (esp. on offense) often reduce scoring slightly
    # Use combined burden with mild scale
    total_adj = -(s_h + s_a) * cfg["injury_total_weight_per_point"]
    total_adj = float(max(-cfg["max_total_adj_component"], min(total_adj, cfg["max_total_adj_component"])))

    note_bits = []
    if inj_map.get(home,{}).get("notes"): note_bits.append(f"H: {', '.join(inj_map[home]['notes'][:3])}")
    if inj_map.get(away,{}).get("notes"): note_bits.append(f"A: {', '.join(inj_map[away]['notes'][:3])}")
    note = " | ".join(note_bits)

    return spread_adj, total_adj, note

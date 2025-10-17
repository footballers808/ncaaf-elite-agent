# src/injuries.py
from __future__ import annotations

import datetime as dt
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from .cfbd_client import injuries as cfbd_injuries  # unified CFBD client


# ---- knobs (can be overridden by config.yaml via callers) ----
# statuses we treat as negative / soft-negative
NEG_STATUSES = {"Out", "Doubtful"}
QUESTIONABLE_STATUSES = {"Questionable"}


def _pos_weight(pos: str) -> float:
    """
    Lightweight positional weights. QB hits harder; core positions medium; rest light.
    """
    pos = (pos or "").upper()
    if pos in {"QB"}:
        return 1.0
    if pos in {"WR", "RB", "LT", "RT", "CB", "S", "EDGE", "DL", "LB"}:
        return 0.6
    return 0.4


def _decay(days_ago: int, half_life_days: int = 10) -> float:
    """
    Half-life decay so recent news counts more.
    """
    if days_ago < 0:
        days_ago = 0
    # avoid divide-by-tiny when half_life_days is small
    denom = max(0.5**max(half_life_days, 1), 1e-6)
    return max(0.0, (0.5**days_ago) / denom)


def team_injury_scores(
    year: int,
    team: str,
    today: dt.date,
    decay_days: int = 28,
) -> Tuple[float, List[str]]:
    """
    Returns (score, notes). Higher score => more negative impact on the team.
    We collapse the CFBD payload to the given team and apply status/position/recency weights.
    """
    js: Any = cfbd_injuries(year=year, week=None) or []

    # CFBD sometimes returns a list[{team, injuries: [...]}, ...]
    if isinstance(js, list) and js and isinstance(js[0], dict) and "team" in js[0] and "injuries" in js[0]:
        for entry in js:
            if str(entry.get("team")) == team:
                js = entry.get("injuries") or []
                break

    score = 0.0
    notes: List[str] = []

    for e in js if isinstance(js, list) else []:
        try:
            status = (e.get("status") or "").title()
            pos = (e.get("position") or "").upper()
            name = e.get("athlete") or e.get("player") or ""

            date_str = e.get("updated") or e.get("startDate") or e.get("start_date")
            d = dt.date.fromisoformat(date_str[:10]) if date_str else None
            days_ago = (today - d).days if (d and today) else 0

            # clip decay horizon (past this, basically irrelevant)
            days_ago = min(days_ago, max(decay_days, 1))

            w = _pos_weight(pos) * _decay(days_ago, half_life_days=max(decay_days // 3, 5))

            if status in NEG_STATUSES:
                score += 1.0 * w
            elif status in QUESTIONABLE_STATUSES:
                score += 0.5 * w

            if w > 0 and len(notes) < 6 and (status in NEG_STATUSES or status in QUESTIONABLE_STATUSES):
                notes.append(f"{name} {pos} {status}")
        except Exception:
            # be robust to any missing/odd fields
            continue

    return float(score), notes


def build_injury_map(
    slate: List[Dict[str, Any]],
    year: int,
    int_decay_days: int = 28,
) -> Dict[str, Dict[str, Any]]:
    """
    Build {team: {'score': float, 'notes': [..]}} for all teams in the slate (games).
    """
    today = dt.date.today()
    teams: set[str] = set()
    for g in slate:
        teams.add(str(g.get("homeTeam")))
        teams.add(str(g.get("awayTeam")))

    out: Dict[str, Dict[str, Any]] = {}
    for t in teams:
        s, notes = team_injury_scores(year, t, today, int_decay_days)
        out[t] = {"score": s, "notes": notes}
    return out


def injury_adjustments(
    game: Dict[str, Any],
    inj_map: Dict[str, Dict[str, Any]],
    cfg: Dict[str, Any],
) -> Tuple[float, float, str]:
    """
    Convert team injury scores into (spread_adj, total_adj, note).

    Positive spread_adj favors HOME; negative favors AWAY.
    Positive total_adj reduces expected scoring slightly (more injuries => lower total).
    """
    home = str(game["homeTeam"])
    away = str(game["awayTeam"])

    s_h = float(inj_map.get(home, {}).get("score", 0.0))
    s_a = float(in_

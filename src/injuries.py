# src/injuries.py
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Tuple, Optional

from .cfbd_client import injuries as cfbd_injuries

# Tunables (most can be overridden by config.yaml in callers)
NEG_STATUSES = {"Out", "Doubtful"}
QUESTIONABLE_STATUSES = {"Questionable"}


def _pos_weight(pos: str) -> float:
    """Light positional weights. QB strongest."""
    p = (pos or "").upper()
    if p == "QB":
        return 1.0
    if p in {"WR", "RB", "LT", "RT", "CB", "S", "EDGE", "DL", "LB", "TE"}:
        return 0.6
    return 0.4


def _decay(days_ago: int, half_life_days: int) -> float:
    """Half-life decay so recent news counts more."""
    days_ago = max(0, days_ago)
    half_life_days = max(1, int(half_life_days))
    # 0.5^(days/half_life). Keep a small floor to avoid 0.
    return max(0.0, 0.5 ** (days_ago / half_life_days))


def _parse_date(d: Optional[str]) -> Optional[dt.date]:
    if not d:
        return None
    try:
        return dt.date.fromisoformat(d[:10])
    except Exception:
        return None


def team_injury_scores(
    year: int,
    team: str,
    today: dt.date,
    decay_days: int = 28,
) -> Tuple[float, List[str]]:
    """
    Returns (score, notes). Higher score => more negative impact on the team.
    Works with CFBD responses that are either:
      A) [{team, injuries:[...]}...]  or
      B) [{team, status, position, updated, ...}, ...]
    """
    raw = cfbd_injuries(year=year, week=None) or []

    # Normalize to a flat list of injury dicts for the target team
    items: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        if raw and isinstance(raw[0], dict) and "injuries" in raw[0]:
            # shape A
            for entry in raw:
                if str(entry.get("team")) == team:
                    items = list(entry.get("injuries") or [])
                    break
        else:
            # shape B
            items = [r for r in raw if str(r.get("team")) == team]

    score = 0.0
    notes: List[str] = []
    half_life = max(5, decay_days // 3)  # recency matters more

    for e in items:
        try:
            status = (e.get("status") or "").title()
            pos = (e.get("position") or "")
            name = e.get("athlete") or e.get("player") or ""

            d = (
                _parse_date(e.get("updated"))
                or _parse_date(e.get("startDate"))
                or _parse_date(e.get("start_date"))
            )
            days_ago = (today - d).days if (d and today) else decay_days
            days_ago = min(days_ago, max(decay_days, 1))

            w = _pos_weight(pos) * _decay(days_ago, half_life)

            if status in NEG_STATUSES:
                score += 1.0 * w
            elif status in QUESTIONABLE_STATUSES:
                score += 0.5 * w

            if w > 0 and len(notes) < 6 and (status in NEG_STATUSES or status in QUESTIONABLE_STATUSES):
                notes.append(f"{name} {pos} {status}")
        except Exception:
            # Be robust to odd rows
            continue

    return float(score), notes


def build_injury_map(
    slate: List[Dict[str, Any]],
    year: int,
    decay_days: int = 28,
) -> Dict[str, Dict[str, Any]]:
    """
    Build {team: {'score': float, 'notes': [..]}} for all teams in the slate.
    """
    today = dt.date.today()
    teams: set[str] = set()
    for g in slate:
        teams.add(str(g.get("homeTeam")))
        teams.add(str(g.get("awayTeam")))

    out: Dict[str, Dict[str, Any]] = {}
    for t in teams:
        s, notes = team_injury_scores(year, t, today, decay_days)
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
    s_a = float(inj_map.get(away, {}).get("score", 0.0))

    # If HOME has more injury burden, move spread toward AWAY (negative for HOME)
    home_minus_away = s_h - s_a

    spread_per_point = float(cfg.get("injury_spread_weight_per_point", 0.5))
    total_per_point = float(cfg.get("injury_total_weight_per_point", 0.3))
    max_spread = float(cfg.get("max_spread_adj_component", 3.0))
    max_total = float(cfg.get("max_total_adj_component", 3.0))

    spread_adj_raw = -home_minus_away * spread_per_point
    total_adj_raw = -(s_h + s_a) * total_per_point  # more injuries -> slightly lower total

    spread_adj = float(max(-max_spread, min(spread_adj_raw, max_spread)))
    total_adj = float(max(-max_total, min(total_adj_raw, max_total)))

    note_bits: List[str] = []
    if inj_map.get(home, {}).get("notes"):
        note_bits.append("H: " + ", ".join(inj_map[home]["notes"][:3]))
    if inj_map.get(away, {}).get("notes"):
        note_bits.append("A: " + ", ".join(inj_map[away]["notes"][:3]))
    note = " | ".join(note_bits)

    return spread_adj, total_adj, note

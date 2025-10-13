from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

CFBD = "https://api.collegefootballdata.com"


# -------------------------- helpers --------------------------

def _headers() -> dict:
    key = os.environ.get("CFBD_API_KEY")
    if not key:
        raise EnvironmentError("Missing CFBD_API_KEY secret.")
    return {"Authorization": f"Bearer {key}"}


def _pick(cols: List[str], choices: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for c in choices:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _to_dt(x: Any) -> Optional[datetime]:
    if not x:
        return None
    try:
        # CFBD sends ISO8601; let pandas parse and convert to UTC
        return pd.to_datetime(x, utc=True).to_pydatetime()
    except Exception:
        return None


# -------------------------- fetch / normalize --------------------------

def fetch_lines(year: int, season_type: str = "regular") -> pd.DataFrame:
    """
    Hit CFBD /lines for a season. Response shape (simplified) is rows with nested 'lines' array.
    We explode to one row per (game, provider) and normalize:
      game_id, home, away, provider, home_spread, total, updated
    Assumption: 'spread' returned by CFBD is the spread from the HOME perspective
    (negative if the home team is favored). If CFBD exposes 'homeSpread', we prefer it.
    """
    url = f"{CFBD}/lines"
    params = {"year": int(year), "seasonType": season_type}
    r = requests.get(url, params=params, headers=_headers(), timeout=60)
    r.raise_for_status()
    raw = pd.DataFrame(r.json())

    if raw.empty:
        return pd.DataFrame(columns=["game_id","home","away","provider","home_spread","total","updated"])

    # explode nested lines
    if "lines" not in raw.columns:
        return pd.DataFrame(columns=["game_id","home","away","provider","home_spread","total","updated"])
    base_cols = ["id","home_team","away_team"]
    tmp = raw[base_cols + ["lines"]].explode("lines", ignore_index=True)
    if tmp.empty:
        return pd.DataFrame(columns=["game_id","home","away","provider","home_spread","total","updated"])
    lines_df = pd.json_normalize(tmp["lines"]).add_prefix("ln.")
    tmp = pd.concat([tmp.drop(columns=["lines"]), lines_df], axis=1)

    # column picking (snake/camel tolerant)
    cols = list(tmp.columns)
    gid  = _pick(cols, ["id","game_id","gameId"])
    home = _pick(cols, ["home_team","home","homeTeam"])
    away = _pick(cols, ["away_team","away","awayTeam"])
    prov = _pick(cols, ["ln.provider","ln.provider_name","ln.providerName"])
    sprA = _pick(cols, ["ln.home_spread","ln.homeSpread","ln.spread","ln.formattedSpread","ln.spreadOpen"])
    totA = _pick(cols, ["ln.over_under","ln.overUnder","ln.total","ln.totalOpen"])
    updA = _pick(cols, ["ln.last_updated","ln.lastUpdated","ln.updated"])

    out = pd.DataFrame({
        "game_id": tmp[gid],
        "home": tmp[home],
        "away": tmp[away],
        "provider": tmp[prov],
        "raw_spread": tmp[sprA] if sprA in tmp else None,
        "raw_total": tmp[totA] if totA in tmp else None,
        "updated": tmp[updA] if updA in tmp else None,
    }).copy()

    # Coerce numeric (handling strings l

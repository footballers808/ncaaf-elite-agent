from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import math
import requests
import pandas as pd
import numpy as np

CFBD = "https://api.collegefootballdata.com"


# ----------------------- your helper (kept) -----------------------
def prob_home_covers(line, mu, sigma=13.0):
    from math import erf
    z = (mu - line) / max(1e-6, sigma)
    return 0.5 * (1 + erf(z / np.sqrt(2)))


def predict_game(game, team_tbl, cfg):
    away, home = game.get("awayTeam") or game.get("away"), game.get("homeTeam") or game.get("home")
    th = team_tbl[team_tbl.team == home]
    ta = team_tbl[team_tbl.team == away]
    if th.empty or ta.empty:
        return None

    ph, pa = th.iloc[0].power, ta.iloc[0].power
    off_h, off_a = th.iloc[0].off_ppg, ta.iloc[0].off_ppg
    def_h, def_a = th.iloc[0].def_ppg, ta.iloc[0].def_ppg
    pace_h, pace_a = th.iloc[0].pace_ppg, ta.iloc[0].pace_ppg

    spread = cfg.get("power_scale", 1.0) * (ph - pa) + (
        0.5 if game.get("neutralSite") else cfg.get("hfa_points", 2.0)
    )
    pace_factor = float(np.clip((pace_h + pace_a) / 140.0, 0.85, 1.15))
    total = float(
        np.clip(
            off_h + off_a + 0.25 * (def_h + def_a),
            cfg.get("min_total_floor", 36.0),
            cfg.get("max_total_cap", 78.0),
        )
    ) * pace_factor
    total = float(
        np.clip(total, cfg.get("min_total_floor", 36.0), cfg.get("max_total_cap", 78.0))
    )

    home_pts = (total + spread) / 2.0
    away_pts = (total - spread) / 2.0
    p_cover = prob_home_covers(spread, spread, sigma=cfg.get("sigma_points", 13.0))

    return dict(
        gameId=game.get("id") or game.get("game_id"),
        start_local=game.get("start_date") or game.get("start_local"),
        home=home,
        away=away,
        spread=round(float(spread), 2),
        total=round(float(total), 1),
        pred_home_pts=int(round(home_pts)),
        pred_away_pts=int(round(away_pts)),
        p_home_cover=round(float(p_cover), 3),
    )


# ----------------------- CFBD helpers -----------------------
def _headers() -> dict:
    api_key = os.environ.get("CFBD_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing CFBD_API_KEY secret.")
    return {"Authorization": f"Bearer {api_key}"}


def _season_year_from_cfg(cfg: Dict[str, Any]) -> int:
    y = cfg.get("season_year")
    if isinstance(y, int):
        return y
    # default: current UTC year
    return datetime.now(timezone.utc).year


def _fetch_upcoming_games(year: int) -> List[Dict[str, Any]]:
    # Use CFBD /games for the season and filter to not-yet-started games (approx).
    url = f"{CFBD}/games"
    params = {"year": year, "seasonType": "regular"}
    r = requests.get(url, params=params, headers=_headers(), timeout=60)
    r.raise_for_status()
    rows = r.json()
    out = []
    for g in rows:
        # Keep rows that have teams but may not yet have final scores
        if g.get("home_team") and g.get("away_team"):
            out.append(
                {
                    "id": g.get("id"),
                    "homeTeam": g.get("home_team"),
                    "awayTeam": g.get("away_team"),
                    "start_local": g.get("start_date"),
                    "neutralSite": g.get("neutral_site", False),
                }
            )
    return out


def _build_naive_team_table(year: int) -> pd.DataFrame:
    """
    VERY NAIVE fallback team table so the pipeline can run in CI.
    If you already build a better table elsewhere, this will get replaced by your logic.
    """
    # Pull teams to at least know the team names; assign neutral powers.
    url = f"{CFBD}/teams/fbs"
    r = requests.get(url, headers=_headers(), timeout=60)
    r.raise_for_status()
    raw = pd.DataFrame(r.json())
    if raw.empty:
        return pd.DataFrame(columns=["team", "power", "off_ppg", "def_ppg", "pace_ppg"])
    df = pd.DataFrame({"team": raw["school"]})
    # Flat baseline so we still produce predictions; you can plug in your real ratings later.
    df["power"] = 0.0
    df["off_ppg"] = 28.0
    df["def_ppg"] = 27.0
    df["pace_ppg"] = 70.0
    return df


# ----------------------- Public pipeline API (adapters) -----------------------
def predict_games(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Adapter entrypoint used by the runner. Produces a list of prediction dicts.
    If your repo already has a richer predictor, swap this to call it.
    """
    try:
        year = _season_year_from_cfg(cfg)
        games = _fetch_upcoming_games(year)
        if not games:
            print("ℹ️ CFBD returned no games; predictions list will be empty.")
            return []
        team_tbl = _build_naive_team_table(year)
        out: List[Dict[str, Any]] = []
        for g in games:
            row = predict_game(g, team_tbl, cfg or {})
            if row:
                # Add some friendly text fields used by the email
                row["matchup"] = f"{row['away']} @ {row['home']}"
                row["model_text"] = f"{row['home']} -{abs(row['spread']):.1f}, total {row['total']:.1f}"
                row["market_text"] = ""  # you can fill this from odds provider later
                out.append(row)
        print(f"✅ Built {len(out)} predictions.")
        return out
    except Exception as e:
        print(f"⚠️ predict_games failed: {e}")
        return []


def load_model(cfg: Dict[str, Any]) -> Any:
    """
    Stub so CI runs. Replace with your real load if you have one.
    """
    return {}


def train_model(cfg: Dict[str, Any], model: Any, labeled_rows: List[Dict[str, Any]]) -> Any:
    """
    Stub (no-op). Replace with your learning logic; we keep it to preserve the pipeline shape.
    """
    return model


def save_model(cfg: Dict[str, Any], model: Any) -> None:
    """
    Stub (no-op). Replace to persist your model file if needed.
    """
    return

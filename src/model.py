from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
import pandas as pd
import numpy as np

from . import news  # NEW: bring in team sentiment

CFBD = "https://api.collegefootballdata.com"


# ----------------------- unchanged helper -----------------------
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

    # ----------------------- NEWS ADJUSTMENT (NEW) -----------------------
    ncfg = (cfg.get("news") or {})
    enable_news = ncfg.get("enabled", True)
    if enable_news:
        w_spread = float(ncfg.get("sentiment_weight_spread", 0.6))  # points per 1.0 sentiment delta
        w_total  = float(ncfg.get("sentiment_weight_total", 0.0))   # optional for totals
        cap      = float(ncfg.get("sentiment_cap", 1.5))            # cap the delta before weighting

        s_home = float(news.get_team_sentiment(home, cfg))
        s_away = float(news.get_team_sentiment(away, cfg))
        # Positive means home has better news vs away
        s_delta = max(-cap, min(cap, (s_home - s_away)))
        spread += w_spread * s_delta
        total  += w_total  * s_delta

    # --------------------------------------------------------------------

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
        # helpful text for email
        matchup=f"{away} @ {home}",
        model_text=f"{home} {spread:+.1f}, total {total:.1f}",
        market_text="",  # fill from odds when ready
        # rationale hints:
        news_note=f"newsΔ={s_delta:+.2f}" if enable_news else "",
        macro_note="",
        pace_note="",
        weather_note="",
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
    return datetime.now(timezone.utc).year


def _fetch_upcoming_games(year: int) -> List[Dict[str, Any]]:
    url = f"{CFBD}/games"
    params = {"year": year, "seasonType": "regular"}
    r = requests.get(url, params=params, headers=_headers(), timeout=60)
    r.raise_for_status()
    rows = r.json()
    out = []
    for g in rows:
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
    url = f"{CFBD}/teams/fbs"
    r = requests.get(url, headers=_headers(), timeout=60)
    r.raise_for_status()
    raw = pd.DataFrame(r.json())
    if raw.empty:
        return pd.DataFrame(columns=["team", "power", "off_ppg", "def_ppg", "pace_ppg"])
    df = pd.DataFrame({"team": raw["school"]})
    df["power"] = 0.0
    df["off_ppg"] = 28.0
    df["def_ppg"] = 27.0
    df["pace_ppg"] = 70.0
    return df


# ----------------------- Public pipeline API (adapters) -----------------------
def predict_games(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
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
                out.append(row)
        print(f"✅ Built {len(out)} predictions.")
        return out
    except Exception as e:
        print(f"⚠️ predict_games failed: {e}")
        return []


def load_model(cfg: Dict[str, Any]) -> Any:
    return {}


def train_model(cfg: Dict[str, Any], model: Any, labeled_rows: List[Dict[str, Any]]) -> Any:
    return model


def save_model(cfg: Dict[str, Any], model: Any) -> None:
    return

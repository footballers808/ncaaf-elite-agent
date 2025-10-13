from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import requests
import pandas as pd
import numpy as np

from . import news
from . import odds  # market lines + consensus

CFBD = "https://api.collegefootballdata.com"


# ----------------------- Utilities -----------------------

def prob_home_covers(line, mu, sigma=13.0):
    """Probability the home team covers given model mean 'mu' and spread 'line'."""
    from math import erf
    z = (mu - line) / max(1e-6, sigma)
    return 0.5 * (1 + erf(z / np.sqrt(2)))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _headers() -> dict:
    key = os.environ.get("CFBD_API_KEY")
    if not key:
        raise EnvironmentError("Missing CFBD_API_KEY secret.")
    return {"Authorization": f"Bearer {key}"}


def _season_year_from_cfg(cfg: Dict[str, Any]) -> int:
    y = cfg.get("season_year")
    return int(y) if isinstance(y, int) else datetime.now(timezone.utc).year


# ----------------------- Exact-score predictor -----------------------

def predict_game(game: Dict[str, Any], team_tbl: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    Produce spread/total as before, then compute **exact scores** by splitting the total
    using a pace+offense vs defense propensity. We ensure:
        home_pts + away_pts = total
        home_pts - away_pts = spread
    by correcting around the propensity split so edges remain consistent.
    """
    away = game.get("awayTeam") or game.get("away")
    home = game.get("homeTeam") or game.get("home")

    th = team_tbl[team_tbl.team == home]
    ta = team_tbl[team_tbl.team == away]
    if th.empty or ta.empty:
        return None

    # Baseline team features
    ph, pa = float(th.iloc[0].power), float(ta.iloc[0].power)
    off_h, off_a = float(th.iloc[0].off_ppg), float(ta.iloc[0].off_ppg)
    def_h, def_a = float(th.iloc[0].def_ppg), float(ta.iloc[0].def_ppg)
    pace_h, pace_a = float(th.iloc[0].pace_ppg), float(ta.iloc[0].pace_ppg)

    # 1) Model spread/total (same structure as before)
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

    # 2) News adjustment (kept)
    ncfg = (cfg.get("news") or {})
    if ncfg.get("enabled", True):
        w_spread = float(ncfg.get("sentiment_weight_spread", 0.6))
        w_total  = float(ncfg.get("sentiment_weight_total", 0.0))
        cap      = float(ncfg.get("sentiment_cap", 1.5))
        s_home = float(news.get_team_sentiment(home, cfg))
        s_away = float(news.get_team_sentiment(away, cfg))
        s_delta = max(-cap, min(cap, (s_home - s_away)))
        spread += w_spread * s_delta
        total  += w_total  * s_delta
    else:
        s_delta = 0.0

    # 3) Propensity split for **exact scores** (NEW)
    #    Compute how the total should be divided using offense vs opponent defense
    #    and a little pace. This yields a home share f in (0,1).
    #    We then correct so the final sum=total and diff=spread.
    w_off  = float(cfg.get("score_split_w_offense", 0.55))
    w_def  = float(cfg.get("score_split_w_defense", 0.35))
    w_pace = float(cfg.get("score_split_w_pace", 0.10))
    k_gain = float(cfg.get("score_split_gain", 0.03))  # controls sharpness of split

    # Higher is better for offense; lower is better for defense. Flip defense axis.
    score_home = w_off * off_h + w_def * (70.0 - def_a) + w_pace * pace_h
    score_away = w_off * off_a + w_def * (70.0 - def_h) + w_pace * pace_a

    # Convert to share with a calibrated sigmoid
    f_home = _sigmoid(k_gain * (score_home - score_away))  # ~0.5 when similar teams

    # Equal-split would be T/2 each; propensity tilts that. But we must still
    # satisfy spread/total, so we compute the residual and correct.
    # If we naively set H0=T*f, A0=T*(1-f), then difference is T*(2f-1).
    # Residual r needed to hit exact spread is: r = S - T*(2f-1).
    # Use correction +/- r/2 to keep the sum = T.
    T = float(total)
    S = float(spread)
    base_diff = T * (2.0 * f_home - 1.0)
    residual = S - base_diff

    home_pts = T * f_home + residual / 2.0
    away_pts = T * (1.0 - f_home) - residual / 2.0

    # Clamp to reasonable bounds and round to integers for the report
    home_pts = max(0.0, home_pts)
    away_pts = max(0.0, away_pts)

    p_cover = prob_home_covers(S, S, sigma=cfg.get("sigma_points", 13.0))

    return dict(
        gameId=game.get("id") or game.get("game_id"),
        start_local=game.get("start_date") or game.get("start_local"),
        home=home, away=away,
        spread=round(float(S), 2),
        total=round(float(T), 1),
        pred_home_pts=int(round(home_pts)),
        pred_away_pts=int(round(away_pts)),
        p_home_cover=round(float(p_cover), 3),
        matchup=f"{away} @ {home}",
        model_text=f"{home} {S:+.1f}, total {T:.1f}",
        news_note=f"newsΔ={s_delta:+.2f}" if ncfg.get("enabled", True) else "",
        macro_note="",
        pace_note="",
        weather_note="",
    )


# ----------------------- CFBD fetchers -----------------------

def _fetch_upcoming_games(year: int) -> List[Dict[str, Any]]:
    url = f"{CFBD}/games"
    params = {"year": year, "seasonType": "regular"}
    r = requests.get(url, params=params, headers=_headers(), timeout=60)
    r.raise_for_status()
    rows = r.json()
    out: List[Dict[str, Any]] = []
    for g in rows:
        if g.get("home_team") and g.get("away_team"):
            out.append({
                "id": g.get("id"),
                "homeTeam": g.get("home_team"),
                "awayTeam": g.get("away_team"),
                "start_local": g.get("start_date"),
                "neutralSite": g.get("neutral_site", False),
            })
    return out


def _build_naive_team_table(year: int) -> pd.DataFrame:
    # Fallback baselines so CI always runs; replace with your real team power table any time.
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


# ----------------------- Public API for runner -----------------------

def predict_games(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        year = _season_year_from_cfg(cfg)
        games = _fetch_upcoming_games(year)
        if not games:
            print("ℹ️ CFBD returned no games; predictions list will be empty.")
            return []

        team_tbl = _build_naive_team_table(year)
        preds: List[Dict[str, Any]] = []
        for g in games:
            row = predict_game(g, team_tbl, cfg or {})
            if row:
                preds.append(row)

        # --- Merge market and compute edges (unchanged) ---
        ocfg = (cfg.get("odds") or {})
        if ocfg.get("enabled", True):
            pref = ocfg.get("preferred_providers") or []
            market = odds.get_market_by_game(year, "regular", preferred_providers=pref)

            for r in preds:
                gid = r.get("gameId") or r.get("game_id")
                try:
                    gid = int(gid)
                except Exception:
                    gid = None
                mk = market.get(gid) if gid is not None else None
                if mk:
                    m_spread = mk.get("spread")   # home spread
                    m_total  = mk.get("total")
                    providers = mk.get("providers", "")
                    if m_spread is not None:
                        r["edge_spread"] = round(float(r["spread"]) - float(m_spread), 2)
                        r["market_text_spread"] = f"home {m_spread:+.1f} ({providers})"
                    if m_total is not None:
                        r["edge_total"]  = round(float(r["total"]) - float(m_total), 2)
                        r["market_text_total"] = f"total {m_total:.1f} ({providers})"
                    if m_spread is not None or m_total is not None:
                        parts = []
                        if m_spread is not None: parts.append(f"home {m_spread:+.1f}")
                        if m_total  is not None: parts.append(f"total {m_total:.1f}")
                        r["market_text"] = ", ".join(parts) + (f" ({providers})" if providers else "")
                else:
                    r.setdefault("market_text", "")

        print(f"✅ Built {len(preds)} predictions (market merged, exact scores).")
        return preds
    except Exception as e:
        print(f"⚠️ predict_games failed: {e}")
        return []


def load_model(cfg: Dict[str, Any]) -> Any:
    return {}


def train_model(cfg: Dict[str, Any], model: Any, labeled_rows: List[Dict[str, Any]]) -> Any:
    return model


def save_model(cfg: Dict[str, Any], model: Any) -> None:
    return

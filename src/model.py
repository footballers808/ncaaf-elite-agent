from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from . import news
from . import odds

CFBD = "https://api.collegefootballdata.com"


# ----------------------- Utilities -----------------------

def _headers() -> dict:
    key = os.environ.get("CFBD_API_KEY")
    if not key:
        raise EnvironmentError("Missing CFBD_API_KEY secret.")
    return {"Authorization": f"Bearer {key}"}


def _season_year_from_cfg(cfg: Dict[str, Any]) -> int:
    y = cfg.get("season_year")
    return int(y) if isinstance(y, int) else datetime.now(timezone.utc).year


def prob_home_covers(line, mu, sigma=13.0):
    from math import erf
    z = (mu - line) / max(1e-6, sigma)
    return 0.5 * (1 + erf(z / np.sqrt(2)))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


# ----------------------- Team power store -----------------------

STORE_DIR = "store"
TEAM_POWER_PATH = os.path.join(STORE_DIR, "team_power.parquet")

def _ensure_store():
    os.makedirs(STORE_DIR, exist_ok=True)

def _load_team_power() -> Optional[pd.DataFrame]:
    if os.path.exists(TEAM_POWER_PATH):
        try:
            df = pd.read_parquet(TEAM_POWER_PATH)
            # ensure required columns
            for col in ["team", "power", "off_ppg", "def_ppg", "pace_ppg"]:
                if col not in df.columns:
                    return None
            return df
        except Exception:
            return None
    return None

def _save_team_power(df: pd.DataFrame) -> None:
    _ensure_store()
    df.to_parquet(TEAM_POWER_PATH, index=False)


# ----------------------- CFBD fetchers / baselines -----------------------

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

def _build_naive_team_table() -> pd.DataFrame:
    url = f"{CFBD}/teams/fbs"
    r = requests.get(url, headers=_headers(), timeout=60)
    r.raise_for_status()
    raw = pd.DataFrame(r.json())
    if raw.empty:
        return pd.DataFrame(columns=["team", "power", "off_ppg", "def_ppg", "pace_ppg"])
    df = pd.DataFrame({"team": raw["school"]})
    # Start neutral; these will learn over time
    df["power"] = 0.0
    df["off_ppg"] = 28.0
    df["def_ppg"] = 27.0
    df["pace_ppg"] = 70.0
    return df

def _get_team_table() -> pd.DataFrame:
    tbl = _load_team_power()
    if tbl is not None and not tbl.empty:
        return tbl
    # First-time: initialize
    tbl = _build_naive_team_table()
    _save_team_power(tbl)
    return tbl


# ----------------------- Prediction with exact scores -----------------------

def predict_game(game: Dict[str, Any], team_tbl: pd.DataFrame, cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    away = game.get("awayTeam") or game.get("away")
    home = game.get("homeTeam") or game.get("home")

    th = team_tbl[team_tbl.team == home]
    ta = team_tbl[team_tbl.team == away]
    if th.empty or ta.empty:
        return None

    ph, pa = float(th.iloc[0].power), float(ta.iloc[0].power)
    off_h, off_a = float(th.iloc[0].off_ppg), float(ta.iloc[0].off_ppg)
    def_h, def_a = float(th.iloc[0].def_ppg), float(ta.iloc[0].def_ppg)
    pace_h, pace_a = float(th.iloc[0].pace_ppg), float(ta.iloc[0].pace_ppg)

    # Spread/Total baselines
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
    total = float(np.clip(total, cfg.get("min_total_floor", 36.0), cfg.get("max_total_cap", 78.0)))

    # News tilt
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

    # Exact-score split
    w_off  = float(cfg.get("score_split_w_offense", 0.55))
    w_def  = float(cfg.get("score_split_w_defense", 0.35))
    w_pace = float(cfg.get("score_split_w_pace", 0.10))
    k_gain = float(cfg.get("score_split_gain", 0.03))

    score_home = w_off * off_h + w_def * (70.0 - def_a) + w_pace * pace_h
    score_away = w_off * off_a + w_def * (70.0 - def_h) + w_pace * pace_a
    f_home = _sigmoid(k_gain * (score_home - score_away))

    T = float(total)
    S = float(spread)
    base_diff = T * (2.0 * f_home - 1.0)
    residual = S - base_diff
    home_pts = max(0.0, T * f_home + residual / 2.0)
    away_pts = max(0.0, T * (1.0 - f_home) - residual / 2.0)

    p_cover = prob_home_covers(S, S, sigma=cfg.get("sigma_points", 13.0))

    return {
        "gameId": game.get("id") or game.get("game_id"),
        "start_local": game.get("start_date") or game.get("start_local"),
        "home": home,
        "away": away,
        "spread": round(S, 2),
        "total": round(T, 1),
        "pred_home_pts": int(round(home_pts)),
        "pred_away_pts": int(round(away_pts)),
        "p_home_cover": round(float(p_cover), 3),
        "matchup": f"{away} @ {home}",
        "model_text": f"{home} {S:+.1f}, total {T:.1f}",
        "news_note": f"newsΔ={s_delta:+.2f}" if ncfg.get("enabled", True) else "",
        "macro_note": "",
        "pace_note": "",
        "weather_note": "",
    }


# ----------------------- Runner API -----------------------

def predict_games(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        year = _season_year_from_cfg(cfg)
        team_tbl = _get_team_table()
        games = _fetch_upcoming_games(year)
        if not games:
            print("ℹ️ CFBD returned no games; predictions list will be empty.")
            return []

        preds: List[Dict[str, Any]] = []
        for g in games:
            row = predict_game(g, team_tbl, cfg or {})
            if row:
                preds.append(row)

        # Merge market + edges
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
                    m_spread = mk.get("spread")
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
    """Return the current team power table as the 'model'."""
    try:
        return _get_team_table()
    except Exception as e:
        print(f"⚠️ load_model: {e}")
        return _build_naive_team_table()


def save_model(cfg: Dict[str, Any], model: Any) -> None:
    """Persist the team power table."""
    try:
        if isinstance(model, pd.DataFrame) and not model.empty:
            _save_team_power(model)
            print(f"💾 Saved team powers to {TEAM_POWER_PATH} ({len(model)} teams).")
    except Exception as e:
        print(f"⚠️ save_model: {e}")


def train_model(cfg: Dict[str, Any], model: Any, labeled_rows: List[Dict[str, Any]]) -> Any:
    """
    Elo-like update on team powers using final scores.
    For each completed game:
      margin = home_points - away_points
      expected = (power_home - power_away + HFA)
      delta = K * clip(margin, ±margin_cap) - K * clip(expected, ±margin_cap)? No:
      We want to move toward margin: power_home += K*(margin - expected), away -= same
    """
    if not isinstance(model, pd.DataFrame) or model.empty:
        model = _get_team_table()

    lcfg = (cfg.get("learn") or {})
    if not lcfg.get("enabled", True):
        return model

    K = float(lcfg.get("k_factor", 0.25))
    cap = float(lcfg.get("margin_cap", 24.0))
    decay = float(lcfg.get("weekly_decay", 0.00))
    hfa_learn = float(lcfg.get("hfa_points_for_learn", cfg.get("hfa_points", 2.0)))

    # optional weekly decay toward 0 power
    if decay > 0 and "power" in model.columns:
        model["power"] = model["power"] * (1.0 - decay)

    # Make quick lookup
    model_index = {t: i for i, t in enumerate(model["team"])}

    updates = 0
    for r in labeled_rows or []:
        home = r.get("home"); away = r.get("away")
        hp = r.get("home_points"); ap = r.get("away_points")
        if home is None or away is None or hp is None or ap is None:
            continue
        try:
            hp = float(hp); ap = float(ap)
        except Exception:
            continue

        ih = model_index.get(home); ia = model_index.get(away)
        if ih is None or ia is None:
            # unseen team? add neutral row
            # (rare for FCS crossover; keeps learning robust)
            for t in [home, away]:
                if t not in model_index:
                    model = pd.concat([model, pd.DataFrame([{
                        "team": t, "power": 0.0, "off_ppg": 28.0, "def_ppg": 27.0, "pace_ppg": 70.0
                    }])], ignore_index=True)
            model_index = {t: i for i, t in enumerate(model["team"])}
            ih = model_index.get(home); ia = model_index.get(away)

        power_h = float(model.at[ih, "power"])
        power_a = float(model.at[ia, "power"])

        margin = hp - ap                                # actual home margin
        margin = float(np.clip(margin, -cap, cap))
        expected = (power_h - power_a + hfa_learn)      # expected home margin from powers
        expected = float(np.clip(expected, -cap, cap))

        delta = K * (margin - expected)
        model.at[ih, "power"] = power_h + delta
        model.at[ia, "power"] = power_a - delta
        updates += 1

    if updates:
        print(f"🧠 Learned from {updates} completed games; power mean={model['power'].mean():+.2f}, std={model['power'].std():.2f}")
    else:
        print("ℹ️ No learnable rows this run.")

    return model

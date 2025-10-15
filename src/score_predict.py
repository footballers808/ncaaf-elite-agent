# SPDX-License-Identifier: MIT
"""
Predict final scores for a given week using trained models:
- artifacts/model_homepts.joblib
- artifacts/model_awaypts.joblib

Writes CSV with: season, week, home_team, away_team, pred_home, pred_away, spread, total
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import load as joblib_load

# ----- CFBD helpers -----------------------------------------------------

def _maybe_import_net():
    try:
        from src.net import cfbd_get  # type: ignore
        return cfbd_get
    except Exception:
        return None

_cfbd = _maybe_import_net()

def _fallback_cfbd_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    import time
    import requests

    base = "https://api.collegefootballdata.com"
    api_key = os.getenv("CFBD_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("CFBD_API_KEY missing")

    min_sleep_ms = max(int(os.getenv("CFBD_MIN_SLEEP_MS", "2000")), 500)
    max_retries  = max(int(os.getenv("CFBD_MAX_RETRIES", "20")), 1)
    backoff_base = float(os.getenv("CFBD_BACKOFF_BASE_S", "2.0"))

    url = f"{base}{path}"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = params or {}

    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=35)
            if r.status_code == 200:
                return r.json()
            last_exc = RuntimeError(f"HTTP {r.status_code} {url} params={params} body={r.text[:200]}")
        except Exception as e:
            last_exc = e
        sleep_s = (min_sleep_ms / 1000.0) + (backoff_base ** attempt) * 0.3
        time.sleep(sleep_s)
    if last_exc:
        raise last_exc
    raise RuntimeError(f"Failed GET {url}")

def cfbd_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    if _cfbd is not None:
        return _cfbd(path, params or {})
    return _fallback_cfbd_get(path, params)

# ----- utilities --------------------------------------------------------

def current_year() -> int:
    return datetime.now(timezone.utc).year

def penalties_parquet() -> str:
    p = "artifacts/features/penalties.parquet"
    if not os.path.exists(p):
        raise FileNotFoundError("penalties.parquet not found. Run learn/predict pipeline first.")
    return p

def find_week_auto(year: int) -> int:
    """Pick the next upcoming week with no scores yet."""
    js = cfbd_get("/games", {"year": year, "seasonType": "regular"})
    future_weeks = []
    for g in js or []:
        if g.get("home_points") is None and g.get("away_points") is None:
            wk = g.get("week")
            if isinstance(wk, int):
                future_weeks.append(wk)
    if not future_weeks:
        # fallback: last week in schedule
        weeks = sorted(set([g.get("week") for g in js if isinstance(g.get("week"), int)]))
        return weeks[-1] if weeks else 1
    return min(future_weeks)

@dataclass
class Game:
    season: int
    week: int
    home_team: str
    away_team: str

def scheduled_games(year: int, week: int) -> List[Game]:
    js = cfbd_get("/games", {"year": year, "week": week, "seasonType": "regular"})
    out: List[Game] = []
    for g in js or []:
        ht = g.get("home_team") or g.get("homeTeam")
        at = g.get("away_team") or g.get("awayTeam")
        if ht and at:
            out.append(Game(season=year, week=int(week), home_team=str(ht), away_team=str(at)))
    return out

# ----- feature assembly for prediction ---------------------------------

def load_penalties_df() -> pd.DataFrame:
    return pd.read_parquet(penalties_parquet())

def build_feature_rows(games: List[Game], pen: pd.DataFrame) -> pd.DataFrame:
    if not games:
        return pd.DataFrame()

    feat_cols = [c for c in pen.columns if c not in ("season", "team", "games")]
    ph = pen.rename(columns={c: f"home_{c}" for c in pen.columns})
    pa = pen.rename(columns={c: f"away_{c}" for c in pen.columns})

    rows = []
    for gm in games:
        rows.append({"season": gm.season, "home_team": gm.home_team, "away_team": gm.away_team})
    base = pd.DataFrame(rows)

    df = base.merge(
        ph, left_on=["season", "home_team"], right_on=["home_season", "home_team"], how="left"
    ).merge(
        pa, left_on=["season", "away_team"], right_on=["away_season", "away_team"], how="left"
    )

    model_feats = []
    for c in feat_cols:
        model_feats.append(f"home_{c}")
        model_feats.append(f"away_{c}")

    for c in model_feats:
        if c not in df.columns:
            df[c] = 0.0
    df[model_feats] = df[model_feats].fillna(0.0)

    # Keep original ids for output
    df["_season"] = base["season"]
    df["_home_team"] = base["home_team"]
    df["_away_team"] = base["away_team"]
    return df[["_season", "_home_team", "_away_team"] + model_feats]

# ----- main predict -----------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser("score_predict")
    ap.add_argument("--week", type=str, default="auto", help="ISO yyyy-ww or 'auto'")
    ap.add_argument("--model-home", type=str, default="artifacts/model_homepts.joblib")
    ap.add_argument("--model-away", type=str, default="artifacts/model_awaypts.joblib")
    ap.add_argument("--out", type=str, default="artifacts/score_predictions.csv")
    args = ap.parse_args()

    if not os.path.exists(args.model_home) or not os.path.exists(args.model_away):
        raise FileNotFoundError("Score models missing. Run learn mode first.")

    # Parse week
    if args.week == "auto":
        yr = current_year()
        wk = find_week_auto(yr)
    else:
        # accepts "yyyy-ww"
        try:
            yr, wk = args.week.split("-")
            yr, wk = int(yr), int(wk)
        except Exception:
            raise ValueError("--week must be 'auto' or 'yyyy-ww'")

    games = scheduled_games(yr, wk)
    if not games:
        print("[score_predict] No scheduled games found.", file=sys.stderr)
        # Still write an empty CSV for consistency
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["season", "week", "home_team", "away_team", "pred_home", "pred_away", "spread", "total"])
        return 0

    pen = load_penalties_df()
    feats = build_feature_rows(games, pen)

    model_home = joblib_load(args.model_home)
    model_away = joblib_load(args.model_away)

    X = feats.drop(columns=["_season", "_home_team", "_away_team"])
    pred_home = np.clip(model_home.predict(X), 0, 100).round(1)
    pred_away = np.clip(model_away.predict(X), 0, 100).round(1)
    spread = (pred_home - pred_away).round(1)
    total = (pred_home + pred_away).round(1)

    out_rows = []
    for i in range(len(feats)):
        out_rows.append([
            int(feats.loc[i, "_season"]),
            int(wk),
            str(feats.loc[i, "_home_team"]),
            str(feats.loc[i, "_away_team"]),
            float(pred_home[i]),
            float(pred_away[i]),
            float(spread[i]),
            float(total[i]),
        ])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["season", "week", "home_team", "away_team", "pred_home", "pred_away", "spread", "total"])
        writer.writerows(out_rows)

    print(f"[score_predict] Wrote {len(out_rows)} rows => {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

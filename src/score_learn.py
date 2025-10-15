# SPDX-License-Identifier: MIT
"""
Train two regression models to predict final scores:
- Home points model
- Away points model

Uses penalties.parquet as features; joins to CFBD games for the last N seasons.
Strict mode: raises on missing data so CI fails and you can re-run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd
from joblib import dump as joblib_dump

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ----- CFBD helpers (same retry knobs as other modules) ----------------

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

# ----- data assembly ----------------------------------------------------

def seasons_last_n(years: int) -> List[int]:
    end = datetime.now(timezone.utc).year
    years = max(1, int(years))
    return list(range(end - years + 1, end + 1))

def fetch_games(seasons: List[int]) -> pd.DataFrame:
    rows = []
    for y in seasons:
        js = cfbd_get("/games", {"year": y, "seasonType": "regular"})
        for g in js or []:
            ht = g.get("home_team") or g.get("homeTeam")
            at = g.get("away_team") or g.get("awayTeam")
            hs = g.get("home_points")
            aw = g.get("away_points")
            if ht and at and hs is not None and aw is not None:
                rows.append({
                    "season": y,
                    "home_team": str(ht),
                    "away_team": str(at),
                    "home_points": int(hs),
                    "away_points": int(aw),
                })
    return pd.DataFrame(rows)

def load_penalties(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    return pd.read_parquet(path)

def assemble(games: pd.DataFrame, pen: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame()

    feat_cols = [c for c in pen.columns if c not in ("season", "team", "games")]
    ph = pen.rename(columns={c: f"home_{c}" for c in pen.columns})
    pa = pen.rename(columns={c: f"away_{c}" for c in pen.columns})

    df = games.merge(
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

    keep = model_feats + ["season", "home_points", "away_points"]
    return df[keep]

# ----- training ---------------------------------------------------------

def train_regressor(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> Pipeline:
    num_cols = list(X.columns)
    prep = ColumnTransformer([("num", StandardScaler(with_mean=False), num_cols)], remainder="drop")

    gbr = GradientBoostingRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42
    )
    pipe = Pipeline([("prep", prep), ("gbr", gbr)])

    # quick CV (by season groups)
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    maes = []
    for tr, va in gkf.split(X, y, groups):
        pipe.fit(X.iloc[tr], y[tr])
        p = pipe.predict(X.iloc[va])
        maes.append(mean_absolute_error(y[va], p))
    print(f"[score_learn] CV MAE: {np.mean(maes):.3f} ± {np.std(maes):.3f}")

    # Final fit on all
    pipe.fit(X, y)
    return pipe

def main() -> int:
    ap = argparse.ArgumentParser("score_learn")
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--features", type=str, default="artifacts/features/penalties.parquet")
    ap.add_argument("--save-home", type=str, default="artifacts/model_homepts.joblib")
    ap.add_argument("--save-away", type=str, default="artifacts/model_awaypts.joblib")
    ap.add_argument("--metrics", type=str, default="artifacts/score_metrics.json")
    args = ap.parse_args()

    seasons = seasons_last_n(args.years)
    pen = load_penalties(args.features)
    games = fetch_games(seasons)
    if games.empty:
        raise RuntimeError("No historical games fetched. Aborting learn.")

    df = assemble(games, pen)
    if df.empty:
        raise RuntimeError("Assembled dataset empty. Aborting learn.")

    X = df.drop(columns=["home_points", "away_points"])
    g = df["season"].values

    # HOME model
    y_home = df["home_points"].values.astype(float)
    home_model = train_regressor(X, y_home, g)
    os.makedirs(os.path.dirname(os.path.abspath(args.save_home)), exist_ok=True)
    joblib_dump(home_model, args.save_home)
    print(f"[score_learn] Saved home-points model => {args.save_home}")

    # AWAY model
    y_away = df["away_points"].values.astype(float)
    away_model = train_regressor(X, y_away, g)
    os.makedirs(os.path.dirname(os.path.abspath(args.save_away)), exist_ok=True)
    joblib_dump(away_model, args.save_away)
    print(f"[score_learn] Saved away-points model => {args.save_away}")

    # quick fit metrics on full data (not for reporting, just sanity)
    m_home = mean_absolute_error(y_home, home_model.predict(X))
    m_away = mean_absolute_error(y_away, away_model.predict(X))
    metrics = {"mae_home_fit": float(m_home), "mae_away_fit": float(m_away)}
    os.makedirs(os.path.dirname(os.path.abspath(args.metrics)), exist_ok=True)
    with open(args.metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[score_learn] Metrics => {args.metrics}: {metrics}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

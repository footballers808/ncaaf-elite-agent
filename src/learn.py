# SPDX-License-Identifier: MIT
"""
Robust training step:
- Tries to fetch games for the last N seasons.
- Joins with penalties features.
- Trains LogisticRegression when data is available.
- If CFBD fetch fails or dataset is empty, falls back to DummyClassifier
  (predicts prior / 0.5) so the pipeline completes.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from joblib import dump as joblib_dump


# Optional net helper (preferred) with retries/backoff
def _maybe_import_net():
    try:
        from src.net import cfbd_get  # type: ignore
        return cfbd_get
    except Exception:
        return None


_cfbd_get = _maybe_import_net()


def _fallback_cfbd_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    import time
    import requests

    base = "https://api.collegefootballdata.com"
    api_key = os.getenv("CFBD_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("CFBD_API_KEY missing for learn().")

    min_sleep_ms = max(int(os.getenv("CFBD_MIN_SLEEP_MS", "1000")), 500)
    max_retries = max(int(os.getenv("CFBD_MAX_RETRIES", "8")), 1)
    backoff_base = float(os.getenv("CFBD_BACKOFF_BASE_S", "1.6"))

    url = f"{base}{path}"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = params or {}

    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            last_exc = RuntimeError(
                f"HTTP {r.status_code} for {url} params={params} body={r.text[:200]}"
            )
        except Exception as e:
            last_exc = e

        sleep_s = (min_sleep_ms / 1000.0) + (backoff_base ** attempt) * 0.2
        time.sleep(sleep_s)

    if last_exc:
        raise last_exc
    raise RuntimeError(f"Failed to GET {url} after {max_retries} attempts")


def _cfbd_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    if _cfbd_get is not None:
        return _cfbd_get(path, params or {})
    return _fallback_cfbd_get(path, params)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("learn")
    p.add_argument("--years", type=int, default=3, help="How many seasons back to learn")
    p.add_argument("--save", type=str, default="artifacts/model.joblib", help="Output model path")
    p.add_argument("--features", type=str, default="artifacts/features/penalties.parquet",
                   help="Penalties features parquet")
    return p.parse_args()


def _season_range(years: int) -> List[int]:
    end = datetime.now(timezone.utc).year
    years = max(1, int(years))
    return list(range(end - years + 1, end + 1))


def _fetch_games_for_seasons(seasons: List[int]) -> pd.DataFrame:
    rows = []
    for y in seasons:
        js = _cfbd_get("/games", {"year": y, "seasonType": "regular"})
        for g in js or []:
            ht = g.get("home_team") or g.get("homeTeam")
            at = g.get("away_team") or g.get("awayTeam")
            hs = g.get("home_points")
            as_ = g.get("away_points")
            if ht and at and hs is not None and as_ is not None:
                rows.append({
                    "season": y,
                    "home_team": str(ht),
                    "away_team": str(at),
                    "home_points": int(hs),
                    "away_points": int(as_),
                })
    return pd.DataFrame(rows)


def _load_penalties(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[learn] WARNING: penalties features not found at {path}. Using empty frame.", file=sys.stderr)
        return pd.DataFrame(columns=["season", "team", "games"])
    try:
        return pd.read_parquet(path)
    except Exception as e:
        print(f"[learn] WARNING: failed to read {path}: {e}. Using empty frame.", file=sys.stderr)
        return pd.DataFrame(columns=["season", "team", "games"])


def _assemble_dataset(games: pd.DataFrame, pen: pd.DataFrame) -> pd.DataFrame:
    """Create one row per game from home perspective with label home_win."""
    if games.empty:
        return pd.DataFrame()

    # Identify feature columns in penalties
    feat_cols = [c for c in pen.columns if c not in ("season", "team", "games")]

    # Home/away merges
    ph = pen.rename(columns={c: f"home_{c}" for c in pen.columns})
    pa = pen.rename(columns={c: f"away_{c}" for c in pen.columns})

    df = games.copy()
    df["home_win"] = (df["home_points"] > df["away_points"]).astype(int)

    df = df.merge(
        ph,
        left_on=["season", "home_team"],
        right_on=["home_season", "home_team"],
        how="left",
    ).merge(
        pa,
        left_on=["season", "away_team"],
        right_on=["away_season", "away_team"],
        how="left",
    )

    # Build model feature list (only penalties rollups)
    model_feats = []
    for c in feat_cols:
        model_feats.append(f"home_{c}")
        model_feats.append(f"away_{c}")

    # Some columns may be missing if pen is empty; create them
    for c in model_feats:
        if c not in df.columns:
            df[c] = 0.0

    # Minimal cleaning
    df[model_feats] = df[model_feats].fillna(0.0)

    keep = model_feats + ["home_win"]
    return df[keep]


def _train_or_fallback(train_df: pd.DataFrame, save_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    if train_df.empty or train_df["home_win"].nunique() < 2:
        print("[learn] No usable training data. Training DummyClassifier (prior).", file=sys.stderr)
        X = np.zeros((1, 1), dtype=float)
        y = np.array([0], dtype=int)
        model = DummyClassifier(strategy="prior")  # predicts class prior
        model.fit(X, y)
        joblib_dump(model, save_path)
        print(f"[learn] Saved fallback model to {save_path}")
        return

    y = train_df["home_win"].astype(int).values
    X = train_df.drop(columns=["home_win"])

    numeric_cols = list(X.columns)
    pre = ColumnTransformer(
        transformers=[("num", StandardScaler(with_mean=False), numeric_cols)],
        remainder="drop",
        sparse_threshold=0.0,
    )
    clf = LogisticRegression(max_iter=500, solver="lbfgs")

    pipe = Pipeline(steps=[("prep", pre), ("clf", clf)])
    pipe.fit(X, y)

    # quick sanity metric
    try:
        p = pipe.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, p)
        print(f"[learn] train AUC={auc:.3f} on {len(y)} samples")
    except Exception:
        pass

    joblib_dump(pipe, save_path)
    print(f"[learn] Saved model to {save_path}")


def main() -> int:
    args = _parse_args()
    seasons = _season_range(args.years)

    # Load features
    pen = _load_penalties(args.features)

    # Fetch games (with robust fallback)
    try:
        games = _fetch_games_for_seasons(seasons)
    except Exception as e:
        print(f"[learn] WARNING: CFBD fetch failed: {e}. Falling back to dummy model.", file=sys.stderr)
        games = pd.DataFrame()

    train_df = _assemble_dataset(games, pen)
    _train_or_fallback(train_df, args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

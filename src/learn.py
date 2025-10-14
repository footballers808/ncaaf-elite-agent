# src/learn.py
from __future__ import annotations
import argparse, json, pathlib
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss

from src.net import cfbd_get
from src.feat_penalties import load_penalty_features, features_for_game

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

def _years_from_latest(n_years: int) -> List[int]:
    from datetime import datetime, timezone
    cur = datetime.now(tz=timezone.utc).year
    return list(range(cur - n_years + 1, cur + 1))

def _fetch_regular_games(year: int) -> pd.DataFrame:
    g = cfbd_get("/games", {"year": year, "seasonType": "regular"}) or []
    if not g:
        return pd.DataFrame()
    df = pd.DataFrame(g)
    keep = [c for c in [
        "season","week","start_date","home_team","away_team","home_points","away_points","neutral_site","venue"
    ] if c in df.columns]
    out = df[keep].copy()
    out["start_dt"] = pd.to_datetime(out["start_date"], utc=True, errors="coerce")
    out = out.dropna(subset=["home_team","away_team","start_dt"])
    out = out.dropna(subset=["home_points","away_points"])
    out["y"] = (out["home_points"].astype(float) > out["away_points"].astype(float)).astype(int)
    return out

def _build_dataset(years_back: int, windows: Tuple[int, ...]) -> Tuple[pd.DataFrame, List[str]]:
    # Ensure penalty features are present
    pen_df = load_penalty_features()
    years = _years_from_latest(years_back)
    rows: List[Dict[str, Any]] = []
    for y in years:
        df = _fetch_regular_games(y)
        for _, g in df.iterrows():
            game = g.to_dict()
            feats = features_for_game(game, pen_df, windows=windows)
            if not feats:
                continue
            feats["y"] = int(g["y"])
            feats["start_dt"] = g["start_dt"]
            rows.append(feats)
    if not rows:
        return pd.DataFrame(), []
    Xy = pd.DataFrame(rows).sort_values("start_dt")
    cols = [c for c in Xy.columns if c not in {"y","start_dt"}]
    return Xy[["y","start_dt"] + cols].reset_index(drop=True), cols

def _time_split(Xy: pd.DataFrame, train_frac: float = 0.85) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(Xy)
    k = max(1, int(n * train_frac))
    return Xy.iloc[:k].copy(), Xy.iloc[k:].copy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", help="(ignored for now)", default=None)
    ap.add_argument("--years", type=int, default=5, help="How many seasons back to train")
    ap.add_argument("--save", default=str(ART / "model.joblib"), help="Path to save model")
    ap.add_argument("--no-scale", action="store_true", help="Disable feature standardization")
    ap.add_argument("--C", type=float, default=1.0, help="LogReg inverse regularization strength")
    ap.add_argument("--max-iter", type=int, default=200)
    args = ap.parse_args()

    windows = (3, 5, 10)
    Xy, cols = _build_dataset(years_back=args.years, windows=windows)
    if Xy.empty:
        raise SystemExit("No training data built. Make sure penalties.parquet exists and CFBD returned games.")

    train_df, valid_df = _time_split(Xy, train_frac=0.85)

    means = {c: float(train_df[c].mean()) for c in cols}
    for c in cols:
        train_df[c] = train_df[c].fillna(means[c])
        valid_df[c] = valid_df[c].fillna(means[c])

    scaler_path = None
    if not args.no_scale:
        scaler = StandardScaler()
        scaler.fit(train_df[cols].values)
        train_X = scaler.transform(train_df[cols].values)
        valid_X = scaler.transform(valid_df[cols].values)
        scaler_path = str(ART / "scaler.joblib")
        joblib.dump(scaler, scaler_path)
    else:
        train_X = train_df[cols].values
        valid_X = valid_df[cols].values

    train_y = train_df["y"].values
    valid_y = valid_df["y"].values

    clf = LogisticRegression(C=args.C, max_iter=args.max_iter, solver="lbfgs")
    clf.fit(train_X, train_y)
    p_valid = clf.predict_proba(valid_X)[:, 1]

    metrics = {
        "n_train": int(len(train_y)),
        "n_valid": int(len(valid_y)),
        "valid_logloss": float(log_loss(valid_y, p_valid, eps=1e-12)),
        "valid_brier": float(brier_score_loss(valid_y, p_valid)),
    }

    bundle = {
        "estimator": clf,
        "meta": {
            "feature_order": cols,
            "feature_means": means,
            "scaler": scaler_path,  # path or None
            "windows": list(windows),
            "metrics": metrics,
        },
    }
    save_path = pathlib.Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, save_path)

    (ART / "learn_summary.json").write_text(json.dumps(metrics, indent=2))

    print(json.dumps({"saved": str(save_path), **metrics}, indent=2))
    print(f"✅ Model saved → {save_path}")

if __name__ == "__main__":
    main()

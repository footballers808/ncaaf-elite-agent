from __future__ import annotations
import pandas as pd, numpy as np, joblib
from .common import iso_year_week, save_parquet, read_parquet
from . import cfbd_api as api
from .build_features import build_features

def _sigma_points():
    # heuristic game total stddev
    return 13.5

def _week_filter(df: pd.DataFrame, year: int, week: int) -> pd.DataFrame:
    return df[(df["season"]==year) & (df["week"]==week)]

def run(iso_week: str, features_path: str, out_path: str, model_path: str = "artifacts/model.joblib"):
    year, week = iso_year_week(iso_week)

    # ensure features exist and are current enough
    try:
        feats = read_parquet(features_path)
    except Exception:
        feats = build_features(years_back=3)

    week_df = _week_filter(feats, year, week).copy()
    if week_df.empty:
        # fallback: rebuild in case cache is old
        feats = build_features(years_back=3)
        week_df = _week_filter(feats, year, week).copy()

    pack = joblib.load(model_path) if model_path and os.path.exists(model_path := model_path) else None
    if not pack:
        # cold-start: no model yet — emit market-based baseline so pipeline still runs
        week_df["home_pred"] = week_df["market_total"]/2 + (week_df["market_spread"]/2)
        week_df["away_pred"] = week_df["market_total"]/2 - (week_df["market_spread"]/2)
    else:
        models = pack["models"]; cols = pack["features"]
        X = week_df[cols].fillna(week_df[cols].median())
        week_df["home_pred"] = models["home_points"].predict(X)
        week_df["away_pred"] = models["away_points"].predict(X)

    week_df["pred_total"] = week_df["home_pred"] + week_df["away_pred"]
    week_df["pred_spread"] = week_df["home_pred"] - week_df["away_pred"]

    # convert spread to win probability via normal CDF
    from math import erf, sqrt
    sigma = _sigma_points()
    z = week_df["pred_spread"] / (sigma)
    week_df["home_win_prob"] = 0.5 * (1 + week_df["pred_spread"].apply(lambda v: erf(v/(sqrt(2)*sigma))))

    out = week_df[["game_id","season","week","home","away",
                   "home_pred","away_pred","pred_spread","pred_total","home_win_prob"]]
    save_parquet(out, out_path)
    return out

if __name__ == "__main__":
    import argparse, os
    p = argparse.ArgumentParser()
    p.add_argument("--iso-week", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="artifacts/model.joblib")
    a = p.parse_args()
    run(a.__dict__["iso_week"], a.features, a.out, a.model)

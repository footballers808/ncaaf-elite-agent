from __future__ import annotations
import pandas as pd, numpy as np, joblib
from .common import read_parquet, save_parquet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor

TARGETS = ["home_points","away_points"]

def _select_features(df: pd.DataFrame):
    drop = ["game_id","season","week","season_type","home","away","venue","weather"] + TARGETS
    cols = [c for c in df.columns if c not in drop]
    # numeric only
    cols = [c for c in cols if df[c].dtype != "O"]
    return cols

def train(features_path: str, labels_path: str, out_path: str):
    X = read_parquet(features_path)
    y = read_parquet(labels_path)

    df = X.merge(y, on=["game_id","season","week","home","away"], how="inner")
    cols = _select_features(df)
    df = df.dropna(subset=cols)

    Xmat = df[cols].fillna(df[cols].median())
    models = {}
    report = {}

    for tgt in TARGETS:
        yvec = df[tgt].astype(float)
        Xtr, Xva, ytr, yva = train_test_split(Xmat, yvec, test_size=0.2, random_state=42)
        model = LGBMRegressor(
            n_estimators=600,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.8,
            max_depth=-1,
            num_leaves=63,
            reg_lambda=2.0,
            random_state=42
        )
        model.fit(Xtr, ytr)
        p = model.predict(Xva)
        report[f"mae_{tgt}"] = float(np.round(mean_absolute_error(yva, p), 3))
        models[tgt] = model

    joblib.dump({"models": models, "features": cols}, out_path)
    with open("artifacts/learn_report.txt","w") as f:
        f.write(str(report))
    return report

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    train(a.features, a.labels, a.out)

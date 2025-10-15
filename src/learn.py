import argparse, pathlib, pandas as pd, numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import joblib, yaml
from .common import safe_read_parquet, ART

TARGETS = ["home_score", "away_score"]

def _build_training(features: pd.DataFrame, labels: pd.DataFrame):
    # Pivot features to have one row per game with home & away columns
    # Then join labels
    def pivot_side(df, prefix, side_mask):
        cols = ["game_id","team","pf_mean","pa_mean","pace_mean","injuries_recent",
                "market_spread","market_total","wx_temp","wx_wind","wx_precip","neutral_site"]
        x = df.loc[side_mask, cols].copy()
        x = x.rename(columns={c: f"{prefix}_{c}" for c in cols if c not in ["game_id"]})
        return x

    home = pivot_side(features, "home", features["team"].eq(features["home_team"]))
    away = pivot_side(features, "away", features["team"].eq(features["away_team"]))
    X = home.merge(away, on="game_id", how="inner")
    y = labels[["game_id","home_score","away_score"]]
    df = X.merge(y, on="game_id", how="inner").dropna()
    # Basic sanity: drop columns not used for fit
    feature_cols = [c for c in df.columns if c not in ["game_id","home_score","away_score","home_team","away_team"]]
    return df[feature_cols], df[TARGETS], df["game_id"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open("config.yaml","r") as f:
        cfg = yaml.safe_load(f)

    F = safe_read_parquet(pathlib.Path(args.features))
    L = safe_read_parquet(pathlib.Path(args.labels))

    X, Y, G = _build_training(F, L)
    # CV by game_id groups (stable leakage guard)
    gkf = GroupKFold(n_splits=5)
    preds = np.zeros_like(Y.values, dtype=float)

    if cfg["model"]["regressor"] == "random_forest":
        base = RandomForestRegressor(
            n_estimators=cfg["model"]["n_estimators"],
            max_depth=cfg["model"]["max_depth"],
            random_state=cfg["model"]["random_state"],
            n_jobs=-1,
        )
    else:
        base = RandomForestRegressor(n_estimators=400, max_depth=12, random_state=42, n_jobs=-1)

    model = MultiOutputRegressor(base)

    for tr, va in gkf.split(X, Y, groups=G):
        model.fit(X.iloc[tr], Y.iloc[tr])
        preds[va] = model.predict(X.iloc[va])

    mae_home = mean_absolute_error(Y.iloc[:,0], preds[:,0])
    mae_away = mean_absolute_error(Y.iloc[:,1], preds[:,1])
    print(f"CV MAE home={mae_home:.2f} away={mae_away:.2f}")

    # Fit on all and save
    model.fit(X, Y)
    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": X.columns.tolist()}, outp)

if __name__ == "__main__":
    main()

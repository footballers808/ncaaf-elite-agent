import argparse, pathlib, yaml, pandas as pd
from .common import ART, safe_read_parquet, safe_write_parquet
from . import cfbd_client as cfbd

def iso_to_year_week(iso: str):
    y, w = iso.split("-")
    return int(y), int(w)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iso-week", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    year, week = iso_to_year_week(args.iso_week)
    with open("config.yaml","r") as f:
        cfg = yaml.safe_load(f)

    F = safe_read_parquet(pathlib.Path(args.features))
    home = F[F["team"]==F["home_team"]]
    away = F[F["team"]==F["away_team"]]
    X_home = home[["game_id","pf_mean","pa_mean","pace_mean","injuries_recent","market_spread","market_total","wx_temp","wx_wind","wx_precip","neutral_site"]].rename(columns=lambda c: c if c=="game_id" else f"home_{c}")
    X_away = away[["game_id","pf_mean","pa_mean","pace_mean","injuries_recent","market_spread","market_total","wx_temp","wx_wind","wx_precip","neutral_site"]].rename(columns=lambda c: c if c=="game_id" else f"away_{c}")

    X = X_home.merge(X_away, on="game_id")
    games = cfbd.games(year, cfg.get("season_type","regular"), week=week)
    valid_ids = {g["id"] for g in games if not g.get("completed")}
    X = X[X["game_id"].isin(valid_ids)].copy()

    import joblib
    model = joblib.load(ART / "model.joblib")["model"]
    feature_cols = [c for c in X.columns if c != "game_id"]
    preds = model.predict(X[feature_cols])

    out = X[["game_id"]].copy()
    out["pred_home_score"] = preds[:,0]
    out["pred_away_score"] = preds[:,1]
    safe_write_parquet(out, pathlib.Path(args.out))

if __name__ == "__main__":
    main()

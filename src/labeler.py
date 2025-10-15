import argparse, pathlib, pandas as pd
from .common import ART, safe_write_parquet
from . import cfbd_client as cfbd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--season-type", type=str, default="regular")
    args = ap.parse_args()

    # Build labels from completed games
    df = []
    games = cfbd.games(args.year, args.season_type)
    for g in games:
        if not g.get("completed"): 
            continue
        df.append({
            "game_id": g["id"],
            "home_team": g["home_team"],
            "away_team": g["away_team"],
            "home_score": g.get("home_points"),
            "away_score": g.get("away_points"),
            "total_actual": (g.get("home_points") or 0) + (g.get("away_points") or 0),
            "spread_actual": (g.get("away_points") or 0) - (g.get("home_points") or 0) # away - home
        })
    labels = pd.DataFrame(df)
    safe_write_parquet(labels, ART / "labels.parquet")

if __name__ == "__main__":
    main()

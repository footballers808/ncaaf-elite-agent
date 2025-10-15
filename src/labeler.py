from __future__ import annotations
import pandas as pd
from . import cfbd_api as api
from .common import save_parquet

def build_labels(year: int, season_type: str = "regular"):
    g = api.games(year, season_type)
    rows = []
    for r in g:
        rows.append({
            "game_id": r.get("id"),
            "season": r.get("season"),
            "week": r.get("week"),
            "home": r.get("home_team") or r.get("homeTeam"),
            "away": r.get("away_team") or r.get("awayTeam"),
            "home_points": r.get("home_points"),
            "away_points": r.get("away_points"),
        })
    df = pd.DataFrame(rows).dropna(subset=["home_points","away_points"])
    save_parquet(df, "artifacts/labels.parquet")
    return df

if __name__ == "__main__":
    import argparse, datetime as dt
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=dt.date.today().year)
    p.add_argument("--season-type", default="regular")
    a = p.parse_args()
    build_labels(a.year, a.season_type)

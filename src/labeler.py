# src/labeler.py
from __future__ import annotations

import os
import time
from typing import Optional, List, Dict

import pandas as pd
import requests

STORE_DIR = "store"
os.makedirs(STORE_DIR, exist_ok=True)

CFBD = "https://api.collegefootballdata.com"

def fetch_completed_games(year: int, week: Optional[int] = None) -> List[Dict]:
    """Fetch games with final scores; requires CFBD_API_KEY in env."""
    api_key = os.environ.get("CFBD_API_KEY")
    if not api_key:
        raise RuntimeError("CFBD_API_KEY is not set in environment.")
    headers = {"Authorization": f"Bearer {api_key}"}

    params: Dict[str, object] = {"year": year}
    if week is not None:
        params["week"] = week

    r = requests.get(f"{CFBD}/games", params=params, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()

    games = [
        g for g in data
        if g.get("home_points") is not None and g.get("away_points") is not None
    ]
    return games

def build_labels(games: List[Dict]) -> pd.DataFrame:
    if not games:
        return pd.DataFrame()
    df = pd.DataFrame(games).rename(columns={
        "id": "game_id",
        "home_team": "home",
        "away_team": "away",
    })
    df["actual_spread"] = df["home_points"] - df["away_points"]
    df["actual_total"]  = df["home_points"] + df["away_points"]
    return df[["game_id", "home", "away", "actual_spread", "actual_total"]]

def main(year: Optional[int] = None, week: Optional[int] = None) -> None:
    if year is None:
        year = int(time.strftime("%Y"))
    games = fetch_completed_games(year, week)
    labels = build_labels(games)

    out_path = os.path.join(STORE_DIR, "labels.parquet")
    if labels.empty:
        print("⚠️ No completed games found; writing placeholder labels.parquet")
        pd.DataFrame(columns=["game_id","home","away","actual_spread","actual_total"]).to_parquet(out_path, index=False)
    else:
        labels.to_parquet(out_path, index=False)
        print(f"✅ Wrote labels: {out_path} (rows={len(labels)})")

if __name__ == "__main__":
    main()
